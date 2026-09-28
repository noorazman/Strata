# V100 Stage 1.8 — QSA attention-path analysis (profile-only, no engine changes)

Production target: **32 GB V100 PCIE (GPU0)** only. Goal of the stage: determine whether
the QSA attention path can be **meaningfully** accelerated on SM70 **before** implementing
anything. This stage is analysis only: profile the complete QSA path, run the fast-vs-slow
A/B (E4 plumbing from `00e1609`), measure the attention kernel against its roofline, and
decide. **No FlashAttention implemented, no kernel rewrites, no production defaults changed.**

Engine: `build-sm70/strata` at `0fea54d` (Stage 1.7 complete). Binary unchanged this stage.

## 1. Method

| Tool | Config | Purpose |
|---|---|---|
| `bench/v100/s18e4.sh` / `s18e4b.sh` | 256 tok, fp16/8192 and int8/32768, `--kv fp16\|int8`, 3-cycle fp16 arms, `drop_caches` between | e2e A/B (steady state, GPU0 free) |
| `bench/v100/s18nsys.sh` | `nsys --cuda-graph-trace=node`, 64-tok window, 4 arms (base / nofastattn / nofastsel / int8-base) | kernel-level breakdown on a fixed workload |
| `bench/v100/s18_trace.py` | sqlite → stream-order QSA segment decoder (kv_append anchors, exact fixed-offset segments) | per-component µs/tok tables |
| `/tmp/s18bench/bench_attn*.cu` | `nvcc -O3 -arch=sm_70` microbench: faithful production-structure replica + 2 WMMA fp16 variants + chunk-size sweep | attention-kernel roofline |

Decode window = from the largest inter-kernel gap (prefill→decode boundary) to the last
kernel; 64-token runs give 25–27 verify rounds (300–324 QSA segments = 12 layers × rounds).
Per-token kernel time (kernel-time sum / tokens) is the robust cross-run metric; the 64-tok
base run contained 3–4 environmental stalls (539/449/368 ms PLE/expert-cache fetches after
`drop_caches`), so "GPU busy %" from those windows is not comparable across arms.

## 2. Full QSA attention-path time breakdown (new binary, per token, kernel time)

Main-model stream kernel time: **base 22,731 µs/tok** (fp16), **22,644 µs/tok** (int8).
MTP draft stream: 809–822 µs/tok (3.4 % of total). `wait_flag_ge` spin: **7,868 µs/tok
(34 % of the main stream)** — the single largest kernel class, larger than the entire QSA
attention mechanism (see §6 finding F1).

**QSA attention mechanism** (attention-specific: projections + indexer + selection +
attention + small kernels; MoE/FFN tail of the layer excluded) — fp16 base, 64-tok nsys:

| component | µs/tok | µs/call | calls/tok | notes |
|---|---|---|---|---|
| projection Q (IQ4_XS GEMV) | 287.2 | 61.3 | 4.7 | largest single component |
| attention chunk (fast) | 155.2 | 33.1 | 4.7 | 66 blocks, single wave |
| projection O (Q6_K/Q5_K GEMV) | 147.4 | 31.4 | 4.7 | |
| Q/idxq norm (rms) | 90.1 | 3.0 | 30.4 | 2× per-token × ~15 |
| indexer Q projection (bf16 mmvf) | 89.5 | 5.9 | 15.2 | |
| Q/idxq rope | 68.7 | 2.3 | 30.4 | |
| indexer K projection (bf16 mmvf) | 68.5 | 4.5 | 15.2 | |
| attn output quantize (q8_1) | 51.6 | 2.7 | 19.5 | |
| K norm | 46.8 | 3.1 | 15.2 | |
| KV cache append | 45.8 | 3.0 | 15.2 | |
| projection K (IQ4_XS GEMV) | 40.4 | 8.6 | 4.7 | |
| indexer key append | 38.8 | 2.6 | 15.2 | |
| gate (sigmoid×attn) | 38.0 | 2.5 | 15.2 | |
| K rope | 35.2 | 2.3 | 15.2 | |
| projection V (Q6_K GEMV) | 33.7 | 7.2 | 4.7 | |
| attention merge (fast) | 27.4 | 5.9 | 4.7 | |
| selection block scores (fast) | 17.7 | 3.8 | 4.7 | |
| step/pos staging (mapped copy) | 13.3 | 2.8 | 4.7 | |
| activation quantize (q8_1) | 11.5 | 2.5 | 4.7 | |
| selection block topk (fast) | 10.7 | 2.3 | 4.7 | |
| **QSA attention mechanism subtotal** | **1,317** | | | **5.8 % of main-stream kernel time** |

Per QSA-layer invocation (T=1): window ≈ 598 µs; attention core (chunk+merge) ≈ 39 µs;
projections ≈ 110 µs; indexer ≈ 150 µs; small kernels ≈ 200 µs; MoE tail ≈ 300 µs.

int8 base: attention mechanism **1,356 µs/tok** (+38 vs fp16: chunk 162.3 vs 155.2, int8
block scores 35.6 vs 17.7 — the dequant inside block scoring; everything else identical).
**int8 KV attention costs ~3 % more than fp16 KV attention** — int8 KV is already the right
choice (cheaper bandwidth, same speed).

## 3. Fast vs slow A/B (E4 plumbing, `--no-fast-attn` / `--no-fast-select`)

### e2e (256 tok, steady state, GPU0 free, `drop_caches` between runs)

| arm (256 tok) | tok/s | Δ vs base | trajectory |
|---|---|---|---|
| fp16 base ×3 | 49.15 / 49.86 / 49.11 (avg 49.37) | — | **all GOLDEN** `c1517d…`, 3/3 deterministic; 111 rounds, 145/210 (0.69) |
| fp16 nofastattn ×3 | 46.30 / 45.00 / 44.40 (avg 45.23) | **−8.4 %** | md5 `537afd…` 3/3 deterministic; diverges from base at tok 42; 105 rounds, 151/238 (0.634); MTP draft 2.09 ms/round (+8.5 %) |
| fp16 nofastsel ×3 | 47.35 / 47.43 / 48.46 (avg 47.75) | **−3.3 %** | md5 **= GOLDEN** `c1517d…` (bit-identical trajectory, 3/3); identical rounds/acceptance |
| fp16 g32 base / nofastattn / nofastsel | 39.28 / 38.74 / 37.21 | — | 32-prefix `cf577e…` = GOLDEN_32 on all three (slow paths bit-identical through 32 tok) |
| int8 base | 48.60 | — | md5 `cdb7f7…`; diverges from fp16 base at tok 135 |
| int8 nofastattn | 43.81 | **−9.9 %** | md5 `1df31f…`; 117 rounds, 141/234 (0.603) |
| int8 nofastsel | **20.17** | **−58.6 %** | md5 `38f859…`; diverges from int8 base at tok 221 then degenerate loop (token 248045 ×9); **MTP draft 8.591 ms/round (vs 1.969)**; pool 21.15 ms/tok (vs ~13.5) |
| int8 g32 base / nofastattn / nofastsel | 37.21 / 36.43 / 37.21 | — | base & nofastsel share the fp16 32-prefix `cf577e…`; nofastattn diverges at tok 6 |

All 15 runs rc=0, no CUDA errors, no hangs.

### kernel level (64-tok nsys, per token — the clean, fixed-workload evidence)

| quantity (µs/tok) | base | nofastattn | nofastsel |
|---|---|---|---|
| attention core, fast (chunk+merge) | **182.7** (33.1+5.9/call) | — | 183.2 |
| attention core, slow (qsa_attend+kv_gather) | — | **1,094.4** (65.8+3.7/call, per-token) | — |
| selection, fast (block scores+topk) | **28.4** | 28.9 | — |
| selection, slow (qsa_index+topk) | — | — | **652.8** (5.5+37.4/call, per-token) |
| main-stream kernel total | 22,731 | 25,114 (+10.2 %) | 23,173 (+1.9 %) |

Slow attention = **6.0×** the fast core per invocation; slow selection = **23×** the fast
selection. Per QSA-layer invocation (T=1): fast core ≈ 39 µs (batched over T) vs slow
≈ 216 µs (T per-token calls) → **+177 µs/invocation**, growing with T because the fast
path batches T tokens in one launch while the slow path issues T per-token launches.

**A/B confound (expected, quantified):** the slow paths change numerics → different
trajectory → different MTP acceptance → different round count and draft time.
- fp16 nofastsel is the clean one: **bit-identical trajectory** (same golden md5) → its
  −3.3 % e2e is the pure cost. Check: +624 µs/tok kernel × 256 tok = +160 ms →
  47.9 tok/s predicted vs 47.75 measured. ✓
- fp16 nofastattn: pure-kernel prediction +912 µs/tok → 47.2 tok/s (−4.3 %); measured
  45.23 (−8.4 %) — the trajectory drift (0.69→0.634 acceptance, +8.5 % draft time) costs
  another ~4 %.
- int8 nofastsel: pure-kernel prediction ≈ −2 % (fp16 nofastsel delta); measured **−58.6 %**
  — the MTP-draft serialization (8.59 vs 1.97 ms/round) + degenerate loop dominate.
  Conclusion: `--no-fast-select` is a diagnostic arm only; in the production int8 config
  the slow selection is catastrophic e2e, far beyond its direct kernel cost.

## 4. Theoretical maximum end-to-end gain

Production wall (fp16): 5,185 ms / 256 tok = **20.26 ms/tok** (49.37 tok/s). int8: 20.58 ms/tok.

| hypothetical | µs/tok saved | e2e (fp16) |
|---|---|---|
| entire QSA attention mechanism → 0 | 1,317 | 5,185 − 337 = 4,848 ms → **52.8 tok/s (+6.9 %)** |
| attention core (chunk+merge) → 0 | 183 | → 49.8 tok/s (+0.9 %) |
| attention core 4× faster (33.7 → 8.4 µs/launch, ≈ roofline) | 137 | → 49.7 tok/s (+0.7 %) |
| int8: QSA mechanism → 0 | 1,356 | 5,267 − 347 = 4,920 ms → 52.0 tok/s (+7.1 %) |

**The ceiling for the whole QSA attention mechanism is ~7 % e2e; for the pure attention
kernel, ~1 %.** Even a perfect SM70 FlashAttention-style kernel buys ~1 % e2e on this
model/config. Attention is real but small: the token is 5.8 % QSA attention, 34 % expert-pool
spin, ~6.5 % QSA-layer MoE, and the rest is the 36 GDN layers + MTP.

## 5. Microbenchmark — attention kernel vs roofline (V100 SM70, `nvcc -O3 -arch=sm_70`)

Production shape: G=12 q-heads/kv-head, HD=256, CHUNK=64 cells, 256 threads, grid 66 blocks
(33 chunks × 2 kv-heads over 2051 top-k cells), fp16 KV. 4.33 MB K/V per launch.

| variant | µs/launch | GFLOP/s | GB/s | regs / smem |
|---|---|---|---|---|
| production structure replica (v_cur) | **33.7** | 2,057 (13 % of FP32 peak) | 128 (14 % of HBM) | 58 / 15,360 B (prod: 44 / 15,872 B) |
| WMMA fp16, shared-staged (v_wmma) | 54.9 | 1,262 | 79 | 46 / 16,384 B |
| WMMA fp16, register-resident (v_wmma2) | 30.5 | — | 142 | **174** / 16,384 B |
| production structure, CHUNK=16 | 23.4 | — | 180 | — |
| production structure, CHUNK=32 | 25.2 | — | 169 | — |
| production structure, CHUNK=64 (prod) | 33.8 | — | 128 | — |
| production structure, CHUNK=128 | 59.8 | — | 75 | — |

Roofline: 4.33 MB @ ~900 GB/s HBM → **5–8 µs/launch floor**; 66 blocks < 80 SMs = single
wave, so the kernel is latency-bound, not bandwidth- or compute-bound. Headroom 4–6×,
**but capturing it all is only ~0.7 % e2e**. Findings:
- The kernel already *is* FlashAttention-style (64-cell tiling, online softmax via the
  merge kernel, no score materialization). "Implementing FlashAttention" would be a
  re-implementation of the existing structure.
- Naive WMMA fp16 is **slower** (54.9 µs): 12→16 row padding wastes 25 % of the mma tiles,
  and the S-stage → shared → softmax → shared → P-load cycle adds 3 `__syncthreads` and
  8 KB of shared traffic per block. The register-resident WMMA variant (30.5 µs) only
  marginally beats the FMA path and needs fp16 Q (a numerics change → opt-in mode).
- **Chunk size is the one free lever**: CHUNK=32 → 25.2 µs (−25 %), CHUNK=16 → 23.4 µs
  (−31 %), no numerics change (same per-cell FMA order; the online-softmax merge is
  chunk-count invariant). e2e ceiling ~0.2 %.
- Toolchain gotcha found: `wmma::store_matrix_sync`/`load_matrix_sync` from/to **local
  memory** (register arrays) faults with `unspecified launch failure` (trap) on CUDA
  12.8/12.9 sm_70; shared/global staging works. Any future SM70 WMMA work must stage
  fragment stores through shared memory.

## 6. Findings and verdicts

**F1 — `wait_flag_ge` spin is the largest decode kernel class: 7,868 µs/tok (34 % of the
main stream; 503.5 ms in the 64-tok window).** Structural change vs the old binary:
**wait calls per layer-round 1.0 (s17x, 5,328 = 48×111) → 3.0 (new, 3,600 = 48×25×3)**;
spin per layer-round 309 µs (old) → 420 µs (new). Re-attributing flag A/B/C on the new
binary is the highest-value profiling item for the next stage — it dwarfs the entire QSA
attention mechanism (1,317 µs/tok). (64-tok window caveat: the base run had environmental
stalls; the call-count change is exact and not affected by that.)

**F2 — The QSA attention mechanism is 5.8 % of the token (ceiling ~7 % e2e). The pure
attention kernel is 0.8 % (ceiling ~1 %).** Attention *can* be made ~4× faster (chunk-size
tuning, or a tuned WMMA path), but the e2e payoff is 0.2–0.7 %. **Answer to the stage
question: attention is not a *major* e2e bottleneck on this model; it is worth a small,
low-risk tuning pass, not a FlashAttention project.**

**F3 — Projections dominate the attention mechanism**: Q GEMV 287 + O GEMV 147 + K/V GEMV
74 = 508 µs/tok (39 % of the mechanism, 2.2 % of the token). These are weight-traffic-bound
GEMVs — the real lever is an int8 tensor-core GEMV path (opt-in, numerics change), not
attention-kernel work.

**F4 — The ~8 small per-token QSA kernels** (norms, ropes, indexer append, KV append,
gate, quantizes, staging) ≈ 370 µs/tok + their launch overhead. Fusing them is a ~1.5–2 %
e2e ceiling project — structural, cross-cutting, worth a dedicated stage, not this one.

**F5 — int8 KV attention ≈ fp16 KV attention** (chunk 162 vs 155 µs; +3 % on the
mechanism). int8 KV stays the production choice. Slow selection, however, is
disproportionately expensive under int8 e2e (−58.6 % with MTP-draft serialization) —
the diagnostic `--no-fast-select` must not be used for long int8 generations.

**Rejected ideas (with why):**
1. **WMMA/tensor-core fp16 QK^T now** — measured 54.9 µs (naive) / 30.5 µs (register-
   resident) vs 33.7 µs current: no clear win at G=12/HD=256/CHUNK=64; 25 % mma waste from
   row padding; requires an fp16-Q mode (numerics change); e2e ceiling 0.9 % anyway.
   *Keep open* for a later int8-TC attention experiment (250 TOPS int8 TC, production is
   int8 KV).
2. **FlashAttention-style rewrite** — the kernel is already tiled/online-softmax; the
   remaining gap is occupancy/latency, not structure.
3. **KV-cache layout changes** — KV append is 46 µs/tok; int8 already neutral-to-slower
   for attention but cheaper elsewhere.
4. **Bigger GEMV batch / multi-token Q projection** — already batched per verify window
   (T tokens per launch); no headroom at T≈2.3.
5. **Occupancy-only tuning** — 66 blocks/80 SMs is one wave; more blocks via smaller
   chunks is the occupancy lever (F2, the free part of it).

**Worth attempting (ranked):**
1. **CHUNK 64→32/16 in the fast chunk kernel** (compile-time or flag): −25–31 % on the
   chunk kernel, no numerics change, e2e +0.15–0.2 % (≈ +0.08–0.1 tok/s). Low risk,
   golden-gated.
2. **Q/O int8-TC GEMV path** (next stage, opt-in `--qsa-gemv-int8`): targets 508 µs/tok,
   realistic ~200–250 µs/tok saved (≈ +1–1.2 % e2e); numerics change → flag-gated +
   32/32 golden re-baseline.
3. **Fused small-kernel pass** (next stage): fuse norm/rope/idx-append/kv-append/gate/
   quantize into ≤3 launches per QSA layer; ceiling ~1.5–2 % e2e, also cuts ~96 launches/tok.
4. **wait_flag re-attribution** (next stage, F1): 34 % of the main stream — the biggest
   single kernel class in decode.

**Proposed next experiment (not implemented): E1 = QSA chunk-size knob.**
Expose CHUNK (64 default; 32 opt-in via `STRATA_QSA_CHUNK=32`) in
`qsa_decode_attn.cu` (chunk count and the merge's `n_chunks` adapt; scratch sizing is
per-chunk). Expectation: chunk 33.7 → ~25 µs (−26 %), merge unchanged, e2e +0.1–0.15 tok/s.
Gate: 32/32 golden (expect bit-identical — the merge is the same online-softmax reduction
over a different partition; verify, don't assume) + 256/256 determinism + no CUDA errors,
then a 3-cycle paired A/B like `s18e4.sh`. If bit-identical and ≥ −20 % chunk time, keep
32 as default candidate; otherwise document and revert.

## 7. Correctness & environment

- fp16 base: 3/3 × 256-tok **GOLDEN** `c1517d02473fbc06b5cf415ea1f8be63`, deterministic.
- fp16 32-tok golden: `cf577e731fbff35d91b879a23b670e56` (all 3 arms identical).
- fp16 nofastattn 3/3 deterministic (`537afd…`); fp16 nofastsel 3/3 = GOLDEN.
- int8 256 arms single-run each (md5s recorded); int8 g32 base shares the fp16 32-prefix.
- All 15 A/B runs + 4 nsys runs rc=0, no CUDA errors, no hangs. Engine binary unchanged
  (no ctest delta expected; 20/22 standing from Stage 1.7).
- GPU0 free during all captures (service stopped for the campaign; VRAM ≤ 19.5 GiB peak).
- Raw data: `Logs/benchmarks/s18e4*.json` (15 A/B runs), `Logs/benchmarks/s18e4b-campaign.log`,
  `Logs/gpu/s18nsys-{base,nofastattn,nofastsel,int8-base}.{nsys-rep,sqlite}`,
  `bench/v100/s18{e4,e4b,nsys}.sh`, `bench/v100/s18check.py`, `bench/v100/s18_trace.py`.
  Microbench source kept at `/tmp/s18bench/` (not committed; re-derivable, SM70-only).
