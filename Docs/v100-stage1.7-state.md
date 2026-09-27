# V100 Stage 1.7 — GPU Kernel & Attention Optimization (WIP STATE CHECKPOINT)

> **WORK-IN-PROGRESS checkpoint** — written at the user's pause request, before any
> Stage 1.7 code change. Baseline is established; profiling + source audit done;
> experiments not yet started. This file is the resume point; it is superseded by
> `Docs/v100-stage1.7-final.md` when the stage completes.

## 1. Objective (from the user)

Investigate remaining GPU-side bottlenecks on the V100 (SM70), profile first:

1. Attention kernels. 2. Flash Attention feasibility/performance on V100/SM70.
3. Tensor-Core utilization. 4. GPU memory bandwidth. 5. Kernel launch overhead.
6. Verify-window kernels. 7. Kernel fusion opportunities.
8. Overlap between GPU work and the existing CPU/pool synchronization.

Rules: profile before implementing; one change at a time (baseline → change →
benchmark → correctness → keep/revert); do not touch PLE default/PLE RAM/pool
workers/2048 µs parking/quantization/model format; no context shifting, no dense
models. A faster kernel that changes numerics is a separate mode, not a silent
replacement. Success = end-to-end tok/s improvement over the ~50 tok/s baseline,
validated on both the 32 GB (GPU0) and 16 GB (GPU4) V100.

## 2. Environment / housekeeping state at pause

- Repo: `~/dsh/strata/Strata`, branch `stage1.3-expert-pool-sync`, HEAD
  `3eafe42` (Stage 1.6 docs) / `1fbe264` (Stage 1.6 code) — **checkpoint, do not
  disturb**. Working tree at pause: only Stage 1.7 artifacts untracked
  (`bench/v100/s17_trace.py`, `Logs/{benchmarks,cpu,gpu}/s17-baseline-*.{json,csv}`);
  the nsys profile + sqlite are gitignored (`Logs/gpu/nsys-*`, `*.sqlite`) but exist
  on disk (regenerate per §5 if lost).
- **`strata.service` (the port-8180 API server on GPU0) was STOPPED at 00:53 for
  clean profiling. RESTART IT at the end of the stage:**
  `echo mustoe8 | sudo -S -p '' systemctl start strata.service`
- GPU0 (32 GB PCIE, dev) free. GPU1/3 busy (llama-server, ninfer — don't touch);
  GPU2/4 (16 GB SXM2) free for validation.
- Build: `build-sm70/strata` (built 22:34 on the 27th, current HEAD, no
  Stage 1.7 edits yet). nsys 2024.6.2; **no ncu on this box** (occupancy/regs come
  from the nsys sqlite `CUPTI_ACTIVITY_KIND_KERNEL.registersPerThread` + smem cols,
  and from source).
- Baseline config (Stage 1.6, frozen): `--pool-workers 24 --ple-io ram` (default),
  `STRATA_POOL_PARK=2048`, `--pcie-frac 0.2` (default), `--spec 4 --spec-min-p 0.5`,
  `--expert-cache auto`, bench decode = 256 tokens, `--kv fp16 --max-context 8192`.

## 3. Stage 1.6 baseline, re-established (GPU0, 256 tokens, interleaved)

| run | decode tok/s | golden | notes |
|---|---:|:--:|---|
| s17-baseline-1 | 49.28 | 32/32 prefix ✓ | cold PLE preload (22 s) |
| s17-baseline-2 | 50.15 | ✓ | 111 rounds ×4, 145/210 accepted (0.690), 2.31 tok/round |
| s17-baseline-3 | 50.07 | ✓ | |

Window sizes T1:29 T2:12 T3:12 T4:58. Peak VRAM 18,896 MiB; GPU peak 100 %;
CPU mean ~7 %; pool 12.6–13.1 ms/tok; MTP draft ~1.9 ms/round. Matches the
known-good 50.0–50.1 tok/s. Golden bar (32 tokens):

```
271 248068 198 760 1156 369 30869 5402 430 328 23202 1288 264 11952 5617 303 220 17 15 17 21 11 321 369 883 310 3184 728 883 836 1118 1834
```

## 4. Stage 1.6 GPU profile — the bottleneck breakdown (PRIMARY DATA OF THIS STAGE)

Capture: `Logs/gpu/s17-baseline-node.nsys-rep` (+ `.sqlite`), nsys 2024.6.2,
`-t cuda --cuda-graph-trace=node`, same 256-token workload. Under node trace the
run is 41.02 tok/s (expected ~18 % profiler overhead vs ~49.3). Decode window =
activity after the 215 ms prefill→decode gap: **span 6,202 ms, 376,564 kernel
events, GPU busy (union) 5,815.5 ms = 93.8 % of span → decode is GPU-busy-bound.**

### 4.1 Functional-area breakdown (share of kernel busy, 5,851 ms)

| area | ms | % | kernels |
|---|---:|---:|---:|
| Expert GEMV (routed MoE IQ3_XXS + shared, native_gu/native_down/native_mmvq*/q5/q6/small) | 1776.0 | 30.4 % | 58,377 |
| `wait_flag_ge_kernel` (CPU-pool doorbell spin) | 1598.0 | 27.3 % | 15,984 |
| GDN mixer `gr_*_multi` (hyper-connection read: down+up+norm) | 989.8 | 16.9 % | 33,870 |
| zero-copy mapped copies + doorbell publish | 303.5 | 5.2 % | 20,598 |
| routing (`route`+combine+sigmoid+group_resident+…) | 285.6 | 4.9 % | 57,282 |
| dense bf16 GEMV `bf16_f32_mmvf` (attn projections, shared, head) | 237.5 | 4.1 % | 40,852 |
| GDN core (`gdn_step_norm_multi`, `gdn_ab_multi`, `gdn_conv_l2_multi`) | 219.9 | 3.8 % | 19,980 |
| elementwise/other small (norm/swiglu/to_bf16/apply/append/add_hits/gate) | 211.2 | 3.6 % | 77,901 |
| activation quantize (q8_0/q8_1) | 114.1 | 2.0 % | 40,165 |
| **QSA attention core (attn_chunk+merge, qsa_*, indexer, topk, kv_append, rope, gate)** | **91.1** | **1.6 %** | 10,286 |
| PLE block | 17.6 | 0.3 % | 373 |
| sampler/MTP/commit helpers | 6.7 | 0.1 % | 896 |

### 4.2 Top kernels (name, n, total, mean µs, grid/block, regs, smem)

| kernel | n | ms | mean µs | grid/block | regs | smem |
|---|---:|---:|---:|---|---:|---:|
| `wait_flag_ge_kernel` | 15,984 | 1598.0 | 100.0 | 1/1 | 16 | 0 |
| `gr_down_multi_kernel` | 11,290 | 541.0 | 47.9 | **41**/256 | 74 | 32,768 |
| `gr_up_multi_kernel` | 11,290 | 290.2 | 25.7 | 160/256 | 40 | 98,304 |
| `bf16_f32_mmvf_kernel<256>` | 40,269 | 226.1 | 5.6 | 512/256 | 29 | 8,192 |
| `native_down_kernel<42>` | 6,660 | 206.5 | 31.0 | 320×10/256 | 60 | 0 |
| `gr_norm_multi_kernel` | 11,290 | 158.6 | 14.1 | 1/256 | 32 | 8,192 |
| `copy_from_mapped_kernel` | 8,534 | 158.2 | 18.5 | 3/256 | 32 | 0 |
| `route` | 15,670 | 148.6 | 9.5 | **1 block** 32×8 | 83 | 0 |
| `native_gu_kernel<21>` | 2,886 | 139.8 | 48.4 | 160×10/256 | 48 | 0 |
| `native_mmvq_multi_kernel<IQ4XS,4,4,4>` | 4,118 | 139.6 | 33.9 | 6144/32×4 | 56 | 16,384 |
| `mmvq_kernel<21>` | 6,105 | 137.7 | 22.6 | 1536/32×4 | 40 | 0 |
| `native_down_kernel<20>` | 3,996 | 137.7 | 34.5 | 320×10/256 | 40 | 0 |
| `gdn_step_norm_multi_kernel` | 7,992 | 120.1 | 15.0 | 48/128×4 | 64 | 8,192 |
| `native_gu_kernel<22>` | 2,220 | 103.3 | 46.5 | 160×10/256 | 48 | 0 |
| `native_mmvq_multi_kernel<...,123 regs>` | 3,538 | 103.3 | 29.2 | 2560/32×4 | **123** | 32,768 |
| `native_gu_kernel<17>` | 2,220 | 95.0 | 42.8 | 160×10/256 | 40 | 0 |
| `copy_i32_from_mapped_kernel` | 6,736 | 90.9 | 13.5 | 1/128 | 16 | 0 |
| `native_mmvq_multi_kernel<...>` (640 blocks) | 870 | 88.2 | **101.4** | 640/32×4 | 64 | 16,384 |
| `native_gu_kernel<16>` | 1,998 | 75.6 | 37.8 | 160×10/256 | 40 | 0 |
| `gdn_ab_multi_kernel` | 3,996 | 73.4 | 18.4 | 12/256 | 42 | 0 |
| `native_q5_k_mmvq_kernel` | 697 | 55.3 | 79.4 | 640/32×4 | 40 | 8,192 |
| `attn_chunk_kernel<false>` (QSA flash-decode) | 1,594 | 57.9 | 36.3 | **33×2**/256 | 44 | 98,304 |
| `doorbell_publish_kernel` | 5,328 | 54.4 | 10.2 | 1/1024 | 16 | 0 |
| `native_q6_k_mmvq_kernel` | 2,059 | 48.9 | 23.8 | 6144/32×4 | 40 | 8,192 |

### 4.3 `wait_flag_ge` attribution (same-stream next-event classifier, Stage 1.3 method)

- total 1598.0 ms: **flag A (plan) 962.9 ms (60.3 %)**, flag B (staging DMA)
  373.1 ms (23.3 %), flag C (CPU rows) 262.1 ms (16.4 %).
- **158 waits > 1 ms account for 1,023.1 ms (max 16.3 ms)** — the round-head
  waits. Engine's own diagnostics agree: `verify window: wait for rings 28.026
  ms/round` (111 rounds), slowest spins `l1/g0 8.9/7.4/5.7/3.9 ms`, `l38/g0 3.7`.
  i.e. the Stage 1.6 round-head flag-A/flag-C visibility lag is the single biggest
  wall-time item; it improved somewhat vs 1.6 (top-5 mean ~5.9 ms vs 9–14 ms) but
  remains ~17 % of round wall.
- flag B = 373 ms (23.3 %) is the residual staging-DMA wait (pcie-frac 0.2).

### 4.4 Memcpy (decode window)

- H2D: 4,340 copies, **7,441 MB** (staging DMA; sizes 1.3–2.3 MB expert blobs).
- D2H: 0 (in-window); kind-8 (116 MB, 4,435 copies ~26 KB) = MTP/commit small copies.

### 4.5 Verify-window structure

The native pack runs verify windows only (no session graphs). Per round (T1–T4):
one captured graph re-runs all 48 layers for T tokens; the CPU expert pool runs
between layers (doorbell flags A/B/C). Window counts: T4×58, T1×29, T2/T3×12.
Per-window host phases (engine log): wait-for-rings 28.0 ms + pool 13.3 ms +
commit 1.0 ms + MTP draft 2.3 ms ≈ 56 ms/round (≈ 24.2 ms/token under profile).
Per-graph-node busy was extracted from `CUDA_GRAPH_NODE_EVENTS` node-id blocks ×
kernel `graphNodeId` (script in §5) — the per-node breakdown (e.g. T4 round-head
`wait_flag_ge` node = 523 ms over 58 executions ≈ 9 ms each; `native_gu<18>` node
6.2 ms/58; `gr_down_multi` 3.1 ms/58) is available by re-running the §5 one-liner.

## 5. How to regenerate / extend the analysis

- Benchmark: `python3 bench/v100/bench.py <label> --gpu 0 --workers 24 --max-new 256
  --max-context 8192 --kv fp16 --stats` (256-token decode, canonical tokens).
- nsys capture: `bench/v100/s13_trace.py`-era command in `/tmp/s17_nsys_node.sh`
  (recreate if lost): `nsys profile -o Logs/gpu/<name> --force-overwrite true -t
  cuda --cuda-graph-trace=node ./build-sm70/strata <canonical flags>` under sudo
  `ulimit -l unlimited`, `CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64`.
  Export: `nsys export --type sqlite -o X.sqlite X.nsys-rep`.
- **New harness (created this stage, commit-ready): `bench/v100/s17_trace.py`** —
  per-kernel aggregation with grid/block/regs/smem, wait_flag_ge A/B/C attribution,
  per-graph-node busy, memcpy stats, decode-window auto-detection. Run:
  `python3 bench/v100/s17_trace.py Logs/gpu/<name>.sqlite`.
  Notes learned: `demangledName` is a StringIds FK (JOIN StringIds); `gridId` is
  per-graph-INSTANCE (unique per launch), stable graph identity = `graphNodeId`
  blocks from `CUDA_GRAPH_NODE_EVENTS` (node-id contiguity).

## 6. Source audit findings (attention / tensor cores / verify windows)

**Model attention architecture** (qwen4exp, 48 layers): 36 GDN (linear attention,
Gated DeltaNet recurrence) + 12 QSA (full attention) layers, at indices ≡ 3 mod 4.

**QSA geometry** (`qsa_real_shapes()`): n_head 24, n_head_kv 2 (GQA 12:1),
head_dim **256**, idx_n_head 4, idx_dim 128, idx_block 4, **idx_top_k 2048**
(selection width = min(n_kv, 2051)), page_size 512, KV fp16 (or int8 via `--kv
int8`), fp32 indexer keys (exactness contract), n_rot 64.

**QSA decode path (current default)** — `qsa_layer` in `src/core/layer.cpp`:
quantize acts → `project_bf16` indexer raw key → gemv_quantized attn_k/v →
norm+rope → `kv_append(_q8)` → `indexer_key_append` → gemv_quantized attn_q →
q/gate split (cudaMemcpy2D) → norm+rope q → project_bf16 idx q → norm+rope →
**selection: `g_fast_select` (default ON): `qsa_block_scores` + `qsa_block_topk`
(radix over blocks)** (else `qsa_index` + `topk_512` binary-lift one-block kernel)
→ **attention: `g_fast_attn` (default ON): `qsa_decode_attn_step`** =
`attn_chunk_kernel` (grid n_chunks×n_kv_heads, CHUNK=64 cells, 256 thr, online
softmax partials m/l/acc, reads K/V straight from the paged pool, no gather) +
`attn_merge_kernel` (24×HD) → `gate` (sigmoid) → quantize q8_1 →
gemv_quantized attn_output.
Flags: `--no-fast-attn`, `--no-fast-select` (A/B back to the P2.S2
gather+`qsa_attend_step` path), `--native-flash-attn-short` (diagnostic pinned
vector attention, ctx ≤ 256 only).

**Flash Attention verdict (preliminary, from source + profile):** the fast path
is *already* a flash-style chunked decode attention (IO-aware online softmax,
paged-pool direct reads, GQA head-sharing via the 12 heads/block sq tile).
Differences vs FA2-on-SM70: no WMMA/HMMA in the score inner loop (fp32 FMAs on
bf16-decoded K/V or int8-decoded), one KV-head's 12 query heads per block,
CHUNK=64. Grid 33×2=66 blocks < 90 SMs → **under-occupied**, 36.3 µs mean.
QSA attention core = 91.1 ms = **1.6 % of kernel busy** at ctx 8192 (selection
capped at 2051 cells). The QSA *layer* as a whole (attn projections via
bf16_f32_mmvf + norms + rope + gate + quantizes) is the bigger chunk (~5–8 %).
So: a SM70 FA port (WMMA 8×8×4, fp16 in/fp32 out) is feasible but the *core
attention* headroom is small (≤ ~1–2 % e2e); the projections (bf16 GEMV, also
memory-bound) dominate the layer. Verify with a per-QSA-layer stage-timing run
before investing (see §7 exp E4).

**Tensor cores:** the only WMMA/CUTLASS path in decode is… none of the hot
kernels. Decode GEMVs are all fp32-FMA (bf16 weights unpacked via
`__uint_as_float`, int8 codes × scale). WMMA appears only in prefill
(`cutlass_70_wmma_tensorop`, `magma_sgemmEx` bf16). The hot decode kernels
(`gr_down/up`, expert `native_*`, `attn_chunk`) are **memory/latency-bound GEMVs**,
so TC peak (≈112 TF fp16) is not the ceiling; the question is achieved bandwidth
vs HBM2 900 GB/s (PCIe card):

- `gr_down_multi`: 47.9 µs for ~6.55 MB of bf16 weights (320×10240×2B) + T×10240
  f32 activations → **~137 GB/s effective (≈15 % of peak)**. Grid **41 blocks** on
  90 SMs (45 % occupancy of SMs), 74 regs, 32 KB smem, one warp per row, 40
  sequential 16B-chunk iterations with a 5-step `warp_sum` each. **Latency-bound
  and under-parallelized → the clearest GPU win available.**
- `gr_up_multi`: 25.7 µs, grid 160/90 SMs (1.78 waves), 98 KB smem (2 blocks/SM
  max), one row per warp, K=320.
- `gr_norm_multi`: 14.1 µs, **grid 1 block** — RMS norm of 4×10240 f32 (40 MB?
  no: 10240×4 f32 = 160 KB) — single-block latency-bound.
- `route`: 9.5 µs, **1 block of 32×8**, 83 regs — expert routing latency-bound.
- `attn_chunk`: 36.3 µs, 66 blocks, 98 KB smem — under-occupied (66/90 SMs).
- `copy_i32_from_mapped`: 13.5 µs, 1 block × 128 thr, reads ≤16 ints of pinned
  mapped memory (the plan) — PCIe round-trip latency per call ×6,736.
- Expert `native_gu/native_down` (grid 160×10 / 320×10): IQ3_XXS weight streams,
  31–48 µs; `native_mmvq_multi` 24–101 µs (the 101 µs one: 640 blocks, 32×4 thr
  blocks = 1,280 warps of 16 → check block-level latency).

**GDN mixer math (all 48 layers run the `gr` hyper-connection read):** per window
48 × (6.55 MB w_down + 6.55 MB w_up) ≈ 629 MB of bf16 weight traffic shared across
the window's T tokens (that's what the `*_multi` fusion buys). 111 windows ≈
70 GB over 6.2 s. At 900 GB/s the floor is ~78 ms; we spend 990 ms → **~12× off
peak; pure latency/parallelism loss.** Split-K (more blocks, partial reduce) is
the natural fix and numerics-preserving if the per-row accumulation order is kept
(warp-per-row order is the parity contract — a K-split needs an fp32 partial sum
joined in a fixed order, which CHANGES rounding vs the current single-warp serial
order → must be validated against the golden; keep the fused single-phase behavior
intact otherwise).

## 7. Candidate experiments (ranked by measured headroom; NONE started yet)

Constraint reminders: one change at a time; numerics-preserving unless we accept
a separate-mode flag; bench = 256-token decode ×3 interleaved + 32/32 golden +
256/256 determinism; final validation on GPU4 16 GB.

- **E1 — gr_down_multi split-K parallelism** (biggest single item: 541 ms).
  Grid 41 → 41×KSPLIT (KSPLIT 2–4), each block handles a K-slice (5120/2560),
  warp-per-row order inside a slice preserved; cross-block partials via a small
  reduce kernel or `atomicAdd` (check order sensitivity vs golden; if rounding
  differs, expose as opt-in or prove 0-token difference over 256). Expected
  30–50 % of 541 ms if latency-bound → 2–4 % e2e. Files: `src/kernels/cuda/fused_gr.cu`
  (`gr_down_multi_kernel`), caller `fused_gr_read_multi`.
- **E2 — gr_norm_multi parallelization** (159 ms, grid 1 block): split the
  4×10240 f32 (4 streams) across blocks with a 2-pass rsqrt (partial ss →
  reduce → scale). Order: ss per stream is a full 10240 sum — fp32 partial sums
  change order → validate golden. Expect most of 159 ms.
- **E3 — route kernel parallelism** (149 ms, 1 block 9.5 µs): block-per-expert
  chunk + warp top-k, or simply a bigger single block (256→1024) if it's a
  serial scan; check `router_top10.cu`/`route` source first. Expect 50–70 % of it.
- **E4 — measure QSA attention properly** before any FA work: run
  `--stage-timing --no-capture --no-pool` (GPU floor, per-stage table splits
  attention block by GDN vs QSA per layer) + `--no-fast-attn` / `--no-fast-select`
  A/B benches, to get the QSA-layer total and the fast-vs-slow attention delta.
  Then decide: optimize `attn_chunk` (more chunks in flight / smaller CHUNK /
  2 heads per block) vs leave it (1.6 % core).
- **E5 — copy_i32_from_mapped / copy_from_mapped** (249 ms): bigger blocks, or
  fold the plan read into the next kernel's prologue; these ride PCIe latency
  per call.
- **E6 — round-head wait overlap** (the 1,023 ms of big waits): check whether the
  verify-window graph can run a plan-independent prefix (embed/PLE/gr-read of
  layer 0) *before* the flag-A wait node, i.e. move the first wait later in the
  graph; or overlap MTP draft with the round head (already partly the case —
  verify the timeline in nsys before touching). Highest prize, highest risk
  (graph topology + doorbell semantics).
- **E7 — launch-overhead sweep**: 376 k kernels/256 tok ≈ 1.5 k/token; the
  77.9 k elementwise small kernels (211 ms, 2.7 µs mean) are launch/latency
  bound. Graphs already hide most of it (93.8 % busy); measure the residual
  inter-kernel gaps per window (the s13 round-gap analysis) before any fusion.
- **Explicitly deprioritized (measured small):** FlashAttention WMMA rewrite of
  the QSA core (≤1–2 % ceiling, numerics-risky); int8/TC paths for the expert
  GEMVs (weights are 3-bit — dequant cost already fused; memory-bound on codes);
  GDN core kernels (220 ms, already T-fused).

## 8. Remaining todo list (resume order)

1. ✅ Baseline established (§3). ✅ Profile + breakdown (§4). ✅ Source audit (§6).
2. E4 first (cheap, decides the attention investment): stage-timing GPU-floor run
   + `--no-fast-attn`/`--no-fast-select` A/B.
3. E1 (gr_down split-K): implement → bench ×3 interleaved → 32/32 golden →
   256/256 determinism → keep/revert. (E2, E3 same loop.)
4. E5/E6 if E1–E3 leave room; E7 only if measured gaps justify it.
5. Final candidate: re-run baseline comparison ×3 on GPU0 + full 16 GB (GPU4)
   validation (32/32 golden there, 256/256, peak VRAM, no CUDA errors).
6. Docs: `Docs/v100-stage1.7-final.md` (13 required report sections), update
   `Docs/STATE.md`, `Docs/CHANGELOG.md`, `Docs/v100-performance.md`; raw data in
   `Logs/{benchmarks,cpu,gpu}` (keep the existing `nsys-*` gitignore policy for
   large captures); short logical commits, preserve the Stage 1.6 checkpoint.
7. **Restart `strata.service`** (was stopped for profiling) and verify :8180.
8. STOP at Stage 1.7 — do not start 1.8.

## 9. Correctness bar (unchanged from 1.1–1.6)

- 32/32 golden prefix (§3 bar) on GPU0 for every candidate run.
- 256/256 deterministic (two identical runs byte-identical) per card.
- No CUDA errors / hangs / races; ctest 20/22 (2 pre-existing environmental).
- Cross-card note: GPU4 16 GB diverges from GPU0 on ~96–108 of 256 tokens
  pre-existing (different resident set) — compare GPU4-vs-GPU4, not vs GPU0.
