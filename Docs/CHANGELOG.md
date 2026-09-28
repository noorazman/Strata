# CHANGELOG

Important historical changes and decisions. No raw logs.

## 2026 — V100 Stage 1.9 (branch `stage1.3-expert-pool-sync`, GPU0 only)

`wait_flag_ge` A/B/C dependency analysis (profile-only stage, no engine changes; binary
`90afb6c`). Question: understand the wait bottleneck before changing any synchronization
code — why 3 waits, which are necessary, what each contributes, can they be merged/
overlapped/moved/eliminated, and is this the Stage 1.6 flag-C visibility issue.
**Verdict: the 3-wait A/B/C protocol is the original design (first commit `f2a08d4`) and all
three waits are load-bearing; the dominant cost is the round-head flag-A visibility lag
(63 % of all wait time) — the Stage 1.6 boundary mechanism landing on the A slot.**

- **Classifier + captures (`bench/v100/s19.sh`, `bench/v100/s19_trace.py`).** nsys
  `--cuda-graph-trace=node` 256-tok fp16/8192 (s19-base) and 256-tok int8/32768 (s19-int8)
  on the current binary + the same-workload old-binary capture (s17x, Stage 1.6 code) +
  the 64-tok F1 source (s18nsys-base). The classifier anchors wait-A on the following
  `copy_i32_from_mapped_kernel` (exactly 1/3 of waits in every capture) and walks the
  fixed A→B→C cycle inside the 48-dispatch round; it reproduces the known s17x and
  s18/F1 totals to the digit (15,984 waits; 503.5 ms = 7,868 µs/tok). Staging streams
  identified by stream id: verify staging = 17 (1,800 copies / 3.11 GB = 0.34
  distinct experts/layer, identical old vs new), adaptive-tier refill = 19 (exactly
  2,503 copies = the engine's swap count).
- **Q1/Q2 (why 3, which are necessary).** Both binaries issue exactly 3 waits/dispatch —
  F1's "1→3" was the ring count (5,328) vs the wait count (15,984); per-dispatch wait is
  308.5 µs (old) → 313.3 µs (new), +1.6 % = no binary regression. F1's 420 µs was the
  64-tok cold window (539.9/449.6/368.3 ms cold stalls + a 20.2 ms round-0 C outlier +
  25-round statistics). A is always necessary (plan gate; publish sfence max 22 µs makes
  it a valid gate for the WB plan data); B is necessary when ≥1 blob staged (≈39/48
  dispatches; 0-blob dispatches pay a ~3–5 µs spin tax ≈ 0.3 % e2e); C is necessary
  whenever the CPU share is non-empty (82.8 % cache hits, 3.63 distinct CPU experts/layer
  → essentially always).
- **Q3 (time each contributes).** 256-tok fp16 (111 rounds): total wait 1,672.3 ms =
  6,532 µs/tok = 29.9 % of the main stream. A 1,103.7 ms (66.0 %): round-head 1,058.4 ms
  = 9.54 ms/round (max 14.6) vs mid-round 8.7 µs mean. B 196.7 ms (11.8 %): head 0.50
  ms/round, mid 27.0 µs, 10 % DMA-tied. C 371.9 ms (22.2 %): head 0.30 ms/round, mid
  64.9 µs. int8/32768: 1,754.8 ms = 6,855 µs/tok (A 63.3 / B 22.2 / C 14.5 %). >1 ms
  cohort: 150 waits = 1,157.6 ms = 69.2 % of all wait time; 125/150 at the round head;
  111 of those are flag A (one per round). Host side of A: doorbell→publish mean 75 µs,
  max 376 µs, 0 > 1 ms.
- **Q4 (merge/overlap/move/eliminate).** Keep 3 waits. Merging B+C turns `max(B+X,C)`
  into `max(B,C)+X` — the current structure overlaps the PCIe-grouped kernels with the
  pool tail and wins in the common C ≥ B case (measured mid C 65 µs ≥ B 27 µs). waitB on
  a side stream is wall-neutral (the PCIe group is DMA-gated either way). Mid-round A
  (8.7 µs) is at the floor (the plan cannot be known before the `moe_route`→doorbell).
  Only removable small item: the ~0.1 ms/round zero-blob waitB tax (needs conditional
  graph nodes).
- **Q5 (relation to Stage 1.6).** Same mechanism, different slot: one ~9–14.6 ms spin per
  round at dispatch 0 — the host publishes A ≤ 376 µs after the doorbell (the store is in
  DRAM on return) but the GPU's read of the mapped line stays stale ~9–14 ms at every
  window boundary. The 1.6-era captures show the identical shape in the l0 flag-C slot
  (25,564 poll iters); the slot moved because the A wait now straddles the boundary
  (waitA starts at doorbell+~15 µs, the A store lands at +80 µs). Host corroboration:
  l0 ring-wait 0.000 ms/round, the slow host-side spin is the l1/g0 shadow (7.9–8.2 ms =
  the GPU's `post(0)`+`pre(1)`), round-end host path 2.90 ms/round (run→commit 0.89 +
  commit→draft 1.98 + draft→run 0.03). The round-0 C outlier (19.8–29.0 ms across
  captures) is the separate first-round cold-start effect.
- **Root cause (confirmed) + theoretical opportunity.** (1) the 3-flag structure is
  correct, per-dispatch spin ~310 µs, no regression; (2) 63 % of all wait time is the
  round-head flag-A visibility lag (1,058.4 ms); (3) the rest is mid-round C pool tail
  (338.8 ms) + B copy-engine queue state (141.1 ms). The lag is on the critical path with
  the host already ready: round-head A → mid-level removes 2,398 µs/tok of wait → e2e
  50.8 → ~64 tok/s (+26 % ceiling; ~+30 % if head B/C normalize). 3× the entire QSA
  attention mechanism (1,317 µs/tok, Stage 1.8).
- **Recommended next experiment (proposed, NOT implemented):** `STRATA_WAIT_FENCE=1` —
  env-gated periodic `__threadfence_system()` every ~1024 poll iterations inside
  `wait_flag_ge_kernel` (today it fences only after the loop exits); direct test of the
  stale-mapped-line hypothesis; arm = nsys round-head A distribution + e2e A/B with
  golden/determinism gates. Fallbacks: pre-warm flag-line read at the window head, then
  MTP-draft reordering.
- **Correctness:** 32-tok 32/32 golden (`cf577e…`), 256-tok ×2 byte-identical = golden
  `c1517d02473fbc06b5cf415ea1f8be63` (50.77/50.76 tok/s, 111 rounds), hostdiag arm 50.76
  tok/s with all four host diagnostics on (wait for rings 26.04 / pool 12.57 / commit
  0.86 ms/round; expert-cache 82.8 % hits), all runs rc=0, no CUDA errors/hangs.
  `strata.service` stopped for the campaign and left stopped (stage instruction).
  Doc: `Docs/v100-stage1.9-final.md`.

## 2026 — V100 Stage 1.8 (branch `stage1.3-expert-pool-sync`, GPU0 only)

QSA attention-path analysis (profile-only stage, no engine changes; binary `0fea54d`).
Question: can attention be meaningfully accelerated on V100 SM70 before implementing
anything? Verdict: **the QSA attention mechanism is 5.8 % of the token (ceiling ~7 % e2e);
the pure attention kernel is 0.8 % (ceiling ~1 %). Attention is not a major e2e bottleneck.**

- **Full QSA path breakdown (nsys, `--cuda-graph-trace=node`, 64-tok window, 4 arms,
  stream-order segment decoder `bench/v100/s18_trace.py`).** Main-stream kernel time
  22,731 µs/tok (fp16 base) / 22,644 (int8). QSA attention mechanism (projections +
  indexer + selection + attention + small kernels, MoE tail excluded) = **1,317 µs/tok**:
  Q GEMV 287.2, attn_chunk 155.2 (33.1 µs/call, 66 blocks single wave), O GEMV 147.4,
  Q/idxq norm 90.1, idxq proj 89.5, Q/idxq rope 68.7, idxk proj 68.5, attn quant 51.6,
  K norm 46.8, KV append 45.8, K GEMV 40.4, idxk append 38.8, gate 38.0, K rope 35.2,
  V GEMV 33.7, attn merge 27.4, sel block scores 17.7, staging 13.3, act quant 11.5,
  sel block topk 10.7. int8 mechanism 1,356 µs/tok (+3 %, int8 KV ≈ fp16 KV speed).
- **Fast-vs-slow A/B (E4 `--no-fast-attn`/`--no-fast-select`, 256 tok, `drop_caches`).**
  fp16 base 49.15/49.86/49.11 tok/s all GOLDEN `c1517d…`; nofastattn 46.30/45.00/44.40
  (−8.4 %, trajectory diverges tok 42, kernel +912 µs/tok, slow attention 6.0× fast core
  per invocation); nofastsel 47.35/47.43/48.46 (−3.3 %, **bit-identical trajectory** =
  pure cost, +624 µs/tok, slow selection 23× fast); int8 base 48.60; int8 nofastattn
  43.81 (−9.9 %); int8 nofastsel **20.17 (−58.6 %)** — MTP draft 8.591 vs 1.969 ms/round
  + degenerate loop at tok 221; slow selection is a diagnostic arm only under int8.
  All 15 runs rc=0, 3/3 deterministic per fp16 arm, no CUDA errors/hangs.
- **Attention-kernel microbench (SM70, production shape G=12/HD=256/CHUNK=64/256 thr,
  66 blocks, 4.33 MB/launch).** Production-structure replica 33.7 µs/launch (2,057
  GFLOP/s = 13 % FP32 peak, 128 GB/s = 14 % HBM; roofline 5–8 µs → latency-bound).
  Naive WMMA fp16 54.9 µs (slower: 12→16 row padding + 3 syncs + shared staging);
  register-resident WMMA 30.5 µs (marginal, needs fp16 Q = numerics change); CHUNK
  sweep on the production structure: c32 25.2 µs (−25 %), c16 23.4 µs (−31 %), c128
  59.8 µs — **chunk size is the one free lever (no numerics change)**. Toolchain note:
  `wmma::load/store_matrix_sync` from/to local (register) memory faults on CUDA
  12.8/12.9 sm_70 (`unspecified launch failure`); stage through shared memory.
- **Theoretical max e2e gain:** whole QSA mechanism → 0 = 5,185 − 337 ms → 52.8 tok/s
  (+6.9 % fp16 / +7.1 % int8); attention core → 0 = +0.9 %; core 4× faster = +0.7 %.
- **Rejected by measurement:** WMMA fp16 now (54.9/30.5 vs 33.7 µs — no clear win, 25 %
  mma waste, needs fp16-Q mode; kept open for a later int8-TC experiment), FlashAttention
  rewrite (kernel is already tiled 64-cell + online softmax via merge), KV-cache layout
  (KV append 46 µs/tok; int8 already neutral for attention), bigger GEMV batches (already
  T-batched per verify window).
- **Largest single finding (not attention):** `wait_flag_ge` spin = 7,868 µs/tok = 34 %
  of the main stream (503.5 ms in the 64-tok window) — the largest decode kernel class,
  larger than the whole QSA mechanism. Structural: wait calls per layer-round 1.0 (old
  binary, s17x) → 3.0 (new); spin per layer-round 309 → 420 µs. Flag A/B/C re-attribution
  on the new binary is the top next-stage profiling item.
- **Proposed next experiment (NOT implemented):** E1 = `STRATA_QSA_CHUNK` knob (64
  default, 32 opt-in): expect chunk 33.7 → ~25 µs, e2e +0.1–0.15 tok/s; golden-gated
  (the online-softmax merge is partition-invariant — verify bit-identity, don't assume).
  Follow-ups ranked: Q/O int8-TC GEMV path (targets 508 µs/tok of projections, +1–1.2 %
  e2e, opt-in), small-kernel fusion (ceiling ~1.5–2 %, cuts ~96 launches/tok), wait_flag
  re-attribution.
- Deliverables: `Docs/v100-stage1.8-attention-analysis.md`, `bench/v100/s18{e4,e4b,nsys}.sh`,
  `bench/v100/s18check.py`, `bench/v100/s18_trace.py`, raw data `Logs/benchmarks/s18e4*.{json,log}`
  + `Logs/benchmarks/s18e4b-campaign.log` + `Logs/gpu/s18nsys-{base,nofastattn,nofastsel,int8-base}.{nsys-rep,sqlite}`.

## 2026 — V100 Stage 1.7 (branch `stage1.3-expert-pool-sync`, GPU0 only)

GPU kernel experiments: profiled the decode GPU kernel breakdown, ran three env-gated
kernels, root-caused the all-on regression, and finalized the keepers. Production target
is the 32 GB V100 (GPU0) only (16 GB / GPU4 not validated this stage).

- **Profiled the decode GPU kernels (nsys 2024.6.2, `--cuda-graph-trace=node`).** The
  MTP/draft kernels and all memcpy are byte-identical across the base/e12/allon
  captures, so the all-on regression (~3.5 %) is 100 % attributable to the three
  experiment kernels themselves. Per-kernel: `gr_down_multi` 47.99 µs (base) vs
  `gr_down_multi_split`+`gr_down_multi_reduce` 54.16+5.60 µs (**E1 +133.1 ms**);
  `gr_norm_multi` 14.04 µs (base) vs `gr_norm_multi_split` 7.86 µs (**E2 −69.8 ms**);
  `route` 9.51 µs (base) vs `route_sort` 15.90 µs (**E3 +100.0 ms**). Net all-on
  +163.3 ms ≈ 3.2 %, matching the raw-bench +3.5 %.
- **KEEP E2 `STRATA_GR_NORM_SPLIT` (per-stream gr_norm split), now the default (ON).**
  The norm was latency-bound at 1 block/token; splitting to 4 blocks/token (one per HC
  stream) halves the kernel time (−44 %) with no scratch round-trip. Bitwise-identical
  to the single-block norm; adds no VRAM. `STRATA_GR_NORM_SPLIT=0` keeps the single block.
  End-to-end on GPU0 (256 tok, 12 tight interleaved pairs, `drop_caches` between):
  baseline 5211 ms (49.13 tok/s) vs E2 5194.9 ms (49.28 tok/s) = **−16.2 ms (−0.31 %),
  +0.15 tok/s**, E2 faster in 9/12 pairs; outlier-trimmed −58.7 ms (−1.13 %), consistent
  with the kernel-level −69.8 ms.
- **REVERT E1 `STRATA_GR_DOWN_SPLIT` (gr_down K-split) to default OFF (opt-in `=1`).**
  The base `gr_down_multi` already runs a single wave (41 blocks < 90 SMs); K-splitting
  to 164 blocks does not shorten the critical path but adds register pressure (95 vs 74
  regs → 2 blocks/SM), a `part`-scratch global round-trip, and a separate reduce-kernel
  launch. Its 320 KiB `grdown_part_` scratch (per MTP + verify pool) is now reserved
  only when E1 is enabled (`fused_gr_down_split_enabled()`), so the production
  (E1-off) footprint no longer carries the unused scratch. (A prior E1 bug — the
  inject-block part-row collision — was found and fixed earlier and is bitwise-verified.)
- **REVERT E3 `STRATA_ROUTE_SORT` (bitonic-sort top-10) to default OFF (opt-in `=1`).**
  The 16-element bitonic sort is 1.68× slower than the iterative-argmax `route` scan
  (15.90 vs 9.51 µs) inside the same single-block single-warp kernel.
- **E4 (slow attention / selection A/B) plumbing** is in place and opt-in
  (`--no-fast-attn` / `--no-fast-select`): the verify window reproduces the fast OR the
  slow (gather + one-block-per-head / cell top-k) QSA arithmetic, so attention can be
  A/B'd inside the window. `layer_verify_compatible` no longer gates on
  `g_fast_attn`/`g_fast_select`; the short flash-attention adapter still does.
- **Correctness:** 32/32 golden every run (golden 256-token md5
  `c1517d02473fbc06b5cf415ea1f8be63`); 256/256 byte-identical deterministic re-runs (E2
  output byte-identical to the all-off baseline); 0 CUDA errors / no hangs; ctest 20/22
  (the 2 failures pre-existing/environmental: `ple_parity` missing Q2_0 shard,
  `platform_memory_test` mlock ulimit). **VRAM/RAM:** peak VRAM 18,900 MiB identical for
  baseline and E2; host RAM stable.

Doc: `Docs/v100-stage1.7-final.md` (state/checkpoint `Docs/v100-stage1.7-state.md`).
Raw data: `Logs/benchmarks/s17{f,x,c}-*` and `Logs/gpu/s17x-*.nsys-rep/.sqlite`.

## 2026 — V100 Stage 1.6 (branch `stage1.3-expert-pool-sync`, on top of the 1.5 commit `95a338e`)

CPU pool scheduling optimization (the seven 1.6 candidates: affinity/governor, SMT/core placement, keep-frequency-high, pool scheduling/drain, park-threshold re-test, the "l1/g0" ring-spin tail, cross-layer pipelining). Profile-first, one-change-at-a-time; kept only what measures end-to-end. Net e2e **48.7 → 50.0–50.1 tok/s (+~3 %)**.

- **Bottleneck found (profiled):** the remaining CPU-side cost is the **round-head wait-C (layer-0 flag-C) visibility lag, 9–14 ms uniform** (~25 % of the ~51 ms round). Measured hop by hop: workers raise flag-C ≤ 0.11–0.20 ms after dispatch; the WC flag store+`sfence` is ≤ 12 µs; and the new `STRATA_WAIT_ITERS=1` probe shows the GPU was polling normally (25,564 iterations at the ~9.3–14 ms spin = ~364–500 ns/iter) — so the flag value is genuinely invisible to the GPU for the full spin even though the CPU-side store+fence completes in µs. Isolation microbenchmarks (`/tmp/flagvis*.cu`) show ~immediate CPU→GPU visibility in every configuration (host/worker core, WC/WB, posted/non-posted, cross-core reset+write, +24 DRAM-traffic threads), so the lag is **context-specific to the window boundary** (PCIe/memory-controller/IOMMU interaction under the round's concurrent staging DMA + MTP, on a socket-0 DRAM row with a steady ~1/s corrected-MCE stream on MC_CHA bank 5; MCE rate is 1.0/s at idle and 1.0/s under load → the MCEs contribute jitter but are not the primary cause).
- **Keepers:** (1) **Arena pinned to the host thread's NUMA node by default** (`arena_preferred_node()` = `main_thread_node()`, `STRATA_ARENA_NODE=<n>` override) — the 40 GB arena's pages previously first-touched non-deterministically on node 1 (loader-thread placement); pinning to node0 (where the host, the 26.8 GB PLE and 13/24 workers sit) gave pool/drain 11.88 → 12.97–13.42 ms/tok and 48.70 → 49.08–50.07 tok/s (**+1.8 % e2e, drain −11 %**). (2) **Async rows dispatch** (`STRATA_POOL_ASYNC`, default ON; `src/kernels/cpu/pool.cpp` `dispatch_multi_native_async` + `ExpertDispatch::rows_async`) — the fused rows phase fires without blocking the host's layer loop; the workers (and the host when it drains) finish the rows and raise flag-C themselves, last-participant-ordered after the row stores, with synchronous fallbacks for empty-plan/njobs==0/njobs>96/`STRATA_POOL_UNFUSE`/in-flight. e2e-neutral standalone (48.99/50.85 vs sync 49.44/49.99 tok/s), golden-passing — a robustness keeper the round-head fix would build on. (3) **Write-combined host→GPU flags + non-posted publish** — `h_flag_`/`h_flagA_`/`h_flagB_` are `cudaHostAllocMapped|WriteCombined`; every flag publish is a plain store + `lock addl $0` (a non-posted locked RMW that retires only once the line's write-back has completed) + `_mm_sfence()`, so the value is in DRAM by the time the store sequence ends (a posted store+sfence retires only when the write is *issued*). flag-B was already a non-posted `cmpxchg`. The row bulk `h_ymiss_` and the plan stay WB (worker streams). (4) `STRATA_POOL_CORES` placement knob.
- **Rejected by measurement (reverted):** SMT workers on node1 (43.15/42.01 tok/s, −11 %) and SMT-node0 workers (42 tok/s) — cross-QPI row streams dominated; governor `performance`/min_freq 2.9 GHz (e2e-neutral — the workers are dispatch-bound, not frequency-bound; deployment recommendation only); park threshold (kept 2048 µs: 512→46.8, 1024→49.5, 2048→48.75 tok/s avg, within MCE noise); full WC (flags+ymiss+plan) — ymiss/plan WC reverted to WB after flags-only WC proved equivalent.
- **Probes added (all off by default, zero-cost when off):** `STRATA_POOL_ASYNC_DIAG=1` (worker wall top-10 slow phases), `STRATA_SFENCE_DIAG=1` (per-site sfence max/avg/total), `STRATA_WAIT_ITERS=1` (per-slot max GPU poll-iteration counts — the probe that proved the GPU polls normally), `STRATA_HEAD_DIAG=1` (round-end host-path decomposition), `STRATA_ALTFLAGC=1` (route the round's flag-C through a fresh page — the address A/B; 23,302 → 15,819 wait-iters, 50.11 vs 50.00 tok/s interleaved, opt-in).
- **Correctness:** 32/32 golden in every run; 256/256 deterministic ×2 (alt-flag and original-flag outputs byte-identical); 16 GB (GPU4) 48.87 tok/s, first-32 identical to the GPU0 golden, 96/256 cross-card diffs (≈ the ~108 pre-existing), no CUDA errors; ctest 20/22 (same 2 pre-existing environmental: `ple_parity`, `platform_memory_test`).
- **Remaining bottleneck (documented, not fully fixable in software):** the ~13 ms round-head flag-C visibility lag. A 13 ms round-head removal would move ~50 → ~65 tok/s if the rest of the round does not stretch.

## 2026 — V100 Stage 1.5 (branch `stage1.3-expert-pool-sync`, commit `95a338e`)

ExpertPool idle parking. The 24 pinned expert-pool workers spin-parked on `_mm_pause()` between batches and burned 24 full logical cores at 100 % while the engine sat idle between requests. Fix (default): park after `STRATA_POOL_PARK` (2048 µs) of spin — idle CPU 24.0 → ~0 cores, decode-neutral (48.9–49.3 tok/s, 32/32 golden, 256/256 deterministic). The park-gap histogram (`STRATA_POOL_PARK_DIAG`) showed 90 % of parks are 256 µs–1 ms and 3.1 % ≥ 4 ms, so no small-gap regime a tiny spin window would protect. Doc: `Docs/v100-stage1.5-final.md`.

## 2026 — V100 Stage 1.4 (branch `stage1.3-expert-pool-sync`, on top of the 1.3 commits @ b407beb)

CPU row production (flag C) optimization. Deliverables: fused single-phase CPU expert pool (default), decode-once AVX2 expert dots (opt-in) + bit-exactness parity harness, Flag C instrumentation, `Docs/v100-stage1.4-final.md` + `Docs/v100-performance.md` (appended) + STATE/CHANGELOG updates, raw data `Logs/benchmarks/s14-*`, `Logs/gpu/nsys-s14-*`. Stage 1.3 frozen (no pcie-frac/PLE/KV/spec/FA changes).

- **Root cause of flag C (measured, not inferred):** 360.5 ms of GPU `wait_flag_ge` on `h_flag_` = 5,328 per-dispatch waits (48 dispatches/round × 111 rounds, mean 67.7 µs, nsys `Logs/gpu/nsys-s14-base.sqlite`). The waits were not for CPU capacity — the pool's 25 threads were already 82–83 % busy (20.5 thread-equiv, `park-wait 0.4 ms`) while the box-wide CPU mean was ~7 % (the other ~31 cores are not in the expert path). The down-projection rows (the ones flag C gates) were serialized behind **two worker barriers plus a host-only intermediate-quantization loop** between them in `ExpertPool::run_split_multi_native` (baseline modes 5+6): no down row could start until every gate/up chunk on every thread finished AND the dispatch thread alone quantized every expert's intermediate (9,130 phases/run; pool 14.376 ms/round inside the verify window vs ~40.9 ms/round total, while decode pace consumes ~19.6 ms/token × ~2.3 tok/round).
- **Change: fused single-phase expert pool (mode 7, now the default)** — gate/up row chunks accumulate a per-expert release counter (`fgate_` atomic); per-expert quantization runs as pool *tasks* (workers overlap it with other experts' work); down chunks wait per-expert at expert boundaries. Same row work, same kernels, same per-row numerics; one barrier and one host-serial step removed per dispatch. `STRATA_POOL_UNFUSE=1` restores the two-phase path. Measured: pool wall 1,358 → 1,205 ms/256 tok (−11.2 % A/B), pool threads 92–93 % busy (23.1–23.2 thread-equiv, 4,565 phases), flag C 360.5 → 193.6–292.3 ms nsys (−19 to −46 %; total `wait_flag_ge` flat ~1.64–1.72 s), decode **48.88/48.62 → 50.99/51.01 tok/s GPU0 (+4.5–5.1 %)**, 47.56/47.20 → 47.83/48.25 GPU4, per-token latency 20.5 → 19.61 ms, VRAM 18,896/16,133 MiB unchanged. The e2e gain is fully accounted for by the pool term of the verify window (14.376 → 12.386 ms/round; wait-for-rings and commit unchanged).
- **H3: decode-once AVX2 expert rows (opt-in `STRATA_IQAVX2=1`, off by default)** — ggml-cpu's AVX2 dots re-decode weight codebooks per token; the new kernels (`src/kernels/cpu/iq_avx2.cpp`) decode each (row, block, sub-block) once and share the integer code/sign expansion across the nt tokens, reproducing the per-token AVX2 op sequence exactly (bit-identical; two real bugs found and fixed on the way: a `keven_signs_q2xs` table transcribed with 1,023 of 1,024 entries, and `_mm256_castsi128_si256` where the original replicates both 128-bit halves — VPSHUFB indexes per-lane, so the high-half sign bytes of IQ2_XS codes 2/3 were indexing zeros). Parity harness `src/kernels/iq_avx2_parity.cpp`: 122,880 rows × nt 1..4 bit-exact vs the per-token ggml-cpu AVX2 dot, all five gu formats (16/17/18/21/22), both GGUF shards; down Q2_K/Q2_0 stay on the per-token path by design. Interleaved 2×2 A/B is e2e-neutral (50.45 vs 50.64 tok/s, pool 12.44 vs 12.31 ms/tok — inside the ±1–2 tok/s MCE drift), consistent with only the gu rows (1/3 of the dot FLOPs) being affected while the pool is compute-bound → kept opt-in per the stage rule, not default.
- **Rejected by measurement (reverted to default):** H1a `--no-host-worker` (49.21 tok/s, pool wall +2.4 % — the host thread's ~1 thread-equiv outweighs its dispatch-loop interference); H3 as default (e2e-neutral).
- **Correctness:** H2 is bit-exact end-to-end — the final 256-token sequence on GPU0 is 0-token-different from the 1.4 baseline sequence (fused re-schedules only); H3 likewise (on vs off). 32/32 golden on both cards; 256/256 token-identical deterministic re-runs per card; the cross-card divergence from token 136 (108 tokens) is identical in pattern to the 1.3 final runs (pre-existing, different resident sets past the golden window). ctest 20/22 (same 2 pre-existing environmental failures as 1.3); no CUDA errors, no hangs, no VRAM regression.
- **Remaining bottleneck for Stage 1.5 (documented, not touched):** flag A 1,100.9 ms (dominant GPU-visible wait; early-layer l1/g0 13.4–13.7 ms spins — frozen in this stage per scope), residual flag C 193.6 ms (the serialized per-dispatch host path: plan + activation quantize + job publish ≈ 1.9 ms/round × 48 dispatches still exceeds the GPU's per-round consumption), flag B 372.9 ms (staging tail at pcie-frac 0.2; a 0.25–0.35 sweep is a noted candidate), round-boundary gap (commit 0.857 + MTP draft ~2.1 ms/round, frozen).

## 2026 — V100 Stage 1.3 (branch `stage1.3-expert-pool-sync`, from `stage1.2b-ple-ram-default` @ e4ebe92)

Expert pool synchronization optimization. Deliverables: `--pcie-frac 0.2` as the native-pack default, sync diagnostics in the stats output, `Docs/v100-stage1.3-final.md` + `Docs/v100-performance.md` (appended) + STATE/CHANGELOG updates. No PLE/pool-workers/KV/speculation/Flash-Attention/dense changes.

- **Finding (nsys 2024.6.2, 256-token decode, `wait_flag_ge` lifecycle):** at the 0.55 baseline the GPU's 2273.6 ms of `wait_flag_ge` time was 1338.9 ms (64 %) of **flag B waiting on the PCIe staging DMA** (the next global event after those waits is the staging copy on the copy engine — measured, not inferred), 769.0 ms flag A (plan), 34.2 ms flag C (CPU rows). The CPU expert pool sat at 7–9 % of the box while the GPU idled on ~1.9 staged expert blobs/layer (45.1 GB H2D per decode).
- **Change: `--pcie-frac` default 0.55 → 0.2 for native packs** (`Options::pcie_frac`, `src/program/generate.cpp`): 0.2 stages 51/256 experts (0.34 distinct/layer) and lets the CPU pool compute the rest (3.63 distinct/layer). Measured curve on this machine: 0.0→47.33, 0.25→43.93 (fails the 32 GB golden at token 6), 0.35→46.31, **0.2→48.62/48.87 tok/s**, 0.55→43.48/43.60 (same-window baseline). Result: total GPU wait 1604.4 ms (−29 %), flag B 1338.9→385.6 ms, flag C 34.2→376.6 ms (CPU rows become the new critical path), H2D 31.6 GB (−30 %), VRAM 18,896 / 16,133 MiB unchanged on both cards. 16 GB (GPU4) survives at 47.56/47.20 tok/s; its 32-token golden is now identical to the 32 GB golden (was divergent at token 6 under 0.55).
- **Rejected by measurement (reverted):** `--pcie-mode direct` (35.81 tok/s, grouped reads of the mapped arena ~30 % slower than VRAM grouped); spin-flush 64 µs instead of 2 ms (interleaved same-window A/B at 0.2: 64 µs = 48.15/48.53/48.82 vs 2 ms = 48.49/49.00/49.27 tok/s — the original 2 ms won every adjacent pair by ~0.4 tok/s; the big l1/g0 13–17 ms ring spins are unchanged at 64 µs, so it is not a device-write visibility effect); commit-graph overlap (dropping `cudaStreamSynchronize` in `Verifier::commit`): the ~0.77 ms/round host saving is cancelled by a +0.30 ms/round MTP-draft slowdown (commit GPU work shares SMs with the draft) — net wash, kept synchronous.
- **Diagnostics added (stats only, `--stats` output):** verify round head (layer 0 wait/pool), top-8 slowest spins/pools with decode-relative timestamps, top-8 slowest pool dispatches with the plan/actq/jobs/run (gu/quant/down) phase split. Root-caused the residual tail: one-time 8 ms `actq` cold start on the first prefill call and ~100 ms clusters of ~5× slower pool `gu` phases (machine-wide CPU transients; a pre-existing stream of correctable MCE machine-check events is present in `dmesg`), cascading into the l1/g0 13–17 ms ring spins.
- **Correctness:** 32/32 golden on both cards (32 GB golden: `271 248068 198 760 1156 369 30869 5402 430 328 23202 1288 264 11952 5617 303 ...`); 256/256 token-identical deterministic re-runs on both cards; ctest 20/22 (the 2 failures pre-existing/environmental: `ple_parity` — its Q2_0 reference shard is absent on this machine; `platform_memory_test` — needs `ulimit -l unlimited`, passes under it); no CUDA errors, no hangs, no VRAM regression.
- **Environment notes:** correctable MCE stream (3,142 `dmesg` entries the day of testing, EDAC ce=0, no UE) likely contributes to the ±1–2 tok/s run-to-run variance; two concurrent strata processes (~67 GB RSS each) OOM-kill on NUMA node 1 of this 125.78 GiB box — validation runs are sequential.

## 2026 — V100 Stage 1.2B (branch `stage1.2b-ple-ram-default`, from `stage1.2a-ple-ram` @ `stage1.2a-ple-ram-pass`)

Made `--ple-io ram` the DEFAULT PLE storage mode (the 1.2A validation made it the preferred configuration on this 128 GB machine). Narrow stage: default flip + documentation + low-RAM guard; no engine/quant/expert-cache/pool/spec/KV/dense changes, no new performance work. Doc: `Docs/v100-stage1.2b-final.md`.

- **Default is canonical in one place:** `Options::ple_io = "ram"` in `src/program/generate.cpp` (the kernel API `PleIoOptions::mode` keeps its low-RAM `Direct` default for library callers). A new ctest `ple_default_mode` pins the program default by asserting the help text shows `ram (default)` with `direct` and `mmap` still selectable. Explicit `--ple-io ram|direct|mmap` (and `--ple-ram`) always override; nothing auto-switches modes on detected RAM.
- **Mode is logged:** startup now prints `PLE on, table N rows of SHARD (PLE I/O mode: ram|direct|mmap)` so every log states the active mode unambiguously.
- **Low-RAM guard, no silent fallback:** before the preload, `PleTable::open` (Ram) requires total system RAM (MemTotal) >= PLE table + `ram_rest_bytes`, where the program supplies 42 GiB (expert arena ~40 GiB measured + 2 GiB headroom; 1.2A measured 67.5 GiB total peak RSS with the table resident). On a short machine the run fails BEFORE preloading with an actionable error naming the fallback (`use --ple-io direct on lower-RAM systems`); a failed anonymous allocation names it too. `ple_reader_test --gguf SHARD --ram --ram-rest-gb N` exercises the guard end-to-end against the real 26.82 GiB table without consuming RAM.
- **All four paths validated** (GPU0, workers 24, Stage 1.1 deterministic workload): no-flag (ram default), `--ple-io ram`, `--ple-io direct` (its 4,488-read/18.8 MB decode I/O intact), `--ple-io mmap` — all 32/32 golden and token-for-token identical. Default-mode performance matches the 1.2A RAM results (prefill 406.2 tok/s chunk, TTFT 6.15 s; decode 42.2–44.0 tok/s; peak RSS 67.55 GiB; peak VRAM unchanged; PLE NVMe reads at inference 0).
- **Environment note:** the system `/usr/bin/cmake` is 3.22.1 (rejects the project's `cmake_minimum_required(3.24)`); the working reconfigure used `~/.local/bin/cmake` (4.4.2).

## 2026 — V100 Stage 1.2A (branch `stage1.2a-ple-ram`, from `stage1.1-performance` c16ec22; Stage 1.1 and Stage 1 baselines untouched)

RAM-resident PLE investigation: is the PLE storage path unnecessarily NVMe-bound, and can this machine's 128 GB RAM eliminate it? Deliverables: `Docs/ple-ram-analysis.md`, `Docs/v100-stage1.2a-final.md`, `Docs/v100-performance.md` (appended), engine `--ple-io ram`, bench helpers, raw data under `Logs/`.

- **Finding: the PLE n-gram table (26.82 GiB, 320,001,536 rows × 90 B IQ4_NL, in shard 1) is the last model component that touches NVMe at inference.** The 39.97 GiB expert blobs were already RAM-resident (pinned arena, loaded once at startup; RAM→GPU at runtime). PLE moved 85.2 MB / 20,376 O_DIRECT reads (prefill, p50 7.9 ms, 2,520 ms blocked) + 18.8 MB (decode, p50 3.8 ms, p99 64 ms) per benchmark — synchronous on the host critical path (one serialized-read worker thread; "64 in flight" = a queue of serialized preads, not parallelism).
- **Why O_DIRECT was the design:** 64 GB-RAM policy — the table must never occupy RAM/page cache; engine keeps a bounded 95 MB row cache instead. Documented in-code; not an I/O-scheduling quirk. On 128 GB the constraint is looser — measured.
- **Implementation (opt-in, SSD default kept):** `PleIo::Ram` + `--ple-io ram` / `--ple-ram` / `--ple-ram-threads`. One-time preload of the validated table region into an anonymous mmap buffer (multi-threaded buffered preads, timed, reported at startup); `issue/collect/gather_batch` serve rows by memcpy through the same dequant and zeroing; mapping released like Direct. `PleTable::is_open()` treats the resident buffer as open state.
- **Measured (workers 24, deterministic Stage 1.1 workload, NVMe counters device-level per run):** GPU0 32 GB — prefill 7,477.6→4,869.9 ms (273.6→420.1 tok/s, −35 %), TTFT 8,036→5,472 ms, decode 35.4→44.7/42.2 tok/s (+19–26 %); GPU4 16 GB — prefill 7,683.9→4,470.9 ms (266.3→457.6 tok/s), decode 35.4→40.0 tok/s (+13 %). PLE NVMe at inference: 20,376 reads/85.2 MB → **0** (diskstats: RAM run's device reads = arena + preload only). Peak VRAM unchanged (18,852 / 16,133 MiB); GPU busy during prefill 53 % → 75–89 % (no longer I/O-starved).
- **Cost:** process peak RSS ~42 → **67.5 GiB** (1 Hz VmRSS sampling; arena 39.97 + PLE 26.82 + 0.7), ~44 GiB headroom on 125.78 GiB; preload 2.33 s @ 11.49 GiB/s from page cache / 16.9–28.5 s cold @ 1.0–1.6 GiB/s.
- **Correctness:** table-level bit-identity mmap = direct = ram over 52,752 rows (`ple_reader_test --gguf --ram`, including straddle/out-of-range); 32/32 golden on BOTH cards in BOTH modes; SSD and RAM token-for-token identical; expert-cache hit/miss, spec and pool stats identical between arms.
- **Recommendation:** keep `direct` as the default (64 GB machines); use `--ple-io ram` on this 128 GB box. Hot/cold caching deferred (per-prompt rows ~99.2 % unique — a hot set doesn't persist across prompts). Decode engine untouched (wait_flag_ge, CUDA events, verify window, pool dequant, expert cache, dense support) per stage scope.

## 2025 — V100 Stage 1.1 (branch `stage1.1-performance`, from `4b34188`; Stage 1 baseline `fa146c9`, tag `stage1-v100-moe-pass`)

Performance optimization, no engine changes. Deliverables: `Docs/v100-{cpu-analysis,gpu-analysis,vram-analysis,performance,stage1.1-final}.md`, `bench/v100/` harness, raw data under `Logs/{benchmarks,cpu,gpu}` (large captures gitignored; sqlite regenerable via `nsys export --type sqlite`).

- **Bottleneck determined by measurement (Option B):** decode is GPU-busy-bound at 85 % (nsys 2025.1.3, in-graph kernels). 95 % of GPU busy time is the spec **verify-window graphs** (1–4-token windows, 34.7–65.0 ms; exec counts match the engine T1:18/T2:19/T3:20/T4:50 distribution exactly). 37 % of GPU busy (2.5 s / 256 tokens) is `wait_flag_ge_kernel` — device spin waiting on the CPU expert pool. The pool drain (8.67 ms/round @ 22.3 GB/s effective) is bandwidth-, not core-limited (worker sweep 8→40 flat above 24; perf shows ~86 % of `ExpertPool::worker` time is the `pause` spin loop).
- **Adopted change (Option A):** `--pool-workers 24` (was 28): 39.65–39.99 vs 39.32–39.55 tok/s (3 runs each side), numeric-invariant (32/32 golden). Cause: with 28 workers + host, physical core 0 is double-occupied and 29 cores spin; 24 removes the worst of the SMT contention.
- **Knob space exhausted (all measured, all reverted except workers 24):** `--spec 2/3` (slower + numerics change), `--spec-min-p 0.7` (−5 %), `--kv int8` (−5 % + numerics change), `--vram-reserve-mib 300` (no-op — cache capped by the profile's 8,000 pairs), `--expert-cache 12000/16000` (no-op, 0 admitted under the no-eviction profile policy), PLE `mmap` (prefill 183 vs 295 tok/s), `--ple-inflight 256` (0.7 %), `--no-ple-prefetch` / `--ple-sync-submit` / `--ple-row-cache 0` (each −0.5…−1.8 %).
- **Prefill/TTFT explained:** 2,047-token prefill = 287.9 tok/s (engine chunk 295.1); 2.8 s of 6.93 s is PLE table I/O (20,376 random O_DIRECT reads, p50 7.9 ms, 64 in flight = the NVMe random-read floor) plus 11,439 expert blobs DMA'd from the pinned arena. Stage 1's 401.1 tok/s was its own prompt (different PLE rows / expert routing). PCIe: 41.6 GB HtoD @ 5.9 GB/s during decode (~45 % of PCIe3 x16; demand-driven by the 12.4 % miss rate, not saturated).
- **Expert-cache residency:** 8,000 slots / 12.93 GiB, profile-filled, 87.6 % hit rate, adaptive tier swaps 2,365 hot experts every 4 rounds. Cache *size* is not the limiter (profile cap); bigger profile = the lever.
- **Engine bug found (not fixed):** `--expert-cache-per-layer` + profile fill fails at startup — `ExpertCache::verify_slot: slot 0 differs from the arena at byte 0 (of 2329600)` after "R4.2g PER-LAYER: each layer owns 166 slots".
- **16 GB (GPU4) regression PASS:** workers 24, peak 16.13 GiB (1.6 % headroom), 6,321 slots / 10.23 GiB, 36.6 tok/s sustained (+11 % vs Stage 1's 32.8 — post-Stage-1 engine commits, not a Stage 1.1 change), no CUDA errors, deterministic output (differs from GPU0 golden at token 6 — expected: different resident set + NOT-CORRECT hit path).
- **Housekeeping:** dev repo moved to `~/dsh/strata/Strata` per user decision (`/mnt/ssd` now models-only); pristine upstream clone kept at `~/dsh/strata/upstream-pristine`; `strata-swift-iq3_xxs.json` paths + workers updated.

## 2026 — V100 Stage 1 (branch `feature/v100-moe`, base `1ee8b66`)

- `5f75af7` **baseline** — original Strata build recorded red on V100 (no sm_70 codegen; ctest registrations unusable). Doc: `Docs/v100-stage1-baseline.md`.
- `6bb6b94` **SM70 support** — `CMAKE_CUDA_ARCHITECTURES=70`, `FA_ALL_QUANTS`, tf32 fallback for the QSA scorer on cards without MMA/ldmatrix (SASS-verified: 0 ldmatrix/0 MMA). Decision: emulate, do not redesign the attention path.
- `ae7b4fb` **ctest usable on V100** — in-tree registrations fixed; 19/20 green (`ple_parity` red by design, needs Q2_0 fixtures). Doc: `Docs/v100-build.md`.
- `12979f4` **dangling-else fix** (`src/kernels/cpu/pool.cpp`) — the `else` after the `sched_getaffinity` branch bound to the inner `if (CPU_ISSET(...))`; on success the fallback core list was pushed once per UNSET mask bit → 54,263 "cores" / pool workers on a 56-CPU machine (~15-minute startup hang in `clone3`). Fixed with two braces. Root-cause repro kept in `/tmp/pool_repro*.cpp`.
- `fa146c9` **Stage 1 docs** — testing (determinism + llama.cpp token-level cross-check: 32/32 identical on the primary prompt), benchmarks (prefill ~400 tok/s; sustained decode 37–40 tok/s; peak VRAM 18.3/15.6 GiB on 32/16 GB), final report (PASS).
- **Fork + PR** — forked to `noorazman/Strata`; branch pushed; PR **Niko1221/Strata#3** opened against `main` (maintainer_can_edit). Local clone `/home/noorazman/dsh/strata/Strata` kept pristine (remote `local-mirror`).
- **PR closed at user request** — user wants the build local and the repo private *for now*; PR Niko1221/Strata#3 closed, no upstream pushes. Fork left public (GitHub forbids making a public fork private without unforking first — deferred). Branch `feature/v100-moe` remains on `origin`, so the PR can be reopened or a new one opened anytime.

### Decisions of record
- Model for Stage 1: user's Swift-1.5 IQ3_XXS GGUF (no downloads).
- `--pool-workers 28` (physical-core count) recommended on this box: +9–13 % decode vs the 55-worker default (SMT contention).
- 2 MB hugepages for the expert arena measured SLOWER on this box (12.2 vs 35–37 tok/s, reclaim pressure on the 72 GB PLE page cache) → pool left empty; the arena auto-uses hugepages when configured.
- Reference engine: vanilla llama.cpp (the `ik_llama.cpp` production fork crashes on this box at CUDA init/NCCL).
- Hit-path caveat stays documented, not patched: the GPU expert-cache path is upstream-declared not-correct and opt-in; token-level match with the reference is the acceptance bar, and it was met on this model/quant.
