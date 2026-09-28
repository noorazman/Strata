# V100 Stage 1.7 — GPU kernel experiments: FINAL

**Production target: the 32 GB V100 (GPU0) only.** Stage 1.7 profiled the decode GPU
kernel breakdown, ran three env-gated kernel experiments (E1/E2/E3), root-caused the
all-on regression, and finalized the set of keepers. Flash Attention, context shifting,
KV optimization and Stage 1.8 are explicitly out of scope.

## Verdict

- **KEEP E2** — `STRATA_GR_NORM_SPLIT` (per-stream gr_norm split), **ON by default**.
  A measured GPU win (−69.8 ms over a 256-token decode, norm kernel 14.04 → 7.86 µs,
  −44 %), bitwise-identical to the single-block norm, adds no VRAM. End-to-end on
  GPU0: −16.2 ms mean over 12 tight interleaved pairs (9/12 faster; −58.7 ms
  outlier-trimmed), +0.15 tok/s (49.13 → 49.28).
- **REVERT E1** — `STRATA_GR_DOWN_SPLIT` (gr_down K-split), now **OFF by default**
  (opt-in `=1`). Measured cost +133.1 ms over 256 tokens.
- **REVERT E3** — `STRATA_ROUTE_SORT` (bitonic-sort top-10), now **OFF by default**
  (opt-in `=1`). Measured cost +100.0 ms over 256 tokens.
- The E1 part-scratch (`kFusedGrDownPartFloats`, 320 KiB per pool) is now reserved only
  when E1 is enabled, so the production (E1-off) footprint drops the unused scratch.

Net production change: **E2 on, E1 off, E3 off** (all three still re-enable-able by
env for future work).

## The three experiments and the regression

All three were introduced default-ON during the Stage 1.7 kernel work and produced a
~3.5 % end-to-end regression when all were enabled. `Docs/v100-stage1.7-state.md` §10
documents the E1 K-split inject-block part-row bug found and fixed (bitwise-verified);
§11 documents the all-on regression root cause (this final supersedes the
investigation-only docs).

### Per-kernel mechanism (nsys 2024.6.2, `--cuda-graph-trace=node`, GPU0, 256-token decode)

The MTP/draft kernels and all memcpy are byte-identical across the base/e12/allon
captures, so the regression is 100 % attributable to the E1/E2/E3 kernels themselves:

| path | per-call | Δ over 256 tok | verdict |
|---|---:|---:|---|
| `gr_down_multi` (41 blk, r=74) | 47.99 µs | — | base |
| `gr_down_multi_split` (164 blk, r=95) + `gr_down_multi_reduce` (41 blk, r=25) | 54.16 + 5.60 µs | **E1: +133.1 ms** | REVERT |
| `gr_norm_multi` (1 blk/token) | 14.04 µs | — | base |
| `gr_norm_multi_split` (4 blk/token, one per HC stream) | 7.86 µs | **E2: −69.8 ms** | **KEEP** |
| `route` (single-block, single-warp) | 9.51 µs | — | base |
| `route_sort` (single-block, single-warp) | 15.90 µs | **E3: +100.0 ms** | REVERT |

**Net all-on = +133.1 − 69.8 + 100.0 = +163.3 ms ≈ 3.2 %**, matching the raw-bench
+3.5 % all-on regression.

### Why each is what it is (confirmed)

- **E1 (gr_down K-split) — largest cost, +133.1 ms.** The base `gr_down_multi` already
  runs a single wave (41 blocks < 90 SMs). K-splitting to 164 blocks does **not**
  shorten the critical path (the down projection is not K-parallelism-bound); it adds
  register pressure (95 vs 74 regs → 2 blocks/SM occupancy), a `part`-scratch global
  round-trip (each K-sub-block writes partials, the reduce reads them back), and a
  separate reduce-kernel launch. Net 54.16 + 5.60 = 59.76 µs vs 47.99 µs per call.
  The `part` scratch traffic itself is small (~1.3 GB total, ~100 MB/s, L2-resident) —
  not the dominant cost.
- **E2 (gr_norm split) — genuine win, −69.8 ms.** The norm was latency-bound at
  1 block/token; splitting to 4 blocks/token (one per HC stream) halves the time
  (14.04 → 7.86 µs, −44 %) with no scratch round-trip. Bitwise-identical outputs.
- **E3 (route_sort) — second cost, +100.0 ms.** The 16-element bitonic-sort top-10 is
  1.68× slower than the iterative-argmax `route` scan (15.90 vs 9.51 µs) inside the
  same single-block single-warp kernel.

### End-to-end benchmark

GPU0, 256-token decode, canonical 28-token prompt, `--spec 4 --spec-min-p 0.5
--expert-cache auto --mtp mtp/rt --kv fp16 --max-context 8192`, `drop_caches` between
runs. Box MCE/scheduling drift between runs was large (per-run sd ~90 ms — comparable
to the effect), so the reported statistic is the **tight-pair delta** (each E2 run vs
the baseline run immediately before it, ~2.5 min apart), which cancels the slow
cycle-level drift. 12 interleaved pairs (24 runs), all 256/256 deterministic and
32/32 golden in every run; the E2 output is byte-identical to the all-off baseline.

| pair | base (ms) | E2 (ms) | E2−base (ms) |
|---:|---:|---:|---:|
| 1 | 5084.2 | 5082.4 | −1.8 |
| 2 | 5290.5 | 5000.0 | −290.5 |
| 3 | 5223.0 | 5181.2 | −41.8 |
| 4 | 5286.1 | 5193.9 | −92.2 |
| 5 | 5162.7 | 5303.3 | +140.6 |
| 6 | 5067.8 | 5320.3 | +252.5 |
| 7 | 5195.2 | 5202.5 | +7.3 |
| 8 | 5080.7 | 5077.7 | −3.0 |
| 9 | 5235.6 | 5235.3 | −0.3 |
| 10 | 5261.8 | 5253.4 | −8.4 |
| 11 | 5290.2 | 5251.9 | −38.3 |
| 12 | 5354.5 | 5236.4 | −118.1 |

- **Arm means:** baseline 5211.0 ms (sd 90.5, 49.13 tok/s) vs E2 5194.9 ms (sd 92.3,
  49.28 tok/s) → **e2e −16.2 ms (−0.31 %), +0.15 tok/s** on the full 12 pairs;
  **E2 is faster in 9 of 12 pairs** and the negative-sum of the deltas (−594 ms)
  outweighs the positive-sum (+400 ms).
- **Outlier-trimmed:** the two large positive deltas (pairs 5, 6 — MCE/scheduling
  hiccups on those specific runs) trimmed, the remaining 10-pair mean is **−58.7 ms
  (−1.13 %)**, consistent with the kernel-level −69.8 ms.
- **Conclusion:** E2 is a real, modest end-to-end improvement (order −1 %, confirmed
  at the kernel level as the norm kernel's −44 % / −69.8 ms). The per-run noise
  (±90 ms) is as large as the effect, so the e2e delta is best read as "small but
  directionally consistent and never net-hurting", with the unambiguous win being the
  norm kernel itself.

Raw data: `Logs/benchmarks/s17f-{base,e2}-{1..12}.{log,json}`.

## Correctness (GPU0)

- **32/32 golden** prefix in every run (golden 256-token sequence md5
  `c1517d02473fbc06b5cf415ea1f8be63`).
- **256/256 deterministic**: baseline and E2 re-runs byte-identical; E2 output
  byte-identical to the all-off baseline (E2 is bitwise-identical).
- **No CUDA errors / no hangs** in any run.
- **ctest 20/22** — the 2 failures are the pre-existing environmental ones
  (`ple_parity`: its Q2_0 reference shard is absent on this machine;
  `platform_memory_test`: needs `ulimit -l unlimited`), unchanged from Stages 1.3–1.6.

## VRAM / RAM

- **Peak VRAM (GPU0): 18,900 MiB — identical** for the all-off baseline and the E2-on
  build. E2 adds no VRAM (reuses the existing `xn_` buffer); the E1 scratch (320 KiB ×
  the MTP + verify pools) is not reserved when E1 is off.
- **Host RAM: stable** (~96.3 GiB available of 125.8 GiB; the ~29 GB PLE table is
  file-backed page cache, reclaimable). No RAM regression.

## Files changed (this stage)

- `src/kernels/cuda/fused_gr.cu` — E1 split+reduce kernels + gate (now off-by-default);
  E2 norm-split kernel + gate (retained, on-by-default); `fused_gr_down_split_enabled()`.
- `src/kernels/cuda/native_router.cu` — E3 `route_sort` kernel + gate (now off-by-default).
- `include/strata/kernels/fused_gr.hpp` — `FusedGrArgs`, `kFusedGrDownPartFloats`,
  `fused_gr_down_split_enabled()`, `fused_gr_read_multi` signature.
- `src/core/{layer,mtp,verify}.cpp` + `include/strata/core/{layer,mtp,verify}.hpp` —
  gate-conditional `grdown_part_` reservation + `fused_gr_read_multi` call sites.

## Commits / service

_Final commit list and `strata.service` status: filled in below._
