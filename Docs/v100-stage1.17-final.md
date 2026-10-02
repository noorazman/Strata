# V100 Stage 1.17 — MoE Prefill Grouped Skinny-GU GEMM (`STRATA_MOE_GEMM_GROUPED=1`) — FINAL

Production target: 32 GB V100 (GPU0) only. Started from clean Stage 1.16 HEAD `21ff71d`.
Baseline config for A/B: production (wide dequant ON + TC OFF + LT OFF + E5/E8 OFF + 8K expert cache + int8 KV).
Leg A = fresh production re-runs (this stage); leg B = `STRATA_MOE_DQ_WIDE=1 STRATA_MOE_GEMM_GROUPED=1`.
The grouped path is OFF-by-default; no production-default change.

**Verdict: WASH (negative) — mission stop rule. The grouped launch consolidation is real and on the critical path
(−188,447 kernel launches per 16 K run; the ~133 K skinny-GU dispatches + their 88 K splitKreduce collapses into
384 grouped G2 launches; G2 GPU time 2.68 s replaces the ~2.43 s of cutlass-64x64 + wmma-32x32 + their reduces,
total kernel busy ≈ neutral), but the dequant-first reordering it requires breaks the baseline's
dequant/GEMM interleave under the 8-slot expert-DMA staging pipeline: compute-stream idle goes 7.69 s → 14.67 s
(p99 inter-kernel gap 97 → 191 µs), so the end-to-end r1 TTFT is consistently SLOWER — 16 K +3.5 s (+8.8 %,
5/5 fresh-engine leg-B determinism, vs fresh leg-A re-run at the exact 1.16 anchor md5) and 32 K +6.8 s (+8.7 %).
Both contexts are far outside the noise band (16 K ±2.7 % / 32 K ±1.4 %). Per the stop rule the stage stops
after this ONE attempt: the path stays OFF-by-default, production is restored + verified live, and the stage is
documented without forcing a win.**

## 1. Mission and stop rule

"Find ONE meaningful optimization for the remaining MoE prefill skinny-GU GEMM bottleneck. Pick ONE of the two
candidate directions (2-D tiled per-expert TC kernel, or grouped/batched skinny GEMM) based on fresh
measurements. PROFILE FIRST. Do not assume. A microbench win alone is NOT sufficient; the end-to-end gain must
be clearly repeatable (target roughly ≥2–3 %). If it is not: STOP and document WASH. If grouped: specifically
measure whether reducing the ~133 K dispatches actually moves the critical path rather than merely reducing
launch count. Stop after ONE optimization attempt."

## 2. Profile first (fresh 1.16-production baseline, `Logs/gpu/s117nsys-16k-off.sqlite`)

16 K serve-mode nsys of the production baseline: **SPAN 37.56 s, GPU busy 30.12 s (80.2 %), GPU idle 7.44 s,
1,395,768 kernels; GEMM-family 11.87 s / 388,819**. The main compute stream (13) gaps BEFORE GEMM kernels sum to
**1.61 s over 388,819 GEMM kernels (avg 4.14 µs)** — the measured critical-path dispatch cost; the skinny-GU
share is ≈0.65 s (main 0.42 s + splitKreduce 0.23 s). Gaps before non-GEMM kernels = 6.08 s (CPU chunk logic +
DMA-event waits, p99 112.5 µs). The 1.16 TC (S0) calibration (Stage 1.16 doc): per-expert TC saved 2.03 s of
GU-small GEMM GPU time (5.46 → 3.43 s, −88.5 K launches) but e2e was WASH because (a) non-GEMM regressed +1.59 s
and (b) part of the removed GPU time was not on the critical path (84 % busy). That calibration pushed this
stage toward the direction with a NEW lever: launch consolidation measured on the critical path (grouped), not
a second per-expert TC variant (a repeat of 1.16's measured −0.08 s span).

## 3. Direction pick (from fresh measurements)

- Microbench (`bench/v100/s117_gemm_variants.cu`, L2-hot W, in-run cuBLAS baseline, median of 60): the per-expert
  2-D tile family S0/S1/S12/S14 only wins at ne ≤ 16 and loses at ne 33–64 (S12 = 75.7 µs vs cuBLAS 43.0 µs at
  ne 64); ncu: S12 is L1-queue-bound (54 % of warp stalls = "L1 instruction queue full", L1/TEX 96 % vs CUTLASS 46 %).
  The per-expert direction is thus a re-measure of 1.16's WASH class.
- Grouped (G = 347 bench experts, one launch): G1 (1-warp tiles) loses to G×cuBLAS 2.9×; G2 (S12-tile body,
  128 threads, 4-way in-block k-split, binary-search expert lookup) = 7.33 ms for the 261-expert skinny batch vs
  5.67 ms for G×cuBLAS under the microbench's L2-warm inter-launch-overlap baseline (that baseline is
  optimistic for the engine, where per-expert cuBLAS has no inter-call overlap; in-engine estimate: G2 ≈
  3.0–3.3 s vs cuBLAS GU-small 5.46 s ≈ −2.2 s GPU), PLUS the ~0.65 s dispatch-gap reduction.
- **Picked: GROUPED (G2).** ONE direction, not both.

## 4. The isolated change (OFF by default)

- `src/prefill/gemm.cu`: `g2_gu_kernel` (the bench G2 verbatim: grid (80, total n16 tiles), 128 threads,
  m16×n16 tiles, per-block expert via binary search over `E[mid].tile0/80`, QCACHE=4, 4-way in-block k-split,
  fixed-order smem reduce → deterministic) + `Gemm::gu_grouped()`. 112 regs, 4.1 KB smem, 0 spills.
- `src/prefill/prefill.cpp`: `STRATA_MOE_GEMM_GROUPED=1` switches the MoE expert loop to three phases per
  chunk-layer: (1) dequant ALL routed experts into PER-EXPERT pools (512 × 9.83 MB = 4.7 GB, plain cudaMalloc —
  NOT the borrow region, see §7); (2) one grouped G2 launch over all 1 ≤ ne ≤ tc_max_ne experts (fat experts
  keep per-expert cuBLAS); (3) per-expert swiglu + D GEMM unchanged. `STRATA_MOE_GEMM_TC_MAXNE` is now honored
  regardless of `STRATA_MOE_GEMM_TC` (it sets the skinny range). `STRATA_MOE_GEMM_TC_VERIFY=1` extends to
  grouped mode (co-compute cuBLAS for the first 8 skinny experts per launch, log `[g2_verify]` nneq/maxAbs).
  When OFF: byte-identical allocation and code path to production (verified: leg-A re-runs reproduce the 1.16
  anchor md5s exactly, §6).

## 5. Correctness and determinism

- Microbench maxAbs vs cuBLAS: all S/G variants 1e-5…1.6e-4 (S3 still broken, kept for the record, dropped).
- In-engine verify hook (29-token run, GROUPED + VERIFY): `[g2_verify]` **maxAbs ≤ 3.1e-06** vs cuBLAS on all
  46 sampled batches (≈1–2 ULP f32, same class as the 1.16 TC family — leg-B outputs legitimately differ from
  leg-A by ~1 ULP and flip a few expert routings, as in 1.16); first token 271 = golden.
- Fresh-engine determinism (leg B, 16 K fill prompt, 5 fresh engines): r1 text md5 `c9358fb2f362…` **5/5
  identical**; leg-B md5 ≠ leg-A md5 (expected, 1.16 semantics). r1 r1-TTFT spread ±0.6 s (±1.4 %).

## 6. A/B (r1 = cold, the anchor)

| | leg A (fresh production) | leg B (grouped) | Δ |
|---|---|---|---|
| 16 K r1 client TTFT | 40,662.8 ms (md5 `6041c5f3…` = 1.16 anchor ✓) | 43,773…45,039 ms (5 runs, md5 `c9358fb2…` 5/5) | **+3.5…+4.4 s (+8.8…+10.8 %)** |
| 16 K r1 engine prefill_end | 35,229 ms | 38,534…39,637 ms (det2–5 mean 38,747) | **+3.52 s (+10.0 %)** |
| 32 K r1 client TTFT | 76,748.8 ms (md5 `1fbe577e…` = 1.16 anchor ✓) | 83,440.4 ms | **+6,691.6 ms (+8.7 %)** |
| 32 K r1 engine prefill_end | 71,130 ms | 77,898 ms | **+6,768 ms (+9.5 %)** |

Both contexts consistently slower, far outside the noise band (16 K ±2.7 % = ±1.09 s; 32 K ±1.4 %). The warm-r2
serve flake (r2 18→34-chunk re-chunking, both legs) is pre-existing and gated out, as in 1.16.

## 7. Why (nsys kernel-level A/B, `s117nsys-16k-off` vs `s117nsys-16k-g2`)

| | baseline | leg B (grouped) | Δ |
|---|---|---|---|
| kernel launches (all streams) | 1,395,768 | 1,207,321 | **−188,447** (mission question: the ~133 K dispatches DO reduce; they are the 1.61 s critical-path gap) |
| GU-skinny GEMM kernels | cutlass-64x64 39,962 / 1.00 s + wmma-32x32 67,781 / 1.15 s | `g2_gu_kernel` 384 / **2.68 s** (avg 7.0 ms/launch, matches microbench) | GPU ≈ −0.0 s, launches −107 K |
| splitKreduce | 140,299 / 1.02 s | 51,556 / 0.35 s | −88.7 K launches, −0.67 s |
| total kernel busy (stream 13) | 27,147 ms | 26,405 ms | **−0.74 s** (≈ neutral) |
| stream-13 span | 34,834 ms | 41,075 ms | +6.24 s |
| stream-13 idle | 7,687 ms (22 %) | **14,670 ms (36 %)** | **+6.98 s** |
| inter-kernel gaps (stream 13) | p50 0.9 µs, p99 97 µs | p50 0.9 µs, p99 **191 µs** | the idle is mid-range CPU-side waits |
| H2D (expert staging) | 110,647 / 195,520 MB / 20.1 s | 111,043 / 195,530 MB / 20.7 s | **identical** (after the pool fix, §7a) |

So: the GEMM/dispatch side did exactly what the profile predicted (launches collapsed, GPU busy neutral — the
dispatch gap itself is only 1.61 s, of which the skinny-GU share is 0.65 s), but the compute stream now idles
+7.0 s. Root cause: the grouped path dequants ALL experts of a chunk-layer before ANY GEMM, so the
compute-to-DMA wait cadence between consecutive staged-expert `cudaStreamWaitEvent`s shrinks from ~100 µs
(dequant + 3 GEMMs, baseline) to ~20 µs (dequants only, grouped) — with one expert blob (1.7 MB) taking ~180 µs
of the 9.4 GB/s DMA engine, the 8-slot/lookahead-7 staging pipeline falls behind and the compute stream stalls
at the p99 191 µs gap. The 1.16 per-expert TC path kept the baseline interleave and paid no such cost.

7a. Implementation correction found by the first nsys: the 4.7 GB per-expert pools were initially allocated from
the prefill **borrow region**, which is bump-allocated from the trailing slots of the 12.93 GiB / 8000-slot
expert cache (`generate.cpp`: lend the slots whose trailing bytes ≥ `bytes_needed()`). That carved ~2,900 slots
out of the cache, pushed ~19 K more experts through the H2D staging path (nsys: H2D 195.5 → 227.4 GB,
+18.7 K copies) and added +9.6 s of compute-stream idle. Fix: pools are plain cudaMalloc (peak VRAM 23.6 GB on
the 32 GB V100, fits with margin; the borrow then sizes exactly as in production and H2D returns to baseline —
§7 H2D row). The correction recovered 3.1 s of the 7.8 s loss but not the reordering cost above.

## 8. ncu on the chosen kernel (G2, 261-expert skinny batch)

`g2_gu_kernel` (80, 448)×128, 112 regs, 4.1 KB smem: Memory Throughput 98.8 % (L1 pipe bound), L1 hit 58.7 %,
L2 hit 91.3 %, DRAM 12.3 % (W streaming), compute (SM) 24 %, warp stalls 63.1 % = L1TEX scoreboard
(memory latency) — the same L1-queue profile class as S12/1.16-TC (CUTLASS avoids it via smem staging;
a smem-staged G2b is the documented next bottleneck, §9).

## 9. State and next bottleneck

- `STRATA_MOE_GEMM_GROUPED` stays OFF-by-default (production unchanged); the code, bench, and nsys artifacts are
  committed for the next stage. Production `strata.service` restored: /health 200, live request verified.
- Next-bottleneck candidates for a future stage, in measurement order:
  1. **The 100–200 µs CPU-side compute-stream waits** (6.08 s in the baseline alone, p99 112.5 µs; worse here):
     deeper DMA staging (32 in-flight slots ≈ 54 MB, cheap) or re-interleaving dequant/GEMM per expert while
     still grouping (per-expert dequant → per-expert G2 launch defeats the purpose; the fix is staging depth).
  2. **smem-staged G2b** (CUTLASS-style LDG→smem staging to escape the L1-queue-bound 54/63 % stall; cuBLAS L1
     46 % vs S12 96 %) — only if the dispatch/idle problem is fixed first, since G2's GPU time is already ≈
     neutral vs the baseline GEMMs.
  3. **Grouped D GEMM** (same skinny problem at N=640/K=2560; the second ~0.65 s dispatch pool).
- The 1.16 TC kernel (`STRATA_MOE_GEMM_TC`) remains the per-expert option if a future stage re-opens the
  per-expert direction with a smem-staged tile.
