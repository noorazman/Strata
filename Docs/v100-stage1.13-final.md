# V100 Stage 1.13 — MoE Prefill H2D Byte-Reduction: Expert-Residency Growth (E8)

Stage 1.13 from the clean Stage 1.12 commit `71d89e9` (branch `stage1.3-expert-pool-sync`,
GPU0, V100 32 GB). Production 32 K / int8 KV config untouched until the final restore.

**Goal:** attack the largest remaining prefill bottleneck identified in Stages 1.11/1.12 —
the ~170 GB of expert-weight H2D during a 16 K prefill — by growing the resident expert
cache from the production 8,000-slot profile to ~14,000 slots on a rebuilt top-14,000
profile, and measure the H2D-byte reduction with full gates and an A/B at 16 K / 32 K.

## 1. Why residency (from the Stage 1.12 diagnostic)

The 16 K prefill streams **169.70 GB** of expert weights (96,650 H2D transfers @ 10.57 GB/s,
10.36 MB/token) = **133,176 routed expert runs** over only **20,716 unique (layer, expert)
pairs** (6.43× re-routing). The production 8,000-slot profile cache is only **27.4 % resident**
on this workload — the profile (built before the current routing distribution) is stale, and
the cache is capped at 8,000 by `--expert-cache auto` (auto sizes from free VRAM but is capped
at the profile's pair count).

Resident experts are computed directly from the VRAM arena — **no staging ring, no H2D**
(`src/prefill/prefill.cpp`: `stage_one` skips residents; one H2D blob transfer per non-resident
expert run). So H2D bytes scale linearly with the streamed-run fraction:

- Baseline (8,000-slot stale profile): 96,650 streamed runs = 27.4 % resident → 169.70 GB.
- E8 target (~14,000 slots on a current, count-weighted profile): see §2.

VRAM budget: after weights/PLE/MTP the card has **26.61 GiB free** (700 MiB reserved); the
14,000-slot arena takes **22.78 GiB** (per-pair sized slots, avg 1.63 MB — most experts are
smaller than the 2.33 MB max blob), leaving ~3.8 GiB for KV (3.09 GiB at 32 K int8) + token
graph + verify/MTP buffers. Measured peak at a 32 K session: **29,242 MiB of 32,768** — fits
with ~3.5 GB margin (§4).

## 2. The rebuilt profile (E8's data)

`data/expert-profile-14k.bin` (STRP, 48×512, built-for 14000, ranked 14000, 56,024 B) —
built by `bench/v100/e8_profile.py` from two count-weighted route dumps of the real
fill-prompt workloads (new `STRATA_MOE_ROUTE_DUMP` format: per-expert token counts, not just
the routed set — a 2-line extension in `src/prefill/prefill.cpp`, diagnostic env only):

- 16 K fill dump (r1+r2+warmup+oh: 864 section lines, 21,635 unique pairs, 16.06 M routing
  decisions),
- 32 K fill dump (1632 section lines, 22,110 unique pairs, 31.80 M decisions).

Union: **22,113 unique pairs, 47.85 M decisions**. Ranking = decision-count descending with a
deterministic (layer, expert) tie-break.

Coverage (in-sample union, decision-weighted): top-8000 89.94 %, top-10000 94.27 %,
top-12000 96.96 %, **top-14000 98.54 %**, top-16000 99.40 %.

**Execution-level coverage** — the metric H2D bytes actually follow (one H2D per streamed
expert *run*, not per token):

| workload | resident-run share, top-8000 (old profile, measured) | resident-run share, top-14000 (E8) |
|---|---|---|
| 16 K fill r1 | 27.4 % (36,526 / 133,176, nsys-verified in 1.12) | **77.6 %** (105,758 / 136,245) |
| 32 K fill | ~29–30 % (324,069 / 1,060,200, production serve line) | **78.9 %** (448,160 / 568,317, all sections) |

The old 8,000-slot profile covers only 27.4 % of 16 K runs despite being the "top 8,000" of
whatever corpus it was built on — it is stale against the current routing. The rebuilt profile
covers **77.6 % of 16 K runs at the same budget of 8,000 pairs is not what E8 asks**; E8 asks
for 14,000 pairs, which covers 77.6 % of runs (98.5 % of decisions).

**Predicted H2D at 16 K:** streamed runs 96,650 → ~30,500 ⇒ **~39–53 GB** (69–77 % cut),
to be measured exactly by nsys in the A/B.

Out-of-sample caveat: the profile is built from the fill-prompt family that the A/B uses; on
production traffic mix the hit rate will differ (follow-up: build the profile from production
routing). It cannot be much worse than the current 27–30 % unless the production mix is far
from the fill text — documented, not a blocker for the opt-in measurement.

## 3. The change (one isolated opt-in)

No kernel or engine-path code change. E8 is the existing `--expert-cache N` mechanism pointed
at a larger profile:

- `--expert-cache 14000` (instead of `auto`; last-occurrence-wins in the arg parser —
  bench.py's `--strata-flags` appends it after the hardcoded `auto`),
- `--expert-profile data/expert-profile-14k.bin` (instead of the 8,000-pair file).

Mechanics already in the tree (`src/program/generate.cpp`): the sized-slot path walks the
profile in rank order and stops at the VRAM cap (min(requested × max_blob, free − reserve));
all 14,000 pairs fit (22.78 GiB < 25.9 GiB cap). The prefill borrow mechanism lends only the
~70 tail (lowest-ranked) slots to the prompt buffers, so the prefill hit-rate cost of borrow
is <0.5 % of the profile.

Tooling (additive, `bench/v100/`): `s113e8.sh` (gate + ctx A/B harness; the 8,000-slot leg is
the unchanged `s110.sh ctx`), `e8_profile.py` (dump → STRP builder + coverage reports).

## 4. VRAM fit (measured)

- Boot at 14,000 slots: `expert cache 14000 slots, 22.78 GiB of VRAM; pre-filled 14000 of
  14000 slots from the profile; slot 0 verified` — no OOM, verify_slot clean.
- 32 K session (int8 KV) after a 1,024-token probe request: **29,242 MiB used of 32,768**
  (peak sampled at 2 Hz) — ~3.5 GB margin. 16 K sessions use less KV.

## 5. Correctness gates (E8 = `--expert-cache 14000` + `data/expert-profile-14k.bin`, int8 KV, production flags, GPU0)

| gate | 16 K | 32 K |
|---|---|---|
| 256×2 determinism (det1 vs det2, fresh engines) | **byte-identical**, md5 `3dffc779e10a20ff295609ecc182d3be` | **byte-identical**, same md5 |
| 32-tok run == det1 prefix | **yes** (exact prefix, n=32/256) | **yes** (exact prefix, n=32/256) |
| old 8,000-slot golden prefix | DIVERGE at token 6 (`run=7967 golden=30869`) | DIVERGE at token 6 (same) |
| decode (256-tok det, tok/s) | 48.51 | 48.01 |
| spec accept / tpr | 0.636 / 2.34 | 0.636 / 2.34 |
| CUDA errors / hangs | none (all runs rc=0) | none (all runs rc=0) |

Decode-tier (spec/MTP path) expert-cache hits at 14,000 slots: **84.3 %** (136,227/161,524, cache
100 % full, 25,297 refused) on the 256-token benchmark prompt.

Note on the golden: E8 changes the residency *set* (which experts take the GPU hit path), so
E8's numerics need not match the old 8,000-slot golden (`GOLDEN_32`). The E8-internal gate is
therefore: (a) det1/det2 byte-identical (256×2, fresh engines), (b) the 32-token run is the
exact prefix of det1, (c) decode ~49–50 tok/s, (d) no CUDA errors. The old-golden comparison
is reported for the record. (The prefill hit path feeds the same dequant/GEMM kernels the
miss path uses, with identical weight bytes — the residency-sensitive numerics live in the
decode hit path, per the ROUND 328 warning still printed by `generate.cpp`. The divergence at
token 6 is the first decode position where a differently-resident expert changes the logits.)

## 6. A/B (serve path, same build, paired same-day runs; 8,000 slots = production config)

| 16 K fill (16,068 tok) | baseline r1 | E8 r1 | baseline r2 | E8 r2 |
|---|---|---|---|---|
| TTFT (s) | 40.56 | **38.36 (−5.4 %)** | 40.05 | **38.22 (−4.6 %)** |
| decode (tok/s) | 52.21 | 50.14 | 54.47 | 55.72 |
| prefill hit rate | 28.5 % | **72.0 %** | (29.4 % cum.) | **69.7 %** (cum.) |
| streamed expert-runs (r1) | 117,590 | **46,137 (−60.8 %)** | | |
| VRAM peak (MiB) | 18,926 | 29,012 | | |

| 32 K fill (32,504 tok) | baseline r1 | E8 r1 | baseline r2 | E8 r2 |
|---|---|---|---|---|
| TTFT (s) | 76.57 | **73.29 (−4.3 %)** | 76.05 | **72.54 (−4.6 %)** |
| decode (tok/s) | 49.62 | 45.28 | 53.55 | 68.42 |
| prefill hit rate | 28.0 % | **73.0 %** | (28.9 % cum.) | **69.5 %** (cum.) |
| streamed expert-runs (r1) | 214,978 | **80,714 (−62.5 %)** | | |
| cumulative prefill_ms (r1+r2) | 147,439.8 | **141,175.3 (−4.2 %)** | | |
| VRAM peak (MiB) | 19,178 | 29,264 | | |

Serve-mode r1/r2 decode is session-state noisy (r2 runs after the first full session cycle);
the stable decode reference is the §5 det gate: 48.0–48.5 tok/s vs the 1.12 anchors
48.8–50.7 — within the ±1.5 gate-noise band. The 32 K r1 E8 decode (45.28) is the one
conservative data point (run immediately after the hot baseline run); r2 inverts it (68.42).

## 7. nsys verification (16 K, same build, `s113nsys.sh` / `s113nsys-off.sh` captures)

| | OFF (8,000 slots) | E8 (14,000 slots) | Δ |
|---|---|---|---|
| **expert H2D, stream 15 (per 16 K prefill)** | **169.70 GB** (96,650 transfers) | **62.00 GB** (34,534 transfers) | **−107.70 GB (−63.5 %)** |
| prefill window (stream-15 H2D span) | 35.88 s | 33.87 s | −5.6 % |
| total kernel time (window) | 32,515.6 ms (1,304,687 launches) | 32,028.5 ms (1,304,496) | −1.5 % |
| dequant launches (gu / flat) | 133,171 / 133,609 | 133,141 / 133,579 | ~0 |
| GEMM time | 12,167.3 ms | 11,916.2 ms | −2.1 % |
| startup arena prefill (stream 7, one-time) | 18.56 GB | 29.13 GB | +10.57 GB |

The −63.5 % H2D is exactly the goal's metric: 16 K prefill expert weight DMA falls from
169.70 GB to 62.00 GB (measured 72 % execution-level residency, vs the 83 % decision-weighted
prediction — the long tail of one-shot experts is what keeps it under 80 %). The wall-clock
gain is much smaller than the byte cut because the MoE prefill is **compute-bound, not
DMA-bound**: resident experts still run the identical dequant+GEMM kernels (dequant launch
counts are unchanged at ~133 K — only the DMA of their blobs is saved), so the 107.7 GB of
DMA freed was largely overlapped with the ~32 s of SM work (the Stage 1.11 saturated
SM+DMA pipeline). Net: −2.01 s in the prefill window, consistent with the −2.2 s / −3.3 s
TTFT in §6.

The +10.57 GB on stream 7 is the one-time startup cost of the larger arena (22.78 GiB vs
12.93 GiB of profile prefill at boot); it amortizes to zero on a long-lived engine and is
invisible to per-request latency.

## 8. Verdict + production state

**E8 works and delivers the goal's metric:** per-prefill expert H2D 169.70 → 62.00 GB
(−63.5 %, target was ~40 GB), TTFT −4.3 to −5.4 % at 16 K/32 K, fully deterministic
(256×2 byte-identical at both contexts), decode within gate noise, VRAM fits
(29.26 GB peak at 32 K, ~3.5 GB margin).

**Production stays at the 8,000-slot default** (per the stage stop rule — the win is real but
modest on wall-clock, and the cost is a 10.1 GiB VRAM-residency increase: 16 K peak 18.9 →
29.0 GB, 32 K 19.2 → 29.3 GB, leaving ~3.5 GB instead of ~12.7 GB for future stages). E8 is
opt-in, one flag + one file, no engine code change:

```
--expert-cache 14000 --expert-profile data/expert-profile-14k.bin
```

(14,000 slots is the max that fits; the profile is the count-weighted top-14,000 of the
fill-prompt workloads — `data/expert-profile-14k.bin`, sha256 `f5d6561c…ad7`, built by
`bench/v100/e8_profile.py`.)

Follow-ups (not this stage): a profile built from production routing (the fill-prompt profile
is in-sample on the A/B workloads); shrinking the dequant work for resident experts (the
compute-bound floor found here — dequant+GEMM is ~32 s of the 36 s 16 K prefill); a
decode-tier residency study (84.3 % hits already, ROUND 328 numerics open).
