# V100 Stage 1.9 — `wait_flag_ge` A/B/C dependency analysis (profile-only, no engine changes)

Production target: **32 GB V100 PCIE (GPU0)** only. Goal of the stage: understand the
`wait_flag_ge` bottleneck **before** changing any synchronization code. Investigate the full
A/B/C flag dependency chain (CPU pool dispatch, flag A/B/C writes, GPU `wait_flag_ge`,
dependent GPU kernels, layer/round boundaries) and answer: (1) why 3 waits instead of 1;
(2) which waits are necessary for correctness; (3) how much time each wait contributes;
(4) whether waits can be merged/overlapped/moved/eliminated; (5) whether this is related to
the Stage 1.6 flag-C round-head visibility issue. **No synchronization behavior changed, no
optimization implemented, FlashAttention/QSA untouched.** Binary `build-sm70/strata` (HEAD
`90afb6c`, Stage 1.8 docs-only) unchanged this stage; all diagnostics used are opt-in
env-gated (`STRATA_WAIT_ITERS=1`, `STRATA_SFENCE_DIAG=1`, `STRATA_HEAD_DIAG=1`,
`STRATA_POOL_ASYNC_DIAG=1`) and were verified to leave the token stream bit-identical.

## 1. Method

| Tool | Config | Purpose |
|---|---|---|
| `bench/v100/s19.sh` | sanity arms (32-tok golden, 256-tok determinism ×2, hostdiag 256-tok with all four host diagnostics) + two `nsys -t cuda --cuda-graph-trace=node` captures (256-tok fp16/8192 and 256-tok int8/32768), canonical 29-token prompt, `--prefill 2048 --spec 4 --spec-min-p 0.5 --pool-workers 24 --mtp mtp/rt`, GPU0 with `strata.service` stopped | A/B/C attribution on the current binary + host-side producer timeline |
| `bench/v100/s19_trace.py` | sqlite → wait-kernel classifier: anchors each wait-A on the `copy_i32_from_mapped_kernel` that follows it (exactly 1/3 of waits, verified), walks the fixed A→B→C cycle inside the 48-dispatch round, round boundaries via `sampler_greedy_kernel` (one per round, 111 = 111); per-flag stats, >1 ms cohort, round-head vs mid-round, doorbell→waitA host gap, waitB↔staging-DMA tie, per-round wait totals | kernel-level A/B/C attribution, comparable across binaries |
| historical data | `Logs/gpu/s17x-base.sqlite` (Stage 1.6 binary, same workload, node trace), `Logs/gpu/s18nsys-base.sqlite` (current binary, 64 tok — the F1 source) | old-binary delta + F1 cross-check |

The classifier reproduces the known s17x totals (15,984 waits = 48 dispatches × 111 rounds ×
3 flags, exactly; 111 rounds = 111 sampler launches) and the s18 64-tok F1 figure (503.5 ms
= 7,868 µs/tok, 34.3 % of the main stream) to the digit. Staging DMA streams were identified
by stream: verify staging = stream 17 (1,800 copies / 3.11 GB in s19-base), adaptive-tier
refill = stream 19 (exactly 2,503 copies = the engine's "2503 experts swapped" line),
prefill expert streaming = stream 15/7 (before the first round).

## 2. The protocol (source, `src/core/verify.cpp`, `expert_source.cpp`, `kernels/cpu/pool.cpp`)

Per (layer l, group grp) dispatch inside a captured verify window (G=1, 48 dispatches/round):

```
GPU stream cs_ (window graph)                     Host driver thread
─────────────────────────────                     ─────────────────────────────────────────
pre(l): embed, PLE, gr_read, attention,
  moe_route(ids)
  └─ doorbell_publish_kernel
       └─ h_seq_ = ring ─── dev→host mapped (posted write)
                                     spins on *h_seq_ (2 ms cudaStreamQuery flush)
post(l):
  └─ wait_flag_ge A ────────────────┐   plan(ids) → h_plan_ (WB pinned)
                                     │   sfence + `lock addl` → h_flagA_ (non-posted, in DRAM on return)
  └─ copy_i32_from_mapped (plan)    │
  └─ VRAM grouped kernels ──────────┤   fetch_dma: H2D staging blobs on copy_ stream
       (∥ DMA + pool below)         │     └─ cudaLaunchHostFunc → h_flagB_ when DMA lands
  └─ wait_flag_ge B ────────────────┤     (raised immediately on host when 0 staged blobs)
  └─ PCIe grouped kernels ──────────┤   act-quant → jobs → pool dispatch (async, 24 workers)
       (reads staging_ VRAM)        │     workers GEMV → h_ymiss_ (WC)
  └─ wait_flag_ge C ────────────────┘   last worker: sfence + `lock addl` → h_flagC_ (non-posted)
  └─ copy_from_mapped (CPU rows)
  └─ add_hits + moe_combine
round tail: head + sampler ─── host sync, commit graph (0.86–1.0 ms), MTP draft (1.9–2.0 ms,
next window: staging writes, PLE gather, flag reset, cudaGraphLaunch → back to pre(0)
```

All three flags are mapped pinned host memory; the GPU spins in
`wait_flag_ge_kernel` (`verify_kernels.cu:418`: 1×1 thread, `while (*flag < value)
{ __nanosleep(100); ++n; }` then `__threadfence_system()`). Flag publishes are non-posted
(lock RMW, Stage 1.6): the value is in DRAM when the store retires.

**History (git):** `git log -S wait_flag_ge` touches only the first commit `f2a08d4` (plus
`2546fcb`). The first commit's `verify.cpp` already issues exactly these three waits per
dispatch (lines 517/545/553: flagA, flagB, flagC) — the 3-flag protocol is the original
design of the expert-pool sync, not a regression.

## 3. Measured wait breakdown (current binary, 256 tok, 111–112 rounds)

`s19-base` (fp16/8192, node trace, window 7,988 ms; e2e of the same workload without nsys:
**50.77 tok/s**, 2.31 tok/round, 45.4 ms/round wall):

| flag | n | total | µs/tok | share | mean | p50 | p90 | p99 | max | round-head (l0, per round) | mid-round (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A (plan) | 5,328 | **1,103.7 ms** | 4,311 | 66.0 % | 207 µs | 5.0 | 18.8 | 10,832 | 14,631 µs | **1,058.4 ms = 9.54 ms/round** (max 14.6) | 45.3 ms = 8.7 µs |
| B (staging DMA) | 5,328 | 196.7 ms | 769 | 11.8 % | 36.9 µs | 4.7 | 68.7 | 457 | 7,791 µs | 55.7 ms = 0.50 ms/round | 141.1 ms = 27.0 µs |
| C (CPU rows) | 5,328 | 371.9 ms | 1,453 | 22.2 % | 69.8 µs | 4.3 | 209.6 | 612 | 19,847 µs | 33.1 ms = 0.30 ms/round | 338.8 ms = 64.9 µs |
| **total** | 15,984 | **1,672.3 ms** | **6,532** | 29.9 % of main-stream kernel time | | | | | | | 15.07 ms/round |

- **Waits > 1 ms: 150 instances = 1,157.6 ms = 69.2 % of ALL wait time.** 125/150 sit at the
  round head (1,113.8 ms); **111 of those are flag A — one per round, 1,058.4 ms (87 % of the
  >1 ms cohort)**. The rest: 29 B (69.7 ms), 10 C (29.5 ms, incl. the round-0 cold-start
  outlier 19.8 ms).
- Host side of A: doorbell-end → flag-A published = mean **75 µs**, p90 94, p99 100, **max
  376 µs, zero > 1 ms** (n=5,328). The host is never the slow party — the round-head 9.5 ms
  is GPU-side visibility of a store that retired ~80 µs after the doorbell.
- waitB: 10 % of B-waits have the staging H2D completing inside the wait (staging stream
  only: 1,800 copies / 3.11 GB, 0.34 distinct experts/layer — identical to the s17x run).
- int8/32768 (production service config, 112 rounds, 39.3 tok/s under nsys): total **1,754.8
  ms = 6,855 µs/tok (30.0 %)**; A 1,110.8 ms (63.3 %, head 9.20 ms/round), B 389.0 ms
  (22.2 %), C 255.0 ms (14.5 %); >1 ms cohort 166 = 1,168.0 ms, 133 at round head.

Old binary, same workload, same analyzer (`s17x`, Stage 1.6 code, 111 rounds):

| flag | total | µs/tok | share | head/round | mid mean |
|---|---|---|---|---|---|
| A | 996.6 ms | 3,893 | 60.6 % | 8.21 ms | 16.3 µs |
| B | 406.9 ms | 1,589 | 24.8 % | 1.02 ms | 56.2 µs |
| C | 240.0 ms | 937 | 14.6 % | 0.27 ms | 40.3 µs |
| total | 1,643.4 ms | 6,420 | 28.7 % | | 308.5 µs/dispatch |

**Per-dispatch wait is unchanged: 308.5 µs (old) → 313.3 µs (new), +1.6 %.** The A/B/C mix
moved (A +107 ms, B −210 ms, C +132 ms) — the B/C difference tracks DMA queue state and
source-page temperature between runs (staging volume is identical to the digit: 1,810 vs
1,800 copies, 3.12 vs 3.11 GB), not a structural change.

## 4. Answers to the stage questions

### Q1 — Why are there 3 waits instead of 1?

The 3-wait structure is the **original design** (first commit `f2a08d4`) and is confirmed in
**both** binaries: old and new issue exactly 15,984 = 48 × 111 × 3 waits per 256-tok run.
Each wait gates a distinct producer writing distinct data through a distinct channel, and
the structure is the *maximum-overlap* arrangement:

- **A** gates the **plan** (`h_plan_`, host-computed from this layer's `moe_route` ids after
  the doorbell). Nothing can run in `post(l)` before the plan: the grouped kernels' counts,
  starts and destinations all come from it.
- **B** gates the **staging DMA** (copy engine, `copy_` stream, `cudaLaunchHostFunc` raises
  B on completion; raised immediately on the host when the dispatch staged 0 blobs). The
  VRAM grouped kernels run **in parallel** with the DMA and the pool — B is what lets that
  overlap happen without the PCIe group reading a half-filled `staging_`.
- **C** gates the **CPU pool rows** (`h_ymiss_`, 24 workers; async since Stage 1.6). The
  PCIe grouped kernels run **in parallel** with the pool tail — C is what lets that overlap
  happen.

With a single flag raised after (plan + DMA + pool), `post(l)` would serialize behind the
slowest of the three (the pool, ~0.5–2 ms/dispatch) instead of overlapping all of them.
The 3-wait structure costs 2 extra ~5 µs spin launches and buys the VRAM-group ∥ (DMA +
pool) and PCIe-group ∥ (pool tail) overlaps — that is precisely the design the Stage 1.3/1.6
work measured into the 48–50 tok/s territory.

**Stage 1.8 F1's "1 → 3" claim does not reproduce.** The old-binary capture of the same
workload (`s17x`) also shows exactly 3 waits per dispatch; the "1" matches only the Stage
1.1 doc's description of a single doorbell spin. F1's comparison was almost certainly the
ring/doorbell count (5,328 = 48 × 111) against the wait count (15,984 = 3 × 5,328). The
per-dispatch wait number F1 cared about (309 → 420 µs) was an artifact of the 64-tok
capture's cold, contaminated window (see §5), not a binary change.

### Q2 — Which waits are necessary for correctness?

- **A: always necessary.** Without it the GPU could read a stale plan (previous dispatch /
  previous window) → wrong group boundaries → wrong results. The plan is too large for the
  graph-launch parameter path; the flag is the only ordering channel. The publish's `sfence`
  (plan drain, site 0: max 1 µs) makes the flag a *valid* gate for the WB plan data.
- **B: necessary when the dispatch staged ≥ 1 blob** (≈ 39 of 48 dispatches/round, 0.81
  blobs/dispatch). When 0 blobs staged, B is raised immediately on the host and the wait is
  a ~3–5 µs spin+kernel tax (≈ 21 dispatches/round ≈ 0.1 ms/round ≈ 0.3 % e2e) — removable
  only with a conditional graph node, a small win for a graph-structure change.
- **C: necessary whenever the CPU share is non-empty** — at the measured 82.8 % expert-cache
  hit rate (126,006/152,169, 26,163 refused, cache 100 % full) with 3.63 distinct CPU
  experts/layer, essentially every dispatch has CPU rows. A zero-CPU-share dispatch could in
  principle skip C, but that case is rare.

All three are load-bearing; none is redundant. The question is not "3 vs 1" but "how much
does each *spin* cost", and that is dominated by the round-head A visibility (§5, Q5).

### Q3 — Time each wait contributes (measured)

From §3: **A 66.0 % / B 11.8 % / C 22.2 %** of the 6,532 µs/tok total wait (fp16);
A 63.3 % / B 22.2 % / C 14.5 % (int8). The decomposition that matters:

- **Round-head (dispatch 0) = 1,147.2 ms of 1,672.3 ms (68.6 %)** — and of that, flag A is
  1,058.4 ms (92.3 %).
- **Mid-round = 525 ms**: C 338.8 (pool tail) + B 141.1 (DMA queue) + A 45.3 (near floor).
- Mid-round per-dispatch = A 8.7 + B 27 + C 65 ≈ 101 µs — this part is already close to the
  producer-latency floor (host plan ≤ 376 µs published early, DMA lead ≈ waitA+grouped,
  pool wall publish→flagC ≤ 0.21 ms/phase).

### Q4 — Can the waits be merged / overlapped / moved / eliminated?

Let X = `copy_from_mapped + add_hits + moe_combine` ≈ 60–100 µs.

- **Merge B+C (one flag after both DMA and pool):** current round-tail = `max(B+X, C)`
  (the PCIe group overlaps the pool tail); merged = `max(B, C) + X`. The merge wins only
  when B+X > C (DMA slower than the pool); in the common case C ≥ B it is X *worse*.
  Measured mid-round C (65 µs) ≥ B (27 µs) in most dispatches → **keep 3 waits; do not merge
  B+C.** (The merge also loses the zero-blob fast path: B is free today when 0 blobs staged.)
- **waitB on a side stream** (graph edge to the PCIe group): the PCIe group still cannot
  start before the DMA, so wall time is unchanged; only the 1-thread spin moves off the main
  SM. Negligible.
- **Move A earlier / publish A earlier:** the plan needs this layer's `moe_route` output —
  the last significant kernel of `pre(l)`, immediately before the doorbell. The host already
  responds in ≤ 376 µs (mean 75). Mid-round A (8.7 µs) is at the floor. **No headroom except
  at the round head (§5).**
- **Eliminate:** the zero-blob waitB tax (~0.1 ms/round) and, hypothetically, zero-CPU-share
  waitC — small, need conditional graph nodes. The only large removable component is the
  round-head A spin (9.54 ms/round) — see Q5/theory.
- **Overlaps already in place and load-bearing:** VRAM-grouped ∥ (DMA + pool),
  PCIe-grouped ∥ (pool tail), MTP draft ∥ (commit + next-window setup). The 3 waits *are*
  the overlap machinery; removing one removes an overlap.

### Q5 — Relation to the Stage 1.6 flag-C round-head visibility issue: **same mechanism, different slot.**

Stage 1.6 measured a 9–14 ms round-head visibility lag in the **l0 flag-C slot** (25,564
poll iterations; CPU store+fence completed in µs but the GPU's PCIe read returned the stale
value for the whole spin; isolated CPU→GPU flag writes showed no lag → context-specific
LLC/PCIe interaction at the window boundary, with the box's steady corrected-MCE stream).
The current data show the **same shape** — one ~9–14.6 ms spin per round at dispatch 0 —
landing in the **l0 flag-A slot**:

| evidence | value |
|---|---|
| kernel-side (nsys, current binary) | A round-head = 111 waits, 9.54 ms/round mean, 14.6 ms max = 1,058.4 ms |
| hostdiag wait-iter slots (max poll iters) | slot 0 (l0, flag **A**) = 3,888 iters = the max slot; B/C slots ≤ 3,865 (l22 B, an isolated B outlier) / 2,559 |
| host side of A | doorbell→A-publish mean 75 µs, max 376 µs, 0 waits > 1 ms → the store retires ~80 µs after the doorbell |
| sfence publish timing | A publish max 22 µs, plan-drain max 1 µs, B raise max 16 µs, pool flag-C max 20 µs → CPU side is µs-fast |
| host round-head ring wait | l0 = 0.000 ms/round; the slow host-side spin is the **l1/g0** shadow (7.9–8.2 ms = the GPU's whole `post(0)`+`pre(1)` duration) |
| host round-end path (`STRATA_HEAD_DIAG`) | run→commit 0.89 + commit→draft(MTP) 1.98 (max 8.77 @ round 1) + draft→run 0.03 = 2.90 ms/round — all before the next window's `pre(0)` |

Why the slot differs from Stage 1.6: the lag belongs to **the wait that straddles the window
boundary**, i.e. the first flag read after the previous round's sync + commit + MTP draft +
staging traffic. The A store lands at doorbell+80 µs while waitA starts at doorbell+~15 µs,
so waitA spans the boundary and absorbs the lag; in the 1.6-era captures the boundary
straddled the C slot instead. The mechanism (stale mapped-flag line on the GPU's read path
at the boundary, µs-fast CPU store) is the documented Stage 1.6 one — the non-posted
publish removed the CPU-side component and the lag remains as a GPU-side visibility effect.

The **round-0 cold-start C outlier** (19.8 ms s19-base / 20.2 ms s18-64tok / 29.0 ms
s19-int8 / 20.6 ms s17x) is a separate, smaller effect: the first verify round's
expert-arena/PLE pages cold after load (the host's own log shows the l0 pool dispatch
taking 18–21 ms at t+42 ms in both binaries).

## 5. F1 cross-check (the "309 → 420 µs per dispatch" mystery)

`s18nsys-base` (current binary, 64 tok, the capture F1 used): total wait 503.5 ms =
7,868 µs/tok = 34.3 % (reproduced exactly), 419.6 µs/dispatch vs s17x's 308.5. That window
is contaminated: the 4.7 s "decode window" includes the prefill tail plus three cold
stalls (539.9/449.6/368.3 ms, the 1.8 doc's known PLE/expert-cache fetches after
`drop_caches`) and only 25 rounds, of which the first carries a 20.2 ms C outlier.
Decomposed: cold C +16.9 µs/dispatch, mid-B elevation +51 µs, mid-C elevation +32 µs,
mid-A +5 µs, head-A comparable (8.33 vs 8.21 ms/round). The s19-base capture (256 tok,
current binary, identical workload to s17x) settles it: **313.3 µs/dispatch — no binary
regression.** F1's "increase" was the short cold window, not the Stage 1.7 code.

## 6. Confirmed root cause

1. **Structural (benign):** the 3-flag A/B/C protocol is the original design and is
   necessary; the per-dispatch spin is ~310 µs (A 8.7 + B 27 + C 65 µs mid-round plus the
   kernel launches) — no regression between the Stage 1.6 and current binaries.
2. **Dominant cost (63 % of all wait time):** the **round-head flag-A visibility lag** —
   1,058.4 ms over 111 rounds (9.54 ms/round, max 14.6 ms). The host publishes A ≤ 376 µs
   after the doorbell; the GPU's read of the mapped line returns the stale value for ~9–14
   ms at every window boundary. This is the Stage 1.6 boundary-visibility mechanism, now
   landing on the A slot (it was the C slot in the 1.6-era captures).
3. **Secondary (37 %):** mid-round C = the pool tail (338.8 ms, 20 % — the pool's
   publish→flagC pipeline vs the grouped kernels finishing first) and B = the copy-engine
   queue depth at boundaries + DMA source-page state (141.1 ms, 8 %, fp16).

## 7. Theoretical optimization opportunity

The round-head A spin is on the critical path (main stream blocked; the host is already
ready — the data is in DRAM 80 µs before the spin starts):

- **Round-head A → mid-round level (8.7 µs):** total wait 1,672 → 614 ms; removes
  **2,398 µs/tok** of wait; round wall 45.4 → ~35.9 ms → **e2e 50.8 → ~64 tok/s (+26 %)**.
- **Round-head A+B+C → mid-round levels:** ~10.3 ms/round → round wall ~35.1 ms →
  **~66 tok/s (+30 %)** ceiling. (Head B/C are partly boundary-effect, partly run-state.)
- For comparison: the entire QSA attention mechanism (Stage 1.8) is 1,317 µs/tok — the
  round-head A lag alone (4,141 µs/tok) is 3× the whole attention mechanism.

The mid-round component (~101 µs/dispatch) is near the producer floor; attacking it
(picker pool throughput, more staging) has diminishing returns vs the boundary.

## 8. Recommended next experiment (Stage 1.10, NOT implemented this stage)

1. **`STRATA_WAIT_FENCE=1`** (env-gated, default off): add `__threadfence_system()` every
   ~1024 poll iterations inside `wait_flag_ge_kernel` (today it fences only after the loop
   exits). A system-scope fence during polling is the direct test of the "stale mapped line
   cached on the GPU read path" hypothesis. Arm: nsys 256-tok fp16 (same as s19-base) + e2e
   A/B with golden/determinism gates. Expect: if the hypothesis is right, round-head A
   collapses from 9.5 ms toward the 80 µs store latency (up to ~+20–25 % e2e); if wrong,
   the distribution is unchanged and the cost is ~1024 extra fences per big wait (ns).
2. If (1) is inert: **pre-warm the flag line at the window head** — a tiny dummy node
   reading `h_flagA_` at the start of `pre(0)` (before the first wait), to trigger an
   invalidation before the spin starts.
3. Only if both fail: **reorder the boundary traffic** (MTP draft before commit, or a
   small inter-round gap) — larger change, same target.

Non-goals for the fix stage: merging B+C (Q4), removing zero-blob waitB (0.3 % e2e),
raising pcie-frac (already measured optimum at 0.2 in Stage 1.3).

## 9. Correctness / sanity evidence

- **32-tok golden:** 32/32 prefix MATCH (md5 `cf577e731fbff35d91b879a23b670e56` = the known
  32-tok golden) — `s19-g32`, default env = production behavior.
- **256-tok determinism ×2:** both runs byte-identical, md5
  `c1517d02473fbc06b5cf415ea1f8be63` = the Stage 1.7/1.8 fp16 256-tok golden; 50.77 /
  50.76 tok/s; 111 rounds; window sizes T1:29 T2:12 T3:12 T4:58, accepted 0:42 1:22 2:18
  3:29 — production shape.
- **hostdiag 256-tok** (all four diagnostics on): 50.76 tok/s; per-round `wait for rings
  26.04 / pool 12.57 / host 0.04 / commit 0.86 ms`; CPU experts 3.63 distinct / 4.91 routed
  per layer; expert-cache hits 82.8 % (26,163 refused, 100 % full) — all within the frozen
  1.6/1.7 baselines.
- **nsys arms** (s19-base, s19-int8): rc=0, no CUDA errors/hangs; `STRATA_WAIT_ITERS=1`
  was the only env set during the captures (diagnostic only, verified bit-identical output
  in the hostdiag arm).
- `strata.service` was stopped for the campaign and **left stopped** (per stage instruction);
  working tree clean after commit.

## 10. Data & artifacts

- Scripts: `bench/v100/s19.sh` (campaign), `bench/v100/s19_trace.py` (analyzer,
  `--json` output).
- New raw: `Logs/gpu/s19-{base,int8}.{nsys-rep,sqlite,run.log}`, `Logs/gpu/s19-{base,int8}.json`,
  `Logs/gpu/s19-campaign{,2}.log`, `Logs/benchmarks/s19-{g32,det-1,det-2,hostdiag}.{json,log}`.
- Reused historical: `Logs/gpu/s17x-base.*` (old binary, same workload),
  `Logs/gpu/s18nsys-base.sqlite` (F1 source).
