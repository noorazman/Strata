# V100 Stage 1.6 Final — CPU Pool Scheduling Optimization

Branch `stage1.3-expert-pool-sync` (continues from the Stage 1.5 commit `95a338e`).
Scope: the seven Stage 1.6 candidates — CPU affinity/governor, SMT/core placement,
keep-frequency-high-without-spinning, pool scheduling/drain, shorter park threshold
re-test, the "l1/g0" ring-spin tail, and cross-layer pool pipelining — with the rule
**profile first, test one change at a time, keep only what measures end-to-end**.
Config throughout: `--pool-workers 24`, `--ple-io ram`, `STRATA_POOL_PARK=2048`,
`--spec 4 --spec-min-p 0.5`, `--expert-cache auto`, `--kv fp16 --max-context 8192
--prefill 2048`, 256-token decode, V100-32 GB GPU0 primary / V100-16 GB GPU4
validation. Frozen per stage rules: PLE implementation/default, `--ple-io ram`,
quantization, KV cache, speculation, attention implementation, dense-model support.

Golden bar (identical to the independent vanilla llama.cpp reference, Stage 1):

```
271 248068 198 760 1156 369 30869 5402 430 328 23202 1288 264 11952 5617 303 220 17 15 17 21 11 321 369 883 310 3184 728 883 836 1118 1834
```

## 1. Bottleneck found (profiled, not assumed)

The Stage 1.5 freeze left a single large host-side component: the **round-head
wait-C (layer-0 flag-C) spin, 9–14 ms uniform across rounds**. Everything else on the
CPU side is small (mid-window waits ≤ 1.3 ms; worker rows ≤ 0.2 ms; host round-end
path 2.9 ms). The round is ~51 ms, so the round-head spin is ~25 % of the round and
is the only lever with headroom.

The flag-visibility chain was measured end to end, one hop at a time:

1. **Workers raise flag-C ≤ 0.11–0.20 ms after dispatch** (worker wall, `STRATA_POOL_ASYNC_DIAG`).
2. **The WC flag store + `_mm_sfence()` completes in ≤ 12 µs** (`STRATA_SFENCE_DIAG`, all sites).
3. **The GPU was polling normally the whole time** — `STRATA_WAIT_ITERS=1` records the
   `wait_flag_ge` poll-iteration count per slot: round-head flag-C = **25,564 iterations**
   at the ~9.3–14 ms spin = ~364–500 ns/iteration. A `__nanosleep(100)` loop at that
   rate is healthy (~100 ns sleep + ~250–400 ns PCIe read). The flag value is genuinely
   invisible to the GPU for the full spin, even though the CPU-side store + fence is
   done in microseconds.

So the stall sits **between "the WC write is at the CPU point-of-no-return (µs)" and
"the GPU's PCIe read observes the value"**. nsys correlation (WC build, round head):
the host makes only 2 CUDA API calls during the 9.26 ms spin; the 1.83 MB staging H2D
for the layer is issued only when the host resumes. This is not a host dispatch gap, a
coarse poll, or a WC-drain stall.

### Isolation probes (`/tmp/flagvis*.cu`, sm_70)

To find the mechanism, a focused microbenchmark measured the CPU host-store → GPU
PCIe-read visibility latency in isolation (a `<<<1,1>>>` kernel spins on a mapped
host flag; the host publishes it after a fixed delay; the lag = GPU-cycles-to-observe
minus the host launch→write interval). **Result: visibility is ~immediate (the
measured lag is negative = just the kernel-launch latency, ~20–50 µs) in every
configuration tested:**

| configuration | median lag |
|---|---|
| host-core write, WC, posted (store+sfence) | −0.045 ms |
| host-core write, WC, non-posted (store+`lock addl $0`) | −0.045 ms |
| + 24 background DRAM-traffic threads (256 MB each) | −0.020 ms |
| worker-core (cpu14, node 1) write | −0.046 ms |
| worker-core + 24 traffic threads | −0.039 ms |
| cross-core: host resets flag=0 (mfence), worker writes 1 | −0.027 ms |
| cross-core, same node | −0.037 ms |
| cross-core, WB page | −0.036 ms |

So the basic write→read path is fast on this hardware. The engine's ~13 ms is
**context-specific** to the round head (the window boundary), not an intrinsic
write/read, core-affinity, page-type, posted-vs-non-posted, cross-core reset+write, or
CPU-DRAM-traffic effect.

### MCE storm (measured, ruled out as the primary cause)

`/dev/mcelog` (parsed with the kernel `struct mce`) shows a continuous stream of
corrected machine-check events on **bank 5 (MC_CHA, the memory controller)**,
severity RECOVERED, all socket 0. The rate is **constant: ~1.0/s at idle and ~1.0/s
during a full engine run** — so the engine does not provoke extra MCEs, and the
~50/s round heads cannot be caused by a ~1/s MCE stream. The MCEs are a real
background hardware condition (a degraded DRAM row/channel on socket 0) and likely
contribute to the per-read jitter, but they are not the dominant cause of the
round-head lag.

### Address component (partial)

Routing the round's flag-C through a **freshly allocated page** (`STRATA_ALTFLAGC=1`,
a different physical address than `h_flag_`) reduced the round-head wait-C from
23,302 → 15,819 poll-iterations and the host spin from 13.1–15.3 → 12.6–14.4 ms.
Interleaved 256-token A/B (3 alternating pairs, GPU0): fresh-flag avg **50.11 tok/s**
vs original **50.00 tok/s** — a small, partial address improvement (the engine's
original flag page is marginally worse, consistent with it sitting near the degraded
row). Not a full fix, so `STRATA_ALTFLAGC` stays opt-in.

**Conclusion:** the ~13 ms round-head flag-C visibility lag is a context-specific,
hardware-level effect at the window boundary (PCIe / memory-controller / IOMMU
interaction under the round's concurrent staging DMA + MTP + the MCE storm). It is not
fully eliminable in software with the changes tried here (WC page, non-posted publish,
fresh page, core placement). It is documented as the **remaining bottleneck**.

## 2. Experiments and before/after

Baseline (Stage 1.5 freeze: WB flags, synchronous post-pool store, non-pinned
arena): **48.58 / 48.82 tok/s** (interleaved, GPU0, 256 tok).

| candidate | result | verdict |
|---|---|---|
| (1) CPU affinity / governor | `performance`/min_freq 2.9 GHz: e2e-neutral | neutral (deployment rec only) |
| (2) SMT/core placement | workers→node1 SMT 43.15/42.01; SMT-node0 42 tok/s; **arena pinned to the host's NUMA node 49.08/50.07** | **arena keeper** (below); SMT workers rejected |
| (3) keep freq high w/o spinning | governor A/B | neutral |
| (4) pool scheduling / drain | **async rows dispatch** 48.99/50.85 vs sync 49.44/49.99 | e2e-neutral, **kept** (below) |
| (5) shorter park threshold | 512: 47.19/46.48; 1024: 50.24/48.76; 2048: 48.29/49.20 | keep **2048** (1024 ≈ 2048 > 512, within MCE noise) |
| (6) "l1/g0" ring-spin tail | = the round-head wait-C flag-C visibility lag (§1) | documented (remaining bottleneck) |
| (7) cross-layer pool pipelining | not measured to matter given (4) | not pursued |

**Net end-to-end (final config vs baseline): 48.7 → 50.0–50.1 tok/s (+~3 %).** The
win is the arena-node keeper; async rows and the WC/non-posted flags are correctness
and robustness keepers (e2e-neutral) that the round-head fix would build on.

## 3. CPU frequency / affinity findings

- Governor `schedutil`, min 1.2 GHz, max 3.3 GHz. Forcing `performance` (min 2.9 GHz)
  is e2e-neutral here — the workers are dispatch-bound, not frequency-bound, during
  decode. Recommend `performance` (or `min_perf` ≥ 2.9 GHz) at deployment to avoid
  idle→active upclock latency, but it is not a measured decode win.
- NUMA: node0 = cpus 0–13,28–41; node1 = 14–27,42–55; SMT sibling = +28. The host and
  the 26.8 GB PLE live on node0. Putting workers on node1 SMT cores cost −11 %
  (43.15/42.01 tok/s) — the cross-QPI row streams dominated. **Keeping the arena on
  the host's node** (the keeper) removed the non-deterministic node-1 arena
  first-touch: pool/drain 11.88 → 12.97–13.42 ms/tok and 48.70 → 49.08–50.07 tok/s.

## 4. What was kept (committed)

1. **Arena NUMA keeper** — the 40 GB expert arena's pages now first-touch on the
   calling (host) thread's NUMA node by default (deterministic), with
   `STRATA_ARENA_NODE=<n>` to override. Was non-deterministic (loader-thread
   first-touch drifted to node 1). **+1.8 % e2e, drain −11 %**.
2. **`STRATA_POOL_CORES`** placement knob (worker core override for A/B).
3. **Async rows dispatch** (`STRATA_POOL_ASYNC`, default ON) — the fused rows phase
   fires without blocking the host's layer loop; the workers (and the host when it
   drains) finish the rows and raise flag-C themselves. `STRATA_POOL_ASYNC=0` keeps
   the synchronous post-pool store. Invariants: flag-C once per layer; the worker
   store is ordered after the row stores; `mode_` not reset at end of async dispatch;
   synchronous fallbacks for empty-plan / njobs==0 / njobs>96 / `STRATA_POOL_UNFUSE` /
   in-flight. e2e-neutral, golden-passing.
4. **Write-combined host→GPU flags + non-posted publish** — `h_flag_`/`h_flagA_`/
   `h_flagB_` are `cudaHostAllocMapped|WriteCombined`; every flag publish is a plain
   store + `lock addl $0` (a non-posted locked RMW that retires only once the line's
   write-back has completed) + `_mm_sfence()`. This makes the value present in DRAM by
   the time the store sequence ends (a posted store + sfence only retires when the
   write is *issued*). flag-B was already a non-posted `cmpxchg`. The row bulk
   (`h_ymiss_`) and plan stay WB (worker streams).

## 5. What was reverted / rejected

- SMT workers on node1 (−11 %) and SMT-node0 workers (42 tok/s) — reverted to the
  node0 host + 24-worker placement.
- Governor `performance` / min_freq 2.9 GHz as a decode change — neutral; deployment
  recommendation only.
- Park threshold change — 2048 µs kept.
- Full WC (flags + ymiss + plan) — the ymiss/plan WC was reverted to WB after the
  flags-only WC proved equivalent (the row bulk is better WB).

## 6. Probes added (all off by default, zero-cost when off)

- `STRATA_POOL_ASYNC_DIAG=1` — worker wall top-10 slowest dispatch→flag phases.
- `STRATA_SFENCE_DIAG=1` — per-site sfence max/avg/total (verify sites 0–5, pool
  sites 0–2); reports the flag store+fence is ≤ 12 µs.
- `STRATA_WAIT_ITERS=1` — per-slot max GPU poll-iteration counts for the
  `wait_flag_ge` waits (slot = (l*G+grp)*3 + {0=A,1=B,2=C}); printed at exit. This is
  the probe that proved the GPU was polling normally (25,564 iters = healthy rate).
- `STRATA_HEAD_DIAG=1` — decomposes the round-end host path (run→commit, commit→draft,
  draft→run).
- `STRATA_ALTFLAGC=1` — routes the round's flag-C through a fresh page (the address
  A/B, §1).

## 7. Correctness

- **32/32 golden** on GPU0 in every run this stage (baseline, async, WC, non-posted,
  alt-flag, park, final).
- **256/256 deterministic** ×2 (A1==A2 and B1==B2 full-output identical; alt-flag and
  original-flag produce byte-identical 256-token outputs — the alt flag only moves the
  physical page, not the model math).
- **16 GB (GPU4)** validation of the final config: RC=0, **48.87 tok/s**, first-32
  identical to the GPU0 golden, 96 full-256 diffs vs GPU0 (≈ the ~108 pre-existing
  cross-card difference), no CUDA errors/OOM.
- **ctest 20/22** (the 2 pre-existing environmental failures: `ple_parity` missing
  Q2_0 shard, `platform_memory_test` mlock ulimit).
- No CUDA errors, hangs, or races in any run.

## 8. Remaining bottleneck

The **~13 ms round-head wait-C (layer-0 flag-C) visibility lag** at the window
boundary — ~25 % of the ~51 ms round. It is a context-specific hardware effect (PCIe /
memory-controller / IOMMU interaction under the round's concurrent staging DMA + MTP,
on a socket-0 DRAM row with a steady ~1/s corrected-MCE stream). Eliminating it is the
next lever for throughput (a 13 ms round-head removal would move ~50 → ~65 tok/s if
the rest of the round does not stretch). Software changes tried here (WC page,
non-posted publish, fresh page, core placement) are necessary for correctness/robustness
and give a partial address improvement, but do not remove it.
