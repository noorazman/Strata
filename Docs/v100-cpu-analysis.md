# V100 CPU Analysis — Strata MoE (Stage 1.1)

Scope: CPU-side bottleneck determination for the Strata expert pool on this
machine. All data from `bench/v100/bench.py` runs (raw CSVs under
`Logs/cpu/`, raw perf data `Logs/cpu/perf-decode-w28.data`).

## Machine facts

- 2× Xeon E5-2680 v4 (Broadwell): 28 physical / 56 logical CPUs, AVX2, no AVX-512.
- NUMA: node0 = CPUs 0–13, 28–41; node1 = CPUs 14–27, 42–55.
- 125 GB RAM; no hugetlb pool (deliberately empty — hugepages measured slower in Stage 1).
- GPU0 PCI `02:00.0`, `numa_node` file empty (-1); socket locality inferred from bus.
- Expert arena: ~40 GiB pinned host RAM (`cudaHostRegister PORTABLE`, 4 KB pages).

## Thread placement during decode (workers 28)

From the 1 Hz thread-placement log (`Logs/cpu/phase0-baseline-256tok.threads.csv`):

- 32–33 live threads, one pinned per core on **node0 only**: host thread on
  core 0 (mean 94 %), pool workers on cores 1–28 (mean ~68 % each), PLE I/O +
  CUDA helper threads on cores 32/34/41.
- Core 0 carries **two** threads (host + one worker) — SMT sibling sharing.
- Total CPU during the 8 s decode window ≈ 53 % of 56 logical CPUs.

Per-core means (decode window): 1 core ≥90 %, 28 cores 20–90 %, rest <5 %.
**No pool core is saturated** — the pool is not CPU-throughput-bound; the
~32 % headroom per worker is time spent waiting (ring arrival, job-steal
empty-queue, drain barrier).

## Worker sweep (256-token sustained decode, GPU0, all else default)

| workers | tok/s | pool ms/tok | drain ms | CPU mean % |
|--------:|------:|------------:|---------:|-----------:|
| 8  | 36.00 | 23.92 | 21.79 | 10.1 |
| 12 | 37.05 | 19.86 | 17.64 | 12.0 |
| 16 | 39.33 | 14.46 | 12.36 | 13.5 |
| 20 | 39.47 | 12.57 | 10.36 | 15.4 |
| 24 | **39.99 / 39.79 / 39.65** | 11.33 | 9.14 | 15.3–17.2 |
| 28 | 39.55 / 39.38 / 39.32 / 39.39 | 11.44–11.50 | 8.67–8.70 | 19.2 |
| 32 | 39.63 | 11.47 | 8.64 | 21.4 |
| 40 | 38.91 | 11.93 | 8.79 | 23.6 |

Output tokens identical across all worker counts (deterministic). Readings:

1. **Diminishing returns above 16 workers.** Drain saturates at ~8.6–9.1 ms/round
   from w24 up — the remaining drain work does not parallelize further
   (memory-bandwidth-limited, see below).
2. **More workers ≠ faster, they just spin more.** CPU % climbs 10 → 24 % from
   w8 to w40 while decode stays flat.
3. **w24 is the best config** (39.65–39.99 over 3 runs vs 39.32–39.55 for w28,
   ~+1 %): with 28 workers + host, core 0 is double-occupied (SMT contention)
   and 29 cores spin; 24 workers avoid the worst of that.

## What the CPU actually does (perf, 4 s of decode, 11,589 samples)

Top self-time:

| % | symbol |
|------:|--------|
| 85.3 | `strata::kernels::cpu::ExpertPool::worker` |
| 2.9 | `ggml_vec_dot_iq3_s_q8_K` (native gate/up rows, CPU side) |
| 2.0 | `strata::core::Verifier::run` (spec-verify host path) |
| 1.6 | `ggml_vec_dot_iq2_xs_q8_K` (native rows) |
| 1.3 | `q2_0_gguf_rows_multi_avx2` (pool drain) |
| 1.2 | `ggml_vec_dot_iq4_nl_q8_0` (native down rows) |

Disassembly of `ExpertPool::worker` (perf annotate): the function is a
**spin-poll loop on the round counter** —

- 35.4 % `pause`, 22.5 % loop `jmp`, 16.9 % round-counter `mov`, 3.6 % `cmp`,
  7.3 % stop-flag `test`, 0.3 % flag load, 0.4 % `lock` decrement, and the
  actual `drain()` call.

So **~86 % of pool-worker time is polling**, ~14 % is useful drain work.
Engine's own phase counters agree: drain 8.67 ms/round vs ~10.6 ms round span
for the pool phase, at an effective **22.3 GB/s** across the dequant+rows
phases ("pool multi gate/up 6.078, quantize 0.180, down 2.402 ms/round;
22.3 GB/s over the rows phases" from `--stats`).

## Verdict

- The CPU pool is **not the decode bottleneck by itself** — 8 workers already
  give 36 tok/s, and drain is bandwidth-limited (22.3 GB/s effective) rather
  than core-limited from w24 up.
- The pool **is on the critical path**: the GPU `post` graph cannot combine
  layer `l`'s experts until the pool finishes; the GPU waits for that inside
  `wait_flag_ge_kernel` (see `v100-gpu-analysis.md` — 2.5 s of GPU busy time).
- Spin-wait accounting: ~19 CPU-seconds per wall-second at w28 are poll cycles.
  This is what makes the per-core utilization (68 %) look "busy" while no core
  is saturated.
- NUMA is not an issue: all worker/host/PLE threads and the arena allocate on
  node0, same socket as GPU0; node1 stays idle.
- **Recommendation: `--pool-workers 24`** (measurable +1 %, see
  `v100-performance.md`). Engine-level wins (dequant bandwidth, spin→event
  handoff) are the next lever, out of scope for Stage 1.1's low-risk phase.
