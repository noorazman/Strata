# V100 Stage 1.1 Final — Performance Optimization

Date: 2025-09-26. Branch `stage1.1-performance` (from 4b34188; Stage 1
baseline fa146c9, tag `stage1-v100-moe-pass`, untouched).

## Bottom line

**Option A (measurable, explained improvement) + Option B (profiling
demonstrates the architectural limit):**

1. **`--pool-workers 24` is the better production setting: 39.65–39.99
   tok/s vs 39.32–39.55 at 28** (3 independent runs each side). Cause:
   with 28 workers + the host thread, physical core 0 is double-occupied
   (SMT contention) and 29 cores spend ~86 % of their time
   spin-polling the round counter (perf: 85 % of all samples in
   `ExpertPool::worker`, of which ~86 % is the `pause`/counter-read spin
   loop). 24 workers remove the worst of both while the pool drain
   (22.3 GB/s effective, bandwidth-limited) is already saturated at 24+.
   Decode improves ~+1.0 %; the change is numeric-invariant (32/32 golden).
2. **Everything else is a measured limit, not an unexplored knob:**
   - Decode is **GPU-busy-bound at 85 %** (nsys, 6,772 ms of 7,912 ms).
     95 % of that busy time is the **spec verify-window graphs**
     (1–4-token windows, 34.7–65.0 ms each; exec counts match the engine's
     T1:18/T2:19/T3:20/T4:50 distribution exactly).
   - **37 % of GPU busy time (2.5 s per 256-token decode) is
     `wait_flag_ge_kernel`** — the device spinning on the doorbell flag for
     the CPU expert pool. The pool drain (8.67 ms/round, 22.3 GB/s) is the
     CPU-side critical path; it is bandwidth-, not core-limited (worker
     sweep 8→40: flat above 24).
   - The remaining GPU time: GDN recurrent multi-token kernels (~15 %),
     native dense BF16 projections (~11 %), zero-copy host reads (~4 %),
     routing/dequant/KV/LM-head (~33 %). SM70 has no `ldmatrix`/Ampere MMA;
     the fused GEMMs are 16×16/32×32 WMMA.
   - Prefill/TTFT is **PLE SSD-latency-bound**: 2.8 s of the 6.93 s
     2,047-token prefill is 20,376 random O_DIRECT reads at p50 7.9 ms with
     64 in flight (the NVMe's random-read floor; `--ple-inflight 256` buys
     0.7 %, `--ple-io mmap` is 2.5× worse). This explains the 287.9 tok/s
     prefill vs Stage 1's 401.1 (their prompt; the PLE access pattern and
     expert routing differ with prompt content).
   - The expert cache (8,000 slots, profile-filled, 87.6 % hits) is capped
     by the **profile's 8,000 pairs**, not VRAM: larger caches (12k/16k)
     give identical hit patterns, and `--vram-reserve-mib` cannot grow it.
3. **16 GB card validated:** same build + workers 24 fits with 6,321
   slots (10.23 GiB cache), peak 16.13 GiB (1.6 % headroom), no CUDA
   errors, 36.6 tok/s sustained — +11 % vs Stage 1's 32.8 sustained,
   attributable to post-Stage-1 engine commits (incl. the pool
   `physical_cores` fix), not to any Stage 1.1 config change.

## Recommended production config (changed from Stage 1)

```
--pool-workers 24        # was 28
```

Everything else unchanged: `--pack packs/swift-iq3_xxs --native <shard1>
--ple-gguf <shard1> --expert-profile data/expert-profile.bin
--expert-cache auto --prefill 2048 --spec 4 --spec-min-p 0.5 --mtp mtp/rt
--max-context 8192 --kv fp16 --ple-io direct` (PLE defaults: prefetch on,
I/O-thread submit, row cache on, 64 in flight).

## What would move the numbers next (engine work, out of Stage 1.1 scope)

1. **Verify-window cost** — the single biggest GPU term. Options: skip
   re-running shared-prefix work between rounds, or accept at the MTP-head
   level for low-risk windows. A 4-token window at 65 ms vs the 0.84 ms
   main token graph is the structural cost of current spec design on SM70.
2. **`wait_flag_ge` → CUDA events / graph dependencies** — stop burning 2.5 s
   of GPU time spinning; lets the next layer's pre-graph overlap the pool.
3. **Pool drain bandwidth** — 22.3 GB/s effective at 28 workers; the
   dequant/multi-row AVX2 kernels are the target (memory-streaming pattern).
4. **Bigger expert profile** — `tools/make_profile.py` is referenced in
   source but missing from the tree; regenerating with >8,000 pairs (VRAM
   budget allows ~16,000 on the 32 GB card) is the clean way to push the
   87.6 % hit rate up.
5. **Prefill PLE** — the 16 rows/position are read per-position; a
   prefill-phase readahead (positions are sequential) could cut the 2.8 s
   floor.

## Known issues found (reported, not fixed — no engine changes in Stage 1.1)

- `--expert-cache-per-layer` + profile fill: startup failure
  `ExpertCache::verify_slot: slot 0 differs from the arena at byte 0
  (of 2329600)` after "R4.2g PER-LAYER: each layer owns 166 slots".
- GPU expert-cache hit path remains upstream-declared NOT CORRECT (opt-in,
  warning printed); Stage 1.1 keeps it on because tokens match the
  independent llama.cpp reference 32/32 on both cards' own resident sets.

## Deliverables

| doc | content |
|---|---|
| `Docs/v100-cpu-analysis.md` | Phase 1: per-core/per-thread, worker sweep, perf spin attribution, NUMA |
| `Docs/v100-gpu-analysis.md` | Phases 0/2: nsys kernel+graph breakdown, GPU busy %, prefill kernels, SM70 limits |
| `Docs/v100-vram-analysis.md` | Phase 4: full VRAM ledger, KV probe, 16 GB fit |
| `Docs/v100-performance.md` | Phases 0/3/6/7: baseline, prefill gap, benchmark matrix, all knob results, PCIe/IO, residency |
| raw data | `Logs/benchmarks/*.log/.json`, `Logs/cpu/*.csv`, `Logs/cpu/perf-decode-w28.data`, `Logs/gpu/*.nsys-rep`, `Logs/gpu/nsys-*.sqlite` |

Bench harness: `bench/v100/bench.py` (+ `sweep_workers.sh`, `spec_sweep.sh`,
`vram_sweep.sh`, `perf_decode.sh`, `prefill_prompt.txt[.tokens]`).
