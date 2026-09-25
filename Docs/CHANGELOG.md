# CHANGELOG

Important historical changes and decisions. No raw logs.

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
