# V100 Performance — Stage 1.1 Results

Environment: GPU0 = Tesla V100 32 GB PCIE (`02:00.0`, dev), GPU4 = V100 16 GB
SXM2 (`84:00.0`, validation). CUDA 12.9, driver 580.178.04, engine built
`CMAKE_CUDA_ARCHITECTURES=70` at `stage1.1-performance` (4b34188). Model:
Swift-1.5 Qwen3.8-Flash-Next 125B IQ3_XXS (70.74 GiB, 2 shards) +
`mtp/rt` + `data/expert-profile.bin` (8,000 ranked pairs). All runs via
`bench/v100/bench.py` (nvidia-smi 10 Hz, /proc/stat per-core 10 Hz; raw data
in `Logs/{benchmarks,cpu,gpu}/`).

**Determinism / correctness bar:** golden 32-token output (identical to the
independent vanilla llama.cpp reference, Stage 1):

```
271 248068 198 760 1156 369 30869 5402 430 328 23202 1288 264 11952
5617 303 220 17 15 17 21 11 321 369 883 310 3184 728 883 836 1118 1834
```

## Baseline (Phase 0), GPU0, workers 28

| metric | value |
|---|---|
| decode, 256 tokens sustained | **39.55 tok/s** (6,473.6 ms) — matches Stage 1 39.7–39.9 |
| decode, 32 tokens | 31.29 tok/s |
| prefill, 2,047-token prompt (1 chunk) | **287.9 tok/s** (7,107.5 ms; engine chunk line 6,934 ms @ 295.1); TTFT 7.26 s |
| prefill, 28 tokens (warmup-heavy) | 20.2 tok/s; TTFT 1.53 s |
| peak VRAM | 18,896 MiB |
| GPU busy (nsys) | 85 % of decode wall (see `v100-gpu-analysis.md`) |
| CPU mean (decode window) | 13–19 % of 56 logical CPUs |
| expert-cache hits | 122,912 / 140,270 = 87.6 % (0 admitted, 17,358 refused, 100 % full) |
| CPU pool | 11.44 ms/round (drain 8.67 ms), 22.3 GB/s effective |
| PLE (decode) | 5,504 rows, 4,488 SSD reads, p50 3.72 ms, blocked 669 ms total |
| spec | 107 rounds × 4, accepted 149/209 (0.713), 2.39 tok/round |

### Prefill gap vs Stage 1 (287.9 vs 401.1 tok/s) — explained

Stage 1's 401.1 tok/s used its own 2,047-token prompt (not preserved in the
repo). With this run's deterministic 2,047-token prompt (same length,
prose) the engine reports: `prefill 2046 tokens in 1 chunks, 6934.0 ms;
experts streamed 11439 (11439 by DMA), resident 6365; PLE 2800.7 ms`.
Breakdown of the 6.93 s:

- **PLE table reads: 2.80 s (40 %)** — 33,008 row demands, 20,376 unique SSD
  reads (85.2 MB) at p50 **7.9 ms per read**, 64 in flight, 38 % row-cache
  hits. The NVMe serves ~8 ms per 4 KB-class random O_DIRECT read here, so
  64-deep pipelining floors out at ~2.5 s (2,518 ms submit-time of the 2,801).
- **Expert streaming: 11,439 expert blobs DMA'd from the pinned arena**
  (≈ PCIe, concurrent with the GPU work).
- Remainder: GPU compute of the 2,048-token chunks (~4 s; prefill GPU ~79 %
  busy in the measured 28-token chunk, see `v100-gpu-analysis.md`).

Re-run variance: 287.86 / 288.68 tok/s across two runs (the prompt content,
hence PLE rows and expert routing, is what moves the number, not the
machine — SSD was idle between runs, verified via /proc/diskstats).

## Benchmark matrix

Decode = 256 tokens greedy from the 29-token Stage 1 prompt. Prefill = the
2,047-token prompt, `--max-new 16`. Correctness = 32-token golden prefix.

| config | decode tok/s | prefill tok/s | TTFT | peak VRAM | 32/32 golden |
|---|---:|---:|---:|---:|:--:|
| Stage 1 baseline (fa146c9 era, workers 28) | 39.7–39.9 | 401.1 (their prompt) | 5.27 s | 18.7 GiB | yes |
| Stage 1.1 Phase 0 (same build, workers 28) | 39.32–39.55 | 287.9 | 7.26 s | 18.90 GiB | yes |
| **best CPU: workers 24** | **39.65–39.99** | n/a | n/a | 18.90 GiB | yes |
| best VRAM/expert-cache variant | 39.23–39.37 (no change, see below) | n/a | n/a | 18.90 GiB | yes |
| **best overall: workers 24, all else default** | **39.8–40.0** | 287.9 (PLE-bound) | 7.26 s | 18.90 GiB | yes |
| 16 GB validation (GPU4, workers 24) | 36.6 | n/a | 1.44 s | 16.13 GiB | GPU4-deterministic (see below) |

## Optimization results (Phase 7 — every knob measured, BEFORE/CHANGE/AFTER)

| change | decode tok/s | effect | kept? |
|---|---|---|---|
| (baseline) workers 28 | 39.32–39.55 | — | — |
| workers 24 | 39.65–39.99 | **+1.0 %**, SMT/ spin relief | **yes** |
| workers 32 / 40 | 39.63 / 38.91 | ±0 | no |
| `--spec 2` | 39.19 | −1 %, output diverges at tok 7 | no |
| `--spec 3` | 39.47 | −0.2 %, output diverges at tok 7 | no |
| `--spec 4 --spec-min-p 0.7` | 37.69 | −5 % (1.83 tok/round) | no |
| `--kv int8` | 37.54 | −5 %, output diverges at tok 7 | no |
| `--vram-reserve-mib 300` | 39.37 | 0 % (cache still 8,000 — profile cap) | no |
| `--expert-cache 12000 / 16000` | 38.57 / 39.23 | 0 % (profile fills 8,000; 0 admitted) | no |
| `--expert-cache-per-layer` | — (rc=1) | **engine bug**: `verify_slot: slot 0 differs from the arena` with profile fill | no (reported) |
| PLE `--no-ple-prefetch` | 39.01 | −1.3 % | no (default prefetch better) |
| PLE `--ple-sync-submit` | 39.35 | −0.5 % | no (default I/O thread better) |
| PLE `--ple-row-cache 0` | 38.83 | −1.8 % | no (default cache better) |
| PLE `--ple-io mmap` (prefill) | — | prefill 183 tok/s vs 295 (PLE 7.07 s vs 2.80 s) | no (direct better) |
| PLE `--ple-inflight 256` (prefill) | — | prefill 297 vs 295 (0.7 %) | no (64 ≈ 256) |
| `--spec 0` (pure greedy) | — | not allowed by the native pack (`--spec T >= 2`) | n/a |

Correctness notes:

- Every config that preserves the resident set and numerics (workers,
  vram-reserve, expert-cache size, PLE arms, spec4-p0.7) reproduces the
  golden 32-token prefix **32/32**.
- `--spec 2/3` and `--kv int8` change the greedy argmax at token 7 — the
  verify window (multi-token GDN fused kernels) and KV quantization are
  numeric paths, not pure acceleration. Documented; baseline semantics kept.
- Worker count is numeric-invariant (pool only, 32/32 at every count).

## Expert cache / residency (Phase 6)

- Cache: 8,000 slots / 12.93 GiB, PROFILE policy (ranked by routing
  frequency, no eviction), pre-filled at startup ("pre-filled 8000 of 8000
  slots from the profile; slot 0 verified"), plus an **adaptive tier** that
  swaps 2,365 hot experts in every 4 rounds (0.151 ms/round).
- Decode: 87.6 % hit rate; 17,358 misses (3.38 routed / 2.27 distinct per
  layer); miss path = CPU pool + PCIe DMA of the expert blob
  (1.89 distinct experts/layer read over PCIe, "141/256 of the misses").
- Cache **size is not the limiter**: 12,000/16,000-slot caches give
  bit-identical hit/miss counts (profile fills 8,000; on-demand admission = 0
  under the no-eviction profile policy). Raising residency requires a bigger
  profile (the generator `tools/make_profile.py` is referenced in-source but
  absent from the tree — the 8,000-pair artifact is tracked in `data/`).
- GPU4 (16 GB): 6,321 slots, 84.6 % sustained hit rate, +work to the pool.
- The GPU hit path is upstream-declared NOT CORRECT (opt-in, startup
  warning). Stage 1.1 keeps it on (tokens match the llama.cpp reference);
  any experiment changing the resident set re-checks the 32/32 golden — done
  for all kept configs.

## PCIe / I/O (Phase 3)

From the nsys MEMCPY records (decode window, 7.04 s):

| direction | volume | rate | transfers |
|---|---:|---:|---:|
| HtoD (pinned arena → GPU) | 41.6 GB | **5.9 GB/s** sustained | 23,410 (23,267 of them 1–8 MB expert blobs) |
| DtoH (GPU → pinned) | 2.4 MB | — | 49 (logits/sampling readbacks) |

- 5.9 GB/s ≈ 45 % of PCIe3 x16 effective (~12–14 GB/s); the link is not
  saturated — the DMA is demand-driven by the 12.4 % miss rate, and it
  overlaps GPU work (async copies; only 2.4 MB crosses DtoH).
- PLE: O_DIRECT NVMe, 4 KB-class random rows. Decode 18.8 MB / 6.5 s
  (pipelined, off critical path except 2.6 ms/token blocked); prefill 85.2 MB
  at p50 7.9 ms/read (see prefill gap above).
- Host↔GPU zero-copy reads inside the token path: ~286 ms per 256-token
  decode (`copy_*_from_mapped` kernels) — small.

## Stability

- 30+ runs across Phases 0/1/3/7/8: engine rc=0 in all, no CUDA errors, no
  hangs; deterministic output (identical token sequences run-to-run for the
  same config; first divergence across GPU0/4 at token 6 as expected).
- Peak VRAM stable: 18,852–18,896 MiB (GPU0), 16,111–16,133 MiB (GPU4).

## Stage 1.2A — RAM-resident PLE (branch `stage1.2a-ple-ram`, from c16ec22)

Full write-up: `Docs/v100-stage1.2a-final.md`; analysis: `Docs/ple-ram-analysis.md`.
`--ple-io ram` (new, opt-in, default stays `direct`): the 26.82 GiB PLE table is
preloaded once at startup into RAM (2.3 s warm / 17–28 s cold) and served by
memcpy instead of 4 KiB O_DIRECT preads. Same deterministic workload as the
Stage 1.1 matrix (workers 24; prefill = 2,047-token prompt; decode = 256 tokens).

| metric, GPU0 32 GB | SSD `direct` | RAM `--ple-io ram` |
|---|---:|---:|
| prefill 2046 tokens | 7,477.6 ms (273.6 tok/s) | **4,869.9 ms (420.1 tok/s, −35 %)** |
| prefill end-to-end | 267.4 tok/s | **406.6 tok/s (+52 %)** |
| TTFT | 8,035.6 ms | **5,471.5 ms (−2.56 s)** |
| PLE NVMe reads at inference | 20,376 / 85.2 MB | **0** (device-level verified) |
| decode 256 tokens | 35.38 / 35.50 tok/s | **44.69 / 42.24 tok/s (+19–26 %)** |
| peak VRAM | 18,852 MiB | 18,852 MiB (unchanged) |
| process peak RSS | ~42 GiB | **67.5 GiB** (measured; 125.78 GiB machine, ~44 GiB headroom) |

16 GB (GPU4): prefill 260.9 tok/s → **418–441 tok/s** (+60–68 %), decode
35.43 → **40.04 tok/s** (+13 %), peak VRAM 16,133 MiB unchanged, 6,321-slot
cache and hit rate identical.

Correctness: table-level bit-identity (mmap = direct = ram, 52,752 rows,
`ple_reader_test --gguf --ram`); 32/32 golden reproduced on BOTH cards in BOTH
modes; token-for-token identical across SSD/RAM. The PLE table (26.82 GiB) was
the last of the model's components still touching NVMe at inference; the expert
blobs were already RAM-resident (pinned arena, loaded once at startup).
Recommendation: keep `direct` as the default (64 GB machines); use
`--ple-io ram` on this 128 GB box.

## Stage 1.2B — `--ple-io ram` is now the default (branch `stage1.2b-ple-ram-default`)

The 1.2A recommendation is adopted as the program default: `strata` with no
`--ple-io` runs in RAM mode (canonical default: `Options::ple_io` in
`src/program/generate.cpp`; pinned by the `ple_default_mode` ctest).
`--ple-io direct` (lower-RAM fallback) and `--ple-io mmap` (alternative/test)
are unchanged and explicit selection always wins. Startup logs the active mode
(`PLE I/O mode: ram`), and a low-RAM guard fails clearly before the preload on
machines short of table + ~42 GiB (naming `--ple-io direct`) instead of
switching modes silently. All four modes re-validated: 32/32 golden,
token-identical; default-mode numbers match the 1.2A RAM results (prefill
406.2 tok/s chunk / TTFT 6.15 s, decode 42.2–44.0 tok/s, peak RSS 67.55 GiB,
zero PLE NVMe at inference). Full write-up: `Docs/v100-stage1.2b-final.md`.

## Stage 1.3 — Expert pool synchronization optimization (branch `stage1.3-expert-pool-sync`)

Full write-up: `Docs/v100-stage1.3-final.md`. Scope: the complete
`wait_flag_ge_kernel` lifecycle on the IQ3_XXS native pack; one optimization at a time;
no PLE/pool-workers/KV/speculation/Flash-Attention/dense changes.

**The bottleneck (measured, nsys 2024.6.2, 256-token decode):** at the 0.55 baseline,
1338.9 ms of the 2273.6 ms total GPU `wait_flag_ge` time was **flag B waiting on the PCIe
staging DMA** (its next global event is the staging copy on the copy engine — the
smoking gun). The CPU pool ran at 7–9 % of the box while the GPU sat idle on ~1.9 expert
blobs/layer of staging.

**The fix: `--pcie-frac 0.2`** (was 0.55) — move most missed experts to the CPU pool,
stage far fewer. Now the program default for native packs.

| metric, GPU0 32 GB, workers 24 | baseline 0.55 | final 0.2 |
|---|---:|---:|
| decode 256 tokens | 43.48 / 43.60 tok/s | **48.62 / 48.87 tok/s (+11.5–12.3 %)** |
| total GPU wait (nsys) | 2273.6 ms | **1604.4 ms (−29 %)** |
| flag B (staging DMA) | 1338.9 ms | 385.6 ms |
| flag C (CPU rows) | 34.2 ms | 376.6 ms (new critical path) |
| H2D staging | 45.1 GB | 31.6 GB (−30 %) |
| peak VRAM | 18,896 MiB | 18,896 MiB (unchanged) |

Measured pcie-frac curve (same window): 0.55→43.48, 0.35→46.31, 0.25→43.93 (fails golden),
0.2→48.6–49.8, 0.0→47.33. `--pcie-mode direct` (no staging)→35.81 (rejected).

**16 GB (GPU4) survives the final config:** 47.56 / 47.20 tok/s, peak VRAM 16,133 MiB
(unchanged), 32/32 golden **now identical to the 32 GB golden** (was divergent at token 6
under 0.55), 256/256 deterministic.

**Rejected (measured, reverted):** spin-flush 64 µs (interleaved A/B at 0.2: 2 ms won every
adjacent pair by ~0.4 tok/s — not a visibility effect); commit-graph overlap (drop the
`Verifier::commit` sync; the host saving is cancelled by +0.30 ms/round MTP-draft
slowdown). Correctness: 32/32 golden on both cards, 256/256 deterministic, ctest 20/22
(2 pre-existing environmental), no CUDA errors, no VRAM regression.

# V100 Performance — Stage 1.4 Results (CPU row production / flag C)

Full write-up: `Docs/v100-stage1.4-final.md`. Config: 1.3 defaults (`--ple-io ram`,
`--pcie-frac 0.2`) + `--pool-workers 24`, 256-token decode. Baseline = 1.4 frozen
(48.88/48.62 tok/s GPU0; 47.56/47.20 GPU4; flag C 360.5 ms of GPU `wait_flag_ge`,
n=5328, mean 67.7 µs).

**Flag C root cause:** the GPU waits per dispatch (48/round × 111 rounds = 5,328 waits)
for the CPU down-projection rows, which in the baseline pipeline were serialized behind
two worker barriers plus a host-serial intermediate-quantization loop. The pool's 25
threads were already 82–83 % busy (20.5 thread-equiv) while the box-wide mean was ~7 % —
the pool is compute/synchronization-bound, not capacity-bound ("more CPU workers"
rejected by measurement, H1a).

**The fix (H2, default):** one fused barrier phase per dispatch (pool mode 7) — gu row
chunks accumulate a per-expert release counter; per-expert quantization runs as pool
tasks overlapping the other experts' work; down chunks wait per-expert. Same row work,
same kernels, same numerics; one barrier and one host-serial step removed per dispatch.
`STRATA_POOL_UNFUSE=1` restores the baseline path.

| metric, 256 tok | 1.4 baseline | final (fused default) |
|---|---:|---:|
| decode, GPU0 32 GB | 48.88 / 48.62 tok/s | **50.99 / 51.01 tok/s (+4.5–5.1 %)** |
| decode, GPU4 16 GB | 47.56 / 47.20 tok/s | **47.83 / 48.25 tok/s (+0.6–2.4 %)** |
| per-token latency, GPU0 | 20.46–20.57 ms | 19.61 ms (−4.3 %) |
| pool wall (256 tok, A/B) | 1,358.3 ms mean | 1,205.4 ms mean (−11.2 %) |
| pool threads | 82 % busy, 9,130 phases | 92–93 % busy, 4,565 phases |
| flag C (nsys) | 360.5 ms | **193.6–292.3 ms (−19 to −46 %)** |
| total `wait_flag_ge` (nsys) | 1,635.6 ms | ~1,667 ms (flat) |
| peak VRAM | 18,896 / 16,133 MiB | unchanged |

**H3 (decode-once AVX2 expert dots):** ggml-cpu's AVX2 dots re-decode weight codebooks per
token; the new kernels decode once per (row, block) and are **bit-identical** to the
per-token AVX2 path (parity harness: 122,880 rows × nt 1..4, all five gu formats, both
GGUF shards). Interleaved 2×2 A/B is e2e-neutral (50.45 vs 50.64 tok/s) — the pack's down
projections are Q2_K/Q2_0, so only 1/3 of the dot FLOPs (gu rows) are affected while the
pool is compute-bound. Shipped opt-in: `STRATA_IQAVX2=1` (off by default).

**Rejected (measured, reverted to default):** H1a `--no-host-worker` (49.21 tok/s, pool
wall +2.4 %); H3 as default (e2e-neutral, see above). Correctness: 32/32 golden on both
cards, 256/256 deterministic per card (H2 and H3 both 0-token-difference vs the 1.4
baseline sequence on GPU0), ctest 20/22 (same 2 pre-existing environmental failures),
no CUDA errors, no VRAM regression. Remaining bottleneck (1.5 candidates, not touched):
flag A 1,100.9 ms (early-layer `l1/g0` tail — frozen here), residual flag C 193.6 ms
(per-dispatch host path ≈ 1.9 ms/round × 48), flag B 372.9 ms, pcie-frac 0.25–0.35 sweep.
