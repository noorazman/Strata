# V100 Stage 1.4 Final — CPU Row Production (Flag C)

Branch `stage1.3-expert-pool-sync` (1.4 work sits on `b407beb`, Stage 1.3 frozen:
`--pcie-frac 0.2` default, untouched). Scope: the ~377 ms of GPU-visible waiting that
`wait_flag_ge` spends on **flag C (`h_flag_`)** — the flag the host raises when the CPU
pool's expert rows (down-projection outputs in `h_ymiss_`) are ready — despite relatively
low overall CPU utilization. Config throughout: `--ple-io ram`, `--pool-workers 24`,
`--pcie-frac 0.2` (default), `--expert-cache auto --spec 4`, 256-token decode, V100-32 GB
GPU0 primary / V100-16 GB GPU4 validation. One hypothesis at a time. Frozen per stage
rules: PLE, PCIe-frac, KV, speculation, Flash Attention, context-shifting, unrelated
kernels; the small Flag A `l1/g0` ring spins are explicitly not optimized in this stage.

Golden bar (identical to the independent vanilla llama.cpp reference, Stage 1):

```
271 248068 198 760 1156 369 30869 5402 430 328 23202 1288 264 11952 5617 303 ...
```

## 1. What flag C actually waits for

Each verify round dispatches the missed experts of all 48 layers to the CPU pool:
**48 dispatches/round × 111 rounds = 5,328 dispatches** (engine stats: `5328 dispatches`).
For each dispatch the host: plans (0.38–0.43 ms/round total), quantizes the input
activations (0.75–0.79), publishes the job tables, runs the pool (`run 10.5–12.4 ms/round`),
then raises flag C. The GPU's `post(l)` for that layer spins in `wait_flag_ge` until the
rows are in memory.

Baseline measurement (nsys, `Logs/gpu/nsys-s14-base.sqlite`, classifier
`flagc_sqlite.py` — attribution by the next same-stream kernel: `copy_i32_from_mapped` =
A, `copy_from_mapped` = C, otherwise B):

| attribution | 1.4 baseline |
|---|---:|
| flag A (plan) | 969.4 ms (n=5328) |
| flag B (staging DMA) | 305.7 ms (n=5328) |
| **flag C (CPU rows)** | **360.5 ms (n=5328, mean 67.7 µs, p99 599 µs, max 19.1 ms)** |
| total `wait_flag_ge` | 1,635.6 ms |

Flag C is not one long wait — it is 5,328 waits of mean 67.7 µs. The question of the stage:
why is the GPU sitting idle for 360 ms on a box whose overall CPU utilization is ~7 %?

## 2. Root cause

**The GPU is not waiting for the CPU to be busy — it is waiting for the serialized
host-side row-production pipeline, whose bottleneck is synchronization and a
host-serial step, not CPU capacity.**

Evidence:

1. **The pool is compute-bound on its own threads, not starved.** Baseline pool threads:
   `busy 26,730 ms = 82 % of 25 threads (20.5 thread-equiv)`, `park-wait 0.4 ms`,
   `wait 0.0` (pool phases). The pool's 25 of 56 logical CPUs are 82–83 % busy while the
   box-wide mean is ~7 % — the other ~31 cores are simply not in the expert path. This is
   why "more CPU workers" is *not* the answer (see H1a, §4).
2. **The baseline native multi-expert path serializes the down rows behind two worker
   barriers plus a host-only quantization loop** (`run_split_multi_native`, pre-change):
   phase 5 = gate/up row chunks (barrier), then the *dispatch thread alone* quantizes every
   expert's intermediate (single-threaded, `for e, for t: native_quant_h`), then phase 6 =
   down row chunks (second barrier). The down rows — the ones flag C gates — cannot start
   until every gu chunk on every thread is done *and* the host finishes the serial
   quantization. Measured cost: 9,130 phases (two barriers/dispatch), 20.5–20.7
   thread-equiv busy, pool 14.376 ms/round inside the verify window.
3. **The host critical path exceeds the GPU's per-round need.** The verify window
   decomposes (engine `--stats`): baseline `wait-for-rings 25.616 + pool 14.376 + host 0.072
   + commit 0.843 = 40.9 ms/round` vs a decode pace of ~19.6 ms/token × ~2.3 tok/round.
   The row-production half (host plan/actq/jobs + pool) is longer than the GPU consumes,
   so every dispatch lands a tail on flag C.
4. **Low box CPU utilization is expected, not a symptom.** 23 thread-equiv of work on 56
   logical CPUs, active for ~12 ms of every ~19.6 ms token, gives single-digit-to-low
   double-digit box-wide percentages while the pool itself is saturated.

## 3. The optimization (H2, kept — now the default)

One fused barrier phase per dispatch instead of two (pool mode 7, default;
`STRATA_POOL_UNFUSE=1` restores the baseline two-phase path):

- **Gate/up row chunks** (work-stealing, as before) now accumulate a per-expert release
  counter (`fgate_` atomic, `fetch_add` with release at expert boundaries).
- **Per-expert quantization** runs as pool *tasks* (nb of them): a worker spins (acquire)
  until its expert's gu rows total FF, quantizes all of that expert's tokens, and bumps the
  counter to FF+1. Quantization overlaps with other experts' gu/down work instead of
  running host-serial between two barriers.
- **Down row chunks** wait (acquire, at expert boundaries only) for their expert's counter
  to pass FF, then compute as before.

Same row work, same kernels, same per-row numerics — one barrier and one host-serial step
removed per dispatch.

Same-window interleaved A/B (256 tok, GPU0, workers 24; `STRATA_POOL_UNFUSE=1` = baseline
arm):

| run | tok/s | pool wall (ms, 256 tok) | pool threads busy |
|---|---:|---:|---:|
| baseline b1 | 48.36 | 1,408.8 | 82 % (20.5 equiv) |
| **fused f1** | **50.79** | **1,171.6** | **93 % (23.4 equiv)** |
| baseline b2 | 49.68 | 1,307.7 | 80 % |
| **fused f2** | 48.06 | **1,239.2** | 92 % (23.2 equiv) |
| **mean** | **49.02 → 49.43 (+0.8 %)** | **1,358.3 → 1,205.4 (−11.2 %)** | 81 % → 92–93 % |

Flag C (nsys, same classifier):

| | 1.4 baseline | fused (H2 validation) | final candidate (re-measured) |
|---|---:|---:|---:|
| flag C total | **360.5 ms** | 292.3 ms (−19 %) | **193.6 ms (−46 %)** |
| flag C mean / p99 / max | 67.7 / 599.2 / 19,144 µs | 54.9 / 485.4 / 46,261 µs | 36.3 / 438.8 / 13,407 µs |
| flag A | 969.4 ms | 1,042.5 ms | 1,100.9 ms |
| flag B | 305.7 ms | 386.9 ms | 372.9 ms |
| total `wait_flag_ge` | 1,635.6 ms | 1,721.7 ms | 1,667.4 ms |

The A/B/C split is heuristic (next-kernel attribution) and run-to-run; the defensible
claim is flag C **360.5 → 193.6–292.3 ms (−19 % to −46 %)** with total `wait_flag_ge`
statistically flat (~1.64–1.72 s) — the fused phase moves time *out of the CPU-rows wait*
without creating new waits elsewhere.

## 4. Hypotheses tested

| H | hypothesis | result |
|---|---|---|
| H1a | `--no-host-worker` (drop the host from the pool) helps | **Rejected** — 49.21 tok/s, pool wall +2.4 %. The host thread's ~1 thread-equiv is worth more than its dispatch-loop interference. |
| H2 | the two-barrier + host-serial-quantization pipeline delays the down rows (flag C) | **Kept (default).** Pool wall −11.2 %, flag C −19 % to −46 %, same-window +0.8 %; 256/256 golden ×7. |
| H3 | ggml-cpu's per-token AVX2 dots re-decode weight codebooks per token; a decode-once kernel removes that redundancy | **Kept as opt-in (`STRATA_IQAVX2=1`), e2e-neutral.** Bit-exact across all five gu formats (122,880 rows × nt 1..4 vs the per-token AVX2 reference, both GGUF shards, `src/kernels/iq_avx2_parity.cpp`), yet interleaved 2×2 A/B is 50.45 vs 50.64 tok/s and 12.44 vs 12.31 ms/tok pool — inside the ±1–2 tok/s MCE drift. Per the stage rule ("keep only if it improves e2e"), it ships off by default. The down projections of this pack (Q2_K/Q2_0) are not IQ types, so only the gu rows (1/3 of the dot FLOPs) are affected — consistent with a neutral result while the pool is compute-bound. |

## 5. Baseline vs final (shipping candidate, 256 tokens, workers 24)

| metric | 1.4 baseline (frozen) | final (fused default) | Δ |
|---|---:|---:|---:|
| decode, GPU0 32 GB | 48.88 / 48.62 tok/s | **50.99 / 51.01 tok/s** | **+4.5 % / +5.1 %** |
| decode, GPU4 16 GB | 47.56 / 47.20 tok/s (1.3 final = 1.4 baseline) | **47.83 / 48.25 tok/s** | +0.6 % / +2.4 % |
| per-token latency, GPU0 | 20.46 / 20.57 ms | **19.61 / 19.61 ms** | −4.3 % |
| pool in verify window | 14.376 ms/round | 12.386 ms/round | −1.99 ms/round (explains the e2e gain: rings 25.616→25.828 and commit 0.843→0.857 are unchanged) |
| pool wall (256 tok) | 1,306–1,413 ms | 1,158–1,270 ms | −11.2 % (A/B) |
| pool threads busy | 82 % (20.5 equiv), 9,130 phases | 92–93 % (23.1–23.2 equiv), 4,565 phases | one barrier/dispatch, no host-serial quant |
| CPU worker behavior | 25 threads + host, drain 12.421 ms/tok, re-park 0.0, wait 0.0 | same staffing; drain 10.39–10.47 ms/tok, re-park 0.0, wait 0.0 | pool stays compute-bound, idle time stays ~zero |
| flag C (nsys) | 360.5 ms | 193.6 ms | −46 % |
| peak VRAM | 18,896 / 16,133 MiB | 18,896 / 16,133 MiB | unchanged |
| box CPU mean | ~7–8 % | 6.4–7.4 % (GPU0), 6.9–9.8 % (GPU4) | unchanged (low by construction, §2.4) |

## 6. Correctness

- **Golden 32/32 on both cards** (both runs of each card, final candidate).
- **256/256 deterministic**: GPU0 run1 == run2 (all 256 tokens); GPU4 run1 == run2.
- **H2 is bit-exact end-to-end**: 1.4 baseline 256-token sequence vs final 256-token
  sequence on GPU0 — **0 token differences** (same build lineage, fused only re-schedules).
- **H3 is bit-exact end-to-end**: H3-on vs H3-off 256-token sequences — 0 differences;
  plus the row-level parity harness (122,880 rows × nt 1..4, gu types 16/17/18/21/22,
  both GGUF shards; down Q2_K/Q2_0 stay on the per-token path by design).
- Cross-card note: GPU0 and GPU4 256-token sequences diverge from token 136 (108
  differences) — **identical pattern to the 1.3 final runs** (same first-divergence index
  and count), i.e. pre-existing (different resident expert sets past the golden window),
  not introduced by 1.4. First 135 tokens identical across cards.
- **ctest 20/22** — the same 2 pre-existing environmental failures as 1.3 (`ple_parity`
  missing Q2_0 reference shard; `platform_memory_test` needs `ulimit -l unlimited`).
- No CUDA errors, no hangs: every run `rc=0` (14 bench runs + 2 nsys profiles).
- Environment: pre-existing correctable MCE stream (hence the ±1–2 tok/s drift and the
  interleaved-A/B protocol) and node-1 OOM under concurrent runs — all validations ran
  sequentially with `drop_caches` between arms.

## 7. Remaining bottleneck (for Stage 1.5 — documented, not touched)

1. **Flag A — 1,100.9 ms (nsys, final):** the dominant GPU-visible wait, mostly the
   early-layer `l1/g0` spin tail (13.4–13.7 ms spins at t+1117/1503/1889) — explicitly
   frozen in this stage. Host plan/publish cost (plan 0.38–0.43 ms/round) is its share.
2. **Residual flag C — 193.6 ms:** the serialized per-dispatch host path (plan +
   activation quantize + job publish ≈ 1.9 ms/round before the pool even starts, ×48
   dispatches/round) still exceeds the GPU's per-round consumption. Further reduction
   needs a smaller per-dispatch host cost or a cross-layer pipeline (e.g. dispatch layer
   l+1 while the pool finishes l) — out of scope here.
3. **Flag B — 372.9 ms:** staging-DMA tail at pcie-frac 0.2 (0.34 distinct experts/layer
   staged). A pcie-frac 0.25–0.35 sweep was noted as a 1.5 candidate and not swept (frozen
   per stage rules).
4. Round-boundary gap (commit 0.857 + MTP draft ~2.1 ms/round) — frozen.

## 8. Environment notes

- GPU0 = Tesla V100 32 GB PCIE (dev), GPU4 = V100 16 GB SXM2 (validation); CUDA 12.9,
  driver 580.178.04; 56 logical / 28 physical cores, AVX2-only (no AVX-512).
- Model: Swift-1.5 Qwen3.8-Flash-Next 125B IQ3_XXS (70.74 GiB, 2 shards); gu expert
  types: IQ2_XXS(16) ×9, IQ2_XS(17) ×10, IQ3_XXS(18) ×6, IQ3_S(21) ×13, IQ2_S(22) ×10
  layers (GGUF `blk.0..blk.11` in shard 1, `blk.12..blk.47` in shard 2); down: Q2_K(20),
  Q2_0(42). Adaptive tier ~2,503 experts, expert cache 8,000 slots (12.93 GiB).
- Canonical run: `bench/v100/bench.py <label> --gpu N --workers 24 --max-new 256 --stats`
  (`--kv fp16`, `--max-context 8192`, `--spec 4 --spec-min-p 0.5`, `--ple-io ram` via the
  1.2B default, `--pcie-frac 0.2` via the 1.3 default).
- Raw data: `Logs/benchmarks/s14-*.{log,json}` (baseline, A/B, H3 A/B, final 32 GB/16 GB),
  `Logs/gpu/nsys-s14-{base,fuse,final}.{nsys-rep,sqlite}` (sqlite regenerable via
  `nsys export --type sqlite`), flag-C classifier output in this doc.
- Knobs: `STRATA_POOL_UNFUSE=1` (two-phase baseline pool), `STRATA_IQAVX2=1` (decode-once
  AVX2 expert dots, opt-in), `STRATA_POOL_TASKS=N` (row-chunk multiplier, experiment).

## Final configuration

`--ple-io ram` (1.2B default) + `--pcie-frac 0.2` (1.3 default) + **fused single-phase CPU
expert pool (1.4, default)** + decode-once AVX2 expert dots (1.4, opt-in, off by default).
GPU0 32 GB: **50.99–51.01 tok/s** (baseline 48.88/48.62), flag C **360.5 → 193.6 ms
(−46 %)**, pool wall **−11.2 %**, VRAM unchanged, 256/256 deterministic. GPU4 16 GB:
**47.83–48.25 tok/s** (baseline 47.56/47.20), 16,133 MiB unchanged. Stage 1.4 complete.
