# V100 Stage 1.3 Final — Expert Pool Synchronization Optimization

Branch `stage1.3-expert-pool-sync` (from `stage1.2b-ple-ram-default` @ e4ebe92). Scope: the
complete lifecycle of `wait_flag_ge_kernel` on the 125B IQ3_XXS native pack (Swift-1.5
Qwen3.8-Flash-Next), `--pool-workers 24`, `--ple-io ram`, 1.2B frozen, V100-32 GB GPU0 primary
/ V100-16 GB GPU4 validation. Baseline `--pool-workers 24 --ple-io ram` at 43.48 tok/s.
One optimization at a time; no PLE/RAM, pool-workers, KV, speculation (MTP), Flash Attention,
dense-model, or unrelated-kernel changes.

## 1. The sync path (what `wait_flag_ge` actually waits for)

Each verify window runs as one captured graph on stream `cs_`. Per layer `l` the GPU
`pre(l)` → doorbell (device write to mapped `h_seq_`) → host spins on the ring →
`expert_pool_dispatch_multi` (CPU: plan → publish flag A → staging DMA fetch → actq →
jobs → pool `run_split_multi_native` = the CPU expert GEMVs) → fence → set flag B (staging
done, raised by a `cudaLaunchHostFunc` on the copy stream) → set flag C (`h_flag_`, CPU rows
in `h_ymiss_` ready). The GPU `post(l)` then executes `wait_flag_ge` three times:

- **flag A** — waits for the CPU *plan* (ptr/ptr2 tables, the residency split). Host work.
- **flag B** — waits for the *staging DMA completion*. nsys shows the smoking gun directly:
  at the 0.55 baseline, 1,729 of the 5,136 wait instances have their next global event on
  *another* stream — the staging copy-engine DMA — i.e. the wait ends exactly when the
  Host→Device staging copy finishes.
- **flag C (`h_flag_`)** — waits for the *CPU expert rows* (the GEMV results the pool wrote
  into mapped memory).

Measured `wait_flag_ge` attribution, nsys 2024.6.2, 256-token decode,
`bench/v100/s13_trace.py` (canonical classifier):

| attribution | baseline `--pcie-frac 0.55` | final `--pcie-frac 0.2` |
|---|---:|---:|
| flag A (next event = plan copy) | 769.0 ms (n=5110) | 709.4 ms (n=5301) |
| flag B, DMA-tie (next event = staging DMA on copy stream) | **1338.9 ms (n=1729)** | 385.6 ms (n=208) |
| flag B, otherwise | 131.4 ms (n=3433) | 132.8 ms (n=5153) |
| flag C (`copy_from_mapped`, CPU rows) | 34.2 ms (n=5136) | **376.6 ms (n=5322)** |
| **total GPU wait** | **2273.6 ms** | **1604.4 ms (−29 %)** |
| H2D staging | 24,508 copies / 45.1 GB / 4811.8 ms busy | 16,749 / 31.6 GB / 3711 ms |

`--pcie-frac` sets how many of the 256 routed experts per layer are *staged to the GPU*
(the PCIe share) versus computed by the CPU pool. 0.55 → 141/256 staged (1.89 distinct/layer
H2D, 2.27 CPU); 0.2 → 51/256 staged (0.34 H2D, 3.63 CPU).

## 2. The actual bottleneck (baseline)

64 % of the GPU's `wait_flag_ge` time (1338.9 ms of 2273.6 ms) was **flag B waiting on the
PCIe staging DMA**. The GPU sits idle while the copy engine moves ~1.9 expert blobs/layer
(45.1 GB total) from the pinned expert arena to VRAM per decode. The CPU pool meanwhile ran
at only 7–9 % of the box. The fix was to move most of the missed experts to the CPU pool and
stage far fewer: `--pcie-frac 0.2`.

## 3. Baseline vs optimized (final configuration, same-window)

| metric (256 tok, GPU0 32 GB, workers 24) | baseline 0.55 | final 0.2 |
|---|---:|---:|
| decode | 43.48 / **43.60** tok/s (same-window re-run) | **48.62 / 48.87 tok/s (+11.5–12.3 %)** |
| wait for rings | 36.6 ms/round | 25.8–26.2 ms/round |
| CPU pool | 11.5 ms/round, 2.27 distinct experts/layer | 13.4–13.9 ms/round, 3.63 |
| commit | 0.86 ms/round | 0.85 ms/round |
| GPU wait (nsys) | 2273.6 ms | 1604.4 ms (−29 %) |
| H2D staging | 45.1 GB | 31.6 GB (−30 %) |
| peak VRAM | 18,896 MiB | 18,896 MiB (unchanged) |
| window sizes (107 / 111 rounds) | T1:18 T2:19 T3:20 T4:50 | T1:29 T2:12 T3:12 T4:58 |

Earlier same-config runs on a faster machine window: 49.31–49.79 tok/s (7 runs). Run-to-run
variance on this machine is ±1–2 tok/s (see §7 environment note).

## 4. Chosen / rejected approaches

| experiment | result | verdict |
|---|---|---|
| **`--pcie-frac 0.2`** (0.55 → 0.2 default) | 43.5 → 48.6–49.8 tok/s, −29 % GPU wait, −30 % H2D, VRAM unchanged | **KEPT** |
| pcie-frac sweep (same window as baseline) | 0.35 → 46.31; 0.25 → 43.93 (fails 32 GB golden at token 6); 0.0 → 47.33 | 0.2 is the clean optimum |
| `--pcie-mode direct` (grouped kernel reads mapped arena, no staging) | 35.81 tok/s (−18 %): grouped reads ~30 % slower than VRAM grouped | REJECTED |
| spin-flush 64 µs (device-write visibility lag, round-195/287 hypothesis) | interleaved A/B at 0.2: 64 µs = 48.15/48.53/48.82, 2 ms = 48.49/49.00/49.27 tok/s; 2 ms won every adjacent pair by ~0.4 tok/s (the l1/g0 13–17 ms spins are unchanged at 64 µs — not a visibility effect) | REVERTED to 2 ms |
| commit-graph overlap (drop `cudaStreamSynchronize` in `Verifier::commit`) | 49.77 tok/s ≈ 0.2 baseline: the ~0.77 ms/round host saving is cancelled by +0.30 ms/round MTP-draft slowdown (commit GPU work now shares SMs with the draft) | REJECTED (reverted) |
| CUDA events / stream dependency as the sync mechanism | not needed: the flags are already stream-ordered (host func on the copy stream; `sfence` before stores); the measured wins come from *moving work between CPU and GPU*, not from re-ordering the waits | not pursued (per mission) |

## 5. Correctness

- **32 GB (GPU0) golden, final config (0.2): first 32 = `271 248068 198 760 1156 369 30869 5402 430 328 23202 1288 264 11952 5617 303 ...`** — 32/32, reproduced by `s13-v32-a` and `s13-v32-b`.
- **Deterministic:** two full 256-token runs of the final config are token-identical (256/256) on GPU0; same on GPU4 (256/256). Cross-config divergence (e.g. 0.2 vs 0.55 at token 98) is real CPU-vs-GPU expert numeric noise (different routing → different GEMV/grouped paths); determinism within a config holds.
- **16 GB (GPU4) golden, final config: first 32 = identical to the 32 GB golden above (32/32 cross-card)** — under the old 0.55 config the 16 GB card diverged at token 6; at 0.2 the two cards now agree for the first 32 tokens.
- **Selftests:** `ctest` 22 tests → 20 pass. The 2 failures are environmental and pre-existing: `ple_parity` (its `Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` reference shard is absent on this machine) and `platform_memory_test` (needs `ulimit -l unlimited`; passes under it). No CUDA errors, no hangs/deadlocks in any run (`rc=0` everywhere).
- **No VRAM regression:** 18,896 MiB (32 GB) and 16,133 MiB (16 GB) — both unchanged from Stage 1.2B.

## 6. 32 GB / 16 GB results

| | GPU0 32 GB | GPU4 16 GB |
|---|---:|---:|
| decode (final config) | 48.62 / 48.87 tok/s | 47.56 / 47.20 tok/s |
| vs same-window baseline | +11.5–12.3 % over 43.60 | (16 GB baseline not re-run; Stage 1.2B 16 GB ≈ 36.6 tok/s at the old config) |
| peak VRAM | 18,896 MiB (unchanged) | 16,133 MiB (unchanged) |
| golden / determinism | 32/32; 256/256 identical | 32/32 (vs 32 GB); 256/256 identical |

The 16 GB card **survives** the final configuration: fits with margin, no VRAM change,
deterministic, and faster than its old-config numbers.

## 7. Remaining bottleneck (for the next stage)

Total GPU wait at the final config: 1604.4 ms (~31 % of the 5.2 s decode).

1. **Flag A tail / early-layer ring spins (dominant):** the slowest spins are consistently
   `l1/g0` at 13–17 ms (5 occurrences per run ≈ 70 ms of decode, ~1.4 %). Cascade: the
   first window layer's dispatch is occasionally slow (a one-time 8 ms `actq` cold start on
   the first prefill call; ~100 ms clusters where the pool's `gu` phase runs ~5× slower),
   the CPU rows land late, the GPU's `post(l0)` stalls at flag C, and `pre(l1)` — hence
   ring 1 — fires late. The machine-wide slowness clusters were not fully attributed
   (per-core sampling at 10 Hz is too coarse for 10 ms events; a background stream of
   correctable MCE machine-check events — 3,142 log entries today, EDAC ce=0, no UE —
   may add retry latency). A follow-up stage should instrument the pool's per-phase
   completion times with a fine clock and the host's `schedstat` to pin down the transient.
2. **Flag C exposure (376.6 ms):** at 0.2 the CPU pool now carries 3.63 distinct
   experts/layer (was 2.27) and its rows are on the GPU's critical path. The pool runs at
   7–10 % of the box, so this is latency, not bandwidth; early-layer wake-up/ordering is
   the candidate.
3. **Round-boundary gap (~3.5 ms/round, 405 ms):** commit 0.86 ms (kept synchronous —
   overlap was measured and is a wash) + MTP draft 2.06 ms (frozen: speculation) + launch/
   driver overhead. Not in this stage's scope.

## 8. Environment notes

- **MCEs:** `dmesg` shows a continuous stream of *correctable* machine-check events
  ("Machine check events logged", ~1/minute, present from at least 12:33 today; EDAC
  `ce_count` = 0 on all 4 controllers, no uncorrectable). Pre-existing background hardware
  noise; likely a contributor to the ±1–2 tok/s run-to-run variance seen this session.
- **Node-1 OOM under concurrent runs:** two simultaneous strata processes (≈67 GB RSS
  each) exceed the 125.78 GiB box under the node-1 memory policy — the 12:32 A/B run and
  the 14:46 concurrent 32/16 GB pair were both OOM-killed (dmesg). Validation runs are
  sequential from then on.
- Raw data: `Logs/gpu/nsys-s13-{base,pcf02}*.nsys-rep` + CSV traces (gitignored),
  `Logs/benchmarks/s13-*` (json + logs + cpu/gpu sampler CSVs).

## Final configuration

`--pcie-frac 0.2` is now the program default for native packs (`src/program/generate.cpp`,
`Options::pcie_frac`); explicit `--pcie-frac`/`--pcie-mode` still override. Everything else
is the Stage 1.2B configuration (`--pool-workers 24`, `--ple-io ram` default, 2 ms spin
flush, `--spec 4 --spec-min-p 0.5 --mtp mtp/rt --kv fp16`, `--max-context 8192`).
