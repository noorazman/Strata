# V100 Stage 1.5 Final — ExpertPool Idle Parking (24-core idle burn → ~0)

Branch `stage1.3-expert-pool-sync` (1.5 work sits on the Stage 1.4 frozen baseline:
fused single-phase expert pool default, `--pcie-frac 0.2`, `--ple-io ram`). Scope: the
**24 pinned expert-pool workers that spin-park on `_mm_pause()` between batches** and burn
24 full logical cores at 100 % while the engine sits idle between requests (measured 24.0
cores on a 56-core box; the idle diagnostic of 27 Sep confirmed the whole idle CPU is this
one component). Goal: eliminate the idle burn **without degrading decode**. Config throughout:
`--pool-workers 24`, `--spec 4 --spec-min-p 0.5`, `--expert-cache auto`, `--kv fp16
--max-context 8192 --prefill 2048`, 256-token decode, V100-32 GB GPU0 primary / V100-16 GB
GPU4 validation. Frozen per stage rules: PLE, PCIe-frac, pool worker count, model/quant,
Flags A/B/C optimization, worker affinity; Stage 1.6 not started.

Golden bar (identical to the independent vanilla llama.cpp reference, Stage 1):

```
271 248068 198 760 1156 369 30869 5402 430 328 23202 1288 264 11952 5617 303 ...
```

## 1. What the workers wait for (measured, not assumed)

Two measurements first (task 1 and 2 of the stage brief):

1. **Worker park-gap distribution** — new `STRATA_POOL_PARK_DIAG=1` knob prints a
   park-duration histogram at pool destruction. One full 256-token decode (spin mode,
   `Logs/benchmarks/s15-diag-spin.log`):

   | gap bucket | parks | share |
   |---|---:|---:|
   | <64 µs | 1 | 0.001 % |
   | **256 µs – 1 ms** | **98,618** | **90.0 %** |
   | 1 – 4 ms | 7,590 | 6.9 % |
   | 4 – 16 ms | 3,240 | 3.0 % |
   | ≥16 ms | 113 | 0.1 % |

   109,562 parks, 195.9 ms total parked (mean ≈ 1.79 ms). The inter-dispatch gap is the
   GPU round-trip + host plan of the verify window (5,328 dispatches = 111 rounds × 48
   layers; ~86 % of layer-rounds dispatch at least once given the 82.8 % expert-cache hit
   rate). **There is no small-gap regime to protect with a tiny spin window** — almost
   every park gap is ≥ 256 µs, and 3.1 % are ≥ 4 ms (the long ones: PCIe staging,
   adaptive-tier swap, round boundary).

2. **Topology / affinity (verified, unchanged)** — 56 logical CPUs = 28 physical cores ×
   SMT (sibling = cpu+28). The 24 workers are pinned to logical CPUs 1–24 = **24 distinct
   physical cores** (cpu1–13 socket 0, cpu14–24 socket 1); their SMT siblings (29–52) are
   free; the host thread is pinned to cpu0. This stage does not move any affinity.

## 2. Why naive futex parking is a trap (the threshold curve)

The hybrid park (spin ≤ N µs, then futex on the epoch word; wake = epoch bump +
`futex_wake`) was implemented with N configurable (`STRATA_POOL_PARK=N`) plus an
always-futex arm ("futex") as the extreme baseline. The 32 GB A/B matrix (same protocol as
1.3/1.4: `bench.py <label> --gpu 0 --workers 24 --max-new 256 --stats`, `drop_caches`
between arms; interleaved baseline for the final window):

| arm | decode tok/s | pool ms/tok | verify-window pool ms/round |
|---|---:|---:|---:|
| spin b1 | 48.64 | 13.94 | 13.95 |
| spin + diag | 49.32 | 13.16 | 13.18 |
| spin b2 | 48.89 | 13.25 | 13.27 |
| futex (0 µs) | 42.02 | 23.12 | 23.14 |
| 64 µs | 42.59 | 22.51 | 22.52 |
| 256 µs | 42.35 | 22.65 | 22.66 |
| 512 µs | 46.42 | 17.31 | 17.32 |
| 1024 µs | 48.28 | 14.08 | 14.09 |
| **2048 µs** | **48.95** (isolated: 50.36) | 13.51 | 13.51 |
| 4096 µs | 49.22 | 13.10 | 13.11 |
| 8192 µs | 48.86 | 13.47 | 13.49 |

The curve is monotonic: **always-futex costs −14 %**, and the penalty shrinks as the spin
window grows, reaching parity at 2 ms. `wait-for-rings` (24.6–27.0) and `commit` (≈0.85)
are unchanged in every arm — the entire delta is the pool phase.

**Root cause (measured, not guessed).** A 200 Hz per-core MHz probe run concurrently with a
short decode, phase-aligned to the engine's own timestamps:

- spin mode, decode window: **all 24 worker cores at 2893 MHz, 100 % of the time** (they
  spin).
- futex mode, decode window: **worker cores at 1197–1267 MHz, 98 % of the time** (hot
  fraction 2 % — the drain bursts are ~20–50 µs, below the 5 ms sampler resolution).

A core that futex-parks across an inter-dispatch gap drops to the **idle P-state
(1.2 GHz)**; the woken drain burst then runs at ~1.2–1.9 GHz instead of 2.9 GHz, so the
DRAM/LLC-bound expert rows cost ~2× longer (pool 17.2 → 31.9 ms/tok on the short 64-token
probe; 13.2 → 23.1 ms/tok on the full bench). The futex *syscall* itself is cheap — the
production-path stress harness (300 dense dispatches, `run_split_multi_native`) measures
only +1.8 % (3.33 vs 3.27 ms/dispatch, futex vs spin). The cost is the P-state: a 2 ms spin
window keeps the pinned cores at full P-state across the decode gaps (the typical gap is
256 µs–1 ms), while gaps ≥ 2 ms (3.1 % of parks) and the between-request idle still
futex-park.

## 3. The shipping change (now the default)

`src/kernels/cpu/pool.cpp` + `include/strata/kernels/cpu/pool.hpp` — the worker park
protocol becomes a **hybrid spin-then-futex park**:

- **`publish_epoch(epoch)`** — the only place the epoch moves (was a bare
  `fetch_add` in `run`, `run_phase`, and the destructor): `epoch.fetch_add(1, release)`
  followed by `futex_wake(INT_MAX)` on the epoch word whenever futex parking is in effect.
  A wake with no waiters is ~100 ns.
- **`park_on_epoch(epoch, stop, seen)`** — check the epoch (acquire); if it moved (or
  `stop`), return; else `futex_wait(expected = seen)`. The loop re-checks after every
  wake; the kernel re-checks the word under its lock, so a publish landing between the
  check and the wait cannot be lost. Spurious wakes just re-read the epoch and re-sleep.
- **Worker park loop** — when the spin budget is finite: check the epoch every iteration,
  check the `steady_clock` deadline every 1024 iterations (a deadline read per iteration
  would cost more than the budget saves), then `park_on_epoch`. The legacy pure-spin loop
  is byte-for-byte unchanged and selected by `STRATA_POOL_PARK=spin`. The
  `parked_`/`done_`/drain claim protocol is untouched — only the waiting changed.
- **glibc 2.35 detail** — this system's libc has no `futex()` wrapper symbol (verified by
  probe: `undefined reference to futex`), so the raw `syscall(SYS_futex, …)` is used
  (`SYS_futex = 202`, constants from `<linux/futex.h>`). Win32 degrades to bounded-spin-then
  `_mm_pause` (legacy-equivalent; no futex).
- **Diagnostics** — `STRATA_POOL_PARK_DIAG=1` records every park duration into a 9-bucket
   histogram (edges 1/4/16/64/256/1024/4096/16384 µs) printed at pool destruction; the
   startup line reports the active mode: `… (worker park: hybrid spin-2048us+futex
   (default))`.

**Configuration** (`STRATA_POOL_PARK`, parsed once per process):

| value | behavior |
|---|---|
| unset (new default) | **hybrid: spin 2048 µs, then futex** — the measured winner |
| `spin` | legacy pure `_mm_pause` spin (the fallback) |
| `futex` / `0` | always-futex (the extreme baseline) |
| `<N>` | hybrid: spin N µs (clamped to 10⁶), then futex |

## 4. Results

### Idle CPU (task 6): 24.0 → 0.0 cores

Resident `--serve` engine, 60 s no-request window, per-thread CPU summed (same method as
the idle diagnostic):

| arm | idle cores (60 s) |
|---|---:|
| spin (GPU0) | **24.0** |
| 2048 µs (GPU0) | **0.000** |
| 2048 µs (GPU4, 16 GB) | **0.000** |
| new default, env unset (GPU0) | 0.000 |

### Decode (task 6): no regression

Interleaved same-window, 32 GB GPU0: spin 48.89 / **2048 µs 48.95 (+0.1 %)** / 4096 µs
49.22 / 8192 µs 48.86 — all inside the ±1–2 tok/s MCE drift. 16 GB GPU4: spin 48.51 /
2048 µs 48.12 & 47.82 (−0.8 %/−1.4 %, inside drift). Per-round verify window (2048 µs):
`wait-for-rings 26.9 + pool 13.5 + host 0.09 + commit 0.85` vs spin
`26.7 / 13.3 / 0.09 / 0.86` — no round got slower.

### Wake latency (task 6)

- Dense-dispatch stress (production `run_split_multi_native` path, 8-worker pool, 300
  dispatches): futex vs spin **+1.8 %** (3.33 vs 3.27 ms/dispatch) — the pure wake overhead.
- Wake after 60 s of deep park (serve mode): spin request wall 2.1 s
  (prefill 1291.8 + decode 791.9 ms); 2048 µs: 2.1 s (1314.7 + 759.4) — indistinguishable;
  golden 32/32 in both. GPU4 2048 µs: 2.8 s (1877.1 + 926.0), consistent with GPU4's
  slower prefill on the same probe.

### Correctness (task 7)

- **Golden 32/32 on every arm** — all 11 32 GB arms and all 3 16 GB runs.
- **256/256 deterministic**: 2048/4096/8192 vs spin on GPU0 (bit-identical full sequences,
  separate windows); GPU4 2048 run-a vs run-b and vs GPU4 spin: 256/256 identical.
- **Cross-card**: GPU4 first-32 == GPU0 first-32 == golden bar (as documented in 1.3/1.4);
  full-256 cross-card divergence (108 diffs) is the same pre-existing pattern as 1.3/1.4
  (different resident expert sets past the golden window).
- **Park-protocol stress**: 100 create/dispatch-30/destroy churn cycles + 300 steady
  dispatches through the production multi-expert path — PASS in every mode (spin, futex,
  1 µs, 16/64/256 µs, 2048 µs, and the new default); first-dispatch fingerprint
  bit-identical to a fresh pool. No hangs (every run `rc=0`), no CUDA errors.
- ctest 20/22 — the same 2 pre-existing environmental failures as 1.3/1.4.

### 16 GB validation (task 8)

GPU4, same bench: decode 48.12/47.82 vs 48.51 spin (inside drift), pool 16.05/16.41 vs
15.74 ms/tok, peak VRAM 16,133 MiB unchanged, idle 0.000 cores, golden + determinism as
above.

## 5. Baseline vs final

| metric | Stage 1.4 baseline (spin park) | final (hybrid 2048 µs default) | Δ |
|---|---:|---:|---:|
| **idle CPU (60 s, 24 workers)** | **24.0 cores** | **0.0 cores** | **−100 %** |
| decode, GPU0 32 GB (interleaved) | 48.89 tok/s | 48.95 tok/s | +0.1 % (drift) |
| decode, GPU4 16 GB | 48.51 tok/s | 48.12 / 47.82 tok/s | −0.8 % / −1.4 % (drift) |
| verify window (2048 µs, GPU0) | 26.7/13.3/0.09/0.86 ms/round | 26.9/13.5/0.09/0.85 ms/round | flat |
| pool drain, 64-tok probe (spin vs futex) | 17.2 ms/tok | 31.9 ms/tok (futex) — why the threshold matters | see §2 |
| golden / determinism | 32/32, 256/256 | 32/32 ×14, 256/256 ×5 | unchanged |
| VRAM | 18,896 / 16,133 MiB | 18,896 / 16,133 MiB | unchanged |

## 6. Rejected arms (task 9)

| arm | result | reason |
|---|---|---|
| always-futex | 42.02 tok/s (−14 %) | P-state: 100 % of park gaps cross the futex; drain runs at idle P-state |
| 64 / 256 µs | 42.59 / 42.35 (−13 %) | same — the typical gap (≥256 µs) still parks futex |
| 512 µs | 46.42 (−5 %) | ~59 % of gaps still park futex |
| 1024 µs | 48.28 (−1.2 % vs best baseline) | borderline; kept as an option, not the default |

No arm showed latency spikes beyond the pool phase, no hangs, no races (stress + 14 runs),
no token divergence (14/14 golden, 5/5 determinism).

## 7. Environment notes

- Same box as 1.3/1.4: 56 logical / 28 physical cores (E5-2680 v4 ×2, AVX2 only),
  128 GB RAM, continuous correctable MCE stream (hence the ±1–2 tok/s drift and the
  interleaved-A/B + drop_caches protocol). One OOM during a *concurrent* GPU0+GPU4 idle
  test (2×26.8 GiB PLE + 2×40 GiB arena loads ≈ 133 GiB) — re-ran sequentially; no effect
  on results.
- The futex P-state mechanism is governor-dependent: under `performance`-mode P-states the
  always-futex arm would likely narrow — a Stage 1.6 system-tuning candidate (per-core
  governor for the 25 pinned CPUs), **not** part of this change.
- Residual: after a long idle the first dispatch pays one ~10 ms cold ramp (P-state +
  caches) — once per request, inside TTFT; the 60 s→request A/B above shows no
  end-to-end penalty for a 32-token request.
- Raw data: `Logs/benchmarks/s15-*.{log,json}` (all arms, both cards, idle probes),
  `Logs/benchmarks/idle-s15-*.log`, `/tmp/dvfs_*.csv` (200 Hz MHz probes),
  `/tmp/park_stress{,.cpp}` (production-path park stress).

## 8. Recommendation

**Ship.** The hybrid spin-2048 µs + futex park is the new default
(`STRATA_POOL_PARK` unset); the pure-spin legacy behavior is the one-knob fallback
(`STRATA_POOL_PARK=spin`), and every other threshold stays selectable for future
governor work. The idle diagnostic's 24.0-core burn becomes ~0 while decode is
statistically unchanged on both cards and all correctness gates pass. Stage 1.5 complete;
STOP (Stage 1.6 not started).
