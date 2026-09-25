# V100 Stage 1.2B Final — `--ple-io ram` is now the default

Branch `stage1.2b-ple-ram-default` (from `stage1.2a-ple-ram` @ stage1.2a-ple-ram-pass).
Narrow scope: flip the program default from `direct` to `ram`, keep all three
modes selectable, make the low-RAM failure explicit. No engine, quantization,
expert-cache, pool, spec, KV or dense-model changes; no new performance work.

## What changed

1. **The default** — `Options::ple_io` in `src/program/generate.cpp` is now
   `"ram"`. That is the single canonical program default; the kernel API
   (`PleIoOptions::mode`) keeps its own low-RAM default (`Direct`) for library
   callers, and the help text states the default explicitly. A ctest
   (`ple_default_mode`) pins the program default by asserting the help text:
   `ram (default)`, and that `direct` and `mmap` remain documented and
   selectable. Explicit `--ple-io ram|direct|mmap` (and the `--ple-ram` alias)
   always override the default; nothing auto-switches modes based on detected
   RAM.
2. **The log says so** — startup now prints
   `strata generate: PLE on, table N rows of SHARD (PLE I/O mode: ram)`, so the
   active mode is unambiguous in every log.
3. **Low-RAM guard, no silent fallback** — before the preload, `PleTable::open`
   (Ram branch) requires total system RAM (from `/proc/meminfo` MemTotal) to be
   at least `table + ram_rest_bytes`, where the program supplies
   `ram_rest_bytes = 42 GiB` (expert arena ~40 GiB measured 39.97 for the
   current model + 2 GiB headroom; Stage 1.2A measured 67.5 GiB total peak RSS
   with the table resident). On a short machine it fails BEFORE the preload
   with an actionable error that names the fallback:

   ```
   PLE ram mode needs 68.82 GiB of system RAM (PLE table 26.82 GiB + the rest of
   the engine 42.00 GiB), but this system has 62.00 GiB total; use --ple-io
   direct on lower-RAM systems
   ```

   The guard is a hard failure, never a mode switch. If the total fits but the
   anonymous allocation itself fails, the mmap error also names `--ple-io
   direct`. On non-Linux systems (no `/proc/meminfo`) the check is skipped and
   the mmap remains the last line of defense. `ple_reader_test --gguf SHARD
   --ram --ram-rest-gb N` exercises the guard end-to-end against the real 26.82
   GiB table without consuming RAM.

## Required behavior — all four paths validated (GPU0, workers 24, Stage 1.1 deterministic workload)

| run | mode (resolved, from the `PLE I/O mode:` log line) | result |
|---|---|---|
| no `--ple-io` | ram (default) | `PLE I/O mode: ram` in the log; 26.82 GiB preloaded and resident; zero PLE NVMe reads at inference (device-level: run delta = arena + preload only); prefill prompt tokens identical to 1.2A |
| `--ple-io ram` | ram | decode token-for-token identical to the default run (32/32 golden both) |
| `--ple-io direct` | direct | startup OK; PLE SSD reads occur as before (4,488 reads / 18.8 MB for the 256-token decode, p50 3.7 ms); 32/32 golden; tokens identical to the RAM runs; no preload (NVMe delta = arena only) |
| `--ple-io mmap` | mmap | functional (existing path, unmodified); 32/32 golden; tokens identical |

Selftest (`ple_reader_selftest`) green; `ple_default_mode` green.

## Performance (default = RAM, so this is the already-validated 1.2A behavior re-confirmed)

Measured this stage, GPU0, workers 24, deterministic workload:

- prefill (2,047-token prompt, default mode): 5,036.4 ms (406.2 tok/s engine
  chunk; 392.8 tok/s end-to-end; TTFT 6.15 s) — in the 1.2A RAM range
  (4,471–4,870 ms; the gap is the same machine drift that moved the 1.2A SSD
  arm too);
- decode 256 tokens: 42.19 tok/s (default run) / 44.04 tok/s (explicit ram) —
  the 1.2A RAM range was 42.2–44.7;
- peak RSS: **67.55 GiB** (1 Hz VmRSS sampling of the default run) — matches
  the 1.2A measurement;
- peak VRAM: 18,852/18,896 MiB — unchanged;
- PLE NVMe reads at inference: **0** (served 33,008 / 5,504 rows from RAM;
  device counters confirm the table only touches NVMe during the preload).

(The purpose of this stage was to prove the default flip does not alter the
validated RAM-mode behavior - it does not; raw runs under Logs/.)

## Documentation state

`ram` is now the default for this 128 GB machine (and, by default, for any
machine large enough to pass the guard). For the currently benchmarked model
the PLE table is ~26.82 GiB - that figure applies to the current
Swift-1.5 table, not to every future model. `direct` remains the lower-RAM
fallback; `mmap` remains available as an alternative/testing mode. RAM mode
removes PLE NVMe reads during inference and is the recommended configuration
where memory allows.

## Issues discovered

- The usage text is printed to stderr (existing behavior); the new ctest
  accounts for that.
- The reconfigured build uses `~/.local/bin/cmake` (4.4.2); the system
  `/usr/bin/cmake` is 3.22.1 and rejects the project's `cmake_minimum_required
  (3.24)` - a pre-existing environment fact, now noted so the next session does
  not trip on it.
