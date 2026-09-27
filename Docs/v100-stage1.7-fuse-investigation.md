# V100 Stage 1.7 pre-work — Historical ~52 tok/s fuse configuration investigation

> Completed at the user's request **before** any Stage 1.7 kernel optimization
> (E4/E1/E2/E3 not started). Resume point remains `Docs/v100-stage1.7-state.md`.
> Verdict up front: **the ~52 tok/s number is the Stage 1.4 fused-pool top-of-distribution
> result (51.88, run `s14-h3-off2`); the fuse setting is still the default and still works,
> but 51.88 itself is not reproduced on the current build — best warm fused run today is
> 50.11, inside the documented ±1–2 tok/s MCE drift envelope.**

## 1. The exact historical fuse setting

**The setting:** the **fused single-phase CPU expert pool** (pool mode 7) — Stage 1.4,
commit `5ed5616` ("1.4: fused single-phase CPU expert pool (default) + pool phase
diagnostics"). It is an engine default, not a CLI flag: fused is active whenever
`STRATA_POOL_UNFUSE` is **unset** (`STRATA_POOL_FUSE=1` is a no-op alias kept for A/B
scripts; `src/kernels/cpu/pool.cpp` `run_split_multi_native`, `dispatch_multi_native_async`).
`STRATA_POOL_UNFUSE=1` restores the pre-1.4 two-phase path (gu barrier → host-serial
per-expert quant → down barrier) and forces the synchronous dispatch (no async rows).

**The ~52 tok/s measurement:** `Logs/benchmarks/s14-h3-off2.{json,log}` —
**51.88 tok/s** (256 tok in 4,934.9 ms), GPU0 V100-32 GB, 2026-09-27. Exact launch
(`bench/v100/bench.py s14-h3-off2 --gpu 0 --workers 24 --max-new 256 --stats`, which
expands to):

```
ulimit -l unlimited
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
./build-sm70/strata --pack packs/swift-iq3_xxs \
  --native <Swift-...-IQ3_XXS-00001-of-00002.gguf> \
  --ple-gguf <same shard> \
  --expert-profile data/expert-profile.bin \
  --expert-cache auto --prefill 2048 --spec 4 --spec-min-p 0.5 \
  --pool-workers 24 --mtp mtp/rt --max-context 8192 \
  --kv fp16 --max-new 256 --tokens <canonical 28 tokens> --stats
```

(fused pool = default; no `STRATA_*` overrides; `--ple-io ram` and `--pcie-frac 0.2`
were the 1.2B/1.3 engine defaults; the build at the time was pre-1.5/1.6 — no park
override, no async rows, no WC flags, arena first-touch still non-deterministic.)

**Context of that number:** it is the **top of a 7-run 1.4-era distribution on the same
config**: 51.88 / 51.12 / 51.01 / 50.99 / 49.97 / 49.78 / 49.39 tok/s
(`s14-h3-off2`, `s14-h3-on1`, `s14-final0-2`, `s14-final0-1`, `s14-prof`, `s14-h3-on2`,
`s14-h3-off1`). The documented 1.4 final is 50.99/51.01. The fuse change itself is the
measured step **48.88/48.62 → 50.99/51.01 tok/s (+4.5–5.1 %)** vs the two-phase baseline
(pool wall −11.2 %, flag C 360.5 → 193.6 ms nsys, bit-exact end-to-end). No commit,
branch, env file, or benchmark script in the repo records a different "fuse" setting
reaching ~52; the full env-var surface is `STRATA_POOL_*` (UNFUSE/FUSE alias, TASKS,
PARK, ASYNC, CORES, *_DIAG), `STRATA_ARENA_NODE`, `STRATA_IQAVX2`, `STRATA_ALTFLAGC`,
`STRATA_WAIT_ITERS`/`SFENCE`/`HEAD`_DIAG, `STRATA_TG_FLUSH_US`, PLE/AVX2 knobs — and the
only one named "fuse" is the pool path above.

## 2. Reproduced on the current Stage 1.6/1.7 baseline

The fuse setting is **already the default on the current build** (HEAD `085cec9`,
`build-sm70/strata` of 2026-09-27 22:34, no Stage 1.7 code changes). Verified directly:
fused runs show `pool multi gate/up ≈11.1–12.1, quantize 0.000, down 0.000` (mode 7,
quantize folded into the gu phase) and "(rows async)"; the `STRATA_POOL_UNFUSE=1` arm
shows `gu ≈8.8–9.1, quantize ≈0.2, down ≈3.3` (two-phase) and sync dispatch.

**This session's 3×3 interleaved A/B** (`bench/v100/s17f_fuse_ab.sh`, GPU0, 256-token
decode, canonical tokens, `drop_caches` between arms, `strata.service` stopped,
`ulimit -l unlimited`, MCE bracketed 0→0 on EDAC while the kernel MCE log shows the
documented socket-0 corrected-MCE stream active throughout):

| run | arm | decode tok/s | pool ms/tok | verify-pool ms/round | golden 32 | notes |
|---|---|---:|---:|---:|:--:|---|
| s17f-base-1 | fused (default) | 48.19 | 13.75 | 13.79 | ✓ | cold first run (PLE preload 23.8 s) |
| s17f-unf-2 | `STRATA_POOL_UNFUSE=1` | 48.27 | 14.94 | 15.18 | ✓ | |
| s17f-base-3 | fused (default) | **50.02** | 12.80 | 12.83 | ✓ | |
| s17f-unf-4 | `STRATA_POOL_UNFUSE=1` | 49.43 | 14.27 | 14.42 | ✓ | |
| s17f-base-5 | fused (default) | **50.11** | 12.66 | 12.69 | ✓ | |
| s17f-unf-6 | `STRATA_POOL_UNFUSE=1` | 49.44 | 14.37 | 14.50 | ✓ | |
| — mean (fused / unfused) | | **49.44 / 49.05** | **13.06 / 14.53** | | 6/6 pass | warm pairs: fused +0.59 / +0.67 tok/s |

Correctness: **32/32 golden on all 6 runs**; **256/256 deterministic** — every fused run
byte-identical to every other (base-1=base-3=base-5), same for the unfused arm
(unf-2=unf-4=unf-6). Both arms are also **0/256-token-different from the historical
`s14-h3-off2` sequence** (and from each other): the current fused configuration is
numerically identical to the historical one; only wall time differs.

## 3. Comparison vs the ~50 tok/s baseline and the historical 51.88

- Current Stage 1.6/1.7 baseline (last session): 49.28 / 50.15 / 50.07 tok/s.
  This session's warm fused runs (50.02 / 50.11) **re-establish that baseline** — no
  regression from the checkpoint.
- **The 51.88 (~52) is not reproduced.** Best warm fused run: 50.11 (−1.77 vs 51.88).
  The closest historical results on the *same* configuration remain the 1.4-era
  51.88 (s14-h3-off2) and 51.01/50.99 (s14-final0-2/1); the closest result measured
  today is 50.11.
- **Why the gap (measured, not guessed):**
  1. **Pool wall drift, not a config difference.** Historical fused pool: 11.51 ms/tok
     (gu 9.60, run 9.61, plan 0.38). Current fused pool: 12.66–12.80 ms/tok
     (gu 11.13–11.49, run 10.67–10.73, plan 0.55–0.58). The +1.1–1.3 ms/tok pool
     slowdown (≈ −6 % on the 19.9 ms token wall ≈ −1.5–2 tok/s if fully additive; the
     round-head overlap makes the observed e2e delta ≈ 1.7 tok/s) explains essentially
     the whole gap. Every other component is unchanged or faster (wait-for-rings
     26.07→26.36, commit 0.843→0.855, MTP 1.92→1.93 ms/round).
  2. **Documented ±1–2 tok/s box drift.** The socket-0 corrected-MCE stream
     (bank 5 MC_CHA, ~1/s at idle, confirmed active in `dmesg` during today's runs;
     Stage 1.6 §1 "MCE storm") plus unpinned pool-worker scheduling (worker core sets
     drift run-to-run in the `.threads.csv` samplers even *within* the 1.4-era runs)
     produces the run-to-run spread seen in every stage: the 1.4-era 7-run spread on
     this very config was 49.39–51.88 (a 2.5 tok/s band). 51.88 is the top of that
     band; today's band is 48.19–50.11 with the cold-start run at the bottom (the
     first run after idle is consistently ~1.5 tok/s slower across stages —
     s17-baseline-1 49.28 vs 50.07/50.15; 1.6 baseline 48.58/48.82 vs 50.0–50.1).
  3. **The fuse effect itself is intact and still positive.** Fused vs unfused on the
     current build: pool wall −1.3 to −2.2 ms/tok (−11 to −14 %, matching the original
     −11.2 % measurement), e2e +0.59/+0.67 tok/s in the two warm pairs (−0.08 in the
     cold pair, within noise). Bit-exact (both arms 0/256-different vs the historical
     sequence). So the fuse configuration is worth keeping — it is the default.

## 4. Exact command/configuration (reproducible)

- **Current fused baseline (the historical ~52 config, on the current build):**
  `python3 bench/v100/bench.py <label> --gpu 0 --workers 24 --max-new 256 --stats`
  with no `STRATA_*` env — i.e. `./build-sm70/strata <canonical flags from §1>`,
  pool mode 7 single-phase (default), async rows (default), park 2048 µs (default),
  arena pinned node0 (default), `--ple-io ram` + `--pcie-frac 0.2` (engine defaults).
- **Two-phase A/B arm:** same + `--env STRATA_POOL_UNFUSE=1`.
- **Driver script:** `bench/v100/s17f_fuse_ab.sh` (3×3 interleaved, drop_caches between
  arms). Raw data: `Logs/benchmarks/s17f-{base,unf}-{1,2,3,4,5,6}.{json,log}`,
  `Logs/gpu/s17f-*.csv`, `Logs/cpu/s17f-*.{csv,threads.csv}`.

## 5. Verdict and disposition

1. **Identified:** exact historical fuse setting = Stage 1.4 fused single-phase CPU
   expert pool, commit `5ed5616`, default via `STRATA_POOL_UNFUSE` unset; the ~52
   figure is run `s14-h3-off2` = 51.88 tok/s (top of the 49.39–51.88 same-config band).
2. **Reproduced:** the setting is the current default and was re-benchmarked on the
   Stage 1.6 baseline (3 fused / 3 unfused, interleaved, GPU0, 256 tok).
3. **~52 not reproduced** (best 50.11): the gap is the current pool running
   +1.1–1.3 ms/tok slower than the 1.4-era best run (12.66 vs 11.51 ms/tok), within
   the documented MCE/scheduling drift — not a lost or changed fuse configuration
   (0/256-token numerical difference proves the math is identical).
4. **Keep the configuration** (it is the default; it still buys −11 to −14 % pool wall
   and +0.6 tok/s e2e vs unfused, bit-exact). Documented cause of the historical step:
   fusing the two worker barriers + host-serial per-expert quantization into one
   phase (mode 7) removed one barrier wake/re-park and the host-serial quant per
   dispatch (Stage 1.4 root-cause analysis, `Docs/v100-stage1.4-final.md` §2–3).
5. **Do not start E4/E1/E2/E3** (per instruction). For the record, the profile
   (`v100-stage1.7-state.md` §4–7) still ranks the GPU kernel work (E1 gr_down split-K,
   E2, E3) and the round-head wait-C lag (the remaining 1.6 bottleneck, ~13 ms/round)
   as the next levers — neither is a fuse matter.
