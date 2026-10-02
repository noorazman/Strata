# V100 Stage 3.0 — upstream sync to 0.1.35 (Release 3 branch)

## What happened

The V100 fork had drifted from upstream since the merge base `1ee8b666` (tag era 0.1.x,
before the QSA indexer rework). Release 3 starts by rebasing the branch onto upstream
`main` tip `d9ab8435f` (tag `v0.1.35`).

- Branch: `release3-qwen35-dense`, cut from `stage1.3-expert-pool-sync` (`e4f93b1`).
- Merge commit: `9a5cbee719bab3e05f904890f88939bea8276322` (two parents: `e4f93b1`, `d9ab8435f`).
- Build fixes after the merge: `0aab9b3`.
- 471 files changed by the merge (+105,944 / −3,254 lines), 195 of them under
  `src/`, `include/`, `CMakeLists.txt`, `serve/`.

Policy: **upstream wins**. Our changes survive only where they support the V100/SM70
adaptation, or where auto-merged call-sites reference them. Everything else is upstream
0.1.35 as-is.

Push note: `origin` (github.com/noorazman/Strata, free account) rejects files over
100 MB (GH001), and the Stage 1.12–1.17 commits carry ten Nsight profiling databases
(`Logs/gpu/s1*nsys-*.sqlite`, 259–533 MB each) in the tree. The branch was therefore
pushed with those ten blobs filtered out (`git filter-repo`, no other change):
origin's tip is `34eb1ba…` vs the local full-history tip, and the tag `v100-release1`
points at the rewritten freeze commit `c3e70e0…` (same message, tree minus the blobs).
The local working clone keeps the full history; the sqlite files are raw profiling
data, referenced by the stage docs, not by the code.

## What 0.1.35 brings

HIP backend (AMD, bundled runtime on Windows), MMQ mixed-precision MoE prefill
(`STRATA_PREFILL_MMQ`), `qsa_attn_pools` + the tensor-core prompt attention
(`qsa_prompt_attn`, CUDA-only, refuses below sm_75), profiling stamps
(`gpu_stamp`, per-stage profiler in the verify window), E-6 device plan
(`resident_plan`, `wait_flag_ge_or`, `copy_or_zero_from_mapped`), PLE FP8 + Q5_0,
MTP E-4/E-9 batched prefill + prompt K/V API, REMOTE experts (`ArenaExpertSource`,
`RemoteExperts` preflight), pool watchdog / batch-claim, `detect_cpu_topology` /
`PoolAffinity` (#272), `STRATA_FSEEK64`, the prefill restructure (stager, `RING_MAX`,
`ring_slots`, mapped group copy `STRATA_GROUP_COPY`, `rebind`), `note(cublasStatus_t,
const char*)`, hipBLASLt (`try_hipblaslt`, HIP-only), `absorb_hipblas_sticky`,
`--p2p` / `--list-devices` wiring on the new device-plan foundation, the 512-expert
draft geometry for pruned targets, and the secure-MTP-CUDA0-allocation ordering before
the host arena registration.

## Per-file conflict resolutions

The five named V100 files, resolved strictly preserving our adaptation:

- `CMakeLists.txt` — arch floor 70 (upstream wants 75+), `enable_testing()`,
  `CUDA::cublasLt` link. Our floor stays: the V100 is the target card.
- `src/kernels/cuda/native_qsa_score.cu` — our SM70 warp-FMA TF32 fallback
  (`__CUDA_ARCH__ < 800`) kept inside the upstream kernel.
- `src/core/device.cu` + `include/strata/core/device.hpp` — runtime cc floor 7.0
  (`kMinCc = 70`), `DevicePlan`, P2P enable at plan time, `--devices` parsing/report.
- `src/kernels/cpu/pool.cpp` — our dangling-else fix, `STRATA_POOL_PARK` hybrid park,
  fused pool mode 7, `STRATA_POOL_TASKS`/`STRATA_POOL_CORES`; upstream's watchdog and
  batch-claim merged around them.
- `src/program/generate.cpp` — CLI: `--devices`, `--ple-io ram` default (pinned by the
  `ple_default_mode` ctest), `--pcie-frac 0.2` (V100 sweep: 0.55 → 43.5, 0.35 → 46.3,
  0.25 → 43.9, 0.2 → 49.5, 0.0 → 47.3 tok/s), `--pool-workers 24`. Plus the Stage 1.10
  TTFT stage timestamps (engine side) and the 9-arg `PoolMultiFn` (`rows_async`) the
  verify window needs.

Other resolutions worth recording:

- `src/prefill/prefill.cpp` — **took upstream verbatim** (1,977 lines vs our 803: the
  stager/ring restructure) and grafted back exactly one thing: the `STRATA_MOE_DQ_WIDE`
  wide-dequant dispatch (the only promoted production env var, per the Release 1 frozen
  config). Our opt-in prefill experiments (`STRATA_MOE_ASYNC_IDS`, `STRATA_TWO_STREAM_DMA`,
  `STRATA_MOE_DEQUANT_GEMM_FUSE`, `STRATA_MOE_GEMM_GROUPED` — Stage 1.17 verdict: wash —
  `STRATA_MOE_ROUTE_DUMP`) are dropped from prefill; the gemm APIs they used survive in
  `gemm.cu`.
- `src/prefill/gemm.cu` — upstream `rebind`/workspace handling + our dispatch chain:
  hipblaslt-try (HIP-only) → tensor-core (`STRATA_MOE_GEMM_TC*`) → cublasLt
  (`STRATA_MOE_GEMM_LT`) → plain `cublasGemmEx`.
- `src/kernels/cuda/iq_kernels.cu` — both sides: our wide dequant kernels
  (`iq_dequant_gu_f16_wide`, `iq_dequant_f16_wide`, `iq_wide_supported`) + upstream's
  `is_iq` superset, `gu_qk`/`d_qk`, `native_expert_supported`, `STRATA_GU_FMTS`.
- `src/core/verify.cpp` + `include/strata/core/verify.hpp` — upstream window with our
  wait-iteration probe (`STRATA_WAIT_ITERS`), flagC2 A/B (`STRATA_ALTFLAGC`),
  `STRATA_WAIT_FENCE`, the E1/E2 GR split hooks and the `PoolMultiFn` 9-arg call.
- `include/strata/kernels/verify_kernels.hpp/.cu` — one `wait_flag_ge` declaration
  (4-arg, `iters` optional); both kernel variants kept in the .cu. The 3-arg upstream
  call-sites bind to the 4-arg declaration with `iters = nullptr`.
- `include/strata/kernels/fused_gr.hpp` + `src/kernels/cuda/fused_gr.cu` — upstream's
  6-arg stamped read (with the template `<int TILEV>` down kernel, 1280 on sm_75 /
  2560 elsewhere) + our 5-arg E1/E2 overloads (`STRATA_GR_DOWN_SPLIT`,
  `STRATA_GR_NORM_SPLIT`), instantiated at `<TILE>` = 2560.
- `src/core/mtp.cpp` + `include/strata/core/mtp.hpp` — upstream MTP with the prompt K/V
  API; our Stage 1.11 E3 chunk-graph overlap (`STRATA_MTP_OVERLAP`, opt-in, no-op
  otherwise) and the batched E-4 prefill.
- `src/kernels/ngram.cpp` — upstream PLE reader (mmap/mlock) + our RAM preload
  (`--ple-io ram`, `PLE_ROW_BYTES` 90 / FP8 160 / Q5_0 110).
- `src/kernels/cpu/iq_avx2.hpp/.cpp` — upstream's `iq256_*` superset; the two parity/
  debug tools renamed to follow.
- `serve/server.py` — upstream 0.1.35 handler (path normalization for issue #55,
  `/settings` `/unload` `/load`, structured outputs, `_capture`, `EngineDied`
  handling) + the Stage 1.10 TTFT stage timestamps (`t0`, `t_json`, `t_prepare`,
  `t_gen_write`, `t_first_engine_tok`, `t_first_sse`, `t_done`), opt-in, off by
  default.

## Build and test

Build (V100, SM70 only):

```
.venv/bin/cmake -S . -B build-sm70 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DFETCHCONTENT_BASE_DIR=/home/noorazman/dsh/strata/fetch
.venv/bin/cmake --build build-sm70 -j 40
```

`cuobjdump --list-elf build-sm70/strata`: 55/55 ELF sections `sm_70` (no other arch).

Test, `CUDA_VISIBLE_DEVICES=0 ctest -R "parity|selftest|hostref"` on this box:
30 of 31 pass (two fixture tests skipped, `iq_parity_fixtures`/`iq_parity` — their
fixtures are not checked out). The one failure is `ple_parity`, environmental: it wants
`../../Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf`, which is not on this
machine. `kv_hybrid_parity` needed the one test-side tweak: the tensor-core prompt
path refuses below sm_75 by design (`qsa_prompt_attn.cu`), so on the V100 the refusal
is the expected old-kernel fallback and now counts as a pass, matching the HIP branch.

The build-fix commit `0aab9b3` records the merge residue found by the first build:
two lost closing braces in the `verify_kernels.cu` keep-both splice, the E1 down-split
call sites needing the `<TILE>` template argument (upstream made `gr_down_multi_kernel`
a template), a duplicate `is_iq` and a lost brace in `iq_kernels.cu`, and the
`iqavx2_*` → `iq256_*` rename in the two test tools.

## Benchmark protocol (Release 3)

Frozen Release 1 config on the merged binary, GPU 0:

- `strata serve` / `generate` with `--spec 4 --spec-min-p 0.5 --pool-workers 24
  --mtp mtp/rt --ple-io ram --expert-cache auto`, 16K and 32K context, int8 KV,
  `STRATA_MOE_DQ_WIDE=1` (fork only). "auto" resolved to the 8,000-slot cache in the
  R1-era engine and to 15,764 slots in the 0.1.35 engine (see below) — the frozen
  config is the flag, not a slot count.
- Decode: tok/s at 16K and 32K prompt (served), acceptance and TTFT from the engine
  (`STRATA_TTFT=1` JSON lines) and the server's TTFT report.
- Anchors (Release 1 / Release 2, this box): decode 48–58 tok/s class
  (51.09/54.27 @16K, 49.51/52.0 @32K served); TTFT 37.98 s @16K, 71.50 s @32K
  (anchor band 37.7–40.7 / 71.4–76.8); VRAM peak ~18.9–19.2 GiB; service boot
  ~95–105 s.
- Comparison: (1) this merged `release3-qwen35-dense` build, (2) `Strata-Pure` —
  upstream `Niko1221/Strata` `main` as-is, pushed private to
  `noorazman/Strata-Pure`, built for SM70 with the upstream floor exception
  (`-DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_EXPERIMENTAL_SM60=ON`; upstream floor is 75
  and the flag permits 70, runtime floor 60; the upstream QSA scorer already has an
  fp32 FMA fallback below sm_80), (3) the previous V100 anchors above.
- `Strata-Pure` is a clone of upstream `main` at `d9ab8435f` (tag `v0.1.35`) with zero
  source changes — only the SM70 build-flags exception, the model pack (`packs/`), the
  MTP head (`mtp/rt/`) and the expert profile (identical md5 on both sides) copied in.

## Benchmark results (Release 3)

All numbers: GPU 0 (V100 32 GB PCIe), this machine, 2026-10-03. Builds: `r3` = the
merged `release3-qwen35-dense` binary with `STRATA_MOE_DQ_WIDE=1`; `pure` =
`Strata-Pure` (upstream 0.1.35 as-is, no V100 env vars); `R1` = the Release 1 anchors
(`Docs/v100-release1.md`), the last pre-merge V100 baseline.

### What 0.1.35 changed for the V100 port

- **Expert cache auto-sizing moved.** Upstream 0.1.35 reserves less VRAM before sizing
  the cache, so `--expert-cache auto` now yields 15,764 slots / 25.63 GiB at both 16K
  and 32K context (R1 era: 8,000 slots / 12.93 GiB). The frozen production config is
  unchanged; "auto" means more. Consequence: VRAM peaks go from ~19 GiB to ~32.1 GiB
  (headroom ~700 MiB on the 32 GB card), and part of the decode speedup below is the
  bigger cache: the 0.1.35 runs show 98 % expert-cache hit on the fill prompt (det
  gates ~96 %), vs 79.9–84.3 % measured on the 8,000–14,000-slot R1-era cache
  (`Docs/v100-release1.md` L245, `Docs/v100-stage1.13-final.md`).
- **Cross-request prompt cache (new, upstream).** 0.1.35 reports
  `usage.prompt_tokens_details.cached_tokens`: a second request with the same prefix
  reuses the first one's prompt KV. The serve benchmark's warm run (r2, same prompt,
  same server) shows 16,115/16,120 tokens cached at 16K and 32,499/32,504 at 32K —
  warm TTFT drops to 3–4 s. The R1 protocol ran the same two requests and saw no
  reuse (37.98 s → 37.08 s), so this is a 0.1.35 capability, not a V100 effect.
- **Numerical trajectory vs the R1 reference.** At 16K with `max-new 256`, the merged
  build reproduces the R1 golden-32 prefix (32/32 MATCH, det1 and det2) — continuity
  with the production anchors. At 32K, the merged build and `pure` agree
  byte-for-byte (det1 md5 `788f1ab8…` both sides), and both diverge from the R1
  32K trajectory at token 7 (`7967` vs the R1 golden's `30869`): the 0.1.35 prefill
  restructure changes the 32K numbers, and it does so identically with or without the
  V100 adaptation — the merge is numerically faithful. Cross-engine non-determinism
  (independent det runs diverging after ~token 32–43 at 16K, ~token 43 at 32K) exists
  in `pure` too (det2 diverges at token 43, det3 at token 181 at 32K; the two 16K
  runs differ from each other as well), so it is an upstream timing property of the
  MTP draft path in 0.1.35, not a merge regression. One V100-visible quirk: at 16K the
  `max-new 32` run (g32) takes the `7967` trajectory while the `max-new 256` runs take
  the R1 trajectory — the trajectory depends on `max-new` in this build, and `pure`
  shows the same g32/det split.

### Serve-mode fix found by the benchmarks (commit `b77303e`)

The first ctx run against the merged serve died in the overhead probe: every request
ended `0 generated … 1 checkpoints (cancelled)` right after prefill. Upstream 0.1.35
yields prompt-progress/heartbeat chunks as `None` (the SSE `: keep-alive` comments),
and the V100 TTFT instrumentation grafted in at Stage 1.10 (`if tt:` blocks in the
OpenAI/Anthropic SSE loops, reading `c["choices"][0]["delta"]` on every chunk, with
`name` unbound on the first Anthropic heartbeat) assumed every chunk carried
`choices`. Any request that emitted a `PP` line crashed the handler (TypeError /
NameError), the stream's `finally` sent `STOP`, and the engine cancelled the request
after prefill. Fix: guard both `tt` blocks against `None`/unbound `name`
(`b77303e`). The pre-merge fork never met `None` chunks (its engine yielded no
heartbeats), which is why it only surfaced after the merge. `pure` has no TTFT block
and was unaffected.

### Engine gates (det, max-new 256, spec 4, int8 KV; `r3` with `STRATA_MOE_DQ_WIDE=1`)

Decode tok/s, fresh engine per run (this box, V100):

| build   | 16K det1/det2/det3 | 32K det1/det2/det3 | 16K golden-32 | VRAM peak |
|---------|--------------------|--------------------|---------------|-----------|
| R1      | 48.36 / 48.96 / 48.50 | 49.44 / 46.34 / 47.44 | 32/32 MATCH, md5 `cdb7f7d0…` | ~18,926 / 19,178 MiB |
| r3      | 64.97 / 52.2 / 66.8 | 68.6 / 62.62 / 59.28 | det1+det2 32/32 MATCH (R1 trajectory); g32 diverges at token 7 | 32,084 / 32,086 MiB |
| pure    | 62.51 / 63.16 / 63.98 | 62.84 / 63.87 / 62.79 | its own 0.1.35 trajectory (g32 == det1) | ~31.9 GiB class |

- Decode is ~+25–40 % over R1 in both new builds; the split of that gain between the
  0.1.35 decode path and the 15,764-slot cache is not separable with the frozen config
  (the cache size is what "auto" now means). Run-to-run spread at 16K in `r3`
  (52–67) is trajectory-dependent (divergent det runs route to different experts);
  `pure` is tighter (62.5–64).
- Spec acceptance, det gates: `r3` 0.62–0.69 (16K 0.624–0.684, 32K 0.637–0.691) vs
  `pure` 0.56–0.63 (16K 0.583–0.595, 32K 0.56–0.631) — the merged build accepts more
  drafts on the same prompts (its 16K trajectory is the R1 one, where the draft head
  was tuned). Expert-cache hit rate is ~0.96 in all det runs (both builds).
- In the serve runs the acceptance follows the trajectory: `r3` 143/181 (16K r1) and
  114/174 (32K r1) → 181/197 (32K r2); `pure` 118/161 (16K r1) and 145/208 (32K r1) →
  179/207 (32K r2).

### Serve (s110 protocol: thinking on, fill prompt, max-new 256, GPU 0)

Client = `s110_client.py` (TTFT = time to first reasoning/content token; decode =
steady inter-arrival over the last 70 % of arrivals). Engine = the engine's own
`strata serve:` summary line for the same request.

| metric | r3 16K r1 | r3 32K r1 | pure 16K r1 | pure 32K r1 | R1 16K r1 | R1 32K r1 |
|---|---|---|---|---|---|---|
| client TTFT (s) | **25.83** | **49.51** | **27.62** | **48.94** | 37.98 | 71.50 |
| engine prefill (tok/s) | 738.3 (21.8 s) | 731.0 (44.5 s) | 733.5 (22.0 s) | 728.7 (44.6 s) | — | 492.4 (66.0 s) |
| engine decode r1 (tok/s) | 66.6 (drafts 143/181) | 53.6 (114/174) | 46.6 (118/161) | 62.5 (145/208) | — | 49.51 class |
| client decode r1 (tok/s) | 65.27 | 51.17 | 42.84 | 63.22 | 51.09 | 49.51 |
| warm r2 TTFT (s) | 4.44 (16,115/16,120 cached) | 3.39 (32,499/32,504 cached) | 4.34 (16,115 cached) | 3.49 (32,499 cached) | 37.08 (no reuse) | 70.72 (no reuse) |
| warm r2 client decode (tok/s) | 55.65 | 86.73 | 59.09 | 82.25 | 54.27 | 52.0 |
| VRAM peak (MiB) | 32,094 | 32,116 | 32,106 | 32,108 | 18,926 | 19,178 |
| r1 fill text md5 | `d6733778…` | `bba460ce…` | `a02c7610…` | `f6e4da33…` | `6041c5f3…` | `1fbe577e…` |
| boot (engine start → ready) | ~95 s (PLE RAM preload) | ~95 s | ~95 s | ~95 s | ~95–105 s | ~95–105 s |

Reading:

- **Prefill throughput is identical across builds** (731–738 tok/s at both contexts,
  `r3` and `pure` within ~1 % of each other): the 0.1.35 prefill restructure (stager,
  ring slots, mapped group copy) works on the V100 without the V100 adaptation, and
  the adaptation's `STRATA_MOE_DQ_WIDE` does not move prefill at the 0.1.35 pipeline
  (it was tuned for the pre-merge prefill). Both are ~+49 % over the R1 anchor
  (492.4 tok/s @32K).
- **TTFT follows prefill**: 25.83 s / 49.51 s at 16K / 32K (`r3`), −32 % / −31 % vs
  R1 (37.98 / 71.50), both inside/below the old noise bands (37.7–40.7 / 71.4–76.8).
- **Decode splits by trajectory**: on the cold r1 run `r3` wins at 16K (66.6 vs 46.6
  tok/s) and `pure` wins at 32K (62.5 vs 53.6 — its r1 trajectory gets 145/208 draft
  acceptance vs 114/174); the warm r2 is 59.09/82.25 (`pure`) vs 55.65/86.73 (`r3`).
  These are different text trajectories with different expert routing, not different
  engine speeds — the det gates above are the comparable decode numbers.
- **Config hygiene**: the `pure` base config initially carried the production vision
  block (pointing at the fork's `strata-vision` encoder); the first `pure` ctx run had
  the encoder resident on the GPU and its expert cache auto-sized to 15,053 experts /
  24.49 GiB instead of ~15,600–15,800 / ~25.4–25.6. The vision block was removed from
  the `pure` base config and the ctx runs re-done; the table above is the text-only
  rerun (its expert cache: 15,634 experts / 25.39 GiB). The `r3` ctx configs were
  always text-only, so the two builds were run under the same conditions.
- **The r1 fill md5s no longer match the R1 anchors** at either context (the 0.1.35
  trajectory change, as above); at 16K the det-gate runs still reproduce the R1
  golden prefix. The serve-mode r2 (warm) md5s differ from r1's, the documented
  warm-trajectory divergence, present in all three builds.
