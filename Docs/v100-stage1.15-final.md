# V100 Stage 1.15 — MoE Prefill GEMM: cuBLASLt Routing (STRATA_MOE_GEMM_LT)

Stage 1.15 from the clean Stage 1.14 commit `38bd20d` (branch `stage1.3-expert-pool-sync`,
GPU0, V100 32 GB PCIe). The optimization is opt-in (`STRATA_MOE_GEMM_LT=1`, default OFF); the
production default and the Stage 1.14 wide-dequant (`STRATA_MOE_DQ_WIDE`) are untouched. The A/B
baseline for this stage is **wide-ON**: leg A = `STRATA_MOE_DQ_WIDE=1` + LT OFF; leg B =
`STRATA_MOE_DQ_WIDE=1` + LT ON.

**Mission:** find the next highest-impact optimization for MoE PREFILL wall-clock latency.
PROFILE FIRST. Pick ONE isolated change, implement it OFF-by-default, and run the full gate
(correctness, determinism, 16 K / 32 K performance, nsys A/B). If it is not a meaningful
improvement, stop and document it — do not force a win.

## 1. Baseline (mission step 1) — profile first, no assumption

Reused the Stage 1.14 wide-ON nsys captures (`s114nsys-{16k,32k}-on.sqlite`, build `38bd20d`) as
the experimental baseline (wide ON = the current best-known prefill config). The 16 K GEMM family
breakdown (mission step 2, "break down the remaining GEMM time"):

| 16 K GEMM family (wide ON) | time | notes |
|---|---|---|
| MoE expert f16 GEMM (GU + D) | ≈ 6.45 s | the target |
| — GU (gate+up) GEMMs | ≈ 4.3 s | 133,176 runs; W = 6.55 MB f16 |
| — D (down) GEMMs | ≈ 2.15 s | W = 3.28 MB f16 |
| dense bf16 magma | 4.45 s | normal-T GEMMs, cuBLAS is fine |
| GEMV / mmvf | 1.7 s | decode-side, out of scope |

Mission step 3 ("is attention/other a larger actionable bottleneck?"): attention ≈ 3.4 s
(QSA, scope-guarded), gdn 1.7 s, dequant 4.82 s (already at the 1.14 roofline),
`wait_flag_ge` 0.99 s. The GU expert GEMM is the largest *actionable* MoE component.

## 2. Bottleneck (mission steps 2–3) — the skinny GU GEMM

The GU GEMM shape is `Y[ne,1280] = X[ne,2560] . W[1280,2560]^T` (f16 in, f32 out), called
133,176× at 16 K. `ne` distribution (`Logs/gpu/s113route-16k.txt`, 300,087 pairs): mean 53.5,
median 17, p90 138, p99 488, max 2043. The dominant ne (1–64) is **memory-bound on the 6.55 MB
f16 W read**; the heavy tail (ne>64) is compute-bound where cuBLAS is already good.

ncu on the GU GEMM at ne=8 (microbench): the main kernel
`cutlass_70_tensorop_s884gemm_f16_64x64_tn_align8` runs at **40.18 % of DRAM peak** (7.93 MB,
24.24 µs; roofline 13.8 µs), 15.5 % warp occupancy, plus a `splitKreduce_kernel` (7.63 µs,
≈24 % of the GEMM time). L2 persistence is unavailable on this V100
(`cudaDevAttrMaxPersistingL2CacheSize` = 0). So the GU GEMM has headroom but is a hard target.

## 3. Candidate levers measured (mission step 4)

| lever | measured win (microbench) | verdict |
|---|---|---|
| Custom f32-FMA kernel (v1/v2/v3, `bench/v100/s115_gemm_custom.cu`) | loses at ne≥4 (X re-read / smem traffic is the bottleneck, not the W read) | rejected |
| **cuBLASLt first-heuristic algo** (`/tmp/cublaslt_test.cu`) | GU 2–9 %, D 4–13 % faster across ne=1..512, no per-ne search | **selected** (cheapest with a measured win) |
| L2 persistence for the f16 W | unavailable (maxp=0) | rejected |
| CUBLAS_COMPUTE_32F_FAST_16F | 1.00× | rejected |

The isolated-kernel microbench (`/tmp/cublaslt_test`) showed cuBLASLt's first heuristic algo
beating `cublasGemmEx(CUBLAS_GEMM_DEFAULT)` on the exact GU and D shapes:

```
GU (N=1280,K=2560):  ne=1..512  lt/def = 0.913, 0.923, 0.923, 0.925, 0.929, 0.946, 0.951, 0.957, 0.970, 0.979
D  (N=2560,K=640):   ne=1..512  lt/def = 0.868, 0.875, 0.875, 0.875, 0.882, 0.941, 0.962, 0.913, 0.966, 0.955
```

## 4. Isolated change (mission step 5) — OFF-by-default `STRATA_MOE_GEMM_LT`

`src/prefill/gemm.cu` + `include/strata/prefill/gemm.hpp`: `Gemm::f16` (the MoE expert GU+D
path) optionally routes through `cublasLtMatmul` with the first heuristic algorithm instead of
`cublasGemmEx(CUBLAS_GEMM_DEFAULT)`. The per-shape Lt objects (op / matrix layouts / heuristic)
are cached in a static `LtCache` keyed by (N, K, T) so the hot loop only issues the GEMM. A
sticky `lt_fail_` falls back to `cublasGemmEx` on any Lt error. `CMakeLists.txt` links
`CUDA::cublasLt` to `strata_prefill`. Default OFF (the env var is read at `Gemm::init`).

## 5. Correctness + determinism (mission step 6)

- 32-tok golden (baseline leg, wide ON / LT OFF): **MATCH** (`s18check --prefix`, GOLDEN_32).
- 32-tok self-consistency: baseline det1==det2 byte-identical; LT det1==det2 byte-identical.
- **LT-ON is byte-identical to the baseline** at 32 tok and at the 16 K / 32 K fill-prompt
  r1 (r1 md5 `6041c5f3…` at 16 K, `1fbe577e…` at 32 K — both match the Stage 1.14 anchors).
  The first-heuristic Lt algo resolves to bit-identical numerics for these shapes, so the
  determinism gate is trivially satisfied on both legs.

## 6. A/B (mission step 7) — wall-clock TTFT (r1 cold anchor; `s110.sh ctx`, wide ON both legs)

| context | A (LT OFF) r1 | B (LT ON) r1 | delta |
|---|---|---|---|
| 16 K | 38.14 s | 38.16 s / 38.58 s (cached) | **wash** (≈ +0.0 … +1.2 %) |
| 32 K run 1 | 72.01 s | 70.96 s | −1.06 s (−1.47 %) |
| 32 K run 2 | 71.80 s | 72.11 s | +0.31 s (+0.43 %) |

The baseline legs match the Stage 1.14 wide-ON anchors (16 K 37.72 s, 32 K 71.36 s). The 16 K
A/B is a wash. The 32 K A/B **flips direction between runs** (run 1: B 1.47 % faster; run 2:
B 0.43 % slower) — a wash at the edge of the r1-TTFT noise band (≈1–1.5 %). Averages: A 71.91 s
vs B 71.53 s (−0.52 %), inside the noise.

## 7. nsys kernel-level A/B (same build, `s114nsys-16k-on` vs `s115nsys-16k-on`, wide ON)

The MoE f16 GEMM family total is **unchanged**: 11,901.8 ms (LT OFF) vs 11,931.2 ms (LT ON).
The kernel names and launch counts are identical between the two captures — cuBLASLt's first
heuristic algo selects the **same kernels** as `cublasGemmEx(CUBLAS_GEMM_DEFAULT)` for the
engine's shapes. So the isolated-kernel microbench win (2–9 %) does not translate to the engine
(the GEMM is dominated by the L2/DRAM state of the running prefill, not the algo choice).

## 8. Verdict (mission step 8) — stop and document

The cuBLASLt routing lever is a **marginal, borderline** improvement: a wash at 16 K and a wash
at 32 K (the A/B direction flips between runs; averages −0.52 %, inside the noise band), with the
nsys GEMM family time unchanged. It is not a clear, meaningful end-to-end improvement, so per the
mission's stop rule the stage **stops here**: the change is kept OFF-by-default (behind
`STRATA_MOE_GEMM_LT`), production config restored, and the stage documented without forcing a win.

The GU expert GEMM remains the largest actionable MoE prefill component (≈4.3 s at 16 K, 40 % of
DRAM peak). Reaching the roofline would require a tensor-core MMA kernel tuned for the skinny
ne≤64 bulk (the f32-FMA family was rejected above) — a larger, higher-risk change for a
subsequent stage.

## 9. Artifacts

- Code: `src/prefill/gemm.cu`, `include/strata/prefill/gemm.hpp`, `CMakeLists.txt`
  (all OFF-by-default; `STRATA_MOE_GEMM_LT=1` to enable).
- Microbench: `bench/v100/s115_gemm_custom.cu` (custom-kernel probe, rejected);
  `bench/v100/s115_gemm_microbench.cu` (exact-shape cublasGemmEx GU+D);
  cuBLASLt probe at `/tmp/cublaslt_test.cu` (reproducible: GU N=1280/K=2560, D N=2560/K=640).
- nsys: `bench/v100/s115nsys-16k-on.sh` → `Logs/gpu/s115nsys-16k-on.sqlite` (LT ON); baseline
  `Logs/gpu/s114nsys-16k-on.sqlite` (LT OFF).
- A/B: `Logs/benchmarks/s115_ab_{A,B}{16,32}.out`, `Logs/benchmarks/s110-client.jsonl`.
