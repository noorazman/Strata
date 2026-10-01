# V100 Stage 1.16 — MoE Prefill Skinny-GU Tensor-Core GEMM (`STRATA_MOE_GEMM_TC=1`) — FINAL

Production target: 32 GB V100 (GPU0) only. Started from clean Stage 1.15 HEAD `5d2b747`.
Baseline config for A/B: 32K ctx / int8 KV / 8K expert cache / E5/E8 OFF / Stage 1.14 wide dequant ON (leg A = wide ON + TC OFF, leg B = both ON). TC is OFF-by-default; no production-default change.

**Verdict: WASH (mission stop rule) — the isolated kernel-level win is real (GU-small GEMM GPU time 5.46 → 3.43 s, −37 %, 88.5 K fewer kernel launches at 16 K), but it does not convert into a clear, repeatable end-to-end TTFT improvement (16 K r1 −1.2…−2.0 %, 32 K r1 −0.4…+1.3 %, inside the ~±1.5 % r1-TTFT noise band; non-GEMM kernels absorb +1.59 s of the GEMM cut). Per the stop rule the stage stops here: the path stays OFF-by-default, production is restored + verified live, and the stage is documented without forcing a win. No other optimization is attempted within Stage 1.16.**

## 1. Mission and stop rule

Find the next highest-impact MoE PREFILL optimization, PROFILE FIRST (no assumption); the largest actionable component is the skinny GU expert GEMM (~4.3 s of the 16 K prefill). Pick ONE isolated change, implement OFF-by-default, prove with microbench + full engine gates + 16K/32K A/B + nsys. PASS only if the tensor-core path produces a clear, repeatable end-to-end prefill/TTFT improvement beyond normal measurement noise (Stage 1.15 precedent: ±1–1.5 % = wash; "clear" ≈ ≥2–3 %). A microbenchmark win alone is NOT sufficient. If wash/slower/numerically problematic: STOP, document, restore production, do not proceed to another optimization within the stage.

## 2. Profile first (no assumption)

16 K serve-mode nsys of the leg-A baseline (`Logs/gpu/s114nsys-16k-on.sqlite`, wide-ON): GEMM family **10.94 s / 248,520 kernels** in a 35.43 s span (GPU busy 84.6 %); non-GEMM 19.02 s. The MoE expert f16 GEMM is 133,157 GU calls (route dump, verified: 20,480 (expert,count) pairs per 2048-token chunk-layer = 2048×top-10) plus 133,157 D calls. **GU GEMM ne distribution (FLOP-weighted): ne 1–8 = 1.9 %, 9–16 = 2.8 %, 17–32 = 5.7 %, 33–64 = 12.8 %, 65–128 = 19.2 %, 129–256 = 23.8 %, 257–512 = 19.4 %, 513–1024 = 10.2 %, >1024 = 4.2 %** — the TC-routable range (ne ≤ 64) is **only 23.2 % of GU GEMM FLOPs but 75.5 % of the call count** (100,543 of 133,157). cuBLAS serves the small-ne calls as TWO kernels (main + splitKreduce; 140,299 reduces in the baseline, 0.97 s). The ~4.3 s GU wall is ~133 K individual cuBLAS dispatches: each carries cuBLAS heuristic/dispatch CPU cost plus, for small ne, a second reduce launch. So a tensor-core kernel for the skinny bulk attacks both GPU kernel time and per-call launch overhead.

## 3. Targeted shape and routing guard

ONE shape is TC-routable in `ModelGeometry`: the GU expert GEMM `Y[ne,1280] = X[ne,2560] · W[1280,2560]ᵀ` (f16 in, f32 out, W 6.55 MB). Guard in `Gemm::f16`: `use_tc_ && T <= tc_max_ne_ && T <= 128 && beta == 0.0f && N == 1280 && K == 2560 && ldy == 1280`; anything else falls through to the cuBLAS/cuBLASLt path unchanged (D GEMM, dense GEMM, decode untouched). Env: `STRATA_MOE_GEMM_TC=1` (OFF default), `STRATA_MOE_GEMM_TC_MAXNE` (default 64, clamped 1..128), `STRATA_MOE_GEMM_TC_VERIFY=1` (debug: co-compute cuBLAS per routed call and log nneq/maxAbs; 2-float buffer, zero cost when off).

## 4. Microbenchmark (`bench/v100/s116_gemm_tc.cu`, L2-hot W, in-run cuBLAS baselines)

| ne | TC µs | cuBLAS µs | ratio |
|----|-------|-----------|-------|
| 1  | 20.5  | 23.6      | 0.87× |
| 2  | 20.5  | 26.6      | 0.77× |
| 4  | 20.5  | 26.6      | 0.77× |
| 8  | 20.5  | 27.6      | 0.74× |
| 16 | 25.6  | 28.7      | 0.89× |
| 32 | 43.0  | 37.9      | 1.14× |
| 64 | 75.8  | 43.0      | 1.76× |
| 128| 128.0 | 43.2      | 2.96× |

maxAbs ≤ 1.4e-04 vs cuBLAS (f32 add-order, not bit-exact — expected for a different MMA accumulation order). Wins only at ne ≤ ~16; intermediate ne (17–64) loses. Two earlier traps were root-caused and fixed before this table was trusted: (a) the first kernel variant summed only 1/4 of K because `mma.sync.m8n8k4` computes ONE k4 dot per instruction (the four lane groups replicate the same k4 tile) — fixed with 4 sequential mmas over a shared k8 window, validated standalone (`/tmp/mma_fix.cu`, nneq=0); (b) the first microbench data generator produced f16-infinity values (LCG bit patterns), making any k-subset sum bit-exact by accident — data gen is now signed finite f16 (W ∈ [−0.173, 0.173], X ∈ [−2.1, 2.1]).

## 5. Implementation (ONE isolated change, OFF-by-default)

`src/prefill/gemm.cu` TC section: `tc_mma816` (m8n8k4 f16→f32 wrapper) + `template<bool FULL> __global__ void __launch_bounds__(256) tc_gu_kernel(const uint16_t* W, const uint16_t* X, float* Y, int ne)`. Design: m16×n128 block, 8 warps (n16-tiles × KW k-splits), uint4 16-byte k8 windows read straight from L2/DRAM, register Q-cache depth `TC_QCACHE=4` (a depth-8 cache pushes Q to local memory and costs ~6.5× — the queue-depth cliff), consume-then-refill per window, smem `part[8][16][16]` fixed-order reduce, C/D output mapping with 4 lane groups redundantly writing identical values (race-benign per the PTX doc). `template<bool FULL>`: for ne ≤ 8 the launcher dispatches `tc_gu_kernel<false>`, which skips the second n8 half's W loads and 4 c0 mmas (halves X over-read); a runtime nt-skip branch was tried first and rejected (it pushed Q to the stack). ptxas: `<true>` 120 regs, `<false>` 86 regs, 0 stack, 0 spills, 8 KB smem — identical to the pre-stage kernel's static footprint. Launcher `tc_gemm_gu`: grid `(1280/16, (T+127)/128)`, `T <= 8 → <false> else <true>`, on the caller's stream. Fallbacks: shape guard (above), sticky init-failure flag, cuBLAS path untouched. Rejected during the stage (documented, not reopened): f32-FMA custom kernels (1.15), L2 persistence, FAST_16F, full-K single-warp TC, m8 m-tiles (X over-read doubled), runtime nt-skip branch, deep Q cache.

## 6. Correctness and determinism gates

- **32-token golden (leg B):** first-32 MATCH (first token 271).
- **Fresh-engine 256-token determinism (leg B):** det×5 = **5/5 identical** (md5 `b482c1b06a18d8b8d014085e5a55d712`).
- **OFF vs ON numerical correctness:** in-engine verify hook (`STRATA_MOE_GEMM_TC_VERIFY=1`, 29-token run): `nneq` = full output size, **maxAbs ≤ 2.4e-06** vs cuBLAS on every routed call — ~1 ULP f32 add-order difference, as designed. This flips top-10 routing at layer 1 for a few experts and cascades, so leg-B trajectories legitimately differ from leg-A at token level; the acceptance signals are leg-B internal determinism + golden + timing (both held).
- **16 K fill r1 correctness:** leg B r1 byte-stable across repeats (md5 `2cc3d466…` 4/5 in the s110 harness, 4/4 in a direct exact-sequence engine run; MAXNE variants each stable 2/2: `155e727b…` @16, `d6733778…` @32). Leg A anchor `6041c5f3…` stable 18+ runs.
- **32 K fill r1 correctness:** leg A `1fbe577e…` stable 3/3; leg B `e291dd85…` (MAXNE 64) and `35d25cda…` (MAXNE 16) stable 2/2 each.
- **No CUDA errors / hangs** in any leg-A or leg-B run (all rc=0).
- **Pre-existing warm-path flake (NOT TC-specific):** the exact s110 sequence [100-tok probe → 1024-tok warmup → 16K r1 (256) → 16K r2 (256)] run 4× at engine level: leg B 4/4 identical; leg A (cuBLAS-only control) 4/4 on r1 but r2 flaked 1/4. Combined with the s110-harness history (leg B r2 2/5 divergent, leg A r2 historically stable but untested in pairs), the intermittent warm r2 divergence exists in BOTH legs — it is a latent property of the warm decode path (expert-cache residency/adapt interaction under the Python serve flow), not an artifact of the TC kernel. The stage's determinism gate (fresh-engine, 5/5) is on the cold path and passes.

## 7. 16 K A/B (r1 cold prefill, ms; `s110.sh ctx 16384`)

| run | leg A (TC off) | MAXNE=16 | MAXNE=32 | MAXNE=64 |
|-----|----------------|----------|----------|----------|
| 1   | 39,262.6       | 37,376.9 | 38,586.1 | 37,896.7 |
| 2   | 38,477.2       | 38,456.0 | 37,921.7 | 38,036.1 |
| 3   | 38,225.9       | —        | —        | 37,627.4 |
| 4   | 38,863.1       | —        | —        | 38,549.9 |
| avg | **38,707**     | **37,917 (−2.0 %)** | **38,254 (−1.2 %)** | **38,028 (−1.8 %)** |

Back-to-back A/B pairs: −3.48 %, −1.15 %, −1.57 %, −0.81 %. Leg A's own run-to-run spread is 2.7 % (±1.3 % half-band); the TC improvement (1.2–2.0 %) is ~1.5× the half-band — directionally favorable at 16 K but not clearly beyond noise.

## 8. 32 K A/B (r1 cold prefill, ms; `s110.sh ctx 32768`)

| run | leg A | MAXNE=64 | MAXNE=16 |
|-----|-------|----------|----------|
| 1   | 71,820.3 | 72,751.5 | 72,059.5 |
| 2   | 72,863.6 | 72,063.7 | 71,816.1 |
| 3   | 72,048.7 | —        | —        |
| avg | **72,244** | **72,408 (+0.2 %)** | **71,938 (−0.4 %)** |

32 K is a flat wash (all within 0.5 %; leg A's own spread is 1.4 %).

## 9. nsys kernel-level A/B (16 K serve, `s114nsys-16k-on` vs `s116nsys-16k-on`)

| | leg A (TC off) | leg B (TC on, MAXNE 64) | Δ |
|---|----------------|--------------------------|---|
| GEMM-family total | 10.94 s / 248,520 kernels | 8.91 s / 160,049 kernels | **−2.03 s, −88,471 launches** |
| tc_gu_kernel | — | 3.430 s / **100,543 calls** (= the exact routed ne ≤ 64 count; SKIP8 `<false>` 42,767 = the exact ne ≤ 8 count, 0.990 s @ 23.1 µs; FULL `<true>` 57,776, 2.440 s @ 42.2 µs) | — |
| GU-small (ne ≤ 64) cuBLAS time | 5.46 s (derived: A GEMM total − B non-TC GEMM; 100,543 mains + the GU share of the 140,299 splitKreduces) | 3.43 s (tc_gu) | **−37 % GPU time on the routed FLOPs** |
| splitKreduce | 140,299 calls / 0.97 s | 51,603 calls / 0.44 s | −88,696 launches |
| non-GEMM kernels | 19.02 s | 20.61 s | **+1.59 s (L2/DRAM contention with the TC W-stream + per-kernel jitter; top non-GEMM kernels individually unchanged: attn 2.70→2.68 s, gdn 1.39→1.39 s, bf16 mmvf 1.43→1.39 s, wide_gu 1.07→1.06 s)** |
| span / busy | 35.43 s / 29.96 s (84.6 %) | 35.35 s / 29.52 s (83.5 %) | −0.08 s |

Kernel-level verdict: the routed GU-small GEMM genuinely improved (5.46 → 3.43 s, −37 %, −88.5 K launches). The e2e gain falls short because (a) +1.59 s of non-GEMM regression absorbs 3/4 of the GEMM cut, and (b) with the GPU 84 % busy, part of the removed kernel time was not on the critical path.

## 10. VRAM / RAM

Peak VRAM **19,032 MiB identical across all legs** (TC adds no device allocations: Q-cache in registers, 8 KB `part` smem same as the pre-stage footprint, shared 32 MB GEMM scratch; nvidia-smi sampled over each 16 K ctx run). Host RAM stable across all runs.

## 11. Verdict

**WASH.** The microbenchmark and kernel-level wins are real (GU-small GEMM −37 % GPU time, −88.5 K launches at 16 K), but the end-to-end r1-TTFT effect is −1.2…−2.0 % at 16 K and −0.4…+1.3 % at 32 K — inside the ±1–1.5 % noise band established in Stage 1.15, and the 32 K direction even flips slightly positive for MAXNE=64. The routed range is only 23.2 % of GU GEMM FLOPs, and the intermediate-ne kernel loss (1.14–1.76× at ne 17–64, 12.8 % of GU FLOPs) cancels most of the small-ne win at the kernel level, while the non-GEMM contention and GPU-busy critical path eat the rest. Per the stop rule: **STOP — the path stays OFF-by-default, production restored + verified, stage documented, no further optimization attempted within Stage 1.16.**

## 12. Restored state and next bottleneck

- `STRATA_MOE_GEMM_TC` remains OFF-by-default (with `STRATA_MOE_GEMM_TC_MAXNE` default 64); all production flags unchanged from the Stage 1.15 baseline (wide dequant ON, LT OFF, E5/E8 OFF, 8 K cache); `strata.service` restored and verified live (`/health` 200 + live reasoning request).
- The GU expert GEMM remains the largest actionable MoE prefill lever (~4.3 s @ 16 K). The follow-up that this stage's data points to: a **2-D-tiled TC kernel** (m64×n16 blocks, each block reads its 64 W rows and 16 X rows ≈ once each) to kill the intermediate-ne X-over-read loss — the microbench says ne 17–64 is 1.14–1.76× today and memory-bound at ~3.4 s for the whole routed range; a ~2 s floor on the routed range would push 16 K toward ~−4 % and could clear the stop rule. Reducing the 133 K per-expert launch count (grouped/batched skinny GEMM) is the second candidate. Both are larger, higher-risk changes for a subsequent stage. The pre-existing warm r2 flake (Section 6) is logged as a separate follow-up.

## Data and tooling

- Microbench: `bench/v100/s116_gemm_tc.cu` (synced to the final kernel; standalone build `/tmp/s116_repo`).
- nsys: `bench/v100/s116nsys-16k-on.sh`, `Logs/gpu/s116nsys-16k-on.{nsys-rep,sqlite,run.log}` (leg B) vs `Logs/gpu/s114nsys-16k-on.sqlite` (leg A baseline).
- A/B data: `Logs/benchmarks/s116B-{g32,det1..det5,16k-r1,16k-r2}.json` (leg B gates), `s110-client.jsonl` (r1 anchors), per-run engine logs `/tmp/{sweep3,ctx32k,ctx32k2}-*.engine.log`, VRAM csvs `/tmp/sweep3-*-vram{1,2}.csv` (all peaks 19,032 MiB).
- Route dumps: `/tmp/route_B_s{1,2}.txt` (16 K per-expert counts, identical across runs).
- Determinism probes: `/tmp/r1det.sh`, `/tmp/fullseq.sh`, `/tmp/dblfill2.sh` (+ leg-A control `/tmp/dblfillA.sh`) — exact s110 sequence at engine level.
