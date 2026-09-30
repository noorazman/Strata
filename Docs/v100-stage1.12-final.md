# V100 Stage 1.12 — MoE Prefill: Expert H2D Byte-Volume Diagnostic + Dequant+GEMM Fusion (E5)

Branch `stage1.3-expert-pool-sync`, on top of `1d9dbe7` (the Stage 1.11 baseline). GPU0 only
(32 GB V100, `CUDA_VISIBLE_DEVICES=0`); the 32 K int8-KV production config untouched until the
final restore. `strata.service` stopped during all runs.

Two work items this stage:

1. **H2D byte-volume diagnostic** (mandated before touching anything): exactly where the ~170 GB of
   expert-weight PCIe traffic in the Stage 1.11 report comes from, what it costs, and whether a
   simple, separate, opt-in byte-reduction exists (→ candidate **E8**, kept apart from E5).
2. **E5 — minimal dequant+GEMM fusion** for the prefill expert path, one isolated opt-in variant
   under `STRATA_MOE_DEQUANT_GEMM_FUSE=1` (default OFF), measured against the OFF baseline.

Scope guard (unchanged from the stage mandate): QSA attention / KV layout / int8 KV / context /
MTP behavior / Stage 1.9 sync / E3 / E4 / E7 and the large dense GEMMs are untouched. No
whole-MoE-path rewrite. If the minimal fusion cannot produce a meaningful improvement, stop,
document, and leave it opt-in (or revert).

---

## 1. H2D byte-volume diagnostic (16 K prefill, completed before any code change)

Source: the committed Stage 1.11 nsys capture (`Logs/gpu/s110nsys-16k-off.sqlite`, 16 K prefill,
36.57 s wall) plus a route-dump instrumentation run (`STRATA_MOE_ROUTE_DUMP`,
`Logs/gpu/s112routedump-16k.txt`, engine log `/tmp/s112-routedump.log`: identical routing to the
nsys run — `experts_streamed: 96650, experts_resident: 36526, prefill_ms: 36068.8`).

### 1.1 Where the bytes go (memcpy table, copyKind = H2D)

| Stream | What | Transfers | Bytes | Notes |
|---|---|---:|---:|---|
| 15 | main-model expert blobs (staging ring) | 96,650 | 169.70 GB | 98.5% of the window. 9 size classes, 1,305,600–2,329,600 B, avg 1,755,800 B (1.756 MB) |
| 16 + 18 | MTP drafter expert blobs | 3,164 | 5.41 GB | 3.1%, all in the tail window (draft layers after the main prefill) |
| 13 | PLE rows | — | 0.23 GB | |
| 7 | startup dense-weight load | 10,049 | 20.18 GB | pre-prefill (t0−85.6 s … t0−0.24 s); the large 67 MB–437 MB transfers run 3.0–4.4 GB/s |

Prefill-window H2D total: **172.2 GB**. The main-expert stream's active throughput is
**10.57 GB/s whole-stream** (16.05 s active of 36.57 s wall — the copy engine idles ~55% of the
window waiting on the ring), and **10.18–10.94 GB/s per size class — flat across all 9 classes**.

### 1.2 Accounting

- **10.36 MB of expert weights per prefill token** (169.70 GB / 16,384 tokens; 10.69 MB incl. MTP).
- **3.535 GB per MoE layer** (48 MoE layers); 441.9 MB per layer-chunk; 251.7 streamed experts per
  layer-chunk.
- 133,176 routed expert slots per 16 K prefill (8 chunks × 48 layers × 2048 routed tokens/10).
  **96,650 slots (72.6%) were streamed → 100% of the H2D bytes; 36,526 slots (27.4%) hit the
  resident cache → 0 bytes.**

### 1.3 Redundancy (route dump, measured on the real 16 K routing)

- **20,716 unique (layer, expert) pairs** across the whole prefill — 84% of the 24,576 possible
  (48 × 512). Average 6.43× re-routing of the same pair.
- Per-layer distinct experts across the 8 chunks: min 225 / max 495 / avg 432 — a per-layer-chunk
  working set of ≈20.7 K blobs ≈ 36 GB, i.e. the *whole* prefill's working set is far larger than
  any single cache, but the consecutive-chunk Jaccard is **0.862 avg (min 0.655, max 0.977)** —
  very high locality between adjacent chunks.
- Perfect intra-prefill dedup upper bound: 20,716 × 1.756 MB = **36.4 GB → up to 133 GB (78.6%) of
  the 169.7 GB is redundant re-streaming of blobs this prefill already transferred.**
- The current staging ring deduplicates only within its 7-deep lookahead window, so the measured
  streaming (96,650) is 4.67× the unique-pair count (20,716).

### 1.4 What is and is *not* the lever

- **Transfer size is not the lever.** The 10 GB/s rate is flat across 1.3–2.3 MB transfers, and the
  pre-prefill 67–437 MB dense-weight transfers run *slower* (3.0–4.4 GB/s). The link is already
  saturated by the current transfer sizes; making transfers larger or more contiguous does not
  raise the rate.
- **Residency quality is the lever.** Simulations on the measured access sequence:
  - LRU on the real sequence: 0% hits at cache caps ≤16 K, **82.65% at 18 K**, 84.4% at 20.7 K —
    a phase transition at the ~16.6 K unique-blob (≈29 GB) per-chunk working set.
  - Static top-C (the production profile policy): 48.06% @8 K, 60.07% @10 K, 72.09% @12 K,
    **82.64% @14 K**, 91.35% @16 K — while the *measured* hit rate at the production 8 K cache is
    **27.4%**. The profile ranking (`data/expert-profile.bin`, 8,000 pairs built for 8,000 slots)
    is under-optimized vs true frequency: 48.1% potential vs 27.4% realized at the same 8 K cap.
- **VRAM headroom:** the 32 K production peak is 19.3 GB → ≈12.7 GB spare → at most ≈14 K extra
  slots (24.6 GB cache, ~31 GB total, int8 KV is only ~415 MB at 32 K). A 14 K static profile
  would reach ~83% residency → **170 GB → ~40 GB (~76% cut)**. The 18 K LRU sweet spot is
  unreachable on 32 GB with a static policy.
- The profile file is a fixed 8,000-pair ranked list; the auto-sizer caps the cache at
  `min(profile.size(), (free − reserve)/blob)`, so residency cannot scale past 8 K without
  rebuilding the profile (the builder `tools/make_profile.py` is not in the tree; a small builder
  ships under `bench/v100/` for this experiment).

### 1.5 Verdict → E8 (separate opt-in candidate, not combined with E5)

A simple measurement reveals a potentially large byte-reduction opportunity: **profile-quality +
cache-size (E8)** can cut the 16 K expert H2D from ~170 GB to ~40 GB (~76%) on the same 32 GB card,
independently of the compute path. Per the stage mandate it is evaluated as its own opt-in
candidate (rebuilt larger profile + `--expert-cache N`), kept out of the first E5 A/B, and the
production cache size/architecture is not changed in this stage. (E8 is not measured to completion
in this stage; the simulation evidence is recorded here as the rationale for the next stage.)

---

## 2. E5 — fused dequant + expert GEMM (`STRATA_MOE_DEQUANT_GEMM_FUSE=1`)

### 2.1 What the isolated path costs (OFF baseline, per expert run)

The prefill per-expert sequence (`src/prefill/prefill.cpp`) for expert `e` of layer `l` is:

1. `iq_dequant_gu_f16` (or `dequant_flat_kernel`) — dequant the expert blob's gate/up rows into
   `dq_gu[q]` (1280×2560 f16 = 6.55 MB) and down rows into `dq_d[q]` (2560×640 f16 = 3.28 MB) in
   **global memory**;
2. `gemm.f16(Xs, dq_gu → GU)` — cuBLAS `cublasGemmEx` f16/f16→f32 (the "magma_sgemmEx_kernel"
   trace entries are cuBLAS internals);
3. `swiglu_interleaved(GU → Hh)`;
4. `gemm.f16(Hh, dq_d → Dm)` — second cuBLAS GEMM;
5. `moe_combine`.

Per expert run that is **2 weight-dequant launches + 2 GEMM launches** and a **~23.6 MB
write-then-immediately-read global round trip** of the dequantized weights (19.8 MB written by the
dequants, 19.8 MB read by the GEMMs, plus the f32 intermediates). Across the 96,650 streamed expert
runs per 16 K prefill that is ~190 K extra kernel launches and ~3.8 TB of global-memory traffic
on weights that exist in another form (packed) already in device memory.

### 2.2 The change (one opt-in flag, default OFF)

`STRATA_MOE_DEQUANT_GEMM_FUSE=1` replaces steps 1+2 and 4 with two fused kernels
(`src/kernels/cuda/moe_fused.cu`): each dequantizes its weight tiles **directly from the packed
GGUF blob into shared memory** and runs the GEMM on the tensor cores in the same kernel. Steps 3
and 5 (swiglu, combine), the staging ring, residency, buffers and stream logic are unchanged.

- Tile: 32 weight rows × 32 token rows per CTA, 128 threads (4 warps), 16×16 wmma f16/f32 MMA.
- GU GEMM (M = 1280, K = 2560): K in 10 strips of 256 (weight rows are block-aligned: 10 × 256-
  value blocks per row); 16 KB shared per CTA.
- Down GEMM (M = 2560, K = 640): one 640 strip (K = n_ff is a multiple of the 32/64-value units of
  its two types); 40 KB shared per CTA.
- The gate/up row interleave (even = gate, odd = up) is preserved, so `swiglu_interleaved` is
  untouched; X and the f32 outputs (GU/Dm) are the same buffers the OFF path uses.
- The dequant math is the llama.cpp per-value formulas, factored out of `iq_kernels.cu` into
  `include/strata/kernels/moe_fused_iq.hpp` (single source of truth for kernel and parity test).
- This model's expert pack is mixed-quant per layer (measured via `STRATA_EXPERT_FMT_DUMP=1`):
  gate/up types {16, 17, 18, 21, 22} = {IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S}, down types
  {20, 42} = {IQ4_NL, Q2_0}; n_ff = 640, n_embd = 2560 on all 48 layers. The fused kernels
  instantiate all seven.

### 2.3 Correctness argument

- **Weights bit-identical:** the per-value dequant is a value-by-value transcription of the
  block-wide `dq_*` functions (same scale math, same codebooks, same f16 rounding). Proven by
  `bench/v100/e5_dequant_parity.cu`: for every one of the 7 types, a pseudo-random packed blob is
  dequantized by the engine's `iq_dequant_f16` and by the fused kernel's exact
  (mrow, k) → (unit, pos) addressing + `iq_value<TYPE>`, and the f16 results are compared
  bit-by-bit. **Result: 0 mismatches on all 7 types** (40,960 values each for the GU types,
  20,480 for the D types), including the gate/up interleave mapping.
- **Product:** same hardware MMA (f16×f16→f32 accumulate) on bit-identical f16 inputs, with an
  ascending-K, no-split-K accumulation. cuBLAS may use a different internal tiling/reduction
  order, so last-ULP differences in the f32 product are possible; that is documented rather than
  assumed away, and the variant stays opt-in (per the stage rule).
- **sm_70 NVVM frontend bug (found in the first gate run, fixed):** the original epilogue staged
  the 16×16 f32 accumulator tile in a *local* `float buf[16*16]` (256 floats/thread). The sm_70
  NVVM frontend (CUDA 12.9, NVVM 7.0.1) eliminated the entire `wmma::store_matrix_sync(buf, …)` +
  masked float4 copy chain from the PTX and replaced the live `t0 < T` continuation with a bare
  `trap` — the first gate run (29-token prompt) died with a sticky "unspecified launch failure"
  surfacing at the PLE launch of layer 1, and compute-sanitizer reported a `Trace/breakpoint trap`
  inside `fused_dq_gemm_kernel<17,256,true>` with zero `HMMA`/`STG` instructions in the SASS.
  Bisect (unmasked epilogue: still dropped; no `__restrict__`: still dropped; `buf` in shared
  memory: survives with 64 `st.global` and no trap) isolated the local tile as the trigger. The
  epilogue now stages the tile in this warp's slice of the already-allocated `__shared__ __half
  ws` (reused byte-wise: GU 16 KB / D 40 KB both cover 4 warps × 256 floats), verified by
  re-disassembly: all 7 instantiations now carry their HMMA chains (256 for the GU KT=256 kernels,
  640 for the D KT=640 kernels) and 64 float4 stores, with no `BPT.TRAP`.

### 2.4 Engine gates (fresh engine, `s112e5.sh gate 16384`)

| Gate | Result |
|---|---|
| 32/32 golden prefix | **MATCH** at both 16 K and 32 K (`GOLDEN_MD5 c1517d02…`) |
| 256×2 determinism | **byte-identical** (md5 `aaff1a3f…` on both runs, both contexts) |
| Decode rate | **49.92 tok/s @16 K, 49.04 tok/s @32 K** (target ~49–50) |
| Spec path | accept 0.62, 2.35 tok/round (unchanged behavior) |
| CUDA errors | none (first run's sticky launch-failure was the §2.3 frontend bug, fixed and re-gated) |

**Output vs OFF:** the E5 deterministic output md5 (`aaff1a3f…`) differs from the OFF int8-16K
determinism reference (`cdb7f7d0…`). Per the stage rule this is the expected outcome: the dequant
is bit-identical (§2.3), the GEMM uses the same hardware MMA on the same f16 inputs, but the
K-accumulation association order differs from cuBLAS's internal tiling, so the f32 products can
differ in last ULPs and an argmax/tokenizer boundary can flip. The variant therefore stays opt-in
(`STRATA_MOE_DEQUANT_GEMM_FUSE=1`, default OFF); it is deterministic run-to-run.

---

## 3. Method (measurement protocol)

- OFF baseline: `bench/v100/s110.sh ctx 16384` / `ctx 32768` (fresh engine, production flags,
  1 K warmup + 2 fill runs with md5, VRAM sampled at 2 Hz) and `s18check.py --prefix`
  (32/32 golden prefix, fresh engine).
- E5: same commands with `STRATA_MOE_DEQUANT_GEMM_FUSE=1`; determinism = two fresh-engine runs
  byte-identical (md5 of the 256-token output); decode rate from the same serve runs.
- A/B: paired OFF/E5 at 16 K and 32 K (`bench/v100/s112e5.sh`, same pattern as the Stage 1.11
  `s111e{3,4,7}.sh` A/B harnesses); each leg is a fresh engine so the comparison is prefill-vs-
  prefill on identical prompts.
- nsys verification (`bench/v100/s112nsys.sh` + `s112nsys_analyze.py`): kernel counts and time by
  category, H2D volume by stream (must be unchanged — E5 touches no DMA), comparing the committed
  Stage 1.10 OFF capture (`Logs/gpu/s110nsys-16k-off.sqlite`) against a new E5 capture. Note: this
  nsys export stores kernel names as integer FKs into a `StringIds` table and timestamps in ns —
  the analyzer joins accordingly.
- Microbenchmark (`bench/v100/e5_microbench.cu`): per-expert-shape event timing (median of 30,
  5 warmup) of the OFF pair (`iq_dequant_gu_f16`/`iq_dequant_f16` + `cublasGemmEx` f16/f32)
  vs the E5 fused kernel at the real shapes (GU: M=1280 K=2560; D: M=2560 K=640) and T=8/32.
- Kernel decomposition (`gu_decomp.cu`, not committed): dequant-only vs GEMM-only variants of the
  fused GU kernel (identical grid/tiles, ws pre-filled) to attribute the fused kernel's time.

## 4. A/B results

### 4.1 Wall-clock (fresh-engine serve runs, fill-prompt, 256 max-new)

| Context | Leg | TTFT (s) | decode (tok/s) | output md5 |
|---|---|---|---|---|
| 16 K | OFF r1 (cold) | **40.89** | 50.66 | `6041c5f3` (known anchor) |
| 16 K | E5 r1 (cold) | **72.98** | 50.77 | `2fecee38` (deterministic; ≠ OFF, §2.4) |
| 32 K | OFF r1 (cold) | **77.43** | 49.45 | `1fbe577e` (known anchor) |
| 32 K | E5 r1 (cold) | **141.64** | 49.88 | `1c21f82b` (deterministic; ≠ OFF, §2.4) |

**TTFT regression: 16 K +32.1 s (1.78×); 32 K +64.2 s (141.64 / 77.43 = 1.83×).** Decode is
unaffected (50.77 / 49.88 vs 50.66 / 49.45 tok/s) because E5 only changes the prefill expert
path — decode already uses the resident expert-pool path. The wall-clock ratio tracks the
per-expert microbenchmark (§4.3): the expert path is a slightly larger share of the 32 K prefill,
so the 32 K ratio is marginally worse.

### 4.2 nsys kernel-level (16 K prefill window)

| Category (kernel time) | OFF | E5 |
|---|---|---|
| Total kernel time | 32.6 s (1.30 M launches) | 66.9 s (0.90 M launches) |
| — i-quant expert path | dequant 10.4 s + GEMM ~2.3 s ≈ **12.7 s** | **fused 50.9 s** |
| — dense / bf16 GEMM (untouched) | ~9.9 s | ~9.9 s (GEMM 5.6 s + rest) |
| — other (QSA/GDN/elementwise) | ~9.1 s | ~9.1 s (identical) |
| H2D main-expert (stream 15) | 169.70 GB | 170.34 GB (unchanged) |
| Launch count | 1 304 693 | 899 126 (−31 %) |

E5 removes ~4 launches per expert run (2 dequant + 2 GEMM → 2 fused) and leaves the DMA volume
identical, exactly as designed. But the fused kernel spends **50.9 s** on the expert path that the
OFF pair does in **12.7 s** — a ~4× blowup on that path. That single number is the whole story:
everything else in the prefill is byte-for-byte the same code.

### 4.3 Root cause (microbenchmark + kernel decomposition)

Per-expert event timing at the real shapes (`e5_microbench.cu`, T = 8 and 32, the routed-expert
size is ne ≈ 6.6 so T = 8 is representative):

| (M × K) | OFF dequant | OFF cuBLAS GEMM | OFF pair | E5 fused | ratio |
|---|---|---|---|---|---|
| GU (1280 × 2560), T = 8 | 57.3 µs | 29.7 µs | **87.0 µs** | **199.7 µs** | **2.29×** |
| D (2560 × 640), T = 8 | 30.7 µs | 18.4 µs | **49.1 µs** | **36.8 µs** | **0.75×** |
| expert total, T = 8 | — | — | **136.1 µs** | **236.5 µs** | **1.74×** |
| GU (1280 × 2560), T = 32 | 56.3 µs | 39.9 µs | **96.2 µs** | **200.7 µs** | **2.09×** |
| D (2560 × 640), T = 32 | 29.7 µs | 20.5 µs | **50.2 µs** | **36.9 µs** | **0.73×** |

Two facts:

1. **The D (down) fusion is a win** (0.73–0.75×). Its dequant is small (32-value units) and its
   GEMM tile (M = 2560 → 80 CTAs) has enough parallelism that folding the dequant in saves a
   launch and the intermediate global round-trip.
2. **The GU (gate/up) fusion is a 2.1–2.3× loss.** Decomposing the fused GU kernel
   (`gu_decomp.cu`, same grid/tiles): **dequant-only = 186 µs (85 %)**, GEMM-only (ws
   pre-filled) = 58 µs (15 %). The dequant is the entire problem.

Why the dequant is 3.3× slower inside the fused kernel (186 µs) than as its own kernel (57 µs)
for the *same values*: the standalone `dequant_gu_kernel` runs a grid of
`dim3(n_ff * per_row, 2) × 32` = **12 800 blocks × 32 threads = 409 600 threads**, each doing a
tiny slice — it is memory-bound and fully parallel. The fused kernel runs the dequant *inside* the
GEMM CTA, and the GEMM tile is fixed at 32 weight-rows per CTA, so the GU grid is only
`(M/32, ceil(T/32)) × 128` = **40 CTAs × 128 = 5 120 threads** — 320× fewer CTAs. The dequant work
(32 rows × 2560 K per CTA, 10 stripes) is now serialized across 4 warps per CTA instead of being
spread over 409 K threads. With the GEMM tile pinned to 32 rows (to match the 16×16 wmma layout
and keep the 4-warp m-split), there is no freedom to add CTAs without also re-partitioning the
weight tile. At ne ≈ 6.6 the GEMM is too skinny to amortize the dequant that was moved into it.

**Net:** the minimal 32×32-tile fusion trades the dequant's standalone parallelism for a GEMM
locality that this shape does not need, and the GU path (the larger dequant) dominates. D wins,
GU loses ~4×, and the expert total lands at 1.74× — matching the 1.78× (16 K) / 1.83× (32 K)
wall-clock regressions.

### 4.4 Verdict (stop rule)

Per the stage stop rule — *"if a minimal fusion can't produce a meaningful improvement, stop,
document, and keep it opt-in (or revert)"* — E5 is **correct and deterministic but slower**, so it
stays **opt-in, default OFF** (`STRATA_MOE_DEQUANT_GEMM_FUSE=1`). It is a clean, low-risk
baseline that isolates exactly one change, and it quantifies the parallelism cost that any real
fusion must beat. Recommended next-stage options if fusion is revisited (all out of scope here):
split-K on the GU path (more CTAs over K, fixed-order partial-sum reduce to keep determinism),
a dequant-parallelism-preserving tile (e.g. 16 weight-rows/CTA with a 2-warp m-split), or fusing
only the D path (already a 0.73× win) and leaving GU on the OFF dequant+GEMM.

## 5. What changed in the tree

Production code (behind the new flag, default OFF — no behavior change when unset):
- `include/strata/kernels/moe_fused.hpp` — new: `moe_fused_gemm_gu` / `moe_fused_gemm_d` launcher
  declarations.
- `include/strata/kernels/moe_fused_iq.hpp` — new: the per-value `iq_value<TYPE>` dequantizer
  (16/17/18/20/21/22/42), transcribed value-by-value from `iq_kernels.cu` so it is bit-identical
  to the separate dequant path.
- `src/kernels/cuda/moe_fused.cu` — new: the two fused wmma kernels (32×32 tile, 4 warps,
  16×16 f16-in/f32-acc, K-strips of 256 for GU / single 640 strip for D, dequant → shared `ws` →
  wmma). Includes the shared-`ws` epilogue fix for the sm_70 NVVM frontend trap (§2.3).
- `src/prefill/prefill.cpp` — read `STRATA_MOE_DEQUANT_GEMM_FUSE` (default 0); the native-expert
  prefill branch dispatches to the fused kernel when set, else the original dequant + `m.gemm.f16`.
  No other prefill change.
- `CMakeLists.txt` — compile `moe_fused.cu` into the kernel library.

Bench tooling (`bench/v100/`, additive only):
- `s112e5.sh` — E5 gate + A/B harness (fresh-engine golden prefix, 256×2 determinism, decode, and
  the `s110.sh`-based serve runs with `STRATA_MOE_DEQUANT_GEMM_FUSE=1`).
- `s112nsys.sh` — nsys capture of the E5 16 K prefill (same flags as the OFF capture).
- `s112nsys_analyze.py` — OFF-vs-E5 kernel/H2D comparison from the nsys sqlite exports
  (handles the `StringIds` FK name table and ns timestamps).
- `e5_dequant_parity.cu` — bit-identical dequant proof for all 7 types (exact kernel addressing
  incl. the GU row-interleave).
- `e5_microbench.cu` — per-expert-shape event timing of OFF pair vs E5 fused.

Data / docs:
- `Logs/gpu/s112routedump-16k.txt` (route dump used for the §1 redundancy accounting),
  `Logs/gpu/s112nsys-16k-e5.sqlite` (force-added like the 1.11 OFF capture),
  `Logs/benchmarks/s112e5-g*.json` + `Logs/cpu|gpu/s112e5-g*.csv|out` (gate/A/B raw results),
  this doc.

Untouched (scope guard): QSA attention, KV layout, int8-KV, context, MTP, Stage 1.9 sync, E3/E4/E7
(all remain OFF), the large dense GEMMs, and the production 32K/int8/8180 config.

## 6. Production state after the stage

- All experimental flags OFF: `STRATA_MOE_DEQUANT_GEMM_FUSE=0` (default), E3/E4/E7 OFF.
- `strata.service` restarted on the production config (32 K / int8 KV / port 8180 /
  `--expert-cache auto`), `/health` green, one live request (max_tokens ≥ 96) returned normally.
- Cold-16K r1 md5 and decode are back to the OFF anchors after restore (verified at commit time).
- E5 remains available for A/B at any time via the env flag; enabling it changes prefill numerics
  in last ULPs (§2.4) and costs ~1.8× prefill wall-clock at 16 K (§4), so it is **not** a
  production-default candidate until the GU parallelism problem is solved.
