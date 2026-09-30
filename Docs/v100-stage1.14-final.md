# V100 Stage 1.14 — MoE Prefill Dequant: Wide Memory Ops (STRATA_MOE_DQ_WIDE)

Stage 1.14 from the clean Stage 1.13 commit `26d0719` (branch `stage1.3-expert-pool-sync`,
GPU0, V100 32 GB PCIe). Production 32 K / int8 KV config (E5 OFF, E8 OFF, 8 K expert cache)
untouched until the final restore; the optimization is opt-in (`STRATA_MOE_DQ_WIDE=1`, default OFF).

**Mission:** find the next highest-impact optimization for MoE PREFILL wall-clock latency.
Pick ONE isolated change, implement it OFF-by-default, and run the full gate (correctness,
determinism, 16 K / 32 K performance, nsys A/B). If it is not a meaningful improvement, stop
and document it — do not force a win.

## 1. Baseline (mission step 1) — fresh nsys of the Stage 1.13 production build

Two fresh serve-mode captures (`bench/v100/s114nsys-{16k,32k}-off.sh`, 16 K / 32 K corpus
fill-prompt + 64 max-new, `--expert-cache auto` + the 8,000-slot production profile):

| 16 K prefill window | 36.03 s | 32 K prefill window | 71.99 s |
|---|---|---|---|
| total kernel time | 32,573.6 ms | 65,747.1 ms |
| **dequant_gu_kernel** | **6,444.9 ms** (133,176 launches) | **12,971.0 ms** (266,722) |
| **dequant_flat_kernel** | **3,118.8 ms** (149,999) | **6,267.4 ms** (300,105) |
| dequant total | 9,563.7 ms | 19,238.4 ms |
| GEMM (cuBLAS expert+dense) | 12,196.5 ms | 24,357.6 ms |
| attn / other | 9,113.4 ms | 18,794.2 ms |
| expert H2D (stream 15) | 169.70 GB (96,650) | 339.65 GB (193,239) |

The 16 K anchors match Stage 1.13 (≈35.9 s window, ≈32.5 s kernels, ≈133 K dequant launches);
the 32 K row is new data for the 32 K A/B gate.

## 2. Bottleneck (mission steps 2–3)

`dequant_gu_kernel` is the largest non-GEMM prefill kernel — **6.44 s of the 36.0 s window at
16 K (19.8 % of total kernel time), 13.0 s at 32 K**. The per-expert MoE path is:
`dequant_gu_kernel` (gate+up, 6,400 superblocks) + `dequant_flat_kernel` (down, 6,400
superblocks) → f16 weight buffers → cuBLAS GEMM. Across the 48 MoE layers the expert quant
types (per-layer dump, `STRATA_EXPERT_FMT_DUMP=1`) are:

- GU (gate+up): type 16 (IQ2_XXS) ×9, 17 (IQ2_XS) ×10, 18 (IQ3_XXS) ×6, 21 (IQ3_S) ×13, 22 (IQ2_S) ×10.
- D (down): type 20 (IQ4_NL) ×22, 42 (Q2_0) ×26.

ncu on the baseline `dequant_gu_kernel` (production shape n_ff=640, n_embd=2560) is
**L1/TEX-pipe bound, not DRAM-bound**:

| metric | baseline dequant_gu | |
|---|---|---|
| duration | 55.6 µs | |
| L1/TEX pipe | **77 %** | |
| DRAM | 17 % | |
| SM | 9 % | |
| issue | 0.105 warp-instr / cycle / SM | |
| cycles / inst | 72.3 | |

The SASS confirms the cause: each thread issues **8 × `LDG.E.U8`** (one byte per grid entry)
and **8 × `STG.E.U16`** (one half per output element). The dequantizer loads its codebook grid
entry one byte at a time and stores its 8 output halves one at a time, so the L1 pipe — not
memory bandwidth — is the limiting resource.

## 3. The optimization (mission steps 4–5) — one isolated change: wide memory ops

Replace the byte-granular loads/stores with wide ones, keeping the per-value math and
evaluation order **verbatim** (so the f16 output is bit-identical — a scheduling/width change,
not a numerics change):

- **Load the codebook grid entry with one wide load.** The grids are `__device__` arrays
  declared at their native width (`iq2xxs_grid`/`iq2xs_grid`/`iq2s_grid` are `uint64_t[]`,
  `iq3xxs_grid`/`iq3s_grid` are `uint32_t[]`), so each grid entry is read with a single
  `LDG.64` / `LDG.32` instead of 8 × `LDG.E.U8`. The 1-byte tables (`kmask_iq2xs`,
  `kvalues_iq4nl`) are read via 4-byte-aligned sub-offsets.
- **Store the 8 output halves with one 16-byte `STG.128`** (`uint4`) instead of 8 × `STG.E.U16`.
- Each thread (32/block) handles one 256-value superblock; `PER` independent superblocks per
  thread add memory-level parallelism. **GU runs PER=1, flat runs PER=2** (microbench optimum).

Validation (`bench/v100/s114_wide_kernel_test.cu`, production shapes, bit-identity vs the
baseline `iq_dequant_gu_f16`/`iq_dequant_f16` for all 7 types at PER 1 and 2, plus a host
f16 bit-pattern oracle):

| dequant | baseline | wide | speedup | bit-ident |
|---|---|---|---|---|
| GU (PER=1) | 56–59 µs | 31.7 µs | 1.75–1.81× | ✓ all 5 GU types |
| D t20 (PER=2) | 20.5 µs | 11.2 µs | 1.82× | ✓ |
| D t42 (PER=2) | 29.7 µs | 10.2 µs | 2.90× | ✓ |

Three bit-identity bugs were found and fixed during validation (the microbench + host oracle
catch all of them): (1) type 16 (IQ2_XXS) — the production grid index `aux8[il]` is **byte `il`
of the 8-byte group** at `qs+8*ib` (for `il=1,3` the high byte of `q2[il/2]`, not the low byte);
(2) type 21 (IQ3_S) — `((uint8_t)(qh << (8-2*il)) & 256)` truncates to 8 bits *before* the `& 256`
so the high-bit contribution is always 0 (production has no cast); (3) type 42 (Q2_0) — the
output mapping is `yy[b*64 + part*8..+7]` (b=tid>>3, part=tid&7), not the GU `32*part+8*b`.

ncu on the wide kernel confirms the mechanism: `wide_gu` 55.6 → **31.1 µs** (L1 77 % → 69 %,
DRAM 17 % → 28 %, cycles/inst 72.3 → 29.5, issue 0.105 → 0.21). It is still L1/TEX-capped
(69 %) and occupancy-capped at 38.9 % (1 warp/block, 32 blocks/SM = 50 % of 64 warps/SM) — so
it plateaus ≈2× above the 15.4 µs memory roofline. Closing that remaining gap (128-thread
blocks, 2+ warps/block) is **documented future work, not part of this isolated change**.

## 4. Implementation (mission step 6) — OFF by default, `STRATA_MOE_DQ_WIDE=1`

- `src/kernels/cuda/iq_kernels.cu` — 7 wide device dequantizers (`wide_dq_iq2_xxs/xs`,
  `wide_dq_iq3_xxs/s`, `wide_dq_iq2_s`, `wide_dq_iq4_nl`, `wide_dq_q2_0`) + helpers
  (`wide_kmask8`, `wide_kv_byte`, `wide_sgn`, `wide_store8`) + a `wide_sel<TY>` dispatch +
  `wide_gu_kernel<TY,PER>` / `wide_flat_kernel<TY,PER>` + launchers
  `iq_dequant_gu_f16_wide` / `iq_dequant_f16_wide` + `iq_wide_supported(t)`. **Types without a
  wide implementation fall back to the baseline kernel inside the launcher**, so the wide
  entry points are drop-in replacements and numerics are invariant for any type.
- `include/strata/kernels/iq_kernels.hpp` — declarations.
- `src/prefill/prefill.cpp` — a `moe_dq_wide` flag parsed from `STRATA_MOE_DQ_WIDE=1`
  (default OFF), routing the two native-pack dequant call sites through the wide launchers.
  Orthogonal to E5: when both are on, E5's fused path wins and the flag is inert for native
  layers.

The change is numerics-invariant by construction (same f16 bits, only load/store width and
scheduling). No QSA / KV / context / MTP / Stage 1.9 / E3 / E4 / E7 / E5 / E8 changes; no
whole-MoE-path rewrite; no production-default change; no multi-GPU.

## 5. Gates (mission step 7)

**Correctness + determinism (fresh engines, int8 KV, 16 K + 32 K, engine-level
`bench/v100/bench.py` 29-tok canonical prompt):**

| check | 16 K | 32 K |
|---|---|---|
| 32-tok golden prefix (OFF) | MATCH | MATCH |
| 32-tok golden prefix (ON) | MATCH | MATCH |
| 256-tok det1 == det2, OFF (determinism) | ✓ `cdb7f7d056f3…` | ✓ `cdb7f7d056f3…` |
| 256-tok det1 == det2, ON (determinism) | ✓ `cdb7f7d056f3…` | ✓ `cdb7f7d056f3…` |
| **OFF det == ON det (bit-identity)** | **✓ byte-identical** | **✓ byte-identical** |

**Serve fill-prompt (the long-context prefill itself, `s110.sh ctx`):** OFF r1 == ON r1
byte-identical at both contexts (16 K `6041c5f3…`, 32 K `1fbe577e…`) — the wide dequant is
bit-identical on the real 16 K / 32 K prefill, not just the short prompt. (The r2/warm fill
run DIVERGES in **both** OFF and ON — a pre-existing warm-state / MTP non-determinism the
baseline itself shows — so r1 is the deterministic anchor and OFF r1 == ON r1 is the
numerics-invariance check.)

**No CUDA errors** in any gate or A/B run.

## 6. A/B (mission step 7) — wall-clock + nsys

**Wall-clock TTFT (`s110.sh ctx`, r1 cold anchor; decode unchanged):**

| context | OFF TTFT | ON TTFT | delta | decode (r1) |
|---|---|---|---|---|
| 16 K | 40.44 s | 37.72 s | **−2.72 s (−6.7 %)** | 51.6 → 51.3 tok/s |
| 32 K | 77.54 s | 71.36 s | **−6.17 s (−8.0 %)** | 49.4 → 48.7 tok/s |

(r2/warm: 16 K −2.81 s / −7.1 %, 32 K −6.77 s / −8.8 % — consistent direction.)

**nsys kernel-level A/B (same build, `s114nsys-{16k,32k}-{off,on}.sh`):**

| metric | 16 K OFF→ON | 32 K OFF→ON |
|---|---|---|
| dequant kernel time | 9,563.7 → 4,821.1 ms | 19,238.4 → 9,685.0 ms |
| **dequant speedup** | **−49.6 %** | **−49.7 %** |
| total kernel time | 32,573.6 → 27,932.3 ms (−14.2 %) | 65,747.1 → 56,483.7 ms (−14.1 %) |
| prefill window (H2D span) | 36.03 → 32.92 s (−3.11 s) | 71.99 → 66.30 s (−5.69 s) |
| expert H2D (stream 15) | 169.70 GB (unchanged) | 339.65 GB (unchanged) |
| dequant launchers | dequant_gu 133,176 → wide_gu 133,176 | dequant_gu 266,722 → wide_gu 266,722 |

The dequant bottleneck — the target — is **halved (−49.6 % / −49.7 %)**, exactly the
microbench prediction. The wall-clock gain (−6.7 % / −8.0 %) is smaller than the dequant
kernel-time cut because the MoE prefill is compute-bound and part of the dequant time was
already overlapped with H2D / off the critical path (the regime documented in Stage 1.13, where
a −63.5 % H2D cut moved TTFT only −4.3…−5.4 %). A −6.7…−8.0 % TTFT cut from the dequant path
alone is the strongest single-path prefill result in this series.

## 7. Verdict (mission step 8)

**PASS — a meaningful, bit-identical, opt-in improvement.** It halves the largest
non-GEMM prefill kernel (the dequant L1-pipe bottleneck) with no numerics change, for a
measured **−6.7 % (16 K) / −8.0 % (32 K) TTFT** and unchanged decode. Kept OFF-by-default
(`STRATA_MOE_DQ_WIDE=1`) per the scope guard; the production 32 K / int8 / 8 K-cache config is
restored with the flag off.

**Documented, not done (mission step 8 — do not force a win):** the wide GU kernel still
plateaus ≈2× above the 15.4 µs memory roofline (ncu: L1/TEX 69 % + 50 %-capped occupancy,
1 warp/block). A further GU redesign (128-thread blocks / 2+ warps/block) is a follow-up
stage, not part of this isolated one-change optimization. The wall gain also leaves headroom:
dequant is now ~17 % of total kernel time (was ~29 %), so the next-largest levers (GEMM,
attn/other) become relatively more important.

## 8. Restore (mission step 9)

`strata.service` restarted on the production config (32 K / int8 KV / port 8180, E5 OFF,
E8 OFF, 8 K expert cache, `STRATA_MOE_DQ_WIDE` OFF/default) and verified live
(`/health` + a live request). See STATE.md.

## 9. Deliverables

- Engine: `src/kernels/cuda/iq_kernels.cu` (wide dequantizers + kernels + launchers),
  `include/strata/kernels/iq_kernels.hpp`, `src/prefill/prefill.cpp` (flag + dispatch).
- Bench: `bench/v100/s114_wide_kernel_test.cu` (microbench + bit-identity + host oracle),
  `s114_dq_microbench.cu` / `s114_dq_roofline.cu` (narrow microbench + roofline),
  `s114nsys-{16k,32k}-{off,on}.sh` (nsys captures), `s112nsys_analyze.py` (A/B analyzer).
- Data: `Logs/gpu/s114nsys-{16k,32k}-{off,on}.sqlite`, `Logs/benchmarks/s114ab/` (A/B
  client + engine-ttft), `Logs/gpu/s114g{16k,32k}-{off,on}-*.{out,json}` (gate runs).
- Doc: this file; STATE.md; CHANGELOG.md.
