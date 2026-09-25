# V100 GPU Analysis — Strata MoE (Stage 1.1)

Scope: kernel-level GPU behavior of the production (native pack) config on
GPU0 (V100 32 GB PCIE, SM70), decode and prefill. All data from nsys
(NSight Systems 2025.1.3) profiles in `Logs/gpu/`:

- `nsys-baseline-256tok.nsys-rep` — graph-level CUDA tracing (run: 36.35 tok/s).
- `nsys-node-256tok.nsys-rep` — `--cuda-graph-trace=node`, in-graph kernels
  (run: 32.36 tok/s; node tracing adds ~18 % overhead vs 39.55 unprofiled).

## Why not the built-in `--gpu-only-full` / `--gpu-stages`

Those flags replay the **session graphs** (48× pre/post), but the native pack
runs verify windows only (`session_capture` is skipped when
`expert_layout().native`, `src/program/generate.cpp` L1253:
"plan v0.3 P6: a native pack runs verify windows only"). In the production
config there is no session graph to replay — the flags exit with
`session_replay_full: not captured with post graphs`. nsys is the
mode-independent measurement; the numbers below are the real per-token GPU
floor.

## Decode (256 tokens, workers 28)

Round structure: 107 spec rounds × 4 drafts, 2.39 tokens/round,
25.29 ms/token wall (engine clock). Per round the engine runs:

| graph | execs | busy | note |
|---|---:|---:|---|
| T4 verify window (4 tokens) | 50 | 3,248 ms | 65.0 ms avg |
| T3 verify window | 20 | 1,126 ms | 56.3 ms avg |
| T2 verify window | 19 | 820 ms | 43.2 ms avg |
| T1 verify window | 18 | 625 ms | 34.7 ms avg |
| per-round token graph | 107 | 90 ms | 0.84 ms avg |
| MTP draft / commit graphs | 265 | 217 ms | 0.2–1.0 ms each |

(The verify-graph exec counts match the engine's window distribution
`T1:18 T2:19 T3:20 T4:50` exactly.)

**GPU busy = 6,127 ms of a 7,221 ms decode span (84.8 %; 85.6 % in the
node-trace run: 6,772 / 7,912 ms).** Decode is GPU-busy-bound: the device is
saturated, not waiting between kernels (only 13 ms of sub-µs micro-gaps in
the span).

### In-graph kernel breakdown (node trace, 7,912 ms decode window)

| kernel | total | share | role |
|---|---:|---:|---|
| `wait_flag_ge_kernel` | 2,515 ms | 37 % | device spin-wait on the CPU pool's doorbell flag |
| `gr_down_multi_kernel` | 529 ms | 8 % | GDN recurrent step, down proj (multi-token) |
| `gr_up_multi_kernel` | 283 ms | 4 % | GDN up proj |
| `bf16_f32_mmvf_kernel<256>` | 223 ms | 3 % | native dense rows (BF16) |
| `native_down_kernel<42/20>` | 353 ms | 5 % | native down projections |
| `copy_from_mapped_kernel` | 156 ms | 2 % | zero-copy read of pinned host memory |
| `gr_norm_multi_kernel` | 153 ms | 2 % | GDN norm |
| `native_gu_kernel<…>` | 435 ms | 6 % | native gate/up projections |
| `route` | 146 ms | 2 % | expert routing |
| `copy_i32_from_mapped_kernel` | 130 ms | 2 % | zero-copy int read |
| `mmvq`/`native_mmvq_multi` | 344 ms | 5 % | quantized mmv |
| rest (dequant, swiglu, KV, LM head) | ~1,000 ms | 15 % | |

`wait_flag_ge` by graph: the T4 window graph carries a single node averaging
**12.2 ms of GPU-side spin per round** (waiting for the CPU pool to drain the
window's expert positions), plus ~40 per-layer waits of 0.5–1 ms each.

### Consequences

1. **The verify-window graphs are 95 % of all GPU busy time.** A 1-token window
   costs 34.7 ms; the main single-token graph is 0.84 ms. Verification
   re-runs the full 48-layer forward per window on the device (GDN multi
   kernels, dense projections, routing, waits).
2. **A third of that busy time is the GPU spinning for the CPU pool.** While
   `wait_flag_ge` occupies the stream, nothing else on it can progress.
3. V100/SM70 specifics: the fused GEMMs run as
   `cutlass_70_wmma_tensorop_s161616` (16×16 / 32×32 WMMA, align8) — no
   `ldmatrix`/Ampere MMA; the cuBLAS SGEMM path (prefill) is the
   `magma_sgemmEx` bf16 kernel. There is no native int8 tensor path for the
   fused kernels; KV/attention uses fp16 WMMA.

## Prefill (28-token prompt, 1 chunk, 1,423 ms wall)

- 25,487 direct (non-graph) kernels in a 1,230 ms span; **968 ms busy (79 %)**.
- Top: `magma_sgemmEx` (cuBLAS batched SGEMM) 396 ms / 432 launches
  (40.9 %), `dequant_gu_kernel` 197 ms, `dequant_flat_kernel` 94 ms,
  cutlass WMMA GEMMs 84 ms, `dequant_kernel` (large weight dequants) 97 ms.
- The prefill is ~2/3 GPU work, ~1/3 PLE SSD latency on the critical path
  (PLE 2,801 ms of 6,934 ms for the 2,047-token prompt — see
  `v100-performance.md`).

## 16 GB card (GPU4)

Same build/config, workers 24: decode 36.6 tok/s, cache 6,321 slots
(10.23 GiB), peak 16.13 GiB. No CUDA errors; the smaller resident set lowers
the hit rate (84.6 % vs 87.6 %) and shifts a little work to the CPU pool —
consistent with the 32 GB card being slightly faster, as in Stage 1.

## Verdict

Decode on V100 is **GPU-busy-bound (85 % utilization)**, and the dominant
GPU work is the spec **verify-window graphs** (95 % of busy), of which ~37 %
is the device spin-wait for the CPU expert pool (`wait_flag_ge_kernel`,
2.5 s per 256-token decode). The remaining GPU time is GDN recurrent kernels
(~15 %), native dense projections (~11 %), zero-copy host reads (~4 %) and
routing/dequant/KV (~33 %). This is an architectural limit of the
verify-window design on SM70 — the low-risk config knobs (see
`v100-performance.md`) do not move it; the engine-level levers are
(i) cheaper verify windows, (ii) a faster pool drain (22.3 GB/s → closer to
memory-bandwidth ceiling), (iii) replacing the flag spin with CUDA
events/graph dependencies.
