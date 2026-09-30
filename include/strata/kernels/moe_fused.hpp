// include/strata/kernels/moe_fused.hpp - Stage 1.12 E5 (opt-in): the prefill expert GEMMs with the
// weight dequant folded into the GEMM, so the intermediate f16 weight buffers (dq_gu/dq_d) and their
// global-memory round trip disappear.  Enabled per run by STRATA_MOE_DEQUANT_GEMM_FUSE=1.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// Stage 1.12 E5.  `moe_fused_gemm_gu`: Y[T, 2*n_ff] (f32, ldy) = X[T, n_embd] (f16, ldx) . W^T, where
/// W[2*n_ff, n_embd] is dequantized IN-KERNEL from the native GGUF expert blob: W row 2r is gate row r
/// (from `gate`), W row 2r+1 is up row r (from `up`) - the exact layout the separate dequant_gu kernel
/// wrote, so the downstream swiglu is unchanged.  `moe_fused_gemm_d`: Y[T, n_embd] (f32) = X[T, n_ff]
/// (f16) . W^T with W[n_embd, n_ff] dequantized from `down` (the flat GGUF layout).
///
/// The dequant uses llama.cpp's own per-type formulas (bit-identical to the block-wide dq_* paths in
/// iq_kernels.cu: the same scale math, the same codebooks, the same f16 rounding).  The product runs on
/// the tensor cores with an ascending-K, no-split-K accumulation - the same hardware MMA and K order as
/// the cuBLAS path it replaces.  Supported types: gate/up 16/17/18/21/22, down 20/42 (this model's set).
/// T is any positive value (the grid pads the last 32-row tile; the padded Y rows land in this expert's
/// own scratch and are overwritten by the next expert's GEMM in stream order, so the epilogue masks
/// them instead of relying on buffer slack).
void moe_fused_gemm_gu(int gu_type, const void* gate, const void* up, int64_t T, int64_t n_ff, int64_t n_embd,
                       const uint16_t* X, int64_t ldx, float* Y, int64_t ldy, void* stream);
void moe_fused_gemm_d(int d_type, const void* down, int64_t T, int64_t n_ff, int64_t n_embd,
                      const uint16_t* X, int64_t ldx, float* Y, int64_t ldy, void* stream);

}  // namespace strata::kernels
