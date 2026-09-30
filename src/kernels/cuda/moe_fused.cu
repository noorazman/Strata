// src/kernels/cuda/moe_fused.cu - see include/strata/kernels/moe_fused.hpp.
//
// Stage 1.12 E5 (opt-in, STRATA_MOE_DEQUANT_GEMM_FUSE=1): the prefill expert GEMM with the weight
// dequant folded in.  The two-stage path (dequant_gu/dequant_flat -> f16 weight buffer -> cuBLAS GEMM)
// writes each expert's dequantized weights to global memory and immediately reads them back; this
// kernel reads the packed GGUF blob directly, dequantizes weight tiles into shared memory with
// llama.cpp's own per-value formulas (bit-identical to the block-wide dq_* functions in iq_kernels.cu),
// and runs a tensor-core f16/f32 GEMM with an ascending-K, no-split-K accumulation - the same hardware
// MMA and K order as the cuBLAS path it replaces.
#include "strata/kernels/moe_fused.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <mma.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include "strata/kernels/moe_fused_iq.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "moe_fused: %s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

using namespace nvcuda;

// ---------------------------------------------------------------- the fused kernel
// CTA: 32 weight rows x 32 token rows, 128 threads (4 warps); each warp owns one 16x16 output tile.
// The weight stripe (32 x KT) is dequantized from the packed blob into shared memory, then the K stripe
// runs through the tensor cores.  GU: KT = 256 (10 stripes for n_embd = 2560); D: KT = n_ff (one stripe).
template<int TYPE, int KT, bool GU>
__global__ void fused_dq_gemm_kernel(const void* __restrict__ s0, const void* __restrict__ s1,
                                     int64_t K, const uint16_t* __restrict__ X, int64_t ldx,
                                     float* __restrict__ Y, int64_t ldy, int64_t T) {
    __shared__ __half ws[32 * KT];
    constexpr int QV = Unit<TYPE>::qv;
    constexpr size_t UB = Unit<TYPE>::bytes;
    const int tid = threadIdx.x;
    const int64_t m0 = (int64_t) blockIdx.x * 32, t0 = (int64_t) blockIdx.y * 32;
    const int w = tid >> 5, wm = w & 1, wt = w >> 1;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> C;
    wmma::fill_fragment(C, 0.0f);
    const int64_t n_stripes = K / KT;
    const int64_t per_row_units = K / QV;
    for (int64_t s = 0; s < n_stripes; ++s) {
        for (int idx = tid; idx < 32 * KT; idx += 128) {
            const int k = idx % KT;
            const int64_t mrow = m0 + (int64_t) (idx / KT);
            const int64_t region_row = GU ? (mrow >> 1) : mrow;
            const void* base = (GU && (mrow & 1)) ? s1 : s0;
            const int k_abs = (int) (s * KT + k);
            const int64_t unit = region_row * per_row_units + (int64_t) (k_abs / QV);
            const float v = iq_value<TYPE>((const uint8_t*) base + unit * UB, k_abs % QV);
            ws[idx] = __float2half(v);
        }
        __syncthreads();
        for (int k0 = 0; k0 < KT; k0 += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> A;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> B;
            wmma::load_matrix_sync(A, (const __half*) (X + (t0 + (int64_t) wt * 16) * ldx + s * KT + k0), (unsigned) ldx);
            wmma::load_matrix_sync(B, &ws[(size_t) wm * 16 * KT + k0], (unsigned) KT);
            wmma::mma_sync(C, A, B, C);
        }
        __syncthreads();
    }
    // epilogue: masked copy (T may not be a multiple of 32 - the padded rows are this expert's own
    // scratch, but the mask keeps the write exactly [o0, o0+T)).  The 16x16 f32 tile is staged in this
    // warp's slice of the weight tile: ws is a __half array no longer read after the last stripe's
    // __syncthreads__, and its bytes are a superset of 4 warps x 256 floats (GU: 16 KB, D: 40 KB).
    // (A local `float buf[16*16]` gets its stores eliminated by the sm_70 NVVM frontend - the wmma
    // store_matrix_sync + masked float4 copy vanished from the PTX and a bare `trap` replaced the
    // epilogue; staging in shared memory is the fix verified by the v_shbuf bisection.)
    const int64_t nr = T - t0;
    if (nr <= 0) return;
    float* buf = (float*) ws + (size_t) (tid >> 5) * (16 * 16);
    wmma::store_matrix_sync(buf, C, 16, wmma::mem_row_major);
    const int64_t t_base = (int64_t) wt * 16, m_base = (int64_t) wm * 16;
    for (int i = 0; i < 16; ++i) {
        const int64_t t = t_base + i;
        if (t < nr) {
            float* y = Y + ((t0 + t) * ldy + m0 + m_base);
            const float* b = buf + (size_t) i * 16;
            for (int m = 0; m < 16; m += 4) {
                float4 v = *reinterpret_cast<const float4*>(b + m);
                *reinterpret_cast<float4*>(y + m) = v;
            }
        }
    }
}

template<int TYPE, int KT, bool GU>
void launch(int64_t M, int64_t K, const void* s0, const void* s1, int64_t T, const uint16_t* X, int64_t ldx,
            float* Y, int64_t ldy, cudaStream_t st) {
    dim3 grid((unsigned) (M / 32), (unsigned) ((T + 31) / 32));
    fused_dq_gemm_kernel<TYPE, KT, GU><<<grid, 128, 0, st>>>(s0, s1, K, X, ldx, Y, ldy, T);
    check(GU ? "moe_fused_gemm_gu" : "moe_fused_gemm_d");
}

}  // namespace

void moe_fused_gemm_gu(int gu_type, const void* gate, const void* up, int64_t T, int64_t n_ff, int64_t n_embd,
                       const uint16_t* X, int64_t ldx, float* Y, int64_t ldy, void* stream) {
    if (T <= 0) return;
    const cudaStream_t st = (cudaStream_t) stream;
    const int64_t M = 2 * n_ff;
    switch (gu_type) {
        case 16: launch<16, 256, true>(M, n_embd, gate, up, T, X, ldx, Y, ldy, st); break;
        case 17: launch<17, 256, true>(M, n_embd, gate, up, T, X, ldx, Y, ldy, st); break;
        case 18: launch<18, 256, true>(M, n_embd, gate, up, T, X, ldx, Y, ldy, st); break;
        case 21: launch<21, 256, true>(M, n_embd, gate, up, T, X, ldx, Y, ldy, st); break;
        case 22: launch<22, 256, true>(M, n_embd, gate, up, T, X, ldx, Y, ldy, st); break;
        default: std::fprintf(stderr, "moe_fused_gemm_gu: type %d is not supported\n", gu_type); std::exit(1);
    }
}

void moe_fused_gemm_d(int d_type, const void* down, int64_t T, int64_t n_ff, int64_t n_embd,
                      const uint16_t* X, int64_t ldx, float* Y, int64_t ldy, void* stream) {
    if (T <= 0) return;
    const cudaStream_t st = (cudaStream_t) stream;
    switch (d_type) {
        case 20: launch<20, 640, false>(n_embd, n_ff, down, nullptr, T, X, ldx, Y, ldy, st); break;
        case 42: launch<42, 640, false>(n_embd, n_ff, down, nullptr, T, X, ldx, Y, ldy, st); break;
        default: std::fprintf(stderr, "moe_fused_gemm_d: type %d is not supported\n", d_type); std::exit(1);
    }
}

}  // namespace strata::kernels
