// include/strata/kernels/moe_fused_iq.hpp - Stage 1.12 E5: the per-value i-quant dequantizer shared by the
// fused dequant+GEMM kernel (moe_fused.cu) and the parity test (bench/v100/e5_dequant_parity.cu).
//
// `iq_value<TYPE>(vb, pos)` dequantizes ONE value: `vb` points at the quant unit (block) that CONTAINS
// `pos` (0-based inside the unit).  The math is transcribed value-by-value from the block-wide dq_*
// functions in iq_kernels.cu (same scale math, same codebooks, same float arithmetic), so it is
// bit-identical to the separate dequant path.  Requires ggml-common.h to be included first with
// GGML_COMMON_DECL_CUDA / GGML_COMMON_IMPL_CUDA so the block structs and codebook LUTs are visible.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

template<int TYPE> __device__ __forceinline__ float iq_value(const void* vb, int pos);

template<> __device__ __forceinline__ float iq_value<16>(const void* vb, int pos) {
    const block_iq2_xxs* x = (const block_iq2_xxs*) vb;
    const int ib = pos / 32, il = (pos % 32) / 8, j = pos % 8;
    const uint16_t* q2 = x->qs + 4 * ib;
    const uint8_t* aux8 = (const uint8_t*) q2;
    const uint8_t* grid = (const uint8_t*) (iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float) x->d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    return d * (float) grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}
template<> __device__ __forceinline__ float iq_value<17>(const void* vb, int pos) {
    const block_iq2_xs* x = (const block_iq2_xs*) vb;
    const int ib = pos / 32, il = (pos % 32) / 8, j = pos % 8;
    const uint16_t* q2 = x->qs + 4 * ib;
    const uint8_t* grid = (const uint8_t*) (iq2xs_grid + (q2[il] & 511));
    const float d = (float) x->d * (0.5f + ((x->scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    return d * (float) grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}
template<> __device__ __forceinline__ float iq_value<18>(const void* vb, int pos) {
    const block_iq3_xxs* x = (const block_iq3_xxs*) vb;
    const int ib = pos / 32, il = (pos % 32) / 8, j = pos % 8;
    const uint8_t* q3 = x->qs + 8 * ib;
    const uint16_t* gas = (const uint16_t*) (x->qs + QK_K / 4) + 2 * ib;
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float) x->d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    if (j < 4) {
        const uint8_t* g1 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 0]);
        return d * (float) g1[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
    }
    const uint8_t* g2 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 1]);
    return d * (float) g2[j - 4] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}
template<> __device__ __forceinline__ float iq_value<21>(const void* vb, int pos) {
    const block_iq3_s* x = (const block_iq3_s*) vb;
    const int ib = pos / 32, il = (pos % 32) / 8, j = pos % 8;
    const uint8_t* qs = x->qs + 8 * ib;
    const float d = (float) x->d * (1 + 2 * ((x->scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint8_t signs = x->signs[4 * ib + il];
    if (j < 4) {
        const uint8_t* g1 = (const uint8_t*) (iq3s_grid + (qs[2 * il + 0] | ((x->qh[ib] << (8 - 2 * il)) & 256)));
        return d * (float) g1[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
    }
    const uint8_t* g2 = (const uint8_t*) (iq3s_grid + (qs[2 * il + 1] | ((x->qh[ib] << (7 - 2 * il)) & 256)));
    return d * (float) g2[j - 4] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}
template<> __device__ __forceinline__ float iq_value<22>(const void* vb, int pos) {
    const block_iq2_s* x = (const block_iq2_s*) vb;
    const int ib = pos / 32, il = (pos % 32) / 8, j = pos % 8;
    const uint8_t* grid = (const uint8_t*) (iq2s_grid + (x->qs[4 * ib + il] | ((x->qh[ib] << (8 - 2 * il)) & 0x300)));
    const float d = (float) x->d * (0.5f + ((x->scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = x->qs[QK_K / 8 + 4 * ib + il];
    return d * (float) grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}
// 32-value unit (the down path of this model).
template<> __device__ __forceinline__ float iq_value<20>(const void* vb, int pos) {
    const block_iq4_nl* x = (const block_iq4_nl*) vb;
    const float d = (float) x->d;
    return (pos < 16) ? d * (float) kvalues_iq4nl[x->qs[pos] & 0xf]
                      : d * (float) kvalues_iq4nl[x->qs[pos - 16] >> 4];
}
// 64-value unit (the down path of this model).
template<> __device__ __forceinline__ float iq_value<42>(const void* vb, int pos) {
    const block_q2_0* x = (const block_q2_0*) vb;
    const int code = (x->qs[pos / 4] >> ((pos % 4) * 2)) & 3;
    return (float) x->d * (float) (code - 1);
}

// Per-type quant-unit geometry: how many values a unit holds and how many bytes it occupies.
template<int TYPE> struct Unit;
template<> struct Unit<16> { static constexpr int qv = 256; static constexpr size_t bytes = sizeof(block_iq2_xxs); };
template<> struct Unit<17> { static constexpr int qv = 256; static constexpr size_t bytes = sizeof(block_iq2_xs); };
template<> struct Unit<18> { static constexpr int qv = 256; static constexpr size_t bytes = sizeof(block_iq3_xxs); };
template<> struct Unit<21> { static constexpr int qv = 256; static constexpr size_t bytes = sizeof(block_iq3_s); };
template<> struct Unit<22> { static constexpr int qv = 256; static constexpr size_t bytes = sizeof(block_iq2_s); };
template<> struct Unit<20> { static constexpr int qv = 32;  static constexpr size_t bytes = sizeof(block_iq4_nl); };
template<> struct Unit<42> { static constexpr int qv = 64;  static constexpr size_t bytes = sizeof(block_q2_0); };

}  // namespace strata::kernels
