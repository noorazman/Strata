// src/kernels/cpu/iq_avx2.cpp - Stage 1.4: AVX2 multi-token i-quant expert rows (decode-once).
//
// ggml-cpu's x86 AVX2 dot products for the i-quant formats (arch/x86/quants.c) are single-token: every
// token re-decodes the weight codebook - 8 to 16 grid lookups, sign expansion, scale extraction - per
// 64/32-value sub-block.  In the expert pool every row is dotted against nt >= 2 tokens, so that decode
// work is nt-fold redundant.  Here it is computed once per (row, block, sub-block) and shared across all
// tokens; the per-(row, token) integer math (sign apply, maddubs, madd, the per-sub-block int sums) and
// the float accumulation (per-block d scaling, the running sum across blocks, the final hsum) are the
// exact op sequence of the ggml-cpu AVX2 kernel, so each result is bit-identical to the per-token path.
//
// Formats: IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21), IQ2_S (22).
#include "strata/kernels/cpu/iq_avx2.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <cpuid.h>
#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cmath>
#include <cstring>
#include <algorithm>

// ggml-cpu keeps fp16->fp32 in a 65536-entry lookup table (defined in ggml-cpu.c, not declared in a header);
// on x86 GGML_CPU_FP16_TO_FP32 is exactly that table, so bit-identity with the ggml-cpu AVX2 dots requires it
// (not a hardware conversion).  Global scope - do not move into a namespace.
extern "C" {
extern float ggml_table_f32_f16[];
}

namespace strata::kernels::cpu {
namespace {

inline float fp16f(ggml_half h) {
    uint16_t s;
    std::memcpy(&s, &h, sizeof s);
    return ggml_table_f32_f16[s];
}

// ggml-cpu's AVX2 hsum_float_8 (arch/x86/quants.c) - same reduction order, bit-identical.
inline float hsum8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// Copy of keven_signs_q2xs[1024] from ggml-cpu/arch/x86/quants.c (file-static there; used by the IQ2_XXS
// and IQ3_XXS sign expansion).
static const int8_t kSigns[1024] = {
        1,    1,    1,    1,    1,    1,    1,    1,   -1,    1,    1,    1,    1,    1,    1,   -1,    1,   -1,    1,    1,    1,    1,    1,   -1,   -1,   -1,    1,    1,    1,    1,    1,    1,
        1,    1,   -1,    1,    1,    1,    1,   -1,   -1,    1,   -1,    1,    1,    1,    1,    1,    1,   -1,   -1,    1,    1,    1,    1,    1,   -1,   -1,   -1,    1,    1,    1,    1,   -1,
        1,    1,    1,   -1,    1,    1,    1,   -1,   -1,    1,    1,   -1,    1,    1,    1,    1,    1,   -1,    1,   -1,    1,    1,    1,    1,   -1,   -1,    1,   -1,    1,    1,    1,   -1,
        1,    1,   -1,   -1,    1,    1,    1,    1,   -1,    1,   -1,   -1,    1,    1,    1,   -1,    1,   -1,   -1,   -1,    1,    1,    1,   -1,   -1,   -1,   -1,   -1,    1,    1,    1,    1,
        1,    1,    1,    1,   -1,    1,    1,   -1,   -1,    1,    1,    1,   -1,    1,    1,    1,    1,   -1,    1,    1,   -1,    1,    1,    1,   -1,   -1,    1,    1,   -1,    1,    1,   -1,
        1,    1,   -1,    1,   -1,    1,    1,    1,   -1,    1,   -1,    1,   -1,    1,    1,   -1,    1,   -1,   -1,    1,   -1,    1,    1,   -1,   -1,   -1,   -1,    1,   -1,    1,    1,    1,
        1,    1,    1,   -1,   -1,    1,    1,    1,   -1,    1,    1,   -1,   -1,    1,    1,   -1,    1,   -1,    1,   -1,   -1,    1,    1,   -1,   -1,   -1,    1,   -1,   -1,    1,    1,    1,
        1,    1,   -1,   -1,   -1,    1,    1,   -1,   -1,    1,   -1,   -1,   -1,    1,    1,    1,    1,   -1,   -1,   -1,   -1,    1,    1,    1,   -1,   -1,   -1,   -1,   -1,    1,    1,   -1,
        1,    1,    1,    1,    1,   -1,    1,   -1,   -1,    1,    1,    1,    1,   -1,    1,    1,    1,   -1,    1,    1,    1,   -1,    1,    1,   -1,   -1,    1,    1,    1,   -1,    1,   -1,
        1,    1,   -1,    1,    1,   -1,    1,    1,   -1,    1,   -1,    1,    1,   -1,    1,   -1,    1,   -1,   -1,    1,    1,   -1,    1,   -1,   -1,   -1,   -1,    1,    1,   -1,    1,    1,
        1,    1,    1,   -1,    1,   -1,    1,    1,   -1,    1,    1,   -1,    1,   -1,    1,   -1,    1,   -1,    1,   -1,    1,   -1,    1,   -1,   -1,   -1,    1,   -1,    1,   -1,    1,    1,
        1,    1,   -1,   -1,    1,   -1,    1,   -1,   -1,    1,   -1,   -1,    1,   -1,    1,    1,    1,   -1,   -1,   -1,    1,   -1,    1,    1,   -1,   -1,   -1,   -1,    1,   -1,    1,   -1,
        1,    1,    1,    1,   -1,   -1,    1,    1,   -1,    1,    1,    1,   -1,   -1,    1,   -1,    1,   -1,    1,    1,   -1,   -1,    1,   -1,   -1,   -1,    1,    1,   -1,   -1,    1,    1,
        1,    1,   -1,    1,   -1,   -1,    1,   -1,   -1,    1,   -1,    1,   -1,   -1,    1,    1,    1,   -1,   -1,    1,   -1,   -1,    1,    1,   -1,   -1,   -1,    1,   -1,   -1,    1,   -1,
        1,    1,    1,   -1,   -1,   -1,    1,   -1,   -1,    1,    1,   -1,   -1,   -1,    1,    1,    1,   -1,    1,   -1,   -1,   -1,    1,    1,   -1,   -1,    1,   -1,   -1,   -1,    1,   -1,
        1,    1,   -1,   -1,   -1,   -1,    1,    1,   -1,    1,   -1,   -1,   -1,   -1,    1,   -1,    1,   -1,   -1,   -1,   -1,   -1,    1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,    1,    1,
        1,    1,    1,    1,    1,    1,   -1,   -1,   -1,    1,    1,    1,    1,    1,   -1,    1,    1,   -1,    1,    1,    1,    1,   -1,    1,   -1,   -1,    1,    1,    1,    1,   -1,   -1,
        1,    1,   -1,    1,    1,    1,   -1,    1,   -1,    1,   -1,    1,    1,    1,   -1,   -1,    1,   -1,   -1,    1,    1,    1,   -1,   -1,   -1,   -1,   -1,    1,    1,    1,   -1,    1,
        1,    1,    1,   -1,    1,    1,   -1,    1,   -1,    1,    1,   -1,    1,    1,   -1,   -1,    1,   -1,    1,   -1,    1,    1,   -1,   -1,   -1,   -1,    1,   -1,    1,    1,   -1,    1,
        1,    1,   -1,   -1,    1,    1,   -1,   -1,   -1,    1,   -1,   -1,    1,    1,   -1,    1,    1,   -1,   -1,   -1,    1,    1,   -1,    1,   -1,   -1,   -1,   -1,    1,    1,   -1,   -1,
        1,    1,    1,    1,   -1,    1,   -1,    1,   -1,    1,    1,    1,   -1,    1,   -1,   -1,    1,   -1,    1,    1,   -1,    1,   -1,   -1,   -1,   -1,    1,    1,   -1,    1,   -1,    1,
        1,    1,   -1,    1,   -1,    1,   -1,   -1,   -1,    1,   -1,    1,   -1,    1,   -1,    1,    1,   -1,   -1,    1,   -1,    1,   -1,    1,   -1,   -1,   -1,    1,   -1,    1,   -1,   -1,
        1,    1,    1,   -1,   -1,    1,   -1,   -1,   -1,    1,    1,   -1,   -1,    1,   -1,    1,    1,   -1,    1,   -1,   -1,    1,   -1,    1,   -1,   -1,    1,   -1,   -1,    1,   -1,   -1,
        1,    1,   -1,   -1,   -1,    1,   -1,    1,   -1,    1,   -1,   -1,   -1,    1,   -1,   -1,    1,   -1,   -1,   -1,   -1,    1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,    1,   -1,    1,
        1,    1,    1,    1,    1,   -1,   -1,    1,   -1,    1,    1,    1,    1,   -1,   -1,   -1,    1,   -1,    1,    1,    1,   -1,   -1,   -1,   -1,   -1,    1,    1,    1,   -1,   -1,    1,
        1,    1,   -1,    1,    1,   -1,   -1,   -1,   -1,    1,   -1,    1,    1,   -1,   -1,    1,    1,   -1,   -1,    1,    1,   -1,   -1,    1,   -1,   -1,   -1,    1,    1,   -1,   -1,   -1,
        1,    1,    1,   -1,    1,   -1,   -1,   -1,   -1,    1,    1,   -1,    1,   -1,   -1,    1,    1,   -1,    1,   -1,    1,   -1,   -1,    1,   -1,   -1,    1,   -1,    1,   -1,   -1,   -1,
        1,    1,   -1,   -1,    1,   -1,   -1,    1,   -1,    1,   -1,   -1,    1,   -1,   -1,   -1,    1,   -1,   -1,   -1,    1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,    1,   -1,   -1,    1,
        1,    1,    1,    1,   -1,   -1,   -1,   -1,   -1,    1,    1,    1,   -1,   -1,   -1,    1,    1,   -1,    1,    1,   -1,   -1,   -1,    1,   -1,   -1,    1,    1,   -1,   -1,   -1,   -1,
        1,    1,   -1,    1,   -1,   -1,   -1,    1,   -1,    1,   -1,    1,   -1,   -1,   -1,   -1,    1,   -1,   -1,    1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,    1,   -1,   -1,   -1,    1,
        1,    1,    1,   -1,   -1,   -1,   -1,    1,   -1,    1,    1,   -1,   -1,   -1,   -1,   -1,    1,   -1,    1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,    1,   -1,   -1,   -1,   -1,    1,
        1,    1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,    1,   -1,   -1,   -1,   -1,   -1,    1,    1,   -1,   -1,   -1,   -1,   -1,   -1,    1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
};

// ---- IQ2_XXS (16): block = d(2) + qs[32] u16 (64 B) = 66 B ------------------------------
// ggml_vec_dot_iq2_xxs_q8_K AVX2: 4 sub-blocks of 64 values; per sub-block 8 code lookups into
// iq2xxs_grid (uint64 entries, 8 x 2-bit values each) + 8 sign lookups from keven_signs_q2xs.
template <int NT>
void row_dot_iq2xxs(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    const uint64_t* signs64 = reinterpret_cast<const uint64_t*>(kSigns);
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * 66;
        const float dx = fp16f(*(const ggml_half*) blk);
        const uint16_t* q2 = (const uint16_t*) (blk + 2);
        __m256i sumi1[NT], sumi2[NT];
        for (int t = 0; t < NT; ++t) {
            sumi1[t] = _mm256_setzero_si256();
            sumi2[t] = _mm256_setzero_si256();
        }
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            uint32_t aux32[4];
            std::memcpy(aux32, q2, 16);
            q2 += 8;
            const uint8_t* aux8 = (const uint8_t*) aux32;
            const __m256i q2_1 = _mm256_set_epi64x(iq2xxs_grid[aux8[3]], iq2xxs_grid[aux8[2]],
                                                   iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]],
                                                   iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
            const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127],
                                                   signs64[(aux32[3] >>  7) & 127], signs64[(aux32[3] >>  0) & 127]);
            const __m256i psc1 = _mm256_set1_epi16(2 * (int) (aux32[1] >> 28) + 1);
            const __m256i psc2 = _mm256_set1_epi16(2 * (int) (aux32[3] >> 28) + 1);
            for (int t = 0; t < NT; ++t) {
                const int8_t* q8 = y[t][i].qs + ib32 * 32;
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) q8);
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) (q8 + 32));
                sumi1[t] = _mm256_add_epi32(sumi1[t],
                                            _mm256_madd_epi16(_mm256_maddubs_epi16(q2_1, _mm256_sign_epi8(q8_1, s2_1)), psc1));
                sumi2[t] = _mm256_add_epi32(sumi2[t],
                                            _mm256_madd_epi16(_mm256_maddubs_epi16(q2_2, _mm256_sign_epi8(q8_2, s2_2)), psc2));
            }
        }
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d),
                                      _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1[t], sumi2[t])), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = 0.125f * hsum8(accf[t]);
}

// ---- IQ2_XS (17): block = d(2) + qs[32] u16 (64 B) + scales[8] = 74 B -------------------
// ggml_vec_dot_iq2_xs_q8_K AVX2: 2 sub-blocks of 128 values; 16 code lookups into iq2xs_grid
// (uint64 entries, 8 x 4-bit values), signs expanded from the high bits of the same u16 codes,
// and 16 scales (4 per 32-value group) pre-computed per block.
template <int NT>
void row_dot_iq2xs(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    static const uint8_t k_bit_helper[32] = {
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
    };
    static const char block_sign_shuffle_mask_1[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    };
    static const char block_sign_shuffle_mask_2[32] = {
        0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e,
    };
    static const uint8_t bit_selector_mask_bytes[32] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };
    // get_scale_shuffle table from ggml-cpu arch/x86/quants.c.
    static const uint8_t k_shuffle[128] = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
        8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
        12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13,
        14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
    };
    const __m256i bit_helper = _mm256_loadu_si256((const __m256i*) k_bit_helper);
    const __m256i bit_selector_mask = _mm256_loadu_si256((const __m256i*) bit_selector_mask_bytes);
    const __m256i block_sign_shuffle_1 = _mm256_loadu_si256((const __m256i*) block_sign_shuffle_mask_1);
    const __m256i block_sign_shuffle_2 = _mm256_loadu_si256((const __m256i*) block_sign_shuffle_mask_2);
    const __m256i mone = _mm256_set1_epi8(1);
    const __m256i m511 = _mm256_set1_epi16(511);
    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);

    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * 74;
        const float dx = fp16f(*(const ggml_half*) blk);
        const uint16_t* q2 = (const uint16_t*) (blk + 2);
        uint64_t aux64;
        std::memcpy(&aux64, blk + 66, 8);
        __m128i stmp = _mm_set1_epi64x((long long) aux64);
        stmp = _mm_unpacklo_epi8(_mm_and_si128(stmp, m4), _mm_and_si128(_mm_srli_epi16(stmp, 4), m4));
        const __m128i scales = _mm_add_epi8(_mm_slli_epi16(stmp, 1), m1);
        __m256i sumi1[NT], sumi2[NT];
        for (int t = 0; t < NT; ++t) {
            sumi1[t] = _mm256_setzero_si256();
            sumi2[t] = _mm256_setzero_si256();
        }
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 4) {
            const __m256i q2_data = _mm256_loadu_si256((const __m256i*) q2);
            q2 += 16;
            const __m256i gindex_vec = _mm256_and_si256(q2_data, m511);
            const uint16_t* gindex = (const uint16_t*) &gindex_vec;
            const __m256i partial_sign_bits = _mm256_srli_epi16(q2_data, 9);
            const __m256i partial_sign_bits_upper = _mm256_srli_epi16(q2_data, 13);
            const __m256i partial_sign_bits_for_counting = _mm256_xor_si256(partial_sign_bits, partial_sign_bits_upper);
            const __m256i odd_bits = _mm256_shuffle_epi8(bit_helper, partial_sign_bits_for_counting);
            const __m256i full_sign_bits = _mm256_or_si256(partial_sign_bits, odd_bits);
            const __m256i q2_1 = _mm256_set_epi64x(iq2xs_grid[gindex[3]], iq2xs_grid[gindex[2]],
                                                   iq2xs_grid[gindex[1]], iq2xs_grid[gindex[0]]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2xs_grid[gindex[7]], iq2xs_grid[gindex[6]],
                                                   iq2xs_grid[gindex[5]], iq2xs_grid[gindex[4]]);
            const __m256i q2_3 = _mm256_set_epi64x(iq2xs_grid[gindex[11]], iq2xs_grid[gindex[10]],
                                                   iq2xs_grid[gindex[9]], iq2xs_grid[gindex[8]]);
            const __m256i q2_4 = _mm256_set_epi64x(iq2xs_grid[gindex[15]], iq2xs_grid[gindex[14]],
                                                   iq2xs_grid[gindex[13]], iq2xs_grid[gindex[12]]);
            const __m128i full_signs_l = _mm256_castsi256_si128(full_sign_bits);
            const __m128i full_signs_h = _mm256_extractf128_si256(full_sign_bits, 1);
            // The original uses MM256_SET_M128I(l, l): both 128-bit halves hold the SAME data.
            // _mm256_castsi128_si256 leaves the high half undefined (GCC materializes 0 here), and
            // VPSHUFB operates per 128-bit lane, so the codes-2/3 sign bytes would index into the
            // high half.  set_m128i(x, x) replicates it into both halves, as the original does.
            const __m256i full_signs_1 = _mm256_set_m128i(full_signs_l, full_signs_l);
            const __m256i full_signs_2 = _mm256_set_m128i(full_signs_h, full_signs_h);
            const __m256i signs_1 = _mm256_cmpeq_epi8(
                _mm256_and_si256(_mm256_shuffle_epi8(full_signs_1, block_sign_shuffle_1), bit_selector_mask), bit_selector_mask);
            const __m256i signs_2 = _mm256_cmpeq_epi8(
                _mm256_and_si256(_mm256_shuffle_epi8(full_signs_1, block_sign_shuffle_2), bit_selector_mask), bit_selector_mask);
            const __m256i signs_3 = _mm256_cmpeq_epi8(
                _mm256_and_si256(_mm256_shuffle_epi8(full_signs_2, block_sign_shuffle_1), bit_selector_mask), bit_selector_mask);
            const __m256i signs_4 = _mm256_cmpeq_epi8(
                _mm256_and_si256(_mm256_shuffle_epi8(full_signs_2, block_sign_shuffle_2), bit_selector_mask), bit_selector_mask);
            // the original does `_mm_loadu_si128((const __m128i*)k_shuffle + i)` - pointer arithmetic in 16-byte
            // units, i.e. a 16*i byte offset.  Cast first, then add.
            const __m256i sc1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i*) k_shuffle + ib32)));
            const __m256i sc2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i*) k_shuffle + ib32 + 1)));
            const __m256i sc3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i*) k_shuffle + ib32 + 2)));
            const __m256i sc4 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i*) k_shuffle + ib32 + 3)));
            for (int t = 0; t < NT; ++t) {
                const int8_t* q8 = y[t][i].qs + ib32 * 32;
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) q8);
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) (q8 + 32));
                const __m256i q8_3 = _mm256_loadu_si256((const __m256i*) (q8 + 64));
                const __m256i q8_4 = _mm256_loadu_si256((const __m256i*) (q8 + 96));
                const __m256i q8s_1 = _mm256_sign_epi8(q8_1, _mm256_or_si256(signs_1, mone));
                const __m256i q8s_2 = _mm256_sign_epi8(q8_2, _mm256_or_si256(signs_2, mone));
                const __m256i q8s_3 = _mm256_sign_epi8(q8_3, _mm256_or_si256(signs_3, mone));
                const __m256i q8s_4 = _mm256_sign_epi8(q8_4, _mm256_or_si256(signs_4, mone));
                const __m256i d1 = _mm256_maddubs_epi16(q2_1, q8s_1);
                const __m256i d2 = _mm256_maddubs_epi16(q2_2, q8s_2);
                const __m256i d3 = _mm256_maddubs_epi16(q2_3, q8s_3);
                const __m256i d4 = _mm256_maddubs_epi16(q2_4, q8s_4);
                sumi1[t] = _mm256_add_epi32(sumi1[t], _mm256_madd_epi16(d1, sc1));
                sumi2[t] = _mm256_add_epi32(sumi2[t], _mm256_madd_epi16(d2, sc2));
                sumi1[t] = _mm256_add_epi32(sumi1[t], _mm256_madd_epi16(d3, sc3));
                sumi2[t] = _mm256_add_epi32(sumi2[t], _mm256_madd_epi16(d4, sc4));
            }
        }
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d),
                                      _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1[t], sumi2[t])), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = 0.125f * hsum8(accf[t]);
}

// ---- IQ2_S (22): block = d(2) + qs[64] + qh[8] + scales[8] = 82 B -----------------------
// ggml_vec_dot_iq2_s_q8_K AVX2: 4 sub-blocks of 64 values; 8 code lookups into iq2s_grid
// (uint64 entries, 8 x 2-bit values) with 2 qh bits folded into the index, signs from the second
// half of qs (uint16), 16 scales pre-computed per block.
template <int NT>
void row_dot_iq2s(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03};
    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    // get_scale_shuffle_k4 table from ggml-cpu arch/x86/quants.c.
    static const uint8_t k_shuffle4[256] = {
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
        4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
        6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
        8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
        12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
        14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
    };
    const __m256i mask1 = _mm256_loadu_si256((const __m256i*) k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i*) k_mask2);
    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);

    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * 82;
        const float dx = fp16f(*(const ggml_half*) blk);
        const uint8_t* qs = blk + 2;
        const uint8_t* qh = blk + 66;
        const uint16_t* signs = (const uint16_t*) (blk + 2 + 32);
        uint64_t aux64;
        std::memcpy(&aux64, blk + 74, 8);
        const __m128i scales8 = _mm_add_epi8(_mm_slli_epi16(_mm_and_si128(_mm_set_epi64x((long long) (aux64 >> 4), (long long) aux64), m4), 1), m1);
        const __m256i scales16 = _mm256_cvtepi8_epi16(scales8);
        __m256i sumi1[NT], sumi2[NT];
        for (int t = 0; t < NT; ++t) {
            sumi1[t] = _mm256_setzero_si256();
            sumi2[t] = _mm256_setzero_si256();
        }
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m256i q2_1 = _mm256_set_epi64x(iq2s_grid[qs[3] | ((qh[ib32 + 0] << 2) & 0x300)],
                                                   iq2s_grid[qs[2] | ((qh[ib32 + 0] << 4) & 0x300)],
                                                   iq2s_grid[qs[1] | ((qh[ib32 + 0] << 6) & 0x300)],
                                                   iq2s_grid[qs[0] | ((qh[ib32 + 0] << 8) & 0x300)]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2s_grid[qs[7] | ((qh[ib32 + 1] << 2) & 0x300)],
                                                   iq2s_grid[qs[6] | ((qh[ib32 + 1] << 4) & 0x300)],
                                                   iq2s_grid[qs[5] | ((qh[ib32 + 1] << 6) & 0x300)],
                                                   iq2s_grid[qs[4] | ((qh[ib32 + 1] << 8) & 0x300)]);
            qs += 8;
            const __m256i s2_1 = _mm256_cmpeq_epi8(
                _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(signs[0] | ((uint32_t) signs[1] << 16)), mask1), mask2), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(
                _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(signs[2] | ((uint32_t) signs[3] << 16)), mask1), mask2), mask2);
            signs += 4;
            // 32-byte units (the original: `_mm256_loadu_si256((const __m256i*)k_shuffle + i)`)
            const __m256i psc1 = _mm256_shuffle_epi8(scales16, _mm256_loadu_si256((const __m256i*) k_shuffle4 + ib32));
            const __m256i psc2 = _mm256_shuffle_epi8(scales16, _mm256_loadu_si256((const __m256i*) k_shuffle4 + ib32 + 1));
            for (int t = 0; t < NT; ++t) {
                const int8_t* q8 = y[t][i].qs + ib32 * 32;
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) q8);
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) (q8 + 32));
                const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);
                const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);
                sumi1[t] = _mm256_add_epi32(sumi1[t], _mm256_madd_epi16(_mm256_maddubs_epi16(q2_1, q8s_1), psc1));
                sumi2[t] = _mm256_add_epi32(sumi2[t], _mm256_madd_epi16(_mm256_maddubs_epi16(q2_2, q8s_2), psc2));
            }
        }
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d),
                                      _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1[t], sumi2[t])), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = 0.125f * hsum8(accf[t]);
}

// ---- IQ3_XXS (18): block = d(2) + qs[32] grid + qs[32] gas = 98 B -----------------------
// ggml_vec_dot_iq3_xxs_q8_K AVX2: 4 sub-blocks of 64 values; 16 code lookups into iq3xxs_grid
// (uint32 entries, 4 x 4-bit values each) + 8 sign lookups from keven_signs_q2xs.
template <int NT>
void row_dot_iq3xxs(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    const uint64_t* signs64 = reinterpret_cast<const uint64_t*>(kSigns);
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * 98;
        const float dx = fp16f(*(const ggml_half*) blk);
        const uint8_t* q3 = blk + 2;
        const uint8_t* gas = blk + 2 + QK_K / 4;
        __m256i sumi1[NT], sumi2[NT];
        for (int t = 0; t < NT; ++t) {
            sumi1[t] = _mm256_setzero_si256();
            sumi2[t] = _mm256_setzero_si256();
        }
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m256i q2_1 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                  iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            const __m256i q2_2 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                  iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            uint32_t aux32[2];
            std::memcpy(aux32, gas, 8);
            gas += 8;
            const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[0] >> 21) & 127], signs64[(aux32[0] >> 14) & 127],
                                                   signs64[(aux32[0] >>  7) & 127], signs64[(aux32[0] >>  0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            const __m256i psc1 = _mm256_set1_epi16(2 * (int) (aux32[0] >> 28) + 1);
            const __m256i psc2 = _mm256_set1_epi16(2 * (int) (aux32[1] >> 28) + 1);
            for (int t = 0; t < NT; ++t) {
                const int8_t* q8 = y[t][i].qs + ib32 * 32;
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) q8);
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) (q8 + 32));
                sumi1[t] = _mm256_add_epi32(sumi1[t],
                                            _mm256_madd_epi16(_mm256_maddubs_epi16(q2_1, _mm256_sign_epi8(q8_1, s2_1)), psc1));
                sumi2[t] = _mm256_add_epi32(sumi2[t],
                                            _mm256_madd_epi16(_mm256_maddubs_epi16(q2_2, _mm256_sign_epi8(q8_2, s2_2)), psc2));
            }
        }
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d),
                                      _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1[t], sumi2[t])), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = 0.25f * hsum8(accf[t]);
}

// ---- IQ3_S (21): block = d(2) + qs[64] + qh[8] + signs[32] + scales[4] = 110 B ----------
// ggml_vec_dot_iq3_s_q8_K AVX2: 4 sub-blocks of 64 values; 16 code lookups into iq3s_grid
// (uint32 entries, 4 x 4-bit values) with one qh bit folded into the high byte of the index,
// signs from the signs[] half-words, 2 scales per sub-block from scales[].
template <int NT>
void row_dot_iq3s(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03};
    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    const __m256i mask1 = _mm256_loadu_si256((const __m256i*) k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i*) k_mask2);
    const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i idx_mask = _mm256_set1_epi32(256);
    typedef union {
        __m256i vec[2];
        uint32_t index[16];
    } index_t;
    index_t idx;

    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * 110;
        const float dx = fp16f(*(const ggml_half*) blk);
        const uint8_t* qs = blk + 2;
        const uint8_t* qh = blk + 66;
        const uint16_t* signs = (const uint16_t*) (blk + 74);
        __m256i sumi1[NT], sumi2[NT];
        for (int t = 0; t < NT; ++t) {
            sumi1[t] = _mm256_setzero_si256();
            sumi2[t] = _mm256_setzero_si256();
        }
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m128i qsl = _mm_loadu_si128((const __m128i*) qs);
            qs += 16;
            const __m256i idx_l = _mm256_cvtepu8_epi16(qsl);
            idx.vec[0] = _mm256_set1_epi32((int) qh[ib32 + 0]);
            idx.vec[1] = _mm256_set1_epi32((int) qh[ib32 + 1]);
            idx.vec[0] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[0], idx_shift), idx_mask);
            idx.vec[1] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[1], idx_shift), idx_mask);
            idx.vec[0] = _mm256_or_si256(idx.vec[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
            idx.vec[1] = _mm256_or_si256(idx.vec[1], _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));
            const __m256i q2_1 = _mm256_set_epi32(iq3s_grid[idx.index[7]], iq3s_grid[idx.index[6]], iq3s_grid[idx.index[5]],
                                                  iq3s_grid[idx.index[4]], iq3s_grid[idx.index[3]], iq3s_grid[idx.index[2]],
                                                  iq3s_grid[idx.index[1]], iq3s_grid[idx.index[0]]);
            const __m256i q2_2 = _mm256_set_epi32(iq3s_grid[idx.index[15]], iq3s_grid[idx.index[14]],
                                                  iq3s_grid[idx.index[13]], iq3s_grid[idx.index[12]],
                                                  iq3s_grid[idx.index[11]], iq3s_grid[idx.index[10]], iq3s_grid[idx.index[9]],
                                                  iq3s_grid[idx.index[8]]);
            const __m256i s2_1 = _mm256_cmpeq_epi8(
                _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(signs[0] | (signs[1] << 16)), mask1), mask2), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(
                _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(signs[2] | (signs[3] << 16)), mask1), mask2), mask2);
            signs += 4;
            const uint16_t ls1 = blk[106 + ib32 / 2] & 0xf;
            const uint16_t ls2 = blk[106 + ib32 / 2] >> 4;
            const __m256i psc1 = _mm256_set1_epi16(2 * (int) ls1 + 1);
            const __m256i psc2 = _mm256_set1_epi16(2 * (int) ls2 + 1);
            for (int t = 0; t < NT; ++t) {
                const int8_t* q8 = y[t][i].qs + ib32 * 32;
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) q8);
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) (q8 + 32));
                const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);
                const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);
                sumi1[t] = _mm256_add_epi32(sumi1[t], _mm256_madd_epi16(_mm256_maddubs_epi16(q2_1, q8s_1), psc1));
                sumi2[t] = _mm256_add_epi32(sumi2[t], _mm256_madd_epi16(_mm256_maddubs_epi16(q2_2, q8s_2), psc2));
            }
        }
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d),
                                      _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1[t], sumi2[t])), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(accf[t]);
}

// ---- dispatch -------------------------------------------------------------------------
template <int TY, int NT>
inline void row_dot_any(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    if (TY == 16) {
        row_dot_iq2xxs<NT>(row, nblocks, y, res);
    } else if (TY == 17) {
        row_dot_iq2xs<NT>(row, nblocks, y, res);
    } else if (TY == 18) {
        row_dot_iq3xxs<NT>(row, nblocks, y, res);
    } else if (TY == 21) {
        row_dot_iq3s<NT>(row, nblocks, y, res);
    } else {
        row_dot_iq2s<NT>(row, nblocks, y, res);
    }
}

template <int TY, int NT>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, float* const* ff,
             int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot_any<TY, NT>(blob + (size_t) r * gu_row, nb, y, g);
        row_dot_any<TY, NT>(blob + up_off + (size_t) r * gu_row, nb, y, u);
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

template <int TY, int NT>
void dot_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot_any<TY, NT>(w + (size_t) r * row_bytes, nb, y, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

template <int TY>
void gu_rows_nt(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                float* const* ff, int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int k = std::min(4, nt - t0);
        switch (k) {
            case 1: gu_rows<TY, 1>(blob, gu_row, up_off, n, act + t0, ff + t0, r0, r1); break;
            case 2: gu_rows<TY, 2>(blob, gu_row, up_off, n, act + t0, ff + t0, r0, r1); break;
            case 3: gu_rows<TY, 3>(blob, gu_row, up_off, n, act + t0, ff + t0, r0, r1); break;
            case 4: gu_rows<TY, 4>(blob, gu_row, up_off, n, act + t0, ff + t0, r0, r1); break;
        }
    }
}

template <int TY>
void dot_rows_nt(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0,
                 int r1) {
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int k = std::min(4, nt - t0);
        switch (k) {
            case 1: dot_rows<TY, 1>(w, row_bytes, n, act + t0, out + t0, r0, r1); break;
            case 2: dot_rows<TY, 2>(w, row_bytes, n, act + t0, out + t0, r0, r1); break;
            case 3: dot_rows<TY, 3>(w, row_bytes, n, act + t0, out + t0, r0, r1); break;
            case 4: dot_rows<TY, 4>(w, row_bytes, n, act + t0, out + t0, r0, r1); break;
        }
    }
}

}  // namespace

bool cpu_avx2_ok() noexcept {
#if defined(__x86_64__)
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    return (ecx & (1u << 5)) && (ecx & (1u << 27));  // AVX2 + FMA3
#else
    return false;
#endif
}

bool iqavx2_supported(int type) noexcept {
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22;
}

void iqavx2_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                    float* const* ff, int r0, int r1) {
    switch (type) {
        case 16: gu_rows_nt<16>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 17: gu_rows_nt<17>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 18: gu_rows_nt<18>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 21: gu_rows_nt<21>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 22: gu_rows_nt<22>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: break;
    }
}

void iqavx2_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                 int r0, int r1) {
    switch (type) {
        case 16: dot_rows_nt<16>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 17: dot_rows_nt<17>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 18: dot_rows_nt<18>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 21: dot_rows_nt<21>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 22: dot_rows_nt<22>(nt, w, row_bytes, n, act, out, r0, r1); break;
        default: break;
    }
}

}  // namespace strata::kernels::cpu
