// bench/v100/e5_dequant_parity.cu - Stage 1.12 E5: proves the fused kernel's per-value dequant
// (include/strata/kernels/moe_fused_iq.hpp) is BIT-IDENTICAL to the engine's separate dequant path
// (src/kernels/cuda/iq_kernels.cu), for every quant type in the production pack
// (gate/up 16/17/18/21/22, down 20/42).
//
// For each type, a deterministic pseudo-random packed blob is dequantized two ways:
//   reference  - iq_dequant_f16 (the block-wide dq_* path the OFF baseline uses)
//   fused-style - one thread per value, with the fused kernel's exact (mrow,k) -> (unit,pos)
//                 addressing and the shared iq_value<TYPE> dequantizer (gate/up interleave for the
//                 GU types, flat rows for the D types); the result lands in a __half tile exactly as
//                 the fused kernel writes its shared-memory weight tile.
// The f16 results are compared bit-by-bit on the host.  Exit 0 = bit-identical for all types.
//
// Build:  /usr/local/cuda/bin/nvcc -O2 -std=c++20 -arch=sm_70 -I ../../include -I ../../third_party/ggml \
//              e5_dequant_parity.cu ../../src/kernels/cuda/iq_kernels.cu -o e5_dequant_parity
// Run:    CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 ./e5_dequant_parity
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/moe_fused_iq.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace strata::kernels;

namespace {

// One thread per (mrow, k): the exact dequant addressing of the fused kernel's weight-stripe pass.
// (dst is a __half tile, matching the real kernel's `__shared__ __half ws`.)
template<int TYPE, bool GU>
__global__ void fused_style(const void* s0, const void* s1, int64_t rows, int64_t cols, __half* dst) {
    constexpr int QV = Unit<TYPE>::qv;
    constexpr size_t UB = Unit<TYPE>::bytes;
    const int64_t M = GU ? 2 * rows : rows;
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * cols) return;
    const int64_t mrow = idx / cols, k = idx % cols;
    const int64_t region_row = GU ? (mrow >> 1) : mrow;
    const void* base = (GU && (mrow & 1)) ? s1 : s0;
    const int64_t unit = region_row * (cols / QV) + k / QV;
    dst[idx] = __float2half(iq_value<TYPE>((const uint8_t*) base + unit * UB, k % QV));
}

uint32_t rng_state = 0x9e3779b9u;
uint8_t rnd() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return (uint8_t) rng_state;
}

float half_as_f16(uint16_t h) { return __half2float(*(const __half*) &h); }

template<bool GU>
int check_type(int type, const char* name) {
    rng_state = 0xABCD0000u + (uint32_t) type;  // deterministic per type
    const int64_t rows = 8, cols = 2560;  // cols = n_embd: 10x256 / 80x32 / 40x64 units per row
    const int64_t n = rows * cols;
    const size_t qv = (size_t) (type == 20 ? 32 : type == 42 ? 64 : 256);
    const size_t ub = (size_t) (type == 20 ? sizeof(block_iq4_nl) : type == 42 ? sizeof(block_q2_0) :
                               type == 16 ? sizeof(block_iq2_xxs) : type == 17 ? sizeof(block_iq2_xs) :
                               type == 18 ? sizeof(block_iq3_xxs) : type == 21 ? sizeof(block_iq3_s) : sizeof(block_iq2_s));
    const size_t unit_bytes = n / qv * ub;

    std::vector<uint8_t> host_a(unit_bytes);
    for (auto& v : host_a) v = rnd();
    std::vector<uint8_t> host_b(unit_bytes);
    for (auto& v : host_b) v = rnd();
    uint8_t *da = nullptr, *db = nullptr;
    __half* dw = nullptr;
    cudaMalloc(&da, unit_bytes);
    cudaMemcpy(da, host_a.data(), unit_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&db, unit_bytes);
    cudaMemcpy(db, host_b.data(), unit_bytes, cudaMemcpyHostToDevice);
    const int64_t M = GU ? 2 * rows : rows;
    const int64_t total = M * cols;
    cudaMalloc(&dw, total * sizeof(__half));
    cudaStream_t st;
    cudaStreamCreate(&st);

    // ---- reference: the engine's block-wide dequant of the flat region(s)
    std::vector<uint16_t> a(n), b(n);
    {
        uint16_t* tmp = nullptr;
        cudaMalloc(&tmp, n * sizeof(uint16_t));
        iq_dequant_f16(type, da, n, tmp, st);
        cudaMemcpy(a.data(), tmp, n * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        if (GU) {
            iq_dequant_f16(type, db, n, tmp, st);
            cudaMemcpy(b.data(), tmp, n * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        }
        cudaFree(tmp);
    }
    std::vector<uint16_t> ref(total);
    if (!GU) {
        ref = std::move(a);
    } else {
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t k = 0; k < cols; ++k) {
                ref[(2 * r + 0) * cols + k] = a[r * cols + k];  // even row = region A (gate)
                ref[(2 * r + 1) * cols + k] = b[r * cols + k];  // odd row  = region B (up)
            }
    }

    // ---- fused-style: one thread per (mrow, k) with the kernel's addressing
    const int threads = 256;
    const unsigned blocks = (unsigned) ((total + threads - 1) / threads);
    switch (type) {
        case 16: fused_style<16, GU><<<blocks, threads, 0, st>>>(da, db, rows, cols, dw); break;
        case 17: fused_style<17, GU><<<blocks, threads, 0, st>>>(da, db, rows, cols, dw); break;
        case 18: fused_style<18, GU><<<blocks, threads, 0, st>>>(da, db, rows, cols, dw); break;
        case 21: fused_style<21, GU><<<blocks, threads, 0, st>>>(da, db, rows, cols, dw); break;
        case 22: fused_style<22, GU><<<blocks, threads, 0, st>>>(da, db, rows, cols, dw); break;
        case 20: fused_style<20, false><<<blocks, threads, 0, st>>>(da, db, rows, cols, dw); break;
        case 42: fused_style<42, false><<<blocks, threads, 0, st>>>(da, db, rows, cols, dw); break;
    }
    cudaStreamSynchronize(st);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: cuda error: %s\n", name, cudaGetErrorString(e)); return 1; }

    std::vector<__half> got(total);
    cudaMemcpy(got.data(), dw, total * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dw);
    cudaStreamDestroy(st);

    int bad_total = 0;
    int64_t first_bad = -1;
    for (int64_t i = 0; i < total; ++i)
        if (__half_as_ushort(got[i]) != ref[i]) {
            if (first_bad < 0) first_bad = i;
            ++bad_total;
        }
    if (first_bad >= 0) {
        const int64_t i = first_bad;
        const int64_t mrow = i / cols, k = i % cols;
        const int64_t region_row = GU ? (mrow >> 1) : mrow;
        const uint8_t* base = (GU && (mrow & 1)) ? db : da;
        const int qv = (type == 20) ? 32 : (type == 42) ? 64 : 256;
        const int64_t blk = (region_row * cols + k) / qv;
        const uint16_t g16 = __half_as_ushort(got[i]);
        std::fprintf(stderr, "%s: first mismatch @%lld (mrow=%lld k=%lld block=%lld): ref %04x (%.6f) got %04x (%.6f)\n",
                     name, (long long) i, (long long) mrow, (long long) k, (long long) blk,
                     ref[i], half_as_f16(ref[i]), g16, half_as_f16(g16));
        std::vector<uint8_t> raw(ub);
        cudaMemcpy(raw.data(), (const void*) (base + blk * ub), ub, cudaMemcpyDeviceToHost);
        std::fprintf(stderr, "  raw block bytes[0..31]:");
        for (size_t x = 0; x < 32 && x < ub; ++x) std::fprintf(stderr, " %02x", raw[x]);
        std::fprintf(stderr, "\n");
    }
    std::fprintf(stderr, "%s: %lld values, %d mismatches\n", name, (long long) total, bad_total);
    return bad_total == 0 ? 0 : 1;
}

}  // namespace

int main() {
    int rc = 0;
    rc |= check_type<true>(16, "gu type 16 (IQ2_XXS)");
    rc |= check_type<true>(17, "gu type 17 (IQ2_XS)");
    rc |= check_type<true>(18, "gu type 18 (IQ3_XXS)");
    rc |= check_type<true>(21, "gu type 21 (IQ3_S)");
    rc |= check_type<true>(22, "gu type 22 (IQ2_S)");
    rc |= check_type<false>(20, "d  type 20 (IQ4_NL)");
    rc |= check_type<false>(42, "d  type 42 (Q2_0)");
    std::fprintf(stderr, rc == 0 ? "E5 dequant parity: BIT-IDENTICAL for all 7 types\n"
                                 : "E5 dequant parity: FAILURES (see above)\n");
    return rc;
}
