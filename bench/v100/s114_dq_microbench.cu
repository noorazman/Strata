// bench/v100/s114_dq_microbench.cu — Stage 1.14: expert-weight dequant kernel variants.
//
// The prefill MoE path dequantizes each expert's gate/up (GU) and down (D) matrices from the
// packed GGUF blob to f16 before the cuBLAS GEMM, once per expert RUN (133 K runs at 16 K;
// dequant is the largest single kernel class, ~9.4 s, latency-bound per the 1.11 ncu SOL:
// dequant_gu SM 9.3 %, mem 48 %, DRAM 18 %).  This microbench times the current kernels
// against variants that raise per-thread memory-level parallelism (multiple independent
// 256-value superblocks per thread), at the real expert shapes (n_ff=640, n_embd=2560) and
// all quant types the model uses (GU: 16/17/18/21/22, D: 20/42).  Every variant must produce
// bit-identical f16 output (the per-value math is unchanged; only the scheduling changes).
//
// Single TU: iq_kernels.cu is included so the variants share its dq_dispatch + ggml-common
// block structs/constants (one source of truth for the dequant math).
//
// Build: /usr/local/cuda/bin/nvcc -O3 -std=c++20 -arch=sm_70 -I include -I third_party/ggml \
//        s114_dq_microbench.cu -o /tmp/s114_dq_mb
// Run:   CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 /tmp/s114_dq_mb
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>

#include "iq_kernels.cu"  // relative from -I src/kernels/cuda

using strata::kernels::iq_row_bytes;
using strata::kernels::iq_dequant_gu_f16;
using strata::kernels::iq_dequant_f16;

namespace {

static const int REPS = 50;

// ---------------- variant kernels
// PER = independent 256-value superblocks per thread.  The per-superblock math and the
// output addressing are the production kernel's, verbatim; only the scheduling changes.
template<int PER>
__global__ void dq_gu_var(int ty, const void* __restrict__ gate, const void* __restrict__ up, int64_t per_row,
                          __half* __restrict__ y) {
    const int64_t base = (int64_t) blockIdx.x * PER;
    const int parity = blockIdx.y;
    const void* vx = parity ? up : gate;
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int64_t i = base + p;
        const int64_t r = i / per_row, c = i % per_row;
        strata::kernels::dq_dispatch<__half>(ty, vx, i, y + ((2 * r + parity) * per_row + c) * 256, threadIdx.x);
    }
}
template<int PER>
__global__ void dq_flat_var(int ty, const void* __restrict__ vx, __half* __restrict__ y) {
    const int64_t base = (int64_t) blockIdx.x * PER;
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int64_t i = base + p;
        strata::kernels::dq_dispatch<__half>(ty, vx, i, y + i * 256, threadIdx.x);
    }
}
// control: 4 independent warps per CTA, one superblock each (same per-warp work as the PER=1 baseline)
__global__ void dq_flat_w128(int ty, const void* __restrict__ vx, __half* __restrict__ y) {
    const int64_t i = (int64_t) (blockIdx.x * 4 + (int) (threadIdx.x >> 5));
    strata::kernels::dq_dispatch<__half>(ty, vx, i, y + i * 256, (int) (threadIdx.x & 31));
}


// GU control: 4 independent warps per CTA, one superblock each (same per-warp work as the PER=1 baseline)
__global__ void dq_gu_w128(int ty, const void* __restrict__ gate, const void* __restrict__ up, int64_t per_row,
                           __half* __restrict__ y) {
    const int64_t i = (int64_t) (blockIdx.x * 4 + (int) (threadIdx.x >> 5));
    const int parity = blockIdx.y;
    const int64_t r = i / per_row, c = i % per_row;
    strata::kernels::dq_dispatch<__half>(ty, parity ? up : gate, i, y + ((2 * r + parity) * per_row + c) * 256,
                                         (int) (threadIdx.x & 31));
}


// compile-time-typed dispatch (no runtime switch): the per-value math is the production
// dq_* device functions, called directly per type.
template<int TY> struct DqSel {
    static __device__ void f(const void* v, int64_t ibs, __half* y, int tid) { strata::kernels::dq_dispatch<__half>(TY, v, ibs, y, tid); }
};
#define DQSEL(ty, fn)     template<> struct DqSel<ty> { static __device__ void f(const void* v, int64_t ibs, __half* y, int tid) { fn<__half>(v, ibs, y, tid); } };
DQSEL(16, strata::kernels::dq_iq2_xxs)
DQSEL(17, strata::kernels::dq_iq2_xs)
DQSEL(18, strata::kernels::dq_iq3_xxs)
DQSEL(20, strata::kernels::dq_iq4_nl)
DQSEL(21, strata::kernels::dq_iq3_s)
DQSEL(22, strata::kernels::dq_iq2_s)
DQSEL(42, strata::kernels::dq_q2_0)

template<int TY, int PER>
__global__ void dq_gu_typed(int ty, const void* __restrict__ gate, const void* __restrict__ up, int64_t per_row,
                            __half* __restrict__ y) {
    const int64_t base = (int64_t) blockIdx.x * PER;
    const int parity = blockIdx.y;
    const void* vx = parity ? up : gate;
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int64_t i = base + p;
        const int64_t r = i / per_row, c = i % per_row;
        DqSel<TY>::f(vx, i, y + ((2 * r + parity) * per_row + c) * 256, (int) threadIdx.x);
    }
}
template<int TY, int PER>
__global__ void dq_flat_typed(int ty, const void* __restrict__ vx, __half* __restrict__ y) {
    const int64_t base = (int64_t) blockIdx.x * PER;
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int64_t i = base + p;
        DqSel<TY>::f(vx, i, y + i * 256, (int) threadIdx.x);
    }
}

template<typename F>
static double us_of(F f, cudaStream_t s) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 10; ++i) f();
    std::vector<double> t;
    for (int i = 0; i < REPS; ++i) {
        cudaEventRecord(a, s);
        f();
        cudaEventRecord(b, s);
        cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    cudaEventDestroy(a); cudaEventDestroy(b);
    return t[REPS / 2] * 1e3;
}

const int64_t N_FF = 640, N_EMBD = 2560;
const int64_t PER_ROW = N_EMBD / 256;
const int64_t GU_SB = N_FF * PER_ROW;      // superblocks per role (gate or up)
const int64_t D_SB = N_EMBD * N_FF / 256;  // D superblocks

void run(int argc, char** argv) {
    cudaStream_t s; cudaStreamCreate(&s);
    const int gu_types[] = {16, 17, 18, 21, 22};
    const int d_types[] = {20, 42};

    for (int ty : gu_types) {
        const size_t role_bytes = (size_t) N_FF * iq_row_bytes(ty, N_EMBD);
        std::vector<uint8_t> blob(role_bytes);
        for (size_t i = 0; i < blob.size(); ++i) blob[i] = (uint8_t) ((i * 2654435761u) >> 8);
        void *gate, *up, *W, *W2;
        cudaMalloc(&gate, role_bytes); cudaMalloc(&up, role_bytes);
        cudaMalloc(&W, (size_t) 2 * N_FF * N_EMBD * 2);
        cudaMalloc(&W2, (size_t) 2 * N_FF * N_EMBD * 2);
        cudaMemcpy(gate, blob.data(), role_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(up, blob.data(), role_bytes, cudaMemcpyHostToDevice);
        iq_dequant_gu_f16(ty, gate, up, N_FF, N_EMBD, (uint16_t*) W, s);
        cudaStreamSynchronize(s);

        auto base = [&] { iq_dequant_gu_f16(ty, gate, up, N_FF, N_EMBD, (uint16_t*) W, s); };
        auto v1 = [&] { dq_gu_var<2><<<(unsigned) (GU_SB / 2), 32, 0, s>>>(ty, gate, up, PER_ROW, (__half*) W2); };
        auto v2 = [&] { dq_gu_var<4><<<(unsigned) (GU_SB / 4), 32, 0, s>>>(ty, gate, up, PER_ROW, (__half*) W2); };
        auto v1w = [&] { dq_gu_w128<<<dim3((unsigned) (GU_SB / 4), 2), 128, 0, s>>>(ty, gate, up, PER_ROW, (__half*) W2); };
        double tb = us_of(base, s), t1 = us_of(v1, s), t2 = us_of(v2, s), t1w = us_of(v1w, s);
        double tt;
        switch (ty) {
            case 16: tt = us_of([&] { dq_gu_typed<16, 2><<<(unsigned)(GU_SB/2), 32, 0, s>>>(ty, gate, up, PER_ROW, (__half*) W2); }, s); break;
            case 17: tt = us_of([&] { dq_gu_typed<17, 2><<<(unsigned)(GU_SB/2), 32, 0, s>>>(ty, gate, up, PER_ROW, (__half*) W2); }, s); break;
            case 18: tt = us_of([&] { dq_gu_typed<18, 2><<<(unsigned)(GU_SB/2), 32, 0, s>>>(ty, gate, up, PER_ROW, (__half*) W2); }, s); break;
            case 21: tt = us_of([&] { dq_gu_typed<21, 2><<<(unsigned)(GU_SB/2), 32, 0, s>>>(ty, gate, up, PER_ROW, (__half*) W2); }, s); break;
            default: tt = us_of([&] { dq_gu_typed<22, 2><<<(unsigned)(GU_SB/2), 32, 0, s>>>(ty, gate, up, PER_ROW, (__half*) W2); }, s); break;
        }
        // (typed-identity check appended after the existing ones)

        std::vector<uint8_t> ref((size_t) 2 * N_FF * N_EMBD * 2), got(ref.size());
        cudaMemcpy(ref.data(), W, ref.size(), cudaMemcpyDeviceToHost);
        bool ok1 = true, ok2 = true, ok1w = true;
        v1(); cudaMemcpy(got.data(), W2, got.size(), cudaMemcpyDeviceToHost); ok1 = (ref == got);
        v2(); cudaMemcpy(got.data(), W2, got.size(), cudaMemcpyDeviceToHost); ok2 = (ref == got);
        v1w(); cudaMemcpy(got.data(), W2, got.size(), cudaMemcpyDeviceToHost); ok1w = (ref == got);
        { bool okt = true;
          auto vt = [&]() {
            switch (ty) {
                case 16: dq_gu_typed<16, 2><<<(unsigned)(GU_SB/2), 32, 0, s>>>(ty, gate, up, PER_ROW, (__half*) W2); break;
                case 17: dq_gu_typed<17, 2><<<(unsigned)(GU_SB/2), 32, 0, s>>>(ty, gate, up, PER_ROW, (__half*) W2); break;
                case 18: dq_gu_typed<18, 2><<<(unsigned)(GU_SB/2), 32, 0, s>>>(ty, gate, up, PER_ROW, (__half*) W2); break;
                case 21: dq_gu_typed<21, 2><<<(unsigned)(GU_SB/2), 32, 0, s>>>(ty, gate, up, PER_ROW, (__half*) W2); break;
                default: dq_gu_typed<22, 2><<<(unsigned)(GU_SB/2), 32, 0, s>>>(ty, gate, up, PER_ROW, (__half*) W2); break;
            }
          };
          vt(); cudaMemcpy(got.data(), W2, got.size(), cudaMemcpyDeviceToHost); okt = (ref == got);
          if (!okt) printf("GU type %d: TYPED VARIANT BIT-MISMATCH\n", ty); }
        printf("GU type %2d: base %8.1f us | 2/thread %8.1f us (%.2fx) | 4/thread %8.1f us (%.2fx) | w128 %8.1f us (%.2fx) | typed-2 %8.1f us (%.2fx) | bit-ident: %s/%s/%s\n",
               ty, tb, t1, tb / t1, t2, tb / t2, t1w, tb / t1w, tt, tb / tt,
               ok1 ? "y" : "N", ok2 ? "y" : "N", ok1w ? "y" : "N");
        cudaFree(gate); cudaFree(up); cudaFree(W); cudaFree(W2);
    }

    for (int ty : d_types) {
        const size_t d_bytes = (size_t) N_EMBD * iq_row_bytes(ty, N_FF);
        std::vector<uint8_t> blob(d_bytes);
        for (size_t i = 0; i < blob.size(); ++i) blob[i] = (uint8_t) ((i * 40503u) >> 8);
        void *down, *W, *W2;
        cudaMalloc(&down, d_bytes); cudaMalloc(&W, (size_t) N_EMBD * N_FF * 2);
        cudaMalloc(&W2, (size_t) N_EMBD * N_FF * 2);
        cudaMemcpy(down, blob.data(), d_bytes, cudaMemcpyHostToDevice);
        iq_dequant_f16(ty, down, N_EMBD * N_FF, (uint16_t*) W, s);
        cudaStreamSynchronize(s);

        auto base = [&] { iq_dequant_f16(ty, down, N_EMBD * N_FF, (uint16_t*) W, s); };
        auto v1 = [&] { dq_flat_var<2><<<(unsigned) (D_SB / 2), 32, 0, s>>>(ty, down, (__half*) W2); };
        auto v2 = [&] { dq_flat_var<4><<<(unsigned) (D_SB / 4), 32, 0, s>>>(ty, down, (__half*) W2); };
        auto w128 = [&] { dq_flat_w128<<<(unsigned) (D_SB / 4), 128, 0, s>>>(ty, down, (__half*) W2); };
        double tb = us_of(base, s), t1 = us_of(v1, s), t2 = us_of(v2, s), tw = us_of(w128, s);

        std::vector<uint8_t> ref((size_t) N_EMBD * N_FF * 2), got(ref.size());
        cudaMemcpy(ref.data(), W, ref.size(), cudaMemcpyDeviceToHost);
        bool ok1 = true, ok2 = true, okw = true;
        v1(); cudaMemcpy(got.data(), W2, got.size(), cudaMemcpyDeviceToHost); ok1 = (ref == got);
        v2(); cudaMemcpy(got.data(), W2, got.size(), cudaMemcpyDeviceToHost); ok2 = (ref == got);
        w128(); cudaMemcpy(got.data(), W2, got.size(), cudaMemcpyDeviceToHost); okw = (ref == got);
        printf("D  type %2d: base %8.1f us | 2/thread %8.1f us (%.2fx) | 4/thread %8.1f us (%.2fx) | w128 %8.1f us (%.2fx) | bit-ident: %s/%s/%s\n",
               ty, tb, t1, tb / t1, t2, tb / t2, tw, tb / tw,
               ok1 ? "y" : "N", ok2 ? "y" : "N", okw ? "y" : "N");
        cudaFree(down); cudaFree(W); cudaFree(W2);
    }
    cudaStreamDestroy(s);
}

}  // namespace

int main(int argc, char** argv) {
    run(argc, argv);
    return 0;
}
