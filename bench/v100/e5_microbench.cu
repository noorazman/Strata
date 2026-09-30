// bench/v100/e5_microbench.cu — Stage 1.12: per-expert-shape cost of (dequant + cuBLAS GEMM) vs the
// E5 fused dequant+GEMM kernel, at the typical routed-expert shape (ne ≈ 6.6 tokens, n_ff=640,
// n_embd=2560 as in swift-iq3_xxs). GPU event timing, median of REPS after warmup.
//
// Build: /usr/local/cuda/bin/nvcc -O2 -std=c++20 -arch=sm_70 -I include -I third_party/ggml \
//        e5_microbench.cu ../../src/kernels/cuda/iq_kernels.cu ../../src/kernels/cuda/moe_fused.cu \
//        -lcublas -o /tmp/e5_microbench
// Run:   CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 /tmp/e5_microbench [T]
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/moe_fused.hpp"

using strata::kernels::iq_dequant_f16;
using strata::kernels::iq_dequant_gu_f16;
using strata::kernels::iq_row_bytes;
using strata::kernels::moe_fused_gemm_gu;
using strata::kernels::moe_fused_gemm_d;

static const int REPS = 30;

template<typename F>
static double ms_of(F f, cudaStream_t s) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 5; ++i) f();          // warmup
    std::vector<double> tms;
    for (int i = 0; i < REPS; ++i) {
        cudaEventRecord(a, s);
        f();
        cudaEventRecord(b, s);
        cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        tms.push_back(ms);
    }
    std::sort(tms.begin(), tms.end());
    cudaEventDestroy(a); cudaEventDestroy(b);
    return tms[REPS / 2];
}

int main(int argc, char** argv) {
    const int64_t n_ff = 640, n_embd = 2560;
    const int64_t T = (argc > 1) ? std::atoll(argv[1]) : 8;

    // packed blobs: gate + up (type 17 IQ2_XS), down (type 42 Q2_0)
    const size_t b_gu = iq_row_bytes(17, n_embd);   // per role row (n_embd values)
    const size_t b_d_row = iq_row_bytes(42, n_ff);  // D row: n_ff values
    const size_t b_down = (size_t) n_embd * b_d_row;

    std::vector<uint8_t> blob_gu((size_t) n_ff * b_gu, 0), blob_down(b_down, 0);
    for (size_t i = 0; i < blob_gu.size(); ++i) blob_gu[i] = (uint8_t) ((i * 2654435761u) >> 8);
    for (size_t i = 0; i < blob_down.size(); ++i) blob_down[i] = (uint8_t) ((i * 40503u) >> 8);

    void *gate, *up, *down, *W, *Wd, *X, *H, *Y;
    const size_t b_role = (size_t) n_ff * b_gu;     // one role (gate or up): n_ff rows x n_embd
    cudaMalloc(&gate, b_role); cudaMalloc(&up, b_role); cudaMalloc(&down, b_down);
    cudaMalloc(&W, (size_t) 2 * n_ff * n_embd * 2);              // GU dequant: 1280 x 2560 f16
    cudaMalloc(&Wd, (size_t) n_embd * n_ff * 2);                 // D dequant: 2560 x 640 f16
    cudaMalloc(&X, (size_t) T * n_embd * 2);
    cudaMalloc(&H, (size_t) T * n_ff * 2);
    cudaMalloc(&Y, (size_t) T * n_embd * 4);
    cudaMemcpy(gate, blob_gu.data(), b_role, cudaMemcpyHostToDevice);
    cudaMemcpy(up, blob_gu.data(), b_role, cudaMemcpyHostToDevice);
    cudaMemcpy(down, blob_down.data(), b_down, cudaMemcpyHostToDevice);
    // deterministic X / H
    std::vector<uint16_t> xv((size_t) T * n_embd), hv((size_t) T * n_ff);
    for (size_t i = 0; i < xv.size(); ++i) xv[i] = (uint16_t) ((i * 7919u) % 65535);
    for (size_t i = 0; i < hv.size(); ++i) hv[i] = (uint16_t) ((i * 104729u) % 65535);
    cudaMemcpy(X, xv.data(), xv.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(H, hv.data(), hv.size() * 2, cudaMemcpyHostToDevice);

    cublasHandle_t ch; cublasCreate(&ch);
    const float alpha = 1.f, beta = 0.f;
    cudaStream_t s; cudaStreamCreate(&s);

    auto gu_dequant = [&] { iq_dequant_gu_f16(17, gate, up, n_ff, n_embd, (uint16_t*) W, s); };
    auto gu_gemm = [&] {
        cublasGemmEx(ch, CUBLAS_OP_T, CUBLAS_OP_N, (int) (2 * n_ff), (int) T, (int) n_embd, &alpha,
                     W, CUDA_R_16F, (int) n_embd, X, CUDA_R_16F, (int) n_embd, &beta, Y, CUDA_R_32F,
                     (int) (2 * n_ff), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    };
    auto gu_fused = [&] {
        moe_fused_gemm_gu(17, gate, up, T, n_ff, n_embd, (const uint16_t*) X, n_embd, (float*) Y,
                          2 * n_ff, s);
    };

    auto d_dequant = [&] { iq_dequant_f16(42, down, (int64_t) n_embd * n_ff, (uint16_t*) Wd, s); };
    auto d_gemm = [&] {
        cublasGemmEx(ch, CUBLAS_OP_T, CUBLAS_OP_N, (int) n_embd, (int) T, (int) n_ff, &alpha,
                     Wd, CUDA_R_16F, (int) n_ff, H, CUDA_R_16F, (int) n_ff, &beta, Y, CUDA_R_32F,
                     (int) n_embd, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    };
    auto d_fused = [&] {
        moe_fused_gemm_d(42, down, T, n_ff, n_embd, (const uint16_t*) H, n_ff, (float*) Y, n_embd, s);
    };

    double a = ms_of(gu_dequant, s);
    double b = ms_of(gu_gemm, s);
    double c = ms_of(gu_fused, s);
    double d = ms_of(d_dequant, s);
    double e = ms_of(d_gemm, s);
    double f = ms_of(d_fused, s);

    printf("shape: T=%ld (per expert; median of %d, %d warmup)\n", (long) T, REPS, 5);
    printf("GU (M=%4lld K=%5lld): dequant %8.1f us | cuBLAS %8.1f us | OFF pair %8.1f us | E5 fused %8.1f us | ratio %.2fx\n",
           (long long) (2 * n_ff), (long long) n_embd, a * 1e3, b * 1e3, (a + b) * 1e3, c * 1e3, c / (a + b));
    printf("D  (M=%4lld K=%5lld): dequant %8.1f us | cuBLAS %8.1f us | OFF pair %8.1f us | E5 fused %8.1f us | ratio %.2fx\n",
           (long long) n_embd, (long long) n_ff, d * 1e3, e * 1e3, (d + e) * 1e3, f * 1e3, f / (d + e));
    printf("expert total: OFF %8.1f us | E5 %8.1f us | ratio %.2fx\n",
           (a + b + d + e) * 1e3, (c + f) * 1e3, (c + f) / (a + b + d + e));
    cudaStreamDestroy(s); cublasDestroy(ch);
    return 0;
}
