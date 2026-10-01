// bench/v100/s115_gemm_microbench.cu — Stage 1.15: MoE expert GEMM characterization.
//
// The prefill MoE path runs, per routed-expert RUN (133 K at 16 K), two f16 cuBLAS GEMMs on
// the dequantized expert weights:
//   GU : Y[ne][1280] = X[ne][2560] . W_gu[1280][2560]^T   (m=1280, k=2560, reads 6.55 MB f16)
//   D  : Y[ne][2560] = H[ne][640]  . W_d[2560][640]^T    (m=2560, k=640,  reads 3.28 MB f16)
// where ne (T) is the per-expert token count (route dump: 1..31, mean ~6.6).  These are
// extremely skinny, memory-bound GEMMs.  This microbench measures the EXACT engine cuBLAS
// call (cublasGemmEx, CUBLAS_OP_T/OP_N, COMPUTE_32F) at the real shapes and reports the
// achieved weight-read bandwidth vs the V100 f16 copy roofline, plus a few cuBLAS variants,
// so we can see how much headroom a better GEMM has.
//
// Build: /usr/local/cuda/bin/nvcc -O3 -std=c++20 -arch=sm_70 s115_gemm_microbench.cu -lcublas -o /tmp/s115_gemm_mb
// Run:   CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 /tmp/s115_gemm_mb
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>
#include <cublas_v2.h>

using u16 = uint16_t;

static const int64_t N_FF = 640, N_EMBD = 2560;
// roofline: V100 measured f16 copy ~474 GB/s (Stage 1.11/1.14).
static const double GBPS_ROOF = 474e9;

static const int REPS = 60;

template<typename F>
static double us_of(F f, cudaStream_t s) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 12; ++i) f();
    std::vector<double> t;
    for (int i = 0; i < REPS; ++i) {
        cudaEventRecord(a, s); f(); cudaEventRecord(b, s);
        cudaEventSynchronize(b); float ms = 0; cudaEventElapsedTime(&ms, a, b); t.push_back(ms);
    }
    std::sort(t.begin(), t.end()); cudaEventDestroy(a); cudaEventDestroy(b);
    return t[REPS / 2] * 1e3;
}

int main() {
    cublasHandle_t h; cublasCreate(&h);
    cudaStream_t s; cudaStreamCreate(&s);
    cublasSetStream(h, s);
    size_t ws = 32u << 20; void* wsp; cudaMalloc(&wsp, ws); cublasSetWorkspace(h, wsp, ws);
    const float alpha = 1.0f, beta = 0.0f;

    // f16 GU weight [1280][2560] and activation [ne][2560], f32 out [ne][1280].
    const int64_t GU_M = 1280, GU_K = 2560;
    const int64_t D_M = 2560, D_K = 640;
    u16 *Wgu, *Xd, *Yu, *Wd, *Hd, *Yd;
    cudaMalloc(&Wgu, (size_t) GU_M * GU_K * 2);
    cudaMalloc(&Wd, (size_t) D_M * D_K * 2);
    for (int64_t nb : { (int64_t) 32768 }) {
        cudaMalloc(&Xd, (size_t) nb * GU_K * 2);
        cudaMalloc(&Yu, (size_t) nb * GU_M * 4);
        cudaMalloc(&Hd, (size_t) nb * D_K * 2);
        cudaMalloc(&Yd, (size_t) nb * D_M * 4);
    }
    // fill with small random-ish f16 values
    { std::vector<u16> b((size_t) 32768 * GU_K);
      for (size_t i = 0; i < b.size(); ++i) b[i] = (u16) ((i * 2654435761u) & 0x7fff);
      cudaMemcpy(Wgu, b.data(), (size_t) GU_M * GU_K * 2, cudaMemcpyHostToDevice);
      cudaMemcpy(Xd, b.data(), (size_t) 32768 * GU_K * 2, cudaMemcpyHostToDevice);
      cudaMemcpy(Hd, b.data(), (size_t) 32768 * D_K * 2, cudaMemcpyHostToDevice);
      std::vector<u16> b2((size_t) D_M * D_K);
      for (size_t i = 0; i < b2.size(); ++i) b2[i] = (u16) ((i * 40503u) & 0x7fff);
      cudaMemcpy(Wd, b2.data(), b2.size() * 2, cudaMemcpyHostToDevice); }

    printf("=== GU GEMM: m=1280 k=2560, weight 6.55 MB f16 ===\n");
    printf("ne   cuBLAS(DEF)us  BW(GB/s) %%roof | FAST16F us  BW | note\n");
    for (int ne : {1, 2, 4, 8, 12, 16, 24, 32}) {
        // exact engine call: m=GU_M, n=ne, k=GU_K
        auto gemm = [&](cublasComputeType_t ct) {
            return [&] {
                cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) GU_M, ne, (int) GU_K, &alpha, Wgu,
                             CUDA_R_16F, (int) GU_K, Xd, CUDA_R_16F, (int) GU_K, &beta, Yu, CUDA_R_32F,
                             (int) GU_M, ct, CUBLAS_GEMM_DEFAULT);
            };
        };
        double t_def = us_of(gemm(CUBLAS_COMPUTE_32F), s);
        double t_fast = us_of(gemm(CUBLAS_COMPUTE_32F_FAST_16F), s);
        double bw_def = 6.553e6 / (t_def * 1e-6) / 1e9;
        double bw_fast = 6.553e6 / (t_fast * 1e-6) / 1e9;
        printf("%3d   %9.1f  %7.0f %5.1f%% | %8.1f %6.0f | F16 %.2fx\n",
               ne, t_def, bw_def, 100.0 * bw_def / GBPS_ROOF, t_fast, bw_fast, t_def / t_fast);
    }
    printf("=== D GEMM: m=2560 k=640, weight 3.28 MB f16 ===\n");
    printf("ne   cuBLAS(DEF)us  BW(GB/s) %%roof | FAST16F us  BW | note\n");
    for (int ne : {1, 2, 4, 8, 12, 16, 24, 32}) {
        auto gemm = [&](cublasComputeType_t ct) {
            return [&] {
                cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) D_M, ne, (int) D_K, &alpha, Wd,
                             CUDA_R_16F, (int) D_K, Hd, CUDA_R_16F, (int) D_K, &beta, Yd, CUDA_R_32F,
                             (int) D_M, ct, CUBLAS_GEMM_DEFAULT);
            };
        };
        double t_def = us_of(gemm(CUBLAS_COMPUTE_32F), s);
        double t_fast = us_of(gemm(CUBLAS_COMPUTE_32F_FAST_16F), s);
        double bw_def = 3.277e6 / (t_def * 1e-6) / 1e9;
        double bw_fast = 3.277e6 / (t_fast * 1e-6) / 1e9;
        printf("%3d   %9.1f  %7.0f %5.1f%% | %8.1f %6.0f | F16 %.2fx\n",
               ne, t_def, bw_def, 100.0 * bw_def / GBPS_ROOF, t_fast, bw_fast, t_def / t_fast);
    }
    printf("\n(roofline 474 GB/s; BW = weight_bytes/time; the GEMM must read the full f16 weight)\n");
    return 0;
}
