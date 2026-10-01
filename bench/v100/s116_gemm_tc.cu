// bench/v100/s116_gemm_tc.cu — Stage 1.16: V100 tensor-core skinny-GU GEMM microbench.
//
// The GU expert GEMM is  Y[ne,1280] = X[ne,2560] . W[1280,2560]^T  (f16 in, f32 out),
// 133 K+ runs at 16 K, dominant ne = 1..64. cuBLAS runs 24-42 us there (isolated microbench)
// plus a 7.6 us splitKreduce; in-engine the ne=16..128 band lands on 64/128-wide tiled
// kernels (23-52 us avg, nsys) where n-tile padding wastes compute and the W read stays
// ~40% of DRAM peak. This file prototypes ONE tensor-core path for skinny ne on V100
// (sm_70; its only f16 tensor-op mma is m8n8k4, which per the PTX ISA computes FOUR k4
// sub-MMAs i.e. effectively m8 x n8 x k16: A/B fragments are 4 k-CONSECUTIVE f16 per lane,
// C is 8 f32 per lane, each 8-lane group holds the full 8x8 result):
//
//   * warp = m16 x n16 output tile (4 sub-tiles: 2 m8 x 2 n8 -> 4 mma per k16 step; the
//     B fragments of the two m8 sub-tiles are shared)
//   * block = 8 warps (n128 chunk); grid = (1280/16) x ceil(ne/128) = 80 x k
//   * full K=2560 per tile (160 k16 steps) -> no split-K, no reduce kernel, fixed
//     accumulation order -> deterministic
//   * A (W) and B (X) fragments are 8B k-contiguous, read straight from L2/DRAM with
//     2-step LDG.64 register prefetch; no smem, no split-K
//   * for ne <= 128: W bytes touch DRAM exactly once (6.55 MB), X <= 640 KB is L2-resident
//     in the engine; warps with n0 >= ne early-exit (no padding FLOPs / W traffic)
//
// Build: /usr/local/cuda/bin/nvcc -O3 -std=c++20 -arch=sm_70 -Wno-deprecated-gpu-targets \
//        -diag-suppress 177,550 s116_gemm_tc.cu -lcublas -o /tmp/s116_gemm_tc
// Run:   CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 /tmp/s116_gemm_tc
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>
#include <cublas_v2.h>

static const int64_t GU_M = 1280, GU_K = 2560;
static const double GBPS_ROOF = 474e9;      // V100 measured f16 copy roofline (1.11/1.14)
static const int REPS = 60;
#ifndef QCACHE
#define QCACHE 4   // k8 windows in flight (power of two); matches the engine's TC_QCACHE (8 spills to local mem)
#endif

static double us_of(const char* tag, const std::function<void()>& f, cudaStream_t s) {
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

static __device__ __forceinline__ void mma816(float* d, unsigned a0, unsigned a1,
                                              unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
                   "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// Block = 8 warps covering m16 x n128 (blockIdx.x = m tile of 16, blockIdx.y = n chunk).
// Warps split the block's work as Nw n-warps (n16 each) x Kw k-warps (K/Kw each),
// Nw*Kw = 8: small ne -> more k-warps (all 8 warps active for ANY ne); large ne -> n-warps.
// Each k-warp accumulates its k-slice in ascending k16 order (deterministic), writes its
// m16 x n16 partial to smem; a fixed-order reduce over k-warps produces the block tile.
template<bool FULL>
__global__ void __launch_bounds__(256)
tc_gu_kernel(const __half* __restrict__ W, const __half* __restrict__ X,
             float* __restrict__ Y, int ne) {
    const int l = threadIdx.x & 31;
    const int w = threadIdx.x >> 5;
    const int m0 = blockIdx.x * 16;
    const int nbase = blockIdx.y * 128;
    // per-block decomposition: cover the block's valid rows with n16 tiles (nwarps of
    // them, power of two <= 8); the remaining warps split K. All 8 warps active for
    // any ne (last block may be partial: only its first `tiles` n-tiles are valid).
    const int nv = min(128, ne - nbase);
    const int tiles = (nv + 15) / 16;                       // 1..8 n16 tiles needed
    const int nwarp = tiles <= 1 ? 1 : (tiles <= 2 ? 2 : (tiles <= 4 ? 4 : 8));
    const int KW = 8 / nwarp;
    const int i = w % nwarp;                                // n-tile index
    const int j = w / nwarp;                                // k-slice index
    const bool active = i < tiles;                          // else: padding warp
    const int n0 = nbase + i * 16;
    const int SUB = (int) GU_K / KW;                        // k values per warp (mult of 16)
    const int64_t kbase = (int64_t) j * SUB;
    const int row = (l & 3) + ((l >> 4) & 1) * 4;           // A row / B col, 0..7
    // One mma.sync.m8n8k4.f16 = ONE k4 dot product (all lanes load the same k4 slice; the 4 lane groups
    const int64_t wa0 = (int64_t)(m0 + row) * GU_K + kbase;
    const int64_t wa1 = (int64_t)(m0 + row + 8) * GU_K + kbase;
    const int64_t xb0 = (int64_t)(n0 + row) * GU_K + kbase;
    const int64_t xb1 = (int64_t)(n0 + row + 8) * GU_K + kbase;

    float c0[2][2][8];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            #pragma unroll
            for (int i = 0; i < 8; ++i) c0[mt][nt][i] = 0.f;

    if (active) {
        // One mma.sync.m8n8k4.f16 = ONE k4 dot product (all lanes load the same k4 slice; the 4
        // lane groups replicate the tile - verified empirically). A k8 window = 4 sequential calls
        // (16B loads: 8 f16 per lane per operand). FULL=false (ne <= 8): skip the second n8
        // sub-tile entirely - halves the X over-read traffic (the microbench win at small ne).
        struct Q { uint4 a0, a1, b0, b1; };
        Q q[QCACHE];
        #pragma unroll
        for (int d = 0; d < QCACHE; ++d) {
            const int off = d * 8;
            q[d].a0 = *(const uint4*)&W[wa0 + off];
            q[d].a1 = *(const uint4*)&W[wa1 + off];
            q[d].b0 = *(const uint4*)&X[xb0 + off];
            if (FULL) q[d].b1 = *(const uint4*)&X[xb1 + off];
        }
        const int NWIN = SUB / 8;
        for (int t = 0; t < NWIN; ++t) {
            const int slot = t & (QCACHE - 1);
            const int koff = t * 8 + QCACHE * 8;
            mma816(c0[0][0], q[slot].a0.x, q[slot].a0.y, q[slot].b0.x, q[slot].b0.y);
            mma816(c0[1][0], q[slot].a1.x, q[slot].a1.y, q[slot].b0.x, q[slot].b0.y);
            mma816(c0[0][0], q[slot].a0.z, q[slot].a0.w, q[slot].b0.z, q[slot].b0.w);
            mma816(c0[1][0], q[slot].a1.z, q[slot].a1.w, q[slot].b0.z, q[slot].b0.w);
            if (FULL) {
                mma816(c0[0][1], q[slot].a0.x, q[slot].a0.y, q[slot].b1.x, q[slot].b1.y);
                mma816(c0[1][1], q[slot].a1.x, q[slot].a1.y, q[slot].b1.x, q[slot].b1.y);
                mma816(c0[0][1], q[slot].a0.z, q[slot].a0.w, q[slot].b1.z, q[slot].b1.w);
                mma816(c0[1][1], q[slot].a1.z, q[slot].a1.w, q[slot].b1.z, q[slot].b1.w);
            }
            if (koff < SUB) {   // consume first, then refill the same slot for t+D
                q[slot].a0 = *(const uint4*)&W[wa0 + koff];
                q[slot].a1 = *(const uint4*)&W[wa1 + koff];
                q[slot].b0 = *(const uint4*)&X[xb0 + koff];
                if (FULL) q[slot].b1 = *(const uint4*)&X[xb1 + koff];
            }
        }
    }
    // C/D layout: ci: row = (l&1) + 2*((i>>1)&1) + 4*((l>>4)&1); col = 4*((i>>2)&1) + 2*((l>>1)&1) + (i&1)
    const int rb = (l & 1) + 4 * ((l >> 4) & 1);
    const int cb = 2 * ((l >> 1) & 1);
    __shared__ float part[8][16][16];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int r = rb + 2 * ((i >> 1) & 1);
                const int c = cb + 4 * ((i >> 2) & 1) + (i & 1);
                part[w][mt * 8 + r][nt * 8 + c] = c0[mt][nt][i];
            }
    __syncthreads();
    // fixed-order reduce over k-warps, then store (rows < ne)
    for (int idx = threadIdx.x; idx < 2048; idx += 256) {
        const int n128 = idx >> 4, m16 = idx & 15;
        const int gn = nbase + n128;
        if (gn < ne) {
            const int ti = n128 >> 4, c16 = n128 & 15;
            float s = 0.f;
            #pragma unroll
            for (int k = 0; k < KW; ++k) s += part[ti + k * nwarp][m16][c16];
            Y[(int64_t) gn * GU_M + (m0 + m16)] = s;
        }
    }
}

static void tc_gu(const __half* W, const __half* X, float* Y, int ne, cudaStream_t s) {
    dim3 grid(80, (ne + 127) / 128);
    if (ne <= 8) tc_gu_kernel<false><<<grid, 256, 0, s>>>(W, X, Y, ne);
    else         tc_gu_kernel<true ><<<grid, 256, 0, s>>>(W, X, Y, ne);
}

int main() {
    cublasHandle_t h; cublasCreate(&h);
    cudaStream_t s; cudaStreamCreate(&s);
    cublasSetStream(h, s);
    size_t wspb = 32u << 20; void* wsp; cudaMalloc(&wsp, wspb); cublasSetWorkspace(h, wsp, wspb);
    const float alpha = 1.0f, beta = 0.0f;

    __half *Wgu, *Xd; float *Yu_ref, *Yu_tc;
    cudaMalloc(&Wgu, (size_t) GU_M * GU_K * 2);
    cudaMalloc(&Xd, (size_t) 640 * GU_K * 2);
    cudaMalloc(&Yu_ref, (size_t) 512 * GU_M * 4);
    cudaMalloc(&Yu_tc, (size_t) 512 * GU_M * 4);
    // Realistic SIGNED FINITE f16 (matches engine magnitudes: W ~ IQ3_XXS dequant in [-0.19,0.15],
    // X ~ post-attention residual in [-2.2,2.0]). The old LCG bit-patterns were all non-negative AND
    // produced +inf products in every k%16 class, which masked the missing-3/4-of-K bug.
    { std::vector<uint16_t> b((size_t) GU_M * GU_K), bx((size_t) 512 * GU_K);
      uint32_t s1 = 0x1234567u, s2 = 0xABCDEF0u;
      for (size_t i = 0; i < b.size(); ++i) {
          s1 = s1 * 1664525u + 1013904223u;
          b[i] = __half_as_ushort(__float2half_rn(0.346f * ((float)(s1 >> 8) / (float) 0xFFFFFF - 0.5f)));
      }
      for (size_t i = 0; i < bx.size(); ++i) {
          s2 = s2 * 1664525u + 1013904223u;
          bx[i] = __half_as_ushort(__float2half_rn(4.2f * ((float)(s2 >> 8) / (float) 0xFFFFFF - 0.5f)));
      }
      cudaMemcpy(Wgu, b.data(), (size_t) GU_M * GU_K * 2, cudaMemcpyHostToDevice);
      cudaMemcpy(Xd, bx.data(), (size_t) 512 * GU_K * 2, cudaMemcpyHostToDevice); }

    auto gemm_ref = [&](int ne) {
        cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) GU_M, ne, (int) GU_K, &alpha, Wgu,
                     CUDA_R_16F, (int) GU_K, Xd, CUDA_R_16F, (int) GU_K, &beta, Yu_ref,
                     CUDA_R_32F, (int) GU_M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    };

    printf("=== GU TC(m8n8k4 k8-window pipeline, L2-direct, FULL/SKIP8) vs cuBLAS(DEF): m=1280 k=2560, W 6.55 MB f16 (median of %d) ===\n", REPS);
    printf("ne    TC us   BW GB/s  %%roof   cuBLAS us  BW GB/s  %%roof   TC/cuBLAS  nneq maxAbs maxRelErr\n");
    const char* only = getenv("NE_ONLY"); int only_ne = only ? atoi(only) : 0;
    std::vector<int> nes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    if (const char* extra = getenv("NE_EXTRA")) {
        char buf[256]; strncpy(buf, extra, sizeof(buf) - 1); buf[sizeof(buf) - 1] = 0;
        for (char* tok = strtok(buf, ","); tok; tok = strtok(nullptr, ",")) {
            int v = atoi(tok);
            if (v > 0 && v <= 512) { bool dup = false; for (int e : nes) if (e == v) dup = true; if (!dup) nes.push_back(v); }
        }
    }
    for (int ne : nes) {
        if (only_ne && ne != only_ne) continue;
        gemm_ref(ne); cudaStreamSynchronize(s);
        tc_gu(Wgu, Xd, Yu_tc, ne, s); cudaStreamSynchronize(s);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) { printf("  !! CUDA error at ne=%d: %s\n", ne, cudaGetErrorString(err)); continue; }
        double maxre = 0, maxabs = 0; size_t nneq = 0;
        { std::vector<float> r((size_t) ne * GU_M), t((size_t) ne * GU_M);
          cudaMemcpy(r.data(), Yu_ref, r.size() * 4, cudaMemcpyDeviceToHost);
          cudaMemcpy(t.data(), Yu_tc, t.size() * 4, cudaMemcpyDeviceToHost);
          for (size_t i = 0; i < r.size(); ++i) {
              double diff = fabs((double) r[i] - (double) t[i]);
              if (diff > 0) nneq++;
              if (diff > maxabs) maxabs = diff;
              double rel = diff / (fabs((double) r[i]) + 1e-6);
              if (rel > maxre) maxre = rel;
          } }
        double t_tc = us_of("tc", [&] { tc_gu(Wgu, Xd, Yu_tc, ne, s); }, s);
        double t_cl = us_of("cl", [&] { gemm_ref(ne); }, s);
        double bw_tc = 6.553e6 / (t_tc * 1e-6) / 1e9;
        double bw_cl = 6.553e6 / (t_cl * 1e-6) / 1e9;
        printf("%3d   %7.1f  %8.0f %6.1f%%   %9.1f %8.0f %6.1f%%    %6.3f  %5zu %.1e %.2e\n",
               ne, t_tc, bw_tc, 100.0 * bw_tc / 474.0, t_cl, bw_cl, 100.0 * bw_cl / 474.0,
               t_tc / t_cl, nneq, maxabs, maxre);
        if (maxre > 1e-4) { printf("  !! maxRelErr %.2e at ne=%d\n", maxre, ne); }
    }
    printf("(roofline 474 GB/s = 6.55 MB W read in 13.8 us; BW = W bytes / kernel time)\n");
    return 0;
}
