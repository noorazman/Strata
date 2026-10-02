// bench/v100/s117_gemm_variants.cu — Stage 1.17: candidate kernels for the GU small/intermediate-ne GEMM.
//
// Y[ne,1280] = X[ne,2560] . W[1280,2560]^T   (f16 in, f32 out; the ONLY TC-routable engine shape).
//
// Single-expert variants (compared at ne = 8,16,32,64,128 vs the in-run cuBLAS baseline):
//   S0: Stage 1.16 production kernel (8 warps, m16 x n128, QCACHE=4, FULL/SKIP8) — reference.
//   S1: 16 warps (512 thr), m16 x n128 — doubles per-SM warps / in-flight loads at the same traffic.
//   S2: 8 warps, m32 x n128 — halves the X over-read (40 m-blocks instead of 80).
//   S3: 8 warps, m64 x n64 — 2-D tile: each m-row and each n-row read ~2x total (16.4 MB at ne=64
//       vs 32.75 MB for S0/S1), the trade from the Stage 1.16 analysis.
// Grouped variant (G experts, ONE launch):
//   G1: work-queue over (expert, m16, n16) tiles, deterministic prefix-sum tile assignment,
//       warp = one m16 x n16 x full-K tile (the S0 tile math, KW=1).
//
// Data: signed finite f16 (same LCG scheme as the Stage 1.16 bench: W in [-0.173,0.173],
// X in [-2.1,2.1]) — the 1.16 trap (f16-infinity LCG bits masking k-subset sums) is gone.
// Correctness: maxAbs vs the cuBLAS reference (f32 add-order, expect ~1e-5..1e-4, NOT 0).
//
// Build: /usr/local/cuda/bin/nvcc -O3 -std=c++20 -arch=sm_70 -Wno-deprecated-gpu-targets \
//        -diag-suppress 177,550 s117_gemm_variants.cu -lcublas -o /tmp/s117_bench
// Run:   CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 /tmp/s117_bench
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>
#include <cublas_v2.h>

static const int64_t M1280 = 1280, K2560 = 2560;
static const int REPS = 60;
static const int QCACHE = 4;

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

static __device__ __forceinline__ void mma816(float* d, unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
                   "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// S0: the Stage 1.16 production kernel (8 warps, m16 x n128, adaptive n/k warp split, FULL/SKIP8).
template<bool FULL>
__global__ void __launch_bounds__(256)
S0(const __half* __restrict__ W, const __half* __restrict__ X, float* __restrict__ Y, int ne) {
    const int l = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int m0 = blockIdx.x * 16, nbase = blockIdx.y * 128;
    const int nv = min(128, ne - nbase);
    const int tiles = (nv + 15) / 16;
    const int nwarp = tiles <= 1 ? 1 : (tiles <= 2 ? 2 : (tiles <= 4 ? 4 : 8));
    const int KW = 8 / nwarp;
    const int i = w % nwarp, j = w / nwarp;
    const bool active = i < tiles;
    const int n0 = nbase + i * 16;
    const int SUB = (int) K2560 / KW, kbase = j * SUB;
    const int row = (l & 3) + ((l >> 4) & 1) * 4;
    const int64_t wa0 = (int64_t)(m0 + row) * K2560 + kbase, wa1 = wa0 + 8 * K2560;
    const int64_t xb0 = (int64_t)(n0 + row) * K2560 + kbase, xb1 = xb0 + 8 * K2560;
    float c0[2][2][8];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) for (int nt = 0; nt < 2; ++nt) for (int ii = 0; ii < 8; ++ii) c0[mt][nt][ii] = 0.f;
    if (active) {
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
            if (koff < SUB) {
                q[slot].a0 = *(const uint4*)&W[wa0 + koff];
                q[slot].a1 = *(const uint4*)&W[wa1 + koff];
                q[slot].b0 = *(const uint4*)&X[xb0 + koff];
                if (FULL) q[slot].b1 = *(const uint4*)&X[xb1 + koff];
            }
        }
    }
    const int rb = (l & 1) + 4 * ((l >> 4) & 1), cb = 2 * ((l >> 1) & 1);
    __shared__ float part[8][16][16];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) for (int nt = 0; nt < 2; ++nt) for (int ii = 0; ii < 8; ++ii) {
        const int r = rb + 2 * ((ii >> 1) & 1), c = cb + 4 * ((ii >> 2) & 1) + (ii & 1);
        part[w][mt * 8 + r][nt * 8 + c] = c0[mt][nt][ii];
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < 2048; idx += 256) {
        const int n128 = idx >> 4, m16 = idx & 15;
        const int gn = nbase + n128;
        if (gn < ne) {
            const int ti = n128 >> 4, c16 = n128 & 15;
            float s = 0.f;
            #pragma unroll
            for (int k = 0; k < KW; ++k) s += part[ti + k * nwarp][m16][c16];
            Y[(int64_t) gn * M1280 + m0 + m16] = s;
        }
    }
}

// S1: 16 warps (512 thr), m16 x n128. Warps: i = w % tiles (n-tile, power-of-two <= 8),
// j = w / tiles (k-slice, KW = 16/tiles). All 16 warps active for any ne >= 1.
__global__ void __launch_bounds__(512)
S1(const __half* __restrict__ W, const __half* __restrict__ X, float* __restrict__ Y, int ne) {
    const int l = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int m0 = blockIdx.x * 16, nbase = blockIdx.y * 128;
    const int nv = min(128, ne - nbase);
    const int tiles = (nv + 15) / 16;
    const int p2 = tiles <= 1 ? 1 : (tiles <= 2 ? 2 : (tiles <= 4 ? 4 : 8));  // n-warps (power of two)
    const int KW = 16 / p2;
    const int i = w % p2, j = w / p2;
    const bool active = i < tiles;
    const int n0 = nbase + i * 16;
    const int SUB = (int) K2560 / KW, kbase = j * SUB;
    const int row = (l & 3) + ((l >> 4) & 1) * 4;
    const int64_t wa0 = (int64_t)(m0 + row) * K2560 + kbase, wa1 = wa0 + 8 * K2560;
    const int64_t xb0 = (int64_t)(n0 + row) * K2560 + kbase, xb1 = xb0 + 8 * K2560;
    float c0[2][2][8];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) for (int nt = 0; nt < 2; ++nt) for (int ii = 0; ii < 8; ++ii) c0[mt][nt][ii] = 0.f;
    if (active) {
        struct Q { uint4 a0, a1, b0, b1; };
        Q q[QCACHE];
        #pragma unroll
        for (int d = 0; d < QCACHE; ++d) {
            const int off = d * 8;
            q[d].a0 = *(const uint4*)&W[wa0 + off];
            q[d].a1 = *(const uint4*)&W[wa1 + off];
            q[d].b0 = *(const uint4*)&X[xb0 + off];
            q[d].b1 = *(const uint4*)&X[xb1 + off];
        }
        const int NWIN = SUB / 8;
        for (int t = 0; t < NWIN; ++t) {
            const int slot = t & (QCACHE - 1);
            const int koff = t * 8 + QCACHE * 8;
            mma816(c0[0][0], q[slot].a0.x, q[slot].a0.y, q[slot].b0.x, q[slot].b0.y);
            mma816(c0[1][0], q[slot].a1.x, q[slot].a1.y, q[slot].b0.x, q[slot].b0.y);
            mma816(c0[0][0], q[slot].a0.z, q[slot].a0.w, q[slot].b0.z, q[slot].b0.w);
            mma816(c0[1][0], q[slot].a1.z, q[slot].a1.w, q[slot].b0.z, q[slot].b0.w);
            mma816(c0[0][1], q[slot].a0.x, q[slot].a0.y, q[slot].b1.x, q[slot].b1.y);
            mma816(c0[1][1], q[slot].a1.x, q[slot].a1.y, q[slot].b1.x, q[slot].b1.y);
            mma816(c0[0][1], q[slot].a0.z, q[slot].a0.w, q[slot].b1.z, q[slot].b1.w);
            mma816(c0[1][1], q[slot].a1.z, q[slot].a1.w, q[slot].b1.z, q[slot].b1.w);
            if (koff < SUB) {
                q[slot].a0 = *(const uint4*)&W[wa0 + koff];
                q[slot].a1 = *(const uint4*)&W[wa1 + koff];
                q[slot].b0 = *(const uint4*)&X[xb0 + koff];
                q[slot].b1 = *(const uint4*)&X[xb1 + koff];
            }
        }
    }
    const int rb = (l & 1) + 4 * ((l >> 4) & 1), cb = 2 * ((l >> 1) & 1);
    __shared__ float part[16][16][16];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) for (int nt = 0; nt < 2; ++nt) for (int ii = 0; ii < 8; ++ii) {
        const int r = rb + 2 * ((ii >> 1) & 1), c = cb + 4 * ((ii >> 2) & 1) + (ii & 1);
        part[w][mt * 8 + r][nt * 8 + c] = c0[mt][nt][ii];
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < 2048; idx += 512) {
        const int n128 = idx >> 4, m16 = idx & 15;
        const int gn = nbase + n128;
        if (gn < ne) {
            const int ti = n128 >> 4, c16 = n128 & 15;   // ti 0..7 = n16 column
            float s = 0.f;
            #pragma unroll
            for (int k = 0; k < KW; ++k) s += part[ti + k * p2][m16][c16];
            Y[(int64_t) gn * M1280 + m0 + m16] = s;
        }
    }
}

// S2: 8 warps, m32 x n64. Warps 0-3 -> m16 half 0, 4-7 -> half 1. Within a half (4 warps):
// i = w2 % p2 (n-tile, p2 in {1,2,4}), j = w2 / p2 (k-slice, KW = 4/p2).
__global__ void __launch_bounds__(256)
S2(const __half* __restrict__ W, const __half* __restrict__ X, float* __restrict__ Y, int ne) {
    const int l = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int m0 = blockIdx.x * 32, nbase = blockIdx.y * 64;
    const int h = w >> 2;                       // m16 half (0/1)
    const int w2 = w & 3;
    const int nv = min(64, ne - nbase);
    const int tiles = (nv + 15) / 16;
    const int p2 = tiles <= 1 ? 1 : (tiles <= 2 ? 2 : 4);
    const int KW = 4 / p2;
    const int i = w2 % p2, j = w2 / p2;
    const bool active = i < tiles;
    const int n0 = nbase + i * 16;
    const int SUB = (int) K2560 / KW, kbase = j * SUB;
    const int row = (l & 3) + ((l >> 4) & 1) * 4;
    const int64_t wa0 = (int64_t)(m0 + h * 16 + row) * K2560 + kbase, wa1 = wa0 + 8 * K2560;
    const int64_t xb0 = (int64_t)(n0 + row) * K2560 + kbase, xb1 = xb0 + 8 * K2560;
    float c0[2][2][8];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) for (int nt = 0; nt < 2; ++nt) for (int ii = 0; ii < 8; ++ii) c0[mt][nt][ii] = 0.f;
    if (active) {
        struct Q { uint4 a0, a1, b0, b1; };
        Q q[QCACHE];
        #pragma unroll
        for (int d = 0; d < QCACHE; ++d) {
            const int off = d * 8;
            q[d].a0 = *(const uint4*)&W[wa0 + off];
            q[d].a1 = *(const uint4*)&W[wa1 + off];
            q[d].b0 = *(const uint4*)&X[xb0 + off];
            q[d].b1 = *(const uint4*)&X[xb1 + off];
        }
        const int NWIN = SUB / 8;
        for (int t = 0; t < NWIN; ++t) {
            const int slot = t & (QCACHE - 1);
            const int koff = t * 8 + QCACHE * 8;
            mma816(c0[0][0], q[slot].a0.x, q[slot].a0.y, q[slot].b0.x, q[slot].b0.y);
            mma816(c0[1][0], q[slot].a1.x, q[slot].a1.y, q[slot].b0.x, q[slot].b0.y);
            mma816(c0[0][0], q[slot].a0.z, q[slot].a0.w, q[slot].b0.z, q[slot].b0.w);
            mma816(c0[1][0], q[slot].a1.z, q[slot].a1.w, q[slot].b0.z, q[slot].b0.w);
            mma816(c0[0][1], q[slot].a0.x, q[slot].a0.y, q[slot].b1.x, q[slot].b1.y);
            mma816(c0[1][1], q[slot].a1.x, q[slot].a1.y, q[slot].b1.x, q[slot].b1.y);
            mma816(c0[0][1], q[slot].a0.z, q[slot].a0.w, q[slot].b1.z, q[slot].b1.w);
            mma816(c0[1][1], q[slot].a1.z, q[slot].a1.w, q[slot].b1.z, q[slot].b1.w);
            if (koff < SUB) {
                q[slot].a0 = *(const uint4*)&W[wa0 + koff];
                q[slot].a1 = *(const uint4*)&W[wa1 + koff];
                q[slot].b0 = *(const uint4*)&X[xb0 + koff];
                q[slot].b1 = *(const uint4*)&X[xb1 + koff];
            }
        }
    }
    const int rb = (l & 1) + 4 * ((l >> 4) & 1), cb = 2 * ((l >> 1) & 1);
    __shared__ float part[8][16][16];   // [warp][m16][n16col]
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) for (int nt = 0; nt < 2; ++nt) for (int ii = 0; ii < 8; ++ii) {
        const int r = rb + 2 * ((ii >> 1) & 1), c = cb + 4 * ((ii >> 2) & 1) + (ii & 1);
        part[w][mt * 8 + r][nt * 8 + c] = c0[mt][nt][ii];
    }
    __syncthreads();
    // 64 n rows x 32 m rows = 2048 cells; warp for (hh, n16col, k): w = hh*4 + n16col + k*p2.
    for (int idx = threadIdx.x; idx < 2048; idx += 256) {
        const int n_row = idx & 63, m_row = idx >> 6;
        const int hh = m_row >> 4, m16 = m_row & 15;
        const int n16col = n_row >> 4, rn16 = n_row & 15;
        const int gn = nbase + n_row;
        if (gn < ne) {
            float s = 0.f;
            #pragma unroll
            for (int k = 0; k < KW; ++k) s += part[hh * 4 + n16col + k * p2][m16][rn16];
            Y[(int64_t) gn * M1280 + m0 + m_row] = s;
        }
    }
}

// S3: 8 warps, m64 x n64. Warp w owns m16 tiles {2*(w&1), 2*(w&1)+1} and n16 tile (w>>1):
// TWO m16 x n16 tiles (adjacent m-halves, same n-column), full K, KW=1. Each m16 x n16 tile of
// the block is computed by exactly one warp. Traffic at ne=64: W x1 + X x20 (13.1 MB vs 32.75
// for S0/S1).
__global__ void __launch_bounds__(256)
S3(const __half* __restrict__ W, const __half* __restrict__ X, float* __restrict__ Y, int ne) {
    const int l = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int m0 = blockIdx.x * 64, nbase = blockIdx.y * 64;
    const int nc = w >> 1;                 // n16 column 0..3
    const int n0 = nbase + nc * 16;
    const bool active = n0 < ne;
    const int row = (l & 3) + ((l >> 4) & 1) * 4;
    const int64_t xb0 = (int64_t)(n0 + row) * K2560, xb1 = xb0 + 8 * K2560;
    float cA[2][2][8], cB[2][2][8];        // tile A = low m16 of the warp's 32-row span, tile B = high
    if (active) {
        const int64_t wbase = (int64_t)(m0 + (w & 1) * 32 + row) * K2560;
        const int64_t wA0 = wbase, wA1 = wbase + 8 * K2560;
        const int64_t wB0 = wA0 + 16 * K2560, wB1 = wB0 + 8 * K2560;
        struct Q { uint4 a0, a1, b0, b1; };   // a = tile A rows, b = tile B rows
        Q q[QCACHE];
        #pragma unroll
        for (int d = 0; d < QCACHE; ++d) {
            const int off = d * 8;
            q[d].a0 = *(const uint4*)&W[wA0 + off];
            q[d].a1 = *(const uint4*)&W[wA1 + off];
            q[d].b0 = *(const uint4*)&W[wB0 + off];
            q[d].b1 = *(const uint4*)&W[wB1 + off];
        }
        uint4 xa = *(const uint4*)&X[xb0];
        uint4 xb = *(const uint4*)&X[xb1];
        const int NWIN = K2560 / 8;
        for (int t = 0; t < NWIN; ++t) {
            const int slot = t & (QCACHE - 1);
            const int koff = t * 8 + QCACHE * 8;
            mma816(cA[0][0], q[slot].a0.x, q[slot].a0.y, xa.x, xa.y);
            mma816(cA[1][0], q[slot].a1.x, q[slot].a1.y, xa.x, xa.y);
            mma816(cA[0][0], q[slot].a0.z, q[slot].a0.w, xa.z, xa.w);
            mma816(cA[1][0], q[slot].a1.z, q[slot].a1.w, xa.z, xa.w);
            mma816(cA[0][1], q[slot].a0.x, q[slot].a0.y, xb.x, xb.y);
            mma816(cA[1][1], q[slot].a1.x, q[slot].a1.y, xb.x, xb.y);
            mma816(cA[0][1], q[slot].a0.z, q[slot].a0.w, xb.z, xb.w);
            mma816(cA[1][1], q[slot].a1.z, q[slot].a1.w, xb.z, xb.w);
            mma816(cB[0][0], q[slot].b0.x, q[slot].b0.y, xa.x, xa.y);
            mma816(cB[1][0], q[slot].b1.x, q[slot].b1.y, xa.x, xa.y);
            mma816(cB[0][0], q[slot].b0.z, q[slot].b0.w, xa.z, xa.w);
            mma816(cB[1][0], q[slot].b1.z, q[slot].b1.w, xa.z, xa.w);
            mma816(cB[0][1], q[slot].b0.x, q[slot].b0.y, xb.x, xb.y);
            mma816(cB[1][1], q[slot].b1.x, q[slot].b1.y, xb.x, xb.y);
            mma816(cB[0][1], q[slot].b0.z, q[slot].b0.w, xb.z, xb.w);
            mma816(cB[1][1], q[slot].b1.z, q[slot].b1.w, xb.z, xb.w);
            if (koff < K2560) {
                q[slot].a0 = *(const uint4*)&W[wA0 + koff];
                q[slot].a1 = *(const uint4*)&W[wA1 + koff];
                q[slot].b0 = *(const uint4*)&W[wB0 + koff];
                q[slot].b1 = *(const uint4*)&W[wB1 + koff];
                xa = *(const uint4*)&X[xb0 + koff];
                xb = *(const uint4*)&X[xb1 + koff];
            }
        }
    }
    const int rb = (l & 1) + 4 * ((l >> 4) & 1), cb = 2 * ((l >> 1) & 1);
    __shared__ float part[8][2][16][16];   // [warp][tile A/B][m16][n16]
    #pragma unroll
    for (int t = 0; t < 2; ++t)
        #pragma unroll
        for (int mt = 0; mt < 2; ++mt) for (int nt = 0; nt < 2; ++nt) for (int ii = 0; ii < 8; ++ii) {
            const int r = rb + 2 * ((ii >> 1) & 1), c = cb + 4 * ((ii >> 2) & 1) + (ii & 1);
            const float v = t ? cB[mt][nt][ii] : cA[mt][nt][ii];
            part[w][t][mt * 8 + r][nt * 8 + c] = v;
        }
    __syncthreads();
    // 64 m rows x 64 n rows; warp for (m16 tile mp, n16 col nc) = (nc << 1) | (mp & 1); tile t = mp >> 1.
    for (int idx = threadIdx.x; idx < 4096; idx += 256) {
        const int n_row = idx & 63, m_row = idx >> 6;
        const int nc2 = n_row >> 4, rn = n_row & 15;
        const int mp = m_row >> 4, m16 = m_row & 15;
        const int gn = nbase + n_row;
        if (gn < ne) {
            const int w2 = (nc2 << 1) | (mp & 1);
            const int t = mp >> 1;
            Y[(int64_t) gn * M1280 + m0 + m_row] = part[w2][t][m16][rn];
        }
    }
}

// S6: 16 warps (512 thr), m8 x n128, grid (160, ceil(ne/128)). The intermediate-ne occupancy fix:
// m8 blocks tile the 1280 rows with no overlap -> W is read ONCE per GEMM (6.55 MB, like S0),
// but 160 blocks x 16 warps = ~29 warps/SM (S0: 80 blocks x 8 warps = ~7 warps/SM at ne=32/64).
__global__ void __launch_bounds__(512)
S6(const __half* __restrict__ W, const __half* __restrict__ X, float* __restrict__ Y, int ne) {
    const int l = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int m0 = blockIdx.x * 8, nbase = blockIdx.y * 128;
    const int nv = min(128, ne - nbase);
    const int tiles = (nv + 15) / 16;
    const int p2 = tiles <= 1 ? 1 : (tiles <= 2 ? 2 : (tiles <= 4 ? 4 : 8));
    const int KW = 16 / p2;
    const int i = w % p2, j = w / p2;
    const bool active = i < tiles;
    const int n0 = nbase + i * 16;
    const int SUB = (int) K2560 / KW, kbase = j * SUB;
    const int row = (l & 3) + ((l >> 4) & 1) * 4;
    const int64_t wa0 = (int64_t)(m0 + row) * K2560 + kbase;
    const int64_t xb0 = (int64_t)(n0 + row) * K2560 + kbase, xb1 = xb0 + 8 * K2560;
    float c0[2][8];
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) for (int ii = 0; ii < 8; ++ii) c0[nt][ii] = 0.f;
    if (active) {
        struct Q { uint4 a0, b0, b1; };
        Q q[QCACHE];
        #pragma unroll
        for (int d = 0; d < QCACHE; ++d) {
            const int off = d * 8;
            q[d].a0 = *(const uint4*)&W[wa0 + off];
            q[d].b0 = *(const uint4*)&X[xb0 + off];
            q[d].b1 = *(const uint4*)&X[xb1 + off];
        }
        const int NWIN = SUB / 8;
        for (int t = 0; t < NWIN; ++t) {
            const int slot = t & (QCACHE - 1);
            const int koff = t * 8 + QCACHE * 8;
            mma816(c0[0], q[slot].a0.x, q[slot].a0.y, q[slot].b0.x, q[slot].b0.y);
            mma816(c0[0], q[slot].a0.z, q[slot].a0.w, q[slot].b0.z, q[slot].b0.w);
            mma816(c0[1], q[slot].a0.x, q[slot].a0.y, q[slot].b1.x, q[slot].b1.y);
            mma816(c0[1], q[slot].a0.z, q[slot].a0.w, q[slot].b1.z, q[slot].b1.w);
            if (koff < SUB) {
                q[slot].a0 = *(const uint4*)&W[wa0 + koff];
                q[slot].b0 = *(const uint4*)&X[xb0 + koff];
                q[slot].b1 = *(const uint4*)&X[xb1 + koff];
            }
        }
    }
    const int rb = (l & 1) + 4 * ((l >> 4) & 1), cb = 2 * ((l >> 1) & 1);
    __shared__ float part[16][16][8];
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt)
        #pragma unroll
        for (int ii = 0; ii < 8; ++ii) {
            const int r = rb + 2 * ((ii >> 1) & 1);
            const int c = cb + 4 * ((ii >> 2) & 1) + (ii & 1);
            part[w][nt * 8 + c][r] = c0[nt][ii];
        }
    __syncthreads();
    for (int idx = threadIdx.x; idx < 1024; idx += 512) {
        const int n128 = idx >> 3, m8 = idx & 7;
        const int gn = nbase + n128;
        if (gn < ne) {
            const int ti = n128 >> 4;
            float s = 0.f;
            #pragma unroll
            for (int k = 0; k < KW; ++k) s += part[ti + k * p2][n128 & 15][m8];
            Y[(int64_t) gn * M1280 + m0 + m8] = s;
        }
    }
}

// S12: m16 x n16 tile, 128 threads (4 warps), in-block 4-way k-split (deterministic smem reduce).
// grid (80, ceil(ne/16)) -> at ne=64: 320 blocks x 4 warps = 16 warps/SM; W read once (6.55 MB).
// The cuBLAS structure (ncu: cutlass s884 64x64 + 12 inter-block k-splits = 240 blocks) but
// deterministic (in-block split) and with no second reduce kernel.
__global__ void __launch_bounds__(128)
S12(const __half* __restrict__ W, const __half* __restrict__ X, float* __restrict__ Y, int ne) {
    const int l = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int m0 = blockIdx.x * 16, nbase = blockIdx.y * 16;
    const int nv = min(16, ne - nbase);
    const bool active = nv > 0;
    const int n0 = nbase;
    const int SUB = (int) K2560 / 4, kbase = w * SUB;
    const int row = (l & 3) + ((l >> 4) & 1) * 4;
    const int64_t wa0 = (int64_t)(m0 + row) * K2560 + kbase, wa1 = wa0 + 8 * K2560;
    const int64_t xb0 = (int64_t)(n0 + row) * K2560 + kbase, xb1 = xb0 + 8 * K2560;
    float c0[2][2][8];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) for (int nt = 0; nt < 2; ++nt) for (int ii = 0; ii < 8; ++ii) c0[mt][nt][ii] = 0.f;
    if (active) {
        struct Q { uint4 a0, a1, b0, b1; };
        Q q[QCACHE];
        #pragma unroll
        for (int d = 0; d < QCACHE; ++d) {
            const int off = d * 8;
            q[d].a0 = *(const uint4*)&W[wa0 + off];
            q[d].a1 = *(const uint4*)&W[wa1 + off];
            q[d].b0 = *(const uint4*)&X[xb0 + off];
            q[d].b1 = *(const uint4*)&X[xb1 + off];
        }
        const int NWIN = SUB / 8;
        for (int t = 0; t < NWIN; ++t) {
            const int slot = t & (QCACHE - 1);
            const int koff = t * 8 + QCACHE * 8;
            mma816(c0[0][0], q[slot].a0.x, q[slot].a0.y, q[slot].b0.x, q[slot].b0.y);
            mma816(c0[1][0], q[slot].a1.x, q[slot].a1.y, q[slot].b0.x, q[slot].b0.y);
            mma816(c0[0][0], q[slot].a0.z, q[slot].a0.w, q[slot].b0.z, q[slot].b0.w);
            mma816(c0[1][0], q[slot].a1.z, q[slot].a1.w, q[slot].b0.z, q[slot].b0.w);
            mma816(c0[0][1], q[slot].a0.x, q[slot].a0.y, q[slot].b1.x, q[slot].b1.y);
            mma816(c0[1][1], q[slot].a1.x, q[slot].a1.y, q[slot].b1.x, q[slot].b1.y);
            mma816(c0[0][1], q[slot].a0.z, q[slot].a0.w, q[slot].b1.z, q[slot].b1.w);
            mma816(c0[1][1], q[slot].a1.z, q[slot].a1.w, q[slot].b1.z, q[slot].b1.w);
            if (koff < SUB) {
                q[slot].a0 = *(const uint4*)&W[wa0 + koff];
                q[slot].a1 = *(const uint4*)&W[wa1 + koff];
                q[slot].b0 = *(const uint4*)&X[xb0 + koff];
                q[slot].b1 = *(const uint4*)&X[xb1 + koff];
            }
        }
    }
    const int rb = (l & 1) + 4 * ((l >> 4) & 1), cb = 2 * ((l >> 1) & 1);
    __shared__ float part[4][16][16];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) for (int nt = 0; nt < 2; ++nt) for (int ii = 0; ii < 8; ++ii) {
        const int r = rb + 2 * ((ii >> 1) & 1), c = cb + 4 * ((ii >> 2) & 1) + (ii & 1);
        part[w][mt * 8 + r][nt * 8 + c] = c0[mt][nt][ii];
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < 256; idx += 128) {
        const int n16 = idx >> 4, m16 = idx & 15;
        const int gn = nbase + n16;
        if (gn < ne) {
            float s = 0.f;
            #pragma unroll
            for (int k = 0; k < 4; ++k) s += part[k][m16][n16];
            Y[(int64_t) gn * M1280 + m0 + m16] = s;
        }
    }
}

// S14: m16 x n16 tile, 256 threads (8 warps), in-block 8-way k-split (deterministic smem reduce).
// 320 blocks x 8 warps at ne=64 -> 32 warps/SM, 8x the in-flight bytes of S12.
__global__ void __launch_bounds__(256)
S14(const __half* __restrict__ W, const __half* __restrict__ X, float* __restrict__ Y, int ne) {
    const int l = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int m0 = blockIdx.x * 16, nbase = blockIdx.y * 16;
    const int nv = min(16, ne - nbase);
    const bool active = nv > 0;
    const int n0 = nbase;
    const int SUB = (int) K2560 / 8, kbase = w * SUB;
    const int row = (l & 3) + ((l >> 4) & 1) * 4;
    const int64_t wa0 = (int64_t)(m0 + row) * K2560 + kbase, wa1 = wa0 + 8 * K2560;
    const int64_t xb0 = (int64_t)(n0 + row) * K2560 + kbase, xb1 = xb0 + 8 * K2560;
    float c0[2][2][8];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) for (int nt = 0; nt < 2; ++nt) for (int ii = 0; ii < 8; ++ii) c0[mt][nt][ii] = 0.f;
    if (active) {
        struct Q { uint4 a0, a1, b0, b1; };
        Q q[QCACHE];
        #pragma unroll
        for (int d = 0; d < QCACHE; ++d) {
            const int off = d * 8;
            q[d].a0 = *(const uint4*)&W[wa0 + off];
            q[d].a1 = *(const uint4*)&W[wa1 + off];
            q[d].b0 = *(const uint4*)&X[xb0 + off];
            q[d].b1 = *(const uint4*)&X[xb1 + off];
        }
        const int NWIN = SUB / 8;
        for (int t = 0; t < NWIN; ++t) {
            const int slot = t & (QCACHE - 1);
            const int koff = t * 8 + QCACHE * 8;
            mma816(c0[0][0], q[slot].a0.x, q[slot].a0.y, q[slot].b0.x, q[slot].b0.y);
            mma816(c0[1][0], q[slot].a1.x, q[slot].a1.y, q[slot].b0.x, q[slot].b0.y);
            mma816(c0[0][0], q[slot].a0.z, q[slot].a0.w, q[slot].b0.z, q[slot].b0.w);
            mma816(c0[1][0], q[slot].a1.z, q[slot].a1.w, q[slot].b0.z, q[slot].b0.w);
            mma816(c0[0][1], q[slot].a0.x, q[slot].a0.y, q[slot].b1.x, q[slot].b1.y);
            mma816(c0[1][1], q[slot].a1.x, q[slot].a1.y, q[slot].b1.x, q[slot].b1.y);
            mma816(c0[0][1], q[slot].a0.z, q[slot].a0.w, q[slot].b1.z, q[slot].b1.w);
            mma816(c0[1][1], q[slot].a1.z, q[slot].a1.w, q[slot].b1.z, q[slot].b1.w);
            if (koff < SUB) {
                q[slot].a0 = *(const uint4*)&W[wa0 + koff];
                q[slot].a1 = *(const uint4*)&W[wa1 + koff];
                q[slot].b0 = *(const uint4*)&X[xb0 + koff];
                q[slot].b1 = *(const uint4*)&X[xb1 + koff];
            }
        }
    }
    const int rb = (l & 1) + 4 * ((l >> 4) & 1), cb = 2 * ((l >> 1) & 1);
    __shared__ float part[8][16][16];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) for (int nt = 0; nt < 2; ++nt) for (int ii = 0; ii < 8; ++ii) {
        const int r = rb + 2 * ((ii >> 1) & 1), c = cb + 4 * ((ii >> 2) & 1) + (ii & 1);
        part[w][mt * 8 + r][nt * 8 + c] = c0[mt][nt][ii];
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < 256; idx += 256) {
        const int n16 = idx >> 4, m16 = idx & 15;
        const int gn = nbase + n16;
        if (gn < ne) {
            float s = 0.f;
            #pragma unroll
            for (int k = 0; k < 8; ++k) s += part[k][m16][n16];
            Y[(int64_t) gn * M1280 + m0 + m16] = s;
        }
    }
}

// G1: grouped work-queue kernel. Tiles = (expert e, m16 m, n16 n); total tiles =
// sum_e 80 * ceil(ne_e/16). Block b handles tiles [8b, 8b+8); warp = one tile, full K (S0 math).
// Deterministic: per-expert tile prefix (device array, binary search per warp). X rows beyond
// ne_e are read by padding n-tiles (finite f16 in the engine's padded Xs); Y rows masked.
struct GExpert { const __half* X; const __half* W; float* Y; int ne; int tile0; };
__global__ void __launch_bounds__(256)
G1(const GExpert* __restrict__ E, int G, int nexperts_valid) {
    const int l = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int t = blockIdx.x * 8 + w;
    if (t >= E[G - 1].tile0 + 80 * ((E[G - 1].ne + 15) / 16)) return;
    // binary search: smallest e in [0, G) with tile prefix[e] > t (lo may reach G); the containing
    // expert is e = lo - 1 (t < total tiles, so lo >= 1 and E[lo-1].tile0 <= t).
    int lo = 0, hi = nexperts_valid;
    while (lo < hi) { const int mid = (lo + hi) >> 1; if (E[mid].tile0 <= t) lo = mid + 1; else hi = mid; }
    const int e = lo - 1;
    const int tel = t - E[e].tile0;
    const int nt = (E[e].ne + 15) / 16;
    const int m = tel % 80, n = tel / 80;
    if (n >= nt) return;
    const GExpert ex = E[e];
    const int m0 = m * 16, n0 = n * 16;
    const int row = (l & 3) + ((l >> 4) & 1) * 4;
    const int64_t wa0 = (int64_t)(m0 + row) * K2560, wa1 = wa0 + 8 * K2560;
    const int64_t xb0 = (int64_t)(n0 + row) * K2560, xb1 = xb0 + 8 * K2560;
    float c0[2][2][8];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) for (int nt = 0; nt < 2; ++nt) for (int ii = 0; ii < 8; ++ii) c0[mt][nt][ii] = 0.f;
    {
        struct Q { uint4 a0, a1, b0, b1; };
        Q q[QCACHE];
        #pragma unroll
        for (int d = 0; d < QCACHE; ++d) {
            const int off = d * 8;
            q[d].a0 = *(const uint4*)&ex.W[wa0 + off];
            q[d].a1 = *(const uint4*)&ex.W[wa1 + off];
            q[d].b0 = *(const uint4*)&ex.X[xb0 + off];
            q[d].b1 = *(const uint4*)&ex.X[xb1 + off];
        }
        const int NWIN = K2560 / 8;
        for (int tt = 0; tt < NWIN; ++tt) {
            const int slot = tt & (QCACHE - 1);
            const int koff = tt * 8 + QCACHE * 8;
            mma816(c0[0][0], q[slot].a0.x, q[slot].a0.y, q[slot].b0.x, q[slot].b0.y);
            mma816(c0[1][0], q[slot].a1.x, q[slot].a1.y, q[slot].b0.x, q[slot].b0.y);
            mma816(c0[0][0], q[slot].a0.z, q[slot].a0.w, q[slot].b0.z, q[slot].b0.w);
            mma816(c0[1][0], q[slot].a1.z, q[slot].a1.w, q[slot].b0.z, q[slot].b0.w);
            mma816(c0[0][1], q[slot].a0.x, q[slot].a0.y, q[slot].b1.x, q[slot].b1.y);
            mma816(c0[1][1], q[slot].a1.x, q[slot].a1.y, q[slot].b1.x, q[slot].b1.y);
            mma816(c0[0][1], q[slot].a0.z, q[slot].a0.w, q[slot].b1.z, q[slot].b1.w);
            mma816(c0[1][1], q[slot].a1.z, q[slot].a1.w, q[slot].b1.z, q[slot].b1.w);
            if (koff < K2560) {
                q[slot].a0 = *(const uint4*)&ex.W[wa0 + koff];
                q[slot].a1 = *(const uint4*)&ex.W[wa1 + koff];
                q[slot].b0 = *(const uint4*)&ex.X[xb0 + koff];
                q[slot].b1 = *(const uint4*)&ex.X[xb1 + koff];
            }
        }
    }
    const int rb = (l & 1) + 4 * ((l >> 4) & 1), cb = 2 * ((l >> 1) & 1);
    // direct store: each lane's c0[ii] maps to (r, c); row = n0 + c, col = m0 + r.
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            #pragma unroll
            for (int ii = 0; ii < 8; ++ii) {
                const int r = rb + 2 * ((ii >> 1) & 1);
                const int c = cb + 4 * ((ii >> 2) & 1) + (ii & 1);
                const int gr = n0 + nt * 8 + c, gc = m0 + mt * 8 + r;
                if (gr < E[e].ne) ex.Y[(int64_t) gr * M1280 + gc] = c0[mt][nt][ii];
            }
}

// G2: grouped S12 — ONE launch over all experts. Block = one (expert, m16, n16) tile running the
// S12 body (128 thr, 4-way in-block k-split, deterministic smem reduce). Per-block expert lookup =
// binary search over the G1 E table (tile0 = 80 x n16 prefix -> n16 index = tile0 / 80).
// Grid: (80, total_n16_tiles).
__global__ void __launch_bounds__(128)
G2(const GExpert* __restrict__ E, int G) {
    const int t = blockIdx.y;
    int lo = 0, hi = G;
    while (lo < hi) { const int mid = (lo + hi) >> 1; if (E[mid].tile0 / 80 <= t) lo = mid + 1; else hi = mid; }
    const GExpert Ee = E[lo - 1];
    const int tel = t - Ee.tile0 / 80;
    const int m0 = blockIdx.x * 16, n0 = tel * 16;
    const int ne = Ee.ne;
    const bool active = n0 < ne;
    const int SUB = (int) K2560 / 4, kbase = (threadIdx.x >> 5) * SUB;
    const int l = threadIdx.x & 31;
    const int row = (l & 3) + ((l >> 4) & 1) * 4;
    const int64_t wa0 = (int64_t)(m0 + row) * K2560 + kbase, wa1 = wa0 + 8 * K2560;
    const int64_t xb0 = (int64_t)(n0 + row) * K2560 + kbase, xb1 = xb0 + 8 * K2560;
    const __half* W = Ee.W; const __half* X = Ee.X; float* Y = Ee.Y;
    float c0[2][2][8];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) for (int nt = 0; nt < 2; ++nt) for (int ii = 0; ii < 8; ++ii) c0[mt][nt][ii] = 0.f;
    if (active) {
        struct Q { uint4 a0, a1, b0, b1; };
        Q q[QCACHE];
        #pragma unroll
        for (int d = 0; d < QCACHE; ++d) {
            const int off = d * 8;
            q[d].a0 = *(const uint4*)&W[wa0 + off];
            q[d].a1 = *(const uint4*)&W[wa1 + off];
            q[d].b0 = *(const uint4*)&X[xb0 + off];
            q[d].b1 = *(const uint4*)&X[xb1 + off];
        }
        const int NWIN = SUB / 8;
        for (int tt = 0; tt < NWIN; ++tt) {
            const int slot = tt & (QCACHE - 1);
            const int koff = tt * 8 + QCACHE * 8;
            mma816(c0[0][0], q[slot].a0.x, q[slot].a0.y, q[slot].b0.x, q[slot].b0.y);
            mma816(c0[1][0], q[slot].a1.x, q[slot].a1.y, q[slot].b0.x, q[slot].b0.y);
            mma816(c0[0][0], q[slot].a0.z, q[slot].a0.w, q[slot].b0.z, q[slot].b0.w);
            mma816(c0[1][0], q[slot].a1.z, q[slot].a1.w, q[slot].b0.z, q[slot].b0.w);
            mma816(c0[0][1], q[slot].a0.x, q[slot].a0.y, q[slot].b1.x, q[slot].b1.y);
            mma816(c0[1][1], q[slot].a1.x, q[slot].a1.y, q[slot].b1.x, q[slot].b1.y);
            mma816(c0[0][1], q[slot].a0.z, q[slot].a0.w, q[slot].b1.z, q[slot].b1.w);
            mma816(c0[1][1], q[slot].a1.z, q[slot].a1.w, q[slot].b1.z, q[slot].b1.w);
            if (koff < SUB) {
                q[slot].a0 = *(const uint4*)&W[wa0 + koff];
                q[slot].a1 = *(const uint4*)&W[wa1 + koff];
                q[slot].b0 = *(const uint4*)&X[xb0 + koff];
                q[slot].b1 = *(const uint4*)&X[xb1 + koff];
            }
        }
    }
    const int rb = (l & 1) + 4 * ((l >> 4) & 1), cb = 2 * ((l >> 1) & 1);
    __shared__ float part[4][16][16];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) for (int nt = 0; nt < 2; ++nt) for (int ii = 0; ii < 8; ++ii) {
        const int r = rb + 2 * ((ii >> 1) & 1), c = cb + 4 * ((ii >> 2) & 1) + (ii & 1);
        part[threadIdx.x >> 5][mt * 8 + r][nt * 8 + c] = c0[mt][nt][ii];
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < 256; idx += 128) {
        const int n16 = idx >> 4, m16 = idx & 15;
        const int gn = n0 + n16;
        if (gn < ne) {
            float s2 = 0.f;
            #pragma unroll
            for (int k = 0; k < 4; ++k) s2 += part[k][m16][n16];
            Y[(int64_t) gn * M1280 + m0 + m16] = s2;
        }
    }
}

// ---------------- host ----------------
__global__ void cmpmax(const float* a, const float* b, int64_t n, float* out) {
    int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    float mx = 0.f;
    if (i < n) mx = fabsf(a[i] - b[i]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
    if ((threadIdx.x & 31) == 0) atomicMax((unsigned*) (void*) out, __float_as_uint(mx));
}
static float host_maxabs(const float* a, const float* b, int64_t n, cudaStream_t s) {
    float* d; cudaMalloc(&d, 4);
    cudaMemsetAsync(d, 0, 4, s);  // atomicMax target: zeroed on s, ordered before cmpmax
    cmpmax<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(a, b, n, d);
    cudaStreamSynchronize(s);  // the D2H below runs on the default stream: no cross-stream read of d
    float h = 0; cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost); cudaFree(d);
    return h;
}

static std::vector<__half> gen_f16(int64_t n, unsigned& seed, float scale) {
    std::vector<__half> v(n);
    for (int64_t i = 0; i < n; ++i) {
        seed = seed * 1664525u + 1013904223u;
        float u = (float) (seed >> 8) / (float) 0xFFFFFF;      // 0..1
        v[i] = __float2half_rn((u - 0.5f) * 2.f * scale);
    }
    return v;
}

int main(int argc, char** argv) {
    bool grouped_only = argc > 1 && !strcmp(argv[1], "grouped");
    cudaStream_t s; cudaStreamCreate(&s);
    cublasHandle_t h; cublasCreate(&h); cublasSetStream(h, s);
    const float alpha = 1.f, beta = 0.f;

    auto cublas = [&](const __half* X, const __half* W, float* Y, int64_t ne) {
        // row-major Y[ne,1280] = X[ne,2560] . W[1280,2560]^T
        return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) M1280, (int) ne, (int) K2560, &alpha,
                            W, CUDA_R_16F, (int) K2560, X, CUDA_R_16F, (int) K2560, &beta,
                            Y, CUDA_R_32F, (int) M1280, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    };

    __half *dW, *dX; float *dY, *dRef;
    cudaMalloc(&dW, M1280 * K2560 * 2);
    cudaMalloc(&dY, 256 * M1280 * 4);
    cudaMalloc(&dRef, 256 * M1280 * 4);
    cudaMalloc(&dX, 256 * K2560 * 2);

    unsigned sw = 0x1234567, sx = 0xABCDEF0;
    std::vector<__half> hW = gen_f16(M1280 * K2560, sw, 0.173f);
    cudaMemcpy(dW, hW.data(), hW.size() * 2, cudaMemcpyHostToDevice);

    printf("== single-expert (in-run cuBLAS baseline; maxAbs vs cuBLAS ref) ==\n");
    for (int ne : {8, 16, 32, 64, 128}) {
        sx = 0xABCDEF0;
        std::vector<__half> hX = gen_f16(ne * K2560, sx, 2.1f);
        cudaMemcpy(dX, hX.data(), hX.size() * 2, cudaMemcpyHostToDevice);
        cublas(dX, dW, dRef, ne);

        struct Row { const char* name; double us; float maxabs; };
        std::vector<Row> rows;
        {
            const dim3 g(M1280 / 16, (ne + 127) / 128);
            double u = us_of("S0", [&] {
                if (ne <= 8) S0<false><<<g, 256, 0, s>>>(dW, dX, dY, ne);
                else S0<true ><<<g, 256, 0, s>>>(dW, dX, dY, ne);
            }, s);
            // Stream-ordered D2D: a default-stream D2D would race the next cublas write of dY on s.
            cudaMemcpyAsync(dRef, dY, ne * M1280 * 4, cudaMemcpyDeviceToDevice, s);
            cublas(dX, dW, dY, ne);
            cudaStreamSynchronize(s);
                        float ma = host_maxabs(dRef, dY, ne * M1280, s);
            rows.push_back({"S0  8w m16x128 (1.16)", u, ma});
        }
        {
            const dim3 g(M1280 / 16, (ne + 127) / 128);
            double u = us_of("S1", [&] { S1<<<g, 512, 0, s>>>(dW, dX, dY, ne); }, s);
            float ma = host_maxabs(dRef, dY, ne * M1280, s);
            rows.push_back({"S1 16w m16x128", u, ma});
        }
        {
            const dim3 g(M1280 / 32, (ne + 63) / 64);
            double u = us_of("S2", [&] { S2<<<g, 256, 0, s>>>(dW, dX, dY, ne); }, s);
            float ma = host_maxabs(dRef, dY, ne * M1280, s);
            rows.push_back({"S2  8w m32x128", u, ma});
        }
        {
            const dim3 g(M1280 / 64, (ne + 63) / 64);
            double u = us_of("S3", [&] { S3<<<g, 256, 0, s>>>(dW, dX, dY, ne); }, s);
            float ma = host_maxabs(dRef, dY, ne * M1280, s);
            rows.push_back({"S3  8w m64x64", u, ma});
        }
        {
            const dim3 g(M1280 / 8, (ne + 127) / 128);
            double u = us_of("S6", [&] { S6<<<g, 512, 0, s>>>(dW, dX, dY, ne); }, s);
            float ma = host_maxabs(dRef, dY, ne * M1280, s);
            rows.push_back({"S6 16w m8x128 (S1 split)", u, ma});
        }
        {
            const dim3 g(M1280 / 16, (ne + 15) / 16);
            double u = us_of("S12", [&] { S12<<<g, 128, 0, s>>>(dW, dX, dY, ne); }, s);
            float ma = host_maxabs(dRef, dY, ne * M1280, s);
            rows.push_back({"S12 4w m16x16 ksplit4", u, ma});
        }
        {
            const dim3 g(M1280 / 16, (ne + 15) / 16);
            double u = us_of("S14", [&] { S14<<<g, 256, 0, s>>>(dW, dX, dY, ne); }, s);
            float ma = host_maxabs(dRef, dY, ne * M1280, s);
            rows.push_back({"S14 8w m16x16 ksplit8", u, ma});
        }
        double ucb = us_of("cublas", [&] { cublas(dX, dW, dY, ne); }, s);
        printf("ne=%3d:\n", ne);
        for (auto& r : rows) printf("  %-20s %7.1f us   maxAbs %.2e\n", r.name, r.us, r.maxabs);
        printf("  %-20s %7.1f us   (baseline)\n", "cuBLAS", ucb);
    }

    if (grouped_only) return 0;

    printf("\n== grouped (G experts, one launch) vs G x cuBLAS ==\n");
    // realistic 16K chunk-layer mix (per the 1.16 route-dump ne distribution, scaled to 347
    // experts, sum ne ~ 20.5K): 111 x ne(1..8), 46 x ne(9..16), 49 x ne(17..32), 55 x ne(33..64),
    // 43 x ne(65..128), fat tail: 20 x ~220, 10 x ~360, 8 x ~700, 5 x ~1500.
    std::vector<int> nes;
    auto fill = [&](int count, int lo, int hi, unsigned& seed) {
        for (int i = 0; i < count; ++i) { seed = seed * 1664525u + 1013904223u; nes.push_back(lo + (int) (seed % (hi - lo + 1))); }
    };
    unsigned sg = 0x5EED1234;
    fill(111, 1, 8, sg); fill(46, 9, 16, sg); fill(49, 17, 32, sg); fill(55, 33, 64, sg); fill(43, 65, 128, sg);
    fill(20, 180, 260, sg); fill(10, 320, 400, sg); fill(8, 600, 800, sg); fill(5, 1200, 1800, sg);
    const int G = (int) nes.size();
    int64_t total_rows = 0; for (int n : nes) total_rows += n;
    printf("G=%d experts, total rows = %lld (avg ne %.1f)\n", G, (long long) total_rows, (double) total_rows / G);

    const int64_t xrows = total_rows + G * 128;   // 128 rows headroom per expert (padding n-tiles)
    __half *dXG; float *dYG; cudaMalloc(&dXG, xrows * K2560 * 2); cudaMalloc(&dYG, total_rows * M1280 * 4);
    std::vector<std::vector<__half>> hWg(G);
    std::vector<__half*> dWg_ptrs(G);
    unsigned swg = 0x77AA55;
    int64_t off = 0;
    for (int e = 0; e < G; ++e) {
        hWg[e] = gen_f16(M1280 * K2560, swg, 0.173f);
        cudaMalloc((void**) &dWg_ptrs[e], M1280 * K2560 * 2);
        cudaMemcpy(dWg_ptrs[e], hWg[e].data(), hWg[e].size() * 2, cudaMemcpyHostToDevice);
        off += nes[e];
    }
    unsigned sxg = 0x33BB44;
    std::vector<__half> hXg(xrows * K2560);
    for (int64_t i = 0; i < (int64_t) xrows * K2560; ++i) {
        sxg = sxg * 1664525u + 1013904223u;
        float u = (float) (sxg >> 8) / (float) 0xFFFFFF;
        hXg[i] = __float2half_rn((u - 0.5f) * 2.f * 2.1f);
    }
    cudaMemcpy(dXG, hXg.data(), hXg.size() * 2, cudaMemcpyHostToDevice);

    // reference: G x cuBLAS (the engine's current behavior)
    std::vector<float> ref(total_rows * M1280);
    {
        float* dref; cudaMalloc(&dref, total_rows * M1280 * 4);
        int64_t o = 0;
        for (int e = 0; e < G; ++e) {
            cublas(dXG + o * K2560, dWg_ptrs[e], dref + o * M1280, nes[e]);
            o += nes[e];
        }
        cudaMemcpy(ref.data(), dref, ref.size() * 4, cudaMemcpyDeviceToHost);
        cudaFree(dref);
    }
    double ucb_g = us_of("cublas-grouped", [&] {
        int64_t o = 0;
        for (int e = 0; e < G; ++e) { cublas(dXG + o * K2560, dWg_ptrs[e], dYG + o * M1280, nes[e]); o += nes[e]; }
    }, s);

    // G1 setup: per-expert device array
    std::vector<GExpert> E(G);
    int64_t o = 0, tiles = 0;
    for (int e = 0; e < G; ++e) {
        E[e].X = dXG + o * K2560; E[e].W = dWg_ptrs[e]; E[e].Y = dYG + o * M1280;
        E[e].ne = nes[e]; E[e].tile0 = (int) tiles;
        tiles += 80 * ((nes[e] + 15) / 16);
        o += nes[e];
    }
    GExpert* dE; cudaMalloc(&dE, G * sizeof(GExpert));
    cudaMemcpy(dE, E.data(), G * sizeof(GExpert), cudaMemcpyHostToDevice);
    const dim3 gg((unsigned) ((tiles + 7) / 8));
    double ug1 = us_of("G1", [&] { G1<<<gg, 256, 0, s>>>(dE, (int) G, G); }, s);
    std::vector<float> got(ref.size());
    cudaMemcpy(got.data(), dYG, got.size() * 4, cudaMemcpyDeviceToHost);
    float ma = 0.f;
    for (int64_t i = 0; i < (int64_t) ref.size(); ++i) ma = fmaxf(ma, fabsf(got[i] - ref[i]));
    printf("G1 grouped (all %d experts, 1 launch): %8.1f us   maxAbs %.2e\n", G, ug1, ma);
    printf("G x cuBLAS (engine baseline, %d launches): %8.1f us\n", G, ucb_g);
    // skinny-only comparison: the first 261 experts are ne <= 128; group only ne<=64 (261-43=218? no: 111+46+49+55=261)
    const int GS = 111 + 46 + 49 + 55;
    int64_t tiles_s = 0; for (int e = 0; e < GS; ++e) tiles_s += 80 * ((nes[e] + 15) / 16);
    int64_t rows_s = 0; for (int e = 0; e < GS; ++e) rows_s += nes[e];
    double ucb_s = us_of("cublas-skinny", [&] {
        int64_t o2 = 0;
        for (int e = 0; e < GS; ++e) { cublas(dXG + o2 * K2560, dWg_ptrs[e], dYG + o2 * M1280, nes[e]); o2 += nes[e]; }
    }, s);
    double ug1_s = us_of("G1-skinny", [&] {
        dim3 g2((unsigned) ((tiles_s + 7) / 8));
        G1<<<g2, 256, 0, s>>>(dE, GS, GS);
    }, s);
    printf("G1 grouped (skinny ne<=64, %d experts, 1 launch): %8.1f us  |  G x cuBLAS: %8.1f us  (rows %lld)\n",
           GS, ug1_s, ucb_s, (long long) rows_s);

    // G2 (grouped S12): same E table (tile0/80 = n16 prefix), grid (80, total n16 tiles).
    {
        const int64_t tiles_all = tiles / 80;
        const dim3 g2a(80, (unsigned) tiles_all);
        double ug2a = us_of("G2-all", [&] { G2<<<g2a, 128, 0, s>>>(dE, G); }, s);
        cudaMemcpy(got.data(), dYG, got.size() * 4, cudaMemcpyDeviceToHost);
        float ma2 = 0.f;
        for (int64_t i = 0; i < (int64_t) ref.size(); ++i) ma2 = fmaxf(ma2, fabsf(got[i] - ref[i]));
        printf("G2 grouped (S12 tiles, all %d experts, 1 launch): %8.1f us   maxAbs %.2e\n", G, ug2a, ma2);
    }
    {
        const int64_t tiles_sk = tiles_s / 80;
        const dim3 g2s(80, (unsigned) tiles_sk);
        double ug2s = us_of("G2-skinny", [&] { G2<<<g2s, 128, 0, s>>>(dE, GS); }, s);
        std::vector<float> gotk(rows_s * M1280);
        cudaMemcpy(gotk.data(), dYG, gotk.size() * 4, cudaMemcpyDeviceToHost);
        float ma2k = 0.f;
        for (int64_t i = 0; i < gotk.size(); ++i) ma2k = fmaxf(ma2k, fabsf(gotk[i] - ref[i]));
        printf("G2 grouped (S12 tiles, skinny %d experts, 1 launch): %8.1f us  |  G x cuBLAS: %8.1f us   maxAbs %.2e\n",
               GS, ug2s, ucb_s, ma2k);
    }
    return 0;
}
