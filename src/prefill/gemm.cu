// src/prefill/gemm.cu - see include/strata/prefill/gemm.hpp.
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
// The HIP compatibility shim maps CUDA shuffle spellings to Strata helpers.
// hipBLASLt's public headers declare native HIP shuffle functions, so keep
// those declarations from being macro-expanded in this translation unit.
#undef __shfl_xor_sync
#undef __shfl_down_sync
#undef __shfl_up_sync
#undef __shfl_sync
#undef __ballot_sync
#endif

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>
#include <cstring>
#include <memory>

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
#include "hipblaslt_tuning.hpp"
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <map>
#include <set>
#include <tuple>
#endif

namespace strata::prefill {
namespace {
// Stage 1.15: route the f16 MoE expert GEMMs (the skinny Y[T,N]=X[T,K].W[N,K]^T products) through
// cublasLtMatmul with the first heuristic algorithm instead of cublasGemmEx(CUBLAS_GEMM_DEFAULT).
// Microbench (bench/v100, V100 sm_70): GU (N=1280,K=2560) is 2-9% faster and D (N=2560,K=640) is
// 4-13% faster across ne=1..512, with no per-ne algo search needed. OFF by default (STRATA_MOE_GEMM_LT).
// Per-shape Lt objects (op/layout/heuristic) are cached so the hot loop only issues the GEMM itself.
struct LtPlan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    cublasLtMatmulHeuristicResult_t hr;
    bool valid = false;
};
struct LtCache {
    std::unordered_map<uint64_t, LtPlan> plans;
    size_t ws_bytes = 0;
    bool pref_ok = false;
    cublasLtMatmulPreference_t pref = nullptr;
    LtPlan& get(cublasLtHandle_t lt, int64_t T, int64_t N, int64_t K, int64_t ldy, size_t ws_bytes) {
        const uint64_t key = ((uint64_t) N << 40) | ((uint64_t) K << 24) | (uint64_t) (T & 0xFFFFFF);
        auto it = plans.find(key);
        if (it != plans.end()) return it->second;
        LtPlan p;
        cublasStatus_t st;
        if ((st = cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F)) == CUBLAS_STATUS_SUCCESS) {
            cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
            cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
            cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
        }
        // Column-major views (row-major tensors read transposed): A=W stored K x N (ld=K), B=X stored K x T (ld=K),
        // C=Y stored N x T (ld=ldy); C[N,T] = A^T[N,K] . B[K,T].
        st = cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16F, K, N, K);
        if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16F, K, T, K);
        if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&p.lc, CUDA_R_32F, N, T, ldy);
        if (st == CUBLAS_STATUS_SUCCESS) {
            if (ws_bytes != this->ws_bytes || !pref_ok) {
                if (pref) cublasLtMatmulPreferenceDestroy(pref);
                pref = nullptr;
                pref_ok = (cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS);
                if (pref_ok) {
                    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes,
                                                         sizeof(ws_bytes));
                    this->ws_bytes = ws_bytes;
                }
            }
            int nres = 0;
            st = cublasLtMatmulAlgoGetHeuristic(lt, p.op, p.la, p.lb, p.lc, p.lc, pref, 1, &p.hr, &nres);
            p.valid = (st == CUBLAS_STATUS_SUCCESS && nres > 0);
        }
        plans.emplace(key, p);
        return plans[key];
    }
};
LtCache lt_cache;

bool lt_gemm_f16(cublasLtHandle_t lt, const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta, void* workspace, size_t ws_bytes, cudaStream_t stream, bool* ok) {
    const float alpha = 1.0f;
    LtPlan& p = lt_cache.get(lt, T, N, K, ldy, ws_bytes);
    if (!p.valid) { *ok = false; return false; }
    cublasStatus_t st = cublasLtMatmul(lt, p.op, &alpha, W, p.la, X, p.lb, &beta, Y, p.lc, Y, p.lc, &p.hr.algo,
                                       workspace, ws_bytes, stream);
    *ok = (st == CUBLAS_STATUS_SUCCESS);
    return *ok;
}

void ck(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: %s: cuBLAS status %d\n", what, (int) s);
        std::exit(1);
    }
}

// Stage 1.16 (opt-in, STRATA_MOE_GEMM_TC): V100 tensor-core path for the skinny GU expert GEMM
//   Y[T,1280] (f32, ldy 1280) = X[T,2560] (f16) . W[1280,2560]^T (f16),  T <= 128.
// Microbench (bench/v100/s116_gemm_tc.cu, V100 sm_70): faster than cuBLAS at T <= 16 (0.74-0.89x
// cuBLAS time; up to 67% of the 474 GB/s W-streaming roofline at T <= 8 via the SKIP variant, which
// drops the second n8 sub-tile's X over-reads), on par at T = 32, slower at T >= 64 (the X tile is
// re-read once per m16 block). f32 accumulate: matches cuBLAS to ~1e-5 abs (not bit-exact - the f32
// add order differs). Only the GU shape (N=1280, K=2560) is routed, for T <= tc_max_ne_ (default 64,
// max 128); D GEMM, dense GEMM and decode are untouched, and the cuBLAS/cuBLASLt paths remain the
// fallback.
//
// sm_70 mma.sync.m8n8k4.f16 is ONE k4 dot product (its 4 lane groups replicate the same k4 tile -
// verified empirically: a single call sums only the k4 slice its lanes loaded). A/B fragments are
// 4 k-contiguous f16 per lane; C/D is 8 f32 per lane. A k8 window = 4 sequential calls (16B loads,
// 8 f16 per lane per operand), all lanes loading the same window.
// Block = 8 warps over an m16 x n128 tile; the warps split as n-warps (n16 tiles, the smallest power of
// two covering the block's valid rows) x k-warps (K split). Small T -> the K split keeps all 8 warps
// active (enough in-flight bytes to stream W near roofline), large T -> n-warps. Each k-warp accumulates
// its k-slice in ascending k8 order, writes its m16 x n16 partial to smem; a fixed-order reduce over the
// k-warps makes the path deterministic. X rows beyond T are only READ by the padding n16 tiles (their
// outputs are masked out), so the caller must keep 128 rows of headroom in X (prefill pads Xs).
namespace {
constexpr int TC_M = 1280, TC_K = 2560, TC_QCACHE = 4;  // queue depth: power of two; 4 = Q stays in registers (8 spills to local mem -> 6x slower)

__device__ __forceinline__ void tc_mma816(float* d, unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
                   "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// FULL=false: ne <= 8 - only the first n8 sub-tile of the block's single n16 tile is valid;
// skip the second one entirely (no b1 loads, no c0[1] mma) - halves the X over-read traffic.
template<bool FULL>
__global__ void __launch_bounds__(256)
tc_gu_kernel(const uint16_t* __restrict__ W, const uint16_t* __restrict__ X, float* __restrict__ Y, int ne) {
    const int l = threadIdx.x & 31;
    const int w = threadIdx.x >> 5;
    const int m0 = blockIdx.x * 16;
    const int nbase = blockIdx.y * 128;
    // Per-block decomposition: cover the block's valid rows with n16 tiles (a power-of-two number of
    // n-warps <= 8); the remaining warps split K. All 8 warps stay active for any ne.
    const int nv = min(128, ne - nbase);
    const int tiles = (nv + 15) / 16;  // 1..8 n16 tiles needed
    const int nwarp = tiles <= 1 ? 1 : (tiles <= 2 ? 2 : (tiles <= 4 ? 4 : 8));
    const int KW = 8 / nwarp;
    const int i = w % nwarp;  // n-tile index
    const int j = w / nwarp;  // k-slice index
    const bool active = i < tiles;
    const int n0 = nbase + i * 16;
    const int SUB = TC_K / KW;  // k values per warp (multiple of 16)
    const int64_t kbase = (int64_t) j * SUB;
    const int row = (l & 3) + ((l >> 4) & 1) * 4;  // A row / B col, 0..7
    const int64_t wa0 = (int64_t)(m0 + row) * TC_K + kbase;
    const int64_t wa1 = (int64_t)(m0 + row + 8) * TC_K + kbase;
    const int64_t xb0 = (int64_t)(n0 + row) * TC_K + kbase;
    const int64_t xb1 = (int64_t)(n0 + row + 8) * TC_K + kbase;

    float c0[2][2][8];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            #pragma unroll
            for (int ii = 0; ii < 8; ++ii) c0[mt][nt][ii] = 0.f;

    if (active) {
        // One mma.sync.m8n8k4.f16 = ONE k4 dot product: all 32 lanes load the SAME k4 slice
        // (A frag = W row x k4, B frag = X row x k4; the 4 lane groups replicate the tile).
        // A k8 window = 4 sequential mma calls (2 per operand half), lanes loading 16B windows
        // (8 f16) straight from L2/DRAM - no smem, no swizzle. Queue holds whole k8 windows.
        struct Q { uint4 a0, a1, b0, b1; };
        Q q[TC_QCACHE];
        #pragma unroll
        for (int d = 0; d < TC_QCACHE; ++d) {
            const int off = d * 8;
            q[d].a0 = *(const uint4*)&W[wa0 + off];
            q[d].a1 = *(const uint4*)&W[wa1 + off];
            q[d].b0 = *(const uint4*)&X[xb0 + off];
            if (FULL) q[d].b1 = *(const uint4*)&X[xb1 + off];
        }
        const int NWIN = SUB / 8;
        for (int t = 0; t < NWIN; ++t) {
            const int slot = t & (TC_QCACHE - 1);
            const int koff = t * 8 + TC_QCACHE * 8;
            tc_mma816(c0[0][0], q[slot].a0.x, q[slot].a0.y, q[slot].b0.x, q[slot].b0.y);
            tc_mma816(c0[1][0], q[slot].a1.x, q[slot].a1.y, q[slot].b0.x, q[slot].b0.y);
            tc_mma816(c0[0][0], q[slot].a0.z, q[slot].a0.w, q[slot].b0.z, q[slot].b0.w);
            tc_mma816(c0[1][0], q[slot].a1.z, q[slot].a1.w, q[slot].b0.z, q[slot].b0.w);
            if (FULL) {
                tc_mma816(c0[0][1], q[slot].a0.x, q[slot].a0.y, q[slot].b1.x, q[slot].b1.y);
                tc_mma816(c0[1][1], q[slot].a1.x, q[slot].a1.y, q[slot].b1.x, q[slot].b1.y);
                tc_mma816(c0[0][1], q[slot].a0.z, q[slot].a0.w, q[slot].b1.z, q[slot].b1.w);
                tc_mma816(c0[1][1], q[slot].a1.z, q[slot].a1.w, q[slot].b1.z, q[slot].b1.w);
            }
            if (koff < SUB) {  // consume first, then refill the same slot for t+D
                q[slot].a0 = *(const uint4*)&W[wa0 + koff];
                q[slot].a1 = *(const uint4*)&W[wa1 + koff];
                q[slot].b0 = *(const uint4*)&X[xb0 + koff];
                if (FULL) q[slot].b1 = *(const uint4*)&X[xb1 + koff];
            }
        }
    }
    // C/D fragment layout (m8n8k4): ci: row = (l&1) + 2*((i>>1)&1) + 4*((l>>4)&1);
    // col = 4*((i>>2)&1) + 2*((l>>1)&1) + (i&1).
    const int rb = (l & 1) + 4 * ((l >> 4) & 1);
    const int cb = 2 * ((l >> 1) & 1);
    __shared__ float part[8][16][16];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            #pragma unroll
            for (int ii = 0; ii < 8; ++ii) {
                const int r = rb + 2 * ((ii >> 1) & 1);
                const int c = cb + 4 * ((ii >> 2) & 1) + (ii & 1);
                part[w][mt * 8 + r][nt * 8 + c] = c0[mt][nt][ii];
            }
    __syncthreads();
    // Fixed-order reduce over the k-warps, then store (rows < ne only).
    for (int idx = threadIdx.x; idx < 2048; idx += 256) {
        const int n128 = idx >> 4, m16 = idx & 15;
        const int gn = nbase + n128;
        if (gn < ne) {
            const int ti = n128 >> 4, c16 = n128 & 15;
            float s = 0.f;
            #pragma unroll
            for (int k = 0; k < KW; ++k) s += part[ti + k * nwarp][m16][c16];
            Y[(int64_t) gn * TC_M + (m0 + m16)] = s;
        }
    }
}

bool tc_gemm_gu(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, cudaStream_t stream) {
    // T <= 128 is guaranteed by the caller (the routing guard in f16()).
    const dim3 g(TC_M / 16, (int) ((T + 127) / 128));
    // T <= 8: only the first n8 sub-tile is valid -> use the SKIP variant (no second n8).
    if (T <= 8) tc_gu_kernel<false><<<g, 256, 0, stream>>>(W, X, Y, (int) T);
    else        tc_gu_kernel<true ><<<g, 256, 0, stream>>>(W, X, Y, (int) T);
    return cudaGetLastError() == cudaSuccess;
}

// Stage 1.17 (STRATA_MOE_GEMM_GROUPED): the G2 grouped skinny-GU kernel.  ONE launch over all the
// skinny experts of a chunk-layer: grid (80, total n16 tiles), block = 128 threads (4 warps).  Each
// block owns one (expert, m16, n16) tile; the expert is found by a binary search over the table's
// tile0 prefix (tile0 is in units of 80 m-blocks).  The tile body is the Stage 1.17 microbench S12
// (the m16 x n16 tile beats the m16 x n128 S0 shape at ne <= 16 and is competitive at 17-64; W is
// read once per GEMM).  The 4 warps split K (SUB = K/4 each); every k-warp accumulates its slice in
// ascending k8 order and writes its partial to smem; the fixed-order reduce over the k-warps makes
// the result deterministic.  X rows past ne are only READ by the last partial n16 tile (the outputs
// are masked), so the caller keeps the 128-row Xs headroom.  ne <= 128 is guaranteed by the caller
// (the tc_max_ne routing guard).
__global__ void __launch_bounds__(128)
g2_gu_kernel(const GroupedExpert* __restrict__ E, int G) {
    const int l = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int t = blockIdx.y;
    int lo = 0, hi = G;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (E[mid].tile0 / 80 <= t) lo = mid + 1; else hi = mid;
    }
    const GroupedExpert Ee = E[lo - 1];
    const int tel = t - Ee.tile0 / 80;
    const int m0 = blockIdx.x * 16, n0 = tel * 16;
    const int ne = Ee.ne;
    const bool active = n0 < ne;
    const int SUB = TC_K / 4, kbase = w * SUB;
    const int row = (l & 3) + ((l >> 4) & 1) * 4;  // A row / B col, 0..7
    const int64_t wa0 = (int64_t)(m0 + row) * TC_K + kbase, wa1 = wa0 + 8 * TC_K;
    const int64_t xb0 = (int64_t)(n0 + row) * TC_K + kbase, xb1 = xb0 + 8 * TC_K;
    const uint16_t* W = Ee.W;
    const uint16_t* X = Ee.X;
    float c0[2][2][8];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            #pragma unroll
            for (int ii = 0; ii < 8; ++ii) c0[mt][nt][ii] = 0.f;
    if (active) {
        struct Q { uint4 a0, a1, b0, b1; };
        Q q[TC_QCACHE];
        #pragma unroll
        for (int d = 0; d < TC_QCACHE; ++d) {
            const int off = d * 8;
            q[d].a0 = *(const uint4*)&W[wa0 + off];
            q[d].a1 = *(const uint4*)&W[wa1 + off];
            q[d].b0 = *(const uint4*)&X[xb0 + off];
            q[d].b1 = *(const uint4*)&X[xb1 + off];
        }
        const int NWIN = SUB / 8;
        for (int t2 = 0; t2 < NWIN; ++t2) {
            const int slot = t2 & (TC_QCACHE - 1);
            const int koff = t2 * 8 + TC_QCACHE * 8;
            tc_mma816(c0[0][0], q[slot].a0.x, q[slot].a0.y, q[slot].b0.x, q[slot].b0.y);
            tc_mma816(c0[1][0], q[slot].a1.x, q[slot].a1.y, q[slot].b0.x, q[slot].b0.y);
            tc_mma816(c0[0][0], q[slot].a0.z, q[slot].a0.w, q[slot].b0.z, q[slot].b0.w);
            tc_mma816(c0[1][0], q[slot].a1.z, q[slot].a1.w, q[slot].b0.z, q[slot].b0.w);
            tc_mma816(c0[0][1], q[slot].a0.x, q[slot].a0.y, q[slot].b1.x, q[slot].b1.y);
            tc_mma816(c0[1][1], q[slot].a1.x, q[slot].a1.y, q[slot].b1.x, q[slot].b1.y);
            tc_mma816(c0[0][1], q[slot].a0.z, q[slot].a0.w, q[slot].b1.z, q[slot].b1.w);
            tc_mma816(c0[1][1], q[slot].a1.z, q[slot].a1.w, q[slot].b1.z, q[slot].b1.w);
            if (koff < SUB) {
                q[slot].a0 = *(const uint4*)&W[wa0 + koff];
                q[slot].a1 = *(const uint4*)&W[wa1 + koff];
                q[slot].b0 = *(const uint4*)&X[xb0 + koff];
                q[slot].b1 = *(const uint4*)&X[xb1 + koff];
            }
        }
    }
    const int rb = (l & 1) + 4 * ((l >> 4) & 1);
    const int cb = 2 * ((l >> 1) & 1);
    __shared__ float part[4][16][16];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            #pragma unroll
            for (int ii = 0; ii < 8; ++ii) {
                const int r = rb + 2 * ((ii >> 1) & 1);
                const int c = cb + 4 * ((ii >> 2) & 1) + (ii & 1);
                part[w][mt * 8 + r][nt * 8 + c] = c0[mt][nt][ii];
            }
    __syncthreads();
    for (int idx = threadIdx.x; idx < 256; idx += 128) {
        const int n16 = idx >> 4, m16 = idx & 15;
        const int gn = n0 + n16;
        if (gn < ne) {
            float s = 0.f;
            #pragma unroll
            for (int k = 0; k < 4; ++k) s += part[k][m16][n16];
            Ee.Y[(int64_t) gn * TC_M + (m0 + m16)] = s;
        }
    }
}

// Stage 1.16 debug (STRATA_MOE_GEMM_TC_VERIFY): count nneq and max |a-b| between two f32 arrays.
__global__ void tc_verify_cmp(const float* Y, const float* ref, int n, float* nneq_f, float* maxabs_f) {
    unsigned cnt = 0;
    unsigned mx = 0u;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float a = Y[i], b = ref[i];
        if (a != b) {
            ++cnt;
            const unsigned dbits = __float_as_uint(fabsf(a - b));
            if (dbits > mx) mx = dbits;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        cnt += __shfl_down_sync(0xffffffffu, cnt, o);
        mx = max(mx, __shfl_down_sync(0xffffffffu, mx, o));
    }
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(nneq_f, (float) cnt);
        atomicMax((unsigned*) (void*) maxabs_f, mx);
    }
}
}  // namespace
// #247/#325: on Windows (seen on gfx1201), hipBLAS can return success with the correct BF16/FP16 product for some
// shapes (hc up once T >= 96, the router) and still leave hipErrorInvalidValue set, which the next kernel's error
// check turns into an exit. The multiply has finished, so that one stale error is cleared after a GEMM that succeeded;
// any other error still stops the engine. Windows only: on Linux a stale hipErrorInvalidValue is a real error from
// an earlier call and keeps being reported. A no-op everywhere else (CUDA compiles none of it).
#if defined(__HIPCC__) && defined(_WIN32)
void absorb_hipblas_sticky(const char* what) {
    const hipError_t sticky = hipGetLastError();
    if (sticky == hipSuccess || sticky == hipErrorInvalidValue) return;
    std::fprintf(stderr, "prefill gemm: %s left %s\n", what, hipGetErrorString(sticky));
    std::exit(1);
}
#define STRATA_ABSORB_HIPBLAS_STICKY(what) absorb_hipblas_sticky(what)
#else
#define STRATA_ABSORB_HIPBLAS_STICKY(what) ((void) 0)
#endif

// A setup call whose failure the engine survives (the handle keeps its defaults), as before #240 - but said.
void note(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) std::fprintf(stderr, "prefill gemm: %s: cuBLAS status %d (continuing)\n", what, (int) s);
}

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
struct HipLtCallKey {
    strata::prefill::hipblaslt::InputType type;
    int t;
    int n;
    int k;
    int ldy;
    uint32_t beta_bits;

    bool operator<(const HipLtCallKey& other) const {
        return std::tie(type, n, k, ldy, t, beta_bits) <
               std::tie(other.type, other.n, other.k, other.ldy, other.t, other.beta_bits);
    }
};

struct HipLtCachedAlgo {
    bool supported = false;
    hipblasLtMatmulAlgo_t algo{};
    size_t workspace_bytes = 0;
};

struct HipLtState {
    hipblasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    strata::prefill::hipblaslt::TuningTable table;
    std::map<HipLtCallKey, HipLtCachedAlgo> cache;
    uint64_t lt_launches = 0;
    uint64_t fallbacks = 0;
    std::set<std::tuple<strata::prefill::hipblaslt::InputType, int, int, int, int>> fallback_shapes;

    ~HipLtState() {
        if (std::getenv("STRATA_HIPBLASLT_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: hipBLASLt summary launches=%llu fallbacks=%llu unique_fallback_shapes=%zu\n",
                         (unsigned long long) lt_launches, (unsigned long long) fallbacks, fallback_shapes.size());
            for (const auto& shape : fallback_shapes) {
                const auto type = std::get<0>(shape);
                std::fprintf(stderr, "prefill gemm: fallback shape dtype=%s T=%d N=%d K=%d ldy=%d\n",
                             type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16",
                             std::get<1>(shape), std::get<2>(shape), std::get<3>(shape), std::get<4>(shape));
            }
        }
        if (handle) hipblasLtDestroy(handle);
    }
};

struct HipLtDescriptors {
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a = nullptr;
    hipblasLtMatrixLayout_t b = nullptr;
    hipblasLtMatrixLayout_t c = nullptr;

    ~HipLtDescriptors() {
        if (op) hipblasLtMatmulDescDestroy(op);
        if (a) hipblasLtMatrixLayoutDestroy(a);
        if (b) hipblasLtMatrixLayoutDestroy(b);
        if (c) hipblasLtMatrixLayoutDestroy(c);
    }

    bool init(hipDataType type, int t, int n, int k, int ldy) {
        const hipblasOperation_t trans_a = HIPBLAS_OP_T;
        const hipblasOperation_t trans_b = HIPBLAS_OP_N;
        if (hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)) !=
                HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)) !=
                HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&a, type, k, n, k) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&b, type, k, t, k) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&c, HIP_R_32F, n, t, ldy) != HIPBLAS_STATUS_SUCCESS) {
            return false;
        }
        return true;
    }
};

std::unique_ptr<HipLtState> create_hipblaslt_state(void* workspace, size_t workspace_bytes) {
    const char* path = std::getenv("STRATA_HIPBLASLT_TUNING");
    if (!path || !*path) return nullptr;

    auto state = std::make_unique<HipLtState>();
    state->workspace = workspace;
    state->workspace_bytes = workspace_bytes;
    if (hipblasLtCreate(&state->handle) != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: hipBLASLt handle creation failed; using hipBLASEx\n");
        return nullptr;
    }

    int version = 0;
    if (hipblasLtGetVersion(state->handle, &version) != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: hipBLASLt version query failed; using hipBLASEx\n");
        return nullptr;
    }
    int device = 0;
    hipDeviceProp_t properties{};
    if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&properties, device) != hipSuccess) {
        std::fprintf(stderr, "prefill gemm: HIP device query failed; using hipBLASEx\n");
        return nullptr;
    }
    std::string arch(properties.gcnArchName);
    const auto suffix = arch.find(':');
    if (suffix != std::string::npos) arch.resize(suffix);

    std::string error;
    if (!state->table.load(path, arch, version, error)) {
        std::fprintf(stderr, "prefill gemm: %s; using hipBLASEx\n", error.c_str());
        return nullptr;
    }
    std::fprintf(stderr, "prefill gemm: hipBLASLt tuning enabled (%zu rows, %s, version %d)\n",
                 state->table.rows().size(), arch.c_str(), version);
    return state;
}

HipLtCachedAlgo resolve_hipblaslt_algo(HipLtState& state, strata::prefill::hipblaslt::InputType type, int t,
                                       int n, int k, int ldy, float beta) {
    uint32_t beta_bits = 0;
    static_assert(sizeof(beta_bits) == sizeof(beta));
    std::memcpy(&beta_bits, &beta, sizeof(beta));
    const HipLtCallKey key{type, t, n, k, ldy, beta_bits};
    const auto cached = state.cache.find(key);
    if (cached != state.cache.end()) return cached->second;

    HipLtCachedAlgo resolved;
    const bool verbose = std::getenv("STRATA_HIPBLASLT_VERBOSE") != nullptr;
    const auto* row = state.table.closest(type, n, k, ldy, t);
    if (!row) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; no calibration for dtype=%s T=%d N=%d K=%d ldy=%d\n",
                         type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16", t, n, k, ldy);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    HipLtDescriptors desc;
    const hipDataType input_type = type == strata::prefill::hipblaslt::InputType::bf16 ? HIP_R_16BF : HIP_R_16F;
    if (!desc.init(input_type, t, n, k, ldy)) {
        return state.cache.emplace(key, resolved).first->second;
    }

    std::vector<int> solution_ids{row->solution_id};
    std::vector<hipblasLtMatmulHeuristicResult_t> candidates;
    if (hipblaslt_ext::getAlgosFromIndex(state.handle, solution_ids, candidates) != HIPBLAS_STATUS_SUCCESS ||
        candidates.empty() || candidates.front().state != HIPBLAS_STATUS_SUCCESS ||
        hipblaslt_ext::getIndexFromAlgo(candidates.front().algo) != row->solution_id) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution %d unavailable for T=%d N=%d K=%d ldy=%d\n",
                         row->solution_id, t, n, k, ldy);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    const float alpha = 1.0f;
    size_t required_workspace = 0;
    auto algo = candidates.front().algo;
    if (hipblaslt_ext::matmulIsAlgoSupported(state.handle, desc.op, &alpha, desc.a, desc.b, &beta, desc.c, desc.c,
                                             algo, required_workspace) != HIPBLAS_STATUS_SUCCESS) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution %d rejects actual T=%d N=%d K=%d ldy=%d beta=%.9g\n",
                         row->solution_id, t, n, k, ldy, beta);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    resolved.supported = true;
    resolved.algo = algo;
    resolved.workspace_bytes = required_workspace;
    if (verbose) {
        std::fprintf(stderr,
                     "prefill gemm: Lt solution=%d dtype=%s T=%d N=%d K=%d ldy=%d beta=%.9g workspace=%zu\n",
                     row->solution_id, type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16", t, n,
                     k, ldy, beta, required_workspace);
    }
    return state.cache.emplace(key, resolved).first->second;
}

bool try_hipblaslt(void* opaque_state, strata::prefill::hipblaslt::InputType type, const uint16_t* x,
                   const uint16_t* w, float* y, int64_t t, int64_t n, int64_t k, int64_t ldy, float beta,
                   void* stream) {
    auto* state = static_cast<HipLtState*>(opaque_state);
    if (!state || t <= 0 || n <= 0 || k <= 0 || t > INT_MAX || n > INT_MAX || k > INT_MAX || ldy > INT_MAX ||
        ldy < n) {
        return false;
    }
    const auto resolved = resolve_hipblaslt_algo(*state, type, (int) t, (int) n, (int) k, (int) ldy, beta);
    if (!resolved.supported) {
        ++state->fallbacks;
        state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
        return false;
    }
    if (resolved.workspace_bytes > state->workspace_bytes) {
        ++state->fallbacks;
        state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
        if (std::getenv("STRATA_HIPBLASLT_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution needs %zu workspace bytes, have %zu\n",
                         resolved.workspace_bytes, state->workspace_bytes);
        }
        return false;
    }

    HipLtDescriptors desc;
    const hipDataType input_type = type == strata::prefill::hipblaslt::InputType::bf16 ? HIP_R_16BF : HIP_R_16F;
    if (!desc.init(input_type, (int) t, (int) n, (int) k, (int) ldy)) return false;
    const float alpha = 1.0f;
    const hipblasStatus_t status = hipblasLtMatmul(state->handle, desc.op, &alpha, w, desc.a, x, desc.b, &beta, y,
                                                   desc.c, y, desc.c, &resolved.algo, state->workspace,
                                                   state->workspace_bytes, (hipStream_t) stream);
    if (status == HIPBLAS_STATUS_SUCCESS) {
        ++state->lt_launches;
        return true;
    }

    std::fprintf(stderr, "prefill gemm: hipBLASLt launch failed with status %d\n", (int) status);
    if (beta != 0.0f) {
        std::fprintf(stderr, "prefill gemm: refusing a fallback after hipBLASLt failed with nonzero beta\n");
        std::exit(1);
    }
    auto* mutable_state = static_cast<HipLtState*>(opaque_state);
    uint32_t beta_bits = 0;
    std::memcpy(&beta_bits, &beta, sizeof(beta_bits));
    auto cached = mutable_state->cache.find(HipLtCallKey{type, (int) t, (int) n, (int) k, (int) ldy, beta_bits});
    if (cached != mutable_state->cache.end()) cached->second.supported = false;
    ++mutable_state->fallbacks;
    mutable_state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
    return false;
}
#endif

}  // namespace

Gemm::~Gemm() {
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    delete static_cast<HipLtState*>(hipblaslt_state_);
#endif
    if (lt_handle_) cublasLtDestroy((cublasLtHandle_t) lt_handle_);
    if (handle_) cublasDestroy((cublasHandle_t) handle_);
    if (verify_dev_) cudaFree(verify_dev_);
    if (!external_) {
        if (scratch_) cudaFree(scratch_);
        if (workspace_) cudaFree(workspace_);
    }
}



bool Gemm::init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                         std::string& err) {
    cublasHandle_t h = nullptr;
    if (const cublasStatus_t s = cublasCreate(&h); s != CUBLAS_STATUS_SUCCESS) {
        err = "prefill gemm: cublasCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    handle_ = h;
    stream_ = stream;
    external_ = true;
    note(cublasSetStream(h, (cudaStream_t) stream), "cublasSetStream");
    workspace_ = workspace;
    ws_bytes_ = ws_bytes;
    note(cublasSetWorkspace(h, workspace_, ws_bytes), "cublasSetWorkspace");
    note(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    if (std::getenv("STRATA_MOE_GEMM_LT") && cublasLtCreate((cublasLtHandle_t*) &lt_handle_) == CUBLAS_STATUS_SUCCESS)
        use_lt_ = true;
    if (const char* e = std::getenv("STRATA_MOE_GEMM_TC"); e && std::string(e) == "1") {
        use_tc_ = true;
        if (const char* m = std::getenv("STRATA_MOE_GEMM_TC_MAXNE")) {
            const long v = std::strtol(m, nullptr, 10);
            if (v >= 1 && v <= 128) tc_max_ne_ = v;
        }
    }
    if (const char* v = std::getenv("STRATA_MOE_GEMM_TC_VERIFY"); v && std::string(v) == "1") {
        tc_verify_ = true;
        if (cudaMalloc((void**) &verify_dev_, 2 * sizeof(float)) != cudaSuccess) tc_verify_ = false;
    }
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    hipblaslt_state_ = create_hipblaslt_state(workspace_, ws_bytes).release();
#endif
    return true;
}

void Gemm::rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes) {
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    workspace_ = workspace;
    cublasSetWorkspace((cublasHandle_t) handle_, workspace_, ws_bytes);
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (hipblaslt_state_) {
        auto* state = static_cast<HipLtState*>(hipblaslt_state_);
        state->workspace = workspace_;
        state->workspace_bytes = ws_bytes;
    }
#endif
}

bool Gemm::init(void* stream, int64_t scratch_elems, std::string& err) {
    // #240: every failure names the call and the real status, so "no VRAM" can be told from a broken install
    cublasHandle_t h = nullptr;
    if (const cublasStatus_t s = cublasCreate(&h); s != CUBLAS_STATUS_SUCCESS) {
        err = "prefill gemm: cublasCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    handle_ = h;
    stream_ = stream;
    note(cublasSetStream(h, (cudaStream_t) stream), "cublasSetStream");
    // A fixed workspace so the handle never allocates on the way (and graphs could capture it later).
    const size_t ws = 32u << 20;
    if (const cudaError_t e = cudaMalloc(&workspace_, ws); e != cudaSuccess) {
        err = std::string("prefill gemm: workspace of 32 MiB: ") + cudaGetErrorString(e);
        return false;
    }
    ws_bytes_ = ws;
    if (std::getenv("STRATA_MOE_GEMM_LT") && cublasLtCreate((cublasLtHandle_t*) &lt_handle_) == CUBLAS_STATUS_SUCCESS)
        use_lt_ = true;
    if (const char* e = std::getenv("STRATA_MOE_GEMM_TC"); e && std::string(e) == "1") {
        use_tc_ = true;
        if (const char* m = std::getenv("STRATA_MOE_GEMM_TC_MAXNE")) {
            const long v = std::strtol(m, nullptr, 10);
            if (v >= 1 && v <= 128) tc_max_ne_ = v;
        }
    }
    if (const char* v = std::getenv("STRATA_MOE_GEMM_TC_VERIFY"); v && std::string(v) == "1") {
        tc_verify_ = true;
        if (cudaMalloc((void**) &verify_dev_, 2 * sizeof(float)) != cudaSuccess) tc_verify_ = false;
    }
    note(cublasSetWorkspace(h, workspace_, ws), "cublasSetWorkspace");
    note(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    hipblaslt_state_ = create_hipblaslt_state(workspace_, ws).release();
#endif
    if (scratch_elems > 0) {
        if (const cudaError_t e = cudaMalloc((void**) &scratch_, (size_t) scratch_elems * 2); e != cudaSuccess) {
            err = "prefill gemm: dequant scratch of " + std::to_string(scratch_elems * 2 >> 20) + " MiB: " +
                  cudaGetErrorString(e);
            return false;
        }
    }
    scratch_elems_ = scratch_elems;
    return true;
}

void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    const float alpha = 1.0f;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (try_hipblaslt(hipblaslt_state_, strata::prefill::hipblaslt::InputType::bf16, X, W, Y, T, N, K, ldy,
                      beta, stream_)) {
        STRATA_ABSORB_HIPBLAS_STICKY("hipBLASLt bf16");
        return;
    }
#endif
    // Column-major view: Y^T[N, T] = W[N, K] (stored K x N col-major, transposed) . X^T[K, T].
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16BF, (int) K, X, CUDA_R_16BF, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx");
    STRATA_ABSORB_HIPBLAS_STICKY("cublasGemmEx");
}

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    const float alpha = 1.0f;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (try_hipblaslt(hipblaslt_state_, strata::prefill::hipblaslt::InputType::f16, X, W, Y, T, N, K, ldy,
                      beta, stream_)) {
        STRATA_ABSORB_HIPBLAS_STICKY("hipBLASLt f16");
        return;
    }
#endif
    // Stage 1.16: the skinny GU shape only (the D shape N=2560/K=640, dense GEMMs and any custom ldy
    // keep their existing paths). T <= 128 keeps the X over-read within the padded Xs.
    if (use_tc_ && T <= tc_max_ne_ && T <= 128 && beta == 0.0f && N == 1280 && K == 2560 && ldy == 1280 &&
        tc_gemm_gu(X, W, Y, T, (cudaStream_t) stream_)) {
        if (tc_verify_) {
            // Debug: compute the cuBLAS reference into the dequant scratch (free during f16 calls) and
            // compare against the TC result on-device; log every diverging call plus the first 20.
            float* ref = (float*) scratch_;
            ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                            CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &beta, ref, CUDA_R_32F, (int) ldy,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
               "cublasGemmEx tc_verify");
            cudaMemsetAsync(verify_dev_, 0, 2 * sizeof(float), (cudaStream_t) stream_);
            tc_verify_cmp<<<8, 256, 0, (cudaStream_t) stream_>>>(Y, ref, (int) (T * ldy), verify_dev_, verify_dev_ + 1);
            float hv[2] = {0.0f, 0.0f};
            cudaMemcpyAsync(hv, verify_dev_, 2 * sizeof(float), cudaMemcpyDeviceToHost, (cudaStream_t) stream_);
            cudaStreamSynchronize((cudaStream_t) stream_);
            static long calls = 0;
            ++calls;
            if (hv[0] != 0.0f || calls <= 20) {
                std::fprintf(stderr,
                             "[tc_verify] call %ld T=%lld nneq=%.0f maxAbs=%.3e\n", calls, (long long) T, hv[0], hv[1]);
            }
            // STRATA_MOE_GEMM_TC_DUMP=<path>: dump the first diverging call's X, W, Y(TC), ref for offline check.
            static bool dumped = false;
            if (!dumped && hv[0] != 0.0f && getenv("STRATA_MOE_GEMM_TC_DUMP")) {
                dumped = true;
                const char* p = getenv("STRATA_MOE_GEMM_TC_DUMP");
                std::vector<uint16_t> xd((size_t) T * K), wd((size_t) N * K);
                std::vector<float> yd((size_t) T * ldy), rd((size_t) T * ldy);
                cudaMemcpy(xd.data(), X, xd.size() * 2, cudaMemcpyDeviceToHost);
                cudaMemcpy(wd.data(), W, wd.size() * 2, cudaMemcpyDeviceToHost);
                cudaMemcpy(yd.data(), Y, yd.size() * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(rd.data(), ref, rd.size() * 4, cudaMemcpyDeviceToHost);
                std::FILE* fx = std::fopen(p, "wb");
                if (fx) {
                    int64_t hdr[4] = {T, N, K, ldy};
                    std::fwrite(hdr, sizeof(int64_t), 4, fx);
                    std::fwrite(xd.data(), 2, xd.size(), fx);
                    std::fwrite(wd.data(), 2, wd.size(), fx);
                    std::fwrite(yd.data(), 4, yd.size(), fx);
                    std::fwrite(rd.data(), 4, rd.size(), fx);
                    std::fclose(fx);
                    std::fprintf(stderr, "[tc_verify] dumped call T=%lld to %s\n", (long long) T, p);
                }
            }
        }
        return;
    }
    if (use_lt_ && !lt_fail_) {
        bool ok = true;
        if (lt_gemm_f16((cublasLtHandle_t) lt_handle_, X, W, Y, T, N, K, ldy, beta, workspace_, ws_bytes_,
                        (cudaStream_t) stream_, &ok))
            return;
        lt_fail_ = true;  // sticky fallback so a persistent Lt error doesn't retry on every call
    }
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx f16");
    STRATA_ABSORB_HIPBLAS_STICKY("cublasGemmEx f16");
}

void Gemm::gu_grouped(const GroupedExpert* etab_dev, const GroupedExpert* etab, int G, int total_tiles) {
    if (G <= 0 || total_tiles <= 0) return;
    const float alpha = 1.0f, beta = 0.0f;
    const int64_t ntot = (int64_t) total_tiles;  // grid.y bound
    const dim3 g(TC_M / 16, (int) ntot);
    g2_gu_kernel<<<g, 128, 0, (cudaStream_t) stream_>>>(etab_dev, G);
    if (tc_verify_ && etab) {
        // Debug: co-compute the cuBLAS reference for the first few skinny experts into the dequant
        // scratch and compare against the G2 outputs (same hook semantics as the per-call TC verify).
        const int nv = (G < 8) ? G : 8;
        float* ref = (float*) scratch_;
        cudaMemsetAsync(verify_dev_, 0, 2 * sizeof(float), (cudaStream_t) stream_);
        static long batches = 0;
        ++batches;
        for (int e = 0; e < nv; ++e) {
            const GroupedExpert& x = etab[e];
            ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, TC_M, x.ne, TC_K, &alpha, x.W,
                            CUDA_R_16F, TC_K, x.X, CUDA_R_16F, TC_K, &beta, ref, CUDA_R_32F, TC_M,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
               "cublasGemmEx g2_verify");
            tc_verify_cmp<<<8, 256, 0, (cudaStream_t) stream_>>>(x.Y, ref, x.ne * TC_M, verify_dev_, verify_dev_ + 1);
        }
        float hv[2] = {0.0f, 0.0f};
        cudaMemcpyAsync(hv, verify_dev_, 2 * sizeof(float), cudaMemcpyDeviceToHost, (cudaStream_t) stream_);
        cudaStreamSynchronize((cudaStream_t) stream_);
        if (hv[0] != 0.0f || batches <= 20) {
            std::fprintf(stderr, "[g2_verify] batch %ld experts=%d nneq=%.0f maxAbs=%.3e\n", batches, nv, hv[0], hv[1]);
        }
    }
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta) {
    if (N * K > scratch_elems_) {
        // Too large for the scratch at once: in row slices.
        const int64_t rows = scratch_elems_ / K;
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            f16(X, scratch_, Y + r0, T, n, K, ldy, beta);
        }
        return;
    }
    strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
    f16(X, scratch_, Y, T, N, K, ldy, beta);
}

}  // namespace strata::prefill
