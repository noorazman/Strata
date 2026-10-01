// src/prefill/gemm.cu - see include/strata/prefill/gemm.hpp.
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>

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
}  // namespace

Gemm::~Gemm() {
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
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { err = "prefill gemm: cublasCreate failed"; return false; }
    handle_ = h;
    stream_ = stream;
    external_ = true;
    cublasSetStream(h, (cudaStream_t) stream);
    workspace_ = workspace;
    ws_bytes_ = ws_bytes;
    cublasSetWorkspace(h, workspace_, ws_bytes);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
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
    return true;
}

bool Gemm::init(void* stream, int64_t scratch_elems, std::string& err) {
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { err = "prefill gemm: cublasCreate failed"; return false; }
    handle_ = h;
    stream_ = stream;
    cublasSetStream(h, (cudaStream_t) stream);
    // A fixed workspace so the handle never allocates on the way (and graphs could capture it later).
    const size_t ws = 32u << 20;
    if (cudaMalloc(&workspace_, ws) != cudaSuccess) { err = "prefill gemm: workspace"; return false; }
    ws_bytes_ = ws;
    cublasSetWorkspace(h, workspace_, ws);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
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
    if (scratch_elems > 0 && cudaMalloc((void**) &scratch_, (size_t) scratch_elems * 2) != cudaSuccess) {
        err = "prefill gemm: dequant scratch of " + std::to_string(scratch_elems * 2 >> 20) + " MiB";
        return false;
    }
    scratch_elems_ = scratch_elems;
    return true;
}

void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    const float alpha = 1.0f;
    // Column-major view: Y^T[N, T] = W[N, K] (stored K x N col-major, transposed) . X^T[K, T].
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16BF, (int) K, X, CUDA_R_16BF, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx");
}

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    const float alpha = 1.0f;
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
