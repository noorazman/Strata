// src/prefill/gemm.cu - see include/strata/prefill/gemm.hpp.
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <unordered_map>

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
}  // namespace

Gemm::~Gemm() {
    if (lt_handle_) cublasLtDestroy((cublasLtHandle_t) lt_handle_);
    if (handle_) cublasDestroy((cublasHandle_t) handle_);
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
