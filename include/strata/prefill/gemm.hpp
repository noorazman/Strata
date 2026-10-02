// include/strata/prefill/gemm.hpp - plan v0.3 P5: the batched projections of prompt processing.
//
// Every projection of a chunk of T tokens is Y[T, N] = X[T, K] . W[N, K]^T with W row-major (the GGUF / pack layout)
// and FP32 outputs.  Weights are BF16 on the device - either already (the pack's BF16 tensors) or dequantized from
// their native GGUF blocks into a reusable scratch (`dequant_bf16`) right before the product - and activations are
// rounded to BF16, which is also what llama.cpp's batched CUDA path does.  Tensor-core GEMM through cuBLAS.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::prefill {

/// Stage 1.17: one row of the grouped skinny-GU table.  `tile0` is the n16-tile prefix in units of
/// 80 m-blocks (the G2 grid is (80, total n16 tiles)); `tile0 / 80` is the expert's first n16 tile.
struct GroupedExpert {
    const uint16_t* X;  // [ne, 2560] f16 activation rows (row-major)
    const uint16_t* W;  // [1280, 2560] f16 dequantized gate/up weight (row-major)
    float* Y;           // [ne, 1280] f32 output (row-major)
    int ne;             // rows (1..128)
    int tile0;          // 80 x (n16 tile prefix)
};

class Gemm {
public:
    Gemm() = default;
    ~Gemm();
    Gemm(const Gemm&) = delete;
    Gemm& operator=(const Gemm&) = delete;

    /// `scratch_elems`: BF16 elements of the dequantization scratch (the largest weight dequantized at once).
    bool init(void* stream, int64_t scratch_elems, std::string& err);
    /// The same with caller-owned device buffers (the prompt path borrowing expert-cache slots).
    bool init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                       std::string& err);

    /// Y[T, N] (fp32, row stride ldy) = X[T, K] (bf16, row-major) . W[N, K]^T (bf16, row-major).  `beta` = 1 adds.
    void bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy = 0,
              float beta = 0.0f);

    /// Y = X . W^T with both in FP16 (bits).
    void f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy = 0,
             float beta = 0.0f);

    /// W given as native GGUF blocks of `ggml_type`, dequantized to FP16 in the scratch, X in FP16.
    void native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                int64_t ldy = 0, float beta = 0.0f);

    /// Stage 1.17: one launch over `G` skinny GU experts (the G2 grouped kernel, m16 x n16 tiles,
    /// 4-way in-block k-split, deterministic smem reduce).  `etab_dev` = the device copy of `etab`
    /// (only the first `G` rows), already on the stream; `total_tiles` = sum over the rows of
    /// ceil(ne / 16).  Shape is fixed (N = 1280, K = 2560, ldy = 1280).  `etab` (host) is used only
    /// for the STRATA_MOE_GEMM_TC_VERIFY co-computation and may be null.
    void gu_grouped(const GroupedExpert* etab_dev, const GroupedExpert* etab, int G, int total_tiles);
    /// Caller-owned buffers only: the scratch and workspace moved (the prompt path laid its buffers out again).
    void rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes);

    uint16_t* scratch() const { return scratch_; }
    int64_t scratch_elems() const { return scratch_elems_; }
    void* stream() const { return stream_; }
    int tc_max_ne() const { return (int) tc_max_ne_; }
    bool tc_verify_on() const { return tc_verify_; }

private:
    void* handle_ = nullptr;
    void* stream_ = nullptr;
    uint16_t* scratch_ = nullptr;
    int64_t scratch_elems_ = 0;
    void* workspace_ = nullptr;
    bool external_ = false;
    size_t ws_bytes_ = 0;
    void* lt_handle_ = nullptr;  // cublasLt handle (Stage 1.15: STRATA_MOE_GEMM_LT)
    bool use_lt_ = false;        // route f16() through cublasLtMatmul when true
    bool lt_fail_ = false;       // sticky: fall back to cublasGemmEx after an Lt error
    bool use_tc_ = false;        // Stage 1.16: route skinny GU f16() GEMMs through the tensor-core kernel
    int64_t tc_max_ne_ = 64;     // Stage 1.16: largest T routed to the TC kernel (STRATA_MOE_GEMM_TC_MAXNE, 1..128)
    bool tc_verify_ = false;     // Stage 1.16 debug: also compute cuBLAS for routed calls and compare (STRATA_MOE_GEMM_TC_VERIFY)
    float* verify_dev_ = nullptr;
    void* hipblaslt_state_ = nullptr;
};



}  // namespace strata::prefill
