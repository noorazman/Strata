// include/strata/kernels/cpu/iq_avx2.hpp - Stage 1.4: AVX2 multi-token i-quant rows for the expert pool.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// true when the CPU has AVX2+FMA (the ISA this file is compiled for).
bool cpu_avx2_ok() noexcept;
/// Whether `iqavx2_gu_rows`/`iqavx2_rows` can serve this ggml type (the i-quant expert formats).
bool iqavx2_supported(int ggml_type) noexcept;

/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]), rows [r0, r1); gate rows at blob, up rows at blob + up_off.
/// Weights are decoded once per row and shared across all `nt` tokens; the per-(row, token) integer and float
/// math is ggml-cpu's AVX2 dot, bit-identical to the per-token path.
void iqavx2_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                    int nt, float* const* ff, int r0, int r1);

/// out[t][r] = w_r . a[t], rows [r0, r1).
void iqavx2_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                 float* const* out, int r0, int r1);

}  // namespace strata::kernels::cpu
