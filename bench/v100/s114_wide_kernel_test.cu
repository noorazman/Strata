// bench/v100/s114_wide_kernel_test.cu — Stage 1.14: wide-memory-op expert dequant kernels.
//
// ncu (microbench + in-engine) shows the production dequant_gu_kernel is L1/TEX-bound (77 % of
// the L1 pipe, DRAM only 17 %, SM 9 %): the SASS issues 8 x LDG.E.U8 per grid entry and 8 x
// STG.E.U16 per thread for 8 output halves.  These kernels load each table entry with ONE
// wide load (the grids are uint64_t / uint32_t device arrays) and pack the 8 output halves
// into ONE 16 B store; PER = independent superblocks per thread for memory-level parallelism.
// The per-value math and evaluation order are the production dq_* functions verbatim, so the
// f16 output is bit-identical to the baseline kernel (verified here against it).
//
// Build: /usr/local/cuda/bin/nvcc -O3 -std=c++20 -arch=sm_70 -I ../../include -I ../../third_party/ggml \
//        -I ../../src/kernels/cuda s114_wide_kernel_test.cu -o /tmp/s114_wide
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>

#include "iq_kernels.cu"  // production kernels + block structs + device tables

// ---------------------------------------------------------------- helpers
// kmask_iq2xs / kvalues_iq4nl are __device__ int8_t / uint8_t arrays (1 B alignment only).
// Read bytes through 4 B-aligned sub-offsets so the wide loads are always aligned.
__device__ __forceinline__ uint64_t kmask8() {
    const uintptr_t a = (uintptr_t) kmask_iq2xs;
    const uint32_t* b = (const uint32_t*) (a & ~3ull);
    uint64_t m = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uintptr_t off = a + j;
        m |= (uint64_t) ((b[(off - (uintptr_t) b) >> 2] >> (8 * (off & 3))) & 0xff) << (8 * j);
    }
    return m;
}
__device__ __forceinline__ uint8_t kv_byte(int j) {
    const uintptr_t a = (uintptr_t) kvalues_iq4nl;
    const uint32_t* b = (const uint32_t*) (a & ~3ull);
    const uintptr_t off = a + j;
    return (uint8_t) ((b[(off - (uintptr_t) b) >> 2] >> (8 * (off & 3))) & 0xff);
}
__device__ __forceinline__ void store8(__half* y, float v[8]) {
    uint4 p;
    p.x = (unsigned) __half_as_ushort(__float2half(v[0])) | ((unsigned) __half_as_ushort(__float2half(v[1])) << 16);
    p.y = (unsigned) __half_as_ushort(__float2half(v[2])) | ((unsigned) __half_as_ushort(__float2half(v[3])) << 16);
    p.z = (unsigned) __half_as_ushort(__float2half(v[4])) | ((unsigned) __half_as_ushort(__float2half(v[5])) << 16);
    p.w = (unsigned) __half_as_ushort(__float2half(v[6])) | ((unsigned) __half_as_ushort(__float2half(v[7])) << 16);
    *(uint4*) (y + 32 * (int) (threadIdx.x & 7) + 8 * (int) (threadIdx.x >> 3)) = p;
}
__device__ __forceinline__ float sgn(float v, int signs, uint64_t mask, int j) { return signs & (int) (mask >> 8 * j) ? -v : v; }

// ---------------------------------------------------------------- wide per-type dequantizers
// tid: 0..31; ib = tid & 7, il = tid >> 3 (the production mapping).  The thread writes its 8
// halves at the production output offset (32*ib + 8*il within the superblock).
__device__ __forceinline__ void wide_iq2_xxs(const void* vx, int64_t ibs, __half* y, int tid) {
    const block_iq2_xxs* x = (const block_iq2_xxs*) vx;
    const int ib = tid & 7, il = tid >> 3;
    // production grid index: aux8[il] = BYTE il of the 8-byte group at qs + 8*ib bytes, i.e.
    // (il even) low byte of q2[il/2] else HIGH byte of q2[il/2]  (q2 = uint16 view of qs + 4*ib).
    const uint16_t q2h = x[ibs].qs[4 * ib + (il >> 1)];
    const uint8_t aux8 = (il & 1) ? (uint8_t) (q2h >> 8) : (uint8_t) q2h;
    const uint32_t aux32 = (uint32_t) x[ibs].qs[4 * ib + 2] | ((uint32_t) x[ibs].qs[4 * ib + 3] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    const uint64_t g64 = iq2xxs_grid[aux8];
    const uint8_t* gb = (const uint8_t*) &g64;
    const uint64_t mk = kmask8();
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = sgn(d * (float) gb[j], signs, mk, j);
    store8(y, v);
}
__device__ __forceinline__ void wide_iq2_xs(const void* vx, int64_t ibs, __half* y, int tid) {
    const block_iq2_xs* x = (const block_iq2_xs*) vx;
    const int ib = tid & 7, il = tid >> 3;
    const uint16_t q2k = x[ibs].qs[4 * ib + il];
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2k >> 9];
    const uint64_t g64 = iq2xs_grid[q2k & 511];
    const uint8_t* gb = (const uint8_t*) &g64;
    const uint64_t mk = kmask8();
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = sgn(d * (float) gb[j], signs, mk, j);
    store8(y, v);
}
__device__ __forceinline__ void wide_iq3_xxs(const void* vx, int64_t ibs, __half* y, int tid) {
    const block_iq3_xxs* x = (const block_iq3_xxs*) vx;
    const int ib = tid & 7, il = tid >> 3;
    const uint16_t q3p = ((const uint16_t*) x[ibs].qs)[4 * ib + il];
    const uint16_t* gas16 = (const uint16_t*) (x[ibs].qs + 64);   // block stride is 2 mod 4: keep 2 B loads
    const uint32_t gas = (uint32_t) gas16[2 * ib] | ((uint32_t) gas16[2 * ib + 1] << 16);
    const float d = (float) x[ibs].d * (0.5f + (gas >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(gas >> 7 * il) & 127];
    const uint32_t g1 = iq3xxs_grid[(uint8_t) q3p];
    const uint32_t g2 = iq3xxs_grid[(uint8_t) (q3p >> 8)];
    const uint8_t* a = (const uint8_t*) &g1, * b = (const uint8_t*) &g2;
    const uint64_t mk = kmask8();
    float v[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j + 0] = sgn(d * (float) a[j], signs, mk, j + 0);
        v[j + 4] = sgn(d * (float) b[j], signs, mk, j + 4);
    }
    store8(y, v);
}
__device__ __forceinline__ void wide_iq3_s(const void* vx, int64_t ibs, __half* y, int tid) {
    const block_iq3_s* x = (const block_iq3_s*) vx;
    const int ib = tid & 7, il = tid >> 3;
    const uint16_t qsp = ((const uint16_t*) x[ibs].qs)[4 * ib + il];
    const uint8_t qh = x[ibs].qh[ib];
    const float d = (float) x[ibs].d * (1 + 2 * ((x[ibs].scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4 * ib + il];
    const uint32_t g1 = iq3s_grid[(uint8_t) qsp | ((qh << (8 - 2 * il)) & 256)];
    const uint32_t g2 = iq3s_grid[(uint8_t) (qsp >> 8) | ((qh << (7 - 2 * il)) & 256)];
    const uint8_t* a = (const uint8_t*) &g1, * b = (const uint8_t*) &g2;
    const uint64_t mk = kmask8();
    float v[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j + 0] = sgn(d * (float) a[j], signs, mk, j + 0);
        v[j + 4] = sgn(d * (float) b[j], signs, mk, j + 4);
    }
    store8(y, v);
}
__device__ __forceinline__ void wide_iq2_s(const void* vx, int64_t ibs, __half* y, int tid) {
    const block_iq2_s* x = (const block_iq2_s*) vx;
    const int ib = tid & 7, il = tid >> 3;
    const uint8_t qsb = x[ibs].qs[4 * ib + il];
    const uint8_t qh = x[ibs].qh[ib];
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K / 8 + 4 * ib + il];
    const uint64_t g64 = iq2s_grid[(uint32_t) qsb | ((uint32_t) qh << (8 - 2 * il)) & 0x300];
    const uint8_t* gb = (const uint8_t*) &g64;
    const uint64_t mk = kmask8();
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = sgn(d * (float) gb[j], signs, mk, j);
    store8(y, v);
}
__device__ __forceinline__ void wide_iq4_nl(const void* vx, int64_t ibs, __half* y, int tid) {
    // 32-value blocks; the thread handles 8 values of block ib: 4 low + 4 high nibbles, at
    // output offsets 32*ib + 4*il (+0..3) and +16 (+0..3) of the superblock.
    const block_iq4_nl* x = (const block_iq4_nl*) vx + ibs * (QK_K / QK4_NL);
    const int ib = tid & 7, il = tid >> 3;
    const uint16_t q01 = ((const uint16_t*) x[ib].qs)[2 * il];
    const uint16_t q23 = ((const uint16_t*) x[ib].qs)[2 * il + 1];
    const float d = (float) x[ib].d;
    const uint32_t q4w = (uint32_t) q01 | ((uint32_t) q23 << 16);   // qs[4il..4il+3]
    const uint8_t* qb = (const uint8_t*) &q4w;
    float v[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j + 0] = d * (float) ((int8_t) kv_byte((int) qb[j] & 0xf));
        v[j + 4] = d * (float) ((int8_t) kv_byte((int) qb[j] >> 4));
    }
    *(uint2*) (y + 32 * ib + 4 * il) = make_uint2(
        (unsigned) __half_as_ushort(__float2half(v[0])) | ((unsigned) __half_as_ushort(__float2half(v[1])) << 16),
        (unsigned) __half_as_ushort(__float2half(v[2])) | ((unsigned) __half_as_ushort(__float2half(v[3])) << 16));
    *(uint2*) (y + 32 * ib + 4 * il + 16) = make_uint2(
        (unsigned) __half_as_ushort(__float2half(v[4])) | ((unsigned) __half_as_ushort(__float2half(v[5])) << 16),
        (unsigned) __half_as_ushort(__float2half(v[6])) | ((unsigned) __half_as_ushort(__float2half(v[7])) << 16));
}
__device__ __forceinline__ void wide_q2_0(const void* vx, int64_t ibs, __half* y, int tid) {
    const block_q2_0* x = (const block_q2_0*) vx + ibs * 4;
    const int b = tid >> 3, part = tid & 7;
    const float d = (float) x[b].d;
    const uint16_t q2 = ((const uint16_t*) x[b].qs)[part];
    const uint8_t lo = (uint8_t) q2, hi = (uint8_t) (q2 >> 8);
    float v[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j + 0] = d * (float) (((lo >> (2 * j)) & 3) - 1);
        v[j + 4] = d * (float) (((hi >> (2 * j)) & 3) - 1);
    }
    // Q2_0 output mapping differs from the GU mapping: thread writes yy[b*64 + part*8 .. +7]
    uint4 p;
    p.x = (unsigned) __half_as_ushort(__float2half(v[0])) | ((unsigned) __half_as_ushort(__float2half(v[1])) << 16);
    p.y = (unsigned) __half_as_ushort(__float2half(v[2])) | ((unsigned) __half_as_ushort(__float2half(v[3])) << 16);
    p.z = (unsigned) __half_as_ushort(__float2half(v[4])) | ((unsigned) __half_as_ushort(__float2half(v[5])) << 16);
    p.w = (unsigned) __half_as_ushort(__float2half(v[6])) | ((unsigned) __half_as_ushort(__float2half(v[7])) << 16);
    *(uint4*) (y + 64 * b + 8 * part) = p;
}

template<int TY> struct WideSel;
template<> struct WideSel<16> { static __device__ void f(const void* v, int64_t i, __half* y, int t) { wide_iq2_xxs(v, i, y, t); } };
template<> struct WideSel<17> { static __device__ void f(const void* v, int64_t i, __half* y, int t) { wide_iq2_xs(v, i, y, t); } };
template<> struct WideSel<18> { static __device__ void f(const void* v, int64_t i, __half* y, int t) { wide_iq3_xxs(v, i, y, t); } };
template<> struct WideSel<20> { static __device__ void f(const void* v, int64_t i, __half* y, int t) { wide_iq4_nl(v, i, y, t); } };
template<> struct WideSel<21> { static __device__ void f(const void* v, int64_t i, __half* y, int t) { wide_iq3_s(v, i, y, t); } };
template<> struct WideSel<22> { static __device__ void f(const void* v, int64_t i, __half* y, int t) { wide_iq2_s(v, i, y, t); } };
template<> struct WideSel<42> { static __device__ void f(const void* v, int64_t i, __half* y, int t) { wide_q2_0(v, i, y, t); } };

template<int TY, int PER>
__global__ void __launch_bounds__(32) wide_gu(int ty, const void* __restrict__ gate, const void* __restrict__ up,
                                              int64_t per_row, __half* __restrict__ y) {
    const int64_t base = (int64_t) blockIdx.x * PER;
    const int parity = blockIdx.y;
    const void* vx = parity ? up : gate;
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int64_t i = base + p;
        const int64_t r = i / per_row, c = i % per_row;
        WideSel<TY>::f(vx, i, y + ((2 * r + parity) * per_row + c) * 256, (int) threadIdx.x);
    }
}
template<int TY, int PER>
__global__ void __launch_bounds__(32) wide_flat(int ty, const void* __restrict__ vx, __half* __restrict__ y) {
    const int64_t base = (int64_t) blockIdx.x * PER;
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int64_t i = base + p;
        WideSel<TY>::f(vx, i, y + i * 256, (int) threadIdx.x);
    }
}

// ---------------------------------------------------------------- alignment probe
__global__ void probe_align(uint64_t* out) {
    out[0] = (uintptr_t) kmask_iq2xs;
    out[1] = (uintptr_t) ksigns_iq2xs;
    out[2] = (uintptr_t) kvalues_iq4nl;
    out[3] = (uintptr_t) iq2xxs_grid;
    out[4] = (uintptr_t) iq2xs_grid;
    out[5] = (uintptr_t) iq2s_grid;
    out[6] = (uintptr_t) iq3xxs_grid;
    out[7] = (uintptr_t) iq3s_grid;
}

// ---------------------------------------------------------------- test harness
const int64_t N_FF = 640, N_EMBD = 2560;
const int64_t PER_ROW = N_EMBD / 256;
const int64_t GU_SB = N_FF * PER_ROW;
const int64_t D_SB = N_EMBD * N_FF / 256;

// Host re-implementation of the production dq_* math (identical evaluation order), for
// debugging: verifies the GPU ref/got against a third independent oracle.
struct HostTables {
    uint8_t kmask[8], ksigns[128], kv[16];
    uint64_t g2xxs[256], g2xs[512], g2s[1024];
    uint32_t g3xxs[256], g3s[512];
};
static HostTables g_ht;
static void init_host_tables() {
    void* dev;
    auto cp = [&](const void* sym, void* dst, size_t bytes) {
        cudaGetSymbolAddress(&dev, sym);
        cudaMemcpy(dst, dev, bytes, cudaMemcpyDeviceToHost);
    };
    cp(kmask_iq2xs, g_ht.kmask, 8);
    cp(ksigns_iq2xs, g_ht.ksigns, 128);
    cp(kvalues_iq4nl, g_ht.kv, 16);
    cp(iq2xxs_grid, g_ht.g2xxs, 256 * 8);
    cp(iq2xs_grid, g_ht.g2xs, 512 * 8);
    cp(iq2s_grid, g_ht.g2s, 1024 * 8);
    cp(iq3xxs_grid, g_ht.g3xxs, 256 * 4);
    cp(iq3s_grid, g_ht.g3s, 512 * 4);
}
static float hsgn(float v, uint8_t signs, int j) { return signs & g_ht.kmask[j] ? -v : v; }
// f16 bit-pattern decode on the host, matching the GPU's native (float)__half (cvt.f32.f16).
// The CUDA host intrinsics route through float, so we decode the IEEE-754 half bits directly.
static float hf(uint16_t bits) {
    const uint32_t sign = (bits >> 15) & 1u;
    const uint32_t exp  = (bits >> 10) & 0x1Fu;
    const uint32_t mant = bits & 0x3FFu;
    const float sgn = sign ? -1.f : 1.f;
    if (exp == 0u) {
        // subnormal: value = mant * 2^-24
        return sgn * (float) mant * (1.f / 16777216.f);
    }
    if (exp == 0x1Fu) {
        return mant ? NAN : (sign ? -INFINITY : INFINITY);
    }
    uint32_t out = (sign << 31) | ((exp + (127u - 15u)) << 23) | (mant << (23u - 10u));
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}
// Host oracle: 8 values for thread `tid` of superblock `ibs` (same mapping + math as production).
// GU types: ib = tid%8, il = tid/8. D type 20: ib = tid%8, il = tid/8. D type 42: b = tid/8, part = tid%8.
static void host_dq(int ty, const uint8_t* blob, size_t block_size, int64_t ibs, int tid, float v[8]) {
    const uint8_t* b = blob + (size_t) ibs * block_size;
    auto h16 = [&](int off) { return (uint16_t) (b[off] | (b[off + 1] << 8)); };
    const float d0 = hf(h16(0));
    const int ib = tid % 8, il = tid / 8;
    if (ty == 16) {
        const uint16_t* q2u = (const uint16_t*) (b + 2);
        const uint8_t* q2b = b + 2;
        const uint8_t aux8 = q2b[8 * ib + il];
        const uint32_t aux32 = (uint32_t) q2u[4 * ib + 2] | ((uint32_t) q2u[4 * ib + 3] << 16);
        const float d = d0 * (0.5f + (aux32 >> 28)) * 0.25f;
        const uint8_t signs = g_ht.ksigns[(aux32 >> 7 * il) & 127];
        const uint64_t g64 = g_ht.g2xxs[aux8];
        const uint8_t* gb = (const uint8_t*) &g64;
        for (int j = 0; j < 8; ++j) v[j] = hsgn(d * (float) gb[j], signs, j);
    } else if (ty == 17) {
        const uint16_t q2k = h16(2 + 8 * ib + 2 * il);
        const float d = d0 * (0.5f + ((b[66 + ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
        const uint8_t signs = g_ht.ksigns[q2k >> 9];
        const uint64_t g64 = g_ht.g2xs[q2k & 511];
        const uint8_t* gb = (const uint8_t*) &g64;
        for (int j = 0; j < 8; ++j) v[j] = hsgn(d * (float) gb[j], signs, j);
    } else if (ty == 18) {
        const uint16_t q3p = h16(2 + 8 * ib + 2 * il);
        const uint32_t gas = (uint32_t) h16(66 + 4 * ib) | ((uint32_t) h16(68 + 4 * ib) << 16);
        const float d = d0 * (0.5f + (gas >> 28)) * 0.5f;
        const uint8_t signs = g_ht.ksigns[(gas >> 7 * il) & 127];
        const uint32_t g1 = g_ht.g3xxs[q3p & 0xff];
        const uint32_t g2 = g_ht.g3xxs[(q3p >> 8) & 0xff];
        const uint8_t* a = (const uint8_t*) &g1, * c = (const uint8_t*) &g2;
        for (int j = 0; j < 4; ++j) {
            v[j + 0] = hsgn(d * (float) a[j], signs, j + 0);
            v[j + 4] = hsgn(d * (float) c[j], signs, j + 4);
        }
    } else if (ty == 21) {
        const uint8_t* qs = b + 2 + 8 * ib;
        const uint8_t qh = b[66 + ib];
        const float d = d0 * (1 + 2 * ((b[106 + ib / 2] >> 4 * (ib % 2)) & 0xf));
        const uint8_t signs = b[74 + 4 * ib + il];
        const uint32_t g1 = g_ht.g3s[(uint8_t) qs[2 * il] | ((qh << (8 - 2 * il)) & 256)];
        const uint32_t g2 = g_ht.g3s[(uint8_t) qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 256)];
        const uint8_t* a = (const uint8_t*) &g1, * c = (const uint8_t*) &g2;
        for (int j = 0; j < 4; ++j) {
            v[j + 0] = hsgn(d * (float) a[j], signs, j + 0);
            v[j + 4] = hsgn(d * (float) c[j], signs, j + 4);
        }
    } else if (ty == 22) {
        const uint8_t qsb = b[2 + 4 * ib + il];
        const uint8_t qh = b[66 + ib];
        const float d = d0 * (0.5f + ((b[74 + ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
        const uint8_t signs = b[34 + 4 * ib + il];
        const uint64_t g64 = g_ht.g2s[(uint32_t) qsb | ((uint32_t) qh << (8 - 2 * il)) & 0x300];
        const uint8_t* gb = (const uint8_t*) &g64;
        for (int j = 0; j < 8; ++j) v[j] = hsgn(d * (float) gb[j], signs, j);
    } else if (ty == 20) {
        const uint8_t* bb = b + 18 * ib;
        const float d = hf(((const uint16_t*) bb)[0]);
        for (int j = 0; j < 4; ++j) {
            const uint8_t q = bb[2 + 4 * il + j];
            v[j + 0] = d * (float) (int8_t) g_ht.kv[q & 0xf];
            v[j + 4] = d * (float) (int8_t) g_ht.kv[q >> 4];
        }
    } else if (ty == 42) {
        const int blk = tid / 8, part = tid % 8;
        const uint8_t* bb = b + 18 * blk;
        const float d = hf(((const uint16_t*) bb)[0]);
        const uint8_t lo = bb[2 + 2 * part], hi = bb[3 + 2 * part];
        for (int j = 0; j < 4; ++j) {
            v[j + 0] = d * (float) (((lo >> (2 * j)) & 3) - 1);
            v[j + 4] = d * (float) (((hi >> (2 * j)) & 3) - 1);
        }
    }
}
// Decode a W-buffer byte position to (superblock, thread, j) and check ref/got against the host oracle.
static void analyze_diff(int ty, bool flat, const std::vector<uint8_t>& blob, const std::vector<uint8_t>& ref,
                         const std::vector<uint8_t>& got, size_t nbytes, size_t block_bytes) {
    if (ty == 21) {  // debug: full position list for the first superblock
        int cnt = 0;
        std::vector<int> pos;
        for (size_t p = 0; p < 512; ++p) if (ref[p] != got[p]) { pos.push_back((int) p); cnt++; }
        printf("  type21 sb0: %d byte-diffs: ", cnt);
        for (int q = 0; q < (int) pos.size() && q < 40; ++q) printf("%d(h%d) ", pos[q], pos[q] / 2);
        printf("\n");
        // detailed input dump for ibs=0, ib=6
        const uint8_t* b = blob.data();
        auto h16h = [&](int off) { return (uint16_t) (b[off] | (b[off + 1] << 8)); };
        const int ib = 6, il = 0;
        const uint8_t qh = b[66 + ib];
        const uint8_t sc = b[106 + ib / 2];
        const uint8_t sg = b[74 + 4 * ib + il];
        const float d = __half2float(h16h(0)) * (1 + 2 * ((sc >> 4 * (ib % 2)) & 0xf));
        printf("  dump: d0 %04x=%g scales3 %02x qh6 %02x signs24 %02x d %g\n", h16h(0), hf(h16h(0)), sc, qh, sg, d);
        for (int ilv = 0; ilv < 4; ++ilv) {
            const uint8_t qs0 = b[2 + 8 * ib + 2 * ilv], qs1 = b[3 + 8 * ib + 2 * ilv];
            const int i1 = qs0 | ((qh << (8 - 2 * ilv)) & 256);
            const int i2 = qs1 | ((qh << (7 - 2 * ilv)) & 256);
            const uint32_t e1 = g_ht.g3s[i1], e2 = g_ht.g3s[i2];
            const uint8_t* a1 = (const uint8_t*) &e1, * a2 = (const uint8_t*) &e2;
            printf("  il %d: qs %02x %02x g1idx %03x {%02x %02x %02x %02x} g2idx %03x {%02x %02x %02x %02x}\n",
                   ilv, qs0, qs1, i1, a1[0], a1[1], a1[2], a1[3], i2, a2[0], a2[1], a2[2], a2[3]);
        }
        for (int ilv = 0; ilv < 4; ++ilv) {
            const int hbase = 32 * ib + 8 * ilv;
            printf("  ref halves %d..%d: ", hbase, hbase + 7);
            for (int j = 0; j < 8; ++j) {
                const uint16_t hv = (uint16_t) (ref[2 * (hbase + j)] | (ref[2 * (hbase + j) + 1] << 8));
                printf("%04x(%g) ", hv, hf(hv));
            }
            printf("\n");
        }
    }
    int shown = 0;
    for (size_t p = 0; p < nbytes && shown < 4; ++p) {
        if (ref[p] == got[p]) continue;
        const size_t k = p / 512;
        const int h = (int) ((p % 512) / 2);
        int64_t ibs; int tid; int j; int which;
        if (flat) {
            ibs = (int64_t) k;
            if (ty == 20) {
                const int ib = h / 32, h32 = h % 32;
                if (h32 < 16) { tid = 8 * (h32 / 4) + ib; j = h32 % 4; which = 0; }
                else { tid = 8 * ((h32 - 16) / 4) + ib; j = (h32 - 16) % 4; which = 1; }
            } else {  // ty == 42
                const int b2 = h / 64, part = (h % 64) / 8;
                tid = 8 * b2 + part; j = h % 8; which = 0;
            }
        } else {
            const int rp = (int) (k / PER_ROW), c = (int) (k % PER_ROW);
            ibs = (int64_t) (rp / 2) * PER_ROW + c;
            (void) rp; (void) c;
            const int ib = h / 32, h32 = h % 32;
            tid = 8 * (h32 / 8) + ib; j = h32 % 8; which = 0;
        }
        float v[8];
        host_dq(ty, blob.data(), block_bytes, ibs, tid, v);
        const uint16_t want = (uint16_t) __half_as_ushort(__float2half(v[which ? j + 4 : j]));
        const uint16_t gotv = (uint16_t) (got[p] | (got[(p & 1) ? p - 1 : p + 1] << 8));
        const uint16_t refv = (uint16_t) (ref[p] | (ref[(p & 1) ? p - 1 : p + 1] << 8));
        printf("  diff@%zu: sb %zu h %d ibs %lld tid %d j %d want %04x ref %04x got %04x %s%s%s\n",
               p, k, h, (long long) ibs, tid, j, want, refv, gotv,
               (want == refv) ? "refOK " : "refBAD", (want == gotv) ? "gotOK" : "gotBAD",
               (want == refv) ? (want == gotv ? " (both OK?)" : " (WIDE WRONG)") : (want == gotv ? " (PROD WRONG!)" : " (both wrong)"));
        shown++;
    }
}

template<typename F>
static double us_of(F f, cudaStream_t s) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 10; ++i) f();
    std::vector<double> t;
    for (int i = 0; i < 50; ++i) {
        cudaEventRecord(a, s); f(); cudaEventRecord(b, s); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b); t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    cudaEventDestroy(a); cudaEventDestroy(b);
    return t[25] * 1e3;
}

#define LAUNCH_WIDE_GU(ty, per) \
    switch (ty) { \
        case 16: wide_gu<16, per><<<dim3((unsigned) (GU_SB / per), 2), 32, 0, s>>>(ty, gate, up, PER_ROW, (__half*) W2); break; \
        case 17: wide_gu<17, per><<<dim3((unsigned) (GU_SB / per), 2), 32, 0, s>>>(ty, gate, up, PER_ROW, (__half*) W2); break; \
        case 18: wide_gu<18, per><<<dim3((unsigned) (GU_SB / per), 2), 32, 0, s>>>(ty, gate, up, PER_ROW, (__half*) W2); break; \
        case 21: wide_gu<21, per><<<dim3((unsigned) (GU_SB / per), 2), 32, 0, s>>>(ty, gate, up, PER_ROW, (__half*) W2); break; \
        case 22: wide_gu<22, per><<<dim3((unsigned) (GU_SB / per), 2), 32, 0, s>>>(ty, gate, up, PER_ROW, (__half*) W2); break; \
        default: std::fprintf(stderr, "GU type %d not in wide set\n", ty); std::exit(1); \
    }
#define LAUNCH_WIDE_FLAT(ty, per) \
    switch (ty) { \
        case 20: wide_flat<20, per><<<(unsigned) (D_SB / per), 32, 0, s>>>(ty, down, (__half*) W2); break; \
        case 42: wide_flat<42, per><<<(unsigned) (D_SB / per), 32, 0, s>>>(ty, down, (__half*) W2); break; \
        default: std::fprintf(stderr, "D type %d not in wide set\n", ty); std::exit(1); \
    }

static size_t block_bytes(int ty) {
    switch (ty) { case 16: return 66; case 17: return 74; case 18: return 98; case 21: return 110; case 22: return 82; default: return 18; }
}
int main() {
    cudaStream_t s; cudaStreamCreate(&s);
    init_host_tables();
    uint64_t* align;
    cudaMalloc(&align, 8 * 8);
    probe_align<<<1, 1, 0, s>>>(align);
    cudaStreamSynchronize(s);
    std::vector<uint64_t> h(8);
    cudaMemcpy(h.data(), align, 64, cudaMemcpyDeviceToHost);
    const char* names[8] = {"kmask_iq2xs", "ksigns_iq2xs", "kvalues_iq4nl", "iq2xxs_grid",
                            "iq2xs_grid", "iq2s_grid", "iq3xxs_grid", "iq3s_grid"};
    for (int i = 0; i < 8; ++i) printf("  %s: %%8=%llu %%4=%llu\n", names[i],
                                       (unsigned long long) (h[i] & 7), (unsigned long long) (h[i] & 3));
    cudaFree(align);

    const int gu_types[] = {16, 17, 18, 21, 22};
    for (int ty : gu_types) {
        const size_t role_bytes = (size_t) N_FF * strata::kernels::iq_row_bytes(ty, N_EMBD);
        std::vector<uint8_t> blob(role_bytes);
        for (size_t i = 0; i < blob.size(); ++i) blob[i] = (uint8_t) ((i * 2654435761u) >> 8);
        void *gate, *up, *W, *W2;
        cudaMalloc(&gate, role_bytes); cudaMalloc(&up, role_bytes);
        cudaMalloc(&W, (size_t) 2 * N_FF * N_EMBD * 2);
        cudaMalloc(&W2, (size_t) 2 * N_FF * N_EMBD * 2);
        cudaMemcpy(gate, blob.data(), role_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(up, blob.data(), role_bytes, cudaMemcpyHostToDevice);
        strata::kernels::iq_dequant_gu_f16(ty, gate, up, N_FF, N_EMBD, (uint16_t*) W, s);
        cudaStreamSynchronize(s);

        auto base = [&] { strata::kernels::iq_dequant_gu_f16(ty, gate, up, N_FF, N_EMBD, (uint16_t*) W, s); };
        auto w1 = [&] { LAUNCH_WIDE_GU(ty, 1); };
        auto w2 = [&] { LAUNCH_WIDE_GU(ty, 2); };
        double tb = us_of(base, s), t1 = us_of(w1, s), t2 = us_of(w2, s);

        std::vector<uint8_t> ref((size_t) 2 * N_FF * N_EMBD * 2), got(ref.size());
        cudaMemcpy(ref.data(), W, ref.size(), cudaMemcpyDeviceToHost);
        w1(); cudaMemcpy(got.data(), W2, got.size(), cudaMemcpyDeviceToHost);
        bool ok1 = (ref == got);
        w2(); cudaMemcpy(got.data(), W2, got.size(), cudaMemcpyDeviceToHost);
        bool ok2 = (ref == got);
        printf("GU type %2d: base %8.1f us | wide-1 %8.1f us (%.2fx) | wide-2 %8.1f us (%.2fx) | bit-ident: %s/%s\n",
               ty, tb, t1, tb / t1, t2, tb / t2, ok1 ? "y" : "N", ok2 ? "y" : "N");
        if (!ok1 || !ok2) {
            for (size_t i = 0; i < ref.size(); ++i) if (ref[i] != got[i]) {
                printf("  first diff at byte %zu: ref %02x got %02x\n", i, ref[i], got[i]); break;
            }
            analyze_diff(ty, false, blob, ref, got, ref.size(), block_bytes(ty));
        }
        cudaFree(gate); cudaFree(up); cudaFree(W); cudaFree(W2);
    }

    const int d_types[] = {20, 42};
    for (int ty : d_types) {
        const size_t d_bytes = (size_t) N_EMBD * strata::kernels::iq_row_bytes(ty, N_FF);
        std::vector<uint8_t> blob(d_bytes);
        for (size_t i = 0; i < blob.size(); ++i) blob[i] = (uint8_t) ((i * 40503u) >> 8);
        void *down, *W, *W2;
        cudaMalloc(&down, d_bytes); cudaMalloc(&W, (size_t) N_EMBD * N_FF * 2);
        cudaMalloc(&W2, (size_t) N_EMBD * N_FF * 2);
        cudaMemcpy(down, blob.data(), d_bytes, cudaMemcpyHostToDevice);
        strata::kernels::iq_dequant_f16(ty, down, N_EMBD * N_FF, (uint16_t*) W, s);
        cudaStreamSynchronize(s);

        auto base = [&] { strata::kernels::iq_dequant_f16(ty, down, N_EMBD * N_FF, (uint16_t*) W, s); };
        auto w1 = [&] { LAUNCH_WIDE_FLAT(ty, 1); };
        auto w2 = [&] { LAUNCH_WIDE_FLAT(ty, 2); };
        double tb = us_of(base, s), t1 = us_of(w1, s), t2 = us_of(w2, s);

        std::vector<uint8_t> ref((size_t) N_EMBD * N_FF * 2), got(ref.size());
        cudaMemcpy(ref.data(), W, ref.size(), cudaMemcpyDeviceToHost);
        w1(); cudaMemcpy(got.data(), W2, got.size(), cudaMemcpyDeviceToHost);
        bool ok1 = (ref == got);
        w2(); cudaMemcpy(got.data(), W2, got.size(), cudaMemcpyDeviceToHost);
        bool ok2 = (ref == got);
        printf("D  type %2d: base %8.1f us | wide-1 %8.1f us (%.2fx) | wide-2 %8.1f us (%.2fx) | bit-ident: %s/%s\n",
               ty, tb, t1, tb / t1, t2, tb / t2, ok1 ? "y" : "N", ok2 ? "y" : "N");
        if (!ok1 || !ok2) {
            for (size_t i = 0; i < ref.size(); ++i) if (ref[i] != got[i]) {
                printf("  first diff at byte %zu: ref %02x got %02x\n", i, ref[i], got[i]); break;
            }
            analyze_diff(ty, true, blob, ref, got, ref.size(), block_bytes(ty));
        }
        cudaFree(down); cudaFree(W); cudaFree(W2);
    }
    cudaStreamDestroy(s);
    return 0;
}
