// bench/v100/s114_dq_roofline.cu — the memory-system ceiling for the dequant I/O pattern:
// each warp reads a small (128 B) scattered chunk and writes a 512 B contiguous span.
// K0: plain f16 copy of the same total output volume (the generic roofline reference).
// K1: the exact dequant access shape (2 x 128 B in, 512 B out per warp), no ALU.
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>
#include <vector>
#include <algorithm>

__global__ void copy_kernel(const __half* __restrict__ in, __half* __restrict__ out, long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}
__global__ void dequant_shape_kernel(const uint8_t* __restrict__ in, __half* __restrict__ out, int n_warps) {
    // warp w: reads 2 x 128 B from in (scattered, 512 B stride), writes 512 B contiguous.
    const int w = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (w >= n_warps) return;
    const int lane = threadIdx.x & 31;
    const uint4* a = (const uint4*) (in + (size_t) w * 512 + lane * 16);
    const uint4* b = a + 8;  // second 128 B, 128 B apart
    uint4 va = *a, vb = *b;
    (void) va; (void) vb;
    uint4* o = (uint4*) (out + (size_t) w * 256 + lane * 8);
    *(o + 0) = make_uint4(__half_as_ushort(1.0f), __half_as_ushort(1.0f), __half_as_ushort(1.0f), __half_as_ushort(1.0f));
    *(o + 1) = make_uint4(__half_as_ushort(1.0f), __half_as_ushort(1.0f), __half_as_ushort(1.0f), __half_as_ushort(1.0f));
}
static double us_of(const char* tag, void* p) { (void) tag; (void) p; return 0; }
template<typename F>
static double us_of_f(F f, cudaStream_t s) {
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
int main() {
    const long long N_OUT = (long long) 1280 * 2560;         // 3.28M halves = 6.55 MB
    const int N_WARPS = (int) (N_OUT / 256);                 // 512 B writes per warp -> 12,800
    const size_t IN_BYTES = (size_t) N_WARPS * 256;          // 2 x 128 B per warp = 3.28 MB
    __half *a, *o; uint8_t* in;
    cudaMalloc(&a, N_OUT * 2); cudaMalloc(&o, N_OUT * 2); cudaMalloc(&in, IN_BYTES);
    cudaStream_t s; cudaStreamCreate(&s);
    double t0 = us_of_f([&] { copy_kernel<<<(unsigned) ((N_OUT + 255) / 256), 256, 0, s>>>(a, o, N_OUT); }, s);
    double t1 = us_of_f([&] { dequant_shape_kernel<<<(unsigned) ((N_WARPS + 7) / 8), 256, 0, s>>>(in, o, N_WARPS); }, s);
    printf("K0 f16 copy 6.55MB->6.55MB: %8.1f us  (%.0f GB/s)\n", t0, 2.0 * N_OUT * 2 / 1e9 / (t0 * 1e-6));
    printf("K1 dequant shape (3.28MB in scattered, 6.55MB out): %8.1f us  (%.0f GB/s total traffic)\n",
           t1, (IN_BYTES + (size_t) N_OUT * 2) / 1e9 / (t1 * 1e-6));
    return 0;
}
