// Stage 1.15: custom GU GEMM microbench.
// GU GEMM: Y[ne][1280] = X[ne][2560] * W[1280][2560]^T  (f16 in, f32 out)
// W is [1280][2560] f16 row-major (6.55 MB). X is [ne][2560] f16. Y is [ne][1280] f32.
// Memory-bound on the W read (roofline 13.8 us at 474 GB/s). cuBLAS ~26.6 us (ne=8).
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <vector>
#include <algorithm>

#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 1;}}while(0)

const int M = 1280;   // GU output rows
const int K = 2560;   // GU input dim

// Kernel v1: one W row per block, 128 threads, K/128=20 f16 per thread.
template<int MAXNE>
__global__ void gu_gemm_v1(const __half* __restrict__ W, const __half* __restrict__ X, float* __restrict__ Y, int ne, int M, int K){
  const int i = blockIdx.x;
  if (i >= M) return;
  const int tid = threadIdx.x;           // 0..127
  constexpr int KPT = 20;
  const int k0 = tid * KPT;
  const __half* Wi = W + (int64_t)i * K + k0;
  float w[KPT];
  #pragma unroll
  for (int k = 0; k < KPT; k += 2){ const __half2 h = *(const __half2*)(Wi+k); w[k]=__low2float(h); w[k+1]=__high2float(h); }
  float partial[MAXNE];
  for (int t = 0; t < ne; t++){
    const __half* Xt = X + (int64_t)t * K + k0;
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < KPT; k += 2){ const __half2 h = *(const __half2*)(Xt+k); acc += w[k]*__low2float(h) + w[k+1]*__high2float(h); }
    partial[t] = acc;
  }
  __shared__ float sr[MAXNE][128];
  for (int t = 0; t < ne; t++) sr[t][tid] = partial[t];
  __syncthreads();
  // 2-stage reduce: 4 warps reduce 32 -> 4 partial sums, then warp0 reduces 4.
  for (int t = 0; t < ne; t++){
    float v = sr[t][tid];
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    if ((tid & 31) == 0) sr[t][tid>>5] = v;
  }
  __syncthreads();
  if (tid < 32){
    for (int t = 0; t < ne; t++){
      float v = (tid < 4) ? sr[t][tid] : 0.f;
      for (int off = 2; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
      if (tid == 0) Y[(int64_t)t * M + i] = v;
    }
  }
}

// Kernel v2: one W row per warp (32 threads), K/32=80 f16 per thread. Higher ILP, no block reduce.
template<int MAXNE>
__global__ void gu_gemm_v2(const __half* __restrict__ W, const __half* __restrict__ X, float* __restrict__ Y, int ne, int M, int K){
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= M) return;
  constexpr int KPT = 80;
  const int k0 = lane * KPT;
  const __half* Wi = W + (int64_t)warp * K + k0;
  float w[KPT];
  #pragma unroll
  for (int k = 0; k < KPT; k += 2){ const __half2 h = *(const __half2*)(Wi+k); w[k]=__low2float(h); w[k+1]=__high2float(h); }
  float partial[MAXNE];
  for (int t = 0; t < ne; t++){
    const __half* Xt = X + (int64_t)t * K + k0;
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < KPT; k += 2){ const __half2 h = *(const __half2*)(Xt+k); acc += w[k]*__low2float(h) + w[k+1]*__high2float(h); }
    partial[t] = acc;
  }
  for (int t = 0; t < ne; t++){
    float v = partial[t];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    if (lane == 0) Y[(int64_t)t * M + warp] = v;
  }
}


// v3: block-based, X in shared memory (loaded once per block, shared across R W rows).
// R W rows per block, each warp handles one W row. W read from DRAM (bottleneck).
template<int R, int MAXNE>
__global__ void gu_gemm_v3(const __half* __restrict__ W, const __half* __restrict__ X, float* __restrict__ Y, int ne, int M, int K){
  extern __shared__ __half xs[];   // ne*K f16
  for (int idx = threadIdx.x; idx < ne * K; idx += blockDim.x) xs[idx] = X[idx];
  __syncthreads();
  const int warp = threadIdx.x >> 5;        // 0..R-1
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * R + warp;
  if (row >= M) return;
  constexpr int KPT = 80;                   // K/32
  const int k0 = lane * KPT;
  const __half* Wi = W + (int64_t)row * K + k0;
  float w[KPT];
  #pragma unroll
  for (int k = 0; k < KPT; k += 2){ const __half2 h = *(const __half2*)(Wi+k); w[k]=__low2float(h); w[k+1]=__high2float(h); }
  float partial[MAXNE];
  for (int t = 0; t < ne; t++){
    const __half* Xt = xs + (int64_t)t * K + k0;
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < KPT; k += 2){ const __half2 h = *(const __half2*)(Xt+k); acc += w[k]*__low2float(h) + w[k+1]*__high2float(h); }
    partial[t] = acc;
  }
  for (int t = 0; t < ne; t++){
    float v = partial[t];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    if (lane == 0) Y[(int64_t)t * M + row] = v;
  }
}

double time_us(int reps, std::function<void()> f){
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (int i=0;i<10;i++) f();
  cudaDeviceSynchronize();
  std::vector<double> t(reps);
  for (int i=0;i<reps;i++){
    cudaEventRecord(a); f(); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms,a,b); t[i]=ms*1000.0;
  }
  std::sort(t.begin(),t.end());
  return t[reps/2];
}

int main(int argc, char** argv){
  const int maxne = 32;
  __half *W, *X; float *Y, *Yref;
  CK(cudaMalloc(&W, (int64_t)M*K*2));
  CK(cudaMalloc(&X, (int64_t)maxne*K*2));
  CK(cudaMalloc(&Y, (int64_t)maxne*M*4));
  CK(cudaMalloc(&Yref, (int64_t)maxne*M*4));
  // fill
  {
    std::vector<__half> hW((int64_t)M*K), hX((int64_t)maxne*K);
    for (auto& h : hW) h = __float2half(0.01f * (rand()%1000 - 500) / 500.f);
    for (auto& h : hX) h = __float2half(0.01f * (rand()%1000 - 500) / 500.f);
    CK(cudaMemcpy(W, hW.data(), hW.size()*2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(X, hX.data(), hX.size()*2, cudaMemcpyHostToDevice));
  }
  cublasHandle_t h; cublasCreate(&h);
  void* ws; CK(cudaMalloc(&ws, 32<<20)); cublasSetWorkspace(h, ws, 32<<20);
  const float alpha=1.f, beta=0.f;

  // reference: cuBLAS
  auto cublas_gemm = [&](int ne){
    // Y[ne][M] (row-major) = X[ne][K] * W[M][K]^T
    // col-major: Y^T[M][ne] = W[M][K] * (X^T)[K][ne]
    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, ne, K,
      &alpha, W, CUDA_R_16F, K, X, CUDA_R_16F, K, &beta, Yref, CUDA_R_32F, M,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  };

  printf("%5s %12s %12s %12s %12s %12s\n", "ne","cublas","v1","v2","v3 R8","v3 R16"); fflush(stdout);
  for (int ne : {1,2,4,8,12,16,24,32}){
    double t_cublas = time_us(60, [&]{ cublas_gemm(ne); });
    double t_v1 = time_us(60, [&]{ gu_gemm_v1<32><<<M,128>>>(W,X,Y,ne,M,K); });
    double t_v2 = time_us(60, [&]{ gu_gemm_v2<32><<<M/8, 256>>>(W,X,Y,ne,M,K); });
    double t_v3 = 0, t_v3b = 0;
    if (ne <= 16){
      int smem = ne*K*2;
      cudaFuncSetAttribute(gu_gemm_v3<8,16>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96*1024);
      t_v3 = time_us(60, [&]{ gu_gemm_v3<8,16><<<M/8, 256, smem>>>(W,X,Y,ne,M,K); });
      cudaFuncSetAttribute(gu_gemm_v3<16,16>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96*1024);
      t_v3b = time_us(60, [&]{ gu_gemm_v3<16,16><<<M/16, 512, smem>>>(W,X,Y,ne,M,K); });
    }
    // correctness: compare v2 vs cublas
    std::vector<float> yref((int64_t)ne*M), y((int64_t)ne*M);
    CK(cudaMemcpy(yref.data(), Yref, yref.size()*4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y.data(), Y, y.size()*4, cudaMemcpyDeviceToHost));
    double maxerr = 0;
    for (size_t i=0;i<y.size();i++) maxerr = std::max(maxerr, (double)std::abs(y[i]-yref[i]));
    fprintf(stderr,"ne=%d t_c=%d t_v1=%d t_v2=%d\n", ne, (int)t_cublas,(int)t_v1,(int)t_v2); fflush(stderr);
    printf("%5d %12.2f %12.2f %12.2f %12.2f %12.2f  (err=%.2g)\n", ne, t_cublas, t_v1, t_v2, t_v3, t_v3b, maxerr); fflush(stdout);
  }
  printf("roofline (6.55MB @474GB/s) = 13.8 us\n");
  return 0;
}
