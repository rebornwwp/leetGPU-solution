#include <cuda_runtime.h>

#include <vector>

#include "solve.h"

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

__global__ void vector_add(float* a, float* b, float* c, int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  int stride = 4 * blockDim.x * gridDim.x;

  for (int i = idx; i + 3 < N; i += stride) {
    float4 a4 = FLOAT4(a[i]);
    float4 b4 = FLOAT4(b[i]);
    float4 c4;
    c4.x = a4.x + b4.x;
    c4.y = a4.y + b4.y;
    c4.z = a4.z + b4.z;
    c4.w = a4.w + b4.w;
    FLOAT4(c[i]) = c4;
  }

  // optional: scalar tail处理 N 不被4整除的情况
  // 只让一个线程做 tail，避免重复
  if (idx == 0) {
    int tail_start = (N / 4) * 4;
#pragma unroll
    for (int i = tail_start; i < N; ++i) {
      c[i] = a[i] + b[i];
    }
  }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int N) {
  // 需要更细致的分配线程
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + 4 * threadsPerBlock - 1) / (4 * threadsPerBlock);

  vector_add<<<blocksPerGrid, threadsPerBlock>>>((float*)A, (float*)B, C, N);
  cudaDeviceSynchronize();
}
