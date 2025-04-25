#include <cuda_runtime.h>

#include <vector>

#include "solve.h"

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

__global__ void vector_add(float* a, float* b, float* c, int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N - 3) {
    float4 reg_a = FLOAT4(a[idx]);
    float4 reg_b = FLOAT4(b[idx]);
    float4 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    FLOAT4(c[idx]) = reg_c;
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
