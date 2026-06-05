#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <stdio.h>

// Use half precision (fp16)
#include <hip/hip_fp16.h>

// GEMM Kernel: C = A * B
// A: [M, K] = [1024, 128]
// B: [K, N] = [128, 2048]
// C: [M, N] = [1024, 2048]

__global__ void gemm_fp16_kernel(const __half* A, const __half* B, __half* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        __half acc = __half(0.0f);
        for (int k = 0; k < K; ++k) {
            __half a = A[row * K + k];
            __half b = B[k * N + col];
            acc = __half(__half2float(acc) + __half2float(a) * __half2float(b));
        }
        C[row * N + col] = acc;
    }
}

int main() {
    const int M = 1024;
    const int K = 128;
    const int N = 2048;

    size_t sizeA = M * K * sizeof(__half);
    size_t sizeB = K * N * sizeof(__half);
    size_t sizeC = M * N * sizeof(__half);

    std::vector<__half> h_A(M * K);
    std::vector<__half> h_B(K * N);
    std::vector<__half> h_C(M * N);

    // Initialize matrices
    for (int i = 0; i < M * K; ++i) h_A[i] = __half(1.0f / (float)K);
    for (int i = 0; i < K * N; ++i) h_B[i] = __half(1.0f / (float)K);

    __half* d_A, *d_B, *d_C;
    hipMalloc(&d_A, sizeA);
    hipMalloc(&d_B, sizeB);
    hipMalloc(&d_C, sizeC);

    hipMemcpy(d_A, h_A.data(), sizeA, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), sizeB, hipMemcpyHostToDevice);

    dim3 blockSize(16, 16);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    // hipLaunchKernelGGL is a statement macro: (kernel, grid, block, sharedMemBytes, stream, args...)
    hipEventRecord(start);
    hipLaunchKernelGGL(gemm_fp16_kernel, gridSize, blockSize, 0, 0, d_A, d_B, d_C, M, K, N);
    hipEventRecord(stop);

    hipError_t error = hipGetLastError();
    if (error != hipSuccess) {
        printf("Kernel launch failed: %s\n", hipGetErrorString(error));
        return -1;
    }
    error = hipDeviceSynchronize();
    if (error != hipSuccess) {
        printf("Kernel execution failed: %s\n", hipGetErrorString(error));
        return -1;
    }

    hipMemcpy(h_C.data(), d_C, sizeC, hipMemcpyDeviceToHost);

    std::cout << "Result C[0][0]: " << __half2float(h_C[0]) << std::endl;
    std::cout << "Result C[1023][2047]: " << __half2float(h_C[M * N - 1]) << std::endl;

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    return 0;
}