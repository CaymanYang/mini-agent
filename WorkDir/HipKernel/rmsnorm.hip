#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h>

// RMSNorm Kernel
// Each block handles one row of the input matrix
// M: number of rows, N: number of elements per row (hidden dimension)
// Input: [M, N], Output: [M, N]
__global__ void rmsnorm_kernel(const float* input, float* output, float* gamma, float epsilon, int M, int N) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        // 1. Calculate sum of squares for the row
        // We use a simple reduction approach: each block calculates the sum for one row
        // To keep it simple for this task, we'll use a shared memory reduction
        __shared__ float shared_sq_sum[256]; // Adjust size based on blockDim.x

        // This is a simplified version where we assume blockDim.x <= 256
        // For a production kernel, we'd use a more robust reduction
        
        // Actually, to keep it simple and correct for a single kernel call:
        // We'll use a two-pass approach or a simpler one-pass if possible.
        // Let's do a simple one-pass where each block handles one row.
        // Each thread in the block contributes its square to the sum.
    }
}

// Let's rewrite to a more standard approach: 
// Each block handles one row. 
// Threads in the block compute the sum of squares.
__global__ void rmsnorm_kernel_v2(const float* input, float* output, float* gamma, float epsilon, int M, int N) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        // We need the sum of squares for the entire row
        // Since we want to avoid complex reductions for now, we'll use a simple 
        // approach: each block calculates the sum of squares for its row.
        // But we need that sum for every element in the row.
    }
}

// Let's go with a simpler design: 
// One block per row. 
// Threads in the block compute the sum of squares using shared memory.
__global__ void rmsnorm_kernel_v3(const float* input, float* output, float* gamma, float epsilon, int M, int N) {
    __shared__ float shared_sq_sum[1024]; // Adjust based on max block size

    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes the square of its element
    float val = input[row * N + col];
    shared_sq_sum[threadIdx.x] = val * val;

    __syncthreads();

    // Simple reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sq_sum[threadIdx.x] += shared_sq_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    float sum_sq = shared_sq_sum[0];
    float inv_rms = rsqrtf(sum_sq / N + epsilon);

    if (row < M && col < N) {
        output[row * N + col] = input[row * N + col] * inv_rms * gamma[col];
    }
}

int main() {
    const int M = 128; // Batch size
    const int N = 512; // Hidden dimension
    const float epsilon = 1e-5f;

    size_t sizeInput = M * N * sizeof(float);
    size_t sizeOutput = M * N * sizeof(float);
    size_t sizeGamma = N * sizeof(float);

    std::vector<float> h_input(M * N);
    std::vector<float> h_output(M * N);
    std::vector<float> h_gamma(N, 1.0f);

    // Initialize input with some values
    for (int i = 0; i < M * N; ++i) {
        h_input[i] = (float)(rand() % 100) / 10.0f;
    }

    float *d_input, *d_output, *d_gamma;
    hipMalloc(&d_input, sizeInput);
    hipMalloc(&d_output, sizeOutput);
    hipMalloc(&d_gamma, sizeGamma);

    hipMemcpy(d_input, h_input.data(), sizeInput, hipMemcpyHostToDevice);
    hipMemcpy(d_gamma, h_gamma.data(), sizeGamma, hipMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, M);

    rmsnorm_kernel_v3<<<gridSize, blockSize>>>(d_input, d_output, d_gamma, epsilon, M, N);

    hipDeviceSynchronize();

    hipMemcpy(h_output.data(), d_output, sizeOutput, hipMemcpyDeviceToHost);

    printf("Input[0][0]: %.4f, Output[0][0]: %.4f\n", h_input[0], h_output[0]);
    printf("Input[0][N-1]: %.4f, Output[0][N-1]: %.4f\n", h_input[N-1], h_output[N-1]);

    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_gamma);

    return 0;
}
