#include <iostream>
#include <cuda_runtime.h>

#define ceil(x,y) (((x) + (y) - 1) / (y))
#define BLOCK_SIZE 256

__device__ int add(int a, int b) {
    return a + b;
}

__global__ void functionKernel(int *h_A, int *h_B, int N) {
    int idx = blockIdx.x * blockDim.x
            + threadIdx.x;

    if (idx < N) {
        h_B[idx] = add(h_A[idx], h_A[idx]);
    }
}

int main() {
    const int N = 1024;
    const int size = N * sizeof(int);

    // Allocate host memory
    int *h_A = (int*)malloc(size);
    int *h_B = (int*)malloc(size);
    for (int i = 0; i < N; i++) h_A[i] = i + 1;

    // Allocate device memory
    int *d_A, *d_B;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid(ceil(N, BLOCK_SIZE));
    functionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    printf("h_B: ");
    for (int i = 0; i < N; i++) printf("%d ", h_B[i]);
    printf("\n");

    return 0;
}
