#include <cuda_runtime.h>

// Persistent Matmul: 고정 블록 수로 여러 행을 순회 처리
// A(M,K) @ B(K,N) = C(M,N)

__global__ void persistent_matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N,
    int NUM_SMS
) {
    int sm_pid = blockIdx.x;   // 블록 인덱스 (0 ~ NUM_SMS-1)
    int col_idx = threadIdx.x; // 열 인덱스 (0 ~ N-1)

    // Persistent 루프: 각 블록이 여러 행을 순회
    for (int row_idx = sm_pid; row_idx < M; row_idx += NUM_SMS) {
        if (col_idx < N) {
            float accumulator = 0.0f;

            // K 축 순회하며 내적 계산
            for (int k = 0; k < K; k++) {
                float a_val = A[row_idx * K + k];
                float b_val = B[k * N + col_idx];
                accumulator += a_val * b_val;
            }

            C[row_idx * N + col_idx] = accumulator;
        }
    }
}

extern "C" void day17_persistent_matmul(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    // SM 수 가져오기
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    int NUM_SMS = props.multiProcessorCount;
    if (NUM_SMS > M) NUM_SMS = M;

    // Persistent: 고정 블록 수만 런칭
    dim3 threadsPerBlock(N);     // 각 스레드가 한 열 담당
    dim3 blocksPerGrid(NUM_SMS); // SM 수만큼만 블록 런칭

    persistent_matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A, B, C, M, K, N, NUM_SMS
    );
    cudaDeviceSynchronize();
}
