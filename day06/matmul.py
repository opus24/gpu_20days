import torch
import triton
import triton.language as tl


@triton.jit
def matrix_multiplication_kernel(A_ptr, B_ptr, C_ptr, M, N, K, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
    # CUDA: inner_K = blockDim.x * blockIdx.x + threadIdx.x
    # CUDA: inner_M = blockDim.y * blockIdx.y + threadIdx.y
    pid_k = tl.program_id(0)  # corresponds to blockIdx.x
    pid_m = tl.program_id(1)  # corresponds to blockIdx.y
    
    k_idx = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)  # threadIdx.x
    m_idx = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # threadIdx.y
    
    mask_m = m_idx < M
    mask_k = k_idx < K
    
    # Initialize accumulator for each (m, k) pair
    accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float32)
    
    # Compute dot product: sum over N dimension
    for n in range(0, N):
        # Load A[inner_M, n] and B[n, inner_K]
        a = tl.load(A_ptr + m_idx[:, None] * N + n, mask=mask_m[:, None], other=0.0)
        b = tl.load(B_ptr + n * K + k_idx[None, :], mask=mask_k[None, :], other=0.0)
        accumulator += a * b
    
    # Store result: C[inner_M, inner_K]
    tl.store(C_ptr + m_idx[:, None] * K + k_idx[None, :], accumulator,
             mask=mask_m[:, None] & mask_k[None, :])


# A, B, C are tensors on the GPU
# A: M x N, B: N x K, C: M x K
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int):
    BLOCK_SIZE_K = 16
    BLOCK_SIZE_M = 16

    def grid(meta):
        return (
            triton.cdiv(K, meta["BLOCK_SIZE_K"]),
            triton.cdiv(M, meta["BLOCK_SIZE_M"]),
        )

    matrix_multiplication_kernel[grid](A, B, C, M, N, K, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_M=BLOCK_SIZE_M)

