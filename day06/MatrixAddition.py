import torch
import triton
import triton.language as tl


@triton.jit
def matrix_add_kernel(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    # CUDA: row = blockIdx.x, col = blockDim.x * blockIdx.y + threadIdx.x
    pid_x = tl.program_id(0)  # blockIdx.x -> row
    pid_y = tl.program_id(1)  # blockIdx.y
    
    row = pid_x
    col = pid_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # threadIdx.x
    
    mask = (row < N) & (col < N)
    
    # Calculate index: idx = col + row * N
    idx = col + row * N
    
    A = tl.load(A_ptr + idx, mask=mask)
    B = tl.load(B_ptr + idx, mask=mask)
    C = A + B
    tl.store(C_ptr + idx, C, mask=mask)


# A, B, C are tensors on the GPU (N x N matrices)
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    BLOCK_SIZE = 256

    def grid(meta):
        return (N, triton.cdiv(N, meta["BLOCK_SIZE"]))

    matrix_add_kernel[grid](A, B, C, N, BLOCK_SIZE=BLOCK_SIZE)

