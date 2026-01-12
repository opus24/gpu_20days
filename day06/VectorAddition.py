import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    A = tl.load(A_ptr + idx, mask=mask)
    B = tl.load(B_ptr + idx, mask=mask)
    C = A + B
    tl.store(C_ptr + idx, C, mask=mask)


# A, B, C are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    BLOCK_SIZE = 256

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    vector_add_kernel[grid](A, B, C, N, BLOCK_SIZE=BLOCK_SIZE)

