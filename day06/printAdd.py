import torch
import triton
import triton.language as tl


@triton.jit
def print_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    input = tl.load(input_ptr + idx, mask=mask)
    # Basic element-wise operation (identity for demonstration)
    tl.store(output_ptr + idx, input, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 256

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    print_kernel[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)

