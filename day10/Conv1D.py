import torch
import triton
import triton.language as tl


@triton.jit
def convolution_1d_kernel(input_ptr, kernel_ptr, output_ptr, input_size, kernel_size, BLOCK_SIZE: tl.constexpr):
    output_size = input_size - kernel_size + 1
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < output_size
    
    # Initialize sum
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Compute convolution: sum over kernel_size
    for i in range(kernel_size):
        input_idx = idx + i
        input_mask = (input_idx < input_size) & mask
        input_val = tl.load(input_ptr + input_idx, mask=input_mask, other=0.0)
        kernel_val = tl.load(kernel_ptr + i)
        sum_val += input_val * kernel_val
    
    tl.store(output_ptr + idx, sum_val, mask=mask)


# input, kernel, output are tensors on the GPU
def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, input_size: int, kernel_size: int):
    BLOCK_SIZE = 256
    output_size = input_size - kernel_size + 1

    def grid(meta):
        return (triton.cdiv(output_size, meta["BLOCK_SIZE"]),)

    convolution_1d_kernel[grid](input, kernel, output, input_size, kernel_size, BLOCK_SIZE=BLOCK_SIZE)

