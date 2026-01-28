"""
Day 17: Persistent Matmul
"""

import torch
import triton
import triton.language as tl


@triton.jit
def day17_persistent_matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    K,
    N,  # A(M,K) @ B(K,N) = C(M,N)
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    sm_pid = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    col_mask = col_offsets < N

    for row_idx in range(sm_pid, M, NUM_SMS):
        accumulator = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
        for k in range(K):
            a_val = tl.load(A_ptr + row_idx * K + k)
            b_vals = tl.load(B_ptr + k * N + col_offsets, mask=col_mask, other=0.0)
            accumulator += a_val * b_vals
        tl.store(C_ptr + row_idx * N + col_offsets, accumulator, mask=col_mask)


def day17_persistent_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Day 17: Persistent matrix multiplication"""
    M, K = A.shape
    _, N = B.shape

    C = torch.zeros(M, N, device=A.device, dtype=A.dtype)
    # 256 단위로 연산되므로 2의 거듭제곱 형태
    BLOCK_SIZE_N = triton.next_power_of_2(N)
    # 런칭할 블록이 SM수보다 작아야 함
    NUM_SMS = min(M, torch.cuda.get_device_properties(0).multi_processor_count)

    day17_persistent_matmul_kernel[(NUM_SMS,)](
        A,
        B,
        C,
        M,
        K,
        N,
        NUM_SMS=NUM_SMS,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return C
