from task import input_t, output_t
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_stages': 5, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 5, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 5, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_stages': 6, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
    ],
    key=['M', 'H', 'K'],
)
@triton.jit
def _swiglu_fused_kernel(
    # Pointers
    X_ptr,      # [M, K]
    W_ptr,      # [K, H]
    V_ptr,      # [K, H]
    B_ptr,      # [H]
    C_ptr,      # [H]
    Y_ptr,      # [M, H]
    # Dimensions
    M,          # batch_size * seq_len
    H,          # hidden_size
    K,          # in_features
    # Strides
    stride_xm, stride_xk,
    stride_wk, stride_wh,
    stride_vk, stride_vh,
    stride_ym, stride_yh,
    # Swish parameter
    beta,
    # Block sizes (constexpr for compilation)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused SwiGLU kernel: computes Swish(xW + b) * (xV + c)
    Both matmuls are computed in a single pass over K to maximize data reuse.
    Uses fp16 for tensor core operations with fp32 accumulation.
    """
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Masks for bounds checking
    mask_m = offs_m < M
    mask_n = offs_n < H

    # Initialize accumulators in fp32 for numerical precision
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_value = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Base pointers for X, W, V
    X_block_ptr = X_ptr + offs_m[:, None] * stride_xm
    W_block_ptr = W_ptr + offs_n[None, :] * stride_wh
    V_block_ptr = V_ptr + offs_n[None, :] * stride_vh

    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        mask_k = k_offs < K

        # Load X tile [BLOCK_M, BLOCK_K]
        x_tile = tl.load(
            X_block_ptr + k_offs[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        )

        # Load W tile [BLOCK_K, BLOCK_N]
        w_tile = tl.load(
            W_block_ptr + k_offs[:, None] * stride_wk,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0
        )

        # Load V tile [BLOCK_K, BLOCK_N]
        v_tile = tl.load(
            V_block_ptr + k_offs[:, None] * stride_vk,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0
        )

        # Convert to fp16 for tensor core operations
        x_fp16 = x_tile.to(tl.float16)
        w_fp16 = w_tile.to(tl.float16)
        v_fp16 = v_tile.to(tl.float16)

        # Compute both dot products using tensor cores with fp32 accumulation
        acc_gate += tl.dot(x_fp16, w_fp16, out_dtype=tl.float32)
        acc_value += tl.dot(x_fp16, v_fp16, out_dtype=tl.float32)

    # Load biases
    bias_b = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    bias_c = tl.load(C_ptr + offs_n, mask=mask_n, other=0.0)

    # Add biases: gate = xW + b, value = xV + c
    gate = acc_gate + bias_b[None, :]
    value = acc_value + bias_c[None, :]

    # Swish activation: gate * sigmoid(beta * gate)
    swish_gate = gate * tl.sigmoid(beta * gate)

    # Final output: Swish(gate) * value
    output = swish_gate * value

    # Store result
    Y_block_ptr = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yh
    tl.store(
        Y_block_ptr,
        output,
        mask=mask_m[:, None] & mask_n[None, :]
    )


def custom_kernel(data: input_t) -> output_t:
    """
    Triton implementation of SwiGLU activation function.
    SwiGLU(x, W, V, b, c, beta) = Swish(xW + b) ⊙ (xV + c)
    where Swish(z) = z * sigmoid(beta * z)

    Args:
        data: tuple of (x, W, V, b, c, beta) where:
            x: input tensor of shape (batch_size, seq_len, in_features)
            W: weight matrix of shape (in_features, hidden_size)
            V: weight matrix of shape (in_features, hidden_size)
            b: bias vector of shape (hidden_size,)
            c: bias vector of shape (hidden_size,)
            beta: scalar value for Swish activation
    Returns:
        Output tensor of shape (batch_size, seq_len, hidden_size)
    """
    x, W, V, b, c, beta = data

    # Get dimensions
    B, S, In = x.shape
    H = W.shape[1]
    M = B * S  # Total number of rows
    K = In     # Input features

    # Reshape x to 2D: [M, K]
    x_2d = x.reshape(M, K)

    # Ensure inputs are contiguous
    x_2d = x_2d.contiguous()
    W = W.contiguous()
    V = V.contiguous()
    b = b.contiguous()
    c = c.contiguous()

    # Allocate output tensor
    y = torch.empty((M, H), device=x.device, dtype=torch.float32)

    # Get strides
    stride_xm, stride_xk = x_2d.stride()
    stride_wk, stride_wh = W.stride()
    stride_vk, stride_vh = V.stride()
    stride_ym, stride_yh = y.stride()

    # Define grid
    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(H, META['BLOCK_N']),
        )

    # Launch kernel
    _swiglu_fused_kernel[grid](
        x_2d, W, V, b, c, y,
        M, H, K,
        stride_xm, stride_xk,
        stride_wk, stride_wh,
        stride_vk, stride_vh,
        stride_ym, stride_yh,
        beta,
    )

    # Reshape output to 3D
    return y.reshape(B, S, H)
