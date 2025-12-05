// submission.cu
// Naive CUDA implementation template for the SwiGLU kernel.
//
// SwiGLU(x, W, V, b, c, beta) = Swish(xW + b) ⊙ (xV + c)
//   where Swish(z) = z * sigmoid(beta * z)
//
// Shapes (see reference.py for details):
//   x: [B, S, In]          (batch_size, seq_len, in_features)
//   W: [In, H]             (in_features, hidden_size)
//   V: [In, H]
//   b: [H]
//   c: [H]
//   output: [B, S, H]

#include <torch/extension.h>
#include <cuda_runtime.h>

// Simple sigmoid helper operating in float
__device__ inline float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Naive SwiGLU kernel.
// Each thread computes one output element (b, s, h).
template <typename scalar_t>
__global__ void kernel_body(
    const scalar_t* __restrict__ x,   // [B, S, In]
    const scalar_t* __restrict__ W,   // [In, H]
    const scalar_t* __restrict__ V,   // [In, H]
    const scalar_t* __restrict__ b,   // [H]
    const scalar_t* __restrict__ c,   // [H]
    float beta,
    int B,
    int S,
    int In,
    int H,
    scalar_t* __restrict__ output     // [B, S, H]
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;   // hidden dim
    int m = blockIdx.y * blockDim.y + threadIdx.y;   // flattened (B * S)

    int M = B * S;
    if (h >= H || m >= M) {
        return;
    }

    int b_idx = m / S;
    int s_idx = m % S;

    // Pointers to the input row (over In) for this (b, s)
    const scalar_t* x_row = x + ( (b_idx * S + s_idx) * In );

    // Compute gate = x_row @ W[:, h] + b[h]
    // and value = x_row @ V[:, h] + c[h]
    float gate = 0.0f;
    float value = 0.0f;
    for (int i = 0; i < In; ++i) {
        float xv = static_cast<float>(x_row[i]);
        float wv = static_cast<float>(W[i * H + h]);
        float vv = static_cast<float>(V[i * H + h]);
        gate  += xv * wv;
        value += xv * vv;
    }

    gate  += static_cast<float>(b[h]);
    value += static_cast<float>(c[h]);

    // Swish(gate) = gate * sigmoid(beta * gate)
    float swish_gate = gate * sigmoidf(beta * gate);
    float out_val = swish_gate * value;

    // Store result
    output[(b_idx * S + s_idx) * H + h] = static_cast<scalar_t>(out_val);
}

// Required entry point: called from Python via wrap_cuda_submission.py
// Signature must match:
//   torch::Tensor custom_kernel(torch::Tensor, torch::Tensor,
//                               torch::Tensor, torch::Tensor,
//                               torch::Tensor, float)
torch::Tensor custom_kernel(
    torch::Tensor x,   // [B, S, In]
    torch::Tensor W,   // [In, H]
    torch::Tensor V,   // [In, H]
    torch::Tensor b,   // [H]
    torch::Tensor c,   // [H]
    float beta
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(W.is_cuda(), "W must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(c.is_cuda(), "c must be a CUDA tensor");

    TORCH_CHECK(x.dim() == 3, "x must have shape [B, S, In]");
    TORCH_CHECK(W.dim() == 2, "W must have shape [In, H]");
    TORCH_CHECK(V.dim() == 2, "V must have shape [In, H]");
    TORCH_CHECK(b.dim() == 1, "b must have shape [H]");
    TORCH_CHECK(c.dim() == 1, "c must have shape [H]");

    const int64_t B = x.size(0);
    const int64_t S = x.size(1);
    const int64_t In = x.size(2);
    const int64_t H = W.size(1);

    TORCH_CHECK(W.size(0) == In, "W.shape[0] must equal x.shape[2]");
    TORCH_CHECK(V.size(0) == In, "V.shape[0] must equal x.shape[2]");
    TORCH_CHECK(V.size(1) == H,  "V.shape[1] must equal W.shape[1]");
    TORCH_CHECK(b.size(0) == H,  "b.shape[0] must equal H");
    TORCH_CHECK(c.size(0) == H,  "c.shape[0] must equal H");

    auto x_contig = x.contiguous();
    auto W_contig = W.contiguous();
    auto V_contig = V.contiguous();
    auto b_contig = b.contiguous();
    auto c_contig = c.contiguous();

    // Output shape: [B, S, H]
    auto output = torch::empty({B, S, H}, x.options());

    const int threads_x = 16;
    const int threads_y = 16;
    dim3 threads(threads_x, threads_y);
    dim3 blocks(
        (H + threads_x - 1) / threads_x,
        (B * S + threads_y - 1) / threads_y
    );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        x_contig.scalar_type(), "swiglu_kernel", ([&] {
            kernel_body<scalar_t><<<blocks, threads>>>(
                x_contig.data_ptr<scalar_t>(),
                W_contig.data_ptr<scalar_t>(),
                V_contig.data_ptr<scalar_t>(),
                b_contig.data_ptr<scalar_t>(),
                c_contig.data_ptr<scalar_t>(),
                beta,
                static_cast<int>(B),
                static_cast<int>(S),
                static_cast<int>(In),
                static_cast<int>(H),
                output.data_ptr<scalar_t>()
            );
        })
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    // For debugging / simplicity; you can remove this for performance
    cudaDeviceSynchronize();

    return output;
}


