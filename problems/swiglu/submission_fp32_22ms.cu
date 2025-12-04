#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>

static cublasHandle_t g_cublas_handle = nullptr;

inline void ensure_cublas_handle() {
    if (g_cublas_handle == nullptr) {
        cublasStatus_t status = cublasCreate(&g_cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLAS handle");
        }
        cublasSetMathMode(g_cublas_handle, CUBLAS_DEFAULT_MATH);
    }
}

__global__ void fused_swiglu_activation_kernel(
    const float* __restrict__ gate,    // [M, H] - result of x @ W
    const float* __restrict__ value,   // [M, H] - result of x @ V
    const float* __restrict__ b,       // [H]
    const float* __restrict__ c,       // [H]
    float beta,
    int M,
    int H,
    float* __restrict__ output         // [M, H]
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * H;
    
    if (idx >= total) return;
    
    const int h = idx % H;
    
    float g = gate[idx] + b[h];
    float v = value[idx] + c[h];
    
    float swish_g = g / (1.0f + expf(-beta * g));
    
    output[idx] = swish_g * v;
}

__global__ void fused_swiglu_activation_kernel_vec4(
    const float* __restrict__ gate,    // [M, H] - result of x @ W
    const float* __restrict__ value,   // [M, H] - result of x @ V
    const float* __restrict__ b,       // [H]
    const float* __restrict__ c,       // [H]
    float beta,
    int M,
    int H,
    float* __restrict__ output         // [M, H]
) {
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int total = M * H;
    
    if (idx + 3 >= total) {
        for (int i = idx; i < total && i < idx + 4; i++) {
            int h = i % H;
            float g = gate[i] + b[h];
            float v = value[i] + c[h];
            float swish_g = g / (1.0f + expf(-beta * g));
            output[i] = swish_g * v;
        }
        return;
    }
    
    float4 g_vec = *reinterpret_cast<const float4*>(&gate[idx]);
    float4 v_vec = *reinterpret_cast<const float4*>(&value[idx]);
    
    int h0 = idx % H;
    int h1 = (idx + 1) % H;
    int h2 = (idx + 2) % H;
    int h3 = (idx + 3) % H;
    
    g_vec.x += b[h0]; v_vec.x += c[h0];
    g_vec.y += b[h1]; v_vec.y += c[h1];
    g_vec.z += b[h2]; v_vec.z += c[h2];
    g_vec.w += b[h3]; v_vec.w += c[h3];
    
    float4 out_vec;
    out_vec.x = (g_vec.x / (1.0f + expf(-beta * g_vec.x))) * v_vec.x;
    out_vec.y = (g_vec.y / (1.0f + expf(-beta * g_vec.y))) * v_vec.y;
    out_vec.z = (g_vec.z / (1.0f + expf(-beta * g_vec.z))) * v_vec.z;
    out_vec.w = (g_vec.w / (1.0f + expf(-beta * g_vec.w))) * v_vec.w;
    
    *reinterpret_cast<float4*>(&output[idx]) = out_vec;
}

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
    
    // Get dimensions
    const int64_t B = x.size(0);
    const int64_t S = x.size(1);
    const int64_t In = x.size(2);
    const int64_t H = W.size(1);
    const int64_t M = B * S;
    
    auto x_contig = x.contiguous().to(torch::kFloat32);
    auto W_contig = W.contiguous().to(torch::kFloat32);
    auto V_contig = V.contiguous().to(torch::kFloat32);
    auto b_contig = b.contiguous().to(torch::kFloat32);
    auto c_contig = c.contiguous().to(torch::kFloat32);
    
    auto x_2d = x_contig.view({M, In});
    
    auto gate = torch::empty({M, H}, x.options().dtype(torch::kFloat32));
    auto value = torch::empty({M, H}, x.options().dtype(torch::kFloat32));
    auto output = torch::empty({M, H}, x.options().dtype(torch::kFloat32));
    
    ensure_cublas_handle();
    
    cublasSetStream(g_cublas_handle, at::cuda::getCurrentCUDAStream());
    
    float alpha = 1.0f;
    float zero = 0.0f;
    
    cublasStatus_t status;
    
    status = cublasSgemm(
        g_cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        static_cast<int>(H),
        static_cast<int>(M),
        static_cast<int>(In),
        &alpha,
        W_contig.data_ptr<float>(), static_cast<int>(H),
        x_2d.data_ptr<float>(), static_cast<int>(In),
        &zero,
        gate.data_ptr<float>(), static_cast<int>(H)
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS GEMM failed for gate computation");
    }
    
    status = cublasSgemm(
        g_cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        static_cast<int>(H),
        static_cast<int>(M),
        static_cast<int>(In),
        &alpha,
        V_contig.data_ptr<float>(), static_cast<int>(H),
        x_2d.data_ptr<float>(), static_cast<int>(In),
        &zero,
        value.data_ptr<float>(), static_cast<int>(H)
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS GEMM failed for value computation");
    }
    
    const int total_elements = static_cast<int>(M * H);
    const int threads_per_block = 256;
    
    if (H % 4 == 0) {
        const int num_vec4 = total_elements / 4;
        const int blocks = (num_vec4 + threads_per_block - 1) / threads_per_block;
        
        fused_swiglu_activation_kernel_vec4<<<blocks, threads_per_block>>>(
            gate.data_ptr<float>(),
            value.data_ptr<float>(),
            b_contig.data_ptr<float>(),
            c_contig.data_ptr<float>(),
            beta,
            static_cast<int>(M),
            static_cast<int>(H),
            output.data_ptr<float>()
        );
    } else {
        const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        
        fused_swiglu_activation_kernel<<<blocks, threads_per_block>>>(
            gate.data_ptr<float>(),
            value.data_ptr<float>(),
            b_contig.data_ptr<float>(),
            c_contig.data_ptr<float>(),
            beta,
            static_cast<int>(M),
            static_cast<int>(H),
            output.data_ptr<float>()
        );
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output.view({B, S, H});
}
