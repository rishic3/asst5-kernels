#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <stdexcept>

static cublasLtHandle_t g_cublasLt_handle = nullptr;

inline void ensure_cublasLt_handle() {
    if (g_cublasLt_handle == nullptr) {
        cublasStatus_t status = cublasLtCreate(&g_cublasLt_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("failed to create cublas handle");
        }
    }
}

__global__ void fused_swiglu_activation_fp16_kernel(
    const half* __restrict__ gate, // M x H
    const half* __restrict__ value, // M x H
    const half* __restrict__ b, // H
    const half* __restrict__ c_bias, // H
    float beta,
    int M,
    int H,
    half* __restrict__ output // M x H
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * H;
    
    if (idx >= total) return;
    
    const int h = idx % H;

    float g = __half2float(gate[idx]) + __half2float(b[h]);
    float v = __half2float(value[idx]) + __half2float(c_bias[h]);

    float swish_g = g / (1.0f + expf(-beta * g));
    
    output[idx] = __float2half(swish_g * v);
}

__global__ void fused_swiglu_activation_fp16_vec2_kernel(
    const half* __restrict__ gate, // M x H
    const half* __restrict__ value, // M x H
    const half* __restrict__ b, // H
    const half* __restrict__ c_bias, // H
    float beta,
    int M,
    int H,
    half* __restrict__ output // M x H
) {
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int total = M * H;
    
    if (idx + 1 >= total) {
        // edge case
        if (idx < total) {
            int h = idx % H;
            float g = __half2float(gate[idx]) + __half2float(b[h]);
            float v = __half2float(value[idx]) + __half2float(c_bias[h]);
            float swish_g = g / (1.0f + expf(-beta * g));
            output[idx] = __float2half(swish_g * v);
        }
        return;
    }
    
    half2 g_vec = *reinterpret_cast<const half2*>(&gate[idx]);
    half2 v_vec = *reinterpret_cast<const half2*>(&value[idx]);
    
    int h0 = idx % H;
    int h1 = (idx + 1) % H;
    
    half2 b_vec = make_half2(b[h0], b[h1]);
    half2 c_vec = make_half2(c_bias[h0], c_bias[h1]);
    
    float g0 = __half2float(g_vec.x) + __half2float(b_vec.x);
    float g1 = __half2float(g_vec.y) + __half2float(b_vec.y);
    float v0 = __half2float(v_vec.x) + __half2float(c_vec.x);
    float v1 = __half2float(v_vec.y) + __half2float(c_vec.y);

    float swish0 = g0 / (1.0f + expf(-beta * g0));
    float swish1 = g1 / (1.0f + expf(-beta * g1));

    half2 out_vec = make_half2(__float2half(swish0 * v0), __float2half(swish1 * v1));
    *reinterpret_cast<half2*>(&output[idx]) = out_vec;
}


void cublasLt_gemm_fp16(
    cublasLtHandle_t handle,
    cudaStream_t stream,
    int M, int N, int K,
    const half* A,
    const half* B,
    half* C,
    void* workspace,
    size_t workspaceSize
) {
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;

    cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));
    
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, N, K, N);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, K, M, K);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, N, M, N);
    
    float alpha = 1.0f;
    float zero = 0.0f;
    
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulPreferenceCreate(&preference);
    cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, 
                                          &workspaceSize, sizeof(workspaceSize));
    
    cublasLtMatmulHeuristicResult_t heuristicResult;
    int returnedResults;
    cublasLtMatmulAlgoGetHeuristic(handle, matmulDesc, Adesc, Bdesc, Cdesc, Cdesc,
                                    preference, 1, &heuristicResult, &returnedResults);
    
    cublasStatus_t status = cublasLtMatmul(
        handle,
        matmulDesc,
        &alpha,
        A, Adesc,
        B, Bdesc,
        &zero,
        C, Cdesc,
        C, Cdesc,
        &heuristicResult.algo,
        workspace,
        heuristicResult.workspaceSize,
        stream
    );
    
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLASLt FP16 GEMM failed");
    }
}

torch::Tensor custom_kernel(
    torch::Tensor x,
    torch::Tensor W,
    torch::Tensor V,
    torch::Tensor b,
    torch::Tensor c,
    float beta
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(W.is_cuda(), "W must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(c.is_cuda(), "c must be a CUDA tensor");
    
    const int64_t B = x.size(0);
    const int64_t S = x.size(1);
    const int64_t In = x.size(2);
    const int64_t H = W.size(1);
    const int64_t M = B * S;
    
    auto x_fp16 = x.contiguous();
    if (x_fp16.dtype() != torch::kFloat16) {
        x_fp16 = x_fp16.to(torch::kFloat16);
    }
    auto W_fp16 = W.contiguous();
    if (W_fp16.dtype() != torch::kFloat16) {
        W_fp16 = W_fp16.to(torch::kFloat16);
    }
    auto V_fp16 = V.contiguous();
    if (V_fp16.dtype() != torch::kFloat16) {
        V_fp16 = V_fp16.to(torch::kFloat16);
    }
    auto b_fp16 = b.contiguous();
    if (b_fp16.dtype() != torch::kFloat16) {
        b_fp16 = b_fp16.to(torch::kFloat16);
    }
    auto c_fp16 = c.contiguous();
    if (c_fp16.dtype() != torch::kFloat16) {
        c_fp16 = c_fp16.to(torch::kFloat16);
    }
    
    auto x_2d = x_fp16.view({M, In});
    
    auto gate = torch::empty({M, H}, x.options().dtype(torch::kFloat16));
    auto value = torch::empty({M, H}, x.options().dtype(torch::kFloat16));
    auto output = torch::empty({M, H}, x.options().dtype(torch::kFloat16));
    
    ensure_cublasLt_handle();
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    size_t workspaceSize = 32 * 1024 * 1024;  // 32MB
    void* workspace;
    cudaMalloc(&workspace, workspaceSize);
    
    cublasLt_gemm_fp16(
        g_cublasLt_handle, stream,
        static_cast<int>(M), static_cast<int>(H), static_cast<int>(In),
        reinterpret_cast<const half*>(W_fp16.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(x_2d.data_ptr<at::Half>()),
        reinterpret_cast<half*>(gate.data_ptr<at::Half>()),
        workspace, workspaceSize
    );
    
    cublasLt_gemm_fp16(
        g_cublasLt_handle, stream,
        static_cast<int>(M), static_cast<int>(H), static_cast<int>(In),
        reinterpret_cast<const half*>(V_fp16.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(x_2d.data_ptr<at::Half>()),
        reinterpret_cast<half*>(value.data_ptr<at::Half>()),
        workspace, workspaceSize
    );
    
    cudaFree(workspace);
    
    const int total_elements = static_cast<int>(M * H);
    const int threads_per_block = 256;
    
    if (H % 2 == 0) {
        const int num_vec2 = total_elements / 2;
        const int blocks = (num_vec2 + threads_per_block - 1) / threads_per_block;
        
        fused_swiglu_activation_fp16_vec2_kernel<<<blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(value.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(b_fp16.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(c_fp16.data_ptr<at::Half>()),
            beta,
            static_cast<int>(M),
            static_cast<int>(H),
            reinterpret_cast<half*>(output.data_ptr<at::Half>())
        );
    } else {
        const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        
        fused_swiglu_activation_fp16_kernel<<<blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(value.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(b_fp16.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(c_fp16.data_ptr<at::Half>()),
            beta,
            static_cast<int>(M),
            static_cast<int>(H),
            reinterpret_cast<half*>(output.data_ptr<at::Half>())
        );
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output.view({B, S, H});  // to 3D
}
