import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
import sys
import io

# CUDA source code loaded from submission.cu
cuda_source = """
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

__global__ void fused_swiglu_interleaved_kernel(
    const half* __restrict__ input, // M x 2H
    const half* __restrict__ b, // H
    const half* __restrict__ c_bias, // H
    float beta,
    int M,
    int H,
    half* __restrict__ output // M x H
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_vec = (M * H) / 2;
    
    if (idx >= total_vec) return;
    
    int i = idx * 2;
    int m = i / H;
    int h = i % H;
    
    int g_vec_idx = (m * 2 * H + h) / 2;
    int v_vec_idx = (m * 2 * H + H + h) / 2;
    
    const half2* input_h2 = reinterpret_cast<const half2*>(input);
    half2 g_vec = input_h2[g_vec_idx];
    half2 v_vec = input_h2[v_vec_idx];
    
    const half2* b_h2 = reinterpret_cast<const half2*>(b);
    const half2* c_h2 = reinterpret_cast<const half2*>(c_bias);
    
    // h is even (from i=idx*2). b, c assumed aligned.
    half2 b_vec = b_h2[h/2];
    half2 c_vec = c_h2[h/2];
    
    float2 g_f2 = __half22float2(g_vec);
    float2 v_f2 = __half22float2(v_vec);
    float2 b_f2 = __half22float2(b_vec);
    float2 c_f2 = __half22float2(c_vec);
    
    float g0 = g_f2.x + b_f2.x;
    float g1 = g_f2.y + b_f2.y;
    float v0 = v_f2.x + c_f2.x;
    float v1 = v_f2.y + c_f2.y;
    
    float swish0 = g0 / (1.0f + expf(-beta * g0));
    float swish1 = g1 / (1.0f + expf(-beta * g1));
    
    half2 out_vec = __float22half2_rn(make_float2(swish0 * v0, swish1 * v1));
    
    reinterpret_cast<half2*>(output)[idx] = out_vec;
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
    torch::Tensor W_fused,
    torch::Tensor b,
    torch::Tensor c,
    float beta
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(W_fused.is_cuda(), "W_fused must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(c.is_cuda(), "c must be a CUDA tensor");
    
    const int64_t B = x.size(0);
    const int64_t S = x.size(1);
    const int64_t In = x.size(2);
    const int64_t H = W_fused.size(1) / 2;
    const int64_t M = B * S;
    
    auto x_fp16 = x.to(torch::kFloat16);
    auto W_fused_fp16 = W_fused.to(torch::kFloat16);
    auto b_fp16 = b.to(torch::kFloat16);
    auto c_fp16 = c.to(torch::kFloat16);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    auto x_2d = x_fp16.view({M, In});
    auto intermediate = torch::empty({M, 2*H}, x.options());
    
    ensure_cublasLt_handle();
    
    size_t workspaceSize = 32 * 1024 * 1024;  // 32MB
    auto workspace_tensor = torch::empty({(long)workspaceSize}, x.options().dtype(torch::kByte));
    void* workspace = workspace_tensor.data_ptr();
    
    cublasLt_gemm_fp16(
        g_cublasLt_handle, stream,
        static_cast<int>(M), static_cast<int>(2*H), static_cast<int>(In),
        reinterpret_cast<const half*>(W_fused_fp16.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(x_2d.data_ptr<at::Half>()),
        reinterpret_cast<half*>(intermediate.data_ptr<at::Half>()),
        workspace, workspaceSize
    );
    
    auto output = torch::empty({M, H}, x.options());
    
    const int total_elements = static_cast<int>(M * H);
    const int threads_per_block = 256;
    const int num_vec = total_elements / 2;
    const int blocks = (num_vec + threads_per_block - 1) / threads_per_block;
    
    fused_swiglu_interleaved_kernel<<<blocks, threads_per_block, 0, stream>>>(
        reinterpret_cast<const half*>(intermediate.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(b_fp16.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(c_fp16.data_ptr<at::Half>()),
        beta,
        static_cast<int>(M),
        static_cast<int>(H),
        reinterpret_cast<half*>(output.data_ptr<at::Half>())
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output.view({B, S, H});
}

"""

# C++ header declaration
cpp_source = """
#include <torch/extension.h>
torch::Tensor custom_kernel(
    torch::Tensor x,
    torch::Tensor W_fused,
    torch::Tensor b,
    torch::Tensor c,
    float beta
);
"""

# Ensure stdout and stderr exist
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

cuda_module = load_inline(
    name='submission_cuda_swiglu_rishic3',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['custom_kernel'],
    verbose=True,  # Enable verbose to see compilation details
    extra_ldflags=['-lcublas', '-lcublasLt'],
    extra_cuda_cflags=['--use_fast_math'],
    # with_cuda=True,
    # build_directory=".",
)

def custom_kernel(data: input_t) -> output_t:
    # SwiGLU input_t: (x, W, V, b, c, beta)
    x, W, V, b, c, beta = data

    W_fused = torch.cat([W, V], dim=1).contiguous()

    return cuda_module.custom_kernel(
        x,
        W_fused,
        b,
        c,
        beta,
    )
