/**
 * add_tensors.cu
 *
 * Hello World CUDA kernel for mini-llm-kernels.
 * 验证编译链路：逐元素加法
 *
 * grid:  ceil(n / 256) blocks
 * block: 256 threads
 */

#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void add_tensors_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "only float32 supported");

    auto out = torch::empty_like(a);
    int64_t n = a.numel();
    int threads = 256;
    int64_t blocks = (n + threads - 1) / threads;

    add_tensors_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );
    return out;
}
