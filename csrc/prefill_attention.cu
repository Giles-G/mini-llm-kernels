/**
 * Paged Prefill Attention Kernel (B1)
 *
 * Layout:
 *   q:           [N, T_q, H_q, D]
 *   k_cache/v:   [num_blocks, block_size, H_kv, D]
 *   block_table: [N, max_blocks]
 *   chunk_lens/history_lens: [N]
 *   out:         [N, T_q, H_q, D]
 *
 * One CUDA block handles one (request, query token, query head). KV is read
 * directly from physical blocks; no padded KV gather is materialized.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <float.h>
#include <math.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

#define MAX_BLOCK_DIM 128

template <typename T>
__device__ __forceinline__ float prefill_to_float(T value);
template <> __device__ __forceinline__ float prefill_to_float(__half value) {
    return __half2float(value);
}
template <> __device__ __forceinline__ float prefill_to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T prefill_from_float(float value);
template <> __device__ __forceinline__ __half prefill_from_float(float value) {
    return __float2half(value);
}
template <> __device__ __forceinline__ __nv_bfloat16 prefill_from_float(float value) {
    return __float2bfloat16(value);
}

__device__ __forceinline__ float prefill_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffff, value, offset);
    return value;
}

template <typename T>
__global__ void paged_prefill_attention_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k_cache,
    const T* __restrict__ v_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ chunk_lens,
    const int* __restrict__ history_lens,
    T* __restrict__ out,
    int n_requests,
    int query_len,
    int h_q,
    int h_kv,
    int d,
    int block_size,
    int max_blocks,
    float scale) {
    const int request = blockIdx.x;
    const int q_head = blockIdx.y;
    const int query = blockIdx.z;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = blockDim.x >> 5;

    if (request >= n_requests || q_head >= h_q || query >= query_len)
        return;

    const int chunk_len = chunk_lens[request];
    const int history_len = history_lens[request];
    if (query >= chunk_len) {
        T* out_ptr = out + (((long long)request * query_len + query) * h_q + q_head) * d;
        for (int i = tid; i < d; i += blockDim.x)
            out_ptr[i] = prefill_from_float<T>(0.0f);
        return;
    }

    const int total_len = history_len + chunk_len;
    const int query_position = history_len + query;
    const int gqa_ratio = h_q / h_kv;
    const int kv_head = q_head / gqa_ratio;

    extern __shared__ char smem[];
    T* k_tile = reinterpret_cast<T*>(smem);
    T* v_tile = k_tile + block_size * d;
    float* warp_partial = reinterpret_cast<float*>(v_tile + block_size * d);

    float output_acc[MAX_BLOCK_DIM] = {};
    float max_acc = -FLT_MAX;
    float sum_acc = 0.0f;

    const T* q_ptr = q + (((long long)request * query_len + query) * h_q + q_head) * d;
    const int num_kv_blocks = (total_len + block_size - 1) / block_size;

    for (int block_idx = 0; block_idx < num_kv_blocks; ++block_idx) {
        const int physical_block = block_table[request * max_blocks + block_idx];
        const int logical_start = block_idx * block_size;
        const int block_len = min(block_size, total_len - logical_start);

        const T* k_base = k_cache + ((long long)physical_block * block_size * h_kv + kv_head) * d;
        const T* v_base = v_cache + ((long long)physical_block * block_size * h_kv + kv_head) * d;

        for (int slot = 0; slot < block_len; ++slot) {
            const T* k_row = k_base + (long long)slot * h_kv * d;
            const T* v_row = v_base + (long long)slot * h_kv * d;
            for (int i = tid; i < d; i += blockDim.x) {
                k_tile[slot * d + i] = k_row[i];
                v_tile[slot * d + i] = v_row[i];
            }
        }
        __syncthreads();

        for (int slot = 0; slot < block_len; ++slot) {
            const int logical_position = logical_start + slot;
            float dot = 0.0f;
            const T* k_row = k_tile + slot * d;
            for (int i = tid; i < d; i += blockDim.x)
                dot += prefill_to_float(q_ptr[i]) * prefill_to_float(k_row[i]);
            dot = prefill_warp_sum(dot);

            if (lane == 0)
                warp_partial[warp] = dot;
            __syncthreads();

            if (warp == 0) {
                dot = (lane < warps) ? warp_partial[lane] : 0.0f;
                dot = prefill_warp_sum(dot);
                if (lane == 0)
                    warp_partial[0] = dot;
            }
            __syncthreads();
            dot = warp_partial[0] * scale;

            if (logical_position <= query_position) {
                const float new_max = fmaxf(max_acc, dot);
                const float old_factor = (max_acc == -FLT_MAX) ? 0.0f : expf(max_acc - new_max);
                const float current_factor = expf(dot - new_max);
                const T* v_row = v_tile + slot * d;
                for (int i = tid; i < d; i += blockDim.x)
                    output_acc[i] = old_factor * output_acc[i] + current_factor * prefill_to_float(v_row[i]);
                sum_acc = old_factor * sum_acc + current_factor;
                max_acc = new_max;
            }
            __syncthreads();
        }
        __syncthreads();
    }

    T* out_ptr = out + (((long long)request * query_len + query) * h_q + q_head) * d;
    for (int i = tid; i < d; i += blockDim.x) {
        const float value = sum_acc > 0.0f ? output_acc[i] / sum_acc : 0.0f;
        out_ptr[i] = prefill_from_float<T>(value);
    }
}

torch::Tensor paged_prefill_attention(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_table,
    torch::Tensor chunk_lens,
    torch::Tensor history_lens) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA tensor");
    TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda(), "KV cache must be CUDA tensors");
    TORCH_CHECK(block_table.is_cuda(), "block_table must be CUDA tensor");
    TORCH_CHECK(chunk_lens.is_cuda() && history_lens.is_cuda(), "length tensors must be CUDA tensors");
    TORCH_CHECK(q.dtype() == torch::kFloat16 || q.dtype() == torch::kBFloat16,
                "only fp16 / bf16 supported");
    TORCH_CHECK(k_cache.scalar_type() == q.scalar_type() && v_cache.scalar_type() == q.scalar_type(),
                "q and KV cache dtypes must match");
    TORCH_CHECK(q.dim() == 4, "q must be [N, T, H_q, D]");
    TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4,
                "KV cache must be [blocks, block_size, H_kv, D]");
    TORCH_CHECK(block_table.dim() == 2 && chunk_lens.dim() == 1 && history_lens.dim() == 1,
                "invalid metadata dimensions");
    TORCH_CHECK(k_cache.size(1) > 0 && k_cache.size(3) == q.size(3), "invalid KV cache shape");

    const int n_requests = q.size(0);
    const int query_len = q.size(1);
    const int h_q = q.size(2);
    const int d = q.size(3);
    const int block_size = k_cache.size(1);
    const int h_kv = k_cache.size(2);
    const int max_blocks = block_table.size(1);

    TORCH_CHECK(block_table.size(0) == n_requests && chunk_lens.size(0) == n_requests &&
                history_lens.size(0) == n_requests, "metadata batch size mismatch");
    TORCH_CHECK(h_q % h_kv == 0, "H_q must be divisible by H_kv (GQA)");
    TORCH_CHECK(d <= MAX_BLOCK_DIM && d % 8 == 0, "head_dim must be <=128 and divisible by 8");

    auto out = torch::empty_like(q);
    const int threads = ((d + 31) / 32) * 32;
    const size_t smem_bytes = 2ULL * block_size * d * q.element_size() +
                              (threads / 32) * sizeof(float);
    dim3 grid(n_requests, h_q, query_len);
    const float scale = 1.0f / sqrtf(static_cast<float>(d));

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    TORCH_CHECK(smem_bytes <= prop.sharedMemPerBlockOptin,
                "paged prefill shared memory demand exceeds device limit");

    if (q.scalar_type() == torch::kFloat16) {
        if (smem_bytes > prop.sharedMemPerBlock)
            cudaFuncSetAttribute(paged_prefill_attention_kernel<__half>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        paged_prefill_attention_kernel<__half><<<grid, threads, smem_bytes>>>(
            reinterpret_cast<const __half*>(q.data_ptr()),
            reinterpret_cast<const __half*>(k_cache.data_ptr()),
            reinterpret_cast<const __half*>(v_cache.data_ptr()),
            block_table.data_ptr<int>(), chunk_lens.data_ptr<int>(), history_lens.data_ptr<int>(),
            reinterpret_cast<__half*>(out.data_ptr()), n_requests, query_len, h_q, h_kv, d,
            block_size, max_blocks, scale);
    } else {
        if (smem_bytes > prop.sharedMemPerBlock)
            cudaFuncSetAttribute(paged_prefill_attention_kernel<__nv_bfloat16>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        paged_prefill_attention_kernel<__nv_bfloat16><<<grid, threads, smem_bytes>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr()),
            block_table.data_ptr<int>(), chunk_lens.data_ptr<int>(), history_lens.data_ptr<int>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n_requests, query_len, h_q, h_kv, d,
            block_size, max_blocks, scale);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
