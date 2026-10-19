/**
 * int4_matmul.cu
 *
 * Fused INT4 dequantize + matmul kernels.  The legacy kernel is retained
 * beside the newer warp-per-row kernel so callers can choose explicitly.
 *
 * Packed layout:
 *   w_packed is [N/2, K] uint8, row-major.  Each byte at [n >> 1, k]
 *   contains the two output rows' values at column k:
 *
 *       low nibble  = even output row n
 *       high nibble = odd output row n
 *
 *   The packed-row stride is K bytes.  In particular, the column index is k,
 *   not k >> 1: each byte stores two rows at the same K position.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>

template<typename T>
__device__ __forceinline__ float _to_float(T v);
template<> __device__ __forceinline__ float _to_float(__half v) { return __half2float(v); }
template<> __device__ __forceinline__ float _to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

template<typename T>
__device__ __forceinline__ T _from_float(float v);
template<> __device__ __forceinline__ __half _from_float<__half>(float v) {
    return __float2half(v);
}
template<> __device__ __forceinline__ __nv_bfloat16 _from_float<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

namespace {

constexpr int kKTile = 64;
constexpr int kLegacyThreads = 128;
constexpr int kLegacyRowsPerBlock = 128;
constexpr int kWarpThreads = 128;
constexpr int kWarpRowsPerBlock = 4;

}  // namespace

// The HEAD implementation: one thread computes one output row.
template<typename T>
__global__ void int4_dequant_matmul_legacy_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w_packed,
    const __half* __restrict__ group_scales,
    T* __restrict__ y,
    int B, int N, int K, int group_size)
{
    const int batch_idx = blockIdx.y;
    const int n_start = blockIdx.x * kLegacyRowsPerBlock;
    const int tid = threadIdx.x;
    const int n_this = min(kLegacyRowsPerBlock, N - n_start);
    if (n_this <= 0) return;

    const int n_groups = K / group_size;
    const T* __restrict__ x_row = x + (long long)batch_idx * K;
    T* __restrict__ y_row = y + (long long)batch_idx * N + n_start;

    __shared__ float smem_nib[16];
    __shared__ T smem_x[kKTile];

    if (tid < 16) smem_nib[tid] = (float)(tid - 8);
    __syncthreads();

    float acc = 0.0f;
    for (int k_start = 0; k_start < K; k_start += kKTile) {
        const int k_this = min(kKTile, K - k_start);
        const int g = k_start / group_size;
        if (tid < k_this) smem_x[tid] = x_row[k_start + tid];
        __syncthreads();

        const int n_local = tid;
        if (n_local < n_this) {
            const int n_global = n_start + n_local;
            const uint8_t* w_pair =
                w_packed + (long long)(n_global >> 1) * K;
            const bool high_row = (n_global & 1) != 0;
            float dot = 0.0f;
            #pragma unroll 4
            for (int kb = 0; kb < k_this; ++kb) {
                const uint8_t byte = w_pair[k_start + kb];
                const int code = high_row ? (int)(byte >> 4) : (int)(byte & 0x0F);
                dot += _to_float(smem_x[kb]) * smem_nib[code];
            }
            acc += dot * _to_float(group_scales[(long long)n_global * n_groups + g]);
        }
        __syncthreads();
    }

    if (tid < n_this) y_row[tid] = _from_float<T>(acc);
}

// New kernel: one warp computes one output row.
template<typename T>
__global__ void int4_dequant_matmul_warp_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w_packed,
    const __half* __restrict__ group_scales,
    T* __restrict__ y,
    int B, int N, int K, int group_size)
{
    const int batch_idx = blockIdx.y;
    const int n_global = blockIdx.x * kWarpRowsPerBlock + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const bool valid_row = n_global < N;
    const int n_groups = K / group_size;

    const T* x_row = x + (long long)batch_idx * K;
    T* y_row = y + (long long)batch_idx * N;
    const uint8_t* w_pair = valid_row
        ? w_packed + (long long)(n_global >> 1) * K
        : nullptr;
    const bool high_row = (n_global & 1) != 0;

    float acc = 0.0f;
    for (int g = 0; valid_row && g < n_groups; ++g) {
        const int k_start = g * kKTile;
        const int k0 = k_start + lane;
        const int k1 = k0 + 32;
        const uint8_t b0 = w_pair[k0];
        const uint8_t b1 = w_pair[k1];
        const int q0 = high_row ? (int)(b0 >> 4) : (int)(b0 & 0x0F);
        const int q1 = high_row ? (int)(b1 >> 4) : (int)(b1 & 0x0F);
        const float scale = _to_float(group_scales[(long long)n_global * n_groups + g]);
        acc += (_to_float(x_row[k0]) * (float)(q0 - 8)
              + _to_float(x_row[k1]) * (float)(q1 - 8)) * scale;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    if (valid_row && lane == 0) y_row[n_global] = _from_float<T>(acc);
}

struct Int4LaunchShape {
    torch::Tensor x_2d;
    torch::Tensor y;
    int64_t batch_elems;
    int N;
    int K;
    int group_size;
};

Int4LaunchShape check_int4_inputs(
    const torch::Tensor& x,
    const torch::Tensor& w_packed,
    const torch::Tensor& group_scales,
    int group_size)
{
    TORCH_CHECK(x.is_cuda() && w_packed.is_cuda() && group_scales.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() >= 1, "x must have rank >= 1");
    TORCH_CHECK(w_packed.dim() == 2, "w_packed must have rank 2");
    TORCH_CHECK(group_scales.dim() == 2, "group_scales must have rank 2");
    TORCH_CHECK(x.dtype() == torch::kFloat16 || x.dtype() == torch::kBFloat16,
                "x must be fp16/bf16");
    TORCH_CHECK(w_packed.dtype() == torch::kUInt8, "w_packed must be uint8");
    TORCH_CHECK(group_scales.dtype() == torch::kFloat16,
                "Gemma4 INT4 group scales must be float16");
    TORCH_CHECK(group_size == kKTile, "INT4 CUDA kernel requires group_size=64");
    TORCH_CHECK(x.is_contiguous() && w_packed.is_contiguous() &&
                    group_scales.is_contiguous(),
                "INT4 CUDA inputs must be contiguous");

    const int64_t K64 = x.size(-1);
    TORCH_CHECK(w_packed.size(0) <= INT_MAX / 2,
                "packed N exceeds int32 range");
    const int64_t N64 = w_packed.size(0) * 2;
    int64_t batch_elems = 1;
    for (int64_t d = 0; d + 1 < x.dim(); ++d) {
        TORCH_CHECK(x.size(d) >= 0, "x dimensions must be non-negative");
        TORCH_CHECK(x.size(d) <= INT_MAX, "x dimension exceeds int32 range");
        batch_elems *= x.size(d);
        TORCH_CHECK(batch_elems <= INT_MAX, "flattened batch exceeds int32 range");
    }
    TORCH_CHECK(K64 > 0 && K64 <= INT_MAX, "x K dimension must be positive and fit int32");
    TORCH_CHECK(N64 > 0 && N64 <= INT_MAX, "unpacked N must be positive and fit int32");
    TORCH_CHECK(w_packed.size(1) == K64, "w_packed K dim must match x last dim");
    TORCH_CHECK(K64 % 2 == 0, "K must be even for INT4 packing");
    TORCH_CHECK(K64 % group_size == 0, "K must be divisible by group_size");
    TORCH_CHECK(group_scales.size(0) == N64, "group_scales N must match unpacked N");
    TORCH_CHECK(group_scales.size(1) == K64 / group_size, "group_scales K dim mismatch");
    const c10::cuda::CUDAGuard device_guard(x.device());
    TORCH_CHECK(w_packed.device() == x.device() &&
                    group_scales.device() == x.device(),
                "INT4 CUDA inputs must be on the same device");
    TORCH_CHECK(batch_elems <= 65535, "INT4 CUDA batch grid.y exceeds CUDA limit");

    auto x_2d = x.reshape({batch_elems, K64});
    auto y = torch::empty({batch_elems, N64}, x.options());
    return {x_2d, y, batch_elems, (int)N64, (int)K64, group_size};
}

template<bool Legacy>
torch::Tensor int4_dequant_matmul_impl(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor group_scales, int group_size)
{
    const c10::cuda::CUDAGuard device_guard(x.device());
    auto shape = check_int4_inputs(x, w_packed, group_scales, group_size);
    auto orig_shape = x.sizes();
    if (shape.batch_elems != 0) {
        const dim3 grid(
            (unsigned)(((int64_t)shape.N + (Legacy ? kLegacyRowsPerBlock : kWarpRowsPerBlock) - 1)
                       / (Legacy ? kLegacyRowsPerBlock : kWarpRowsPerBlock)),
            (unsigned)shape.batch_elems);
        auto stream = at::cuda::getCurrentCUDAStream();
        if (x.dtype() == torch::kFloat16) {
            if constexpr (Legacy) {
                int4_dequant_matmul_legacy_kernel<__half><<<grid, kLegacyThreads, 0, stream>>>(
                    (const __half*)shape.x_2d.data_ptr(), w_packed.data_ptr<uint8_t>(),
                    (const __half*)group_scales.data_ptr(), (__half*)shape.y.data_ptr(),
                    (int)shape.batch_elems, shape.N, shape.K, shape.group_size);
            } else {
                int4_dequant_matmul_warp_kernel<__half><<<grid, kWarpThreads, 0, stream>>>(
                    (const __half*)shape.x_2d.data_ptr(), w_packed.data_ptr<uint8_t>(),
                    (const __half*)group_scales.data_ptr(), (__half*)shape.y.data_ptr(),
                    (int)shape.batch_elems, shape.N, shape.K, shape.group_size);
            }
        } else {
            if constexpr (Legacy) {
                int4_dequant_matmul_legacy_kernel<__nv_bfloat16><<<grid, kLegacyThreads, 0, stream>>>(
                    (const __nv_bfloat16*)shape.x_2d.data_ptr(), w_packed.data_ptr<uint8_t>(),
                    (const __half*)group_scales.data_ptr(), (__nv_bfloat16*)shape.y.data_ptr(),
                    (int)shape.batch_elems, shape.N, shape.K, shape.group_size);
            } else {
                int4_dequant_matmul_warp_kernel<__nv_bfloat16><<<grid, kWarpThreads, 0, stream>>>(
                    (const __nv_bfloat16*)shape.x_2d.data_ptr(), w_packed.data_ptr<uint8_t>(),
                    (const __half*)group_scales.data_ptr(), (__nv_bfloat16*)shape.y.data_ptr(),
                    (int)shape.batch_elems, shape.N, shape.K, shape.group_size);
            }
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (orig_shape.size() != 2) {
        std::vector<int64_t> out_shape(orig_shape.begin(), orig_shape.end() - 1);
        out_shape.push_back(shape.N);
        shape.y = shape.y.reshape(out_shape);
    }
    return shape.y;
}

torch::Tensor int4_dequant_matmul(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor group_scales, int group_size)
{
    return int4_dequant_matmul_impl<false>(x, w_packed, group_scales, group_size);
}

torch::Tensor int4_dequant_matmul_legacy(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor group_scales, int group_size)
{
    return int4_dequant_matmul_impl<true>(x, w_packed, group_scales, group_size);
}
