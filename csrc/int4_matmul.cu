/**
 * int4_matmul.cu
 *
 * Fused INT4 dequantize + matmul kernel (D1b optimisation).
 *
 * On-the-fly unpacks 2×INT4 per byte into fp16, applies group-scale
 * dequantisation, and accumulates the dot product in fp32 registers.
 * Avoids materialising the full fp16 weight matrix in HBM.
 *
 * Target: batch=1..4 decode M≤4, where Tensor Cores are under-utilised.
 * Uses SIMT warp-level dot product with FMAD instructions.
 *
 * ── Packed layout (must match ``gemma4_quant._pack_int4``) ──
 *   w_packed is [N/2, K] uint8, row-major, so ONE PACKED ROW IS K BYTES
 *   (= 2*K nibbles) and covers TWO output rows:
 *
 *       byte (n >> 1, k >> 1) = (w[n|1][k] << 4) | (w[n&~1][k] & 0xF)
 *
 *   i.e. even output row n → LOW nibble, odd output row n → HIGH nibble,
 *   values unsigned [0,15] biased by 8 from signed [-8,7].
 *
 *   The row-address stride is therefore ``K`` bytes and a thread must load
 *   exactly ONE nibble (selected by the output row's parity). Summing both
 *   nibbles of a byte mixes in the neighbouring output row's weights, and
 *   using a K/2 stride walks off the end of the row entirely.
 *
 * Thread layout:
 *   grid:  [B, (N + N_TILE - 1) / N_TILE],  block: 256 threads (8 warps)
 *
 *   The block cooperatively loads the [2,16] nibble lookup table and the
 *   K_TILE x-values into shared memory, then every thread walks the K tiles
 *   for its own output rows. Weights are read as one unpacked byte per
 *   output row and broadcast through shared memory, so the x tile is read
 *   from shared memory and the weight traffic is the theoretical 4-bit
 *   minimum (no dense fp16 weight is ever materialised).
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

// type conversion helpers (templated for fp16 / bf16)
template<typename T>
__device__ __forceinline__ float _to_float(T v);
template<> __device__ __forceinline__ float _to_float(__half v)         { return __half2float(v); }
template<> __device__ __forceinline__ float _to_float(__nv_bfloat16 v)  { return __bfloat162float(v); }

template<typename T>
__device__ __forceinline__ T _from_float(float v);
template<> __device__ __forceinline__ __half        _from_float<__half>(float v)        { return __float2half(v); }
template<> __device__ __forceinline__ __nv_bfloat16 _from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

namespace {

constexpr int kNThreads = 128;   // 4 warps
constexpr int kNTile    = 128;   // output rows per block
constexpr int kKTile    = 64;    // K elements per tile (== one scale group)

}  // namespace

// ───────────────────────────────────────────────────────────────────
// Main kernel
// ───────────────────────────────────────────────────────────────────
template<typename T>
__global__ void int4_dequant_matmul_kernel(
    const T*        __restrict__ x,            // [B, K]
    const uint8_t*  __restrict__ w_packed,     // [N/2, K] uint8 (2 INT4/byte)
    const __half*   __restrict__ group_scales, // [N, K/group_size], fp16
    T*              __restrict__ y,            // [B, N]
    int B, int N, int K, int group_size)
{
    const int batch_idx = blockIdx.x;
    const int n_start   = blockIdx.y * kNTile;
    const int tid       = threadIdx.x;

    const int n_this = min(kNTile, N - n_start);
    if (n_this <= 0) return;

    const int n_groups = K / group_size;

    const T* __restrict__ x_row = x + (long long)batch_idx * K;
    T* __restrict__ y_row       = y + (long long)batch_idx * N + n_start;

    __shared__ float smem_nib[16];
    __shared__ T     smem_x[kKTile];

    // Nibble code -> signed value.
    if (tid < 16) {
        smem_nib[tid] = (float)(tid - 8);
    }
    __syncthreads();

    float acc = 0.0f;

    for (int k_start = 0; k_start < K; k_start += kKTile) {
        const int k_this = min(kKTile, K - k_start);
        const int k_bytes = k_this;                // one packed byte per column
        const int g = k_start / group_size;

        if (tid < k_this) {
            smem_x[tid] = x_row[k_start + tid];
        }
        __syncthreads();

        const int n_local = tid;
        if (n_local < n_this) {
            const int n_global = n_start + n_local;
            const uint8_t* __restrict__ w_pair =
                w_packed + (long long)(n_global >> 1) * K;
            const bool high_row = (n_global & 1) != 0;
            float dot = 0.0f;
            #pragma unroll 4
            for (int kb = 0; kb < k_bytes; kb++) {
                const uint8_t byte = w_pair[k_start + kb];
                const int code = high_row ? (int)(byte >> 4) : (int)(byte & 0x0F);
                dot += _to_float(smem_x[kb]) * smem_nib[code];
            }
            acc += dot * _to_float(group_scales[(long long)n_global * n_groups + g]);
        }
        __syncthreads();
    }

    if (tid < n_this)
        y_row[tid] = _from_float<T>(acc);
}

// ───────────────────────────────────────────────────────────────────
// Python binding
// ───────────────────────────────────────────────────────────────────
torch::Tensor int4_dequant_matmul(
    torch::Tensor x,               // [*, K] fp16 (automatically batches over leading dims)
    torch::Tensor w_packed,        // [N//2, K] uint8
    torch::Tensor group_scales,    // [N, K//group_size] fp16
    int group_size = 64)
{
    TORCH_CHECK(x.is_cuda() && w_packed.is_cuda() && group_scales.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(x.dtype() == torch::kFloat16 || x.dtype() == torch::kBFloat16,
                "x must be fp16/bf16");
    TORCH_CHECK(w_packed.dtype() == torch::kUInt8, "w_packed must be uint8");
    TORCH_CHECK(group_scales.dtype() == torch::kFloat16,
                "Gemma4 INT4 group scales must be float16");
    TORCH_CHECK(group_size == kKTile,
                "Gemma4 INT4 CUDA kernel currently requires group_size=64");

    // Handle arbitrary batch dimensions by flattening
    auto orig_shape = x.sizes();
    int64_t batch_elems = 1;
    for (int64_t d = 0; d < (int64_t)orig_shape.size() - 1; d++)
        batch_elems *= orig_shape[d];
    int K = (int)orig_shape.back();
    int N = w_packed.size(0) * 2;  // 2 output dims per packed row

    TORCH_CHECK(x.numel() == batch_elems * K, "x shape mismatch");
    TORCH_CHECK(w_packed.size(1) == K, "w_packed K dim must match x last dim");
    TORCH_CHECK(K % 2 == 0, "K must be even for INT4 packing");
    TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");
    TORCH_CHECK(group_scales.size(0) == N, "group_scales N must match unpacked N");
    TORCH_CHECK(group_scales.size(1) == K / group_size, "group_scales K dim mismatch");
    TORCH_CHECK(x.is_contiguous() && w_packed.is_contiguous() &&
                    group_scales.is_contiguous(),
                "INT4 CUDA inputs must be contiguous");
    const c10::cuda::CUDAGuard device_guard(x.device());
    TORCH_CHECK(w_packed.device() == x.device() &&
                    group_scales.device() == x.device(),
                "INT4 CUDA inputs must be on the same device");
    // The kernel loads a full K_TILE of activations into shared memory.
    TORCH_CHECK(K % kKTile == 0 || K >= kKTile,
                "K must be at least one K_TILE for the shared-memory staging path");

    auto x_2d = x.reshape({batch_elems, K});
    auto y = torch::empty({batch_elems, N}, x.options());

    dim3 grid((unsigned)batch_elems, (unsigned)((N + kNTile - 1) / kNTile));

    if (x.dtype() == torch::kFloat16) {
        int4_dequant_matmul_kernel<__half><<<
            grid, kNThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(x_2d.data_ptr()),
            w_packed.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(group_scales.data_ptr()),
            reinterpret_cast<__half*>(y.data_ptr()),
            (int)batch_elems, N, K, group_size
        );
    } else {
        int4_dequant_matmul_kernel<__nv_bfloat16><<<
            grid, kNThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(x_2d.data_ptr()),
            w_packed.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(group_scales.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
            (int)batch_elems, N, K, group_size
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Restore original batch shape
    if (orig_shape.size() > 2) {
        std::vector<int64_t> out_shape(orig_shape.begin(), orig_shape.end() - 1);
        out_shape.push_back(N);
        y = y.reshape(out_shape);
    }
    return y;
}
