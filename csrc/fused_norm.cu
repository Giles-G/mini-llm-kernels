/**
 * fused_norm.cu
 *
 * 融合 RMSNorm + Residual Add Kernel（Stage 8 M3 优化）
 *
 * 功能：
 *   一个 kernel 完成两步操作：
 *   1. residual_out = x + residual
 *   2. x_normed = RMSNorm(residual_out, gamma, eps)
 *
 * 替换主仓库 model_runner.py 每层两处独立的：
 *   residual = hidden_states
 *   x = rms_norm(hidden_states, gamma, eps)   ← 读 hidden, 写 x_norm
 *   hidden_states = residual + output          ← 读 residual+output, 写 hidden
 *
 * HBM 访问节省：
 *   原来（2 kernel）：读 3 次（hidden×2 + output×1），写 2 次
 *   融合后（1 kernel）：读 2 次（x + residual），写 2 次
 *
 * 线程组织：
 *   grid:  [N]，每个 block 处理一行（一个 token）
 *   block: min(H, 1024) threads，处理 hidden_size 方向
 *   warp reduce：__shfl_xor_sync 计算 sum(x²)
 *
 * 数据类型：
 *   输入/输出：fp16 或 bf16
 *   内部计算：fp32（保精度）
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

// ─────────────────────────────────────────────────────────────────────────────
// 辅助：half / bfloat16 → float 互转
// ─────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__ float to_float(const __half v) {
    return __half2float(v);
}
__device__ __forceinline__ float to_float(const __nv_bfloat16 v) {
    return __bfloat162float(v);
}
__device__ __forceinline__ __half from_float_half(float v) {
    return __float2half(v);
}
__device__ __forceinline__ __nv_bfloat16 from_float_bf16(float v) {
    return __float2bfloat16(v);
}

// ─────────────────────────────────────────────────────────────────────────────
// 核心 kernel（模板化，支持 half / bfloat16）
// ─────────────────────────────────────────────────────────────────────────────

template <typename T>
__global__ void fused_add_rms_norm_kernel(
    const T*     __restrict__ x,            // [N, H]  attention/mlp 输出
    const T*     __restrict__ residual,     // [N, H]  上一层的 residual
    const T*     __restrict__ gamma,        // [H]     RMSNorm 可学习缩放
    T*           __restrict__ x_normed,     // [N, H]  输出：norm 后的结果
    T*           __restrict__ residual_out, // [N, H]  输出：更新后的 residual（= x + residual）
    int N, int H, float eps)
{
    // 每个 block 处理一行（一个 token）
    int row = blockIdx.x;
    if (row >= N) return;

    const T* x_row        = x        + row * H;
    const T* res_row      = residual + row * H;
    T*       out_row      = x_normed     + row * H;
    T*       res_out_row  = residual_out + row * H;

    // ── shared memory: warp reduce + val buffer ──
    extern __shared__ float smem[];  // [num_warps + H]
    float* reduce_smem = smem;               // [0 .. num_warps-1]
    float* val_smem     = smem + num_warps;  // [num_warps .. num_warps+H-1]

    // ── Step 1: residual_out = x + residual，同时累积 sum(val²) 并保存 val ──
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float val = to_float(x_row[i]) + to_float(res_row[i]);
        if constexpr (std::is_same_v<T, __half>) {
            res_out_row[i] = from_float_half(val);
        } else {
            res_out_row[i] = from_float_bf16(val);
        }
        val_smem[i] = val;  // 保存到 shared memory，避免 Step 4 重复读 HBM
        sum_sq += val * val;
    }

    // ── Step 2: warp reduce sum_sq ──
    // 先 warp 内 reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);

    // 需要跨 warp reduce（当 blockDim.x > 32 时）
    // 用 shared memory 聚合各 warp 的结果
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = (blockDim.x + 31) / 32;

    if (lane_id == 0) reduce_smem[warp_id] = sum_sq;
    __syncthreads();

    // 只有第 0 个 warp 做最终 reduce
    if (warp_id == 0) {
        sum_sq = (lane_id < num_warps) ? reduce_smem[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
        if (lane_id == 0) reduce_smem[0] = sum_sq;
    }
    __syncthreads();
    sum_sq = reduce_smem[0];

    // ── Step 3: RMS = 1 / sqrt(mean(x²) + eps) ──
    float rms_scale = rsqrtf(sum_sq / H + eps);

    // ── Step 4: normalize + scale → x_normed（从 shared memory 读 val，避免重复 HBM 读） ──
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float normed = val_smem[i] * rms_scale * to_float(gamma[i]);
        if constexpr (std::is_same_v<T, __half>) {
            out_row[i] = from_float_half(normed);
        } else {
            out_row[i] = from_float_bf16(normed);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Python 绑定入口
// ─────────────────────────────────────────────────────────────────────────────

/**
 * fused_add_rms_norm
 *
 * Args:
 *   x:        [N, H] fp16/bf16，attention/mlp 的输出（"delta"）
 *   residual: [N, H] fp16/bf16，上一子层的残差流
 *   gamma:    [H]    fp16/bf16，RMSNorm 的可学习缩放参数
 *   eps:      float，防除零小量
 *
 * Returns:
 *   (x_normed, residual_out)
 *   x_normed:     [N, H]  归一化后的结果，传给下一算子（QKV / MLP）
 *   residual_out: [N, H]  x + residual，作为下一子层的残差输入
 */
std::pair<torch::Tensor, torch::Tensor> fused_add_rms_norm(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor gamma,
    float eps)
{
    TORCH_CHECK(x.is_cuda(),        "x must be CUDA tensor");
    TORCH_CHECK(residual.is_cuda(), "residual must be CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(),    "gamma must be CUDA tensor");
    TORCH_CHECK(x.dim() == 2,       "x must be 2-D [N, H]");
    TORCH_CHECK(x.sizes() == residual.sizes(), "x and residual must have same shape");
    TORCH_CHECK(x.dtype() == residual.dtype(), "x and residual must have same dtype");
    TORCH_CHECK(x.dtype() == gamma.dtype(),    "x and gamma must have same dtype");
    TORCH_CHECK(x.dtype() == torch::kFloat16 || x.dtype() == torch::kBFloat16,
                "only fp16 / bf16 supported");

    int N = x.size(0);
    int H = x.size(1);

    auto x_normed     = torch::empty_like(x);
    auto residual_out = torch::empty_like(x);

    int threads = std::min(H, 1024);
    // 向上对齐到 32（warp 大小）
    threads = ((threads + 31) / 32) * 32;
    int num_warps = threads / 32;
    int smem_bytes = (num_warps + H) * sizeof(float);  // reduce + val buffer

    if (x.dtype() == torch::kFloat16) {
        fused_add_rms_norm_kernel<__half><<<N, threads, smem_bytes>>>(
            reinterpret_cast<const __half*>(x.data_ptr()),
            reinterpret_cast<const __half*>(residual.data_ptr()),
            reinterpret_cast<const __half*>(gamma.data_ptr()),
            reinterpret_cast<__half*>(x_normed.data_ptr()),
            reinterpret_cast<__half*>(residual_out.data_ptr()),
            N, H, eps
        );
    } else {
        fused_add_rms_norm_kernel<__nv_bfloat16><<<N, threads, smem_bytes>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(x_normed.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr()),
            N, H, eps
        );
    }

    return {x_normed, residual_out};
}
