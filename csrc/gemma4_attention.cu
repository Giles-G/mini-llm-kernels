/**
 * gemma4_attention.cu
 *
 * Paged decode attention for Gemma4's heterogeneous head dimensions.
 *
 * Why a dedicated kernel instead of ``decode_paged_attention``:
 *   - Gemma4 uses head_dim=256 on sliding layers and head_dim=512 on full
 *     layers. The generic kernel caps head_dim at MAX_BLOCK_DIM=128 and
 *     keeps ``o_acc[MAX_BLOCK_DIM]`` in registers per thread.
 *   - Sliding layers only attend to the last ``sliding_window`` tokens, so
 *     the kernel must skip older KV instead of walking every block. Without
 *     that a 128K context would re-read the whole history every step.
 *
 * ⚠ STATUS: NOT USABLE YET — the Python wrapper disables this kernel.
 *   Two designs were tried and both fail at runtime: a per-tile loop version
 *   deadlocked, and the current split-K version also hangs (reproduced in a
 *   standalone .cu with the same nvcc flags: execution stops at the first
 *   warp shuffle that follows a divergently-defined ``p``). Root cause is not
 *   yet identified — a minimal repro of that exact shuffle pattern passes, so
 *   the trigger is elsewhere in this file.
 *
 *   ``mini_llm_kernels.cuda_attention_usable()`` therefore probes this kernel
 *   once at import and falls back to the bit-exact PyTorch implementation
 *   unless it returns the right answer. Do not route production traffic here
 *   until that probe passes. Set MINI_LLM_GEMMA4_ATTN_KERNEL=1 to force it on
 *   while debugging.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Design: split-K over the context, two kernels.
 *
 *   partial kernel  grid = [N, H_q, num_chunks], BLOCK_DIM threads
 *     Each block owns one KV chunk of KV_CHUNK tokens and one warp per token,
 *     so the q·k reduction stays inside a warp (shuffle) and never needs a
 *     cross-warp scalar reduction. The block produces its own softmax
 *     statistics (m, l) plus the unnormalised weighted sum o, written to
 *     global scratch.
 *
 *   merge kernel    grid = [N, H_q], BLOCK_DIM threads
 *     Combines the per-chunk results with the standard online-softmax merge:
 *       m = max_c m_c
 *       l = Σ_c l_c · exp(m_c − m)
 *       o = Σ_c o_c · exp(m_c − m)
 *       out = o / l
 *
 *   Splitting the context this way also spreads long contexts over many more
 *   blocks than a single-block-per-head scan would.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Tensor layouts (matching Gemma4PagedKVCacheManager):
 *   Q:            [N, H_q, D]
 *   K_cache:      [num_blocks, block_size, H_kv, D]
 *   V_cache:      [num_blocks, block_size, H_kv, D]
 *   block_table:  [N, max_blocks]  int32
 *   context_lens: [N]              int32
 *   out:          [N, H_q, D]
 *
 * GQA: kv_head = q_head / (H_q / H_kv).
 * Sliding window: window_size > 0 keeps only the most recent ``window_size``
 * positions, which is equivalent to masking older keys for a single query.
 *
 * Output dtype is fp32 for the partial buffers (accuracy of the merge) and
 * the input dtype for the final result.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <float.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <limits>

// ─────────────────────────────────────────────────────────────────────────────
// 编译时常量
// ─────────────────────────────────────────────────────────────────────────────
#define G4_MAX_BLOCK_DIM 256   // threads per block
#define G4_KV_TILE       128   // KV positions staged per online-softmax step

// ─────────────────────────────────────────────────────────────────────────────
// 辅助：类型转换
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
__device__ __forceinline__ float g4_to_float(T v);
template<> __device__ __forceinline__ float g4_to_float(__half v)         { return __half2float(v); }
template<> __device__ __forceinline__ float g4_to_float(__nv_bfloat16 v)  { return __bfloat162float(v); }

template<typename T>
__device__ __forceinline__ T g4_from_float(float v);
template<> __device__ __forceinline__ __half        g4_from_float<__half>(float v)        { return __float2half(v); }
template<> __device__ __forceinline__ __nv_bfloat16 g4_from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// ─────────────────────────────────────────────────────────────────────────────
// warp 内归约
//
// 全掩码 shuffle，因此调用点必须在统一控制流上：如果部分 lane 已经跳过该
// 调用（例如放在 `if (tid < tile_count)` 里），__shfl_xor_sync 的行为未定义，
// 实测会直接挂死。
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float g4_warp_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    return v;
}

__device__ __forceinline__ float g4_warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// 主 kernel
// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
// Split-K 两段式实现
//
// 第一段（partial）：grid = [N, H_q, num_kv_blocks]，每个 block 只处理一段
//   KV（KV_CHUNK 个 token），一个 warp 负责一个 token。段内没有跨迭代的
//   barrier，也没有共享槽位复用，因此不存在前几版那种归约互相覆盖 /
//   条件屏障导致的死锁。每段输出自己的 softmax 统计量 (m, l) 和归一化前的
//   部分输出 o，写回 global memory。
//
// 第二段（merge）：grid = [N, H_q]，把各段按标准 online-softmax 合并：
//   m = max_b m_b ; l = Σ_b l_b * exp(m_b - m) ; o = Σ_b o_b * exp(m_b - m)
//   out = o / l
//
// 这个结构对长上下文也更友好：段之间并行，SM 利用率远高于单 block 扫全程。
// ─────────────────────────────────────────────────────────────────────────────
constexpr int KV_CHUNK = 128;   // 每个 partial block 处理的 token 数

template<typename T, int BLOCK_DIM>
__global__ void gemma4_attention_partial_kernel(
    const T*     __restrict__ Q,           // [N, H_q, D]
    const T*     __restrict__ K_cache,     // [num_blocks, block_size, H_kv, D]
    const T*     __restrict__ V_cache,     // [num_blocks, block_size, H_kv, D]
    const int*   __restrict__ block_table, // [N, max_blocks]
    const int*   __restrict__ context_lens,// [N]
    float*       __restrict__ partial_o,   // [N, H_q, num_chunks, D] fp32
    float*       __restrict__ partial_m,   // [N, H_q, num_chunks]
    float*       __restrict__ partial_l,   // [N, H_q, num_chunks]
    int H_q, int H_kv, int D,
    int block_size, int max_blocks,
    int window_size, float scale, int num_chunks)
{
    static_assert(BLOCK_DIM % 32 == 0, "BLOCK_DIM must be a multiple of the warp size");
    constexpr int WARPS = BLOCK_DIM / 32;

    const int batch_idx  = blockIdx.x;
    const int q_head_idx = blockIdx.y;
    const int chunk      = blockIdx.z;
    const int gqa_ratio  = H_q / H_kv;
    const int kv_head_idx = q_head_idx / gqa_ratio;
    const int tid        = threadIdx.x;
    const int lane       = tid & 31;
    const int warp       = tid >> 5;

    const int ctx_len = context_lens[batch_idx];
    const int start = (window_size > 0 && ctx_len > window_size) ? (ctx_len - window_size) : 0;
    const int num_kv = max(0, ctx_len - start);

    const int kv_begin = chunk * KV_CHUNK;
    const int kv_end   = min(num_kv, kv_begin + KV_CHUNK);

    const T* __restrict__ q_row = Q + ((long long)batch_idx * H_q + q_head_idx) * D;
    constexpr int MAX_NUM_D = (512 + BLOCK_DIM - 1) / BLOCK_DIM;

    float o_acc[MAX_NUM_D];
    float q_reg[MAX_NUM_D];
    #pragma unroll
    for (int j = 0; j < MAX_NUM_D; j++) {
        const int d = tid + j * BLOCK_DIM;
        o_acc[j] = 0.0f;
        q_reg[j] = (d < D) ? g4_to_float(q_row[d]) : 0.0f;
    }

    __shared__ float smem_max[WARPS];
    __shared__ float smem_sum[WARPS];

    const int t_local = tid % KV_CHUNK;                 // 本线程负责的段内 token
    const int pos = kv_begin + t_local;
    const bool active = (pos < kv_end);

    // ── 本 warp 负责的 token 的 logit（点积在 warp 内求和） ──
    float p = -FLT_MAX;
    if (active) {
        const int logical = start + pos;
        const int block_id = block_table[(long long)batch_idx * max_blocks + logical / block_size];
        const int slot     = logical % block_size;
        const T* k_row = K_cache + (((long long)block_id * block_size + slot) * H_kv + kv_head_idx) * D;
        float partial = 0.0f;
        #pragma unroll
        for (int j = 0; j < MAX_NUM_D; j++) {
            const int d = lane + j * 32;
            if (d < D) partial += q_reg[j] * g4_to_float(k_row[d]);
        }
        p = g4_warp_sum(partial) * scale;
    }

    // 段内最大 logit（-FLT_MAX 为中性元，空 lane 不影响结果）
    const float wmax = g4_warp_max(p);
    if (lane == 0) smem_max[warp] = wmax;
    __syncthreads();
    if (tid == 0) {
        float m = -FLT_MAX;
        #pragma unroll
        for (int wi = 0; wi < WARPS; wi++) m = fmaxf(m, smem_max[wi]);
        smem_max[0] = m;
    }
    __syncthreads();
    const float m_chunk = smem_max[0];

    const float w = active ? expf(p - m_chunk) : 0.0f;

    // ── 段内加权和 o = Σ w * v ──
    if (active) {
        const int logical = start + pos;
        const int block_id = block_table[(long long)batch_idx * max_blocks + logical / block_size];
        const int slot     = logical % block_size;
        const T* v_row = V_cache + (((long long)block_id * block_size + slot) * H_kv + kv_head_idx) * D;
        #pragma unroll
        for (int j = 0; j < MAX_NUM_D; j++) {
            const int d = tid + j * BLOCK_DIM;
            if (d < D) o_acc[j] += w * g4_to_float(v_row[d]);
        }
    }

    // ── 段内权重和 l = Σ w ──
    const float wsum = g4_warp_sum(w);
    if (lane == 0) smem_sum[warp] = wsum;
    __syncthreads();
    if (tid == 0) {
        float sw = 0.0f;
        #pragma unroll
        for (int wi = 0; wi < WARPS; wi++) sw += smem_sum[wi];
        smem_sum[0] = sw;
    }
    __syncthreads();
    const float l_chunk = smem_sum[0];

    // ── 写回本段结果 ──
    const long long head_base = ((long long)batch_idx * H_q + q_head_idx) * num_chunks + chunk;
    if (tid == 0) {
        partial_m[head_base] = (l_chunk > 0.0f) ? m_chunk : 0.0f;
        partial_l[head_base] = l_chunk;
    }
    float* __restrict__ o_out = partial_o + ((long long)batch_idx * H_q + q_head_idx) * num_chunks * D
                                         + (long long)chunk * D;
    #pragma unroll
    for (int j = 0; j < MAX_NUM_D; j++) {
        const int d = tid + j * BLOCK_DIM;
        if (d < D) o_out[d] = o_acc[j];
    }
}

template<typename T, int BLOCK_DIM>
__global__ void gemma4_attention_merge_kernel(
    const float* __restrict__ partial_o,   // [N, H_q, num_chunks, D]
    const float* __restrict__ partial_m,   // [N, H_q, num_chunks]
    const float* __restrict__ partial_l,   // [N, H_q, num_chunks]
    T*           __restrict__ out,         // [N, H_q, D]
    int N, int H_q, int D, int num_chunks)
{
    const int batch_idx  = blockIdx.x;
    const int q_head_idx = blockIdx.y;
    const int tid        = threadIdx.x;

    const long long head_base = ((long long)batch_idx * H_q + q_head_idx) * num_chunks;
    __shared__ float smem_m;
    __shared__ float smem_l;

    if (tid == 0) {
        float m = -FLT_MAX;
        for (int c = 0; c < num_chunks; c++) m = fmaxf(m, partial_m[head_base + c]);
        float l = 0.0f;
        for (int c = 0; c < num_chunks; c++) l += partial_l[head_base + c] * expf(partial_m[head_base + c] - m);
        smem_m = m;
        smem_l = l;
    }
    __syncthreads();
    const float inv_l = (smem_l > 0.0f) ? (1.0f / smem_l) : 0.0f;

    const float* __restrict__ po = partial_o + head_base * D;
    T* __restrict__ out_row = out + ((long long)batch_idx * H_q + q_head_idx) * D;
    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int c = 0; c < num_chunks; c++) {
            acc += po[(long long)c * D + d] * expf(partial_m[head_base + c] - smem_m);
        }
        out_row[d] = g4_from_float<T>(acc * inv_l);
    }
}

// Safe baseline kernel for Gemma4. One block owns one (request, query-head)
// pair and scans only the visible KV range. Every lane executes the same
// shuffle calls; inactive tokens are represented by a uniform loop bound
// instead of divergent warp participation. This is intentionally simple and
// serves as the correctness baseline before any split-K optimization.
template<typename T, int BLOCK_DIM>
__global__ void gemma4_attention_safe_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k_cache,
    const T* __restrict__ v_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ context_lens,
    T* __restrict__ out,
    int h_q,
    int h_kv,
    int d,
    int block_size,
    int max_blocks,
    int window_size)
{
    const int request = blockIdx.x;
    const int q_head = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr int WARPS = BLOCK_DIM / 32;
    constexpr int MAX_VALUES = (512 + BLOCK_DIM - 1) / BLOCK_DIM;

    const int context_len = context_lens[request];
    const int start = window_size > 0
        ? max(0, context_len - window_size)
        : 0;
    const int kv_head = q_head / (h_q / h_kv);
    const T* q_row = q + ((long long)request * h_q + q_head) * d;
    T* out_row = out + ((long long)request * h_q + q_head) * d;

    float output[MAX_VALUES] = {0.0f};
    __shared__ float warp_sums[WARPS];
    __shared__ float max_score;
    __shared__ float sum_score;
    if (tid == 0) {
        max_score = -FLT_MAX;
        sum_score = 0.0f;
    }
    __syncthreads();

    for (int logical = start; logical < context_len; ++logical) {
        const int physical_block =
            block_table[(long long)request * max_blocks + logical / block_size];
        const int slot = logical % block_size;
        const T* k_row = k_cache +
            (((long long)physical_block * block_size + slot) * h_kv + kv_head) * d;
        const T* v_row = v_cache +
            (((long long)physical_block * block_size + slot) * h_kv + kv_head) * d;

        float dot = 0.0f;
        #pragma unroll
        for (int j = 0; j < MAX_VALUES; ++j) {
            const int dim = tid + j * BLOCK_DIM;
            if (dim < d)
                dot += g4_to_float(q_row[dim]) * g4_to_float(k_row[dim]);
        }
        dot = g4_warp_sum(dot);
        if (lane == 0)
            warp_sums[warp] = dot;
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            #pragma unroll
            for (int w = 0; w < WARPS; ++w)
                total += warp_sums[w];
            warp_sums[0] = total;
        }
        __syncthreads();
        const float score = warp_sums[0];  // Gemma4 reference scale is 1.0.

        // Online softmax needs a scalar max/sum shared by every lane. Since
        // this kernel processes one token at a time, retain the stable
        // recurrence in two shared scalars.
        if (tid == 0) {
            const float old_max = max_score;
            const float new_max = fmaxf(old_max, score);
            const float old_factor = old_max == -FLT_MAX
                ? 0.0f : expf(old_max - new_max);
            const float current = expf(score - new_max);
            max_score = new_max;
            sum_score = old_factor * sum_score + current;
            warp_sums[0] = old_factor;
            warp_sums[1] = current;
        }
        __syncthreads();
        const float old_factor = warp_sums[0];
        const float current = warp_sums[1];
        for (int j = 0; j < MAX_VALUES; ++j) {
            const int dim = tid + j * BLOCK_DIM;
            if (dim < d) {
                output[j] = old_factor * output[j]
                    + current * g4_to_float(v_row[dim]);
            }
        }
        __syncthreads();
    }

    const float inv_sum = sum_score > 0.0f ? 1.0f / sum_score : 0.0f;
    for (int j = 0; j < MAX_VALUES; ++j) {
        const int dim = tid + j * BLOCK_DIM;
        if (dim < d)
            out_row[dim] = g4_from_float<T>(output[j] * inv_sum);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Python 绑定
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor gemma4_decode_attention(
    torch::Tensor q,               // [N, H_q, D] fp16/bf16
    torch::Tensor k_cache,         // [num_blocks, block_size, H_kv, D]
    torch::Tensor v_cache,         // [num_blocks, block_size, H_kv, D]
    torch::Tensor block_table,     // [N, max_blocks] int32
    torch::Tensor context_lens,    // [N] int32
    int64_t window_size)
{
    TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda(),
                "gemma4_decode_attention: q/k_cache/v_cache must be CUDA tensors");
    TORCH_CHECK(block_table.is_cuda() && context_lens.is_cuda(),
                "gemma4_decode_attention: metadata must be CUDA tensors");
    TORCH_CHECK(q.dtype() == torch::kFloat16 || q.dtype() == torch::kBFloat16,
                "gemma4_decode_attention: q must be fp16/bf16");
    TORCH_CHECK(k_cache.dtype() == q.dtype() && v_cache.dtype() == q.dtype(),
                "gemma4_decode_attention: k/v cache dtype must match q");
    TORCH_CHECK(q.dim() == 3 && k_cache.dim() == 4 && v_cache.dim() == 4,
                "gemma4_decode_attention: expected q [N,H_q,D] and caches [blocks,bs,H_kv,D]");

    const c10::cuda::CUDAGuard device_guard(q.device());
    TORCH_CHECK(k_cache.device() == q.device() && v_cache.device() == q.device() &&
                    block_table.device() == q.device() &&
                    context_lens.device() == q.device(),
                "gemma4_decode_attention: all tensors must be on the same device");
    auto q_c = q.contiguous();
    auto k_c = k_cache.contiguous();
    auto v_c = v_cache.contiguous();
    auto bt  = block_table.to(torch::kInt32).contiguous();
    auto ctx = context_lens.to(torch::kInt32).contiguous();

    const int N    = (int)q_c.size(0);
    const int H_q  = (int)q_c.size(1);
    const int D    = (int)q_c.size(2);
    const int H_kv = (int)k_c.size(2);
    const int block_size = (int)k_c.size(1);
    const int max_blocks = (int)bt.size(1);

    TORCH_CHECK(H_kv > 0 && H_q % H_kv == 0, "gemma4_decode_attention: H_q must be a multiple of H_kv");
    TORCH_CHECK(k_c.size(0) == v_c.size(0) && k_c.size(3) == D && v_c.size(3) == D,
                "gemma4_decode_attention: cache shape mismatch");
    TORCH_CHECK(k_c.size(1) == v_c.size(1) && k_c.size(2) == v_c.size(2),
                "gemma4_decode_attention: K/V cache shape mismatch");
    TORCH_CHECK(bt.size(0) == N && ctx.size(0) == N,
                "gemma4_decode_attention: block_table/context_lens must have N rows");
    TORCH_CHECK(D <= 512, "gemma4_decode_attention: head_dim above 512 is not supported");

    auto out = torch::empty_like(q_c);
    if (N == 0 || H_q == 0) return out;

    const int win = (int)window_size;
    TORCH_CHECK(win >= 0, "gemma4_decode_attention: window_size must be non-negative");
    TORCH_CHECK(D == 256 || D == 512,
                "gemma4_decode_attention: expected head_dim 256 or 512");
    TORCH_CHECK(ctx.numel() == 0 || ctx.max().item<int>() <=
                    max_blocks * block_size,
                "gemma4_decode_attention: context length exceeds block table capacity");
    auto out = torch::empty_like(q_c);
    const int block_dim = D == 512 ? 256 : 128;
    dim3 grid((unsigned)N, (unsigned)H_q);
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (q_c.dtype() == torch::kFloat16) {
        if (block_dim == 128)
            gemma4_attention_safe_kernel<__half, 128><<<grid, 128, 0, stream>>>(
                reinterpret_cast<const __half*>(q_c.data_ptr()),
                reinterpret_cast<const __half*>(k_c.data_ptr()),
                reinterpret_cast<const __half*>(v_c.data_ptr()),
                bt.data_ptr<int>(), ctx.data_ptr<int>(),
                reinterpret_cast<__half*>(out.data_ptr()),
                H_q, H_kv, D, block_size, max_blocks, win);
        else
            gemma4_attention_safe_kernel<__half, 256><<<grid, 256, 0, stream>>>(
                reinterpret_cast<const __half*>(q_c.data_ptr()),
                reinterpret_cast<const __half*>(k_c.data_ptr()),
                reinterpret_cast<const __half*>(v_c.data_ptr()),
                bt.data_ptr<int>(), ctx.data_ptr<int>(),
                reinterpret_cast<__half*>(out.data_ptr()),
                H_q, H_kv, D, block_size, max_blocks, win);
    } else {
        if (block_dim == 128)
            gemma4_attention_safe_kernel<__nv_bfloat16, 128><<<grid, 128, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(q_c.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(k_c.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(v_c.data_ptr()),
                bt.data_ptr<int>(), ctx.data_ptr<int>(),
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                H_q, H_kv, D, block_size, max_blocks, win);
        else
            gemma4_attention_safe_kernel<__nv_bfloat16, 256><<<grid, 256, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(q_c.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(k_c.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(v_c.data_ptr()),
                bt.data_ptr<int>(), ctx.data_ptr<int>(),
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                H_q, H_kv, D, block_size, max_blocks, win);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
