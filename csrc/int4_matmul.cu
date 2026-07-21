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
 * Thread layout:
 *   grid:  [B, (N + 127) / 128]
 *   block: 128 threads (4 warps)
 *
 * Each block processes 128 output dimensions at a time.
 * Each thread owns 2 output dimensions (tid, tid+64).
 * The K dimension is blocked into K_TILE=64 tiles.
 *
 * INT4 packed format:
 *   2 consecutive output rows share 1 byte: byte = (w[2n+1]<<4) | (w[2n]&0xF)
 *   where each w-value is unsigned [0,15] offset from signed [-8,7].
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

// warp reduce helpers (inline in this file to avoid extra dependency)
__device__ __forceinline__ float _warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
}

// ───────────────────────────────────────────────────────────────────
// Main kernel
// ───────────────────────────────────────────────────────────────────
template<typename T>
__global__ void int4_dequant_matmul_kernel(
    const T*        __restrict__ x,            // [B, K] fp16
    const uint8_t*  __restrict__ w_packed,     // [N//2, K] uint8 (2 INT4/byte)
    const T*        __restrict__ group_scales, // [N, K/group_size] fp16
    T*              __restrict__ y,            // [B, N] fp16
    int B, int N, int K, int group_size,
    int use_awq,                               // 1 = apply AWQ pre-scale (via x)
    int K_TILE)                                // compile-time tile: 64
{
    int batch_idx = blockIdx.x;
    int n_block   = blockIdx.y;       // which 128-out slice
    int tid       = threadIdx.x;      // 0..127

    int N_START = n_block * 128;
    int N_THIS  = (N_START + 128 <= N) ? 128 : (N - N_START);
    if (N_THIS <= 0) return;

    // ── pointers ──
    const T* x_ptr = x + batch_idx * (long long)K;
    T* y_ptr       = y + batch_idx * (long long)N + N_START;

    // ── shared memory: weight tile ──
    // [128 out, K_TILE/2 packed bytes] = 128 × 32 = 4 KB
    __shared__ uint8_t smem_w[128][32];  // 32 = K_TILE/2 = 64/2

    // ── registers: accumulator ──
    float acc[2] = {0.0f, 0.0f};
    int out0 = tid;          // 0..127
    int out1 = tid + 64;     // 64..191 (may exceed N_THIS)

    // ── iterate over K ──
    for (int k_start = 0; k_start < K; k_start += K_TILE) {
        int K_THIS = (k_start + K_TILE <= K) ? K_TILE : (K - k_start);

        // ---- cooperative load weight tile ----
        // Each thread loads 8 uint8 values (float2) → 16 bytes
        int k_bytes = (K_THIS + 1) / 2;  // K_THIS INT4 values → K_THIS/2 bytes
        for (int i = tid; i < N_THIS * k_bytes; i += blockDim.x) {
            int out_off = i / k_bytes;
            int k_off   = i % k_bytes;
            int out_row = N_START + out_off;
            smem_w[out_off][k_off] = w_packed[(out_row / 2) * K + (k_start / 2) + k_off];
        }
        __syncthreads();

        // ---- load x for this K tile ----
        float x_regs[2] = {0.0f, 0.0f};  // 2 fp16 values per iteration
        // Stride: each thread loads 8 half values at a time (float4)

        // ---- compute dot products ----
        // For each k in [0, K_THIS) step 4 (process 4 INT4 values = 2 bytes per output)
        for (int kb = 0; kb < k_bytes; kb += 4) {
            int k0 = kb * 2;  // first K dimension index in this block
            int k_limit = min(4, k_bytes - kb);

            // Load activation fragment
            float2 xf2 = *reinterpret_cast<const float2*>(x_ptr + k_start + k0);
            float x_vals[4];
            x_vals[0] = xf2.x; x_vals[1] = xf2.y;
            xf2 = *reinterpret_cast<const float2*>(x_ptr + k_start + k0 + 2);
            x_vals[2] = xf2.x; x_vals[3] = xf2.y;

            #pragma unroll
            for (int ki = 0; ki < k_limit && (k0 + ki) * 2 < K_THIS; ki++) {
                int k_idx = kb + ki;

                // Unpack weights for out0 and out1
                uint8_t w_byte = smem_w[out0][k_idx];
                int w_low  = (int)(w_byte & 0x0F) - 8;  // signed [-8,7]
                int w_high = (int)(w_byte >> 4)   - 8;

                // Group scale lookup
                int g0 = (k_start + k_idx * 2) / group_size;
                int g1 = g0;
                float s0 = __half2float(group_scales[out0 * (K / group_size) + g0]);
                float s1 = (out1 < N_THIS)
                         ? __half2float(group_scales[out1 * (K / group_size) + g1]) : 0.0f;

                float w0a = (float)w_low  * s0;
                float w0b = (float)w_high * s0;
                // Unpack for out1 (if in range)
                float w1a = 0.0f, w1b = 0.0f;
                if (out1 < N_THIS) {
                    uint8_t w_byte1 = smem_w[out1][k_idx];
                    int w1_low  = (int)(w_byte1 & 0x0F) - 8;
                    int w1_high = (int)(w_byte1 >> 4)   - 8;
                    w1a = (float)w1_low  * s1;
                    w1b = (float)w1_high * s1;
                }

                float xa = __half2float(x_vals[ki * 2]);
                float xb = __half2float(x_vals[ki * 2 + 1]);

                acc[0] += xa * w0a + xb * w0b;
                if (out1 < N_THIS)
                    acc[1] += xa * w1a + xb * w1b;
            }
        }
        __syncthreads();
    }

    // ── warp reduce (dot partial sums from different threads) ──
    // Each thread computed partial dot for 2 outputs; they are NOT shared
    // across threads because each thread handles different output dims.
    // Warp reduce is needed only if multiple threads contribute to one output.
    // With current layout: 1 thread = 2 outputs, no sharing. Skip reduce.

    // ── write output ──
    if (out0 < N_THIS)
        y_ptr[out0] = __float2half(acc[0]);
    if (out1 < N_THIS)
        y_ptr[out1] = __float2half(acc[1]);
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
    TORCH_CHECK(group_scales.dtype() == x.dtype(), "scales dtype must match x");

    // Handle arbitrary batch dimensions by flattening
    auto orig_shape = x.sizes();
    int64_t batch_elems = 1;
    for (int64_t d = 0; d < (int64_t)orig_shape.size() - 1; d++)
        batch_elems *= orig_shape[d];
    int K = orig_shape.back().as_int().value_or(0);
    int N = w_packed.size(0) * 2;  // 2 output dims per packed row

    TORCH_CHECK(x.numel() == batch_elems * K, "x shape mismatch");
    TORCH_CHECK(w_packed.size(1) == K, "w_packed K dim must match x last dim");
    TORCH_CHECK(group_scales.size(0) == N, "group_scales N must match unpacked N");
    TORCH_CHECK(group_scales.size(1) == (K + group_size - 1) / group_size,
                "group_scales K dim mismatch");

    auto x_2d = x.reshape({batch_elems, K});
    auto y = torch::empty({batch_elems, N}, x.options());

    dim3 grid((unsigned)batch_elems, (unsigned)((N + 127) / 128));
    int K_TILE = 64;

    if (x.dtype() == torch::kFloat16) {
        int4_dequant_matmul_kernel<__half><<<grid, 128>>>(
            reinterpret_cast<const __half*>(x_2d.data_ptr()),
            w_packed.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(group_scales.data_ptr()),
            reinterpret_cast<__half*>(y.data_ptr()),
            (int)batch_elems, N, K, group_size,
            0, K_TILE
        );
    } else {
        int4_dequant_matmul_kernel<__nv_bfloat16><<<grid, 128>>>(
            reinterpret_cast<const __nv_bfloat16*>(x_2d.data_ptr()),
            w_packed.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(group_scales.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
            (int)batch_elems, N, K, group_size,
            0, K_TILE
        );
    }

    // Restore original batch shape
    if (orig_shape.size() > 2) {
        std::vector<int64_t> out_shape(orig_shape.begin(), orig_shape.end() - 1);
        out_shape.push_back(N);
        y = y.reshape(out_shape);
    }
    return y;
}
