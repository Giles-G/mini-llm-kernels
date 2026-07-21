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
#include <cuda_bf16.h>
#include <torch/extension.h>

// type conversion helpers (templated for fp16 / bf16)
template<typename T>
__device__ __forceinline__ float _to_float(T v);
template<> __device__ __forceinline__ float _to_float(__half v)         { return __half2float(v); }
template<> __device__ __forceinline__ float _to_float(__nv_bfloat16 v)  { return __bfloat162float(v); }

template<typename T>
__device__ __forceinline__ T _from_float(float v);
template<> __device__ __forceinline__ __half        _from_float<__half>(float v)        { return __float2half(v); }
template<> __device__ __forceinline__ __nv_bfloat16 _from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

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
    // Each thread owns one output dimension. The packed row stores two
    // output dimensions, but that packing is only a storage format.
    int out0 = tid;          // 0..127

    // ── iterate over K ──
    for (int k_start = 0; k_start < K; k_start += K_TILE) {
        int K_THIS = (k_start + K_TILE <= K) ? K_TILE : (K - k_start);

        // ---- cooperative load weight tile ----
        // Each thread loads 8 uint8 values (float2) → 16 bytes
        int k_bytes = (K_THIS + 1) / 2;  // K_THIS INT4 values → K_THIS/2 bytes
        for (int i = tid; i < N_THIS * k_bytes; i += blockDim.x) {
            int out_off = i / k_bytes;
            int k_off   = i % k_bytes;
            // Each thread handles one output row; packed rows contain two output rows.
            int packed_row = (N_START + out_off) / 2;
            smem_w[out_off][k_off] = w_packed[packed_row * K + (k_start / 2) + k_off];
        }
        __syncthreads();

        // ---- load x for this K tile (fp16 → float, per element) ----
        // ---- compute dot products ----
        for (int kb = 0; kb < k_bytes; kb += 4) {
            int k_limit = min(4, k_bytes - kb);  // bytes to process

            #pragma unroll
            for (int ki = 0; ki < k_limit; ki++) {
                int k_idx = kb + ki;  // byte index in k_bytes
                int k_base = (k_idx * 2);  // K dim index

                if (k_base + 1 >= K_THIS) break;

                // Skip computation for out-of-range outputs (avoids reading uninitialized smem)
                if (out0 >= N_THIS) continue;

                // Load 2 activation values (fp16/bf16 → float)
                float xa = _to_float(x_ptr[k_start + k_base]);
                float xb = _to_float(x_ptr[k_start + k_base + 1]);

                // Unpack the selected output row's two consecutive K values.
                uint8_t w_byte0 = smem_w[out0][k_idx];
                int w0_low  = (int)(w_byte0 & 0x0F) - 8;
                int w0_high = (int)(w_byte0 >> 4)   - 8;

                // Group scale (use GLOBAL output index N_START + out0)
                int g_idx = (k_start + k_base) / group_size;
                float s0 = _to_float(group_scales[(N_START + out0) * (K / group_size) + g_idx]);

                float w0a = (float)w0_low  * s0;
                float w0b = (float)w0_high * s0;

                acc[0] += xa * w0a + xb * w0b;
            }
        }
        __syncthreads();
    }

    // ── warp reduce (dot partial sums from different threads) ──
    // Each thread owns one output row, so no cross-thread reduction is needed.

    // ── write output ──
    if (out0 < N_THIS)
        y_ptr[out0] = _from_float<T>(acc[0]);
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
    int K = (int)orig_shape.back();
    int N = w_packed.size(0) * 2;  // 2 output dims per packed row

    TORCH_CHECK(x.numel() == batch_elems * K, "x shape mismatch");
    TORCH_CHECK(w_packed.size(1) == K, "w_packed K dim must match x last dim");
    // The packed storage is [N/2, K], while group scales stay [N, K/groups].
    TORCH_CHECK(w_packed.size(0) * 2 == N, "packed weight row count mismatch");
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
            K_TILE
        );
    } else {
        int4_dequant_matmul_kernel<__nv_bfloat16><<<grid, 128>>>(
            reinterpret_cast<const __nv_bfloat16*>(x_2d.data_ptr()),
            w_packed.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(group_scales.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
            (int)batch_elems, N, K, group_size,
            K_TILE
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
