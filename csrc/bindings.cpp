/**
 * bindings.cpp — Python module registration for mini_llm_kernels._C
 *
 * All CUDA kernel functions are declared extern and registered here.
 * Without this file, the compiled .so has no PyInit__C symbol.
 */

#include <torch/extension.h>
#include <torch/csrc/utils/pybind.h>

// ── Declarations (defined in .cu files) ──
torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b);
std::pair<torch::Tensor, torch::Tensor> fused_add_rms_norm(
    torch::Tensor x, torch::Tensor residual, torch::Tensor gamma, float eps);
torch::Tensor decode_paged_attention(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor context_lens);
torch::Tensor decode_paged_attention_partitioned(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor context_lens, int num_partitions);
torch::Tensor paged_prefill_attention(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor chunk_lens, torch::Tensor history_lens);
torch::Tensor int4_dequant_matmul(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor group_scales,
    int group_size);
torch::Tensor int4_dequant_matmul_legacy(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor group_scales,
    int group_size);
torch::Tensor int4_dequant_matmul_pair(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor group_scales,
    int group_size);
torch::Tensor int4_dequant_matmul_pair_tuned(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor group_scales,
    int group_size, int pairs_per_block);
torch::Tensor int4_dequant_matmul_pair_vec(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor group_scales,
    int group_size);
torch::Tensor int4_dequant_dense(
    torch::Tensor w_packed, torch::Tensor group_scales,
    int64_t group_size, bool bf16);
torch::Tensor int4_dequant_matmul_pair_vec8(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor group_scales,
    int group_size);
torch::Tensor int4_dequant_matmul_pair_batch(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor group_scales,
    int group_size);
torch::Tensor int4_dequant_matmul_pair_batch16(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor group_scales,
    int group_size);
torch::Tensor int4_dequant_matmul_pair_batch_parallel(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor group_scales,
    int group_size);
torch::Tensor int4_dequant_matmul_tc(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor group_scales,
    int group_size);
torch::Tensor int4_dequant_matmul_tc_variant(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor group_scales,
    int group_size, int64_t variant);
torch::Tensor int4_prepack_tiles(torch::Tensor w_packed);
torch::Tensor int4_dequant_matmul_tc_tiled(
    torch::Tensor x, torch::Tensor w_tiled, torch::Tensor group_scales);
torch::Tensor int4_prepack_frag(torch::Tensor w_packed);
torch::Tensor int4_dequant_matmul_tc_frag(
    torch::Tensor x, torch::Tensor w_frag, torch::Tensor group_scales,
    int64_t parts);
torch::Tensor int4_dequant_matmul_tc_frag_ga(
    torch::Tensor x, torch::Tensor w_frag, torch::Tensor group_scales,
    int64_t parts);
torch::Tensor int4_dequant_matmul_tc_frag_warps(
    torch::Tensor x, torch::Tensor w_frag, torch::Tensor group_scales,
    int64_t parts, int64_t warps);
torch::Tensor int4_dequant_matmul_tc_frag_ldm(
    torch::Tensor x, torch::Tensor w_frag, torch::Tensor group_scales,
    int64_t parts, int64_t warps);
std::vector<int64_t> int4_pair_kernel_attributes(
    int pairs_per_block, bool fp16, bool vectorized);
std::vector<int64_t> tc_kernel_attributes(int64_t which);
torch::Tensor gemma4_rope_fused(
    torch::Tensor x, torch::Tensor cos, torch::Tensor sin);
torch::Tensor gemma4_gelu_mul(torch::Tensor gate, torch::Tensor up);
torch::Tensor gemma4_add_mul(
    torch::Tensor a, torch::Tensor b, torch::Tensor scalar);
std::vector<torch::Tensor> gemma4_qkv_norm_rope(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor qw, torch::Tensor kw, torch::Tensor vw,
    torch::Tensor cos, torch::Tensor sin, double eps);
std::vector<torch::Tensor> gemma4_qkv_norm_rope_warp(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor qw, torch::Tensor kw, torch::Tensor vw,
    torch::Tensor cos, torch::Tensor sin, double eps);
torch::Tensor gemma4_decode_attention(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor context_lens,
    int64_t window_size);
torch::Tensor gemma4_decode_attention_warpsplit(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor context_lens,
    int64_t window_size);
torch::Tensor gemma4_decode_attention_splitk(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor context_lens,
    int64_t window_size, torch::Tensor partial_o,
    torch::Tensor partial_m, torch::Tensor partial_l,
    int64_t chunk_size);

// ── Module registration ──
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_tensors", &add_tensors, "Element-wise tensor addition");
    m.def("fused_add_rms_norm", &fused_add_rms_norm,
          "Fused residual add + RMSNorm");
    m.def("decode_paged_attention", &decode_paged_attention,
          "Block-aware paged decode attention");
    m.def("decode_paged_attention_partitioned", &decode_paged_attention_partitioned,
          "KV-sequence-partitioned decode attention");
    m.def("paged_prefill_attention", &paged_prefill_attention,
          "Block-aware paged prefill attention");
    m.def("int4_dequant_matmul", &int4_dequant_matmul,
          "Fused INT4 dequantize + matmul");
    m.def("int4_dequant_matmul_legacy", &int4_dequant_matmul_legacy,
          "Legacy fused INT4 dequantize + matmul");
    m.def("int4_dequant_matmul_pair", &int4_dequant_matmul_pair,
          "Row-pair warp INT4 dequantize + matmul");
    m.def("int4_dequant_matmul_pair_tuned", &int4_dequant_matmul_pair_tuned,
          "Row-pair warp INT4 dequantize + matmul (tunable rows/block)");
    m.def("int4_dequant_matmul_pair_vec", &int4_dequant_matmul_pair_vec,
          "Vectorized row-pair INT4 dequantize + matmul");
    m.def("int4_dequant_dense", &int4_dequant_dense,
          "INT4 dequantize to dense FP16/BF16");
    m.def("int4_dequant_matmul_pair_vec8", &int4_dequant_matmul_pair_vec8,
          "8-byte row-pair INT4 decode GEMV");
    m.def("int4_dequant_matmul_pair_batch", &int4_dequant_matmul_pair_batch,
          "Weight-stationary 2..4 row INT4 decode GEMV");
    m.def("int4_dequant_matmul_pair_batch16", &int4_dequant_matmul_pair_batch16,
          "Weight-stationary 5..16 row INT4 decode GEMV");
    m.def("int4_dequant_matmul_pair_batch_parallel",
          &int4_dequant_matmul_pair_batch_parallel,
          "Parallel weight-stationary 1..4 row INT4 decode GEMV");
    m.def("int4_dequant_matmul_tc", &int4_dequant_matmul_tc,
          "Tensor-core tile-dequant 2..16 row INT4 decode matmul");
    m.def("int4_dequant_matmul_tc_variant", &int4_dequant_matmul_tc_variant,
          "Experimental tensor-core INT4 dispatch variant tuner");
    m.def("int4_prepack_tiles", &int4_prepack_tiles,
          "Repack INT4 weights into tile-major 512-byte blocks");
    m.def("int4_dequant_matmul_tc_tiled", &int4_dequant_matmul_tc_tiled,
          "Tensor-core INT4 matmul with tile-major packed weights");
    m.def("int4_prepack_frag", &int4_prepack_frag,
          "Repack INT4 weights into MMA B-fragment order");
    m.def("int4_dequant_matmul_tc_frag", &int4_dequant_matmul_tc_frag,
          "Tensor-core INT4 matmul with direct B-fragment registers");
    m.def("int4_dequant_matmul_tc_frag_ga", &int4_dequant_matmul_tc_frag_ga,
          "Direct-B fragment matmul with global A loads");
    m.def("int4_dequant_matmul_tc_frag_warps", &int4_dequant_matmul_tc_frag_warps,
          "Direct-B fragment matmul with 4 or 8 warps per block");
    m.def("int4_dequant_matmul_tc_frag_ldm", &int4_dequant_matmul_tc_frag_ldm,
          "Direct-B fragment matmul with swizzled ldmatrix A loads");
    m.def("int4_pair_kernel_attributes", &int4_pair_kernel_attributes,
          "Pair kernel register/smem/occupancy attributes");
    m.def("tc_kernel_attributes", &tc_kernel_attributes,
          "Tensor-core kernel register/smem/occupancy attributes");
    m.def("gemma4_rope_fused", &gemma4_rope_fused,
          "Fused FP16 Gemma4 rotary embedding");
    m.def("gemma4_gelu_mul", &gemma4_gelu_mul,
          "Fused FP16 GELU(tanh) multiply");
    m.def("gemma4_add_mul", &gemma4_add_mul,
          "Fused FP16 residual add and scalar multiply");
    m.def("gemma4_qkv_norm_rope", &gemma4_qkv_norm_rope,
          "Fused FP16 q/k/v RMSNorm and q/k RoPE");
    m.def("gemma4_qkv_norm_rope_warp", &gemma4_qkv_norm_rope_warp,
          "Warp-per-head fused FP16 q/k/v RMSNorm and q/k RoPE");
    m.def("gemma4_decode_attention", &gemma4_decode_attention,
          "Gemma4 paged decode attention (head_dim 256/512, sliding window)");
    m.def("gemma4_decode_attention_warpsplit", &gemma4_decode_attention_warpsplit,
          "Warp-split long-context Gemma4 decode attention");
    m.def("gemma4_decode_attention_splitk", &gemma4_decode_attention_splitk,
          "Global split-K Gemma4 decode attention (scratch supplied by caller)");
}
