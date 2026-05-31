一、背景与现状
项目信息
* 主仓库：/Users/gengzhiqiang/User_Program/mini-serve-llm
    * 本地位置：/Users/gengzhiqiang/User_Program/mini-serve-llm

* 目标新仓库：/Users/gengzhiqiang/User_Program/mini-llm-kernels（Stage 8 新建）
    * 本地位置：


已实现功能（Stage 1-7 累计）
阶段
核心功能
Stage 1-3
基础推理引擎、HF 模型加载、单请求推理
Stage 4
手写 Transformer 前向（脱离 HF forward），权重直接操作
Stage 5
Paged KV Cache（block 分配、block_table 映射）
Stage 6
连续批处理（Batched Prefill/Decode）、chunked prefill、调度稳定性
Stage 7
分桶策略（shape 稳定）、torch.compile 接入（MPS fallback）
当前关键代码路径
miniservellm/
├── runtime/
│   ├── nn_ops.py            ← 基础算子（rms_norm/silu_and_mul/linear/rope）
│   ├── model_runner.py      ← Transformer 前向核心（TransformerBlockRunner）
│   └── inference_engine.py  ← 引擎主循环
├── cache/kv_cache.py        ← Paged KV Cache 管理
├── scheduler/scheduler.py   ← 调度器（batching/preemption）
└── config.py                ← 引擎配置
当前性能瓶颈（Stage 7 基准，MPS 设备，Qwen2.5-0.5B）
场景
吞吐量
batch=1, decode
~43 tok/s
batch=8, decode（分桶后）
~304 tok/s
batch=16, decode（分桶后）
~200 tok/s

二、当前执行路径与瓶颈分析
flowchart TD
    A["输入 hidden_states [N, hidden]"] --> B["rms_norm()"]
    B --> C["residual 保存"]
    C --> D["q_proj(x) — GEMM 1"]
    C --> E["k_proj(x) — GEMM 2"]
    C --> F["v_proj(x) — GEMM 3"]
    D & E & F --> G["apply_rope()"]
    G --> H["write_kv_for_tokens_indexed()"]
    H --> I["advanced indexing gather\nk_cache[layer, block_ids, offsets]\n⚠️ 额外 HBM 拷贝"]
    I --> J["gathered_paged_kv_decode_attention()\n标准 QK^T matmul → N×N 矩阵\n⚠️ 全量显存读写"]
    J --> K["o_proj()"]
    K --> L["residual + attn_out\n⚠️ 独立 add kernel"]
    L --> M["rms_norm() — 独立调用\n⚠️ 再次读写 hidden_states"]
    M --> N["MLP: gate_proj + up_proj + silu_and_mul + down_proj"]
    N --> O["residual + mlp_out\n⚠️ 独立 add kernel"]

    style I fill:#ffcccc
    style J fill:#ffcccc
    style L fill:#ffe0b2
    style M fill:#ffe0b2
    style D fill:#fff9c4
    style E fill:#fff9c4
    style F fill:#fff9c4
红色 = 主要瓶颈，橙色 = 次要瓶颈，黄色 = 可优化项

三、Stage 8 优化目标
Stage 8 实施三项优化，分别对应不同的性能瓶颈：
graph LR
    A["Stage 8 优化"] --> B["优化1\nBlock-aware Attention Kernel\n消除 gather 拷贝 + online softmax"]
    A --> C["优化2\nRMSNorm + Residual 融合\n减少 HBM 读写次数"]
    A --> D["优化3\nQKV 权重合并\n3次GEMM → 1次大GEMM"]
    B --> E["新仓库\nmini-llm-kernels\n(CUDA kernel)"]
    C --> E
    D --> F["主仓库改动\n(Python 层)"]

四、优化一：Block-aware Decode Attention Kernel
问题所在
model_runner.py:556-569 的 decode attention 路径：
# 当前：先 gather（物化 KV 到连续内存）
k_padded = self.kv_cache_manager.k_cache[
    self.layer_idx, gather_ctx.block_ids, gather_ctx.block_offsets
]  # ← advanced indexing = 实际数据拷贝到新 tensor
v_padded = ...

# 再做标准 attention
attn_out = gathered_paged_kv_decode_attention(q, k_padded, v_padded, ...)
两个问题：
1. k_cache[..., block_ids, block_offsets] 是一次显存拷贝，KV 从分页物理地址复制到连续 tensor
2. gathered_paged_kv_decode_attention 内部要生成完整的 [N, ctx_len] scores 矩阵，HBM 读写量 O(N×ctx_len)

目标：单个 CUDA Kernel 完成所有步骤
sequenceDiagram
    participant CPU
    participant HBM as GPU HBM（显存）
    participant SM as GPU SM（shared memory）

    Note over CPU,SM: 当前路径（两步）
    CPU->>HBM: 读 block_ids/block_offsets
    HBM->>HBM: gather：物理block → 连续k_padded/v_padded（拷贝）
    HBM->>SM: 读 q
    HBM->>SM: 读 k_padded/v_padded（连续）
    SM->>HBM: 写 scores [N, ctx_len]（全量）
    HBM->>SM: 读回 scores（softmax）
    SM->>HBM: 写 attn_out

    Note over CPU,SM: 目标路径（单 kernel）
    CPU->>HBM: 传 block_table
    HBM->>SM: 读 q（一次）
    loop 每个 KV block tile
        HBM->>SM: 按 block_table 直接读 K tile / V tile
        SM->>SM: online softmax 更新 m, l, o（寄存器）
    end
    SM->>HBM: 写 attn_out（一次）
算法（q_len=1 的简化 Flash Attention）
输入：Q [N, H_q, D], block_table [N, max_blocks], context_lens [N]
      K_cache [num_blocks, block_size, H_kv, D]

对每个 (batch_idx, head_idx) 并行执行：
  m = -inf, l = 0.0, o = zeros(D)   ← 存寄存器

  for block_i in range(num_blocks_for_request):
      block_id = block_table[batch_idx, block_i]
      加载 K_tile = K_cache[block_id, :, head_kv, :]  → shared memory
      加载 V_tile = V_cache[block_id, :, head_kv, :]  → shared memory

      scores = Q[batch_idx, head, :] · K_tile^T        # [block_size]
      scores[超出 context_len 的位置] = -inf            # mask

      m_new = max(m, max(scores))
      l_new = exp(m - m_new) * l + sum(exp(scores - m_new))
      o     = exp(m - m_new) * o + exp(scores - m_new) · V_tile

      m, l = m_new, l_new

  输出 o / l → attn_out[batch_idx, head, :]
代码位置
* 新文件：mini-llm-kernels/csrc/decode_attention.cu
* Python 绑定：mini-llm-kernels/kernels/decode_attention.py
* 主仓库调用点：nn_ops.py: gathered_paged_kv_decode_attention() 替换实现


五、优化二：RMSNorm + Residual Add 融合
问题所在
每个 Transformer Block 执行两次相同模式（model_runner.py:402-403, 455-458, 485-491）：
# 模式：读 hidden_states → norm → 读 hidden_states + attn_out → 写
residual = hidden_states           # 保存引用（无开销）
x = rms_norm(hidden_states, ...)   # kernel 1：读 hidden, 写 x_norm
hidden_states = residual + output  # kernel 2：读 residual + output, 写 hidden
# 下一子层重复一次
每层 2 次 × num_layers 层 = 大量碎片化 HBM 读写。
融合后的 CUDA Kernel
// 一个 kernel 完成：
// 输入：x [N, H], residual [N, H], gamma [H]
// 输出：x_normed [N, H], residual_out [N, H]（= x + residual，供下一层用）

__global__ void fused_add_rms_norm_kernel(
    const half* x, const half* residual, const half* gamma,
    half* x_normed, half* residual_out,
    int N, int H, float eps)
{
    // 1. residual_out = x + residual
    // 2. 对 residual_out 做 RMSNorm → x_normed
    // 使用 warp shuffle reduce 计算 mean(x²)
    float sum_sq = 0;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float val = __half2float(x[row*H+i]) + __half2float(residual[row*H+i]);
        residual_out[row*H+i] = __float2half(val);
        sum_sq += val * val;
    }
    // warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    float rms = rsqrtf(sum_sq / H + eps);
    // normalize + scale
    for (int i = ...) x_normed[...] = residual_out[...] * rms * gamma[i];
}
HBM 访问对比：

读次数
写次数
现在（2个kernel）
3次（hidden×2 + output×1）
2次
融合后（1个kernel）
2次（x + residual）
2次（norm_out + residual_out）
主仓库调用改动
# 现在（model_runner.py 每层两处）
residual = hidden_states
x = rms_norm(hidden_states, self.weights.input_layernorm, eps)
hidden_states = residual + attn_out

# 改后
x, hidden_states = fused_add_rms_norm(
    hidden_states, attn_out,          # x=hidden, residual=attn_out
    self.weights.input_layernorm, eps  # 同时更新 hidden_states
)

六、优化三：QKV Projection 融合
问题所在
model_runner.py:222-228 的 _project_qkv：
q = linear(x, self.weights.q_proj, ...)  # GEMM 1：x [N,H] × W_q [H_q,H]
k = linear(x, self.weights.k_proj, ...)  # GEMM 2：x [N,H] × W_k [H_k,H]
v = linear(x, self.weights.v_proj, ...)  # GEMM 3：x [N,H] × W_v [H_k,H]
x 被读取 3 次，3 个独立 kernel launch，cuBLAS 每次 GEMM 有调度开销。
改法（纯 Python，无需 CUDA Kernel）
Step 1：权重加载时合并（hf_loader.py 或 TransformerLayerWeights）：
# 加载时 concat
qkv_weight = torch.cat([
    state_dict["q_proj.weight"],  # [q_dim, H]
    state_dict["k_proj.weight"],  # [kv_dim, H]
    state_dict["v_proj.weight"],  # [kv_dim, H]
], dim=0)  # → [q_dim + kv_dim + kv_dim, H]
Step 2：_project_qkv 改为单次 GEMM + split：
def _project_qkv(self, x):
    qkv = linear(x, self.weights.qkv_proj)   # 1次 GEMM
    q_dim = self.num_q_heads * self.head_dim
    kv_dim = self.num_kv_heads * self.head_dim
    q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)  # split 几乎零开销
    q = q.view(x.shape[0], self.num_q_heads, self.head_dim)
    k = k.view(x.shape[0], self.num_kv_heads, self.head_dim)
    v = v.view(x.shape[0], self.num_kv_heads, self.head_dim)
    return q, k, v
注意：需处理 Qwen2 等模型带 bias 的情况，bias 同样 concat。

七、两仓库架构与交互
graph TB
    subgraph "mini-serve-llm（主仓库）"
        A["nn_ops.py\n_HAS_CUSTOM_KERNELS 开关"] -->|"有 kernel"| B["decode_paged_attention()"]
        A -->|"无 kernel / fallback"| C["原有 gather+attention 逻辑"]
        D["model_runner.py\nTransformerBlockRunner"] --> A
        E["config.py\nuse_custom_kernels=True/False"] --> A
    end

    subgraph "mini-llm-kernels（新仓库）"
        F["csrc/decode_attention.cu\nBlock-aware decode attention"]
        G["csrc/fused_norm.cu\nRMSNorm + residual add"]
        H["kernels/__init__.py\npybind11 Python 绑定"]
        F & G --> H
    end

    H -->|"pip install -e ."| A
    B --> F
    I["fused_add_rms_norm()"] --> G
fallback 逻辑（nn_ops.py 顶部）：
try:
    from mini_llm_kernels import decode_paged_attention, fused_add_rms_norm
    _HAS_CUSTOM_KERNELS = True
except ImportError:
    _HAS_CUSTOM_KERNELS = False
    # 自动退回原有 PyTorch 实现，MPS/CPU 环境正常运行

八、M1-M6 里程碑详细工作
M1：搭建 mini-llm-kernels 仓库框架
工作内容：
* 创建 /Users/gengzhiqiang/User_Program/mini-llm-kernels 仓库
* 初始化目录结构：csrc/, kernels/, tests/, setup.py
* 配置 setup.py（torch.utils.cpp_extension.CUDAExtension）
* 写一个 Hello World CUDA kernel 验证编译链路（add_tensors.cu）
* mini-serve-llm 的 nn_ops.py 加上 try/import 的 fallback 框架

验收：python -c "import mini_llm_kernels; print('ok')" 可以运行
预期吞吐：与 Stage 7 持平（本里程碑只搭框架）

M2：QKV Projection 融合（Python 层）
工作内容：
* TransformerLayerWeights 新增 qkv_proj 字段，废弃 q_proj/k_proj/v_proj
* hf_loader.py 权重加载逻辑：cat 三个权重矩阵
* model_runner.py: _project_qkv() 改为单次 linear + split
* 处理 bias（Qwen2 有 q/k/v bias，同样 cat）
* 跑 benchmark 验证输出正确性和性能

工作量：纯 Python，改动 3 个文件约 50 行
预期吞吐：
* batch=1：43 → ~55 tok/s（+28%，3次kernel launch → 1次）
* batch=8：304 → ~330 tok/s（+8%，大 batch GEMM 调度开销相对小）


M3：RMSNorm + Residual Add 融合 Kernel
工作内容：
* mini-llm-kernels/csrc/fused_norm.cu：实现 fused_add_rms_norm kernel
* warp shuffle reduce 计算 mean(x²)
* 支持 fp16/bf16 输入，fp32 中间计算
* 支持 batch（N 个 token 并行）
* kernels/fused_norm.py：pybind11 绑定
* nn_ops.py 新增 fused_add_rms_norm() 函数（带 fallback）
* model_runner.py 每层 2 处替换调用（prefill batch / decode batch 各 2 处）
* 单元测试：对比 PyTorch 实现数值误差 < 1e-3

核心 CUDA 难点：warp/block reduce（需要掌握 __shfl_xor_sync）
预期吞吐（累计 M2+M3）：
* batch=1：~55 → ~75 tok/s（+36%）
* batch=8：~330 → ~400 tok/s（+21%）


M4：Block-aware Decode Attention Kernel（核心）
工作内容：
* mini-llm-kernels/csrc/decode_attention.cu：主体 kernel
    * grid：[batch_size, num_q_heads]，每个线程块处理一个 (batch, head)
    * block：[WARP_SIZE × num_warps]，处理 head_dim 方向
    * shared memory：存 K_tile / V_tile（[block_size, head_dim]）
    * 寄存器：存 m/l/o（online softmax 状态）
    * 按 block_table 寻址，支持 GQA（head 分组映射）

* 边界处理：最后一个 block 可能不满，用 context_lens mask
* 支持 fp16/bf16
* 数值验证：与原 gather+attention 输出对齐
* 集成到 nn_ops.py: gathered_paged_kv_decode_attention()

核心难点：
* GQA 场景：Q head idx → KV head idx 的映射（head_kv = head_q // gqa_ratio）
* 最后一个不完整 block 的 mask 处理
* shared memory bank conflict 避免

预期吞吐（累计 M2+M3+M4）：
* batch=1：~75 → ~150-200 tok/s（+100-170%，消除 gather 拷贝收益最大）
* batch=8：~400 → ~600-800 tok/s（+50-100%）


M5：集成、Fallback 验证与端到端测试
工作内容：
* 完整 fallback 路径测试（use_custom_kernels=False 时原逻辑不受影响）
* MPS/CPU 环境验证（仍走 fallback，行为正确）
* bench_engine.py 新增 --use-custom-kernels 开关
* A/B 对比脚本（scripts/bench_compare.py 扩展）：kernel on vs off 吞吐对比
* 生成输出文本的正确性验证（和 Stage 7 输出对比）
* README 更新：说明 mini-llm-kernels 的安装步骤

预期吞吐：与 M4 持平（本里程碑不新增优化）

M6：Benchmark 对比与调优
工作内容：
* 系统性 benchmark：不同 batch size（1/4/8/16/32）× 不同 seq_len（128/512/1024/2048）
* Nsight Compute 分析 M4 kernel 的 occupancy、memory bandwidth 利用率
* 调优方向：
    * block_size 选择（32 vs 64 vs 128）
    * shared memory padding（避免 bank conflict）
    * 向量化访存（float4 128-bit load）

* 最终与 Stage 7（PyTorch 路径）的对比报告

预期吞吐（最终 Stage 8）：
场景
Stage 7
Stage 8 预期
batch=1, decode
~43 tok/s
~150-200 tok/s
batch=8, decode
~304 tok/s
~600-800 tok/s
batch=16, decode
~200 tok/s
~500-700 tok/s

九、整体里程碑进度与吞吐提升曲线
gantt
    title 第八阶段里程碑
    dateFormat  YYYY-MM-DD
    section 新仓库
    M1 仓库框架搭建       :m1, 2026-05-30, 3d
    section Python优化
    M2 QKV融合            :m2, after m1, 3d
    section CUDA Kernel
    M3 RMSNorm融合kernel  :m3, after m2, 5d
    M4 Decode Attention   :m4, after m3, 7d
    section 验证
    M5 集成与Fallback验证 :m5, after m4, 3d
    M6 Benchmark调优      :m6, after m5, 4d
吞吐提升预期（batch=8, CUDA GPU）：
Stage 7 基线:  ████░░░░░░░░░░░░░░  304 tok/s
M2 完成后:     █████░░░░░░░░░░░░░  330 tok/s  (+8%)
M3 完成后:     ███████░░░░░░░░░░░  400 tok/s  (+32%)
M4 完成后:     █████████████░░░░░  650 tok/s  (+114%)
M6 调优后:     ████████████████░░  750 tok/s  (+147%)
注：以上数据基于 CUDA GPU（A100/RTX 类）估算。MPS 设备因不支持自定义 CUDA kernel，全程走 fallback，性能与 Stage 7 持平。