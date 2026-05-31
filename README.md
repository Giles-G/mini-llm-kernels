# mini-llm-kernels

自定义 CUDA kernel 库，为 [mini-serve-llm](../mini-serve-llm) Stage 8 提供推理加速。

## 优化内容

| 里程碑 | 优化 | 效果 |
|--------|------|------|
| M3 | RMSNorm + Residual Add 融合 | 每层减少 1 次 HBM 读写 |
| M4 | Block-aware Decode Attention | 消除 KV gather 拷贝 + online softmax |
| M2 | QKV Projection 融合（Python 层） | 3 次 GEMM → 1 次大 GEMM |

## 安装

### 有 CUDA 环境（A100 / RTX 30xx / RTX 40xx）

```bash
cd /path/to/mini-llm-kernels
pip install -e .
```

编译时自动检测 `nvcc`，支持 `sm_80`（A100）、`sm_86`（RTX 30xx）、`sm_89`（RTX 40xx）。

验证安装：

```bash
python -c "import mini_llm_kernels; print('ok'); print('_HAS_CUDA_OPS =', mini_llm_kernels._HAS_CUDA_OPS)"
# 有 CUDA 编译时输出：ok，_HAS_CUDA_OPS = True
```

### M1/CPU 环境（无 CUDA）

```bash
pip install -e .
# [mini-llm-kernels] No CUDA detected — installing Python-only package (fallback mode).
```

此时 `_HAS_CUDA_OPS = False`，主仓库自动退回纯 PyTorch 实现，行为与 Stage 7 完全相同。

## 在 mini-serve-llm 中使用

安装后，主仓库的 `nn_ops.py` 会自动检测并启用 kernel：

```python
# miniservellm/runtime/nn_ops.py（自动）
import mini_llm_kernels as _mkl
_HAS_CUSTOM_KERNELS = _mkl._HAS_CUDA_OPS  # True / False
```

### 强制关闭 kernel（A/B 对比用）

```bash
# 方式一：环境变量
MINI_LLM_NO_CUSTOM_KERNELS=1 python scripts/bench_engine.py ...

# 方式二：bench_engine.py 参数
python scripts/bench_engine.py --no-custom-kernels ...

# 方式三：bench_compare.py A/B 对比
python scripts/bench_compare.py --kernel-ab --batch-list 1,4,8 --max-new 64 --runs 2
```

## 目录结构

```
mini-llm-kernels/
├── csrc/
│   ├── add_tensors.cu       # Hello World kernel（M1 编译链路验证）
│   ├── fused_norm.cu        # M3: fused_add_rms_norm（fp16/bf16）
│   └── decode_attention.cu  # M4: block-aware decode attention（fp16/bf16，GQA）
├── mini_llm_kernels/
│   ├── __init__.py          # 包入口，导出 fused_add_rms_norm / decode_paged_attention
│   └── kernels/
│       ├── fused_norm.py        # M3 Python 绑定 + PyTorch fallback
│       └── decode_attention.py  # M4 Python 绑定 + PyTorch fallback
├── tests/
│   ├── test_import.py           # M1 验收：import 正常
│   └── test_integration_m5.py  # M5 集成验收：全路径验证
└── setup.py
```

## 运行集成测试

```bash
# 在 mini-serve-llm 目录下（因为测试依赖 miniservellm）
cd /path/to/mini-serve-llm
python /path/to/mini-llm-kernels/tests/test_integration_m5.py
```

期望输出：`6/6 tests passed`
