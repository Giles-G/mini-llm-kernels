"""
setup.py for mini-llm-kernels

在有 CUDA 工具链的环境下编译自定义 kernel。
在 M1/CPU 环境下（无 nvcc），跳过 CUDA 扩展，仅安装 Python 包。

安装命令：
    pip install -e .           # 自动检测 CUDA
    FORCE_CUDA=1 pip install . # 强制编译 CUDA
"""

import os
import sys

from setuptools import setup, find_packages

# --------------------------------------------------------------------------- #
# CUDA 扩展（可选）
# --------------------------------------------------------------------------- #
ext_modules = []

def cuda_available() -> bool:
    """检查 nvcc 是否存在"""
    import shutil
    return shutil.which("nvcc") is not None or os.environ.get("FORCE_CUDA") == "1"

if cuda_available():
    try:
        from torch.utils.cpp_extension import CUDAExtension, BuildExtension

        cuda_ext = CUDAExtension(
            name="mini_llm_kernels._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/add_tensors.cu",
                "csrc/fused_norm.cu",
                "csrc/decode_attention.cu",
                "csrc/int4_matmul.cu",
            ],
            include_dirs=["csrc"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode", "arch=compute_80,code=sm_80",   # A100
                    "-gencode", "arch=compute_86,code=sm_86",   # RTX 30xx
                    "-gencode", "arch=compute_89,code=sm_89",   # RTX 40xx
                ],
            },
        )
        ext_modules.append(cuda_ext)
        print("[mini-llm-kernels] CUDA extension will be compiled.")
    except Exception as e:
        print(f"[mini-llm-kernels] WARNING: failed to configure CUDA extension: {e}")
else:
    print("[mini-llm-kernels] No CUDA detected — installing Python-only package (fallback mode).")

# --------------------------------------------------------------------------- #
# 包元信息
# --------------------------------------------------------------------------- #
setup(
    name="mini_llm_kernels",
    version="0.1.0",
    description="Custom CUDA kernels for mini-serve-llm (Stage 8)",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=["torch>=2.0"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": __import__("torch.utils.cpp_extension", fromlist=["BuildExtension"]).BuildExtension} if ext_modules else {},
)
