"""构建 paged_kv_cache_ext._C CUDA 扩展。

标准 PyTorch CUDA 扩展布局（参考 eole）：
- csrc/              → 纯 C/CUDA 源码，无 __init__.py
- paged_kv_cache_ext/ → Python 包，导入编译产物 _C

构建命令：
    uv run python setup.py build_ext --inplace
"""


from setuptools import setup, find_packages

def get_ext_modules_and_cmdclass():
    try:
        import torch
    except ImportError:
        print("WARNING: torch 不可用，跳过 CUDA 扩展构建")
        return [], {}

    if not torch.cuda.is_available():
        print("WARNING: CUDA 不可用，跳过 CUDA 扩展构建")
        return [], {}

    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    # 自动检测 GPU 计算能力
    arch_flags = []
    try:
        major, minor = torch.cuda.get_device_capability()
        arch = major * 10 + minor
        arch_flags += [
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
            f"-gencode=arch=compute_{arch},code=compute_{arch}",
        ]
    except Exception:
        arch_flags += [
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_80,code=compute_80",
        ]

    cxx_args = ["-O3", "-std=c++17"]
    nvcc_args = [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "--expt-relaxed-constexpr",
    ] + arch_flags

    ext_modules = [
        CUDAExtension(
            name="paged_kv_cache_ext._C",
            sources=["csrc/store_kvcache.cu"],
            extra_compile_args={"cxx": cxx_args, "nvcc": nvcc_args},
        ),
    ]

    return ext_modules, {"build_ext": BuildExtension}


ext_modules, cmdclass = get_ext_modules_and_cmdclass()

setup(
    name="paged_kv_cache_ext",
    version="0.2.0",
    description="High-performance Paged KV Cache Store — torch.compile compatible",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.14",
    install_requires=["torch"],
)