"""
Paged KV Cache Store — 高性能 CUDA 扩展
=======================================

torch.compile 兼容的 in‑place 算子，将 key/value 张量按 slot_mapping
写入分页 KV cache。

用法::

    from paged_kv_cache_ext import store_kvcache_cuda

    # key, value:  [num_tokens, num_kv_heads, head_dim]
    # k_cache:     [num_blocks, block_size, num_kv_heads, head_dim]
    # slot_mapping: [num_tokens]  展平槽索引，-1 表示跳过
    store_kvcache_cuda(key, value, k_cache, v_cache, slot_mapping)

Stride‑aware：key/value 可以是非连续张量（如 .split()+.view() 的切片）。
Cache 必须连续。
"""

import torch


# 导入 _C 触发 store_kvcache.cu 中的 TORCH_LIBRARY 注册
try:
    from . import _C
except ImportError as e:
    raise ImportError(
        "无法导入编译后的 C++ 扩展 (_C)。"
        "请先运行 'python setup.py build_ext --inplace' 构建。"
    ) from e


def store_kvcache_cuda(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """
    将 key/value 存入分页 KV cache（CUDA，in‑place）。

    torch.compile 兼容 —— 通过 TORCH_LIBRARY 注册，含 Meta 后端实现。

    key/value 可以非连续；cache 必须连续。
    """
    # Cache 连续性诊断检查（Python 层，提供比 C++ 更友好的错误信息）
    if not k_cache.is_contiguous():
        raise RuntimeError(
            f"k_cache 必须连续。\n"
            f"  当前 stride: {tuple(k_cache.stride())}\n"
            f"  常见原因：k_cache 来自非连续操作（transpose / permute / 切片 / "
            f"对非连续张量调用 view）。\n"
            f"  修复：k_cache = k_cache.contiguous()"
        )
    if not v_cache.is_contiguous():
        raise RuntimeError(
            f"v_cache 必须连续。\n"
            f"  当前 stride: {tuple(v_cache.stride())}\n"
            f"  修复：v_cache = v_cache.contiguous()"
        )

    torch.ops.paged_kv_cache.paged_kv_cache_store(
        key, value, k_cache, v_cache, slot_mapping
    )


__all__ = ["store_kvcache_cuda"]