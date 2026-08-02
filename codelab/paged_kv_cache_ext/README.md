# Paged KV Cache Store — 从零构建 CUDA 扩展

## 0. Background

在 LLM 推理中，KV 缓存保存历史 token 的 key/value 状态以复用计算。
传统连续缓存会因序列长度不一造成严重内存碎片。**分页 KV 缓存 (Paged KV Cache)** 把缓存
划分成固定大小的块，通过页表动态映射，实现“按需分配、零碎回收”，大幅提升内存效率
（如 vLLM 的 PagedAttention）。

每轮推理产生的新 token 需要写入分页缓存。`store_kvcache` 操作就是做这件事：
根据 `slot_mapping`（展平的槽索引，`-1` 表示无效 token 不写入），
把形状为 `[num_tokens, num_heads, head_dim]` 的 key/value 张量，
分散写入 `[num_blocks, block_size, num_heads, head_dim]` 的 K/V 缓存张量。

**文件结构**

```
paged_kv_cache_ext/                  # 空项目根目录
├── csrc/                            # 纯 C/CUDA 源码，无 __init__.py
│   └── store_kvcache.cu             # kernel + dispatch + 算子注册
├── paged_kv_cache_ext/              # Python 包（面向用户）
│   ├── __init__.py                  # from . import _C → store_kvcache_cuda()
│   └── _C.cpython-3xx-...so         # 编译后生成 `_C.cpython-3xx-...so`。
├── .gitignore                       # gitignore
├── main.py                          # 测试 & 基准（torch / triton / cuda）
├── pyproject.toml                   # PEP 517 构建入口
├── README.md                        # 本文件
└── setup.py                         # CUDAExtension 编译配置
```

## 1. Init

```bash
$ mkdir paged_kv_cache_ext && cd paged_kv_cache_ext
$ uv init
$ uv add torch numpy
```

```toml
[project]
name = "paged-kv-cache-ext"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.14"
dependencies = [
    "numpy>=2.5.1",
    "torch>=2.13.0",
]

[[tool.uv.index]]
name = "tuna"
url = "https://pypi.tuna.tsinghua.edu.cn/simple/"
```

## 2. Test & Benchmark

测试三种实现（torch / triton / cuda）的精度一致性，然后跑三组规模的性能基准。

```python title="main.py"
import logging
logging.getLogger("torch._inductor.cudagraph_utils").setLevel(logging.ERROR)

import torch
import triton
import triton.language as tl

torch.manual_seed(2026)

# pure torch
def store_kvcache_torch(
    key: torch.Tensor, 
    value: torch.Tensor, 
    k_cache: torch.Tensor, 
    v_cache: torch.Tensor, 
    slot_mapping: torch.Tensor
):
    """
    Store key/value into paged KV cache using slot_mapping.
    Handles -1 (invalid slots) via masked scatter to avoid OOB access.
    
    Args:
        key: [num_tokens, num_kv_heads, head_dim]
        value: [num_tokens, num_kv_heads, head_dim]
        k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        slot_mapping: [num_tokens] flat indices into (num_blocks * block_size), -1 for invalid
    
    Demonstration: 
                                ↓ token 0 will be stored in slot 11
        slot_mapping:  tensor([11,  3, 13, 25, 23, 29, -1, 20, 15, 27, 26,  4, 10, 19, 14, -1]
                                                        ↑ token 6 won't be stored

        valid_mask:    tensor([True,True,True,True,True,True,False,True,True,True,True,True,True,True,True,False])
        valid_slots:   tensor([11,  3, 13, 25, 23, 29, 20, 15, 27, 26,  4, 10, 19, 14])

        num_blocks = 8, block_size = 4
        block_indices = valid_slots // block_size
        block_offsets = valid_slots % block_size

        block_indices: tensor([2, 0, 3, 6, 5, 7, 5, 3, 6, 6, 1, 2, 4, 3])
        block_offsets: tensor([3, 3, 1, 1, 3, 1, 0, 3, 3, 2, 0, 2, 3, 2])
                                   ↑
        For example: here, token 0 will be stored in slot 11, which is block 2, offset 3 in kvcache [2, 3, num_kv_heads, head_dim]
    """
    # Filter out invalid slots (-1)
    valid_mask = slot_mapping != -1
    if not valid_mask.any():
        return
    
    valid_slots = slot_mapping[valid_mask]
    valid_keys = key[valid_mask]
    valid_values = value[valid_mask]
    
    # Convert flat slot index to (block_idx, block_offset)
    block_size = k_cache.size(1)
    block_indices = valid_slots // block_size
    block_offsets = valid_slots % block_size

    # Direct indexing write - much faster than scatter for contiguous-ish access patterns
    k_cache[block_indices, block_offsets] = valid_keys
    v_cache[block_indices, block_offsets] = valid_values

    r"""
    # The following code is equivalent to the above
    #     ```python
    #     mask = slot_mapping != -1
    #     if not mask.any():
    #         return
    #     valid_slots = slot_mapping[mask]
    #     k_cache.view(-1, *key.shape[1:])[valid_slots] = key[mask]
    #     v_cache.view(-1, *value.shape[1:])[valid_slots] = value[mask]
    #     ```
    """

# v2 triton
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)

def store_kvcache_triton(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

# v3: CUDA kernel

from paged_kv_cache_ext import store_kvcache_cuda
# csrc/store_kvcache.cu → setup.py → paged_kv_cache_ext/_C.xxx.so → paged_kv_cache_ext/__init__.py

# Benchmark
def benchmark(fn, key, value, k_cache_init, v_cache_init, slot_mapping, warmup=5, repeat=100):

    for _ in range(warmup):
        fn(key, value, k_cache_init.clone(), v_cache_init.clone(), slot_mapping)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times = []

    for _ in range(repeat):
        k_tmp = k_cache_init.clone()
        v_tmp = v_cache_init.clone()
        torch.cuda.synchronize()
        start_event.record()
        fn(key, value, k_tmp, v_tmp, slot_mapping)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    return sorted(times)[len(times) // 2] # 返回中位数（毫秒）

if __name__ == "__main__":

    num_tokens = 16
    num_kv_heads = 4
    head_dim = 32

    num_blocks = 8
    block_size = 4

    key = torch.randn(num_tokens, num_kv_heads, head_dim, device="cuda")
    value = torch.randn(num_tokens, num_kv_heads, head_dim, device="cuda")

    num_slots = num_blocks * block_size 
    all_slots = torch.randperm(num_slots, device='cuda')[:num_tokens] # 随机选择无重复的槽
    mask_invalid = torch.rand(num_tokens, device='cuda') < 0.1        # 10% 设为无效
    slot_mapping = all_slots.clone()
    slot_mapping[mask_invalid] = -1 # 混入无效槽

    k_cache_init = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device='cuda')
    v_cache_init = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device='cuda')

    k1, v1 = k_cache_init.clone(), v_cache_init.clone()
    k2, v2 = k_cache_init.clone(), v_cache_init.clone()
    k3, v3 = k_cache_init.clone(), v_cache_init.clone()

    store_kvcache_torch(key, value, k1, v1, slot_mapping)
    print(f"K: {key.shape}")
    print(f"V: {value.shape}")
    print(f"K cache: {k1.shape}")
    print(f"V cache: {v1.shape}")
    print(f"slot mapping: {slot_mapping.shape}")
    print(f"{'token':>6} {'slot':>6} {'(block, offset)':>16} {'init':>12} {'key[token,0,0]':>16} {'k_cache[block,offset,0,0]':>25}")
    valid = slot_mapping != -1
    for tok in range(num_tokens):
        if valid[tok]:
            slot = slot_mapping[tok].item()
            blk = slot // block_size
            off = slot % block_size
            before = k_cache_init[blk, off, 0, 0].item()
            k_val = key[tok, 0, 0].item()
            cache_val = k1[blk, off, 0, 0].item()
            print(f"{tok:6d} {slot:6d} {str((blk, off)):>16} {before:16.6f} {k_val:16.6f} {cache_val:25.6f}")

    store_kvcache_triton(key, value, k2, v2, slot_mapping)
    print("K cache k1 vs k2:", torch.equal(k1, k2))
    print("V cache v1 vs v2:", torch.equal(v1, v2))
    cos = torch.nn.functional.cosine_similarity
    print("K cosine similarity:", cos(k1.flatten().float(), k2.flatten().float(), dim=0).item())
    print("V cosine similarity:", cos(v1.flatten().float(), v2.flatten().float(), dim=0).item())

    store_kvcache_cuda(key, value, k3, v3, slot_mapping)
    print("K cache k2 vs k3:", torch.equal(k2, k3))
    print("V cache v2 vs v3:", torch.equal(v2, v3))


    configs = [
        ("Small  (32x8x64)   ", 32, 8, 64, 16),
        ("Medium (4096x32x128)", 4096, 32, 128, 64),
        ("Large  (16384x8x128)", 16384, 8, 128, 128),
    ]
    print("\n" + "=" * 90)
    print("综合性能对比 (中位数时间 / ms)")
    print("=" * 90)
    header = f"{'配置':<25} {'v1 (torch)':>15} {'v2 (triton)':>12} {'v3 (cuda)':>12}"
    print(header)
    print("-" * 90)

    summary_data = []
    for name, num_tokens, num_kv_heads, head_dim, block_size in configs:
        num_slots_needed = num_tokens * 2
        num_blocks = (num_slots_needed + block_size - 1) // block_size
        num_slots = num_blocks * block_size
        key = torch.randn(num_tokens, num_kv_heads, head_dim, device="cuda")
        value = torch.randn(num_tokens, num_kv_heads, head_dim, device="cuda")
        k_cache_init = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device="cuda")
        v_cache_init = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device="cuda")
        all_slots = torch.randperm(num_slots, device="cuda")[:num_tokens]
        mask_invalid = torch.rand(num_tokens, device="cuda") < 0.1
        slot_mapping = all_slots.clone()
        slot_mapping[mask_invalid] = -1

        t1 = benchmark(store_kvcache_torch, key, value, k_cache_init, v_cache_init, slot_mapping)
        t2 = benchmark(store_kvcache_triton, key, value, k_cache_init, v_cache_init, slot_mapping)
        t3 = benchmark(store_kvcache_cuda, key, value, k_cache_init, v_cache_init, slot_mapping)
        summary_data.append((name, t1, t2, t3))

    for name, t1, t2, t3 in summary_data:
        print(f"{name:<25} {t1:15.4f} {t2:12.4f} {t3:12.4f}")
```

## 3. CUDA Extension

```cu title="csrc/store_kvcache.cu"
/**
 * =============================================================================
 *  Paged KV Cache Store — CUDA 内核实现
 * =============================================================================
 *
 * 将 key / value 张量写入分页 KV cache 的高性能、向量化 CUDA 内核。
 * 通过 TORCH_LIBRARY 注册算子，torch.compile 可完整追踪（无图断）。
 *
 *  内存布局与 stride 设计
 *  ----------------------
 *  key  :  [num_tokens,              num_kv_heads, head_dim]
 *  cache:  [num_blocks, block_size,  num_kv_heads, head_dim]
 *                                  └────── D ──────┘（展平）
 *
 *  单个 token 内部的 D 个元素始终连续（stride(2)=1, stride(1)=head_dim），
 *  但 token 之间的间距可能大于 D（当输入是 .split()+.view() 的切片时）。
 *  因此内核接受显式的 key_stride / value_stride，与 triton 的
 *  key_ptr + idx * key_stride + tl.arange(0, D) 模式对齐。
 *
 *  Cache 张量假设连续（模型自有 buffer，始终新分配）。
 */

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/library.h>


// ============================================================================
//  CUDA 内核
// ============================================================================

/**
 * 向量化分页 KV cache 存储内核（stride‑aware）。
 *
 * 每个线程块负责一个 token，块内线程协作复制该 token 的 D 个元素。
 * 使用 uint4（16 字节）向量化访存 + 字节级余数循环。
 *
 * @tparam scalar_t  张量 dtype 对应的 C++ 类型（float / half / bf16）
 * @param key_stride   token 间步长（元素数）。连续张量 = D；切片/视图可能更大
 * @param value_stride 同上
 */
template <typename scalar_t>
__global__ void paged_kv_cache_store_kernel(
    const scalar_t* __restrict__ key_ptr,
    const int              key_stride,
    const scalar_t* __restrict__ value_ptr,
    const int              value_stride,
    scalar_t* __restrict__ k_cache_ptr,
    scalar_t* __restrict__ v_cache_ptr,
    const int64_t* __restrict__ slot_mapping_ptr,
    const int num_tokens,
    const int D,
    const int block_size
) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    // 用共享内存读取一次 slot，供块内所有线程使用
    __shared__ int64_t shared_slot;
    if (threadIdx.x == 0) {
        shared_slot = slot_mapping_ptr[token_idx];
    }
    __syncthreads();

    int64_t slot = shared_slot;
    if (slot == -1) return;   // 无效槽 → 跳过

    // 拆解 flat slot → (块索引, 块内偏移)
    int block_idx    = slot / block_size;
    int block_offset = slot % block_size;

    // 计算源/目标指针
    // 源：使用 key_stride 而非 D，正确处理非连续输入
    //     对应 triton: key_ptr + idx * key_stride + tl.arange(0, D)
    const scalar_t* k_src = key_ptr   + token_idx * key_stride;
    const scalar_t* v_src = value_ptr + token_idx * value_stride;

    // 目标：cache 假设连续 → 用 D 作为 slot 内步长
    scalar_t* k_dst = k_cache_ptr + block_idx    * (block_size * D)
                                  + block_offset * D;
    scalar_t* v_dst = v_cache_ptr + block_idx    * (block_size * D)
                                  + block_offset * D;

    // 向量化复制（每事务 16 字节 = uint4）
    // token 内部的 D 个元素始终连续，因此向量化始终安全
    int elem_size   = sizeof(scalar_t);
    int total_bytes = D * elem_size;
    int vec_count   = total_bytes / 16;

    const uint4* k_src_u4 = reinterpret_cast<const uint4*>(k_src);
    const uint4* v_src_u4 = reinterpret_cast<const uint4*>(v_src);
    uint4* k_dst_u4 = reinterpret_cast<uint4*>(k_dst);
    uint4* v_dst_u4 = reinterpret_cast<uint4*>(v_dst);

    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
        k_dst_u4[i] = k_src_u4[i];
        v_dst_u4[i] = v_src_u4[i];
    }

    // 字节级余数复制（0-15 字节）
    int remaining_start = vec_count * 16;
    if (remaining_start < total_bytes) {
        const char* k_src_c = reinterpret_cast<const char*>(k_src) + remaining_start;
        const char* v_src_c = reinterpret_cast<const char*>(v_src) + remaining_start;
        char* k_dst_c = reinterpret_cast<char*>(k_dst) + remaining_start;
        char* v_dst_c = reinterpret_cast<char*>(v_dst) + remaining_start;

        for (int i = threadIdx.x; i < (total_bytes - remaining_start); i += blockDim.x) {
            k_dst_c[i] = k_src_c[i];
            v_dst_c[i] = v_src_c[i];
        }
    }
}


// ============================================================================
//  C++ 主机端 Dispatch
// ============================================================================

/**
 * 校验输入、提取 stride、计算启动参数并启动模板化内核。
 *
 * key/value 可以非连续（如 .split()+.view() 的切片），内核用显式 stride 处理。
 * Cache 必须连续 —— 它们是模型自有 buffer，始终新分配。
 */
void paged_kv_cache_store_cuda(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor slot_mapping
) {
    // 设备校验
    TORCH_CHECK(key.is_cuda(),         "key 必须在 CUDA 设备上");
    TORCH_CHECK(value.is_cuda(),       "value 必须在 CUDA 设备上");
    TORCH_CHECK(k_cache.is_cuda(),     "k_cache 必须在 CUDA 设备上");
    TORCH_CHECK(v_cache.is_cuda(),     "v_cache 必须在 CUDA 设备上");
    TORCH_CHECK(slot_mapping.is_cuda(),"slot_mapping 必须在 CUDA 设备上");

    // slot_mapping 类型标准化：接受 int32 或 int64
    if (slot_mapping.scalar_type() == torch::kInt) {
        slot_mapping = slot_mapping.to(torch::kLong);
    }
    TORCH_CHECK(slot_mapping.scalar_type() == torch::kLong,
                "slot_mapping 必须是 int32 或 int64，当前为 ", slot_mapping.scalar_type());

    // 形状/类型校验
    TORCH_CHECK(key.dim() == 3,        "key 必须是 3 维 [num_tokens, num_kv_heads, head_dim]");
    TORCH_CHECK(value.dim() == 3,      "value 必须是 3 维");
    TORCH_CHECK(k_cache.dim() == 4,    "k_cache 必须是 4 维 [num_blocks, block_size, num_kv_heads, head_dim]");
    TORCH_CHECK(v_cache.dim() == 4,    "v_cache 必须是 4 维");

    TORCH_CHECK(key.sizes() == value.sizes(),    "key 和 value 形状必须一致");
    TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "k_cache 和 v_cache 形状必须一致");
    TORCH_CHECK(key.size(1) == k_cache.size(2) && key.size(2) == k_cache.size(3),
                "num_kv_heads 和 head_dim 在 key 与 k_cache 间必须匹配");
    TORCH_CHECK(key.dtype() == k_cache.dtype(),  "key 与 k_cache dtype 不匹配");
    TORCH_CHECK(value.dtype() == v_cache.dtype(), "value 与 v_cache dtype 不匹配");
    TORCH_CHECK(slot_mapping.dim() == 1 && slot_mapping.size(0) == key.size(0),
                "slot_mapping 必须是 1 维且长度等于 num_tokens");

    // Cache 连续性校验（key/value 不做此要求）
    TORCH_CHECK(k_cache.is_contiguous(),
        "k_cache 必须连续。"
        "当前 stride ", k_cache.strides(), "，连续 stride 应为 ",
        torch::IntArrayRef({k_cache.size(1) * k_cache.size(2) * k_cache.size(3),
                            k_cache.size(2) * k_cache.size(3),
                            k_cache.size(3), 1}), "。"
        "Cache 张量是模型自有 buffer，应始终新分配且连续。"
    );
    TORCH_CHECK(v_cache.is_contiguous(),
        "v_cache 必须连续。"
        "当前 stride ", v_cache.strides(), "，连续 stride 应为 ",
        torch::IntArrayRef({v_cache.size(1) * v_cache.size(2) * v_cache.size(3),
                            v_cache.size(2) * v_cache.size(3),
                            v_cache.size(3), 1}), "。"
    );

    // 启动参数
    int num_tokens   = key.size(0);
    int num_kv_heads = key.size(1);
    int head_dim     = key.size(2);
    int D            = num_kv_heads * head_dim;
    int block_size   = k_cache.size(1);

    // stride(0) 是相邻 token 间的元素间距
    // 连续张量 = D；split+view 切片可能更大
    int key_stride   = key.stride(0);
    int value_stride = value.stride(0);

    // 线程数启发式：D 越大越多线程
    int threads_per_block = 256;
    if (D * static_cast<int>(sizeof(at::Half)) / 16 < 128) {
        threads_per_block = 128;
    }

    dim3 grid(num_tokens);   // 每 token 一个 block
    dim3 block(threads_per_block);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 按标量类型分发
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        key.scalar_type(), "paged_kv_cache_store", [&] {
            paged_kv_cache_store_kernel<scalar_t><<<grid, block, 0, stream>>>(
                key.data_ptr<scalar_t>(),
                key_stride,
                value.data_ptr<scalar_t>(),
                value_stride,
                k_cache.data_ptr<scalar_t>(),
                v_cache.data_ptr<scalar_t>(),
                slot_mapping.data_ptr<int64_t>(),
                num_tokens, D, block_size
            );
        }
    );
}


// ============================================================================
//  Meta（FakeTensor）实现 —— torch.compile 追踪用
// ============================================================================

/**
 * torch.compile 在 FakeTensor 传播阶段调用。不执行计算，仅告知 tracer：
 * “这是一个合法的 in‑place 算子，无返回值”。
 */
void paged_kv_cache_store_meta(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor slot_mapping
) {
    // 空操作：k_cache 和 v_cache 就地修改，无需描述输出张量
}


// ============================================================================
//  算子注册（torch.library）
// ============================================================================

/**
 * TORCH_LIBRARY   — 在命名空间中定义算子 schema
 * TORCH_LIBRARY_IMPL — 绑定各后端的实现
 *
 * Schema 注解：
 *   Tensor(a!) → k_cache 就地修改（别名组 a）
 *   Tensor(b!) → v_cache 就地修改（别名组 b）
 *   -> ()      → 无返回值
 */

TORCH_LIBRARY(paged_kv_cache, m) {
    m.def(
        "paged_kv_cache_store("
        "Tensor key, "
        "Tensor value, "
        "Tensor(a!) k_cache, "
        "Tensor(b!) v_cache, "
        "Tensor slot_mapping"
        ") -> ()"
    );
}

TORCH_LIBRARY_IMPL(paged_kv_cache, CUDA, m) {
    m.impl("paged_kv_cache_store", &paged_kv_cache_store_cuda);
}

TORCH_LIBRARY_IMPL(paged_kv_cache, Meta, m) {
    m.impl("paged_kv_cache_store", &paged_kv_cache_store_meta);
}


// ============================================================================
//  最小 pybind11 模块 —— 提供 PyInit__C 入口
// ============================================================================

/**
 * 算子注册由 TORCH_LIBRARY 完成，此模块不暴露任何函数。
 * 仅提供 Python import _C 所需的 PyInit__C 符号。
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Paged KV Cache Store — compiled CUDA extension";
}
```

设计要点：
- **stride‑aware**：`key_ptr + token_idx * key_stride`，与 triton `idx * key_stride` 对齐
- **uint4 向量化**：16 字节事务，充分利用带宽；余数由字节循环兜底
- **int32/64 兼容**：dispatch 中自动将 `int32` 转为 `int64`
- **cache 连续校验**：带诊断信息（显示 actual vs expected strides）
- **Meta 注册**：`torch.compile` 追踪时不触发 `NotImplementedError`

```python title="setup.py"
"""构建 paged_kv_cache_ext._C CUDA 扩展。

标准 PyTorch CUDA 扩展布局（参考 eole）：
- csrc/              → 纯 C/CUDA 源码，无 __init__.py
- paged_kv_cache_ext/ → Python 包，导入编译产物 _C

构建命令：
    uv run python setup.py build_ext --inplace
"""

import os
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
    python_requires=">=3.10",
    install_requires=["torch"],
)
```

关键点：
- `CUDAExtension(name="paged_kv_cache_ext._C")` — 编译产物放在 `paged_kv_cache_ext/` 下，名为 `_C`
- `-std=c++17` 是 `TORCH_LIBRARY` 宏的最低要求
- `--expt-relaxed-constexpr` 使 nvcc 兼容 PyTorch 头文件中的 `constexpr` 用法
- 自动检测 GPU 架构，编译出针对当前硬件的优化代码

```python title="paged_kv_cache_ext/__init__.py"
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
```

双层防御：Python 层提供友好诊断 → C++ 层做最终校验。

## 4. Compile & Run

```bash
uv run python setup.py build_ext --inplace
```

编译产物 `_C.cpython-3xx-...so` 被复制到 `paged_kv_cache_ext/` 下。

预期输出：

```bash
(paged-kv-cache-ext) root@sheldon:/home/sheldon/paged_kv_cache_ext# uv run main.py 
K: torch.Size([16, 4, 32])
V: torch.Size([16, 4, 32])
K cache: torch.Size([8, 4, 4, 32])
V cache: torch.Size([8, 4, 4, 32])
slot mapping: torch.Size([16])
 token   slot  (block, offset)         init   key[token,0,0] k_cache[block,offset,0,0]
     0     11           (2, 3)        -0.103608        -0.181456                 -0.181456
     1      3           (0, 3)        -0.300611         0.670980                  0.670980
     2     13           (3, 1)        -0.561376         0.398255                  0.398255
     3     25           (6, 1)        -0.084684        -0.525674                 -0.525674
     4     23           (5, 3)         1.075368         1.037455                  1.037455
     5     29           (7, 1)        -1.673548         0.363962                  0.363962
     7     20           (5, 0)         1.417951         0.312266                  0.312266
     8     15           (3, 3)         1.268803         0.912131                  0.912131
     9     27           (6, 3)         0.178813         1.093807                  1.093807
    10     26           (6, 2)        -0.590735         1.777324                  1.777324
    11      4           (1, 0)         0.313605        -2.006953                 -2.006953
    12     10           (2, 2)        -1.848917        -0.917820                 -0.917820
    13     19           (4, 3)        -1.564591         0.136705                  0.136705
    14     14           (3, 2)         1.228593        -1.083561                 -1.083561
K cache k1 vs k2: True
V cache v1 vs v2: True
K cosine similarity: 0.9999997615814209
V cosine similarity: 1.0
K cache k2 vs k3: True
V cache v2 vs v3: True

==========================================================================================
综合性能对比 (中位数时间 / ms)
==========================================================================================
配置                             v1 (torch)  v2 (triton)    v3 (cuda)
------------------------------------------------------------------------------------------
Small  (32x8x64)                   0.6157       0.0130       0.0202
Medium (4096x32x128)               2.2253       0.7266       0.7261
Large  (16384x8x128)               2.1820       0.7234       0.7194
```

- `K cache k1 vs k2: True`、`V cache v1 vs v2: True`（triton == torch）
- `K cache k2 vs k3: True`、`V cache v2 vs v3: True`（cuda == triton）
- 性能表：cuda 与 triton 接近，比 torch 快 3–50×

## 5. github

…or create a new repository on the command line

```bash
git init
$ git config user.name "Username"
$ git config user.email "Email"
$ git remote -v 

git add .
git commit -m "first commit"
git branch -M master
git remote add origin git@github.com:username/code.git
git push -u origin master
```

…or push an existing repository from the command line

```bash
git remote add origin git@github.com:username/code.git
git branch -M master
git push -u origin master
```

## 6. Useful Prompts

该教学案例具有以下鲜明特点：

* **示例驱动注释**  
  main.py 中 `store_kvcache_torch` 的 docstring 用一组具体的 `slot_mapping` 值（如 `11, 3, 13, 25…`）**手动演算**出 `block_indices` 与 `block_offsets`，并配以箭头注释（`↓ token 0 will be stored in slot 11`），形成**可逐行对照的数字拆解图**。  
  这种风格可称为“**数值追踪式注释（traceable numeric illustration）**”——阅读者无需运行代码就能在脑中复现内存变换过程。

* **逐级递进的实现对比**  
  整个算子从 **纯 PyTorch（直观理解）→ Triton（中间性能锚点）→ 手写 CUDA（生产级高性能 + 跨平台编译）** 依次展开。  
  - **torch**：利用高级索引直接完成逻辑，代码最短，作为“黄金参考”。  
  - **triton**：接近硬件但保持 Python 可读性，提供独立的性能基线。  
  - **cuda**：展示如何用 `uint4` 向量化、显式 stride、`TORCH_LIBRARY` 注册来获得 `torch.compile` 兼容的最终产物。

* **内置精度保险丝**  
  每引入一种新实现，立即与原版比对：`torch.equal(k1, k2)`、余弦相似度（`cosine_similarity`），并在打印中显式输出 `True`，防止读者在后续基准中跑错误代码而不自知。

* **可复现的微基准方法论**  
  基准函数采用 **warmup + 克隆缓存 + CUDA event 计时 + 中位数** 的严谨流程，覆盖三种规模（Small/Medium/Large），并在表格中横向对比三种实现的中位数耗时。这种“三规模×三实现”的矩阵让人一眼看清不同方案在负载尺度下的表现。

* **Stride‑Aware 的 CUDA 内核设计**  
  内核接受显式 `key_stride` / `value_stride`，不仅能处理连续张量，还兼容 `.split()+.view()` 产生的不连续视图。这与 Triton 的 `idx * key_stride + tl.arange(0, D)` 模式完全对齐，注释中明确解释了 stride 的源头，避免常见的内存越界 BUG。

* **双语言/双层次注释**  
  - **Python 层**：用自然语言解释算法意图，例如“Handles -1 via masked scatter to avoid OOB access”。  
  - **CUDA 层**：同时保留英文技术文档和中文核心注解（如“向量化复制（每事务 16 字节 = uint4）”），方便中文社区开发者理解；每个核心变量的含义都在声明旁注释（如 `int D = num_kv_heads * head_dim;`）。

* **编译与加载的工程化分离**  
  - **项目结构**：`csrc/`（纯源码）与 `paged_kv_cache_ext/`（Python 包）物理分离，编译产物 `_C.xxx.so` 落于包内，用户只需 `import`。  
  - **跨平台与 torch.compile 兼容**：通过 `TORCH_LIBRARY` 注册算子而非 raw pybind，同时提供 Meta 实现，保证 `torch.compile` 追踪无断点。  
  - **自动设备能力检测**：`setup.py` 自动探测 GPU 计算能力并设置 `gencode` 选项，避免硬编码。

* **防御性的用户提示**  
  Python 层与 C++ 层双重检查 cache 连续性，错误信息直接展示**当前 stride 与期望 stride** 的对比，并给出 `k_cache = k_cache.contiguous()` 的修复建议。这种诊断信息大幅降低初次使用者的调试成本。

* **文档与背景前置**  
  在项目开头的 `Background` 中先以两段话讲清“为什么需要分页 KV 缓存”和“store 操作在推理中的作用”，让即使不熟悉 vLLM 架构的读者也能迅速建立上下文。

以下精准词汇可刻画该教案的独特风格，每一项均可作为教学设计的标签：

- **数值追踪式注释** (*traceable numeric illustration*)  
- **逐级递进对比** (*progressive multi-backend comparison*)  
- **黄金参考实现** (*golden reference implementation*)  
- **精度保险丝** (*accuracy fuse*)  
- **可复现微基准矩阵** (*reproducible micro-benchmarking matrix*)  
- **显式步长感知** (*explicit stride-awareness*)  
- **双语言并行注释** (*bilingual dual-layer annotation*)  
- **编译时算子注册** (*compile-time operator registration*)  
- **Meta 空壳实现** (*Meta backend stub*)  
- **源码包物理分离** (*source-package physical separation*)  
- **防御性逐层校验** (*defensive tiered validation*)  
- **连续性诊断提示** (*contiguity diagnostic hint*)  
- **背景前置** (*prefaced background context*)  
- **跨平台架构自动探测** (*auto-detection of GPU compute capability*)  

这些术语覆盖了注释风格、代码结构、验证方法、性能测量、内核设计、工程部署等维度，共同构成了“从直观理解到生产级代码”的完整教学脉络。
