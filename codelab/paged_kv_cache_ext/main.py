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