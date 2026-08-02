
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
