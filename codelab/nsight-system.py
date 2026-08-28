r"""
# ============================================================
# Eager 基线：普通 PyTorch 执行，无算子融合，无 CUDA Graph
# ============================================================
nsys profile \                                  # 启动 Nsight Systems 采集
  -t cuda,nvtx,osrt,cudnn,cublas \              # 跟踪 CUDA、NVTX、OS runtime、cuDNN、cuBLAS API
  -s none \                                     # 禁用采样，记录所有事件（保证完整性）
  --cpuctxsw=none \                             # 不跟踪 CPU 上下文切换
  --cuda-graph-trace=node \                     # 跟踪 CUDA Graph 节点级信息（虽然本模式未用图）
  --capture-range=cudaProfilerApi \             # 仅采集 cudaProfilerStart/Stop 之间的范围
  --capture-range-end=stop-shutdown \           # 当 profiler API 停止或进程结束时结束采集
  -o eager \                                    # 输出报告文件名前缀（生成 eager.nsys-rep）
  -f true \                                     # 若文件已存在则强制覆盖
  python bench.py --mode eager                  # 运行目标脚本，模式为 eager

# ============================================================
# torch.compile 算子融合（无 CUDA Graph）
# ============================================================
nsys profile \                                  # 启动 Nsight Systems 采集
  -t cuda,nvtx,osrt,cudnn,cublas \              # 跟踪 CUDA、NVTX、OS runtime、cuDNN、cuBLAS API
  -s none \                                     # 禁用采样，记录所有事件
  --cpuctxsw=none \                             # 不跟踪 CPU 上下文切换
  --cuda-graph-trace=node \                     # 跟踪 CUDA Graph 节点级信息
  --capture-range=cudaProfilerApi \             # 仅采集 cudaProfilerStart/Stop 之间的范围
  --capture-range-end=stop-shutdown \           # 当 profiler API 停止或进程结束时结束采集
  -o compile \                                  # 输出报告文件名前缀（生成 compile.nsys-rep）
  -f true \                                     # 若文件已存在则强制覆盖
  python bench.py --mode compile                # 运行目标脚本，模式为 compile

# ============================================================
# 手动 CUDA Graph：捕获整段前向，回放降低 launch 开销
# ============================================================
nsys profile \                                  # 启动 Nsight Systems 采集
  -t cuda,nvtx,osrt,cudnn,cublas \              # 跟踪 CUDA、NVTX、OS runtime、cuDNN、cuBLAS API
  -s none \                                     # 禁用采样，记录所有事件
  --cpuctxsw=none \                             # 不跟踪 CPU 上下文切换
  --cuda-graph-trace=node \                     # 跟踪 CUDA Graph 节点级信息（本模式使用图，该选项重要）
  --capture-range=cudaProfilerApi \             # 仅采集 cudaProfilerStart/Stop 之间的范围
  --capture-range-end=stop-shutdown \           # 当 profiler API 停止或进程结束时结束采集
  -o cudagraph \                                # 输出报告文件名前缀（生成 cudagraph.nsys-rep）
  -f true \                                     # 若文件已存在则强制覆盖
  python bench.py --mode cudagraph              # 运行目标脚本，模式为 cudagraph

# ============================================================
# torch.compile + CUDA Graph（内置 reduce-overhead）
# ============================================================
nsys profile \                                  # 启动 Nsight Systems 采集
  -t cuda,nvtx,osrt,cudnn,cublas \              # 跟踪 CUDA、NVTX、OS runtime、cuDNN、cuBLAS API
  -s none \                                     # 禁用采样，记录所有事件
  --cpuctxsw=none \                             # 不跟踪 CPU 上下文切换
  --cuda-graph-trace=node \                     # 跟踪 CUDA Graph 节点级信息
  --capture-range=cudaProfilerApi \             # 仅采集 cudaProfilerStart/Stop 之间的范围
  --capture-range-end=stop-shutdown \           # 当 profiler API 停止或进程结束时结束采集
  -o compile_cudagraph \                        # 输出报告文件名前缀（生成 compile_cudagraph.nsys-rep）
  -f true \                                     # 若文件已存在则强制覆盖
  python bench.py --mode compile+cudagraph      # 运行目标脚本，模式为 compile+cudagraph

# ============================================================
# 可视化报告（GUI）
# ============================================================
nsys-ui eager.nsys-rep                         # 打开 eager 报告图形界面
# 或通过 GUI 菜单 File → Open 依次加载其他 .nsys-rep 文件

# ============================================================
# 统计报告：GPU kernel 汇总
# ============================================================

nsys stats eager.nsys-rep --report cuda_gpu_kern_sum
# 报告类型 cuda_gpu_kern_sum：统计所有 CUDA kernel 的名称、实例数、平均耗时等
# eager 模式下应看到大量独立的 elementwise kernel（如 aten::gelu、aten::mul）

nsys stats compile.nsys-rep --report cuda_gpu_kern_sum
# 观察 triton_ 前缀 kernel 出现，elementwise 类 kernel 消失（被融合）

nsys stats compile_cudagraph.nsys-rep --report cuda_gpu_kern_sum
# 同样应出现 triton_ 前缀 kernel，且 kernel 总数进一步减少

# ============================================================
# 统计报告：CUDA API 调用汇总
# ============================================================
nsys stats cudagraph.nsys-rep --report cuda_api_sum
# 报告类型 cuda_api_sum：统计 CUDA API 调用次数与耗时
# 关键观察：cudaGraphLaunch 出现，cudaLaunchKernel 次数骤减

nsys stats compile_cudagraph.nsys-rep --report cuda_api_sum
# 同样应出现 cudaGraphLaunch，且 launch 总次数显著低于 eager 模式

# ============================================================
# 总结：eager > compile > cuda_graph > compile + cuda_graph
# ============================================================

- 1个算子层，
- 4种推理方式：
    - eager
    - torch.compile / fused kernels
    - cudagraphs
    - torch.compile(reduce-overhead) 
- 看2个指标：
    - cuda_gpu_kern_sum = GPU 侧真正执行了什么
    - cuda_api_sum      = CPU 侧花了多长时间去启动

> nsys profile -t cuda,nvtx,osrt,cudnn,cublas -s none --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown -o eager -f true python bench.py --mode eager
> nsys stats eager.nsys-rep --report cuda_gpu_kern_sum
> nsys stats eager.nsys-rep --report cuda_api_sum

> nsys profile -t cuda,nvtx,osrt,cudnn,cublas -s none --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown -o compile -f true python bench.py --mode compile
> nsys stats compile.nsys-rep --report cuda_gpu_kern_sum
> nsys stats compile.nsys-rep --report cuda_api_sum

> nsys profile -t cuda,nvtx,osrt,cudnn,cublas -s none --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown -o cudagraph -f true python bench.py --mode cudagraph
> nsys stats cudagraph.nsys-rep --report cuda_gpu_kern_sum
> nsys stats cudagraph.nsys-rep --report cuda_api_sum

> nsys profile -t cuda,nvtx,osrt,cudnn,cublas -s none --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown -o compile_cudagraph -f true python bench.py --mode compile+cudagraph
> nsys stats compile_cudagraph.nsys-rep --report cuda_gpu_kern_sum
> nsys stats compile_cudagraph.nsys-rep --report cuda_api_sum
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.profiler as profiler
import torch.cuda.nvtx as nvtx

torch.set_float32_matmul_precision('high') # GPU 支持 TF32 矩阵乘，PyTorch 默认 FP32 精度不开启。

class TinyMLP(nn.Module):
    def __init__(self, d_in=1024, d_hidden=2048, d_out=512):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        # 故意串联 pointwise 算子：eager 下 gelu/mul/add 各是一个独立 kernel，
        # torch.compile 后会被 Inductor 融合为单个 triton kernel，方便在 timeline 上直观对比
        h = F.gelu(self.fc1(x)) * 1.1 + 0.2
        h = F.gelu(self.fc2(h)) * 0.9 + 0.1
        return self.fc3(h)


def build(mode, model, static_x):
    if mode == "eager":
        def step():
            with torch.no_grad():
                return model(static_x)

    elif mode == "compile":                       # 仅算子融合（无 CUDA Graph）
        compiled = torch.compile(model)
        def step():
            with torch.no_grad():
                return compiled(static_x)

    elif mode == "cudagraph": # 手动 CUDA Graph，捕获整段前向
        # 1) 旁路 stream warmup：预热 cuBLAS handle / allocator
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                with torch.no_grad():
                    model(static_x)
        torch.cuda.current_stream().wait_stream(s)

        # 2) 捕获（推理版只需包一次 forward，比训练版简单得多）
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad():
                static_out = model(static_x)

        def step():
            # 真实推理要换数据时，只能 copy_，不能重新赋值 static_x：
            #   static_x.copy_(torch.randn_like(static_x))
            g.replay()
            return static_out

    elif mode == "compile+cudagraph":             # torch.compile 内置 cudagraphs
        compiled = torch.compile(model, mode="reduce-overhead")
        def step():
            with torch.no_grad():
                return compiled(static_x)

    return step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["eager", "compile", "cudagraph", "compile+cudagraph"], required=True)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda")

    model = TinyMLP().to(dev).eval() # 推理：eval() 关 dropout；没有 optimizer / loss / backward

    # mock 输入：torch.randn 一次性构造，地址固定
    static_x = torch.randn(args.batch, 1024, device=dev)

    step = build(args.mode, model, static_x)

    # warmup：让 compile 完成 JIT、cudagraph 完成捕获，不污染 profile
    for _ in range(10):
        step()
    torch.cuda.synchronize()

    # ===== nsys 采集范围（对应 --capture-range=cudaProfilerApi） =====
    profiler.start()
    nvtx.range_push(f"MODE:{args.mode}")
    for i in range(args.steps):
        nvtx.range_push(f"step_{i}")
        out = step()
        nvtx.range_pop()
    nvtx.range_pop()
    torch.cuda.synchronize()
    profiler.stop()


if __name__ == "__main__":
    main()


r"""
nsys profile -t cuda,nvtx,osrt,cudnn,cublas -s none --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown -o eager -f true python bench.py --mode eager
Capture range started in the application.
Capture range ended in the application.
Generating '/tmp/nsys-report-f57e.qdstrm'
[1/1] [0%                          ] eager.nsys-repProcessing events...
[1/1] [========================100%] eager.nsys-rep
Generated:
        /home/beaver/eager.nsys-rep


```
Capture range started in the application.   ← 程序调用了 cudaProfilerStart()，开始记录（对应代码里的 profiler.start()）
Capture range ended in the application.     ← 调用了 cudaProfilerStop()，停止记录
Generating '/tmp/nsys-report-8ae4.qdstrm'   ← 先生成原始事件流（中间文件）
[1/1] [====100%] eager.nsys-rep             ← 把事件流加工成最终报告
Generated: /home/beaver/eager.nsys-rep      ← 最终产物，GUI 和 stats 都吃这个文件
```
只有看到 **started + ended 两行都才说明 `--capture-range=cudaProfilerApi` 生效了
（如果没出现，报告会是空的）。
"""

r"""
> nsys stats eager.nsys-rep --report cuda_gpu_kern_sum
[eager] ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
     75.8        9,501,639         60  158,360.6  185,468.0   101,862   360,951     59,402.9  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_align4>(T1::Params)             
     18.7        2,337,879         30   77,929.3   54,052.0    53,219   462,622     91,597.8  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_align4>(T1::Params)              
      2.0          255,632         60    4,260.5    4,240.5     4,096     4,512         87.2  void at::native::vectorized_elementwise_kernel<(int)4, at::native::GeluCUDAKernelImpl(at::TensorIte…
      1.7          217,674         60    3,627.9    3,600.5     3,393     4,160        180.3  void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, floa…
      1.7          214,866         60    3,581.1    3,568.5     3,392     3,809        119.6  void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, st…

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # 1. fc1 和 fc3 的 GEMM: `cutlass::Kernel2<...128x128_16x5...>` 2*30 
    # 2. fc2 的 GEMM: `cutlass::Kernel2<...128x64_16x6...>` 1*30
    # 3. 两个 `gelu`: `vectorized_elementwise_kernel<..., GeluCUDAKernelImpl>` 2*30
    # 4. 两个 `mul`（*1.1, *0.9）: `vectorized_elementwise_kernel<..., AUnaryFunctor>` 2*30
    # 5. 两个 `add`（+0.2, +0.1）: `vectorized_elementwise_kernel<..., CUDAFunctorOnSelf_add>` 2*30
    # - **GEMM 有两种不同的 cutlass 配置**：fc1 和 fc3 因为输入/输出维度不同但可能被映射到同一种优化配置（`128x128_16x5`），而 fc2 的维度是 2048→2048，使用了另一种配置（`128x64_16x6`），因此 GEMM 部分共 2 行。
    # - **elementwise 有三种不同的算子实现**：gelu、mul（AUnaryFunctor）、add（CUDAFunctorOnSelf_add），各一行，共 3 行。
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

> nsys stats eager.nsys-rep --report cuda_api_sum
[eager]  ** CUDA API Summary (cuda_api_sum):

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)          Name         
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ---------------------
     69.7        6,223,094          1  6,223,094.0  6,223,094.0  6,223,094  6,223,094          0.0  cudaDeviceSynchronize
     14.4        1,284,270        180      7,134.8      5,410.5      3,226     41,508      5,460.4  cudaLaunchKernel     
      7.4          657,145         90      7,301.6      4,598.5      2,274    103,727     11,409.9  cudaMemsetAsync      
      7.3          647,829         90      7,198.1      4,528.5      2,805     47,740      8,037.3  cuLaunchKernel       
      0.5           42,731          1     42,731.0     42,731.0     42,731     42,731          0.0  cuProfilerStart      
      0.5           42,629         90        473.7        320.0        160      5,721        726.4  cuKernelGetFunction
      0.3           25,194        180        140.0        110.0         60      1,853        184.7  cuKernelGetName

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # cudaLaunchKernel` 180 次 + cuLaunchKernel 90 次 = 270次 = 9*30
    # - **cuLaunchKernel**（90 次）：通常由 cuBLAS/cutlass 内部使用，对应 3 个 GEMM * 30 step = 90 次。
    # - **cudaLaunchKernel**（180 次）：PyTorch 的 elementwise 算子（gelu、mul、add）通常通过该 API 启动，对应 6 个 elementwise * 30 step = 180 次。
    # - 此外还有 `cudaMemsetAsync` 90 次，每个 Linear 的输出张量在分配后需要清零，3 个 Linear * 30 step = 90 次。
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

> nsys stats compile.nsys-rep --report cuda_gpu_kern_sum
[compile] ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                                           Name                                         
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  --------------------------------------------------------------------------------------
     53.5        5,814,073         30  193,802.4  186,699.0   185,675   378,359     34,931.8  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_tn_align4>(T1::Params)
     43.7        4,750,175         60   79,169.6   78,885.5    53,571   108,006     25,022.3  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_align4>(T1::Params)
      1.6          175,692         30    5,856.4    5,824.0     5,760     6,273        126.7  triton_poi_fused_add_addmm_gelu_mul_0                                                 
      1.1          122,728         30    4,090.9    4,064.5     3,936     4,577        171.6  triton_poi_fused_add_addmm_gelu_mul_1  

> nsys stats compile.nsys-rep --report cuda_api_sum
[compile]  ** CUDA API Summary (cuda_api_sum):

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)          Name         
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ---------------------
     78.5        8,054,567          1  8,054,567.0  8,054,567.0  8,054,567  8,054,567          0.0  cudaDeviceSynchronize
     14.3        1,466,016        150      9,773.4      4,513.5      2,705    488,235     39,907.7  cuLaunchKernel       
      6.6          680,836         60     11,347.3      3,732.0      2,054    382,816     49,076.9  cudaMemsetAsync      
      0.3           34,168         90        379.6        211.0        161      4,589        569.1  cuKernelGetFunction  
      0.2           22,002          1     22,002.0     22,002.0     22,002     22,002          0.0  cuProfilerStart      

      # compile 模式：融合后每个 step 只有 5 个 kernel（3 GEMM + 2 triton fused），30 step 共 150 次启动。
      # 报告中列出了 4 行：两行 cutlass GEMM（共 90 实例）+ 两行 triton fused（各 30 实例）。
      # 融合后的两个 triton kernel 分别对应 fc1 后和 fc2 后的 pointwise 链。

> nsys stats cudagraph.nsys-rep --report cuda_gpu_kern_sum
[cudagraph] ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
     80.3        9,547,569         60  159,126.1  186,887.5   103,858   625,645     76,581.3  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_align4>(T1::Params)             
     13.7        1,624,523         30   54,150.8   53,977.0    53,433    55,513        546.0  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_align4>(T1::Params)              
      2.3          268,888         60    4,481.5    4,479.0     4,351     4,832         89.2  void at::native::vectorized_elementwise_kernel<(int)4, at::native::GeluCUDAKernelImpl(at::TensorIte…
      2.0          236,125         60    3,935.4    3,744.0     3,583    14,174      1,350.1  void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, st…
      1.8          210,471         60    3,507.8    3,488.0     3,391     3,679         75.6  void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, floa…

> nsys stats cudagraph.nsys-rep --report cuda_api_sum
[cudagraph] ** CUDA API Summary (cuda_api_sum):

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)              Name            
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  ----------------------------
     93.7       12,297,053          1  12,297,053.0  12,297,053.0  12,297,053  12,297,053          0.0  cudaDeviceSynchronize       
      6.0          793,365         30      26,445.5      19,081.0      15,690     130,357     23,057.8  cudaGraphLaunch_v10000      
      0.1           19,020         30         634.0         170.0         151       7,885      1,601.5  cudaStreamIsCapturing_v10000
      0.1            7,584          1       7,584.0       7,584.0       7,584       7,584          0.0  cuProfilerStart     

    # cudagraph 模式：kernel 与 eager 完全相同，所以 cuda_gpu_kern_sum 报告也是 5 行，总实例 270。
    # API 方面所有 kernel 被封装进图，因此只有 cudaGraphLaunch 30 次。

> nsys stats compile_cudagraph.nsys-rep --report cuda_gpu_kern_sum
[compile_cudagraph] ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
     52.8        6,474,246         30  215,808.2  186,865.0   184,289   744,613    113,918.0  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_tn_align4>(T1::Params)              
     41.2        5,047,553         60   84,125.9  102,529.0    53,185   351,106     43,093.2  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_align4>(T1::Params)              
      3.6          438,114         30   14,603.8   14,464.0    14,112    15,424        419.1  void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::TensorListMetadata<(in…
      1.5          178,563         30    5,952.1    5,888.0     5,792     6,624        197.6  triton_poi_fused_add_addmm_gelu_mul_0                                                               
      1.0          127,424         30    4,247.5    4,128.0     4,064     5,376        329.3  triton_poi_fused_add_addmm_gelu_mul_1    

    # multi_tensor_apply_kernel: reduce-overhead 每步把新输入拷进静态 buffer（foreach copy）
    # 代码注释里"换数据只能 copy_"的自动版——torch.compile 的 cudagraphs 帮你自动做了，代价是每步多一个小拷贝 kernel。

> nsys stats compile_cudagraph.nsys-rep --report cuda_api_sum
[compile_cudagraph] ** CUDA API Summary (cuda_api_sum):

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)              Name            
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  ----------------------------
     88.9       10,524,367          1  10,524,367.0  10,524,367.0  10,524,367  10,524,367          0.0  cudaDeviceSynchronize       
      6.7          789,855         30      26,328.5      25,708.5      12,764      86,944     13,225.7  cudaGraphLaunch_v10000      
      4.1          487,191         30      16,239.7       8,736.5       6,142     147,630     26,079.0  cudaLaunchKernel            
      0.1           15,970         30         532.3         385.5         230       3,506        589.0  cudaStreamIsCapturing_v10000
      0.1           10,159          1      10,159.0      10,159.0      10,159      10,159          0.0  cuProfilerStart             
      0.1            8,143         30         271.4         200.0          80       1,964        341.3  cuKernelGetName          

    # compile+cudagraph 模式：kernel 数量为 6 个/step（3 GEMM + 2 triton + 1 额外 multi_tensor_apply），30 step 共 180 实例，
    # 报告中显示了 5 行（两行 GEMM + 两行 triton + 一行 multi_tensor_apply）。
    # API 方面 cudaGraphLaunch 30 次 + cudaLaunchKernel 30 次（那个额外的 kernel 未捕获进图）。   
"""