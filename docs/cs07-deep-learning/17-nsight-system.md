# Nsight Systems: eager, torch.compile and CUDA Graph

> **这篇文章适合谁**：会用 PyTorch 训练/推理，听说过“算子融合”“CUDA Graph”但没亲眼见过它们的证据；想学会用 Nsight Systems（nsys）分析 GPU 程序，却面对一堆表格不知从何看起的同学。
>
> **你将学到**：① 一份精心设计的基准脚本如何把四种推理模式变成可观测的对照实验；② nsys 每个命令行参数、每张统计表的每一列怎么读；③ 如何从真实测量数据中“读出”算子融合与 CUDA Graph 的效果，以及如何不被噪声骗到。

---

## 目录

1. [实验设计：1 个模型 × 4 种模式 × 2 个视角](#一)
2. [被测代码 bench.py 精讲](#二)
3. [采集命令逐参数精讲](#三)
4. [nsys stats 表格怎么读](#四)
5. [四种模式逐个复盘（真实数据）](#五)
6. [横向对比与六个结论](#六)
7. [GUI（nsys-ui）时间线怎么看](#七)
8. [Kernel 名解码速查表](#八)
9. [视野扩展：推理优化的全景图](#九)
10. [新手学习路径与资料](#十)
11. [动手练习](#十一)
12. [附录：复现命令清单](#附录)

---

<a id="一"></a>

## 一、实验设计：1 个模型 × 4 种模式 × 2 个视角

同一个三层 MLP，用四种方式做推理，各采集一份 nsys 报告：

| 模式 | 优化手段 | 预期变化 |
|---|---|---|
| `eager` | 无（基线） | 每步 9 个独立 kernel、9 次 launch |
| `compile` | torch.compile 算子融合 | pointwise 链被 Inductor 融合成 Triton kernel，kernel 数减少 |
| `cudagraph` | 手动 CUDA Graph | kernel 不变，但每步只用 **1 次** `cudaGraphLaunch` |
| `compile+cudagraph` | `mode="reduce-overhead"` | 融合 + 图回放 + 自动输入拷贝 |

分析时永远盯住**两条线**，这是全文的纲：

> - **`cuda_gpu_kern_sum`**（GPU kernel 汇总）= **GPU 侧真正执行了什么**，每个活干了多久；
> - **`cuda_api_sum`**（CUDA API 汇总）= **CPU 侧花了多少功夫去“安排”这些活**。

优化无非两件事：**让 GPU 干的活更少（算子融合）** + **让 CPU 安排得更省（CUDA Graph）**。

**名词小词典**（后文反复出现）：

| 名词 | 含义 |
|---|---|
| kernel | GPU 上执行的一个函数实例。eager 模式下每个 PyTorch 算子通常对应一个 kernel |
| launch | CPU 调用 CUDA API 把 kernel 提交给 GPU 的动作，每次有几微秒到几十微秒的 CPU 开销 |
| 算子融合 | 把多个小 kernel（如 gelu→mul→add）合并成一个大 kernel，省掉中间结果的显存读写和多次 launch |
| CUDA Graph | 把一整段 kernel 序列录制（capture）成一张“图”，之后一次 API 调用整图回放，消除逐个 launch 的开销 |
| NVTX | NVIDIA 的用户自定义标注 API，在时间线上打“书签”，方便定位代码区间 |

---

<a id="二"></a>

## 二、被测代码 bench.py 精讲

```python
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.profiler as profiler
import torch.cuda.nvtx as nvtx
torch.set_float32_matmul_precision('high')  # 允许 GEMM 使用 TF32 Tensor Core

class TinyMLP(nn.Module):
    def __init__(self, d_in=1024, d_hidden=2048, d_out=512):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_out)
    def forward(self, x):
        # 故意串联 pointwise 算子：eager 下 gelu/mul/add 各是一个独立 kernel，
        # torch.compile 后会被 Inductor 融合为单个 triton kernel
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
    elif mode == "cudagraph":                     # 手动 CUDA Graph
        # 1) 旁路 stream warmup：预热 cuBLAS handle / allocator
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                with torch.no_grad():
                    model(static_x)
        torch.cuda.current_stream().wait_stream(s)
        # 2) 捕获：推理版只需包一次 forward
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad():
                static_out = model(static_x)
        def step():
            # 真实推理要换数据时，只能 copy_，不能重新赋值 static_x
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
    model = TinyMLP().to(dev).eval()   # 推理：eval() 关 dropout；无 optimizer / backward
    static_x = torch.randn(args.batch, 1024, device=dev)  # 地址固定的静态输入
    step = build(args.mode, model, static_x)
    # warmup：让 compile 完成 JIT、cudagraph 完成捕获，不污染 profile
    for _ in range(10):
        step()
    torch.cuda.synchronize()
    # ===== nsys 采集范围（对应 --capture-range=cudaProfilerApi）=====
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
```

**七个关键设计，每一个都是为了“让证据清晰”：**

| # | 设计 | 为什么 |
|---|---|---|
| 1 | `gelu → *常数 → +常数` 串联两次 | pointwise 算子是融合的最佳素材：eager 下是 6 个小 kernel/步，compile 后应变成 2 个 Triton kernel/步，对比一目了然 |
| 2 | `static_x` 一次性构造、地址固定 | CUDA Graph 回放时读取的是**捕获时的内存地址**，输入地址必须固定（换数据只能 `copy_` 进去） |
| 3 | 手动图模式先在**旁路 stream** 上 warmup 3 次 | cuBLAS handle、显存分配器的惰性初始化不能发生在捕获期间（捕获期禁止 CPU 同步等操作），必须提前“热身” |
| 4 | 10 步 warmup + `synchronize()` 在 `profiler.start()` **之前** | torch.compile 的 JIT 编译、图的捕获都很耗时，放在采集范围外，报告才干净 |
| 5 | `profiler.start()/stop()` 包住测量循环 | 与 nsys 的 `--capture-range=cudaProfilerApi` 呼应，精确控制“录什么” |
| 6 | NVTX 嵌套区间 `MODE:xxx > step_i` | 在 GUI 时间线上变成可双击定位的书签 |
| 7 | `torch.set_float32_matmul_precision('high')` | 开启 TF32 矩阵乘。它的效果会直接写在 kernel 名里（见第五章），不开则是另一番景象（见第六章结论 6） |

> **推理版图捕获为什么比训练版简单？** 只需包一次 `forward`：没有 loss、没有 backward、没有 optimizer 的多阶段流程。这也是 CUDA Graph 在推理场景远比训练场景普及的原因之一。

---

<a id="三"></a>

## 三、采集命令逐参数精讲

四条命令**完全相同**，只有 `-o`（输出名）和 `--mode` 不同。以 eager 为例：

```bash
nsys profile \
  -t cuda,nvtx,osrt,cudnn,cublas \
  -s none \
  --cpuctxsw=none \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop-shutdown \
  -o eager -f true \
  python bench.py --mode eager
```

| 参数 | 作用 | 为什么这么用 |
|---|---|---|
| `-t cuda,nvtx,osrt,cudnn,cublas` | 跟踪 CUDA API/kernel、NVTX 标注、OS 运行时、cuDNN/cuBLAS 库级调用 | 覆盖我们关心的所有事件层 |
| `-s none` | **关闭 CPU 采样** | 我们要做事件级完整追踪而非 CPU 热点分析；关采样可减小开销和报告体积 |
| `--cpuctxsw=none` | 不跟踪 CPU 上下文切换 | 同上，分析 CPU 调度行为时才需要 |
| `--cuda-graph-trace=node` | **把 CUDA Graph 内的每个 kernel 单独记录** | 图模式下不加此项，整张图在时间线上是一整块“黑盒”，看不到内部 kernel |
| `--capture-range=cudaProfilerApi` | 只采集 `cudaProfilerStart/Stop` 之间 | 对应代码中的 `profiler.start()/stop()`，配合 warmup 排除编译/捕获噪声 |
| `--capture-range-end=stop-shutdown` | profiler 停止或进程结束时结束采集并生成报告 | 标准搭配 |
| `-o eager` | 输出文件名前缀（生成 `eager.nsys-rep`） | 四个模式分别命名，便于对比 |
| `-f true` | 已存在则强制覆盖 | 重跑实验时省心 |

四个变体：

| 输出 | 命令尾部 |
|---|---|
| `eager.nsys-rep` | `... -o eager -f true python bench.py --mode eager` |
| `compile.nsys-rep` | `... -o compile -f true python bench.py --mode compile` |
| `cudagraph.nsys-rep` | `... -o cudagraph -f true python bench.py --mode cudagraph` |
| `compile_cudagraph.nsys-rep` | `... -o compile_cudagraph -f true python bench.py --mode compile+cudagraph` |

### 采集日志逐行读

```
Capture range started in the application.   ← 程序调用了 cudaProfilerStart()（即 profiler.start()），开始记录
Capture range ended in the application.     ← 调用了 cudaProfilerStop()，停止记录
Generating '/tmp/nsys-report-xxxx.qdstrm'   ← 先落地原始事件流（中间文件）
[1/1] [====100%] eager.nsys-rep             ← 把事件流加工成最终报告
Generated: /home/beaver/eager.nsys-rep      ← 最终产物；GUI 和 stats 都吃这个文件
```

**只有 `started` + `ended` 两行都出现，才说明 `--capture-range=cudaProfilerApi` 生效了**——少任何一行，报告很可能是空的（最常见原因：被测程序根本没执行到 `profiler.start()`）。

### compile 模式多出来的几行输出，都不是错误

| 输出 | 含义 | 处理 |
|---|---|---|
| `Not enough SMs to use max_autotune_gemm mode` | GPU 的 SM 数较少，Inductor 放弃 GEMM 自动调优，退回默认路径 | 无害，知道即可 |
| `resource_tracker: leaked semaphore objects` | Inductor 编译子进程退出时的已知告警 | 忽略 |
| （若**没有** `set_float32_matmul_precision('high')`，会出现 `TensorFloat32 ... available but not enabled` 警告） | GEMM 会用纯 FP32 CUDA Core（kernel 名含 `simt`），而不是 TF32 Tensor Core | 想要更快就加上那一行 |

---

<a id="四"></a>

## 四、nsys stats 表格怎么读

以这条命令为例：

```bash
nsys stats eager.nsys-rep --report cuda_gpu_kern_sum
```

原始输出（先看一眼长什么样）：

```text
 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):
 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)  Name
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  ----
     75.8        9,501,639         60  158,360.6  185,468.0   101,862   360,951     59,402.9  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_align4>(T1::Params)
     ...
```

**每一列的含义：**

| 列 | 含义 | 阅读技巧 |
|---|---|---|
| Time (%) | 该 kernel 总时间占全部 kernel 总时间的比例 | 找大头：谁占 90% 谁就是瓶颈候选 |
| Total Time | 所有实例耗时之和 | **单位是 ns！** 心算时除以 1000 变 µs、再除以 1000 变 ms |
| Instances | 实例个数 | **除以步数（30）= “每步几个”**，是验证猜想的关键 |
| Avg / Med | 平均 / 中位数耗时 | Med 抗离群值干扰；Avg 与 Med 差得多 → 分布有长尾或混了不同的活 |
| Min / Max | 极值 | Max 远大于 Med → 有偶发卡顿（干扰/降频） |
| StdDev | 标准差 | **StdDev 很大 = 这行数据不可信，慎下结论** |
| Name | kernel 名 | 终端太窄会截断成 `…`；GUI 里可看全名，或导出 sqlite/CSV |

两个实用提示：

- `nsys stats` 会先把 `.nsys-rep` 导出成同名 `.sqlite` 缓存。**如果重新 profile 了同名报告，要加 `--force-export=true`**，否则读的是旧缓存。
- `nsys stats --help-reports` 可列出全部报告类型。除本文两个外，推荐 `cuda_gpu_trace`（逐条 kernel 明细，看顺序与空隙）和 `nvtx_kern_sum`（按 NVTX 区间聚合，直接回答“每个 step 里 GPU 忙了多久”）。

---

<a id="五"></a>

## 五、四种模式逐个复盘（真实数据）

以下表格均整理自真实输出：单位换算成 µs，`每步` = 实例数 ÷ 30。本实验所有 kernel 都在默认流上**串行**执行，所以各 kernel 时间之和 ≈ GPU 忙碌总时长。

### 5.1 eager：教科书式的“未融合”基线

**GPU 侧（cuda_gpu_kern_sum）**——5 行，共 270 实例 = **9 个 kernel/步**：

| kernel（简写） | 实例 | 每步 | Avg (µs) | Med (µs) | StdDev (µs) | 对应算子 |
|---|---|---|---|---|---|---|
| `cutlass_80_tensorop_s1688gemm_128x128_16x5` | 60 | 2 | 158.4 | 185.5 | 59.4 | fc1、fc3 的 GEMM |
| `cutlass_80_tensorop_s1688gemm_128x64_16x6` | 30 | 1 | 77.9 | 54.1 | **91.6** | fc2 的 GEMM |
| `vectorized_elementwise<GeluCUDAKernelImpl>` | 60 | 2 | 4.26 | 4.24 | 0.09 | 2 × gelu |
| `vectorized_elementwise<AUnaryFunctor>` | 60 | 2 | 3.63 | 3.60 | 0.18 | 2 × 标量乘 |
| `vectorized_elementwise<CUDAFunctorOnSelf_add>` | 60 | 2 | 3.58 | 3.57 | 0.12 | 2 × 标量加 |

**读出来的事实：**

1. **9/步与代码严丝合缝**：3 个 GEMM（fc1/fc2/fc3）+ 2×(gelu+mul+add)。`forward` 里写什么，kernel 表里就是什么——这就是 eager 的“直译”特性。
2. **GEMM 两种配置**：fc1 与 fc3 维度不同但恰好映射到同一种 cutlass 分块配置（128×128，共 60 实例）；fc2（2048→2048）用另一种（128×64，30 实例）。同一行里混着两个不同形状的 GEMM，这也是 Avg 与 Med 差距较大的原因之一。
3. **时间占比**：GEMM 每步 394.7 µs，pointwise 每步 22.9 µs——**GEMM 占了 GPU 时间的 94.5%**。记住这个数字，第六章要用。
4. **噪声预警**：第二行 StdDev（91.6 µs）比 Med（54.1 µs）还大，Max 冲到 462 µs——这次测量受了明显干扰。对比第 5.3 节同一 kernel 的数据，你会看到“噪声长什么样”。

**CPU 侧（cuda_api_sum）**：

| API | 调用次数 | 每步 | 总时间 | 解读 |
|---|---|---|---|---|
| `cudaDeviceSynchronize` | 1 | — | 6.22 ms | 循环结束后的 `torch.cuda.synchronize()`，**是“尾部等待”，不是浪费**（见下） |
| `cudaLaunchKernel` | 180 | 6 | 1.28 ms | PyTorch 原生算子（gelu/mul/add）走的 **Runtime API** 通道 |
| `cuLaunchKernel` | 90 | 3 | 0.65 ms | cuBLAS/cutlass GEMM 内部走的 **Driver API** 通道 |
| `cudaMemsetAsync` | 90 | 3 | 0.66 ms | 每个 GEMM 调用伴随的缓冲/工作区清零操作 |
| `cuKernelGetFunction` / `cuKernelGetName` | 90 / 180 | — | ~0.07 ms | kernel 元数据查询，首次使用/跟踪时发生，忽略 |

**180 + 90 = 270 = 9 × 30，与 kernel 数完全一致**——每个 kernel 恰好一次 launch，一个不多一个不少。这就是“eager 无优化”的量化画像。

**一个重要推算（后面反复用）：**

> 循环墙钟 ≈ CPU 发射总耗时 + 末尾 sync 等待。
> 当 GPU 无空隙时：**CPU 发射总耗时 ≈ GPU kernel 总时间 − cudaDeviceSynchronize 时间**。
> （`cudaDeviceSynchronize` 的时长 = CPU 交完所有工作后，还要等多久让 GPU 收尾。）

eager：12.53 ms − 6.22 ms ≈ **6.3 ms 的 CPU 发射时间 ≈ 210 µs/步**。也就是说，CPU 每步要花约 210 µs 才能把 9 个 kernel 喂出去（其中 CUDA API 只占约 86 µs，其余是 Python 解释器和 PyTorch dispatcher——这部分**不会**出现在 `cuda_api_sum` 里，只在 GUI 的 CPU 时间线上可见），而 GPU 每步只需 418 µs 消化。GPU 是瓶颈，但 CPU 的余量只有 2 倍——把 batch 调小，天平立刻倾斜（见练习 2）。

```
eager 每步示意（9 个 kernel）：
CPU: [launch GEMM][launch gelu][launch mul][launch add][launch GEMM]…   ← ~210 µs
GPU:  [GEMM][gelu][mul][add][GEMM][gelu][mul][add][GEMM]                ← ~418 µs
```

### 5.2 compile：融合的直接证据

**GPU 侧**——4 行，共 150 实例 = **5 个 kernel/步**：

| kernel（简写） | 实例 | 每步 | Avg (µs) | 对应内容 |
|---|---|---|---|---|
| `cutlass_80_tensorop_s1688gemm_256x64_16x4` | 30 | 1 | 193.8 | 某个 GEMM（Inductor 选了与 eager 不同的分块） |
| `cutlass_80_tensorop_s1688gemm_128x64_16x6` | 60 | 2 | 79.2 | 另两个 GEMM |
| `triton_poi_fused_add_addmm_gelu_mul_0` | 30 | 1 | 5.86 | **融合 kernel ①** |
| `triton_poi_fused_add_addmm_gelu_mul_1` | 30 | 1 | 4.09 | **融合 kernel ②** |

**读出来的事实：**

1. **kernel 名自述了融合内容**：`triton_poi_fused_add_addmm_gelu_mul`——`addmm`（带 bias 的线性层输出）、`gelu`、`mul`、`add` 被编进**一个** Triton kernel；`poi` = pointwise，`_0/_1` = 两处融合点（fc1 后与 fc2 后的链）。
2. **融合前后对比**：pointwise 从 6 个 kernel、22.9 µs/步 → 2 个 kernel、**9.9 µs/步**；kernel 总数 9 → 5。
3. **GEMM 依旧是 GEMM**：该是 GEMM 的还是 GEMM，只是 Inductor 换了分块配置（每步 GEMM 合计 352.1 µs，低于 eager 的 394.7 µs；注意其中一部分是配置差异，一部分是运行噪声，两者数量级相当，别把功劳全记在配置上）。
4. **GPU 总时间 12.53 → 10.86 ms（−13%）**——本实验 GEMM 占比太高，融合的 GPU 收益被摊薄了；在 pointwise 密集的模型（LayerNorm/Dropout/激活多的网络）里，这个数字会大得多。

**CPU 侧：**

| API | 次数 | 每步 | 解读 |
|---|---|---|---|
| `cudaDeviceSynchronize` | 1 | — | 8.05 ms |
| `cuLaunchKernel` | 150 | 5 | **只剩 Driver API**：Triton 和 cutlass 都走这条路；`cudaLaunchKernel` 彻底消失 |
| `cudaMemsetAsync` | 60 | 2 | 从 90 降到 60（Inductor 的内存规划消掉了一部分） |
CPU 发射时间 ≈ 10.86 − 8.05 ≈ 2.8 ms ≈ **94 µs/步**，比 eager 的 210 µs 省一半以上——**融合不仅省 GPU 时间，也省 CPU 的 dispatch 次数**，这是很多人忽略的附带收益。

### 5.3 cudagraph：图不改变 GPU 干的活，只改变“怎么发”

**GPU 侧**——与 eager **完全相同**的 5 行、270 实例（9/步）：同样的 GEMM、同样的 gelu/mul/add。这是理解 CUDA Graph 的关键一课：

> **CUDA Graph 是“发射方式的优化”，不是“计算方式的优化”。** GPU 上执行的 kernel 一个没少、一个没变。

顺带看一个绝佳的噪声教学案例：`128x64_16x6` 这个 GEMM 在 eager 里是 77.9 µs ± 91.6，在这次 cudagraph 运行里是 **54.2 µs ± 0.55**——同一个 kernel、同一个形状，Avg 差了 44%！区别只在于后者 StdDev 只有 546 ns，数据极其稳定；前者的均值被干扰拖高了。**教训：比较耗时先看 StdDev，再看 Avg/Med；跨运行的百分比级差异，先怀疑噪声。**

**CPU 侧：**

| API | 次数 | 每步 | 解读 |
|---|---|---|---|
| `cudaDeviceSynchronize` | 1 | — | 12.30 ms（占 93.7%） |
| `cudaGraphLaunch_v10000` | 30 | **1** | **每步一次整图回放，替代 9 次 launch** |
| `cudaStreamIsCapturing_v10000` | 30 | 1 | PyTorch 每步回放前的例行检查（防止捕获态下的非法操作），亚微秒级，忽略 |
| `cudaLaunchKernel` / `cuLaunchKernel` / memset | **0** | 0 | **全部消失——都被录进图里了** |

两个细节值得咀嚼：

- **`cudaLaunchKernel` 从表里消失**，就是“launch 开销被消除”的直接证据；30 次图启动的 API 总耗时仅 0.79 ms（26.4 µs/次）。
- `cudaDeviceSynchronize` 占 93.7%——CPU 几乎瞬间交完所有活，然后全程干等 GPU。**CPU 瓶颈被彻底移除**。（有趣的是这次 sync 12.30 ms 甚至略大于 kernel 总和 11.89 ms——因为 kern_sum 不统计图内的 memset 节点、kernel 间还有微小间隙。**每张表只覆盖一类事件，跨表推算要留余量**，这也是一课。）

```
cudagraph 每步示意（同样的 9 个 kernel）：
CPU: [cudaGraphLaunch]                                          ← ~26 µs
GPU:  [GEMM][gelu][mul][add][GEMM][gelu][mul][add][GEMM]        ← 整图一次性提交
```

### 5.4 compile+cudagraph：reduce-overhead 的“自动版”

**GPU 侧**——5 行，共 180 实例 = **6 个 kernel/步**：

| kernel（简写） | 实例 | 每步 | Avg (µs) | 对应内容 |
|---|---|---|---|---|
| `cutlass_80_tensorop_s1688gemm_256x64_16x4` | 30 | 1 | 215.8 | GEMM |
| `cutlass_80_tensorop_s1688gemm_128x64_16x6` | 60 | 2 | 84.1 | GEMM ×2 |
| `multi_tensor_apply_kernel` | 30 | 1 | 14.6 | **新面孔：输入拷贝** |
| `triton_poi_fused_add_addmm_gelu_mul_0` | 30 | 1 | 5.95 | 融合 kernel ① |
| `triton_poi_fused_add_addmm_gelu_mul_1` | 30 | 1 | 4.25 | 融合 kernel ② |

**`multi_tensor_apply_kernel` 是谁？** 它就是手动模式注释里那句“换数据只能 `copy_`”的**自动化版本**：`reduce-overhead` 模式下，torch.compile 的 cudagraphs 后端每步先把你的新输入**拷贝进静态缓冲区**（foreach 批量拷贝），再回放整图。代价是每步多一个 14.6 µs 的小 kernel——用一个小拷贝换来“你可以随便传新 tensor、不用自己管静态地址”的便利。

**CPU 侧：**

| API | 次数 | 每步 | 解读 |
|---|---|---|---|
| `cudaDeviceSynchronize` | 1 | — | 10.52 ms（88.9%） |
| `cudaGraphLaunch_v10000` | 30 | 1 | 图回放 |
| `cudaLaunchKernel` | 30 | 1 | **那个输入拷贝 kernel 在图外**，每步正常 launch 一次 |
| `cudaStreamIsCapturing` / `cuKernelGetName` | 30 / 30 | — | 例行检查与图节点名称查询，忽略 |

每步 = 1 次图 + 1 次 kernel launch = 60 次总计。CPU 发射时间 ≈ 12.27 − 10.52 ≈ 1.75 ms ≈ **58 µs/步**。

---

<a id="六"></a>

## 六、横向对比与六个结论

### 每步指标总表（全部由上面的原始数据换算而来）

| 指标（每步） | eager | compile | cudagraph | compile+cudagraph |
|---|---|---|---|---|
| GPU kernel 个数 | **9** | **5** | 9 | **6**（含 1 个拷贝） |
| 其中 pointwise | 6 个 / 22.9 µs | 2 个 triton / 9.9 µs | 6 个 / 23.8 µs | 2 triton + 1 拷贝 / 24.8 µs |
| GEMM 耗时 | 394.7 µs | 352.1 µs | 372.4 µs | 384.0 µs |
| GPU 忙时合计 | 417.6 µs | 362.1 µs | 396.3 µs | 408.9 µs |
| GEMM 占 GPU 时间 | 94.5% | 97.2% | 94.0% | 93.9% |
| launch API 次数/步 | **9**（180 runtime + 90 driver） | 5 | **1** | 2（1 图 + 1 kernel） |
| 30 步 launch 总数 | 270 | 150 | 30 | 60 |
| CPU 发射耗时/步（估算） | ~210 µs | ~94 µs | **~26 µs**（纯 API） | ~58 µs |
| 末尾 sync 等待 | 6.22 ms | 8.05 ms | 12.30 ms | 10.52 ms |

### 六个结论

**结论 1：融合的证据是确凿的。** pointwise 从 6 个 kernel、22.9 µs/步 → 2 个 `triton_poi_fused` kernel、9.9 µs/步（时间 −57%，数量 −67%）。kernel 名里直接写着它融合了什么。

**结论 2：CUDA Graph 的证据也是确凿的。** kernel 数不变（270），但 launch 从 270 次降到 30 次 `cudaGraphLaunch`，`cudaLaunchKernel`/`cuLaunchKernel`/memset API 全部从表里消失。CPU 每步成本从 ~210 µs 降到 ~26 µs，**约 8 倍**。

**结论 3：reduce-overhead = 融合 + 图 + 一个“买路钱”。** 每步多一个 14.6 µs 的 `multi_tensor_apply`（自动输入拷贝），换来无需手动管理静态缓冲的开发体验。天下没有免费的午餐，但这份账单很便宜。

**结论 4：这个实验是 GPU-bound 工况，四种模式墙钟差距不大。** GEMM 占了 94–97% 的 GPU 时间，CPU（最慢的 eager 也只要 210 µs/步）始终喂得上 GPU（≥362 µs/步）。所以本实验里 launch 优化的墙钟收益有限——**launch 开销优化的价值要在 CPU-bound 场景（小 batch、小模型、kernel 又小又多）才能兑现**。把 `--batch` 改成 16 重跑一遍，你会看到完全不同的故事（练习 2）。

**结论 5：不要被单次测量的噪声骗。** 同一个 GEMM kernel 两次运行 Avg 相差 44%（StdDev 91.6 µs vs 0.55 µs）；compile 与 eager 的 GEMM 差异也在噪声量级。**稳健的指标是“次数”和“名称”（270→150→30，triton 出现、multi_tensor_apply 出现），时间类指标要看 Med/StdDev 并多次运行。**

**结论 6：一行精度设置可能胜过所有 launch 优化。** 作为参照，在同一台机器上不设 `set_float32_matmul_precision('high')` 时，GEMM 走的是 `simt_sgemm`（纯 FP32 CUDA Core），每步 GEMM 约 650 µs；开启后走 `tensorop_s1688gemm`（TF32 Tensor Core），降到约 395 µs——**一行代码的收益（约 1.6 倍）超过了本文所有 launch 层优化的总和**。教训：先找最大头（本例是 GEMM 及其精度），再抠发射开销。

### 实践建议：什么时候用哪个

| 场景 | 推荐 | 注意事项 |
|---|---|---|
| 调试、科研原型 | eager | 行为最直译，报错最友好 |
| 通用提速 | `torch.compile` | 首次运行有编译开销；动态 shape 会触发重编译 |
| 极致低延迟、输入/输出地址可控 | 手动 `CUDAGraph` | 换数据只能 `copy_`；捕获期间不能有 CPU 同步、动态控制流；务必先旁路 warmup |
| 想兼得融合与低开销 | `mode="reduce-overhead"` | **输出张量会被下一次回放覆盖**，跨步持有结果要先 `clone()`；个别不支持的算子会退化为图外 launch（正如我们看到的那个拷贝 kernel） |

---

<a id="七"></a>

## 七、GUI（nsys-ui）时间线怎么看

```bash
nsys-ui eager.nsys-rep        # 或 GUI 菜单 File → Open 加载其他报告
```

启动时可能看到：

```
OpenGL version is too low (0). Falling back to Mesa software rendering.
Skipping OpenGL version check on WSL.
```

这只是说 WSL 里没有 GPU 硬件加速的 OpenGL，回退到软件渲染——**GUI 完全能用，只是缩放平移略卡**。想要流畅体验：`.nsys-rep` 是跨平台格式，把它拷到 Windows（如 `cp eager.nsys-rep /mnt/c/Users/你/Desktop/`），用 Windows 版 Nsight Systems 打开即可。

### 界面结构（自上而下）

```
时间标尺（顶部）
└─ python 进程
   ├─ NVTX            ← MODE:eager 大区间，内嵌 step_0 … step_29（你 push 的书签）
   ├─ CPU 线程         ← Python 解释、PyTorch dispatch 都发生在这里（api_sum 里看不到的部分）
   ├─ CUDA API        ← cudaLaunchKernel / cudaGraphLaunch / cudaDeviceSynchronize 小色块
   ├─ OS runtime      ← 驱动与系统调用
   └─ CUDA → GPU 0
      ├─ Kernel/Stream 行 ← 真正执行的 kernel 色块（重点看这行）
      └─ Memory        ← memset 等显存操作
```

### 核心操作

- **滚轮缩放、拖拽平移**；在标尺上横向拖选可测量区间时长；
- **双击某个 NVTX 区间（如 `step_5`）→ 自动放大聚焦到这一步**（最高频动作）；
- **单击 kernel 色块** → 详情面板显示完整名称（解决终端截断问题）、grid/block、时长、stream、correlation id——顺着它还能找到发射它的 CPU API 调用；
- 对比两个模式：File → Open 打开第二份报告，各自双击同一个 `step_i`，肉眼对比 GPU 行的形状。

### 四份报告各自的“标准景象”

| 报告 | CUDA API 行 | GPU Kernel 行 |
|---|---|---|
| eager | 每步 9 个小刻度 + 3 个 memset | 每步 9 块，节奏 `[GEMM, gelu, mul, add] ×2 + GEMM`；**观察块间空隙**——空隙大说明发射跟不上 |
| compile | 每步 5 个刻度 | 每步 5 块，三小块合并成一块 triton（悬停看名） |
| cudagraph | 每步只剩 1 个 `cudaGraphLaunch` 刻度 | kernel **背靠背挤成一整排**（整图已提交，无发射间隙）——与 eager 的“有缝”对比最震撼 |
| compile+cudagraph | 每步 1 图 + 1 kernel 刻度 | 每步开头一个小拷贝块 + 一整包图 kernel |
> 提醒：图模式能看到包内每个 kernel，靠的就是采集时的 `--cuda-graph-trace=node`。没有它，整张图是单个黑盒。

---

<a id="八"></a>

## 八、Kernel 名解码速查表

初学者最大的障碍是“这串名字是什么鬼”。收藏这张表：

| 名称片段 | 含义 |
|---|---|
| `at::native::vectorized_elementwise_kernel` | PyTorch 原生逐元素 kernel（**未融合**的标志） |
| `GeluCUDAKernelImpl` / `AUnaryFunctor` / `CUDAFunctorOnSelf_add` | 具体算子：gelu / 一元运算（标量乘）/ 自身加 |
| `cutlass::Kernel2<...>` | NVIDIA CUTLASS 模板库生成的 GEMM |
| `cutlass_80_` | 面向 SM80（Ampere）架构 |
| `tensorop_s1688gemm` | **TF32 Tensor Core** GEMM（对应 `set_float32_matmul_precision('high')`） |
| `simt_sgemm` | 纯 FP32 CUDA Core GEMM（未开 TF32 时出现） |
| `128x128_16x5_tn_align4` | GEMM 分块（tile）配置：128×128 分块、TN 布局、对齐 4——同一个 GEMM 换个分块，性能可能差很多 |
| `triton_poi_fused_xxx_yyy_zzz` | **Inductor 生成的 Triton 融合 kernel**；`poi`=pointwise，后面列出被融合进来的算子链 |
| `multi_tensor_apply_kernel` | foreach 批量张量操作（本例中 = 输入拷入静态缓冲） |
| `cudaGraphLaunch` | 整图一次回放的 API（CUDA Graph 的标志） |
| `cudaStreamIsCapturing` | PyTorch 回放前的“是否在捕获中”例行检查，开销可忽略 |

---

<a id="九"></a>

## 九、视野扩展：推理优化的全景图

本文实验覆盖的是推理优化三大基石中的两个，把它们放进地图里看：

| 基石 | 优化层面 | 解决的问题 | 本文对应 |
|---|---|---|---|
| **算子融合** | 计算图 / kernel 执行 | 减少 kernel 数量与中间显存读写，提升 GPU 计算效率 | compile 模式 ✅ |
| **CUDA Graph** | CPU–GPU 交互 | 消除 CPU launch 瓶颈 | cudagraph 模式 ✅ |
| **量化** | 数值精度 | 降低计算量与内存带宽，充分利用 Tensor Core | ❌（结论 6 的 TF32 只是入门尝鲜） |

除此之外，推理优化还有一整个工具箱，按层次粗分：

| 层次 | 代表技术 |
|---|---|
| 内存与显存 | 内存池复用激活、FlashAttention 式算子内显存规划、权重压缩 |
| 数值精度 | FP16/BF16/TF32/INT8/FP8 混合精度、QAT/PTQ 量化、低精度指令（DP4A 等） |
| 计算图 | 常量折叠、死代码消除、公共子表达式消除、图替换、Winograd/FFT 卷积 |
| 推理引擎 | TensorRT、ONNX Runtime、OpenVINO——把上述手段打包并针对硬件调优 |
| 模型级 | 知识蒸馏、剪枝、NAS、早退机制 |
| 硬件利用 | Tensor Core 与数据布局（NHWC 等）、多 GPU 并行、多流重叠计算与拷贝 |
| 服务与调度 | 动态批处理、Continuous Batching、请求级调度 |
| 系统部署 | 模型 mmap 加载、编译缓存、kernel 自动调优（cuDNN benchmark / Triton autotune） |

它们通常**叠加使用**：TensorRT = 融合 + INT8 + 内存池 + 多流；`torch.compile` = 融合 + CUDA Graph +（可选）混合精度；Triton Inference Server = 动态批处理 + 多模型并发。本文练的“看报告”功夫，是验证所有这些手段是否真正生效的通用显微镜。

---

<a id="十"></a>

## 十、新手学习路径与资料

按入门顺序：

1. **Nsight Systems 官方文档**（docs.nvidia.com/nsight-systems）：User Guide 里的 *Getting Started*、*Reports*（每种 `--report` 的字段解释——正好对应本文第四章）、*Timeline View*（GUI 每一行的含义）。
2. **命令自查**：`nsys stats --help-reports` 列出全部报告；重点补 `cuda_gpu_trace`（逐条 kernel 明细）和 `nvtx_kern_sum`（按 NVTX 区间聚合）。
3. **PyTorch 官方教程**：
   - PyTorch Profiler recipe（`torch.profiler`，产出 Chrome trace，用 [Perfetto](https://perfetto.dev) 打开——比 nsys 更容易上手，适合先建立“时间线”概念）；
   - 博客 *Accelerating PyTorch with CUDA Graphs* + 文档 `torch.cuda.graphs`（本文手动图模式的标准出处）；
   - *torch.compile Tutorial*（讲 Inductor、`reduce-overhead` 与 cudagraphs 的关系）。
4. **NVIDIA 开发者博客的 CUDA Graphs 系列**：解释图回放为何能消除 launch 开销。
5. **中文实操笔记**：知乎/公众号搜 “Nsight Systems 教程 / timeline 分析”，图例丰富，可作为辅助（以官方文档为准）。
**推荐路径**：先用 `torch.profiler` + Perfetto 熟悉“看时间线” → 再用 nsys 看系统级细节（launch 开销、CUDA Graph、cuBLAS 行为）→ 最后用 `nsys stats`/sqlite 做定量对比。本文练的正是最后一步。

---

<a id="十一"></a>

## 十一、动手练习

1. **数 launch**：跑 `nsys stats eager.nsys-rep --report cuda_api_sum`，验证 launch 总数 = 270；再用 `--report cuda_gpu_trace` 找出一步内 9 个 kernel 的执行顺序，与 `forward` 代码逐行对应。
2. **制造一个 CPU-bound 场景**：`--batch 16` 重跑四种模式。预期：eager 的 GPU kernel 间出现大空隙（发射跟不上），cudagraph 的墙钟收益从“几乎为零”变成“数倍”。这是理解两种瓶颈最好的实验。
3. **按区间统计**：用 `--report nvtx_kern_sum` 回答“每个 `step_i` 里 GPU 忙了多少 µs”，与本文“GPU 忙时/步”对账。
4. **TF32 消融**：注释掉 `set_float32_matmul_precision('high')` 重跑 eager，观察 GEMM kernel 名从 `tensorop_s1688` 变回 `simt`，耗时如何变化（结论 6 的亲自验证版）。
5. **缓存陷阱**：重新 profile 同名报告后，`nsys stats` 加 `--force-export=true`，体会 sqlite 缓存机制。
6. **GUI 测量**：双击 `step_10`，用标尺量出一步总时长，减去 kernel 忙时，差值就是发射间隙——把这个数和第六章的 CPU 估算表对照。

---

<a id="附录"></a>

## 附录：复现命令清单

```bash
# 四种模式采集（仅 -o 与 --mode 不同）
nsys profile -t cuda,nvtx,osrt,cudnn,cublas -s none --cpuctxsw=none \
  --cuda-graph-trace=node --capture-range=cudaProfilerApi \
  --capture-range-end=stop-shutdown -o eager -f true python bench.py --mode eager
nsys profile ... -o compile    -f true python bench.py --mode compile
nsys profile ... -o cudagraph  -f true python bench.py --mode cudagraph
nsys profile ... -o compile_cudagraph -f true python bench.py --mode compile+cudagraph

# 每份报告看两张表
nsys stats eager.nsys-rep             --report cuda_gpu_kern_sum   # GPU 干了什么
nsys stats eager.nsys-rep             --report cuda_api_sum        # CPU 花了多大功夫

nsys stats compile.nsys-rep           --report cuda_gpu_kern_sum
nsys stats compile.nsys-rep           --report cuda_api_sum

nsys stats cudagraph.nsys-rep         --report cuda_gpu_kern_sum
nsys stats cudagraph.nsys-rep         --report cuda_api_sum

nsys stats compile_cudagraph.nsys-rep --report cuda_gpu_kern_sum
nsys stats compile_cudagraph.nsys-rep --report cuda_api_sum

# 可视化
nsys-ui eager.nsys-rep
```

---

## 结语

把整篇实验压缩成三句话：

> **融合**让 pointwise 从 6 个 kernel 变 2 个（GPU 侧少干活）；**CUDA Graph** 让每步 launch 从 9 次变 1 次（CPU 侧少费劲）；而在这台 GPU 上，**GEMM 占了 94% 以上的时间**——先解决大头的精度与效率，再抠发射开销，才是正确的优化顺序。

而 nsys 教给我们的方法论同样只有三句话：

> **`cuda_gpu_kern_sum` 看 GPU 干了什么，`cuda_api_sum` 看 CPU 付了多大代价，GUI 时间线看两者如何在时间上交错。** 次数与名称是最稳健的证据，时间要盯住 Med 和 StdDev，跨表推算记得留余量。

祝 profiling 愉快——当你能从一张表格里“读出故事”时，性能优化就不再是一门玄学。

## 附录：完整代码

````python title="bench.py"
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

Capture range started in the application.   ← 程序调用了 cudaProfilerStart()，开始记录（对应代码里的 profiler.start()）
Capture range ended in the application.     ← 调用了 cudaProfilerStop()，停止记录
Generating '/tmp/nsys-report-8ae4.qdstrm'   ← 先生成原始事件流（中间文件）
[1/1] [====100%] eager.nsys-rep             ← 把事件流加工成最终报告
Generated: /home/beaver/eager.nsys-rep      ← 最终产物，GUI 和 stats 都吃这个文件

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
````
