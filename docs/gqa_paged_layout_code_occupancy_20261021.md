# GQA 分页读取、8 行布局与 occupancy：从代码到证据

2026-10-21，LeanScale EXP-0013。比较对象是 EXP-0012 的 stage1 版本与 EXP-0013 的 `decode_layout=1, decode_rows=8`；两者均为 4 warps、1 stage、16 segments。下述改动用于 decode，prefill 保持原路径。

CoreX 驱动对真实编译内核的查询结果：理论最大驻留块数从 6 增加到 7，按驱动报告的资源口径计算，理论 occupancy 从 18.75% 增加到 21.875%。这支持“提高了理论 occupancy”，但本实验没有测量运行时 achieved occupancy，也不能把全部加速归因于 occupancy。

## 1. 先明确一个线程块处理什么

模型有 16 个 Q head、2 个 KV head，head dimension 为 128。每 8 个 Q head 共用一个 KV head：

```text
Q head  0,1,2,3,4,5,6,7   → KV head 0
Q head  8,9,10,11,12,13,14,15 → KV head 1
```

decode 时，每条请求只有一个新 query token。现有 launch grid 为 `(nseq, 2, 16)`，分别对应请求、KV 分组、历史 KV 的分段。一个线程块处理“一个请求的一个 KV 分组的一段历史”，输出该段的中间结果，再由原有 `_reduce` 合并。此次没有改变 head 分组、分段数或归约方式。

[冻结源码](https://github.com/yyq0210/FlagGems/blob/1c526ca99d412e203ed3caa8b1663566976ea38c/src/flag_gems/runtime/backend/_iluvatar/fused/paged_gqa.py)。查询行见 L55–L74，页表读取见 L75–L99，launch 见 L175–L177。

## 2. 分页读取：减少重复的页号查找

KV cache 已经是分页存储。优化改的是页表 `BT` 的读取表达式，不是重新分页、压缩 KV 或减少历史 token。

下面以 `PAGE=16`、`TILE=32`、`base=0` 为例。每页放 16 个 token，一次循环处理 32 个历史 token。假设这条请求的页表前两项为 `[7, 23]`，即逻辑第 0 页位于物理第 7 页，逻辑第 1 页位于物理第 23 页。页号仅为讲解示例。

原来的代码按 token 建立页表读取索引：

```python
t = tl.arange(0, TILE)                 # 0,1,...,31
tokens = base + t
blocks = tl.load(
    BT + seq * BTS + tokens // PAGE,
    tokens < length, 0,
).to(tl.int64)
```

`seq * BTS` 定位本请求的页表，`tokens // PAGE` 是逻辑页号。代入示例：

```text
tokens // 16 = [0,0,...,0, 1,1,...,1]  # 各 16 项
blocks       = [7,7,...,7, 23,23,...,23]
```

源码表达的是一个 32 元素的页号加载向量，里面只有两个不同的页表地址。

候选代码先取两个页号，再广播到 token 对应的位置：

```python
page_ids = base // PAGE + tl.arange(0, TILE // PAGE)
page_blocks = tl.load(
    BT + seq * BTS + page_ids,
    page_ids * PAGE < length, 0,
).to(tl.int64)
blocks = tl.reshape(
    tl.broadcast_to(page_blocks[:, None], (TILE // PAGE, PAGE)),
    (TILE,),
)
```

逐步代入：

```text
page_ids:       [0, 1]                  shape [2]
page_blocks:    [7, 23]                 shape [2]
加一个维度:      [[7], [23]]             shape [2,1]
广播:           [[7,...,7], [23,...,23]] shape [2,16]
展平 blocks:    [7,...,7,23,...,23]      shape [32]
```

两种方法得到相同的 `blocks`。后续读取 K/V 的地址不变，例如 K 的地址仍为：

```python
k = tl.load(
    K + blocks[None, :] * K0
      + (tokens % PAGE)[None, :] * K1
      + group * K2 + d[:, None],
    tokens[None, :] < length, 0,
)
```

其中 `blocks * K0` 定位物理页，`tokens % PAGE` 定位页内 token，`group` 选择 KV head，`d` 选择 head dimension。页号先转 int64 再乘 stride，保留大于 2^31 元素地址的正确性。

广播路径只在 decode 且 `PAGE <= TILE`、`TILE % PAGE == 0` 时启用，循环起点与 TILE 对齐；其他布局沿用原读取路径。IR 确认页号加载张量由 32 元素变为 2 元素，但这不等于显存事务减少 16 倍：旧读取可能被缓存、合并或编译器处理，真实事务数需要硬件计数器。K/V 本体的读取量不因此减少。

## 3. 8 行布局：去掉 8 行无效的逻辑状态

实际代码用 `BM` 定义一个分组的查询矩阵行数：

```python
r = tl.arange(0, BM)
d = tl.arange(0, 128)
pos = tl.full((BM,), 0, tl.int32)  # decode：同一个新 token
valid = r < 8                    # 一个 KV 分组只有 8 个 Q head
head = group * 8 + r % 8
q = tl.load(
    Q + (start + pos[:, None]) * QS + head[:, None] * QH + d[None, :],
    valid[:, None], 0,
)
m = tl.full((BM,), float('-inf'), tl.float32)
l = tl.zeros((BM,), tl.float32)
acc = tl.zeros((BM, 128), tl.float32)
```

旧配置 `BM=16`：`r=0..15`，但 `valid` 只有前 8 项为真。后 8 行加载为 0，后续 attention 分数被 mask 掉，最终不产生有效输出。新配置 `BM=8`：`r=0..7`，正好对应全部 8 个有效 head。

| 逻辑张量 | BM=16 | BM=8 |
| --- | --- | --- |
| Q | 16×128，只有前 8 行有效 | 8×128，全部有效 |
| QK 分数，TILE=32 | 16×32 | 8×32 |
| softmax 的 m/l | 各 16 元素 | 各 8 元素 |
| 输出累加 acc | 16×128 | 8×128 |

这只是内核中间张量的逻辑布局。模型仍有 16 个 Q head，KV cache 物理布局未改，线程块也仍为 4 warps。CoreX 的 warp_size 为 64，因此两版都是每块 256 个线程，不是把线程数从 16 变成 8。

实际 launch 由下面的参数传入 `BM`：

```python
_attention[(nseq, 2, segments)](
    *args, decode_tile, decode_rows, True, decode_layout,
    num_warps=decode_warps, num_stages=decode_stages,
)
# 参考：decode_rows=16, decode_layout=0
# 候选：decode_rows=8,  decode_layout=1
```

TTGIR 中 Q/acc 的逻辑行数确实从 16 降为 8，但矩阵指令 `instrShape=[16,16,16]` 保持不变。因此不能宣称 Tensor Core 乘加数量减半。编译器最终报告寄存器 96→83、shared 20480→18432 字节，局部内存均为 0；这些是独立可核验的资源结果。

## 4. occupancy 是否增加：本机驱动查询支持理论增加

2026-10-21 20:35，在共享 GPU 锁内执行 `probe_occupancy.py`，对四个消融配置、三个工作点，共 12 个编译内核查询 CoreX 原生接口：

```python
ixdrvOccupancyMaxActiveBlocksPerMultiprocessor(
    &active_blocks, kernel.function,
    blockSize=256,
    dynamicSMemSize=kernel.metadata.shared,
)
```

上面是便于阅读的调用示意，实际可执行 ctypes 版本见 [查询脚本](../experiments/EXP-0013/probe_occupancy.py)。函数句柄来自实际 Triton 编译和加载；三个工作点为 B64/L4700、B30/L16384、B1/L4096。所有 12 个内核的 TTIR/TTGIR/LLVM IR/ASM 哈希与原筛选记录完全一致。

本机 CoreX 驱动报告的每 SM 资源：shared 128 KiB、262144 个寄存器、最多 8192 个线程；warp_size=64。以下 occupancy 使用该驱动的 SM/线程口径，不能直接代入 NVIDIA 某代芯片的上限进行比较。

| 配置 | 寄存器/线程 | shared/块 | 线程/块 | 驱动预测块/SM | 预测 warp/SM | 理论 occupancy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| layout0/rows16 | 96 | 20 KiB | 256 | 6 | 24 | 18.75% |
| layout1/rows16 | 96 | 20 KiB | 256 | 6 | 24 | 18.75% |
| layout0/rows8 | 83 | 18 KiB | 256 | 7 | 28 | 21.875% |
| layout1/rows8 | 83 | 18 KiB | 256 | 7 | 28 | 21.875% |

三个工作点的结果一致；静态 shared 为 0，表中 shared 为动态 shared。

共享内存的整数门槛恰好解释了 6→7：

```text
旧版：floor(128 / 20) = 6 块；7×20 = 140 KiB，放不下
新版：floor(128 / 18) = 7 块；7×18 = 126 KiB，可以放下
```

忽略分配粒度时，寄存器预算分别为：

```text
旧版：floor(262144 / (96×256)) = 10 块
新版：floor(262144 / (83×256)) = 12 块
```

寄存器压力降低，但上述预算都高于 shared 允许的 6/7 块。实际驱动查询也返回 6/7，所以本次跨越的直接可解释门槛是 shared；不应写成“寄存器减少 13 个，所以 occupancy 必然增加”。寄存器/共享内存有分配粒度、分区和其他限制，通用手算必须由实际平台接口校验。

```text
理论 occupancy = 预测驻留块数 × 每块线程数 / 每 SM 最大线程数
旧版 = 6 × 256 / 8192 = 18.75%
新版 = 7 × 256 / 8192 = 21.875%
增加 3.125 个百分点，相对增加 16.67%
```

occupancy 的一般定义和资源限制可参考 [CUDA Best Practices 的 occupancy 章节](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy)，本报告的芯片数值来自 CoreX 查询。理论预测与运行时 achieved occupancy 是不同证据：后者还取决于网格工作量、块的执行时长、尾部负载和调度，需 profiler 计数器确认。

## 5. 为什么不能把所有收益归因于 occupancy

第一，layout1/rows16 的理论 occupancy 与旧版完全相同，但 B64/L4700 完整 GQA 时间已减少 12.483%。页号表达式的改进能在不改变理论 occupancy 的情况下产生收益；具体指令/访存贡献仍需单独 profiling。

第二，B1 时主核网格只有 `1×2×16=32` 个块，驱动报告 16 个 SM。网格总工作量不足以同时填满全设备原来的 `16×6=96` 个驻留槽，更不可能填满新版的 112 个槽。B1 仍有约 23% 时间下降，进一步说明“最高可驻留块数增加”不是全部加速的充分解释。

第三，单独 rows8 的收益与“减少无效逻辑状态”和“资源上限增加”同时发生；当前消融没有把两者完全隔离。结论应是：理论并发容量提高，且页表读取与中间张量布局共同改善了性能；尚未精确分解每个机制的贡献。

## 6. 扩大验证和后续门槛

扩大验证于 20:30:05 完成并归档到提交 `03a4a87d9a02f1e61807f17881fa231846cc48b1`。26 项数值/Graph 检查全部通过，含实际大于 2^31 元素的 KV 地址；34 个计时点均无超过 3% 回退，前后基线漂移均小于 0.5%。30 个纯 decode 点时间减少约 22.42%–27.96%，两个 mixed 点减少约 2.35%–2.39%，纯 prefill 的变化约 +0.018%/−0.066%。

这是单算子验证，不是整模收益。下一步接入独立引擎实验，验证实际服务走到 layout1/rows8 后，再按官方 4K、16K 配置各做完整四轮、丢弃首轮。完整端到端结果出炉前，已选服务基线保持 EXP-0012。

证据：[occupancy 原始查询](../experiments/EXP-0013/artifacts/20261021_203527_occupancy/occupancy.json)、[12 个内核的 IR 一致性核验](../experiments/EXP-0013/artifacts/20261021_203527_occupancy/screen_hash_verification.json)、[扩大验证](../experiments/EXP-0013/artifacts/20261021_202733_extended/validation.json)、[初筛结果](gqa_paged_layout_screen_20261021.md)。
