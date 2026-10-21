# GQA 在天数 BI-V150 上的实现与下一轮优化分析

日期：2026-10-21。队名：LeanScale。

结论：值得继续优化，优先针对 decode 主内核的线程与数据布局、分段策略和 KV 访存。现有实现已经具有 GQA 共享 KV、在线 softmax、QK/PV 融合和 split-KV 并行，不能把这些已有机制重新算作待实现收益。此次只分析源码、已导出的 trace 与编译证据；新候选尚未测出收益，正在运行的实验源码没有改动。

## 1. 证据及统计口径

固定插件提交 `07d9fa0cfdc5f09db178e29508e3c0bcb5c1e908`，FlagGems 提交 `484413ea0db18221dd9e63e742361d837df9ff13`。

本次使用 EXP-0009 的官方形状 4096 输入 / 1024 输出 / 并发上限 64 / 256 请求，在 `NONE/late` 窗口采到的 16 个纯 decode 迭代。每步 B=64、每请求一个 query token，CPU 元数据中的上下文长度上界范围为 4618–4761。672 次 `_attention` 与 672 次 `_reduce` 均可逐步核对为 16 步 × 42 层，所有 7072 个 GPU kernel 都关联到 step。

| 项目 | 次数 | 平均每次 μs | 窗口 GPU 总耗时 ms | 全部可见 kernel 时间占比 |
|---|---:|---:|---:|---:|
| GQA 主内核 `_attention` | 672 | 1672.156 | 1123.689 | 78.340% |
| 分段归约 `_reduce` | 672 | 20.395 | 13.705 | 0.955% |
| 两者合计 | — | 1692.550 / 层 | 1137.394 | 79.296% |

原始 trace SHA256：`9ae8571d71e9dac3590844e6f6890f1ba3421c14cc3b20af950b8e6c8c646222`。逐算子 CSV、metadata、源码哈希及编译片段见 [证据目录](evidence/gqa_analysis_20261021/)。

这是带 profiler 的 NONE 模式诊断，不是官方四轮吞吐，也不是生产 FULL_DECODE_ONLY 的延迟占比。Graph replay 内部 kernel 在当前探针中不可见。GPU kernel 总和也不能直接当作 TTFT、TPOT 或端到端墙钟时间。

## 2. 当前内核具体怎样计算

入口为 [IluvatarGQAImpl.forward](https://github.com/yyq0210/vllm-plugin-FL/blob/07d9fa0cfdc5f09db178e29508e3c0bcb5c1e908/vllm_fl/dispatch/backends/flaggems/impl/iluvatar_gqa.py#L58)，调用 [FlagGems paged_gqa](https://github.com/yyq0210/FlagGems/blob/484413ea0db18221dd9e63e742361d837df9ff13/src/flag_gems/runtime/backend/_iluvatar/fused/paged_gqa.py#L131)。

模型每层 Q heads=16、KV heads=2、head dimension=128。因此每个 KV head 服务 8 个 Q heads；内核直接共享 K/V，不做 `repeat_interleave`。参考检查中的 repeat 只是 FP32 oracle 的写法。

decode 的实际主 launch 为 `(B, 2, 16)`。在本窗口就是 `(64,2,16)`，共 2048 个 program。每个 program 负责一个请求、一个 KV head、一个 KV 长度分段：

1. 读取该组 8 个 Q heads，用 `BM=16` 的点积块承载；只有 `r<8` 的行有效。
2. 上下文分成 16 段，每段以 32-token KV tile 迭代。分页索引经 block table 转换到物理地址，乘 stride 前转换为 int64。
3. 在同一个 kernel 内计算 `Q @ Kᵀ`、因果/有效范围 mask、在线 softmax、`P @ V`，不把完整 attention score 矩阵写入显存。
4. 写出分段的 FP32 累加结果和 softmax 的最大值、指数和。
5. `_reduce[(B,16)]` 用各段最大值重标定，合并成最终输出。

分段合并为：`M=max(m_s)`，`O=sum(exp(m_s-M)*acc_s)/sum(exp(m_s-M)*l_s)`。因此合并顺序可调整，但必须保留重标定和分母；它不是简单平均分段输出。此模式与 [Flash-Decoding 的 split-KV 和最终归约](https://pytorch.org/blog/flash-decoding/)一致，相关算法收益已经部分包含在当前版本中。

prefill 使用不同分支：`BM=64` 对应 8 个 query tokens × 8 个共享 KV 的 Q heads，以增加每次 KV load 的 query 复用。它用 CU 前缀和二分查找所属请求，再处理 ragged block。混合批同时执行 prefill 主内核、decode 主内核和 reduce；不同分支会跳过不属于自己的序列。

## 3. 主要优化候选及依据

| 优先级 | 候选 | 代码与实测依据 | 需要验证的代价 |
|---|---|---|---|
| P1 | decode 独立调 `num_warps` 与流水级数 | 当前 4 warps；后端 warp_size=64，trace 实际 block=256 线程、111 registers/thread、20 KiB shared memory；BM16 只有 8 行有效 | 先测 2/4 warps、1/2/3 stages；1 warp 或 BM8 必须先确认后端点积支持，不能按 NVIDIA 的 32-thread warp 套参数 |
| P1 | 按工作点选择 split 数 | 固定 16 段并非所有 B、L 都最优，已有 segments=8 的局部数据 | 主 kernel 与 reduce 合计计时；小 batch 可能因并行不足大幅退化；长度分桶要与 CUDA Graph 捕获/重放兼容 |
| P1 | paged KV load 和布局优化 | 每 32 tokens 重做分页地址、读取 K/V，再经过点积与 softmax 布局转换 | 分别验证真实 NHD/HND、共享大 stride、随机/连续物理页；尝试页粒度索引、合法对齐提示、load/stage 调度；保留 int64 地址安全 |
| P1（启动成本） | 避免 decode 对无用 NSEQ 值特化 | NSEQ 为 constexpr，源码只在 prefill 分支使用；decode 仍随 nseq 传入 | 独立 decode 入口去除该参数，或只在 decode 传固定哨兵值；核验缓存 miss/编译次数与 Graph 行为。稳态内核时间未必改变 |
| P2 | 为 8 Q heads 写独立的 decode 计算布局 | 现有 16 行点积有 8 行 padding；IR 可见 dot、归约之间的 layout conversion 和 shared-memory staging | 比较小矩阵 TCU 与向量/SIMT 方案；必须保留 KV 共享，避免按 Q head 独立加载造成额外请求；不能保证 BM8 或去 padding 就快 |
| P2 | 更精确的 prefill/mixed query-block 映射 | upper-bound grid 与二分查找会产生无效 program；纯 prefill 仍启动会提前返回的 decode/reduce | 优先利用已有 CPU metadata；新增映射/H2D 的成本必须计入，不允许 `.item()` 读取 GPU 长度引入同步 |
| P3 | 单 kernel attention+reduce 或 reduce 向量化 | reduce 仅占当前 GQA 两部分耗时约 1.20% | 无 split 会损失并行；保留 split 则需要跨 CTA 协调，复杂度与同步成本可能超过收益 |

### 线程布局是明确的测试方向，但还不是已证明的根因

本机编译元数据 `target.backend=corex, arch=71, warp_size=64`，与 trace 的 `(256,1,1)` block 一致。实际 decode trace 的资源字段记录 `registers per thread=111`、`shared memory=20480`；这些是 kernel 资源元数据，不是动态硬件利用率。本文不使用尚未在 CoreX 上校准的 profiler occupancy 估算作定量归因。

相关 decode 编译样本的 VGPR count 同样为 111、SGPR count=99、SGPR spill count=16、VGPR spill count=0、ScratchSize=0。这提示检查标量寄存器压力，但**不能解释成每线程有 16 次显存 spill**；没有证据说明其产生了 global scratch traffic。prefill 编译样本的 shared memory 为 32 KiB。样本 source location 指向 EXP-0005，同一 GQA 源码沿用至 EXP-0009；trace 未记录 cache key，故对具体编译实例的归属仍保留限制。

8/16 行有效只说明计算形状存在 padding，不等于一半时间可直接消除。TCU 最小 tile、softmax 归约、寄存器布局与流水重叠可能抵消较小 tile 的收益。需要把候选生成的 IR、汇编资源及实际耗时一起对比。

### 分段参数有历史数据，必须区分 batch

来自 EXP-0005 的 CUDA Graph event 微基准，单位 μs，均包含 paged_gqa 的主计算与 reduce；KV tile=32：

| 工作点 | segments=8 | segments=16 | 判断 |
|---|---:|---:|---|
| B1，L4096 | 87.15 | 48.50 | 8 段明显更慢，不可全局改成 8 |
| B64，L4096 | 1399.16 | 1426.52 | 8 段约少 1.9% 时间，是待复测的小收益 |
| B64，L16384 | 5436.00 | 5407.35 | 16 段略快，不能把 4K 结论推广到长文 |

当前 persistent scratch 的 segment 容量固定为 16，8/16 可在现接口内测试；32/64 需要修改 metadata 分配、容量判断和 Graph 的稳定地址，不能只改一个常量。先在 16 以内确认方向。

选择参数时不能在 Graph capture 中按当时的 Python 长度分支，然后默认 replay 时该选择仍适用。需要使用与图 key 一致的 shape bucket，或固定网格配合 GPU 数据分支，并验证同一图在上下文增长、batch padding 与请求替换后的结果。

### KV 读取量大，但尚不能断言已经打满带宽

BF16、Q16/KV2/D128 下，每层每个请求每个历史 token 的最小逻辑 K+V 数据量为：

`2(K/V) × 2(KV heads) × 128 × 2 bytes = 1024 bytes`。

取代表性 B64、L4700，一次层级 decode 约需 308 MB 的逻辑 KV 数据；除以当前主 kernel 约 1.672 ms，得到约 184 GB/s 的“有效 KV 字节/时间”。这是模型估算，不是 DRAM 硬件计数器：L 来自代表值，缓存命中、事务放大、重复取数与 scratch traffic 均未实测。有效点积工作量约为每 KV 字节 8 FLOPs，padding 会增加无效计算。

应测试访存效率，同时检查小矩阵布局、归约及调度停顿；不能从“attention 通常受带宽影响”直接推出本内核已经接近硬件带宽上限。保持当前共享 KV 的优势，比盲目增加 Q-head 并行更重要。

## 4. 哪些看起来可优化、实际上应降低优先级

**额外发现一个编译维度。** `_attention` 的 `NSEQ: tl.constexpr` 只用于 prefill 的二分查找；decode 的 seq 来自 grid，循环边界来自 GPU `LEN`，却仍随 batch 传入不同的 NSEQ。这是可以消除的冗余特化维度，应记录不同 B 下的编译缓存 key/miss 数后验证。它主要影响冷启动与未预热 shape，不应计作稳态主内核加速；生产 FULL_DECODE_ONLY 已预热一组图，收益还可能更小。此次 NONE 负载收尾有明显慢请求，但尚未逐编译事件归因，不能把尾延迟直接归咎于 NSEQ，其他 GEMM/采样等 shape 编译也要检查。

**只扩大 KV tile 已有负面证据。** B64/L4096 时 tile32/segments16 约 1426.52 μs，tile64 约 2377.97 μs，tile128 约 7402.76 μs。较大 tile 并不自动提升访存效率；之前 prefill 的收益来自增加 query 复用，不能套用到 decode。

**仅把 `acc += tl.dot(p,v)` 改写为 `tl.dot(p,v,acc)`，不能当作新融合收益。** 已检查的 optimized TTIR 已把 PV 点积的第三个操作数接到当前 accumulator，而不是零；编译器已经完成该合并。

**仅将 `tl.exp` 拼写改为 `exp2`，也不代表消除了昂贵指数实现。** 当前 LLVM IR 已包含 `llvm.exp2.f32`。如果改成从 QK scale 到分段归约统一使用 log2 域，可能减少部分标量运算，但需要逐项数值验证和 IR 对比，优先级低于主布局/访存。可参考 [Triton 官方 attention 的 exp2 与 accumulator 写法](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)，不能照搬其 CUDA/AMD 特化参数到 CoreX。

**减少一次 reduce launch 不是主要算术收益。** 就这个窗口而言，即使 reduce 的全部 GPU 时间都消失，全模型可见 kernel 总和也只减少约 0.96%；host launch 成本未包含在这个估算中，生产 decode 又已使用 Graph。去掉 split 反而可能使主内核变慢。B64/S16 的 FP32 partial output/max/sum 一写一读约 16.25 MiB/层，减少该流量可以测试，但仍需连同主 kernel 并行度评价。

**不能把两个不同 KV heads 直接拼进 16 行、却共用同一 K/V 点积。** 两组 Q 对应不同 K/V；这种“填满无效行”会改变 attention 语义。需要正确的 batched-dot/分组布局，且未必比独立组更快。

## 5. 可复现的下一轮实验顺序

1. 固定当前 GQA 提交作为直接对照。使用独立工作树，在现有 profiling 和调度队列释放 GPU 后运行，不在正式测试期间抢 GPU 或改源码。
2. 首轮固定 tile32、BM16，测 warps={2,4}、stages={1,2,3}、segments={8,16}；编译不支持的配置记录失败，不静默删除。有效配置逐项比较主 kernel、reduce、合计三种耗时。
3. 工作点覆盖 B={1,8,16,30,32,64}，L={4096,4700,5120,16384,17408}，并保留 ragged、空 query、短序列、prefill2048 和 mixed。先在 B64/4K 与 B30/16K 筛选，再扩展验证。
4. 再单独引入页表 load/地址布局优化，之后才比较独立 8-head decode 布局。每次只改变一类因素，防止无法解释组合收益。
5. 无损候选仍需独立 FP32 causal oracle、BF16/FP16、非连续 Q、NHD/HND、int64 大偏移、零长度与分段尾部、修改 Q/KV/lengths 后的 Graph replay 检查。允许暂缓完整任务精度，不等于免除这些检查。
6. 微基准采用预热、Graph event 计时、多次夹测当前实现；然后真实模型接入 42 层，按官方 4K/16K 各四轮、弃首轮。正式测量关闭 profiler，报告吞吐、TTFT、TPOT/ITL、波动和调度参数。

按该诊断窗口的 78.34% 主 kernel 占比做条件估算：若它单独加速 1.2×，其余 kernel 不变，kernel 总和约加速 1.15×；若加速 2×，约为 1.64×。这些仅用于判断优先级，不是生产模式或端到端收益承诺。

这轮候选均保持完整 KV 和 attention 定义。KV 压缩属于另一个实验维度，必须单独做官方任务精度与触发压缩的长文检查，不能用上述无损参数实验代替。
