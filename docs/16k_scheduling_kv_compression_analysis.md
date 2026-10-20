# MiniCPM5-2B：16K 排队、调度优化与 KV 压缩适用性

核查日期：2026-10-20；设备为单卡 BI-V150 32 GiB。本文使用已经完成的官方 16K 基线，讨论系统与算法层的优化空间。算子和 GQA 的具体实现见[基线、融合与 GQA 分析](baseline_fusion_gqa_analysis.md)。

**结论：16K 排队具有明确的容量原因，同时存在 prefill/decode 干扰的优化空间。调度可以降低额外延迟，无法凭空增加 KV 容量。H2O、SnapKV 等有希望同时减少容量和读取成本，但需要改变 paged cache、attention metadata 与调度准入，且属于近似计算；目前不能直接当作符合赛事约束的低风险替换。**

## 1. 已完成测评暴露了什么问题

官方配置为输入 16384、输出 1024、最大并发 64、总请求数 128，每组四轮舍弃首轮。后三轮平均：

| 指标 | 4K 基线 | 16K 基线 | 解读 |
| --- | ---: | ---: | --- |
| total tok/s | 2008.97 | 917.27 | 输入 token 也计入吞吐，不等于用户生成速度 |
| output tok/s | 401.79 | 53.95 | 长上下文下生成吞吐明显下降 |
| 单轮时长 | 652.52 s | 2429.18 s | 16K 每轮约 40.49 分钟 |
| Mean TTFT | 11.66 s | 597.81 s | 首 token 包含队列等待和提示计算 |
| P99 TTFT | 70.08 s | 1158.46 s | 此处是三个轮次各自 P99 的平均 |
| Mean TPOT | 147.38 ms | 486.77 ms | decode 也受到上下文和调度影响 |
| P99 ITL | 753.46 ms | 3300.22 ms | 长尾相邻 token 间隔约 3.30 s |

原始 16K 四轮全部 128/128 成功，每轮输入 2,097,152、输出 131,072 token。不存在以少生成 token 换取吞吐的情况。

对恢复后基线服务的周期日志进行统计：

| 日志样本集合 | 样本数 | Running 中位数 | Waiting 中位数 | KV usage 中位数 |
| --- | ---: | ---: | ---: | ---: |
| Running 或 Waiting 非零 | 971 | 30 | 34 | 95.3% |
| Waiting > 0 | 941 | 30 | 34 | 95.4% |
| Waiting > 0 且 KV usage ≥90% | 784 | 30 | 34 | 96.2% |

有排队的样本中 prefix hit rate 中位数为 0.0%。这些是日志采样点的中位数，不是逐请求统计，也不是精确的时间加权占比。不能把 `941/971` 当作每个请求的排队概率。

证据：[原始日志选段](../experiments/EXP-0004/artifacts/analysis_20261020/runtime_log_evidence.json)、[统计脚本与结果](../experiments/EXP-0004/artifacts/analysis_20261020/analysis_statistics.json)、[完整基线归档](../experiments/EXP-0004/artifacts/runs/20261019_202342/baseline_16k/)。

## 2. 先算容量：为什么 64 并发大致只有 30 个驻留

模型是 42 层，K/V 各 2 heads，每 head 128 维，BF16 每元素 2 bytes。

```text
每 token 的 KV = 42 × 2(K,V) × 2(KV heads) × 128 × 2
               = 43,008 bytes = 42 KiB
```

| 容量项目 | 原始 KV 数据量 |
| --- | ---: |
| 一条 16K prompt | 672 MiB |
| 一条 16K prompt + 1K output，按全长度预算 | 714 MiB |
| 64 条 16K prompt | 42.0 GiB |
| 64 条 16K + 1K | **44.625 GiB** |
| 本次实际 KV 池：521,936 token slots | **20.906 GiB** |

单卡一共 32 GiB，KV 池还要给约 4.69 GiB 模型权重、activation、运行时与 Graph 留空间。因此即使把可用显存比例提高到 100%，也不能装下 44.625 GiB 的 KV，更不用说模型本身。

```text
521,936 / 16,384 ≈ 31.86 条完整 prompt
521,936 / (16,384+1,024) ≈ 29.98 条完整输入输出预算
```

这是理想容量估算，不是“同时恰好能放 30 条完整 17408 token”的硬保证。block 对齐、输出尚未全部生成、最后采样 token 尚未写 cache、共享前缀、空块和运行阶段都会影响实际数量。日志出现 Running=29–31 与该量级一致。

作为对比，4K 的 `64 × (4096+1024)=327,680` token 低于池容量。4K 仍可在起步准入时排队，但没有同样的全长驻留压力。

PagedAttention 改善的是分配碎片与共享，并不会压缩每个 K/V 数值或删去历史；当前环境已使用 paged KV。不能把“开启分页”当作尚未实现的容量方案。[PagedAttention 原论文](https://arxiv.org/abs/2309.06180)

## 3. 排队、prefill 和 decode 干扰必须分开诊断

```text
TTFT = 客户端/接口与输入处理
     + engine 等待时间
     + 各 prefill chunk 的执行与 chunk 间等待
     + 首 token 采样及传输
```

本次没有逐请求阶段时间戳，不能把 597.81 秒全部称为纯队列时间，也不能反过来归因于某个慢 kernel。Running 计数包含部分 prefill 的请求，不等于正在同时完成单 token decode 的数量。

16K 比 4K 不仅多存 4 倍 prompt KV：完整 prefill attention 的计算随长度近似二次增长，而 decode attention 读取与计算随历史长度增长；插入长 prefill chunk 还会延长其他请求两次 decode 之间的间隔。P99 ITL 的长尾与这种干扰相容，但仅凭汇总值不能确定各原因占比。

此前被中止的旧服务 metrics 曾出现累计 preemption=2。它覆盖整个服务生命周期，不能认定恢复后 16K 的四轮各发生两次抢占，也不能证明目前主要损失来自反复重算。新的比较必须采集每轮前后的 counter 差值、请求重算 token 和抢占事件。

## 4. 调度器现在已经做了什么

以下为固定安装源码，详见[源码摘录与哈希](../experiments/EXP-0004/artifacts/analysis_20261020/source_excerpts.md)：

| 代码位置 | 已有机制 | 对优化的含义 |
| --- | --- | --- |
| `Scheduler.schedule:388`，token budget `:409` | 用本步 token 预算调度，不是全局先 prefill 后 decode | 已支持连续批处理与 chunked prefill |
| 同文件 `:432–587` | 先遍历 Running；分配新增 token/KV；空间不足时可抢占 | Running 中也可有尚未完成的 prefill，不能简写成严格 decode 优先 |
| 同文件 `:643–900` | 在预算剩余时从 Waiting 准入 | 并发 64 是客户端限制，不保证 64 条立即驻留 |
| `config/scheduler.py:140` | `scheduler_reserve_full_isl=True` 默认开启 | 已有“整条输入能否装下”的准入门槛 |
| `Scheduler.schedule:874–883` → `KVCacheManager.allocate_slots:372` | 将 full-sequence 检查传入 block 管理 | 只检查当时的完整序列，不等于给未来全部输出预留空间 |
| `KVCacheManager.allocate_slots:360–425` | watermark、reserved blocks、free block 检查 | 现成的保护接口，但不能把默认 0 watermark 写成启用保护 |
| `Scheduler._preempt_request:1107` | 释放 blocks、computed_tokens 归零、重新入等待队列 | 可能引入重算；可复用 prefix 能减轻多少需看实际命中 |
| `KVCacheManager.free:460` 及 scheduler 结束处理 | 释放请求引用、维护缓存生命周期 | 不应盲目清空仍有引用的缓存 |

最终生效的历史日志确认了 async scheduling、chunked prefill、prefix caching 和 `FULL_DECODE_ONLY`。整条输入准入与 watermark 的数值来自固定源码默认值，未取得已退出进程的完整 SchedulerConfig 对象快照；不要把默认值伪装成进程插桩结果。compile range endpoint=2048 也不能单独证明 token budget 恰为 2048。

### 4.1 值得做的系统优化

| 方案 | 预期改善 | 局限与实现注意 |
| --- | --- | --- |
| **为 decode 增长保留自适应余量** | 避免新 prefill 刚准入就挤压活跃 decode，减少抢占重算 | 基线已有整段输入检查；新增的是输出增长余量。保留过多会降低利用率和吞吐 |
| **按预计执行成本安排 prefill chunk** | 限制长上下文 chunk 对 ITL 的干扰 | 相同 token 数不等于相同执行成本；小 chunk 会增加调度和 launch 次数，也可能使 TTFT 更差 |
| **对混合 attention 分流** | 让长历史 decode 继续使用合适的分段内核 | 多一次 launch、metadata 构造和索引成本；需要与共享 GEMM 批处理共存 |
| Graph 友好的批次组织 | 提高纯 decode FULL Graph 的使用率，减少主机发射 | 等待凑批会直接增加 TTFT；不能为 Graph 人为延迟请求却只报吞吐 |
| 优化 CPU metadata / block table / output 处理 | 若 CPU 存在空洞，可降低 GPU 空闲 | 当前没有 CPU/GPU trace 证明它是主要热点；异步调度本来已启用 |
| 及时释放结束请求、减少内部碎片 | 增加可准入空间 | 已有释放机制，须先证明额外滞留或碎片，不能期待把 20.9 GiB 变为 44.6 GiB |

Sarathi-Serve 研究了 chunked prefill 与控制 decode 停顿的调度，值得借鉴其“每步工作量和延迟”的思路。但本环境已有 chunked prefill，下一步应针对该实现做成本建模和测量，不是简单复制一个开关。[Sarathi-Serve 原论文](https://arxiv.org/abs/2403.02310)

### 4.2 一个容易忽略的调度—内核耦合

`triton_unified_attention.py:969–978` 只要发现 `max_seqlen_q>1`，整个该 attention 调用就不走 3D split-K 路径。约 30 条纯 decode 可以沿历史长度切成 16 段；混入一条 prefill 后，那些 decode 也进入 2D 分支。`FULL_DECODE_ONLY` 同时令混合批次无法使用对应的 full decode Graph。

这不证明混合批次一定更慢：更大的共享 GEMM 可能更高效。它提供一个比“把所有请求都优先 decode”更具体的实验假设：保留共同的模型计算，在 attention 处给 decode 和 prefill 分别建立子 metadata、派发不同 kernel，然后还原输出顺序。先比较新增开销与长历史 decode 的节省，再考虑调度器层面的分离。

### 4.3 不应当期待的捷径

降低客户端并发到 30 会缩短外部队列，却改变官方负载；把 engine `max_num_seqs` 调低，也只是改变准入位置，不能增加物理容量。提高 `max_num_batched_tokens` 可能改善 prefill 吞吐但恶化 ITL，降低则可能改善 ITL但延长提示完成时间。修改显存比例、最大长度、量化/投机采样超参，不能作为这次固定官方命令下的优化成绩。

随机提示的前缀命中很低，因此 prefix caching 很难成为主解。32 GiB 单卡上拆成 prefill/decode 两个服务也不会产生更多显存，反而可能重复模型与缓存。CPU offload 能扩大容量，但长历史的搬运量很大；必须证明传输和重算代价被覆盖，不能默认 offload 就会更快。

## 5. H2O 等论文分别解决哪一部分

下表仅概括原论文机制，不把其他 GPU、其他模型或框架的倍数迁移成本项目的预测收益。

| 方法 | 压缩/优化机制 | 对当前模型的适用判断 |
| --- | --- | --- |
| [H2O，2023](https://arxiv.org/abs/2306.14048) | 利用累计 attention 重要性，保留 heavy hitters 与最近 token，生成时持续更新 | 能减少历史保留量，但需要取得重要性分数、在线选择与 cache 淘汰；不是现成的 vLLM 开关 |
| [SnapKV，2024](https://arxiv.org/abs/2404.14469) | 从 prompt 尾部观察窗口估计关注位置，聚合/选择重要 prompt KV | 对长输入、相对短输出的本负载值得研究；prefill 完成后的压缩与动态 decode 淘汰成本不同 |
| [PyramidKV，2024](https://arxiv.org/abs/2406.02069) | 不同层分配不同 KV 预算，利用层间注意力分布差异 | 需要按层独立的容量/长度管理；不能仅改一个全局 seq_len |
| [KV-Compress，2024](https://arxiv.org/abs/2410.00161) | 面向分页 KV 的压缩，处理 query-group、按 head/layer 不同预算和块回收 | 最值得参考其系统接入设计；同时说明真正省容量需改 block 管理，而非只 mask token |
| [StreamingLLM，2023](https://arxiv.org/abs/2309.17453) | 保留少量 attention sinks 与近期窗口以支持流式生成 | 简洁，但丢失远处内容，不能保证任意 16K 信息检索/推理；不宜作为本模型长文能力的默认替代 |
| [KIVI，2024](https://arxiv.org/abs/2402.02750) | K 按 channel、V 按 token 的低比特量化，并处理近期残留部分 | 研究上可减小每个 KV 的字节数；涉及量化和专门解码 kernel，当前赛事约束下不能直接启用作为成绩 |

这些方法的“无需训练”不等于“数值等价”，论文中的质量结论也不等于 MiniCPM5-2B 的 MATH500/长文精度已经通过。当前用户暂缓精度测试，本轮没有运行这些方法或形成性能结论。

### 5.1 H2O 在这里为什么不是换一行 attention

**以下是结合本模型和已安装源码的接入分析。**

1. **当前内核不输出完整 attention 矩阵。** 它用分块在线 softmax，`softmax_step` 的中间 P 依赖逐步更新的最大值与归一化因子。不能把每个 tile 未最终归一化的 P 直接累计成 H2O 分数。需要利用最终 LSE 正确归一化、额外计算/统计，或设计有明确误差的新近似；3D 分段还需全局归一化。
2. **GQA 共享物理 KV。** 8 个 Q head 共同访问一个 KV head。若按 Q head 各自保留集合并复制 KV，会把已有共享优势吃掉。可以研究组内分数求和/均值/最大值或集合并集，得到一个 KV-group 保留集合；这些聚合策略需要独立质量验证，不能假定完全等同原算法。
3. **删 token 必须真的释放物理块。** 只在 attention mask 中屏蔽 token，可能少算一些，却仍占着原 cache；无法让等待队列多准入请求。需要搬移剩余 KV、形成可回收块，并安全更新 free list、引用和 block table。
4. **不同层/不同 head 的保留位置可能不同。** 当前 full-attention cache 按请求共享逻辑序列布局。独立淘汰需要扩展每层/KV head 的 block table 和长度；或者先限制为所有层共享保留集合，降低工程复杂度，但这种策略质量并不等价。
5. **物理长度与逻辑位置要分离。** 处理过 16384 个 prompt token、cache 只剩 7000 个，不意味着下一个 token 的 RoPE position 变成 7000。对保留原位置的 H2O/SnapKV 路径，应保持真实绝对位置，移动已经旋转的 K 不应重复旋转。causal mask、slot mapping、query_start_loc、seq_lens 的语义也必须匹配新布局。StreamingLLM 的位置处理是其独立方法设计，不能混用。
6. **避免在已捕获 Graph 外偷偷改变地址。** 可预分配数据/索引/统计缓冲区，但 Graph replay 时的逻辑长度、淘汰次数和 side effect 必须正确；动态 Python top-k、逐层 CPU 同步可能抵消节省。
7. **调度器必须看见压缩后的占用。** 继续按原 full-sequence 长度准入，会错误拒绝已经有空间的请求；只修改 block manager 又不修改运行输入，会造成 attention 读错位置。
8. **前缀缓存复用语义需要重审。** 不能把已经删掉中间位置、依赖后续 query 选择的缓存，当成完整且可供其他请求复用的同一个前缀。

上述改动横跨 kernel、runner 和 scheduler。仅把论文 Hugging Face 示例移植为一个算子，不能完成高吞吐 continuous batching 的接入。

### 5.2 压缩比例要达到多少，才可能缓解 64 并发

在当前原始 KV 池容量下，均分给 64 条请求：

```text
理想预算 = 521,936 / 64 = 8,155.25 个保留 token / 请求
若完整保留 1,024 个生成 token：
prompt 保留预算约为 7,131.25，约占 16K prompt 的 43.53%
```

这只是忽略统计状态、索引、对齐和临时区的上限，还没有包含同时进行 prefill 的峰值缓存。仅将 prompt KV 压到 50% 并保留全部输出，`64 × (8192+1024)=589,824`，仍超过 521,936。

若把包括输出在内的总保留长度固定到 8192，64 条共需 524,288，也略超基线容量。不能把“8K budget”直接写成必能达到 64 并发。实际要留下余量，并以 allocator 的真实可用块数验证。

先完整 prefill、然后压缩的方案仍需承担单条 16K 的 prefill 计算和临时峰值；释放已进入 decode 请求的空间，有助于后续 prompt 准入，却不保证最初 64 个提示同时展开。若要在 chunked prefill 期间就淘汰，则会改变后续 prompt token 可见的历史，应作为另一种算法单独验证。

## 6. 哪条路线更适合现在做

### 6.1 赛事主线：优先保持完整上下文的优化

1. 针对 D=128、Q:KV=8:1 的 GQA 实现做天数 tile/分段调优，尤其是长上下文 decode。
2. 分析混合批次导致的 2D/3D 路径切换与 Graph 变化，尝试改善派发而保留原数学结果。
3. 在实际出现可归因的抢占重算后，再做输出增长余量、准入与 chunk 成本优化。
4. RoPE+KV update 等局部融合继续作为减少 launch/访存的路线，但不把它当成容量压缩。

固定 serve 参数与 benchmark 是当前比赛要求。具体源码级 scheduler 改动是否属于可提交范围，需依据最终官方口径核对；本文不宣称只改调度就一定符合算子赛题。规则整理见[比赛要求](competition.md)，原始入口见[赛事页面](https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn)。

### 6.2 研究支线：先静态 prompt 压缩，再考虑持续淘汰

若研究目标允许近似 KV，建议先做 **GQA group 共享保留集合、固定预算、prefill 末尾压缩** 的小范围原型。SnapKV 的选择时机与 KV-Compress 的分页回收设计可分别作为算法和系统参考。它比每一步 decode 都更新/淘汰的 H2O 更容易把选择与搬移开销摊薄到本次 1024 个输出 token 上，但这只是工程判断，收益和质量仍待测。

进一步再探索 H2O 持续重要性更新、分层预算和不同 KV head 预算。保留集合越灵活，对 metadata、分页、批处理和 Graph 的改动越大，不能把算法压缩率直接当作端到端收益。

KIVI 等量化路径暂列研究对照；不能绕过现有禁止通过量化超参取得成绩的约束，更不能通过把开关藏进代码来规避规则。对于 token 淘汰，现有规则记录不能充分证明允许或禁止，须在正式采用前确认，并补完整任务质量验证。

## 7. 能证伪这些判断的实验设计

本节是待执行方案，不是新实验结果；实施前按仓库约定分配新 EXP 编号。

| 实验问题 | 控制条件 | 必须记录 |
| --- | --- | --- |
| 排队主要受容量还是抢占重算影响？ | 原官方两组配置；每轮独立统计 | waiting/running、free blocks、每轮 preemption 增量、重复计算 token、实际 prefix 命中 |
| 混合 prefill 是否拖慢长 decode？ | 同一模型/调度预算，收集诊断 trace | 每步 query 长度分布、2D/3D kernel、Graph 类型、decode ITL、step GPU 时间 |
| 保留增长余量是否值得？ | 请求内容与总数不变，仅改准入策略 | 吞吐、平均及尾部 TTFT/ITL、抢占减少量、空闲容量与公平性 |
| 压缩有没有真正增加可用容量？ | 同一 BF16 原模型；禁止只 mask 冒充回收 | 实际归还 blocks、压缩 metadata/score 的内存、峰值显存、驻留并发、排队时间 |
| 压缩成本是否被覆盖？ | 分开计时评分/选择/搬移/attention | prefill 增量成本、decode 节省、端到端总时间；包括编译/Graph 行为变化 |
| 近似算法能否用于模型任务？ | 恢复精度阶段后再执行 | 官方 MATH500 Level 3、长文检索/跨段推理、不同长度和预算；不能只看随机 prompt 能否生成 |

调度优化至少要同时看 TTFT 与 ITL，避免把延迟从一个阶段转移到另一个阶段。压缩优化至少要同时看实际 free blocks、attention 速度和质量，避免出现“缓存张量看上去变短，调度器仍然不能准入”的假收益。

当前最有证据支持的优先事项是完整 GQA 路径及调度—kernel 交互；KV 压缩则应作为需要明确质量与系统成本验证的独立研究方向。
