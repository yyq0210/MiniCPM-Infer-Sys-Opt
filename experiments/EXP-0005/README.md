# EXP-0005：GQA 解码、混合批次 attention 与整模接入

本实验承接 [流水线与融合分析](../../docs/baseline_fusion_gqa_analysis.md) 和
[16K 调度与 KV 压缩分析](../../docs/16k_scheduling_kv_compression_analysis.md)。
2026-10-20 按用户要求停止 EXP-0004 的 RoPE 单项长测，先选定本轮方案、完成代码接入和验证，
再统一启动官方性能测评。所有失败尝试、诊断和正式结果分别保存。

## 本轮选择与边界

| 方向 | 决定 | 理由和验证要求 |
| --- | --- | --- |
| GQA decode 分段计算 | 实现并验证 | Q16/KV2/D128 已知，按 KV head 复用 8 个 Q head 的 KV 加载；减少通用索引开销，诊断选择分段和 tile |
| 混合 prefill/decode attention 分流 | 实现并验证 | 原实现遇到任意 prefill 会关闭整个批次的 3D 分段；新内核在 GPU 读取 query length 后分流，decode 保留并行 |
| RoPE 独立优化 | 保留现有候选作为组合方案的一部分 | 已有基本数值/Graph 验证，但独立整模收益未完成；组合收益不得归因为 RoPE 单项 |
| RoPE + KV cache update | 本轮不纳入 | 当前 functional RoPE 与原融合 pass 模式不匹配；还需证明唯一缓存写入、依赖顺序和 alias 语义，不能只打开开关 |
| RMSNorm + QKV GEMM、MLP 全链融合 | 本轮不纳入 | QKV 和 gate/up 已合并；大 GEMM 与 row reduction 的并行和数据类型边界不同，缺少可保持厂商 GEMM 效率的实现证据 |
| residual add + RMSNorm | 保留现有融合 | 基线已存在；把残差写成 BF16 再归一化会改变原 FP32 累加语义 |
| 调度器改参数 | 不改官方配置 | async、chunked prefill、full ISL reservation 已启用；先从日志与指标判断实际 preemption，不能把物理容量不足当队列算法问题 |
| H2O / SnapKV / KV-Compress | 保留研究记录，不接入本轮服务 | 属于近似 KV 裁剪，需要质量验证和真正释放物理页；当前暂缓完整精度测评，无法据此确定裁剪预算 |

因此，本轮可验证的工程候选为“原有 functional RoPE + 有界 GQA 专用内核”。
“候选”不表示已经有整模收益，也不表示满足比赛精度要求。算子数值、真实引擎接入和完整官方性能
都是独立门槛；失败时保存现场，不启动后续长测。

## 基线和分支

- 官方插件基线：`13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`。
- 官方 FlagGems 基线：`a7620cc191a0b42e040194622c5758b22a7a25dc`。
- 本轮开发起点：插件 `0ad2b82abdd8563e6a27d20b67341a109f2c2e7c`；
  FlagGems `646668a82b64c9ddb2c79e50ecf25a5c229b8560`。
- 两个独立 worktree 均使用 `leanscale/iluvatar-gqa-mixed`，不修改官方基线、共享 site-packages 或旧实验源码。
- 通过检查后记录最终提交及文件 SHA256；测评启动后不继续修改这两个 worktree。

## 真实接入路径

原有 `CachedOp("attention_backend")` 的策略、严格模式、fallback 和 IO 路径保持。
当它选中 FlagOS，且平台为 Iluvatar 时，FlagOS 实现返回继承 `TritonAttentionBackend`
的 `IluvatarGQABackend`。其他平台与原有显式 backend 选项保留原路径。

新类继承原 metadata builder、KV layout、cache update、Graph 支持和特性声明，
只替换满足条件的 `forward` 中的 attention 计算：

```text
LlamaAttention → Attention → unified_attention_with_output
  → IluvatarGQAImpl.forward
    → 检查形状 / dtype / 特性 / scratch 容量
      → FlagGems _iluvatar/fused/paged_gqa.py
        → prefill 因果 paged attention（批次含多 token query 时）
        → 按 sequence × KV head × segment 的 decode attention
        → 稳定 softmax 分段归并
      → 不满足条件：原 TritonAttentionImpl.forward
```

这不是通过切换环境变量获得收益；新增算子源代码属于候选变更。最终必须由运行日志确认
实际模型层进入新实现，并检查整个正式测评期间未发生异常或悄然退回其他实现。

## 算子设计

1. 每个 KV head 服务 8 个 Q head，不生成 repeat 后的 KV 副本。
2. 纯 decode 使用 sequence 索引直接定位 query，避免通用 q-block 二分查找。
3. 混合批次的 decode 与 prefill 在 GPU 内依据 `cu_seqlens_q` 分流，不做 `.item()`、CPU 同步或 query gather。
4. decode 中间结果按 sequence 编号写入已有 `[capacity,16,16,128]` FP32 buffer。
   混合批次中不能按全局 query token 编号写 scratch：一个长 prefill 就会越过原有 64-sequence 容量。
5. partial max、exp sum 和 weighted value 按稳定 softmax 公式归并。空 segment 写 `-inf/0/0`，
   padding query 不写输出，避免读未初始化中间值。
6. K/V 读取使用实际 block table 和 strides，覆盖 NHD/HND 与 QKV split 的非连续 query。
7. 新实现不分配服务期 scratch、不丢弃 token、不改变 block manager 或 KV token 容量。
8. 非 Q16/KV2/D128、非 FP16/BF16、非因果、滑窗、sink、ALiBi、prefix-LM、量化 KV、
   batch invariant、输出量化或 scratch 不匹配时走原实现。

## 验证顺序与首轮发现

`validate_gqa.py` 将算子诊断与官方端到端结果区分。基础检查覆盖 FP16/BF16、16/32 大小的页、
NHD/HND、非连续 QKV、散布的物理页、空 query、ragged 长度、短 prefill、长 decode 和混合请求。
Graph 测试在固定地址修改 Q 后重放，和 eager 输出逐元素对照。

首轮 `20261020_130842` 在 6 项纯 decode 检查后因混合 prefill 与原实现不一致而停止，
没有执行性能矩阵。随后保留原失败脚本，用独立 FP32 softmax(QK)V 作对照：
在 seed=777 的短 prefill 样例中，现有基线路径的最大绝对偏差可达约 0.497，
新内核在相同采样位置约为 0.00372。这只说明这些诊断输入上的差异；
尚未定位现有路径的具体根因，不能扩展为模型精度结论。

第二版检查以独立 FP32 因果 attention 作为判据，维持 `atol=0.003, rtol=0.03`，
保留基线偏差字段；短序列检查全部 query，长序列抽查首、中、末 query。
纯 decode 仍另外对照原实现。Graph 另与相同输入的 eager 结果逐元素完全对照。
不得为了“通过”而扩大容差、删除失败或把原实现当作唯一正确答案。

## 性能及调度观测

诊断矩阵覆盖 B=1/8/30/64、KV=4K/16K，以及 30 decode + 2018-token prefill 两种请求顺序。
它用于比较 tile/segment 配置，不代表官方数据。计时前完成编译，以 CUDA Graph + event 计时，
记录所有候选和前后两次基线，检查顺序漂移。

只有选定配置、引擎接口检查、实际模型编译/Graph 和 smoke 均通过后，才启动原官方脚本：
`[[4096,1024,64,256],[16384,1024,64,128]]`，每组四轮、舍弃第一轮。
保持官方启动参数，不缩短请求、输出或轮数。完整基线来源和两个服务会话的差异保留。

正式测评保存 running/waiting/KV 使用率日志及测前测后指标，
用于区分容量排队、prefill/decode 混合计算、preemption 和吞吐变化。
新 GQA 内核不压缩 KV，故不能承诺让 64 个 16K 请求同时进入显存。

完整 MATH500 按用户要求暂缓。即使性能提升，也只报告性能结果与已完成的算子验证，
不声明赛事精度达标，不自动创建上游 PR。

## 本轮最终选定与接入门槛

固定配置为 segments=16、decode_tile=32、prefill_tile=32、prefill_rows=64。两轮诊断全部完成，固定配置的额外 20 项数值/Graph 检查、实际 backend forward、回退和常量 V 不变量检查通过。性能和淘汰方案见 [DIAGNOSTICS.md](DIAGNOSTICS.md)。随后冻结两个代码提交，启动完整模型编译/Graph 和 smoke；只有观察到 42 层实际进入 GQA 实现后，才执行官方 4K/16K benchmark。最终状态见 [STATUS.md](STATUS.md)，整模结果见 [RESULTS.md](RESULTS.md)。
