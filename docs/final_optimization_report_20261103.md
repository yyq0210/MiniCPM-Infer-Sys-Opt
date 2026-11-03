# LeanScale：MiniCPM5-2B 在天数 BI-V150 上的推理优化技术报告

最终测评完成：2026-11-02。最终独立审核与报告：2026-11-03。

## 1. 最终结论与交付边界

最终保留 **RoPE + KV cache 写入融合、面向 Q16/KV2 的分页 GQA、decode 八行布局与页号广播、prefill 调优与页号广播、top-p 行并行**。保持官方调度、模型权重、KV 精度和测评配置。

| 官方负载 | 原始完整基线 Total tok/s | 原选定 EXP20 | 最终配置 Total tok/s | 相对原始基线的累计变化 |
|---|---:|---:|---:|---:|
| 4K | 2008.97 | 3579.12 | **3687.315** | **+83.543%（1.835×）** |
| 16K | 917.27 | 2541.43 | **2557.84** | **+178.854%（2.789×）** |

4K 最终值是 EXP29 两个开启开关会话的官方 summary 等权均值；16K 是 EXP30 开启开关会话的官方 summary。每个会话均完整四轮、只弃首轮。原始基线和 EXP20 来自早期不同日期，累计变化用于展示项目阶段结果，不能代替同期因果对照，也不能把各阶段增量简单相加。

最后一项 top-p 改动的同期证据是：

- **4K 预登记 ABBA：3539.200 → 3687.315 tok/s，+4.185%；TPOT -3.846%。** 两次固定配对均有收益，4096 请求零失败，全部预登记工程门槛通过。
- **16K 顺序 A/B：2542.44 → 2557.84 tok/s，+0.606%；TPOT -0.680%。** 1024 请求零失败，KV 容量没有下降、抢占没有增加，非劣化门槛通过。推送故障使两会话存在时间间隔；这个小幅差额不构成统计显著性结论。

这是本项目的**最终性能配置**。用户要求本轮结束后收尾，因此不再启动新的优化实验。完整模型任务精度（包括 MATH500）按用户要求未测；已有算子/Graph/采样检查不等于赛事完整精度达标。KV 压缩没有进入最终配置。

## 2. 环境、模型与测量口径

| 项目 | 实际配置 |
|---|---|
| 设备 | 单卡 Iluvatar BI-V150，32 GiB，16 SM |
| 软件 | Python 3.12；PyTorch 2.10.0+corex.4.5.0；Triton 3.6.0；vLLM 0.24.0+empty |
| 模型架构 | LlamaForCausalLM，42 层，hidden=2048，intermediate=6144 |
| Attention | 16 个 Q heads、2 个 KV heads、head_dim=128，GQA 分组比 8 |
| 词表/精度 | vocab=130560，BF16 模型与 KV |
| Serve | FULL_DECODE_ONLY；gpu-memory-utilization=0.85；max-model-len=131072 |
| 4K | [4096,1024,64,256]：输入、输出、最大并发、请求数 |
| 16K | [16384,1024,64,128] |
| 数据与采样 | 沿用官方 random 数据配置、默认采样与 ignore-eos |
| 汇总 | 每会话四轮，仅弃第一轮；保留全部 raw CSV、日志与 summary |

官方 benchmark SHA256：
0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565。

Total tok/s 同时包含输入和输出 token；Output tok/s 仅统计生成。4K 与 16K 的输入/输出比例不同，不能把两个场景的 Total tok/s 当成相同生成工作量直接比较。报告中的 CV 使用保留三轮的样本标准差除以均值；它是波动描述，不是置信区间。

正式测量串行使用 GPU。各 A/B 会话分别重建服务和 VLLM/Triton/Inductor 缓存；profiling、shadow 数值检查与正式性能测量分离。编译缓存/日志/实际调用用于证明代码接线，而非仅凭环境变量判断。

## 3. 真实推理流水线与热点

请求经过服务入口、调度器和 KV block 分配，再由 model runner 构造 position、slot mapping、block table 和 attention metadata。模型 embedding 后执行 42 个 decoder block，最后 final norm、LM head、采样和输出。

每层的核心计算顺序为：

```text
residual add / RMSNorm
  → 已合并的 QKV linear
  → split/view 得到 Q、K、V
  → Q/K RoPE + K/V 写入分页 KV cache        ← 本项目融合
  → GQA：prefill / decode / mixed 路径      ← 本项目主要 attention 优化
  → O projection
  → residual add / RMSNorm
  → 已合并的 gate/up linear
  → SwiGLU
  → down projection
  → 下一层；末层进入 final norm → LM head → sampling
```

QKV projection、gate/up projection 的合并以及部分 residual/norm、SwiGLU 的编译融合是已有路径，不能算成本项目新增成果。实际 SwiGLU 是 native F.silu + mul 经 Inductor 处理。

EXP18 的诊断窗口中，FULL 配置的 mixed 窗口实际采用 NONE 执行：prefill GQA 占可见 kernel 工作量约 45.20%，gate/up 23.61%，down 12.23%；NONE/B64 decode 窗口中主 GQA 约 65.78%，采样约 12.70%；尾部 B8–16 的四类层内投影约 44.54%。这些占比对应不同窗口，不能相加或当作整轮请求时延占比。当前 profiler 未捕获 FULL decode Graph 内部全部 kernel，因此没有虚构整程热点百分比。

详细函数映射与窗口口径见[真实推理流水线](minicpm_real_inference_pipeline.md)、[基线/融合/GQA 分析](baseline_fusion_gqa_analysis.md)及[4K 分阶段热点报告](operator_profile_4k_20261023.md)。

## 4. 最终保留的优化：前后思路和接入

### 4.1 RoPE 与 KV 写入融合

原路径先完成 Q/K 旋转，再由独立 KV update 读取 K/V，根据 slot mapping 写入缓存。候选把旋转和分页缓存写入放入同一个 kernel，减少一次独立 KV update 启动及相关重复读写。QKV GEMM 仍在融合边界之外，不能称作“RMSNorm + QKV + attention 全部融合”。

框架侧由 [fuse_rope_kv](https://github.com/yyq0210/vllm-plugin-FL/blob/c50b5f0526e8112c72f1de918487300cf7376378/vllm_fl/compilation/rope_kv_fusion.py#L34) 识别编译图，再经 [rope_kv_dispatch](https://github.com/yyq0210/vllm-plugin-FL/blob/c50b5f0526e8112c72f1de918487300cf7376378/vllm_fl/ops/rope_kv_dispatch.py#L17) 调用 FlagGems 的 [rope_kv_cache / _rope_kv](https://github.com/yyq0210/FlagGems/blob/de958ddfc4deeb43d6c94bb8652a3d4d3bc905fe/src/flag_gems/runtime/backend/_iluvatar/fused/rope_kv_cache.py#L9)。

验证关注 position/slot 映射、负 slot、分页地址、别名、Graph padding、重复执行和缓存副作用。图重写必须保留 KV 写入及 attention 输出缓冲区的副作用，不能让通用死代码消除误删写操作。最终每次正式服务均核验 42 层融合节点，原独立 unified_kv_cache_update 不应残留在对应融合图中。

### 4.2 分页 GQA：decode 分段、八行布局与页号广播

模型每个 KV head 对应 8 个 Q heads。专用路径在一个 KV 分组内利用 K/V 共享，decode 沿上下文分成 16 段计算，再用独立 reduce 合并 partial softmax 输出。它保持完整注意力，不裁剪历史 KV。

最终 decode 配置为 rows=8、num_warps=4、num_stages=1、segments=16、tile=32、layout=1。八行对应实际的 8 个 Q heads，减少无效行带来的资源占用。页表物理 block 大小为 16，32-token tile 只需读取 2 个页号，再以 int64 广播到各 token lane；不再逐 token 重复页表寻址。地址计算保持 int64，避免长上下文 stride 乘法溢出。

最终编译证据确认 decode 的 8×128 Q 读取、2 个页号加载和 2×16 int64 广播，共享内存 18 KiB。资源下降有利于容纳更多 resident blocks，但本项目没有硬件 occupancy 计数器测量，不能直接把寄存器/共享内存下降等同于“实测 occupancy 已提高”。也不能把 CUDA 的 32-thread warp 假设直接套到该后端。

实际入口为 [IluvatarGQAImpl._supports_gqa / forward](https://github.com/yyq0210/vllm-plugin-FL/blob/c50b5f0526e8112c72f1de918487300cf7376378/vllm_fl/dispatch/backends/flaggems/impl/iluvatar_gqa.py#L21)，设备实现为 [paged_gqa.py：_attention、_reduce、paged_gqa](https://github.com/yyq0210/FlagGems/blob/de958ddfc4deeb43d6c94bb8652a3d4d3bc905fe/src/flag_gems/runtime/backend/_iluvatar/fused/paged_gqa.py#L14)。

支持条件明确限定 Q16/KV2/D128、普通 causal decoder、支持的 BF16/FP16 KV 和布局；ALiBi、sinks、sliding window、cascade、额外量化/输出 scale 等不支持情况回退原 TritonAttentionImpl。没有改变模型的 attention 语义来换取速度。

### 4.3 Prefill：tile/stage 调优和页号广播

prefill 以多 query rows 与 K/V tile 做矩阵运算，与 decode 的工作形状不同。最终固定 rows=64、warps=4、stages=1、tile=32、layout=1；保留单独的 decode 配置，避免用一组参数覆盖两个阶段。

在已验证的 tile/stage 配置上，再把 prefill 页号加载改为两页号加 int64 广播。实际 TTIR 确认 64×128 Q 读取、tile32 页表路径，共享内存 32 KiB。不是通过临时改变官方并发、数据长度或 KV 精度得到收益。

EXP15/16 完成 prefill 参数扩大检查和真实引擎接入，EXP19/20 完成页号广播检查及官方两组测评。EXP20 的组合成绩为 3579.12 / 2541.43 tok/s；这是当时组合配置的成绩，不是单独页号广播的独立贡献。

### 4.4 Top-p：把批内串行改为行并行

原 launcher 使用 min(16,B) 个 program，B64 时一个 program 串行处理四行。候选使用 min(64,B)，B64 时每个 program 处理一行。仍调用原 top-k/top-p JIT，保留 tile、表、warps/stages、筛选逻辑和 RNG 语义；没有新造一个近似采样算法，也没有新增算子融合。

实现见 [topp_row_parallel.py](https://github.com/yyq0210/vllm-plugin-FL/blob/c50b5f0526e8112c72f1de918487300cf7376378/vllm_fl/patches/topp_row_parallel.py#L74)。通过真实 general-plugin 注册同时替换 base/sampler 两个别名。仅在 BI-V150 单 GPU、FP32 logits、V=130560、B17..64、top-p-only 和受支持 stride 上生效；其他条件回退。源码默认关闭，最终交付通过明确的环境开关启用。

scratch 从 16 行的 8,355,840 bytes 扩为共享 64 行的 33,423,360 bytes，名义增量 23.90625 MiB。已有 39 项设备入口/Graph/fallback 检查，以及真实模型 B17..64 各两次、共 96 次精确 mask/RNG 检查通过。正式性能测评 shadow=0。B64 单算子截断阶段耗时下降 66.73%，最终 4K 整模吞吐提升 4.185%；两种口径不能替代。

16K 受容量约束，实际 batch 分布和 prefill 比重不同，因此这项优化可覆盖的工作量不同。4K 收益较大、16K 差额较小与这个机制一致，但本轮没有做足以精确拆分贡献的分项归因。

## 5. 最后两轮正式测评

### 5.1 4K：固定代码 ABBA

| 会话 | top-p | 第1轮（弃用） | 第2轮 | 第3轮 | 第4轮 | 官方 summary | 保留轮CV |
|---|---|---:|---:|---:|---:|---:|---:|
| A1 | 关闭 | 2069.34 | 3343.09 | 3633.02 | 3686.22 | 3554.11 | 5.196% |
| B1 | 开启 | 2122.33 | 3463.24 | 3770.33 | 3832.32 | 3688.63 | 5.358% |
| B2 | 开启 | 2119.92 | 3454.43 | 3770.56 | 3833.01 | 3686.00 | 5.506% |
| A2 | 关闭 | 2119.88 | 3527.51 | 3367.60 | 3677.76 | 3524.29 | 4.401% |

A1→B1→B2→A2 为事先固定顺序。每组四轮、每轮256请求；慢轮 A2 第三轮照常保留。A 与 B 各两个会话等权平均，B/A +4.185%，固定 B1/A1 +3.785%、B2/A2 +4.588%；A 首尾漂移 -0.839%，B -0.071%。

| 指标 | A 均值 | B 均值 | 变化 |
|---|---:|---:|---:|
| Total tok/s | 3539.200 | 3687.315 | +4.185% |
| Output tok/s | 707.840 | 737.465 | +4.185% |
| Mean TTFT（ms） | 5121.290 | 5099.410 | -0.427% |
| Mean TPOT / ITL（ms） | 84.115 | 80.880 | -3.846% |

全部4096请求成功。每会话原 GQA/MLP/RoPEKV/FULL 路由与 top-p 接线均审核；工程门槛通过。单个 ABBA 区块仍不能称为统计显著，服务内轮次存在趋势。

详见 [4K ABBA 完整报告](topp_row_parallel_abba_20261030.md)与[独立审核](../experiments/EXP-0029/review_20261030/audit.json)。

### 5.2 16K：最终 A/B 与容量检查

A：11月2日16:50–17:59；B：19:55–21:03。两组之间因推送故障暂停，A 数据保留，仅续跑从未启动的 B。没有重测或挑选 A，也没有临时改变性能门槛。

| 会话 | 第1轮（弃用） | 第2轮 | 第3轮 | 第4轮 | 官方 summary | 保留轮CV |
|---|---:|---:|---:|---:|---:|---:|
| A：关闭 | 2191.34 | 2543.61 | 2542.69 | 2541.03 | 2542.44 | 0.05143% |
| B：开启 | 2201.76 | 2558.03 | 2557.97 | 2557.51 | 2557.84 | 0.01112% |

| 指标 | A | B | 变化/解释 |
|---|---:|---:|---|
| Total tok/s | 2542.44 | 2557.84 | +0.606% |
| Output tok/s | 149.55 | 150.46 | +0.608% |
| Mean TTFT（ms） | 203174.54 | 202097.05 | -0.530% |
| Mean TPOT / ITL（ms） | 178.05 | 176.84 | -0.680% |
| KV容量（tokens） | 522256 | 522352 | B多96，不能据此归因分配变化 |
| 四轮总抢占 | 16 | 16 | 没有增加，也没有消除 |
| 四轮平均排队（s） | 203.7483 | 202.6960 | 包含被弃用首轮 |
| 日志采样KV峰值 | 100% | 100% | 容量压力仍然存在 |

每组512请求成功、失败0；每轮输入2,097,152、输出131,072 tokens。吞吐下降不超过1%、TTFT增加不超过5%、TPOT增加不超过2%、KV至少99%、抢占不增加等预登记非劣化门槛全部通过。

独立审核核验每组20份原始/归档文件及本地 Git commit blob、官方 CSV/summary、源代码/环境、36个 decode 与31个 prefill 编译变体、42个 MLP 模块/84个原权重节点、42次 RoPEKV 融合、35个 FULL 捕获尺寸。B 有31条真实 top-p grid记录，最大记录调用计数16384，正式 shadow检查0；不是仅设置了开关而未走候选路径。

本轮结束后所有服务子进程已清理。详见[最终审核](../experiments/EXP-0030/review_20261102/audit_final.json)、[原始归档](../experiments/EXP-0030/artifacts/)和[最终配置与决策](../experiments/EXP-0030/final_configuration.json)。

## 6. 未采用的方案及原因

| 方向 | 项目证据 | 最终处理 |
|---|---|---|
| RMSNorm + QKV GEMM、大范围 MLP GEMM 融合 | 原型中归约、tile/寄存器及矩阵效率代价超过省下的启动/访存；已有明显劣化 | 不接入；不能把流水线上相邻函数都塞进一个 kernel |
| MLP gate/up、down 权重布局 | 单算子/小M曾改善，但 EXP25 同日 ABBA A3564.105、B3546.580 tok/s，-0.492%，两配对均无收益 | 不采用，保持原布局 |
| 调度/KV增长预留 | 部分方案劣化；用户明确要求停止并回退 | 使用原调度，不将排队变化冒充算子收益 |
| padding、K加载转置等候选 | 已记录无收益或退化的实验 | 不纳入最终配置 |
| H2O等KV压缩 | 仅有适用性分析，没有完成物理回收、长文质量及官方精度验证 | 未部署、无性能成果声明 |

“算子融合越多越快”不成立。RMSNorm 需要完整行归约，GEMM 以输出 tile 并行；强行融合可能重复归约或破坏矩阵单元效率。down projection 后跨层做 residual/norm 还涉及跨 tile 同步及 BF16 舍入位置，不能只看省掉的 kernel 数量。

实际每个 token 的42层 BF16 KV需要：

```text
42层 × 2(K/V) × 2 KV heads × 128维 × 2 bytes = 43008 bytes = 42 KiB
64 × (16384 + 1024) tokens × 42 KiB = 44.625 GiB
```

它超过当前约20.9 GiB的 KV 池。加速 cache 写入或调整等待队列不会增加物理容量，所以最终16K仍有100%缓存利用和16次抢占。KV压缩若继续研究，必须同时证明物理块确实回收、搬移/metadata成本计入、压缩确实触发及质量通过；仅mask旧KV不能算内存释放。本项目按用户要求已收尾，不再追加此实验。

参见[调度/KV压缩分析](16k_scheduling_kv_compression_analysis.md)、[MLP ABBA](mlp_weight_layout_abba_20261029.md)和[实验总账](optimization_journal.md)。

## 7. 最终代码、开关与复现

| 仓库 | 个人fork分支 | 最终测评提交 |
|---|---|---|
| vllm-plugin-FL | leanscale/iluvatar-topp-shadow-coverage | c50b5f0526e8112c72f1de918487300cf7376378 |
| FlagGems | leanscale/iluvatar-gqa-prefill-memory | de958ddfc4deeb43d6c94bb8652a3d4d3bc905fe |
| MiniCPM-Infer-Sys-Opt | main | 本报告、实验原始归档和独立审核 |

两个代码分支的远端提交已核验。官方基线固定为插件13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0与FlagGems a7620cc191a0b42e040194622c5758b22a7a25dc，没有通过合并上游最新代码更换基线。源码里保留的未采用原型不代表它们在服务中启用。

最终开关为 top-p=1、shadow=0、arm文件为空、MLP布局改动=0。top-p源码默认仍关闭，必须显式启用；关闭该开关即可回到同提交的 A 路径。前一已选性能配置 EXP20 的 ce19f93/de958dd 记录也保留，便于完整回退。收尾只更新交付配置，不启动常驻推理服务。

以下是与实际测评一致的复现入口；本次收尾未再次运行这些 GPU 命令。每组应使用独立服务和新的缓存目录。

```bash
export PYTHONPATH=/workspace/LeanScale/worktrees/EXP-0027-r1/vllm-plugin-FL:/workspace/LeanScale/worktrees/EXP-0027-r1/FlagGems/src
export VLLM_PLUGINS=fl
export VLLM_FL_TOPP_ROW_PARALLEL=1
export VLLM_FL_TOPP_ROW_PARALLEL_SHADOW=0
export VLLM_FL_TOPP_ROW_PARALLEL_SHADOW_ARM_FILE=
export VLLM_FL_MLP_WEIGHT_LAYOUT=0

# 每次独立会话换用新的 RUN_ROOT
RUN_ROOT=/workspace/LeanScale/artifacts/manual_final_reproduction
export VLLM_CACHE_ROOT="$RUN_ROOT/vllm_cache"
export TRITON_CACHE_DIR="$RUN_ROOT/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/inductor_cache"

vllm serve /workspace/MiniCPM5-2B --port 9031 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --served-model-name minicpm --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

在另一终端使用相同源码环境，服务就绪后运行官方脚本；脚本内部执行四轮并只弃首轮：

```bash
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases '[[4096,1024,64,256]]'

# 16K需另起独立服务及缓存，将上一条test-cases改为：
# '[[16384,1024,64,128]]'
```

历史 EXP29/30 控制器包含当日预算与禁止重复运行保护，不应直接修改日期后冒充同一实验。复现需新目录；本项目没有安排新的复现任务。

## 8. 可追溯性、失败与最终状态

最终审核没有只读取均值：逐文件检查 raw/归档 SHA、commit blob、四轮请求/token、官方参数、实际源码与编译路径，再独立重算 summary、CV、门槛及排队/抢占。生产控制器原有的“待审核”和“待推送”状态保留为历史快照，最新状态由独立审核、最终配置及发布回执说明。

本轮保留了三类非数值失败：

1. A完成后GitHub推送超时，控制器停止；数据已本地提交。后续临时SSH重试覆盖了仓库专用IdentityFile/UserKnownHostsFile，又造成主机密钥核验失败。恢复仓库原SSH配置后，c44b9798eb37969f97eee9b682c5de4f099719a8及其祖先已成功补推并核验。没有关闭主机密钥验证，也没有复制私钥。
2. 恢复审核r0错误套用4K的64+64编译变体数。真实16K当前A和历史EXP20均为36+31。修正为遍历全部_attention.json、与证据清单一一匹配并逐个核验语义/IR/hash；没有改推理实现、测量数据或门槛。
3. 最终审核r1遗漏“本地提交已核验”的发布状态分支，错误寻找不存在的B恢复回执。r2只修正该分支，保留所有原始/归档/commit blob检查，最终审核通过。没有追加GPU重测。

| 最终16K文件 | SHA256 |
|---|---|
| A raw CSV | 1c0f415b9c4b037b825ad046f764d867b3d665f92199e2b25ae71c2348d99282 |
| A summary | 1ec61f2dec76a79ea0f2239d65dd397aa11fc72d9f7377384450e56652525c58 |
| B raw CSV | 728c0bfab11da1473e0569a6aadfcf7179830a3241e793506646ace745f6bb74 |
| B summary | e088fde61b042af359a8dea5cad83d72cef8a5e8879ae9d7764aa2d83ec2de7e |
| 最终独立审核脚本 | d8cf85b357f8945d50c571f22b3cec25d716792d70c36df192ab85b04ced2d93 |

最终结论是：采用经过4K复核、16K非劣化检查的明确组合，停止无收益候选，保留精度未测和16K容量瓶颈这两项限制。实验已按用户要求结束，后续自动任务仅负责本次报告与交付收尾，完成后暂停。
