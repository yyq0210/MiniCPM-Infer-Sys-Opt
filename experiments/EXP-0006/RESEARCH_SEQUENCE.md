# LeanScale 后续实验顺序与验证门槛

## EXP-0006：融合

本轮已实现三个候选，有限搜索范围内选择经复测确认的版本；不宣称全局最优。

| 候选 | 参数搜索 | 接入状态 |
| --- | --- | --- |
| RoPE + paged KV cache 写入 | head tile 2/4/8/16，token tile 1/4；六组有效组合 | 已写图切分前替换和回退，CPU 检查通过，待 GPU 和真实模型验证 |
| gate/up GEMM + SwiGLU | BM/BN/BK/warps 四组组合 | 仅诊断；未接入服务 |
| residual + RMSNorm + QKV GEMM | 三组 GEMM tile | 仅诊断；未接入服务 |

RoPE 计时为 N=1/8/30/64/128/512/2048；MLP 为 N=1/8/30/64/512/2048；
Norm+QKV 为 N=1/8/30/64/512。每个场景夹测原实现，CUDA Graph 预热后 event 计时，
保存每次采样与失败。计时采用全部有效 slot，负 slot/padding 单独检查。

RoPE 验证包括 FP16/BF16、NHD/HND、非连续 QKV、NeoX/interleaved、页面边界、
未写页保持、projection 不变、改变输入与 slot 后 Graph 重放、超过 2^31 元素的地址。
RoPE 及 cache 与此前专用 RoPE+cache 路径逐元素一致，另对独立 FP32 公式检查。

Norm 融合保留已安装 vLLM IR 的舍入点：FP32 residual sum；归一化后先转 gamma dtype，
再乘 gamma 并转换投影输入 dtype。GEMM 候选固定 BF16 数值容差，不在失败后放宽。
这些检查只防止接线、数学与内存错误；按用户要求暂缓无损路径的完整模型任务精度。

筛选后必须再次测量候选，确认 42 层真实融合、仅一次 KV 更新、Graph/fallback 正常，
然后保持原官方 serve 和两组 benchmark 配置做整模比较。没有收益的融合保留记录而不启用。
已有 EXP-0005 官方测评仍运行时，所有新 GPU 工作排队，不修改其源码。

## EXP-0007：系统调度

融合版本确认后单独开展，避免混合改动导致无法归因。
从 16K 日志和 counter 分析 waiting/running、KV 占用、抢占及重算；容量不足与计算干扰分开判断。
先尝试为已有 decode 的输出增长保留动态 KV 余量，利用 allocator 已有 admission watermark
接口而保留原请求、token budget、最大长度和显存比例。只有实际存在抢占/重算才将减少它视为收益。
成本感知 prefill chunk 为后续候选，不能只把官方参数换成更有利的数字。

评价同时包括总吞吐、输出吞吐、TTFT、TPOT/ITL 尾延迟、队列时间、抢占与公平性。
无损调度暂缓完整模型任务精度，但必须检查请求不丢失、不重复、无饥饿、KV 生命周期正确。

## EXP-0008：KV 压缩

用户已明确授权并要求验证 KV 压缩精度。先建立完整上下文的质量基线，
再研究 prompt 尾部观察窗口选择和分页物理回收；持续 H2O 淘汰属于后续更复杂候选。
GQA 的保留集合以共享 KV head 为单位，逻辑绝对位置与压缩后物理位置分离，
不能把只增加 attention mask 当成释放物理缓存。

服务器赛题说明记录的官方 MATH500 Level 3 配置为：
batch=8、temperature=1.0、top_p=0.95、max_tokens=32768、timeout=3600，要求分数至少 0.95。
实际执行前记录数据、脚本与服务代码哈希；所有失败/超时均须计入完整性审查。
这是服务器说明中的要求；2026-10-20 赛事网页经检索工具无法打开，尚未获得新的网页规则修订。

MATH500 的短 prompt 可能没有触发静态 prompt 压缩，因此还必须记录每个请求是否真实裁剪，
并加入 4K/8K/16K/32K 的多位置检索和跨段推理。单靠短题分数不能证明 16K 压缩质量。
选预算用独立校准任务，不用官方评测答案调参。压缩版本需同时满足：

1. 完整上下文/压缩版本使用同一质量评测配置，保存逐样本结果与失败；
2. 官方质量门槛和长文任务通过，边缘结果需复测而非选择最好一次；
3. 真正归还物理块、增加可用容量，并计入评分、选择和搬移成本；
4. 真实引擎中 position、slot、prefix sharing、Graph 与释放引用正确；
5. 与完整上下文版本分别报告质量和性能。未验证的近似方案不并入最终候选。

上述“先静态 prompt 压缩”的排序是结合本项目工程成本的判断，不是论文对本模型的性能承诺。
[SnapKV](https://arxiv.org/abs/2404.14469) 提供尾部观察窗口的选择思路；
[KV-Compress](https://arxiv.org/abs/2410.00161) 提供分页回收的系统设计参考；
[H2O](https://arxiv.org/abs/2306.14048) 提供累计重要性与近期 token 的动态保留策略。
[Sarathi-Serve](https://arxiv.org/abs/2403.02310) 是调度成本与延迟权衡的参考。
