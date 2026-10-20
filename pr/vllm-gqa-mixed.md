标题：**【FlagOS开放计算全球挑战赛S2.赛题2】【LeanScale】天数 GQA 分段解码与混合批次推理优化**

目标个人 fork yyq0210/vllm-plugin-FL；head=`leanscale/iluvatar-gqa-mixed`；base=`leanscale/flagos-s2-baseline`。状态 Draft；本文是正文草稿，未创建上游 PR。

### PR Category

Vendor

### PR Type

Performance

### Description

MiniCPM5-2B 的 Q16/KV2/D128 attention 在 prefill/decode 混合批次中原本整体退到 2D kernel。为 Iluvatar 接入 FlagGems paged GQA 内核：GPU 内分流两类请求，decode 保留分段 softmax，prefill 每块复用 8 个 query token 的 KV 读取。继承原 Triton metadata、KV layout/update 和 Graph 支持；特性不支持时调用原 forward。

分支同时包含此前 functional RoPE 接入。保留原 attention_backend 的 CachedOp 策略/选择顺序；不通过调参或切换环境变量代替算子优化。

### Related Issues

无关联 issue。[实验、失败记录及正式成绩状态](https://github.com/yyq0210/MiniCPM-Infer-Sys-Opt/tree/main/experiments/EXP-0005)。

### Changes

- FlagOS/Iluvatar 分支返回继承 Triton 的实现类，其他平台与显式 backend 选项保持原路径。
- 有界支持 BF16/FP16、因果 Q16/KV2/D128；量化、滑窗、sink、ALiBi、prefix-LM、batch invariance、TD 等回退。
- 使用现有 FP32 scratch，不增加 KV 副本或修改调度参数。

### Testing

- 40 项算子数值/分页/Graph 检查及两轮诊断矩阵完成。
- 固定配置边界、实际 backend forward、回退、常量 V 不变量和模型接入证据见实验目录。
- 官方两场景必须完整测量后才能报告整模收益；MATH500 按用户要求暂缓。
- 已发现现有短 prefill 路径与独立参考不一致，保留证据，未宣称完整模型精度达标。

### Checklist

- [ ] I have run the existing tests and they pass
- [x] I have added tests for my changes (if applicable)：复现与验证脚本在实验仓库
- [x] I have updated the documentation (if applicable)
