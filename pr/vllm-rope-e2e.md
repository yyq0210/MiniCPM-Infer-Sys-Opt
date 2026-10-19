标题：**【FlagOS开放计算全球挑战赛S2.赛题2】【LeanScale】将天数 RoPE 分组并行实现接入模型编译路径**

目标：个人 fork `yyq0210/vllm-plugin-FL`；base=`leanscale/flagos-s2-baseline`；head=`leanscale/iluvatar-rope-e2e`；状态 Draft。本文是 PR 正文草稿，不表示已创建 PR。

### PR Category

Vendor

### PR Type

Performance

### Description

官方参数下 `custom_ops=['none']` 使 RotaryEmbeddingFL 使用 native 路径，独立 FlagGems RoPE 原型未进入模型。为天数、FP16/BF16、完整 D=128 且具有 K 的路径增加功能性 custom op，继续通过现有 CachedOp 选择实现，使分组并行 RoPE 可以进入 Inductor 模型图和 decode Graph。其他形状或设备保留原 native 路径。

### Related Issues

无关联 issue。关联算子分支：[FlagGems](https://github.com/yyq0210/FlagGems/tree/leanscale/iluvatar-rope-e2e)。实验：[EXP-0004](https://github.com/yyq0210/MiniCPM-Infer-Sys-Opt/tree/main/experiments/EXP-0004)。

### Changes

- 覆盖受支持条件下的 forward_native，并保持官方服务参数及默认 custom-op 配置。
- 新增不修改输入的 RoPE 编译接口和一致的 fake 实现；Q/K 独立输出，V 保持原值。
- 保留 CachedOp 的策略更新、strict 行为、失败回退和 IO-dump 接口。
- 基于比赛指定提交 13eb9be，不合并后续上游代码。

### Testing

天数 BI-V150 上，56 组基础数值/输入完整性检查通过；FP32、partial RoPE、缺省 K 回退，以及策略切换与注入失败回退检查通过。M=1/64/257/2048 的 torch.compile 与 Graph replay 检查通过，四种尺寸与 compiled-native 输出逐位一致。完整原始结果和首次测试断言失败均保存在 EXP-0004。

真实模型编译接入和官方两场景端到端结果，以 EXP-0004 最新状态与证据为准；不把局部 RoPE 微基准写成整模提升。官方配置固定为 `[4096,1024,64,256]`、`[16384,1024,64,128]`，每组四轮、舍弃首轮。完整 MATH500 精度按用户安排暂缓。

### Checklist

- [ ] I have run the existing tests and they pass
- [x] I have added tests for my changes (targeted validation in the experiment repository)
- [x] I have updated the documentation
- [ ] 官方两场景端到端 A/B 完整通过
- [ ] 完整模型任务精度通过（当前暂缓）
