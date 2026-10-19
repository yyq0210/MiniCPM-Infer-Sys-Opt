标题：**【FlagOS开放计算全球挑战赛S2.赛题2】【LeanScale】天数 D128 RoPE 分组并行与功能性输出**

目标：个人 fork `yyq0210/FlagGems`；base=`leanscale/flagos-s2-baseline`；head=`leanscale/iluvatar-rope-e2e`；状态 Draft。本文是 PR 正文草稿，不表示已创建 PR。

### PR Category

Operator

### Type of Change

Performance Optimization

### Description

为天数 FP16/BF16、D=128 的 RoPE 注册形状受限的后端实现。按 token × head group 分配 program，同一 lane 读取旋转对后计算两个输出；Q/K 分组共享一次 kernel launch。支持原地和独立输出，适配 QKV 投影切片的 stride；未覆盖 dtype、形状、位置或 stride 条件时回退原通用实现。

分支基于比赛指定官方 v5.3.5（a7620cc），保留通用实现，不合并后续上游优化。关联推理分支通过功能性输出将其接入真实模型的 native 编译入口。

### Issue

无关联 issue。关联推理分支：[vllm-plugin-FL](https://github.com/yyq0210/vllm-plugin-FL/tree/leanscale/iluvatar-rope-e2e)。[优化前后说明](https://github.com/yyq0210/MiniCPM-Infer-Sys-Opt/blob/main/docs/rope_before_after.md)。

### Progress

- [ ] Change is properly reviewed (1 reviewer required, 2 recommended).
- [ ] Change is responded to an issue.
- [ ] Change is fully covered by a UT.
- [x] 56 组目标形状、布局、原地/功能性输出检查通过；V 与原始输入完整性检查通过。
- [x] int32/int64 positions、通用回退与实际插件编译/Graph 检查通过。
- [ ] 官方两场景端到端 A/B 完整通过。
- [ ] 完整模型任务精度通过（按用户安排暂缓）。

### Performance

EXP-0002 中，独立通用 FlagGems 内核的 M64 微基准为 5.450 → 2.394 µs。EXP-0004 改为对比实际 native 代码的局部编译版本，并包含功能性调度接口：

| M | compiled-native µs | compiled-dispatch µs | 局部加速比 |
| ---: | ---: | ---: | ---: |
| 1 | 4.498 | 1.867 | 2.41× |
| 64 | 12.687 | 2.465 | 5.15× |
| 257 | 37.426 | 4.540 | 8.24× |
| 2048 | 244.853 | 32.954 | 7.43× |

以上为 BF16、Q/KV=16/2、D=128 的局部 Graph 重放中位数；与整层编译和 HTTP 整模吞吐属于不同测量层级。前后夹测原生实现，保留全部试验数据。官方端到端成绩未完成前，不宣称整模收益或赛事达标。
