标题：**【FlagOS开放计算全球挑战赛S2.赛题2】【LeanScale】天数 RoPE 按 Head 分组并行内核原型**

目标：个人 fork `yyq0210/FlagGems`；base=`leanscale/flagos-s2-baseline`，head=`leanscale/iluvatar-rope-v1`；状态 Draft。

### PR Category

Operator

### Type of Change

Performance Optimization

### Description

新增天数 RoPE 分组并行内核原型，将每个 token 内的逐 head 循环改为按 head 分组分配 program，并由同一线程处理一对旋转元素的读取、计算和写回。保留现有默认路由，未接入正式模型执行。

该分支基于比赛指定 FlagGems v5.3.5（a7620cc），未合并后续上游优化。当前默认 vLLM 编译路径使用 native RoPE，因此下面的微基准不能代表整模吞吐收益。

### Issue

无关联 issue。实验记录：[EXP-0002](https://github.com/yyq0210/MiniCPM-Infer-Sys-Opt/tree/main/experiments/EXP-0002-rope)。

### Progress

- [ ] Change is properly reviewed (1 reviewer required, 2 recommended).
- [ ] Change is responded to an issue.
- [ ] Change is fully covered by a UT.
- [x] 完成 180 次候选/独立参考检查，实际 BF16/FP16 D=128 与原内核逐位一致。
- [ ] 完成默认整模执行路径接入及 A/B。
- [ ] 完成 math_500 Level 3 精度验证。

### Performance

BI-V150 32GB，BF16，Q/KV=16/2，D=128。CUDA Graph 重放、7 轮中位数；每行是本轮 sweep 最优配置。

| token 数 | 原 FlagGems µs | 候选 µs | 内核加速 |
| ---: | ---: | ---: | ---: |
| 64 | 5.450 | 2.394 | 2.28× |
| 2048 | 31.256 | 17.857 | 1.75× |

FP32 扩展形状与原实现的差异仍需定位，失败日志和独立参考结果均已保留。没有整模加速或精度达标声明。

复现命令、完整 sweep 和原始记录见实验仓库 EXP-0002。提交前不勾选尚未运行的上游 UT 或尚未完成的评测。
