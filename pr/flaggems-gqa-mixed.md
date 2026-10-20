标题：**【FlagOS开放计算全球挑战赛S2.赛题2】【LeanScale】天数分页 GQA 分段与 Query 复用优化**

目标个人 fork yyq0210/FlagGems；head=`leanscale/iluvatar-gqa-mixed`；base=`leanscale/flagos-s2-baseline`。状态 Draft；本文是正文草稿，未创建上游 PR。

### PR Category

Operator

### Type of Change

Performance Optimization

### Description

增加 Iluvatar Q16/KV2/D128 分页因果 attention：每个 KV head 复用 8 个 Q heads；decode 按 sequence × KV head × 16 segments 启动，prefill 的 64 行分块覆盖 8 个 query token。混合批次在 GPU 内分流，不把 decode 强制放入 prefill 内核。固定 tile32，使用现有持久 FP32 buffer 合并稳定 softmax；保留真实 block table 和 stride，不压缩 KV。

### Issue

无关联 issue。[推理接入分支](https://github.com/yyq0210/vllm-plugin-FL/tree/leanscale/iluvatar-gqa-mixed)。

### Progress

- [ ] Change is properly reviewed (1 reviewer required, 2 recommended).
- [ ] Change is responded to an issue.
- [ ] Change is fully covered by a UT.
- [x] 独立实验仓库保存数值、布局、Graph、回退及全部选参/失败记录。
- [ ] 官方整模性能与完整 MATH500 全部完成。

### Performance

BI-V150 算子诊断：B30/16K decode 约 2849→2611 µs（1.091×）；30 decode+2018-token prefill 约 73.3→22.3 ms（3.28×）；2048-token prefill、16K 总 KV 约 70.6→19.5 ms（3.62×）。B8/16K 约慢 1.5%，不宣称所有形状均收益。此表不是整模成绩。

[全部配置、基线漂移与失败](https://github.com/yyq0210/MiniCPM-Infer-Sys-Opt/blob/main/experiments/EXP-0005/DIAGNOSTICS.md)。分支包含此前 RoPE；正式测评报告组合收益，不做未经验证的单项归因。
