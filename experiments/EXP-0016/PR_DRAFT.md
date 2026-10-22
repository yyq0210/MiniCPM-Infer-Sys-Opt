# Draft：赛事PR文案

标题：【FlagOS开放计算全球挑战赛S2.赛题2】【LeanScale】接入单级流水GQA prefill优化

## 变更
MiniCPM5-2B Q16/KV2/D128因果注意力在已验证分页decode布局上，显式选择prefill64行/4warps/stage1/tile32。
适用条件、回退、RoPE+KV融合和调度与父版本一致；FlagGems默认stage2保留。

## 验证
42项扩大数值/Graph检查、36个计时点通过；纯prefill及mixed单算子有收益，decode无显著回归。
真实42层接入与官方4K/16K四轮弃首轮测评已完成，吞吐分别3476.54/2314.37 tok/s，较EXP14观测+4.33%/+6.08%。4K增幅与历史波动接近，EXP17配对复核中。完整MATH500精度暂缓，保持Draft，不创建上游PR。
