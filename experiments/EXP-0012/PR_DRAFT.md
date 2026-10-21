# Draft：赛事PR文案

标题：【FlagOS开放计算全球挑战赛S2.赛题2】【LeanScale】接入天数GQA单级流水优化

支持的Q16/KV2/D128 Iluvatar GQA入口显式使用4warps、1stage、16segments，保留现有fallback、RoPE+KV融合和原调度。
此前decode固定2 stages，候选改为1 stage；FlagGems参数扫描及扩大数值/Graph验证见EXP-0011。
单算子代表点耗时减少13.8%～20.8%，不是整模收益。真实42层接入、官方4K/16K四轮结果待补。
完整任务精度按用户要求暂缓，维持Draft，不宣称满足最终赛事精度要求，未创建上游PR。
