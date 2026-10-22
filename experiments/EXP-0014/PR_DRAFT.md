# Draft：赛事PR文案

标题：【FlagOS开放计算全球挑战赛S2.赛题2】【LeanScale】接入GQA分页页号广播与8行查询布局

Q16/KV2/D128的decode按页读取页表并广播到token，逻辑查询行从16收紧到有效的8个head。
插件显式选择decode_layout=1、decode_rows=8，4warps/1stage/16segments及原RoPE+KV融合、调度、回退保持。
FlagGems候选沿用EXP-0013冻结提交，26项扩大数值/Graph检查和34计时点已通过；单算子收益不代表整模收益。
真实服务接入、官方4K/16K四轮结果待补；完整任务精度暂缓。保持Draft，不创建上游PR。
