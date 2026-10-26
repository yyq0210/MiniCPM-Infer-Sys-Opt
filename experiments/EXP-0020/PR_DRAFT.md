# Draft：赛事PR

标题：【FlagOS开放计算全球挑战赛S2.赛题2】【LeanScale】接入GQA prefill页号广播优化

MiniCPM5-2B Q16/KV2/D128因果注意力显式选择prefill_layout=1，减少同页token的重复页表读取。
原decode组合、RoPE+KV融合、支持边界、fallback和调度保持。FlagGems默认0保留。
EXP19通过42项全query数值/Graph检查、39444个query与42个计时点；纯prefill耗时减少16.76%–24.10%，mixed减少15.52%–18.80%，decode变化-0.07%..+0.43%。
真实42层接入和官方4K/16K测评待完成，尚无整模新增收益；完整任务精度暂缓，保持Draft，不创建上游PR。


2026-10-26：真实42层、FULL_DECODE_ONLY、64decode+64prefill实际IR已通过。官方4K四轮弃首后3579.12 tok/s，相对EXP16观测+2.95%；TTFT-8.21%、TPOT-2.61%。因跨日历史对照及CV3.27%，增量待配对复核；16K回归正在运行，仍为Draft，不宣称新增稳定收益或完整精度达标。


16K更新：官方四轮弃首后2541.43 tok/s，较EXP16观测+9.81%、原始+177.06%，TTFT-10.03%、TPOT-8.60%；保留轮CV0.06484%，36decode+31prefill实际编译证据通过。两组正式测试均已完成，4K同期固定ABBA归因复核正在运行；完整MATH500尚未测，仍Draft。
