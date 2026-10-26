# Draft

标题：【FlagOS开放计算全球挑战赛S2.赛题2】【LeanScale】探索天数GQA prefill分页读取与K加载布局

增加prefill_layout实验接口：0保持现有路径，1页号广播，2按token/dim加载K再转置，3组合；通用默认0，decode分派及reduce保持。基于当前4K热点验证有限候选，尚未通过GPU数值/Graph/性能或接入检查，不宣称收益。服务继续使用已选EXP16。CPU派发与地址模型检查不代替GPU验证；完整模型任务精度暂缓，保持Draft。


2026-10-26更新：四布局初筛80项检查/56个计时点完成；layout1扩大42项全query/Graph检查及42计时点通过，39444个query含实际int64地址。详见docs/gqa_prefill_memory_20261023.md。EXP20引擎接入准备完成，官方端到端收益与完整任务精度尚未确认。
