# Draft

标题：【FlagOS开放计算全球挑战赛S2.赛题2】【LeanScale】探索天数GQA prefill分页读取与K加载布局

增加prefill_layout实验接口：0保持现有路径，1页号广播，2按token/dim加载K再转置，3组合；通用默认0，decode分派及reduce保持。基于当前4K热点验证有限候选，尚未通过GPU数值/Graph/性能或接入检查，不宣称收益。服务继续使用已选EXP16。CPU派发与地址模型检查不代替GPU验证；完整模型任务精度暂缓，保持Draft。
