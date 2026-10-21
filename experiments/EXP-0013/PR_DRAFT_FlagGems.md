# Draft

标题：【FlagOS开放计算全球挑战赛S2.赛题2】【LeanScale】探索天数 GQA 分页读取与查询行布局

在 EXP-0012 stage1 基线上增加实验性 decode_layout/decode_rows 参数，默认保持原路径；prefill、归约、完整KV、int64页地址及原调度保持。比较按页广播索引、K转置读取、8行逻辑查询及其组合。
8组共160项初测通过；layout1/rows8在三个工作点完整GQA耗时减少23.1%～24.9%，扩大验证和整模收益待定；无新引擎接入，维持Draft，不宣称整模收益或完整任务精度达标。
