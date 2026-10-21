# Draft

标题：【FlagOS开放计算全球挑战赛S2.赛题2】【LeanScale】探索天数 GQA 分页读取与查询行布局

在 EXP-0012 stage1 基线上增加实验性 decode_layout/decode_rows 参数，默认保持原路径；prefill、归约、完整KV、int64页地址及原调度保持。比较按页广播索引、K转置读取、8行逻辑查询及其组合。
单算子正确性与性能待测，扩大验证及整模测评待定；无新引擎接入，维持Draft，不宣称整模收益或完整任务精度达标。
