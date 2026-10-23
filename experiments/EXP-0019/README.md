# EXP-0019：4K GQA prefill 分页读取与 K 加载布局

2026-10-23，先登记假设，再开发。EXP18 当前版本真实 4K mixed 窗口中 prefill 占可见 kernel 工作量45.20%；16步实际B10–17/M2048。详见[当前热点](../../docs/operator_profile_4k_20261023.md)。这不是整程占比或新的性能收益。

假设：当前页号广播与按[token,dim]加载K后转置仅用于decode。将两种访存方式分别应用到prefill，可能减少重复页号读取或改善加载布局；也可能增加转换、寄存器和shared开销。先比较四个固定配置：layout0原路径、layout1仅页号广播、layout2仅K加载方向、layout3组合。固定rows64/warps4/stage1/tile32，decode原配置不变，不重复EXP15参数网格。

直接基线与当前已选版本是EXP16：插件588a6147cd291e272324ab6c812024cd587c6287，Gems fd209df717648002ebf5ba4ec351424dcc997a53。独立工作树，新增实验接口默认prefill_layout=0，保持通用默认和服务版本；候选尚未实现/验证/接入。

由EXP18逐步metadata导出真实query长度与CPU context上界，用随机Q/K/V构造有来源的形状诊断，不冒充原请求实际tensor或设备length。固定选取initial两步纯prefill、mixed的37/38/51/52步，并保留B64合成mixed、16K及小batch decode回归。工作点先固定，不按结果挑点。

先FP32 oracle、BF16/FP16、随机物理页、NHD/HND、ragged/empty、页边界与Graph修改Q/K/V/length/BT；初筛胜者再扩大all-query和真实int64大地址验证。预热后原版/候选/原版夹测完整GQA，另列prefill-only与编译资源，保留全部原始采样和失败。至少一个主要prefill/mixed点耗时减少5%，其他主要点不退化超过3%，decode回归不超过5%，夹测基线漂移不超过3%；仅作为候选门槛，不等于端到端收益。

通过扩大检查后才改插件显式参数并验证实际42层接入，再测官方4K四轮弃首，16K回归。完整模型任务精度按用户要求暂缓；不混入调度、采样或KV压缩。

当前只登记CPU开发准备。EXP18三小时短窗口仅覆盖已完成的两组profiling，不自动继承给本实验。任何GPU启动必须另行记录当日有限预算、共享锁/实际空闲、源码与脚本哈希以及自动保存推送策略；今天实际关机时间仍未知。
