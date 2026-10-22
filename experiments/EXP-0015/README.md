# EXP-0015：GQA prefill 参数初筛

2026-10-22，LeanScale。先记录假设，再修改源码。本实验继续既定 GQA 优化顺序；用户要求的一轮结束后 4K 专项暂不提前执行。4K 与 16K 同时作为主要工作点，完整任务精度按用户授权暂缓，算子数值和 Graph 校验必须执行。

## 直接对照与假设

EXP-0014 已完成两组官方四轮测评（弃首轮）：4K 3332.27、16K 2181.74 total tok/s，已选为性能基线。插件 `82c6a143f6984f946df42125f1a8699799e8045f`，FlagGems `1c526ca99d412e203ed3caa8b1663566976ea38c`。

入口为 IluvatarGQAImpl.forward → paged_gqa → _attention(DECODE=False)。当前 prefill 固定 rows64/tile32/warps4/stages2；64 行对应 8 个 query tokens，每个 query 的 8 个 Q heads 共享 KV。调整行数和流水可能提高 KV 复用或降低资源压力，也可能因并行度、shared、寄存器及 padding 退化，需实测。

仅为 prefill launch 增加独立 warps/stages 参数，默认4/2保持原行为；_attention 与 _reduce 内核源码、decode 参数和原调度完全不改。现有 prefill_rows/prefill_tile 参数用于诊断。插件工作树保持已选提交，不接入未经验证的参数。

## 初筛矩阵与校验

固定 tile32，rows={32,64,128} × warps={2,4} × stages={1,2}，共12组。第一组 rows64/warps4/stages2 是等价对照；该组失败立即停止并排查。其他配置的编译失败、资源超限、数值失败和超时均保留，不只报告优胜结果。

每组先执行20项 BF16/FP16、NHD/HND、page8/16/32/64、非连续Q、空/ragged/mixed、query块边界与尾部检查；比较冻结基线的全部输出，并对每序列抽样query使用独立FP32 causal oracle。Graph重放修改Q/K/V/lengths/block table，与eager严格一致并再查oracle。扩大验证阶段另覆盖所有query、CU布局变化、长地址和更多形状，初筛不替代扩大验证。

计时为整个 paged_gqa 调用，包括混合批次的prefill/decode/reduce；prefill主核时间与编译资源单列用于解释，不用其时间替代整个调用。工作点：

1. prefill query2048/context4096。
2. prefill query2048/context16384。
3. mixed4K：[63个decode + query1985]，decode context4700、prefill context4096。
4. 同上，prefill放在首序列。
5. mixed16K：[30个decode + query2018]，context16384。
6. 同上，prefill放在首序列。
7. decode B64/L4700。
8. decode B30/L16384。
9. decode B1/L4096。

mixed是声明的合成诊断点，不宣称是实际trace中的每个批次。全部使用同一随机种子和随机物理页；各配置独立进程预热，Graph event每组7样本、每样本8次调用，对照/候选/对照夹测。记录全部采样、IR哈希、registers/spills/shared；资源数字不是动态occupancy。

筛选门槛：全部计时点基线漂移≤3%；前六个pure/mixed点均不退化>3%，至少一个耗时降低≥5%；三个decode点不退化>5%。只生成候选，不自动改变服务版本。若收益方向明确，再以单因素方式复核tile64/stages3或工作点分派；不预先启动第二阶段。

## 接入、归档与执行窗口

通过初筛后，扩大 BF16/FP16、布局、地址、query块映射和Graph检查；获选后再接入真实42层并核验实际编译参数，完整测评官方4K与16K各四轮弃首轮。纯参数调整不改变完整KV或attention定义，仍不得省略算子数值验证。

控制器独占共享GPU锁，每配置最多10分钟且不超过当日23:35计算截止；临近截止不启动新配置。每次结果原子落盘，控制器结束后保存并推送全部紧凑结果和失败日志，推送失败仍保留本地。23:50保存窗口、00:00关机。原始基线及旧最优工作树冻结；不恢复EXP-0010劣化调度，不混入KV压缩。
