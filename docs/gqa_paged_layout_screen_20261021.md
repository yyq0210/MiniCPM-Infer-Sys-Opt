# GQA 分页索引与 8 行查询：筛选结果

2026-10-21，EXP-0013。基线是已完成官方 E2E 的 EXP-0012 stage1。8组全部通过20项初步检查，共160项；全部配置与采样保留。
按页加载后广播＋8行逻辑查询（layout1/rows8）是三个工作点中收益最大的组合。尚未接入引擎，本报告不是端到端收益。

| 配置 | B64/L4700 时间变化 | B30/L16384 时间变化 | B1/L4096 时间变化 | 前后基线稳定 |
| --- | ---: | ---: | ---: | --- |
| layout0_rows16 | -0.081% | +0.211% | -0.221% | True |
| layout1_rows16 | -12.483% | -13.807% | -17.647% | True |
| layout2_rows16 | -7.184% | -6.171% | -1.352% | True |
| layout3_rows16 | -11.622% | -10.158% | -10.381% | True |
| layout0_rows8 | -13.605% | -13.957% | -5.986% | True |
| layout1_rows8 | -23.423% | -24.949% | -23.106% | True |
| layout2_rows8 | -11.644% | -11.062% | -0.740% | True |
| layout3_rows8 | -16.562% | -16.051% | -9.603% | True |

负数表示完整 GQA 调用耗时减少，包含主核和归约。不要把这些百分比当成整模吞吐提升。
等价对照的时间变化在0.23%内；所有配置的前后基线漂移低于0.26%。每组7个样本，每次Graph包含8次调用。
layout1是page16/tile32时按页加载两个页号并广播到32个token；layout2改变K读取方向；layout3组合两者。rows8将逻辑查询行减到真实8个Q head。prefill保持原路径。
单独页广播和单独rows8均有收益，组合进一步改善；K转置读取虽然减少shared字段，但寄存器字段增多，整体逊于layout1/rows8。
最优组合资源字段 n_regs 96→83、shared 20480→18432、n_spills仍0。TTGIR展示页号载荷2元素、查询及累加张量8行，而MMA instrShape仍[16,16,16]；不能据此声称Tensor Core计算量减半或确切occupancy提升。
初测覆盖page8/16/32/64、BF16/FP16、NHD/HND、空context/query、页边界、ragged/mixed、非连续Q、FP32 oracle和Graph内容变化。
下一步固定同一源码和layout1/rows8，扩大到26项数值/Graph/实际>2^31元素地址检查、34个计时点；纯prefill与mixed也要检查回归。通过后才考虑引擎接入及官方四轮。若收益不能复现或出现不可接受回归，保存负结果转GQA prefill调优。

[全部原始筛选结果](../experiments/EXP-0013/artifacts/20261021_201938_screen/RESULTS.md)。
[IR证据与哈希](../experiments/EXP-0013/screen_ir_evidence.json)。原始IR完整保留服务器；不上传二进制缓存。

## 扩大验证与 occupancy 后续证据

扩大验证已于20:30:05完成：26项数值/Graph检查通过，34个计时点无超过3%回退，全部基线漂移低于0.5%。归档提交03a4a87d9a02f1e61807f17881fa231846cc48b1。

20:35对真实编译内核查询CoreX occupancy API，预测每SM驻留块数6→7，理论occupancy 18.75%→21.875%；12个内核IR哈希与原筛选完全一致。这补充了此前“单凭资源字段不能推断”的证据，但仍未测得运行时achieved occupancy。代码逐步示例、资源门槛计算和适用边界见[专门说明](gqa_paged_layout_code_occupancy_20261021.md)。
