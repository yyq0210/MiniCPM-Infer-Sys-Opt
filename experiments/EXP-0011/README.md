# EXP-0011：GQA decode 参数优化

用户于2026-10-21要求停止劣化调度方案，恢复原最优融合方案，转入GQA优化。
本实验直接基于EXP-0006的插件07d9fa0c与FlagGems484413ea，不包含EXP-0010的KV准入调度器。
官方原始基线保持不动，失败及中断实验保留。

## 假设与路径（修改代码前记录）

实际入口为IluvatarGQAImpl.forward → FlagGems paged_gqa → _attention(decode) → _reduce。
Q16/KV2/D128，8个Q heads共享KV，decode BM16只有8行有效。
已有诊断显示主kernel占NONE/late可见GPU kernel时间78.34%，但该占比不代表生产Graph延迟。
先固定tile32与prefill64行，独立扫描decode warps={2,4}、stages={1,2,3}、segments={8,16}。
不改KV内容、采样、官方负载或服务参数；不同时引入NSEQ/分页布局修改。

## 筛选与验证

首轮为单算子诊断，不是官方端到端成绩。12种配置分别用独立进程，失败配置完整保留。
数值检查包括BF16/NHD、FP16/HND、非连续Q、ragged/空query、mixed，独立FP32 causal参考。
Graph重放更改Q、K/V、length和block table内容，同时保持地址稳定，核对eager与Graph。
计时工作点为B64/L4700、B30/L16384、B1/L4096，包含主kernel及reduce。
每种配置以原最优版本前后夹测，7次Graph event采样；另列主kernel、reduce计时，不混同总调用计时。
预设筛选：主要两个工作点均不退化超过3%，至少一个改善3%；小batch若退化超过5%则只能考虑明确的shape分支，不能全局替换。
之后才做扩大工作点、FP16/BF16全布局、int64大偏移、prefill2048/mixed和完整Graph检查；通过后接入42层真实模型，再跑官方四轮。
完整模型精度按用户要求暂缓；KV压缩不在本实验范围。

## 截止与保存

遵守artifacts/run_window.json的23:35计算截止。控制器独占原GPU锁，子进程超时不超过截止时间。
每个trial结束即原子保存；控制器结束后发布紧凑结果和日志，不上传缓存、模型或数据集。
