# EXP-0006：跨算子融合筛选
## 授权与验证边界
用户于 2026-10-20 要求开展算子融合实验，筛选实测较优版本，随后研究系统调度和 KV 压缩。无损候选暂缓模型任务精度，仅做必要数值/别名/分页/Graph/接入检查；KV 压缩必须比较完整上下文基线的模型精度和长文任务。不能声称有限搜索得到全局最优。
## 起点
EXP-0005 插件 b4c932d9d6c3a1fe0af71444caac2e83d97d1d06；算子 25a16284ba3a4cfeb01133e1665260050ca5e22a。保留 pinned 官方基线和所有 dispatch/fallback；独立 worktree，不改写正在测评的 EXP-0005。
## 阶段
1. RoPE + paged KV update：functional Q/K 输出与缓存副作用合并，保留 attention 数据依赖；比较 head 分组、token 分块与是否输出 K。实际图需验证每层只有一次缓存写入。
2. MLP gate/up GEMM + SwiGLU 局部融合：与原厂 GEMM 加编译激活公平比较，小 M decode 和大 M prefill 分开；数值舍入必须明确。
3. 算子候选通过后，用真实引擎图、Graph 和固定官方负载验证；保留没有收益的配置。
4. EXP-0007 调度研究：先从完整 16K 日志确认重算/容量/混合计算，研究 decode 增长余量和成本感知 prefill，保持官方参数和请求不变。
5. EXP-0008 KV 压缩：先建立 MATH500 与长文精度基线，再做有限预算的 prompt 压缩原型；记录选择/搬移开销、真实释放物理页和调度准入。精度未通过不得作为最终方案。
## 执行纪律
GPU 实验持有共享 runner.lock，与 EXP-0005 串行。微基准不冒充整模结果；所有尝试保留。实验结果与脚本同步文档仓库，代码分别同步两个 fork；不自动创建上游赛事 PR。

## 已实现候选和后续工作

新增 residual+RMSNorm+QKV GEMM 诊断候选；完整参数矩阵、精度门槛和后续阶段见 [RESEARCH_SEQUENCE.md](RESEARCH_SEQUENCE.md)。8 项 CPU 检查已通过，GPU 待测；不能据此声称完整模型接入或新增性能收益。
