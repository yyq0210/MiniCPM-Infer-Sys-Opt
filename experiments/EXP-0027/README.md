# EXP-0027：top-p 行并行度生产接入与官方4K验证

在修改生产代码前登记。EXP26 的原内核 grid64 随机 logits 诊断通过140项精确/Graph/RNG检查，B64截断阶段平均-66.73%，尚无真实模型或E2E收益。当前基线为EXP20，插件ce19f93eee4c73f5a26911554700457e879f891d、Gems de958ddfc4deeb43d6c94bb8652a3d4d3bc905fe。

候选默认关闭，仅天数BI-V150、已核验vLLM源码及原TOPP内核版本、float32、V130560、单GPU路径启用；保持原算法/8192-4096 tile/warps-stage/RNG，不修改官方请求参数。先限制top-p-only且B17..64，其他batch/dtype/stride/设备/参数回退原实现；若扩展范围须额外记录和检查。B8/16原grid不变，B<8继续原PyTorch分支。

新launcher复用原JIT对象及原scratch缓存；扩容最多64行，不能双份常驻，不可越界。必须验证扩容前后旧Graph重放、不同B和行stride、缓存reset/设备隔离、默认关闭、安装幂等、上下游函数引用真正换成新入口及fallback结果。补B17/30/31/33/48/63等边界和小B回退。引擎源代码个人fork独立分支leanscale/iluvatar-topp-row-parallel，从EXP20创建，禁止继承未采用的MLP布局。

先真实服务diagnostic shadow：记录真实B/V/dtype/stride/p/k分支，有限次原输入上旧/新mask逐元素比较和RNG状态；采样参数CPU读取/比较仅诊断开关启用。不要把完整logits或请求原文推到远端。检查通过后关闭shadow，独立服务与缓存进行官方[4096,1024,64,256]四轮弃首，记录吞吐/延迟/失败、KV容量、实际路径及source hash。计时不包含shadow，也不把micro当收益。

每阶段重新分配完整预算，今日2026-10-29 23:35停止计算、23:50保存、次日00:00关机；不足则完成文档和源码，不能用过期窗口启动。完整官方结果前selected保持EXP20。整模任务精度暂缓，采样数值与RNG/tie检查不省略。只推个人fork，不创建上游PR；未完成验证的候选保持Draft状态。


## 实际服务流程

30项独立引擎入口检查与归档hash已复核，候选2a4cf99f5a765dc7de3a3337303e91183bc9decd已推个人fork。先shadow=1运行官方benchmark展开后的单轮4K/1024/64/256命令（只作诊断），要求计时区间真实B64/p=.95至少一次exact-mask和RNG检查。接线通过后停止服务、归档；再shadow=0独立新服务/缓存运行原官方脚本四轮弃首。诊断数据不合并进性能。原42MLP布局/GQA/RoPEKV/FULL捕获与实际top-p grid/counter/TTIR均检查。对照为同日EXP25 A2（3573.06 tok/s），并单列EXP20历史和ABBA均值。独立审核前不自动升级。
