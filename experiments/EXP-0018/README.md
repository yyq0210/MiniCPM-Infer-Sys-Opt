# EXP-0018：选定版本的 4K 分阶段热点诊断

2026-10-23。假设先于代码修改登记：GQA decode 布局与 prefill stage1 改变了算子成本，EXP9 的旧窗口占比不能代表当前版本。需要用真实 4K 请求形状，区分 mixed、大 batch decode 与尾部小 batch，再决定下一个优化算子或完整子图。

直接基线为已审核选定的 EXP16：插件 588a6147cd291e272324ab6c812024cd587c6287，FlagGems fd209df717648002ebf5ba4ec351424dcc997a53。选择与 ABBA 漂移限制见 ../../docs/gqa_prefill_selection_20261023.md。建立独立 detached 工作树，不修改推理或算子实现。

每个诊断 pass 使用原官方 command builder、4096 输入 / 1024 输出 / 最大并发 64 / 256 请求、默认采样、ignore_eos、原调度与完整 KV。FULL_DECODE_ONLY 与 NONE 各独立服务和缓存，各一次完整请求集；这是有 profiler 开销的诊断，不是官方四轮成绩，禁止比较其吞吐为优化收益。

四类有界窗口：initial、mixed、large_decode（纯 decode B≥48）、tail（本次主负载 256 请求已进入且 B≤16）。除 initial 外由 CPU metadata 触发；触发条件不保证整个窗口相同，以每步实际 phase 为准。预热和客户端自检必须排除在主负载计数外。未遇到条件或尾部不足 16 步时记录实际覆盖，不以时间回退伪造阶段。每请求 query lengths、CPU context 上界只在采样窗口保存，不读取 GPU 标量或改变 tensor/返回值。

保留 kernel、CPU 自身时间与阶段关联；只用 correlation/调用祖先关联，未关联项如实保留。GPU kernel 总时间不是墙钟延迟；不同模式、窗口不合并成整程占比。FULL Graph 内部不可见时使用 NONE 解释算子归属，不能拿 NONE 替代 FULL 官方性能。

必须验证：CPU 归因与 hook 行为、当前选定提交及哈希、42 层 GQA 与 RoPEKV 接入、实际 decode/prefill 编译 IR、FULL 图捕获。GPU 运行前获取共享锁、检查实际空闲和端口，采用当天有效计算截止；当前只准备，今日关机时间待回复。结果/失败自动落盘并按授权推送；不重启 EXP10，不重跑 EXP15 参数网格，不混入 KV 压缩。完整模型精度继续暂缓。
