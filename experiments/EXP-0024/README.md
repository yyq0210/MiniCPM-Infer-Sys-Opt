# EXP-0024：MLP权重加载布局接入与端到端验证

实现前假设：EXP23初筛与独立完整MLP复测已通过。gate/up与down同时改为[N,K] stride(1,N)，三个独立种子的M8/16完整MLP平均耗时下降12.63%，所有测试形状无回归。加载时一次性重排并替换原存储有望改善4K吞吐；尚无此项整模收益。
基线EXP20：plugin ce19f93eee4c73f5a26911554700457e879f891d、Gems de958ddfc4deeb43d6c94bb8652a3d4d3bc905fe。当前服务保持它们。独立工作树与分支开发，不修改冻结源；本项只改变权重布局，不称新增算子融合，不改变调度/量化/KV压缩。

拟接入UnquantizedLinearMethod.process_weights_after_loading；先核实真实加载生命周期，再实现默认关闭、可回退、幂等的精确范围适配。候选仅MiniCPM5-2B匹配配置、天数设备、BF16、非量化、TP1和当前已支持的无bias MLP两矩阵。VLLM_BATCH_INVARIANT、LoRA/其他未验证路径须守卫。保留原方法语义与Parameter元数据，unsupported场景原样返回。
重排前后权重逐元素完全相同，替换后释放原存储，不保留双份常驻权重挤占KV。只允许模型加载阶段执行；前向仍用原生linear，SwiGLU沿用真实生产编译路径。须记录当前实际weight stride及新stride，不以推导替代真实接线。

验证次序：CPU适用范围/回退/幂等检查；真实GPU参数元数据/值/布局/旧存储释放；真实服务42层gateup+42层down接线日志及Graph、原42GQA/RoPEKV/35FULL捕获尺寸、原调度、无逐次重排、KV容量；随后官方4K完整四轮弃首，独立结果审计。正向且预算足够时再16K/配对。完整模型任务精度按用户暂缓，不能声称赛事精度达标。

每阶段新有限窗口、GPU共享锁、两次空闲/端口/源hash检查、失败保留、结果与文档自动保存push。10月28日23:35计算截止、23:50保存，10月29日00:00机器关闭。比赛PR保持Draft，标题【FlagOS开放计算全球挑战赛S2.赛题2】【LeanScale】描述；无上游PR。
