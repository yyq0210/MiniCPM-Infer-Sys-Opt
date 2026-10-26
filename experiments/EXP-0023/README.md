# EXP-0023：原生线性层权重存储布局诊断

实现前假设：EXP22独立补零全部不满足收益门槛。保留原生dispatch_unquantized_gemm→default_unquantized_gemm→F.linear，加载时一次性W.T.contiguous().T将权重[N,K]从stride(K,1)变为stride(1,N)，检查原生库不同输入存储布局的效率。
基线为EXP20 plugin ce19f93eee4c73f5a26911554700457e879f891d、Gems de958ddfc4deeb43d6c94bb8652a3d4d3bc905fe；不改服务代码和调度。
本轮只用随机张量；原布局由create_weights源码推导，尚无本轮真实服务weight stride探针。QKV/O/gate_up/down形状与EXP22相同。
固定M=[1,4,8,16,30,32,64,128,512,2048]，hot与weights8两模式，7样本×8Graph调用、旧/新/旧夹测，80个计时点。M30是NONE诊断，生产尾部主目标M8/16。
40个BF16数值/Graph检查以及M1/8/16/2048共16个FP16检查，共56；固定atol0.01/rtol0.02，FP32参照和输入/权重更新Graph重放。另记录一次性packing耗时/内存和M8/2048 CPU算子事件，不能把CPU事件当作底层库无内部拷贝的证明。
预注册投影级门槛：该投影M8/16在两模式均至少降低5%耗时，所有其余M/模式均不回归超过3%，所有夹测绝对漂移<=3%，无观察到的aten::clone/contiguous/copy_等隐式拷贝事件。完整覆盖后才判定候选；不改阈值、不删慢点。
候选仍须独立复测、实际权重stride及集成路径核查、完整子图、引擎接入和官方E2E；不能从微基准直接宣称整模收益。
每投影使用8独立权重仅是缓存敏感性代理。持久双份权重不在本轮部署方案内，以免挤占KV；prefill退化不得简单全局切换。
全模型精度按用户暂缓，算子检查必须通过；失败保留。使用新的有限时间窗口和GPU锁，最晚23:35停止计算/23:50保存。
