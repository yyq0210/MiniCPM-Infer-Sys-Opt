# EXP24 官方4K接线检查修复

前次19:15:16–19:20:33服务已完成84权重重排、42 GQA/42 RoPEKV/35 FULL捕获与标准路由检查，KV容量522352 tokens与EXP20相同。尚未开始benchmark，0轮计时。失败原因是诊断脚本假设所有MLP位于一个生成模块；实际vLLM在attention边界分图，42个不同MLP编译模块各含gate/up与down一次（另有首段非MLP模块）。原服务代码、检查源码与失败完整保留，归档4fe573a2ec5ca754ffd55b459f6032d67c379e59。

只修编译证据组织检查：要求42个不同源码哈希的模块各一对投影，所有matmul仍直接读取原参数且stride严格匹配；同时解析完整FX图，逐一核对层0..41的gate/up和down原生linear节点各一次。保持84真实加载日志、数值相等、旧GQA/RoPEKV/Graph/容量与所有benchmark条件。未放宽精度或性能门槛。先在失败run的只读编译缓存上离线验证新检查，再新run、新缓存、新窗口执行。推理插件41fd7a4与FlagGems de958完全不变。
