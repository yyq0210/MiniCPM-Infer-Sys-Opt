# 【FlagOS开放计算全球挑战赛S2.赛题2】【LeanScale】Iluvatar 全注意力 KV 增长准入优化

状态：Draft 文案。尚未创建上游 PR；性能和真实服务接入仍待验证。

### PR Category

Vendor

### PR Type

Performance

### Description

长输入场景中，新请求的完整输入虽能暂时装入 KV 池，已有请求后续的输入分块和
decode 增长仍可能耗尽空闲块，触发抢占与重算。本改动在原分配器之前增加只读
准入判断：用真实块管理器计算所有运行请求和新请求到最大输出上界的剩余需求，
容量不足时让新请求留在原队列中等待。

保留官方服务参数、请求内容、输出长度、采样设置和完整注意力。此改动不进行 KV
压缩；不能消除物理容量限制，可能以更长排队或更低并发换取更少重算，因此必须
同时评价端到端吞吐和延迟。

### Related Issues

FlagOS S2 赛题 2；队名 LeanScale。

### Changes

- Iluvatar 平台仅在原本启用异步调度、未显式指定其他调度器等支持条件下选择新的
  `AsyncScheduler` 子类；运行时进一步检查单组、单卡、full attention。
- 使用实例级分配器代理，不修改共享 vLLM 安装或全局类。
- 保留原分配与释放实现，正确计算共享和可驱逐前缀块；准入失败不改变缓存状态。
- 首请求和不支持的请求/配置回退原逻辑；保留 FCFS/priority 行为。
- 增加准入计数日志、独立 CPU 分配器测试与完整官方测评的串行续跑脚本。
- 本分支建立在 EXP-0006 RoPE＋KV 融合候选之上；父分支算子代码保持不变。

### Testing

- Iluvatar 服务器，vLLM `0.24.0+empty`，PyTorch `2.10.0+corex.4.5.0`。
- `python3 tests/standalone/validate_kv_reservation_cpu.py /tmp/kv-admission.json`：
  13 项 CPU 检查通过；使用真实 KVCacheManager，未初始化 CUDA。
- 生命周期检查包含启用/关闭 prefix cache 各 96 条不同长度请求，分块 prefill、
  decode、取消、释放和后续准入，运行请求未出现容量分配失败。
- 官方 16K 优先，4K 作回归检查；均四轮、弃首轮。尚无本调度候选的性能结论。
- 完整模型任务精度按用户要求暂缓；不声明比赛精度达标。

### Checklist

- [ ] I have run the existing tests and they pass
- [x] I have added tests for my changes (if applicable)
- [x] I have updated the documentation (if applicable)
