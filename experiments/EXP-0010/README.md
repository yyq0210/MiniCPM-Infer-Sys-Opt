# EXP-0010：KV admission 调度接口修复与正式评测

日期：2026-10-21，队名 LeanScale。

假设：EXP-0007 的失败来自未接收 EngineCore 传入的 throttle_prefills 参数。继承实际 AsyncScheduler 接口并原样透传；覆盖启用/禁用、位置/关键字参数和异常恢复路径。通过 CPU 生命周期检查及真实 HTTP smoke 后，以 EXP-0006 相同内核运行官方 16K/4K 四轮测评（首轮舍弃）；若退化应保留结果并撤回候选。不修改 KV 数据，不实施 KV 压缩。

固定内核基线：插件 EXP-0006 07d9fa0cfdc5f09db178e29508e3c0bcb5c1e908；FlagGems 484413ea0db18221dd9e63e742361d837df9ff13。EXP-0010 插件从失败候选 ef63a19a8309c3b3ba686802220883ac2f8836ab 修复。

用户确认今晚 00:00 关机。23:35 计算截止，23:50 前归档。正式测评和诊断严格串行；中断轮次不拼接。
