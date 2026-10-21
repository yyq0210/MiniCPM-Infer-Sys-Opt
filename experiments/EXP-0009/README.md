# EXP-0009：4K profiling 修复与补采集

日期：2026-10-21，队名 LeanScale。

假设：CoreX trace 导出器不转义 LS_META 中 JSON 双引号导致文件无效；改用 base64 标注并在真实设备上导出验证。NONE 模式必须先发送真实 smoke 请求，再检查 42 层 GQA 路由。保留 EXP-0008 原始失败文件，仅对已验证特定标注生成派生修复副本。CUDA Graph 内核可见性单独报告，不把 NONE 诊断当官方吞吐成绩。

固定内核基线：插件 EXP-0006 07d9fa0cfdc5f09db178e29508e3c0bcb5c1e908；FlagGems 484413ea0db18221dd9e63e742361d837df9ff13。EXP-0010 插件从失败候选 ef63a19a8309c3b3ba686802220883ac2f8836ab 修复。

用户确认今晚 00:00 关机。23:35 计算截止，23:50 前归档。正式测评和诊断严格串行；中断轮次不拼接。
