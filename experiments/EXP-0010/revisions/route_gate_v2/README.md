# EXP-0010 测评校验修复 v2

2026-10-21 14:05，真实 smoke 已成功输出32 tokens，但正式benchmark尚未启动。
原检查要求配置阶段的 Selecting LeanScale 提示，该行未出现在日志中；实际EngineCore
已明确记录 KVReservationAsyncScheduler、enabled=True 和 admitted=1 的运行计数。
假设：配置意图日志不是可靠的运行接入证据；改为同时要求实际类、enabled与准入活动。
保留42层GQA、42层融合、生成图无旧KV更新和35个decode图检查。
正式benchmark结束再要求 checked>0，确认非首请求的容量判断被实际执行。
此修改仅涉及实验校验器，插件4534fcd和FlagGems484413ea保持不变。
旧失败目录、旧校验源码、队列状态与归档不覆盖；新测试从完整四轮开始。
