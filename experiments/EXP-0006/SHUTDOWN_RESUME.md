# 停机与接续说明

用户计划北京时间 2026-10-20 18:00 关闭机器，18:30 重启。17:50 前结束 GPU 工作，留十分钟归档推送。

本次先跑完整 4K 组，16K 在重启后另启服务跑完整四轮。所有负载参数、原 benchmark 脚本和 serve 参数保持不变；每组舍弃首轮。两个服务会话必须分别记录，整合前校验相同代码提交与配置。部分轮次只能作为诊断；重启后重新跑该组四轮，不能和前半段拼接。

启动脚本会拒绝停机窗口内运行、拒绝重复协调进程和未完成发布；当前 deadline 会在启动等待或 benchmark 超时后停止子进程及服务，结果先落盘再尝试推送。重启后先确认 GPU、模型文件、仓库和提交仍在，检查最新 receipt 与发布状态。

重启后先读取 `/workspace/LeanScale/artifacts/EXP-0006-latest-4k.json`；只有其运行 `status.json` 为 `performance_case_complete` 且归档 `completion_validation.json` 通过，才保留该组。

在 18:30 之后确认服务恢复后运行：

```bash
python3 /workspace/LeanScale/MiniCPM-Infer-Sys-Opt/experiments/EXP-0006/launch_split.py --case 16k
```

若 4K 未完成则先将 case 改成 `4k`；当前 launcher 在停机前还要求 `--finish-before 2026-10-20T17:50:00+08:00`。每个运行的独立 finisher 会保存 CSV、日志、命令、哈希、接入证据并推送。4K/16K 分别写入 RESULTS_4096.md / RESULTS_16384.md；两组完成后再汇总官方整体结果。不要把单组完成、接入失败或局部微基准称为整个实验完成。

当前冻结提交：`{'FlagGems': '484413ea0db18221dd9e63e742361d837df9ff13', 'vllm-plugin-FL': '07d9fa0cfdc5f09db178e29508e3c0bcb5c1e908'}`。
