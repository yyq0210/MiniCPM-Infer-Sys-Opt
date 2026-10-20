# 2026-10-20 分析证据快照

本目录为只读核查与文档证据，不是新 GPU 实验。基线 4K 与 16K 各四轮完整；候选只有三轮 4K，整个 A/B 未完成。

- `results/status.json` 和 `results/completion_handler_status.json` 保留中断前原始状态。它们不是当前进程存活证明；当前判定见 `interruption_check.json` 和 EXP-0004/RESULTS.md。
- `results/` 原 CSV 不做修改；基线来自两个服务会话。
- `runtime_log_evidence.json` 是完整服务器日志中的固定规则选段，保留每条原时间与数值。
- `analysis_statistics.json` 可从仓库根目录运行 `python3 scripts/analyze_baseline_evidence.py experiments/EXP-0004/artifacts/analysis_20261020` 重算。
- `source_index.json` 是安装源码及插件源码哈希；`source_excerpts.md` 给出重点函数原行号，未改动推理源码。
- 未运行新精度测试，未生成新的官方候选汇总。
