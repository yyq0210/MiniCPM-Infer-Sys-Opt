# EXP-0004 官方性能实验结果

归档时间：2026-10-20T12:42:06+08:00。

- 运行阶段：failed。
- 完整性检查：尚未通过/未完成。
- 官方负载：[4096,1024,64,256] 与 [16384,1024,64,128]；每组四轮，舍弃第一轮。
- 基线 4K 使用中止前已完整完成的四轮；16K 使用恢复后的四轮，两个服务会话的来源均保留。
- 完整 MATH500 精度按用户要求暂缓，不声明赛事精度达标。
- 原始结果、日志、命令及哈希：[本次归档](artifacts/runs/20261019_202342/)。

未得到可报告的完整官方 A/B。保留失败现场，不把不完整数据写成成功。

状态/错误：Coordinator, serving process and completion handler no longer exist; container PID 1 restarted at 2026-10-20 12:20:34 +08:00; logs stop at 00:00:55. Exact interruption cause unknown. Baseline complete; candidate only 3 of 4 4K rounds.

## 已完成基线与中断核查

4K 与 16K 基线各四轮完整，后三轮 total tok/s 为 2008.97 和 917.27；16K Mean TTFT 597805.89 ms。候选只有三轮 4K，无法形成官方汇总。详见[基线、融合与 GQA 文档](../../docs/baseline_fusion_gqa_analysis.md)和[16K 系统分析](../../docs/16k_scheduling_kv_compression_analysis.md)。
