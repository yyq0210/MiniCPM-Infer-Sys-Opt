# LeanScale · MiniCPM 推理系统优化

MiniCPM5-2B 在天数 BI-V150 上的优化实验已收尾。最终测评完成于2026-11-02，独立审核与技术报告完成于2026-11-03；按用户要求不再启动新的优化实验。

**[最终优化技术报告](docs/final_optimization_report_20261103.md)** · [最终配置与决策](experiments/EXP-0030/final_configuration.json) · [16K独立审核](experiments/EXP-0030/review_20261102/audit_final.json)

| 官方负载 | 原始基线 Total tok/s | 最终配置 Total tok/s | 跨阶段累计提升 |
|---|---:|---:|---:|
| 4K | 2008.97 | 3687.315 | +83.543% |
| 16K | 917.27 | 2557.84 | +178.854% |

累计提升涉及不同日期；最终top-p增量另有同期对照：4K预登记ABBA +4.185%，16K顺序A/B +0.606%且非劣化门槛通过。每会话均官方四轮、只弃首；4K最终值是两个开启会话的官方均值等权平均。不能把顺序小差额当作统计显著。

最终采用RoPE+KV写入融合、分页GQA decode八行布局/页号广播、prefill调优/页号广播和top-p行并行；原调度与MLP布局保持。KV压缩未部署。完整模型任务精度按用户要求暂缓，不能据此声明赛事完整精度达标。

## 交付代码

| 个人仓库 | 分支 | 测评SHA |
|---|---|---|
| [vllm-plugin-FL](https://github.com/yyq0210/vllm-plugin-FL) | leanscale/iluvatar-topp-shadow-coverage | c50b5f0526e8112c72f1de918487300cf7376378 |
| [FlagGems](https://github.com/yyq0210/FlagGems) | leanscale/iluvatar-gqa-prefill-memory | de958ddfc4deeb43d6c94bb8652a3d4d3bc905fe |
| [MiniCPM-Infer-Sys-Opt](https://github.com/yyq0210/MiniCPM-Infer-Sys-Opt) | main | 实验、原始归档、独立审核和报告 |

两个代码分支远端提交已核验。top-p源码默认关闭；最终运行配置明确设置VLLM_FL_TOPP_ROW_PARALLEL=1、shadow=0、MLP布局改动=0，详见报告复现命令。没有创建上游赛事PR，也没有在收尾额外启动服务。

## 阅读入口

- [最终4K ABBA结果与限制](docs/topp_row_parallel_abba_20261030.md)
- [最终16K结果](experiments/EXP-0030/RESULTS.md)
- [真实推理流水线](docs/minicpm_real_inference_pipeline.md)
- [基线、融合边界与GQA实现](docs/baseline_fusion_gqa_analysis.md)
- [热点算子与逐项实验记录](docs/iluvatar_operator_optimization_backlog.md)
- [4K分阶段profiling](docs/operator_profile_4k_20261023.md)
- [16K调度与KV压缩分析](docs/16k_scheduling_kv_compression_analysis.md)
- [优化总账](docs/optimization_journal.md)
- [比赛要求与PR规范](docs/competition.md)
- [收尾前README历史快照](README.history_20261103.md)

所有失败与原始轮次保留。测评脚本中的旧截止日期是历史运行保护，不应覆盖后重跑为同一实验。后续复现若另行安排，应使用新的目录和时间窗口。
