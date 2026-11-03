# LeanScale · MiniCPM 推理系统优化

FlagOS 开放计算全球挑战赛 S2，赛题 2。记录 MiniCPM5-2B 在天数 BI-V150 上的环境、代码改动、实验步骤和结果，包括未成功的尝试。

## 当前状态

| 实验 | 内容 | 状态 |
| --- | --- | --- |
| [EXP-0001](experiments/EXP-0001-environment/README.md) | 比赛分支、官方标签及源码导入修正 | 完成，镜像原有依赖约束报告已记录 |
| [EXP-0002](experiments/EXP-0002-rope/README.md) | RoPE 按 head 分组并行原型 | 独立测试完成；未接入默认整模路径 |
| [EXP-0004](experiments/EXP-0004/README.md) | RoPE 单项接入与端到端尝试 | 按用户要求中止，未形成完整候选成绩；保留归档 |
| [EXP-0003](experiments/EXP-0003-baseline/README.md) | 官方 4K / 16K 基线（均完整完成） | 4K 原始会话 + EXP-0004 恢复的 16K 会话；[核验结果](docs/baseline_fusion_gqa_analysis.md) |
| [EXP-0005](experiments/EXP-0005/RESULTS.md) | RoPE＋GQA / mixed 优化 | 官方两组完整完成；4K/16K 吞吐较原始基线约 +22.75%/+107.13% |
| [EXP-0006](experiments/EXP-0006/README.md) | RoPE＋KV 写入融合 | 42 层接入通过；4K 完整完成，16K 正在测评；以各组结果完整性检查为准 |
| [EXP-0007](experiments/EXP-0007/README.md) | KV 增长预留调度 | 13 项 CPU 检查通过；已排队等待融合测评完成后串行验证 |

2026-10-20 晚状态：EXP-0006 4K 为 2686.33 total tok/s，较原始基线约 +33.72%。
组合收益不等于单算子贡献；跨会话比较需结合逐轮波动。完整 MATH500 精度按用户要求暂缓，不能据此声明赛事精度达标。

**后续逐项优化入口：[天数热点算子清单、源码位置、优先级与验收模板](docs/iluvatar_operator_optimization_backlog.md)。**
该清单区分已有实测证据与待 profiler 确认的热点候选。

## 三仓职责

| 仓库 | 内容 | 分支 |
| --- | --- | --- |
| [vllm-plugin-FL](https://github.com/yyq0210/vllm-plugin-FL) | 推理框架代码 | `leanscale/flagos-s2-baseline`，固定 `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0` |
| [FlagGems](https://github.com/yyq0210/FlagGems) | 算子代码 | `leanscale/iluvatar-rope-v1`，基于官方 `v5.3.5` |
| [MiniCPM-Infer-Sys-Opt](https://github.com/yyq0210/MiniCPM-Infer-Sys-Opt) | 实验、日志、文档及复现脚本 | `main` |

以上列出早期基线/原型分支；当前融合使用 `leanscale/iluvatar-fusion`，调度使用插件分支 `leanscale/iluvatar-kv-reservation`。最新 SHA 与验证结果见各实验的 `candidate_commits.json`；早期发布过程见 [发布记录](docs/publication.md)。

## 服务器目录

```text
/workspace/LeanScale/
├── vllm-plugin-FL/          # 独立的推理代码工作区
├── FlagGems/                # 独立的算子代码工作区
├── MiniCPM-Infer-Sys-Opt/   # 本仓库
└── artifacts/              # Git bundle 等交付文件，不纳入本仓库

/workspace/experiments/codex_20261019/  # 本轮原始运行文件，保留原位
```

原始 `/workspace/vllm-plugin-FL` 和 `/workspace/FlagGems` 是本轮基线实际加载的工作区。独立优化工作区的修改不会自动作用于基线服务。

## 记录与复现

- [优化步骤总账](docs/optimization_journal.md)
- [实验记录规范](docs/recording.md)
- [环境与已知依赖情况](experiments/EXP-0001-environment/README.md)
- [比赛要求与 PR 规范](docs/competition.md)
- [算子 PR 正文](pr/flaggems-rope-v1.md)
- [推理代码交付状态](pr/vllm-plugin-fl.md)

新实验应先创建编号和记录，再修改代码与运行。每次失败也要归档命令和日志，不以新结果覆盖旧结果。完整 GPU 实验串行执行。

```bash
python scripts/new_experiment.py --title "实验标题" --hypothesis "预期改善的实际执行路径"
python scripts/collect_baseline.py
```

只采集已有结果的命令不运行 GPU 任务。单算子复现命令和官方 benchmark 命令分别见对应实验目录。

- [RoPE 优化前后的思路与接入说明](docs/rope_before_after.md)

- [模型真实推理流水线：请求、调度、42 层计算、KV cache 与编译路径](docs/minicpm_real_inference_pipeline.md)

- [RoPE 推理接入 PR 草稿](pr/vllm-rope-e2e.md)
- [RoPE 算子集成 PR 草稿](pr/flaggems-rope-e2e.md)

- [本次实验自动归档与推送](docs/automatic_result_archiving.md)

## 2026-10-20 早期基线与中断分析（12:42 快照）

基线官方后三轮均值：4K 2008.97 total tok/s、16K 917.27 total tok/s。16K Mean TTFT 为 597.81 s。RoPE 候选在第 4 轮 4K 中断，不能宣称端到端加速。执行环境已重启，已恢复本地归档；准确中断原因未知。

- [文档一：基线复现、真实函数实现、融合边界与 GQA 优化](docs/baseline_fusion_gqa_analysis.md)
- [文档二：16K 排队、调度优化与 H2O 等 KV 压缩适用性](docs/16k_scheduling_kv_compression_analysis.md)
- [原始 CSV、日志统计与源码摘录](experiments/EXP-0004/artifacts/analysis_20261020/)

### 2026-10-20 续跑

按用户要求在新服务会话重跑候选官方 4K/16K，各四轮排除首轮；复用已验证的完整基线。运行编号 20261020_125406。中断前数据继续保留，最终以 [EXP-0004/RESULTS.md](experiments/EXP-0004/RESULTS.md) 为准，完整任务精度仍暂缓。

本次候选性能测试已启动。服务器每分钟更新本地状态；GitHub 结果页面在本次结束后自动同步，期间可能仍显示上次中断快照。两份实际候选源码均已发布到个人 fork 的 leanscale/iluvatar-rope-e2e 分支，详见[发布记录](docs/publication.md)。

## EXP-0005：GQA 与混合批次优化

[方案与接入](experiments/EXP-0005/README.md) · [全部诊断与淘汰配置](experiments/EXP-0005/DIAGNOSTICS.md) · [当前状态](experiments/EXP-0005/STATUS.md) · [官方性能结果](experiments/EXP-0005/RESULTS.md)。
旧 RoPE 单项长测按用户要求主动中止；本轮先完成实现与验证，再统一测评。

[EXP-0005 接入及启动记录（性能现已完成，见上述结果页）](experiments/EXP-0005/LAUNCH.md)。
