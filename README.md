# LeanScale · MiniCPM 推理系统优化

FlagOS 开放计算全球挑战赛 S2，赛题 2。记录 MiniCPM5-2B 在天数 BI-V150 上的环境、代码改动、实验步骤和结果，包括未成功的尝试。

## 当前状态

| 实验 | 内容 | 状态 |
| --- | --- | --- |
| [EXP-0001](experiments/EXP-0001-environment/README.md) | 比赛分支、官方标签及源码导入修正 | 完成，镜像原有依赖约束报告已记录 |
| [EXP-0002](experiments/EXP-0002-rope/README.md) | RoPE 按 head 分组并行原型 | 独立测试完成；未接入默认整模路径 |
| [EXP-0004](experiments/EXP-0004/README.md) | RoPE 实际编译路径接入与官方端到端 A/B | 旧候选中断；2026-10-20 已安排新会话完整续跑；[归档状态](experiments/EXP-0004/STATUS.md)；精度暂缓 |
| [EXP-0003](experiments/EXP-0003-baseline/README.md) | 官方 4K / 16K 基线（均完整完成） | 4K 原始会话 + EXP-0004 恢复的 16K 会话；[核验结果](docs/baseline_fusion_gqa_analysis.md) |

RoPE 在 64-token 批次的独立内核测试约为 2.28× 加速。**这不是整模吞吐提升，也不是可提交比赛成绩。** 当前尚未完成优化后的官方端到端 A/B；math_500 Level 3 完整精度按用户要求暂缓。已完成基础接入、编译和 Graph 检查。

## 三仓职责

| 仓库 | 内容 | 分支 |
| --- | --- | --- |
| [vllm-plugin-FL](https://github.com/yyq0210/vllm-plugin-FL) | 推理框架代码 | `leanscale/flagos-s2-baseline`，固定 `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0` |
| [FlagGems](https://github.com/yyq0210/FlagGems) | 算子代码 | `leanscale/iluvatar-rope-v1`，基于官方 `v5.3.5` |
| [MiniCPM-Infer-Sys-Opt](https://github.com/yyq0210/MiniCPM-Infer-Sys-Opt) | 实验、日志、文档及复现脚本 | `main` |

以上为目标仓库与分支；发布是否成功以 [发布记录](docs/publication.md) 为准。

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

## 2026-10-20 结果与分析

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

[EXP-0005 已通过实际模型接入并开始官方测评](experiments/EXP-0005/LAUNCH.md)。
