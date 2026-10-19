# LeanScale · MiniCPM 推理系统优化

FlagOS 开放计算全球挑战赛 S2，赛题 2。记录 MiniCPM5-2B 在天数 BI-V150 上的环境、代码改动、实验步骤和结果，包括未成功的尝试。

## 当前状态

| 实验 | 内容 | 状态 |
| --- | --- | --- |
| [EXP-0001](experiments/EXP-0001-environment/README.md) | 比赛分支、官方标签及源码导入修正 | 完成，镜像原有依赖约束报告已记录 |
| [EXP-0002](experiments/EXP-0002-rope/README.md) | RoPE 按 head 分组并行原型 | 独立测试完成；未接入默认整模路径 |
| [EXP-0003](experiments/EXP-0003-baseline/README.md) | 官方 4K / 16K 完整基线 | 以 [自动采集状态](experiments/EXP-0003-baseline/STATUS.md) 为准 |

RoPE 在 64-token 批次的独立内核测试约为 2.28× 加速。**这不是整模吞吐提升，也不是可提交比赛成绩。** 当前尚未完成优化后的端到端 A/B 和 math_500 Level 3 精度验证。

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
