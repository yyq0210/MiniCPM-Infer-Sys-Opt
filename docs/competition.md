# 比赛要求和 PR 约定

来源：[赛题页面](https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn)、[官方 PR 示例](https://github.com/flagos-ai/vllm-plugin-FL/pull/463)。核对日期：2026-10-19。

## 固定条件

- MiniCPM5-2B；目标设备为官方天数 BI-V150。
- vllm-plugin-FL 的 `flagos-2026-s2` 分支；FlagGems 的官方 `v5.3.5` 标签。
- 保持官方 `vllm serve` 参数和 benchmark 脚本；禁止通过超参启用量化/投机采样、修改 benchmark、无算子优化地切换实现、删除主要算子选择逻辑或直接合并框架最新分支获取性能。
- 4K 场景 `[4096,1024,64,256]`，16K 场景 `[16384,1024,64,128]`。
- 主要指标为 total tokens/s，同时核对 TTFT；页面规定 1% 的波动/限制范围。
- math_500 Level 3 精度要求不低于 0.95。数学精度尚未完成时不宣称提交资格已满足。

## 固定启动命令

```bash
export VLLM_PLUGINS=fl
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072

python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

## PR 格式与范围

标题：`【FlagOS开放计算全球挑战赛S2.赛题2】【LeanScale】具体特性描述`。

推理仓库正文遵循其 `.github/PULL_REQUEST_TEMPLATE.md`：PR Category、PR Type、Description、Related Issues、Changes、Testing、Checklist。

算子仓库正文遵循其模板：PR Category、Type of Change、Description、Issue、Progress、Performance。附设计、复现命令、精度和适用范围，避免将微基准误写成整模提升。

本轮目标是向个人 fork 保存工作。原型 PR 的 base 为个人 fork 中从官方提交固定的基线分支；不向默认主分支混入比赛基线之后的上游更新。正式赛事上游 PR 应根据最终实现、完整性能结果和精度结果重新整理。
