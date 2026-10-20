# EXP-0005 算子诊断与最终配置

这些数据是 BI-V150 上的算子诊断，不是官方 HTTP 整模成绩。完整数据和失败记录保存在 artifacts/diagnostics 下。

## 已完成检查

- 第一轮：40 项 FP16/BF16、分页、ragged/空 query、NHD/HND、非连续 Q、CUDA Graph 检查通过；12 场景 × 5 参数组合完成。
- 第二轮：5 个长 prefill / mixed 场景 × 4 参数组合完成，包含修改 Q 后的 Graph/eager 完全一致检查。
- 固定配置另有 20 项边界数值/Graph 检查通过；实际 backend forward、noncausal/cascade 回退检查通过，结果保存在 20261020_132456。
- 完整 MATH500 暂缓；模型精度与赛事合格性未验证。

## 固定配置

`segments=16, decode_tile=32, prefill_tile=32, prefill_rows=64`。服务期不 autotune。

decode 每个 program 处理同一 KV head 下的 8 个 query heads。prefill 每个 program 处理 64 行，即同一 KV head 下的 8 个 query token × 8 个 heads。基线是每块 2 个 query token，因此新配置增加 KV tile 的 query 复用；它不是增加 KV head 或压缩 token。

## 固定配置对应的计时

时间为 CUDA Graph 重放、event 计时的中位数；前后夹测基线均在原始 JSON 中。纯 decode 不执行 prefill kernel，因此第一轮相同 decode 参数的测量适用于最终配置。

| 场景 | 原实现 µs | 候选 µs | 局部加速比 |
| --- | ---: | ---: | ---: |
| decode_b1_k4096 | 71.25 | 48.50 | 1.469× |
| decode_b1_k16384 | 262.28 | 178.05 | 1.473× |
| decode_b8_k4096 | 214.76 | 204.94 | 1.048× |
| decode_b8_k16384 | 742.99 | 754.17 | 0.985× |
| decode_b30_k4096 | 755.08 | 675.94 | 1.117× |
| decode_b30_k16384 | 2849.02 | 2611.39 | 1.091× |
| decode_b64_k4096 | 1584.32 | 1426.52 | 1.111× |
| decode_b64_k16384 | 5897.53 | 5407.35 | 1.091× |
| mixed_decode30_prefill2018_16k | 73263.23 | 22337.33 | 3.280× |
| mixed_prefill2018_decode30_16k | 73314.28 | 22295.32 | 3.288× |
| prefill2048_context16k | 70554.81 | 19487.24 | 3.621× |
| prefill2048_context4k | 14238.88 | 4037.09 | 3.527× |
| prefill2048_context2k | 4961.59 | 1457.61 | 3.404× |

B8/16K 的候选在该次诊断中慢约 1.5%，不能声称所有形状均加速。B8/4K 两次基线存在约 6% 漂移，微小收益应谨慎解释。官方基线 B64/4K 和容量受限的 B≈30/16K 是更相关的工作点，但最终仍以官方整模结果为准。

## 被淘汰的配置和失败

- 把 decode tile 增到 64/128 在较大 batch 下明显变慢。例如 B64/16K 的 tile128 约 25.97 ms，原实现约 5.90 ms，故不采用。
- 只把 prefill tile 增到 64、而不增加 query 复用，收益不足；第二轮 32-token KV tile、64 query-head rows 在所有 5 个场景中优于其余复核组合。
- 第一轮 20261020_130842 因与现有 baseline 的 mixed prefill 数值不一致停止，未进入计时。保留原脚本、日志和 FP32 复核。
- 随后独立 FP32 检查表明原实现的短 prefill 输出存在明显偏差。40 项中候选 BF16 最大绝对误差为 0.00813675，FP16 为 0.00103974；最大绝对误差不是通过判据的全部，使用固定 atol=0.003、rtol=0.03 逐元素判定。
- 原实现偏差的具体根因尚未定位；新内核按独立因果 attention 定义验证，不复制已观察到的偏差。完整模型任务精度仍待测，不将算子检查当作 MATH500 成绩。

## 复现

在对应两个代码分支下设置 PYTHONPATH 为插件目录及 FlagGems/src，设置 VLLM_PLUGINS=fl；GPU 工作必须持有共用 runner.lock。

```bash
python3 experiments/EXP-0005/validate_gqa.py --output diagnostic.json
python3 experiments/EXP-0005/refine_gqa.py diagnostic.json refine.json
python3 experiments/EXP-0005/validate_gqa.py --output selected.json --correctness-only --selected-only
python3 experiments/EXP-0005/validate_integration.py integration.json
```

正式测评使用 run_optimized_e2e.py；它校验固定提交、诊断哈希、代码哈希、官方 benchmark 哈希和安装环境，在实际模型编译/Graph、42 层路由和 smoke 成功后才启动官方两场景。

## 无参考内核的常量 V 复核

固定配置接入检查中，设置 K=0、V=1，任意有效因果 attention 输出应为 1。5 个场景中新内核最大绝对误差均为 0；原实现短 prefill 场景偏差为 0.015625、0.0078125，新建 2048-token prefill 的最大偏差为 0.5；16K 长上下文两个场景为 0。这是独立不变量检查，具体根因仍需单独定位，不能据此直接推断 MATH500 分数。

## 最终提交的地址宽化复核

将物理页号转为 int64 后再乘 stride，避免大/交错 KV 布局的 int32 偏移溢出。专项测试访问元素偏移 2,147,491,840，独立 FP32 及 Graph 检查通过。此前 40+20 项检查属于地址宽化前版本；以下三组是最终源码的复核。

| 场景 | 原实现 µs | 最终候选 µs | 局部加速比 |
| --- | ---: | ---: | ---: |
| decode_b30_k16384 | 2852.98 | 2593.33 | 1.100× |
| decode_b64_k4096 | 1580.07 | 1422.56 | 1.111× |
| mixed_decode30_prefill2018_16k | 73291.86 | 21974.20 | 3.335× |
