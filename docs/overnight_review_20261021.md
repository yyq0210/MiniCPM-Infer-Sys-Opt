# 2026-10-21：昨晚实验结果与待恢复事项

本次在北京时间 10:00 后核验 GitHub 已上传的文件。服务器 SSH 返回 `Permission denied (publickey)`，没有重试或启动实验。自动接续已暂停，等待用户确认机器恢复后再更新运行窗口。

已核验远端 `main`：`b57268fe8951414e69b22c4c1e3b1d8560fd56c5`。昨晚 23:50 的[检查点](https://github.com/yyq0210/MiniCPM-Infer-Sys-Opt/blob/b57268fe8951414e69b22c4c1e3b1d8560fd56c5/docs/evidence/night_20261020/final/checkpoint.json)已在远端，记录当时没有仍在运行的实验控制器。以下结论来自实际保存的结果，不表示昨晚全部计划均已成功。

## 已完成：EXP-0006 融合版本的两组官方测评

两组均为完整四轮、弃首轮，推理与算子版本一致：

- vllm-plugin-FL：`07d9fa0cfdc5f09db178e29508e3c0bcb5c1e908`。
- FlagGems：`484413ea0db18221dd9e63e742361d837df9ff13`。

| 输入长度 | 原始基线 total tok/s | 融合版本 total tok/s | 相对原始基线 | 相对 EXP-0005 | 融合版本 TTFT |
|---|---:|---:|---:|---:|---:|
| 4K | 2008.97 | 2686.33 | +33.72% | +8.93% | 5920.51 ms |
| 16K | 917.27 | 1911.35 | +108.37% | +0.60% | 267408.36 ms |

来源：[4K 结果](https://github.com/yyq0210/MiniCPM-Infer-Sys-Opt/blob/b57268fe8951414e69b22c4c1e3b1d8560fd56c5/experiments/EXP-0006/RESULTS_4096.md)、[16K 结果](https://github.com/yyq0210/MiniCPM-Infer-Sys-Opt/blob/b57268fe8951414e69b22c4c1e3b1d8560fd56c5/experiments/EXP-0006/RESULTS_16384.md)。

16K 于 20:26:09 完成，保留三轮为 1912.89、1911.62、1909.54 tok/s，CV 为 0.0885%。相对原始基线的提升包含前面的 GQA/RoPE 优化；RoPE+KV 融合相对 EXP-0005 的增量只有约 0.60%。各版本分属不同服务会话，不把该差异表述为严格因果估计。4K 的 EXP-0005 CV 为 7.73%，融合版本 CV 为 3.64%，仍需注意波动。完整 MATH500 精度评估按用户要求暂缓。

## 部分完成：EXP-0008 的 4K profiling

`FULL_DECODE_ONLY` pass 完成了 256 个请求、1,048,576 个输入 token、262,144 个输出 token，并保存 initial、steady、late 三个压缩 trace。原始文件及哈希已列入夜间检查点；带 profiler 的负载结果不作为官方吞吐成绩。

但目前**没有可靠的完整算子耗时排名**，原因包括：

1. 设备能力探针捕获到 eager GEMM 的 3 个 GPU kernel 事件，但 CUDA Graph replay 内的设备事件为 **0**。因此不能把生产 Graph trace 中可见的外围 kernel 当成整个模型的耗时覆盖。
2. 辅助 `NONE` pass 在发送真实请求前检查了运行时 GQA 层数，得到 `gqa_layers=0`、`fused_runtime_layers=42`，因此被断言停止。这个启动前检查时机需要修正；不能直接把“尚未执行真实请求”判定为运行时 GQA 接入失败，也不能跳过真实请求后的接入校验。
3. 汇总脚本读取已有 trace 时在第 822 行遇到 `JSONDecodeError`，统计未完成。原始 gzip 与失败记录已保留。需要取得原始文件定位错误后再处理；当前不能声称知道损坏的确切内容。一个待验证的原因是诊断命名范围内嵌 JSON 的引号与该运行时的 trace 导出兼容性，不应在未检查原文件时直接认定。

来源：[能力探针](https://github.com/yyq0210/MiniCPM-Infer-Sys-Opt/blob/b57268fe8951414e69b22c4c1e3b1d8560fd56c5/experiments/EXP-0008/artifacts/20261020_195936_profile_4k/probe_result.json)、[负载完成检查](https://github.com/yyq0210/MiniCPM-Infer-Sys-Opt/blob/b57268fe8951414e69b22c4c1e3b1d8560fd56c5/experiments/EXP-0008/artifacts/20261020_195936_profile_4k/FULL_DECODE_ONLY/workload_validation.json)、[失败堆栈](https://github.com/yyq0210/MiniCPM-Infer-Sys-Opt/blob/b57268fe8951414e69b22c4c1e3b1d8560fd56c5/experiments/EXP-0008/artifacts/20261020_195936_profile_4k/error.txt)。

## 未完成：EXP-0007 调度版本的真实服务验证

调度版本在 20:51:10 的 smoke request 中失败，尚未进入正式性能测试：

```text
TypeError: KVReservationAsyncScheduler.schedule() takes 1 positional argument but 2 were given
```

实际引擎调用 `scheduler.schedule(self._should_throttle_prefills())`，本次编写的包装只定义了 `schedule(self)`，遗漏了调度接口参数。保存的基类源码接口为 `schedule(self, throttle_prefills: bool = False)`。修复时必须在优化开启、关闭及回退路径中都正确转发该参数，保留原生节流语义。

这也是验证覆盖不足：此前的 CPU 测试检查了分配器及请求生命周期，没有覆盖 EngineCore 的实际调用契约。后续需要补充真实参数调用、异常后原分配器恢复和真实服务 smoke，不能仅凭现有 CPU 检查声称已接通引擎。

来源：[服务器错误日志](https://github.com/yyq0210/MiniCPM-Infer-Sys-Opt/blob/b57268fe8951414e69b22c4c1e3b1d8560fd56c5/experiments/EXP-0007/artifacts/runs/20261020_204644_official_16k/candidate_benchmark/server.log)。本机下载后的日志 SHA-256 与已上传 manifest 一致：`64e6ca6835cac39ce5725814df289512dd2a8a2a76c9c1fd9a96a8366bb22de7`。

没有有效调度性能成绩；4K 调度回归测评和 KV 压缩尚未接续。

## 恢复后的顺序

1. 用户确认机器恢复后，验证连接、GPU 状态和残留进程，更新已过期的运行窗口。
2. 优先取得并保留 4K 原始 trace，定位导出/解析问题，修正辅助模式校验时机。验证实际设备事件覆盖，再生成分阶段耗时表；Graph 模式与辅助模式分别报告。
3. 修复调度接口契约并补 CPU 与真实服务检查，冻结新版本后按官方配置重新完整测评，保留此次失败。
4. 完成调度收益判断后继续 KV 压缩；压缩必须通过完整缓存对照、官方精度和确实触发压缩的长文质量验证。

本核验文档暂存本机；原始实验结果、失败记录和昨晚检查点均已在 GitHub。恢复访问后将本核验文档同步到实验仓库。
