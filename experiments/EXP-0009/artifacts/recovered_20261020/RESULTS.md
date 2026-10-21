# EXP-0009：4K 算子 profiling 结果

输入 4096 / 输出 1024 / 并发上限 64 / 请求 256；每个 pass 一次完整负载。
这是带 profiler 的诊断实验，不是官方吞吐成绩；各窗口与 Graph 模式单独汇总。
kernel 总时间是设备工作量之和，可能跨 stream 重叠，不等于请求延迟。
CPU inclusive 时间不可相加。未关联 kernel 保留 unlinked，不按相近时间戳猜测阶段。

设备采集能力检查：`{"torch": "2.10.0", "device_kernel_events": 3, "device_kernel_us": 23.548, "device_kernel_names": ["void Gemm_tcu_mr_kernel::gemm_tcu_h<64u, 64u, 64u, 16u, 16u, 2u, (ixblasTrans_t)1, (ixblasTrans_t)1, (ixblasTCUGEMMCategory_t)0, (ixmma::MMEdgeCheckMode)0, (ixblasGEMMBoundary_t)2, (ixblasLtEpilogue_t)1, false, float, (ixblasLtMatmulMatrixScale_t)1002, (ixblasLtMatmulMatrixScale_t)1002, (ixblasLtMatmulMatrixScale_t)1002, false, false, 4u, 4u, false, float, float, float, float, float>(float*, float*, float const*, float const*, int, int, int, int, int, int, int, int, float, float, long, long, long, long, long, long, ixblasGEMMBatchMode_t, int, void*, void*, void*, unsigned int)"], "graph_device_kernel_events": 0}`。

## FULL_DECODE_ONLY / initial / trace 0

设备事件 8188；kernel 总计 4136.722 ms；设备区间并集 4136.722 ms。
关联情况：`{"linked_to_step": 8188}`；实际阶段：`{"prefill": 2, "mixed": 14}`。

观察到模型 RoPE+KV kernel：`True`。若为 false，本窗口不得用于声称完整模型算子覆盖。

| 阶段 | 算子/候选归属 | 调用数 | GPU 总耗时 ms | kernel 时间占比 |
|---|---|---:|---:|---:|
| mixed | OP05_06_GQA | 1176 | 1833.512 | 44.32% |
| mixed | OP09_gate_up | 588 | 824.308 | 19.93% |
| mixed | OP11_down | 588 | 426.506 | 10.31% |
| prefill | OP05_06_GQA | 168 | 249.097 | 6.02% |
| mixed | OP03_QKV | 588 | 178.225 | 4.31% |
| mixed | OP08_O | 588 | 143.033 | 3.46% |
| prefill | OP09_gate_up | 84 | 117.505 | 2.84% |
| mixed | OP10_SwiGLU_or_compiler_fusion | 588 | 80.700 | 1.95% |
| mixed | OP02_norm_or_fused_norm | 1190 | 78.712 | 1.90% |
| prefill | OP11_down | 84 | 60.984 | 1.47% |
| mixed | unattributed | 636 | 40.706 | 0.98% |
| prefill | OP03_QKV | 84 | 25.425 | 0.61% |
| prefill | OP08_O | 84 | 20.581 | 0.50% |
| mixed | OP04_RoPE_KV | 588 | 19.611 | 0.47% |
| prefill | OP10_SwiGLU_or_compiler_fusion | 84 | 11.530 | 0.28% |
| prefill | OP02_norm_or_fused_norm | 170 | 11.317 | 0.27% |
| mixed | sampling_candidate | 42 | 4.309 | 0.10% |
| prefill | unattributed | 96 | 3.678 | 0.09% |
| mixed | OP07_GQA_reduce | 588 | 3.665 | 0.09% |
| prefill | OP04_RoPE_KV | 84 | 2.802 | 0.07% |

归属方法、均值、中位数、P95 和原始 kernel 名见对应 CSV。

## FULL_DECODE_ONLY / steady / trace 0

设备事件 7744；kernel 总计 4630.742 ms；设备区间并集 4630.742 ms。
关联情况：`{"linked_to_step": 7744}`；实际阶段：`{"mixed": 16}`。

观察到模型 RoPE+KV kernel：`True`。若为 false，本窗口不得用于声称完整模型算子覆盖。

| 阶段 | 算子/候选归属 | 调用数 | GPU 总耗时 ms | kernel 时间占比 |
|---|---|---:|---:|---:|
| mixed | OP05_06_GQA | 1344 | 2527.523 | 54.58% |
| mixed | OP09_gate_up | 672 | 941.272 | 20.33% |
| mixed | OP11_down | 672 | 488.219 | 10.54% |
| mixed | OP03_QKV | 672 | 203.925 | 4.40% |
| mixed | OP08_O | 672 | 162.957 | 3.52% |
| mixed | OP10_SwiGLU_or_compiler_fusion | 672 | 92.069 | 1.99% |
| mixed | OP02_norm_or_fused_norm | 1360 | 90.092 | 1.95% |
| mixed | sampling_candidate | 48 | 69.389 | 1.50% |
| mixed | unattributed | 288 | 24.153 | 0.52% |
| mixed | OP04_RoPE_KV | 672 | 22.492 | 0.49% |
| mixed | OP07_GQA_reduce | 672 | 8.649 | 0.19% |

归属方法、均值、中位数、P95 和原始 kernel 名见对应 CSV。

## FULL_DECODE_ONLY / late / trace 0

设备事件 336；kernel 总计 101.235 ms；设备区间并集 101.235 ms。
关联情况：`{"linked_to_step": 336}`；实际阶段：`{"decode": 16}`。

观察到模型 RoPE+KV kernel：`False`。若为 false，本窗口不得用于声称完整模型算子覆盖。

| 阶段 | 算子/候选归属 | 调用数 | GPU 总耗时 ms | kernel 时间占比 |
|---|---|---:|---:|---:|
| decode | sampling_candidate | 48 | 70.874 | 70.01% |
| decode | unattributed | 288 | 30.361 | 29.99% |

归属方法、均值、中位数、P95 和原始 kernel 名见对应 CSV。

