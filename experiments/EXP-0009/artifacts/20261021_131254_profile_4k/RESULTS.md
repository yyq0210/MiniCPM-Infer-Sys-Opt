# EXP-0009：4K 算子 profiling 结果

输入 4096 / 输出 1024 / 并发上限 64 / 请求 256；每个 pass 一次完整负载。
这是带 profiler 的诊断实验，不是官方吞吐成绩；各窗口与 Graph 模式单独汇总。
kernel 总时间是设备工作量之和，可能跨 stream 重叠，不等于请求延迟。
CPU inclusive 时间不可相加。未关联 kernel 保留 unlinked，不按相近时间戳猜测阶段。

设备采集能力检查：`{"safe_metadata_real_export_passed": true, "torch": "2.10.0", "device_kernel_events": 3, "device_kernel_us": 22.54, "device_kernel_names": ["void Gemm_tcu_mr_kernel::gemm_tcu_h<64u, 64u, 64u, 16u, 16u, 2u, (ixblasTrans_t)1, (ixblasTrans_t)1, (ixblasTCUGEMMCategory_t)0, (ixmma::MMEdgeCheckMode)0, (ixblasGEMMBoundary_t)2, (ixblasLtEpilogue_t)1, false, float, (ixblasLtMatmulMatrixScale_t)1002, (ixblasLtMatmulMatrixScale_t)1002, (ixblasLtMatmulMatrixScale_t)1002, false, false, 4u, 4u, false, float, float, float, float, float>(float*, float*, float const*, float const*, int, int, int, int, int, int, int, int, float, float, long, long, long, long, long, long, ixblasGEMMBatchMode_t, int, void*, void*, void*, unsigned int)"], "graph_device_kernel_events": 0}`。

## FULL_DECODE_ONLY / initial / trace 0

设备事件 8188；kernel 总计 4139.824 ms；设备区间并集 4139.824 ms。
关联情况：`{"linked_to_step": 8188}`；实际阶段：`{"prefill": 2, "mixed": 14}`。

观察到模型 RoPE+KV kernel：`True`。若为 false，本窗口不得用于声称完整模型算子覆盖。

| 阶段 | 算子/候选归属 | 调用数 | GPU 总耗时 ms | kernel 时间占比 |
|---|---|---:|---:|---:|
| mixed | OP05_06_GQA | 1176 | 1833.394 | 44.29% |
| mixed | OP09_gate_up | 588 | 824.111 | 19.91% |
| mixed | OP11_down | 588 | 426.491 | 10.30% |
| prefill | OP05_06_GQA | 168 | 249.686 | 6.03% |
| mixed | OP03_QKV | 588 | 178.022 | 4.30% |
| mixed | OP08_O | 588 | 143.457 | 3.47% |
| prefill | OP09_gate_up | 84 | 117.790 | 2.85% |
| mixed | OP10_SwiGLU_or_compiler_fusion | 588 | 81.063 | 1.96% |
| mixed | OP02_norm_or_fused_norm | 1190 | 78.950 | 1.91% |
| prefill | OP11_down | 84 | 60.936 | 1.47% |
| mixed | unattributed | 636 | 41.008 | 0.99% |
| prefill | OP03_QKV | 84 | 25.409 | 0.61% |
| prefill | OP08_O | 84 | 20.618 | 0.50% |
| mixed | OP04_RoPE_KV | 588 | 19.886 | 0.48% |
| prefill | OP10_SwiGLU_or_compiler_fusion | 84 | 11.586 | 0.28% |
| prefill | OP02_norm_or_fused_norm | 170 | 11.346 | 0.27% |
| mixed | sampling_candidate | 42 | 4.546 | 0.11% |
| mixed | OP07_GQA_reduce | 588 | 4.332 | 0.10% |
| prefill | unattributed | 96 | 3.726 | 0.09% |
| prefill | OP04_RoPE_KV | 84 | 2.855 | 0.07% |

归属方法、均值、中位数、P95 和原始 kernel 名见对应 CSV。

## FULL_DECODE_ONLY / steady / trace 0

设备事件 7744；kernel 总计 4684.703 ms；设备区间并集 4684.703 ms。
关联情况：`{"linked_to_step": 7744}`；实际阶段：`{"mixed": 16}`。

观察到模型 RoPE+KV kernel：`True`。若为 false，本窗口不得用于声称完整模型算子覆盖。

| 阶段 | 算子/候选归属 | 调用数 | GPU 总耗时 ms | kernel 时间占比 |
|---|---|---:|---:|---:|
| mixed | OP05_06_GQA | 1344 | 2567.267 | 54.80% |
| mixed | OP09_gate_up | 672 | 941.074 | 20.09% |
| mixed | OP11_down | 672 | 487.745 | 10.41% |
| mixed | OP03_QKV | 672 | 203.752 | 4.35% |
| mixed | OP08_O | 672 | 163.836 | 3.50% |
| mixed | OP10_SwiGLU_or_compiler_fusion | 672 | 92.808 | 1.98% |
| mixed | OP02_norm_or_fused_norm | 1360 | 90.399 | 1.93% |
| mixed | sampling_candidate | 48 | 79.300 | 1.69% |
| mixed | unattributed | 288 | 25.583 | 0.55% |
| mixed | OP04_RoPE_KV | 672 | 22.794 | 0.49% |
| mixed | OP07_GQA_reduce | 672 | 10.145 | 0.22% |

归属方法、均值、中位数、P95 和原始 kernel 名见对应 CSV。

## FULL_DECODE_ONLY / late / trace 0

设备事件 336；kernel 总计 116.578 ms；设备区间并集 116.578 ms。
关联情况：`{"linked_to_step": 336}`；实际阶段：`{"decode": 16}`。

观察到模型 RoPE+KV kernel：`False`。若为 false，本窗口不得用于声称完整模型算子覆盖。

| 阶段 | 算子/候选归属 | 调用数 | GPU 总耗时 ms | kernel 时间占比 |
|---|---|---:|---:|---:|
| decode | sampling_candidate | 48 | 86.198 | 73.94% |
| decode | unattributed | 288 | 30.380 | 26.06% |

归属方法、均值、中位数、P95 和原始 kernel 名见对应 CSV。

## NONE / initial / trace 0

设备事件 8188；kernel 总计 4139.416 ms；设备区间并集 4139.416 ms。
关联情况：`{"linked_to_step": 8188}`；实际阶段：`{"prefill": 2, "mixed": 14}`。

观察到模型 RoPE+KV kernel：`True`。若为 false，本窗口不得用于声称完整模型算子覆盖。

| 阶段 | 算子/候选归属 | 调用数 | GPU 总耗时 ms | kernel 时间占比 |
|---|---|---:|---:|---:|
| mixed | OP05_06_GQA | 1176 | 1833.928 | 44.30% |
| mixed | OP09_gate_up | 588 | 823.259 | 19.89% |
| mixed | OP11_down | 588 | 426.684 | 10.31% |
| prefill | OP05_06_GQA | 168 | 249.730 | 6.03% |
| mixed | OP03_QKV | 588 | 177.969 | 4.30% |
| mixed | OP08_O | 588 | 143.464 | 3.47% |
| prefill | OP09_gate_up | 84 | 117.768 | 2.85% |
| mixed | OP10_SwiGLU_or_compiler_fusion | 588 | 81.074 | 1.96% |
| mixed | OP02_norm_or_fused_norm | 1190 | 78.761 | 1.90% |
| prefill | OP11_down | 84 | 60.981 | 1.47% |
| mixed | unattributed | 636 | 40.976 | 0.99% |
| prefill | OP03_QKV | 84 | 25.432 | 0.61% |
| prefill | OP08_O | 84 | 20.634 | 0.50% |
| mixed | OP04_RoPE_KV | 588 | 19.929 | 0.48% |
| prefill | OP10_SwiGLU_or_compiler_fusion | 84 | 11.581 | 0.28% |
| prefill | OP02_norm_or_fused_norm | 170 | 11.277 | 0.27% |
| mixed | sampling_candidate | 42 | 4.469 | 0.11% |
| mixed | OP07_GQA_reduce | 588 | 4.324 | 0.10% |
| prefill | unattributed | 96 | 3.713 | 0.09% |
| prefill | OP04_RoPE_KV | 84 | 2.856 | 0.07% |

归属方法、均值、中位数、P95 和原始 kernel 名见对应 CSV。

## NONE / steady / trace 0

设备事件 7744；kernel 总计 4664.735 ms；设备区间并集 4664.735 ms。
关联情况：`{"linked_to_step": 7744}`；实际阶段：`{"mixed": 16}`。

观察到模型 RoPE+KV kernel：`True`。若为 false，本窗口不得用于声称完整模型算子覆盖。

| 阶段 | 算子/候选归属 | 调用数 | GPU 总耗时 ms | kernel 时间占比 |
|---|---|---:|---:|---:|
| mixed | OP05_06_GQA | 1344 | 2548.521 | 54.63% |
| mixed | OP09_gate_up | 672 | 940.570 | 20.16% |
| mixed | OP11_down | 672 | 488.258 | 10.47% |
| mixed | OP03_QKV | 672 | 203.766 | 4.37% |
| mixed | OP08_O | 672 | 163.962 | 3.51% |
| mixed | OP10_SwiGLU_or_compiler_fusion | 672 | 92.858 | 1.99% |
| mixed | OP02_norm_or_fused_norm | 1360 | 90.211 | 1.93% |
| mixed | sampling_candidate | 48 | 78.714 | 1.69% |
| mixed | unattributed | 288 | 25.198 | 0.54% |
| mixed | OP04_RoPE_KV | 672 | 22.942 | 0.49% |
| mixed | OP07_GQA_reduce | 672 | 9.736 | 0.21% |

归属方法、均值、中位数、P95 和原始 kernel 名见对应 CSV。

## NONE / late / trace 0

设备事件 7072；kernel 总计 1434.369 ms；设备区间并集 1434.369 ms。
关联情况：`{"linked_to_step": 7072}`；实际阶段：`{"decode": 16}`。

观察到模型 RoPE+KV kernel：`True`。若为 false，本窗口不得用于声称完整模型算子覆盖。

| 阶段 | 算子/候选归属 | 调用数 | GPU 总耗时 ms | kernel 时间占比 |
|---|---|---:|---:|---:|
| decode | OP05_06_GQA | 672 | 1123.689 | 78.34% |
| decode | sampling_candidate | 48 | 95.904 | 6.69% |
| decode | OP09_gate_up | 672 | 63.766 | 4.45% |
| decode | OP11_down | 672 | 40.116 | 2.80% |
| decode | unattributed | 288 | 30.526 | 2.13% |
| decode | OP03_QKV | 672 | 24.711 | 1.72% |
| decode | OP08_O | 672 | 19.387 | 1.35% |
| decode | OP07_GQA_reduce | 672 | 13.705 | 0.96% |
| decode | OP02_norm_or_fused_norm | 1360 | 11.447 | 0.80% |
| decode | OP04_RoPE_KV | 672 | 5.789 | 0.40% |
| decode | OP10_SwiGLU_or_compiler_fusion | 672 | 5.329 | 0.37% |

归属方法、均值、中位数、P95 和原始 kernel 名见对应 CSV。

