# EXP-0018：选定版本 4K 分阶段热点诊断

每个 pass 使用 4096 / 1024 / 64 / 256 一次完整请求集；带 profiler，不能作为官方吞吐成绩。
窗口名仅表示触发目标，实际阶段和 batch size 来自逐步 metadata。缺失或不足 16 步的覆盖如实保留。
CPU context 长度是异步 CPU 上界。GPU kernel 总时间可能跨 stream 重叠，不等于墙钟延迟。
FULL Graph 内部可能不可见；NONE 仅辅助归属。不同窗口/模式不合并成整程占比。

| 模式 | 窗口 | trace 数 | 满足目标的步数 | 覆盖 |
|---|---|---:|---:|---|
| FULL_DECODE_ONLY | initial | 1 | 16 | at_least_16_steps |
| FULL_DECODE_ONLY | mixed | 1 | 16 | at_least_16_steps |
| FULL_DECODE_ONLY | large_decode | 1 | 16 | at_least_16_steps |
| FULL_DECODE_ONLY | tail | 1 | 16 | at_least_16_steps |
| NONE | initial | 1 | 16 | at_least_16_steps |
| NONE | mixed | 1 | 16 | at_least_16_steps |
| NONE | large_decode | 1 | 16 | at_least_16_steps |
| NONE | tail | 1 | 16 | at_least_16_steps |

## FULL_DECODE_ONLY / initial / 0

实际阶段 {'prefill': 2, 'mixed': 14}；batch sizes [1, 2, 3, 4, 5, 6, 7, 8]；满足目标 16/16 步。
GPU 事件 8188，kernel 总和 3897.269 ms，区间并集 3897.269 ms。
关联情况 {'linked_to_step': 8188}；可见 RoPEKV 模型 kernel：True。

| 实际阶段 | 算子 | 调用数 | GPU 总 ms | 此窗口可见 kernel 时间占比 |
|---|---|---:|---:|---:|
| mixed | OP05_GQA_prefill_branch | 588 | 1572.960 | 40.36% |
| mixed | OP09_gate_up | 588 | 824.400 | 21.15% |
| mixed | OP11_down | 588 | 426.155 | 10.93% |
| prefill | OP05_GQA_prefill_branch | 84 | 222.858 | 5.72% |
| mixed | OP03_QKV | 588 | 178.073 | 4.57% |
| mixed | OP08_O | 588 | 142.867 | 3.67% |
| prefill | OP09_gate_up | 84 | 117.850 | 3.02% |
| mixed | OP10_SwiGLU_or_compiler_fusion | 588 | 80.780 | 2.07% |
| mixed | OP02_norm_or_fused_norm | 1190 | 78.808 | 2.02% |
| prefill | OP11_down | 84 | 60.838 | 1.56% |
| mixed | OP06_GQA_decode_branch | 588 | 47.280 | 1.21% |
| mixed | OP13_sampling | 496 | 32.133 | 0.82% |
| prefill | OP03_QKV | 84 | 25.444 | 0.65% |
| prefill | OP08_O | 84 | 20.589 | 0.53% |
| mixed | OP04_RoPE_KV | 588 | 19.608 | 0.50% |
| mixed | OP12_LM_head | 14 | 11.964 | 0.31% |
| prefill | OP10_SwiGLU_or_compiler_fusion | 84 | 11.517 | 0.30% |
| prefill | OP02_norm_or_fused_norm | 170 | 11.344 | 0.29% |
| mixed | OP07_GQA_reduce | 588 | 3.686 | 0.09% |
| prefill | OP04_RoPE_KV | 84 | 2.801 | 0.07% |

## FULL_DECODE_ONLY / mixed / 0

实际阶段 {'mixed': 16}；batch sizes [10, 11, 12, 13, 14, 15, 16, 17]；满足目标 16/16 步。
GPU 事件 7744，kernel 总和 3982.715 ms，区间并集 3982.714 ms。
关联情况 {'linked_to_step': 7744}；可见 RoPEKV 模型 kernel：True。

| 实际阶段 | 算子 | 调用数 | GPU 总 ms | 此窗口可见 kernel 时间占比 |
|---|---|---:|---:|---:|
| mixed | OP05_GQA_prefill_branch | 672 | 1800.060 | 45.20% |
| mixed | OP09_gate_up | 672 | 940.369 | 23.61% |
| mixed | OP11_down | 672 | 487.046 | 12.23% |
| mixed | OP03_QKV | 672 | 203.690 | 5.11% |
| mixed | OP08_O | 672 | 162.643 | 4.08% |
| mixed | OP06_GQA_decode_branch | 672 | 129.234 | 3.24% |
| mixed | OP10_SwiGLU_or_compiler_fusion | 672 | 92.138 | 2.31% |
| mixed | OP02_norm_or_fused_norm | 1360 | 90.106 | 2.26% |
| mixed | OP13_sampling | 128 | 34.055 | 0.86% |
| mixed | OP04_RoPE_KV | 672 | 22.411 | 0.56% |
| mixed | OP12_LM_head | 16 | 14.885 | 0.37% |
| mixed | OP07_GQA_reduce | 672 | 5.205 | 0.13% |
| mixed | unattributed | 192 | 0.873 | 0.02% |

## FULL_DECODE_ONLY / large_decode / 0

实际阶段 {'decode': 16}；batch sizes [64]；满足目标 16/16 步。
GPU 事件 336，kernel 总和 136.946 ms，区间并集 136.946 ms。
关联情况 {'linked_to_step': 336}；可见 RoPEKV 模型 kernel：False。

| 实际阶段 | 算子 | 调用数 | GPU 总 ms | 此窗口可见 kernel 时间占比 |
|---|---|---:|---:|---:|
| decode | OP13_sampling | 128 | 118.143 | 86.27% |
| decode | OP12_LM_head | 16 | 18.010 | 13.15% |
| decode | unattributed | 192 | 0.793 | 0.58% |

## FULL_DECODE_ONLY / tail / 0

实际阶段 {'decode': 16}；batch sizes [6, 7, 8, 9, 10, 11, 12, 13]；满足目标 16/16 步。
GPU 事件 513，kernel 总和 49.550 ms，区间并集 49.550 ms。
关联情况 {'linked_to_step': 513}；可见 RoPEKV 模型 kernel：False。

| 实际阶段 | 算子 | 调用数 | GPU 总 ms | 此窗口可见 kernel 时间占比 |
|---|---|---:|---:|---:|
| decode | OP13_sampling | 256 | 34.344 | 69.31% |
| decode | OP12_LM_head | 16 | 14.211 | 28.68% |
| decode | unattributed | 241 | 0.995 | 2.01% |

## NONE / initial / 0

实际阶段 {'prefill': 2, 'mixed': 14}；batch sizes [1, 2, 3, 4, 5, 6, 7, 8]；满足目标 16/16 步。
GPU 事件 8188，kernel 总和 3897.968 ms，区间并集 3897.968 ms。
关联情况 {'linked_to_step': 8188}；可见 RoPEKV 模型 kernel：True。

| 实际阶段 | 算子 | 调用数 | GPU 总 ms | 此窗口可见 kernel 时间占比 |
|---|---|---:|---:|---:|
| mixed | OP05_GQA_prefill_branch | 588 | 1572.945 | 40.35% |
| mixed | OP09_gate_up | 588 | 824.577 | 21.15% |
| mixed | OP11_down | 588 | 426.388 | 10.94% |
| prefill | OP05_GQA_prefill_branch | 84 | 222.856 | 5.72% |
| mixed | OP03_QKV | 588 | 178.013 | 4.57% |
| mixed | OP08_O | 588 | 143.110 | 3.67% |
| prefill | OP09_gate_up | 84 | 117.773 | 3.02% |
| mixed | OP10_SwiGLU_or_compiler_fusion | 588 | 80.729 | 2.07% |
| mixed | OP02_norm_or_fused_norm | 1190 | 78.600 | 2.02% |
| prefill | OP11_down | 84 | 60.939 | 1.56% |
| mixed | OP06_GQA_decode_branch | 588 | 47.236 | 1.21% |
| mixed | OP13_sampling | 496 | 32.575 | 0.84% |
| prefill | OP03_QKV | 84 | 25.390 | 0.65% |
| prefill | OP08_O | 84 | 20.579 | 0.53% |
| mixed | OP04_RoPE_KV | 588 | 19.648 | 0.50% |
| mixed | OP12_LM_head | 14 | 11.960 | 0.31% |
| prefill | OP10_SwiGLU_or_compiler_fusion | 84 | 11.515 | 0.30% |
| prefill | OP02_norm_or_fused_norm | 170 | 11.289 | 0.29% |
| mixed | OP07_GQA_reduce | 588 | 3.728 | 0.10% |
| prefill | OP04_RoPE_KV | 84 | 2.807 | 0.07% |

## NONE / mixed / 0

实际阶段 {'mixed': 16}；batch sizes [10, 11, 12, 13, 14, 15, 16, 17]；满足目标 16/16 步。
GPU 事件 7744，kernel 总和 3981.992 ms，区间并集 3981.992 ms。
关联情况 {'linked_to_step': 7744}；可见 RoPEKV 模型 kernel：True。

| 实际阶段 | 算子 | 调用数 | GPU 总 ms | 此窗口可见 kernel 时间占比 |
|---|---|---:|---:|---:|
| mixed | OP05_GQA_prefill_branch | 672 | 1799.896 | 45.20% |
| mixed | OP09_gate_up | 672 | 940.593 | 23.62% |
| mixed | OP11_down | 672 | 487.325 | 12.24% |
| mixed | OP03_QKV | 672 | 203.418 | 5.11% |
| mixed | OP08_O | 672 | 162.858 | 4.09% |
| mixed | OP06_GQA_decode_branch | 672 | 129.204 | 3.24% |
| mixed | OP10_SwiGLU_or_compiler_fusion | 672 | 92.063 | 2.31% |
| mixed | OP02_norm_or_fused_norm | 1360 | 89.858 | 2.26% |
| mixed | OP13_sampling | 128 | 33.433 | 0.84% |
| mixed | OP04_RoPE_KV | 672 | 22.394 | 0.56% |
| mixed | OP12_LM_head | 16 | 14.880 | 0.37% |
| mixed | OP07_GQA_reduce | 672 | 5.187 | 0.13% |
| mixed | unattributed | 192 | 0.884 | 0.02% |

## NONE / large_decode / 0

实际阶段 {'decode': 16}；batch sizes [64]；满足目标 16/16 步。
GPU 事件 7072，kernel 总和 921.468 ms，区间并集 921.468 ms。
关联情况 {'linked_to_step': 7072}；可见 RoPEKV 模型 kernel：True。

| 实际阶段 | 算子 | 调用数 | GPU 总 ms | 此窗口可见 kernel 时间占比 |
|---|---|---:|---:|---:|
| decode | OP06_GQA_decode_branch | 672 | 606.153 | 65.78% |
| decode | OP13_sampling | 128 | 116.984 | 12.70% |
| decode | OP09_gate_up | 672 | 65.227 | 7.08% |
| decode | OP11_down | 672 | 39.482 | 4.28% |
| decode | OP03_QKV | 672 | 24.168 | 2.62% |
| decode | OP08_O | 672 | 18.529 | 2.01% |
| decode | OP12_LM_head | 16 | 18.227 | 1.98% |
| decode | OP07_GQA_reduce | 672 | 12.928 | 1.40% |
| decode | OP02_norm_or_fused_norm | 1360 | 9.899 | 1.07% |
| decode | OP10_SwiGLU_or_compiler_fusion | 672 | 4.632 | 0.50% |
| decode | OP04_RoPE_KV | 672 | 4.463 | 0.48% |
| decode | unattributed | 192 | 0.776 | 0.08% |

## NONE / tail / 0

实际阶段 {'decode': 16}；batch sizes [8, 9, 10, 11, 12, 13, 14, 15, 16]；满足目标 16/16 步。
GPU 事件 7080，kernel 总和 388.932 ms，区间并集 388.932 ms。
关联情况 {'linked_to_step': 7080}；可见 RoPEKV 模型 kernel：True。

| 实际阶段 | 算子 | 调用数 | GPU 总 ms | 此窗口可见 kernel 时间占比 |
|---|---|---:|---:|---:|
| decode | OP06_GQA_decode_branch | 672 | 149.907 | 38.54% |
| decode | OP11_down | 672 | 62.531 | 16.08% |
| decode | OP09_gate_up | 672 | 56.718 | 14.58% |
| decode | OP03_QKV | 672 | 31.273 | 8.04% |
| decode | OP13_sampling | 128 | 28.286 | 7.27% |
| decode | OP08_O | 672 | 22.708 | 5.84% |
| decode | OP12_LM_head | 16 | 14.617 | 3.76% |
| decode | OP02_norm_or_fused_norm | 1360 | 8.823 | 2.27% |
| decode | OP07_GQA_reduce | 672 | 5.236 | 1.35% |
| decode | OP04_RoPE_KV | 672 | 4.669 | 1.20% |
| decode | OP10_SwiGLU_or_compiler_fusion | 672 | 3.322 | 0.85% |
| decode | unattributed | 200 | 0.844 | 0.22% |
