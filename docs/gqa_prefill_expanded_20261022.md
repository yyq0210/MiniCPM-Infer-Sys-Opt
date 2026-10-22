# GQA prefill 扩大验证与引擎接入

2026-10-22，EXP-0015→EXP-0016。单算子验证结果，整模收益尚待官方4K/16K。
42项数值/Graph检查全部通过，变动前后累计39444个query独立FP32对照；包括真实>2^31元素prefill地址。
36工作点均无>3%回归，最大前后基线漂移0.769%。负数为耗时降低。

| 工作点 | 完整GQA耗时变化 | 前后基线漂移 |
|---|---:|---:|
| prefill_q2_L4096 | -13.643% | -0.002% |
| prefill_q2_L16384 | -13.413% | -0.000% |
| prefill_q8_L4096 | -13.651% | +0.002% |
| prefill_q8_L16384 | -14.784% | +0.000% |
| prefill_q17_L4096 | -12.695% | -0.003% |
| prefill_q17_L16384 | -12.421% | +0.006% |
| prefill_q64_L4096 | -11.219% | +0.051% |
| prefill_q64_L16384 | -20.568% | -0.001% |
| prefill_q128_L4096 | -11.500% | -0.015% |
| prefill_q128_L16384 | -26.155% | -0.003% |
| prefill_q512_L4096 | -9.863% | +0.042% |
| prefill_q512_L16384 | -24.856% | +0.010% |
| prefill_q2048_L4096 | -9.431% | +0.534% |
| prefill_q2048_L16384 | -11.070% | +0.107% |
| prefill_first_chunk | -9.833% | -0.119% |
| multiple_prefills | -6.165% | +0.057% |
| ragged_prefills | -15.830% | -0.015% |
| mixed4k_decode_first | -10.054% | +0.036% |
| mixed4k_prefill_first | -9.974% | +0.040% |
| mixed16k_decode_first | -11.444% | -0.011% |
| mixed16k_prefill_first | -11.404% | +0.169% |
| small_mixed_decode_first | -24.478% | -0.117% |
| small_mixed_prefill_first | -24.766% | +0.071% |
| medium_mixed | -23.783% | -0.045% |
| decode_B1_L4700 | +0.032% | -0.022% |
| decode_B1_L16384 | +0.057% | -0.027% |
| decode_B8_L4700 | +0.004% | +0.113% |
| decode_B8_L16384 | +0.041% | +0.014% |
| decode_B16_L4700 | +0.151% | -0.159% |
| decode_B16_L16384 | -0.038% | +0.069% |
| decode_B30_L4700 | +0.089% | -0.769% |
| decode_B30_L16384 | +0.043% | +0.009% |
| decode_B32_L4700 | +0.023% | -0.127% |
| decode_B32_L16384 | +0.069% | -0.033% |
| decode_B64_L4700 | +0.033% | -0.130% |
| decode_B64_L16384 | -0.033% | +0.060% |

原方案prefill64行/4warps/stage2/tile32；新方案仅把软件流水级数改为1，decode组合保持已选版本。
更大的tile64和stage3实测退化，因此未采用。该选择基于实测，不推断任意形状都获益；以上36点完整列出。
插件显式传入prefill参数，Gems通用默认仍stage2。新引擎在独立EXP-0016树；反向替换完全恢复父文件，fallback未变。
实际编译检查按8行decode与64行prefill分开识别，两者都须stage1，分别校验18KiB/32KiB shared与tile形状。
[完整原始数据](../experiments/EXP-0015/artifacts/20261022_144537_extended/RESULTS.md)。
[接入方案与测评门槛](../experiments/EXP-0016/README.md)。
