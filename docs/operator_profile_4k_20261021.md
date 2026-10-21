# 4K 实际推理算子耗时：完整 profiling 结果

完成时间：2026-10-21 13:59:52（北京时间）。EXP-0009。

两种模式分别完成官方形状 4096 输入 / 1024 输出 / 最大并发64 / 256请求；每种模式实际处理1048576输入tokens、262144输出tokens。各有 initial/steady/late 三个窗口，六个 trace 均解析成功。这不是官方四轮性能测试，带profiler的吞吐不用于候选收益排名。

## 模式与实际阶段

| 配置模式 | 窗口 | 实际阶段/迭代数 | GPU kernel 数 | 模型内部 kernel 可见性 |
|---|---|---|---:|---|
| FULL_DECODE_ONLY | initial | prefill×2，mixed×14 | 8188 | 观察到模型主kernel |
| FULL_DECODE_ONLY | steady | mixed×16 | 7744 | 观察到模型主kernel |
| FULL_DECODE_ONLY | late | decode×16 | 336 | Graph内部不可见 |
| NONE | initial | prefill×2，mixed×14 | 8188 | 观察到模型主kernel |
| NONE | steady | mixed×16 | 7744 | 观察到模型主kernel |
| NONE | late | decode×16 | 7072 | 观察到模型主kernel |

FULL_DECODE_ONLY 只对decode启用完整图重放；其mixed窗口仍可观察模型kernel。当前CoreX profiler看不到decode图内部kernel，不能用FULL/late外围采样占比推导全模型热点。NONE/late用于补充纯decode的算子归属和诊断耗时，不能替代生产Graph时延。

## 归属复核方法

所有六个trace与原分析的SHA256、kernel数量及duration总和完全相同；只补充分类，不修改原始trace。新增脚本 analyze_profile_attribution.py 与旧版并存。

- LM head：CUDA correlation关联到aten::linear，输入[64,2048]、权重[130560,2048]；按F.linear权重转置语义识别，不能当普通mm的[K,N]权重。
- 采样：关联到Worker.sample_tokens的LS_SAMPLE标注，包含其内部全部GPU kernel；旧分类只识别名称中的topk/softmax等，范围较窄。
- GQA：固定源码和segments=16下，主kernel grid第三维16对应decode分支，第三维1对应prefill分支；mixed窗口据此分开。分支可能包含提前返回的program，调用数不等于全部program都做了有效attention。
- norm分类包括编译器融合的embedding＋首层norm；不能从一个融合kernel拆出独立embedding耗时。

## 可见 mixed 窗口的算子耗时

生产配置FULL_DECODE_ONLY / steady，实际16个mixed迭代。均值是该分类每次kernel调用的平均时长；norm、采样等分类包含不同kernel，细项和P95见CSV。百分比分母为该窗口全部GPU kernel时间总和。

| 算子分类 | 调用数 | 平均 μs/调用 | GPU总耗时 ms | 占比 |
|---|---:|---:|---:|---:|
| GQA prefill 分支 | 672 | 2952.844 | 1984.311 | 42.357% |
| gate/up 线性投影 | 672 | 1400.407 | 941.074 | 20.088% |
| GQA decode 分支 | 672 | 867.494 | 582.956 | 12.444% |
| down 线性投影 | 672 | 725.811 | 487.745 | 10.411% |
| 合并 QKV 投影 | 672 | 303.203 | 203.752 | 4.349% |
| O 投影 | 672 | 243.804 | 163.836 | 3.497% |
| SwiGLU／对应编译融合 | 672 | 138.107 | 92.808 | 1.981% |
| norm 及编译器融合 | 1360 | 66.470 | 90.399 | 1.930% |
| 采样（Worker.sample_tokens 范围内） | 128 | 678.267 | 86.818 | 1.853% |
| RoPE＋KV 写入 | 672 | 33.919 | 22.794 | 0.487% |
| LM head | 16 | 1069.707 | 17.115 | 0.365% |
| GQA 分段归约 | 672 | 15.097 | 10.145 | 0.217% |
| 其余未分类 kernel | 192 | 4.946 | 0.950 | 0.020% |

## 纯 decode 窗口的算子耗时

NONE / late，16个decode迭代，B=64。CPU元数据上下文上界4618–4761。每步42次主attention和42次reduce，LM head每步一次；以下不是Graph模式排名。

| 算子分类 | 调用数 | 平均 μs/调用 | GPU总耗时 ms | 占比 |
|---|---:|---:|---:|---:|
| GQA decode 分支 | 672 | 1672.156 | 1123.689 | 78.340% |
| 采样（Worker.sample_tokens 范围内） | 128 | 838.978 | 107.389 | 7.487% |
| gate/up 线性投影 | 672 | 94.890 | 63.766 | 4.446% |
| down 线性投影 | 672 | 59.697 | 40.116 | 2.797% |
| 合并 QKV 投影 | 672 | 36.773 | 24.711 | 1.723% |
| O 投影 | 672 | 28.849 | 19.387 | 1.352% |
| LM head | 16 | 1138.398 | 18.214 | 1.270% |
| GQA 分段归约 | 672 | 20.395 | 13.705 | 0.955% |
| norm 及编译器融合 | 1360 | 8.417 | 11.447 | 0.798% |
| RoPE＋KV 写入 | 672 | 8.614 | 5.789 | 0.404% |
| SwiGLU／对应编译融合 | 672 | 7.930 | 5.329 | 0.372% |
| 其余未分类 kernel | 192 | 4.303 | 0.826 | 0.058% |

采样内单独的 `_topk_topp_kernel` 为16次，平均5717.856 μs，占全部可见kernel时间约6.378%；整个sample_tokens范围为7.487%。官方默认采样参数保持不变，不能通过改成greedy来宣称算子优化收益。

## 优化顺序

mixed阶段先关注GQA prefill（42.357%）、gate/up（20.088%）、GQA decode分支（12.444%）和down（10.411%）。纯decode诊断先关注GQA主内核（78.340%），之后是采样（7.487%）、gate/up（4.446%）与down（2.797%）。RoPE＋KV已融合，纯decode诊断仅约0.404%，继续扩大此处融合的收益优先级较低。

GQA具体实现、111寄存器/线程、20KiB shared、warp64和分段选择的依据见 [GQA专项分析](gqa_optimization_analysis_20261021.md)。该记录提供下一轮候选方向，尚无新GQA版本收益。

CPU inclusive时间不可直接相加；kernel总和是设备工作量，不是请求延迟，跨stream可能重叠。采集忽略frontend，不能据此计算HTTP排队或完整EngineCore调度开销；这些由正式测评与Prometheus另行统计。

原始trace留服务器；各窗口CSV、逐步metadata、原始trace哈希和脚本见 [完整归属复核](../experiments/EXP-0009/artifacts/20261021_131254_profile_4k/ATTRIBUTION_REVIEW.md)。
