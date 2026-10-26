# EXP19 prefill页号广播初筛与扩大验证

4个配置均完成20项数值/Graph检查、14个完整GQA计时点，共80项检查/56个计时点。原始采样与IR哈希已独立重算。
layout1仅页号广播在11个prefill/mixed点全部优于其他已测配置，耗时下降15.53%–20.71%；decode回归约±0.11%。这是单算子诊断，尚无新增整模收益。

| 配置 | prefill/mixed耗时变化 | decode最大绝对变化 | 最大夹测漂移 |
|---|---|---:|---:|
| layout0 | -0.197% .. +0.292% | 0.108% | 0.385% |
| layout1 | -20.713% .. -15.534% | 0.106% | 0.378% |
| layout2 | -10.235% .. -2.562% | 0.030% | 0.736% |
| layout3 | -16.420% .. -9.255% | 0.071% | 0.841% |

代表性prefill编译资源：原路径99 registers/32KiB shared，页号广播98 registers/32KiB shared；spill均0。共享内存不变，不能据此宣称occupancy提高。K转置变体117 registers/24KiB shared，虽shared更少但实测不如仅页号广播，资源数字不能代替计时或动态occupancy。
实际TTIR核验layout1/3每tile读取2个页号并以int64广播到32个token lane；layout0/2仍为32个token lane的页号读取。
现在固定layout1进入扩大验证：42项all-query FP32/Graph检查，含两dtype、page8/16/32/64、NHD/HND、CU变化和真实超过2^31元素的KV地址；42个完整GQA计时点，包括6个EXP18形状导出点。初筛的CPU地址模型不代替这些GPU检查。
扩大验证前后均固定EXP16为直接对照，服务选择仍是EXP16；通过后才接入引擎和正式4K/16K测试。
[初筛全部结果](../experiments/EXP-0019/artifacts/20261023_144047_screen/RESULTS.md) · [独立初筛审核](../experiments/EXP-0019/expanded/screen_audit.json)。


## 扩大验证完成与引擎接入

42项全query数值/Graph检查全部通过，变动前后39444个query独立FP32对照；含8项CU变化、两个dtype真实超过2^31元素KV地址。42个完整GQA夹测点全部保留，最大基线漂移2.4494%，无>3%退化。

| 分组 | 点数 | 耗时变化 |
|---|---:|---:|
| pure_prefill | 17 | -24.099% .. -16.764% |
| mixed | 7 | -18.804% .. -15.523% |
| decode | 12 | -0.069% .. +0.424% |
| exp18_shape_derived | 6 | -20.787% .. -15.384% |

EXP18形状导出点使用CPU上下文上界与随机QKV，不是原请求重放。完整数据见[扩大验证](../experiments/EXP-0019/artifacts/20261023_145808_extended/RESULTS.md)。

EXP20仅把prefill_layout=1显式接入插件，逆向移除参数与日志可恢复EXP16父文件。CPU路由拒绝检查通过；真实服务42层与官方4K/16K尚待测，服务选择仍EXP16。
