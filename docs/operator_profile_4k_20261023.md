# 当前性能版本的 4K 热点：混合批次、稳定 decode 与尾部

2026-10-23，EXP-0018。两组诊断于 14:01 完成，各 256 个请求全部成功、0 失败；每组输入 1,048,576 tokens、输出 262,144 tokens。8 个目标窗口各有 16 步，共 128 步，原始 trace、逐步 metadata 和实际编译缓存已独立复核。

直接基线是已选定的 EXP-0016：插件 `588a6147cd291e272324ab6c812024cd587c6287`，FlagGems `fd209df717648002ebf5ba4ec351424dcc997a53`。本次只加入 worker 外围的诊断标注，推理和算子代码没有变化。

[自动归档](../experiments/EXP-0018/artifacts/20261023_132558_profile_4k/) · [独立审核](../experiments/EXP-0018/review_20261023/audit.json) · [完整逐窗口结果](../experiments/EXP-0018/RESULTS.md) · [分形状汇总](../experiments/EXP-0018/review_20261023/shape_summary.csv)。

## 测量口径

每组使用原官方 command builder，输入4096、输出1024、最大并发64、256请求，默认采样、ignore_eos、原调度和完整 KV。FULL_DECODE_ONLY 与 NONE 分别启动服务、建立独立编译缓存，各跑一次完整请求集。

**本次是有 profiler 开销的诊断，不是官方四轮成绩。** 不以两组诊断吞吐比较优化收益，不合并两个模式或不同窗口来计算整程算子占比。下面的百分比均以对应窗口可见 GPU kernel 的 duration 总和为分母；不同 stream 可能重叠，因此它也不等于墙钟延迟占比。

设备探针能捕获 eager kernel，但 Graph replay 内部设备事件为0。生产配置下，混合批次实际走 NONE、可以观察模型算子；纯 decode 实际走 FULL、模型 Graph 内部不可见。后者只看得到 LM head 和采样等外围算子，不能据其86.27%的采样占比得出“采样是整模最大热点”。纯 decode 内部归属来自独立 NONE 诊断。

窗口由实际 CPU metadata 触发，最终逐步核验阶段；预热和客户端初始自检不计入256个主请求。context 数值是异步 CPU 上界，不能当成从 GPU 读取的实际序列长度。只通过 correlation 和 CPU 调用祖先关联 kernel，未关联项保留。

## 当前热点与真实形状

| 窗口 | 实际形状 | GQA 主核 | gate/up | down | 采样 | 解释 |
|---|---|---:|---:|---:|---:|---|
| FULL配置 / mixed | 16步；B10–17；M=2048 | prefill 45.20%，decode 3.24% | 23.61% | 12.23% | 0.86% | 这些步实际未用Graph，模型内部可见 |
| NONE / large_decode | 16步；B=M=64；CPU context上界4099–4242 | decode 65.78% | 7.08% | 4.28% | 12.70% | 只用于decode算子归属 |
| NONE / tail | 16步；B=M为8–16；全部256请求已入场 | decode 38.54% | 14.58% | 16.08% | 7.27% | 小batch投影值得单独诊断 |

mixed 中 prefill 主核单次均值2678.66 μs，gate/up为1399.36 μs，down为724.77 μs。每步均有42层对应调用，不是通过FLOPs估算耗时。

NONE/B64 的 decode 主核均值902.01 μs，reduce为19.24 μs；尾部 decode 主核均值223.08 μs，reduce为7.79 μs。B、context与执行模式不同，不能据这两行计算某个优化的加速比。

initial 窗口实际是2步纯prefill加14步mixed，并非16步纯prefill。两个模式的mixed窗口均覆盖B10–17的早期混合批次，**不能代表所有运行期间的混合批次**。后续微基准还应保留B64混合与16K回归工作点。

生产 FULL 尾部窗口实际B6–13、Graph padding M8/16；NONE 尾部B8–16、不做Graph padding。后续小M GEMM必须同时测真实M与生产padding后的M，不能直接把NONE尾部收益当作生产Graph收益。

## 与源码对应，以及优先级

### 1. 优先验证 prefill 的分页读取和 K 加载布局

入口仍是 `IluvatarGQAImpl.forward → paged_gqa → _attention`。当前[算子源码](https://github.com/yyq0210/FlagGems/blob/fd209df717648002ebf5ba4ec351424dcc997a53/src/flag_gems/runtime/backend/_iluvatar/fused/paged_gqa.py)的分页页号广播只应用于decode：

```python
if (DECODE and (DECODE_LAYOUT == 1 or DECODE_LAYOUT == 3)
        and PAGE <= TILE and TILE % PAGE == 0):
    # 每个逻辑页读取一次页号，再广播到token lanes
    ...
else:
    blocks = tl.load(BT + seq * BTS + tokens // PAGE, ...)
```

当前服务PAGE=16、prefill TILE=32，因此每个KV tile逻辑上只有两个页号，prefill路径仍表达为32个token lane的页号读取。改为两次页号读取再广播，有机会减少冗余索引工作；K改为先按[token, dim]加载再转置，也可能改变设备内存访问和编译布局。这些只是新假设，编译器可能已合并部分工作，额外转换也可能使寄存器或shared memory上升，必须实测。

下一轮固定已选的prefill rows64、warps4、stage1、tile32，比较三种独立变体：仅页号广播、仅K加载布局、两者组合。decode保留当前配置。使用本次实际query/context上界向量构造标注清楚的形状诊断，计时完整GQA调用，再单列prefill kernel与资源。沿用FP32 oracle、两种dtype、随机物理页、NHD/HND、页边界、ragged/empty、int64地址和Graph输入更新检查。

这不是重复EXP15的rows/warps/stages网格，也不是已取得的新收益。通过筛选和扩大验证后，才接入引擎并跑完整官方4K；16K做回归。

### 2. 第二优先级：尾部小 M 的投影路径

尾部 down 与gate/up合计占该NONE窗口30.66%；加QKV、O后，四类层内投影约44.54%。其中down单次均值93.05 μs，而B64窗口为58.75 μs。小batch工作量更少，却出现更高单次设备耗时，值得检查实际厂商GEMM算法、权重布局和M分派，不能直接断言是带宽或occupancy问题。

实际路径是 `LlamaMLP.forward → down_proj / gate_up_proj → UnquantizedLinearMethod.apply → dispatch_unquantized_gemm → F.linear`，必须修改真实接入路径，不能只修改插件中未被调用的同名实现。优先测M=8/16/32/64及生产Graph padding边界，使用预热后的完整子图计时。

不直接恢复已淘汰的融合原型：EXP6的gate/up＋SwiGLU在M30时原实现110.41 μs、最快原型151.59 μs；norm＋QKV原实现24.56 μs、原型118.62 μs。只有提出新的GEMM效率或复用方案，才有理由重试。详见[历史融合选择](../experiments/EXP-0006/FUSION_DECISION.md)。

### 3. 保留 decode 与采样为后续方向

B64的decode GQA仍是最大可见算子；已有tile、分段、warp/stage和8行分页布局结果要复用，后续只探索有新证据支持的形状分派或访存方案。采样在该NONE窗口占12.70%，但任何优化必须保持官方默认采样、随机数与分布语义，不能改成greedy来“加速”。

本轮先不将采样改动、调度或KV压缩与attention访存混在一起。无损方案的完整模型任务精度继续按用户要求暂缓；KV压缩仍必须独立验证精度。

## 审核与限制

- 两个完整请求集均零失败；8个trace、128步metadata、逐请求query/context、实际Graph模式与padding全部重算核对。
- 两个服务各有64个decode及64个prefill的实际编译记录，全部重新核验CoreX参数、TTIR、哈希。各有1份融合图源码；FULL日志记录35个捕获尺寸，NONE未捕获FULL图。
- 保留的源码、benchmark、脚本和安装profiler相关源码哈希一致；结果自动推送提交为`a4fa36f3df04ca2db6447bd6999f4a17d0bcbfc4`。
- trace中匹配到的`Call CompiledFxGraph`是调用已编译图的标注，不能当成重新编译事件。没有证据据此解释历史4K波动，尚不作编译因果结论。
- 当前是有限目标窗口，没有测得整程每阶段权重；不能把旧EXP9与本次窗口占比差异拆成单个优化收益。
- 本次没有更改已选性能版本，也没有产生新的官方吞吐分数。GPU已释放，原始trace与日志保留服务器。
