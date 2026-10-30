# 4K top-p 行并行度：原内核调度诊断

选择 grid64 进入引擎验证，当前服务仍为 EXP20。B64 随机 logits 的截断阶段平均耗时下降 66.73%，不是整模吞吐提升；真实服务后续完成一轮诊断，但B64精确检查覆盖不足，正式四轮尚未启动，详见[失败复盘](topp_row_parallel_service_failure_20261030.md)。

## 为什么转向这里

EXP25 的 MLP 权重布局 ABBA 无端到端收益，已停止推广。EXP18 的 NONE/large_decode 诊断中，_topk_topp_kernel 每步约 6320.50 us，占该窗口可见 kernel 总工作量 10.97%；全部采样为 12.70%。FULL decode 的图内 kernel 不可见，所以不能将该比例套到生产 FULL 整程。

官方 benchmark 未显式覆盖 temperature/top-p；模型 generation_config 为 temperature=1、top_p=.95、do_sample=true。词表为 130560。PlatformFL 为 OOT，native sampler 在 B>=8 时进入 Triton top-k/top-p 截断；B<8 的 PyTorch 分支不是本轮对象。真实服务参数还会在下一阶段记录。

## 前后实现

原 wrapper 启动 min(SM数,B) 个 program，本机为16；B64 时每个 program 循环处理4行。新诊断仅把 grid 改为32或64，同时扩容每个 program 的 scratch。调用同一个原始 JIT 内核，8192/4096 tile、默认 warps/stages、查找表、算术和筛选算法全部保持；不修改 top-p，不改成贪心采样，不替换 RNG。

更多 program 提供更多可调度工作，是否提高实际 occupancy 需要硬件计数器，当前只报告耗时，没有测 occupancy。

## 全部诊断结果

| B | program 数 | 平均耗时变化 | 各点范围 |
|---:|---:|---:|---:|
| 64 | 32 | -45.718% | -46.751% ～ -43.804% |
| 64 | 64 | -66.732% | -69.878% ～ -65.305% |
| 32 | 32 | -46.545% | -47.531% ～ -45.358% |

3种子、3种logits分布(scale .5/1/3)，27个旧/新/旧夹测点完整保存；B8/16 的 grid 未变，只作控制检查。7样本×4次Graph重放，每次都包含同样的输入恢复copy，避免反复截断已mask数据；不把结果当作单纯kernel设备时间。最大绝对夹测漂移低于1%。

140项检查通过：masked logits逐元素相同、-inf掩码相同；Graph更新input/p/k后与eager一致；uniform/tied BF16/sparse -inf/peaked/行padding；同RNG状态及per-request generator下随机采样token与后续RNG状态一致。没有放宽容差。此结论相对当前生产实现，未声称其算法被穷尽证明。

64行scratch有效载荷为33,423,360字节，原16行为8,355,840字节；若在生产中扩容同一cache而非双份保留，名义差额25,067,520字节（23.90625 MiB）。微基准为对照持有独立buffer，尚不能称真实引擎增量或KV损失已测得。必须处理缓存扩容、旧Graph引用、reset和fallback，并测服务KV容量。

## 下一阶段 EXP27

先固定 grid64 候选与适用范围，再验证真实模型logits和top-p分支，补17/30/31/33/48/63等中间batch、缓存生命周期/Graph/fallback。引擎接入保持默认关闭，GPU和版本不支持时原路径回退；采样概率阈值、随机数生成和token选择定义保持。实际服务shadow检查与正式性能分开，禁止把额外比较开销混入官方计时。通过接线检查后再按官方4K四轮弃首测试。

每个后续阶段刷新日期和预算，受23:35计算/23:50保存限制；当前EXP26窗口已结束，不能继续套用。完整模型任务精度按用户暂缓，KV压缩仍需独立精度。

- [独立审核](../experiments/EXP-0026/review_20261029/audit.json)
- [预注册协议](../experiments/EXP-0026/README.md)
- [原始计时](../experiments/EXP-0026/artifacts/20261029_190815_grid_diagnostic/timings.jsonl)
- [全部检查](../experiments/EXP-0026/artifacts/20261029_190815_grid_diagnostic/checks.jsonl)
