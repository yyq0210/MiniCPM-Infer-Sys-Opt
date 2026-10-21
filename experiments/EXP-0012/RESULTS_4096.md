# EXP-0012：GQA stage1 官方 4K 完整结果

2026-10-21：官方四轮全部成功，按协议弃首轮。后处理要求了已撤销调度的专用日志，导致控制器在测量完成后报错。
原 status.json、日志、CSV、哈希及失败归档 e097a1b 保持原样；本次只做离线复核，未重新测量，也未拼接任何轮次。
补充证据见 [postprocessing_recovery.json](artifacts/runs/20261021_162416_official_4k/postprocessing_recovery.json)。

| 指标 | EXP-0006 | GQA stage1 | 变化 |
| --- | ---: | ---: | ---: |
| Total tok/s | 2686.33 | 2936.01 | +9.29% |
| Output tok/s | 537.27 | 587.20 | +9.29% |
| Mean TTFT (ms) | 5920.51 | 5874.60 | -0.78% |
| Mean TPOT (ms) | 111.60 | 101.45 | -9.09% |
| Mean ITL (ms) | 111.60 | 101.45 | -9.09% |

保留三轮吞吐 2908.21 / 2844.26 / 3055.57 tok/s，平均 2936.01 tok/s，CV 3.6909%；EXP-0006 CV 3.6373%。
相对原始官方基线 2008.97 tok/s，吞吐提升 46.15%。对照使用历史服务会话，不是交错 A/B。
首轮 1648.24 tok/s 按既定协议丢弃，所有数据均保留。不能用最优单轮 3055.57 代替均值。

| 调度指标（包含首轮的全部四轮） | EXP-0006 | GQA stage1 |
| --- | ---: | ---: |
| mean_queue_seconds_all_four_rounds | 8.721664513393534 | 8.404443143763274 |
| peak_running_sampled | 64 | 64 |
| peak_waiting_sampled | 60 | 61 |
| peak_cache_usage_percent_sampled | 62.0 | 62.0 |
| preemptions | 0.0 | 0.0 |

排队计数覆盖全部四轮及测评脚本的端点探测请求，吞吐和延迟表弃首轮，口径不可混用；日志峰值只是采样值。
原调度专用准入日志不存在是正确状态；修正后处理后仍拒绝缺失计数器、计数器重置和劣化调度路径。
42 层 GQA stage1、42 层 RoPE+KV、35 张 CUDA Graph 及实际编译元数据再次离线核验通过。
插件 08d1711e58fdd0f4cd4f746f9e21cb59800ec1d6；FlagGems 19aa059af4ce456ef2e34e3bb83eb949d7e7b702。
16K 待测；最优服务选择暂不变。完整任务精度按用户要求暂缓，不宣称赛事精度达标。
