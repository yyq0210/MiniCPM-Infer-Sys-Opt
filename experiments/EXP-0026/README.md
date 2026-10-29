# EXP-0026：4K top-p 行并行度诊断

## 假设与真实路径

在写候选之前登记：当前服务仍为 EXP20。官方 benchmark 没有显式 temperature/top-p 覆盖；模型 generation_config 为 do_sample=true、temperature=1.0、top_p=0.95，逻辑词表 130560。PlatformFL 为 OOT，TopKTopPSampler 使用 forward_native；B>=8 转入 apply_top_k_top_p_triton，B<8 保持原 PyTorch 路径。既有 EXP18/NONE/large_decode 的真实 _topk_topp_kernel 每步均值 6320.4975 us，占该窗口可见 kernel 总时间 10.9747%；不能称生产 FULL 整程占比。

原 wrapper NUM_PROGRAMS=min(num_sm,B)，本设备 num_sm=16，B64 时每 program 顺序处理四行。候选只比较 min(2*num_sm,B)、min(4*num_sm,B)，使用同一个原始 JIT 内核、同一 8192/4096 tile、默认 warps/stages、原表和输入 dtype。每 program 的 scratch 必须相应扩容，不能沿用 16 行大小而越界。算法、归约、top-k/top-p 阈值、随机数生成和请求配置不变。未接入推理引擎。

## 固定诊断协议

三种子，B8/16/32/64，V=130560，输入为正态 scale0.5/1/3，主测 p=.95/k=None；B8/16 的 grid 与原版相同，只做控制检查，不虚报加速。B32 实测 grid32；B64 比较 grid32/grid64。候选相同 grid 去重。另测 uniform/ties、稀疏 -inf、peaked、p=.0001/.5/.95/1、top-k=50、top-k 与 top-p 组合，以及带行间 padding 的 stride。

所有 masked logits 必须与当前原实现逐元素完全相同（包括 -inf 掩码），不放宽阈值；输入、p 和 k 的 Graph 重放更新后再次比较。masking 不得改变 RNG。使用同一 RNG 状态和相同 per-request generators 复验后续随机采样 token 与 RNG 状态一致。参考为当前生产实现，不宣称其数学算法已穷尽验证。

计时是 GPU event 的旧/新/旧夹测，7 样本、每样本4次 Graph 重放；每次都含 input→logits restore copy，避免原地 masking 重复处理已截断数据。两组含同样恢复开销。记录主目标全部计时、漂移、CV、scratch 字节与编译资源，不只挑最快点；随机张量诊断不等同真实 logits 或端到端收益。

选择进入下一阶段须 B64 所有三个种子×三个分布均至少5%改善、B32不回退超过3%、夹测漂移绝对值不超过3%，所有精确值/Graph/RNG检查通过。未达标则不接入。若仅某个 batch 的 grid 更好，必须另行固定 dispatch 并复核，不能在端到端测量中按成绩自适应挑选。

## 时间、保存与后续

一次性后台进程，GPU共享锁、两次空闲及端口检查，独立 Triton 缓存。启动时刷新时间，分配45分钟计算预算和10分钟归档余量，受2026-10-29 23:35停计算、23:50保存约束；预算不足不启动。完成或失败均保存 JSONL/状态/编译证据并自动推个人文档 fork，不覆盖原始失败，不自动重启。整模精度暂缓；这不是KV压缩或调度方案。

通过后先用真实服务收集采样分支和真实logits诊断，完成加载、scratch生命周期、Graph/fallback接线，再官方4K。原生随机数和筛选语义必须保留，不改为 greedy，不把微基准当E2E。源码集成到个人fork时按比赛Draft格式记录，当前只提交实验。
