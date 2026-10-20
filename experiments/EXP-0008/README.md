# EXP-0008：官方 4K 负载的独立算子 profiling

## 目的与假设（采集前）

2026-10-20 用户要求在 GPU 空闲时分析 4K 数据的各算子耗时。19:43 检查时 GPU 利用率 94%、显存 28,578 / 32,768 MiB，EXP-0006 的 16K 测评正在运行。保留该实验，完成并归档后优先执行本诊断，再恢复 EXP-0007 调度实验。

目前热点清单包含真实调用路径和部分微基准，尚无全模型设备时间排序。假设线性层、GQA、LM head 与数据准备的相对开销随 prefill、decode、mixed 阶段变化；以 trace 验证，不预设某一算子最慢。

## 冻结版本与负载

- vllm-plugin-FL：`07d9fa0cfdc5f09db178e29508e3c0bcb5c1e908`。
- FlagGems：`484413ea0db18221dd9e63e742361d837df9ff13`。
- 独立 detached worktree，沿用 EXP-0006 已完成 4K 的 RoPE+KV 融合及 GQA 版本。
- MiniCPM5-2B；输入 4096、输出 1024、最大并发 64、请求 256；沿用官方脚本的 random dataset、ignore-eos 及未显式覆盖的采样参数。
- 每个诊断 pass 运行一次上述完整请求集；这不是官方“四轮弃首轮”成绩。所有带 profiler 的吞吐、延迟只用于检查负载完成，禁止与正式成绩比较。
- 主 pass 使用 `FULL_DECODE_ONLY`；另一个 `NONE` pass 仅辅助形状与算子归属。两者单独汇总，禁止将关闭 Graph 的耗时替代生产 Graph 耗时。

## 采集和保护

1. 等待 EXP-0006 16K 控制器及归档器结束，获取共享 GPU 文件锁；检查 GPU 低利用率、低显存占用和服务端口空闲。不停止其他 GPU 服务。
2. 先运行极小的 PyTorch CPU+CUDA profiler 能力检查，必须实际导出设备事件。失败保存证据，禁止把 CPU 时间冒充 GPU 时间。
3. 独立服务使用原生 `--profiler-config`。只采集有限迭代窗口，记录 shapes，禁用内存采集和前端全程 profiling；启动、稳态及后续请求进入阶段分别采样。
4. 诊断 worker extension 只增加 CPU 元数据和命名范围：读取已经在 CPU 上的 query length、prompt/computed token counts、上下文长度及 padding/Graph 决策；不读取 token 内容，不增加 GPU 同步，不改算子或输出。
5. 每个窗口保留原始压缩 trace、采集时间、运行命令、提交和脚本哈希、route 证据、采样阶段与各 kernel 统计。设备时间取 GPU kernel 事件；CPU self time、拷贝及同步分别报告，嵌套 CPU inclusive 时间不相加。
6. 已识别 kernel 映射到热点清单 OP 编号；无法可靠区分的 GEMM 或编译融合算子保留未归属类别。只在相关性和张量形状提供证据时细分 QKV/O/gate-up/down/LM head。
7. 结果与小型 CSV 自动保存、提交并推送实验仓库；大 trace 留在服务器并记录哈希。失败也保留状态和日志。只清理本任务启动的子进程，释放 GPU 后恢复 EXP-0007。

## 待产物

`RESULTS.md`、`STATUS.json`、分 pass/window 的 kernel 与 CPU operator CSV、语义归属表、原始 trace manifest、运行及路由证据。尚未完成采集时不得填写虚构耗时。

## 部署记录

19:59 后台控制器已入队，等待 EXP-0006；EXP-0007 的等待队列已加入前置门槛。15 类 CPU 检查通过。首次提交被 `git diff --check` 的文件末尾空行检查拦截，尚未推送；在 GPU profiling 尚未开始时去除空行并重新冻结脚本哈希，不涉及推理逻辑变化。


## 今晚关机截止

用户更新：机器北京时间 2026-10-21 00:00 关闭。本实验服从 2026-10-20 23:35 计算截止及 23:50 保存目标；未完成现场单独保留，重启后再续。[完整安排与守护脚本](../../ops/night_20261020/README.md)。
