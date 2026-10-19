# EXP-0002：天数 RoPE 按 head 分组并行

状态：完成独立内核实验，候选尚未接入默认整模推理路径。

## 假设与修改

原 FlagGems 内核在一个 token 的 program 中循环遍历 Q/K heads。候选把 head 分组分配给独立 program，每个线程先读取一对旋转元素，再计算和写回两项结果，减少逐 head 循环。

算子源码目标：yyq0210/FlagGems，`leanscale/iluvatar-rope-v1` 分支，`src/flag_gems/runtime/backend/_iluvatar/fused/rotary_embedding_parallel.py`。函数没有注册为默认替代实现。原始工作区仍保持官方基线。

## 正确性与失败记录

- 36 组 dtype/形状/布局 × 5 个 head 分组，共 180 次候选对独立数学参考检查通过。
- 实际模型 BF16/FP16、head_dim=128，与原内核逐位一致；覆盖非连续 Q/K 切片，并检查 V 未改写。
- FP32 扩展测试与原内核出现差异，包括首次 D=80 和另一次 M=257/D=128。候选对独立参考的容差检查通过，原内核差异的机制尚未定位，不把它归因于某个已证实的根因。
- 原失败日志保存在 artifacts/；失败后才增加独立参考、区分精度和对照检查。未删除失败结果。

## 测量方式

BF16，Q/KV=16/2，head_dim=128，固定随机 seed 1234。正确性使用随机输入；计时输入置零以避免重复原地旋转造成数值漂移。预热 5 次，CUDA Graph 捕获 100 次内核调用，每轮重放 5 次，7 轮取中位数；sweep 后复测原实现。

| token 数 | 原内核 µs | 候选 µs | 内核加速比 | BH | warps |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.188 | 1.701 | 3.05× | 2 | 4 |
| 8 | 5.294 | 1.731 | 3.06× | 4 | 4 |
| 32 | 5.415 | 2.003 | 2.70× | 8 | 4 |
| 64 | 5.450 | 2.394 | 2.28× | 8 | 4 |
| 128 | 5.903 | 3.007 | 1.96× | 8 | 4 |
| 512 | 10.821 | 6.350 | 1.70× | 16 | 4 |
| 2048 | 31.256 | 17.857 | 1.75× | 16 | 4 |

表格是本轮 sweep 的最快配置，完整配置与前后基线在 artifacts/rope_results.json；不推断为其他形状的保证收益。

## 实际接入限制

官方默认编译图中 RoPE 走 native/Inductor 路径，未调用这里的 FlagGems RoPE。当前结果只比较两个独立内核。下一步须以实际编译路径为对照，设计合法接入和回退，再验证整模吞吐、TTFT 和数学精度。

## 复现

GPU 空闲时，在具备上述环境的服务器执行：

```bash
cd /workspace/LeanScale/MiniCPM-Infer-Sys-Opt
python experiments/EXP-0002-rope/bench_rope.py \
  --kernel /workspace/LeanScale/FlagGems/src/flag_gems/runtime/backend/_iluvatar/fused/rotary_embedding_parallel.py \
  --output-dir /workspace/experiments/rope-reproduction-001
```

原始测试脚本与源码的哈希由 artifact_manifest.json 记录。新增复现入口只调整源码加载和输出目录，不修改原始测试及计时逻辑。
