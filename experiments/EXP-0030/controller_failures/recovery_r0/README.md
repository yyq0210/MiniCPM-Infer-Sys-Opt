# 恢复预检 r0 失败

原16K A已完成，恢复工具在CPU独立审核时错误沿用4K的64decode+64prefill变体数量。实际16K A为36+31，与历史EXP20 16K相同；此前逐内核语义/IR/hash检查已通过。没有开始B、没有修改或重测A。

r1核验缓存中全部_attention.json与证据清单一一对应、无遗漏重复，并继续逐个核验warps/stages/shared/load/page广播和MLP/融合Graph。只修正审核口径，不更改推理代码、benchmark、数据或性能门槛。原恢复脚本和失败保留。
