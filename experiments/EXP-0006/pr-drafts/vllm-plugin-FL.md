# 【FlagOS开放计算全球挑战赛S2.赛题2】【LeanScale】接入 RoPE 与分页 KV 写入融合原型

## PR Category
Vendor

## PR Type
Performance

## Description
为天数 Q16/KV2/D128 路径新增 functional RoPE + KV 写入融合。原图在 KV update 前切分，因此在切分前匹配并替换，保留 attention 的显式 dummy dependency。输入 projection 不变，缓存写入只有一次。当前为原型，GPU 与整模收益尚未验证。

## Related Issues
LeanScale EXP-0006；无关联上游 issue。

## Changes
保留原 rotary_embedding 策略选择；非 FlagOS、IO dump、其他 backend 或不支持形状走原 RoPE 与 cache update。拒绝多缓存写入和不安全的 value 重排。匹配幂等，不修改共享 site-packages。

## Testing
8 项 CPU 图匹配/拒绝/幂等/回退检查通过。GPU 数值、Graph、64 位缓存地址与微基准已排队；尚未确认真实模型 42 层接入。完整官方性能未完成。无损改动的完整任务精度按用户要求暂缓。

## Checklist
- [x] 独立分支并保留官方基线
- [x] CPU 结构和回退检查
- [ ] GPU 数值和 Graph
- [ ] 真实模型接入与官方性能
- [ ] 最终比赛精度验证
