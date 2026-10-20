# EXP-0006 实际模型融合接入

验证时间：2026-10-20T16:56:15.127244+08:00。运行：20261020_165011_official_4k。

修复前的启动被接入门槛阻止，未执行 benchmark。修复后的真实模型证据：

- 图重写匹配 42 层；实际 fused RoPE+KV 调用 42 层。
- GQA 实际启用 42 层。
- computation_graph 有 42 个 fl_rope_kv_dispatch，旧 fl_rotary_embedding_dispatch 和 unified_kv_cache_update 均为 0。
- FULL_DECODE_ONLY 的 35 个图完成捕获；API 请求生成 32 tokens。
- 4K 官方测评已启动，完整四轮后才报告成绩；此证据不等于 MATH500 精度验证。

[原始接入证据](artifacts/integration/20261020_165011_official_4k/route_evidence.json)。

冻结提交：{'vllm-plugin-FL': '07d9fa0cfdc5f09db178e29508e3c0bcb5c1e908', 'FlagGems': '484413ea0db18221dd9e63e742361d837df9ff13'}。
