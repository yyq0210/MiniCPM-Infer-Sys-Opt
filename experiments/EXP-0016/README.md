# EXP-0016：GQA prefill stage1接入官方端到端测评

假设先于修改记录：EXP-0015的rows64/warps4/stage1/tile32已通过独立复测、42项扩大数值/Graph检查和36个计时点。
在EXP-0014现有最佳引擎上增加prefill显式参数，decode分页页号广播/8行/4warps/1stage/16segments、RoPE+KV融合及原调度保持。
FlagGems沿用EXP-0015冻结提交fd209df7，其通用默认prefill_stages仍为2；只有此插件明确选stage1。
插件反向去除新参数和新日志必须与EXP-0014逐字一致；supports/fallback与形状边界不变。

实际接入门槛：42层GQA新参数日志、42层RoPE+KV融合、35张FULL_DECODE_ONLY图；独立服务缓存中所有attention均CoreX/4warps/1stage。
decode必须有8行Q、两页号与2×16广播、18KiB shared；prefill必须有64行Q、tile32的页号/K读取和QK张量、32KiB shared。两类实际编译证据缺一不可。
CPU合成日志/IR拒绝检查仅验证检查器，真实服务检查仍在每组正式测评前后执行。

官方参数保持：4K[4096,1024,64,256]、16K[16384,1024,64,128]，每组四轮弃首轮，原benchmark哈希不变。
对照EXP-0014与原始完整基线；历史会话不是交错A/B，4K父CV4.43%，小幅收益需谨慎归因。
4K吞吐若退化超过3%则暂停16K分析。两组完整判定前，EXP-0014仍为已选性能版本。
完整模型精度按用户要求暂缓，数值/Graph验证不可省略。今晚23:35计算截止、23:50保存；55/90分钟完整case起跑预算。
完成后评估本轮GQA方案，再重点做4K专项。所有失败、原始结果和文档保存并推送fork；不创建上游PR。
