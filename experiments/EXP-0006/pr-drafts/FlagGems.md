# 【FlagOS开放计算全球挑战赛S2.赛题2】【LeanScale】实现天数 RoPE 缓存写入及 GEMM 局部融合候选

## PR Category
Operator

## Type of Change
New kernel / Performance

## Description
新增 functional RoPE 输出与 paged KV 写入单次 launch 内核，支持真实 cache strides、负 slot、padding 和 int64 地址。
另提供 gate/up GEMM+SwiGLU、residual+RMSNorm+QKV GEMM 诊断候选；后两者尚未接入服务。Norm 候选保留投影输入前的 dtype 舍入点。

## Issue
LeanScale EXP-0006。

## Progress
源码已推送个人 fork；原型待 GPU 参数搜索和复测。保留所有失败及未加速配置，仅选择通过检查并有实测收益的路径。

## Performance
尚无本轮融合性能结果。已排队进行 RoPE 六组、MLP 四组、Norm 三组参数矩阵，微基准与官方整模结果分开报告。
