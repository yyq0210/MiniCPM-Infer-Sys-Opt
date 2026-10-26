# EXP-0022：4K尾部小M原生投影与补零诊断

实现前假设：EXP18 NONE尾部B8–16中四投影合计44.54%设备kernel时间，down mean93.05us，而B64窗口58.75us；这是不同窗口/shape观察，不是收益。
生产FULL尾部实际padding M8/16。先保留已有原生库GEMM，检查输入补到16/32/64行能否改善小M执行效率，且收益必须覆盖补零成本。
本轮不恢复EXP6的自写norm+QKV或gateup+SwiGLU慢原型，不修改模型/调度/已选代码。

实际调用链：UnquantizedLinearMethod.apply → dispatch_unquantized_gemm → default_unquantized_gemm → torch.nn.functional.linear。
模型hidden2048、intermediate6144、Q16/KV2/head128；预期合并QKV[N,K]=[2560,2048]、O[2048,2048]、gate_up[12288,2048]、down[2048,6144]。
权重BF16；连续[N,K]布局根据create_weights源码推导并在合成张量中显式使用，尚未在新服务中探针记录实际weight stride。
只读checkpoint header验证分离权重shape/dtype，不读取或发布权重数据。随机权重/激活，不是模型请求重放。

固定M=[1,4,8,16,30,32,64,2048]，目标padding=[16,32,64]，仅target>M时测候选；M30是NONE诊断参照，不声称生产FULL使用M30。
完整候选路径为F.pad + 原库linear + 前M行视图，不只计GEMM。M8/16是预先声明的生产尾部主目标。
hot模式8次重用同权重；weights8模式每次轮换8个独立权重矩阵，观察缓存敏感性，不能等同于42层真实工作集。
每种模式旧/新/旧夹测，7样本×8次Graph调用，记录所有配置与退化点。64个基线曲线点、112个候选计时点。
56项BF16与16项FP16数值/Graph检查，共72项：FP32对照、与原库输出比较，改变input/weight后Graph replay与eager严格一致。
固定atol0.01/rtol0.02，TF32关闭；完整模型精度仍按用户暂缓。

初筛门槛：同一投影/M8或16/target在hot和weights8两模式均完整路径耗时下降至少5%，两次夹测漂移均<=3%。
通过只代表进入独立复测，之后仍需更完整数值边界/完整子图/真实引擎接入和正式E2E，不能直接切换代码。
无候选就记录本假设被淘汰；不删慢点、不改阈值挑结果。全部结果自动保存推送。
新40分钟窗口，child最多30分钟，受今日23:35计算截止、23:50保存约束；GPU锁/空闲/端口/代码与脚本哈希前后检查。
