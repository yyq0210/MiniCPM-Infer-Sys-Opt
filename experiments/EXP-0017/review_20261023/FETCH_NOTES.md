# 结果读取记录

2026-10-23：服务器 SSH 返回 publickey 认证失败；未启动任何 GPU 工作。
web 工具无法抓取 GitHub 页面；curl 可读取公开 raw 结果和 main 分支固定提交。
GitHub tree API 请求返回 HTTP403，首次离线审核脚本在下载前失败。
改用同一个固定提交下已公开的 source_hashes.json 和 script_hashes.json 清单读取结果文件，不使用凭据，不更换提交，不改结果。
审核报告不声称核对了 Git blob SHA；记录下载 SHA256，并核对原始数据源 SHA256 和冻结脚本 SHA256。

首次审核把35个CUDA Graph捕获尺寸误当成35份computation_graph.py源码，断言失败。
检查四个会话的真实归档：每组都只有1份编译计算图源码，而日志完整显示35/35捕获。
修正为分别检查非空计算图源码及35/35捕获，保留这一审核脚本错误记录；实验数据和运行代码均未修改。
