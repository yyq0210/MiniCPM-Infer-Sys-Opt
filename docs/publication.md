# 发布记录

队名：LeanScale。用户已明确授权将推理源码、算子源码和实验文档分别推送到指定个人仓库。

当前准备的分支：

- yyq0210/vllm-plugin-FL：`leanscale/flagos-s2-baseline`，官方比赛提交。
- yyq0210/FlagGems：`leanscale/flagos-s2-baseline`（固定官方标签）、`leanscale/iluvatar-rope-v1`（原型）。
- yyq0210/MiniCPM-Infer-Sys-Opt：`main`。

认证检查：当前 GitHub connector 账户为 YuYanJ02，对三个目标仓库仅有读取权限。本机和服务器现有 SSH 均未获得 GitHub 公钥认证；本机现有 HTTPS credential 的 push --dry-run 也认证失败。

在认证恢复前，不能声称远端已推送、PR 已创建。服务器保存本地提交和 Git bundle；后续成功推送后在此追加远端 SHA 和 PR 链接。

## 已准备的本地提交

| 仓库 | 提交 | 内容 |
| --- | --- | --- |
| vllm-plugin-FL | 13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0 | 官方推理基线 |
| FlagGems | 3cee7d462b76aefa4e19961b20884044b6389a35 | 独立 RoPE 原型 |

文档根提交：b85533cc0ce9e5fde7aedabf5fb0a82ed1861204，之后继续追加采集与发布记录。

服务器 artifacts 目录保存可校验的 Git bundle；算子 bundle 是相对官方 a7620cc 的增量，需要目标仓库已有该基线。scripts/publish.sh 可在具备有效 GitHub 凭据后一次性推送。

## 2026-10-19T19:33:38+08:00 身份恢复与源码推送

用户已将服务器专用公钥加入 yyq0210；GitHub SSH 返回 yyq0210，身份验证通过。仅为三个项目配置专用推送凭据，不在仓库中保存密钥或服务器连接信息。

以下分支已推送，并通过远端 refs 核对：

| 仓库 | 分支 | 远端提交 |
| --- | --- | --- |
| [vllm-plugin-FL](https://github.com/yyq0210/vllm-plugin-FL/tree/leanscale/flagos-s2-baseline) | leanscale/flagos-s2-baseline | 13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0 |
| [FlagGems](https://github.com/yyq0210/FlagGems/tree/leanscale/flagos-s2-baseline) | leanscale/flagos-s2-baseline | a7620cc191a0b42e040194622c5758b22a7a25dc |
| [FlagGems](https://github.com/yyq0210/FlagGems/tree/leanscale/iluvatar-rope-v1) | leanscale/iluvatar-rope-v1 | 3cee7d462b76aefa4e19961b20884044b6389a35 |

本次文档提交包含采集时的 EXP-0003 快照。基线尚未完成，采集器继续等待当前任务，不自动推送。

PR 尚未创建。算子原型的比赛格式标题和正文见 [PR 草稿](../pr/flaggems-rope-v1.md)，[分支对比](https://github.com/yyq0210/FlagGems/compare/leanscale/flagos-s2-baseline...leanscale/iluvatar-rope-v1)。推理分支当前只有固定官方基线，没有优化差异，不创建空 PR。
