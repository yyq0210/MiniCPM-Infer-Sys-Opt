# EXP-0001：固定比赛代码与运行环境

状态：完成目标模型路径的运行验证；保留镜像原有依赖约束报告。

| 项目 | 调整前 | 调整后 |
| --- | --- | --- |
| vllm-plugin-FL | main @ 13eb9be | flagos-2026-s2 @ 13eb9be |
| FlagGems | 本地 v5.3.5 @ 4df52d9 | 官方 v5.3.5 @ a7620cc |
| Python 导入 | site-packages 普通安装副本 | /workspace 下的可编辑源码 |
| 厂商运行时 | 镜像内置 CoreX | 保留不变 |

远端 refs 核对：插件 `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`，FlagGems `a7620cc191a0b42e040194622c5758b22a7a25dc`。原始引用保留在原仓库。

```bash
python -m pip install --no-build-isolation --no-deps \
  -e /workspace/vllm-plugin-FL -e /workspace/FlagGems
```

实际硬件为单张 BI-V150 32GB；PyTorch 发行包为 2.10.0+corex.4.5.0，Triton 可导入运行时为 3.6.0，vLLM 为 0.24.0+empty。记录见 environment.json 和 requirements-image.txt。

验证：GPU 张量计算、Triton 内核编译与正确性测试、官方模型服务启动、HTTP /health=200、真实文本生成均通过。此结论仅覆盖本次 BF16 文本推理路径。

## 已知依赖情况

pip-check.txt 保留四条镜像原有报告：xgrammar 找不到标准 triton distribution 元数据；compressed-tensors、fastapi 和 opencv-python-headless 不满足 vLLM wheel 的版本声明。实际 Triton 模块来自厂商环境，不能为消除元数据提示直接安装上游 Triton。未宣称所有功能的依赖已验证。

## 复现依据

以 Git 提交为版本依据。FlagGems 可编辑构建的包版本可能带日期后缀，不代表使用了不同源码。原始目录 `/workspace/experiments/codex_20261019/` 中保留安装日志与旧状态。
