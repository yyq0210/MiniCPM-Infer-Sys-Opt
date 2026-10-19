"""Create an experiment record before changing code or running a measurement."""
import argparse
from datetime import datetime
from pathlib import Path
import re

parser=argparse.ArgumentParser()
parser.add_argument("--title",required=True)
parser.add_argument("--hypothesis",required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
parent=root/"experiments"
used=[int(m.group(1)) for p in parent.iterdir() if (m:=re.match(r"EXP-(\d+)",p.name))]
n=max(used,default=0)+1
while True:
    experiment=f"EXP-{n:04d}"
    try:
        path=parent/experiment
        path.mkdir()
        break
    except FileExistsError:
        n+=1
now=datetime.now().astimezone().isoformat(timespec="seconds")
content=f"""# {experiment}：{args.title}

创建时间：{now}。状态：planned。

## 假设和实际执行路径

{args.hypothesis}

## 基线与改动

待记录：三个仓库的 commit、工作区状态、改动文件和理由、回退条件。

## 运行命令和环境

待记录：硬件、依赖、完整命令、输入形状、数据集、计时方法。

## 步骤日志

| 时间 | 动作 | 结果/证据 |
| --- | --- | --- |
| {now} | 建立实验记录 | 尚未修改代码或运行 |

## 正确性与失败

未运行。

## 性能结果

未运行。明确区分单算子与整模，保留全部配置和原始数据。

## 结论、代码提交和下一步

待记录。
"""
(path/"README.md").write_text(content)
with (root/"docs/optimization_journal.md").open("a") as f:
    f.write(f"\n- {now}：创建 [{experiment}](../experiments/{experiment}/README.md) — {args.title}，尚未运行。\n")
print(path)
