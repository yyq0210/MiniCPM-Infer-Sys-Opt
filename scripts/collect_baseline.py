"""Archive the already-running baseline; never launches GPU work or pushes Git."""
import argparse
import csv
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "experiments/EXP-0003-baseline"


def sanitize(text):
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP_REDACTED]", text)
    text = re.sub(r"uuid=[^,)]+", "uuid=[REDACTED]", text)
    text = re.sub(r"(?i)(authorization\s*[:=]\s*bearer\s+)\S+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)([?&](?:token|key|secret|signature)=)[^&\s]+", r"\1[REDACTED]", text)
    return text


def read_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def collect(source):
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    state = json.loads((source / "status.json").read_text())
    stage = state.get("stage", "unknown")
    live = DEST / "artifacts"
    live.mkdir(parents=True, exist_ok=True)
    for rel in ["status.json", "runner.log", "baseline_official/commands.json",
                "baseline_official/smoke.json", "baseline_official/server.log",
                "baseline_official/benchmark.log"]:
        path = source / rel
        if path.is_file():
            target = live / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(sanitize(path.read_text(errors="replace")))
    result_dir = source / "baseline_official/benchmark_results"
    for path in result_dir.glob("*.csv"):
        target = live / "baseline_official/benchmark_results" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())

    summaries = sorted(result_dir.glob("summary_*.csv"))
    raw = sorted(result_dir.glob("raw_runs_*.csv"))
    summary_rows = read_rows(summaries[-1]) if summaries else []
    raw_rows = read_rows(raw[-1]) if raw else []
    valid_cases = set()
    try:
        valid_cases = {int(float(row["Prefill"])) for row in summary_rows}
        raw_ok = len(raw_rows) == 8 and all(
            int(float(row["Successful Requests"])) == int(float(row["Num Prompts"]))
            and float(row["Total Input Tokens"]) > 0
            and float(row["Total Output Tokens"]) > 0
            for row in raw_rows
        )
    except (KeyError, TypeError, ValueError):
        raw_ok = False
    complete = stage == "throughput_complete" and len(summary_rows) == 2 and valid_cases == {4096,16384} and raw_ok
    validation = {"captured_at":now, "source_stage":stage,
                  "summary_rows":len(summary_rows), "raw_rows":len(raw_rows),
                  "all_expected_results_present":complete,
                  "accuracy":"not_run", "optimized_end_to_end":"not_run"}
    (live / "validation.json").write_text(json.dumps(validation, indent=2))

    state_log = DEST / "state_history.jsonl"
    old = [json.loads(line) for line in state_log.read_text().splitlines()] if state_log.exists() else []
    if not old or old[-1]["source_stage"] != stage or old[-1]["raw_rows"] != len(raw_rows):
        with state_log.open("a") as f:
            f.write(json.dumps(validation,ensure_ascii=False)+"\n")
    label = "完整性校验通过" if complete else ("失败/中止" if stage == "failed" else "尚未得到完整结果")
    text = f"# EXP-0003 状态快照\n\n采集时间：{now}。\n\n- 运行状态：`{stage}`\n- 判定：{label}\n- 官方汇总：{len(summary_rows)}/2 行\n- 原始轮次：{len(raw_rows)}/8 行\n- 数学精度：未运行\n- 优化后整模 A/B：未运行\n\n"
    if summary_rows:
        text += "| Prefill | Total tok/s | Mean TTFT (ms) |\n| ---: | ---: | ---: |\n"
        for row in summary_rows:
            text += f"| {row.get('Prefill')} | {row.get('Total tok/s')} | {row.get('Mean TTFT (ms)')} |\n"
    text += "\n本文件是采集时的快照，GitHub 不会随服务器自动刷新。原始结果和校验数据见 artifacts/。\n"
    (DEST / "STATUS.md").write_text(text)
    hashes = {str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(live.rglob("*")) if p.is_file()}
    (DEST / "artifact_manifest.json").write_text(json.dumps(hashes,indent=2))
    print(json.dumps(validation,ensure_ascii=False),flush=True)
    return stage


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,default=Path("/workspace/experiments/codex_20261019"))
    parser.add_argument("--wait",action="store_true",help="Wait only for the existing run, then archive its final state")
    args=parser.parse_args()
    lock=(DEST / ".collect.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        stage=collect(args.source)
        if not args.wait or stage in {"throughput_complete","failed"}:
            break
        state=json.loads((args.source/"status.json").read_text())
        pid=state.get("runner_pid")
        stat=Path(f"/proc/{pid}/stat")
        if pid and (not stat.exists() or stat.read_text().split(") ",1)[1].startswith("Z")):
            with (DEST/"STATUS.md").open("a") as f:
                f.write("\n采集器发现 runner 已退出但未记录终态，需要人工检查；不判定实验成功。\n")
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
