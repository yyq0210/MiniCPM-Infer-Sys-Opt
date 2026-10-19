"""One-shot completion handler for an already-running EXP-0004 experiment.

No GPU work and no new benchmark runs. Archive before attempting Git publication;
a push failure cannot discard local results. Never pull, rebase or force-push.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT/'experiments/EXP-0004'
BASE = Path('/workspace/experiments/codex_20261019/baseline_official')
CASES = {(4096,1024,64,256),(16384,1024,64,128)}
TERMINAL = {'performance_complete', 'failed'}


def read_json(path):
    for attempt in range(5):
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            if attempt == 4:
                raise
            time.sleep(.2)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)


def sanitize(text):
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP_REDACTED]', text)
    text = re.sub(r'uuid=[^,)]+', 'uuid=[REDACTED]', text)
    text = re.sub(r'(?i)(authorization\s*[:=]\s*bearer\s+)\S+', r'\1[REDACTED]', text)
    return text


def running(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
        return fields[0] != 'Z' and b'run_rope_e2e.py' in cmd
    except FileNotFoundError:
        return False


def selected_files(run):
    for name in ['status.json','validation.log','validation.command.json',
                 'validation/results.json','comparison.json','baseline_combined.json']:
        path = run/name
        if path.is_file():
            yield path
    for folder in ['candidate_preflight','baseline_16k','candidate_benchmark']:
        for name in ['server.log','server.command.json','smoke.json','route_evidence.json',
                     'benchmark.log','benchmark.command.json']:
            path = run/folder/name
            if path.is_file():
                yield path
        yield from sorted((run/folder/'benchmark_results').glob('*.csv'))


def csv_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def verify_rows(raw, summary):
    def key(r):
        return tuple(int(float(r[k])) for k in ('Prefill','Decode','Conc','Num Prompts'))
    assert Counter(key(r) for r in raw) == Counter({c:4 for c in CASES})
    assert len(summary) == 2 and {key(r) for r in summary} == CASES
    for r in raw:
        prefill, decode, _, n = key(r)
        assert r['Run Status'] == 'SUCCESS'
        assert int(float(r['Successful Requests'])) == n
        assert int(float(r['Total Input Tokens'])) == prefill*n
        assert int(float(r['Total Output Tokens'])) == decode*n
    for r in summary:
        trials = [v for v in raw if key(v) == key(r)][1:]
        for metric in ['Total tok/s','Mean TTFT (ms)']:
            expected = sum(float(v[metric]) for v in trials)/3
            assert abs(float(r[metric])-expected) <= .011, (metric, r, expected)


def archive(run, state):
    dest = EXP/'artifacts/runs'/run.name
    source_hashes = {}
    for source in selected_files(run):
        rel = source.relative_to(run)
        data = source.read_bytes()
        target = dest/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(sanitize(data.decode(errors='replace')))
        source_hashes[str(rel)] = hashlib.sha256(data).hexdigest()
    # Preserve original baseline 4K CSVs alongside the resumed 16K run.
    for source in sorted((BASE/'benchmark_results').glob('*.csv')):
        target = dest/'retained_baseline_4k'/source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    write_json(dest/'source_hashes.json', source_hashes)
    integrity = {'complete': False, 'accuracy': 'deferred_by_user'}
    report = []
    if state.get('stage') == 'performance_complete':
        try:
            baseline = read_json(run/'baseline_combined.json')
            verify_rows(baseline['raw'], baseline['summary'])
            result_dir = run/'candidate_benchmark/benchmark_results'
            raw = csv_rows(sorted(result_dir.glob('raw_runs_*.csv'))[-1])
            summary = csv_rows(sorted(result_dir.glob('summary_*.csv'))[-1])
            verify_rows(raw, summary)
            report = read_json(run/'comparison.json')
            assert len(report) == 2 and {r['prefill'] for r in report} == {4096,16384}
            for item in report:
                old = next(r for r in baseline['summary'] if int(r['Prefill']) == item['prefill'])
                new = next(r for r in summary if int(r['Prefill']) == item['prefill'])
                for metric, old_key, new_key in [
                    ('Total tok/s','baseline_total_tokens_s','candidate_total_tokens_s'),
                    ('Mean TTFT (ms)','baseline_ttft_ms','candidate_ttft_ms')]:
                    assert float(old[metric]) == item[old_key]
                    assert float(new[metric]) == item[new_key]
            integrity.update(complete=True, baseline_raw_runs=8, candidate_raw_runs=8)
        except Exception as exc:
            integrity['error'] = repr(exc)
    write_json(dest/'completion_validation.json', integrity)
    stamp = datetime.now().astimezone().isoformat(timespec='seconds')
    lines = ['# EXP-0004 官方性能实验结果', '', f'归档时间：{stamp}。', '',
             f"- 运行阶段：{state.get('stage')}。",
             f"- 完整性检查：{'通过' if integrity['complete'] else '尚未通过/未完成'}。",
             '- 官方负载：[4096,1024,64,256] 与 [16384,1024,64,128]；每组四轮，舍弃第一轮。',
             '- 基线 4K 使用中止前已完整完成的四轮；16K 使用恢复后的四轮，两个服务会话的来源均保留。',
             '- 完整 MATH500 精度按用户要求暂缓，不声明赛事精度达标。',
             f'- 原始结果、日志、命令及哈希：[本次归档](artifacts/runs/{run.name}/)。', '']
    if integrity['complete']:
        lines += ['| Prefill | 基线 total tok/s | 优化 total tok/s | 吞吐变化 | 基线 TTFT ms | 优化 TTFT ms | TTFT 变化 |',
                  '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in report:
            lines.append(f"| {r['prefill']} | {r['baseline_total_tokens_s']:.2f} | {r['candidate_total_tokens_s']:.2f} | {r['throughput_change_percent']:+.2f}% | {r['baseline_ttft_ms']:.2f} | {r['candidate_ttft_ms']:.2f} | {r['ttft_change_percent']:+.2f}% |")
        lines += ['', 'TTFT 变化为正表示延迟变差；逐轮数据和变异系数见 comparison.json。局部内核加速不替代此表的整模结果。']
    else:
        lines += ['未得到可报告的完整官方 A/B。保留失败现场，不把不完整数据写成成功。',
                  '', '状态/错误：'+sanitize(str(state.get('error') or integrity.get('error') or '运行尚未结束'))]
    (EXP/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (EXP/'STATUS.md').write_text(f"# EXP-0004 状态\n\n更新：{stamp}。阶段：{state.get('stage')}。\n\n[结果与完整性检查](RESULTS.md)。完整精度暂缓。\n")
    hashes = {str(p.relative_to(EXP)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((EXP/'artifacts').rglob('*')) if p.is_file()}
    write_json(EXP/'artifact_manifest.json', hashes)
    return integrity


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args], text=True,
                                   stderr=subprocess.STDOUT, timeout=120).strip()


def publish(run):
    assert git('branch','--show-current') == 'main'
    assert not git('diff','--cached','--name-only'), 'Existing staged changes; leaving them untouched'
    paths = ['experiments/EXP-0004/STATUS.json','experiments/EXP-0004/STATUS.md',
             'experiments/EXP-0004/RESULTS.md','experiments/EXP-0004/artifact_manifest.json',
             'experiments/EXP-0004/artifacts/runs/'+run.name]
    forbidden = re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|lab_inst_[A-Za-z0-9]+|ssh\.flagos\.io')
    for rel in paths:
        p = ROOT/rel
        for item in p.rglob('*') if p.is_dir() else [p]:
            if item.is_file():
                assert not forbidden.search(item.read_bytes()), 'Sensitive data detected: '+str(item)
    git('add','--',*paths)
    git('diff','--cached','--check')
    if git('diff','--cached','--name-only'):
        git('commit','-m','experiments: archive completed official RoPE performance run')
    head = git('rev-parse','HEAD')
    push_url = git('remote','get-url','--push','origin')
    failures = []
    for attempt in range(3):
        try:
            git('push','origin','main')
            remote = git('ls-remote',push_url,'refs/heads/main').split()[0]
            assert remote == head, (remote, head)
            return {'status':'pushed','commit':head,'remote_verified':True}
        except Exception as exc:
            failures.append(str(exc))
            if attempt < 2:
                time.sleep(30)
    return {'status':'push_failed_local_saved','commit':head,'errors':failures}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--once',action='store_true',help='Archive current snapshot; never push')
    parser.add_argument('--publish',action='store_true',help='Publish after terminal state only')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    assert run.parent == Path('/workspace/LeanScale/artifacts/EXP-0004')
    lock = (EXP/'.finish.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    receipt = run/'completion_handler_status.json'
    write_json(receipt, {'status':'waiting','pid':os.getpid(),'run_directory':str(run),
                         'publish_on_finish':args.publish and not args.once})
    deadline = time.monotonic()+200000
    last_snapshot = 0
    try:
        while True:
            state = read_json(run/'status.json')
            alive = running(state['pid'])
            terminal = state.get('stage') in TERMINAL
            if not alive and not terminal:
                state.update(stage='failed',error='Coordinator exited without a terminal status; original status retained')
                terminal = True
            if time.monotonic() > deadline:
                state.update(stage='watcher_timeout',error='Completion handler wait exceeded 200000 seconds; experiment process was not stopped')
                terminal = True
            if args.once or time.monotonic()-last_snapshot >= 60 or (terminal and not alive):
                integrity = archive(run,state)
                last_snapshot = time.monotonic()
            if args.once or (terminal and not alive) or state['stage']=='watcher_timeout':
                break
            time.sleep(15)
        result = {'status':'local_saved','experiment_stage':state['stage'],
                  'complete_results':integrity['complete'],'local_archive':str(EXP/'artifacts/runs'/run.name)}
        if args.publish and not args.once:
            try:
                result.update(publish(run))
            except Exception as exc:
                result.update(status='publish_failed_local_saved',error=sanitize(str(exc)))
                traceback.print_exc()
        write_json(receipt,result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except BaseException as exc:
        write_json(receipt,{'status':'handler_failed_raw_data_retained','error':sanitize(repr(exc)),
                            'raw_directory':str(run)})
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
