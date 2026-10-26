"""One-shot completion handler for an already-running EXP-0016 experiment.

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

ROOT = Path('/workspace/LeanScale/MiniCPM-Infer-Sys-Opt')
EXP = ROOT/'experiments/EXP-0016'
BASE = Path('/workspace/experiments/codex_20261019/baseline_official')
CASES = {(4096,1024,64,256),(16384,1024,64,128)}
TERMINAL = {'performance_case_complete', 'failed'}


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


def running(pid, process_start_ticks=None):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
        return (fields[0] != 'Z' and b'run_split_e2e.py' in cmd
                and (process_start_ticks is None or fields[19] == str(process_start_ticks)))
    except FileNotFoundError:
        return False


def selected_files(run):
    for name in ['status.json','validation.log','validation.command.json',
                 'validation/results.json','comparison.json','baseline_combined.json',
                 'resume_provenance.json','environment_check.json','stage_history.jsonl',
                 'progress.json','comparison_exp14.json','exp14_reference.json','integration.json','integration.log','integration.command.json',
                 'candidate_validation.json','parent_reference_verification.json','coordinator.log','scheduler_comparison.json']:
        path = run/name
        if path.is_file():
            yield path
    for folder in ['candidate_preflight','baseline_16k','candidate_benchmark']:
        for name in ['server.log','server.command.json','smoke.json','route_evidence.json',
                     'benchmark.log','benchmark.command.json','metrics_before.prom','metrics_after.prom']:
            path = run/folder/name
            if path.is_file():
                yield path
        yield from sorted((run/folder/'benchmark_results').glob('*.csv'))
    yield from sorted((run/'retained_baseline').rglob('*.csv'))


def csv_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def verify_rows(raw, summary, cases=CASES):
    def key(r):
        return tuple(int(float(r[k])) for k in ('Prefill','Decode','Conc','Num Prompts'))
    assert Counter(key(r) for r in raw) == Counter({c:4 for c in cases})
    assert len(summary) == len(cases) and {key(r) for r in summary} == cases
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
    counts = {}
    for path in sorted((run/'candidate_benchmark/benchmark_results').glob('raw_runs_*.csv')):
        for row in csv_rows(path):
            if row.get('Run Status') == 'SUCCESS':
                case = row['Prefill']
                counts[case] = counts.get(case, 0)+1
    write_json(run/'progress.json', {
        'updated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'completed_candidate_rounds': counts, 'required_rounds_per_case': 4,
        'stage': state.get('stage'),
    })
    source_hashes = {}
    for source in selected_files(run):
        rel = source.relative_to(run)
        data = source.read_bytes()
        target = dest/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(sanitize(data.decode(errors='replace')).replace('\r\n','\n'))
        source_hashes[str(rel)] = hashlib.sha256(data).hexdigest()
    # Preserve original baseline 4K CSVs alongside the resumed 16K run.
    for source in sorted((BASE/'benchmark_results').glob('*.csv')):
        target = dest/'retained_baseline_4k'/source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes().replace(b'\r\n',b'\n'))
    write_json(dest/'source_hashes.json', source_hashes)
    integrity = {'complete': False, 'accuracy': 'deferred_by_user'}
    report = []
    if state.get('stage') == 'performance_case_complete':
        try:
            baseline = read_json(run/'baseline_combined.json')
            verify_rows(baseline['raw'], baseline['summary'])
            result_dir = run/'candidate_benchmark/benchmark_results'
            raw = csv_rows(sorted(result_dir.glob('raw_runs_*.csv'))[-1])
            summary = csv_rows(sorted(result_dir.glob('summary_*.csv'))[-1])
            selected_cases = {tuple(c) for c in state['selected_cases']}
            verify_rows(raw, summary, selected_cases)
            report = read_json(run/'comparison.json')
            assert len(report) == len(selected_cases) and {r['prefill'] for r in report} == {c[0] for c in selected_cases}
            for item in report:
                old = next(r for r in baseline['summary'] if int(r['Prefill']) == item['prefill'])
                new = next(r for r in summary if int(r['Prefill']) == item['prefill'])
                for metric, old_key, new_key in [
                    ('Total tok/s','baseline_total_tokens_s','candidate_total_tokens_s'),
                    ('Mean TTFT (ms)','baseline_ttft_ms','candidate_ttft_ms')]:
                    assert float(old[metric]) == item[old_key]
                    assert float(new[metric]) == item[new_key]
            integrity.update(complete=True, baseline_raw_runs=8, candidate_raw_runs=len(raw), completed_cases=state['selected_cases'], all_cases_complete=False)
        except Exception as exc:
            integrity['error'] = repr(exc)
    write_json(dest/'completion_validation.json', integrity)
    stamp = datetime.now().astimezone().isoformat(timespec='seconds')
    lines = ['# EXP-0016 官方单组性能结果：GQA prefill 单级流水', '', f'归档时间：{stamp}。', '',
             f"- 运行阶段：{state.get('stage')}。",
             f"- 完整性检查：{'通过' if integrity['complete'] else '尚未通过/未完成'}。",
             f"- 本次完整负载组：{state.get('selected_cases')}；四轮，舍弃第一轮。整体两组需分别完成并校验版本后汇总。",
             '- 原始基线 4K 使用中止前已完整完成的四轮；16K 使用恢复后的四轮，两个服务会话的来源均保留。',
             '- 本次候选在 EXP-0014 融合版本上显式选择 GQA prefill 单级流水，保持原调度；下表对照原始基线，增量收益另见 comparison_exp14.json。',
             '- 完整 MATH500 精度按用户要求暂缓，不声明赛事精度达标。',
             f'- 原始结果、日志、命令及哈希：[本次归档](artifacts/runs/{run.name}/)。', '']
    if integrity['complete']:
        lines += ['| Prefill | 基线 total tok/s | 优化 total tok/s | 吞吐变化 | 基线 TTFT ms | 优化 TTFT ms | TTFT 变化 |',
                  '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in report:
            lines.append(f"| {r['prefill']} | {r['baseline_total_tokens_s']:.2f} | {r['candidate_total_tokens_s']:.2f} | {r['throughput_change_percent']:+.2f}% | {r['baseline_ttft_ms']:.2f} | {r['candidate_ttft_ms']:.2f} | {r['ttft_change_percent']:+.2f}% |")
        lines += ['', 'TTFT 变化为正表示延迟变差；逐轮数据和变异系数见 comparison.json。局部内核加速不替代此表的整模结果。']
        incremental = read_json(run/'comparison_exp14.json')
        lines += ['', '## 相对 EXP-0014 的 GQA 增量', '',
                  '| Prefill | EXP-0014 tok/s | GQA tok/s | 变化 |',
                  '| ---: | ---: | ---: | ---: |']
        for r in incremental:
            lines.append(f"| {r['prefill']} | {r['baseline_total_tokens_s']:.2f} | {r['candidate_total_tokens_s']:.2f} | {r['throughput_change_percent']:+.2f}% |")
        lines += ['', '两个服务会话的对照，不是交错 A/B。结合逐轮波动判断；4K 的 EXP-0014 CV 为 4.43%，小幅差异不能单凭本表归因于GQA参数。']
        accounting = read_json(run/'scheduler_comparison.json')
        lines += ['', '## 调度指标（全部四轮）', '',
                  '| 指标 | EXP-0014 | GQA 候选 |',
                  '| --- | ---: | ---: |']
        before, after = accounting['reference'], accounting['candidate']
        for title, key in [
            ('平均排队秒数', 'mean_queue_seconds_all_four_rounds'),
            ('日志采样最大运行请求数', 'peak_running_sampled'),
            ('日志采样最大等待请求数', 'peak_waiting_sampled'),
            ('日志采样最大 KV 占用百分比', 'peak_cache_usage_percent_sampled'),
        ]:
            lines.append(f"| {title} | {before[key]} | {after[key]} |")
        lines.append(
            f"| 抢占计数增量 | {before['counter_delta']['num_preemptions_total']} "
            f"| {after['counter_delta']['num_preemptions_total']} |")
        lines += ['', '计数器增量包含首轮；吞吐汇总丢弃首轮，两者口径不同。'
                  '采样峰值不是连续观测的精确峰值。保持原调度，劣化调度的准入日志必须不存在；原始数据见 scheduler_comparison.json。']
    else:
        lines += ['未得到可报告的完整官方 A/B。保留失败现场，不把不完整数据写成成功。',
                  '', '状态/错误：'+sanitize(str(state.get('error') or integrity.get('error') or '运行尚未结束'))]
    (EXP/('RESULTS_'+str(state['selected_cases'][0][0])+'.md')).write_text('\n'.join(lines)+'\n')
    (EXP/'STATUS.md').write_text(f"# EXP-0016 状态\n\n更新：{stamp}。阶段：{state.get('stage')}。\n\n[4K 结果](RESULTS_4096.md) / [16K 结果](RESULTS_16384.md)。完整精度暂缓。\n")
    if state.get('stage') in TERMINAL:
        write_json(EXP/'STATUS.json', state)
    hashes = {str(p.relative_to(EXP)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((EXP/'artifacts').rglob('*')) if p.is_file()}
    write_json(EXP/'artifact_manifest.json', hashes)
    return integrity


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args], text=True,
                                   stderr=subprocess.STDOUT, timeout=120).strip()


def _publish_unlocked(run):
    assert git('branch','--show-current') == 'main'
    assert not git('diff','--cached','--name-only'), 'Existing staged changes; leaving them untouched'
    paths = ['experiments/EXP-0016/STATUS.json','experiments/EXP-0016/STATUS.md',
             'experiments/EXP-0016/RESULTS_'+str(read_json(run/'status.json')['selected_cases'][0][0])+'.md','experiments/EXP-0016/artifact_manifest.json',
             'experiments/EXP-0016/artifacts/runs/'+run.name]
    forbidden = re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|lab_inst_[A-Za-z0-9]+|ssh\.flagos\.io')
    for rel in paths:
        p = ROOT/rel
        for item in p.rglob('*') if p.is_dir() else [p]:
            if item.is_file():
                assert not forbidden.search(item.read_bytes()), 'Sensitive data detected: '+str(item)
    git('add','--',*paths)
    git('diff','--cached','--check')
    if git('diff','--cached','--name-only'):
        stage = read_json(ROOT/'experiments/EXP-0016/STATUS.json').get('stage')
        message = ('experiments: archive completed official GQA prefill-stage1 performance case'
                   if stage == 'performance_case_complete'
                   else 'experiments: archive interrupted official GQA prefill-stage1 performance case')
        git('commit','--only','-m',message,'--',*paths)
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


def publish(run):
    with Path('/workspace/LeanScale/artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _publish_unlocked(run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--once',action='store_true',help='Archive current snapshot; never push')
    parser.add_argument('--publish',action='store_true',help='Publish after terminal state only')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    assert run.parent == Path('/workspace/LeanScale/artifacts/EXP-0016')
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
            alive = running(state['pid'], state.get('process_start_ticks'))
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
