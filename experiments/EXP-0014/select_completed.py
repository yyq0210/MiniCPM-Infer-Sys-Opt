"""Audit both complete EXP14 cases and select the performance baseline, without GPU work."""
from pathlib import Path
import csv
import datetime
import fcntl
import hashlib
import json
import re
import shutil
import statistics
import subprocess

ROOT = Path('/workspace/LeanScale')
REPO = ROOT/'MiniCPM-Infer-Sys-Opt'
EXP = REPO/'experiments/EXP-0014'
WT = ROOT/'worktrees/EXP-0014'


def read(p):
    return json.loads(p.read_text())


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True, timeout=120).strip()


def rows(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def write(p, data):
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')


def main():
    assert not (EXP/'selection.json').exists(), 'Inspect an existing selection instead of replacing it'
    commits = read(EXP/'candidate_commits.json')
    active = read(ROOT/'artifacts/active_inference_plan.json')
    parent = read(EXP.parent/'EXP-0012/selection.json')
    assert active['selected_experiment'] == 'EXP-0012'
    assert active['selected_serving_commits'] == parent['selected_serving_commits']
    assert read(ROOT/'artifacts/EXP-0014-queue.json')['stage'] == 'complete'
    for name, sha in commits.items():
        assert git(WT/name, 'rev-parse', 'HEAD') == sha
        assert not git(WT/name, 'status', '--porcelain')
    gate = read(EXP/'candidate_validation.json')
    assert gate['status'] == 'passed' and gate['candidate_commits'] == commits
    for name, sha in gate['diagnostic_hashes'].items():
        assert digest(Path(name)) == sha
    for repo, files in gate['code_hashes'].items():
        for name, sha in files.items():
            assert digest(WT/repo/name) == sha
    results = []
    for case, prefill, count in [('4k', 4096, 256), ('16k', 16384, 128)]:
        run = Path(read(ROOT/f'artifacts/EXP-0014-latest-{case}.json')['run_directory'])
        status = read(run/'status.json')
        handler = read(run/'completion_handler_status.json')
        assert status['stage'] == 'performance_case_complete' and status['commits'] == commits
        assert handler['status'] == 'pushed' and handler['complete_results'] and handler['remote_verified']
        git(REPO, 'merge-base', '--is-ancestor', handler['commit'], 'HEAD')
        folder = run/'candidate_benchmark'
        route = read(folder/'route_evidence.json')
        assert route['status'] == 'passed' and route['gqa_layers'] == route['fused_runtime_layers'] == 42
        assert route['gqa_launch'] == dict(warps=4, stages=1, segments=16, decode_layout=1, decode_rows=8)
        assert route['kernels'] and route['graphs']
        for item in route['kernels']:
            p = run/item['path']
            assert digest(p) == item['sha256'] and digest(p.with_suffix('.ttir')) == item['ttir_sha256']
            assert item['shared'] == 18432 and item['logical_query_rows'] == 8 and item['page_table_load_elements'] == 2
        for item in route['graphs']:
            assert digest(run/item['path']) == item['sha256']
            assert item['counts'] == dict(fl_rope_kv_dispatch=42, fl_rotary_embedding_dispatch=0, unified_kv_cache_update=0)
        assert read(folder/'smoke.json')['usage']['completion_tokens'] > 0
        raw_path, = (folder/'benchmark_results').glob('raw_runs_*.csv')
        summary_path, = (folder/'benchmark_results').glob('summary_*.csv')
        raw, summary = rows(raw_path), rows(summary_path)
        assert len(raw) == 4 and len(summary) == 1
        assert (folder/'benchmark.log').read_text().count('Failed requests:                         0') == 4
        for row in raw:
            assert row['Run Status'] == 'SUCCESS'
            assert tuple(int(float(row[k])) for k in ['Prefill', 'Decode', 'Conc', 'Num Prompts']) == (prefill, 1024, 64, count)
            assert int(float(row['Successful Requests'])) == count
            assert int(float(row['Total Input Tokens'])) == prefill*count
            assert int(float(row['Total Output Tokens'])) == 1024*count
        parent_item = next(x for x in parent['case_results'] if x['case'] == case)
        parent_folder = ROOT/'artifacts/EXP-0012'/parent_item['run_name']/'candidate_benchmark/benchmark_results'
        parent_raw, = parent_folder.glob('raw_runs_*.csv')
        parent_summary, = parent_folder.glob('summary_*.csv')
        assert digest(parent_raw) == parent_item['raw_sha256']
        assert digest(parent_summary) == parent_item['summary_sha256']
        before, after = rows(parent_summary)[0], summary[0]
        metrics = {}
        for key in ['Total tok/s', 'Output tok/s', 'Mean TTFT (ms)', 'Mean TPOT (ms)', 'Mean ITL (ms)']:
            a, b = float(before[key]), float(after[key])
            assert abs(b-statistics.mean(float(x[key]) for x in raw[1:])) <= .011
            metrics[key] = dict(exp12=a, paged_layout=b, change_percent=(b/a-1)*100)
        assert metrics['Total tok/s']['change_percent'] > 0
        assert metrics['Mean TPOT (ms)']['change_percent'] < 0
        assert metrics['Mean TTFT (ms)']['change_percent'] < 1
        comparison = read(run/'comparison_exp12.json')[0]
        assert comparison['candidate_total_tokens_s'] == float(after['Total tok/s'])
        retained = [float(x['Total tok/s']) for x in raw[1:]]
        assert comparison['candidate_retained_runs'] == retained
        results.append(dict(case=case, run_name=run.name, metrics=metrics,
                            comparison_exp12=comparison, comparison_original=status['performance'][0],
                            accounting=status['scheduler_accounting'],
                            raw_sha256=digest(raw_path), summary_sha256=digest(summary_path),
                            route_sha256=digest(folder/'route_evidence.json'),
                            archive_commit=handler['commit'],
                            checks=dict(four_successful_rounds=True, discard_first=True,
                                        partial_runs_combined=False, route_revalidated=True)))
    selection = dict(selected_experiment='EXP-0014', selected_serving_commits=commits,
                     previous_selection=active, selected_at=datetime.datetime.now().astimezone().isoformat(),
                     basis='Both complete official cases improve total throughput and TPOT versus EXP12; 4K TTFT +0.45%, 16K TTFT -8.05%.',
                     scope='Performance baseline; historical sessions, not interleaved A/B; full task accuracy deferred.',
                     case_results=results, scheduler_candidate='rejected_by_user',
                     next_direction='Original plan: GQA prefill tuning, then later dedicated 4K optimization')
    report = '''# 分页 KV 读取与 8 行 GQA 的完整端到端结果

日期：2026-10-22。队名：LeanScale。EXP-0014 两组官方测评均完整结束，各四轮弃首轮、无失败请求；选择为下一轮性能实验基线。

| 输入长度 | 原始基线 tok/s | EXP-0012 tok/s | EXP-0014 tok/s | 相对上一版 | 相对原始基线 |
|---|---:|---:|---:|---:|---:|
| 4K | 2008.97 | 2936.01 | 3332.27 | +13.50% | +65.87% |
| 16K | 917.27 | 1973.96 | 2181.74 | +10.53% | +137.85% |

| 输入长度 | TTFT 上一版→本版 ms | TPOT 上一版→本版 ms | 本版吞吐 CV |
|---|---:|---:|---:|
| 4K | 5874.60 → 5901.08（+0.45%） | 101.45 → 88.93（-12.34%） | 4.43% |
| 16K | 262412.54 → 241283.65（-8.05%） | 229.76 → 206.35（-10.19%） | 0.102% |

4K 保留三轮吞吐为 3190.65 / 3321.02 / 3485.14；16K 为 2183.60 / 2182.35 / 2179.27。比较来自独立历史服务会话，未做交错 A/B；CV 是三个保留样本的描述统计，不是置信区间。

## 改动及接入证据

在 EXP-0012 的 4 warps / 1 stage / 16 segments 基础上，显式启用 `decode_layout=1, decode_rows=8`。按页加载两个页号后广播到 32 token lanes，减少页表重复读取与地址计算；逻辑查询行由 16 改为 8，匹配每组 8 个 Q heads。完整 KV、因果 attention、原调度、prefill 和既有 RoPE+KV 融合保持。

冻结插件 `82c6a143f6984f946df42125f1a8699799e8045f`；FlagGems `1c526ca99d412e203ed3caa8b1663566976ea38c`。两组服务均确认 42 层 GQA、42 层 RoPE+KV 融合、35 张 FULL_DECODE_ONLY 图。真实编译缓存的 8×128 Q、2 项页表读取、2×16 int64 广播、18 KiB shared 均通过；图中旧独立 KV update 为 0。

EXP-0013 微基准中 decode 完整 GQA 时间减少 22.42%–27.96%，不能与本表整模吞吐提升混用。寄存器 96→83、shared 20→18 KiB 对应驱动预测驻留 blocks 6→7；尚未测动态 achieved occupancy，不能将全部收益归因于 occupancy。

## 调度及能力边界

16K 全四轮平均队列时间 261.727→240.579 秒，抢占仍为 16 次，采样 KV 峰值仍为 99.9%，采样最大 running 仍为 31。计算更快使队列更快消化，但没有增加 KV 容量，也没有消除长文排队。4K 全四轮平均队列时间 8.404→8.339 秒，抢占 0，KV 峰值 62%。队列统计含首轮；吞吐与延迟表为弃首轮统计，口径不同。

这是性能基线选择，完整模型任务精度仍按用户要求暂缓；此前算子数值与 Graph 检查通过。尚不宣称赛事精度验收完成。原始基线、EXP-0012 和所有失败现场保留。

## 下一步

遵照用户澄清：继续原计划的 GQA prefill 参数/实现调优，通过扩大验证再接入引擎、完整测评 4K 和 16K。完成这一轮优化后再重点做 4K 专项，不提前改变当前实验顺序或验收范围。

逐项数值、源文件哈希和归档提交见 `experiments/EXP-0014/selection.json`；两组原始归档及 `RESULTS_4096.md`、`RESULTS_16384.md` 均已推送。
'''
    with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not git(REPO, 'diff', '--cached', '--name-only')
        write(EXP/'selection.json', selection)
        shutil.copy2(__file__, EXP/'select_completed.py')
        (REPO/'docs/gqa_paged_layout_e2e_20261022.md').write_text(report)
        with (EXP/'steps.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=selection['selected_at'], action='select_completed_performance_baseline',
                                    commits=commits, official_cases=['4k', '16k'], gpu_work=False))+'\n')
        with (REPO/'docs/optimization_journal.md').open('a') as f:
            f.write('\n\n### 2026-10-22：EXP-0014 两组完成，选择分页布局性能基线\n\n'
                    '4K 3332.27 tok/s、16K 2181.74 tok/s，相对EXP-0012提升13.50%/10.53%。'
                    '完整四轮、弃首轮、代码/实际路由/请求token数/源哈希/归档提交复核通过。'
                    '4K TTFT近似不变，16K TTFT下降8.05%，两组TPOT改善。'
                    '保持原调度和完整KV；按原计划继续GQA prefill，之后再做4K专项。\n')
        paths = ['experiments/EXP-0014/selection.json', 'experiments/EXP-0014/select_completed.py',
                 'experiments/EXP-0014/steps.jsonl', 'docs/gqa_paged_layout_e2e_20261022.md', 'docs/optimization_journal.md']
        forbidden = re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|lab_inst_[A-Za-z0-9]+|ssh\.flagos\.io')
        for p in paths:
            assert not forbidden.search((REPO/p).read_bytes()), p
        git(REPO, 'add', '--', *paths)
        git(REPO, 'diff', '--cached', '--check')
        git(REPO, 'commit', '--only', '-m', 'experiments: select verified paged GQA E2E performance baseline', '--', *paths)
        git(REPO, 'push', 'origin', 'main')
        head = git(REPO, 'rev-parse', 'HEAD')
        assert git(REPO, 'ls-remote', git(REPO, 'remote', 'get-url', '--push', 'origin'), 'refs/heads/main').split()[0] == head
    assert read(ROOT/'artifacts/active_inference_plan.json') == active
    active.update(previous_serving_commits=active['selected_serving_commits'], selected_serving_commits=commits,
                  selected_experiment='EXP-0014', current_experiment='EXP-0014', selection_document_commit=head,
                  status='Both official cases completed; paged GQA selected as performance baseline; prefill tuning next',
                  next_direction=selection['next_direction'])
    tmp = ROOT/'artifacts/active_inference_plan.json.tmp'
    write(tmp, active)
    tmp.replace(ROOT/'artifacts/active_inference_plan.json')
    print(json.dumps(dict(status='selected', commits=commits, document_commit=head, case_count=len(results))))


if __name__ == '__main__':
    main()
