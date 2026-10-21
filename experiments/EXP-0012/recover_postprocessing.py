"""CPU-only recovery of an already complete four-round case; retain failure evidence."""
from pathlib import Path
import ast
import copy
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile

ROOT = Path('/workspace/LeanScale')
REPO = ROOT/'MiniCPM-Infer-Sys-Opt'
EXP = REPO/'experiments/EXP-0012'
RUN = ROOT/'artifacts/EXP-0012/20261021_162416_official_4k'
STAGING = Path(__file__).resolve().parent
REV = EXP/'revisions/original_scheduler_postprocessing'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write_json(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')

def git(*args):
    return subprocess.check_output(['git','-C',str(REPO),*args], text=True, stderr=subprocess.STDOUT, timeout=120).strip()

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    # Both execution and publishing are idle; do not patch a running experiment.
    lock = (ROOT/'../experiments/codex_20261019/runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    docs_lock = (ROOT/'artifacts/docs-publish.lock').open('a')
    fcntl.flock(docs_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for receipt in [RUN/'status.json', ROOT/'artifacts/EXP-0012-queue.json']:
        d = json.loads(receipt.read_text())
        p = Path('/proc')/str(d['pid'])
        if p.exists():
            st = (p/'stat').read_text().rsplit(') ',1)[1].split()
            assert st[0] == 'Z' or st[19] != d['process_start_ticks']
    assert json.loads((RUN/'completion_handler_status.json').read_text())['status'] == 'pushed'
    assert git('branch','--show-current') == 'main'
    assert not git('status','--porcelain'), 'Preserve other work; investigate before deployment'
    assert digest(EXP/'run_split_e2e.py') == 'e655d53827c32d24554dca8863d98b793a78f4301f03d36a8983f56c4693b59d'
    state = json.loads((RUN/'status.json').read_text())
    assert state['stage'] == 'failed' and state['error'] == "ValueError('No scheduler admission telemetry')"
    assert not REV.exists()
    originals = [p for p in RUN.rglob('*') if p.is_file() and (
        p.parent == RUN or p.is_relative_to(RUN/'candidate_benchmark'))]
    original_hashes = {str(p.relative_to(RUN)): digest(p) for p in originals}
    REV.mkdir(parents=True)
    stamp = datetime.datetime.now().astimezone().isoformat()
    with (EXP/'steps.jsonl').open('a') as f:
        f.write(json.dumps({'time':stamp,'action':'repair_postprocessing_before_16k',
            'reason':'All four 4K rounds succeeded; inherited EXP-0010 admission telemetry requirement failed after measurement.',
            'policy':'Preserve raw status/logs/CSVs and failure archive; offline revalidate complete case; no partial stitching.'})+'\n')
    old = load_module('old_accounting', EXP/'analyze_schedule.py')
    parent = Path(json.loads((RUN/'exp6_reference.json').read_text())['source'])
    try:
        old.compare(parent,RUN/'candidate_benchmark',REV/'unexpected_old_success.json')
    except ValueError as e:
        assert str(e) == 'No scheduler admission telemetry'
    else:
        raise AssertionError('Original failure did not reproduce')
    for name in ['analyze_schedule.py','run_split_e2e.py','finish_split.py']:
        ast.parse((STAGING/name).read_text())
        shutil.copy2(EXP/name,REV/('before_'+name))
        shutil.copy2(STAGING/name,EXP/name)
    shutil.copy2(__file__,EXP/'recover_postprocessing.py')
    sys.path.insert(0,str(EXP))
    runner = load_module('recovery_runner',EXP/'run_split_e2e.py')
    finish = load_module('recovery_finish',EXP/'finish_split.py')
    analysis = load_module('recovery_analysis',EXP/'analyze_schedule.py')
    runner.RUN = RUN
    cases = {(4096,1024,64,256)}
    raw, summary = runner.validate_benchmark(RUN/'candidate_benchmark',cases)
    finish.verify_rows(raw,summary,cases)
    assert digest(runner.BENCH_FILE) == runner.BENCH_SHA
    assert state['commits'] == runner.CANDIDATE_COMMITS
    checks = ['original_failure_reproduced','complete_four_rounds_and_counts','official_summary_discard_first']
    for label,bad_raw in [('partial_rounds',raw[:-1]),('failed_request',copy.deepcopy(raw))]:
        if label == 'failed_request': bad_raw[2]['Successful Requests']='255'
        try: finish.verify_rows(bad_raw,summary,cases)
        except AssertionError: checks.append('reject_'+label)
        else: raise AssertionError(label)
    with tempfile.TemporaryDirectory(prefix='exp12-recovery-') as tmp:
        t=Path(tmp)
        shutil.copy2(RUN/'candidate_benchmark/server.log',t/'server.log')
        runner.verify_route(t)
        assert json.loads((t/'route_evidence.json').read_text()) == json.loads((RUN/'candidate_benchmark/route_evidence.json').read_text())
        checks.append('all_original_route_evidence_revalidated')
        for name in ['metrics_before.prom','metrics_after.prom']:
            shutil.copy2(RUN/'candidate_benchmark'/name,t/name)
        accounting=analysis.compare(parent,t,t/'accounting.json')
        # vLLM bench may also send endpoint probe requests inside the bracket.
        assert accounting['candidate']['counter_delta']['request_queue_time_seconds_count'] >= 4*256
        checks.append('original_scheduler_ordinary_counters_pass')
        original_log=(t/'server.log').read_text()
        for label,suffix in [
            ('withdrawn_class','\nKVReservationAsyncScheduler\n'),
            ('withdrawn_enabled','\nKV growth admission enabled=True\n'),
            ('withdrawn_telemetry','\nLeanScale KV admission stats {"checked": 1}\n'),
        ]:
            (t/'server.log').write_text(original_log+suffix)
            try: analysis.compare(parent,t,t/'bad.json')
            except ValueError as e: assert 'Withdrawn scheduler' in str(e);checks.append('reject_'+label)
            else: raise AssertionError(label)
        (t/'server.log').write_text(original_log)
        after=(t/'metrics_after.prom').read_text()
        (t/'metrics_after.prom').write_text('\n'.join(l for l in after.splitlines() if not l.startswith('vllm:num_preemptions_total'))+'\n')
        try: analysis.compare(parent,t,t/'bad.json')
        except ValueError as e: assert 'Missing counters' in str(e);checks.append('reject_missing_counter')
        else: raise AssertionError('missing counter')
        (t/'metrics_after.prom').write_text(after)
        (t/'metrics_before.prom').write_text(after)
        (t/'metrics_after.prom').write_bytes((RUN/'candidate_benchmark/metrics_before.prom').read_bytes())
        try: analysis.compare(parent,t,t/'bad.json')
        except ValueError as e: assert 'Counter reset' in str(e);checks.append('reject_counter_reset')
        else: raise AssertionError('counter reset')
    assert all(digest(RUN/rel)==sha for rel,sha in original_hashes.items())
    comparisons=json.loads((RUN/'comparison_exp6.json').read_text())
    assert comparisons[0]['candidate_total_tokens_s'] == float(summary[0]['Total tok/s'])
    assert comparisons[0]['throughput_change_percent'] > -3
    reference=runner.validate_benchmark(parent,cases)
    b=reference[1][0];c=summary[0]
    metrics={key:{'reference':float(b[key]),'candidate':float(c[key]),
                  'change_percent':(float(c[key])/float(b[key])-1)*100}
             for key in ['Total tok/s','Output tok/s','Mean TTFT (ms)','Mean TPOT (ms)','Mean ITL (ms)']}
    recovery={'status':'complete_case_revalidated_offline','time':stamp,
              'measurement_completed_at':state['updated_at'],'raw_stage_preserved':state['stage'],
              'raw_error_preserved':state['error'],'original_failure_archive_commit':'e097a1b6162321423f90c193722990aa16abf74f',
              'case':[4096,1024,64,256],'all_four_rounds_successful':True,'discard_first':True,
              'partial_runs_combined':False,'checks_passed':checks,'source_hashes_before_and_after':original_hashes,
              'helper_sha256':digest(EXP/'analyze_schedule.py'),'future_runner_sha256':digest(EXP/'run_split_e2e.py'),
              'metrics_vs_exp6':metrics,'selected_best_unchanged':True,
              'accuracy_status':'deferred_by_user; math_500_not_run'}
    write_json(RUN/'postprocessing_recovery.json',recovery)
    write_json(RUN/'postprocessing_recovery_schedule.json',accounting)
    archive=EXP/'artifacts/runs'/RUN.name
    for name in ['postprocessing_recovery.json','postprocessing_recovery_schedule.json']:
        (archive/name).write_text(finish.sanitize((RUN/name).read_text()))
    write_json(REV/'validation.json',{'status':'passed','checks':checks,'original_files_unchanged':True,
                                   'new_hashes':{n:digest(EXP/n) for n in ['analyze_schedule.py','run_split_e2e.py','finish_split.py']}})
    lines=['# EXP-0012：GQA stage1 官方 4K 完整结果','',
      '2026-10-21：官方四轮全部成功，按协议弃首轮。后处理要求了已撤销调度的专用日志，导致控制器在测量完成后报错。',
      '原 status.json、日志、CSV、哈希及失败归档 e097a1b 保持原样；本次只做离线复核，未重新测量，也未拼接任何轮次。',
      '补充证据见 [postprocessing_recovery.json](artifacts/runs/'+RUN.name+'/postprocessing_recovery.json)。','',
      '| 指标 | EXP-0006 | GQA stage1 | 变化 |','| --- | ---: | ---: | ---: |']
    for key,d in metrics.items(): lines.append(f"| {key} | {d['reference']:.2f} | {d['candidate']:.2f} | {d['change_percent']:+.2f}% |")
    lines += ['', '保留三轮吞吐 2908.21 / 2844.26 / 3055.57 tok/s，平均 2936.01 tok/s，CV 3.6909%；EXP-0006 CV 3.6373%。',
      '相对原始官方基线 2008.97 tok/s，吞吐提升 46.15%。对照使用历史服务会话，不是交错 A/B。',
      '首轮 1648.24 tok/s 按既定协议丢弃，所有数据均保留。不能用最优单轮 3055.57 代替均值。','',
      '| 调度指标（包含首轮的全部四轮） | EXP-0006 | GQA stage1 |','| --- | ---: | ---: |']
    for key in ['mean_queue_seconds_all_four_rounds','peak_running_sampled','peak_waiting_sampled','peak_cache_usage_percent_sampled']:
        lines.append(f"| {key} | {accounting['reference'][key]} | {accounting['candidate'][key]} |")
    lines.append(f"| preemptions | {accounting['reference']['counter_delta']['num_preemptions_total']} | {accounting['candidate']['counter_delta']['num_preemptions_total']} |")
    lines += ['', '排队计数覆盖全部四轮及测评脚本的端点探测请求，吞吐和延迟表弃首轮，口径不可混用；日志峰值只是采样值。',
      '原调度专用准入日志不存在是正确状态；修正后处理后仍拒绝缺失计数器、计数器重置和劣化调度路径。',
      '42 层 GQA stage1、42 层 RoPE+KV、35 张 CUDA Graph 及实际编译元数据再次离线核验通过。',
      '插件 08d1711e58fdd0f4cd4f746f9e21cb59800ec1d6；FlagGems 19aa059af4ce456ef2e34e3bb83eb949d7e7b702。',
      '16K 待测；最优服务选择暂不变。完整任务精度按用户要求暂缓，不宣称赛事精度达标。']
    (EXP/'RESULTS_4096.md').write_text('\n'.join(lines)+'\n')
    with (EXP/'README.md').open('a') as f:
        f.write('\n4K 四轮完整数据已离线复核：[结果](RESULTS_4096.md)。后处理失败记录保留；16K 使用修正后的原调度计数分析，推理源码和官方参数不变。\n')
    p=EXP/'PR_DRAFT.md'
    p.write_text(p.read_text().replace('真实42层接入、官方4K/16K四轮结果待补。',
        '真实42层接入通过；官方4K四轮弃首轮为2936.01 tok/s，相对EXP-0006提升9.29%，CV3.69%；16K待完成。历史会话对照，不是交错A/B。'))
    (EXP/'STATUS.md').write_text('# EXP-0012 状态\n\n4K 已完成四轮并离线复核；原后处理失败状态保留。[4K结果](RESULTS_4096.md)。16K待接续，最终选择与完整任务精度尚未完成。\n')
    write_json(ROOT/'artifacts/EXP-0012-4k-recovery.json',{'run_directory':str(RUN),**recovery})
    hashes={str(p.relative_to(EXP)):digest(p) for p in sorted((EXP/'artifacts').rglob('*')) if p.is_file()}
    write_json(EXP/'artifact_manifest.json',hashes)
    with (EXP/'steps.jsonl').open('a') as f:
        f.write(json.dumps({'time':datetime.datetime.now().astimezone().isoformat(),
            'action':'postprocessing_recovery_validated','checks':checks,
            'result':'4K 2936.01 tok/s, +9.294% vs EXP6; original failure retained; ready for 16K.'})+'\n')
    paths=['experiments/EXP-0012/'+p for p in ['analyze_schedule.py','run_split_e2e.py','finish_split.py','recover_postprocessing.py',
        'revisions/original_scheduler_postprocessing','steps.jsonl','RESULTS_4096.md','STATUS.md','README.md','PR_DRAFT.md',
        'artifact_manifest.json','artifacts/runs/'+RUN.name+'/postprocessing_recovery.json',
        'artifacts/runs/'+RUN.name+'/postprocessing_recovery_schedule.json']]
    assert not git('diff','--cached','--name-only')
    forbidden=re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|lab_inst_[A-Za-z0-9]+|ssh\.flagos\.io')
    for rel in paths:
        p=REPO/rel
        for item in p.rglob('*') if p.is_dir() else [p]:
            if item.is_file(): assert not forbidden.search(item.read_bytes()), str(item)
    git('add','--',*paths)
    git('diff','--cached','--check')
    git('commit','--only','-m','experiments: recover complete GQA 4K results and fix original-scheduler accounting','--',*paths)
    head=git('rev-parse','HEAD')
    git('push','origin','main')
    assert git('ls-remote',git('remote','get-url','--push','origin'),'refs/heads/main').split()[0] == head
    write_json(ROOT/'artifacts/EXP-0012-4k-recovery-publish.json',{'commit':head,'remote_verified':True})
    print(json.dumps({'commit':head,'remote_verified':True,'checks':checks,'metrics':metrics,
                      'future_runner_sha256':recovery['future_runner_sha256']},ensure_ascii=False),flush=True)
    docs_lock.close();lock.close()

if __name__=='__main__':
    main()
