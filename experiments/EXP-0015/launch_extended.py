"""Audit completed refinement, publish the validation plan, then launch once."""
from pathlib import Path
import ast
import datetime
import fcntl
import hashlib
import json
import re
import shutil
import subprocess
import time

ROOT = Path('/workspace/LeanScale')
REPO = ROOT / 'MiniCPM-Infer-Sys-Opt'
EXP = REPO / 'experiments/EXP-0015'
STAGING = Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True, stderr=subprocess.STDOUT, timeout=120).strip()


def main():
    prior = read(ROOT / 'artifacts/EXP-0015-refined-request.json')
    run = Path(prior['run_directory'])
    assert prior['stage'] == 'screen_complete' and prior['safe_to_resume_gpu']
    publication = read(run / 'publication.json')
    assert publication['status'] == 'pushed' and publication['remote_verified']
    git('merge-base', '--is-ancestor', publication['commit'], 'HEAD')
    summary = read(run / 'screen_summary.json')
    winner, = [x for x in summary['rows'] if x['global_candidate_gate_passed']]
    assert winner['configuration'] == 'rows64_w4_s1_tile32'
    assert len(prior['trials']) == 3
    source_hashes = {}
    for item in prior['trials']:
        p = run / (item['configuration'] + '.json')
        d = read(p)
        assert item['status'] == 'passed' and d['status'] == 'passed'
        assert len(d['checks']) == 20 and len(d['timings']) == 9
        assert all(abs(t['baseline_drift_percent']) <= 3 for t in d['timings'])
        source_hashes[p.name] = sha(p)
    winning_data = read(run / (winner['configuration'] + '.json'))
    coarse = Path(read(ROOT / 'artifacts/EXP-0015-request.json')['run_directory'])
    coarse_winner = read(coarse / 'rows64_w4_s1.json')
    assert coarse_winner['status'] == 'passed' and winning_data['config'] == coarse_winner['config']
    latest = ROOT / 'artifacts/EXP-0015-expanded-request.json'
    assert not latest.exists(), 'Inspect existing expanded run before retrying'
    deadline = datetime.datetime.fromisoformat(read(ROOT / 'artifacts/run_window.json')['compute_stop_at'])
    assert (deadline - datetime.datetime.now().astimezone()).total_seconds() > 1800
    files = ['validate_extended.py', 'run_extended.py', 'launch_extended.py']
    with (ROOT / 'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not git('status', '--porcelain')
        for name in files:
            ast.parse((STAGING / name).read_text())
            shutil.copy2(STAGING / name, EXP / name)
        choice = dict(config=winning_data['config'], refinement_run=run.name, refinement_publication=publication,
                      refinement_source_hashes=source_hashes, refinement_summary_sha256=sha(run / 'screen_summary.json'),
                      coarse_winner_sha256=sha(coarse / 'rows64_w4_s1.json'), selected_for_serving=False,
                      validation_script_hashes={n: sha(EXP / n) for n in files + ['trial.py', 'run_screen.py']},
                      hypothesis='Reproduced stage1/tile32 prefill gain remains correct over every query, dynamic graph metadata, large addresses and wider shapes without decode regression.',
                      validation_checks=42, timing_points=36, serving_selection_unchanged='EXP-0014',
                      integration_gate='All numerical/Graph checks pass; review every timing regression and baseline drift >3% before engine integration.')
        (EXP / 'expanded_choice.json').write_text(json.dumps(choice, indent=2, ensure_ascii=False) + '\n')
        lines = ['', '## 邻域复核已完成，进入扩大验证', '',
                 '14:31三组全部完成，新增60项数值/Graph检查、27个计时点；所有原始数据已自动推送。',
                 'rows64/warps4/stage1/tile32再次降低pure/mixed耗时9.25%–11.52%；decode变化在0.14%内。',
                 'tile64使pure/mixed耗时增加39.38%–52.85%，stage3增加36.56%–47.32%，两者淘汰。',
                 '因此固定stage1/tile32进入扩大验证，保持decode分页布局和8行配置不变。', '',
                 '扩大检查包含BF16/FP16×page8/16/32/64与NHD/HND、query边界/尾部/混合批次、2048-query长文的每个query FP32因果对照、Graph重放时Q/K/V/length/block-table变化，以及固定地址下CU/query分布变化。另以BF16和FP16实际访问超过2^31元素偏移的prefill KV页。共42项。',
                 '36个计时点覆盖query2/8/17/64/128/512/2048、4K/16K context、多prefill、mixed两顺序及decode批次1/8/16/30/32/64；7×8次Graph调用、前后基线夹测，报告完整GQA耗时。',
                 '先完整保存并复核每个>3%回归或漂移，再决定接入引擎；不自动改变已选EXP-0014。完整任务精度仍暂缓。', '',
                 '[邻域完整数据](../experiments/EXP-0015/artifacts/' + run.name + '/RESULTS.md)。']
        with (REPO / 'docs/gqa_prefill_screen_20261022.md').open('a') as f:
            f.write('\n'.join(lines) + '\n')
        with (EXP / 'steps.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),
                                   action='refinement_complete_prepare_expanded_validation', source_run=run.name,
                                   winner=winner['configuration'], checks=42, timing_points=36,
                                   serving_selection_unchanged='EXP-0014')) + '\n')
        paths = ['experiments/EXP-0015/' + n for n in files + ['expanded_choice.json', 'steps.jsonl']]
        paths += ['docs/gqa_prefill_screen_20261022.md']
        forbidden = re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|lab_inst_[A-Za-z0-9]+|ssh\.flagos\.io')
        for rel in paths:
            assert not forbidden.search((REPO / rel).read_bytes()), rel
        git('add', '--', *paths)
        git('diff', '--cached', '--check')
        git('commit', '--only', '-m', 'experiments: freeze prefill winner and expand numerical and performance validation', '--', *paths)
        head = git('rev-parse', 'HEAD')
        git('push', 'origin', 'main')
        assert git('ls-remote', git('remote', 'get-url', '--push', 'origin'), 'refs/heads/main').split()[0] == head
    with (ROOT / 'artifacts/EXP-0015-extended-controller.log').open('a') as log:
        child = subprocess.Popen(['python3', str(EXP / 'run_extended.py')], stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(100):
        if latest.exists():
            break
        if child.poll() is not None:
            raise RuntimeError('Expanded controller exited before receipt')
        time.sleep(.2)
    else:
        raise RuntimeError('No expanded receipt')
    receipt = read(latest)
    assert receipt['pid'] == child.pid
    (ROOT / 'artifacts/EXP-0015-extended-launch.json').write_text(json.dumps(dict(documentation_commit=head, **receipt), indent=2) + '\n')
    print(json.dumps(dict(documentation_commit=head, receipt=receipt)), flush=True)


if __name__ == '__main__':
    main()
