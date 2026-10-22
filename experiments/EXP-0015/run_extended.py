"""Deadline-bounded expanded prefill validation; preserve the serving selection."""
from run_screen import ROOT, REPO, EXP, WT, write, stop
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import re
import subprocess
import traceback


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True, timeout=120).strip()


def publish(run):
    dest = EXP / 'artifacts' / run.name
    with (ROOT / 'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        dest.mkdir(parents=True, exist_ok=True)
        for p in run.iterdir():
            if p.is_file() and p.suffix in ('.json', '.log', '.md', '.csv'):
                text = p.read_text(errors='replace').replace('\r\n', '\n')
                text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP_REDACTED]', text)
                text = re.sub(r'uuid=[^,)]+', 'uuid=[REDACTED]', text)
                (dest / p.name).write_text(text)
        forbidden = re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|lab_inst_[A-Za-z0-9]+|ssh\.flagos\.io')
        for p in dest.rglob('*'):
            if p.is_file():
                assert not forbidden.search(p.read_bytes()), str(p)
        paths = [str(dest.relative_to(REPO))]
        assert not git('diff', '--cached', '--name-only')
        git('add', '--', *paths)
        git('diff', '--cached', '--check')
        git('commit', '--only', '-m', 'experiments: archive expanded GQA prefill validation', '--', *paths)
        head = git('rev-parse', 'HEAD')
        git('push', 'origin', 'main')
        assert git('ls-remote', git('remote', 'get-url', '--push', 'origin'), 'refs/heads/main').split()[0] == head
        write(run / 'publication.json', dict(status='pushed', commit=head, remote_verified=True))


def main():
    now = datetime.datetime.now().astimezone()
    run = ROOT / 'artifacts/EXP-0015' / now.strftime('%Y%m%d_%H%M%S_extended')
    run.mkdir(parents=True, exist_ok=False)
    choice = read(EXP / 'expanded_choice.json')
    deadline = datetime.datetime.fromisoformat(read(ROOT / 'artifacts/run_window.json')['compute_stop_at'])
    state = dict(pid=os.getpid(), process_start_ticks=Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()[19],
                 run_directory=str(run), started_at=now.isoformat(), compute_stop_at=deadline.isoformat(),
                 stage='preparing', config=choice['config'], commits=read(EXP / 'candidate_commits.json'),
                 selected_for_serving=False, kind='expanded_operator_validation_not_official_e2e',
                 choice_sha256=sha(EXP / 'expanded_choice.json'))

    def update(stage, **kw):
        state.update(stage=stage, updated_at=datetime.datetime.now().astimezone().isoformat(), **kw)
        write(run / 'status.json', state)
        write(ROOT / 'artifacts/EXP-0015-expanded-request.json', state)

    lock = Path('/workspace/experiments/codex_20261019/runner.lock').open('a')
    child = None
    locked = False
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        remaining = (deadline - datetime.datetime.now().astimezone()).total_seconds()
        assert remaining > 1800, 'Need a complete 30-minute validation window'
        prior = read(ROOT / 'artifacts/EXP-0015-refined-request.json')
        assert prior['stage'] == 'screen_complete' and prior['safe_to_resume_gpu']
        assert Path(prior['run_directory']).name == choice['refinement_run']
        for n, expected in choice['validation_script_hashes'].items():
            assert sha(EXP / n) == expected, n
        for n, commit in state['commits'].items():
            assert subprocess.check_output(['git', '-C', str(WT / n), 'rev-parse', 'HEAD'], text=True).strip() == commit
            assert not subprocess.check_output(['git', '-C', str(WT / n), 'status', '--porcelain'], text=True)
        frozen = ROOT / 'worktrees/EXP-0014/FlagGems'
        assert subprocess.check_output(['git', '-C', str(frozen), 'rev-parse', 'HEAD'], text=True).strip() == '1c526ca99d412e203ed3caa8b1663566976ea38c'
        assert not subprocess.check_output(['git', '-C', str(frozen), 'status', '--porcelain'], text=True)
        state['validation_script_hashes'] = choice['validation_script_hashes']
        write(run / 'expanded_choice.json', choice)
        env = dict(os.environ, PYTHONPATH=f'{WT}/vllm-plugin-FL:{WT}/FlagGems/src', PYTHONUNBUFFERED='1',
                   VLLM_PLUGINS='fl', TRITON_CACHE_DIR=str(run / 'triton_cache'))
        with (run / 'validation.log').open('w') as log:
            child = subprocess.Popen(['python3', str(EXP / 'validate_extended.py'), str(run / 'validation.json')],
                                     env=env, cwd=run, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            update('validating', child_pid=child.pid)
            code = child.wait(timeout=min(1800, (deadline-datetime.datetime.now().astimezone()).total_seconds()))
        assert code == 0, f'Validation exit {code}'
        data = read(run / 'validation.json')
        assert data['status'] == 'passed' and data['config'] == choice['config']
        assert len(data['checks']) == 42 and len(data['timings']) == 36
        assert all(c['graph_passed'] for c in data['checks'])
        regressions = [x['name'] for x in data['timings'] if x['time_change_percent'] > 3]
        drift = [x['name'] for x in data['timings'] if abs(x['baseline_drift_percent']) > 3]
        update('validation_complete', checks=42, timing_points=36,
               regression_names=regressions, drift_names=drift, requires_review_before_integration=True)
        lines = ['# EXP-0015 扩大验证', '',
                 '单算子验证，未改变服务选择；42项全query数值/Graph检查通过，36个完整GQA计时点。',
                 '前后基线漂移与所有回归均保留，集成前人工复核；不能当作整模收益。', '',
                 '| 工作点 | 时间变化 | 前后基线漂移 |', '|---|---:|---:|']
        lines += [f"| {x['name']} | {x['time_change_percent']:+.3f}% | {x['baseline_drift_percent']:+.3f}% |" for x in data['timings']]
        lines += ['', f'超过3%回归：{regressions}。超过3%基线漂移：{drift}。']
        (run / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    except BaseException as exc:
        update('failed', error=repr(exc))
        traceback.print_exc()
    finally:
        stop(child)
        update(state['stage'], owned_children_reaped=True, safe_to_resume_gpu=locked)
        lock.close()
        try:
            publish(run)
        except BaseException as exc:
            write(run / 'publication.json', dict(status='local_saved_push_failed', error=repr(exc)))
            traceback.print_exc()


if __name__ == '__main__':
    main()
