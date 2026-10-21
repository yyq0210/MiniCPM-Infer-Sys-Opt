"""Freeze the API repair after real allocator CPU checks; retain failed parent."""
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path('/workspace/LeanScale')
REPO = ROOT / 'MiniCPM-Infer-Sys-Opt'
EXP = REPO / 'experiments/EXP-0010'
PLUGIN = ROOT / 'worktrees/EXP-0010/vllm-plugin-FL'
GEMS = ROOT / 'worktrees/EXP-0010/FlagGems'
CPU = ROOT / 'artifacts/EXP-0010/20261021_cpu'
CODE = ['vllm_fl/worker/kv_reservation_scheduler.py', 'tests/standalone/validate_kv_reservation_cpu.py']
FORBIDDEN = re.compile(rb'PRIVATE KEY|lab_inst_|ssh\.flagos\.io|github_pat_|gh[pousr]_[A-Za-z0-9]{20,}')

def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True, timeout=120).strip()

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, data):
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')

assert git(PLUGIN, 'rev-parse', 'HEAD') == 'ef63a19a8309c3b3ba686802220883ac2f8836ab'
assert git(GEMS, 'rev-parse', 'HEAD') == '484413ea0db18221dd9e63e742361d837df9ff13'
assert not git(GEMS, 'status', '--porcelain')
result = json.loads((CPU / 'results.json').read_text())
assert result['status'] == 'passed' and not result['cuda_initialized']
assert len(result['checks']) == 14 and all(x['passed'] for x in result['checks'])
assert not git(PLUGIN, 'diff', '--cached', '--name-only')
assert set(git(PLUGIN, 'diff', '--name-only').splitlines()) == set(CODE)
for name in CODE:
    data = (PLUGIN / name).read_bytes()
    assert not FORBIDDEN.search(data)
    ast.parse(data)
git(PLUGIN, 'add', '--', *CODE)
git(PLUGIN, 'diff', '--cached', '--check')
git(PLUGIN, 'commit', '-m', 'fix(iluvatar): forward scheduler prefill throttle and verify entry contract')
branch = 'leanscale/iluvatar-kv-reservation-v2'
git(PLUGIN, 'push', '-u', 'leanscale', branch)
candidate = {'vllm-plugin-FL': git(PLUGIN, 'rev-parse', 'HEAD'), 'FlagGems': git(GEMS, 'rev-parse', 'HEAD')}
url = git(PLUGIN, 'remote', 'get-url', '--push', 'leanscale')
assert git(PLUGIN, 'ls-remote', url, 'refs/heads/' + branch).split()[0] == candidate['vllm-plugin-FL']
parent = {'vllm-plugin-FL': '07d9fa0cfdc5f09db178e29508e3c0bcb5c1e908',
          'FlagGems': '484413ea0db18221dd9e63e742361d837df9ff13'}
write(EXP / 'candidate_commits.json', candidate)
write(EXP / 'parent_commits.json', parent)
write(EXP / 'candidate_validation.json', {
    'status': 'passed', 'scope': '14 CPU checks including actual parent API contract and allocator lifecycle; service smoke pending',
    'candidate_commits': candidate, 'parent_commits': parent,
    'failed_scheduler_parent': 'ef63a19a8309c3b3ba686802220883ac2f8836ab',
    'diagnostic_hashes': {str(CPU / 'results.json'): sha(CPU / 'results.json')},
    'code_hashes': {'vllm-plugin-FL': {name: sha(PLUGIN / name) for name in CODE}, 'FlagGems': {}},
    'cuda_initialized': False, 'full_model_accuracy': 'deferred_by_user',
})
dest = EXP / 'artifacts/20261021_cpu'
dest.mkdir(parents=True, exist_ok=True)
shutil.copy2(CPU / 'results.json', dest / 'results.json')
log = (CPU / 'validation.log').read_text()
log = re.sub(r'uuid=[^,)]+', 'uuid=[REDACTED]', log)
log = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP_REDACTED]', log)
(dest / 'validation.log').write_text(log)
(EXP / 'STATUS.md').write_text('# EXP-0010 状态\n\n14 项 CPU 检查通过，包括 EngineCore 位置参数/关键字参数契约、启用/禁用路径、异常后 manager 恢复。代码已冻结并推送；GPU 真实服务 smoke 和官方四轮评测尚待执行。\n')
print(json.dumps({'candidate_commits': candidate, 'fork_push_verified': True}))
