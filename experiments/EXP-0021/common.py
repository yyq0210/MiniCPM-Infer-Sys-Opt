from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import re
import subprocess

ROOT = Path('/workspace/LeanScale')
REPO = ROOT/'MiniCPM-Infer-Sys-Opt'
EXP = REPO/'experiments/EXP-0021'
WT = ROOT/'worktrees/EXP-0021'
ART = ROOT/'artifacts/EXP-0021'
COMMITS = {
    'A': {'vllm-plugin-FL':'588a6147cd291e272324ab6c812024cd587c6287', 'FlagGems':'fd209df717648002ebf5ba4ec351424dcc997a53'},
    'B': {'vllm-plugin-FL':'ce19f93eee4c73f5a26911554700457e879f891d', 'FlagGems':'de958ddfc4deeb43d6c94bb8652a3d4d3bc905fe'},
}
ORDER = ['A','B','B','A']
def now(): return datetime.datetime.now().astimezone()
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def git(p,*a): return subprocess.check_output(['git','-C',str(p),*a],text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def step(action,**kw):
    with (EXP/'steps.jsonl').open('a') as f: f.write(json.dumps(dict(time=now().isoformat(),action=action,**kw),ensure_ascii=False)+'\n')
def verify_sources(arm):
    for name, commit in COMMITS[arm].items():
        assert git(WT/arm/name,'rev-parse','HEAD')==commit
        assert not git(WT/arm/name,'status','--porcelain')
    for name,digest in read(EXP/'script_hashes.json').items(): assert sha(EXP/name)==digest,name
def publish(paths, message):
    # Exact paths only; raw files remain in artifacts even if publication fails.
    with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert git(REPO,'branch','--show-current')=='main'
        assert not git(REPO,'diff','--cached','--name-only')
        bad=re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|lab_inst_[A-Za-z0-9]+|ssh\.flagos\.io')
        for rel in paths:
            p=REPO/rel
            for f in p.rglob('*') if p.is_dir() else [p]:
                if f.is_file() and f.suffix!='.pyc': assert not bad.search(f.read_bytes()),str(f)
        git(REPO,'add','--',*paths);git(REPO,'diff','--cached','--check')
        if git(REPO,'diff','--cached','--name-only'): git(REPO,'commit','--only','-m',message,'--',*paths)
        head=git(REPO,'rev-parse','HEAD');git(REPO,'push','origin','main')
        assert git(REPO,'ls-remote',git(REPO,'remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
        return dict(status='pushed',commit=head,remote_verified=True)
