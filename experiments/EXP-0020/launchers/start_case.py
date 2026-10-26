"""Start one case under a fresh finite budget within the confirmed daily window."""
from pathlib import Path
from datetime import datetime,timedelta
import argparse
import fcntl
import hashlib
import json
import subprocess

ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt';EXP=REPO/'experiments/EXP-0020';ART=ROOT/'artifacts/EXP-0020'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args],text=True,stderr=subprocess.STDOUT,timeout=120).strip()

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['4k','16k'],required=True);a=p.parse_args()
    now=datetime.now().astimezone();assert now.strftime('%Y-%m-%d')=='2026-10-26'
    assert not (ROOT/f'artifacts/EXP-0020-latest-{a.case}.json').exists(),'Inspect existing attempt; no automatic rerun'
    setup=read(ROOT/'artifacts/EXP-0020-setup.json');assert setup['status']=='prepared' and setup['remote_verified']
    git('merge-base','--is-ancestor',setup['docs_commit'],'HEAD')
    for n,h in read(EXP/'candidate_validation.json')['script_hashes'].items():assert hashlib.sha256((EXP/n).read_bytes()).hexdigest()==h
    if a.case=='16k':
        review=read(EXP/'review_4k.json');assert review['status']=='complete_case_audited'
        assert review['candidate_commits']==setup['candidate_commits'] and review['allow_16k']
    ART.mkdir(exist_ok=True)
    daily=dict(date='2026-10-26',machine_shutdown_at=now.replace(hour=0,minute=0,second=0,microsecond=0)+timedelta(days=1),
       compute_stop_at=now.replace(hour=23,minute=35,second=0,microsecond=0),archive_target_at=now.replace(hour=23,minute=50,second=0,microsecond=0),
       user_confirmed_shutdown=True,user_reply='今晚 00:00 关闭')
    daily={k:v.isoformat() if isinstance(v,datetime) else v for k,v in daily.items()}
    end=min(now+timedelta(minutes=70 if a.case=='4k' else 105),datetime.fromisoformat(daily['compute_stop_at']))
    assert (end-now).total_seconds()>=(55 if a.case=='4k' else 90)*60
    policy=dict(experiment='EXP-0020',case=a.case,created_at=now.isoformat(),compute_stop_at=end.isoformat(),
       archive_target_at=min(end+timedelta(minutes=10),datetime.fromisoformat(daily['archive_target_at'])).isoformat(),
       machine_shutdown_at=daily['machine_shutdown_at'],user_confirmed_shutdown=True,daily_window=daily,
       scope='One complete official case only; new stage requires review and a new finite budget')
    write(ART/'run_window.json',policy)
    launchdir=EXP/'launches';launchdir.mkdir(exist_ok=True)
    stamp=now.strftime('%Y%m%d_%H%M%S_')+a.case
    write(launchdir/(stamp+'_policy.json'),policy)
    write(launchdir/'daily_20261026.json',daily)
    command=['python3',str(EXP/'launch_split.py'),'--case',a.case,'--finish-before',end.isoformat()]
    out=subprocess.check_output(command,text=True,timeout=45);receipt=json.loads(out.strip().splitlines()[-1])
    write(launchdir/(stamp+'_receipt.json'),receipt)
    run=Path(receipt['run_directory'])
    write(run/'integration.json',dict(launch=receipt,policy=policy,daily_window=daily,candidate_commits=setup['candidate_commits'],preparation_commit=setup['docs_commit']))
    with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(),action='Launch single official case with fresh budget and automatic archive',case=a.case,receipt=receipt,policy=policy),ensure_ascii=False)+'\n')
    with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        # Later confirmation supersedes the conservative date-unknown note.
        with (EXP/'README.md').open('a') as f:f.write('\n\n2026-10-26：用户确认今晚00:00关机。今日23:35计算截止、23:50保存；每组仍单独限时并审核，不再沿用未知关机时的16:25上限。实际case窗口和回执见launches/。\n')
        paths=['experiments/EXP-0020/launchers','experiments/EXP-0020/launches','experiments/EXP-0020/steps.jsonl','experiments/EXP-0020/README.md']
        assert not git('diff','--cached','--name-only')
        git('add','--',*paths);git('diff','--cached','--check')
        git('commit','--only','-m','experiments: launch official '+a.case+' with confirmed daily window','--',*paths)
        head=git('rev-parse','HEAD');git('push','origin','main')
        assert git('ls-remote',git('remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
    write(ART/('launch_publication_'+a.case+'.json'),dict(commit=head,remote_verified=True,receipt=receipt))
    print(json.dumps(dict(commit=head,receipt=receipt,policy=policy)),flush=True)

if __name__=='__main__':main()
