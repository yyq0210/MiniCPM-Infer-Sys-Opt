"""Audit the completed screen, freeze all-query checks, issue a fresh budget."""
from datetime import datetime,timedelta
from pathlib import Path
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import time

ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt';EXP=REPO/'experiments/EXP-0019'
DEST=EXP/'expanded';STAGING=Path(__file__).resolve().parent;ART=ROOT/'artifacts/EXP-0019'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True,timeout=120).strip()
def step(action,**kw):
    with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(),action=action,**kw),ensure_ascii=False)+'\n')

def main():
    assert not DEST.exists() and not (ROOT/'artifacts/EXP-0019-expanded-request.json').exists()
    prior=read(ROOT/'artifacts/EXP-0019-request.json');run=Path(prior['run_directory'])
    assert prior['stage']=='screen_complete' and prior['owned_children_reaped']
    pub=read(run/'publication.json');assert pub['remote_verified']
    git('merge-base','--is-ancestor',pub['commit'],'HEAD')
    assert [t['configuration'] for t in prior['trials']]==['layout0','layout1','layout2','layout3']
    assert all(t['returncode']==0 and t['status']=='passed' for t in prior['trials'])
    assert {p.name:sha(p) for p in EXP.glob('*.py')}==read(EXP/'script_manifest.json')
    summary=read(run/'screen_summary.json');assert len(summary['order'])==14
    summaries=[];raw={};dataset={}
    for layout in range(4):
        name=f'layout{layout}';p=run/(name+'.json');a=read(p);dataset[name]=a;raw[p.name]=sha(p)
        assert a['status']=='passed' and len(a['checks'])==20 and len(a['timings'])==14
        assert a['config']['prefill_layout']==layout
        for key,value in dict(prefill_rows=64,prefill_tile=32,prefill_warps=4,prefill_stages=1,decode_rows=8,decode_layout=1,decode_stages=1,segments=16).items():assert a['config'][key]==value
        assert all(c['graph_replay_passed'] for c in a['checks'])
        assert [t['name'] for t in a['timings']]==summary['order']
        changes=[];drifts=[]
        for t in a['timings']:
            for tag in ['baseline_before','candidate_total','baseline_after']:
                v=t[tag];assert len(v['samples_us'])==7 and v['median_us']==statistics.median(v['samples_us'])
            before=t['baseline_before']['median_us'];after=t['baseline_after']['median_us'];candidate=t['candidate_total']['median_us']
            change=(candidate/((before+after)/2)-1)*100;drift=(after/before-1)*100
            assert math.isclose(change,t['time_change_percent'],abs_tol=1e-10)
            assert math.isclose(drift,t['baseline_drift_percent'],abs_tol=1e-10)
            changes.append(change);drifts.append(drift)
            if 'resources' in t:
                for kind,digest in t['resources']['ir_sha256'].items():assert sha(run/'ir'/name/(t['name']+'.'+kind))==digest
                ir=(run/'ir'/name/(t['name']+'.ttir')).read_text()
                assert re.search(r'tt\.load[^\n]*tensor<64x128x!tt\.ptr<(?:bf16|f16)>>',ir)
                width=2 if layout in (1,3) else 32
                assert re.search(r'tt\.load[^\n]*tensor<'+str(width)+r'x!tt\.ptr<i32>>',ir)
                if layout in (1,3):
                    assert re.search(r'tt\.broadcast[^\n]*tensor<2x1xi64> -> tensor<2x16xi64>',ir)
                    assert re.search(r'tt\.reshape[^\n]*tensor<2x16xi64> -> tensor<32xi64>',ir)
        assert max(abs(x) for x in drifts)<=3
        old=next(x for x in summary['rows'] if x['configuration']==name)
        assert changes==old['time_change_percent']
        summaries.append(dict(configuration=name,primary_min_percent=min(changes[:11]),primary_max_percent=max(changes[:11]),
            decode_max_abs_percent=max(abs(x) for x in changes[11:]),max_baseline_drift_percent=max(abs(x) for x in drifts),
            representative_resources=a['timings'][0]['resources']))
    # Layout1 is faster at every primary point, not merely the best average.
    winner=dataset['layout1']
    for i in range(11):
        assert winner['timings'][i]['time_change_percent'] < min(dataset[k]['timings'][i]['time_change_percent'] for k in ['layout0','layout2','layout3'])
    assert next(x for x in summary['rows'] if x['configuration']=='layout1')['global_candidate_gate_passed']
    DEST.mkdir()
    for n in ['validate_extended.py','run_extended.py','prepare_extended.py']:
        shutil.copy2(STAGING/n,DEST/n)
    subprocess.run(['python3','-m','py_compile',*[str(p) for p in DEST.glob('*.py')]],check=True)
    files=['trial.py','run_screen.py','expanded/validate_extended.py','expanded/run_extended.py','expanded/prepare_extended.py']
    choice=dict(config=winner['config'],screen_run=run.name,screen_publication=pub,screen_source_hashes=raw,
                selected_for_serving=False,validation_checks=42,timing_points=42,
                validation_script_hashes={n:sha(EXP/n) for n in files},
                hypothesis='Prefill page-id broadcast remains numerically correct for all queries and mutable Graph metadata, including actual int64 offsets, with wider shape benefits',
                gate='Review every regression or drift over3percent before integrating; no full-model performance claim')
    write(DEST/'choice.json',choice)
    write(DEST/'screen_audit.json',dict(status='passed',screen_run=run.name,checks=80,timings=56,raw_hashes=raw,rows=summaries,
          winner='layout1',dominates_each_primary_point=True,source_manifest_unchanged=True))
    lines=['# EXP19 prefill页号广播初筛与扩大验证','','4个配置均完成20项数值/Graph检查、14个完整GQA计时点，共80项检查/56个计时点。原始采样与IR哈希已独立重算。',
           'layout1仅页号广播在11个prefill/mixed点全部优于其他已测配置，耗时下降15.53%–20.71%；decode回归约±0.11%。这是单算子诊断，尚无新增整模收益。','',
           '| 配置 | prefill/mixed耗时变化 | decode最大绝对变化 | 最大夹测漂移 |', '|---|---|---:|---:|']
    for x in summaries:lines.append(f"| {x['configuration']} | {x['primary_min_percent']:+.3f}% .. {x['primary_max_percent']:+.3f}% | {x['decode_max_abs_percent']:.3f}% | {x['max_baseline_drift_percent']:.3f}% |")
    lines+=['','代表性prefill编译资源：原路径99 registers/32KiB shared，页号广播98 registers/32KiB shared；spill均0。共享内存不变，不能据此宣称occupancy提高。K转置变体117 registers/24KiB shared，虽shared更少但实测不如仅页号广播，资源数字不能代替计时或动态occupancy。',
            '实际TTIR核验layout1/3每tile读取2个页号并以int64广播到32个token lane；layout0/2仍为32个token lane的页号读取。',
            '现在固定layout1进入扩大验证：42项all-query FP32/Graph检查，含两dtype、page8/16/32/64、NHD/HND、CU变化和真实超过2^31元素的KV地址；42个完整GQA计时点，包括6个EXP18形状导出点。初筛的CPU地址模型不代替这些GPU检查。',
            '扩大验证前后均固定EXP16为直接对照，服务选择仍是EXP16；通过后才接入引擎和正式4K/16K测试。',
            '[初筛全部结果](../experiments/EXP-0019/artifacts/'+run.name+'/RESULTS.md) · [独立初筛审核](../experiments/EXP-0019/expanded/screen_audit.json)。']
    (REPO/'docs/gqa_prefill_memory_20261023.md').write_text('\n'.join(lines)+'\n')
    step('Screen independently audited; choose layout1 for expanded validation only',checks=80,timings=56,config=choice['config'])
    write(EXP/'STATUS.json',dict(stage='screen_audited_expanded_prepared',selected_for_serving=False,choice=choice['config']))
    with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        paths=['experiments/EXP-0019/expanded','experiments/EXP-0019/steps.jsonl','experiments/EXP-0019/STATUS.json',
               'experiments/EXP-0019/screen_window.json','docs/gqa_prefill_memory_20261023.md']
        git('add','--',*paths);git('diff','--cached','--check')
        assert '__pycache__' not in git('diff','--cached','--name-only')
        git('commit','--only','-m','experiments: audit prefill memory winner and prepare all-query validation','--',*paths)
        commit=git('rev-parse','HEAD');git('push','origin','main');assert git('ls-remote','origin','refs/heads/main').split()[0]==commit
    write(ART/'expanded_preparation.json',dict(status='prepared',documentation_commit=commit,remote_verified=True))
    # Separate finite budget for the newly audited validation stage. No night
    # availability is assumed and the prior screen budget is not extended.
    now=datetime.now().astimezone();assert now.strftime('%Y-%m-%d')=='2026-10-23'
    end=min(now+timedelta(minutes=35),now.replace(hour=16,minute=25,second=0,microsecond=0))
    assert (end-now).total_seconds()>31*60,'Insufficient fresh validation budget; prepared files retained'
    policy=dict(experiment='EXP-0019-expanded',created_at=now.isoformat(),compute_stop_at=end.isoformat(),
                archive_target_at=(end+timedelta(minutes=10)).isoformat(),machine_shutdown_at=None,user_confirmed_shutdown=False,
                scope='Fresh finite budget for layout1 all-query/Graph/int64 validation only; no automatic subsequent GPU work')
    write(ART/'expanded_window.json',policy)
    step('Launch independently budgeted expanded validation after audit and published plan',policy=policy)
    with (ART/'extended-controller.log').open('w') as log:
        child=subprocess.Popen(['python3',str(DEST/'run_extended.py')],cwd=DEST,stdin=subprocess.DEVNULL,stdout=log,
                               stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONUNBUFFERED='1'))
    latest=ROOT/'artifacts/EXP-0019-expanded-request.json'
    for _ in range(100):
        if latest.exists():break
        assert child.poll() is None,'Controller exited before receipt';time.sleep(.2)
    else:raise RuntimeError('No expanded receipt')
    receipt=read(latest);assert receipt['pid']==child.pid
    write(ART/'expanded_launch.json',dict(documentation_commit=commit,**receipt))
    print(json.dumps(dict(documentation_commit=commit,receipt=receipt)))

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        write(ART/'expanded_preparation_error.json',dict(error=traceback.format_exc()));step('Expanded preparation failed; preserve existing screen results',error=traceback.format_exc());raise
