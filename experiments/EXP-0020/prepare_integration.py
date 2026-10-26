"""Audit EXP19, record a hypothesis, and freeze its engine integration."""
from pathlib import Path
from datetime import datetime
import ast
import fcntl
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
import traceback

ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt';EXP=REPO/'experiments/EXP-0020';WT=ROOT/'worktrees/EXP-0020'
STAGING=Path(__file__).resolve().parent;EX19=REPO/'experiments/EXP-0019'
PARENT={'vllm-plugin-FL':'588a6147cd291e272324ab6c812024cd587c6287','FlagGems':'fd209df717648002ebf5ba4ec351424dcc997a53'}
GEMS='de958ddfc4deeb43d6c94bb8652a3d4d3bc905fe';BRANCH='leanscale/iluvatar-gqa-prefill-memory'
FILES=['run_split_e2e.py','launch_split.py','finish_split.py','analyze_schedule.py','build_scripts.py','route_checks.py','prepare_integration.py']
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(p,*args):return subprocess.check_output(['git','-C',str(p),*args],text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def step(action,**kw):
    with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(),action=action,**kw),ensure_ascii=False)+'\n')

def audit():
    s=read(ROOT/'artifacts/EXP-0019-expanded-request.json');run=Path(s['run_directory']);d=read(run/'validation.json')
    assert s['stage']=='validation_complete' and s['owned_children_reaped'] and s['safe_to_resume_gpu']
    assert not s['regression_names'] and not s['drift_names']
    assert d['status']=='passed' and d['correctness_passed'] and len(d['checks'])==42 and len(d['timings'])==42
    choice=read(EX19/'expanded/choice.json');assert sha(EX19/'expanded/choice.json')==s['choice_sha256']
    assert d['config']==choice['config']==s['config']
    for n,h in choice['validation_script_hashes'].items():assert sha(EX19/n)==h
    assert all(c['graph_passed'] for c in d['checks'])
    assert len({c['name'] for c in d['checks']})==42
    assert sum(c['changed_cu'] for c in d['checks'])==8
    big=[c for c in d['checks'] if c['name'].startswith('prefill_large_offset')]
    assert len(big)==2 and all(c['page2_offset_elements']==2*(2**30+4096)>2**31 for c in big)
    queries=sum(c['all_query_fp32']['queries_checked']+c['mutated_all_query_fp32']['queries_checked'] for c in d['checks']);assert queries==39444
    assert len({t['name'] for t in d['timings']})==42
    for t in d['timings']:
        for key in ['baseline_before','candidate','baseline_after']:
            v=t[key];assert len(v['samples_us'])==7 and all(math.isfinite(x) and x>0 for x in v['samples_us'])
            assert v['median_us']==statistics.median(v['samples_us'])
        a=t['baseline_before']['median_us'];b=t['baseline_after']['median_us'];c=t['candidate']['median_us']
        assert math.isclose((c/((a+b)/2)-1)*100,t['time_change_percent'],abs_tol=1e-10)
        assert math.isclose((b/a-1)*100,t['baseline_drift_percent'],abs_tol=1e-10)
        assert t['time_change_percent']<=3 and abs(t['baseline_drift_percent'])<=3
    groups={
      'pure_prefill':[t for t in d['timings'] if t['name'].startswith('prefill') or t['name'] in ['multiple_prefills','ragged_prefills']],
      'mixed':[t for t in d['timings'] if 'mixed' in t['name'] and not t['name'].startswith('exp18')],
      'decode':[t for t in d['timings'] if t['name'].startswith('decode')],
      'exp18_shape_derived':[t for t in d['timings'] if t['name'].startswith('exp18')]}
    assert [len(x) for x in groups.values()]==[17,7,12,6]
    pub=read(run/'publication.json');assert pub['remote_verified'] and pub['status']=='pushed'
    git(REPO,'merge-base','--is-ancestor',pub['commit'],'HEAD')
    assert s['commits']=={'vllm-plugin-FL':PARENT['vllm-plugin-FL'],'FlagGems':GEMS}
    for name,commit in s['commits'].items():
        p=ROOT/'worktrees/EXP-0019'/name;assert git(p,'rev-parse','HEAD')==commit and not git(p,'status','--porcelain')
    info=dict(status='passed',source_run=run.name,publication=pub,checks=42,queries_checked=queries,timings=42,
      max_baseline_drift_percent=max(abs(t['baseline_drift_percent']) for t in d['timings']),
      groups={k:dict(count=len(v),min_percent=min(t['time_change_percent'] for t in v),max_percent=max(t['time_change_percent'] for t in v)) for k,v in groups.items()},
      hashes={n:sha(run/n) for n in ['status.json','validation.json','expanded_choice.json','validation.log']},selected_for_serving=False)
    return run,d,info

def main():
    assert not EXP.exists() and not WT.exists(),'Inspect prior preparation, do not overwrite'
    run,d,info=audit()
    assert read(ROOT/'artifacts/active_inference_plan.json')['selected_serving_commits']==PARENT
    assert not git(REPO,'diff','--cached','--name-only')
    EXP.mkdir();WT.mkdir(parents=True)
    (EXP/'README.md').write_text('''# EXP-0020：prefill页号广播接入官方4K测评

修改代码前的假设：EXP19独立扩大验证42项检查/39444个query/42工作点通过，prefill页号广播可减少重复页表读取并改善真实mixed阶段耗时。
从已选EXP16独立建树，仅在插件显式传入prefill_layout=1并增加日志；Gems复用EXP19冻结de958ddf，通用默认仍0。
prefill64行/4warps/stage1/tile32、decode8行/4warps/stage1/16segments/layout1、RoPE+KV融合及原调度保持。
插件逆向移除新增参数与日志必须逐字恢复父文件，支持条件/fallback不变。

正式测评前后要求真实42层GQA与RoPE+KV融合、35个FULL_DECODE_ONLY捕获尺寸、所有实际attention编译元数据与TTIR。
decode8行/18KiB shared，prefill64行/32KiB shared；两者均tile32且每tile读2个页号、int64广播到32个token lane。
CPU故障注入只验证检查器；不能代替真实服务路由。

先4K[4096,1024,64,256]，再16K[16384,1024,64,128]；原官方脚本与服务参数不变，各四轮弃首轮。
保留全部原始轮次；EXP16为直接历史对照，原始基线为累计对照。历史会话有漂移，4K父CV3.16%；小幅变化需交错复核。
4K若退化超过3%暂停16K并分析。完整评估前EXP16仍为已选性能基线，微基准不作为整模收益。
完整MATH500按用户要求暂缓，算子数值/Graph检查已完成。无KV压缩，无上游PR。
每组启动前单独记录新预算，4K至少55分钟、16K至少90分钟；今天关机时间未确认，暂以16:25为白天计算上限。
不延长EXP19旧窗口、不默认夜间运行；运行期间冻结脚本/代码，失败和结果自动保存并推送。
''')
    write(EXP/'parent_commits.json',PARENT);write(EXP/'expanded_audit.json',info)
    step('Record integration hypothesis after independent expanded-validation audit',audit=info)
    git(ROOT/'worktrees/EXP-0016/vllm-plugin-FL','worktree','add','-b',BRANCH,str(WT/'vllm-plugin-FL'),PARENT['vllm-plugin-FL'])
    git(ROOT/'worktrees/EXP-0019/FlagGems','worktree','add','--detach',str(WT/'FlagGems'),GEMS)
    rel='vllm_fl/dispatch/backends/flaggems/impl/iluvatar_gqa.py';p=WT/'vllm-plugin-FL'/rel;old=p.read_text()
    call='prefill_rows=64, prefill_warps=4, prefill_stages=1, prefill_tile=32)'
    newcall=call[:-1]+', prefill_layout=1)'
    log='prefill_rows=64 prefill_warps=4 prefill_stages=1 prefill_tile=32';newlog=log+' prefill_layout=1'
    assert old.count(call)==1 and old.count(log)==1
    new=old.replace(call,newcall).replace(log,newlog)
    assert new.replace(newcall,call).replace(newlog,log)==old
    node=next(n for n in ast.walk(ast.parse(new)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='paged_gqa')
    kwargs={k.arg:ast.literal_eval(k.value) for k in node.keywords};assert kwargs==d['config']
    p.write_text(new);subprocess.run(['python3','-m','py_compile',str(p)],check=True)
    assert subprocess.run(['git','-C',str(WT/'vllm-plugin-FL'),'grep','-n','KVReservationAsyncScheduler'],capture_output=True).returncode==1
    git(WT/'vllm-plugin-FL','add','--',rel);git(WT/'vllm-plugin-FL','diff','--cached','--check')
    git(WT/'vllm-plugin-FL','commit','--only','-m','perf(iluvatar): select validated prefill page-table broadcast','--',rel)
    commits={'vllm-plugin-FL':git(WT/'vllm-plugin-FL','rev-parse','HEAD'),'FlagGems':GEMS};write(EXP/'candidate_commits.json',commits)
    for n in FILES:shutil.copy2(STAGING/n,EXP/n)
    shutil.copytree(STAGING/'parent_exp16',EXP/'parent_exp16')
    for p in (EXP/'parent_exp16').iterdir():assert sha(p)==sha(REPO/'experiments/EXP-0016'/p.name)
    before={n:sha(EXP/n) for n in FILES};subprocess.run(['python3',str(EXP/'build_scripts.py')],check=True)
    assert before=={n:sha(EXP/n) for n in FILES}
    gate=dict(status='passed',candidate_commits=commits,launch=kwargs,diagnostic_hashes={str(run/'validation.json'):sha(run/'validation.json')},
      code_hashes={'vllm-plugin-FL':{rel:sha(WT/'vllm-plugin-FL'/rel)},'FlagGems':{'src/flag_gems/runtime/backend/_iluvatar/fused/paged_gqa.py':sha(WT/'FlagGems/src/flag_gems/runtime/backend/_iluvatar/fused/paged_gqa.py')}},
      validation_scope='42 all-query/Graph checks, 39444 queries, 42 timings; full-model task accuracy deferred',
      integration_cpu_check='Inverse one kwarg and log field restores parent exactly',script_hashes=before)
    write(EXP/'candidate_validation.json',gate)
    subprocess.run(['python3',str(EXP/'route_checks.py')],check=True)
    subprocess.run(['python3','-m','py_compile',*[str(p) for p in EXP.glob('*.py')]],check=True)
    checks=read(EXP/'route_cpu_checks.json');step('Freeze validated layout1 and pass route rejection checks',commits=commits,checks=len(checks['checks']))
    git(WT/'vllm-plugin-FL','push','leanscale','HEAD:refs/heads/'+BRANCH)
    for n,commit in commits.items():
        p=WT/n;url=git(p,'remote','get-url','--push','leanscale');assert git(p,'ls-remote',url,'refs/heads/'+BRANCH).split()[0]==commit
    write(EXP/'STATUS.json',dict(stage='prepared_for_official_4k',candidate_commits=commits,selected_for_serving=False))
    (EXP/'PR_DRAFT.md').write_text('''# Draft：赛事PR

标题：【FlagOS开放计算全球挑战赛S2.赛题2】【LeanScale】接入GQA prefill页号广播优化

MiniCPM5-2B Q16/KV2/D128因果注意力显式选择prefill_layout=1，减少同页token的重复页表读取。
原decode组合、RoPE+KV融合、支持边界、fallback和调度保持。FlagGems默认0保留。
EXP19通过42项全query数值/Graph检查、39444个query与42个计时点；纯prefill耗时减少16.76%–24.10%，mixed减少15.52%–18.80%，decode变化-0.07%..+0.43%。
真实42层接入和官方4K/16K测评待完成，尚无整模新增收益；完整任务精度暂缓，保持Draft，不创建上游PR。
''')
    with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        review=EX19/'expanded/review';review.mkdir(exist_ok=False);write(review/'audit.json',info)
        for n in ['expanded_window.json','expanded_preparation.json','expanded_preparation_error.json']:
            src=ROOT/'artifacts/EXP-0019'/n
            if src.exists():shutil.copy2(src,review/n)
        write(EX19/'STATUS.json',dict(stage='expanded_audited_integration_prepared',selected_for_serving=False,next_experiment='EXP-0020',audit=info))
        doc=REPO/'docs/gqa_prefill_memory_20261023.md'
        with doc.open('a') as f:
            f.write('\n\n## 扩大验证完成与引擎接入\n\n42项全query数值/Graph检查全部通过，变动前后39444个query独立FP32对照；含8项CU变化、两个dtype真实超过2^31元素KV地址。42个完整GQA夹测点全部保留，最大基线漂移2.4494%，无>3%退化。\n\n| 分组 | 点数 | 耗时变化 |\n|---|---:|---:|\n')
            for k,g in info['groups'].items():f.write(f"| {k} | {g['count']} | {g['min_percent']:+.3f}% .. {g['max_percent']:+.3f}% |\n")
            f.write('\nEXP18形状导出点使用CPU上下文上界与随机QKV，不是原请求重放。完整数据见[扩大验证](../experiments/EXP-0019/artifacts/'+run.name+'/RESULTS.md)。\n\nEXP20仅把prefill_layout=1显式接入插件，逆向移除参数与日志可恢复EXP16父文件。CPU路由拒绝检查通过；真实服务42层与官方4K/16K尚待测，服务选择仍EXP16。\n')
        with (REPO/'docs/optimization_journal.md').open('a') as f:f.write('\n\n### 2026-10-26：接续EXP19扩大验证结果，准备EXP20接入\n\n页号广播通过42项数值/Graph检查、39444个query和42计时点。纯prefill减少16.76%–24.10%，mixed减少15.52%–18.80%，decode最大退化0.43%，最大夹测漂移2.45%。插件仅新增prefill_layout=1和日志；真实路由与官方4K优先待测，尚未更换性能基线。\n')
        draft=EX19/'PR_DRAFT_FlagGems.md'
        with draft.open('a') as f:f.write('\n\n2026-10-26更新：四布局初筛80项检查/56个计时点完成；layout1扩大42项全query/Graph检查及42计时点通过，39444个query含实际int64地址。详见docs/gqa_prefill_memory_20261023.md。EXP20引擎接入准备完成，官方端到端收益与完整任务精度尚未确认。\n')
        paths=['experiments/EXP-0020','experiments/EXP-0019/steps.jsonl','experiments/EXP-0019/STATUS.json','experiments/EXP-0019/expanded/review','experiments/EXP-0019/PR_DRAFT_FlagGems.md','docs/gqa_prefill_memory_20261023.md','docs/optimization_journal.md']
        assert not git(REPO,'diff','--cached','--name-only')
        forbidden=re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|lab_inst_[A-Za-z0-9]+|ssh\.flagos\.io')
        for rel in paths:
            p=REPO/rel
            for x in p.rglob('*') if p.is_dir() else [p]:
                if x.is_file() and x.suffix!='.pyc':assert not forbidden.search(x.read_bytes()),str(x)
        git(REPO,'add','--',*paths);git(REPO,'diff','--cached','--check');assert '__pycache__' not in git(REPO,'diff','--cached','--name-only')
        git(REPO,'commit','--only','-m','experiments: audit prefill memory gains and integrate layout1 for official E2E','--',*paths)
        head=git(REPO,'rev-parse','HEAD');git(REPO,'push','origin','main')
        assert git(REPO,'ls-remote',git(REPO,'remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
    write(ROOT/'artifacts/EXP-0020-setup.json',dict(status='prepared',candidate_commits=commits,docs_commit=head,remote_verified=True,route_checks=len(checks['checks'])))
    active=read(ROOT/'artifacts/active_inference_plan.json');assert active['selected_serving_commits']==PARENT
    active.update(current_experiment='EXP-0020',candidate_commits=commits,status='Validated prefill page broadcast integrated; actual route and official E2E pending; EXP16 remains selected')
    write(ROOT/'artifacts/active_inference_plan.json',active)
    print(json.dumps(dict(status='prepared',commits=commits,docs_commit=head,checks=len(checks['checks']))),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        if EXP.is_dir():step('Integration preparation failed; preserve partial files',error=traceback.format_exc())
        write(ROOT/'artifacts/EXP-0020-preparation-error.json',dict(error=traceback.format_exc()));raise
