"""EXP24: actual-route gate, then the unchanged official four-round workload."""
from pathlib import Path
from datetime import datetime
import argparse, ast, fcntl, hashlib, importlib.util, json, os, re
import signal, socket, subprocess, sys, time, traceback

ROOT=Path('/workspace/LeanScale'); DOC=ROOT/'MiniCPM-Infer-Sys-Opt'
EXP=DOC/'experiments/EXP-0024'; ART=ROOT/'artifacts/EXP-0024'
WORK=ROOT/'worktrees/EXP-0024'; PARENT=DOC/'experiments/EXP-0020'
HELPERS={'run_split_e2e.py':'2e599679dfe365de907b5071cee469b08489b66ff371588bf78c0f1782281ab1',
         'analyze_schedule.py':'574042b1d7d97454691b3608f2fe0073cc760a6c0314a1cd1315bc95a769ea9f'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):
    temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');temp.replace(p)
def git(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def sanitize(text):
    text=re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b','[IP_REDACTED]',text)
    text=re.sub(r'uuid=[^,)]+','uuid=[REDACTED]',text)
    text=re.sub(r'GPU-[0-9a-fA-F-]{16,}','[GPU_ID_REDACTED]',text)
    return '\n'.join(line.rstrip() for line in text.splitlines())+'\n'
def step(action,**kw):
    with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(),action=action,**kw))+'\n')

def load_core():
    for name,h in HELPERS.items():assert sha(PARENT/name)==h
    sys.path.insert(0,str(PARENT))
    spec=importlib.util.spec_from_file_location('exp20_core',PARENT/'run_split_e2e.py')
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    return core

def validate_sources(run):
    plan=read(EXP/'e2e_plan.json')
    for name,h in plan['script_hashes'].items():assert sha(EXP/name)==h,name
    for name,commit in plan['candidate_commits'].items():
        assert git(WORK/name,'rev-parse','HEAD')==commit
        assert not git(WORK/name,'status','--porcelain')
    validation=ART/'validation_r3'
    assert read(validation/'status.json')['status']=='passed'
    assert read(validation/'status.json')['owned_children_reaped']
    assert read(validation/'publication.json')['remote_verified']
    assert read(validation/'code_publication.json')['commits']==plan['candidate_commits']
    for name,h in read(validation/'source_hashes.json').items():assert sha(WORK/'vllm-plugin-FL'/name)==h
    write(run/'candidate_validation.json',dict(status='passed',plan=plan,
         diagnostic_status_sha256=sha(validation/'status.json'),source_hashes=read(validation/'source_hashes.json')))
    return plan

def parent_reference(core,run,case,cases):
    selection=read(PARENT/'selection.json')
    assert selection['selected_experiment']=='EXP-0020'
    item=next(r for r in selection['case_results'] if r['case']==case)
    folder=ROOT/'artifacts/EXP-0020'/item['run_name']/'candidate_benchmark'
    raw=next((folder/'benchmark_results').glob('raw_runs_*.csv'))
    summary=next((folder/'benchmark_results').glob('summary_*.csv'))
    assert sha(raw)==item['raw_sha256'] and sha(summary)==item['summary_sha256']
    assert sha(folder/'route_evidence.json')==item['route_sha256']
    result=core.validate_benchmark(folder,cases)
    write(run/'parent_reference.json',dict(source=str(folder),raw=result[0],summary=result[1],
          selected_commits=selection['selected_serving_commits'],selection_sha256=sha(PARENT/'selection.json'),
          raw_sha256=sha(raw),summary_sha256=sha(summary),route_sha256=item['route_sha256']))
    return folder,result

def layout_route(run,folder,reference):
    log=(folder/'server.log').read_text(errors='replace')
    pattern=re.compile(r'LeanScale MLP layout active: prefix=model\.layers\.(\d+)\.mlp\.(gate_up_proj|down_proj) shape=\((\d+), (\d+)\) old_stride=\((\d+), (\d+)\) new_stride=\((\d+), (\d+)\) exact_values=True parameter_preserved=True bytes=(\d+)')
    matches=pattern.findall(log)
    assert len(matches)==84, ('Expected 84 one-time layout conversions',len(matches))
    expected={(str(i),p) for i in range(42) for p in ['gate_up_proj','down_proj']}
    assert {(r[0],r[1]) for r in matches}==expected
    for r in matches:
        n,k=(12288,2048) if r[1]=='gate_up_proj' else (2048,6144)
        assert tuple(map(int,r[2:]))==(n,k,k,1,1,n,n*k*2),r
    # Require generated matmuls to read engine weight arguments directly in
    # the new orientation. A fresh buffer/copy in the forward path is rejected.
    modules=[]
    for p in list((run/'candidate_cache').rglob('*.py'))+list((run/'inductor_cache').rglob('*.py')):
        text=p.read_text(errors='replace')
        if 'extern_kernels.mm' not in text:continue
        tree=ast.parse(text); inputs=set(); counts={'gate_up_proj':0,'down_proj':0}
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign) and isinstance(node.value,ast.Name) and node.value.id=='args':
                inputs.update(n.id for target in node.targets for n in ast.walk(target) if isinstance(n,ast.Name))
        for node in ast.walk(tree):
            if not (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                    and isinstance(node.func.value,ast.Name) and node.func.value.id=='extern_kernels'
                    and node.func.attr=='mm' and len(node.args)>=2):continue
            weight=node.args[1]
            if not (isinstance(weight,ast.Call) and isinstance(weight.func,ast.Name)
                    and weight.func.id=='reinterpret_tensor' and len(weight.args)>=3):continue
            try:shape=ast.literal_eval(weight.args[1]);stride=ast.literal_eval(weight.args[2])
            except (ValueError,TypeError):continue
            if shape not in [(2048,12288),(6144,2048)]:continue
            name='gate_up_proj' if shape==(2048,12288) else 'down_proj'
            assert stride==(shape[1],1),(p.name,name,stride)
            assert isinstance(weight.args[0],ast.Name) and weight.args[0].id in inputs, 'Forward weight packing detected'
            counts[name]+=1
        if any(counts.values()):
            modules.append(dict(path=str(p.relative_to(run)),sha256=sha(p),direct_weight_mm_counts=counts))
    assert any(m['direct_weight_mm_counts']=={'gate_up_proj':42,'down_proj':42} for m in modules), 'No complete 42-layer generated MLP layout proof'
    def capacity(p):
        values=re.findall(r'GPU KV cache size: ([\d,]+) tokens',p.read_text())
        assert len(values)==1,values
        return int(values[0].replace(',',''))
    old,new=capacity(reference/'server.log'),capacity(folder/'server.log')
    assert new>=.99*old, ('KV capacity regressed by more than predeclared 1%',old,new)
    evidence=dict(status='passed',converted_gate_up=42,converted_down=42,
                  exact_weight_values=True,parameter_objects_preserved=True,
                  one_time_conversions=84,compiled_modules=modules,
                  reference_kv_tokens=old,candidate_kv_tokens=new,
                  kv_capacity_change_percent=(new/old-1)*100)
    write(folder/'layout_route_evidence.json',evidence)
    return evidence

def archive(run,core):
    dest=EXP/'artifacts/runs'/run.name;dest.mkdir(parents=True,exist_ok=False)
    files=[p for p in run.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl','.log')]
    folder=run/'candidate_benchmark'
    for name in ['server.log','server.command.json','smoke.json','route_evidence.json','layout_route_evidence.json',
                 'benchmark.log','benchmark.command.json','metrics_before.prom','metrics_after.prom']:
        if (folder/name).is_file():files.append(folder/name)
    files+=list((folder/'benchmark_results').glob('*.csv'))
    manifest={}
    for p in files:
        target=dest/p.relative_to(run);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(sanitize(p.read_text(errors='replace')))
        manifest[str(p.relative_to(run))]=dict(raw_sha256=sha(p),archived_sha256=sha(target))
    write(dest/'artifact_manifest.json',manifest)
    result=core.STATE
    lines=['# EXP24 官方测评记录','',f"阶段：{result['stage']}。",'',
           '官方配置，四轮弃首；对照已审核的 EXP20 历史服务会话。整模任务精度按用户要求暂缓。',
           '只调整 MLP 权重加载布局；保留原 GQA、RoPE+KV、编译模式和调度。',
           '历史对照受运行时漂移影响，不能直接将增量视为稳定因果收益。','']
    if result['stage']=='performance_case_complete':
        c=read(run/'comparison.json')[0]
        lines += [f"EXP20：{c['baseline_total_tokens_s']:.2f} tok/s；本次：{c['candidate_total_tokens_s']:.2f} tok/s；变化 {c['throughput_change_percent']:+.2f}%。",
                  f"保留三轮：{c['candidate_retained_runs']}；CV {c['candidate_cv_percent']:.3f}%。"]
    else:lines += ['未完成整组，不报告端到端收益；错误与原始记录已保存。']
    (dest/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    step('Archive official MLP-layout attempt before publication',run_name=run.name,stage=result['stage'])
    with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);assert not git(DOC,'diff','--cached','--name-only')
        paths=['experiments/EXP-0024/artifacts/runs/'+run.name,'experiments/EXP-0024/STATUS.json','experiments/EXP-0024/steps.jsonl']
        forbidden=re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|lab_inst_[A-Za-z0-9]+|ssh\.flagos\.io')
        for rel in paths:
            p=DOC/rel
            for f in p.rglob('*') if p.is_dir() else [p]:
                if f.is_file():assert not forbidden.search(f.read_bytes()),str(f)
        git(DOC,'add','--',*paths);git(DOC,'diff','--cached','--check')
        git(DOC,'commit','--only','-m','experiments: archive official MLP weight layout run','--',*paths)
        head=git(DOC,'rev-parse','HEAD')
        for attempt in range(3):
            try:
                git(DOC,'push','origin','main')
                assert git(DOC,'ls-remote',git(DOC,'remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
                write(run/'publication.json',dict(commit=head,remote_verified=True));return
            except Exception:
                if attempt==2:raise
                time.sleep(10)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['4k','16k'],required=True)
    parser.add_argument('--run-dir',type=Path,required=True);args=parser.parse_args()
    run=args.run_dir.resolve();assert run.parent==ART;run.mkdir(exist_ok=False)
    policy=read(ART/('e2e_window_'+args.case+'.json'))
    end=datetime.fromisoformat(policy['compute_stop_at'])
    assert datetime.now().astimezone()<end and end.hour<=23
    core=load_core();core.RUN=run;core.EXP=EXP;core.WORKTREES=WORK
    core.CANDIDATE_COMMITS=read(EXP/'candidate_commits.json')
    original_env=core.env
    def candidate_env(candidate):
        env=original_env(candidate)
        env.update(VLLM_FL_MLP_WEIGHT_LAYOUT='1',TORCHINDUCTOR_CACHE_DIR=str(run/'inductor_cache'))
        return env
    core.env=candidate_env
    case=(4096,1024,64,256) if args.case=='4k' else (16384,1024,64,128)
    core.STATE.update(pid=os.getpid(),process_start_ticks=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19],
         started_at=datetime.now().astimezone().isoformat(),selected_cases=[list(case)],commits=core.CANDIDATE_COMMITS,
         planned_finish_before=end.isoformat(),accuracy_status='full-model task accuracy deferred by user')
    core.state('preparing')
    write(ART/('official_'+args.case+'_request.json'),dict(run_directory=str(run),**core.STATE))
    lock=Path('/workspace/experiments/codex_20261019/runner.lock').open('a');server=None
    def abort(*unused):raise TimeoutError('Experiment deadline or termination signal')
    signal.signal(signal.SIGTERM,abort);signal.signal(signal.SIGALRM,abort)
    signal.setitimer(signal.ITIMER_REAL,(end-datetime.now().astimezone()).total_seconds())
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        observations=[]
        for port in (9031,9032):
            with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port))!=0
        for _ in range(2):
            out=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True,timeout=20)
            m=re.search(r'(\d+)MiB\s*/\s*(\d+)MiB\s*\|\s*(\d+)%',out)
            assert m and int(m[1])<=1024 and int(m[3])<=5
            observations.append(dict(memory_mib=int(m[1]),util_percent=int(m[3])));time.sleep(3)
        write(run/'idle_evidence.json',observations)
        validate_sources(run);core.verify_installed_environment()
        assert sha(core.BENCH_FILE)==core.BENCH_SHA
        reference,before=parent_reference(core,run,args.case,{case})
        folder=run/'candidate_benchmark'
        core.state('candidate_official_starting')
        server=core.launch(folder,True);core.smoke(folder)
        core.verify_route(folder);layout_route(run,folder,reference)
        core.metrics(folder,'metrics_before.prom')
        core.state('candidate_throughput')
        command=core.BENCH[:-1]+[json.dumps([list(case)],separators=(',',':'))]
        core.run_command(command,folder,'benchmark',timeout=(end-datetime.now().astimezone()).total_seconds())
        core.metrics(folder,'metrics_after.prom')
        core.verify_route(folder);layout_route(run,folder,reference);validate_sources(run)
        assert sha(core.BENCH_FILE)==core.BENCH_SHA
        after=core.validate_benchmark(folder,{case})
        comparison=core.comparison(before,after)
        accounting=core.compare_schedule(reference,folder,run/'scheduler_comparison.json')
        core.state('performance_case_complete',comparison_exp20=comparison,scheduler_accounting=accounting)
    except BaseException:
        core.state('failed',error=traceback.format_exc());traceback.print_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        core.stop(server);lock.close()
        core.state(core.STATE['stage'],owned_children_reaped=True,finished_at=datetime.now().astimezone().isoformat())
        write(ART/('official_'+args.case+'_request.json'),dict(run_directory=str(run),**core.STATE))
    archive(run,core)

if __name__=='__main__':main()
