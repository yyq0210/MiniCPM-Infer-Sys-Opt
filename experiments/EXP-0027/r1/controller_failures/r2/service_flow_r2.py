"""EXP27: bounded real-model shadow validation, then a fresh official 4K case."""
from pathlib import Path
from datetime import datetime,timedelta
import argparse,fcntl,hashlib,importlib.util,json,os,re,shlex,signal,socket,subprocess,sys,time,traceback
ROOT=Path('/workspace/LeanScale');DOC=ROOT/'MiniCPM-Infer-Sys-Opt';EXP=DOC/'experiments/EXP-0027/r1'
ART=ROOT/'artifacts/EXP-0027-r1/service_r2';WT=ROOT/'worktrees/EXP-0027-r1';PARENT=DOC/'experiments/EXP-0020'
VALIDATION=Path(json.loads((ROOT/'artifacts/EXP-0027-r1/validation_launch.json').read_text())['run_directory'])
REFERENCE=ROOT/'artifacts/EXP-0025/20261029_160250_4_A_official_4k/candidate_benchmark'
BENCH_SHA='0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565'
def now():return datetime.now().astimezone()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def git(repo,*a):return subprocess.check_output(['git','-C',str(repo),*a],text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def step(action,**kw):
    with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(time=now().isoformat(),action=action,**kw))+'\n')
def publish(paths,message):
    sys.path.insert(0,str(DOC/'experiments/EXP-0025'));from common import publish as fn
    return fn(paths,message)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def verify_sources():
    for n,h in read(EXP/'service_plan_r2.json')['source_hashes'].items():assert sha(Path(n))==h,n
    commits=read(EXP/'candidate_commits.json')
    for n,h in commits.items():assert git(WT/n,'rev-parse','HEAD')==h and not git(WT/n,'status','--porcelain')
    for n,h in read(EXP/'implementation_sources.json').items():assert sha(WT/'vllm-plugin-FL'/n)==h
    assert read(VALIDATION/'status.json')['stage']=='passed' and read(VALIDATION/'status.json')['owned_children_reaped']
    assert read(VALIDATION/'code_publication.json')['commits']==commits and read(VALIDATION/'code_publication.json')['remote_verified']
    assert read(VALIDATION/'publication.json')['remote_verified']

def verify_environment(core):
    # This frozen helper resolves EXP4 via EXP.parent. Keep nested status paths
    # intact, and redirect only the environment check to the original root.
    saved = core.EXP
    try:
        core.EXP = PARENT
        core.verify_installed_environment()
    finally:
        core.EXP = saved
    assert core.EXP == saved

def prepare():
    bound=datetime.fromisoformat('2026-10-30T17:00:00+08:00')
    assert now().date()==bound.date() and now()+timedelta(minutes=125)<=bound, 'Refresh fresh afternoon run budget before preparation'
    assert not ART.exists() and not (EXP/'service_flow_r2.py').exists()
    assert read(VALIDATION/'result.json')['status']=='passed'
    checks=[json.loads(l) for l in (VALIDATION/'checks.jsonl').read_text().splitlines()]
    assert len(checks)==39 and all(r['passed'] is True for r in checks)
    archived=EXP/'validation'/VALIDATION.name
    for n,h in read(archived/'artifact_manifest.json').items():assert sha(VALIDATION/n)==h['raw_sha256'] and sha(archived/n)==h['archived_sha256']
    assert read(DOC/'experiments/EXP-0025/review_20261029/audit.json')['status']=='passed'
    assert sha(PARENT/'run_split_e2e.py')=='2e599679dfe365de907b5071cee469b08489b66ff371588bf78c0f1782281ab1'
    assert sha(PARENT/'analyze_schedule.py')=='574042b1d7d97454691b3608f2fe0073cc760a6c0314a1cd1315bc95a769ea9f'
    assert sha(DOC/'experiments/EXP-0025/layout_checks.py')=='b8915b5d6575a7ba4c9fa6eaffb5bee10e7d05988b41cbc6fd98baae35ba5e07'
    ART.mkdir(parents=True)
    preflight=ART/'environment_preflight';preflight.mkdir()
    checker=load('exp27_r2_preflight',PARENT/'run_split_e2e.py')
    checker.EXP=EXP;checker.RUN=preflight;checker.CANDIDATE_COMMITS=read(EXP/'candidate_commits.json')
    verify_environment(checker)
    assert checker.EXP==EXP and (preflight/'environment_check.json').exists()
    write(EXP/'environment_preflight_r2.json',dict(status='passed',checked_at=now().isoformat(),environment=read(preflight/'environment_check.json'),nested_status_root_preserved=True))
    step('Controller-only retry: original installed-source index restored for environment verification; preflight passed',source_candidate=read(EXP/'candidate_commits.json'),previous_attempt='service/20261030_134524_shadow')
    import shutil
    shutil.copy2(__file__,EXP/'service_flow_r2.py')
    write(EXP/'integration_review_r2.json',dict(status='passed',checks=39,checked_at=now().isoformat(),
        checks_sha256=sha(VALIDATION/'checks.jsonl'),candidate_commits=read(EXP/'candidate_commits.json'),
        code_publication=read(VALIDATION/'code_publication.json'),test_publication=read(VALIDATION/'publication.json')))
    sources=[EXP/'service_flow_r2.py',PARENT/'run_split_e2e.py',PARENT/'analyze_schedule.py',
        DOC/'experiments/EXP-0025/layout_checks.py',DOC/'experiments/EXP-0025/common.py',
        Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')]
    write(EXP/'service_plan_r2.json',dict(source_hashes={str(p):sha(p) for p in sources},
        diagnostic='one exact official inner4K workload (256 requests) with shadow=1; not a complete official case',
        official='new service/cache,shadow=0,official four rounds discard first',
        reference=str(REFERENCE),case=[4096,1024,64,256],diagnostic_max_minutes=35,official_max_minutes=70))
    with (EXP/'README.md').open('a') as f:f.write('\n\n## 实际服务流程\n\n39项独立引擎入口检查与归档hash已复核，r1候选提交由candidate_commits.json固定，已推个人fork。先shadow=1运行官方benchmark展开后的单轮4K/1024/64/256命令（只作诊断），要求计时区间真实B64/p=.95至少一次exact-mask和RNG检查。接线通过后停止服务、归档；再shadow=0独立新服务/缓存运行原官方脚本四轮弃首。诊断数据不合并进性能。原42MLP布局/GQA/RoPEKV/FULL捕获与实际top-p grid/counter/TTIR均检查。对照为历史EXP25 A2（3573.06 tok/s），并单列EXP20历史和ABBA均值。独立审核前不自动升级。\n')
    verify_sources();step('Audit39 integration checks; register real-service shadow then fresh official4K',commits=read(EXP/'candidate_commits.json'))
    pub=publish(['experiments/EXP-0027/r1'],'experiments: prepare real-model top-p shadow and official 4K')
    current=now();daily=datetime.fromisoformat('2026-10-30T17:00:00+08:00')
    assert current.date()==daily.date() and current+timedelta(minutes=115)<=daily,'Need full diagnostic+official budget'
    deadline=current+timedelta(minutes=115)
    window=dict(created_at=current.isoformat(),compute_stop_at=deadline.isoformat(),save_target_at=(deadline+timedelta(minutes=10)).isoformat(),
        run_upper_bound=daily.isoformat(),scope='Fresh afternoon run; self-imposed 17:00 upper bound, not a shutdown confirmation')
    write(ART/'run_window.json',window);write(EXP/'service_window_r2_at_launch.json',window)
    pub=publish(['experiments/EXP-0027/r1/service_window_r2_at_launch.json'],'experiments: allocate fresh top-p service comparison window')
    assert (deadline-now()).total_seconds()>110*60
    with (ART/'controller.log').open('w') as log:
        proc=subprocess.Popen(['python3',str(EXP/'service_flow_r2.py'),'--run'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    receipt=dict(pid=proc.pid,process_start_ticks=(Path('/proc')/str(proc.pid)/'stat').read_text().rsplit(') ',1)[1].split()[19],
        started_at=now().isoformat(),window=window,preparation_publication=pub)
    write(ART/'launch.json',receipt)
    active=read(ROOT/'artifacts/active_inference_plan.json');active.update(current_experiment='EXP-0027-r1',status='Real-service shadow then fresh official4K controller running',
        next_direction='Audit top-p official4K before any selection change',launch_receipt=str(ART/'launch.json'))
    write(ROOT/'artifacts/active_inference_plan.json',active)
    print(json.dumps(receipt),flush=True)

def sanitize(t):
    t=re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b','[IP_REDACTED]',t)
    t=re.sub(r'uuid=[^,)]+','uuid=[REDACTED]',t)
    return '\n'.join(l.rstrip() for l in t.splitlines())+'\n'
def archive(run):
    dest=EXP/'service_artifacts_r2'/run.name;dest.mkdir(parents=True,exist_ok=False)
    files=[p for p in run.iterdir() if p.is_file() and p.suffix in ['.json','.jsonl']]
    folder=run/'candidate_benchmark'
    files += [p for p in folder.iterdir() if p.is_file() and p.suffix in ['.json','.log','.prom']] if folder.exists() else []
    files += list((folder/'benchmark_results').glob('*.csv'))
    manifest={}
    for p in files:
        rel=p.relative_to(run);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(sanitize(p.read_text(errors='replace')))
        manifest[str(rel)]=dict(raw_sha256=sha(p),archived_sha256=sha(target))
    write(dest/'artifact_manifest.json',manifest)
    try:pub=publish(['experiments/EXP-0027/r1/service_artifacts_r2/'+run.name,'experiments/EXP-0027/r1/steps.jsonl','experiments/EXP-0027/r1/STATUS.json'],
        'experiments: archive top-p real-service stage '+run.name)
    except Exception as exc:pub=dict(status='push_failed_local_saved',error=repr(exc))
    write(run/'publication.json',pub)
    return pub

def execute():
    assert not (ART/'status.json').exists(),'One-shot flow, inspect partial completion'
    deadline=datetime.fromisoformat(read(ART/'run_window.json')['compute_stop_at'])
    assert now().date()==deadline.date() and (deadline-now()).total_seconds()>=110*60
    sys.path.insert(0,str(PARENT));core=load('exp27_core',PARENT/'run_split_e2e.py')
    sys.path.insert(0,str(DOC/'experiments/EXP-0025'));layout=load('exp27_parent_layout',DOC/'experiments/EXP-0025/layout_checks.py')
    commits=read(EXP/'candidate_commits.json');original_env=core.env
    STATE=dict(stage='preparing',started_at=now().isoformat(),pid=os.getpid(),deadline=deadline.isoformat(),completed_stages=[])
    def state(stage,**kw):STATE.update(stage=stage,updated_at=now().isoformat(),**kw);write(ART/'status.json',STATE);print(json.dumps(STATE),flush=True)
    def abort(*a):raise TimeoutError('Service phase deadline or termination signal')
    signal.signal(signal.SIGTERM,abort);signal.signal(signal.SIGALRM,abort)
    lock=Path('/workspace/experiments/codex_20261019/runner.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for phase,minutes in [('shadow',35),('official4k',70)]:
            assert (deadline-now()).total_seconds()>=minutes*60,'Insufficient complete-stage budget'
            run=ART/now().strftime('%Y%m%d_%H%M%S_'+phase);run.mkdir(exist_ok=False)
            phase_end=min(deadline,now()+timedelta(minutes=minutes));folder=run/'candidate_benchmark';server=None
            core.RUN=run;core.EXP=EXP;core.WORKTREES=WT;core.CANDIDATE_COMMITS=commits
            core.STATE=dict(pid=os.getpid(),started_at=now().isoformat(),phase=phase,commits=commits,planned_finish_before=phase_end.isoformat())
            def environment(candidate):
                env=original_env(candidate);env.update(VLLM_FL_TOPP_ROW_PARALLEL='1',VLLM_FL_TOPP_ROW_PARALLEL_SHADOW='1' if phase=='shadow' else '0',
                    VLLM_FL_MLP_WEIGHT_LAYOUT='0',TORCHINDUCTOR_CACHE_DIR=str(run/'inductor_cache'),
                    VLLM_FL_TOPP_ROW_PARALLEL_SHADOW_ARM_FILE=str(run/'shadow.arm') if phase=='shadow' else '')
                return env
            core.env=environment
            state('starting_'+phase,current_run=str(run),phase_deadline=phase_end.isoformat())
            step('Start top-p real service phase',phase=phase,run_name=run.name,deadline=phase_end.isoformat())
            signal.setitimer(signal.ITIMER_REAL,(phase_end-now()).total_seconds())
            try:
                verify_sources()
                for port in [9031,9032]:
                    with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',port))!=0
                idle=[]
                for _ in range(2):
                    out=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True,timeout=20)
                    m=re.search(r'(\d+)MiB\s*/\s*(\d+)MiB\s*\|\s*(\d+)%',out);assert m and int(m[1])<=1024 and int(m[3])<=5
                    idle.append(dict(memory_mib=int(m[1]),util_percent=int(m[3])));time.sleep(3)
                write(run/'idle.json',idle);verify_environment(core);assert sha(core.BENCH_FILE)==BENCH_SHA
                env=environment(True);write(run/'explicit_environment.json',{k:env[k] for k in ['PYTHONPATH','VLLM_PLUGINS','VLLM_CACHE_ROOT','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','VLLM_FL_TOPP_ROW_PARALLEL','VLLM_FL_TOPP_ROW_PARALLEL_SHADOW','VLLM_FL_TOPP_ROW_PARALLEL_SHADOW_ARM_FILE']})
                core.state('service_starting');server=core.launch(folder,True);core.smoke(folder)
                core.verify_route(folder);layout.layout_route(run,folder,REFERENCE,'A')
                log=(folder/'server.log').read_text();assert 'LeanScale top-p grid hook installed:' in log
                assert ('shadow=True' if phase=='shadow' else 'shadow=False') in log
                log_offset=len(log)
                write(run/'workload_start.json',dict(started_at=now().isoformat(),server_log_character_offset=log_offset))
                if phase=='shadow':
                    assert 'LeanScale top-p shadow passed:' not in log, 'Warmup consumed shadow checks'
                    (run/'shadow.arm').touch(exist_ok=False)
                    write(run/'arm_evidence.json',dict(armed_at=now().isoformat(),before_workload_dispatch=True,warmup_shadow_checks=0))
                core.metrics(folder,'metrics_before.prom');core.state('running_'+phase);state('running_'+phase)
                if phase=='shadow':
                    command=shlex.split(next(l for l in (REFERENCE/'benchmark.log').read_text().splitlines() if l.startswith('vllm bench serve ')))
                    for key,value in [('--random-input-len','4096'),('--random-output-len','1024'),('--max-concurrency','64'),('--num-prompts','256')]:assert command[command.index(key)+1]==value
                    assert '--temperature' not in command and '--top-p' not in command and '--ignore-eos' in command
                    core.run_command(command,folder,'diagnostic',timeout=(phase_end-now()).total_seconds())
                    text=(folder/'diagnostic.log').read_text()
                    assert len(re.findall(r'Successful requests:\s+256\b',text))==1 and len(re.findall(r'Failed requests:\s+0\b',text))==1
                    tail=(folder/'server.log').read_text()[log_offset:]
                    shadows=re.findall(r'LeanScale top-p shadow passed: B=(\d+) V=(\d+) stride=(.*?) p_min=([^ ]+) p_max=([^ ]+) exact_mask=True rng_unchanged=True check=(\d+)',tail)
                    assert any(int(r[0])==64 and int(r[1])==130560 and abs(float(r[3])-.95)<1e-6 and abs(float(r[4])-.95)<1e-6 for r in shadows),'No real-workload B64/.95 exact shadow evidence'
                    write(run/'shadow_evidence.json',dict(status='passed',successful_requests=256,failed_requests=0,
                        workload='one official inner workload, diagnostic only',checks_during_workload=shadows))
                else:
                    command=core.BENCH[:-1]+['[[4096,1024,64,256]]']
                    core.run_command(command,folder,'benchmark',timeout=(phase_end-now()).total_seconds())
                    after=core.validate_benchmark(folder,{(4096,1024,64,256)})
                    assert (folder/'benchmark.log').read_text().count('Failed requests:                         0')==4
                    before=core.validate_benchmark(REFERENCE,{(4096,1024,64,256)})
                    comparison=core.comparison(before,after)
                    write(run/'reference.json',dict(source=str(REFERENCE),raw_sha256=sha(next((REFERENCE/'benchmark_results').glob('raw_runs_*.csv'))),
                        summary_sha256=sha(next((REFERENCE/'benchmark_results').glob('summary_*.csv'))),
                        reference_scope='previous-day EXP25 A2, separate session; ABBA Amean3564.105 and historical EXP20 3579.12 are separate context'))
                    core.STATE.update(comparison=comparison)
                core.metrics(folder,'metrics_after.prom')
                if phase=='official4k':core.compare_schedule(REFERENCE,folder,run/'scheduler_comparison.json')
                core.verify_route(folder);layout.layout_route(run,folder,REFERENCE,'A');verify_sources();verify_environment(core)
                text=(folder/'server.log').read_text();tail=text[log_offset:]
                routes=re.findall(r'LeanScale top-p grid active: B=(\d+) V=(\d+) programs=(\d+) scratch_rows=(\d+) scratch_bytes=(\d+) k=None calls=(\d+) shadow=(True|False)',tail)
                assert routes and any(int(r[0])==64 and int(r[2])==64 for r in routes)
                assert all(int(r[1])==130560 and int(r[2])==int(r[0]) and int(r[3])==64 and int(r[4])==33423360 for r in routes)
                if phase=='official4k':
                    assert 'LeanScale top-p shadow passed:' not in text and all(r[6]=='False' for r in routes)
                    assert max(int(r[5]) for r in routes)>=1024
                kernels=[]
                for p in (run/'triton_cache').glob('*/_topk_topp_kernel.json'):
                    ir=p.with_suffix('.ttir');m=read(p);assert ir.exists() and m['target']['backend']=='corex'
                    kernels.append(dict(path=str(p.relative_to(run)),sha256=sha(p),ttir_sha256=sha(ir),num_warps=m['num_warps'],num_stages=m['num_stages'],shared=m.get('shared')))
                assert kernels
                write(folder/'topp_route_evidence.json',dict(status='passed',phase=phase,grid_events=routes,kernels=kernels,
                    unchanged_original_kernel=True,maximum_scratch_payload_bytes=33423360))
                core.state('complete_'+phase);step('Complete top-p real-service phase',phase=phase,run_name=run.name)
            except BaseException:
                core.state('failed',error=traceback.format_exc());step('Top-p service phase failed; no automatic retry',phase=phase,error=traceback.format_exc());raise
            finally:
                signal.setitimer(signal.ITIMER_REAL,0)
                try:core.stop(server);assert not core.ready()
                finally:
                    core.state(core.STATE['stage'],owned_children_reaped=not core.ready(),finished_at=now().isoformat())
                    pub=archive(run)
            STATE['completed_stages'].append(dict(phase=phase,run_directory=str(run),publication=pub))
            state('archived_'+phase)
        state('complete',selected_experiment='EXP-0020',selection_changed=False,reference_scope='previous-day context; fresh same-day comparison required before selection')
        write(ART/'publication.json',STATE['completed_stages'][-1]['publication'])
    except BaseException:
        state('failed',error=traceback.format_exc());traceback.print_exc()
    finally:signal.setitimer(signal.ITIMER_REAL,0);lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');args=p.parse_args()
    if args.run:execute()
    else:prepare()
