"""EXP29: fixed-code official 4K ABBA; new services/caches and per-arm archives."""
from pathlib import Path
from datetime import datetime, timedelta
import argparse, csv, fcntl, hashlib, importlib.util, json, os, re, signal, socket, statistics, subprocess, sys, time, traceback

ROOT = Path('/workspace/LeanScale')
DOC = ROOT / 'MiniCPM-Infer-Sys-Opt'
EXP = DOC / 'experiments/EXP-0029'
ART = ROOT / 'artifacts/EXP-0029'
PARENT = DOC / 'experiments/EXP-0020'
HELPERS = DOC / 'experiments/EXP-0025'
WT = ROOT / 'worktrees/EXP-0027-r1'
REFERENCE = ROOT / 'artifacts/EXP-0028/r1/20261030_155458_official4k_off/candidate_benchmark'
COMMITS = {'vllm-plugin-FL':'c50b5f0526e8112c72f1de918487300cf7376378', 'FlagGems':'de958ddfc4deeb43d6c94bb8652a3d4d3bc905fe'}
ORDER = ['A1','B1','B2','A2']
ENVKEYS = ['PYTHONPATH','VLLM_PLUGINS','VLLM_CACHE_ROOT','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR',
           'VLLM_FL_TOPP_ROW_PARALLEL','VLLM_FL_TOPP_ROW_PARALLEL_SHADOW','VLLM_FL_TOPP_ROW_PARALLEL_SHADOW_ARM_FILE','VLLM_FL_MLP_WEIGHT_LAYOUT']
ORIGINAL_TOPP_SHA = '64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8'
STOP = datetime.fromisoformat('2026-10-30T23:35:00+08:00')
SAVE = datetime.fromisoformat('2026-10-30T23:50:00+08:00')

def now(): return datetime.now().astimezone()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp=p.with_name(p.name+'.tmp'); tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n'); tmp.replace(p)
def git(p,*args): return subprocess.check_output(['git','-C',str(p),*args],text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def step(action,**kw):
    with (EXP/'steps.jsonl').open('a') as f: f.write(json.dumps(dict(time=now().isoformat(),action=action,**kw),ensure_ascii=False)+'\n')
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
def publish(paths,message):
    sys.path.insert(0,str(HELPERS))
    from common import publish as fn
    return fn(paths,message)

def make_core(run,arm):
    # The real child and preparation use the same module search paths.
    sys.path.insert(0,str(PARENT)); sys.path.insert(0,str(HELPERS))
    core=load('exp29_core_'+arm,PARENT/'run_split_e2e.py')
    layout=load('exp29_layout_'+arm,HELPERS/'layout_checks.py')
    core.EXP=EXP; core.RUN=run; core.WORKTREES=WT; core.CANDIDATE_COMMITS=COMMITS
    original=core.env
    def environment(candidate):
        value=original(candidate)
        value.update(VLLM_FL_TOPP_ROW_PARALLEL='1' if arm.startswith('B') else '0',
                     VLLM_FL_TOPP_ROW_PARALLEL_SHADOW='0',VLLM_FL_TOPP_ROW_PARALLEL_SHADOW_ARM_FILE='',
                     VLLM_FL_MLP_WEIGHT_LAYOUT='0',TORCHINDUCTOR_CACHE_DIR=str(run/'inductor_cache'))
        return value
    core.env=environment
    return core,layout

def verify():
    plan=read(EXP/'plan.json')
    for name,value in plan['source_hashes'].items(): assert sha(Path(name))==value,name
    for name,commit in COMMITS.items():
        assert git(WT/name,'rev-parse','HEAD')==commit and not git(WT/name,'status','--porcelain')
    for name,value in read(DOC/'experiments/EXP-0027/r1/implementation_sources.json').items():
        assert sha(WT/'vllm-plugin-FL'/name)==value,name
    assert sha(Path('/usr/local/lib/python3.12/site-packages/vllm/v1/sample/ops/topk_topp_triton.py'))==ORIGINAL_TOPP_SHA

GATE = """import os
import vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar
import vllm.v1.sample.ops.topk_topp_triton as b
import vllm.v1.sample.ops.topk_topp_sampler as s
from vllm_fl.patches import topp_row_parallel as p
expected=os.environ['VLLM_FL_TOPP_ROW_PARALLEL']=='1'
function=b.apply_top_k_top_p_triton
assert function is s.apply_top_k_top_p_triton
assert bool(getattr(function,'_leanscale_topp_row_parallel',False))==expected
if not expected:
    assert not p.install_topp_row_parallel()
    assert b.apply_top_k_top_p_triton is function and s.apply_top_k_top_p_triton is function
assert os.environ['VLLM_FL_TOPP_ROW_PARALLEL_SHADOW']=='0'
assert os.environ['VLLM_FL_TOPP_ROW_PARALLEL_SHADOW_ARM_FILE']==''
print('TOPP_ALIAS_GATE_PASSED enabled='+str(expected))
"""

def alias_gate(core,folder,arm):
    core.run_command(['python3','-c',GATE],folder,'alias_gate',timeout=180)
    expected='TOPP_ALIAS_GATE_PASSED enabled='+str(arm.startswith('B'))
    assert expected in (folder/'alias_gate.log').read_text()

def prepare():
    assert now().date()==STOP.date() and (STOP-now()).total_seconds()>=280*60, 'No full preparation and ABBA window'
    assert not ART.exists(); ART.mkdir()
    try:
        assert read(DOC/'experiments/EXP-0028/review_20261030/audit.json')['status']=='passed'
        assert read(ROOT/'artifacts/EXP-0028/r1/review_publication.json')['remote_verified']
        paths=[EXP/'run_abba.py',EXP/'protocol.json',PARENT/'run_split_e2e.py',PARENT/'analyze_schedule.py',
               HELPERS/'layout_checks.py',HELPERS/'common.py',
               DOC/'experiments/EXP-0027/r1/implementation_sources.json',
               Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')]
        write(EXP/'plan.json',dict(source_hashes={str(p):sha(p) for p in paths},commits=COMMITS,
              order=ORDER,case=[4096,1024,64,256],max_minutes_per_arm=60,archive_margin_minutes_per_arm=5,
              final_margin_minutes=10,automatic_selection=False))
        verify()
        for arm in ['A','B']:
            run=ART/('preflight_'+arm);run.mkdir()
            core,layout=make_core(run,arm)
            assert callable(core.launch) and callable(layout.layout_route)
            core.verify_installed_environment()
            write(run/'explicit_environment.json',{k:core.env(True)[k] for k in ENVKEYS})
            alias_gate(core,run,arm)
            core.run_command(['vllm','bench','serve','--help=all'],run,'client_help',timeout=180)
            assert '--num-prompts' in (run/'client_help.log').read_text()
        preflight=dict(status='passed',checked_at=now().isoformat(),real_dependency_imports=True,
                       actual_subprocess_aliases_and_cli_both_arms=True,model_loaded=False,
                       gpu_numerical_retests=False,source_hashes=read(EXP/'plan.json')['source_hashes'])
        write(EXP/'preflight.json',preflight)
        step('Pass real imports and A/B alias/client preflight before ABBA launch')
        result=publish(['experiments/EXP-0029'],'experiments: prepare fixed-code top-p ABBA controller')
        write(ART/'preparation_publication.json',result)
        print(json.dumps(dict(preflight=preflight,publication=result)),flush=True)
    except BaseException:
        failure=dict(time=now().isoformat(),error=traceback.format_exc(),gpu_benchmark_started=False)
        write(EXP/'prepare_failure.json',failure);write(ART/'prepare_failure.json',failure)
        step('ABBA preparation failed; preserve and do not launch',error=failure['error'])
        raise

def launch():
    verify()
    assert read(EXP/'preflight.json')['status']=='passed'
    assert read(ART/'preparation_publication.json')['remote_verified']
    assert not (ART/'launch.json').exists() and not (ART/'status.json').exists()
    assert now().date()==STOP.date() and (STOP-now()).total_seconds()>=270*60, 'No complete 4-arm budget'
    window=dict(created_at=now().isoformat(),compute_stop_at=STOP.isoformat(),save_target_at=SAVE.isoformat(),
                minimum_complete_flow_minutes=270,scope='Continuation authorized; recent recurring midnight schedule, daily change question pending')
    write(EXP/'run_window_at_launch.json',window);write(ART/'run_window.json',window)
    pub=publish(['experiments/EXP-0029/run_window_at_launch.json'],'experiments: allocate complete evening ABBA window')
    assert (STOP-now()).total_seconds()>=270*60
    with (ART/'controller.log').open('w') as log:
        proc=subprocess.Popen(['python3',str(EXP/'run_abba.py'),'--run'],stdout=log,stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL,start_new_session=True)
    receipt=dict(pid=proc.pid,process_start_ticks=(Path('/proc')/str(proc.pid)/'stat').read_text().rsplit(') ',1)[1].split()[19],
                 started_at=now().isoformat(),window=window,window_publication=pub)
    write(ART/'launch.json',receipt)
    p=ROOT/'artifacts/active_inference_plan.json';active=read(p)
    active.update(current_experiment='EXP-0029',status='Fixed-code top-p ABBA running; EXP20 still selected',
                  next_direction='Audit all four sessions against preregistered gates before 16K evaluation',
                  launch_receipt=str(ART/'launch.json'))
    write(p,active);print(json.dumps(receipt),flush=True)

def idle():
    samples=[]
    for _ in range(2):
        out=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True,timeout=20)
        m=re.search(r'(\d+)MiB\s*/\s*(\d+)MiB\s*\|\s*(\d+)%',out)
        assert m and int(m[1])<=1024 and int(m[3])<=5, 'GPU is not idle'
        samples.append(dict(memory_mib=int(m[1]),util_percent=int(m[3])));time.sleep(3)
    for port in [9031,9032]:
        with socket.socket() as sock: assert sock.connect_ex(('127.0.0.1',port))!=0
    return samples

def top_p_evidence(run,arm):
    folder=run/'candidate_benchmark';text=(folder/'server.log').read_text()
    tail=text[read(run/'workload_start.json')['server_log_character_offset']:]
    assert 'LeanScale top-p shadow passed:' not in text
    events=re.findall(r'LeanScale top-p grid active: B=(\d+) V=(\d+) programs=(\d+) scratch_rows=(\d+) scratch_bytes=(\d+) k=None calls=(\d+) shadow=(True|False)',tail)
    if arm.startswith('B'):
        assert 'LeanScale top-p grid hook installed:' in text and 'shadow=False' in text
        assert events and any(int(x[0])==64 for x in events)
        for b,v,programs,count,size,calls,shadow in events:
            assert 17<=int(b)<=64 and int(v)==130560 and int(programs)==int(b)
            assert int(count)==64 and int(size)==33423360 and shadow=='False'
        assert max(int(x[5]) for x in events)>=1024
    else:
        for forbidden in ['LeanScale top-p grid hook installed:','LeanScale top-p grid active:']:
            assert forbidden not in text
        assert not events
    kernels=[]
    for p in (run/'triton_cache').glob('*/_topk_topp_kernel.json'):
        ir=p.with_suffix('.ttir');metadata=read(p)
        assert ir.exists() and metadata['target']['backend']=='corex'
        kernels.append(dict(path=str(p.relative_to(run)),sha256=sha(p),ttir_sha256=sha(ir),
                            num_warps=metadata['num_warps'],num_stages=metadata['num_stages'],shared=metadata.get('shared')))
    assert kernels
    value=dict(status='passed',arm=arm,candidate_enabled=arm.startswith('B'),shadow_checks=0,
               aliases_gate_passed=True,grid_events=events,kernels=kernels,original_launcher_source_sha256=ORIGINAL_TOPP_SHA)
    write(folder/'topp_route_evidence.json',value)
    return value

def archive(run):
    dest=EXP/'artifacts'/run.name;dest.mkdir(parents=True,exist_ok=False)
    folder=run/'candidate_benchmark'
    files=[p for p in run.iterdir() if p.is_file() and p.suffix in ['.json','.jsonl']]
    if folder.exists(): files += [p for p in folder.iterdir() if p.is_file() and p.suffix in ['.json','.log','.prom','.error']]
    files += list((folder/'benchmark_results').glob('*.csv'))
    manifest={}
    for p in files:
        rel=p.relative_to(run);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
        text=re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b','[IP_REDACTED]',p.read_text(errors='replace'))
        text=re.sub(r'uuid=[^,)]+','uuid=[REDACTED]',text)
        target.write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n')
        manifest[str(rel)]=dict(raw_sha256=sha(p),archived_sha256=sha(target))
    write(dest/'artifact_manifest.json',manifest)
    try: pub=publish(['experiments/EXP-0029/artifacts/'+run.name,'experiments/EXP-0029/steps.jsonl','experiments/EXP-0029/STATUS.json'],
                     'experiments: archive top-p ABBA '+run.name)
    except Exception: pub=dict(status='push_failed_local_saved',error=traceback.format_exc(),remote_verified=False)
    write(run/'publication.json',pub)
    return pub

def aggregate(completed):
    assert [x['arm'] for x in completed]==ORDER
    sessions={}
    for item in completed:
        run=Path(item['run_directory']);folder=run/'candidate_benchmark'
        core,_=make_core(run,item['arm'])
        raw,summary=core.validate_benchmark(folder,{(4096,1024,64,256)})
        throughput=[float(r['Total tok/s']) for r in raw]
        def counter(p,key):
            return sum(float(line.rsplit(' ',1)[1]) for line in p.read_text().splitlines()
                       if line.startswith('vllm:'+key+'{') or line.startswith('vllm:'+key+' '))
        delta={key:counter(folder/'metrics_after.prom',key)-counter(folder/'metrics_before.prom',key)
               for key in ['num_preemptions_total','request_queue_time_seconds_count','request_queue_time_seconds_sum']}
        assert delta['request_queue_time_seconds_count']==1024
        kv=read(folder/'layout_route_evidence.json')['candidate_kv_tokens']
        sessions[item['arm']]=dict(run_directory=str(run),summary=summary[0],all_four_total_tokens_s=throughput,
              retained_cv_percent=statistics.stdev(throughput[1:])/statistics.mean(throughput[1:])*100,
              kv_tokens=kv,counter_delta=delta,mean_queue_seconds_all_four_rounds=delta['request_queue_time_seconds_sum']/1024)
    metric=lambda arm,key:float(sessions[arm]['summary'][key])
    means={group:{key:statistics.mean(metric(arm,key) for arm in ORDER if arm.startswith(group))
                  for key in ['Total tok/s','Output tok/s','Mean TTFT (ms)','Mean TPOT (ms)','Mean ITL (ms)']} for group in ['A','B']}
    gain=(means['B']['Total tok/s']/means['A']['Total tok/s']-1)*100
    pairs=[]
    for a,b in [('A1','B1'),('A2','B2')]:
        pair=dict(A=a,B=b,throughput_change_percent=(metric(b,'Total tok/s')/metric(a,'Total tok/s')-1)*100,
                  ttft_change_percent=(metric(b,'Mean TTFT (ms)')/metric(a,'Mean TTFT (ms)')-1)*100,
                  kv_ratio=sessions[b]['kv_tokens']/sessions[a]['kv_tokens'],
                  preemptions_A=sessions[a]['counter_delta']['num_preemptions_total'],
                  preemptions_B=sessions[b]['counter_delta']['num_preemptions_total'])
        pairs.append(pair)
    drift={group:(metric(group+'2','Total tok/s')/metric(group+'1','Total tok/s')-1)*100 for group in ['A','B']}
    gates=dict(mean_gain_gt_1_percent=gain>1,both_pair_gains_positive=all(p['throughput_change_percent']>0 for p in pairs),
               pair_ttft_regression_le_5_percent=all(p['ttft_change_percent']<=5 for p in pairs),
               mean_tpot_not_worse=means['B']['Mean TPOT (ms)']<=means['A']['Mean TPOT (ms)'],
               within_arm_summary_drift_le_2_percent=all(abs(x)<=2 for x in drift.values()),
               pair_kv_capacity_at_least_99_percent=all(p['kv_ratio']>=.99 for p in pairs),
               no_extra_preemptions=all(p['preemptions_B']<=p['preemptions_A'] for p in pairs))
    return dict(status='complete_pending_independent_audit',sessions=sessions,means=means,throughput_change_percent=gain,
                paired_comparisons=pairs,within_arm_summary_drift_percent=drift,registered_gates=gates,
                passes_registered_engineering_gates=all(gates.values()),selection_changed=False,selected_experiment='EXP-0020',
                successful_requests=4096,failed_requests=0,full_model_task_accuracy='deferred_by_user',
                interpretation='Engineering replication gate, not statistical significance; independent audit and 16K still required.')

def execute():
    assert not (ART/'status.json').exists()
    window=read(ART/'run_window.json');deadline=datetime.fromisoformat(window['compute_stop_at'])
    flow=dict(stage='preparing',started_at=now().isoformat(),pid=os.getpid(),completed_arms=[],selection_changed=False)
    def state(stage,**kw):
        flow.update(stage=stage,updated_at=now().isoformat(),**kw);write(ART/'status.json',flow)
        print(json.dumps(flow,ensure_ascii=False),flush=True)
    def abort(*args): raise TimeoutError('EXP29 compute deadline or termination signal')
    signal.signal(signal.SIGTERM,abort);signal.signal(signal.SIGALRM,abort)
    lock=Path('/workspace/experiments/codex_20261019/runner.lock').open('a')
    try:
        assert now().date()==deadline.date() and (deadline-now()).total_seconds()>=270*60
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
        for index,arm in enumerate(ORDER):
            needed=(len(ORDER)-index)*65+10
            assert (deadline-now()).total_seconds()>=needed*60,'Insufficient complete remaining arms budget'
            run=ART/now().strftime('%Y%m%d_%H%M%S_'+arm);run.mkdir()
            core,layout=make_core(run,arm);folder=run/'candidate_benchmark';server=None
            end=min(deadline,now()+timedelta(minutes=60))
            core.STATE=dict(pid=os.getpid(),started_at=now().isoformat(),arm=arm,commits=COMMITS,
                            planned_finish_before=end.isoformat(),selection_changed=False)
            state('starting_'+arm,current_arm=arm,current_run=str(run),arm_deadline=end.isoformat())
            step('Start fresh official ABBA arm',arm=arm,run_name=run.name)
            signal.setitimer(signal.ITIMER_REAL,(end-now()).total_seconds())
            try:
                verify();write(run/'idle.json',idle());core.verify_installed_environment()
                write(run/'explicit_environment.json',{k:core.env(True)[k] for k in ENVKEYS})
                alias_gate(core,folder,arm)
                core.state('starting');server=core.launch(folder,True);core.smoke(folder)
                core.verify_route(folder);layout.layout_route(run,folder,REFERENCE,'A')
                log=(folder/'server.log').read_text()
                assert ('LeanScale top-p grid hook installed:' in log)==arm.startswith('B')
                write(run/'workload_start.json',dict(started_at=now().isoformat(),server_log_character_offset=len(log)))
                core.metrics(folder,'metrics_before.prom');core.state('running_official4k');state('running_'+arm)
                core.run_command(core.BENCH[:-1]+['[[4096,1024,64,256]]'],folder,'benchmark',timeout=(end-now()).total_seconds())
                core.validate_benchmark(folder,{(4096,1024,64,256)})
                assert re.findall(r'Failed requests:\s+(\d+)',(folder/'benchmark.log').read_text())==['0']*4
                core.metrics(folder,'metrics_after.prom')
                core.verify_route(folder);layout.layout_route(run,folder,REFERENCE,'A');top_p_evidence(run,arm)
                verify();core.verify_installed_environment()
                core.state('complete_official4k');step('Complete all four official rounds',arm=arm,run_name=run.name)
            except BaseException:
                core.state('failed',error=traceback.format_exc());step('ABBA arm failed; no automatic retry',arm=arm,error=traceback.format_exc());raise
            finally:
                signal.setitimer(signal.ITIMER_REAL,0)
                try: core.stop(server)
                finally:
                    core.state(core.STATE['stage'],finished_at=now().isoformat(),owned_children_reaped=not core.ready())
                    pub=archive(run)
            assert core.STATE['owned_children_reaped'] and pub['remote_verified'], 'Stop after cleanup or archive failure'
            flow['completed_arms'].append(dict(arm=arm,run_directory=str(run),publication=pub))
            state('archived_'+arm)
        result=aggregate(flow['completed_arms']);write(EXP/'RESULTS.json',result);write(ART/'RESULTS.json',result)
        lines=['# EXP-0029 ABBA 结果（待独立审核）','','| 会话 | Total tok/s | 后三轮CV |','|---|---:|---:|']
        for arm in ORDER:
            s=result['sessions'][arm];lines.append('| '+arm+' | '+s['summary']['Total tok/s']+' | '+format(s['retained_cv_percent'],'.4f')+'% |')
        lines+=['','A均值 '+str(result['means']['A']['Total tok/s'])+'，B均值 '+str(result['means']['B']['Total tok/s'])+
                '；变化 '+format(result['throughput_change_percent'],'.4f')+'%。',
                '','[完整统计和预登记门槛](RESULTS.json)。此为控制器汇总，仍须独立审核；不自动替换EXP20，不等于统计显著或完整模型精度通过。']
        (EXP/'RESULTS.md').write_text('\n'.join(lines)+'\n')
        step('Complete four-arm fixed-code ABBA; independent audit pending',gain_percent=result['throughput_change_percent'],
             gates=result['registered_gates'],selection_changed=False)
        state('complete_pending_independent_audit',selected_experiment='EXP-0020',result_path=str(EXP/'RESULTS.json'))
        write(EXP/'FLOW_STATUS.json',flow)
        pub=publish(['experiments/EXP-0029/RESULTS.json','experiments/EXP-0029/RESULTS.md','experiments/EXP-0029/FLOW_STATUS.json','experiments/EXP-0029/steps.jsonl'],
                    'experiments: summarize fixed-code top-p ABBA without selection change')
        write(ART/'publication.json',pub)
        p=ROOT/'artifacts/active_inference_plan.json';active=read(p)
        active.update(status='EXP29 ABBA complete; independent audit pending; EXP20 selected',next_direction='Independently audit EXP29 before 16K evaluation')
        write(p,active)
    except BaseException:
        state('failed',error=traceback.format_exc());write(EXP/'FLOW_STATUS.json',flow)
        step('ABBA flow stopped; preserve completed and partial arms',error=flow['error'])
        try:
            pub=publish(['experiments/EXP-0029/FLOW_STATUS.json','experiments/EXP-0029/steps.jsonl'],'experiments: preserve ABBA flow interruption')
            write(ART/'failure_publication.json',pub)
        except Exception: traceback.print_exc()
        traceback.print_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);lock.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--launch',action='store_true');group.add_argument('--run',action='store_true')
    args=parser.parse_args()
    try:
        if args.prepare: prepare()
        elif args.launch: launch()
        else: execute()
    except BaseException:
        failure=dict(recorded_at=now().isoformat(),error=traceback.format_exc(),action='prepare' if args.prepare else 'launch' if args.launch else 'run')
        write(ART/('outer_'+failure['action']+'_failure.json'),failure)
        step('ABBA outer controller failure',failure=failure)
        raise
