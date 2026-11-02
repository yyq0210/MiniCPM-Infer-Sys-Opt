"""Independent CPU audit of EXP30 fixed-code 16K off/on; no GPU imports or producer writes."""
from pathlib import Path
from collections import Counter
from datetime import datetime
import argparse, ast, csv, hashlib, json, re, statistics, subprocess

ROOT=Path('/workspace/LeanScale')
DOC=ROOT/'MiniCPM-Infer-Sys-Opt'
EXP=DOC/'experiments/EXP-0030'
ART=ROOT/'artifacts/EXP-0030'
ORDER=['A','B']
COMMITS={'vllm-plugin-FL':'c50b5f0526e8112c72f1de918487300cf7376378','FlagGems':'de958ddfc4deeb43d6c94bb8652a3d4d3bc905fe'}
BENCH_SHA='0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565'
CONTROLLER_SHA='6e2097a5d6b1799e7240ea443a1bb345a31d9ed3e9b698ed9d082943ba988034'
HELPER_SHA='b82c526ce398875c7e715340bac2574f74572c814c650de610e8c0857c1a0ea0'
ORIGINAL_TOPP_SHA='64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def import_review_helpers():
    p=DOC/'experiments/EXP-0028/review_20261030/audit_off_control.py'
    assert sha(p)==HELPER_SHA
    names={'rows','counter','route_verifier'}
    nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==names
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),globals())

def benchmark(run):
    folder=run/'candidate_benchmark'
    raws=list((folder/'benchmark_results').glob('raw_runs_*.csv'))
    sums=list((folder/'benchmark_results').glob('summary_*.csv'))
    assert len(raws)==len(sums)==1
    raw,summary=rows(raws[0]),rows(sums[0])
    assert len(raw)==4 and len(summary)==1
    fields=['Prefill','Decode','Conc','Num Prompts']
    expected=(16384,1024,64,128)
    assert tuple(int(summary[0][k]) for k in fields)==expected
    for row in raw:
        assert tuple(int(row[k]) for k in fields)==expected
        assert row['Run Status']=='SUCCESS' and float(row['Successful Requests'])==128
        assert float(row['Total Input Tokens'])==2097152 and float(row['Total Output Tokens'])==131072
    for key,value in summary[0].items():
        if key not in fields:
            assert abs(float(value)-statistics.mean(float(row[key]) for row in raw[1:]))<=.011,(key,value)
    command=read(folder/'benchmark.command.json')
    assert command[1]=='/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py'
    assert json.loads(command[command.index('--test-cases')+1])==[[16384,1024,64,128]]
    log=(folder/'benchmark.log').read_text()
    for label,value in [('Successful requests',128),('Failed requests',0),('Total input tokens',2097152),('Total generated tokens',131072)]:
        assert re.findall(label+r':\s+(\d+)',log)==[str(value)]*4,label
    retained=[float(row['Total tok/s']) for row in raw[1:]]
    return dict(raw_sha256=sha(raws[0]),summary_sha256=sha(sums[0]),summary=summary[0],
                all_four_total_tokens_s=[float(row['Total tok/s']) for row in raw],
                retained_total_tokens_s=retained,retained_cv_percent=statistics.stdev(retained)/statistics.mean(retained)*100,
                retained_first_to_last_change_percent=(retained[-1]/retained[0]-1)*100,successful_requests=512,failed_requests=0)

def schedule(run):
    folder=run/'candidate_benchmark'
    delta={k:counter(folder/'metrics_after.prom',k)-counter(folder/'metrics_before.prom',k)
           for k in ['num_preemptions_total','request_queue_time_seconds_count','request_queue_time_seconds_sum']}
    assert delta['request_queue_time_seconds_count']==512
    peaks=[float(x) for x in re.findall(r'GPU KV cache usage: ([\d.]+)%',(folder/'server.log').read_text())]
    assert peaks
    return dict(counter_delta=delta,mean_queue_seconds_all_four_rounds=delta['request_queue_time_seconds_sum']/512,
                sampled_peak_kv_percent=max(peaks))

def verify_sources():
    assert sha(EXP/'run_pair.py')==CONTROLLER_SHA
    for name,value in read(EXP/'plan.json')['source_hashes'].items():assert sha(Path(name))==value,name
    assert read(EXP/'protocol.json')['order']==ORDER
    assert read(EXP/'preflight.json')['status']=='passed'
    for name,commit in COMMITS.items():
        p=ROOT/'worktrees/EXP-0027-r1'/name
        assert subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()==commit
        assert not subprocess.check_output(['git','-C',str(p),'status','--porcelain'],text=True).strip()
    for name,value in read(DOC/'experiments/EXP-0027/r1/implementation_sources.json').items():
        assert sha(ROOT/'worktrees/EXP-0027-r1/vllm-plugin-FL'/name)==value
    assert sha(Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py'))==BENCH_SHA

def inspect(entry):
    run=Path(entry['run_directory']);arm=entry['arm'];assert arm in ORDER and run.parent==ART
    status=read(run/'status.json')
    assert status['arm']==arm and status['commits']==COMMITS and not status['selection_changed']
    assert status['stage']=='complete_official16k' and status['owned_children_reaped']
    assert datetime.fromisoformat(status['finished_at'])<datetime.fromisoformat(status['planned_finish_before'])
    publication=read(run/'publication.json');assert publication==entry['publication'] and publication['remote_verified']
    archive=EXP/'artifacts'/run.name;manifest=read(archive/'artifact_manifest.json')
    required={'status.json','environment_check.json','explicit_environment.json',
              'candidate_benchmark/alias_gate.log','candidate_benchmark/server.log','candidate_benchmark/benchmark.log',
              'candidate_benchmark/topp_route_evidence.json','candidate_benchmark/metrics_before.prom','candidate_benchmark/metrics_after.prom'}
    assert required<=set(manifest)
    for name,value in manifest.items():
        assert sha(run/name)==value['raw_sha256'],name
        assert sha(archive/name)==value['archived_sha256'],name
    folder=run/'candidate_benchmark';env=read(run/'explicit_environment.json')
    enabled=arm.startswith('B')
    assert env['VLLM_FL_TOPP_ROW_PARALLEL']==str(int(enabled))
    assert env['VLLM_FL_TOPP_ROW_PARALLEL_SHADOW']=='0' and env['VLLM_FL_TOPP_ROW_PARALLEL_SHADOW_ARM_FILE']==''
    assert env['VLLM_FL_MLP_WEIGHT_LAYOUT']=='0'
    for key,name in [('VLLM_CACHE_ROOT','candidate_cache'),('TRITON_CACHE_DIR','triton_cache'),('TORCHINDUCTOR_CACHE_DIR','inductor_cache')]:
        assert env[key]==str(run/name)
    gate=(folder/'alias_gate.log').read_text()
    assert 'TOPP_ALIAS_GATE_PASSED enabled='+str(enabled) in gate
    command=read(folder/'alias_gate.command.json')
    assert command[:2]==['python3','-c'] and 'load_general_plugins()' in command[2]
    assert 'function is s.apply_top_k_top_p_triton' in command[2]
    server=read(folder/'server.command.json')
    assert json.loads(server[server.index('--compilation-config')+1])=={'cudagraph_mode':'FULL_DECODE_ONLY'}
    assert server[server.index('--gpu-memory-utilization')+1]=='0.85' and server[server.index('--max-model-len')+1]=='131072'
    installed=read(run/'environment_check.json')
    assert installed['candidate_commits']==COMMITS and installed['benchmark_sha256']==BENCH_SHA
    assert installed['versions']=={'torch':'2.10.0+corex.4.5.0','triton':'3.6.0','vllm':'0.24.0+empty'}
    for name,value in installed['installed_source_hashes'].items():assert sha(Path('/usr/local/lib/python3.12/site-packages')/name)==value
    verifier,verifier_sha=route_verifier();original_route=verifier(run,'A')
    evidence=read(folder/'topp_route_evidence.json')
    assert evidence['status']=='passed' and evidence['arm']==arm and evidence['candidate_enabled']==enabled
    assert evidence['aliases_gate_passed'] and evidence['shadow_checks']==0
    assert evidence['original_launcher_source_sha256']==ORIGINAL_TOPP_SHA
    assert sha(Path('/usr/local/lib/python3.12/site-packages/vllm/v1/sample/ops/topk_topp_triton.py'))==ORIGINAL_TOPP_SHA
    log=(folder/'server.log').read_text()
    assert 'LeanScale top-p shadow passed:' not in log
    assert ('LeanScale top-p grid hook installed:' in log)==enabled
    offset=read(run/'workload_start.json')['server_log_character_offset']
    assert 0<=offset<=len(log)
    events=re.findall(r'LeanScale top-p grid active: B=(\d+) V=(\d+) programs=(\d+) scratch_rows=(\d+) scratch_bytes=(\d+) k=None calls=(\d+) shadow=(True|False)',log[offset:])
    assert [list(x) for x in events]==evidence['grid_events']
    if enabled:
        assert events and any(17<=int(x[0])<=64 for x in events)
        assert max(int(x[5]) for x in events)>=1024
        for b,v,programs,count,size,calls,shadow in events:
            assert 17<=int(b)<=64 and int(v)==130560 and int(programs)==int(b)
            assert int(count)==64 and int(size)==33423360 and shadow=='False'
    else:
        assert not events and 'LeanScale top-p grid active:' not in log
    assert evidence['kernels']
    for k in evidence['kernels']:
        p=run/k['path'];assert p.resolve().is_relative_to(run)
        assert sha(p)==k['sha256'] and sha(p.with_suffix('.ttir'))==k['ttir_sha256']
        metadata=read(p);assert metadata['target']['backend']=='corex'
        assert all(metadata[key]==k[key] for key in ['num_warps','num_stages','shared'])
    measured=benchmark(run);scheduling=schedule(run)
    for path in (folder/'benchmark_results').glob('*.csv'):
        assert str(path.relative_to(run)) in manifest
    return dict(status='passed',arm=arm,run_directory=str(run),started_at=status['started_at'],finished_at=status['finished_at'],
                archive_files_checked=len(manifest),publication=publication,original_route=original_route,
                route_verifier_sha256=verifier_sha,top_p_variants=len(evidence['kernels']),top_p_grid_events=len(events),
                max_logged_candidate_calls=max((int(x[5]) for x in events),default=0),shadow_checks=0,
                measured=measured,scheduling=scheduling)

def summarize(sessions):
    assert list(sessions)==ORDER
    keys=['Total tok/s','Output tok/s','Mean TTFT (ms)','Mean TPOT (ms)','Mean ITL (ms)']
    means={arm:{k:float(sessions[arm]['measured']['summary'][k]) for k in keys} for arm in ORDER}
    changes={k:(means['B'][k]/means['A'][k]-1)*100 for k in keys}
    kv_ratio=sessions['B']['original_route']['kv_tokens']/sessions['A']['original_route']['kv_tokens']
    gates=dict(throughput_regression_le_1_percent=changes['Total tok/s']>=-1,
               ttft_regression_le_5_percent=changes['Mean TTFT (ms)']<=5,
               tpot_regression_le_2_percent=changes['Mean TPOT (ms)']<=2,
               kv_capacity_at_least_99_percent=kv_ratio>=.99,
               no_extra_preemptions=sessions['B']['scheduling']['counter_delta']['num_preemptions_total']<=sessions['A']['scheduling']['counter_delta']['num_preemptions_total'])
    producer=read(EXP/'RESULTS.json')
    assert producer['means']==means and producer['B_on_vs_A_off_percent']==changes
    assert producer['kv_ratio']==kv_ratio and producer['registered_gates']==gates
    assert abs(producer['throughput_change_percent']-changes['Total tok/s'])<1e-9
    assert producer['passes_registered_engineering_gates']==all(gates.values())
    assert producer['successful_requests']==1024 and producer['failed_requests']==0 and not producer['selection_changed']
    for arm,item in sessions.items():
        stored=producer['sessions'][arm];m=item['measured'];scheduler=item['scheduling']
        assert stored['summary']==m['summary'] and stored['all_four_total_tokens_s']==m['all_four_total_tokens_s']
        assert abs(stored['retained_cv_percent']-m['retained_cv_percent'])<1e-9
        assert stored['counter_delta']==scheduler['counter_delta']
        assert abs(stored['mean_queue_seconds_all_four_rounds']-scheduler['mean_queue_seconds_all_four_rounds'])<1e-9
        assert stored['kv_tokens']==item['original_route']['kv_tokens']
    assert datetime.fromisoformat(sessions['A']['finished_at'])<datetime.fromisoformat(sessions['B']['started_at'])
    return dict(means=means,B_on_vs_A_off_percent=changes,kv_ratio=kv_ratio,registered_engineering_gates=gates,
                passes_registered_engineering_gates=all(gates.values()),successful_requests=1024,failed_requests=0)

def audit(arm=None):
    import_review_helpers();verify_sources()
    flow=read(ART/'status.json')
    entries=flow['completed_arms']
    if arm:
        entry=next(x for x in entries if x['arm']==arm)
        return dict(status='passed_completed_arm_only',checked_at=datetime.now().astimezone().isoformat(),
                    source_commits=COMMITS,helper_sha256=HELPER_SHA,session=inspect(entry),whole_pair_audited=False,selection_changed=False)
    assert flow['stage']=='complete_pending_independent_audit' and [x['arm'] for x in entries]==ORDER
    assert read(ART/'publication.json')['remote_verified']
    sessions={entry['arm']:inspect(entry) for entry in entries}
    result=summarize(sessions)
    return dict(status='passed',checked_at=datetime.now().astimezone().isoformat(),source_commits=COMMITS,
                helper_sha256=HELPER_SHA,sessions=sessions,**result,selection_changed=False,selected_experiment='EXP-0020',
                full_model_task_accuracy='deferred_by_user',candidate_16k='official_pair_audited',
                scope='One preregistered sequential A/B pair; 16K nonregression screen, not a significance test.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ORDER);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assert not args.output.exists()
    result=audit(args.arm)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))
