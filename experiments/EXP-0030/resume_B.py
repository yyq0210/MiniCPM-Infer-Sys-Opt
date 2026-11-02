"""Resume only unstarted B after EXP30 A completed and publication interrupted."""
from pathlib import Path
from datetime import datetime,timedelta
import argparse,fcntl,hashlib,importlib.util,json,os,signal,subprocess,time,traceback

ROOT=Path('/workspace/LeanScale')
DOC=ROOT/'MiniCPM-Infer-Sys-Opt'
EXP=DOC/'experiments/EXP-0030'
ART=ROOT/'artifacts/EXP-0030'
REC=ART/'recovery_20261102'
A=ART/'20261102_165006_A'
STOP=datetime.fromisoformat('2026-11-02T23:35:00+08:00')
SAVE=datetime.fromisoformat('2026-11-02T23:50:00+08:00')
def now():return datetime.now().astimezone()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
r=load('exp30_frozen',EXP/'run_pair.py')
audit=load('exp30_recovery_review',EXP/'review_20261102/audit_pair_recovered.py')

def configure_publication():
    result=read(ART/'publication_network_recovery.json')['result']
    options=' -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=10 -o ServerAliveCountMax=2'
    os.environ['GIT_SSH_COMMAND']=('ssh -p 443 -o Hostname=ssh.github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes' if result.get('route')=='github443' else 'ssh')+options

_original_publish=r.publish
def saved(pub):return pub.get('remote_verified') or pub.get('local_commit_verified')
def safe_publish(paths,message):
    try:return _original_publish(paths,message)
    except Exception:
        # The user's local-save fallback applies only after a verified local commit.
        status=r.git(DOC,'status','--porcelain','--',*paths)
        assert not status, 'Publication failed before local commit: '+status
        head=r.git(DOC,'rev-parse','HEAD')
        return dict(status='local_committed_pending_push',commit=head,remote_verified=False,local_commit_verified=True,error=traceback.format_exc())
r.publish=safe_publish

def prior_stopped():
    launch=read(ART/'launch.json')
    p=Path('/proc')/str(launch['pid'])/'stat'
    parts=p.read_text().rsplit(') ',1)[1].split() if p.exists() else None
    assert parts is None or parts[19]!=launch['process_start_ticks'] or parts[0]=='Z','Original controller still active'
    assert read(ART/'status.json')['stage']=='failed'
    assert read(A/'status.json')['stage']=='complete_official16k'
    assert read(A/'status.json')['owned_children_reaped']

def A_entry():
    return dict(arm='A',run_directory=str(A),publication=read(A/'publication_recovered.json')['publication'])

def verify():
    r.verify();prior_stopped()
    for name,digest in read(EXP/'recovery_plan.json')['source_hashes'].items():assert sha(Path(name))==digest,name
    assert read(EXP/'recovery_protocol.json')['order']==['A','B']

def prepare():
    assert now().date()==STOP.date() and (STOP-now()).total_seconds()>=130*60
    assert not REC.exists();REC.mkdir()
    configure_publication();prior_stopped()
    net=read(ART/'publication_network_recovery.json')['result']
    if not net['remote_verified']:
        assert not r.git(DOC,'diff','HEAD','--','experiments/EXP-0030/artifacts/'+A.name)
        net=dict(status='local_committed_pending_push',commit=r.git(DOC,'rev-parse','HEAD'),remote_verified=False,local_commit_verified=True)
    write(A/'publication_recovered.json',dict(original_publication_sha256=sha(A/'publication.json'),publication=net,
         note='Original failure unchanged; archive verified against local commit; remote publication recorded separately'))
    paths=[EXP/'resume_B.py',EXP/'recovery_protocol.json',EXP/'review_20261102/audit_pair_recovered.py']
    write(EXP/'recovery_plan.json',dict(source_hashes={str(p):sha(p) for p in paths},original_controller_sha256=sha(EXP/'run_pair.py')))
    verify();audit.import_review_helpers();audit.verify_sources()
    result=audit.inspect(A_entry())
    write(EXP/'review_20261102/A_recovery.json',dict(status='passed_completed_arm_only',checked_at=now().isoformat(),session=result,whole_pair_audited=False))
    write(EXP/'recovery_preflight.json',dict(status='passed',checked_at=now().isoformat(),A_independently_audited=True,no_A_rerun=True,
         reused_real_alias_and_cli_preflight=str(EXP/'preflight.json'),source_changes=False,remaining_arm='B'))
    r.step('Independent A audit passed after publication recovery; freeze manual B-only continuation',A_total_tokens_s=result['measured']['summary']['Total tok/s'])
    pub=r.publish(['experiments/EXP-0030/resume_B.py','experiments/EXP-0030/recovery_protocol.json',
       'experiments/EXP-0030/recovery_plan.json','experiments/EXP-0030/recovery_preflight.json',
       'experiments/EXP-0030/review_20261102/audit_pair_recovered.py','experiments/EXP-0030/review_20261102/A_recovery.json',
       'experiments/EXP-0030/steps.jsonl'],'experiments: preserve publication interruption and prepare B-only continuation')
    write(REC/'preparation_publication.json',pub)
    print(json.dumps(dict(status='prepared',publication=pub,A_total_tokens_s=result['measured']['summary']['Total tok/s'])),flush=True)

def launch():
    configure_publication();verify()
    assert read(EXP/'recovery_preflight.json')['status']=='passed' and saved(read(REC/'preparation_publication.json'))
    assert not (REC/'launch.json').exists() and not (REC/'status.json').exists()
    assert now().date()==STOP.date() and (STOP-now()).total_seconds()>=120*60
    window=dict(created_at=now().isoformat(),compute_stop_at=STOP.isoformat(),save_target_at=SAVE.isoformat(),
                minimum_remaining_flow_minutes=120,scope='Nov2 user-confirmed cutoff; completed A retained, only B unstarted')
    write(EXP/'recovery_window.json',window)
    pub=r.publish(['experiments/EXP-0030/recovery_window.json'],'experiments: allocate B continuation window')
    assert (STOP-now()).total_seconds()>=120*60
    with (REC/'controller.log').open('w') as f:
        p=subprocess.Popen(['python3',str(EXP/'resume_B.py'),'--run'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(pid=p.pid,process_start_ticks=(Path('/proc')/str(p.pid)/'stat').read_text().rsplit(') ',1)[1].split()[19],
                 started_at=now().isoformat(),window=window,publication=pub)
    write(REC/'launch.json',receipt)
    active=read(ROOT/'artifacts/active_inference_plan.json')
    active.update(current_experiment='EXP-0030',status='B-only continuation after A publication interruption; EXP20 selected',
                  launch_receipt=str(REC/'launch.json'),next_direction='Independently audit full 16K pair; then focus 4K')
    write(ROOT/'artifacts/active_inference_plan.json',active)
    print(json.dumps(receipt),flush=True)

def execute():
    configure_publication();verify()
    assert not (REC/'status.json').exists()
    flow=dict(stage='preparing_B',started_at=now().isoformat(),pid=os.getpid(),completed_arms=[A_entry()],
              selection_changed=False,recovery_reason='A fully completed; remote archive timeout before B launch; user-authorized verified local-save fallback')
    def state(stage,**kw):
        flow.update(stage=stage,updated_at=now().isoformat(),**kw);write(REC/'status.json',flow);print(json.dumps(flow),flush=True)
    def abort(*args):raise TimeoutError('B continuation deadline or termination signal')
    signal.signal(signal.SIGTERM,abort);signal.signal(signal.SIGALRM,abort)
    lock=Path('/workspace/experiments/codex_20261019/runner.lock').open('a')
    try:
        assert now().date()==STOP.date() and (STOP-now()).total_seconds()>=120*60
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run=ART/now().strftime('%Y%m%d_%H%M%S_B');run.mkdir()
        core,layout=r.make_core(run,'B');folder=run/'candidate_benchmark';server=None
        end=min(STOP,now()+timedelta(minutes=105))
        core.STATE=dict(pid=os.getpid(),started_at=now().isoformat(),arm='B',commits=r.COMMITS,
                        planned_finish_before=end.isoformat(),selection_changed=False)
        state('starting_B',current_arm='B',current_run=str(run),arm_deadline=end.isoformat())
        r.step('Start only unstarted B with original official16K parameters',run_name=run.name)
        signal.setitimer(signal.ITIMER_REAL,(end-now()).total_seconds())
        try:
            verify();write(run/'idle.json',r.idle());core.verify_installed_environment()
            write(run/'explicit_environment.json',{k:core.env(True)[k] for k in r.ENVKEYS})
            r.alias_gate(core,folder,'B');core.state('starting')
            server=core.launch(folder,True);core.smoke(folder)
            core.verify_route(folder);layout.layout_route(run,folder,r.REFERENCE,'A')
            log=(folder/'server.log').read_text();assert 'LeanScale top-p grid hook installed:' in log
            write(run/'workload_start.json',dict(started_at=now().isoformat(),server_log_character_offset=len(log)))
            core.metrics(folder,'metrics_before.prom');core.state('running_official16k');state('running_B')
            core.run_command(core.BENCH[:-1]+['[[16384,1024,64,128]]'],folder,'benchmark',timeout=(end-now()).total_seconds())
            core.validate_benchmark(folder,{(16384,1024,64,128)})
            assert r.re.findall(r'Failed requests:\s+(\d+)',(folder/'benchmark.log').read_text())==['0']*4
            core.metrics(folder,'metrics_after.prom')
            core.verify_route(folder);layout.layout_route(run,folder,r.REFERENCE,'A');r.top_p_evidence(run,'B')
            verify();core.verify_installed_environment()
            core.state('complete_official16k');r.step('Complete B four official rounds',run_name=run.name)
        except BaseException:
            core.state('failed',error=traceback.format_exc());r.step('B continuation failed; preserve, no automatic retry',error=traceback.format_exc());raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            try:core.stop(server)
            finally:
                core.state(core.STATE['stage'],finished_at=now().isoformat(),owned_children_reaped=not core.ready())
                pub=r.archive(run)
        assert core.STATE['owned_children_reaped'] and saved(pub)
        flow['completed_arms'].append(dict(arm='B',run_directory=str(run),publication=pub))
        result=r.aggregate(flow['completed_arms'])
        write(EXP/'RESULTS.json',result);write(REC/'RESULTS.json',result)
        state('complete_pending_independent_audit',result_path=str(EXP/'RESULTS.json'),selected_experiment='EXP-0020')
        write(EXP/'RECOVERY_FLOW_STATUS.json',flow)
        r.step('Both original arms complete; independent audit pending',throughput_change_percent=result['throughput_change_percent'],selection_changed=False)
        pub=r.publish(['experiments/EXP-0030/RESULTS.json','experiments/EXP-0030/RECOVERY_FLOW_STATUS.json','experiments/EXP-0030/steps.jsonl'],
                      'experiments: summarize completed 16K pair after publication recovery')
        write(REC/'publication.json',pub)
    except BaseException:
        state('failed',error=traceback.format_exc());write(EXP/'RECOVERY_FLOW_STATUS.json',flow)
        r.step('B-only continuation stopped; all evidence retained',error=flow['error'])
        try:
            write(REC/'failure_publication.json',r.publish(['experiments/EXP-0030/RECOVERY_FLOW_STATUS.json','experiments/EXP-0030/steps.jsonl'],
                       'experiments: preserve B continuation failure'))
        except Exception:traceback.print_exc()
        traceback.print_exc()
    finally:signal.setitimer(signal.ITIMER_REAL,0);lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true');g.add_argument('--launch',action='store_true');g.add_argument('--run',action='store_true')
    args=p.parse_args()
    try:
        if args.prepare:prepare()
        elif args.launch:launch()
        else:execute()
    except BaseException:
        write(ART/'recovery_outer_failure.json',dict(time=now().isoformat(),error=traceback.format_exc()))
        raise
