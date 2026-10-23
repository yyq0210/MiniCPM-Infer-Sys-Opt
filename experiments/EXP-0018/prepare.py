"""CPU-only preparation. Never launch a model or device profiler here."""
from datetime import datetime
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback

ROOT=Path('/workspace/LeanScale')
REPO=ROOT/'MiniCPM-Infer-Sys-Opt'
EXP=REPO/'experiments/EXP-0018'
ART=ROOT/'artifacts/EXP-0018'
WORK=ROOT/'worktrees/EXP-0018'
STAGING=Path(__file__).resolve().parent
COMMITS={'vllm-plugin-FL':'588a6147cd291e272324ab6c812024cd587c6287','FlagGems':'fd209df717648002ebf5ba4ec351424dcc997a53'}

def write(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()
def step(text,**kw):
    with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(),step=text,**kw),ensure_ascii=False)+'\n')

def route_checks():
    spec=importlib.util.spec_from_file_location('exp18_route',EXP/'route_checks.py')
    route=importlib.util.module_from_spec(spec);spec.loader.exec_module(route)
    ref=ROOT/'artifacts/EXP-0016/20261022_145853_official_4k'
    checks=[]
    with tempfile.TemporaryDirectory(prefix='exp18_route_') as temp:
        d=Path(temp)
        # Read-only links to complete actual service cache; never alter it.
        (d/'cache').symlink_to(ref/'candidate_cache',target_is_directory=True)
        (d/'triton_cache').symlink_to(ref/'triton_cache',target_is_directory=True)
        log=(ref/'candidate_benchmark/server.log').read_text()
        (d/'server.log').write_text(log)
        route.verify_route(d,'FULL_DECODE_ONLY')
        actual=read(d/'route_evidence.json')
        write(EXP/'actual_service_route_cpu_check.json',dict(status='passed',source_experiment='EXP-0016',
               source_run=ref.name,kernel_counts=actual['kernel_counts'],gqa_launch=actual['gqa_launch']))
        checks.append('all_actual_EXP16_4K_decode_prefill_IR_and_graphs')
        # Negative mutations affect only small temporary fixtures.
        (d/'triton_cache').unlink();(d/'triton_cache').mkdir()
        for kind,shape in [('decode','8x128x!tt.ptr<'),('prefill','64x128x!tt.ptr<')]:
            p=next(p for p in (ref/'triton_cache').glob('*/_attention.json') if shape in p.with_suffix('.ttir').read_text())
            target=d/'triton_cache'/kind/p.name;target.parent.mkdir()
            shutil.copy2(p,target);shutil.copy2(p.with_suffix('.ttir'),target.with_suffix('.ttir'))
        def test(name,mode,expected):
            error=None
            try:route.verify_route(d,mode);ok=True
            except AssertionError as exc:ok=False;error=str(exc)
            assert ok==expected,(name,mode,error)
            checks.append(name)
        test('positive_frozen_kernel_fixture','FULL_DECODE_ONLY',True)
        test('NONE_rejects_FULL_capture','NONE',False)
        # Progress output can share a line with the 42nd layer's logs.
        # Mutate only the capture marker, never remove complete log lines.
        none_log=log.replace('Capturing CUDA graphs (decode, FULL)','FIXTURE_without_FULL_capture')
        (d/'server.log').write_text(none_log)
        test('NONE_gate_without_capture_fixture_only','NONE',True)
        test('FULL_rejects_missing_capture','FULL_DECODE_ONLY',False)
        (d/'server.log').write_text(log)
        for kind in ['decode','prefill']:
            p=d/'triton_cache'/kind/'_attention.json';original=p.read_text()
            m=read(p);m['num_stages']=3;write(p,m);test(kind+'_reject_wrong_stages','FULL_DECODE_ONLY',False);p.write_text(original)
            m=read(p);m['shared']+=2048;write(p,m);test(kind+'_reject_wrong_shared','FULL_DECODE_ONLY',False);p.write_text(original)
            ir=p.with_suffix('.ttir');original_ir=ir.read_text()
            changed=original_ir.replace('8x128x!tt.ptr<','16x128x!tt.ptr<') if kind=='decode' else original_ir.replace('64x128x!tt.ptr<','32x128x!tt.ptr<')
            assert changed!=original_ir;ir.write_text(changed);test(kind+'_reject_wrong_rows','FULL_DECODE_ONLY',False);ir.write_text(original_ir)
        for name,bad in [('missing_GQA_layer',log.replace('LeanScale GQA active:','missing:',1)),
                         ('wrong_prefill_stage_log',log.replace('prefill_stages=1','prefill_stages=2',1)),
                         ('rejected_scheduler',log+'\nKVReservationAsyncScheduler\n')]:
            (d/'server.log').write_text(bad);test('reject_'+name,'FULL_DECODE_ONLY',False)
    return checks

def main():
    assert (EXP/'README.md').exists(),'Register hypothesis before preparing code'
    resume=sys.argv[1:]==['--resume-cpu']
    if resume:
        assert WORK.exists() and (ART/'preparation_error.json').exists()
        assert not (ART/'preparation.json').exists() and not (ROOT/'artifacts/EXP-0018-request.json').exists()
        fail=EXP/'failures';fail.mkdir(exist_ok=True)
        target=fail/f'preparation_{len(list(fail.glob("*.json")))+1}.json'
        shutil.copy2(ART/'preparation_error.json',target)
    else:
        assert not WORK.exists() and not ART.exists(),'Do not overwrite existing prepared experiment'
    active=read(ROOT/'artifacts/active_inference_plan.json')
    assert active['selected_experiment']=='EXP-0016' and active['selected_serving_commits']==COMMITS
    WORK.mkdir(exist_ok=resume);ART.mkdir(exist_ok=resume)
    try:
        step('Resume CPU-only preparation after retaining fixture failure' if resume else 'Begin CPU-only preparation; no GPU run window assumed',commits=COMMITS)
        if resume:step('Fix only synthetic NONE fixture: replace graph marker without deleting shared progress/layer log line; inference and route gate unchanged')
        for name,commit in COMMITS.items():
            if (WORK/name).exists():
                assert resume and git('-C',str(WORK/name),'rev-parse','HEAD')==commit
                assert not git('-C',str(WORK/name),'status','--porcelain')
            else:
                subprocess.run(['git','worktree','add','--detach',str(WORK/name),commit],cwd=ROOT/'worktrees/EXP-0016'/name,check=True)
        for p in STAGING.iterdir():
            if p.is_file() and (p.suffix=='.py' or p.name in ('source_reuse.json','analysis_main.py.fragment')):
                if p.resolve() != (EXP/p.name).resolve():
                    shutil.copy2(p,EXP/p.name)
        subprocess.run(['python3','-m','py_compile',*[str(p) for p in EXP.glob('*.py')]],check=True)
        validation=subprocess.run(['python3',str(EXP/'validate_cpu.py')],capture_output=True,text=True)
        (EXP/'cpu_checks.log').write_text(validation.stdout+validation.stderr)
        assert validation.returncode==0,'CPU validation failed; see retained log'
        checks=read(EXP/'cpu_checks.log')
        route=route_checks()
        checks['route_checks']=route
        write(EXP/'cpu_checks.json',checks)
        write(EXP/'candidate_commits.json',COMMITS)
        write(EXP/'script_manifest.json',{p.name:sha(p) for p in EXP.glob('*.py')})
        sys.path.insert(0,str(EXP));import run_profile
        provenance=run_profile.verify_sources();write(EXP/'preparation_provenance.json',provenance)
        step('Pass CPU attribution, hook, cutoff and actual-cache route checks; freeze independent worktrees',
             checks=len(checks['checks'])+len(route),gpu_work_started=False)
        write(EXP/'STATUS.json',dict(stage='prepared_awaiting_today_window',gpu_work_started=False,
              checks=len(checks['checks'])+len(route),updated_at=datetime.now().astimezone().isoformat()))
        with (REPO/'docs/optimization_journal.md').open('a') as f:
            f.write('\n\n### 2026-10-23：EXP18 当前版本 4K 热点诊断准备\n\nEXP17 服务器原始缓存复核与版本选择已发布 6362da9；选定 EXP16 stage1 作为后续性能基线。EXP18 复用已核验 EXP9 归因、CPU metadata 与 EXP17 编译门槛，增加主负载计数重置、每请求长度、实际阶段触发和尾部覆盖记录；FULL/NONE 分开。独立工作树和 CPU 检查完成，今日关机时间待回复，因此尚未开始 GPU 诊断。见 [EXP18](../experiments/EXP-0018/README.md)。\n')
        paths=['experiments/EXP-0018','docs/optimization_journal.md']
        lock=(ROOT/'artifacts/docs-publish.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            subprocess.run(['git','add','--',*paths],cwd=REPO,check=True)
            diff=git('diff','--cached','--name-only','--',*paths).splitlines()
            assert diff and not any('__pycache__' in p for p in diff),diff
            for rel in diff:
                p=REPO/rel
                assert p.stat().st_size<8*1024*1024
                assert not any(x in p.read_text() for x in ('PRIVATE'+' KEY','ssh.'+'flagos.io','lab'+'_inst_'))
            subprocess.run(['git','commit','-m','experiments: prepare selected-version 4K phase and tail profiling','--only','--',*paths],cwd=REPO,check=True)
            commit=git('rev-parse','HEAD')
            subprocess.run(['git','push','origin','main'],cwd=REPO,check=True,timeout=120)
            assert git('ls-remote','origin','refs/heads/main').split()[0]==commit
            publication=dict(status='pushed',commit=commit,remote_verified=True)
            write(ART/'preparation.json',dict(status='prepared_awaiting_today_window',publication=publication,checks=len(checks['checks'])+len(route)))
        finally:lock.close()
        active.update(current_experiment='EXP-0018',status='CPU preparation complete; waiting for today shutdown window before GPU profiling',next_direction='Run selected-version 4K phase/hotspot diagnostics, then choose optimization target')
        write(ROOT/'artifacts/active_inference_plan.json',active)
        print(json.dumps(dict(status='prepared',publication=publication,checks=len(checks['checks'])+len(route))))
    except BaseException:
        write(ART/'preparation_error.json',dict(status='failed',error=traceback.format_exc()))
        step('CPU preparation failed; retain partial files and error',error=traceback.format_exc())
        raise

if __name__=='__main__':main()
