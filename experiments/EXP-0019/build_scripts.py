"""Derive a memory-layout screen from verified EXP15 checks, not its grid."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
PARENT=BASE/'flagos-exp15-dev' if (BASE/'flagos-exp15-dev').exists() else BASE/'EXP-0015'
def replace(s,a,b):
    assert s.count(a)==1,(a[:80],s.count(a))
    return s.replace(a,b)

p=PARENT/'trial.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='1d8f01bce546e1a6ebdefd9ad41aa353842336b15bfc0b1c47eb0c9f9e238bb0'
s=p.read_text().replace('/worktrees/EXP-0014/','/worktrees/EXP-0016/').replace('exp15_frozen_gqa','exp19_frozen_gqa')
s=replace(s,"POINTS = [",'''BASE_CONFIG = dict(DECODE, prefill_rows=64, prefill_warps=4, prefill_stages=1, prefill_tile=32)
OBSERVED = json.loads(Path(__file__).with_name('observed_workloads.json').read_text())['cases']
POINTS = [(c['name'], c['query_lengths'], c['context_lengths_cpu_upper_bound']) for c in OBSERVED] + [''')
s=replace(s,"    ('prefill4k', [2048], [4096]),\n",'')
s=replace(s,"*args,cfg['prefill_tile'],cfg['prefill_rows'],False,0,","*args,cfg['prefill_tile'],cfg['prefill_rows'],False,cfg['prefill_layout'],")
s=replace(s,"    p.add_argument('--rows',type=int,choices=(32,64,128),required=True)\n    p.add_argument('--warps',type=int,choices=(2,4),required=True)\n    p.add_argument('--stages',type=int,choices=(1,2),required=True)","    p.add_argument('--layout',type=int,choices=(0,1,2,3),required=True)")
s=replace(s,'cfg=dict(DECODE,prefill_rows=a.rows,prefill_warps=a.warps,prefill_stages=a.stages,prefill_tile=32)','cfg=dict(BASE_CONFIG,prefill_layout=a.layout)')
assert s.count('baseline.paged_gqa(**x,**DECODE)')==2
s=s.replace('baseline.paged_gqa(**x,**DECODE)','baseline.paged_gqa(**x,**BASE_CONFIG)')
s=replace(s,"save();torch.manual_seed(20261022)","save();torch.manual_seed(20261023)")
s=replace(s,"                torch.testing.assert_close(x['out'],expected,atol=3e-3,rtol=3e-2)","                torch.testing.assert_close(x['out'],expected,atol=0 if a.layout==0 else 3e-3,rtol=0 if a.layout==0 else 3e-2)")
s=replace(s,"'checks':[],'timings':[],'started_at':time.strftime('%FT%T%z')}","'checks':[],'timings':[],'started_at':time.strftime('%FT%T%z'),\n       'workload_note':'Six EXP18 CPU-shape-derived points with random tensors, plus eight declared synthetic regressions'}")
(HERE/'trial.py').write_text(s)

p=PARENT/'run_screen.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='5d067b4631c0716d0b20970fd7287d82ac039d2b26347a2f5737a2d3fc44b391'
s=p.read_text().replace('EXP-0015','EXP-0019').replace('prefill parameter screening','prefill memory-layout screening')
s=replace(s,"    run=a.run_dir.resolve();assert run.parent==ROOT/'artifacts/EXP-0019';run.mkdir(parents=True,exist_ok=False)","    assert not REQUEST.exists(), 'Inspect existing run; do not silently restart'\n    run=a.run_dir.resolve();assert run.parent==ROOT/'artifacts/EXP-0019';run.mkdir(parents=True,exist_ok=False)")
s=replace(s,"(ROOT/'artifacts/run_window.json')","(ROOT/'artifacts/EXP-0019/run_window.json')")
s=replace(s,"    deadline=datetime.datetime.fromisoformat(policy['compute_stop_at'])",'''    deadline=datetime.datetime.fromisoformat(policy['compute_stop_at'])
    assert policy['experiment']=='EXP-0019' and deadline.date()==datetime.datetime.now().astimezone().date()
    write(run/'run_window.json',policy)''')
s=replace(s,"        assert remaining()>600,'Insufficient time to start a fresh sweep'",'''        assert remaining()>35*60,'Insufficient independent budget for four configurations'
        import socket
        observations=[]
        for _ in range(2):
            output=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True,timeout=20)
            match=re.search(r'(\\d+)MiB\\s*/\\s*(\\d+)MiB\\s*\\|\\s*(\\d+)%',output)
            assert match and int(match[1])<=1024 and int(match[3])<=5,'GPU is not idle'
            observations.append(dict(memory_mib=int(match[1]),util_percent=int(match[3])))
            time.sleep(3)
        for port in (9031,9032):
            with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',port))!=0,'Service port occupied'
        write(run/'idle_evidence.json',observations)
        manifest=json.loads((EXP/'script_manifest.json').read_text())
        assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in EXP.glob('*.py')}==manifest
        assert hashlib.sha256((EXP/'observed_workloads.json').read_bytes()).hexdigest()==json.loads((EXP/'source_provenance.json').read_text())['observed_workloads_sha256']''')
start=s.index('        configs=');end=s.index("            write(run/(name+'.command.json'),command)",start)
s=s[:start]+'''        configs=[0,1,2,3]
        for layout in configs:
            if remaining()<120:
                update('deferred_for_shutdown');break
            name=f'layout{layout}'
            update('screening',current_configuration=name,completed_configurations=len(state['trials']))
            command=['python3',str(EXP/'trial.py'),'--layout',str(layout),'--output',str(run/(name+'.json'))]
'''+s[end:]
s=replace(s,'child.wait(timeout=min(600,remaining()))','child.wait(timeout=min(480,remaining()))')
s=replace(s,"if name=='rows64_w4_s2'", "if name=='layout0'")
s=replace(s,'assert len(changes)==9','assert len(changes)==14')
s=replace(s,"primary=stable and max(changes[:6])<=3 and min(changes[:6])<=-5","primary=stable and max(changes[:11])<=3 and min(changes[:11])<=-5")
s=replace(s,"primary and max(changes[6:])<=5","primary and max(changes[11:])<=5")
s=replace(s,"        rows=[]",'''        assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in EXP.glob('*.py')}==manifest
        for name,sha in state['commits'].items():
            assert subprocess.check_output(['git','-C',str(WT/name),'rev-parse','HEAD'],text=True).strip()==sha
            assert not subprocess.check_output(['git','-C',str(WT/name),'status','--porcelain'],text=True)
        rows=[]''')
start=s.index("        write(run/'screen_summary.json'")
end=s.index("    except BaseException as e:",start)
s=s[:start]+'''        observed=json.loads((EXP/'observed_workloads.json').read_text())['cases']
        order=[c['name'] for c in observed]+['prefill16k','mixed4k_decode_first','mixed4k_prefill_first','mixed16k_decode_first','mixed16k_prefill_first','decode64_4k','decode30_16k','decode1_4k']
        write(run/'screen_summary.json',dict(kind=state['kind'],order=order,rows=rows,selected_for_serving=False,
              next_gate='Expanded all-query/int64/Graph checks, then actual engine integration and official4K/16K'))
        lines=['# EXP-0019 GQA prefill 访存筛选','','单算子诊断；服务仍选EXP16。前6点由EXP18 CPU形状导出，随机Q/K/V；其余8点为声明的合成回归。',
               '固定64rows/4warps/stage1/tile32，仅改变prefill读取布局。完整GQA夹测为筛选依据，prefill-only与资源另列。','',
               '| 配置 | 11个prefill/mixed点耗时变化范围 | 3个decode回归点变化范围 | 基线稳定 | 候选门槛 |',
               '|---|---|---|---|---|']
        for row in rows:
            c=row['time_change_percent'];span=lambda x:f'{min(x):+.3f}% .. {max(x):+.3f}%'
            lines.append(f"| {row['configuration']} | {span(c[:11])} | {span(c[11:])} | {row['stable_baseline']} | {row['global_candidate_gate_passed']} |")
        lines+=['','负数为耗时减少。保留全部失败、原始采样和夹测漂移；不自动更改服务版本，不当作官方整模收益。']
        (run/'RESULTS.md').write_text('\\n'.join(lines)+'\\n')
'''+s[end:]
s=replace(s,"    child=None;locked=False", "    child=None;locked=False\n    singleton=(ROOT/'artifacts/EXP-0019-controller.lock').open('a');fcntl.flock(singleton,fcntl.LOCK_EX|fcntl.LOCK_NB)\n    update('preparing')")
s=replace(s,"        lock.close()", "        lock.close()\n        singleton.close()")
(HERE/'run_screen.py').write_text(s)
(HERE/'source_reuse.json').write_text(json.dumps({'EXP-0015/trial.py':'1d8f01bce546e1a6ebdefd9ad41aa353842336b15bfc0b1c47eb0c9f9e238bb0','EXP-0015/run_screen.py':'5d067b4631c0716d0b20970fd7287d82ac039d2b26347a2f5737a2d3fc44b391'},indent=2)+'\n')
