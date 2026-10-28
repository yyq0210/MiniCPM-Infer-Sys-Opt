"""Preserve the unmeasured run; fix only partition-aware compiled-route auditing."""
from pathlib import Path
from datetime import datetime,timedelta
import ast,fcntl,hashlib,importlib.util,json,shutil,subprocess,time

ROOT=Path('/workspace/LeanScale');DOC=ROOT/'MiniCPM-Infer-Sys-Opt';EXP=DOC/'experiments/EXP-0024'
SCRIPTS=EXP/'e2e-r1';ART=ROOT/'artifacts/EXP-0024/e2e-r1'
FAILED=ROOT/'artifacts/EXP-0024/20261028_191515_official_4k'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git','-C',str(DOC),*a],text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def step(action,**kw):
    with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(),action=action,**kw))+'\n')
def main():
    assert not SCRIPTS.exists() and not ART.exists()
    state=read(FAILED/'status.json')
    assert state['stage']=='failed' and state['owned_children_reaped']
    assert 'No complete 42-layer generated MLP layout proof' in state['error']
    assert not (FAILED/'candidate_benchmark/benchmark.command.json').exists()
    assert read(FAILED/'publication.json')['remote_verified']
    prior=read(EXP/'e2e_plan.json')
    for name,h in prior['script_hashes'].items():assert sha(EXP/name)==h
    assert read(FAILED/'candidate_benchmark/route_evidence.json')['status']=='passed'
    SCRIPTS.mkdir();ART.mkdir(parents=True)
    (SCRIPTS/'README.md').write_text('''# EXP24 官方4K接线检查修复\n\n前次19:15:16–19:20:33服务已完成84权重重排、42 GQA/42 RoPEKV/35 FULL捕获与标准路由检查，KV容量522352 tokens与EXP20相同。尚未开始benchmark，0轮计时。失败原因是诊断脚本假设所有MLP位于一个生成模块；实际vLLM在attention边界分图，42个不同MLP编译模块各含gate/up与down一次（另有首段非MLP模块）。原服务代码、检查源码与失败完整保留，归档4fe573a2ec5ca754ffd55b459f6032d67c379e59。\n\n只修编译证据组织检查：要求42个不同源码哈希的模块各一对投影，所有matmul仍直接读取原参数且stride严格匹配；同时解析完整FX图，逐一核对层0..41的gate/up和down原生linear节点各一次。保持84真实加载日志、数值相等、旧GQA/RoPEKV/Graph/容量与所有benchmark条件。未放宽精度或性能门槛。先在失败run的只读编译缓存上离线验证新检查，再新run、新缓存、新窗口执行。推理插件41fd7a4与FlagGems de958完全不变。\n''')
    step('Register partition-aware route audit fix before editing copied controller',failed_run=FAILED.name,
         completed_benchmark_rounds=0,failed_archive=read(FAILED/'publication.json')['commit'])
    old=(EXP/'run_e2e.py').read_text()
    old_assert="    assert any(m['direct_weight_mm_counts']=={'gate_up_proj':42,'down_proj':42} for m in modules), 'No complete 42-layer generated MLP layout proof'"
    new_assert='''    # vLLM partitions at attention boundaries: one MLP per compiled block.
    unique_modules = {m['sha256']: m for m in modules}
    assert len(unique_modules) == 42, ('Expected 42 distinct MLP blocks', len(unique_modules))
    assert all(m['direct_weight_mm_counts'] == {'gate_up_proj': 1, 'down_proj': 1}
               for m in unique_modules.values())
    fx_coverage = []
    for graph_path in (run/'candidate_cache').rglob('computation_graph.py'):
        coverage = {}
        graph_tree = ast.parse(graph_path.read_text())
        for node in ast.walk(graph_tree):
            if not (isinstance(node, ast.Call) and ast.unparse(node.func) == 'torch._C._nn.linear'
                    and len(node.args) >= 2 and isinstance(node.args[1], ast.Name)):
                continue
            match = re.search(r'modules_layers_modules_(\\d+)_modules_mlp_modules_(gate_up_proj|down_proj)_parameters_weight_',
                              node.args[1].id, re.IGNORECASE)
            if match:
                key = (match[1], match[2])
                coverage[key] = coverage.get(key, 0) + 1
        if coverage:
            assert set(coverage) == expected and all(n == 1 for n in coverage.values()), coverage
            fx_coverage.append(dict(path=str(graph_path.relative_to(run)),sha256=sha(graph_path),
                                    logical_layers=42,gate_up_nodes=42,down_nodes=42))
    assert fx_coverage, 'No full FX graph with all 84 original MLP weight arguments'
'''.rstrip()
    assert old.count(old_assert)==1
    new=old.replace(old_assert,new_assert)
    replacements={
        "EXP=DOC/'experiments/EXP-0024'; ART=ROOT/'artifacts/EXP-0024'":"EXP=DOC/'experiments/EXP-0024'; SCRIPTS=EXP/'e2e-r1'; ART=ROOT/'artifacts/EXP-0024/e2e-r1'",
        "plan=read(EXP/'e2e_plan.json')":"plan=read(SCRIPTS/'e2e_plan.json')",
        "sha(EXP/name)==h,name":"sha(SCRIPTS/name)==h,name",
        "validation=ART/'validation_r3'":"validation=ART.parent/'validation_r3'",
        "one_time_conversions=84,compiled_modules=modules,":"one_time_conversions=84,compiled_modules=modules,fx_coverage=fx_coverage,compiled_mlp_blocks=len(unique_modules),",
    }
    for before,after in replacements.items():
        assert new.count(before)==1,before
        new=new.replace(before,after)
    ast.parse(new);(SCRIPTS/'run_e2e.py').write_text(new)
    shutil.copy2(Path(__file__),SCRIPTS/'prepare_route_retry.py')
    # Test only the corrected read-only graph audit, preserving the old run.
    diagnostic=ART/'diagnostic_previous_run';diagnostic.mkdir()
    shutil.copy2(FAILED/'candidate_benchmark/server.log',diagnostic/'server.log')
    spec=importlib.util.spec_from_file_location('route_retry',SCRIPTS/'run_e2e.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    reference=ROOT/'artifacts/EXP-0020/20261026_125756_official_4k/candidate_benchmark'
    evidence=module.layout_route(FAILED,diagnostic,reference)
    assert evidence['compiled_mlp_blocks']==42 and evidence['candidate_kv_tokens']==522352
    write(SCRIPTS/'previous_run_partition_audit.json',evidence)
    write(SCRIPTS/'change_record.json',dict(parent_runner_sha256=sha(EXP/'run_e2e.py'),
          runner_sha256=sha(SCRIPTS/'run_e2e.py'),change='partition-aware route evidence and isolated paths only',
          candidate_commits=prior['candidate_commits'],parent_failed_status_sha256=sha(FAILED/'status.json')))
    plan=dict(prior,script_hashes={n:sha(SCRIPTS/n) for n in ['run_e2e.py','prepare_route_retry.py']},
              retry_reason='Actual MLP code split across42 blocks; same 84 logical weights and strict strides',
              parent_script_hashes=prior['script_hashes'])
    write(SCRIPTS/'e2e_plan.json',plan)
    step('Corrected strict route audit passed on preserved cache without GPU work',compiled_blocks=42,
         logical_mlp_layers=42,kv_tokens=522352,candidate_commits=prior['candidate_commits'])
    with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);assert not git('diff','--cached','--name-only')
        paths=['experiments/EXP-0024/e2e-r1','experiments/EXP-0024/steps.jsonl']
        git('add','--',*paths);git('diff','--cached','--check')
        git('commit','--only','-m','experiments: validate partition-aware MLP route proof before retry','--',*paths)
        head=git('rev-parse','HEAD');git('push','origin','main');assert git('ls-remote',git('remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
    write(ART/'preparation.json',dict(commit=head,remote_verified=True))
    now=datetime.now().astimezone();assert now.strftime('%Y-%m-%d')=='2026-10-28'
    daily=now.replace(hour=23,minute=35,second=0,microsecond=0)
    end=min(now+timedelta(minutes=110),daily);assert (end-now).total_seconds()>70*60
    policy=dict(created_at=now.isoformat(),compute_stop_at=end.isoformat(),
                archive_target_at=min(end+timedelta(minutes=10),daily.replace(minute=50)).isoformat(),
                daily_compute_stop_at=daily.isoformat(),daily_save_by=daily.replace(minute=50).isoformat())
    write(ART/'e2e_window_4k.json',policy)
    run=ART/(now.strftime('%Y%m%d_%H%M%S')+'_official_4k_r1')
    with (ART/'official_4k.controller.log').open('w') as log:
        p=subprocess.Popen(['python3',str(SCRIPTS/'run_e2e.py'),'--case','4k','--run-dir',str(run)],
                           stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    for _ in range(100):
        if (ART/'official_4k_request.json').exists():break
        assert p.poll() is None;time.sleep(.2)
    else:raise RuntimeError('No retry receipt')
    receipt=read(ART/'official_4k_request.json');assert receipt['pid']==p.pid
    write(ART/'launch.json',dict(preparation_commit=head,receipt=receipt,policy=policy))
    print(json.dumps(dict(commit=head,receipt=receipt,policy=policy)),flush=True)
if __name__=='__main__':main()
