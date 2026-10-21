"""Freeze the proven GQA arguments in the original fusion engine, before official E2E."""
from pathlib import Path
import ast
import datetime
import fcntl
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile

ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt';EXP=REPO/'experiments/EXP-0012';WT=ROOT/'worktrees/EXP-0012'
PARENT={'vllm-plugin-FL':'07d9fa0cfdc5f09db178e29508e3c0bcb5c1e908','FlagGems':'484413ea0db18221dd9e63e742361d837df9ff13'}
GEMS='19aa059af4ce456ef2e34e3bb83eb949d7e7b702'
def read(p):return json.loads(p.read_text())
def git(p,*a):return subprocess.check_output(['git','-C',str(p),*a],text=True,timeout=120).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
screen=Path(read(ROOT/'artifacts/EXP-0011-request.json')['run_directory'])
ext=Path(read(ROOT/'artifacts/EXP-0011-expanded-request.json')['run_directory'])
repeat=Path(read(ROOT/'artifacts/EXP-0011-drift-repeat.json')['run_directory'])
es=read(ext/'status.json');assert es['stage']=='validation_complete' and es['safe_to_resume_gpu']
assert not es['regression_names'] and es['drift_names']==['B8_L17408']
ev=read(ext/'validation.json');assert ev['status']=='passed' and len(ev['checks'])==26 and len(ev['timings'])==34
rv=read(repeat/'validation.json');assert rv['status']=='passed' and len(rv['timings'])==3
assert all(abs(x['baseline_drift_percent'])<=3 and x['time_change_percent']<=-3 for x in rv['timings'])
assert not EXP.exists() and not WT.exists()
EXP.mkdir(parents=True);WT.mkdir(parents=True)
(EXP/'README.md').write_text('''# EXP-0012：单级流水GQA接入与官方端到端测评

假设先于接入修改记录：EXP-0011的4warps/1stage/16segments在34个工作点无>3%退化，26项数值/Graph检查通过。
唯一基线漂移点B8/L17408三次预声明复测均通过，耗时减少18.27%～18.87%、基线漂移绝对值<=1.86%。
将这组常量显式接入IluvatarGQAImpl.forward，保持原最优EXP-0006融合、全部supports/fallback和官方调度不变。
FlagGems使用已验证的19aa059a，默认参数仍保持原值，只有受支持的Iluvatar路径明确请求stage1。
不会重新启用用户已否定的KV reservation调度器。原最优服务选择在E2E判定前保持不变。

## 门槛

- 接入前验证源码只新增三个明确launch参数及日志，supports和fallback逐字不变；校验代码和所有数值证据SHA。
- 实际服务smoke成功；42层GQA和42层RoPE+KV融合日志通过；35张FULL_DECODE_ONLY图完整捕获；图中旧分离KV写入为0。
- 隔离Triton缓存中必须找到实际服务编译的4warps/1stage CoreX attention元数据；调度候选类不得出现。
- 官方脚本、随机数据、采样与服务参数不变。4K先行，16K接续，各四轮弃首轮；超过3%吞吐退化则停止下一case并审查。
- 对照原始完整基线及EXP-0006完整case，报告吞吐、TTFT、TPOT/ITL、波动、排队与抢占；历史会话不是交错A/B。
- 完整任务精度按用户要求暂缓；数值、Graph、大地址和实际接入验证保留。

截止2026-10-21 23:35；控制器在启动和负载阶段均受绝对截止限制，归档器保留结果并推送。
''')
(EXP/'parent_commits.json').write_text(json.dumps(PARENT,indent=2)+'\n')
(EXP/'steps.jsonl').write_text(json.dumps({'time':datetime.datetime.now().astimezone().isoformat(),'action':'Hypothesis and full diagnostic gates recorded before engine edit'})+'\n')
print(git(ROOT/'worktrees/EXP-0006/vllm-plugin-FL','worktree','add','-b','leanscale/iluvatar-gqa-stage1',str(WT/'vllm-plugin-FL'),PARENT['vllm-plugin-FL']),flush=True)
print(git(ROOT/'worktrees/EXP-0011/FlagGems','worktree','add','--detach',str(WT/'FlagGems'),GEMS),flush=True)
rel='vllm_fl/dispatch/backends/flaggems/impl/iluvatar_gqa.py';p=WT/'vllm-plugin-FL'/rel;old=p.read_text()
old_call='                  meta.softmax_segm_max, meta.softmax_segm_expsum)'
new_call='                  meta.softmax_segm_max, meta.softmax_segm_expsum,\n                  segments=16, decode_warps=4, decode_stages=1)'
old_log='segmented decode and mixed batch; layer=%s'
new_log='segmented decode and mixed batch; decode_warps=4 decode_stages=1 segments=16; layer=%s'
assert old.count(old_call)==1 and old.count(old_log)==1
new=old.replace(old_call,new_call).replace(old_log,new_log);p.write_text(new)
assert new.replace(new_call,old_call).replace(new_log,old_log)==old
tree=ast.parse(new)
call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='paged_gqa')
assert {k.arg:ast.literal_eval(k.value) for k in call.keywords}=={'segments':16,'decode_warps':4,'decode_stages':1}
assert subprocess.run(['git','-C',str(WT/'vllm-plugin-FL'),'grep','-n','KVReservationAsyncScheduler'],capture_output=True).returncode==1
subprocess.run(['python3','-m','py_compile',str(p)],check=True)
git(WT/'vllm-plugin-FL','add',rel);git(WT/'vllm-plugin-FL','diff','--cached','--check')
print(git(WT/'vllm-plugin-FL','commit','-m','perf(iluvatar): select validated single-stage GQA decode'),flush=True)
commits={'vllm-plugin-FL':git(WT/'vllm-plugin-FL','rev-parse','HEAD'),'FlagGems':GEMS}
(EXP/'candidate_commits.json').write_text(json.dumps(commits,indent=2)+'\n')
for name in ('run_split_e2e.py','launch_split.py','finish_split.py','analyze_schedule.py','queue_official.py','build_scripts.py','prepare_integration.py'):
    shutil.copy2(ROOT/'staging/exp12'/name,EXP/name)
gate={'status':'passed','candidate_commits':commits,'launch':{'warps':4,'stages':1,'segments':16},
      'diagnostic_hashes':{str(p):sha(p) for p in (screen/'w4_t1_s16.json',ext/'validation.json',repeat/'validation.json')},
      'code_hashes':{'vllm-plugin-FL':{rel:sha(WT/'vllm-plugin-FL'/rel)},
                    'FlagGems':{'src/flag_gems/runtime/backend/_iluvatar/fused/paged_gqa.py':sha(WT/'FlagGems/src/flag_gems/runtime/backend/_iluvatar/fused/paged_gqa.py')}},
      'validation_scope':'26 numerical/Graph cases,34 shape timings,3 drift repeats; complete task accuracy deferred',
      'integration_cpu_check':'Only explicit validated launch kwargs and runtime log changed; reversing substitutions gives exact original source; scheduler class absent'}
(EXP/'candidate_validation.json').write_text(json.dumps(gate,indent=2)+'\n')
# CPU negative-route tests use an archived real engine log/graph plus explicit fixtures.
sys.path.insert(0,str(EXP))
spec=importlib.util.spec_from_file_location('exp12_gate',EXP/'run_split_e2e.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
reference=ROOT/'artifacts/EXP-0006/20261020_165011_official_4k'
text=(reference/'candidate_benchmark/server.log').read_text()
assert text.count('LeanScale GQA active:')==42
fixture=text.replace('segmented decode and mixed batch;',
                    'segmented decode and mixed batch; decode_warps=4 decode_stages=1 segments=16;')
checks=[]
with tempfile.TemporaryDirectory(prefix='exp12_route_') as tmp:
    tmp=Path(tmp);mod.RUN=tmp;folder=tmp/'candidate_benchmark';folder.mkdir()
    graph=next(q for q in (reference/'candidate_cache').rglob('computation_graph.py') if 'torch.ops.vllm.fl_rope_kv_dispatch' in q.read_text())
    dst=tmp/'candidate_cache/fixture/computation_graph.py';dst.parent.mkdir(parents=True);shutil.copy2(graph,dst)
    meta=next(q for q in (ext/'triton_cache').glob('*/_attention.json') if read(q).get('num_stages')==1 and read(q).get('num_warps')==4)
    dst=tmp/'triton_cache/fixture/_attention.json';dst.parent.mkdir(parents=True);shutil.copy2(meta,dst)
    for name,log,passed in [('positive_fixture',fixture,True),
        ('reject_missing_stage1_log',text,False),('reject_scheduler',fixture+'\nKVReservationAsyncScheduler\n',False),
        ('reject_missing_layer',fixture.replace('LeanScale GQA active:','missing:',1),False)]:
        (folder/'server.log').write_text(log)
        try:mod.verify_route(folder);ok=True
        except AssertionError:ok=False
        assert ok==passed,name;checks.append({'name':name,'passed':True})
    (folder/'server.log').write_text(fixture);dst.unlink()
    try:mod.verify_route(folder)
    except AssertionError:checks.append({'name':'reject_missing_service_compile_evidence','passed':True})
    else:raise AssertionError('Missing compile evidence accepted')
(EXP/'route_cpu_checks.json').write_text(json.dumps({'status':'passed','checks':checks,'scope':'Synthetic log mutations for harness validation; not actual new engine execution'},indent=2)+'\n')
subprocess.run(['python3','-m','py_compile',*[str(EXP/n) for n in ('run_split_e2e.py','launch_split.py','finish_split.py','queue_official.py')]],check=True)
for name,branch in [('vllm-plugin-FL','leanscale/iluvatar-gqa-stage1'),('FlagGems','leanscale/iluvatar-gqa-tuning')]:
    repo=WT/name
    if name=='vllm-plugin-FL':print(git(repo,'push','leanscale','HEAD:refs/heads/'+branch),flush=True)
    assert git(repo,'ls-remote',git(repo,'remote','get-url','--push','leanscale'),'refs/heads/'+branch).split()[0]==commits[name]
# Preserve the drift rerun and its setup failure without changing earlier artifacts.
e11=REPO/'experiments/EXP-0011';target=e11/'artifacts'/repeat.name;target.mkdir(parents=True)
shutil.copy2(repeat/'validation.json',target/'validation.json')
with (REPO/'docs/gqa_stage1_screen_20261021.md').open('a') as f:
    f.write('\n\n扩大验证：26项数值/Graph全部通过，34点无>3%退化。B8/L17408原基线漂移3.54%，'
            '预声明三次复测全部保留并通过：时间减少18.87%/18.40%/18.27%，漂移-0.19%/+0.97%/+1.86%。'
            '复测首次因后端提前创建缓存目录使脚本初始化失败，未产生测量，已记录并修复。'
            'EXP-0012现显式接入stage1候选，官方E2E尚待完成。\n')
with (REPO/'docs/optimization_journal.md').open('a') as f:
    f.write('\n\n### 2026-10-21：GQA stage1通过扩大验证，进入EXP-0012整模测试\n\n'
            '12组筛选、26项扩大数值/Graph、34计时点及单点三次漂移复测完成。'
            '仅在支持的Iluvatar入口显式传入4warps/1stage/16segments，原融合和fallback保留，'
            'Gems使用同一已验证提交，未启用劣化调度。5项正反日志门槛fixture通过，真实42层接入与E2E待验证。\n')
(EXP/'PR_DRAFT.md').write_text('''# Draft：赛事PR文案

标题：【FlagOS开放计算全球挑战赛S2.赛题2】【LeanScale】接入天数GQA单级流水优化

支持的Q16/KV2/D128 Iluvatar GQA入口显式使用4warps、1stage、16segments，保留现有fallback、RoPE+KV融合和原调度。
此前decode固定2 stages，候选改为1 stage；FlagGems参数扫描及扩大数值/Graph验证见EXP-0011。
单算子代表点耗时减少13.8%～20.8%，不是整模收益。真实42层接入、官方4K/16K四轮结果待补。
完整任务精度按用户要求暂缓，维持Draft，不宣称满足最终赛事精度要求，未创建上游PR。
''')
with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    paths=['experiments/EXP-0012','experiments/EXP-0011/repeat_drift.py','experiments/EXP-0011/revisions',
           'experiments/EXP-0011/steps.jsonl','experiments/EXP-0011/artifacts/'+repeat.name,
           'docs/gqa_stage1_screen_20261021.md','docs/optimization_journal.md']
    git(REPO,'add','--',*paths);git(REPO,'diff','--cached','--check')
    print(git(REPO,'commit','--only','-m','experiments: validate and integrate GQA stage-one candidate','--',*paths),flush=True)
    git(REPO,'push','origin','main');head=git(REPO,'rev-parse','HEAD')
    assert git(REPO,'ls-remote',git(REPO,'remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
print(json.dumps({'candidate_commits':commits,'docs_commit':head,'route_cpu_checks':checks}),flush=True)
