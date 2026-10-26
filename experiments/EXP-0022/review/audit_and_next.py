from pathlib import Path
from datetime import datetime
import fcntl,hashlib,json,math,statistics,subprocess

ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt';EXP=REPO/'experiments/EXP-0022';ART=ROOT/'artifacts/EXP-0022'
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a],text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10),(a,b)

state=read(ROOT/'artifacts/EXP-0022-request.json');run=Path(state['run_directory']);data=read(run/'result.json')
assert state['stage']=='screen_complete' and state['owned_children_reaped']
assert data['status']=='passed' and read(run/'publication.json')['remote_verified']
assert datetime.fromisoformat(data['finished_at'])<datetime.fromisoformat(state['compute_stop_at'])
plan=read(EXP/'plan.json')
for n,h in plan['script_hashes'].items():assert sha(EXP/n)==h,n
for p,h in plan['source_hashes'].items():assert sha(Path(p))==h,p
for n,c in plan['selected_commits'].items():
 p=ROOT/'worktrees/EXP-0020'/n
 assert subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()==c
 assert not subprocess.check_output(['git','-C',str(p),'status','--porcelain'],text=True).strip()
assert read(ROOT/'artifacts/active_inference_plan.json')['selected_serving_commits']==plan['selected_commits']
projections={'qkv': [2560,2048],'o':[2048,2048],'gate_up':[12288,2048],'down':[2048,6144]}
assert data['shapes']==projections
ms=[1,4,8,16,30,32,64,2048];targets=[16,32,64]
assert data['Ms']==ms and data['targets']==targets
expected={(p,m,t,mode) for p in projections for m in ms for t in targets if t>m for mode in ['hot','weights8']}
seen=set()
def timing(x):
 assert x['calls_per_replay']==8 and len(x['samples_us'])==7 and all(math.isfinite(y) and y>0 for y in x['samples_us'])
 close(x['median_us'],statistics.median(x['samples_us']))
 return x['median_us']
for x in data['timings']:
 key=(x['projection'],x['M'],x['target'],x['mode']);assert key not in seen;seen.add(key)
 b,n,a=[timing(x[k]) for k in ['baseline_before','candidate','baseline_after']]
 close(x['time_change_percent'],100*(n/((a+b)/2)-1));close(x['baseline_drift_percent'],100*(a/b-1))
 assert x['complete_padding_linear_slice'] and x['production_tail_M']==(x['M'] in [8,16])
assert seen==expected and len(seen)==112
seen=set()
for x in data['curves']:
 key=(x['projection'],x['M'],x['mode']);assert key not in seen;seen.add(key)
 b,a=[timing(x[k]) for k in ['baseline_before','baseline_after']]
 close(x['baseline_drift_percent'],100*(a/b-1))
assert seen=={(p,m,mode) for p in projections for m in ms for mode in ['hot','weights8']}
expected={(p,m,t,'torch.bfloat16') for p in projections for m in ms for t in targets if t>m}|{(p,m,t,'torch.float16') for p in projections for m in [8,16] for t in [32,64]}
seen=set()
for x in data['checks']:
 key=(x['projection'],x['M'],x['target'],x['dtype']);assert key not in seen;seen.add(key)
 assert x['graph_exact_replay'] and x['graph_mutations']==['input','weight']
 for k in ['candidate_fp32_max_abs','candidate_baseline_max_abs','mutated_fp32_max_abs']:assert math.isfinite(x[k]) and x[k]>=0
assert seen==expected and len(seen)==72
candidates=[]
for p in projections:
 for m in [8,16]:
  for t in targets:
   pair=[x for x in data['timings'] if (x['projection'],x['M'],x['target'])==(p,m,t)]
   if len(pair)==2 and all(x['time_change_percent']<=-5 and abs(x['baseline_drift_percent'])<=3 for x in pair):candidates.append((p,m,t))
assert candidates==read(run/'screen_summary.json')['candidates']==[]
arch=EXP/'artifacts'/run.name
for n in ['result.json','screen_summary.json','status.json']:assert sha(run/n)==sha(arch/n),n
stats={p:dict(min_time_change_percent=min(x['time_change_percent'] for x in data['timings'] if x['projection']==p and x['M'] in [8,16]),max_time_change_percent=max(x['time_change_percent'] for x in data['timings'] if x['projection']==p and x['M'] in [8,16]),max_abs_drift_percent=max(abs(x['baseline_drift_percent']) for x in data['timings'] if x['projection']==p and x['M'] in [8,16])) for p in projections}
auditdir=EXP/'review';assert not auditdir.exists();auditdir.mkdir()
write(auditdir/'audit.json',dict(status='complete',reviewed_at=datetime.now().astimezone().isoformat(),result_sha256=sha(run/'result.json'),counts=dict(checks=72,curves=64,timings=112),candidates=[],primary_M8_M16=stats,decision='reject padding path; retain selected EXP20'))
(auditdir/'audit_and_next.py').write_text(Path(__file__).read_text())
write(EXP/'STATUS.json',dict(stage='audited_complete',outcome='rejected_padding',selected_for_serving=False,artifact_publication=read(run/'publication.json'),checks=72,curve_points=64,timing_points=112))
with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(),action='Independently audit all samples/formulas/coverage/source/archive hashes; reject padding; retain EXP20'))+'\n')
doc=['# 4K小batch投影：补零实验与权重布局假设','','EXP22为算子诊断，不是官方端到端测评。当前工作基线仍为EXP20。','','## 补零结果','','72项数值/Graph检查通过，64个基线曲线点、112个旧/新/旧夹测点已全部保存并独立重算。生产尾部主目标M8/16的完整pad+linear+slice均更慢，无候选达到预设门槛。','','| 投影 | M8/16耗时变化范围 | 最大绝对夹测漂移 |','|---|---:|---:|']
for p,s in stats.items():doc.append(f"| {p} | +{s['min_time_change_percent']:.2f}%～+{s['max_time_change_percent']:.2f}% | {s['max_abs_drift_percent']:.2f}% |")
doc += ['', '两种模式分别重用同一权重和轮换8个权重，只用于缓存敏感性诊断。不能把profile不同窗口的down时间差当成补零收益；本轮结果直接否定独立补零这一具体方案。保留全部退化点，不修改阈值挑点。', '', '## 下一项独立假设', '', 'EXP23将保持原生linear、输入M与权重数值，尝试加载时一次性把[N,K]权重从stride(K,1)重排为stride(1,N)。运行路径只读取预排权重，重排成本和临时内存单独记录；不在每次推理中转置拷贝。先做随机张量微基准，不直接接入引擎。', '', '必须同时检查小M收益、M64及prefill M128/512/2048回归、数值与Graph重放、隐式copy事件。若仅小M受益而prefill退化，不能全局切换；本轮不采用双份权重常驻以免侵占KV空间。']
(REPO/'docs/small_batch_projection_20261026.md').write_text('\n'.join(doc)+'\n')
new=REPO/'experiments/EXP-0023';assert not new.exists();new.mkdir()
(new/'README.md').write_text('''# EXP-0023：原生线性层权重存储布局诊断

实现前假设：EXP22独立补零全部不满足收益门槛。保留原生dispatch_unquantized_gemm→default_unquantized_gemm→F.linear，加载时一次性W.T.contiguous().T将权重[N,K]从stride(K,1)变为stride(1,N)，检查原生库不同输入存储布局的效率。
基线为EXP20 plugin ce19f93eee4c73f5a26911554700457e879f891d、Gems de958ddfc4deeb43d6c94bb8652a3d4d3bc905fe；不改服务代码和调度。
本轮只用随机张量；原布局由create_weights源码推导，尚无本轮真实服务weight stride探针。QKV/O/gate_up/down形状与EXP22相同。
固定M=[1,4,8,16,30,32,64,128,512,2048]，hot与weights8两模式，7样本×8Graph调用、旧/新/旧夹测，80个计时点。M30是NONE诊断，生产尾部主目标M8/16。
40个BF16数值/Graph检查以及M1/8/16/2048共16个FP16检查，共56；固定atol0.01/rtol0.02，FP32参照和输入/权重更新Graph重放。另记录一次性packing耗时/内存和M8/2048 CPU算子事件，不能把CPU事件当作底层库无内部拷贝的证明。
预注册投影级门槛：该投影M8/16在两模式均至少降低5%耗时，所有其余M/模式均不回归超过3%，所有夹测绝对漂移<=3%，无观察到的aten::clone/contiguous/copy_等隐式拷贝事件。完整覆盖后才判定候选；不改阈值、不删慢点。
候选仍须独立复测、实际权重stride及集成路径核查、完整子图、引擎接入和官方E2E；不能从微基准直接宣称整模收益。
每投影使用8独立权重仅是缓存敏感性代理。持久双份权重不在本轮部署方案内，以免挤占KV；prefill退化不得简单全局切换。
全模型精度按用户暂缓，算子检查必须通过；失败保留。使用新的有限时间窗口和GPU锁，最晚23:35停止计算/23:50保存。
''')
with (new/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(),action='Record weight layout hypothesis before implementing experiment code'))+'\n')
with (REPO/'docs/optimization_journal.md').open('a') as f:f.write('\n\n### 2026-10-26：EXP22否定补零，EXP23布局假设\n\nEXP22已独立审核72检查/64曲线/112计时，M8/16完整路径全部退化，拒绝接入，保留EXP20。具体范围见docs/small_batch_projection_20261026.md。下一EXP23在实现前记录原生GEMM权重存储布局假设，完整覆盖decode与prefill，不采用常驻双份权重。\n')
with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);assert not git('diff','--cached','--name-only')
 paths=['experiments/EXP-0022','experiments/EXP-0023','docs/small_batch_projection_20261026.md','docs/optimization_journal.md']
 git('add','--',*paths);git('diff','--cached','--check');git('commit','--only','-m','experiments: reject padding and register native weight layout hypothesis','--',*paths)
 head=git('rev-parse','HEAD');git('push','origin','main');assert git('ls-remote',git('remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
write(ART/'review_publication.json',dict(commit=head,remote_verified=True))
print(json.dumps(dict(commit=head,remote_verified=True,primary_stats=stats,next='EXP23 hypothesis registered; no GPU launch yet')))
