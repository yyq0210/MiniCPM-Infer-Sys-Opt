from pathlib import Path
from datetime import datetime
import fcntl,hashlib,json,math,statistics,subprocess
ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt';EXP=REPO/'experiments/EXP-0023';ART=ROOT/'artifacts/EXP-0023'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a],text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10),(a,b)
state=read(ROOT/'artifacts/EXP-0023-request.json');run=Path(state['run_directory']);data=read(run/'result.json');plan=read(EXP/'plan.json')
assert state['stage']=='screen_complete' and state['owned_children_reaped'] and data['status']=='passed'
assert read(run/'publication.json')['remote_verified']
assert datetime.fromisoformat(data['finished_at'])<datetime.fromisoformat(state['compute_stop_at'])
for n,h in plan['script_hashes'].items():assert sha(EXP/n)==h,n
for p,h in plan['source_hashes'].items():assert sha(Path(p))==h,p
active=read(ROOT/'artifacts/active_inference_plan.json');assert active['selected_serving_commits']==plan['selected_commits']
for n,c in plan['selected_commits'].items():
 p=ROOT/'worktrees/EXP-0020'/n
 assert subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()==c
 assert not subprocess.check_output(['git','-C',str(p),'status','--porcelain'],text=True).strip()
shapes={'qkv':[2560,2048],'o':[2048,2048],'gate_up':[12288,2048],'down':[2048,6144]};ms=[1,4,8,16,30,32,64,128,512,2048]
assert data['shapes']==shapes and data['Ms']==ms
def timed(x):
 assert len(x['samples_us'])==7 and x['calls_per_replay']==8 and all(math.isfinite(v) and v>0 for v in x['samples_us'])
 close(x['median_us'],statistics.median(x['samples_us']));return x['median_us']
seen=set()
for x in data['timings']:
 key=(x['projection'],x['M'],x['mode']);assert key not in seen;seen.add(key)
 b,n,a=[timed(x[k]) for k in ['baseline_before','candidate','baseline_after']]
 close(x['time_change_percent'],100*(n/((b+a)/2)-1));close(x['baseline_drift_percent'],100*(a/b-1))
 assert x['one_time_packing_excluded'] and x['production_tail_M']==(x['M'] in [8,16])
assert seen=={(p,m,mode) for p in shapes for m in ms for mode in ['hot','weights8']} and len(seen)==80
seen=set()
for x in data['checks']:
 key=(x['projection'],x['M'],x['dtype']);assert key not in seen;seen.add(key)
 n,k=shapes[x['projection']];assert x['original_stride']==[k,1] and x['candidate_stride']==[1,n]
 assert x['graph_exact_replay'] and x['graph_mutations']==['input','weight']
 for k in ['candidate_fp32_max_abs','candidate_baseline_max_abs','mutated_fp32_max_abs']:assert math.isfinite(x[k]) and x[k]>=0
assert seen=={(p,m,'torch.bfloat16') for p in shapes for m in ms}|{(p,m,'torch.float16') for p in shapes for m in [1,8,16,2048]}
assert len(data['checks'])==56 and len(data['packing'])==4 and {x['projection'] for x in data['packing']}==set(shapes)
for x in data['packing']:
 n,k=shapes[x['projection']];assert x['count']==8 and x['packed_tensor_bytes']==8*n*k*2
 assert x['original_stride']==[k,1] and x['candidate_stride']==[1,n]
 assert x['total_elapsed_us']>0 and x['peak_allocated_delta']>=x['memory_allocated_delta']>=x['packed_tensor_bytes']
seen=set()
for x in data['operator_events']:
 key=(x['projection'],x['M'],x['layout']);assert key not in seen;seen.add(key)
 assert x['observed_copy_events']==[v for v in x['events'] if v['name'] in ['aten::clone','aten::contiguous','aten::copy_','aten::_to_copy']]
 assert all(v['count']>0 for v in x['events'])
assert seen=={(p,m,l) for p in shapes for m in [8,2048] for l in ['row_major','column_major']}
candidates=[];stats={}
for p in shapes:
 rows=[x for x in data['timings'] if x['projection']==p];primary=[x for x in rows if x['M'] in [8,16]];other=[x for x in rows if x['M'] not in [8,16]]
 ev=[x for x in data['operator_events'] if x['projection']==p and x['layout']=='column_major']
 passed=all(x['time_change_percent']<=-5 for x in primary) and all(x['time_change_percent']<=3 for x in other) and all(abs(x['baseline_drift_percent'])<=3 for x in rows) and not any(x['observed_copy_events'] for x in ev)
 if passed:candidates.append(p)
 stats[p]=dict(primary_min_time_change_percent=min(x['time_change_percent'] for x in primary),primary_max_time_change_percent=max(x['time_change_percent'] for x in primary),worst_other_time_change_percent=max(x['time_change_percent'] for x in other),max_abs_drift_percent=max(abs(x['baseline_drift_percent']) for x in rows),candidate=passed)
assert candidates==read(run/'screen_summary.json')['candidates']==['gate_up','down']
arch=EXP/'artifacts'/run.name
for n in ['result.json','screen_summary.json','status.json']:assert sha(run/n)==sha(arch/n),n
dest=EXP/'review';assert not dest.exists();dest.mkdir();(dest/'audit.py').write_text(Path(__file__).read_text())
write(dest/'audit.json',dict(status='complete',reviewed_at=datetime.now().astimezone().isoformat(),result_sha256=sha(run/'result.json'),counts=dict(checks=56,timings=80,packing=4,events=16),candidates=candidates,summary=stats,decision='independent complete MLP subgraph refinement required; serving EXP20 retained'))
write(EXP/'STATUS.json',dict(stage='screen_audited',candidates=candidates,selected_for_serving=False,next='independent MLP subgraph refinement',publication=read(run/'publication.json')))
with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(),action='Audit coverage/sample formulas/numerical flags/packing/events/source/archive; retain gateup/down only for independent refinement'))+'\n')
doc=['','## EXP23权重布局初筛结果','', '56项数值/Graph检查、80个旧/新/旧计时点、4项一次性packing记录和16组CPU事件已完成独立审核。权重数值不变，候选的stride从(K,1)变为(1,N)。','','| 投影 | M8/16耗时变化范围 | 其余形状最差变化 | 判定 |','|---|---:|---:|---|']
for p,s in stats.items():doc.append(f"| {p} | {s['primary_min_time_change_percent']:+.2f}%～{s['primary_max_time_change_percent']:+.2f}% | {s['worst_other_time_change_percent']:+.2f}% | {'独立复测' if s['candidate'] else '不采用'} |")
doc += ['', 'QKV与O在M1明显退化，因此不因M8/16单点收益而纳入。gate/up与down通过预注册投影级门槛：小M均至少5%收益，其余点不退化超过3%，漂移合格。down M512约慢1.2%，应保留并在完整子图复核。', '', '当前结果仍是随机张量微基准，不是模型重放或端到端收益。原权重stride由源码推导，下一阶段必须在真实加载路径核实。CPU事件未见aten拷贝，不证明底层库没有内部拷贝。打包在加载阶段一次执行，稳态计时不包含它；以下给出8组权重的一次性实测总成本。', '', '| 投影 | 8份重排总耗时(us) | 重排张量总字节 | 实测分配增量(bytes) |','|---|---:|---:|---:|']
for x in data['packing']:doc.append(f"| {x['projection']} | {x['total_elapsed_us']:.2f} | {x['packed_tensor_bytes']} | {x['memory_allocated_delta']} |")
doc += ['', '未改推理引擎，未建立双份常驻权重方案，工作基线仍EXP20。下一阶段比较原布局、仅gate/up、仅down、两者组合的完整gate/up→实际SwiGLU→down子图；独立种子与时间窗口、覆盖小M和prefill。先记录具体检查与门槛再写代码。']
with (REPO/'docs/small_batch_projection_20261026.md').open('a') as f:f.write('\n'.join(doc)+'\n')
with (REPO/'docs/optimization_journal.md').open('a') as f:f.write('\n\n### 2026-10-26：EXP23权重布局初筛审核\n\n56检查/80计时/4packing/16CPU事件已独立审核。gate/up M8/16约降低6.58%–7.65%，down降低21.31%–23.15%，达到初筛门槛；QKV和O因M1显著退化不采用。未接入服务，下一步独立完整MLP子图验证及实际加载路径核查，工作基线仍EXP20。详见docs/small_batch_projection_20261026.md。\n')
with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);assert not git('diff','--cached','--name-only')
 paths=['experiments/EXP-0023','docs/small_batch_projection_20261026.md','docs/optimization_journal.md']
 git('add','--',*paths);git('diff','--cached','--check');git('commit','--only','-m','experiments: audit MLP weight layout candidates and retain negative cases','--',*paths)
 head=git('rev-parse','HEAD');git('push','origin','main');assert git('ls-remote',git('remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
write(ART/'review_publication.json',dict(commit=head,remote_verified=True))
active.update(current_experiment='EXP-0023',status='Weight layout screen audited; gateup/down require independent complete MLP refinement; no serving code change',next_direction='4K complete MLP subgraph layout refinement and actual service weight-stride verification')
write(ROOT/'artifacts/active_inference_plan.json',active)
print(json.dumps(dict(commit=head,remote_verified=True,candidates=candidates,summary=stats,packing=data['packing'])))
