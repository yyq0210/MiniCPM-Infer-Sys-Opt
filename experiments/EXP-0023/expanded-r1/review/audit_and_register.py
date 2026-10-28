"""Independently audit the complete MLP result and register bounded integration."""
from pathlib import Path
from datetime import datetime
import fcntl,hashlib,json,math,statistics,subprocess
ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt';EXP=REPO/'experiments/EXP-0023/expanded-r1';ART=ROOT/'artifacts/EXP-0023/expanded-r1'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a],text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10),(a,b)
state=read(ROOT/'artifacts/EXP-0023-expanded-r1-request.json');run=Path(state['run_directory']);d=read(run/'result.json');plan=read(EXP/'plan.json')
assert state['stage']=='expanded_complete' and state['owned_children_reaped'] and d['status']=='passed' and read(run/'publication.json')['remote_verified']
assert datetime.fromisoformat(d['finished_at'])<datetime.fromisoformat(state['compute_stop_at'])
old=EXP.parent/'expanded';failed_state=read(ROOT/'artifacts/EXP-0023-expanded-request.json');failed_run=Path(failed_state['run_directory']);failed=read(failed_run/'result.json')
assert failed['checks']==[] and failed['timings']==[] and failed_state['owned_children_reaped']
assert datetime.fromisoformat(failed['finished_at'])<datetime.fromisoformat(d['started_at'])
assert sha(failed_run/'result.json')==plan['parent_failed_result_sha256']
needle="configs=[str(v) for v in getattr(kernel,'configs',[])]"
replacement="configs_available=getattr(kernel,'configs',None) is not None,configs=[str(v) for v in (getattr(kernel,'configs',None) or [])]"
assert (EXP/'trial.py').read_text().replace(replacement,needle)==(old/'trial.py').read_text()
for n,h in plan['script_hashes'].items():assert sha(EXP/n)==h,n
for p,h in plan['source_hashes'].items():assert sha(Path(p))==h,p
active=read(ROOT/'artifacts/active_inference_plan.json');assert active['selected_serving_commits']==plan['selected_commits']
for n,c in plan['selected_commits'].items():
 p=ROOT/'worktrees/EXP-0020'/n
 assert subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()==c
 assert not subprocess.check_output(['git','-C',str(p),'status','--porcelain'],text=True).strip()
ms=[1,4,8,16,30,32,64,128,512,2048];seeds=[202610261,202610262,202610263];variants=['gate_up','down','both']
assert d['Ms']==ms and d['seeds']==seeds and d['variants']==variants and d['tolerance']==dict(atol=.03,rtol=.03)
def timing(x):
 assert len(x['samples_us'])==7 and x['calls_per_replay']==8 and all(math.isfinite(v) and v>0 for v in x['samples_us'])
 close(x['median_us'],statistics.median(x['samples_us']));return x['median_us']
seen=set()
for x in d['timings']:
 key=(x['seed'],x['M'],x['variant'],x['mode']);assert key not in seen;seen.add(key)
 b,n,a=[timing(x[k]) for k in ['baseline_before','candidate','baseline_after']]
 close(x['time_change_percent'],100*(n/((b+a)/2)-1));close(x['baseline_drift_percent'],100*(a/b-1));assert x['complete_MLP']
assert seen=={(s,m,v,mode) for s in seeds for m in ms for v in variants for mode in ['hot','weights8']} and len(seen)==180
expected={(s,m,v,'torch.bfloat16','normal') for s in seeds for m in ms for v in variants}|{(202610264,m,v,'torch.float16','normal') for m in [1,8,16,512,2048] for v in variants}|{(202610265,m,v,'torch.bfloat16',dist) for m in [8,2048] for v in variants for dist in ['zero','constant']}
seen=set()
for x in d['checks']:
 key=(x['seed'],x['M'],x['variant'],x['dtype'],x['distribution']);assert key not in seen;seen.add(key)
 assert x['graph_exact_replay'] and x['graph_mutations']==['input','gate_up_weight','down_weight']
 assert 'dtype rounding at gate_up and fused activation product boundaries' in x['reference']
 for k in ['candidate_baseline_max_abs','candidate_baseline_max_scaled','candidate_reference_max_abs','mutated_reference_max_abs']:assert math.isfinite(x[k]) and x[k]>=0
assert seen==expected and len(seen)==117
source=read(EXP/'production_silu_source.json')['source'];name='triton_poi_fused_mul_silu_slice_1'
assert hashlib.sha256(source.encode()).hexdigest()==plan['production_kernel_source_sha256']
derived=source.replace("'*bf16'","'*fp16'").replace(name,name+'_fp16')
for dtype,text,production in [('torch.bfloat16',source,True),('torch.float16',derived,False)]:
 row=d['activation_sources'][dtype]
 assert row['source_sha256']==hashlib.sha256(text.encode()).hexdigest() and row['production_source']==production and row['derived_fp16_signature_only']==(not production) and row['warmup_M']==2048
 assert row['configs_available'] is False and row['configs']==[]
kernels=[]
for p in (run/'triton_cache').rglob('*.json'):
 try:meta=read(p)
 except (ValueError,UnicodeError):continue
 if not isinstance(meta,dict) or 'silu' not in str(meta.get('name','')):continue
 assert meta['name'] in [name,name+'_fp16'] and meta['num_warps'] in [4,8] and meta['num_stages']==1 and meta['shared']==0 and meta['warp_size']==64
 ir=p.with_suffix('.ttir');text=ir.read_text();dtype='f16' if meta['name'].endswith('_fp16') else 'bf16'
 assert '!tt.ptr<'+dtype+'>' in text
 files={f.name:sha(f) for f in p.parent.iterdir() if f.is_file() and f.suffix in ['.json','.ttir','.ttgir','.llir','.ptx','.cubin']}
 kernels.append(dict(path=str(p.relative_to(run)),name=meta['name'],num_warps=meta['num_warps'],num_stages=meta['num_stages'],shared=meta['shared'],dtype=dtype,hashes=files))
assert len(kernels)==4 and {(x['name'],x['num_warps']) for x in kernels}=={(n,w) for n in [name,name+'_fp16'] for w in [4,8]}
candidates=[];scores={};stats={};by_m=[]
for v in variants:
 rows=[x for x in d['timings'] if x['variant']==v];primary=[x for x in rows if x['M'] in [8,16]];other=[x for x in rows if x['M'] not in [8,16]]
 passed=all(x['time_change_percent']<=-5 for x in primary) and all(x['time_change_percent']<=3 for x in other) and all(abs(x['baseline_drift_percent'])<=3 for x in rows)
 mean=statistics.mean(x['time_change_percent'] for x in primary)
 if passed:candidates.append(v);scores[v]=mean
 stats[v]=dict(primary_min=min(x['time_change_percent'] for x in primary),primary_max=max(x['time_change_percent'] for x in primary),primary_mean=mean,other_worst=max(x['time_change_percent'] for x in other),max_abs_drift=max(abs(x['baseline_drift_percent']) for x in rows),passed=passed)
 for m in ms:
  values=[x['time_change_percent'] for x in rows if x['M']==m];by_m.append(dict(variant=v,M=m,min=min(values),max=max(values),mean=statistics.mean(values)))
summary=read(run/'screen_summary.json');assert candidates==summary['candidates']==['down','both']
for v in candidates:close(scores[v],summary['primary_mean_changes'][v])
best=min(scores.values());near=[v for v in candidates if scores[v]-best<1];selected=min(near,key=lambda v:(2 if v=='both' else 1,scores[v]))
assert selected==summary['selected_for_refinement']=='both'
for n in ['result.json','status.json','screen_summary.json']:assert sha(run/n)==sha(EXP/'artifacts'/run.name/n)
dest=EXP/'review';assert not dest.exists();dest.mkdir();(dest/'audit_and_register.py').write_text(Path(__file__).read_text())
errors={k:max(x[k] for x in d['checks']) for k in ['candidate_baseline_max_abs','candidate_baseline_max_scaled','candidate_reference_max_abs','mutated_reference_max_abs']}
write(dest/'audit.json',dict(status='complete',reviewed_at=datetime.now().astimezone().isoformat(),result_sha256=sha(run/'result.json'),counts=dict(checks=117,timings=180),candidates=candidates,selected_for_integration=selected,stats=stats,by_M=by_m,error_maxima=errors,actual_compilation=kernels,serving_selection_unchanged=True))
write(EXP/'STATUS.json',dict(stage='audited_complete',selected_for_integration=selected,selected_for_serving=False,publication=read(run/'publication.json')))
with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(),action='Independently audit all117checks/180timings, metadata-only diff,4actual compiled variants, source/archives; select both for integration only'))+'\n')
doc=['','## 2026-10-28：独立完整MLP复测与路由纠正','','正式服务计算图实际使用forward_native的SiLU与乘法，编译后合成动态SwiGLU内核；本轮复用该生成源，在独立缓存编译和预热。没有采用尚未生效的CachedOp类作为实际路径。FP16仅为派生回归，生产模型为BF16。','','初次尝试在可选编译器元数据configs=None处失败，0检查/0计时；失败完整保存。expanded-r1只修复该记录逻辑，原候选/种子/门槛不变。117项数值/Graph检查全部通过，180个完整MLP计时点已独立重算，4个实际BF16/FP16×4/8warps编译候选的元数据与IR已核验；这些是编译候选集合，不代表逐调用选中了哪一个。','','| 布局 | M8/16耗时变化 | 主目标平均变化 | 其余点最差变化 | 门槛 |','|---|---:|---:|---:|---|']
for v,s in stats.items():doc.append(f"| {v} | {s['primary_min']:+.2f}%～{s['primary_max']:+.2f}% | {s['primary_mean']:+.2f}% | {s['other_worst']:+.2f}% | {'通过' if s['passed'] else '未通过5%收益门槛'} |")
doc += ['', '选择gate/up与down同时重排进入引擎集成。仅gate/up仍有收益，但没有在全部主目标点达到预定5%；不是精度失败。both的M512耗时下降1.84%–2.22%，M2048下降1.89%–2.00%，全部60点均改善。', '', '候选与原布局输出最大绝对差0.001953125；与分段FP32参照比较时，常量输入及权重变动后的最大绝对差0.25仍通过预先固定的绝对+相对容差，不能把此数值误说成全部误差小于0.03。Graph重放与同布局eager严格一致。这是算子/子图检查，整模MATH500仍按用户暂缓。', '', '三种子、hot与8组权重轮换都是随机张量诊断，不是请求重放。完整MLP平均耗时下降12.63%不能当作整模收益；当前已选服务仍EXP20。下一EXP24先做加载时布局接入、真实权重/Graph/路由核验，然后官方4K端到端评测。']
with (REPO/'docs/small_batch_projection_20261026.md').open('a') as f:f.write('\n'.join(doc)+'\n')
new=REPO/'experiments/EXP-0024';assert not new.exists();new.mkdir()
(new/'README.md').write_text('''# EXP-0024：MLP权重加载布局接入与端到端验证

实现前假设：EXP23初筛与独立完整MLP复测已通过。gate/up与down同时改为[N,K] stride(1,N)，三个独立种子的M8/16完整MLP平均耗时下降12.63%，所有测试形状无回归。加载时一次性重排并替换原存储有望改善4K吞吐；尚无此项整模收益。
基线EXP20：plugin ce19f93eee4c73f5a26911554700457e879f891d、Gems de958ddfc4deeb43d6c94bb8652a3d4d3bc905fe。当前服务保持它们。独立工作树与分支开发，不修改冻结源；本项只改变权重布局，不称新增算子融合，不改变调度/量化/KV压缩。

拟接入UnquantizedLinearMethod.process_weights_after_loading；先核实真实加载生命周期，再实现默认关闭、可回退、幂等的精确范围适配。候选仅MiniCPM5-2B匹配配置、天数设备、BF16、非量化、TP1和当前已支持的无bias MLP两矩阵。VLLM_BATCH_INVARIANT、LoRA/其他未验证路径须守卫。保留原方法语义与Parameter元数据，unsupported场景原样返回。
重排前后权重逐元素完全相同，替换后释放原存储，不保留双份常驻权重挤占KV。只允许模型加载阶段执行；前向仍用原生linear，SwiGLU沿用真实生产编译路径。须记录当前实际weight stride及新stride，不以推导替代真实接线。

验证次序：CPU适用范围/回退/幂等检查；真实GPU参数元数据/值/布局/旧存储释放；真实服务42层gateup+42层down接线日志及Graph、原42GQA/RoPEKV/35FULL捕获尺寸、原调度、无逐次重排、KV容量；随后官方4K完整四轮弃首，独立结果审计。正向且预算足够时再16K/配对。完整模型任务精度按用户暂缓，不能声称赛事精度达标。

每阶段新有限窗口、GPU共享锁、两次空闲/端口/源hash检查、失败保留、结果与文档自动保存push。10月28日23:35计算截止、23:50保存，10月29日00:00机器关闭。比赛PR保持Draft，标题【FlagOS开放计算全球挑战赛S2.赛题2】【LeanScale】描述；无上游PR。
''')
(new/'steps.jsonl').write_text(json.dumps(dict(time=datetime.now().astimezone().isoformat(),action='Register engine integration hypothesis before modifying inference code',parent_experiment='EXP-0023/expanded-r1',chosen_layout='both'))+'\n')
with (REPO/'docs/optimization_journal.md').open('a') as f:f.write('\n\n### 2026-10-28：完整MLP审核通过，登记EXP24集成\n\nexpanded-r1的117检查/180计时/4实际编译候选及源/结果哈希独立核验通过。both在M8/16平均耗时降低12.63%，全部形状无回归；只down约7.84%，只gateup未完全达到5%门槛。原失败保留，工作服务仍EXP20。先登记EXP24加载时双MLP矩阵布局集成假设，下一步接线和官方E2E，未宣称整模收益。\n')
with (ROOT/'artifacts/docs-publish.lock').open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX);assert not git('diff','--cached','--name-only')
 paths=['experiments/EXP-0023/expanded-r1','experiments/EXP-0024','docs/small_batch_projection_20261026.md','docs/optimization_journal.md'];git('add','--',*paths);git('diff','--cached','--check')
 git('commit','--only','-m','experiments: audit complete MLP gains and register engine integration','--',*paths)
 head=git('rev-parse','HEAD');git('push','origin','main');assert git('ls-remote',git('remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
write(ART/'review_publication.json',dict(commit=head,remote_verified=True,selected_for_integration='both'))
active.update(current_experiment='EXP-0024',status='EXP23 complete MLP independently audited; EXP24 integration hypothesis registered; serving remains EXP20',next_direction='load-time gateup+down layout integration and official4K E2E')
write(ROOT/'artifacts/active_inference_plan.json',active)
print(json.dumps(dict(commit=head,remote_verified=True,selected_for_integration='both',summary=stats,actual_compiled_variants=len(kernels))))
