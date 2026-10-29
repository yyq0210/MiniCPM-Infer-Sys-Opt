"""Independent CPU-only EXP25 audit; original scripts, logs and results stay frozen."""
from pathlib import Path
from collections import Counter
from datetime import datetime
import ast, csv, hashlib, json, re, shutil, statistics, subprocess, sys

ROOT=Path('/workspace/LeanScale'); DOC=ROOT/'MiniCPM-Infer-Sys-Opt'
EXP=DOC/'experiments/EXP-0025'; ART=ROOT/'artifacts/EXP-0025'; REVIEW=EXP/'review_20261029'
sys.path.insert(0,str(EXP))
from common import publish
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def write(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def rows(p):
    with p.open() as f: return list(csv.DictReader(f))
def git(repo,*a): return subprocess.check_output(['git','-C',str(repo),*a],text=True,stderr=subprocess.STDOUT,timeout=120).strip()

def inspect_route(run,arm):
    folder=run/'candidate_benchmark';log=(folder/'server.log').read_text()
    assert log.count('LeanScale GQA active:')==42 and log.count('LeanScale fused RoPE+KV active:')==42
    assert 'Capturing CUDA graphs (decode, FULL)' in log and '35/35' in log
    assert 'KVReservationAsyncScheduler' not in log and 'KV growth admission enabled=True' not in log
    assert log.count('LeanScale MLP layout active:')==(84 if arm=='B' else 0)
    expected={(str(i),p) for i in range(42) for p in ['gate_up_proj','down_proj']}
    loaded=re.findall(r'LeanScale MLP layout active: prefix=model\.layers\.(\d+)\.mlp\.(gate_up_proj|down_proj)',log)
    assert Counter(loaded)==(Counter({k:1 for k in expected}) if arm=='B' else Counter())
    route=read(folder/'route_evidence.json');layout=read(folder/'layout_route_evidence.json')
    assert route['status']==layout['status']=='passed' and layout['arm']==arm
    kinds=Counter()
    for k in route['kernels']:
        p=run/k['path'];assert p.resolve().is_relative_to(run)
        assert sha(p)==k['sha256'] and sha(p.with_suffix('.ttir'))==k['ttir_sha256']
        data=read(p);ir=p.with_suffix('.ttir').read_text()
        assert data['num_warps']==4 and data['num_stages']==1 and data['target']['backend']=='corex'
        assert data['shared']==(18432 if k['kind']=='decode' else 32768)
        n=8 if k['kind']=='decode' else 64
        assert re.search(r'tt\.load[^\n]*tensor<'+str(n)+r'x128x!tt\.ptr<(?:bf16|f16)>>',ir)
        assert re.search(r'tt\.load[^\n]*tensor<2x!tt\.ptr<i32>>',ir)
        assert re.search(r'tt\.broadcast[^\n]*tensor<2x1xi64> -> tensor<2x16xi64>',ir)
        kinds[k['kind']]+=1
    assert dict(kinds)==route['kernel_counts']=={'decode':64,'prefill':64}
    modules={}
    for m in layout['compiled_modules']:
        p=run/m['path'];assert sha(p)==m['sha256']
        tree=ast.parse(p.read_text());inputs=set();counts=Counter()
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign) and isinstance(node.value,ast.Name) and node.value.id=='args':
                inputs.update(n.id for target in node.targets for n in ast.walk(target) if isinstance(n,ast.Name))
        for node in ast.walk(tree):
            if not(isinstance(node,ast.Call) and ast.unparse(node.func)=='extern_kernels.mm' and len(node.args)>=2):continue
            w=node.args[1]
            if not(isinstance(w,ast.Call) and ast.unparse(w.func)=='reinterpret_tensor' and len(w.args)>=3):continue
            try: shape=ast.literal_eval(w.args[1]);stride=ast.literal_eval(w.args[2])
            except (ValueError,TypeError):continue
            if shape not in [(2048,12288),(6144,2048)]:continue
            assert stride==((shape[1],1) if arm=='B' else (1,shape[0]))
            assert isinstance(w.args[0],ast.Name) and w.args[0].id in inputs
            counts['gate_up_proj' if shape==(2048,12288) else 'down_proj']+=1
        assert dict(counts)==m['direct_weight_mm_counts']=={'gate_up_proj':1,'down_proj':1}
        modules[m['sha256']]=str(p.relative_to(run))
    assert len(modules)==layout['compiled_mlp_blocks']==42 and layout['fx_coverage']
    for graph in layout['fx_coverage']:
        p=run/graph['path'];assert sha(p)==graph['sha256'];coverage=Counter()
        for node in ast.walk(ast.parse(p.read_text())):
            if not(isinstance(node,ast.Call) and ast.unparse(node.func)=='torch._C._nn.linear' and len(node.args)>=2 and isinstance(node.args[1],ast.Name)):continue
            m=re.search(r'modules_layers_modules_(\d+)_modules_mlp_modules_(gate_up_proj|down_proj)_parameters_weight_',node.args[1].id,re.I)
            if m:coverage[m[1],m[2]]+=1
        assert coverage==Counter({k:1 for k in expected})
    assert route['graphs']
    for g in route['graphs']:
        p=run/g['path'];assert sha(p)==g['sha256']
        text='\n'.join(l for l in p.read_text().splitlines() if not l.lstrip().startswith('#'))
        assert text.count('torch.ops.vllm.fl_rope_kv_dispatch')==42
        assert 'torch.ops.vllm.unified_kv_cache_update' not in text
    kv=int(re.search(r'GPU KV cache size: ([\d,]+) tokens',log)[1].replace(',',''))
    assert kv==layout['candidate_kv_tokens']
    return dict(kv_tokens=kv,gqa_kernels=dict(kinds),compiled_mlp_blocks=42,logical_weight_nodes=84,
        gqa_route_sha256=sha(folder/'route_evidence.json'),layout_route_sha256=sha(folder/'layout_route_evidence.json'))

def main():
    assert not REVIEW.exists(),'Review is one-shot; inspect any partial review'
    result=read(EXP/'RESULTS.json');state=read(ART/'status.json')
    assert result['status']=='complete_ABBA' and state['stage']=='complete'
    assert result['order']==['A','B','B','A'] and len(result['sessions'])==4
    assert read(ART/'publication.json')['remote_verified']
    for name,h in read(EXP/'script_hashes.json').items():assert sha(EXP/name)==h
    commits=read(EXP/'arms.json')['commits']
    for arm in commits:
        for name,h in commits[arm].items():
            repo=ROOT/'worktrees/EXP-0025'/arm/name
            assert git(repo,'rev-parse','HEAD')==h and not git(repo,'status','--porcelain')
    sessions=[]
    for i,r in enumerate(result['sessions']):
        assert r['arm']==result['order'][i] and r['index']==i+1 and r['commits']==commits[r['arm']]
        run=ART/r['run_name'];folder=run/'candidate_benchmark';s=read(run/'status.json')
        assert s['stage']=='performance_case_complete' and s['owned_children_reaped']
        assert datetime.fromisoformat(s['finished_at'])<datetime.fromisoformat(s['planned_finish_before'])
        assert read(run/'publication.json')['remote_verified']
        raw_path=next((folder/'benchmark_results').glob('raw_runs_*.csv'));summary_path=next((folder/'benchmark_results').glob('summary_*.csv'))
        raw=rows(raw_path);summary=rows(summary_path);assert len(raw)==4 and len(summary)==1
        assert summary[0]==r['summary']
        for v in raw:
            assert tuple(int(v[k]) for k in ['Prefill','Decode','Conc','Num Prompts'])==(4096,1024,64,256)
            assert v['Run Status']=='SUCCESS' and float(v['Successful Requests'])==256
            assert float(v['Total Input Tokens'])==1048576 and float(v['Total Output Tokens'])==262144
        for k,v in summary[0].items():
            if k not in ['Prefill','Decode','Conc','Num Prompts']:
                assert abs(float(v)-statistics.mean(float(x[k]) for x in raw[1:]))<=.011,(k,v)
        values=[float(x['Total tok/s']) for x in raw[1:]]
        assert values==r['retained_total_tokens_s']
        assert abs(statistics.stdev(values)/statistics.mean(values)*100-r['cv_percent'])<1e-9
        log=(folder/'benchmark.log').read_text()
        assert log.count('Failed requests:                         0')==4
        command=read(folder/'benchmark.command.json')
        assert json.loads(command[command.index('--test-cases')+1])==[[4096,1024,64,256]]
        server=read(folder/'server.command.json')
        assert json.loads(server[server.index('--compilation-config')+1])=={'cudagraph_mode':'FULL_DECODE_ONLY'}
        assert server[server.index('--gpu-memory-utilization')+1]=='0.85' and server[server.index('--max-model-len')+1]=='131072'
        env=read(run/'explicit_environment.json')
        assert env['VLLM_FL_MLP_WEIGHT_LAYOUT']==('0' if r['arm']=='A' else '1')
        for key,name in [('VLLM_CACHE_ROOT','candidate_cache'),('TRITON_CACHE_DIR','triton_cache'),('TORCHINDUCTOR_CACHE_DIR','inductor_cache')]:assert env[key]==str(run/name)
        installed=read(run/'environment_check.json');assert installed['candidate_commits']==r['commits']
        assert sha(Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py'))==installed['benchmark_sha256']=='0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565'
        for p,h in installed['installed_source_hashes'].items():assert sha(Path('/usr/local/lib/python3.12/site-packages')/p)==h
        route=inspect_route(run,r['arm']);assert route['kv_tokens']==r['kv_tokens']
        archived=EXP/'artifacts'/run.name;manifest=read(archived/'artifact_manifest.json')
        for rel,h in manifest.items():assert sha(run/rel)==h['raw_sha256'] and sha(archived/rel)==h['archived_sha256'],rel
        assert rows(archived/'candidate_benchmark/benchmark_results'/raw_path.name)==raw
        sessions.append(dict(index=i+1,arm=r['arm'],run_name=run.name,finished_at=s['finished_at'],
            retained_total_tokens_s=values,summary=summary[0],cv_percent=r['cv_percent'],
            discarded_first_tokens_s=float(raw[0]['Total tok/s']),raw_sha256=sha(raw_path),summary_sha256=sha(summary_path),
            archive_files_checked=len(manifest),publication=read(run/'publication.json'),**route))
    metrics={}
    for key,reported in result['metrics'].items():
        v=[float(r['summary'][key]) for r in sessions];a=(v[0]+v[3])/2;b=(v[1]+v[2])/2
        m=dict(A=a,B=b,change_percent=(b/a-1)*100,pair1_change_percent=(v[1]/v[0]-1)*100,
            pair2_change_percent=(v[2]/v[3]-1)*100,A_endpoint_change_percent=(v[3]/v[0]-1)*100,B_repeat_change_percent=(v[2]/v[1]-1)*100)
        for k in m: assert abs(m[k]-reported[k])<1e-9,(key,k)
        metrics[key]=m
    # Independently recompute Prometheus counter deltas; retain sampled-peak scope.
    def counter(p,key):
        vals=[float(line.rsplit(' ',1)[1]) for line in p.read_text().splitlines() if line.startswith('vllm:'+key+'{') or line.startswith('vllm:'+key+' ')]
        assert vals,key
        return sum(vals)
    schedules=[]
    for r in sessions:
        f=ART/r['run_name']/'candidate_benchmark';delta={}
        for key in ['num_preemptions_total','request_queue_time_seconds_count','request_queue_time_seconds_sum']:
            delta[key]=counter(f/'metrics_after.prom',key)-counter(f/'metrics_before.prom',key)
        assert delta['num_preemptions_total']==0 and delta['request_queue_time_seconds_count']==1024
        schedules.append(dict(delta=delta,mean_queue_seconds_all_four_rounds=delta['request_queue_time_seconds_sum']/1024))
    for label,ai,bi in [('pair1',0,1),('pair2',3,2)]:
        accounting=read(EXP/(label+'_scheduler.json'))
        for key,i in [('reference',ai),('candidate',bi)]:
            assert accounting[key]['counter_delta']==schedules[i]['delta']
            assert abs(accounting[key]['mean_queue_seconds_all_four_rounds']-schedules[i]['mean_queue_seconds_all_four_rounds'])<1e-9
    conclusion='No demonstrated 4K gain: paired throughput -0.457%/-0.526%, mean -0.492%. Retain EXP20; do not claim statistically significant regression with two sessions per arm.'
    audit=dict(status='passed',checked_at=datetime.now().astimezone().isoformat(),metrics=metrics,sessions=sessions,
        all_successful_requests=4096,failures=0,schedules=schedules,selected_experiment='EXP-0020',conclusion=conclusion,
        original_results_sha256=sha(EXP/'RESULTS.json'),publication=read(ART/'publication.json'),
        scope_correction='The frozen scheduler helper says historical, not interleaved in a template field. These inputs are actually same-day ABBA sessions. Numeric counters are independently verified; original evidence is preserved.',
        full_model_task_accuracy='deferred_by_user')
    REVIEW.mkdir();write(REVIEW/'audit.json',audit);shutil.copy2(__file__,REVIEW/'audit_abba.py')
    write(REVIEW/'decision.json',dict(selected_experiment='EXP-0020',selected_commits=commits['A'],
        candidate_experiment='EXP-0024',decision='not_selected_no_demonstrated_e2e_gain',reason=conclusion,
        next='Inspect actual production sampling path and 4K stage evidence; register a separate bounded experiment'))
    lines=['# MLP 权重布局：同日 4K ABBA 复测结论','',
        '保留 EXP20。加载时重排 gate/up 与 down 权重已正确接入，但本次同日交错对照没有端到端收益；候选分支保持默认关闭，不继续推广到 16K。',
        '', '## 官方性能结果','',
        '2026-10-29 固定 A1→B1→B2→A2。A 是 EXP20，B 是 EXP24；各自独立服务与编译缓存，全部采用官方 [4096,1024,64,256] 四轮弃首，合计 4096 请求全部成功。没有额外剔除慢轮或混入历史数据。','',
        '| 会话 | 保留三轮 tok/s | 均值 tok/s | CV | KV 容量 tokens |','|---|---|---:|---:|---:|']
    for r in sessions:lines.append(f"| {r['index']}/{r['arm']} | {r['retained_total_tokens_s']} | {float(r['summary']['Total tok/s']):.2f} | {r['cv_percent']:.3f}% | {r['kv_tokens']} |")
    lines+=['','| 指标 | A：EXP20 | B：新布局 | 变化 |','|---|---:|---:|---:|']
    for key,m in metrics.items():lines.append(f"| {key} | {m['A']:.3f} | {m['B']:.3f} | {m['change_percent']:+.3f}% |")
    lines+=['', '两对吞吐分别为 -0.457% 和 -0.526%，均未改善；A 首尾变化 +0.504%，B 重复变化 +0.434%。平均 -0.492% 是这四个会话的描述统计，不是已证明的固定回退幅度。每组只有两个独立会话，保留轮有 3.34%–5.37% 的 CV；ABBA 仅平衡近似线性漂移，不作统计显著性声明。',
        '', '四个会话的保留轮次均呈上升趋势。虽然慢轮会影响均值，仍遵守事先规定仅弃首轮；不能在看见数据后改成只取末轮。下一步也不继续反复测试同一方案直到出现正数。',
        '', '## 执行路径与容量核验','',
        'A 的全部 MLP 权重保持原 stride(K,1)，B 仅在加载时转换为 stride(1,N)，前向转置视图分别为 (1,K) 与 (N,1)。每个会话均验证 42 个编译分图、84 个原模型权重节点；B 的 84 次重排逐元素保持值及 Parameter 对象，A 没有重排。没有新增 GEMM 融合。',
        '', '四个服务均保留 42 层 GQA/RoPE+KV、35 FULL Graph 捕获尺寸，实际 64 decode+64 prefill 编译内核哈希及分页广播 IR 通过核验。代码、官方脚本、安装源、缓存隔离和归档哈希均复核。',
        '', 'KV 容量 A1 为 522256，其余三次均为 522352。昨日 B 曾少 96 tokens，今日 A1 也出现相同量级差异，因此不能将这 6 个 block 的变化直接归因为权重重排；具体分配原因仍未测定。全部会话抢占为 0，日志采样 KV 峰值约 62%。',
        '', '平均排队时间（含弃用首轮）依次为 '+', '.join(f"{r['mean_queue_seconds_all_four_rounds']:.6f}s" for r in schedules)+'。这一指标不与保留三轮吞吐混为同一统计范围。原调度分析 helper 的 limitations 字段沿用“历史、非交错”的旧模板；本实验真实输入为同日 ABBA，原文件不覆盖，审核记录纠正描述并独立重算全部计数器。',
        '', '## 为什么没有采用','',
        '完整 MLP 微基准在 M8/16 约快 12.63%，而 M2048 只有约 2%。真实请求同时包含 attention、其他投影、采样、不同 batch、42 层权重工作集和主机开销。微基准热权重/8 组轮换并不等价于全模型，无法由它推出吞吐收益。现有证据只支持“不采用该全局布局方案”，还没有定位收益被抵消的具体内核或阶段。',
        '', '候选已接线成功，因此这不是“开关没生效”的假阴性。保留默认关闭的实现、失败尝试和全部测量，不改变用户已选的 EXP20。整模任务精度按用户要求暂缓，没有宣称比赛精度达标。',
        '', '## 后续方向','',
        '转入 4K 的另一个可见热点：采样路径。先核对官方请求实际参数、生产 sampler 分支、logits dtype/词表尺寸及既有 profile，再登记单独实验。不能把 NONE 模式 12.70% 采样占比当作生产 FULL 整程占比，也不能在没有确认采样语义时替换为 greedy argmax。只在真实路径明确后做无损候选及数值/随机数/tie/Graph 检查。KV 压缩仍需独立精度验证。',
        '', '## 复查入口','',
        '- [原始汇总与逐会话归档](../experiments/EXP-0025/RESULTS.md)',
        '- [独立审核](../experiments/EXP-0025/review_20261029/audit.json)',
        '- [前一天结果及前后实现思路](mlp_weight_layout_e2e_20261028.md)',
        '- 插件候选 commit `41fd7a47e3f39badb9e455fb04b2fe7c888d3db1`；当前保留 `ce19f93eee4c73f5a26911554700457e879f891d`，FlagGems 均为 `de958ddfc4deeb43d6c94bb8652a3d4d3bc905fe`。']
    report=DOC/'docs/mlp_weight_layout_abba_20261029.md';report.write_text('\n'.join(lines)+'\n')
    for name in ['optimization_journal.md','small_batch_projection_20261026.md','mlp_weight_layout_e2e_20261028.md','iluvatar_operator_optimization_backlog.md']:
        with (DOC/'docs'/name).open('a') as f:
            f.write('\n\n## 2026-10-29：EXP25 同日 4K ABBA 已审核\n\nA 均值 3564.105、B 3546.580 tok/s（-0.492%），两相邻配对均未改善（-0.457%/-0.526%）；16 轮、4096 请求全部成功。保持 EXP20，停止推广全局 MLP 权重布局，不把微基准收益当作整模收益。下一步先核实实际采样路径。详见[完整分析](mlp_weight_layout_abba_20261029.md)。\n')
    with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(),action='Independent full ABBA audit passed; retain EXP20',conclusion=conclusion))+'\n')
    paths=['experiments/EXP-0025/review_20261029','experiments/EXP-0025/steps.jsonl']+['docs/'+n for n in ['mlp_weight_layout_abba_20261029.md','optimization_journal.md','small_batch_projection_20261026.md','mlp_weight_layout_e2e_20261028.md','iluvatar_operator_optimization_backlog.md']]
    pub=publish(paths,'experiments: audit MLP layout ABBA and retain EXP20')
    write(ART/'review_publication.json',pub)
    active=read(ROOT/'artifacts/active_inference_plan.json')
    assert active['selected_serving_commits']==commits['A']
    active.update(current_experiment='EXP-0025',status='ABBA audited; no demonstrated MLP layout gain; EXP20 retained',
        next_direction='Inspect actual sampling path and 4K profile before a separate experiment',review_commit=pub['commit'])
    write(ROOT/'artifacts/active_inference_plan.json',active)
    print(json.dumps(dict(publication=pub,metrics=metrics,requests=4096,conclusion=conclusion)),flush=True)

if __name__=='__main__':main()
