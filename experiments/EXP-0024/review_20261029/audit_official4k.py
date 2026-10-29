"""Independent, CPU-only audit of the completed EXP24 official 4K case."""
from pathlib import Path
from datetime import datetime
from collections import Counter
import ast,csv,fcntl,hashlib,json,re,shutil,statistics,subprocess

ROOT=Path('/workspace/LeanScale');DOC=ROOT/'MiniCPM-Infer-Sys-Opt';EXP=DOC/'experiments/EXP-0024'
RUN=ROOT/'artifacts/EXP-0024/e2e-r1/20261028_192854_official_4k_r1'
REVIEW=EXP/'review_20261029';FOLDER=RUN/'candidate_benchmark'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open() as f:return list(csv.DictReader(f))
def git(repo,*a):return subprocess.check_output(['git','-C',str(repo),*a],text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def main():
    assert not REVIEW.exists()
    state=read(RUN/'status.json');assert state['stage']=='performance_case_complete' and state['owned_children_reaped']
    assert read(RUN/'publication.json')['remote_verified']
    raw_path=next((FOLDER/'benchmark_results').glob('raw_runs_*.csv'))
    summary_path=next((FOLDER/'benchmark_results').glob('summary_*.csv'))
    raw=rows(raw_path);summary=rows(summary_path);assert len(raw)==4 and len(summary)==1
    for r in raw:
        assert tuple(int(r[k]) for k in ['Prefill','Decode','Conc','Num Prompts'])==(4096,1024,64,256)
        assert r['Run Status']=='SUCCESS' and float(r['Successful Requests'])==256
        assert float(r['Total Input Tokens'])==4096*256 and float(r['Total Output Tokens'])==1024*256
    for k,v in summary[0].items():
        if k in ['Prefill','Decode','Conc','Num Prompts']:continue
        assert abs(float(v)-statistics.mean(float(r[k]) for r in raw[1:]))<=.011,(k,v)
    command=read(FOLDER/'benchmark.command.json')
    assert json.loads(command[command.index('--test-cases')+1])==[[4096,1024,64,256]]
    server=read(FOLDER/'server.command.json')
    assert json.loads(server[server.index('--compilation-config')+1])=={'cudagraph_mode':'FULL_DECODE_ONLY'}
    assert server[server.index('--gpu-memory-utilization')+1]=='0.85'
    assert server[server.index('--max-model-len')+1]=='131072'
    env=read(RUN/'environment_check.json')
    assert sha(Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py'))==env['benchmark_sha256']=='0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565'
    for p,h in env['installed_source_hashes'].items():assert sha(Path('/usr/local/lib/python3.12/site-packages')/p)==h,p
    validation=read(RUN/'candidate_validation.json');plan=validation['plan']
    for n,h in plan['script_hashes'].items():assert sha(EXP/'e2e-r1'/n)==h,n
    assert state['commits']==plan['candidate_commits']==env['candidate_commits']
    for name,h in state['commits'].items():
        repo=ROOT/'worktrees/EXP-0024'/name
        assert git(repo,'rev-parse','HEAD')==h and not git(repo,'status','--porcelain')
    for n,h in validation['source_hashes'].items():assert sha(ROOT/'worktrees/EXP-0024/vllm-plugin-FL'/n)==h
    parent=read(RUN/'parent_reference.json');ref=Path(parent['source'])
    assert sha(next((ref/'benchmark_results').glob('raw_runs_*.csv')))==parent['raw_sha256']
    assert sha(next((ref/'benchmark_results').glob('summary_*.csv')))==parent['summary_sha256']
    assert sha(ref/'route_evidence.json')==parent['route_sha256']
    retained=[float(r['Total tok/s']) for r in raw[1:]]
    metrics={}
    for k in ['Total tok/s','Output tok/s','Mean TTFT (ms)','Mean TPOT (ms)','Mean ITL (ms)']:
        old=float(parent['summary'][0][k]);new=float(summary[0][k])
        metrics[k]=dict(exp20=old,exp24=new,change_percent=(new/old-1)*100)
    comparison=read(RUN/'comparison.json')[0]
    cv=statistics.stdev(retained)/statistics.mean(retained)*100
    assert comparison['candidate_retained_runs']==retained
    assert abs(comparison['candidate_cv_percent']-cv)<1e-9
    assert abs(comparison['throughput_change_percent']-metrics['Total tok/s']['change_percent'])<1e-9
    assert abs(comparison['ttft_change_percent']-metrics['Mean TTFT (ms)']['change_percent'])<1e-9
    log=(FOLDER/'server.log').read_text()
    assert log.count('LeanScale MLP layout active:')==84
    assert log.count('LeanScale GQA active:')==42 and log.count('LeanScale fused RoPE+KV active:')==42
    assert 'Capturing CUDA graphs (decode, FULL)' in log and '35/35' in log
    assert 'KVReservationAsyncScheduler' not in log and 'KV growth admission enabled=True' not in log
    loaded=re.findall(r'LeanScale MLP layout active: prefix=model\.layers\.(\d+)\.mlp\.(gate_up_proj|down_proj)',log)
    expected={(str(i),p) for i in range(42) for p in ['gate_up_proj','down_proj']}
    assert Counter(loaded)==Counter({k:1 for k in expected})
    route=read(FOLDER/'route_evidence.json');layout=read(FOLDER/'layout_route_evidence.json')
    assert route['status']==layout['status']=='passed'
    kinds=Counter()
    for k in route['kernels']:
        p=RUN/k['path'];assert p.resolve().is_relative_to(RUN)
        assert sha(p)==k['sha256'] and sha(p.with_suffix('.ttir'))==k['ttir_sha256']
        data=read(p);assert data['num_warps']==4 and data['num_stages']==1 and data['target']['backend']=='corex'
        assert data['shared']==(18432 if k['kind']=='decode' else 32768)
        ir=p.with_suffix('.ttir').read_text()
        assert re.search(r'tt\.broadcast[^\n]*tensor<2x1xi64> -> tensor<2x16xi64>',ir)
        kinds[k['kind']]+=1
    assert dict(kinds)==route['kernel_counts']=={'decode':64,'prefill':64}
    modules={}
    for m in layout['compiled_modules']:
        p=RUN/m['path'];assert sha(p)==m['sha256']
        tree=ast.parse(p.read_text());inputs=set();counts=Counter()
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign) and isinstance(node.value,ast.Name) and node.value.id=='args':
                inputs.update(n.id for target in node.targets for n in ast.walk(target) if isinstance(n,ast.Name))
        for node in ast.walk(tree):
            if not(isinstance(node,ast.Call) and ast.unparse(node.func)=='extern_kernels.mm' and len(node.args)>=2):continue
            w=node.args[1]
            if not(isinstance(w,ast.Call) and ast.unparse(w.func)=='reinterpret_tensor' and len(w.args)>=3):continue
            try:shape=ast.literal_eval(w.args[1]);stride=ast.literal_eval(w.args[2])
            except (ValueError,TypeError):continue
            if shape not in [(2048,12288),(6144,2048)]:continue
            assert stride==(shape[1],1) and isinstance(w.args[0],ast.Name) and w.args[0].id in inputs
            counts['gate_up_proj' if shape==(2048,12288) else 'down_proj']+=1
        assert dict(counts)==m['direct_weight_mm_counts']=={'gate_up_proj':1,'down_proj':1}
        modules[m['sha256']]=str(p.relative_to(RUN))
    assert len(modules)==layout['compiled_mlp_blocks']==42
    for graph in layout['fx_coverage']:
        p=RUN/graph['path'];assert sha(p)==graph['sha256']
        coverage=Counter()
        for node in ast.walk(ast.parse(p.read_text())):
            if not(isinstance(node,ast.Call) and ast.unparse(node.func)=='torch._C._nn.linear' and len(node.args)>=2 and isinstance(node.args[1],ast.Name)):continue
            m=re.search(r'modules_layers_modules_(\d+)_modules_mlp_modules_(gate_up_proj|down_proj)_parameters_weight_',node.args[1].id,re.I)
            if m:coverage[m[1],m[2]]+=1
        assert coverage==Counter({k:1 for k in expected})
    assert layout['fx_coverage']
    for g in route['graphs']:
        p=RUN/g['path'];assert sha(p)==g['sha256']
        t='\n'.join(l for l in p.read_text().splitlines() if not l.lstrip().startswith('#'))
        assert t.count('torch.ops.vllm.fl_rope_kv_dispatch')==42
        assert 'torch.ops.vllm.unified_kv_cache_update' not in t
    kv=int(re.search(r'GPU KV cache size: ([\d,]+) tokens',log)[1].replace(',',''))
    assert kv==layout['candidate_kv_tokens']==522256 and layout['reference_kv_tokens']==522352
    archived=EXP/'artifacts/runs'/RUN.name
    archive_manifest=read(archived/'artifact_manifest.json')
    for rel,hashes in archive_manifest.items():
        assert sha(RUN/rel)==hashes['raw_sha256'] and sha(archived/rel)==hashes['archived_sha256'],rel
    assert rows(archived/'candidate_benchmark/benchmark_results'/raw_path.name)==raw
    assert rows(archived/'candidate_benchmark/benchmark_results'/summary_path.name)==summary
    schedule=read(RUN/'scheduler_comparison.json')
    assert schedule['candidate']['counter_delta']['num_preemptions_total']==0
    assert schedule['candidate']['counter_delta']['request_queue_time_seconds_count']==1024
    conclusion='No demonstrated 4K end-to-end gain; historical-session difference -1.07% versus retained CV5.38%; keep EXP20 selected, no causal-regression claim'
    audit=dict(status='passed',checked_at=datetime.now().astimezone().isoformat(),run_name=RUN.name,
         run_finished_at=state['finished_at'],candidate_commits=state['commits'],metrics=metrics,
         all_four_successful_requests=1024,all_four_failures=0,discarded_warmup_total_tokens_s=float(raw[0]['Total tok/s']),
         retained_total_tokens_s=retained,retained_cv_percent=cv,raw_sha256=sha(raw_path),summary_sha256=sha(summary_path),
         layout_route_sha256=sha(FOLDER/'layout_route_evidence.json'),gqa_route_sha256=sha(FOLDER/'route_evidence.json'),
         compiled_mlp_blocks=42,logical_weight_arguments=84,gqa_kernels=dict(kinds),
         kv_reference_tokens=522352,kv_candidate_tokens=kv,kv_change_tokens=kv-522352,
         kv_capacity_change_percent=(kv/522352-1)*100,archive_files_checked=len(archive_manifest),
         archive_commit=read(RUN/'publication.json')['commit'],conclusion=conclusion,
         full_model_task_accuracy='deferred_by_user')
    REVIEW.mkdir();write(REVIEW/'audit.json',audit);shutil.copy2(Path(__file__),REVIEW/'audit_official4k.py')
    lines=['# MLP权重布局优化：接入成功，4K端到端收益尚未成立','',
      '结论：完整MLP微基准的改善未在这次官方4K会话中转化为可确认的整模收益。继续使用EXP20；EXP24保留为默认关闭的实验分支，不进入当前最优配置。',
      '', '## 测量结果','',
      '2026-10-28 19:28:54–20:05:12，官方[4096,1024,64,256]，四轮弃首；4×256请求全部成功。基准脚本、采样、模型、编译模式、调度保持原配置。与EXP20历史会话比较，未做同日交错配对。','',
      '| 指标 | EXP20 | EXP24 | 变化 |','| --- | ---: | ---: | ---: |']
    for k,m in metrics.items():lines.append(f"| {k} | {m['exp20']:.2f} | {m['exp24']:.2f} | {m['change_percent']:+.3f}% |")
    lines += ['',f'EXP24保留三轮吞吐：{retained} tok/s，样本CV {cv:.3f}%；EXP20 CV 3.269%。首轮2083.27按预注册协议舍弃，第二轮3322.74虽慢仍保留。不能再去掉慢轮制造提升。',
      '', '吞吐差异约-1.07%，小于本次保留轮次的波动幅度。这里只有一次候选会话和历史对照；既不能称优化有效，也不能据此断言布局导致稳定的1.07%回退。CV是描述统计，不是置信区间。',
      '', '## 接入与资源证据','',
      '- 42层gate/up和42层down均在权重加载后一次性重排，逻辑值完全相同，Parameter对象及元数据保留。测试证明旧GPU存储释放；前向仍调用原生linear。',
      '- 42个独立编译模块各有两个MLP matmul，直接读取模型参数的新转置视图。完整FX图逐一覆盖全部84个原始权重参数；原42GQA、42RoPE+KV、35 FULL捕获尺寸保持。实际缓存64个decode与64个prefill GQA核的元数据和TTIR重新核验。',
      '- 正式成功会话KV容量522256 tokens，相对EXP20的522352少96 tokens（6个block，-0.01838%），不是完全相等；前一次未计时的启动曾得到522352。记录这一差异，不将其称为KV压缩收益，也不据此推断确定的分配原因。',
      '- 两会话全部四轮的抢占均为0，日志采样KV峰值均62.0%。候选平均排队7.27278秒，对照7.27999秒；这是包含弃用首轮的计数器口径，不能混作保留三轮吞吐口径。',
      '', '## 为什么微基准收益不等于整模收益','',
      'EXP23三种子完整MLP在M8/16平均降低12.63%，但M2048只降低约1.9%–2.0%。官方负载还包含prefill attention、其他投影、KV访问、调度和不同batch阶段；只加速其中一段，整模增益必然被其余工作稀释。M8/16尾部占比及MLP实际贡献还需生产图测量。',
      '', '微基准的热权重/8组权重轮换也不是42层完整模型的内存访问重放。本次会话保留轮次存在明显漂移，可能掩盖小幅差异。这些是待验证解释，现有数据不能选定一个原因。',
      '', '## 前后实现思路与下一步','',
      '原实现W逻辑[N,K]且stride(K,1)，linear转置视图stride(1,K)。新实现仅在加载时W.T.contiguous().T，使W stride(1,N)，linear转置视图stride(N,1)。没有新增算子融合，没有改变SwiGLU数学定义、调度或KV策略。开关默认关闭，并纳入编译缓存标识以隔离两种布局。',
      '', '下一步优先对EXP20与EXP24做预注册的同日4K配对，保留全部会话和慢轮；必要时对相同负载阶段进行Graph内热点诊断，确认MLP真实占比。当前不直接推广至16K，不替换EXP20，也不开展KV压缩。新实验必须另开时间窗口；完整模型任务精度仍按用户要求暂缓。',
      '', '## 可追溯记录','',
      '- 代码：[个人fork分支](https://github.com/yyq0210/vllm-plugin-FL/tree/leanscale/iluvatar-mlp-weight-layout)，commit `41fd7a47e3f39badb9e455fb04b2fe7c888d3db1`；FlagGems保持`de958ddfc4deeb43d6c94bb8652a3d4d3bc905fe`。',
      f'- [完整四轮归档](../experiments/EXP-0024/artifacts/runs/{RUN.name}/RESULTS.md)，自动推送`dab7c4a70699d87b210cd3b41770381efe316b5f`。',
      '- [独立审核及复算脚本](../experiments/EXP-0024/review_20261029/audit.json)。',
      '- CPU/GPU测试准备问题和首个整模检查脚本的“单模块包含42层”错误均保留。修复仅适配实际分图组织，先在原缓存离线验证，再独立启动；首个服务0轮计时，没有拼接或选择性复跑性能数据。']
    report=DOC/'docs/mlp_weight_layout_e2e_20261028.md';report.write_text('\n'.join(lines)+'\n')
    for filename,text in [('optimization_journal.md','2026-10-29：审核昨晚EXP24官方4K'),('small_batch_projection_20261026.md','EXP24整模测评结论')]:
        with (DOC/'docs'/filename).open('a') as f:f.write('\n\n### '+text+'\n\nEXP24已接入并完成官方四轮4K：3540.86 tok/s，对EXP20的3579.12低1.069%，保留轮CV5.382%。未确认整模收益，保持EXP20；不把微基准12.63%误报为E2E。KV522256 vs522352（-96 tokens）完整记录。[详细结果与思路对比](mlp_weight_layout_e2e_20261028.md)。\n')
    write(EXP/'review_20261029/decision.json',dict(selected_experiment='EXP-0020',candidate='EXP-0024',decision='retain_opt_in_experiment_not_selected',reason=conclusion,next='preregistered same-day 4K paired comparison after a fresh run window'))
    with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(),action='Independent completed official4K audit; retain EXP20, no E2E benefit claim',audit=audit))+'\n')
    with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);assert not git(DOC,'diff','--cached','--name-only')
        paths=['experiments/EXP-0024/review_20261029','experiments/EXP-0024/steps.jsonl','docs/mlp_weight_layout_e2e_20261028.md','docs/optimization_journal.md','docs/small_batch_projection_20261026.md']
        git(DOC,'add','--',*paths);git(DOC,'diff','--cached','--check')
        git(DOC,'commit','--only','-m','experiments: audit MLP layout 4K result and retain prior best','--',*paths)
        head=git(DOC,'rev-parse','HEAD');git(DOC,'push','origin','main');assert git(DOC,'ls-remote',git(DOC,'remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
    write(RUN/'review_publication.json',dict(commit=head,remote_verified=True,selected_experiment='EXP-0020'))
    active=read(ROOT/'artifacts/active_inference_plan.json')
    assert active['selected_experiment']=='EXP-0020'
    active.update(current_experiment='EXP-0024',status='Official4K audited; no demonstrated gain; EXP20 retained',
                  next_direction='same-day 4K paired comparison with fresh time window',review_commit=head)
    write(ROOT/'artifacts/active_inference_plan.json',active)
    print(json.dumps(dict(review_commit=head,remote_verified=True,audit=audit)),flush=True)

if __name__=='__main__':main()
