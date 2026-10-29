"""EXP26: unchanged top-p kernel, independently tune row program concurrency."""
from pathlib import Path
from datetime import datetime, timedelta
import argparse, ast, csv, fcntl, hashlib, json, os, re, shutil, signal, socket
import statistics, subprocess, sys, time, traceback

ROOT=Path('/workspace/LeanScale');DOC=ROOT/'MiniCPM-Infer-Sys-Opt'
EXP=DOC/'experiments/EXP-0026';ART=ROOT/'artifacts/EXP-0026'
SOURCE=Path('/usr/local/lib/python3.12/site-packages/vllm/v1/sample/ops/topk_topp_triton.py')
SAMPLER=SOURCE.with_name('topk_topp_sampler.py')
PARENT=ROOT/'worktrees/EXP-0020'
COMMITS={'vllm-plugin-FL':'ce19f93eee4c73f5a26911554700457e879f891d','FlagGems':'de958ddfc4deeb43d6c94bb8652a3d4d3bc905fe'}
def now():return datetime.now().astimezone()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def git(repo,*a):return subprocess.check_output(['git','-C',str(repo),*a],text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def step(action,**kw):
    with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(time=now().isoformat(),action=action,**kw),ensure_ascii=False)+'\n')
def publish(paths,message):
    # Reuse the already audited exact-path publication helper, not its runtime.
    helper=DOC/'experiments/EXP-0025/common.py'
    assert sha(helper)=='bb51ee4f2d240293a0f9fdfb128ccc55f1183f531eedd3dad035c79b67251cb3'
    import importlib.util
    spec=importlib.util.spec_from_file_location('exp26_publication_helper',helper)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m.publish(paths,message)
def verify_sources():
    for repo,commit in COMMITS.items():
        assert git(PARENT/repo,'rev-parse','HEAD')==commit and not git(PARENT/repo,'status','--porcelain')
    for name,h in read(EXP/'source_hashes.json').items():assert sha(Path(name))==h,name
    assert sha(EXP/'sampling_grid_experiment.py')==read(EXP/'plan.json')['script_sha256']

def prepare():
    assert not EXP.exists() and not ART.exists(),'Inspect prior preparation rather than overwrite'
    assert now().strftime('%Y-%m-%d')=='2026-10-29'
    assert read(DOC/'experiments/EXP-0025/review_20261029/audit.json')['status']=='passed'
    assert read(ROOT/'artifacts/active_inference_plan.json')['selected_serving_commits']==COMMITS
    EXP.mkdir();ART.mkdir()
    (EXP/'README.md').write_text('''# EXP-0026：4K top-p 行并行度诊断

## 假设与真实路径

在写候选之前登记：当前服务仍为 EXP20。官方 benchmark 没有显式 temperature/top-p 覆盖；模型 generation_config 为 do_sample=true、temperature=1.0、top_p=0.95，逻辑词表 130560。PlatformFL 为 OOT，TopKTopPSampler 使用 forward_native；B>=8 转入 apply_top_k_top_p_triton，B<8 保持原 PyTorch 路径。既有 EXP18/NONE/large_decode 的真实 _topk_topp_kernel 每步均值 6320.4975 us，占该窗口可见 kernel 总时间 10.9747%；不能称生产 FULL 整程占比。

原 wrapper NUM_PROGRAMS=min(num_sm,B)，本设备 num_sm=16，B64 时每 program 顺序处理四行。候选只比较 min(2*num_sm,B)、min(4*num_sm,B)，使用同一个原始 JIT 内核、同一 8192/4096 tile、默认 warps/stages、原表和输入 dtype。每 program 的 scratch 必须相应扩容，不能沿用 16 行大小而越界。算法、归约、top-k/top-p 阈值、随机数生成和请求配置不变。未接入推理引擎。

## 固定诊断协议

三种子，B8/16/32/64，V=130560，输入为正态 scale0.5/1/3，主测 p=.95/k=None；B8/16 的 grid 与原版相同，只做控制检查，不虚报加速。B32 实测 grid32；B64 比较 grid32/grid64。候选相同 grid 去重。另测 uniform/ties、稀疏 -inf、peaked、p=.0001/.5/.95/1、top-k=50、top-k 与 top-p 组合，以及带行间 padding 的 stride。

所有 masked logits 必须与当前原实现逐元素完全相同（包括 -inf 掩码），不放宽阈值；输入、p 和 k 的 Graph 重放更新后再次比较。masking 不得改变 RNG。使用同一 RNG 状态和相同 per-request generators 复验后续随机采样 token 与 RNG 状态一致。参考为当前生产实现，不宣称其数学算法已穷尽验证。

计时是 GPU event 的旧/新/旧夹测，7 样本、每样本4次 Graph 重放；每次都含 input→logits restore copy，避免原地 masking 重复处理已截断数据。两组含同样恢复开销。记录主目标全部计时、漂移、CV、scratch 字节与编译资源，不只挑最快点；随机张量诊断不等同真实 logits 或端到端收益。

选择进入下一阶段须 B64 所有三个种子×三个分布均至少5%改善、B32不回退超过3%、夹测漂移绝对值不超过3%，所有精确值/Graph/RNG检查通过。未达标则不接入。若仅某个 batch 的 grid 更好，必须另行固定 dispatch 并复核，不能在端到端测量中按成绩自适应挑选。

## 时间、保存与后续

一次性后台进程，GPU共享锁、两次空闲及端口检查，独立 Triton 缓存。启动时刷新时间，分配45分钟计算预算和10分钟归档余量，受2026-10-29 23:35停计算、23:50保存约束；预算不足不启动。完成或失败均保存 JSONL/状态/编译证据并自动推个人文档 fork，不覆盖原始失败，不自动重启。整模精度暂缓；这不是KV压缩或调度方案。

通过后先用真实服务收集采样分支和真实logits诊断，完成加载、scratch生命周期、Graph/fallback接线，再官方4K。原生随机数和筛选语义必须保留，不改为 greedy，不把微基准当E2E。源码集成到个人fork时按比赛Draft格式记录，当前只提交实验。
''')
    step('Register top-p grid-only hypothesis before candidate execution',baseline=COMMITS,GPU_started=False)
    shutil.copy2(__file__,EXP/'sampling_grid_experiment.py')
    source_files=[SOURCE,SAMPLER,Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py'),
        PARENT/'vllm-plugin-FL/vllm_fl/platform.py',PARENT/'vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/iluvatar.py']
    write(EXP/'source_hashes.json',{str(p):sha(p) for p in source_files})
    write(EXP/'plan.json',dict(experiment='EXP-0026',baseline_commits=COMMITS,script_sha256=sha(EXP/'sampling_grid_experiment.py'),
        seeds=[202610291,202610292,202610293],batches=[8,16,32,64],vocab_size=130560,scales=[.5,1.,3.],
        program_multipliers=[2,4],samples=7,graph_repeats=4,compute_minutes=45,
        daily_compute_stop='2026-10-29T23:35:00+08:00',daily_save='2026-10-29T23:50:00+08:00'))
    for p in EXP.glob('*.py'):ast.parse(p.read_text())
    verify_sources()
    pub=publish(['experiments/EXP-0026'],'experiments: preregister unchanged top-p kernel row concurrency')
    write(ART/'preparation.json',dict(status='prepared',publication=pub))
    current=now();daily=datetime.fromisoformat(read(EXP/'plan.json')['daily_compute_stop'])
    assert current.date()==daily.date() and current+timedelta(minutes=45)<=daily,'Need full diagnostic budget'
    run=ART/current.strftime('%Y%m%d_%H%M%S_grid_diagnostic')
    window=dict(created_at=current.isoformat(),compute_stop_at=(current+timedelta(minutes=45)).isoformat(),
        save_target_at=(current+timedelta(minutes=55)).isoformat(),daily_compute_stop=daily.isoformat(),daily_save='2026-10-29T23:50:00+08:00')
    write(ART/'run_window.json',window);write(EXP/'run_window_at_launch.json',window)
    launch_pub=publish(['experiments/EXP-0026/run_window_at_launch.json'],'experiments: allocate fresh top-p diagnostic budget')
    assert (datetime.fromisoformat(window['compute_stop_at'])-now()).total_seconds()>40*60
    env=dict(os.environ,PYTHONPATH=str(PARENT/'vllm-plugin-FL')+':'+str(PARENT/'FlagGems/src'),
        VLLM_PLUGINS='fl',PYTHONUNBUFFERED='1',TRITON_CACHE_DIR=str(run/'triton_cache'))
    with (ART/'controller.log').open('w') as log:
        proc=subprocess.Popen(['python3',str(EXP/'sampling_grid_experiment.py'),'--run',str(run)],
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
    ticks=(Path('/proc')/str(proc.pid)/'stat').read_text().rsplit(') ',1)[1].split()[19]
    receipt=dict(pid=proc.pid,process_start_ticks=ticks,run_directory=str(run),window=window,publication=launch_pub)
    write(ART/'launch.json',receipt)
    active=read(ROOT/'artifacts/active_inference_plan.json');active.update(current_experiment='EXP-0026',
        status='top-p unchanged-kernel row concurrency diagnostic running',next_direction='Review exact-mask/Graph/RNG and grid timing before any serving integration')
    write(ROOT/'artifacts/active_inference_plan.json',active)
    print(json.dumps(receipt),flush=True)

def execute(run):
    assert run.parent==ART;run.mkdir(exist_ok=False)
    end=datetime.fromisoformat(read(ART/'run_window.json')['compute_stop_at'])
    assert now()<end and now().date()==end.date()
    status=dict(stage='starting',started_at=now().isoformat(),pid=os.getpid(),deadline=end.isoformat())
    def state(stage,**kw):status.update(stage=stage,updated_at=now().isoformat(),**kw);write(run/'status.json',status);print(json.dumps(status),flush=True)
    def record(filename,x):
        with (run/filename).open('a') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    def abort(*a):raise TimeoutError('Bounded diagnostic deadline or termination signal')
    signal.signal(signal.SIGTERM,abort);signal.signal(signal.SIGALRM,abort)
    signal.setitimer(signal.ITIMER_REAL,(end-now()).total_seconds())
    lock=Path('/workspace/experiments/codex_20261019/runner.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify_sources()
        for port in [9031,9032]:
            with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',port))!=0
        idle=[]
        for _ in range(2):
            out=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True,timeout=20)
            m=re.search(r'(\d+)MiB\s*/\s*(\d+)MiB\s*\|\s*(\d+)%',out)
            assert m and int(m[1])<=1024 and int(m[3])<=5
            idle.append(dict(memory_mib=int(m[1]),util_percent=int(m[3])));time.sleep(3)
        write(run/'idle.json',idle)
        import torch
        # Apply the same vendor compatibility shims as the selected worker.
        import vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar
        import vllm.v1.sample.ops.topk_topp_triton as base
        import vllm.v1.sample.ops.topk_topp_sampler as sampler
        assert Path(base.__file__).resolve()==SOURCE and torch.cuda.get_device_name()=='Iluvatar BI-V150'
        plan=read(EXP/'plan.json');v=plan['vocab_size'];sm=base.num_compute_units(0)
        assert sm==16
        baseline=base.apply_top_k_top_p_triton
        buffers={};timings=[];checks=[]
        def candidate(x,k,p,programs):
            # Identical production kernel and compile-time arguments.
            key=(programs,x.shape[1])
            if key not in buffers:buffers[key]=x.new_empty((programs,x.shape[1]))
            normal,percentile=base._TRITON_TABLE_CACHE[x.device]
            base._topk_topp_kernel[(programs,)](x,x.stride(0),buffers[key],percentile,normal,
                x if k is None else k,x if p is None else p,BATCH_SIZE=x.shape[0],MASK_VALUE=-float('inf'),
                VOCAB_SIZE=x.shape[1],BLOCK_SIZE=8192,BLOCK_SIZE_TRUNC=4096,
                TOPK_ENABLED=k is not None,TOPP_ENABLED=p is not None)
            return x
        def exact(a,b):return bool(torch.equal(a,b))
        def validate(x,k,p,grid,label,random_check=False):
            a=x.clone();b=x.clone();rng=torch.cuda.get_rng_state().clone()
            baseline(a,k,p);candidate(b,k,p,grid);torch.cuda.synchronize()
            equal=exact(a,b);rng_same=torch.equal(rng,torch.cuda.get_rng_state())
            item=dict(label=label,B=x.shape[0],grid=grid,exact_masked_logits=equal,masking_rng_unchanged=bool(rng_same),
                baseline_masked=int(torch.isneginf(a).sum().item()),candidate_masked=int(torch.isneginf(b).sum().item()))
            record('checks.jsonl',item);checks.append(item)
            assert equal and rng_same,item
            if random_check:
                for per_request in [False,True]:
                    saved=torch.cuda.get_rng_state().clone()
                    def gens():return {0:torch.Generator(device='cuda').manual_seed(7123)} if per_request else {}
                    ga=gens();gb=gens();outa=sampler.random_sample(a.softmax(-1,dtype=torch.float32),ga)
                    aftera=torch.cuda.get_rng_state().clone();statesa={i:g.get_state().clone() for i,g in ga.items()}
                    torch.cuda.set_rng_state(saved)
                    outb=sampler.random_sample(b.softmax(-1,dtype=torch.float32),gb)
                    same=exact(outa,outb) and torch.equal(aftera,torch.cuda.get_rng_state()) and all(torch.equal(statesa[i],g.get_state()) for i,g in gb.items())
                    item=dict(label=label+'_rng',B=x.shape[0],grid=grid,per_request_generator=per_request,exact_tokens_and_rng=bool(same))
                    record('checks.jsonl',item);checks.append(item);assert same,item
        def graph(fn,x,k,p):
            output=x.clone()
            def call():output.copy_(x);fn(output,k,p)
            for _ in range(3):call()
            torch.cuda.synchronize()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):call()
            g.replay();torch.cuda.synchronize()
            return g,output
        def measure(g):
            for _ in range(3):g.replay()
            torch.cuda.synchronize();values=[]
            for _ in range(plan['samples']):
                start=torch.cuda.Event(enable_timing=True);finish=torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(plan['graph_repeats']):g.replay()
                finish.record();finish.synchronize()
                values.append(start.elapsed_time(finish)*1000/plan['graph_repeats'])
            return values
        state('correctness_and_timing',num_compute_units=sm)
        for seed in plan['seeds']:
            torch.manual_seed(seed)
            for b in plan['batches']:
                for scale in plan['scales']:
                    x=torch.randn((b,v),device='cuda',dtype=torch.float32)*scale
                    p=torch.full((b,),.95,device='cuda',dtype=torch.float32)
                    grids=sorted({min(b,sm*m) for m in plan['program_multipliers']})
                    for grid in grids:validate(x,None,p,grid,f'seed{seed}_scale{scale}',random_check=b==64 and scale==1.)
                    ga,oa=graph(baseline,x,None,p)
                    for grid in grids:
                        if grid==min(b,sm):continue
                        gb,ob=graph(lambda t,k,p:candidate(t,k,p,grid),x,None,p)
                        # Input and p values change at fixed pointers; neither graph may return stale data.
                        x.mul_(1.125);p.fill_(.9);ga.replay();gb.replay();torch.cuda.synchronize()
                        expected=baseline(x.clone(),None,p)
                        same=exact(oa,ob) and exact(oa,expected)
                        item=dict(label='graph_input_and_p_update',seed=seed,B=b,scale=scale,grid=grid,exact=bool(same))
                        checks.append(item);record('checks.jsonl',item);assert same,item
                        x.div_(1.125);p.fill_(.95)
                        before=measure(ga);new=measure(gb);after=measure(ga)
                        old=(statistics.median(before)+statistics.median(after))/2
                        row=dict(seed=seed,B=b,scale=scale,grid=grid,reference_grid=min(b,sm),
                            baseline_before_us=before,candidate_us=new,baseline_after_us=after,
                            baseline_median_us=old,candidate_median_us=statistics.median(new),
                            change_percent=(statistics.median(new)/old-1)*100,
                            drift_percent=(statistics.median(after)/statistics.median(before)-1)*100,
                            candidate_cv_percent=statistics.stdev(new)/statistics.mean(new)*100,
                            candidate_scratch_bytes=grid*v*4,baseline_scratch_bytes=min(b,sm)*v*4,
                            timing_scope='restore copy + unchanged masking kernel via CUDA Graph; no RNG or softmax')
                        record('timings.jsonl',row);timings.append(row)
                        del gb,ob
                    del ga,oa,x,p
        state('edge_and_rng_checks',timing_points=len(timings),checks=len(checks))
        b=64
        cases=[]
        for kind in ['uniform','tied_bf16','sparse_neginf','peaked','row_stride']:
            x=torch.randn((b,v),device='cuda',dtype=torch.float32)
            if kind=='uniform':x.zero_()
            elif kind=='tied_bf16':x=x.to(torch.bfloat16).float()
            elif kind=='sparse_neginf':x[:,::3]=-float('inf')
            elif kind=='peaked':x.mul_(.01);x[:,17]=30
            elif kind=='row_stride':
                padded=torch.empty((b,v+32),device='cuda');padded[:,:v].copy_(x);x=padded[:,:v]
            for mode in ['p_small','p_half','p_full','k_only','kp']:
                p=None if mode=='k_only' else torch.full((b,),{'p_small':.0001,'p_half':.5,'p_full':1.,'kp':.95}[mode],device='cuda')
                k=torch.full((b,),50,device='cuda',dtype=torch.int32) if mode in ['k_only','kp'] else None
                for grid in [32,64]:validate(x,k,p,grid,kind+'_'+mode,random_check=kind=='tied_bf16' and mode=='kp')
        # Preserve arbitrary row stride explicitly; clone in validate would normalize it.
        storage=torch.randn((64,v+32),device='cuda');x=storage[:,:v];p=torch.full((64,),.95,device='cuda')
        expected=baseline(x.clone(),None,p)
        candidate(x,None,p,64);same=exact(x,expected)
        item=dict(label='actual_padded_row_stride',stride=list(x.stride()),exact=bool(same))
        checks.append(item);record('checks.jsonl',item);assert same,item
        # k is read dynamically too, not baked into graph capture.
        x=torch.randn((64,v),device='cuda');k=torch.full((64,),50,device='cuda',dtype=torch.int32);p=torch.full((64,),.95,device='cuda')
        ga,oa=graph(baseline,x,k,p);gb,ob=graph(lambda t,k,p:candidate(t,k,p,64),x,k,p)
        k.fill_(127);p.fill_(.8);x.add_(.01);ga.replay();gb.replay();torch.cuda.synchronize()
        same=exact(oa,ob) and exact(oa,baseline(x.clone(),k,p))
        item=dict(label='graph_k_p_input_update',exact=bool(same));checks.append(item);record('checks.jsonl',item);assert same,item
        del ga,gb,oa,ob,x,k,p
        verify_sources()
        metadata=[]
        for p in (run/'triton_cache').glob('*/_topk_topp_kernel.json'):
            data=read(p);ir=p.with_suffix('.ttir');assert ir.exists()
            metadata.append(dict(path=str(p.relative_to(run)),sha256=sha(p),ttir_sha256=sha(ir),
                num_warps=data['num_warps'],num_stages=data['num_stages'],shared=data.get('shared'),
                n_regs=data.get('n_regs'),backend=data['target']['backend']))
        assert metadata and all(m['backend']=='corex' for m in metadata)
        write(run/'kernel_evidence.json',metadata)
        candidates=[]
        for grid in [32,64]:
            target=[r for r in timings if r['B']==64 and r['grid']==grid]
            control=[r for r in timings if r['B']==32 and r['grid']==32]
            assert len(target)==9 and len(control)==9
            passed=all(r['change_percent']<=-5 and abs(r['drift_percent'])<=3 for r in target) and all(r['change_percent']<=3 and abs(r['drift_percent'])<=3 for r in control)
            candidates.append(dict(B64_grid=grid,passed=passed,mean_change_percent=statistics.mean(r['change_percent'] for r in target),
                range_change_percent=[min(r['change_percent'] for r in target),max(r['change_percent'] for r in target)],
                max_abs_drift_percent=max(abs(r['drift_percent']) for r in target+control),
                additional_scratch_bytes=(grid-sm)*v*4))
        write(run/'results.json',dict(status='complete_diagnostic',candidates=candidates,timing_points=len(timings),checks=len(checks),
            all_exact_checks_passed=True,selected_serving_experiment='EXP-0020',end_to_end_gain='not_measured',
            next='Independent audit; no automatic serving integration'))
        state('complete',timing_points=len(timings),checks=len(checks),candidates=candidates)
        step('Complete top-p grid diagnostic; not an end-to-end result',run_name=run.name,candidates=candidates)
    except BaseException:
        state('failed',error=traceback.format_exc());step('Top-p grid diagnostic failed; preserve all partial evidence',run_name=run.name,error=traceback.format_exc())
        traceback.print_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);status['finished_at']=now().isoformat();write(run/'status.json',status)
        lock.close()
    # GPU allocations vanish when this process exits; no vLLM service was started.
    dest=EXP/'artifacts'/run.name;dest.mkdir(parents=True,exist_ok=False);manifest={}
    for f in run.iterdir():
        if f.is_file() and f.suffix in ['.json','.jsonl']:
            target=dest/f.name;target.write_bytes(f.read_bytes());manifest[f.name]=sha(f)
    write(dest/'source_hashes.json',manifest);write(EXP/'STATUS.json',status)
    try:pub=publish(['experiments/EXP-0026/artifacts/'+run.name,'experiments/EXP-0026/STATUS.json','experiments/EXP-0026/steps.jsonl'],
        'experiments: preserve top-p row concurrency diagnostic')
    except Exception as exc:pub=dict(status='push_failed_local_saved',error=repr(exc))
    write(run/'publication.json',pub);write(ART/'publication.json',pub)
    print(json.dumps(dict(status=status,publication=pub)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);args=p.parse_args()
    if args.run:execute(args.run)
    else:prepare()
