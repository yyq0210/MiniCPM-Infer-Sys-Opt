"""Prepare CPU-only paired validation while the separate 16K case is running."""
import shutil
import subprocess
import tempfile
import traceback
from common import *

STAGING=Path(__file__).resolve().parent
REFS={'A':ROOT/'artifacts/EXP-0016/20261022_145853_official_4k',
      'B':ROOT/'artifacts/EXP-0020/20261026_125756_official_4k'}

def checks():
    from route_checks import verify_route
    results=[]
    with tempfile.TemporaryDirectory(prefix='exp21_route_') as temp:
        temp=Path(temp)
        for arm,ref in REFS.items():
            output=temp/(arm+'_full_route.json');verify_route(ref,arm,output)
            evidence=read(output);assert evidence['kernel_counts']=={'decode':64,'prefill':64}
            results.append(dict(name=arm+'_all128_actual_service_kernels',passed=True,sha256=sha(output)))
            f=temp/arm;folder=f/'candidate_benchmark';folder.mkdir(parents=True)
            log=(ref/'candidate_benchmark/server.log').read_text();(folder/'server.log').write_text(log)
            graph=next(p for p in (ref/'candidate_cache').rglob('computation_graph.py') if 'torch.ops.vllm.fl_rope_kv_dispatch' in p.read_text())
            dst=f/'candidate_cache/fixture/computation_graph.py';dst.parent.mkdir(parents=True);shutil.copy2(graph,dst)
            metas={}
            for kind,shape in [('decode','8x128x!tt.ptr<'),('prefill','64x128x!tt.ptr<')]:
                src=next(p for p in (ref/'triton_cache').glob('*/_attention.json') if shape in p.with_suffix('.ttir').read_text())
                dst=f/'triton_cache'/kind/src.name;dst.parent.mkdir(parents=True)
                shutil.copy2(src,dst);shutil.copy2(src.with_suffix('.ttir'),dst.with_suffix('.ttir'));metas[kind]=dst
            def test(name,expected):
                try:verify_route(f,arm);ok=True
                except AssertionError:ok=False
                assert ok==expected,(arm,name)
                results.append(dict(name=arm+'_'+name,passed=True))
            test('positive_fixture',True)
            for kind,p in metas.items():
                original=p.read_text()
                for key,value in [('num_stages',2),('num_warps',8),('shared',read(p)['shared']+2048)]:
                    m=json.loads(original);m[key]=value;write(p,m);test(kind+'_wrong_'+key,False);p.write_text(original)
                p.unlink();test(kind+'_missing_compile',False);p.write_text(original)
                ir=p.with_suffix('.ttir');original=ir.read_text()
                old,new=('8x128x!tt.ptr<','16x128x!tt.ptr<') if kind=='decode' else ('64x128x!tt.ptr<','32x128x!tt.ptr<')
                changed=original.replace(old,new);assert changed!=original;ir.write_text(changed);test(kind+'_wrong_Q_rows',False);ir.write_text(original)
                width=32 if arm=='A' and kind=='prefill' else 2
                old=f'tensor<{width}x!tt.ptr<i32>>';new=f'tensor<{2 if width==32 else 32}x!tt.ptr<i32>>'
                changed=original.replace(old,new);assert changed!=original;ir.write_text(changed);test(kind+'_wrong_page_loads',False);ir.write_text(original)
                if width==2:
                    changed=original.replace('tensor<2x1xi64> -> tensor<2x16xi64>','tensor<2x1xi32> -> tensor<2x16xi32>')
                    assert changed!=original;ir.write_text(changed);test(kind+'_non_int64_broadcast',False);ir.write_text(original)
            for name,bad in [('wrong_decode_log',log.replace('decode_rows=8','decode_rows=16',1)),
              ('wrong_prefill_stage_log',log.replace('prefill_stages=1','prefill_stages=2',1)),
              ('withdrawn_scheduler',log+'\nKVReservationAsyncScheduler\n'),
              ('missing_layer',log.replace('LeanScale GQA active:','missing:',1))]:
                (folder/'server.log').write_text(bad);test(name,False)
            changed=log.replace('prefill_layout=1','prefill_layout=0',1) if arm=='B' else log+'\nprefill_layout=1\n'
            (folder/'server.log').write_text(changed);test('wrong_prefill_layout_log',False)
            (folder/'server.log').write_text(log)
            try:verify_route(f,'A' if arm=='B' else 'B');other=True
            except AssertionError:other=False
            assert not other;results.append(dict(name=arm+'_reject_other_arm',passed=True))
    return results

def main():
    assert not EXP.exists() and not WT.exists() and not ART.exists(),'Inspect partial setup rather than overwrite'
    assert not git(REPO,'diff','--cached','--name-only')
    official=read(REPO/'experiments/EXP-0020/review_4k.json')
    assert official['status']=='complete_case_audited' and official['candidate_commits']==COMMITS['B']
    assert read(ROOT/'artifacts/active_inference_plan.json')['selected_serving_commits']==COMMITS['A']
    daily=read(REPO/'experiments/EXP-0020/launches/daily_20261026.json');assert daily['user_confirmed_shutdown']
    assert now().strftime('%Y-%m-%d')=='2026-10-26'
    EXP.mkdir();WT.mkdir();ART.mkdir()
    (EXP/'README.md').write_text('''# EXP-0021：prefill页号广播的官方4K固定ABBA复测

建立工作树之前记录假设：EXP20相对EXP16的历史4K吞吐观测+2.95%，与保留轮CV3.27%接近，需在同期独立会话中复核。
本实验只做配对归因，不改推理/算子代码。A=EXP16原prefill页表读取，B=EXP20页号广播；两者均64行/4warps/stage1/tile32。
decode8行页号广播、RoPE+KV融合、原调度、完整KV、服务和官方benchmark参数保持。

固定A1→B1→B2→A2。每会话新服务、独立vLLM/Triton缓存、完整4K[4096,1024,64,256]四轮弃首。
全部会话均保留首轮与慢轮，不把任何历史结果加入本实验，不按性能结果提前截断；仅失败/绝对截止才停止。
启动前需EXP20完整4K和16K独立审核，吞吐不得退化>3%、TTFT/TPOT不得退化>5%；此条件在ABBA开始前评估，不是中途挑选。
准备阶段可与16K并行做CPU检查，不能占用GPU或修改活动代码。准备完成不表示ABBA已经启动。

两组独立detached工作树固定提交；每会话前后核对代码/脚本/benchmark哈希。共享GPU锁覆盖全组，每会话检测空闲和端口。
实际路由要求42层GQA和RoPEKV、35个FULL_DECODE_ONLY捕获尺寸；decode8行/18KiB/pagetable2，prefill64行/32KiB。
A prefill必须实际32页号加载；B必须2页号、int64广播至32 token lane；两组都stage1，不套用旧stage2对照门槛。
CPU检查先验证两版真实已完成4K的各128个编译核，并故障注入错误页表读取/位宽/stage/shape/shared/log/缺层/劣化调度。

启动时分配新的235分钟窗口，单会话55分钟上限，受当日23:35计算截止约束；23:50保存，00:00关机。
全部ABBA起跑前必须有完整预算，既不缩短轮数，也不继承EXP20的窗口。每会话停止服务后保存、自动push，再进入下一会话。
失败现场保留，不自动重启、切换服务选择或创建上游PR。完整任务精度按用户暂缓。

报告两arm的会话均值、A1→B1/A2→B2配对、A首尾漂移、B重复差异，以及全部逐轮数据。
ABBA只平衡近似线性时间漂移，无法排除非线性变化；每arm只有两独立会话，轮次不作独立样本，不声称统计显著。
与EXP20历史结果分别报告，最后综合性能和延迟决定工作基线，再继续4K小M投影优化。
''')
    step('Record fixed ABBA hypothesis and protocol before creating worktrees',order=ORDER,commits=COMMITS,preparation_is_CPU_only=True)
    for arm,commits in COMMITS.items():
        (WT/arm).mkdir()
        for name,commit in commits.items():git(ROOT/'worktrees/EXP-0020'/name,'worktree','add','--detach',str(WT/arm/name),commit)
    for p in STAGING.glob('*.py'):shutil.copy2(p,EXP/p.name)
    for name in ['parent_exp17','parent_exp20']:shutil.copytree(STAGING/name,EXP/name)
    for p in (EXP/'parent_exp17').glob('*.py'):assert sha(p)==sha(REPO/'experiments/EXP-0017'/p.name)
    assert sha(EXP/'parent_exp20/run_split_e2e.py')==sha(REPO/'experiments/EXP-0020/run_split_e2e.py')
    write(EXP/'candidate_commits.json',COMMITS['B']);write(EXP/'arms.json',dict(commits=COMMITS,order=ORDER))
    write(EXP/'daily_window_at_setup.json',daily)
    before={p.name:sha(p) for p in EXP.glob('*.py')}
    subprocess.run(['python3',str(EXP/'build_scripts.py')],check=True)
    assert before=={p.name:sha(p) for p in EXP.glob('*.py')}
    subprocess.run(['python3','-m','py_compile',*[str(p) for p in EXP.glob('*.py')]],check=True)
    results=checks();write(EXP/'route_cpu_checks.json',dict(status='passed',checks=results))
    write(EXP/'script_hashes.json',{p.name:sha(p) for p in EXP.glob('*.py')})
    write(EXP/'source_provenance.json',{str(p.relative_to(EXP)):sha(p) for d in ['parent_exp17','parent_exp20'] for p in (EXP/d).glob('*.py')})
    for arm in COMMITS:verify_sources(arm)
    step('Complete CPU preparation: both actual service routes plus negative checks; await16K review before GPU launch',checks=len(results))
    write(EXP/'STATUS.json',dict(stage='cpu_prepared_waiting_for_16k_review',GPU_started=False,checks=len(results)))
    pub=publish(['experiments/EXP-0021'],'experiments: prepare fixed 4K ABBA for prefill page broadcast')
    write(ART/'preparation.json',dict(status='prepared',checks=len(results),publication=pub,GPU_started=False))
    print(json.dumps(dict(status='cpu_prepared',checks=len(results),publication=pub)),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        if EXP.exists():step('CPU preparation failed; preserve partial setup',error=traceback.format_exc())
        write(ROOT/'artifacts/EXP-0021-preparation-error.json',dict(error=traceback.format_exc()));raise
