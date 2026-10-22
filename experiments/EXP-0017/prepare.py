"""Record the hypothesis, freeze both revisions, test gates, and publish before launch."""
import datetime
from pathlib import Path
import shutil
import subprocess
import tempfile
from common import *

STAGING=Path(__file__).resolve().parent
def checks():
    from route_checks import verify_route
    results=[]
    refs={'A':ROOT/'artifacts/EXP-0014/20261022_120817_official_4k',
          'B':ROOT/'artifacts/EXP-0016/20261022_145853_official_4k'}
    with tempfile.TemporaryDirectory(prefix='exp17_checks_') as t:
        t=Path(t)
        for arm,ref in refs.items():
            verify_route(ref,arm,t/(arm+'_actual_route.json'))
            results.append(dict(name=arm+'_all_actual_service_kernels',passed=True))
            f=t/arm;folder=f/'candidate_benchmark';folder.mkdir(parents=True)
            log=(ref/'candidate_benchmark/server.log').read_text();(folder/'server.log').write_text(log)
            graph=next(p for p in (ref/'candidate_cache').rglob('computation_graph.py') if 'torch.ops.vllm.fl_rope_kv_dispatch' in p.read_text())
            g=f/'candidate_cache/graph/computation_graph.py';g.parent.mkdir(parents=True);shutil.copy2(graph,g)
            metas={}
            for kind,shape in [('decode','8x128x!tt.ptr<'),('prefill','64x128x!tt.ptr<')]:
                src=next(p for p in (ref/'triton_cache').glob('*/_attention.json') if shape in p.with_suffix('.ttir').read_text())
                dst=f/'triton_cache'/kind/src.name;dst.parent.mkdir(parents=True)
                shutil.copy2(src,dst);shutil.copy2(src.with_suffix('.ttir'),dst.with_suffix('.ttir'));metas[kind]=dst
            def check(name,expected):
                try:verify_route(f,arm);ok=True
                except AssertionError:ok=False
                assert ok==expected,(arm,name)
                results.append(dict(name=arm+'_'+name,passed=True))
            check('fixture_positive',True)
            for kind,p in metas.items():
                original=p.read_text();m=read(p)
                m['num_stages']=3;write(p,m);check(kind+'_reject_wrong_stages',False);p.write_text(original)
                m=read(p);m['shared']+=2048;write(p,m);check(kind+'_reject_wrong_shared',False);p.write_text(original)
                p.unlink();check(kind+'_reject_missing',False);p.write_text(original)
                ir=p.with_suffix('.ttir');original=ir.read_text()
                changed=original.replace('8x128x!tt.ptr<','16x128x!tt.ptr<') if kind=='decode' else original.replace('64x128x!tt.ptr<','32x128x!tt.ptr<')
                assert changed!=original;ir.write_text(changed);check(kind+'_reject_wrong_rows',False);ir.write_text(original)
            for name,bad in [('wrong_decode_log',log.replace('decode_rows=8','decode_rows=16',1)),
                             ('withdrawn_scheduler',log+'\nKVReservationAsyncScheduler\n'),
                             ('missing_layer',log.replace('LeanScale GQA active:','missing:',1))]:
                (folder/'server.log').write_text(bad);check(name,False)
            (folder/'server.log').write_text(log)
    return results

def main():
    assert not EXP.exists() and not WT.exists() and not ART.exists(),'Do not overwrite existing experiment'
    assert not git(REPO,'status','--porcelain')
    policy=read(ROOT/'artifacts/run_window.json')
    assert (datetime.datetime.fromisoformat(policy['compute_stop_at'])-now()).total_seconds()>4*55*60+15*60
    audit=[read(REPO/f'experiments/EXP-0016/audited_{case}.json') for case in ['4k','16k']]
    assert all(a['status']=='complete_case_audited' and a['candidate_commits']==COMMITS['B'] for a in audit)
    EXP.mkdir();WT.mkdir();ART.mkdir()
    (EXP/'README.md').write_text('''# EXP-0017：4K 完整官方协议 ABBA 复测

假设先于修改记录：EXP-0016 stage1 prefill 的官方4K相对EXP-0014提升4.33%，与三轮CV（3.16%和4.43%）量级接近。
16K完整四轮验证提升6.08%，但4K因历史服务会话的预热/时间漂移需要同期复核。CV不是置信区间。

固定顺序 A1→B1→B2→A2。A为EXP-0014（prefill stage2），B为EXP-0016（prefill stage1）。
每个会话独立服务及缓存，严格官方4K[4096,1024,64,256]、四轮舍弃首轮。
保留原benchmark和serve参数、RoPEKV融合、分页decode、原调度及完整KV；不改推理或算子代码。
独立detached工作树分别固定A/B提交；每次启动前后核查提交、工作区、脚本与原benchmark哈希。
CPU门槛验证两版本实际服务编译IR，并拒绝错误stage、shape、shared、日志和已拒调度。
正式每次测评前后均检查42层接入、35个FULL_DECODE_ONLY捕获及真实8行decode/64行prefill编译证据。

每组四轮数据独立完整，不合并EXP-0016数据或中断轮次。预先固定全部ABBA，不按性能好坏提前截断。
评估A/B各两个会话均值，同时报告A1→B1、A2→B2对照、A首尾漂移和B重复差异。
ABBA可平衡近似线性时间漂移，无法排除非线性变化；每版本仅两个会话，不宣称统计显著，不把轮次当独立实验。
配对复核后才确定性能版本，随后进入重点4K热点优化。全模型精度按用户要求暂缓。

每个case起跑预算55分钟，启动整组需230分钟。单个case上限55分钟且不超过23:35绝对计算截止。
共享GPU锁覆盖全部ABBA；独立服务停止后才进入下一组。结果逐组自动保存推送；发布失败保留本地，不为此重测。
完整ABBA或失败都不自动重启、不自动切换最佳版本、不创建上游PR。
''')
    step('Record hypothesis, fixed ABBA order, interpretation and shutdown budget before preparation',order=ORDER,commits=COMMITS)
    for arm,commits in COMMITS.items():
        (WT/arm).mkdir()
        for name,commit in commits.items():
            git(ROOT/'worktrees/EXP-0016'/name,'worktree','add','--detach',str(WT/arm/name),commit)
    for f in STAGING.glob('*.py'):shutil.copy2(f,EXP/f.name)
    write(EXP/'candidate_commits.json',COMMITS['B'])
    write(EXP/'arms.json',dict(commits=COMMITS,order=ORDER))
    write(EXP/'run_window_at_setup.json',policy)
    assert sha(EXP/'frozen_runtime.py')=='0c28df633c175694967f390325f0d34be304de52c95244e064c9077556f2c466'
    assert sha(EXP/'analyze_schedule.py')=='574042b1d7d97454691b3608f2fe0073cc760a6c0314a1cd1315bc95a769ea9f'
    subprocess.run(['python3',str(EXP/'build_route.py')],check=True)
    assert sha(EXP/'route_checks.py')==sha(STAGING/'route_checks.py')
    subprocess.run(['python3','-m','py_compile',*[str(f) for f in EXP.glob('*.py')]],check=True)
    results=checks();write(EXP/'route_cpu_checks.json',dict(status='passed',checks=results))
    write(EXP/'script_hashes.json',{f.name:sha(f) for f in EXP.glob('*.py')})
    for arm in COMMITS:verify_sources(arm)
    step('Freeze independent A/B worktrees and pass actual-service plus negative CPU gate checks',checks=len(results))
    lines=['# GQA prefill stage1 官方端到端结果','','2026-10-22；EXP-0016。两组均完成官方四轮，弃首轮，全部零失败。','',
           '| 负载 | 原始基线 tok/s | EXP14 tok/s | stage1 tok/s | 本轮增量 | 累计增量 |','|---|---:|---:|---:|---:|---:|']
    for a in audit:
        m=a['metrics']['Total tok/s'];o=a['comparison_original'];lines.append(f"| {a['case']} | {o['baseline_total_tokens_s']:.2f} | {m['exp14']:.2f} | {m['prefill_stage1']:.2f} | {m['change_percent']:+.2f}% | {o['throughput_change_percent']:+.2f}% |")
    lines += ['','4K TTFT下降5.79%，TPOT下降4.03%；16K TTFT下降6.34%，TPOT下降5.56%。',
              '42层实际接入、FULL_DECODE_ONLY图及decode/prefill编译IR检查通过。数值/Graph检查已在EXP15完成，完整MATH500精度暂缓。',
              '16K全四轮平均排队240.58→225.38秒，抢占仍为16次，KV采样峰值99.8%；容量压力未解决。',
              '4K保留轮3358.63/3495.20/3575.80 tok/s，CV3.16%；父版本CV4.43%。历史会话的小幅差异不能全部归因于参数。',
              '因此保留EXP14为已选版本，启动[EXP17完整4K ABBA复测](../experiments/EXP-0017/README.md)，之后再作选择和4K热点专项。',
              '[4K完整审核](../experiments/EXP-0016/audited_4k.json) / [16K完整审核](../experiments/EXP-0016/audited_16k.json)。']
    (REPO/'docs/gqa_prefill_e2e_20261022.md').write_text('\n'.join(lines)+'\n')
    with (REPO/'docs/optimization_journal.md').open('a') as f:f.write('\n\n### 2026-10-22：EXP16完整审核与EXP17配对复核\n\n官方4K3476.54、16K2314.37 total tok/s，相对EXP14分别+4.33%/+6.08%。保留全部原始四轮、实际接入与数值验证证据。4K历史会话波动接近收益，先固定ABBA完整官方复测；不提前切换已选版本。详见[gqa_prefill_e2e_20261022.md](gqa_prefill_e2e_20261022.md)。\n')
    draft=REPO/'experiments/EXP-0016/PR_DRAFT.md'
    s=draft.read_text().replace('真实42层接入和官方4K/16K四轮弃首轮测评待补。','真实42层接入与官方4K/16K四轮弃首轮测评已完成，吞吐分别3476.54/2314.37 tok/s，较EXP14观测+4.33%/+6.08%。4K增幅与历史波动接近，EXP17配对复核中。')
    draft.write_text(s)
    paths=['experiments/EXP-0017','docs/gqa_prefill_e2e_20261022.md','docs/optimization_journal.md','experiments/EXP-0016/PR_DRAFT.md']
    pub=publish(paths,'experiments: complete prefill E2E review and prepare 4K ABBA recheck')
    write(ART/'preparation.json',dict(status='prepared',checks=len(results),publication=pub))
    active=read(ROOT/'artifacts/active_inference_plan.json');assert active['selected_serving_commits']==COMMITS['A']
    active.update(current_experiment='EXP-0017',status='EXP16 full 4K/16K audited; paired 4K ABBA pending before selection',next_direction='Complete ABBA attribution recheck, then dedicated 4K hotspot optimization')
    write(ROOT/'artifacts/active_inference_plan.json',active)
    print(json.dumps(dict(status='prepared',checks=len(results),publication=pub)),flush=True)
if __name__=='__main__':main()
