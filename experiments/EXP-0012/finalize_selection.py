"""Select the fully measured GQA stage1 performance baseline; no GPU work."""
from pathlib import Path
import csv
import datetime
import fcntl
import hashlib
import json
import shutil
import subprocess
import sys

ROOT=Path('/workspace/LeanScale')
REPO=ROOT/'MiniCPM-Infer-Sys-Opt'
EXP=REPO/'experiments/EXP-0012'
sys.path.insert(0,str(EXP))
from finish_split import verify_rows

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def rows(p):
    with p.open() as f:return list(csv.DictReader(f))
def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args],text=True,stderr=subprocess.STDOUT,timeout=120).strip()

def main():
    with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert not git('status','--porcelain'),'Investigate other changes before publication'
        shutil.copy2(__file__,EXP/'finalize_selection.py')
        commits=read(EXP/'candidate_commits.json')
        report=[]
        for case,shape in [('4k',(4096,1024,64,256)),('16k',(16384,1024,64,128))]:
            run=Path(read(ROOT/f'artifacts/EXP-0012-latest-{case}.json')['run_directory'])
            state=read(run/'status.json')
            assert state['commits']==commits
            if case=='4k':
                rec=read(run/'postprocessing_recovery.json')
                assert rec['status']=='complete_case_revalidated_offline'
                assert all(sha(run/rel)==v for rel,v in rec['source_hashes_before_and_after'].items())
                assert read(ROOT/'artifacts/EXP-0012-4k-recovery-publish.json')['remote_verified']
                accounting=read(run/'postprocessing_recovery_schedule.json')
            else:
                assert state['stage']=='performance_case_complete'
                h=read(run/'completion_handler_status.json')
                assert h['complete_results'] and h['status']=='pushed' and h['remote_verified']
                accounting=read(run/'scheduler_comparison.json')
            f=run/'candidate_benchmark/benchmark_results'
            raw_file=next(f.glob('raw_runs_*.csv'));summary_file=next(f.glob('summary_*.csv'))
            raw=rows(raw_file);summary=rows(summary_file);verify_rows(raw,summary,{shape})
            old=read(run/'exp6_reference.json')['summary'][0];new=summary[0]
            metrics={k:{'exp6':float(old[k]),'gqa_stage1':float(new[k]),'change_percent':(float(new[k])/float(old[k])-1)*100}
                     for k in ['Total tok/s','Output tok/s','Mean TTFT (ms)','Mean TPOT (ms)','Mean ITL (ms)']}
            comp=read(run/'comparison_exp6.json')[0]
            assert comp['throughput_change_percent']>0
            assert metrics['Mean TTFT (ms)']['change_percent']<=0
            assert metrics['Mean TPOT (ms)']['change_percent']<=0
            report.append({'case':case,'run_name':run.name,'metrics':metrics,'comparison_exp6':comp,
                           'comparison_original':read(run/'comparison.json')[0],'accounting':accounting,
                           'raw_sha256':sha(raw_file),'summary_sha256':sha(summary_file)})
        previous=read(ROOT/'artifacts/active_inference_plan.json')
        assert previous['selected_serving_commits']==read(EXP/'parent_commits.json')
        selection={'selected_experiment':'EXP-0012','selected_serving_commits':commits,
                   'previous_selection':previous,'selected_at':datetime.datetime.now().astimezone().isoformat(),
                   'basis':'Both complete official cases improve throughput and mean latency against EXP-0006 historical sessions.',
                   'scope':'Performance experiment baseline; full task accuracy deferred; not a claim of contest readiness.',
                   'case_results':report,'scheduler_candidate':'rejected_by_user','next_direction':'GQA paged KV access/head layout, then prefill tuning'}
        write(EXP/'selection.json',selection)
        text=['# GQA 单级流水端到端结果与版本选择','',
              '2026-10-21：将 EXP-0012 作为后续性能实验基线。保持原调度、原有 RoPE+KV 融合及完整 KV，仅将受支持的 GQA decode 从 2 stages 改为 1 stage。',
              '4K、16K 分别在独立服务会话完成官方四轮并弃首轮；对照 EXP-0006 是历史会话，不是交错 A/B。完整任务精度按用户要求暂缓，不宣称已满足赛事精度要求。','',
              '| 负载 | EXP-0006 tok/s | GQA tok/s | 吞吐变化 | TTFT ms：前→后 | TPOT/ITL ms：前→后 | 候选 CV |',
              '| --- | ---: | ---: | ---: | --- | --- | ---: |']
        for d in report:
            m=d['metrics'];c=d['comparison_exp6']
            text.append(f"| {d['case']} | {m['Total tok/s']['exp6']:.2f} | {m['Total tok/s']['gqa_stage1']:.2f} | {m['Total tok/s']['change_percent']:+.2f}% | {m['Mean TTFT (ms)']['exp6']:.2f} → {m['Mean TTFT (ms)']['gqa_stage1']:.2f} | {m['Mean TPOT (ms)']['exp6']:.2f} → {m['Mean TPOT (ms)']['gqa_stage1']:.2f} | {c['candidate_cv_percent']:.4f}% |")
        text+=['','## 优化前后与收益边界','',
               '原实现已经包含分组 Q head 复用 KV、QK/softmax/PV 融合、split-K decode 与归约；本轮没有新增 head 并行，也没有新融合。',
               'EXP-0011 扫描 12 组 launch 配置，筛出 4 warps / 1 stage / 16 segments。原 4 warps / 2 stages / 16 segments 保留为对照，默认 FlagGems 参数不改，由支持该路径的引擎入口显式传参。',
               '代表点单算子时间减少约 13.8%～20.8%；编译元数据 n_regs 111→96、shared 20480 不变。寄存器减少是观察结果，不能据此断言动态 occupancy 或解释全部收益。',
               '实际服务验证 42 层 GQA stage1、42 层 RoPE+KV、35 张 FULL_DECODE_ONLY 图及隔离缓存编译元数据。微基准数值、Graph、不同 dtype/layout、长地址验证见 EXP-0011。',
               '16K 的整模收益较小，与现有 prefill 开销和 KV 容量约束的观察一致；尚未进行因果隔离，不能将吞吐差全部归因于某项硬件因素。','',
               '## 长文本仍有容量约束','']
        d=report[1];b=d['accounting']['reference'];a=d['accounting']['candidate']
        text += [f"16K 全四轮抢占计数 {b['counter_delta']['num_preemptions_total']:.0f}→{a['counter_delta']['num_preemptions_total']:.0f}；采样最大运行请求数 {b['peak_running_sampled']}→{a['peak_running_sampled']}，最大等待请求数 {b['peak_waiting_sampled']}→{a['peak_waiting_sampled']}，KV 使用率峰值 {b['peak_cache_usage_percent_sampled']}%→{a['peak_cache_usage_percent_sampled']}%。",
                 f"全部四轮平均排队时间 {b['mean_queue_seconds_all_four_rounds']:.2f}→{a['mean_queue_seconds_all_four_rounds']:.2f} 秒。计数包含弃首轮；吞吐和延迟表弃首轮，两者口径不同。采样峰值不是精确连续峰值。",
                 '此次 GQA 加速没有消除 KV 容量瓶颈；劣化调度候选保持停用。KV 压缩留作独立实验，需完整 KV 对照、官方任务及触发压缩的长文本精度验证，并计入评分、搬移与回收成本。','',
                 '## 可复现性与失败记录','',
                 '4K 四轮数据完整，但后处理继承了已撤销调度的专用日志要求而报错。原 status、CSV、日志和失败归档均未改写；离线复核证据独立追加，未重跑或拼接任何轮次。16K 使用修正后的原调度普通计数分析，正常完成并归档。',
                 '- [4K 完整结果](../experiments/EXP-0012/RESULTS_4096.md)',
                 '- [16K 完整结果](../experiments/EXP-0012/RESULTS_16384.md)',
                 '- [版本选择和原始结果哈希](../experiments/EXP-0012/selection.json)',
                 '- [GQA 参数筛选](gqa_stage1_screen_20261021.md)','',
                 f"插件提交：`{commits['vllm-plugin-FL']}`；FlagGems 提交：`{commits['FlagGems']}`。",
                 '后续优先验证分页 KV 寻址/读取布局与 8-head 专用布局，再测试 prefill tile/warps/stages。每项独立记录、串行测量，数值/Graph 检查通过且微基准收益稳定后才进入整模测评。']
        (REPO/'docs/gqa_stage1_e2e_20261021.md').write_text('\n'.join(text)+'\n')
        with (EXP/'README.md').open('a') as f:f.write('\n4K 与 16K 均完成，EXP-0012 已选为后续性能实验基线：[汇总与边界](../../docs/gqa_stage1_e2e_20261021.md)。完整任务精度仍暂缓。\n')
        p=EXP/'PR_DRAFT.md';p.write_text(p.read_text().replace('16K待完成。','16K为1973.96 tok/s，相对EXP-0006提升3.28%，CV0.054%。'))
        (EXP/'STATUS.md').write_text('# EXP-0012 状态\n\n两组官方四轮均已完成并推送，已选为后续性能实验基线。[汇总](../../docs/gqa_stage1_e2e_20261021.md)。4K后处理失败与离线复核记录保留；完整任务精度暂缓，维持Draft。\n')
        with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps({'time':selection['selected_at'],'action':'select_gqa_stage1_performance_baseline','reason':selection['basis'],'full_task_accuracy':'deferred'})+'\n')
        paths=['docs/gqa_stage1_e2e_20261021.md']+['experiments/EXP-0012/'+n for n in ['selection.json','finalize_selection.py','README.md','PR_DRAFT.md','STATUS.md','steps.jsonl']]
        git('add','--',*paths);git('diff','--cached','--check')
        git('commit','--only','-m','experiments: select measured GQA stage1 performance baseline','--',*paths)
        head=git('rev-parse','HEAD');git('push','origin','main')
        assert git('ls-remote',git('remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
        active={'selected_serving_commits':commits,'selected_experiment':'EXP-0012','previous_serving_commits':previous['selected_serving_commits'],
                'scheduler_candidate':'rejected_by_user','current_experiment':None,'status':'Official 4K and 16K complete; selected performance baseline; full task accuracy deferred',
                'selection_document_commit':head,'next_direction':selection['next_direction']}
        write(ROOT/'artifacts/active_inference_plan.json',active)
        write(ROOT/'artifacts/EXP-0012-selection-publish.json',{'commit':head,'remote_verified':True,'selected':active})
        print(json.dumps({'commit':head,'remote_verified':True,'selected_commits':commits,'results':[{k:d[k] for k in ['case','metrics','comparison_original']} for d in report]},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
