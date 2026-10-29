"""Independently audit EXP26; preregister production validation without running it."""
from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,statistics,subprocess,sys
ROOT=Path('/workspace/LeanScale');DOC=ROOT/'MiniCPM-Infer-Sys-Opt';EXP=DOC/'experiments/EXP-0026'
ART=ROOT/'artifacts/EXP-0026';RUN=ART/'20261029_190815_grid_diagnostic';REVIEW=EXP/'review_20261029'
NEXT=DOC/'experiments/EXP-0027'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def main():
    assert not REVIEW.exists() and not NEXT.exists()
    s=read(RUN/'status.json');assert s['stage']=='complete' and datetime.fromisoformat(s['finished_at'])<datetime.fromisoformat(s['deadline'])
    assert read(RUN/'publication.json')['remote_verified']
    plan=read(EXP/'plan.json');assert sha(EXP/'sampling_grid_experiment.py')==plan['script_sha256']
    for p,h in read(EXP/'source_hashes.json').items():assert sha(Path(p))==h
    checks=[json.loads(l) for l in (RUN/'checks.jsonl').read_text().splitlines()]
    timings=[json.loads(l) for l in (RUN/'timings.jsonl').read_text().splitlines()]
    assert len(checks)==s['checks']==140 and len(timings)==s['timing_points']==27
    for r in checks:
        keys=[k for k in ['exact_masked_logits','masking_rng_unchanged','exact_tokens_and_rng','exact'] if k in r]
        assert keys and all(r[k] is True for k in keys),r
        if 'baseline_masked' in r:assert r['baseline_masked']==r['candidate_masked']
    assert any(r['label']=='actual_padded_row_stride' for r in checks)
    assert any(r['label']=='graph_k_p_input_update' for r in checks)
    assert sum(r['label']=='graph_input_and_p_update' for r in checks)==27
    assert sum('exact_tokens_and_rng' in r for r in checks)==16
    expected={(seed,b,scale,g) for seed in plan['seeds'] for b in [32,64] for scale in plan['scales'] for g in ([32] if b==32 else [32,64])}
    assert {(r['seed'],r['B'],r['scale'],r['grid']) for r in timings}==expected
    for r in timings:
        a,b,c=r['baseline_before_us'],r['candidate_us'],r['baseline_after_us']
        assert len(a)==len(b)==len(c)==7 and all(x>0 for x in a+b+c)
        old=(statistics.median(a)+statistics.median(c))/2
        computed={'baseline_median_us':old,'candidate_median_us':statistics.median(b),
            'change_percent':(statistics.median(b)/old-1)*100,
            'drift_percent':(statistics.median(c)/statistics.median(a)-1)*100,
            'candidate_cv_percent':statistics.stdev(b)/statistics.mean(b)*100}
        for k,v in computed.items():assert abs(v-r[k])<1e-8,(k,v,r[k])
        assert r['baseline_scratch_bytes']==16*130560*4 and r['candidate_scratch_bytes']==r['grid']*130560*4
    result=read(RUN/'results.json');summary=[]
    for grid in [32,64]:
        target=[r for r in timings if r['B']==64 and r['grid']==grid]
        control=[r for r in timings if r['B']==32 and r['grid']==32]
        assert len(target)==len(control)==9
        assert all(r['change_percent']<=-5 and abs(r['drift_percent'])<=3 for r in target)
        assert all(r['change_percent']<=3 and abs(r['drift_percent'])<=3 for r in control)
        actual=next(r for r in result['candidates'] if r['B64_grid']==grid)
        assert actual['passed'] and abs(actual['mean_change_percent']-statistics.mean(r['change_percent'] for r in target))<1e-9
        summary.append(dict(B=64,grid=grid,mean_change_percent=actual['mean_change_percent'],
            min_change_percent=min(r['change_percent'] for r in target),max_change_percent=max(r['change_percent'] for r in target),
            max_abs_drift_percent=max(abs(r['drift_percent']) for r in target+control),
            scratch_payload_bytes=grid*130560*4,nominal_increment_over16_bytes=(grid-16)*130560*4))
    control=[r for r in timings if r['B']==32]
    summary.append(dict(B=32,grid=32,mean_change_percent=statistics.mean(r['change_percent'] for r in control),
        min_change_percent=min(r['change_percent'] for r in control),max_change_percent=max(r['change_percent'] for r in control)))
    evidence=read(RUN/'kernel_evidence.json');assert evidence
    for r in evidence:
        p=RUN/r['path'];assert p.resolve().is_relative_to(RUN)
        assert sha(p)==r['sha256'] and sha(p.with_suffix('.ttir'))==r['ttir_sha256']
        meta=read(p);assert meta['target']['backend']=='corex' and meta['num_warps']==r['num_warps'] and meta['num_stages']==r['num_stages']
        assert '_topk_topp_kernel' in p.with_suffix('.ttir').read_text()
    archive=EXP/'artifacts'/RUN.name;manifest=read(archive/'source_hashes.json')
    for name,h in manifest.items():assert sha(RUN/name)==sha(archive/name)==h
    assert read(ROOT/'artifacts/active_inference_plan.json')['selected_experiment']=='EXP-0020'
    audit=dict(status='passed',checked_at=datetime.now().astimezone().isoformat(),run=RUN.name,
        checks=140,timing_points=27,summary=summary,actual_kernel_variants=len(evidence),
        exact_mask_and_graph=True,random_tokens_and_rng_state_preserved=True,
        selected_diagnostic_grid=64,selected_serving_experiment='EXP-0020',end_to_end_gain='not_measured',
        timing_scope='restore copy + unchanged original masking kernel; CUDA Graph; random tensors, not real model replay',
        memory_scope='payload difference assumes replacing16-row scratch; microbenchmark holds separate buffers for compared grids. Not measured production incremental allocation or KV loss.',
        source_publication=read(RUN/'publication.json'),checks_sha256=sha(RUN/'checks.jsonl'),timings_sha256=sha(RUN/'timings.jsonl'))
    REVIEW.mkdir();write(REVIEW/'audit.json',audit);shutil.copy2(__file__,REVIEW/'audit_grid.py')
    report=DOC/'docs/topp_row_parallel_20261029.md'
    lines=['# 4K top-p 行并行度：原内核调度诊断','',
        '选择 grid64 进入引擎验证，当前服务仍为 EXP20。B64 随机 logits 的截断阶段平均耗时下降 66.73%，不是整模吞吐提升；生产请求尚未测量。',
        '', '## 为什么转向这里','',
        'EXP25 的 MLP 权重布局 ABBA 无端到端收益，已停止推广。EXP18 的 NONE/large_decode 诊断中，_topk_topp_kernel 每步约 6320.50 us，占该窗口可见 kernel 总工作量 10.97%；全部采样为 12.70%。FULL decode 的图内 kernel 不可见，所以不能将该比例套到生产 FULL 整程。',
        '', '官方 benchmark 未显式覆盖 temperature/top-p；模型 generation_config 为 temperature=1、top_p=.95、do_sample=true。词表为 130560。PlatformFL 为 OOT，native sampler 在 B>=8 时进入 Triton top-k/top-p 截断；B<8 的 PyTorch 分支不是本轮对象。真实服务参数还会在下一阶段记录。',
        '', '## 前后实现','',
        '原 wrapper 启动 min(SM数,B) 个 program，本机为16；B64 时每个 program 循环处理4行。新诊断仅把 grid 改为32或64，同时扩容每个 program 的 scratch。调用同一个原始 JIT 内核，8192/4096 tile、默认 warps/stages、查找表、算术和筛选算法全部保持；不修改 top-p，不改成贪心采样，不替换 RNG。',
        '', '更多 program 提供更多可调度工作，是否提高实际 occupancy 需要硬件计数器，当前只报告耗时，没有测 occupancy。',
        '', '## 全部诊断结果','',
        '| B | program 数 | 平均耗时变化 | 各点范围 |','|---:|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['B']} | {r['grid']} | {r['mean_change_percent']:+.3f}% | {r['min_change_percent']:+.3f}% ～ {r['max_change_percent']:+.3f}% |")
    lines+=['', '3种子、3种logits分布(scale .5/1/3)，27个旧/新/旧夹测点完整保存；B8/16 的 grid 未变，只作控制检查。7样本×4次Graph重放，每次都包含同样的输入恢复copy，避免反复截断已mask数据；不把结果当作单纯kernel设备时间。最大绝对夹测漂移低于1%。',
        '', '140项检查通过：masked logits逐元素相同、-inf掩码相同；Graph更新input/p/k后与eager一致；uniform/tied BF16/sparse -inf/peaked/行padding；同RNG状态及per-request generator下随机采样token与后续RNG状态一致。没有放宽容差。此结论相对当前生产实现，未声称其算法被穷尽证明。',
        '', '64行scratch有效载荷为33,423,360字节，原16行为8,355,840字节；若在生产中扩容同一cache而非双份保留，名义差额25,067,520字节（23.90625 MiB）。微基准为对照持有独立buffer，尚不能称真实引擎增量或KV损失已测得。必须处理缓存扩容、旧Graph引用、reset和fallback，并测服务KV容量。',
        '', '## 下一阶段 EXP27','',
        '先固定 grid64 候选与适用范围，再验证真实模型logits和top-p分支，补17/30/31/33/48/63等中间batch、缓存生命周期/Graph/fallback。引擎接入保持默认关闭，GPU和版本不支持时原路径回退；采样概率阈值、随机数生成和token选择定义保持。实际服务shadow检查与正式性能分开，禁止把额外比较开销混入官方计时。通过接线检查后再按官方4K四轮弃首测试。',
        '', '每个后续阶段刷新日期和预算，受23:35计算/23:50保存限制；当前EXP26窗口已结束，不能继续套用。完整模型任务精度按用户暂缓，KV压缩仍需独立精度。',
        '', '- [独立审核](../experiments/EXP-0026/review_20261029/audit.json)',
        '- [预注册协议](../experiments/EXP-0026/README.md)',
        '- [原始计时](../experiments/EXP-0026/artifacts/'+RUN.name+'/timings.jsonl)',
        '- [全部检查](../experiments/EXP-0026/artifacts/'+RUN.name+'/checks.jsonl)']
    report.write_text('\n'.join(lines)+'\n')
    NEXT.mkdir()
    (NEXT/'README.md').write_text('''# EXP-0027：top-p 行并行度生产接入与官方4K验证

在修改生产代码前登记。EXP26 的原内核 grid64 随机 logits 诊断通过140项精确/Graph/RNG检查，B64截断阶段平均-66.73%，尚无真实模型或E2E收益。当前基线为EXP20，插件ce19f93eee4c73f5a26911554700457e879f891d、Gems de958ddfc4deeb43d6c94bb8652a3d4d3bc905fe。

候选默认关闭，仅天数BI-V150、已核验vLLM源码及原TOPP内核版本、float32、V130560、单GPU路径启用；保持原算法/8192-4096 tile/warps-stage/RNG，不修改官方请求参数。先限制top-p-only且B17..64，其他batch/dtype/stride/设备/参数回退原实现；若扩展范围须额外记录和检查。B8/16原grid不变，B<8继续原PyTorch分支。

新launcher复用原JIT对象及原scratch缓存；扩容最多64行，不能双份常驻，不可越界。必须验证扩容前后旧Graph重放、不同B和行stride、缓存reset/设备隔离、默认关闭、安装幂等、上下游函数引用真正换成新入口及fallback结果。补B17/30/31/33/48/63等边界和小B回退。引擎源代码个人fork独立分支leanscale/iluvatar-topp-row-parallel，从EXP20创建，禁止继承未采用的MLP布局。

先真实服务diagnostic shadow：记录真实B/V/dtype/stride/p/k分支，有限次原输入上旧/新mask逐元素比较和RNG状态；采样参数CPU读取/比较仅诊断开关启用。不要把完整logits或请求原文推到远端。检查通过后关闭shadow，独立服务与缓存进行官方[4096,1024,64,256]四轮弃首，记录吞吐/延迟/失败、KV容量、实际路径及source hash。计时不包含shadow，也不把micro当收益。

每阶段重新分配完整预算，今日2026-10-29 23:35停止计算、23:50保存、次日00:00关机；不足则完成文档和源码，不能用过期窗口启动。完整官方结果前selected保持EXP20。整模任务精度暂缓，采样数值与RNG/tie检查不省略。只推个人fork，不创建上游PR；未完成验证的候选保持Draft状态。
''')
    write(NEXT/'STATUS.json',dict(stage='hypothesis_registered_not_implemented',GPU_started=False,
        parent_experiment='EXP-0020',source_diagnostic='EXP-0026',next='Implement bounded launcher and validate exact path before benchmark'))
    with (DOC/'docs/optimization_journal.md').open('a') as f:f.write('\n\n## 2026-10-29：EXP26 top-p grid 诊断通过\n\n27计时点、140项精确/Graph/RNG检查通过；B64 grid32/64平均耗时分别-45.72%/-66.73%。选择grid64进入EXP27生产验证，服务仍EXP20；名义scratch差额23.90625MiB需在引擎测量。详见[top-p行并行报告](topp_row_parallel_20261029.md)。\n')
    with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(),action='Independent grid audit passed; register EXP27 before implementation',summary=summary))+'\n')
    sys.path.insert(0,str(DOC/'experiments/EXP-0025'));from common import publish
    pub=publish(['experiments/EXP-0026/review_20261029','experiments/EXP-0026/steps.jsonl','experiments/EXP-0027','docs/topp_row_parallel_20261029.md','docs/optimization_journal.md'],
        'experiments: audit top-p grid speedup and register production validation')
    write(ART/'review_publication.json',pub)
    active=read(ROOT/'artifacts/active_inference_plan.json');active.update(current_experiment='EXP-0027',
        status='Top-p grid diagnostic audited; production validation preregistered, not implemented',
        next_direction='Implement bounded top-p grid launcher from EXP20, then real-service shadow checks and official4K',review_commit=pub['commit'])
    write(ROOT/'artifacts/active_inference_plan.json',active)
    print(json.dumps(dict(publication=pub,audit=audit)),flush=True)
if __name__=='__main__':main()
