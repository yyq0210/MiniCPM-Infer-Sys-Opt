"""Publish completed EXP18 audit and register the next hypothesis, CPU only."""
from datetime import datetime
import fcntl
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt'
ART=ROOT/'artifacts/EXP-0018';EXP=REPO/'experiments/EXP-0018'
SOURCE=ART/'review_20261023_1';DEST=EXP/'review_20261023'
NEXT=REPO/'experiments/EXP-0019'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True,timeout=120).strip()

assert not DEST.exists() and not NEXT.exists(),'Do not overwrite completed review or registered experiment'
assert read(SOURCE/'audit.json')['status']=='passed'
assert not git('status','--porcelain')
assert git('rev-parse','HEAD')=='a4fa36f3df04ca2db6447bd6999f4a17d0bcbfc4'
lock=(ROOT/'artifacts/docs-publish.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
try:
    shutil.copytree(SOURCE,DEST)
    shutil.copy2(ART/'audit_20261023.py',DEST/'audit.py')
    shutil.copy2(Path(__file__),DEST/'publish_review.py')
    shutil.copy2(ART/'operator_profile_4k_20261023.md',REPO/'docs/operator_profile_4k_20261023.md')
    names=read(SOURCE/'compile_name_candidates.json')
    assert all(all(name.startswith('## Call CompiledFxGraph ') for name in item['names']) for item in names)
    NEXT.mkdir()
    (NEXT/'README.md').write_text('''# EXP-0019：4K GQA prefill 分页读取与 K 加载布局

2026-10-23，先登记假设，再开发。EXP18 当前版本真实 4K mixed 窗口中 prefill 占可见 kernel 工作量45.20%；16步实际B10–17/M2048。详见[当前热点](../../docs/operator_profile_4k_20261023.md)。这不是整程占比或新的性能收益。

假设：当前页号广播与按[token,dim]加载K后转置仅用于decode。将两种访存方式分别应用到prefill，可能减少重复页号读取或改善加载布局；也可能增加转换、寄存器和shared开销。先比较四个固定配置：layout0原路径、layout1仅页号广播、layout2仅K加载方向、layout3组合。固定rows64/warps4/stage1/tile32，decode原配置不变，不重复EXP15参数网格。

直接基线与当前已选版本是EXP16：插件588a6147cd291e272324ab6c812024cd587c6287，Gems fd209df717648002ebf5ba4ec351424dcc997a53。独立工作树，新增实验接口默认prefill_layout=0，保持通用默认和服务版本；候选尚未实现/验证/接入。

由EXP18逐步metadata导出真实query长度与CPU context上界，用随机Q/K/V构造有来源的形状诊断，不冒充原请求实际tensor或设备length。固定选取initial两步纯prefill、mixed的37/38/51/52步，并保留B64合成mixed、16K及小batch decode回归。工作点先固定，不按结果挑点。

先FP32 oracle、BF16/FP16、随机物理页、NHD/HND、ragged/empty、页边界与Graph修改Q/K/V/length/BT；初筛胜者再扩大all-query和真实int64大地址验证。预热后原版/候选/原版夹测完整GQA，另列prefill-only与编译资源，保留全部原始采样和失败。至少一个主要prefill/mixed点耗时减少5%，其他主要点不退化超过3%，decode回归不超过5%，夹测基线漂移不超过3%；仅作为候选门槛，不等于端到端收益。

通过扩大检查后才改插件显式参数并验证实际42层接入，再测官方4K四轮弃首，16K回归。完整模型任务精度按用户要求暂缓；不混入调度、采样或KV压缩。

当前只登记CPU开发准备。EXP18三小时短窗口仅覆盖已完成的两组profiling，不自动继承给本实验。任何GPU启动必须另行记录当日有限预算、共享锁/实际空闲、源码与脚本哈希以及自动保存推送策略；今天实际关机时间仍未知。
''')
    run=ROOT/'artifacts/EXP-0018/20261023_132558_profile_4k'
    workloads=[]
    for window,steps in [('initial',None),('mixed',{37,38,51,52})]:
        meta=read(run/'summary/FULL_DECODE_ONLY'/window/'0/step_metadata.json')
        for key,m in meta.items():
            if (steps is None and m['phase']=='prefill') or (steps is not None and int(key) in steps):
                workloads.append(dict(name=f'exp18_{window}_step{key}',source_mode='FULL_DECODE_ONLY',source_window=window,
                    source_step=int(key),query_lengths=m['query_lengths'],context_lengths_cpu_upper_bound=m['context_lengths_cpu_upper_bound'],
                    phase=m['phase'],B=m['B'],M=m['M_unpadded']))
    assert len(workloads)==6
    write(NEXT/'observed_workloads.json',dict(source_run=run.name,source_commit='a4fa36f3df04ca2db6447bd6999f4a17d0bcbfc4',
          interpretation='Observed CPU shape vectors; future random tensor diagnostic is not replay of original request tensors or GPU lengths',cases=workloads))
    write(NEXT/'parent_commits.json',read(EXP/'candidate_commits.json'))
    write(NEXT/'STATUS.json',dict(stage='hypothesis_registered_cpu_preparation',gpu_work_started=False,updated_at=datetime.now().astimezone().isoformat()))
    (NEXT/'steps.jsonl').write_text(json.dumps(dict(time=datetime.now().astimezone().isoformat(),step='Register prefill memory-layout hypothesis and fixed profile-derived cases before modifying code'),ensure_ascii=False)+'\n')
    p=REPO/'docs/iluvatar_operator_optimization_backlog.md';s=p.read_text()
    old='**当前还没有完整 GPU profiler 的算子时间占比排名。**'
    assert old in s
    s=s.replace(old,'**2026-10-23更新：已完成当前选定版本EXP18的分阶段GPU profiling；最新结论见本文末尾及[新报告](operator_profile_4k_20261023.md)。以下2026-10-20建档内容保留为历史快照，不代表当前实验状态。**',1)
    s+='''\n\n## 2026-10-23：当前选定版本的4K热点与后续顺序\n\nEXP18两组各256请求全部成功，8个窗口各16步，原始trace/metadata/实际编译缓存已独立复核。FULL配置mixed（实际NONE执行）中prefill45.20%、gate/up23.61%、down12.23%；NONE/B64 decode主核65.78%、采样12.70%；NONE尾部B8–16的四类层内投影约44.54%。这些是各窗口可见kernel总工作量占比，不能合为整程占比。FULL decode Graph内部不可见，不能用外围采样占比给整模排名。\n\nOP05优先开展EXP19的prefill页号广播与K加载布局，固定已选参数；OP03/08/09/11后续针对小M及生产Graph padding研究真实GEMM路径。先前GEMM融合慢原型不直接恢复。当前选择仍EXP16；官方4K/16K、EXP17 ABBA均完成，调度EXP10已由用户拒绝并回退，KV压缩精度工作保持独立。详见[最新报告](operator_profile_4k_20261023.md)与[EXP19假设](../experiments/EXP-0019/README.md)。\n'''
    p.write_text(s)
    with (REPO/'docs/optimization_journal.md').open('a') as f:
        f.write('\n\n### 2026-10-23：EXP18完成、独立审核与EXP19假设\n\n两种模式各256/256成功、零失败，8个窗口各16步；14:01完成，自动归档a4fa36f。独立重算trace、128步长度/phase/padding、42层调用和128个GQA编译记录/模式，源码及原始证据哈希通过。GPU已释放。当前热点及解释见[报告](operator_profile_4k_20261023.md)。按实际mixed prefill热点登记EXP19分页读取/K布局假设，保留原参数与decode；尚无新增性能或接入结论。\n')
    with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(),step='Independent CPU audit passed: eight raw traces,128 steps,two complete256-request workloads,actual route and hashes; GPU released; register EXP19 hypothesis'),ensure_ascii=False)+'\n')
    paths=['experiments/EXP-0018/review_20261023','experiments/EXP-0018/steps.jsonl','experiments/EXP-0019',
           'docs/operator_profile_4k_20261023.md','docs/iluvatar_operator_optimization_backlog.md','docs/optimization_journal.md']
    for rel in paths:
        p=REPO/rel
        files=[p] if p.is_file() else [x for x in p.rglob('*') if x.is_file()]
        for f in files:
            assert f.stat().st_size<8*1024*1024
            assert not any(x in f.read_text(errors='replace') for x in ('PRIVATE'+' KEY','ssh.'+'flagos.io','lab'+'_inst_'))
    git('add','--',*paths);git('diff','--cached','--check')
    git('commit','--only','-m','docs: audit current 4K hotspots and register prefill memory experiment','--',*paths)
    commit=git('rev-parse','HEAD');git('push','origin','main');assert git('ls-remote','origin','refs/heads/main').split()[0]==commit
    write(ART/'review_publication.json',dict(status='pushed',commit=commit,remote_verified=True))
    active=read(ROOT/'artifacts/active_inference_plan.json');active.update(current_experiment='EXP-0019',
        status='EXP18 audited and published; EXP19 prefill memory-layout CPU preparation; serving remains EXP16',
        next_direction='Prefill page-table broadcast/K-load layout; then small-M projection diagnosis')
    write(ROOT/'artifacts/active_inference_plan.json',active)
    print(json.dumps(dict(status='published',commit=commit,remote_verified=True,next_experiment='EXP-0019')))
finally:lock.close()
