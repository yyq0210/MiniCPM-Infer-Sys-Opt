"""Record the complete coarse sweep and launch a frozen three-point refinement."""
from pathlib import Path
import ast
import datetime
import fcntl
import hashlib
import json
import re
import shutil
import subprocess
import time

ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt';EXP=REPO/'experiments/EXP-0015'
STAGING=Path(__file__).resolve().parent


def read(p):return json.loads(p.read_text())
def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args],text=True,stderr=subprocess.STDOUT,timeout=120).strip()


def main():
    prior=read(ROOT/'artifacts/EXP-0015-request.json');run=Path(prior['run_directory'])
    assert prior['stage']=='screen_complete' and prior['safe_to_resume_gpu']
    assert len(prior['trials'])==12 and all(t['status']=='passed' for t in prior['trials'])
    assert read(run/'publication.json')['status']=='pushed'
    summary=read(run/'screen_summary.json')
    winner,=[x for x in summary['rows'] if x['global_candidate_gate_passed']]
    assert winner['configuration']=='rows64_w4_s1'
    for rec in prior['trials']:
        data=read(run/(rec['configuration']+'.json'))
        assert data['status']=='passed' and len(data['checks'])==20 and len(data['timings'])==9
    latest=ROOT/'artifacts/EXP-0015-refined-request.json'
    assert not latest.exists(),'Inspect an existing refinement before retrying'
    deadline=datetime.datetime.fromisoformat(read(ROOT/'artifacts/run_window.json')['compute_stop_at'])
    assert (deadline-datetime.datetime.now().astimezone()).total_seconds()>600
    files=['build_refine.py','trial_refine.py','run_refine.py','launch_refine.py']
    with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert not git('status','--porcelain')
        for name in files:
            ast.parse((STAGING/name).read_text());shutil.copy2(STAGING/name,EXP/name)
        hashes={n:hashlib.sha256((EXP/n).read_bytes()).hexdigest() for n in files}
        subprocess.run(['python3',str(EXP/'build_refine.py')],check=True)
        assert hashes=={n:hashlib.sha256((EXP/n).read_bytes()).hexdigest() for n in files}
        evidence=dict(source_run=run.name,screen_publication=read(run/'publication.json'),
            source_summary_sha256=hashlib.sha256((run/'screen_summary.json').read_bytes()).hexdigest(),
            complete_configurations=12,checks=240,timing_points=108,winner=winner,
            next_hypothesis='At fixed rows64/warps4, compare winner stage1/tile32 against stage1/tile64 and stage3/tile32; one factor at a time.',
            selected_for_serving=False,script_hashes=hashes)
        (EXP/'refinement_plan.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+'\n')
        lines=['# GQA prefill 参数初筛与邻域复核','',
            '2026-10-22，EXP-0015。12组全部通过20项数值/Graph检查，共240项；每组9个计时点，共108个。基线是已完成官方4K/16K的EXP-0014。',
            '唯一通过全局候选门槛的是rows64/warps4/stages1，tile32保持。尚未接入模型；表中是完整GQA调用耗时变化，不是整模吞吐收益。','',
            '| 配置 | pure4K | pure16K | mixed4K两顺序 | mixed16K两顺序 | decode三点 |',
            '|---|---:|---:|---|---|---|']
        fmt=lambda xs:' / '.join(f'{x:+.3f}%' for x in xs)
        for row in summary['rows']:
            c=row['time_change_percent']
            lines.append(f"| {row['configuration']} | {fmt(c[:1])} | {fmt(c[1:2])} | {fmt(c[2:4])} | {fmt(c[4:6])} | {fmt(c[6:])} |")
        lines+=['','负数表示耗时下降。stage1候选pure/mixed减少9.336%–11.300%，decode变化在0.24%内；所有配置基线漂移低于0.78%。',
            '32行、2warps及128行组合没有超过原基线；不能从理论KV复用或占用率预先推断加速。',
            '下一步只复核3点：重复优胜rows64/w4/s1/tile32，单独改tile64，单独改stage3。保持同一算子提交、decode参数、9工作点与门槛；不改变原筛选文件。',
            '若邻域无额外收益，固定stage1进入扩大数值/Graph/形状验证，再真实引擎接入和官方4K/16K四轮。',
            '所有采样、失败和资源记录保留；完整任务精度仍暂缓，当前已选服务版本仍EXP-0014。','',
            '[完整初筛结果](../experiments/EXP-0015/artifacts/'+run.name+'/RESULTS.md)。']
        (REPO/'docs/gqa_prefill_screen_20261022.md').write_text('\n'.join(lines)+'\n')
        with (EXP/'steps.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),action='complete_coarse_screen_and_prepare_single_factor_refinement',
                configurations=12,checks=240,winner='rows64_w4_s1',serving_selection_unchanged='EXP-0014'))+'\n')
        paths=['experiments/EXP-0015/'+n for n in files+['refinement_plan.json','steps.jsonl']]+['docs/gqa_prefill_screen_20261022.md']
        forbidden=re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|lab_inst_[A-Za-z0-9]+|ssh\.flagos\.io')
        for rel in paths:assert not forbidden.search((REPO/rel).read_bytes()),rel
        git('add','--',*paths);git('diff','--cached','--check')
        git('commit','--only','-m','experiments: record GQA prefill winner and prepare local parameter refinement','--',*paths)
        head=git('rev-parse','HEAD');git('push','origin','main')
        assert git('ls-remote',git('remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
    target=ROOT/'artifacts/EXP-0015'/datetime.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_refine')
    with (ROOT/'artifacts/EXP-0015-refine-controller.log').open('a') as log:
        child=subprocess.Popen(['python3',str(EXP/'run_refine.py'),'--run-dir',str(target)],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(100):
        if latest.exists():break
        if child.poll() is not None:raise RuntimeError('Refinement exited before receipt')
        time.sleep(.2)
    else:raise RuntimeError('No refinement receipt')
    receipt=read(latest);assert receipt['pid']==child.pid
    (ROOT/'artifacts/EXP-0015-refine-launch.json').write_text(json.dumps(dict(documentation_commit=head,**receipt),indent=2)+'\n')
    print(json.dumps(dict(documentation_commit=head,receipt=receipt)),flush=True)


if __name__=='__main__':main()
