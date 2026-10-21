"""Publish the complete screen and launch expanded validation for layout1/rows8."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import shutil
import subprocess
import time

ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt';EXP=REPO/'experiments/EXP-0013'
STAGING=Path(__file__).resolve().parent

def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a],text=True,stderr=subprocess.STDOUT,timeout=120).strip()

def main():
    s=json.loads((ROOT/'artifacts/EXP-0013-request.json').read_text());run=Path(s['run_directory'])
    assert s['stage']=='screen_complete' and s['safe_to_resume_gpu']
    assert json.loads((run/'publication.json').read_text())['status']=='pushed'
    latest=ROOT/'artifacts/EXP-0013-expanded-request.json';assert not latest.exists(),'Inspect an existing run before retrying'
    data=json.loads((run/'layout1_rows8.json').read_text());assert data['status']=='passed'
    summary=json.loads((run/'screen_summary.json').read_text())
    assert next(x for x in summary['rows'] if x['configuration']=='layout1_rows8')['global_candidate_gate_passed']
    deadline=datetime.datetime.fromisoformat(json.loads((ROOT/'artifacts/run_window.json').read_text())['compute_stop_at'])
    assert (deadline-datetime.datetime.now().astimezone()).total_seconds()>600
    assert not git('status','--porcelain')
    for n in ['build_scripts.py','validate_extended.py','run_extended.py','launch_extended.py']:
        shutil.copy2(STAGING/n,EXP/n)
    evidence={}
    for config in ['layout0_rows16','layout1_rows16','layout0_rows8','layout1_rows8']:
        p=run/'ir'/config/'B64_L4700.ttgir';text=p.read_text()
        evidence[config]={'ttgir_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'mma_layout':[l for l in text.splitlines() if '#ttg.iluvatar_mma' in l],
            'page_related_loads':[l.strip() for l in text.splitlines() if 'tt.load' in l and ('blocks' in l or 'page_' in l)],
            'query_and_accumulator_shapes':[l.strip() for l in text.splitlines() if 'arith.constant' in l and ('16x128x' in l or '8x128x' in l)]}
    (EXP/'screen_ir_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    lines=['# GQA 分页索引与 8 行查询：筛选结果','',
           '2026-10-21，EXP-0013。基线是已完成官方 E2E 的 EXP-0012 stage1。8组全部通过20项初步检查，共160项；全部配置与采样保留。',
           '按页加载后广播＋8行逻辑查询（layout1/rows8）是三个工作点中收益最大的组合。尚未接入引擎，本报告不是端到端收益。','',
           '| 配置 | B64/L4700 时间变化 | B30/L16384 时间变化 | B1/L4096 时间变化 | 前后基线稳定 |',
           '| --- | ---: | ---: | ---: | --- |']
    for r in summary['rows']:
        c=r['time_change_percent'];lines.append(f"| {r['configuration']} | {c[0]:+.3f}% | {c[1]:+.3f}% | {c[2]:+.3f}% | {r['stable_baseline']} |")
    lines += ['', '负数表示完整 GQA 调用耗时减少，包含主核和归约。不要把这些百分比当成整模吞吐提升。',
        '等价对照的时间变化在0.23%内；所有配置的前后基线漂移低于0.26%。每组7个样本，每次Graph包含8次调用。',
        'layout1是page16/tile32时按页加载两个页号并广播到32个token；layout2改变K读取方向；layout3组合两者。rows8将逻辑查询行减到真实8个Q head。prefill保持原路径。',
        '单独页广播和单独rows8均有收益，组合进一步改善；K转置读取虽然减少shared字段，但寄存器字段增多，整体逊于layout1/rows8。',
        '最优组合资源字段 n_regs 96→83、shared 20480→18432、n_spills仍0。TTGIR展示页号载荷2元素、查询及累加张量8行，而MMA instrShape仍[16,16,16]；不能据此声称Tensor Core计算量减半或确切occupancy提升。',
        '初测覆盖page8/16/32/64、BF16/FP16、NHD/HND、空context/query、页边界、ragged/mixed、非连续Q、FP32 oracle和Graph内容变化。',
        '下一步固定同一源码和layout1/rows8，扩大到26项数值/Graph/实际>2^31元素地址检查、34个计时点；纯prefill与mixed也要检查回归。通过后才考虑引擎接入及官方四轮。若收益不能复现或出现不可接受回归，保存负结果转GQA prefill调优。','',
        '[全部原始筛选结果](../experiments/EXP-0013/artifacts/'+run.name+'/RESULTS.md)。',
        '[IR证据与哈希](../experiments/EXP-0013/screen_ir_evidence.json)。原始IR完整保留服务器；不上传二进制缓存。']
    (REPO/'docs/gqa_paged_layout_screen_20261021.md').write_text('\n'.join(lines)+'\n')
    with (EXP/'README.md').open('a') as f:f.write('\n8组已完成，160项初测通过；layout1/rows8进入扩大验证：[筛选报告](../../docs/gqa_paged_layout_screen_20261021.md)。\n')
    p=EXP/'PR_DRAFT_FlagGems.md';p.write_text(p.read_text().replace('单算子正确性与性能待测，扩大验证及整模测评待定；',
        '8组共160项初测通过；layout1/rows8在三个工作点完整GQA耗时减少23.1%～24.9%，扩大验证和整模收益待定；'))
    with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps({'time':datetime.datetime.now().astimezone().isoformat(),
        'action':'all eight screen configurations passed; select layout1/rows8 for expanded validation only',
        'source_run':run.name,'expanded_checks':26,'timing_points':34,'serving_selection_unchanged':'EXP-0012'})+'\n')
    with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        paths=['docs/gqa_paged_layout_screen_20261021.md']+['experiments/EXP-0013/'+n for n in [
            'build_scripts.py','validate_extended.py','run_extended.py','launch_extended.py','screen_ir_evidence.json',
            'README.md','PR_DRAFT_FlagGems.md','steps.jsonl']]
        assert not git('diff','--cached','--name-only')
        git('add','--',*paths);git('diff','--cached','--check')
        git('commit','--only','-m','experiments: validate paged KV broadcast and eight-row GQA candidate','--',*paths)
        head=git('rev-parse','HEAD');git('push','origin','main')
        assert git('ls-remote',git('remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
    with (ROOT/'artifacts/EXP-0013-extended-controller.log').open('a') as log:
        child=subprocess.Popen(['python3',str(EXP/'run_extended.py')],stdout=log,stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL,start_new_session=True)
    for _ in range(100):
        if latest.exists():break
        if child.poll() is not None:raise RuntimeError('Expanded controller exited before receipt')
        time.sleep(.2)
    receipt=json.loads(latest.read_text());assert receipt['pid']==child.pid
    print(json.dumps({'documentation_commit':head,'receipt':receipt}),flush=True)

if __name__=='__main__':main()
