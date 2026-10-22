"""Freeze the validated prefill choice in a new engine branch and publish evidence."""
from pathlib import Path
import ast
import datetime
import fcntl
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path('/workspace/LeanScale')
REPO = ROOT/'MiniCPM-Infer-Sys-Opt'
EXP = REPO/'experiments/EXP-0016'
WT = ROOT/'worktrees/EXP-0016'
STAGING = Path(__file__).resolve().parent
PARENT = {'vllm-plugin-FL':'82c6a143f6984f946df42125f1a8699799e8045f',
          'FlagGems':'1c526ca99d412e203ed3caa8b1663566976ea38c'}
GEMS = 'fd209df717648002ebf5ba4ec351424dcc997a53'
BRANCH = 'leanscale/iluvatar-gqa-prefill-stage1'
EXT = ROOT/'artifacts/EXP-0015/20261022_144537_extended'
REFINE = ROOT/'artifacts/EXP-0015/20261022_142759_refine'
OLD_LOG = 'decode_warps=4 decode_stages=1 segments=16 decode_layout=1 decode_rows=8'
NEW_LOG = OLD_LOG+' prefill_rows=64 prefill_warps=4 prefill_stages=1 prefill_tile=32'
FILES = ['run_split_e2e.py','launch_split.py','finish_split.py','queue_official.py',
         'analyze_schedule.py','build_scripts.py','prepare_integration.py']


def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(p,*args):return subprocess.check_output(['git','-C',str(p),*args],text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def step(action,**data):
    with (EXP/'steps.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),action=action,**data),ensure_ascii=False)+'\n')


def check_routes():
    sys.path.insert(0,str(EXP))
    spec=importlib.util.spec_from_file_location('exp16_gate',EXP/'run_split_e2e.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    reference=ROOT/'artifacts/EXP-0014/20261022_124748_official_16k'
    old_log=(reference/'candidate_benchmark/server.log').read_text()
    fixture=old_log.replace(OLD_LOG,NEW_LOG)
    assert fixture.count(NEW_LOG)==42
    graph=next(p for p in (reference/'candidate_cache').rglob('computation_graph.py') if 'torch.ops.vllm.fl_rope_kv_dispatch' in p.read_text())
    decode=next(p for p in (reference/'triton_cache').glob('*/_attention.json')
                if read(p).get('num_stages')==1 and read(p).get('shared')==18432)
    expected=read(REFINE/'rows64_w4_s1_tile32.json')['timings'][0]['resources']['ir_sha256']['ttir']
    prefill=next(p for p in (REFINE/'triton_cache').glob('*/_attention.json')
                 if p.with_suffix('.ttir').exists() and sha(p.with_suffix('.ttir'))==expected)
    checks=[]
    with tempfile.TemporaryDirectory(prefix='exp16_route_') as tmp:
        tmp=Path(tmp);mod.RUN=tmp
        folder=tmp/'candidate_benchmark';folder.mkdir()
        dst=tmp/'candidate_cache/fixture/computation_graph.py';dst.parent.mkdir(parents=True);shutil.copy2(graph,dst)
        metas={}
        for kind,p in [('decode',decode),('prefill',prefill)]:
            dst=tmp/'triton_cache'/kind/'_attention.json';dst.parent.mkdir(parents=True)
            shutil.copy2(p,dst);shutil.copy2(p.with_suffix('.ttir'),dst.with_suffix('.ttir'));metas[kind]=dst
        def run_check(name,passed):
            try:mod.verify_route(folder);ok=True
            except AssertionError:ok=False
            assert ok==passed,name
            checks.append(dict(name=name,passed=True))
        for name,log,passed in [
            ('positive_distinct_decode_prefill',fixture,True),
            ('reject_old_prefill_log',old_log,False),
            ('reject_withdrawn_scheduler',fixture+'\nKVReservationAsyncScheduler\n',False),
            ('reject_missing_layer',fixture.replace('LeanScale GQA active:','missing:',1),False),
            ('reject_wrong_prefill_argument',fixture.replace('prefill_stages=1','prefill_stages=2',1),False),
            ('reject_wrong_decode_argument',fixture.replace('decode_rows=8','decode_rows=16',1),False)]:
            (folder/'server.log').write_text(log);run_check(name,passed)
        (folder/'server.log').write_text(fixture)
        for kind in ('decode','prefill'):
            p=metas[kind];original=p.read_text();p.unlink()
            run_check('reject_missing_'+kind+'_compile',False);p.write_text(original)
            bad=json.loads(original);bad['num_stages']=2;write(p,bad)
            run_check('reject_'+kind+'_stage2',False);p.write_text(original)
            bad=json.loads(original);bad['shared']+=2048;write(p,bad)
            run_check('reject_'+kind+'_wrong_shared',False);p.write_text(original)
        for kind,old,new in [('decode','8x128x!tt.ptr<bf16>','16x128x!tt.ptr<bf16>'),
                             ('decode','2x!tt.ptr<i32>','32x!tt.ptr<i32>'),
                             ('prefill','64x128x!tt.ptr<bf16>','32x128x!tt.ptr<bf16>'),
                             ('prefill','128x32x!tt.ptr<bf16>','128x64x!tt.ptr<bf16>')]:
            p=metas[kind].with_suffix('.ttir');original=p.read_text();bad=original.replace(old,new)
            assert bad!=original;p.write_text(bad)
            run_check('reject_'+kind+'_IR_'+old,False);p.write_text(original)
        for case,cases in [('4k',{(4096,1024,64,256)}),('16k',{(16384,1024,64,128)})]:
            mod.verified_parent_reference(case,cases);checks.append(dict(name='complete_parent_'+case,passed=True))
        mod.verify_candidate_gate();checks.append(dict(name='candidate_commit_code_and_diagnostic_hashes',passed=True))
    write(EXP/'route_cpu_checks.json',dict(status='passed',checks=checks,
          scope='Synthetic mutations using actual parent decode and candidate prefill IR; actual new service route still pending'))
    return checks


def publish_prepared(commits,checks):
    with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with (REPO/'docs/optimization_journal.md').open('a') as f:
            f.write('\n\n### 2026-10-22：EXP-0015扩大验证完成，EXP-0016接入prefill stage1\n\n'
                    '42项数值/Graph检查、39444个前后变动query的FP32对照通过；36个计时点无>3%回归或漂移。'
                    '17个pure prefill时间下降6.16%–26.15%，7个mixed下降9.97%–24.77%，12个decode变化在0.16%内。'
                    '独立EXP-0016仅增加prefill显式参数与日志，逆向替换恢复EXP-0014原文件。'
                    '真实服务检查分别识别8行decode与64行prefill，要求各自的stage1编译证据，之后才执行官方4K/16K。'
                    '整模收益尚未测，EXP-0014仍是当前性能基线；完成本轮后进入4K专项。\n')
        paths=['experiments/EXP-0016','docs/optimization_journal.md','docs/gqa_prefill_expanded_20261022.md']
        assert not git(REPO,'diff','--cached','--name-only')
        forbidden=re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|lab_inst_[A-Za-z0-9]+|ssh\.flagos\.io')
        for rel in paths:
            p=REPO/rel
            for item in p.rglob('*') if p.is_dir() else [p]:
                if item.is_file() and item.suffix!='.pyc':assert not forbidden.search(item.read_bytes()),str(item)
        git(REPO,'add','--',*paths);git(REPO,'diff','--cached','--check')
        print(git(REPO,'commit','--only','-m','experiments: integrate validated GQA prefill stage1 for official E2E','--',*paths),flush=True)
        head=git(REPO,'rev-parse','HEAD');git(REPO,'push','origin','main')
        assert git(REPO,'ls-remote',git(REPO,'remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
    write(ROOT/'artifacts/EXP-0016-setup.json',dict(status='prepared',candidate_commits=commits,docs_commit=head,checks=checks))
    active=read(ROOT/'artifacts/active_inference_plan.json');assert active['selected_experiment']=='EXP-0014'
    active.update(current_experiment='EXP-0016',candidate_commits=commits,
                  status='GQA prefill stage1 integrated; official E2E pending; EXP14 remains selected')
    write(ROOT/'artifacts/active_inference_plan.json',active)
    print(json.dumps(dict(status='prepared',commits=commits,docs_commit=head,checks=len(checks))),flush=True)


def main():
    state=read(EXT/'status.json');d=read(EXT/'validation.json')
    assert state['stage']=='validation_complete' and state['safe_to_resume_gpu']
    assert not state['regression_names'] and not state['drift_names']
    assert d['status']=='passed' and len(d['checks'])==42 and len(d['timings'])==36
    assert all(abs(x['baseline_drift_percent'])<=3 and x['time_change_percent']<=3 for x in d['timings'])
    assert all(c['graph_passed'] for c in d['checks'])
    assert sum(c['all_query_fp32']['queries_checked']+c['mutated_all_query_fp32']['queries_checked'] for c in d['checks'])==39444
    assert state['commits']=={'vllm-plugin-FL':PARENT['vllm-plugin-FL'],'FlagGems':GEMS}
    publication=read(EXT/'publication.json');assert publication['status']=='pushed' and publication['remote_verified']
    git(REPO,'merge-base','--is-ancestor',publication['commit'],'HEAD')
    assert read(ROOT/'artifacts/active_inference_plan.json')['selected_serving_commits']==PARENT
    assert not EXP.exists() and not WT.exists() and not git(REPO,'status','--porcelain')
    EXP.mkdir(parents=True);WT.mkdir(parents=True)
    (EXP/'README.md').write_text('''# EXP-0016：GQA prefill stage1接入官方端到端测评

假设先于修改记录：EXP-0015的rows64/warps4/stage1/tile32已通过独立复测、42项扩大数值/Graph检查和36个计时点。
在EXP-0014现有最佳引擎上增加prefill显式参数，decode分页页号广播/8行/4warps/1stage/16segments、RoPE+KV融合及原调度保持。
FlagGems沿用EXP-0015冻结提交fd209df7，其通用默认prefill_stages仍为2；只有此插件明确选stage1。
插件反向去除新参数和新日志必须与EXP-0014逐字一致；supports/fallback与形状边界不变。

实际接入门槛：42层GQA新参数日志、42层RoPE+KV融合、35张FULL_DECODE_ONLY图；独立服务缓存中所有attention均CoreX/4warps/1stage。
decode必须有8行Q、两页号与2×16广播、18KiB shared；prefill必须有64行Q、tile32的页号/K读取和QK张量、32KiB shared。两类实际编译证据缺一不可。
CPU合成日志/IR拒绝检查仅验证检查器，真实服务检查仍在每组正式测评前后执行。

官方参数保持：4K[4096,1024,64,256]、16K[16384,1024,64,128]，每组四轮弃首轮，原benchmark哈希不变。
对照EXP-0014与原始完整基线；历史会话不是交错A/B，4K父CV4.43%，小幅收益需谨慎归因。
4K吞吐若退化超过3%则暂停16K分析。两组完整判定前，EXP-0014仍为已选性能版本。
完整模型精度按用户要求暂缓，数值/Graph验证不可省略。今晚23:35计算截止、23:50保存；55/90分钟完整case起跑预算。
完成后评估本轮GQA方案，再重点做4K专项。所有失败、原始结果和文档保存并推送fork；不创建上游PR。
''')
    write(EXP/'parent_commits.json',PARENT);write(EXP/'run_window_at_setup.json',read(ROOT/'artifacts/run_window.json'))
    step('Record prefill hypothesis, all expanded results, and official protocol before source edit')
    try:
        print(git(ROOT/'worktrees/EXP-0014/vllm-plugin-FL','worktree','add','-b',BRANCH,str(WT/'vllm-plugin-FL'),PARENT['vllm-plugin-FL']),flush=True)
        print(git(ROOT/'worktrees/EXP-0015/FlagGems','worktree','add','--detach',str(WT/'FlagGems'),GEMS),flush=True)
        rel='vllm_fl/dispatch/backends/flaggems/impl/iluvatar_gqa.py';p=WT/'vllm-plugin-FL'/rel;old=p.read_text()
        old_call='decode_layout=1, decode_rows=8)'
        new_call='decode_layout=1, decode_rows=8,\n                  prefill_rows=64, prefill_warps=4, prefill_stages=1, prefill_tile=32)'
        assert old.count(old_call)==1 and old.count(OLD_LOG)==1
        new=old.replace(old_call,new_call).replace(OLD_LOG,NEW_LOG)
        assert new.replace(new_call,old_call).replace(NEW_LOG,OLD_LOG)==old
        tree=ast.parse(new)
        call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='paged_gqa')
        expected={k.arg:ast.literal_eval(k.value) for k in call.keywords};assert expected==d['config']
        p.write_text(new);subprocess.run(['python3','-m','py_compile',str(p)],check=True)
        assert subprocess.run(['git','-C',str(WT/'vllm-plugin-FL'),'grep','-n','KVReservationAsyncScheduler'],capture_output=True).returncode==1
        git(WT/'vllm-plugin-FL','add',rel);git(WT/'vllm-plugin-FL','diff','--cached','--check')
        print(git(WT/'vllm-plugin-FL','commit','-m','perf(iluvatar): select validated single-stage GQA prefill'),flush=True)
        commits={'vllm-plugin-FL':git(WT/'vllm-plugin-FL','rev-parse','HEAD'),'FlagGems':GEMS}
        write(EXP/'candidate_commits.json',commits)
        for n in FILES:shutil.copy2(STAGING/n,EXP/n)
        shutil.copytree(STAGING/'parent_exp14',EXP/'parent_exp14')
        for p in (EXP/'parent_exp14').iterdir():assert sha(p)==sha(REPO/'experiments/EXP-0014'/p.name)
        before={n:sha(EXP/n) for n in FILES}
        subprocess.run(['python3',str(EXP/'build_scripts.py')],check=True)
        assert before=={n:sha(EXP/n) for n in FILES}
        gate=dict(status='passed',candidate_commits=commits,launch=expected,
                  diagnostic_hashes={str(p):sha(p) for p in [REFINE/'rows64_w4_s1_tile32.json',EXT/'validation.json']},
                  code_hashes={'vllm-plugin-FL':{rel:sha(WT/'vllm-plugin-FL'/rel)},'FlagGems':{
                      'src/flag_gems/runtime/backend/_iluvatar/fused/paged_gqa.py':sha(WT/'FlagGems/src/flag_gems/runtime/backend/_iluvatar/fused/paged_gqa.py')}},
                  validation_scope='42 numerical/Graph checks, 39444 query checks and 36 timings; full model accuracy deferred',
                  integration_cpu_check='Inverse explicit kwargs and runtime log replacement restores parent exactly',script_hashes=before)
        write(EXP/'candidate_validation.json',gate)
        step('Freeze explicit prefill parameters; preserve decode, fallback and scheduling exactly',commits=commits)
        checks=check_routes()
        subprocess.run(['python3','-m','py_compile',*[str(p) for p in EXP.glob('*.py')]],check=True)
        assert sha(EXP/'analyze_schedule.py')=='574042b1d7d97454691b3608f2fe0073cc760a6c0314a1cd1315bc95a769ea9f'
        step('Pass distinct decode/prefill route rejection checks, complete parent data and candidate hashes',checks=len(checks))
        print(git(WT/'vllm-plugin-FL','push','leanscale','HEAD:refs/heads/'+BRANCH),flush=True)
        for name,branch in [('vllm-plugin-FL',BRANCH),('FlagGems','leanscale/iluvatar-gqa-prefill-tuning')]:
            repo=WT/name;assert git(repo,'ls-remote',git(repo,'remote','get-url','--push','leanscale'),'refs/heads/'+branch).split()[0]==commits[name]
        step('Verify plugin and kernel forks at frozen candidate commits')
        lines=['# GQA prefill 扩大验证与引擎接入','',
               '2026-10-22，EXP-0015→EXP-0016。单算子验证结果，整模收益尚待官方4K/16K。',
               '42项数值/Graph检查全部通过，变动前后累计39444个query独立FP32对照；包括真实>2^31元素prefill地址。',
               '36工作点均无>3%回归，最大前后基线漂移0.769%。负数为耗时降低。','',
               '| 工作点 | 完整GQA耗时变化 | 前后基线漂移 |','|---|---:|---:|']
        lines += [f"| {x['name']} | {x['time_change_percent']:+.3f}% | {x['baseline_drift_percent']:+.3f}% |" for x in d['timings']]
        lines += ['','原方案prefill64行/4warps/stage2/tile32；新方案仅把软件流水级数改为1，decode组合保持已选版本。',
                  '更大的tile64和stage3实测退化，因此未采用。该选择基于实测，不推断任意形状都获益；以上36点完整列出。',
                  '插件显式传入prefill参数，Gems通用默认仍stage2。新引擎在独立EXP-0016树；反向替换完全恢复父文件，fallback未变。',
                  '实际编译检查按8行decode与64行prefill分开识别，两者都须stage1，分别校验18KiB/32KiB shared与tile形状。',
                  '[完整原始数据](../experiments/EXP-0015/artifacts/'+EXT.name+'/RESULTS.md)。',
                  '[接入方案与测评门槛](../experiments/EXP-0016/README.md)。']
        (REPO/'docs/gqa_prefill_expanded_20261022.md').write_text('\n'.join(lines)+'\n')
        (EXP/'PR_DRAFT.md').write_text('''# Draft：赛事PR文案

标题：【FlagOS开放计算全球挑战赛S2.赛题2】【LeanScale】接入单级流水GQA prefill优化

## 变更
MiniCPM5-2B Q16/KV2/D128因果注意力在已验证分页decode布局上，显式选择prefill64行/4warps/stage1/tile32。
适用条件、回退、RoPE+KV融合和调度与父版本一致；FlagGems默认stage2保留。

## 验证
42项扩大数值/Graph检查、36个计时点通过；纯prefill及mixed单算子有收益，decode无显著回归。
真实42层接入和官方4K/16K四轮弃首轮测评待补。完整MATH500精度暂缓，保持Draft，不创建上游PR。
''')
        publish_prepared(commits,checks)
    except BaseException as exc:
        step('Integration preparation failed; preserve all files and diagnose before retry',error=repr(exc));raise


if __name__=='__main__':main()
