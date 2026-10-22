"""Freeze validated EXP13 layout1/rows8 in an isolated EXP14 serving branch."""
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

ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt'
EXP=REPO/'experiments/EXP-0014';WT=ROOT/'worktrees/EXP-0014'
PARENT={'vllm-plugin-FL':'08d1711e58fdd0f4cd4f746f9e21cb59800ec1d6',
        'FlagGems':'19aa059af4ce456ef2e34e3bb83eb949d7e7b702'}
GEMS='1c526ca99d412e203ed3caa8b1663566976ea38c'
BRANCH='leanscale/iluvatar-gqa-paged-layout'
STAGING=Path(__file__).resolve().parent
SCREEN=ROOT/'artifacts/EXP-0013/20261021_201938_screen'
EXT=ROOT/'artifacts/EXP-0013/20261021_202733_extended'
def read(p):return json.loads(p.read_text())
def write(p,data):p.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(p,*a):return subprocess.check_output(['git','-C',str(p),*a],text=True,timeout=120).strip()
def step(action,**data):
    with (EXP/'steps.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),action=action,**data),ensure_ascii=False)+'\n')

def finish_preparation(commits,old_log,new_log):
    sys.path.insert(0,str(EXP))
    spec=importlib.util.spec_from_file_location('exp14_gate',EXP/'run_split_e2e.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    reference=ROOT/'artifacts/EXP-0012/20261021_184434_official_16k'
    old_log_text=(reference/'candidate_benchmark/server.log').read_text()
    fixture=old_log_text.replace(old_log.split('; layer=')[0],new_log.split('; layer=')[0])
    assert fixture.count(new_log.split('; layer=')[0])==42
    graph=next(q for q in (reference/'candidate_cache').rglob('computation_graph.py') if 'torch.ops.vllm.fl_rope_kv_dispatch' in q.read_text())
    wanted=sha(SCREEN/'ir/layout1_rows8/B64_L4700.ttir')
    meta=next(q for q in (SCREEN/'triton_cache').glob('*/_attention.json')
              if q.with_suffix('.ttir').is_file() and sha(q.with_suffix('.ttir'))==wanted)
    checks=[]
    with tempfile.TemporaryDirectory(prefix='exp14_route_') as tmp:
        tmp=Path(tmp);mod.RUN=tmp;folder=tmp/'candidate_benchmark';folder.mkdir()
        dst=tmp/'candidate_cache/fixture/computation_graph.py';dst.parent.mkdir(parents=True);shutil.copy2(graph,dst)
        dst=tmp/'triton_cache/fixture/_attention.json';dst.parent.mkdir(parents=True)
        shutil.copy2(meta,dst);shutil.copy2(meta.with_suffix('.ttir'),dst.with_suffix('.ttir'))
        for name,log,passed in [('positive_fixture',fixture,True),('reject_parent_logs',old_log_text,False),
            ('reject_withdrawn_scheduler',fixture+'\nKVReservationAsyncScheduler\n',False),
            ('reject_missing_layer',fixture.replace('LeanScale GQA active:','missing:',1),False),
            ('reject_wrong_row_argument',fixture.replace('decode_rows=8','decode_rows=16',1),False)]:
            (folder/'server.log').write_text(log)
            try:mod.verify_route(folder);ok=True
            except AssertionError:ok=False
            assert ok==passed,name;checks.append(dict(name=name,passed=True))
        (folder/'server.log').write_text(fixture)
        actual_ir=dst.with_suffix('.ttir').read_text()
        for name,ir in [('reject_wrong_compiled_rows',actual_ir.replace('8x128x!tt.ptr<bf16>','16x128x!tt.ptr<bf16>')),
                        ('reject_missing_compiled_page_load',actual_ir.replace('2x!tt.ptr<i32>','32x!tt.ptr<i32>'))]:
            assert ir!=actual_ir
            dst.with_suffix('.ttir').write_text(ir)
            try:mod.verify_route(folder)
            except AssertionError:checks.append(dict(name=name,passed=True))
            else:raise AssertionError(name)
        dst.with_suffix('.ttir').write_text(actual_ir)
        original_meta=dst.read_text();bad=json.loads(original_meta);bad['shared']=20480;write(dst,bad)
        try:mod.verify_route(folder)
        except AssertionError:checks.append(dict(name='reject_wrong_compiled_resources',passed=True))
        else:raise AssertionError('Wrong resources accepted')
        dst.write_text(original_meta);dst.unlink()
        try:mod.verify_route(folder)
        except AssertionError:checks.append(dict(name='reject_missing_service_compile_evidence',passed=True))
        else:raise AssertionError('Missing compile evidence accepted')
        for case,cases in [('4k',{(4096,1024,64,256)}),('16k',{(16384,1024,64,128)})]:
            mod.verified_parent_reference(case,cases)
            checks.append(dict(name='verify_complete_parent_'+case,passed=True))
        mod.verify_candidate_gate()
    write(EXP/'route_cpu_checks.json',dict(status='passed',checks=checks,
          scope='Synthetic route mutations and real audited parent references; actual new service route pending'))
    parent_runner=(STAGING/'parent_exp12/run_split_e2e.py').read_text();child_runner=(EXP/'run_split_e2e.py').read_text()
    assert parent_runner[parent_runner.index('SERVER ='):parent_runner.index('STATE =')] == child_runner[child_runner.index('SERVER ='):child_runner.index('STATE =')]
    assert sha(EXP/'analyze_schedule.py')=='574042b1d7d97454691b3608f2fe0073cc760a6c0314a1cd1315bc95a769ea9f'
    subprocess.run(['python3','-m','py_compile',*[str(p) for p in EXP.glob('*.py')]],check=True)
    step('Pass CPU route rejection checks, both complete parent references, and unchanged workload/analyzer verification',checks=len(checks))
    print(git(WT/'vllm-plugin-FL','push','leanscale','HEAD:refs/heads/'+BRANCH),flush=True)
    for name,branch in [('vllm-plugin-FL',BRANCH),('FlagGems','leanscale/iluvatar-gqa-kv-layout')]:
        repo=WT/name
        assert git(repo,'ls-remote',git(repo,'remote','get-url','--push','leanscale'),'refs/heads/'+branch).split()[0]==commits[name]
    step('Verify both forks match frozen candidate commits')
    (EXP/'PR_DRAFT.md').write_text('''# Draft：赛事PR文案

标题：【FlagOS开放计算全球挑战赛S2.赛题2】【LeanScale】接入GQA分页页号广播与8行查询布局

Q16/KV2/D128的decode按页读取页表并广播到token，逻辑查询行从16收紧到有效的8个head。
插件显式选择decode_layout=1、decode_rows=8，4warps/1stage/16segments及原RoPE+KV融合、调度、回退保持。
FlagGems候选沿用EXP-0013冻结提交，26项扩大数值/Graph检查和34计时点已通过；单算子收益不代表整模收益。
真实服务接入、官方4K/16K四轮结果待补；完整任务精度暂缓。保持Draft，不创建上游PR。
''')
    with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with (REPO/'docs/optimization_journal.md').open('a') as f:
            f.write('\n\n### 2026-10-22：恢复服务器，进入EXP-0014分页布局官方测评\n\n'
                    '昨日EXP-0013扩大验证全部通过；新建独立引擎分支显式接入layout1/rows8。'
                    '编译IR门槛同时核验8行Q、2项页号读取与广播，11项CPU门槛/父结果复核通过。'
                    '采用修正后的原调度分析器；父4K使用带原始哈希的离线恢复收据。'
                    '官方4K/16K完整四轮测评待执行，23:35计算截止、23:50保存。\n')
        paths=['experiments/EXP-0014','docs/optimization_journal.md']
        assert not git(REPO,'diff','--cached','--name-only')
        forbidden=re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|lab_inst_[A-Za-z0-9]+|ssh\\.flagos\\.io')
        for p in EXP.rglob('*'):
            if p.is_file() and p.suffix!='.pyc':assert not forbidden.search(p.read_bytes()),str(p)
        git(REPO,'add','--',*paths);git(REPO,'diff','--cached','--check')
        print(git(REPO,'commit','--only','-m','experiments: integrate validated paged GQA layout for official E2E','--',*paths),flush=True)
        git(REPO,'push','origin','main');head=git(REPO,'rev-parse','HEAD')
        assert git(REPO,'ls-remote',git(REPO,'remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
    write(ROOT/'artifacts/EXP-0014-setup.json',dict(status='prepared',candidate_commits=commits,docs_commit=head,checks=checks))
    active=read(ROOT/'artifacts/active_inference_plan.json')
    assert active['selected_experiment']=='EXP-0012'
    active.update(current_experiment='EXP-0014',candidate_commits=commits,
                  status='Paged KV broadcast/eight-row GQA integrated; official E2E pending; EXP12 remains selected')
    write(ROOT/'artifacts/active_inference_plan.json',active)
    print(json.dumps(dict(status='prepared',commits=commits,docs_commit=head,checks=len(checks))),flush=True)

def main():
    state=read(EXT/'status.json');validation=read(EXT/'validation.json')
    assert state['stage']=='validation_complete' and state['safe_to_resume_gpu']
    assert not state['regression_names'] and not state['drift_names']
    assert validation['status']=='passed' and len(validation['checks'])==26 and len(validation['timings'])==34
    assert all(abs(x['baseline_drift_percent'])<=3 and x['time_change_percent']<=3 for x in validation['timings'])
    assert read(SCREEN/'layout1_rows8.json')['status']=='passed'
    assert state['commits']=={'vllm-plugin-FL':PARENT['vllm-plugin-FL'],'FlagGems':GEMS}
    assert read(EXT/'publication.json')['status']=='pushed'
    assert not EXP.exists() and not WT.exists()
    EXP.mkdir(parents=True);WT.mkdir(parents=True)
    (EXP/'README.md').write_text('''# EXP-0014：分页 KV 读取与8行 GQA接入官方端到端测评

假设先于代码修改记录：EXP-0013页号广播与8行decode组合已通过26项扩大数值/Graph检查和34个计时点。
30个decode工作点时间减少22.42%–27.96%，mixed减少约2.35%–2.39%，纯prefill近似不变；这些不是整模收益。
在EXP-0012最优引擎上，仅增加decode_layout=1、decode_rows=8的显式参数及可核验日志。
固定Gems提交1c526ca9，维持4warps/1stage/16segments、原RoPE+KV融合、supports/fallback和原调度。

## 接入与性能门槛

- 反向替换两个参数和日志后，插件文件必须与EXP-0012逐字相同；不修改默认Gems参数和prefill。
- 真实HTTP smoke成功；42层GQA均带新参数，42层RoPE+KV融合，35张FULL_DECODE_ONLY图完成。
- 隔离服务缓存的全部stage1 attention元数据为CoreX/4warps/shared18432；实际TTIR必须有8×128 Q加载、2个页号加载、2×16广播和32元素展开。
- 官方serve命令、benchmark脚本与随机数据配置不变：4K[4096,1024,64,256]，16K[16384,1024,64,128]，各四轮丢弃首轮。
- 对照EXP-0012完整成绩与原始基线；历史会话非交错A/B。父4K引用已审核的离线恢复，不删除或掩盖原后处理失败。
- 4K吞吐退化超过3%则暂停16K并分析；完整E2E判定前保留EXP-0012作为已选版本。
- 原调度普通计数器必须存在，撤销的准入调度日志必须不存在。完整任务精度按用户要求暂缓。
- 使用当天run_window的23:35计算截止、23:50归档目标；每个case启动须有55/90分钟余量，自动保存并推送所有结果。
''')
    write(EXP/'parent_commits.json',PARENT)
    write(EXP/'run_window_at_setup.json',read(ROOT/'artifacts/run_window.json'))
    step('Record hypothesis, validated input evidence and unchanged official protocol before source edit')
    try:
        print(git(ROOT/'worktrees/EXP-0012/vllm-plugin-FL','worktree','add','-b',BRANCH,str(WT/'vllm-plugin-FL'),PARENT['vllm-plugin-FL']),flush=True)
        print(git(ROOT/'worktrees/EXP-0013/FlagGems','worktree','add','--detach',str(WT/'FlagGems'),GEMS),flush=True)
        rel='vllm_fl/dispatch/backends/flaggems/impl/iluvatar_gqa.py'
        p=WT/'vllm-plugin-FL'/rel;old=p.read_text()
        old_call='segments=16, decode_warps=4, decode_stages=1)'
        new_call='segments=16, decode_warps=4, decode_stages=1,\n                  decode_layout=1, decode_rows=8)'
        old_log='decode_warps=4 decode_stages=1 segments=16; layer=%s'
        new_log='decode_warps=4 decode_stages=1 segments=16 decode_layout=1 decode_rows=8; layer=%s'
        assert old.count(old_call)==1 and old.count(old_log)==1
        new=old.replace(old_call,new_call).replace(old_log,new_log)
        assert new.replace(new_call,old_call).replace(new_log,old_log)==old
        tree=ast.parse(new)
        call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='paged_gqa')
        expected={'segments':16,'decode_warps':4,'decode_stages':1,'decode_layout':1,'decode_rows':8}
        assert {k.arg:ast.literal_eval(k.value) for k in call.keywords}==expected
        p.write_text(new)
        assert subprocess.run(['git','-C',str(WT/'vllm-plugin-FL'),'grep','-n','KVReservationAsyncScheduler'],capture_output=True).returncode==1
        subprocess.run(['python3','-m','py_compile',str(p)],check=True)
        git(WT/'vllm-plugin-FL','add',rel);git(WT/'vllm-plugin-FL','diff','--cached','--check')
        print(git(WT/'vllm-plugin-FL','commit','-m','perf(iluvatar): select paged KV broadcast and eight-row GQA decode'),flush=True)
        commits={'vllm-plugin-FL':git(WT/'vllm-plugin-FL','rev-parse','HEAD'),'FlagGems':GEMS}
        write(EXP/'candidate_commits.json',commits)
        for name in ('run_split_e2e.py','launch_split.py','finish_split.py','queue_official.py','analyze_schedule.py','build_scripts.py','prepare_integration.py'):
            shutil.copy2(STAGING/name,EXP/name)
        shutil.copytree(STAGING/'parent_exp12',EXP/'parent_exp12')
        step('Freeze engine-only explicit parameters; inverse substitution matches parent exactly',commits=commits)
        gate=dict(status='passed',candidate_commits=commits,launch=expected,
                  diagnostic_hashes={str(p):sha(p) for p in (SCREEN/'layout1_rows8.json',EXT/'validation.json')},
                  code_hashes={'vllm-plugin-FL':{rel:sha(WT/'vllm-plugin-FL'/rel)},
                    'FlagGems':{'src/flag_gems/runtime/backend/_iluvatar/fused/paged_gqa.py':sha(WT/'FlagGems/src/flag_gems/runtime/backend/_iluvatar/fused/paged_gqa.py')}},
                  validation_scope='EXP13 26 numerical/Graph checks and 34 timings; full task accuracy deferred',
                  integration_cpu_check='Only explicit layout/row kwargs and runtime log changed; reverse substitutions match parent exactly')
        write(EXP/'candidate_validation.json',gate)
        finish_preparation(commits,old_log,new_log)
    except BaseException as exc:
        step('Integration setup failed; preserve files and diagnose before resuming',error=repr(exc))
        raise

if __name__=='__main__':main()
