"""Record a native-GEMM hypothesis and run one finite operator screen."""
from pathlib import Path
from datetime import datetime,timedelta
import ast
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
import traceback

ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt';EXP=REPO/'experiments/EXP-0022';ART=ROOT/'artifacts/EXP-0022'
STAGING=Path(__file__).resolve().parent
COMMITS={'vllm-plugin-FL':'ce19f93eee4c73f5a26911554700457e879f891d','FlagGems':'de958ddfc4deeb43d6c94bb8652a3d4d3bc905fe'}
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args],text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def step(action,**kw):
    with (EXP/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(),action=action,**kw),ensure_ascii=False)+'\n')

def main():
    assert not EXP.exists() and not ART.exists() and not (ROOT/'artifacts/EXP-0022-request.json').exists()
    assert read(ROOT/'artifacts/active_inference_plan.json')['selected_serving_commits']==COMMITS
    assert read(ROOT/'artifacts/EXP-0021/status.json')['stage']=='complete'
    assert read(ROOT/'artifacts/EXP-0021/selection_publication.json')['remote_verified']
    now=datetime.now().astimezone();assert now.strftime('%Y-%m-%d')=='2026-10-26'
    EXP.mkdir();ART.mkdir()
    (EXP/'README.md').write_text('''# EXP-0022：4K尾部小M原生投影与补零诊断

实现前假设：EXP18 NONE尾部B8–16中四投影合计44.54%设备kernel时间，down mean93.05us，而B64窗口58.75us；这是不同窗口/shape观察，不是收益。
生产FULL尾部实际padding M8/16。先保留已有原生库GEMM，检查输入补到16/32/64行能否改善小M执行效率，且收益必须覆盖补零成本。
本轮不恢复EXP6的自写norm+QKV或gateup+SwiGLU慢原型，不修改模型/调度/已选代码。

实际调用链：UnquantizedLinearMethod.apply → dispatch_unquantized_gemm → default_unquantized_gemm → torch.nn.functional.linear。
模型hidden2048、intermediate6144、Q16/KV2/head128；预期合并QKV[N,K]=[2560,2048]、O[2048,2048]、gate_up[12288,2048]、down[2048,6144]。
权重BF16；连续[N,K]布局根据create_weights源码推导并在合成张量中显式使用，尚未在新服务中探针记录实际weight stride。
只读checkpoint header验证分离权重shape/dtype，不读取或发布权重数据。随机权重/激活，不是模型请求重放。

固定M=[1,4,8,16,30,32,64,2048]，目标padding=[16,32,64]，仅target>M时测候选；M30是NONE诊断参照，不声称生产FULL使用M30。
完整候选路径为F.pad + 原库linear + 前M行视图，不只计GEMM。M8/16是预先声明的生产尾部主目标。
hot模式8次重用同权重；weights8模式每次轮换8个独立权重矩阵，观察缓存敏感性，不能等同于42层真实工作集。
每种模式旧/新/旧夹测，7样本×8次Graph调用，记录所有配置与退化点。64个基线曲线点、112个候选计时点。
56项BF16与16项FP16数值/Graph检查，共72项：FP32对照、与原库输出比较，改变input/weight后Graph replay与eager严格一致。
固定atol0.01/rtol0.02，TF32关闭；完整模型精度仍按用户暂缓。

初筛门槛：同一投影/M8或16/target在hot和weights8两模式均完整路径耗时下降至少5%，两次夹测漂移均<=3%。
通过只代表进入独立复测，之后仍需更完整数值边界/完整子图/真实引擎接入和正式E2E，不能直接切换代码。
无候选就记录本假设被淘汰；不删慢点、不改阈值挑结果。全部结果自动保存推送。
新40分钟窗口，child最多30分钟，受今日23:35计算截止、23:50保存约束；GPU锁/空闲/端口/代码与脚本哈希前后检查。
''')
    step('Record shape-specific native-GEMM padding hypothesis before installing experiment code',selected_commits=COMMITS)
    cfgpath=Path('/workspace/MiniCPM5-2B/config.json');cfg=read(cfgpath)
    for k,v in dict(hidden_size=2048,intermediate_size=6144,num_attention_heads=16,num_key_value_heads=2,head_dim=128,num_hidden_layers=42,torch_dtype='bfloat16').items():assert cfg[k]==v
    utils=Path('/usr/local/lib/python3.12/site-packages/vllm/model_executor/layers/utils.py')
    linear=utils.with_name('linear.py')
    tree=ast.parse(utils.read_text());default=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='default_unquantized_gemm')
    dispatch=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='dispatch_unquantized_gemm')
    assert 'torch.nn.functional.linear' in ast.get_source_segment(utils.read_text(),default)
    assert 'return default_unquantized_gemm' in ast.get_source_segment(utils.read_text(),dispatch)
    assert 'return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)' in linear.read_text()
    model=cfgpath.parent;mapping=read(model/'model.safetensors.index.json')['weight_map'];headers={};shapes={}
    for key,file in mapping.items():
        if key.startswith('model.layers.0.') and key.endswith('_proj.weight'):
            if file not in headers:
                with (model/file).open('rb') as f:
                    size=int.from_bytes(f.read(8),'little');assert size<100000000;raw=f.read(size)
                headers[file]=dict(data=json.loads(raw),sha256=hashlib.sha256(raw).hexdigest())
            meta=headers[file]['data'][key];assert meta['dtype']=='BF16'
            shapes[key]=dict(shape=meta['shape'],dtype=meta['dtype'])
    assert len(shapes)==7
    assert shapes['model.layers.0.mlp.down_proj.weight']['shape']==[2048,6144]
    for name in ['trial.py','run_screen.py','prepare_and_start.py']:
        shutil.copy2(STAGING/name,EXP/name);ast.parse((EXP/name).read_text(),filename=name)
    write(EXP/'source_evidence.json',dict(config={k:cfg[k] for k in ['hidden_size','intermediate_size','num_attention_heads','num_key_value_heads','head_dim','num_hidden_layers','torch_dtype']},
       first_layer_weights=shapes,header_hashes={k:v['sha256'] for k,v in headers.items()},
       source_derived_weight_layout='row-major contiguous[N,K]; not yet actual service stride probe',
       default_linear_source=ast.get_source_segment(utils.read_text(),default),dispatch_source=ast.get_source_segment(utils.read_text(),dispatch)))
    plan=dict(selected_commits=COMMITS,script_hashes={p.name:sha(p) for p in EXP.glob('*.py')},
       source_hashes={str(p):sha(p) for p in [utils,linear,cfgpath]},expected_checks=72,baseline_curve_points=64,timing_points=112,
       candidate_gate='M8/16,>=5percent time decrease and drift<=3percent in both hot and weights8; no automatic integration')
    write(EXP/'plan.json',plan);write(EXP/'STATUS.json',dict(stage='prepared',GPU_started=False))
    with (ROOT/'artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);assert not git('diff','--cached','--name-only')
        with (REPO/'docs/optimization_journal.md').open('a') as f:f.write('\n\n### 2026-10-26：EXP22小M投影诊断准备\n\n已选EXP20，开始检验4K尾部M8/16的原生GEMM补零假设。保留库GEMM、完整计入pad/linear/slice，hot与8权重轮换两模式，固定72检查/64曲线/112计时点。随机张量且布局源推导，不是模型重放，不预先宣称收益。详见experiments/EXP-0022/README.md。\n')
        paths=['experiments/EXP-0022','docs/optimization_journal.md'];git('add','--',*paths);git('diff','--cached','--check')
        assert '__pycache__' not in git('diff','--cached','--name-only')
        git('commit','--only','-m','experiments: prepare small-M native linear padding diagnosis','--',*paths)
        head=git('rev-parse','HEAD');git('push','origin','main')
        assert git('ls-remote',git('remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
    write(ART/'preparation.json',dict(status='prepared',commit=head,remote_verified=True))
    now=datetime.now().astimezone();assert now.strftime('%Y-%m-%d')=='2026-10-26'
    daily=read(REPO/'experiments/EXP-0020/launches/daily_20261026.json')
    end=min(now+timedelta(minutes=40),datetime.fromisoformat(daily['compute_stop_at']))
    assert (end-now).total_seconds()>35*60,'Prepared only; insufficient fresh screen window'
    policy=dict(experiment='EXP-0022',created_at=now.isoformat(),compute_stop_at=end.isoformat(),archive_target_at=(end+timedelta(minutes=10)).isoformat(),daily_window=daily)
    write(ART/'run_window.json',policy);step('Launch separate finite operator screen after published plan',policy=policy)
    with (ART/'controller.log').open('w') as log:
        child=subprocess.Popen(['python3',str(EXP/'run_screen.py')],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    latest=ROOT/'artifacts/EXP-0022-request.json'
    for _ in range(100):
        if latest.exists():break
        assert child.poll() is None;time.sleep(.2)
    else:raise RuntimeError('No screen receipt')
    receipt=read(latest);assert receipt['pid']==child.pid
    write(ART/'launch.json',dict(preparation_commit=head,receipt=receipt,policy=policy));print(json.dumps(dict(commit=head,receipt=receipt,policy=policy)),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        if EXP.exists():step('Preparation/start failed; retain files and do not blindly rerun',error=traceback.format_exc())
        write(ROOT/'artifacts/EXP-0022-preparation-error.json',dict(error=traceback.format_exc()));raise
