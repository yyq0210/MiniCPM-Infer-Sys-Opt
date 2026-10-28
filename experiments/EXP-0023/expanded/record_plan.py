"""Publish the revised protocol after inspecting actual production graph code."""
from pathlib import Path
from datetime import datetime
import ast,fcntl,hashlib,json,shutil,subprocess
ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt';EXP=REPO/'experiments/EXP-0023/expanded';STAGING=Path(__file__).resolve().parent
RUN=ROOT/'artifacts/EXP-0020/20261026_125756_official_4k'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a],text=True,stderr=subprocess.STDOUT,timeout=120).strip()
assert not EXP.exists()
graph=next((RUN/'candidate_cache').rglob('computation_graph.py'));s=graph.read_text()
assert s.count('torch.nn.functional.silu(')==42 and 'activation.py:141 in forward_native' in s
assert 'triton_poi_fused' not in s
seen={}
for p in (RUN/'candidate_cache').rglob('*.py'):
 text=p.read_text(errors='replace')
 if 'async_compile.triton' not in text or 'silu' not in text:continue
 for n in ast.parse(text).body:
  if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='triton' and n.value.args and isinstance(n.value.args[0],ast.Constant) and 'silu' in str(n.value.args[0].value):
   source=n.value.args[1].value;h=hashlib.sha256(source.encode()).hexdigest()
   seen.setdefault(h,dict(source=source,name=n.value.args[0].value,path=str(p),module_sha256=sha(p)))
assert list(seen)==['2077eeed3cb635af436273a4ca3c69e6fb2d32e303bbf50b739950086df2cb6a']
h,item=next(iter(seen.items()));assert item['name']=='triton_poi_fused_mul_silu_slice_1'
assert not any(l.strip().startswith('xnumel =') for l in item['source'].splitlines())
assert item['module_sha256']=='feaa3bf64672cb789a89946d63b026e3f063bfe80c1b1b4366be672aa543fe9c'
assert json.loads((ROOT/'artifacts/EXP-0023/review_publication.json').read_text())['remote_verified']
EXP.mkdir();shutil.copy2(STAGING/'README.md',EXP/'README.md')
(EXP/'steps.jsonl').write_text(json.dumps(dict(time=datetime.now().astimezone().isoformat(),action='Record revised hypothesis and fixed protocol before implementation; original CachedOp-log preflight failed with no GPU work; production graph shows native compiled SwiGLU'))+'\n')
write(EXP/'route_evidence.json',dict(graph_path=str(graph),graph_sha256=sha(graph),native_silu_count=42,kernel_name=item['name'],kernel_source_sha256=h,module_path=item['path'],module_sha256=item['module_sha256'],
 source_kind='curated single production-generated kernel source, not full cache',initial_preflight=dict(status='failed',reason='No silu_and_mul default.flagos first-use log; actual computation graph uses forward_native',GPU_started=False),status='actual graph and generated kernel verified; runtime execution pending'))
write(EXP/'production_silu_source.json',dict(source=item['source'],source_sha256=h))
(EXP/'record_plan.py').write_text(Path(__file__).read_text())
write(EXP/'daily_window.json',dict(date='2026-10-28',machine_shutdown_at='2026-10-29T00:00:00+08:00',compute_stop_at='2026-10-28T23:35:00+08:00',archive_target_at='2026-10-28T23:50:00+08:00',user_confirmed_shutdown=True))
with (REPO/'docs/optimization_journal.md').open('a') as f:f.write('\n\n### 2026-10-28：EXP23完整MLP路由核实与复测方案\n\n恢复后核实已选代码和历史结果保留。原CPU预检查因缺少SwiGLU的CachedOp使用日志停止，未启动GPU；实际服务计算图42次forward_native及唯一生成SwiGLU内核说明使用编译后原生路径，不能套用插件中存在但未生效的类。修正expanded/README，先登记117检查/180完整子图计时与固定门槛，再实现。仅提取实际生成激活核，不复制整缓存；新增今日23:35计算/23:50保存截止。\n')
with (ROOT/'artifacts/docs-publish.lock').open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX);assert not git('diff','--cached','--name-only')
 paths=['experiments/EXP-0023/expanded','docs/optimization_journal.md'];git('add','--',*paths);git('diff','--cached','--check');git('commit','--only','-m','experiments: register MLP refinement using actual compiled activation','--',*paths)
 head=git('rev-parse','HEAD');git('push','origin','main');assert git('ls-remote',git('remote','get-url','--push','origin'),'refs/heads/main').split()[0]==head
write(ROOT/'artifacts/EXP-0023/expanded_hypothesis_publication.json',dict(commit=head,remote_verified=True))
print(json.dumps(dict(commit=head,remote_verified=True,actual_activation='native forward compiled to production triton kernel',kernel_source_sha256=h)))
