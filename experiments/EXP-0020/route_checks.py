"""Exercise the route gate against real IR and controlled CPU mutations."""
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile

ROOT=Path('/workspace/LeanScale');REPO=ROOT/'MiniCPM-Infer-Sys-Opt'
EXP=Path(__file__).resolve().parent
OLD_LOG='decode_warps=4 decode_stages=1 segments=16 decode_layout=1 decode_rows=8 prefill_rows=64 prefill_warps=4 prefill_stages=1 prefill_tile=32'
NEW_LOG=OLD_LOG+' prefill_layout=1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    sys.path.insert(0,str(EXP))
    spec=importlib.util.spec_from_file_location('exp20_gate',EXP/'run_split_e2e.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    ref=ROOT/'artifacts/EXP-0016/20261022_145853_official_4k'
    screen=ROOT/'artifacts/EXP-0019/20261023_144047_screen'
    original=(ref/'candidate_benchmark/server.log').read_text()
    fixture=original.replace(OLD_LOG,NEW_LOG);assert fixture.count(NEW_LOG)==42
    graph=next(p for p in (ref/'candidate_cache').rglob('computation_graph.py') if 'torch.ops.vllm.fl_rope_kv_dispatch' in p.read_text())
    decode=next(p for p in (ref/'triton_cache').glob('*/_attention.json') if read(p).get('shared')==18432 and read(p).get('num_stages')==1)
    expected=read(screen/'layout1.json')['timings'][0]['resources']['ir_sha256']['ttir']
    prefill=next(p for p in (screen/'triton_cache').rglob('_attention.json') if p.with_suffix('.ttir').is_file() and sha(p.with_suffix('.ttir'))==expected)
    checks=[]
    with tempfile.TemporaryDirectory(prefix='exp20_gate_') as tmp:
        tmp=Path(tmp);mod.RUN=tmp;folder=tmp/'candidate_benchmark';folder.mkdir()
        target=tmp/'candidate_cache/fixture/computation_graph.py';target.parent.mkdir(parents=True);shutil.copy2(graph,target)
        metas={}
        for kind,p in [('decode',decode),('prefill',prefill)]:
            dst=tmp/'triton_cache'/kind/'_attention.json';dst.parent.mkdir(parents=True)
            shutil.copy2(p,dst);shutil.copy2(p.with_suffix('.ttir'),dst.with_suffix('.ttir'));metas[kind]=dst
        def check(name,expected):
            try:mod.verify_route(folder);ok=True
            except AssertionError:ok=False
            assert ok==expected,name
            checks.append(dict(name=name,passed=True))
        for name,log,passed in [
          ('positive_distinct_decode_prefill',fixture,True),('reject_parent_log',original,False),
          ('reject_withdrawn_scheduler',fixture+'\nKVReservationAsyncScheduler\n',False),
          ('reject_missing_layer',fixture.replace('LeanScale GQA active:','missing:',1),False),
          ('reject_wrong_prefill_layout',fixture.replace('prefill_layout=1','prefill_layout=0',1),False),
          ('reject_wrong_prefill_stage',fixture.replace('prefill_stages=1','prefill_stages=2',1),False),
          ('reject_wrong_decode_layout',fixture.replace('decode_layout=1','decode_layout=0',1),False)]:
            (folder/'server.log').write_text(log);check(name,passed)
        (folder/'server.log').write_text(fixture)
        for kind,p in metas.items():
            text=p.read_text();p.unlink();check('reject_missing_'+kind,False);p.write_text(text)
            for key,value in [('num_stages',2),('num_warps',8),('shared',read(p)['shared']+2048)]:
                bad=json.loads(text);bad[key]=value;p.write_text(json.dumps(bad))
                check('reject_'+kind+'_'+key,False);p.write_text(text)
        for kind,old,new in [
          ('decode','8x128x!tt.ptr<bf16>','16x128x!tt.ptr<bf16>'),
          ('prefill','64x128x!tt.ptr<bf16>','32x128x!tt.ptr<bf16>'),
          ('prefill','128x32x!tt.ptr<bf16>','128x64x!tt.ptr<bf16>'),
          *[(k,'2x!tt.ptr<i32>','32x!tt.ptr<i32>') for k in ('decode','prefill')],
          *[(k,'tensor<2x1xi64> -> tensor<2x16xi64>','tensor<2x1xi32> -> tensor<2x16xi32>') for k in ('decode','prefill')]]:
            p=metas[kind].with_suffix('.ttir');text=p.read_text();bad=text.replace(old,new)
            assert bad!=text;p.write_text(bad);check('reject_'+kind+'_'+old,False);p.write_text(text)
        for case,cases in [('4k',{(4096,1024,64,256)}),('16k',{(16384,1024,64,128)})]:
            mod.verified_parent_reference(case,cases);checks.append(dict(name='parent_complete_'+case,passed=True))
        mod.verify_candidate_gate();checks.append(dict(name='frozen_candidate_and_diagnostics',passed=True))
    (EXP/'route_cpu_checks.json').write_text(json.dumps(dict(status='passed',checks=checks,
       scope='CPU mutations of real decode/prefill IR; new actual service evidence remains pending'),indent=2)+'\n')
    print('CPU route checks:',len(checks))

if __name__=='__main__':main()
