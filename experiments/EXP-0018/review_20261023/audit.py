"""Independent CPU audit of EXP18 raw evidence; never launches GPU work."""
from collections import Counter, defaultdict
from datetime import datetime
import base64
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT=Path('/workspace/LeanScale')
REPO=ROOT/'MiniCPM-Infer-Sys-Opt'
EXP=REPO/'experiments/EXP-0018'
RUN=ROOT/'artifacts/EXP-0018/20261023_132558_profile_4k'
OUT=Path(sys.argv[1])
COMMITS={'vllm-plugin-FL':'588a6147cd291e272324ab6c812024cd587c6287','FlagGems':'fd209df717648002ebf5ba4ec351424dcc997a53'}
MODES=('FULL_DECODE_ONLY','NONE')
WINDOWS=('initial','mixed','large_decode','tail')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def csv_out(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

assert not OUT.exists(),'Use a new audit destination; do not overwrite a prior audit'
OUT.mkdir(parents=True)
try:
    status=read(RUN/'status.json');pub=read(RUN/'publication.json');provenance=read(RUN/'provenance.json')
    assert status['stage']=='complete' and status['owned_children_reaped']
    assert pub['remote_verified'] and pub['commit']=='a4fa36f3df04ca2db6447bd6999f4a17d0bcbfc4'
    assert provenance['commits']==COMMITS
    for name,commit in COMMITS.items():
        tree=ROOT/'worktrees/EXP-0018'/name
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip()==commit
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=tree,text=True).strip()
    for name,digest in read(EXP/'script_manifest.json').items():assert sha(EXP/name)==digest,name
    assert sha(Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py'))==provenance['benchmark_sha256']
    for rel,digest in provenance['installed_hashes'].items():assert sha(Path('/usr/local/lib/python3.12/site-packages/vllm')/rel)==digest
    archive=EXP/'artifacts'/RUN.name
    analysis=read(RUN/'analysis.json')
    assert read(archive/'analysis.json')==analysis
    assert len(analysis['traces'])==8 and all(c['qualifying_steps']==16 for c in analysis['coverage'])
    spec=importlib.util.spec_from_file_location('exp18_route_audit',EXP/'route_checks.py')
    route=importlib.util.module_from_spec(spec);spec.loader.exec_module(route)
    workloads=[];routes=[];traces=[];operator_rows=[];shape_rows=[];cpu_candidates=[]
    for mode in MODES:
        folder=RUN/mode
        text=(folder/'client.log').read_text()
        expected={'Successful requests':256,'Failed requests':0,'Total input tokens':1048576,'Total generated tokens':262144}
        for label,value in expected.items():
            match=re.findall(r'^'+re.escape(label)+r':\s*(\d+)\s*$',text,re.M)
            assert match==[str(value)],(mode,label,match)
        workloads.append(dict(mode=mode,successful=256,failed=0,input_tokens=1048576,output_tokens=262144,
                              client_log_sha256=sha(folder/'client.log'),server_log_sha256=sha(folder/'server.log')))
        assert read(folder/'workload_validation.json')['actual_profiled_batch_steps']==64
        with tempfile.TemporaryDirectory(prefix='exp18_independent_route_') as t:
            out=Path(t)/'route.json';route.verify_route(folder,mode,out)
            assert read(out)==read(folder/'route_evidence.json')
        routes.append(dict(mode=mode,kernel_counts=read(folder/'route_evidence.json')['kernel_counts'],
                           graph_sources=len(read(folder/'route_evidence.json')['graphs']),
                           route_evidence_sha256=sha(folder/'route_evidence.json')))
        metadata=[json.loads(x) for x in (folder/'step_metadata.jsonl').read_text().splitlines()]
        records={}
        for m in metadata:records.setdefault(m['step'],{}).update(m)
        assert len(records)==64
        assert read(folder/'latest_batch.json')['new_requests_total']==256
        for window in WINDOWS:
            files=list((folder/'traces'/window).glob('*.pt.trace.json*'));assert len(files)==1
            p=files[0];opener=gzip.open if p.suffix=='.gz' else open
            with opener(p,'rt') as f:events=json.load(f)['traceEvents']
            kernels=[e for e in events if e.get('ph')=='X' and e.get('dur',0)>=0 and 'kernel' in e.get('cat','').lower()]
            meta={}
            for e in events:
                if e.get('name','').startswith('LS_META64:'):
                    m=json.loads(base64.urlsafe_b64decode(e['name'].split(':',1)[1]));meta.setdefault(m['step'],{}).update(m)
            report=next(x for x in analysis['traces'] if x['mode']==mode and x['window']==window)
            assert sha(p)==report['trace_sha256']
            assert len(kernels)==report['kernel_events']
            assert math.isclose(sum(e['dur'] for e in kernels),report['kernel_sum_us'],rel_tol=1e-12)
            assert len(meta)==16
            assert all(records[step]==m for step,m in meta.items())
            exported=read(RUN/'summary'/mode/window/'0/step_metadata.json')
            assert {str(k):v for k,v in meta.items()}==exported
            for step,m in meta.items():
                q=m['query_lengths'];l=m['context_lengths_cpu_upper_bound'];c=m['computed_tokens_cpu_upper_bound'];ph=m['request_phases']
                assert m['B']==len(q)==len(l)==len(c)==len(ph)
                assert sum(q)==m['M_unpadded'] and m['M_padded']>=sum(q)
                assert all(a+b==z for a,b,z in zip(c,q,l))
                assert ph==['prefill' if v<4096 else 'decode' for v in c]
                assert m['prefill_requests']==ph.count('prefill')
                assert m['phase']==('prefill' if all(x=='prefill' for x in ph) else 'decode' if all(x=='decode' for x in ph) else 'mixed')
                assert m['main_requests_entered']<=256
                if window=='mixed':assert m['phase']=='mixed'
                if window=='large_decode':assert m['phase']=='decode' and m['B']>=48
                if window=='tail':assert m['phase']=='decode' and m['main_requests_entered']==256 and 0<m['B']<=16
                assert m['graph_mode']==('FULL' if mode=='FULL_DECODE_ONLY' and window in ('large_decode','tail') else 'NONE')
            mapped=read(RUN/'summary'/mode/window/'0/kernel_mapping.json')
            assert len(mapped)==len(kernels)
            for e,m in zip(kernels,mapped):assert e['name']==m['name'] and e['dur']==m['duration_us']
            groups=defaultdict(list)
            for m in mapped:
                groups[(m['phase'],m['op'])].append(m['duration_us'])
                if m['step'] is not None:assert m['phase']==meta[m['step']]['phase']
            for (phase,op),v in groups.items():
                operator_rows.append(dict(mode=mode,window=window,phase=phase,operator=op,calls=len(v),
                     mean_us=sum(v)/len(v),total_us=sum(v),percent_visible_kernel_sum=100*sum(v)/report['kernel_sum_us']))
            full_model_visible=mode=='NONE' or window in ('initial','mixed')
            if full_model_visible:
                for step,m in meta.items():
                    counts=Counter(x['op'] for x in mapped if x['step']==step)
                    for op in ['OP03_QKV','OP08_O','OP09_gate_up','OP11_down','OP04_RoPE_KV','OP06_GQA_decode_branch','OP07_GQA_reduce']:
                        # Pure prefill has no useful decode branch, but both
                        # dispatches still launch and early-return per program.
                        assert counts[op]==42,(mode,window,step,op,counts)
                    if m['phase']!='decode':assert counts['OP05_GQA_prefill_branch']==42
            else:
                assert not any(x['op'] in ('OP03_QKV','OP06_GQA_decode_branch') for x in mapped)
            shapes={}
            for step,m in meta.items():
                key=(m['B'],m['M_padded'])
                item=shapes.setdefault(key,dict(mode=mode,window=window,B=key[0],M_padded=key[1],steps=0,
                    context_min_cpu_upper_bound=999999,context_max_cpu_upper_bound=0,down_us=0.0,down_calls=0,gate_up_us=0.0,gate_up_calls=0))
                item['steps']+=1;item['context_min_cpu_upper_bound']=min(item['context_min_cpu_upper_bound'],min(m['context_lengths_cpu_upper_bound']))
                item['context_max_cpu_upper_bound']=max(item['context_max_cpu_upper_bound'],max(m['context_lengths_cpu_upper_bound']))
                for x in mapped:
                    if x['step']==step:
                        prefix={'OP11_down':'down','OP09_gate_up':'gate_up'}.get(x['op'])
                        if prefix:item[prefix+'_us']+=x['duration_us'];item[prefix+'_calls']+=1
            shape_rows.extend(shapes.values())
            names=Counter(e['name'] for e in events if e.get('ph')=='X' and re.search('compile|autotun',e.get('name',''),re.I))
            cpu_candidates.append(dict(mode=mode,window=window,names=dict(names),interpretation='Name matches are candidates, not confirmed compile events'))
            traces.append(dict(mode=mode,window=window,sha256=sha(p),steps=len(meta),phase_counts=dict(Counter(m['phase'] for m in meta.values())),
                batch_sizes=sorted({m['B'] for m in meta.values()}),padded_tokens=sorted({m['M_padded'] for m in meta.values()}),
                kernel_events=len(kernels),kernel_sum_us=report['kernel_sum_us'],model_visible=full_model_visible,
                unlinked_kernels=sum(x['step'] is None for x in mapped)))
    expected_raw=read(archive/'raw_evidence_hashes.json')
    actual_raw=[]
    for mode in MODES:
        for p in sorted((RUN/mode).rglob('*')):
            if p.is_file() and ('traces' in p.parts or p.name in ('server.log','client.log','step_metadata.jsonl','batch_timeline.jsonl')):
                actual_raw.append(dict(mode=mode,kind='trace' if 'traces' in p.parts else p.name,
                  window=p.parent.name if 'traces' in p.parts else None,sha256=sha(p),bytes=p.stat().st_size))
    assert expected_raw==actual_raw
    csv_out(OUT/'operator_summary.csv',operator_rows);csv_out(OUT/'shape_summary.csv',shape_rows)
    write(OUT/'compile_name_candidates.json',cpu_candidates)
    audit=dict(status='passed',audited_at=datetime.now().astimezone().isoformat(),run=RUN.name,publication=pub,
        commits=COMMITS,workloads=workloads,routes=routes,traces=traces,raw_evidence_files=len(actual_raw),
        diagnostic_only=True,performance_comparison_allowed=False,accuracy='Full task accuracy deferred',
        coverage_note='128 recorded steps across 8 targeted windows; not whole-run attribution',
        graph_note='FULL decode Graph internals invisible; NONE is attribution-only',
        next_hypothesis='Prefill page-table broadcast and K-load layout; fixed selected launch parameters. Small-M projection is secondary.')
    write(OUT/'audit.json',audit)
    print(json.dumps(dict(status='passed',traces=len(traces),steps=128,raw_files=len(actual_raw),workloads=workloads,routes=routes)))
except BaseException:
    import traceback
    write(OUT/'audit_error.json',dict(error=traceback.format_exc()))
    raise
