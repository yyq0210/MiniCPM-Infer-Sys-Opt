"""CPU-only regression checks for attribution, hooks and truthful window gates."""
import base64
import contextlib
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types

from analyze_profile import analyze_trace, category, union_us
from run_profile import choose_window, validate_window

CHECKS=[]
def passed(name): CHECKS.append(name)

assert union_us([(0,5),(3,8),(9,12)])==11
assert category('vendor_gemm', [[[64,2048],[2048,2560]]])[0]=='OP03_QKV'
assert category('vendor_gemm', [])[0]=='GEMM_unresolved_projection'
passed('overlap_union_and_conservative_shape_mapping')

def event(name,cat,ts,dur,args=None,tid=1):
    return dict(name=name,cat=cat,ph='X',ts=ts,dur=dur,pid=1,tid=tid,args=args or {})
payload=base64.urlsafe_b64encode(b'{"step":7,"kind":"batch","phase":"decode","B":64}').decode()
events=[event('LS_STEP:7','user_annotation',0,100),event('LS_META64:'+payload,'user_annotation',1,1),
        event('aten::mm','cpu_op',5,80,{'External id':5,'Input Dims':[[64,2048],[2048,2560]]}),
        event('cudaLaunchKernel','cuda_runtime',10,3,{'correlation':99,'External id':5}),
        event('vendor_gemm','kernel',110,20,{'correlation':99},2),
        event('unknown','kernel',40,5,{'correlation':777},2),
        event('LS_STEP:8','user_annotation',200,100),
        event('aten::linear','cpu_op',205,80,{'Input Dims':[[64,2048],[130560,2048]]}),
        event('cudaLaunchKernel','cuda_runtime',210,3,{'correlation':100}),
        event('vendor_gemm','kernel',300,7,{'correlation':100},2),
        event('LS_SAMPLE:8','user_annotation',400,100),
        event('cudaLaunchKernel','cuda_runtime',410,3,{'correlation':101}),
        event('random_kernel','kernel',500,8,{'correlation':101},2),
        event('_attention','kernel',600,9,{'grid':[64,2,16]},2),
        event('_attention','kernel',700,10,{'grid':[256,2,1]},2)]
with tempfile.TemporaryDirectory() as d:
    p=Path(d);trace=p/'trace.json';trace.write_text(json.dumps(dict(traceEvents=events)));out=p/'out';out.mkdir()
    report=analyze_trace(trace,out)
    assert report['kernel_events']==6 and report['kernel_sum_us']==59
    assert report['correlations']==dict(linked_to_step=3,unlinked_to_step=3)
    rows=json.loads((out/'kernel_mapping.json').read_text())
    assert rows[0]['op']=='OP03_QKV' and rows[0]['phase']=='decode'
    assert rows[1]['step'] is None and rows[1]['phase']=='unlinked'
    assert rows[2]['op']=='OP12_LM_head'
    assert rows[3]['op']=='OP13_sampling'
    assert rows[4]['op']=='OP06_GQA_decode_branch' and rows[5]['op']=='OP05_GQA_prefill_branch'
    import csv
    cpu=list(csv.DictReader((out/'cpu_operator_times.csv').open()))
    mm=next(x for x in cpu if x['operator']=='aten::mm')
    assert float(mm['self_cpu_us'])==77 and float(mm['inclusive_cpu_us'])==80
    assert all(b'\r\n' not in f.read_bytes() for f in out.glob('*.csv'))
    passed('GPU_correlation_without_timestamp_guessing')
    passed('CPU_nested_self_time_not_inclusive_sum')
    passed('linear_weight_orientation_sampling_scope_and_S16_grid')
    passed('derived_csv_LF')

base=dict(main_workload_started=True,window='initial',window_profiled_steps=16,
          epoch=100,profiling=False,phase='decode',B=64,new_requests_total=64)
assert choose_window(base,{'initial'},'initial',100)=='large_decode'
assert choose_window(dict(base,phase='mixed'),{'initial'},'initial',100)=='mixed'
assert choose_window(dict(base,B=8,new_requests_total=256),{'initial'},'initial',100)=='tail'
for override in [dict(main_workload_started=False),dict(window='tail'),dict(window_profiled_steps=15),
                 dict(profiling=True),dict(epoch=90)]:
    assert choose_window(dict(base,**override),{'initial'},'initial',100) is None
assert choose_window(dict(base,B=8,new_requests_total=255),{'initial'},'initial',100) is None
assert choose_window(dict(base,B=8,new_requests_total=258),{'initial'},'initial',100) is None
passed('window_gates_reject_warmup_stale_overlap_incomplete_and_false_tail')
now=datetime.fromisoformat('2026-10-23T13:00:00+08:00')
assert validate_window(dict(compute_stop_at='2026-10-23T23:35:00+08:00'),now)>0
for cutoff in ['2026-10-22T23:35:00+08:00','2026-10-24T23:35:00+08:00','2026-10-23T12:00:00+08:00']:
    try:validate_window(dict(compute_stop_at=cutoff),now)
    except AssertionError:pass
    else:raise AssertionError('Accepted invalid dated window')
passed('expired_wrong_date_cutoffs_rejected')

import numpy as np
with tempfile.TemporaryDirectory() as d:
    root=Path(d);os.environ['LEANSCALE_PROFILE_PASS']=str(root)
    # No torch.cuda API is provided: any GPU metadata access fails this test.
    sys.modules['torch']=types.SimpleNamespace(profiler=types.SimpleNamespace(record_function=lambda n:contextlib.nullcontext()))
    class FakeWorker:
        def profile(self,is_start=True,profile_prefix=None):return is_start
        def sample_tokens(self,*args,**kwargs):return 'sample_output'
    sys.modules['vllm_fl.worker.worker']=types.SimpleNamespace(WorkerFL=FakeWorker)
    spec=importlib.util.spec_from_file_location('hooks_test',Path(__file__).with_name('profile_hooks.py'))
    hooks=importlib.util.module_from_spec(spec);spec.loader.exec_module(hooks)
    output=object();padding=('FULL',types.SimpleNamespace(num_tokens=272,num_reqs=3))
    batch=types.SimpleNamespace(num_reqs=3,num_computed_tokens_cpu=np.array([4096,0,4097]),num_prompt_tokens=np.array([4096]*3))
    runner=types.SimpleNamespace(input_batch=batch,_prepare_inputs=lambda a,b:output,_determine_batch_execution_and_padding=lambda **kw:padding)
    worker=FakeWorker();worker.model_runner=runner
    fake=lambda active:types.SimpleNamespace(_running=active,step=lambda:None,stop=lambda:None)
    worker.profiler=fake(False);worker.profiler_config=types.SimpleNamespace(torch_profiler_dir='')
    scheduled=types.SimpleNamespace(scheduled_new_reqs=[object(),object()])
    with worker.annotate_profile(scheduled):assert runner._prepare_inputs(scheduled,np.array([1,256,1])) is output
    assert hooks.CURRENT['new_requests_total']==2
    (root/'window_control.json').write_text('{"window":"initial"}')
    assert worker.profile() is True and hooks.CURRENT['new_requests_total']==0
    worker.profiler=fake(True)
    with worker.annotate_profile(scheduled):
        assert runner._prepare_inputs(scheduled,np.array([1,256,1])) is output
        assert runner._determine_batch_execution_and_padding() is padding
    meta=[json.loads(s) for s in (root/'step_metadata.jsonl').read_text().splitlines() if json.loads(s)['kind']=='batch'][0]
    assert meta['phase']=='mixed' and meta['M_unpadded']==258
    assert meta['query_lengths']==[1,256,1] and meta['context_lengths_cpu_upper_bound']==[4097,256,4098]
    assert meta['request_phases']==['decode','prefill','decode'] and meta['main_requests_entered']==2
    (root/'window_control.json').write_text('{"window":"mixed"}');worker.profile()
    assert hooks.CURRENT['new_requests_total']==2,'Do not reset main counter at later windows'
    worker.profiler=fake(True)
    batch.num_computed_tokens_cpu=np.array([4096,4100,4097])
    with worker.annotate_profile(types.SimpleNamespace(scheduled_new_reqs=[])):
        runner._prepare_inputs(types.SimpleNamespace(scheduled_new_reqs=[]),np.array([1,1,1]))
        assert hooks.CURRENT['metadata']['phase']=='decode'
    assert worker.sample_tokens()=='sample_output' and not hooks.CURRENT['active']
    passed('CPU_only_hook_output_identity_and_mixed_decode_metadata')
    passed('warmup_counter_reset_once_before_main_workload')
    passed('per_request_context_query_and_phase_metadata')
print(json.dumps(dict(status='passed',checks=CHECKS)))
