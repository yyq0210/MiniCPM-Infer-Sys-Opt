"""Native linear padding screen, including padding cost and weight cycling."""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import statistics
import time
import traceback
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm

SHAPES={'qkv':(2560,2048),'o':(2048,2048),'gate_up':(12288,2048),'down':(2048,6144)}
MS=[1,4,8,16,30,32,64,2048]
TARGETS=[16,32,64]
ATOL=1e-2;RTOL=2e-2

def timed(fn,cycling):
    for i in range(8):fn(i if cycling else 0)
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        values=[fn(i if cycling else 0) for i in range(8)]
    g.replay();torch.cuda.synchronize();samples=[]
    for _ in range(7):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();g.replay();b.record();b.synchronize();samples.append(a.elapsed_time(b)*1000/8)
    return dict(samples_us=samples,median_us=statistics.median(samples),calls_per_replay=8)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    handler=dispatch_unquantized_gemm();source=inspect.getsource(handler)
    assert handler.__name__=='default_unquantized_gemm' and 'torch.nn.functional.linear' in source
    assert F.linear.__module__=='torch._C._nn','Unexpected global linear replacement'
    result=dict(status='running',kind='operator_diagnostic_not_official_e2e',started_at=time.strftime('%FT%T%z'),
      handler=handler.__module__+'.'+handler.__name__,handler_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
      tolerance=dict(atol=ATOL,rtol=RTOL),shapes=SHAPES,Ms=MS,targets=TARGETS,checks=[],curves=[],timings=[],
      caveat='Random activations/weights, row-major source-derived layout; hot-weight and8-weight cycling proxies, not model replay')
    def save():
        tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(a.output)
    def validate(name,x,w,target,dtype):
        base=handler(None,x,w,None)
        def call():return handler(None,F.pad(x,(0,0,0,target-x.shape[0])),w,None)[:x.shape[0]]
        y=call();ref=F.linear(x.float(),w.float())
        torch.testing.assert_close(base.float(),ref,atol=ATOL,rtol=RTOL)
        torch.testing.assert_close(y.float(),ref,atol=ATOL,rtol=RTOL)
        torch.testing.assert_close(y,base,atol=ATOL,rtol=RTOL)
        first=dict(candidate_fp32_max_abs=(y.float()-ref).abs().max().item(),candidate_baseline_max_abs=(y-base).abs().max().item())
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=call()
        x.mul_(.97);w.add_(.0001)
        expected=call();ref=F.linear(x.float(),w.float());graph.replay();torch.cuda.synchronize()
        torch.testing.assert_close(captured,expected,atol=0,rtol=0)
        torch.testing.assert_close(captured.float(),ref,atol=ATOL,rtol=RTOL)
        assert captured.shape==(x.shape[0],w.shape[0]) and captured.is_contiguous()
        result['checks'].append(dict(projection=name,dtype=str(dtype),M=x.shape[0],target=target,
            graph_mutations=['input','weight'],graph_exact_replay=True,mutated_fp32_max_abs=(captured.float()-ref).abs().max().item(),**first));save()
    torch.manual_seed(20261026);torch.backends.cuda.matmul.allow_tf32=False
    try:
        for name,(n,k) in SHAPES.items():
            # Eight independent weights exceed the one-weight working set. This
            # is a declared cache-sensitivity proxy, not a full42-layer replay.
            weights=[torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.02 for _ in range(8)]
            for m in MS:
                x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
                baseline=lambda i:handler(None,x,weights[i],None)
                for mode,cycle in [('hot',False),('weights8',True)]:
                    before=timed(baseline,cycle);after=timed(baseline,cycle)
                    result['curves'].append(dict(projection=name,M=m,mode=mode,baseline_before=before,baseline_after=after,
                        baseline_drift_percent=(after['median_us']/before['median_us']-1)*100));save()
                for target in TARGETS:
                    if target<=m:continue
                    validate(name,x,weights[0],target,torch.bfloat16)
                    padded=lambda i:handler(None,F.pad(x,(0,0,0,target-m)),weights[i],None)[:m]
                    for mode,cycle in [('hot',False),('weights8',True)]:
                        before=timed(baseline,cycle);new=timed(padded,cycle);after=timed(baseline,cycle)
                        change=(new['median_us']/((before['median_us']+after['median_us'])/2)-1)*100
                        drift=(after['median_us']/before['median_us']-1)*100
                        result['timings'].append(dict(projection=name,M=m,target=target,mode=mode,
                          baseline_before=before,candidate=new,baseline_after=after,time_change_percent=change,baseline_drift_percent=drift,
                          complete_padding_linear_slice=True,production_tail_M=m in (8,16)))
                        save();print(json.dumps(result['timings'][-1]),flush=True)
                del x;torch.cuda.empty_cache()
            del weights;torch.cuda.empty_cache()
        # Independent FP16 numerical/Graph regressions; production is BF16.
        for name,(n,k) in SHAPES.items():
            w=torch.randn(n,k,device='cuda',dtype=torch.float16)*.02
            for m in (8,16):
                for target in (32,64):
                    x=torch.randn(m,k,device='cuda',dtype=torch.float16)
                    validate(name,x,w,target,torch.float16);del x
            del w;torch.cuda.empty_cache()
        assert len(result['checks'])==72 and len(result['curves'])==64 and len(result['timings'])==112
        result['status']='passed'
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));traceback.print_exc();raise
    finally:result['finished_at']=time.strftime('%FT%T%z');save()

if __name__=='__main__':main()
