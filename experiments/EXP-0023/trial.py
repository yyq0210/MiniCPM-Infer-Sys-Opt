"""Native GEMM weight-layout screen; packing is performed once, outside calls."""
from pathlib import Path
import argparse,hashlib,inspect,json,statistics,time,traceback
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm

SHAPES={'qkv':(2560,2048),'o':(2048,2048),'gate_up':(12288,2048),'down':(2048,6144)}
MS=[1,4,8,16,30,32,64,128,512,2048]
ATOL=.01;RTOL=.02

def timed(fn,cycle):
    for i in range(8):fn(i if cycle else 0)
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):values=[fn(i if cycle else 0) for i in range(8)]
    g.replay();torch.cuda.synchronize();samples=[]
    for _ in range(7):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();g.replay();b.record();b.synchronize();samples.append(a.elapsed_time(b)*1000/8)
    return dict(samples_us=samples,median_us=statistics.median(samples),calls_per_replay=8)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    handler=dispatch_unquantized_gemm();source=inspect.getsource(handler)
    assert handler.__name__=='default_unquantized_gemm' and 'torch.nn.functional.linear' in source
    assert F.linear.__module__=='torch._C._nn'
    result=dict(status='running',kind='operator_diagnostic_not_official_e2e',started_at=time.strftime('%FT%T%z'),
      handler=handler.__module__+'.'+handler.__name__,handler_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
      shapes=SHAPES,Ms=MS,tolerance=dict(atol=ATOL,rtol=RTOL),checks=[],timings=[],packing=[],operator_events=[],
      caveat='Random tensors, source-derived original layout; no actual model stride probe; 8-weight cache proxy, not 42-layer replay; CPU events cannot exclude internal library copies')
    def save():
        tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(args.output)
    def validate(name,x,w,packed):
        n,k=w.shape
        assert w.stride()==(k,1) and packed.stride()==(1,n) and torch.equal(w,packed)
        base=handler(None,x,w,None);candidate=handler(None,x,packed,None);ref=F.linear(x.float(),w.float())
        torch.testing.assert_close(base.float(),ref,atol=ATOL,rtol=RTOL)
        torch.testing.assert_close(candidate.float(),ref,atol=ATOL,rtol=RTOL)
        torch.testing.assert_close(candidate,base,atol=ATOL,rtol=RTOL)
        errors=dict(candidate_fp32_max_abs=(candidate.float()-ref).abs().max().item(),candidate_baseline_max_abs=(candidate-base).abs().max().item())
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=handler(None,x,packed,None)
        x.mul_(.97);w.add_(.0001);packed.add_(.0001)
        assert torch.equal(w,packed)
        eager=handler(None,x,packed,None);ref=F.linear(x.float(),w.float());graph.replay();torch.cuda.synchronize()
        torch.testing.assert_close(captured,eager,atol=0,rtol=0)
        torch.testing.assert_close(captured.float(),ref,atol=ATOL,rtol=RTOL)
        assert captured.shape==(x.shape[0],n) and captured.is_contiguous()
        result['checks'].append(dict(projection=name,M=x.shape[0],dtype=str(x.dtype),original_stride=list(w.stride()),candidate_stride=list(packed.stride()),
           graph_mutations=['input','weight'],graph_exact_replay=True,mutated_fp32_max_abs=(captured.float()-ref).abs().max().item(),**errors));save()
    def events(name,m,layout,fn):
        fn();torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            value=fn();torch.cuda.synchronize()
        rows=[dict(name=e.key,count=e.count) for e in prof.key_averages()]
        result['operator_events'].append(dict(projection=name,M=m,layout=layout,events=rows,
           observed_copy_events=[x for x in rows if x['name'] in ['aten::clone','aten::contiguous','aten::copy_','aten::_to_copy']]))
        save()
    torch.manual_seed(20261026);torch.backends.cuda.matmul.allow_tf32=False
    try:
        for name,(n,k) in SHAPES.items():
            weights=[torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.02 for _ in range(8)]
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();memory_before=torch.cuda.memory_allocated()
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();packed=[w.t().contiguous().t() for w in weights];b.record();b.synchronize()
            assert all(v.stride()==(1,n) for v in packed)
            result['packing'].append(dict(projection=name,count=8,total_elapsed_us=a.elapsed_time(b)*1000,
              original_stride=[k,1],candidate_stride=[1,n],packed_tensor_bytes=sum(v.numel()*v.element_size() for v in packed),
              memory_allocated_delta=torch.cuda.memory_allocated()-memory_before,peak_allocated_delta=torch.cuda.max_memory_allocated()-memory_before,
              timing_scope='one-time8weight packing; excluded from steady state linear'));save()
            for m in MS:
                x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
                validate(name,x,weights[0],packed[0])
                old=lambda i:handler(None,x,weights[i],None)
                new=lambda i:handler(None,x,packed[i],None)
                for mode,cycle in [('hot',False),('weights8',True)]:
                    before=timed(old,cycle);candidate=timed(new,cycle);after=timed(old,cycle)
                    row=dict(projection=name,M=m,mode=mode,baseline_before=before,candidate=candidate,baseline_after=after,
                      time_change_percent=100*(candidate['median_us']/((before['median_us']+after['median_us'])/2)-1),
                      baseline_drift_percent=100*(after['median_us']/before['median_us']-1),one_time_packing_excluded=True,production_tail_M=m in [8,16])
                    result['timings'].append(row);save();print(json.dumps(row),flush=True)
                if m in [8,2048]:
                    events(name,m,'row_major',lambda:old(0));events(name,m,'column_major',lambda:new(0))
                del x;torch.cuda.empty_cache()
            del packed,weights;torch.cuda.empty_cache()
        for name,(n,k) in SHAPES.items():
            w=torch.randn(n,k,device='cuda',dtype=torch.float16)*.02;packed=w.t().contiguous().t()
            for m in [1,8,16,2048]:
                x=torch.randn(m,k,device='cuda',dtype=torch.float16);validate(name,x,w,packed);del x
            del w,packed;torch.cuda.empty_cache()
        assert (len(result['checks']),len(result['timings']),len(result['packing']),len(result['operator_events']))==(56,80,4,16)
        result['status']='passed'
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));traceback.print_exc();raise
    finally:result['finished_at']=time.strftime('%FT%T%z');save()

if __name__=='__main__':main()
