"""Complete native-linear / production-generated SwiGLU / native-linear screen."""
from pathlib import Path
import argparse,hashlib,inspect,json,statistics,time,traceback
import torch
import torch.nn.functional as F
from torch._inductor.async_compile import AsyncCompile
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm

MS=[1,4,8,16,30,32,64,128,512,2048]
SEEDS=[202610261,202610262,202610263]
VARIANTS=['gate_up','down','both']
ATOL=.03;RTOL=.03
SOURCE_HASH='2077eeed3cb635af436273a4ca3c69e6fb2d32e303bbf50b739950086df2cb6a'

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
    root=Path(__file__).resolve().parent
    source=json.loads((root/'production_silu_source.json').read_text())['source']
    assert hashlib.sha256(source.encode()).hexdigest()==SOURCE_HASH
    handler=dispatch_unquantized_gemm();assert handler.__name__=='default_unquantized_gemm' and F.linear.__module__=='torch._C._nn'
    result=dict(status='running',kind='complete_MLP_operator_diagnostic_not_official_e2e',started_at=time.strftime('%FT%T%z'),
      Ms=MS,seeds=SEEDS,variants=VARIANTS,tolerance=dict(atol=ATOL,rtol=RTOL),checks=[],timings=[],activation_sources={},
      native_linear=dict(callable=handler.__module__+'.'+handler.__name__,source_sha256=hashlib.sha256(inspect.getsource(handler).encode()).hexdigest()),
      caveat='Random tensors; same production-generated activation source compiled in separate cache, with fresh autotuning; not full model request replay; original weight layout inferred from production graph')
    def save():
        tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(args.output)
    kernels={}
    def build_kernel(dtype):
        name='triton_poi_fused_mul_silu_slice_1';derived=source
        if dtype==torch.float16:
            assert source.count("'*bf16'")==2
            derived=source.replace("'*bf16'","'*fp16'").replace(name,name+'_fp16');name+='_fp16'
        compiler=AsyncCompile();scope={name:compiler.triton(name,derived,device_str='cuda')};compiler.wait(scope);kernel=scope[name]
        # Production service profiles M2048; warm the same element count before
        # measuring. New-cache autotuning is recorded separately from source reuse.
        x=torch.zeros(2048,12288,device='cuda',dtype=dtype);out=torch.empty(2048,6144,device='cuda',dtype=dtype)
        kernel.run(x,out,out.numel(),stream=torch.cuda.current_stream().cuda_stream);torch.cuda.synchronize()
        torch.testing.assert_close(out,torch.zeros_like(out),atol=0,rtol=0)
        kernels[dtype]=kernel
        result['activation_sources'][str(dtype)]=dict(source_sha256=hashlib.sha256(derived.encode()).hexdigest(),kernel_name=name,
          production_source=dtype==torch.bfloat16,derived_fp16_signature_only=dtype==torch.float16,
          warmup_M=2048,best_config=str(getattr(kernel,'best_config',None)),configs_available=getattr(kernel,'configs',None) is not None,configs=[str(v) for v in (getattr(kernel,'configs',None) or [])])
        save()
    def act(x):
        out=torch.empty(x.shape[0],6144,device=x.device,dtype=x.dtype)
        kernels[x.dtype].run(x,out,out.numel(),stream=torch.cuda.current_stream().cuda_stream)
        return out
    def weights(dtype,count):
        gu=[torch.randn(12288,2048,device='cuda',dtype=dtype)*.02 for _ in range(count)]
        down=[torch.randn(2048,6144,device='cuda',dtype=dtype)*.02 for _ in range(count)]
        return gu,down,[w.t().contiguous().t() for w in gu],[w.t().contiguous().t() for w in down]
    def call(x,ws,i,variant):
        gu,down,pgu,pdown=ws
        g=pgu[i] if variant in ['gate_up','both'] else gu[i]
        d=pdown[i] if variant in ['down','both'] else down[i]
        return handler(None,act(handler(None,x,g,None)),d,None)
    def reference(x,wg,wd):
        g=F.linear(x.float(),wg.float()).to(x.dtype)
        a=(F.silu(g[:,:6144].float())*g[:,6144:].float()).to(x.dtype)
        out=F.linear(a.float(),wd.float()).to(x.dtype)
        return g,a,out
    def validate(x,ws,variant,seed,distribution):
        gu,down,pgu,pdown=ws
        for w,p in [(gu[0],pgu[0]),(down[0],pdown[0])]:
            assert w.stride()==(w.shape[1],1) and p.stride()==(1,w.shape[0]) and torch.equal(w,p)
        base=call(x,ws,0,'baseline');candidate=call(x,ws,0,variant)
        g,a,ref=reference(x,gu[0],down[0])
        cg=handler(None,x,pgu[0] if variant in ['gate_up','both'] else gu[0],None)
        ca=act(cg)
        torch.testing.assert_close(cg,g,atol=ATOL,rtol=RTOL)
        torch.testing.assert_close(ca,a,atol=ATOL,rtol=RTOL)
        for value in [base,candidate]:torch.testing.assert_close(value,ref,atol=ATOL,rtol=RTOL)
        torch.testing.assert_close(candidate,base,atol=ATOL,rtol=RTOL)
        err=(candidate.float()-base.float()).abs()
        errors=dict(candidate_baseline_max_abs=err.max().item(),candidate_baseline_max_scaled= (err/base.float().abs().clamp_min(1)).max().item(),candidate_reference_max_abs=(candidate.float()-ref.float()).abs().max().item())
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=call(x,ws,0,variant)
        x.mul_(.97).add_(.01)
        for v in [gu[0],down[0],pgu[0],pdown[0]]:v.add_(.0001)
        expected=call(x,ws,0,variant);_,_,ref=reference(x,gu[0],down[0]);graph.replay();torch.cuda.synchronize()
        torch.testing.assert_close(captured,expected,atol=0,rtol=0)
        torch.testing.assert_close(captured,ref,atol=ATOL,rtol=RTOL)
        assert captured.shape==(x.shape[0],2048) and captured.is_contiguous()
        result['checks'].append(dict(seed=seed,M=x.shape[0],dtype=str(x.dtype),variant=variant,distribution=distribution,
          graph_mutations=['input','gate_up_weight','down_weight'],graph_exact_replay=True,reference='FP32 linear with dtype rounding at gate_up and fused activation product boundaries',
          mutated_reference_max_abs=(captured.float()-ref.float()).abs().max().item(),**errors));save()
    torch.backends.cuda.matmul.allow_tf32=False
    try:
        build_kernel(torch.bfloat16);build_kernel(torch.float16)
        for seed_index,seed in enumerate(SEEDS):
            torch.manual_seed(seed);ws=weights(torch.bfloat16,8)
            order=VARIANTS[seed_index:]+VARIANTS[:seed_index]
            for m in MS:
                x=torch.randn(m,2048,device='cuda',dtype=torch.bfloat16)
                for variant in order:
                    validate(x,ws,variant,seed,'normal')
                    old=lambda i:call(x,ws,i,'baseline');new=lambda i:call(x,ws,i,variant)
                    for mode,cycle in [('hot',False),('weights8',True)]:
                        before=timed(old,cycle);candidate=timed(new,cycle);after=timed(old,cycle)
                        row=dict(seed=seed,M=m,variant=variant,mode=mode,baseline_before=before,candidate=candidate,baseline_after=after,
                          time_change_percent=100*(candidate['median_us']/((before['median_us']+after['median_us'])/2)-1),
                          baseline_drift_percent=100*(after['median_us']/before['median_us']-1),complete_MLP=True)
                        result['timings'].append(row);save();print(json.dumps(row),flush=True)
                del x
            del ws;torch.cuda.empty_cache()
        seed=202610264;torch.manual_seed(seed);ws=weights(torch.float16,1)
        for m in [1,8,16,512,2048]:
            x=torch.randn(m,2048,device='cuda',dtype=torch.float16)
            for variant in VARIANTS:validate(x,ws,variant,seed,'normal')
            del x
        del ws;torch.cuda.empty_cache()
        seed=202610265;torch.manual_seed(seed);ws=weights(torch.bfloat16,1)
        for m in [8,2048]:
            for distribution in ['zero','constant']:
                for variant in VARIANTS:
                    x=torch.full((m,2048),0. if distribution=='zero' else 1.,device='cuda',dtype=torch.bfloat16)
                    validate(x,ws,variant,seed,distribution);del x
        assert len(result['checks'])==117 and len(result['timings'])==180
        result['status']='passed'
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));traceback.print_exc();raise
    finally:result['finished_at']=time.strftime('%FT%T%z');save()
if __name__=='__main__':main()
