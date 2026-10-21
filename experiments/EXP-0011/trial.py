"""One isolated GQA configuration: oracle/Graph checks and bracketed GPU timings."""
import argparse
import importlib.util
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

import torch
import triton
from importlib import import_module
candidate = import_module('flag_gems.runtime.backend._iluvatar.fused.paged_gqa')
path=Path('/workspace/LeanScale/worktrees/EXP-0006/FlagGems/src/flag_gems/runtime/backend/_iluvatar/fused/paged_gqa.py')
spec=importlib.util.spec_from_file_location('exp11_frozen_gqa',path)
baseline=importlib.util.module_from_spec(spec);sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)

def make(qlens,lengths,dtype=torch.bfloat16,page=16,hnd=False):
    n=len(qlens);width=max(1,math.ceil(max(lengths,default=0)/page));pages=max(1,width*n)
    raw=torch.randn((pages,2,2,page,128) if hnd else (pages,2,page,2,128),device='cuda',dtype=dtype)
    cache=raw.permute(0,1,3,2,4) if hnd else raw;k,v=cache.unbind(1)
    table=torch.randperm(pages,device='cuda').int().reshape(n,width)
    q=torch.randn(sum(qlens),20,128,device='cuda',dtype=dtype)[:,:16]
    po=torch.empty(max(n,64),16,16,128,device='cuda',dtype=torch.float32)
    pm=torch.empty(max(n,64),16,16,device='cuda',dtype=torch.float32)
    return dict(q=q,k=k,v=v,out=torch.empty_like(q),
        cu=torch.tensor([0]+list(itertools.accumulate(qlens)),device='cuda',dtype=torch.int32),
        lengths=torch.tensor(lengths,device='cuda',dtype=torch.int32),block_table=table,
        scale=128**-.5,max_query_len=max(qlens,default=0),partial_out=po,partial_max=pm,partial_sum=torch.empty_like(pm))

def oracle(x,qlens,lengths,selected=None):
    start=0;err=0.0
    for seq,(qn,ln) in enumerate(zip(qlens,lengths)):
        if qn and (selected is None or seq in selected):
            tok=torch.arange(ln,device='cuda');page=x['k'].shape[1]
            ids=x['block_table'][seq,tok//page]
            k=x['k'][ids,tok%page].float().repeat_interleave(8,1)
            v=x['v'][ids,tok%page].float().repeat_interleave(8,1)
            for i in sorted({0,qn//2,qn-1}):
                if ln==0:ref=torch.zeros((16,128),device='cuda')
                else:
                    scores=torch.einsum('hd,lhd->hl',x['q'][start+i].float(),k)*x['scale']
                    scores[:,ln-qn+i+1:]=-torch.inf
                    ref=torch.einsum('hl,lhd->hd',scores.softmax(-1),v)
                got=x['out'][start+i].float()
                torch.testing.assert_close(got,ref,atol=3e-3,rtol=3e-2)
                err=max(err,(got-ref).abs().max().item())
        start+=qn
    return err

def timed(fn):
    for _ in range(3):fn()
    torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(8):fn()
    g.replay();torch.cuda.synchronize()
    vals=[]
    for _ in range(7):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();g.replay();b.record();b.synchronize();vals.append(a.elapsed_time(b)*1000/8)
    return {'median_us':statistics.median(vals),'samples_us':vals}

def components(x,cfg):
    n=x['lengths'].numel();s=cfg['segments'];cap=x['partial_max'].shape[-1]
    args=(x['q'],x['k'],x['v'],x['out'],x['cu'],x['lengths'],x['block_table'],
          x['partial_out'],x['partial_max'],x['partial_sum'],x['scale'],n,
          x['q'].stride(0),x['q'].stride(1),x['out'].stride(0),x['out'].stride(1),
          *x['k'].stride()[:3],*x['v'].stride()[:3],x['block_table'].stride(0),x['k'].shape[1],cap,s)
    def main():
        return candidate._attention[(n,2,s)](*args,32,16,True,
                 num_warps=cfg['decode_warps'],num_stages=cfg['decode_stages'])
    def reduce():
        return candidate._reduce[(n,16)](x['partial_out'],x['partial_max'],x['partial_sum'],
                 x['out'],x['cu'],x['out'].stride(0),x['out'].stride(1),cap,s,
                 triton.next_power_of_2(s),num_warps=4)
    return main,reduce

def main():
    p=argparse.ArgumentParser();p.add_argument('--warps',type=int,required=True)
    p.add_argument('--stages',type=int,required=True);p.add_argument('--segments',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=dict(decode_warps=a.warps,decode_stages=a.stages,segments=a.segments)
    r={'status':'running','kind':'single_operator_diagnostic_not_official_e2e','config':cfg,
       'checks':[],'timings':[],'started_at':time.strftime('%FT%T%z')}
    def save():
        tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(a.output)
    save();torch.manual_seed(20261021)
    try:
        tests=[([1,0,1,1],[17,0,33,4097]),([1,37,0,1,7],[16385,201,0,4096,71]),([0],[0])]
        for dtype,page,hnd in ((torch.bfloat16,16,False),(torch.float16,32,True)):
            for qlens,lengths in tests:
                x=make(qlens,lengths,dtype,page,hnd)
                baseline.paged_gqa(**x);expected=x['out'].clone()
                candidate.paged_gqa(**x,**cfg)
                torch.testing.assert_close(x['out'],expected,atol=3e-3,rtol=3e-2)
                err=oracle(x,qlens,lengths)
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):candidate.paged_gqa(**x,**cfg)
                x['q'].add_(.013);x['k'].mul_(1.01);x['v'].mul_(.99)
                changed=[max(q,l-3) if l else 0 for q,l in zip(qlens,lengths)]
                x['lengths'].copy_(torch.tensor(changed,device='cuda',dtype=torch.int32))
                x['block_table'].copy_(x['block_table'].flip(1))
                candidate.paged_gqa(**x,**cfg);eager=x['out'].clone()
                g.replay();torch.cuda.synchronize()
                torch.testing.assert_close(x['out'],eager,atol=0,rtol=0)
                err2=oracle(x,qlens,changed)
                r['checks'].append(dict(dtype=str(dtype),page=page,hnd=hnd,qlens=qlens,lengths=lengths,
                    fp32_error=err,mutated_fp32_error=err2,graph_replay_passed=True,
                    mutations=['Q','K','V','lengths','block_table']))
                save();del x,g,eager,expected;torch.cuda.empty_cache()
        r['correctness_passed']=True;save()
        for b,l in ((64,4700),(30,16384),(1,4096)):
            qlens=[1]*b;lengths=[l-i%17 for i in range(b)];x=make(qlens,lengths)
            old=lambda:baseline.paged_gqa(**x)
            new=lambda:candidate.paged_gqa(**x,**cfg)
            old();expected=x['out'].clone();new()
            torch.testing.assert_close(x['out'],expected,atol=3e-3,rtol=3e-2)
            err=oracle(x,qlens,lengths,selected={0,b//2,b-1})
            before=timed(old);after_candidate=timed(new);after=timed(old)
            main_fn,reduce_fn=components(x,cfg)
            kernel=main_fn();reduce_fn();torch.cuda.synchronize()
            resources={k:getattr(kernel,k,None) for k in ('n_regs','n_spills')}
            resources['shared']=getattr(getattr(kernel,'metadata',None),'shared',None)
            main_time=timed(main_fn);reduce_time=timed(reduce_fn)
            ref=(before['median_us']+after['median_us'])/2
            row={'B':b,'L':l,'baseline_before':before,'candidate_total':after_candidate,
                 'baseline_after':after,'speedup':ref/after_candidate['median_us'],
                 'time_change_percent':(after_candidate['median_us']/ref-1)*100,
                 'baseline_drift_percent':(after['median_us']/before['median_us']-1)*100,
                 'main_only':main_time,'reduce_only':reduce_time,'resources':resources,'fp32_error':err}
            r['timings'].append(row);print(json.dumps(row),flush=True);save()
            del x,expected,kernel,main_fn,reduce_fn;torch.cuda.empty_cache()
        r['status']='passed'
    except BaseException as e:
        r.update(status='failed',error=repr(e));traceback.print_exc();raise
    finally:
        r['finished_at']=time.strftime('%FT%T%z');save()

if __name__=='__main__':main()
