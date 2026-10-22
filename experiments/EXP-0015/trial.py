"""One isolated GQA configuration: oracle/Graph checks and bracketed GPU timings."""
import argparse
import importlib.util
import hashlib
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
path=Path('/workspace/LeanScale/worktrees/EXP-0014/FlagGems/src/flag_gems/runtime/backend/_iluvatar/fused/paged_gqa.py')
spec=importlib.util.spec_from_file_location('exp15_frozen_gqa',path)
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

DECODE = dict(decode_warps=4, decode_stages=1, segments=16, decode_layout=1, decode_rows=8)
POINTS = [
    ('prefill4k', [2048], [4096]),
    ('prefill16k', [2048], [16384]),
    ('mixed4k_decode_first', [1]*63+[1985], [4700]*63+[4096]),
    ('mixed4k_prefill_first', [1985]+[1]*63, [4096]+[4700]*63),
    ('mixed16k_decode_first', [1]*30+[2018], [16384]*31),
    ('mixed16k_prefill_first', [2018]+[1]*30, [16384]*31),
    ('decode64_4k', [1]*64, [4700-i%17 for i in range(64)]),
    ('decode30_16k', [1]*30, [16384-i%17 for i in range(30)]),
    ('decode1_4k', [1], [4096]),
]


def prefill_component(x, cfg):
    n=x['lengths'].numel();cap=x['partial_max'].shape[-1]
    args=(x['q'],x['k'],x['v'],x['out'],x['cu'],x['lengths'],x['block_table'],
          x['partial_out'],x['partial_max'],x['partial_sum'],x['scale'],n,
          x['q'].stride(0),x['q'].stride(1),x['out'].stride(0),x['out'].stride(1),
          *x['k'].stride()[:3],*x['v'].stride()[:3],x['block_table'].stride(0),x['k'].shape[1],cap,16)
    def launch():
        return candidate._attention[(x['q'].shape[0]//(cfg['prefill_rows']//8)+n,2)](
            *args,cfg['prefill_tile'],cfg['prefill_rows'],False,0,
            num_warps=cfg['prefill_warps'],num_stages=cfg['prefill_stages'])
    return launch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rows',type=int,choices=(32,64,128),required=True)
    p.add_argument('--warps',type=int,choices=(2,4),required=True)
    p.add_argument('--stages',type=int,choices=(1,2),required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=dict(DECODE,prefill_rows=a.rows,prefill_warps=a.warps,prefill_stages=a.stages,prefill_tile=32)
    r={'status':'running','kind':'single_operator_diagnostic_not_official_e2e','config':cfg,
       'checks':[],'timings':[],'started_at':time.strftime('%FT%T%z')}
    def save():
        tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(a.output)
    save();torch.manual_seed(20261022)
    try:
        tests=[([1,0,1,1],[17,0,33,4097]),
               ([1,37,0,1,7],[16385,201,0,4096,71]),
               ([0],[0]),
               ([3,17,65,129],[3,33,1025,4096]),
               ([1,257,1,0],[31,16384,33,0])]
        for dtype,page,hnd in ((torch.bfloat16,16,False),(torch.float16,32,True),
                               (torch.bfloat16,8,False),(torch.bfloat16,64,True)):
            for qs,ls in tests:
                x=make(qs,ls,dtype,page,hnd)
                baseline.paged_gqa(**x,**DECODE);expected=x['out'].clone()
                candidate.paged_gqa(**x,**cfg)
                torch.testing.assert_close(x['out'],expected,atol=3e-3,rtol=3e-2)
                err=oracle(x,qs,ls)
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):candidate.paged_gqa(**x,**cfg)
                x['q'].add_(.013);x['k'].mul_(1.01);x['v'].mul_(.99)
                changed=[max(q,l-3) if l else 0 for q,l in zip(qs,ls)]
                x['lengths'].copy_(torch.tensor(changed,device='cuda',dtype=torch.int32))
                x['block_table'].copy_(x['block_table'].flip(1))
                candidate.paged_gqa(**x,**cfg);eager=x['out'].clone()
                g.replay();torch.cuda.synchronize()
                torch.testing.assert_close(x['out'],eager,atol=0,rtol=0)
                err2=oracle(x,qs,changed)
                r['checks'].append(dict(dtype=str(dtype),page=page,hnd=hnd,qlens=qs,lengths=ls,
                    fp32_error=err,mutated_fp32_error=err2,graph_replay_passed=True,
                    mutations=['Q','K','V','lengths','block_table']))
                save();del x,g,eager,expected;torch.cuda.empty_cache()
        r['correctness_passed']=True;save()
        for name,qs,ls in POINTS:
            x=make(qs,ls)
            old=lambda:baseline.paged_gqa(**x,**DECODE)
            new=lambda:candidate.paged_gqa(**x,**cfg)
            old();expected=x['out'].clone();new()
            torch.testing.assert_close(x['out'],expected,atol=3e-3,rtol=3e-2)
            err=oracle(x,qs,ls,selected={0,len(qs)//2,len(qs)-1})
            before=timed(old);current=timed(new);after=timed(old)
            ref=(before['median_us']+after['median_us'])/2
            row=dict(name=name,baseline_before=before,candidate_total=current,baseline_after=after,
                time_change_percent=(current['median_us']/ref-1)*100,
                baseline_drift_percent=(after['median_us']/before['median_us']-1)*100,fp32_error=err)
            if max(qs)>1:
                fn=prefill_component(x,cfg);kernel=fn();torch.cuda.synchronize()
                resources={k:getattr(kernel,k,None) for k in ('n_regs','n_spills')}
                resources['shared']=getattr(getattr(kernel,'metadata',None),'shared',None)
                resources['ir_sha256']={}
                ir_dir=a.output.parent/'ir'/a.output.stem;ir_dir.mkdir(parents=True,exist_ok=True)
                for kind,code in kernel.asm.items():
                    if isinstance(code,str) and kind in ('ttir','ttgir','llir','asm'):
                        resources['ir_sha256'][kind]=hashlib.sha256(code.encode()).hexdigest()
                        (ir_dir/f'{name}.{kind}').write_text(code)
                row.update(prefill_only=timed(fn),resources=resources)
                del kernel,fn
            r['timings'].append(row);save();print(json.dumps(row),flush=True)
            del x,expected;torch.cuda.empty_cache()
        r['status']='passed'
    except BaseException as e:
        r.update(status='failed',error=repr(e));traceback.print_exc();raise
    finally:
        r['finished_at']=time.strftime('%FT%T%z');save()


if __name__=='__main__':main()
