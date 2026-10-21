"""Expanded checks for w4/stage1/split16, still not a model accuracy or E2E test."""
import json
from pathlib import Path
import sys
import time
import traceback
import torch
from trial import make, oracle, timed, baseline, candidate

CFG=dict(decode_warps=4,decode_stages=1,segments=16)

def main():
    output=Path(sys.argv[1]);r={'status':'running','config':CFG,'checks':[],'timings':[],
        'kind':'expanded_operator_validation_not_official_e2e','started_at':time.strftime('%FT%T%z')}
    def save():
        tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(output)
    def check(name,x,qs,ls,mutate_metadata=False):
        baseline.paged_gqa(**x);old=x['out'].clone();candidate.paged_gqa(**x,**CFG)
        torch.testing.assert_close(x['out'],old,atol=3e-3,rtol=3e-2)
        err=oracle(x,qs,ls)
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):candidate.paged_gqa(**x,**CFG)
        x['q'].mul_(.97);x['k'].mul_(1.01);x['v'].add_(.01)
        if mutate_metadata:
            qs=[0,1,1,1];ls=[0,49,7,4100]
            x['cu'].copy_(torch.tensor([0,0,1,2,3],device='cuda',dtype=torch.int32))
            x['lengths'].copy_(torch.tensor(ls,device='cuda',dtype=torch.int32))
            x['block_table'].copy_(x['block_table'].flip(1))
        candidate.paged_gqa(**x,**CFG);eager=x['out'].clone()
        g.replay();torch.cuda.synchronize();torch.testing.assert_close(x['out'],eager,atol=0,rtol=0)
        replay_err=oracle(x,qs,ls)
        r['checks'].append({'name':name,'fp32_max_error':err,'replay_fp32_max_error':replay_err,
                            'graph_passed':True,'changed_query_layout':mutate_metadata})
        save()
    save();torch.manual_seed(20261022)
    try:
        cases=[('short_and_empty',[1,0,1,1],[1,0,17,33]),
               ('ragged_and_replaced',[1,0,1,1],[17,0,33,4097]),
               ('long30',[1]*30,[17408-i*7 for i in range(30)]),
               ('mixed',[1,37,0,1,7],[16385,201,0,4096,71]),
               ('prefill2048',[2048],[16384]),
               ('prefill_tail',[3,17,67],[3,33,1025])]
        for dtype in (torch.bfloat16,torch.float16):
            for page,hnd in ((16,False),(32,True)):
                for name,qs,ls in cases:
                    x=make(qs,ls,dtype,page,hnd)
                    check(f'{dtype}_{page}_{hnd}_{name}',x,qs,ls,mutate_metadata=name=='ragged_and_replaced')
                    del x;torch.cuda.empty_cache()
        # Real device addresses above the int32 element offset range.
        for dtype in (torch.bfloat16,torch.float16):
            x=make([1],[17],dtype);stride=2**30+4096;elements=2*stride+4096
            kb=torch.empty(elements,device='cuda',dtype=dtype);vb=torch.empty_like(kb)
            x['k']=kb.as_strided((3,16,2,128),(stride,256,128,1))
            x['v']=vb.as_strided((3,16,2,128),(stride,256,128,1))
            x['k'].normal_(std=.3);x['v'].normal_()
            x['block_table']=torch.tensor([[2,1]],device='cuda',dtype=torch.int32)
            check(f'large_offset_{dtype}',x,[1],[17])
            r['checks'][-1]['page2_offset_elements']=2*stride;save()
            del x,kb,vb;torch.cuda.empty_cache()
        r['correctness_passed']=True;save()
        points=[(f'B{b}_L{l}',[1]*b,[l-i%17 for i in range(b)])
                for b in (1,8,16,30,32,64) for l in (4096,4700,5120,16384,17408)]
        points += [('mixed_decode_first',[1]*30+[2018],[16384]*31),
                   ('mixed_prefill_first',[2018]+[1]*30,[16384]*31),
                   ('prefill4k',[2048],[4096]),('prefill16k',[2048],[16384])]
        for name,qs,ls in points:
            x=make(qs,ls);old=lambda:baseline.paged_gqa(**x);new=lambda:candidate.paged_gqa(**x,**CFG)
            old();expected=x['out'].clone();new()
            torch.testing.assert_close(x['out'],expected,atol=3e-3,rtol=3e-2)
            err=oracle(x,qs,ls,selected={0,len(qs)//2,len(qs)-1})
            before=timed(old);current=timed(new);after=timed(old)
            ref=(before['median_us']+after['median_us'])/2
            row={'name':name,'baseline_before':before,'candidate':current,'baseline_after':after,
                 'time_change_percent':(current['median_us']/ref-1)*100,
                 'baseline_drift_percent':(after['median_us']/before['median_us']-1)*100,'fp32_error':err}
            r['timings'].append(row);save();print(json.dumps(row),flush=True)
            del x,expected;torch.cuda.empty_cache()
        r['status']='passed'
    except BaseException as e:r.update(status='failed',error=repr(e));traceback.print_exc();raise
    finally:r['finished_at']=time.strftime('%FT%T%z');save()

if __name__=='__main__':main()
