"""Expanded prefill validation, including every query, mutable CU, and actual large offsets."""
from pathlib import Path
import itertools
import json
import sys
import time
import traceback
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from trial import make, oracle, timed, baseline, candidate, BASE_CONFIG, POINTS

CFG=json.loads((Path(__file__).parent/'choice.json').read_text())['config']


def all_query_oracle(x,qs,ls):
    """Independent FP32 causal attention, computed in bounded query chunks."""
    start=0;error=0.0;checked=0
    for seq,(qn,ln) in enumerate(zip(qs,ls)):
        if not qn:continue
        if ln:
            tokens=torch.arange(ln,device='cuda');page=x['k'].shape[1]
            ids=x['block_table'][seq,tokens//page]
            k=x['k'][ids,tokens%page].float().repeat_interleave(8,1).permute(1,2,0)
            v=x['v'][ids,tokens%page].float().repeat_interleave(8,1).permute(1,0,2)
        for offset in range(0,qn,32):
            end=min(offset+32,qn)
            got=x['out'][start+offset:start+end].float()
            if not ln:ref=torch.zeros_like(got)
            else:
                q=x['q'][start+offset:start+end].float().permute(1,0,2)
                scores=torch.matmul(q,k)*x['scale']
                pos=torch.arange(offset,end,device='cuda')
                scores.masked_fill_(tokens[None,None,:]>(ln-qn+pos)[None,:,None],-torch.inf)
                ref=torch.matmul(scores.softmax(-1),v).permute(1,0,2)
            torch.testing.assert_close(got,ref,atol=3e-3,rtol=3e-2)
            error=max(error,(got-ref).abs().max().item());checked+=end-offset
        start+=qn
    return dict(max_abs_error=error,queries_checked=checked)


def main():
    output=Path(sys.argv[1])
    result=dict(status='running',config=CFG,checks=[],timings=[],
        kind='expanded_operator_validation_not_official_e2e',started_at=time.strftime('%FT%T%z'))
    def save():
        tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(output)
    def check(name,x,qs,ls,change_cu=False):
        baseline.paged_gqa(**x,**BASE_CONFIG);old=x['out'].clone()
        candidate.paged_gqa(**x,**CFG)
        torch.testing.assert_close(x['out'],old,atol=3e-3,rtol=3e-2)
        first=all_query_oracle(x,qs,ls)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):candidate.paged_gqa(**x,**CFG)
        x['q'].mul_(.97);x['k'].mul_(1.01);x['v'].add_(.01)
        if change_cu:
            qs=[1,17,0,15];ls=[4095,65,0,129]
            x['cu'].copy_(torch.tensor([0]+list(itertools.accumulate(qs)),device='cuda',dtype=torch.int32))
        else:ls=[max(q,l-3) if l else 0 for q,l in zip(qs,ls)]
        x['lengths'].copy_(torch.tensor(ls,device='cuda',dtype=torch.int32))
        x['block_table'].copy_(x['block_table'].flip(1))
        baseline.paged_gqa(**x,**BASE_CONFIG);mutated_old=x['out'].clone()
        candidate.paged_gqa(**x,**CFG);eager=x['out'].clone()
        torch.testing.assert_close(eager,mutated_old,atol=3e-3,rtol=3e-2)
        graph.replay();torch.cuda.synchronize()
        torch.testing.assert_close(x['out'],eager,atol=0,rtol=0)
        second=all_query_oracle(x,qs,ls)
        result['checks'].append(dict(name=name,all_query_fp32=first,mutated_all_query_fp32=second,
            graph_passed=True,changed_cu=change_cu,mutations=['Q','K','V','lengths','block_table']+(['CU'] if change_cu else [])))
        save()
    save();torch.manual_seed(20261023)
    torch.backends.cuda.matmul.allow_tf32=False
    try:
        cases=[('query_boundaries',[0,2,3,7,8,9,15,16,17,63,65],[0,2,17,31,33,64,65,129,257,4096,16385]),
               ('ragged_mixed',[1,37,0,1,7],[16385,201,0,4096,71]),
               ('prefill_tail',[129],[4096]),
               ('prefill2048',[2048],[16384]),
               ('mutable_cu',[17,0,1,15],[129,0,4096,64])]
        for dtype in (torch.bfloat16,torch.float16):
            for page,hnd in ((8,False),(16,True),(32,False),(64,True)):
                for name,qs,ls in cases:
                    x=make(qs,ls,dtype,page,hnd)
                    check(f'{dtype}_{page}_{hnd}_{name}',x,qs,ls,change_cu=name=='mutable_cu')
                    del x;torch.cuda.empty_cache()
        for dtype in (torch.bfloat16,torch.float16):
            x=make([17],[33],dtype);stride=2**30+4096;elements=2*stride+4096
            kb=torch.empty(elements,device='cuda',dtype=dtype);vb=torch.empty_like(kb)
            x['k']=kb.as_strided((3,16,2,128),(stride,256,128,1))
            x['v']=vb.as_strided((3,16,2,128),(stride,256,128,1))
            x['k'].normal_(std=.3);x['v'].normal_()
            x['block_table']=torch.tensor([[2,1,0]],device='cuda',dtype=torch.int32)
            check(f'prefill_large_offset_{dtype}',x,[17],[33])
            result['checks'][-1]['page2_offset_elements']=2*stride;save()
            del x,kb,vb;torch.cuda.empty_cache()
        result['correctness_passed']=True;save()
        points=[(f'prefill_q{q}_L{l}',[q],[l]) for q in (2,8,17,64,128,512,2048) for l in (4096,16384)]
        points += [('prefill_first_chunk',[2048],[2048]),
                   ('multiple_prefills',[64]*8,[4096]*8),
                   ('ragged_prefills',[3,65,511,127],[17,4096,16384,1024])]
        points += [p for p in POINTS if p[0] in ('mixed4k_decode_first','mixed4k_prefill_first','mixed16k_decode_first','mixed16k_prefill_first')]
        points += [('small_mixed_decode_first',[1]*7+[17],[4700]*7+[4096]),
                   ('small_mixed_prefill_first',[17]+[1]*7,[4096]+[4700]*7),
                   ('medium_mixed',[1]*31+[512],[4700]*31+[16384])]
        points += [(f'decode_B{b}_L{l}',[1]*b,[l-i%17 for i in range(b)])
                   for b in (1,8,16,30,32,64) for l in (4700,16384)]
        points += [p for p in POINTS if p[0].startswith('exp18_')]
        assert len(points)==42
        for name,qs,ls in points:
            x=make(qs,ls)
            old=lambda:baseline.paged_gqa(**x,**BASE_CONFIG)
            new=lambda:candidate.paged_gqa(**x,**CFG)
            old();expected=x['out'].clone();new()
            torch.testing.assert_close(x['out'],expected,atol=3e-3,rtol=3e-2)
            err=oracle(x,qs,ls,selected={0,len(qs)//2,len(qs)-1})
            before=timed(old);current=timed(new);after=timed(old)
            ref=(before['median_us']+after['median_us'])/2
            row=dict(name=name,baseline_before=before,candidate=current,baseline_after=after,
                time_change_percent=(current['median_us']/ref-1)*100,
                baseline_drift_percent=(after['median_us']/before['median_us']-1)*100,fp32_error=err)
            result['timings'].append(row);save();print(json.dumps(row),flush=True)
            del x,expected;torch.cuda.empty_cache()
        result['status']='passed'
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));traceback.print_exc();raise
    finally:result['finished_at']=time.strftime('%FT%T%z');save()


if __name__=='__main__':main()
