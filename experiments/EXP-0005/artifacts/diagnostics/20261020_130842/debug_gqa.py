import sys,json,torch
sys.path.insert(0,'/workspace/LeanScale/MiniCPM-Infer-Sys-Opt/experiments/EXP-0005')
from validate_gqa import make_case,baseline,paged_gqa
torch.manual_seed(777)
result=[]
for qlens,lengths in [([7],[71]),([37],[201]),([1,37,0,1,7],[16385,201,0,4096,71])]:
 x,_,_=make_case(qlens,lengths)
 baseline(x,lengths); old=x['out'].clone()
 paged_gqa(**x,segments=8,decode_tile=32,prefill_tile=32,prefill_rows=16); new=x['out'].clone()
 start=0
 for seq,(n,L) in enumerate(zip(qlens,lengths)):
  if n:
   ids=torch.arange(L,device='cuda');blocks=x['block_table'][seq,ids//16]
   k=x['k'][blocks,ids%16].float().repeat_interleave(8,1);v=x['v'][blocks,ids%16].float().repeat_interleave(8,1)
   for i in sorted({0,n//2,n-1}):
    logits=torch.einsum('hd,lhd->hl',x['q'][start+i].float(),k)*x['scale'];logits[:,L-n+i+1:]=-torch.inf
    ref=torch.einsum('hl,lhd->hd',logits.softmax(-1),v)
    result.append(dict(qlens=qlens,seq=seq,i=i,baseline_error=(old[start+i]-ref).abs().max().item(),candidate_error=(new[start+i]-ref).abs().max().item(),baseline_sample=old[start+i,0,:4].float().tolist(),candidate_sample=new[start+i,0,:4].float().tolist(),ref_sample=ref[0,:4].tolist()))
  start+=n
print(json.dumps(result,indent=2))
