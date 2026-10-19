"""Correctness plus CUDA-graph timings, never alters the official benchmark."""
import json
import statistics
from pathlib import Path

import torch
from flag_gems.fused.rotary_embedding import apply_rotary_pos_emb
import argparse
import importlib.util
import os

parser=argparse.ArgumentParser()
parser.add_argument("--kernel",type=Path,required=True)
parser.add_argument("--output-dir",type=Path,required=True)
args=parser.parse_args()
spec=importlib.util.spec_from_file_location("rope_candidate",args.kernel.resolve())
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rope_parallel=module.rope_parallel
args.output_dir.mkdir(parents=True,exist_ok=False)
os.chdir(args.output_dir)


def graph_us(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(100):
            fn()
    g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(7):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            g.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / 500)
    return statistics.median(times)


def tensors(m, hq=16, hk=2, dim=128, dtype=torch.bfloat16):
    x = torch.randn(m, hq + 2 * hk, dim, device="cuda", dtype=dtype)
    # Strided slices match a fused QKV projection, including untouched V.
    q, k = x[:, :hq], x[:, hq:hq + hk]
    angles = torch.randn(32768, dim // 2, device="cuda")
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(dtype)
    c, s = cache.chunk(2, dim=-1)
    p = torch.randint(0, cache.shape[0], (m,), device="cuda", dtype=torch.int64)
    return x, q, k, c, s, p


def main():
    torch.manual_seed(1234)
    results = {"correctness": [], "timings": [], "note": "isolated kernel microbenchmark; not end-to-end speedup"}
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for m, hq, hk, dim in [(1,16,2,128),(7,16,2,128),(64,16,2,128),(257,16,2,128),(64,12,3,80),(64,8,1,64)]:
            for interleaved in [False, True]:
                x,q,k,c,s,p=tensors(m,hq,hk,dim,dtype)
                expected=x.clone()
                apply_rotary_pos_emb(expected[:,:hq],expected[:,hq:hq+hk],c,s,p,interleaved,True)
                reference=x.clone()
                for view in [reference[:,:hq],reference[:,hq:hq+hk]]:
                    cf, sf = c[p].float()[:,None,:], s[p].float()[:,None,:]
                    a = view[...,0::2].float().clone() if interleaved else view[...,:dim//2].float().clone()
                    b = view[...,1::2].float().clone() if interleaved else view[...,dim//2:].float().clone()
                    lower, upper = a*cf-b*sf, b*cf+a*sf
                    if interleaved:
                        view[...,0::2],view[...,1::2]=lower,upper
                    else:
                        view[...,:dim//2],view[...,dim//2:]=lower,upper
                tol = 2e-2 if dtype==torch.bfloat16 else (2e-3 if dtype==torch.float16 else 2e-6)
                old_error=(expected.float()-reference.float()).abs().max().item()
                exact=True
                for bh in [1,2,4,8,16]:
                    actual=x.clone()
                    rope_parallel(actual[:,:hq],actual[:,hq:hq+hk],c,s,p,bh=bh,interleaved=interleaved)
                    torch.testing.assert_close(actual, reference, rtol=tol, atol=tol,
                                               msg=f"dtype={dtype}, M={m}, D={dim}, BH={bh}, interleaved={interleaved}")
                    exact=exact and torch.equal(actual,expected)
                    if dim==128 and dtype in (torch.bfloat16, torch.float16):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                item={"dtype":str(dtype),"m":m,"hq":hq,"hk":hk,"dim":dim,"interleaved":interleaved,"exact_vs_existing":exact,"existing_error_vs_reference":old_error}
                results["correctness"].append(item)
                print(json.dumps(item),flush=True)
    print("CORRECTNESS: 180 candidate/reference comparisons passed; BF16/FP16 D=128 bitwise matches existing; FP32 differences are recorded for diagnosis",flush=True)
    for m in [1,8,32,64,128,512,2048]:
        x,q,k,c,s,p=tensors(m)
        x.zero_()  # Stable values during repeated in-place graph replays.
        base=lambda:apply_rotary_pos_emb(q,k,c,s,p,False,True)
        baseline=graph_us(base)
        candidates=[]
        for bh in [1,2,4,8,16]:
            for warps in [4,8]:
                us=graph_us(lambda:rope_parallel(q,k,c,s,p,bh=bh,warps=warps))
                candidates.append({"bh":bh,"warps":warps,"us":us,"speedup":baseline/us})
        baseline_after=graph_us(base)
        item={"m":m,"baseline_us":baseline,"baseline_after_us":baseline_after,"candidates":candidates}
        results["timings"].append(item)
        print(json.dumps(item),flush=True)
        Path("rope_results.json").write_text(json.dumps(results,indent=2))
    print("DONE",flush=True)


if __name__ == "__main__":
    main()
