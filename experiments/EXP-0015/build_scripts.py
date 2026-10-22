"""Derive the prefill sweep from the completed EXP13 validation harness."""
from pathlib import Path
import ast

HERE = Path(__file__).resolve().parent
OLD = HERE/'parent_exp13' if (HERE/'parent_exp13').is_dir() else HERE.parent/'flagos-exp13-dev'


def replace(text, old, new):
    assert old in text, old
    return text.replace(old, new)


s = (OLD/'trial.py').read_text().split('def components(')[0]
s = s.replace('worktrees/EXP-0012/FlagGems', 'worktrees/EXP-0014/FlagGems').replace('exp13_frozen_gqa', 'exp15_frozen_gqa')
s += '''DECODE = dict(decode_warps=4, decode_stages=1, segments=16, decode_layout=1, decode_rows=8)
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
        tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2)+'\\n');tmp.replace(a.output)
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
'''
ast.parse(s)
(HERE/'trial.py').write_text(s)

s = (OLD/'run_screen.py').read_text().replace('EXP-0013', 'EXP-0015')
s = replace(s, "configs=[(layout,rows) for rows in (16,8) for layout in (0,1,2,3)]",
            "configs=[(64,4,2)]+[x for x in itertools.product((32,64,128),(2,4),(1,2)) if x!=(64,4,2)]")
s = replace(s, 'for layout,decode_rows in configs:', 'for rows,warps,stages in configs:')
s = replace(s, "name=f'layout{layout}_rows{decode_rows}'", "name=f'rows{rows}_w{warps}_s{stages}'")
s = replace(s, "'--layout',str(layout),'--rows',str(decode_rows),",
            "'--rows',str(rows),'--warps',str(warps),'--stages',str(stages),")
s = replace(s, "name=='layout0_rows16'", "name=='rows64_w4_s2'")
s = replace(s, 'primary=stable and max(changes[:2])<=3 and min(changes[:2])<=-3',
            'assert len(changes)==9\n            primary=stable and max(changes[:6])<=3 and min(changes[:6])<=-5')
s = replace(s, "primary and changes[2]<=5", "primary and max(changes[6:])<=5")
s = replace(s, "'order':['B64/L4700','B30/L16384','B1/L4096']",
            "'order':['prefill4k','prefill16k','mixed4k_decode_first','mixed4k_prefill_first',"
            "'mixed16k_decode_first','mixed16k_prefill_first','decode64_4k','decode30_16k','decode1_4k']")
start = s.index('        lines=')
end = s.index("        (run/'RESULTS.md')", start)
s = s[:start]+'''        lines=['# EXP-0015 GQA prefill 参数筛选','','单算子诊断；未改变服务版本，未测官方E2E。',
               '全部9个工作点及原始采样见各配置JSON；完整GQA总耗时为筛选依据，prefill-only单列解释。','',
               '| 配置 | prefill4K | prefill16K | mixed4K两顺序 | mixed16K两顺序 | decode三个回归点 | 基线稳定 | 全局候选门槛 |',
               '|---|---:|---:|---|---|---|---|---|']
        for row in rows:
            c=row['time_change_percent']
            fmt=lambda xs:' / '.join(f'{x:+.3f}%' for x in xs)
            lines.append(f"| {row['configuration']} | {fmt(c[:1])} | {fmt(c[1:2])} | {fmt(c[2:4])} | {fmt(c[4:6])} | {fmt(c[6:])} | {row['stable_baseline']} | {row['global_candidate_gate_passed']} |")
        lines += ['','负数表示时间减少。基线漂移≤3%；六个prefill/mixed点均不退化>3%、至少一点减少≥5%，decode回归≤5%。',
                  '失败/超时、资源超限以及全部采样保留；获选后还须扩大数值/Graph/地址检查及官方4K/16K。']
''' + s[end:]
s = s.replace('paged KV and query layout screening', 'prefill parameter screening')
needle = "        git('add','--',*paths);git('diff','--cached','--check')"
s = replace(s, needle, """        forbidden=re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|lab_inst_[A-Za-z0-9]+|ssh\\.flagos\\.io')
        for item in dest.rglob('*'):
            if item.is_file():assert not forbidden.search(item.read_bytes()),str(item)
"""+needle)
ast.parse(s)
(HERE/'run_screen.py').write_text(s)
print('Built prefill trial and serialized controller')
