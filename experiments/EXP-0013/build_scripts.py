"""Build EXP-0013 from the measured EXP-0011 harness, with a stage1 reference."""
from pathlib import Path
import ast

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'flagos-exp11-dev'

def replace(s,old,new):
    assert old in s,old
    return s.replace(old,new)

s=(OLD/'trial.py').read_text()
s=replace(s,'import importlib.util','import importlib.util\nimport hashlib')
s=replace(s,'worktrees/EXP-0006/FlagGems','worktrees/EXP-0012/FlagGems')
s=replace(s,'exp11_frozen_gqa','exp13_frozen_gqa')
s=replace(s,'baseline.paged_gqa(**x)','baseline.paged_gqa(**x,decode_warps=4,decode_stages=1,segments=16)')
s=replace(s,"*args,32,16,True,","*args,32,cfg['decode_rows'],True,cfg['decode_layout'],")
s=replace(s,"p=argparse.ArgumentParser();p.add_argument('--warps',type=int,required=True)\n    p.add_argument('--stages',type=int,required=True);p.add_argument('--segments',type=int,required=True)",
            "p=argparse.ArgumentParser();p.add_argument('--layout',type=int,choices=(0,1,2,3),required=True)\n    p.add_argument('--rows',type=int,choices=(8,16),required=True)")
s=replace(s,'cfg=dict(decode_warps=a.warps,decode_stages=a.stages,segments=a.segments)',
            'cfg=dict(decode_warps=4,decode_stages=1,segments=16,decode_layout=a.layout,decode_rows=a.rows)')
s=replace(s,"tests=[([1,0,1,1],[17,0,33,4097]),([1,37,0,1,7],[16385,201,0,4096,71]),([0],[0])]",
            "tests=[([1,0,1,1],[17,0,33,4097]),([1,37,0,1,7],[16385,201,0,4096,71]),([0],[0]),\n               ([1,0,1,1],[0,0,15,16]),([1,1,1],[31,32,33])]")
s=replace(s,'((torch.bfloat16,16,False),(torch.float16,32,True))',
            '((torch.bfloat16,16,False),(torch.float16,32,True),(torch.bfloat16,8,False),(torch.bfloat16,64,True))')
s=replace(s,"resources['shared']=getattr(getattr(kernel,'metadata',None),'shared',None)",
            "resources['shared']=getattr(getattr(kernel,'metadata',None),'shared',None)\n            resources['ir_sha256']={}\n            ir_dir=a.output.parent/'ir'/a.output.stem;ir_dir.mkdir(parents=True,exist_ok=True)\n            for kind,code in kernel.asm.items():\n                if isinstance(code,str) and kind in ('ttir','ttgir','llir','asm'):\n                    resources['ir_sha256'][kind]=hashlib.sha256(code.encode()).hexdigest()\n                    (ir_dir/f'B{b}_L{l}.{kind}').write_text(code)")
ast.parse(s);(HERE/'trial.py').write_text(s)

s=(OLD/'run_screen.py').read_text().replace('EXP-0011','EXP-0013')
s=replace(s,'import os','import os\nimport re')
s=replace(s,'shutil.copy2(p,dest/p.name)',
            "text=p.read_text(errors='replace').replace('\\r\\n','\\n')\n            text=re.sub(r'\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b','[IP_REDACTED]',text)\n            text=re.sub(r'uuid=[^,)]+','uuid=[REDACTED]',text)\n            (dest/p.name).write_text(text)")
s=replace(s,'configs=[(4,2,16)]+[x for x in itertools.product((2,4),(1,2,3),(8,16)) if x!=(4,2,16)]',
            'configs=[(layout,rows) for rows in (16,8) for layout in (0,1,2,3)]')
s=replace(s,'for w,t,s in configs:','for layout,decode_rows in configs:')
s=replace(s,"name=f'w{w}_t{t}_s{s}'","name=f'layout{layout}_rows{decode_rows}'")
s=replace(s,"command=['python3',str(EXP/'trial.py'),'--warps',str(w),'--stages',str(t),'--segments',str(s),",
            "command=['python3',str(EXP/'trial.py'),'--layout',str(layout),'--rows',str(decode_rows),")
s=replace(s,"name=='w4_t2_s16'","name=='layout0_rows16'")
s=replace(s,'GQA 参数筛选','GQA 分页读取和查询行布局筛选')
s=replace(s,"'experiments: archive EXP-0013 GQA parameter screening'","'experiments: archive EXP-0013 paged KV and query layout screening'")
s=replace(s,"paths=[str(dest.relative_to(REPO))]", "paths=[str(dest.relative_to(REPO))]\n        assert not git('diff','--cached','--name-only'),'Existing staged changes'")
ast.parse(s);(HERE/'run_screen.py').write_text(s)
print('Built trial.py and run_screen.py')

s=(OLD/'validate_extended.py').read_text()
s=replace(s,'w4/stage1/split16','paged-index broadcast / 8 query rows at stage1')
s=replace(s,'CFG=dict(decode_warps=4,decode_stages=1,segments=16)',
            'CFG=dict(decode_warps=4,decode_stages=1,segments=16,decode_layout=1,decode_rows=8)')
s=replace(s,'baseline.paged_gqa(**x)','baseline.paged_gqa(**x,decode_warps=4,decode_stages=1,segments=16)')
ast.parse(s);(HERE/'validate_extended.py').write_text(s)
s=(OLD/'run_extended.py').read_text().replace('EXP-0011','EXP-0013')
s=replace(s,"'w4_t1_s16'","'layout1_rows8'")
s=replace(s,"'config':{'decode_warps':4,'decode_stages':1,'segments':16}",
            "'config':{'decode_warps':4,'decode_stages':1,'segments':16,'decode_layout':1,'decode_rows':8}")
ast.parse(s);(HERE/'run_extended.py').write_text(s)
print('Built extended validation scripts')
