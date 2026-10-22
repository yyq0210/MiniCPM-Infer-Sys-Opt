"""Build a three-point, single-factor neighborhood check from the frozen sweep."""
from pathlib import Path
import ast

HERE=Path(__file__).resolve().parent


def replace(s,a,b):
    assert a in s,a
    return s.replace(a,b)


s=(HERE/'trial.py').read_text()
s=replace(s,"choices=(1,2),required=True)","choices=(1,2,3),required=True)\n    p.add_argument('--tile',type=int,choices=(32,64),required=True)")
s=replace(s,'prefill_stages=a.stages,prefill_tile=32','prefill_stages=a.stages,prefill_tile=a.tile')
ast.parse(s);(HERE/'trial_refine.py').write_text(s)

s=(HERE/'run_screen.py').read_text()
s=replace(s,"REQUEST=ROOT/'artifacts/EXP-0015-request.json'","REQUEST=ROOT/'artifacts/EXP-0015-refined-request.json'")
s=replace(s,"('trial.py','run_screen.py')","('trial_refine.py','run_refine.py')")
s=replace(s,'configs=[(64,4,2)]+[x for x in itertools.product((32,64,128),(2,4),(1,2)) if x!=(64,4,2)]',
            'configs=[(64,4,1,32),(64,4,1,64),(64,4,3,32)]')
s=replace(s,'for rows,warps,stages in configs:','for rows,warps,stages,tile in configs:')
s=replace(s,"name=f'rows{rows}_w{warps}_s{stages}'","name=f'rows{rows}_w{warps}_s{stages}_tile{tile}'")
s=replace(s,"str(EXP/'trial.py')","str(EXP/'trial_refine.py')")
s=replace(s,"'--stages',str(stages),","'--stages',str(stages),'--tile',str(tile),")
s=replace(s,"name=='rows64_w4_s2'","name=='rows64_w4_s1_tile32'")
s=s.replace('Baseline-equivalent configuration failed; stop sweep for diagnosis','Previously winning configuration failed; stop refinement for diagnosis')
s=s.replace('prefill parameter screening','prefill single-factor refinement')
s=s.replace('# EXP-0015 GQA prefill 参数筛选','# EXP-0015 GQA prefill 邻域复核')
ast.parse(s);(HERE/'run_refine.py').write_text(s)
print('Built three-point prefill refinement')
