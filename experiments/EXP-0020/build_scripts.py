"""Derive EXP20 from verified EXP16 without changing the official workload."""
from pathlib import Path
import ast
import hashlib
import shutil

OUT=Path(__file__).resolve().parent
PARENT=OUT/'parent_exp16'
EXPECTED={
 'run_split_e2e.py':'0c28df633c175694967f390325f0d34be304de52c95244e064c9077556f2c466',
 'launch_split.py':'3654a50588861caf4e873fc0dda083e588ceaacec8a0b58fbd4d6a74067ab26d',
 'finish_split.py':'3006f4b1815200038fb957c6371f15b912dbb0d6afda873819787acc7cfab227',
 'analyze_schedule.py':'574042b1d7d97454691b3608f2fe0073cc760a6c0314a1cd1315bc95a769ea9f'}
for name,digest in EXPECTED.items():assert hashlib.sha256((PARENT/name).read_bytes()).hexdigest()==digest

def once(t,a,b):
    assert t.count(a)==1,(a,t.count(a))
    return t.replace(a,b)

def renumber(t):
    return t.replace('EXP-0016','EXP-TEMP').replace('EXP-0014','EXP-0016').replace('EXP-TEMP','EXP-0020').replace('exp14','exp16').replace('EXP-0015','EXP-0019')

t=renumber((PARENT/'run_split_e2e.py').read_text())
t=t.replace('GQA prefill stage1','GQA prefill page-id broadcast').replace('route_gate_version=5','route_gate_version=6')
old='prefill_rows=64 prefill_warps=4 prefill_stages=1 prefill_tile=32'
t=once(t,"log.count('decode_warps=4 decode_stages=1 segments=16 decode_layout=1 decode_rows=8 "+old+"')",
         "log.count('decode_warps=4 decode_stages=1 segments=16 decode_layout=1 decode_rows=8 "+old+" prefill_layout=1')")
t=once(t,"assert re.search(r'tt\\.load[^\\n]*tensor<32x!tt\\.ptr<i32>>', ir), 'Expected prefill tile32 page-table loads'",
       "assert re.search(r'tt\\.load[^\\n]*tensor<2x!tt\\.ptr<i32>>', ir), 'Expected prefill two-page loads'\n"
       "            assert re.search(r'tt\\.broadcast[^\\n]*tensor<2x1xi64> -> tensor<2x16xi64>', ir), 'Missing prefill page-id broadcast'\n"
       "            assert re.search(r'tt\\.reshape[^\\n]*tensor<2x16xi64> -> tensor<32xi64>', ir), 'Missing prefill token-lane expansion'")
t=once(t,'page_table_load_elements=2 if decode else 32','page_table_load_elements=2')
t=once(t,'prefill_rows=64,prefill_warps=4,prefill_stages=1,prefill_tile=32), kernels=kernels,',
       'prefill_rows=64,prefill_warps=4,prefill_stages=1,prefill_tile=32,prefill_layout=1), kernels=kernels,')
t=once(t,"item['metrics']['Total tok/s']['paged_layout']","item['metrics']['Total tok/s']['prefill_stage1']")
t=t.replace("ROOT/'artifacts/run_window.json'","ROOT/'artifacts/EXP-0020/run_window.json'")
t=once(t,"assert args.finish_before > datetime.now().astimezone()", "assert args.finish_before > datetime.now().astimezone()\n    assert args.finish_before.date()==datetime.now().astimezone().date()")
t=once(t,"    assert gate['candidate_commits'] == CANDIDATE_COMMITS", "    assert gate['candidate_commits'] == CANDIDATE_COMMITS\n"
       "    for name, expected in gate['script_hashes'].items():\n"
       "        assert hashlib.sha256((EXP/name).read_bytes()).hexdigest()==expected, name\n"
       "    for name, expected in CANDIDATE_COMMITS.items():\n"
       "        assert subprocess.check_output(['git','-C',str(WORKTREES/name),'rev-parse','HEAD'],text=True).strip()==expected\n"
       "        assert not subprocess.check_output(['git','-C',str(WORKTREES/name),'status','--porcelain'],text=True)")
t=once(t,"        assert not ready(), 'Port 9031 is occupied'", "        assert not ready(), 'Port 9031 is occupied'\n"
       "        observations=[]\n"
       "        import socket\n"
       "        for port in (9031,9032):\n"
       "            with socket.socket() as sock: assert sock.connect_ex(('127.0.0.1',port))!=0\n"
       "        for _ in range(2):\n"
       "            output=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True,timeout=20)\n"
       "            m=re.search(r'(\\d+)MiB\\s*/\\s*(\\d+)MiB\\s*\\|\\s*(\\d+)%',output)\n"
       "            assert m and int(m[1])<=1024 and int(m[3])<=5,'GPU not idle'\n"
       "            observations.append(dict(memory_mib=int(m[1]),util_percent=int(m[3])));time.sleep(3)\n"
       "        (RUN/'idle_evidence.json').write_text(json.dumps(observations,indent=2)+'\\n')")
t=once(t,"        optimized = validate_benchmark(RUN/'candidate_benchmark', selected_cases)",
         "        verify_candidate_gate()\n        optimized = validate_benchmark(RUN/'candidate_benchmark', selected_cases)")
(OUT/'run_split_e2e.py').write_text(t)
for name in ('launch_split.py','finish_split.py'):
    t=renumber((PARENT/name).read_text())
    t=t.replace('GQA prefill 单级流水','GQA prefill 页号广播').replace('GQA prefill-stage1','GQA prefill-page-broadcast')
    t=t.replace('4K 的 EXP-0016 CV 为 4.43%','4K 的 EXP-0016 CV 为 3.16%')
    t=t.replace("root/'artifacts/run_window.json'","root/'artifacts/EXP-0020/run_window.json'")
    if name=='launch_split.py':
        t=once(t,'if args.finish_before:\n    minimum_seconds', 'assert args.finish_before.date()==now.date()\nif args.finish_before:\n    minimum_seconds')
        t=once(t,'latest = root/"artifacts/EXP-0020-latest-official.json"',
               'assert not (root/("artifacts/EXP-0020-latest-"+args.case+".json")).exists(), "Prior attempt exists; inspect instead of rerunning"\nlatest = root/"artifacts/EXP-0020-latest-official.json"')
    else:
        t=once(t,"'candidate_validation.json','parent_reference_verification.json'", "'candidate_validation.json','idle_evidence.json','parent_reference_verification.json'")
    (OUT/name).write_text(t)
shutil.copy2(PARENT/'analyze_schedule.py',OUT/'analyze_schedule.py')
for p in OUT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
assert (OUT/'run_split_e2e.py').read_text().split('SERVER =')[1].split('STATE =')[0]==(PARENT/'run_split_e2e.py').read_text().split('SERVER =')[1].split('STATE =')[0]
print('EXP20 harness derived; official workload unchanged.')
