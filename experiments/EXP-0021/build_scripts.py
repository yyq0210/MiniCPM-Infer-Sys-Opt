"""Derive fixed-order ABBA using audited EXP17 orchestration and EXP20 runtime."""
from pathlib import Path
import ast
import hashlib
import shutil

P=Path(__file__).resolve().parent
OLD=P/'parent_exp17';RT=P/'parent_exp20/run_split_e2e.py'
EXPECTED={'common.py':'c4eb93394efd902a949ef9bf9c1c3d8c7e60fa6d50f0dc39b1fb178238da6249',
 'run_abba.py':'e8acab6d055ca52feae095bc13d15a2c4ee5013cc559faa47909d60a52734baa',
 'analyze_schedule.py':'574042b1d7d97454691b3608f2fe0073cc760a6c0314a1cd1315bc95a769ea9f'}
for n,h in EXPECTED.items():assert hashlib.sha256((OLD/n).read_bytes()).hexdigest()==h
assert hashlib.sha256(RT.read_bytes()).hexdigest()=='2e599679dfe365de907b5071cee469b08489b66ff371588bf78c0f1782281ab1'
def once(t,a,b):
    assert t.count(a)==1,(a,t.count(a));return t.replace(a,b)

t=(OLD/'common.py').read_text().replace('EXP-0017','EXP-0021')
t=t.replace('82c6a143f6984f946df42125f1a8699799e8045f','588a6147cd291e272324ab6c812024cd587c6287').replace('1c526ca99d412e203ed3caa8b1663566976ea38c','fd209df717648002ebf5ba4ec351424dcc997a53')
old="'B': {'vllm-plugin-FL':'588a6147cd291e272324ab6c812024cd587c6287', 'FlagGems':'fd209df717648002ebf5ba4ec351424dcc997a53'}"
t=once(t,old,"'B': {'vllm-plugin-FL':'ce19f93eee4c73f5a26911554700457e879f891d', 'FlagGems':'de958ddfc4deeb43d6c94bb8652a3d4d3bc905fe'}")
(P/'common.py').write_text(t)
shutil.copy2(RT,P/'frozen_runtime.py');shutil.copy2(OLD/'analyze_schedule.py',P/'analyze_schedule.py')
source=RT.read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='verify_route')
s=ast.get_source_segment(source,node)
s=once(s,'def verify_route(folder):','def verify_route(run, arm, output=None):\n    RUN=run\n    folder=run/"candidate_benchmark"\n    assert arm in ("A","B")\n    prefill_layout=0 if arm=="A" else 1')
log='decode_warps=4 decode_stages=1 segments=16 decode_layout=1 decode_rows=8 prefill_rows=64 prefill_warps=4 prefill_stages=1 prefill_tile=32'
s=once(s,"    assert log.count('"+log+" prefill_layout=1') == 42",
       "    expected_log='"+log+"'+(' prefill_layout=1' if arm=='B' else '')\n    assert log.count(expected_log)==42\n    if arm=='A': assert 'prefill_layout=' not in log")
old=r"""            assert re.search(r'tt\.load[^\n]*tensor<2x!tt\.ptr<i32>>', ir), 'Expected prefill two-page loads'
            assert re.search(r'tt\.broadcast[^\n]*tensor<2x1xi64> -> tensor<2x16xi64>', ir), 'Missing prefill page-id broadcast'
            assert re.search(r'tt\.reshape[^\n]*tensor<2x16xi64> -> tensor<32xi64>', ir), 'Missing prefill token-lane expansion'"""
new=r"""            if arm=='A':
                assert re.search(r'tt\.load[^\n]*tensor<32x!tt\.ptr<i32>>',ir), 'Expected baseline 32 page-table loads'
            else:
                assert re.search(r'tt\.load[^\n]*tensor<2x!tt\.ptr<i32>>',ir), 'Expected candidate two-page loads'
                assert re.search(r'tt\.broadcast[^\n]*tensor<2x1xi64> -> tensor<2x16xi64>',ir), 'Missing candidate int64 broadcast'
                assert re.search(r'tt\.reshape[^\n]*tensor<2x16xi64> -> tensor<32xi64>',ir), 'Missing candidate expansion'"""
s=once(s,old,new)
s=once(s,'page_table_load_elements=2))','page_table_load_elements=2 if decode or arm=="B" else 32))')
s=once(s,'prefill_tile=32,prefill_layout=1)','prefill_tile=32,prefill_layout=prefill_layout)')
s=s.replace("(folder/'route_evidence.json').write_text","(output or folder/'route_evidence.json').write_text")
(P/'route_checks.py').write_text('import hashlib\nimport json\nimport re\nfrom collections import Counter\n\n'+s+'\n')
t=(OLD/'run_abba.py').read_text().replace('EXP-0017','EXP-0021').replace('EXP-0016','EXP-0020').replace('EXP16 official result','EXP20 official result')
t=t.replace('A：原stage2 | B：stage1','A：原页表读取 | B：页号广播')
t=t.replace("read(ROOT/'artifacts/run_window.json')","read(ART/'run_window.json')")
t=once(t,"    assert remaining(deadline)>=4*55*60+10*60,'Insufficient budget for predetermined ABBA'", "    assert deadline.date()==now().date()\n    assert remaining(deadline)>=4*55*60+10*60,'Insufficient budget for predetermined ABBA'")
t=once(t,"        assert sha(rt.BENCH_FILE)==rt.BENCH_SHA", r"""        import socket
        idle=[]
        for port in (9031,9032):
            with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',port))!=0
        for _ in range(2):
            out=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True,timeout=20)
            m=re.search(r'(\d+)MiB\s*/\s*(\d+)MiB\s*\|\s*(\d+)%',out)
            assert m and int(m[1])<=1024 and int(m[3])<=5,'GPU not idle'
            idle.append(dict(memory_mib=int(m[1]),util_percent=int(m[3])));time.sleep(3)
        write(run/'idle_evidence.json',idle)
        assert sha(rt.BENCH_FILE)==rt.BENCH_SHA""")
t=once(t,"['status.json','environment_check.json','result.json','error.json','stage_history.jsonl']","['status.json','environment_check.json','result.json','error.json','stage_history.jsonl','idle_evidence.json']")
(P/'run_abba.py').write_text(t)
for x in P.glob('*.py'):ast.parse(x.read_text(),filename=str(x))
print('EXP21 generated: unchanged official workload, both arms stage1, only prefill page layout differs.')
