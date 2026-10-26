"""Derive a two-arm route gate from the frozen, audited EXP16 checker."""
import ast
from pathlib import Path

p = Path(__file__).resolve().parent
source = (p/'frozen_runtime.py').read_text()
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'verify_route')
s = ast.get_source_segment(source, node)
s = s.replace('def verify_route(folder):', 'def verify_route(run, arm, output=None):\n    RUN = run\n    folder = run/"candidate_benchmark"\n    expected_prefill_stages = 2 if arm == "A" else 1\n    assert arm in ("A", "B")')
old = 'decode_warps=4 decode_stages=1 segments=16 decode_layout=1 decode_rows=8'
new = old+' prefill_rows=64 prefill_warps=4 prefill_stages=1 prefill_tile=32'
s = s.replace(f"    assert log.count('{new}') == 42", f"    expected_log = '{old}' if arm == 'A' else '{new}'\n    assert log.count(expected_log) == 42\n    if arm == 'A': assert 'prefill_stages=' not in log")
s = s.replace("metadata.get('num_warps') == 4 and metadata.get('num_stages') == 1", "metadata.get('num_warps') == 4")
s = s.replace("        if decode:\n", "        expected_stages = 1 if decode else expected_prefill_stages\n        assert metadata.get('num_stages') == expected_stages, 'Unexpected decode/prefill pipeline stages'\n        if decode:\n")
s = s.replace('num_warps=4, num_stages=1, shared=', 'num_warps=4, num_stages=expected_stages, shared=')
s = s.replace("(folder/'route_evidence.json').write_text", "(output or folder/'route_evidence.json').write_text")
s = s.replace('prefill_stages=1,prefill_tile=32)', 'prefill_stages=expected_prefill_stages,prefill_tile=32)')
assert 'num_stages=1, shared=' not in s and 'expected_stages' in s
ast.parse(s)
(p/'route_checks.py').write_text('import hashlib\nimport json\nimport re\nfrom collections import Counter\n\n'+s+'\n')
