"""Derive the prefill E2E harness from frozen EXP14; preserve official workload."""
from pathlib import Path
import ast
import shutil

OUT = Path(__file__).resolve().parent
PARENT = OUT / 'parent_exp14'


def replace_once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def renumber(text):
    return text.replace('EXP-0014', 'EXP-TEMP').replace('EXP-0012', 'EXP-0014').replace('EXP-TEMP', 'EXP-0016').replace('exp12', 'exp14').replace('EXP-0013', 'EXP-0015')


t = renumber((PARENT / 'run_split_e2e.py').read_text())
t = t.replace('GQA paged KV and eight-row experiment', 'GQA prefill stage1 experiment')
t = t.replace('GQA paged KV broadcast and eight-row decode', 'GQA prefill stage1')
t = t.replace('route_gate_version=4', 'route_gate_version=5')
old_log = 'decode_warps=4 decode_stages=1 segments=16 decode_layout=1 decode_rows=8'
new_log = old_log + ' prefill_rows=64 prefill_warps=4 prefill_stages=1 prefill_tile=32'
t = t.replace("log.count('" + old_log + "')", "log.count('" + new_log + "')")
start = t.index('    kernels = []\n', t.index('def verify_route'))
end = t.index('\n\ndef run_command', start)
replacement = r'''    kernels = []
    paths = list((RUN/'triton_cache').glob('*/_attention.json'))
    assert paths, 'No isolated service-cache GQA metadata'
    for path in paths:
        metadata = json.loads(path.read_text())
        assert metadata.get('num_warps') == 4 and metadata.get('num_stages') == 1, 'Unexpected GQA launch parameters'
        assert metadata['target']['backend'] == 'corex'
        irpath = path.with_suffix('.ttir')
        assert irpath.is_file(), 'Missing actual service TTIR'
        ir = irpath.read_text()
        decode = bool(re.search(r'tt\.load[^\n]*tensor<8x128x!tt\.ptr<(?:bf16|f16)>>', ir))
        prefill = bool(re.search(r'tt\.load[^\n]*tensor<64x128x!tt\.ptr<(?:bf16|f16)>>', ir))
        assert decode != prefill, 'Ambiguous or unsupported GQA query layout'
        if decode:
            assert metadata.get('shared') == 18432, 'Unexpected decode shared-memory allocation'
            assert re.search(r'tt\.load[^\n]*tensor<2x!tt\.ptr<i32>>', ir), 'Expected two page-table entries'
            assert re.search(r'tt\.broadcast[^\n]*tensor<2x1xi64> -> tensor<2x16xi64>', ir), 'Missing page-id broadcast'
            assert re.search(r'tt\.reshape[^\n]*tensor<2x16xi64> -> tensor<32xi64>', ir), 'Missing token-lane expansion'
        else:
            assert metadata.get('shared') == 32768, 'Unexpected prefill shared-memory allocation'
            assert re.search(r'tt\.load[^\n]*tensor<32x!tt\.ptr<i32>>', ir), 'Expected prefill tile32 page-table loads'
            assert re.search(r'tt\.load[^\n]*tensor<128x32x!tt\.ptr<(?:bf16|f16)>>', ir), 'Expected prefill K tile32'
            assert re.search(r'tt\.dot[^\n]*tensor<64x32xf32>', ir), 'Expected prefill QK accumulator layout'
        kernels.append(dict(path=str(path.relative_to(RUN)), kind='decode' if decode else 'prefill',
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            ttir_sha256=hashlib.sha256(irpath.read_bytes()).hexdigest(),
                            num_warps=4, num_stages=1, shared=metadata['shared'],
                            logical_query_rows=8 if decode else 64, tile=32,
                            page_table_load_elements=2 if decode else 32))
    counts = Counter(k['kind'] for k in kernels)
    assert counts['decode'] and counts['prefill'], 'Both actual decode and prefill compile evidence required'
    (folder/'route_evidence.json').write_text(json.dumps(
        dict(status='passed', gqa_launch=dict(decode_warps=4,decode_stages=1,segments=16,decode_layout=1,decode_rows=8,
             prefill_rows=64,prefill_warps=4,prefill_stages=1,prefill_tile=32), kernels=kernels,
             kernel_counts=dict(counts), fused_runtime_layers=42, gqa_layers=42,
             old_cache_updates_in_fused_graph=0, graphs=graphs), indent=2)+'\n')
'''
t = t[:start] + replacement + t[end:]
start = t.index('def verified_parent_reference(')
end = t.index('\ndef main():', start)
helper = '''def verified_parent_reference(case, selected_cases):
    selection_path = EXP.parent/'EXP-0014/selection.json'
    selection = json.loads(selection_path.read_text())
    parent_commits = json.loads((EXP/'parent_commits.json').read_text())
    assert selection['selected_experiment'] == 'EXP-0014'
    assert selection['selected_serving_commits'] == parent_commits
    item = next(x for x in selection['case_results'] if x['case'] == case)
    parent_run = ROOT/'artifacts/EXP-0014'/item['run_name']
    raw_state = json.loads((parent_run/'status.json').read_text())
    assert raw_state['commits'] == parent_commits
    assert raw_state['stage'] == 'performance_case_complete'
    handler = json.loads((parent_run/'completion_handler_status.json').read_text())
    assert handler['complete_results'] and handler['status'] == 'pushed' and handler['remote_verified']
    folder = parent_run/'candidate_benchmark'
    route_path = folder/'route_evidence.json'
    assert json.loads(route_path.read_text())['status'] == 'passed'
    assert hashlib.sha256(route_path.read_bytes()).hexdigest() == item['route_sha256']
    raw_path = sorted((folder/'benchmark_results').glob('raw_runs_*.csv'))[-1]
    summary_path = sorted((folder/'benchmark_results').glob('summary_*.csv'))[-1]
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == item['raw_sha256']
    assert hashlib.sha256(summary_path.read_bytes()).hexdigest() == item['summary_sha256']
    result = validate_benchmark(folder, selected_cases)
    assert float(result[1][0]['Total tok/s']) == item['metrics']['Total tok/s']['paged_layout']
    (RUN/'parent_reference_verification.json').write_text(json.dumps(dict(
        status='passed', selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        parent_commits=parent_commits, source=str(folder), raw_stage_preserved=raw_state['stage'],
        raw_sha256=item['raw_sha256'], summary_sha256=item['summary_sha256'], route_sha256=item['route_sha256']), indent=2)+'\\n')
    return folder, result

'''
t = t[:start] + helper + t[end:]
(OUT/'run_split_e2e.py').write_text(t)
for name in ('launch_split.py', 'queue_official.py', 'finish_split.py'):
    t = renumber((PARENT/name).read_text())
    t = t.replace('GQA 分页读取与8行布局', 'GQA prefill 单级流水')
    t = t.replace('GQA paged-layout performance case', 'GQA prefill-stage1 performance case')
    t = t.replace('4K 的 EXP-0014 CV 为 3.69%', '4K 的 EXP-0014 CV 为 4.43%')
    t = t.replace('显式选择分页页号广播与8行decode布局', '显式选择prefill stage1，并保留分页页号广播与8行decode布局')
    (OUT/name).write_text(t)
shutil.copy2(PARENT/'analyze_schedule.py', OUT/'analyze_schedule.py')
for p in OUT.glob('*.py'):
    ast.parse(p.read_text(), filename=str(p))
assert (PARENT/'run_split_e2e.py').read_text().split('SERVER =')[1].split('STATE =')[0] == (OUT/'run_split_e2e.py').read_text().split('SERVER =')[1].split('STATE =')[0]
print('EXP16 harness generated; fixed server/benchmark parameters unchanged.')
