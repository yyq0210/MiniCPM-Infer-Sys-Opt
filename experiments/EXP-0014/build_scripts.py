"""Derive EXP14 from the corrected, completed EXP12 harness without editing it."""
from pathlib import Path
import ast
import shutil

OUT = Path(__file__).resolve().parent
PARENT = OUT/'parent_exp12'

def replace_once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)

t = (PARENT/'run_split_e2e.py').read_text().replace('EXP-0012', 'EXP-0014')
t = t.replace('EXP-0006', 'EXP-0012').replace('exp6', 'exp12')
t = t.replace('EXP-0011', 'EXP-0013').replace('route_gate_version=3', 'route_gate_version=4')
t = t.replace('GQA decode stage1', 'GQA paged KV broadcast and eight-row decode')
t = t.replace('GQA stage1 experiment', 'GQA paged KV and eight-row experiment')
t = t.replace('import os\n', 'import os\nimport re\n')
t = t.replace("log.count('decode_warps=4 decode_stages=1 segments=16')",
              "log.count('decode_warps=4 decode_stages=1 segments=16 decode_layout=1 decode_rows=8')")
old = '''            kernels.append(dict(path=str(path.relative_to(RUN)),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                num_warps=4, num_stages=1, shared=metadata.get('shared')))'''
new = '''            assert metadata.get('shared') == 18432, 'Unexpected decode shared-memory allocation'
            irpath = path.with_suffix('.ttir')
            assert irpath.is_file(), 'Missing actual service TTIR'
            ir = irpath.read_text()
            assert re.search(r'tt\\.load[^\\n]*tensor<8x128x!tt\\.ptr<(?:bf16|f16)>>', ir), 'Decode Q must have eight rows'
            assert re.search(r'tt\\.load[^\\n]*tensor<2x!tt\\.ptr<i32>>', ir), 'Expected two page-table entries'
            assert re.search(r'tt\\.broadcast[^\\n]*tensor<2x1xi64> -> tensor<2x16xi64>', ir), 'Missing page-id broadcast'
            assert re.search(r'tt\\.reshape[^\\n]*tensor<2x16xi64> -> tensor<32xi64>', ir), 'Missing token-lane expansion'
            kernels.append(dict(path=str(path.relative_to(RUN)),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                ttir_sha256=hashlib.sha256(irpath.read_bytes()).hexdigest(),
                                num_warps=4, num_stages=1, shared=metadata.get('shared'),
                                logical_query_rows=8, page_table_load_elements=2,
                                page_id_broadcast_shape=[2,16]))'''
t = replace_once(t, old, new)
t = t.replace('gqa_launch=dict(warps=4,stages=1,segments=16)',
              'gqa_launch=dict(warps=4,stages=1,segments=16,decode_layout=1,decode_rows=8)')
old = '''        parent_receipt = json.loads((ROOT/('artifacts/EXP-0012-latest-'+args.case+'.json')).read_text())
        parent_run = Path(parent_receipt['run_directory'])
        parent_state = json.loads((parent_run/'status.json').read_text())
        assert parent_state['stage'] == 'performance_case_complete'
        assert parent_state['commits'] == json.loads((EXP/'parent_commits.json').read_text())
        assert json.loads((parent_run/'completion_handler_status.json').read_text())['complete_results']
        previous_path = parent_run/'candidate_benchmark'
        previous = filter_cases(validate_benchmark(previous_path, selected_cases))'''
new = '''        previous_path, previous = verified_parent_reference(args.case, selected_cases)'''
t = replace_once(t, old, new)
helper = '''def verified_parent_reference(case, selected_cases):
    # EXP12 4K completed all four rounds but its original postprocessor failed.
    # Accept only its explicit audited recovery; never erase that raw failure.
    selection_path = EXP.parent/'EXP-0012/selection.json'
    selection = json.loads(selection_path.read_text())
    parent_commits = json.loads((EXP/'parent_commits.json').read_text())
    assert selection['selected_experiment'] == 'EXP-0012'
    assert selection['selected_serving_commits'] == parent_commits
    item = next(x for x in selection['case_results'] if x['case'] == case)
    parent_run = ROOT/'artifacts/EXP-0012'/item['run_name']
    raw_state = json.loads((parent_run/'status.json').read_text())
    assert raw_state['commits'] == parent_commits
    recovery = None
    if raw_state['stage'] != 'performance_case_complete':
        assert case == '4k' and raw_state['stage'] == 'failed'
        recovery_path = parent_run/'postprocessing_recovery.json'
        recovery = json.loads(recovery_path.read_text())
        assert recovery['status'] == 'complete_case_revalidated_offline'
        assert recovery['all_four_rounds_successful'] and recovery['discard_first']
        assert not recovery['partial_runs_combined']
        assert recovery['raw_error_preserved'] == raw_state['error']
        for relative, expected in recovery['source_hashes_before_and_after'].items():
            assert hashlib.sha256((parent_run/relative).read_bytes()).hexdigest() == expected
    else:
        assert json.loads((parent_run/'completion_handler_status.json').read_text())['complete_results']
    folder = parent_run/'candidate_benchmark'
    assert json.loads((folder/'route_evidence.json').read_text())['status'] == 'passed'
    raw_path = sorted((folder/'benchmark_results').glob('raw_runs_*.csv'))[-1]
    summary_path = sorted((folder/'benchmark_results').glob('summary_*.csv'))[-1]
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == item['raw_sha256']
    assert hashlib.sha256(summary_path.read_bytes()).hexdigest() == item['summary_sha256']
    result = validate_benchmark(folder, selected_cases)
    expected = item['metrics']['Total tok/s']['gqa_stage1']
    assert float(result[1][0]['Total tok/s']) == expected
    (RUN/'parent_reference_verification.json').write_text(json.dumps(dict(
        status='passed', selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        parent_commits=parent_commits, source=str(folder), raw_stage_preserved=raw_state['stage'],
        recovery_status=recovery['status'] if recovery else None,
        raw_sha256=item['raw_sha256'], summary_sha256=item['summary_sha256']), indent=2)+'\\n')
    return folder, result


'''
t = t.replace('def main():\n', helper+'def main():\n')
# Enforce the current dated window even if a runner is invoked without its launcher.
t = replace_once(t, '    args = parser.parse_args()\n', '''    args = parser.parse_args()
    policy = json.loads((ROOT/'artifacts/run_window.json').read_text())
    assert args.finish_before == datetime.fromisoformat(policy['compute_stop_at'])
    assert args.finish_before > datetime.now().astimezone()
''')
(OUT/'run_split_e2e.py').write_text(t)

for name in ('launch_split.py','queue_official.py','finish_split.py'):
    t = (PARENT/name).read_text().replace('EXP-0012','EXP-0014')
    t = t.replace('EXP-0006','EXP-0012').replace('exp6','exp12')
    t = t.replace('GQA decode 单级流水','GQA 分页读取与8行布局')
    t = t.replace('GQA decode stage1 参数','GQA 分页读取与8行布局参数')
    t = t.replace('GQA stage1 performance case','GQA paged-layout performance case')
    t = t.replace('显式选择 GQA decode 单级流水','显式选择分页页号广播与8行decode布局')
    t = t.replace('4K 的 EXP-0012 CV 为 3.64%', '4K 的 EXP-0012 CV 为 3.69%')
    t = t.replace("'candidate_validation.json','coordinator.log','scheduler_comparison.json'",
                  "'candidate_validation.json','parent_reference_verification.json','coordinator.log','scheduler_comparison.json'")
    (OUT/name).write_text(t)
shutil.copy2(PARENT/'analyze_schedule.py', OUT/'analyze_schedule.py')
for p in OUT.glob('*.py'):
    ast.parse(p.read_text(), filename=str(p))
print('Generated EXP14 runner, launch, publisher and queue; official workload unchanged.')
