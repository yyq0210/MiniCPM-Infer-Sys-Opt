"""CPU-only source/trace audit; never imports torch or changes running code."""
import ast
from collections import Counter
from datetime import datetime
import hashlib
import gzip
import json
from pathlib import Path
import subprocess

ROOT = Path('/workspace/LeanScale')
REPO = ROOT / 'MiniCPM-Infer-Sys-Opt'
DEST = REPO / 'docs/evidence/gqa_analysis_20261021'
DEST.mkdir(parents=True, exist_ok=True)
RUN = ROOT / 'artifacts/EXP-0009/20261021_131254_profile_4k'
SUMMARY = RUN / 'interim_audit_1338/late'

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

evidence = {'at': datetime.now().astimezone().isoformat(), 'source_index': {},
            'profile_run': str(RUN), 'profile_status_at_review': read(RUN / 'status.json'),
            'benchmark_class': 'diagnostic NONE/late, not official throughput or production Graph latency'}
paths = {'FlagGems': 'src/flag_gems/runtime/backend/_iluvatar/fused/paged_gqa.py',
         'vllm-plugin-FL': 'vllm_fl/dispatch/backends/flaggems/impl/iluvatar_gqa.py'}
for repo, rel in paths.items():
    work = ROOT / 'worktrees/EXP-0009' / repo
    p = work / rel
    tree = ast.parse(p.read_text())
    evidence['source_index'][repo] = {
        'commit': subprocess.check_output(['git', '-C', str(work), 'rev-parse', 'HEAD'], text=True).strip(),
        'path': rel, 'sha256': sha(p),
        'definitions': {n.name: n.lineno for n in ast.walk(tree)
                        if isinstance(n, (ast.FunctionDef, ast.ClassDef))},
    }
for name in ('summary.json', 'step_metadata.json', 'operator_times.csv', 'kernel_times.csv'):
    p = SUMMARY / name
    body = p.read_bytes()
    # Derived CSV serialization only; raw trace and numeric fields are untouched.
    (DEST / name).write_bytes(body.replace(b'\r\n', b'\n') if p.suffix == '.csv' else body)
meta = read(SUMMARY / 'step_metadata.json')
mapping = read(SUMMARY / 'kernel_mapping.json')
gqa = [x for x in mapping if x['op'] in ('OP05_06_GQA', 'OP07_GQA_reduce')]
counts = Counter((x['step'], x['name']) for x in gqa)
assert len(meta) == 16 and all(m['phase'] == 'decode' and m['B'] == 64 for m in meta.values())
assert len(counts) == 32 and all(n == 42 for n in counts.values())
evidence['per_step_gqa_kernel_calls'] = [dict(step=s, kernel=k, calls=n) for (s, k), n in sorted(counts.items())]
evidence['launch_shapes'] = [dict(kernel=k, grid=json.loads(g), block=json.loads(b))
                           for k, g, b in sorted({(x['name'], json.dumps(x['grid']), json.dumps(x['block'])) for x in gqa})]
trace = next((RUN / 'NONE/traces/late').glob('*.gz'))
events = json.loads(gzip.decompress(trace.read_bytes()))['traceEvents']
resources = Counter((e['args'].get('registers per thread'), e['args'].get('shared memory'),
                     json.dumps(e['args'].get('block')))
                    for e in events if e.get('name') == '_attention' and e.get('cat') == 'kernel')
evidence['runtime_kernel_resource_metadata'] = [dict(registers_per_thread=r, shared_bytes=s,
                                                    block=json.loads(b), calls=n)
                                               for (r, s, b), n in resources.items()]
evidence['context_bounds'] = {'min_cpu_upper_bound': min(m['L_min_cpu_upper_bound'] for m in meta.values()),
                              'max_cpu_upper_bound': max(m['L_max_cpu_upper_bound'] for m in meta.values()),
                              'note': 'CPU metadata upper bounds, not direct device sequence-length readings'}
evidence['compiler_cache_samples'] = []
for folder in ['OA5DGT6NS7MN5UDZZOLMFJOYTJU4ZW4HMSZL7X423WHSZ7NWKEAQ',
               'WMDHRNRC7RKEYWYFFVELVUYQY3TAPSGGOFRBHGTHDTPQWN44MWXQ',
               'MANRINY3MZDJNS4G777VVZSM4VEXPFNGJOD45Z22IZB27IVSC3NA']:
    p = Path('/root/.triton/cache') / folder / '_attention.json'
    metadata = read(p)
    ttir, ttgir, llir = [p.with_suffix('.' + suffix) for suffix in ('ttir', 'ttgir', 'llir')]
    assembly = p.with_suffix('.asm')
    evidence['compiler_cache_samples'].append({
        'cache_key': folder, 'metadata_sha256': sha(p),
        'target': metadata['target'],
        'config': {k: metadata[k] for k in ('num_warps', 'num_stages', 'shared')},
        'ir_sha256': {f.suffix: sha(f) for f in (ttir, ttgir, llir)},
        'source_location': ttir.read_text().splitlines()[0],
        'dot_ir': [l.strip() for l in ttir.read_text().splitlines() if 'tt.dot' in l],
        'llvm_exp2_call_count': llir.read_text().count('call float @llvm.exp2.f32'),
        'layout_conversion_count': ttgir.read_text().count('ttg.convert_layout'),
        'assembly_resource_lines': [l.strip() for l in assembly.read_text().splitlines()
                                    if any(k in l for k in ('.sgpr_count:', '.vgpr_count:',
                                                           '.sgpr_spill_count:', '.vgpr_spill_count:', 'ScratchSize:'))],
        'limitation': 'Historical compile-cache sample for the same source family; trace provides no cache-key identity. No claim of exact per-layer register or occupancy values.',
    })
prior = ROOT / 'artifacts/EXP-0005/20261020_131536/diagnostic.json'
original = read(prior) if prior.exists() else None
if original is not None:
    evidence['prior_decode_sweeps'] = [t for t in original['timings'] if t['name'].startswith('decode_')]
    evidence['prior_decode_source'] = dict(path=str(prior), sha256=sha(prior))
(DEST / 'evidence.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'evidence_directory': str(DEST), 'source_index': evidence['source_index'],
                  'profile_stage': evidence['profile_status_at_review']['stage'],
                  'per_step_attention_reduce_calls_verified': True}, ensure_ascii=False))
