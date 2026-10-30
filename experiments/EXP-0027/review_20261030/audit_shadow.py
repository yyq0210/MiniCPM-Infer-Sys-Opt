"""Audit the immutable EXP27 archive; never treat diagnostic timing as E2E."""
import argparse
import collections
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

ARCHIVE_COMMIT = 'd5437268cd536214363ab4dac3038fb69c810c86'


def audit(root):
    manifest = json.loads((root / 'artifact_manifest.json').read_text())
    verified = {}
    for name, hashes in manifest.items():
        value = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert value == hashes['archived_sha256'], (name, value, hashes)
        verified[name] = value
    status = json.loads((root / 'status.json').read_text())
    assert status['stage'] == 'failed' and status['owned_children_reaped']
    assert 'No real-workload B64/.95 exact shadow evidence' in status['error']
    start = json.loads((root / 'workload_start.json').read_text())
    began = datetime.fromisoformat(start['started_at'])
    server = (root / 'candidate_benchmark/server.log').read_text()
    # Redaction/normalization changes offsets: use timestamps on archived text.
    def timestamp(line):
        match = re.search(r'INFO (\d\d-\d\d \d\d:\d\d:\d\d)', line)
        assert match, line
        return datetime.strptime(str(began.year) + '-' + match[1], '%Y-%m-%d %H:%M:%S').replace(tzinfo=began.tzinfo)
    shadows = []
    grids = []
    for line in server.splitlines():
        match = re.search(r'LeanScale top-p shadow passed: B=(\d+) V=(\d+).*?p_min=([^ ]+) p_max=([^ ]+) exact_mask=True rng_unchanged=True check=(\d+)', line)
        if match:
            b, v, low, high, check = match.groups()
            shadows.append(dict(time=timestamp(line).isoformat(), batch=int(b), vocab=int(v), p_min=float(low), p_max=float(high), check=int(check)))
        match = re.search(r'LeanScale top-p grid active: B=(\d+) V=(\d+) programs=(\d+) scratch_rows=(\d+) scratch_bytes=(\d+) k=None calls=(\d+) shadow=(True|False)', line)
        if match:
            b, v, programs, rows, size, calls, shadow = match.groups()
            grids.append(dict(time=timestamp(line).isoformat(), batch=int(b), vocab=int(v), programs=int(programs), scratch_rows=int(rows), scratch_bytes=int(size), calls=int(calls), shadow=shadow))
    assert len(shadows) == 32 and [r['check'] for r in shadows] == list(range(1, 33))
    assert all(datetime.fromisoformat(r['time']) > began and r['vocab'] == 130560 and abs(r['p_min'] - .95) < 1e-6 and abs(r['p_max'] - .95) < 1e-6 for r in shadows)
    assert set(r['batch'] for r in shadows) == set(range(17, 33))
    assert all(r['batch'] == r['programs'] and r['scratch_rows'] == 64 and r['scratch_bytes'] == 33423360 and r['shadow'] == 'True' for r in grids)
    first64 = next(r for r in grids if r['batch'] == 64)
    assert first64['calls'] == 96 and first64['time'] > shadows[-1]['time']
    diagnostic = (root / 'candidate_benchmark/diagnostic.log').read_text()
    assert re.findall(r'Successful requests:\s+(\d+)', diagnostic) == ['256']
    assert re.findall(r'Failed requests:\s+(\d+)', diagnostic) == ['0']
    assert re.findall(r'Total input tokens:\s+(\d+)', diagnostic) == ['1048576']
    assert re.findall(r'Total generated tokens:\s+(\d+)', diagnostic) == ['262144']
    command = json.loads((root / 'candidate_benchmark/diagnostic.command.json').read_text())
    for key, value in [('--random-input-len', '4096'), ('--random-output-len', '1024'), ('--max-concurrency', '64'), ('--num-prompts', '256')]:
        assert command[command.index(key) + 1] == value
    assert '--ignore-eos' in command and '--temperature' not in command and '--top-p' not in command
    layout = json.loads((root / 'candidate_benchmark/layout_route_evidence.json').read_text())
    assert layout['status'] == 'passed' and layout['arm'] == 'A' and layout['compiled_mlp_blocks'] == 42
    assert layout['converted_gate_up'] == layout['converted_down'] == 0
    assert layout['candidate_kv_tokens'] == layout['reference_kv_tokens'] == 522352
    return dict(status='passed', scope='Offline audit of archived diagnostic failure; not GPU validation or official E2E', archive_commit=ARCHIVE_COMMIT,
                archived_hashes=verified, raw_hashes_rechecked=False, successful_requests=256, failed_requests=0,
                input_tokens=1048576, output_tokens=262144, workload_started_at=began.isoformat(),
                shadow_checks=32, shadow_batch_counts=dict(collections.Counter(r['batch'] for r in shadows)),
                first_shadow=shadows[0], last_shadow=shadows[-1], first_batch64_grid=first64, active_grid_events=len(grids),
                coverage_failure='Global first32 budget exhausted at B32 before first B64 call96',
                numeric_mismatch_reported=False, official_e2e_completed=False, selected_experiment='EXP-0020',
                diagnostic_total_tok_s=2100.13, diagnostic_timing_is_performance_evidence=False,
                startup_route_evidence_only=True, final_topp_metadata_gate_completed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.archive)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'archived_hashes'}, ensure_ascii=False))
