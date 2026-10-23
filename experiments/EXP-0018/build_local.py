"""Derive a new diagnostic from verified, immutable earlier experiments."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SOURCES = {}

def source(rel, expected=None):
    p = BASE / rel
    text = p.read_text()
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if expected:
        assert digest == expected, rel
    SOURCES[rel] = digest
    return text

def replace(s, old, new):
    assert s.count(old) == 1, (old[:100], s.count(old))
    return s.replace(old, new)

hooks = source('flagos-exp9-dev/profile_hooks.py', '567b3f4199643a615a41dce4989f1c990aec8db0995d0a0109ecf484c582adc6')
hooks = replace(hooks, 'assert window in ("initial", "steady", "late")', '''assert window in ("initial", "mixed", "large_decode", "tail")
        if window == "initial":
            # Official serve.py starts profiling AFTER endpoint check/warmup
            # and BEFORE submitting the main 256 requests. Reset only here.
            assert not CURRENT.get("main_workload_started", False)
            CURRENT["main_workload_started"] = True
            CURRENT["new_requests_total"] = 0
        CURRENT["window_profiled_steps"] = 0''')
hooks = replace(hooks, 'CURRENT["active"] = bool(self.profiler._running)', '''CURRENT["active"] = bool(self.profiler._running)
        if CURRENT["active"]:
            CURRENT["window_profiled_steps"] = CURRENT.get("window_profiled_steps", 0) + 1''')
hooks = replace(hooks, 'q = [int(x) for x in counts]', '''q = [int(x) for x in counts]
            assert len(q) == n and n > 0''')
hooks = replace(hooks, 'emit_meta("batch", meta)', '''if CURRENT["active"]:
                detailed = dict(meta, query_lengths=q,
                                context_lengths_cpu_upper_bound=lengths,
                                computed_tokens_cpu_upper_bound=[int(x) for x in computed],
                                request_phases=["prefill" if int(c) < int(p) else "decode"
                                                for c, p in zip(computed, prompt)],
                                main_requests_entered=CURRENT["new_requests_total"],
                                window=CURRENT.get("window"))
                emit_meta("batch", detailed)''')
hooks = replace(hooks, 'profiling=CURRENT["active"])', '''profiling=CURRENT["active"],
                              window=CURRENT.get("window"),
                              window_profiled_steps=CURRENT.get("window_profiled_steps", 0),
                              main_workload_started=CURRENT.get("main_workload_started", False))''')
hooks = replace(hooks, 'tmp.replace(ROOT / "latest_batch.json")', '''tmp.replace(ROOT / "latest_batch.json")
                with (ROOT / "batch_timeline.jsonl").open("a") as f:
                    f.write(json.dumps(status) + "\\n")''')
(HERE/'profile_hooks.py').write_text(hooks)

route = source('flagos-exp17-dev/route_checks.py', 'd18e816bf4c35158057b159f01dcf455d9ef1a60a53607f8eda01348492fbcda')
route = replace(route, 'def verify_route(run, arm, output=None):', 'def verify_route(run, mode, output=None):')
route = replace(route, 'folder = run/"candidate_benchmark"', 'folder = run')
route = replace(route, 'expected_prefill_stages = 2 if arm == "A" else 1\n    assert arm in ("A", "B")', 'expected_prefill_stages = 1\n    assert mode in ("FULL_DECODE_ONLY", "NONE")')
route = replace(route, "RUN/'candidate_cache'", "RUN/'cache'")
route = replace(route, '''    assert ('Capturing CUDA graphs (decode, FULL)' in log and '35/35' in log), (
        'Complete FULL_DECODE_ONLY graph capture not observed')''', '''    captured = 'Capturing CUDA graphs (decode, FULL)' in log and '35/35' in log
    if mode == "FULL_DECODE_ONLY":
        assert captured, 'Complete FULL_DECODE_ONLY graph capture not observed'
    else:
        assert not captured, 'NONE attribution pass unexpectedly captured FULL graphs' ''')
start = route.index('    expected_log = ')
end = route.index("    assert 'KVReservationAsyncScheduler'", start)
route = route[:start]+'''    expected_log = 'decode_warps=4 decode_stages=1 segments=16 decode_layout=1 decode_rows=8 prefill_rows=64 prefill_warps=4 prefill_stages=1 prefill_tile=32'
    assert log.count(expected_log) == 42
'''+route[end:]
route = replace(route, "dict(status='passed', gqa_launch=", "dict(status='passed', mode=mode, graph_capture_35=captured, gqa_launch=")
(HERE/'route_checks.py').write_text(route)

analysis = source('flagos-profile-final-20261021/analyze_profile_attribution.py', '0b07f3b750660461ff9142ecd9d5ae78318d8d3b5a07d9089b7eadf19fe8f1e1')
analysis = analysis[:analysis.index('\ndef main(run):')]
analysis = replace(analysis, 'csv.DictWriter(f, fieldnames=fields)', 'csv.DictWriter(f, fieldnames=fields, lineterminator="\\n")')
analysis += (HERE/'analysis_main.py.fragment').read_text()
(HERE/'analyze_profile.py').write_text(analysis)
(HERE/'probe_profiler.py').write_text(source('flagos-exp9-dev/probe_profiler.py', '7729287ede94a218182fc0209150807dfe2d01b6096aa1d4fc5a07157120dd50'))
(HERE/'source_reuse.json').write_text(json.dumps(SOURCES, indent=2)+'\n')
