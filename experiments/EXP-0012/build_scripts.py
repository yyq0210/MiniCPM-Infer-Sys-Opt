"""Derive a frozen GQA E2E harness from previously exercised experiment tools."""
from pathlib import Path
root=Path(__file__).resolve().parent.parent;out=Path(__file__).resolve().parent
old=(root/'flagos-exp10-dev/run_split_e2e.py').read_text()
fusion=(root/'flagos-exp6-dev/run_split_e2e.py').read_text()
route=fusion[fusion.index('def verify_route('):fusion.index('\ndef run_command(')]
needle="    (folder/'route_evidence.json').write_text(json.dumps("
extra='''    assert log.count('decode_warps=4 decode_stages=1 segments=16') == 42
    assert 'KVReservationAsyncScheduler' not in log
    assert 'KV growth admission enabled=True' not in log
    kernels = []
    for path in (RUN/'triton_cache').glob('*/_attention.json'):
        metadata = json.loads(path.read_text())
        if metadata.get('num_warps') == 4 and metadata.get('num_stages') == 1:
            assert metadata['target']['backend'] == 'corex'
            kernels.append(dict(path=str(path.relative_to(RUN)),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                num_warps=4, num_stages=1, shared=metadata.get('shared')))
    assert kernels, 'No isolated service-cache stage1 GQA kernel metadata found'
'''
route=route.replace(needle,extra+needle).replace("dict(status='passed', fused_runtime_layers=42, gqa_layers=42,",
      "dict(status='passed', gqa_launch=dict(warps=4,stages=1,segments=16), kernels=kernels,\n             fused_runtime_layers=42, gqa_layers=42,")
t=old[:old.index('def verify_route(')]+route+old[old.index('\ndef run_command('):]
t=t.replace('EXP-0010','EXP-0012')
t=t.replace("assert path.is_relative_to(ROOT/'artifacts/EXP-0012')","assert path.is_relative_to(ROOT/'artifacts/EXP-0011')")
t=t.replace("verify_route(RUN/'candidate_benchmark', require_capacity_checks=True)","verify_route(RUN/'candidate_benchmark')")
t=t.replace("result['VLLM_CACHE_ROOT'] = str(RUN/'candidate_cache')", "result['VLLM_CACHE_ROOT'] = str(RUN/'candidate_cache')\n        result['TRITON_CACHE_DIR'] = str(RUN/'triton_cache')")
t=t.replace('route_gate_version=2','route_gate_version=3')
t=t.replace('Full max_tokens KV growth admission on EXP-0006 fusion;','GQA decode stage1 on EXP-0006 fusion with original scheduling;')
t=t.replace('EXP-0012 official KV admission experiment, preserving the EXP-0006 kernels.',
            'EXP-0012 official GQA stage1 experiment, preserving EXP-0006 fusion and scheduling.')
(out/'run_split_e2e.py').write_text(t)
t=(root/'flagos-exp10-dev/launch_split.py').read_text().replace('EXP-0010','EXP-0012')
t=t.replace('now = datetime.datetime.now().astimezone()', '''now = datetime.datetime.now().astimezone()
policy = json.loads((root/'artifacts/run_window.json').read_text())
assert args.finish_before is not None
assert args.finish_before == datetime.datetime.fromisoformat(policy['compute_stop_at'])''')
(out/'launch_split.py').write_text(t)
t=(root/'flagos-exp10-dev/finish_split.py').read_text().replace('EXP-0010','EXP-0012')
t=t.replace('KV 增长准入','GQA decode 单级流水').replace('完整输出上界的 KV 增长准入','GQA decode stage1 参数')
t=t.replace('相对 EXP-0006 的调度增量','相对 EXP-0006 的 GQA 增量').replace('调度 tok/s','GQA tok/s').replace('调度候选','GQA 候选')
t=t.replace('归因于调度','归因于GQA参数').replace('KV admission performance case','GQA stage1 performance case')
t=t.replace("target.write_text(sanitize(data.decode(errors='replace')))","target.write_text(sanitize(data.decode(errors='replace')).replace('\\r\\n','\\n'))")
t=t.replace('target.write_bytes(source.read_bytes())',"target.write_bytes(source.read_bytes().replace(b'\\r\\n',b'\\n'))")
t=t.replace('def publish(run):','def _publish_unlocked(run):')
t=t.replace("git('commit','-m',message)","git('commit','--only','-m',message,'--',*paths)")
insert='''def publish(run):
    with Path('/workspace/LeanScale/artifacts/docs-publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _publish_unlocked(run)


'''
t=t.replace('def main():',insert+'def main():')
(out/'finish_split.py').write_text(t)
(out/'analyze_schedule.py').write_text((root/'flagos-exp10-dev/analyze_schedule.py').read_text())
