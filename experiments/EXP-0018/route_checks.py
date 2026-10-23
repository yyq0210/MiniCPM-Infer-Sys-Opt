import hashlib
import json
import re
from collections import Counter

def verify_route(run, mode, output=None):
    RUN = run
    folder = run
    expected_prefill_stages = 1
    assert mode in ("FULL_DECODE_ONLY", "NONE")
    log = (folder/'server.log').read_text(errors='replace')
    assert "Op 'rotary_embedding' using 'default.flagos'" in log
    assert "Op 'rotary_embedding' switched" not in log
    assert log.count('LeanScale GQA active:') == 42, 'Expected 42 GQA layers'
    assert log.count('LeanScale fused RoPE+KV active:') == 42, 'Expected 42 fused runtime layers'
    assert 'LeanScale RoPE+KV graph fusion: replaced=42' in log
    graphs = []
    for path in (RUN/'cache').rglob('computation_graph.py'):
        lines = [line.strip() for line in path.read_text().splitlines()
                 if not line.lstrip().startswith('#')]
        counts = {name: sum('torch.ops.vllm.'+name in line for line in lines)
                  for name in ('fl_rope_kv_dispatch', 'fl_rotary_embedding_dispatch',
                               'unified_kv_cache_update')}
        if counts['fl_rope_kv_dispatch']:
            assert counts == {'fl_rope_kv_dispatch':42,
                              'fl_rotary_embedding_dispatch':0,
                              'unified_kv_cache_update':0}, counts
            graphs.append(dict(path=str(path.relative_to(RUN)), counts=counts,
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert graphs, 'No complete fused model graph found'
    captured = 'Capturing CUDA graphs (decode, FULL)' in log and '35/35' in log
    if mode == "FULL_DECODE_ONLY":
        assert captured, 'Complete FULL_DECODE_ONLY graph capture not observed'
    else:
        assert not captured, 'NONE attribution pass unexpectedly captured FULL graphs' 
    expected_log = 'decode_warps=4 decode_stages=1 segments=16 decode_layout=1 decode_rows=8 prefill_rows=64 prefill_warps=4 prefill_stages=1 prefill_tile=32'
    assert log.count(expected_log) == 42
    assert 'KVReservationAsyncScheduler' not in log
    assert 'KV growth admission enabled=True' not in log
    kernels = []
    paths = list((RUN/'triton_cache').glob('*/_attention.json'))
    assert paths, 'No isolated service-cache GQA metadata'
    for path in paths:
        metadata = json.loads(path.read_text())
        assert metadata.get('num_warps') == 4, 'Unexpected GQA launch parameters'
        assert metadata['target']['backend'] == 'corex'
        irpath = path.with_suffix('.ttir')
        assert irpath.is_file(), 'Missing actual service TTIR'
        ir = irpath.read_text()
        decode = bool(re.search(r'tt\.load[^\n]*tensor<8x128x!tt\.ptr<(?:bf16|f16)>>', ir))
        prefill = bool(re.search(r'tt\.load[^\n]*tensor<64x128x!tt\.ptr<(?:bf16|f16)>>', ir))
        assert decode != prefill, 'Ambiguous or unsupported GQA query layout'
        expected_stages = 1 if decode else expected_prefill_stages
        assert metadata.get('num_stages') == expected_stages, 'Unexpected decode/prefill pipeline stages'
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
                            num_warps=4, num_stages=expected_stages, shared=metadata['shared'],
                            logical_query_rows=8 if decode else 64, tile=32,
                            page_table_load_elements=2 if decode else 32))
    counts = Counter(k['kind'] for k in kernels)
    assert counts['decode'] and counts['prefill'], 'Both actual decode and prefill compile evidence required'
    (output or folder/'route_evidence.json').write_text(json.dumps(
        dict(status='passed', mode=mode, graph_capture_35=captured, gqa_launch=dict(decode_warps=4,decode_stages=1,segments=16,decode_layout=1,decode_rows=8,
             prefill_rows=64,prefill_warps=4,prefill_stages=expected_prefill_stages,prefill_tile=32), kernels=kernels,
             kernel_counts=dict(counts), fused_runtime_layers=42, gqa_layers=42,
             old_cache_updates_in_fused_graph=0, graphs=graphs), indent=2)+'\n')
