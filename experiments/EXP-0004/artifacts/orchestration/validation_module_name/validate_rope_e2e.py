"""Numerical, aliasing, dispatch, compile and graph checks on the target GPU."""
import json
from pathlib import Path
import statistics
import sys
from unittest.mock import patch

import torch
import flag_gems
from flag_gems.fused.rotary_embedding import apply_rotary_pos_emb as generic
from flag_gems.runtime.backend._iluvatar.fused.rotary_embedding_parallel import rope_parallel
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm_fl.ops.rotary_embedding import RotaryEmbeddingFL
from vllm_fl.ops.rotary_embedding_dispatch import rotary_embedding_dispatch
from vllm_fl.dispatch import SelectionPolicy, policy_context, get_default_manager, reset_default_manager

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'artifacts/validation'
OUT.mkdir(parents=True, exist_ok=True)
results = {'checks': [], 'compiled': [], 'timings': [], 'status': 'running'}


def save():
    (OUT / 'results.json').write_text(json.dumps(results, indent=2))


def make_inputs(m, dtype, dim=128):
    x = torch.randn(m, 20, dim, dtype=dtype, device='cuda')
    angles = torch.randn(131072, dim // 2, device='cuda')
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(dtype)
    positions = torch.randint(0, 131072, (m * 2,), device='cuda')[::2]
    if m:
        positions[0] = 131071
    return x, cache, positions


def reference(x, c, s, p, interleaved):
    a, b = (x[..., ::2], x[..., 1::2]) if interleaved else x.chunk(2, -1)
    a, b = a.float(), b.float()
    c, s = c[p].float()[:, None, :], s[p].float()[:, None, :]
    lo, hi = a * c - b * s, b * c + a * s
    if interleaved:
        return torch.stack((lo, hi), -1).flatten(-2).to(x.dtype)
    return torch.cat((lo, hi), -1).to(x.dtype)


def graph_us(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(50):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(7):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / 250)
    return {'median_us': statistics.median(times), 'trials_us': times}


def main():
    torch.manual_seed(20261019)
    results['imports'] = {'flag_gems': flag_gems.__file__,
                          'registered_rope': flag_gems.apply_rotary_pos_emb.__module__}
    assert results['imports']['registered_rope'].endswith('._iluvatar.fused.rotary_embedding')
    for dtype in (torch.bfloat16, torch.float16):
        for m in (0, 1, 7, 32, 64, 257, 2048):
            x, cache, p = make_inputs(m, dtype)
            c, s = cache.chunk(2, -1)
            for interleaved in (False, True):
                expected = [reference(v, c, s, p, interleaved) for v in (x[:, :16], x[:, 16:18])]
                for inplace in (False, True):
                    original = x.clone()
                    out = flag_gems.apply_rotary_pos_emb(x[:, :16], x[:, 16:18], c, s, p, interleaved, inplace)
                    tol = 2e-2 if dtype == torch.bfloat16 else 2e-3
                    for a, b in zip(out, expected):
                        torch.testing.assert_close(a, b, atol=tol, rtol=tol)
                    if inplace:
                        torch.testing.assert_close(x[:, 18:], original[:, 18:], rtol=0, atol=0)
                    else:
                        assert torch.equal(x, original), 'Functional RoPE mutated QKV'
                        if m:
                            assert out[0].data_ptr() != x.data_ptr()
                    results['checks'].append({'dtype': str(dtype), 'm': m, 'interleaved': interleaved,
                                              'inplace': inplace, 'passed': True})
                    x.copy_(original)
    # The production registration must fall back for non-specialized dimensions.
    for dtype, dim in ((torch.float32, 128), (torch.bfloat16, 64)):
        x, cache, p = make_inputs(7, dtype, dim)
        c, s = cache.chunk(2, -1)
        got = flag_gems.apply_rotary_pos_emb(x[:, :16], x[:, 16:18], c, s, p)
        expected = generic(x[:, :16], x[:, 16:18], c, s, p)
        for a, b in zip(got, expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    results['fallbacks_passed'] = True
    x, cache, p = make_inputs(64, torch.bfloat16)
    c, s = cache.chunk(2, -1)
    int32_out = flag_gems.apply_rotary_pos_emb(x[:, :16], x[:, 16:18], c, s, p.int())
    int64_out = flag_gems.apply_rotary_pos_emb(x[:, :16], x[:, 16:18], c, s, p)
    for a, b in zip(int32_out, int64_out):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    results['int32_positions_passed'] = True
    x, cache, p = make_inputs(7, torch.bfloat16)
    c, s = cache.chunk(2, -1)
    for backend in ('flagos', 'reference', 'flagos'):
        policy = SelectionPolicy.from_dict(per_op_order={'rotary_embedding': [backend]}, strict=True)
        with policy_context(policy):
            out = rotary_embedding_dispatch(x[:, :16], x[:, 16:18], c, s, p, False)
            selected = get_default_manager()._called_ops['rotary_embedding']
            assert ('default.' in selected) if backend == 'flagos' else ('reference.' in selected)
            for a, b in zip(out, (reference(x[:, :16], c, s, p, False), reference(x[:, 16:18], c, s, p, False))):
                torch.testing.assert_close(a, b, rtol=2e-2, atol=2e-2)
    for strict in (True, False):
        reset_default_manager()
        policy = SelectionPolicy.from_dict(per_op_order={'rotary_embedding': ['flagos', 'reference']}, strict=strict)
        with policy_context(policy), patch('flag_gems.apply_rotary_pos_emb', side_effect=RuntimeError('injected test failure')):
            try:
                rotary_embedding_dispatch(x[:, :16], x[:, 16:18], c, s, p, False)
            except RuntimeError as exc:
                assert strict and 'injected test failure' in str(exc)
            else:
                assert not strict
                assert 'reference.' in get_default_manager()._called_ops['rotary_embedding']
    reset_default_manager()
    results['policy_refresh_and_strict_fallback_passed'] = True
    save()

    with set_current_vllm_config(VllmConfig()):
        # Unsupported layer signatures must retain exact native behavior.
        for dim, dtype, key_present in ((64, torch.bfloat16, True),
                                         (128, torch.float32, True),
                                         (128, torch.bfloat16, False)):
            fallback = RotaryEmbeddingFL(128, dim, 256, 5000000.0, True, dtype).to('cuda')
            x = torch.randn(7, 20, 128, dtype=dtype, device='cuda')
            q = x[:, :16].reshape(7, -1)
            k = x[:, 16:18].reshape(7, -1) if key_present else None
            p = torch.arange(7, device='cuda')
            got = fallback.forward_native(p, q, k)
            ref = RotaryEmbedding.forward_native(fallback, p, q, k)
            torch.testing.assert_close(got[0], ref[0], rtol=0, atol=0)
            if key_present:
                torch.testing.assert_close(got[1], ref[1], rtol=0, atol=0)
            else:
                assert got[1] is None
        results['layer_fallbacks_passed'] = True
        layer = RotaryEmbeddingFL(128, 128, 131072, 5000000.0, True, torch.bfloat16).to('cuda')
        native = lambda p, q, k: RotaryEmbedding.forward_static(p, q, k, 128, 128, layer.cos_sin_cache, True)
        compiled_native = torch.compile(native, fullgraph=True, dynamic=True)
        compiled_new = torch.compile(layer.forward_native, fullgraph=True, dynamic=True)
        for m in (1, 64, 257, 2048):
            x, _, p = make_inputs(m, torch.bfloat16)
            q, k = x[:, :16].reshape(m, -1), x[:, 16:18].reshape(m, -1)
            original = x.clone()
            ref = compiled_native(p, q, k)
            actual = compiled_new(p, q, k)
            errors = []
            for a, b in zip(actual, ref):
                torch.testing.assert_close(a, b, rtol=2e-2, atol=2e-2)
                errors.append({'max_abs': (a.float()-b.float()).abs().max().item(),
                               'exact_fraction': (a == b).float().mean().item()})
            assert torch.equal(x, original)
            # The default forward method itself must reach the optimized path.
            default_out = layer(p, q, k)
            for a, b in zip(actual, default_out):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                captured = compiled_new(p, q, k)
            graph.replay()
            torch.cuda.synchronize()
            for a, b in zip(captured, actual):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            results['compiled'].append({'m': m, 'errors_vs_compiled_native': errors,
                                         'input_unchanged': True, 'graph_replay_passed': True})
            old_t = graph_us(lambda: compiled_native(p, q, k))
            new_t = graph_us(lambda: compiled_new(p, q, k))
            old_after = graph_us(lambda: compiled_native(p, q, k))
            results['timings'].append({'m': m, 'compiled_native': old_t, 'compiled_dispatch': new_t,
                                      'native_after': old_after,
                                      'speedup': old_t['median_us']/new_t['median_us']})
            print(json.dumps(results['timings'][-1]), flush=True)
            save()
    results['status'] = 'passed'
    save()
    print('VALIDATION_PASSED', flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        results['status'] = 'failed'
        results['error'] = repr(exc)
        save()
        raise
