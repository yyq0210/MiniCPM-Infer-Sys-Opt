"""CPU-only structural and fallback tests; never initialize CUDA tensors."""
import json
import operator
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import fx
from vllm_fl.ops import rotary_embedding_dispatch  # noqa: F401
# The vendor platform probes CUDA capability at import time. Stub only that
# probe in this CPU-only test process; never change the actual service.
with patch("torch.cuda.get_device_capability", return_value=(7, 0)):
    from vllm_fl.ops import rope_kv_dispatch as dispatch
    from vllm_fl.compilation.rope_kv_fusion import fuse_rope_kv
    from vllm_fl.dispatch.backends.flaggems.impl.iluvatar_gqa import IluvatarGQAImpl  # noqa: F401


def make_graph(double_writer=False, nonlinear_v=False, tuple_user=False, aten=False,
               duplicate_getitem=False, packet=False, model_output_only=False):
    g = fx.Graph()
    def op(name):
        value = getattr(torch.ops.vllm, name)
        return value if packet else value.default
    args = [g.placeholder(n) for n in ["q", "k", "v", "cos", "sin", "positions", "out"]]
    q, k, v, cos, sin, positions, out = args
    rope = g.call_function(op("fl_rotary_embedding_dispatch"),
                           (q, k, cos, sin, positions, False))
    qo = g.call_function(operator.getitem, (rope, 0))
    ko = g.call_function(operator.getitem, (rope, 1))
    if aten:
        vv = g.call_function(torch.ops.aten.view.default, (v, [-1, 2, 128]))
        kk = g.call_function(torch.ops.aten.view.default, (ko, [-1, 2, 128]))
    else:
        vv = g.call_method("view", (v, -1, 2, 128))
        kk = g.call_method("reshape", (ko, -1, 2, 128))
    if nonlinear_v:
        vv = g.call_function(torch.sin, (vv,))
    dep = g.call_function(op("unified_kv_cache_update"), (kk, vv, "layer0"))
    g.call_function(op("unified_attention_with_output"),
                    (qo, kk, vv, out, "layer0"), {"kv_cache_dummy_dep": dep})
    extras = []
    if double_writer:
        second_k = (g.call_function(operator.getitem, (rope, 1))
                    if duplicate_getitem else kk)
        extras.append(g.call_function(torch.ops.vllm.unified_kv_cache_update.default,
                                      (second_k, vv, "layer1")))
    if tuple_user:
        extras.append(rope)
    g.output(out if model_output_only else (qo, ko, out, dep, *extras))
    return fx.GraphModule({}, g)


def main():
    results = {"status": "running", "checks": []}
    path = Path(sys.argv[1])
    try:
        for options, expected in [
            ({}, 1), ({"aten": True}, 1),
            ({"packet": True, "model_output_only": True}, 1),
            ({"model_output_only": True}, 1), ({"double_writer": True}, 0),
            ({"double_writer": True, "duplicate_getitem": True}, 0),
            ({"nonlinear_v": True}, 0), ({"tuple_user": True}, 0),
        ]:
            gm = make_graph(**options)
            count = fuse_rope_kv(gm)
            assert count == expected, (options, count)
            gm.graph.lint()
            assert fuse_rope_kv(gm) == 0, "rewrite must be idempotent"
            if expected:
                nodes = list(gm.graph.nodes)
                assert sum("fl_rope_kv_dispatch" in str(n.target) for n in nodes) == 1
                assert not any("unified_kv_cache_update" in str(n.target) for n in nodes)
                attention = next(n for n in nodes if "unified_attention_with_output" in str(n.target))
                dep = attention.kwargs["kv_cache_dummy_dep"]
                assert dep.target is operator.getitem and dep.args[1] == 2
                assert "fl_rope_kv_dispatch" in str(dep.args[0].target)
            results["checks"].append({"options": options, "matches": count, "passed": True})
        q = torch.randn(2, 16, 128)
        k = torch.randn(2, 2, 128)
        v = torch.randn_like(k)
        cache = torch.empty(1)
        positions = torch.tensor([0, 1])
        manager = SimpleNamespace(_resolve_impl=lambda _: SimpleNamespace(impl_id="reference"))
        calls = []
        impl = SimpleNamespace(do_kv_cache_update=lambda *args: calls.append(args))
        layer = SimpleNamespace(impl=impl)
        for slots in [torch.tensor([0, 1]), None]:
            calls.clear()
            with patch.object(dispatch, "get_default_manager", return_value=manager), \
                 patch.object(dispatch, "get_attention_context", return_value=(None, layer, cache, slots)), \
                 patch.object(dispatch, "_rope", return_value=(q.clone(), k.clone())) as rope:
                a, b, dep = dispatch.rope_kv_dispatch(q, k, v, torch.empty(2, 64),
                                                    torch.empty(2, 64), positions, False, "layer0")
                assert rope.call_count == 1 and len(calls) == (slots is not None)
                assert a.data_ptr() != q.data_ptr() and b.data_ptr() != k.data_ptr()
                assert dep.numel() == 0
                if calls:
                    assert calls[0][1] is b and calls[0][2] is v
            results["checks"].append({"fallback": "slots" if slots is not None else "profile",
                                       "single_cache_write": True, "passed": True})
        results["status"] = "passed"
    except BaseException as exc:
        results.update(status="failed", error=repr(exc))
        raise
    finally:
        path.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
