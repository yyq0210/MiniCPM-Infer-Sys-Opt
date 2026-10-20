"""Check the selected backend, actual forward, and unsupported-feature fallback."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm_fl.dispatch import CachedOp, SelectionPolicy, policy_context
from vllm_fl.dispatch.backends.flaggems.impl.iluvatar_gqa import (
    IluvatarGQABackend, IluvatarGQAImpl,
)
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend, TritonAttentionImpl
from validate_gqa import make_case, fp32_check, baseline, paged_gqa


def main():
    result = {"status": "running", "checks": []}
    path = Path(sys.argv[1])
    try:
        op = CachedOp("attention_backend")
        with policy_context(SelectionPolicy.from_dict(
                per_op_order={"attention_backend": ["flagos"]}, strict=True)):
            route = op()
        assert route.endswith(".iluvatar_gqa.IluvatarGQABackend"), route
        result["selected_backend"] = route
        assert IluvatarGQABackend.get_builder_cls() is TritonAttentionBackend.get_builder_cls()
        assert IluvatarGQABackend.get_kv_cache_shape(8, 16, 2, 128) == (8, 2, 16, 2, 128)
        impl = IluvatarGQAImpl(16, 128, 128 ** -0.5, 2, None, None, "auto")
        for qlens, lengths in [([1]*8, [4096]*8), ([1]*30+[127], [16384]*30+[1024])]:
            x, _, _ = make_case(qlens, lengths)
            kv = torch.stack((x["k"], x["v"]), 1)
            meta = SimpleNamespace(
                num_actual_tokens=sum(qlens), query_start_loc=x["cu"],
                seq_lens=x["lengths"], block_table=x["block_table"],
                max_query_len=max(qlens), max_seq_len=max(lengths),
                causal=True, use_cascade=False, mm_prefix_range_tensor=None,
                softmax_segm_output=x["partial_out"],
                softmax_segm_max=x["partial_max"], softmax_segm_expsum=x["partial_sum"])
            layer = SimpleNamespace(layer_name="integration_check")
            assert impl._supports_gqa(x["q"], kv, meta, x["out"], None, None)
            result_tensor = impl.forward(layer, x["q"], None, None, kv, meta, x["out"])
            assert result_tensor is x["out"]
            error = fp32_check(x, qlens, lengths)
            result["checks"].append({"qlens": qlens, "max_abs_error": error})
            for attribute, value in [("causal", False), ("use_cascade", True)]:
                previous = getattr(meta, attribute)
                setattr(meta, attribute, value)
                sentinel = object()
                with patch.object(TritonAttentionImpl, "forward", return_value=sentinel) as fallback:
                    assert impl.forward(layer, x["q"], None, None, kv, meta, x["out"]) is sentinel
                    assert fallback.call_count == 1
                setattr(meta, attribute, previous)
            result["checks"].append({"fallbacks": ["noncausal", "cascade"], "passed": True})
        # An independent exact invariant, requiring no FP32 reference kernel:
        # zero keys and all-one values yield all-one attention at every token.
        result["constant_value_invariant"] = []
        for qlens, lengths in [([7], [71]), ([37], [201]), ([2048], [2048]),
                                ([2048], [16384]), ([1]*30+[127], [16384]*31)]:
            x, _, _ = make_case(qlens, lengths)
            x["k"].zero_()
            x["v"].fill_(1)
            baseline(x, lengths)
            old_error = (x["out"].float()-1).abs().max().item()
            paged_gqa(**x)
            error = (x["out"].float()-1).abs().max().item()
            assert error <= 0.005, error
            result["constant_value_invariant"].append(
                dict(qlens=qlens, lengths=lengths,
                     baseline_max_abs_error=old_error, candidate_max_abs_error=error))
        result["status"] = "passed"
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        path.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
