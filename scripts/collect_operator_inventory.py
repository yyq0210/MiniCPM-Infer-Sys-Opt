"""Read-only code/experiment inventory; no torch import or GPU work."""
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path("/workspace/LeanScale")
REPO = ROOT / "MiniCPM-Infer-Sys-Opt"
OUTPUT = REPO / "docs/evidence/operator_hotspots_20261020"
BASES = {
    "vllm": Path("/usr/local/lib/python3.12/site-packages/vllm"),
    "plugin": ROOT / "worktrees/EXP-0006/vllm-plugin-FL",
    "gems": ROOT / "worktrees/EXP-0006/FlagGems",
    "scheduler_plugin": ROOT / "worktrees/EXP-0007/vllm-plugin-FL",
}
FILES = {
    "vllm": [
        "model_executor/models/llama.py",
        "model_executor/layers/linear.py",
        "model_executor/layers/utils.py",
        "model_executor/layers/layernorm.py",
        "ir/ops/layernorm.py",
        "model_executor/layers/activation.py",
        "model_executor/layers/rotary_embedding/base.py",
        "model_executor/layers/vocab_parallel_embedding.py",
        "model_executor/layers/logits_processor.py",
        "v1/attention/backends/triton_attn.py",
        "v1/attention/ops/triton_reshape_and_cache_flash.py",
        "v1/attention/ops/triton_unified_attention.py",
        "v1/sample/sampler.py",
        "v1/sample/ops/topk_topp_sampler.py",
        "v1/core/sched/scheduler.py",
        "v1/core/kv_cache_manager.py",
        "v1/core/single_type_kv_cache_manager.py",
        "v1/engine/core.py",
        "benchmarks/serve.py",
        "benchmarks/lib/endpoint_request_func.py",
    ],
    "plugin": [
        "vllm_fl/platform.py",
        "vllm_fl/worker/model_runner.py",
        "vllm_fl/dispatch/backends/flaggems/flaggems.py",
        "vllm_fl/dispatch/backends/flaggems/impl/iluvatar_gqa.py",
        "vllm_fl/ops/rotary_embedding.py",
        "vllm_fl/ops/rope_kv_dispatch.py",
        "vllm_fl/compilation/rope_kv_fusion.py",
    ],
    "gems": [
        "src/flag_gems/runtime/backend/_iluvatar/fused/paged_gqa.py",
        "src/flag_gems/runtime/backend/_iluvatar/fused/rope_kv_cache.py",
        "src/flag_gems/runtime/backend/_iluvatar/fused/gate_up_swiglu.py",
        "src/flag_gems/runtime/backend/_iluvatar/fused/rms_qkv.py",
    ],
    "scheduler_plugin": [
        "vllm_fl/worker/kv_reservation_config.py",
        "vllm_fl/worker/kv_reservation_scheduler.py",
    ],
}


class Symbols(ast.NodeVisitor):
    def __init__(self):
        self.prefix = []
        self.rows = {}

    def record(self, node):
        self.rows[".".join(self.prefix + [node.name])] = node.lineno
        self.prefix.append(node.name)
        self.generic_visit(node)
        self.prefix.pop()

    visit_ClassDef = record
    visit_FunctionDef = record
    visit_AsyncFunctionDef = record


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sources = {}
    for label, files in FILES.items():
        for relative in files:
            path = BASES[label] / relative
            data = path.read_bytes()
            symbols = Symbols()
            symbols.visit(ast.parse(data))
            sources[label + "/" + relative] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "lines": len(data.splitlines()),
                "symbols": symbols.rows,
            }
    commits = {
        key: subprocess.check_output(
            ["git", "-C", str(BASES[key]), "rev-parse", "HEAD"], text=True).strip()
        for key in ("plugin", "gems", "scheduler_plugin")
    }
    model = json.loads(Path("/workspace/MiniCPM5-2B/config.json").read_text())
    config = {k: model.get(k) for k in (
        "architectures", "model_type", "num_hidden_layers", "hidden_size",
        "intermediate_size", "num_attention_heads", "num_key_value_heads",
        "vocab_size", "rms_norm_eps", "max_position_embeddings",
        "tie_word_embeddings", "rope_theta")}
    runs = {}
    for case in ("4k", "16k"):
        receipt = json.loads((ROOT / f"artifacts/EXP-0006-latest-{case}.json").read_text())
        run = Path(receipt["run_directory"])
        status = json.loads((run / "status.json").read_text())
        route = run / "candidate_benchmark/route_evidence.json"
        runs[case] = {
            "run": str(run.relative_to(ROOT)),
            "stage": status["stage"], "commits": status["commits"],
            "route_evidence": json.loads(route.read_text()) if route.exists() else None,
            "performance": status.get("performance"),
            "incremental_vs_exp5": status.get("incremental_vs_exp5"),
        }
    result = {
        "captured_at": datetime.now().astimezone().isoformat(),
        "purpose": "Source/shape/route inventory, not a GPU profiler report",
        "commits": commits, "model": config,
        "runtime_snapshot": runs, "sources": sources,
    }
    (OUTPUT / "source_index.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"files": len(sources), "commits": commits,
                      "captured_at": result["captured_at"]}), flush=True)


if __name__ == "__main__":
    main()
