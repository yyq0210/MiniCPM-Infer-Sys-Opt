"""Collect CPU-only evidence for the installed MiniCPM inference pipeline."""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/EXP-0004/artifacts/pipeline'
VLLM = Path('/usr/local/lib/python3.12/site-packages/vllm')
PLUGIN = Path('/workspace/vllm-plugin-FL')
MODEL = Path('/workspace/MiniCPM5-2B')
BASE = Path('/workspace/experiments/codex_20261019')


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sanitize(s):
    s = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP_REDACTED]', s)
    s = re.sub(r'uuid=[^,)]+', 'uuid=[REDACTED]', s)
    return s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((MODEL / 'config.json').read_text())
    total, dtypes, examples = 0, {}, {}
    for p in sorted(MODEL.glob('*.safetensors')):
        with p.open('rb') as f:
            size = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(size))
        for name, meta in header.items():
            if name == '__metadata__':
                continue
            n = math.prod(meta['shape'])
            total += n
            dtypes[meta['dtype']] = dtypes.get(meta['dtype'], 0) + n
            if 'layers.0.' in name or name in {'lm_head.weight', 'model.embed_tokens.weight', 'model.norm.weight'}:
                examples[name] = meta['shape']
    (OUT / 'model.json').write_text(json.dumps({
        'config': config, 'config_sha256': digest(MODEL/'config.json'),
        'safetensors_header_parameter_count': total, 'parameter_dtypes': dtypes,
        'weight_shapes': examples, 'weights_copied': False,
    }, indent=2))

    sources = {
        'vllm': (VLLM, [
            'model_executor/models/llama.py', 'model_executor/layers/rotary_embedding/base.py',
            'model_executor/layers/rotary_embedding/common.py', 'model_executor/custom_op.py',
            'model_executor/layers/attention/attention.py', 'model_executor/layers/layernorm.py',
            'model_executor/layers/logits_processor.py', 'v1/engine/async_llm.py',
            'v1/engine/core.py', 'v1/engine/input_processor.py', 'v1/engine/output_processor.py',
            'v1/core/sched/scheduler.py', 'v1/core/kv_cache_manager.py',
            'v1/attention/backends/registry.py', 'v1/attention/backends/triton_attn.py',
            'v1/attention/ops/triton_unified_attention.py',
            'v1/attention/ops/triton_reshape_and_cache_flash.py', 'v1/sample/sampler.py',
            'entrypoints/openai/completion/api_router.py', 'entrypoints/openai/completion/serving.py',
        ]),
        'plugin': (PLUGIN, [
            'vllm_fl/platform.py', 'vllm_fl/worker/worker.py', 'vllm_fl/worker/model_runner.py',
            'vllm_fl/ops/rotary_embedding.py', 'vllm_fl/dispatch/backends/flaggems/flaggems.py',
            'vllm_fl/dispatch/backends/flaggems/impl/custom_attention.py',
            'benchmarks/benchmark_throughput_serve.py',
        ]),
    }
    records = []
    for group, (base, paths) in sources.items():
        for rel in paths:
            p = base / rel
            if not p.is_file():
                raise FileNotFoundError(p)
            definitions = [{'line': n, 'text': line.strip()}
                           for n, line in enumerate(p.read_text().splitlines(), 1)
                           if re.match(r'(class |    (async )?def |(async )?def )', line)]
            records.append({'group': group, 'relative_path': rel, 'sha256': digest(p),
                            'definitions': definitions})
    (OUT / 'source_index.json').write_text(json.dumps(records, indent=2))

    state = json.loads((BASE/'status.json').read_text())
    lines = (BASE/'baseline_official/server.log').read_text(errors='replace').splitlines()
    startup = [{'line': n, 'text': sanitize(line)} for n, line in enumerate(lines, 1)
               if any(x in line for x in ['Resolved architecture:', 'Initializing a V1 LLM engine',
                                          'Asynchronous scheduling is enabled',
                                          "Op 'attention_backend'", 'GPU KV cache size:',
                                          'Maximum concurrency for', 'Capturing CUDA graphs (decode, FULL): 100%'])]
    live = [{'line': n, 'text': sanitize(line)} for n, line in enumerate(lines, 1)
            if 'GPU KV cache usage:' in line][-3:]
    (OUT/'runtime.json').write_text(json.dumps({
        'captured_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'baseline_commits': state['commits'], 'benchmark_sha256': state['benchmark_sha256'],
        'startup': startup, 'recent_scheduler_observations': live,
        'attention_backend_class_evidence': 'source selector plus baseline environment and graph; not an in-process object dump',
    }, indent=2))

    graph = Path('/root/.cache/vllm/torch_compile_cache/f7af72be36/rank_0_0/backbone/computation_graph.py')
    code = graph.read_text()
    patterns = ['torch._C._nn.linear(', 'torch.nn.functional.silu(',
                'torch.ops.vllm_ir.rms_norm.default(', 'torch.ops.vllm_ir.fused_add_rms_norm.maybe_inplace(',
                'torch.ops.vllm.unified_kv_cache_update(', 'torch.ops.vllm.unified_attention_with_output(',
                'torch.ops.vllm.fl_rotary_embedding_dispatch(', '.index_select(0, flatten']
    executable = '\n'.join(l for l in code.splitlines() if not l.lstrip().startswith('#'))
    (OUT/'compiled_graph.json').write_text(json.dumps({
        'sha256': digest(graph),
        'call_expression_counts': {x: executable.count(x) for x in patterns},
        'note': 'FX call expressions are not GPU kernel launch counts; Inductor can fuse or lower them.',
    }, indent=2))
    excerpt = '\n'.join(f'{n}: {l}' for n, l in enumerate(code.splitlines(), 1) if 561 <= n <= 760)
    (OUT/'compiled_graph_layer0.txt').write_text(excerpt+'\n')
    hashes = {p.name: digest(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(hashes, indent=2))
    print(json.dumps({'output': str(OUT), 'files': len(hashes), 'parameters': total}))


if __name__ == '__main__':
    main()
