"""Check the selected default launch policy at boundaries and graph padding."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import traceback

import torch
from validate_fusion import make_rope, old_rope
from flag_gems.runtime.backend._iluvatar.fused.rope_kv_cache import (
    rope_kv_cache, select_rope_kv_config,
)

root = Path("/workspace/LeanScale")
run = root/"artifacts/EXP-0006"/datetime.datetime.now().strftime("%Y%m%d_%H%M%S_selected")
run.mkdir(parents=True)
source = root/"worktrees/EXP-0006/FlagGems/src/flag_gems/runtime/backend/_iluvatar/fused/rope_kv_cache.py"
result = dict(status="running", source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              checks=[])
(run/source.name).write_bytes(source.read_bytes())
(run/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
lock = open("/workspace/experiments/codex_20261019/runner.lock", "a+")
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.manual_seed(20261022)
    for dtype in (torch.bfloat16, torch.float16):
        for n in (0, 1, 7, 31, 32, 33, 63, 64, 127, 128, 129, 257, 2048, 4096):
            x, qkv, cache = make_rope(n, dtype=dtype, hnd=True, short_slots=True)
            original = qkv.clone()
            if n:
                oq, ok = old_rope(x)
                expected = cache.clone()
            else:
                oq, ok, expected = x["q"], x["k"], cache.clone()
            cache.fill_(0.125)
            q, k = rope_kv_cache(**x)
            for a, b in ((q, oq), (k, ok), (cache, expected), (qkv, original)):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            if n:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    gq, gk = rope_kv_cache(**x)
                qkv.mul_(0.7)
                x["positions"].copy_(x["positions"].roll(1))
                x["slots"].copy_(x["slots"].roll(1))
                cache.fill_(0.125)
                oq, ok = old_rope(x)
                expected = cache.clone()
                cache.fill_(0.125)
                graph.replay()
                torch.cuda.synchronize()
                for a, b in ((gq, oq), (gk, ok), (cache, expected)):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
            result["checks"].append(dict(n=n, dtype=str(dtype),
                                         selected_config=select_rope_kv_config(n), passed=True))
    result["status"] = "passed"
except BaseException as exc:
    result.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
    raise
finally:
    (run/"selected.json").write_text(json.dumps(result, indent=2)+"\n")
    lock.close()
    print(json.dumps(dict(run_directory=str(run), status=result["status"],
                         checks=len(result["checks"]))), flush=True)
