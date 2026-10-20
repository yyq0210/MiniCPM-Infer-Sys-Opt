"""Confirm RoPE+KV finalists with reordered trials, under the shared GPU lock."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import random
import shutil
import traceback

import torch
from validate_fusion import make_rope, old_rope, timed
from flag_gems.runtime.backend._iluvatar.fused.rope_kv_cache import rope_kv_cache
from flag_gems.runtime.backend._iluvatar.fused.rotary_embedding_parallel import rope_parallel
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash

root = Path("/workspace/LeanScale")
run = root/"artifacts/EXP-0006"/datetime.datetime.now().strftime("%Y%m%d_%H%M%S_confirmation")
run.mkdir(parents=True)
result = dict(status="running", kind="isolated_confirmation_not_official",
              rounds=[], source_manifest={})
files = [Path(__file__), Path(__file__).with_name("validate_fusion.py"),
         root/"worktrees/EXP-0006/FlagGems/src/flag_gems/runtime/backend/_iluvatar/fused/rope_kv_cache.py"]
for path in files:
    result["source_manifest"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    shutil.copy2(path, run/path.name)


def save():
    result["updated_at"] = datetime.datetime.now().astimezone().isoformat()
    target = run/"confirmation.json"
    target.with_suffix(".tmp").write_text(json.dumps(result, indent=2)+"\n")
    target.with_suffix(".tmp").replace(target)


lock = open("/workspace/experiments/codex_20261019/runner.lock", "a+")
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.manual_seed(20261021)
    rng = random.Random(20261021)
    configs = [dict(head_tile=h, token_tile=t, num_warps=4)
               for h, t in [(4, 1), (8, 1), (16, 1), (4, 4)]]
    for n in (1, 8, 30, 32, 64, 128, 129, 256, 512, 1024, 2048, 4096):
        x, qkv, cache = make_rope(n)
        x["slots"] = torch.randperm(x["k_cache"].shape[0]*16, device="cuda")[:n]
        oq, ok = old_rope(x)
        expected = cache.clone()
        for config in configs:
            cache.fill_(0.125)
            q, k = rope_kv_cache(**x, **config)
            torch.testing.assert_close(q, oq, atol=0, rtol=0)
            torch.testing.assert_close(k, ok, atol=0, rtol=0)
            torch.testing.assert_close(cache, expected, atol=0, rtol=0)
        scale = torch.ones((), device="cuda")

        def baseline():
            q, k = rope_parallel(x["q"], x["k"], x["cos"], x["sin"],
                                 x["positions"], inplace=False)
            triton_reshape_and_cache_flash(k, x["v"], x["k_cache"], x["v_cache"],
                                           x["slots"], "auto", scale, scale)
            return q, k

        for repeat in range(3):
            order = list(configs)
            rng.shuffle(order)
            row = dict(n=n, repeat=repeat, baseline=timed(baseline), candidates=[])
            for config in order:
                row["candidates"].append(dict(
                    config=config, **timed(lambda: rope_kv_cache(**x, **config))))
            row["baseline_after"] = timed(baseline)
            result["rounds"].append(row)
            save()
        print(json.dumps(dict(n=n, repeats=3, exact_checks=4)), flush=True)
        del x, qkv, cache, expected, oq, ok, q, k
        torch.cuda.empty_cache()
    result["status"] = "passed"
except BaseException as exc:
    result.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
    raise
finally:
    save()
    lock.close()
    print(json.dumps(dict(run_directory=str(run), status=result["status"])), flush=True)
