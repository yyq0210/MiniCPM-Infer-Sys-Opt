"""Run only after acquiring the GPU lock; require real device trace events."""
import gzip
import json
from pathlib import Path
import sys

import torch

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
a = torch.randn((128, 128), device="cuda")
for _ in range(3):
    b = a @ a
torch.cuda.synchronize()
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for _ in range(3):
        with torch.profiler.record_function("probe_matmul"):
            b = a @ a
    torch.cuda.synchronize()
trace = out / "probe.json"
prof.export_chrome_trace(str(trace))
data = json.loads(trace.read_text())
device_events = [e for e in data["traceEvents"]
                 if e.get("ph") == "X" and "kernel" in e.get("cat", "").lower()
                 and e.get("dur", 0) > 0]
result = {"torch": torch.__version__, "device_kernel_events": len(device_events),
          "device_kernel_us": sum(e["dur"] for e in device_events),
          "device_kernel_names": sorted({e["name"] for e in device_events})}
# Graph replay visibility is checked separately: a profiler can observe eager
# kernels while omitting individual kernels inside a replay on a vendor runtime.
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    c = a @ a
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
) as graph_prof:
    graph.replay()
    torch.cuda.synchronize()
graph_path = out / "graph_probe.json"
graph_prof.export_chrome_trace(str(graph_path))
graph_events = json.loads(graph_path.read_text())["traceEvents"]
result["graph_device_kernel_events"] = sum(
    e.get("ph") == "X" and "kernel" in e.get("cat", "").lower() and e.get("dur", 0) > 0
    for e in graph_events)
(out / "probe_result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
assert device_events, "Profiler produced no GPU kernel events; CPU timing is not a substitute"
with gzip.open(out / "probe.json.gz", "wb") as f:
    f.write(trace.read_bytes())
