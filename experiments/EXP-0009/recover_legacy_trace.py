"""Repair only the verified unescaped LS_META name field in derivative files."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys


def repair(source, destination):
    raw = source.read_bytes()
    text = gzip.decompress(raw).decode()
    changes = []
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if '"name": "LS_META:{' not in line:
            continue
        match = re.fullmatch(r'(.*"name": )"LS_META:(\{.*\})"(, "pid": .*\n?)', line)
        assert match, (i + 1, "Unexpected exporter format; refuse broad repair")
        metadata = json.loads(match[2])
        assert metadata["kind"] in ("batch", "padding") and isinstance(metadata["step"], int)
        lines[i] = match[1] + json.dumps("LS_META:" + match[2]) + match[3]
        changes.append({"line": i + 1, "step": metadata["step"], "kind": metadata["kind"]})
    repaired = "".join(lines)
    data = json.loads(repaired)
    assert changes and "traceEvents" in data
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(gzip.compress(repaired.encode(), mtime=0))
    assert hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(raw).digest()
    return {"source": str(source), "source_sha256": hashlib.sha256(raw).hexdigest(),
            "derived": str(destination), "derived_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "changes": changes, "event_count": len(data["traceEvents"]),
            "changed_fields": "Only JSON escaping of LS_META annotation names; no timestamps/durations changed"}


if __name__ == "__main__":
    source, dest = map(Path, sys.argv[1:3])
    dest.mkdir(parents=True, exist_ok=True)
    receipts = []
    for trace in sorted(source.glob("FULL_DECODE_ONLY/traces/*/*.pt.trace.json.gz")):
        receipts.append(repair(trace, dest / trace.relative_to(source)))
    assert len(receipts) == 3
    (dest / "recovery_manifest.json").write_text(json.dumps(receipts, indent=2) + "\n")
    import shutil
    (dest / "probe").mkdir(exist_ok=True)
    shutil.copy2(source / "probe/probe_result.json", dest / "probe/probe_result.json")
    from analyze_profile import main
    main(dest)
