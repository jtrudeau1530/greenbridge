"""Group every carved file by signature, using only the per-archive index.json (no file reads).

Classes:
  bink2        Bink 2 video (KB2x)
  rbf0         Rockstar binary metadata ('RBF0', carries its own field names)
  respage      resource page ('42 40 00 00': tagged-pointer structures for textures/meshes/etc.)
  gpu-zerohdr  starts with 16+ zero bytes (GPU data: textures, vertex/index buffers)
  other:<hex>  anything else, keyed by its first 4 bytes

Usage:
    python -I classify_carve.py <work>\\carve
Writes types.json (per archive and overall counts/bytes per class, plus size stats) and prints a summary.
"""
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict


def classify(r):
    if r.get("type") == "bink2":
        return "bink2"
    h = r["head"]
    if h.startswith("52424630"):
        return "rbf0"
    if h.startswith("42400000"):
        return "respage"
    if h.startswith("00000000000000000000000000000000"):
        return "gpu-zerohdr"
    return "other:" + h[:8]


def main():
    root = sys.argv[1]
    per = {}
    total = defaultdict(lambda: [0, 0])
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name, "index.json")
        if not os.path.isfile(p):
            continue
        files = json.load(open(p, encoding="utf-8"))["files"]
        c = defaultdict(lambda: [0, 0])
        for r in files:
            k = classify(r)
            c[k][0] += 1
            c[k][1] += r["size"]
            total[k][0] += 1
            total[k][1] += r["size"]
        per[name] = {k: {"files": v[0], "bytes": v[1]} for k, v in sorted(c.items(), key=lambda kv: -kv[1][1])}
    top = sorted(total.items(), key=lambda kv: -kv[1][1])
    out = {"overall": {k: {"files": v[0], "bytes": v[1]} for k, v in top}, "per_archive": per}
    fd, temp = tempfile.mkstemp(dir=root, prefix=".types-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, os.path.join(root, "types.json"))
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise
    print(f"{'class':24} {'files':>10} {'GiB':>8}")
    for k, v in top[:25]:
        print(f"{k:24} {v[0]:>10} {v[1] / 2**30:8.2f}")
    print(f"{len(top)} classes in total")


if __name__ == "__main__":
    main()
