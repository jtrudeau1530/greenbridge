"""Build a joaat (Jenkins one-at-a-time, lowercased) hash -> string dictionary from every string we can find.

Sources: printable runs in carved blobs (selected archives fully; PSO/RBF blobs everywhere), shader names, and
any extra text/binary files given (e.g. the exe once it arrives: run with --extra path\\to\\exe).
When multiple names share a hash, the dictionary keeps the lexicographically first spelling.

Usage:
    python -I hashdict.py <carve dir> --out hashes.json [--archives a,b] [--extra f1 f2 ...] [--min 3]
    python -I hashdict.py --lookup <hashes.json> 0x28cca8df ...
"""
import argparse
import json
import os
import re
import sys

STR = re.compile(rb"[A-Za-z_][A-Za-z0-9_./:\\-]{2,200}")


def joaat(s: str) -> int:
    h = 0
    for c in s.lower().encode("latin-1", "replace"):
        h = (h + c) & 0xFFFFFFFF
        h = (h + (h << 10)) & 0xFFFFFFFF
        h ^= h >> 6
    h = (h + (h << 3)) & 0xFFFFFFFF
    h ^= h >> 11
    return (h + (h << 15)) & 0xFFFFFFFF


def harvest(data: bytes, out: set, minlen: int):
    for m in STR.finditer(data):
        s = m.group(0).decode("latin-1")
        if len(s) >= minlen:
            out.add(s)
            # also the pieces of paths and dotted names, which are hashed on their own
            for part in re.split(r"[./:\\]", s):
                if len(part) >= minlen:
                    out.add(part)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carve", nargs="?")
    ap.add_argument("--out", default="hashes.json")
    ap.add_argument("--archives", default="common,data,script,pack_story000_story000_script,textures")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--min", type=int, default=3)
    ap.add_argument("--lookup", nargs="+", metavar="VALUE", help="dictionary path followed by one or more hashes")
    a = ap.parse_args()
    if a.lookup:
        if len(a.lookup) < 2:
            ap.error("--lookup requires a dictionary and at least one hash")
        values = []
        for h in a.lookup[1:]:
            try:
                values.append((h, int(h, 16)))
            except ValueError:
                ap.error(f"--lookup: {h!r} is not a hexadecimal hash")
        try:
            with open(a.lookup[0], encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError) as e:
            print(f"ERROR: cannot read the dictionary {a.lookup[0]}: {getattr(e, 'strerror', None) or e}",
                  file=sys.stderr)
            return 2
        for h, v in values:
            print(h, d.get(f"{v:08x}", "?"))
        return 0
    if a.carve is None:
        ap.error("a carve directory is required to build a dictionary")
    if not os.path.isdir(a.carve):
        what = "is not a folder" if os.path.exists(a.carve) else "does not exist"
        print(f"ERROR: the carve folder {a.carve} {what}", file=sys.stderr)
        return 2
    strings = set()
    full = set(a.archives.split(","))
    missing = []                    # carved blobs an index.json lists that can't be read: skipped, reported
    for arch in sorted(os.listdir(a.carve)):
        p = os.path.join(a.carve, arch, "index.json")
        if not os.path.isfile(p):
            continue
        with open(p, encoding="utf-8") as f:
            rows = json.load(f)["files"]
        for r in rows:
            if r.get("type") == "bink2":
                continue
            meta = r["head"].startswith("5053494e") or r["head"].startswith("52424630")
            if arch in full or meta:
                blob = os.path.join(a.carve, arch, f"{r['off']:012x}.bin")
                try:
                    with open(blob, "rb") as f:
                        data = f.read()
                except OSError:
                    missing.append(blob)
                    continue
                harvest(data, strings, a.min)
        print(f"{arch}: {len(strings)} strings so far", flush=True)
    if missing:
        print(f"WARNING: {len(missing)} carved blobs listed in an index.json could not be read and were skipped "
              f"(first: {missing[0]}). Re-run the carve to get their strings.", file=sys.stderr)
    for x in a.extra:
        if os.path.isdir(x):
            for root, _, files in os.walk(x):
                for fn in files:
                    strings.add(os.path.splitext(fn)[0])
        elif os.path.isfile(x):
            harvest(open(x, "rb").read(), strings, a.min)
    d = {}
    for s in sorted(strings):
        d.setdefault(f"{joaat(s):08x}", s)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(d, f)
    print(f"{len(strings)} strings -> {len(d)} hashes -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
