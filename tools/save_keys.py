#!/usr/bin/env python3
"""save_keys: collect the hashed `$0x…` keys of GTA VI saves and resolve them against a local name dictionary.

Save keys (block ids, group keys and field names in JSAV blocks and ASAV settings JSON) are the **case-sensitive**
joaat (Jenkins one-at-a-time, no lower-casing, seed 0) of the engine's names, the same rule as the engine member
hashes in docs/world_names.md. Lower-cased joaat, FNV-1/1a, CRC-32 and MurmurHash3 (32-bit, and both halves of the
64-bit) resolve none of them. See docs/saves.md.

Standard library only. Run with `python -I tools/save_keys.py <command> ...`.

Commands
  collect <input>...                 count the distinct `$0x…` keys in JSAV/ASAV files, `saves.py extract` folders
                                     or loose .json files
  resolve <input>... --dict F... [--compose N] [--out FILE]
                                     harvest identifier-like strings from the dictionary files (any text or binary,
                                     e.g. carved archive streams), hash them, and report how many keys resolve with
                                     each hash family. --compose N also tries one- and two-word Pascal/camel/snake
                                     names built from the N most frequent words and prints the expected number of
                                     chance matches. --out writes hash -> names as JSON (a local work product:
                                     resolved names are game data, so never commit or post it).

Resolved names are game strings. Only counts belong in issues and pull requests.
"""
import argparse
import collections
import json
import os
import re
import struct
import sys
import zlib

KEY = re.compile(rb'"\$0x([0-9a-fA-F]{8})"')
IDENT = re.compile(rb"[A-Za-z_][A-Za-z0-9_]{1,80}")
WORD = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
M32 = 0xFFFFFFFF


# ---------------------------------------------------------------------------------------------- hash families
def joaat_step(h, data):
    for c in data:
        h = (h + c) & M32
        h = (h + (h << 10)) & M32
        h ^= h >> 6
    return h


def joaat_final(h):
    h = (h + (h << 3)) & M32
    h ^= h >> 11
    return (h + (h << 15)) & M32


def joaat(data, seed=0):
    """Case-sensitive joaat: the save-key hash."""
    return joaat_final(joaat_step(seed, data))


def fnv1a(data):
    h = 0x811C9DC5
    for c in data:
        h = ((h ^ c) * 0x01000193) & M32
    return h


def fnv1(data):
    h = 0x811C9DC5
    for c in data:
        h = ((h * 0x01000193) & M32) ^ c
    return h


def murmur3_32(data, seed=0):
    c1, c2 = 0xCC9E2D51, 0x1B873593
    h, n = seed, len(data) // 4
    for i in range(n):
        k = struct.unpack_from("<I", data, i * 4)[0]
        k = (k * c1) & M32
        k = ((k << 15) | (k >> 17)) & M32
        h ^= (k * c2) & M32
        h = ((h << 13) | (h >> 19)) & M32
        h = (h * 5 + 0xE6546B64) & M32
    k = 0
    for i, b in enumerate(data[n * 4:]):
        k |= b << (8 * i)
    if len(data) & 3:
        k = (k * c1) & M32
        k = ((k << 15) | (k >> 17)) & M32
        h ^= (k * c2) & M32
    h ^= len(data)
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & M32
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & M32
    return h ^ (h >> 16)


def murmur3_x64_64(data, seed=0):
    """Low 64 bits of MurmurHash3 x64 128 (the 64-bit name key in docs/world_names.md)."""
    m = (1 << 64) - 1

    def rotl(x, r):
        return ((x << r) | (x >> (64 - r))) & m

    def fmix(k):
        k ^= k >> 33
        k = (k * 0xFF51AFD7ED558CCD) & m
        k ^= k >> 33
        k = (k * 0xC4CEB9FE1A85EC53) & m
        return k ^ (k >> 33)

    c1, c2 = 0x87C37B91114253D5, 0x4CF5AD432745937F
    h1 = h2 = seed
    n = len(data) // 16
    for i in range(n):
        k1, k2 = struct.unpack_from("<QQ", data, i * 16)
        h1 ^= (rotl((k1 * c1) & m, 31) * c2) & m
        h1 = (rotl(h1, 27) + h2) & m
        h1 = (h1 * 5 + 0x52DCE729) & m
        h2 ^= (rotl((k2 * c2) & m, 33) * c1) & m
        h2 = (rotl(h2, 31) + h1) & m
        h2 = (h2 * 5 + 0x38495AB5) & m
    tail = data[n * 16:]
    k1 = k2 = 0
    for i, b in enumerate(tail):
        if i < 8:
            k1 |= b << (8 * i)
        else:
            k2 |= b << (8 * (i - 8))
    if len(tail) > 8:
        h2 ^= (rotl((k2 * c2) & m, 33) * c1) & m
    if tail:
        h1 ^= (rotl((k1 * c1) & m, 31) * c2) & m
    h1 ^= len(data)
    h2 ^= len(data)
    h1 = (h1 + h2) & m
    h2 = (h2 + h1) & m
    h1, h2 = fmix(h1), fmix(h2)
    return (h1 + h2) & m


FAMILIES = {
    "joaat": joaat,
    "joaat_lower": lambda b: joaat(b.lower()),
    "fnv1a": fnv1a,
    "fnv1": fnv1,
    "crc32": zlib.crc32,
    "murmur3_32": murmur3_32,
    "murmur3_64_lo32": lambda b: murmur3_x64_64(b) & M32,
    "murmur3_64_hi32": lambda b: murmur3_x64_64(b) >> 32,
}


# ---------------------------------------------------------------------------------------------- inputs
def read(p):
    with open(p, "rb") as f:
        return f.read()


INPUT_EXTENSIONS = (".jsav", ".asav", ".json", ".bin")


def iter_inputs(paths):
    """Yield the bytes of every save file (.jsav, .asav, .bin or no extension) or .json found under the given paths."""
    for p in paths:
        if os.path.isdir(p):
            for dp, _, fs in os.walk(p):
                for f in sorted(fs):
                    if f.lower().endswith(INPUT_EXTENSIONS) or "." not in f:
                        yield read(os.path.join(dp, f))
        else:
            yield read(p)


def collect_keys(paths):
    keys = collections.Counter()
    for d in iter_inputs(paths):
        if d[:4] == b"JSAV":                      # index hashes are block ids too
            # A truncated or lying entry count must not abort a whole batch with a struct.error traceback.
            # Skip only the table: the $0x... text keys below are still readable from the same bytes.
            if len(d) < 16:
                print(f"warning: JSAV header is only {len(d)} bytes; skipping its entry table", file=sys.stderr)
                n = 0
            else:
                n = sum(struct.unpack_from("<II", d, 8))
            if n and 16 + 20 * n > len(d):
                print(f"warning: JSAV header declares {n} entries but the file has {len(d)} bytes; "
                      "skipping its entry table", file=sys.stderr)
                n = 0
            for i in range(n):
                keys[struct.unpack_from("<I", d, 16 + i * 20)[0]] += 1
        for m in KEY.finditer(d):
            keys[int(m.group(1), 16)] += 1
    return keys


def harvest(paths):
    """Identifier-like strings from raw bytes and from every zlib stream found in them."""
    out = set()
    zhdr = re.compile(rb"\x78[\x01\x5e\x9c\xda]")
    for p in paths:
        d = read(p)
        out.update(m.group(0) for m in IDENT.finditer(d))
        pos = 0
        while True:
            m = zhdr.search(d, pos)
            if not m:
                break
            try:
                z = zlib.decompressobj()
                data = z.decompress(d[m.start():m.start() + (8 << 20)], 64 << 20)
                if z.eof and data:
                    out.update(x.group(0) for x in IDENT.finditer(data))
                    pos = m.start() + max(2, len(d[m.start():m.start() + (8 << 20)]) - len(z.unused_data))
                    continue
            except zlib.error:
                pass
            pos = m.start() + 2
    return out


def compose(strings, keys, n_words):
    """One- and two-word names (PascalCase, camelCase, snake_case) from the n most frequent words."""
    words = collections.Counter()
    for s in strings:
        for w in WORD.findall(s.decode("latin-1")):
            if len(w) >= 2 or w.isdigit():
                words[w.lower()] += 1
    vocab = [w for w, _ in words.most_common(n_words)]
    pascal = [w.capitalize().encode() for w in vocab]
    lower = [w.encode() for w in vocab]
    hits, tried = collections.defaultdict(set), 0
    for p, low in zip(pascal, lower):
        for c in (p, low, p.upper()):
            tried += 1
            v = joaat(c)
            if v in keys:
                hits[v].add(c)
    for firsts, sep, seconds in ((pascal, b"", pascal), (lower, b"", pascal), (lower, b"_", lower)):
        for a in firsts:
            ha = joaat_step(joaat_step(0, a), sep)
            for b in seconds:
                tried += 1
                v = joaat_final(joaat_step(ha, b))
                if v in keys:
                    hits[v].add(a + sep + b)
    return hits, tried


# ---------------------------------------------------------------------------------------------- commands
def cmd_collect(a):
    keys = collect_keys(a.inputs)
    print(f"{len(keys)} distinct keys, {sum(keys.values())} occurrences")


def cmd_resolve(a):
    keys = set(collect_keys(a.inputs))
    strings = harvest(a.dict)
    chance = len(strings) * len(keys) / 2 ** 32
    print(f"{len(keys)} distinct keys; {len(strings)} dictionary strings "
          f"(about {chance:.1f} keys per family expected by chance)")
    resolved = {}
    for name, fn in FAMILIES.items():
        hit = collections.defaultdict(set)
        for s in strings:
            v = fn(s)
            if v in keys:
                hit[v].add(s)
        extra = ""
        if name == "joaat":
            resolved = hit
        else:
            only = len(set(hit) - set(resolved))
            extra = f" ({only} not already resolved by joaat)"
        print(f"  {name:16} {len(hit):6} keys resolve{extra}")
    if a.compose:
        hits, tried = compose(strings, keys, a.compose)
        new = set(hits) - set(resolved)
        print(f"  joaat, composed  {len(new):6} more keys from {tried} one/two-word names "
              f"(about {tried * len(keys) / 2 ** 32:.1f} expected by chance)")
        for k in new:
            resolved[k] |= hits[k]
    print(f"  total (joaat)    {len(resolved):6} of {len(keys)} keys ({100 * len(resolved) / max(1, len(keys)):.1f}%)")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"%08x" % k: sorted(x.decode("latin-1") for x in v) for k, v in sorted(resolved.items())},
                      f, indent=1)
        print(f"wrote {a.out} (local only: it contains game names)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("inputs", nargs="+")
    r = sub.add_parser("resolve")
    r.add_argument("inputs", nargs="+")
    r.add_argument("--dict", nargs="+", required=True)
    r.add_argument("--compose", type=int, default=0, metavar="N")
    r.add_argument("--out")
    a = ap.parse_args(argv)
    {"collect": cmd_collect, "resolve": cmd_resolve}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
