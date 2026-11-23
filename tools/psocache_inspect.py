"""Check a PSO/shader cache blob against the measured 0x80/0x70/0x18 layout.

The blob starts with the magic ``PS\\0\\x09`` and, in the measured cache, has this layout (little-endian):

| Offset | Region | Size |
|---:|---|---:|
| 0x00 | header | 0x80 |
| 0x80 | graphics PSO records | count * 0x70 |
| | compute PSO records | count * 0x18 |
| | meta index | entries * 8 |
| | shader index | entries * 8 |
| | meta blob | u32 at 0x60 |
| | code blob | u32 at 0x64 |

The four counts sit at 0x50, 0x54, 0x58 and 0x5c; the two sizes at 0x60 and 0x64. The header also holds two
NUL-terminated ASCII strings (0x04 and 0x24, 10 and 11 bytes in the measured cache) and an 8-byte value at 0x48.
The regions are laid end to end and, when the header is right, they add up to the file size exactly.

Both indexes are lists of ``(u32 offset, u32 size)`` pairs. Meta-index offsets run from 0 over the meta blob;
shader-index offsets run from 0 over the code blob. A meta entry is a short header (0x10, 0x14, 0x18 or 0x1c in
the measured cache; up to 0x40 is accepted) followed by 32-byte chunks whose first word is 0x41, 0x42 or 0x43.
The 0x41 chunk's second word is the descriptor kind that the checked note names SRV 0x300, CBV 0x302, UAV 0x303
and sampler 0x002; the measured cache carries those four and also 0x401 and 0x100. The code blob is RDNA2 code:
each program ends at an ``s_endpgm`` (0xBF810000) and the space after it is filled with ``s_code_end``
(0xBF9F0000) words.

Every count, offset and size in the report comes from the file's header; the only constants are the magic, the
field offsets above and the two marker words. The tool never rewrites the file and prints counts, offsets and
sizes only. When the file disagrees with the model it prints what does not fit and keeps the checks that are
still in bounds.

Usage:
    python -I psocache_inspect.py <cache> [--json] [--quiet]

Exit code: 0 when every region fits and the regions add up to the file size, 1 when the file disagrees with the
model, 2 when it cannot be read or does not start with the magic.
"""

import argparse
import collections
import json
import mmap
import os
import struct
import sys

MAGIC = b"PS\x00\x09"
HEADER_SIZE = 0x80
GRAPHICS_STRIDE = 0x70
COMPUTE_STRIDE = 0x18
INDEX_STRIDE = 8
CHUNK_SIZE = 32
CHUNK_TYPES = (0x41, 0x42, 0x43)
ENTRY_HEADER_MAX = 0x40  # largest per-entry meta header the model allows
S_END_PGM = b"\x00\x00\x81\xbf"  # s_endpgm, 0xBF810000 little-endian
S_CODE_END = b"\x00\x00\x9f\xbf"  # s_code_end, 0xBF9F0000 little-endian
HASH_OFF = 0x48
STRING_OFFS = (0x04, 0x24)
COUNT_OFFS = (0x50, 0x54, 0x58, 0x5C)
SIZE_OFFS = (0x60, 0x64)


def u32(m, off):
    return struct.unpack_from("<I", m, off)[0]


def count_marker(m, pattern, lo, hi):
    """Occurrences of a 4-byte pattern in [lo, hi), read-only through the map."""
    n = 0
    p = lo
    while True:
        j = m.find(pattern, p, hi)
        if j < 0:
            return n
        n += 1
        p = j + len(pattern)


def region(name, offset, count, stride, size, file_size):
    """One fixed-stride or byte-count region, bounded against the file."""
    nbytes = count * stride if stride is not None else size
    end = offset + nbytes
    return {
        "offset": offset,
        "count": count,
        "stride": stride,
        "bytes": nbytes,
        "end": end,
        "fits": end <= file_size,
        "shortfall": max(0, end - file_size),
    }


def parse_chunks(m, base, start, size):
    """Parse one meta entry as ``header + n * 32`` chunks.

    The first header length, in 4-byte steps, that covers the entry exactly with 32-byte chunks all of type
    0x41/0x42/0x43 wins. Returns ``(header_size, [(tag, word1), ...])`` or ``(None, None)`` when no length fits.
    """
    for header in range(0, ENTRY_HEADER_MAX + 1, 4):
        if size < header or (size - header) % CHUNK_SIZE:
            continue
        tags = []
        ok = True
        for pos in range(start + header, start + size, CHUNK_SIZE):
            tag = u32(m, base + pos)
            if tag not in CHUNK_TYPES:
                ok = False
                break
            tags.append((tag, u32(m, base + pos + 4)))
        if ok:
            return header, tags
    return None, None


def inspect(path):
    """Read ``path`` and return the report as a dict. Raises OSError/ValueError on unusable input."""
    with open(path, "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        if size < HEADER_SIZE:
            raise ValueError(
                "file is %d bytes, shorter than the 0x%x-byte header"
                % (size, HEADER_SIZE)
            )
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
            magic = bytes(m[:4])
            if magic != MAGIC:
                raise ValueError("no PS\\0\\x09 magic at 0x0 (found %s)" % magic.hex())
            report = {
                "file": path,
                "size": size,
                "magic_ok": True,
                "header_size": HEADER_SIZE,
            }
            counts = dict(
                zip(
                    ("graphics", "compute", "meta_index", "shader_index"),
                    (u32(m, off) for off in COUNT_OFFS),
                )
            )
            sizes = dict(zip(("meta", "code"), (u32(m, off) for off in SIZE_OFFS)))
            report["counts"] = counts
            report["sizes"] = sizes
            report["header_hash"] = m[HASH_OFF : HASH_OFF + 8].hex()
            report["header_strings"] = []
            for off in STRING_OFFS:
                end = m.find(b"\x00", off, HEADER_SIZE)
                report["header_strings"].append(
                    {"offset": off, "length": (end - off) if end >= 0 else 0}
                )
            problems = []
            if any(
                m[off : off + 4] != b"\x00\x00\x00\x00"
                for off in range(0x68, HEADER_SIZE, 4)
            ):
                problems.append("header words at 0x68..0x7f are not zero")

            off = 0x80
            regions = collections.OrderedDict()
            for name, count, stride, nbytes in (
                ("graphics", counts["graphics"], GRAPHICS_STRIDE, None),
                ("compute", counts["compute"], COMPUTE_STRIDE, None),
                ("meta_index", counts["meta_index"], INDEX_STRIDE, None),
                ("shader_index", counts["shader_index"], INDEX_STRIDE, None),
                ("meta_blob", None, None, sizes["meta"]),
                ("code_blob", None, None, sizes["code"]),
            ):
                reg = region(name, off, count, stride, nbytes, size)
                regions[name] = reg
                off = reg["end"]
                if not reg["fits"]:
                    problems.append(
                        "%s region ends at 0x%x, 0x%x bytes past the file end"
                        % (name, reg["end"], reg["shortfall"])
                    )
            report["regions"] = regions
            report["regions_end"] = off
            report["slack"] = size - off
            if off != size:
                problems.append(
                    "regions end at 0x%x, %d bytes %s the file size 0x%x"
                    % (off, abs(off - size), "past" if off > size else "short of", size)
                )
            report["problems"] = problems
            report["references"] = None
            report["meta_index"] = None
            report["shader_index"] = None
            report["chunks"] = None
            report["markers"] = None

            meta_off = regions["meta_blob"]["offset"]
            code_off = regions["code_blob"]["offset"]
            meta_size = sizes["meta"]
            code_size = sizes["code"]

            # The PSO records name meta-index entries in their small u32 fields (0xffffffff is a hole). All of
            # them must stay below the meta-index count: a wrong count or stride shifts the records and shows up
            # here as out-of-range references.
            if (
                regions["graphics"]["fits"]
                and regions["compute"]["fits"]
                and counts["meta_index"]
            ):
                limit = counts["meta_index"]
                bad_graphics = 0
                for i in range(counts["graphics"]):
                    base = regions["graphics"]["offset"] + i * GRAPHICS_STRIDE
                    for delta in (8, 0xC, 0x50, 0x54, 0x58, 0x5C):
                        value = u32(m, base + delta)
                        if value != 0xFFFFFFFF and value >= limit:
                            bad_graphics += 1
                bad_compute = 0
                for i in range(counts["compute"]):
                    base = regions["compute"]["offset"] + i * COMPUTE_STRIDE
                    for delta in (8, 0xC, 0x10, 0x14):
                        if u32(m, base + delta) >= limit:
                            bad_compute += 1
                report["references"] = {
                    "limit": limit,
                    "graphics_out_of_range": bad_graphics,
                    "compute_out_of_range": bad_compute,
                }
                if bad_graphics or bad_compute:
                    problems.append(
                        "%d graphics and %d compute record references are outside the %d-entry meta index"
                        % (bad_graphics, bad_compute, limit)
                    )

            # Meta index: contiguous (offset, size) pairs covering the meta blob from 0.
            if regions["meta_index"]["fits"] and regions["meta_blob"]["fits"]:
                entries = []
                bad = 0
                for i in range(counts["meta_index"]):
                    a, b = struct.unpack_from(
                        "<II", m, regions["meta_index"]["offset"] + i * INDEX_STRIDE
                    )
                    if a + b > meta_size:
                        bad += 1
                    entries.append((a, b))
                gaps = overlaps = 0
                expected = 0
                for a, b in entries:
                    if a > expected:
                        gaps += 1
                    elif a < expected:
                        overlaps += 1
                    expected = max(expected, a + b)
                report["meta_index"] = {
                    "entries": len(entries),
                    "out_of_range": bad,
                    "gaps": gaps,
                    "overlaps": overlaps,
                    "coverage": expected,
                    "declared": meta_size,
                    "contiguous": gaps == 0
                    and overlaps == 0
                    and expected == meta_size
                    and bad == 0,
                }
                if bad:
                    problems.append(
                        "%d meta index entries point past the meta blob" % bad
                    )
                if gaps or overlaps:
                    problems.append(
                        "meta index is not contiguous: %d gaps, %d overlaps"
                        % (gaps, overlaps)
                    )
                if expected != meta_size:
                    problems.append(
                        "meta index covers 0x%x bytes of the 0x%x-byte meta blob"
                        % (expected, meta_size)
                    )

                # Chunk histogram. Entries that are not 0x41/0x42/0x43 lists are counted, not guessed.
                parsed = unparsed = no_chunks = 0
                headers = collections.Counter()
                types = collections.Counter()
                user_tags = collections.Counter()
                for a, b in entries:
                    if a + b > meta_size:
                        unparsed += 1
                        continue
                    header, tags = parse_chunks(m, meta_off, a, b)
                    if header is None:
                        unparsed += 1
                        continue
                    headers[header] += 1
                    if not tags:
                        no_chunks += 1
                        continue
                    parsed += 1
                    for tag, word1 in tags:
                        types[tag] += 1
                        if tag == 0x41:
                            user_tags[word1] += 1
                report["chunks"] = {
                    "entries": len(entries),
                    "chunk_lists": parsed,
                    "empty_lists": no_chunks,
                    "unparsed": unparsed,
                    "entry_headers": {hex(k): v for k, v in sorted(headers.items())},
                    "types": {hex(k): v for k, v in sorted(types.items())},
                    "user_tags": {hex(k): v for k, v in sorted(user_tags.items())},
                }

            # Shader index: (offset, size) pairs inside the code blob, in ascending offset order.
            if regions["shader_index"]["fits"] and regions["code_blob"]["fits"]:
                entries = []
                bad = 0
                for i in range(counts["shader_index"]):
                    a, b = struct.unpack_from(
                        "<II", m, regions["shader_index"]["offset"] + i * INDEX_STRIDE
                    )
                    if a + b > code_size:
                        bad += 1
                    entries.append((a, b))
                ascending = all(
                    entries[i][0] <= entries[i + 1][0] for i in range(len(entries) - 1)
                )
                max_end = max((a + b for a, b in entries), default=0)
                report["shader_index"] = {
                    "entries": len(entries),
                    "out_of_range": bad,
                    "ascending": ascending,
                    "max_end": max_end,
                    "code_size": code_size,
                    "slack": code_size - max_end,
                }
                if bad:
                    problems.append(
                        "%d shader index entries point past the code blob" % bad
                    )
                if not ascending:
                    problems.append("shader index offsets are not ascending")
                if max_end > code_size:
                    problems.append(
                        "shader index reaches 0x%x, past the 0x%x-byte code blob"
                        % (max_end, code_size)
                    )

                # Markers: total counts, and how many s_endpgm each shader entry holds.
                per_entry = collections.Counter()
                for a, b in entries:
                    if a + b > code_size:
                        continue
                    per_entry[
                        min(
                            4,
                            count_marker(m, S_END_PGM, code_off + a, code_off + a + b),
                        )
                    ] += 1
                report["markers"] = {
                    "s_endpgm": count_marker(
                        m, S_END_PGM, code_off, code_off + code_size
                    ),
                    "s_code_end": count_marker(
                        m, S_CODE_END, code_off, code_off + code_size
                    ),
                    "entries_with_none": per_entry.get(0, 0),
                    "entries_with_one": per_entry.get(1, 0),
                    "entries_with_two": per_entry.get(2, 0),
                    "entries_with_three_or_more": per_entry.get(3, 0)
                    + per_entry.get(4, 0),
                }

            report["problems"] = problems
            report["verdict"] = (
                "model holds" if not problems else "disagrees with the model"
            )
            return report


def print_report(r, out=sys.stdout):
    p = lambda *a: print(*a, file=out)
    p("file: %d bytes (0x%x)" % (r["size"], r["size"]))
    p("magic: PS\\0\\x09 at 0x0, header %d bytes" % r["header_size"])
    p(
        "header: hash at 0x%x, strings %s"
        % (
            HASH_OFF,
            ", ".join(
                "0x%x (%d bytes)" % (s["offset"], s["length"])
                for s in r["header_strings"]
            ),
        )
    )
    p(
        "counts: graphics=%d compute=%d meta_index=%d shader_index=%d"
        % (
            r["counts"]["graphics"],
            r["counts"]["compute"],
            r["counts"]["meta_index"],
            r["counts"]["shader_index"],
        )
    )
    p("sizes: meta=%d code=%d" % (r["sizes"]["meta"], r["sizes"]["code"]))
    p("regions:")
    for name, g in r["regions"].items():
        unit = ("x 0x%x" % g["stride"]) if g["stride"] is not None else "bytes"
        count = ("%d " % g["count"]) if g["count"] is not None else ""
        p(
            "  %-12s off 0x%08x  %s%s = %d  end 0x%08x  %s"
            % (
                name,
                g["offset"],
                count,
                unit,
                g["bytes"],
                g["end"],
                "fits"
                if g["fits"]
                else "DOES NOT FIT (%d bytes short)" % g["shortfall"],
            )
        )
    p(
        "regions end 0x%x, file 0x%x, slack %d"
        % (r["regions_end"], r["size"], r["slack"])
    )
    refs = r["references"]
    if refs:
        p(
            "record references: meta index limit %d, graphics out of range %d, compute out of range %d"
            % (
                refs["limit"],
                refs["graphics_out_of_range"],
                refs["compute_out_of_range"],
            )
        )
    mi = r["meta_index"]
    if mi:
        p(
            "meta index: %d entries, coverage %d of %d, gaps %d, overlaps %d, out of range %d, %s"
            % (
                mi["entries"],
                mi["coverage"],
                mi["declared"],
                mi["gaps"],
                mi["overlaps"],
                mi["out_of_range"],
                "contiguous" if mi["contiguous"] else "NOT contiguous",
            )
        )
    si = r["shader_index"]
    if si:
        p(
            "shader index: %d entries, max end 0x%x of 0x%x, slack %d, %s, out of range %d"
            % (
                si["entries"],
                si["max_end"],
                si["code_size"],
                si["slack"],
                "ascending" if si["ascending"] else "NOT ascending",
                si["out_of_range"],
            )
        )
    ch = r["chunks"]
    if ch:
        p(
            "meta chunks: %d of %d entries are chunk lists (%d empty, %d unparsed); headers %s"
            % (
                ch["chunk_lists"],
                ch["entries"],
                ch["empty_lists"],
                ch["unparsed"],
                " ".join("%s=%d" % (k, v) for k, v in ch["entry_headers"].items()),
            )
        )
        p(
            "  chunk types: %s"
            % " ".join("%s=%d" % (k, v) for k, v in ch["types"].items())
        )
        p(
            "  0x41 user tags: %s"
            % " ".join("%s=%d" % (k, v) for k, v in ch["user_tags"].items())
        )
    mk = r["markers"]
    if mk:
        p(
            "markers: s_endpgm %d, s_code_end %d in the code blob"
            % (mk["s_endpgm"], mk["s_code_end"])
        )
        p(
            "  shader entries by s_endpgm count: 0=%d 1=%d 2=%d 3+=%d"
            % (
                mk["entries_with_none"],
                mk["entries_with_one"],
                mk["entries_with_two"],
                mk["entries_with_three_or_more"],
            )
        )
    for problem in r["problems"]:
        p("problem: %s" % problem)
    p("verdict: %s" % r["verdict"])


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Check a PS\\0\\x09 PSO/shader cache against the 0x80/0x70/0x18 layout."
    )
    ap.add_argument("cache", help="cache file (read-only)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    ap.add_argument(
        "--quiet", action="store_true", help="print nothing, only set the exit code"
    )
    a = ap.parse_args(argv)
    try:
        report = inspect(a.cache)
    except (OSError, ValueError) as e:
        print("%s: %s" % (a.cache, e), file=sys.stderr)
        return 2
    if not a.quiet:
        if a.json:
            print(json.dumps(report, indent=1))
        else:
            print_report(report)
    return 0 if not report["problems"] else 1


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
