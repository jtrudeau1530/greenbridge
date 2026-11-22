"""Index the `63 30 00 00` mesh pages (vertex blocks plus meshlet groups) in carved streams or an RPF9 archive.

`docs/story.md` lists 45 inflated streams in `shared_minigame000.rpf` that start with `63 30 00 00`. They are
geometry pages: a run of 16-byte-aligned chunks, each either a vertex block or a meshlet group.

Vertex block (header 0x60 bytes, first u32 is 0x00003063):
    +0x00..+0x58  23 slot words. A non-zero word is tag (low byte) | offset/2 (high 24 bits); the offset is
                  from the block start. Slot 0 is always tag 0x63 at offset 0x60.
    +0x5C         u32, 0x200 in every block measured.
    slot 0  tag 0x63  12 bytes/vertex  float3 position
    slot 1  tag 0x26   4 bytes/vertex  RGBA8 (absent in some blocks)
    slot 2  tag 0x63  12 bytes/vertex  float3 normal (absent in some blocks)
    slot 3  tag 0x42   8 bytes/vertex  float2
    slot 4  tag 0x42   8 bytes/vertex  float2 (absent in some blocks)
    The streams are stored in slot order 0, 2, 1, 3, 4. Only slot 0 is required. The vertex count is the span
    from slot 0 to the next stream in offset order, divided by 12; the block ends 8 * count bytes after its last
    slot.

Meshlet group (first u32 has bit 31 set):
    +0   u32  0x80000000 | meshlet count K
    +4   u32  triangle count T
    +8   u32  vertex reference count V
    +12  K * 16-byte meshlet records:
         u32 w0: bits 0-6 = vertices - 1, bits 7-13 = triangles - 1, bits 14-31 not decoded
         u32 w1: low u16 = first vertex reference, high u16 = first triangle (running sums)
         u32 w2, w3: not decoded (0x007FF7FF and 0x7FF in most records)
    then T u32 triangles: three 7-bit meshlet-local indices (bits 0-6, 7-13, 14-20), meshlet index in bits 21-31
    then V u16 vertex references into a vertex block

A page starts with either kind: 45 streams start with a block (`63 30 00 00`) and others start with a group
(continuation pages). The walk stops, without an error, at a `42 40 00 00` block (the same slot-word header with
other vertex formats, not sized here) or at a group that continues past the end of the stream. Only counts and layout checks are written; no vertex, index or float value leaves this tool.

Usage:
    python -I tools/meshlet_page.py <file|dir|archive.rpf> [...] [--json out.json] [--quiet]
    python -I tools/meshlet_page.py <graphics-segment> --draw-system <system-segment> --geometry-offset <offset> [--json out.json]
    python -I tools/meshlet_page.py <graphics-segment> --draw-system <system-segment> --drawable-offset <offset>
"""
import argparse
import json
import math
import mmap
import os
import re
import struct
import sys
import zlib

BLOCK_MAGIC = 0x3063
OTHER_BLOCK_MAGIC = 0x4042
HEADER_SIZE = 0x60
SLOT_COUNT = 23
SLOT_STRIDE = {0: 12, 1: 4, 2: 12, 3: 8, 4: 8}
SLOT_TAG = {0: 0x63, 1: 0x26, 2: 0x63, 3: 0x42, 4: 0x42}
ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]")


class PageError(ValueError):
    pass


def _u32(data, at):
    if at + 4 > len(data):
        raise PageError(f"truncated at +0x{at:x}")
    return struct.unpack_from("<I", data, at)[0]


def parse_block(data, at):
    """Parse a vertex block at `at`. Returns (record, end offset)."""
    if at + HEADER_SIZE > len(data):
        raise PageError(f"block header truncated at +0x{at:x}")
    words = struct.unpack_from("<24I", data, at)
    if words[0] != BLOCK_MAGIC:
        raise PageError(f"no block magic at +0x{at:x}")
    slots = {i: (w & 0xFF, (w >> 8) * 2) for i, w in enumerate(words[:SLOT_COUNT]) if w}
    if 0 not in slots:
        raise PageError(f"block at +0x{at:x} has no slot 0")
    if slots[0][1] != HEADER_SIZE:
        raise PageError(f"block at +0x{at:x}: slot 0 is not at +0x60")
    order = sorted(slots, key=lambda i: slots[i][1])
    if len(order) < 2 or order[0] != 0:
        raise PageError(f"block at +0x{at:x}: slot 0 is not followed by another stream")
    span = slots[order[1]][1] - slots[0][1]
    if span <= 0 or span % SLOT_STRIDE[0]:
        raise PageError(f"block at +0x{at:x}: slot {order[1]} offset is not 12 * count past slot 0")
    count = span // SLOT_STRIDE[0]
    for here, nxt in zip(order, order[1:]):
        stride = SLOT_STRIDE.get(here)
        if stride is None or slots[nxt][1] - slots[here][1] != stride * count:
            raise PageError(f"block at +0x{at:x}: slot {here} size is not stride * count")
    last = order[-1]
    if last not in (3, 4):
        raise PageError(f"block at +0x{at:x}: last slot is {last}, expected a float2 slot")
    end = at + slots[last][1] + 8 * count
    if end > len(data):
        raise PageError(f"block at +0x{at:x} runs past the page end")
    record = {"kind": "block", "offset": at, "vertices": count, "flags": words[23],
              "slots": [[i, slots[i][0], slots[i][1]] for i in sorted(slots)],
              "tags_expected": all(SLOT_TAG.get(i) == t for i, (t, _) in slots.items())}
    return record, end


def parse_group(data, at):
    """Parse and validate a meshlet group at `at`. Returns (record, end offset)."""
    head = _u32(data, at)
    if not head & 0x80000000:
        raise PageError(f"no group header at +0x{at:x}")
    meshlets = head & 0x7FFFFFFF
    triangles, refs = _u32(data, at + 4), _u32(data, at + 8)
    rec_at = at + 12
    tri_at = rec_at + 16 * meshlets
    ref_at = tri_at + 4 * triangles
    end = ref_at + 2 * refs
    if end > len(data):
        raise PageError(f"group at +0x{at:x} runs past the page end")
    vbase = tbase = 0
    full = 0
    for m in range(meshlets):
        w0, w1, _, _ = struct.unpack_from("<4I", data, rec_at + 16 * m)
        vcount, tcount = (w0 & 0x7F) + 1, ((w0 >> 7) & 0x7F) + 1
        if (w1 & 0xFFFF) != (vbase & 0xFFFF) or (w1 >> 16) != (tbase & 0xFFFF):
            raise PageError(f"group at +0x{at:x}: meshlet {m} offsets are not running sums")
        if tbase + tcount > triangles or vbase + vcount > refs:
            raise PageError(f"group at +0x{at:x}: meshlet {m} runs past the group")
        for t in struct.unpack_from(f"<{tcount}I", data, tri_at + 4 * tbase):
            if (t >> 21) != m or max(t & 0x7F, (t >> 7) & 0x7F, (t >> 14) & 0x7F) >= vcount:
                raise PageError(f"group at +0x{at:x}: bad triangle in meshlet {m}")
        full += vcount == 128
        vbase += vcount
        tbase += tcount
    if vbase != refs or tbase != triangles:
        raise PageError(f"group at +0x{at:x}: meshlet sums do not match the header")
    vref = struct.unpack_from(f"<{refs}H", data, ref_at) if refs else ()
    span = (max(vref) + 1) if vref else 0
    record = {"kind": "group", "offset": at, "meshlets": meshlets, "triangles": triangles,
              "vertex_refs": refs, "vertex_span": span, "unique_refs": len(set(vref)),
              "full_meshlets": full}
    return record, end


def is_mesh_page(data):
    """True for a page that starts with a vertex block, or with a meshlet group that validates."""
    if len(data) < 16:
        return False
    word = struct.unpack_from("<I", data, 0)[0]
    if word == BLOCK_MAGIC:
        return True
    if not word & 0x80000000:
        return False
    try:
        parse_group(data, 0)
    except PageError:
        return False
    return True


def parse_page(data):
    """Walk a whole page. Returns a list of chunk records; raises PageError on anything unexpected."""
    if len(data) < 4 or not is_mesh_page(data):
        raise PageError("page starts with neither a vertex block nor a meshlet group")
    chunks = []
    pos = 0
    while True:
        while pos < len(data) and data[pos] == 0:
            pos += 1
        if pos >= len(data):
            break
        pos &= ~15
        while _u32(data, pos) == 0:
            pos += 4
        if pos % 16:
            raise PageError(f"chunk at +0x{pos:x} is not 16-byte aligned")
        word = _u32(data, pos)
        if word == BLOCK_MAGIC:
            record, pos = parse_block(data, pos)
        elif word == OTHER_BLOCK_MAGIC:
            # A `42 40 00 00` block: same slot-word header, other vertex formats; not sized here.
            chunks.append({"kind": "stop", "offset": pos, "reason": "42 40 00 00 block"})
            break
        elif word & 0x80000000:
            try:
                record, pos = parse_group(data, pos)
            except PageError as exc:
                if "past the page end" not in str(exc):
                    raise
                chunks.append({"kind": "stop", "offset": pos, "reason": "group continues past the stream"})
                break
        else:
            raise PageError(f"unknown chunk at +0x{pos:x}")
        chunks.append(record)
    return chunks


def block_checks(data, block):
    """Count-only plausibility checks for one block's streams. Returns a dict of counts."""
    n = block["vertices"]
    at = block["offset"]
    slots = {i: off for i, _, off in block["slots"]}
    out = {"vertices": n, "positions_finite": 0, "positions_within_300m": 0, "normals_unit": 0,
           "colour_alpha_255": 0, "colour_vertices": 0, "uv_finite": 0, "uv_vertices": 0,
           "position_low_byte_zero": 0}
    pos = struct.unpack_from(f"<{3 * n}f", data, at + slots[0])
    for v in range(n):
        p = pos[3 * v:3 * v + 3]
        if all(math.isfinite(c) for c in p):
            out["positions_finite"] += 1
            out["positions_within_300m"] += all(abs(c) <= 300 for c in p)
    if 2 in slots:
        nrm = struct.unpack_from(f"<{3 * n}f", data, at + slots[2])
        for v in range(n):
            q = nrm[3 * v:3 * v + 3]
            if all(math.isfinite(c) for c in q) and abs(math.sqrt(sum(c * c for c in q)) - 1) < 0.02:
                out["normals_unit"] += 1
    raw = data[at + slots[0]:at + slots[0] + 12 * n]
    out["position_low_byte_zero"] = sum(1 for i in range(0, len(raw), 4) if raw[i] == 0)
    if 1 in slots:
        out["colour_vertices"] = n
        out["colour_alpha_255"] = sum(1 for i in range(3, 4 * n, 4) if data[at + slots[1] + i] == 255)
    for s in (3, 4):
        if s in slots:
            uv = struct.unpack_from(f"<{2 * n}f", data, at + slots[s])
            out["uv_vertices"] += n
            out["uv_finite"] += sum(1 for v in range(n) if math.isfinite(uv[2 * v]) and math.isfinite(uv[2 * v + 1]))
    return out


def summarise_page(data, name):
    try:
        chunks = parse_page(data)
    except PageError as exc:
        return {"name": name, "size": len(data), "error": str(exc)}
    blocks = [c for c in chunks if c["kind"] == "block"]
    groups = [c for c in chunks if c["kind"] == "group"]
    checks = {}
    for b in blocks:
        for k, v in block_checks(data, b).items():
            checks[k] = checks.get(k, 0) + v
    counts = {b["vertices"] for b in blocks}
    stop = [c for c in chunks if c["kind"] == "stop"]
    return {"name": name, "size": len(data), "starts_with": chunks[0]["kind"],
            "sequence": "".join({"block": "B", "group": "G", "stop": "."}[c["kind"]] for c in chunks),
            "stop": stop[0]["reason"] if stop else None,
            "blocks": len(blocks), "groups": len(groups),
            "blocks_with_colour": sum(1 for b in blocks if any(s[0] == 1 for s in b["slots"])),
            "blocks_with_uv1": sum(1 for b in blocks if any(s[0] == 4 for s in b["slots"])),
            "block_flags": sorted({b["flags"] for b in blocks}),
            "tags_expected": all(b["tags_expected"] for b in blocks),
            "meshlets": sum(g["meshlets"] for g in groups),
            "full_meshlets": sum(g["full_meshlets"] for g in groups),
            "triangles": sum(g["triangles"] for g in groups),
            "vertex_refs": sum(g["vertex_refs"] for g in groups),
            "groups_dense": sum(1 for g in groups if g["unique_refs"] == g["vertex_span"]),
            "groups_matching_block_here": sum(1 for g in groups if g["vertex_span"] in counts),
            "group_spans": [g["vertex_span"] for g in groups],
            "block_counts": [b["vertices"] for b in blocks],
            "checks": checks}


def iter_archive(path):
    """Yield (name, inflated bytes) for every zlib stream in an RPF9 data area that is_mesh_page accepts."""
    with open(path, "rb") as fh:
        if not os.fstat(fh.fileno()).st_size:
            return
        m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            size = len(m)
            pos = struct.unpack_from("<I", m, 4)[0] if size >= 8 else size
            while pos < size:
                match = ZHDR.search(m, pos)
                if not match:
                    break
                s = match.start()
                d = zlib.decompressobj()
                chunks, p = [], s
                try:
                    while not d.eof and p < size:
                        block = m[p:p + (1 << 20)]
                        chunks.append(d.decompress(block))
                        p += len(block)
                        if d.unused_data or d.eof:
                            break
                    out = b"".join(chunks) + d.flush()
                except zlib.error:
                    pos = s + 1
                    continue
                if not d.eof or len(out) < 64:
                    pos = s + 1
                    continue
                if is_mesh_page(out):
                    yield f"0x{s:x}", out
                pos = p - len(d.unused_data)
        finally:
            m.close()


def iter_inputs(paths):
    for path in paths:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for f in sorted(files):
                    full = os.path.join(root, f)
                    with open(full, "rb") as fh:
                        head = fh.read(4)
                        if len(head) == 4 and (head == b"\x63\x30\x00\x00" or head[3] & 0x80):
                            data = head + fh.read()
                            if is_mesh_page(data):
                                yield os.path.relpath(full, path), data
        elif path.lower().endswith(".rpf"):
            yield from iter_archive(path)
        else:
            with open(path, "rb") as fh:
                yield os.path.basename(path), fh.read()


def totals(pages):
    keys = ("blocks", "groups", "blocks_with_colour", "blocks_with_uv1", "meshlets", "full_meshlets",
            "triangles", "vertex_refs", "groups_dense", "groups_matching_block_here")
    out = {"pages": len(pages), "pages_parsed": sum(1 for p in pages if "error" not in p),
           "block_led_pages": sum(1 for p in pages if p.get("starts_with") == "block"),
           "group_led_pages": sum(1 for p in pages if p.get("starts_with") == "group"),
           "stopped_at_4240_block": sum(1 for p in pages if p.get("stop") == "42 40 00 00 block"),
           "stopped_in_group": sum(1 for p in pages if p.get("stop") == "group continues past the stream"),
           "bytes": sum(p["size"] for p in pages)}
    good = [p for p in pages if "error" not in p]
    for k in keys:
        out[k] = sum(p[k] for p in good)
    out["vertices"] = sum(sum(p["block_counts"]) for p in good)
    checks = {}
    for p in good:
        for k, v in p["checks"].items():
            checks[k] = checks.get(k, 0) + v
    out["checks"] = checks
    out["block_flags"] = sorted({f for p in good for f in p["block_flags"]})
    # A group's vertex span that matches a block in the same page or a neighbouring page (input order).
    near = 0
    for i, p in enumerate(good):
        pool = set(p["block_counts"])
        for j in (i - 1, i + 1):
            if 0 <= j < len(good):
                pool |= set(good[j]["block_counts"])
        near += sum(1 for s in p["group_spans"] if s in pool)
    out["groups_matching_block_near"] = near
    return out


def _resource_pointer(system, at, base, segment, size):
    if not 0 <= at <= len(system) - 8:
        raise PageError("pointer field outside the system segment")
    address = struct.unpack_from("<Q", system, at)[0] - base
    if not 0 <= address <= len(segment) - size:
        raise PageError("buffer or metadata pointer outside its segment")
    return address


def drawable_geometries(system, drawable):
    """Read the primary LOD geometry list at a caller-identified drawable."""
    def pointer(at, size):
        return _resource_pointer(system, at, 0x50000000, system, size)

    def count(at):
        used, capacity = struct.unpack_from("<HH", system, at)
        if not 1 <= used <= 256 or used != capacity:
            raise PageError("invalid drawable array count")
        return used

    if not 0 <= drawable <= len(system) - 176:
        raise PageError("drawable outside the system segment")
    lod = pointer(drawable + 80, 16)
    models = count(lod + 8)
    table = pointer(lod, 8 * models)
    for i in range(models):
        model = pointer(table + 8 * i, 56)
        geometries = count(model + 16)
        if struct.unpack_from("<H", system, model + 54)[0] != geometries:
            raise PageError("inconsistent model geometry count")
        entries = pointer(model + 8, 8 * geometries)
        for j in range(geometries):
            yield pointer(entries + 8 * j, 160)


def draw_binding(system, graphics, geometry):
    """Resolve a known geometry record in paired, inflated resource segments."""

    def system_pointer(at, size):
        return _resource_pointer(system, at, 0x50000000, system, size)

    def buffer(at):
        count, stride = struct.unpack_from("<IH", system, at + 8)
        if not count or not 1 <= stride <= 32:
            raise PageError("invalid buffer extent")
        length = count * stride
        return _resource_pointer(system, at + 24, 0x60000000, graphics, length), length, stride

    if not 0 <= geometry <= len(system) - 160:
        raise PageError("geometry outside the system segment")
    vb = system_pointer(geometry + 8, 80)
    layout = system_pointer(vb + 64, 320)
    descriptor = system_pointer(geometry + 64, 16)
    ib = system_pointer(descriptor, 64)
    vertex, vertex_bytes, _ = buffer(vb)
    index, index_bytes, stride = buffer(ib)
    flags = _u32(system, layout + 0x138)
    count = flags >> 10
    if (not flags & 1 or not count or _u32(system, layout) != 128
            or system[layout + 0xD0] != 8 or system[layout + 0x104] != 16
            or vertex_bytes < 128 + count * 8 or _u32(graphics, vertex) != OTHER_BLOCK_MAGIC):
        raise PageError("unsupported vertex layout or invalid position extent")
    meshlets, triangles, binding = struct.unpack_from("<HHI", system, descriptor + 8)
    refs = (binding >> 8) & 0x7FFFFF
    if (stride != 4 or index_bytes < 12 or binding >> 31 or not (meshlets and triangles and refs)
            or struct.unpack_from("<III", graphics, index) != (0x80000000 | meshlets, triangles, refs)):
        raise PageError("unsupported or inconsistent meshlet descriptor")
    data = memoryview(graphics)[index:index + index_bytes]
    group, end = parse_group(data, 0)
    if (group["vertex_span"] > count or ((end + 15) & ~15) != index_bytes or any(data[end:])):
        raise PageError("group references or padded extent exceed the declared layout")
    return {"vertex_block": vertex, "meshlet_group": index, "vertices": count}


def same_output_path(first, second):
    if os.path.normcase(os.path.realpath(first)) == os.path.normcase(os.path.realpath(second)):
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def is_scanned_input(path, input_path):
    """True when a report path is one of the files the scan reads: writing it would destroy the input."""
    if same_output_path(path, input_path):
        return True
    if not os.path.isdir(input_path) or not os.path.isfile(path):
        return False
    manifest = os.path.join(input_path, "index.json")
    if os.path.isfile(manifest) and same_output_path(path, manifest):
        return True
    root = os.path.normcase(os.path.abspath(input_path))
    target = os.path.normcase(os.path.abspath(path))
    if os.path.dirname(target) != root and not target.startswith(root + os.sep):
        return False
    base = os.path.basename(target)
    return base.endswith(".bin") or base.endswith(".rpf")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+", help="carved page files, folders of them, or .rpf archives")
    ap.add_argument("--json", help="write per-page records and totals here")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--draw-system", help="inflated system segment paired with one graphics segment")
    select = ap.add_mutually_exclusive_group()
    select.add_argument("--geometry-offset", action="append", type=lambda s: int(s, 0),
                        help="known geometry record offset in the system segment (repeatable)")
    select.add_argument("--drawable-offset", type=lambda s: int(s, 0),
                        help="walk primary LOD geometry records from this system-segment drawable")
    args = ap.parse_args(argv)
    if args.json:
        if os.path.isdir(args.json):
            print(f"error: --json {args.json} is a directory", file=sys.stderr)
            return 2
        candidates = list(args.inputs)
        if args.draw_system:
            candidates.append(args.draw_system)
        for inp in candidates:
            if is_scanned_input(args.json, inp):
                print(f"error: --json would overwrite the input {inp}", file=sys.stderr)
                return 2
    if args.draw_system is not None or args.geometry_offset is not None or args.drawable_offset is not None:
        if not args.draw_system or (not args.geometry_offset and args.drawable_offset is None) or len(args.inputs) != 1:
            ap.error("draw binding needs one graphics segment, --draw-system and a geometry or drawable offset")
        try:
            with open(args.draw_system, "rb") as source:
                system = source.read()
            with open(args.inputs[0], "rb") as source:
                graphics = source.read()
            rows = []
            offsets = args.geometry_offset or drawable_geometries(system, args.drawable_offset)
            try:
                for offset in offsets:
                    try:
                        rows.append({"geometry": offset, **draw_binding(system, graphics, offset)})
                    except PageError as exc:
                        rows.append({"geometry": offset, "unresolved": str(exc)})
            except PageError as exc:
                rows.append({"drawable": args.drawable_offset, "unresolved": str(exc)})
            report = json.dumps({"bindings": rows}, indent=2)
            if args.json:
                try:
                    with open(args.json, "w", encoding="utf-8", newline="\n") as out:
                        out.write(report + "\n")
                except OSError as exc:
                    print(f"error: cannot write {args.json}: {exc.strerror or exc}", file=sys.stderr)
                    return 2
            if not args.quiet:
                print(report)
            return int(any("unresolved" in row for row in rows))
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    for path in args.inputs:
        if not os.path.exists(path):
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2
    pages = [summarise_page(data, name) for name, data in iter_inputs(args.inputs)]
    total = totals(pages)
    if not args.quiet:
        for p in pages:
            if "error" in p:
                print(f"{p['name']}: {p['size']} bytes: ERROR {p['error']}")
            else:
                print(f"{p['name']}: {p['size']} bytes {p['sequence']} blocks={p['blocks']} groups={p['groups']} "
                      f"meshlets={p['meshlets']} triangles={p['triangles']}")
        print(json.dumps(total, indent=1))
    if args.json:
        try:
            with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
                json.dump({"totals": total, "pages": pages}, fh, indent=1)
        except OSError as exc:
            print(f"error: cannot write {args.json}: {exc.strerror or exc}", file=sys.stderr)
            return 2
    return 1 if total["pages"] != total["pages_parsed"] else 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
