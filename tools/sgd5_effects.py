"""Look up a stored effect/technique pass in the Xbox 1.0.2.1 shaders.rpf.

python -I tools/sgd5_effects.py PATH/TO/shaders.rpf
python -I tools/sgd5_effects.py PATH/TO/shaders.rpf --list-effects
python -I tools/sgd5_effects.py PATH/TO/shaders.rpf --effect NAME
python -I tools/sgd5_effects.py PATH/TO/shaders.rpf --effect NAME --technique NAME

The default JSON contains only aggregate counts. A named query also returns
the database, group, ordinal and archive offsets of the selected programs.
Group 0 contains vertex and mesh programs, 1 pixel, 2 geometry and 5 compute.
Groups 3 and 4 are empty in this build. All returned offsets are absolute.
Use carve_offset to join to shader_carve's index.json by offset. Repeated
effect or technique names return every match with its offset and row index.
This reads plain metadata in place. Runtime selection of a technique is unknown.
"""
import argparse
import json
from pathlib import Path
import re
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shader_carve import ENTRY

# SGD5 signature, first native group count, end of the padded effect region.
DATABASES = ((0x9000, 0x10E39, 0x10287000), (0x10287000, 0x10289C81, 0x10579000))
IDENTIFIER = re.compile(rb"[A-Za-z_][A-Za-z0-9_]{0,254}\x00")
POOL = re.compile(rb"[\x20-\x7e\x00]+")
STRING = re.compile(rb"[^\x00]+\x00")


def u32(data, pos=0):
    if pos + 4 > len(data):
        raise ValueError("truncated SGD5 count")
    return struct.unpack_from("<I", data, pos)[0]


def program_groups(stream, limit):
    groups = []
    for _ in range(6):
        count = u32(stream.read(4))
        offsets = []
        for _ in range(count):
            start = stream.tell()
            length = stream.read(1)
            if not length or length[0] < 2:
                raise ValueError("invalid program label length")
            label = stream.read(length[0])
            if len(label) != length[0] or label[-1:] != b"\x00":
                raise ValueError("truncated program label")
            stream.seek(1, 1)  # tag
            size = u32(stream.read(4))
            if size < 12 or stream.tell() + size + 156 > limit:
                raise ValueError("program crosses the database boundary")
            stream.seek(size + 152, 1)
            trailer = u32(stream.read(4))
            if stream.tell() + trailer > limit:
                raise ValueError("program trailer crosses the database boundary")
            stream.seek(trailer, 1)
            offsets.append(start)
        groups.append(offsets)
    return groups


def program_identity(stream, group, ordinal, start):
    stream.seek(start)
    length = stream.read(1)[0]
    # shader_carve.ENTRY can include a printable native length byte (and up to
    # 200 printable bytes in all). Match the same wire header to recover its key.
    terminator = start + length
    window_start = max(0, terminator - 200)
    stream.seek(window_start)
    window = stream.read(terminator + 18 - window_start)
    pos = terminator - window_start
    while pos and 0x21 <= window[pos - 1] <= 0x7e:
        pos -= 1
    match = ENTRY.fullmatch(window, pos)
    if match is None:
        raise ValueError(f"no shader_carve entry matching program at {start:#x}")
    return {"group": group, "ordinal": ordinal, "record_offset": start, "carve_offset": window_start + pos}


def program_lists(data, pos, groups):
    lists = []
    for group in groups:
        if pos + 4 > len(data):
            return None
        count = u32(data, pos)
        pos += 4
        after = pos + 4 * count
        if after > len(data):
            return None
        ordinals = []
        for (ordinal,) in struct.iter_unpack("<I", memoryview(data)[pos:after]):
            if ordinal >= len(group):
                return None
            ordinals.append(ordinal)
        lists.append(ordinals)
        pos = after
    return lists, pos


def effect_headers(data, groups):
    """Find counted native identifiers and validate all six arrays of database-local ordinals."""
    for match in IDENTIFIER.finditer(data):
        end = match.end()
        # If the prefix itself looks like an identifier character, it is part
        # of the regex match. The length byte decides where the label starts.
        for start in range(max(0, match.start() - 1), end - 2):
            if data[start] != end - start - 1 or not IDENTIFIER.fullmatch(data, start + 1, end):
                continue
            arrays = program_lists(data, end + 4, groups)  # the word following the name is opaque
            if arrays is not None:
                lists, pos = arrays
                yield start, data[start + 1:end - 1].decode("ascii"), lists, pos


def string_pool(data, pos, end):
    # The length excludes its own u32, includes the final NUL, and the pool
    # ends at the next effect boundary (or just before the terminal marker).
    found = []
    for prefix in range(pos, end - 4):
        if (u32(data, prefix) == end - prefix - 4 and data[prefix + 4] == data[end - 1] == 0
                and POOL.fullmatch(data, prefix + 4, end)):
            found.append(prefix)
    if len(found) != 1:
        raise ValueError(f"expected one string pool at effect boundary {end:#x}, found {len(found)}")
    return found[0]


def techniques(data, lists, pos, boundary):
    """Yield the stored name and passes, including zero-program passes."""
    pool = string_pool(data, pos, boundary)
    names = {m.start() - pool - 4: m.group()[:-1].decode("ascii")
             for m in STRING.finditer(data, pool + 4, boundary)}
    pos += 80
    for width in (16, 24, 68, 36):
        if pos + 4 > pool:
            raise ValueError("state table crosses the string pool")
        pos += 4 + width * u32(data, pos)
    if pos + 4 > pool:
        raise ValueError("missing technique count")
    count = u32(data, pos)
    pass_count_at = pos + 4 + 4 * count
    if pass_count_at + 4 > pool:
        raise ValueError("technique rows cross the string pool")
    cursor = pass_count_at + 4
    pass_end = cursor + 24 * u32(data, pass_count_at)
    if pass_end > pool:
        raise ValueError("pass rows cross the string pool")
    for name_bits, _, length in struct.iter_unpack("<HBB", data[pos + 4:pass_count_at]):
        name = names.get(name_bits & 0x7fff)
        if name is None or cursor + 24 * length > pass_end:
            raise ValueError("invalid technique name offset or pass count")
        passes = []
        for _ in range(length):
            programs = []
            for group, selector in enumerate(data[cursor:cursor + 6]):
                if selector > len(lists[group]):
                    raise ValueError("pass selector is outside the effect's program list")
                if selector:
                    programs.append((group, lists[group][selector - 1]))
            passes.append(programs)
            cursor += 24
        yield name, passes
    if cursor != pass_end:
        raise ValueError("technique pass counts do not sum to the table count")


def read_database(stream, layout, effect=None, technique=None, *, list_effects=False):
    database, first_group, end = layout
    stream.seek(database)
    if stream.read(4) != b"SGD5":
        raise ValueError(f"expected Xbox 1.0.2.1 SGD5 database at {database:#x}")
    stream.seek(first_group)
    groups = program_groups(stream, end)
    count = u32(stream.read(4))
    base = stream.tell()
    if base >= end:
        raise ValueError("missing effect region")
    region = stream.read(end - base)
    tail = len(region.rstrip(b"\x00")) - 1
    if len(region) != end - base or tail < 0 or region[tail] != 1 or not 4 <= len(region) - tail <= 4096:
        raise ValueError("truncated effect region or invalid terminal marker")
    data = region[:tail]
    headers = list(effect_headers(data, groups))
    if len(headers) != count or (headers and headers[0][0] != 0) or (not headers and data):
        raise ValueError(f"found {len(headers)} effect headers for {count} declared effects")
    result = {"database": database, "groups": list(map(len, groups)), "effects": count,
              "named_effects": 0, "techniques": 0, "passes": 0, "matches": []}
    ends = [header[0] for header in headers[1:]] + [len(data)]
    for (start, name, lists, pos), boundary in zip(headers, ends):
        if list_effects:
            result["matches"].append({"effect": name, "effect_offset": base + start})
        listing = []
        for index, (stored_name, passes) in enumerate(techniques(data, lists, pos, boundary)):
            result["named_effects"] += index == 0
            result["techniques"] += 1
            result["passes"] += len(passes)
            if name != effect:
                continue
            if technique is None:
                listing.append({"technique": stored_name, "technique_index": index, "pass_count": len(passes)})
            elif stored_name == technique:
                resolved = [[program_identity(stream, g, ordinal, groups[g][ordinal]) for g, ordinal in row]
                            for row in passes]
                result["matches"].append({"effect_offset": base + start, "technique_index": index, "passes": resolved})
        if name == effect and technique is None:
            result["matches"].append({"effect": name, "effect_offset": base + start, "techniques": listing})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", type=Path)
    query = parser.add_mutually_exclusive_group()
    query.add_argument("--list-effects", action="store_true", help="list effect names and archive offsets locally")
    query.add_argument("--effect", help="list techniques for this exact effect name")
    parser.add_argument("--technique", help="exact technique name from that effect")
    args = parser.parse_args(argv)
    if args.technique is not None and args.effect is None:
        parser.error("--technique requires --effect")
    try:
        with args.archive.open("rb") as stream:
            results = [read_database(stream, layout, args.effect, args.technique, list_effects=args.list_effects)
                       for layout in DATABASES]
        if args.effect is not None and not any(row["matches"] for row in results):
            parser.error("no matching effect or technique; use --list-effects, then --effect NAME")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
