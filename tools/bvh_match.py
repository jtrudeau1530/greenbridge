"""Find the drawable geometry a dehydrated BLAS was built from, over `rpf9dump`'s model export.

A dehydrated BLAS (docs/bvh.md §3.1, §4a) keeps, for every leaf, the index of the geometry it came from and the
primitive it uses. That index counts the geometries of one model in the drawable's primary LOD, leaving out the
geometries whose shader the BVH does not keep. So a BLAS is a triangle count per geometry index, and the drawable it
came from has exactly those counts among the geometries of one of its models:

1. read that signature from the BLAS leaves (a leaf holding two primitives counts twice);
2. keep every (drawable, model) of a `rpf9dump models` manifest whose per-geometry triangle counts contain the
   signature as an ordered subsequence - the geometries a model uses are `MeshTris`, grouped by `MeshModels`;
3. for each survivor, read the geometries it points at from the model's OBJ (positions are the decoder's exact
   float32, faces are in primitive order) and check the rehydration against what the BLAS stores with
   tools/bvh_geometry_check.py (bounds, overrides, paired-leaf edges, internal area).

The geometries whose shader the BVH leaves out are game data: pass their name hashes with --skip-shaders (or
BVH_SKIP_SHADERS). When a geometry's shader index is not in the manifest's material list the rule cannot be applied
to that drawable, and every geometry is a candidate; the checks then decide.

The instance transform is the 3x4 at +0x50 of the BLAS's 208-byte instance descriptor (docs/bvh.md §3.2). Without
it, every leaf of a rotated structure falls outside the header bounds and a correct geometry is rejected.

Standard library only. Reads the user's own package and writes nothing by default.

Usage:
    python -I tools/bvh_match.py <dehydrated.bin> <manifest.json> --descriptor <instance208.bin>
                                [--models DIR] [--header-offset N] [--descriptor-offset N]
                                [--transform 12 floats] [--skip-shaders HEX,HEX] [--unordered]
                                [--embeddings N] [--top N] [--names] [--json FILE]
"""
import argparse
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bvh_geometry_check as C  # noqa: E402
import bvh_rehydrate as R  # noqa: E402

INSTANCE_TRANSFORM_OFFSET = 0x50
SKIP_SHADERS_ENV = "BVH_SKIP_SHADERS"


def leaf_signature(src, header_offset=0):
    """Triangles per leaf geometry index; a leaf that holds two primitives counts as two triangles."""
    hdr = R.Header(src, header_offset)
    body = bytes(src[header_offset:header_offset + hdr.total_bytes])   # the leaf arrays count from the header
    per = {}
    for _, geometry, _, _, prim1, _ in R.leaf_records(body, hdr):
        per[geometry] = per.get(geometry, 0) + (1 if prim1 is None else 2)
    return [per.get(i, 0) for i in range(max(per) + 1)] if per else []


def parse_shader_hashes(text):
    """'0x1234abcd,89abcdef' -> frozenset of ints; empty or None -> empty set (the rule is not applied)."""
    return frozenset(int(x, 16) for x in (text or "").replace(" ", "").split(",") if x)


def read_transform(path, at=0):
    """The 3x4 at +0x50 of a 208-byte instance descriptor, as three rows of four floats."""
    with open(path, "rb") as f:
        f.seek(at + INSTANCE_TRANSFORM_OFFSET)
        raw = f.read(48)
    if len(raw) != 48:
        raise ValueError("%s is shorter than a 208-byte instance descriptor at %d" % (path, at))
    values = struct.unpack("<12f", raw)
    return tuple(values[4 * row:4 * row + 4] for row in range(3))


def embeddings(signature, counts, cap, ordered=True):
    """Every way to pick distinct geometries of `counts` matching `signature` (index lists), at most `cap` of them.

    ordered: the picked indices must increase (a prefix is the common case); otherwise any order is allowed.
    """
    out = []

    def walk(i, j, acc):
        if len(out) >= cap:
            return
        if i == len(signature):
            out.append(list(acc))
            return
        for k in (range(j, len(counts)) if ordered else range(len(counts))):
            if counts[k] == signature[i] and k not in acc:
                acc.append(k)
                walk(i + 1, k + 1, acc)
                acc.pop()

    walk(0, 0, [])
    return out


def model_geometries(item):
    """[(mesh index, model, geometry, triangles, shader hash or None)] for one manifest item, in mesh order.

    The mesh index is the geometry's `o` group in the item's OBJ: the manifest arrays and the OBJ groups are
    written in the same order.
    """
    tris = item.get("MeshTris")
    if tris is None:
        raise ValueError("the manifest has no MeshTris: re-export with an rpf9dump that writes per-geometry counts")
    models = item.get("MeshModels") or [0] * len(tris)
    geometries = item.get("MeshGeoms") or list(range(len(tris)))
    shaders = item.get("MeshShaders") or [None] * len(tris)
    materials = item.get("Materials") or []
    hashes = []
    for index in shaders:
        if index is None or not 0 <= index < len(materials) or "Shader" not in materials[index]:
            hashes.append(None)
        else:
            hashes.append(int(materials[index]["Shader"], 16))
    return list(zip(range(len(tris)), models, geometries, tris, hashes))


def match_candidates(items, signature, skip, cap, ordered=True):
    """Every (item, model, used geometry indices, kept geometries) whose triangle counts contain the signature."""
    out = []
    for item in items:
        meshes = model_geometries(item)
        for model in sorted({mesh[1] for mesh in meshes}):
            group = [mesh for mesh in meshes if mesh[1] == model]
            # A geometry whose shader is one of the excluded hashes is not in the BLAS, so it takes no index in the
            # signature. Without every hash of the group the rule cannot be applied: keep all of them.
            rule = bool(skip) and all(mesh[4] is not None for mesh in group)
            kept = [mesh for mesh in group if not rule or mesh[4] not in skip]
            for used in embeddings(signature, [mesh[3] for mesh in kept], cap, ordered):
                out.append({"item": item, "model": model, "used": used, "kept": kept,
                            "rule": rule, "skipped": bool(skip)})
    return out


def read_groups(path):
    """[(positions, faces)] per `o` group of an OBJ, in file order; faces are 0-based indices into that group.

    tools/rpf9dump reads `RageModelDecoder.CreateObj` output: one group per geometry, its vertices in buffer order
    and its faces in primitive order (docs/bvh.md §4a: the leaves address triangles in primitive order).
    """
    groups, positions, faces, base, started = [], [], [], 0, False

    def flush():
        nonlocal positions, faces, base, started
        if started or positions or faces:
            groups.append((positions, faces))
            base += len(positions)
        positions, faces, started = [], [], False

    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("ascii", "replace").strip()
            if len(line) < 2:
                continue
            tag, _, rest = line.partition(" ")
            if tag == "o":
                flush()
                started = True
            elif tag == "v":
                x, y, z = rest.split()[:3]
                positions.append((float(x), float(y), float(z)))
            elif tag == "f":
                corners = rest.split()
                if len(corners) != 3:
                    raise ValueError("%s: a face with %d corners; only triangles are expected" % (path, len(corners)))
                faces.append(tuple(int(corner.split("/")[0]) - 1 - base for corner in corners))
    flush()
    return groups


def build_geometry(group, transform=None):
    """A bvh_rehydrate.Geometry from one OBJ group: float3 vertices, uint32 triangles, the instance transform."""
    positions, faces = group
    if not faces:
        raise ValueError("a geometry without triangles cannot be an index of a BLAS leaf")
    for face in faces:
        if min(face) < 0 or max(face) >= len(positions):
            raise ValueError("an OBJ face refers to a vertex outside its group")
    vb = b"".join(struct.pack("<3f", *p) for p in positions)
    ib = b"".join(struct.pack("<3I", *face) for face in faces)
    return R.Geometry(vb, 12, ib, False, transform)


def score(src, candidate, groups, transform, header_offset=0):
    """Rehydrate the candidate's geometries and run bvh_geometry_check over them."""
    geometries = [build_geometry(groups[mesh[0]], transform) for mesh in
                  [candidate["kept"][k] for k in candidate["used"]]]
    return C.check(src, geometries, header_offset)


def describe(candidate, names):
    item = candidate["item"]
    used = [candidate["kept"][k][2] for k in candidate["used"]]
    label = ("%-40s " % item["Name"]) if names else ""
    note = " (shader rule not applied)" if candidate["skipped"] and not candidate["rule"] else ""
    return "%sstream 0x%-10x model %d geometry %s%s" % (label, item["Offset"], candidate["model"], used, note)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Find the drawable geometry a dehydrated BLAS was built from.")
    ap.add_argument("dehydrated")
    ap.add_argument("manifest")
    ap.add_argument("--models", help="directory of the OBJs (default: the manifest's directory)")
    ap.add_argument("--descriptor", help="208-byte instance descriptor; its 3x4 at +0x50 is the geometry transform")
    ap.add_argument("--descriptor-offset", type=int, default=0)
    ap.add_argument("--transform", help="the same 3x4 as 12 comma-separated floats, instead of --descriptor")
    ap.add_argument("--header-offset", type=int, default=0, help="the 64-byte header's offset in the file")
    ap.add_argument("--skip-shaders", default=os.environ.get(SKIP_SHADERS_ENV, ""),
                    help="shader name hashes of the geometries the BVH leaves out (game data; local use only)")
    ap.add_argument("--unordered", action="store_true", help="also try the geometries in any order")
    ap.add_argument("--embeddings", type=int, default=64, help="at most this many index choices per model")
    ap.add_argument("--top", type=int, default=8, help="how many scored candidates to print")
    ap.add_argument("--names", action="store_true", help="print drawable names (game data: local use only)")
    ap.add_argument("--json", help="write the report here")
    a = ap.parse_args(argv)
    try:
        with open(a.dehydrated, "rb") as f:
            src = f.read()
        with open(a.manifest, encoding="utf-8") as f:
            manifest = json.load(f)
        items = manifest.get("Items") or []
        if not items:
            raise ValueError("%s has no Items" % a.manifest)
        models_dir = a.models or os.path.dirname(os.path.abspath(a.manifest))
        transform = None
        if a.transform:
            transform = R.parse_transform(a.transform)
        elif a.descriptor:
            transform = read_transform(a.descriptor, a.descriptor_offset)
        signature = leaf_signature(src, a.header_offset)
        if not signature:
            raise ValueError("the BLAS has no leaves")
        candidates = match_candidates(items, signature, parse_shader_hashes(a.skip_shaders), a.embeddings, not a.unordered)
    except (OSError, ValueError, KeyError, struct.error) as e:
        print("error: %s" % e, file=sys.stderr)
        return 2
    if transform is None:
        print("no --descriptor or --transform: the geometry is checked without the instance transform", file=sys.stderr)
    print("triangles per geometry index: %s" % signature)
    print("%d candidates from %d models in the manifest" % (len(candidates), len(items)))
    results, groups, errors = [], {}, 0
    for candidate in candidates:
        item = candidate["item"]
        path = os.path.join(models_dir, item["File"])
        try:
            if path not in groups:
                groups[path] = read_groups(path)
            report = score(src, candidate, groups[path], transform, a.header_offset)
        except (OSError, ValueError, KeyError, struct.error) as e:
            errors += 1
            print("   %s: %s" % (describe(candidate, a.names), e))
            continue
        results.append((not report["consistent"], report["internal_area"], report, candidate))
    results.sort(key=lambda r: (r[0], r[1]))
    best = next((r[1] for r in results if not r[0]), None)
    for bad, area, report, candidate in results[:max(1, a.top)]:
        ratio = "" if best is None or not best else " (x%.3f)" % (area / best)
        print("   %-10s %s | outside %d of %d, gap %.1e, overrides %d/%d, area %.6g%s" % (
            "consistent" if not bad else "rejected", describe(candidate, a.names), report["leaves_outside"], report["leaves"],
            report["bounds_gap"], report["overrides_consistent"], report["overrides_inconsistent"], area, ratio))
    consistent = [r for r in results if not r[0]]
    print("%d of %d scored candidates are consistent%s" % (len(consistent), len(results), "" if not errors else " (%d could not be read)" % errors))
    if len(consistent) > 1:
        print("more than one candidate is consistent: the signature and the checks do not decide between them; "
              "the internal box area (the smallest one) is the tie-break")
    if a.json:
        match = consistent[0][3] if consistent else None
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"signature": signature, "consistent": bool(consistent), "scored": len(results),
                       "matched": None if match is None else {
                           "offset": match["item"]["Offset"], "model": match["model"],
                           "geometry": [match["kept"][k][2] for k in match["used"]],
                           "triangles": [match["kept"][k][3] for k in match["used"]],
                           "name": match["item"]["Name"] if a.names else None}},
                      f, indent=1)
    return 0 if consistent else 1


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
