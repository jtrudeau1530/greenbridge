"""Tests for tools/world_index.py, built from synthetic 0x7a headers (no game data).

    python -I tests/test_world_index.py
"""
import contextlib
import csv
import importlib.util
import io
import json
import os
import struct
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_tool():
    path = os.path.join(ROOT, "tools", "world_index.py")
    spec = importlib.util.spec_from_file_location("world_index_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WI = load_tool()
PTR = 0x50000000


def ptr(off):
    return PTR | off


def member(hash_, off, ty, sub=0, ref=0, extra=0):
    return struct.pack("<IIBBHI", hash_, off, ty, sub, ref, extra)


def struct_def(hash_, size, members, at, member_at):
    return struct.pack("<IIIIQIHH", hash_, 0, 0, 0, ptr(member_at), size, 0, len(members))


def make_resource(map_file=True):
    """A minimal, valid 0x7a resource with one entity and (optionally) a map root."""
    b = bytearray(0x700)
    b[0x10:0x14] = b"0DRP"
    b[0x14] = 0x7A
    struct.pack_into("<I", b, 0x20, 1)       # root objects
    struct.pack_into("<Q", b, 0x28, ptr(0x100))
    struct.pack_into("<Q", b, 0x38, ptr(0x200))
    nst = 2 if map_file else 1
    nbl = 2 if map_file else 1
    struct.pack_into("<HHHH", b, 0x48, nst, 0, nbl, 0)

    members = [
        member(WI.FIELD["archetypeName"], 0x44, 0x4A),
        member(WI.FIELD["position"], 0x38, 0x32),
        member(WI.FIELD["rotation"], 0x10, 0x34),
        member(WI.FIELD["scaleXY"], 0x54, 0x21),
        member(WI.FIELD["scaleZ"], 0x58, 0x21),
        member(WI.FIELD["flags"], 0x48, 0x15),
        member(WI.FIELD["guid"], 0x20, 0x4F),
        member(WI.FIELD["lodLevel"], 0x7E, 0x60),
        member(WI.FIELD["lodDist"], 0x60, 0x21),
        member(WI.ARCHETYPE_KEY, 0x20, 0x4F),
    ]
    entity_members = 0x300
    for i, m in enumerate(members):
        b[entity_members + 16 * i:entity_members + 16 * (i + 1)] = m
    b[0x100:0x120] = struct_def(WI.ENTITY, 0x90, members, 0x100, entity_members)
    struct.pack_into("<IIQ", b, 0x200, WI.ENTITY, 0x90, ptr(0x400))

    e = 0x400
    struct.pack_into("<4f", b, e + 0x10, 0.0, 0.0, 0.0, 1.0)
    struct.pack_into("<Q", b, e + 0x20, 0x1122334455667788)
    struct.pack_into("<3f", b, e + 0x38, 1.5, 2.5, 3.5)
    struct.pack_into("<I", b, e + 0x44, 0xDEADBEEF)
    struct.pack_into("<I", b, e + 0x48, 5)
    struct.pack_into("<2f", b, e + 0x54, 1.0, 1.0)
    struct.pack_into("<f", b, e + 0x60, 250.0)
    b[e + 0x7E] = 1  # LOD

    if map_file:
        mm = [
            member(WI.FIELD["name"], 0x08, 0x4A),
            member(WI.FIELD["name64"], 0x10, 0x4F),
            member(WI.FIELD["parent"], 0x68, 0x4A),
            member(WI.FIELD["streamingExtentsMin"], 0x70, 0x32),
            member(WI.FIELD["streamingExtentsMax"], 0x7C, 0x32),
            member(WI.FIELD["flags_map"], 0x88, 0x15),
            member(WI.FIELD["contentFlags"], 0x8C, 0x15),
        ]
        map_members = 0x3A0
        for i, m in enumerate(mm):
            b[map_members + 16 * i:map_members + 16 * (i + 1)] = m
        b[0x120:0x140] = struct_def(WI.MAPDATA, 0x120, mm, 0x120, map_members)
        struct.pack_into("<IIQ", b, 0x210, WI.MAPDATA, 0x120, ptr(0x500))
        root = 0x500
        struct.pack_into("<I", b, root + 0x08, 0x11223344)
        struct.pack_into("<Q", b, root + 0x10, 0x1122334455667788)
        struct.pack_into("<I", b, root + 0x68, 0xAABBCCDD)
        struct.pack_into("<3f", b, root + 0x70, -10.0, -20.0, -30.0)
        struct.pack_into("<3f", b, root + 0x7C, 10.0, 20.0, 30.0)
        struct.pack_into("<I", b, root + 0x88, 2)
        struct.pack_into("<I", b, root + 0x8C, 65)
    return bytes(b)


def make_rpf_streams(parts, start=0x40):
    """A minimal RPF9-looking file: header word + compressed parts, 16-byte aligned."""
    out = bytearray(start)
    out[:4] = b"9FPR"
    struct.pack_into("<I", out, 4, start)
    for part in parts:
        pos = (len(out) + 15) & ~15
        out.extend(b"\0" * (pos - len(out)))
        out.extend(zlib.compress(part))
    return bytes(out)


class WorldIndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_map_resource_fields(self):
        row = WI.parse_resource(make_resource())
        self.assertIsNotNone(row)
        self.assertEqual(row["kind"], "map")
        self.assertEqual(row["version"], 122)
        self.assertEqual(row["entities"], 1)
        self.assertEqual(row["entity_size"], 0x90)
        self.assertEqual(row["archetypes"], 1)
        self.assertEqual(row["map_name"], 0x11223344)
        self.assertEqual(row["map_name64"], 0x1122334455667788)
        self.assertEqual(row["parent"], 0xAABBCCDD)
        self.assertEqual(row["extent_min"], [-10.0, -20.0, -30.0])
        self.assertEqual(row["extent_max"], [10.0, 20.0, 30.0])
        self.assertEqual(row["position_min"], [1.5, 2.5, 3.5])
        self.assertEqual(row["lod"], {"LOD": 1})
        self.assertEqual(row["lod_dist_min"], 250.0)

    def test_non_map_resource_is_meta(self):
        row = WI.parse_resource(make_resource(map_file=False))
        self.assertIsNotNone(row)
        self.assertEqual(row["kind"], "meta")
        self.assertNotIn("entities", row)

    def test_bad_version_is_rejected(self):
        b = bytearray(make_resource())
        b[0x14] = 0x79
        self.assertIsNone(WI.parse_resource(bytes(b)))

    def test_bad_table_pointer_is_rejected(self):
        b = bytearray(make_resource())
        struct.pack_into("<Q", b, 0x28, 0x60000000)
        self.assertIsNone(WI.parse_resource(bytes(b)))

    def test_split_resource_needs_continuation(self):
        self.assertFalse(WI._needs_continuation(make_resource()))
        self.assertTrue(WI._needs_continuation(make_resource()[:0x150]))

    def test_rpf_scan_joins_a_split_stream(self):
        blob = make_resource()
        path = os.path.join(self.dir, "sample.rpf")
        with open(path, "wb") as f:
            f.write(make_rpf_streams([blob[:0x150], blob[0x150:]]))
        rows, stats = WI.scan_rpf(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(stats["streams"], 2)
        self.assertEqual(rows[0]["kind"], "map")
        self.assertEqual(rows[0]["entities"], 1)

    def test_scan_window_options(self):
        blob = make_resource()
        path = os.path.join(self.dir, "sample.rpf")
        with open(path, "wb") as f:
            f.write(make_rpf_streams([blob]))
        size = os.path.getsize(path)
        out = os.path.join(self.dir, "out.json")
        # A one-byte window at the data start: the scan ends one byte later.
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(WI.main([path, "--out", out, "--start", "64", "--limit", "1"]), 0)
        with open(out, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["summary"]["scan_end"], 0x41)
        os.remove(out)
        # --limit 0 used to read as "no limit" (the whole archive); negative values were used as given.
        for args in (["--limit", "0"], ["--limit", "-16"], ["--start", "-1"],
                     ["--max-inflate", "0"], ["--max-inflate", "-1"]):
            with self.subTest(args=args), self.assertRaises(SystemExit) as ctx, \
                    contextlib.redirect_stderr(io.StringIO()):
                WI.main([path, "--out", out, *args])
            self.assertEqual(ctx.exception.code, 2)
        # A start inside the RPF9 header or past the end is refused, and no report is written.
        for start in ("16", str(size)):
            with self.subTest(start=start):
                err = io.StringIO()
                with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(WI.main([path, "--out", out, "--start", start]), 2)
                self.assertIn("outside the data area", err.getvalue())
        self.assertFalse(os.path.exists(out))

    def test_scan_limit_excludes_crossing_stream_and_accepts_hex(self):
        blob = make_resource()
        stored = len(zlib.compress(blob))
        path = os.path.join(self.dir, "sample.rpf")
        with open(path, "wb") as f:
            f.write(make_rpf_streams([blob]))
        out = os.path.join(self.dir, "out.json")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(WI.main([path, "--out", out, "--start", "0x40", "--limit", hex(stored)]), 0)
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["resources"], 1)
        self.assertEqual(doc["summary"]["scanned_bytes"], stored)

        rows, stats = WI.scan_rpf(path, start=0x40, limit=stored - 1)
        self.assertEqual(rows, [])
        self.assertEqual(stats["streams"], 0)
        self.assertEqual(stats["scanned_bytes"], 0)
        self.assertEqual(stats["scan_end"], 0x40 + stored - 1)

    def test_missing_path_is_an_error_not_a_traceback(self):
        out = os.path.join(self.dir, "missing.json")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(WI.main([os.path.join(self.dir, "nope.rpf"), "--out", out]), 2)
        self.assertIn("ERROR:", err.getvalue())
        self.assertFalse(os.path.exists(out))

    def test_rpf_scan_survives_a_damaged_candidate_continuation(self):
        blob = make_resource()
        partial = blob[:0x150]
        unrelated = bytes(72)
        archive = bytearray(make_rpf_streams([partial, unrelated, blob]))
        continuation = (0x40 + len(zlib.compress(partial)) + 15) & ~15
        end = continuation + len(zlib.compress(unrelated))
        archive[end - 1] ^= 1  # Corrupt only the second stream's checksum.
        path = os.path.join(self.dir, "damaged_continuation.rpf")
        with open(path, "wb") as f:
            f.write(archive)
        rows, stats = WI.scan_rpf(path)
        self.assertEqual(stats["streams"], 2)
        self.assertEqual(rows[-1]["offset"], (end + 15) & ~15)
        self.assertEqual(rows[-1]["size"], len(blob))

    def assert_report_collision(self, args, message):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            WI.main(args)
        self.assertEqual(cm.exception.code, 2)
        self.assertIn(message, err.getvalue())

    def test_reports_cannot_overwrite_a_single_input(self):
        resource = make_resource()
        for name, content in (("resource.bin", resource), ("archive.rpf", make_rpf_streams([resource]))):
            path = os.path.join(self.dir, name)
            for flag in ("--out", "--csv", "--archetypes-out", "--entities-out"):
                with self.subTest(name=name, flag=flag):
                    with open(path, "wb") as f:
                        f.write(content)
                    out = os.path.join(self.dir, "summary.json")
                    args = [path, flag, path]
                    if flag != "--out":
                        args.extend(["--out", out])
                    self.assert_report_collision(args, f"{flag} would overwrite the input")
                    with open(path, "rb") as f:
                        self.assertEqual(f.read(), content)
                    self.assertFalse(os.path.exists(out))

    def test_entities_report_cannot_create_the_missing_input(self):
        path = os.path.join(self.dir, "missing.bin")
        out = os.path.join(self.dir, "summary.json")
        self.assert_report_collision(
            [path, "--out", out, "--entities-out", path], "--entities-out would overwrite the input")
        self.assertFalse(os.path.exists(path))
        self.assertFalse(os.path.exists(out))

    def test_report_cannot_overwrite_a_nested_input_through_a_hard_link(self):
        carve = os.path.join(self.dir, "carve")
        nested = os.path.join(carve, "nested")
        os.makedirs(nested)
        resource = make_resource()
        path = os.path.join(nested, "resource.bin")
        with open(path, "wb") as f:
            f.write(resource)
        report = os.path.join(self.dir, "entities.jsonl")
        try:
            os.link(path, report)
        except (OSError, NotImplementedError) as e:
            self.skipTest(f"hard links unavailable: {e}")
        out = os.path.join(self.dir, "summary.json")
        self.assert_report_collision(
            [carve, "--out", out, "--entities-out", report], "--entities-out would overwrite the input")
        with open(path, "rb") as f:
            self.assertEqual(f.read(), resource)
        self.assertFalse(os.path.exists(out))

    def test_report_alias_is_rejected_before_opening_any_output(self):
        path = os.path.join(self.dir, "resource.bin")
        with open(path, "wb") as f:
            f.write(make_resource())
        out = os.path.join(self.dir, "report")
        alias = os.path.join(self.dir, ".", "report")
        entities = os.path.join(self.dir, "new", "entities.jsonl")
        self.assert_report_collision(
            [path, "--out", out, "--csv", alias, "--entities-out", entities],
            "--out and --csv must use different paths")
        self.assertFalse(os.path.exists(out))
        self.assertFalse(os.path.exists(os.path.dirname(entities)))

    def test_reports_cannot_alias_through_hard_links(self):
        path = os.path.join(self.dir, "resource.bin")
        with open(path, "wb") as f:
            f.write(make_resource())
        out = os.path.join(self.dir, "summary.json")
        entities = os.path.join(self.dir, "entities.jsonl")
        with open(out, "wb") as f:
            f.write(b"previous report")
        try:
            os.link(out, entities)
        except (OSError, NotImplementedError) as e:
            self.skipTest(f"hard links unavailable: {e}")
        self.assert_report_collision(
            [path, "--out", out, "--entities-out", entities], "--out and --entities-out must use different paths")
        with open(out, "rb") as f:
            self.assertEqual(f.read(), b"previous report")

    def test_failed_scan_keeps_existing_placement_report(self):
        archive = os.path.join(self.dir, "archive.rpf")
        resource = make_resource()
        with open(archive, "wb") as f:
            f.write(make_rpf_streams([resource]))
        carve = os.path.join(self.dir, "carve")
        os.makedirs(carve)
        with open(os.path.join(carve, "01-resource.bin"), "wb") as f:
            f.write(resource)
        with open(os.path.join(carve, "02-archive.rpf"), "wb") as f:
            f.write(make_rpf_streams([resource]))

        outputs = os.path.join(self.dir, "outputs")
        os.makedirs(outputs)
        summary = os.path.join(outputs, "summary.json")
        entities = os.path.join(outputs, "entities.jsonl")
        previous = b'{"previous":true}\n'
        for source, arguments, diagnostic in (
            (os.path.join(self.dir, "missing.rpf"), [], "missing.rpf"),
            (archive, ["--start", "0"], "outside the data area"),
            # The carved resource emits one row before the archive rejects this window.
            (carve, ["--start", "0"], "outside the data area"),
        ):
            with self.subTest(source=source):
                with open(entities, "wb") as f:
                    f.write(previous)
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    self.assertEqual(WI.main([source, "--out", summary, "--entities-out", entities, *arguments]), 2)
                self.assertIn(diagnostic, err.getvalue())
                self.assertEqual(os.listdir(outputs), ["entities.jsonl"])
                with open(entities, "rb") as f:
                    self.assertEqual(f.read(), previous)

    def test_summary_write_failure_keeps_existing_placement_report(self):
        source = os.path.join(self.dir, "resource.bin")
        with open(source, "wb") as f:
            f.write(make_resource())
        entities = os.path.join(self.dir, "entities.jsonl")
        with open(entities, "wb") as f:
            f.write(b"previous report")
        summary = os.path.join(self.dir, "summary.json")
        os.makedirs(summary)

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(WI.main([source, "--out", summary, "--entities-out", entities]), 2)
        self.assertIn("ERROR:", err.getvalue())
        self.assertCountEqual(os.listdir(self.dir), ["resource.bin", "summary.json", "entities.jsonl"])
        with open(entities, "rb") as f:
            self.assertEqual(f.read(), b"previous report")

    def test_success_replaces_existing_placement_report(self):
        source = os.path.join(self.dir, "resource.bin")
        with open(source, "wb") as f:
            f.write(make_resource())
        entities = os.path.join(self.dir, "entities.jsonl")
        with open(entities, "wb") as f:
            f.write(b"previous report")
        if os.name != "nt":
            os.chmod(entities, 0o640)
        previous_mode = os.stat(entities).st_mode
        summary = os.path.join(self.dir, "summary.json")

        self.assertEqual(WI.main([source, "--out", summary, "--entities-out", entities]), 0)
        self.assertCountEqual(os.listdir(self.dir), ["resource.bin", "summary.json", "entities.jsonl"])
        self.assertEqual(os.stat(entities).st_mode, previous_mode)
        with open(entities, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        self.assertEqual([row["archetype"] for row in rows], ["deadbeef"])

    def test_directory_scan_and_outputs(self):
        with open(os.path.join(self.dir, "map.bin"), "wb") as f:
            f.write(make_resource())
        with open(os.path.join(self.dir, "meta.bin"), "wb") as f:
            f.write(make_resource(map_file=False))
        rows, stats = WI.scan_path(self.dir)
        self.assertEqual(len(rows), 2)
        self.assertEqual(stats["files"], 2)
        self.assertEqual(sum(1 for r in rows if r["kind"] == "map"), 1)
        self.assertEqual(sum(r.get("entities") or 0 for r in rows), 1)

        out = os.path.join(self.dir, "out.json")
        csv_out = os.path.join(self.dir, "out.csv")
        entities = os.path.join(self.dir, "entities.jsonl")
        self.assertEqual(WI.main([self.dir, "--out", out, "--csv", csv_out, "--entities-out", entities]), 0)
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["files"], 2)
        self.assertEqual(doc["summary"]["resources"], 2)
        with open(entities, encoding="utf-8") as f:
            self.assertEqual(len([json.loads(line) for line in f]), 1)
        self.assertEqual(doc["summary"]["maps"], 1)
        self.assertEqual(doc["summary"]["entities"], 1)
        with open(csv_out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["signature"], "0DRPz")
        self.assertEqual(rows[0]["version"], "122")

    def test_directory_inflate_cap_does_not_exclude_large_archives(self):
        resource = make_resource()
        cap = len(resource)
        with open(os.path.join(self.dir, "large.rpf"), "wb") as f:
            f.write(make_rpf_streams([resource]) + bytes(cap))
        with open(os.path.join(self.dir, "oversized.bin"), "wb") as f:
            f.write(resource + b"\0")
        rows, stats = WI.scan_path(self.dir, max_inflate=cap)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["kind"], "map")
        self.assertEqual(stats["files"], 1)

    def test_archetypes_out_counts_each_entity_once(self):
        blob = make_resource()
        path = os.path.join(self.dir, "sample.rpf")
        with open(path, "wb") as f:
            # One resource split over two streams (joined, parsed again) and one whole copy.
            f.write(make_rpf_streams([blob[:0x150], blob[0x150:], blob, make_resource(map_file=False)]))
        out = os.path.join(self.dir, "out.json")
        arch = os.path.join(self.dir, "archetypes.json")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(WI.main([path, "--out", out, "--archetypes-out", arch]), 0)
        with open(arch, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["archetypes"], {"deadbeef": 2})
        self.assertEqual(doc["keys"], {"deadbeef": "1122334455667788"})
        self.assertEqual(doc["summary"], {"entities": 2, "archetypes": 1, "with_key": 1, "key_conflicts": 0,
                                          "incomplete_entities": 0})

    def test_conflicting_keys_leave_the_hash_unverified(self):
        agg = WI.ArchetypeCounts()
        agg.append({"complete": True, "archetype": 1, "archetype_key": 5})
        agg.append({"complete": True, "archetype": 1, "archetype_key": 6})
        agg.append({"complete": True, "archetype": 2, "archetype_key": 7})
        agg.append({"complete": True, "archetype": None, "archetype_key": 8})
        doc = agg.to_json()
        self.assertEqual(doc["archetypes"], {"00000001": 2, "00000002": 1})
        self.assertEqual(doc["keys"], {"00000002": "0000000000000007"})
        self.assertEqual(doc["conflicting_keys"], {"00000001": ["0000000000000005", "0000000000000006"]})
        self.assertEqual(doc["summary"]["key_conflicts"], 1)

    def test_entity_past_the_inflated_bytes_is_not_aggregated(self):
        # The entity block starts at 0x400 (0x90 bytes); the bytes end at 0x448, just after archetypeName.
        agg = WI.ArchetypeCounts()
        row = WI.parse_resource(make_resource()[:0x448], agg)
        self.assertIsNotNone(row)
        self.assertEqual((agg.counts, agg.incomplete), ({}, 1))

    def test_entity_block_starting_past_the_inflated_bytes_is_incomplete(self):
        # The entity block is moved to 0x700, after the bytes kept; its entities must still be reported.
        b = bytearray(make_resource()) + bytearray(0x90)
        struct.pack_into("<IIQ", b, 0x200, WI.ENTITY, 0x90, ptr(0x700))
        b[0x700:0x790] = b[0x400:0x490]
        full = WI.ArchetypeCounts()
        WI.parse_resource(bytes(b), full)
        self.assertEqual((full.counts, full.incomplete), ({0xDEADBEEF: 1}, 0))
        for cut in (0x710, 0x700):
            with self.subTest(cut=cut):
                agg = WI.ArchetypeCounts()
                self.assertIsNotNone(WI.parse_resource(bytes(b[:cut]), agg))
                self.assertEqual((agg.counts, agg.incomplete), ({}, 1))

    def test_archetypes_out_is_a_world_names_input(self):
        spec = importlib.util.spec_from_file_location("world_names_for_index", os.path.join(ROOT, "tools", "world_names.py"))
        WN = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(WN)
        agg = WI.ArchetypeCounts()
        WI.parse_resource(make_resource(), agg)
        stats = WN.resolve_archetypes(agg.to_json(), WN.NameDictionary())
        self.assertEqual((stats["entities"], stats["unique_hashes"], stats["unresolved_hashes"]), (1, 1, 1))


def setUpModule():
    # The tools' main() prints its summary JSON; keep it out of the test log (tests that check it redirect again).
    quiet = contextlib.redirect_stdout(io.StringIO())
    quiet.__enter__()
    unittest.addModuleCleanup(quiet.__exit__, None, None, None)


if __name__ == "__main__":
    unittest.main()
