"""Tests for tools/world_census.py, built from synthetic 0x7a resources and RPF9 streams (no game data).

    python -I tests/test_world_census.py
"""
import importlib.util
import os
import struct
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_tool():
    path = os.path.join(ROOT, "tools", "world_census.py")
    spec = importlib.util.spec_from_file_location("world_census_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WC = load_tool()
WI = WC.WI
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
    struct.pack_into("<I", b, 0x20, 1)
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
    b[e + 0x7E] = 1

    if map_file:
        mm = [
            member(WI.FIELD["name"], 0x08, 0x4A),
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


class CensusTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_inflate_keeps_a_map_header(self):
        resource = make_resource()
        blob = zlib.compress(resource)
        data, used, eof, is_meta, inflated, truncated = WC.inflate(blob, 0, len(blob), keep=True)
        self.assertEqual(data, resource)
        self.assertEqual(used, len(blob))
        self.assertTrue(eof)
        self.assertTrue(is_meta)
        self.assertEqual(inflated, len(resource))
        self.assertFalse(truncated)

    def test_assemble_joins_a_split_resource(self):
        resource = make_resource()
        blob = make_rpf_streams([resource[:0x150], resource[0x150:]])
        res = WC.assemble(blob, 0x40, len(blob))
        self.assertIsNotNone(res)
        self.assertEqual(res["data"], resource)
        self.assertTrue(res["joined"])
        self.assertTrue(res["complete"])
        self.assertEqual(res["streams"], 2)

    def test_shard_and_summary_count_one_map(self):
        resource = make_resource()
        archive = os.path.join(self.dir, "sample.rpf")
        with open(archive, "wb") as f:
            f.write(make_rpf_streams([resource]))
        with open(archive, "rb") as f:
            data_start = struct.unpack_from("<I", f.read(8), 4)[0]
        size = os.path.getsize(archive)
        out = os.path.join(self.dir, "census")
        os.makedirs(out)
        result = WC.scan_shard((archive, 0, data_start, size, out, data_start))
        self.assertEqual((result["maps"], result["entities"]), (1, 1))
        summary = WC.summarize(archive, out, data_start, size, size - data_start, data_start)
        self.assertTrue(summary["complete"])
        self.assertTrue(summary["maps_exact"])
        self.assertTrue(summary["entities_exact"])
        self.assertEqual((summary["maps"], summary["entities"]), (1, 1))
        self.assertEqual(summary["totals"]["entity_unresolved"], 0)
        self.assertEqual(summary["coverage"]["raw_maps"], 0)
        self.assertEqual(summary["coverage"]["missed_zlib_meta"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
