import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
TOOL = os.path.join(ROOT, "tools", "rootsig_census.py")
import rootsig_census as R


def container(chunks):
    """A minimal DXBC container: magic, 24 zero bytes, chunk count at 28, offsets at 32."""
    head = bytearray(b"DXBC" + b"\0" * 24)
    head += struct.pack("<I", len(chunks))
    offset = 32 + 4 * len(chunks)
    offsets, body = [], bytearray()
    for tag, payload in chunks:
        offsets.append(offset + len(body))
        body += tag + struct.pack("<I", len(payload)) + payload
    head += b"".join(struct.pack("<I", o) for o in offsets)
    return bytes(head + body)


def rts0(version=2, params=(), samplers=0, flags=0):
    """An RTS0 payload. params: ("table", vis, nranges) | ("constants", vis, count) | ("cbv"|"srv"|"uav", vis)."""
    v11 = version >= 2
    base = 24 + 12 * len(params)
    entries, body, tables = bytearray(), bytearray(), []

    def push(data):
        off = base + len(body)
        body.extend(data)
        return off

    for param in params:
        kind, vis = param[0], param[1]
        if kind == "table":
            off = push(struct.pack("<2I", param[2], 0))
            tables.append((off - base, param[2]))
            entries += struct.pack("<3I", 0, vis, off)
        elif kind == "constants":
            off = push(struct.pack("<4I", 0, 0, param[2], 0))
            entries += struct.pack("<3I", 1, vis, off)
        else:
            dtype = {"cbv": 2, "srv": 3, "uav": 4}[kind]
            off = push(struct.pack("<3I", 0, 0, 0) if v11 else struct.pack("<2I", 0, 0))
            entries += struct.pack("<3I", dtype, vis, off)
    for at, nranges in tables:
        if not nranges:
            continue
        ranges_off = base + len(body)
        for j in range(nranges):
            body += struct.pack("<6I", 0, 1, 0, 0, 0, j) if v11 else struct.pack("<5I", 0, 1, 0, 0, j)
        struct.pack_into("<I", body, at + 4, ranges_off)
    samplers_off = base + len(body) if samplers else 0
    body += b"\0" * (((14 if version == 3 else 13) * 4) * samplers)
    return struct.pack("<6I", version, len(params), 24, samplers, samplers_off, flags) + bytes(entries) + bytes(body)


class TestRts0(unittest.TestCase):
    def test_tables_and_root_descriptors(self):
        sig, err = R.parse_rts0(rts0(params=[("table", 0, 1), ("cbv", 0), ("uav", 0)]))
        self.assertEqual(err, "")
        self.assertEqual(sig["version"], "1.1")
        self.assertEqual([p["type"] for p in sig["params"]], ["table", "cbv", "uav"])
        self.assertEqual(sig["params"][0]["ranges"], 1)
        self.assertEqual(R.span(sig, 2), 6)
        self.assertEqual(R.span(sig, 1), 5)

    def test_constants_take_their_count(self):
        sig, err = R.parse_rts0(rts0(params=[("constants", 0, 4), ("cbv", 0)]))
        self.assertEqual(err, "")
        self.assertEqual(sig["params"][0]["count"], 4)
        self.assertEqual(R.span(sig, 2), 6)
        self.assertEqual(R.span(sig, 1), 6)

    def test_version_1_0_has_20_byte_ranges(self):
        sig, err = R.parse_rts0(rts0(version=1, params=[("table", 0, 2), ("cbv", 0)]))
        self.assertEqual(err, "")
        self.assertEqual(sig["version"], "1.0")
        self.assertEqual(sig["params"][0]["ranges"], 2)
        self.assertEqual(R.span(sig, 1), 3)

    def test_version_1_2_samplers_carry_flags(self):
        sig, err = R.parse_rts0(rts0(version=3, params=[("table", 0, 0)], samplers=2, flags=0x03000000))
        self.assertEqual(err, "")
        self.assertEqual(sig["version"], "1.2")
        self.assertEqual(sig["flags"], 0x03000000)
        self.assertEqual(len(sig["static_samplers"]), 2)

    def test_stage_visibility_is_kept(self):
        sig, err = R.parse_rts0(rts0(params=[("table", 5, 1), ("cbv", 0)]))
        self.assertEqual(err, "")
        self.assertEqual(sig["params"][0]["visibility"], 5)
        self.assertEqual(R.span(sig, 2), 4)
        self.assertEqual(R.span(sig, 2, visibility=0), 2)

    def test_truncated_payload_is_an_error(self):
        sig, err = R.parse_rts0(rts0(params=[("table", 0, 1)])[:10])
        self.assertIsNone(sig)
        self.assertIn("truncated", err)

    def test_truncated_range_table_is_an_error(self):
        payload = rts0(params=[("table", 0, 3)])
        sig, err = R.parse_rts0(payload[:-40])
        self.assertIsNone(sig)
        self.assertIn("truncated", err)

    def test_unknown_version_is_an_error(self):
        sig, err = R.parse_rts0(rts0(version=9))
        self.assertIsNone(sig)
        self.assertIn("version", err)


class TestContainer(unittest.TestCase):
    def test_parts_finds_rts0(self):
        payload = rts0(params=[("cbv", 0)])
        self.assertEqual(R.parts(container([(b"PSV0", b"pv"), (b"RTS0", payload)])),
                         {b"PSV0": b"pv", b"RTS0": payload})

    def test_a_game_style_container_has_no_rts0(self):
        # Every one of the 26,503 game containers has these parts and no RTS0.
        tags = [b"SFI0", b"ISG1", b"OSG1", b"PSV0", b"HASH", b"DXIL", b"PRIV"]
        self.assertNotIn(b"RTS0", R.parts(container([(t, b"x") for t in tags])))

    def test_malformed_container_is_empty(self):
        self.assertEqual(R.parts(b"DXBC" + b"\0" * 10), {})
        self.assertEqual(R.parts(b"not a container"), {})


class TestCensus(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write(self, name, chunks, user_sgprs=None):
        with open(os.path.join(self.tmp.name, name + ".dxbc"), "wb") as fh:
            fh.write(container(chunks))
        if user_sgprs is not None:
            os.makedirs(os.path.join(self.tmp.name, "fspx"), exist_ok=True)
            with open(os.path.join(self.tmp.name, "fspx", name + ".json"), "w", encoding="utf-8") as fh:
                json.dump({"user_sgprs": user_sgprs}, fh)

    def test_join_reports_the_spare_dwords(self):
        # Two tables and two CBVs: 8 dwords under the layer's model, 6 under one dword per table.
        self.write("sig", [(b"RTS0", rts0(params=[("table", 0, 1), ("table", 0, 1), ("cbv", 0), ("cbv", 0)]))],
                   user_sgprs=14)
        rows, errors = R.census(self.tmp.name)
        self.assertEqual(errors, [])
        self.assertEqual((rows[0]["span_table2"], rows[0]["span_table1"]), (8, 6))
        self.assertEqual((rows[0]["spare_table2"], rows[0]["spare_table1"]), (6, 8))

    def test_a_container_without_rts0_is_reported_not_dropped(self):
        self.write("plain", [(b"PSV0", b"pv")], user_sgprs=8)
        rows, errors = R.census(self.tmp.name)
        self.assertEqual(errors, [])
        self.assertEqual(rows[0]["error"], "no RTS0 part")
        self.assertEqual(rows[0]["user_sgprs"], 8)

    def test_cli_writes_the_json_and_exits_zero(self):
        self.write("sig", [(b"RTS0", rts0(params=[("cbv", 0)]))], user_sgprs=8)
        out = os.path.join(self.tmp.name, "census.json")
        r = subprocess.run([sys.executable, "-I", TOOL, self.tmp.name, "--json", out],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("spare table2", r.stdout)
        with open(out, encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data["summary"]["parsed"], 1)
        self.assertEqual(data["summary"]["user_sgprs"], {"8": 1})

    def test_a_broken_signature_exits_one(self):
        self.write("broken", [(b"RTS0", rts0()[:10])])
        r = subprocess.run([sys.executable, "-I", TOOL, self.tmp.name], capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("error:", r.stderr)


if __name__ == "__main__":
    unittest.main()
