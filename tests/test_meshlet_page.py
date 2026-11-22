"""Synthetic-data tests for tools/meshlet_page.py.

    python -I tests/test_meshlet_page.py

Every byte here is built by the test. No game file, value or name is used.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "meshlet_page.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import meshlet_page as mp  # noqa: E402


def pad16(buf):
    buf += b"\0" * (-len(buf) % 16)
    return buf


def block(n, colour=True, uv1=True, normal=True, flags=0x200):
    """A vertex block with n vertices: slots 0 (pos), 1 (rgba), 2 (normal), 3 (uv0), 4 (uv1).

    The streams are stored in slot order 0, 2, 1, 3, 4, as measured.
    """
    order = [(0, 0x63, 12), (2, 0x63, 12), (1, 0x26, 4), (3, 0x42, 8), (4, 0x42, 8)]
    if not normal:
        order = [o for o in order if o[0] != 2]
    if not colour:
        order = [o for o in order if o[0] != 1]
    if not uv1:
        order = [o for o in order if o[0] != 4]
    head = bytearray(0x60)
    off = 0x60
    body = bytearray()
    for slot, tag, stride in order:
        struct.pack_into("<I", head, 4 * slot, tag | ((off // 2) << 8))
        for v in range(n):
            if slot == 0:
                body += struct.pack("<3f", v * 0.5, -v * 0.25, 1.0)
            elif slot == 1:
                body += bytes((v & 0xFF, 0, 0, 255))
            elif slot == 2:
                body += struct.pack("<3f", 0.0, 0.0, 1.0)
            else:
                body += struct.pack("<2f", v / max(n, 1), 0.5)
        off += stride * n
    struct.pack_into("<I", head, 0x5C, flags)
    return bytes(head + body)


def group(sizes):
    """A meshlet group. sizes is a list of (vertices, triangles) per meshlet (each <= 128)."""
    recs = bytearray()
    tris = bytearray()
    refs = bytearray()
    vbase = tbase = 0
    for m, (vc, tc) in enumerate(sizes):
        recs += struct.pack("<4I", (vc - 1) | ((tc - 1) << 7), vbase | (tbase << 16), 0x7FF7FF, 0x7FF)
        for t in range(tc):
            a, b, c = t % vc, (t + 1) % vc, (t + 2) % vc
            tris += struct.pack("<I", a | (b << 7) | (c << 14) | (m << 21))
        for v in range(vc):
            refs += struct.pack("<H", vbase + v)
        vbase += vc
        tbase += tc
    return struct.pack("<3I", 0x80000000 | len(sizes), tbase, vbase) + bytes(recs + tris + refs)


def page(*chunks):
    buf = bytearray()
    for c in chunks:
        buf = pad16(buf)
        buf += c
    return bytes(pad16(buf))


class DrawBindingTests(unittest.TestCase):
    def setUp(self):
        self.system = bytearray(704)
        self.graphics = bytearray(640)
        # Two same-sized blocks. The group near the first binds the second.
        for at in (0, 384):
            struct.pack_into("<I", self.graphics, at, 0x4042)
        data = bytes(pad16(bytearray(group([(8, 4)]))))
        self.graphics[256:256 + len(data)] = data
        for at, target in ((8, 160), (64, 240), (224, 320), (240, 256)):
            struct.pack_into("<Q", self.system, at, 0x50000000 + target)
        struct.pack_into("<IH", self.system, 168, 192, 1)
        struct.pack_into("<Q", self.system, 184, 0x60000000 + 384)
        struct.pack_into("<HHI", self.system, 248, 1, 4, 8 << 8)
        struct.pack_into("<IH", self.system, 264, len(data) // 4, 4)
        struct.pack_into("<Q", self.system, 280, 0x60000000 + 256)
        struct.pack_into("<I", self.system, 320, 128)
        self.system[320 + 208] = 8
        self.system[320 + 260] = 16
        struct.pack_into("<I", self.system, 320 + 312, (8 << 10) | 1)

    def test_binding_uses_resource_relative_pointers(self):
        result = mp.draw_binding(self.system, self.graphics, 0)
        self.assertEqual(result, {"vertex_block": 384, "meshlet_group": 256, "vertices": 8})

    def test_binding_to_group_in_later_page(self):
        graphics = self.graphics + self.graphics[256:336]
        struct.pack_into("<Q", self.system, 280, 0x60000000 + 640)
        self.assertEqual(mp.draw_binding(self.system, graphics, 0)["meshlet_group"], 640)

    def test_primary_lod_traversal(self):
        self.system.extend(bytes(304))
        for at, target in ((784, 880), (880, 896), (896, 912), (920, 976), (976, 0)):
            struct.pack_into("<Q", self.system, at, 0x50000000 + target)
        for at in (888, 928):
            struct.pack_into("<HH", self.system, at, 1, 1)
        struct.pack_into("<H", self.system, 966, 1)
        self.assertEqual(list(mp.drawable_geometries(self.system, 704)), [0])
        self.system[966] = 2
        with self.assertRaises(mp.PageError):
            list(mp.drawable_geometries(self.system, 704))

    def test_inconsistent_bindings(self):
        for offset, fmt, value in ((64, "Q", 0x50010000), (248, "H", 2),
                                   (280, "Q", 0x60010000), (632, "I", (7 << 10) | 1)):
            with self.subTest(offset=offset):
                damaged = self.system.copy()
                struct.pack_into("<" + fmt, damaged, offset, value)
                with self.assertRaises(mp.PageError):
                    mp.draw_binding(damaged, self.graphics, 0)

    def test_cli_preserves_unresolved_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            system = os.path.join(folder, "system")
            graphics = os.path.join(folder, "graphics")
            for path, data in ((system, self.system), (graphics, self.graphics)):
                with open(path, "wb") as out:
                    out.write(data)
            run = subprocess.run([sys.executable, "-I", TOOL, graphics, "--draw-system", system,
                                  "--geometry-offset", "0", "--geometry-offset", "0xffff"],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 1)
            rows = json.loads(run.stdout)["bindings"]
            self.assertEqual(rows[0]["vertex_block"], 384)
            self.assertIn("unresolved", rows[1])


class ParseTests(unittest.TestCase):
    def test_block_then_group(self):
        data = page(block(200), group([(128, 120), (72, 70)]))
        chunks = mp.parse_page(data)
        self.assertEqual([c["kind"] for c in chunks], ["block", "group"])
        self.assertEqual(chunks[0]["vertices"], 200)
        self.assertEqual(chunks[0]["flags"], 0x200)
        self.assertTrue(chunks[0]["tags_expected"])
        g = chunks[1]
        self.assertEqual((g["meshlets"], g["triangles"], g["vertex_refs"]), (2, 190, 200))
        self.assertEqual((g["vertex_span"], g["unique_refs"], g["full_meshlets"]), (200, 200, 1))

    def test_three_slot_block_and_odd_ref_count(self):
        # 3 vertices -> odd u16 reference count, so the next chunk needs the 16-byte realignment
        data = page(block(3, colour=False, uv1=False), group([(3, 1)]), block(5, colour=False, uv1=False))
        chunks = mp.parse_page(data)
        self.assertEqual([c["kind"] for c in chunks], ["block", "group", "block"])
        self.assertEqual([s[0] for s in chunks[0]["slots"]], [0, 2, 3])

    def test_block_without_normals(self):
        # slots 0, 3, 4 only: the vertex count comes from slot 0's span up to the next stream (slot 3)
        data = page(block(7, colour=False, normal=False), group([(7, 5)]))
        chunks = mp.parse_page(data)
        self.assertEqual([c["kind"] for c in chunks], ["block", "group"])
        self.assertEqual(chunks[0]["vertices"], 7)
        self.assertEqual([s[0] for s in chunks[0]["slots"]], [0, 3, 4])
        self.assertTrue(chunks[0]["tags_expected"])
        summary = mp.summarise_page(data, "x")
        self.assertNotIn("error", summary)
        c = summary["checks"]
        self.assertEqual((c["vertices"], c["positions_finite"], c["normals_unit"]), (7, 7, 0))
        self.assertEqual((c["uv_vertices"], c["uv_finite"]), (14, 14))

    def test_group_led_page(self):
        data = page(group([(10, 8)]), block(10))
        self.assertTrue(mp.is_mesh_page(data))
        summary = mp.summarise_page(data, "x")
        self.assertEqual(summary["starts_with"], "group")
        self.assertEqual(summary["sequence"], "GB")
        self.assertEqual(summary["groups_matching_block_here"], 1)

    def test_stop_at_other_block(self):
        other = struct.pack("<I", 0x4042) + b"\x26\x40\x03\x00" + b"\x55" * 40
        summary = mp.summarise_page(page(group([(4, 2)]), other), "x")
        self.assertEqual(summary["sequence"], "G.")
        self.assertEqual(summary["stop"], "42 40 00 00 block")

    def test_stop_in_truncated_group(self):
        data = page(block(4), group([(4, 2)]))
        g = group([(100, 90)])
        summary = mp.summarise_page(data + g[:64], "x")
        self.assertEqual(summary["sequence"], "BG.")
        self.assertEqual(summary["stop"], "group continues past the stream")

    def test_checks_count_only(self):
        summary = mp.summarise_page(page(block(16), group([(16, 14)])), "x")
        c = summary["checks"]
        self.assertEqual(c["vertices"], 16)
        self.assertEqual(c["positions_finite"], 16)
        self.assertEqual(c["normals_unit"], 16)
        self.assertEqual(c["colour_alpha_255"], 16)
        self.assertEqual(c["uv_vertices"], 32)
        for value in summary.values():
            self.assertNotIsInstance(value, float)


class RejectTests(unittest.TestCase):
    def test_not_a_page(self):
        with self.assertRaises(mp.PageError):
            mp.parse_page(b"\x01\x02\x03\x04" + b"\0" * 60)
        self.assertFalse(mp.is_mesh_page(b"\x63\x30\x00"))

    def test_bad_running_sum(self):
        g = bytearray(group([(8, 6), (8, 6)]))
        struct.pack_into("<I", g, 12 + 16 + 4, 9)   # second meshlet's first vertex reference
        with self.assertRaises(mp.PageError):
            mp.parse_group(bytes(g), 0)

    def test_bad_triangle_meshlet_id(self):
        g = bytearray(group([(8, 6)]))
        tri_at = 12 + 16
        struct.pack_into("<I", g, tri_at, 0 | (1 << 7) | (2 << 14) | (5 << 21))
        with self.assertRaises(mp.PageError):
            mp.parse_group(bytes(g), 0)

    def test_index_past_meshlet(self):
        g = bytearray(group([(8, 6)]))
        struct.pack_into("<I", g, 12 + 16, 0 | (1 << 7) | (9 << 14))
        with self.assertRaises(mp.PageError):
            mp.parse_group(bytes(g), 0)

    def test_header_sum_mismatch(self):
        g = bytearray(group([(8, 6)]))
        struct.pack_into("<I", g, 8, 9)   # one more reference than the meshlets cover
        g += b"\0\0"
        with self.assertRaises(mp.PageError):
            mp.parse_group(bytes(g), 0)

    def test_slot_size_mismatch(self):
        b = bytearray(block(10))
        struct.pack_into("<I", b, 4, 0x26 | (((0x60 + 120 + 8) // 2) << 8))
        with self.assertRaises(mp.PageError):
            mp.parse_block(bytes(b), 0)

    def test_block_with_only_slot_0(self):
        b = bytearray(0x60 + 12 * 4)
        struct.pack_into("<I", b, 0, 0x63 | ((0x60 // 2) << 8))
        with self.assertRaises(mp.PageError):
            mp.parse_block(bytes(b), 0)

    def test_unknown_chunk(self):
        data = page(block(4)) + struct.pack("<I", 0x1234) + b"\0" * 12
        summary = mp.summarise_page(data, "x")
        self.assertIn("unknown chunk", summary["error"])


class CliTests(unittest.TestCase):
    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True)

    def test_directory_and_archive(self):
        pages = [page(block(20), group([(20, 18)])), page(group([(6, 4)]), block(6, colour=False, uv1=False))]
        with tempfile.TemporaryDirectory() as tmp:
            folder = os.path.join(tmp, "carve")
            os.mkdir(folder)
            for i, data in enumerate(pages):
                with open(os.path.join(folder, f"{i:04x}.bin"), "wb") as fh:
                    fh.write(data)
            with open(os.path.join(folder, "other.bin"), "wb") as fh:
                fh.write(b"\0" * 64)
            out = os.path.join(tmp, "dir.json")
            r = self.run_tool(folder, "--json", out, "--quiet")
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(out, encoding="utf-8") as fh:
                totals = json.load(fh)["totals"]
            self.assertEqual((totals["pages"], totals["block_led_pages"], totals["group_led_pages"]), (2, 1, 1))
            self.assertEqual((totals["blocks"], totals["groups"], totals["vertices"]), (2, 2, 26))
            self.assertEqual(totals["groups_matching_block_near"], 2)

            # A minimal RPF9-like file: data offset at +4, then zlib streams (one is not a mesh page).
            rpf = os.path.join(tmp, "a.rpf")
            body = b"".join(zlib.compress(p) for p in pages + [b"\x11" * 128])
            with open(rpf, "wb") as fh:
                fh.write(b"9FPR" + struct.pack("<I", 16) + b"\0" * 8 + body)
            out2 = os.path.join(tmp, "rpf.json")
            r = self.run_tool(rpf, "--json", out2, "--quiet")
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(out2, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["totals"]["pages"], 2)

    def test_page_report_write_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "page.bin")
            with open(source, "wb") as fh:
                fh.write(page(block(4)))
            out = os.path.join(tmp, "missing", "report.json")

            result = self.run_tool(source, "--json", out, "--quiet")

        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn(f"cannot write {out}", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_neighbor_counts_keep_malformed_page_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            block_path = os.path.join(tmp, "block.bin")
            bad_path = os.path.join(tmp, "malformed.bin")
            group_path = os.path.join(tmp, "group.bin")
            out = os.path.join(tmp, "report.json")
            for path, data in ((block_path, page(block(4))),
                               (bad_path, b"bad"),
                               (group_path, page(group([(4, 2)])))):
                with open(path, "wb") as fh:
                    fh.write(data)

            for paths, exit_code, matches in (
                ((block_path, group_path), 0, 1),
                ((group_path, block_path), 0, 1),
                ((block_path, bad_path, group_path), 1, 0),
            ):
                with self.subTest(paths=paths):
                    result = self.run_tool(*paths, "--json", out, "--quiet")
                    self.assertEqual(result.returncode, exit_code, result.stderr)
                    with open(out, encoding="utf-8") as fh:
                        totals = json.load(fh)["totals"]
                    self.assertEqual(totals["groups_matching_block_near"], matches)

    def test_missing_input(self):
        r = self.run_tool(os.path.join(tempfile.gettempdir(), "no-such-meshlet-page.bin"))
        self.assertEqual(r.returncode, 2)

    def test_error_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.bin")
            with open(path, "wb") as fh:
                fh.write(page(block(4)) + struct.pack("<I", 0x1234) + b"\0" * 12)
            r = self.run_tool(path, "--quiet")
            self.assertEqual(r.returncode, 1)

    def test_json_that_is_a_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "page.bin")
            with open(source, "wb") as fh:
                fh.write(page(block(4)))
            r = self.run_tool(source, "--json", tmp, "--quiet")
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("is a directory", r.stderr)
            self.assertNotIn("Traceback", r.stderr)

    def test_json_over_input_archive_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            rpf = os.path.join(tmp, "test.rpf")
            data = b"9FPR" + struct.pack("<I", 16) + b"\0" * 8 + zlib.compress(page(block(4)))
            with open(rpf, "wb") as fh:
                fh.write(data)
            alias = os.path.join(tmp, ".", "test.rpf")
            r = self.run_tool(rpf, "--json", alias, "--quiet")
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("would overwrite the input", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            with open(rpf, "rb") as fh:
                self.assertEqual(fh.read(), data)

    def test_json_over_input_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "page.bin")
            data = page(block(4))
            with open(source, "wb") as fh:
                fh.write(data)
            r = self.run_tool(source, "--json", source, "--quiet")
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("would overwrite the input", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            with open(source, "rb") as fh:
                self.assertEqual(fh.read(), data)

    def test_json_over_carve_dir_manifest_or_payload_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            carve = os.path.join(tmp, "carve")
            os.mkdir(carve)
            manifest = os.path.join(carve, "index.json")
            with open(manifest, "w", encoding="utf-8") as fh:
                fh.write('{"files": []}')
            bin_path = os.path.join(carve, "0000.bin")
            bin_data = page(block(4))
            with open(bin_path, "wb") as fh:
                fh.write(bin_data)

            # Refuse overwrite of index.json
            r = self.run_tool(carve, "--json", manifest, "--quiet")
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("would overwrite the input", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            with open(manifest, "r", encoding="utf-8") as fh:
                self.assertEqual(fh.read(), '{"files": []}')

            # Refuse overwrite of carved .bin payload
            r = self.run_tool(carve, "--json", bin_path, "--quiet")
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("would overwrite the input", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            with open(bin_path, "rb") as fh:
                self.assertEqual(fh.read(), bin_data)

    def test_fresh_json_inside_carve_dir_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            carve = os.path.join(tmp, "carve")
            os.mkdir(carve)
            bin_path = os.path.join(carve, "0000.bin")
            with open(bin_path, "wb") as fh:
                fh.write(page(block(4)))
            out = os.path.join(carve, "report.json")
            r = self.run_tool(carve, "--json", out, "--quiet")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(os.path.isfile(out))
            with open(out, "r", encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["totals"]["pages"], 1)

    def test_json_over_draw_system_inputs_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            sys_path = os.path.join(tmp, "system.bin")
            gfx_path = os.path.join(tmp, "graphics.bin")
            with open(sys_path, "wb") as fh:
                fh.write(b"\0" * 256)
            with open(gfx_path, "wb") as fh:
                fh.write(b"\0" * 256)

            # Refuse overwrite of graphics segment
            r = self.run_tool(gfx_path, "--draw-system", sys_path, "--geometry-offset", "0", "--json", gfx_path, "--quiet")
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("would overwrite the input", r.stderr)
            self.assertNotIn("Traceback", r.stderr)

            # Refuse overwrite of system segment
            r = self.run_tool(gfx_path, "--draw-system", sys_path, "--geometry-offset", "0", "--json", sys_path, "--quiet")
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("would overwrite the input", r.stderr)
            self.assertNotIn("Traceback", r.stderr)


if __name__ == "__main__":
    unittest.main()
