"""Synthetic BVHs for tools/xdxr_bvh.py: layout, the traversal step, and closest hits against brute force.
No game data is read.

    python -I tests/test_xdxr_bvh.py
"""
import contextlib
import io
import math
import os
import random
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import xdxr_bvh as X  # noqa: E402

TRI = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def rays(rng, n, spread=6.0, aim=3.0):
    for _ in range(n):
        o = [rng.uniform(-spread, spread) for _ in range(3)]
        t = [rng.uniform(-aim, aim) for _ in range(3)]
        yield o, [t[c] - o[c] for c in range(3)]


def same_hit(a, b):
    if a is None or b is None:
        return a is None and b is None
    return (a["primitive"] == b["primitive"] and a["instance"] == b["instance"]
            and math.isclose(a["t"], b["t"], rel_tol=1e-9, abs_tol=1e-12)
            and all(math.isclose(a["bary"][k], b["bary"][k], abs_tol=1e-9) for k in range(2)))


class Layout(unittest.TestCase):
    def test_pointer_encoding(self):
        self.assertEqual(X.ptr(0x9B640 - 0x100 + 0x100 - 0x9B640 + 9941 * 64, X.BOX16), 0x136AC)  # docs/bvh.md
        self.assertEqual(X.ptr_offset(0x136AC), 9941 * 64)
        self.assertEqual(X.ptr_type(0x136AC), X.BOX16)
        with self.assertRaises(ValueError):
            X.ptr(32, X.TRI0)

    def test_header_fields(self):
        data, info = X.build_blas([TRI])
        flags, units, crc, n, *rest = struct.unpack_from("<4I6fIiIIII", data, 0)
        self.assertEqual(flags, X.FLAG_BOTTOM_LEVEL | X.VERSION_2403)
        self.assertEqual((units + 1) * 64, len(data))
        self.assertEqual(n, 1)
        self.assertEqual(rest[0:6], [0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        nodes, _remap, root, elements, parents, cp = rest[6:]
        self.assertEqual(root, info["root"])
        self.assertEqual(X.ptr_type(root), X.BOX32)
        self.assertEqual(X.ptr_offset(root), nodes)
        self.assertEqual(elements % 64, 0)
        self.assertEqual(parents, 64)
        self.assertEqual(cp, 0)

    def test_leaf_record(self):
        data, info = X.build_blas([TRI], geometry_index=5, flags=0, opaque=True)
        off = info["elements"]
        self.assertEqual(struct.unpack_from("<9f", data, off), (0, 0, 0, 1, 0, 0, 0, 1, 0))
        gif, base, prim2, remap = struct.unpack_from("<4I", data, off + 48)
        self.assertEqual(gif, 5 << 8 | 1)
        self.assertEqual((base, prim2), (0, X.INVALID))
        self.assertEqual(remap & 0xFF, X.REMAP_IDENTITY)

    def test_unused_children_miss(self):
        data, info = X.build_blas([TRI])
        root = X.ptr_offset(info["root"])
        self.assertEqual(struct.unpack_from("<4I", data, root)[1:], (X.INVALID,) * 3)
        lo, hi = struct.unpack_from("<3f", data, root + 40), struct.unpack_from("<3f", data, root + 52)
        self.assertTrue(all(lo[c] > hi[c] for c in range(3)))

    def test_box16_is_conservative(self):
        rng = random.Random(7)
        for _ in range(2000):
            lo = [rng.uniform(-70000, 70000) for _ in range(3)]
            hi = [x + rng.uniform(0, 3) for x in lo]
            dlo, dhi = X.decode_box16(X.encode_box16(lo, hi))
            for c in range(3):
                self.assertLessEqual(dlo[c], lo[c])
                self.assertGreaterEqual(dhi[c], hi[c])

    def test_parent_table(self):
        data, info = X.build_blas(X.random_scene(random.Random(2), 40))
        self.assertEqual(X.validate(data), [])
        root_unit = (X.ptr_offset(info["root"]) - info["elements"]) // 64
        self.assertEqual(struct.unpack_from("<I", data, info["parents"] + 4 * root_unit)[0], X.INVALID)


class Step(unittest.TestCase):
    def test_triangle_hit_values(self):
        t_num, t_den, i, j = X.triangle_test(TRI, [0.25, 0.25, -2.0], [0.0, 0.0, 1.0])
        self.assertAlmostEqual(t_num / t_den, 2.0)
        self.assertAlmostEqual(i / t_den, 0.25)
        self.assertAlmostEqual(j / t_den, 0.25)
        self.assertEqual(X.triangle_test(TRI, [2.0, 2.0, -2.0], [0.0, 0.0, 1.0])[0], math.inf)
        self.assertEqual(X.triangle_test(TRI, [0.25, 0.25, 2.0], [0.0, 0.0, 1.0])[0], math.inf)   # behind

    def test_remap_selects_edge_functions(self):
        r = [X.triangle_test(TRI, [0.1, 0.3, -1.0], [0.0, 0.0, 1.0], remap)
             for remap in (1 | 2 << 2, 2 | 1 << 2, 0 | 1 << 2)]
        bary = [(x[2] / x[1], x[3] / x[1]) for x in r]
        self.assertAlmostEqual(bary[0][0], 0.1)
        self.assertAlmostEqual(bary[0][1], 0.3)
        self.assertAlmostEqual(bary[1][0], 0.3)
        self.assertAlmostEqual(bary[2][0], 0.6)

    def test_tri1_reads_slots_1_3_2(self):
        quad = [[[0, 0, 0], [1, 0, 0], [1, 1, 0]], [[0, 0, 0], [1, 1, 0], [0, 1, 0]]]
        data, info = X.build_blas(quad, pair=True)
        self.assertEqual(info["leaves"], 1)
        mem = X.Memory()
        mem.place(0x1000, data)
        leaf = info["elements"]
        p1 = X.ptr(leaf, X.TRI1)
        o, d = [0.2, 0.7, -1.0], [0.0, 0.0, 1.0]
        res = X.intersect_node(mem, 0x1000, p1, o, d, X._inverse_dir(d), math.inf)
        self.assertNotEqual(res[0], math.inf)
        hit = X.trace(mem, 0x1000, o, d)
        self.assertEqual(hit["primitive"], 1)
        want = X.brute_force([(None, 0xFF, quad)], o, d)
        self.assertTrue(same_hit(hit, want), (hit, want))

    def test_box_step_sorts_and_culls(self):
        # four triangles: three stacked on the ray at z = 3, 1, 2 and one off to the side
        tris = [[[0, 0, z], [1, 0, z], [0, 1, z]] for z in (3.0, 1.0, 2.0)] + [[[10, 0, 1], [11, 0, 1], [10, 1, 1]]]
        data, info = X.build_blas(tris)
        mem = X.Memory()
        mem.place(0, data)
        o, d = [0.2, 0.2, 0.0], [0.0, 0.0, 1.0]
        kids = X.intersect_node(mem, 0, info["root"], o, d, X._inverse_dir(d), math.inf)
        prim = lambda p: struct.unpack_from("<I", data, X.ptr_offset(p) + 52)[0]
        self.assertEqual(kids[3], X.INVALID)
        self.assertEqual([prim(p) for p in kids[:3]], [1, 2, 0])          # nearest first
        unsorted = X.intersect_node(mem, 0, info["root"], o, d, X._inverse_dir(d), math.inf, sort=False)
        self.assertEqual(sorted(unsorted), sorted(kids))
        culled = X.intersect_node(mem, 0, info["root"], o, d, X._inverse_dir(d), 1.5)
        self.assertEqual([prim(p) for p in culled if p != X.INVALID], [1])   # boxes beyond the extent miss


class ClosestHit(unittest.TestCase):
    def check_blas(self, tris, **kw):
        rng = random.Random(11)
        data, info = X.build_blas(tris, **kw)
        self.assertEqual(X.validate(data, expect_bottom=True), [])
        mem = X.Memory()
        mem.place(0x40000, data)
        hits = 0
        for o, d in rays(rng, 600):
            a = X.trace(mem, 0x40000, o, d)
            b = X.brute_force([(None, 0xFF, tris)], o, d)
            self.assertTrue(same_hit(a, b), (o, d, a, b))
            hits += a is not None
        self.assertGreater(hits, 30)

    def test_random_box32(self):
        self.check_blas(X.random_scene(random.Random(1), 80))

    def test_random_box16(self):
        self.check_blas(X.random_scene(random.Random(1), 80), box16=True)

    def test_grid_paired(self):
        self.check_blas(X.grid_mesh(), pair=True)

    def test_grid_paired_box16(self):
        self.check_blas(X.grid_mesh(), pair=True, box16=True)

    def test_tlas(self):
        rng = random.Random(5)
        mesh_a, mesh_b = X.grid_mesh(4, 2.0), X.random_scene(rng, 30, 1.0)
        blas_a, ia = X.build_blas(mesh_a, pair=True)
        blas_b, ib = X.build_blas(mesh_b, box16=True)
        mem = X.Memory()
        va_a, va_b = mem.place(0x100000, blas_a), mem.place(0x200000, blas_b)
        c, s = math.cos(0.7), math.sin(0.7)
        xforms = [
            ([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], va_a, ia, 0x01),
            ([[c, -s, 0, 3], [s, c, 0, -1], [0, 0, 1, 0.5]], va_a, ia, 0x02),
            ([[2, 0, 0, -3], [0, 0.5, 0, 2], [0, 0, 1.5, 1]], va_b, ib, 0x01),
            ([[c, 0, s, 0], [0, 1, 0, 3], [-s, 0, c, -2]], va_b, ib, 0x04),
        ]
        insts = [X.Instance(m, va, info["root"], info["bounds"], instance_id=10 + k, mask=mask)
                 for k, (m, va, info, mask) in enumerate(xforms)]
        tlas, it = X.build_tlas(insts)
        self.assertEqual(X.validate(tlas, expect_bottom=False), [])
        va_t = mem.place(0x300000, tlas)
        rec = X.ptr_offset(struct.unpack_from("<I", tlas, 48)[0])
        self.assertEqual(it["instances"], 4)
        scene = [(m, mask, mesh_a if va == va_a else mesh_b) for m, va, _, mask in xforms]
        hits = 0
        for mask in (0xFF, 0x01, 0x06):
            for o, d in rays(rng, 300, spread=8.0, aim=4.0):
                a = X.trace(mem, va_t, o, d, mask=mask)
                b = X.brute_force(scene, o, d, mask=mask)
                if a is not None:
                    hits += 1
                    # world t survives the object-space transform; tolerance for the float32 inverse matrix
                    self.assertIsNotNone(b)
                    self.assertEqual((a["instance"], a["primitive"]), (b["instance"], b["primitive"]))
                    self.assertTrue(math.isclose(a["t"], b["t"], rel_tol=1e-5))
                else:
                    self.assertIsNone(b)
        self.assertGreater(hits, 50)
        self.assertGreater(rec, 0)

    def test_instance_record(self):
        data, info = X.build_blas([TRI])
        inst = X.Instance([[1, 0, 0, 5], [0, 1, 0, 0], [0, 0, 1, 0]], 0x12300, info["root"], info["bounds"],
                          instance_id=0x123456, mask=0xAB, contribution=7, flags=0x3)
        tlas, _ = X.build_tlas([inst])
        off = X.ptr_offset(struct.unpack_from("<4I", tlas, X.ptr_offset(struct.unpack_from("<I", tlas, 48)[0]))[0])
        id_mask, contrib, internal, blas = struct.unpack_from("<4I", tlas, off + 48)
        self.assertEqual(id_mask, 0xAB123456)
        self.assertEqual(contrib, 0x03000007)
        self.assertEqual(internal, info["root"] | X.INST_IDENTITY_ROTATION)
        self.assertEqual(blas, 0x123)
        self.assertEqual(struct.unpack_from("<4f", tlas, off + 64), (1.0, 0.0, 0.0, -5.0))
        with self.assertRaises(ValueError):
            X.build_tlas([X.Instance(inst.transform, 0x12340, info["root"], info["bounds"])])


class Validate(unittest.TestCase):
    def setUp(self):
        self.data, self.info = X.build_blas(X.random_scene(random.Random(4), 30))

    def corrupt(self, off, fmt, *vals):
        b = bytearray(self.data)
        struct.pack_into(fmt, b, off, *vals)
        return X.validate(bytes(b))

    def test_size_mismatch(self):
        self.assertTrue(any("totalSizeIn64BNodes" in e for e in X.validate(self.data + bytes(64))))

    def test_element_count(self):
        self.assertTrue(any("numElements" in e for e in self.corrupt(12, "<I", 29)))

    def test_child_outside_parent(self):
        root = X.ptr_offset(self.info["root"])
        errs = self.corrupt(root + 16, "<3f", 1e6, 1e6, 1e6)
        self.assertTrue(errs)

    def test_parent_entry(self):
        b = bytearray(self.data)
        root = X.ptr_offset(self.info["root"])
        child = struct.unpack_from("<I", b, root)[0]
        if X.ptr_type(child) > X.TRI3:
            u = (X.ptr_offset(child) - self.info["elements"]) // 64
            struct.pack_into("<I", b, self.info["parents"] + 4 * u, 0x1234)
            self.assertTrue(any("parent table" in e for e in X.validate(bytes(b))))

    def test_dehydrated_refused(self):
        errs = self.corrupt(0, "<I", 0x81020006 | X.FLAG_DEHYDRATED)
        self.assertTrue(any("dehydrated" in e for e in errs))

    def test_short_buffer(self):
        self.assertEqual(X.validate(b"\0" * 10), ["buffer shorter than the 64-byte header"])


class CheckBuild(unittest.TestCase):
    """check-build on cases written the way vk_xdxr_build_test writes them, built here by the reference."""

    def write_cases(self, d, corrupt=False):
        import json
        tris = X.grid_mesh(4, 3.0)
        blas, info = X.build_blas(tris)
        if corrupt:                                    # move one leaf's first vertex: traces must disagree
            b = bytearray(blas)
            for leaf in range(info["triangles"]):
                struct.pack_into("<f", b, info["elements"] + 64 * leaf, 50.0)
            blas = bytes(b)
        va, bias = 0x12345600, 0x7F0000000000
        with open(os.path.join(d, "blas_grid.bin"), "wb") as f:
            f.write(blas)
        with open(os.path.join(d, "blas_grid.json"), "w") as f:
            json.dump({"kind": "blas", "address": va + bias,
                       "triangles": [t + [0, i] for i, t in enumerate(tris)]}, f)
        m = [[1, 0, 0, 2], [0, 1, 0, -1], [0, 0, 1, 0.5]]
        tlas, _ = X.build_tlas([X.Instance(m, va, info["root"], info["bounds"])])
        with open(os.path.join(d, "tlas.bin"), "wb") as f:
            f.write(tlas)
        with open(os.path.join(d, "tlas.json"), "w") as f:
            json.dump({"kind": "tlas", "address": 0x40000000, "bias": bias, "blas": ["blas_grid"],
                       "instances": [{"transform": sum(m, []), "mask": 255, "blas": 0}]}, f)

    def run_check(self, corrupt):
        with tempfile.TemporaryDirectory() as d:
            self.write_cases(d, corrupt)
            with contextlib.redirect_stdout(io.StringIO()):
                return X.check_build(d, rays=200)

    def test_reference_cases_pass(self):
        self.assertEqual(self.run_check(False), [])

    def test_corrupted_leaves_fail(self):
        self.assertTrue(self.run_check(True))

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(X.check_build(d), [f"no cases in {d}"])


class Cli(unittest.TestCase):
    def test_validate_file(self):
        data, _ = X.build_blas(X.grid_mesh(3), pair=True)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "blas.bin")
            with open(p, "wb") as f:
                f.write(data)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(X.main(["validate", p]), 0)
            self.assertIn("valid", out.getvalue())
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(X.main(["validate", os.path.join(d, "missing.bin")]), 2)

    def test_selftest(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(X.main(["selftest", "--rays", "50"]), 0)


if __name__ == "__main__":
    unittest.main()
