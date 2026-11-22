"""Pure synthetic unit tests for tools/ped_assemble.py (zero game files).

    python -I tests/test_ped_assemble.py

Constructs synthetic overlapping modular body-part meshes (upper torso + lower
pelvis sharing a coplanar waist overlap band, plus female nipple and genital
patches, plus nearby non-coplanar geometry) to verify:
- exact coplanar overlap triangle removal counts,
- watertight seam preservation (zero unintended boundary-edge holes, zero
  non-manifold overlap edges after stitching),
- preservation of nearby non-coplanar triangles,
- standing-space +1.0 m SKEL_ROOT alignment and RAGE Z-up -> glTF Y-up math,
- valid binary glTF 2.0 (.glb) export (static textured and 8-influence skinned,
  cross-checked with tools/glb_writer.py and tools/glb_skinning.py),
- CLI execution (`--help`, `assemble`, `cull-overlap`, `inspect-seams`).
"""
from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import glb_skinning  # noqa: E402
import glb_writer  # noqa: E402
import ped_assemble as pa  # noqa: E402


def _make_1x1_png(r: int, g: int, b: int, a: int = 255) -> bytes:
    """Build a minimal valid 1x1 RGBA PNG using only stdlib zlib/struct."""
    def _chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    raw_scanline = bytes([0, r, g, b, a])
    idat = zlib.compress(raw_scanline)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )


def _cylinder_band(
    z_levels: list[float],
    n_seg: int = 16,
    radius: float = 0.15,
    *,
    split_islands: bool = False,
    radial_offset: float = 0.0,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]], list[tuple[float, float]]]:
    """Generate a synthetic tubular mesh in RAGE Z-up spanning ``z_levels``.

    When ``split_islands=True``, each quad band is emitted with duplicated
    vertex indices (mirroring RAGE 9's physically split ``_nude_cull`` islands
    whose positions still weld seamlessly).
    """
    positions: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    r = radius + radial_offset

    if not split_islands:
        for zi, z in enumerate(z_levels):
            v_coord = zi / max(len(z_levels) - 1, 1)
            for si in range(n_seg):
                ang = 2.0 * math.pi * si / n_seg
                positions.append((r * math.cos(ang), r * math.sin(ang), z))
                uvs.append((si / n_seg, v_coord))
        for zi in range(len(z_levels) - 1):
            row0 = zi * n_seg
            row1 = (zi + 1) * n_seg
            for si in range(n_seg):
                sn = (si + 1) % n_seg
                a, b = row0 + si, row0 + sn
                c, d = row1 + si, row1 + sn
                triangles.append((a, b, d))
                triangles.append((a, d, c))
        return positions, triangles, uvs

    for zi in range(len(z_levels) - 1):
        z0, z1 = z_levels[zi], z_levels[zi + 1]
        v0 = zi / max(len(z_levels) - 1, 1)
        v1 = (zi + 1) / max(len(z_levels) - 1, 1)
        for si in range(n_seg):
            sn = (si + 1) % n_seg
            a0 = 2.0 * math.pi * si / n_seg
            a1 = 2.0 * math.pi * sn / n_seg
            base = len(positions)
            positions.extend([
                (r * math.cos(a0), r * math.sin(a0), z0),
                (r * math.cos(a1), r * math.sin(a1), z0),
                (r * math.cos(a0), r * math.sin(a0), z1),
                (r * math.cos(a1), r * math.sin(a1), z1),
            ])
            uvs.extend([
                (si / n_seg, v0),
                ((si + 1) / n_seg, v0),
                (si / n_seg, v1),
                ((si + 1) / n_seg, v1),
            ])
            triangles.append((base + 0, base + 1, base + 3))
            triangles.append((base + 0, base + 3, base + 2))
    return positions, triangles, uvs


def _extract_sub_patch(
    positions: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
    tri_indices: list[int],
    *,
    normal_shift: float = 0.0,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """Extract a subset of triangles as a standalone patch, optionally nudged slightly along +radius."""
    remap: dict[int, int] = {}
    out_pos: list[tuple[float, float, float]] = []
    out_tris: list[tuple[int, int, int]] = []
    for ti in tri_indices:
        tri = triangles[ti]
        nt = []
        for vi in tri:
            if vi not in remap:
                x, y, z = positions[vi]
                if normal_shift != 0.0:
                    rad = math.hypot(x, y) or 1.0
                    x += normal_shift * (x / rad)
                    y += normal_shift * (y / rad)
                remap[vi] = len(out_pos)
                out_pos.append((x, y, z))
            nt.append(remap[vi])
        out_tris.append((nt[0], nt[1], nt[2]))
    return out_pos, out_tris


class StandingSpaceTests(unittest.TestCase):
    def test_standing_offset_and_zup_to_yup(self):
        # Bind-space biped: feet at Z = -1.0 m, head top at Z = +0.72 m (1.72 m tall)
        pos_bind = [(0.0, 0.0, -1.0), (0.1, 0.2, 0.0), (0.0, 0.0, 0.72)]
        self.assertTrue(pa.is_bind_space_biped(pos_bind, coord_space="z-up"))

        shifted_z = pa.apply_standing_offset(pos_bind, offset_m=1.0, coord_space="z-up")
        self.assertFalse(pa.is_bind_space_biped(shifted_z, coord_space="z-up"))
        self.assertAlmostEqual(shifted_z[0][2], 0.0, places=7)
        self.assertAlmostEqual(shifted_z[1][2], 1.0, places=7)
        self.assertAlmostEqual(shifted_z[2][2], 1.72, places=7)

        part = pa.MeshPart(
            name="hand_fr1_000_nude_cull",
            positions=pos_bind,
            triangles=[(0, 1, 2)],
            normals=[(0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)],
            coord_space="z-up",
        )
        aligned = pa.align_part_to_standing_space(part, standing_offset=1.0, target_space="y-up")
        self.assertEqual(aligned.coord_space, "y-up")
        # RAGE (0, 0, 0.0) -> glTF (0, 0, 0); RAGE (0.1, 0.2, 1.0) -> glTF (0.1, 1.0, -0.2)
        self.assertAlmostEqual(aligned.positions[0][1], 0.0, places=7)
        self.assertAlmostEqual(aligned.positions[1][0], 0.1, places=7)
        self.assertAlmostEqual(aligned.positions[1][1], 1.0, places=7)
        self.assertAlmostEqual(aligned.positions[1][2], -0.2, places=7)
        self.assertAlmostEqual(aligned.positions[2][1], 1.72, places=7)
        # Normals rotated identically: (0,1,0)->(0,0,-1), (0,0,1)->(0,1,0)
        self.assertEqual(aligned.normals[0], (0.0, 0.0, -1.0))
        self.assertEqual(aligned.normals[1], (0.0, 1.0, 0.0))
        # Roundtrip back to RAGE Z-up
        for p in aligned.positions:
            back = pa.rage_to_yup_vec3(pa.yup_to_rage_vec3(p))
            for a, b in zip(p, back):
                self.assertAlmostEqual(a, b, places=7)

    def test_role_classification(self):
        self.assertEqual(pa.classify_part_role("hand_fr1_000_nude_cull"), pa.ROLE_UPPER_BODY)
        self.assertEqual(pa.classify_part_role("hand_mr1_000_nude_cull.glb"), pa.ROLE_UPPER_BODY)
        self.assertEqual(pa.classify_part_role("feet_fr1_000_nude_cull"), pa.ROLE_LOWER_BODY)
        self.assertEqual(pa.classify_part_role("bodynipplesgen_fr1_000_regular_ws00_full"), pa.ROLE_NIPPLES)
        self.assertEqual(pa.classify_part_role("bodygenitalsgen_fr1_008_regular_ws00_full"), pa.ROLE_GENITALS_FEMALE)
        self.assertEqual(pa.classify_part_role("bodygenitalsgen_mr1_000_regular_ws00_full"), pa.ROLE_GENITALS_MALE)
        self.assertEqual(pa.classify_part_role("uppr_000"), pa.ROLE_UPPER_BODY)
        self.assertEqual(pa.classify_part_role("lowr_000"), pa.ROLE_LOWER_BODY)


class CoplanarCullingAndSeamTests(unittest.TestCase):
    def _build_synthetic_body_parts(
        self,
        *,
        waist_radial_jitter: float = 0.0,
        patch_normal_shift: float = 0.0,
    ):
        # Upper torso spans Z = [0.90, 1.40] with 5 bands (6 z-levels), 16 segments -> 5 * 32 = 160 tris.
        # Lower pelvis spans Z = [0.50, 1.04] with 4 bands (5 z-levels), 16 segments -> 4 * 32 = 128 tris.
        # Exact overlap band is Z = [0.90, 0.97, 1.04] -> 2 bands * 32 tris = 64 coplanar waist overlap tris!
        n_seg = 16
        upper_z = [0.90, 0.97, 1.04, 1.16, 1.28, 1.40]
        lower_z = [0.50, 0.70, 0.90, 0.97, 1.04]

        u_pos, u_tris, u_uvs = _cylinder_band(upper_z, n_seg=n_seg, split_islands=True)
        l_pos, l_tris, l_uvs = _cylinder_band(lower_z, n_seg=n_seg, split_islands=True, radial_offset=waist_radial_jitter)

        # Nipple patch overlaps 4 triangles on upper chest band (band index 3: Z = [1.16, 1.28], segments 0 and 8)
        nip_tri_ids = [3 * (2 * n_seg) + 0, 3 * (2 * n_seg) + 1, 3 * (2 * n_seg) + 16, 3 * (2 * n_seg) + 17]
        n_pos, n_tris = _extract_sub_patch(u_pos, u_tris, nip_tri_ids, normal_shift=patch_normal_shift)

        # Female genital patch overlaps 6 triangles on lower pelvis band (band index 1: Z = [0.70, 0.90])
        gen_tri_ids = [1 * (2 * n_seg) + k for k in range(6)]
        g_pos, g_tris = _extract_sub_patch(l_pos, l_tris, gen_tri_ids, normal_shift=patch_normal_shift)

        upper = pa.MeshPart(
            name="hand_fr1_000_nude_cull",
            positions=u_pos,
            triangles=u_tris,
            uvs=u_uvs,
            coord_space="z-up",
            material=pa.MaterialSpec(name="upper_skin", base_color_factor=(0.82, 0.65, 0.55, 1.0)),
        )
        lower = pa.MeshPart(
            name="feet_fr1_000_nude_cull",
            positions=l_pos,
            triangles=l_tris,
            uvs=l_uvs,
            coord_space="z-up",
            material=pa.MaterialSpec(name="lower_skin", base_color_factor=(0.82, 0.65, 0.55, 1.0)),
        )
        nipples = pa.MeshPart(
            name="bodynipplesgen_fr1_000_regular_ws00_full",
            positions=n_pos,
            triangles=n_tris,
            coord_space="z-up",
        )
        genitals = pa.MeshPart(
            name="bodygenitalsgen_fr1_008_regular_ws00_full",
            positions=g_pos,
            triangles=g_tris,
            coord_space="z-up",
        )
        return upper, lower, nipples, genitals

    def test_exact_waist_nipple_genital_culling_and_watertight_stitch(self):
        upper, lower, nipples, genitals = self._build_synthetic_body_parts(
            waist_radial_jitter=0.0,
            patch_normal_shift=0.0,
        )

        # Physical islands on split_islands upper mesh = 5 bands * 16 quads = 80 islands, welded = 1 island
        phys = pa.count_mesh_islands(upper.positions, upper.triangles, by_position=False)
        welded = pa.count_mesh_islands(upper.positions, upper.triangles, by_position=True)
        self.assertEqual(phys["island_count"], 80)
        self.assertEqual(welded["island_count"], 1)

        # Before culling: waist overlap produces non-manifold double-drawn edges
        before_body = pa.analyze_edge_topology([upper, lower], weld_tol=1e-4)
        self.assertGreater(before_body["non_manifold_edges"], 0)

        # Run standard 3-pass culling
        stitched, reports = pa.apply_standard_ped_culling([upper, lower, nipples, genitals])
        by_pass = {r["pass"]: r for r in reports}
        self.assertEqual(by_pass["upper_vs_lower_waist"]["culled_total"], 64)
        self.assertEqual(by_pass["upper_vs_nipples"]["culled_total"], 4)
        self.assertEqual(by_pass["lower_vs_female_genitals"]["culled_total"], 6)

        stitched_upper, stitched_lower, stitched_nip, stitched_gen = stitched
        self.assertEqual(len(stitched_upper.triangles), 160 - 64 - 4)
        self.assertEqual(len(stitched_lower.triangles), 128 - 6)
        self.assertEqual(len(stitched_nip.triangles), 4)
        self.assertEqual(len(stitched_gen.triangles), 6)

        # Watertight check across all 4 stitched parts at 0.1 mm weld tolerance:
        # zero non-manifold edges, and boundary edges equal ONLY the 16 top-cap +
        # 16 bottom-cap rim edges = 32 (zero unintended holes along any seam!).
        after_topo = pa.analyze_edge_topology(stitched, weld_tol=1e-4)
        self.assertEqual(after_topo["non_manifold_edges"], 0)
        self.assertEqual(after_topo["boundary_edges"], 32)

        # Re-inspecting seams must show 0 remaining pairwise coplanar overlaps
        after_seams = pa.inspect_seams(stitched, weld_tol=1e-4, overlap_tol=0.004)
        self.assertEqual(after_seams["pairwise_overlaps"], [])

    def test_submillimeter_perturbed_waist_and_patch_overlap_culling(self):
        # 0.8 mm radial offset on waist (< 1.5 mm measured RAGE 9 tolerance) and 0.5 mm shift on
        # nipple/genital patches fail exact vertex-key matching and are caught by spatial-hashed
        # coplanar proximity culling.
        upper, lower, nipples, genitals = self._build_synthetic_body_parts(
            waist_radial_jitter=0.0008,
            patch_normal_shift=0.0005,
        )
        stitched, reports = pa.apply_standard_ped_culling([upper, lower, nipples, genitals])
        by_pass = {r["pass"]: r for r in reports}
        self.assertEqual(by_pass["upper_vs_lower_waist"]["culled_exact_key"], 0)
        self.assertEqual(by_pass["upper_vs_lower_waist"]["culled_coplanar_proximity"], 64)
        self.assertEqual(by_pass["upper_vs_nipples"]["culled_total"], 4)
        self.assertEqual(by_pass["lower_vs_female_genitals"]["culled_total"], 6)
        self.assertEqual(len(stitched[0].triangles), 160 - 64 - 4)
        self.assertEqual(len(stitched[1].triangles), 128 - 6)

    def test_non_coplanar_nearby_triangles_are_preserved(self):
        # Reference triangle lies in XY plane (Z = 0, normal = +Z), centroid at (0.01, 0.01, 0.0)
        ref = pa.MeshPart(
            name="feet_fr1_000_nude_cull",
            positions=[(0.0, 0.0, 0.0), (0.03, 0.0, 0.0), (0.0, 0.03, 0.0)],
            triangles=[(0, 1, 2)],
            coord_space="z-up",
        )
        # Target contains:
        #   tri 0: coplanar duplicate (Z = 0.0005, normal = +Z) -> MUST BE CULLED
        #   tri 1: perpendicular fin sharing exact same centroid (normal = +Y) -> MUST BE PRESERVED
        #   tri 2: parallel offset layer 6 mm above (Z = 0.006, normal = +Z) -> MUST BE PRESERVED
        #   tri 3: tilted 45-deg wedge sharing exact same centroid -> MUST BE PRESERVED
        target = pa.MeshPart(
            name="hand_fr1_000_nude_cull",
            positions=[
                # tri 0 (coplanar)
                (0.0, 0.0, 0.0005), (0.03, 0.0, 0.0005), (0.0, 0.03, 0.0005),
                # tri 1 (perpendicular XZ plane passing through centroid (0.01, 0.01, 0.0))
                (0.0, 0.01, -0.01), (0.03, 0.01, -0.01), (0.0, 0.01, 0.02),
                # tri 2 (parallel cloth hovering 6 mm outside skin)
                (0.0, 0.0, 0.006), (0.03, 0.0, 0.006), (0.0, 0.03, 0.006),
                # tri 3 (tilted ~45 deg passing through centroid)
                (0.0, 0.0, -0.01), (0.03, 0.0, 0.01), (0.0, 0.03, 0.0),
            ],
            triangles=[(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)],
            coord_space="z-up",
        )
        culled, stats = pa.cull_part_overlap(
            target,
            [ref],
            centroid_tol=0.004,
            plane_tol=0.002,
            normal_cos_min=0.90,
        )
        self.assertEqual(stats["culled_total"], 1)
        self.assertEqual(culled.triangles, [(3, 4, 5), (6, 7, 8), (9, 10, 11)])

    def test_compact_vertices_remaps_attributes(self):
        part = pa.MeshPart(
            name="test_part",
            positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (5, 5, 5), (6, 5, 5), (5, 6, 5)],
            triangles=[(3, 4, 5)],
            uvs=[(0, 0), (1, 0), (0, 1), (0.5, 0.5), (0.6, 0.5), (0.5, 0.6)],
            coord_space="z-up",
        )
        compacted = pa.compact_part_vertices(part)
        self.assertEqual(len(compacted.positions), 3)
        self.assertEqual(compacted.triangles, [(0, 1, 2)])
        self.assertEqual(compacted.uvs, [(0.5, 0.5), (0.6, 0.5), (0.5, 0.6)])


class GlbExportAndCliTests(unittest.TestCase):
    def test_static_textured_glb_roundtrip(self):
        png = _make_1x1_png(220, 180, 150)
        part_a = pa.MeshPart(
            name="hand_fr1_000_nude_cull",
            positions=[(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.1)],
            triangles=[(0, 1, 2)],
            uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
            coord_space="z-up",
            material=pa.MaterialSpec(name="skin_upper", base_color_png=png, normal_png=png),
        )
        glb_bytes, stats = pa.build_assembled_glb([part_a], standing_offset=1.0, apply_standing=True)
        self.assertEqual(stats["parts"], 1)
        self.assertEqual(stats["triangles"], 1)
        self.assertEqual(stats["textures"], 1)  # deduplicated identical PNG bytes

        summary = glb_writer.read_glb_summary(glb_bytes)
        self.assertEqual(summary["version"], 2)
        self.assertEqual(len(summary["json"]["meshes"]), 1)
        self.assertEqual(len(summary["json"]["materials"]), 1)
        self.assertEqual(len(summary["json"]["images"]), 1)

        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "assembled.glb")
            pa.write_assembled_glb(out_path, [part_a], standing_offset=1.0, apply_standing=True)
            reloaded = pa.read_glb_parts(out_path)
            self.assertEqual(len(reloaded), 1)
            self.assertEqual(reloaded[0].coord_space, "y-up")
            # Original Z=0.0 + standing offset 1.0 -> Y=1.0 in glTF Y-up
            self.assertAlmostEqual(reloaded[0].positions[0][1], 1.0, places=5)
            self.assertAlmostEqual(reloaded[0].positions[2][1], 1.1, places=5)

    def test_skinned_glb_validates_with_glb_skinning(self):
        # 2-joint skeleton in RAGE Z-up with identity IBMs and 8-influence vertex weights
        skel = pa.SkeletonSpec(
            name="ambient_femalebody",
            coord_space="z-up",
            joints=[
                pa.SkeletonJoint(name="SKEL_ROOT", parent=-1, translation=(0.0, 0.0, 0.0)),
                pa.SkeletonJoint(name="SKEL_Pelvis", parent=0, translation=(0.0, 0.0, 0.02)),
            ],
        )
        part = pa.MeshPart(
            name="hand_fr1_000_nude_cull",
            positions=[(0.0, 0.0, -0.1), (0.1, 0.0, -0.1), (0.0, 0.0, 0.2)],
            triangles=[(0, 1, 2)],
            joints_0=[(0, 1, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0)],
            weights_0=[(0.5, 0.5, 0.0, 0.0), (0.25, 0.75, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
            joints_1=[(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
            weights_1=[(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)],
            coord_space="z-up",
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "skinned.glb")
            stats = pa.write_assembled_glb(out_path, [part], skeleton=skel, standing_offset=1.0)
            self.assertEqual(stats["skins"], 1)
            self.assertEqual(stats["joints"], 2)

            # Cross-validate the produced GLB with Leonida's strict glb_skinning validator!
            proc = subprocess.run(
                [sys.executable, "-I", os.path.join(TOOLS, "glb_skinning.py"), out_path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            # Also confirm standing-space root wrapper translation [0, 1.0, 0]
            with open(out_path, "rb") as f:
                summary = glb_writer.read_glb_summary(f.read())
            root_idx = summary["json"]["skins"][0]["skeleton"]
            self.assertEqual(summary["json"]["nodes"][root_idx]["translation"], [0.0, 1.0, 0.0])

    def test_cli_subcommands_end_to_end(self):
        script = os.path.join(TOOLS, "ped_assemble.py")
        # 1. --help
        r_help = subprocess.run(
            [sys.executable, "-I", script, "--help"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(r_help.returncode, 0, r_help.stderr)
        self.assertIn("assemble", r_help.stdout)
        self.assertIn("cull-overlap", r_help.stdout)
        self.assertIn("inspect-seams", r_help.stdout)

        with tempfile.TemporaryDirectory() as tmp:
            # Build overlapping synthetic upper + lower JSON parts
            u_pos, u_tris, u_uvs = _cylinder_band([0.90, 0.97, 1.04, 1.20], n_seg=8, split_islands=True)
            l_pos, l_tris, l_uvs = _cylinder_band([0.60, 0.90, 0.97, 1.04], n_seg=8, split_islands=True)
            upper_json = os.path.join(tmp, "hand_mr1_000_nude_cull.json")
            lower_json = os.path.join(tmp, "feet_mr1_000_nude_cull.json")
            with open(upper_json, "w", encoding="utf-8") as f:
                json.dump({"name": "hand_mr1_000_nude_cull", "positions": u_pos, "triangles": u_tris, "uvs": u_uvs}, f)
            with open(lower_json, "w", encoding="utf-8") as f:
                json.dump({"name": "feet_mr1_000_nude_cull", "positions": l_pos, "triangles": l_tris, "uvs": l_uvs}, f)

            # 2. inspect-seams
            seam_report = os.path.join(tmp, "seams.json")
            r_seam = subprocess.run(
                [sys.executable, "-I", script, "inspect-seams", upper_json, lower_json, "--json", seam_report],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(r_seam.returncode, 0, r_seam.stderr)
            with open(seam_report, encoding="utf-8") as f:
                sdata = json.load(f)
            self.assertEqual(sdata["part_count"], 2)
            self.assertEqual(len(sdata["pairwise_overlaps"]), 1)
            self.assertEqual(sdata["pairwise_overlaps"][0]["overlapping_triangles_in_a"], 32)

            # 3. cull-overlap
            culled_json = os.path.join(tmp, "culled_upper.json")
            cull_stats = os.path.join(tmp, "cull_stats.json")
            r_cull = subprocess.run(
                [
                    sys.executable, "-I", script, "cull-overlap",
                    "--target", upper_json, "--ref", lower_json,
                    "-o", culled_json, "--json", cull_stats, "--compact-vertices",
                ],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(r_cull.returncode, 0, r_cull.stderr)
            with open(cull_stats, encoding="utf-8") as f:
                cdata = json.load(f)
            self.assertEqual(cdata["culled_total"], 32)
            self.assertEqual(cdata["target_triangles_after"], 16)

            # 4. assemble -> GLB + report (also test reading back OBJ + GLB inputs)
            out_glb = os.path.join(tmp, "ped.glb")
            asm_report = os.path.join(tmp, "asm.json")
            r_asm = subprocess.run(
                [
                    sys.executable, "-I", script, "assemble",
                    upper_json, lower_json, "-o", out_glb, "--json", asm_report,
                ],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(r_asm.returncode, 0, r_asm.stderr)
            with open(asm_report, encoding="utf-8") as f:
                adata = json.load(f)
            self.assertEqual(adata["after_seams"]["combined_topology"]["non_manifold_edges"], 0)
            self.assertEqual(adata["after_seams"]["pairwise_overlaps"], [])
            self.assertEqual(adata["glb"]["triangles"], 16 + 48)
            self.assertTrue(os.path.isfile(out_glb))

            # 5. Re-inspect the assembled GLB directly via inspect-seams
            r_glb_seam = subprocess.run(
                [sys.executable, "-I", script, "inspect-seams", out_glb, "--source-space", "y-up"],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(r_glb_seam.returncode, 0, r_glb_seam.stderr)
            self.assertIn("non_manifold=0", r_glb_seam.stdout)
            self.assertIn("overlapping_pairs=0", r_glb_seam.stdout)

    def test_obj_manifest_and_error_paths(self):
        script = os.path.join(TOOLS, "ped_assemble.py")
        with tempfile.TemporaryDirectory() as tmp:
            obj_path = os.path.join(tmp, "feet_fr1_000_nude_cull.obj")
            with open(obj_path, "w", encoding="utf-8") as f:
                f.write("o feet_fr1_000_nude_cull\n")
                f.write("v 0.0 0.0 -1.0\n")
                f.write("v 0.2 0.0 -1.0\n")
                f.write("v 0.0 0.2 -0.8\n")
                f.write("f 1 2 3\n")

            manifest_path = os.path.join(tmp, "ped_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump({
                    "coord_space": "z-up",
                    "parts": [{"path": "feet_fr1_000_nude_cull.obj", "name": "feet_fr1_000_nude_cull"}],
                }, f)

            out_glb = os.path.join(tmp, "from_manifest.glb")
            r_asm = subprocess.run(
                [sys.executable, "-I", script, "assemble", "--manifest", manifest_path, "-o", out_glb],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(r_asm.returncode, 0, r_asm.stderr)
            reloaded = pa.read_glb_parts(out_glb)
            self.assertEqual(len(reloaded), 1)
            # Z = -1.0 + 1.0 standing offset -> Y = 0.0 on ground plane
            self.assertAlmostEqual(reloaded[0].positions[0][1], 0.0, places=5)

            # Unsupported extension exits cleanly with status 2
            bad_path = os.path.join(tmp, "bad.xyz")
            with open(bad_path, "w", encoding="utf-8") as f:
                f.write("x")
            r_bad = subprocess.run(
                [sys.executable, "-I", script, "inspect-seams", bad_path],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(r_bad.returncode, 2)
            self.assertIn("ERROR:", r_bad.stderr)


if __name__ == "__main__":
    unittest.main()
