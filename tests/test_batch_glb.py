"""Unit tests for tools/batch_glb.py batch OBJ -> glTF 2.0 (.glb) converter.

Usage:
    python -I tests/test_batch_glb.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import batch_glb
import dds_png


def _create_minimal_bc1_dds(color: int = 0xF800) -> bytes:
    hdr = bytearray(128)
    hdr[0:4] = b"DDS "
    struct.pack_into("<I", hdr, 4, 124)
    struct.pack_into("<I", hdr, 8, 0x1007)
    struct.pack_into("<I", hdr, 12, 2)
    struct.pack_into("<I", hdr, 16, 2)
    struct.pack_into("<I", hdr, 20, 8)
    struct.pack_into("<I", hdr, 76, 32)
    struct.pack_into("<I", hdr, 80, 0x4)
    hdr[84:88] = b"DXT1"
    struct.pack_into("<I", hdr, 108, 0x1000)
    bc1_block = struct.pack("<HHBBBB", color, 0x07E0, 0, 0, 0, 0)
    return bytes(hdr) + bc1_block


class TestBatchGlb(unittest.TestCase):
    def test_cli_reports_directory_setup_errors(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            missing = os.path.join(root, "missing")
            blocked = os.path.join(root, "blocked")
            with open(blocked, "wb"):
                pass

            for source, output, error in [
                (missing, os.path.join(root, "out"), "Input directory does not exist"),
                (root, blocked, "blocked"),
            ]:
                with self.subTest(source=source, output=output):
                    result = subprocess.run(
                        [sys.executable, "-I", os.path.join(TOOLS, "batch_glb.py"), source, output],
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(result.returncode, 1)
                    self.assertTrue(result.stderr.startswith("ERROR: "), result.stderr)
                    self.assertIn(error, result.stderr)
                    self.assertEqual(len(result.stderr.splitlines()), 1)

    def test_batch_convert_synthetic(self) -> None:
        with tempfile.TemporaryDirectory() as in_dir, tempfile.TemporaryDirectory() as out_dir:
            for i in range(3):
                obj_path = os.path.join(in_dir, f"test_model_{i}.obj")
                with open(obj_path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(
                        "o test\n"
                        "v 0.0 0.0 0.0\n"
                        "v 1.0 0.0 0.0\n"
                        "v 0.0 1.0 0.0\n"
                        "f 1 2 3\n"
                    )

            stats = batch_glb.batch_convert(in_dir, out_dir, up_axis="z", jobs=2)
            self.assertEqual(stats["total"], 3)
            self.assertEqual(stats["succeeded"], 3)
            self.assertEqual(stats["failed"], 0)

            for i in range(3):
                glb_file = os.path.join(out_dir, f"test_model_{i}.glb")
                self.assertTrue(os.path.isfile(glb_file))
                self.assertGreater(os.path.getsize(glb_file), 100)

    def test_batch_convert_with_png_texture(self) -> None:
        with tempfile.TemporaryDirectory() as in_dir, \
             tempfile.TemporaryDirectory() as out_dir, \
             tempfile.TemporaryDirectory() as tex_dir:
            # Create a 2x2 PNG texture
            dds_data = _create_minimal_bc1_dds()
            png_bytes = dds_png.dds_to_png_bytes(dds_data)
            tex_file = os.path.join(tex_dir, "mat_custom.png")
            with open(tex_file, "wb") as f:
                f.write(png_bytes)

            # Create an OBJ with UVs and material reference
            obj_path = os.path.join(in_dir, "textured_cube.obj")
            with open(obj_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(
                    "o textured_cube\n"
                    "usemtl mat_custom\n"
                    "v 0.0 0.0 0.0\n"
                    "v 1.0 0.0 0.0\n"
                    "v 0.0 1.0 0.0\n"
                    "vt 0.0 0.0\n"
                    "vt 1.0 0.0\n"
                    "vt 0.0 1.0\n"
                    "f 1/1 2/2 3/3\n"
                )

            stats = batch_glb.batch_convert(
                in_dir, out_dir, up_axis="y", texture_dir=tex_dir
            )
            self.assertEqual(stats["total"], 1)
            self.assertEqual(stats["succeeded"], 1)
            self.assertEqual(stats["failed"], 0)

            glb_file = os.path.join(out_dir, "textured_cube.glb")
            self.assertTrue(os.path.isfile(glb_file))
            with open(glb_file, "rb") as f:
                content = f.read()
            self.assertIn(b"image/png", content)
            self.assertIn(b"mat_custom", content)

    def test_unnamed_primitive_keeps_its_own_texture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            in_dir, out_dir, tex_dir = root / "obj", root / "glb", root / "textures"
            in_dir.mkdir()
            tex_dir.mkdir()
            (in_dir / "panel.detail.obj").write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
                "vt 0 0\nvt 1 0\nvt 0 1\n"
                "f 1/1 2/2 3/3\nusemtl panel\nf 1/1 2/2 3/3\n",
                encoding="utf-8",
            )
            red_png = dds_png.dds_to_png_bytes(_create_minimal_bc1_dds())
            blue_png = dds_png.dds_to_png_bytes(_create_minimal_bc1_dds(0x001F))
            (tex_dir / "panel.detail.png").write_bytes(red_png)
            (tex_dir / "panel.png").write_bytes(blue_png)

            stats = batch_glb.batch_convert(str(in_dir), str(out_dir), texture_dir=str(tex_dir), jobs=1)
            self.assertEqual(stats, {"total": 1, "succeeded": 1, "failed": 0})
            glb = (out_dir / "panel.detail.glb").read_bytes()
            json_size, = struct.unpack_from("<I", glb, 12)
            gltf = json.loads(glb[20:20 + json_size])
            binary = glb[28 + json_size:]
            textures = []
            for primitive in gltf["meshes"][0]["primitives"]:
                material = gltf["materials"][primitive["material"]]
                texture = material["pbrMetallicRoughness"]["baseColorTexture"]["index"]
                image = gltf["images"][gltf["textures"][texture]["source"]]
                view = gltf["bufferViews"][image["bufferView"]]
                offset = view.get("byteOffset", 0)
                textures.append(binary[offset:offset + view["byteLength"]])

            self.assertEqual(textures, [red_png, blue_png])

    def test_batch_convert_with_dds_texture(self) -> None:
        with tempfile.TemporaryDirectory() as in_dir, \
             tempfile.TemporaryDirectory() as out_dir, \
             tempfile.TemporaryDirectory() as tex_dir:
            # Create a DDS texture directly
            dds_data = _create_minimal_bc1_dds()
            tex_file = os.path.join(tex_dir, "car_body.dds")
            with open(tex_file, "wb") as f:
                f.write(dds_data)

            obj_path = os.path.join(in_dir, "car_body.obj")
            with open(obj_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(
                    "o car_body\n"
                    "v 0.0 0.0 0.0\n"
                    "v 1.0 0.0 0.0\n"
                    "v 0.0 1.0 0.0\n"
                    "vt 0.0 0.0\n"
                    "vt 1.0 0.0\n"
                    "vt 0.0 1.0\n"
                    "f 1/1 2/2 3/3\n"
                )

            stats = batch_glb.batch_convert(
                in_dir, out_dir, up_axis="z", texture_dir=tex_dir
            )
            self.assertEqual(stats["total"], 1)
            self.assertEqual(stats["succeeded"], 1)

            glb_file = os.path.join(out_dir, "car_body.glb")
            self.assertTrue(os.path.isfile(glb_file))
            with open(glb_file, "rb") as f:
                content = f.read()
            self.assertIn(b"image/png", content)


if __name__ == "__main__":
    unittest.main()
