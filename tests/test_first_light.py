"""Package-gated first-light test: draw the real model a first_light.py run assembled.

tools/first_light.py writes a directory with vs.spv, ps.spv, mesh.bin and manifest.json from the user's own
package. This test runs vk_harness on that directory and checks the pixels against the manifest: a coverage
window, a corner the mesh does not cover (the harness clear value) and the UV the model's last triangle writes at
the centre pixel. The oracle is also exercised on synthetic images, without a GPU or a package, so the
all-opaque and swapped-UV negatives stay red.

    LEONIDA_FIRST_LIGHT_DIR=~/leonida-work/first_light/out python -I tests/test_first_light.py

The image is written to the directory as image_first_light.raw/.ppm; no game data is in the repository.
"""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import first_light as fl
from spirv_stubs import build_ps_stub, build_vs_stub

CLEAR = fl.CLEAR
check_image = fl.check_image
write_ppm = fl.write_ppm


def harness() -> str | None:
    exe = ".exe" if os.name == "nt" else ""
    for path in (
        os.environ.get("LEONIDA_VK_HARNESS"),
        os.path.join(ROOT, "runtime", "build", "linux", f"vk_harness{exe}"),
        os.path.join(ROOT, "runtime", "build", "windows", f"vk_harness{exe}"),
        os.path.join(ROOT, "runtime", "build", f"vk_harness{exe}"),
    ):
        if path and os.path.isfile(path):
            return path
    return None


class MeshOutputTests(unittest.TestCase):
    def test_fit_keeps_relative_depth_for_an_offset_mesh(self) -> None:
        import tempfile

        vertices = [(-2.0, -1.0, 900.0, 0.0, 0.0),
                    (2.0, -1.0, 901.0, 1.0, 0.0),
                    (0.0, 1.0, 902.0, 0.5, 1.0)]
        fitted = fl.fit(vertices, 128, 128, True, False)
        with tempfile.TemporaryDirectory() as directory:
            mesh = os.path.join(directory, "mesh.bin")
            fl.write_mesh(mesh, fitted, [0, 1, 2])
            with open(mesh, "rb") as handle:
                encoded = handle.read()

        records = encoded[fl.MESH_HEADER.size:fl.MESH_HEADER.size + 3 * 20]
        depth = b"".join(records[offset + 8:offset + 12] for offset in (0, 20, 40))
        self.assertEqual(depth, struct.pack("<3f", 0.05, 0.5, 0.95))
        self.assertEqual(encoded[-12:], struct.pack("<3I", 0, 1, 2))


class FirstLightTests(unittest.TestCase):
    def setUp(self):
        self.out = os.environ.get("LEONIDA_FIRST_LIGHT_DIR")
        if not self.out:
            self.skipTest(
                "LEONIDA_FIRST_LIGHT_DIR is not set (run tools/first_light.py first)"
            )
        self.out = os.path.abspath(os.path.expanduser(self.out))
        self.harness = harness()
        if not self.harness:
            self.skipTest(
                "vk_harness is not built (runtime/build/linux or runtime/build/windows, target "
                "vk_harness); set LEONIDA_VK_HARNESS"
            )
        with open(os.path.join(self.out, "manifest.json"), encoding="utf-8") as f:
            self.manifest = json.load(f)

    def test_draw_matches_the_manifest(self):
        raw = os.path.join(self.out, "image_first_light.raw")
        if os.path.exists(raw):
            os.unlink(
                raw
            )  # a fresh run: a harness that writes nothing cannot pass on a stale file
        env = fl.harness_env(self.manifest, self.out)
        subprocess.run(
            [
                self.harness,
                os.path.join(self.out, "vs.spv"),
                os.path.join(self.out, "ps.spv"),
                raw,
                str(self.manifest["width"]),
                str(self.manifest["height"]),
            ],
            check=True,
            env=env,
        )
        self.assertTrue(os.path.isfile(raw), "the harness did not write a fresh image")
        with open(raw, "rb") as f:
            data = f.read()
        stats = check_image(data, self.manifest)
        write_ppm(
            os.path.join(self.out, "image_first_light.ppm"),
            stats["pixels"],
            self.manifest["width"],
            self.manifest["height"],
        )
        self.assertGreater(stats["covered"], 0)

    def test_check_cli_matches_the_oracle(self):
        """The tool's --check must accept and reject the same images the test does."""
        subprocess.run(
            [
                sys.executable,
                "-I",
                os.path.join(ROOT, "tools", "first_light.py"),
                "--check",
                "--out",
                self.out,
                "--harness",
                self.harness,
            ],
            check=True,
        )

    def test_manifest_names_the_assets(self):
        self.assertGreater(self.manifest["vertex_count"], 0)
        self.assertGreater(self.manifest["triangle_count"], 0)
        for key in ("vs", "ps"):
            self.assertTrue(
                self.manifest[key].get("fssm_sha256"), f"{key} has no FSSM hash"
            )
            self.assertTrue(
                os.path.isfile(os.path.join(self.out, self.manifest["files"][key]))
            )


class PpmTests(unittest.TestCase):
    def test_nonfinite_readback_channels_write_complete_ppm(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        pixels = (float("nan"), float("inf"), float("-inf"), 1.0,
                  -0.25, 0.5, 1.25, 1.0)
        with TemporaryDirectory() as temp:
            path = Path(temp) / "preview.ppm"
            write_ppm(str(path), pixels, 2, 1)
            self.assertEqual(path.read_bytes(), b"P6\n2 1\n255\n" + bytes((0, 255, 0, 0, 128, 255)))


class ShaderPreparationTests(unittest.TestCase):
    def test_same_display_name_keeps_both_stage_files(self) -> None:
        modules = {"vs": build_vs_stub([0]), "ps": build_ps_stub([0])}
        entries = [
            {"name": "shared", "file": "shared.dxbc", "stage": "vs"},
            {"name": "shared", "file": "shared~1.dxbc", "stage": "ps"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            dxbc = work / "shaders" / "dxbc"
            dxbc.mkdir(parents=True)
            (dxbc / "index.json").write_text(json.dumps(entries), encoding="utf-8", newline="\n")
            for entry in entries:
                (dxbc / entry["file"]).write_bytes(entry["stage"].encode("ascii"))

            def write_translation(argv: list[str]) -> None:
                if Path(argv[0]).name != "rdna2spirv.py":
                    return
                staged = work / "first_light" / "dxbc"
                picked = json.loads((staged / "index.json").read_text(encoding="utf-8"))
                self.assertEqual(picked, entries)
                spirv = Path(argv[argv.index("--out") + 1])
                spirv.mkdir()
                index = {}
                for entry in picked:
                    stage = entry["stage"]
                    self.assertEqual((staged / entry["file"]).read_bytes(), stage.encode("ascii"))
                    stem = Path(entry["file"]).stem
                    (spirv / f"{stem}.spv").write_bytes(modules[stage])
                    index[stem] = {"stage": stage, "fssm_sha256": stage + "-fixture"}
                (spirv / "index.json").write_text(json.dumps(index), encoding="utf-8", newline="\n")

            with patch.object(fl, "run", side_effect=write_translation), redirect_stdout(io.StringIO()):
                self.assertEqual(fl.main(["--game", tmp, "--work", tmp, "--no-fit",
                                          "--vs", "shared", "--ps", "shared"]), 0)
            out = work / "first_light" / "out"
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            runtime_index = json.loads((out / "index.json").read_text(encoding="utf-8"))
            for stage, module in modules.items():
                self.assertEqual((out / manifest["files"][stage]).read_bytes(), module)
                self.assertEqual((out / f"{stage}.dxbc").read_bytes(), stage.encode("ascii"))
                self.assertEqual(runtime_index[stage]["fssm_sha256"], stage + "-fixture")
                self.assertEqual(manifest[stage]["fssm_sha256"], stage + "-fixture")


class OracleTests(unittest.TestCase):
    """The readback oracle on synthetic images: the controls run without a GPU or a game package."""

    def setUp(self):
        self.manifest = {
            "width": 4,
            "height": 4,
            "min_coverage": 0.2,
            "max_coverage": 0.5,
            "centre_pixel": [2, 2],
            "centre_uv": [0.25, 0.75],
            "uv_probes": [{"pixel": [2, 2], "uv": [0.25, 0.75]}],
            "clear_pixel": [0, 0],
            "clear": CLEAR,
        }

    def image(self, centre=(0.25, 0.75), fill=None) -> bytes:
        pixels = [[CLEAR] * 4 for _ in range(16)]
        if fill is not None:
            pixels = [[fill, fill, fill, 1.0] for _ in range(16)]
        else:
            for x, y in ((1, 1), (2, 1), (1, 2), (2, 2)):
                pixels[y * 4 + x] = [centre[0], centre[1], 0.0, 1.0]
        return struct.pack("<64f", *(v for pixel in pixels for v in pixel))

    def test_accepts_the_expected_image(self):
        check_image(self.image(), self.manifest)

    def test_rejects_an_all_opaque_image(self):
        with self.assertRaises(AssertionError):
            check_image(self.image(fill=0.5), self.manifest)

    def test_rejects_a_swapped_uv(self):
        with self.assertRaises(AssertionError):
            check_image(self.image(centre=(0.75, 0.25)), self.manifest)

    def test_rejects_a_short_image(self):
        with self.assertRaises(AssertionError):
            check_image(b"", self.manifest)


class ReadObjTests(unittest.TestCase):
    """Wavefront OBJ parsing in tools/first_light.py without a GPU or game package."""

    def _write_obj(self, text: str) -> str:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "mesh.obj")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        return path

    def test_negative_relative_indices_inline_comments_and_continuations(self):
        positive = self._write_obj(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
            "vt 0.1 0.2\nvt 0.3 0.4\nvt 0.5 0.6\n"
            "f 1/1 2/2 3/3\n"
        )
        relative = self._write_obj(
            "v 0 0 0 # origin\n"
            "v 1 0 0\n"
            "v 0 1 0\n"
            "vt 0.1 0.2 # uv0\n"
            "vt 0.3 0.4\n"
            "vt 0.5 0.6\n"
            "f -3/-3 \\\n"
            "  -2/-2 -1/-1 # trailing comment\n"
        )
        self.assertEqual(fl.read_obj(relative), fl.read_obj(positive))

    def test_rejects_zero_and_out_of_range_vertex_indices(self):
        for bad in (
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n",
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4\n",
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -4 -2 -1\n",
            "v 0 0\nf 1 1 1\n",
        ):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                fl.read_obj(self._write_obj(bad))


class TexturedOracleTests(unittest.TestCase):
    """Synthetic tests for the textured first-light path (TEST-524): no GPU or game files."""

    def setUp(self):
        self.manifest = {
            "width": 4,
            "height": 4,
            "min_coverage": 0.2,
            "max_coverage": 0.5,
            "centre_pixel": [2, 2],
            "centre_rgba": [0.2, 0.4, 0.6, 1.0],
            "rgba_probes": [{"pixel": [1, 1], "uv": [0.25, 0.75], "rgba": [0.8, 0.2, 0.1, 1.0]}],
            "rgba_tolerance": 0.02,
            "clear_pixel": [0, 0],
            "clear": CLEAR,
            "files": {"mesh": "mesh.bin", "texture": "texture.rgba"},
            "ps": {"subgroup_size": 32},
            "texture": {
                "width": 2,
                "height": 2,
                "ptr_sgpr": 9,
                "tsharp_va": 0x100,
                "ssharp": [0x92, 0xC00000, 0x14500000, 0],
            },
        }

    def image(self, centre=(0.2, 0.4, 0.6, 1.0), probe=(0.8, 0.2, 0.1, 1.0)) -> bytes:
        pixels = [[CLEAR] * 4 for _ in range(16)]
        for x, y in ((1, 1), (2, 1), (1, 2), (2, 2)):
            pixels[y * 4 + x] = list(centre)
        pixels[1 * 4 + 1] = list(probe)
        return struct.pack("<64f", *(v for pixel in pixels for v in pixel))

    def test_accepts_expected_textured_image(self):
        check_image(self.image(), self.manifest)

    def test_rejects_magenta_fallback_texture(self):
        with self.assertRaises(AssertionError):
            check_image(self.image(centre=(1.0, 0.0, 1.0, 1.0)), self.manifest)

    def test_rejects_wrong_textured_probe(self):
        with self.assertRaises(AssertionError):
            check_image(self.image(probe=(0.1, 0.2, 0.8, 1.0)), self.manifest)

    def test_sample_rgba_bilinear(self):
        rgba = bytes(
            [
                255, 0, 0, 255,    0, 255, 0, 255,
                0, 0, 255, 255,    255, 255, 0, 255,
            ]
        )
        # Centre (0.5, 0.5) averages all 4 texels equally -> (0.5, 0.5, 0.25, 1.0).
        mid = fl.sample_rgba_bilinear(2, 2, rgba, 0.5, 0.5)
        for got, want in zip(mid, (0.5, 0.5, 0.25, 1.0)):
            self.assertAlmostEqual(got, want, places=4)
        # Clamped corner (0.0, 0.0) hits texel (0, 0) -> red.
        tl = fl.sample_rgba_bilinear(2, 2, rgba, -0.5, -0.5)
        for got, want in zip(tl, (1.0, 0.0, 0.0, 1.0)):
            self.assertAlmostEqual(got, want, places=4)

    def test_harness_env_sets_texture_vars(self):
        env = fl.harness_env(self.manifest, "/tmp/fl_out", base_env={})
        self.assertEqual(env["VKH_TSHARP_VA"], "0x100")
        self.assertEqual(env["VKH_SSHARP"], "0x92,0xc00000,0x14500000,0x0")
        sgprs = env["VKH_SGPRS"].split(",")
        self.assertEqual(len(sgprs), 10)
        self.assertEqual(sgprs[9], "0x100")
        self.assertEqual(env["VKH_SUBGROUP_SIZE"], "32")
        self.assertTrue(env["VKH_TEX_RGBA"].endswith("texture.rgba"))

    def test_harness_rejects_wrapped_tsharp_va_overflow(self):
        h = harness()
        if not h:
            self.skipTest("vk_harness is not built")
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            rgba_path = os.path.join(td, "tex.rgba")
            fl.write_texture_rgba(rgba_path, 1, 1, bytes([255, 0, 0, 255]))
            env = dict(
                os.environ,
                VKH_TEX_RGBA=rgba_path,
                VKH_TSHARP_VA="0xFFFFFFE0",
            )
            proc = subprocess.run(
                [h, "unused_vs.spv", "unused_ps.spv", os.path.join(td, "out.raw"), "4", "4"],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("VKH_TSHARP_VA", proc.stderr)

    def test_decode_synthetic_dds_and_write_rgba(self):
        import tempfile

        # Minimal valid 2x2 uncompressed R8G8B8A8 DDS (128-byte header + 16 bytes pixels).
        dds = bytearray(128)
        dds[0:4] = b"DDS "
        struct.pack_into("<I", dds, 4, 124)       # dwSize
        struct.pack_into("<I", dds, 8, 0x1007)    # DDSD_CAPS|HEIGHT|WIDTH|PIXELFORMAT
        struct.pack_into("<I", dds, 12, 2)        # height
        struct.pack_into("<I", dds, 16, 2)        # width
        struct.pack_into("<I", dds, 76, 32)       # ddspf.dwSize
        struct.pack_into("<I", dds, 80, 0x41)     # DDPF_RGB | DDPF_ALPHAPIXELS
        struct.pack_into("<I", dds, 88, 32)       # RGBBitCount
        struct.pack_into("<IIII", dds, 92, 0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000)
        struct.pack_into("<I", dds, 108, 0x1000)  # DDSCAPS_TEXTURE
        pixels = bytes([10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255, 100, 110, 120, 255])
        dds.extend(pixels)
        with tempfile.TemporaryDirectory() as td:
            dds_path = os.path.join(td, "tex.dds")
            rgba_path = os.path.join(td, "tex.rgba")
            with open(dds_path, "wb") as f:
                f.write(dds)
            w, h, rgba, sha = fl.decode_dds_to_rgba(dds_path)
            self.assertEqual((w, h), (2, 2))
            self.assertEqual(rgba, pixels)
            self.assertEqual(len(sha), 64)
            fl.write_texture_rgba(rgba_path, w, h, rgba)
            with open(rgba_path, "rb") as f:
                blob = f.read()
            self.assertEqual(blob[:8], struct.pack("<II", 2, 2))
            self.assertEqual(blob[8:], pixels)


if __name__ == "__main__":
    sys.exit(unittest.main())
