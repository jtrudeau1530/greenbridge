"""Package-gated first-light test: draw the real model a first_light.py run assembled.

tools/first_light.py writes a directory with vs.spv, ps.spv, mesh.bin and manifest.json from the user's own
package. This test runs vk_harness on that directory and checks the pixels against the manifest: a coverage
window, a corner the mesh does not cover (the harness clear value) and the UV the model's last triangle writes at
the centre pixel. The oracle is also exercised on synthetic images, without a GPU or a package, so the
all-opaque and swapped-UV negatives stay red.

    LEONIDA_FIRST_LIGHT_DIR=~/leonida-work/first_light/out python -I tests/test_first_light.py

The image is written to the directory as image_first_light.raw/.ppm; no game data is in the repository.
"""

import json
import os
import struct
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import first_light as fl

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
        env = dict(
            os.environ,
            VKH_DRAW_MESH=os.path.join(self.out, "mesh.bin"),
            VKH_VERTEX_LAYOUT="0:3f,1:2f",
        )
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


if __name__ == "__main__":
    sys.exit(unittest.main())
