"""Package-gated first-light test: draw the real model a first_light.py run assembled.

tools/first_light.py writes a directory with vs.spv, ps.spv, mesh.bin and manifest.json from the user's own
package. This test runs vk_harness on that directory and checks that the draw produced the mesh (coverage and a
covered centre), not just a valid pipeline. It needs no game files itself and skips when the directory is unset:

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


def harness() -> str | None:
    for path in (
        os.environ.get("LEONIDA_VK_HARNESS"),
        os.path.join(ROOT, "runtime", "build", "linux", "vk_harness"),
    ):
        if path and os.path.isfile(path):
            return path
    return None


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
                "vk_harness is not built (runtime/build/linux, target vk_harness)"
            )
        with open(os.path.join(self.out, "manifest.json"), encoding="utf-8") as f:
            self.manifest = json.load(f)

    def test_draw_covers_the_model(self):
        raw = os.path.join(self.out, "image_first_light.raw")
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
        with open(raw, "rb") as f:
            data = f.read()
        pixels = struct.unpack(f"<{len(data) // 4}f", data)
        width, height = self.manifest["width"], self.manifest["height"]
        self.assertEqual(len(pixels), width * height * 4)
        covered = sum(1 for i in range(3, len(pixels), 4) if pixels[i] > 0.5)
        ratio = covered / (width * height)
        self.assertGreaterEqual(
            ratio,
            self.manifest["min_coverage"],
            f"only {covered}/{width * height} pixels were written",
        )
        centre = (height // 2 * width + width // 2) * 4
        self.assertAlmostEqual(
            pixels[centre + 3], 1.0, places=3, msg="centre pixel was not written"
        )
        self.assertNotEqual(
            list(pixels[centre : centre + 3]),
            [-1.0, -1.0, -1.0],
            "centre is the clear value",
        )
        # A PPM for the user's own inspection; it is derived from their copy and stays in their work folder.
        with open(os.path.join(self.out, "image_first_light.ppm"), "wb") as f:
            f.write(f"P6\n{width} {height}\n255\n".encode())
            f.writelines(
                bytes(max(0, min(255, round(pixels[i + c] * 255))) for c in range(3))
                for i in range(0, len(pixels), 4)
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


if __name__ == "__main__":
    sys.exit(unittest.main())
