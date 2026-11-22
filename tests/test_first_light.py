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
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLEAR = -1.0


def harness() -> str | None:
    for path in (
        os.environ.get("LEONIDA_VK_HARNESS"),
        os.path.join(ROOT, "runtime", "build", "linux", "vk_harness"),
    ):
        if path and os.path.isfile(path):
            return path
    return None


def check_image(data: bytes, manifest: dict) -> dict:
    """The readback oracle: exact shape, coverage window, a clear corner and the centre UV."""
    width, height = manifest["width"], manifest["height"]
    if len(data) != width * height * 16:
        raise AssertionError(
            f"raw is {len(data)} bytes, expected {width * height * 16}"
        )
    pixels = struct.unpack(f"<{len(data) // 4}f", data)
    covered = sum(1 for i in range(3, len(pixels), 4) if pixels[i] > 0.5)
    ratio = covered / (width * height)
    if (
        not manifest.get("min_coverage", 0.0)
        <= ratio
        <= manifest.get("max_coverage", 1.0)
    ):
        raise AssertionError(
            f"coverage {ratio:.4f} outside "
            f"[{manifest.get('min_coverage', 0.0)}, {manifest.get('max_coverage', 1.0)}]"
        )
    if (clear := manifest.get("clear_pixel")) is not None:
        i = (clear[1] * width + clear[0]) * 4
        want = manifest.get("clear", CLEAR)
        if any(abs(pixels[i + c] - want) > 1e-5 for c in range(4)):
            raise AssertionError(
                f"clear pixel {clear} is {pixels[i : i + 4]}, expected {want}"
            )
    if (uv := manifest.get("centre_uv")) is not None:
        centre = manifest.get("centre_pixel") or [width // 2, height // 2]
        i = (centre[1] * width + centre[0]) * 4
        for c, want in enumerate((uv[0], uv[1], 0.0, 1.0)):
            if abs(pixels[i + c] - want) > 2e-3:
                raise AssertionError(
                    f"centre {centre} is {pixels[i : i + 4]}, expected {want} at {c}"
                )
    return {"covered": covered, "ratio": ratio, "pixels": pixels}


def write_ppm(path: str, pixels, width: int, height: int) -> None:
    with open(path, "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode())
        f.writelines(
            bytes(max(0, min(255, round(pixels[i + c] * 255))) for c in range(3))
            for i in range(0, len(pixels), 4)
        )


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

    def test_draw_matches_the_manifest(self):
        fd, raw = tempfile.mkstemp(prefix="first_light_", suffix=".raw", dir=self.out)
        os.close(fd)
        os.unlink(
            raw
        )  # a fresh path: a harness that writes nothing cannot pass on a stale file
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
