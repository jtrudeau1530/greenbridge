"""Wave64 mesh fixtures require a confirmed producer size; unsupported hardware is not a pixel failure."""
import contextlib
import io
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_gpu_tests as runner
import check


def pixels(good=True):
    return b"".join(struct.pack("<4f", (x + .5) / 64, (y + .5) / 64, .5 if good else 0, 1)
                    for y in range(64) for x in range(64))


class Wave64MeshContract(unittest.TestCase):
    def suite(self, name, *, code=0, message="mesh subgroup size: 64\n", good=True, emulated=False):
        seen = []
        def command(cmd, env=None, **_kw):
            if cmd[0] == sys.executable:
                out = Path(cmd[cmd.index("--out") + 1])
                for shader in (name, "fs_passthru"):
                    (out / (shader + ".spv")).touch()
                return 0, ""
            seen.append(dict(env))
            if code == 0:
                Path(cmd[-1]).write_bytes(pixels(good))
            return code, message

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as output:
            args = SimpleNamespace(filter=name, llvm_mc="llvm-mc", build=tmp)
            with (patch.object(runner, "assemble_to_s", side_effect=lambda _m, _s, dst, _n: Path(dst).touch()),
                  patch.object(runner, "spirv_as"), patch.object(runner, "run_cmd", side_effect=command),
                  patch.object(runner, "mesh_env", return_value={"VKH_MESH": "1"}),
                  patch.object(runner, "w64_emulated", return_value=emulated)):
                result = runner.run_suite(args, tmp)
            return result, output.getvalue(), seen

    def test_both_wave64_cases_request_and_check_pixels(self):
        for name in ("ms_w64x_ballot", "ms_wave_info_s3"):
            with self.subTest(name=name):
                rc, out, envs = self.suite(name)
                self.assertEqual(rc, 0, out)
                self.assertIn("1/1 passed", out)
                self.assertEqual(envs, [{"VKH_MESH": "1", "VKH_SUBGROUP_SIZE": "64", "VKH_REQUIRE_MESH_SUBGROUP_SIZE": "1"}])
                rc, out, _ = self.suite(name, good=False)
                self.assertEqual(rc, 1, out)
                self.assertIn("0/1 passed", out)

    def test_unavailable_size_is_an_explicit_skip(self):
        for name in ("ms_w64x_ballot", "ms_wave_info_s3"):
            with self.subTest(name=name):
                rc, out, _ = self.suite(name, code=77,
                    message="unsupported: mesh producer cannot run subgroup size 64\n")
                self.assertEqual(rc, 0, out)
                self.assertIn("0/1 passed, 1 skipped", out)
                self.assertIn("mesh producer cannot run subgroup size 64", out)

    def test_range_alone_does_not_confirm_the_stage(self):
        rc, out, _ = self.suite("ms_wave_info_s3", message="device: fake (subgroup 32..64)\n")
        self.assertEqual(rc, 1, out)
        self.assertIn("did not confirm", out)

    def test_other_errors_are_failures(self):
        rc, out, _ = self.suite("ms_wave_info_s3", code=1, message="vkCreateDevice failed\n")
        self.assertEqual(rc, 1, out)
        self.assertIn("0/1 passed", out)

    def test_emulation_checks_pixels_without_requesting_wave64(self):
        for good in (True, False):
            rc, out, envs = self.suite("ms_wave_info_s3", emulated=True, good=good,
                                      message="device: fake (subgroup 8..8)\n")
            self.assertEqual(rc, 0 if good else 1, out)
            self.assertEqual(envs, [{"VKH_MESH": "1"}])
            self.assertNotIn("SKIP", out)

    def test_plain_triangle_does_not_require_wave64(self):
        rc, out, envs = self.suite("ms_tri", message="device: fake (subgroup 8..8)\n")
        self.assertEqual(rc, 0, out)
        self.assertEqual(envs, [{"VKH_MESH": "1"}])

    def test_standalone_checker_requires_confirmation_and_checks_pixels(self):
        for message, good, emulated, success in (
            ("mesh subgroup size: 64\n", True, False, True),
            ("mesh subgroup size: 64\n", False, False, False),
            ("device: fake (subgroup 32..64)\n", True, False, False),
            ("device: fake (subgroup 8..8)\n", True, True, True),
            ("device: fake (subgroup 8..8)\n", False, True, False),
        ):
            with self.subTest(message=message, good=good, emulated=emulated), tempfile.TemporaryDirectory() as tmp:
                for name in ("ms_wave_info_s3", "fs_passthru"):
                    (Path(tmp) / (name + ".spv")).touch()
                def harness(cmd, **_kw):
                    Path(cmd[-1]).write_bytes(pixels(good))
                    return subprocess.CompletedProcess(cmd, 0, "", message)
                with patch.object(check.subprocess, "run", side_effect=harness), \
                        patch.object(check, "W64_EMULATED", emulated):
                    problem = check.mesh_w64x_test(tmp, tmp, "ms_wave_info_s3")
                self.assertEqual(not problem, success, problem)


if __name__ == "__main__":
    unittest.main()
