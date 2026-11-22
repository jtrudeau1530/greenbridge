"""Unit tests for tools/manifest_verifier.py using synthetic file fixtures.

Verifies SHA-256 manifest integrity in parallel with live progress reporting.
Zero game data used.

    python -I tests/test_manifest_verifier.py
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "manifest_verifier.py")

# Add root directory to sys.path so manifest_verifier can be imported directly
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tools.manifest_verifier as mv

SYNTHETIC_SIZES = {
    "alpha.bin": 256 * 1024,
    "bravo.bin": 64 * 1024,
    "charlie.bin": 128 * 1024,
    "delta.bin": 32 * 1024,
    "nested/echo.bin": 16 * 1024,
    "nested/deep/foxtrot.bin": 8 * 1024,
}


def make_payload(size: int, seed: str) -> bytes:
    block = hashlib.sha256(seed.encode("utf-8")).digest()
    return (block * (size // len(block) + 1))[:size]


class TestManifestVerifier(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.game_dir = os.path.join(self.root, "game")
        os.makedirs(self.game_dir, exist_ok=True)
        self.payloads: dict[str, bytes] = {}

        for rel, size in SYNTHETIC_SIZES.items():
            abs_p = os.path.join(self.game_dir, *rel.split("/"))
            os.makedirs(os.path.dirname(abs_p), exist_ok=True)
            data = make_payload(size, rel)
            with open(abs_p, "wb") as f:
                f.write(data)
            self.payloads[rel] = data

        self.manifest_path = os.path.join(self.root, "manifest.json")
        self.write_manifest(self.manifest_path, self.payloads)

    def write_manifest(
        self,
        dest: str,
        files_map: dict[str, bytes],
        wrapper: bool = True,
        upper_hash: bool = False,
    ) -> str:
        entries = {}
        for rel, data in files_map.items():
            h = hashlib.sha256(data).hexdigest()
            if upper_hash:
                h = h.upper()
            entries[rel] = {"size": len(data), "sha256": h}
        doc = {"created": "2026-11-22 00:00:00", "files": entries} if wrapper else entries
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        return dest

    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-I", TOOL, *args],
            cwd=self.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def test_sha256_file(self) -> None:
        sample_path = os.path.join(self.game_dir, "alpha.bin")
        expected = hashlib.sha256(self.payloads["alpha.bin"]).hexdigest()
        self.assertEqual(mv.sha256_file(sample_path), expected)

    def test_walk_normalization(self) -> None:
        items = dict(mv.walk(self.game_dir))
        self.assertEqual(sorted(items.keys()), sorted(SYNTHETIC_SIZES.keys()))
        for rel in items:
            self.assertNotIn("\\", rel)

    def test_load_manifest_valid_and_case_insensitive(self) -> None:
        manifest_upper = os.path.join(self.root, "manifest_upper.json")
        self.write_manifest(manifest_upper, self.payloads, upper_hash=True)
        entries = mv.load_manifest(manifest_upper)
        self.assertEqual(len(entries), len(SYNTHETIC_SIZES))
        for rel, info in entries.items():
            self.assertEqual(info["sha256"], info["sha256"].lower())
            self.assertEqual(info["size"], SYNTHETIC_SIZES[rel])

    def test_load_manifest_flat_dictionary(self) -> None:
        flat_path = os.path.join(self.root, "manifest_flat.json")
        self.write_manifest(flat_path, self.payloads, wrapper=False)
        entries = mv.load_manifest(flat_path)
        self.assertEqual(len(entries), len(SYNTHETIC_SIZES))

    def test_load_manifest_invalid_json(self) -> None:
        bad_json = os.path.join(self.root, "bad.json")
        with open(bad_json, "w", encoding="utf-8") as f:
            f.write("not json at all {")
        with self.assertRaises(mv.ManifestError):
            mv.load_manifest(bad_json)

    def test_load_manifest_missing_file(self) -> None:
        missing_path = os.path.join(self.root, "non_existent.json")
        with self.assertRaises(mv.ManifestError):
            mv.load_manifest(missing_path)

    def test_load_manifest_malformed_entries(self) -> None:
        invalid_entries = os.path.join(self.root, "invalid_entries.json")
        with open(invalid_entries, "w", encoding="utf-8") as f:
            json.dump({"files": {"bad.bin": {"size": "not-an-int", "sha256": "abc"}}}, f)
        with self.assertRaises(mv.ManifestError):
            mv.load_manifest(invalid_entries)

    def test_verify_target_dir_missing(self) -> None:
        with self.assertRaises(mv.ManifestError):
            mv.verify_manifest(os.path.join(self.root, "missing_dir"), self.manifest_path)

    def test_clean_verification_serial_and_parallel(self) -> None:
        serial_report = mv.verify_manifest(self.game_dir, self.manifest_path, jobs=1)
        parallel_report = mv.verify_manifest(self.game_dir, self.manifest_path, jobs=4)

        self.assertTrue(serial_report.passed)
        self.assertTrue(parallel_report.passed)
        self.assertEqual(serial_report.problem_count, 0)
        self.assertEqual(parallel_report.problem_count, 0)
        self.assertEqual(sorted(serial_report.matched), sorted(SYNTHETIC_SIZES.keys()))
        self.assertEqual(serial_report.matched, parallel_report.matched)
        self.assertEqual(serial_report.bytes_verified, sum(SYNTHETIC_SIZES.values()))
        self.assertEqual(parallel_report.bytes_verified, sum(SYNTHETIC_SIZES.values()))

    def test_problem_detection_and_deterministic_order(self) -> None:
        # 1. CHANGED (same size, modified byte)
        alpha_path = os.path.join(self.game_dir, "alpha.bin")
        with open(alpha_path, "r+b") as f:
            b = f.read(1)
            f.seek(0)
            f.write(bytes([b[0] ^ 0xFF]))

        # 2. SIZE (truncated)
        bravo_path = os.path.join(self.game_dir, "bravo.bin")
        with open(bravo_path, "r+b") as f:
            f.truncate(1024)

        # 3. MISSING
        charlie_path = os.path.join(self.game_dir, "charlie.bin")
        os.remove(charlie_path)

        # 4. NEW (untracked)
        new_path = os.path.join(self.game_dir, "extra.bin")
        with open(new_path, "wb") as f:
            f.write(b"untracked extra file")

        serial = mv.verify_manifest(self.game_dir, self.manifest_path, jobs=1)
        parallel = mv.verify_manifest(self.game_dir, self.manifest_path, jobs=4)

        self.assertFalse(serial.passed)
        self.assertFalse(parallel.passed)
        self.assertEqual(serial.problem_count, 3)
        self.assertEqual(parallel.problem_count, 3)
        self.assertEqual(serial.problems, parallel.problems)
        self.assertEqual(serial.problems["alpha.bin"], "CHANGED")
        self.assertEqual(serial.problems["bravo.bin"], "SIZE")
        self.assertEqual(serial.problems["charlie.bin"], "MISSING")
        self.assertEqual(serial.untracked, ["extra.bin"])
        self.assertEqual(parallel.untracked, ["extra.bin"])

        # Check stdout formatted printing
        out = io.StringIO()
        mv.print_report(serial, out=out)
        lines = out.getvalue().splitlines()
        expected = [
            "CHANGED   alpha.bin",
            "SIZE      bravo.bin",
            "MISSING   charlie.bin",
            "NEW       extra.bin",
            "3 problem(s)",
        ]
        self.assertEqual(lines, expected)

    def test_untracked_file_not_failing_without_strict(self) -> None:
        new_path = os.path.join(self.game_dir, "bonus.bin")
        with open(new_path, "wb") as f:
            f.write(b"bonus")

        report_default = mv.verify_manifest(self.game_dir, self.manifest_path, jobs=2, strict=False)
        self.assertTrue(report_default.passed)
        self.assertEqual(report_default.problem_count, 0)
        self.assertEqual(report_default.untracked, ["bonus.bin"])

        report_strict = mv.verify_manifest(self.game_dir, self.manifest_path, jobs=2, strict=True)
        self.assertFalse(report_strict.passed)
        self.assertEqual(report_strict.untracked, ["bonus.bin"])

    def test_progress_reporting_callback_and_stream(self) -> None:
        progress_events: list[mv.ProgressInfo] = []
        stream = io.StringIO()

        report = mv.verify_manifest(
            self.game_dir,
            self.manifest_path,
            jobs=2,
            progress_callback=progress_events.append,
            progress_stream=stream,
        )

        self.assertTrue(report.passed)
        self.assertEqual(len(progress_events), len(SYNTHETIC_SIZES))

        # Check progress monotonicity
        prev_files = 0
        prev_bytes = 0
        for ev in progress_events:
            self.assertGreaterEqual(ev.completed_files, prev_files)
            self.assertGreaterEqual(ev.completed_bytes, prev_bytes)
            self.assertGreaterEqual(ev.pct, 0.0)
            self.assertLessEqual(ev.pct, 1.0001)
            prev_files = ev.completed_files
            prev_bytes = ev.completed_bytes

        self.assertEqual(progress_events[-1].completed_files, len(SYNTHETIC_SIZES))
        self.assertEqual(progress_events[-1].completed_bytes, sum(SYNTHETIC_SIZES.values()))
        self.assertAlmostEqual(progress_events[-1].pct, 1.0, places=4)

        stream_output = stream.getvalue()
        self.assertTrue(any("[" in line for line in stream_output.splitlines()))

    def test_unreadable_file_graceful_handling(self) -> None:
        # Simulate unreadable file by patching sha256_file to raise OSError
        def mock_sha(path: str, chunk_size: int = mv.CHUNK) -> str:
            if "alpha.bin" in path:
                raise OSError("Simulated permission denied")
            with open(path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()

        with patch("tools.manifest_verifier.sha256_file", side_effect=mock_sha):
            report = mv.verify_manifest(self.game_dir, self.manifest_path, jobs=1)
            self.assertFalse(report.passed)
            self.assertEqual(report.problems.get("alpha.bin"), "UNREADABLE")
            self.assertEqual(len(report.unreadable), 1)
            self.assertEqual(report.unreadable[0][0], "alpha.bin")

    def test_empty_manifest_and_zero_byte_file(self) -> None:
        zero_p = os.path.join(self.game_dir, "empty.bin")
        with open(zero_p, "wb"):
            pass
        empty_manifest = os.path.join(self.root, "empty.json")
        with open(empty_manifest, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "files": {
                        "empty.bin": {
                            "size": 0,
                            "sha256": hashlib.sha256(b"").hexdigest(),
                        }
                    }
                },
                f,
            )
        report = mv.verify_manifest(self.game_dir, empty_manifest, jobs=1)
        self.assertTrue(report.passed)
        self.assertIn("empty.bin", report.matched)

    def test_cli_clean_verification(self) -> None:
        res = self.run_cli(self.game_dir, "--manifest", self.manifest_path, "--jobs", "2")
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertEqual(res.stdout.strip(), "OK")

    def test_cli_verify_subcommand_syntax(self) -> None:
        res = self.run_cli("verify", self.game_dir, "--manifest", self.manifest_path)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertEqual(res.stdout.strip(), "OK")

    def test_cli_verbose_output(self) -> None:
        res = self.run_cli(self.game_dir, "--manifest", self.manifest_path, "--verbose")
        self.assertEqual(res.returncode, 0)
        self.assertIn("OK (", res.stdout)
        self.assertIn("GiB in", res.stdout)

    def test_cli_json_output(self) -> None:
        res = self.run_cli(self.game_dir, "--manifest", self.manifest_path, "--json")
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        data = json.loads(res.stdout)
        self.assertTrue(data["passed"])
        self.assertEqual(data["problem_count"], 0)
        self.assertEqual(data["matched_count"], len(SYNTHETIC_SIZES))

    def test_cli_tampered_returns_exit_code_1(self) -> None:
        delta_p = os.path.join(self.game_dir, "delta.bin")
        with open(delta_p, "r+b") as f:
            f.write(b"tampered")
        res = self.run_cli(self.game_dir, "--manifest", self.manifest_path)
        self.assertEqual(res.returncode, 1)
        self.assertIn("CHANGED   delta.bin", res.stdout)
        self.assertIn("1 problem(s)", res.stdout)

    def test_cli_strict_with_untracked_fails(self) -> None:
        extra_p = os.path.join(self.game_dir, "new_file.dat")
        with open(extra_p, "wb") as f:
            f.write(b"extra")

        res_lax = self.run_cli(self.game_dir, "--manifest", self.manifest_path)
        self.assertEqual(res_lax.returncode, 0)
        self.assertIn("NEW       new_file.dat", res_lax.stdout)
        self.assertIn("OK", res_lax.stdout)

        res_strict = self.run_cli(self.game_dir, "--manifest", self.manifest_path, "--strict")
        self.assertEqual(res_strict.returncode, 1)
        self.assertIn("NEW       new_file.dat", res_strict.stdout)
        self.assertIn("1 problem(s)", res_strict.stdout)

    def test_cli_strict_ignores_only_selected_in_tree_manifest(self) -> None:
        in_tree = self.write_manifest(os.path.join(self.game_dir, "nested", "manifest.json"), self.payloads)
        args = (self.game_dir, "--manifest", in_tree, "--strict", "--json")
        res = self.run_cli(*args)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertEqual(json.loads(res.stdout)["untracked"], [])

        self.write_manifest(os.path.join(self.game_dir, "manifest.json"), self.payloads)
        res = self.run_cli(*args)
        self.assertEqual(res.returncode, 1, res.stdout + res.stderr)
        self.assertEqual(json.loads(res.stdout)["untracked"], ["manifest.json"])

    def test_cli_missing_manifest_returns_exit_code_2(self) -> None:
        res = self.run_cli(self.game_dir, "--manifest", os.path.join(self.root, "none.json"))
        self.assertEqual(res.returncode, 2)
        self.assertIn("ERROR:", res.stderr)

    def test_cli_missing_target_dir_returns_exit_code_2(self) -> None:
        res = self.run_cli(os.path.join(self.root, "none_dir"), "--manifest", self.manifest_path)
        self.assertEqual(res.returncode, 2)
        self.assertIn("ERROR:", res.stderr)

    def test_cli_quiet_suppresses_progress(self) -> None:
        res_default = self.run_cli(self.game_dir, "--manifest", self.manifest_path)
        self.assertEqual(res_default.returncode, 0)
        # default outputs progress to stderr
        self.assertTrue(len(res_default.stderr) > 0)

        res_quiet = self.run_cli(self.game_dir, "--manifest", self.manifest_path, "--quiet")
        self.assertEqual(res_quiet.returncode, 0)
        self.assertEqual(res_quiet.stderr.strip(), "")


if __name__ == "__main__":
    unittest.main()
