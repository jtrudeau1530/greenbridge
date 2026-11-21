"""Tests for tools/manifest.py covering directory hashing, manifest creation, verification, and corruption detection.

    python -I tests/test_manifest.py
"""
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "manifest.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import manifest  # noqa: E402


class ManifestSha256Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_sha256_computes_expected_hash(self):
        payload = b"synthetic manifest test payload\nwith multiple lines and deterministic data."
        path = os.path.join(self.tmp.name, "sample.bin")
        with open(path, "wb") as f:
            f.write(payload)

        expected = hashlib.sha256(payload).hexdigest()
        self.assertEqual(manifest.sha256(path), expected)

    def test_sha256_empty_file(self):
        path = os.path.join(self.tmp.name, "empty.bin")
        with open(path, "wb") as f:
            pass

        expected = hashlib.sha256(b"").hexdigest()
        self.assertEqual(manifest.sha256(path), expected)


class ManifestWalkTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pkg = os.path.join(self.tmp.name, "pkg")
        os.makedirs(os.path.join(self.pkg, "sub", "deep"))

        with open(os.path.join(self.pkg, "root.dat"), "wb") as f:
            f.write(b"root file")
        with open(os.path.join(self.pkg, "sub", "mid.dat"), "wb") as f:
            f.write(b"mid level file")
        with open(os.path.join(self.pkg, "sub", "deep", "leaf.dat"), "wb") as f:
            f.write(b"leaf file")

    def test_walk_normalizes_to_forward_slashes_and_yields_all_files(self):
        entries = list(manifest.walk(self.pkg))
        rel_paths = [rel for rel, _ in entries]

        self.assertIn("root.dat", rel_paths)
        self.assertIn("sub/mid.dat", rel_paths)
        self.assertIn("sub/deep/leaf.dat", rel_paths)
        for rel in rel_paths:
            self.assertNotIn("\\", rel)

        for rel, abs_path in entries:
            self.assertTrue(os.path.isabs(abs_path))
            self.assertTrue(os.path.exists(abs_path))


class ManifestCreateVerifyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pkg = os.path.join(self.tmp.name, "game_pkg")
        os.makedirs(os.path.join(self.pkg, "content", "data"))

        self.files = {
            "alpha.bin": b"alpha payload bytes " * 100,
            "bravo.bin": b"bravo payload bytes " * 50,
            "content/asset1.dat": b"asset 1 synthetic binary payload " * 20,
            "content/data/table.bin": b"table data entries and records " * 10,
        }
        for rel, data in self.files.items():
            full = os.path.join(self.pkg, rel.replace("/", os.sep))
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "wb") as f:
                f.write(data)

        self.out_manifest = os.path.join(self.tmp.name, "reports", "test_manifest.json")

    def test_create_manifest_format_and_hashes(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = manifest.create(self.pkg, self.out_manifest, jobs=1)

        self.assertEqual(rc, 0)
        self.assertTrue(os.path.isfile(self.out_manifest))

        with open(self.out_manifest, encoding="utf-8") as f:
            doc = json.load(f)

        self.assertIn("created", doc)
        self.assertIn("files", doc)
        entries = doc["files"]
        self.assertEqual(len(entries), len(self.files))

        for rel, data in self.files.items():
            self.assertIn(rel, entries)
            self.assertEqual(entries[rel]["size"], len(data))
            self.assertEqual(entries[rel]["sha256"], hashlib.sha256(data).hexdigest())

    def test_create_is_identical_for_serial_and_parallel(self):
        out_serial = os.path.join(self.tmp.name, "serial.json")
        out_parallel = os.path.join(self.tmp.name, "parallel.json")

        with contextlib.redirect_stdout(io.StringIO()):
            manifest.create(self.pkg, out_serial, jobs=1)
            manifest.create(self.pkg, out_parallel, jobs=4)

        with open(out_serial, encoding="utf-8") as f:
            serial_doc = json.load(f)
        with open(out_parallel, encoding="utf-8") as f:
            parallel_doc = json.load(f)

        self.assertEqual(serial_doc["files"], parallel_doc["files"])
        self.assertEqual(json.dumps(serial_doc["files"]), json.dumps(parallel_doc["files"]))

    def test_verify_intact_directory_returns_zero(self):
        with contextlib.redirect_stdout(io.StringIO()):
            manifest.create(self.pkg, self.out_manifest, jobs=1)

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = manifest.verify(self.pkg, self.out_manifest, jobs=1)

        self.assertEqual(rc, 0)
        self.assertIn("OK", out.getvalue())

    def test_verify_detects_missing_file(self):
        with contextlib.redirect_stdout(io.StringIO()):
            manifest.create(self.pkg, self.out_manifest, jobs=1)

        missing_target = os.path.join(self.pkg, "content", "asset1.dat")
        os.remove(missing_target)

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = manifest.verify(self.pkg, self.out_manifest, jobs=1)

        self.assertEqual(rc, 1)
        output = out.getvalue()
        self.assertIn("MISSING   content/asset1.dat", output)
        self.assertIn("1 problem(s)", output)

    def test_verify_detects_size_mismatch(self):
        with contextlib.redirect_stdout(io.StringIO()):
            manifest.create(self.pkg, self.out_manifest, jobs=1)

        target = os.path.join(self.pkg, "content", "asset1.dat")
        with open(target, "ab") as f:
            f.write(b"appended extra bytes")

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = manifest.verify(self.pkg, self.out_manifest, jobs=1)

        self.assertEqual(rc, 1)
        output = out.getvalue()
        self.assertIn("SIZE      content/asset1.dat", output)
        self.assertIn("1 problem(s)", output)

    def test_verify_detects_changed_content_same_size(self):
        with contextlib.redirect_stdout(io.StringIO()):
            manifest.create(self.pkg, self.out_manifest, jobs=1)

        target = os.path.join(self.pkg, "content", "asset1.dat")
        with open(target, "r+b") as f:
            original = f.read()
            mutated = bytes([original[0] ^ 0xFF]) + original[1:]
            f.seek(0)
            f.write(mutated)

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = manifest.verify(self.pkg, self.out_manifest, jobs=1)

        self.assertEqual(rc, 1)
        output = out.getvalue()
        self.assertIn("CHANGED   content/asset1.dat", output)
        self.assertIn("1 problem(s)", output)

    def test_verify_reports_new_untracked_file(self):
        with contextlib.redirect_stdout(io.StringIO()):
            manifest.create(self.pkg, self.out_manifest, jobs=1)

        extra = os.path.join(self.pkg, "content", "unexpected_new.dat")
        with open(extra, "wb") as f:
            f.write(b"new rogue asset")

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = manifest.verify(self.pkg, self.out_manifest, jobs=1)

        self.assertEqual(rc, 0)
        output = out.getvalue()
        self.assertIn("NEW       content/unexpected_new.dat", output)
        self.assertIn("OK", output)

    def test_verify_accumulates_multiple_corruptions(self):
        with contextlib.redirect_stdout(io.StringIO()):
            manifest.create(self.pkg, self.out_manifest, jobs=1)

        os.remove(os.path.join(self.pkg, "alpha.bin"))

        with open(os.path.join(self.pkg, "bravo.bin"), "ab") as f:
            f.write(b"extra")

        tbl = os.path.join(self.pkg, "content", "data", "table.bin")
        with open(tbl, "r+b") as f:
            b = f.read()
            f.seek(0)
            f.write(bytes([b[0] ^ 0xFF]) + b[1:])

        with open(os.path.join(self.pkg, "new_file.dat"), "wb") as f:
            f.write(b"extra")

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = manifest.verify(self.pkg, self.out_manifest, jobs=1)

        self.assertEqual(rc, 1)
        output = out.getvalue()
        self.assertIn("MISSING   alpha.bin", output)
        self.assertIn("SIZE      bravo.bin", output)
        self.assertIn("CHANGED   content/data/table.bin", output)
        self.assertIn("NEW       new_file.dat", output)
        self.assertIn("3 problem(s)", output)

    def test_verify_messages_do_not_depend_on_jobs(self):
        with contextlib.redirect_stdout(io.StringIO()):
            manifest.create(self.pkg, self.out_manifest, jobs=1)

        target = os.path.join(self.pkg, "content", "asset1.dat")
        with open(target, "r+b") as f:
            orig = f.read()
            f.seek(0)
            f.write(bytes([orig[0] ^ 0xFF]) + orig[1:])

        out_serial = io.StringIO()
        with contextlib.redirect_stdout(out_serial):
            rc_serial = manifest.verify(self.pkg, self.out_manifest, jobs=1)

        out_parallel = io.StringIO()
        with contextlib.redirect_stdout(out_parallel):
            rc_parallel = manifest.verify(self.pkg, self.out_manifest, jobs=4)

        self.assertEqual(rc_serial, rc_parallel)
        self.assertEqual(out_serial.getvalue(), out_parallel.getvalue())


class ManifestCliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pkg = os.path.join(self.tmp.name, "pkg")
        os.makedirs(self.pkg)
        with open(os.path.join(self.pkg, "sample.dat"), "wb") as f:
            f.write(b"cli test payload bytes")
        self.out_manifest = os.path.join(self.tmp.name, "manifest.json")

    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True)

    def test_cli_create_and_verify(self):
        p_create = self.run_cli("create", self.pkg, "--out", self.out_manifest, "--jobs", "2")
        self.assertEqual(p_create.returncode, 0, p_create.stderr)
        self.assertIn("1 files", p_create.stdout)

        p_verify = self.run_cli("verify", self.pkg, "--manifest", self.out_manifest, "--jobs", "2")
        self.assertEqual(p_verify.returncode, 0, p_verify.stderr)
        self.assertIn("OK", p_verify.stdout)

    def test_cli_verify_failure_exit_code(self):
        self.assertEqual(self.run_cli("create", self.pkg, "--out", self.out_manifest, "--jobs", "1").returncode, 0)

        with open(os.path.join(self.pkg, "sample.dat"), "wb") as f:
            f.write(b"corrupted bytes!")

        p_verify = self.run_cli("verify", self.pkg, "--manifest", self.out_manifest, "--jobs", "1")
        self.assertEqual(p_verify.returncode, 1)
        self.assertIn("problem(s)", p_verify.stdout)


class ManifestBelowNormalTest(unittest.TestCase):
    def test_below_normal_callable(self):
        manifest.below_normal()


if __name__ == "__main__":
    unittest.main(verbosity=2)
