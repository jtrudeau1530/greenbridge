"""Manifest create/verify must not depend on how many threads hash the files.

    python -I tests/test_manifest.py

The tree and manifests are generated in a temporary directory; nothing is committed.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "manifest.py")

# alpha is much larger than the rest so worker completion order differs from walk order.
SIZES = {
    "alpha.bin": 2 * 1024 * 1024,
    "bravo.bin": 512 * 1024,
    "charlie.bin": 384 * 1024,
    "delta.bin": 256 * 1024,
    "nested/echo.bin": 128 * 1024,
}


def payload(size: int, seed: str) -> bytes:
    block = hashlib.sha256(seed.encode()).digest()
    return (block * (size // len(block) + 1))[:size]


class ManifestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.game = os.path.join(self.root, "game")
        self.contents = {}
        for rel, size in SIZES.items():
            path = os.path.join(self.game, *rel.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = payload(size, rel)
            with open(path, "wb") as f:
                f.write(data)
            self.contents[rel] = data

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], cwd=self.root,
                              capture_output=True, text=True, encoding="utf-8", errors="replace")

    def create(self, jobs: int, name: str) -> str:
        out = os.path.join(self.root, name)
        p = self.run_tool("create", self.game, "--out", out, "--jobs", str(jobs))
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        return out

    def verify(self, manifest: str, jobs: int):
        return self.run_tool("verify", self.game, "--manifest", manifest, "--jobs", str(jobs))

    def read(self, path: str) -> dict:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_create_is_identical_for_serial_and_parallel(self):
        one = self.read(self.create(1, "one.json"))
        four = self.read(self.create(4, "four.json"))
        self.assertEqual(one["files"], four["files"])
        # dict equality ignores order, so also compare the serialized form: the JSON must be
        # byte-for-byte the same apart from the "created" timestamp.
        self.assertEqual(json.dumps(one["files"]), json.dumps(four["files"]))
        self.assertEqual(list(one["files"]), sorted(SIZES))
        for rel, data in self.contents.items():
            self.assertEqual(one["files"][rel],
                             {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()})

    def test_verify_clean(self):
        p = self.verify(self.create(4, "clean.json"), 1)
        self.assertEqual(p.returncode, 0, p.stdout)
        self.assertEqual(p.stdout.strip(), "OK")

    def test_verify_messages_do_not_depend_on_jobs(self):
        manifest = self.create(1, "mutated.json")
        with open(os.path.join(self.game, "alpha.bin"), "r+b") as f:
            first = f.read(1)  # CHANGED, same size; the largest file is listed first
            f.seek(0)
            f.write(bytes([first[0] ^ 0xFF]))
        with open(os.path.join(self.game, "bravo.bin"), "r+b") as f:
            f.truncate(SIZES["bravo.bin"] // 2)  # SIZE
        os.remove(os.path.join(self.game, "charlie.bin"))  # MISSING
        with open(os.path.join(self.game, "new.bin"), "wb") as f:
            f.write(b"new")  # NEW

        serial = self.verify(manifest, 1)
        parallel = self.verify(manifest, 4)
        expected = [
            "CHANGED   alpha.bin",
            "SIZE      bravo.bin",
            "MISSING   charlie.bin",
            "NEW       new.bin",
            "3 problem(s)",
        ]
        self.assertEqual(serial.returncode, 1)
        self.assertEqual(serial.stdout.splitlines(), expected)
        self.assertEqual(parallel.returncode, serial.returncode)
        self.assertEqual(parallel.stdout, serial.stdout)

    def test_new_file_is_reported_but_not_a_problem(self):
        manifest = self.create(1, "only-new.json")
        with open(os.path.join(self.game, "new.bin"), "wb") as f:
            f.write(b"new")
        p = self.verify(manifest, 4)
        self.assertEqual(p.returncode, 0)
        self.assertEqual(p.stdout.splitlines(), ["NEW       new.bin", "OK"])

    def test_parallel_manifest_verifies_serially(self):
        p = self.verify(self.create(4, "parallel.json"), 1)
        self.assertEqual(p.returncode, 0, p.stdout)
        self.assertEqual(p.stdout.strip(), "OK")


if __name__ == "__main__":
    unittest.main()
