"""GLB export file ownership and overlapping-write regressions; no game data.

    python -I tests/test_glb_atomic.py -v
"""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import glb_writer


class GlbAtomicTests(unittest.TestCase):
    @staticmethod
    def primitives(size=1):
        return glb_writer.parse_obj(
            f"v 0 0 0\nv {size} 0 0\nv 0 {size} 0\nf 1 2 3\n",
            is_path=False,
        )

    def test_success_preserves_existing_sibling_and_output_bytes(self):
        prims = self.primitives()
        expected = glb_writer.build_glb(prims)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "mesh.glb"
            sibling = Path(str(target) + ".tmp")
            sibling.write_bytes(b"unrelated file")
            target.write_bytes(b"previous export")

            stats = glb_writer.write_glb(target, prims)

            self.assertEqual(target.read_bytes(), expected)
            self.assertEqual(sibling.read_bytes(), b"unrelated file")
            self.assertEqual(stats["bytes"], len(expected))
            self.assertEqual(set(os.listdir(directory)), {target.name, sibling.name})

    def test_failed_replace_preserves_destination_and_sibling(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "mesh.glb"
            sibling = Path(str(target) + ".tmp")
            target.write_bytes(b"previous export")
            sibling.write_bytes(b"unrelated file")
            with mock.patch.object(glb_writer.os, "replace", side_effect=PermissionError("busy")):
                with self.assertRaisesRegex(PermissionError, "busy"):
                    glb_writer.write_glb(target, self.primitives())

            self.assertEqual(target.read_bytes(), b"previous export")
            self.assertEqual(sibling.read_bytes(), b"unrelated file")
            self.assertEqual(set(os.listdir(directory)), {target.name, sibling.name})

    def test_overlapping_writes_each_publish_their_own_complete_bytes(self):
        first, second = self.primitives(1), self.primitives(2)
        first_bytes, second_bytes = map(glb_writer.build_glb, (first, second))
        replace = os.replace
        staging_paths = []
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "mesh.glb"

            def interleave(source, destination):
                staging_paths.append(os.fspath(source))
                self.assertEqual(Path(source).parent.resolve(), target.parent.resolve())
                if len(staging_paths) == 1:
                    # The first writer has closed its file but has not published it.
                    glb_writer.write_glb(target, second)
                    self.assertEqual(target.read_bytes(), second_bytes)
                replace(source, destination)

            with mock.patch.object(glb_writer.os, "replace", side_effect=interleave):
                glb_writer.write_glb(target, first)

            self.assertEqual(target.read_bytes(), first_bytes)
            self.assertEqual(len(set(staging_paths)), 2)
            self.assertEqual(os.listdir(directory), [target.name])

    def test_partial_write_failure_cleans_only_its_own_staging_file(self):
        create_temp = tempfile.NamedTemporaryFile

        def failing_temp(*args, **kwargs):
            handle = create_temp(*args, **kwargs)
            write = handle.write

            def fail_write(payload):
                write(payload[:12])
                raise OSError("simulated full disk")

            handle.write = fail_write
            return handle

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "mesh.glb"
            sibling = Path(str(target) + ".tmp")
            target.write_bytes(b"previous export")
            sibling.write_bytes(b"unrelated file")
            with mock.patch.object(glb_writer.tempfile, "NamedTemporaryFile", side_effect=failing_temp):
                with self.assertRaisesRegex(OSError, "simulated full disk"):
                    glb_writer.write_glb(target, self.primitives())

            self.assertEqual(target.read_bytes(), b"previous export")
            self.assertEqual(sibling.read_bytes(), b"unrelated file")
            self.assertEqual(set(os.listdir(directory)), {target.name, sibling.name})

    def test_creates_missing_parent_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "mesh.glb"
            prims = self.primitives()
            glb_writer.write_glb(target, prims)
            self.assertEqual(target.read_bytes(), glb_writer.build_glb(prims))
            self.assertEqual(os.listdir(target.parent), [target.name])


if __name__ == "__main__":
    unittest.main()
