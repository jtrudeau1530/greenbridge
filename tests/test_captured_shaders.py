"""make_game.py's captured_shaders(): the containers a run could not resolve, turned into modules.

The title builds some shader programs at run time, so the carve never produces them and a run stops at the pipeline
that asks for one (#8851 A40/A44). `LEONIDA_SHADER_CAPTURE=<work>/shaders/captured` makes the runtime write those
containers with a sidecar naming the stage; this step runs them through the same three tools the carve uses.

The translator itself is not exercised here - a real program cannot be synthesised - so `make_game.py`'s `py()` is
replaced by a stub that plays those three tools. What is tested is this repository's own plumbing: the folder and
sidecar discovery, the stage it hands on, the staging layout rdna2spirv.py reads, skipping a program already in the
library, and the merge that leaves a valid index behind. Standard library only; no game files.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import make_game  # noqa: E402

CONTAINER = b"DXBC\x00\x01\x00\x00FSSM\x01\x01\x00\x00" + bytes(range(16))


class CapturedShadersTest(unittest.TestCase):
    def setUp(self):
        self.work = Path(tempfile.mkdtemp(prefix="leonida_captured_"))
        self.addCleanup(shutil.rmtree, self.work, True)
        self.caught = self.work / "shaders" / "captured"
        self.spirv = self.work / "shaders" / "spirv"
        self.spirv.mkdir(parents=True)
        self.library = {"existing_shader": {"stage": "ps", "fssm_sha256": "0" * 64}}
        (self.spirv / "index.json").write_text(json.dumps(self.library), encoding="utf-8")
        self.calls = []

    # -- the stub that plays fssm_unpack.py, shader_disasm.py and rdna2spirv.py -------------------------------
    def fake_py(self, tool, *args):
        self.calls.append((tool, args))
        staging = Path(args[0]).parent if tool == "fssm_unpack.py" else None
        if tool == "fssm_unpack.py":
            root = Path(args[0])
            entries = json.loads((root / "index.json").read_text(encoding="utf-8"))
            (root / "fssm").mkdir(exist_ok=True)
            for entry in entries:
                stem = entry["file"][: -len(".dxbc")]
                # The program a container holds: here its own bytes, so the hash of a container is the hash the
                # library keys it by, as fssm_unpack.py + rdna2spirv.py give for a real one.
                (root / "fssm" / (stem + ".fssm")).write_bytes((root / (stem + ".dxbc")).read_bytes())
        elif tool == "shader_disasm.py":
            listing = Path(args[2]) if len(args) > 2 and args[1] == "--out" else None
            listing.mkdir(parents=True, exist_ok=True)
            for blob in Path(args[0]).glob("*.fssm"):
                (listing / (blob.stem + ".s")).write_text("; synthetic\n", encoding="utf-8")
        elif tool == "rdna2spirv.py":
            root = Path(args[0]).parent
            out = Path(args[args.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            entries = json.loads((root / "index.json").read_text(encoding="utf-8"))
            fresh = {}
            for entry in entries:
                stem = entry["file"][: -len(".dxbc")]
                import hashlib
                digest = hashlib.sha256((root / "fssm" / (stem + ".fssm")).read_bytes()).hexdigest()
                fresh[stem] = {"stage": entry["stage"], "fssm_sha256": digest}
                (out / (stem + ".spv")).write_bytes(b"spirv" + stem.encode())
            (out / "index.json").write_text(json.dumps(fresh), encoding="utf-8")
        else:  # pragma: no cover - a tool this test does not expect
            raise AssertionError(f"unexpected tool {tool}")

    def run_step(self):
        with mock.patch.object(make_game, "py", self.fake_py):
            make_game.captured_shaders(str(self.work), 2)

    def write_capture(self, name, stage="cs", body=None):
        """A capture as the runtime writes it: the file is named after the program's SHA-256."""
        self.caught.mkdir(parents=True, exist_ok=True)
        program = body if body is not None else CONTAINER + name.encode()
        digest = hashlib.sha256(program).hexdigest()
        (self.caught / (digest + ".dxbc")).write_bytes(program)
        if stage is not None:
            (self.caught / (digest + ".json")).write_text(json.dumps({"stage": stage}), encoding="utf-8")
        return digest

    def library_now(self):
        return json.loads((self.spirv / "index.json").read_text(encoding="utf-8"))

    # -- tests -----------------------------------------------------------------------------------------------
    def test_no_captures_is_a_no_op(self):
        self.run_step()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.library_now(), self.library)

    def test_a_capture_becomes_a_module_and_keeps_the_rest_of_the_library(self):
        digest = self.write_capture("a" * 64)
        self.run_step()
        library = self.library_now()
        self.assertIn("existing_shader", library, "the library's own entries survive")
        added = [name for name in library if name.startswith("captured_")]
        self.assertEqual(added, ["captured_" + digest[:16]])
        self.assertEqual(library[added[0]]["stage"], "cs")
        self.assertTrue((self.spirv / (added[0] + ".spv")).is_file(), "the module is copied beside the index")
        self.assertFalse((self.work / "shaders" / "captured_work").exists(), "the staging folder is removed")

    def test_a_second_run_translates_nothing_new(self):
        self.write_capture("b" * 64)
        self.run_step()
        self.calls.clear()
        self.run_step()
        self.assertEqual(self.calls, [], "a translated program is not translated again")
        self.assertEqual(len([n for n in self.library_now() if n.startswith("captured_")]), 1)

    def test_a_capture_without_a_stage_is_skipped(self):
        self.write_capture("c" * 64, stage=None)
        self.run_step()
        self.assertEqual(self.calls, [])
        self.assertFalse([n for n in self.library_now() if n.startswith("captured_")])

    def test_a_capture_with_an_unknown_stage_is_skipped(self):
        self.write_capture("d" * 64, stage="xx")
        self.run_step()
        self.assertEqual(self.calls, [])
        self.assertFalse([n for n in self.library_now() if n.startswith("captured_")])

    def test_the_stage_reaches_the_translator(self):
        digest = self.write_capture("e" * 64, stage="ps")
        self.run_step()
        staging_index = None
        for tool, args in self.calls:
            if tool == "fssm_unpack.py":
                staging_index = Path(args[0]) / "index.json"
        self.assertIsNotNone(staging_index, "fssm_unpack.py ran on the staging folder")
        # The stub rewrote nothing: captured_shaders() wrote the index the translator reads, and the merged entry
        # carries that same stage.
        self.assertEqual(self.library_now()["captured_" + digest[:16]]["stage"], "ps")

    def test_two_captures_sharing_a_prefix_both_land(self):
        """A capture's name comes from its file name, so two that agree on their first 8 characters must not collide.

        The runtime names captures after their program, so this is a hand-renamed pair rather than something it
        writes; the step still has to keep both.
        """
        self.caught.mkdir(parents=True, exist_ok=True)
        first, second = "abcdef01" + "1" * 11 + "x", "abcdef01" + "2" * 11 + "y"
        for stem, body, stage in ((first, CONTAINER + b"one", "vs"), (second, CONTAINER + b"two", "cs")):
            (self.caught / (stem + ".dxbc")).write_bytes(body)
            (self.caught / (stem + ".json")).write_text(json.dumps({"stage": stage}), encoding="utf-8")
        self.run_step()
        added = sorted(n for n in self.library_now() if n.startswith("captured_"))
        self.assertEqual(added, sorted(["captured_" + first[:16], "captured_" + second[:16]]),
                         "both captures land, and neither overwrites the other's staging file")

    def test_two_captures_both_land(self):
        one = self.write_capture("f" * 64, stage="vs")
        two = self.write_capture("1" * 64, stage="cs")
        self.run_step()
        added = sorted(n for n in self.library_now() if n.startswith("captured_"))
        self.assertEqual(added, sorted(["captured_" + one[:16], "captured_" + two[:16]]))


if __name__ == "__main__":
    unittest.main()
