import ctypes
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "carve_all.ps1"
INDEX = {"summary": {"files": 1, "bink2_files": 0, "span": 10, "compressed_bytes": 5, "unread_bytes": 5,
                     "inflated_bytes": 5}, "files": []}


@unittest.skipUnless(os.name == "nt", "PowerShell research script")
class CarveAllGamePathTest(unittest.TestCase):
    def test_failed_carver_stops_without_replacing_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            game, out, tools = root / "game", root / "out", root / "tools"
            for folder in (game, out, tools):
                folder.mkdir()
            (game / "first.rpf").write_bytes(b"x")
            (game / "second.rpf").write_bytes(b"xx")
            (tools / "python.cmd").write_bytes(b"@exit /b 7\r\n")
            coverage = out / "coverage.json"
            coverage.write_bytes(b"[]\n")
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
                 "-Game", str(game), "-Out", str(out)],
                env={**os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"]},
                capture_output=True, text=True, timeout=30,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Carving first.rpf failed (exit 7)", result.stderr)
            self.assertNotIn("second.rpf", result.stdout + result.stderr)
            self.assertEqual(coverage.read_bytes(), b"[]\n")

    def test_relative_and_trailing_separator_game_paths_find_existing_carves(self):
        with tempfile.TemporaryDirectory() as tmp:
            game, out = Path(tmp, "game"), Path(tmp, "out")
            (game / "pack" / "p1").mkdir(parents=True)
            (game / "common.rpf").write_bytes(b"x")
            (game / "pack" / "p1" / "p1.rpf").write_bytes(b"xy")
            # Both archives are already carved, so no carve should start.
            for name in ("common", "pack_p1_p1"):
                (out / name).mkdir(parents=True)
                (out / name / "index.json").write_text(json.dumps(INDEX), encoding="utf-8")
            args = ["game", ".\\game", str(game) + "\\"]
            short_game = ctypes.create_unicode_buffer(32768)
            short_path_length = ctypes.windll.kernel32.GetShortPathNameW(str(game), short_game, len(short_game))
            if short_path_length and short_game.value.casefold() != str(game).casefold():
                args.append(short_game.value + "\\")
            for arg in args:
                with self.subTest(game=arg):
                    result = subprocess.run(
                        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
                         "-Game", arg, "-Out", str(out)], cwd=tmp, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertNotIn("carving", result.stdout)
                    rows = json.loads((out / "coverage.json").read_text(encoding="utf-8-sig"))
                    self.assertEqual([r["archive"] for r in rows], ["common.rpf", "pack\\p1\\p1.rpf"])
                    self.assertEqual(sorted(p.name for p in out.iterdir()),
                                     ["common", "coverage.json", "pack_p1_p1"])

    def test_game_path_with_83_short_name_component_finds_existing_carves(self):
        # Resolve-Path keeps an 8.3 component (e.g. RUNNER~1) but Get-ChildItem emits canonical
        # FullNames. A trailing separator plus such a component used to cut the rel path at the
        # wrong offset, re-carving archives that were already present.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp, "LongDirectoryNameForCarveAllShortPath")
            game, out = base / "game", base / "out"
            (game / "pack" / "p1").mkdir(parents=True)
            (game / "common.rpf").write_bytes(b"x")
            (game / "pack" / "p1" / "p1.rpf").write_bytes(b"xy")
            for name in ("common", "pack_p1_p1"):
                (out / name).mkdir(parents=True)
                (out / name / "index.json").write_text(json.dumps(INDEX), encoding="utf-8")
            buf = ctypes.create_unicode_buffer(1024)
            get_short = ctypes.windll.kernel32.GetShortPathNameW
            get_short.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
            get_short.restype = ctypes.c_uint
            if not get_short(str(game), buf, len(buf)):
                self.skipTest("GetShortPathNameW failed")
            short_game = buf.value + "\\"
            if short_game == str(game) + "\\":
                self.skipTest("8.3 short names are unavailable on this volume")
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
                 "-Game", short_game, "-Out", str(out)], cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("carving", result.stdout)
            rows = json.loads((out / "coverage.json").read_text(encoding="utf-8-sig"))
            self.assertEqual([r["archive"] for r in rows], ["common.rpf", "pack\\p1\\p1.rpf"])

    def test_colliding_archive_names_do_not_share_a_cached_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            game, out = Path(tmp, "game"), Path(tmp, "out")
            (game / "pack").mkdir(parents=True)
            (game / "pack" / "scene.rpf").write_bytes(b"x")
            (game / "PACK_scene.rpf").write_bytes(b"xx")
            cached = out / "pack_scene"
            cached.mkdir(parents=True)
            (cached / "index.json").write_text(json.dumps(INDEX), encoding="utf-8")
            coverage = out / "coverage.json"
            coverage.write_bytes(b"[]\n")

            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
                 "-Game", str(game), "-Out", str(out)], capture_output=True, text=True, timeout=30,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("pack\\scene.rpf", result.stderr)
            self.assertIn("PACK_scene.rpf", result.stderr)
            self.assertNotIn("carving", result.stdout)
            self.assertEqual(coverage.read_bytes(), b"[]\n")


if __name__ == "__main__":
    unittest.main()
