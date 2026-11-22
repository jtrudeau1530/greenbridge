"""gamedata_xml.py report paths: create the parent, never overwrite an input, reject a directory.

Run with ``python -I tests/test_gamedata_xml_report_paths.py``. Set ``LEONIDA_GAMEDATA_XML`` to another
copy of the tool to use as a negative control (main's version tracebacks on a missing folder and a
directory, and overwrites inputs). All inputs are hand-built; no game data is used.
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOL = pathlib.Path(os.environ.get("LEONIDA_GAMEDATA_XML", str(ROOT / "tools" / "gamedata_xml.py")))
XML = (b'<?xml version="1.0" encoding="utf-8"?>\n'
       b'<CDataFileMgr__ContentChangeSetDef version="1"><id>fixture</id></CDataFileMgr__ContentChangeSetDef>')


class ReportPathTest(unittest.TestCase):
    def make_carve(self, folder):
        carve = pathlib.Path(folder) / "x.rpf"
        carve.mkdir()
        (carve / "index.json").write_text(json.dumps({"files": []}), encoding="utf-8")
        return carve

    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", str(TOOL), *map(str, args)],
                              cwd=ROOT, capture_output=True, text=True, timeout=60)

    def test_missing_json_folder_is_created(self):
        with tempfile.TemporaryDirectory() as td:
            carve = self.make_carve(td)
            out = pathlib.Path(td) / "missing" / "sub" / "gamedata.json"
            result = self.run_tool(carve, "--json", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(out.is_file(), result.stderr)
            self.assertIn("summary", json.loads(out.read_text(encoding="utf-8")))

    def test_missing_csv_folder_is_created(self):
        with tempfile.TemporaryDirectory() as td:
            carve = self.make_carve(td)
            out = pathlib.Path(td) / "missing" / "sub" / "gamedata.csv"
            result = self.run_tool(carve, "--csv", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(out.is_file(), result.stderr)
            self.assertTrue(out.read_text(encoding="utf-8").startswith("source,"))

    def test_json_cannot_overwrite_the_carve_index(self):
        with tempfile.TemporaryDirectory() as td:
            carve = self.make_carve(td)
            index = carve / "index.json"
            before = index.read_text(encoding="utf-8")
            result = self.run_tool(carve, "--json", index)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("would overwrite the input", result.stderr)
            self.assertEqual(index.read_text(encoding="utf-8"), before)

    def test_csv_cannot_overwrite_the_carve_index(self):
        with tempfile.TemporaryDirectory() as td:
            carve = self.make_carve(td)
            index = carve / "index.json"
            before = index.read_text(encoding="utf-8")
            result = self.run_tool(carve, "--csv", index)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("would overwrite the input", result.stderr)
            self.assertEqual(index.read_text(encoding="utf-8"), before)

    def test_json_cannot_overwrite_an_xml_target(self):
        with tempfile.TemporaryDirectory() as td:
            doc = pathlib.Path(td) / "document.xml"
            doc.write_bytes(XML)
            before = doc.read_bytes()
            result = self.run_tool(doc, "--json", doc)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("would overwrite the input", result.stderr)
            self.assertEqual(doc.read_bytes(), before)

    def test_json_cannot_overwrite_a_nested_xml(self):
        with tempfile.TemporaryDirectory() as td:
            folder = pathlib.Path(td) / "tree"
            folder.mkdir()
            doc = folder / "nested.xml"
            doc.write_bytes(XML)
            before = doc.read_bytes()
            result = self.run_tool(folder, "--json", doc)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("would overwrite the input", result.stderr)
            self.assertEqual(doc.read_bytes(), before)

    def test_csv_cannot_overwrite_a_nested_xml(self):
        with tempfile.TemporaryDirectory() as td:
            folder = pathlib.Path(td) / "tree"
            folder.mkdir()
            doc = folder / "nested.xml"
            doc.write_bytes(XML)
            before = doc.read_bytes()
            result = self.run_tool(folder, "--csv", doc)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("would overwrite the input", result.stderr)
            self.assertEqual(doc.read_bytes(), before)

    def test_nested_xml_alias_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            folder = pathlib.Path(td) / "tree"
            folder.mkdir()
            (folder / "sub").mkdir()
            doc = folder / "nested.xml"
            doc.write_bytes(XML)
            before = doc.read_bytes()
            result = self.run_tool(folder, "--json", folder / "sub" / ".." / "nested.xml")
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("would overwrite the input", result.stderr)
            self.assertEqual(doc.read_bytes(), before)

    def test_json_cannot_overwrite_a_carve_stream(self):
        with tempfile.TemporaryDirectory() as td:
            carve = pathlib.Path(td) / "x.rpf"
            carve.mkdir()
            stream = carve / "000000000010.bin"
            stream.write_bytes(b"not really xml")
            before = stream.read_bytes()
            (carve / "index.json").write_text(json.dumps({"files": [
                {"off": 0x10, "stored": 14, "size": 14, "type": "xml", "head": ""}
            ]}), encoding="utf-8")
            result = self.run_tool(carve, "--json", stream)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("would overwrite the input", result.stderr)
            self.assertEqual(stream.read_bytes(), before)

    def test_report_inside_the_scanned_folder_still_works(self):
        with tempfile.TemporaryDirectory() as td:
            folder = pathlib.Path(td) / "tree"
            folder.mkdir()
            (folder / "a.xml").write_bytes(XML)
            out = folder / "report.json"
            result = self.run_tool(folder, "--json", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(out.is_file(), result.stderr)

    def test_report_path_cannot_be_a_directory(self):
        with tempfile.TemporaryDirectory() as td:
            carve = self.make_carve(td)
            outdir = pathlib.Path(td) / "outdir"
            outdir.mkdir()
            result = self.run_tool(carve, "--json", outdir)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("is a directory", result.stderr)

    def test_json_and_csv_must_differ(self):
        with tempfile.TemporaryDirectory() as td:
            carve = self.make_carve(td)
            out = pathlib.Path(td) / "report.bin"
            result = self.run_tool(carve, "--json", out, "--csv", out)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("must use different paths", result.stderr)

    def test_reports_in_a_writable_folder_still_work(self):
        with tempfile.TemporaryDirectory() as td:
            carve = self.make_carve(td)
            js = pathlib.Path(td) / "report.json"
            csv_path = pathlib.Path(td) / "report.csv"
            result = self.run_tool(carve, "--json", js, "--csv", csv_path)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(js.is_file() and csv_path.is_file(), result.stderr)


if __name__ == "__main__":
    unittest.main()
