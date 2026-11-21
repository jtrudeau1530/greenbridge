"""gamedata_index.py must group RPF9 zlib blocks into payloads without game data.

    python -I tests/test_gamedata_index.py

Every fixture here is synthetic: small XML documents, a text list, a binary blob and two archive
magics. The archive test builds a minimal RPF9-shaped file with a zlib stream and zero padding.
"""
import csv
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "gamedata_index.py")


def chunked(data: bytes, first: int = 65536):
    """Split a payload into a 64 KiB block and the remainder, the way RPF9 stores large files."""
    return [data[:first], data[first:]] if len(data) > first else [data]


class GamedataIndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.carve = os.path.join(self.root, "carve")
        os.makedirs(self.carve)
        self.rows = []
        self.next_off = 0x1000

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, payload, chunks=None):
        parts = chunks if chunks is not None else [payload]
        for part in parts:
            off = self.next_off
            self.next_off += 0x1000
            with open(os.path.join(self.carve, f"{off:012x}.bin"), "wb") as f:
                f.write(part)
            self.rows.append({
                "off": off,
                "stored": len(part),
                "size": len(part),
                "head": part[:16].hex(),
                "pre": "",
            })

    def finish(self):
        with open(os.path.join(self.carve, "index.json"), "w", encoding="utf-8") as f:
            json.dump({"summary": {"archive": "synthetic", "files": len(self.rows)}, "files": self.rows}, f)

    def run_tool(self, target, *args):
        return subprocess.run([sys.executable, "-I", TOOL, target, *args], cwd=self.root,
                              capture_output=True, text=True, encoding="utf-8", errors="replace")

    def test_directory_groups_chunks_and_roots(self):
        self.add(b'<?xml version="1.0"?>\n<RootA><!-- synthetic fixture, longer than 64 bytes --><Item value="1"/></RootA>')
        second = b'<?xml version="1.0"?>\n<RootB>' + b"x" * (65536 - len(b'<?xml version="1.0"?>\n<RootB>'))
        self.add(second, chunked(second + b"</RootB>"))
        text = b"line of text\n" * 6000
        self.add(text, chunked(text))
        self.add(b"RBF0" + bytes(60))
        self.add(b"PSIN" + bytes(60))
        self.add(bytes(range(256)) * 2)
        self.add(b"MHIU" + b"\x17" * 100, [b"MHIU" + b"\x17" * 65532, b"\x17" * 100])
        self.add(b"<Item/>\n</TailRoot>")       # no declaration, root inferred from the close tag
        self.add(b"<Item>mid")                  # continuation whose start block was not carved
        self.finish()

        out = os.path.join(self.root, "index.json")
        p = self.run_tool(self.carve, "--csv", os.path.join(self.root, "index.csv"), "--json", out)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        s = data["summary"]
        self.assertEqual(s["types"]["xml"]["files"], 4)      # RootA, RootB, TailRoot, fragment
        self.assertEqual(s["types"]["text"]["files"], 1)
        self.assertEqual(s["types"]["rbf0"]["files"], 1)
        self.assertEqual(s["types"]["psin"]["files"], 1)
        self.assertEqual(s["types"]["binary"]["files"], 1)
        self.assertEqual(s["types"]["other"]["files"], 1)
        self.assertEqual(s["xml_documents"], 3)
        self.assertEqual(s["xml_with_header"], 2)
        self.assertEqual(s["xml_tail_only"], 1)
        self.assertEqual(s["xml_incomplete"], 1)
        self.assertEqual(set(s["roots"]), {"RootA", "RootB", "TailRoot"})
        self.assertEqual(len(data["rows"]), 9)
        self.assertEqual([r["root"] for r in data["rows"] if r["type"] == "xml" and r["complete"]],
                         ["RootA", "RootB", "TailRoot"])
        self.assertEqual(data["rows"][1]["chunks"], 2)
        self.assertEqual(data["rows"][2]["chunks"], 2)
        self.assertEqual(data["rows"][6]["detail"], "MHIU")
        self.assertEqual(data["rows"][7]["type"], "xml")
        self.assertTrue(data["rows"][7]["complete"])
        self.assertFalse(data["rows"][7]["header"])
        self.assertFalse(data["rows"][8]["complete"])

        with open(os.path.join(self.root, "index.csv"), newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 9)
        self.assertEqual(rows[0]["root"], "RootA")
        self.assertEqual(rows[0]["type"], "xml")

    def test_archive_runs_the_carver_and_reads_roots(self):
        xml = b'<?xml version="1.0"?>\n<RootA><!-- synthetic fixture, longer than 64 bytes --><Item value="1"/></RootA>'
        blob = b"\x00\x01\x02binary\xff" * 32
        start = 16
        body = struct.pack("<I", start)  # RPF9 header: magic + data start
        archive = bytearray(b"9FPR" + body + bytes(8))
        first = zlib.compress(xml)
        archive += first
        archive += bytes(16)
        second = zlib.compress(blob)
        archive += second
        archive += bytes(2 * 1024 * 1024)  # keeps the last stream away from EOF for the carver
        rpf = os.path.join(self.root, "synthetic.rpf")
        with open(rpf, "wb") as f:
            f.write(archive)

        out = os.path.join(self.root, "archive.json")
        p = self.run_tool(rpf, "--json", out)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        s = data["summary"]
        self.assertEqual(s["streams"], 2)
        self.assertEqual(s["types"]["xml"]["files"], 1)
        self.assertEqual(s["types"]["xml"]["inflated"], len(xml))
        self.assertEqual(s["types"]["binary"]["files"], 1)
        self.assertEqual(s["roots"], {"RootA": {"files": 1, "inflated": len(xml)}})


if __name__ == "__main__":
    unittest.main()
