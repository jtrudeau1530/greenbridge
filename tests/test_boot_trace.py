"""Unit tests for tools/boot_trace.py on synthetic traces (no game data, no Windows, no GPU)."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import boot_trace as bt

STARTUP = [
    {"seq": 1, "t": 0.0, "kind": "trace_open", "pid": 123},
    {
        "seq": 2,
        "kind": "map",
        "module": "game.exe",
        "base": "0x140000000",
        "size": "0x1000",
        "relocated": False,
    },
    {
        "seq": 3,
        "kind": "map",
        "module": "x.dll",
        "base": "0x180000000",
        "size": "0x2000",
        "relocated": True,
    },
    {
        "seq": 4,
        "kind": "tls_callback",
        "module": "x.dll",
        "at": "x.dll+0x10",
        "reason": 1,
    },
    {"seq": 5, "kind": "dllmain", "module": "x.dll", "result": 1},
    {"seq": 6, "kind": "dllmain", "module": "y.dll", "result": 0},
    {
        "seq": 7,
        "kind": "unimplemented_call",
        "import": "x.dll!Foo",
        "caller": "game.exe+0x10",
    },
    {
        "seq": 8,
        "kind": "unimplemented_call",
        "import": "x.dll!Foo",
        "caller": "game.exe+0x20",
    },
    {
        "seq": 9,
        "kind": "unimplemented_call",
        "import": "y.dll!Bar",
        "caller": "game.exe+0x30",
    },
    {"seq": 10, "kind": "debug_string", "text": "secret game text"},
    {
        "seq": 11,
        "kind": "exception",
        "code": "0xc0000005",
        "at": "game.exe+0x40",
        "access": "read",
        "address": "0x0",
        "stack": ["game.exe+0x40", "game.exe+0x120"],
    },
    {"seq": 12, "kind": "exit", "code": 0},
]

D3D12 = [
    "[rt] ordinary diagnostic",
    "[d3d12-trace] #1 D3D12XboxCreateDevice ok",
    "[d3d12-trace] #2 CreateCommandQueue failed",
    "[d3d12-trace] #3 Foo unsupported",
    "[d3d12-trace] #4 D3D12XboxCreateDevice ok",
]


class BootTraceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.startup = os.path.join(self.tmp.name, "boot.jsonl")
        self.d3d12 = os.path.join(self.tmp.name, "d3d12.log")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, path, events):
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(event) + "\n" for event in events)

    def test_startup_summary(self):
        self.write(self.startup, STARTUP)
        trace = bt.parse_startup(self.startup)
        self.assertEqual(trace["pid"], 123)
        self.assertEqual(trace["events"], len(STARTUP))
        self.assertEqual(trace["bad_lines"], 0)
        self.assertEqual(trace["kinds"]["map"], 2)
        self.assertEqual([m["module"] for m in trace["modules"]], ["game.exe", "x.dll"])
        self.assertEqual(trace["tls_callbacks"], 1)
        self.assertEqual(
            trace["dllmain"], {"ok": 1, "failed": 1, "failed_modules": ["y.dll"]}
        )
        self.assertEqual(
            [u["import"] for u in trace["unimplemented"]], ["x.dll!Foo", "y.dll!Bar"]
        )
        self.assertEqual(trace["unimplemented"][0]["caller"], "game.exe+0x10")
        self.assertEqual(trace["unimplemented_by_dll"], {"x.dll": 1, "y.dll": 1})
        self.assertEqual(trace["debug_strings"], 1)
        self.assertEqual(trace["exception_count"], 1)
        self.assertEqual(
            trace["exceptions"][0]["stack"], ["game.exe+0x40", "game.exe+0x120"]
        )
        self.assertEqual(trace["exit"], 0)

    def test_debug_text_is_not_printed(self):
        self.write(self.startup, STARTUP)
        text = bt.format_text(bt.summarize(bt.parse_startup(self.startup), None), 10)
        self.assertIn("debug strings: 1", text)
        self.assertNotIn("secret game text", text)

    def test_truncated_line_is_counted_not_fatal(self):
        self.write(self.startup, STARTUP)
        with open(self.startup, "a", encoding="utf-8") as f:
            f.write('{"seq": 13, "kind":')
        trace = bt.parse_startup(self.startup)
        self.assertEqual(trace["events"], len(STARTUP))
        self.assertEqual(trace["bad_lines"], 1)
        self.assertEqual(trace["bad_line_numbers"], [len(STARTUP) + 1])

    def test_d3d12_summary(self):
        with open(self.d3d12, "w", encoding="utf-8") as f:
            f.write("\n".join(D3D12) + "\n")
        trace = bt.parse_d3d12(self.d3d12)
        self.assertEqual(trace["calls"], 4)
        self.assertEqual(trace["ignored_lines"], 1)
        self.assertEqual(trace["outcomes"], {"ok": 2, "failed": 1, "unsupported": 1})
        self.assertEqual(
            trace["methods"], ["D3D12XboxCreateDevice", "CreateCommandQueue", "Foo"]
        )
        self.assertEqual(
            trace["first"]["D3D12XboxCreateDevice"], {"seq": 1, "outcome": "ok"}
        )
        self.assertEqual(trace["failed"], [{"seq": 2, "method": "CreateCommandQueue"}])
        self.assertEqual(trace["unsupported"], [{"seq": 3, "method": "Foo"}])
        self.assertEqual(trace["last_seq"], 4)

    def test_json_cli(self):
        self.write(self.startup, STARTUP)
        with open(self.d3d12, "w", encoding="utf-8") as f:
            f.write("\n".join(D3D12) + "\n")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(
                bt.main([self.startup, "--d3d12", self.d3d12, "--json"]), 0
            )
        data = json.loads(out.getvalue())
        self.assertEqual(data["startup"]["pid"], 123)
        self.assertEqual(data["d3d12"]["calls"], 4)

    def test_missing_and_empty_files_are_errors(self):
        self.assertEqual(bt.main([os.path.join(self.tmp.name, "missing.jsonl")]), 2)
        open(self.startup, "w").close()
        self.assertEqual(bt.main([self.startup]), 2)
        self.assertEqual(bt.main(["--limit", "0", self.startup]), 2)

    def test_text_report_has_the_triage_units(self):
        self.write(self.startup, STARTUP)
        text = bt.format_text(bt.summarize(bt.parse_startup(self.startup), None), 10)
        for needle in (
            "pid 123",
            "modules (2)",
            "dllmain: 1 ok, 1 failed",
            "unimplemented: 2 unique",
            "x.dll!Foo from game.exe+0x10",
            "exceptions: 1",
            "stack: game.exe+0x40, game.exe+0x120",
            "exit: 0",
        ):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
