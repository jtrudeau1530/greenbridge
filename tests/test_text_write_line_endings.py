"""Every text-mode file write in the Python tools names its line ending (BUILD-468).

`open(path, "w")` with no `newline=` turns each "\\n" into "\\r\\n" on Windows, so the same JSON report, manifest or
assembly listing is CRLF there and LF on Linux and macOS: the same input gives byte-different output per platform
(checksums and diffs differ). The CSV writers were fixed in #1573; this covers the other text writers. This test reads
the source with `ast` (standard library, no game data, no GPU) and fails for any `open` / `os.fdopen` in a text write
mode (`w`, `a`, `x`) or any `Path.write_text` under tools/, scripts/, make_game.py and runtime/test/shaders/ that has
no `newline=` argument. Pass `newline="\\n"` for LF output (or `newline=""` when the text carries its own line ends,
like a csv writer).
"""
import ast
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCAN = ("tools", "scripts", "make_game.py", os.path.join("runtime", "test", "shaders"))


def python_files():
    for entry in SCAN:
        path = os.path.join(ROOT, entry)
        if os.path.isfile(path):
            yield path
            continue
        for base, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", "build") and not d.startswith(".")]
            for name in sorted(files):
                if name.endswith(".py"):
                    yield os.path.join(base, name)


def unpinned_text_writes(source: str):
    """Line numbers of text-mode writes in `source` that pass no `newline=`."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "attr", getattr(func, "id", None))
        if any(k.arg == "newline" for k in node.keywords):
            continue
        if name in ("open", "fdopen"):
            mode = None
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value
            for k in node.keywords:
                if k.arg == "mode" and isinstance(k.value, ast.Constant):
                    mode = k.value.value
            if isinstance(mode, str) and "b" not in mode and any(c in mode for c in "wax"):
                found.append(node.lineno)
        elif name == "write_text":
            found.append(node.lineno)
    return found


class TextWriteLineEndings(unittest.TestCase):
    def test_detector_finds_the_unpinned_forms(self):
        src = (
            "open('a', 'w')\n"                                     # 1
            "open('a', 'w', encoding='utf-8')\n"                   # 2
            "open('a', mode='a')\n"                                # 3
            "os.fdopen(3, 'w', encoding='utf-8')\n"                # 4
            "p.write_text('x', encoding='utf-8')\n"                # 5
            "open('a', 'w', newline='\\n')\n"                      # pinned
            "open('a', 'w', newline='')\n"                         # pinned
            "open('a', 'wb')\n"                                    # binary
            "open('a')\n"                                          # read
            "open('a', 'r', encoding='utf-8')\n"                   # read
            "p.write_text('x', newline='\\n')\n"                   # pinned
        )
        self.assertEqual(unpinned_text_writes(src), [1, 2, 3, 4, 5])

    def test_every_text_write_pins_its_newline(self):
        bad = []
        count = 0
        for path in python_files():
            count += 1
            with open(path, encoding="utf-8") as f:
                for line in unpinned_text_writes(f.read()):
                    bad.append(f"{os.path.relpath(path, ROOT)}:{line}")
        self.assertGreater(count, 50, "the scan found almost no Python files; is the layout different?")
        self.assertEqual(bad, [], "text writes without newline= (CRLF on Windows, LF elsewhere): " + ", ".join(bad))


if __name__ == "__main__":
    unittest.main()
