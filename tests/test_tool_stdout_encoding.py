"""Tools must not die printing a path that the console's code page can't write.

On Windows a redirected stdout (tool.py > log.txt, or a pipe from make_game.py) is encoded with the console's ANSI
code page, and `python -I` ignores PYTHONUTF8 and PYTHONIOENCODING. A caller path with a character outside that
code page, such as a Cyrillic, Greek or CJK user name, or a Polish one under a code page without it, reached print()
as UnicodeEncodeError and ended the run. stderr escapes such characters (backslashreplace); stdout was strict. #1406
fixed make_game.py. Every CLI tool now starts its __main__ block by setting errors="replace" on stdout.

The paths below use a private-use character that no ANSI code page maps, so before the fix these tools died on any
Windows locale. On a UTF-8 console the character prints, and the test passes either way. Standard library only;
no game files (the inputs are zero bytes).
"""
import os
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
MARK = ""


class EveryToolGuardsStdout(unittest.TestCase):
    def test_every_main_block_reconfigures_stdout(self):
        missing = []
        for name in sorted(os.listdir(TOOLS)):
            if not name.endswith(".py"):
                continue
            with open(os.path.join(TOOLS, name), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r"^if __name__ == ['\"]__main__['\"]:\n", text, re.M)
            if m and 'sys.stdout.reconfigure(errors="replace")' not in text[m.end():m.end() + 400]:
                missing.append(name)
        self.assertEqual(missing, [], "tools whose __main__ block doesn't set errors='replace' on stdout")


class MeasuredCrashes(unittest.TestCase):
    # Each of these prints its input path to stdout and died with UnicodeEncodeError on Windows 11 given a zeroed
    # input under a private-use-named folder, on every run (audio_wav, hashdict and shader_profile die the same way
    # only once their --out already exists, so they are covered by the static check above rather than here).
    CASES = [("analyze_4240_full.py", False), ("meshlet_page.py", False)]

    def test_a_path_outside_the_code_page_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = os.path.join(tmp, "in-" + MARK)
            os.makedirs(folder)
            src = os.path.join(folder, "x" + MARK + ".bin")
            with open(src, "wb") as f:
                f.write(bytes(4096))
            out = os.path.join(tmp, "out-" + MARK)
            for tool, with_out in self.CASES:
                with self.subTest(tool=tool):
                    argv = [src] + (["--out", out] if with_out else [])
                    r = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, tool)] + argv,
                                       capture_output=True, cwd=ROOT, timeout=300)
                    err = r.stderr.decode("utf-8", "replace")
                    self.assertNotIn("UnicodeEncodeError", err)


if __name__ == "__main__":
    unittest.main()
