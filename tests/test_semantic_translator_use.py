"""Every tests/semantic/test_*.py actually runs tools/rdna2spirv.py.

    python -I tests/test_semantic_translator_use.py

A translator regression test that reimplements the answer in Python and never calls the translator
passes whatever the translator does, so it is not a regression test.  This guard reads the test
files (it does not import them) and requires each one to reach ``rdna2spirv``:

* a direct ``import rdna2spirv`` / ``from rdna2spirv import ...``;
* an import of a helper in tests/semantic (``support``, ``regen_fixtures``, another test module)
  that itself reaches the translator, followed transitively;
* a subprocess/runpy test or a meta-test whose own string names ``rdna2spirv.py``.

``ALLOWLIST`` is deliberately short and reviewed: it holds the interpreter self-tests, the fixture
drift check, and the files whose subject is hand-written SPIR-V.  A new entry needs a reason and a
name that says why the file must not call the translator.

This checks that a file *reaches* the translator, not that every assertion depends on it: a file that
imports ``support`` and asserts nothing about its output still passes.  The pattern it is meant to
stop is the one that imports nothing from the repository at all and models the instruction itself.
"""
import ast
import os
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEMANTIC = os.path.join(ROOT, "tests", "semantic")

# Files that legitimately do not invoke the translator.  Keep this list short; every entry is a
# promise that the file tests something other than translation (or tests the translator's input).
ALLOWLIST = {
    "test_interp.py": "self-test of spirv_mini_interp.py against hand-written SPIR-V",
    "test_spirv_wave_interp.py": "self-test of spirv_wave_interp.py against hand-written SPIR-V",
    "test_fixtures.py": "checks fixtures/disasm/*.s against regen_fixtures.py; the listing is the subject",
    "test_alu_frexp.py": "hand-written SPIR-V run by spirv_mini_interp.py; no translator output",
    "test_alu_ldexp.py": "hand-written SPIR-V run by spirv_mini_interp.py; no translator output",
    "test_alu_lshl_add.py": "hand-written copy of the emitted SPIR-V run by spirv_mini_interp.py",
    "test_alu_pack_in_place.py": "hand-written SPIR-V run by spirv_mini_interp.py; no translator output",
    "test_alu_sad_hi_u32.py": "pure oracle and hand-written SPIR-V run by spirv_mini_interp.py",
    # Debt from before this guard.  Replace these with translating tests rather than adding to this list.
    "test_alu_dot2acc_f16.py": "DEBT: private oracle; v_dot2acc_f16 is not implemented in the translator yet",
    "test_ballot_votes.py": "DEBT: private SPIR-V interpreter; should translate a hand-written listing instead",
    "test_valu_div_fixup.py": "DEBT: private oracle whose rules RDNA2 disagrees with; TEST-545 replaces it",
}


def _parse(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return ast.parse(f.read(), path)


def _imports(tree):
    """(module or helper name, line) for every import, at any nesting depth, relative ones included."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            if node.module:
                yield node.module, node.lineno
        elif isinstance(node, ast.ImportFrom) and node.level == 1:
            # from . import support / from .support import x
            if node.module:
                yield node.module, node.lineno
            else:
                for alias in node.names:
                    yield alias.name, node.lineno


def _docstring_literals(tree):
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                found.add(id(first.value))
    return found


def _process_names_translator(tree):
    """The line of a non-docstring string naming rdna2spirv.py, when the file also drives a process."""
    imports_process = any(name in ("subprocess", "runpy") for name, _ in _imports(tree))
    if not imports_process:
        return None
    docstrings = _docstring_literals(tree)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and "rdna2spirv.py" in node.value and id(node) not in docstrings):
            return node.lineno
    return None


def _first_line(path):
    """A useful line number for a file that never reaches the translator: its first import, else 1."""
    try:
        tree = _parse(path)
    except SyntaxError:
        return 1
    lines = [line for _, line in _imports(tree)]
    return min(lines) if lines else 1


def _reaches(path, seen, semantic_dir):
    """True when `path` imports the translator, a helper that does, or names it to a subprocess."""
    path = os.path.normpath(path)
    if path in seen:
        return False
    seen = seen | {path}
    tree = _parse(path)
    for name, _ in _imports(tree):
        if name.split(".")[0] == "rdna2spirv":
            return True
    if _process_names_translator(tree) is not None:
        return True
    for name, _ in _imports(tree):
        helper = os.path.join(semantic_dir, name.split(".")[0] + ".py")
        if os.path.isfile(helper) and _reaches(helper, seen, semantic_dir):
            return True
    return False


def offenders(semantic_dir=SEMANTIC, allowlist=None):
    """[(relative path, line, reason)] for test files that do not reach rdna2spirv and are not allowed."""
    allowlist = ALLOWLIST if allowlist is None else allowlist
    found = []
    for name in sorted(os.listdir(semantic_dir)):
        if not (name.startswith("test_") and name.endswith(".py")):
            continue
        if name in allowlist:
            continue
        path = os.path.join(semantic_dir, name)
        if _reaches(path, set(), semantic_dir):
            continue
        found.append((os.path.join("tests", "semantic", name), _first_line(path),
                      "imports no rdna2spirv and no helper that does"))
    return found


class TranslatorUseTests(unittest.TestCase):
    def test_every_semantic_test_reaches_the_translator(self):
        bad = offenders()
        if bad:
            self.fail(
                "translator regression tests that never run tools/rdna2spirv.py:\n"
                + "\n".join(f"  {path}:{line}: {why}" for path, line, why in bad)
                + "\n\nCall support.translate_fixture/translate_lines/translate_listing (or import "
                  "rdna2spirv) so the test observes the translator, or add a reviewed ALLOWLIST entry "
                  "in tests/test_semantic_translator_use.py saying why the file must not.")

    def test_allowlist_entries_are_real_and_still_needed(self):
        """A stale entry would silently exempt a file that now calls the translator."""
        for name, why in ALLOWLIST.items():
            path = os.path.join(SEMANTIC, name)
            self.assertTrue(os.path.isfile(path), f"ALLOWLIST names a missing file: {name}")
            self.assertTrue(why.strip(), f"ALLOWLIST entry {name} has no reason")
            self.assertFalse(_reaches(path, set(), SEMANTIC),
                             f"{name} now reaches rdna2spirv; remove its ALLOWLIST entry")

    def test_detector_on_a_synthetic_tree(self):
        """A guard that accepts everything cannot pass: check direct, helper, process and silent files."""
        with tempfile.TemporaryDirectory() as tmp:
            files = {
                "support.py": "import rdna2spirv\n",
                "test_direct.py": "import rdna2spirv\n",
                "test_helper.py": "import support\n",
                "test_nested.py": "def f():\n    import support\n",
                "test_process.py": "import subprocess\nCMD = 'tools/rdna2spirv.py'\n",
                "test_silent.py": "import spirv_mini_interp\n",
            }
            for name, text in files.items():
                with open(os.path.join(tmp, name), "w", encoding="utf-8", newline="\n") as f:
                    f.write(text)
            self.assertEqual(offenders(tmp, allowlist={}), [
                (os.path.join("tests", "semantic", "test_silent.py"), 1,
                 "imports no rdna2spirv and no helper that does")])


if __name__ == "__main__":
    unittest.main()
