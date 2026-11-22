"""The Runtime workflow's path filter still lists every file under tools/ that its jobs read.

    python -I tests/test_runtime_paths.py

``.github/workflows/runtime.yml`` starts on the paths it lists, not on everything outside an ignore list
(BUILD-856): a change to a tool the runtime build never reads should not queue a Windows, Wine and llvmpipe run.
The price is that the list can fall behind. If ``tools/rdna2spirv.py`` starts importing a new ``tools/foo.py``,
that pull request still runs Runtime (it touches ``rdna2spirv.py``), but a later one that changes only
``foo.py`` would skip the GPU tests without anyone noticing. This test recomputes what the jobs read and fails
when an entry is missing:

* the seeds are every tracked file under ``runtime/``, the workflow itself and the scripts it runs: a
  ``tools/<name>`` they spell out is needed, and so is a ``tools`` module a Python seed imports (the scripts and
  shader checks put ``tools/`` on ``sys.path``);
* from there, each needed tool's sibling imports, the ``tools/`` data files it names (``gfx10_formats.json`` is
  loaded by name, which a ``.py``-only scan misses), modules named in strings (``importlib``) and ``tools/``
  subdirectories it names (``hw_oracle``), to a fixed point.

It also checks that the ``push`` and ``pull_request`` lists are the same, and that no ``tools/`` entry names a
file that no longer exists. Listing more than the jobs need is allowed on purpose: an extra run costs one
queue slot, a missing entry loses the build for a tool the workflow reads.

Only tracked sources and the workflow are read; no game files, no build tools, no PyYAML.
"""
import ast
import os
import re
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW = ".github/workflows/runtime.yml"

# Entries that are needed for reasons a source scan cannot see: the build's own trees, the workflow, and the
# file that decides the line endings every checked-out source gets.
ALWAYS = ("runtime/**", "third_party/**", WORKFLOW, ".gitattributes")

TOOLS_REF = re.compile(r"tools/([A-Za-z0-9_][A-Za-z0-9_.-]*)")
SCRIPT_REF = re.compile(r"scripts/[A-Za-z0-9_./-]+\.py")


def read(root, rel):
    with open(os.path.join(root, rel), encoding="utf-8", errors="replace") as f:
        return f.read()


def path_lists(root):
    """{event: [paths]} for the events that filter on paths, exactly as the workflow spells them."""
    text = read(root, WORKFLOW)
    lists = {}
    for event in ("push", "pull_request"):
        match = re.search(r"^  %s:\n((?:    .*\n)+)" % event, text, re.M)
        if not match:
            raise AssertionError(f"{WORKFLOW} has no `{event}:` trigger the way this test expects it")
        body = match.group(1)
        paths = re.search(r"^    paths:\n((?:      - .*\n)+)", body, re.M)
        if not paths:
            raise AssertionError(f"{WORKFLOW} `{event}:` has no `paths:` list; update this test with the workflow")
        lists[event] = [line.strip()[1:].strip().strip("\"'") for line in paths.group(1).splitlines()]
    return lists


def covered(path, patterns):
    for pattern in patterns:
        if pattern.endswith("/**"):
            if path.startswith(pattern[:-2]):
                return True
        elif path == pattern:
            return True
    return False


def tracked(root, top):
    result = subprocess.run(["git", "ls-files", "-z", "--", top], cwd=root, capture_output=True, text=True)
    if result.returncode == 0 and result.stdout:
        return [n for n in result.stdout.split("\0") if n]
    names = []                                     # no git: walk, as tests/test_docs_links.py does
    for base, dirs, files in os.walk(os.path.join(root, top)):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "build")]
        for name in files:
            names.append(os.path.relpath(os.path.join(base, name), root).replace(os.sep, "/"))
    return names


def imported_names(source):
    """Top-level module names a Python file imports, plus every string constant (importlib, file names)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set(), set()
    modules, strings = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".")[0])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.add(node.value)
    return modules, strings


def needed(root):
    """{tools entry: why} for everything under tools/ the Runtime jobs read."""
    tools = os.path.join(root, "tools")
    entries = os.listdir(tools)
    modules = {e[:-3] for e in entries if e.endswith(".py")}
    data = {e for e in entries if os.path.isfile(os.path.join(tools, e)) and not e.endswith(".py")}
    subdirs = {e for e in entries if os.path.isdir(os.path.join(tools, e)) and not e.startswith((".", "__"))}

    def entry(name):
        name = name.rstrip(".")
        if name in subdirs:
            return f"tools/{name}/**"
        if name in modules or name.endswith(".py") and name[:-3] in modules:
            return f"tools/{name if name.endswith('.py') else name + '.py'}"
        if name in data:
            return f"tools/{name}"
        return None

    seeds = [f for f in tracked(root, "runtime")] + [WORKFLOW]
    seeds += sorted(set(SCRIPT_REF.findall(read(root, WORKFLOW))))
    found = {}
    for seed in seeds:
        if not os.path.isfile(os.path.join(root, seed)):
            continue
        text = read(root, seed)
        if seed == WORKFLOW:
            # Only what the jobs run: the trigger's own list would make every listed entry "needed".
            text = text[text.find("\njobs:"):] if "\njobs:" in text else ""
        for name in TOOLS_REF.findall(text):
            first = name.split("/")[0]
            hit = entry(first if first in subdirs else name)
            if hit:
                found.setdefault(hit, f"named in {seed}")
        if seed.endswith(".py"):
            own = os.path.dirname(os.path.join(root, seed))
            for module in imported_names(text)[0]:
                if module in modules and not os.path.exists(os.path.join(own, module + ".py")):
                    found.setdefault(f"tools/{module}.py", f"imported by {seed}")

    queue = list(found)
    while queue:
        current = queue.pop()
        if current.endswith("/**"):
            sources = [f for f in tracked(root, current[:-3]) if f.endswith(".py")]
        elif current.endswith(".py"):
            sources = [current]
        else:
            continue
        for source in sources:
            imports, strings = imported_names(read(root, source))
            names = {m + ".py" for m in imports if m in modules}
            for s in strings:
                base = s.rstrip("/").split("/")[-1]
                if s in modules:                       # importlib.import_module("name")
                    names.add(s + ".py")
                elif base in data or base in subdirs or base.endswith(".py") and base[:-3] in modules:
                    names.add(base)                    # a file or directory the tool opens or runs by name
            for name in names:
                hit = entry(name)
                if hit and hit not in found:
                    found[hit] = f"via {source}"
                    queue.append(hit)
    return found


def problems(root):
    lists = path_lists(root)
    issues = []
    events = sorted(lists)
    for other in events[1:]:
        if lists[other] != lists[events[0]]:
            issues.append(f"the `{events[0]}` and `{other}` path lists differ")
    patterns = lists[events[0]]
    for must in ALWAYS:
        if must not in patterns:
            issues.append(f"{must}: missing (always needed)")
    for script in sorted(set(SCRIPT_REF.findall(read(root, WORKFLOW)))):
        if not covered(script, patterns):
            issues.append(f"{script}: run by the workflow but not listed")
    for path, why in sorted(needed(root).items()):
        if not covered(path, patterns):
            issues.append(f"{path}: missing ({why})")
    for pattern in patterns:
        if pattern.startswith("tools/") and not pattern.endswith("/**"):
            if not os.path.exists(os.path.join(root, pattern)):
                issues.append(f"{pattern}: listed but does not exist")
    return issues


class RepositoryRuntimePathsTest(unittest.TestCase):
    def test_the_scan_sees_the_repository(self):
        found = needed(ROOT)
        # The translator the GPU tests run through and the stub generator CMake runs at configure time.
        self.assertIn("tools/rdna2spirv.py", found)
        self.assertIn("tools/gen_d3d12_stubs.py", found)

    def test_every_tool_the_jobs_read_triggers_the_workflow(self):
        issues = problems(ROOT)
        self.assertFalse(issues, f"{WORKFLOW} path filter:\n" + "\n".join(f"  {i}" for i in issues))


class SyntheticTreeTest(unittest.TestCase):
    """The scan flags what a path grep misses, on a tree built for the purpose."""

    def build(self, tmp, listed):
        def write(rel, text):
            path = os.path.join(tmp, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)

        paths = "".join(f'      - "{p}"\n' for p in ALWAYS + ("scripts/**",) + tuple(listed))
        write(WORKFLOW, "name: Runtime\n\non:\n  push:\n    branches: [main]\n    paths:\n" + paths +
              "  pull_request:\n    paths:\n" + paths + "\njobs:\n  x:\n    steps:\n"
              "      - run: python3 -I scripts/run_gpu_tests.py\n")
        write("runtime/CMakeLists.txt", "add_custom_command(COMMAND python tools/translator.py)\n")
        write("scripts/run_gpu_tests.py", "import sys\nimport subgroup_info\n")
        write("tools/translator.py", "import lanes\nimport json\nTABLE = 'formats.json'\nRUN = 'tools/runner.py'\n")
        write("tools/runner.py", "X = 1\n")
        write("tools/lanes.py", "import importlib\nM = importlib.import_module('late')\n")
        write("tools/late.py", "ORACLE = 'oracle'\n")
        write("tools/oracle/run.py", "import helper\n")
        write("tools/helper.py", "X = 1\n")
        write("tools/formats.json", "{}\n")
        write("tools/subgroup_info.py", "X = 1\n")
        write("tools/unrelated.py", "X = 1\n")

    def test_a_complete_list_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.build(tmp, ["tools/translator.py", "tools/lanes.py", "tools/late.py", "tools/oracle/**",
                             "tools/helper.py", "tools/formats.json", "tools/subgroup_info.py",
                             "tools/runner.py"])
            self.assertEqual(problems(tmp), [])
            self.assertNotIn("tools/unrelated.py", needed(tmp))

    def test_each_kind_of_dependency_is_reported_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.build(tmp, ["tools/translator.py", "tools/gone.py"])
            issues = "\n".join(problems(tmp))
            for missing in ("tools/lanes.py", "tools/late.py", "tools/oracle/**", "tools/helper.py",
                            "tools/formats.json", "tools/subgroup_info.py", "tools/runner.py"):
                self.assertIn(f"{missing}: missing", issues)
            self.assertIn("tools/gone.py: listed but does not exist", issues)

    def test_lists_that_differ_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.build(tmp, [])
            path = os.path.join(tmp, WORKFLOW)
            text = read(tmp, WORKFLOW).replace('    paths:\n', '    paths:\n      - "tools/extra.py"\n', 1)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            self.assertTrue(any("path lists differ" in i for i in problems(tmp)))


if __name__ == "__main__":
    unittest.main()
