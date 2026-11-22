"""tools/progress_dashboard.py: parsers, classification, the no-game-data rule and the output-folder rule.

    python -I tests/test_progress_dashboard.py

Every input is written here by hand (Markdown tables, JUnit XML, a CTest log, a fake work folder whose shader names
are sentinels). No game files, no GPU, no network: the GitHub path is exercised through its parsers only.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import progress_dashboard as pd  # noqa: E402

TOOL = os.path.join(ROOT, "tools", "progress_dashboard.py")
SENTINEL = "SENTINEL_SHADER_NAME_must_not_appear"


class Markdown(unittest.TestCase):
    def test_tables_keep_pipes_inside_code(self):
        t = pd.md_tables("## Translator status and runtime contract\n\n| Piece | Contract |\n|---|---|\n| A | `x|y` is **GPU-checked** |\n")
        self.assertEqual(len(t), 1)
        heading, header, rows, line = t[0]
        self.assertEqual(heading, "Translator status and runtime contract")
        self.assertEqual(rows[0][1], "`x|y` is **GPU-checked**")
        self.assertEqual(line, 3)

    def test_plain_strips_links_and_emphasis(self):
        self.assertEqual(pd.plain("**GPU-checked** ([measurement](http://x)) `v_mac_f32`"), "GPU-checked (measurement) v_mac_f32")


class Contract(unittest.TestCase):
    def test_classification_order_and_evidence(self):
        cases = {
            "**not yet GPU-verified**; Source: LLVM `SIInstrInfo.cpp`": ("not_gpu", "partial"),
            "**GPU-checked** (TEST-298) on an NVIDIA GeForce RTX 2060": ("gpu", "verified"),
            "Source: LLVM `VOPProfileI2F`": ("reference", "implemented"),
            "a count over the corpus: 0 of the 26,503 shaders": ("corpus", "implemented"),
            "CPU-checked by DenormalFlushTests": ("cpu", "implemented"),
            "The translator emits OpFAdd.": ("unstated", "unknown"),
        }
        for text, (kind, status) in cases.items():
            with self.subTest(text=text):
                k, s, words = pd.classify_contract(text)
                self.assertEqual((k, s), (kind, status))
                if kind != "unstated":
                    self.assertIn(words.lower(), text.lower())

    def test_open_items_are_detected_only_by_explicit_words(self):
        self.assertTrue(pd.OPEN.search("Still open: on some inputs the result is undefined"))
        self.assertFalse(pd.OPEN.search("s_getreg_b32 is unsupported and there are no traps"))


class Tests(unittest.TestCase):
    def test_junit_ctest_and_pytest_flavours(self):
        ctest = '<testsuite><testcase name="loader" classname="loader" status="run" time="0.9"/>' \
                '<testcase name="xmem" classname="xmem" status="fail"><failure message="C:\\Users\\me\\x\\out.txt bad"/></testcase></testsuite>'
        rows = pd.parse_junit(ctest)
        self.assertEqual([(n, r) for n, r, _, _ in rows], [("loader", "pass"), ("xmem", "fail")])
        self.assertNotIn("me", rows[1][3])          # paths are reduced to their last component
        pytest = '<testsuites><testsuite><testcase classname="tests.a" name="t[1-2]"/>' \
                 '<testcase classname="tests.a" name="t[1_2]"><skipped/></testcase></testsuite></testsuites>'
        self.assertEqual([(n, r) for n, r, _, _ in pd.parse_junit(pytest)], [("tests.a::t[1-2]", "pass"), ("tests.a::t[1_2]", "skip")])

    def test_lasttest_log(self):
        log = "Start testing: Nov 22 10:00 X\n1/2 Test: loader\nTest time =   0.91 sec\nTest Passed.\n2/2 Test: xmem\nTest Failed.\n"
        self.assertEqual(pd.parse_lasttest(log), [("loader", "pass", 0.91), ("xmem", "fail", None)])

    def test_ci_selection_matches_the_workflow_rule(self):
        files = pd.ci_test_files()
        self.assertIn("tests/test_progress_dashboard.py", files)
        self.assertFalse(any(f.startswith("tests/semantic/") for f in files))
        for f in files[:20]:
            with open(os.path.join(ROOT, f), encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("__main__", text)
            self.assertIsNone(pd.PYTEST_MARKERS.search(text))


class Freshness(unittest.TestCase):
    def collect(self, results):
        with tempfile.TemporaryDirectory() as out:
            with open(os.path.join(out, "unit_results.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(results, f)
            col = pd.Collector()
            pd.collect_unit(col, out, type("A", (), {"run_tests": False})())
            return {c["name"]: (c["status"], c["freshness"]) for c in col.components}

    def base(self, digest, code):
        return {"ran_at": "2026-11-22T10:00:00Z", "commit": None, "code_digest": code,
                "results": [{"test": "tests/test_progress_dashboard.py", "result": "pass", "digest": digest}]}

    def test_unchanged_test_and_code_is_current(self):
        got = self.collect(self.base(pd.file_digest("tests/test_progress_dashboard.py"), pd.code_digest()))
        self.assertEqual(got["test_progress_dashboard.py"], ("verified", "current"))

    def test_changed_test_file_needs_revalidation(self):
        got = self.collect(self.base("0" * 16, pd.code_digest()))
        self.assertEqual(got["test_progress_dashboard.py"], ("partial", "revalidation_required"))

    def test_changed_tools_need_revalidation(self):
        got = self.collect(self.base(pd.file_digest("tests/test_progress_dashboard.py"), "0" * 16))
        self.assertEqual(got["test_progress_dashboard.py"], ("partial", "revalidation_required"))

    def test_code_edited_during_the_run_is_never_current(self):
        got = self.collect(self.base(pd.file_digest("tests/test_progress_dashboard.py"), None))
        self.assertEqual(got["test_progress_dashboard.py"], ("verified", "unknown"))


class Github(unittest.TestCase):
    def test_d3d_states(self):
        self.assertEqual(pd.d3d_state("@a", "#291 merged, #819, #831 on main", "")[0], "partial")
        self.assertEqual(pd.d3d_state("@a", "#1489 merged, #907 closed", "")[0], "implemented")
        self.assertEqual(pd.d3d_state("@a", "—", "")[0], "unknown")
        self.assertEqual(pd.d3d_state("—", "—", "")[0], "missing")


class Corpus(unittest.TestCase):
    def test_counts_only_never_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = os.path.join(tmp, "work")
            os.makedirs(os.path.join(work, "dxbc"))
            os.makedirs(os.path.join(work, "spirv"))
            carved = [{"name": f"{SENTINEL}_{i}", "stage": "pixel" if i < 3 else "compute"} for i in range(5)]
            with open(os.path.join(work, "dxbc", "index.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(carved, f)
            with open(os.path.join(work, "spirv", "index.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump({f"{SENTINEL}_0": {"stage": "pixel"}, f"{SENTINEL}_3": {"stage": "compute"},
                           f"{SENTINEL}_4": {"error": "x"}}, f)
            col = pd.Collector()
            pd.collect_corpus(col, work)
            got = {(c["group"], c["status"]): c["count"] for c in col.components}
            self.assertEqual(got, {("pixel", "implemented"): 1, ("pixel", "failed"): 2,
                                   ("compute", "implemented"): 1, ("compute", "failed"): 1})
            m = next(m for m in col.metrics if m["id"] == "corpus.validated")
            self.assertEqual((m["numerator"], m["denominator"]), (2, 5))
            self.assertNotIn(SENTINEL, json.dumps({"c": col.components, "s": col.sources, "m": col.metrics}))
            self.assertNotIn(tmp, json.dumps(col.sources))


class Outputs(unittest.TestCase):
    def test_out_inside_the_repository_is_refused(self):
        r = subprocess.run([sys.executable, "-I", TOOL, "--out", os.path.join(ROOT, "progress_out_test")],
                           capture_output=True, cwd=ROOT, timeout=120)
        self.assertEqual(r.returncode, 2)
        self.assertFalse(os.path.exists(os.path.join(ROOT, "progress_out_test")))

    def test_page_snapshot_history_and_dedupe(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            cmd = [sys.executable, "-I", TOOL, "--out", out]
            r1 = subprocess.run(cmd, capture_output=True, cwd=ROOT, timeout=300)
            self.assertEqual(r1.returncode, 0, r1.stderr.decode("utf-8", "replace"))
            with open(os.path.join(out, "progress.html"), encoding="utf-8") as f:
                page = f.read()
            self.assertNotIn("/*__PROGRESS_DATA__*/null", page)
            self.assertIn('"schema": "leonida.progress/1"', page)
            with open(os.path.join(out, "progress.html"), "rb") as f:
                self.assertNotIn(b"\r\n", f.read())             # LF on every platform
            r2 = subprocess.run(cmd, capture_output=True, cwd=ROOT, timeout=300)
            self.assertIn(b"no change since the last snapshot", r2.stdout)
            self.assertEqual(len(os.listdir(os.path.join(out, "snapshots"))), 1)
            with open(os.path.join(out, "history.json"), encoding="utf-8") as f:
                self.assertEqual(len(json.load(f)), 1)

    def test_script_element_cannot_be_closed_by_data(self):
        snap = {"schema": pd.SCHEMA, "x": "</script><b>"}
        self.assertNotIn("</script><b>", pd.render_page(snap, []))


if __name__ == "__main__":
    unittest.main()
