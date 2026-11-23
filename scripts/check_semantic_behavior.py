"""Behavioral sabotage gate for tests/semantic/test_*.py (standard library only).

    python -I scripts/check_semantic_behavior.py tests/semantic/test_alu_int_misc.py
    python -I scripts/check_semantic_behavior.py --base HEAD~1 --head HEAD

`tests/test_semantic_translator_use.py` checks that a semantic test *reaches* `tools/rdna2spirv.py` via AST,
which cannot distinguish a real regression test from one that calls `translate_lines()` and ignores the output.
This gate runs each target test file twice:
  1. Against the real `tools/rdna2spirv.py` (must pass).
  2. Against a temporary copy of `tools/rdna2spirv.py` whose emitted ALU, comparison, bitop, and atomic
     SPIR-V instructions are deterministically inverted (without corrupting `OpConstant` loop bounds or
     register indices). A test that still passes under sabotaged translator output does not assert on the
     translator's behavior and is rejected.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SABOTAGE_SHIM = r'''
# --- LEONIDA SEMANTIC BEHAVIOR GATE SABOTAGE ---
_SAB_OPS = [
    (" = OpIAdd ", " = OpISub "),
    (" = OpISub ", " = OpIAdd "),
    (" = OpFAdd ", " = OpFSub "),
    (" = OpIMul ", " = OpIAdd "),
    (" = OpFMul ", " = OpFAdd "),
    (" = OpBitwiseAnd ", " = OpBitwiseXor "),
    (" = OpBitwiseOr ", " = OpBitwiseAnd "),
    (" = OpBitwiseXor ", " = OpBitwiseOr "),
    (" = OpBitFieldSExtract ", " = OpBitFieldUExtract "),
    (" = OpShiftLeftLogical ", " = OpShiftRightLogical "),
    (" = OpShiftRightArithmetic ", " = OpShiftRightLogical "),
    (" = OpAtomicIAdd ", " = OpAtomicISub "),
    (" = OpFOrdEqual ", " = OpFOrdNotEqual "),
    (" = OpFOrdGreaterThan ", " = OpFOrdLessThan "),
    (" = OpIEqual ", " = OpINotEqual "),
    (" = OpULessThan ", " = OpUGreaterThan "),
    (" = OpSLessThan ", " = OpSGreaterThan "),
    (" = OpLogicalAnd ", " = OpLogicalOr "),
    (" NMin ", " NMax "),
    (" FMin ", " FMax "),
    (" Fract ", " FAbs "),
]


def _sabotage_spv(text):
    if not isinstance(text, str):
        return text
    out = []
    for line in text.splitlines():
        if "OpAccessChain" not in line and "OpBranch" not in line and "OpLoopMerge" not in line:
            for src, dst in _SAB_OPS:
                if src in line:
                    line = line.replace(src, dst, 1)
                    break
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


_orig_translate = translate


def translate(*args, **kwargs):
    return _sabotage_spv(_orig_translate(*args, **kwargs))


if hasattr(Gen, "emit"):
    _orig_emit = Gen.emit

    def _sabotaged_emit(self, line, *args, **kwargs):
        if isinstance(line, str) and "OpAccessChain" not in line and "OpBranch" not in line:
            for src, dst in _SAB_OPS:
                if src in line:
                    line = line.replace(src, dst, 1)
                    break
        return _orig_emit(self, line, *args, **kwargs)

    Gen.emit = _sabotaged_emit
'''


def load_exemptions(root):
    """Read ALLOWLIST and DEBT from tests/test_semantic_translator_use.py when present."""
    tests_dir = os.path.join(root, "tests")
    guard_path = os.path.join(tests_dir, "test_semantic_translator_use.py")
    if not os.path.isfile(guard_path):
        return set(), set()
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    import test_semantic_translator_use as guard
    return set(guard.ALLOWLIST), set(guard.DEBT)


def make_sabotaged_tools(root, tmpdir):
    dst = os.path.join(tmpdir, "tools")
    shutil.copytree(
        os.path.join(root, "tools"),
        dst,
        ignore=shutil.ignore_patterns("__pycache__", "rpf9dump", "bcpack_calibration"),
    )
    target = os.path.join(dst, "rdna2spirv.py")
    with open(target, encoding="utf-8") as f:
        src = f.read()
    with open(target, "w", encoding="utf-8", newline="\n") as f:
        f.write(src + "\n" + SABOTAGE_SHIM)
    return dst


def run_module(semantic_dir, tools_dir, module_name, timeout=30, use_sabotage_env=False):
    py_path = [semantic_dir]
    if use_sabotage_env:
        py_path.insert(0, tools_dir)
    else:
        py_path.append(tools_dir)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=os.pathsep.join(py_path))
    if use_sabotage_env:
        env["LEONIDA_TOOLS_DIR"] = tools_dir
    t0 = time.perf_counter()
    try:
        p = subprocess.run(
            [sys.executable, "-m", "unittest", module_name],
            cwd=semantic_dir,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return p.returncode, round((time.perf_counter() - t0) * 1000.0, 1)
    except subprocess.TimeoutExpired:
        return -9, float(timeout * 1000.0)


def check_files(root, test_paths, timeout=30, include_debt=False):
    root = os.path.abspath(root)
    allowlist, debt = load_exemptions(root)
    real_tools = os.path.join(root, "tools")
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        sab_tools = make_sabotaged_tools(root, tmp)
        for raw_path in test_paths:
            abs_path = os.path.abspath(raw_path if os.path.isabs(raw_path) else os.path.join(root, raw_path))
            if not os.path.isfile(abs_path):
                continue
            fname = os.path.basename(abs_path)
            if fname in allowlist or (not include_debt and fname in debt):
                results.append({"file": fname, "verdict": "EXEMPT", "base_rc": 0, "sab_rc": None, "ms_base": 0.0, "ms_sab": 0.0})
                continue
            sem_dir = os.path.dirname(abs_path)
            mod = os.path.splitext(fname)[0]
            rc_base, ms_base = run_module(sem_dir, real_tools, mod, timeout=timeout, use_sabotage_env=False)
            if rc_base != 0:
                verdict = "TIMEOUT" if rc_base == -9 else "BASE_FAIL"
                results.append({"file": fname, "verdict": verdict, "base_rc": rc_base, "sab_rc": None, "ms_base": ms_base, "ms_sab": 0.0})
                continue
            rc_sab, ms_sab = run_module(sem_dir, sab_tools, mod, timeout=timeout, use_sabotage_env=True)
            if rc_sab == 0:
                verdict = "FAKE_CAUGHT"
            elif rc_sab == -9:
                verdict = "TIMEOUT"
            else:
                verdict = "REAL_PASS"
            results.append({"file": fname, "verdict": verdict, "base_rc": rc_base, "sab_rc": rc_sab, "ms_base": ms_base, "ms_sab": ms_sab})
    return results


def changed_semantic_tests(root, base_ref, head_ref):
    p = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=AM", base_ref, head_ref, "--", "tests/semantic"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    out = []
    for rel in p.stdout.splitlines():
        name = os.path.basename(rel)
        if name.startswith("test_") and name.endswith(".py"):
            out.append(rel)
    return sorted(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Verify tests/semantic/ files actually depend on rdna2spirv.py output.")
    ap.add_argument("files", nargs="*", help="Semantic test file paths to check")
    ap.add_argument("--root", default=ROOT, help="Repository root")
    ap.add_argument("--base", help="Git base ref for diff-based selection")
    ap.add_argument("--head", default="HEAD", help="Git head ref (default: HEAD)")
    ap.add_argument("--timeout", type=int, default=30, help="Per-module timeout in seconds")
    ap.add_argument("--include-debt", action="store_true", help="Also check files listed in DEBT")
    ap.add_argument("--json", dest="json_out", help="Write JSON report to path")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    targets = list(args.files)
    if args.base:
        targets.extend(changed_semantic_tests(root, args.base, args.head))
    targets = sorted(set(targets))

    if not targets:
        print("semantic-behavior-gate: no semantic tests selected")
        return 0

    rows = check_files(root, targets, timeout=args.timeout, include_debt=args.include_debt)
    bad = [r for r in rows if r["verdict"] in ("FAKE_CAUGHT", "BASE_FAIL", "TIMEOUT")]
    report = {"ok": not bad, "checked": len(rows), "failed": len(bad), "rows": rows}
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, indent=2)
            f.write("\n")

    for r in rows:
        print(f"  {r['verdict']:11s} {r['file']} (base={r['base_rc']} sab={r['sab_rc']} {r['ms_sab']} ms)")
    if bad:
        print(
            "semantic-behavior-gate: FAIL — tests marked FAKE_CAUGHT still passed under sabotaged "
            "tools/rdna2spirv.py output. Assert on executed SPIR-V output or translated instructions."
        )
        return 1
    print(f"semantic-behavior-gate: PASS ({len(rows)} checked)")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
