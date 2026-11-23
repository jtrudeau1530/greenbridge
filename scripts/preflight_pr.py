"""Unified local preflight mirroring Leonida's fast CI checks (standard library only).

    python -I scripts/preflight_pr.py
    python -I scripts/preflight_pr.py --semantic

Runs without game data, LLVM, or a GPU:
  1. `python -m compileall -q make_game.py tools tests`
  2. `python -I make_game.py --help`
  3. No-game-data tracked-extension check and `scripts/check_no_game_data.py`
  4. Fast repository-invariant tests (line endings, UTF-8 encoding, CSV, stdout guard, report_paths,
     docs links/flags/paths/tables/lists/python deps, CI registration, semantic translator use)
  5. Viewer streaming tests (`node --test tests/test_*.cjs`) when Node.js is installed
  6. Optional `--semantic`: `python -m unittest discover -s tests/semantic`
"""
import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import time

BAD_EXT_RE = re.compile(
    r"\.(rpf|ttpf|xvd|msixvc|dxbc|fssm|spv|spvasm|dds|bk2|awc|r9modeldata|obj|glb|ogg|wav|exe|dll)$",
    re.I,
)

REPO_RULE_TESTS = [
    "tests/test_check_no_game_data.py",
    "tests/test_ci_registration.py",
    "tests/test_semantic_translator_use.py",
    "tests/test_text_write_line_endings.py",
    "tests/test_text_encoding.py",
    "tests/test_csv_line_endings.py",
    "tests/test_tool_stdout_encoding.py",
    "tests/test_report_paths.py",
    "tests/test_docs_links.py",
    "tests/test_docs_cli_flags.py",
    "tests/test_docs_repo_paths.py",
    "tests/test_docs_table_rows.py",
    "tests/test_docs_design_rows.py",
    "tests/test_design_contract_rows.py",
    "tests/test_design_contracts.py",
    "tests/test_docs_list_items.py",
    "tests/test_docs_python_deps.py",
    "tests/test_preflight_pr.py",
]


def run_step(label, cmd, cwd):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = (time.perf_counter() - t0) * 1000.0
    tail = (p.stdout + p.stderr).strip().splitlines()[-8:] if p.returncode != 0 else []
    return {
        "label": label,
        "ok": p.returncode == 0,
        "rc": p.returncode,
        "ms": round(dt, 1),
        "tail": tail,
    }


def check_tracked_extensions(cwd):
    t0 = time.perf_counter()
    p = subprocess.run(["git", "ls-files"], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        files = []
        for base, dirs, names in os.walk(cwd):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "build", "node_modules")]
            for name in names:
                files.append(os.path.relpath(os.path.join(base, name), cwd).replace(os.sep, "/"))
    else:
        files = p.stdout.splitlines()
    bad = [f for f in files if BAD_EXT_RE.search(f)]
    dt = (time.perf_counter() - t0) * 1000.0
    return {
        "label": "no-game-data:extensions",
        "ok": not bad,
        "rc": 0 if not bad else 1,
        "ms": round(dt, 1),
        "tail": bad[:8],
    }


def collect_jobs(root, python_exe, include_semantic=False):
    jobs = []
    if os.path.isfile(os.path.join(root, "make_game.py")):
        compile_targets = [t for t in ("make_game.py", "tools", "tests") if os.path.exists(os.path.join(root, t))]
        jobs.append(("compileall", [python_exe, "-m", "compileall", "-q", *compile_targets]))
        jobs.append(("make_game:--help", [python_exe, "-I", "make_game.py", "--help"]))
    no_game = os.path.join(root, "scripts", "check_no_game_data.py")
    if os.path.isfile(no_game):
        jobs.append(("no-game-data:magics", [python_exe, "-I", os.path.join("scripts", "check_no_game_data.py")]))
    for rel in REPO_RULE_TESTS:
        if os.path.isfile(os.path.join(root, *rel.split("/"))):
            jobs.append((rel, [python_exe, "-I", os.path.join(*rel.split("/"))]))
    node = shutil.which("node")
    tests_dir = os.path.join(root, "tests")
    if node and os.path.isdir(tests_dir):
        cjs = sorted(
            os.path.join("tests", n)
            for n in os.listdir(tests_dir)
            if n.startswith("test_") and n.endswith(".cjs")
        )
        if cjs:
            jobs.append(("viewer:node-test", [node, "--test", *cjs]))
    if include_semantic and os.path.isdir(os.path.join(root, "tests", "semantic")):
        jobs.append(("semantic:unittest", [python_exe, "-m", "unittest", "discover", "-s", os.path.join("tests", "semantic")]))
    return jobs


def run_preflight(root, jobs_count=0, include_semantic=False, python_exe=None):
    root = os.path.abspath(root)
    python_exe = python_exe or sys.executable
    jobs = collect_jobs(root, python_exe, include_semantic=include_semantic)
    workers = jobs_count if jobs_count > 0 else min(8, os.cpu_count() or 4)
    t0 = time.perf_counter()
    results = []
    if workers <= 1:
        results.append(check_tracked_extensions(root))
        for label, cmd in jobs:
            results.append(run_step(label, cmd, root))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(check_tracked_extensions, root)]
            futs.extend(pool.submit(run_step, label, cmd, root) for label, cmd in jobs)
            for fut in futs:
                results.append(fut.result())
    wall_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    failed = [r for r in results if not r["ok"]]
    return {
        "root": root,
        "ok": not failed,
        "passed": len(results) - len(failed),
        "failed": len(failed),
        "wall_ms": wall_ms,
        "workers": workers,
        "results": sorted(results, key=lambda r: r["label"]),
        "failures": sorted(failed, key=lambda r: r["label"]),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run Leonida's fast local preflight checks.")
    ap.add_argument("--root", default=os.getcwd(), help="Repository root (default: current directory)")
    ap.add_argument("--jobs", type=int, default=0, help="Parallel worker count (default: min(8, cpu_count))")
    ap.add_argument("--semantic", action="store_true", help="Also run tests/semantic via unittest discover")
    ap.add_argument("--list", action="store_true", help="List checks that would run and exit")
    ap.add_argument("--json", dest="json_out", help="Write the JSON report to this path")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    if args.list:
        print("no-game-data:extensions")
        for label, _ in collect_jobs(root, sys.executable, include_semantic=args.semantic):
            print(label)
        return 0

    report = run_preflight(root, jobs_count=args.jobs, include_semantic=args.semantic)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, indent=2)
            f.write("\n")

    status = "PASS" if report["ok"] else "FAIL"
    print(f"preflight: {status} ({report['passed']} passed, {report['failed']} failed, {report['wall_ms']:.0f} ms, {report['workers']} workers)")
    for fail in report["failures"]:
        print(f"  FAIL {fail['label']} (rc={fail['rc']}, {fail['ms']} ms)")
        for line in fail["tail"][-4:]:
            print(f"       {line}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
