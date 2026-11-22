"""Run the Windows build's CTest tests under Wine, on Linux.

Usage:
    python -I scripts/run_wine_tests.py --tests ctest.json --windows-root D:/a/Leonida/Leonida
        [--root DIR] [--wine CMD] [--require NAME ...] [--timeout SECONDS] [--summary FILE] [--label TEXT]

--tests is the output of `ctest --show-only=json-v1` from the Windows build, and --windows-root the checkout it was
made in (forward slashes, as `cygpath -m` prints it). Paths under it are mapped to --root, the Linux checkout (default:
this repository), which must hold the Windows build's runtime/build folder at the same place.

Every test is run and reported. The exit code is 1 if a test named by --require fails or is missing, else 0: the
others show what does and doesn't work under Wine yet without failing the job.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def norm(path: str) -> str:
    return path.replace("\\", "/").rstrip("/")


def map_path(arg: str, windows_root: str, root: str) -> str | None:
    """The Linux path for a path under the Windows checkout, or None if it isn't one."""
    a = norm(arg)
    w = norm(windows_root)
    if a.lower() == w.lower():
        return root
    if a.lower().startswith(w.lower() + "/"):
        return os.path.join(root, *a[len(w) + 1:].split("/"))
    return None


def wine_path(path: str) -> str:
    """Wine's Z: drive is the Unix root, so a Windows program reads /x/y as Z:\\x\\y."""
    return "Z:" + os.path.abspath(path).replace("/", "\\")


def properties(test: dict) -> dict:
    return {p["name"]: p["value"] for p in test.get("properties", [])}


def truthy(value) -> bool:
    """A ctest boolean property. The JSON carries it as a string, so "0" must stay false."""
    return str(value).strip().lower() in ("1", "on", "true", "yes")


def run_test(test: dict, args, wine: list[str]) -> tuple[str, str]:
    """Returns (status, detail): status is pass, fail or skip."""
    command = test.get("command")
    if not command:
        return "fail", "no command (target not built?)"
    exe = map_path(command[0], args.windows_root, args.root)
    if exe is None:
        # A test that runs a tool of the Windows runner (cmake -P, python) rather than one this build made.
        return "skip", f"runs {command[0]}, not a program from the build"
    if not os.path.isfile(exe):
        return "fail", f"executable not in the artifact: {command[0]}"
    argv = list(wine) + [exe]
    for a in command[1:]:
        mapped = map_path(a, args.windows_root, args.root)
        argv.append(wine_path(mapped) if mapped is not None else a)
    props = properties(test)
    cwd = props.get("WORKING_DIRECTORY")
    cwd = map_path(cwd, args.windows_root, args.root) if cwd else os.path.dirname(exe)
    if not cwd or not os.path.isdir(cwd):
        cwd = os.path.dirname(exe)
    start = time.monotonic()
    try:
        proc = subprocess.run(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=args.timeout)
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode("utf-8", "replace")
        return "fail", f"timed out after {args.timeout} s\n{out}"
    except OSError as e:                  # a mistyped --wine, or a missing one: report the test, do not abort the run
        return "fail", f"could not start {argv[0]}: {e.strerror or e}"
    elapsed = time.monotonic() - start
    out = proc.stdout.decode("utf-8", "replace")
    code = proc.returncode
    if code < 0:
        # A negative return code is a host-signal termination (e.g. SIGSEGV), not a process exit code:
        # report it as a failure before SKIP_RETURN_CODE or WILL_FAIL can reinterpret it.
        return "fail", f"killed by signal {-code} in {elapsed:.1f} s\n{out}"
    skip_code = props.get("SKIP_RETURN_CODE")
    if skip_code is not None and code == int(skip_code):
        return "skip", f"exit {code} (skip) in {elapsed:.1f} s\n{out}"
    detail = f"exit {code} in {elapsed:.1f} s\n{out}"
    required = props.get("PASS_REGULAR_EXPRESSION", [])
    forbidden = props.get("FAIL_REGULAR_EXPRESSION", [])
    # A required match takes the place of the exit code. A forbidden match overrides either kind of success.
    try:
        passed = any(re.search(pattern, out, re.DOTALL) for pattern in required) if required else code == 0
        if any(re.search(pattern, out, re.DOTALL) for pattern in forbidden):
            passed = False
    except re.error as e:
        return "fail", f"invalid output regular expression: {e}\n{detail}"
    if truthy(props.get("WILL_FAIL")):
        passed = not passed
    return ("pass" if passed else "fail"), detail


def positive_seconds(text: str) -> float:
    """--timeout: a value of 0 or less made subprocess.run raise ValueError before the test could be reported."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid float value: {text!r}") from None
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return value


def read_tests(path: str) -> list[dict] | None:
    """The tests in a `ctest --show-only=json-v1` file, or None when it is not one."""
    try:
        with open(path, encoding="utf-8-sig") as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        print(f"ERROR: cannot read {path}: {e}", file=sys.stderr)
        return None
    if not isinstance(doc, dict) or not isinstance(doc.get("tests", []), list):
        print(f"ERROR: {path} is not the output of ctest --show-only=json-v1", file=sys.stderr)
        return None
    return doc.get("tests", [])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", required=True, help="ctest --show-only=json-v1 output from the Windows build")
    ap.add_argument("--windows-root", required=True, help="the Windows checkout the tests were listed in")
    ap.add_argument("--root", default=ROOT, help="the Linux checkout (default: this repository)")
    ap.add_argument("--wine", default=os.environ.get("WINE", "wine"), help="Wine command (default: $WINE or wine)")
    ap.add_argument("--require", action="append", default=[], help="a test that must pass (repeatable)")
    ap.add_argument("--timeout", type=positive_seconds, default=120.0, help="seconds per test (default 120)")
    ap.add_argument("--summary", help="append a Markdown table to this file (e.g. $GITHUB_STEP_SUMMARY)")
    ap.add_argument("--label", help="named in the summary's heading, e.g. the Wine version")
    args = ap.parse_args()

    tests = read_tests(args.tests)
    if tests is None:
        return 2
    wine = shlex.split(args.wine)
    results = {}
    for test in tests:
        name = test.get("name", "?")
        status, detail = run_test(test, args, wine)
        results[name] = status
        print(f"[{status.upper()}] {name}", flush=True)
        if status != "pass":
            for line in detail.rstrip().splitlines()[-40:]:
                print(f"    {line}")

    missing = [n for n in args.require if n not in results]
    failed_required = [n for n in args.require if results.get(n) != "pass"]
    counts = {s: sum(1 for v in results.values() if v == s) for s in ("pass", "fail", "skip")}
    print(f"\n{counts['pass']} passed, {counts['fail']} failed, {counts['skip']} skipped under Wine "
          f"({len(results)} tests)")
    for n in missing:
        print(f"required test not listed by the Windows build: {n}")
    if failed_required:
        print("required tests that did not pass: " + ", ".join(failed_required))

    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as f:
            f.write(f"### Windows tests under {args.label or 'Wine'}\n\n")
            f.write(f"{counts['pass']} passed, {counts['fail']} failed, {counts['skip']} skipped.\n\n")
            f.write("| Test | Result | Required |\n|---|---|---|\n")
            for name, status in results.items():
                f.write(f"| `{name}` | {status} | {'yes' if name in args.require else ''} |\n")
            f.write("\n")
    return 1 if failed_required else 0


if __name__ == "__main__":
    sys.exit(main())
