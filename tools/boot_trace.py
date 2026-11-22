"""Summarize `gta6rt`'s two first-boot traces: the startup JSONL and the `[d3d12-trace]` lines.

The loader writes the startup trace with `gta6rt --trace FILE` (one JSON object per line, flushed per event so it
survives a crash; `runtime/README.md`, "Startup trace"), and the D3D12 proxy writes one `[d3d12-trace] #<seq>
<method> <outcome>` line to stderr with `--trace-d3d12`. This tool turns both into the small report a first-boot
triage needs: modules, imports, unimplemented calls and their callers, DllMain/TLS results, exceptions and the
exit, then the D3D12 calls by outcome and the first call per method.

Usage:
    python -I boot_trace.py boot.jsonl [--d3d12 d3d12.log] [--json] [--limit N]

The traces can quote what the game prints, so they stay on the player's machine: the text report counts debug
strings instead of printing them, and only the tests' synthetic fixtures go into the repository. Share counts and
error classes from a real run, not its trace (AGENTS.md, "No game data").

Standard library only; reads files, never runs them. A truncated last line (the expected shape after a crash) is
counted, not fatal; a file with no valid event is an error (exit 2).
"""

from __future__ import annotations

import argparse
import json
import re
import sys

D3D12_LINE = re.compile(r"^\[d3d12-trace\] #(\d+) (\S+) (\S+)\s*$")


class TraceError(ValueError):
    """The file is unreadable or holds no valid trace event."""


def parse_startup(path: str, limit: int = 10) -> dict:
    """Read the `--trace` JSONL into the units the first-boot triage uses."""
    trace = {
        "path": path,
        "pid": None,
        "events": 0,
        "bad_lines": 0,
        "bad_line_numbers": [],
        "kinds": {},
        "modules": [],
        "tls_callbacks": 0,
        "dllmain": {"ok": 0, "failed": 0, "failed_modules": []},
        "unimplemented": [],
        "unimplemented_by_dll": {},
        "debug_strings": 0,
        "exceptions": [],
        "exception_count": 0,
        "exit": None,
        "error": None,
    }
    seen_imports = set()
    try:
        fh = open(path, encoding="utf-8", errors="replace")  # noqa: SIM115 - closed by the with below
    except OSError as e:
        raise TraceError(f"cannot read {path}: {e.strerror or e}") from e
    with fh:
        for number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                trace["bad_lines"] += 1
                if len(trace["bad_line_numbers"]) < limit:
                    trace["bad_line_numbers"].append(number)
                continue
            if not isinstance(event, dict) or "kind" not in event:
                trace["bad_lines"] += 1
                if len(trace["bad_line_numbers"]) < limit:
                    trace["bad_line_numbers"].append(number)
                continue
            trace["events"] += 1
            kind = str(event.get("kind"))
            trace["kinds"][kind] = trace["kinds"].get(kind, 0) + 1
            if kind == "trace_open":
                trace["pid"] = event.get("pid")
            elif kind == "map":
                trace["modules"].append(
                    {
                        "module": event.get("module"),
                        "base": event.get("base"),
                        "size": event.get("size"),
                        "relocated": event.get("relocated"),
                    }
                )
            elif kind == "tls_callback":
                trace["tls_callbacks"] += 1
            elif kind == "dllmain":
                if event.get("result"):
                    trace["dllmain"]["ok"] += 1
                else:
                    trace["dllmain"]["failed"] += 1
                    if len(trace["dllmain"]["failed_modules"]) < limit:
                        trace["dllmain"]["failed_modules"].append(event.get("module"))
            elif kind == "unimplemented_call":
                name = str(event.get("import", "?"))
                if name not in seen_imports:
                    seen_imports.add(name)
                    trace["unimplemented"].append(
                        {"import": name, "caller": event.get("caller")}
                    )
                    dll = name.split("!", 1)[0]
                    trace["unimplemented_by_dll"][dll] = (
                        trace["unimplemented_by_dll"].get(dll, 0) + 1
                    )
            elif kind == "debug_string":
                trace["debug_strings"] += 1
            elif kind == "exception":
                trace["exception_count"] += 1
                if len(trace["exceptions"]) < limit:
                    stack = event.get("stack")
                    trace["exceptions"].append(
                        {
                            "code": event.get("code"),
                            "at": event.get("at"),
                            "cpp": event.get("cpp"),
                            "access": event.get("access"),
                            "address": event.get("address"),
                            "stack": stack[:8] if isinstance(stack, list) else [],
                        }
                    )
            elif kind == "exit":
                trace["exit"] = event.get("code")
            elif kind == "error":
                trace["error"] = event.get("what")
    if trace["events"] == 0:
        raise TraceError(f"{path} has no valid startup event")
    return trace


def parse_d3d12(path: str, limit: int = 10) -> dict:
    """Read the `--trace-d3d12` stderr lines, ignoring the runtime's ordinary diagnostics."""
    trace = {
        "path": path,
        "calls": 0,
        "ignored_lines": 0,
        "outcomes": {},
        "methods": [],
        "first": {},
        "failed": [],
        "unsupported": [],
        "last_seq": 0,
    }
    try:
        fh = open(path, encoding="utf-8", errors="replace")  # noqa: SIM115 - closed by the with below
    except OSError as e:
        raise TraceError(f"cannot read {path}: {e.strerror or e}") from e
    with fh:
        for line in fh:
            match = D3D12_LINE.match(line.rstrip("\n"))
            if not match:
                if line.strip():
                    trace["ignored_lines"] += 1
                continue
            seq, method, outcome = int(match.group(1)), match.group(2), match.group(3)
            trace["calls"] += 1
            trace["last_seq"] = seq
            trace["outcomes"][outcome] = trace["outcomes"].get(outcome, 0) + 1
            if method not in trace["first"]:
                trace["methods"].append(method)
                trace["first"][method] = {"seq": seq, "outcome": outcome}
            if outcome in ("failed", "unsupported"):
                bucket = trace[outcome]
                if len(bucket) < limit:
                    bucket.append({"seq": seq, "method": method})
    return trace


def summarize(startup: dict | None, d3d12: dict | None) -> dict:
    return {"startup": startup, "d3d12": d3d12}


def _tail(items, limit: int) -> str:
    shown = ", ".join(str(x) for x in items[:limit])
    return shown + (f", ... {len(items) - limit} more" if len(items) > limit else "")


def format_text(summary: dict, limit: int) -> str:
    out = []
    startup = summary.get("startup")
    if startup:
        out.append(f"startup trace: {startup['path']}")
        if startup["events"]:
            out.append(
                f"  pid {startup['pid']}, {startup['events']} event(s), {startup['bad_lines']} bad line(s)"
            )
            kinds = ", ".join(f"{k} {v}" for k, v in sorted(startup["kinds"].items()))
            out.append(f"  kinds: {kinds}")
        if startup["modules"]:
            names = [m["module"] for m in startup["modules"]]
            out.append(f"  modules ({len(names)}): {_tail(names, limit)}")
        out.append(f"  tls callbacks: {startup['tls_callbacks']}")
        dm = startup["dllmain"]
        line = f"  dllmain: {dm['ok']} ok, {dm['failed']} failed"
        if dm["failed_modules"]:
            line += f" ({_tail(dm['failed_modules'], limit)})"
        out.append(line)
        if startup["unimplemented"]:
            out.append(f"  unimplemented: {len(startup['unimplemented'])} unique")
            for item in startup["unimplemented"][:limit]:
                out.append(f"    {item['import']} from {item['caller']}")
            if len(startup["unimplemented"]) > limit:
                out.append(f"    ... {len(startup['unimplemented']) - limit} more")
            by_dll = ", ".join(
                f"{dll} {count}"
                for dll, count in sorted(startup["unimplemented_by_dll"].items())
            )
            out.append(f"  by dll: {by_dll}")
        else:
            out.append("  unimplemented: none")
        if startup["debug_strings"]:
            out.append(f"  debug strings: {startup['debug_strings']} (text not shown)")
        if startup["exception_count"]:
            out.append(f"  exceptions: {startup['exception_count']}")
            for exc in startup["exceptions"][:limit]:
                extra = f" ({exc['cpp']})" if exc["cpp"] else ""
                out.append(f"    code {exc['code']} at {exc['at']}{extra}")
                if exc.get("stack"):
                    out.append(f"      stack: {_tail(exc['stack'], limit)}")
        if startup["error"]:
            out.append(f"  error: {startup['error']}")
        if startup["exit"] is not None:
            out.append(f"  exit: {startup['exit']}")
    d3d12 = summary.get("d3d12")
    if d3d12:
        out.append(f"d3d12 trace: {d3d12['path']}")
        outcomes = ", ".join(f"{k} {v}" for k, v in sorted(d3d12["outcomes"].items()))
        out.append(
            f"  {d3d12['calls']} call(s), {d3d12['ignored_lines']} other line(s)"
            + (f": {outcomes}" if outcomes else "")
        )
        if d3d12["methods"]:
            out.append(f"  first per method: {_tail(d3d12['methods'], limit)}")
        if d3d12["unsupported"]:
            unsupported = [f"#{c['seq']} {c['method']}" for c in d3d12["unsupported"]]
            out.append(f"  unsupported: {_tail(unsupported, limit)}")
        if d3d12["failed"]:
            failed = [f"#{c['seq']} {c['method']}" for c in d3d12["failed"]]
            out.append(f"  failed: {_tail(failed, limit)}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Summarize gta6rt's startup and D3D12 first-boot traces."
    )
    ap.add_argument("startup", nargs="?", help="the JSONL written by gta6rt --trace")
    ap.add_argument("--d3d12", help="the stderr log written with --trace-d3d12")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    ap.add_argument(
        "--limit", type=int, default=10, help="items listed per section (default 10)"
    )
    a = ap.parse_args(argv)
    if not a.startup and not a.d3d12:
        print("error: give the startup trace, --d3d12, or both", file=sys.stderr)
        return 2
    if a.limit < 1:
        print("error: --limit must be at least 1", file=sys.stderr)
        return 2
    try:
        startup = parse_startup(a.startup, a.limit) if a.startup else None
        d3d12 = parse_d3d12(a.d3d12, a.limit) if a.d3d12 else None
    except TraceError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    summary = summarize(startup, d3d12)
    if a.json:
        print(json.dumps(summary, indent=1))
    else:
        print(format_text(summary, a.limit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
