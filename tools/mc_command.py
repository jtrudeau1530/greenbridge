"""Resolve LLVM_MC to an argv list.

LLVM_MC may be one executable (a path, spaces and all, with or without its .exe) or a whole command line. Windows
needs the command-line form when llvm-mc runs inside WSL, because the official Windows LLVM build has no AMDGPU
target:

    set LLVM_MC=wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc

run() starts such a command and turns a missing llvm-mc into one line instead of a FileNotFoundError traceback.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess


def split_command(raw: str) -> list[str]:
    """Split a command line without eating the backslashes of a Windows path (POSIX shlex would)."""
    raw = raw.strip()
    if not raw:
        return []
    if os.path.isfile(raw):
        return [raw]                    # a plain path, spaces and all
    if ("/" in raw or "\\" in raw) and shutil.which(raw):
        # A path whose extension is left out (C:\Program Files\LLVM\bin\llvm-mc): Windows finds llvm-mc.exe
        # for it, but it is no file under that exact name, so it used to be split at its spaces.
        return [raw]
    try:
        if os.name == "nt":
            # Quotes can start inside an argument, as in -o="path with spaces".
            lexer = shlex.shlex(raw, posix=True)
            lexer.quotes = '"'
            lexer.escape = ""
            lexer.commenters = ""
            lexer.whitespace_split = True
            return list(lexer)
        parts = shlex.split(raw)
    except ValueError as e:
        # shader_disasm.py calls mc_command() at import time: a bad LLVM_MC must give one clear line,
        # not a ValueError traceback before any tool's own argument parsing runs.
        raise SystemExit(f"LLVM_MC: invalid command line {raw!r}: {e}") from None
    return [p[1:-1] if len(p) > 1 and p[0] == p[-1] == '"' else p for p in parts]


def mc_command(raw: str | None = None) -> list:
    """LLVM_MC is a path, or a command line: 'wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc'."""
    if raw is None:
        raw = os.environ.get("LLVM_MC", "llvm-mc")
    return split_command(raw) or ["llvm-mc"]


def not_found(exe: str, error: OSError, flag: str | None = None) -> str:
    """The message for an llvm-mc that can't be started (shader_disasm.py's wording). `flag` is the tool's own
    option for the command, if it has one (run_gpu_tests.py's --llvm-mc); the others read LLVM_MC only."""
    where = f"set LLVM_MC (or pass {flag})" if flag else "set LLVM_MC"
    return (f"'{exe}' could not be started: {error}. Put llvm-mc on PATH or {where} to it\n"
            f"(on Windows, scripts\\build_llvm_mc.ps1 builds one and sets it), e.g.\n"
            f"  LLVM_MC=wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc")


def run(cmd: list, flag: str | None = None, **kwargs) -> subprocess.CompletedProcess:
    """subprocess.run for an llvm-mc command line; one that can't be started ends the tool with not_found()."""
    try:
        return subprocess.run(cmd, **kwargs)
    except OSError as e:
        raise SystemExit(not_found(cmd[0], e, flag)) from None
