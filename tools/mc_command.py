"""Resolve LLVM_MC to an argv list.

LLVM_MC may be one executable (a path, spaces and all) or a whole command line. Windows needs the command-line
form when llvm-mc runs inside WSL, because the official Windows LLVM build has no AMDGPU target:

    set LLVM_MC=wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc
"""
from __future__ import annotations

import os
import shlex


def split_command(raw: str) -> list:
    """Split a command line without eating the backslashes of a Windows path (POSIX shlex would)."""
    raw = raw.strip()
    if not raw:
        return []
    if os.path.isfile(raw):
        return [raw]                    # a plain path, spaces and all
    if os.name != "nt":
        return shlex.split(raw)
    parts = shlex.split(raw, posix=False)
    return [p[1:-1] if len(p) > 1 and p[0] == p[-1] == '"' else p for p in parts]


def mc_command(raw: str | None = None) -> list:
    """LLVM_MC is a path, or a command line: 'wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc'."""
    if raw is None:
        raw = os.environ.get("LLVM_MC", "llvm-mc")
    return split_command(raw) or ["llvm-mc"]
