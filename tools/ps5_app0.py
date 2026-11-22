"""Inspect a readable PlayStation 5 application directory."""

import json
import os
from pathlib import Path
import re
import struct


ELF_HEADER = struct.Struct("<16sHHIQQQIHHHHHH")
PROGRAM_HEADER = struct.Struct("<IIQQQQQQ")
SELF_MAGICS = (b"\x54\x14\xf5\xee", b"\x4f\x15\x3d\x1d")


def executable_header(path: Path) -> None:
    if not path.is_file():
        raise ValueError("The app0 directory is missing eboot.bin.")
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        header = stream.read(ELF_HEADER.size)
        if header[:4] in SELF_MAGICS:
            raise ValueError("eboot.bin is a SELF container. The PS5 check currently accepts a readable ELF.")
        if len(header) != ELF_HEADER.size or header[:7] != b"\x7fELF\x02\x01\x01":
            raise ValueError("eboot.bin must be a little-endian 64-bit ELF.")
        _, kind, machine, version, entry, table, _, _, ehsize, phsize, count, _, _, _ = ELF_HEADER.unpack(header)
        if machine != 62 or version != 1 or kind not in (2, 3, 0xFE10):
            raise ValueError("eboot.bin must be an x86-64 ELF executable.")
        if ehsize != ELF_HEADER.size or phsize != PROGRAM_HEADER.size or not count:
            raise ValueError("eboot.bin has an invalid ELF program-header layout.")
        if table < ELF_HEADER.size or table + count * phsize > size:
            raise ValueError("eboot.bin has a program-header table outside the file.")
        stream.seek(table)
        executable_entry = False
        for _ in range(count):
            program_header = stream.read(phsize)
            if len(program_header) != phsize:
                raise ValueError("eboot.bin changed while reading its program headers.")
            kind, flags, offset, address, _, stored, memory, _ = PROGRAM_HEADER.unpack(program_header)
            if kind != 1:
                continue
            if stored > memory or offset + stored > size or address + memory > 1 << 64:
                raise ValueError("eboot.bin has an invalid ELF load segment.")
            executable_entry |= bool(flags & 1) and address <= entry < address + stored
        if not executable_entry:
            raise ValueError("eboot.bin has no executable, file-backed entry point.")


def check_app0(directory: str) -> str:
    app0 = Path(directory)
    if not app0.is_dir():
        raise ValueError("Specify a readable PS5 app0 directory.")
    executable_header(app0 / "eboot.bin")
    metadata = app0 / "sce_sys" / "param.json"
    if not metadata.is_file():
        raise ValueError("The app0 directory is missing sce_sys/param.json.")
    with metadata.open(encoding="utf-8-sig") as stream:
        param = json.load(stream)
    title = param.get("titleId") if isinstance(param, dict) else None
    if not isinstance(title, str) or re.fullmatch(r"PPSA[0-9]{5}", title) is None:
        raise ValueError("sce_sys/param.json is missing a PS5 title ID.")
    return title
