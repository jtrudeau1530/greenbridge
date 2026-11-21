"""Synthetic FSSM compute headers shared by both GPU test runners."""
import re
import struct
from pathlib import Path


def compute_header(asm_path):
    # Default: four user SGPRs (V#), TGID_X, 512 bytes LDS, 64 threads.
    rsrc2, group = 0x8088, (64, 1, 1)
    for line in Path(asm_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.startswith(";"):
            break
        size = re.match(r";\s*group:\s*(\d+),(\d+),(\d+)", line)
        if size:
            group = tuple(map(int, size.groups()))
        elif line[1:].lstrip().startswith("rsrc2:"):
            rsrc2 = int(line.split(":", 1)[1].strip(), 0)
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, rsrc2, *group)
    return header
