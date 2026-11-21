"""Synthetic FSSM compute headers shared by both GPU test runners."""
import struct


def compute_header(name):
    # Default: four user SGPRs (V#), TGID_X, 512 bytes LDS, 64 threads.
    rsrc2, group = 0x8088, (64, 1, 1)
    if name == "cs_workgroup_yz":
        rsrc2, group = 0x308, (1, 1, 1)  # four user SGPRs, TGID_Y and TGID_Z
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, rsrc2, *group)
    return header
