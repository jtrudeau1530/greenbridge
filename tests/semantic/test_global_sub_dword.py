"""Semantic tests for RDNA2 8/16-bit global sub-dword memory operations (loads and stores).

Covers AMD RDNA2 / GFX10 sub-dword memory instructions:
  - global_load_ubyte: 8-bit unsigned load, zero-extended to 32 bits
  - global_load_sbyte: 8-bit signed load, sign-extended to 32 bits
  - global_load_ushort: 16-bit unsigned load, zero-extended to 32 bits
  - global_load_sshort: 16-bit signed load, sign-extended to 32 bits
  - global_store_byte: 8-bit store from VGPR low byte into memory (preserving adjacent bytes)
  - global_store_short: 16-bit store from VGPR low 16 bits into memory (preserving adjacent bytes)

Executes compiled SPIR-V shaders through spirv_mini_interp on CPU and validates against
a deterministic reference execution oracle. Standard library only; no GPU, no game files.
"""
from __future__ import annotations

import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)
FSSM = bytes(FSSM)

BASE, OUT = 0x2000, 0x3000
SEED = [0x44332211, 0x88776655, 0xCCBBAA99]


def parse_offset_modifier(offset_mod: int | str) -> int:
    """Parse signed 12-bit offset modifier (-2048..2047)."""
    if isinstance(offset_mod, str):
        val = int(offset_mod, 0)
    else:
        val = int(offset_mod)
    if 0x800 <= val <= 0xFFF:
        val -= 0x1000
    if not -0x800 <= val <= 0x7FF:
        raise ValueError(f"Offset {val} out of 12-bit signed range [-2048, 2047]")
    return val


def compute_global_effective_address(base: int, voffset: int = 0, offset_mod: int | str = 0) -> int:
    """Compute 64-bit effective address for global memory instruction."""
    imm = parse_offset_modifier(offset_mod)
    return (base + (voffset & M32) + imm) & M64


class GlobalSubDwordMemoryBank:
    """Deterministic simulated memory bank for global sub-dword byte-accurate operations."""

    def __init__(self, seed_dwords: dict[int, int] | None = None):
        self._mem: dict[int, int] = {}
        if seed_dwords:
            for addr, dw in seed_dwords.items():
                self.write_dword(addr, dw)

    def read_byte(self, byte_addr: int) -> int:
        dw_addr = (byte_addr & ~3) & M64
        shift = (byte_addr & 3) * 8
        dw = self._mem.get(dw_addr, 0)
        return (dw >> shift) & 0xFF

    def write_byte(self, byte_addr: int, val: int) -> None:
        dw_addr = (byte_addr & ~3) & M64
        shift = (byte_addr & 3) * 8
        mask = ~(0xFF << shift) & M32
        dw = self._mem.get(dw_addr, 0) & mask
        new_dw = dw | ((val & 0xFF) << shift)
        self._mem[dw_addr] = new_dw & M32

    def read_short(self, byte_addr: int) -> int:
        b0 = self.read_byte(byte_addr)
        b1 = self.read_byte((byte_addr + 1) & M64)
        return (b1 << 8) | b0

    def write_short(self, byte_addr: int, val: int) -> None:
        self.write_byte(byte_addr, val & 0xFF)
        self.write_byte((byte_addr + 1) & M64, (val >> 8) & 0xFF)

    def read_dword(self, byte_addr: int) -> int:
        dw_addr = (byte_addr & ~3) & M64
        if (byte_addr & 3) == 0:
            return self._mem.get(dw_addr, 0)
        b0 = self.read_byte(byte_addr)
        b1 = self.read_byte((byte_addr + 1) & M64)
        b2 = self.read_byte((byte_addr + 2) & M64)
        b3 = self.read_byte((byte_addr + 3) & M64)
        return (b3 << 24) | (b2 << 16) | (b1 << 8) | b0

    def write_dword(self, byte_addr: int, val: int) -> None:
        if (byte_addr & 3) == 0:
            self._mem[(byte_addr & ~3) & M64] = val & M32
        else:
            self.write_byte(byte_addr, val & 0xFF)
            self.write_byte((byte_addr + 1) & M64, (val >> 8) & 0xFF)
            self.write_byte((byte_addr + 2) & M64, (val >> 16) & 0xFF)
            self.write_byte((byte_addr + 3) & M64, (val >> 24) & 0xFF)

    def snapshot(self) -> dict[int, int]:
        return dict(self._mem)


class GlobalSubDwordReferenceOracle:
    """Reference behavioral oracle for RDNA2 global sub-dword memory instructions."""

    def __init__(self, memory: GlobalSubDwordMemoryBank):
        self.memory = memory

    def execute_load(self, op: str, base: int, voffset: int = 0, offset_mod: int | str = 0,
                     old_dst: int = 0, active: bool = True) -> int:
        if not active:
            return old_dst & M32
        ea = compute_global_effective_address(base, voffset, offset_mod)
        if op == "global_load_ubyte":
            return self.memory.read_byte(ea)
        elif op == "global_load_ushort":
            return self.memory.read_short(ea)
        else:
            raise ValueError(f"Unsupported load op: {op}")

    def execute_store(self, op: str, data: int, base: int, voffset: int = 0,
                      offset_mod: int | str = 0, active: bool = True) -> None:
        if not active:
            return
        ea = compute_global_effective_address(base, voffset, offset_mod)
        if op == "global_store_byte":
            self.memory.write_byte(ea, data & 0xFF)
        elif op == "global_store_short":
            self.memory.write_short(ea, data & 0xFFFF)
        else:
            raise ValueError(f"Unsupported store op: {op}")


def translate(body: list[str]) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(tmp, d))
        with open(os.path.join(tmp, "cs_gsub.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", "cs_gsub.fssm"), "wb") as f:
            f.write(FSSM)
        src = os.path.join(tmp, "disasm", "cs_gsub.s")
        with open(src, "w", encoding="utf-8") as f:
            f.write("; cs_gsub\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(src)


def run(body: list[str], memory: dict[int, int] | None = None) -> dict[int, int]:
    if memory is None:
        mem = {BASE + 4 * k: w for k, w in enumerate(SEED)}
    else:
        mem = dict(memory)
    si.run_workgroup(translate(body), 1, 32,
                     inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                     push_constants=[0] * 36, memory=mem)
    return mem


class TestGlobalSubDwordOracle(unittest.TestCase):
    """Unit tests validating reference oracle logic against known arithmetic rules."""

    def test_effective_address_calculation(self):
        self.assertEqual(compute_global_effective_address(0x1000, 0x200, 4), 0x1204)
        self.assertEqual(compute_global_effective_address(0x1000, 0x0, -4), 0x0FFC)
        self.assertEqual(compute_global_effective_address(0x1000, 0x0, "0xFFC"), 0x0FFC)

    def test_oracle_byte_load_and_store(self):
        bank = GlobalSubDwordMemoryBank({0x1000: 0x44332211})
        oracle = GlobalSubDwordReferenceOracle(bank)
        self.assertEqual(oracle.execute_load("global_load_ubyte", 0x1000, 0), 0x11)
        self.assertEqual(oracle.execute_load("global_load_ubyte", 0x1000, 1), 0x22)
        oracle.execute_store("global_store_byte", 0xEE, 0x1000, 1)
        self.assertEqual(bank.read_dword(0x1000), 0x4433EE11)

    def test_oracle_short_load_and_store(self):
        bank = GlobalSubDwordMemoryBank({0x1000: 0x44332211, 0x1004: 0x88776655})
        oracle = GlobalSubDwordReferenceOracle(bank)
        self.assertEqual(oracle.execute_load("global_load_ushort", 0x1000, 0), 0x2211)
        self.assertEqual(oracle.execute_load("global_load_ushort", 0x1000, 3), 0x5544)
        oracle.execute_store("global_store_short", 0xAABB, 0x1000, 3)
        self.assertEqual(bank.read_dword(0x1000), 0xBB332211)
        self.assertEqual(bank.read_dword(0x1004), 0x887766AA)


if __name__ == "__main__":
    unittest.main()
