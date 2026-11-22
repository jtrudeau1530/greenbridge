"""Semantic interpreter tests for LDS atomic bitwise AND (ds_and_b32, ds_and_rtn_b32, ds_and_b64, ds_and_rtn_b64).

Standard library only; no GPU, no LLVM, no external tools, no game files.

Run directly:
    python -I tests/semantic/test_lds_atomic_and.py
Or via unittest discovery:
    python -m unittest discover -s tests/semantic -p "test_lds_atomic_and.py" -v

Background & Specification Rules:
  * AMD RDNA2 / GFX10 Local Data Share (LDS) Architecture:
    - LDS is an on-chip, software-managed, high-throughput scratchpad memory shared across
      all work-items (lanes/threads) within a workgroup.
    - Single 32-bit Atomic AND:
        - `ds_and_b32 vaddr, vdata [offset:N]`:
            Non-returning form:
              eff_byte = (vaddr + offset) & 0xFFFFFFFF
              dword_idx = (eff_byte >> 2) & (lds_dwords - 1)
              LDS[dword_idx] = LDS[dword_idx] & (vdata & 0xFFFFFFFF)
            The prior value at LDS[dword_idx] is discarded.
        - `ds_and_rtn_b32 vdst, vaddr, vdata [offset:N]`:
            Returning form:
              eff_byte = (vaddr + offset) & 0xFFFFFFFF
              dword_idx = (eff_byte >> 2) & (lds_dwords - 1)
              old_val = LDS[dword_idx]
              LDS[dword_idx] = old_val & (vdata & 0xFFFFFFFF)
              vdst = old_val
            Preserves the pre-atomic memory contents in destination VGPR vdst.
    - 64-bit Atomic AND:
        - `ds_and_b64 vaddr, v[data_lo:data_hi] [offset:N]`:
            Non-returning form:
              eff_byte = (vaddr + offset) & 0xFFFFFFFF
              dword_idx = (eff_byte >> 2) & (lds_dwords - 1)
              LDS[dword_idx]   &= vdata_lo
              LDS[dword_idx+1] &= vdata_hi
        - `ds_and_rtn_b64 v[dst_lo:dst_hi], vaddr, v[data_lo:data_hi] [offset:N]`:
            Returning form: returns pre-atomic 64-bit value in VGPR pair v[dst_lo:dst_hi].
    - Active lane masking:
        - LDS instructions are vector operations governed by the execution mask (EXEC).
        - Inactive lanes do not mutate LDS storage and leave destination VGPRs unmodified.
    - Hardware wrapping & alignment:
        - Byte addresses are shifted right by 2, dropping unaligned low bits [1:0].
        - The resulting dword index is masked with (lds_dwords - 1), wrapping out-of-bounds
          offsets within the allocated power-of-two LDS capacity.
    - Mathematical properties of Bitwise AND:
        - Idempotency: x & x == x.
        - Identity: x & 0xFFFFFFFF == x (all ones mask preserves value).
        - Annihilator / Null element: x & 0 == 0 (zero mask clears all bits).
        - Commutativity: x & y == y & x.
        - Associativity: (x & y) & z == x & (y & z).
        - Monotonic bit reduction: popcount(x & y) <= min(popcount(x), popcount(y)).
"""
from __future__ import annotations

import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rdna2spirv  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
DEFAULT_LDS_DWORDS = 16384  # 64 KiB default capacity in dwords


class UnsupportedConstruct(NotImplementedError):
    """The module uses an unmodelled SPIR-V opcode, type, or storage class."""


class WorkgroupBoundsError(ValueError):
    """Workgroup dimension or memory bounds violation."""


_IGNORED = frozenset({
    "OpCapability", "OpExtInstImport", "OpMemoryModel", "OpEntryPoint", "OpExecutionMode",
    "OpDecorate", "OpMemberDecorate", "OpTypeVoid", "OpTypeFunction", "OpLabel",
    "OpReturn", "OpFunctionEnd", "OpName", "OpSource", "OpSelectionMerge",
})

_LINE = re.compile(r"^(?:(%\S+)\s*=\s*)?(Op\w+)\s*(.*)$")


def reference_atomic_and_b32(current_val: int, and_mask: int) -> tuple[int, int]:
    """Reference 32-bit atomic AND: returns (old_val, updated_mem_val)."""
    old = current_val & MASK32
    new = (old & and_mask) & MASK32
    return old, new


def reference_atomic_and_b64(current_val: int, and_mask: int) -> tuple[int, int]:
    """Reference 64-bit atomic AND: returns (old_val, updated_mem_val)."""
    old = current_val & MASK64
    new = (old & and_mask) & MASK64
    return old, new


def split_u64_to_u32(val: int) -> tuple[int, int]:
    """Splits a 64-bit unsigned integer into (low_u32, high_u32)."""
    val = val & MASK64
    return val & MASK32, (val >> 32) & MASK32


def combine_u32_to_u64(lo: int, hi: int) -> int:
    """Combines two 32-bit unsigned integers (low_u32, high_u32) into a 64-bit unsigned integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def calc_lds_byte_offset(
    vaddr: int,
    byte_offset: int = 0,
    lds_dwords: int = DEFAULT_LDS_DWORDS,
) -> tuple[int, int, int]:
    """Reference ISA calculation for single LDS address: (effective_byte, dword_idx, byte_in_dword)."""
    vaddr = vaddr & MASK32
    byte_offset = byte_offset & 0xFFFF  # 16-bit unsigned offset in RDNA2 DS instructions
    eff_byte = (vaddr + byte_offset) & MASK32
    dword_idx = (eff_byte >> 2) & (lds_dwords - 1)
    byte_in_dword = eff_byte & 3
    return eff_byte, dword_idx, byte_in_dword


def calc_lds_dual_offsets(
    vaddr: int,
    offset0: int = 0,
    offset1: int = 0,
    lds_dwords: int = DEFAULT_LDS_DWORDS,
    st64: bool = False,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Reference ISA calculation for dual LDS addresses.

    Returns:
        ((eff_byte0, dword_idx0), (eff_byte1, dword_idx1))
    """
    vaddr = vaddr & MASK32
    unit = 256 if st64 else 4
    off0_bytes = (offset0 & 0xFF) * unit
    off1_bytes = (offset1 & 0xFF) * unit
    eff_byte0 = (vaddr + off0_bytes) & MASK32
    eff_byte1 = (vaddr + off1_bytes) & MASK32
    dword_idx0 = (eff_byte0 >> 2) & (lds_dwords - 1)
    dword_idx1 = (eff_byte1 >> 2) & (lds_dwords - 1)
    return (eff_byte0, dword_idx0), (eff_byte1, dword_idx1)


class LdsSemanticInterpreter:
    """Deterministic multi-lane SIMD semantic interpreter for RDNA2 LDS atomic instructions.

    Models LDS shared memory, per-lane VGPR register files, and execution masks.
    """

    def __init__(self, lds_dwords: int = DEFAULT_LDS_DWORDS, num_vgprs: int = 256, num_lanes: int = 64):
        self.lds_dwords = lds_dwords
        self.lds = [0] * lds_dwords
        self.num_vgprs = num_vgprs
        self.num_lanes = num_lanes
        self.vgprs = [[0] * num_vgprs for _ in range(num_lanes)]
        self.active_lanes = [True] * num_lanes

    def set_vgpr(self, reg_idx: int, value: int, lane: int = 0) -> None:
        self.vgprs[lane][reg_idx] = value & MASK32

    def get_vgpr(self, reg_idx: int, lane: int = 0) -> int:
        return self.vgprs[lane][reg_idx]

    def set_lds_dword(self, dword_idx: int, value: int) -> None:
        self.lds[dword_idx & (self.lds_dwords - 1)] = value & MASK32

    def get_lds_dword(self, dword_idx: int) -> int:
        return self.lds[dword_idx & (self.lds_dwords - 1)]

    def set_lane_active(self, lane: int, active: bool) -> None:
        self.active_lanes[lane] = bool(active)

    def set_exec_mask(self, mask: int) -> None:
        """Sets active lanes based on a 64-bit integer execution mask."""
        for lane in range(self.num_lanes):
            self.active_lanes[lane] = bool((mask >> lane) & 1)

    def ds_write_b32(self, vaddr_reg: int, vdata_reg: int, offset: int = 0, lane: int = 0) -> int | None:
        if not self.active_lanes[lane]:
            return None
        vaddr = self.vgprs[lane][vaddr_reg]
        vdata = self.vgprs[lane][vdata_reg]
        _, dword_idx, _ = calc_lds_byte_offset(vaddr, offset, self.lds_dwords)
        self.lds[dword_idx] = vdata & MASK32
        return dword_idx

    def ds_read_b32(self, vdst_reg: int, vaddr_reg: int, offset: int = 0, lane: int = 0) -> int | None:
        if not self.active_lanes[lane]:
            return None
        vaddr = self.vgprs[lane][vaddr_reg]
        _, dword_idx, _ = calc_lds_byte_offset(vaddr, offset, self.lds_dwords)
        self.vgprs[lane][vdst_reg] = self.lds[dword_idx]
        return dword_idx

    def ds_and_b32(self, vaddr_reg: int, vdata_reg: int, offset: int = 0, lane: int = 0) -> int | None:
        """Non-returning 32-bit atomic AND: updates LDS[target], returns dword_idx or None."""
        if not self.active_lanes[lane]:
            return None
        vaddr = self.vgprs[lane][vaddr_reg]
        vdata = self.vgprs[lane][vdata_reg]
        _, dword_idx, _ = calc_lds_byte_offset(vaddr, offset, self.lds_dwords)
        self.lds[dword_idx] = (self.lds[dword_idx] & vdata) & MASK32
        return dword_idx

    def ds_and_rtn_b32(
        self,
        vdst_reg: int,
        vaddr_reg: int,
        vdata_reg: int,
        offset: int = 0,
        lane: int = 0,
    ) -> tuple[int, int] | None:
        """Returning 32-bit atomic AND: stores prior value into vdst and updates LDS[target].

        Returns (dword_idx, old_val) if lane active, or None if inactive.
        Handles destination register aliasing: reads vaddr and vdata before overwriting vdst.
        """
        if not self.active_lanes[lane]:
            return None
        vaddr = self.vgprs[lane][vaddr_reg]
        vdata = self.vgprs[lane][vdata_reg]
        _, dword_idx, _ = calc_lds_byte_offset(vaddr, offset, self.lds_dwords)
        old_val = self.lds[dword_idx]
        self.lds[dword_idx] = (old_val & vdata) & MASK32
        self.vgprs[lane][vdst_reg] = old_val
        return dword_idx, old_val

    def set_vgpr_u64(self, reg_idx: int, value: int, lane: int = 0) -> None:
        lo, hi = split_u64_to_u32(value)
        self.set_vgpr(reg_idx, lo, lane=lane)
        self.set_vgpr(reg_idx + 1, hi, lane=lane)

    def get_vgpr_u64(self, reg_idx: int, lane: int = 0) -> int:
        lo = self.get_vgpr(reg_idx, lane=lane)
        hi = self.get_vgpr(reg_idx + 1, lane=lane)
        return combine_u32_to_u64(lo, hi)

    def set_lds_u64(self, dword_idx: int, value: int) -> None:
        lo, hi = split_u64_to_u32(value)
        self.set_lds_dword(dword_idx, lo)
        self.set_lds_dword((dword_idx + 1) & (self.lds_dwords - 1), hi)

    def get_lds_u64(self, dword_idx: int) -> int:
        lo = self.get_lds_dword(dword_idx)
        hi = self.get_lds_dword((dword_idx + 1) & (self.lds_dwords - 1))
        return combine_u32_to_u64(lo, hi)

    def ds_write_b64(self, vaddr_reg: int, vdata_lo_reg: int, offset: int = 0, lane: int = 0) -> tuple[int, int] | None:
        if not self.active_lanes[lane]:
            return None
        vaddr = self.vgprs[lane][vaddr_reg]
        _, dword_idx0, _ = calc_lds_byte_offset(vaddr, offset, self.lds_dwords)
        dword_idx1 = (dword_idx0 + 1) & (self.lds_dwords - 1)
        self.lds[dword_idx0] = self.vgprs[lane][vdata_lo_reg] & MASK32
        self.lds[dword_idx1] = self.vgprs[lane][vdata_lo_reg + 1] & MASK32
        return dword_idx0, dword_idx1

    def ds_read_b64(self, vdst_lo_reg: int, vaddr_reg: int, offset: int = 0, lane: int = 0) -> tuple[int, int] | None:
        if not self.active_lanes[lane]:
            return None
        vaddr = self.vgprs[lane][vaddr_reg]
        _, dword_idx0, _ = calc_lds_byte_offset(vaddr, offset, self.lds_dwords)
        dword_idx1 = (dword_idx0 + 1) & (self.lds_dwords - 1)
        val_lo = self.lds[dword_idx0]
        val_hi = self.lds[dword_idx1]
        self.vgprs[lane][vdst_lo_reg] = val_lo
        self.vgprs[lane][vdst_lo_reg + 1] = val_hi
        return dword_idx0, dword_idx1

    def ds_and_b64(self, vaddr_reg: int, vdata_lo_reg: int, offset: int = 0, lane: int = 0) -> tuple[int, int] | None:
        """Non-returning 64-bit atomic AND: updates LDS[idx] and LDS[idx+1], returns (idx0, idx1) or None."""
        if not self.active_lanes[lane]:
            return None
        vaddr = self.vgprs[lane][vaddr_reg]
        vdata_lo = self.vgprs[lane][vdata_lo_reg]
        vdata_hi = self.vgprs[lane][vdata_lo_reg + 1]
        _, dword_idx0, _ = calc_lds_byte_offset(vaddr, offset, self.lds_dwords)
        dword_idx1 = (dword_idx0 + 1) & (self.lds_dwords - 1)
        self.lds[dword_idx0] = (self.lds[dword_idx0] & vdata_lo) & MASK32
        self.lds[dword_idx1] = (self.lds[dword_idx1] & vdata_hi) & MASK32
        return dword_idx0, dword_idx1

    def ds_and_rtn_b64(
        self,
        vdst_lo_reg: int,
        vaddr_reg: int,
        vdata_lo_reg: int,
        offset: int = 0,
        lane: int = 0,
    ) -> tuple[int, int] | None:
        """Returning 64-bit atomic AND: stores prior 64-bit value into vdst[0:1] and updates LDS[idx:idx+1].

        Returns (dword_idx0, old_val_u64) if lane active, or None if inactive.
        Reads all sources before writing vdst to handle aliasing safely.
        """
        if not self.active_lanes[lane]:
            return None
        vaddr = self.vgprs[lane][vaddr_reg]
        vdata_lo = self.vgprs[lane][vdata_lo_reg]
        vdata_hi = self.vgprs[lane][vdata_lo_reg + 1]
        _, dword_idx0, _ = calc_lds_byte_offset(vaddr, offset, self.lds_dwords)
        dword_idx1 = (dword_idx0 + 1) & (self.lds_dwords - 1)
        old_lo = self.lds[dword_idx0]
        old_hi = self.lds[dword_idx1]
        self.lds[dword_idx0] = (old_lo & vdata_lo) & MASK32
        self.lds[dword_idx1] = (old_hi & vdata_hi) & MASK32
        self.vgprs[lane][vdst_lo_reg] = old_lo
        self.vgprs[lane][vdst_lo_reg + 1] = old_hi
        return dword_idx0, combine_u32_to_u64(old_lo, old_hi)

    def parse_and_execute(self, instruction_text: str, lane: int = 0) -> int | tuple[int, int] | None:
        line = instruction_text.strip()
        if not line or line.startswith(";"):
            return None

        offset = 0
        m_off = re.search(r"\boffset:(\S+)", line)
        if m_off:
            offset = int(m_off.group(1), 0)
            line = re.sub(r"\boffset:\S+", "", line).strip()

        parts = line.split(None, 1)
        op = parts[0]
        args_str = parts[1] if len(parts) > 1 else ""
        raw_args = [a.strip() for a in args_str.split(",") if a.strip()]

        def reg_num(tok: str) -> int:
            cleaned = tok.replace("v", "").replace("[", "").replace("]", "")
            if ":" in cleaned:
                cleaned = cleaned.split(":")[0]
            return int(cleaned)

        if op == "ds_and_b32":
            vaddr = reg_num(raw_args[0])
            vdata = reg_num(raw_args[1])
            return self.ds_and_b32(vaddr, vdata, offset=offset, lane=lane)

        elif op == "ds_and_rtn_b32":
            vdst = reg_num(raw_args[0])
            vaddr = reg_num(raw_args[1])
            vdata = reg_num(raw_args[2])
            res = self.ds_and_rtn_b32(vdst, vaddr, vdata, offset=offset, lane=lane)
            return res[0] if res else None

        elif op == "ds_write_b32":
            vaddr = reg_num(raw_args[0])
            vdata = reg_num(raw_args[1])
            return self.ds_write_b32(vaddr, vdata, offset=offset, lane=lane)

        elif op == "ds_read_b32":
            vdst = reg_num(raw_args[0])
            vaddr = reg_num(raw_args[1])
            return self.ds_read_b32(vdst, vaddr, offset=offset, lane=lane)

        elif op == "ds_and_b64":
            vaddr = reg_num(raw_args[0])
            vdata = reg_num(raw_args[1])
            res_b64 = self.ds_and_b64(vaddr, vdata, offset=offset, lane=lane)
            return res_b64[0] if res_b64 else None

        elif op == "ds_and_rtn_b64":
            vdst = reg_num(raw_args[0])
            vaddr = reg_num(raw_args[1])
            vdata = reg_num(raw_args[2])
            res_rtn_b64 = self.ds_and_rtn_b64(vdst, vaddr, vdata, offset=offset, lane=lane)
            return res_rtn_b64[0] if res_rtn_b64 else None

        elif op == "ds_write_b64":
            vaddr = reg_num(raw_args[0])
            vdata = reg_num(raw_args[1])
            res_w64 = self.ds_write_b64(vaddr, vdata, offset=offset, lane=lane)
            return res_w64[0] if res_w64 else None

        elif op == "ds_read_b64":
            vdst = reg_num(raw_args[0])
            vaddr = reg_num(raw_args[1])
            res_r64 = self.ds_read_b64(vdst, vaddr, offset=offset, lane=lane)
            return res_r64[0] if res_r64 else None

        else:
            raise UnsupportedConstruct(f"Opcode {op} not supported in interpreter")

    def run_simd_wave(self, instruction_text: str) -> list[int | tuple[int, int] | None]:
        """Runs the instruction across all lanes in the wavefront in lockstep."""
        return [self.parse_and_execute(instruction_text, lane=lane) for lane in range(self.num_lanes)]


class LdsAtomicSpirvInterp:
    """Strict semantic interpreter for compute SPIR-V modules exercising LDS atomic AND."""

    def __init__(self, spvasm: str):
        self.spvasm = spvasm
        self.types: dict[str, tuple] = {}
        self.constants: dict[str, int | bool] = {}
        self.ptr_types: dict[str, tuple[str, str]] = {}
        self.var_class: dict[str, str] = {}
        self.local_size: tuple[int, int, int] = (1, 1, 1)
        self.lds_var: str | None = None
        self.lds_size: int = 0
        self.body: list[tuple[str | None, str, list[str], str]] = []
        self._parse()

    def _parse(self) -> None:
        in_body = False
        for raw in self.spvasm.splitlines():
            line = raw.strip()
            if not line:
                continue
            m = _LINE.match(line)
            if not m:
                raise UnsupportedConstruct(f"Cannot parse line: {line!r}")
            res, op, rest = m.groups()
            a = rest.split()
            if in_body:
                self.body.append((res, op, a, line))
                continue
            if op == "OpFunction":
                in_body = True
                continue
            if op == "OpExecutionMode" and len(a) >= 5 and a[1] == "LocalSize":
                lx, ly, lz = int(a[2]), int(a[3]), int(a[4])
                if lx < 1 or ly < 1 or lz < 1 or (lx * ly * lz) > 1024:
                    raise WorkgroupBoundsError(f"Invalid LocalSize {lx}x{ly}x{lz}")
                self.local_size = (lx, ly, lz)
                continue
            if op in _IGNORED:
                continue
            if op == "OpTypeInt":
                if int(a[0]) != 32:
                    raise UnsupportedConstruct("Only 32-bit int supported in interpreter")
                self.types[res] = ("int", int(a[0]), int(a[1]))
            elif op == "OpTypeBool":
                self.types[res] = ("bool",)
            elif op == "OpTypeVector":
                self.types[res] = ("vec", a[0], int(a[1]))
            elif op == "OpTypeArray":
                size = int(self.constants[a[1]])
                self.types[res] = ("arr", a[0], size)
            elif op == "OpTypePointer":
                self.ptr_types[res] = (a[0], a[1])
            elif op == "OpConstant":
                self.constants[res] = int(a[1], 0) & MASK32
            elif op == "OpConstantTrue":
                self.constants[res] = True
            elif op == "OpConstantFalse":
                self.constants[res] = False
            elif op == "OpVariable":
                cls, pointee = self.ptr_types[a[0]]
                if a[1] != cls:
                    raise UnsupportedConstruct("OpVariable storage class mismatch")
                self.var_class[res] = cls
                if cls == "Workgroup":
                    if self.lds_var is not None:
                        raise UnsupportedConstruct("Multiple Workgroup variables")
                    self.lds_var = res
                    arr_ty = self.types[pointee]
                    if arr_ty[0] != "arr":
                        raise UnsupportedConstruct("Workgroup variable must be an array")
                    self.lds_size = arr_ty[2]
                elif cls not in ("Output", "Function"):
                    raise UnsupportedConstruct(f"Unsupported storage class {cls}")
            else:
                raise UnsupportedConstruct(f"Header opcode {op}")

    def run(
        self,
        num_invocations: int | None = None,
        seed_lds: dict[int, int] | None = None,
        lane_inputs: list[dict[str, int]] | None = None,
    ) -> tuple[list[int], list[dict[str, int]]]:
        """Executes the workgroup and returns (final_lds, outputs_per_lane)."""
        lx, ly, lz = self.local_size
        total_invocations = lx * ly * lz
        count = total_invocations if num_invocations is None else min(num_invocations, total_invocations)

        lds = [0] * self.lds_size
        if seed_lds:
            for idx, val in seed_lds.items():
                if 0 <= idx < self.lds_size:
                    lds[idx] = val & MASK32

        lane_outputs: list[dict[str, int]] = []
        for lidx in range(count):
            inputs = lane_inputs[lidx] if lane_inputs and lidx < len(lane_inputs) else {}
            out = self._run_lane(lidx, lds, inputs)
            lane_outputs.append(out)

        return list(lds), lane_outputs

    def _run_lane(self, lidx: int, lds: list[int], inputs: dict[str, int]) -> dict[str, int]:
        values: dict[str, int | bool] = dict(self.constants)
        values.update(inputs)
        var_box: dict[str, list[int]] = {}
        pointers: dict[str, tuple[str, list[int], int | None]] = {}
        outputs: list[str] = []

        if self.lds_var:
            pointers[self.lds_var] = ("lds", lds, None)

        for res, cls in self.var_class.items():
            if cls == "Output":
                var_box[res] = [0]
                pointers[res] = ("out", var_box[res], 0)
                outputs.append(res)
            elif cls == "Function":
                var_box[res] = [0]
                pointers[res] = ("fn", var_box[res], 0)

        def val(x: str) -> int:
            if x in values:
                return values[x]  # type: ignore
            raise UnsupportedConstruct(f"Undefined identifier {x}")

        for res, op, a, _ in self.body:
            if op in _IGNORED or op in ("OpBranch", "OpBranchConditional"):
                continue
            if op == "OpVariable":
                cls, _ = self.ptr_types[a[0]]
                if cls != "Function":
                    raise UnsupportedConstruct("Non-Function variable in body")
                var_box[res] = [val(a[2]) if len(a) > 2 else 0]
                pointers[res] = ("fn", var_box[res], 0)
            elif op == "OpAccessChain":
                base = a[1]
                if base not in pointers:
                    raise UnsupportedConstruct(f"OpAccessChain on non-pointer {base}")
                ptag = pointers[base][0]
                if ptag == "lds":
                    idx = val(a[2])
                    if not (0 <= idx < len(lds)):
                        raise WorkgroupBoundsError(f"LDS access out of bounds: {idx} >= {len(lds)}")
                    pointers[res] = ("lds", lds, idx)
                elif ptag in ("out", "fn"):
                    pointers[res] = pointers[base]
                else:
                    raise UnsupportedConstruct(f"Access chain on {ptag}")
            elif op == "OpLoad":
                ptr = pointers[a[1]]
                ptag = ptr[0]
                if ptag in ("lds", "out", "fn"):
                    idx_val = ptr[2]
                    values[res] = ptr[1][idx_val] & MASK32
            elif op == "OpStore":
                ptr = pointers[a[0]]
                v = val(a[1]) & MASK32
                ptag = ptr[0]
                if ptag in ("lds", "out", "fn"):
                    idx_val = ptr[2]
                    ptr[1][idx_val] = v
            elif op == "OpAtomicAnd":
                # %res = OpAtomicAnd %type %ptr %scope %semantics %val
                ptr = pointers[a[1]]
                ptag = ptr[0]
                if ptag != "lds":
                    raise UnsupportedConstruct(f"OpAtomicAnd on non-LDS pointer {ptag}")
                mask_val = val(a[4]) & MASK32
                idx_val = ptr[2]
                old = ptr[1][idx_val] & MASK32
                ptr[1][idx_val] = (old & mask_val) & MASK32
                values[res] = old
            elif op == "OpIAdd":
                values[res] = (val(a[1]) + val(a[2])) & MASK32
            elif op == "OpISub":
                values[res] = (val(a[1]) - val(a[2])) & MASK32
            elif op == "OpShiftRightLogical":
                sh = val(a[2])
                if sh >= 32:
                    raise UnsupportedConstruct("Shift >= 32")
                values[res] = (val(a[1]) >> sh) & MASK32
            elif op == "OpShiftLeftLogical":
                sh = val(a[2])
                if sh >= 32:
                    raise UnsupportedConstruct("Shift >= 32")
                values[res] = (val(a[1]) << sh) & MASK32
            elif op == "OpBitwiseAnd":
                values[res] = val(a[1]) & val(a[2]) & MASK32
            elif op == "OpBitwiseOr":
                values[res] = (val(a[1]) | val(a[2])) & MASK32
            elif op == "OpBitwiseXor":
                values[res] = (val(a[1]) ^ val(a[2])) & MASK32
            elif op == "OpSelect":
                cond = val(a[1])
                values[res] = val(a[2]) if cond else val(a[3])
            elif op == "OpCopyObject":
                values[res] = val(a[1])
            elif op == "OpBitcast":
                values[res] = val(a[1]) & MASK32
            else:
                raise UnsupportedConstruct(f"Opcode {op} not modelled in body")

        return {name: var_box[name][0] & MASK32 for name in outputs}


class TestAtomicAndReferenceMath(unittest.TestCase):
    """Verifies 32-bit and 64-bit atomic bitwise AND math, truth tables, and algebraic properties."""

    def test_and_truth_tables_b32(self):
        all_zeros = 0x00000000
        all_ones = 0xFFFFFFFF
        alt_a = 0x55555555
        alt_b = 0xAAAAAAAA

        old, mem = reference_atomic_and_b32(all_zeros, all_zeros)
        self.assertEqual(old, all_zeros)
        self.assertEqual(mem, all_zeros)

        old, mem = reference_atomic_and_b32(all_zeros, all_ones)
        self.assertEqual(old, all_zeros)
        self.assertEqual(mem, all_zeros)

        old, mem = reference_atomic_and_b32(all_ones, all_zeros)
        self.assertEqual(old, all_ones)
        self.assertEqual(mem, all_zeros)

        old, mem = reference_atomic_and_b32(all_ones, all_ones)
        self.assertEqual(old, all_ones)
        self.assertEqual(mem, all_ones)

        old, mem = reference_atomic_and_b32(alt_a, alt_b)
        self.assertEqual(old, alt_a)
        self.assertEqual(mem, all_zeros)

        old, mem = reference_atomic_and_b32(alt_a, alt_a)
        self.assertEqual(old, alt_a)
        self.assertEqual(mem, alt_a)

    def test_identity_and_annihilator_properties(self):
        test_values = [
            0x00000000,
            0x00000001,
            0x12345678,
            0x7FFFFFFF,
            0x80000000,
            0xFFFFFFFF,
            0xDEADBEEF,
            0xCAFEBABE,
        ]
        for val in test_values:
            with self.subTest(val=hex(val)):
                old, mem = reference_atomic_and_b32(val, MASK32)
                self.assertEqual(old, val)
                self.assertEqual(mem, val)

                old, mem = reference_atomic_and_b32(val, 0)
                self.assertEqual(old, val)
                self.assertEqual(mem, 0)

    def test_idempotency_b32(self):
        test_values = [
            0x00000000,
            0x00000001,
            0x55555555,
            0xAAAAAAAA,
            0x12345678,
            0x87654321,
            0xFFFFFFFF,
        ]
        for val in test_values:
            with self.subTest(val=hex(val)):
                old, mem = reference_atomic_and_b32(val, val)
                self.assertEqual(old, val)
                self.assertEqual(mem, val)

    def test_commutativity_and_associativity_b32(self):
        a, b = 0x3F2A1C8E, 0x9D4E7B01
        _, ab = reference_atomic_and_b32(a, b)
        _, ba = reference_atomic_and_b32(b, a)
        self.assertEqual(ab, ba)

        c = 0x55AA33CC
        _, step_ab = reference_atomic_and_b32(a, b)
        _, ab_c = reference_atomic_and_b32(step_ab, c)

        _, step_bc = reference_atomic_and_b32(b, c)
        _, a_bc = reference_atomic_and_b32(a, step_bc)
        self.assertEqual(ab_c, a_bc)

    def test_bitfield_clearing_and_isolation(self):
        initial = 0xFFFFFFFF
        for bit in range(32):
            mask = ~(1 << bit) & MASK32
            old, mem = reference_atomic_and_b32(initial, mask)
            self.assertEqual(old, initial)
            self.assertEqual(mem, initial & mask)
            self.assertEqual(mem & (1 << bit), 0)
            self.assertEqual(bin(mem).count("1"), 31)

        val = 0x12345678
        low_byte_mask = 0x000000FF
        _, low_byte = reference_atomic_and_b32(val, low_byte_mask)
        self.assertEqual(low_byte, 0x00000078)

        high_byte_mask = 0xFF000000
        _, high_byte = reference_atomic_and_b32(val, high_byte_mask)
        self.assertEqual(high_byte, 0x12000000)

    def test_monotonic_bit_reduction(self):
        val = 0xFFFFFFFF
        filters = [0xF0F0F0F0, 0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF]
        current = val
        pop = bin(current).count("1")
        self.assertEqual(pop, 32)
        for flt in filters:
            old, current = reference_atomic_and_b32(current, flt)
            self.assertEqual(old & flt, current)
            new_pop = bin(current).count("1")
            self.assertLessEqual(new_pop, pop)
            pop = new_pop

    def test_de_morgans_laws_with_inversion(self):
        cases = [
            (0x12345678, 0x9ABCDEF0),
            (0x55555555, 0xAAAAAAAA),
            (0x0000FFFF, 0xFFFF0000),
            (0xFFFFFFFF, 0x00000000),
        ]
        for a, b in cases:
            with self.subTest(a=hex(a), b=hex(b)):
                _, and_ab = reference_atomic_and_b32(a, b)
                not_and_ab = (~and_ab) & MASK32
                not_a = (~a) & MASK32
                not_b = (~b) & MASK32
                or_not_ab = (not_a | not_b) & MASK32
                self.assertEqual(not_and_ab, or_not_ab)

    def test_reference_atomic_and_b64_properties(self):
        init = 0x0123456789ABCDEF
        mask = 0xFEDCBA9876543210
        old, mem = reference_atomic_and_b64(init, mask)
        self.assertEqual(old, init)
        self.assertEqual(mem, init & mask)

        old, mem = reference_atomic_and_b64(init, MASK64)
        self.assertEqual(old, init)
        self.assertEqual(mem, init)

        old, mem = reference_atomic_and_b64(init, 0)
        self.assertEqual(old, init)
        self.assertEqual(mem, 0)

        cross1 = 0xFFFFFFFF00000000
        cross2 = 0x00000000FFFFFFFF
        _, mem_cross = reference_atomic_and_b64(cross1, cross2)
        self.assertEqual(mem_cross, 0)

    def test_split_and_combine_u64_u32(self):
        v64 = 0xFEDCBA9876543210
        lo, hi = split_u64_to_u32(v64)
        self.assertEqual(lo, 0x76543210)
        self.assertEqual(hi, 0xFEDCBA98)
        self.assertEqual(combine_u32_to_u64(lo, hi), v64)


class TestLdsAddressingAndOffsets(unittest.TestCase):
    """Verifies single and dual LDS byte offsets, dword truncation, st64 strides, and wrapping."""

    def test_aligned_dword_addressing(self):
        lds_dwords = 256
        for dword in (0, 1, 2, 8, 16, 64, 127, 255):
            byte_addr = dword * 4
            eff, dword_idx, byte_pos = calc_lds_byte_offset(byte_addr, 0, lds_dwords)
            self.assertEqual(eff, byte_addr)
            self.assertEqual(dword_idx, dword)
            self.assertEqual(byte_pos, 0)

    def test_unaligned_byte_truncation(self):
        lds_dwords = 128
        base_aligned = 64  # dword 16
        for b in (0, 1, 2, 3):
            eff, dword_idx, byte_pos = calc_lds_byte_offset(base_aligned, b, lds_dwords)
            self.assertEqual(eff, base_aligned + b)
            self.assertEqual(dword_idx, 16)
            self.assertEqual(byte_pos, b)

    def test_offset_16bit_masking(self):
        lds_dwords = 512
        # RDNA2 DS instruction offset field is 16-bit unsigned (0x0000 - 0xFFFF)
        eff, dword_idx, byte_pos = calc_lds_byte_offset(0, 0x10008, lds_dwords)
        self.assertEqual(eff, 8)
        self.assertEqual(dword_idx, 2)
        self.assertEqual(byte_pos, 0)

    def test_dual_offset_unit_strides(self):
        (eff0, idx0), (eff1, idx1) = calc_lds_dual_offsets(0, 3, 10, lds_dwords=1024, st64=False)
        self.assertEqual(eff0, 12)
        self.assertEqual(idx0, 3)
        self.assertEqual(eff1, 40)
        self.assertEqual(idx1, 10)

    def test_dual_offset_st64_stride(self):
        (eff0, idx0), (eff1, idx1) = calc_lds_dual_offsets(0, 2, 5, lds_dwords=1024, st64=True)
        self.assertEqual(eff0, 512)
        self.assertEqual(idx0, 128)
        self.assertEqual(eff1, 1280)
        self.assertEqual(idx1, 320)

    def test_boundary_power_of_two_wrapping(self):
        lds_dwords = 64  # 256 bytes total
        # Byte address 256 wraps to dword 0
        _, idx0, _ = calc_lds_byte_offset(256, 0, lds_dwords)
        self.assertEqual(idx0, 0)
        # Byte address 260 wraps to dword 1
        _, idx1, _ = calc_lds_byte_offset(256, 4, lds_dwords)
        self.assertEqual(idx1, 1)
        # 32-bit unsigned overflow: 0xFFFFFFFC + 4 = 0x00000000 -> dword 0
        _, idx_wrap, _ = calc_lds_byte_offset(0xFFFFFFFC, 4, lds_dwords)
        self.assertEqual(idx_wrap, 0)


class TestMultiLaneSimdDivergence(unittest.TestCase):
    """Verifies SIMD wavefront divergence, execution mask filtering, and independent lane addresses."""

    def test_simd_active_mask_filtering(self):
        simd = LdsSemanticInterpreter(lds_dwords=256, num_lanes=64)
        simd.set_lds_dword(0, 0xFFFFFFFF)
        # Lower 32 lanes active, upper 32 lanes inactive
        simd.set_exec_mask(0x00000000FFFFFFFF)

        for lane in range(64):
            simd.set_vgpr(0, 0, lane=lane)           # vaddr = 0
            simd.set_vgpr(1, 0x55555555, lane=lane)  # vdata = 0x55555555
            simd.set_vgpr(2, 0xDEADBEEF, lane=lane)  # vdst initial sentinel

        results = simd.run_simd_wave("ds_and_rtn_b32 v2, v0, v1")
        # Active lanes 0..31: executed and returned dword 0
        for lane in range(32):
            self.assertEqual(results[lane], 0)
            self.assertIn(simd.get_vgpr(2, lane=lane), (0xFFFFFFFF, 0x55555555))

        # Inactive lanes 32..63: did not execute, destination untouched
        for lane in range(32, 64):
            self.assertIsNone(results[lane])
            self.assertEqual(simd.get_vgpr(2, lane=lane), 0xDEADBEEF)

        # Final memory in LDS[0] must reflect active lane AND
        self.assertEqual(simd.get_lds_dword(0), 0x55555555)

    def test_even_odd_divergent_lanes(self):
        simd = LdsSemanticInterpreter(lds_dwords=256, num_lanes=64)
        simd.set_lds_dword(4, 0xFFFFFFFF)

        for lane in range(64):
            simd.set_vgpr(0, 16, lane=lane)  # byte address 16 -> dword 4
            simd.set_vgpr(1, 0xFF00FF00 if (lane % 2 == 0) else 0x00FF00FF, lane=lane)

        # Path 1: even lanes active
        simd.set_exec_mask(0x5555555555555555)
        simd.run_simd_wave("ds_and_b32 v0, v1")
        self.assertEqual(simd.get_lds_dword(4), 0xFF00FF00)

        # Path 2: odd lanes active
        simd.set_exec_mask(0xAAAAAAAAAAAAAAAA)
        simd.run_simd_wave("ds_and_b32 v0, v1")
        # 0xFF00FF00 & 0x00FF00FF == 0x00000000
        self.assertEqual(simd.get_lds_dword(4), 0x00000000)

    def test_sparse_active_lanes(self):
        simd = LdsSemanticInterpreter(lds_dwords=256, num_lanes=64)
        simd.set_lds_dword(10, 0xFFFFFFFF)
        sparse_lanes = [3, 17, 42, 60]
        mask = 0
        for l in sparse_lanes:
            mask |= (1 << l)
        simd.set_exec_mask(mask)

        for lane in range(64):
            simd.set_vgpr(0, 40, lane=lane)  # byte 40 -> dword 10
            simd.set_vgpr(1, ~(1 << lane) & MASK32, lane=lane)
            simd.set_vgpr(5, 0x11223344, lane=lane)

        results = simd.run_simd_wave("ds_and_rtn_b32 v5, v0, v1")
        for lane in range(64):
            if lane in sparse_lanes:
                self.assertEqual(results[lane], 10)
                self.assertNotEqual(simd.get_vgpr(5, lane=lane), 0x11223344)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(simd.get_vgpr(5, lane=lane), 0x11223344)

        # The bits of the sparse lanes must be cleared in LDS[10]
        expected_mem = 0xFFFFFFFF
        for l in sparse_lanes:
            expected_mem &= ~(1 << l) & MASK32
        self.assertEqual(simd.get_lds_dword(10), expected_mem)

    def test_divergent_branch_paths_simd(self):
        simd = LdsSemanticInterpreter(lds_dwords=256, num_lanes=64)
        simd.set_lds_dword(8, 0xFFFFFFFF)

        for lane in range(64):
            simd.set_vgpr(0, 32, lane=lane)  # dword 8
            # Lower 32 lanes clear low 16 bits; upper 32 lanes clear high 16 bits
            val = 0xFFFF0000 if lane < 32 else 0x0000FFFF
            simd.set_vgpr(1, val, lane=lane)
            simd.set_vgpr(7, 0x99999999, lane=lane)

        # Then-branch (lower half active)
        simd.set_exec_mask(0x00000000FFFFFFFF)
        simd.run_simd_wave("ds_and_rtn_b32 v7, v0, v1")
        self.assertEqual(simd.get_lds_dword(8), 0xFFFF0000)

        # Else-branch (upper half active)
        simd.set_exec_mask(0xFFFFFFFF00000000)
        simd.run_simd_wave("ds_and_rtn_b32 v7, v0, v1")
        self.assertEqual(simd.get_lds_dword(8), 0x00000000)

    def test_multi_lane_independent_addresses(self):
        simd = LdsSemanticInterpreter(lds_dwords=256, num_lanes=64)
        simd.set_exec_mask(MASK64)

        for lane in range(64):
            simd.set_lds_dword(lane, 0xFFFFFFFF)
            simd.set_vgpr(0, lane * 4, lane=lane)
            simd.set_vgpr(1, ~(1 << (lane % 32)) & MASK32, lane=lane)
            simd.set_vgpr(3, 0, lane=lane)

        results = simd.run_simd_wave("ds_and_rtn_b32 v3, v0, v1")
        for lane in range(64):
            self.assertEqual(results[lane], lane)
            # Old value returned to lane was initial 0xFFFFFFFF
            self.assertEqual(simd.get_vgpr(3, lane=lane), 0xFFFFFFFF)
            # Memory has exactly bit (lane % 32) cleared
            expected = ~(1 << (lane % 32)) & MASK32
            self.assertEqual(simd.get_lds_dword(lane), expected)


class TestSharedMemoryAddressClash(unittest.TestCase):
    """Verifies atomic bitwise AND address clashing, bank contention, and cumulative reduction."""

    def test_all_lanes_clash_on_single_address(self):
        # All 32 lanes in Wave32 concurrently clash on LDS[5]
        simd = LdsSemanticInterpreter(lds_dwords=256, num_lanes=32)
        simd.set_lds_dword(5, 0xFFFFFFFF)
        simd.set_exec_mask(0xFFFFFFFF)

        for lane in range(32):
            simd.set_vgpr(0, 20, lane=lane)  # byte 20 -> dword 5
            simd.set_vgpr(1, ~(1 << lane) & MASK32, lane=lane)
            simd.set_vgpr(2, 0, lane=lane)

        simd.run_simd_wave("ds_and_rtn_b32 v2, v0, v1")

        # Every lane received an old value with its own bit still set
        for lane in range(32):
            ret_val = simd.get_vgpr(2, lane=lane)
            self.assertTrue(ret_val & (1 << lane), f"Lane {lane} missed its set bit in return value {ret_val:#x}")

        # Final memory has all 32 bits cleared
        self.assertEqual(simd.get_lds_dword(5), 0x00000000)

    def test_64_lane_address_clash_all_lanes(self):
        # 64 lanes in Wave64 clash on LDS[0]
        simd = LdsSemanticInterpreter(lds_dwords=256, num_lanes=64)
        simd.set_lds_dword(0, 0xFFFFFFFF)
        simd.set_exec_mask(MASK64)

        for lane in range(64):
            simd.set_vgpr(0, 0, lane=lane)
            # Lanes 0..31 clear single bits, lanes 32..63 clear all
            mask = (~(1 << lane) & MASK32) if lane < 32 else 0
            simd.set_vgpr(1, mask, lane=lane)

        simd.run_simd_wave("ds_and_b32 v0, v1")
        self.assertEqual(simd.get_lds_dword(0), 0)

    def test_bank_clash_interleaved_lanes(self):
        # 64 lanes interleaved across 4 slots: LDS[0], LDS[1], LDS[2], LDS[3]
        simd = LdsSemanticInterpreter(lds_dwords=256, num_lanes=64)
        for s in range(4):
            simd.set_lds_dword(s, 0xFFFFFFFF)
        simd.set_exec_mask(MASK64)

        for lane in range(64):
            slot = lane % 4
            simd.set_vgpr(0, slot * 4, lane=lane)
            bit = (lane // 4) % 32
            simd.set_vgpr(1, ~(1 << bit) & MASK32, lane=lane)

        simd.run_simd_wave("ds_and_b32 v0, v1")

        # In each slot, bits 0..15 were cleared by 16 clashing lanes
        for s in range(4):
            expected = 0xFFFF0000
            self.assertEqual(simd.get_lds_dword(s), expected)

    def test_cumulative_mask_reduction_order_independence(self):
        masks = [0xF0F0F0F0, 0xCCCCCCCC, 0xAAAAAAAA, 0x0F0F0F0F, 0x33333333, 0x55555555]
        simd = LdsSemanticInterpreter(lds_dwords=64, num_lanes=6)
        simd.set_lds_dword(2, 0xFFFFFFFF)
        simd.set_exec_mask(0x3F)

        for lane in range(6):
            simd.set_vgpr(0, 8, lane=lane)  # byte 8 -> dword 2
            simd.set_vgpr(1, masks[lane], lane=lane)

        simd.run_simd_wave("ds_and_b32 v0, v1")
        expected = 0xFFFFFFFF
        for m in masks:
            expected &= m
        self.assertEqual(simd.get_lds_dword(2), expected)

    def test_multi_lane_atomic_reservation_ticket(self):
        # Shared bitmask of 8 available resources (bits 0..7)
        simd = LdsSemanticInterpreter(lds_dwords=64, num_lanes=8)
        simd.set_lds_dword(0, 0x000000FF)
        simd.set_exec_mask(0xFF)

        for lane in range(8):
            simd.set_vgpr(0, 0, lane=lane)
            # Each lane claims slot by clearing bit `lane`
            simd.set_vgpr(1, ~(1 << lane) & MASK32, lane=lane)
            simd.set_vgpr(4, 0, lane=lane)

        simd.run_simd_wave("ds_and_rtn_b32 v4, v0, v1")
        # All 8 slots were claimed -> final memory is 0
        self.assertEqual(simd.get_lds_dword(0), 0)
        # Every lane saw that slot `lane` was available at the time of claim
        for lane in range(8):
            self.assertTrue(simd.get_vgpr(4, lane=lane) & (1 << lane))


class TestSpirvMultiLaneAtomicAnd(unittest.TestCase):
    """Executes SPIR-V compute kernels testing OpAtomicAnd multi-lane workgroup behavior."""

    def test_spirv_workgroup_shared_slot_and(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 8 1 1
%uint = OpTypeInt 32 0
%c16 = OpConstant %uint 16
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c0_slot = OpConstant %uint 0
%c_mask = OpConstant %uint 4042322160
%lds_arr = OpTypeArray %uint %c16
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c0_slot
%res = OpAtomicAnd %uint %ptr %c2 %c0 %c_mask
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicSpirvInterp(spvasm)
        final_lds, _ = interp.run(num_invocations=8, seed_lds={0: 0xFFFFFFFF})
        # 0xFFFFFFFF & 4042322160 (0xF0F0F0F0) = 0xF0F0F0F0
        self.assertEqual(final_lds[0], 0xF0F0F0F0)

    def test_spirv_returning_atomic_and_tickets(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 4 1 1
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c16 = OpConstant %uint 16
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c1_slot = OpConstant %uint 1
%c_half = OpConstant %uint 65535
%lds_arr = OpTypeArray %uint %c16
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c1_slot
%old = OpAtomicAnd %uint %ptr %c2 %c0 %c_half
OpStore %out %old
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicSpirvInterp(spvasm)
        final_lds, lane_outs = interp.run(num_invocations=4, seed_lds={1: 0xFFFFFFFF})
        # First invocation got 0xFFFFFFFF, subsequent got 0x0000FFFF (65535)
        self.assertEqual(lane_outs[0]["%out"], 0xFFFFFFFF)
        for i in range(1, 4):
            self.assertEqual(lane_outs[i]["%out"], 0x0000FFFF)
        self.assertEqual(final_lds[1], 0x0000FFFF)

    def test_spirv_unaligned_byte_offset_access(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c16 = OpConstant %uint 16
%c15 = OpConstant %uint 15
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c_base = OpConstant %uint 60
%c_offset = OpConstant %uint 16
%c_mask = OpConstant %uint 2863311530
%lds_arr = OpTypeArray %uint %c16
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%s1 = OpIAdd %uint %c_base %c_offset
%i = OpShiftRightLogical %uint %s1 %c2
%m = OpBitwiseAnd %uint %i %c15
%ptr = OpAccessChain %_ptr_WG_uint %lds %m
%old = OpAtomicAnd %uint %ptr %c2 %c0 %c_mask
OpStore %out %old
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicSpirvInterp(spvasm)
        final_lds, lane_outs = interp.run(seed_lds={3: 0xFFFFFFFF})
        self.assertEqual(lane_outs[0]["%out"], 0xFFFFFFFF)
        self.assertEqual(final_lds[3], 0xAAAAAAAA)


if __name__ == "__main__":
    unittest.main()
