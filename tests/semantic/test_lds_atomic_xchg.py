"""Semantic interpreter tests for LDS atomic exchange operations (ds_wrxchg_rtn_b32, ds_wrxchg2_rtn_b32).

Standard library only; no GPU, no LLVM, no external tools, no game files.

Run directly:
    python -I tests/semantic/test_lds_atomic_xchg.py
Or via unittest discovery:
    python -m unittest discover -s tests/semantic -v

Background & Specification Rules:
  * AMD RDNA2 / GFX10 Local Data Share (LDS) Architecture:
    - LDS is an on-chip software-managed scratchpad memory shared across all invocations
      in a workgroup (threadgroup).
    - Single atomic exchange (ds_wrxchg_rtn_b32):
        Syntax: `ds_wrxchg_rtn_b32 vdst, vaddr, vdata [offset:N]`
        Atomically swaps memory contents with vdata and returns prior value:
          effective_byte_address = (vaddr + offset) & 0xFFFFFFFF
          dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
          old_val = LDS[dword_index]
          LDS[dword_index] = vdata & 0xFFFFFFFF
          vdst = old_val
    - Dual atomic exchange (ds_wrxchg2_rtn_b32, ds_wrxchg2st64_rtn_b32):
        Syntax: `ds_wrxchg2_rtn_b32 vdst, vaddr, vdata0, vdata1 [offset0:N] [offset1:M]`
        Executes two 32-bit atomic exchanges from a single base address with two 8-bit offsets.
        Unit stride is 4 bytes (1 dword) for `ds_wrxchg2_rtn_b32`, or 256 bytes (64 dwords)
        for `ds_wrxchg2st64_rtn_b32`.
        Address calculation:
          eff_byte0 = (vaddr + offset0 * unit) & 0xFFFFFFFF
          dword_idx0 = (eff_byte0 >> 2) & (lds_dwords - 1)
          eff_byte1 = (vaddr + offset1 * unit) & 0xFFFFFFFF
          dword_idx1 = (eff_byte1 >> 2) & (lds_dwords - 1)
        Serialization / Address Collision:
          When dword_idx0 == dword_idx1 (both offsets alias to the same LDS location),
          hardware serializes the operations: port 0 executes first, then port 1.
          Therefore:
            vdst[0] = initial_old_value
            LDS[dword_idx] = vdata0
            vdst[1] = vdata0  (observes port 0 write)
            LDS[dword_idx] = vdata1 (final state)
    - Active lane masking:
        LDS instructions are vector operations governed by the execution mask (EXEC).
        Only active lanes participate in LDS mutations and destination register updates.
        Inactive lanes leave LDS memory untouched and preserve destination VGPR contents.
    - Modulo wrapping & dword alignment:
        Byte addresses are converted to dword indices by truncating the low 2 bits (>> 2).
        Indices wrap modulo lds_dwords (a power-of-two capacity, 16..16384 dwords).
        32-bit address overflow wraps strictly modulo 2^32.
  * SPIR-V Representation:
    - LDS is allocated in the Workgroup storage class as an array of uint:
        %lds_arr = OpTypeArray %uint <lds_dwords>
        %_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
        %_ptr_WG_uint = OpTypePointer Workgroup %uint
        %lds = OpVariable %_ptr_WG_lds Workgroup
    - Atomic exchange maps directly to OpAtomicExchange:
        %old = OpAtomicExchange %uint %ptr %scope %semantics %val
        where %scope is Workgroup (2) and %semantics is Relaxed (0).
    - Dual atomic exchange translates into two sequential OpAtomicExchange instructions
      with their respective OpAccessChain pointers.
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

MASK = 0xFFFFFFFF
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


def reference_atomic_xchg(current_val: int, new_val: int) -> tuple[int, int]:
    """Reference 32-bit atomic exchange: returns (old_val, updated_mem_val)."""
    old = current_val & MASK
    new = new_val & MASK
    return old, new


def reference_atomic_xchg2(
    current_val0: int,
    current_val1: int,
    new_val0: int,
    new_val1: int,
    aliased: bool = False,
) -> tuple[int, int, int, int]:
    """Reference dual atomic exchange: returns (old0, old1, final_mem0, final_mem1).

    If aliased is True, both ports access the exact same dword address.
    Per AMD GFX10 / RDNA2 specification, port 0 executes first followed by port 1,
    so old0 is current_val0, old1 is new_val0, and the final memory holds new_val1.
    """
    if aliased:
        old0 = current_val0 & MASK
        old1 = new_val0 & MASK
        final0 = new_val1 & MASK
        final1 = final0
        return old0, old1, final0, final1
    else:
        old0 = current_val0 & MASK
        old1 = current_val1 & MASK
        final0 = new_val0 & MASK
        final1 = new_val1 & MASK
        return old0, old1, final0, final1


def calc_lds_byte_offset(
    vaddr: int,
    byte_offset: int = 0,
    lds_dwords: int = DEFAULT_LDS_DWORDS,
) -> tuple[int, int, int]:
    """Reference ISA calculation for single LDS address: (effective_byte, dword_idx, byte_in_dword)."""
    vaddr = vaddr & MASK
    byte_offset = byte_offset & 0xFFFF  # 16-bit unsigned offset in RDNA2 DS instructions
    eff_byte = (vaddr + byte_offset) & MASK
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
    vaddr = vaddr & MASK
    unit = 256 if st64 else 4
    off0_bytes = (offset0 & 0xFF) * unit
    off1_bytes = (offset1 & 0xFF) * unit
    eff_byte0 = (vaddr + off0_bytes) & MASK
    eff_byte1 = (vaddr + off1_bytes) & MASK
    dword_idx0 = (eff_byte0 >> 2) & (lds_dwords - 1)
    dword_idx1 = (eff_byte1 >> 2) & (lds_dwords - 1)
    return (eff_byte0, dword_idx0), (eff_byte1, dword_idx1)


class LdsSemanticInterpreter:
    """Deterministic semantic interpreter for RDNA2 LDS atomic exchange and memory instructions.

    Models LDS shared scratchpad memory, per-lane VGPR register files, and execution masks.
    """

    def __init__(self, lds_dwords: int = DEFAULT_LDS_DWORDS, num_vgprs: int = 256, num_lanes: int = 64):
        self.lds_dwords = lds_dwords
        self.lds = [0] * lds_dwords
        self.num_vgprs = num_vgprs
        self.num_lanes = num_lanes
        self.vgprs = [[0] * num_vgprs for _ in range(num_lanes)]
        self.active_lanes = [True] * num_lanes

    def set_vgpr(self, reg_idx: int, value: int, lane: int = 0) -> None:
        self.vgprs[lane][reg_idx] = value & MASK

    def get_vgpr(self, reg_idx: int, lane: int = 0) -> int:
        return self.vgprs[lane][reg_idx]

    def set_lds_dword(self, dword_idx: int, value: int) -> None:
        self.lds[dword_idx & (self.lds_dwords - 1)] = value & MASK

    def get_lds_dword(self, dword_idx: int) -> int:
        return self.lds[dword_idx & (self.lds_dwords - 1)]

    def set_lane_active(self, lane: int, active: bool) -> None:
        self.active_lanes[lane] = bool(active)

    def ds_write_b32(self, vaddr_reg: int, vdata_reg: int, offset: int = 0, lane: int = 0) -> int | None:
        """Execute ds_write_b32 vaddr, vdata [offset:offset]."""
        if not self.active_lanes[lane]:
            return None
        vaddr = self.vgprs[lane][vaddr_reg]
        vdata = self.vgprs[lane][vdata_reg]
        _, dword_idx, _ = calc_lds_byte_offset(vaddr, offset, self.lds_dwords)
        self.lds[dword_idx] = vdata & MASK
        return dword_idx

    def ds_read_b32(self, vdst_reg: int, vaddr_reg: int, offset: int = 0, lane: int = 0) -> int | None:
        """Execute ds_read_b32 vdst, vaddr [offset:offset]."""
        if not self.active_lanes[lane]:
            return None
        vaddr = self.vgprs[lane][vaddr_reg]
        _, dword_idx, _ = calc_lds_byte_offset(vaddr, offset, self.lds_dwords)
        self.vgprs[lane][vdst_reg] = self.lds[dword_idx]
        return dword_idx

    def ds_wrxchg_rtn_b32(
        self,
        vdst_reg: int,
        vaddr_reg: int,
        vdata_reg: int,
        offset: int = 0,
        lane: int = 0,
    ) -> tuple[int, int] | None:
        """Execute ds_wrxchg_rtn_b32 vdst, vaddr, vdata [offset:offset].

        Returns (dword_idx, old_val) if active, or None if inactive.
        """
        if not self.active_lanes[lane]:
            return None
        vaddr = self.vgprs[lane][vaddr_reg]
        vdata = self.vgprs[lane][vdata_reg]
        _, dword_idx, _ = calc_lds_byte_offset(vaddr, offset, self.lds_dwords)
        old_val = self.lds[dword_idx]
        self.lds[dword_idx] = vdata & MASK
        self.vgprs[lane][vdst_reg] = old_val
        return dword_idx, old_val

    def ds_wrxchg2_rtn_b32(
        self,
        vdst_reg: int,
        vaddr_reg: int,
        vdata0_reg: int,
        vdata1_reg: int,
        offset0: int = 0,
        offset1: int = 0,
        st64: bool = False,
        lane: int = 0,
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """Execute ds_wrxchg2_rtn_b32 / ds_wrxchg2st64_rtn_b32.

        Dual atomic exchange at two separate offsets from vaddr.
        vdst occupies two consecutive registers: vdst and vdst+1.
        Returns ((idx0, old0), (idx1, old1)) if active, or None if inactive.
        """
        if not self.active_lanes[lane]:
            return None
        vaddr = self.vgprs[lane][vaddr_reg]
        vdata0 = self.vgprs[lane][vdata0_reg]
        vdata1 = self.vgprs[lane][vdata1_reg]
        (_, idx0), (_, idx1) = calc_lds_dual_offsets(vaddr, offset0, offset1, self.lds_dwords, st64=st64)

        if idx0 == idx1:
            # Address collision: port 0 executes first, port 1 observes port 0 write
            old0 = self.lds[idx0]
            self.lds[idx0] = vdata0 & MASK
            old1 = self.lds[idx1]  # equals vdata0
            self.lds[idx1] = vdata1 & MASK
        else:
            old0 = self.lds[idx0]
            self.lds[idx0] = vdata0 & MASK
            old1 = self.lds[idx1]
            self.lds[idx1] = vdata1 & MASK

        self.vgprs[lane][vdst_reg] = old0
        self.vgprs[lane][vdst_reg + 1] = old1
        return (idx0, old0), (idx1, old1)

    def parse_and_execute(self, instruction_text: str, lane: int = 0) -> tuple[int, ...] | int | None:
        """Parse and execute an LDS assembly instruction string for the specified lane."""
        line = instruction_text.strip()
        if not line or line.startswith(";"):
            return None

        # Extract modifiers: offset:N, offset0:N, offset1:N
        offset = 0
        m_off = re.search(r"\boffset:(\S+)", line)
        if m_off:
            offset = int(m_off.group(1), 0)
            line = re.sub(r"\boffset:\S+", "", line).strip()

        offset0 = 0
        m_off0 = re.search(r"\boffset0:(\S+)", line)
        if m_off0:
            offset0 = int(m_off0.group(1), 0)
            line = re.sub(r"\boffset0:\S+", "", line).strip()

        offset1 = 0
        m_off1 = re.search(r"\boffset1:(\S+)", line)
        if m_off1:
            offset1 = int(m_off1.group(1), 0)
            line = re.sub(r"\boffset1:\S+", "", line).strip()

        parts = line.split(None, 1)
        op = parts[0]
        args_str = parts[1] if len(parts) > 1 else ""
        raw_args = [a.strip() for a in args_str.split(",") if a.strip()]

        def reg_num(tok: str) -> int:
            cleaned = tok.replace("v", "").replace("[", "").replace("]", "")
            if ":" in cleaned:
                cleaned = cleaned.split(":")[0]
            return int(cleaned)

        if op == "ds_wrxchg_rtn_b32":
            vdst = reg_num(raw_args[0])
            vaddr = reg_num(raw_args[1])
            vdata = reg_num(raw_args[2])
            res = self.ds_wrxchg_rtn_b32(vdst, vaddr, vdata, offset=offset, lane=lane)
            return res[0] if res else None

        elif op in ("ds_wrxchg2_rtn_b32", "ds_wrxchg2st64_rtn_b32"):
            st64 = "st64" in op
            vdst = reg_num(raw_args[0])
            vaddr = reg_num(raw_args[1])
            vdata0 = reg_num(raw_args[2])
            vdata1 = reg_num(raw_args[3])
            res = self.ds_wrxchg2_rtn_b32(
                vdst, vaddr, vdata0, vdata1, offset0=offset0, offset1=offset1, st64=st64, lane=lane
            )
            return (res[0][0], res[1][0]) if res else None

        elif op == "ds_write_b32":
            vaddr = reg_num(raw_args[0])
            vdata = reg_num(raw_args[1])
            return self.ds_write_b32(vaddr, vdata, offset=offset, lane=lane)

        elif op == "ds_read_b32":
            vdst = reg_num(raw_args[0])
            vaddr = reg_num(raw_args[1])
            return self.ds_read_b32(vdst, vaddr, offset=offset, lane=lane)

        else:
            raise UnsupportedConstruct(f"Opcode {op} not supported by LdsSemanticInterpreter")


class LdsAtomicSpirvInterp:
    """Strict semantic interpreter for compute SPIR-V modules exercising LDS atomic exchange."""

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
                    raise UnsupportedConstruct("Only 32-bit int supported")
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
                self.constants[res] = int(a[1], 0) & MASK
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
                    lds[idx] = val & MASK

        lane_outputs = []
        for lidx in range(count):
            inputs = lane_inputs[lidx] if lane_inputs and lidx < len(lane_inputs) else {}
            out = self._run_lane(lidx, lds, inputs)
            lane_outputs.append(out)

        return list(lds), lane_outputs

    def _run_lane(self, lidx: int, lds: list[int], inputs: dict[str, int]) -> dict[str, int]:
        values = dict(self.constants)
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
                return values[x]
            raise UnsupportedConstruct(f"Undefined identifier {x}")

        for res, op, a, _ in self.body:
            if op in _IGNORED:
                continue
            if op == "OpBranch" or op == "OpBranchConditional":
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
                if ptag == "lds":
                    values[res] = ptr[1][ptr[2]] & MASK
                elif ptag in ("out", "fn"):
                    values[res] = ptr[1][ptr[2]] & MASK
            elif op == "OpStore":
                ptr = pointers[a[0]]
                v = val(a[1]) & MASK
                ptag = ptr[0]
                if ptag == "lds":
                    ptr[1][ptr[2]] = v
                elif ptag in ("out", "fn"):
                    ptr[1][ptr[2]] = v
            elif op == "OpAtomicExchange":
                # %res = OpAtomicExchange %type %ptr %scope %semantics %val
                ptr = pointers[a[1]]
                ptag = ptr[0]
                if ptag != "lds":
                    raise UnsupportedConstruct(f"OpAtomicExchange on non-LDS pointer {ptag}")
                new_val = val(a[4]) & MASK
                old = ptr[1][ptr[2]] & MASK
                ptr[1][ptr[2]] = new_val
                values[res] = old
            elif op == "OpIAdd":
                values[res] = (val(a[1]) + val(a[2])) & MASK
            elif op == "OpISub":
                values[res] = (val(a[1]) - val(a[2])) & MASK
            elif op == "OpShiftRightLogical":
                sh = val(a[2])
                if sh >= 32:
                    raise UnsupportedConstruct("Shift >= 32")
                values[res] = (val(a[1]) >> sh) & MASK
            elif op == "OpShiftLeftLogical":
                sh = val(a[2])
                if sh >= 32:
                    raise UnsupportedConstruct("Shift >= 32")
                values[res] = (val(a[1]) << sh) & MASK
            elif op == "OpBitwiseAnd":
                values[res] = val(a[1]) & val(a[2]) & MASK
            elif op == "OpBitwiseOr":
                values[res] = (val(a[1]) | val(a[2])) & MASK
            elif op == "OpBitwiseXor":
                values[res] = (val(a[1]) ^ val(a[2])) & MASK
            elif op == "OpSelect":
                cond = val(a[1])
                values[res] = val(a[2]) if cond else val(a[3])
            elif op == "OpCopyObject":
                values[res] = val(a[1])
            elif op == "OpBitcast":
                values[res] = val(a[1]) & MASK
            else:
                raise UnsupportedConstruct(f"Opcode {op} not modelled in body")

        return {name: var_box[name][0] & MASK for name in outputs}


class TestAtomicXchgReferenceMath(unittest.TestCase):
    """Verifies 32-bit exchange math, boundary representations, and collision serialization."""

    def test_single_exchange_basic(self):
        old, mem = reference_atomic_xchg(0x12345678, 0x9ABCDEF0)
        self.assertEqual(old, 0x12345678)
        self.assertEqual(mem, 0x9ABCDEF0)

    def test_single_exchange_boundary_values(self):
        cases = [
            (0, 0, 0, 0),
            (0xFFFFFFFF, 0, 0xFFFFFFFF, 0),
            (0, 0xFFFFFFFF, 0, 0xFFFFFFFF),
            (0x80000000, 0x7FFFFFFF, 0x80000000, 0x7FFFFFFF),
            (0x55555555, 0xAAAAAAAA, 0x55555555, 0xAAAAAAAA),
        ]
        for cur, new_val, exp_old, exp_mem in cases:
            with self.subTest(cur=hex(cur), new_val=hex(new_val)):
                old, mem = reference_atomic_xchg(cur, new_val)
                self.assertEqual(old, exp_old)
                self.assertEqual(mem, exp_mem)

    def test_repeated_sequential_exchanges(self):
        # Chaining 10 updates through a single memory cell
        val = 0
        for i in range(1, 11):
            old, val = reference_atomic_xchg(val, i * 100)
            self.assertEqual(old, (i - 1) * 100)
        self.assertEqual(val, 1000)

    def test_dual_exchange_distinct_addresses(self):
        old0, old1, mem0, mem1 = reference_atomic_xchg2(10, 20, 100, 200, aliased=False)
        self.assertEqual(old0, 10)
        self.assertEqual(old1, 20)
        self.assertEqual(mem0, 100)
        self.assertEqual(mem1, 200)

    def test_dual_exchange_aliased_addresses_serialization(self):
        # Port 0 writes 100 to slot (old=10); Port 1 reads 100 and writes 200
        old0, old1, mem0, mem1 = reference_atomic_xchg2(10, 10, 100, 200, aliased=True)
        self.assertEqual(old0, 10)
        self.assertEqual(old1, 100)
        self.assertEqual(mem0, 200)
        self.assertEqual(mem1, 200)


class TestLdsAddressingAndOffsets(unittest.TestCase):
    """Verifies single and dual LDS byte offsets, dword truncation, st64 strides, and wrapping."""

    def test_single_offset_alignment_and_truncation(self):
        lds_dwords = 256
        # Aligned dword addresses
        eff, dword, byte_off = calc_lds_byte_offset(64, 0, lds_dwords)
        self.assertEqual(eff, 64)
        self.assertEqual(dword, 16)
        self.assertEqual(byte_off, 0)

        # Unaligned low bits truncated by >> 2
        for b in (0, 1, 2, 3):
            _, dword, byte_pos = calc_lds_byte_offset(64, b, lds_dwords)
            self.assertEqual(dword, 16)
            self.assertEqual(byte_pos, b)

    def test_single_offset_16bit_masking_and_wrapping(self):
        lds_dwords = 512
        # Offset is 16-bit unsigned (masked to 0xFFFF)
        eff, dword, _ = calc_lds_byte_offset(0, 0x10004, lds_dwords)
        self.assertEqual(eff, 4)
        self.assertEqual(dword, 1)

    def test_dual_offset_unit_strides(self):
        # ds_wrxchg2_rtn_b32 has 4-byte (1 dword) stride
        (eff0, idx0), (eff1, idx1) = calc_lds_dual_offsets(0, 5, 12, lds_dwords=1024, st64=False)
        self.assertEqual(eff0, 20)
        self.assertEqual(idx0, 5)
        self.assertEqual(eff1, 48)
        self.assertEqual(idx1, 12)

    def test_dual_offset_st64_stride(self):
        # ds_wrxchg2st64_rtn_b32 has 256-byte (64 dwords) stride
        (eff0, idx0), (eff1, idx1) = calc_lds_dual_offsets(0, 1, 3, lds_dwords=1024, st64=True)
        self.assertEqual(eff0, 256)
        self.assertEqual(idx0, 64)
        self.assertEqual(eff1, 768)
        self.assertEqual(idx1, 192)

    def test_boundary_power_of_two_wrapping(self):
        # LDS with 64 dwords (256 bytes)
        lds_dwords = 64
        # Byte 256 wraps to dword 0
        _, idx, _ = calc_lds_byte_offset(256, 0, lds_dwords)
        self.assertEqual(idx, 0)
        # Byte 260 wraps to dword 1
        _, idx, _ = calc_lds_byte_offset(256, 4, lds_dwords)
        self.assertEqual(idx, 1)
        # 32-bit wrap
        _, idx, _ = calc_lds_byte_offset(0xFFFFFFFC, 4, lds_dwords)
        self.assertEqual(idx, 0)

    def test_dual_offset_aliasing_detection(self):
        # When offset0 == offset1, idx0 == idx1
        (eff0, idx0), (eff1, idx1) = calc_lds_dual_offsets(100, 4, 4, lds_dwords=256)
        self.assertEqual(eff0, eff1)
        self.assertEqual(idx0, idx1)


class TestLdsSemanticInterpreter(unittest.TestCase):
    """Executes RDNA2 instructions in the software semantic interpreter."""

    def setUp(self):
        self.interp = LdsSemanticInterpreter(lds_dwords=256, num_vgprs=64)

    def test_ds_wrxchg_rtn_b32_basic(self):
        # Seed LDS[10] = 0x12345678
        self.interp.set_lds_dword(10, 0x12345678)
        self.interp.set_vgpr(0, 40)          # vaddr: byte 40 -> dword 10
        self.interp.set_vgpr(1, 0xABCDEF01)    # vdata
        self.interp.set_vgpr(2, 0)             # vdst

        res = self.interp.ds_wrxchg_rtn_b32(vdst_reg=2, vaddr_reg=0, vdata_reg=1, offset=0)
        self.assertEqual(res, (10, 0x12345678))
        # Destination register receives old value
        self.assertEqual(self.interp.get_vgpr(2), 0x12345678)
        # Memory location updated with new value
        self.assertEqual(self.interp.get_lds_dword(10), 0xABCDEF01)
        # Surrounding slots untouched
        self.assertEqual(self.interp.get_lds_dword(9), 0)
        self.assertEqual(self.interp.get_lds_dword(11), 0)

    def test_ds_wrxchg_rtn_b32_with_offset(self):
        self.interp.set_lds_dword(8, 0x5555)
        self.interp.set_vgpr(0, 16)          # byte 16 (dword 4)
        self.interp.set_vgpr(1, 0x7777)
        # offset 16 bytes (+4 dwords) -> dword 8
        self.interp.ds_wrxchg_rtn_b32(vdst_reg=3, vaddr_reg=0, vdata_reg=1, offset=16)
        self.assertEqual(self.interp.get_vgpr(3), 0x5555)
        self.assertEqual(self.interp.get_lds_dword(8), 0x7777)

    def test_ds_wrxchg2_rtn_b32_distinct_locations(self):
        self.interp.set_lds_dword(5, 0xAAAA)
        self.interp.set_lds_dword(7, 0xBBBB)
        self.interp.set_vgpr(0, 0)           # vaddr = 0
        self.interp.set_vgpr(1, 0x1111)      # vdata0
        self.interp.set_vgpr(2, 0x2222)      # vdata1
        # vdst = v10 (v10:v11)
        res = self.interp.ds_wrxchg2_rtn_b32(
            vdst_reg=10, vaddr_reg=0, vdata0_reg=1, vdata1_reg=2, offset0=5, offset1=7
        )
        self.assertEqual(res, ((5, 0xAAAA), (7, 0xBBBB)))
        self.assertEqual(self.interp.get_vgpr(10), 0xAAAA)
        self.assertEqual(self.interp.get_vgpr(11), 0xBBBB)
        self.assertEqual(self.interp.get_lds_dword(5), 0x1111)
        self.assertEqual(self.interp.get_lds_dword(7), 0x2222)

    def test_ds_wrxchg2_rtn_b32_aliased_collision(self):
        # Both offset0 and offset1 point to dword 4
        self.interp.set_lds_dword(4, 0x9999)
        self.interp.set_vgpr(0, 0)
        self.interp.set_vgpr(1, 0x1234)      # vdata0
        self.interp.set_vgpr(2, 0x5678)      # vdata1
        self.interp.ds_wrxchg2_rtn_b32(
            vdst_reg=8, vaddr_reg=0, vdata0_reg=1, vdata1_reg=2, offset0=4, offset1=4
        )
        # Port 0 read initial 0x9999
        self.assertEqual(self.interp.get_vgpr(8), 0x9999)
        # Port 1 read port 0's written value 0x1234
        self.assertEqual(self.interp.get_vgpr(9), 0x1234)
        # Final memory holds port 1's written value 0x5678
        self.assertEqual(self.interp.get_lds_dword(4), 0x5678)

    def test_ds_wrxchg2st64_rtn_b32_execution(self):
        self.interp.set_lds_dword(64, 0x100)
        self.interp.set_lds_dword(128, 0x200)
        self.interp.set_vgpr(0, 0)
        self.interp.set_vgpr(1, 0x300)
        self.interp.set_vgpr(2, 0x400)
        # st64 multiplies offsets 1 and 2 by 64 dwords
        self.interp.ds_wrxchg2_rtn_b32(
            vdst_reg=4, vaddr_reg=0, vdata0_reg=1, vdata1_reg=2, offset0=1, offset1=2, st64=True
        )
        self.assertEqual(self.interp.get_vgpr(4), 0x100)
        self.assertEqual(self.interp.get_vgpr(5), 0x200)
        self.assertEqual(self.interp.get_lds_dword(64), 0x300)
        self.assertEqual(self.interp.get_lds_dword(128), 0x400)

    def test_inactive_lane_preserves_state(self):
        self.interp.set_lds_dword(2, 0x5555)
        self.interp.set_vgpr(0, 8, lane=1)   # byte 8 -> dword 2
        self.interp.set_vgpr(1, 0x9999, lane=1)
        self.interp.set_vgpr(5, 0xDEAD, lane=1)  # preset vdst
        self.interp.set_lane_active(1, False)

        res = self.interp.ds_wrxchg_rtn_b32(vdst_reg=5, vaddr_reg=0, vdata_reg=1, lane=1)
        self.assertIsNone(res)
        # Destination unchanged
        self.assertEqual(self.interp.get_vgpr(5, lane=1), 0xDEAD)
        # LDS unchanged
        self.assertEqual(self.interp.get_lds_dword(2), 0x5555)

    def test_assembly_parser_dispatch(self):
        self.interp.set_lds_dword(3, 0xCAFE)
        self.interp.set_vgpr(0, 12)          # byte 12 -> dword 3
        self.interp.set_vgpr(1, 0xBEEF)

        # Single exchange with offset
        self.interp.parse_and_execute("ds_wrxchg_rtn_b32 v4, v0, v1 offset:0")
        self.assertEqual(self.interp.get_vgpr(4), 0xCAFE)
        self.assertEqual(self.interp.get_lds_dword(3), 0xBEEF)

        # Dual exchange with bracketed register pair
        self.interp.set_lds_dword(10, 0x11)
        self.interp.set_lds_dword(20, 0x22)
        self.interp.set_vgpr(2, 0)
        self.interp.set_vgpr(3, 0x33)
        self.interp.set_vgpr(4, 0x44)
        self.interp.parse_and_execute("ds_wrxchg2_rtn_b32 v[6:7], v2, v3, v4 offset0:10 offset1:20")
        self.assertEqual(self.interp.get_vgpr(6), 0x11)
        self.assertEqual(self.interp.get_vgpr(7), 0x22)
        self.assertEqual(self.interp.get_lds_dword(10), 0x33)
        self.assertEqual(self.interp.get_lds_dword(20), 0x44)

    def test_spinlock_acquire_release_pattern(self):
        # Modeling a mutual exclusion spinlock on LDS[0]
        # 0 = Unlocked, 1 = Locked
        self.interp.set_lds_dword(0, 0)      # initially unlocked
        self.interp.set_vgpr(0, 0)           # vaddr = 0
        self.interp.set_vgpr(1, 1)           # lock payload = 1
        self.interp.set_vgpr(2, 0)           # unlock payload = 0

        # Thread 0 acquires lock
        self.interp.ds_wrxchg_rtn_b32(vdst_reg=5, vaddr_reg=0, vdata_reg=1, lane=0)
        self.assertEqual(self.interp.get_vgpr(5, lane=0), 0)   # 0 -> Acquired!
        self.assertEqual(self.interp.get_lds_dword(0), 1)      # LDS is now 1 (locked)

        # Thread 1 attempts to acquire lock
        self.interp.set_vgpr(0, 0, lane=1)
        self.interp.set_vgpr(1, 1, lane=1)
        self.interp.ds_wrxchg_rtn_b32(vdst_reg=5, vaddr_reg=0, vdata_reg=1, lane=1)
        self.assertEqual(self.interp.get_vgpr(5, lane=1), 1)   # 1 -> Failed/Contended!

        # Thread 0 releases lock
        self.interp.ds_wrxchg_rtn_b32(vdst_reg=6, vaddr_reg=0, vdata_reg=2, lane=0)
        self.assertEqual(self.interp.get_lds_dword(0), 0)      # Unlocked

        # Thread 1 retries and acquires
        self.interp.ds_wrxchg_rtn_b32(vdst_reg=5, vaddr_reg=0, vdata_reg=1, lane=1)
        self.assertEqual(self.interp.get_vgpr(5, lane=1), 0)   # 0 -> Acquired!


class TestSpirvAtomicXchgInterpreter(unittest.TestCase):
    """Executes SPIR-V compute shaders with OpAtomicExchange on LDS memory."""

    def test_op_atomic_exchange_basic(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c64 = OpConstant %uint 64
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c7 = OpConstant %uint 7
%c999 = OpConstant %uint 999
%lds_arr = OpTypeArray %uint %c64
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c7
%old = OpAtomicExchange %uint %ptr %c2 %c0 %c999
OpStore %out %old
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicSpirvInterp(spvasm)
        final_lds, lane_out = interp.run(seed_lds={7: 0x1234})
        # Output gets old value
        self.assertEqual(lane_out[0]["%out"], 0x1234)
        # LDS holds new value
        self.assertEqual(final_lds[7], 999)
        # Other slots pristine
        self.assertEqual(final_lds[0], 0)
        self.assertEqual(final_lds[6], 0)

    def test_chained_atomic_exchanges_single_invocation(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out1 %out2 %out3
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out1 = OpVariable %_ptr_Output_uint Output
%out2 = OpVariable %_ptr_Output_uint Output
%out3 = OpVariable %_ptr_Output_uint Output
%c32 = OpConstant %uint 32
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c3 = OpConstant %uint 3
%v1 = OpConstant %uint 100
%v2 = OpConstant %uint 200
%v3 = OpConstant %uint 300
%lds_arr = OpTypeArray %uint %c32
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c3
%r1 = OpAtomicExchange %uint %ptr %c2 %c0 %v1
%r2 = OpAtomicExchange %uint %ptr %c2 %c0 %v2
%r3 = OpAtomicExchange %uint %ptr %c2 %c0 %v3
OpStore %out1 %r1
OpStore %out2 %r2
OpStore %out3 %r3
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicSpirvInterp(spvasm)
        final_lds, lane_out = interp.run(seed_lds={3: 50})
        # Sequence of observations:
        # r1 observes initial 50 (slot becomes 100)
        # r2 observes 100 (slot becomes 200)
        # r3 observes 200 (slot becomes 300)
        self.assertEqual(lane_out[0]["%out1"], 50)
        self.assertEqual(lane_out[0]["%out2"], 100)
        self.assertEqual(lane_out[0]["%out3"], 200)
        self.assertEqual(final_lds[3], 300)

    def test_dual_exchange_spirv_sequence(self):
        # Emulating ds_wrxchg2_rtn_b32 via two OpAtomicExchange instructions
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out0 %out1
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out0 = OpVariable %_ptr_Output_uint Output
%out1 = OpVariable %_ptr_Output_uint Output
%c64 = OpConstant %uint 64
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%idx0 = OpConstant %uint 4
%idx1 = OpConstant %uint 8
%val0 = OpConstant %uint 1111
%val1 = OpConstant %uint 2222
%lds_arr = OpTypeArray %uint %c64
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr0 = OpAccessChain %_ptr_WG_uint %lds %idx0
%r0 = OpAtomicExchange %uint %ptr0 %c2 %c0 %val0
%ptr1 = OpAccessChain %_ptr_WG_uint %lds %idx1
%r1 = OpAtomicExchange %uint %ptr1 %c2 %c0 %val1
OpStore %out0 %r0
OpStore %out1 %r1
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicSpirvInterp(spvasm)
        final_lds, lane_out = interp.run(seed_lds={4: 10, 8: 20})
        self.assertEqual(lane_out[0]["%out0"], 10)
        self.assertEqual(lane_out[0]["%out1"], 20)
        self.assertEqual(final_lds[4], 1111)
        self.assertEqual(final_lds[8], 2222)

    def test_spirv_address_calculation_pipeline(self):
        # Simulating address math: (base + offset) >> 2 & (lds_dwords - 1)
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
%c_base = OpConstant %uint 44
%c_offset = OpConstant %uint 16
%c_val = OpConstant %uint 8888
%lds_arr = OpTypeArray %uint %c16
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%sum = OpIAdd %uint %c_base %c_offset
%idx = OpShiftRightLogical %uint %sum %c2
%masked = OpBitwiseAnd %uint %idx %c15
%ptr = OpAccessChain %_ptr_WG_uint %lds %masked
%old = OpAtomicExchange %uint %ptr %c2 %c0 %c_val
OpStore %out %old
OpReturn
OpFunctionEnd
"""
        # (44 + 16) = 60; 60 >> 2 = 15; 15 & 15 = 15
        interp = LdsAtomicSpirvInterp(spvasm)
        final_lds, lane_out = interp.run(seed_lds={15: 777})
        self.assertEqual(lane_out[0]["%out"], 777)
        self.assertEqual(final_lds[15], 8888)


class TestWorkgroupMultiLaneAtomicXchg(unittest.TestCase):
    """Verifies workgroup multi-lane contention, permutation conservation, and mutex allocation."""

    def test_multi_lane_shared_slot_exchange_conservation(self):
        # 8 lanes in a workgroup each swap their unique identifier into LDS[0]
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 8 1 1
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c16 = OpConstant %uint 16
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c0_slot = OpConstant %uint 0
%lds_arr = OpTypeArray %uint %c16
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c0_slot
%res = OpAtomicExchange %uint %ptr %c2 %c0 %payload
OpStore %out %res
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicSpirvInterp(spvasm)
        seed_value = 0x5555
        lane_payloads = [{"%payload": 100 + i} for i in range(8)]
        final_lds, lane_outs = interp.run(
            num_invocations=8,
            seed_lds={0: seed_value},
            lane_inputs=lane_payloads,
        )

        returned_values = [o["%out"] for o in lane_outs]
        # Every returned value must be distinct
        self.assertEqual(len(set(returned_values)), 8)

        # The multiset of (all returned values + final value in LDS) MUST equal
        # the initial seed plus all 8 swapped values (conservation of values).
        all_values = set(returned_values) | {final_lds[0]}
        expected_set = {seed_value} | {100 + i for i in range(8)}
        self.assertEqual(all_values, expected_set)
        # Exactly one lane observed the initial seed
        self.assertIn(seed_value, returned_values)

    def test_multi_lane_mutex_race(self):
        # 16 lanes race to swap 1 into LDS[0]. Seed is 0 (unlocked).
        # Exactly ONE lane receives 0 (lock winner); 15 lanes receive 1 (contenders).
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 16 1 1
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c16 = OpConstant %uint 16
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c0_slot = OpConstant %uint 0
%c1 = OpConstant %uint 1
%lds_arr = OpTypeArray %uint %c16
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c0_slot
%ticket = OpAtomicExchange %uint %ptr %c2 %c0 %c1
OpStore %out %ticket
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicSpirvInterp(spvasm)
        final_lds, lane_outs = interp.run(num_invocations=16, seed_lds={0: 0})
        # Final state of lock is 1 (held)
        self.assertEqual(final_lds[0], 1)
        tickets = [o["%out"] for o in lane_outs]
        # Exactly one lane got 0 (acquired)
        self.assertEqual(tickets.count(0), 1)
        # 15 lanes got 1 (spin / busy)
        self.assertEqual(tickets.count(1), 15)

    def test_wave32_token_ring_exchange(self):
        # 32 lanes in wave32 pass tokens through individual LDS slots
        interp = LdsSemanticInterpreter(lds_dwords=32, num_vgprs=8, num_lanes=32)
        # Seed each LDS slot with initial token = slot * 10
        for slot in range(32):
            interp.set_lds_dword(slot, slot * 10)

        # Each lane i exchanges with slot (i + 1) % 32 using payload 1000 + i
        for lane in range(32):
            target_slot = (lane + 1) % 32
            interp.set_vgpr(0, target_slot * 4, lane=lane)  # vaddr
            interp.set_vgpr(1, 1000 + lane, lane=lane)       # vdata
            interp.ds_wrxchg_rtn_b32(vdst_reg=2, vaddr_reg=0, vdata_reg=1, lane=lane)

        # Verify each lane received the token previously in target_slot
        for lane in range(32):
            target_slot = (lane + 1) % 32
            self.assertEqual(interp.get_vgpr(2, lane=lane), target_slot * 10)
            self.assertEqual(interp.get_lds_dword(target_slot), 1000 + lane)


class TestTranslatorIntegration(unittest.TestCase):
    """Verifies tools/rdna2spirv.py handling and SPIR-V code generation for LDS atomic exchanges."""

    def _make_mock_gen(self, stage: str = "cs"):
        class MockGen:
            def __init__(self):
                self.stage = stage
                self.uses_lds = False
                self.lds_dwords = 256
                self.lds_bound = None       # not bounds-checked (SHADER-309): the wrapped access
                self.wave64 = True
                self._tmp = 0
                self.emitted: list[str] = []
                self.stored_dsts: list[tuple[dict, str]] = []

            def tmp(self) -> str:
                self._tmp += 1
                return f"%t{self._tmp}"

            def cu(self, v: int) -> str:
                return f"%c{v}"

            def active_b(self) -> str:
                return "%exec_active"

            def src_u(self, arg: dict) -> str:
                return f"%v_{arg['r']}"

            def dst(self, d: dict, v: str) -> None:
                self.stored_dsts.append((d, v))

            def emit(self, line: str) -> None:
                self.emitted.append(line)

            LDS_ATOMICS = rdna2spirv.Gen.LDS_ATOMICS
            mod_int = staticmethod(rdna2spirv.Gen.mod_int)
            lds_ptr = rdna2spirv.Gen.lds_ptr
            lds_access = rdna2spirv.Gen.lds_access
            lds_atomic = rdna2spirv.Gen.lds_atomic
            lds_op = rdna2spirv.Gen.lds_op

        return MockGen()

    def test_lds_atomics_table_contains_ds_wrxchg_rtn_b32(self):
        # Gen.LDS_ATOMICS maps ds_wrxchg_rtn_b32 to OpAtomicExchange
        self.assertIn("ds_wrxchg_rtn_b32", rdna2spirv.Gen.LDS_ATOMICS)
        self.assertEqual(rdna2spirv.Gen.LDS_ATOMICS["ds_wrxchg_rtn_b32"], "OpAtomicExchange")

    def test_ds_wrxchg_rtn_b32_compute_emission(self):
        gen = self._make_mock_gen(stage="cs")
        # ds_wrxchg_rtn_b32 v5, v0, v1, offset:16 (dst=v5, addr=v0, data=v1)
        gen.lds_op("ds_wrxchg_rtn_b32", [{"r": 5}, {"r": 0}, {"r": 1}], ["offset:16"])
        self.assertTrue(gen.uses_lds)
        emitted_str = "\n".join(gen.emitted)

        # Must guard with active mask
        self.assertIn("OpBranchConditional %exec_active", emitted_str)
        # Must compute pointer with offset 16
        self.assertIn("OpIAdd %uint %v_0 %c16", emitted_str)
        self.assertIn("OpShiftRightLogical %uint", emitted_str)
        self.assertIn("OpBitwiseAnd %uint", emitted_str)
        self.assertIn("OpAccessChain %_ptr_WG_uint %lds", emitted_str)

        # Must emit OpAtomicExchange with scope Workgroup (2) and Relaxed (0)
        self.assertIn("OpAtomicExchange %uint", emitted_str)
        self.assertIn("%c2 %c0 %v_1", emitted_str)

        # Returning form MUST store old value into destination v5
        self.assertEqual(len(gen.stored_dsts), 1)
        dest_reg, val_id = gen.stored_dsts[0]
        self.assertEqual(dest_reg["r"], 5)
        self.assertTrue(val_id.startswith("%t"))

    def test_ds_wrxchg_rtn_b32_mesh_stage_emission(self):
        # Mesh shader (ms) is compute-like and also permits LDS operations
        gen = self._make_mock_gen(stage="ms")
        gen.lds_op("ds_wrxchg_rtn_b32", [{"r": 2}, {"r": 3}, {"r": 4}], [])
        self.assertTrue(gen.uses_lds)
        emitted_str = "\n".join(gen.emitted)
        self.assertIn("OpAtomicExchange %uint", emitted_str)
        self.assertEqual(len(gen.stored_dsts), 1)

    def test_ds_wrxchg_rtn_b32_non_compute_rejected(self):
        # Pixel (ps), vertex (vs), geometry (gs) shaders reject LDS instructions
        for invalid_stage in ("ps", "vs", "gs"):
            with self.subTest(stage=invalid_stage):
                gen = self._make_mock_gen(stage=invalid_stage)
                with self.assertRaises(rdna2spirv.Unsupported):
                    gen.lds_op("ds_wrxchg_rtn_b32", [{"r": 0}, {"r": 1}, {"r": 2}], [])

    def test_ds_wrxchg2_rtn_b32_translator_emission(self):
        gen = self._make_mock_gen(stage="cs")
        dst = {"k": "vgpr", "r": 0, "n": 2}
        gen.lds_op("ds_wrxchg2_rtn_b32", [dst, {"r": 2}, {"r": 3}, {"r": 4}], ["offset0:1", "offset1:2"])
        self.assertTrue(gen.uses_lds)
        emitted_str = "\n".join(gen.emitted)
        self.assertEqual(emitted_str.count("OpAtomicExchange %uint"), 2)
        self.assertEqual(len(gen.stored_dsts), 2)
        self.assertEqual(gen.stored_dsts[0][0]["r"], 0)
        self.assertEqual(gen.stored_dsts[1][0]["r"], 1)

    def test_ds_wrxchg2st64_rtn_b32_translator_emission(self):
        gen = self._make_mock_gen(stage="cs")
        dst = {"k": "vgpr", "r": 6, "n": 2}
        gen.lds_op("ds_wrxchg2st64_rtn_b32", [dst, {"r": 1}, {"r": 2}, {"r": 3}], ["offset0:1", "offset1:3"])
        self.assertTrue(gen.uses_lds)
        emitted_str = "\n".join(gen.emitted)
        self.assertEqual(emitted_str.count("OpAtomicExchange %uint"), 2)
        self.assertIn("OpIAdd %uint %v_1 %c256", emitted_str)
        self.assertIn("OpIAdd %uint %v_1 %c768", emitted_str)
        self.assertEqual(len(gen.stored_dsts), 2)

    def test_ds_wrxchg2_rtn_b32_invalid_dst_rejected(self):
        gen = self._make_mock_gen(stage="cs")
        with self.assertRaises(rdna2spirv.Unsupported):
            gen.lds_op("ds_wrxchg2_rtn_b32", [{"k": "vgpr", "r": 0, "n": 1}, {"r": 1}, {"r": 2}, {"r": 3}], [])
        with self.assertRaises(rdna2spirv.Unsupported):
            gen.lds_op("ds_wrxchg2_rtn_b32", [{"k": "sgpr", "r": 0, "n": 2}, {"r": 1}, {"r": 2}, {"r": 3}], [])


class TestNegativeControlsAndValidation(unittest.TestCase):
    """Verifies error handling for malformed SPIR-V, out-of-bounds sizes, and unsupported constructs."""

    def test_unsupported_storage_class_rejected(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%_ptr_UC = OpTypePointer UniformConstant %uint
%u = OpVariable %_ptr_UC UniformConstant
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"""
        with self.assertRaises(UnsupportedConstruct):
            LdsAtomicSpirvInterp(spvasm)

    def test_invalid_workgroup_size_rejected(self):
        for dims in [(0, 1, 1), (1025, 1, 1), (32, 33, 1)]:
            with self.subTest(dims=dims):
                spvasm = f"""
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize {dims[0]} {dims[1]} {dims[2]}
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"""
                with self.assertRaises(WorkgroupBoundsError):
                    LdsAtomicSpirvInterp(spvasm)

    def test_atomic_exchange_on_non_lds_pointer_rejected(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%_ptr_fn_uint = OpTypePointer Function %uint
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c5 = OpConstant %uint 5
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%local_v = OpVariable %_ptr_fn_uint Function
%bad = OpAtomicExchange %uint %local_v %c2 %c0 %c5
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicSpirvInterp(spvasm)
        with self.assertRaises(UnsupportedConstruct):
            interp.run()

    def test_unmodelled_opcode_in_body_rejected(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%c16 = OpConstant %uint 16
%lds_arr = OpTypeArray %uint %c16
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%bad = OpUDiv %uint %c16 %c16
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicSpirvInterp(spvasm)
        with self.assertRaises(UnsupportedConstruct):
            interp.run()


if __name__ == "__main__":
    unittest.main()
