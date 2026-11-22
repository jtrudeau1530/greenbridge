from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

# RDNA2 Architectural Bitmasks & Constants
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
EXEC_ALL_ONES: int = 0xFFFF_FFFF_FFFF_FFFF
EXEC_ALL_ZEROS: int = 0x0000_0000_0000_0000
SCC_SET: int = 1
SCC_CLEARED: int = 0


def u64_to_u32_pair(val: int) -> Tuple[int, int]:
    """Split a 64-bit scalar value into (lo, hi) 32-bit dwords."""
    return val & MASK32, (val >> 32) & MASK32


def u32_pair_to_u64(lo: int, hi: int) -> int:
    """Combine two 32-bit dwords into a 64-bit scalar register value."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def oracle_s_orn2_saveexec_b64(src0: int, exec_in: int) -> Tuple[int, int, int]:
    """Reference mathematical oracle for RDNA2 s_orn2_saveexec_b64.

    Instruction Definition:
        SOP1 / Wave64 Execution Flow Control:
        S_ORN2_SAVEEXEC_B64 SDST, SSRC0

    Architectural Operation:
        SDST = EXEC;
        EXEC = SSRC0 | ~EXEC;
        SCC  = (EXEC != 0);

    Parameters:
        src0: 64-bit scalar source operand (SGPR pair, literal constant, or inline constant).
        exec_in: 64-bit current EXEC mask register value before execution.

    Returns:
        Tuple of (saved_exec, new_exec, scc):
            saved_exec: the pre-instruction EXEC mask destined for SDST.
            new_exec: the updated EXEC mask after applying OR with bitwise NOT of previous EXEC.
            scc: scalar condition code flag (1 if new_exec is non-zero, else 0).
    """
    saved_exec = exec_in & MASK64
    not_exec = (~exec_in) & MASK64
    new_exec = (src0 | not_exec) & MASK64
    scc = SCC_SET if new_exec != 0 else SCC_CLEARED
    return saved_exec, new_exec, scc


@dataclass
class ArchitecturalModel:
    """Architectural register model capturing the hardware execution state."""

    sgprs: Dict[int, int] = field(default_factory=dict)
    exec_mask: int = EXEC_ALL_ONES
    scc: int = SCC_CLEARED

    def get_sgpr32(self, idx: int) -> int:
        return self.sgprs.get(idx, 0) & MASK32

    def set_sgpr32(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK32

    def get_sgpr64(self, idx: int) -> int:
        lo = self.get_sgpr32(idx)
        hi = self.get_sgpr32(idx + 1)
        return u32_pair_to_u64(lo, hi)

    def set_sgpr64(self, idx: int, val: int) -> None:
        lo, hi = u64_to_u32_pair(val)
        self.set_sgpr32(idx, lo)
        self.set_sgpr32(idx + 1, hi)

    def step_orn2_saveexec(self, dst_idx: Optional[int], src0_val: int) -> Tuple[int, int, int]:
        saved, new_exec, scc = oracle_s_orn2_saveexec_b64(src0_val, self.exec_mask)
        if dst_idx is not None:
            self.set_sgpr64(dst_idx, saved)
        self.exec_mask = new_exec
        self.scc = scc
        return saved, new_exec, scc


# -----------------------------------------------------------------------------
# Commit 2: Symbolic AST Opcode Definitions & Typed Expressions
# -----------------------------------------------------------------------------

class RegType(Enum):
    SGPR = auto()
    EXEC = auto()
    SCC = auto()
    NULL = auto()
    LITERAL = auto()


class RegWidth(Enum):
    B32 = 32
    B64 = 64


@dataclass(frozen=True)
class RegRef:
    name: str
    reg_type: RegType
    index: int = 0
    width: RegWidth = RegWidth.B64

    def format_asm(self) -> str:
        if self.reg_type == RegType.SGPR:
            if self.width == RegWidth.B64:
                return f"s[{self.index}:{self.index + 1}]"
            return f"s{self.index}"
        elif self.reg_type == RegType.EXEC:
            return "exec" if self.width == RegWidth.B64 else ("exec_lo" if self.index == 0 else "exec_hi")
        elif self.reg_type == RegType.NULL:
            return "null"
        elif self.reg_type == RegType.SCC:
            return "scc"
        return self.name


@dataclass
class TypedExpr:
    width: RegWidth = RegWidth.B64

    def evaluate(self, env: Dict[str, int]) -> int:
        raise NotImplementedError


@dataclass
class RegExpr(TypedExpr):
    reg: RegRef = field(default_factory=lambda: RegRef("s[0:1]", RegType.SGPR, 0, RegWidth.B64))

    def evaluate(self, env: Dict[str, int]) -> int:
        return env.get(self.reg.name, 0) & (MASK64 if self.width == RegWidth.B64 else MASK32)


@dataclass
class LiteralExpr(TypedExpr):
    val: int = 0

    def evaluate(self, env: Dict[str, int]) -> int:
        return self.val & (MASK64 if self.width == RegWidth.B64 else MASK32)


@dataclass
class BitwiseNotExpr(TypedExpr):
    operand: TypedExpr = field(default_factory=TypedExpr)

    def evaluate(self, env: Dict[str, int]) -> int:
        mask = MASK64 if self.width == RegWidth.B64 else MASK32
        return (~self.operand.evaluate(env)) & mask


@dataclass
class BitwiseOrExpr(TypedExpr):
    lhs: TypedExpr = field(default_factory=TypedExpr)
    rhs: TypedExpr = field(default_factory=TypedExpr)

    def evaluate(self, env: Dict[str, int]) -> int:
        mask = MASK64 if self.width == RegWidth.B64 else MASK32
        return (self.lhs.evaluate(env) | self.rhs.evaluate(env)) & mask


@dataclass
class SOrn2SaveExecB64ASTNode:
    """Symbolic AST node representing s_orn2_saveexec_b64 instruction."""

    sdst: RegRef
    ssrc0: TypedExpr
    exec_ref: RegRef = field(default_factory=lambda: RegRef("exec", RegType.EXEC, 0, RegWidth.B64))
    scc_ref: RegRef = field(default_factory=lambda: RegRef("scc", RegType.SCC, 0, RegWidth.B32))

    def type_check(self) -> bool:
        if self.sdst.reg_type not in (RegType.SGPR, RegType.NULL, RegType.EXEC):
            return False
        if self.sdst.width != RegWidth.B64 and self.sdst.reg_type != RegType.NULL:
            return False
        if self.ssrc0.width != RegWidth.B64:
            return False
        if self.exec_ref.reg_type != RegType.EXEC or self.exec_ref.width != RegWidth.B64:
            return False
        return True

    def to_assembly(self) -> str:
        dst_str = self.sdst.format_asm()
        if isinstance(self.ssrc0, RegExpr):
            src_str = self.ssrc0.reg.format_asm()
        elif isinstance(self.ssrc0, LiteralExpr):
            src_str = f"0x{self.ssrc0.val:X}"
        else:
            src_str = str(self.ssrc0)
        return f"s_orn2_saveexec_b64 {dst_str}, {src_str}"

    def build_symbolic_exec_expr(self) -> BitwiseOrExpr:
        """Construct the symbolic expression tree: ssrc0 | ~EXEC."""
        not_exec = BitwiseNotExpr(width=RegWidth.B64, operand=RegExpr(width=RegWidth.B64, reg=self.exec_ref))
        return BitwiseOrExpr(width=RegWidth.B64, lhs=self.ssrc0, rhs=not_exec)

    def eval_symbolic(self, env: Dict[str, int]) -> Dict[str, int]:
        saved_exec = env.get(self.exec_ref.name, EXEC_ALL_ONES) & MASK64
        expr = self.build_symbolic_exec_expr()
        new_exec = expr.evaluate(env)
        scc = SCC_SET if new_exec != 0 else SCC_CLEARED

        out_env = dict(env)
        if self.sdst.reg_type == RegType.SGPR:
            out_env[self.sdst.name] = saved_exec
        out_env[self.exec_ref.name] = new_exec
        out_env[self.scc_ref.name] = scc
        return out_env


# -----------------------------------------------------------------------------
# Commit 3: Deterministic Scalar Instruction Interpreter
# -----------------------------------------------------------------------------

@dataclass
class ScalarMachineState:
    """Deterministic hardware state representing the RDNA2 Scalar ALU."""

    sgpr: List[int] = field(default_factory=lambda: [0] * 104)
    exec_mask: int = EXEC_ALL_ONES
    scc: int = SCC_CLEARED

    def read_sgpr32(self, idx: int) -> int:
        if not (0 <= idx < 104):
            raise IndexError(f"SGPR index out of range: {idx}")
        return self.sgpr[idx] & MASK32

    def write_sgpr32(self, idx: int, val: int) -> None:
        if not (0 <= idx < 104):
            raise IndexError(f"SGPR index out of range: {idx}")
        self.sgpr[idx] = val & MASK32

    def read_sgpr64(self, idx: int) -> int:
        if idx % 2 != 0:
            raise ValueError(f"64-bit SGPR access must be even-aligned: s{idx}")
        lo = self.read_sgpr32(idx)
        hi = self.read_sgpr32(idx + 1)
        return u32_pair_to_u64(lo, hi)

    def write_sgpr64(self, idx: int, val: int) -> None:
        if idx % 2 != 0:
            raise ValueError(f"64-bit SGPR access must be even-aligned: s{idx}")
        lo, hi = u64_to_u32_pair(val)
        self.write_sgpr32(idx, lo)
        self.write_sgpr32(idx + 1, hi)

    def read_exec(self) -> int:
        return self.exec_mask & MASK64

    def write_exec(self, val: int) -> None:
        self.exec_mask = val & MASK64

    def read_scc(self) -> int:
        return self.scc & 1

    def write_scc(self, val: int) -> None:
        self.scc = 1 if val else 0


@dataclass
class ScalarInstruction:
    """Decoded scalar instruction."""

    opcode: str
    dst: str
    src0: Union[int, str]


class ScalarInterpreter:
    """Deterministic interpreter executing SALU instructions."""

    @staticmethod
    def parse_src64(operand: Union[int, str], state: ScalarMachineState) -> int:
        if isinstance(operand, int):
            return operand & MASK64
        op = operand.strip().lower()
        if op == "exec":
            return state.read_exec()
        if op == "null":
            return 0
        if op.startswith("s[") and op.endswith("]"):
            inner = op[2:-1]
            parts = inner.split(":")
            idx = int(parts[0])
            return state.read_sgpr64(idx)
        if op.startswith("s") and not op.startswith("s["):
            idx = int(op[1:])
            return state.read_sgpr64(idx)
        if op.startswith("0x"):
            return int(op, 16) & MASK64
        return int(op) & MASK64

    def step(self, instr: ScalarInstruction, state: ScalarMachineState) -> Tuple[int, int, int]:
        if instr.opcode != "s_orn2_saveexec_b64":
            raise ValueError(f"Unsupported opcode in ScalarInterpreter: {instr.opcode}")

        src0_val = self.parse_src64(instr.src0, state)
        old_exec = state.read_exec()

        saved_exec, new_exec, scc = oracle_s_orn2_saveexec_b64(src0_val, old_exec)

        dst = instr.dst.strip().lower()
        if dst == "null":
            pass
        elif dst == "exec":
            pass
        elif dst.startswith("s[") and dst.endswith("]"):
            parts = dst[2:-1].split(":")
            dst_idx = int(parts[0])
            state.write_sgpr64(dst_idx, saved_exec)
        elif dst.startswith("s") and not dst.startswith("s["):
            dst_idx = int(dst[1:])
            state.write_sgpr64(dst_idx, saved_exec)
        else:
            raise ValueError(f"Invalid destination register: {instr.dst}")

        state.write_exec(new_exec)
        state.write_scc(scc)

        return saved_exec, new_exec, scc


# -----------------------------------------------------------------------------
# Commit 4: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# -----------------------------------------------------------------------------

class WaveMode(Enum):
    WAVE32 = 32
    WAVE64 = 64


@dataclass
class SIMDMultiLaneState:
    """Multi-lane SIMD wave state capturing lane divergence under EXEC masks."""

    mode: WaveMode = WaveMode.WAVE64
    exec_mask: int = EXEC_ALL_ONES
    vgprs: List[List[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        num_lanes = self.mode.value
        if not self.vgprs:
            self.vgprs = [[0] * 32 for _ in range(num_lanes)]

    def is_lane_active(self, lane_id: int) -> bool:
        if not (0 <= lane_id < self.mode.value):
            return False
        return bool((self.exec_mask >> lane_id) & 1)

    def active_lane_count(self) -> int:
        mask = MASK32 if self.mode == WaveMode.WAVE32 else MASK64
        return bin(self.exec_mask & mask).count("1")

    def read_vgpr(self, lane_id: int, vgpr_idx: int) -> int:
        return self.vgprs[lane_id][vgpr_idx] & MASK32

    def write_vgpr_masked(self, vgpr_idx: int, values_per_lane: List[int]) -> int:
        """Vector ALU write: only active lanes are modified. Inactive lanes preserve contents."""
        updated = 0
        for lane in range(self.mode.value):
            if self.is_lane_active(lane):
                self.vgprs[lane][vgpr_idx] = values_per_lane[lane] & MASK32
                updated += 1
        return updated


class SIMDMultiLaneController:
    """Coordinates SALU saveexec flow control with VALU multi-lane execution."""

    def __init__(self, wave: SIMDMultiLaneState, scalar: ScalarMachineState) -> None:
        self.wave = wave
        self.scalar = scalar
        self.interp = ScalarInterpreter()

    def execute_orn2_saveexec(self, dst_sgpr: str, src0: Union[int, str]) -> Tuple[int, int, int]:
        self.scalar.write_exec(self.wave.exec_mask)
        instr = ScalarInstruction(opcode="s_orn2_saveexec_b64", dst=dst_sgpr, src0=src0)
        saved_exec, new_exec, scc = self.interp.step(instr, self.scalar)
        if self.wave.mode == WaveMode.WAVE32:
            self.wave.exec_mask = new_exec & MASK32
        else:
            self.wave.exec_mask = new_exec & MASK64
        return saved_exec, new_exec, scc

    def restore_exec(self, saved_sgpr: str) -> None:
        restored = self.interp.parse_src64(saved_sgpr, self.scalar)
        self.scalar.write_exec(restored)
        if self.wave.mode == WaveMode.WAVE32:
            self.wave.exec_mask = restored & MASK32
        else:
            self.wave.exec_mask = restored & MASK64


# -----------------------------------------------------------------------------
# Test Suites
# -----------------------------------------------------------------------------

class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model tests."""

    def test_truth_table_bit_by_bit(self) -> None:
        """Exhaustively verify 1-bit combinations of (src0, exec_in)."""
        cases = [
            (0, 0, 0, 1, 1),
            (0, 1, 1, 0, 0),
            (1, 0, 0, 1, 1),
            (1, 1, 1, 1, 1),
        ]
        for s0_bit, ex_bit, exp_saved, exp_new, exp_scc in cases:
            with self.subTest(s0_bit=s0_bit, ex_bit=ex_bit):
                s0 = MASK64 if s0_bit else 0
                ex = MASK64 if ex_bit else 0
                want_saved = MASK64 if exp_saved else 0
                want_new = MASK64 if exp_new else 0
                saved, new_exec, scc = oracle_s_orn2_saveexec_b64(s0, ex)
                self.assertEqual(saved, want_saved)
                self.assertEqual(new_exec, want_new)
                self.assertEqual(scc, exp_scc)

    def test_all_zeros_and_all_ones(self) -> None:
        """Verify boundary extremes: all zeros and all ones in both operands."""
        saved, new_exec, scc = oracle_s_orn2_saveexec_b64(EXEC_ALL_ZEROS, EXEC_ALL_ZEROS)
        self.assertEqual(saved, EXEC_ALL_ZEROS)
        self.assertEqual(new_exec, EXEC_ALL_ONES)
        self.assertEqual(scc, SCC_SET)

        saved, new_exec, scc = oracle_s_orn2_saveexec_b64(EXEC_ALL_ZEROS, EXEC_ALL_ONES)
        self.assertEqual(saved, EXEC_ALL_ONES)
        self.assertEqual(new_exec, EXEC_ALL_ZEROS)
        self.assertEqual(scc, SCC_CLEARED)

        saved, new_exec, scc = oracle_s_orn2_saveexec_b64(EXEC_ALL_ONES, EXEC_ALL_ZEROS)
        self.assertEqual(saved, EXEC_ALL_ZEROS)
        self.assertEqual(new_exec, EXEC_ALL_ONES)
        self.assertEqual(scc, SCC_SET)

        saved, new_exec, scc = oracle_s_orn2_saveexec_b64(EXEC_ALL_ONES, EXEC_ALL_ONES)
        self.assertEqual(saved, EXEC_ALL_ONES)
        self.assertEqual(new_exec, EXEC_ALL_ONES)
        self.assertEqual(scc, SCC_SET)

    def test_alternating_bit_patterns(self) -> None:
        """Test alternating checkerboard patterns 0x5555... and 0xAAAA..."""
        p5 = 0x5555_5555_5555_5555
        pA = 0xAAAA_AAAA_AAAA_AAAA

        saved, new_exec, scc = oracle_s_orn2_saveexec_b64(p5, p5)
        self.assertEqual(saved, p5)
        self.assertEqual(new_exec, EXEC_ALL_ONES)
        self.assertEqual(scc, SCC_SET)

        saved, new_exec, scc = oracle_s_orn2_saveexec_b64(p5, pA)
        self.assertEqual(saved, pA)
        self.assertEqual(new_exec, p5)
        self.assertEqual(scc, SCC_SET)

    def test_walking_ones_and_zeros(self) -> None:
        """Verify walking ones across all 64 bit lanes."""
        for i in range(64):
            bit = 1 << i
            saved, new_exec, scc = oracle_s_orn2_saveexec_b64(bit, EXEC_ALL_ZEROS)
            self.assertEqual(saved, EXEC_ALL_ZEROS)
            self.assertEqual(new_exec, EXEC_ALL_ONES)
            self.assertEqual(scc, SCC_SET)

            saved, new_exec, scc = oracle_s_orn2_saveexec_b64(0, bit)
            self.assertEqual(saved, bit)
            self.assertEqual(new_exec, (~bit) & MASK64)
            self.assertEqual(scc, SCC_SET)

    def test_scc_zero_uniqueness(self) -> None:
        """Prove that SCC == 0 occurs if and only if src0 == 0 and exec_in == ALL_ONES."""
        saved, new_exec, scc = oracle_s_orn2_saveexec_b64(0, EXEC_ALL_ONES)
        self.assertEqual(new_exec, 0)
        self.assertEqual(scc, SCC_CLEARED)

        for i in range(64):
            _, new_exec, scc = oracle_s_orn2_saveexec_b64(1 << i, EXEC_ALL_ONES)
            self.assertNotEqual(new_exec, 0)
            self.assertEqual(scc, SCC_SET)

        for i in range(64):
            exec_val = EXEC_ALL_ONES ^ (1 << i)
            _, new_exec, scc = oracle_s_orn2_saveexec_b64(0, exec_val)
            self.assertNotEqual(new_exec, 0)
            self.assertEqual(scc, SCC_SET)

    def test_architectural_model_execution(self) -> None:
        """Verify architectural model state transitions."""
        model = ArchitecturalModel(exec_mask=0x1234_5678_9ABC_DEF0)
        saved, new_ex, scc = model.step_orn2_saveexec(dst_idx=4, src0_val=0xF000_0000_0000_000F)

        self.assertEqual(saved, 0x1234_5678_9ABC_DEF0)
        self.assertEqual(model.get_sgpr64(4), 0x1234_5678_9ABC_DEF0)

        expected_new = (0xF000_0000_0000_000F | (~0x1234_5678_9ABC_DEF0 & MASK64)) & MASK64
        self.assertEqual(model.exec_mask, expected_new)
        self.assertEqual(model.scc, SCC_SET if expected_new != 0 else SCC_CLEARED)


class TestSymbolicASTDefinitions(unittest.TestCase):
    """Commit 2: Symbolic AST opcode definitions and typed expressions tests."""

    def test_ast_type_checking(self) -> None:
        """Verify AST type check enforcement on registers and expression widths."""
        dst_valid = RegRef("s[4:5]", RegType.SGPR, 4, RegWidth.B64)
        src_valid = RegExpr(width=RegWidth.B64, reg=RegRef("s[6:7]", RegType.SGPR, 6, RegWidth.B64))
        node = SOrn2SaveExecB64ASTNode(sdst=dst_valid, ssrc0=src_valid)
        self.assertTrue(node.type_check())

        dst_invalid = RegRef("s4", RegType.SGPR, 4, RegWidth.B32)
        node_bad_dst = SOrn2SaveExecB64ASTNode(sdst=dst_invalid, ssrc0=src_valid)
        self.assertFalse(node_bad_dst.type_check())

        src_invalid = LiteralExpr(width=RegWidth.B32, val=0x1234)
        node_bad_src = SOrn2SaveExecB64ASTNode(sdst=dst_valid, ssrc0=src_invalid)
        self.assertFalse(node_bad_src.type_check())

        dst_null = RegRef("null", RegType.NULL, 0, RegWidth.B64)
        node_null = SOrn2SaveExecB64ASTNode(sdst=dst_null, ssrc0=src_valid)
        self.assertTrue(node_null.type_check())

    def test_ast_to_assembly(self) -> None:
        """Verify AST string formatting into valid disassembly syntax."""
        dst = RegRef("s[8:9]", RegType.SGPR, 8, RegWidth.B64)
        src_reg = RegExpr(width=RegWidth.B64, reg=RegRef("s[10:11]", RegType.SGPR, 10, RegWidth.B64))
        node_reg = SOrn2SaveExecB64ASTNode(sdst=dst, ssrc0=src_reg)
        self.assertEqual(node_reg.to_assembly(), "s_orn2_saveexec_b64 s[8:9], s[10:11]")

        src_lit = LiteralExpr(width=RegWidth.B64, val=0xDEAD_BEEF_CAFE_BABE)
        node_lit = SOrn2SaveExecB64ASTNode(sdst=dst, ssrc0=src_lit)
        self.assertEqual(node_lit.to_assembly(), "s_orn2_saveexec_b64 s[8:9], 0xDEADBEEFCAFEBABE")

    def test_ast_symbolic_expression_tree(self) -> None:
        """Validate symbolic expression evaluation matches mathematical oracle."""
        node = SOrn2SaveExecB64ASTNode(
            sdst=RegRef("s[0:1]", RegType.SGPR, 0, RegWidth.B64),
            ssrc0=RegExpr(width=RegWidth.B64, reg=RegRef("s[2:3]", RegType.SGPR, 2, RegWidth.B64)),
        )
        expr_tree = node.build_symbolic_exec_expr()

        env = {
            "s[2:3]": 0x0F0F_0F0F_0F0F_0F0F,
            "exec": 0x3333_3333_3333_3333,
        }
        res = expr_tree.evaluate(env)
        _, expected_exec, _ = oracle_s_orn2_saveexec_b64(env["s[2:3]"], env["exec"])
        self.assertEqual(res, expected_exec)

    def test_ast_eval_symbolic_environment(self) -> None:
        """Check symbolic AST environment update behavior."""
        node = SOrn2SaveExecB64ASTNode(
            sdst=RegRef("s[12:13]", RegType.SGPR, 12, RegWidth.B64),
            ssrc0=LiteralExpr(width=RegWidth.B64, val=0x00FF_00FF_00FF_00FF),
        )
        initial_env = {
            "exec": 0xAAAA_5555_AAAA_5555,
            "scc": 0,
        }
        final_env = node.eval_symbolic(initial_env)
        self.assertEqual(final_env["s[12:13]"], 0xAAAA_5555_AAAA_5555)
        expected_exec = (0x00FF_00FF_00FF_00FF | (~0xAAAA_5555_AAAA_5555 & MASK64)) & MASK64
        self.assertEqual(final_env["exec"], expected_exec)
        self.assertEqual(final_env["scc"], SCC_SET if expected_exec != 0 else SCC_CLEARED)


class TestDeterministicScalarInterpreter(unittest.TestCase):
    """Commit 3: Deterministic scalar instruction interpreter tests."""

    def setUp(self) -> None:
        self.state = ScalarMachineState()
        self.interp = ScalarInterpreter()

    def test_interpreter_basic_execution(self) -> None:
        """Test step execution with SGPR operands."""
        self.state.write_sgpr64(4, 0x1111_2222_3333_4444)
        self.state.write_exec(0x5555_6666_7777_8888)

        instr = ScalarInstruction(opcode="s_orn2_saveexec_b64", dst="s[0:1]", src0="s[4:5]")
        saved, new_exec, scc = self.interp.step(instr, self.state)

        self.assertEqual(saved, 0x5555_6666_7777_8888)
        self.assertEqual(self.state.read_sgpr64(0), 0x5555_6666_7777_8888)
        expected_exec = (0x1111_2222_3333_4444 | (~0x5555_6666_7777_8888 & MASK64)) & MASK64
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(self.state.read_exec(), expected_exec)
        self.assertEqual(self.state.read_scc(), SCC_SET if expected_exec != 0 else SCC_CLEARED)

    def test_interpreter_null_destination(self) -> None:
        """Verify writing to null destination discards saved exec without altering SGPRs."""
        self.state.write_sgpr64(0, 0xCAFE_BABE_DEAD_BEEF)
        self.state.write_exec(0xFFFF_0000_FFFF_0000)

        instr = ScalarInstruction(opcode="s_orn2_saveexec_b64", dst="null", src0=0x0000_FFFF_0000_FFFF)
        saved, new_exec, scc = self.interp.step(instr, self.state)

        self.assertEqual(saved, 0xFFFF_0000_FFFF_0000)
        self.assertEqual(self.state.read_sgpr64(0), 0xCAFE_BABE_DEAD_BEEF)
        self.assertEqual(self.state.read_exec(), 0x0000_FFFF_0000_FFFF)
        self.assertEqual(self.state.read_scc(), SCC_SET)

    def test_interpreter_inline_constants(self) -> None:
        """Verify behavior with inline constants 0, -1, 1."""
        self.state.write_exec(EXEC_ALL_ONES)
        instr0 = ScalarInstruction(opcode="s_orn2_saveexec_b64", dst="s[2:3]", src0=0)
        _, new_exec, scc = self.interp.step(instr0, self.state)
        self.assertEqual(new_exec, 0)
        self.assertEqual(scc, SCC_CLEARED)
        self.assertEqual(self.state.read_scc(), SCC_CLEARED)

        instr_neg1 = ScalarInstruction(opcode="s_orn2_saveexec_b64", dst="s[4:5]", src0=-1)
        _, new_exec, scc = self.interp.step(instr_neg1, self.state)
        self.assertEqual(new_exec, EXEC_ALL_ONES)
        self.assertEqual(scc, SCC_SET)
        self.assertEqual(self.state.read_scc(), SCC_SET)

    def test_interpreter_sequential_program(self) -> None:
        """Verify a multi-step sequence simulating control flow branching."""
        self.state.write_exec(0xFFFF_FFFF_0000_0000)
        self.state.write_sgpr64(6, 0x0000_0000_FFFF_0000)

        instr1 = ScalarInstruction(opcode="s_orn2_saveexec_b64", dst="s[8:9]", src0="s[6:7]")
        self.interp.step(instr1, self.state)
        self.assertEqual(self.state.read_sgpr64(8), 0xFFFF_FFFF_0000_0000)
        self.assertEqual(self.state.read_exec(), 0x0000_0000_FFFF_FFFF)

        instr2 = ScalarInstruction(opcode="s_orn2_saveexec_b64", dst="s[10:11]", src0=0)
        self.interp.step(instr2, self.state)
        self.assertEqual(self.state.read_sgpr64(10), 0x0000_0000_FFFF_FFFF)
        self.assertEqual(self.state.read_exec(), 0xFFFF_FFFF_0000_0000)

    def test_interpreter_alignment_enforcement(self) -> None:
        """Verify unaligned 64-bit SGPR accesses raise ValueError."""
        with self.assertRaises(ValueError):
            self.state.read_sgpr64(1)
        with self.assertRaises(ValueError):
            self.state.write_sgpr64(3, 123)


class TestSIMDMultiLaneDivergenceAndExecMasking(unittest.TestCase):
    """Commit 4: SIMD multi-lane divergence and 32/64-bit EXEC masking tests."""

    def test_wave64_multi_lane_divergence(self) -> None:
        """Verify that vector ALU updates strictly follow the EXEC mask derived from s_orn2_saveexec_b64."""
        wave = SIMDMultiLaneState(mode=WaveMode.WAVE64, exec_mask=0x5555_5555_5555_5555)  # Even lanes active
        scalar = ScalarMachineState()
        controller = SIMDMultiLaneController(wave, scalar)

        # Vector write under initial exec: only even lanes should receive 100 + lane
        values_initial = [100 + i for i in range(64)]
        updated_lanes = wave.write_vgpr_masked(0, values_initial)
        self.assertEqual(updated_lanes, 32)
        for i in range(64):
            if i % 2 == 0:
                self.assertEqual(wave.read_vgpr(i, 0), 100 + i)
            else:
                self.assertEqual(wave.read_vgpr(i, 0), 0)

        # Apply s_orn2_saveexec_b64 with src0 mask setting low 16 lanes
        # Old exec = 0x5555_5555_5555_5555
        # ~Old exec = 0xAAAA_AAAA_AAAA_AAAA
        # src0 = 0x0000_0000_0000_FFFF
        # New exec = 0x0000_0000_0000_FFFF | 0xAAAA_AAAA_AAAA_AAAA
        #          = 0xAAAA_AAAA_AAAA_FFFF
        src0_mask = 0x0000_0000_0000_FFFF
        saved, new_exec, scc = controller.execute_orn2_saveexec("s[16:17]", src0_mask)

        self.assertEqual(saved, 0x5555_5555_5555_5555)
        self.assertEqual(scalar.read_sgpr64(16), 0x5555_5555_5555_5555)
        self.assertEqual(new_exec, 0xAAAA_AAAA_AAAA_FFFF)
        self.assertEqual(wave.exec_mask, 0xAAAA_AAAA_AAAA_FFFF)
        self.assertEqual(scc, SCC_SET)

        # Vector write under new exec
        values_second = [500 + i for i in range(64)]
        wave.write_vgpr_masked(1, values_second)
        for i in range(64):
            expected_active = bool((0xAAAA_AAAA_AAAA_FFFF >> i) & 1)
            if expected_active:
                self.assertEqual(wave.read_vgpr(i, 1), 500 + i)
            else:
                self.assertEqual(wave.read_vgpr(i, 1), 0)

    def test_wave32_divergence_masking(self) -> None:
        """Verify Wave32 restricts execution to lanes 0..31 and limits active count."""
        wave = SIMDMultiLaneState(mode=WaveMode.WAVE32, exec_mask=0xFFFF_FFFF)
        scalar = ScalarMachineState()
        controller = SIMDMultiLaneController(wave, scalar)

        self.assertEqual(wave.active_lane_count(), 32)

        # Saveexec with src0 = 0 -> in 64-bit opcode, exec_in has high 32 bits as 0, so ~exec has high 32 bits as 1!
        # Thus new_exec = 0xFFFF_FFFF_0000_0000. Wave32 active lanes (low 32 bits) are 0, but 64-bit SCC is 1!
        controller.execute_orn2_saveexec("s[20:21]", 0)
        self.assertEqual(wave.active_lane_count(), 0)
        self.assertEqual(scalar.read_scc(), SCC_SET)

        # Write attempt under empty exec: 0 lanes updated
        values = [999] * 32
        updated = wave.write_vgpr_masked(0, values)
        self.assertEqual(updated, 0)

    def test_reconvergence_after_saveexec(self) -> None:
        """Simulate shader flow control: saveexec -> diverge -> restore original mask."""
        initial_exec = 0x1234_5678_9ABC_DEF0
        wave = SIMDMultiLaneState(mode=WaveMode.WAVE64, exec_mask=initial_exec)
        scalar = ScalarMachineState()
        controller = SIMDMultiLaneController(wave, scalar)

        # Saveexec
        controller.execute_orn2_saveexec("s[30:31]", 0x0000_FFFF_0000_FFFF)
        self.assertNotEqual(wave.exec_mask, initial_exec)

        # Restore
        controller.restore_exec("s[30:31]")
        self.assertEqual(wave.exec_mask, initial_exec)
        self.assertEqual(scalar.read_exec(), initial_exec)


# -----------------------------------------------------------------------------
# Commit 5: Register Aliasing and In-Place Operand Preservation
# -----------------------------------------------------------------------------

class TestRegisterAliasingAndInPlacePreservation(unittest.TestCase):
    """Commit 5: Register aliasing, overlapping operands, and in-place preservation tests."""

    def setUp(self) -> None:
        self.state = ScalarMachineState()
        self.interp = ScalarInterpreter()

    def test_exact_src0_dst_aliasing_order_of_evaluation(self) -> None:
        """Verify sdst == ssrc0: src0 must be captured before sdst is written with saved exec."""
        val_src = 0x0000_FFFF_0000_FFFF
        old_exec = 0x5555_5555_5555_5555
        self.state.write_sgpr64(4, val_src)
        self.state.write_exec(old_exec)

        instr = ScalarInstruction(opcode="s_orn2_saveexec_b64", dst="s[4:5]", src0="s[4:5]")
        saved, new_exec, scc = self.interp.step(instr, self.state)

        # Saved EXEC returned and placed into s[4:5]
        self.assertEqual(saved, old_exec)
        self.assertEqual(self.state.read_sgpr64(4), old_exec)

        # New EXEC must be computed from val_src, NOT from old_exec!
        expected_new_exec = (val_src | (~old_exec & MASK64)) & MASK64
        self.assertNotEqual(expected_new_exec, EXEC_ALL_ONES)
        self.assertEqual(new_exec, expected_new_exec)
        self.assertEqual(self.state.read_exec(), expected_new_exec)

    def test_src0_is_exec_tautological_ones(self) -> None:
        """Verify ssrc0 == exec: EXEC | ~EXEC is mathematically always ALL_ONES."""
        arbitrary_execs = [
            0x0000_0000_0000_0000,
            0xFFFF_FFFF_FFFF_FFFF,
            0x1234_5678_9ABC_DEF0,
            0xAAAA_5555_AAAA_5555,
            0x8000_0000_0000_0001,
        ]
        for ex in arbitrary_execs:
            with self.subTest(ex=hex(ex)):
                self.state.write_exec(ex)
                instr = ScalarInstruction(opcode="s_orn2_saveexec_b64", dst="s[6:7]", src0="exec")
                saved, new_exec, scc = self.interp.step(instr, self.state)
                self.assertEqual(saved, ex)
                self.assertEqual(self.state.read_sgpr64(6), ex)
                self.assertEqual(new_exec, EXEC_ALL_ONES)
                self.assertEqual(self.state.read_exec(), EXEC_ALL_ONES)
                self.assertEqual(scc, SCC_SET)

    def test_dst_is_exec_overwriting(self) -> None:
        """Verify sdst == exec: new EXEC calculation takes precedence."""
        self.state.write_sgpr64(8, 0x1111_1111_1111_1111)
        self.state.write_exec(0x2222_2222_2222_2222)

        instr = ScalarInstruction(opcode="s_orn2_saveexec_b64", dst="exec", src0="s[8:9]")
        saved, new_exec, scc = self.interp.step(instr, self.state)

        self.assertEqual(saved, 0x2222_2222_2222_2222)
        expected_exec = (0x1111_1111_1111_1111 | (~0x2222_2222_2222_2222 & MASK64)) & MASK64
        self.assertEqual(new_exec, expected_exec)
        self.assertEqual(self.state.read_exec(), expected_exec)

    def test_adjacent_register_non_interference(self) -> None:
        """Verify non-targeted SGPR registers retain their exact contents without bleeding."""
        sentinels = {}
        for idx in range(0, 32, 2):
            val = (0xABCD_0000_0000_0000 | (idx << 32) | (idx * 7919)) & MASK64
            sentinels[idx] = val
            self.state.write_sgpr64(idx, val)

        self.state.write_exec(0xF0F0_F0F0_F0F0_F0F0)

        # Target only s[14:15] with s[8:9]
        instr = ScalarInstruction(opcode="s_orn2_saveexec_b64", dst="s[14:15]", src0="s[8:9]")
        self.interp.step(instr, self.state)

        # Check s[14:15] updated with saved exec
        self.assertEqual(self.state.read_sgpr64(14), 0xF0F0_F0F0_F0F0_F0F0)

        # Check all other SGPRs unaltered
        for idx, expected_val in sentinels.items():
            if idx == 14:
                continue
            self.assertEqual(self.state.read_sgpr64(idx), expected_val, f"SGPR s[{idx}:{idx+1}] corrupted!")

    def test_null_destination_with_aliased_source(self) -> None:
        """Verify sdst=null with ssrc0 being an SGPR preserves ssrc0 completely."""
        self.state.write_sgpr64(18, 0x9999_8888_7777_6666)
        self.state.write_exec(0x1234_1234_1234_1234)

        instr = ScalarInstruction(opcode="s_orn2_saveexec_b64", dst="null", src0="s[18:19]")
        saved, new_exec, scc = self.interp.step(instr, self.state)

        self.assertEqual(saved, 0x1234_1234_1234_1234)
        self.assertEqual(self.state.read_sgpr64(18), 0x9999_8888_7777_6666)
        expected_exec = (0x9999_8888_7777_6666 | (~0x1234_1234_1234_1234 & MASK64)) & MASK64
        self.assertEqual(self.state.read_exec(), expected_exec)


# -----------------------------------------------------------------------------
# Commit 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# -----------------------------------------------------------------------------

@dataclass
class FlowControlStatusFlags:
    """Detailed status metrics evaluating execution mask state transitions."""

    all_lanes_active: bool
    no_lanes_active: bool
    partial_lanes_active: bool
    divergence_altered: bool
    lanes_enabled_count: int
    lanes_disabled_count: int


def evaluate_saveexec_status_flags(old_exec: int, new_exec: int) -> FlowControlStatusFlags:
    """Calculate lane transition dynamics and convergence status."""
    all_active = (new_exec == MASK64)
    no_active = (new_exec == 0)
    partial_active = (0 < new_exec < MASK64)
    divergence_altered = (new_exec != old_exec)
    lanes_enabled = bin(new_exec & (~old_exec & MASK64)).count("1")
    lanes_disabled = bin((~new_exec & MASK64) & old_exec).count("1")
    return FlowControlStatusFlags(
        all_lanes_active=all_active,
        no_lanes_active=no_active,
        partial_lanes_active=partial_active,
        divergence_altered=divergence_altered,
        lanes_enabled_count=lanes_enabled,
        lanes_disabled_count=lanes_disabled,
    )


def scalar_addc_u32(s0: int, s1: int, carry_in: int) -> Tuple[int, int]:
    """Simulate RDNA2 s_addc_u32 consuming SCC carry bit. Returns (sum32, carry_out)."""
    total = (s0 & MASK32) + (s1 & MASK32) + (carry_in & 1)
    return total & MASK32, 1 if total > MASK32 else 0


def scalar_subb_u32(s0: int, s1: int, borrow_in: int) -> Tuple[int, int]:
    """Simulate RDNA2 s_subb_u32 consuming SCC borrow bit. Returns (diff32, borrow_out)."""
    diff = (s0 & MASK32) - (s1 & MASK32) - (borrow_in & 1)
    return diff & MASK32, 1 if diff < 0 else 0


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Commit 6: Condition codes, status flags, and carry/borrow generation tests."""

    def test_scc_exhaustive_zero_conditions(self) -> None:
        """Verify that SCC is 0 ONLY when src0 is 0 and exec is all-ones, 1 otherwise."""
        _, new_exec, scc = oracle_s_orn2_saveexec_b64(0, EXEC_ALL_ONES)
        self.assertEqual(new_exec, 0)
        self.assertEqual(scc, SCC_CLEARED)

        for bit in range(64):
            _, new_exec, scc = oracle_s_orn2_saveexec_b64(1 << bit, EXEC_ALL_ONES)
            self.assertNotEqual(new_exec, 0)
            self.assertEqual(scc, SCC_SET)

        for bit in range(64):
            _, new_exec, scc = oracle_s_orn2_saveexec_b64(0, EXEC_ALL_ONES ^ (1 << bit))
            self.assertNotEqual(new_exec, 0)
            self.assertEqual(scc, SCC_SET)

    def test_status_flags_metrics(self) -> None:
        """Verify status metrics across transitions: full, empty, partial divergence."""
        old_exec = 0x0000_0000_FFFF_FFFF
        src0 = 0xFFFF_0000_0000_0000
        _, new_exec, _ = oracle_s_orn2_saveexec_b64(src0, old_exec)
        flags = evaluate_saveexec_status_flags(old_exec, new_exec)
        self.assertFalse(flags.all_lanes_active)
        self.assertFalse(flags.no_lanes_active)
        self.assertTrue(flags.partial_lanes_active)
        self.assertTrue(flags.divergence_altered)
        self.assertEqual(flags.lanes_enabled_count, 32)
        self.assertEqual(flags.lanes_disabled_count, 32)

        _, new_exec_all, _ = oracle_s_orn2_saveexec_b64(EXEC_ALL_ONES, old_exec)
        flags_all = evaluate_saveexec_status_flags(old_exec, new_exec_all)
        self.assertTrue(flags_all.all_lanes_active)
        self.assertFalse(flags_all.no_lanes_active)
        self.assertEqual(flags_all.lanes_enabled_count, 32)
        self.assertEqual(flags_all.lanes_disabled_count, 0)

        _, new_exec_none, _ = oracle_s_orn2_saveexec_b64(0, EXEC_ALL_ONES)
        flags_none = evaluate_saveexec_status_flags(EXEC_ALL_ONES, new_exec_none)
        self.assertFalse(flags_none.all_lanes_active)
        self.assertTrue(flags_none.no_lanes_active)
        self.assertEqual(flags_none.lanes_enabled_count, 0)
        self.assertEqual(flags_none.lanes_disabled_count, 64)

    def test_scc_forwarding_to_scalar_addc(self) -> None:
        """Test carry generation chain: s_orn2_saveexec_b64 -> SCC -> s_addc_u32."""
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        state.write_exec(0)
        interp.step(ScalarInstruction(opcode="s_orn2_saveexec_b64", dst="s[0:1]", src0=0), state)
        self.assertEqual(state.read_scc(), 1)

        sum_val, carry_out = scalar_addc_u32(0xFFFF_FFFF, 0, state.read_scc())
        self.assertEqual(sum_val, 0)
        self.assertEqual(carry_out, 1)

        state.write_exec(EXEC_ALL_ONES)
        interp.step(ScalarInstruction(opcode="s_orn2_saveexec_b64", dst="s[0:1]", src0=0), state)
        self.assertEqual(state.read_scc(), 0)

        sum_val2, carry_out2 = scalar_addc_u32(0xFFFF_FFFF, 0, state.read_scc())
        self.assertEqual(sum_val2, 0xFFFF_FFFF)
        self.assertEqual(carry_out2, 0)

    def test_scc_forwarding_to_scalar_subb(self) -> None:
        """Test borrow generation chain: s_orn2_saveexec_b64 -> SCC -> s_subb_u32."""
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        state.write_exec(0)
        interp.step(ScalarInstruction(opcode="s_orn2_saveexec_b64", dst="s[0:1]", src0=0), state)
        self.assertEqual(state.read_scc(), 1)

        diff, borrow_out = scalar_subb_u32(0, 0, state.read_scc())
        self.assertEqual(diff, 0xFFFF_FFFF)
        self.assertEqual(borrow_out, 1)

        state.write_exec(EXEC_ALL_ONES)
        interp.step(ScalarInstruction(opcode="s_orn2_saveexec_b64", dst="s[0:1]", src0=0), state)
        self.assertEqual(state.read_scc(), 0)

        diff2, borrow_out2 = scalar_subb_u32(0, 0, state.read_scc())
        self.assertEqual(diff2, 0)
        self.assertEqual(borrow_out2, 0)


# -----------------------------------------------------------------------------
# Commit 7: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# -----------------------------------------------------------------------------

# IEEE 754 Binary64 architectural constants
IEEE_F64_POS_ZERO: int = 0x0000_0000_0000_0000
IEEE_F64_NEG_ZERO: int = 0x8000_0000_0000_0000
IEEE_F64_POS_INF: int = 0x7FF0_0000_0000_0000
IEEE_F64_NEG_INF: int = 0xFFF0_0000_0000_0000
IEEE_F64_QNAN: int = 0x7FF8_0000_0000_0000
IEEE_F64_SNAN: int = 0x7FF0_0000_0000_0001
IEEE_F64_MIN_SUBNORMAL: int = 0x0000_0000_0000_0001
IEEE_F64_MAX_SUBNORMAL: int = 0x000F_FFFF_FFFF_FFFF
IEEE_F64_MIN_NORMAL: int = 0x0010_0000_0000_0000
IEEE_F64_MAX_NORMAL: int = 0x7FEF_FFFF_FFFF_FFFF

IEEE_F64_BOUNDARY_PATTERNS = [
    ("POS_ZERO", IEEE_F64_POS_ZERO),
    ("NEG_ZERO", IEEE_F64_NEG_ZERO),
    ("POS_INF", IEEE_F64_POS_INF),
    ("NEG_INF", IEEE_F64_NEG_INF),
    ("QNAN", IEEE_F64_QNAN),
    ("SNAN", IEEE_F64_SNAN),
    ("MIN_SUBNORMAL", IEEE_F64_MIN_SUBNORMAL),
    ("MAX_SUBNORMAL", IEEE_F64_MAX_SUBNORMAL),
    ("MIN_NORMAL", IEEE_F64_MIN_NORMAL),
    ("MAX_NORMAL", IEEE_F64_MAX_NORMAL),
]


class TestIEEE754BoundaryExtremesAndBitPatterns(unittest.TestCase):
    """Commit 7: IEEE 754 boundary extremes, subnormals, and NaN bit preservation tests."""

    def test_subnormal_no_flush_preservation(self) -> None:
        """Verify subnormal bit patterns are preserved strictly without hardware denormal flushing."""
        saved, new_exec, scc = oracle_s_orn2_saveexec_b64(IEEE_F64_MIN_SUBNORMAL, EXEC_ALL_ONES)
        self.assertEqual(saved, EXEC_ALL_ONES)
        self.assertEqual(new_exec, IEEE_F64_MIN_SUBNORMAL)
        self.assertEqual(scc, SCC_SET)

        saved, new_exec, scc = oracle_s_orn2_saveexec_b64(IEEE_F64_MAX_SUBNORMAL, EXEC_ALL_ONES)
        self.assertEqual(saved, EXEC_ALL_ONES)
        self.assertEqual(new_exec, IEEE_F64_MAX_SUBNORMAL)
        self.assertEqual(scc, SCC_SET)

    def test_nan_payload_bit_preservation(self) -> None:
        """Verify that signaling and diagnostic NaN payloads are never modified or quietened."""
        custom_snan = 0x7FF0_0000_DEAD_BEEF
        saved, new_exec, scc = oracle_s_orn2_saveexec_b64(custom_snan, EXEC_ALL_ONES)
        self.assertEqual(new_exec, custom_snan)
        self.assertEqual(new_exec & 0xFFFF_FFFF, 0xDEAD_BEEF)

    def test_signed_zero_complementation(self) -> None:
        """Verify distinct behavior between +0.0 and -0.0 under bitwise complementation."""
        _, new_exec_pos, scc_pos = oracle_s_orn2_saveexec_b64(0, IEEE_F64_POS_ZERO)
        self.assertEqual(new_exec_pos, EXEC_ALL_ONES)
        self.assertEqual(scc_pos, SCC_SET)

        _, new_exec_neg, scc_neg = oracle_s_orn2_saveexec_b64(0, IEEE_F64_NEG_ZERO)
        self.assertEqual(new_exec_neg, 0x7FFF_FFFF_FFFF_FFFF)
        self.assertEqual(scc_neg, SCC_SET)
        self.assertEqual((new_exec_neg >> 63) & 1, 0)
        self.assertEqual(new_exec_neg & 0x7FFF_FFFF_FFFF_FFFF, 0x7FFF_FFFF_FFFF_FFFF)

    def test_infinity_exponent_mask_interaction(self) -> None:
        """Verify interaction between +/- infinity exponent bit patterns."""
        _, new_exec, scc = oracle_s_orn2_saveexec_b64(0, IEEE_F64_POS_INF)
        expected = (~IEEE_F64_POS_INF) & MASK64
        self.assertEqual(new_exec, expected)
        self.assertEqual((new_exec >> 52) & 0x7FF, 0)
        self.assertEqual(scc, SCC_SET)

    def test_all_ieee754_extremes_cross_product(self) -> None:
        """Exhaustively verify 10x10 combinations of IEEE 754 boundary extremes as src0 and exec."""
        for s0_name, s0_val in IEEE_F64_BOUNDARY_PATTERNS:
            for ex_name, ex_val in IEEE_F64_BOUNDARY_PATTERNS:
                with self.subTest(src0=s0_name, exec_in=ex_name):
                    saved, new_exec, scc = oracle_s_orn2_saveexec_b64(s0_val, ex_val)
                    self.assertEqual(saved, ex_val)
                    expected = (s0_val | (~ex_val & MASK64)) & MASK64
                    self.assertEqual(new_exec, expected)
                    self.assertEqual(scc, SCC_SET if expected != 0 else SCC_CLEARED)


# -----------------------------------------------------------------------------
# Commit 8: Negative Controls, Syntax Rejection, and Mutation Detection
# -----------------------------------------------------------------------------

def mutant_or_instead_of_orn2(src0: int, exec_in: int) -> Tuple[int, int, int]:
    """Defect: computes OR instead of ORN2 (fails to invert second operand)."""
    saved = exec_in & MASK64
    new_exec = (src0 | exec_in) & MASK64
    scc = SCC_SET if new_exec != 0 else SCC_CLEARED
    return saved, new_exec, scc


def mutant_orn1_instead_of_orn2(src0: int, exec_in: int) -> Tuple[int, int, int]:
    """Defect: inverts first operand instead of second operand (~src0 | exec)."""
    saved = exec_in & MASK64
    new_exec = ((~src0 & MASK64) | exec_in) & MASK64
    scc = SCC_SET if new_exec != 0 else SCC_CLEARED
    return saved, new_exec, scc


def mutant_andn2_instead_of_orn2(src0: int, exec_in: int) -> Tuple[int, int, int]:
    """Defect: performs ANDN2 instead of ORN2 (AND instead of OR)."""
    saved = exec_in & MASK64
    new_exec = (src0 & (~exec_in & MASK64)) & MASK64
    scc = SCC_SET if new_exec != 0 else SCC_CLEARED
    return saved, new_exec, scc


def mutant_drop_saved_exec(src0: int, exec_in: int) -> Tuple[int, int, int]:
    """Defect: corrupts or zeroes the saved EXEC return value."""
    saved = 0
    new_exec = (src0 | (~exec_in & MASK64)) & MASK64
    scc = SCC_SET if new_exec != 0 else SCC_CLEARED
    return saved, new_exec, scc


def mutant_inverted_scc_condition(src0: int, exec_in: int) -> Tuple[int, int, int]:
    """Defect: inverts SCC condition flag logic."""
    saved = exec_in & MASK64
    new_exec = (src0 | (~exec_in & MASK64)) & MASK64
    scc = SCC_CLEARED if new_exec != 0 else SCC_SET
    return saved, new_exec, scc


def mutant_32bit_truncation(src0: int, exec_in: int) -> Tuple[int, int, int]:
    """Defect: truncates calculations to 32 bits, dropping high dword."""
    saved = exec_in & MASK32
    new_exec = ((src0 & MASK32) | (~exec_in & MASK32)) & MASK32
    scc = SCC_SET if new_exec != 0 else SCC_CLEARED
    return saved, new_exec, scc


def validate_instruction_syntax(raw_asm: str) -> None:
    """Strict syntax parser and checker for s_orn2_saveexec_b64."""
    line = raw_asm.strip()
    if not line:
        raise ValueError("Empty assembly line")
    parts = line.split(None, 1)
    opcode = parts[0]
    if opcode != "s_orn2_saveexec_b64":
        raise ValueError(f"Unrecognized opcode: {opcode}")
    if len(parts) < 2:
        raise ValueError(f"Missing operands for {opcode}")

    operands = [op.strip() for op in parts[1].split(",")]
    if len(operands) != 2:
        raise ValueError(f"Expected exactly 2 operands, got {len(operands)}")

    dst = operands[0]

    # Validate destination
    if dst == "null" or dst == "exec":
        pass
    elif dst.startswith("v"):
        raise ValueError(f"Vector register {dst} invalid for scalar ALU destination")
    elif dst.startswith("s[") and dst.endswith("]"):
        inner = dst[2:-1]
        range_parts = inner.split(":")
        if len(range_parts) != 2:
            raise ValueError(f"Malformed SGPR range: {dst}")
        lo, hi = int(range_parts[0]), int(range_parts[1])
        if hi != lo + 1:
            raise ValueError(f"64-bit SGPR pair must span exactly 2 dwords: {dst}")
        if lo % 2 != 0:
            raise ValueError(f"64-bit SGPR destination must be even-aligned: {dst}")
        if not (0 <= lo <= 102):
            raise ValueError(f"SGPR index out of architectural bounds [0..102]: {dst}")
    else:
        raise ValueError(f"Invalid scalar destination operand: {dst}")


class TestNegativeControlsAndMutationDetection(unittest.TestCase):
    """Commit 8: Negative controls, syntax rejection, and mutation detection tests."""

    def test_mutation_or_instead_of_orn2_caught(self) -> None:
        """Verify OR mutant produces incorrect result on test vector."""
        src0 = 0x0000_0000_0000_0000
        exec_in = 0x0000_0000_0000_0000
        _, correct_exec, _ = oracle_s_orn2_saveexec_b64(src0, exec_in)
        _, mutant_exec, _ = mutant_or_instead_of_orn2(src0, exec_in)
        self.assertEqual(correct_exec, EXEC_ALL_ONES)
        self.assertEqual(mutant_exec, 0)
        self.assertNotEqual(correct_exec, mutant_exec)

    def test_mutation_inverted_operand_order_caught(self) -> None:
        """Verify ORN1 mutant (~src0 | exec) is caught due to non-commutativity."""
        src0 = 0xFFFF_FFFF_0000_0000
        exec_in = 0x0000_0000_0000_0000
        _, correct_exec, _ = oracle_s_orn2_saveexec_b64(src0, exec_in)
        _, mutant_exec, _ = mutant_orn1_instead_of_orn2(src0, exec_in)
        self.assertEqual(correct_exec, EXEC_ALL_ONES)
        self.assertEqual(mutant_exec, 0x0000_0000_FFFF_FFFF)
        self.assertNotEqual(correct_exec, mutant_exec)

    def test_mutation_andn2_caught(self) -> None:
        """Verify ANDN2 mutant is distinguished from ORN2."""
        src0 = 0x5555_5555_5555_5555
        exec_in = 0x5555_5555_5555_5555
        _, correct_exec, _ = oracle_s_orn2_saveexec_b64(src0, exec_in)
        _, mutant_exec, _ = mutant_andn2_instead_of_orn2(src0, exec_in)
        self.assertEqual(correct_exec, EXEC_ALL_ONES)
        self.assertEqual(mutant_exec, 0)
        self.assertNotEqual(correct_exec, mutant_exec)

    def test_mutation_dropped_save_caught(self) -> None:
        """Verify mutant dropping saved EXEC is caught."""
        exec_in = 0x1234_5678_9ABC_DEF0
        saved_corr, _, _ = oracle_s_orn2_saveexec_b64(0, exec_in)
        saved_mut, _, _ = mutant_drop_saved_exec(0, exec_in)
        self.assertEqual(saved_corr, exec_in)
        self.assertEqual(saved_mut, 0)
        self.assertNotEqual(saved_corr, saved_mut)

    def test_mutation_inverted_scc_caught(self) -> None:
        """Verify mutant inverting SCC flag is caught on zero and non-zero cases."""
        _, _, scc_corr = oracle_s_orn2_saveexec_b64(0, EXEC_ALL_ONES)
        _, _, scc_mut = mutant_inverted_scc_condition(0, EXEC_ALL_ONES)
        self.assertEqual(scc_corr, SCC_CLEARED)
        self.assertEqual(scc_mut, SCC_SET)
        self.assertNotEqual(scc_corr, scc_mut)

    def test_mutation_32bit_truncation_caught(self) -> None:
        """Verify 32-bit truncation mutant is caught when upper 32 lanes differ."""
        src0 = 0xAAAA_0000_0000_0000
        exec_in = 0xFFFF_FFFF_FFFF_FFFF
        _, correct_exec, _ = oracle_s_orn2_saveexec_b64(src0, exec_in)
        _, mutant_exec, _ = mutant_32bit_truncation(src0, exec_in)
        self.assertEqual(correct_exec, 0xAAAA_0000_0000_0000)
        self.assertEqual(mutant_exec, 0)
        self.assertNotEqual(correct_exec, mutant_exec)

    def test_syntax_rejection_unaligned_sgpr(self) -> None:
        """Verify rejection of unaligned SGPR pair destination."""
        with self.assertRaises(ValueError) as ctx:
            validate_instruction_syntax("s_orn2_saveexec_b64 s[1:2], s[4:5]")
        self.assertIn("even-aligned", str(ctx.exception))

    def test_syntax_rejection_out_of_bounds_sgpr(self) -> None:
        """Verify rejection of out-of-bounds SGPR destination."""
        with self.assertRaises(ValueError) as ctx:
            validate_instruction_syntax("s_orn2_saveexec_b64 s[104:105], s[0:1]")
        self.assertIn("bounds", str(ctx.exception))

    def test_syntax_rejection_vgpr_as_scalar_dst(self) -> None:
        """Verify rejection of VGPR used in scalar saveexec destination."""
        with self.assertRaises(ValueError) as ctx:
            validate_instruction_syntax("s_orn2_saveexec_b64 v[0:1], s[0:1]")
        self.assertIn("Vector register", str(ctx.exception))

    def test_syntax_rejection_malformed_syntax(self) -> None:
        """Verify rejection of invalid opcode and missing operands."""
        with self.assertRaises(ValueError):
            validate_instruction_syntax("s_orn2_saveexec_b32 s[0:1], s[2:3]")
        with self.assertRaises(ValueError):
            validate_instruction_syntax("s_orn2_saveexec_b64 s[0:1]")
        with self.assertRaises(ValueError):
            validate_instruction_syntax("")


if __name__ == "__main__":
    unittest.main()
