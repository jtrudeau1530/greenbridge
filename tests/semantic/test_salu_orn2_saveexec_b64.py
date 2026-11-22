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

        # Invalid dst width (32-bit SGPR for 64-bit saveexec)
        dst_invalid = RegRef("s4", RegType.SGPR, 4, RegWidth.B32)
        node_bad_dst = SOrn2SaveExecB64ASTNode(sdst=dst_invalid, ssrc0=src_valid)
        self.assertFalse(node_bad_dst.type_check())

        # Invalid src width
        src_invalid = LiteralExpr(width=RegWidth.B32, val=0x1234)
        node_bad_src = SOrn2SaveExecB64ASTNode(sdst=dst_valid, ssrc0=src_invalid)
        self.assertFalse(node_bad_src.type_check())

        # NULL destination is valid
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


if __name__ == "__main__":
    unittest.main()
