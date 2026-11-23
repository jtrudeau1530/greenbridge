"""wave64_halves' operand helpers: base_op suffix stripping, the register keys an operand names (_regs, _half)
and operands(), which splits an instruction dict into (dests, srcs, implicit defs, implicit uses). All inputs are
hand-written instruction dicts in the shape tools/shader_disasm.py emits; no game data is used.

    python -I tests/test_wave64_halves_operands.py
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import wave64_halves as w64  # noqa: E402


def s(r, n=1):
    return {"k": "sgpr", "r": r, "n": n}


def v(r, n=1):
    return {"k": "vgpr", "r": r, "n": n}


def sp(t):
    return {"k": "special", "t": t}


def imm(x):
    return {"k": "imm", "v": x}


def ins(op, *args, mods=()):
    return {"op": op, "args": list(args), "mods": list(mods)}


class BaseOp(unittest.TestCase):
    def test_strips_encoding_suffixes(self):
        self.assertEqual(w64.base_op("v_mov_b32_e32"), "v_mov_b32")
        self.assertEqual(w64.base_op("v_add_co_u32_e64"), "v_add_co_u32")
        self.assertEqual(w64.base_op("v_lshlrev_b32_sdwa"), "v_lshlrev_b32")
        self.assertEqual(w64.base_op("v_or_b32_dpp"), "v_or_b32")

    def test_leaves_other_names_alone(self):
        self.assertEqual(w64.base_op("s_endpgm"), "s_endpgm")
        self.assertEqual(w64.base_op("s_movk_i32"), "s_movk_i32")   # _i32 is not an encoding suffix
        self.assertEqual(w64.base_op(""), "")


class Regs(unittest.TestCase):
    def test_scalar_registers_expand_by_width(self):
        self.assertEqual(w64._regs(s(4, 2)), [("s", 4), ("s", 5)])
        self.assertEqual(w64._regs({"k": "sgpr", "r": 4}), [("s", 4)])      # n defaults to 1

    def test_vector_registers(self):
        self.assertEqual(w64._regs(v(3)), [("v", 3)])
        self.assertEqual(w64._regs(v(3, 2)), [("v", 3), ("v", 4)])
        self.assertEqual(w64._regs({"k": "vlist", "r": [5, 7]}), [("v", 5), ("v", 7)])

    def test_special_registers(self):
        self.assertEqual(w64._regs(sp("vcc")), ["vcc"])
        self.assertEqual(w64._regs(sp("vcc_lo")), ["vcc"])      # either half names the pair
        self.assertEqual(w64._regs(sp("exec_hi")), ["exec"])
        self.assertEqual(w64._regs(sp("scc")), ["scc"])
        self.assertEqual(w64._regs(sp("m0")), ["m0"])

    def test_everything_else_has_no_registers(self):
        self.assertEqual(w64._regs(sp("null")), [])
        self.assertEqual(w64._regs(sp("tba")), [])
        self.assertEqual(w64._regs(imm(5)), [])
        self.assertEqual(w64._regs({"k": "fimm", "v": 1.0}), [])
        self.assertEqual(w64._regs({}), [])


class Half(unittest.TestCase):
    def test_named_32_bit_halves(self):
        self.assertEqual(w64._half(sp("exec_lo")), "lo")
        self.assertEqual(w64._half(sp("vcc_lo")), "lo")
        self.assertEqual(w64._half(sp("exec_hi")), "hi")
        self.assertEqual(w64._half(sp("vcc_hi")), "hi")

    def test_whole_registers_and_other_operands_are_not_halves(self):
        self.assertIsNone(w64._half(sp("exec")))
        self.assertIsNone(w64._half(sp("vcc")))
        self.assertIsNone(w64._half(s(1)))
        self.assertIsNone(w64._half(imm(0)))


class Operands(unittest.TestCase):
    def test_plain_alu_first_operand_is_the_dest(self):
        self.assertEqual(w64.operands(ins("v_add_f32_e32", v(0), v(1), v(2))),
                         ([v(0)], [v(1), v(2)], [], []))
        self.assertEqual(w64.operands(ins("s_mov_b32", s(1), imm(5))),
                         ([s(1)], [imm(5)], [], []))

    def test_no_dest_ops_treat_every_operand_as_a_source(self):
        self.assertEqual(w64.operands(ins("exp", imm(0), v(0), v(1), v(2), v(3))),
                         ([], [imm(0), v(0), v(1), v(2), v(3)], [], []))
        self.assertEqual(w64.operands(ins("buffer_store_dword", v(0), v(1), s(0, 4), imm(0))),
                         ([], [v(0), v(1), s(0, 4), imm(0)], [], []))
        self.assertEqual(w64.operands(ins("s_nop")), ([], [], [], []))
        self.assertEqual(w64.operands(ins("v_nop")), ([], [], [], []))
        self.assertEqual(w64.operands(ins("s_waitcnt", imm(0))), ([], [imm(0)], [], []))

    def test_scalar_compares_define_scc(self):
        self.assertEqual(w64.operands(ins("s_cmp_eq_u32", s(0), imm(1))),
                         ([], [s(0), imm(1)], ["scc"], []))
        self.assertEqual(w64.operands(ins("s_bitcmp1_b32", s(0), imm(1))),
                         ([], [s(0), imm(1)], ["scc"], []))

    def test_vector_compares(self):
        # an explicit destination (the _e64 form) is a normal dest; the two-operand form defines VCC implicitly
        self.assertEqual(w64.operands(ins("v_cmp_gt_f32_e32", sp("vcc_lo"), imm(0), v(1))),
                         ([sp("vcc_lo")], [imm(0), v(1)], [], []))
        self.assertEqual(w64.operands(ins("v_cmp_gt_f32_e32", imm(0), v(1))),
                         ([], [imm(0), v(1)], ["vcc"], []))

    def test_v_cmpx_reports_its_exec_def(self):
        # GFX10 v_cmpx has no destination operand and writes EXEC only; operands() reports that implicit EXEC
        # write (it used to report none, which this test pinned until round 44), so the transfer function models
        # the narrowing.
        self.assertEqual(w64.operands(ins("v_cmpx_eq_u32_e32", s(6), v(1))),
                         ([], [s(6), v(1)], ["exec"], []))

    def test_atomics_return_a_value_only_with_glc(self):
        args = (v(0), v(1), s(0, 4), imm(0))
        self.assertEqual(w64.operands(ins("buffer_atomic_add", *args)),
                         ([], list(args), [], []))
        self.assertEqual(w64.operands(ins("buffer_atomic_add", *args, mods=("glc",))),
                         ([v(0)], list(args), [], []))        # the destination stays in srcs too
        self.assertEqual(w64.operands(ins("flat_atomic_add", v(0), v(1), mods=("glc",))),
                         ([v(0)], [v(0), v(1)], [], []))

    def test_ds_ops_with_a_return_value(self):
        self.assertEqual(w64.operands(ins("ds_read_b32", v(0), v(1))), ([v(0)], [v(1)], [], []))
        self.assertEqual(w64.operands(ins("ds_add_rtn_u32", v(0), v(1), v(2))),
                         ([v(0)], [v(1), v(2)], [], []))
        self.assertEqual(w64.operands(ins("ds_write_b32", v(0), v(1))),
                         ([], [v(0), v(1)], [], []))
        self.assertEqual(w64.operands(ins("ds_nop")), ([], [], [], []))

    def test_dual_dest_vector_ops(self):
        self.assertEqual(w64.operands(ins("v_add_co_u32_e64", v(0), s(6, 2), v(1), v(2))),
                         ([v(0), s(6, 2)], [v(1), v(2)], [], []))
        self.assertEqual(w64.operands(ins("v_add_co_u32", v(0), sp("vcc"), v(1), v(2))),
                         ([v(0), sp("vcc")], [v(1), v(2)], [], []))
        # a carry-out that is not scalar/special stays a source: single dest
        self.assertEqual(w64.operands(ins("v_add_co_u32", v(0), v(1), v(2), v(3))),
                         ([v(0)], [v(1), v(2), v(3)], [], []))
        self.assertEqual(w64.operands(ins("v_mad_u64_u32", v(0, 2), s(6, 2), v(2), v(3), v(4, 2))),
                         ([v(0, 2), s(6, 2)], [v(2), v(3), v(4, 2)], [], []))

    def test_saveexec_defines_exec_and_scc(self):
        self.assertEqual(w64.operands(ins("s_and_saveexec_b64", s(6, 2), sp("exec"))),
                         ([s(6, 2)], [sp("exec")], ["exec", "scc"], []))

    def test_scalar_ops_define_scc_unless_exempted(self):
        self.assertEqual(w64.operands(ins("s_and_b64", s(6, 2), s(2, 2), sp("exec"))),
                         ([s(6, 2)], [s(2, 2), sp("exec")], ["scc"], []))
        self.assertEqual(w64.operands(ins("s_mov_b32", s(1), imm(5))),
                         ([s(1)], [imm(5)], [], []))
        # conservative: an s_ op the model does not exempt is assumed to define SCC
        self.assertEqual(w64.operands(ins("s_sleep", imm(7))), ([], [imm(7)], ["scc"], []))

    def test_s_set_is_a_prefix_so_s_setpc_has_no_dest(self):
        # "s_set" is in NO_DEST and prefix-matching catches s_setpc_* too (the transfer function calls it
        # an unmodelled op regardless).
        self.assertEqual(w64.operands(ins("s_setpc_b64", s(4, 2), s(2, 2))),
                         ([], [s(4, 2), s(2, 2)], [], []))

    def test_implicit_vcc_use(self):
        self.assertEqual(w64.operands(ins("v_cndmask_b32", v(0), v(1), v(2))),
                         ([v(0)], [v(1), v(2)], [], ["vcc"]))
        # an explicit VCC operand is not listed again, and over-long operand lists opt out
        self.assertEqual(w64.operands(ins("v_cndmask_b32", v(0), v(1), v(2), sp("vcc"))),
                         ([v(0)], [v(1), v(2), sp("vcc")], [], []))
        self.assertEqual(w64.operands(ins("v_cndmask_b32", v(0), v(1), v(2), v(3), v(4))),
                         ([v(0)], [v(1), v(2), v(3), v(4)], [], []))

    def test_implicit_scc_use(self):
        self.assertEqual(w64.operands(ins("s_cselect_b32", s(0), s(1), s(2))),
                         ([s(0)], [s(1), s(2)], [], ["scc"]))
        self.assertEqual(w64.operands(ins("s_cbranch_scc1", imm(12))),
                         ([], [imm(12)], [], ["scc"]))

    def test_vcc_and_exec_branches(self):
        self.assertEqual(w64.operands(ins("s_cbranch_vccz", imm(4))), ([], [imm(4)], [], ["vcc"]))
        self.assertEqual(w64.operands(ins("s_branch", imm(4))), ([], [imm(4)], [], []))
        # EXEC branches name their mask implicitly: "exec" is not added to uses
        self.assertEqual(w64.operands(ins("s_cbranch_execnz", imm(4))), ([], [imm(4)], [], []))

    def test_empty_args_give_empty_lists(self):
        self.assertEqual(w64.operands(ins("v_mov_b32_e32")), ([], [], [], []))
        self.assertEqual(w64.operands(ins("s_endpgm")), ([], [], [], []))


if __name__ == "__main__":
    unittest.main()
