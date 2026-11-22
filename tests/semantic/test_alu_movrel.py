from __future__ import annotations

"""Semantic interpreter tests and reference oracle for relative register move (v_movrels_b32).

RDNA2 ISA (VOP1 opcode 71 / 0x47):
  v_movrels_b32   D = VGPR[(S0.addr + M0) % 256]
  v_movreld_b32   VGPR[(D.addr + M0) % 256] = S0
  v_movrelsd_b32  VGPR[(D.addr + M0) % 256] = VGPR[(S0.addr + M0) % 256]

The relative register move instruction indexes the vector register file (VGPR)
dynamically relative to the value stored in the scalar M0 register.

In RDNA2 hardware, VGPR indexing is governed by the scalar index register M0.
When performing v_movrels_b32, the source register index is dynamically calculated
as (S0 + M0) mod 256, allowing indexed lookup table reads, dynamic array indexing,
and dynamic register indirection across the 256-entry physical register file.

This module provides:
  1. A mathematical reference oracle (MovrelOracle) and pure-Python register file model
     implementing dynamic VGPR indexing with modulo-256 arithmetic.
  2. Synthetic shader translation and execution via rdna2spirv and spirv_mini_interp /
     spirv_wave_interp to verify translator lowering and execution semantics.
  3. Golden vector verification ensuring parity between the formal ISA model and
     the translator pipeline.
"""

from collections.abc import Sequence
import gc
import os
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import STUB_DXBC, mrt0_words, rdna2spirv  # noqa: E402

MASK32 = 0xFFFFFFFF
NUM_VGPRS = 256  # Size of the VGPR file (256 32-bit registers)


def sentinel(i: int) -> int:
    """The value v_mov_b32 loads into v<i> before the movrel instruction."""
    return 0xC0DE0000 | ((i * 0x111) & 0xFFFF)


SENTINELS: dict[int, int] = {i: sentinel(i) for i in range(1, 9)}
V0_VALUE = 0x5EED0000  # Loaded into v0 by cases that write or seed v0


def compute_movrels_target_vgpr(base_vgpr: int, m0_offset: int, modulo: int = NUM_VGPRS) -> int:
    """Compute effective VGPR index under RDNA2 dynamic relative addressing: (base + M0) mod 256."""
    return (base_vgpr + m0_offset) % modulo


def ref_movrels_b32(
    dst: int,
    src: int,
    m0: int,
    vgpr_file: Sequence[int],
    modulo: int = NUM_VGPRS,
) -> list[int]:
    """Pure-Python reference oracle for v_movrels_b32 execution.

    Loads VGPR[(src + m0) % 256] and writes it to VGPR[dst].
    Returns the updated register file list.
    """
    out = list(vgpr_file)
    target_idx = compute_movrels_target_vgpr(src, m0, modulo)
    out[dst] = out[target_idx] & MASK32
    return out


class MovrelOracle:
    """Stateful architectural register file oracle for M0-relative instructions."""

    def __init__(self, initial_values: dict[int, int] | None = None) -> None:
        self.vgpr = [0] * NUM_VGPRS
        self.m0 = 0
        if initial_values:
            for reg, val in initial_values.items():
                self.vgpr[reg % NUM_VGPRS] = val & MASK32

    def set_m0(self, val: int) -> None:
        self.m0 = val & MASK32

    def set_vgpr(self, reg: int, val: int) -> None:
        self.vgpr[reg % NUM_VGPRS] = val & MASK32

    def get_vgpr(self, reg: int) -> int:
        return self.vgpr[reg % NUM_VGPRS]

    def execute_v_movrels(self, dst: int, src: int) -> int:
        """Execute v_movrels_b32: VGPR[dst] = VGPR[(src + M0) % 256]. Returns value written."""
        src_idx = compute_movrels_target_vgpr(src, self.m0, NUM_VGPRS)
        val = self.vgpr[src_idx]
        self.vgpr[dst % NUM_VGPRS] = val
        return val

    def execute_v_movreld(self, dst: int, src: int) -> int:
        """Execute v_movreld_b32: VGPR[(dst + M0) % 256] = VGPR[src]. Returns value written."""
        dst_idx = compute_movrels_target_vgpr(dst, self.m0, NUM_VGPRS)
        val = self.vgpr[src % NUM_VGPRS]
        self.vgpr[dst_idx] = val
        return val

    def execute_v_movrelsd(self, dst: int, src: int) -> int:
        """Execute v_movrelsd_b32: VGPR[(dst + M0) % 256] = VGPR[(src + M0) % 256]."""
        src_idx = compute_movrels_target_vgpr(src, self.m0, NUM_VGPRS)
        dst_idx = compute_movrels_target_vgpr(dst, self.m0, NUM_VGPRS)
        val = self.vgpr[src_idx]
        self.vgpr[dst_idx] = val
        return val

    def dump_registers(self, regs: Sequence[int]) -> tuple[int, ...]:
        return tuple(self.vgpr[r % NUM_VGPRS] for r in regs)


def model(op: str, dst: int, src: int, m0: int, init: dict[int, int]) -> list[int]:
    """Plain-Python register file model applying `op dst, src` with M0 = m0."""
    vgpr = [0] * NUM_VGPRS
    for r, v in init.items():
        vgpr[r] = v & MASK32
    if op == "v_movrels_b32":
        vgpr[dst] = vgpr[compute_movrels_target_vgpr(src, m0, NUM_VGPRS)]
    elif op == "v_movreld_b32":
        vgpr[compute_movrels_target_vgpr(dst, m0, NUM_VGPRS)] = vgpr[src]
    elif op == "v_movrelsd_b32":
        vgpr[compute_movrels_target_vgpr(dst, m0, NUM_VGPRS)] = vgpr[
            compute_movrels_target_vgpr(src, m0, NUM_VGPRS)
        ]
    else:
        raise ValueError(op)
    return vgpr


S = SENTINELS
# (op, dst, src, m0, load v0?, exported registers, expected words) -- expected words worked out by hand.
CASES = [
    # movrels: D = VGPR[S0 + M0]
    ("v_movrels_b32", 0, 1, 0, False, (0, 1, 2, 3), (S[1], S[1], S[2], S[3])),     # M0 = 0 reads S0 itself
    ("v_movrels_b32", 0, 2, 3, False, (0, 2, 4, 5), (S[5], S[2], S[4], S[5])),     # v2 + 3 = v5
    ("v_movrels_b32", 0, 1, 7, False, (0, 1, 7, 8), (S[8], S[1], S[7], S[8])),     # v1 + 7 = v8
    ("v_movrels_b32", 9, 0, 4, True, (9, 0, 4, 5), (S[4], V0_VALUE, S[4], S[5])),  # base v0, M0 alone indexes
    # movreld: VGPR[D + M0] = S0; neighbours keep their sentinels
    ("v_movreld_b32", 1, 0, 0, True, (0, 1, 2, 3), (V0_VALUE, V0_VALUE, S[2], S[3])),
    ("v_movreld_b32", 3, 0, 2, True, (3, 4, 5, 6), (S[3], S[4], V0_VALUE, S[6])),  # writes v5, not v3
    ("v_movreld_b32", 1, 8, 6, False, (1, 6, 7, 8), (S[1], S[6], S[8], S[8])),     # v1 + 6 = v7 <- v8
    # movrelsd: VGPR[D + M0] = VGPR[S0 + M0]; M0 offsets both sides
    ("v_movrelsd_b32", 2, 1, 0, False, (1, 2, 3, 4), (S[1], S[1], S[3], S[4])),
    ("v_movrelsd_b32", 6, 1, 2, False, (6, 7, 8, 3), (S[6], S[7], S[3], S[3])),    # v8 <- v3
    ("v_movrelsd_b32", 1, 3, 4, False, (1, 3, 5, 7), (S[1], S[3], S[7], S[7])),    # v5 <- v7
]

_TMP: tempfile.TemporaryDirectory | None = None


def setUpModule() -> None:
    global _TMP
    _TMP = tempfile.TemporaryDirectory()
    os.mkdir(os.path.join(_TMP.name, "disasm"))
    with open(os.path.join(_TMP.name, "ps_movrel.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP
    if _TMP is not None:
        gc.collect()
        _TMP.cleanup()
        _TMP = None


def translate_listing(body: Sequence[str]) -> str:
    """Write `body` (instruction lines) as ps_movrel.s, translate it and return the SPIR-V text."""
    assert _TMP is not None, "setUpModule was not executed"
    lines = ["; test_movrel"] + [f"\t{line}" for line in body] + ["\ts_endpgm", ""]
    path = os.path.join(_TMP.name, "disasm", "ps_movrel.s")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(path)
        gc.collect()
    return spv


def initial_regs(load_v0: bool) -> dict[int, int]:
    regs = dict(SENTINELS)
    if load_v0:
        regs[0] = V0_VALUE
    return regs


def run_case(
    op: str,
    dst: int,
    src: int,
    m0: int,
    load_v0: bool,
    exported: Sequence[int],
) -> tuple[int, ...]:
    body = [f"v_mov_b32 v{r}, {v:#x}" for r, v in sorted(initial_regs(load_v0).items())]
    body.append(f"s_mov_b32 m0, {m0}")
    body.append(f"{op} v{dst}, v{src}")
    body.append("exp mrt0 " + ", ".join(f"v{r}" for r in exported) + " done vm")
    return tuple(mrt0_words(translate_listing(body)))


def case_id(case: tuple) -> str:
    op, dst, src, m0 = case[:4]
    return f"{op} v{dst}, v{src} (m0={m0})"


class TestMovrelsReferenceOracle(unittest.TestCase):
    """Rigorous mathematical tests for the v_movrels_b32 reference oracle."""

    def test_oracle_zero_offset(self) -> None:
        """When M0 = 0, dynamic indexing evaluates directly to the source register."""
        for reg_idx in range(NUM_VGPRS):
            target = compute_movrels_target_vgpr(reg_idx, 0)
            self.assertEqual(target, reg_idx)

        oracle = MovrelOracle({10: 0x12345678, 20: 0x87654321})
        oracle.set_m0(0)
        val = oracle.execute_v_movrels(dst=5, src=10)
        self.assertEqual(val, 0x12345678)
        self.assertEqual(oracle.get_vgpr(5), 0x12345678)
        self.assertEqual(oracle.get_vgpr(10), 0x12345678)

    def test_oracle_positive_displacement(self) -> None:
        """Target register index is (S0 + M0) mod 256 for positive displacements."""
        oracle = MovrelOracle()
        oracle.set_vgpr(15, 0xCAFEBABE)
        oracle.set_m0(5)
        # Reading from base v10 with M0=5 resolves to v15
        val = oracle.execute_v_movrels(dst=0, src=10)
        self.assertEqual(val, 0xCAFEBABE)
        self.assertEqual(oracle.get_vgpr(0), 0xCAFEBABE)

    def test_oracle_modulo_256_wraparound(self) -> None:
        """Indexing beyond register 255 wraps cleanly to the beginning of the file."""
        self.assertEqual(compute_movrels_target_vgpr(250, 10), 4)
        self.assertEqual(compute_movrels_target_vgpr(255, 1), 0)
        self.assertEqual(compute_movrels_target_vgpr(255, 255), 254)
        self.assertEqual(compute_movrels_target_vgpr(128, 128), 0)

        oracle = MovrelOracle()
        oracle.set_vgpr(4, 0xDEADBEEF)
        oracle.set_m0(10)
        oracle.execute_v_movrels(dst=1, src=250)
        self.assertEqual(oracle.get_vgpr(1), 0xDEADBEEF)

    def test_oracle_large_m0_values(self) -> None:
        """Arbitrary 32-bit M0 offsets undergo modulo-256 reduction in the register file."""
        self.assertEqual(compute_movrels_target_vgpr(10, 256), 10)
        self.assertEqual(compute_movrels_target_vgpr(5, 512 + 7), 12)
        self.assertEqual(compute_movrels_target_vgpr(0, 1024), 0)
        self.assertEqual(compute_movrels_target_vgpr(1, 0x10000), 1)
        self.assertEqual(compute_movrels_target_vgpr(10, 0x10000003), 13)

        oracle = MovrelOracle()
        oracle.set_vgpr(13, 0xA1B2C3D4)
        oracle.set_m0(0x10000003)
        oracle.execute_v_movrels(dst=2, src=10)
        self.assertEqual(oracle.get_vgpr(2), 0xA1B2C3D4)

    def test_oracle_preserves_other_vgprs(self) -> None:
        """Only the destination register is written; all other registers remain unchanged."""
        oracle = MovrelOracle({r: sentinel(r) for r in range(1, 9)})
        oracle.set_m0(3)
        # S0 = v2, M0 = 3 -> reads v5 (sentinel 5), writes to v0
        oracle.execute_v_movrels(dst=0, src=2)
        self.assertEqual(oracle.get_vgpr(0), SENTINELS[5])
        for r in range(1, 9):
            self.assertEqual(oracle.get_vgpr(r), SENTINELS[r])

    def test_oracle_self_move(self) -> None:
        """When destination matches effective source index, the value is preserved."""
        oracle = MovrelOracle({5: 0x55555555})
        oracle.set_m0(3)
        # dst = 5, src = 2 + 3 = 5
        val = oracle.execute_v_movrels(dst=5, src=2)
        self.assertEqual(val, 0x55555555)
        self.assertEqual(oracle.get_vgpr(5), 0x55555555)

    def test_oracle_overwrite_source(self) -> None:
        """When destination matches base source register but M0 > 0, base register is overwritten."""
        oracle = MovrelOracle({2: 0x22222222, 5: 0x55555555})
        oracle.set_m0(3)
        # dst = 2, src = 2 + 3 = 5 -> v2 becomes 0x55555555
        oracle.execute_v_movrels(dst=2, src=2)
        self.assertEqual(oracle.get_vgpr(2), 0x55555555)
        self.assertEqual(oracle.get_vgpr(5), 0x55555555)

    def test_oracle_sequential_chain(self) -> None:
        """Sequential relative moves correctly update and carry forward register state."""
        oracle = MovrelOracle()
        oracle.set_vgpr(10, 0xAAAAAAAA)
        oracle.set_vgpr(20, 0xBBBBBBBB)
        oracle.set_vgpr(30, 0xCCCCCCCC)

        oracle.set_m0(10)
        oracle.execute_v_movrels(dst=0, src=0)  # v0 <- v10
        oracle.execute_v_movrels(dst=1, src=10) # v1 <- v20

        oracle.set_m0(20)
        oracle.execute_v_movrels(dst=2, src=10) # v2 <- v30

        self.assertEqual(oracle.dump_registers((0, 1, 2)), (0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC))

    def test_pure_function_parity(self) -> None:
        """Parity between stateful MovrelOracle and pure-function ref_movrels_b32."""
        regs = [sentinel(i % 16) for i in range(NUM_VGPRS)]
        oracle = MovrelOracle({i: regs[i] for i in range(NUM_VGPRS)})
        oracle.set_m0(7)
        oracle.execute_v_movrels(dst=50, src=20)

        updated_pure = ref_movrels_b32(dst=50, src=20, m0=7, vgpr_file=regs)
        self.assertEqual(oracle.vgpr, updated_pure)


class ModelMatchesHandConstants(unittest.TestCase):
    def test_model(self):
        for case in CASES:
            op, dst, src, m0, load_v0, exported, expected = case
            with self.subTest(case=case_id(case)):
                vgpr = model(op, dst, src, m0, initial_regs(load_v0))
                self.assertEqual(tuple(vgpr[r] for r in exported), expected)

    def test_sentinels_distinct(self):
        values = list(SENTINELS.values()) + [V0_VALUE]
        self.assertEqual(len(set(values)), len(values))


class TranslatedValues(unittest.TestCase):
    def check(self, op: str) -> None:
        cases = [c for c in CASES if c[0] == op]
        self.assertTrue(cases)
        for case in cases:
            _, dst, src, m0, load_v0, exported, expected = case
            with self.subTest(case=case_id(case)):
                got = run_case(op, dst, src, m0, load_v0, exported)
                self.assertEqual(
                    got,
                    expected,
                    f"{case_id(case)}: got {[hex(w) for w in got]}, expected {[hex(w) for w in expected]}",
                )

    def test_v_movrels_b32(self) -> None:
        self.check("v_movrels_b32")

    def test_v_movreld_b32(self) -> None:
        self.check("v_movreld_b32")

    def test_v_movrelsd_b32(self) -> None:
        self.check("v_movrelsd_b32")


if __name__ == "__main__":
    unittest.main()
