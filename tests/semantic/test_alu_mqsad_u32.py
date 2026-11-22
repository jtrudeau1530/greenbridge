"""Semantic interpreter tests for RDNA2 v_mqsad_u32_u8.

Four masked byte-SADs with full 32-bit accumulators (RDNA ISA pseudocode). S0 is 64 bits, S1 32 bits, S2 and D are
128 bits (four VGPRs):
  D[k] = (S2[k] + sum over the 4 bytes of |W_k.byte - S1.byte|, skipping bytes where S1's byte is 0) mod 2^32,
  W_k = S0 bits [8k+31 : 8k]
so the four windows slide over S0 one byte at a time, as for v_qsad_pk_u16_u8, but each result is a whole dword.
Each case translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py
and compares D with a plain-Python reference. D leaves through exp as eight 16-bit halves ORed into 0x40000000, so no
export channel is a NaN or denormal pattern that a float round trip could change.
"""
import os
import random
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def ref(x: int, y: int, accs) -> list:
    out = []
    for k in range(4):
        win = (x >> (8 * k)) & M32
        s = sum(abs(((win >> (8 * i)) & 0xFF) - ((y >> (8 * i)) & 0xFF)) for i in range(4) if (y >> (8 * i)) & 0xFF)
        out.append((accs[k] + s) & M32)
    return out


X64 = [0, M64, 0x0807060504030201, 0xFF00FF00FF00FF00, 0x0123456789ABCDEF]
Y32 = [0, 0xFFFFFFFF, 0x04030201, 0x00FF00FF, 0x12000034]
ACCS = [[0, 0, 0, 0], [M32, M32 - 1, 0x7FFFFFFF, 1], [1, 2, 3, 4]]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_mqsad_u32.s")
    with open(os.path.join(_TMP_DIR.name, "ps_mqsad_u32.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_lines(lines) -> list:
    """Run the lines, return v[8:11] (exported as eight 16-bit halves, see the module docstring)."""
    if _TMP_DIR is None:
        setUpModule()
    tail = []
    for k in range(4):
        lo, hi = 20 + 2 * k, 21 + 2 * k
        tail += [f"v_and_b32 v{lo}, 0xffff, v{8 + k}", f"v_lshrrev_b32 v{hi}, 16, v{8 + k}",
                 f"v_or_b32 v{lo}, 0x40000000, v{lo}", f"v_or_b32 v{hi}, 0x40000000, v{hi}"]
    asm = ("; test_mqsad_u32\n" + "".join(f"\t{ln}\n" for ln in list(lines) + tail)
           + "\texp mrt0 v20, v21, v22, v23 vm\n\texp mrt1 v24, v25, v26, v27 done vm\n\ts_endpgm\n")
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    out = si.run_fragment(spv)
    h = list(out["%out_mrt0"]) + list(out["%out_mrt1"])
    assert all(v >> 16 == 0x4000 for v in h), [hex(v) for v in h]
    return [(h[2 * k] & 0xFFFF) | (h[2 * k + 1] & 0xFFFF) << 16 for k in range(4)]


def run_op(x: int, y: int, accs) -> list:
    return run_lines([f"v_mov_b32 v2, {x & M32:#x}", f"v_mov_b32 v3, {x >> 32:#x}", f"v_mov_b32 v1, {y:#x}"]
                     + [f"v_mov_b32 v{4 + k}, {accs[k]:#x}" for k in range(4)]
                     + ["v_mqsad_u32_u8 v[8:11], v[2:3], v1, v[4:7]"])


class MqsadU32Tests(unittest.TestCase):
    def test_reference(self):
        # Windows of 0x0807060504030201 against 0x04030201: SADs 0, 4, 8, 12.
        self.assertEqual(ref(0x0807060504030201, 0x04030201, [0] * 4), [0, 4, 8, 12])
        self.assertEqual(ref(M64, 0, [5, 6, 7, 8]), [5, 6, 7, 8])                     # every byte masked
        self.assertEqual(ref(0, 0xFFFFFFFF, [M32] * 4), [1019] * 4)                    # 1020 - 1, wrapped

    def test_edge_values(self):
        for x in X64:
            for y in Y32:
                accs = ACCS[(x ^ y) % len(ACCS)]
                with self.subTest(x=hex(x), y=hex(y), accs=[hex(a) for a in accs]):
                    self.assertEqual(run_op(x, y, accs), ref(x, y, accs))

    def test_seeded(self):
        rng = random.Random(0x3AD)
        for _ in range(25):
            x, y = rng.getrandbits(64), rng.getrandbits(32)
            if rng.random() < 0.5:
                y &= rng.choice([0xFF00FF00, 0x00FFFF00, 0xFFFFFF00])
            accs = [rng.getrandbits(32) for _ in range(4)]
            with self.subTest(x=hex(x), y=hex(y)):
                self.assertEqual(run_op(x, y, accs), ref(x, y, accs))

    def test_sgpr_source_pair(self):
        got = run_lines(["s_mov_b32 s2, 0x04030201", "s_mov_b32 s3, 0x08070605", "v_mov_b32 v1, 0x04030201"]
                        + [f"v_mov_b32 v{4 + k}, 0" for k in range(4)]
                        + ["v_mqsad_u32_u8 v[8:11], s[2:3], v1, v[4:7]"])
        self.assertEqual(got, [0, 4, 8, 12])

    def test_clamp_is_rejected_not_guessed(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            run_lines(["v_mqsad_u32_u8 v[8:11], v[2:3], v1, v[4:7] clamp"])


if __name__ == "__main__":
    unittest.main()
