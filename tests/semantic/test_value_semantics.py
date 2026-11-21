"""Value-level regression tests for the recent RDNA2 -> SPIR-V semantic fixes.

Each pixel shader ends with `exp mrt0 v.., v.., v.., v..`, so four 32-bit words are checked per shader. The
expectations are written down by hand as constants and recomputed by plain-Python reference functions (checked
against each other in OracleTests). Neither the constants nor the references use the translator or the SPIR-V
interpreter.

ISA semantics used (AMD RDNA2 ISA reference, VALU/VOP3/VOP3P chapters):
  v_mul_legacy_f32      0 * anything, inf/NaN included, is +0.0; otherwise a normal IEEE binary32 multiply.
  v_lshlrev_b16         D.lo = (S1.lo << (S0.lo & 15)) & 0xffff; the VOP3 form keeps D's high half.
  v_lshrrev_b16         same with >>; v_ashrrev_i16 sign-extends S1.lo before the arithmetic shift.
  v_pk_lshrev_b16 /     per half: D.h = (S1.half <</>> (S0.half & 15)) & 0xffff. op_sel picks the source half
  v_pk_lshlrev_b16 /      used for the low result half, op_sel_hi for the high result half.
  v_pk_ashrrev_i16
  v_add_nc_u16          D.lo = (S0.lo + S1.lo) & 0xffff; the VOP3 form keeps D's high half (carry is dropped).
  v_sin_f16/v_cos_f16   input is a number of turns: sin/cos(S * 2*pi), rounded to binary16; high half kept.
  s_buffer_load_dwordx2 with destination vcc: dword 0 -> vcc_lo, dword 1 -> vcc_hi.
  v_interp_mov_f32      reads the flat (provoking-vertex) input attribute; the translator must declare it Flat.

The appended cases for ps_ops5/ps_ops6 also pin the op_sel source-half selection for packed shifts, which the
GPU-checked ps_ops4 does not exercise. cs_bounds (compute) pins the buffer num_records bounds checks: an
out-of-bounds store must not modify memory and an out-of-bounds load must return 0.
"""
import math
import struct
import unittest

import support

# Independent constants, written down by hand (derivations are in the reference functions).
EXPECTED_LEGACY_MUL = [0x00000000, 0x00000000, 0x00000000, 0x40C00000]
EXPECTED_OPS4 = [0x34000010, 0x00120800, 0xBEEF0000, 0x12343C00]
EXPECTED_OPS5 = [0x12340010, 0xBEEF0800, 0xAAAAFFFF, 0xDEADBC00]
EXPECTED_OPS6 = [0xF800FFFF, 0x01000010, 0x00800800, 0x8000FFF0]
EXPECTED_VCC_X2 = [0x11223344, 0x55667788, 0x00000000, 0x00000000]
EXPECTED_FLAT = [0x00000101, 0x00000102, 0x3F000000, 0x3E800000]
EXPECTED_BOUNDS = {0x00: 0x12345678, 0x04: 0x12345678, 0x08: 0x00000000, 0x0C: 0xEEEEEEEE}


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _half(x):
    return struct.unpack("<H", struct.pack("<e", x))[0]


def _unhalf(h):
    return struct.unpack("<e", struct.pack("<H", h))[0]


def _s16(x):
    return x - 0x10000 if x & 0x8000 else x


def _legacy_mul(x, y):
    """v_mul_legacy_f32: zero wins over inf and NaN, including a negative zero."""
    return 0 if x == 0.0 or y == 0.0 else _f32_bits(x * y)


def reference_legacy_mul():
    return [_legacy_mul(0.0, float("inf")), _legacy_mul(0.0, float("nan")),
            _legacy_mul(-0.0, float("-inf")), _legacy_mul(2.0, 3.0)]


def _pk_shift(kind, sh, val, sel=(0, 0), selhi=(1, 1)):
    """One packed 16-bit shift. sel selects (shift-half, value-half) for the low result, selhi for the high."""
    def half(x, n):
        return (x >> (16 * n)) & 0xFFFF

    def one(shalf, vhalf):
        amount = half(sh, shalf) & 15
        x = half(val, vhalf)
        if kind == "ashr":
            return (_s16(x) >> amount) & 0xFFFF
        return ((x << amount) if kind == "lshl" else (x >> amount)) & 0xFFFF

    return one(sel[0], sel[1]) | (one(selhi[0], selhi[1]) << 16)


def reference_ops4():
    v0, shift = 0x12348001, 0x00080004
    sin = _half(math.sin(_unhalf(0x3400) * 2.0 * math.pi))          # 0.25 turns -> 1.0
    add = (0xBEEF0000 & 0xFFFF0000) | ((0x0001 + 0xFFFF) & 0xFFFF)  # carry out of the low half is dropped
    return [_pk_shift("lshl", shift, v0), _pk_shift("lshr", shift, v0), add,
            0x12340000 | sin]


def reference_ops5():
    lshl = 0x12340000 | ((0x8001 << 4) & 0xFFFF)
    lshr = 0xBEEF0000 | ((0x8001 >> 4) & 0xFFFF)
    ashr = 0xAAAA0000 | ((_s16(0xFFF0) >> 4) & 0xFFFF)
    cos = 0xDEAD0000 | _half(math.cos(_unhalf(0x3800) * 2.0 * math.pi))   # 0.5 turns -> -1.0
    return [lshl, lshr, ashr, cos]


def reference_ops6():
    return [_pk_shift("ashr", 0x00040004, 0x8000FFF0),
            # op_sel:[1,0,0] -> low result shifts by the high half of src0; op_sel_hi:[0,0,0] -> high result
            # shifts by the low half. Both use the low half of the value.
            _pk_shift("lshl", 0x00040008, 0x12348001, sel=(1, 0), selhi=(0, 0)),
            _pk_shift("lshr", 0x00040008, 0xBEEF8001, sel=(1, 0), selhi=(0, 0)),
            _pk_shift("ashr", 0, 0x8000FFF0)]


def reference_vcc_x2():
    return [0x11223344, 0x55667788, 0, 0]


def reference_flat():
    return [0x101, 0x101 + 1, _f32_bits(0.5), _f32_bits(0.25)]


def reference_bounds():
    # num_records = 16: store 0 and load 0 are in bounds, the store and load at offset 16 are not. The OOB load
    # feeds a store of 0 into offset 8; the OOB store must leave offset 16 untouched (it is not in the map).
    return {0x00: 0x12345678, 0x04: 0x12345678, 0x08: 0, 0x0C: 0xEEEEEEEE}


def _isg1_dxbc(elements):
    """A DXBC container with one ISG1 input signature. elements: (register, mask, system value, component type)."""
    payload = bytearray(struct.pack("<II", len(elements), 8))
    for reg, mask, sysv, ctype in elements:
        e = bytearray(32)
        struct.pack_into("<6I", e, 0, 0, 0, 0, sysv, ctype, reg)
        e[24], e[25] = mask, 1                                # read/write mask: the shader reads it
        payload += e
    part = b"ISG1" + struct.pack("<I", len(payload)) + bytes(payload)
    part += bytes((-len(part)) % 4)
    header = b"DXBC" + bytes(16) + struct.pack("<III", 1, 0, 1)
    return header + struct.pack("<I", len(header) + 8) + bytes(4) + part


def _fssm(rsrc2, group):
    """The compute-shader header words the translator reads: COMPUTE_PGM_RSRC2 and the thread-group size."""
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, rsrc2, *group)
    return bytes(header)


# attr0 = TEXCOORD0 (x/y interpolated), attr1 = TEXCOORD1 (read through v_interp_mov only, so Flat).
PS_FLAT_DXBC = _isg1_dxbc([(0, 0x3, 0, 3), (1, 0x1, 0, 3)])
FLAT_ATTR1_X = struct.unpack("<f", struct.pack("<I", 0x101))[0]
# 4 user SGPRs (the output V#), workgroup id in s4, 512 bytes of LDS, 64x1x1 threads: the header cs_bounds was
# written for in runtime/test/shaders/build.py.
CS_BOUNDS_FSSM = _fssm(0x8088, (64, 1, 1))


class OracleTests(unittest.TestCase):
    """The hand-written constants and the plain-Python reference agree (so the oracle is self-consistent)."""

    def test_reference_matches_constants(self):
        self.assertEqual(reference_legacy_mul(), EXPECTED_LEGACY_MUL)
        self.assertEqual(reference_ops4(), EXPECTED_OPS4)
        self.assertEqual(reference_ops5(), EXPECTED_OPS5)
        self.assertEqual(reference_ops6(), EXPECTED_OPS6)
        self.assertEqual(reference_vcc_x2(), EXPECTED_VCC_X2)
        self.assertEqual(reference_flat(), EXPECTED_FLAT)
        self.assertEqual(reference_bounds(), EXPECTED_BOUNDS)


class TranslatorValueSemantics(unittest.TestCase):
    def check(self, name, expected, dxbc=None, **run_options):
        spv = support.translate_fixture(name, dxbc=dxbc) if dxbc is not None else support.translate_fixture(name)
        words = support.mrt0_words(spv, **run_options)
        self.assertEqual(len(words), 4, "mrt0 has four components")
        for i, (got, want) in enumerate(zip(words, expected)):
            with self.subTest(shader=name, component="xyzw"[i]):
                self.assertEqual(f"{got:#010x}", f"{want:#010x}")

    def test_v_mul_legacy_f32_zero_times_inf_nan(self):
        # 0*inf, 0*NaN and -0*-inf all give +0.0; 2*3 is an ordinary multiply.
        self.check("ps_legacy_mul", EXPECTED_LEGACY_MUL)

    def test_packed_b16_shifts_add_nc_u16_and_sin_f16(self):
        # Per-half shifts with independent amounts (4 and 8), a 16-bit add whose carry is dropped while the
        # destination's high half is kept, and sin_f16 reading its input as turns.
        self.check("ps_ops4", EXPECTED_OPS4)

    def test_scalar_b16_shifts_and_cos_f16(self):
        # VOP3 forms keep the destination's high half; ashrrev sign-extends the 16-bit value before shifting.
        self.check("ps_ops5", EXPECTED_OPS5)

    def test_packed_b16_shifts_with_op_sel(self):
        # Packed ashrrev (both halves negative) and lshl/lshr selecting source halves through op_sel/op_sel_hi.
        self.check("ps_ops6", EXPECTED_OPS6)

    def test_s_buffer_load_dwordx2_into_vcc(self):
        # The V# in s[0:3] points at 0x1000; the two dwords must come back in vcc_lo and vcc_hi, not be lost.
        push = [0x1000, 0, 16, 0] + [0] * 28
        self.check("ps_vcc_x2", EXPECTED_VCC_X2, push_constants=push,
                   memory={0x1000: 0x11223344, 0x1004: 0x55667788})

    def test_flat_input_and_interpolated_input(self):
        # attr1.x is read only through v_interp_mov, so it carries the provoking vertex's 0x101 unchanged and
        # integer maths works on it; attr0.xy is read through v_interp and keeps the supplied values.
        self.check("ps_flat", EXPECTED_FLAT, dxbc=PS_FLAT_DXBC,
                   inputs={"%in_attr0": (0.5, 0.25, 0.0, 1.0), "%in_attr1": (FLAT_ATTR1_X,) * 4})

    def test_flat_input_is_declared_flat(self):
        spv = support.translate_fixture("ps_flat", dxbc=PS_FLAT_DXBC)
        self.assertIn("OpDecorate %in_attr1 Flat", spv)
        self.assertNotIn("OpDecorate %in_attr0 Flat", spv)

    def test_buffer_num_records_bounds(self):
        # cs_bounds builds a V# with base 0, stride 0, num_records 16 and runs one thread. The output V# in
        # s[0:3] is the same memory. The out-of-bounds store at offset 16 must not land anywhere, and the OOB
        # load must come back as 0.
        spv = support.translate_fixture("cs_bounds", fssm=CS_BOUNDS_FSSM)
        memory = {0x0C: 0xEEEEEEEE}
        outputs = support.run_fragment(
            spv, inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
            push_constants=[0, 0, 16, 0] + [0] * 28, memory=memory)
        self.assertEqual(outputs, {})
        for address, want in EXPECTED_BOUNDS.items():
            with self.subTest(address=address):
                self.assertEqual(memory.get(address), want)
        with self.subTest(what="out-of-bounds store"):
            self.assertNotIn(0x10, memory)


if __name__ == "__main__":
    unittest.main()
