"""Value tests for RDNA2 8/16-bit global stores and d16 loads, run through spirv_mini_interp on the CPU.

global_store_byte/short write the low 8/16 bits of the data VGPR (the _d16_hi forms its high half) and leave the
neighbouring bytes alone. global_load_{u,s}byte_d16 and global_load_short_d16 write the low 16 bits of vdst (the _hi
forms the high 16) and keep the other half; ubyte zero-extends and sbyte sign-extends the byte to 16 bits. The address
is the global_* one. Every byte offset in a dword is tried, so a 16-bit access at byte 3 crosses into the next dword.
Standard library only; no GPU, no game files.
"""
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
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)
FSSM = bytes(FSSM)
BASE, OUT = 0x2000, 0x3000
SEED = [0x44332211, 0x88776655, 0xCCBBAA99]       # the bytes at BASE .. BASE + 11 are 11 22 33 ... CC


def translate(body):
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


def run(body):
    mem = {BASE + 4 * k: w for k, w in enumerate(SEED)}
    si.run_workgroup(translate(body), 1, 32,
                     inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                     push_constants=[0] * 36, memory=mem)
    return mem


def seed_bytes():
    return bytearray(b"".join(struct.pack("<I", w) for w in SEED))


def words(buf):
    return [struct.unpack_from("<I", buf, 4 * k)[0] for k in range(len(SEED))]


STORES = {"global_store_byte": (1, 0), "global_store_byte_d16_hi": (1, 16),
          "global_store_short": (2, 0), "global_store_short_d16_hi": (2, 16)}
LOADS = {"global_load_ubyte_d16": (1, False, 0), "global_load_ubyte_d16_hi": (1, False, 16),
         "global_load_sbyte_d16": (1, True, 0), "global_load_sbyte_d16_hi": (1, True, 16),
         "global_load_short_d16": (2, False, 0), "global_load_short_d16_hi": (2, False, 16)}


class GlobalSubDword(unittest.TestCase):
    def test_stores_keep_neighbouring_bytes(self):
        data = 0xF1E2D3C4
        for op, (n, shift) in STORES.items():
            for byte in range(1, 8):                # byte 3 and 7: a short crosses into the next dword
                with self.subTest(op=op, byte=byte):
                    mem = run([f"v_mov_b32_e32 v1, {BASE + byte:#x}", f"v_mov_b32_e32 v2, {data:#x}",
                               f"{op} v1, v2, s[0:1]"])
                    want = seed_bytes()
                    want[byte:byte + n] = ((data >> shift) & ((1 << 8 * n) - 1)).to_bytes(n, "little")
                    self.assertEqual([mem[BASE + 4 * k] for k in range(3)], words(want))

    def test_d16_loads_write_one_half(self):
        keep = 0xBEEF1234
        buf = seed_bytes()
        for op, (n, signed, shift) in LOADS.items():
            for byte in range(0, 8):
                with self.subTest(op=op, byte=byte):
                    mem = run([f"v_mov_b32_e32 v1, {BASE + byte:#x}", f"v_mov_b32_e32 v5, {keep:#x}",
                               f"{op} v5, v1, s[0:1]", "s_waitcnt vmcnt(0)", f"v_mov_b32_e32 v6, {OUT:#x}",
                               "global_store_dword v6, v5, s[0:1]"])
                    v = int.from_bytes(buf[byte:byte + n], "little")
                    if signed and v >> 7:
                        v -= 0x100
                    v &= 0xFFFF
                    want = (keep & ~(0xFFFF << shift) | v << shift) & M32
                    self.assertEqual(mem[OUT], want)

    def test_inactive_lane_does_nothing(self):
        # Two lanes run; only lane 1 is active, so the byte gets lane 1's data.
        body = ["v_cmpx_eq_u32_e32 1, v0", f"v_mov_b32_e32 v1, {BASE + 1:#x}", "v_add_nc_u32_e32 v2, 0x70, v0",
                "global_store_byte v1, v2, s[0:1]"]
        mem = {BASE + 4 * k: w for k, w in enumerate(SEED)}
        si.run_workgroup(translate(body), 2, 32,
                         inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                         push_constants=[0] * 36, memory=mem)
        self.assertEqual(mem[BASE], (SEED[0] & ~0xFF00) | 0x71 << 8)

    def test_rejected_forms(self):
        for line in ("global_load_dword_addtid v0, s[0:1]", "global_store_byte v1, v[2:3], s[0:1]",
                     "global_store_short v1, v2, s[0:1] slc"):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(["v_mov_b32_e32 v1, 0", line])

    def test_first_read_analysis_counts_the_kept_half(self):
        from shader_ir import parse_line
        self.assertEqual(rdna2spirv.read_before_write_vgprs([parse_line("\tglobal_load_short_d16 v5, v1, s[0:1]")]),
                         {1, 5})
        self.assertEqual(rdna2spirv.read_before_write_vgprs([parse_line("\tglobal_load_ushort v5, v1, s[0:1]")]),
                         {1})


if __name__ == "__main__":
    unittest.main()
