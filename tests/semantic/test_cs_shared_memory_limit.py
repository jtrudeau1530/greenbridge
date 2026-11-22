"""SHADER-309: a compute shader's Workgroup memory fits in maxComputeSharedMemorySize (#1721).

`--wave64-emulate` adds its own Workgroup buffer, `%x_buf` (2L + 6W + 7 dwords for L invocations in W waves), on top
of the shader's LDS array. With a 64 KiB LDS that made modules over the 64 KiB device limit, which `spirv-val` accepts
and RADV dies on (#1721: 16,385 uints = 65,540 B). The translator now counts both against
`rdna2spirv.MAX_SHARED_BYTES` (`--max-shared-memory`, default 65536, the hard cap): the LDS array, a power of two at
least the program's need, shrinks to the room left when the need still fits (indices wrap with OpUMod instead of a
mask); otherwise the shader is refused (`Unsupported`, with the byte counts). The device's own limit (32768 on the
AMD Windows and Intel drivers) is the runtime's check, against `lds_bytes`, which must be the whole total.

The Workgroup bytes are counted from the emitted SPIR-V. The programs are written here; no game data.
"""
import os
import re
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # python -I leaves the script's folder off sys.path

import support  # noqa: E402  (support puts tools/ on sys.path, honouring LEONIDA_TOOLS_DIR)

rdna2spirv = support.rdna2spirv

# One wave of 64: write each lane's id to LDS, read the neighbour's dword back, store it to the output V# s[0:3].
PROGRAM = ["s_mov_b64 exec, -1",
           "v_lshlrev_b32_e32 v1, 2, v0",
           "v_readfirstlane_b32 s16, v1",
           "ds_write_b32 v1, v0",
           "s_waitcnt lgkmcnt(0)",
           "ds_read_b32 v2, v1 offset:4",
           "s_waitcnt lgkmcnt(0)",
           "buffer_store_dword v2, v1, s[0:3], 0 offen",
           "s_endpgm"]
X_BUF = 4 * (2 * 64 + 6 * 1 + 7)          # %x_buf for one wave of 64: 564 bytes


def fssm(lds_units, threads=64):
    """Compute header: COMPUTE_PGM_RSRC2 with 4 user SGPRs, TGID_X and LDS_SIZE (512-byte units) at [23:15]."""
    return bytes(0x138) + struct.pack("<4I", 0x88 | (lds_units << 15), threads, 1, 1)


def translate(lds_units, emulate=True, limit=None):
    prev = rdna2spirv.WAVE64_EMULATE, rdna2spirv.MAX_SHARED_BYTES
    rdna2spirv.WAVE64_EMULATE = emulate
    if limit is not None:
        rdna2spirv.MAX_SHARED_BYTES = limit
    try:
        spv = support.translate_lines("cs_shared_limit", PROGRAM, fssm=fssm(lds_units))
    finally:
        rdna2spirv.WAVE64_EMULATE, rdna2spirv.MAX_SHARED_BYTES = prev
    assert rdna2spirv.translate.last_gen.wave64, "the test program must translate as wave64"
    return spv


def workgroup_bytes(spv):
    """{variable: bytes} of every Workgroup-storage-class OpVariable, sized from its pointee type."""
    consts, types, ptrs, out = {}, {}, {}, {}
    for ln in spv.splitlines():
        t = ln.split()
        if len(t) < 3 or t[1] != "=":
            continue
        rid, op, args = t[0], t[2], t[3:]
        if op == "OpConstant":
            consts[rid] = int(args[1], 0)
        elif op == "OpTypeInt" or op == "OpTypeFloat":
            types[rid] = int(args[0]) // 8
        elif op == "OpTypeArray":
            types[rid] = types[args[0]] * consts[args[1]]
        elif op == "OpTypeStruct":
            types[rid] = sum(types[a] for a in args)
        elif op == "OpTypePointer" and args[0] == "Workgroup":
            ptrs[rid] = args[1]
        elif op == "OpVariable" and args[1:2] == ["Workgroup"]:
            out[rid] = types[ptrs[args[0]]]
    return out


def lds_index_ops(spv):
    """The op each LDS access wraps its dword index with (OpBitwiseAnd for a power of two, OpUMod when shrunk)."""
    return set(re.findall(r"= (OpBitwiseAnd|OpUMod) %uint %\w+ %\w+\n\s*%\w+ = OpAccessChain %_ptr_WG_uint\w* %lds ",
                          spv))


class SharedMemoryLimitTests(unittest.TestCase):
    def test_under_limit_is_unchanged(self):
        # 32 KiB of LDS + the exchange: well under 64 KiB, the power-of-two array as before
        wg = workgroup_bytes(translate(64))
        self.assertEqual(wg, {"%lds": 32768, "%x_buf": X_BUF})
        self.assertLessEqual(sum(wg.values()), rdna2spirv.MAX_SHARED_BYTES)

    def test_full_64k_lds_is_refused(self):
        # #1721: 64 KiB of LDS leaves no room for the exchange, and the program needs all of it
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            translate(128)
        msg = str(cm.exception)
        self.assertIn(f"{65536 + X_BUF} B", msg)
        self.assertIn("LDS 65536 B", msg)
        self.assertIn(f"wave64 exchange {X_BUF} B", msg)
        self.assertIn("65536 B", msg)

    def test_rounded_up_lds_shrinks_to_exactly_the_limit(self):
        # LDS_SIZE 100 units = 51,200 B rounds up to a 64 KiB array; the need fits, so the array takes what is left
        spv = translate(100)
        wg = workgroup_bytes(spv)
        self.assertEqual(wg["%x_buf"], X_BUF)
        self.assertEqual(sum(wg.values()), 65536)            # exactly at the limit (65536 - 564 is a dword multiple)
        self.assertGreaterEqual(wg["%lds"], 51200)
        self.assertEqual(lds_index_ops(spv), {"OpUMod"})

    def test_exact_limit_is_accepted_one_dword_less_is_refused(self):
        # 32 KiB of LDS (LDS_SIZE 64, all of it needed) + the exchange, against a limit at exactly that sum ...
        exact = 32768 + X_BUF
        spv = translate(64, limit=exact)
        self.assertEqual(sum(workgroup_bytes(spv).values()), exact)
        self.assertEqual(lds_index_ops(spv), {"OpBitwiseAnd"})
        # ... and one dword under it: the need no longer fits, so there is nothing to shrink to
        with self.assertRaises(rdna2spirv.Unsupported):
            translate(64, limit=exact - 4)

    def test_metadata_lds_bytes_is_every_workgroup_byte(self):
        # index.json's lds_bytes is what the runtime compares with the device's maxComputeSharedMemorySize (32768 on
        # some drivers): it has to be the module's whole Workgroup total, the emulation's %x_buf included
        for units, emulate in ((64, True), (100, True), (128, False), (64, False)):
            with self.subTest(units=units, emulate=emulate):
                prev = rdna2spirv.WAVE64_EMULATE
                spv = translate(units, emulate=emulate)
                rdna2spirv.WAVE64_EMULATE = emulate
                try:
                    meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)
                finally:
                    rdna2spirv.WAVE64_EMULATE = prev
                self.assertEqual(meta["lds_bytes"], sum(workgroup_bytes(spv).values()))

    def test_native_wave64_full_64k_lds_still_translates(self):
        # without the flag there is no %x_buf: a 64 KiB LDS is exactly the limit
        wg = workgroup_bytes(translate(128, emulate=False))
        self.assertEqual(wg, {"%lds": 65536})


if __name__ == "__main__":
    unittest.main()
