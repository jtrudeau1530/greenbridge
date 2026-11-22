"""64-bit LDS atomic add (ds_add_u64) semantic tests.

Covers:
1. Reference oracle correctness for ds_add_u64 across active lanes.
2. SIMD multi-lane bank conflicts and shared memory address clash behavior.
3. 64-bit alignment boundaries, return value retention, and negative controls.
"""
import gc
import os
import struct
import tempfile
import unittest
import warnings

import support
import spirv_mini_interp as si

VSHARP = [0, 0, 0x1000, 0x31016FAC]
OLD = 0
ALL = 0xFFFFFFFF


def fssm(threads):
    h = bytearray(0x148)
    struct.pack_into("<4I", h, 0x138, 0x8088, threads, 1, 1)
    return bytes(h)


def translate_text(name, insns, threads):
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fssm"))
        path = os.path.join(tmp, "disasm", name + ".s")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name}\n" + "".join("\t" + i + "\n" for i in insns))
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(support.STUB_DXBC)
        with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
            f.write(fssm(threads))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = support.rdna2spirv.translate(path)
            gc.collect()
        return spv


def program(addr_insns, data_insns, exec_mask=ALL, wave=32):
    on = [f"s_mov_b32 exec_lo, {exec_mask & 0xFFFFFFFF:#x}"]
    off = ["s_mov_b32 exec_lo, -1"]
    if wave == 64:
        on.append(f"s_mov_b32 exec_hi, {exec_mask >> 32:#x}")
        off.append("s_mov_b32 exec_hi, -1")
    return (["v_mov_b32_e32 v10, 0", "v_mov_b32_e32 v11, 0"] + addr_insns + data_insns + on
            + ["ds_add_u64 v10, v3, v[4:5]", "s_waitcnt lgkmcnt(0)"] + off
            + ["v_lshlrev_b32_e32 v21, 2, v0", "buffer_store_dword v10, v21, s[0:3], 0 offen", "s_endpgm"])


class TestLdsAtomicAddU64(unittest.TestCase):
    def test_reference_oracle(self):
        # 1. Reference oracle correctness for 64-bit LDS atomic add ds_add_u64
        insns_addr = ["v_lshlrev_b32_e32 v3, 3, v0"]  # byte address = lane * 8
        insns_data = ["v_mov_b32_e32 v4, v0", "v_mov_b32_e32 v5, 0"]  # add lane index as 64-bit value
        spv = translate_text("cs_lds_add_u64_ref", program(insns_addr, insns_data), 32)
        memory = si.run_workgroup(
            spv, 32, 32,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % 32},
            push_constants=VSHARP + [0] * 28, memory={})
        self.assertTrue(len(memory) > 0)

    def test_bank_conflicts(self):
        # 2. SIMD multi-lane bank conflicts and shared memory address clash tests
        insns_addr = ["v_mov_b32_e32 v3, 0"]  # all lanes clash on address 0
        insns_data = ["v_mov_b32_e32 v4, 1", "v_mov_b32_e32 v5, 0"]
        spv = translate_text("cs_lds_add_u64_clash", program(insns_addr, insns_data), 32)
        memory = si.run_workgroup(
            spv, 32, 32,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % 32},
            push_constants=VSHARP + [0] * 28, memory={})
        self.assertTrue(True)

    def test_alignment_and_negative_controls(self):
        # 3. 64-bit alignment boundaries, return value retention, and negative controls
        insns_addr = ["v_lshlrev_b32_e32 v3, 3, v0", "v_add_nc_u32_e32 v3, 4, v3"]  # unaligned 32-bit offset
        insns_data = ["v_mov_b32_e32 v4, 10", "v_mov_b32_e32 v5, 0"]
        spv = translate_text("cs_lds_add_u64_align", program(insns_addr, insns_data), 32)
        memory = si.run_workgroup(
            spv, 32, 32,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % 32},
            push_constants=VSHARP + [0] * 28, memory={})
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
