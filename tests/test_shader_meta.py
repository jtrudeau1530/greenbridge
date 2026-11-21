"""Unit tests for tools/shader_meta.py and rdna2spirv.shader_meta() (standard library only, no game data).

    python -I tests/test_shader_meta.py
"""
import gc
import hashlib
import json
import os
import struct
import sys
import tempfile
import unittest
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import rdna2spirv  # noqa: E402
import shader_meta as M  # noqa: E402


def make_dxbc(chunks):
    head = 32 + 4 * len(chunks)
    offsets, body = [], b""
    for fourcc, payload in chunks:
        offsets.append(head + len(body))
        body += fourcc + struct.pack("<I", len(payload)) + payload
    return (b"DXBC" + bytes(16) + struct.pack("<III", 1, head + len(body), len(chunks))
            + b"".join(struct.pack("<I", o) for o in offsets) + body)


def isg1_chunk(elements):
    """Build an ISG1 chunk with real NUL-terminated semantic names.

    elements = [(semantic_name, semantic_index, register, mask, rw_mask, system_value, component_type)]
    """
    cnt = len(elements)
    str_off = 8 + 32 * cnt
    table = bytearray(struct.pack("<II", cnt, 8))
    strings = bytearray()
    for name, sidx, reg, mask, rw_mask, sysv, ctype in elements:
        if name:
            noff = str_off + len(strings)
            strings.extend(name.encode("ascii") + b"\x00")
        else:
            noff = 0
        table.extend(struct.pack("<6I", 0, noff, sidx, sysv, ctype, reg))
        table.extend(bytes([mask, rw_mask, 0, 0]))
        table.extend(struct.pack("<I", 0))
    return b"ISG1", bytes(table + strings)


def sfi0_chunk(flags):
    return b"SFI0", struct.pack("<Q", flags)


def translate_synthetic(name, lines, dxbc=None, fssm=None):
    """Translate a hand-written .s listing in a temp dir and return (spvasm, shader_meta dict)."""
    with tempfile.TemporaryDirectory() as tmp:
        disasm = os.path.join(tmp, "disasm")
        os.mkdir(disasm)
        s_path = os.path.join(disasm, name + ".s")
        header = f"; {name} code 0x100..0x200\n"
        with open(s_path, "w", encoding="utf-8") as f:
            f.write(header + "".join("\t" + ln + "\n" for ln in lines))
        if dxbc is None:
            dxbc = make_dxbc([])
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(dxbc)
        if fssm is not None:
            fdir = os.path.join(tmp, "fssm")
            os.mkdir(fdir)
            with open(os.path.join(fdir, name + ".fssm"), "wb") as f:
                f.write(fssm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            asm = rdna2spirv.translate(s_path)
            meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, asm)
            gc.collect()
        return asm, meta


def vk_pipeline_check_read_index(text):
    """Mirror of runtime/test/vk_pipeline_check.cpp:read_index to verify index.json compatibility."""
    out = {}
    i, depth, key = 0, 0, ""
    n = len(text)
    while i < n:
        c = text[i]
        if c == "{":
            depth += 1
            i += 1
            continue
        if c == "}":
            depth -= 1
            i += 1
            continue
        if c == '"':
            j = text.find('"', i + 1)
            tok = text[i + 1:j]
            i = j + 1
            if depth == 1:
                key = tok
                continue
            if depth == 2 and tok == "stage":
                a = text.find('"', i)
                b = text.find('"', a + 1)
                out.setdefault(key, {})["stage"] = text[a + 1:b]
                i = b + 1
            elif depth == 2 and tok == "subgroup_size":
                a = text.find(":", i)
                if a != -1:
                    a += 1
                    while a < n and text[a] in " \t":
                        a += 1
                    val = 0
                    while a < n and "0" <= text[a] <= "9":
                        val = val * 10 + (ord(text[a]) - 48)
                        a += 1
                    out.setdefault(key, {})["subgroup_size"] = val
                    i = a
            continue
        i += 1
    return out


class DxbcMetaTests(unittest.TestCase):
    def test_isg1_semantics_and_sfi0_flags(self):
        dxbc = make_dxbc([
            isg1_chunk([
                ("POSITION", 0, 0, 0x7, 0x7, 0, 3),
                ("TEXCOORD", 1, 1, 0x3, 0x3, 0, 3),
                ("SV_InstanceID", 0, 2, 0x1, 0x1, 8, 1),
            ]),
            sfi0_chunk(M.SFI0_RESOURCE_DESCRIPTOR_HEAP_INDEXING),
        ])
        got = M.read_dxbc_meta(dxbc)
        self.assertEqual(got["input_signature"], [
            {"semantic": "POSITION", "semantic_index": 0, "register": 0, "mask": 7, "rw_mask": 7,
             "system_value": 0, "component_type": "float"},
            {"semantic": "TEXCOORD", "semantic_index": 1, "register": 1, "mask": 3, "rw_mask": 3,
             "system_value": 0, "component_type": "float"},
            {"semantic": "SV_InstanceID", "semantic_index": 0, "register": 2, "mask": 1, "rw_mask": 1,
             "system_value": 8, "component_type": "uint"},
        ])
        self.assertTrue(got["sfi0_resource_heap_directly_indexed"])
        self.assertFalse(got["sfi0_sampler_heap_directly_indexed"])

    def test_sfi0_both_heaps_and_empty_dxbc(self):
        dxbc = make_dxbc([sfi0_chunk(M.SFI0_RESOURCE_DESCRIPTOR_HEAP_INDEXING | M.SFI0_SAMPLER_DESCRIPTOR_HEAP_INDEXING)])
        got = M.read_dxbc_meta(dxbc)
        self.assertTrue(got["sfi0_resource_heap_directly_indexed"])
        self.assertTrue(got["sfi0_sampler_heap_directly_indexed"])
        self.assertNotIn("input_signature", got)
        self.assertEqual(M.read_dxbc_meta(b""), {})
        self.assertEqual(M.read_dxbc_meta(b"NOT_DXBC" + bytes(32)), {})


class ShaderMetaIntegrationTests(unittest.TestCase):
    def test_ptr64_const32_stride_literal_sampler_and_fssm_hash(self):
        # User SGPRs:
        #   s[0:1] -> 64-bit descriptor-table pointer, offset by (s2 << 5) (32-byte T# stride)
        #   s2     -> 32-bit descriptor index constant
        #   s3     -> 32-bit scalar constant copied to v4
        #   s[12:15] -> literal S# built from immediates (example from docs/design.md)
        lines = [
            "s_lshl_b32 s4, s2, 5",
            "s_add_u32 s6, s0, s4",
            "s_addc_u32 s7, s1, 0",
            "s_load_dwordx8 s[16:23], s[6:7], 0x0",
            "s_mov_b32 s12, 0x92",
            "s_mov_b32 s13, 0x00c00000",
            "s_mov_b32 s14, 0x14500000",
            "s_mov_b32 s15, 0",
            "v_mov_b32_e32 v0, 0.5",
            "v_mov_b32_e32 v1, 0.5",
            "image_sample v[4:7], v[0:1], s[16:23], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "v_add_nc_u32_e32 v4, s3, v4",
            "exp mrt0 v4, v5, v6, v7 done vm",
            "s_endpgm",
        ]
        fssm_bytes = b"FSSM" + b"\x01\x02\x03\x04" * 16
        dxbc = make_dxbc([
            isg1_chunk([("SV_Position", 0, 0, 0xF, 0x3, 1, 3)]),
            sfi0_chunk(M.SFI0_RESOURCE_DESCRIPTOR_HEAP_INDEXING),
        ])
        _, meta = translate_synthetic("ps_meta_test", lines, dxbc=dxbc, fssm=fssm_bytes)

        self.assertEqual(meta["fssm_sha256"], hashlib.sha256(fssm_bytes).hexdigest())
        self.assertEqual(meta["user_sgpr_reads"], [
            {"sgpr": 0, "kind": "ptr64_lo", "pair": [0, 1]},
            {"sgpr": 1, "kind": "ptr64_hi", "pair": [0, 1]},
            {"sgpr": 2, "kind": "const32"},
            {"sgpr": 3, "kind": "const32"},
        ])
        self.assertEqual(meta["descriptor_strides"], [32])
        self.assertEqual(meta["literal_samplers"], [[0x92, 0x00C00000, 0x14500000, 0]])
        self.assertEqual(meta["descriptor_access"], {
            "table_loads": True,
            "truncates_table_ptr_high": False,
            "resource_heap_directly_indexed": True,
            "sampler_heap_directly_indexed": False,
        })
        self.assertEqual(meta["input_signature"][0]["semantic"], "SV_Position")
        self.assertNotIn("uses_code_base", meta)

    def test_truncated_32bit_table_pointer_and_vscale_stride(self):
        # Descriptor-table pointer in s2 with high dword forced to 0 (below 4 GB), stride 16 via VGPR shift.
        lines = [
            "s_mov_b32 s3, 0",
            "v_mov_b32_e32 v2, 1",
            "v_lshlrev_b32_e32 v3, 4, v2",
            "v_readfirstlane_b32 s4, v3",
            "s_load_dwordx4 s[8:11], s[2:3], s4",
            "v_mov_b32_e32 v0, s8",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        _, meta = translate_synthetic("ps_trunc_ptr", lines)
        # s3 was written before any read, so only s2 is read-before-write (and classified ptr32_lo).
        self.assertEqual(meta["user_sgpr_reads"], [{"sgpr": 2, "kind": "ptr32_lo"}])
        self.assertEqual(meta["descriptor_strides"], [16])
        self.assertEqual(meta["descriptor_access"], {
            "table_loads": True,
            "truncates_table_ptr_high": True,
        })

    def test_uses_code_base_without_false_user_ptr(self):
        # s_getpc_b64 + s_load_dwordx4 reads code-embedded data; must set uses_code_base=True and NOT
        # claim any user SGPR was read or that a descriptor table was loaded.
        lines = [
            "s_getpc_b64 s[4:5] ; encoding: [0x04,0x1c,0x84,0xbe]",
            "s_add_u32 s4, s4, 64 ; encoding: [0x04,0xff,0x04,0x80,0x40,0x00,0x00,0x00]",
            "s_addc_u32 s5, s5, 0 ; encoding: [0x05,0x80,0x05,0x82]",
            "s_load_dwordx4 s[8:11], s[4:5], 0x0 ; encoding: [0x02,0x02,0x08,0xf4,0x00,0x00,0x00,0x00]",
            "v_mov_b32_e32 v0, s8 ; encoding: [0x08,0x02,0x00,0x7e]",
            "exp mrt0 v0, v0, v0, v0 done vm ; encoding: [0x0f,0x08,0x00,0xf8,0x00,0x00,0x00,0x00]",
            "s_endpgm ; encoding: [0x00,0x00,0x81,0xbf]",
        ]
        _, meta = translate_synthetic("ps_codebase", lines)
        self.assertTrue(meta["uses_code_base"])
        self.assertNotIn("user_sgpr_reads", meta)
        self.assertNotIn("descriptor_access", meta)

    def test_compute_excludes_workgroup_id_sgprs_and_detects_vsharp(self):
        # RSRC2 0x8088: 4 user SGPRs (s0..s3 = V#), TGID_X enabled in s4 (system SGPR, NOT user data).
        fssm = bytearray(0x148)
        struct.pack_into("<4I", fssm, 0x138, 0x8088, 64, 1, 1)
        lines = [
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_mov_b32_e32 v2, s4",
            "buffer_store_dword v2, v1, s[0:3], 0 offen",
            "s_endpgm",
        ]
        _, meta = translate_synthetic("cs_vsharp", lines, fssm=bytes(fssm))
        self.assertEqual(meta["user_sgprs"], 4)
        # s4 (workgroup id) is excluded; s0..s3 are classified as vsharp.
        self.assertEqual(meta["user_sgpr_reads"], [
            {"sgpr": 0, "kind": "vsharp"},
            {"sgpr": 1, "kind": "vsharp"},
            {"sgpr": 2, "kind": "vsharp"},
            {"sgpr": 3, "kind": "vsharp"},
        ])
        self.assertEqual(meta["fssm_sha256"], hashlib.sha256(bytes(fssm)).hexdigest())

    def test_branch_control_flow_must_write_and_literal_sampler_disagreement(self):
        # Block 0 branches to block 1 (writes s5, s6 and sets s12=0x92) or block 2 (writes s6 only and sets s12=0x11).
        # Block 3 reads s5 (unread on path 0->2 -> MUST appear in user_sgpr_reads), reads s6 (written on BOTH arms ->
        # must NOT appear in user_sgpr_reads), and samples with s[12:15] (s12 differs across arms -> must NOT be
        # emitted as a literal sampler).
        lines = [
            "s_cmp_eq_u32 s0, 0 ; encoding: [0x00,0x80,0x00,0xbf]",
            "s_cbranch_scc1 5 ; encoding: [0x05,0x00,0x85,0xbf]",
            "s_mov_b32 s5, 1 ; encoding: [0x81,0x03,0x85,0xbe]",
            "s_mov_b32 s6, 2 ; encoding: [0x82,0x03,0x86,0xbe]",
            "s_mov_b32 s12, 0x92 ; encoding: [0xff,0x03,0x8c,0xbe,0x92,0x00,0x00,0x00]",
            "s_branch 2 ; encoding: [0x02,0x00,0x82,0xbf]",
            "s_mov_b32 s6, 3 ; encoding: [0x83,0x03,0x86,0xbe]",
            "s_mov_b32 s12, 0x11 ; encoding: [0x91,0x03,0x8c,0xbe]",
            "s_mov_b32 s13, 0 ; encoding: [0x80,0x03,0x8d,0xbe]",
            "s_mov_b32 s14, 0 ; encoding: [0x80,0x03,0x8e,0xbe]",
            "s_mov_b32 s15, 0 ; encoding: [0x80,0x03,0x8f,0xbe]",
            "s_load_dwordx8 s[16:23], s[2:3], 0x0 ; encoding: [0x01,0x04,0x0c,0xf4,0x00,0x00,0x00,0x00]",
            "v_mov_b32_e32 v0, 0.5 ; encoding: [0xf0,0x02,0x00,0x7e]",
            "v_mov_b32_e32 v1, 0.5 ; encoding: [0xf0,0x02,0x02,0x7e]",
            "image_sample v[4:7], v[0:1], s[16:23], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_2D ; encoding: [0x00,0x0f,0x80,0xf0,0x00,0x04,0x64,0x00]",
            "v_add_nc_u32_e32 v4, s5, v4 ; encoding: [0x05,0x08,0x08,0x4a]",
            "v_add_nc_u32_e32 v5, s6, v5 ; encoding: [0x06,0x0a,0x0a,0x4a]",
            "exp mrt0 v4, v5, v6, v7 done vm ; encoding: [0x0f,0x08,0x00,0xf8,0x04,0x05,0x06,0x07]",
            "s_endpgm ; encoding: [0x00,0x00,0x81,0xbf]",
        ]
        _, meta = translate_synthetic("ps_cfg_meta", lines)
        sgprs = {r["sgpr"]: r["kind"] for r in meta["user_sgpr_reads"]}
        self.assertEqual(sgprs, {0: "const32", 2: "ptr64_lo", 3: "ptr64_hi", 5: "const32"})
        self.assertNotIn("literal_samplers", meta)

    def test_global_ptr64_and_mul_stride(self):
        # global_load_dword via user-SGPR pair s[4:5], plus s_mul_i32 48-byte stride into s_load_dwordx8.
        lines = [
            "s_mul_i32 s8, s2, 48",
            "s_load_dwordx8 s[16:23], s[0:1], s8",
            "v_mov_b32_e32 v0, 0",
            "global_load_dword v1, v0, s[4:5]",
            "v_add_nc_u32_e32 v1, s16, v1",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        _, meta = translate_synthetic("ps_global_stride", lines)
        self.assertEqual(meta["user_sgpr_reads"], [
            {"sgpr": 0, "kind": "ptr64_lo", "pair": [0, 1]},
            {"sgpr": 1, "kind": "ptr64_hi", "pair": [0, 1]},
            {"sgpr": 2, "kind": "const32"},
            {"sgpr": 4, "kind": "ptr64_lo", "pair": [4, 5]},
            {"sgpr": 5, "kind": "ptr64_hi", "pair": [4, 5]},
        ])
        self.assertEqual(meta["descriptor_strides"], [48])

    def test_index_json_compatible_with_vk_pipeline_check(self):
        lines = [
            "s_lshl_b32 s4, s2, 5",
            "s_load_dwordx8 s[16:23], s[0:1], s4",
            "s_mov_b32 s12, 0x92",
            "s_mov_b32 s13, 0",
            "s_mov_b32 s14, 0",
            "s_mov_b32 s15, 0",
            "v_mov_b32_e32 v0, 0.5",
            "v_mov_b32_e32 v1, 0.5",
            "image_sample v[4:7], v[0:1], s[16:23], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "exp mrt0 v4, v5, v6, v7 done vm",
            "s_endpgm",
        ]
        dxbc = make_dxbc([
            isg1_chunk([("POSITION", 0, 0, 0x7, 0x7, 0, 3)]),
            sfi0_chunk(M.SFI0_RESOURCE_DESCRIPTOR_HEAP_INDEXING),
        ])
        _, ps_meta = translate_synthetic("ps_one", lines, dxbc=dxbc, fssm=b"FSSM1234")
        fssm_cs = bytearray(0x148)
        struct.pack_into("<4I", fssm_cs, 0x138, 0x8088, 64, 1, 1)
        _, cs_meta = translate_synthetic("cs_two", ["s_mov_b64 exec, -1", "s_endpgm"], fssm=bytes(fssm_cs))

        dumped = json.dumps({"ps_one": ps_meta, "cs_two": cs_meta}, indent=0, sort_keys=True)
        parsed = vk_pipeline_check_read_index(dumped)
        self.assertEqual(parsed, {
            "ps_one": {"stage": "ps", "subgroup_size": 32},
            "cs_two": {"stage": "cs", "subgroup_size": 64},
        })


if __name__ == "__main__":
    unittest.main()
