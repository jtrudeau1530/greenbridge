"""Unit tests for scripts/audit_shader4_regs.py (standard library only, synthetic .s/.fssm/.dxbc)."""
import json
import os
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import audit_shader4_regs as audit_mod  # noqa: E402


def make_dxbc(chunks):
    hdr_size = 32 + 4 * len(chunks)
    offsets, body = [], bytearray()
    for tag, payload in chunks:
        offsets.append(hdr_size + len(body))
        body += tag + struct.pack("<I", len(payload)) + payload
    hdr = bytearray(b"DXBC") + bytes(16) + struct.pack("<III", 1, hdr_size + len(body), len(chunks))
    for off in offsets:
        hdr += struct.pack("<I", off)
    return bytes(hdr) + bytes(body)


def make_ms_psv0(tx=128, ty=1, tz=1, maxv=64, maxp=64, topo=2):
    r = bytearray(48)
    struct.pack_into("<HH", r, 12, maxv, maxp)
    r[24] = 13
    r[27] = topo
    struct.pack_into("<III", r, 36, tx, ty, tz)
    return b"PSV0", struct.pack("<I", len(r)) + bytes(r)


class AuditShader4RegsTests(unittest.TestCase):
    def test_synthetic_cs_ms_gs_audit_and_s3_patterns(self):
        with tempfile.TemporaryDirectory() as tmp:
            disasm = os.path.join(tmp, "disasm")
            fssm = os.path.join(tmp, "fssm")
            os.mkdir(disasm)
            os.mkdir(fssm)

            # Synthetic CS: RSRC2 = 0x8088 (user=4, tgid=100, tg_size=0, tidig=0, scratch=0, lds=512B)
            with open(os.path.join(disasm, "cs_sample.s"), "w", encoding="utf-8") as f:
                f.write("; cs_sample\n"
                        "\ts_lshl_b32 s6, s4, 6 ; encoding: [0x00,0x00,0x00,0x00]\n"
                        "\tv_add_nc_u32_e32 v1, s6, v0 ; encoding: [0x00,0x00,0x00,0x00]\n"
                        "\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]\n")
            with open(os.path.join(fssm, "cs_sample.fssm"), "wb") as f:
                f.write(bytes(0x138) + struct.pack("<4I", 0x8088, 64, 1, 1))

            # Synthetic MS: reads s3 via s_bfe_u32 D, s3, 0x40018 and reads v5
            with open(os.path.join(disasm, "ms_sample.s"), "w", encoding="utf-8") as f:
                f.write("; ms_sample\n"
                        "\ts_bfe_u32 s4, s3, 0x40018 ; encoding: [0x00,0x00,0x00,0x00]\n"
                        "\tv_readfirstlane_b32 s5, v5 ; encoding: [0x00,0x00,0x00,0x00]\n"
                        "\ts_mov_b64 exec, -1 ; encoding: [0x00,0x00,0x00,0x00]\n"
                        "\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]\n")
            with open(os.path.join(tmp, "ms_sample.dxbc"), "wb") as f:
                f.write(make_dxbc([make_ms_psv0(128, 1, 1, 64, 64, 2)]))

            # Synthetic GS: reads v0, v1 and s3
            with open(os.path.join(disasm, "gs_sample.s"), "w", encoding="utf-8") as f:
                f.write("; gs_sample\n"
                        "\ts_bfe_u32 s4, s3, 0x80008 ; encoding: [0x00,0x00,0x00,0x00]\n"
                        "\tv_mov_b32_e32 v2, v0 ; encoding: [0x00,0x00,0x00,0x00]\n"
                        "\tv_mov_b32_e32 v3, v1 ; encoding: [0x00,0x00,0x00,0x00]\n"
                        "\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]\n")

            out_json = os.path.join(tmp, "report.json")
            self.assertEqual(audit_mod.main([disasm, "--json", out_json]), 0)
            with open(out_json, encoding="utf-8") as f:
                rep = json.load(f)

            self.assertEqual(rep["cs"]["count"], 1)
            self.assertEqual(rep["cs"]["rbw_sgpr_counts"], {"s4": 1})
            self.assertEqual(rep["cs"]["rbw_vgpr_counts"], {"v0": 1})
            self.assertIn("user=4 tgid=100 tg_size=0 tidig=0 scratch=0 trap=0 lds=512B", rep["cs"]["rsrc2_combos"])

            self.assertEqual(rep["ms"]["count"], 1)
            self.assertEqual(rep["ms"]["wave64_heuristic"], 1)
            self.assertEqual(rep["ms"]["rbw_sgpr_counts"], {"s3": 1})
            self.assertEqual(rep["ms"]["rbw_vgpr_counts"], {"v5": 1})
            self.assertEqual(rep["ms"]["s3_rbw_op_patterns"], {"s_bfe_u32 D, s3, 0x40018": 1})
            self.assertIn("stage=13 topo=2 threads=128x1x1 maxv=64 maxp=64 payload=0B shared=0B",
                          rep["ms"]["psv0_combos"])

            self.assertEqual(rep["gs"]["count"], 1)
            self.assertEqual(rep["gs"]["rbw_vgpr_counts"], {"v0": 1, "v1": 1})
            self.assertEqual(rep["gs"]["s3_rbw_op_patterns"], {"s_bfe_u32 D, s3, 0x80008": 1})


if __name__ == "__main__":
    unittest.main()
