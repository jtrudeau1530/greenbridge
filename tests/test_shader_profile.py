"""Unit tests for tools/shader_profile.py.

Validates instruction classification (ALU vs memory vs branch), operation ratio counting,
register footprint tracking (VGPR, SGPR, ranges, special registers), and CLI corpus profiling.
Built entirely with synthetic disassembly text; no game data or proprietary assets.

Run:
    python -I tests/test_shader_profile.py
    python -m unittest discover -s tests -v
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import shader_profile as P  # noqa: E402


class TestFamilyClassification(unittest.TestCase):
    def test_known_families(self):
        cases = [
            ("s_buffer_load_dword", "smem"),
            ("s_load_dwordx4", "smem"),
            ("s_mov_b32", "scalar"),
            ("s_add_u32", "scalar"),
            ("v_mov_b32", "vector"),
            ("v_fma_f32", "vector"),
            ("ds_read_b32", "lds"),
            ("ds_write_b32", "lds"),
            ("buffer_load_dword", "buffer"),
            ("tbuffer_load_format_x", "buffer"),
            ("image_sample", "image"),
            ("image_load", "image"),
            ("global_load_dword", "global"),
            ("global_store_dword", "global"),
            ("flat_load_dword", "flat"),
            ("flat_store_dword", "flat"),
            ("scratch_load_dword", "scratch"),
            ("scratch_store_dword", "scratch"),
            ("exp", "export"),
            ("unknown_instruction", "other"),
        ]
        for mnemonic, expected_family in cases:
            with self.subTest(mnemonic=mnemonic):
                self.assertEqual(P.family(mnemonic), expected_family)


class TestInstructionClassification(unittest.TestCase):
    def test_alu_instructions(self):
        alu_cases = [
            "v_mov_b32", "v_add_f32", "v_sub_f32", "v_mul_f32", "v_fma_f32",
            "v_cmp_gt_f32", "v_cndmask_b32", "v_cvt_f32_u32", "v_and_b32",
            "v_lshlrev_b32", "v_med3_f32", "v_bfi_b32", "v_bfe_u32",
            "s_mov_b32", "s_add_u32", "s_sub_u32", "s_and_b64", "s_or_b64",
            "s_xor_b64", "s_lshl_b64", "s_cmp_eq_u32", "s_cselect_b32",
        ]
        for m in alu_cases:
            with self.subTest(mnemonic=m):
                self.assertEqual(P.classify_instruction(m), "alu")
                self.assertTrue(P.is_alu(m))
                self.assertFalse(P.is_memory(m))
                self.assertFalse(P.is_branch(m))

    def test_memory_instructions(self):
        mem_cases = [
            "s_load_dword", "s_load_dwordx4", "s_buffer_load_dword",
            "ds_read_b32", "ds_write_b32", "buffer_load_dword",
            "buffer_store_dword", "tbuffer_load_format_x",
            "tbuffer_store_format_xyzw", "image_sample", "image_sample_lz",
            "image_load", "image_store", "global_load_dword",
            "global_store_dword", "flat_load_dword", "flat_store_dword",
            "scratch_load_dword", "scratch_store_dword",
        ]
        for m in mem_cases:
            with self.subTest(mnemonic=m):
                self.assertEqual(P.classify_instruction(m), "memory")
                self.assertTrue(P.is_memory(m))
                self.assertFalse(P.is_alu(m))
                self.assertFalse(P.is_branch(m))

    def test_branch_instructions(self):
        branch_cases = [
            "s_branch", "s_cbranch_scc0", "s_cbranch_scc1",
            "s_cbranch_vccz", "s_cbranch_vccnz", "s_cbranch_execz",
            "s_cbranch_execnz", "s_setpc_b64", "s_swappc_b64",
            "s_rfe_b64", "s_call_b64",
        ]
        for m in branch_cases:
            with self.subTest(mnemonic=m):
                self.assertEqual(P.classify_instruction(m), "branch")
                self.assertTrue(P.is_branch(m))
                self.assertFalse(P.is_alu(m))
                self.assertFalse(P.is_memory(m))

    def test_export_and_other_instructions(self):
        self.assertEqual(P.classify_instruction("exp"), "export")
        self.assertEqual(P.classify_instruction("s_nop"), "other")
        self.assertEqual(P.classify_instruction("s_endpgm"), "other")
        self.assertEqual(P.classify_instruction("s_barrier"), "other")
        self.assertEqual(P.classify_instruction("s_waitcnt"), "other")
        self.assertEqual(P.classify_instruction("s_waitcnt_vscnt"), "other")
        self.assertEqual(P.classify_instruction("s_sendmsg"), "other")
        self.assertEqual(P.classify_instruction(""), "other")
        self.assertEqual(P.classify_instruction("nonexistent_opcode"), "other")


class TestRatiosAndCategoryCounting(unittest.TestCase):
    def test_count_categories_from_list(self):
        mnemonics = [
            "v_mov_b32", "v_add_f32", "s_load_dword", "s_cbranch_scc0",
            "exp", "s_nop", "s_endpgm"
        ]
        counts = P.count_categories(mnemonics)
        self.assertEqual(counts["alu"], 2)
        self.assertEqual(counts["memory"], 1)
        self.assertEqual(counts["branch"], 1)
        self.assertEqual(counts["export"], 1)
        self.assertEqual(counts["other"], 2)
        self.assertEqual(counts["total"], 7)

    def test_count_categories_from_dict(self):
        mnemonics = {
            "v_add_f32": 10,
            "v_mul_f32": 5,
            "buffer_load_dword": 5,
            "s_branch": 2,
        }
        counts = P.count_categories(mnemonics)
        self.assertEqual(counts["alu"], 15)
        self.assertEqual(counts["memory"], 5)
        self.assertEqual(counts["branch"], 2)
        self.assertEqual(counts["export"], 0)
        self.assertEqual(counts["other"], 0)
        self.assertEqual(counts["total"], 22)

    def test_compute_ratios_balanced(self):
        counts = {"alu": 50, "memory": 25, "branch": 25, "export": 0, "other": 0, "total": 100}
        r = P.compute_ratios(counts)
        self.assertEqual(r["alu_ratio"], 0.50)
        self.assertEqual(r["memory_ratio"], 0.25)
        self.assertEqual(r["branch_ratio"], 0.25)
        self.assertEqual(r["alu_to_memory"], 2.0)

    def test_compute_ratios_empty(self):
        counts = {"alu": 0, "memory": 0, "branch": 0, "export": 0, "other": 0, "total": 0}
        r = P.compute_ratios(counts)
        self.assertEqual(r["alu_ratio"], 0.0)
        self.assertEqual(r["memory_ratio"], 0.0)
        self.assertEqual(r["branch_ratio"], 0.0)
        self.assertEqual(r["alu_to_memory"], 0.0)

    def test_compute_ratios_zero_memory(self):
        counts = {"alu": 10, "memory": 0, "branch": 2, "export": 0, "other": 0, "total": 12}
        r = P.compute_ratios(counts)
        self.assertEqual(r["alu_to_memory"], float("inf"))


class TestRegisterFootprint(unittest.TestCase):
    def test_single_registers(self):
        line = "\tv_add_f32 v5, v2, v0"
        regs = P.parse_registers(line)
        self.assertEqual(regs["vgprs"], {0, 2, 5})
        self.assertEqual(regs["sgprs"], set())
        self.assertEqual(regs["special"], set())

    def test_register_ranges(self):
        line = "\ts_load_dwordx4 s[4:7], s[0:1], 0x0"
        regs = P.parse_registers(line)
        self.assertEqual(regs["sgprs"], {0, 1, 4, 5, 6, 7})
        self.assertEqual(regs["vgprs"], set())

    def test_mixed_vgpr_and_sgpr(self):
        line = "\timage_sample v[0:3], v[8:9], s[16:23], s[24:27] dmask:0xf"
        regs = P.parse_registers(line)
        self.assertEqual(regs["vgprs"], {0, 1, 2, 3, 8, 9})
        self.assertEqual(regs["sgprs"], {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27})

    def test_special_registers(self):
        line = "\tv_cndmask_b32 v0, v1, v2, vcc"
        regs = P.parse_registers(line)
        self.assertIn("vcc", regs["special"])
        self.assertEqual(regs["vgprs"], {0, 1, 2})

        line2 = "\ts_mov_b64 exec, s[0:1]"
        regs2 = P.parse_registers(line2)
        self.assertIn("exec", regs2["special"])
        self.assertEqual(regs2["sgprs"], {0, 1})

    def test_comments_stripped_for_registers(self):
        line = "\tv_mov_b32 v1, 1.0 ; encoding: [0x00,0x00,0x80,0xbf] v255 s99"
        regs = P.parse_registers(line)
        self.assertEqual(regs["vgprs"], {1})
        self.assertEqual(regs["sgprs"], set())

    def test_footprint_calculation(self):
        shader = (
            "; synthetic pixel shader\n"
            "\ts_load_dwordx2 s[0:1], s[2:3], 0x10\n"
            "\tv_mov_b32 v0, 1.0\n"
            "\tv_add_f32 v31, v0, v0\n"
            "\ts_cbranch_scc0 0x10\n"
            "\ts_endpgm\n"
        )
        fp = P.register_footprint(shader)
        self.assertEqual(fp["max_vgpr"], 31)
        self.assertEqual(fp["vgpr_count"], 32)
        self.assertEqual(fp["unique_vgprs"], 2)
        self.assertEqual(fp["vgprs"], [0, 31])

        self.assertEqual(fp["max_sgpr"], 3)
        self.assertEqual(fp["sgpr_count"], 4)
        self.assertEqual(fp["unique_sgprs"], 4)
        self.assertEqual(fp["sgprs"], [0, 1, 2, 3])

    def test_footprint_empty(self):
        fp = P.register_footprint([])
        self.assertEqual(fp["max_vgpr"], -1)
        self.assertEqual(fp["max_sgpr"], -1)
        self.assertEqual(fp["vgpr_count"], 0)
        self.assertEqual(fp["sgpr_count"], 0)
        self.assertEqual(fp["vgprs"], [])
        self.assertEqual(fp["sgprs"], [])


class TestProfileShader(unittest.TestCase):
    def test_profile_shader_synthetic(self):
        text = (
            "; ps_sample code 0x0..0x40\n"
            "\ts_load_dwordx4 s[0:3], s[4:5], 0x0\n"
            "\tv_mov_b32 v0, 1.0\n"
            "\tv_add_f32 v1, v0, v0\n"
            "\tv_mul_f32 v2, v1, v0\n"
            "\ts_cbranch_scc0 0x20\n"
            "\texp mrt0 v0, v1, v2, v2 done vm\n"
            "\ts_endpgm\n"
        )
        prof = P.profile_shader(text)
        self.assertEqual(prof["total_instructions"], 7)
        cats = prof["categories"]
        self.assertEqual(cats["alu"], 3)     # v_mov_b32, v_add_f32, v_mul_f32
        self.assertEqual(cats["memory"], 1)  # s_load_dwordx4
        self.assertEqual(cats["branch"], 1)  # s_cbranch_scc0
        self.assertEqual(cats["export"], 1)  # exp
        self.assertEqual(cats["other"], 1)   # s_endpgm

        ratios = prof["ratios"]
        self.assertAlmostEqual(ratios["alu_ratio"], 3 / 7, places=5)
        self.assertAlmostEqual(ratios["memory_ratio"], 1 / 7, places=5)
        self.assertAlmostEqual(ratios["branch_ratio"], 1 / 7, places=5)
        self.assertEqual(ratios["alu_to_memory"], 3.0)

        regs = prof["registers"]
        self.assertEqual(regs["max_vgpr"], 2)
        self.assertEqual(regs["vgpr_count"], 3)
        self.assertEqual(regs["max_sgpr"], 5)
        self.assertEqual(regs["sgpr_count"], 6)


class TestCommandLineEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.disasm_dir = os.path.join(self.tmp.name, "disasm")
        os.makedirs(self.disasm_dir, exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def run_profile(self, extra=()):
        out_json = os.path.join(self.tmp.name, "custom_profile.json")
        cmd = [sys.executable, "-I", os.path.join(ROOT, "tools", "shader_profile.py"), self.disasm_dir, "--out", out_json]
        res = subprocess.run(cmd + list(extra), capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"CLI failed: {res.stderr}")
        with open(out_json, encoding="utf-8") as f:
            return res.stdout, json.load(f, parse_constant=self.fail)

    def test_cli_alu_without_memory_writes_null(self):
        with open(os.path.join(self.disasm_dir, "ps_001.s"), "w", encoding="utf-8") as f:
            f.write("\tv_mov_b32 v0, 1.0\n\ts_endpgm\n")
        _, data = self.run_profile()
        self.assertEqual(data["categories"]["alu"], 1)
        self.assertEqual(data["categories"]["memory"], 0)
        self.assertIsNone(data["ratios"]["alu_to_memory"])

    def test_cli_empty_corpus(self):
        _, data = self.run_profile()
        self.assertEqual(data["shaders"], 0)
        self.assertEqual(data["ratios"]["alu_to_memory"], 0.0)

    def test_cli_execution_and_json_output(self):
        s1 = (
            "; ps_001 code 0x0..0x20\n"
            "\tv_mov_b32 v0, 1.0\n"
            "\tv_add_f32 v1, v0, v0\n"
            "\ts_load_dword s0, s[2:3], 0x0\n"
            "\ts_endpgm\n"
        )
        s2 = (
            "; vs_002 code 0x0..0x20\n"
            "\ts_branch 0x10\n"
            "\texp pos0 v0, v1, v2, v3 done vm\n"
            "\ts_nop 0\n"
            "; <stdin>:5:1: warning: invalid instruction encoding\n"
            "; 0x12 0x34 0x56 0x78\n"
        )
        with open(os.path.join(self.disasm_dir, "ps_001.s"), "w", encoding="utf-8") as f:
            f.write(s1)
        with open(os.path.join(self.disasm_dir, "vs_002.s"), "w", encoding="utf-8") as f:
            f.write(s2)

        stdout, data = self.run_profile()
        self.assertIn("2 shaders", stdout)
        self.assertIn("ALU:", stdout)
        self.assertIn("Memory:", stdout)
        self.assertIn("Branch:", stdout)

        self.assertEqual(data["shaders"], 2)
        self.assertIn("families", data)
        self.assertIn("categories", data)
        self.assertIn("ratios", data)
        self.assertIn("by_mnemonic", data)

        cats = data["categories"]
        self.assertEqual(cats["alu"], 2)
        self.assertEqual(cats["memory"], 1)
        self.assertEqual(cats["branch"], 1)
        self.assertEqual(cats["export"], 1)
        self.assertEqual(cats["other"], 2)

        ratios = data["ratios"]
        self.assertGreater(ratios["alu_ratio"], 0.0)
        self.assertGreater(ratios["memory_ratio"], 0.0)
        self.assertEqual(ratios["alu_to_memory"], 2.0)

        # Register pressure per stage: ps_001 uses v0-v1 and s0-s3, vs_002 uses v0-v3 and no SGPR.
        regs = data["registers"]
        self.assertEqual(regs["ps"], {"shaders": 1, "vgpr_max": 2, "vgpr_mean": 2.0, "sgpr_max": 4,
                                      "sgpr_mean": 4.0, "vgpr_hist": {"32": 1}})
        self.assertEqual(regs["vs"]["vgpr_max"], 4)
        self.assertEqual(regs["vs"]["sgpr_max"], 0)
        self.assertIn("ps registers: VGPR max 2", stdout)
        self.assertNotIn("per_shader", data)

        _, data = self.run_profile(["--per-shader"])
        self.assertEqual(data["per_shader"], [
            {"name": "ps_001", "stage": "ps", "instructions": 4, "vgpr_count": 2, "sgpr_count": 4},
            {"name": "vs_002", "stage": "vs", "instructions": 3, "vgpr_count": 4, "sgpr_count": 0},
        ])

    def test_register_summary_histogram(self):
        summary = P.register_summary({"cs": [(10, 4), (40, 8), (100, 20), (300, 2)]})
        self.assertEqual(summary["cs"]["vgpr_hist"], {"32": 1, "64": 1, "128": 1, ">256": 1})
        self.assertEqual(summary["cs"]["vgpr_max"], 300)
        self.assertEqual(summary["cs"]["sgpr_mean"], 8.5)
        self.assertEqual(P.register_summary({}), {})


if __name__ == "__main__":
    unittest.main()
