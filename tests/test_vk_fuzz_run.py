"""tools/vk_fuzz_run.py (the compute-program wrapper behind the --oracle hw,vk modes of the differential fuzzers) and the
--oracle vk guard of each fuzzer that has the mode. Standard library only; no GPU, no llvm-mc.

    python -I tests/test_vk_fuzz_run.py
"""
import subprocess
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
sys.dont_write_bytecode = True

import vk_fuzz_run as R  # noqa: E402


class ProgramText(unittest.TestCase):
    def test_wave32_shell(self):
        text = R.program_text("cs_x", ["v_mov_b32 v1, 5"], 4, 32).splitlines()
        self.assertEqual(text[:5], ["; cs_x", "; group: 32,1,1", "; rsrc2: 0x8088", "s_mov_b32 exec_lo, -1",
                                    "v_mov_b32 v20, v0"])
        self.assertIn("v_lshlrev_b32 v21, 4, v20", text)
        self.assertEqual(text[-1], "s_endpgm")
        self.assertEqual(sum(ln.startswith("buffer_store_dword") for ln in text), 4)

    def test_store_stride_follows_the_word_count(self):
        for words, shift in ((1, 2), (2, 3), (4, 4), (8, 5)):
            self.assertIn(f"v_lshlrev_b32 v21, {shift}, v20", R.program_text("c", [], words, 32).splitlines())

    def test_wave64_rewrites_masks_and_vcc_data(self):
        w = R.wave64_text
        self.assertEqual(w("s_mov_b32 vcc_lo, 1"), "s_mov_b64 vcc, 1")
        self.assertEqual(w("s_mov_b32 vcc_lo, 0"), "s_mov_b64 vcc, 0")
        self.assertEqual(w("v_cndmask_b32_e64 v11, 0, 1, vcc_lo"), "v_cndmask_b32_e64 v11, 0, 1, vcc")
        self.assertEqual(w("v_cmp_lt_u32_e64 s20, v1, v2"), "v_cmp_lt_u32_e64 s[20:21], v1, v2")
        self.assertEqual(w("v_cndmask_b32_e64 v1, 0, 1, s21"), "v_cndmask_b32_e64 v1, 0, 1, s[22:23]")
        self.assertEqual(w("s_mov_b32 s4, 7"), "s_mov_b32 s4, 7")

    def test_rewrite_false_keeps_lines_that_already_spell_wave64(self):
        lines = ["v_cmp_lt_u32_e64 s[20:21], v1, v2", "s_mov_b32 s20, 3"]
        text = R.program_text("c", lines, 4, 64, rewrite=False).splitlines()
        self.assertIn("; mattr: +wavefrontsize64,-wavefrontsize32", text)
        self.assertIn("s_mov_b64 exec, -1", text)
        self.assertIn("s_mov_b32 s20, 3", text)
        self.assertIn("v_cmp_lt_u32_e64 s[20:21], v1, v2", text)

    def test_opaque_seeds_compute_the_same_values_at_run_time(self):
        self.assertEqual(R.opaque_seed("v_mov_b32_e32 v3, 0x3f800000"), "v_xor_b32_e32 v3, 0xd16eeeee, v22")
        self.assertEqual(R.opaque_seed("v_mov_b32 v1, 5"), "v_xor_b32_e32 v1, 0xeeeeeeeb, v22")
        for line in ("s_mov_b32 s4, 1", "v_mov_b32 v1, v2", "v_mov_b32 v20, 3", "v_add_f32 v1, v2, v3"):
            self.assertEqual(R.opaque_seed(line), line)
        text = R.program_text("c", ["v_mov_b32 v1, 5"], 4, 32, opaque=True).splitlines()
        self.assertIn("buffer_load_dword v22, v21, s[0:3], 0 offen", text)
        self.assertLess(text.index("s_waitcnt vmcnt(0)"), text.index("v_xor_b32_e32 v1, 0xeeeeeeeb, v22"))
        for k in (0x0, 0x1234, 0xFFFFFFFF, 0x7FC00000):
            self.assertEqual(k ^ R.FILL ^ R.FILL, k)

    def test_exact_verdict(self):
        lines = ["v_add_u32 v1, v2, v3", "v_mov_b32 v8, v1", "v_mov_b32 v9, v2", "v_mov_b32 v10, v3", "v_mov_b32 v11, v4"]
        self.assertEqual(R.exact_verdict(lines, [1, 2, 3, 4], [1, 2, 3, 4]), ("match", ""))
        klass, why = R.exact_verdict(lines, [1, 2, 3, 4], [1, 9, 3, 4])
        self.assertEqual(klass, "mismatch")
        self.assertIn("word 1 (v2)", why)

    def test_judge_counts_skips_and_mismatches(self):
        import unittest.mock as mock
        obs = ["v_mov_b32 v8, v1", "v_mov_b32 v9, v2", "v_mov_b32 v10, v3", "v_mov_b32 v11, v4"]
        progs = [["a"] + obs, ["b"] + obs, ["c"] + obs]
        with mock.patch.object(R, "run_programs", return_value=[[1, 2, 3, 4], [1, 2, 3, 5], "llvm-mc: bad"]):
            counts, bad = R.judge(progs, [[1, 2, 3, 4], [1, 2, 3, 4], [0, 0, 0, 0]], R.exact_verdict)
        self.assertEqual(counts, {"match": 1, "mismatch": 1, "skip: llvm-mc: bad": 1})
        self.assertEqual([b[0] for b in bad], [1])


class Guard(unittest.TestCase):
    def test_vk_without_hw_is_a_usage_error(self):
        for tool in ("pk16_diff.py", "vint_diff.py", "sdwa_diff.py", "wint_diff.py"):
            with self.subTest(tool=tool):
                r = subprocess.run([sys.executable, "-I", str(TOOLS / tool), "--oracle", "vk", "--programs", "1"],
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
                self.assertIn("--oracle hw,vk", r.stderr)


if __name__ == "__main__":
    unittest.main()
