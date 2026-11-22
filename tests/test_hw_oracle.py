"""tools/hw_oracle: kernel text, literal and opcode filters, error mapping, and the exit-77 contract. No GPU needed;
the last test runs on the hardware when the oracle tools are present and is skipped otherwise.
Standard library only."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
import unittest.mock

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from hw_oracle import asm, runner  # noqa: E402

PROG = ["s_mov_b32 s4, 0x12345678", "s_mov_b32 s5, 7", "s_add_u32 s6, s4, s5", "s_cselect_b32 s30, 1, 0",
        "v_mov_b32 v8, s4", "v_mov_b32 v9, s5", "v_mov_b32 v10, s6", "v_mov_b32 v11, s30"]


class Text(unittest.TestCase):
    def test_families(self):
        self.assertEqual(asm.family("gfx1150"), "gfx11")
        self.assertEqual(asm.family("gfx1030"), "gfx10")
        self.assertEqual(asm.family("gfx1013"), "gfx10")
        self.assertEqual(asm.family("gfx1201"), "gfx12")
        with self.assertRaises(ValueError):
            asm.family("gfx90a")

    def test_kernel_uses_the_mnemonics_of_its_family(self):
        k11, k10 = asm.kernel_text("k0", PROG, "gfx1150"), asm.kernel_text("k0", PROG, "gfx1030")
        k12 = asm.kernel_text("k0", PROG, "gfx1201")
        for k in (k11, k12):
            self.assertIn("s_load_b64 s[2:3], s[0:1], 0x0", k)
            self.assertIn("global_store_b128 v0, v[8:11], s[2:3]", k)
        self.assertIn("s_load_dwordx2", k10)
        self.assertIn("global_store_dwordx4", k10)
        for k in (k11, k12, k10):
            self.assertIn("  s_add_u32 s6, s4, s5\n", k)
            self.assertIn(".amdhsa_wavefront_size32 1", k)
            # the program runs between the output-pointer load and the store
            self.assertLess(k.index("s_waitcnt lgkmcnt(0)"), k.index("s_add_u32"))
            self.assertLess(k.index("s_add_u32"), k.index("v[8:11]"))

    def test_metadata_is_required_and_lists_every_kernel(self):
        text, names, starts = asm.module_text([PROG, PROG, PROG], "gfx1150")
        self.assertEqual(names, ["k0", "k1", "k2"])
        self.assertIn('.amdgcn_target "amdgcn-amd-amdhsa--gfx1150"', text)
        self.assertEqual(text.count(".symbol: k"), 3)
        self.assertIn(".end_amdgpu_metadata", text)
        self.assertEqual(text.count("global_store_b128"), 3)

    def test_error_lines_map_back_to_kernels(self):
        text, names, starts = asm.module_text([PROG, PROG, PROG], "gfx1150")
        lines = text.split("\n")
        for k, start in enumerate(starts):
            self.assertTrue(lines[start - 1].startswith(".globl k"), (k, lines[start - 1]))
        err = f"k.s:{starts[1] + 7}:3: error: invalid operand\n  s_foo\n"
        self.assertEqual(asm.failing_kernels(err, starts), {1})
        err2 = err + f"k.s:{starts[2]}:1: error: x\n"
        self.assertEqual(asm.failing_kernels(err2, starts), {1, 2})


class Filters(unittest.TestCase):
    def test_literals(self):
        self.assertEqual(asm.literals("s_add_u32 s4, 0x100, s5"), {0x100})
        self.assertEqual(asm.literals("s_add_u32 s4, 64, -16"), set())
        self.assertEqual(asm.literals("s_and_b32 s4, 0x100, 0x100"), {0x100})
        self.assertEqual(len(asm.literals("s_and_b32 s4, 0x100, 0x200")), 2)
        self.assertEqual(asm.literals("s_and_b32 s4, -1000, 0xfffffc18"), {0xFFFFFC18})   # one 32-bit value
        self.assertEqual(asm.literals("s_cmpk_lt_u32 s4, 0x8000"), set())     # SIMM16, not a literal
        self.assertEqual(asm.literals("s_movk_i32 s4, 0xffff"), set())

    def test_skip_reasons(self):
        self.assertIsNone(asm.skip_reason(PROG, "gfx1150"))
        self.assertIn("s_ff0", asm.skip_reason(PROG + ["s_ff0_i32_b32 s4, s5"], "gfx1150"))
        self.assertIsNone(asm.skip_reason(PROG + ["s_ff0_i32_b32 s4, s5"], "gfx1030"))      # GFX10 still has it
        self.assertIn("two different literals", asm.skip_reason(["s_and_b32 s4, 0x100, 0x200"], "gfx1030"))
        # GFX12 keeps GFX11's removals and drops the SOPK compares, which GFX11 still has.
        cmpk = PROG + ["s_cmpk_lt_i32 s4, 0x12"]
        self.assertIsNone(asm.skip_reason(PROG, "gfx1201"))
        self.assertIsNone(asm.skip_reason(cmpk, "gfx1150"))
        self.assertIn("GFX12", asm.skip_reason(cmpk, "gfx1201"))
        self.assertIn("s_ff0", asm.skip_reason(PROG + ["s_ff0_i32_b64 s[4:5], s[6:7]"], "gfx1201"))
        self.assertEqual(len([op for op in asm.GFX12_REMOVED if op.startswith("s_cmpk_")]), 12)


class Unavailable(unittest.TestCase):
    def test_missing_tools_are_reported_not_raised_as_crashes(self):
        with unittest.mock.patch.dict(os.environ, {"LEONIDA_AMD_CLANG": str(TOOLS / "no-such-clang.exe")}):
            with self.assertRaises(runner.Unavailable):
                runner.find_clang()
        with unittest.mock.patch.dict(os.environ, {"LEONIDA_HIP_DLL": str(TOOLS / "no-such.dll")}):
            with self.assertRaises(runner.Unavailable):
                runner.find_hip_dll()

    def test_salu_diff_exits_77_without_the_tools(self):
        env = dict(os.environ, LEONIDA_AMD_CLANG=str(TOOLS / "no-such-clang.exe"))
        r = subprocess.run([sys.executable, "-I", str(TOOLS / "salu_diff.py"), "--oracle", "hw", "--programs", "3"],
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 77, r.stdout + r.stderr)
        self.assertIn("unavailable", r.stdout)

    def test_an_unsupported_architecture_is_unavailable_not_a_traceback(self):
        # Oracle() checks the architecture right after finding the tools; stub those so this runs anywhere.
        with unittest.mock.patch.object(runner, "find_clang", return_value="clang"), \
                unittest.mock.patch.object(runner, "_find_lld", return_value="ld.lld"), \
                unittest.mock.patch.object(runner, "find_hip_dll", return_value="amdhip64_7.dll"):
            with self.assertRaises(runner.Unavailable) as cm:
                runner.Oracle(arch="gfx90a")
        self.assertIn("gfx90a", str(cm.exception))

    def test_unknown_oracle_is_a_usage_error(self):
        r = subprocess.run([sys.executable, "-I", str(TOOLS / "salu_diff.py"), "--oracle", "gpu"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)


class OnHardware(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.oracle = runner.open_oracle()
        except runner.Unavailable as e:
            raise unittest.SkipTest(f"hardware oracle unavailable: {e}") from None

    @classmethod
    def tearDownClass(cls):
        cls.oracle.close()

    def test_the_dll_override_reaches_the_launcher(self):
        with unittest.mock.patch.dict(os.environ, {"LEONIDA_HIP_DLL": str(Path(__file__))}):   # exists, not a DLL
            with self.assertRaises(runner.Unavailable):
                runner.open_oracle()

    def test_a_chunk_the_assembler_rejects_entirely_is_a_harness_error(self):
        with self.assertRaises(RuntimeError):
            self.oracle.run([["s_add_u32 s4, 0x100, 0x200"] + PROG[-4:]])

    def test_runs_a_program_and_reports_a_rejected_one(self):
        bad = ["s_add_u32 s4, 0x100, 0x200"] + PROG[-4:]     # two literals: the assembler refuses it
        good, rejected = self.oracle.run([PROG, bad])
        self.assertEqual(good, [0x12345678, 7, 0x1234567F, 0])
        self.assertIsInstance(rejected, str)
        self.assertIn("rejected", rejected)


if __name__ == "__main__":
    unittest.main()
