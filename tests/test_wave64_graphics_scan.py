"""tools/wave64_graphics_scan.py: classification, counting and privacy; no game data.

    python -I tests/test_wave64_graphics_scan.py -v
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import wave64_graphics_scan as scan  # noqa: E402


def fixture(name, lines):
    """A synthetic disassembly: the real files start with shader_disasm's '; name' header, then asm."""
    return name, "; %s  code 0x0..0x10 (header), gfx1013, 0 invalid words\n" % name[:-2] + "\n".join(lines) + "\n"


PS_W64_SENSITIVE = fixture("ps_w64a.s", [
    "v_mbcnt_hi_u32_b32 v1, v2, v3",
    "s_cbranch_execz 42",
    "v_add_f32_e32 v0, v1, v2",
])
PS_W64_CLEAN = fixture("ps_w64clean.s", [
    # _saveexec_b64 marks the shader wave64 for WAVE64_RE, but mask arithmetic alone is per-lane and exact
    "s_and_saveexec_b64 s[0:1], s[2:3], s[4:5]",
    "v_add_f32_e32 v0, v1, v2",
])
PS_W32 = fixture("ps_w32a.s", ["v_add_f32_e32 v0, v1, v2"])
VS_W64 = fixture("vs_w64b.s", [
    # a real wave64 shader always touches the 64-bit exec, which is what WAVE64_RE sees
    "s_or_saveexec_b64 s[0:1], s[2:3], exec",
    "v_readlane_b32 v1, v2, s4",
    "v_readlane_b32 v3, v4, s5",
])
GS_W64 = fixture("gs_w64d.s", ["s_ff1_i32_b64 s0, s[2:3]"])
CS_W64 = fixture("cs_w64c.s", ["v_mbcnt_hi_u32_b32 v1, v2, v3"])  # compute: never counted
MS_W64 = fixture("ms_w64e.s", ["s_cbranch_execz 7"])             # mesh: never counted
VS_UNKNOWN = fixture("xx_w64f.s", ["v_readfirstlane_b32 s1, v1"])  # stage '?': skipped


class ScanTests(unittest.TestCase):
    def scan(self, *fixtures):
        with tempfile.TemporaryDirectory() as directory:
            for name, text in fixtures:
                with open(os.path.join(directory, name), "w", encoding="utf-8") as f:
                    f.write(text)
            return scan.scan_directory(directory)

    def test_counts_wave64_graphics_only(self):
        report = self.scan(PS_W64_SENSITIVE, PS_W64_CLEAN, PS_W32, VS_W64, GS_W64, CS_W64, MS_W64, VS_UNKNOWN)
        ps, vs, gs = report["per_stage"]["ps"], report["per_stage"]["vs"], report["per_stage"]["gs"]
        self.assertEqual(ps["shaders"], 3)
        self.assertEqual(ps["wave64"], 2)
        self.assertEqual(ps["with_cross_half"], 1)
        self.assertEqual(vs["wave64"], 1)
        self.assertEqual(gs["wave64"], 1)
        self.assertEqual(report["other_stages"], {"shaders": 3, "wave64": 2})  # cs + ms wave64, xx unknown

    def test_occurrence_counts_per_op(self):
        report = self.scan(PS_W64_SENSITIVE, VS_W64, GS_W64)
        ps = report["per_stage"]["ps"]["occurrences"]
        self.assertEqual(ps, {"v_mbcnt_hi_u32_b32": 1, "s_cbranch_execz": 1})
        self.assertEqual(report["per_stage"]["vs"]["occurrences"], {"v_readlane_b32": 2})
        self.assertEqual(report["per_stage"]["gs"]["occurrences"], {"s_ff1_i32_b64": 1})

    def test_clean_wave64_shader_has_no_cross_half(self):
        report = self.scan(PS_W64_CLEAN)
        b = report["per_stage"]["ps"]
        self.assertEqual((b["wave64"], b["with_cross_half"], b["occurrences"]), (1, 0, {}))

    def test_stage_from_index_json_overrides_the_name(self):
        # stage_of reads <disasm>/../index.json (the carve index), like the translator does
        with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as disasm:
            with open(os.path.join(work, "index.json"), "w", encoding="utf-8") as f:
                json.dump([{"file": "ps_0001.dxbc", "stage": "ps"}], f)
            real = os.path.join(work, "dxbc", "disasm")
            os.makedirs(real)
            name, text = PS_W64_SENSITIVE
            with open(os.path.join(real, "ps_0001.s"), "w", encoding="utf-8") as f:
                f.write(text)  # name prefix says ps_ but the index is what counts
            report = scan.scan_directory(real)
            self.assertEqual(report["per_stage"]["ps"]["with_cross_half"], 1)

    def test_main_json_has_no_shader_names(self):
        # privacy rule: nothing from the game leaves the work folder, not even file names
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outdir:
            for name, text in (PS_W64_SENSITIVE, VS_W64):
                with open(os.path.join(directory, name), "w", encoding="utf-8") as f:
                    f.write(text)
            out = os.path.join(outdir, "scan.json")
            report = scan.scan_directory(directory)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(report, f)
            blob = open(out, encoding="utf-8").read()
            self.assertNotIn(".s", blob)
            self.assertNotIn("ps_w64a", blob)
            self.assertIn("v_mbcnt_hi_u32_b32", blob)


if __name__ == "__main__":
    unittest.main(verbosity=2)
