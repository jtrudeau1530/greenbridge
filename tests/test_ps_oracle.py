"""tools/ps_oracle.py: the msgpack subset, the PAL ELF rewrite that puts new pixel-shader code in a pipeline binary,
the binary file format, and (with LEONIDA_PS_ORACLE_BUILD set to a runtime build and an AMD GPU selected) a real
draw with patched code. Standard library only."""
import os
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import ps_oracle as P  # noqa: E402

S_ENDPGM = struct.pack("<I", 0xBF810000)
S_VERSION_W64 = struct.pack("<I", 0xB0802004)


def _meta(vgprs=4, sgprs=6):
    return {"amdpal.version": [3, 6], "amdpal.pipelines": [{
        ".api": "Vulkan",
        ".hardware_stages": {".gs": {".vgpr_count": 46, ".sgpr_count": 20},
                             ".ps": {".vgpr_count": vgprs, ".sgpr_count": sgprs, ".entry_point_symbol": P.PS_SYMBOL,
                                     ".checksum_value": 780697862}},
        ".graphics_registers": {".spi_ps_input_cntl": [{".offset": 0, ".fp16_interp_mode": False}],
                                ".spi_ps_in_control": {".num_interps": 1}},
        ".scale": P.F32(0.5),
    }]}


def _elf(gs_code, ps_code, ps_at, meta):
    """A small ELF laid out like the AMD driver's pipeline binary: .shstrtab, .text (gs then ps), .comment, .strtab,
    .symtab and the AMDGPU metadata note, no program headers."""
    names = [b"", b".shstrtab", b".text", b".comment", b".strtab", b".symtab", b".note"]
    shstr = b"\0".join(names) + b"\0"
    off = {n: shstr.index(n + b"\0") if n else 0 for n in names}
    text = gs_code + b"\0" * (ps_at - len(gs_code)) + ps_code
    strtab = b"\0_amdgpu_gs_main\0" + P.PS_SYMBOL.encode() + b"\0"
    sym = b"\0" * 24
    sym += struct.pack("<IBBHQQ", 1, 0x12, 0, 2, 0, len(gs_code))
    sym += struct.pack("<IBBHQQ", strtab.index(P.PS_SYMBOL.encode()), 0x12, 0, 2, ps_at, len(ps_code))
    desc = P.mp_dump(meta)
    note = struct.pack("<III", 7, len(desc), 32) + b"AMDGPU\0\0" + desc
    note += b"\0" * (-len(note) % 4)
    secs = [(".shstrtab", 3, shstr, 0, 0), (".text", 1, text, 0, 0), (".comment", 1, b"test\0", 0, 0),
            (".strtab", 3, strtab, 0, 0), (".symtab", 2, sym, 4, 24), (".note", 7, note, 0, 0)]
    body, hdrs, at = bytearray(), [(0,) * 10], 64
    for name, typ, data, link, ent in secs:
        hdrs.append((off[name.encode()], typ, 6 if name == ".text" else 0, 0, at, len(data), link, 0,
                     256 if name == ".text" else 0, ent))
        body += data
        at += len(data)
    while (64 + len(body)) % 8:
        body += b"\0"
    head = bytearray(64)
    head[:16] = b"\x7fELF\x02\x01\x01\x41" + bytes(8)
    struct.pack_into("<HHIQQQIHHHHHH", head, 16, 1, 0xE0, 1, 0, 0, 64 + len(body), 0x945, 64, 0, 0, 64,
                     len(hdrs), 1)
    return bytes(head) + bytes(body) + b"".join(struct.pack(P._SH, *h) for h in hdrs)


class Msgpack(unittest.TestCase):
    def test_round_trip(self):
        values = [0, 127, 128, 255, 256, 65536, 1 << 40, -1, -32, -33, -200, -40000, -(1 << 40), True, False, None,
                  "", "x" * 31, "y" * 32, "z" * 300, b"\x01\x02", [], list(range(20)), {"a": 1, "b": [1, {"c": None}]},
                  {str(i): i for i in range(20)}, 1.5, P.F32(0.25)]
        for v in values:
            with self.subTest(v=v if not isinstance(v, str) else v[:5]):
                got, end = P.mp_load(P.mp_dump(v))
                self.assertEqual(got, v)
                self.assertEqual(end, len(P.mp_dump(v)))
                self.assertEqual(type(got) is P.F32, type(v) is P.F32)

    def test_minimal_encodings(self):
        self.assertEqual(P.mp_dump(5), b"\x05")
        self.assertEqual(P.mp_dump(200), b"\xcc\xc8")
        self.assertEqual(P.mp_dump(-5), b"\xfb")
        self.assertEqual(P.mp_dump(P.F32(1.0)), b"\xca\x3f\x80\x00\x00")
        self.assertEqual(P.mp_dump({"a": 1}), b"\x81\xa1a\x01")
        with self.assertRaises(ValueError):
            P.mp_load(b"\xc1")


class PatchPixelShader(unittest.TestCase):
    def setUp(self):
        self.gs = S_VERSION_W64 + S_ENDPGM
        self.ps = S_VERSION_W64 + struct.pack("<II", 0x7E0002FF, 0x3F800000) + S_ENDPGM
        self.elf = _elf(self.gs, self.ps, 0x100, _meta())

    def test_the_test_elf_reads_back(self):
        e = P.Elf(self.elf)
        self.assertEqual([s[0] for s in e.symbols()], ["", "_amdgpu_gs_main", P.PS_SYMBOL])
        self.assertEqual(P.ps_stage(e.metadata())[".vgpr_count"], 4)
        self.assertEqual(P.ps_code(self.elf), (0x100, self.ps))
        self.assertTrue(P.is_wave64(self.elf))

    def test_longer_code_grows_text_and_keeps_everything_else(self):
        code = S_VERSION_W64 + struct.pack("<II", 0x7E0002FF, 0x12345678) * 50 + S_ENDPGM
        out = P.patch_ps(self.elf, code, ["v_mov_b32 v40, 0", "s_mov_b32 s[20:21], 0"],
                         [(".graphics_registers/.spi_ps_input_cntl/0/.fp16_interp_mode", True)])
        e = P.Elf(out)
        text = bytes(e.data[e.index(".text")])
        self.assertEqual(text[:len(self.gs)], self.gs)                    # the vertex stage is untouched
        self.assertEqual(text[0x100:0x100 + len(code)], code)
        pad = text[0x100 + len(code):]
        self.assertGreaterEqual(len(pad), 256)                            # s_code_end past the end, as the driver
        self.assertEqual(pad, P.S_CODE_END * (len(pad) // 4))
        self.assertEqual((0x100 + len(code) + len(pad)) % 64, 0)
        sym = {s[0]: s for s in e.symbols()}
        self.assertEqual(sym[P.PS_SYMBOL][1:3], (0x100, len(code) + (len(pad) - 256)))
        meta = e.metadata()
        ps = P.ps_stage(meta)
        self.assertEqual((ps[".vgpr_count"], ps[".sgpr_count"]), (41, 22))   # raised to the registers named
        self.assertEqual(ps[".checksum_value"], 780697862)
        regs = meta["amdpal.pipelines"][0][".graphics_registers"]
        self.assertIs(regs[".spi_ps_input_cntl"][0][".fp16_interp_mode"], True)
        self.assertIs(type(meta["amdpal.pipelines"][0][".scale"]), P.F32)
        self.assertEqual(bytes(e.data[e.index(".comment")]), b"test\0")
        self.assertEqual(P.Elf(e.to_bytes()).to_bytes(), out)            # stable when written again

    def test_counts_are_never_lowered_and_an_untouched_note_is_kept(self):
        out = P.patch_ps(_elf(self.gs, self.ps, 0x100, _meta(vgprs=64, sgprs=40)), self.ps, ["v_mov_b32 v1, s2"])
        e = P.Elf(out)
        self.assertEqual(P.ps_stage(e.metadata())[".vgpr_count"], 64)
        self.assertEqual(bytes(e.data[e.index(".note")]),
                         bytes(P.Elf(_elf(self.gs, self.ps, 0x100, _meta(64, 40))).data[e.index(".note")]))

    def test_refusals(self):
        e = P.Elf(self.elf)                            # the vertex stage moved after the pixel stage
        struct.pack_into("<Q", e.data[e.index(".symtab")], 24 + 8, 0x200)
        with self.assertRaises(ValueError):
            P.patch_ps(e.to_bytes(), self.ps)
        with self.assertRaises(ValueError):
            P.patch_ps(self.elf, self.ps, [], [(".graphics_registers/.no_such_register", 1)])
        with self.assertRaises(P.Unavailable):
            P.Elf(b"MZ" + bytes(100))

    def test_registers_named(self):
        self.assertEqual(P.regs_used(["v_interp_p1_f32_e32 v4, v0, attr0.z", "s_mov_b32 m0, s2",
                                      "v_mad_u64_u32 v[8:9], s[10:11], v1, v1, v[2:3] ; v200"]), (10, 12))


class BinaryFile(unittest.TestCase):
    def test_round_trip(self):
        bins = [(16, bytes(range(32)), b"\x7fELF" + bytes(40)), (32, b"\xff" * 32, b"")]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "b.bin")
            P.write_binaries(p, bins)
            self.assertEqual(P.read_binaries(p), bins)


class Cli(unittest.TestCase):
    def test_nothing_to_do_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as cm:
            P.main(["--build", "."])
        self.assertEqual(cm.exception.code, 2)

    def test_no_harness_is_unavailable(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(P.main(["--build", d, "--show-template"]), 77)


@unittest.skipUnless(os.environ.get("LEONIDA_PS_ORACLE_BUILD"), "set LEONIDA_PS_ORACLE_BUILD to a runtime build "
                     "and select an AMD GPU (LEONIDA_VK_DEVICE) to run the pixel shader on hardware")
class OnHardware(unittest.TestCase):
    def test_patched_code_runs_as_the_pixel_shader(self):
        # interp2: location 0 is (u, v, 0, 1), u = 0 at vertex 0 and 2 at vertex 1, so on a 4 x 1 target pixel x has
        # u = (x + 0.5) / 4. The parameter words in LDS are P0 = 0 (vertex 0) and P10 = 2.0 (vertex 1 - vertex 0).
        try:
            o = P.Oracle(os.environ["LEONIDA_PS_ORACLE_BUILD"], template="interp2", size=(4, 1))
        except P.Unavailable as e:
            self.skipTest(str(e))
        try:
            px = o.run(["s_mov_b32 m0, s2", "v_interp_p1_f32_e32 v2, v0, attr0.x", "v_interp_p2_f32_e32 v2, v1, attr0.x",
                        "v_interp_mov_f32_e32 v3, p0, attr0.x", "v_interp_mov_f32_e32 v4, p10, attr0.x",
                        "v_mov_b32_e32 v40, 0x12345678", "v_mov_b32_e32 v5, v40",
                        "exp mrt0, v2, v3, v4, v5 done vm", "s_endpgm"])
        finally:
            o.close()
        want_u = [0x3E000000, 0x3EC00000, 0x3F200000, 0x3F600000]    # 0.125, 0.375, 0.625, 0.875
        self.assertEqual(px, [[u, 0, 0x40000000, 0x12345678] for u in want_u])


if __name__ == "__main__":
    unittest.main()
