"""Unit tests for tools/pe_inspect.py: standard-library PE parsing, directory walks and the report it writes.

The fixtures are built here, byte by byte, so the test needs no game files and no third-party packages. The 32-bit
image exists because a delay-import descriptor without dlattrRva stores virtual addresses in 32-bit fields: that form
can only be built for an image whose base fits in 32 bits, and it is what the parser has to convert.

    python -I tests/test_pe_inspect.py
"""
import contextlib
import io
import json
import os
import struct
import sys
import shutil
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import pe_inspect  # noqa: E402

IMAGE_BASE = 0x180000000
BASE32 = 0x10000000
TEXT_RVA = 0x1000
TEXT_RAW = 0x200
TEXT_SIZE = 0x1000


def place(buf: bytearray, rva: int, blob: bytes) -> None:
    at = TEXT_RAW + (rva - TEXT_RVA)
    assert at + len(blob) <= len(buf), f"fixture does not fit at {rva:#x}"
    buf[at:at + len(blob)] = blob


def build_image(is64: bool = True, with_directories: bool = True) -> bytes:
    """A one-section image with imports (by name and by ordinal), a delay import, exports (two named, one
    ordinal-only), .pdata entries, TLS callbacks and a relocation directory. The 32-bit build also carries a
    delay-import descriptor whose fields are virtual addresses rather than RVAs."""
    base = IMAGE_BASE if is64 else BASE32
    opt_size = 240 if is64 else 224
    assert 0x58 + opt_size + 40 <= TEXT_RAW
    buf = bytearray(TEXT_RAW + TEXT_SIZE)

    buf[0:2] = b"MZ"
    struct.pack_into("<I", buf, 0x3C, 0x40)
    buf[0x40:0x44] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", buf, 0x44,
                     0x8664 if is64 else 0x14C,     # machine
                     1,                            # sections
                     0x5F000000,                   # timestamp
                     0, 0,                         # symbol table
                     opt_size,
                     0x2022 if is64 else 0x0102)   # DLL+executable+large-address, or executable+32-bit

    opt = 0x58
    if is64:
        struct.pack_into("<H", buf, opt, 0x20B)
        struct.pack_into("<I", buf, opt + 16, TEXT_RVA)               # AddressOfEntryPoint
        struct.pack_into("<Q", buf, opt + 24, base)                   # ImageBase
        struct.pack_into("<I", buf, opt + 108, 16)                    # NumberOfRvaAndSizes
    else:
        struct.pack_into("<H", buf, opt, 0x10B)
        struct.pack_into("<I", buf, opt + 16, TEXT_RVA)
        struct.pack_into("<I", buf, opt + 28, base)
        struct.pack_into("<I", buf, opt + 92, 16)
    struct.pack_into("<I", buf, opt + 32, 0x1000)                     # SectionAlignment
    struct.pack_into("<I", buf, opt + 36, 0x200)                      # FileAlignment
    struct.pack_into("<I", buf, opt + 56, 0x2000)                     # SizeOfImage
    struct.pack_into("<I", buf, opt + 60, TEXT_RAW)                   # SizeOfHeaders
    struct.pack_into("<H", buf, opt + 68, 2)                          # Subsystem: GUI
    struct.pack_into("<H", buf, opt + 70, 0x40)                       # DllCharacteristics: dynamic base

    directories = {}
    if with_directories:
        directories = {
            pe_inspect.DIR_EXPORT: (0x1700, 40),
            pe_inspect.DIR_IMPORT: (0x1100, 40),
            pe_inspect.DIR_EXCEPTION: (0x1800, 36),
            pe_inspect.DIR_BASERELOC: (0x1A00, 12),
            pe_inspect.DIR_TLS: (0x1900, 40 if is64 else 24),
            pe_inspect.DIR_DELAY_IMPORT: (0x1500, 32 if is64 else 64),
        }
    dirs_at = opt + (112 if is64 else 96)
    for index, (rva, size) in directories.items():
        struct.pack_into("<II", buf, dirs_at + index * 8, rva, size)

    section = opt + opt_size
    buf[section:section + 8] = b".text\0\0\0"
    struct.pack_into("<IIII", buf, section + 8, TEXT_SIZE, TEXT_RVA, TEXT_SIZE, TEXT_RAW)
    struct.pack_into("<I", buf, section + 36, 0x60000020)              # code, execute, read

    if not with_directories:
        return bytes(buf)

    thunk = "Q" if is64 else "I"
    ordinal = (1 << (63 if is64 else 31)) | 7

    # Import table: one import by name, one by ordinal.
    place(buf, 0x1100, struct.pack("<IIIII", 0x1200, 0, 0, 0x1300, 0x1240))
    place(buf, 0x1114, struct.pack("<IIIII", 0x1280, 0, 0, 0x1320, 0x12C0))
    place(buf, 0x1128, bytes(20))                                      # terminator
    place(buf, 0x1200, struct.pack(f"<{thunk}{thunk}", 0x1400, 0))
    place(buf, 0x1240, bytes(16))
    place(buf, 0x1280, struct.pack(f"<{thunk}{thunk}", ordinal, 0))
    place(buf, 0x12C0, bytes(16))
    place(buf, 0x1300, b"kernel32.dll\0")
    place(buf, 0x1320, b"xmem.dll\0")

    # Delay-import table. An x64 image can only use dlattrRva: without it the fields are 32-bit virtual addresses,
    # which cannot hold a base of 0x180000000.
    place(buf, 0x1500, struct.pack("<IIIIIIII", 1, 0x1340, 0, 0x1600, 0x1580, 0, 0, 0))
    if not is64:
        place(buf, 0x1520, struct.pack("<IIIIIIII", 0, base + 0x1360, 0, base + 0x1640,
                                       base + 0x1620, 0, 0, 0))
    place(buf, 0x1540, bytes(32))                                      # terminator
    place(buf, 0x1580, struct.pack(f"<{thunk}{thunk}", 0x1420, 0))
    place(buf, 0x1600, bytes(16))
    place(buf, 0x1620, struct.pack(f"<{thunk}{thunk}", 0x1440, 0))
    place(buf, 0x1640, bytes(16))
    place(buf, 0x1340, b"xmem.dll\0")
    place(buf, 0x1360, b"d3d12_xs.dll\0")
    place(buf, 0x1400, struct.pack("<H", 0) + b"GetCurrentProcessId\0")
    place(buf, 0x1420, struct.pack("<H", 0) + b"XMemAlloc\0")
    place(buf, 0x1440, struct.pack("<H", 0) + b"D3D12SerializeVersionedRootSignature\0")

    # Export directory: functions 5 and 7 are named, function 6 is ordinal-only.
    place(buf, 0x1700, struct.pack("<IIHHIIIIIII", 0, 0, 0, 0, 0x1780, 5, 3, 2, 0x1730, 0x1740, 0x1750))
    place(buf, 0x1730, struct.pack("<III", 0x1000, 0x1008, 0x1010))
    place(buf, 0x1740, struct.pack("<II", 0x1760, 0x1770))
    place(buf, 0x1750, struct.pack("<HH", 0, 2))
    place(buf, 0x1760, b"ExportedA\0")
    place(buf, 0x1770, b"ExportedB\0")
    place(buf, 0x1780, b"sample.dll\0")

    # .pdata: three RUNTIME_FUNCTION entries. TLS: two callbacks then the zero terminator.
    place(buf, 0x1800, bytes(36))
    tls = bytearray(40 if is64 else 24)
    struct.pack_into("<Q" if is64 else "<I", tls, 24 if is64 else 12, base + 0x1940)
    place(buf, 0x1900, bytes(tls))
    cb = "QQQ" if is64 else "III"
    place(buf, 0x1940, struct.pack(f"<{cb}", base + 0x1000, base + 0x1008, 0))
    place(buf, 0x1A00, bytes(12))
    return bytes(buf)


EXPORT_FUNCS = 0x1F00  # AddressOfFunctions for the rebuilt export tables below: last in the section, so a cut can reach it
FORWARDER = 0x1790     # inside the export directory once its size is widened: a forwarder string lives here


def with_exports(slots: list[int], names: dict[int, str], base: int = 10, is64: bool = True) -> bytes:
    """build_image() with its export directory replaced: `slots` is AddressOfFunctions, `names` maps an index into
    it to the export name that points at it."""
    buf = bytearray(build_image(is64=is64))
    img = pe_inspect.PeImage(bytes(buf))
    dir_entry = (0x58 + (112 if is64 else 96)) + pe_inspect.DIR_EXPORT * 8
    struct.pack_into("<II", buf, dir_entry, 0x1700, 0x100)
    ordered = sorted(names.items(), key=lambda kv: kv[1])
    place(buf, 0x1700, struct.pack("<IIHHIIIIIII", 0, 0, 0, 0, 0x1780, base, len(slots), len(ordered),
                                   EXPORT_FUNCS, 0x1C00, 0x1C80))
    place(buf, 0x1780, b"sample.dll\0")
    place(buf, FORWARDER, b"other.Target\0")
    place(buf, EXPORT_FUNCS, struct.pack(f"<{len(slots)}I", *slots))
    text = 0x1D00
    for n, (index, name) in enumerate(ordered):
        place(buf, 0x1C00 + 4 * n, struct.pack("<I", text))
        place(buf, 0x1C80 + 2 * n, struct.pack("<H", index))
        place(buf, text, name.encode() + b"\0")
        text += len(name) + 1
    assert img.offset(EXPORT_FUNCS) is not None
    return bytes(buf)


class PeInspectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = cls.tmp.name
        cls.image = os.path.join(cls.dir, "sample.dll")
        with open(cls.image, "wb") as f:
            f.write(build_image())
        cls.image32 = os.path.join(cls.dir, "sample32.exe")
        with open(cls.image32, "wb") as f:
            f.write(build_image(is64=False))
        cls.report = pe_inspect.inspect(cls.image, cls.dir)
        cls.report32 = pe_inspect.inspect(cls.image32, cls.dir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_headers(self):
        r = self.report
        self.assertEqual(r["machine"], "x86-64")
        self.assertTrue(r["is_dll"])
        self.assertEqual(r["subsystem"], "GUI")
        self.assertEqual(r["image_base"], "0x180000000")
        self.assertEqual(r["size_of_image"], 0x2000)
        self.assertEqual(r["entry_rva"], "0x1000")
        self.assertTrue(r["has_relocs"])
        self.assertTrue(r["dynamic_base"])

    def test_sections(self):
        sections = self.report["sections"]
        self.assertEqual([s["name"] for s in sections], [".text"])
        self.assertTrue(sections[0]["exec"])
        self.assertFalse(sections[0]["write"])
        self.assertFalse(sections[0]["looks_encrypted"])
        self.assertGreater(sections[0]["entropy"], 0.0)

    def test_imports_by_name_and_ordinal(self):
        imports = {(i["dll"], i["kind"]): i for i in self.report["imports"]}
        self.assertEqual(imports[("kernel32.dll", "normal")]["functions"], ["GetCurrentProcessId"])
        self.assertEqual(imports[("xmem.dll", "normal")]["functions"], ["#7"])

    def test_delay_import_with_rva_descriptor(self):
        delay = [i for i in self.report["imports"] if i["kind"] == "delay"]
        self.assertEqual(len(delay), 1)
        self.assertEqual(delay[0]["dll"], "xmem.dll")
        self.assertEqual(delay[0]["functions"], ["XMemAlloc"])

    def test_delay_import_descriptor_with_virtual_addresses(self):
        imports = {(i["dll"], i["kind"]): i for i in self.report32["imports"]}
        self.assertEqual(imports[("xmem.dll", "delay")]["functions"], ["XMemAlloc"])
        self.assertEqual(imports[("d3d12_xs.dll", "delay")]["functions"],
                         ["D3D12SerializeVersionedRootSignature"])

    def test_exports_include_ordinal_only_entry(self):
        for report in (self.report, self.report32):
            self.assertEqual(report["export_count"], 3)
            self.assertEqual(set(report["exports"]), {"ExportedA", "ExportedB", "#6"})

    def exports_of(self, data: bytes) -> list[str]:
        return pe_inspect._exports(pe_inspect.PeImage(data))

    def test_unused_ordinal_slot_is_not_an_export(self):
        for is64 in (True, False):
            data = with_exports([0x1000, 0, 0x1010], {0: "Alpha"}, is64=is64)
            self.assertEqual(self.exports_of(data), ["Alpha", "#12"])
            path = os.path.join(self.dir, "gap.dll")
            with open(path, "wb") as f:
                f.write(data)
            report = pe_inspect.inspect(path, None)
            self.assertEqual(report["exports"], ["Alpha", "#12"])
            self.assertEqual(report["export_count"], 2)

    def test_unused_slots_in_a_table_without_names(self):
        data = with_exports([0, 0x1000, 0, 0x1010, 0], {})
        self.assertEqual(self.exports_of(data), ["#11", "#13"])

    def test_unnamed_forwarder_is_still_an_export(self):
        data = with_exports([0x1000, FORWARDER, 0], {0: "Alpha"})
        self.assertEqual(self.exports_of(data), ["Alpha", "#11"])

    def test_function_table_cut_by_end_of_file_lists_the_readable_slots(self):
        data = with_exports([0x1000, 0, 0x1010, 0x1020], {})
        at = pe_inspect.PeImage(data).offset(EXPORT_FUNCS)
        for keep, expected in ((8, ["#10"]), (10, ["#10"]), (12, ["#10", "#12"]), (0, [])):
            with self.subTest(bytes_kept=keep):
                self.assertEqual(self.exports_of(data[:at + keep]), expected)

    def test_table_without_gaps_is_listed_as_before(self):
        data = with_exports([0x1000, 0x1008, 0x1010, 0x1018], {1: "Beta", 3: "Delta"}, base=1)
        self.assertEqual(self.exports_of(data), ["Beta", "Delta", "#1", "#3"])

    def test_pdata_and_tls_counts(self):
        for report in (self.report, self.report32):
            self.assertEqual(report["pdata_functions"], 3)
            self.assertEqual(report["tls_callbacks"], 2)

    def test_pe32_headers_and_thunks(self):
        r = self.report32
        self.assertEqual(r["machine"], "x86")
        self.assertEqual(r["image_base"], "0x10000000")
        self.assertFalse(r["is_dll"])
        imports = {(i["dll"], i["kind"]): i for i in r["imports"]}
        self.assertEqual(imports[("kernel32.dll", "normal")]["functions"], ["GetCurrentProcessId"])
        self.assertEqual(imports[("xmem.dll", "normal")]["functions"], ["#7"])

    def test_bundled_module_is_classified_from_the_game_dir(self):
        with tempfile.TemporaryDirectory() as game:
            path = os.path.join(game, "xmem.dll")
            open(path, "wb").close()
            self.assertEqual(pe_inspect.classify_dll("xmem.dll", game), "bundled")
        self.assertEqual(pe_inspect.classify_dll("xmem.dll", None), "shim")

    @unittest.skipUnless(os.name == "nt", "system DLL classification needs Windows")
    def test_system_dll_is_classified_system(self):
        self.assertEqual(pe_inspect.classify_dll("kernel32.dll", None), "system")

    def test_apiset_query_returns_a_bool_and_unknown_contracts_do_not_resolve(self):
        self.assertIsInstance(pe_inspect._apiset_resolves("api-ms-win-core-file-l1-2-0"), bool)
        self.assertFalse(pe_inspect._apiset_resolves("api-ms-win-not-a-real-contract-l9-9-9"))

    def test_markdown_report(self):
        md = pe_inspect.to_markdown(self.report)
        self.assertIn("# sample.dll", md)
        self.assertIn("| dll | kind | resolves | functions |", md)
        self.assertIn("| xmem.dll | normal | shim | 1 |", md)
        self.assertIn("## Needs checking or a shim", md)
        self.assertIn("`XMemAlloc`", md)

    def test_non_pe_input_is_rejected(self):
        bad = os.path.join(self.dir, "not_pe.bin")
        with open(bad, "wb") as f:
            f.write(b"not a PE file")
        with self.assertRaises(pe_inspect.PeFormatError):
            pe_inspect.inspect(bad, None)

    def test_minimal_image_without_directories(self):
        path = os.path.join(self.dir, "bare.bin")
        with open(path, "wb") as f:
            f.write(build_image(with_directories=False))
        r = pe_inspect.inspect(path, None)
        self.assertEqual(r["imports"], [])
        self.assertEqual(r["exports"], [])
        self.assertEqual(r["tls_callbacks"], 0)
        self.assertEqual(r["pdata_functions"], 0)
        self.assertFalse(r["has_relocs"])

    def test_cli_writes_json_and_markdown(self):
        out = os.path.join(self.dir, "reports")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(pe_inspect.main([self.image, "--out", out]), 0)
        with open(os.path.join(out, "sample.dll.json"), encoding="utf-8") as f:
            written = json.load(f)
        self.assertEqual(written["machine"], "x86-64")
        self.assertEqual(written["imports"][0]["dll"], "kernel32.dll")
        with open(os.path.join(out, "sample.dll.md"), encoding="utf-8") as f:
            self.assertIn("# sample.dll", f.read())

    def run_cli(self, images: list[str], out: str) -> str:
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            self.assertEqual(pe_inspect.main([*images, "--out", out]), 0)
        return err.getvalue()

    def read_report(self, out: str, name: str) -> tuple[dict, str]:
        with open(os.path.join(out, name + ".json"), encoding="utf-8") as f:
            data = json.load(f)
        with open(os.path.join(out, name + ".md"), encoding="utf-8") as f:
            return data, f.read()

    def test_cli_exe_and_dll_with_the_same_stem_get_separate_reports(self):
        sub = os.path.join(self.dir, "stem")
        os.makedirs(sub)
        exe, dll = os.path.join(sub, "game.exe"), os.path.join(sub, "game.dll")
        with open(exe, "wb") as f:
            f.write(build_image(is64=False))
        with open(dll, "wb") as f:
            f.write(build_image())
        out = os.path.join(self.dir, "reports_stem")
        self.run_cli([exe, dll], out)
        self.assertEqual(sorted(os.listdir(out)), ["game.dll.json", "game.dll.md", "game.exe.json", "game.exe.md"])
        exe_r, exe_md = self.read_report(out, "game.exe")
        dll_r, dll_md = self.read_report(out, "game.dll")
        self.assertEqual((exe_r["file"], exe_r["machine"]), ("game.exe", self.report32["machine"]))
        self.assertEqual((dll_r["file"], dll_r["machine"]), ("game.dll", "x86-64"))
        self.assertNotEqual(exe_r["machine"], dll_r["machine"])
        self.assertIn("# game.exe", exe_md)
        self.assertIn("# game.dll", dll_md)

    def test_cli_same_file_name_from_two_directories_gets_a_numbered_suffix(self):
        a, b = os.path.join(self.dir, "a"), os.path.join(self.dir, "b")
        os.makedirs(a)
        os.makedirs(b)
        first, second = os.path.join(a, "x.exe"), os.path.join(b, "x.exe")
        with open(first, "wb") as f:
            f.write(build_image())
        with open(second, "wb") as f:
            f.write(build_image(is64=False))
        out = os.path.join(self.dir, "reports_dup")
        err = self.run_cli([first, second, first], out)
        self.assertEqual(sorted(os.listdir(out)), ["x.exe-2.json", "x.exe-2.md", "x.exe.json", "x.exe.md"])
        r1, _ = self.read_report(out, "x.exe")
        r2, _ = self.read_report(out, "x.exe-2")
        self.assertEqual(r1["machine"], "x86-64")
        self.assertEqual(r2["machine"], self.report32["machine"])
        self.assertNotEqual(r1["machine"], r2["machine"])
        self.assertIn("report written as x.exe-2.json", err)
        self.assertIn("given more than once", err)

    def check_same_image_under_another_folder(self, tag: str, link) -> None:
        a, b = os.path.join(self.dir, tag + "_a"), os.path.join(self.dir, tag + "_b")
        os.makedirs(a)
        os.makedirs(b)
        first, second = os.path.join(a, "x.exe"), os.path.join(b, "x.exe")
        with open(first, "wb") as f:
            f.write(build_image())
        link(first, second)
        open(os.path.join(b, "xmem.dll"), "wb").close()
        out = os.path.join(self.dir, "reports_" + tag)
        self.run_cli([first, second], out)
        self.assertEqual(sorted(os.listdir(out)), ["x.exe-2.json", "x.exe-2.md", "x.exe.json", "x.exe.md"])

        def resolves(report: dict) -> dict:
            return {(i["dll"], i["kind"]): i["where"] for i in report["imports"]}

        for name, img in (("x.exe", first), ("x.exe-2", second)):
            single = os.path.join(self.dir, "reports_" + tag + "_single_" + name)
            self.run_cli([img], single)
            combined, _ = self.read_report(out, name)
            alone, _ = self.read_report(single, "x.exe")
            self.assertEqual(resolves(combined), resolves(alone))
        self.assertEqual(resolves(self.read_report(out, "x.exe")[0])[("xmem.dll", "normal")], "shim")
        self.assertEqual(resolves(self.read_report(out, "x.exe-2")[0])[("xmem.dll", "normal")], "bundled")

    def test_cli_symlinked_image_under_another_folder_gets_its_own_report(self):
        def link(src, dst):
            try:
                os.symlink(src, dst)
            except (OSError, NotImplementedError) as e:
                self.skipTest(f"os.symlink unavailable: {e}")
        self.check_same_image_under_another_folder("symlink", link)

    def test_cli_copied_image_under_another_folder_gets_its_own_report(self):
        self.check_same_image_under_another_folder("copy", shutil.copyfile)

    def test_cli_reports_a_non_pe_file(self):
        bad = os.path.join(self.dir, "not_pe_cli.bin")
        with open(bad, "wb") as f:
            f.write(b"MZ" + bytes(10))
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            self.assertEqual(pe_inspect.main([bad, "--out", os.path.join(self.dir, "reports2")]), 1)
        self.assertIn("not a PE image", err.getvalue())


class TruncatedImageTest(unittest.TestCase):
    """A file cut off inside a header or table is a bad image, not a struct.error that stops the run."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name

    def write(self, name: str, data: bytes) -> str:
        path = os.path.join(self.dir, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def cut_inside(self, full: bytes, index: int, into: int) -> bytes:
        """The image cut `into` bytes after the start of data directory `index`."""
        rva, _size = pe_inspect.PeImage(full).directory(index)
        return full[:pe_inspect.PeImage(full).offset(rva) + into]

    def test_each_table_cut_short_is_a_pe_format_error(self):
        for is64 in (True, False):
            full = build_image(is64=is64)
            cases = {
                "optional header": full[:0x58 + 2 + 16],     # magic present, AddressOfEntryPoint cut off
                "export directory": self.cut_inside(full, pe_inspect.DIR_EXPORT, 10),
                "tls directory": self.cut_inside(full, pe_inspect.DIR_TLS, 20 if is64 else 10),
            }
            for what, data in cases.items():
                with self.subTest(is64=is64, cut=what):
                    path = self.write("cut.dll", data)
                    with self.assertRaises(pe_inspect.PeFormatError):
                        pe_inspect.inspect(path, None)

    def test_every_cut_length_gives_a_report_or_a_pe_format_error(self):
        for is64 in (True, False):
            full = build_image(is64=is64)
            for n in range(0, len(full), 16):
                with self.subTest(is64=is64, length=n):
                    path = self.write("sweep.dll", full[:n])
                    try:
                        pe_inspect.inspect(path, None)
                    except pe_inspect.PeFormatError:
                        pass

    def test_cli_reports_a_truncated_file_and_keeps_going(self):
        cut = self.write("cut.dll", build_image()[:0x58 + 2 + 16])
        good = self.write("good.dll", build_image())
        out = os.path.join(self.dir, "reports")
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            self.assertEqual(pe_inspect.main([cut, good, "--out", out]), 1)
        self.assertIn("cut.dll: not a PE image (truncated optional header)", err.getvalue())
        self.assertEqual(sorted(os.listdir(out)), ["good.dll.json", "good.dll.md"])

    def test_export_counts_past_the_table_cap_do_not_run_away(self):
        full = bytearray(build_image())
        img = pe_inspect.PeImage(bytes(full))
        at = img.offset(img.directory(pe_inspect.DIR_EXPORT)[0])
        struct.pack_into("<I", full, at + 20, 0xFFFFFFFF)  # NumberOfFunctions
        path = self.write("huge_exports.dll", bytes(full))
        exports = pe_inspect.inspect(path, None)["exports"]
        self.assertLessEqual(len(exports), pe_inspect.MAX_ENTRIES + 2)


if __name__ == "__main__":
    unittest.main()
