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
        with open(os.path.join(out, "sample.json"), encoding="utf-8") as f:
            written = json.load(f)
        self.assertEqual(written["machine"], "x86-64")
        self.assertEqual(written["imports"][0]["dll"], "kernel32.dll")
        with open(os.path.join(out, "sample.md"), encoding="utf-8") as f:
            self.assertIn("# sample.dll", f.read())

    def test_cli_reports_a_non_pe_file(self):
        bad = os.path.join(self.dir, "not_pe_cli.bin")
        with open(bad, "wb") as f:
            f.write(b"MZ" + bytes(10))
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            self.assertEqual(pe_inspect.main([bad, "--out", os.path.join(self.dir, "reports2")]), 1)
        self.assertIn("not a PE image", err.getvalue())


if __name__ == "__main__":
    unittest.main()
