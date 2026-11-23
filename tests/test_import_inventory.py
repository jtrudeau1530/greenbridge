"""Unit tests for tools/import_inventory.py: the classify() port, the import-list reader, the SHIM_TABLE
cross-reference and the report aggregation.

The PE fixture is built here byte by byte — a minimal PE32+ image importing one function by name and one by
ordinal — so the test needs no game files and no third-party packages.

    python -I tests/test_import_inventory.py
"""
import contextlib
import io
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
SHIMS_DIR = os.path.join(ROOT, "runtime", "shims")

import import_inventory as inv  # noqa: E402

# (dll, name, expected class): every registry.cpp classify() family, plus a negative for each.
CLASSIFY_CASES = [
    ("d3d12.dll", "D3D12CreateDevice", inv.HRESULT),
    ("d3d12.dll", "D3D12SerializeVersionedRootSignature", inv.HRESULT),
    ("d3d12.dll", "D3D12GetDebugInterface", inv.HRESULT),
    ("d3d12.dll", "D3D12EnableExperimentalFeatures", inv.HRESULT),
    ("D3D12_XS.DLL", "D3D12XboxCreateDevice", inv.HRESULT),          # the DLL name is case-insensitive
    ("d3d12_xs.dll", "D3D12XboxCreateSyntheticDevice", inv.HRESULT),
    ("d3d12_xs.dll", "D3D12XboxSetProcessDebugFlags", inv.OTHER),    # left out on purpose
    ("d3d12.dll", "D3D12QueryInterface", inv.OTHER),
    ("dxgi.dll", "CreateDXGIFactory", inv.HRESULT),
    ("dxgi.dll", "CreateDXGIFactory2", inv.HRESULT),
    ("dxgi.dll", "DXGIGetDebugInterface1", inv.HRESULT),
    ("dxgi.dll", "DXGIDeclareAdapterRemovalSupport", inv.HRESULT),
    ("dxgi.dll", "DXGIGetDebugInterface", inv.OTHER),                # only the *1 name is listed
    ("xgameruntime.dll", "XGameRuntimeInitialize", inv.HRESULT),
    ("xgameruntime.dll", "XGameEventWrite", inv.HRESULT),
    ("xgameruntime.dll", "XUserAddAsync", inv.HRESULT),
    ("xgameruntime.dll", "XTaskQueueCreate", inv.HRESULT),
    ("xgameruntime.dll", "XGameSaveGetXcsHandleFoo", inv.HRESULT),
    ("xgameruntime.dll", "XGameUiShowFoo", inv.HRESULT),
    ("xgameruntime.dll", "XTaskQueueDispatch", inv.OTHER),           # kNotHResult: returns bool
    ("xgameruntime.dll", "XUserCloseHandle", inv.OTHER),             # kNotHResult: void
    ("xgameruntime.dll", "XStoreIsLicenseValid", inv.OTHER),
    ("xgameruntime.dll", "XPackageIsPackagedProcess", inv.OTHER),
    ("xgameruntime.dll", "XSystemGetDeviceType", inv.OTHER),         # mixed family, left out
    ("xgameruntime.dll", "XGameRuntimeUninitialize", inv.OTHER),
    ("xaudio2_9.dll", "XAudio2Create", inv.HRESULT),
    ("XAUDIO2_9REDIST.DLL", "XAudio2Create", inv.HRESULT),
    ("xaudio2_9.dll", "#1", inv.HRESULT),                            # package modules import it by ordinal
    ("xaudio2_9redist.dll", "#1", inv.HRESULT),
    ("xaudio2_9.dll", "#8", inv.HRESULT),                            # XAudio2CreateV2_9 on the host copy
    ("xaudio2_9.dll", "#9", inv.HRESULT),
    ("xaudio2_9.dll", "#10", inv.HRESULT),
    ("xaudio2_9.dll", "#11", inv.OTHER),
    ("xaudio2_9redist.dll", "#8", inv.OTHER),                        # redist ordinal table unconfirmed
    ("xaudio2_9.dll", "XAudio2SomethingElse", inv.OTHER),
    ("gameinput.dll", "GameInputCreate", inv.HRESULT),
    ("gameinput.dll", "#1", inv.HRESULT),
    ("gameinput.dll", "#2", inv.OTHER),
    ("gameinput.dll", "GameInputOther", inv.OTHER),
    ("dstorage.dll", "DStorageGetFactory", inv.HRESULT),
    ("dstorage_xs.dll", "DStorageGetFactory", inv.HRESULT),
    ("dstorage.dll", "DStorageEnqueueRequest", inv.OTHER),
    ("kernel32.dll", "XUserAddAsync", inv.OTHER),                    # the DLL matters, not just the name
    ("unknown.dll", "Whatever", inv.OTHER),
    ("xmem.dll", "#7", inv.OTHER),                                   # ordinal imports are "other"
]

TEXT_RVA = 0x1000
TEXT_RAW = 0x200
TEXT_SIZE = 0x1000


def make_pe32plus():
    """A one-section PE32+ image and its place(rva, blob) helper; import tables go under the caller."""
    opt_size = 240
    buf = bytearray(TEXT_RAW + TEXT_SIZE)
    buf[0:2] = b"MZ"
    struct.pack_into("<I", buf, 0x3C, 0x40)
    buf[0x40:0x44] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", buf, 0x44, 0x8664, 1, 0, 0, 0, opt_size, 0x0022)
    opt = 0x58
    struct.pack_into("<H", buf, opt, 0x20B)
    struct.pack_into("<I", buf, opt + 56, 0x2000)                    # SizeOfImage
    struct.pack_into("<I", buf, opt + 60, TEXT_RAW)                  # SizeOfHeaders
    struct.pack_into("<I", buf, opt + 108, 16)                       # NumberOfRvaAndSizes
    sec = opt + opt_size
    buf[sec:sec + 8] = b".text\0\0\0"
    struct.pack_into("<IIII", buf, sec + 8, TEXT_SIZE, TEXT_RVA, TEXT_SIZE, TEXT_RAW)

    def place(rva: int, blob: bytes) -> None:
        at = TEXT_RAW + (rva - TEXT_RVA)
        buf[at:at + len(blob)] = blob

    def directory(entry: int, rva: int, size: int) -> None:
        struct.pack_into("<II", buf, opt + 112 + entry * 8, rva, size)

    return buf, place, directory


def build_pe32plus() -> bytes:
    """An image importing xtest.dll!XTestFunc by name and xmem.dll!#7 by ordinal."""
    buf, place, directory = make_pe32plus()
    directory(1, 0x1100, 40)
    place(0x1100, struct.pack("<IIIII", 0x1200, 0, 0, 0x1300, 0x1240))  # xtest.dll descriptor
    place(0x1114, struct.pack("<IIIII", 0x1280, 0, 0, 0x1320, 0x12C0))  # xmem.dll descriptor
    place(0x1128, bytes(20))                                            # terminator
    place(0x1200, struct.pack("<QQ", 0x1400, 0))
    place(0x1280, struct.pack("<QQ", (1 << 63) | 7, 0))
    place(0x1300, b"xtest.dll\0")
    place(0x1320, b"xmem.dll\0")
    place(0x1400, b"\x00\x00" + b"XTestFunc\0")                         # hint + name
    return bytes(buf)


def write_image(tmp: str, name: str, blob: bytes) -> str:
    path = os.path.join(tmp, name)
    with open(path, "wb") as f:
        f.write(blob)
    return path


class ClassifyTest(unittest.TestCase):
    def test_families(self):
        for dll, name, expected in CLASSIFY_CASES:
            with self.subTest(import_=f"{dll}!{name}"):
                self.assertEqual(inv.classify(dll, name), expected)

    def test_function_name_is_case_sensitive(self):
        # classify() compares the name exactly; only the DLL is lower-cased.
        self.assertEqual(inv.classify("d3d12.dll", "d3d12createdevice"), inv.OTHER)


class ImportListTest(unittest.TestCase):
    def test_parse(self):
        imports, bad = inv.parse_import_lines([
            "d3d12_xs.dll!D3D12XboxCreateDevice",
            "",
            "   ",
            "# a comment",
            " XMem.DLL ! XMemAlloc ",
            "xmem.dll!#7",
        ])
        self.assertEqual(imports, [("d3d12_xs.dll", "D3D12XboxCreateDevice"),
                                   ("xmem.dll", "XMemAlloc"), ("xmem.dll", "#7")])
        self.assertEqual(bad, 0)

    def test_malformed_lines_are_skipped(self):
        imports, bad = inv.parse_import_lines(["no bang here", "!name", "dll!", "ok.dll!Fine"])
        self.assertEqual(imports, [("ok.dll", "Fine")])
        self.assertEqual(bad, 3)

    def test_dll_names_normalise_like_the_loader(self):
        # pe_loader.cpp's dll_key: ".dll" is added when there is no extension, one trailing '.' is
        # dropped, a directory part is stripped — "GameInput" and "gameinput.dll." key gameinput.dll.
        imports, bad = inv.parse_import_lines([
            "GameInput!GameInputCreate",
            "gameinput.dll.!GameInputCreate",
            "dir\\XMem.DLL!XMemAlloc",
        ])
        self.assertEqual(imports, [("gameinput.dll", "GameInputCreate"),
                                   ("gameinput.dll", "GameInputCreate"),
                                   ("xmem.dll", "XMemAlloc")])
        self.assertEqual(bad, 0)

    def test_dll_key_edge_cases(self):
        # The same rules the C++ applies, including the odd corners: only one trailing dot is
        # dropped, and a name that is all extension keeps it.
        self.assertEqual(inv.dll_key("GameInput"), "gameinput.dll")
        self.assertEqual(inv.dll_key("gameinput.dll."), "gameinput.dll")
        self.assertEqual(inv.dll_key("gameinput.."), "gameinput.")
        self.assertEqual(inv.dll_key("api-ms-win-core-file-l2-1-0.dll"),
                         "api-ms-win-core-file-l2-1-0.dll")


class PeParseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.image = os.path.join(cls.tmp.name, "sample.exe")
        with open(cls.image, "wb") as f:
            f.write(build_pe32plus())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_named_and_ordinal_imports(self):
        self.assertEqual(sorted(inv.pe_imports(self.image)),
                         [("xmem.dll", "#7"), ("xtest.dll", "XTestFunc")])

    def test_non_pe_input_is_rejected(self):
        bad = os.path.join(self.tmp.name, "not_pe.bin")
        with open(bad, "wb") as f:
            f.write(b"not a PE file")
        with self.assertRaises(inv.PeFormatError):
            inv.pe_imports(bad)

    def test_delay_imports_are_listed(self):
        # Directory 13, IMAGE_DELAYLOAD_DESCRIPTOR: RvaBased attr set, name/IAT/INT as RVAs — the loader
        # binds these eagerly, so they are part of the image's import surface.
        buf, place, directory = make_pe32plus()
        directory(13, 0x1180, 64)
        place(0x1180, struct.pack("<8I", 1, 0x1300, 0x1400, 0x1240, 0x1200, 0, 0, 0))
        place(0x11A0, bytes(32))                                          # terminator
        place(0x1200, struct.pack("<QQ", 0x1420, 0))                        # INT: one named import
        place(0x1300, b"xdelay.dll\0")
        place(0x1420, b"\x00\x00" + b"XDelayFunc\0")
        path = write_image(self.tmp.name, "delay.exe", bytes(buf))
        self.assertEqual(inv.pe_imports(path), [("xdelay.dll", "XDelayFunc")])

    def test_delay_descriptor_without_rva_attribute_fails(self):
        # Same throw the loader makes: a version-1 descriptor's fields are virtual addresses.
        buf, place, directory = make_pe32plus()
        directory(13, 0x1180, 64)
        place(0x1180, struct.pack("<8I", 0, 0x1300, 0x1400, 0x1240, 0x1200, 0, 0, 0))
        place(0x1300, b"xdelay.dll\0")
        path = write_image(self.tmp.name, "bad_delay.exe", bytes(buf))
        with self.assertRaisesRegex(inv.PeFormatError, "RVA attribute"):
            inv.pe_imports(path)

    def test_descriptor_table_stops_at_name_zero(self):
        # The loader ends the table at the first descriptor with Name == 0, even when other fields are
        # set — the parser must not read past it (string_at(0) decodes the MZ header).
        buf, place, directory = make_pe32plus()
        directory(1, 0x1100, 60)
        place(0x1100, struct.pack("<IIIII", 0x1200, 0, 0, 0x1300, 0x1240))  # xtest.dll descriptor
        place(0x1114, struct.pack("<IIIII", 0x1280, 0, 0, 0, 0x12C0))       # Name == 0: table ends
        place(0x1128, struct.pack("<IIIII", 0x1280, 0, 0, 0x1320, 0x12C0))  # unreachable xmem.dll
        place(0x1200, struct.pack("<QQ", 0x1400, 0))
        place(0x1280, struct.pack("<QQ", (1 << 63) | 7, 0))
        place(0x1300, b"xtest.dll\0")
        place(0x1320, b"xmem.dll\0")
        place(0x1400, b"\x00\x00" + b"XTestFunc\0")
        path = write_image(self.tmp.name, "early_end.exe", bytes(buf))
        self.assertEqual(inv.pe_imports(path), [("xtest.dll", "XTestFunc")])

    def test_unreadable_descriptor_name_fails(self):
        # Where the tool skipped the descriptor, the loader throws on the same image.
        buf, place, directory = make_pe32plus()
        directory(1, 0x1100, 40)
        place(0x1100, struct.pack("<IIIII", 0x1200, 0, 0, 0x3FF0, 0x1240))  # name outside the image
        path = write_image(self.tmp.name, "bad_name.exe", bytes(buf))
        with self.assertRaisesRegex(inv.PeFormatError, "DLL name"):
            inv.pe_imports(path)

    def test_unreadable_thunk_name_fails(self):
        buf, place, directory = make_pe32plus()
        directory(1, 0x1100, 40)
        place(0x1100, struct.pack("<IIIII", 0x1200, 0, 0, 0x1300, 0x1240))
        place(0x1200, struct.pack("<QQ", 0x3FF0, 0))                        # name outside the image
        place(0x1300, b"xtest.dll\0")
        path = write_image(self.tmp.name, "bad_sym.exe", bytes(buf))
        with self.assertRaisesRegex(inv.PeFormatError, "unreadable name"):
            inv.pe_imports(path)

    def test_shared_thunk_table_reports_each_dll(self):
        # Two descriptors pointing at one lookup table parse it once, but the loader binds each
        # descriptor's IAT separately, so both DLLs get the association.
        buf, place, directory = make_pe32plus()
        directory(1, 0x1100, 60)
        place(0x1100, struct.pack("<IIIII", 0x1200, 0, 0, 0x1300, 0x1240))
        place(0x1114, struct.pack("<IIIII", 0x1200, 0, 0, 0x1320, 0x12C0))  # same lookup rva
        place(0x1200, struct.pack("<QQ", 0x1400, 0))
        place(0x1300, b"xtest.dll\0")
        place(0x1320, b"xtest2.dll\0")
        place(0x1400, b"\x00\x00" + b"XTestFunc\0")
        path = write_image(self.tmp.name, "shared.exe", bytes(buf))
        self.assertEqual(inv.pe_imports(path),
                         [("xtest.dll", "XTestFunc"), ("xtest2.dll", "XTestFunc")])

    def test_descriptor_dll_name_normalises_like_the_loader(self):
        # A descriptor naming "GameInput" binds the module the shims register — pe_loader.cpp's
        # dll_key adds the extension; a bare name must not report a "gameinput" host import.
        buf, place, directory = make_pe32plus()
        directory(1, 0x1100, 40)
        place(0x1100, struct.pack("<IIIII", 0x1200, 0, 0, 0x1300, 0x1240))
        place(0x1200, struct.pack("<QQ", 0x1400, 0))
        place(0x1300, b"GameInput\0")
        place(0x1400, b"\x00\x00" + b"GameInputCreate\0")
        path = write_image(self.tmp.name, "bare.exe", bytes(buf))
        self.assertEqual(inv.pe_imports(path), [("gameinput.dll", "GameInputCreate")])


class ShimTableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entries = inv.shim_table_entries(SHIMS_DIR)

    def test_literal_entries(self):
        self.assertIn("xmem.dll!XMemAlloc", self.entries)
        self.assertIn("gameinput.dll!GameInputCreate", self.entries)
        self.assertIn("d3d12_xs.dll!D3D12XboxCreateDevice", self.entries)

    def test_macro_expanded_entries(self):
        # XGR(name) -> {"xgameruntime.dll", #name, ...}
        self.assertIn("xgameruntime.dll!XTaskQueueCreate", self.entries)
        self.assertIn("xgameruntime.dll!XGameRuntimeInitialize", self.entries)
        # FILE_API(dll) -> {dll, "CreateFileA", ...}, ...
        self.assertIn("kernel32.dll!CreateFileW", self.entries)
        self.assertIn("api-ms-win-core-file-l2-1-0.dll!FindNextFileA", self.entries)
        # LOADER(dll)
        self.assertIn("kernelbase.dll!GetProcAddress", self.entries)
        # UCRT(dll) and COMMAND_LINE(dll) expand to several entries each
        self.assertIn("api-ms-win-crt-runtime-l1-1-0.dll!__p___argv", self.entries)
        self.assertIn("kernel32.dll!GetCommandLineW", self.entries)

    def test_unregistered_pair_is_absent(self):
        self.assertNotIn("xmem.dll!XNope", self.entries)
        self.assertNotIn("d3d12.dll!D3D12CreateDevice", self.entries)  # host DLL: classed, not shimmed


class ReportTest(unittest.TestCase):
    IMPORTS = [
        ("d3d12_xs.dll", "D3D12XboxCreateDevice"),      # hresult, implemented
        ("d3d12_xs.dll", "D3D12XboxSetProcessDebugFlags"),  # other, thunk
        ("xgameruntime.dll", "XTaskQueueCreate"),       # hresult, implemented
        ("xgameruntime.dll", "XUserGetId"),             # hresult, thunk
        ("xmem.dll", "XMemAlloc"),                      # other, implemented
        ("kernel32.dll", "ExitProcess"),                # other, implemented
        ("kernel32.dll", "NotAFunction"),               # other, thunk
        ("unknown.dll", "Func"),                        # other, thunk
    ]

    @classmethod
    def setUpClass(cls):
        cls.implemented = inv.shim_table_entries(SHIMS_DIR)
        cls.rows, cls.per_dll = inv.summarize(cls.IMPORTS, cls.implemented)

    def test_rows_classify_and_bind(self):
        by_name = {f"{d}!{n}": (c, b) for d, n, c, b in self.rows}
        self.assertEqual(by_name["d3d12_xs.dll!D3D12XboxCreateDevice"], (inv.HRESULT, inv.IMPLEMENTED))
        self.assertEqual(by_name["d3d12_xs.dll!D3D12XboxSetProcessDebugFlags"], (inv.OTHER, inv.THUNK))
        self.assertEqual(by_name["xgameruntime.dll!XTaskQueueCreate"], (inv.HRESULT, inv.IMPLEMENTED))
        self.assertEqual(by_name["xgameruntime.dll!XUserGetId"], (inv.HRESULT, inv.THUNK))
        self.assertEqual(by_name["xmem.dll!XMemAlloc"], (inv.OTHER, inv.IMPLEMENTED))
        self.assertEqual(by_name["kernel32.dll!ExitProcess"], (inv.OTHER, inv.IMPLEMENTED))
        self.assertEqual(by_name["kernel32.dll!NotAFunction"], (inv.OTHER, inv.HOST))
        self.assertEqual(by_name["unknown.dll!Func"], (inv.OTHER, inv.HOST))

    def test_per_dll_counts(self):
        # [imports, hresult, other, implemented, host, unimplemented thunk]
        self.assertEqual(self.per_dll["d3d12_xs.dll"], [2, 1, 1, 1, 0, 1])
        self.assertEqual(self.per_dll["xgameruntime.dll"], [2, 2, 0, 1, 0, 1])
        self.assertEqual(self.per_dll["xmem.dll"], [1, 0, 1, 1, 0, 0])
        self.assertEqual(self.per_dll["kernel32.dll"], [2, 0, 2, 1, 1, 0])
        self.assertEqual(self.per_dll["unknown.dll"], [1, 0, 1, 0, 1, 0])

    def test_duplicates_collapse(self):
        rows, per_dll = inv.summarize(self.IMPORTS + [self.IMPORTS[0]], self.implemented)
        self.assertEqual(len(rows), len(self.rows))
        self.assertEqual(per_dll, self.per_dll)

    def test_unnormalised_names_route_as_their_key(self):
        # summarize() must route on the loader's dll_key too: "GameInput" is the registered
        # gameinput.dll, not a "gameinput" host module.
        rows, per_dll = inv.summarize([("GameInput", "GameInputCreate")], self.implemented)
        self.assertEqual(rows, [("gameinput.dll", "GameInputCreate", inv.HRESULT, inv.IMPLEMENTED)])
        self.assertEqual(per_dll["gameinput.dll"], [1, 1, 0, 1, 0, 0])

    def test_report_text(self):
        text = inv.format_report(self.IMPORTS, self.implemented)
        self.assertIn("d3d12_xs.dll!D3D12XboxCreateDevice", text)
        self.assertIn("hresult", text)
        self.assertIn("unimplemented thunk", text)
        total = [ln for ln in text.splitlines() if ln.startswith("total")]
        self.assertEqual(len(total), 1)
        # 8 imports, 3 hresult, 5 other, 4 implemented, 2 host-bound, 2 thunks
        self.assertRegex(total[0], r"total\s+8\s+3\s+5\s+4\s+2\s+2$")


class CliTest(unittest.TestCase):
    def test_main_on_a_list_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "imports.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("xmem.dll!XMemAlloc\nunknown.dll!Func\n")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(inv.main([path]), 0)
        text = stdout.getvalue()
        self.assertIn("xmem.dll!XMemAlloc", text)
        self.assertIn("implemented", text)
        self.assertIn("unknown.dll!Func", text)
        self.assertIn("host", text)

    def test_main_reports_bad_lines(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "imports.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("garbage\nxmem.dll!XMemFree\n")
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(inv.main([path]), 0)
        self.assertIn("skipped", stderr.getvalue())
        self.assertIn("xmem.dll!XMemFree", stdout.getvalue())

    def test_main_without_inputs(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(inv.main([]), 1)
        self.assertIn("no inputs", stderr.getvalue())

    def test_main_on_a_pe_image(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sample.exe")
            with open(path, "wb") as f:
                f.write(build_pe32plus())
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(inv.main(["--pe", path]), 0)
        text = stdout.getvalue()
        self.assertIn("xtest.dll!XTestFunc", text)
        self.assertIn("xmem.dll!#7", text)

    def test_main_on_an_unreadable_pe(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.exe")
            with open(path, "wb") as f:
                f.write(b"no")
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                self.assertEqual(inv.main(["--pe", path]), 1)
        self.assertIn("not a readable PE32+ image", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
