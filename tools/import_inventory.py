"""Report how a set of PE imports maps onto the runtime's shim layer.

Usage:
    python -I import_inventory.py <imports.txt> [<imports.txt> ...]
    python -I import_inventory.py --pe <image.exe> [--pe <image.dll>]

An imports file lists one ``dll!function`` per line; the DLL part is case-insensitive, blank lines and ``#``
comments are ignored. ``--pe`` reads the import and delay-import directories of a PE32+ image directly (the
loader binds delay imports eagerly, so both are its import surface); ordinal-only imports report as
``dll!#ordinal``.

Each import gets the return class runtime/shims/registry.cpp's classify() gives it — the same table that decides
what an unimplemented thunk returns under LEONIDA_UNIMPLEMENTED=fail — and is checked against the SHIM_TABLE
registrations in runtime/shims/*.cpp, mirroring the loader's routing: a registered ``dll!name`` binds to the
shim; an unregistered import on a module the shims own binds to the unimplemented-import thunk; anything else
binds to the host Windows DLL (when the host has it — the tool reports the route, not the host's exports).

DLL names are reported under the loader's key — pe_loader.cpp's dll_key(), the LoadLibrary file-name rule —
so an import naming ``GameInput`` or ``gameinput.dll.`` groups and routes as ``gameinput.dll``.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import struct
import sys

HRESULT = "hresult"
OTHER = "other"

# Ported from runtime/shims/registry.cpp: the xgameruntime families classify() treats as HRESULT-returning
# (kXFamilies) and the members the GDK reference shows as void/bool/int (kNotHResult).
NOT_HRESULT = frozenset({
    "XAsyncCancel", "XAsyncComplete",
    "XTaskQueueCloseHandle", "XTaskQueueDispatch", "XTaskQueueGetCurrentProcessTaskQueue",
    "XTaskQueueSetCurrentProcessTaskQueue", "XTaskQueueUnregisterMonitor", "XTaskQueueUnregisterWaiter",
    "XUserCloseHandle", "XUserCloseSignOutDeferralHandle", "XUserCompare", "XUserIsStoreUser",
    "XUserUnregisterForChangeEvent", "XUserUnregisterForDefaultAudioEndpointUtf16Changed",
    "XUserUnregisterForDeviceAssociationChanged",
    "XStoreCloseContextHandle", "XStoreCloseLicenseHandle", "XStoreCloseProductsQueryHandle",
    "XStoreIsLicenseValid", "XStoreProductsQueryHasMorePages", "XStoreUnregisterGameLicenseChanged",
    "XStoreUnregisterPackageLicenseLost",
    "XGameSaveCloseContainer", "XGameSaveCloseProvider", "XGameSaveCloseUpdate",
    "XPackageCloseInstallationMonitorHandle", "XPackageCloseMountHandle", "XPackageGetInstallationProgress",
    "XPackageIsPackagedProcess", "XPackageUninstallPackage", "XPackageUnregisterInstallationProgressChanged",
    "XPackageUnregisterPackageInstalled", "XPackageUpdateInstallationMonitor",
})
X_FAMILIES = ("XAsync", "XTaskQueue", "XUser", "XStore", "XGameSave", "XPackage", "XGameUi")


def classify(dll: str, name: str) -> str:
    """"hresult" or "other", the same split registry.cpp's classify() makes. `dll` is case-insensitive;
    `name` must match exactly, as the registry's string comparisons do."""
    dll = dll.lower()
    if dll in ("d3d12.dll", "d3d12_xs.dll"):
        if (name.startswith(("D3D12Create", "D3D12Serialize"))
                or name in ("D3D12GetDebugInterface", "D3D12EnableExperimentalFeatures",
                            "D3D12XboxCreateDevice", "D3D12XboxCreateSyntheticDevice")):
            return HRESULT
    elif dll == "dxgi.dll":
        if name.startswith("CreateDXGIFactory") or name in ("DXGIGetDebugInterface1",
                                                           "DXGIDeclareAdapterRemovalSupport"):
            return HRESULT
    elif dll in ("xaudio2_9.dll", "xaudio2_9redist.dll"):
        if name == "XAudio2Create" or name == "#1":
            return HRESULT
        # Host xaudio2_9.dll ordinals 8-10 are XAudio2CreateV2_9 / XAudio2CreateWithVersionInfo /
        # XAudio2CreateWithSharedContexts (measured on the System32 export table); the redist's ordinal
        # table is not confirmed, so only the system copy gets them — same split as registry.cpp.
        if dll == "xaudio2_9.dll" and name in ("#8", "#9", "#10"):
            return HRESULT
    elif dll == "gameinput.dll":
        if name == "GameInputCreate" or name == "#1":
            return HRESULT
    elif dll in ("dstorage.dll", "dstorage_xs.dll"):
        if name == "DStorageGetFactory":
            return HRESULT
    elif dll == "xgameruntime.dll":
        if name in ("XGameRuntimeInitialize", "XGameEventWrite"):
            return HRESULT
        if name in NOT_HRESULT:
            return OTHER
        if name.startswith(X_FAMILIES):
            return HRESULT
    return OTHER


def dll_key(name: str) -> str:
    """Port of pe_loader.cpp's dll_key(): the LoadLibrary file-name rule on a module name — basename only,
    lower-case, one trailing '.' dropped, '.dll' added when the name has no extension. An import naming
    "GameInput" or "gameinput.dll." resolves to gameinput.dll, the module the shims register."""
    name = name[max(name.rfind("\\"), name.rfind("/")) + 1:].lower()
    if name.endswith("."):
        return name[:-1]
    return name if "." in name else name + ".dll"


def parse_import_lines(lines) -> tuple[list[tuple[str, str]], int]:
    """(dll, name) pairs from `dll!function` lines, and the count of lines skipped as malformed. The DLL part
    comes back under the loader's dll_key."""
    imports, bad = [], 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        dll, bang, name = line.partition("!")
        if not bang or not dll.strip() or not name.strip():
            bad += 1
            continue
        imports.append((dll_key(dll.strip()), name.strip()))
    return imports, bad


class PeFormatError(ValueError):
    """The file is not a PE32+ image this parser can read."""


MAX_ENTRIES = 65536  # a table this long is not a table; stops a malformed image from looping forever
MAX_TOTAL_ENTRIES = 262144  # bounds the table entries parsed and the (dll, name) pairs emitted


def _cstring(data: bytes, offset: int, limit: int = 4097) -> str | None:
    end = data.find(b"\0", offset, min(len(data), offset + limit))
    if end < 0:
        return None
    return data[offset:end].decode(errors="replace")


def pe_imports(path: str) -> list[tuple[str, str]]:
    """(dll, name) pairs from a PE32+ image's import and delay-import directories, matching the tables
    pe_loader.cpp's bind_imports walks. Ordinal-only imports come back as '#ordinal'."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 0x40 or data[:2] != b"MZ":
        raise PeFormatError("no MZ header")
    nt = struct.unpack_from("<I", data, 0x3C)[0]
    if nt + 24 > len(data) or data[nt:nt + 4] != b"PE\0\0":
        raise PeFormatError("no PE signature")
    (_machine, section_count, _ts, _symtab, _syms, opt_size, _chars) = struct.unpack_from(
        "<HHIIIHH", data, nt + 4)
    opt = nt + 24
    if opt + opt_size > len(data) or opt_size < 116:
        raise PeFormatError("truncated optional header")
    magic = struct.unpack_from("<H", data, opt)[0]
    if magic != 0x20B:
        raise PeFormatError(f"not a PE32+ image (magic 0x{magic:x})")
    size_of_headers = struct.unpack_from("<I", data, opt + 60)[0]
    dir_count = struct.unpack_from("<I", data, opt + 108)[0]

    sections = []
    table = opt + opt_size
    for i in range(section_count):
        at = table + i * 40
        if at + 40 > len(data):
            raise PeFormatError("truncated section table")
        vsize, rva, rawsize, rawptr = struct.unpack_from("<IIII", data, at + 8)
        sections.append((vsize, rva, rawsize, rawptr))

    def offset(rva: int) -> int | None:
        """File offset of an RVA, or None when no file bytes back it."""
        if rva < size_of_headers:
            return rva if rva < len(data) else None
        for vsize, srva, rawsize, rawptr in sections:
            span = vsize or rawsize
            if srva <= rva < srva + span:
                if rva - srva >= min(rawsize, span):
                    return None
                at = rawptr + rva - srva
                return at if at < len(data) else None
        return None

    def string_at(rva: int) -> str | None:
        at = offset(rva)
        return None if at is None else _cstring(data, at)

    out = []
    tables = {}  # lookup-table RVA -> symbol names, parsed once per table
    total = 0

    def walk_table(lookup: int) -> list[str]:
        """The symbol names one lookup table lists. Descriptors may share a table; the loader still binds
        each descriptor's IAT separately, so the names emit per DLL, but the table parses only once."""
        nonlocal total
        names = tables.get(lookup)
        if names is not None:
            return names
        names = []
        for j in range(MAX_ENTRIES):
            entry = offset(lookup + j * 8)
            if entry is None or entry + 8 > len(data):
                break
            value = struct.unpack_from("<Q", data, entry)[0]
            if not value:
                break
            total += 1
            if total > MAX_TOTAL_ENTRIES:
                raise PeFormatError(f"more than {MAX_TOTAL_ENTRIES} import entries")
            if value & (1 << 63):
                names.append(f"#{value & 0xFFFF}")
                continue
            name = string_at((value & 0x7FFFFFFF) + 2)  # skip the 2-byte hint
            if name is None:
                raise PeFormatError("an import thunk points at an unreadable name")
            names.append(name)
        tables[lookup] = names
        return names

    def walk_thunks(dll: str, lookup: int) -> None:
        if not lookup:
            return
        names = walk_table(lookup)
        if len(out) + len(names) > MAX_TOTAL_ENTRIES:
            raise PeFormatError(f"more than {MAX_TOTAL_ENTRIES} import entries")
        out.extend((dll, name) for name in names)

    # Data directory entry 1. The loader stops the table at the first descriptor with Name == 0 and fails on an
    # unreadable name; read_string throws there.
    if dir_count > 1 and opt + 120 <= len(data):
        imp_rva, _imp_size = struct.unpack_from("<II", data, opt + 112 + 8)
        base = offset(imp_rva) if imp_rva else None
        if base is not None:
            for i in range(MAX_ENTRIES):
                at = base + i * 20  # IMAGE_IMPORT_DESCRIPTOR
                if at + 20 > len(data):
                    break
                oft, _stamp, _forwarder, name_rva, iat_rva = struct.unpack_from("<IIIII", data, at)
                if not name_rva:
                    break
                dll = string_at(name_rva)
                if dll is None:
                    raise PeFormatError("an import descriptor points at an unreadable DLL name")
                walk_thunks(dll_key(dll), oft if oft else iat_rva)

    # Data directory entry 13: delay imports, which the loader binds eagerly — the same checks it makes on an
    # IMAGE_DELAYLOAD_DESCRIPTOR (32 bytes; version-2 fields are RVAs).
    if dir_count > 13 and opt + 112 + 14 * 8 <= len(data):
        dd_rva, _dd_size = struct.unpack_from("<II", data, opt + 112 + 13 * 8)
        dbase = offset(dd_rva) if dd_rva else None
        if dbase is not None:
            for i in range(MAX_ENTRIES):
                at = dbase + i * 32
                if at + 32 > len(data):
                    break
                (attrs, name_rva, _module, iat_rva, int_rva,
                 _bound, _unload, _stamp) = struct.unpack_from("<8I", data, at)
                if not name_rva:
                    break
                if not (attrs & 1):
                    raise PeFormatError("a delay-import descriptor without the RVA attribute")
                if not iat_rva or not int_rva:
                    raise PeFormatError("a delay-import descriptor without an address or name table")
                dll = string_at(name_rva)
                if dll is None:
                    raise PeFormatError("a delay-import descriptor points at an unreadable DLL name")
                walk_thunks(dll_key(dll), int_rva)
    return out


# ---- SHIM_TABLE cross-reference ------------------------------------------------------------------------------

STRING = re.compile(r'"((?:[^"\\]|\\.)*)"')
ENTRY_GROUP = re.compile(r"\{([^{}]*)\}")
TABLE_RE = re.compile(r"\bSHIM_TABLE\s*\(")
MACRO_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
# Only function-like macros with one parameter: the shim files build their entry lists from helpers like
# FILE_API(dll) and XGR(name). SHIM_TABLE(id, ...) itself takes two and is matched by TABLE_RE instead.
DEFINE_RE = re.compile(
    r"(?m)^\s*#\s*define\s+([A-Za-z_]\w*)\s*\(\s*([A-Za-z_]\w*)\s*\)[ \t]*((?:[^\n]*\\[ \t]*\r?\n)*[^\n]*)")


def _table_arguments(text: str) -> list[str]:
    """The argument list of each SHIM_TABLE(...) call in a shim source, comments removed."""
    args = []
    pos = 0
    while True:
        m = TABLE_RE.search(text, pos)
        if not m:
            return args
        i = m.end()  # just past '('
        depth, in_str = 1, False
        out = []
        while i < len(text):
            c = text[i]
            if in_str:
                out.append(c)
                if c == "\\" and i + 1 < len(text):
                    out.append(text[i + 1])
                    i += 2
                    continue
                if c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
                out.append(c)
            elif text.startswith("//", i):
                end = text.find("\n", i)
                i = len(text) if end < 0 else end
                continue
            elif text.startswith("/*", i):
                end = text.find("*/", i + 2)
                i = len(text) if end < 0 else end + 2
                continue
            elif c == "(":
                depth += 1
                out.append(c)
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
                out.append(c)
            else:
                out.append(c)
            i += 1
        args.append("".join(out))
        pos = i + 1


def _expand_macros(text: str, macros: dict[str, tuple[str, str]], limit: int = 16) -> str:
    """SHIM_TABLE argument text with the file's one-parameter entry macros applied."""
    for _ in range(limit):
        found, out, pos = False, [], 0
        for m in MACRO_CALL.finditer(text):
            if m.group(1) not in macros:
                continue
            open_paren = m.end() - 1
            close = _balanced(text, open_paren)
            if close < 0:
                continue
            param, body = macros[m.group(1)]
            arg = text[open_paren + 1:close].strip()
            expanded = body.replace("#" + param, '"' + arg.strip('"').strip() + '"')
            expanded = re.sub(r"\b" + re.escape(param) + r"\b", arg, expanded)
            out.append(text[pos:m.start()])
            out.append(expanded)
            pos = close + 1
            found = True
        if not found:
            return text
        text = "".join(out) + text[pos:]
    return text


def _balanced(text: str, start: int) -> int:
    """Index of the ')' matching the '(' at `start`, or -1."""
    depth, in_str = 0, False
    i = start
    while i < len(text):
        c = text[i]
        if in_str:
            if c == "\\":
                i += 1
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def shim_table_entries(shims_dir: str) -> set[str]:
    """The "dll!name" keys the SHIM_TABLE registrations in shims_dir's .cpp files implement."""
    entries = set()
    for path in sorted(glob.glob(os.path.join(shims_dir, "*.cpp"))):
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        macros = {}
        for name, param, body in DEFINE_RE.findall(text):
            macros[name] = (param, re.sub(r"\\[ \t]*\r?\n", " ", body))
        for args in _table_arguments(text):
            body = args.split(",", 1)[1] if "," in args else ""
            for group in ENTRY_GROUP.findall(_expand_macros(body, macros)):
                strings = STRING.findall(group)
                if len(strings) >= 2:
                    entries.add(strings[0].lower() + "!" + strings[1])
    return entries


IMPLEMENTED = "implemented"
HOST = "host"
THUNK = "unimplemented thunk"

# Ported from registry.cpp's owns_dll(): the modules that always stay on the host even though the shim
# table registers names for them, and the API-set prefixes it rules out before the table is consulted.
HOST_DLLS = frozenset({
    "kernel32.dll", "kernelbase.dll", "msvcrt.dll", "ucrtbase.dll", "ucrtbased.dll",
    "vulkan-1.dll", "ntdll.dll", "user32.dll",
})


def owned_dlls(implemented: set[str]) -> frozenset[str]:
    """The DLLs registry.cpp's owns_dll() calls shim-owned: any SHIM_TABLE entry exists for them and they
    are not the host-bound system modules or API sets it excludes."""
    owned = {key.split("!", 1)[0] for key in implemented}
    return frozenset(d for d in owned
                     if d not in HOST_DLLS and not d.startswith(("api-ms-", "ext-ms-")))


def summarize(imports: list[tuple[str, str]], implemented: set[str]):
    """(rows, per_dll): rows are (dll, name, class, binding) sorted and deduplicated on the loader's
    dll_key, where binding is IMPLEMENTED, HOST or THUNK, mirroring the loader's routing order; per_dll
    maps a dll to [imports, hresult, other, implemented, host, unimplemented thunk]."""
    owned = owned_dlls(implemented)
    rows = []
    for dll, name in sorted({(dll_key(d), n) for d, n in imports}):
        cls = classify(dll, name)
        if f"{dll}!{name}" in implemented:
            binding = IMPLEMENTED
        elif dll in owned:
            binding = THUNK
        else:
            binding = HOST
        rows.append((dll, name, cls, binding))
    per_dll: dict[str, list[int]] = {}
    for dll, _name, cls, binding in rows:
        c = per_dll.setdefault(dll, [0, 0, 0, 0, 0, 0])
        c[0] += 1
        c[1] += cls == HRESULT
        c[2] += cls == OTHER
        c[3] += binding == IMPLEMENTED
        c[4] += binding == HOST
        c[5] += binding == THUNK
    return rows, per_dll


def format_report(imports: list[tuple[str, str]], implemented: set[str]) -> str:
    """The per-import and per-DLL report for (dll, name) pairs against the shim table."""
    rows, per_dll = summarize(imports, implemented)

    out = []
    width = max((len(d) + len(n) + 1 for d, n, _c, _b in rows), default=6)
    for dll, name, cls, binding in rows:
        out.append(f"{dll + '!' + name:<{width}}  {cls:<7}  {binding}")
    out.append("")

    head = ("dll", "imports", "hresult", "other", "implemented", "host", "unimplemented")
    dw = max([len(d) for d in per_dll] + [len(head[0]), len("total")])
    out.append(f"{head[0]:<{dw}}  {head[1]:>7}  {head[2]:>7}  {head[3]:>5}  {head[4]:>11}  "
               f"{head[5]:>4}  {head[6]:>13}")
    totals = [0, 0, 0, 0, 0, 0]
    for dll in sorted(per_dll):
        n, hresult, other, impl, host, thunk = per_dll[dll]
        totals = [a + b for a, b in zip(totals, per_dll[dll])]
        out.append(f"{dll:<{dw}}  {n:>7}  {hresult:>7}  {other:>5}  {impl:>11}  {host:>4}  {thunk:>13}")
    out.append(f"{'total':<{dw}}  {totals[0]:>7}  {totals[1]:>7}  {totals[2]:>5}  "
               f"{totals[3]:>11}  {totals[4]:>4}  {totals[5]:>13}")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="import_inventory.py",
                                 description="Report how PE imports map onto the runtime's shim layer.")
    ap.add_argument("imports", nargs="*", metavar="LIST",
                    help="text file(s) of dll!function lines")
    ap.add_argument("--pe", action="append", default=[], metavar="IMAGE",
                    help="read imports from a PE32+ image")
    ap.add_argument("--shims", metavar="DIR",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runtime", "shims"),
                    help="directory of shim .cpp files to cross-reference")
    a = ap.parse_args(argv)
    if not a.imports and not a.pe:
        print("no inputs: give an imports file or --pe IMAGE", file=sys.stderr)
        return 1

    imports = []
    rc = 0
    for path in a.imports:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                parsed, bad = parse_import_lines(f)
        except OSError as e:
            print(f"{path}: cannot read ({e.strerror or e})", file=sys.stderr)
            rc = 1
            continue
        if bad:
            print(f"{path}: {bad} line(s) without dll!name skipped", file=sys.stderr)
        imports.extend(parsed)
    for path in a.pe:
        try:
            imports.extend(pe_imports(path))
        except (PeFormatError, OSError) as e:
            print(f"{path}: not a readable PE32+ image ({e})", file=sys.stderr)
            rc = 1

    implemented = shim_table_entries(a.shims)
    print(format_report(imports, implemented), end="")
    return rc


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
