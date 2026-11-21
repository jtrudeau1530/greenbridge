"""Inspect a PE image (exe or dll) for porting work.

Usage:
    python -I pe_inspect.py <image> [--game-dir DIR] [--out DIR]

Writes <out>/<name>.json and <out>/<name>.md. Reads the file only; never runs it.

Standard library only: the headers, the section table, the import, delay-import, export, TLS and relocation
directories are parsed with struct, so the tool runs under `python -I` without third-party packages and is covered
by the same standard-library-only CI job as the other tools.
"""
import argparse
import json
import math
import os
import struct
import sys
from collections import Counter

SYSTEM32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")

MACHINES = {0x8664: "x86-64", 0x14C: "x86", 0xAA64: "ARM64"}
SUBSYSTEMS = {2: "GUI", 3: "Console", 14: "Xbox"}

DIR_EXPORT = 0
DIR_IMPORT = 1
DIR_EXCEPTION = 3
DIR_BASERELOC = 5
DIR_TLS = 9
DIR_DELAY_IMPORT = 13

MAX_ENTRIES = 65536  # a table this long is not a table; stops a malformed image from looping forever


class PeFormatError(ValueError):
    """The file is not a PE image this parser can read."""


def entropy(data: bytes) -> float:
    if not data:
        return 0.0
    n = len(data)
    return -sum(c / n * math.log2(c / n) for c in Counter(data).values())


def _cstring(data: bytes, offset: int, limit: int = 512) -> str | None:
    end = data.find(b"\0", offset, min(len(data), offset + limit))
    if end < 0:
        return None
    return data[offset:end].decode(errors="replace")


class PeImage:
    """The parts of a PE file this tool reports on, read in place from the file bytes."""

    def __init__(self, data: bytes):
        self.data = data
        if len(data) < 0x40 or data[:2] != b"MZ":
            raise PeFormatError("no MZ header")
        self.nt = struct.unpack_from("<I", data, 0x3C)[0]
        if self.nt + 24 > len(data) or data[self.nt:self.nt + 4] != b"PE\0\0":
            raise PeFormatError("no PE signature")
        (self.machine, self.section_count, self.timestamp, _symbol_table, _symbols, opt_size,
         self.characteristics) = struct.unpack_from("<HHIIIHH", data, self.nt + 4)
        opt = self.nt + 24
        if opt + 2 > len(data):
            raise PeFormatError("truncated optional header")
        magic = struct.unpack_from("<H", data, opt)[0]
        if magic not in (0x10B, 0x20B):
            raise PeFormatError(f"unknown optional header magic 0x{magic:x}")
        self.is64 = magic == 0x20B
        self.entry_rva = struct.unpack_from("<I", data, opt + 16)[0]
        self.image_base = struct.unpack_from("<Q" if self.is64 else "<I", data, opt + (24 if self.is64 else 28))[0]
        self.size_of_image = struct.unpack_from("<I", data, opt + 56)[0]
        self.size_of_headers = struct.unpack_from("<I", data, opt + 60)[0]
        self.subsystem = struct.unpack_from("<H", data, opt + 68)[0]
        self.dll_characteristics = struct.unpack_from("<H", data, opt + 70)[0]

        opt_end = min(len(data), opt + opt_size)
        dirs = opt + (112 if self.is64 else 96)
        self.directories = []
        for i in range(16):
            at = dirs + i * 8
            self.directories.append(struct.unpack_from("<II", data, at) if at + 8 <= opt_end else (0, 0))

        # A section's data can be virtual only, so lookups past SizeOfRawData have no file bytes to read.
        self.sections = []
        table = opt + opt_size
        for i in range(self.section_count):
            at = table + i * 40
            if at + 40 > len(data):
                raise PeFormatError("truncated section table")
            vsize, rva, rawsize, rawptr = struct.unpack_from("<IIII", data, at + 8)
            chars = struct.unpack_from("<I", data, at + 36)[0]
            self.sections.append({
                "name": data[at:at + 8].rstrip(b"\0").decode(errors="replace"),
                "rva": rva, "vsize": vsize, "rawsize": rawsize, "rawptr": rawptr,
                "exec": bool(chars & 0x20000000), "write": bool(chars & 0x80000000),
            })

    def directory(self, index: int) -> tuple[int, int]:
        return self.directories[index] if index < len(self.directories) else (0, 0)

    def offset(self, rva: int) -> int | None:
        """File offset of an RVA, or None when no file bytes back it."""
        if rva < self.size_of_headers:
            return rva if rva < len(self.data) else None
        for s in self.sections:
            span = max(s["vsize"], s["rawsize"])
            if s["rva"] <= rva < s["rva"] + span:
                delta = rva - s["rva"]
                if delta >= s["rawsize"]:
                    return None
                at = s["rawptr"] + delta
                return at if at < len(self.data) else None
        return None

    def section_data(self, section: dict) -> bytes:
        at = section["rawptr"]
        if at >= len(self.data):
            return b""
        return self.data[at:at + min(section["rawsize"], len(self.data) - at)]

    def string_at(self, rva: int) -> str | None:
        at = self.offset(rva)
        return None if at is None else _cstring(self.data, at)

    def word(self, at: int) -> int:
        return struct.unpack_from("<Q" if self.is64 else "<I", self.data, at)[0]

    def thunk_names(self, lookup_rva: int) -> list[str]:
        """Names (or '#ordinal') of one lookup table, which is terminated by a zero entry."""
        ordinal_flag = 1 << (63 if self.is64 else 31)
        out = []
        width = 8 if self.is64 else 4
        for i in range(MAX_ENTRIES):
            at = self.offset(lookup_rva + i * width)
            if at is None:
                break
            value = self.word(at)
            if not value:
                break
            if value & ordinal_flag:
                out.append(f"#{value & 0xFFFF}")
                continue
            name = self.string_at((value & 0x7FFFFFFF) + 2)  # skip the 2-byte hint
            out.append(name if name is not None else "?")
        return out


def _imports(img: PeImage, kind: str, index: int, delay: bool = False) -> list[dict]:
    rva, size = img.directory(index)
    if not rva or not size:
        return []
    at = img.offset(rva)
    if at is None:
        return []

    entries = []
    step = 32 if delay else 20
    for i in range(MAX_ENTRIES):
        base = at + i * step
        if base + step > len(img.data):
            break
        if delay:
            (attributes, dll_rva, _handle, iat_rva, int_rva, _bound, _unload,
             _stamp) = struct.unpack_from("<IIIIIIII", img.data, base)
            if not dll_rva:
                break
            # dlattrRva (0x1) is set on every descriptor Windows writes; without it the fields are virtual
            # addresses, which the linker uses for 32-bit images. Convert them so both kinds read the same here.
            if not attributes & 1:
                dll_rva = dll_rva - img.image_base
                iat_rva = iat_rva - img.image_base
                int_rva = int_rva - img.image_base
        else:
            oft, _stamp, _forwarder, dll_rva, iat_rva = struct.unpack_from("<IIIII", img.data, base)
            int_rva = oft
            if not dll_rva:
                break
        if dll_rva < 0:
            continue
        dll = img.string_at(dll_rva)
        if dll is None:
            continue
        lookup = int_rva if int_rva > 0 else iat_rva
        entries.append({"dll": dll, "kind": kind, "functions": img.thunk_names(lookup) if lookup > 0 else []})
    return entries


def _exports(img: PeImage) -> list[str]:
    rva, size = img.directory(DIR_EXPORT)
    if not rva or size < 40:
        return []
    at = img.offset(rva)
    if at is None:
        return []
    (_chars, _stamp, _major, _minor, _name, base, functions, name_count, addr_functions,
     addr_names, addr_ordinals) = struct.unpack_from("<IIHHIIIIIII", img.data, at)

    named, names = set(), []
    for i in range(name_count):
        name_at = img.offset(addr_names + i * 4)
        ord_at = img.offset(addr_ordinals + i * 2)
        if name_at is None or ord_at is None:
            break
        symbol = img.string_at(struct.unpack_from("<I", img.data, name_at)[0])
        names.append(symbol if symbol is not None else "?")
        named.add(struct.unpack_from("<H", img.data, ord_at)[0])
    for i in range(functions):
        if i not in named:
            names.append(f"#{base + i}")
    return names


def _tls_callbacks(img: PeImage) -> int:
    rva, size = img.directory(DIR_TLS)
    if not rva or size < (40 if img.is64 else 24):
        return 0
    at = img.offset(rva)
    if at is None:
        return 0
    callbacks_va = struct.unpack_from("<Q" if img.is64 else "<I", img.data, at + (24 if img.is64 else 12))[0]
    if not callbacks_va or callbacks_va < img.image_base:
        return 0
    callbacks = callbacks_va - img.image_base
    count = 0
    width = 8 if img.is64 else 4
    for i in range(MAX_ENTRIES):
        entry = img.offset(callbacks + i * width)
        if entry is None or not img.word(entry):
            break
        count += 1
    return count


_apiset_cache: dict[str, bool] = {}


def _apiset_resolves(name: str) -> bool:
    """True if this Windows' API-set schema maps the contract to a host DLL (ntdll query; nothing is loaded)."""
    if name not in _apiset_cache:
        _apiset_cache[name] = _query_apiset(name)
    return _apiset_cache[name]


def _query_apiset(name: str) -> bool:
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class UNICODE_STRING(ctypes.Structure):
            _fields_ = [("Length", wintypes.USHORT), ("MaximumLength", wintypes.USHORT), ("Buffer", wintypes.LPWSTR)]

        query = ctypes.WinDLL("ntdll").ApiSetQueryApiSetPresence
        query.argtypes = [ctypes.POINTER(UNICODE_STRING), ctypes.POINTER(ctypes.c_ubyte)]
        query.restype = ctypes.c_long
        base = name[:-4] if name.lower().endswith(".dll") else name
        buf = ctypes.create_unicode_buffer(base)
        text = ctypes.cast(buf, wintypes.LPWSTR)
        us = UNICODE_STRING(len(base) * 2, (len(base) + 1) * 2, text)
        present = ctypes.c_ubyte(0)
        status = query(ctypes.byref(us), ctypes.byref(present))
        return status == 0 and bool(present.value)
    except (OSError, AttributeError, ValueError):
        return False


def classify_dll(name: str, game_dir: str | None) -> str:
    """bundled = ships beside the game, system = present on this Windows, shim = must be reimplemented."""
    if game_dir and os.path.isfile(os.path.join(game_dir, name)):
        return "bundled"
    low = name.lower()
    if low.startswith("api-ms-win-") or low.startswith("ext-ms-win-"):
        # API sets resolve through the OS schema: ask this Windows whether it knows the contract.
        return "system" if _apiset_resolves(name) else "shim"
    if os.path.isfile(os.path.join(SYSTEM32, name)):
        return "system"
    return "shim"


def inspect(path: str, game_dir: str | None) -> dict:
    with open(path, "rb") as f:
        img = PeImage(f.read())
    sections = []
    for s in img.sections:
        data = img.section_data(s)
        e = entropy(data)
        sections.append({
            "name": s["name"],
            "rva": s["rva"], "vsize": s["vsize"], "rawsize": s["rawsize"],
            "exec": s["exec"],
            "write": s["write"],
            "entropy": round(e, 3),
            "looks_encrypted": e > 7.9 and len(data) > 4096,
        })

    imports = _imports(img, "normal", DIR_IMPORT) + _imports(img, "delay", DIR_DELAY_IMPORT, delay=True)
    for entry in imports:
        entry["where"] = classify_dll(entry["dll"], game_dir)

    exports = _exports(img)
    pdata_rva, pdata_size = img.directory(DIR_EXCEPTION)
    reloc_rva, reloc_size = img.directory(DIR_BASERELOC)

    return {
        "file": os.path.basename(path),
        "size": os.path.getsize(path),
        "machine": MACHINES.get(img.machine, hex(img.machine)),
        "timestamp": img.timestamp,
        "is_dll": bool(img.characteristics & 0x2000),
        "subsystem": SUBSYSTEMS.get(img.subsystem, img.subsystem),
        "image_base": hex(img.image_base),
        "size_of_image": img.size_of_image,
        "entry_rva": hex(img.entry_rva),
        "has_relocs": bool(reloc_rva and reloc_size),
        "dynamic_base": bool(img.dll_characteristics & 0x40),
        "sections": sections,
        "imports": imports,
        "export_count": len(exports),
        "exports": exports,
        "pdata_functions": pdata_size // 12 if pdata_rva and pdata_size >= 12 else 0,
        "tls_callbacks": _tls_callbacks(img),
    }


def to_markdown(r: dict) -> str:
    out = [f"# {r['file']}", "",
           f"- {r['machine']}, {'DLL' if r['is_dll'] else 'EXE'}, subsystem {r['subsystem']}, {r['size']:,} bytes",
           f"- image base {r['image_base']}, size {r['size_of_image']:,}, entry RVA {r['entry_rva']}",
           f"- relocations: {'yes' if r['has_relocs'] else 'NO (must load at preferred base)'}; "
           f"ASLR flag: {r['dynamic_base']}",
           f"- .pdata functions: {r['pdata_functions']:,}; TLS callbacks: {r['tls_callbacks']}; exports: {r['export_count']}",
           "", "## Sections", "", "| name | rva | vsize | exec | write | entropy | |", "|---|---|---|---|---|---|---|"]
    for s in r["sections"]:
        flag = "**encrypted?**" if s["looks_encrypted"] else ""
        out.append(f"| {s['name']} | {s['rva']:#x} | {s['vsize']:,} | {s['exec']} | {s['write']} | {s['entropy']} | {flag} |")
    out += ["", "## Imports", ""]
    by_where = Counter()
    for imp in r["imports"]:
        by_where[imp["where"]] += len(imp["functions"])
    out.append(", ".join(f"{k}: {v}" for k, v in sorted(by_where.items())) or "none")
    out += ["", "| dll | kind | resolves | functions |", "|---|---|---|---|"]
    for imp in sorted(r["imports"], key=lambda i: (i["where"] != "shim", i["dll"].lower())):
        out.append(f"| {imp['dll']} | {imp['kind']} | {imp['where']} | {len(imp['functions'])} |")
    shims = [i for i in r["imports"] if i["where"] == "shim"]
    if shims:
        out += ["", "## Needs checking or a shim", ""]
        for imp in shims:
            out.append(f"### {imp['dll']} ({imp['where']})")
            out.append("")
            out.append(", ".join(f"`{f}`" for f in imp["functions"]))
            out.append("")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="+")
    ap.add_argument("--game-dir")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "reports", "pe"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rc = 0
    for img in a.image:
        try:
            r = inspect(img, a.game_dir or os.path.dirname(os.path.abspath(img)))
        except (PeFormatError, OSError) as e:
            print(f"{img}: not a PE image ({e})", file=sys.stderr)
            rc = 1
            continue
        stem = os.path.join(a.out, os.path.splitext(r["file"])[0])
        with open(stem + ".json", "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1)
        with open(stem + ".md", "w", encoding="utf-8") as f:
            f.write(to_markdown(r))
        enc = [s["name"] for s in r["sections"] if s["looks_encrypted"]]
        shim = sum(len(i["functions"]) for i in r["imports"] if i["where"] == "shim")
        print(f"{r['file']:<40} {r['machine']:<7} subsys={r['subsystem']!s:<7} pdata={r['pdata_functions']:<7} "
              f"shim-imports={shim:<5} encrypted={','.join(enc) or '-'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
