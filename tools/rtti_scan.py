"""Recover C++ class names, hierarchies and vtables from an x64 MSVC PE via RTTI.

x64 RTTI uses image-relative (RVA) fields:
  TypeDescriptor             { u64 pVFTable; u64 spare; char name[] = ".?AVClass@ns@@" }
  CompleteObjectLocator      { u32 signature(=1); u32 offset; u32 cdOffset; u32 rvaTypeDescriptor;
                               u32 rvaClassHierarchy; u32 rvaSelf }
  ClassHierarchyDescriptor   { u32 signature; u32 attributes; u32 numBaseClasses; u32 rvaBaseClassArray }
  BaseClassDescriptor        { u32 rvaTypeDescriptor; u32 numContainedBases; i32 mdisp, pdisp, vdisp; u32 attrs;
                               u32 rvaClassHierarchy }
A vtable is preceded (at -8) by a pointer to its CompleteObjectLocator.

Usage:
    python -I rtti_scan.py <image> [--out file.json]
Output: classes (demangled-ish name, type descriptor RVA, base classes) and vtables (RVA, class, offset in the
complete object, number of slots that point into executable sections).
"""
import argparse
import json
import os
import re
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from file_mode import published_mode  # noqa: E402



IMAGE_SCN_MEM_EXECUTE = 0x20000000
PE32PLUS_MAGIC = 0x020B
MAX_PE_IMAGE_SIZE = 0x80000000  # 2 GiB (PE32+ spec maximum)


def load_pe64_image(data: bytes) -> tuple[int, bytes, list[dict]]:
    """Parse a 64-bit PE32+ image using only the Python standard library.

    Returns (image_base, memory_mapped_image, sections) where each section dict contains
    'name', 'virtual_address', 'virtual_size', 'raw_ptr', 'raw_size', and 'characteristics'.
    """
    if len(data) < 0x40 or data[:2] != b"MZ":
        raise ValueError("not a PE image: missing MZ header")
    (e_lfanew,) = struct.unpack_from("<I", data, 0x3C)
    if e_lfanew + 24 > len(data) or data[e_lfanew : e_lfanew + 4] != b"PE\x00\x00":
        raise ValueError("not a PE image: missing PE signature")

    _machine, num_sections, _ts, _sym, _nsym, opt_size, _chars = struct.unpack_from(
        "<HHIIIHH", data, e_lfanew + 4
    )
    opt_off = e_lfanew + 24
    if opt_size < 64 or opt_off + opt_size > len(data):
        raise ValueError("truncated PE optional header")
    (magic,) = struct.unpack_from("<H", data, opt_off)
    if magic != PE32PLUS_MAGIC:
        raise ValueError(f"unsupported PE optional header magic 0x{magic:04X} (expected PE32+ 0x020B)")

    (image_base,) = struct.unpack_from("<Q", data, opt_off + 24)
    size_of_image, size_of_headers = struct.unpack_from("<II", data, opt_off + 56)

    sec_table_off = opt_off + opt_size
    if sec_table_off + num_sections * 40 > len(data):
        raise ValueError("truncated PE section table")

    sections: list[dict] = []
    mapped_size = size_of_image
    for i in range(num_sections):
        sec_off = sec_table_off + i * 40
        raw_name, vsize, vaddr, rsize, rptr, _rel, _ln, _nrel, _nln, chars = struct.unpack_from(
            "<8sIIIIIIHHI", data, sec_off
        )
        span = vsize if vsize > 0 else rsize
        end = vaddr + max(vsize, rsize)
        if end > mapped_size:
            mapped_size = end
        sections.append({
            "name": raw_name.split(b"\x00", 1)[0].decode("latin-1"),
            "virtual_address": vaddr,
            "virtual_size": span,
            "raw_ptr": rptr,
            "raw_size": rsize,
            "characteristics": chars,
        })

    if mapped_size <= 0 or mapped_size > MAX_PE_IMAGE_SIZE:
        raise ValueError(f"invalid PE mapped image size: {mapped_size}")

    img = bytearray(mapped_size)
    hdr_copy = min(size_of_headers, len(data), mapped_size)
    if hdr_copy > 0:
        img[0:hdr_copy] = data[0:hdr_copy]

    for sec in sections:
        rptr = sec["raw_ptr"]
        rsize = sec["raw_size"]
        vaddr = sec["virtual_address"]
        if rsize <= 0 or rptr >= len(data) or vaddr >= mapped_size:
            continue
        copy_len = min(rsize, len(data) - rptr, mapped_size - vaddr)
        if copy_len > 0:
            img[vaddr : vaddr + copy_len] = data[rptr : rptr + copy_len]

    return image_base, bytes(img), sections


def simple_name(mangled: str) -> str:
    """'.?AVFoo@bar@@' -> 'bar::Foo' (enough to read; full demangling isn't needed here)."""
    m = re.match(r"\.\?A[VUW](.*)@@$", mangled)
    if not m:
        return mangled
    parts = [p for p in m.group(1).split("@") if p]
    return "::".join(reversed(parts))


def scan_pe64_rtti(data: bytes) -> dict:
    """Recover x64 MSVC TypeDescriptors, class hierarchies, and vtables from raw PE32+ bytes."""
    base, img, sections = load_pe64_image(data)
    exec_ranges = [
        (s["virtual_address"], min(s["virtual_address"] + s["virtual_size"], len(img)))
        for s in sections
        if s["characteristics"] & IMAGE_SCN_MEM_EXECUTE
    ]

    def is_exec(rva):
        return any(lo <= rva < hi for lo, hi in exec_ranges)

    def u32(rva):
        return struct.unpack_from("<I", img, rva)[0]

    # 1. Type descriptors: '.?AV' / '.?AU' / '.?AW' strings, 16 bytes after the descriptor start.
    tds = {}
    for m in re.finditer(rb"\.\?A[VUW][\x21-\x7e]{1,2000}?@@\x00", img):
        rva = m.start() - 16
        if rva >= 0:
            tds[rva] = m.group(0)[:-1].decode("latin-1")

    # 2. Complete object locators: signature 1 and rvaSelf == own RVA.
    cols = {}
    for m in re.finditer(rb"\x01\x00\x00\x00", img):
        rva = m.start()
        if rva % 4 or rva + 24 > len(img):
            continue
        sig, off, cd, rtd, rchd, rself = struct.unpack_from("<6I", img, rva)
        if rself == rva and rtd in tds:
            cols[rva] = {"offset": off, "td": rtd, "chd": rchd}

    # 3. Class hierarchies.
    classes = {}
    for rva, col in cols.items():
        td = col["td"]
        if td in classes:
            continue
        chd = col["chd"]
        try:
            _, attrs, nbase, rarr = struct.unpack_from("<4I", img, chd)
            bases = []
            for i in range(1, min(nbase, 256)):
                bcd = u32(rarr + 4 * i)
                btd = u32(bcd)
                if btd in tds:
                    bases.append(simple_name(tds[btd]))
        except struct.error:
            attrs, bases = 0, []
        classes[td] = {"name": simple_name(tds[td]), "mangled": tds[td], "td_rva": td,
                       "multiple": bool(attrs & 1), "virtual": bool(attrs & 2), "bases": bases}

    # 4. Vtables: a qword equal to (image base + COL rva), followed by code pointers.
    vtables = []
    col_vas = {base + rva: rva for rva in cols}
    for sec in sections:
        if sec["characteristics"] & IMAGE_SCN_MEM_EXECUTE:
            continue
        lo = sec["virtual_address"]
        hi = min(lo + sec["virtual_size"], len(img))
        for p in range(lo - lo % 8, hi - 8, 8):
            q = struct.unpack_from("<Q", img, p)[0]
            if q in col_vas:
                vt = p + 8
                n = 0
                while vt + 8 * (n + 1) <= hi:
                    f = struct.unpack_from("<Q", img, vt + 8 * n)[0] - base
                    if not (0 <= f < len(img) and is_exec(f)):
                        break
                    n += 1
                col = cols[col_vas[q]]
                vtables.append({"rva": vt, "class": classes.get(col["td"], {}).get("name", "?"),
                                "offset_in_object": col["offset"], "slots": n})

    return {
        "type_descriptors": len(tds),
        "classes": sorted(classes.values(), key=lambda c: c["name"]),
        "vtables": sorted(vtables, key=lambda v: v["rva"]),
    }


def write_report(path, text):
    """Write the report next to its destination and swap it in, so a write that fails part-way (a full disk, a
    locked range) leaves the previous report as it was instead of an empty or truncated file. The result keeps
    the previous report's permission bits, or gets the usual ones for a new file."""
    parent = os.path.dirname(os.path.abspath(path))
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=parent, prefix=".rtti-",
                                         suffix=".tmp", delete=False) as f:
            tmp = f.name
            f.write(text)
        os.chmod(tmp, published_mode(path))
        os.replace(tmp, path)
    except BaseException:
        if tmp is not None:
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out")
    a = ap.parse_args()
    path = a.out or re.sub(r"\.(exe|dll)$", "", a.image, flags=re.I) + ".rtti.json"
    same_path = os.path.normcase(os.path.realpath(a.image)) == os.path.normcase(os.path.realpath(path))
    try:
        same_path = same_path or os.path.samefile(a.image, path)
    except OSError:
        pass  # A missing or inaccessible path is handled by the normal read/write error path.
    if same_path:
        ap.error("output must not overwrite the source image")
    try:
        with open(a.image, "rb") as f:
            data = f.read()
        scanned = scan_pe64_rtti(data)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    out = {"image": a.image, **scanned}
    try:
        write_report(path, json.dumps(out, indent=1))
    except OSError as exc:
        print(f"ERROR: {path}: {exc.strerror or exc}", file=sys.stderr)
        return 1
    print(f"{a.image}: {out['type_descriptors']} type descriptors, {len(out['classes'])} classes with RTTI, {len(out['vtables'])} vtables -> {path}")
    for c in out["classes"][:5]:
        print("  ", c["name"], "<-", ", ".join(c["bases"][:4]))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
