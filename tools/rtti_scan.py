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
import re
import struct
import sys



def simple_name(mangled: str) -> str:
    """'.?AVFoo@bar@@' -> 'bar::Foo' (enough to read; full demangling isn't needed here)."""
    m = re.match(r"\.\?A[VUW](.*)@@$", mangled)
    if not m:
        return mangled
    parts = [p for p in m.group(1).split("@") if p]
    return "::".join(reversed(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out")
    a = ap.parse_args()
    import pefile  # only the CLI needs it; the helpers import without it
    pe = pefile.PE(a.image, fast_load=True)
    base = pe.OPTIONAL_HEADER.ImageBase
    img = pe.get_memory_mapped_image()
    exec_ranges = [(s.VirtualAddress, s.VirtualAddress + s.Misc_VirtualSize) for s in pe.sections
                   if s.Characteristics & 0x20000000]

    def is_exec(rva):
        return any(lo <= rva < hi for lo, hi in exec_ranges)

    def u32(rva):
        return struct.unpack_from("<I", img, rva)[0]

    # 1. Type descriptors: '.?AV' / '.?AU' strings, 16 bytes after the descriptor start.
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
    for sec in pe.sections:
        if sec.Characteristics & 0x20000000:
            continue
        lo, hi = sec.VirtualAddress, sec.VirtualAddress + sec.Misc_VirtualSize
        for p in range(lo - lo % 8, hi - 8, 8):
            q = struct.unpack_from("<Q", img, p)[0]
            if q in col_vas:
                vt = p + 8
                n = 0
                while vt + 8 * (n + 1) <= len(img):
                    f = struct.unpack_from("<Q", img, vt + 8 * n)[0] - base
                    if not (0 <= f < len(img) and is_exec(f)):
                        break
                    n += 1
                col = cols[col_vas[q]]
                vtables.append({"rva": vt, "class": classes.get(col["td"], {}).get("name", "?"),
                                "offset_in_object": col["offset"], "slots": n})
    out = {"image": a.image, "type_descriptors": len(tds), "classes": sorted(classes.values(), key=lambda c: c["name"]),
           "vtables": sorted(vtables, key=lambda v: v["rva"])}
    path = a.out or re.sub(r"\.(exe|dll)$", "", a.image, flags=re.I) + ".rtti.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"{a.image}: {len(tds)} type descriptors, {len(classes)} classes with RTTI, {len(vtables)} vtables -> {path}")
    for c in out["classes"][:5]:
        print("  ", c["name"], "<-", ", ".join(c["bases"][:4]))


if __name__ == "__main__":
    sys.exit(main())
