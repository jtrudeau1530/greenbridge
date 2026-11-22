"""Index a carved RPF9 archive by payload type and XML root.

RPF9 stores files as one or more zlib streams, usually 64 KiB of inflated data per stream. The table of
contents is encrypted, so a carve has offsets and sizes but no names. This tool turns that stream list into
a compact payload index:

    python -I gamedata_index.py <carve_dir> [--csv FILE] [--json FILE]
    python -I gamedata_index.py <archive.rpf> [--csv FILE] [--json FILE] [--carve-dir DIR]

<carve_dir> is an rpf9_carve.py output directory (index.json plus <offset>.bin files). For an .rpf, the
carver is run with --index-only in a temporary directory and the streams are inflated on demand, so no
carved files are written. Pass --carve-dir to reuse an existing index.json.

The rows contain offsets, sizes, a coarse type and an XML root element (a schema/class name). They do not
contain element values, entry names or payload bytes, so the CSV/JSON can be reviewed without copying the
game data. Non-XML payloads follow the 64 KiB block rule; a file whose inflated size is an exact multiple of
64 KiB can merge with the next stream. XML documents close on their root element, so their grouping is exact.
The tool is standard library only and never writes outside the requested output files.
"""
from __future__ import annotations

import argparse
import csv
import json
import mmap
import os
import re
import subprocess
import sys
import tempfile
import zlib
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
CARVER = os.path.join(ROOT, "rpf9_carve.py")
BOM = b"\xef\xbb\xbf"
XML_DECL = re.compile(rb"<\?xml[^>]*\?>")
XML_OPEN = re.compile(rb"<([A-Za-z_][\w.:-]*)")
XML_CLOSE_TAIL = re.compile(rb"</([A-Za-z_][\w.:-]*)>\s*$")
MAGIC = {b"MHIU": "MHIU", b"PAMH": "PAMH", b"SPS1Z": "SPS1Z", b"RBF0": "RBF0", b"PSIN": "PSIN"}


def printable(data: bytes) -> float:
    sample = data[:4096]
    if not sample:
        return 0.0
    good = sum(1 for c in sample if c in (9, 10, 13) or 32 <= c < 127)
    return good / len(sample)


def has_xml_decl(data: bytes) -> bool:
    if data.startswith(BOM) and XML_DECL.match(data[3:]):
        return True
    return bool(XML_DECL.match(data) or XML_DECL.match(data.lstrip(b" \t\r\n")))


def xml_root(data: bytes) -> str:
    """First element name after the XML declaration/comments/PIs, or ""."""
    text = data[3:] if data.startswith(BOM) else data
    text = text.lstrip(b" \t\r\n")
    while True:
        m = XML_DECL.match(text)
        if m:
            text = text[m.end():].lstrip(b" \t\r\n")
            continue
        if text.startswith(b"<!--"):
            end = text.find(b"-->")
            text = text[end + 3:].lstrip(b" \t\r\n") if end >= 0 else b""
            continue
        if text.startswith(b"<?"):
            end = text.find(b"?>")
            text = text[end + 2:].lstrip(b" \t\r\n") if end >= 0 else b""
            continue
        break
    m = XML_OPEN.match(text)
    return m.group(1).decode("ascii", "replace") if m else ""


def tail_root(data: bytes) -> str:
    m = XML_CLOSE_TAIL.search(data[-512:])
    return m.group(1).decode("ascii", "replace") if m else ""


def xml_complete(root: str, data: bytes) -> bool:
    if not root:
        return False
    close = b"</" + root.encode("ascii", "replace") + b">"
    if close in data[-65536:]:
        return True
    return bool(re.search(rb"<" + re.escape(root.encode("ascii", "replace")) + rb"\b[^>]*/>\s*$",
                          data[-4096:]))


def standalone_xml(data: bytes) -> bool:
    """A complete document in one stream, with or without a declaration."""
    root = xml_root(data)
    return bool(root) and tail_root(data) == root and xml_complete(root, data)


def stream_kind(data: bytes, row: dict) -> str:
    if row.get("type") == "bink2" or data.startswith(b"KB2"):
        return "bink2"
    if has_xml_decl(data):
        return "decl"
    if data.startswith(b"RBF0"):
        return "rbf0"
    if data.startswith(b"PSIN"):
        return "psin"
    for magic in MAGIC:
        if data.startswith(magic):
            return "other"
    if printable(data) > 0.95:
        return "text"
    return "binary"


def magic_name(data: bytes) -> str:
    for magic, name in MAGIC.items():
        if data.startswith(magic):
            return name
    return ""


class DirectorySource:
    """Reads .bin/.bk2 files next to an rpf9_carve.py index.json."""

    def __init__(self, directory: str):
        self.directory = directory
        with open(os.path.join(directory, "index.json"), encoding="utf-8") as f:
            self.index = json.load(f)
        self.rows = sorted(self.index["files"], key=lambda r: r["off"])

    def read(self, row: dict) -> bytes:
        ext = ".bk2" if row.get("type") == "bink2" else ".bin"
        path = os.path.join(self.directory, f'{row["off"]:012x}{ext}')
        try:
            with open(path, "rb") as f:
                return f.read()
        except OSError:
            return bytes.fromhex(row.get("head", ""))

    @property
    def summary(self) -> dict:
        return self.index.get("summary", {})


class ArchiveSource:
    """Reads streams on demand from the archive, using an index-only carve."""

    def __init__(self, archive: str, index_dir: str | None = None):
        self.archive = archive
        self._tmp = None
        if index_dir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="gamedata-index-")
            index_dir = self._tmp.name
            r = subprocess.run([sys.executable, "-I", CARVER, archive, index_dir, "--index-only"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode != 0:
                raise RuntimeError((r.stdout + r.stderr).strip() or "rpf9_carve.py failed")
        with open(os.path.join(index_dir, "index.json"), encoding="utf-8") as f:
            self.index = json.load(f)
        self.rows = sorted(self.index["files"], key=lambda r: r["off"])
        self._fh = open(archive, "rb")
        self._map = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def read(self, row: dict) -> bytes:
        if row.get("type") == "bink2":
            return self._map[row["off"]:row["off"] + min(row["stored"], 64)]
        try:
            d = zlib.decompressobj()
            out = d.decompress(self._map[row["off"]:row["off"] + row["stored"]])
            return out + d.flush()
        except zlib.error:
            return bytes.fromhex(row.get("head", ""))

    @property
    def summary(self) -> dict:
        return self.index.get("summary", {})

    def close(self) -> None:
        self._map.close()
        self._fh.close()
        if self._tmp is not None:
            self._tmp.cleanup()


def build_rows(rows: list, source) -> list:
    """Group streams into payloads; XML documents are closed by their root element."""
    out = []
    n = len(rows)
    i = 0
    while i < n:
        data = source.read(rows[i])
        kind = stream_kind(data, rows[i])

        if kind == "decl":
            root = xml_root(data)
            group = [i]
            blob = data
            j = i + 1
            while j < n and not xml_complete(root, blob):
                nxt = source.read(rows[j])
                nk = stream_kind(nxt, rows[j])
                if nk in ("decl", "bink2", "rbf0", "psin", "other", "binary"):
                    break
                blob += nxt
                group.append(j)
                j += 1
            out.append(row_for("xml", rows, group, root=root, header=True,
                               complete=xml_complete(root, blob)))
            i = j
            continue

        if kind == "other" or kind == "bink2":
            group = [i]
            detail = magic_name(data)
            j = i + 1
            while j < n and rows[j - 1]["size"] == 65536:
                nxt = source.read(rows[j])
                nk = stream_kind(nxt, rows[j])
                if nk in ("decl", "bink2", "rbf0", "psin", "other"):
                    break
                group.append(j)
                j += 1
            typ = "bink2" if kind == "bink2" else "other"
            out.append(row_for(typ, rows, group, detail=detail))
            i = j
            continue

        if kind in ("rbf0", "psin", "text", "binary"):
            group = [i]
            j = i + 1
            while j < n and rows[j - 1]["size"] == 65536:
                nxt = source.read(rows[j])
                if standalone_xml(nxt):
                    break
                nk = stream_kind(nxt, rows[j])
                if nk in ("decl", "bink2", "rbf0", "psin", "other"):
                    break
                group.append(j)
                j += 1
            if kind == "text" and b"<" in data and tail_root(b"".join(source.read(rows[x]) for x in group)):
                root = tail_root(b"".join(source.read(rows[x]) for x in group))
                out.append(row_for("xml", rows, group, root=root, header=False, complete=True))
            elif kind == "text" and b"<" in data:
                out.append(row_for("xml", rows, group, complete=False))
            else:
                out.append(row_for(kind, rows, group))
            i = j
            continue

        # An XML continuation stream with no declaration: a tail-only document if it closes a root.
        group = [i]
        blob = data
        j = i + 1
        while j < n and rows[j - 1]["size"] == 65536:
            nxt = source.read(rows[j])
            if stream_kind(nxt, rows[j]) in ("decl", "bink2", "rbf0", "psin", "other", "binary"):
                break
            blob += nxt
            group.append(j)
            j += 1
        root = tail_root(blob)
        out.append(row_for("xml", rows, group, root=root, header=False,
                           complete=bool(root) and xml_complete(root, blob)))
        i = j
    return out


def row_for(typ: str, rows: list, group: list, root: str = "", detail: str = "",
            header: bool = False, complete: bool = True) -> dict:
    first = rows[group[0]]
    return {
        "offset": first["off"],
        "offset_hex": f'{first["off"]:x}',
        "stored": sum(rows[x]["stored"] for x in group),
        "size": sum(rows[x]["size"] for x in group),
        "type": typ,
        "root": root,
        "detail": detail,
        "chunks": len(group),
        "complete": complete,
        "header": header,
    }


def summarize(rows: list, source_summary: dict, target: str) -> dict:
    by_type = defaultdict(lambda: {"files": 0, "streams": 0, "stored": 0, "inflated": 0})
    by_root = defaultdict(lambda: {"files": 0, "inflated": 0})
    for r in rows:
        t = by_type[r["type"]]
        t["files"] += 1
        t["streams"] += r["chunks"]
        t["stored"] += r["stored"]
        t["inflated"] += r["size"]
        if r["root"]:
            by_root[r["root"]]["files"] += 1
            by_root[r["root"]]["inflated"] += r["size"]
    xml = [r for r in rows if r["type"] == "xml"]
    streams = source_summary.get("files", 0)
    if isinstance(streams, list):
        streams = len(streams)
    return {
        "target": target,
        "streams": streams,
        "stored": sum(r["stored"] for r in rows),
        "inflated": sum(r["size"] for r in rows),
        "zero_padding": source_summary.get("zero_padding"),
        "unread_bytes": source_summary.get("unread_bytes"),
        "types": {k: dict(v) for k, v in sorted(by_type.items(), key=lambda kv: -kv[1]["inflated"])},
        "xml_documents": sum(1 for r in xml if r["complete"]),
        "xml_with_header": sum(1 for r in xml if r["header"]),
        "xml_tail_only": sum(1 for r in xml if r["complete"] and not r["header"]),
        "xml_incomplete": sum(1 for r in xml if not r["complete"]),
        "roots": {k: dict(v) for k, v in sorted(by_root.items(), key=lambda kv: (-kv[1]["files"], kv[0]))},
    }


def write_csv(path: str, rows: list) -> None:
    fields = ["offset", "offset_hex", "stored", "size", "type", "root", "detail", "chunks",
              "complete", "header"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", help="carve directory or .rpf archive")
    ap.add_argument("--carve-dir", help="existing rpf9_carve.py index.json directory for an .rpf")
    ap.add_argument("--csv", help="write one row per payload to this CSV")
    ap.add_argument("--json", help="write the summary and rows to this JSON")
    a = ap.parse_args()

    source = None
    try:
        if os.path.isdir(a.target):
            source = DirectorySource(a.target)
        else:
            source = ArchiveSource(a.target, a.carve_dir)
        rows = build_rows(source.rows, source)
        summary = summarize(rows, source.summary, a.target)
    finally:
        if isinstance(source, ArchiveSource):
            source.close()

    if a.csv:
        write_csv(a.csv, rows)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "rows": rows}, f, indent=1)

    print(f'{summary["streams"]} streams, {summary["stored"]:,} stored, '
          f'{summary["inflated"]:,} inflated')
    for name, t in summary["types"].items():
        print(f'  {name:8} {t["files"]:4} payloads {t["streams"]:5} streams '
              f'{t["stored"]:10,} stored {t["inflated"]:12,} inflated')
    print(f'XML: {summary["xml_documents"]} complete documents '
          f'({summary["xml_with_header"]} with a declaration, {summary["xml_tail_only"]} tail-only), '
          f'{summary["xml_incomplete"]} incomplete, {len(summary["roots"])} roots')
    if a.csv:
        print("wrote", a.csv)
    if a.json:
        print("wrote", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
