"""Write stream names, dialogue labels and subtitles into the Ogg Opus files that awc_opus.py exports.

awc_opus.py writes <out>/<bank>/<stream id>.ogg with an empty OpusTags header. This tool fills that header in place,
from two local work products:

``--names``     awc_names.py CSV (source, bank, stream_id, name, rule). A file is matched by its stream id (the file
                name; awc_opus.py's ``<id>~2.ogg`` for a second stream with the same id gets the same name) and its
                bank (the folder path below the Opus folder; if that has no match, the path with its leading folders
                removed one at a time, so the Opus folder may hold several archives' outputs).
``--dialogue``  text_tables.py dialogue CSV (wave, key, label, speaker, text_<lang> ...). A named stream whose name is
                a dialogue wave gets that line's label, speaker and subtitles.

Comments written (RFC 7845 section 5.2; empty values are left out):

    TITLE=<stream name>
    LABEL=<dialogue line label>          SPEAKER=<speaker code from the label>
    SUBTITLE=<English text, or the first language with text>
    SUBTITLE_<LANG>=<text>               one per language column with text

Subtitle markup (``~n~``, ``~sl:...~``, ``~COLOR_...~`` and other ``~...~`` tokens) becomes a space. Only the OpusTags
page (page 1, which holds that packet alone) is replaced, so the audio pages are copied byte for byte and the page
sequence numbers do not change. Streams without a name are left untouched; running the tool again replaces the
tags it wrote. A tags packet larger than one Ogg page (65,024 bytes) is skipped and counted.

Usage:
    python -I opus_tags.py <Opus folder> --names names.csv [--dialogue dialogue.csv] [--source LABEL] [--dry-run]

Standard library only. The CSVs and the tagged files hold game names and text: keep them in your local work folder.
"""
import argparse
import collections
import csv
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from awc_opus import ogg_page, write_file  # noqa: E402

VENDOR = b"leonida "
MARKUP = re.compile(r"~[^~]*~")
SPACES = re.compile(r"\s+")
STREAM_FILE = re.compile(r"^([0-9A-Fa-f]{8})(?:~\d+)?\.ogg$")   # <id>~2 ...: a later stream with the same id
MAX_PACKET = 255 * 255 - 1          # lacing values of one page: 254 full segments and a final short one


def clean(text: str) -> str:
    return SPACES.sub(" ", MARKUP.sub(" ", text)).strip()


def tags_packet(comments) -> bytes:
    """An OpusTags packet: magic, vendor string, then the comments as KEY=value (RFC 7845 section 5.2)."""
    items = [f"{k}={v}".encode("utf-8") for k, v in comments if v]
    out = b"OpusTags" + struct.pack("<I", len(VENDOR)) + VENDOR + struct.pack("<I", len(items))
    return out + b"".join(struct.pack("<I", len(i)) + i for i in items)


def pages(data: bytes):
    """(start, end) of each Ogg page. Raises ValueError on anything that is not a page."""
    at = 0
    while at < len(data):
        if data[at:at + 4] != b"OggS" or at + 27 > len(data):
            raise ValueError(f"no Ogg page at byte {at}")
        segments = data[at + 26]
        end = at + 27 + segments + sum(data[at + 27:at + 27 + segments])
        if end > len(data):
            raise ValueError(f"page at byte {at} runs past the end of the file")
        yield at, end
        at = end


def retag(data: bytes, packet: bytes) -> bytes:
    """`data` with its OpusTags page replaced by one holding `packet`."""
    if len(packet) > MAX_PACKET:
        raise ValueError("tags do not fit in one page")
    it = pages(data)
    try:
        _, (tags_at, tags_end) = next(it), next(it)
    except StopIteration:
        raise ValueError("fewer than two Ogg pages") from None
    body = tags_at + 27 + data[tags_at + 26]
    if data[body:body + 8] != b"OpusTags":
        raise ValueError("page 1 does not start an OpusTags packet")
    segments = data[tags_at + 27:tags_at + 27 + data[tags_at + 26]]
    if not segments or segments[-1] == 255:
        raise ValueError("the OpusTags packet continues past page 1")
    serial, seq = struct.unpack_from("<II", data, tags_at + 14)
    if seq != 1:
        raise ValueError("the OpusTags page is not page 1")
    return data[:tags_at] + ogg_page(serial, 1, 0, [packet], 0x00) + data[tags_end:]


def load_names(path, source=None):
    """{bank label: {29-bit stream id: name}} from awc_names.py CSV."""
    out = collections.defaultdict(dict)
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if source and row.get("source") != source:
                continue
            out[row["bank"].replace("\\", "/")][int(row["stream_id"], 16) & 0x1FFFFFFF] = row["name"]
    return dict(out)


def load_dialogue(path):
    """{wave: row} from text_tables.py dialogue CSV, and the language codes in column order."""
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        langs = [f[5:] for f in reader.fieldnames or [] if f.startswith("text_")]
        return {row["wave"].lower(): row for row in reader}, langs


def comments(name, line, langs):
    out = [("TITLE", name)]
    if line:
        texts = [(lang, clean(line.get(f"text_{lang}", ""))) for lang in langs]
        texts = [(lang, t) for lang, t in texts if t]
        main = dict(texts).get("en") or (texts[0][1] if texts else "")
        out += [("LABEL", line.get("label", "")), ("SPEAKER", line.get("speaker", "")), ("SUBTITLE", main)]
        out += [(f"SUBTITLE_{lang.upper()}", t) for lang, t in texts]
    return out


def bank_names(names, rel_dir):
    """The stream names of the bank in `rel_dir`, trying shorter paths from the left when the full path is unknown."""
    parts = rel_dir.split("/")
    for i in range(len(parts)):
        found = names.get("/".join(parts[i:]))
        if found is not None:
            return found
    return None


def tag_folder(root, names, dialogue=None, langs=(), dry_run=False):
    stats = collections.Counter()
    dialogue = dialogue or {}
    for folder, _dirs, files in os.walk(root):
        rel = os.path.relpath(folder, root).replace(os.sep, "/")
        streams = None
        for fname in sorted(files):
            m = STREAM_FILE.match(fname)
            if not m:
                continue
            stats["ogg files"] += 1
            if streams is None:
                streams = bank_names(names, rel) if rel != "." else None
                streams = streams or {}
            name = streams.get(int(m.group(1), 16) & 0x1FFFFFFF)
            if not name:
                stats["without a name"] += 1
                continue
            line = dialogue.get(name.lower())
            fields = comments(name, line, langs)
            packet = tags_packet(fields)
            path = os.path.join(folder, fname)
            try:
                with open(path, "rb") as fh:
                    data = fh.read()
                new = retag(data, packet)
            except ValueError as e:
                stats["skipped: " + str(e).split(" at byte")[0]] += 1
                continue
            stats["tagged"] += 1
            if line:
                stats["with a dialogue line"] += 1
                if dict(fields).get("SUBTITLE"):
                    stats["with a subtitle"] += 1
            if not dry_run and new != data:
                write_file(path, new)
    return dict(sorted(stats.items()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("opus", help="folder written by awc_opus.py")
    ap.add_argument("--names", required=True, help="awc_names.py CSV")
    ap.add_argument("--dialogue", help="text_tables.py dialogue CSV")
    ap.add_argument("--source", help="only use names rows with this source label")
    ap.add_argument("--dry-run", action="store_true", help="count what would be tagged; write nothing")
    a = ap.parse_args(argv)
    for path in [a.opus, a.names] + ([a.dialogue] if a.dialogue else []):
        if not os.path.exists(path):
            print(f"opus_tags: error: {path}: no such file or folder", file=sys.stderr)
            return 2
    if not os.path.isdir(a.opus):
        print(f"opus_tags: error: {a.opus}: not a folder", file=sys.stderr)
        return 2
    try:
        names = load_names(a.names, a.source)
        dialogue, langs = load_dialogue(a.dialogue) if a.dialogue else ({}, [])
    except (OSError, KeyError, ValueError) as e:
        print(f"opus_tags: error: cannot read a CSV: {e}", file=sys.stderr)
        return 2
    try:
        stats = tag_folder(a.opus, names, dialogue, langs, a.dry_run)
    except OSError as e:
        print(f"opus_tags: error: {e}", file=sys.stderr)
        return 2
    for k, v in stats.items():
        print(f"{k:32} {v:,}")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
