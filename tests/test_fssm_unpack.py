"""Unit tests for tools/fssm_unpack.py covering XBCH carving, FSSM headers, alignment, and CRC bounds.

    python -I tests/test_fssm_unpack.py

Zero game data: all tests use deterministic, synthetic byte sequences constructed in-memory.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import fssm_unpack  # noqa: E402

TOOL = os.path.join(ROOT, "tools", "fssm_unpack.py")
S_ENDPGM = b"\x00\x00\x81\xbf"


def make_fssm(version=0x0101, instructions=None, pad_to=0x200):
    blob = bytearray(pad_to)
    blob[0:4] = b"FSSM"
    struct.pack_into("<H", blob, 4, version)
    if instructions:
        for off, code in instructions:
            blob[off:off + len(code)] = code
    return bytes(blob)


def make_xbch(fssm_data, flags_crc=0, stored_override=None, unpacked_override=None, corrupt_zlib=False):
    compressed = zlib.compress(fssm_data)
    if corrupt_zlib:
        corrupted = bytearray(compressed)
        corrupted[-4:] = b"\x00\x00\x00\x00"
        compressed = bytes(corrupted)
    stored = len(compressed) if stored_override is None else stored_override
    unpacked = len(fssm_data) if unpacked_override is None else unpacked_override
    header = struct.pack("<4sIIIII", b"XBCH", flags_crc, stored, 0, unpacked, 0)
    return header + compressed


def make_xdat(entries, pre_padding=b""):
    count = len(entries)
    hdr_size = 8
    table_len = 12 + 12 * count
    dxil_base_offset = len(pre_padding) + table_len

    table_entries = []
    payloads = bytearray()
    for tag in sorted(entries.keys()):
        payload = entries[tag]
        off = dxil_base_offset + len(payloads)
        table_entries.append(struct.pack("<III", tag, off, len(payload)))
        payloads.extend(payload)

    xdat_header = b"XDAT" + struct.pack("<II", count, hdr_size) + b"".join(table_entries)
    return pre_padding + xdat_header + bytes(payloads)


def make_xdat_list(entries, pre_padding=b""):
    """Like make_xdat, but entries is an ordered list so duplicate tags can be represented."""
    count = len(entries)
    table_len = 12 + 12 * count
    dxil_base_offset = len(pre_padding) + table_len
    table_entries = []
    payloads = bytearray()
    for tag, payload in entries:
        off = dxil_base_offset + len(payloads)
        table_entries.append(struct.pack("<III", tag, off, len(payload)))
        payloads.extend(payload)
    xdat_header = b"XDAT" + struct.pack("<II", count, 8) + b"".join(table_entries)
    return pre_padding + xdat_header + bytes(payloads)


def make_dxbc(chunks, digest=bytes(16), version=1):
    count = len(chunks)
    head_len = 32 + 4 * count
    offsets = []
    body = bytearray()
    for fourcc, payload in chunks:
        offsets.append(head_len + len(body))
        body.extend(fourcc)
        body.extend(struct.pack("<I", len(payload)))
        body.extend(payload)
    total_size = head_len + len(body)
    header = b"DXBC" + digest + struct.pack("<III", version, total_size, count)
    offset_table = b"".join(struct.pack("<I", off) for off in offsets)
    return header + offset_table + bytes(body)


class TestPartsParsing(unittest.TestCase):
    def test_empty_chunks(self):
        container = make_dxbc([])
        self.assertEqual(fssm_unpack.parts(container), {})

    def test_single_chunk(self):
        container = make_dxbc([(b"DXIL", b"sample-dxil-data")])
        parts = fssm_unpack.parts(container)
        self.assertEqual(parts, {b"DXIL": b"sample-dxil-data"})

    def test_multiple_chunks(self):
        chunks = [
            (b"DXIL", b"dxil-content"),
            (b"PRIV", b"priv-content"),
            (b"ISG1", b"isg1-content"),
        ]
        container = make_dxbc(chunks)
        parts = fssm_unpack.parts(container)
        self.assertEqual(parts, {
            b"DXIL": b"dxil-content",
            b"PRIV": b"priv-content",
            b"ISG1": b"isg1-content",
        })

    def test_chunk_with_empty_payload(self):
        container = make_dxbc([(b"DXIL", b"")])
        self.assertEqual(fssm_unpack.parts(container), {b"DXIL": b""})


class TestXdatEntries(unittest.TestCase):
    def test_missing_xdat_magic(self):
        dxil = b"arbitrary-dxil-bytes-without-magic"
        self.assertEqual(fssm_unpack.xdat_entries(dxil), {})

    def test_xdat_with_zero_count(self):
        dxil = b"prefix" + b"XDAT" + struct.pack("<II", 0, 8)
        self.assertEqual(fssm_unpack.xdat_entries(dxil), {})

    def test_xdat_single_entry(self):
        entries = {0x0D: b"xbch-blob-data"}
        dxil = make_xdat(entries)
        parsed = fssm_unpack.xdat_entries(dxil)
        self.assertEqual(parsed, entries)

    def test_xdat_multiple_entries_and_relative_offsets(self):
        entries = {
            0x07: b"dxil-program-header",
            0x0D: b"xbch-fssm-container",
            0x12: b"\xab" * 16,
        }
        pre_padding = b"\x00" * 32
        dxil = make_xdat(entries, pre_padding=pre_padding)
        parsed = fssm_unpack.xdat_entries(dxil)
        self.assertEqual(parsed, entries)

    def test_xdat_zero_size_entry(self):
        entries = {0x01: b"", 0x02: b"non-empty"}
        dxil = make_xdat(entries)
        parsed = fssm_unpack.xdat_entries(dxil)
        self.assertEqual(parsed, entries)


class TestDuplicateXdatPayloads(unittest.TestCase):
    def test_duplicate_tag_keeps_order_and_selects_first_or_last(self):
        first = make_fssm(version=0x0101, instructions=[(0x100, S_ENDPGM)])
        last = make_fssm(version=0x0102, instructions=[(0x100, S_ENDPGM)])
        xdat = make_xdat_list([
            (0x12, b"\x11" * 16),
            (0x0D, make_xbch(first)),
            (0x0D, make_xbch(last)),
        ])
        entries = fssm_unpack.xdat_entries_all(xdat)
        self.assertEqual([tag for tag, _ in entries], [0x12, 0x0D, 0x0D])
        self.assertEqual(entries[1][1], make_xbch(first))
        self.assertEqual(entries[2][1], make_xbch(last))
        self.assertEqual(fssm_unpack.xdat_entries(xdat)[0x0D], make_xbch(last))
        dxbc = make_dxbc([(b"DXIL", xdat)])
        data, _ = fssm_unpack.unpack(dxbc)
        self.assertEqual(data, last)
        data, _ = fssm_unpack.unpack(dxbc, payload="first")
        self.assertEqual(data, first)
        with self.assertRaises(ValueError):
            fssm_unpack.unpack(dxbc, payload="middle")

    def test_cli_payload_first(self):
        first = make_fssm(version=0x0101, instructions=[(0x100, S_ENDPGM)])
        last = make_fssm(version=0x0102, instructions=[(0x100, S_ENDPGM)])
        xdat = make_xdat_list([(0x0D, make_xbch(first)), (0x0D, make_xbch(last))])
        dxbc = make_dxbc([(b"DXIL", xdat)])
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "dup.dxbc"), "wb") as f:
                f.write(dxbc)
            p = subprocess.run([sys.executable, "-I", TOOL, tmp, "--payload", "first"],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            with open(os.path.join(tmp, "fssm", "dup.fssm"), "rb") as f:
                self.assertEqual(f.read(), first)


class TestXbchContainerCarvingAndUnpack(unittest.TestCase):
    def test_unpack_valid_container(self):
        fssm_payload = make_fssm(version=0x0101, instructions=[(0x100, S_ENDPGM)])
        xbch_blob = make_xbch(fssm_payload)
        dxil = make_xdat({0x0D: xbch_blob, 0x12: b"\x11" * 16})
        dxbc = make_dxbc([(b"DXIL", dxil), (b"PRIV", b"priv-data")])

        data, ents = fssm_unpack.unpack(dxbc)
        self.assertEqual(data, fssm_payload)
        self.assertIn(0x0D, ents)
        self.assertIn(0x12, ents)

    def test_unpack_missing_dxil_chunk(self):
        dxbc = make_dxbc([(b"PRIV", b"priv-data-only")])
        data, ents = fssm_unpack.unpack(dxbc)
        self.assertIsNone(data)
        self.assertEqual(ents, {})

    def test_unpack_missing_xdat_magic(self):
        dxbc = make_dxbc([(b"DXIL", b"no-xdat-here")])
        data, ents = fssm_unpack.unpack(dxbc)
        self.assertIsNone(data)
        self.assertEqual(ents, {})

    def test_unpack_missing_xbch_tag(self):
        dxil = make_xdat({0x07: b"other-tag-content"})
        dxbc = make_dxbc([(b"DXIL", dxil)])
        data, ents = fssm_unpack.unpack(dxbc)
        self.assertIsNone(data)
        self.assertEqual(list(ents.keys()), [0x07])

    def test_unpack_invalid_xbch_magic(self):
        bad_blob = b"NOBC" + bytes(20) + zlib.compress(b"FSSM\x00\x00")
        dxil = make_xdat({0x0D: bad_blob})
        dxbc = make_dxbc([(b"DXIL", dxil)])
        data, ents = fssm_unpack.unpack(dxbc)
        self.assertIsNone(data)
        self.assertIn(0x0D, ents)

    def test_unpack_truncated_xbch_header(self):
        bad_blob = b"XBCH" + bytes(8)
        dxil = make_xdat({0x0D: bad_blob})
        dxbc = make_dxbc([(b"DXIL", dxil)])
        with self.assertRaises(struct.error):
            fssm_unpack.unpack(dxbc)

    def test_unpack_size_mismatch_raises_value_error(self):
        fssm_payload = make_fssm()
        xbch_blob = make_xbch(fssm_payload, unpacked_override=len(fssm_payload) + 10)
        dxil = make_xdat({0x0D: xbch_blob})
        dxbc = make_dxbc([(b"DXIL", dxil)])
        with self.assertRaises(ValueError) as ctx:
            fssm_unpack.unpack(dxbc)
        self.assertIn("size mismatch", str(ctx.exception))

    def test_unpack_corrupt_zlib_raises_error(self):
        fssm_payload = make_fssm()
        xbch_blob = make_xbch(fssm_payload, corrupt_zlib=True)
        dxil = make_xdat({0x0D: xbch_blob})
        dxbc = make_dxbc([(b"DXIL", dxil)])
        with self.assertRaises(zlib.error):
            fssm_unpack.unpack(dxbc)


class TestFssmChunkHeadersAndMetadata(unittest.TestCase):
    def test_fssm_magic_and_version(self):
        blob_v257 = make_fssm(version=0x0101)
        self.assertEqual(blob_v257[:4].decode("latin-1"), "FSSM")
        self.assertEqual(struct.unpack_from("<H", blob_v257, 4)[0], 0x0101)

        blob_v512 = make_fssm(version=0x0200)
        self.assertEqual(blob_v512[:4].decode("latin-1"), "FSSM")
        self.assertEqual(struct.unpack_from("<H", blob_v512, 4)[0], 0x0200)


class TestPayloadAlignment(unittest.TestCase):
    def test_sendpgm_aligned_offsets_detected(self):
        blob = bytearray(make_fssm(pad_to=0x100))
        blob[0x20:0x24] = S_ENDPGM
        blob[0x80:0x84] = S_ENDPGM
        ends = [i for i in range(0, len(blob) - 3, 4) if blob[i:i + 4] == S_ENDPGM]
        self.assertEqual(ends, [0x20, 0x80])

    def test_sendpgm_unaligned_offsets_rejected(self):
        blob = bytearray(make_fssm(pad_to=0x100))
        blob[0x21:0x25] = S_ENDPGM
        blob[0x43:0x47] = S_ENDPGM
        ends = [i for i in range(0, len(blob) - 3, 4) if blob[i:i + 4] == S_ENDPGM]
        self.assertEqual(ends, [])

    def test_sendpgm_mixed_aligned_and_unaligned(self):
        blob = bytearray(make_fssm(pad_to=0x100))
        blob[0x11:0x15] = S_ENDPGM
        blob[0x20:0x24] = S_ENDPGM
        blob[0x33:0x37] = S_ENDPGM
        ends = [i for i in range(0, len(blob) - 3, 4) if blob[i:i + 4] == S_ENDPGM]
        self.assertEqual(ends, [0x20])

    def test_sendpgm_short_fssm_bounds(self):
        for size in (0, 1, 2, 3):
            blob = b"FSS"[:size]
            ends = [i for i in range(0, len(blob) - 3, 4) if blob[i:i + 4] == S_ENDPGM]
            self.assertEqual(ends, [])


class TestCrcAndChecksumBounds(unittest.TestCase):
    def test_dxbc_container_digest_bounds(self):
        fssm_data = make_fssm()
        xbch = make_xbch(fssm_data)
        dxil = make_xdat({0x0D: xbch})

        digests = [
            bytes(16),
            b"\xff" * 16,
            b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
        ]
        for digest in digests:
            dxbc = make_dxbc([(b"DXIL", dxil)], digest=digest)
            data, ents = fssm_unpack.unpack(dxbc)
            self.assertEqual(data, fssm_data)

    def test_xdat_hash_tag_0x12_preservation(self):
        hash_16 = b"\xca\xfe\xba\xbe\xde\xad\xbe\xef\x01\x02\x03\x04\x05\x06\x07\x08"
        dxil = make_xdat({0x0D: make_xbch(make_fssm()), 0x12: hash_16})
        dxbc = make_dxbc([(b"DXIL", dxil)])
        _, ents = fssm_unpack.unpack(dxbc)
        self.assertEqual(ents[0x12], hash_16)

    def test_xbch_flags_crc_bounds(self):
        fssm_data = make_fssm()
        test_crcs = [0x00000000, 0xFFFFFFFF, 0x80000000, 0x12345678]
        for crc_val in test_crcs:
            xbch = make_xbch(fssm_data, flags_crc=crc_val)
            dxil = make_xdat({0x0D: xbch})
            dxbc = make_dxbc([(b"DXIL", dxil)])
            data, _ = fssm_unpack.unpack(dxbc)
            self.assertEqual(data, fssm_data)

    def test_zlib_adler32_integrity_rejection(self):
        fssm_data = make_fssm()
        raw_compressed = zlib.compress(fssm_data)
        corrupted_adler = raw_compressed[:-4] + bytes([raw_compressed[-4] ^ 0xFF]) + raw_compressed[-3:]
        header = struct.pack("<4sIIIII", b"XBCH", 0, len(corrupted_adler), 0, len(fssm_data), 0)
        xbch = header + corrupted_adler
        dxil = make_xdat({0x0D: xbch})
        dxbc = make_dxbc([(b"DXIL", dxil)])
        with self.assertRaises(zlib.error):
            fssm_unpack.unpack(dxbc)


class TestMainDirectoryProcessing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_main_cli_processing_and_index_generation(self):
        fssm_1 = make_fssm(version=0x0101, instructions=[(0x40, S_ENDPGM)])
        fssm_2 = make_fssm(version=0x0102, instructions=[(0x20, S_ENDPGM), (0x60, S_ENDPGM)])

        dxbc_1 = make_dxbc([(b"DXIL", make_xdat({0x0D: make_xbch(fssm_1), 0x12: b"\xaa" * 16}))])
        dxbc_2 = make_dxbc([(b"DXIL", make_xdat({0x0D: make_xbch(fssm_2)}))])
        dxbc_no_xbch = make_dxbc([(b"DXIL", make_xdat({0x07: b"dxil-hdr"}))])
        dxbc_mismatch = make_dxbc([(b"DXIL", make_xdat({0x0D: make_xbch(fssm_1, unpacked_override=9999)}))])
        dxbc_bad_zlib = make_dxbc([(b"DXIL", make_xdat({0x0D: make_xbch(fssm_1, corrupt_zlib=True)}))])

        with open(os.path.join(self.src, "shader_a.dxbc"), "wb") as f:
            f.write(dxbc_1)
        with open(os.path.join(self.src, "shader_b.dxbc"), "wb") as f:
            f.write(dxbc_2)
        with open(os.path.join(self.src, "no_xbch.dxbc"), "wb") as f:
            f.write(dxbc_no_xbch)
        with open(os.path.join(self.src, "mismatch.dxbc"), "wb") as f:
            f.write(dxbc_mismatch)
        with open(os.path.join(self.src, "bad_zlib.dxbc"), "wb") as f:
            f.write(dxbc_bad_zlib)
        with open(os.path.join(self.src, "ignored.txt"), "w") as f:
            f.write("not a dxbc container")

        res = subprocess.run([sys.executable, "-I", TOOL, self.src], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

        self.assertIn("2 FSSM blobs", res.stdout)
        self.assertIn("no XBCH", res.stdout)
        self.assertIn("ValueError", res.stdout)
        self.assertIn("error", res.stdout)
        self.assertIn("magic: {'FSSM': 2}", res.stdout)

        fssm_dir = os.path.join(self.src, "fssm")
        self.assertTrue(os.path.isdir(fssm_dir))
        out_files = sorted(os.listdir(fssm_dir))
        self.assertEqual(out_files, ["shader_a.fssm", "shader_b.fssm"])

        with open(os.path.join(fssm_dir, "shader_a.fssm"), "rb") as f:
            self.assertEqual(f.read(), fssm_1)
        with open(os.path.join(fssm_dir, "shader_b.fssm"), "rb") as f:
            self.assertEqual(f.read(), fssm_2)

        index_path = os.path.join(self.src, "fssm_index.json")
        self.assertTrue(os.path.isfile(index_path))
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)

        self.assertEqual(len(index), 2)
        entry_a = next(r for r in index if r["name"] == "shader_a")
        self.assertEqual(entry_a["size"], len(fssm_1))
        self.assertEqual(entry_a["magic"], "FSSM")
        self.assertEqual(entry_a["version"], 0x0101)
        self.assertEqual(entry_a["endpgm"], [0x40])
        self.assertEqual(entry_a["tags"], ["0x12", "0xd"])

        entry_b = next(r for r in index if r["name"] == "shader_b")
        self.assertEqual(entry_b["size"], len(fssm_2))
        self.assertEqual(entry_b["magic"], "FSSM")
        self.assertEqual(entry_b["version"], 0x0102)
        self.assertEqual(entry_b["endpgm"], [0x20, 0x60])
        self.assertEqual(entry_b["tags"], ["0xd"])


if __name__ == "__main__":
    unittest.main()
