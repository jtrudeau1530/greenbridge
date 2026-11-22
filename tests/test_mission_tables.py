"""Synthetic tests for tools/mission_tables.py (no game data: every byte below is built here)."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "mission_tables.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import mission_tables as M  # noqa: E402


def record(key, kind, offset, count, col_b=1, band=0):
    """One 32-byte record. The offset sits in the high half of the dword at +16."""
    word = (kind & 0xFFFF) | ((offset & 0xFFFF) << 16)
    return struct.pack("<8I", key, 0, col_b, 0, word, band, (count << 16) | count, 0)


def element(archetype, pos, rot, link, hash_value=None, extra=0):
    """One 64-byte pool element: hash, 3 zeros, pos(x,y,z,x), rot(x,y,z,x), hash, link, extra, 0."""
    head = struct.pack("<4I", archetype, 0, 0, 0)
    pos_bytes = struct.pack("<4f", pos[0], pos[1], pos[2], pos[0])
    rot_bytes = struct.pack("<4f", rot[0], rot[1], rot[2], rot[0])
    tail = struct.pack("<4I", archetype if hash_value is None else hash_value, link, extra, 0)
    return head + pos_bytes + rot_bytes + tail


def blob_with(records, elements):
    """records: [(address, bytes)]; elements: [(address, bytes)]. Returns a zero-padded bytearray."""
    end = max([address + len(data) for address, data in records + elements] + [0])
    blob = bytearray(end)
    for address, data in records + elements:
        blob[address:address + len(data)] = data
    return blob


def good_element(archetype=0x11223344, pos=(1.5, -2.25, 3.75), rot=(0.25, 0.5, -0.75),
                 link=0x55667788, extra=0):
    return element(archetype, pos, rot, link, extra=extra)


class TestRecord(unittest.TestCase):
    def test_big_endian_offset_half(self):
        # The measured bytes of a real record: 03 00 80 07. kind is 3 and the offset is 0x780.
        raw = struct.pack("<8I", 0x3B643C5E, 0, 1, 0, 0x07800003, 0, 0x00010001, 0)
        self.assertEqual(raw[16:20], bytes((0x03, 0x00, 0x80, 0x07)))
        decoded = M.decode_record(raw, 0)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded["key"], 0x3B643C5E)
        self.assertEqual(decoded["kind"], 3)
        self.assertEqual(decoded["poolOffset"], 0x780)   # not 0x8007: the u16 is still little-endian
        self.assertEqual(decoded["count"], 1)
        self.assertEqual(decoded["band"], 0)
        self.assertEqual(decoded["colB"], 1)

    def test_duplicated_count_and_zero_fields_are_required(self):
        self.assertIsNone(M.decode_record(bytes(16), 0))   # shorter than a record
        bad_count = bytearray(record(0x12345678, 1, 0x100, 1))
        struct.pack_into("<I", bad_count, 24, 0x00010002)
        self.assertIsNone(M.decode_record(bytes(bad_count), 0))
        bad_zero = bytearray(record(0x12345678, 1, 0x100, 1))
        struct.pack_into("<I", bad_zero, 28, 1)
        self.assertIsNone(M.decode_record(bytes(bad_zero), 0))

    def test_joaat_folds_case(self):
        self.assertEqual(M.joaat("AlphaBeta"), M.joaat("alphabeta"))
        self.assertEqual(M.joaat("ALPHABETA"), M.joaat("alphabeta"))
        # An independent transcription of Jenkins one-at-a-time, seeded at 0.
        def reference(text):
            value = 0
            for byte in text.lower().encode("latin-1"):
                value = (value + byte) & 0xFFFFFFFF
                value = (value + (value << 10)) & 0xFFFFFFFF
                value ^= value >> 6
            value = (value + (value << 3)) & 0xFFFFFFFF
            value ^= value >> 11
            return (value + (value << 15)) & 0xFFFFFFFF
        for name in ("", "StateOne", "RoadJob", "a_b-c.d"):
            self.assertEqual(M.joaat(name), reference(name))


class TestElement(unittest.TestCase):
    def test_round_trip(self):
        state_one = 0x11111111
        state_two = 0x22222222
        blob = blob_with(
            [(0x100, record(state_one, 7, 0x1C0, 1)),
             (0x120, record(state_two, 7, 0x200, 1))],
            [(0x1C0, good_element(archetype=0xAABBCCDD, link=0x01020304)),
             (0x200, good_element(archetype=0x99887766, link=0x05060708,
                                  pos=(-10.0, 20.0, 30.0), rot=(0.0, 1.0, 2.0)))],
        )
        result = M.parse_blob(bytes(blob), "round-trip.bin")
        self.assertEqual(result["records"], 2)
        self.assertEqual(result["chunks"], 1)
        self.assertEqual(result["state_keys"], 2)
        self.assertEqual(result["entities"], 2)
        self.assertEqual(result["unresolved_records"], 0)
        self.assertEqual(result["skipped_slots"], 0)
        by_archetype = {entity["archetype"]: entity for entity in result["entity_list"]}
        self.assertEqual(set(by_archetype), {0xAABBCCDD, 0x99887766})
        first = by_archetype[0xAABBCCDD]
        self.assertEqual(first["link"], 0x01020304)
        self.assertEqual([round(value, 4) for value in first["pos"]], [1.5, -2.25, 3.75])
        self.assertEqual([round(value, 4) for value in first["rot"]], [0.25, 0.5, -0.75])
        second = by_archetype[0x99887766]
        self.assertEqual([round(value, 4) for value in second["pos"]], [-10.0, 20.0, 30.0])

    def test_mismatched_duplicated_position_is_not_an_entity(self):
        broken = bytearray(good_element(archetype=0xDEADBEEF, link=0x00000001))
        struct.pack_into("<I", broken, 0x1C, 0x7FC00000)  # pos.x copy no longer equals pos.x
        blob = blob_with(
            [(0x100, record(0x11111111, 7, 0x180, 1)),
             (0x120, record(0x22222222, 7, 0x1C0, 1))],
            [(0x180, bytes(broken)),
             (0x1C0, good_element(archetype=0xCAFEBABE, link=0x00000002))],
        )
        result = M.parse_blob(bytes(blob), "mismatch.bin")
        self.assertEqual(result["records"], 2)
        self.assertEqual(result["entities"], 1)
        self.assertEqual(result["skipped_slots"], 1)
        self.assertEqual(result["unresolved_records"], 1)
        self.assertEqual([entity["archetype"] for entity in result["entity_list"]], [0xCAFEBABE])

    def test_mismatched_duplicated_rotation_is_not_an_entity(self):
        broken = bytearray(good_element(archetype=0xDEADBEEF))
        struct.pack_into("<I", broken, 0x2C, 0x40000000)  # rot.x copy no longer equals rot.x
        blob = blob_with(
            [(0x100, record(0x11111111, 7, 0x180, 1)),
             (0x120, record(0x22222222, 7, 0x180, 1))],
            [(0x180, bytes(broken))],
        )
        result = M.parse_blob(bytes(blob), "mismatch-rot.bin")
        self.assertEqual(result["entities"], 0)
        self.assertEqual(result["skipped_slots"], 2)

    def test_extra_tail_hash_does_not_hide_an_entity(self):
        # Some elements carry a second hash in the third-last dword; it must not invalidate the element.
        blob = blob_with(
            [(0x100, record(0x11111111, 7, 0x180, 1)),
             (0x120, record(0x22222222, 7, 0x180, 1))],
            [(0x180, good_element(archetype=0x1234ABCD, link=0x0BADF00D, extra=0xFEEDFACE))],
        )
        result = M.parse_blob(bytes(blob), "extra.bin")
        self.assertEqual(result["entities"], 1)
        self.assertEqual(result["entity_list"][0]["extra"], 0xFEEDFACE)


class TestBrokenInput(unittest.TestCase):
    def test_truncated_element_is_skipped_without_a_crash(self):
        blob = blob_with(
            [(0x100, record(0x11111111, 7, 0x1C0, 1)),
             (0x120, record(0x22222222, 7, 0x1C0, 1))],
            [(0x1C0, good_element())],
        )
        truncated = bytes(blob[:0x1C0 + 40])   # the element loses its last 24 bytes
        result = M.parse_blob(truncated, "truncated.bin")
        self.assertEqual(result["records"], 2)
        self.assertEqual(result["entities"], 0)
        self.assertEqual(result["skipped_slots"], 2)
        self.assertEqual(result["trailing_bytes"], len(truncated) - 0x140)
        self.assertEqual(result["skipped_bytes"], len(truncated) - 2 * M.RECORD_SIZE)

    def test_shorter_than_one_record_is_all_skipped(self):
        result = M.parse_blob(b"\x01" * 20, "tiny.bin")
        self.assertEqual(result["records"], 0)
        self.assertEqual(result["entities"], 0)
        self.assertEqual(result["skipped_bytes"], 20)
        self.assertEqual(result["trailing_bytes"], 20)

    def test_offset_outside_the_file_is_a_record_but_not_an_entity(self):
        blob = blob_with(
            [(0x100, record(0x11111111, 7, 0x180, 1)),
             (0x120, record(0x22222222, 7, 0xFFF0, 1))],
            [(0x180, good_element(archetype=0xABCDEF01, link=0x00000003))],
        )
        result = M.parse_blob(bytes(blob), "outside.bin")
        self.assertEqual(result["records"], 2)
        self.assertEqual(result["state_keys"], 2)
        self.assertEqual(result["entities"], 1)
        self.assertEqual(result["unresolved_records"], 1)
        self.assertEqual(result["entity_list"][0]["archetype"], 0xABCDEF01)
        self.assertIn(0x22222222, {state["key"] for state in result["states"]})


class TestCli(unittest.TestCase):
    def test_directory_is_sorted_and_json_is_machine_readable(self):
        one = bytes(blob_with(
            [(0x100, record(0x11111111, 7, 0x180, 1)),
             (0x120, record(0x22222222, 7, 0x180, 1))],
            [(0x180, good_element())],
        ))
        with tempfile.TemporaryDirectory() as folder:
            for name in ("b.bin", "a.bin"):
                with open(os.path.join(folder, name), "wb") as handle:
                    handle.write(one)
            with open(os.path.join(folder, "notes.txt"), "w", encoding="utf-8") as handle:
                handle.write("not a stream")
            result = M.collect(folder)
            self.assertEqual([record["file"] for record in result], ["a.bin", "b.bin"])
            completed = subprocess.run(
                [sys.executable, "-I", TOOL, folder, "--json"],
                capture_output=True, text=True, check=True,
            )
            payload = json.loads(completed.stdout)
            self.assertEqual([record["file"] for record in payload["files"]], ["a.bin", "b.bin"])
            self.assertEqual(payload["summary"]["files"], 2)
            self.assertEqual(payload["summary"]["records"], 4)
            self.assertEqual(payload["summary"]["entities"], 2)
            self.assertRegex(payload["files"][0]["entity_list"][0]["archetype"], r"^[0-9a-f]{8}$")

    def test_limit_stops_after_the_requested_file_count(self):
        one = bytes(blob_with(
            [(0x100, record(0x11111111, 7, 0x180, 1)),
             (0x120, record(0x22222222, 7, 0x180, 1))],
            [(0x180, good_element())],
        ))
        with tempfile.TemporaryDirectory() as folder:
            for name in ("a.bin", "b.bin", "c.bin"):
                with open(os.path.join(folder, name), "wb") as handle:
                    handle.write(one)
            completed = subprocess.run(
                [sys.executable, "-I", TOOL, folder, "--limit", "2"],
                capture_output=True, text=True, check=True,
            )
            self.assertEqual(completed.stdout.count(" records="), 2)
            self.assertIn("2 files,", completed.stdout)


if __name__ == "__main__":
    unittest.main()
