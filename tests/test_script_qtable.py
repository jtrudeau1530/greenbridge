"""script_qtable.py must decode synthetic 0x71717171 object-tree containers without game data.

    python -I tests/test_script_qtable.py

Every fixture is synthetic: 40-byte header + u64 offset table + 16-byte-aligned
struct, wrapper and array records linked by 0x71717171-tagged pointers.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TOOL = os.path.join(TOOLS, "script_qtable.py")
sys.path.insert(0, TOOLS)

import script_qtable  # noqa: E402

Q_PTR_HIGH = 0x71717171 << 32


def _pad16(buf: bytes) -> bytes:
    rem = len(buf) % 16
    return buf if rem == 0 else buf + b"\x00" * (16 - rem)


def build_synthetic_qtable(*, pad_to_bytes: int = 0) -> bytes:
    """Build a valid 7-record, 2-tree synthetic object-tree payload."""
    # Preorder record layout for 7 records (odd count => zero gap after 0x28 + 7*8):
    #   Tree 0 (root 0, hash 0x11111111):
    #     [0] 32B wrapper struct (type 0xAAAA0001, flags 0xFFFF0000) -> ptr array [1]
    #     [1] pointer array (count=2, stride=8, flags=0x10000, elem=0xBBBB0001) -> [2], [3] (64B)
    #     [2] 48B struct (type 0xAAAA0002, leaf or 0 ptrs)
    #     [3] 64B struct (type 0xAAAA0003)
    #   Tree 1 (root 4, hash 0x22222222):
    #     [4] 48B struct (type 0xAAAA0004) -> [5] and [6]
    #     [5] value array (count=3, stride=12, flags=0, elem=0xBBBB0002) -> align16(48+36)=96B
    #     [6] 32B leaf struct (type 0xAAAA0005)
    sizes = [32, 64, 48, 64, 48, 96, 32]
    if pad_to_bytes:
        # Enlarge last leaf value array? Better: add padding elements to record [5] so align16 stays exact.
        extra_elems = max(0, (pad_to_bytes - 0x200) // 12)
        cnt5 = 3 + extra_elems
        sizes[5] = ((48 + cnt5 * 12) + 15) & ~15
    else:
        cnt5 = 3

    table_end = 40 + len(sizes) * 8  # 96 = 0x60, 16-byte aligned because 7 is odd
    offsets = []
    cur = table_end
    for sz in sizes:
        offsets.append(cur)
        cur += sz

    def qptr(target_idx: int) -> int:
        return Q_PTR_HIGH | (offsets[target_idx] + 8)

    recs = [
        # [0] 32B wrapper -> [1]
        struct.pack("<4I2Q", 0x11111111, 0xFFFF0000, 0xAAAA0001, 0, qptr(1), 0),
        # [1] ptr array count=2, stride=8, flags=0x10000 -> 48 + 16 = 64B
        struct.pack("<12I2Q", 0, 0, 0x14, 0, 2, 2, 8, 0x10000, 0, 0, 0xBBBB0001, 0, qptr(2), qptr(3)),
        # [2] 48B struct (leaf)
        struct.pack("<4I", 0, 0, 0xAAAA0002, 0) + b"\x01" * 32,
        # [3] 64B struct (leaf)
        struct.pack("<4I", 0, 0, 0xAAAA0003, 0) + b"\x02" * 48,
        # [4] 48B struct (root 0x22222222) -> qptr(5) at +0x10 and qptr(6) at +0x20
        struct.pack("<4IQ2Q", 0x22222222, 0, 0xAAAA0004, 0, qptr(5), 0, qptr(6)) + bytes(8),
        # [5] value array (cnt5, stride=12, flags=0, elem=0xBBBB0002)
        _pad16(struct.pack("<12I", 0, 0, 0x14, 0, cnt5, cnt5, 12, 0, 0, 0, 0xBBBB0002, 0) + b"\x03" * (cnt5 * 12)),
        # [6] 32B leaf struct
        struct.pack("<4I2Q", 0, 0, 0xAAAA0005, 0, 42, 99),
    ]
    for i, r in enumerate(recs):
        assert len(r) == sizes[i], f"rec {i} len {len(r)} != {sizes[i]}"

    header = struct.pack("<10I", 0, 0, 0, 0, 0, len(offsets), 0, 0, 0, 0)
    offset_table = struct.pack(f"<{len(offsets)}Q", *offsets)
    return header + offset_table + b"".join(recs)


class ScriptQtableTests(unittest.TestCase):
    def test_parse_validates_synthetic_forest_and_invariants(self):
        blob = build_synthetic_qtable()
        rep = script_qtable.parse_qtable(blob)
        self.assertTrue(rep["valid"])
        self.assertEqual(rep["record_count"], 7)
        self.assertEqual(rep["root_count"], 2)
        self.assertEqual(rep["distinct_root_hashes"], 2)
        self.assertTrue(rep["root_bijection_holds"])
        self.assertTrue(rep["contiguous_preorder"])
        self.assertEqual(rep["total_pointers"], 5)
        self.assertEqual(rep["resolved_pointers"], 5)
        self.assertEqual(rep["unresolved_pointers"], 0)
        self.assertEqual(rep["array_records"], 2)
        self.assertEqual(rep["array_size_exact"], 2)
        self.assertEqual(rep["array_pointer_records"], 1)
        self.assertEqual(rep["array_value_records"], 1)
        self.assertEqual(rep["struct_records"], 5)
        self.assertEqual(rep["struct_wrappers"], 1)
        self.assertEqual(rep["array_strides"], {"8": 1, "12": 1})
        self.assertEqual(rep["struct_sizes"], {"32": 2, "48": 2, "64": 1})

    def test_load_joins_split_64k_streams_from_carve_dir(self):
        big = build_synthetic_qtable(pad_to_bytes=70_000)
        self.assertGreater(len(big), 65536)
        chunk0 = big[:65536]
        chunk1 = big[65536:]

        with tempfile.TemporaryDirectory() as tmp:
            carve = os.path.join(tmp, "carve")
            os.makedirs(carve)
            rows = [
                {"off": 0x1000, "stored": 500, "size": len(b"PSIN" + bytes(120)), "head": (b"PSIN" + bytes(12)).hex()},
                {"off": 0x2000, "stored": 10000, "size": len(chunk0), "head": chunk0[:16].hex()},
                {"off": 0x5000, "stored": 2000, "size": len(chunk1), "head": chunk1[:16].hex()},
                {"off": 0x9000, "stored": 400, "size": 128, "head": (b"PSIN" + bytes(12)).hex()},
            ]
            with open(os.path.join(carve, f"{rows[0]['off']:012x}.bin"), "wb") as f:
                f.write(b"PSIN" + bytes(124))
            with open(os.path.join(carve, f"{rows[1]['off']:012x}.bin"), "wb") as f:
                f.write(chunk0)
            with open(os.path.join(carve, f"{rows[2]['off']:012x}.bin"), "wb") as f:
                f.write(chunk1)
            with open(os.path.join(carve, f"{rows[3]['off']:012x}.bin"), "wb") as f:
                f.write(b"PSIN" + bytes(124))
            with open(os.path.join(carve, "index.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump({"summary": {"archive": "synthetic", "files": len(rows)}, "files": rows}, f)

            out_json = os.path.join(tmp, "out", "qtable.json")
            p = subprocess.run(
                [sys.executable, "-I", TOOL, carve, "--root-hash", "0x11111111", "--type-hash", "0xBBBB0002", "--json", out_json],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            with open(out_json, "rb") as f:
                raw = f.read()
            self.assertIn(b"\n", raw)
            self.assertNotIn(b"\r\n", raw)
            rep = json.loads(raw.decode("utf-8"))
            self.assertEqual(rep["location"]["offset_hex"], "0x2000")
            self.assertEqual(rep["location"]["chunks"], 2)
            self.assertEqual(rep["root_query"]["matches"], 1)
            self.assertEqual(rep["root_query"]["trees"][0]["nodes"], 4)
            self.assertEqual(rep["type_query"]["array_matches"], 1)
            self.assertEqual(rep["type_query"]["distinct_trees"], 1)

    def test_detects_corrupt_pointer_and_refuses_input_overwrite(self):
        bad = bytearray(build_synthetic_qtable())
        # Corrupt the last record's u64 at +0x10 into a dangling 0x71717171 pointer
        struct.pack_into("<Q", bad, len(bad) - 16, Q_PTR_HIGH | 0xDEADBEEF)
        rep = script_qtable.parse_qtable(bytes(bad))
        self.assertFalse(rep["valid"])
        self.assertEqual(rep["unresolved_pointers"], 1)

        with tempfile.TemporaryDirectory() as tmp:
            bin_path = os.path.join(tmp, "qtable.bin")
            with open(bin_path, "wb") as f:
                f.write(build_synthetic_qtable())
            p = subprocess.run(
                [sys.executable, "-I", TOOL, bin_path, "--json", bin_path],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            self.assertEqual(p.returncode, 2)
            self.assertIn("overwrite", p.stderr)

            bad_file = os.path.join(tmp, "not_qtable.bin")
            with open(bad_file, "wb") as f:
                f.write(b"NOTQ" * 16)
            p2 = subprocess.run(
                [sys.executable, "-I", TOOL, bad_file],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            self.assertEqual(p2.returncode, 1)


if __name__ == "__main__":
    unittest.main()
