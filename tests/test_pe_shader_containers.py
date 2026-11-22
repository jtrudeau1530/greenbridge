"""Unit tests for tools/pe_shader_containers.py, on synthetic containers (no game data, no PE needed).

The containers are built by hand here: a u32 offset table at +32, and each chunk beginning with its fourcc
and a u32 size. The scan's job is to accept exactly that shape and to reject the `DXBC` immediates and error
strings that occur in ordinary code.
"""

import json
import os
import random
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import pe_shader_containers as psc


def container(parts) -> bytes:
    """Assemble a DXBC container from (fourcc, payload) pairs."""
    header = 32 + 4 * len(parts)
    offsets, body, end = [], b"", header
    for fourcc, payload in parts:
        offsets.append(header + len(body))
        body += fourcc + struct.pack("<I", len(payload)) + payload
        end = header + len(body)
        body += bytes(
            (-len(body)) % 4
        )  # chunks are 4-byte aligned; nothing follows the last one
    total = end
    out = b"DXBC" + bytes(16) + struct.pack("<III", 1, total, len(parts))
    out += b"".join(struct.pack("<I", o) for o in offsets)
    return out + body[: total - header]


def xdat(entries, base: int) -> bytes:
    """An XDAT directory whose payload offsets are relative to the DXIL part, as the package has it."""
    head = base + 12 + 12 * len(entries)
    directory, payload = b"", b""
    for tag, value in entries:
        directory += struct.pack("<III", tag, head + len(payload), len(value))
        payload += value
    return b"XDAT" + struct.pack("<II", len(entries), 0) + directory + payload


def dxil_payload(name: str, xbch: bool = True, bitcode: bool = True) -> bytes:
    entries = [(0x12, bytes(16))]
    if xbch:
        entries.append((0x0D, b"XBCH" + bytes(20) + b"payload"))
    if bitcode:
        entries.append((0x07, b"BC\xc0\xde" + bytes(8)))
    prefix = (
        b"e\x00\x05\x00"
        + struct.pack("<I", 24)
        + b"DXIL"
        + b"\x05\x01\x00\x00"
        + struct.pack("<I", 0x70)
        + struct.pack("<I", 0x668)
    )
    return prefix + xdat(entries, len(prefix))


def ildn(name: str) -> bytes:
    path = (
        "X:\\americas\\src\\dev\\rage\\3rdparty\\Microsoft\\XboxBvhBuilder\\Build\\Release\\Output\\"
        f"XboxBvhBuilder\\{name}.pdb"
    ).encode()
    return b"\x00\x00" + struct.pack("<H", len(path)) + path


class ContainerTests(unittest.TestCase):
    def test_scan_reads_one_container(self):
        data = container(
            [
                (b"SFI0", bytes(8)),
                (b"ILDN", ildn("BitonicPreSort64CS")),
                (b"DXIL", dxil_payload("BitonicPreSort64CS")),
            ]
        )
        found, rejected = psc.inspect(b"MZ" + bytes(64) + data, "fake.dll")
        self.assertEqual(rejected, [])
        self.assertEqual(len(found), 1)
        r = found[0]
        self.assertEqual(r["offset"], 66)
        self.assertEqual(r["size"], len(data))
        self.assertEqual(r["name"], "BitonicPreSort64CS")
        self.assertTrue(r["has_xbch"])
        self.assertTrue(r["has_bitcode"])
        self.assertFalse(r["has_fspx"])
        self.assertEqual([c["fourcc"] for c in r["chunks"]], ["SFI0", "ILDN", "DXIL"])

    def test_scan_rejects_a_bare_magic(self):
        data = b"junk" + b"DXBC" + bytes(40)
        found, rejected = psc.inspect(data, "fake.dll")
        self.assertEqual(found, [])
        self.assertEqual(rejected, [4])

    def test_scan_rejects_a_truncated_chunk(self):
        good = container([(b"SFI0", bytes(8)), (b"DXIL", dxil_payload("x"))])
        data = good[:-8]
        found, rejected = psc.inspect(data, "fake.dll")
        self.assertEqual(found, [])
        self.assertEqual(len(rejected), 1)

    def test_fspx_is_reported_and_has_no_xbch(self):
        data = container([(b"SFI0", bytes(8)), (b"ACCC", b"FSPX" + bytes(12))])
        found, _ = psc.inspect(data, "fsr.dll")
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0]["has_fspx"])
        self.assertFalse(found[0]["has_xbch"])
        self.assertFalse(found[0]["has_bitcode"])
        self.assertIsNone(found[0]["name"])

    def test_xdat_directory(self):
        tags = psc.xdat_tags(dxil_payload("x"))
        self.assertEqual([t["tag"] for t in tags], [0x12, 0x0D, 0x07])
        self.assertEqual(tags[1]["magic"], "XBCH")

    def test_module_name_forms(self):
        self.assertEqual(psc.module_name(ildn("ConstructBVH4")), "ConstructBVH4")
        self.assertEqual(psc.module_name(b"\x00\x00\x0a\x00C:\\a\\b.pdb"), "b")
        self.assertIsNone(psc.module_name(b""))

    def test_extract_writes_container_and_index(self):
        data = container(
            [(b"ILDN", ildn("RefitBVH4")), (b"DXIL", dxil_payload("RefitBVH4"))]
        )
        found, _ = psc.inspect(data, "fake.dll")
        with tempfile.TemporaryDirectory() as tmp:
            index = psc.extract(data, found, tmp)
            with open(os.path.join(tmp, index[0]["file"]), "rb") as f:
                self.assertEqual(f.read(), data)
            with open(os.path.join(tmp, "index.json"), "w", encoding="utf-8") as f:
                json.dump(index, f)
            with open(os.path.join(tmp, "index.json"), encoding="utf-8") as f:
                self.assertEqual(
                    json.load(f)[0],
                    {"name": "RefitBVH4", "file": "RefitBVH4.dxbc", "stage": "cs"},
                )

    def test_safe_stem(self):
        self.assertEqual(psc.safe_stem("a b/c", 1), "a_b_c")
        self.assertEqual(psc.safe_stem(None, 7), "container_07")

    def test_random_buffers_raise_only_container_error(self):
        """The scan must never traceback on bytes that are not a container (dcastell1's review)."""
        rng = random.Random(0xDBC)
        for _ in range(2000):
            size = rng.randrange(0, 400)
            data = bytearray(rng.randbytes(size))
            if size >= 4:
                at = rng.randrange(0, size - 3)
                data[at : at + 4] = b"DXBC"
                try:
                    psc.parse_container(bytes(data), at)
                except psc.ContainerError:
                    pass
                except Exception as e:  # noqa: BLE001 - any other type is the failure being asserted
                    self.fail(f"parse_container raised {type(e).__name__}: {e}")
            try:
                psc.inspect(bytes(data), "fuzz.dll")
            except Exception as e:  # noqa: BLE001 - any exception is the failure being asserted
                self.fail(f"inspect raised {type(e).__name__}: {e}")


if __name__ == "__main__":
    unittest.main()
