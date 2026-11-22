"""world_index.scan_rpf joining a resource stored as three zlib streams (synthetic bytes only).

    python -I tests/test_world_index_three_streams.py
"""
import os
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_world_index import WI, make_resource, make_rpf_streams  # noqa: E402


def stream_ends(parts, start=0x40):
    """(start, end) of each compressed part as make_rpf_streams lays them out."""
    out, pos = [], start
    for part in parts:
        pos = (pos + 15) & ~15
        out.append((pos, pos + len(zlib.compress(part))))
        pos = out[-1][1]
    return out


class ThreeStreamJoinTest(unittest.TestCase):
    def test_third_stream_follows_the_second_streams_real_end(self):
        blob = make_resource()
        # The split points put 2 + 15 bytes of alignment padding after the first two streams, so
        # "first stream start + both streams' lengths", rounded up to 16, lands inside the second stream.
        parts = [blob[:0x150], blob[0x150:0x420], blob[0x420:]]
        spans = stream_ends(parts)
        pads = [spans[i + 1][0] - spans[i][1] for i in range(2)]
        self.assertGreaterEqual(sum(pads), 16)  # precondition for the case
        archive = make_rpf_streams(parts)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "three.rpf")
            with open(path, "wb") as f:
                f.write(archive)
            rows, stats = WI.scan_rpf(path)
        self.assertEqual(stats["streams"], 3)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["size"], len(blob))
        self.assertEqual(rows[0]["map_name"], 0x11223344)  # the CMapData block lives in the third stream
        self.assertEqual(stats["scanned_bytes"], sum(e - s for s, e in spans))


if __name__ == "__main__":
    unittest.main()
