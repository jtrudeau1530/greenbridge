"""tools/bvh_gpu_smoke.py without a GPU or game files: the heap image, the word count and the judging of runs, with
a fake vk_compute_test in place of the real one."""
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import bvh_gpu_smoke as smoke  # noqa: E402


def fake_runner(behaviour):
    """A runner that writes the heap image plus per-program changes to the dump, as vk_compute_test's VKC_DUMP does.
    behaviour: name -> ("ok", writes) | ("exit 1",) | ("validation",) | ("timeout",) | ("flaky", writes)."""
    calls = {}

    def run(vkc, spv, init, dump, heap_words, validation, timeout):
        name = Path(spv).stem
        calls[name] = calls.get(name, 0) + 1
        kind = behaviour.get(name, ("ok", {}))
        if kind[0] not in ("ok", "flaky"):
            return kind[0], "boom"
        heap = bytearray(Path(init).read_bytes())
        assert len(heap) == heap_words * 4
        for offset, word in kind[1].items():
            if kind[0] == "flaky" and calls[name] == 2:
                word ^= 1
            struct.pack_into("<I", heap, offset, word)
        Path(dump).write_bytes(bytes(heap))
        return "ok", "out[0..7] = ..."
    return run, calls


class HeapImage(unittest.TestCase):
    def test_table_then_zeros(self):
        image = smoke.heap_image(0x2000, 8192)
        self.assertEqual(len(image), 8192)
        self.assertEqual(struct.unpack_from("<Q", image, 0)[0], 0x2000)
        self.assertEqual(struct.unpack_from("<Q", image, 4088)[0], 0x2000)
        self.assertEqual(image[4096:], bytes(4096))

    def test_too_small(self):
        with self.assertRaises(ValueError):
            smoke.heap_image(0, 1024)

    def test_changed_words(self):
        a = bytes(16)
        b = struct.pack("<4I", 0, 7, 0, 9)
        self.assertEqual(smoke.changed_words(a, b), 2)


class Smoke(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.spirv = Path(self.dir.name)
        for name in ("Alpha", "Beta", "Gamma", "Delta", "Echo", "LeafRehydration"):
            (self.spirv / (name + ".spv")).write_bytes(b"\x03\x02\x23\x07")

    def tearDown(self):
        self.dir.cleanup()

    def run_smoke(self, behaviour, **kw):
        run, calls = fake_runner(behaviour)
        args = dict(table_qword=0x2000, heap_bytes=8192, validation=True, timeout=5, include_rehydration=False)
        args.update(kw)
        return smoke.smoke(self.spirv, "vkc", runner=run, **args), calls

    def test_statuses(self):
        results, calls = self.run_smoke({
            "Alpha": ("ok", {0x1000: 5, 0x1004: 6}),
            "Beta": ("exit 1",),
            "Gamma": ("validation",),
            "Delta": ("timeout",),
            "Echo": ("flaky", {0x1800: 1}),
        })
        got = {r["name"]: (r["status"], r["words_changed"]) for r in results}
        self.assertEqual(got, {"Alpha": ("pass", 2), "Beta": ("exit 1", None), "Gamma": ("validation", None),
                               "Delta": ("timeout", None), "Echo": ("nondeterministic", 1)})
        self.assertEqual(calls["Alpha"], 2, "a passing program runs twice")
        self.assertEqual(calls["Beta"], 1, "a failing first run is not repeated")
        self.assertEqual([r["name"] for r in results], sorted(got), "name order")

    def test_rehydration_is_left_to_test_376(self):
        results, calls = self.run_smoke({})
        self.assertNotIn("LeafRehydration", calls)
        results, calls = self.run_smoke({}, include_rehydration=True)
        self.assertIn("LeafRehydration", [r["name"] for r in results])


class Cli(unittest.TestCase):
    def test_missing_vk_compute_test(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as e:
                smoke.main(["--spirv", d, "--build", d])
            self.assertIn("vk_compute_test", str(e.exception))

    def test_bad_heap_size(self):
        with self.assertRaises(SystemExit):
            smoke.main(["--spirv", ".", "--heap-bytes", "100"])

    def test_missing_dll(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as e:
                smoke.translate(d, d, 32768)
            self.assertIn("XboxBvhBuilder.dll", str(e.exception))


if __name__ == "__main__":
    unittest.main()
