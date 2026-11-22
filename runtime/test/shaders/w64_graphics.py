"""SHADER-10 GPU checks: wave64 pixel shaders on 32-lane subgroups (docs/design.md, "Wave64 graphics: partition
invariance").

    python3 -I runtime/test/shaders/w64_graphics.py <build dir with spirv/> [tools dir] [name ...]

A wave64 pixel shader is exact on 32-wide subgroups when its outputs do not change if the wave is cut into two
32-lane halves. The test draws each ps_w64_* fixture with the fragment stage's required subgroup size set to 64 (one
hardware wave, the reference) and to 32 (the halves) through VKH_SUBGROUP_SIZE of vk_harness, and compares:

  tier A, B   both images equal, and equal to the words the fixture's header derives (tier B's atomic compaction
              compares the set of stored slots, not their order)
  tier C      the words the header names differ between the two sizes, which shows the translator's detector is
              right to flag the shader (the other words stay equal)

A device that can require only one size (NVIDIA: 32 only) cannot run the comparison. There the default-size image is
checked against the derived words (tiers A, B), the tier C check is skipped, and two environment variables let a
second device stand in for the missing size: W64G_SAVE_REF=<dir> stores each 64-lane run of a device that can require
it, and W64G_REF=<dir> compares every run of the other device with those files (equal for A and B, different for C).
"""
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

EXE = ".exe" if os.name == "nt" else ""
UNSUPPORTED = 77
W = H = 64


class Skip(str):
    """Neither a pass nor a failure."""


def _x(i):
    return i % W


def _y(i):
    return i // W


def _a_skip(i):
    x, y = _x(i), _y(i)
    return (7 * y + x if x % 4 == 1 else 1000 + x, int(x % 4 < 2), 2000 + y + (9 if y % 4 == 2 else 0), x % 4 + 1)


def _b_waterfall(i):
    x, y = _x(i), _y(i)
    a = x % 4 + 4 * (y % 4)
    return (17 * a + x, a, 0, 0)


def _b_readfirstlane(i):
    return (_x(i) + 123, 277, 0, 0)


def _b_readlane_uniform(i):
    return (_x(i) + 123, 200, 0, 0)


def _b_ballot(i):
    return (int(_x(i) % 4 == 1), 0, 0, 0)


def _c_waterfall_count(i):
    x, y = _x(i), _y(i)
    return (17 * (x % 8 + 8 * (y % 8)) + x, None, 0, 0)       # None: the trip count, partition dependent


def _c_bcnt(i):
    return (None, int(_x(i) % 4 == 1), 0, 0)


def _c_readlane(i):
    return (None, _y(i) * W + _x(i), None, 0)                  # r2 = lane 7 of the wave: also partition dependent


def _c_lane_id(i):
    return (None, None, _x(i), 0)                              # r1 = lane id & 31 depends on the lane a pixel got


def _c_vote_stale(i):
    return (None, int(_x(i) == 0 and _y(i) == 0), 0, 0)


def _on(i):
    return int(_x(i) == 5 and _y(i) == 7)


def _a_one_lane(i):
    x, y, on = _x(i), _y(i), _on(i)
    return (7777 if on else 1000 + x, 3000 + 11 * on, 4000 + y + 13 * on, x % 4)


def _c_vote_else(i):
    return (7 if _on(i) else None, _on(i), 0, 0)               # r0 of the other pixels: 0 or 5 by the partition


def _c_scc_vote(i):
    return (1 if _on(i) else None, _on(i), 0, 0)               # r0 = SCC of the vote: the wave's or the half's


def _c_execz_sgpr(i):
    return (7 if _on(i) else None, _on(i), 0, 0)               # r0 = s5: 7 only where the active side ran


# name -> (tier, expected words per pixel (None = not fixed), environment of the draw)
FIXTURES = {
    "ps_w64_a_skip": ("A", _a_skip, {}),
    "ps_w64_a_one_lane": ("A", _a_one_lane, {}),
    "ps_w64_b_waterfall": ("B", _b_waterfall, {}),
    "ps_w64_b_readfirstlane": ("B", _b_readfirstlane, {"VKH_SGPRS": "0,0,0,0,123"}),
    "ps_w64_b_readlane_uniform": ("B", _b_readlane_uniform, {"VKH_SGPRS": "0,0,0,0,123"}),
    "ps_w64_b_ballot_atomic": ("B", _b_ballot, {"VKH_SGPRS": "0,0,65536,0x30000000", "VKH_HEAP": "0=0"}),
    "ps_w64_c_waterfall_count": ("C", _c_waterfall_count, {}),
    "ps_w64_c_bcnt_export": ("C", _c_bcnt, {}),
    "ps_w64_c_readlane": ("C", _c_readlane, {}),
    "ps_w64_c_lane_id": ("C", _c_lane_id, {}),
    "ps_w64_c_vote_stale": ("C", _c_vote_stale, {}),
    "ps_w64_c_vote_else": ("C", _c_vote_else, {}),
    "ps_w64_c_scc_vote": ("C", _c_scc_vote, {}),
    "ps_w64_c_execz_sgpr": ("C", _c_execz_sgpr, {}),
}


def draw(tools, spirv, name, size, tmp):
    """(words per pixel, heap words or None, applied) of one draw. `applied`: the required size took effect (size
    None: no size asked, applied is None). The words are None when the draw failed; then `heap` is the message."""
    tier, expect, extra = FIXTURES[name]
    env = dict(os.environ, **extra)
    env.pop("VKH_SUBGROUP_SIZE", None)
    if size:
        env["VKH_SUBGROUP_SIZE"] = str(size)
    raw, heap = Path(tmp) / f"{name}.{size}.raw", Path(tmp) / f"{name}.{size}.heap"
    if "VKH_HEAP" in extra:
        env["VKH_HEAP_DUMP"] = str(heap)
    cmd = [os.path.join(tools, "vk_harness" + EXE), os.path.join(spirv, "vs_ids.spv"), os.path.join(spirv, name + ".spv"),
           str(raw)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    out = r.stdout + r.stderr
    if r.returncode == UNSUPPORTED:
        return None, "unsupported: " + out.strip().splitlines()[-1], None
    if r.returncode or not raw.is_file():
        return None, f"exit {r.returncode}: {' | '.join(out.strip().splitlines()[-3:])}", None
    data = raw.read_bytes()
    if len(data) != W * H * 16:
        return None, f"{len(data)} bytes of pixels, expected {W * H * 16}", None
    applied = None
    if size:
        applied = f"fragment subgroup size: {size}\n" in out.replace("\r", "")
    words = list(struct.iter_unpack("<4I", data))
    words_heap = None
    if "VKH_HEAP" in extra:
        words_heap = list(struct.unpack("<%dI" % (65536 // 4), heap.read_bytes()))
    return words, words_heap, applied


def expected_problem(name, words, heap):
    """'' when the words (and heap) are what the fixture's header derives, else the first discrepancy."""
    tier, expect, _ = FIXTURES[name]
    wrong = []
    for i, w in enumerate(words):
        want = expect(i)
        if any(g is not None and g != v for g, v in zip(want, w)):
            wrong.append((i, w, want))
    if wrong:
        i, w, want = wrong[0]
        return f"{len(wrong)} of {len(words)} pixels differ from the derived words, first ({_x(i)},{_y(i)}): got {w}, want {want}"
    if name == "ps_w64_b_ballot_atomic":
        want = sorted(_y(i) * W + _x(i) + 1 for i in range(W * H) if _x(i) % 4 == 1)
        n = len(want)
        if heap[0] != n:
            return f"counter is {heap[0]}, want {n}"
        slots = heap[64:64 + n]
        if sorted(slots) != want:
            return f"slot words are not the {n} wanted pixel numbers once each ({len(set(slots))} distinct)"
        if any(heap[64 + n:]) or any(heap[1:64]):
            return "words outside the counter and the slots were written"
    return ""


def comparable(name, words, heap):
    """What two draws of the same fixture must share when the shader is partition invariant."""
    stable = [tuple(0 if g is None else v for g, v in zip(FIXTURES[name][1](i), w)) for i, w in enumerate(words)]
    if name == "ps_w64_b_ballot_atomic":
        return stable, sorted(heap[64:64 + heap[0]]), heap[0]
    return stable, None, None


def partition_dependent(name, a, b):
    """Whether the words the fixture's header names as partition dependent differ between two draws."""
    return any(any(g is None and x != y for g, x, y in zip(FIXTURES[name][1](i), p, q))
               for i, (p, q) in enumerate(zip(a, b)))


def check(tools, spirv, name):
    """"" when `name` behaves as its tier says on this device, a Skip when the device cannot run the comparison."""
    tier = FIXTURES[name][0]
    if not os.path.isfile(os.path.join(spirv, name + ".spv")):
        return f"missing {name}.spv"
    save, ref = os.environ.get("W64G_SAVE_REF"), os.environ.get("W64G_REF")
    with tempfile.TemporaryDirectory(prefix="leonida-w64g-") as tmp:
        runs = {}
        for size in (64, 32, None):
            words, heap, applied = draw(tools, spirv, name, size, tmp)
            if words is None:
                if heap.startswith("unsupported"):
                    return Skip("SKIP (" + heap + ")")
                return f"{size or 'default'}: {heap}"
            if size is None or applied:
                runs[size] = (words, heap)
        compared = False
        if save and 64 in runs:
            Path(save).mkdir(parents=True, exist_ok=True)
            (Path(save) / (name + ".ref64.raw")).write_bytes(b"".join(struct.pack("<4I", *w) for w in runs[64][0]))
        if tier in ("A", "B"):
            for size, (words, heap) in runs.items():
                if size is None and len(runs) > 1:
                    continue                      # the default draw repeats one of the forced sizes
                p = expected_problem(name, words, heap)
                if p:
                    return f"{size or 'default size'}: {p}"
            if 64 in runs and 32 in runs and comparable(name, *runs[64]) != comparable(name, *runs[32]):
                return "the images at required subgroup size 64 and 32 differ"
        elif 64 in runs and 32 in runs:
            if comparable(name, *runs[64]) != comparable(name, *runs[32]):
                return "a word that must not depend on the partition differs between 64 and 32"
            if not partition_dependent(name, runs[64][0], runs[32][0]):
                return "tier C but the images at required subgroup size 64 and 32 are equal: the detector is wrong"
            compared = True
        if ref and (Path(ref) / (name + ".ref64.raw")).is_file():
            data = (Path(ref) / (name + ".ref64.raw")).read_bytes()
            ref_words = list(struct.iter_unpack("<4I", data))
            for size, (words, heap) in runs.items():
                if size == 64:
                    continue
                if tier == "C":
                    if not partition_dependent(name, ref_words, words):
                        return "tier C but the run equals the 64-lane reference: the detector is wrong"
                    compared = True
                elif name != "ps_w64_b_ballot_atomic" and comparable(name, ref_words, None) != comparable(name, words, None):
                    return "the run differs from the 64-lane reference"
        if tier == "C" and not compared:
            return Skip("SKIP (tier C needs a device that can require both 64 and 32, or W64G_REF)")
    return ""


def main(argv):
    if not argv:
        sys.exit(__doc__)
    spirv = os.path.join(argv[0], "spirv")
    tools = argv[1] if len(argv) > 1 and os.path.isdir(argv[1]) else os.path.join(os.path.dirname(__file__), "..", "..", "build")
    names = [a for a in argv[1:] if a in FIXTURES] or sorted(FIXTURES)
    failed = 0
    for name in names:
        problem = check(os.path.abspath(tools), spirv, name)
        print(f"{name:28} {FIXTURES[name][0]}  {'ok' if not problem else problem}")
        failed += bool(problem) and not isinstance(problem, Skip)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
