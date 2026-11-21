"""Run the test shaders that build.py made on this machine's GPU and compare with what each one must give.

    python -I runtime/test/shaders/check.py <out dir of build.py> [dir with the test programs]

The test programs are vk_compute_test, vk_store_test and vk_harness (runtime/build by default). Each line is one
shader: ok, or what came back instead. The wave32 test ps_bitmask_perm is only judged on lanes 0-31, since a GPU
with 64-wide subgroups puts half the pixels in lanes it wasn't written for.
The two whole-wave mask checks are skipped when the device cannot provide a full 64-wide compute subgroup.
The cs_w64x_* / ms_w64x_* cross-half tests (lanes 32-63 against lanes 0-31, see their .asm headers) are skipped there
too. With LEONIDA_W64_EMULATE=1 they run without that requirement, on a device whose subgroups are narrower (NVIDIA),
to check a translator that emulates wave64 on them. ms_w64x_ballot is a mesh shader that build.py does not build:
scripts/run_gpu_tests.py makes it (and fs_passthru.spv); here it runs when both .spv files are in <out dir>/spirv.
"""
import collections
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = ".exe" if os.name == "nt" else ""
UNSUPPORTED = 77                                       # a test tool exits with this when the device lacks a feature

COMPUTE = [("cs_lds", "4", "mirror64"), ("cs_exec64", "1", "exec64"), ("cs_bvh", "1", "bvh")]
CS_STORE = [0x400080ff, 0xe00003ff, 0x3c004000, 0xeeeeeeee, 0x40003c00, 0xbc003800]
CS_SMEM = [0x5eed0024, 0x5eed0028, 0x5eed002c, 0x5eed0030, 0x5eed0010, 0x5eed0014, 0x5eed0018, 0x5eed001c]
WAVE32_DUMPS = ("cs_store", "cs_smem_soffset_imm")
CS_DUMPS = [("cs_store", CS_STORE), ("cs_smem_soffset_imm", CS_SMEM), ("cs_wave64_ballot", [32, 16, 24, 40, 8, 48]),
            ("cs_wm_half_ops", [32, 16, 16, 0, 16]), ("cs_wm_half_cmp", [1, 0, 1, 1]),
            ("cs_wm_bfm64", [16, 8, 8, 40]), ("cs_wm_scc_rfl", [55, 48])]
# Cross-half wave64 tests (x = cross-half): exact results need 64 lanes, or an emulation of them on narrower subgroups.
W64_EMULATED = os.environ.get("LEONIDA_W64_EMULATE") == "1"
W64X_DUMPS = [("cs_w64x_readlane", [121, 121, 16, 97, 16, 100, 100, 121]),
              ("cs_w64x_writelane", [4660, 77, 99, 2748, 0, 0, 4660, 99]),
              ("cs_w64x_readfirstlane", [140, 140, 163, 132, 100, 105, 133, 48]),
              ("cs_w64x_bpermute", [36, 316, 253, 218, 435, 225, 0, 442]),
              ("cs_w64x_branch", [0, 0, 0, 0, 5, 5, 9, 9]),
              ("cs_w64x_ballot", [2053, 2061, 2069, 2077, 2085, 2093, 2101, 2109]),
              ("cs_w64x_wg128_branches", [121, 121, 343, 343, 14, 31, 19812, 22728]),   # 128-thread group
              ("cs_w64x_lds_raw", [132, 137, 100, 108, 131, 101, 149, 118]),
              ("cs_w64x_lds_war", [1032, 1063, 1000, 1031, 2032, 2063, 2000, 2031]),
              ("cs_w64x_sbarrier", [605, 605, 686, 686, 3032, 3013, 3032, 3031]),         # 128-thread group
              ("cs_w64x_load_ballot_load", [43, 74, 75, 42, 42, 18, 41, 32])]
# A 40-thread group (one partial wave) cannot use --require-wave64 (full subgroups), so it asks for subgroup size 64.
W64X_PARTIAL = [("cs_w64x_partial", [40, 40, 528, 0, 0, 532, 539, 524820])]
PIXEL_BITS = {"ps_legacy_mul": (0, 0, 0, 0x40c00000),
              "ps_ops": (0x48804300, 5, 7, 0xfffffffc), "ps_ops2": (0xabcd4400, 0x46001234, 0xfffffffa, 7),
              "ps_ops3": (0xfffffffa, 0x44660066, 0x44aabbcc, 16), "ps_ops4": (0x34000010, 0x00120800, 0xbeef0000, 0x12343c00),
              "ps_ops5": (0x12340010, 0xbeef0800, 0xaaaaffff, 0xdeadbc00),
              "ps_ops6": (0xf800ffff, 0x01000010, 0x00800800, 0x8000fff0),
              "ps_scc": (1, 1, 3, 0x80000000), "ps_sdwa": (0x000000f1, 0xfffefff2, 0x3c003344, 0xfff70000)}
PIXEL_TESTS = sorted((*PIXEL_BITS, "ps_bary", "ps_flat", "ps_bitmask_perm", "ps_uniform_branch"))


def run(*cmd, env=None):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    except FileNotFoundError:
        tools = os.path.abspath(os.path.dirname(cmd[0]))
        missing = [t + EXE for t in ("vk_compute_test", "vk_store_test", "vk_harness")
                   if not os.path.isfile(os.path.join(tools, t + EXE))]
        sys.exit(f"Test programs not found in {tools}: {', '.join(missing)}\n"
                 "Build them with cmake -S runtime -B runtime/build, then cmake --build runtime/build.\n"
                 "CMake only builds them when it finds Vulkan. If it printed \"Vulkan SDK not found\", install the "
                 "Vulkan headers (vulkan-headers on Arch) and configure again.")
    return r.returncode, r.stdout + r.stderr


def failure(code, output, lines):
    details = "\n".join(output.strip().splitlines()[-lines:])
    return f"exit {code}: {details or 'no test output'}"


class Skip(str):
    """A result that is neither a pass nor a failure: the device can't run the test."""


def partial_wave_test(tools, spirv, name, expected_words):
    """A group smaller than a wave: run with subgroup size 64 requested (no full subgroups); None if the device
    has no 64-wide subgroup (unless LEONIDA_W64_EMULATE=1)."""
    shader = os.path.join(spirv, name + ".spv")
    if not os.path.isfile(shader):
        return f"missing {name}.spv"
    old = os.environ.get("VKC_SUBGROUP_SIZE")
    os.environ["VKC_SUBGROUP_SIZE"] = "64"
    try:
        code, out = run(os.path.join(tools, "vk_compute_test" + EXE), shader, "1", "dump")
    finally:
        if old is None:
            os.environ.pop("VKC_SUBGROUP_SIZE", None)
        else:
            os.environ["VKC_SUBGROUP_SIZE"] = old
    if code != 0:
        return failure(code, out, 2)
    if not W64_EMULATED and not any(int(x) >= 64 for x in re.findall(r"subgroup \d+\.\.(\d+)", out)):
        return None
    m = re.search(r"^out\[0\.\.7\] = ((?:[0-9]+ ){7}[0-9]+)$", out, re.MULTILINE)
    if m and [int(x) for x in m.group(1).split()] == expected_words:
        return ""
    return failure(code, out, 2)


def compute_test(tools, spirv, name, groups, mode, *, require_wave64=False, expected_words=None):
    """Return an empty string for a pass, None for unsupported wave64, or a failure diagnostic."""
    shader = os.path.join(spirv, name + ".spv")
    args = []
    if require_wave64:
        if not os.path.isfile(shader):
            return f"missing {name}.spv"
        args.append("--require-wave64")
    code, out = run(os.path.join(tools, "vk_compute_test" + EXE), shader, groups, mode, *args)
    if require_wave64 and code == 77:  # vk_compute_test: the device cannot run a 64-wide compute subgroup
        return None
    if code != 0:
        return failure(code, out, 2)
    if require_wave64 and "compute subgroup size: 64" not in out.splitlines():
        return "64-wide subgroup was not confirmed (rebuild vk_compute_test)"
    if expected_words is not None:
        m = re.search(r"^out\[0\.\.7\] = ((?:[0-9]+ ){7}[0-9]+)$", out, re.MULTILINE)
        if m and [int(x) for x in m.group(1).split()][:len(expected_words)] == expected_words:
            return ""
    else:
        m = re.search(r"(\d+) mismatches", out)
        if m and m.group(1) == "0":
            return ""
    return failure(code, out, 2)


def mesh_w64x_test(tools, spirv, name):
    """ms_w64x_ballot: ms_tri's picture, blue 0.5 only if the ballot/mbcnt compaction was exact across both halves."""
    shader, fs = os.path.join(spirv, name + ".spv"), os.path.join(spirv, "fs_passthru.spv")
    if not (os.path.isfile(shader) and os.path.isfile(fs)):
        return Skip("SKIP (mesh shader: build it with scripts/run_gpu_tests.py, build.py does not)")
    env = dict(os.environ, VKH_MESH="1")
    if not W64_EMULATED:
        env["VKH_SUBGROUP_SIZE"] = "64"
    with tempfile.TemporaryDirectory(prefix="leonida-mesh-") as tmp:
        raw = Path(tmp) / "pixels.raw"
        r = subprocess.run([os.path.join(tools, "vk_harness" + EXE), shader, fs, raw], capture_output=True, text=True, env=env)
        if r.returncode or not raw.is_file():
            return failure(r.returncode, r.stdout + r.stderr, 2)
        data = raw.read_bytes()
    if len(data) != 64 * 64 * 16:
        return "vk_harness returned an unexpected pixel count"
    px = list(struct.iter_unpack("<4f", data))
    if not W64_EMULATED and 64 not in [int(x) for x in re.findall(r"subgroup \d+\.\.(\d+)", r.stderr)]:
        return None   # the device has no 64-wide subgroups
    wrong = sum(abs(p[0] - ((i % 64) + .5) / 64) > 1e-3 or abs(p[1] - ((i // 64) + .5) / 64) > 1e-3
                or abs(p[2] - .5) > 1e-3 or abs(p[3] - 1) > 1e-3 for i, p in enumerate(px))
    return "" if not wrong else f"{wrong} of {len(px)} pixels differ (blue {px[0][2]:.3g} at pixel 0, want 0.5)"


def subgroup_size(spirv, name):
    """The wave size the translator recorded for a shader (index.json), or None."""
    try:
        with open(os.path.join(spirv, "index.json"), encoding="utf-8") as f:
            return json.load(f).get(name, {}).get("subgroup_size")
    except (OSError, ValueError):
        return None


def pixels(tools, spirv, name, fmt):
    # Draw at the shader's own wave size, as run_gpu_tests.py does (#146): a wave32 pixel shader run as wave64
    # (RADV's default on RDNA1) reads lanes it was never written for.
    sg = subgroup_size(spirv, name)
    env = dict(os.environ, VKH_SUBGROUP_SIZE=str(sg)) if sg else None
    with tempfile.TemporaryDirectory(prefix="leonida-pixels-") as tmp:
        raw = Path(tmp) / "pixels.raw"
        code, _ = run(os.path.join(tools, "vk_harness" + EXE), os.path.join(spirv, "vs_ids.spv"),
                      os.path.join(spirv, name + ".spv"), raw, env=env)
        if code or not raw.is_file():
            return None
        data = raw.read_bytes()
    if len(data) != 64 * 64 * 16:  # default harness viewport, four 32-bit components per pixel
        return None
    return list(struct.iter_unpack(fmt, data))


def pixel_test(tools, spirv, name):
    path = os.path.join(spirv, name + ".spv")
    if not os.path.isfile(path):
        return f"missing {name}.spv"
    px = pixels(tools, spirv, name, "<4f" if name == "ps_bary" else "<2I2f" if name in ("ps_flat", "ps_uniform_branch") else "<4I")
    if not px:
        return "vk_harness failed or returned no pixels"
    n = len(px)
    if name in PIXEL_BITS:
        wrong = sum(p != PIXEL_BITS[name] for p in px)
        return "" if not wrong else f"{wrong} of {n} pixels differ, e.g. {[hex(x) for x in next(p for p in px if p != PIXEL_BITS[name])]}"
    if name == "ps_bary":                              # raw i, j against the interpolated attr0.xy / 2
        if any(not math.isfinite(v) for p in px for v in p):
            return "barycentrics contain non-finite values"
        err = max(max(abs(a / 2 - i), abs(b / 2 - j)) for a, b, i, j in px)
        return "" if err < 1e-6 else f"barycentrics off by {err:.3g}"
    if name == "ps_flat":
        if any(not math.isfinite(v) for p in px for v in p[2:]):
            return "UVs contain non-finite values"
        w = int(n ** 0.5)
        wrong = sum((a, b) != (0x101, 0x102) for a, b, _, _ in px)
        err = max(max(abs(u - (i % w + 0.5) / w), abs(v - (i // w + 0.5) / w)) for i, (_, _, u, v) in enumerate(px))
        return "" if not wrong and err < 1e-6 else f"{wrong} pixels without the first vertex's value, uv off by {err:.3g}"
    if name == "ps_bitmask_perm":
        low = [p for p in px if p[0] < 32]
        wrong = sum(not (a == lane ^ 9 and b == ((lane & 5) | 16) ^ 1) for lane, a, b, _ in low)
        return "" if low and not wrong else f"{wrong} of {len(low)} pixels in lanes 0-31 wrong"
    if name == "ps_uniform_branch":                    # (s5, own VCC bit): 8 wherever the bit is set, never 7 there
        c = collections.Counter((a, b) for a, b, _, _ in px)
        fine = c[(8, 1)] and c[(8, 0)] and c[(7, 0)] and set(c) <= {(8, 1), (8, 0), (7, 0)}
        return "" if fine else f"(s5, bit) counts {dict(c)}"
    return "no check written for this shader"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    spirv = os.path.join(sys.argv[1], "spirv")
    tools = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "..", "build")
    results = []
    for name, groups, expect in COMPUTE:
        results.append((name, compute_test(tools, spirv, name, groups, expect)))
    results.append(("cs_wavemask", compute_test(tools, spirv, "cs_wavemask", "1", "wavemask",
                                                require_wave64=not W64_EMULATED)))
    for name, want in CS_DUMPS:   # all but cs_store need a 64-wide subgroup unless wave64 is emulated
        results.append((name, compute_test(tools, spirv, name, "1", "dump",
                                           require_wave64=name not in WAVE32_DUMPS and not W64_EMULATED,
                                           expected_words=want)))
    for name, want in W64X_DUMPS:   # without --require-wave64 when LEONIDA_W64_EMULATE=1
        results.append((name, compute_test(tools, spirv, name, "1", "dump", require_wave64=not W64_EMULATED,
                                           expected_words=want)))
    for name, want in W64X_PARTIAL:
        results.append((name, partial_wave_test(tools, spirv, name, want)))
    results.append(("ms_w64x_ballot", mesh_w64x_test(tools, spirv, "ms_w64x_ballot")))
    for name, args in (("cs_imgstore", []), ("cs_imgload", ["load"])):
        code, out = run(os.path.join(tools, "vk_store_test" + EXE), os.path.join(spirv, name + ".spv"), *args)
        if code == UNSUPPORTED:   # vk_store_test: the device lacks a feature this test needs
            m = re.search(r"unsupported: .*", out)
            results.append((name, Skip(f"SKIP ({m.group(0) if m else 'unsupported device'})")))
            continue
        bad = re.findall(r"(\d+) bad", out)
        results.append((name, "" if code == 0 and len(bad) == 3 and set(bad) == {"0"} else failure(code, out, 3)))
    for name in PIXEL_TESTS:
        results.append((name, pixel_test(tools, spirv, name)))
    for name, problem in results:
        if problem is None:
            problem = "SKIP (full 64-wide compute subgroups unavailable)"
        print(f"{name:20} {'ok' if not problem else problem}")
    skipped = sum(p is None or isinstance(p, Skip) for _, p in results)
    failed = sum(bool(p) for _, p in results) - sum(isinstance(p, Skip) for _, p in results)
    print(f"{len(results) - failed - skipped} of {len(results)} pass, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
