"""Run the test shaders that build.py made on this machine's GPU and compare with what each one must give.

    python -I runtime/test/shaders/check.py <out dir of build.py> [dir with the test programs]

The test programs are vk_compute_test, vk_store_test, vk_harness and vk_bindless_test (runtime/build by default). With
the Visual Studio generator they are in runtime/build/RelWithDebInfo, Release and so on: that subfolder is found by
itself (the newest complete one if there are several) and its name is printed.
Each line is one shader: ok, or what came back instead. The wave32 test ps_bitmask_perm is only judged on lanes
0-31, since a GPU with 64-wide subgroups puts half the pixels in lanes it wasn't written for.
The two whole-wave mask checks are skipped when the device cannot provide a full 64-wide compute subgroup.
The cs_w64x_* / ms_w64x_* cross-half tests (lanes 32-63 against lanes 0-31, see their .asm headers) are skipped there
too. With LEONIDA_W64_EMULATE=1 they run without that requirement, on a device whose subgroups are narrower (NVIDIA),
to check a translator that emulates wave64 on them. scripts/run_gpu_tests.py builds the mesh fixtures and
fs_passthru.spv; here the mesh checks run when those files are in <out dir>/spirv.
ps_tex is skipped when the device lacks a feature the bindless set or the shader needs.
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

# The programs every run needs. Multi-config CMake generators (Visual Studio, Xcode, Ninja Multi-Config) put each of
# them in build/<Config> instead of build/ (TEST-38).
TEST_PROGRAMS = ("vk_compute_test", "vk_store_test", "vk_harness")
CONFIG_DIRS = ("RelWithDebInfo", "Release", "Debug", "MinSizeRel")


def complete_build_dirs(build, programs):
    """`build` and its configuration subfolders, in that order, that hold all of `programs` (file names with extension)."""
    return [d for d in (build, *(os.path.join(build, c) for c in CONFIG_DIRS))
            if all(os.path.isfile(os.path.join(d, n)) for n in programs)]


def resolve_build_dir(build, programs):
    """The folder that holds all of `programs`.

    Makefile and Ninja builds leave them in `build`. With the Visual Studio generator they land in build/RelWithDebInfo,
    build/Release and so on. `build` itself wins when it has them all; otherwise the configuration folder that has them
    all and was built last. If none has them, `build` is returned, so a "not found" message names the folder given."""
    found = complete_build_dirs(build, programs)
    if not found or found[0] == build:
        return build
    return max(found, key=lambda d: max(os.path.getmtime(os.path.join(d, n)) for n in programs))


def pick_build_dir(build, programs):
    """resolve_build_dir, and say which folder was picked when it is a configuration subfolder, so a report can name
    the build it ran and a stale pick is visible."""
    chosen = resolve_build_dir(build, programs)
    if chosen != build:
        names = [os.path.basename(d) for d in complete_build_dirs(build, programs)]
        print(f"test programs: {chosen}" + (f" (newest of {', '.join(names)})" if len(names) > 1 else ""))
    return chosen


def not_found_message(build, missing):
    return (f"Test programs not found in {build}: {', '.join(missing)}\n"
            f"(its {', '.join(CONFIG_DIRS)} subfolders were searched too)\n"
            "Build them with cmake -S runtime -B runtime/build, then cmake --build runtime/build.\n"
            "CMake only builds them when it finds Vulkan. If it printed \"Vulkan SDK not found\", install the "
            "Vulkan headers (vulkan-headers on Arch) and configure again.")

COMPUTE = [("cs_lds", "4", "mirror64"), ("cs_exec64", "1", "exec64"), ("cs_bvh", "1", "bvh"),
           ("cs_workgroup_yz", "2x3x4", "workgroup_yz")]
CS_STORE = [0x400080ff, 0xe00003ff, 0x3c004000, 0xeeeeeeee, 0x40003c00, 0xbc003800]
CS_SMEM = [0x5eed0024, 0x5eed0028, 0x5eed002c, 0x5eed0030, 0x5eed0010, 0x5eed0014, 0x5eed0018, 0x5eed001c]
WAVE32_DUMPS = ("cs_store", "cs_smem_soffset_imm", "cs_wqm_w32")
CS_DUMPS = [("cs_store", CS_STORE), ("cs_smem_soffset_imm", CS_SMEM), ("cs_wave64_ballot", [32, 16, 24, 40, 8, 48]),
            ("cs_wm_half_ops", [32, 16, 16, 0, 16]), ("cs_wm_half_cmp", [1, 0, 1, 1]),
            ("cs_wm_bfm64", [16, 8, 8, 40]), ("cs_wm_scc_rfl", [55, 48]),
            # mask-only wave64: s_bcnt1_i32_b64 on ordinary SGPRs, no WAVE64_RE EXEC markers (#210 / SHADER-28)
            ("cs_w64_mask_detect", [48]),
            ("cs_wqm_w64", [20, 4, 12, 8, 1, 0, 15728895, 20]), ("cs_wqm_w32", [12, 1, 4, 0, 8, 15728880, 4, 32]),
            ("cs_wqm_w64_half", [11, 14, 0, 1, 6, 8, 12, 2])]
# num_records bounds check (cs_bounds.asm): one thread, so no 64-wide subgroup requirement.
BOUNDS = [("cs_bounds", [0x12345678, 0x12345678, 0x00000000, 0xEEEEEEEE])]
CMPX_CLASS_WORDS = [0xEEEEEEEE] * 3 + [0x22] + [0x11] * 4
# Cross-half wave64 tests (x = cross-half): exact results need 64 lanes, or an emulation of them on narrower subgroups.
W64_EMULATED = os.environ.get("LEONIDA_W64_EMULATE") == "1"
W64X_DUMPS = [("cs_w64x_readlane", [121, 121, 16, 97, 16, 100, 100, 121]),
              ("cs_w64x_writelane", [4660, 77, 99, 2748, 0, 0, 4660, 99]),
              ("cs_w64x_readfirstlane", [140, 140, 163, 132, 100, 105, 133, 48]),
              ("cs_w64x_bpermute", [36, 316, 253, 218, 435, 225, 0, 442]),
              ("cs_w64x_branch", [0, 0, 0, 0, 5, 5, 9, 9]),
              ("cs_w64x_mask_reuse", [8, 40, 8, 40, 2, 20, 10, 64]),
              ("cs_w64x_mask_reuse2", [10, 20, 8, 8, 5, 2, 132, 131]),
              ("cs_w64x_chain_barrier", [564, 563, 500, 627]),                   # 128-thread group
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
              "ps_ops7": (0x00ff8077, 0xff0000ff, 0x89abcdef, 0x00070003),
              "ps_ops8": (0x80000000, 0xffffffff, 0x000f100e, 0x0f0f0f1f),
              "ps_scc": (1, 1, 3, 0x80000000), "ps_sdwa": (0x000000f1, 0xfffefff2, 0x3c003344, 0xfff70000),
              "ps_vcc_x2": (0x11223344, 0x55667788, 0, 0),
              "ps_scc_minmax": (0x00030205, 0x00030207, 0x00030209, 0x00030203),
              "ps_fma_literal": (0x41b80000, 0x41800000, 0x40e00000, 0x41a40000)}
# Pixel shaders that read memory: vk_harness puts this V# in s[0:3] and seeds the heap it points at, as
# tests/semantic/test_value_semantics.py does for the interpreter.
PIXEL_ENV = {"ps_vcc_x2": {"VKH_SGPRS": "0x1000,0,16,0", "VKH_HEAP": "0x1000=0x11223344,0x1004=0x55667788"}}
CLIP_TESTS = ("vs_clip", "vs_clip_param4", "vs_clip_undeclared")
PIXEL_TESTS = sorted((*PIXEL_BITS, "ps_bary", "ps_flat", "ps_bitmask_perm", "ps_uniform_branch"))
PIXEL_TESTS = sorted((*PIXEL_BITS, "ps_bary", "ps_flat", "ps_bitmask_perm", "ps_uniform_branch", "ps_wqm"))


# Windows error codes for a program the system refuses to start, and who refused it.
BLOCKED_BY = {4551: "Windows Application Control (Smart App Control or an App Control policy)",
              1260: "a Windows software restriction policy",
              225: "Microsoft Defender, which reported it as a threat"}


def start_failure(exe, err):
    """One result line for a test program that could not be started, instead of a traceback that ends the run.
    Smart App Control blocks unsigned programs it has no reputation for, so a fresh build can be refused."""
    name = os.path.basename(str(exe))
    who = BLOCKED_BY.get(getattr(err, "winerror", None))
    if who is None:
        return f"could not start {name}: {err}"
    return (f"could not start {name}: blocked by {who}. A freshly built test program has no reputation yet: "
            "allow it, or turn Smart App Control off (Windows Security > App & browser control), and run again")


def run(*cmd, env=None):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    except OSError as e:
        if not isinstance(e, FileNotFoundError):
            return 1, start_failure(cmd[0], e)
        tools = os.path.abspath(os.path.dirname(cmd[0]))
        missing = [t + EXE for t in TEST_PROGRAMS if not os.path.isfile(os.path.join(tools, t + EXE))]
        sys.exit(not_found_message(tools, missing))
    return r.returncode, r.stdout + r.stderr


def failure(code, output, lines):
    details = "\n".join(output.strip().splitlines()[-lines:])
    return f"exit {code}: {details or 'no test output'}"


class Skip(str):
    """A result that is neither a pass nor a failure: the device can't run the test."""


def check_mesh_v5(data):
    if len(data) != 24 * 16:
        return f"expected 24 RGBA32F pixels (384 bytes), got {len(data)} bytes"
    pixels = iter(struct.iter_unpack("<4f", data))
    errors = []
    for z in range(4):
        for y in range(3):
            for x in range(2):
                pixel = next(pixels)
                # v5 = linear group index * NumThreads + lane (#209); the shader records lanes 0, 1 and 31 of 32
                g = x + 2*y + 6*z
                want = (32 * g, 32 * g + 1, 32 * g + 31, 1)
                if pixel != want:
                    errors.append(f"({x},{y},{z}): got {pixel}, expected {want}")
    return f"{len(errors)}/24 mismatches; first {errors[0]}" if errors else ""


def mesh_v5_test(tools, spirv, raw):
    shader = Path(spirv) / "ms_v5_workgroup.spv"
    if not shader.is_file():
        return Skip("SKIP (build the mesh fixture with scripts/run_gpu_tests.py)")
    fs = Path(spirv) / "fs_passthru.spv"
    if not fs.is_file():
        return "missing fs_passthru.spv"
    code, out = run(os.path.join(tools, "vk_harness" + EXE), shader, fs, raw, "--mesh-group-readback", "2", "3", "4")
    if code == UNSUPPORTED:
        return Skip("SKIP (mesh shading unavailable)")
    if code or not raw.is_file():
        return failure(code, out, 3)
    return check_mesh_v5(raw.read_bytes())


def partial_wave_test(tools, spirv, name, expected_words):
    """A group smaller than a wave: run with subgroup size 64 requested (no full subgroups); None if the device
    has no 64-wide subgroup. With LEONIDA_W64_EMULATE=1 it asks for 32 instead, the width the emulation is for, so
    a device that defaults to 64 (AMD) runs the emulated path at 32 too (#166)."""
    shader = os.path.join(spirv, name + ".spv")
    if not os.path.isfile(shader):
        return f"missing {name}.spv"
    old = os.environ.get("VKC_SUBGROUP_SIZE")
    os.environ["VKC_SUBGROUP_SIZE"] = "32" if W64_EMULATED else "64"
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
    if mode == "workgroup_yz":
        if "2x3x4 workgroup Y/Z: 0 mismatches" in out.splitlines():
            return ""
    elif expected_words is not None:
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
        harness = os.path.join(tools, "vk_harness" + EXE)
        try:
            r = subprocess.run([harness, shader, fs, raw], capture_output=True, text=True, env=env)
        except OSError as e:
            return start_failure(harness, e)
        if r.returncode == UNSUPPORTED:   # vk_harness: the device has no VK_EXT_mesh_shader
            m = re.search(r"unsupported: .*", r.stderr)
            return Skip(f"SKIP ({m.group(0) if m else 'unsupported device'})")
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


def bindless_test(tools, spirv):
    """ps_tex: a texture found through the descriptor hash table and the bindless set (vk_bindless_test)."""
    shaders = [os.path.join(spirv, name + ".spv") for name in ("vs_ids", "ps_tex")]
    for path in shaders:
        if not os.path.isfile(path):
            return f"missing {os.path.basename(path)}"
    code, out = run(os.path.join(tools, "vk_bindless_test" + EXE), *shaders)
    if code == UNSUPPORTED:
        m = re.search(r"unsupported: .*", out)
        return Skip(f"SKIP ({m.group(0) if m else 'unsupported device'})")
    m = re.search(r"(\d+) draws read back, (\d+) bad pixels", out)
    return "" if code == 0 and m and m.group(2) == "0" else failure(code, out, 3)


def pixels(tools, spirv, name, fmt):
    # Draw at the shader's own wave size, as run_gpu_tests.py does (#146): a wave32 pixel shader run as wave64
    # (RADV's default on RDNA1) reads lanes it was never written for.
    sg = subgroup_size(spirv, name)
    env = dict(os.environ, VKH_SUBGROUP_SIZE=str(sg)) if sg else None
    if name in PIXEL_ENV:
        env = dict(env or os.environ, **PIXEL_ENV[name])
    with tempfile.TemporaryDirectory(prefix="leonida-pixels-") as tmp:
        raw = Path(tmp) / "pixels.raw"
        code, out = run(os.path.join(tools, "vk_harness" + EXE), os.path.join(spirv, "vs_ids.spv"),
                        os.path.join(spirv, name + ".spv"), raw, env=env)
        if code == UNSUPPORTED:   # vk_harness: the shader needs a device feature this GPU lacks
            return Skip(""), out
        if code or not raw.is_file():
            return None, out
        data = raw.read_bytes()
    if len(data) != 64 * 64 * 16:  # default harness viewport, four 32-bit components per pixel
        return None, out
    return list(struct.iter_unpack(fmt, data)), out


def pixel_test(tools, spirv, name):
    path = os.path.join(spirv, name + ".spv")
    if not os.path.isfile(path):
        return f"missing {name}.spv"
    px, out = pixels(tools, spirv, name, "<4f" if name == "ps_bary" else "<2I2f" if name in ("ps_flat", "ps_uniform_branch") else "<4I")
    if isinstance(px, Skip):   # exit 77: the device cannot run this shader (e.g. no barycentrics)
        m = re.search(r"unsupported: .*", out)
        return Skip(f"SKIP ({m.group(0) if m else 'unsupported device'})")
    if name == "ps_bitmask_perm":
        # Decide from the reported width before the pixel count: a narrow device's frame can be short, and with no
        # width line the lanes 0-31 this shader writes are not known to exist.
        m = re.search(r"subgroup \d+\.\.(\d+)", out)
        if not m:
            return Skip("SKIP (vk_harness reported no subgroup width)")
        if int(m.group(1)) < 32:
            return Skip(f"SKIP (subgroup {m.group(1)} is narrower than 32 lanes)")
    if not px:
        return out if out.startswith("could not start") else "vk_harness failed or returned no pixels"
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
    if name == "ps_wqm":                               # (wqm, own, wqm, own) per pixel, ps_wqm.asm
        w = int(n ** 0.5)
        wrong = sum(p != (int(i % w % 4 < 2 and i // w % 4 >= 2), int(i % w % 4 == 1 and i // w % 4 == 2)) * 2 for i, p in
                    enumerate(px))
        return "" if not wrong else f"{wrong} of {n} pixels differ from the whole-quad expectation"
    if name == "ps_uniform_branch":                    # (s5, own VCC bit): 8 wherever the bit is set, never 7 there
        c = collections.Counter((a, b) for a, b, _, _ in px)
        fine = c[(8, 1)] and c[(8, 0)] and c[(7, 0)] and set(c) <= {(8, 1), (8, 0), (7, 0)}
        return "" if fine else f"(s5, bit) counts {dict(c)}"
    return "no check written for this shader"


def clip_test(tools, spirv, name):
    with tempfile.TemporaryDirectory(prefix="leonida-clip-") as tmp:
        raw = Path(tmp) / "pixels.raw"
        for clip_x, clip_y in ((False, False), (True, False), (False, True), (True, True)):
            code, output = run(os.path.join(tools, "vk_harness" + EXE), os.path.join(spirv, name + ".spv"),
                               os.path.join(spirv, "ps_ops.spv"), raw, "64", "64",
                               "0" if clip_x else "4", "0" if clip_y else "4")
            if code == UNSUPPORTED:
                return Skip("SKIP (shaderClipDistance unavailable)")
            if code:
                return failure(code, output, 2)
            if not raw.is_file():
                return "missing framebuffer"
            data = raw.read_bytes()
            raw.unlink()
            if len(data) != 64 * 64 * 16:
                return "incomplete framebuffer"
            clear = (0xbf800000,) * 4
            wrong = 0
            for i, pixel in enumerate(struct.iter_unpack("<4I", data)):
                inside = (not clip_x or i % 64 < 32) and (not clip_y or i // 64 < 16)
                drawn = inside or name == "vs_clip_undeclared"
                wrong += pixel != (PIXEL_BITS["ps_ops"] if drawn else clear)
            if wrong:
                return f"x={clip_x} y={clip_y}: {wrong} of 4096 pixels differ"
    return ""


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    spirv = os.path.join(sys.argv[1], "spirv")
    tools = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "..", "build")
    tools = pick_build_dir(os.path.abspath(tools), tuple(t + EXE for t in TEST_PROGRAMS))
    results = []
    for name, groups, expect in COMPUTE:
        results.append((name, compute_test(tools, spirv, name, groups, expect)))
    results.append(("cs_wavemask", compute_test(tools, spirv, "cs_wavemask", "1", "wavemask",
                                                require_wave64=not W64_EMULATED)))
    for name, want in CS_DUMPS:   # all but cs_store need a 64-wide subgroup unless wave64 is emulated
        results.append((name, compute_test(tools, spirv, name, "1", "dump",
                                           require_wave64=name not in WAVE32_DUMPS and not W64_EMULATED,
                                           expected_words=want)))
    for name, want in BOUNDS:
        results.append((name, compute_test(tools, spirv, name, "1", "dump", expected_words=want)))
    results.append(("cs_cmpx_class_f32", compute_test(tools, spirv, "cs_cmpx_class_f32", "1", "dump",
                                                      expected_words=CMPX_CLASS_WORDS)))
    for name, want in W64X_DUMPS:   # without --require-wave64 when LEONIDA_W64_EMULATE=1
        results.append((name, compute_test(tools, spirv, name, "1", "dump", require_wave64=not W64_EMULATED,
                                           expected_words=want)))
    for name, want in W64X_PARTIAL:
        results.append((name, partial_wave_test(tools, spirv, name, want)))
    results.append(("ms_w64x_ballot", mesh_w64x_test(tools, spirv, "ms_w64x_ballot")))
    with tempfile.TemporaryDirectory(prefix="leonida-mesh-") as tmp:
        results.append(("ms_v5_workgroup", mesh_v5_test(tools, spirv, Path(tmp) / "groups.raw")))
    for name, args in (("cs_imgstore", []), ("cs_imgload", ["load"])):
        code, out = run(os.path.join(tools, "vk_store_test" + EXE), os.path.join(spirv, name + ".spv"), *args)
        if code == UNSUPPORTED:   # vk_store_test: the device lacks a feature this test needs
            m = re.search(r"unsupported: .*", out)
            results.append((name, Skip(f"SKIP ({m.group(0) if m else 'unsupported device'})")))
            continue
        bad = re.findall(r"(\d+) bad", out)
        results.append((name, "" if code == 0 and len(bad) == 3 and set(bad) == {"0"} else failure(code, out, 3)))
    for name in CLIP_TESTS:
        results.append((name, clip_test(tools, spirv, name)))
    for name in PIXEL_TESTS:
        results.append((name, pixel_test(tools, spirv, name)))
    results.append(("ps_tex", bindless_test(tools, spirv)))
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
