"""Assemble, translate, and run the synthetic RDNA2->SPIR-V GPU test suite on Vulkan.

Usage:
    python -I scripts/run_gpu_tests.py [--llvm-mc PATH] [--build DIR] [--work DIR] [--filter SUBSTR]
"""
import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHADER_DIR = os.path.join(ROOT, "runtime", "test", "shaders")
TRANSLATOR = os.path.join(ROOT, "tools", "rdna2spirv.py")


def make_dxbc(chunks: list[tuple[bytes, bytes]]) -> bytes:
    """Build a minimal DXBC container from (fourcc, payload) pairs."""
    n = len(chunks)
    hdr_size = 32 + 4 * n
    offsets = []
    body = bytearray()
    for tag, payload in chunks:
        offsets.append(hdr_size + len(body))
        body += tag + struct.pack("<I", len(payload)) + payload
    total = hdr_size + len(body)
    hdr = bytearray(b"DXBC") + bytes(16) + struct.pack("<III", 1, total, n)
    for off in offsets:
        hdr += struct.pack("<I", off)
    return bytes(hdr) + bytes(body)


def make_sig_chunk(elements: list[tuple[int, int, int, int, int]]) -> bytes:
    """Build ISG1/OSG1 payload: elements of (sysv, ctype, reg, mask, rw_mask)."""
    out = bytearray(struct.pack("<II", len(elements), 8))
    for sysv, ctype, reg, mask, rw in elements:
        e = bytearray(32)
        struct.pack_into("<6I", e, 0, 0, 0, 0, sysv, ctype, reg)
        e[24] = mask
        e[25] = rw
        out += e
    return bytes(out)


def make_ms_psv0(threads_x: int, maxv: int, maxp: int, topology: int = 2) -> bytes:
    """Build PSV0 payload for a mesh shader (stage 13)."""
    r = bytearray(48)
    struct.pack_into("<HH", r, 12, maxv, maxp)
    r[24] = 13  # mesh stage
    r[27] = topology  # 2 = triangles
    struct.pack_into("<III", r, 36, threads_x, 1, 1)
    return struct.pack("<I", len(r)) + bytes(r)


def make_cs_fssm(rsrc2: int = 0x8088, lx: int = 64, ly: int = 1, lz: int = 1) -> bytes:
    """Build minimal FSSM header (0x148 bytes) with COMPUTE_PGM_RSRC2 and group size."""
    buf = bytearray(0x148)
    struct.pack_into("<4I", buf, 0x138, rsrc2, lx, ly, lz)
    return bytes(buf)


def assemble_to_s(llvm_mc: str, asm_path: str, s_path: str, name: str) -> None:
    """Assemble raw RDNA2 .asm with llvm-mc -show-encoding into .s format for rdna2spirv.py."""
    raw = open(asm_path, encoding="utf-8").read()
    cmd = [llvm_mc, "-triple=amdgcn--amdpal", "-mcpu=gfx1013"]
    for ln in raw.splitlines():
        s = ln.strip()
        if not s:
            continue
        if not s.startswith(";"):
            break
        m = re.match(r";\s*mattr:\s*(\S+)", s)
        if m:
            cmd.append(f"-mattr={m.group(1)}")
            break
    cmd.append("-show-encoding")
    p = subprocess.run(
        cmd,
        input=raw,
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"llvm-mc failed on {name}:\n{p.stderr}")
    lines = [f"; {name}  code 0x0..0x0 (synthetic), gfx1013, 0 invalid words"]
    for ln in p.stdout.splitlines():
        if ln.startswith("\t.") or ln.strip().startswith("."):
            continue
        if ln.startswith("\t"):
            lines.append(ln)
    with open(s_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def spirv_as(asm_text: str, out_spv: str) -> None:
    p = subprocess.run(["spirv-as", "--target-env", "vulkan1.2", "-o", out_spv, "-"],
                       input=asm_text, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"spirv-as failed for {out_spv}:\n{p.stderr}")


VS_TRI_SPVASM = """\
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %gl_VertexIndex %gl_Position %out0 %out1
               OpDecorate %gl_VertexIndex BuiltIn VertexIndex
               OpDecorate %gl_Position BuiltIn Position
               OpDecorate %out0 Location 0
               OpDecorate %out1 Location 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
   %ptr_in_i = OpTypePointer Input %int
  %ptr_out_v = OpTypePointer Output %v4float
     %uint_3 = OpConstant %uint 3
     %arr_v4 = OpTypeArray %v4float %uint_3
 %ptr_fn_arr = OpTypePointer Function %arr_v4
 %ptr_fn_v4  = OpTypePointer Function %v4float
        %fm1 = OpConstant %float -1
         %f0 = OpConstant %float 0
         %f1 = OpConstant %float 1
         %f2 = OpConstant %float 2
         %f3 = OpConstant %float 3
      %pos_0 = OpConstantComposite %v4float %fm1 %fm1 %f0 %f1
      %pos_1 = OpConstantComposite %v4float %f3 %fm1 %f0 %f1
      %pos_2 = OpConstantComposite %v4float %fm1 %f3 %f0 %f1
    %pos_arr = OpConstantComposite %arr_v4 %pos_0 %pos_1 %pos_2
      %uv0_0 = OpConstantComposite %v4float %f0 %f0 %f0 %f1
      %uv0_1 = OpConstantComposite %v4float %f2 %f0 %f0 %f1
      %uv0_2 = OpConstantComposite %v4float %f0 %f2 %f0 %f1
    %uv0_arr = OpConstantComposite %arr_v4 %uv0_0 %uv0_1 %uv0_2
      %uv1_0 = OpConstantComposite %v4float %f0 %f0 %f0 %f0
      %uv1_1 = OpConstantComposite %v4float %f2 %f0 %f0 %f0
      %uv1_2 = OpConstantComposite %v4float %f0 %f2 %f0 %f0
    %uv1_arr = OpConstantComposite %arr_v4 %uv1_0 %uv1_1 %uv1_2
%gl_VertexIndex = OpVariable %ptr_in_i Input
%gl_Position = OpVariable %ptr_out_v Output
       %out0 = OpVariable %ptr_out_v Output
       %out1 = OpVariable %ptr_out_v Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
      %p_var = OpVariable %ptr_fn_arr Function
      %0_var = OpVariable %ptr_fn_arr Function
      %1_var = OpVariable %ptr_fn_arr Function
               OpStore %p_var %pos_arr
               OpStore %0_var %uv0_arr
               OpStore %1_var %uv1_arr
        %idx = OpLoad %int %gl_VertexIndex
       %p_pt = OpAccessChain %ptr_fn_v4 %p_var %idx
       %p_vl = OpLoad %v4float %p_pt
               OpStore %gl_Position %p_vl
       %0_pt = OpAccessChain %ptr_fn_v4 %0_var %idx
       %0_vl = OpLoad %v4float %0_pt
               OpStore %out0 %0_vl
       %1_pt = OpAccessChain %ptr_fn_v4 %1_var %idx
       %1_vl = OpLoad %v4float %1_pt
               OpStore %out1 %1_vl
               OpReturn
               OpFunctionEnd
"""


def make_fs_passthru(loc: int) -> str:
    return f"""\
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %in_v %out_v
               OpExecutionMode %main OriginUpperLeft
               OpDecorate %in_v Location {loc}
               OpDecorate %out_v Location 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
   %ptr_in_v = OpTypePointer Input %v4float
  %ptr_out_v = OpTypePointer Output %v4float
       %in_v = OpVariable %ptr_in_v Input
      %out_v = OpVariable %ptr_out_v Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %v = OpLoad %v4float %in_v
               OpStore %out_v %v
               OpReturn
               OpFunctionEnd
"""


def read_raw_rgba32f(path: str) -> list[tuple[float, float, float, float]]:
    data = open(path, "rb").read()
    n = len(data) // 16
    return [struct.unpack_from("<4f", data, 16 * i) for i in range(n)]


def read_raw_u32(path: str) -> list[tuple[int, int, int, int]]:
    data = open(path, "rb").read()
    n = len(data) // 16
    return [struct.unpack_from("<4I", data, 16 * i) for i in range(n)]


def run_cmd(cmd: list[str], env: dict | None = None) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return p.returncode, (p.stdout + "\n" + p.stderr).strip()


def check_pixel_const(raw_path: str, want: tuple[int, int, int, int]) -> tuple[bool, str]:
    px = read_raw_u32(raw_path)
    if len(px) != 64 * 64:
        return False, f"expected 4096 pixels, got {len(px)}"
    bad = [(i, p) for i, p in enumerate(px) if p != want]
    if bad:
        i, p = bad[0]
        return False, f"{len(bad)}/4096 mismatches; first at ({i%64},{i//64}): got {[hex(x) for x in p]}, want {[hex(x) for x in want]}"
    return True, "4096/4096 pixels exact"


def check_bary(raw_path: str) -> tuple[bool, str]:
    px = read_raw_rgba32f(raw_path)
    bad = 0
    first = ""
    for i, (ax, ay, bi, bj) in enumerate(px):
        if abs(ax * 0.5 - bi) > 1e-4 or abs(ay * 0.5 - bj) > 1e-4:
            if not bad:
                first = f"at ({i%64},{i//64}): attr/2=({ax*0.5:.5f},{ay*0.5:.5f}) raw=({bi:.5f},{bj:.5f})"
            bad += 1
    return (bad == 0), (first if bad else "4096/4096 pixels exact")


def check_bitmask_perm(raw_path: str) -> tuple[bool, str]:
    px = read_raw_u32(raw_path)
    bad = 0
    first = ""
    judged = 0
    for i, (lane, v1, v2, v3) in enumerate(px):
        if lane >= 32:      # wave32 test: in a 64-wide subgroup lanes 32-63 hold values it never wrote
            continue
        judged += 1
        w1 = lane ^ 9
        w2 = ((lane & 5) | 16) ^ 1
        if v1 != w1 or v2 != w2 or v3 != 0:
            if not bad:
                first = f"at ({i%64},{i//64}) lane={lane}: got ({v1},{v2},{v3}), want ({w1},{w2},0)"
            bad += 1
    return (bad == 0 and judged > 0), (first if bad else f"{judged}/{judged} pixels in lanes 0-31 exact")


def check_flat(raw_path: str) -> tuple[bool, str]:
    """ps_flat with vs_ids: the integer IDs come from the first (provoking) vertex, the UVs still interpolate."""
    data = open(raw_path, "rb").read()
    px = list(struct.iter_unpack("<2I2f", data))
    w = int(len(px) ** 0.5)
    wrong = sum((a, b) != (0x101, 0x102) for a, b, _, _ in px)
    err = max(max(abs(u - (i % w + 0.5) / w), abs(v - (i // w + 0.5) / w)) for i, (_, _, u, v) in enumerate(px))
    ok = wrong == 0 and err < 1e-6
    return ok, (f"{len(px)} pixels: flat IDs exact, uv within {err:.1g}" if ok else
                f"{wrong} pixels without the first vertex's value, uv off by {err:.3g}")


def check_uniform_branch(raw_path: str) -> tuple[bool, str]:
    px = read_raw_rgba32f(raw_path)
    u32 = read_raw_u32(raw_path)
    bad = 0
    n8 = 0
    n7 = 0
    first = ""
    for i, ((_, _, bary_i, _), (s5, bit, _, _)) in enumerate(zip(px, u32)):
        if bit == 1:
            if bary_i <= 0.27 or s5 != 8:
                if not bad:
                    first = f"bit=1 at ({i%64},{i//64}): s5={s5}, i={bary_i:.4f}"
                bad += 1
        elif bit == 0:
            if bary_i > 0.2701 or s5 not in (7, 8):
                if not bad:
                    first = f"bit=0 at ({i%64},{i//64}): s5={s5}, i={bary_i:.4f}"
                bad += 1
        else:
            if not bad:
                first = f"unexpected bit={bit}"
            bad += 1
        if s5 == 8:
            n8 += 1
        elif s5 == 7:
            n7 += 1
    return (bad == 0 and n8 > 0 and n7 > 0), (first if bad else f"s5==8: {n8}, s5==7: {n7} (4096 px)")


def check_grad(raw_path: str, want_b: float, want_a: float) -> tuple[bool, str]:
    px = read_raw_rgba32f(raw_path)
    bad = 0
    first = ""
    for i, (r, g, b, a) in enumerate(px):
        wx = ((i % 64) + 0.5) / 64.0
        wy = ((i // 64) + 0.5) / 64.0
        if abs(r - wx) > 1e-3 or abs(g - wy) > 1e-3 or abs(b - want_b) > 1e-3 or abs(a - want_a) > 1e-3:
            if not bad:
                first = f"at ({i%64},{i//64}): got ({r:.4f},{g:.4f},{b:.4f},{a:.4f}), want ({wx:.4f},{wy:.4f},{want_b},{want_a})"
            bad += 1
    return (bad == 0), (first if bad else "4096/4096 pixels exact")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llvm-mc", default=os.environ.get("LLVM_MC", "llvm-mc"))
    ap.add_argument("--build", default=os.path.join(ROOT, "runtime", "build"))
    ap.add_argument("--work", default=None)
    ap.add_argument("--filter", default="")
    a = ap.parse_args()

    work = a.work or tempfile.mkdtemp(prefix="leonida_gpu_tests_")
    try:
        return run_suite(a, work)
    finally:
        if a.work is None:
            shutil.rmtree(work, ignore_errors=True)


def run_suite(a, work: str) -> int:
    fssm_dir = os.path.join(work, "fssm")
    disasm_dir = os.path.join(work, "disasm")
    spirv_dir = os.path.join(work, "spirv")
    raw_dir = os.path.join(work, "raw")
    for d in (fssm_dir, disasm_dir, spirv_dir, raw_dir):
        os.makedirs(d, exist_ok=True)

    asm_files = sorted(f for f in os.listdir(SHADER_DIR) if f.endswith(".asm"))
    if a.filter:
        asm_files = [f for f in asm_files if a.filter in f]

    # 1. Create synthetic DXBC / FSSM metadata + assemble .asm -> .s
    for fname in asm_files:
        name = os.path.splitext(fname)[0]
        if name == "gs_tri":
            dxbc = make_dxbc([
                (b"ISG1", make_sig_chunk([(1, 3, 0, 0xF, 0xF), (0, 3, 1, 0xF, 0xF)])),
                (b"OSG1", make_sig_chunk([(1, 3, 0, 0xF, 0xF), (0, 3, 1, 0xF, 0xF)])),
            ])
        elif name == "ms_tri":
            dxbc = make_dxbc([(b"PSV0", make_ms_psv0(128, 64, 64, 2))])
        else:
            dxbc = make_dxbc([])
        with open(os.path.join(work, f"{name}.dxbc"), "wb") as f:
            f.write(dxbc)
        if name.startswith("cs_"):
            with open(os.path.join(fssm_dir, f"{name}.fssm"), "wb") as f:
                f.write(make_cs_fssm())
        assemble_to_s(a.llvm_mc, os.path.join(SHADER_DIR, fname), os.path.join(disasm_dir, f"{name}.s"), name)

    # 2. Translate all .s -> .spv
    rc, out = run_cmd([sys.executable, "-I", TRANSLATOR, disasm_dir, "--out", spirv_dir])
    print("=== rdna2spirv.py ===")
    print(out)
    if rc != 0:
        return 1

    # 3. Build helper SPIR-V shaders
    vs_tri = os.path.join(spirv_dir, "vs_tri.spv")
    fs_passthru = os.path.join(spirv_dir, "fs_passthru.spv")
    fs_loc1 = os.path.join(spirv_dir, "fs_loc1.spv")
    spirv_as(VS_TRI_SPVASM, vs_tri)
    spirv_as(make_fs_passthru(0), fs_passthru)
    spirv_as(make_fs_passthru(1), fs_loc1)

    index = {}
    idx_path = os.path.join(spirv_dir, "index.json")
    if os.path.isfile(idx_path):
        index = json.load(open(idx_path, encoding="utf-8"))

    ext = ".exe" if os.name == "nt" else ""
    vk_compute = os.path.join(a.build, "vk_compute_test" + ext)
    vk_store = os.path.join(a.build, "vk_store_test" + ext)
    vk_harness = os.path.join(a.build, "vk_harness" + ext)

    results: list[tuple[str, bool, str]] = []
    for fname in asm_files:
        name = os.path.splitext(fname)[0]
        spv = os.path.join(spirv_dir, f"{name}.spv")
        if not os.path.isfile(spv):
            results.append((name, False, "missing .spv (translation failed)"))
            continue
        sg = index.get(name, {}).get("subgroup_size")
        cs_env = dict(os.environ, VKC_SUBGROUP_SIZE=str(sg)) if sg else None
        gfx_env = dict(os.environ, VKH_SUBGROUP_SIZE=str(sg)) if sg else None

        if name == "cs_lds":
            rc, msg = run_cmd([vk_compute, spv, "4", "mirror64"], env=cs_env)
            results.append((name, rc == 0, msg.splitlines()[-1] if msg else ""))
        elif name == "cs_exec64":
            rc, msg = run_cmd([vk_compute, spv, "1", "exec64"], env=cs_env)
            results.append((name, rc == 0, msg.splitlines()[-1] if msg else ""))
        elif name == "cs_bvh":
            rc, msg = run_cmd([vk_compute, spv, "1", "bvh"], env=cs_env)
            results.append((name, rc == 0, msg.splitlines()[-1] if msg else ""))
        elif name == "cs_wavemask":
            # Whole-wave mask test, as in check.py: needs a 64-wide subgroup, skipped where the device has none.
            rc, msg = run_cmd([vk_compute, spv, "1", "wavemask", "--require-wave64"], env=cs_env)
            if rc == 77:
                results.append((name, None, "SKIP (no 64-wide compute subgroup on this device)"))
            elif rc == 0 and "compute subgroup size: 64" not in msg.splitlines():
                results.append((name, False, "64-wide subgroup was not confirmed (rebuild vk_compute_test)"))
            else:
                results.append((name, rc == 0, msg.splitlines()[-1] if msg else ""))
        elif name in ("cs_store", "cs_wave64_ballot", "cs_wm_half_ops", "cs_wm_half_cmp", "cs_wm_bfm64", "cs_wm_scc_rfl"):
            cs_dump_wants = {
                "cs_store": [0x400080FF, 0xE00003FF, 0x3C004000, 0xEEEEEEEE, 0x40003C00, 0xBC003800],
                "cs_wave64_ballot": [32, 16, 24, 40, 8, 48],
                "cs_wm_half_ops": [32, 16, 16, 0, 16],
                "cs_wm_half_cmp": [1, 0, 1, 1],
                "cs_wm_bfm64": [16, 8, 8, 40],
                "cs_wm_scc_rfl": [55, 48],
            }
            want = cs_dump_wants[name]
            # Every dump but cs_store is a whole-wave (wave64) mask test: ask for a 64-wide subgroup and skip, as
            # check.py does, where the device has none (vk_compute_test exits 77) instead of reporting a failure.
            wave64 = name != "cs_store"
            rc, msg = run_cmd([vk_compute, spv, "1", "dump"] + (["--require-wave64"] if wave64 else []), env=cs_env)
            dump_line = next((ln for ln in msg.splitlines() if "out[0..7] =" in ln), "")
            if wave64 and rc == 77:
                results.append((name, None, "SKIP (no 64-wide compute subgroup on this device)"))
            elif rc == 0 and dump_line:
                vals = [int(x) for x in dump_line.split("=")[1].split()[:len(want)]]
                ok = vals == want
                fmt = hex if name == "cs_store" else str
                results.append((name, ok, " ".join(fmt(x) for x in vals)))
            else:
                results.append((name, False, msg))
        elif name == "cs_imgstore":
            rc, msg = run_cmd([vk_store, spv])
            results.append((name, rc == 0, msg.splitlines()[-1] if msg else ""))
        elif name == "cs_imgload":
            rc, msg = run_cmd([vk_store, spv, "load"])
            results.append((name, rc == 0, msg.splitlines()[-1] if msg else ""))
        elif name.startswith("ps_"):
            raw = os.path.join(raw_dir, f"{name}.raw")
            vs = os.path.join(spirv_dir, "vs_ids.spv") if name == "ps_flat" else vs_tri   # check.py draws ps_flat with vs_ids
            rc, msg = run_cmd([vk_harness, vs, spv, raw], env=gfx_env)
            if rc != 0:
                results.append((name, False, msg))
                continue
            const_wants = {
                "ps_ops": (0x48804300, 5, 7, 0xFFFFFFFC),
                "ps_ops2": (0xABCD4400, 0x46001234, 0xFFFFFFFA, 7),
                "ps_ops3": (0xFFFFFFFA, 0x44660066, 0x44AABBCC, 16),
                "ps_ops4": (0x34000010, 0x00120800, 0xBEEF0000, 0x12343C00),
                "ps_scc": (1, 1, 3, 0x80000000),
                "ps_sdwa": (0x000000F1, 0xFFFEFFF2, 0x3C003344, 0xFFF70000),
                "ps_legacy_mul": (0, 0, 0, 0x40C00000),        # same expectation as check.py
            }
            if name in const_wants:
                ok, detail = check_pixel_const(raw, const_wants[name])
            elif name == "ps_bary":
                ok, detail = check_bary(raw)
            elif name == "ps_bitmask_perm":
                ok, detail = check_bitmask_perm(raw)
            elif name == "ps_uniform_branch":
                ok, detail = check_uniform_branch(raw)
            elif name == "ps_flat":
                ok, detail = check_flat(raw)
            else:
                ok, detail = False, "no checker defined"
            results.append((name, ok, detail))
        elif name == "gs_tri":
            raw = os.path.join(raw_dir, f"{name}.raw")
            env = dict(gfx_env or os.environ, VKH_GS=spv)
            rc, msg = run_cmd([vk_harness, vs_tri, fs_loc1, raw], env=env)
            if rc != 0:
                results.append((name, False, msg))
            else:
                ok, detail = check_grad(raw, 0.0, 0.0)
                results.append((name, ok, detail))
        elif name == "ms_tri":
            raw = os.path.join(raw_dir, f"{name}.raw")
            env = dict(gfx_env or os.environ, VKH_MESH="1")
            rc, msg = run_cmd([vk_harness, spv, fs_passthru, raw], env=env)
            if rc != 0:
                results.append((name, False, msg))
            else:
                ok, detail = check_grad(raw, 0.5, 1.0)
                results.append((name, ok, detail))

    print("\n=== GPU TEST RESULTS ===")
    passed = skipped = 0
    for name, ok, detail in results:
        tag = "SKIP" if ok is None else "PASS" if ok else "FAIL"
        passed += ok is True
        skipped += ok is None
        print(f"  {tag:4s}  {name:<20s}  {detail}")
    print(f"\n{passed}/{len(results)} passed" + (f", {skipped} skipped" if skipped else ""))
    return 0 if passed + skipped == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
