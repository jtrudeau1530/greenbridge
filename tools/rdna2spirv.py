"""RDNA (gfx10) shader -> SPIR-V assembly, version 0.

Model: one SPIR-V invocation runs one lane. VGPRs and SGPRs are Function-scope uint arrays; float ops bitcast,
so register contents keep exact bit patterns. EXEC is treated as all-ones (valid for branch-free code).

Supported in v0 (anything else is reported, never guessed):
  vector ALU  v_mov_b32 v_add/sub/subrev/mul/mul_legacy/fma/fmac/fmaak/fmamk/mac/max/min_f32 v_rcp/rsq/sqrt/
              exp/log/floor/fract/ceil/trunc_f32 v_cvt_f32_u32/i32/ubyte0..3 v_cvt_u32/i32_f32
              v_add/sub_nc_u32 v_and/or/xor_b32 v_lshlrev/lshrrev/ashrrev_i32 v_cndmask_b32 v_mul_u32_u24
              v_bfe_u32 v_med3_f32 v_mad_u32_u24
  scalar      s_mov_b32 s_add_u32 s_and_b32 s_or_b32 s_lshl_b32 s_lshr_b32 s_mul_i32
  no-ops      s_waitcnt* s_nop s_inst_prefetch s_endpgm s_setprio s_clause s_delay_alu
  export      exp pos0..3 / param0..31 (vertex) and mrt0..7 (pixel)
Vertex inputs: v0 = VertexIndex (the full-screen-triangle shaders read only that).

Usage (needs spirv-as/spirv-val on PATH, from the Vulkan SDK or spirv-tools):
    python3 -I rdna2spirv.py <disasm dir or .s> --out DIR [--limit N]
Writes <name>.spvasm and <name>.spv for each translated shader and prints coverage + validation results.
"""
import argparse
import os
import re
import struct
import subprocess
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from shader_ir import parse_file  # noqa: E402

# GFX10 buffer format codes (Mesa src/amd/registers/gfx10-rsrc.json, enum GFX10_FORMAT).
import json as _json  # noqa: E402
GFX10_FORMATS = {int(k): v for k, v in _json.load(open(os.path.join(os.path.dirname(__file__), "gfx10_formats.json"))).items()}

# User SGPRs s0..s(USER_SGPRS-1) are initialised from push constants (128 bytes = Vulkan's guaranteed minimum).
USER_SGPRS = 32

NOOPS = {"s_sleep", "v_nop", "s_nop", "s_inst_prefetch", "s_endpgm", "s_setprio", "s_clause", "s_delay_alu", "s_waitcnt",
         "s_waitcnt_vscnt", "s_waitcnt_vmcnt", "s_waitcnt_expcnt", "s_waitcnt_lgkmcnt", "s_waitcnt_depctr"}

F2 = {"v_add_f32": "OpFAdd", "v_sub_f32": "OpFSub", "v_mul_f32": "OpFMul", "v_mul_legacy_f32": "OpFMul",
      "v_max_f32": "max", "v_min_f32": "min"}
U2 = {"v_add_nc_u32": "OpIAdd", "v_sub_nc_u32": "OpISub", "v_and_b32": "OpBitwiseAnd", "v_or_b32": "OpBitwiseOr",
      "v_xor_b32": "OpBitwiseXor", "v_mul_u32_u24": "mul24"}
SHIFT_REV = {"v_lshlrev_b32": "OpShiftLeftLogical", "v_lshrrev_b32": "OpShiftRightLogical",
             "v_ashrrev_i32": "OpShiftRightArithmetic"}
F1 = {"v_rcp_f32": "rcp", "v_rsq_f32": "InverseSqrt", "v_sqrt_f32": "Sqrt", "v_exp_f32": "Exp2",
      "v_log_f32": "Log2", "v_floor_f32": "Floor", "v_fract_f32": "Fract", "v_ceil_f32": "Ceil",
      "v_trunc_f32": "Trunc"}
S2 = {"s_and_b32": "OpBitwiseAnd", "s_or_b32": "OpBitwiseOr",
      "s_lshl_b32": "OpShiftLeftLogical", "s_lshr_b32": "OpShiftRightLogical", "s_mul_i32": "OpIMul"}


def base_op(op):
    for suf in ("_e32", "_e64", "_sdwa", "_dpp"):
        if op.endswith(suf):
            return op[: -len(suf)]
    return op


class Unsupported(Exception):
    pass


class Gen:
    def __init__(self, stage):
        self.stage = stage
        self.body = []
        self.consts = {}
        self.n = 0
        self.outputs = {}
        self.inputs = {}
        self.uses_psb = False
        self.uses_active = False
        self.uses_tex = False
        self.uses_subgroup = False
        self.uses_tex_array = False
        self.uses_storage = False
        self.uses_query = False
        self.uses_tex_1d = False
        self.uses_tex_3d = False
        self.uses_v4uint = False
        self.uses_store_views = set()  # image_store dims used (index into STORE_DIMS)
        self.writes_depth = False
        self.uses_smul = False
        self.uses_minlod = False
        self.uses_img_offset = False
        self.uses_pops = False
        self.uses_vote = False         # wave-uniform branches (OpGroupNonUniformAny)
        self.uses_lds = False
        self.lds_dwords = 16384        # CS: set from RSRC2 LDS_SIZE (power of two, covers every offset used)
        self.cs = None             # CS: user SGPR count, enabled workgroup ids, thread-id VGPR count, group size
        self.uses_storage3d = False
        self.uses_heap = False
        self.uses_codebase = False
        self.uses_long = False
        self.vertex_inputs = []    # (D3D input register, component type 1=uint 2=sint 3=float) for fetch-calling VS
        self.frag_inputs = {}      # PS: VGPR -> gl_FragCoord component, from the verified system-VGPR layout
        self.blocks = []      # per basic block: list of SPIR-V lines (dispatch-loop form), empty = straight-line
        self.block_label = "%entry"   # label of the SPIR-V block the current instruction starts in

    def tmp(self):
        self.n += 1
        return f"%t{self.n}"

    def cu(self, v):
        v &= 0xFFFFFFFF
        if v not in self.consts:
            self.consts[v] = f"%c{v:x}"
        return self.consts[v]

    def emit(self, s):
        self.body.append("         " + s)

    # ---- register access ----
    def load_reg(self, kind, r):
        arr = "%vgpr" if kind == "vgpr" else "%sgpr"
        p, t = self.tmp(), self.tmp()
        self.emit(f"{p} = OpAccessChain %_ptr_Function_uint {arr} {self.cu(r)}")
        self.emit(f"{t} = OpLoad %uint {p}")
        return t

    def store_reg(self, kind, r, val):
        arr = "%vgpr" if kind == "vgpr" else "%sgpr"
        p = self.tmp()
        self.emit(f"{p} = OpAccessChain %_ptr_Function_uint {arr} {self.cu(r)}")
        self.emit(f"OpStore {p} {val}")

    def src_u(self, a):
        k = a["k"]
        if k in ("vgpr", "sgpr"):
            if a.get("n", 1) != 1:
                raise Unsupported("64-bit operand")
            v = self.load_reg(k, a["r"])
        elif k == "special" and a["t"] == "null":
            v = self.cu(0)                # RDNA 'null' source reads as zero (e.g. soffset)
        elif k == "special" and a["t"] in ("exec_lo", "exec", "exec_hi"):
            v = self.exec_value()
        elif k == "special" and a["t"] == "m0":
            v = self.tmp()
            self.emit(f"{v} = OpLoad %uint %m0")
        elif k == "special" and a["t"] == "vcc_hi":
            # wave32: vcc_hi is an ordinary SGPR (the lane mask lives in vcc_lo).
            v = self.tmp()
            self.emit(f"{v} = OpLoad %uint %vcc_hi")
        elif k == "special" and a["t"] == "scc":
            v = self.tmp()
            self.emit(f"{v} = OpLoad %uint %scc")
        elif k == "special" and a["t"] == "src_pops_exiting_wave_id":
            # POPS (primitive-ordered pixel shading): the hardware wait loop compares this against the
            # collision word. Ordering comes from fragment-shader interlock around the whole shader instead
            # (see assemble), so the id reads as 0 and the runtime puts 0 (no collision) in the collision SGPR.
            v = self.cu(0)
            self.uses_pops = True
        elif k == "imm":
            v = self.cu(a["v"])
        elif k == "fimm":
            v = self.cu(struct.unpack("<I", struct.pack("<f", a["v"]))[0])
        elif k == "special" and a["t"] in ("vcc_lo", "vcc"):
            v = self.tmp()
            self.emit(f"{v} = OpLoad %uint %vcc")
        else:
            raise Unsupported(f"operand {a.get('t', k)}")
        if "sdwa_sel" in a:
            v = self.sdwa_src(v, a)
        return self.modify(v, a)

    def modify(self, v, a):
        if a.get("abs") or a.get("neg"):
            f = self.tof(v)
            if a.get("abs"):
                g = self.tmp()
                self.emit(f"{g} = OpExtInst %float %glsl FAbs {f}")
                f = g
            if a.get("neg"):
                g = self.tmp()
                self.emit(f"{g} = OpFNegate %float {f}")
                f = g
            v = self.tou(f)
        return v

    def write_lo16(self, d, lo16, preserve_high=True):
        """Write a 16-bit result into the low half of VGPR d. On GFX10, f16 VOP1/VOP2 ops and opsel-capable VOP3
        ops are partial writes that keep the high half (Mesa ACO aco_ir.cpp: instr_is_16bit / can_use_opsel);
        non-opsel VOP2 16-bit integer ops define the whole register (high half modelled as zero)."""
        if not preserve_high or d["k"] != "vgpr":
            self.dst(d, lo16)
            return
        old, keep, r = self.tmp(), self.tmp(), self.tmp()
        self.emit(f"{old} = OpCopyObject %uint {self.load_reg('vgpr', d['r'])}")
        self.emit(f"{keep} = OpBitwiseAnd %uint {old} {self.cu(0xFFFF0000)}")
        self.emit(f"{r} = OpBitwiseOr %uint {keep} {lo16}")
        self.dst(d, r)

    def f16_rtz(self, f):
        """f32 -> f16 bits (low 16) with round-toward-zero, as v_cvt_pkrtz does: pack to nearest, and if that
        rounded away from zero (|half| > |x|), step one f16 ulp back toward zero. Inf/NaN inputs keep the nearest
        result (inf/NaN). Finite values above the f16 range become +-65504, never inf."""
        v2, p, lo, back2, back, ab, ax, gt, fin, nz, dec, r = (self.tmp() for _ in range(12))
        self.emit(f"{v2} = OpCompositeConstruct %v2float {f} %f0")
        self.emit(f"{p} = OpExtInst %uint %glsl PackHalf2x16 {v2}")
        self.emit(f"{lo} = OpBitwiseAnd %uint {p} {self.cu(0xFFFF)}")
        self.emit(f"{back2} = OpExtInst %v2float %glsl UnpackHalf2x16 {lo}")
        self.emit(f"{back} = OpCompositeExtract %float {back2} 0")
        self.emit(f"{ab} = OpExtInst %float %glsl FAbs {back}")
        self.emit(f"{ax} = OpExtInst %float %glsl FAbs {f}")
        self.emit(f"{gt} = OpFOrdGreaterThan %bool {ab} {ax}")       # false for NaN
        isinf, isnan = self.tmp(), self.tmp()
        self.emit(f"{isinf} = OpIsInf %bool {f}")
        self.emit(f"{isnan} = OpIsNan %bool {f}")
        special = self.tmp()
        self.emit(f"{special} = OpLogicalOr %bool {isinf} {isnan}")
        self.emit(f"{fin} = OpLogicalNot %bool {special}")
        self.emit(f"{nz} = OpLogicalAnd %bool {gt} {fin}")
        self.emit(f"{dec} = OpISub %uint {lo} {self.cu(1)}")          # magnitude bits - 1 = one ulp toward zero
        self.emit(f"{r} = OpSelect %uint {nz} {dec} {lo}")
        return r

    def modify_f(self, f, a):
        return self.tof(self.modify(self.tou(f), a))

    def host_addr(self, xbox_va):
        """Xbox GPU virtual address (ulong id) -> host buffer device address: heap base (push constant
        dwords 32..33) + VA. The runtime maps the console's GPU address window into one host buffer."""
        lo, hi, l64, h64, sh, base, r = (self.tmp() for _ in range(7))
        self.emit(f"{lo} = OpLoad %uint %ud_base_lo_p")
        self.emit(f"{hi} = OpLoad %uint %ud_base_hi_p")
        self.emit(f"{l64} = OpUConvert %ulong {lo}")
        self.emit(f"{h64} = OpUConvert %ulong {hi}")
        self.emit(f"{sh} = OpShiftLeftLogical %ulong {h64} %c64_32")
        self.emit(f"{base} = OpBitwiseOr %ulong {sh} {l64}")
        self.emit(f"{r} = OpIAdd %ulong {base} {xbox_va}")
        self.uses_heap = True
        self.cu(1), self.cu(2)        # member indices of the push-constant block, declared before assembly
        return r

    def tof(self, u):
        f = self.tmp()
        self.emit(f"{f} = OpBitcast %float {u}")
        return f

    def tou(self, f):
        u = self.tmp()
        self.emit(f"{u} = OpBitcast %uint {f}")
        return u

    def src_f(self, a):
        return self.tof(self.src_u(a))

    # ---- EXEC (this lane's active bit) ----
    def active_b(self):
        b = self.tmp()
        self.emit(f"{b} = OpLoad %bool %active")
        return b

    def exec_value(self):
        """EXEC as a lane mask for this invocation: all-ones if active, else 0."""
        r = self.tmp()
        self.emit(f"{r} = OpSelect %uint {self.active_b()} {self.cu(0xFFFFFFFF)} {self.cu(0)}")
        return r

    def set_exec(self, mask_u):
        self.emit(f"OpStore %active {self.tmp_bool_nz(mask_u)}")
        self.uses_active = True

    def dst(self, a, val):
        if a.get("sdwa_dst") and a["k"] == "vgpr":
            val = self.sdwa_dst(a, val)
        if a["k"] == "special" and a["t"] == "null":
            return                                    # result discarded
        if a["k"] == "special" and a["t"] in ("exec_lo", "exec"):
            self.set_exec(val)
            return
        if a["k"] == "vgpr" and a.get("n", 1) == 1:
            # VALU writes only land in active lanes.
            p, old, r = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{p} = OpAccessChain %_ptr_Function_uint %vgpr {self.cu(a['r'])}")
            self.emit(f"{old} = OpLoad %uint {p}")
            self.emit(f"{r} = OpSelect %uint {self.active_b()} {val} {old}")
            self.emit(f"OpStore {p} {r}")
        elif a["k"] == "sgpr" and a.get("n", 1) == 1:
            self.store_reg(a["k"], a["r"], val)
        elif a["k"] == "special" and a["t"] in ("vcc_lo", "vcc"):
            self.emit(f"OpStore %vcc {val}")
        elif a["k"] == "special" and a["t"] == "m0":
            self.emit(f"OpStore %m0 {val}")
        elif a["k"] == "special" and a["t"] == "vcc_hi":
            self.emit(f"OpStore %vcc_hi {val}")
        elif a["k"] == "special" and a["t"] == "scc":
            self.emit(f"OpStore %scc {val}")
        else:
            raise Unsupported(f"destination {a.get('t', a['k'])}")

    def mov64(self, d, s):
        """64-bit scalar move, including EXEC save/restore (EXEC = this lane's active bit)."""
        if d["k"] == "special" and d["t"] == "exec":
            self.set_exec(self.mask_src(s))
            return
        if d["k"] == "special" and d["t"] in ("vcc", "vcc_lo"):
            self.emit(f"OpStore %vcc {self.mask_src(s)}")
            return
        if d["k"] != "sgpr" or d.get("n") != 2:
            raise Unsupported("s_mov_b64 destination")
        if s["k"] == "special" and s["t"] == "exec":
            lo = hi = self.exec_value()
        elif s["k"] == "sgpr" and s.get("n") == 2:
            lo, hi = self.load_reg("sgpr", s["r"]), self.load_reg("sgpr", s["r"] + 1)
        elif s["k"] == "imm":
            v = s["v"] & 0xFFFFFFFFFFFFFFFF
            if s["v"] < 0:
                v = s["v"] & 0xFFFFFFFFFFFFFFFF
            lo, hi = self.cu(v & 0xFFFFFFFF), self.cu(v >> 32)
        elif s["k"] == "special" and s["t"] == "vcc":
            lo = self.tmp()
            self.emit(f"{lo} = OpLoad %uint %vcc")
            hi = self.cu(0)
        else:
            raise Unsupported("s_mov_b64 source")
        self.store_reg("sgpr", d["r"], lo)
        self.store_reg("sgpr", d["r"] + 1, hi)

    def compare(self, op, a, scalar, exec_only=False):
        """v_cmp_* / s_cmp_*: lane result into VCC/SGPR mask (vector), SCC (scalar), or EXEC (v_cmpx)."""
        parts = op.split("_")          # v_cmp_lt_f32 / s_cmp_eq_u32
        cond, ty = parts[2], parts[3]
        srcs = a[0:2] if scalar else a[-2:]
        dst = None if (scalar or exec_only) else a[0]
        if ty in ("f32", "f16"):
            if ty == "f32":
                x, y = self.src_f(srcs[0]), self.src_f(srcs[1])
            else:   # f16 compare: the low half of each source, compared as f32 (exact for f16 values)
                vals = []
                for s in srcs:
                    u = self.src_u({**s, "neg": False, "abs": False})
                    v2, f = self.tmp(), self.tmp()
                    self.emit(f"{v2} = OpExtInst %v2float %glsl UnpackHalf2x16 {u}")
                    self.emit(f"{f} = OpCompositeExtract %float {v2} 0")
                    vals.append(self.modify_f(f, s))
                x, y = vals
            ops = {"lt": "OpFOrdLessThan", "le": "OpFOrdLessThanEqual", "gt": "OpFOrdGreaterThan",
                   "ge": "OpFOrdGreaterThanEqual", "eq": "OpFOrdEqual", "neq": "OpFUnordNotEqual",
                   "lg": "OpFOrdNotEqual", "nlt": "OpFUnordGreaterThanEqual", "ngt": "OpFUnordLessThanEqual",
                   "nle": "OpFUnordGreaterThan", "nge": "OpFUnordLessThan"}
        elif ty in ("u32", "i32"):
            x, y = self.src_u(srcs[0]), self.src_u(srcs[1])
            if ty == "i32":
                x, y = self.as_int(x), self.as_int(y)
            s = ty == "i32"
            ops = {"lt": "OpSLessThan" if s else "OpULessThan", "le": "OpSLessThanEqual" if s else "OpULessThanEqual",
                   "gt": "OpSGreaterThan" if s else "OpUGreaterThan",
                   "ge": "OpSGreaterThanEqual" if s else "OpUGreaterThanEqual",
                   "eq": "OpIEqual", "ne": "OpINotEqual", "lg": "OpINotEqual"}
        else:
            raise Unsupported(op)
        if cond not in ops:
            raise Unsupported(op)
        b, r = self.tmp(), self.tmp()
        self.emit(f"{b} = {ops[cond]} %bool {x} {y}")
        # Lane masks are all-ones/zero per invocation (one lane), SCC is 1/0. A VOPC result bit is 0 for lanes
        # that are off in EXEC (as on hardware), so wave-wide reductions of the mask (s_cbranch_vccz) are exact.
        bm = b
        if not scalar and not exec_only:
            bm = self.tmp()
            self.emit(f"{bm} = OpLogicalAnd %bool {b} {self.active_b()}")
        self.emit(f"{r} = OpSelect %uint {bm} {self.cu(1 if scalar else 0xFFFFFFFF)} {self.cu(0)}")
        if scalar:
            self.emit(f"OpStore %scc {r}")
        elif dst is None:          # v_cmpx: EXEC &= cond
            a2 = self.tmp()
            self.emit(f"{a2} = OpLoad %bool %active")
            a3 = self.tmp()
            self.emit(f"{a3} = OpLogicalAnd %bool {a2} {b}")
            self.emit(f"OpStore %active {a3}")
            self.uses_active = True
        elif dst["k"] == "special" and dst["t"] in ("vcc", "vcc_lo"):
            self.emit(f"OpStore %vcc {r}")
        elif dst["k"] == "sgpr":
            self.store_reg("sgpr", dst["r"], r)
            if dst.get("n") == 2:
                self.store_reg("sgpr", dst["r"] + 1, r)
        else:
            raise Unsupported("v_cmp destination")

    def compare_x(self, op, a):
        """v_cmpx_*: like v_cmp but the result ANDs into EXEC. e32 form has only the two sources; e64 also names
        a destination (vcc/sgpr) which gets the compare result too."""
        srcs = a[-2:]
        self.compare("v_cmp_" + op[len("v_cmpx_"):], [None] + list(srcs), scalar=False, exec_only=True)

    def swizzle(self, a, mods, ins):
        """ds_swizzle_b32 vD, vS offset:swizzle(...): a subgroup shuffle (one invocation per lane)."""
        line = " ".join(m for m in mods) + " " + " ".join(x.get("t", "") for x in a if x["k"] in ("special", "raw"))
        import re as _re
        m = _re.search(r"swizzle\((\w+)((?:,\s*\d+)*)\)", line + " " + str(ins.get("raw", "")))
        if not m:
            raise Unsupported("ds_swizzle pattern")
        kind = m.group(1)
        args = [int(x) for x in m.group(2).split(",")[1:]] if m.group(2) else []
        self.uses_subgroup = True
        lane = self.tmp()
        self.emit(f"{lane} = OpLoad %uint %sg_lane")
        if kind == "QUAD_PERM":
            sel, base, src = self.tmp(), self.tmp(), self.cu(0)
            self.emit(f"{sel} = OpBitwiseAnd %uint {lane} {self.cu(3)}")
            self.emit(f"{base} = OpBitwiseAnd %uint {lane} {self.cu(0xFFFFFFFC)}")
            pick = self.cu(args[0])
            for i in (1, 2, 3):
                e, nxt = self.tmp(), self.tmp()
                self.emit(f"{e} = OpIEqual %bool {sel} {self.cu(i)}")
                self.emit(f"{nxt} = OpSelect %uint {e} {self.cu(args[i])} {pick}")
                pick = nxt
            src = self.tmp()
            self.emit(f"{src} = OpBitwiseOr %uint {base} {pick}")
        elif kind == "SWAP":
            src = self.tmp()
            self.emit(f"{src} = OpBitwiseXor %uint {lane} {self.cu(args[0])}")
        elif kind == "REVERSE":
            src = self.tmp()
            self.emit(f"{src} = OpBitwiseXor %uint {lane} {self.cu(args[0] - 1)}")
        elif kind == "BROADCAST":
            src, keep = self.tmp(), self.tmp()
            self.emit(f"{keep} = OpBitwiseAnd %uint {lane} {self.cu(~(args[0] - 1) & 0xFFFFFFFF)}")
            self.emit(f"{src} = OpBitwiseOr %uint {keep} {self.cu(args[1])}")
        else:
            raise Unsupported("ds_swizzle " + kind)
        v, r = self.src_u(a[1]), self.tmp()
        self.emit(f"{r} = OpGroupNonUniformShuffle %uint %c3 {v} {src}")
        self.dst(a[0], r)

    def buffer_load(self, op, a, mods):
        """buffer_load_dword[xN] vD, vaddr, s[V#], soffset [offen|idxen] offset:N  -> buffer-device-address loads.
        addr = V#.base + soffset + inst_offset + (offen ? vaddr : 0) + (idxen ? vindex * V#.stride : 0)."""
        if any(m in mods for m in ("lds", "tfe")) or "format" in op:
            raise Unsupported(op + " form")
        n = 1 if op.endswith("dword") else int(op.rsplit("x", 1)[1])
        vd, vaddr, rsrc, soff = a[0], a[1], a[2], a[3] if len(a) > 3 else {"k": "imm", "v": 0}
        if rsrc["k"] != "sgpr" or rsrc.get("n") != 4:
            raise Unsupported("buffer_load V#")
        d0, d1 = self.load_reg("sgpr", rsrc["r"]), self.load_reg("sgpr", rsrc["r"] + 1)
        hi, stride = self.tmp(), self.tmp()
        self.emit(f"{hi} = OpBitwiseAnd %uint {d1} {self.cu(0xFFFF)}")
        self.emit(f"{stride} = OpBitFieldUExtract %uint {d1} {self.cu(16)} {self.cu(14)}")
        off = self.src_u(soff)
        inst = next((int(m.split(":")[1], 0) for m in mods if m.startswith("offset:")), 0)
        total = self.tmp()
        self.emit(f"{total} = OpIAdd %uint {off} {self.cu(inst)}")
        offen, idxen = "offen" in mods, "idxen" in mods
        vregs = [] if vaddr["k"] == "special" else [vaddr["r"] + i for i in range(vaddr.get("n", 1))]
        if idxen:
            idx = self.load_reg("vgpr", vregs[0])
            m, t = self.tmp(), self.tmp()
            self.emit(f"{m} = OpIMul %uint {idx} {stride}")
            self.emit(f"{t} = OpIAdd %uint {total} {m}")
            total = t
        if offen:
            o = self.load_reg("vgpr", vregs[1] if idxen else vregs[0])
            t = self.tmp()
            self.emit(f"{t} = OpIAdd %uint {total} {o}")
            total = t
        l64, h64, sh, addr, o64, full, ptr = (self.tmp() for _ in range(7))
        self.emit(f"{l64} = OpUConvert %ulong {d0}")
        self.emit(f"{h64} = OpUConvert %ulong {hi}")
        self.emit(f"{sh} = OpShiftLeftLogical %ulong {h64} %c64_32")
        self.emit(f"{addr} = OpBitwiseOr %ulong {sh} {l64}")
        self.emit(f"{o64} = OpUConvert %ulong {total}")
        self.emit(f"{full} = OpIAdd %ulong {addr} {o64}")
        self.emit(f"{ptr} = OpConvertUToPtr %_ptr_PSB_arr {self.host_addr(full)}")
        self.uses_psb = True
        for i in range(n):
            p, v = self.tmp(), self.tmp()
            self.emit(f"{p} = OpAccessChain %_ptr_PSB_uint {ptr} {self.cu(i)}")
            self.emit(f"{v} = OpLoad %uint {p} Aligned 4")
            self.dst({"k": "vgpr", "r": vd["r"] + i, "n": 1}, v)

    # GFX10 buffer formats (from Mesa src/amd/registers/gfx10-rsrc.json, GFX10_FORMAT):
    #   code: (bytes, kind, component bit widths)
    FORMATS = {22: (4, "f32", (32,)), 64: (8, "f32", (32, 32)), 74: (12, "f32", (32, 32, 32)),
               77: (16, "f32", (32, 32, 32, 32)), 20: (4, "u32", (32,)), 62: (8, "u32", (32, 32)),
               75: (16, "u32", (32, 32, 32, 32)), 13: (2, "f16", (16,)), 29: (4, "f16", (16, 16)),
               71: (8, "f16", (16, 16, 16, 16)), 1: (1, "unorm", (8,)), 14: (2, "unorm", (8, 8)),
               56: (4, "unorm", (8, 8, 8, 8)), 57: (4, "snorm", (8, 8, 8, 8)), 60: (4, "uint", (8, 8, 8, 8)),
               7: (2, "unorm", (16,)), 23: (4, "unorm", (16, 16)), 24: (4, "snorm", (16, 16)),
               65: (8, "unorm", (16, 16, 16, 16)), 66: (8, "snorm", (16, 16, 16, 16)),
               44: (4, "unorm", (10, 10, 10, 2)), 50: (4, "unorm", (2, 10, 10, 10)),
               69: (8, "uint", (16, 16, 16, 16)), 27: (4, "uint", (16, 16))}

    def decode_format(self, words, code):
        """Decode one element of format `code` from raw dwords -> 4 uint ids holding f32 bits (or uint for *_UINT/uint)."""
        size, kind, widths = self.FORMATS[code]
        if kind in ("f32", "u32"):
            vals = list(words[:len(widths)])
            if kind == "u32":
                vals = vals    # integers are passed through as their bits
            return vals + [self.cu(0)] * (3 - len(vals) + 1 if len(vals) < 4 else 0)
        out, bit = [], 0
        for w in widths:
            src = words[bit // 32]
            f = self.tmp()
            self.emit(f"{f} = OpBitFieldUExtract %uint {src} {self.cu(bit % 32)} {self.cu(w)}")
            if kind == "f16":
                v2, x = self.tmp(), self.tmp()
                self.emit(f"{v2} = OpExtInst %v2float %glsl UnpackHalf2x16 {f}")
                self.emit(f"{x} = OpCompositeExtract %float {v2} 0")
                out.append(self.tou(x))
            elif kind == "unorm":
                c, x = self.tmp(), self.tmp()
                self.emit(f"{c} = OpConvertUToF %float {f}")
                self.emit(f"{x} = OpFMul %float {c} {self.fconst(1.0 / ((1 << w) - 1))}")
                out.append(self.tou(x))
            elif kind == "snorm":
                si, c, x, cl = self.tmp(), self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{si} = OpBitFieldSExtract %int {self.as_int(src)} {self.cu(bit % 32)} {self.cu(w)}")
                self.emit(f"{c} = OpConvertSToF %float {si}")
                self.emit(f"{x} = OpFMul %float {c} {self.fconst(1.0 / ((1 << (w - 1)) - 1))}")
                self.emit(f"{cl} = OpExtInst %float %glsl FMax {x} {self.fconst(-1.0)}")
                out.append(self.tou(cl))
            else:   # uint: raw integer
                out.append(f)
            bit += w
        one = self.cu(0x3F800000) if kind in ("f16", "unorm", "snorm") else self.cu(1)
        while len(out) < 4:
            out.append(self.cu(0) if len(out) < 3 else one)
        return out

    BUF_ATOMICS = {"add": ("OpAtomicIAdd", 0), "sub": ("OpAtomicISub", 0), "umin": ("OpAtomicUMin", 0xFFFFFFFF),
                   "umax": ("OpAtomicUMax", 0), "smin": ("OpAtomicSMin", 0x7FFFFFFF), "smax": ("OpAtomicSMax", 0x80000000),
                   "and": ("OpAtomicAnd", 0xFFFFFFFF), "or": ("OpAtomicOr", 0), "xor": ("OpAtomicXor", 0),
                   "swap": ("OpAtomicExchange", None), "cmpswap": ("OpAtomicCompareExchange", None)}

    def buffer_atomic(self, op, a, mods):
        """buffer_atomic_<op> vdata, vaddr, s[V#], soffset ...: 32-bit atomic at the element address (buffer device
        address). With 'glc' the pre-op value is returned in vdata. Inactive lanes don't touch memory."""
        kind = op[len("buffer_atomic_"):]
        if kind not in self.BUF_ATOMICS or "x2" in kind:
            raise Unsupported(op)
        spv, neutral = self.BUF_ATOMICS[kind]
        addr, _ = self.buffer_address(a, mods)
        vd = a[0]
        ptr, p = self.tmp(), self.tmp()
        self.emit(f"{ptr} = OpConvertUToPtr %_ptr_PSB_arr {self.host_addr(addr)}")
        self.emit(f"{p} = OpAccessChain %_ptr_PSB_uint {ptr} {self.cu(0)}")
        val = self.load_reg("vgpr", vd["r"])
        act = self.active_b()
        if neutral is not None:
            safe, old = self.tmp(), self.tmp()
            self.emit(f"{safe} = OpSelect %uint {act} {val} {self.cu(neutral)}")
            self.emit(f"{old} = {spv} %uint {p} {self.cu(1)} {self.cu(0)} {safe}")
        else:
            # swap/cmpswap have no neutral operand: guard with a structured if and merge the result.
            lbl_do, lbl_end, res_do, old = self.tmp() + "_do", self.tmp() + "_end", self.tmp(), self.tmp()
            cur = self.tmp()
            self.emit(f"{cur} = OpLoad %uint {p} Aligned 4")
            self.emit(f"OpSelectionMerge {lbl_end} None")
            self.emit(f"OpBranchConditional {act} {lbl_do} {lbl_end}")
            pre_label = self.current_label_placeholder()
            self.emit(f"{lbl_do} = OpLabel")
            if kind == "cmpswap":
                cmp = self.load_reg("vgpr", vd["r"] + 1)
                self.emit(f"{res_do} = OpAtomicCompareExchange %uint {p} {self.cu(1)} {self.cu(0)} {self.cu(0)} {val} {cmp}")
            else:
                self.emit(f"{res_do} = OpAtomicExchange %uint {p} {self.cu(1)} {self.cu(0)} {val}")
            self.emit(f"OpBranch {lbl_end}")
            self.emit(f"{lbl_end} = OpLabel")
            self.emit(f"{old} = OpPhi %uint {res_do} {lbl_do} {cur} {pre_label}")
        if "glc" in mods:
            self.dst({"k": "vgpr", "r": vd["r"], "n": 1}, old)

    def current_label_placeholder(self):
        """Label of the block currently being emitted (needed for OpPhi). Blocks are opened by OpLabel lines."""
        for line in reversed(self.body):
            s = line.strip()
            if s.endswith("= OpLabel"):
                return s.split(" = ")[0]
        return self.block_label

    def image_get_lod(self, a, mods):
        """image_get_lod vD, vaddr, T#, S#: (clamped lod, unclamped lod) like OpImageQueryLod (2D, 2D array). For
        arrays the query takes only the 2D coordinate; the layer doesn't affect the LOD."""
        dim = next((m for m in mods if m.startswith("dim:")), "dim:SQ_RSRC_IMG_2D")
        if dim not in ("dim:SQ_RSRC_IMG_2D", "dim:SQ_RSRC_IMG_2D_ARRAY") or self.stage != "ps" or "a16" in mods:
            raise Unsupported("image_get_lod form")
        arr = dim.endswith("_ARRAY")
        if arr:
            self.uses_tex_array = True
        dm = next((m for m in mods if m.startswith("dmask:")), "dmask:0x3")
        dmask = int(dm.split(":")[1], 0)
        vdata, addr = a[0], self.addr_regs(a[1])
        img_idx = self.desc_lookup(self.sgpr_words(a[2], 8))
        smp_idx = self.desc_lookup(self.sgpr_words(a[3], 4) + [self.cu(0xFFFFFFFF)] * 4)
        ip, img, sp, smp, si = (self.tmp() for _ in range(5))
        self.emit(f"{ip} = OpAccessChain %{'_ptr_UC_imga' if arr else '_ptr_UC_img'} "
                  f"{'%images_arr' if arr else '%images'} {img_idx}")
        self.emit(f"{img} = OpLoad {'%img2da' if arr else '%img2d'} {ip}")
        self.emit(f"{sp} = OpAccessChain %_ptr_UC_smp %samplers {smp_idx}")
        self.emit(f"{smp} = OpLoad %smp {sp}")
        self.emit(f"{si} = OpSampledImage {'%simg2da' if arr else '%simg2d'} {img} {smp}")
        f = [self.tof(self.load_reg("vgpr", r)) for r in addr[:2]]
        uv, lod = self.tmp(), self.tmp()
        self.emit(f"{uv} = OpCompositeConstruct %v2float {f[0]} {f[1]}")
        self.emit(f"{lod} = OpImageQueryLod %v2float {si} {uv}")
        self.uses_query = True
        out = 0
        for c in range(2):
            if dmask >> c & 1:
                comp = self.tmp()
                self.emit(f"{comp} = OpCompositeExtract %float {lod} {c}")
                self.dst({"k": "vgpr", "r": vdata["r"] + out, "n": 1}, self.tou(comp))
                out += 1

    def buffer_load_format(self, op, a, mods, fixed_fmt=None):
        """buffer_load_format_[x|xy|xyz|xyzw]: element at V#.base + offsets, decoded per the V# FORMAT field
        (word3 bits 12-18), then swizzled by DST_SEL_X/Y/Z/W (word3 bits 0-11: 0=0, 1=1, 4..7 = X..W).
        Formats outside FORMATS decode to 0. num_records bounds checking isn't modelled."""
        if any(m in mods for m in ("lds", "tfe", "d16")):
            raise Unsupported(op + " form")
        ncomp = len(op.rsplit("_", 1)[1])
        vd, vaddr, rsrc = a[0], a[1], a[2]
        soff = a[3] if len(a) > 3 else {"k": "imm", "v": 0}
        if rsrc["k"] != "sgpr" or rsrc.get("n") != 4:
            raise Unsupported("buffer_load_format V#")
        d0, d1, d3 = (self.load_reg("sgpr", rsrc["r"] + i) for i in (0, 1, 3))
        hi, stride = self.tmp(), self.tmp()
        self.emit(f"{hi} = OpBitwiseAnd %uint {d1} {self.cu(0xFFFF)}")
        self.emit(f"{stride} = OpBitFieldUExtract %uint {d1} {self.cu(16)} {self.cu(14)}")
        if fixed_fmt is not None:
            fmt = self.cu(fixed_fmt)          # tbuffer: format from the instruction
        else:
            fmt = self.tmp()
            self.emit(f"{fmt} = OpBitFieldUExtract %uint {d3} {self.cu(12)} {self.cu(7)}")
        inst = next((int(m.split(":")[1], 0) for m in mods if m.startswith("offset:")), 0)
        total = self.tmp()
        self.emit(f"{total} = OpIAdd %uint {self.src_u(soff)} {self.cu(inst)}")
        vregs = [] if vaddr["k"] == "special" else [vaddr["r"] + i for i in range(vaddr.get("n", 1))]
        if "idxen" in mods:
            idx = self.load_reg("vgpr", vregs[0])
            m, t = self.tmp(), self.tmp()
            self.emit(f"{m} = OpIMul %uint {idx} {stride}")
            self.emit(f"{t} = OpIAdd %uint {total} {m}")
            total = t
        if "offen" in mods:
            o = self.load_reg("vgpr", vregs[1] if "idxen" in mods else vregs[0])
            t = self.tmp()
            self.emit(f"{t} = OpIAdd %uint {total} {o}")
            total = t
        self.uses_psb = True
        l64, h64, sh, base = self.tmp(), self.tmp(), self.tmp(), self.tmp()
        self.emit(f"{l64} = OpUConvert %ulong {d0}")
        self.emit(f"{h64} = OpUConvert %ulong {hi}")
        self.emit(f"{sh} = OpShiftLeftLogical %ulong {h64} %c64_32")
        self.emit(f"{base} = OpBitwiseOr %ulong {sh} {l64}")
        # Byte address of the element, split into a dword-aligned address and a byte shift (for 1/2-byte formats).
        o64, addr, alo = self.tmp(), self.tmp(), self.tmp()
        self.emit(f"{o64} = OpUConvert %ulong {total}")
        self.emit(f"{addr} = OpIAdd %ulong {base} {o64}")
        self.emit(f"{alo} = OpUConvert %uint {addr}")
        bsh, al64, aligned = self.tmp(), self.tmp(), self.tmp()
        b3 = self.tmp()
        self.emit(f"{b3} = OpBitwiseAnd %uint {alo} {self.cu(3)}")
        self.emit(f"{bsh} = OpIMul %uint {b3} {self.cu(8)}")
        self.emit(f"{al64} = OpUConvert %ulong {b3}")
        self.emit(f"{aligned} = OpISub %ulong {addr} {al64}")
        # How many dwords this element needs, by format (others: 1). Extra loads re-read dword 0, so a load never
        # goes past the element (the last element of a buffer stays in bounds).
        need = self.cu(1)
        for code, (size, _, _) in self.FORMATS.items():
            e, n2 = self.tmp(), self.tmp()
            self.emit(f"{e} = OpIEqual %bool {fmt} {self.cu(code)}")
            self.emit(f"{n2} = OpSelect %uint {e} {self.cu(max(1, (size + 3) // 4))} {need}")
            need = n2
        ptr = self.tmp()
        self.emit(f"{ptr} = OpConvertUToPtr %_ptr_PSB_arr {self.host_addr(aligned)}")
        words = []
        for i in range(4):
            ok, idx, p, v = self.tmp(), self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{ok} = OpULessThan %bool {self.cu(i)} {need}")
            self.emit(f"{idx} = OpSelect %uint {ok} {self.cu(i)} {self.cu(0)}")
            self.emit(f"{p} = OpAccessChain %_ptr_PSB_uint {ptr} {idx}")
            self.emit(f"{v} = OpLoad %uint {p} Aligned 4")
            words.append(v)
        # Sub-dword formats start at the byte shift inside dword 0 (and may spill into dword 1).
        w0s, w1s, nz, w1sh, merged, bw0 = (self.tmp() for _ in range(6))
        self.emit(f"{w0s} = OpShiftRightLogical %uint {words[0]} {bsh}")
        self.emit(f"{w1sh} = OpISub %uint {self.cu(32)} {bsh}")
        self.emit(f"{w1s} = OpShiftLeftLogical %uint {words[1]} {w1sh}")
        self.emit(f"{nz} = OpINotEqual %bool {bsh} {self.cu(0)}")
        self.emit(f"{merged} = OpBitwiseOr %uint {w0s} {w1s}")
        self.emit(f"{bw0} = OpSelect %uint {nz} {merged} {words[0]}")
        words = [bw0] + words[1:]
        # Decode every supported format, then pick by FORMAT.
        comps = [self.cu(0)] * 4
        for code in self.FORMATS:
            dec = self.decode_format(words, code)
            e = self.tmp()
            self.emit(f"{e} = OpIEqual %bool {fmt} {self.cu(code)}")
            nxt = []
            for c in range(4):
                s = self.tmp()
                self.emit(f"{s} = OpSelect %uint {e} {dec[c]} {comps[c]}")
                nxt.append(s)
            comps = nxt
        # DST_SEL swizzle: 0 -> 0, 1 -> 1.0, 4..7 -> X..W.
        for c in range(ncomp):
            sel = self.tmp()
            self.emit(f"{sel} = OpBitFieldUExtract %uint {d3} {self.cu(3 * c)} {self.cu(3)}")
            r = self.cu(0)
            for code_v, src in ((1, self.cu(0x3F800000)), (4, comps[0]), (5, comps[1]), (6, comps[2]), (7, comps[3])):
                e, s = self.tmp(), self.tmp()
                self.emit(f"{e} = OpIEqual %bool {sel} {self.cu(code_v)}")
                self.emit(f"{s} = OpSelect %uint {e} {src} {r}")
                r = s
            self.dst({"k": "vgpr", "r": vd["r"] + c, "n": 1}, r)

    def global_mem(self, op, a, mods):
        """global_load_dword[xN] vD, vaddr, saddr|off offset:N / global_store_dword[xN] vaddr, vD, saddr|off.
        Address = (saddr: 64-bit SGPR pair + zero-extended 32-bit vaddr | off: 64-bit VGPR pair) + offset,
        an Xbox GPU VA translated through the heap base. Stores and loads only take effect in active lanes."""
        load = op.startswith("global_load")
        if any(m in mods for m in ("lds", "tfe", "d16")):
            raise Unsupported(op + " form")
        n = 1 if op.endswith("dword") else int(op.rsplit("x", 1)[1])
        data, vaddr, saddr = (a[0], a[1], a[2]) if load else (a[1], a[0], a[2])
        inst = next((int(m.split(":")[1], 0) for m in mods if m.startswith("offset:")), 0)
        if inst & 0x800:
            inst -= 0x1000            # 12-bit signed offset
        if saddr["k"] == "sgpr" and saddr.get("n") == 2:
            lo, hi = self.load_reg("sgpr", saddr["r"]), self.load_reg("sgpr", saddr["r"] + 1)
            voff = self.load_reg("vgpr", vaddr["r"])
            l64, h64, sh, base, v64, va = (self.tmp() for _ in range(6))
            self.emit(f"{l64} = OpUConvert %ulong {lo}")
            self.emit(f"{h64} = OpUConvert %ulong {hi}")
            self.emit(f"{sh} = OpShiftLeftLogical %ulong {h64} %c64_32")
            self.emit(f"{base} = OpBitwiseOr %ulong {sh} {l64}")
            self.emit(f"{v64} = OpUConvert %ulong {voff}")
            self.emit(f"{va} = OpIAdd %ulong {base} {v64}")
        elif saddr["k"] == "special" and saddr["t"] == "off" and vaddr["k"] == "vgpr" and vaddr.get("n") == 2:
            lo, hi = self.load_reg("vgpr", vaddr["r"]), self.load_reg("vgpr", vaddr["r"] + 1)
            l64, h64, sh, va = (self.tmp() for _ in range(4))
            self.emit(f"{l64} = OpUConvert %ulong {lo}")
            self.emit(f"{h64} = OpUConvert %ulong {hi}")
            self.emit(f"{sh} = OpShiftLeftLogical %ulong {h64} %c64_32")
            self.emit(f"{va} = OpBitwiseOr %ulong {sh} {l64}")
        else:
            raise Unsupported(op + " addressing")
        self.uses_psb = True
        off64, full = self.tmp(), self.tmp()
        ioff = self.tmp()
        self.emit(f"{ioff} = OpBitcast %uint {self.as_int(self.cu(inst & 0xFFFFFFFF))}")
        self.emit(f"{off64} = OpSConvert %long {self.as_int(ioff)}")
        o2 = self.tmp()
        self.emit(f"{o2} = OpBitcast %ulong {off64}")
        self.emit(f"{full} = OpIAdd %ulong {va} {o2}")
        self.uses_long = True
        act = self.active_b()
        lbl_do, lbl_end = self.tmp() + "_do", self.tmp() + "_end"
        if load:
            # Inactive lanes don't load (and keep their registers): guard the access, merge with OpPhi per dword.
            olds = [self.load_reg("vgpr", data["r"] + i) for i in range(n)]
            pre = self.current_label_placeholder()
            self.emit(f"OpSelectionMerge {lbl_end} None")
            self.emit(f"OpBranchConditional {act} {lbl_do} {lbl_end}")
            self.emit(f"{lbl_do} = OpLabel")
            ptr = self.tmp()
            self.emit(f"{ptr} = OpConvertUToPtr %_ptr_PSB_arr {self.host_addr(full)}")
            vals = []
            for i in range(n):
                p, v = self.tmp(), self.tmp()
                self.emit(f"{p} = OpAccessChain %_ptr_PSB_uint {ptr} {self.cu(i)}")
                self.emit(f"{v} = OpLoad %uint {p} Aligned 4")
                vals.append(v)
            self.emit(f"OpBranch {lbl_end}")
            self.emit(f"{lbl_end} = OpLabel")
            phis = []
            for i in range(n):                 # all OpPhi first: they must open the merge block
                ph = self.tmp()
                self.emit(f"{ph} = OpPhi %uint {vals[i]} {lbl_do} {olds[i]} {pre}")
                phis.append(ph)
            for i, ph in enumerate(phis):
                self.store_reg("vgpr", data["r"] + i, ph)
        else:
            vals = [self.load_reg("vgpr", data["r"] + i) for i in range(n)]
            self.emit(f"OpSelectionMerge {lbl_end} None")
            self.emit(f"OpBranchConditional {act} {lbl_do} {lbl_end}")
            self.emit(f"{lbl_do} = OpLabel")
            ptr = self.tmp()
            self.emit(f"{ptr} = OpConvertUToPtr %_ptr_PSB_arr {self.host_addr(full)}")
            for i in range(n):
                p = self.tmp()
                self.emit(f"{p} = OpAccessChain %_ptr_PSB_uint {ptr} {self.cu(i)}")
                self.emit(f"OpStore {p} {vals[i]} Aligned 4")
            self.emit(f"OpBranch {lbl_end}")
            self.emit(f"{lbl_end} = OpLabel")

    def buffer_address(self, a, mods):
        """(64-bit element address id, V# word3 id) for a MUBUF instruction: vdata, vaddr, srsrc, soffset."""
        vaddr, rsrc = a[1], a[2]
        soff = a[3] if len(a) > 3 else {"k": "imm", "v": 0}
        if rsrc["k"] != "sgpr" or rsrc.get("n") != 4:
            raise Unsupported("buffer V#")
        d0, d1, d3 = (self.load_reg("sgpr", rsrc["r"] + i) for i in (0, 1, 3))
        hi, stride = self.tmp(), self.tmp()
        self.emit(f"{hi} = OpBitwiseAnd %uint {d1} {self.cu(0xFFFF)}")
        self.emit(f"{stride} = OpBitFieldUExtract %uint {d1} {self.cu(16)} {self.cu(14)}")
        inst = next((int(m.split(":")[1], 0) for m in mods if m.startswith("offset:")), 0)
        total = self.tmp()
        self.emit(f"{total} = OpIAdd %uint {self.src_u(soff)} {self.cu(inst)}")
        vregs = [] if vaddr["k"] == "special" else [vaddr["r"] + i for i in range(vaddr.get("n", 1))]
        if "idxen" in mods:
            idx = self.load_reg("vgpr", vregs[0])
            m, t = self.tmp(), self.tmp()
            self.emit(f"{m} = OpIMul %uint {idx} {stride}")
            self.emit(f"{t} = OpIAdd %uint {total} {m}")
            total = t
        if "offen" in mods:
            o = self.load_reg("vgpr", vregs[1] if "idxen" in mods else vregs[0])
            t = self.tmp()
            self.emit(f"{t} = OpIAdd %uint {total} {o}")
            total = t
        self.uses_psb = True
        l64, h64, sh, base, o64, addr = (self.tmp() for _ in range(6))
        self.emit(f"{l64} = OpUConvert %ulong {d0}")
        self.emit(f"{h64} = OpUConvert %ulong {hi}")
        self.emit(f"{sh} = OpShiftLeftLogical %ulong {h64} %c64_32")
        self.emit(f"{base} = OpBitwiseOr %ulong {sh} {l64}")
        self.emit(f"{o64} = OpUConvert %ulong {total}")
        self.emit(f"{addr} = OpIAdd %ulong {base} {o64}")
        return addr, d3

    def buffer_store(self, op, a, mods):
        """buffer_store_dword[xN] (raw) and buffer_store_format_* (f32 formats stored raw, f16x2/x4, unorm8x4).
        Only active lanes store, inside a structured if."""
        if any(m in mods for m in ("lds", "tfe", "d16")):
            raise Unsupported(op + " form")
        vd = a[0]
        addr, d3 = self.buffer_address(a, mods)
        if op.startswith("buffer_store_dword"):
            n = 1 if op.endswith("dword") else int(op.rsplit("x", 1)[1])
            words = [self.load_reg("vgpr", vd["r"] + i) for i in range(n)]
            plans = None
        else:
            n = len(op.rsplit("_", 1)[1])
            comps = [self.load_reg("vgpr", vd["r"] + i) for i in range(n)]
            fmt = self.tmp()
            self.emit(f"{fmt} = OpBitFieldUExtract %uint {d3} {self.cu(12)} {self.cu(7)}")
            # Encodings for the formats that appear in store-heavy shaders; others store nothing.
            f32 = {22: 1, 64: 2, 74: 3, 77: 4}
            # Float views of the components, computed before any branch so every branch can use them.
            fl = [self.tof(c) for c in comps] + ["%f0"] * (4 - len(comps))
            plans = (fmt, comps, f32, fl)
            words = None
        act = self.active_b()
        lbl_do, lbl_end = self.tmp() + "_do", self.tmp() + "_end"
        self.emit(f"OpSelectionMerge {lbl_end} None")
        self.emit(f"OpBranchConditional {act} {lbl_do} {lbl_end}")
        self.emit(f"{lbl_do} = OpLabel")
        ptr = self.tmp()
        self.emit(f"{ptr} = OpConvertUToPtr %_ptr_PSB_arr {self.host_addr(addr)}")
        if words is not None:
            for i, w in enumerate(words):
                p = self.tmp()
                self.emit(f"{p} = OpAccessChain %_ptr_PSB_uint {ptr} {self.cu(i)}")
                self.emit(f"OpStore {p} {w} Aligned 4")
        else:
            fmt, comps, f32, fl = plans
            # f32 formats: store min(n, format components) raw dwords; f16x4 (71) packs pairs; unorm8x4 (56) packs.
            for code, cnt in f32.items():
                e = self.tmp()
                self.emit(f"{e} = OpIEqual %bool {fmt} {self.cu(code)}")
                ld, le = self.tmp() + "_f", self.tmp() + "_fe"
                self.emit(f"OpSelectionMerge {le} None")
                self.emit(f"OpBranchConditional {e} {ld} {le}")
                self.emit(f"{ld} = OpLabel")
                for i in range(min(cnt, len(comps))):
                    p = self.tmp()
                    self.emit(f"{p} = OpAccessChain %_ptr_PSB_uint {ptr} {self.cu(i)}")
                    self.emit(f"OpStore {p} {comps[i]} Aligned 4")
                self.emit(f"OpBranch {le}")
                self.emit(f"{le} = OpLabel")
            e = self.tmp()
            self.emit(f"{e} = OpIEqual %bool {fmt} {self.cu(71)}")
            ld, le = self.tmp() + "_h", self.tmp() + "_he"
            self.emit(f"OpSelectionMerge {le} None")
            self.emit(f"OpBranchConditional {e} {ld} {le}")
            self.emit(f"{ld} = OpLabel")
            for i in range(2):
                v, pk, p = self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{v} = OpCompositeConstruct %v2float {fl[2 * i]} {fl[2 * i + 1]}")
                self.emit(f"{pk} = OpExtInst %uint %glsl PackHalf2x16 {v}")
                self.emit(f"{p} = OpAccessChain %_ptr_PSB_uint {ptr} {self.cu(i)}")
                self.emit(f"OpStore {p} {pk} Aligned 4")
            self.emit(f"OpBranch {le}")
            self.emit(f"{le} = OpLabel")
            e = self.tmp()
            self.emit(f"{e} = OpIEqual %bool {fmt} {self.cu(56)}")
            ld, le = self.tmp() + "_u", self.tmp() + "_ue"
            self.emit(f"OpSelectionMerge {le} None")
            self.emit(f"OpBranchConditional {e} {ld} {le}")
            self.emit(f"{ld} = OpLabel")
            v4, pk, p = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{v4} = OpCompositeConstruct %v4float {' '.join(fl[:4])}")
            self.emit(f"{pk} = OpExtInst %uint %glsl PackUnorm4x8 {v4}")
            self.emit(f"{p} = OpAccessChain %_ptr_PSB_uint {ptr} {self.cu(0)}")
            self.emit(f"OpStore {p} {pk} Aligned 4")
            self.emit(f"OpBranch {le}")
            self.emit(f"{le} = OpLabel")
        self.emit(f"OpBranch {lbl_end}")
        self.emit(f"{lbl_end} = OpLabel")

    def cmp_class(self, a):
        """v_cmp_class_f32 dst, x, mask: true if x's class bit is set in mask.
        Bits: 0 sNaN, 1 qNaN, 2 -inf, 3 -normal, 4 -denormal, 5 -0, 6 +0, 7 +denormal, 8 +normal, 9 +inf."""
        x, mask = self.src_f(a[-2]), self.src_u(a[-1])
        u = self.tou(x)
        sign, absu = self.tmp(), self.tmp()
        self.emit(f"{sign} = OpShiftRightLogical %uint {u} {self.cu(31)}")
        self.emit(f"{absu} = OpBitwiseAnd %uint {u} {self.cu(0x7FFFFFFF)}")
        exp_, man = self.tmp(), self.tmp()
        self.emit(f"{exp_} = OpShiftRightLogical %uint {absu} {self.cu(23)}")
        self.emit(f"{man} = OpBitwiseAnd %uint {absu} {self.cu(0x7FFFFF)}")
        neg = self.tmp_bool_nz(sign)

        def b(cond_line):
            t = self.tmp()
            self.emit(f"{t} = {cond_line}")
            return t
        is_zero = b(f"OpIEqual %bool {absu} {self.cu(0)}")
        exp0 = b(f"OpIEqual %bool {exp_} {self.cu(0)}")
        expmax = b(f"OpIEqual %bool {exp_} {self.cu(255)}")
        man0 = b(f"OpIEqual %bool {man} {self.cu(0)}")
        man_nz = b(f"OpLogicalNot %bool {man0}")
        is_inf = b(f"OpLogicalAnd %bool {expmax} {man0}")
        is_nan = b(f"OpLogicalAnd %bool {expmax} {man_nz}")
        quiet = b(f"OpUGreaterThanEqual %bool {man} {self.cu(0x400000)}")
        snan = b(f"OpLogicalAnd %bool {is_nan} " + b(f"OpLogicalNot %bool {quiet}"))
        qnan = b(f"OpLogicalAnd %bool {is_nan} {quiet}")
        denorm = b(f"OpLogicalAnd %bool {exp0} {man_nz}")
        normal = b(f"OpLogicalAnd %bool " + b(f"OpLogicalNot %bool {exp0}") + " " + b(f"OpLogicalNot %bool {expmax}"))
        pos = b(f"OpLogicalNot %bool {neg}")
        classes = [snan, qnan, b(f"OpLogicalAnd %bool {is_inf} {neg}"), b(f"OpLogicalAnd %bool {normal} {neg}"),
                   b(f"OpLogicalAnd %bool {denorm} {neg}"), b(f"OpLogicalAnd %bool {is_zero} {neg}"),
                   b(f"OpLogicalAnd %bool {is_zero} {pos}"), b(f"OpLogicalAnd %bool {denorm} {pos}"),
                   b(f"OpLogicalAnd %bool {normal} {pos}"), b(f"OpLogicalAnd %bool {is_inf} {pos}")]
        acc = "%false"
        for i, c in enumerate(classes):
            bit = b(f"OpBitwiseAnd %uint {mask} {self.cu(1 << i)}")
            on = b(f"OpINotEqual %bool {bit} {self.cu(0)}")
            hit = b(f"OpLogicalAnd %bool {on} {c}")
            acc = b(f"OpLogicalOr %bool {acc} {hit}")
        acc = b(f"OpLogicalAnd %bool {acc} {self.active_b()}")   # inactive lanes write 0
        r = b(f"OpSelect %uint {acc} {self.cu(0xFFFFFFFF)} {self.cu(0)}")
        d = a[0]
        if len(a) == 2 or (d["k"] == "special" and d["t"] in ("vcc", "vcc_lo")):
            self.emit(f"OpStore %vcc {r}")
        else:
            self.mask_dst(d, r)

    @staticmethod
    def mod_int(mods, name, default):
        m = next((m for m in mods if m.startswith(name + ":")), None)
        return int(m.split(":", 1)[1], 0) if m else default

    # LDS atomics (non-returning name -> SPIR-V atomic); _rtn_ forms share the entry.
    LDS_ATOMICS = {"ds_add_u32": "OpAtomicIAdd", "ds_sub_u32": "OpAtomicISub", "ds_min_u32": "OpAtomicUMin",
                   "ds_max_u32": "OpAtomicUMax", "ds_min_i32": "OpAtomicSMin", "ds_max_i32": "OpAtomicSMax",
                   "ds_and_b32": "OpAtomicAnd", "ds_or_b32": "OpAtomicOr", "ds_xor_b32": "OpAtomicXor",
                   "ds_wrxchg_rtn_b32": "OpAtomicExchange"}

    def lds_ptr(self, addr, byte_off):
        """Pointer to the LDS dword at (addr + byte_off) / 4, wrapped to the array (LDS is a Workgroup uint array)."""
        s1, i, m, p = self.tmp(), self.tmp(), self.tmp(), self.tmp()
        self.emit(f"{s1} = OpIAdd %uint {addr} {self.cu(byte_off)}")
        self.emit(f"{i} = OpShiftRightLogical %uint {s1} {self.cu(2)}")
        self.emit(f"{m} = OpBitwiseAnd %uint {i} {self.cu(self.lds_dwords - 1)}")
        self.emit(f"{p} = OpAccessChain %_ptr_WG_uint %lds {m}")
        return p

    def lds_op(self, op, a, mods):
        """LDS (GFX10 DS encoding). Addresses are bytes. offset:N is bytes; read2/write2 offset0/offset1 count
        data-size units (x64 for st64). Writes and atomics run only in active lanes; results land in VGPRs through
        dst (inactive lanes unchanged). Sub-dword writes use atomic and/or so neighbouring lanes don't race."""
        if self.stage != "cs":
            raise Unsupported(op + " outside compute")
        self.uses_lds = True
        rtn = "_rtn_" in op
        name = op if op == "ds_wrxchg_rtn_b32" else op.replace("_rtn_", "_")
        act = self.active_b()
        lbl_do, lbl_end = self.tmp() + "_do", self.tmp() + "_end"
        self.emit(f"OpSelectionMerge {lbl_end} None")
        self.emit(f"OpBranchConditional {act} {lbl_do} {lbl_end}")
        self.emit(f"{lbl_do} = OpLabel")
        off = self.mod_int(mods, "offset", 0)
        sc, sem = self.cu(2), self.cu(0)
        sizes = {"b32": 1, "b64": 2, "b96": 3, "b128": 4}
        if op in ("ds_read_b32", "ds_read_b64", "ds_read_b96", "ds_read_b128"):
            addr = self.src_u(a[1])
            for k in range(sizes[op.rsplit("_", 1)[1]]):
                v = self.tmp()
                self.emit(f"{v} = OpLoad %uint {self.lds_ptr(addr, off + 4 * k)}")
                self.dst({"k": "vgpr", "r": a[0]["r"] + k, "n": 1}, v)
        elif op in ("ds_write_b32", "ds_write_b64", "ds_write_b96", "ds_write_b128"):
            addr = self.src_u(a[0])
            for k in range(sizes[op.rsplit("_", 1)[1]]):
                v = self.load_reg("vgpr", a[1]["r"] + k)
                self.emit(f"OpStore {self.lds_ptr(addr, off + 4 * k)} {v}")
        elif op in ("ds_read2_b32", "ds_read2st64_b32", "ds_write2_b32", "ds_write2st64_b32"):
            unit = 4 * (64 if "st64" in op else 1)
            offs = (self.mod_int(mods, "offset0", 0) * unit, self.mod_int(mods, "offset1", 0) * unit)
            if op.startswith("ds_read2"):
                addr = self.src_u(a[1])
                for k, o in enumerate(offs):
                    v = self.tmp()
                    self.emit(f"{v} = OpLoad %uint {self.lds_ptr(addr, o)}")
                    self.dst({"k": "vgpr", "r": a[0]["r"] + k, "n": 1}, v)
            else:
                addr = self.src_u(a[0])
                for k, o in enumerate(offs):
                    self.emit(f"OpStore {self.lds_ptr(addr, o)} {self.src_u(a[1 + k])}")
        elif op in ("ds_read_u16", "ds_read_i16", "ds_read_u8", "ds_read_i8", "ds_read_u16_d16_hi", "ds_read_u16_d16"):
            bits = 16 if "16" in op.split("_")[2] else 8
            addr = self.src_u(a[1])
            s1, sh, sh2, w, v = (self.tmp() for _ in range(5))
            self.emit(f"{s1} = OpIAdd %uint {addr} {self.cu(off)}")
            self.emit(f"{sh} = OpShiftLeftLogical %uint {s1} {self.cu(3)}")     # byte address -> bit offset
            self.emit(f"{sh2} = OpBitwiseAnd %uint {sh} {self.cu(31)}")
            self.emit(f"{w} = OpLoad %uint {self.lds_ptr(addr, off)}")
            f = "OpBitFieldSExtract" if op.split("_")[2].startswith("i") else "OpBitFieldUExtract"
            self.emit(f"{v} = {f} %uint {w} {sh2} {self.cu(bits)}")
            d = a[0]
            if "_d16" in op:
                hi = op.endswith("_hi")
                r = self.tmp()
                self.emit(f"{r} = OpBitFieldInsert %uint {self.load_reg('vgpr', d['r'])} {v} "
                          f"{self.cu(16 if hi else 0)} {self.cu(16)}")
                v = r
            self.dst(d, v)
        elif op in ("ds_write_b16", "ds_write_b8", "ds_write_b16_d16_hi", "ds_write_b8_d16_hi"):
            bits = 16 if "b16" in op else 8
            addr = self.src_u(a[0])
            val = self.src_u(a[1])
            if op.endswith("_d16_hi"):
                hv = self.tmp()
                self.emit(f"{hv} = OpShiftRightLogical %uint {val} {self.cu(16)}")
                val = hv
            s1, sh, sh2, fld, pos, msk, inv = (self.tmp() for _ in range(7))
            self.emit(f"{s1} = OpIAdd %uint {addr} {self.cu(off)}")
            self.emit(f"{sh} = OpShiftLeftLogical %uint {s1} {self.cu(3)}")
            self.emit(f"{sh2} = OpBitwiseAnd %uint {sh} {self.cu(31)}")
            self.emit(f"{fld} = OpBitwiseAnd %uint {val} {self.cu((1 << bits) - 1)}")
            self.emit(f"{pos} = OpShiftLeftLogical %uint {fld} {sh2}")
            self.emit(f"{msk} = OpShiftLeftLogical %uint {self.cu((1 << bits) - 1)} {sh2}")
            self.emit(f"{inv} = OpNot %uint {msk}")
            p = self.lds_ptr(addr, off)
            t1, t2 = self.tmp(), self.tmp()
            self.emit(f"{t1} = OpAtomicAnd %uint {p} {sc} {sem} {inv}")
            self.emit(f"{t2} = OpAtomicOr %uint {p} {sc} {sem} {pos}")
        elif name in self.LDS_ATOMICS:
            d, ad, dat = (a[0], a[1], a[2]) if rtn else (None, a[0], a[1])
            p = self.lds_ptr(self.src_u(ad), off)
            old = self.tmp()
            self.emit(f"{old} = {self.LDS_ATOMICS[name]} %uint {p} {sc} {sem} {self.src_u(dat)}")
            if d is not None:
                self.dst(d, old)
        elif op == "ds_cmpst_rtn_b32":
            # tmp = M[A]; M[A] = (tmp == DATA) ? DATA2 : tmp; return tmp
            d, ad, cmp_, src = a[0], a[1], a[2], a[3]
            p = self.lds_ptr(self.src_u(ad), off)
            old = self.tmp()
            self.emit(f"{old} = OpAtomicCompareExchange %uint {p} {sc} {sem} {sem} {self.src_u(src)} {self.src_u(cmp_)}")
            self.dst(d, old)
        else:
            raise Unsupported(op)
        self.emit(f"OpBranch {lbl_end}")
        self.emit(f"{lbl_end} = OpLabel")

    def tmp_bool_nz(self, v):
        b = self.tmp()
        self.emit(f"{b} = OpINotEqual %bool {v} {self.cu(0)}")
        return b

    def mask_src(self, a):
        """A lane mask operand (sgpr pair, vcc, exec, imm) as one uint for this lane."""
        if a["k"] == "sgpr":
            return self.load_reg("sgpr", a["r"])
        if a["k"] == "special" and a["t"] in ("vcc", "vcc_lo"):
            v = self.tmp()
            self.emit(f"{v} = OpLoad %uint %vcc")
            return v
        if a["k"] == "special" and a["t"] in ("exec", "exec_lo"):
            return self.exec_value()
        if a["k"] == "imm":
            return self.cu(0xFFFFFFFF if a["v"] & 1 else 0)
        raise Unsupported("mask operand")

    def mask_dst(self, d, v):
        if d["k"] == "special" and d["t"] in ("exec", "exec_lo"):
            self.set_exec(v)
        elif d["k"] == "sgpr":
            self.store_reg("sgpr", d["r"], v)
            if d.get("n") == 2:
                self.store_reg("sgpr", d["r"] + 1, v)
        elif d["k"] == "special" and d["t"] in ("vcc", "vcc_lo"):
            self.emit(f"OpStore %vcc {v}")
        else:
            raise Unsupported("mask destination")

    def clamp_mods(self, v, mods):
        # Output modifier (mul:2 / mul:4 / div:2) is applied before clamp, as on hardware.
        for m in mods:
            if m in ("mul:2", "mul:4", "div:2"):
                k = {"mul:2": "2", "mul:4": "4", "div:2": "0.5"}[m]
                c = self.fconst(float(k))
                g = self.tmp()
                self.emit(f"{g} = OpFMul %float {v} {c}")
                v = g
        if "clamp" in mods:
            g = self.tmp()
            self.emit(f"{g} = OpExtInst %float %glsl FClamp {v} %f0 %f1")
            return g
        return v

    def fconst(self, x):
        bits = struct.unpack("<I", struct.pack("<f", x))[0]
        return self.tof(self.cu(bits))

    # ---- instructions ----
    SDWA_SEL = {"BYTE_0": (0, 8), "BYTE_1": (8, 8), "BYTE_2": (16, 8), "BYTE_3": (24, 8),
                "WORD_0": (0, 16), "WORD_1": (16, 16), "DWORD": (0, 32)}

    def insn(self, ins):
        """SDWA (sub-dword addressing): each source is narrowed to src<i>_sel (zero-extended, or sign-extended
        under sext()) before the op and its neg/abs, and a VGPR result is placed at dst_sel with dst_unused
        deciding the other bits (PAD = zero, SEXT = sign-extend, PRESERVE = keep the old bits). VOPC has no
        dst_sel; v_cmpx (GFX10) has no written destination operand."""
        if not ins["op"].endswith("_sdwa"):
            return self._insn(ins)
        op, mods = base_op(ins["op"]), ins["mods"]
        kv = dict(m.split(":", 1) for m in mods if m.split(":", 1)[0] in ("src0_sel", "src1_sel", "dst_sel", "dst_unused"))
        args = [dict(x) for x in ins["args"]]
        first = 0 if op.startswith("v_cmpx") else 1
        for i, key in enumerate(("src0_sel", "src1_sel")):
            j = first + i
            if j < len(args) and (kv.get(key, "DWORD") != "DWORD" or args[j].get("sext")):
                args[j]["sdwa_sel"] = kv.get(key, "DWORD")
        if not op.startswith("v_cmp") and args and (kv.get("dst_sel", "DWORD") != "DWORD"):
            args[0]["sdwa_dst"] = (kv["dst_sel"], kv.get("dst_unused", "UNUSED_PAD"))
        return self._insn({**ins, "args": args})

    def sdwa_src(self, v, a):
        off, width = self.SDWA_SEL[a["sdwa_sel"]]
        if width == 32:
            return v
        r = self.tmp()
        f = "OpBitFieldSExtract" if a.get("sext") else "OpBitFieldUExtract"
        self.emit(f"{r} = {f} %uint {v} {self.cu(off)} {self.cu(width)}")
        return r

    def sdwa_dst(self, d, val):
        sel, unused = d["sdwa_dst"]
        off, width = self.SDWA_SEL[sel]
        if width == 32:
            return val
        r = self.tmp()
        if unused == "UNUSED_PRESERVE":
            self.emit(f"{r} = OpBitFieldInsert %uint {self.load_reg('vgpr', d['r'])} {val} {self.cu(off)} {self.cu(width)}")
            return r
        lo = self.tmp()
        ext = "OpBitFieldSExtract" if unused == "UNUSED_SEXT" else "OpBitFieldUExtract"
        self.emit(f"{lo} = {ext} %uint {val} {self.cu(0)} {self.cu(width)}")
        # SEXT: the sign-extended field shifted up leaves sign copies above it; bits below stay zero.
        self.emit(f"{r} = OpShiftLeftLogical %uint {lo} {self.cu(off)}")
        return r

    def _insn(self, ins):
        op, a, mods = base_op(ins["op"]), ins["args"], ins["mods"]
        if op in NOOPS or op.startswith("s_waitcnt"):
            return
        if op == "s_sendmsg" and a and "MSG_ORDERED_PS_DONE" in a[0].get("t", ""):
            # End of the POPS ordered section. The fragment-shader interlock already ends at shader exit (a longer
            # critical section, same ordering), so nothing to do here.
            return
        if op in ("s_setreg_imm32_b32", "s_setreg_b32", "s_denorm_mode", "s_round_mode"):
            # MODE register (rounding/denormals). SPIR-V float controls aren't modelled yet: Vulkan defaults are
            # used, which can differ in denormal handling. Recorded here so the gap is visible.
            return
        if op == "v_mov_b32" or op == "s_mov_b32":
            self.dst(a[0], self.src_u(a[1]))
        elif op == "s_mov_b64":
            self.mov64(a[0], a[1])
        elif op in ("v_max3_f32", "v_min3_f32"):
            x, y, z = (self.src_f(t) for t in a[1:4])
            f = "FMax" if "max" in op else "FMin"
            m, r = self.tmp(), self.tmp()
            self.emit(f"{m} = OpExtInst %float %glsl {f} {x} {y}")
            self.emit(f"{r} = OpExtInst %float %glsl {f} {m} {z}")
            self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
        elif op == "v_and_or_b32":
            x, y, z = (self.src_u(t) for t in a[1:4])
            m, r = self.tmp(), self.tmp()
            self.emit(f"{m} = OpBitwiseAnd %uint {x} {y}")
            self.emit(f"{r} = OpBitwiseOr %uint {m} {z}")
            self.dst(a[0], r)
        elif op == "s_brev_b32":
            x = self.src_u(a[1])
            r = self.tmp()
            self.emit(f"{r} = OpBitReverse %uint {x}")
            self.dst(a[0], r)
        elif op == "v_cvt_f16_f32":
            # Low 16 bits get the f16; on GFX10 the high half is kept (partial write: Mesa ACO instr_is_16bit lists
            # v_cvt_f16_f32 as 16-bit for gfx_level >= GFX10).
            x = self.src_f(a[1])
            v, p, r = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{v} = OpCompositeConstruct %v2float {x} %f0")
            self.emit(f"{p} = OpExtInst %uint %glsl PackHalf2x16 {v}")
            self.emit(f"{r} = OpBitwiseAnd %uint {p} {self.cu(0xFFFF)}")
            self.write_lo16(a[0], r)
        elif op == "s_mulk_i32":
            v = a[1]["v"] & 0xFFFF
            k = v - 0x10000 if v & 0x8000 else v
            x = self.load_reg("sgpr", a[0]["r"])
            r = self.tmp()
            self.emit(f"{r} = OpIMul %uint {x} {self.cu(k)}")
            self.dst(a[0], r)
        elif op == "v_readfirstlane_b32":
            self.dst(a[0], self.src_u(a[1]))      # one lane per invocation: the first lane is this lane
        elif op in ("image_sample_o", "image_sample_lz_o", "image_sample_l_o", "image_sample_b_o",
                    "image_sample_c_lz_o") or                 op in ("image_sample", "image_sample_lz", "image_sample_l", "image_sample_b", "image_sample_c_lz",
                    "image_sample_d", "image_sample_d_cl", "image_load", "image_gather4_lz"):
            self.image_op(op, a, mods)
        elif op in ("v_sin_f32", "v_cos_f32"):
            # RDNA takes the angle in turns: sin(x * 2*pi).
            x = self.src_f(a[1])
            t, r = self.tmp(), self.tmp()
            self.emit(f"{t} = OpFMul %float {x} {self.fconst(6.283185307179586)}")
            self.emit(f"{r} = OpExtInst %float %glsl {'Sin' if 'sin' in op else 'Cos'} {t}")
            self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
        elif op == "v_rndne_f32":
            x = self.src_f(a[1])
            r = self.tmp()
            self.emit(f"{r} = OpExtInst %float %glsl RoundEven {x}")
            self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
        elif op in ("v_madak_f32", "v_madmk_f32"):
            if op == "v_madak_f32":
                x, y, z = self.src_f(a[1]), self.src_f(a[2]), self.src_f(a[3])
            else:
                x, y, z = self.src_f(a[1]), self.src_f(a[3]), self.src_f(a[2])
            m, r = self.tmp(), self.tmp()
            self.emit(f"{m} = OpFMul %float {x} {y}")
            self.emit(f"{r} = OpFAdd %float {m} {z}")
            self.dst(a[0], self.tou(r))
        elif op in ("v_bfi_b32", "v_lshl_or_b32", "v_lshl_add_u32", "v_add_lshl_u32", "v_or3_b32", "v_xor3_b32",
                    "v_add3_u32", "v_and_or_b32_"):
            x, y, z = (self.src_u(t) for t in a[1:4])
            r = self.tmp()
            if op == "v_bfi_b32":
                p, n, q = self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{p} = OpBitwiseAnd %uint {x} {y}")
                self.emit(f"{n} = OpNot %uint {x}")
                self.emit(f"{q} = OpBitwiseAnd %uint {n} {z}")
                self.emit(f"{r} = OpBitwiseOr %uint {p} {q}")
            elif op in ("v_lshl_or_b32", "v_lshl_add_u32"):
                s, m = self.tmp(), self.tmp()
                self.emit(f"{m} = OpBitwiseAnd %uint {y} {self.cu(31)}")
                self.emit(f"{s} = OpShiftLeftLogical %uint {x} {m}")
                self.emit(f"{r} = {'OpBitwiseOr' if op == 'v_lshl_or_b32' else 'OpIAdd'} %uint {s} {z}")
            elif op == "v_add_lshl_u32":
                s, m = self.tmp(), self.tmp()
                self.emit(f"{s} = OpIAdd %uint {x} {y}")
                self.emit(f"{m} = OpBitwiseAnd %uint {z} {self.cu(31)}")
                self.emit(f"{r} = OpShiftLeftLogical %uint {s} {m}")
            else:
                k = {"v_or3_b32": "OpBitwiseOr", "v_xor3_b32": "OpBitwiseXor", "v_add3_u32": "OpIAdd"}[op]
                s = self.tmp()
                self.emit(f"{s} = {k} %uint {x} {y}")
                self.emit(f"{r} = {k} %uint {s} {z}")
            self.dst(a[0], r)
        elif op == "v_pack_b32_f16":
            # dst = f16(src0) | f16(src1) << 16, each taken from the low half (op_sel picks high halves).
            sel = next((m for m in mods if m.startswith("op_sel:")), "op_sel:[0,0,0]")
            sb = [int(x) for x in sel.split("[")[1].rstrip("]").split(",")]
            halves = []
            for i, src in enumerate(a[1:3]):
                u = self.src_u({**src, "neg": False, "abs": False})
                if sb[i]:
                    s = self.tmp()
                    self.emit(f"{s} = OpShiftRightLogical %uint {u} {self.cu(16)}")
                    u = s
                m = self.tmp()
                self.emit(f"{m} = OpBitwiseAnd %uint {u} {self.cu(0xFFFF)}")
                halves.append(m)
            if any(src.get("neg") or src.get("abs") for src in a[1:3]):
                raise Unsupported("v_pack_b32_f16 modifiers")
            hi, r = self.tmp(), self.tmp()
            self.emit(f"{hi} = OpShiftLeftLogical %uint {halves[1]} {self.cu(16)}")
            self.emit(f"{r} = OpBitwiseOr %uint {halves[0]} {hi}")
            self.dst(a[0], r)
        elif op == "v_cmp_class_f32":
            self.cmp_class(a)
        elif op in ("v_ffbl_b32", "v_ffbh_u32", "s_ff1_i32_b32", "s_flbit_i32_b32"):
            x = self.src_u(a[1])
            r = self.tmp()
            if op in ("v_ffbl_b32", "s_ff1_i32_b32"):
                # Index of lowest set bit, -1 (0xFFFFFFFF) if none: FindILsb has the same convention.
                ri = self.tmp()
                self.emit(f"{ri} = OpExtInst %int %glsl FindILsb {self.as_int(x)}")
                self.emit(f"{r} = OpBitcast %uint {ri}")
            else:
                # Leading-zero count from the MSB side, -1 if zero: 31 - FindUMsb(x), with FindUMsb(0) = -1.
                m, z, d = self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{m} = OpExtInst %int %glsl FindUMsb {x}")
                mu = self.tmp()
                self.emit(f"{mu} = OpBitcast %uint {m}")
                self.emit(f"{d} = OpISub %uint {self.cu(31)} {mu}")
                self.emit(f"{z} = OpIEqual %bool {x} {self.cu(0)}")
                self.emit(f"{r} = OpSelect %uint {z} {self.cu(0xFFFFFFFF)} {d}")
            self.dst(a[0], r)
        elif op in ("image_get_resinfo", "image_load_mip", "image_store") or op in self.IMG_ATOMICS:
            self.image_misc(op, a, mods)
        elif op in ("v_cubema_f32", "v_cubeid_f32", "v_cubesc_f32", "v_cubetc_f32"):
            # Cube face selection (standard table): major axis Z if |z|>=|x| and |z|>=|y|, else Y if |y|>=|x|,
            # else X.   face: +X0 -X1 +Y2 -Y3 +Z4 -Z5
            #   +X: sc=-z tc=-y   -X: sc=+z tc=-y   +Y: sc=+x tc=+z   -Y: sc=+x tc=-z   +Z: sc=+x tc=-y   -Z: sc=-x tc=-y
            #   cubema = 2 * (signed major-axis coordinate)
            x, y, z = (self.src_f(t) for t in a[1:4])
            ax, ay, az = (self.tmp() for _ in range(3))
            for t, s in ((ax, x), (ay, y), (az, z)):
                self.emit(f"{t} = OpExtInst %float %glsl FAbs {s}")
            zx, zy, zmaj, yx, ymaj = (self.tmp() for _ in range(5))
            self.emit(f"{zx} = OpFOrdGreaterThanEqual %bool {az} {ax}")
            self.emit(f"{zy} = OpFOrdGreaterThanEqual %bool {az} {ay}")
            self.emit(f"{zmaj} = OpLogicalAnd %bool {zx} {zy}")
            self.emit(f"{yx} = OpFOrdGreaterThanEqual %bool {ay} {ax}")
            nz = self.tmp()
            self.emit(f"{nz} = OpLogicalNot %bool {zmaj}")
            self.emit(f"{ymaj} = OpLogicalAnd %bool {nz} {yx}")

            def neg(v):
                n = self.tmp()
                self.emit(f"{n} = OpFNegate %float {v}")
                return n

            def lt0(v):
                b = self.tmp()
                self.emit(f"{b} = OpFOrdLessThan %bool {v} %f0")
                return b

            def pick(zv, yv, xv):
                t1, t2 = self.tmp(), self.tmp()
                self.emit(f"{t1} = OpSelect %float {ymaj} {yv} {xv}")
                self.emit(f"{t2} = OpSelect %float {zmaj} {zv} {t1}")
                return t2

            def sel(c, t, f):
                r = self.tmp()
                self.emit(f"{r} = OpSelect %float {c} {t} {f}")
                return r
            xn, yn, zn = lt0(x), lt0(y), lt0(z)
            if op == "v_cubema_f32":
                m = pick(z, y, x)
                r = self.tmp()
                self.emit(f"{r} = OpFMul %float {m} {self.fconst(2.0)}")
            elif op == "v_cubeid_f32":
                r = pick(sel(zn, self.fconst(5.0), self.fconst(4.0)), sel(yn, self.fconst(3.0), self.fconst(2.0)),
                         sel(xn, self.fconst(1.0), self.fconst(0.0)))
            elif op == "v_cubesc_f32":
                r = pick(sel(zn, neg(x), x), x, sel(xn, z, neg(z)))
            else:
                r = pick(neg(y), sel(yn, neg(z), z), neg(y))
            self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
        elif op in ("v_mbcnt_lo_u32_b32", "v_mbcnt_hi_u32_b32"):
            # Count of set bits in the mask for lanes below this one (lo: lanes 0-31, hi: lanes 32-63), plus src1.
            self.uses_subgroup = True
            mask, acc = self.src_u(a[1]), self.src_u(a[2])
            lane, rel, below, inrange, one, m1, cnt, r = (self.tmp() for _ in range(8))
            self.emit(f"{lane} = OpLoad %uint %sg_lane")
            if op.endswith("lo_u32_b32"):
                self.emit(f"{rel} = OpCopyObject %uint {lane}")
            else:
                self.emit(f"{rel} = OpISub %uint {lane} {self.cu(32)}")
            # bits below `rel`, clamped to [0, 32]
            self.emit(f"{inrange} = OpULessThan %bool {rel} {self.cu(32)}")
            self.emit(f"{one} = OpShiftLeftLogical %uint {self.cu(1)} {rel}")
            self.emit(f"{m1} = OpISub %uint {one} {self.cu(1)}")
            big = self.tmp()
            # lo: lanes >= 32 count all 32 bits; hi: lanes < 32 count none (rel wraps above 32).
            self.emit(f"{big} = OpSelect %uint {inrange} {m1} "
                      f"{self.cu(0xFFFFFFFF) if op.endswith('lo_u32_b32') else self.cu(0)}")
            if op.endswith("hi_u32_b32"):
                lt32, fixed = self.tmp(), self.tmp()
                self.emit(f"{lt32} = OpULessThan %bool {lane} {self.cu(32)}")
                self.emit(f"{fixed} = OpSelect %uint {lt32} {self.cu(0)} {big}")
                big = fixed
            self.emit(f"{below} = OpBitwiseAnd %uint {mask} {big}")
            self.emit(f"{cnt} = OpBitCount %uint {below}")
            self.emit(f"{r} = OpIAdd %uint {cnt} {acc}")
            self.dst(a[0], r)
        elif op == "v_bfrev_b32":
            x = self.src_u(a[1])
            r = self.tmp()
            self.emit(f"{r} = OpBitReverse %uint {x}")
            self.dst(a[0], r)
        elif op == "v_bfm_b32":
            w, o = self.src_u(a[1]), self.src_u(a[2])
            wm, om, one, m1, r = (self.tmp() for _ in range(5))
            self.emit(f"{wm} = OpBitwiseAnd %uint {w} {self.cu(31)}")
            self.emit(f"{om} = OpBitwiseAnd %uint {o} {self.cu(31)}")
            self.emit(f"{one} = OpShiftLeftLogical %uint {self.cu(1)} {wm}")
            self.emit(f"{m1} = OpISub %uint {one} {self.cu(1)}")
            self.emit(f"{r} = OpShiftLeftLogical %uint {m1} {om}")
            self.dst(a[0], r)
        elif op in ("s_bcnt1_i32_b64", "s_bcnt1_i32_b32"):
            if op.endswith("b64") and a[1]["k"] == "sgpr" and a[1].get("n") == 2:
                lo, hi = self.load_reg("sgpr", a[1]["r"]), self.load_reg("sgpr", a[1]["r"] + 1)
            else:
                lo, hi = (self.mask_src(a[1]), self.cu(0)) if op.endswith("b64") else (self.src_u(a[1]), self.cu(0))
            c1, c2, r = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{c1} = OpBitCount %uint {lo}")
            self.emit(f"{c2} = OpBitCount %uint {hi}")
            self.emit(f"{r} = OpIAdd %uint {c1} {c2}")
            self.dst(a[0], r)
            z = self.tmp()
            self.emit(f"{z} = OpSelect %uint {self.tmp_bool_nz(r)} {self.cu(1)} {self.cu(0)}")
            self.emit(f"OpStore %scc {z}")
        elif op == "v_subrev_nc_u32":
            x, y = self.src_u(a[1]), self.src_u(a[2])
            r = self.tmp()
            self.emit(f"{r} = OpISub %uint {y} {x}")
            self.dst(a[0], r)
        elif op == "s_ff1_i32_b64":
            # Lowest set bit of a 64-bit mask, -1 if none.
            if a[1]["k"] != "sgpr" or a[1].get("n") != 2:
                lo, hi = self.mask_src(a[1]), self.cu(0)
            else:
                lo, hi = self.load_reg("sgpr", a[1]["r"]), self.load_reg("sgpr", a[1]["r"] + 1)
            l, h, hp, lz, r = (self.tmp() for _ in range(5))
            self.emit(f"{l} = OpExtInst %int %glsl FindILsb {self.as_int(lo)}")
            self.emit(f"{h} = OpExtInst %int %glsl FindILsb {self.as_int(hi)}")
            hu = self.tmp()
            self.emit(f"{hu} = OpBitcast %uint {h}")
            hz = self.tmp()
            self.emit(f"{hz} = OpIEqual %bool {hi} {self.cu(0)}")
            self.emit(f"{hp} = OpIAdd %uint {hu} {self.cu(32)}")
            hsel = self.tmp()
            self.emit(f"{hsel} = OpSelect %uint {hz} {self.cu(0xFFFFFFFF)} {hp}")
            self.emit(f"{lz} = OpIEqual %bool {lo} {self.cu(0)}")
            lu = self.tmp()
            self.emit(f"{lu} = OpBitcast %uint {l}")
            self.emit(f"{r} = OpSelect %uint {lz} {hsel} {lu}")
            self.dst(a[0], r)
        elif op == "ds_swizzle_b32":
            self.swizzle(a, mods, ins)
        elif op == "ds_bpermute_b32":
            # vD = vData from lane ((vAddr + offset) / 4) mod 32: a subgroup shuffle (no LDS memory involved).
            d, ad, dat = a[0], self.src_u(a[1]), self.src_u(a[2])
            off = self.mod_int(mods, "offset", 0)
            s1, ln, ln2, r = self.tmp(), self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{s1} = OpIAdd %uint {ad} {self.cu(off)}")
            self.emit(f"{ln} = OpShiftRightLogical %uint {s1} {self.cu(2)}")
            self.emit(f"{ln2} = OpBitwiseAnd %uint {ln} {self.cu(31)}")
            self.emit(f"{r} = OpGroupNonUniformShuffle %uint %c3 {dat} {ln2}")
            self.uses_subgroup = True
            self.dst(d, r)
        elif op.startswith("ds_"):
            self.lds_op(op, a, mods)
        elif op == "s_barrier":
            # Workgroup execution barrier; AMD orders LDS through s_waitcnt before it, so include workgroup memory.
            if self.stage != "cs":
                raise Unsupported("s_barrier outside compute")
            self.emit(f"OpControlBarrier {self.cu(2)} {self.cu(2)} {self.cu(0x108)}")
        elif op == "v_readlane_b32":
            # sD = vS[lane]: a subgroup shuffle from that lane (the result is uniform).
            self.uses_subgroup = True
            v, ln = self.src_u(a[1]), self.src_u(a[2])
            m, r = self.tmp(), self.tmp()
            self.emit(f"{m} = OpBitwiseAnd %uint {ln} {self.cu(63)}")
            self.emit(f"{r} = OpGroupNonUniformShuffle %uint %c3 {v} {m}")
            self.dst(a[0], r)
        elif op == "v_writelane_b32":
            # vD[lane] = sS: only the named lane changes (independent of EXEC, as on hardware).
            self.uses_subgroup = True
            val, ln = self.src_u(a[1]), self.src_u(a[2])
            me, m, eq, old, r = (self.tmp() for _ in range(5))
            self.emit(f"{me} = OpLoad %uint %sg_lane")
            self.emit(f"{m} = OpBitwiseAnd %uint {ln} {self.cu(63)}")
            self.emit(f"{eq} = OpIEqual %bool {me} {m}")
            p = self.tmp()
            self.emit(f"{p} = OpAccessChain %_ptr_Function_uint %vgpr {self.cu(a[0]['r'])}")
            self.emit(f"{old} = OpLoad %uint {p}")
            self.emit(f"{r} = OpSelect %uint {eq} {val} {old}")
            self.emit(f"OpStore {p} {r}")
        elif op.startswith("buffer_store_dword") or op in ("buffer_store_format_x", "buffer_store_format_xy",
                                                            "buffer_store_format_xyz", "buffer_store_format_xyzw"):
            self.buffer_store(op, a, mods)
        elif op in ("v_permlanex16_b32", "v_permlane16_b32"):
            # Lane l reads lane sel[l & 15] of (x16: the other 16-lane half of its row; plain: the same half).
            # The 4-bit selectors for lanes 0-7 are in src1, for lanes 8-15 in src2.
            self.uses_subgroup = True
            v, s1, s2 = self.src_u(a[1]), self.src_u(a[2]), self.src_u(a[3])
            lane, within, hi8, sh, nib1, nib2, nib, row, src = (self.tmp() for _ in range(9))
            self.emit(f"{lane} = OpLoad %uint %sg_lane")
            self.emit(f"{within} = OpBitwiseAnd %uint {lane} {self.cu(15)}")
            self.emit(f"{hi8} = OpUGreaterThanEqual %bool {within} {self.cu(8)}")
            w7 = self.tmp()
            self.emit(f"{w7} = OpBitwiseAnd %uint {within} {self.cu(7)}")
            self.emit(f"{sh} = OpIMul %uint {w7} {self.cu(4)}")
            self.emit(f"{nib1} = OpBitFieldUExtract %uint {s1} {sh} {self.cu(4)}")
            self.emit(f"{nib2} = OpBitFieldUExtract %uint {s2} {sh} {self.cu(4)}")
            self.emit(f"{nib} = OpSelect %uint {hi8} {nib2} {nib1}")
            half = self.tmp()
            self.emit(f"{half} = OpBitwiseAnd %uint {lane} {self.cu(0xFFFFFFF0)}")
            if op == "v_permlanex16_b32":
                self.emit(f"{row} = OpBitwiseXor %uint {half} {self.cu(16)}")
            else:
                self.emit(f"{row} = OpCopyObject %uint {half}")
            self.emit(f"{src} = OpBitwiseOr %uint {row} {nib}")
            r = self.tmp()
            self.emit(f"{r} = OpGroupNonUniformShuffle %uint %c3 {v} {src}")
            self.dst(a[0], r)
        elif op.startswith("buffer_load_dword"):
            self.buffer_load(op, a, mods)
        elif op in ("buffer_load_format_x", "buffer_load_format_xy", "buffer_load_format_xyz",
                    "buffer_load_format_xyzw"):
            self.buffer_load_format(op, a, mods)
        elif op in ("tbuffer_load_format_x", "tbuffer_load_format_xy", "tbuffer_load_format_xyz",
                    "tbuffer_load_format_xyzw"):
            # Typed load with the format in the instruction (format:[BUF_FMT_...]) instead of the V#.
            fm = next((m for m in mods if m.startswith("format:")), None)
            if fm is None:
                raise Unsupported(op + " without format")
            name = "GFX10_FORMAT_" + fm.split("BUF_FMT_")[-1].rstrip("]")
            code = next((c for c, n in GFX10_FORMATS.items() if n == name), None)
            if code is None or code not in self.FORMATS:
                raise Unsupported(op + " " + name)
            self.buffer_load_format(op[1:], a, mods, fixed_fmt=code)
        elif op == "s_bitset1_b32":
            # sD |= 1 << (src & 31)   (sD is both source and destination)
            b = self.src_u(a[1])
            m, sh, old, r = (self.tmp() for _ in range(4))
            self.emit(f"{m} = OpBitwiseAnd %uint {b} {self.cu(31)}")
            self.emit(f"{sh} = OpShiftLeftLogical %uint {self.cu(1)} {m}")
            self.emit(f"{old} = OpCopyObject %uint {self.load_reg('sgpr', a[0]['r'])}")
            self.emit(f"{r} = OpBitwiseOr %uint {old} {sh}")
            self.dst(a[0], r)
        elif op == "v_alignbit_b32":
            # (({src0, src1} as 64 bits) >> (src2 & 31)) low 32 bits
            hi_, lo_, sh = (self.src_u(t) for t in a[1:4])
            m, inv, a1, b1, nz, r0, r = (self.tmp() for _ in range(7))
            self.emit(f"{m} = OpBitwiseAnd %uint {sh} {self.cu(31)}")
            self.emit(f"{inv} = OpISub %uint {self.cu(32)} {m}")
            self.emit(f"{a1} = OpShiftRightLogical %uint {lo_} {m}")
            self.emit(f"{b1} = OpShiftLeftLogical %uint {hi_} {inv}")
            self.emit(f"{r0} = OpBitwiseOr %uint {a1} {b1}")
            self.emit(f"{nz} = OpINotEqual %bool {m} {self.cu(0)}")
            self.emit(f"{r} = OpSelect %uint {nz} {r0} {lo_}")
            self.dst(a[0], r)
        elif op == "v_lshlrev_b64":
            # v[d:d+1] = v[s:s+1] << (src0 & 63)
            sh, src = a[1], a[2]
            if a[0].get("n") != 2:
                raise Unsupported(op + " operands")
            if src["k"] in ("vgpr", "sgpr") and src.get("n") == 2:
                lo, hi = self.load_reg(src["k"], src["r"]), self.load_reg(src["k"], src["r"] + 1)
            elif src["k"] == "imm":
                # 64-bit inline/literal constant: sign-extended from 32 bits.
                v = src["v"] & 0xFFFFFFFFFFFFFFFF
                lo, hi = self.cu(v & 0xFFFFFFFF), self.cu((v >> 32) & 0xFFFFFFFF)
            else:
                raise Unsupported(op + " operands")
            s = self.src_u(sh)
            l64, h64, x, sh64, m, res, rlo, rhi, t = (self.tmp() for _ in range(9))
            self.emit(f"{l64} = OpUConvert %ulong {lo}")
            self.emit(f"{h64} = OpUConvert %ulong {hi}")
            self.emit(f"{t} = OpShiftLeftLogical %ulong {h64} %c64_32")
            self.emit(f"{x} = OpBitwiseOr %ulong {t} {l64}")
            self.emit(f"{m} = OpBitwiseAnd %uint {s} {self.cu(63)}")
            self.emit(f"{sh64} = OpUConvert %ulong {m}")
            self.emit(f"{res} = OpShiftLeftLogical %ulong {x} {sh64}")
            self.emit(f"{rlo} = OpUConvert %uint {res}")
            hs = self.tmp()
            self.emit(f"{hs} = OpShiftRightLogical %ulong {res} %c64_32")
            self.emit(f"{rhi} = OpUConvert %uint {hs}")
            self.uses_psb = True        # needs Int64 (declared with the PSB types)
            self.dst({"k": "vgpr", "r": a[0]["r"], "n": 1}, rlo)
            self.dst({"k": "vgpr", "r": a[0]["r"] + 1, "n": 1}, rhi)
        elif op == "v_fmac_f16":
            if any(m.startswith("op_sel") for m in mods):
                raise Unsupported(op + " op_sel")
            vals = []
            for s in (a[1], a[2], a[0]):
                u = self.src_u({**s, "neg": False, "abs": False})
                v2, f = self.tmp(), self.tmp()
                self.emit(f"{v2} = OpExtInst %v2float %glsl UnpackHalf2x16 {u}")
                self.emit(f"{f} = OpCompositeExtract %float {v2} 0")
                vals.append(self.modify_f(f, s))
            r, v2, p, old, keep, newv = (self.tmp() for _ in range(6))
            self.emit(f"{r} = OpExtInst %float %glsl Fma {vals[0]} {vals[1]} {vals[2]}")
            self.emit(f"{v2} = OpCompositeConstruct %v2float {r} %f0")
            self.emit(f"{p} = OpExtInst %uint %glsl PackHalf2x16 {v2}")
            lo = self.tmp()
            self.emit(f"{lo} = OpBitwiseAnd %uint {p} {self.cu(0xFFFF)}")
            self.write_lo16(a[0], lo)  # GFX10: f16 ops keep the high half
        elif op in ("v_lshrrev_b16", "v_lshlrev_b16", "v_ashrrev_i16", "v_add_u16", "v_sub_u16", "v_subrev_u16",
                    "v_mul_lo_u16", "v_max_u16", "v_min_u16", "v_max_i16", "v_min_i16", "v_and_b16", "v_or_b16",
                    "v_xor_b16"):
            # 16-bit integer ops on the low halves (op_sel picks high halves in the VOP3 form). Per Mesa ACO
            # (can_use_opsel): the opsel-capable VOP3 (_e64) forms are partial writes that keep the high half,
            # and the VOP2 (_e32) forms define the whole register (high half modelled as zero).
            e64 = ins["op"].endswith("_e64")
            sel = next((m for m in mods if m.startswith("op_sel:")), "op_sel:[0,0,0]")
            sb = [int(x) for x in sel.split("[")[1].rstrip("]").split(",")] + [0, 0, 0]
            if not e64 and sb[:2] != [0, 0]:
                raise Unsupported(op + " op_sel")
            halves = []
            for k, s in enumerate(a[1:3]):
                u = self.src_u(s)
                if sb[k]:
                    t = self.tmp()
                    self.emit(f"{t} = OpShiftRightLogical %uint {u} {self.cu(16)}")
                    u = t
                h = self.tmp()
                self.emit(f"{h} = OpBitwiseAnd %uint {u} {self.cu(0xFFFF)}")
                halves.append(h)
            x, y = halves
            r = self.tmp()
            if op in ("v_lshrrev_b16", "v_lshlrev_b16", "v_ashrrev_i16"):
                sh = self.tmp()                      # shift amount is src0 (reversed operand order), low 4 bits
                self.emit(f"{sh} = OpBitwiseAnd %uint {x} {self.cu(15)}")
                if op == "v_lshrrev_b16":
                    self.emit(f"{r} = OpShiftRightLogical %uint {y} {sh}")
                elif op == "v_lshlrev_b16":
                    self.emit(f"{r} = OpShiftLeftLogical %uint {y} {sh}")
                else:
                    yi, ri = self.tmp(), self.tmp()
                    self.emit(f"{yi} = OpBitFieldSExtract %int {self.as_int(y)} {self.cu(0)} {self.cu(16)}")
                    self.emit(f"{ri} = OpShiftRightArithmetic %int {yi} {sh}")
                    self.emit(f"{r} = OpBitcast %uint {ri}")
            elif op in ("v_max_i16", "v_min_i16"):
                xi, yi, ri = self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{xi} = OpBitFieldSExtract %int {self.as_int(x)} {self.cu(0)} {self.cu(16)}")
                self.emit(f"{yi} = OpBitFieldSExtract %int {self.as_int(y)} {self.cu(0)} {self.cu(16)}")
                self.emit(f"{ri} = OpExtInst %int %glsl {'SMax' if 'max' in op else 'SMin'} {xi} {yi}")
                self.emit(f"{r} = OpBitcast %uint {ri}")
            else:
                k = {"v_add_u16": "OpIAdd", "v_sub_u16": "OpISub", "v_mul_lo_u16": "OpIMul", "v_and_b16": "OpBitwiseAnd",
                     "v_or_b16": "OpBitwiseOr", "v_xor_b16": "OpBitwiseXor"}.get(op)
                if op == "v_subrev_u16":
                    self.emit(f"{r} = OpISub %uint {y} {x}")
                elif k:
                    self.emit(f"{r} = {k} %uint {x} {y}")
                else:
                    self.emit(f"{r} = OpExtInst %uint %glsl {'UMax' if 'max' in op else 'UMin'} {x} {y}")
            lo = self.tmp()
            self.emit(f"{lo} = OpBitwiseAnd %uint {r} {self.cu(0xFFFF)}")
            if e64 and sb[2]:
                # op_sel destination bit: the result goes to the high half, the low half is kept.
                old, keep, sh16, v = (self.tmp() for _ in range(4))
                self.emit(f"{old} = OpCopyObject %uint {self.load_reg('vgpr', a[0]['r'])}")
                self.emit(f"{keep} = OpBitwiseAnd %uint {old} {self.cu(0xFFFF)}")
                self.emit(f"{sh16} = OpShiftLeftLogical %uint {lo} {self.cu(16)}")
                self.emit(f"{v} = OpBitwiseOr %uint {keep} {sh16}")
                self.dst(a[0], v)
            else:
                self.write_lo16(a[0], lo, preserve_high=e64)
        elif op == "v_bcnt_u32_b32":
            x, acc = self.src_u(a[1]), self.src_u(a[2])
            c, r = self.tmp(), self.tmp()
            self.emit(f"{c} = OpBitCount %uint {x}")
            self.emit(f"{r} = OpIAdd %uint {c} {acc}")
            self.dst(a[0], r)
        elif op == "s_bitset0_b32":
            b = self.src_u(a[1])
            m, sh, n, r = (self.tmp() for _ in range(4))
            self.emit(f"{m} = OpBitwiseAnd %uint {b} {self.cu(31)}")
            self.emit(f"{sh} = OpShiftLeftLogical %uint {self.cu(1)} {m}")
            self.emit(f"{n} = OpNot %uint {sh}")
            self.emit(f"{r} = OpBitwiseAnd %uint {self.load_reg('sgpr', a[0]['r'])} {n}")
            self.dst(a[0], r)
        elif op in ("s_min_i32", "s_max_i32"):
            x, y = self.src_u(a[1]), self.src_u(a[2])
            r = self.tmp()
            self.emit(f"{r} = OpExtInst %uint %glsl {'SMin' if 'min' in op else 'SMax'} {x} {y}")
            self.dst(a[0], r)
            # SCC = (src0 selected) per ISA; select-src0 iff src0 <= src1 (min) / >= (max)
            c, z = self.tmp(), self.tmp()
            xi, yi = self.as_int(x), self.as_int(y)
            self.emit(f"{c} = {'OpSLessThanEqual' if 'min' in op else 'OpSGreaterThanEqual'} %bool {xi} {yi}")
            self.emit(f"{z} = OpSelect %uint {c} {self.cu(1)} {self.cu(0)}")
            self.emit(f"OpStore %scc {z}")
        elif op == "v_fma_f16":
            # VOP3 op_sel:[s0,s1,s2,dst]: a set source bit reads that source's high half; a set dst bit writes the
            # result to the high half and keeps the low half (opsel-capable VOP3, see "16-bit results").
            m = next((m for m in mods if m.startswith("op_sel:")), "op_sel:[0,0,0,0]")
            osel = [int(x) for x in m.split("[")[1].rstrip("]").split(",")] + [0, 0, 0, 0]
            vals = []
            for i, s in enumerate(a[1:4]):
                u = self.src_u({**s, "neg": False, "abs": False})
                v2, f = self.tmp(), self.tmp()
                self.emit(f"{v2} = OpExtInst %v2float %glsl UnpackHalf2x16 {u}")
                self.emit(f"{f} = OpCompositeExtract %float {v2} {osel[i]}")
                vals.append(self.modify_f(f, s))
            r, w2, p, lo = (self.tmp() for _ in range(4))
            self.emit(f"{r} = OpExtInst %float %glsl Fma {vals[0]} {vals[1]} {vals[2]}")
            r = self.clamp_mods(r, mods)
            self.emit(f"{w2} = OpCompositeConstruct %v2float {r} %f0")
            self.emit(f"{p} = OpExtInst %uint %glsl PackHalf2x16 {w2}")
            self.emit(f"{lo} = OpBitwiseAnd %uint {p} {self.cu(0xFFFF)}")
            if osel[3]:
                if a[0]["k"] != "vgpr":
                    raise Unsupported(op + " op_sel destination")
                hi16 = self.tmp()
                self.emit(f"{hi16} = OpBitFieldInsert %uint {self.load_reg('vgpr', a[0]['r'])} {lo} {self.cu(16)} {self.cu(16)}")
                self.dst(a[0], hi16)
            else:
                self.write_lo16(a[0], lo)  # GFX10: f16 ops keep the high half
        elif op in ("v_pk_add_f16", "v_pk_mul_f16", "v_pk_max_f16", "v_pk_min_f16", "v_pk_fma_f16"):
            # Packed f16x2 (VOP3P). Result lo: source i takes its half from op_sel[i] (0 = lo, 1 = hi), negated by
            # neg_lo[i]. Result hi: half from op_sel_hi[i], negated by neg_hi[i]. Defaults: op_sel [0,0],
            # op_sel_hi [1,1], no negation.
            def bits(name, default):
                m = next((m for m in mods if m.startswith(name + ":")), None)
                return [int(v) for v in m.split("[")[1].rstrip("]").split(",")] if m else default
            ns = 3 if op == "v_pk_fma_f16" else 2
            sel, selhi = bits("op_sel", [0] * ns), bits("op_sel_hi", [1] * ns)
            nlo, nhi = bits("neg_lo", [0] * ns), bits("neg_hi", [0] * ns)
            srcs = []
            for i in range(ns):
                u = self.src_u({**a[1 + i], "neg": False, "abs": False})
                v2 = self.tmp()
                self.emit(f"{v2} = OpExtInst %v2float %glsl UnpackHalf2x16 {u}")
                comps = []
                for half_sel, neg in ((sel[i], nlo[i]), (selhi[i], nhi[i])):
                    c = self.tmp()
                    self.emit(f"{c} = OpCompositeExtract %float {v2} {half_sel}")
                    if neg:
                        n2 = self.tmp()
                        self.emit(f"{n2} = OpFNegate %float {c}")
                        c = n2
                    comps.append(c)
                vv = self.tmp()
                self.emit(f"{vv} = OpCompositeConstruct %v2float {comps[0]} {comps[1]}")
                srcs.append(vv)
            vx, vy = srcs[:2]
            r, p = self.tmp(), self.tmp()
            if op == "v_pk_fma_f16":
                self.emit(f"{r} = OpExtInst %v2float %glsl Fma {vx} {vy} {srcs[2]}")
            elif op in ("v_pk_add_f16", "v_pk_mul_f16"):
                self.emit(f"{r} = {'OpFAdd' if 'add' in op else 'OpFMul'} %v2float {vx} {vy}")
            else:
                self.emit(f"{r} = OpExtInst %v2float %glsl {'FMax' if 'max' in op else 'FMin'} {vx} {vy}")
            if "clamp" in mods:
                c = self.tmp()
                z2, o2 = self.tmp(), self.tmp()
                self.emit(f"{z2} = OpCompositeConstruct %v2float %f0 %f0")
                self.emit(f"{o2} = OpCompositeConstruct %v2float %f1 %f1")
                self.emit(f"{c} = OpExtInst %v2float %glsl FClamp {r} {z2} {o2}")
                r = c
            self.emit(f"{p} = OpExtInst %uint %glsl PackHalf2x16 {r}")
            self.dst(a[0], p)
        elif op in ("v_movrels_b32", "v_movreld_b32", "v_movrelsd_b32"):
            # Relative VGPR indexing through M0: movrels reads VGPR[src + M0], movreld writes VGPR[dst + M0],
            # movrelsd does both. The register file is an array, so these become dynamic array accesses.
            d, s = a[0], a[1]
            if d["k"] != "vgpr" or s["k"] != "vgpr":
                raise Unsupported(op + " operands")
            m0 = self.tmp()
            self.emit(f"{m0} = OpLoad %uint %m0")
            if op in ("v_movrels_b32", "v_movrelsd_b32"):
                si, sp, v = self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{si} = OpIAdd %uint {self.cu(s['r'])} {m0}")
                self.emit(f"{sp} = OpAccessChain %_ptr_Function_uint %vgpr {si}")
                self.emit(f"{v} = OpLoad %uint {sp}")
            else:
                v = self.load_reg("vgpr", s["r"])
            if op in ("v_movreld_b32", "v_movrelsd_b32"):
                di, dp, old, sel = (self.tmp() for _ in range(4))
                self.emit(f"{di} = OpIAdd %uint {self.cu(d['r'])} {m0}")
                self.emit(f"{dp} = OpAccessChain %_ptr_Function_uint %vgpr {di}")
                self.emit(f"{old} = OpLoad %uint {dp}")
                self.emit(f"{sel} = OpSelect %uint {self.active_b()} {v} {old}")
                self.emit(f"OpStore {dp} {sel}")
            else:
                self.dst(d, v)
        elif op == "v_med3_f16":
            if any(m.startswith("op_sel") for m in mods):
                raise Unsupported(op + " op_sel")
            vals = []
            for s in a[1:4]:
                u = self.src_u({**s, "neg": False, "abs": False})
                v2, f = self.tmp(), self.tmp()
                self.emit(f"{v2} = OpExtInst %v2float %glsl UnpackHalf2x16 {u}")
                self.emit(f"{f} = OpCompositeExtract %float {v2} 0")
                vals.append(self.modify_f(f, s))
            lo_, hi_, r = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{lo_} = OpExtInst %float %glsl FMin {vals[0]} {vals[1]}")
            self.emit(f"{hi_} = OpExtInst %float %glsl FMax {vals[0]} {vals[1]}")
            self.emit(f"{r} = OpExtInst %float %glsl FClamp {vals[2]} {lo_} {hi_}")
            r = self.clamp_mods(r, mods)
            w2, p, lo = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{w2} = OpCompositeConstruct %v2float {r} %f0")
            self.emit(f"{p} = OpExtInst %uint %glsl PackHalf2x16 {w2}")
            self.emit(f"{lo} = OpBitwiseAnd %uint {p} {self.cu(0xFFFF)}")
            self.write_lo16(a[0], lo)     # opsel-capable VOP3: keeps the high half (Mesa ACO can_use_opsel)
        elif op == "v_xad_u32":
            x, y, z = (self.src_u(t) for t in a[1:4])
            t, r = self.tmp(), self.tmp()
            self.emit(f"{t} = OpBitwiseXor %uint {x} {y}")
            self.emit(f"{r} = OpIAdd %uint {t} {z}")
            self.dst(a[0], r)
        elif op == "s_not_b64":
            v = self.mask_src(a[1])
            r = self.tmp()
            self.emit(f"{r} = OpNot %uint {v}")
            self.mask_dst(a[0], r)
            z = self.tmp()
            self.emit(f"{z} = OpSelect %uint {self.tmp_bool_nz(r)} {self.cu(1)} {self.cu(0)}")
            self.emit(f"OpStore %scc {z}")
        elif op in ("v_rsq_f16", "v_rcp_f16", "v_sqrt_f16", "v_exp_f16", "v_log_f16", "v_floor_f16", "v_fract_f16",
                    "v_ceil_f16", "v_trunc_f16", "v_rndne_f16"):
            # f16 unary on the low half; result in the low half, high half cleared (non-op_sel form).
            if any(m.startswith("op_sel") for m in mods):
                raise Unsupported(op + " op_sel")
            u = self.src_u({**a[1], "neg": False, "abs": False})
            v2, f = self.tmp(), self.tmp()
            self.emit(f"{v2} = OpExtInst %v2float %glsl UnpackHalf2x16 {u}")
            self.emit(f"{f} = OpCompositeExtract %float {v2} 0")
            f = self.modify_f(f, a[1])
            r = self.tmp()
            fn = {"v_rsq_f16": "InverseSqrt", "v_sqrt_f16": "Sqrt", "v_exp_f16": "Exp2", "v_log_f16": "Log2",
                  "v_floor_f16": "Floor", "v_fract_f16": "Fract", "v_ceil_f16": "Ceil", "v_trunc_f16": "Trunc",
                  "v_rndne_f16": "RoundEven"}.get(op)
            if fn:
                self.emit(f"{r} = OpExtInst %float %glsl {fn} {f}")
            else:
                self.emit(f"{r} = OpFDiv %float %f1 {f}")
            r = self.clamp_mods(r, mods)
            w2, p, lo = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{w2} = OpCompositeConstruct %v2float {r} %f0")
            self.emit(f"{p} = OpExtInst %uint %glsl PackHalf2x16 {w2}")
            self.emit(f"{lo} = OpBitwiseAnd %uint {p} {self.cu(0xFFFF)}")
            self.write_lo16(a[0], lo)  # GFX10: f16 ops keep the high half
        elif op == "v_not_b32":
            x = self.src_u(a[1])
            r = self.tmp()
            self.emit(f"{r} = OpNot %uint {x}")
            self.dst(a[0], r)
        elif op == "v_mul_hi_i32":
            x, y = self.as_int(self.src_u(a[1])), self.as_int(self.src_u(a[2]))
            s, hi, r = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{s} = OpSMulExtended %smul {x} {y}")
            self.emit(f"{hi} = OpCompositeExtract %int {s} 1")
            self.emit(f"{r} = OpBitcast %uint {hi}")
            self.uses_smul = True
            self.dst(a[0], r)
        elif op in ("v_add_f16", "v_sub_f16", "v_mul_f16", "v_max_f16", "v_min_f16"):
            # f16 math on the low halves; the result's high half is cleared (non-op_sel form).
            if any(m.startswith("op_sel") for m in mods):
                raise Unsupported(op + " op_sel")
            vals = []
            for s in a[1:3]:
                u = self.src_u({**s, "neg": False, "abs": False})
                v2, f = self.tmp(), self.tmp()
                self.emit(f"{v2} = OpExtInst %v2float %glsl UnpackHalf2x16 {u}")
                self.emit(f"{f} = OpCompositeExtract %float {v2} 0")
                vals.append(self.modify_f(f, s))
            r = self.tmp()
            k = {"v_add_f16": "OpFAdd", "v_sub_f16": "OpFSub", "v_mul_f16": "OpFMul"}.get(op)
            if k:
                self.emit(f"{r} = {k} %float {vals[0]} {vals[1]}")
            else:
                self.emit(f"{r} = OpExtInst %float %glsl {'FMax' if 'max' in op else 'FMin'} {vals[0]} {vals[1]}")
            r = self.clamp_mods(r, mods)
            v2, p, lo = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{v2} = OpCompositeConstruct %v2float {r} %f0")
            self.emit(f"{p} = OpExtInst %uint %glsl PackHalf2x16 {v2}")
            self.emit(f"{lo} = OpBitwiseAnd %uint {p} {self.cu(0xFFFF)}")
            self.write_lo16(a[0], lo)  # GFX10: f16 ops keep the high half
        elif op.startswith(("v_cmp_", "v_cmpx_")) and op.endswith(("_u64", "_i64")) and op.split("_")[2] in ("eq", "ne", "lg"):
            # 64-bit equality compare: both halves of the register pairs. GFX10 v_cmpx has no destination operand:
            # it writes EXEC only (inactive lanes stay inactive).
            cmpx = op.startswith("v_cmpx_")
            d, x, y = (None, a[-2], a[-1]) if cmpx else (a[0], a[1], a[2])
            def pair(s):
                if s["k"] in ("vgpr", "sgpr") and s.get("n") == 2:
                    return self.load_reg(s["k"], s["r"]), self.load_reg(s["k"], s["r"] + 1)
                if s["k"] == "imm":
                    v = s["v"] & 0xFFFFFFFFFFFFFFFF
                    return self.cu(v & 0xFFFFFFFF), self.cu(v >> 32)
                raise Unsupported(op + " operand")
            (x0, x1), (y0, y1) = pair(x), pair(y)
            e0, e1, eq, r = (self.tmp() for _ in range(4))
            self.emit(f"{e0} = OpIEqual %bool {x0} {y0}")
            self.emit(f"{e1} = OpIEqual %bool {x1} {y1}")
            self.emit(f"{eq} = OpLogicalAnd %bool {e0} {e1}")
            if op.split("_")[2] != "eq":
                ne = self.tmp()
                self.emit(f"{ne} = OpLogicalNot %bool {eq}")
                eq = ne
            if cmpx:
                both = self.tmp()
                self.emit(f"{both} = OpLogicalAnd %bool {eq} {self.active_b()}")
                self.emit(f"OpStore %active {both}")
                self.uses_active = True
            else:
                eqa = self.tmp()
                self.emit(f"{eqa} = OpLogicalAnd %bool {eq} {self.active_b()}")   # inactive lanes write 0
                self.emit(f"{r} = OpSelect %uint {eqa} {self.cu(0xFFFFFFFF)} {self.cu(0)}")
                if d["k"] == "special" and d["t"] in ("vcc", "vcc_lo"):
                    self.emit(f"OpStore %vcc {r}")
                else:
                    self.mask_dst(d, r)
        elif op == "s_not_b32":
            x = self.src_u(a[1])
            r = self.tmp()
            self.emit(f"{r} = OpNot %uint {x}")
            self.dst(a[0], r)
            z = self.tmp()
            self.emit(f"{z} = OpSelect %uint {self.tmp_bool_nz(r)} {self.cu(1)} {self.cu(0)}")
            self.emit(f"OpStore %scc {z}")
        elif op in ("s_cmp_eq_u64", "s_cmp_lg_u64"):
            # 64-bit compare of masks/pointers: equal iff both halves are equal.
            def halves(s):
                if s["k"] == "sgpr" and s.get("n") == 2:
                    return self.load_reg("sgpr", s["r"]), self.load_reg("sgpr", s["r"] + 1)
                if s["k"] == "imm":
                    v = s["v"] & 0xFFFFFFFFFFFFFFFF
                    return self.cu(v & 0xFFFFFFFF), self.cu(v >> 32)
                m = self.mask_src(s)
                return m, m
            (x0, x1), (y0, y1) = halves(a[0]), halves(a[1])
            e0, e1, eq, r = (self.tmp() for _ in range(4))
            self.emit(f"{e0} = OpIEqual %bool {x0} {y0}")
            self.emit(f"{e1} = OpIEqual %bool {x1} {y1}")
            self.emit(f"{eq} = OpLogicalAnd %bool {e0} {e1}")
            if op == "s_cmp_lg_u64":
                ne = self.tmp()
                self.emit(f"{ne} = OpLogicalNot %bool {eq}")
                eq = ne
            self.emit(f"{r} = OpSelect %uint {eq} {self.cu(1)} {self.cu(0)}")
            self.emit(f"OpStore %scc {r}")
        elif op.startswith("buffer_atomic_"):
            self.buffer_atomic(op, a, mods)
        elif op == "image_get_lod":
            self.image_get_lod(a, mods)
        elif op == "s_addk_i32":
            v = a[1]["v"] & 0xFFFF
            k = v - 0x10000 if v & 0x8000 else v
            x = self.src_u(a[0])            # the destination can be m0/vcc_hi as well as an SGPR
            r = self.tmp()
            self.emit(f"{r} = OpIAdd %uint {x} {self.cu(k)}")
            self.dst(a[0], r)
        elif op in ("v_add_co_u32", "v_add_co_ci_u32", "v_sub_co_u32", "v_sub_co_ci_u32", "v_subrev_co_u32",
                    "v_subrev_co_ci_u32"):
            # Lane masks are all-ones/zero: carry in = (vcc or the named mask) != 0, carry out stored the same way.
            vd, cd = a[0], a[1]
            srcs = a[2:]
            x, y = self.src_u(srcs[0]), self.src_u(srcs[1])
            if op.startswith("v_subrev_co"):
                x, y = y, x
            sub = "sub" in op
            if op.endswith("ci_u32"):
                cin_m = self.mask_src(srcs[2]) if len(srcs) > 2 else self.mask_src({"k": "special", "t": "vcc"})
                cin = self.tmp()
                self.emit(f"{cin} = OpSelect %uint {self.tmp_bool_nz(cin_m)} {self.cu(1)} {self.cu(0)}")
            else:
                cin = self.cu(0)
            s1, c1, s2, c2, r, carry = (self.tmp() for _ in range(6))
            if sub:
                self.emit(f"{s1} = OpISubBorrow %umul {x} {y}")
            else:
                self.emit(f"{s1} = OpIAddCarry %umul {x} {y}")
            r1 = self.tmp()
            self.emit(f"{r1} = OpCompositeExtract %uint {s1} 0")
            self.emit(f"{c1} = OpCompositeExtract %uint {s1} 1")
            self.emit(f"{s2} = {'OpISubBorrow' if sub else 'OpIAddCarry'} %umul {r1} {cin}")
            self.emit(f"{r} = OpCompositeExtract %uint {s2} 0")
            self.emit(f"{c2} = OpCompositeExtract %uint {s2} 1")
            self.emit(f"{carry} = OpBitwiseOr %uint {c1} {c2}")
            self.dst(vd, r)
            m = self.tmp()
            self.emit(f"{m} = OpSelect %uint {self.tmp_bool_nz(carry)} {self.cu(0xFFFFFFFF)} {self.cu(0)}")
            if not (cd["k"] == "special" and cd["t"] == "null"):
                self.mask_dst(cd, m)
        elif op == "v_mad_legacy_f32":
            # Legacy multiply treats 0 * anything (inf/NaN included) as 0.
            x, y, z = (self.src_f(t) for t in a[1:4])
            m, zx, zy, either, sel, r = (self.tmp() for _ in range(6))
            self.emit(f"{m} = OpFMul %float {x} {y}")
            self.emit(f"{zx} = OpFOrdEqual %bool {x} %f0")
            self.emit(f"{zy} = OpFOrdEqual %bool {y} %f0")
            self.emit(f"{either} = OpLogicalOr %bool {zx} {zy}")
            self.emit(f"{sel} = OpSelect %float {either} %f0 {m}")
            self.emit(f"{r} = OpFAdd %float {sel} {z}")
            self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
        elif op in ("v_min3_i32", "v_min3_u32", "v_max3_i32", "v_max3_u32"):
            x, y, z = (self.src_u(t) for t in a[1:4])
            f = ("S" if op.endswith("i32") else "U") + ("Min" if "min" in op else "Max")
            m, r = self.tmp(), self.tmp()
            self.emit(f"{m} = OpExtInst %uint %glsl {f} {x} {y}")
            self.emit(f"{r} = OpExtInst %uint %glsl {f} {m} {z}")
            self.dst(a[0], r)
        elif op in ("v_med3_i32", "v_med3_u32"):
            x, y, z = (self.src_u(t) for t in a[1:4])
            s = "S" if op.endswith("i32") else "U"
            lo, hi, r = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{lo} = OpExtInst %uint %glsl {s}Min {x} {y}")
            self.emit(f"{hi} = OpExtInst %uint %glsl {s}Max {x} {y}")
            self.emit(f"{r} = OpExtInst %uint %glsl {s}Clamp {z} {lo} {hi}")
            self.dst(a[0], r)
        elif op.startswith("s_load_dword") or op.startswith("s_buffer_load_dword"):
            self.scalar_load(op, a, ins["mods"])
        elif op in ("s_lshl_b64", "s_lshr_b64"):
            # 64-bit shift of an SGPR pair by (src1 & 63); SCC = result != 0.
            d, x = a[0], a[1]
            if d["k"] != "sgpr" or d.get("n") != 2:
                raise Unsupported(op + " destination")
            if x["k"] == "sgpr" and x.get("n") == 2:
                lo, hi = self.load_reg("sgpr", x["r"]), self.load_reg("sgpr", x["r"] + 1)
            elif x["k"] == "imm":
                v = x["v"] & 0xFFFFFFFFFFFFFFFF
                lo, hi = self.cu(v & 0xFFFFFFFF), self.cu(v >> 32)
            else:
                raise Unsupported(op + " source")
            sh = self.src_u(a[2])
            l64, h64, t, v64, m, s64, r, rlo, rh, rhi = (self.tmp() for _ in range(10))
            self.emit(f"{l64} = OpUConvert %ulong {lo}")
            self.emit(f"{h64} = OpUConvert %ulong {hi}")
            self.emit(f"{t} = OpShiftLeftLogical %ulong {h64} %c64_32")
            self.emit(f"{v64} = OpBitwiseOr %ulong {t} {l64}")
            self.emit(f"{m} = OpBitwiseAnd %uint {sh} {self.cu(63)}")
            self.emit(f"{s64} = OpUConvert %ulong {m}")
            self.emit(f"{r} = {'OpShiftLeftLogical' if op == 's_lshl_b64' else 'OpShiftRightLogical'} %ulong {v64} {s64}")
            self.emit(f"{rlo} = OpUConvert %uint {r}")
            self.emit(f"{rh} = OpShiftRightLogical %ulong {r} %c64_32")
            self.emit(f"{rhi} = OpUConvert %uint {rh}")
            self.store_reg("sgpr", d["r"], rlo)
            self.store_reg("sgpr", d["r"] + 1, rhi)
            nz, z = self.tmp(), self.tmp()
            orr = self.tmp()
            self.emit(f"{orr} = OpBitwiseOr %uint {rlo} {rhi}")
            self.emit(f"{z} = OpSelect %uint {self.tmp_bool_nz(orr)} {self.cu(1)} {self.cu(0)}")
            self.emit(f"OpStore %scc {z}")
            self.uses_psb = True        # Int64 capability and %ulong come with the PSB declarations
        elif op == "s_ashr_i32":
            x, sh = self.src_u(a[1]), self.src_u(a[2])
            m, xi, ri, r = (self.tmp() for _ in range(4))
            self.emit(f"{m} = OpBitwiseAnd %uint {sh} {self.cu(31)}")
            self.emit(f"{xi} = OpBitcast %int {x}")
            self.emit(f"{ri} = OpShiftRightArithmetic %int {xi} {m}")
            self.emit(f"{r} = OpBitcast %uint {ri}")
            self.dst(a[0], r)
            z = self.tmp()
            self.emit(f"{z} = OpSelect %uint {self.tmp_bool_nz(r)} {self.cu(1)} {self.cu(0)}")
            self.emit(f"OpStore %scc {z}")
        elif op.startswith(("global_load_dword", "global_store_dword")):
            self.global_mem(op, a, mods)
        elif op in ("s_add_u32", "s_addc_u32", "s_sub_u32", "s_subb_u32"):
            # 32-bit add/sub with carry/borrow through SCC (used for 64-bit address arithmetic, e.g. s_getpc + offset).
            x, y = self.src_u(a[1]), self.src_u(a[2])
            sub = op.startswith("s_sub")
            s1, r1, c1 = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{s1} = {'OpISubBorrow' if sub else 'OpIAddCarry'} %umul {x} {y}")
            self.emit(f"{r1} = OpCompositeExtract %uint {s1} 0")
            self.emit(f"{c1} = OpCompositeExtract %uint {s1} 1")
            r, carry = r1, c1
            if op in ("s_addc_u32", "s_subb_u32"):
                cin, s2, r2, c2, cc = (self.tmp() for _ in range(5))
                self.emit(f"{cin} = OpLoad %uint %scc")
                self.emit(f"{s2} = {'OpISubBorrow' if sub else 'OpIAddCarry'} %umul {r1} {cin}")
                self.emit(f"{r2} = OpCompositeExtract %uint {s2} 0")
                self.emit(f"{c2} = OpCompositeExtract %uint {s2} 1")
                self.emit(f"{cc} = OpBitwiseOr %uint {c1} {c2}")
                r, carry = r2, cc
            self.dst(a[0], r)
            self.emit(f"OpStore %scc {carry}")
        elif op == "s_getpc_b64":
            # Address of the next instruction: code base (push constants 34-35: where the runtime put this shader's
            # FSSM blob in the Xbox VA window) + offset of the next instruction within the blob.
            if a[0]["k"] != "sgpr" or a[0].get("n") != 2:
                raise Unsupported("s_getpc_b64 destination")
            nxt = ins["pc"] + len(ins["enc"]) // 2
            lo, hi, s, rlo, c, rhi = (self.tmp() for _ in range(6))
            self.emit(f"{lo} = OpLoad %uint %ud_code_lo_p")
            self.emit(f"{hi} = OpLoad %uint %ud_code_hi_p")
            self.emit(f"{s} = OpIAddCarry %umul {lo} {self.cu(nxt)}")
            self.emit(f"{rlo} = OpCompositeExtract %uint {s} 0")
            self.emit(f"{c} = OpCompositeExtract %uint {s} 1")
            self.emit(f"{rhi} = OpIAdd %uint {hi} {c}")
            self.store_reg("sgpr", a[0]["r"], rlo)
            self.store_reg("sgpr", a[0]["r"] + 1, rhi)
            self.uses_codebase = True
            self.cu(3), self.cu(4)
        elif op in ("s_add_i32", "s_sub_i32", "s_min_u32", "s_max_u32", "s_xor_b32", "s_andn2_b32",
                    "s_cselect_b32"):
            x, y = self.src_u(a[1]), self.src_u(a[2])
            r = self.tmp()
            if op == "s_cselect_b32":
                c, b = self.tmp(), self.tmp()
                self.emit(f"{c} = OpLoad %uint %scc")
                self.emit(f"{b} = OpINotEqual %bool {c} {self.cu(0)}")
                self.emit(f"{r} = OpSelect %uint {b} {x} {y}")
            elif op == "s_andn2_b32":
                n = self.tmp()
                self.emit(f"{n} = OpNot %uint {y}")
                self.emit(f"{r} = OpBitwiseAnd %uint {x} {n}")
            elif op in ("s_min_u32", "s_max_u32"):
                self.emit(f"{r} = OpExtInst %uint %glsl {'UMin' if 'min' in op else 'UMax'} {x} {y}")
            else:
                k = {"s_add_i32": "OpIAdd", "s_sub_i32": "OpISub", "s_sub_u32": "OpISub",
                     "s_xor_b32": "OpBitwiseXor"}[op]
                self.emit(f"{r} = {k} %uint {x} {y}")
            self.dst(a[0], r)
        elif op in ("s_wqm_b64", "s_wqm_b32"):
            # Whole-quad mode: a lane becomes active if any lane of its 2x2 quad is. One lane per invocation can't
            # see its neighbours, so this is the identity (APPROXIMATION: helper lanes stay as they were).
            v = self.mask_src(a[1])
            self.mask_dst(a[0], v)
            z = self.tmp()
            self.emit(f"{z} = OpSelect %uint {self.tmp_bool_nz(v)} {self.cu(1)} {self.cu(0)}")
            self.emit(f"OpStore %scc {z}")
        elif op == "v_fma_mix_f32":
            # Each source is f32, or (op_sel_hi bit set) an f16 from the low/high half chosen by op_sel.
            sel = next((m for m in mods if m.startswith("op_sel:")), "op_sel:[0,0,0]")
            selhi = next((m for m in mods if m.startswith("op_sel_hi:")), "op_sel_hi:[0,0,0]")
            sb = [int(x) for x in sel.split("[")[1].rstrip("]").split(",")]
            hb = [int(x) for x in selhi.split("[")[1].rstrip("]").split(",")]
            vals = []
            for i, src in enumerate(a[1:4]):
                if hb[i]:
                    u = self.src_u({**src, "neg": False, "abs": False})
                    v2, f = self.tmp(), self.tmp()
                    self.emit(f"{v2} = OpExtInst %v2float %glsl UnpackHalf2x16 {u}")
                    self.emit(f"{f} = OpCompositeExtract %float {v2} {sb[i]}")
                    f = self.tof(self.modify(self.tou(f), src))
                    vals.append(f)
                else:
                    vals.append(self.src_f(src))
            r = self.tmp()
            self.emit(f"{r} = OpExtInst %float %glsl Fma {vals[0]} {vals[1]} {vals[2]}")
            self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
        elif op == "v_mad_f32":
            x, y, z = (self.src_f(t) for t in a[1:4])
            m, r = self.tmp(), self.tmp()
            self.emit(f"{m} = OpFMul %float {x} {y}")
            self.emit(f"{r} = OpFAdd %float {m} {z}")
            self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
        elif op in ("v_mul_lo_u32", "v_min_u32", "v_max_u32", "v_min_i32", "v_max_i32"):
            x, y = self.src_u(a[1]), self.src_u(a[2])
            r = self.tmp()
            if op == "v_mul_lo_u32":
                self.emit(f"{r} = OpIMul %uint {x} {y}")
            else:
                f = {"v_min_u32": "UMin", "v_max_u32": "UMax", "v_min_i32": "SMin", "v_max_i32": "SMax"}[op]
                self.emit(f"{r} = OpExtInst %uint %glsl {f} {x} {y}")
            self.dst(a[0], r)
        elif op == "v_cvt_f32_f16":
            u = self.src_u({**a[1], "neg": False, "abs": False})
            v2, f = self.tmp(), self.tmp()
            self.emit(f"{v2} = OpExtInst %v2float %glsl UnpackHalf2x16 {u}")
            self.emit(f"{f} = OpCompositeExtract %float {v2} 0")
            self.dst(a[0], self.tou(self.modify_f(f, a[1])))
        elif op in ("v_interp_p1_f32", "v_interp_p2_f32", "v_interp_mov_f32"):
            # Single-lane model: the barycentric p1/p2 pair (and the flat mov) both yield the attribute value.
            attr = next((x for x in a if x["k"] == "attr"), None)
            if attr is None or self.stage != "ps":
                raise Unsupported(op + " form")
            self.dst(a[0], self.tou(self.attribute(attr["a"], attr["c"])))
        elif op == "s_movk_i32":
            v = a[1]["v"] & 0xFFFF
            self.dst(a[0], self.cu(v - 0x10000 if v & 0x8000 else v))
        elif op.startswith("v_cmp_") and not op.startswith("v_cmp_class"):
            self.compare(op, a, scalar=False)
        elif op.startswith("v_cmpx_") and not op.startswith("v_cmpx_class"):
            self.compare_x(op, a)
        elif op in ("s_and_saveexec_b64", "s_or_saveexec_b64", "s_andn2_saveexec_b64", "s_and_saveexec_b32",
                    "s_or_saveexec_b32", "s_andn2_saveexec_b32"):
            saved = self.exec_value()
            src = self.mask_src(a[1])
            r = self.tmp()
            kind = op[2:].split("_saveexec")[0]
            if kind == "andn2":
                n = self.tmp()
                self.emit(f"{n} = OpNot %uint {saved}")
                self.emit(f"{r} = OpBitwiseAnd %uint {src} {n}")
            else:
                self.emit(f"{r} = {'OpBitwiseAnd' if kind == 'and' else 'OpBitwiseOr'} %uint {src} {saved}")
            self.mask_dst(a[0], saved)
            self.set_exec(r)
            z = self.tmp()
            self.emit(f"{z} = OpSelect %uint {self.tmp_bool_nz(r)} {self.cu(1)} {self.cu(0)}")
            self.emit(f"OpStore %scc {z}")
        elif op in ("s_andn2_wrexec_b64", "s_andn2_wrexec_b32"):
            # RDNA ISA: EXEC = S0 & ~EXEC; D = EXEC; SCC = (EXEC != 0). Two operands: sdst, ssrc0.
            x, y = self.mask_src(a[1]), self.exec_value()
            n, r = self.tmp(), self.tmp()
            self.emit(f"{n} = OpNot %uint {y}")
            self.emit(f"{r} = OpBitwiseAnd %uint {x} {n}")
            self.set_exec(r)
            self.mask_dst(a[0], r)
            z = self.tmp()
            self.emit(f"{z} = OpSelect %uint {self.tmp_bool_nz(r)} {self.cu(1)} {self.cu(0)}")
            self.emit(f"OpStore %scc {z}")
        elif op in ("s_bfe_u32", "s_bfe_i32"):
            # src1 packs offset in bits [4:0] and width in bits [22:16].
            x, ctl = self.src_u(a[1]), self.src_u(a[2])
            off, wid, r = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{off} = OpBitwiseAnd %uint {ctl} {self.cu(31)}")
            self.emit(f"{wid} = OpBitFieldUExtract %uint {ctl} {self.cu(16)} {self.cu(7)}")
            if op == "s_bfe_u32":
                self.emit(f"{r} = OpBitFieldUExtract %uint {x} {off} {wid}")
            else:
                xi, ri = self.as_int(x), self.tmp()
                self.emit(f"{ri} = OpBitFieldSExtract %int {xi} {off} {wid}")
                self.emit(f"{r} = OpBitcast %uint {ri}")
            self.dst(a[0], r)
            z = self.tmp()
            self.emit(f"{z} = OpSelect %uint {self.tmp_bool_nz(r)} {self.cu(1)} {self.cu(0)}")
            self.emit(f"OpStore %scc {z}")
        elif op == "v_perm_b32":
            # Byte permute: result byte i = selector byte i picks from the 8 bytes {src0, src1} (src1 is low).
            # Selector values 0-7 pick a byte; 12 -> 0x00, >=13 -> 0xFF (sign/other cases not used here are 0).
            s0, s1, sel = (self.src_u(t) for t in a[1:4])
            out = self.cu(0)
            for i in range(4):
                sb, lo_pick, hi_pick, k, b, lo_sh, hi_sh, chosen, inrange, val, shifted, acc = \
                    (self.tmp() for _ in range(12))
                self.emit(f"{sb} = OpBitFieldUExtract %uint {sel} {self.cu(8 * i)} {self.cu(8)}")
                self.emit(f"{k} = OpBitwiseAnd %uint {sb} {self.cu(7)}")
                self.emit(f"{b} = OpULessThan %bool {k} {self.cu(4)}")
                self.emit(f"{lo_sh} = OpIMul %uint {k} {self.cu(8)}")
                self.emit(f"{lo_pick} = OpShiftRightLogical %uint {s1} {lo_sh}")
                self.emit(f"{hi_sh} = OpISub %uint {lo_sh} {self.cu(32)}")
                self.emit(f"{hi_pick} = OpShiftRightLogical %uint {s0} {hi_sh}")
                self.emit(f"{chosen} = OpSelect %uint {b} {lo_pick} {hi_pick}")
                self.emit(f"{inrange} = OpULessThan %bool {sb} {self.cu(8)}")
                ff, big = self.tmp(), self.tmp()
                self.emit(f"{big} = OpUGreaterThanEqual %bool {sb} {self.cu(13)}")
                self.emit(f"{ff} = OpSelect %uint {big} {self.cu(0xFF)} {self.cu(0)}")
                self.emit(f"{val} = OpSelect %uint {inrange} {chosen} {ff}")
                m8 = self.tmp()
                self.emit(f"{m8} = OpBitwiseAnd %uint {val} {self.cu(0xFF)}")
                self.emit(f"{shifted} = OpShiftLeftLogical %uint {m8} {self.cu(8 * i)}")
                self.emit(f"{acc} = OpBitwiseOr %uint {out} {shifted}")
                out = acc
            self.dst(a[0], out)
        elif op in ("s_and_b64", "s_or_b64", "s_xor_b64", "s_andn2_b64", "s_orn2_b64",
                    "s_and_b32_mask", "s_or_b32_mask"):
            x, y = self.mask_src(a[1]), self.mask_src(a[2])
            r = self.tmp()
            if op.endswith("n2_b64"):
                n = self.tmp()
                self.emit(f"{n} = OpNot %uint {y}")
                y = n
            k = {"and": "OpBitwiseAnd", "or": "OpBitwiseOr", "xor": "OpBitwiseXor", "andn2": "OpBitwiseAnd",
                 "orn2": "OpBitwiseOr"}[op[2:].rsplit("_", 1)[0]]
            self.emit(f"{r} = {k} %uint {x} {y}")
            self.mask_dst(a[0], r)
            z = self.tmp()
            self.emit(f"{z} = OpSelect %uint {self.tmp_bool_nz(r)} {self.cu(1)} {self.cu(0)}")
            self.emit(f"OpStore %scc {z}")
        elif op == "s_cselect_b64":
            x, y = self.mask_src(a[1]), self.mask_src(a[2])
            c, b, r = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{c} = OpLoad %uint %scc")
            self.emit(f"{b} = OpINotEqual %bool {c} {self.cu(0)}")
            self.emit(f"{r} = OpSelect %uint {b} {x} {y}")
            self.mask_dst(a[0], r)
        elif op in ("v_interp_p1ll_f16", "v_interp_p1lv_f16", "v_interp_p2_f16"):
            attr = next((x for x in a if x["k"] == "attr"), None)
            if attr is None or self.stage != "ps":
                raise Unsupported(op + " form")
            f = self.attribute(attr["a"], attr["c"])
            if op == "v_interp_p2_f16":
                v, p, r = self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{v} = OpCompositeConstruct %v2float {f} %f0")
                self.emit(f"{p} = OpExtInst %uint %glsl PackHalf2x16 {v}")
                self.emit(f"{r} = OpBitwiseAnd %uint {p} {self.cu(0xFFFF)}")
                self.dst(a[0], r)
            else:   # p1ll/p1lv produce the f32 intermediate consumed by p2_f16
                self.dst(a[0], self.tou(f))
        elif op.startswith("s_cmp_"):
            self.compare(op, a, scalar=True)
        elif op.startswith("s_cmpk_"):
            # s_cmpk_<cond>_<ty> sX, imm16: zero-extended for u32, sign-extended for i32.
            ty = op.rsplit("_", 1)[1]
            v = a[1]["v"] & 0xFFFF
            if ty == "i32" and v & 0x8000:
                v -= 0x10000
            self.compare("s_cmp_" + op[len("s_cmpk_"):], [a[0], {"k": "imm", "v": v}], scalar=True)
        elif op == "s_bfm_b32":
            # Bitfield mask: ((1 << width) - 1) << offset, width/offset from the low 5 bits of the sources.
            w, o = self.src_u(a[1]), self.src_u(a[2])
            wm, om, one, m1, r = (self.tmp() for _ in range(5))
            self.emit(f"{wm} = OpBitwiseAnd %uint {w} {self.cu(31)}")
            self.emit(f"{om} = OpBitwiseAnd %uint {o} {self.cu(31)}")
            self.emit(f"{one} = OpShiftLeftLogical %uint {self.cu(1)} {wm}")
            self.emit(f"{m1} = OpISub %uint {one} {self.cu(1)}")
            self.emit(f"{r} = OpShiftLeftLogical %uint {m1} {om}")
            self.dst(a[0], r)
        elif op == "s_mul_hi_u32":
            x, y = self.src_u(a[1]), self.src_u(a[2])
            s, r = self.tmp(), self.tmp()
            self.emit(f"{s} = OpUMulExtended %umul {x} {y}")
            self.emit(f"{r} = OpCompositeExtract %uint {s} 1")
            self.dst(a[0], r)
        elif op in ("v_fma_mixlo_f16", "v_fma_mixhi_f16"):
            sel = next((m for m in mods if m.startswith("op_sel:")), "op_sel:[0,0,0]")
            selhi = next((m for m in mods if m.startswith("op_sel_hi:")), "op_sel_hi:[0,0,0]")
            sb = [int(x) for x in sel.split("[")[1].rstrip("]").split(",")]
            hb = [int(x) for x in selhi.split("[")[1].rstrip("]").split(",")]
            vals = []
            for i, src in enumerate(a[1:4]):
                if hb[i]:
                    u = self.src_u({**src, "neg": False, "abs": False})
                    v2, f = self.tmp(), self.tmp()
                    self.emit(f"{v2} = OpExtInst %v2float %glsl UnpackHalf2x16 {u}")
                    self.emit(f"{f} = OpCompositeExtract %float {v2} {sb[i]}")
                    vals.append(self.modify_f(f, src))
                else:
                    vals.append(self.src_f(src))
            r, v, p, h16 = self.tmp(), self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{r} = OpExtInst %float %glsl Fma {vals[0]} {vals[1]} {vals[2]}")
            r = self.clamp_mods(r, mods)
            self.emit(f"{v} = OpCompositeConstruct %v2float {r} %f0")
            self.emit(f"{p} = OpExtInst %uint %glsl PackHalf2x16 {v}")
            self.emit(f"{h16} = OpBitwiseAnd %uint {p} {self.cu(0xFFFF)}")
            old = self.load_reg("vgpr", a[0]["r"])
            keep, newv = self.tmp(), self.tmp()
            if op == "v_fma_mixlo_f16":
                self.emit(f"{keep} = OpBitwiseAnd %uint {old} {self.cu(0xFFFF0000)}")
                self.emit(f"{newv} = OpBitwiseOr %uint {keep} {h16}")
            else:
                sh = self.tmp()
                self.emit(f"{keep} = OpBitwiseAnd %uint {old} {self.cu(0xFFFF)}")
                self.emit(f"{sh} = OpShiftLeftLogical %uint {h16} {self.cu(16)}")
                self.emit(f"{newv} = OpBitwiseOr %uint {keep} {sh}")
            self.dst(a[0], newv)
        elif op == "v_cvt_pk_u8_f32":
            # Convert src0 to u8 (round to nearest even, clamp) and insert it at byte src1 of src2.
            x, byte, base = self.src_f(a[1]), self.src_u(a[2]), self.src_u(a[3])
            c, rn, u, sh, mask, nmask, keep, ins_, r = (self.tmp() for _ in range(9))
            self.emit(f"{c} = OpExtInst %float %glsl FClamp {x} %f0 {self.fconst(255.0)}")
            self.emit(f"{rn} = OpExtInst %float %glsl RoundEven {c}")
            self.emit(f"{u} = OpConvertFToU %uint {rn}")
            b3 = self.tmp()
            self.emit(f"{b3} = OpBitwiseAnd %uint {byte} {self.cu(3)}")
            self.emit(f"{sh} = OpIMul %uint {b3} {self.cu(8)}")
            self.emit(f"{mask} = OpShiftLeftLogical %uint {self.cu(0xFF)} {sh}")
            self.emit(f"{nmask} = OpNot %uint {mask}")
            self.emit(f"{keep} = OpBitwiseAnd %uint {base} {nmask}")
            self.emit(f"{ins_} = OpShiftLeftLogical %uint {u} {sh}")
            self.emit(f"{r} = OpBitwiseOr %uint {keep} {ins_}")
            self.dst(a[0], r)
        elif op == "v_mul_hi_u32":
            x, y = self.src_u(a[1]), self.src_u(a[2])
            s, r = self.tmp(), self.tmp()
            self.emit(f"{s} = OpUMulExtended %umul {x} {y}")
            self.emit(f"{r} = OpCompositeExtract %uint {s} 1")
            self.dst(a[0], r)
        elif op == "v_cvt_pkrtz_f16_f32":
            # Packs two f32 as f16 into one dword with round-toward-zero (exact, see f16_rtz). The earlier
            # round-to-nearest version produced inf where hardware gives 65504 (found by the GPU smoke test).
            x, y = self.src_f(a[1]), self.src_f(a[2])
            lo, hi = self.f16_rtz(x), self.f16_rtz(y)
            sh, r = self.tmp(), self.tmp()
            self.emit(f"{sh} = OpShiftLeftLogical %uint {hi} {self.cu(16)}")
            self.emit(f"{r} = OpBitwiseOr %uint {lo} {sh}")
            self.dst(a[0], r)
        elif op == "v_rcp_iflag_f32":
            x = self.src_f(a[1])
            r = self.tmp()
            self.emit(f"{r} = OpFDiv %float %f1 {x}")
            self.dst(a[0], self.tou(r))
        elif op in F2:
            x, y = self.src_f(a[1]), self.src_f(a[2])
            r = self.tmp()
            if F2[op] in ("max", "min"):
                self.emit(f"{r} = OpExtInst %float %glsl {'FMax' if F2[op] == 'max' else 'FMin'} {x} {y}")
            else:
                self.emit(f"{r} = {F2[op]} %float {x} {y}")
            self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
        elif op == "v_subrev_f32":
            x, y = self.src_f(a[1]), self.src_f(a[2])
            r = self.tmp()
            self.emit(f"{r} = OpFSub %float {y} {x}")
            self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
        elif op in ("v_fma_f32", "v_fmac_f32", "v_mac_f32", "v_fmaak_f32", "v_fmamk_f32", "v_med3_f32"):
            if op == "v_fma_f32":
                x, y, z = (self.src_f(t) for t in a[1:4])
            elif op in ("v_fmac_f32", "v_mac_f32"):
                x, y, z = self.src_f(a[1]), self.src_f(a[2]), self.src_f(a[0])
            elif op == "v_fmaak_f32":
                x, y, z = self.src_f(a[1]), self.src_f(a[2]), self.src_f(a[3])
            elif op == "v_fmamk_f32":
                x, y, z = self.src_f(a[1]), self.src_f(a[3]), self.src_f(a[2])
            else:
                x, y, z = (self.src_f(t) for t in a[1:4])
                lo, hi, r = self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{lo} = OpExtInst %float %glsl FMin {x} {y}")
                self.emit(f"{hi} = OpExtInst %float %glsl FMax {x} {y}")
                self.emit(f"{r} = OpExtInst %float %glsl FClamp {z} {lo} {hi}")
                self.dst(a[0], self.tou(r))
                return
            r = self.tmp()
            self.emit(f"{r} = OpExtInst %float %glsl Fma {x} {y} {z}")
            self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
        elif op in F1:
            x = self.src_f(a[1])
            r = self.tmp()
            if F1[op] == "rcp":
                self.emit(f"{r} = OpFDiv %float %f1 {x}")
            else:
                self.emit(f"{r} = OpExtInst %float %glsl {F1[op]} {x}")
            self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
        elif op in ("v_cvt_f32_u32", "v_cvt_f32_i32"):
            x = self.src_u(a[1])
            r = self.tmp()
            self.emit(f"{r} = {'OpConvertUToF' if op.endswith('u32') else 'OpConvertSToF'} %float "
                      f"{x if op.endswith('u32') else self.as_int(x)}")
            self.dst(a[0], self.tou(r))
        elif op.startswith("v_cvt_f32_ubyte"):
            n = int(op[-1])
            x = self.src_u(a[1])
            s, m, r = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{s} = OpShiftRightLogical %uint {x} {self.cu(8 * n)}")
            self.emit(f"{m} = OpBitwiseAnd %uint {s} {self.cu(0xFF)}")
            self.emit(f"{r} = OpConvertUToF %float {m}")
            self.dst(a[0], self.tou(r))
        elif op in ("v_cvt_u32_f32", "v_cvt_i32_f32"):
            x = self.src_f(a[1])
            r = self.tmp()
            if op == "v_cvt_u32_f32":
                self.emit(f"{r} = OpConvertFToU %uint {x}")
            else:
                i = self.tmp()
                self.emit(f"{i} = OpConvertFToS %int {x}")
                self.emit(f"{r} = OpBitcast %uint {i}")
            self.dst(a[0], r)
        elif op in U2 or op in S2:
            x, y = self.src_u(a[1]), self.src_u(a[2])
            r = self.tmp()
            kind = U2.get(op) or S2.get(op)
            if kind == "mul24":
                xm, ym = self.tmp(), self.tmp()
                self.emit(f"{xm} = OpBitwiseAnd %uint {x} {self.cu(0xFFFFFF)}")
                self.emit(f"{ym} = OpBitwiseAnd %uint {y} {self.cu(0xFFFFFF)}")
                self.emit(f"{r} = OpIMul %uint {xm} {ym}")
            else:
                self.emit(f"{r} = {kind} %uint {x} {y}")
            self.dst(a[0], r)
        elif op in SHIFT_REV:
            sh, x = self.src_u(a[1]), self.src_u(a[2])
            m, r = self.tmp(), self.tmp()
            self.emit(f"{m} = OpBitwiseAnd %uint {sh} {self.cu(31)}")
            self.emit(f"{r} = {SHIFT_REV[op]} %uint {x} {m}")
            self.dst(a[0], r)
        elif op in ("v_bfe_u32", "v_bfe_i32"):
            # offset and width are taken mod 32 (a width of 0 gives 0)
            x, o, w = (self.src_u(t) for t in a[1:4])
            om, wm, r = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{om} = OpBitwiseAnd %uint {o} {self.cu(31)}")
            self.emit(f"{wm} = OpBitwiseAnd %uint {w} {self.cu(31)}")
            f = "OpBitFieldSExtract" if op.endswith("i32") else "OpBitFieldUExtract"
            self.emit(f"{r} = {f} %uint {x} {om} {wm}")
            self.dst(a[0], r)
        elif op == "v_mad_u32_u24":
            x, y, z = (self.src_u(t) for t in a[1:4])
            xm, ym, p, r = self.tmp(), self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{xm} = OpBitwiseAnd %uint {x} {self.cu(0xFFFFFF)}")
            self.emit(f"{ym} = OpBitwiseAnd %uint {y} {self.cu(0xFFFFFF)}")
            self.emit(f"{p} = OpIMul %uint {xm} {ym}")
            self.emit(f"{r} = OpIAdd %uint {p} {z}")
            self.dst(a[0], r)
        elif op == "v_cndmask_b32":
            # e32: dst, src0, src1 (vcc) ; e64: dst, src0, src1, sN[:N+1]/vcc
            x, y = self.src_u(a[1]), self.src_u(a[2])
            c = self.mask_src(a[3]) if len(a) > 3 else self.mask_src({"k": "special", "t": "vcc"})
            b, r = self.tmp(), self.tmp()
            self.emit(f"{b} = OpINotEqual %bool {c} {self.cu(0)}")
            self.emit(f"{r} = OpSelect %uint {b} {y} {x}")
            self.dst(a[0], r)
        elif op == "exp":
            self.export(ins)
        else:
            raise Unsupported(op)

    def scalar_load(self, op, a, mods):
        """s_load_dwordxN sD, s[p:p+1], off   -> N dwords at (64-bit pointer + off)
        s_buffer_load_dwordxN sD, s[v:v+3], off -> N dwords at (V# base address + off)
        Both become PhysicalStorageBuffer loads (Vulkan buffer device address). V# base = dword0 | (dword1 & 0xFFFF) << 32
        (AMD RDNA ISA, buffer resource descriptor)."""
        n = 1 if op.endswith("dword") else int(op.rsplit("x", 1)[1])
        d, base = a[0], a[1]
        # A single dword may also land in vcc_lo / m0 / vcc_hi (seen: s_buffer_load_dword vcc_lo, ...).
        single_special = n == 1 and d["k"] == "special" and d["t"] in ("vcc_lo", "vcc", "m0", "vcc_hi")
        if (d["k"] != "sgpr" and not single_special) or base["k"] != "sgpr":
            raise Unsupported(f"{op} operands {d.get('t', d['k'])} {base.get('t', base['k'])}")
        lo, hi = self.load_reg("sgpr", base["r"]), self.load_reg("sgpr", base["r"] + 1)
        if op.startswith("s_buffer"):
            if base.get("n") != 4:
                raise Unsupported("s_buffer_load without V#")
            m = self.tmp()
            self.emit(f"{m} = OpBitwiseAnd %uint {hi} {self.cu(0xFFFF)}")
            hi = m
        else:
            if base.get("n") != 2:
                raise Unsupported("s_load without 64-bit pointer")
        off = a[2] if len(a) > 2 else {"k": "imm", "v": 0}
        offv = self.src_u(off)
        l64, h64, sh, addr, o64, full = (self.tmp() for _ in range(6))
        self.emit(f"{l64} = OpUConvert %ulong {lo}")
        self.emit(f"{h64} = OpUConvert %ulong {hi}")
        self.emit(f"{sh} = OpShiftLeftLogical %ulong {h64} %c64_32")
        self.emit(f"{addr} = OpBitwiseOr %ulong {sh} {l64}")
        self.emit(f"{o64} = OpUConvert %ulong {offv}")
        self.emit(f"{full} = OpIAdd %ulong {addr} {o64}")
        ptr = self.tmp()
        self.emit(f"{ptr} = OpConvertUToPtr %_ptr_PSB_arr {self.host_addr(full)}")
        self.uses_psb = True
        for i in range(n):
            p, v = self.tmp(), self.tmp()
            self.emit(f"{p} = OpAccessChain %_ptr_PSB_uint {ptr} {self.cu(i)}")
            self.emit(f"{v} = OpLoad %uint {p} Aligned 4")
            if single_special:
                self.dst(d, v)
            else:
                self.store_reg("sgpr", d["r"] + i, v)

    # ---- images: T#/S# -> bindless index through a runtime-filled hash table ----
    # Table layout (storage buffer, set 0 binding 2): uint mask; then entries of 10 uints:
    #   key[0..7] (descriptor dwords; S# uses key[0..3] and key[4..7] = 0xFFFFFFFF), index, pad.
    # Probes 8 slots from hash(key) & mask; a miss yields index 0 (the runtime puts a fallback there).
    PROBES = 8

    def desc_lookup(self, words):
        """words: 8 uint ids. Returns a uint id with the bindless index."""
        self.uses_tex = True
        h = self.cu(2166136261)
        for w in words:                      # FNV-1a over the 8 dwords
            x, m = self.tmp(), self.tmp()
            self.emit(f"{x} = OpBitwiseXor %uint {h} {w}")
            self.emit(f"{m} = OpIMul %uint {x} {self.cu(16777619)}")
            h = m
        mp, mask = self.tmp(), self.tmp()
        self.emit(f"{mp} = OpAccessChain %_ptr_SB_uint %desc_map %c0 %c0")
        self.emit(f"{mask} = OpLoad %uint {mp}")
        found, idx = "%false", self.cu(0)
        for k in range(self.PROBES):
            s, slot, base = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{s} = OpIAdd %uint {h} {self.cu(k)}")
            self.emit(f"{slot} = OpBitwiseAnd %uint {s} {mask}")
            self.emit(f"{base} = OpIMul %uint {slot} {self.cu(10)}")
            match = "%true"
            for j, w in enumerate(words):
                o, p, v, e, m2 = self.tmp(), self.tmp(), self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{o} = OpIAdd %uint {base} {self.cu(1 + j)}")
                self.emit(f"{p} = OpAccessChain %_ptr_SB_uint %desc_map %c0 {o}")
                self.emit(f"{v} = OpLoad %uint {p}")
                self.emit(f"{e} = OpIEqual %bool {v} {w}")
                self.emit(f"{m2} = OpLogicalAnd %bool {match} {e}")
                match = m2
            o, p, v = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{o} = OpIAdd %uint {base} {self.cu(9)}")
            self.emit(f"{p} = OpAccessChain %_ptr_SB_uint %desc_map %c0 {o}")
            self.emit(f"{v} = OpLoad %uint {p}")
            nf, take, nidx, nfound = self.tmp(), self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{nf} = OpLogicalNot %bool {found}")
            self.emit(f"{take} = OpLogicalAnd %bool {nf} {match}")
            self.emit(f"{nidx} = OpSelect %uint {take} {v} {idx}")
            self.emit(f"{nfound} = OpLogicalOr %bool {found} {match}")
            idx, found = nidx, nfound
        return idx

    def sgpr_words(self, a, n):
        if a["k"] != "sgpr" or a.get("n", 1) < n:
            raise Unsupported("descriptor operand")
        return [self.load_reg("sgpr", a["r"] + i) for i in range(n)]

    def addr_regs(self, a):
        """Image address operand: contiguous v[x:y] or an NSA list."""
        if a["k"] == "vgpr":
            return [a["r"] + i for i in range(a.get("n", 1))]
        if a["k"] == "vlist":
            return a["r"]
        raise Unsupported("image address operand")

    def image_op(self, op, a, mods):
        a16 = "a16" in mods
        d16 = "d16" in mods
        dim = next((m for m in mods if m.startswith("dim:")), "dim:SQ_RSRC_IMG_2D")
        kinds = {"dim:SQ_RSRC_IMG_2D": ("img2d", "%images", "simg2d", 2, "_ptr_UC_img"),
                 "dim:SQ_RSRC_IMG_2D_ARRAY": ("img2da", "%images_arr", "simg2da", 3, "_ptr_UC_imga"),
                 # Cube: the shader already did face selection (v_cube*), so (s, t, face) samples a 6-layer
                 # 2D array view of the cube. The runtime binds cube textures as 2D-array views.
                 "dim:SQ_RSRC_IMG_CUBE": ("img2da", "%images_arr", "simg2da", 3, "_ptr_UC_imga"),
                 "dim:SQ_RSRC_IMG_1D": ("img1d", "%images_1d", "simg1d", 1, "_ptr_UC_img1"),
                 "dim:SQ_RSRC_IMG_3D": ("img3d", "%images_3d", "simg3d", -3, "_ptr_UC_img3")}
        if dim not in kinds:
            raise Unsupported(op + " " + dim.split("SQ_RSRC_IMG_")[-1])
        itype, iarr, stype, ncoord, iptr = kinds[dim]
        is3d = ncoord == -3          # 3D: three real coordinates (and 3-component gradients)
        if is3d:
            ncoord = 3
            self.uses_tex_3d = True
        elif ncoord == 3:
            self.uses_tex_array = True
        if ncoord == 1:
            self.uses_tex_1d = True
        if is3d and op.endswith("_d"):
            raise Unsupported(op + " 3D gradients")
        dm = next((m for m in mods if m.startswith("dmask:")), "dmask:0x1")
        dmask = int(dm.split(":")[1], 0)
        vdata, addr = a[0], self.addr_regs(a[1])
        t = self.sgpr_words(a[2], 8)
        img_idx = self.desc_lookup(t)
        ip, img = self.tmp(), self.tmp()
        self.emit(f"{ip} = OpAccessChain %{iptr} {iarr} {img_idx}")
        self.emit(f"{img} = OpLoad %{itype} {ip}")
        coords = [self.load_reg("vgpr", r) for r in addr]
        if a16:
            # 16-bit addresses: each register packs two f16 coordinates (low half first).
            unpacked = []
            for c in coords:
                v2 = self.tmp()
                self.emit(f"{v2} = OpExtInst %v2float %glsl UnpackHalf2x16 {c}")
                for k in (0, 1):
                    f = self.tmp()
                    self.emit(f"{f} = OpCompositeExtract %float {v2} {k}")
                    unpacked.append(self.tou(f))
            coords = unpacked
            if op == "image_load":
                # 16-bit integer coordinates: each half is an unsigned 16-bit int, not an f16.
                ints = []
                for c in [self.load_reg("vgpr", r) for r in addr]:
                    lo, hi = self.tmp(), self.tmp()
                    self.emit(f"{lo} = OpBitwiseAnd %uint {c} {self.cu(0xFFFF)}")
                    self.emit(f"{hi} = OpShiftRightLogical %uint {c} {self.cu(16)}")
                    ints += [lo, hi]
                coords = ints
        if op == "image_load":
            if len(coords) < ncoord:
                raise Unsupported("image_load coords")
            ints = [self.as_int(c) for c in coords[:ncoord]]
            if ncoord == 1:
                iv = ints[0]
            else:
                iv = self.tmp()
                self.emit(f"{iv} = OpCompositeConstruct {'%v2int' if ncoord == 2 else '%v3int'} {' '.join(ints)}")
            res = self.tmp()
            self.emit(f"{res} = OpImageFetch %v4float {img} {iv} Lod {self.cu(0)}")
            ncomp = bin(dmask).count("1")
        else:
            s = self.sgpr_words(a[3], 4)
            smp_idx = self.desc_lookup(s + [self.cu(0xFFFFFFFF)] * 4)
            sp, smp, si = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{sp} = OpAccessChain %_ptr_UC_smp %samplers {smp_idx}")
            self.emit(f"{smp} = OpLoad %smp {sp}")
            self.emit(f"{si} = OpSampledImage %{stype} {img} {smp}")
            f = [self.tof(c) for c in coords]
            if op == "image_gather4_lz":
                # Gather the 2x2 footprint of one channel (dmask picks it) at mip 0 -> 4 results.
                if bin(dmask).count("1") != 1 or ncoord not in (2, 3) or is3d or len(f) < ncoord:
                    raise Unsupported("image_gather4_lz form")
                uv, res = self.tmp(), self.tmp()
                if ncoord == 3:          # 2D array (or cube as a 6-layer array): layer is the 3rd coordinate
                    self.emit(f"{uv} = OpCompositeConstruct %v3float {f[0]} {f[1]} {f[2]}")
                else:
                    self.emit(f"{uv} = OpCompositeConstruct %v2float {f[0]} {f[1]}")
                self.emit(f"{res} = OpImageGather %v4float {si} {uv} {self.cu(dmask.bit_length() - 1)}")
                comps = []
                for c in range(4):
                    comp = self.tmp()
                    self.emit(f"{comp} = OpCompositeExtract %float {res} {c}")
                    comps.append(comp)
                if d16:
                    for i in range(0, 4, 2):
                        v, p = self.tmp(), self.tmp()
                        self.emit(f"{v} = OpCompositeConstruct %v2float {comps[i]} {comps[i + 1]}")
                        self.emit(f"{p} = OpExtInst %uint %glsl PackHalf2x16 {v}")
                        self.dst({"k": "vgpr", "r": vdata["r"] + i // 2, "n": 1}, p)
                else:
                    for i, comp in enumerate(comps):
                        self.dst({"k": "vgpr", "r": vdata["r"] + i, "n": 1}, self.tou(comp))
                return
            kind = op[len("image_sample"):]          # '', '_lz', '_l', '_b', '_c_lz', '_d', '_d_cl' (+ '_o')
            off = None
            if kind.endswith("_o"):
                # _o: the first address dword packs signed 6-bit texel offsets, x = [5:0], y = [13:8], z = [21:16]
                # (GFX10 MIMG address order: offset, bias, z-compare, derivatives, coordinates). Mapped to the
                # SPIR-V Offset operand; Vulkan only guarantees offsets within minTexelOffset..maxTexelOffset.
                if a16:
                    raise Unsupported(op + " a16")
                kind = kind[:-2]
                raw, f = coords[0], f[1:]
                n_off = 3 if is3d else (1 if ncoord == 1 else 2)
                parts = []
                for k in range(n_off):
                    u, i = self.tmp(), self.tmp()
                    self.emit(f"{u} = OpBitFieldSExtract %uint {raw} {self.cu(8 * k)} {self.cu(6)}")
                    self.emit(f"{i} = OpBitcast %int {u}")
                    parts.append(i)
                if n_off == 1:
                    off = parts[0]
                else:
                    off = self.tmp()
                    self.emit(f"{off} = OpCompositeConstruct {'%v3int' if n_off == 3 else '%v2int'} {' '.join(parts)}")
                self.uses_img_offset = True
            minlod = None
            if kind == "_d_cl":
                # Same as _d plus a trailing min-LOD clamp coordinate.
                minlod, f, kind = f[-1], f[:-1], "_d"
                self.uses_minlod = True
            ref = None
            if kind.startswith("_c"):
                ref, f = f[0], f[1:]
            bias = None
            if kind in ("_b", "_c_b"):
                bias, f = f[0], f[1:]
            grads = None
            if kind == "_d":
                # Address: gradients first, then coords. 2D and 2D-array have 2-component gradients (the array
                # layer has none): ds/dx, dt/dx, ds/dy, dt/dy. 1D has scalar ds/dx, ds/dy.
                if ncoord == 1:
                    if len(f) < 3:
                        raise Unsupported(op + " coords")
                    grads, f = (f[0], f[1]), f[2:]
                else:
                    if len(f) < 4 + ncoord:
                        raise Unsupported(op + " coords")
                    gx, gy = self.tmp(), self.tmp()
                    self.emit(f"{gx} = OpCompositeConstruct %v2float {f[0]} {f[1]}")
                    self.emit(f"{gy} = OpCompositeConstruct %v2float {f[2]} {f[3]}")
                    grads, f = (gx, gy), f[4:]
            if len(f) < ncoord:
                raise Unsupported(op + " coords")
            if ncoord == 1:
                uv = f[0]
            elif ncoord == 3:
                # The array layer arrives as a float coordinate; Vulkan rounds it the same way.
                uv = self.tmp()
                self.emit(f"{uv} = OpCompositeConstruct %v3float {f[0]} {f[1]} {f[2]}")
            else:
                uv = self.tmp()
                self.emit(f"{uv} = OpCompositeConstruct %v2float {f[0]} {f[1]}")
            res = self.tmp()
            if off is not None and kind not in ("", "_lz", "_l", "_b", "_c_lz"):
                raise Unsupported(op)

            def opnd(mask, args):
                """Image operands with the texel offset appended (Offset = 0x10 sorts after Lod/Bias)."""
                if off is None:
                    return f" {mask} {args}" if mask else ""
                return f" {mask + '|' if mask else ''}Offset {args + ' ' if args else ''}{off}"
            if kind == "_d":
                if minlod is not None:
                    self.emit(f"{res} = OpImageSampleExplicitLod %v4float {si} {uv} Grad|MinLod {grads[0]} {grads[1]} {minlod}")
                else:
                    self.emit(f"{res} = OpImageSampleExplicitLod %v4float {si} {uv} Grad {grads[0]} {grads[1]}")
            elif kind == "":
                self.emit(f"{res} = OpImageSampleImplicitLod %v4float {si} {uv}{opnd('', '')}" if self.stage == "ps" else
                          f"{res} = OpImageSampleExplicitLod %v4float {si} {uv}{opnd('Lod', '%f0')}")
            elif kind == "_lz":
                self.emit(f"{res} = OpImageSampleExplicitLod %v4float {si} {uv}{opnd('Lod', '%f0')}")
            elif kind == "_l":
                self.emit(f"{res} = OpImageSampleExplicitLod %v4float {si} {uv}{opnd('Lod', f[ncoord] if len(f) > ncoord else '%f0')}")
            elif kind == "_b" and self.stage == "ps":
                self.emit(f"{res} = OpImageSampleImplicitLod %v4float {si} {uv}{opnd('Bias', bias)}")
            elif kind == "_c_lz":
                d = self.tmp()
                self.emit(f"{d} = OpImageSampleDrefExplicitLod %float {si} {uv} {ref}{opnd('Lod', '%f0')}")
                self.dst({"k": "vgpr", "r": vdata["r"], "n": 1}, self.tou(d))
                return
            else:
                raise Unsupported(op)
            ncomp = bin(dmask).count("1")
        comps = []
        for c in range(4):
            if dmask >> c & 1:
                comp = self.tmp()
                self.emit(f"{comp} = OpCompositeExtract %float {res} {c}")
                comps.append(comp)
        if len(comps) != ncomp:
            raise Unsupported("dmask")
        if d16:
            # d16: components are packed two per register as f16 (x|y<<16, z|w<<16).
            for i in range(0, len(comps), 2):
                second = comps[i + 1] if i + 1 < len(comps) else "%f0"
                v, p = self.tmp(), self.tmp()
                self.emit(f"{v} = OpCompositeConstruct %v2float {comps[i]} {second}")
                self.emit(f"{p} = OpExtInst %uint %glsl PackHalf2x16 {v}")
                self.dst({"k": "vgpr", "r": vdata["r"] + i // 2, "n": 1}, p)
        else:
            for i, comp in enumerate(comps):
                self.dst({"k": "vgpr", "r": vdata["r"] + i, "n": 1}, self.tou(comp))

    # image atomic -> (SPIR-V op, value that leaves memory unchanged, used for inactive lanes)
    IMG_ATOMICS = {"image_atomic_umin": ("OpAtomicUMin", 0xFFFFFFFF), "image_atomic_umax": ("OpAtomicUMax", 0),
                   "image_atomic_smin": ("OpAtomicSMin", 0x7FFFFFFF), "image_atomic_smax": ("OpAtomicSMax", 0x80000000),
                   "image_atomic_or": ("OpAtomicOr", 0), "image_atomic_xor": ("OpAtomicXor", 0),
                   "image_atomic_and": ("OpAtomicAnd", 0xFFFFFFFF), "image_atomic_add": ("OpAtomicIAdd", 0),
                   "image_atomic_sub": ("OpAtomicISub", 0)}

    # image_store views: dim -> (index, SPIR-V Dim, arrayed, coordinate count). Bindings 8 + 3 * index + class,
    # class 0 = float, 1 = uint, 2 = sint (Vulkan needs the shader's component type to match the image format).
    STORE_DIMS = {"dim:SQ_RSRC_IMG_2D": (0, "2D", 0, 2), "dim:SQ_RSRC_IMG_2D_ARRAY": (1, "2D", 1, 3),
                  "dim:SQ_RSRC_IMG_3D": (2, "3D", 0, 3)}
    INT_FORMATS = None          # (uint mask words, sint mask words) over the 512 GFX10 formats

    @classmethod
    def int_format_masks(cls):
        if cls.INT_FORMATS is None:
            u, si = [0] * 16, [0] * 16
            for k, name in GFX10_FORMATS.items():
                if name.endswith("_UINT"):
                    u[k >> 5] |= 1 << (k & 31)
                elif name.endswith("_SINT"):
                    si[k >> 5] |= 1 << (k & 31)
            cls.INT_FORMATS = (u, si)
        return cls.INT_FORMATS

    def image_store(self, a, mods, dim):
        """image_store vData, vAddr, T#: write the dmask components to a storage image. The view is picked at run
        time from T# word1 FORMAT [28:20]: *_UINT -> uint view, *_SINT -> sint view, anything else -> float view
        (UNORM/SNORM/FLOAT convert from the float bits the shader wrote). Missing components are 0."""
        if dim not in self.STORE_DIMS or "a16" in mods or "d16" in mods:
            raise Unsupported("image_store " + dim.split("SQ_RSRC_IMG_")[-1] + (" a16" if "a16" in mods else "")
                              + (" d16" if "d16" in mods else ""))
        di, _, _, nc = self.STORE_DIMS[dim]
        dm = next((m for m in mods if m.startswith("dmask:")), "dmask:0xf")
        ncomp = bin(int(dm.split(":")[1], 0)).count("1")
        vdata, addr = a[0], self.addr_regs(a[1])
        if len(addr) < nc:
            raise Unsupported("image_store coords")
        vals = [self.load_reg("vgpr", vdata["r"] + i) for i in range(ncomp)] + [self.cu(0)] * (4 - ncomp)
        coords = [self.as_int(self.load_reg("vgpr", r)) for r in addr[:nc]]
        t = self.sgpr_words(a[2], 8)
        idx = self.desc_lookup(t)
        self.uses_store_views.add(di)
        for w in [16] + sum(self.int_format_masks(), []):      # constants used by assemble's format masks
            self.cu(w)
        iv = self.tmp()
        self.emit(f"{iv} = OpCompositeConstruct {'%v2int' if nc == 2 else '%v3int'} {' '.join(coords)}")
        fmt, word, bit = self.tmp(), self.tmp(), self.tmp()
        self.emit(f"{fmt} = OpBitFieldUExtract %uint {t[1]} {self.cu(20)} {self.cu(9)}")
        self.emit(f"{word} = OpShiftRightLogical %uint {fmt} {self.cu(5)}")
        self.emit(f"{bit} = OpBitwiseAnd %uint {fmt} {self.cu(31)}")
        flags = []
        for mask in ("%fmt_uint_mask", "%fmt_sint_mask"):
            p, w, b, nz = self.tmp(), self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{p} = OpAccessChain %_ptr_Private_uint {mask} {word}")
            self.emit(f"{w} = OpLoad %uint {p}")
            self.emit(f"{b} = OpBitFieldUExtract %uint {w} {bit} {self.cu(1)}")
            self.emit(f"{nz} = OpINotEqual %bool {b} {self.cu(0)}")
            flags.append(nz)
        is_u, is_s = flags
        fv = [self.tof(v) for v in vals]
        sv = [self.as_int(v) for v in vals]
        v4f, v4u, v4s = self.tmp(), self.tmp(), self.tmp()
        self.emit(f"{v4f} = OpCompositeConstruct %v4float {' '.join(fv)}")
        self.emit(f"{v4u} = OpCompositeConstruct %v4uint_st {' '.join(vals)}")
        self.emit(f"{v4s} = OpCompositeConstruct %v4int_st {' '.join(sv)}")
        act = self.active_b()
        L = {k: self.tmp() + "_" + k for k in ("do", "end", "s", "rest", "e2", "u", "f", "e3")}

        def write(cls, texel):
            ip, img = self.tmp(), self.tmp()
            self.emit(f"{ip} = OpAccessChain %_ptr_UC_stv{cls}_{di} %stv{cls}_{di} {idx}")
            self.emit(f"{img} = OpLoad %stimg{cls}_{di} {ip}")
            self.emit(f"OpImageWrite {img} {iv} {texel}")

        self.emit(f"OpSelectionMerge {L['end']} None")
        self.emit(f"OpBranchConditional {act} {L['do']} {L['end']}")
        self.emit(f"{L['do']} = OpLabel")
        self.emit(f"OpSelectionMerge {L['e2']} None")
        self.emit(f"OpBranchConditional {is_s} {L['s']} {L['rest']}")
        self.emit(f"{L['s']} = OpLabel")
        write(2, v4s)
        self.emit(f"OpBranch {L['e2']}")
        self.emit(f"{L['rest']} = OpLabel")
        self.emit(f"OpSelectionMerge {L['e3']} None")
        self.emit(f"OpBranchConditional {is_u} {L['u']} {L['f']}")
        self.emit(f"{L['u']} = OpLabel")
        write(1, v4u)
        self.emit(f"OpBranch {L['e3']}")
        self.emit(f"{L['f']} = OpLabel")
        write(0, v4f)
        self.emit(f"OpBranch {L['e3']}")
        self.emit(f"{L['e3']} = OpLabel")
        self.emit(f"OpBranch {L['e2']}")
        self.emit(f"{L['e2']} = OpLabel")
        self.emit(f"OpBranch {L['end']}")
        self.emit(f"{L['end']} = OpLabel")

    def image_misc(self, op, a, mods):
        """image_get_resinfo / image_load_mip on 2D sampled images, image_atomic_umin on 2D r32ui storage images."""
        dim = next((m for m in mods if m.startswith("dim:")), "dim:SQ_RSRC_IMG_2D")
        is3d_atomic = dim == "dim:SQ_RSRC_IMG_3D" and op in self.IMG_ATOMICS
        if op == "image_store":
            self.image_store(a, mods, dim)
            return
        if (dim != "dim:SQ_RSRC_IMG_2D" and not is3d_atomic) or "d16" in mods or "a16" in mods:
            raise Unsupported(op + " " + dim.split("SQ_RSRC_IMG_")[-1])
        vdata, addr = a[0], self.addr_regs(a[1])
        t = self.sgpr_words(a[2], 8)
        idx = self.desc_lookup(t)
        if is3d_atomic:
            # 3D r32ui storage image atomics (binding 7).
            self.uses_storage3d = True
            coords = [self.as_int(self.load_reg("vgpr", r)) for r in addr[:3]]
            if len(coords) < 3:
                raise Unsupported(op + " coords")
            iv, ip, tp, old, safe = (self.tmp() for _ in range(5))
            self.emit(f"{iv} = OpCompositeConstruct %v3int {coords[0]} {coords[1]} {coords[2]}")
            self.emit(f"{ip} = OpAccessChain %_ptr_UC_simg3 %simages3d {idx}")
            val = self.load_reg("vgpr", vdata["r"])
            spv, neutral = self.IMG_ATOMICS[op]
            self.emit(f"{tp} = OpImageTexelPointer %_ptr_Image_uint {ip} {iv} {self.cu(0)}")
            self.emit(f"{safe} = OpSelect %uint {self.active_b()} {val} {self.cu(neutral)}")
            self.emit(f"{old} = {spv} %uint {tp} {self.cu(1)} {self.cu(0)} {safe}")
            if "glc" in mods:
                self.dst({"k": "vgpr", "r": vdata["r"], "n": 1}, old)
            return
        if op == "image_store":
            self.image_store(a, mods, dim)
            return
        if op in self.IMG_ATOMICS:
            # Storage image (2D r32ui). Atomics: vdata holds the operand; with 'glc' the pre-op value comes back.
            self.uses_storage = True
            coords = [self.as_int(self.load_reg("vgpr", r)) for r in addr[:2]]
            iv = self.tmp()
            self.emit(f"{iv} = OpCompositeConstruct %v2int {coords[0]} {coords[1]}")
            ip = self.tmp()
            self.emit(f"{ip} = OpAccessChain %_ptr_UC_simg %simages {idx}")
            val = self.load_reg("vgpr", vdata["r"])
            if op == "image_store":
                # Writes dmask component 0 as r32ui; only active lanes store.
                img, lbl_do, lbl_end = self.tmp(), self.tmp() + "_do", self.tmp() + "_end"
                self.emit(f"{img} = OpLoad %simg2d_u {ip}")
                v4 = self.tmp()
                self.emit(f"{v4} = OpCompositeConstruct %v4uint {val} {self.cu(0)} {self.cu(0)} {self.cu(0)}")
                act = self.active_b()      # must be computed before OpSelectionMerge (merge directly precedes branch)
                self.emit(f"OpSelectionMerge {lbl_end} None")
                self.emit(f"OpBranchConditional {act} {lbl_do} {lbl_end}")
                self.emit(f"{lbl_do} = OpLabel")
                self.emit(f"OpImageWrite {img} {iv} {v4}")
                self.emit(f"OpBranch {lbl_end}")
                self.emit(f"{lbl_end} = OpLabel")
                self.uses_v4uint = True
                return
            spv, neutral = self.IMG_ATOMICS[op]
            tp, old, safe = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{tp} = OpImageTexelPointer %_ptr_Image_uint {ip} {iv} {self.cu(0)}")
            self.emit(f"{safe} = OpSelect %uint {self.active_b()} {val} {self.cu(neutral)}")
            self.emit(f"{old} = {spv} %uint {tp} {self.cu(1)} {self.cu(0)} {safe}")
            if "glc" in mods:
                self.dst({"k": "vgpr", "r": vdata["r"], "n": 1}, old)
            return
        ip, img = self.tmp(), self.tmp()
        self.emit(f"{ip} = OpAccessChain %_ptr_UC_img %images {idx}")
        self.emit(f"{img} = OpLoad %img2d {ip}")
        dm = next((m for m in mods if m.startswith("dmask:")), "dmask:0x1")
        dmask = int(dm.split(":")[1], 0)
        self.uses_query = True
        if op == "image_get_resinfo":
            lod = self.as_int(self.load_reg("vgpr", addr[0]))
            sz, lv = self.tmp(), self.tmp()
            self.emit(f"{sz} = OpImageQuerySizeLod %v2int {img} {lod}")
            self.emit(f"{lv} = OpImageQueryLevels %int {img}")
            w, h = self.tmp(), self.tmp()
            self.emit(f"{w} = OpCompositeExtract %int {sz} 0")
            self.emit(f"{h} = OpCompositeExtract %int {sz} 1")
            vals = [w, h, self.as_int(self.cu(1)), lv]          # x=width, y=height, z=depth(1), w=mip levels
            out = 0
            for c in range(4):
                if dmask >> c & 1:
                    u = self.tmp()
                    self.emit(f"{u} = OpBitcast %uint {vals[c]}")
                    self.dst({"k": "vgpr", "r": vdata["r"] + out, "n": 1}, u)
                    out += 1
            return
        # image_load_mip: x, y, mip
        if len(addr) < 3:
            raise Unsupported("image_load_mip coords")
        xs = [self.as_int(self.load_reg("vgpr", r)) for r in addr[:3]]
        iv, res = self.tmp(), self.tmp()
        self.emit(f"{iv} = OpCompositeConstruct %v2int {xs[0]} {xs[1]}")
        self.emit(f"{res} = OpImageFetch %v4float {img} {iv} Lod {xs[2]}")
        out = 0
        for c in range(4):
            if dmask >> c & 1:
                comp = self.tmp()
                self.emit(f"{comp} = OpCompositeExtract %float {res} {c}")
                self.dst({"k": "vgpr", "r": vdata["r"] + out, "n": 1}, self.tou(comp))
                out += 1

    def attribute(self, n, comp):
        """Component `comp` of fragment input attribute n (declared as a vec4 Input at Location n)."""
        name = f"%in_attr{n}"
        self.inputs[name] = n
        p, v = self.tmp(), self.tmp()
        self.emit(f"{p} = OpAccessChain %_ptr_Input_float {name} {self.cu(comp)}")
        self.emit(f"{v} = OpLoad %float {p}")
        return v

    def as_int(self, u):
        i = self.tmp()
        self.emit(f"{i} = OpBitcast %int {u}")
        return i

    def export(self, ins):
        a = ins["args"]
        tgt = a[0]["t"] if a and a[0]["k"] in ("target", "special") else None
        if tgt is None or tgt == "null":
            return
        comps = []
        if "compr" in ins["mods"]:
            # Compressed export: two registers, each holding two f16 values (x,y) and (z,w). LLVM lists each register
            # twice (one slot per half): 'exp mrt0 v4, v4, v5, v5 compr', so the pair is slots 1 and 3.
            # (Found by the GPU texture test: taking the first two operands exported (r,g,r,g).)
            srcs = a[1:5]
            regs = [srcs[0], srcs[2]] if len(srcs) >= 3 else srcs[:1]
            for x in regs:
                if x["k"] == "special" and x["t"] == "off":
                    comps += ["%f0", "%f0"]
                    continue
                u = self.src_u(x)
                v2 = self.tmp()
                self.emit(f"{v2} = OpExtInst %v2float %glsl UnpackHalf2x16 {u}")
                for c in (0, 1):
                    f = self.tmp()
                    self.emit(f"{f} = OpCompositeExtract %float {v2} {c}")
                    comps.append(f)
        else:
            for x in a[1:5]:
                if x["k"] == "special" and x["t"] == "off":
                    comps.append("%f0")
                else:
                    comps.append(self.src_f(x))
        while len(comps) < 4:
            comps.append("%f0")
        comps = comps[:4]
        v = self.tmp()
        self.emit(f"{v} = OpCompositeConstruct %v4float {' '.join(comps)}")
        name = {"pos0": "%out_pos"}.get(tgt)
        if tgt == "mrtz" and self.stage == "ps":
            # Depth export: component x is the fragment depth (stencil/mask exports aren't modelled).
            self.writes_depth = True
            old, sel = self.tmp(), self.tmp()
            self.emit(f"{old} = OpLoad %float %out_depth")
            self.emit(f"{sel} = OpSelect %float {self.active_b()} {comps[0]} {old}")
            self.emit(f"OpStore %out_depth {sel}")
            return
        if name is None:
            if tgt.startswith("param") and self.stage == "vs":
                name = f"%out_param{int(tgt[5:])}"
            elif tgt.startswith("mrt") and tgt != "mrtz" and self.stage == "ps":
                name = f"%out_mrt{int(tgt[3:])}"
            elif tgt.startswith("pos"):
                return  # pos1+ (clip/cull/point size): ignored in v0
            else:
                raise Unsupported(f"export {tgt}")
        self.outputs[name] = tgt
        # Exports from inactive lanes don't happen: keep the previous output value.
        old, sel = self.tmp(), self.tmp()
        self.emit(f"{old} = OpLoad %v4float {name}")
        self.emit(f"{sel} = OpSelect %v4float {self.active_b()} {v} {old}")
        self.emit(f"OpStore {name} {sel}")


_TYPE_DECL = re.compile(r"\s*(%\w+) = (OpTypeVector|OpTypePointer) (.*)$")
_ID_REF = re.compile(r"%\w+")


def dedupe_types(text):
    """SPIR-V forbids duplicate non-aggregate types. Separate features may each declare e.g. a uint4 vector
    (%v4uint for r32ui storage, %v4uint_st for typed stores, %v4uint_in for vertex inputs): keep the first
    declaration of each OpTypeVector / OpTypePointer and rename the others everywhere."""
    alias, seen, out = {}, {}, []
    for line in text.split("\n"):
        m = _TYPE_DECL.match(line)
        if m:
            name, opc, rest = m.groups()
            key = (opc, _ID_REF.sub(lambda t: alias.get(t.group(0), t.group(0)), rest))
            if key in seen:
                alias[name] = seen[key]
                continue
            seen[key] = name
        out.append(line)
    if not alias:
        return text
    return "\n".join(_ID_REF.sub(lambda t: alias.get(t.group(0), t.group(0)), l) for l in out)


def assemble(gen):
    vs = gen.stage == "vs"
    cs = gen.stage == "cs"
    model = "Vertex" if vs else "GLCompute" if cs else "Fragment"
    iface = list(gen.outputs) + list(gen.inputs) + (["%vertex_index"] if vs else []) +         (["%wg_id", "%local_id"] if cs else []) + (["%lds"] if gen.uses_lds else []) +         ((["%fmt_uint_mask", "%fmt_sint_mask"] +
          [f"%stv{c}_{d}" for d in sorted(gen.uses_store_views) for c in range(3)]) if gen.uses_store_views else [])
    psb = gen.uses_psb
    tex = gen.uses_tex
    sg = gen.uses_subgroup
    if tex:
        iface += ["%images", "%samplers", "%desc_map"] + (["%images_arr"] if gen.uses_tex_array else []) + \
                 (["%simages"] if gen.uses_storage else []) + (["%images_1d"] if gen.uses_tex_1d else []) + \
                 (["%images_3d"] if gen.uses_tex_3d else []) + (["%simages3d"] if gen.uses_storage3d else [])
    if sg:
        iface += ["%sg_lane"]
    if gen.writes_depth:
        iface += ["%out_depth"]
    iface += [f"%vin{reg}" for reg, _ in gen.vertex_inputs] + ["%user_data"]
    bary = any(str(c).startswith("bary") for c in gen.frag_inputs.values())
    if gen.frag_inputs:
        iface += ["%frag_coord"] + (["%front_facing"] if "front" in gen.frag_inputs.values() else []) +             (["%bary"] if bary else [])
    for i in range(USER_SGPRS):          # index constants for the user-data copy at entry
        gen.cu(i)
    # SPIR-V module layout: every OpCapability, then every OpExtension, then imports.
    caps = ["Shader"] + (["Int64", "PhysicalStorageBufferAddresses"] if psb else []) + \
        (["RuntimeDescriptorArray"] if tex else []) + (["GroupNonUniform", "GroupNonUniformShuffle"] if sg else []) + \
        (["ImageQuery"] if gen.uses_query else []) + (["Sampled1D"] if gen.uses_tex_1d else []) + \
        (["MinLod"] if gen.uses_minlod else []) + (["ImageGatherExtended"] if gen.uses_img_offset else []) +         (["FragmentBarycentricKHR"] if bary else [])
    if gen.uses_vote:
        caps += [c for c in ("GroupNonUniform", "GroupNonUniformVote") if c not in caps]
    if gen.uses_store_views:
        caps.append("StorageImageWriteWithoutFormat")     # shaderStorageImageWriteWithoutFormat
    pops = gen.uses_pops and gen.stage == "ps"
    exts = (["SPV_KHR_physical_storage_buffer"] if psb else []) + (["SPV_EXT_descriptor_indexing"] if tex else [])
    if bary:
        exts.append("SPV_KHR_fragment_shader_barycentric")
    if pops:
        caps.append("FragmentShaderPixelInterlockEXT")
        exts.append("SPV_EXT_fragment_shader_interlock")
    hdr = [f"OpCapability {c}" for c in caps] + [f'OpExtension "{e}"' for e in exts] + \
          ['%glsl = OpExtInstImport "GLSL.std.450"',
           f"OpMemoryModel {'PhysicalStorageBuffer64' if psb else 'Logical'} GLSL450",
           f"OpEntryPoint {model} %main \"main\" {' '.join(iface)}"]
    if cs:
        hdr.append("OpExecutionMode %main LocalSize {} {} {}".format(*gen.cs["size"]))
    if gen.stage == "ps":
        hdr.append("OpExecutionMode %main OriginUpperLeft")
        if pops:
            hdr.append("OpExecutionMode %main PixelInterlockOrderedEXT")
        if gen.writes_depth:
            hdr.append("OpExecutionMode %main DepthReplacing")
    dec = []
    if vs:
        dec.append("OpDecorate %vertex_index BuiltIn VertexIndex")
    if cs:
        dec += ["OpDecorate %wg_id BuiltIn WorkgroupId", "OpDecorate %local_id BuiltIn LocalInvocationId"]
    for name, tgt in gen.outputs.items():
        if name == "%out_pos":
            dec.append("OpDecorate %out_pos BuiltIn Position")
        else:
            dec.append(f"OpDecorate {name} Location {int(''.join(ch for ch in tgt if ch.isdigit()))}")
    for name, n in gen.inputs.items():
        dec.append(f"OpDecorate {name} Location {n}")
    types = ["%void = OpTypeVoid", "%fn = OpTypeFunction %void", "%bool = OpTypeBool", "%uint = OpTypeInt 32 0",
             "%int = OpTypeInt 32 1", "%float = OpTypeFloat 32", "%v4float = OpTypeVector %float 4",
             "%v2float = OpTypeVector %float 2", "%umul = OpTypeStruct %uint %uint",
             "%smul = OpTypeStruct %int %int",
             "%n256 = OpConstant %uint 256", "%n128 = OpConstant %uint 128",
             "%arr_v = OpTypeArray %uint %n256", "%arr_s = OpTypeArray %uint %n128",
             "%null_v = OpConstantNull %arr_v", "%null_s = OpConstantNull %arr_s",
             "%_ptr_Function_arr_v = OpTypePointer Function %arr_v", "%_ptr_Function_arr_s = OpTypePointer Function %arr_s",
             "%_ptr_Function_uint = OpTypePointer Function %uint", "%_ptr_Output_v4 = OpTypePointer Output %v4float",
             "%_ptr_Input_int = OpTypePointer Input %int", "%_ptr_Input_v4 = OpTypePointer Input %v4float",
             "%_ptr_Input_float = OpTypePointer Input %float",
             "%f0 = OpConstant %float 0", "%f1 = OpConstant %float 1", "%true = OpConstantTrue %bool",
             "%false = OpConstantFalse %bool",
             "%_ptr_Function_bool = OpTypePointer Function %bool"]
    if psb:
        types += ["%ulong = OpTypeInt 64 0", "%c64_32 = OpConstant %ulong 32", "%long = OpTypeInt 64 1",
                  "%rt_uint = OpTypeRuntimeArray %uint", "%_ptr_PSB_arr = OpTypePointer PhysicalStorageBuffer %rt_uint",
                  "%_ptr_PSB_uint = OpTypePointer PhysicalStorageBuffer %uint"]
        dec.append("OpDecorate %rt_uint ArrayStride 4")
    if tex:
        # Bindless set 0: images (binding 0), samplers (binding 1), descriptor hash table (binding 2).
        types += ["%img2d = OpTypeImage %float 2D 0 0 0 1 Unknown", "%smp = OpTypeSampler",
                  "%simg2d = OpTypeSampledImage %img2d", "%v2int = OpTypeVector %int 2",
                  "%rt_img = OpTypeRuntimeArray %img2d", "%rt_smp = OpTypeRuntimeArray %smp",
                  "%_ptr_UC_rt_img = OpTypePointer UniformConstant %rt_img",
                  "%_ptr_UC_rt_smp = OpTypePointer UniformConstant %rt_smp",
                  "%_ptr_UC_img = OpTypePointer UniformConstant %img2d", "%_ptr_UC_smp = OpTypePointer UniformConstant %smp",
                  "%rt_map = OpTypeRuntimeArray %uint", "%DescMap = OpTypeStruct %rt_map",
                  "%_ptr_SB_DescMap = OpTypePointer StorageBuffer %DescMap", "%_ptr_SB_uint = OpTypePointer StorageBuffer %uint"]
        dec += ["OpDecorate %images DescriptorSet 0", "OpDecorate %images Binding 0",
                "OpDecorate %samplers DescriptorSet 0", "OpDecorate %samplers Binding 1",
                "OpDecorate %desc_map DescriptorSet 0", "OpDecorate %desc_map Binding 2",
                "OpDecorate %DescMap Block", "OpMemberDecorate %DescMap 0 Offset 0",
                "OpDecorate %rt_map ArrayStride 4", "OpDecorate %desc_map NonWritable"]
        types += ["%v3float = OpTypeVector %float 3", "%v3int = OpTypeVector %int 3"]
        if gen.uses_tex_array:
            types += ["%img2da = OpTypeImage %float 2D 0 1 0 1 Unknown", "%simg2da = OpTypeSampledImage %img2da",
                      "%rt_imga = OpTypeRuntimeArray %img2da",
                      "%_ptr_UC_rt_imga = OpTypePointer UniformConstant %rt_imga",
                      "%_ptr_UC_imga = OpTypePointer UniformConstant %img2da"]
            dec += ["OpDecorate %images_arr DescriptorSet 0", "OpDecorate %images_arr Binding 3"]
        if gen.uses_storage3d:
            types += ["%simg3d_u = OpTypeImage %uint 3D 0 0 0 2 R32ui", "%rt_simg3 = OpTypeRuntimeArray %simg3d_u",
                      "%_ptr_UC_rt_simg3 = OpTypePointer UniformConstant %rt_simg3",
                      "%_ptr_UC_simg3 = OpTypePointer UniformConstant %simg3d_u"]
            if not gen.uses_storage:
                types += ["%_ptr_Image_uint = OpTypePointer Image %uint"]
            dec += ["OpDecorate %simages3d DescriptorSet 0", "OpDecorate %simages3d Binding 7"]
        if gen.uses_tex_3d:
            types += ["%img3d = OpTypeImage %float 3D 0 0 0 1 Unknown", "%simg3d = OpTypeSampledImage %img3d",
                      "%rt_img3 = OpTypeRuntimeArray %img3d",
                      "%_ptr_UC_rt_img3 = OpTypePointer UniformConstant %rt_img3",
                      "%_ptr_UC_img3 = OpTypePointer UniformConstant %img3d"]
            dec += ["OpDecorate %images_3d DescriptorSet 0", "OpDecorate %images_3d Binding 6"]
        if gen.uses_tex_1d:
            types += ["%img1d = OpTypeImage %float 1D 0 0 0 1 Unknown", "%simg1d = OpTypeSampledImage %img1d",
                      "%rt_img1 = OpTypeRuntimeArray %img1d",
                      "%_ptr_UC_rt_img1 = OpTypePointer UniformConstant %rt_img1",
                      "%_ptr_UC_img1 = OpTypePointer UniformConstant %img1d"]
            dec += ["OpDecorate %images_1d DescriptorSet 0", "OpDecorate %images_1d Binding 5"]
        if gen.uses_storage:
            types += ["%simg2d_u = OpTypeImage %uint 2D 0 0 0 2 R32ui", "%rt_simg = OpTypeRuntimeArray %simg2d_u",
                      "%_ptr_UC_rt_simg = OpTypePointer UniformConstant %rt_simg",
                      "%_ptr_UC_simg = OpTypePointer UniformConstant %simg2d_u",
                      "%_ptr_Image_uint = OpTypePointer Image %uint", "%v4uint = OpTypeVector %uint 4"]
            dec += ["OpDecorate %simages DescriptorSet 0", "OpDecorate %simages Binding 4"]
    if sg:
        types += ["%_ptr_Input_uint = OpTypePointer Input %uint"]
        dec += ["OpDecorate %sg_lane BuiltIn SubgroupLocalInvocationId"]
        if gen.stage == "ps":
            dec += ["OpDecorate %sg_lane Flat"]
        gen.consts.setdefault(3, "%c3")
    gen.consts.setdefault(0, "%c0")
    for v, name in sorted(gen.consts.items()):
        types.append(f"{name} = OpConstant %uint {v}")
    gvars = [f"{n} = OpVariable %_ptr_Output_v4 Output" for n in gen.outputs]
    # Push constants: s0..s31 user data, then the 64-bit host address of the Xbox GPU VA window (heap base).
    types += [f"%n_ud = OpConstant %uint {USER_SGPRS}", "%arr_ud = OpTypeArray %uint %n_ud",
              "%UserData = OpTypeStruct %arr_ud %uint %uint %uint %uint",
              "%_ptr_PC_UserData = OpTypePointer PushConstant %UserData",
              "%_ptr_PC_uint = OpTypePointer PushConstant %uint"]
    dec += ["OpDecorate %arr_ud ArrayStride 4", "OpDecorate %UserData Block", "OpMemberDecorate %UserData 0 Offset 0",
            f"OpMemberDecorate %UserData 1 Offset {4 * USER_SGPRS}",
            f"OpMemberDecorate %UserData 2 Offset {4 * USER_SGPRS + 4}",
            f"OpMemberDecorate %UserData 3 Offset {4 * USER_SGPRS + 8}",
            f"OpMemberDecorate %UserData 4 Offset {4 * USER_SGPRS + 12}"]
    gvars += ["%user_data = OpVariable %_ptr_PC_UserData PushConstant"]
    if gen.uses_store_views:
        u, si = Gen.int_format_masks()
        types += ["%v4uint_st = OpTypeVector %uint 4", "%v4int_st = OpTypeVector %int 4",
                  f"%fmt_mask_arr = OpTypeArray %uint {gen.cu(16)}",
                  "%_ptr_Private_fmt = OpTypePointer Private %fmt_mask_arr",
                  "%_ptr_Private_uint = OpTypePointer Private %uint"]
        for nm, words in (("uint", u), ("sint", si)):
            ids = " ".join(gen.cu(w) for w in words)
            types += [f"%fmt_{nm}_init = OpConstantComposite %fmt_mask_arr {ids}"]
            gvars += [f"%fmt_{nm}_mask = OpVariable %_ptr_Private_fmt Private %fmt_{nm}_init"]
        for di in sorted(gen.uses_store_views):
            _, sdim, arr, _ = next(v for v in Gen.STORE_DIMS.values() if v[0] == di)
            for cls, st in enumerate(("%float", "%uint", "%int")):
                types += [f"%stimg{cls}_{di} = OpTypeImage {st} {sdim} 0 {arr} 0 2 Unknown",
                          f"%rt_stv{cls}_{di} = OpTypeRuntimeArray %stimg{cls}_{di}",
                          f"%_ptr_UC_rt_stv{cls}_{di} = OpTypePointer UniformConstant %rt_stv{cls}_{di}",
                          f"%_ptr_UC_stv{cls}_{di} = OpTypePointer UniformConstant %stimg{cls}_{di}"]
                dec += [f"OpDecorate %stv{cls}_{di} DescriptorSet 0", f"OpDecorate %stv{cls}_{di} Binding {8 + 3 * di + cls}"]
                gvars += [f"%stv{cls}_{di} = OpVariable %_ptr_UC_rt_stv{cls}_{di} UniformConstant"]
    if cs:
        types += ["%v3uint = OpTypeVector %uint 3", "%_ptr_Input_v3u = OpTypePointer Input %v3uint"]
        gvars += ["%wg_id = OpVariable %_ptr_Input_v3u Input", "%local_id = OpVariable %_ptr_Input_v3u Input"]
        if gen.uses_lds:
            types += [f"%lds_arr = OpTypeArray %uint {gen.cu(gen.lds_dwords)}",
                      "%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr", "%_ptr_WG_uint = OpTypePointer Workgroup %uint"]
            gvars += ["%lds = OpVariable %_ptr_WG_lds Workgroup"]
    if gen.frag_inputs:
        dec += ["OpDecorate %frag_coord BuiltIn FragCoord"]
        gvars += ["%frag_coord = OpVariable %_ptr_Input_v4 Input"]
        if "front" in gen.frag_inputs.values():
            types += ["%_ptr_Input_bool = OpTypePointer Input %bool"]
            dec += ["OpDecorate %front_facing BuiltIn FrontFacing"]
            gvars += ["%front_facing = OpVariable %_ptr_Input_bool Input"]
        if bary:
            if not tex:
                types += ["%v3float = OpTypeVector %float 3"]
            types += ["%_ptr_Input_v3f = OpTypePointer Input %v3float"]
            dec += ["OpDecorate %bary BuiltIn BaryCoordKHR"]
            gvars += ["%bary = OpVariable %_ptr_Input_v3f Input"]
    if gen.vertex_inputs:
        types += ["%v4uint_in = OpTypeVector %uint 4", "%v4int_in = OpTypeVector %int 4",
                  "%_ptr_Input_v4u = OpTypePointer Input %v4uint_in", "%_ptr_Input_v4i = OpTypePointer Input %v4int_in"]
        for reg, ctype in gen.vertex_inputs:
            dec.append(f"OpDecorate %vin{reg} Location {reg}")
    if gen.writes_depth:
        types += ["%_ptr_Output_float = OpTypePointer Output %float"]
        dec += ["OpDecorate %out_depth BuiltIn FragDepth"]
        gvars += ["%out_depth = OpVariable %_ptr_Output_float Output"]
    gvars += [f"{n} = OpVariable %_ptr_Input_v4 Input" for n in gen.inputs]
    if sg:
        gvars += ["%sg_lane = OpVariable %_ptr_Input_uint Input"]
    if tex:
        gvars += ["%images = OpVariable %_ptr_UC_rt_img UniformConstant",
                  "%samplers = OpVariable %_ptr_UC_rt_smp UniformConstant",
                  "%desc_map = OpVariable %_ptr_SB_DescMap StorageBuffer"]
        if gen.uses_tex_array:
            gvars += ["%images_arr = OpVariable %_ptr_UC_rt_imga UniformConstant"]
        if gen.uses_storage:
            gvars += ["%simages = OpVariable %_ptr_UC_rt_simg UniformConstant"]
        if gen.uses_tex_1d:
            gvars += ["%images_1d = OpVariable %_ptr_UC_rt_img1 UniformConstant"]
        if gen.uses_tex_3d:
            gvars += ["%images_3d = OpVariable %_ptr_UC_rt_img3 UniformConstant"]
        if gen.uses_storage3d:
            gvars += ["%simages3d = OpVariable %_ptr_UC_rt_simg3 UniformConstant"]
    if vs:
        gvars.append("%vertex_index = OpVariable %_ptr_Input_int Input")
    for reg, ctype in gen.vertex_inputs:
        ptr = {1: "%_ptr_Input_v4u", 2: "%_ptr_Input_v4i"}.get(ctype, "%_ptr_Input_v4")
        gvars.append(f"%vin{reg} = OpVariable {ptr} Input")
    fn = ["%main = OpFunction %void None %fn", "%entry = OpLabel",
          # Registers start at zero (deterministic); real inputs are written explicitly at entry.
          "%vgpr = OpVariable %_ptr_Function_arr_v Function %null_v",
          "%sgpr = OpVariable %_ptr_Function_arr_s Function %null_s",
          "%vcc = OpVariable %_ptr_Function_uint Function %c0", "%m0 = OpVariable %_ptr_Function_uint Function %c0",
          "%vcc_hi = OpVariable %_ptr_Function_uint Function %c0",
          "%scc = OpVariable %_ptr_Function_uint Function %c0",
          "%active = OpVariable %_ptr_Function_bool Function %true"]
    # User data: s0..s31 are loaded at entry from a push-constant block (the driver's user-SGPR setup).
    if gen.uses_heap:
        gen.consts.setdefault(1, "%c1")
        gen.consts.setdefault(2, "%c2")
        fn += ["%ud_base_lo_p = OpAccessChain %_ptr_PC_uint %user_data %c1",
               "%ud_base_hi_p = OpAccessChain %_ptr_PC_uint %user_data %c2"]
    if gen.uses_codebase:
        fn += ["%ud_code_lo_p = OpAccessChain %_ptr_PC_uint %user_data %c3",
               "%ud_code_hi_p = OpAccessChain %_ptr_PC_uint %user_data %c4"]
    for i in range(USER_SGPRS):
        fn += [f"%ud{i}p = OpAccessChain %_ptr_PC_uint %user_data %c0 {gen.cu(i)}",
               f"%ud{i} = OpLoad %uint %ud{i}p",
               f"%ud{i}s = OpAccessChain %_ptr_Function_uint %sgpr {gen.cu(i)}",
               f"OpStore %ud{i}s %ud{i}"]
    if vs:
        fn += ["%vi = OpLoad %int %vertex_index", "%viu = OpBitcast %uint %vi",
               "%v0p = OpAccessChain %_ptr_Function_uint %vgpr %c0", "OpStore %v0p %viu"]
    if cs:
        # After the user-data copy: workgroup ids overwrite s[user..], thread ids go to v0..v2.
        fn += ["%wgi = OpLoad %v3uint %wg_id", "%lci = OpLoad %v3uint %local_id"]
        for k, c in enumerate(gen.cs["tgid"]):
            r = gen.cs["user"] + k
            fn += [f"%wg{c} = OpCompositeExtract %uint %wgi {c}",
                   f"%wgp{c} = OpAccessChain %_ptr_Function_uint %sgpr {gen.cu(r)}", f"OpStore %wgp{c} %wg{c}"]
        for c in range(gen.cs["tid"] + 1):
            fn += [f"%li{c} = OpCompositeExtract %uint %lci {c}",
                   f"%lip{c} = OpAccessChain %_ptr_Function_uint %vgpr {gen.cu(c)}", f"OpStore %lip{c} %li{c}"]
    if gen.frag_inputs:
        fn += ["%fc = OpLoad %v4float %frag_coord"]
        if bary:
            fn += ["%bc = OpLoad %v3float %bary"]
        for r, c in gen.frag_inputs.items():
            if str(c).startswith("bary"):
                k = 1 if c == "bary_i" else 2
                fn += [f"%bc{k} = OpCompositeExtract %float %bc {k}", f"%bcu{k} = OpBitcast %uint %bc{k}",
                       f"%bcp{k} = OpAccessChain %_ptr_Function_uint %vgpr {gen.cu(r)}", f"OpStore %bcp{k} %bcu{k}"]
                continue
            if c == "front":
                fn += ["%ff = OpLoad %bool %front_facing", "%ffm1 = OpFNegate %float %f1",
                       "%ffv = OpSelect %float %ff %f1 %ffm1", "%ffu = OpBitcast %uint %ffv",
                       f"%ffp = OpAccessChain %_ptr_Function_uint %vgpr {gen.cu(r)}", "OpStore %ffp %ffu"]
                continue
            fn += [f"%fc{c} = OpCompositeExtract %float %fc {c}", f"%fcu{c} = OpBitcast %uint %fc{c}",
                   f"%fcp{c} = OpAccessChain %_ptr_Function_uint %vgpr {gen.cu(r)}", f"OpStore %fcp{c} %fcu{c}"]
    for reg, ctype in gen.vertex_inputs:
        vt = {1: "%v4uint_in", 2: "%v4int_in"}.get(ctype, "%v4float")
        st = {1: "%uint", 2: "%int"}.get(ctype, "%float")
        fn.append(f"%vinv{reg} = OpLoad {vt} %vin{reg}")
        for c in range(4):
            vg = 4 + 4 * reg + c
            fn += [f"%vinc{reg}_{c} = OpCompositeExtract {st} %vinv{reg} {c}",
                   f"%vinb{reg}_{c} = OpBitcast %uint %vinc{reg}_{c}" if ctype != 1 else
                   f"%vinb{reg}_{c} = OpCopyObject %uint %vinc{reg}_{c}",
                   f"%vinp{reg}_{c} = OpAccessChain %_ptr_Function_uint %vgpr {gen.cu(vg)}",
                   f"OpStore %vinp{reg}_{c} %vinb{reg}_{c}"]
    out = hdr + dec + types + gvars + fn
    if gen.blocks:
        # Dispatch loop: any CFG (forward/backward branches) in structured SPIR-V form.
        n = len(gen.blocks)
        lines = ["%pc = OpVariable %_ptr_Function_uint Function %c0",
                 "%done = OpVariable %_ptr_Function_bool Function %false",
                 "OpBranch %loop_head", "%loop_head = OpLabel", "OpLoopMerge %loop_exit %loop_cont None",
                 "OpBranch %dispatch", "%dispatch = OpLabel", "%pcv = OpLoad %uint %pc",
                 "OpSelectionMerge %sw_merge None",
                 "OpSwitch %pcv %sw_default " + " ".join(f"{i} %blk{i}" for i in range(n))]
        body = []
        for i, blines in enumerate(gen.blocks):
            body.append(f"%blk{i} = OpLabel")
            body += blines
            body.append("OpBranch %sw_merge")
        body += ["%sw_default = OpLabel", "OpStore %done %true", "OpBranch %sw_merge",
                 "%sw_merge = OpLabel", "OpBranch %loop_cont", "%loop_cont = OpLabel",
                 "%dn = OpLoad %bool %done", "OpBranchConditional %dn %loop_exit %loop_head",
                 "%loop_exit = OpLabel"]
        # %pc/%done must be declared with the other Function variables, at the top of the entry block,
        # before the vertex-shader prologue.
        fn_vars = lines[:2]
        flow = lines[2:]
        last_var = max(i for i, l in enumerate(fn) if " = OpVariable " in l)
        fn = fn[:last_var + 1] + fn_vars + fn[last_var + 1:]
        if pops:
            fn.append("OpBeginInvocationInterlockEXT")   # end of the entry block: uniform, executed once
        out = hdr + dec + types + gvars + fn
        gen.body = ["         " + l if not (l.startswith("%") and " = OpLabel" in l) else l for l in flow + body]
    end_lock = "         OpEndInvocationInterlockEXT\n" if pops else ""
    if gen.uses_active and gen.stage == "ps":
        # A pixel lane switched off by v_cmpx at the final export writes nothing: equivalent to discard.
        tail = end_lock + ("         %act_end = OpLoad %bool %active\n"
                "         OpSelectionMerge %live None\n"
                "         OpBranchConditional %act_end %live %dead\n"
                "%dead = OpLabel\n         OpKill\n"
                "%live = OpLabel\n         OpReturn\n         OpFunctionEnd\n")
    else:
        tail = end_lock + "         OpReturn\n         OpFunctionEnd\n"
    return dedupe_types("\n".join("         " + l if not l.startswith("%") or " = " not in l else l for l in out) +
                        "\n" + "\n".join(gen.body) + "\n" + tail)


_STAGES = None


def stage_of(name, path):
    """Shader stage from the carve index (the DXIL program header in each container), not the name."""
    global _STAGES
    if _STAGES is None:
        import json
        idx = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(path))), "index.json")
        _STAGES = {}
        if os.path.isfile(idx):
            for e in json.load(open(idx, encoding="utf-8")):
                _STAGES[e["file"][:-5]] = e["stage"]
    return _STAGES.get(name) or (name[:2].lower() if name[2:3] == "_" else "?")


def reachable_code(insns):
    """Instructions reachable from the first one, following fall-through and branch targets. Fall-through stops
    after s_endpgm, s_branch and s_setpc_b64. Raises Unsupported if a branch target isn't an instruction start."""
    by_pc = {i["pc"]: k for k, i in enumerate(insns)}
    seen, work = set(), [0]
    while work:
        k = work.pop()
        while k is not None and k < len(insns) and k not in seen:
            seen.add(k)
            ins = insns[k]
            op = base_op(ins["op"])
            nxt = ins["pc"] + len(ins["enc"]) // 2
            if op in BRANCHES:
                imm = next((x for x in ins["args"] if x["k"] == "imm"), None)
                v = (imm["v"] & 0xFFFF) if imm else 0
                tgt = nxt + 4 * (v - 0x10000 if v & 0x8000 else v)
                if tgt not in by_pc:
                    raise Unsupported("branch into data")
                work.append(by_pc[tgt])
                if op == "s_branch":
                    break
            if op in ("s_endpgm", "s_setpc_b64"):
                break
            k = by_pc.get(nxt)
            if k is None and op not in ("s_endpgm",):
                break
    return [insns[k] for k in sorted(seen)]


def read_before_write_vgprs(insns):
    """VGPRs a shader reads before writing them: for a fetch-calling VS these are the fetched attributes."""
    written, first = set(), []
    for ins in insns:
        a, op = ins["args"], ins["op"]
        if not a:
            continue
        reads_all = op.startswith(("exp", "buffer_store", "image_store", "v_cmp", "v_cmpx", "ds_write", "global_store"))
        regs = []
        for x in (a if reads_all else a[1:]):
            if x["k"] == "vgpr":
                regs += range(x["r"], x["r"] + x.get("n", 1))
            elif x["k"] == "vlist":
                regs += x["r"]
        if base_op(op) in ("v_fmac_f32", "v_mac_f32", "v_fmac_f16") and a[0]["k"] == "vgpr":
            regs.append(a[0]["r"])
        for r in regs:
            if r not in written and r not in first:
                first.append(r)
        if not reads_all and a[0]["k"] == "vgpr":
            written.update(range(a[0]["r"], a[0]["r"] + a[0].get("n", 1)))
    return set(first)


def input_signature_full(path, name):
    """ISG1 elements as (system value, declared mask, read/write mask)."""
    dxbc = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(path))), name + ".dxbc")
    d = open(dxbc, "rb").read()
    for k in range(struct.unpack_from("<I", d, 28)[0]):
        off = struct.unpack_from("<I", d, 32 + 4 * k)[0]
        if d[off:off + 4] == b"ISG1":
            p = d[off + 8:off + 8 + struct.unpack_from("<I", d, off + 4)[0]]
            cnt, eo = struct.unpack_from("<II", p, 0)
            return [(struct.unpack_from("<I", p, eo + 32 * i + 12)[0], p[eo + 32 * i + 24], p[eo + 32 * i + 25])
                    for i in range(cnt)]
    return []


def input_signature(path, name):
    """ISG1 elements of the shader's DXBC container: (register, mask, system value, component type)."""
    dxbc = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(path))), name + ".dxbc")
    d = open(dxbc, "rb").read()
    for k in range(struct.unpack_from("<I", d, 28)[0]):
        off = struct.unpack_from("<I", d, 32 + 4 * k)[0]
        if d[off:off + 4] == b"ISG1":
            p = d[off + 8:off + 8 + struct.unpack_from("<I", d, off + 4)[0]]
            cnt, eo = struct.unpack_from("<II", p, 0)
            out = []
            for i in range(cnt):
                e = p[eo + 32 * i:eo + 32 * i + 32]
                _, _, _, sysv, ctype, reg = struct.unpack_from("<6I", e, 0)
                out.append((reg, e[24], sysv, ctype))
            return out
    return []


def same_encoding(raw, printed, op):
    """True if the bytes in the blob are the instruction LLVM printed. Exact match, except for fields the
    hardware ignores that LLVM re-encodes with its default: VOP3P op_sel_hi[2] (bit 14) on two-source packed
    ops (seen: v_pk_add_f16 stored with 0, printed with 1)."""
    if raw == printed:
        return True
    if len(raw) != len(printed) or len(raw) < 8:
        return False
    diff = int.from_bytes(raw, "little") ^ int.from_bytes(printed, "little")
    vop3p = printed[3] == 0xCC
    two_src = op.startswith("v_pk_") and not any(k in op for k in ("fma", "mad"))
    return vop3p and two_src and diff == 1 << 14


def translate(path):
    info = parse_file(path)
    stage = stage_of(info["name"], path)
    if stage not in ("vs", "ps", "cs"):
        raise Unsupported(f"stage {stage}")
    insns = info["insns"]
    if info["invalid"]:
        # Undecodable words are constant tables embedded in the code range (147/155 such VS/PS use s_getpc_b64 to
        # read them). LLVM skips them, so re-anchor every instruction to its true offset in the FSSM blob, then
        # translate only code reachable from entry; a branch into data is rejected below.
        fssm = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(path))), "fssm", info["name"] + ".fssm")
        blob = open(fssm, "rb").read()
        pos = info["code_start"]
        for ins in insns:
            enc = bytes.fromhex(ins["enc"])
            skipped, at = 0, pos
            while not same_encoding(blob[at:at + len(enc)], enc, base_op(ins["op"])):
                at += 4
                skipped += 4
                if skipped > 4096 or at >= len(blob):
                    at = None
                    break
            if at is None:
                # LLVM prints the encoding of the instruction it decoded, not the bytes it read: data words that
                # decode with ignored fields set (e.g. 0xbe8d2242 -> s_rfe_b64, printed 0xbe802242) never match.
                # The decoder consumed the same number of bytes, so the instruction sits at the current position.
                # Such words are data in every case seen; reachable_code drops them unless a branch lands there.
                if pos + len(enc) > len(blob):
                    raise Unsupported("undecodable words (cannot re-anchor)")
                at = pos
                ins["normalized"] = True
            ins["pc"] = at
            pos = at + len(enc)
        insns = reachable_code(insns)
        if any(i.get("normalized") for i in insns):
            raise Unsupported("reachable instruction with normalized encoding")
    g = Gen(stage)
    if stage == "cs":
        # Compute system registers come from the FSSM header. Word 0x138 is COMPUTE_PGM_RSRC2 (checked against
        # what 2,689 CS read before writing: 2,281 fit exactly, the rest are loop artifacts of the straight-line
        # check): USER_SGPR [5:1], TGID_X/Y/Z_EN [9:7], TG_SIZE_EN [10], TIDIG_COMP_CNT [12:11]. Words
        # 0x13C/0x140/0x144 are the thread-group size. Layout (Mesa radv_shader_args.c): s0..user-1 = user data,
        # then the enabled workgroup ids; v0, v1, v2 = thread id x, y, z (separate VGPRs on GFX10).
        fssm = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(path))), "fssm", info["name"] + ".fssm")
        hdr = open(fssm, "rb").read(0x148)
        rsrc2, lx, ly, lz = struct.unpack_from("<4I", hdr, 0x138)
        if (rsrc2 >> 10) & 1:
            raise Unsupported("CS thread-group-size SGPR")
        if not (1 <= lx * ly * lz <= 1024):
            raise Unsupported("CS group size")
        user = (rsrc2 >> 1) & 31
        g.cs = {"user": user, "tgid": [c for c in range(3) if (rsrc2 >> (7 + c)) & 1],
                "tid": (rsrc2 >> 11) & 3, "size": (lx, ly, lz)}
        # LDS_SIZE [23:15] in 512-byte units (470 of 486 LDS-using CS stay below it; the rest index past it with
        # constant offsets, so the array also covers the largest offset in the code).
        need = ((rsrc2 >> 15) & 0x1FF) * 128
        for ins in insns:
            if not ins["op"].startswith("ds_") or ins["op"].startswith(("ds_swizzle", "ds_bpermute")):
                continue
            for m in ins["mods"]:
                k, _, v = m.partition(":")
                if not v[:1].isdigit():
                    continue
                if k == "offset":
                    need = max(need, int(v, 0) // 4 + 4)
                elif k in ("offset0", "offset1"):
                    need = max(need, int(v, 0) * (64 if "st64" in ins["op"] else 1) * 2 + 4)
        n = 16
        while n < need:
            n *= 2
        g.lds_dwords = min(n, 16384)
        g.cu(g.lds_dwords)
        g.cu(g.lds_dwords - 1)
        for i in range(user, user + 3):
            g.cu(i)
    if stage == "ps":
        # Pixel-shader system VGPRs (verified on 2,531 of 3,000 PS): v0-v1 = barycentrics (modelled through
        # v_interp -> attributes), then v2.. = the SV_Position components the shader reads, packed in order.
        # Anything else a PS reads before writing is an input layout we haven't confirmed: reject it.
        sig = input_signature_full(path, info["name"])
        pos = [rw for sv, m, rw in sig if sv == 1]
        used = pos[0] if pos else 0
        # Position w: AMD's POS_W_FLOAT input holds 1/w. Mesa (src/amd/common/nir/ac_nir_lower_intrinsics_to_args.c)
        # lowers nir_intrinsic_load_frag_coord_w_rcp straight to frag_pos[3]. That's the same value as Vulkan's
        # FragCoord.w, so all four components map directly.
        # Hardware order (Mesa src/amd/vulkan/radv_shader_args.c): barycentric pairs, ..., POS_X/Y/Z/W, FRONT_FACE,
        # ANCILLARY, SAMPLE_COVERAGE, POS_FIXED_PT. Only enabled inputs are packed. Front face is a float whose
        # sign gives facing (Mesa: load_front_face = front_face > 0; load_front_face_fsign = the raw value).
        layout = {}
        nxt = 2
        for c in range(4):
            if used >> c & 1:
                layout[nxt] = c
                nxt += 1
        if any(sv == 9 for sv, _, _ in sig):           # D3D_NAME_IS_FRONT_FACE
            layout[nxt] = "front"
            nxt += 1
        no_bary = [dict(i, args=[x for k, x in enumerate(i["args"])
                                 if not (i["op"].startswith("v_interp") and x["k"] == "vgpr" and k != 0)])
                   for i in insns]
        reads = read_before_write_vgprs(no_bary)
        # Reads inside the system-input range that aren't in the layout (raw barycentrics, unknown inputs): reject.
        # Reads past the last hardware input can't be hardware inputs: they're values undefined on some path, and
        # the zero-initialised registers stand in for them.
        # Raw reads of the barycentric pair (v0 = i, v1 = j) outside v_interp, e.g. v_readfirstlane_b32 s41, v1 in
        # 315 PS: AMD interpolates P = P0 + i(P1 - P0) + j(P2 - P0), so i and j are the weights of vertices 1 and 2,
        # which VK_KHR_fragment_shader_barycentric gives as BaryCoordKHR.y / .z. Assumes the pair is the
        # perspective-center one, the only pair this layout models.
        for r, comp in ((0, "bary_i"), (1, "bary_j")):
            if r in reads:
                layout[r] = comp
        unexplained = {r for r in reads if r < nxt and r not in layout}
        if unexplained:
            raise Unsupported("PS system inputs")
        g.frag_inputs = layout
        for r in layout:
            g.cu(r)
    if stage == "vs" and any(base_op(i["op"]) == "s_swappc_b64" for i in insns):
        # The call goes to a driver-generated fetch program (not in the blob). Verified rule over 788 shaders:
        # input-signature register r, component c is fetched into v(4 + 4r + c). Replace the call with Vulkan
        # vertex inputs (location = register) written into those VGPRs at entry.
        sig = [s for s in input_signature(path, info["name"]) if s[2] == 0]
        # Any component of a declared register may be read: components the vertex format lacks come back as the
        # format defaults (0, 0, 1 for y, z, w) from the hardware fetch, and Vulkan fills a vec4 input from a
        # narrower format the same way. Seen: .w of an xyz position.
        predicted = {4 + 4 * reg + c for reg, mask, _, _ in sig for c in range(4)}
        reads = {r for r in read_before_write_vgprs(insns) if r >= 4}
        if not reads <= predicted:
            raise Unsupported("fetch mapping")
        g.vertex_inputs = [(reg, ctype) for reg, _, _, ctype in sig]
        for reg, _ in g.vertex_inputs:          # register the VGPR index constants before assembly
            for c in range(4):
                g.cu(4 + 4 * reg + c)
        insns = [i for i in insns if base_op(i["op"]) != "s_swappc_b64"]
    branchy = any(base_op(i["op"]) in BRANCHES for i in insns)
    if not branchy:
        for ins in insns:
            if base_op(ins["op"]) == "s_endpgm":
                break
            g.insn(ins)
        return assemble(g)

    # ---- basic blocks ----
    def next_pc(ins):
        return ins["pc"] + len(ins["enc"]) // 2

    def target(ins):
        imm = next((x for x in ins["args"] if x["k"] == "imm"), None)
        if imm is None:
            raise Unsupported("branch without offset")
        v = imm["v"] & 0xFFFF
        return next_pc(ins) + 4 * (v - 0x10000 if v & 0x8000 else v)

    leaders = {insns[0]["pc"]}
    for ins in insns:
        if base_op(ins["op"]) in BRANCHES:
            leaders.add(target(ins))
            leaders.add(next_pc(ins))
    starts = sorted(p for p in leaders if any(i["pc"] == p for i in insns))
    if any(p not in starts for p in (target(i) for i in insns if base_op(i["op"]) in BRANCHES)):
        raise Unsupported("branch target outside code")
    index = {p: n for n, p in enumerate(starts)}
    blocks = [[] for _ in starts]
    cur = -1
    for ins in insns:
        if ins["pc"] in index:
            cur = index[ins["pc"]]
        blocks[cur].append(ins)

    for n, blk in enumerate(blocks):
        g.body = []
        g.block_label = f"%blk{n}"
        term = None
        for ins in blk:
            op = base_op(ins["op"])
            if op == "s_endpgm":
                term = ("end",)
                break
            if op in BRANCHES:
                term = ("br", op, index[target(ins)])
                break
            g.insn(ins)
        fall = n + 1 if n + 1 < len(blocks) else None
        if term is None:
            g.emit(f"OpStore %pc {g.cu(fall)}" if fall is not None else "OpStore %done %true")
        elif term[0] == "end":
            g.emit("OpStore %done %true")
        else:
            _, op, tgt = term
            if op == "s_branch":
                g.emit(f"OpStore %pc {g.cu(tgt)}")
            else:
                cond = branch_condition(g, op)
                alt = g.cu(fall) if fall is not None else g.cu(len(blocks))   # past the end = default -> done
                r = g.tmp()
                g.emit(f"{r} = OpSelect %uint {cond} {g.cu(tgt)} {alt}")
                g.emit(f"OpStore %pc {r}")
        g.blocks.append([l.strip() for l in g.body])
    g.body = []
    return assemble(g)


BRANCHES = {"s_branch", "s_cbranch_scc0", "s_cbranch_scc1", "s_cbranch_vccz", "s_cbranch_vccnz",
            "s_cbranch_execz", "s_cbranch_execnz"}


def branch_condition(g, op):
    """Boolean SPIR-V id: whether the branch is taken for this lane."""
    if op in ("s_cbranch_scc0", "s_cbranch_scc1"):
        s = g.tmp()
        g.emit(f"{s} = OpLoad %uint %scc")
        b = g.tmp()
        g.emit(f"{b} = {'OpIEqual' if op.endswith('0') else 'OpINotEqual'} %bool {s} {g.cu(0)}")
        return b
    # VCC and EXEC are wave-wide registers: the hardware branch is taken (or not) by the whole wave. With one
    # invocation per lane, each lane holds only its own bit, so the condition is reduced across the subgroup
    # (= the wave). Deciding per lane would let lanes skip scalar instructions the wave executes, leaving their
    # SGPRs stale, and would break cross-lane ops and barriers inside the block.
    if op in ("s_cbranch_vccz", "s_cbranch_vccnz"):
        s = g.tmp()
        g.emit(f"{s} = OpLoad %uint %vcc")
        lane = g.tmp()
        g.emit(f"{lane} = OpINotEqual %bool {s} {g.cu(0)}")
    else:
        lane = g.active_b()
    any_ = g.tmp()
    g.emit(f"{any_} = OpGroupNonUniformAny %bool {g.cu(3)} {lane}")
    g.uses_vote = True
    if op in ("s_cbranch_vccnz", "s_cbranch_execnz"):
        return any_
    b = g.tmp()
    g.emit(f"{b} = OpLogicalNot %bool {any_}")
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    files = [a.src] if a.src.endswith(".s") else sorted(os.path.join(a.src, f) for f in os.listdir(a.src) if f.endswith(".s"))
    files = files[: a.limit] if a.limit else files
    os.makedirs(a.out, exist_ok=True)
    why, ok, valid = Counter(), 0, 0
    for p in files:
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            asm = translate(p)
        except Unsupported as e:
            why[str(e)] += 1
            continue
        except (IndexError, KeyError, ValueError) as e:
            why[f"translator error {type(e).__name__}"] += 1
            continue
        ok += 1
        sa = os.path.join(a.out, name + ".spvasm")
        sv = os.path.join(a.out, name + ".spv")
        with open(sa, "w") as f:
            f.write(asm)
        r = subprocess.run(["spirv-as", sa, "-o", sv, "--target-env", "vulkan1.2"], capture_output=True, text=True)
        if r.returncode == 0:
            r = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", "--allow-offset-texture-operand", sv],
                               capture_output=True, text=True)  # Offset on non-gather: VK_KHR_maintenance8
        if r.returncode == 0:
            valid += 1
        else:
            why["INVALID: " + (r.stderr.strip().splitlines() or ["?"])[0][:90]] += 1
    print(f"{len(files)} shaders: {ok} translated, {valid} pass spirv-val")
    for k, v in why.most_common(25):
        print(f"  {v:6}  {k}")


if __name__ == "__main__":
    sys.exit(main())
