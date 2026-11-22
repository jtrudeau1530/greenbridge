r"""Differential fuzzer for the translator's LDS and ds_* instructions (FUZZ-6), standard library only.

Generates seeded random compute programs (one 64-thread group, wave64 or wave32, 4 KiB LDS) that run a sequence
of ds_* instructions, translates them with rdna2spirv.py, runs them on the GPU with vk_compute_test and compares
every thread's result registers and the final LDS contents, bit for bit, with an independent model of the RDNA2
instructions written here (it does not import the translator).

    python -I tools/lds_fuzz.py --programs 200 --seed 1 [--wave 64|32|both] [--emulate] [--build runtime/build]
    python -I tools/lds_fuzz.py --show 1:7        # print program 7 of seed 1 and the model's result
    python -I tools/lds_fuzz.py --model-only ...  # generate and run the model only (no GPU)

Programs are deterministic: every op either touches addresses no other lane touches in the same op, or is an
order-independent atomic without return (add/sub/min/max/and/or/xor). ds_swizzle runs with all
lanes active (the ISA behaviour of a source lane that is off in EXEC isn't covered here); ds_bpermute also runs
under random EXEC masks (an inactive source lane reads 0). Swizzle offsets are
limited to the two patterns the LLVM swizzle() macros encode: bit 15 set = quad permute (4 x 2-bit selectors in
bits 7:0), clear = bit mask (and = bits 4:0, or = bits 9:5, xor = bits 14:10, within each 32-lane group).

Each mismatch is printed with the words that differ and the op that last wrote them, and a --show line that
prints the program. Exit code: 0 when every program that ran matches, 1 on a mismatch, 2 on a harness error
(assembler, translator or vk_compute_test failure) or when no program could run (every one skipped).
"""
import argparse
import json
import os
import random
import struct
import subprocess
import sys
import shutil
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "runtime", "test", "shaders"))
import mc_command  # noqa: E402
from compute_fixture import compute_header  # noqa: E402

THREADS = 64
LDS_BYTES = 4096
LDS_DWORDS = LDS_BYTES // 4
POOL = list(range(20, 36))            # result / data registers stored at the end
ADDR = 6                              # per-op address register
TMP = 7
RSRC2 = 0x8088 & ~(0x1FF << 15) | ((LDS_BYTES // 512) << 15)
OUT_WORDS = THREADS * len(POOL) + LDS_DWORDS
M32 = 0xFFFFFFFF

READS = {   # name: (bytes, kind)
    "ds_read_b32": (4, "b32"), "ds_read_b64": (8, "wide"), "ds_read_b96": (12, "wide"),
    "ds_read_b128": (16, "wide"),
    "ds_read_u8": (1, "u"), "ds_read_i8": (1, "i"), "ds_read_u16": (2, "u"), "ds_read_i16": (2, "i"),
    "ds_read_u16_d16": (2, "u_lo"), "ds_read_u16_d16_hi": (2, "u_hi"),
    "ds_read_u8_d16": (1, "u_lo"), "ds_read_u8_d16_hi": (1, "u_hi"),
    "ds_read_i8_d16": (1, "i_lo"), "ds_read_i8_d16_hi": (1, "i_hi"),
}
# The translator rejects ds_permute_b32, so it is not generated.
WRITES = {  # name: (bytes, data bit offset)
    "ds_write_b32": (4, 0), "ds_write_b64": (8, 0), "ds_write_b96": (12, 0), "ds_write_b128": (16, 0),
    "ds_write_b8": (1, 0), "ds_write_b16": (2, 0), "ds_write_b8_d16_hi": (1, 16), "ds_write_b16_d16_hi": (2, 16),
}
SHARED_ATOMICS = ("add", "sub", "min_i32", "max_i32", "min_u32", "max_u32", "and", "or", "xor")
RTN_ATOMICS = SHARED_ATOMICS
EXEC_KINDS = ("atomic", "atomic_rtn", "bpermute", "cmpst", "read", "read2", "write", "write2", "wrxchg")
ATOMIC_NAME = {"add": "ds_add%s_u32", "sub": "ds_sub%s_u32", "min_i32": "ds_min%s_i32", "max_i32": "ds_max%s_i32",
               "min_u32": "ds_min%s_u32", "max_u32": "ds_max%s_u32", "and": "ds_and%s_b32", "or": "ds_or%s_b32",
               "xor": "ds_xor%s_b32"}


def s32(x):
    x &= M32
    return x - (1 << 32) if x & 0x80000000 else x


def atomic(op, old, data):
    if op == "add":
        return (old + data) & M32
    if op == "sub":
        return (old - data) & M32
    if op == "min_i32":
        return old if s32(old) <= s32(data) else data
    if op == "max_i32":
        return old if s32(old) >= s32(data) else data
    if op == "min_u32":
        return min(old, data)
    if op == "max_u32":
        return max(old, data)
    if op == "and":
        return old & data
    if op == "or":
        return old | data
    if op == "xor":
        return old ^ data
    raise ValueError(op)


def swizzle_src(lane, offset):
    """Source lane for ds_swizzle_b32 (lane = index in the wave)."""
    if offset & 0x8000:
        sel = (offset >> (2 * (lane & 3))) & 3
        return (lane & ~3) | sel
    and_m, or_m, xor_m = offset & 31, (offset >> 5) & 31, (offset >> 10) & 31
    return (lane & ~31) | ((((lane & 31) & and_m) | or_m) ^ xor_m)


class Program:
    """A generated program: a list of ops, the wave size and the initial values."""

    def __init__(self, seed, index, wave, no_exec=()):
        self.rng = random.Random(f"{seed}:{index}:{wave}")
        self.name = f"cs_ldsfuzz_s{seed}_{index}_w{wave}"
        self.wave = wave
        rng = self.rng
        self.lds_xor = rng.getrandbits(32)
        self.lds_mul = rng.getrandbits(32) | 1
        self.pool_init = {r: (rng.randrange(1, 1 << 24), rng.getrandbits(32)) for r in POOL}
        self.ops = [self.random_op() for _ in range(rng.randrange(4, 13))]
        for op in self.ops:      # the mask is still drawn, so the rest of the program is the same
            if op["kind"] in no_exec:
                op["exec"] = None

    def exec_mask(self):
        rng = self.rng
        if rng.random() < 0.6:
            return None
        choice = rng.choice(("rand", "low", "high", "one", "none_hi"))
        if choice == "rand":
            return rng.getrandbits(64)
        if choice == "low":
            return 0x00000000FFFFFFFF
        if choice == "high":
            return 0xFFFFFFFF00000000
        if choice == "one":
            return 1 << rng.randrange(64)
        return rng.getrandbits(32)

    def lane_addresses(self, stride, base):
        return [(t * stride + base) for t in range(THREADS)]

    def disjoint(self, spans):
        """spans: per thread, list of (byte address, bytes). True when no byte is touched by two threads."""
        owner = {}
        for t, ss in enumerate(spans):
            for a, n in ss:
                if a < 0 or a + n > LDS_BYTES:
                    return False
                for b in range(a, a + n):
                    if owner.setdefault(b, t) != t:
                        return False
        return True

    def pick_addr(self, size, align, distinct, extra=()):
        """(stride, base, offset) so that every thread's access fits and, if distinct, threads don't overlap.
        extra: further byte offsets each thread also touches (read2/write2 second slot)."""
        rng = self.rng
        for _ in range(200):
            if distinct:
                stride = rng.choice([s for s in (align, 2 * align, 4, 8, 16, 32, 48, 64) if s % align == 0 and s >= 1])
            else:
                stride = rng.choice([0, align, 4, 8, 16, 32]) // align * align
            top = LDS_BYTES - stride * (THREADS - 1) - size - max(extra, default=0)
            if top < 0:
                continue
            off = rng.choice([0, 0, rng.randrange(0, min(top, 0xFFFF) + 1)]) // align * align
            base_top = top - off
            if base_top < 0:
                continue
            base = rng.randrange(0, base_top + 1) // align * align
            spans = [[(a + off, size)] + [(a + off + e, size) for e in extra] for a in self.lane_addresses(stride, base)]
            if not distinct or self.disjoint(spans):
                return stride, base, off
        return None

    def random_op(self):
        rng = self.rng
        while True:
            kind = rng.choice(("read", "read", "write", "write", "read2", "write2", "atomic", "atomic_rtn",
                               "wrxchg", "cmpst", "swizzle", "bpermute"))
            op = self.make(kind)
            if op:
                return op

    def make(self, kind):
        rng = self.rng
        if kind == "read":
            name = rng.choice(sorted(READS))
            size = READS[name][0]
            align = min(size, 4) if size < 4 else (16 if size > 8 else size)
            a = self.pick_addr(size, align, distinct=False)
            if not a:
                return None
            n = (size + 3) // 4 if size >= 4 else 1
            dst = rng.randrange(POOL[0], POOL[-1] - n + 2)
            return {"kind": kind, "name": name, "addr": a, "dst": dst, "exec": self.exec_mask()}
        if kind == "write":
            name = rng.choice(sorted(WRITES))
            size = WRITES[name][0]
            align = min(size, 4) if size < 4 else (16 if size > 8 else size)
            a = self.pick_addr(size, align, distinct=True)
            if not a:
                return None
            n = (size + 3) // 4 if size >= 4 else 1
            src = rng.randrange(POOL[0], POOL[-1] - n + 2)
            return {"kind": kind, "name": name, "addr": a, "src": src, "exec": self.exec_mask()}
        if kind in ("read2", "write2"):
            st64 = rng.random() < 0.3
            unit = 256 if st64 else 4
            o0, o1 = rng.randrange(0, 4 if st64 else 64), rng.randrange(0, 4 if st64 else 64)
            if kind == "write2" and o0 == o1:
                return None
            lo = min(o0, o1) * unit
            extra = (max(o0, o1) * unit - lo,)
            a = self.pick_addr(4, 4, distinct=(kind == "write2"), extra=extra)
            if not a:
                return None
            stride, base, _ = a
            # the instruction's offsets are offset0/offset1; place the lower one at the picked spot
            base = base + a[2]
            if base - lo < 0:
                return None
            base -= lo
            op = {"kind": kind, "st64": st64, "o0": o0, "o1": o1, "addr": (stride, base, 0), "exec": self.exec_mask()}
            if kind == "read2":
                op["dst"] = rng.randrange(POOL[0], POOL[-1])
            else:
                op["src0"], op["src1"] = rng.choice(POOL), rng.choice(POOL)
            return op
        if kind == "atomic":
            a = self.pick_addr(4, 4, distinct=False)
            if not a:
                return None
            return {"kind": kind, "op": rng.choice(SHARED_ATOMICS), "addr": a, "src": rng.choice(POOL),
                    "exec": self.exec_mask()}
        if kind in ("atomic_rtn", "wrxchg", "cmpst"):
            a = self.pick_addr(4, 4, distinct=True)
            if not a:
                return None
            op = {"kind": kind, "addr": a, "src": rng.choice(POOL), "dst": rng.choice(POOL), "exec": self.exec_mask()}
            if kind == "atomic_rtn":
                op["op"] = rng.choice(RTN_ATOMICS)
            if kind == "cmpst":
                op["src2"] = rng.choice(POOL)
                op["cmp_hit"] = rng.random() < 0.6   # make the compare value equal the memory word for some lanes
            return op
        if kind == "swizzle":
            if rng.random() < 0.5:
                offset = 0x8000 | rng.getrandbits(8)
            else:
                offset = rng.getrandbits(15)
            return {"kind": kind, "offset": offset, "src": rng.choice(POOL), "dst": rng.choice(POOL), "exec": None}
        if kind == "bpermute":
            return {"kind": kind, "mul": rng.randrange(0, 64), "add": rng.randrange(0, 64),
                    "offset": rng.choice([0, 0, 4 * rng.randrange(0, 64)]), "src": rng.choice(POOL),
                    "dst": rng.choice(POOL), "exec": self.exec_mask()}
        raise ValueError(kind)

    def asm(self):
        w64 = self.wave == 64
        out = [f"; {self.name}: tools/lds_fuzz.py (FUZZ-6)", "; group: 64,1,1", f"; rsrc2: {RSRC2:#x}"]
        if w64:
            out.append("; mattr: +wavefrontsize64,-wavefrontsize32")
        full = "s_mov_b64 exec, -1" if w64 else "s_mov_b32 exec_lo, -1"
        out.append(full)
        for r, (k, c) in self.pool_init.items():
            out += [f"v_mul_u32_u24_e32 v{r}, {k:#x}, v0", f"v_xor_b32_e32 v{r}, {c:#x}, v{r}"]
        for j in range(LDS_DWORDS // THREADS):
            out += [f"v_add_nc_u32_e32 v{TMP}, {64 * j}, v0",
                    f"v_lshlrev_b32_e32 v{ADDR}, 2, v{TMP}",
                    f"v_mul_lo_u32 v{TMP}, v{TMP}, {self.lds_mul:#x}",
                    f"v_xor_b32_e32 v{TMP}, {self.lds_xor:#x}, v{TMP}",
                    f"ds_write_b32 v{ADDR}, v{TMP}"]
        out += ["s_waitcnt lgkmcnt(0)", "s_barrier"]
        for op in self.ops:
            out += self.asm_op(op, w64, full)
            out += ["s_waitcnt lgkmcnt(0)", "s_barrier"]
        for i, r in enumerate(POOL):
            out += [f"v_mul_u32_u24_e32 v{ADDR}, {4 * len(POOL)}, v0",
                    f"buffer_store_dword v{r}, v{ADDR}, s[0:3], 0 offen offset:{4 * i}"]
        for j in range(LDS_DWORDS // THREADS):
            out += [f"v_add_nc_u32_e32 v{TMP}, {64 * j}, v0",
                    f"v_lshlrev_b32_e32 v{TMP}, 2, v{TMP}",
                    f"ds_read_b32 v{ADDR}, v{TMP}",
                    "s_waitcnt lgkmcnt(0)",
                    f"v_add_nc_u32_e32 v{TMP}, {4 * THREADS * len(POOL)}, v{TMP}",
                    f"buffer_store_dword v{ADDR}, v{TMP}, s[0:3], 0 offen"]
        out.append("s_endpgm")
        return "\n".join(out) + "\n"

    def asm_addr(self, stride, base):
        return [f"v_mul_u32_u24_e32 v{ADDR}, {stride}, v0", f"v_add_nc_u32_e32 v{ADDR}, {base:#x}, v{ADDR}"]

    def asm_exec(self, mask, w64):
        if mask is None:
            return []
        if w64:
            return [f"s_mov_b32 exec_lo, {mask & M32:#x}", f"s_mov_b32 exec_hi, {mask >> 32:#x}"]
        return [f"s_mov_b32 exec_lo, {mask & M32:#x}"]

    def asm_op(self, op, w64, full):
        k = op["kind"]
        lines = []
        if k in ("read", "write", "atomic", "atomic_rtn", "wrxchg", "cmpst", "read2", "write2"):
            stride, base, off = op["addr"]
            lines += self.asm_addr(stride, base)
        if k == "cmpst" and op["cmp_hit"]:
            # compare value = the word currently in memory for every lane (so the store happens): read it first
            lines += [f"ds_read_b32 v{op['src']}, v{ADDR} offset:{op['addr'][2]}", "s_waitcnt lgkmcnt(0)"]
        if k == "bpermute":
            lines += [f"v_mul_u32_u24_e32 v{ADDR}, {op['mul']}, v0", f"v_add_nc_u32_e32 v{ADDR}, {op['add']}, v{ADDR}",
                      f"v_lshlrev_b32_e32 v{ADDR}, 2, v{ADDR}"]
        lines += self.asm_exec(op["exec"], w64)
        if k == "read":
            size = READS[op["name"]][0]
            n = (size + 3) // 4 if size >= 4 else 1
            dst = f"v{op['dst']}" if n == 1 else f"v[{op['dst']}:{op['dst'] + n - 1}]"
            lines.append(f"{op['name']} {dst}, v{ADDR} offset:{op['addr'][2]}")
        elif k == "write":
            size = WRITES[op["name"]][0]
            n = (size + 3) // 4 if size >= 4 else 1
            src = f"v{op['src']}" if n == 1 else f"v[{op['src']}:{op['src'] + n - 1}]"
            lines.append(f"{op['name']} v{ADDR}, {src} offset:{op['addr'][2]}")
        elif k == "read2":
            nm = "ds_read2st64_b32" if op["st64"] else "ds_read2_b32"
            lines.append(f"{nm} v[{op['dst']}:{op['dst'] + 1}], v{ADDR} offset0:{op['o0']} offset1:{op['o1']}")
        elif k == "write2":
            nm = "ds_write2st64_b32" if op["st64"] else "ds_write2_b32"
            lines.append(f"{nm} v{ADDR}, v{op['src0']}, v{op['src1']} offset0:{op['o0']} offset1:{op['o1']}")
        elif k == "atomic":
            lines.append(f"{ATOMIC_NAME[op['op']] % ''} v{ADDR}, v{op['src']} offset:{op['addr'][2]}")
        elif k == "atomic_rtn":
            lines.append(f"{ATOMIC_NAME[op['op']] % '_rtn'} v{op['dst']}, v{ADDR}, v{op['src']} offset:{op['addr'][2]}")
        elif k == "wrxchg":
            lines.append(f"ds_wrxchg_rtn_b32 v{op['dst']}, v{ADDR}, v{op['src']} offset:{op['addr'][2]}")
        elif k == "cmpst":
            lines.append(f"ds_cmpst_rtn_b32 v{op['dst']}, v{ADDR}, v{op['src']}, v{op['src2']} offset:{op['addr'][2]}")
        elif k == "swizzle":
            lines.append(f"ds_swizzle_b32 v{op['dst']}, v{op['src']} offset:{op['offset']:#x}")
        elif k == "bpermute":
            lines.append(f"ds_bpermute_b32 v{op['dst']}, v{ADDR}, v{op['src']} offset:{op['offset']}")
        if op["exec"] is not None:
            lines.append(full)
        return lines

    def model(self):
        """Run the program on the reference model. Returns (regs[t][r], lds bytearray)."""
        regs = [{r: ((t * k) & M32) ^ c for r, (k, c) in self.pool_init.items()} for t in range(THREADS)]
        lds = bytearray(LDS_BYTES)
        for i in range(LDS_DWORDS):
            struct.pack_into("<I", lds, 4 * i, ((i * self.lds_mul) & M32) ^ self.lds_xor)
        self.writer = [{r: None for r in POOL} for _ in range(THREADS)]
        for i, op in enumerate(self.ops):
            if op["kind"] == "cmpst" and op["cmp_hit"]:
                for t in range(THREADS):
                    self.writer[t][op["src"]] = i
            self.model_op(op, regs, lds)
            for t in range(THREADS):
                if self.active(t, op["exec"]):
                    for r in self.dst_regs(op):
                        self.writer[t][r] = i
        return regs, lds

    @staticmethod
    def dst_regs(op):
        k = op["kind"]
        if k == "read":
            size = READS[op["name"]][0]
            return range(op["dst"], op["dst"] + (size // 4 if size >= 4 else 1))
        if k == "read2":
            return range(op["dst"], op["dst"] + 2)
        if k in ("atomic_rtn", "wrxchg", "cmpst", "swizzle", "bpermute"):
            return (op["dst"],)
        return ()

    def describe(self, i):
        if i is None:
            return "initial value"
        op = self.ops[i]
        what = op.get("name") or op.get("op") or ("st64" if op.get("st64") else "")
        mask = "" if op["exec"] is None else f", EXEC {op['exec']:#x}"
        return " ".join(x for x in (f"op {i}", op["kind"], what) if x) + mask

    def active(self, t, mask):
        if mask is None:
            return True
        if self.wave == 64:
            return bool((mask >> t) & 1)
        return bool((mask >> (t & 31)) & 1)

    def wave_lane(self, t):
        return t & (self.wave - 1)

    def model_op(self, op, regs, lds):
        k = op["kind"]
        rd = lambda a: struct.unpack_from("<I", lds, a)[0]               # noqa: E731
        act = [self.active(t, op["exec"]) for t in range(THREADS)]
        if "addr" in op:
            stride, base, off = op["addr"]
            addr = [t * stride + base + off for t in range(THREADS)]
        if k == "read":
            size, how = READS[op["name"]]
            new = []
            for t in range(THREADS):
                if not act[t]:
                    new.append(None)
                    continue
                a = addr[t]
                if size >= 4:
                    new.append([rd(a + 4 * i) for i in range(size // 4)])
                    continue
                raw = int.from_bytes(lds[a:a + size], "little")
                bits = 8 * size
                sext = raw - (1 << bits) if raw >> (bits - 1) else raw
                old = regs[t][op["dst"]]
                if how == "u":
                    v = raw
                elif how == "i":
                    v = sext & M32
                elif how in ("u_lo", "i_lo"):
                    v = (old & 0xFFFF0000) | ((raw if how == "u_lo" else sext) & 0xFFFF)
                else:
                    v = (old & 0xFFFF) | (((raw if how == "u_hi" else sext) & 0xFFFF) << 16)
                new.append([v])
            for t in range(THREADS):
                if new[t] is not None:
                    for i, v in enumerate(new[t]):
                        regs[t][op["dst"] + i] = v
        elif k == "write":
            size, shift = WRITES[op["name"]]
            for t in range(THREADS):
                if not act[t]:
                    continue
                if size >= 4:
                    for i in range(size // 4):
                        struct.pack_into("<I", lds, addr[t] + 4 * i, regs[t][op["src"] + i])
                else:
                    v = (regs[t][op["src"]] >> shift) & ((1 << (8 * size)) - 1)
                    lds[addr[t]:addr[t] + size] = v.to_bytes(size, "little")
        elif k in ("read2", "write2"):
            unit = 256 if op["st64"] else 4
            stride, base, _ = op["addr"]
            for t in range(THREADS):
                if not act[t]:
                    continue
                a = t * stride + base
                a0, a1 = a + op["o0"] * unit, a + op["o1"] * unit
                if k == "read2":
                    regs[t][op["dst"]], regs[t][op["dst"] + 1] = rd(a0), rd(a1)
                else:
                    struct.pack_into("<I", lds, a0, regs[t][op["src0"]])
                    struct.pack_into("<I", lds, a1, regs[t][op["src1"]])
        elif k == "atomic":
            for t in range(THREADS):
                if act[t]:
                    struct.pack_into("<I", lds, addr[t], atomic(op["op"], rd(addr[t]), regs[t][op["src"]]))
        elif k in ("atomic_rtn", "wrxchg", "cmpst"):
            if k == "cmpst" and op["cmp_hit"]:
                for t in range(THREADS):          # the ds_read_b32 before the EXEC change runs in every lane
                    regs[t][op["src"]] = rd(addr[t])
            ret = {}
            for t in range(THREADS):
                if not act[t]:
                    continue
                old = rd(addr[t])
                data = regs[t][op["src"]]
                if k == "atomic_rtn":
                    new = atomic(op["op"], old, data)
                elif k == "wrxchg":
                    new = data
                else:
                    new = regs[t][op["src2"]] if old == data else old
                struct.pack_into("<I", lds, addr[t], new)
                ret[t] = old
            for t, v in ret.items():
                regs[t][op["dst"]] = v
        elif k == "swizzle":
            src = [regs[t][op["src"]] for t in range(THREADS)]
            for t in range(THREADS):
                wl = self.wave_lane(t)
                regs[t][op["dst"]] = src[t - wl + swizzle_src(wl, op["offset"])]
        elif k == "bpermute":
            # each 32-lane group (a wave32, or a half of a wave64) reads its own lanes; an inactive source gives 0
            src = [regs[t][op["src"]] for t in range(THREADS)]
            new = {}
            for t in range(THREADS):
                if act[t]:
                    s = (t & ~31) | ((((t * op["mul"] + op["add"]) * 4 + op["offset"]) >> 2) & 31)
                    new[t] = src[s] if act[s] else 0
            for t, v in new.items():
                regs[t][op["dst"]] = v
        else:
            raise ValueError(k)


def programs(seed, count, waves, no_exec=()):
    return [Program(seed, i, w, no_exec) for i in range(count) for w in waves]


def parse_dump(data):
    words = struct.unpack(f"<{len(data) // 4}I", data)
    regs = [{r: words[t * len(POOL) + i] for i, r in enumerate(POOL)} for t in range(THREADS)]
    lds = words[THREADS * len(POOL):THREADS * len(POOL) + LDS_DWORDS]
    return regs, lds


def compare(p, got_regs, got_lds):
    want_regs, want_lds = p.model()
    diffs = []
    for t in range(THREADS):
        for r in POOL:
            if got_regs[t][r] != want_regs[t][r]:
                diffs.append(f"thread {t} v{r}: got {got_regs[t][r]:#010x}, want {want_regs[t][r]:#010x}"
                             f" (last written by {p.describe(p.writer[t][r])})")
    for i in range(LDS_DWORDS):
        w = struct.unpack_from("<I", want_lds, 4 * i)[0]
        if got_lds[i] != w:
            diffs.append(f"LDS dword {i} (byte {4 * i:#x}): got {got_lds[i]:#010x}, want {w:#010x}")
    return diffs


def translate(progs, work, emulate):
    disasm, fssm, spirv = (work / d for d in ("disasm", "fssm", "spirv"))
    for d in (disasm, fssm, spirv):   # a reused --work must not leave an earlier run's listings or SPIR-V behind
        if d.exists():
            shutil.rmtree(d)
    for d in (disasm, fssm):
        d.mkdir(parents=True)
    mc = mc_command.mc_command()
    for p in progs:
        text = p.asm()
        asm_path = work / (p.name + ".asm")
        asm_path.write_text(text, encoding="utf-8")
        cmd = mc + ["-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-show-encoding"]
        if p.wave == 64:
            cmd.append("-mattr=+wavefrontsize64,-wavefrontsize32")
        r = subprocess.run(cmd, input=text, capture_output=True, text=True)
        if r.returncode or "error:" in r.stderr:
            raise RuntimeError(f"{p.name} did not assemble:\n{r.stderr}")
        code = "".join(ln for ln in r.stdout.splitlines(keepends=True) if not ln.lstrip().startswith("."))
        (disasm / (p.name + ".s")).write_text(f"; {p.name}\n{code}", encoding="utf-8")
        (fssm / (p.name + ".fssm")).write_bytes(compute_header(asm_path))
    cmd = [sys.executable, "-I", os.path.join(HERE, "rdna2spirv.py"), str(disasm), "--out", str(spirv)]
    if emulate:
        cmd.append("--wave64-emulate")
    r = subprocess.run(cmd, capture_output=True, text=True)
    index = spirv / "index.json"
    if not index.exists():
        raise RuntimeError(f"rdna2spirv.py exited {r.returncode} without an index.json:\n{(r.stdout + r.stderr)[-2000:]}")
    return json.loads(index.read_text(encoding="utf-8")) if index.exists() else {}


def run_gpu(p, spv, build, index_entry, emulate, dump):
    exe = os.path.join(build, "vk_compute_test" + (".exe" if os.name == "nt" else ""))
    env = dict(os.environ, VKC_OUT_WORDS=str(OUT_WORDS), VKC_DUMP=str(dump))
    args = [exe, str(spv), "1", "dump"]
    if p.wave == 64 and not emulate:
        args.append("--require-wave64")
    elif p.wave == 32 or emulate:
        env["VKC_SUBGROUP_SIZE"] = "32"
    if index_entry.get("subgroup_size") and not emulate:
        env["VKC_SUBGROUP_SIZE"] = str(index_entry["subgroup_size"])
    r = subprocess.run(args, capture_output=True, text=True, env=env)
    return r.returncode, (r.stdout + r.stderr).strip()


def show(seed, index, wave, no_exec=()):
    p = Program(seed, index, wave, no_exec)
    print(p.asm())
    regs, _ = p.model()
    print("; model, thread 0:", " ".join(f"v{r}={regs[0][r]:#x}" for r in POOL))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--programs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--wave", choices=("64", "32", "both"), default="both")
    ap.add_argument("--emulate", action="store_true", help="translate with --wave64-emulate, run at subgroup 32")
    ap.add_argument("--build", default=os.path.join(ROOT, "runtime", "build"))
    ap.add_argument("--work", default=None, help="keep the generated .asm and SPIR-V here")
    ap.add_argument("--model-only", action="store_true")
    ap.add_argument("--show", default=None, metavar="SEED:INDEX[:WAVE]")
    ap.add_argument("--max-report", type=int, default=12)
    ap.add_argument("--no-exec", action="append", default=[], choices=EXEC_KINDS,
                    help="run this op kind (e.g. bpermute) with all lanes active, to isolate a known finding")
    a = ap.parse_args()
    if a.programs < 1 or a.max_report < 1:
        ap.error("--programs and --max-report must be at least 1")
    if a.show is not None:
        parts = a.show.split(":")
        try:
            if len(parts) not in (2, 3):
                raise ValueError
            seed, index = int(parts[0]), int(parts[1])
            wave = int(parts[2]) if len(parts) == 3 else 64
        except ValueError:
            ap.error("--show expects SEED:INDEX[:WAVE] with INDEX >= 0 and WAVE in 32, 64")
        if index < 0 or wave not in (32, 64):
            ap.error("--show expects SEED:INDEX[:WAVE] with INDEX >= 0 and WAVE in 32, 64")
        show(seed, index, wave, tuple(a.no_exec))
        return 0
    waves = (64, 32) if a.wave == "both" else (int(a.wave),)
    progs = programs(a.seed, a.programs, waves, tuple(a.no_exec))
    if a.model_only:
        for p in progs:
            p.model()
        print(f"{len(progs)} programs generated and modelled")
        return 0
    work = Path(a.work).resolve() if a.work else Path(tempfile.mkdtemp(prefix="lds_fuzz_"))
    work.mkdir(parents=True, exist_ok=True)
    try:
        index = translate(progs, work, a.emulate)
    except RuntimeError as e:
        print(e)
        return 2
    failed = untranslated = skipped = errors = 0
    no_exec = "".join(f" --no-exec {k}" for k in a.no_exec)
    for p in progs:
        spv = work / "spirv" / (p.name + ".spv")
        if p.name not in index or not spv.is_file():
            untranslated += 1
            print(f"NOT TRANSLATED {p.name}")
            continue
        dump = work / (p.name + ".bin")
        dump.unlink(missing_ok=True)
        rc, msg = run_gpu(p, spv, a.build, index[p.name], a.emulate, dump)
        if rc == 77:
            skipped += 1
            print(f"SKIP {p.name}: {msg.splitlines()[-1] if msg else 'unsupported'}")
            continue
        data = dump.read_bytes() if dump.is_file() else b""
        if rc != 0 or len(data) != 4 * OUT_WORDS:   # dump mode itself never fails a run
            errors += 1
            print(f"ERROR {p.name}: exit {rc}, {len(data)} of {4 * OUT_WORDS} dump bytes: {msg[-400:]}")
            continue
        regs, lds = parse_dump(data)
        diffs = compare(p, regs, lds)
        if diffs:
            failed += 1
            print(f"MISMATCH {p.name} ({len(diffs)} words); print it with --show {a.seed}:"
                  f"{p.name.split('_')[3]}:{p.wave}{no_exec}")
            for d in diffs[:a.max_report]:
                print("   ", d)
    mode = "emulated" if a.emulate else "native"
    ran = len(progs) - untranslated - skipped - errors
    print(f"{ran - failed}/{ran} programs that ran match ({mode}); {failed} mismatched, {skipped} skipped, "
          f"{untranslated} not translated, {errors} harness errors; work dir {work}")
    if errors or ran == 0:
        return 2
    return 1 if failed or untranslated else 0


if __name__ == "__main__":
    sys.exit(main())
