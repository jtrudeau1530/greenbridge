"""Tests of tests/semantic/spirv_wave_interp.py against hand-written SPIR-V (standard library only).

The interpreter is the SPIR-V side of tools/cf_diff.py, so its convergence rules are pinned here on modules
written by hand, with the expected words derived from the SPIR-V specification in each test:
  * a subgroup operation inside a selection sees only the invocations that took that side;
  * after the merge block every invocation is back and sees the whole subgroup;
  * an invocation that leaves a loop stops taking part in the loop's subgroup operations;
  * OpKill removes an invocation (no outputs, no later participation);
  * an undefined value (a shift by the bit width) may be computed and discarded, but not branched on.

    python -I tests/semantic/test_spirv_wave_interp.py
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from spirv_wave_interp import UnsupportedConstruct, run_wave  # noqa: E402

HEAD = """OpCapability Shader
OpCapability GroupNonUniformBallot
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %lane %out
OpDecorate %lane BuiltIn SubgroupLocalInvocationId
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%bool = OpTypeBool
%v4uint = OpTypeVector %uint 4
%pin = OpTypePointer Input %uint
%pout = OpTypePointer Output %v4uint
%lane = OpVariable %pin Input
%out = OpVariable %pout Output
%true = OpConstantTrue %bool
%false = OpConstantFalse %bool
%c0 = OpConstant %uint 0
%c1 = OpConstant %uint 1
%c3 = OpConstant %uint 3
%c8 = OpConstant %uint 8
%c16 = OpConstant %uint 16
%c32 = OpConstant %uint 32
%main = OpFunction %void None %fn
%entry = OpLabel
%l = OpLoad %uint %lane
"""
TAIL = "OpReturn\nOpFunctionEnd\n"


def outs(body, wave=32):
    return [o["%out"] if o is not None else None for o in run_wave(HEAD + body + TAIL, wave=wave)["outputs"]]


class SelectionConvergence(unittest.TestCase):
    BODY = """%lt = OpULessThan %bool %l %c8
OpSelectionMerge %merge None
OpBranchConditional %lt %then %merge
%then = OpLabel
%b1 = OpGroupNonUniformBallot %v4uint %c3 %true
%x = OpCompositeExtract %uint %b1 0
OpBranch %merge
%merge = OpLabel
%p = OpPhi %uint %x %then %c0 %entry
%b2 = OpGroupNonUniformBallot %v4uint %c3 %true
%y = OpCompositeExtract %uint %b2 0
%z = OpCompositeExtract %uint %b2 1
%v = OpCompositeConstruct %v4uint %p %y %z %l
OpStore %out %v
"""

    def test_wave32(self):
        # Inside the selection only lanes 0-7 run: their ballot is 0xff. After the merge all 32 lanes are back.
        got = outs(self.BODY, 32)
        for lane in range(32):
            self.assertEqual(got[lane], (0xFF if lane < 8 else 0, 0xFFFFFFFF, 0, lane))

    def test_wave64(self):
        got = outs(self.BODY, 64)
        for lane in range(64):
            self.assertEqual(got[lane], (0xFF if lane < 8 else 0, 0xFFFFFFFF, 0xFFFFFFFF, lane))


class LoopConvergence(unittest.TestCase):
    # Lane L runs (L & 3) + 1 iterations. Each iteration adds the number of lanes still in the loop, so with
    # 32 lanes: iteration 0 has 32, 1 has 24 (L & 3 >= 1), 2 has 16, 3 has 8.
    BODY = """OpBranch %head
%head = OpLabel
%i = OpPhi %uint %c0 %entry %inext %cont
%acc = OpPhi %uint %c0 %entry %accn %cont
OpLoopMerge %end %cont None
OpBranch %body
%body = OpLabel
%bal = OpGroupNonUniformBallot %v4uint %c3 %true
%n = OpGroupNonUniformBallotBitCount %uint %c3 Reduce %bal
%accn = OpIAdd %uint %acc %n
%inext = OpIAdd %uint %i %c1
%lim = OpBitwiseAnd %uint %l %c3
%go = OpULessThanEqual %bool %inext %lim
OpBranchConditional %go %cont %end
%cont = OpLabel
OpBranch %head
%end = OpLabel
%after = OpGroupNonUniformBallot %v4uint %c3 %true
%all = OpCompositeExtract %uint %after 0
%v = OpCompositeConstruct %v4uint %accn %inext %all %l
OpStore %out %v
"""

    def test_lanes_leave_the_loop_one_by_one(self):
        expect = {0: 32, 1: 32 + 24, 2: 32 + 24 + 16, 3: 32 + 24 + 16 + 8}
        got = outs(self.BODY, 32)
        for lane in range(32):
            self.assertEqual(got[lane], (expect[lane & 3], (lane & 3) + 1, 0xFFFFFFFF, lane))


class Kill(unittest.TestCase):
    BODY = """%hi = OpUGreaterThanEqual %bool %l %c16
OpSelectionMerge %merge None
OpBranchConditional %hi %dead %merge
%dead = OpLabel
OpKill
%merge = OpLabel
%b = OpGroupNonUniformBallot %v4uint %c3 %true
%x = OpCompositeExtract %uint %b 0
%v = OpCompositeConstruct %v4uint %x %l %c0 %c0
OpStore %out %v
"""

    def test_killed_lanes_write_nothing_and_leave_the_subgroup(self):
        got = outs(self.BODY, 32)
        for lane in range(32):
            self.assertEqual(got[lane], (0xFFFF, lane, 0, 0) if lane < 16 else None)


class UndefinedValues(unittest.TestCase):
    def test_discarded_shift_by_width_is_fine(self):
        body = """%s = OpShiftLeftLogical %uint %c1 %c32
%m = OpISub %uint %s %c1
%sel = OpSelect %uint %false %m %c3
%v = OpCompositeConstruct %v4uint %sel %c0 %c0 %c0
OpStore %out %v
"""
        self.assertEqual(outs(body)[5], (3, 0, 0, 0))

    def test_branch_on_undefined_value_raises(self):
        body = """%s = OpShiftLeftLogical %uint %c1 %c32
%z = OpIEqual %bool %s %c0
OpSelectionMerge %merge None
OpBranchConditional %z %merge %merge
%merge = OpLabel
"""
        with self.assertRaises(UnsupportedConstruct):
            outs(body)

    def test_undefined_value_in_an_output_raises(self):
        body = """%s = OpShiftLeftLogical %uint %c1 %c32
%v = OpCompositeConstruct %v4uint %s %c0 %c0 %c0
OpStore %out %v
"""
        with self.assertRaises(UnsupportedConstruct):
            outs(body)


class Strictness(unittest.TestCase):
    def test_float_arithmetic_is_rejected(self):
        body = """%f = OpBitcast %uint %c1
%g = OpFAdd %uint %f %f
"""
        with self.assertRaises(UnsupportedConstruct):
            outs(body)

    def test_split_without_merge_is_rejected(self):
        body = """%lt = OpULessThan %bool %l %c8
OpBranchConditional %lt %a %b
%a = OpLabel
OpReturn
%b = OpLabel
"""
        with self.assertRaises(UnsupportedConstruct):
            outs(body)




class WaveInterpExtensions(unittest.TestCase):
    def test_location_inputs_float_ops_and_convert(self):
        mod = """OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out %in0 %l
OpDecorate %out Location 0
OpDecorate %in0 Location 0
OpDecorate %l BuiltIn SubgroupLocalInvocationId
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%v4uint = OpTypeVector %uint 4
%_ptr_Output = OpTypePointer Output %v4uint
%_ptr_Input = OpTypePointer Input %v4float
%_ptr_Input_f = OpTypePointer Input %float
%_ptr_Builtin = OpTypePointer Input %uint
%out = OpVariable %_ptr_Output Output
%in0 = OpVariable %_ptr_Input Input
%l = OpVariable %_ptr_Builtin Input
%c0 = OpConstant %uint 0
%c1 = OpConstant %uint 1
%f64 = OpConstant %float 64.0
%main = OpFunction %void None %fn
%entry = OpLabel
%px = OpAccessChain %_ptr_Input_f %in0 %c0
%py = OpAccessChain %_ptr_Input_f %in0 %c1
%fx = OpLoad %float %px
%fy = OpLoad %float %py
%sx = OpFMul %float %fx %f64
%sy = OpFMul %float %fy %f64
%ux = OpConvertFToU %uint %sx
%uy = OpConvertFToU %uint %sy
%sum = OpFAdd %float %fx %fy
%us = OpConvertFToU %uint %sum
%v = OpCompositeConstruct %v4uint %ux %uy %us %c0
OpStore %out %v
OpReturn
OpFunctionEnd
"""
        res = run_wave(mod, wave=32, inputs={0: lambda lane: ((lane + 0.5) / 64.0, 1.5 / 64.0, 0.0, 0.0)})
        for lane in range(32):
            self.assertEqual(res["outputs"][lane]["%out"], (lane, 1, 0, 0))

    def test_physical_storage_buffer_atomic_and_store(self):
        mod = """OpCapability Shader
OpCapability Int64
OpCapability PhysicalStorageBufferAddresses
OpMemoryModel PhysicalStorageBuffer64 GLSL450
OpEntryPoint GLCompute %main "main" %out %l
OpDecorate %out Location 0
OpDecorate %l BuiltIn SubgroupLocalInvocationId
OpDecorate %arr ArrayStride 4
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%ulong = OpTypeInt 64 0
%v4uint = OpTypeVector %uint 4
%arr = OpTypeRuntimeArray %uint
%_ptr_Output = OpTypePointer Output %v4uint
%_ptr_Builtin = OpTypePointer Input %uint
%_ptr_PSB_arr = OpTypePointer PhysicalStorageBuffer %arr
%_ptr_PSB_u = OpTypePointer PhysicalStorageBuffer %uint
%out = OpVariable %_ptr_Output Output
%l = OpVariable %_ptr_Builtin Input
%c0 = OpConstant %uint 0
%c1 = OpConstant %uint 1
%c0_64 = OpConstant %ulong 0
%c64_64 = OpConstant %ulong 64
%c4_64 = OpConstant %ulong 4
%main = OpFunction %void None %fn
%entry = OpLabel
%lane = OpLoad %uint %l
%base = OpConvertUToPtr %_ptr_PSB_arr %c0_64
%p0 = OpAccessChain %_ptr_PSB_u %base %c0
%slot = OpAtomicIAdd %uint %p0 %c1 %c0 %c1
%slot64 = OpUConvert %ulong %slot
%off = OpIMul %ulong %slot64 %c4_64
%addr = OpIAdd %ulong %c64_64 %off
%sptr_arr = OpConvertUToPtr %_ptr_PSB_arr %addr
%sptr = OpAccessChain %_ptr_PSB_u %sptr_arr %c0
OpStore %sptr %lane Aligned 4
%v = OpCompositeConstruct %v4uint %slot %c0 %c0 %c0
OpStore %out %v
OpReturn
OpFunctionEnd
"""
        mem = {0: 0}
        res = run_wave(mod, wave=32, memory=mem)
        self.assertEqual(mem[0], 32)
        self.assertEqual({mem[64 + 4 * i] for i in range(32)}, set(range(32)))

    def test_quad_swap_and_broadcast(self):
        mod = """OpCapability Shader
OpCapability GroupNonUniformQuad
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out %l
OpDecorate %out Location 0
OpDecorate %l BuiltIn SubgroupLocalInvocationId
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%v4uint = OpTypeVector %uint 4
%_ptr_Output = OpTypePointer Output %v4uint
%_ptr_Builtin = OpTypePointer Input %uint
%out = OpVariable %_ptr_Output Output
%l = OpVariable %_ptr_Builtin Input
%c0 = OpConstant %uint 0
%c1 = OpConstant %uint 1
%c2 = OpConstant %uint 2
%c3 = OpConstant %uint 3
%main = OpFunction %void None %fn
%entry = OpLabel
%lane = OpLoad %uint %l
%sh = OpGroupNonUniformQuadSwap %uint %c3 %lane %c0
%sv = OpGroupNonUniformQuadSwap %uint %c3 %lane %c1
%sd = OpGroupNonUniformQuadSwap %uint %c3 %lane %c2
%sb = OpGroupNonUniformQuadBroadcast %uint %c3 %lane %c0
%v = OpCompositeConstruct %v4uint %sh %sv %sd %sb
OpStore %out %v
OpReturn
OpFunctionEnd
"""
        res = run_wave(mod, wave=32)
        for lane in range(32):
            self.assertEqual(res["outputs"][lane]["%out"], (lane ^ 1, lane ^ 2, lane ^ 3, lane & ~3))


if __name__ == "__main__":
    unittest.main()
