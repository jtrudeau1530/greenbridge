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


class SwitchLiterals(unittest.TestCase):
    FRAGMENT = """OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %color
OpExecutionMode %main OriginUpperLeft
OpDecorate %color Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%int = OpTypeInt 32 1
%out_int = OpTypePointer Output %int
%color = OpVariable %out_int Output
%one = OpConstant %int 1
%two = OpConstant %int 2
%ten = OpConstant %int 10
%zero = OpConstantNull %int
%main = OpFunction %void None %fn
%entry = OpLabel
%selector = OpISub %int %one %two
OpSelectionMerge %merge None
OpSwitch %selector %default -1 %selected
%selected = OpLabel
OpStore %color %ten
OpBranch %merge
%default = OpLabel
OpStore %color %one
OpBranch %merge
%merge = OpLabel
OpReturn
OpFunctionEnd
"""
    WIDE_SELECTOR = (FRAGMENT.replace("OpCapability Shader", "OpCapability Shader\nOpCapability Int64")
                     .replace("%int = OpTypeInt 32 1", "%int = OpTypeInt 32 1\n%long = OpTypeInt 64 1")
                     .replace("%two = OpConstant %int 2", "%long_one = OpConstant %long 1\n%long_two = OpConstant %long 2")
                     .replace("OpISub %int %one %two", "OpISub %long %long_one %long_two"))

    def test_signed_literal_matches_the_selector_bits(self) -> None:
        for width, text in ((32, self.FRAGMENT), (64, self.WIDE_SELECTOR)):
            with self.subTest(width=width):
                self.assertEqual(run_wave(text)["outputs"], [{"%color": (10,)}] * 32)

    def test_constant_null_selector_still_matches_zero(self) -> None:
        text = self.FRAGMENT.replace("OpSwitch %selector %default -1", "OpSwitch %zero %default 0")
        self.assertEqual(run_wave(text)["outputs"], [{"%color": (10,)}] * 32)


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


if __name__ == "__main__":
    unittest.main()
