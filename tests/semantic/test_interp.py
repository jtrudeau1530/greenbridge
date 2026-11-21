"""Self-tests for the mini SPIR-V interpreter, using hand-written modules and expectations taken from the SPIR-V
and GLSL.std.450 specifications (not from the translator)."""
import unittest

import spirv_mini_interp as si

HEADER = """\
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%bool = OpTypeBool
%v2float = OpTypeVector %float 2
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c0 = OpConstant %uint 0
%c1 = OpConstant %uint 1
%c2 = OpConstant %uint 2
%c4 = OpConstant %uint 4
%c8 = OpConstant %uint 8
%c32 = OpConstant %uint 32
%cF0 = OpConstant %uint 4294967280
%cFF = OpConstant %uint 4294967295
%c80 = OpConstant %uint 128
%ch = OpConstant %uint 1073757184
%f1 = OpConstant %float 1
%f2 = OpConstant %float 2
%f3 = OpConstant %float 3
%f70000 = OpConstant %float 70000
%t = OpConstantTrue %bool
%f = OpConstantFalse %bool
%_ptr_Function_uint = OpTypePointer Function %uint
%arr = OpTypeArray %uint %c4
%_ptr_Function_arr = OpTypePointer Function %arr
%null_arr = OpConstantNull %arr
%ud = OpTypeStruct %uint
%umul = OpTypeStruct %uint %uint
%_ptr_PC = OpTypePointer PushConstant %ud
%pc = OpVariable %_ptr_PC PushConstant
%_ptr_PC_uint = OpTypePointer PushConstant %uint
%main = OpFunction %void None %fn
%entry = OpLabel
"""
FOOTER = "OpReturn\nOpFunctionEnd\n"


def run_word(body):
    """Run `body` (instructions that must define %res as a uint) and return the word stored to %out."""
    return si.run_fragment(HEADER + body + "OpStore %out %res\n" + FOOTER)["%out"][0]


class InterpreterTests(unittest.TestCase):
    def test_bitfield_extract(self):
        # SPIR-V: Count 0 gives 0; SExtract replicates the field's top bit; UExtract zero-extends.
        self.assertEqual(run_word("%res = OpBitFieldSExtract %uint %cF0 %c2 %c4\n"), 0xFFFFFFFC)   # 0b1100 -> -4
        self.assertEqual(run_word("%res = OpBitFieldUExtract %uint %cF0 %c2 %c4\n"), 0xC)
        self.assertEqual(run_word("%res = OpBitFieldSExtract %uint %c80 %c0 %c8\n"), 0xFFFFFF80)
        self.assertEqual(run_word("%res = OpBitFieldSExtract %uint %cFF %c2 %c0\n"), 0)

    def test_bitfield_insert(self):
        self.assertEqual(run_word("%res = OpBitFieldInsert %uint %cF0 %c1 %c0 %c4\n"), 0xFFFFFFF1)

    def test_half_pack_unpack(self):
        self.assertEqual(run_word("%v = OpCompositeConstruct %v2float %f1 %f2\n"
                                  "%res = OpExtInst %uint %glsl PackHalf2x16 %v\n"), 0x40003C00)
        self.assertEqual(run_word("%v = OpCompositeConstruct %v2float %f70000 %f1\n"
                                  "%res = OpExtInst %uint %glsl PackHalf2x16 %v\n"), 0x3C007C00)  # 70000 -> +inf
        self.assertEqual(run_word("%u = OpExtInst %v2float %glsl UnpackHalf2x16 %ch\n"          # %ch = 0x40003c00
                                  "%x = OpCompositeExtract %float %u 1\n"
                                  "%res = OpBitcast %uint %x\n"), 0x40000000)                   # 2.0f

    def test_fma_select_bitcast(self):
        self.assertEqual(run_word("%x = OpExtInst %float %glsl Fma %f2 %f3 %f1\n%res = OpBitcast %uint %x\n"),
                         0x40E00000)                                                              # 7.0f
        self.assertEqual(run_word("%res = OpSelect %uint %t %c1 %c2\n"), 1)
        self.assertEqual(run_word("%res = OpSelect %uint %f %c1 %c2\n"), 2)

    def test_signed_unsigned_min_max(self):
        self.assertEqual(run_word("%res = OpExtInst %uint %glsl SMax %cFF %c1\n"), 1)        # -1 vs 1 signed
        self.assertEqual(run_word("%res = OpExtInst %uint %glsl UMin %cFF %c1\n"), 1)        # 2^32-1 vs 1 unsigned

    def test_integer_ops_wrap(self):
        self.assertEqual(run_word("%res = OpIAdd %uint %cFF %c2\n"), 1)
        self.assertEqual(run_word("%res = OpISub %uint %c1 %c2\n"), 0xFFFFFFFF)
        self.assertEqual(run_word("%res = OpIMul %uint %cFF %c2\n"), 0xFFFFFFFE)
        self.assertEqual(run_word("%res = OpShiftLeftLogical %uint %cFF %c4\n"), 0xFFFFFFF0)
        self.assertEqual(run_word("%res = OpShiftRightLogical %uint %cF0 %c4\n"), 0x0FFFFFFF)
        self.assertEqual(run_word("%res = OpBitwiseXor %uint %cF0 %cFF\n"), 0xF)
        self.assertEqual(run_word("%res = OpNot %uint %cF0\n"), 0xF)
        self.assertEqual(run_word("%res = OpBitCount %uint %cFF\n"), 32)
        self.assertEqual(run_word("%res = OpBitCount %uint %c0\n"), 0)

    def test_float_ops(self):
        self.assertEqual(run_word("%x = OpFMul %float %f2 %f3\n%res = OpBitcast %uint %x\n"), 0x40C00000)
        self.assertEqual(run_word("%x = OpFSub %float %f2 %f3\n%res = OpBitcast %uint %x\n"), 0xBF800000)
        self.assertEqual(run_word("%x = OpFAdd %float %f1 %f2\n%res = OpBitcast %uint %x\n"), 0x40400000)
        self.assertEqual(run_word("%x = OpFNegate %float %f3\n%res = OpBitcast %uint %x\n"), 0xC0400000)
        self.assertEqual(run_word("%b = OpFOrdEqual %bool %f2 %f2\n%res = OpSelect %uint %b %c1 %c0\n"), 1)
        self.assertEqual(run_word("%b = OpFOrdLessThan %bool %f1 %f2\n%res = OpSelect %uint %b %c1 %c0\n"), 1)
        self.assertEqual(run_word("%b = OpFUnordNotEqual %bool %f1 %f1\n"
                                  "%res = OpSelect %uint %b %c1 %c0\n"), 0)

    def test_64bit_arithmetic(self):
        text = HEADER.replace("%c0 = OpConstant %uint 0",
                              "%ulong = OpTypeInt 64 0\n"
                              "%c64_1 = OpConstant %ulong 1\n"
                              "%c64_32 = OpConstant %ulong 32\n"
                              "%c0 = OpConstant %uint 0")
        text += ("%sh = OpShiftLeftLogical %ulong %c64_1 %c64_32\n"
                 "%sum = OpIAdd %ulong %sh %c64_1\n"
                 "%prod = OpIMul %ulong %sum %c64_1\n"
                 "%res = OpUConvert %uint %prod\n" + "OpStore %out %res\n" + FOOTER)
        self.assertEqual(si.run_fragment(text)["%out"][0], 1)

    def test_64bit_pointer_load(self):
        text = HEADER.replace("OpMemoryModel Logical GLSL450",
                              'OpCapability PhysicalStorageBufferAddresses\n'
                              'OpExtension "SPV_KHR_physical_storage_buffer"\n'
                              "OpMemoryModel PhysicalStorageBuffer64 GLSL450")
        decls = ("%ulong = OpTypeInt 64 0\n"
                 "%rt = OpTypeRuntimeArray %uint\n"
                 "%_ptr_PSB_rt = OpTypePointer PhysicalStorageBuffer %rt\n"
                 "%_ptr_PSB_uint = OpTypePointer PhysicalStorageBuffer %uint\n"
                 "%addr = OpConstant %ulong 4096\n")
        text = text.replace("%main = OpFunction", decls + "%main = OpFunction")
        text += ("%p = OpConvertUToPtr %_ptr_PSB_rt %addr\n"
                 "%q = OpAccessChain %_ptr_PSB_uint %p %c1\n"
                 "%res = OpLoad %uint %q\n"
                 "OpStore %out %res\n" + FOOTER)
        self.assertEqual(si.run_fragment(text, memory={4100: 0x12345678})["%out"][0], 0x12345678)

    def test_inputs_and_push_constants_can_be_seeded(self):
        text = HEADER.replace('OpEntryPoint Fragment %main "main" %out',
                              'OpEntryPoint Fragment %main "main" %out %in')
        decls = ("%v4float = OpTypeVector %float 4\n"
                 "%_ptr_Input_v4 = OpTypePointer Input %v4float\n"
                 "%_ptr_Input_float = OpTypePointer Input %float\n"
                 "%in = OpVariable %_ptr_Input_v4 Input\n")
        text = text.replace("%main = OpFunction", decls + "%main = OpFunction")
        text += ("%ip = OpAccessChain %_ptr_Input_float %in %c2\n"
                 "%iv = OpLoad %float %ip\n"
                 "%ib = OpBitcast %uint %iv\n"
                 "%pp = OpAccessChain %_ptr_PC_uint %pc %c0\n"
                 "%pv = OpLoad %uint %pp\n"
                 "%res = OpIAdd %uint %ib %pv\n"
                 "OpStore %out %res\n" + FOOTER)
        out = si.run_fragment(text, inputs={"%in": (0.0, 0.0, 1.5, 0.0)}, push_constants=[0x100])
        self.assertEqual(out["%out"][0], 0x3FC00100)

    def test_sub_borrow_and_comparisons(self):
        self.assertEqual(run_word("%s = OpISubBorrow %umul %c1 %c2\n"
                                  "%d = OpCompositeExtract %uint %s 0\n"
                                  "%b = OpCompositeExtract %uint %s 1\n"
                                  "%res = OpBitwiseAnd %uint %d %b\n"), 1)
        self.assertEqual(run_word("%s = OpISubBorrow %umul %c2 %c1\n"
                                  "%res = OpCompositeExtract %uint %s 1\n"), 0)
        self.assertEqual(run_word("%b = OpIEqual %bool %c2 %c2\n%res = OpSelect %uint %b %c1 %c0\n"), 1)
        self.assertEqual(run_word("%b = OpINotEqual %bool %c2 %c1\n%res = OpSelect %uint %b %c1 %c0\n"), 1)
        self.assertEqual(run_word("%b = OpSLessThan %bool %cFF %c0\n%res = OpSelect %uint %b %c1 %c0\n"), 1)
        self.assertEqual(run_word("%b = OpLogicalAnd %bool %t %f\n%res = OpSelect %uint %b %c1 %c2\n"), 2)
        self.assertEqual(run_word("%b = OpLogicalOr %bool %f %t\n%res = OpSelect %uint %b %c1 %c2\n"), 1)

    def test_float_mul_ord_equal_findsmsb_sin(self):
        self.assertEqual(run_word("%x = OpFMul %float %f2 %f3\n%res = OpBitcast %uint %x\n"), 0x40C00000)
        self.assertEqual(run_word("%b = OpFOrdEqual %bool %f2 %f2\n%res = OpSelect %uint %b %c1 %c0\n"), 1)
        self.assertEqual(run_word("%nf = OpBitcast %float %cFF\n"
                                  "%b = OpFOrdEqual %bool %nf %nf\n%res = OpSelect %uint %b %c1 %c0\n"), 0)
        self.assertEqual(run_word("%res = OpExtInst %uint %glsl FindSMsb %c0\n"), 0xFFFFFFFF)
        self.assertEqual(run_word("%res = OpExtInst %uint %glsl FindSMsb %cFF\n"), 0xFFFFFFFF)
        self.assertEqual(run_word("%res = OpExtInst %uint %glsl FindSMsb %c80\n"), 7)
        self.assertEqual(run_word("%res = OpExtInst %uint %glsl FindSMsb %cF0\n"), 3)
        self.assertEqual(run_word("%x = OpBitcast %float %c0\n%s = OpExtInst %float %glsl Sin %x\n"
                                  "%res = OpBitcast %uint %s\n"), 0)

    def test_forward_branch_and_kill(self):
        live = (HEADER + "OpStore %out %c1\nOpSelectionMerge %live None\n"
                "OpBranchConditional %t %live %dead\n%dead = OpLabel\nOpKill\n%live = OpLabel\n" + FOOTER)
        dead = (HEADER + "OpStore %out %c1\nOpSelectionMerge %live None\n"
                "OpBranchConditional %f %live %dead\n%dead = OpLabel\nOpKill\n%live = OpLabel\n" + FOOTER)
        self.assertEqual(si.run_fragment(live)["%out"][0], 1)
        self.assertEqual(si.run_fragment(dead), {})

    def test_phi_selects_the_value_from_the_taken_predecessor(self):
        for cond, want in (("%t", 1), ("%f", 2)):
            text = (HEADER + f"OpSelectionMerge %merge None\nOpBranchConditional {cond} %a %b\n"
                    "%a = OpLabel\nOpBranch %merge\n"
                    "%b = OpLabel\nOpBranch %merge\n"
                    "%merge = OpLabel\n%res = OpPhi %uint %c1 %a %c2 %b\n"
                    "OpStore %out %res\n" + FOOTER)
            with self.subTest(cond=cond):
                self.assertEqual(si.run_fragment(text)["%out"][0], want)

    def test_function_array_access(self):
        self.assertEqual(run_word("%arr_v = OpVariable %_ptr_Function_arr Function %null_arr\n"
                                  "%p = OpAccessChain %_ptr_Function_uint %arr_v %c2\n"
                                  "OpStore %p %cF0\n"
                                  "%q = OpAccessChain %_ptr_Function_uint %arr_v %c2\n"
                                  "%res = OpLoad %uint %q\n"), 0xFFFFFFF0)

    def test_unsupported_constructs_are_rejected(self):
        for body in ("%res = OpUDiv %uint %c2 %c1\n",                                     # opcode not modelled
                     "%res = OpExtInst %uint %glsl FindUMsb %c2\n",                      # ext inst not modelled
                     "%res = OpBitFieldUExtract %uint %cFF %c32 %c1\n",                  # undefined per spec
                     "%res = OpShiftLeftLogical %uint %cFF %c32\n",                     # undefined per spec
                     "%res = OpShiftRightLogical %uint %cFF %c32\n",                    # undefined per spec
                     "%res = OpSelect %uint %c1 %c1 %c2\n",                              # condition not a bool
                     "OpBranch %entry\n%res = OpCopyObject %uint %c1\n",                 # unconditional branch
                     "OpBranchConditional %t %entry %entry\n%res = OpCopyObject %uint %c1\n"):  # backward branch
            with self.subTest(body=body.splitlines()[0]):
                with self.assertRaises(si.UnsupportedConstruct):
                    run_word(body)

    def test_store_to_push_constant_is_rejected(self):
        text = HEADER + "%p = OpAccessChain %_ptr_PC_uint %pc %c0\nOpStore %p %c1\n" + FOOTER
        with self.assertRaises(si.UnsupportedConstruct):
            si.run_fragment(text)

    def test_push_constant_reads_zero(self):
        text = HEADER + "%p = OpAccessChain %_ptr_PC_uint %pc %c0\n%res = OpLoad %uint %p\nOpStore %out %res\n" + FOOTER
        self.assertEqual(si.run_fragment(text)["%out"][0], 0)


if __name__ == "__main__":
    unittest.main()
