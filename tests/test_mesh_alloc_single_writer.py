"""Allocation requests from any wave publish race-free workgroup counts."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import rdna2spirv
import spirv_asm


class MeshAllocationTests(unittest.TestCase):
    def test_counts_are_published_atomically_by_the_senders(self):
        # The GPU ms_wave_info_s3 fixture sends only from wave 1. Reject a local-index-zero gate,
        # while retaining SHADER-766's requirement that concurrent senders do not race on a store.
        for mesh_compute in (False, True):
            with self.subTest(mesh_compute=mesh_compute), patch.object(rdna2spirv, "MESH_COMPUTE", mesh_compute):
                gen = rdna2spirv.Gen("ms")
                gen.ms = {"threads": 128, "maxv": 64, "maxp": 64, "clip": 0, "cull": 0, "params": []}
                gen._insn({"op": "s_sendmsg", "args": [{"k": "special", "t": "sendmsg(MSG_GS_ALLOC_REQ)"}],
                           "mods": []})
                body = "\n".join(gen.body)
                self.assertIn("OpAtomicExchange %uint %ms_counts %c2 %c0", body)
                self.assertNotIn("OpStore %ms_counts", body)
                self.assertNotIn("OpBranchConditional", body)
                self.assertTrue(gen.ms_alloc)
                # Assemble the emitted operation in a complete compute module, including its memory operands.
                module = """OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %ms_counts
OpExecutionMode %main LocalSize 128 1 1
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%c0 = OpConstant %uint 0
%c2 = OpConstant %uint 2
%_ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%_ptr_Function_uint = OpTypePointer Function %uint
%ms_counts = OpVariable %_ptr_Workgroup_uint Workgroup
%fn_t = OpTypeFunction %void
%main = OpFunction %void None %fn_t
%entry = OpLabel
%m0 = OpVariable %_ptr_Function_uint Function
OpStore %m0 %c2
""" + body + "\nOpReturn\nOpFunctionEnd\n"
                self.assertGreater(len(spirv_asm.assemble(module)), 0)


if __name__ == "__main__":
    unittest.main()
