"""Exercise the GPU atlas with synthetic vulkaninfo reports."""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
import turing_atlas

MATRIX_KEYS = ("vulkaninfo", "device", "subgroup", "subgroup_size_control", "mesh",
               "barycentric", "maintenance8", "descriptor_heap", "storageInputOutput16")


def report():
    return {"capabilities": {"device": {
        "extensions": {"VK_EXT_mesh_shader": 1},
        "features": {},
        "properties": {
            "VkPhysicalDeviceProperties": {"deviceName": "Example GPU", "driverVersion": 7, "apiVersion": 42},
        },
    }}}


class VulkanMatrixTests(unittest.TestCase):
    def probe(self, document, returncode=0):
        def run(cmd, cwd):
            Path(cwd, "VP_VULKANINFO_Example.json").write_text(json.dumps(document), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, returncode)

        with patch.object(turing_atlas.shutil, "which", return_value="vulkaninfo"):
            with patch.object(turing_atlas, "run", run):
                return turing_atlas.vulkan_matrix()

    def assert_unknown(self, matrix):
        self.assertEqual(matrix, dict.fromkeys(MATRIX_KEYS))

    def test_recognized_profile(self):
        doc = report()
        device = doc["capabilities"]["device"]
        device["properties"]["VkPhysicalDeviceVulkan11Properties"] = {"subgroupSize": 32}
        device["properties"]["VkPhysicalDeviceVulkan13Properties"] = {"minSubgroupSize": 16, "maxSubgroupSize": 64}
        device["features"]["VkPhysicalDeviceVulkan11Features"] = {"storageInputOutput16": False}
        device["features"]["VkPhysicalDeviceVulkan13Features"] = {"subgroupSizeControl": True}
        matrix = self.probe(doc)
        self.assertEqual(tuple(matrix), MATRIX_KEYS)
        self.assertEqual(matrix["vulkaninfo"], "ok")
        self.assertEqual(matrix["device"], {"name": "Example GPU", "driver": 7, "api": 42})
        self.assertEqual(matrix["subgroup"]["size"], 32)
        self.assertEqual(matrix["subgroup"]["min"], 16)
        self.assertEqual(matrix["subgroup"]["max"], 64)
        self.assertIs(matrix["subgroup_size_control"]["subgroupSizeControl"], True)
        self.assertIs(matrix["mesh"], True)
        self.assertIs(matrix["barycentric"], False)
        self.assertIs(matrix["storageInputOutput16"], False)

    def test_older_device_without_versioned_structs(self):
        matrix = self.probe(report())
        self.assertEqual(matrix["vulkaninfo"], "ok")
        self.assertIsNone(matrix["subgroup"]["size"])
        self.assertIsNone(matrix["subgroup_size_control"]["subgroupSizeControl"])
        self.assertIsNone(matrix["storageInputOutput16"])

    def test_failed_subprocess_discards_even_a_valid_report(self):
        self.assert_unknown(self.probe(report(), returncode=1))

    def test_unusable_profiles_leave_the_matrix_unknown(self):
        no_name = report()
        del no_name["capabilities"]["device"]["properties"]["VkPhysicalDeviceProperties"]["deviceName"]
        missing_extensions = report()
        del missing_extensions["capabilities"]["device"]["extensions"]
        invalid_extensions = report()
        invalid_extensions["capabilities"]["device"]["extensions"] = []
        invalid_feature_block = report()
        invalid_feature_block["capabilities"]["device"]["features"]["VkPhysicalDeviceVulkan11Features"] = []
        for doc in ([], {}, {"capabilities": []}, no_name, missing_extensions, invalid_extensions, invalid_feature_block):
            with self.subTest(document=doc):
                self.assert_unknown(self.probe(doc))


if __name__ == "__main__":
    unittest.main()
