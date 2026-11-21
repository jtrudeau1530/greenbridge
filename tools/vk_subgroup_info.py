"""Ask the Vulkan driver whether this PC's GPU can run 64-wide compute and mesh subgroups.

    python -I tools/vk_subgroup_info.py          # prints one JSON line, exit 0 if a device answered, 1 if not

make_game.py uses it to choose between native wave64 translation and --wave64-emulate. It talks to the system's
Vulkan loader through ctypes (standard library only), so it works before the runtime stage has built anything.

Device choice follows the runtime (runtime/shims/d3d12_vulkan_device.cpp): Vulkan 1.3 devices with dynamicRendering,
synchronization2, timelineSemaphore and separateDepthStencilLayouts and a graphics queue family; a discrete GPU wins
(the last one), otherwise the first. The runtime also needs the rest of its device setup (surface, extensions);
that is not checked here. vk_pipeline_check picks by device type only, so on a PC where the two differ the pipelines
stage can run on another GPU than the one asked here. VK_LOADER_DRIVERS_SELECT (e.g. '*amd*') narrows which
drivers the loader offers.

native64 matches vk_pipeline_check's can_require_sg(64, bit) for the compute and the mesh stage: the
subgroupSizeControl feature, minSubgroupSize <= 64 <= maxSubgroupSize, and the stage in requiredSubgroupSizeStages.
Mesh shaders are always wave64 natively, so both stage bits are needed.
"""
import ctypes
import json
import os
import sys

VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 = 1000059000
VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 = 1000059001
VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES = 51
VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES = 53
VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES = 1000225000
VK_SHADER_STAGE_COMPUTE_BIT = 0x20
VK_SHADER_STAGE_MESH_BIT_EXT = 0x80
VK_QUEUE_GRAPHICS_BIT = 1
VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2
VK_API_VERSION_1_3 = (1 << 22) | (3 << 12)

PROPS_SIZE = 824            # sizeof(VkPhysicalDeviceProperties) on 64-bit
FEATURES2_SIZE = 240        # sizeof(VkPhysicalDeviceFeatures2): 16 + 55 VkBool32, padded to 8
F12_SIZE = 208              # sizeof(VkPhysicalDeviceVulkan12Features): 16 + 47 VkBool32, padded to 8
F13_SIZE = 80               # sizeof(VkPhysicalDeviceVulkan13Features): 16 + 15 VkBool32, padded to 8
# Byte offsets of the VkBool32 members read below (16-byte header, then 4 bytes per member in header order)
F12_SEPARATE_DEPTH_STENCIL_LAYOUTS = 16 + 4 * 35
F12_TIMELINE_SEMAPHORE = 16 + 4 * 37
F13_SUBGROUP_SIZE_CONTROL = 16 + 4 * 7
F13_SYNCHRONIZATION2 = 16 + 4 * 9
F13_DYNAMIC_RENDERING = 16 + 4 * 12
PROPS_DEVICE_TYPE = 16      # offsets in VkPhysicalDeviceProperties
PROPS_DEVICE_NAME = 20      # char[256]
PROPS_API_VERSION = 0
QUEUE_FAMILY_SIZE = 24      # sizeof(VkQueueFamilyProperties)


class AppInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("pApplicationName", ctypes.c_char_p),
                ("applicationVersion", ctypes.c_uint32), ("pEngineName", ctypes.c_char_p),
                ("engineVersion", ctypes.c_uint32), ("apiVersion", ctypes.c_uint32)]


class InstanceInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("pApplicationInfo", ctypes.POINTER(AppInfo)), ("enabledLayerCount", ctypes.c_uint32),
                ("ppEnabledLayerNames", ctypes.c_void_p), ("enabledExtensionCount", ctypes.c_uint32),
                ("ppEnabledExtensionNames", ctypes.c_void_p)]


class SubgroupSizeControl(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("minSubgroupSize", ctypes.c_uint32),
                ("maxSubgroupSize", ctypes.c_uint32), ("maxComputeWorkgroupSubgroups", ctypes.c_uint32),
                ("requiredSubgroupSizeStages", ctypes.c_uint32)]


def block(size: int, stype: int):
    """A zeroed struct of `size` bytes with sType set (sType and pNext head every Vulkan struct used here)."""
    buf = ctypes.create_string_buffer(size)
    ctypes.c_uint32.from_buffer(buf, 0).value = stype
    return buf


def set_next(buf, other) -> None:
    ctypes.c_void_p.from_buffer(buf, 8).value = ctypes.addressof(other)


def u32(buf, off: int) -> int:
    return ctypes.c_uint32.from_buffer(buf, off).value


def load_vulkan():
    names = ["vulkan-1.dll"] if os.name == "nt" else ["libvulkan.so.1", "libvulkan.1.dylib"]
    for n in names:
        try:
            return ctypes.CDLL(n)
        except OSError:
            pass
    raise RuntimeError("no Vulkan loader found (" + ", ".join(names) + ")")


def query() -> dict:
    """The answer for the device the runtime would pick; raises RuntimeError with the reason when there is none."""
    if sys.maxsize <= 2**32:
        # the struct sizes below are the 64-bit ones, and 32-bit Windows Vulkan is stdcall, not CDLL's cdecl
        raise RuntimeError("needs a 64-bit Python")
    vk = load_vulkan()
    vp = ctypes.c_void_p
    vk.vkCreateInstance.argtypes = [vp, vp, vp]
    vk.vkEnumeratePhysicalDevices.argtypes = [vp, vp, vp]
    vk.vkGetPhysicalDeviceProperties2.argtypes = [vp, vp]
    vk.vkGetPhysicalDeviceFeatures2.argtypes = [vp, vp]
    vk.vkGetPhysicalDeviceQueueFamilyProperties.argtypes = [vp, vp, vp]
    vk.vkDestroyInstance.argtypes = [vp, vp]
    app = AppInfo(VK_STRUCTURE_TYPE_APPLICATION_INFO, None, b"leonida-subgroup-info", 1, None, 0, VK_API_VERSION_1_3)
    info = InstanceInfo(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, None, 0, ctypes.pointer(app), 0, None, 0, None)
    inst = ctypes.c_void_p()
    r = vk.vkCreateInstance(ctypes.byref(info), None, ctypes.byref(inst))
    if r != 0:
        raise RuntimeError(f"vkCreateInstance failed ({r})")
    try:
        n = ctypes.c_uint32(0)
        vk.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), None)
        if n.value == 0:
            raise RuntimeError("no Vulkan device")
        devs = (ctypes.c_void_p * n.value)()
        vk.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), devs)
        chosen = None
        for d in devs[: n.value]:
            d = ctypes.c_void_p(d)
            sg = SubgroupSizeControl(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES)
            props = block(16 + PROPS_SIZE, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2)
            ctypes.c_void_p.from_buffer(props, 8).value = ctypes.addressof(sg)
            vk.vkGetPhysicalDeviceProperties2(d, props)
            raw = bytes(props)[16:]
            if int.from_bytes(raw[PROPS_API_VERSION:PROPS_API_VERSION + 4], "little") < VK_API_VERSION_1_3:
                continue
            f13 = block(F13_SIZE, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES)
            f12 = block(F12_SIZE, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
            set_next(f12, f13)
            f2 = block(FEATURES2_SIZE, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
            set_next(f2, f12)
            vk.vkGetPhysicalDeviceFeatures2(d, f2)
            if not (u32(f13, F13_DYNAMIC_RENDERING) and u32(f13, F13_SYNCHRONIZATION2)
                    and u32(f12, F12_TIMELINE_SEMAPHORE) and u32(f12, F12_SEPARATE_DEPTH_STENCIL_LAYOUTS)):
                continue
            fam = ctypes.c_uint32(0)
            vk.vkGetPhysicalDeviceQueueFamilyProperties(d, ctypes.byref(fam), None)
            fams = ctypes.create_string_buffer(QUEUE_FAMILY_SIZE * max(fam.value, 1))
            vk.vkGetPhysicalDeviceQueueFamilyProperties(d, ctypes.byref(fam), fams)
            if not any(u32(fams, QUEUE_FAMILY_SIZE * i) & VK_QUEUE_GRAPHICS_BIT for i in range(fam.value)):
                continue
            dtype = int.from_bytes(raw[PROPS_DEVICE_TYPE:PROPS_DEVICE_TYPE + 4], "little")
            # the runtime's rule: the first usable device, replaced by every later discrete GPU
            if chosen is None or dtype == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                chosen = (raw[PROPS_DEVICE_NAME:PROPS_DEVICE_NAME + 256].split(b"\0")[0].decode("utf-8", "replace"),
                          sg.minSubgroupSize, sg.maxSubgroupSize, sg.requiredSubgroupSizeStages,
                          bool(u32(f13, F13_SUBGROUP_SIZE_CONTROL)))
        if chosen is None:
            raise RuntimeError("no Vulkan 1.3 GPU with the features and graphics queue the runtime needs")
        name, lo, hi, stages, sgc = chosen
        comp = bool(stages & VK_SHADER_STAGE_COMPUTE_BIT)
        mesh = bool(stages & VK_SHADER_STAGE_MESH_BIT_EXT)
        return {"device": name, "min_subgroup_size": lo, "max_subgroup_size": hi,
                "subgroup_size_control": sgc, "compute_in_required_stages": comp, "mesh_in_required_stages": mesh,
                "native64": sgc and lo <= 64 <= hi and comp and mesh}
    finally:
        vk.vkDestroyInstance(inst, None)


def main() -> int:
    try:
        print(json.dumps(query()))
        return 0
    except Exception as e:      # any failure means "unknown": the caller falls back to emulation
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
