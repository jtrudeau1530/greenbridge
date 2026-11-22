// Headless Vulkan harness: draws one triangle (3 vertices, no vertex buffers) with a given VS + FS into a
// W x H RGBA32F image and writes the raw floats to a file. Used to check translated shaders on the real GPU.
//
//   vk_harness <vs.spv> <fs.spv> <out.raw> [width height [clip-test-s0 clip-test-s1]]
//   vk_harness <ms.spv> <fs.spv> <out.raw> --mesh-group-readback X Y Z
// The latter captures one full-screen triangle per mesh workgroup, in Vulkan rasterization order.
//
// Push constants follow the contract in docs/design.md: s0..s31, then the heap base lo/hi. They are zero unless
//   VKH_SGPRS=w0,w1,...          sets s0, s1, ... (decimal or 0x hex)
//   VKH_HEAP=addr=word,...       allocates a zeroed 64 KiB heap (Xbox VA 0 = its device address), writes each word at
//                                its address and passes the heap base, so a V# in VKH_SGPRS can point into it.
//   VKH_HEAP_DUMP=<file>         after the draw, writes the 64 KiB heap (needs VKH_HEAP, which may be empty-valued as
//                                "0=0"), so a test can read what the pixel shader stored or atomically added.
//   VKH_DRAW_MESH=<file> + VKH_VERTEX_LAYOUT=0:3f,1:2f
//                                draws an indexed mesh instead of the 3-vertex triangle. The file is u32
//                                vertex_count, u32 index_count, vertex_count*stride bytes, index_count u32
//                                indices; the layout lists location:float-components, one item per attribute.
//   VKH_SUBGROUP_SIZE=32|64      requires that subgroup size of the fragment shader (or of the compute or geometry
//                                stage under test); stderr says "fragment subgroup size: N" when it took effect and
//                                "... not available" when the device cannot require it (exit code stays 0).
//
#define VK_NO_PROTOTYPES
#include "vulkan_loader.h"
#include "vk_device_pick.h"
#include "vk_shader_fma.h"

#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <utility>
#include <vector>

static PFN_vkGetInstanceProcAddr gipa;
#define LOADI(name) auto name = reinterpret_cast<PFN_##name>(gipa(inst, #name))
#define CHECK(x)                                                                 \
    do {                                                                         \
        VkResult r_ = (x);                                                       \
        if (r_ != VK_SUCCESS) {                                                  \
            std::fprintf(stderr, "FAILED %s -> %d (line %d)\n", #x, r_, __LINE__); \
            return 1;                                                            \
        }                                                                        \
    } while (0)

static std::vector<uint32_t> read_spv(const char* path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    std::vector<uint32_t> w(b.size() / 4);
    std::memcpy(w.data(), b.data(), w.size() * 4);
    return w;
}

static bool parse_u32(const char*& s, uint32_t& v) {
    const bool hex = s[0] == '0' && (s[1] == 'x' || s[1] == 'X');
    const char* digits = hex ? s + 2 : s;
    if (!(hex ? std::isxdigit(uint8_t(*digits)) : std::isdigit(uint8_t(*digits)))) return false;   // no sign or space
    char* end;
    unsigned long long x = std::strtoull(digits, &end, hex ? 16 : 10);
    if (x > 0xFFFFFFFFull) return false;
    v = uint32_t(x); s = end;
    return true;
}

// A list separator must be followed by another item: "1,2," is rejected.
static bool next_item(const char*& s) {
    if (!*s) return true;
    return *s++ == ',' && *s;
}

static bool parse_sgprs(const char* s, uint32_t* sgprs) {
    for (uint32_t i = 0; *s; i++) {
        if (i == 32 || !parse_u32(s, sgprs[i]) || !next_item(s)) return false;
    }
    return true;
}

static bool parse_heap(const char* s, std::vector<std::pair<uint32_t, uint32_t>>& words, uint32_t heap_bytes) {
    while (*s) {
        uint32_t addr, word;
        if (!parse_u32(s, addr) || *s++ != '=' || !parse_u32(s, word) || !next_item(s)) return false;
        if (addr % 4 || addr > heap_bytes - 4) return false;
        words.emplace_back(addr, word);
    }
    return true;
}

// VKH_VERTEX_LAYOUT=0:3f,1:2f: one attribute per item, float components only. Location 0 is the first float3.
struct VertexAttr {
    uint32_t location, components, offset;
    VkFormat format;
};

static bool parse_vertex_layout(const char* s, VertexAttr* attrs, uint32_t& count, uint32_t& stride) {
    count = 0; stride = 0;
    while (*s) {
        uint32_t loc, comps;
        if (count == 8 || !parse_u32(s, loc) || *s++ != ':' || !parse_u32(s, comps) ||
            comps < 1 || comps > 4 || *s++ != 'f' || !next_item(s))
            return false;
        static const VkFormat formats[4] = {VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32G32_SFLOAT,
                                            VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT};
        attrs[count++] = {loc, comps, stride, formats[comps - 1]};
        stride += comps * 4;
    }
    return count > 0;
}

// VKH_DRAW_MESH file: u32 vertex_count, u32 index_count, vertex_count*stride bytes, index_count u32 indices.
static bool read_mesh(const char* path, uint32_t stride, std::vector<uint8_t>& vertices,
                      std::vector<uint32_t>& indices) {
    std::ifstream f(path, std::ios::binary);
    uint32_t vcount = 0, icount = 0;
    f.read(reinterpret_cast<char*>(&vcount), 4);
    f.read(reinterpret_cast<char*>(&icount), 4);
    if (!f || vcount == 0 || icount == 0 || vcount > 10000000u || icount > 30000000u) return false;
    vertices.resize(size_t(vcount) * stride);
    indices.resize(icount);
    f.read(reinterpret_cast<char*>(vertices.data()), std::streamsize(vertices.size()));
    f.read(reinterpret_cast<char*>(indices.data()), std::streamsize(indices.size() * 4));
    return bool(f);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: vk_harness vs.spv fs.spv out.raw [w h [clip-test-s0 clip-test-s1]]\n"
                             "       vk_harness ms.spv fs.spv out.raw --mesh-group-readback X Y Z\n");
        return 2;
    }
    const bool group_readback = argc > 4 && !std::strcmp(argv[4], "--mesh-group-readback");
    const bool clip_shifts = !group_readback && argc == 8;
    uint32_t groups[3] = {1, 1, 1};
    uint32_t W = 64, H = 64;
    if (group_readback) {
        if (argc != 8) {
            std::fprintf(stderr, "--mesh-group-readback requires X Y Z\n");
            return 2;
        }
        W = H = 1;
        for (unsigned i = 0; i < 3; ++i) {
            char* end;
            long n = std::strtol(argv[5 + i], &end, 10);
            if (*end || n < 1 || n > 256 / W) {
                std::fprintf(stderr, "mesh grid must contain 1..256 groups (8-bit stencil)\n");
                return 2;
            }
            groups[i] = uint32_t(n);
            W *= groups[i];
        }
    } else if (argc > 5) {
        W = atoi(argv[4]); H = atoi(argv[5]);
    }
    // Push constants: s0..s31, heap base (32..33), code base (34..35), --mesh-compute output address (36..37).
    uint32_t pc[38] = {};
    const char* draw_mesh_path = std::getenv("VKH_DRAW_MESH");
    if (draw_mesh_path && !*draw_mesh_path) draw_mesh_path = nullptr;
    VertexAttr vertex_attrs[8];
    uint32_t vertex_attr_count = 0, vertex_stride = 0;
    if (draw_mesh_path) {
        const char* layout = std::getenv("VKH_VERTEX_LAYOUT");
        if (!layout || !parse_vertex_layout(layout, vertex_attrs, vertex_attr_count, vertex_stride)) {
            std::fprintf(stderr, "VKH_VERTEX_LAYOUT: expected locations like 0:3f,1:2f with VKH_DRAW_MESH\n");
            return 2;
        }
    }
    const char* sgprs_env = std::getenv("VKH_SGPRS");
    if (sgprs_env && !parse_sgprs(sgprs_env, pc)) {
        std::fprintf(stderr, "VKH_SGPRS: expected up to 32 comma-separated words, got \"%s\"\n", sgprs_env);
        return 2;
    }
    const uint32_t heap_bytes = 64 * 1024;
    const char* heap_env = std::getenv("VKH_HEAP");
    if (heap_env && !*heap_env) heap_env = nullptr;
    std::vector<std::pair<uint32_t, uint32_t>> heap_words;
    if (heap_env && !parse_heap(heap_env, heap_words, heap_bytes)) {
        std::fprintf(stderr, "VKH_HEAP: expected addr=word pairs, 4-byte aligned below 0x%x, got \"%s\"\n",
                     heap_bytes, heap_env);
        return 2;
    }
    gipa = load_vulkan();
    if (!gipa) return 1;
    VkInstance inst = VK_NULL_HANDLE;
    LOADI(vkCreateInstance);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    CHECK(vkCreateInstance(&ici, nullptr, &inst));
    LOADI(vkEnumeratePhysicalDevices); LOADI(vkGetPhysicalDeviceProperties);
    LOADI(vkGetPhysicalDeviceProperties2); LOADI(vkGetPhysicalDeviceFeatures2);
    LOADI(vkGetPhysicalDeviceQueueFamilyProperties); LOADI(vkGetPhysicalDeviceMemoryProperties);
    LOADI(vkCreateDevice); LOADI(vkGetDeviceProcAddr);
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(inst, &n, pds.data());
    std::vector<VkPhysicalDeviceProperties> candidates(pds.size());
    for (std::size_t i = 0; i < pds.size(); ++i) vkGetPhysicalDeviceProperties(pds[i], &candidates[i]);
    const std::size_t chosen = leonida::vk_test::pick_device(candidates);
    VkPhysicalDevice pd = chosen < pds.size() ? pds[chosen] : VK_NULL_HANDLE;
    if (!pd) {
        leonida::vk_test::report_no_device();
        return 1;
    }
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd, &props);
    VkPhysicalDeviceSubgroupSizeControlProperties sgProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &sgProps;
    vkGetPhysicalDeviceProperties2(pd, &props2);
    std::fprintf(stderr, "device: %s (subgroup %u..%u)\n", props.deviceName, sgProps.minSubgroupSize, sgProps.maxSubgroupSize);
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
    uint32_t qi = 0;
    while (qi < qn && !(qf[qi].queueFlags & VK_QUEUE_GRAPHICS_BIT)) qi++;
    float prio = 1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    // Optional features some translated shaders need (enabled only when the device has them):
    // raw barycentric reads (BaryCoordKHR), POPS ordering (pixel interlock), non-constant texel offsets (maintenance8).
    auto vkEnumerateDeviceExtensionProperties = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
        gipa(inst, "vkEnumerateDeviceExtensionProperties"));
    uint32_t en = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, nullptr);
    std::vector<VkExtensionProperties> exts(en);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, exts.data());
    auto has = [&](const char* e) {
        for (auto& x : exts) if (!std::strcmp(x.extensionName, e)) return true;
        return false;
    };
    // A translated pixel shader that reads BaryCoordKHR aborts the driver when the device lacks barycentrics
    // (llvmpipe and other software devices), so the suite reports it as unsupported instead. 77 is the existing
    // "this device cannot run the test" code (runtime/test/shaders/check.py, scripts/run_gpu_tests.py).
    const std::vector<uint32_t> fs_code = read_spv(argv[2]);
    bool needs_barycentric = false, needs_float64 = false;
    for (std::size_t i = 5; i + 1 < fs_code.size();) {   // instructions follow the 5-word SPIR-V header
        const uint32_t words = fs_code[i] >> 16, op = fs_code[i] & 0xffffu;
        if (!words) break;
        if (op == 17 && fs_code[i + 1] == 5284)          // OpCapability FragmentBarycentricKHR
            needs_barycentric = true;
        if (op == 17 && fs_code[i + 1] == 10)            // OpCapability Float64 (v_add_f64 and friends)
            needs_float64 = true;
        i += words;
    }
    if (needs_barycentric && !has(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME)) {
        std::fprintf(stderr, "unsupported: %s reads BaryCoordKHR and the device has no %s\n",
                     argv[2], VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
        return 77;
    }
    // OpFmaKHR (rdna2spirv.py --fma khr) needs VK_KHR_shader_fma; the fragment shader decides whether it is refused.
    leonida::vk_test::ShaderFma fma;
    fma.query(gipa, inst, pd);
    if (!fma.accepts(fs_code.data(), fs_code.size() * 4, argv[2])) return 77;
    std::vector<const char*> enable;
    void* chain = nullptr;
    if (fma.supported()) {
        enable.push_back(fma.extension());
#ifdef VK_KHR_SHADER_FMA_EXTENSION_NAME  // older Vulkan headers have no VkPhysicalDeviceShaderFmaFeaturesKHR
        fma.features.pNext = chain; chain = &fma.features;
#endif
    }
    VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR bary{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
    VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT lock{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT};
    if (has(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME)) {
        enable.push_back(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
        bary.fragmentShaderBarycentric = VK_TRUE; bary.pNext = chain; chain = &bary;
    }
    if (has(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME)) {
        enable.push_back(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME);
        lock.fragmentShaderPixelInterlock = VK_TRUE; lock.pNext = chain; chain = &lock;
    }
    // VKH_MESH_COMPUTE=<vs.spv>: the first shader is a mesh shader translated with rdna2spirv.py --mesh-compute.
    // It is dispatched as compute over the mesh grid (1x1x1, or --mesh-group-readback's), writing its slots to an
    // output buffer (VKH_MESH_SLOT bytes per workgroup, address in push-constant dwords 36..37). Then
    // vs.spv (tools/mesh_compute_vs.py) draws groups * VKH_MESH_PRIMS * 3 vertices with the fragment shader. No
    // VK_EXT_mesh_shader is needed.
    const char* mesh_vs_path = std::getenv("VKH_MESH_COMPUTE");
    if (mesh_vs_path && !*mesh_vs_path) mesh_vs_path = nullptr;
    const bool mesh_compute = mesh_vs_path != nullptr;
    uint64_t mesh_slot = 0;
    uint32_t mesh_prims = 0;
    if (mesh_compute) {
        const char* slot_env = std::getenv("VKH_MESH_SLOT");
        const char* prims_env = std::getenv("VKH_MESH_PRIMS");
        mesh_slot = slot_env ? std::strtoull(slot_env, nullptr, 0) : 0;
        mesh_prims = prims_env ? uint32_t(std::strtoul(prims_env, nullptr, 0)) : 0;
        if (!mesh_slot || !mesh_prims) {
            std::fprintf(stderr, "VKH_MESH_COMPUTE needs VKH_MESH_SLOT and VKH_MESH_PRIMS (index.json mesh_compute)\n");
            return 2;
        }
    }
    // Translated shaders declare a 144-byte push-constant block (user SGPRs, heap base, code base; see VKH_SGPRS /
    // VKH_HEAP); --mesh-compute modules and their vertex shaders add the output address (152 bytes). A device with
    // fewer bytes (128, the Vulkan minimum, as on some Intel Arc drivers and AMD's Windows driver for RDNA2 iGPUs)
    // cannot take that range: declaring it anyway is invalid, and there a shader reading past the limit crashed or
    // hung the device. 77, as vk_texture_test and vk_bindless_test: a device limit, not a wrong result (D3D-14).
    const uint32_t pc_bytes = mesh_compute ? 152u : 144u;
    if (props.limits.maxPushConstantsSize < pc_bytes) {
        std::fprintf(stderr, "unsupported: need %u push constant bytes, device has %u\n", pc_bytes,
                     props.limits.maxPushConstantsSize);
        return 77;
    }
    // VKH_MESH=1: the first shader is a mesh shader (VK_EXT_mesh_shader), drawn with vkCmdDrawMeshTasksEXT(1,1,1).
    const bool mesh = !mesh_compute && (group_readback || std::getenv("VKH_MESH") != nullptr);
    VkPhysicalDeviceMeshShaderFeaturesEXT meshf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    if (mesh) {
        // Query the feature, not just the extension. 77: the device cannot run this test (as vk_compute_test and
        // vk_store_test), so the runners report SKIP.
        VkPhysicalDeviceFeatures2 mesh_query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        mesh_query.pNext = &meshf;
        vkGetPhysicalDeviceFeatures2(pd, &mesh_query);
        if (!has(VK_EXT_MESH_SHADER_EXTENSION_NAME) || !meshf.meshShader) {
            std::fprintf(stderr, "unsupported: no VK_EXT_mesh_shader\n");
            return 77;
        }
        enable.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);
        meshf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
        meshf.meshShader = VK_TRUE; meshf.pNext = chain; chain = &meshf;
    }
#ifdef VK_KHR_MAINTENANCE_8_EXTENSION_NAME
    VkPhysicalDeviceMaintenance8FeaturesKHR m8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR};
    if (has(VK_KHR_MAINTENANCE_8_EXTENSION_NAME)) {
        enable.push_back(VK_KHR_MAINTENANCE_8_EXTENSION_NAME);
        m8.maintenance8 = VK_TRUE; m8.pNext = chain; chain = &m8;
    }
#endif
    VkPhysicalDeviceVulkan12Features f12q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13q.pNext = &f12q;
    VkPhysicalDeviceFeatures2 f2q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2q.pNext = &f13q;
    vkGetPhysicalDeviceFeatures2(pd, &f2q);
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    if (f13q.subgroupSizeControl) {
        f13.subgroupSizeControl = VK_TRUE;
        f13.pNext = chain;
        chain = &f13;
    }
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    if (heap_env || mesh_compute) {
        // 77 is the "this device cannot run the test" code that check.py and run_gpu_tests.py report as SKIP.
        if (!f12q.bufferDeviceAddress) {
            std::fprintf(stderr, "unsupported: %s needs bufferDeviceAddress\n", heap_env ? "VKH_HEAP" : "VKH_MESH_COMPUTE");
            return 77;
        }
        f12.bufferDeviceAddress = VK_TRUE;
        f12.pNext = chain;
        chain = &f12;
    }
    for (auto e : enable) std::fprintf(stderr, "enabled %s\n", e);
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = uint32_t(enable.size()); dci.ppEnabledExtensionNames = enable.data();
    dci.pNext = chain;
    // VKH_GS=<geometry.spv>: insert a geometry stage between the vertex and fragment shaders. VKH_SUBGROUP_SIZE then
    // applies to it, unless VKH_GS_HELPER=1 says the GS only feeds the pixel shader under test.
    const char* gs_path = std::getenv("VKH_GS");
    if (gs_path && !f2q.features.geometryShader) {   // MoltenVK: Metal has no geometry stage
        std::fprintf(stderr, "unsupported: no geometryShader\n");
        return 77;
    }
    VkPhysicalDeviceFeatures feats{};
    if (clip_shifts && !f2q.features.shaderClipDistance) {
        std::fprintf(stderr, "unsupported: shaderClipDistance\n");
        return 77;
    }
    feats.geometryShader = gs_path ? VK_TRUE : VK_FALSE;
    feats.shaderClipDistance = f2q.features.shaderClipDistance;
    feats.shaderCullDistance = f2q.features.shaderCullDistance;
    if (needs_float64) {
        if (!f2q.features.shaderFloat64) {
            std::fprintf(stderr, "unsupported: %s uses Float64 and the device has no shaderFloat64\n", argv[2]);
            return 77;
        }
        feats.shaderFloat64 = VK_TRUE;
    }
    if (heap_env || mesh_compute) {
        if (!f2q.features.shaderInt64) { std::fprintf(stderr, "unsupported: 64-bit addresses need shaderInt64\n"); return 77; }
        feats.shaderInt64 = VK_TRUE;   // heap and --mesh-compute output addresses are 64-bit
    }
    dci.pEnabledFeatures = &feats;
    VkDevice dev;
    CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
#define LOADD(name) auto name = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(dev, #name))
    LOADD(vkGetDeviceQueue); LOADD(vkCreateImage); LOADD(vkGetImageMemoryRequirements); LOADD(vkAllocateMemory);
    LOADD(vkBindImageMemory); LOADD(vkCreateImageView); LOADD(vkCreateRenderPass); LOADD(vkCreateFramebuffer);
    LOADD(vkCreateShaderModule); LOADD(vkCreatePipelineLayout); LOADD(vkCreateGraphicsPipelines);
    LOADD(vkCreateCommandPool); LOADD(vkAllocateCommandBuffers); LOADD(vkBeginCommandBuffer);
    LOADD(vkCmdBeginRenderPass); LOADD(vkCmdBindPipeline); LOADD(vkCmdDraw); LOADD(vkCmdEndRenderPass);
    LOADD(vkCmdBindVertexBuffers); LOADD(vkCmdBindIndexBuffer); LOADD(vkCmdDrawIndexed);
    LOADD(vkCmdCopyImageToBuffer); LOADD(vkEndCommandBuffer); LOADD(vkQueueSubmit); LOADD(vkQueueWaitIdle);
    LOADD(vkCreateBuffer); LOADD(vkGetBufferMemoryRequirements); LOADD(vkBindBufferMemory); LOADD(vkMapMemory);
    VkQueue q; vkGetDeviceQueue(dev, qi, 0, &q);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    auto memtype = [&](uint32_t bits, VkMemoryPropertyFlags want) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
        return 0u;
    };
    auto attachment = [&](VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                          VkImage& image, VkImageView& view) -> int {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D; ii.format = format; ii.extent = {W, H, 1}; ii.mipLevels = 1; ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL; ii.usage = usage;
        CHECK(vkCreateImage(dev, &ii, nullptr, &image));
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, image, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size; mai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory mem; CHECK(vkAllocateMemory(dev, &mai, nullptr, &mem));
        CHECK(vkBindImageMemory(dev, image, mem, 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = format;
        vi.subresourceRange = {aspect, 0, 1, 0, 1};
        CHECK(vkCreateImageView(dev, &vi, nullptr, &view));
        return 0;
    };
    const VkFormat fmt = VK_FORMAT_R32G32B32A32_SFLOAT;
    VkImage img;
    VkImageView views[2]{};
    if (attachment(fmt, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                   VK_IMAGE_ASPECT_COLOR_BIT, img, views[0])) return 1;
    VkAttachmentDescription attachments[2]{};
    auto& ad = attachments[0];
    ad.format = fmt; ad.samples = VK_SAMPLE_COUNT_1_BIT; ad.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    ad.storeOp = VK_ATTACHMENT_STORE_OP_STORE; ad.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    ad.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; ad.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ad.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkAttachmentReference ar{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sd{}; sd.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sd.colorAttachmentCount = 1; sd.pColorAttachments = &ar;
    VkAttachmentReference stencil_ref{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    if (group_readback) {
        LOADI(vkGetPhysicalDeviceFormatProperties);
        auto& ds = attachments[1];
        for (auto format : {VK_FORMAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT}) {
            VkFormatProperties fp;
            vkGetPhysicalDeviceFormatProperties(pd, format, &fp);
            if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
                ds.format = format;
                break;
            }
        }
        if (!ds.format) { std::fprintf(stderr, "no stencil attachment format\n"); return 1; }
        VkImage stencil;
        VkImageAspectFlags aspects = VK_IMAGE_ASPECT_STENCIL_BIT;
        if (ds.format != VK_FORMAT_S8_UINT) aspects |= VK_IMAGE_ASPECT_DEPTH_BIT;
        if (attachment(ds.format, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, aspects, stencil, views[1])) return 1;
        ds.samples = VK_SAMPLE_COUNT_1_BIT;
        ds.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; ds.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        ds.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; ds.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        ds.finalLayout = stencil_ref.layout;
        sd.pDepthStencilAttachment = &stencil_ref;
    }
    VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpi.attachmentCount = group_readback ? 2 : 1; rpi.pAttachments = attachments; rpi.subpassCount = 1; rpi.pSubpasses = &sd;
    VkSubpassDependency copy_dep{0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    rpi.dependencyCount = 1; rpi.pDependencies = &copy_dep;
    VkRenderPass rp; CHECK(vkCreateRenderPass(dev, &rpi, nullptr, &rp));
    VkFramebufferCreateInfo fbi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbi.renderPass = rp; fbi.attachmentCount = rpi.attachmentCount; fbi.pAttachments = views; fbi.width = W; fbi.height = H; fbi.layers = 1;
    VkFramebuffer fb; CHECK(vkCreateFramebuffer(dev, &fbi, nullptr, &fb));
    auto module = [&](const char* p, VkShaderModule* m) {
        auto code = read_spv(p);
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smi.codeSize = code.size() * 4; smi.pCode = code.data();
        return vkCreateShaderModule(dev, &smi, nullptr, m);
    };
    VkShaderModule vs, fs, csm = VK_NULL_HANDLE;
    CHECK(module(mesh_compute ? mesh_vs_path : argv[1], &vs)); CHECK(module(argv[2], &fs));
    if (mesh_compute) CHECK(module(argv[1], &csm));
    VkPipelineShaderStageCreateInfo stages[3]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                              {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                              {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    stages[0].stage = mesh ? VK_SHADER_STAGE_MESH_BIT_EXT : VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs; stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = fs; stages[1].pName = "main";
    uint32_t nstages = 2;
    if (gs_path) {
        VkShaderModule gsm;
        CHECK(module(gs_path, &gsm));
        stages[2].stage = VK_SHADER_STAGE_GEOMETRY_BIT; stages[2].module = gsm; stages[2].pName = "main";
        nstages = 3;
    }
    const char* sg_env = std::getenv("VKH_SUBGROUP_SIZE");
    const uint32_t req_sg = sg_env ? uint32_t(atoi(sg_env)) : 0;
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo reqSg{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    reqSg.requiredSubgroupSize = req_sg;
    if (req_sg > 0 && f13q.subgroupSizeControl &&
        req_sg >= sgProps.minSubgroupSize && req_sg <= sgProps.maxSubgroupSize) {
        // VKH_GS_HELPER=1: the geometry shader only feeds the pixel shader under test (the --bary-fallback
        // pass-through), so the size still goes to the fragment stage.
        const bool gs_tested = gs_path && !std::getenv("VKH_GS_HELPER");
        if (mesh && (sgProps.requiredSubgroupSizeStages & VK_SHADER_STAGE_MESH_BIT_EXT))
            stages[0].pNext = &reqSg;
        else if (gs_tested && (sgProps.requiredSubgroupSizeStages & VK_SHADER_STAGE_GEOMETRY_BIT))
            stages[2].pNext = &reqSg;
        else if (!mesh && !gs_tested && (sgProps.requiredSubgroupSizeStages & VK_SHADER_STAGE_FRAGMENT_BIT))
            stages[1].pNext = &reqSg;
        // A device with one subgroup size runs every stage at it, required or not (AMD GCN: 64, with only COMPUTE
        // in requiredSubgroupSizeStages), so the fragment stage of a plain draw already has the size asked for.
        const bool fixed = !mesh && !gs_tested && sgProps.minSubgroupSize == sgProps.maxSubgroupSize;
        if (stages[1].pNext == &reqSg || fixed) std::fprintf(stderr, "fragment subgroup size: %u\n", req_sg);
    } else if (req_sg > 0) {
        std::fprintf(stderr, "fragment subgroup size: %u not available\n", req_sg);
    }
    VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkVertexInputBindingDescription vbind{0, vertex_stride, VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription vattr[8]{};
    if (draw_mesh_path) {
        for (uint32_t i = 0; i < vertex_attr_count; ++i)
            vattr[i] = {vertex_attrs[i].location, 0, vertex_attrs[i].format, vertex_attrs[i].offset};
        vin.vertexBindingDescriptionCount = 1;
        vin.pVertexBindingDescriptions = &vbind;
        vin.vertexAttributeDescriptionCount = vertex_attr_count;
        vin.pVertexAttributeDescriptions = vattr;
    }
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkViewport vp{0, 0, float(W), float(H), 0, 1}; VkRect2D sc{{0, 0}, {W, H}};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = 1; vps.pViewports = &vp; vps.scissorCount = 1; vps.pScissors = &sc;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState cba{}; cba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1; cb.pAttachments = &cba;
    // The push-constant block (pc_bytes, checked against the device limit above).
    const VkShaderStageFlags pcs = (mesh ? VK_SHADER_STAGE_MESH_BIT_EXT : VK_SHADER_STAGE_VERTEX_BIT) |
                                   VK_SHADER_STAGE_FRAGMENT_BIT | (gs_path ? VK_SHADER_STAGE_GEOMETRY_BIT : 0) |
                                   (mesh_compute ? VK_SHADER_STAGE_COMPUTE_BIT : 0);
    VkPushConstantRange pcr{pcs, 0, pc_bytes};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pl; CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));
    VkGraphicsPipelineCreateInfo gpi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpi.stageCount = nstages; gpi.pStages = stages;
    gpi.pVertexInputState = mesh ? nullptr : &vin; gpi.pInputAssemblyState = mesh ? nullptr : &ia;
    gpi.pViewportState = &vps; gpi.pRasterizationState = &rs; gpi.pMultisampleState = &ms; gpi.pColorBlendState = &cb;
    VkPipelineDepthStencilStateCreateInfo ds_state{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    if (group_readback) {
        // Pixel i starts at stencil i. Every triangle decrements it, but only the i-th sees zero and writes color.
        ds_state.stencilTestEnable = VK_TRUE;
        auto& face = ds_state.front;
        face.compareOp = VK_COMPARE_OP_EQUAL;
        face.compareMask = face.writeMask = 0xff;
        face.failOp = face.passOp = VK_STENCIL_OP_DECREMENT_AND_WRAP;
        ds_state.back = face;
        gpi.pDepthStencilState = &ds_state;
    }
    gpi.layout = pl; gpi.renderPass = rp;
    VkPipeline pipe; CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpi, nullptr, &pipe));
    VkPipeline cpipe = VK_NULL_HANDLE;
    if (mesh_compute) {
        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = csm; cpi.stage.pName = "main";
        // VKH_SUBGROUP_SIZE goes to the mesh shader under test, now a compute stage.
        if (req_sg > 0 && f13q.subgroupSizeControl && req_sg >= sgProps.minSubgroupSize &&
            req_sg <= sgProps.maxSubgroupSize && (sgProps.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT))
            cpi.stage.pNext = &reqSg;
        cpi.layout = pl;
        LOADD(vkCreateComputePipelines);
        CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &cpipe));
    }
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = VkDeviceSize(W) * H * 16; bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer buf; CHECK(vkCreateBuffer(dev, &bi, nullptr, &buf));
    VkMemoryRequirements br; vkGetBufferMemoryRequirements(dev, buf, &br);
    VkMemoryAllocateInfo bai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    bai.allocationSize = br.size;
    bai.memoryTypeIndex = memtype(br.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory bmem; CHECK(vkAllocateMemory(dev, &bai, nullptr, &bmem));
    CHECK(vkBindBufferMemory(dev, buf, bmem, 0));
    std::vector<uint8_t> mesh_vertices;
    std::vector<uint32_t> mesh_indices;
    VkBuffer vbo = VK_NULL_HANDLE, ibo = VK_NULL_HANDLE;
    VkDeviceMemory vbo_mem = VK_NULL_HANDLE, ibo_mem = VK_NULL_HANDLE;
    if (draw_mesh_path) {
        if (!read_mesh(draw_mesh_path, vertex_stride, mesh_vertices, mesh_indices)) {
            std::fprintf(stderr, "VKH_DRAW_MESH: cannot read %s\n", draw_mesh_path);
            return 1;
        }
        auto host_buffer = [&](VkDeviceSize bytes, VkBufferUsageFlags usage, VkBuffer& b,
                               VkDeviceMemory& m, void*& p) -> bool {
            VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            ci.size = bytes; ci.usage = usage;
            if (vkCreateBuffer(dev, &ci, nullptr, &b) != VK_SUCCESS) return false;
            VkMemoryRequirements rq; vkGetBufferMemoryRequirements(dev, b, &rq);
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.allocationSize = rq.size;
            ai.memoryTypeIndex = memtype(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            return vkAllocateMemory(dev, &ai, nullptr, &m) == VK_SUCCESS &&
                   vkBindBufferMemory(dev, b, m, 0) == VK_SUCCESS &&
                   vkMapMemory(dev, m, 0, bytes, 0, &p) == VK_SUCCESS;
        };
        void* vmap = nullptr;
        void* imap = nullptr;
        if (!host_buffer(VkDeviceSize(mesh_vertices.size()), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vbo, vbo_mem, vmap) ||
            !host_buffer(VkDeviceSize(mesh_indices.size() * 4), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, ibo, ibo_mem, imap)) {
            std::fprintf(stderr, "VKH_DRAW_MESH: cannot create the vertex or index buffer\n");
            return 1;
        }
        std::memcpy(vmap, mesh_vertices.data(), mesh_vertices.size());
        std::memcpy(imap, mesh_indices.data(), mesh_indices.size() * 4);
    }
    void* heap_map = nullptr;
    if (heap_env) {
        LOADD(vkGetBufferDeviceAddress);
        VkBufferCreateInfo hbi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        hbi.size = heap_bytes;
        hbi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        VkBuffer heap; CHECK(vkCreateBuffer(dev, &hbi, nullptr, &heap));
        VkMemoryRequirements hr; vkGetBufferMemoryRequirements(dev, heap, &hr);
        VkMemoryAllocateFlagsInfo haf{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        haf.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo hai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        hai.pNext = &haf; hai.allocationSize = hr.size;
        hai.memoryTypeIndex = memtype(hr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDeviceMemory hmem; CHECK(vkAllocateMemory(dev, &hai, nullptr, &hmem));
        CHECK(vkBindBufferMemory(dev, heap, hmem, 0));
        void* hp; CHECK(vkMapMemory(dev, hmem, 0, heap_bytes, 0, &hp));
        heap_map = hp;
        std::memset(hp, 0, heap_bytes);
        for (auto [addr, word] : heap_words) std::memcpy(static_cast<char*>(hp) + addr, &word, 4);
        VkBufferDeviceAddressInfo hda{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        hda.buffer = heap;
        const uint64_t base = vkGetBufferDeviceAddress(dev, &hda);
        pc[32] = uint32_t(base); pc[33] = uint32_t(base >> 32);
    }
    const uint64_t mesh_groups = uint64_t(groups[0]) * groups[1] * groups[2];
    if (mesh_compute) {
        LOADD(vkGetBufferDeviceAddress);
        VkBufferCreateInfo obi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        obi.size = mesh_groups * mesh_slot;
        obi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        VkBuffer obuf; CHECK(vkCreateBuffer(dev, &obi, nullptr, &obuf));
        VkMemoryRequirements orq; vkGetBufferMemoryRequirements(dev, obuf, &orq);
        VkMemoryAllocateFlagsInfo oaf{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        oaf.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo oai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        oai.pNext = &oaf; oai.allocationSize = orq.size;
        oai.memoryTypeIndex = memtype(orq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDeviceMemory omem; CHECK(vkAllocateMemory(dev, &oai, nullptr, &omem));
        CHECK(vkBindBufferMemory(dev, obuf, omem, 0));
        void* op; CHECK(vkMapMemory(dev, omem, 0, obi.size, 0, &op));
        std::memset(op, 0, size_t(obi.size));   // a workgroup that never writes its header draws nothing
        VkBufferDeviceAddressInfo oda{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        oda.buffer = obuf;
        const uint64_t oaddr = vkGetBufferDeviceAddress(dev, &oda);
        pc[36] = uint32_t(oaddr); pc[37] = uint32_t(oaddr >> 32);
        std::fprintf(stderr, "mesh compute: %llu workgroups x %llu bytes, %u primitives each\n",
                     (unsigned long long)mesh_groups, (unsigned long long)mesh_slot, mesh_prims);
    }
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.queueFamilyIndex = qi;
    VkCommandPool pool; CHECK(vkCreateCommandPool(dev, &cpi, nullptr, &pool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cmd; CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHECK(vkBeginCommandBuffer(cmd, &cbi));
    VkClearValue clears[2]{};
    for (float& c : clears[0].color.float32) c = -1;   // marks untouched pixels
    if (clip_shifts) {
        const float shifts[] = {float(std::atof(argv[6])), float(std::atof(argv[7]))};
        std::memcpy(pc, shifts, sizeof(shifts));
    }
    auto push = reinterpret_cast<PFN_vkCmdPushConstants>(vkGetDeviceProcAddr(dev, "vkCmdPushConstants"));
    if (mesh_compute) {
        // The mesh shader as compute, then make its stores visible to the vertex shader's reads.
        LOADD(vkCmdDispatch); LOADD(vkCmdPipelineBarrier);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cpipe);
        push(cmd, pl, pcs, 0, pc_bytes, pc);
        vkCmdDispatch(cmd, groups[0], groups[1], groups[2]);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0,
                             1, &mb, 0, nullptr, 0, nullptr);
    }
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = rp; rbi.framebuffer = fb; rbi.renderArea = sc; rbi.clearValueCount = rpi.attachmentCount; rbi.pClearValues = clears;
    vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    if (group_readback) {
        LOADD(vkCmdClearAttachments);
        VkClearAttachment seed{VK_IMAGE_ASPECT_STENCIL_BIT};
        for (uint32_t i = 1; i < W; ++i) {
            seed.clearValue.depthStencil.stencil = i;
            VkClearRect pixel{{{int32_t(i), 0}, {1, 1}}, 0, 1};
            vkCmdClearAttachments(cmd, 1, &seed, 1, &pixel);
        }
        std::fprintf(stderr, "mesh workgroups: %u x %u x %u\n", groups[0], groups[1], groups[2]);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    push(cmd, pl, pcs, 0, pc_bytes, pc);
    // VKH_VERTICES=<n>: draw n vertices instead of one triangle, so a vertex shader fills whole waves (TEST-251).
    const char* vertices_env = std::getenv("VKH_VERTICES");
    if (draw_mesh_path) {
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vbo, &offset);
        vkCmdBindIndexBuffer(cmd, ibo, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, uint32_t(mesh_indices.size()), 1, 0, 0, 0);
    } else if (const unsigned vertices = vertices_env ? unsigned(std::strtoul(vertices_env, nullptr, 10)) : 0;
               vertices && !mesh)
        vkCmdDraw(cmd, vertices, 1, 0, 0);
    else if (mesh_compute)
        vkCmdDraw(cmd, uint32_t(mesh_groups * mesh_prims * 3), 1, 0, 0);
    else if (mesh)
        reinterpret_cast<PFN_vkCmdDrawMeshTasksEXT>(vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksEXT"))(cmd, groups[0], groups[1], groups[2]);
    else
        vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
    VkBufferImageCopy reg{}; reg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; reg.imageExtent = {W, H, 1};
    vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &reg);
    CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    CHECK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
    CHECK(vkQueueWaitIdle(q));
    void* p; CHECK(vkMapMemory(dev, bmem, 0, bi.size, 0, &p));
    std::ofstream output(argv[3], std::ios::binary);
    output.write(static_cast<const char*>(p), std::streamsize(bi.size));
    output.close();
    if (!output) { std::fprintf(stderr, "failed to write %s\n", argv[3]); return 1; }
    if (const char* dump = std::getenv("VKH_HEAP_DUMP"); dump && *dump && heap_map) {
        std::ofstream hf(dump, std::ios::binary);
        hf.write(static_cast<const char*>(heap_map), heap_bytes);
        hf.close();
        if (!hf) { std::fprintf(stderr, "failed to write %s\n", dump); return 1; }
    }
    std::fprintf(stderr, "ok %ux%u -> %s\n", W, H, argv[3]);
    return 0;
}
