// Headless Vulkan compute test for translated compute shaders.
// Usage: vk_compute_test cs.spv groups expect [--require-wave64]
//   expect = "mirror64": out[g*64 + t] == g*64 + (63 - t)   (runtime/test/shaders/cs_lds.asm)
//            "workgroup_yz": pass 2x3x4 for groups; check the Y/Z histogram (cs_workgroup_yz.asm)
//            "exec64", "wavemask", "bvh": see cs_exec64.asm, cs_wavemask.asm, cs_bvh.asm; anything else prints out[0..7]
//   VKC_BENCH=N (env): dispatch N more times after the checked one and print the median / min / max wall time of one
//   dispatch (submit to queue idle) as "bench ms: median=.. min=.. max=.. n=..". The buffer is not cleared in between.
//   VKC_OUT_WORDS=N (env): make the output buffer at least N words (default groups * 64), so a single group can
//   write more than one word per thread. VKC_DUMP=<file> (env): write the whole buffer to <file> after the dispatch.
//   --require-wave64: require full 64-wide subgroups (local size X must be a multiple of 64), or exit 77.
//   VKC_SGPRS=w4,w5,... (env): set s4, s5, ... (decimal or 0x hex, up to s31), the user SGPRs after the output V#,
//   so a test can take input words (cs_user_sgprs.asm). They are zero otherwise. Same syntax as vk_harness's VKH_SGPRS.
// Contract (docs/design.md): push constants = s0..s31 (128 bytes), heap base lo/hi, code base lo/hi (144 bytes).
// LEONIDA_PC_TAIL_UBO=1 (modules from rdna2spirv.py --pc-tail-ubo): s0..s31 only, the rest at set 1 (vk_pc_tail.h).
// The output buffer is the "heap": Xbox VA 0 maps to its device address, and s[0:3] hold a V# with base 0.
#include "vulkan_loader.h"
#include "vk_device_pick.h"
#include "vk_shader_fma.h"
#include "vk_pc_tail.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

static PFN_vkGetInstanceProcAddr gipa;
#define CHECK(x)                                                                         \
    do {                                                                                 \
        VkResult r_ = (x);                                                               \
        if (r_ != VK_SUCCESS) {                                                          \
            std::fprintf(stderr, "FAILED %s -> %d (line %d)\n", #x, r_, __LINE__);       \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

// The same word list parser as vk_harness: decimal or 0x hex, no sign or space, and "1,2," is rejected.
static bool parse_u32(const char*& s, uint32_t& v) {
    const bool hex = s[0] == '0' && (s[1] == 'x' || s[1] == 'X');
    const char* digits = hex ? s + 2 : s;
    if (!(hex ? std::isxdigit(uint8_t(*digits)) : std::isdigit(uint8_t(*digits)))) return false;
    char* end;
    unsigned long long x = std::strtoull(digits, &end, hex ? 16 : 10);
    if (x > 0xFFFFFFFFull) return false;
    v = uint32_t(x); s = end;
    return true;
}

static bool next_item(const char*& s) {
    if (!*s) return true;
    return *s++ == ',' && *s;
}

// VKC_SGPRS: s4, s5, ... in order, at most s31 (s[0:3] stay the output V#).
static bool parse_user_sgprs(const char* s, uint32_t* pc) {
    for (uint32_t i = 4; *s; i++) {
        if (i == 32 || !parse_u32(s, pc[i]) || !next_item(s)) return false;
    }
    return true;
}

int main(int argc, char** argv) {
    const bool require_wave64 = argc == 5 && !std::strcmp(argv[4], "--require-wave64");
    if (argc != 4 && !require_wave64) {
        std::fprintf(stderr, "usage: vk_compute_test cs.spv groups expect [--require-wave64]\n");
        return 2;
    }
    uint32_t pc[36] = {};
    const char* sgprs_env = std::getenv("VKC_SGPRS");
    if (sgprs_env && !parse_user_sgprs(sgprs_env, pc)) {
        std::fprintf(stderr, "VKC_SGPRS: expected up to 28 comma-separated words for s4..s31, got \"%s\"\n", sgprs_env);
        return 2;
    }
    const bool workgroup_yz = !std::strcmp(argv[3], "workgroup_yz");
    if (workgroup_yz && std::strcmp(argv[2], "2x3x4")) {
        std::fprintf(stderr, "workgroup_yz expects a 2x3x4 dispatch\n");
        return 2;
    }
    const uint32_t groups = workgroup_yz ? 2u : uint32_t(atoi(argv[2]));
    const uint32_t groups_y = workgroup_yz ? 3u : 1u, groups_z = workgroup_yz ? 4u : 1u;
    const uint32_t yz_bins = (groups_y + 1) * (groups_z + 1);
    const uint32_t origins = workgroup_yz ? groups * groups_y * groups_z : 0;
    const char* out_words_env = std::getenv("VKC_OUT_WORDS");
    const char* dump_path = std::getenv("VKC_DUMP");
    uint32_t out_words = 0;
    if (out_words_env) {
        char* end;
        const unsigned long n = std::strtoul(out_words_env, &end, 10);
        if (*out_words_env < '0' || *out_words_env > '9' || *end || n == 0 || n > (1ul << 24)) {
            std::fprintf(stderr, "VKC_OUT_WORDS: expected 1..16777216, got \"%s\"\n", out_words_env);
            return 2;
        }
        out_words = uint32_t(n);
    }
    gipa = load_vulkan();
    if (!gipa) return 1;
#define LOADI(name) auto name = reinterpret_cast<PFN_##name>(gipa(inst, #name))
    VkInstance inst = VK_NULL_HANDLE;
    LOADI(vkCreateInstance);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    CHECK(vkCreateInstance(&ici, nullptr, &inst));
    LOADI(vkEnumeratePhysicalDevices); LOADI(vkGetPhysicalDeviceProperties); LOADI(vkCreateDevice);
    LOADI(vkGetPhysicalDeviceProperties2); LOADI(vkGetPhysicalDeviceFeatures2);
    LOADI(vkGetPhysicalDeviceQueueFamilyProperties); LOADI(vkGetDeviceProcAddr);
    LOADI(vkGetPhysicalDeviceMemoryProperties);
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
    VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    subgroup.pNext = &sgProps;
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &subgroup;
    vkGetPhysicalDeviceProperties2(pd, &props2);
    std::fprintf(stderr, "device: %s (subgroup %u..%u)\n", props.deviceName, sgProps.minSubgroupSize, sgProps.maxSubgroupSize);
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
    uint32_t qi = 0;
    while (qi < qn && !(qf[qi].queueFlags & VK_QUEUE_COMPUTE_BIT)) qi++;
    float prio = 1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkPhysicalDeviceVulkan13Features f13q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceFeatures2 f2q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2q.pNext = &f13q;
    vkGetPhysicalDeviceFeatures2(pd, &f2q);
    const char* sg_env = std::getenv("VKC_SUBGROUP_SIZE");
    const uint32_t req_sg = require_wave64 ? 64u : (sg_env ? uint32_t(std::atoi(sg_env)) : 0u);
    const bool can_request_size = req_sg > 0 && f13q.subgroupSizeControl &&
        req_sg >= sgProps.minSubgroupSize && req_sg <= sgProps.maxSubgroupSize &&
        (sgProps.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT);
    if (require_wave64 && (!f13q.computeFullSubgroups || (!can_request_size && subgroup.subgroupSize != 64))) {
        std::fprintf(stderr, "full 64-wide compute subgroups unavailable\n");
        return 77;
    }
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<char> code((std::istreambuf_iterator<char>(f)), {});
    const bool needs_float64 = leonida::vk_test::spirv_declares_capability(code.data(), code.size(), 10);  // Float64
    if (needs_float64 && !f2q.features.shaderFloat64) {
        std::fprintf(stderr, "unsupported: %s uses Float64 and the device has no shaderFloat64\n", argv[1]);
        return 77;
    }
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = VK_TRUE;
    if (f13q.subgroupSizeControl || require_wave64) {
        f13.subgroupSizeControl = f13q.subgroupSizeControl;
        f13.computeFullSubgroups = require_wave64 ? VK_TRUE : VK_FALSE;
        f12.pNext = &f13;
    }
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.features.shaderInt64 = VK_TRUE;
    f2.features.shaderFloat64 = needs_float64;
    f2.pNext = &f12;
    // OpFmaKHR (rdna2spirv.py --fma khr): VK_KHR_shader_fma, when the device has it.
    leonida::vk_test::ShaderFma fma;
    fma.query(gipa, inst, pd);
    fma.link_after(&f2);
    const char* fma_extension = fma.extension();
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci; dci.pNext = &f2;
    if (fma.supported()) { dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = &fma_extension; }
    VkDevice dev;
    CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
#define LOADD(name) auto name = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(dev, #name))
    LOADD(vkGetDeviceQueue); LOADD(vkCreateBuffer); LOADD(vkGetBufferMemoryRequirements); LOADD(vkAllocateMemory);
    LOADD(vkBindBufferMemory); LOADD(vkMapMemory); LOADD(vkGetBufferDeviceAddress); LOADD(vkCreateShaderModule);
    LOADD(vkCreatePipelineLayout); LOADD(vkCreateComputePipelines); LOADD(vkCreateCommandPool);
    LOADD(vkAllocateCommandBuffers); LOADD(vkBeginCommandBuffer); LOADD(vkCmdBindPipeline);
    LOADD(vkCmdPushConstants); LOADD(vkCmdDispatch); LOADD(vkEndCommandBuffer); LOADD(vkQueueSubmit);
    LOADD(vkQueueWaitIdle); LOADD(vkCmdPipelineBarrier); LOADD(vkCmdDispatchBase); LOADD(vkCmdBindDescriptorSets);
    VkQueue q; vkGetDeviceQueue(dev, qi, 0, &q);

    // Output buffer = heap. Host-visible so the result can be read back directly.
    VkDeviceSize words = workgroup_yz ? yz_bins * (1 + origins) : VkDeviceSize(groups) * 64;
    if (out_words > words) words = out_words;
    // "bvh" writes its nodes at heap offsets 0x100 to 0x264 below, past the 256 bytes one group's 64 words give: the
    // shader then read them through the device address outside the buffer (VUID-RuntimeSpirv-PhysicalStorageBuffer64-11819).
    const bool bvh = !std::strcmp(argv[3], "bvh");
    if (bvh && words < 0x280 / sizeof(uint32_t)) words = 0x280 / sizeof(uint32_t);
    const VkDeviceSize bytes = words * sizeof(uint32_t);
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VkBuffer buf; CHECK(vkCreateBuffer(dev, &bci, nullptr, &buf));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, buf, &mr);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t mt = 0;
    for (; mt < mp.memoryTypeCount; mt++)
        if ((mr.memoryTypeBits >> mt & 1) && (mp.memoryTypes[mt].propertyFlags &
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) break;
    VkMemoryAllocateFlagsInfo maf{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    maf.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size; mai.memoryTypeIndex = mt; mai.pNext = &maf;
    VkDeviceMemory mem; CHECK(vkAllocateMemory(dev, &mai, nullptr, &mem));
    CHECK(vkBindBufferMemory(dev, buf, mem, 0));
    void* map; CHECK(vkMapMemory(dev, mem, 0, bytes, 0, &map));
    std::memset(map, workgroup_yz || !std::strcmp(argv[3], "exec64") ? 0 : 0xEE, size_t(bytes));
    if (!std::strcmp(argv[3], "bvh")) {
        // AMD-format BVH in the heap (runtime/test/shaders/cs_bvh.asm): box32 root at 0x100 whose children are a
        // triangle at z=10 (node 0x40, addr 0x200), a triangle at z=5 (node 0x48, addr 0x240), a box the ray
        // misses, and an inactive (NaN) box. Node id = address >> 3 | type.
        auto f32 = [&](uint32_t at, float v) { std::memcpy(static_cast<char*>(map) + at, &v, 4); };
        auto u32 = [&](uint32_t at, uint32_t v) { std::memcpy(static_cast<char*>(map) + at, &v, 4); };
        u32(0x100, 0x40); u32(0x104, 0x48); u32(0x108, 0x60); u32(0x10C, 0x68);
        const float boxes[4][6] = {{0, 0, 10, 1, 1, 10}, {0, 0, 5, 1, 1, 5}, {5, 0, 0, 6, 1, 1}, {NAN, 0, 0, 1, 1, 1}};
        for (int i = 0; i < 4; i++) for (int c = 0; c < 6; c++) f32(0x110 + 24 * i + 4 * c, boxes[i][c]);
        const float tri[2][9] = {{0, 0, 10, 1, 0, 10, 0, 1, 10}, {0, 0, 5, 1, 0, 5, 0, 1, 5}};
        for (int t = 0; t < 2; t++) for (int c = 0; c < 9; c++) f32(0x200 + 0x40 * t + 4 * c, tri[t][c]);
        // Triangle-ID word at node +60: byte 0 (TRI0) = 0x09 returns vertex 1's and vertex 2's weights, what a
        // builder writes (AMD GPURT WriteTriangleIdField); the translator reads it since the remap fix.
        for (int t = 0; t < 2; t++) u32(0x200 + 0x40 * t + 60, 0x09);
    }
    VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bai.buffer = buf;
    const uint64_t heap = vkGetBufferDeviceAddress(dev, &bai);

    if (!fma.accepts(code.data(), code.size(), argv[1])) return 77;
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = code.size(); smi.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule sm; CHECK(vkCreateShaderModule(dev, &smi, nullptr, &sm));
    // 144 push-constant bytes: s0..s31, heap base, code base. With LEONIDA_PC_TAIL_UBO=1 (rdna2spirv.py
    // --pc-tail-ubo) only s0..s31; the tail, if the module reads it, is a uniform block at set 1 (vk_pc_tail.h), one
    // slot for the dispatch and one per workgroup_yz origin.
    const bool pc_tail = leonida::vk_test::pc_tail_requested();
    const uint32_t pc_bytes = pc_tail ? 128u : 144u;
    leonida::vk_test::PcTail tail;
    const bool tail_block = pc_tail &&
        leonida::vk_test::spirv_declares_set(code.data(), code.size(), leonida::vk_test::pc_tail_set);
    if (tail_block && !tail.create(vkGetDeviceProcAddr, dev, mp, props.limits.minUniformBufferOffsetAlignment,
                                   VK_SHADER_STAGE_COMPUTE_BIT, 1 + origins)) {
        std::fprintf(stderr, "FAILED creating the --pc-tail-ubo block\n");
        return 1;
    }
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_bytes};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    const VkDescriptorSetLayout tail_layouts[] = {tail.empty_layout, tail.set_layout};
    if (tail_block) { pli.setLayoutCount = 2; pli.pSetLayouts = tail_layouts; }
    VkPipelineLayout pl; CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    if (workgroup_yz) cpi.flags |= VK_PIPELINE_CREATE_DISPATCH_BASE_BIT;
    cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = sm; cpi.stage.pName = "main";
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo reqSg{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    reqSg.requiredSubgroupSize = req_sg;
    if (can_request_size)
        cpi.stage.pNext = &reqSg;
    if (require_wave64)
        cpi.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    cpi.layout = pl;
    VkPipeline pipe; CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe));
    if (require_wave64) std::printf("compute subgroup size: 64\n");
    else if (req_sg) {
        // VKC_SUBGROUP_SIZE: say whether the pipeline really runs at that size, so a runner can skip a test the device
        // can't pin instead of judging words from whatever size the driver picked (#774).
        const bool fixed = sgProps.minSubgroupSize == sgProps.maxSubgroupSize && subgroup.subgroupSize == req_sg;
        if (can_request_size || fixed) std::printf("compute subgroup size: %u\n", req_sg);
        else std::printf("compute subgroup size: %u not available\n", req_sg);
    }

    pc[0] = 0; pc[1] = 0; pc[2] = 0xFFFFFFFFu; pc[3] = 0x31016FACu;     // V#: base 0, stride 0, num_records max, OOB_SELECT 3 (raw)
    pc[32] = uint32_t(heap); pc[33] = uint32_t(heap >> 32);    // heap base
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex = qi;
    VkCommandPool pool; CHECK(vkCreateCommandPool(dev, &cpci, nullptr, &pool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cmd; CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHECK(vkBeginCommandBuffer(cmd, &cbi));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_bytes, pc);
    auto bind_tail = [&](uint32_t slot) {
        if (!tail_block) return;
        tail.write(slot, pc, 4);
        const uint32_t offset = tail.offset(slot);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, leonida::vk_test::pc_tail_set, 1, &tail.set,
                                1, &offset);
    };
    bind_tail(0);
    vkCmdDispatch(cmd, groups, groups_y, groups_z);
    // A permutation within one axis can leave the full-grid histogram unchanged. Dispatch one group at
    // each known origin into its own histogram, selected by the host independently of the shader's SGPRs.
    for (uint32_t g = 0; g < origins; ++g) {
        const uint64_t base = heap + (g + 1) * yz_bins * sizeof(uint32_t);
        const uint32_t address[] = {uint32_t(base), uint32_t(base >> 32)};
        if (pc_tail) {
            pc[32] = address[0]; pc[33] = address[1];     // the heap base is the tail's dwords 0..1
            bind_tail(g + 1);
        } else {
            vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 32 * sizeof(uint32_t), sizeof(address), address);
        }
        vkCmdDispatchBase(cmd, g % groups, (g / groups) % groups_y, g / (groups * groups_y), 1, 1, 1);
    }
    VkMemoryBarrier readback{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    readback.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    readback.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                         1, &readback, 0, nullptr, 0, nullptr);
    CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    CHECK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
    CHECK(vkQueueWaitIdle(q));
    if (const char* b = std::getenv("VKC_BENCH")) {
        std::vector<double> ms;
        for (int i = 0, n = std::atoi(b); i < n; i++) {
            auto t0 = std::chrono::steady_clock::now();
            CHECK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
            CHECK(vkQueueWaitIdle(q));
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        if (!ms.empty()) {
            std::sort(ms.begin(), ms.end());
            std::printf("bench ms: median=%.3f min=%.3f max=%.3f n=%zu\n", ms[ms.size() / 2], ms.front(), ms.back(), ms.size());
        }
        uint32_t h = 2166136261u;   // FNV-1a of the whole output buffer, to compare configurations
        for (VkDeviceSize i = 0; i < bytes; i++) h = (h ^ static_cast<const unsigned char*>(map)[i]) * 16777619u;
        std::printf("bench out fnv1a: %08x\n", h);
    }

    if (dump_path) {
        std::ofstream dump(dump_path, std::ios::binary);
        dump.write(static_cast<const char*>(map), std::streamsize(bytes));
        dump.close();
        if (dump.fail()) {
            std::fprintf(stderr, "VKC_DUMP: could not write %s\n", dump_path);
            return 1;
        }
    }

    const uint32_t* out = static_cast<const uint32_t*>(map);
    uint32_t bad = 0, first_bad = ~0u;
    if (workgroup_yz) {
        // X is disabled in RSRC2. Both X groups must contribute to each valid (Y,Z) bin.
        // The last column and row catch out-of-range register values without indexing past the buffer.
        for (uint32_t z = 0; z <= groups_z; ++z) {
            for (uint32_t y = 0; y <= groups_y; ++y) {
                const uint32_t want = y < groups_y && z < groups_z ? groups : 0;
                const uint32_t got = out[z * (groups_y + 1) + y];
                if (got != want) {
                    std::fprintf(stderr, "(Y=%u, Z=%u): got %u, expected %u\n", y, z, got, want);
                    ++bad;
                }
            }
        }
        std::printf("2x3x4 workgroup Y/Z: %u mismatches\n", bad);
        uint32_t wrong_origins = 0;
        for (uint32_t g = 0; g < origins; ++g) {
            const uint32_t y = (g / groups) % groups_y, z = g / (groups * groups_y);
            const uint32_t* histogram = out + (g + 1) * yz_bins;
            for (uint32_t bin = 0; bin < yz_bins; ++bin) {
                const uint32_t want = bin == z * (groups_y + 1) + y ? 1u : 0u;
                if (histogram[bin] != want) {
                    std::fprintf(stderr, "origin (%u,%u,%u), bin %u: got %u, expected %u\n",
                                 g % groups, y, z, bin, histogram[bin], want);
                    ++wrong_origins;
                    break;
                }
            }
        }
        std::printf("single-group origins: %u/%u matched\n", origins - wrong_origins, origins);
        return bad || wrong_origins ? 1 : 0;
    }
    if (!std::strcmp(argv[3], "mirror64")) {
        for (uint32_t g = 0; g < groups; g++)
            for (uint32_t t = 0; t < 64; t++) {
                uint32_t want = g * 64 + (63 - t), got = out[g * 64 + t];
                if (got != want) { if (!bad) first_bad = g * 64 + t; bad++; }
            }
    }
    if (!std::strcmp(argv[3], "exec64")) {    // runtime/test/shaders/cs_exec64.asm
        for (uint32_t g = 0; g < groups; g++)
            for (uint32_t t = 0; t < 64; t++) {
                uint32_t want = (t >= 16 && t < 48) ? 7u : 0u, got = out[g * 64 + t];
                if (got != want) { if (!bad) first_bad = g * 64 + t; bad++; }
            }
    }
    if (!std::strcmp(argv[3], "wavemask")) {  // runtime/test/shaders/cs_wavemask.asm
        for (uint32_t g = 0; g < groups; g++)
            for (uint32_t t = 0; t < 64; t++) {
                // lanes 16-47: on again after the parked mask (0x80) and after the one through a VGPR (0x40),
                // plus how many set lanes are below this one
                uint32_t top = (t >= 16 && t < 48) ? 0xc0u + t - 16 : 0;
                uint32_t want = 65u | 16u << 8 | 8u << 16 | top << 24, got = out[g * 64 + t];
                if (got != want) { if (!bad) first_bad = g * 64 + t; bad++; }
            }
    }
    if (!std::strcmp(argv[3], "bvh")) {
        const uint32_t want[8] = {0x48, 0x40, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xC1200000u, 0xBF800000u, 0xBE800000u, 0xBE800000u};
        for (int i = 0; i < 8; i++) if (out[i] != want[i]) { if (!bad) first_bad = i; bad++; }
        std::printf("bvh out = %08x %08x %08x %08x | %08x %08x %08x %08x\n", out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
    }
    std::printf("%u groups x 64: %u mismatches", groups, bad);
    if (bad) std::printf(" (first at %u: got 0x%08x)", first_bad, out[first_bad]);
    std::printf("\nout[0..7] = %u %u %u %u %u %u %u %u\n", out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
    return bad ? 1 : 0;
}
