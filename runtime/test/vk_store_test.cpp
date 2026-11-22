// Headless Vulkan test for translated typed image stores and loads (runtime/test/shaders/cs_imgstore.asm and
// cs_imgload.asm).
// Usage: vk_store_test cs.spv [load|load-d16]
// The shader builds three T#s (word1 FORMAT = 56 rgba8 unorm, 20 r32 uint, 21 r32 sint) and stores to three 8x8
// images. The translator picks the float / uint / sint view (set 0, bindings 8 / 9 / 10 for 2D) from the FORMAT
// field, and the bindless index from the descriptor hash table (binding 2). As in the runtime, each image has its
// own index in the 2D arrays (1, 2, 3; index 0 is the fallback) and its view in its own format at that index of the
// sampled array (binding 0), which typed image_load fetches from. Checks every texel.
// "load" (cs_imgload.asm): the uint and sint images are filled before the dispatch, and the shader loads both
// texels with image_load and stores their sum into the sint image. The device is made without
// shaderStorageImageReadWithoutFormat, and a module that declares StorageImageReadWithoutFormat fails (SHADER-151).
#include "vulkan_loader.h"
#include "vk_device_pick.h"
#include "vk_shader_fma.h"

#include <cmath>
#include <cstdio>
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

static uint32_t key_hash(const uint32_t k[8]) {     // same FNV-1a as the translator's desc_lookup
    uint32_t h = 2166136261u;
    for (int i = 0; i < 8; i++) h = (h ^ k[i]) * 16777619u;
    return h;
}

static void table_insert(std::vector<uint32_t>& t, const uint32_t key[8], uint32_t index) {
    uint32_t mask = t[0], h = key_hash(key);
    for (uint32_t k = 0; k < 8; k++) {
        uint32_t base = ((h + k) & mask) * 10;
        bool used = false;
        for (int j = 0; j < 8; j++) used |= t[base + 1 + j] != 0;
        if (used) continue;
        for (int j = 0; j < 8; j++) t[base + 1 + j] = key[j];
        t[base + 9] = index;
        return;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: vk_store_test cs.spv\n"); return 2; }
    gipa = load_vulkan();
    if (!gipa) return 1;
#define LOADI(name) auto name = reinterpret_cast<PFN_##name>(gipa(inst, #name))
    VkInstance inst = VK_NULL_HANDLE;
    LOADI(vkCreateInstance);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    CHECK(vkCreateInstance(&ici, nullptr, &inst));
    LOADI(vkEnumeratePhysicalDevices); LOADI(vkGetPhysicalDeviceProperties); LOADI(vkCreateDevice);
    LOADI(vkGetPhysicalDeviceQueueFamilyProperties); LOADI(vkGetDeviceProcAddr);
    LOADI(vkGetPhysicalDeviceMemoryProperties); LOADI(vkGetPhysicalDeviceFeatures);
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
    std::fprintf(stderr, "device: %s\n", props.deviceName);
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
    uint32_t qi = 0;
    while (qi < qn && !(qf[qi].queueFlags & VK_QUEUE_COMPUTE_BIT)) qi++;
    float prio = 1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    // The typed stores need format-less storage image writes: without them, report that and exit 77 (unsupported,
    // as vk_compute_test does) instead of failing vkCreateDevice. Typed loads need no storage image feature.
    VkPhysicalDeviceFeatures have{};
    vkGetPhysicalDeviceFeatures(pd, &have);
    if (!have.shaderStorageImageWriteWithoutFormat || !have.shaderInt64) {
        std::printf("unsupported: shaderStorageImageWriteWithoutFormat=%u shaderInt64=%u\n",
                    have.shaderStorageImageWriteWithoutFormat, have.shaderInt64);
        return 77;
    }
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.runtimeDescriptorArray = VK_TRUE; f12.bufferDeviceAddress = VK_TRUE;
    f12.shaderStorageImageArrayNonUniformIndexing = VK_TRUE;
    f12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    // "load" mode (cs_imgload.asm): the sint image receives the sum of the two texels read back with image_load.
    const bool packed_load = argc > 2 && !std::strcmp(argv[2], "load-d16");
    const bool load_mode = packed_load || (argc > 2 && !std::strcmp(argv[2], "load"));
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<char> code((std::istreambuf_iterator<char>(f)), {});
    if (code.size() < 20 || code.size() % 4) { std::fprintf(stderr, "%s: not a SPIR-V module\n", argv[1]); return 1; }
    if (load_mode) {
        // Typed image_load fetches through the sampled arrays, so it must not need a feature Intel Xe-LP lacks
        // (SHADER-151). shaderStorageImageReadWithoutFormat stays off; a module declaring it would be invalid here.
        std::vector<uint32_t> words(code.size() / 4);
        std::memcpy(words.data(), code.data(), code.size());
        for (std::size_t i = 5; i < words.size() && (words[i] >> 16);) {
            const uint32_t op = words[i] & 0xffff, count = words[i] >> 16;
            // OpCapability (17) StorageImageReadWithoutFormat (55; spirv/unified1/spirv.h)
            if (op == 17 && count == 2 && i + 1 < words.size() && words[i + 1] == 55) {
                std::printf("FAILED: the module declares StorageImageReadWithoutFormat\n");
                return 1;
            }
            i += count;
        }
    }
    f2.features.shaderInt64 = VK_TRUE;
    f2.pNext = &f12;
    leonida::vk_test::ShaderFma fma;      // OpFmaKHR (rdna2spirv.py --fma khr)
    fma.query(gipa, inst, pd);
    if (!fma.accepts(code.data(), code.size(), argv[1])) return 77;
    fma.link_after(&f2);
    const char* fma_extension = fma.extension();
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci; dci.pNext = &f2;
    if (fma.supported()) { dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = &fma_extension; }
    VkDevice dev;
    CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
#define LOADD(name) auto name = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(dev, #name))
    LOADD(vkGetDeviceQueue); LOADD(vkCreateBuffer); LOADD(vkGetBufferMemoryRequirements); LOADD(vkAllocateMemory);
    LOADD(vkBindBufferMemory); LOADD(vkMapMemory); LOADD(vkCreateShaderModule); LOADD(vkCreateImage);
    LOADD(vkGetImageMemoryRequirements); LOADD(vkBindImageMemory); LOADD(vkCreateImageView); LOADD(vkCreateSampler);
    LOADD(vkCreateDescriptorSetLayout); LOADD(vkCreatePipelineLayout); LOADD(vkCreateComputePipelines);
    LOADD(vkCreateDescriptorPool); LOADD(vkAllocateDescriptorSets); LOADD(vkUpdateDescriptorSets);
    LOADD(vkCreateCommandPool); LOADD(vkAllocateCommandBuffers); LOADD(vkBeginCommandBuffer);
    LOADD(vkCmdBindPipeline); LOADD(vkCmdBindDescriptorSets); LOADD(vkCmdPushConstants); LOADD(vkCmdDispatch);
    LOADD(vkCmdPipelineBarrier); LOADD(vkCmdCopyImageToBuffer); LOADD(vkCmdCopyBufferToImage);
    LOADD(vkEndCommandBuffer); LOADD(vkQueueSubmit);
    LOADD(vkQueueWaitIdle);
    VkQueue q; vkGetDeviceQueue(dev, qi, 0, &q);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    auto memtype = [&](uint32_t bits, VkMemoryPropertyFlags want) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((bits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
        return 0u;
    };
    auto make_buffer = [&](VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer* b, void** map) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = size; bi.usage = usage;
        VkResult r = vkCreateBuffer(dev, &bi, nullptr, b);
        if (r) return r;
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, *b, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDeviceMemory m;
        if ((r = vkAllocateMemory(dev, &mai, nullptr, &m))) return r;
        if ((r = vkBindBufferMemory(dev, *b, m, 0))) return r;
        return vkMapMemory(dev, m, 0, size, 0, map);
    };

    // T#s exactly as the shader builds them (cs_imgstore.asm): word0 = 0x1000 * (k + 1), word1 = FORMAT << 20,
    // words 2..7 = 0x11 * (k + 1).
    const uint32_t fmts[3] = {56, 20, 21};
    const VkFormat vkf[3] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT};
    std::vector<uint32_t> table(1 + 16 * 10 + 10, 0);
    table[0] = 15;
    for (uint32_t k = 0; k < 3; k++) {
        uint32_t t[8] = {0x1000 * (k + 1), fmts[k] << 20};
        for (int j = 2; j < 8; j++) t[j] = 0x11 * (k + 1);
        table_insert(table, t, k + 1);     // its index in the 2D arrays; 0 is the fallback, as in the runtime
    }
    VkBuffer tb; void* tbp;
    CHECK(make_buffer(table.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &tb, &tbp));
    std::memcpy(tbp, table.data(), table.size() * 4);

    const uint32_t W = 8;
    VkImage img[3]; VkImageView view[3];
    for (int k = 0; k < 3; k++) {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D; ii.format = vkf[k]; ii.extent = {W, W, 1}; ii.mipLevels = 1; ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                   VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        CHECK(vkCreateImage(dev, &ii, nullptr, &img[k]));
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, img[k], &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size; mai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory m; CHECK(vkAllocateMemory(dev, &mai, nullptr, &m));
        CHECK(vkBindImageMemory(dev, img[k], m, 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = img[k]; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = vkf[k];
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        CHECK(vkCreateImageView(dev, &vi, nullptr, &view[k]));
    }
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    VkSampler sampler; CHECK(vkCreateSampler(dev, &sci, nullptr, &sampler));
    // Set 0: binding 0 = sampled images, 1 = samplers, 2 = hash table, 8/9/10 = float/uint/sint 2D storage image arrays.
    // Four elements each: index 0 stands for the runtime's fallback, image k is at index k + 1. Binding 0 holds each
    // image's view in its own format (typed image_load reads the uint and sint ones through integer aliases); every
    // element of a storage array holds the image of its class, so all of them are valid descriptors.
    constexpr uint32_t N = 4;
    VkDescriptorSetLayoutBinding b[6]{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, N, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    for (int k = 0; k < 3; k++) b[3 + k] = {uint32_t(8 + k), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, N, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 6; dli.pBindings = b;
    VkDescriptorSetLayout dsl; CHECK(vkCreateDescriptorSetLayout(dev, &dli, nullptr, &dsl));
    VkDescriptorPoolSize ps[4] = {
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, N},
        {VK_DESCRIPTOR_TYPE_SAMPLER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 * N},
    };
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1; dpi.poolSizeCount = 4; dpi.pPoolSizes = ps;
    VkDescriptorPool dp; CHECK(vkCreateDescriptorPool(dev, &dpi, nullptr, &dp));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = dp; dai.descriptorSetCount = 1; dai.pSetLayouts = &dsl;
    VkDescriptorSet ds; CHECK(vkAllocateDescriptorSets(dev, &dai, &ds));
    VkDescriptorBufferInfo tbi{tb, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo sii[N] = {{VK_NULL_HANDLE, view[0], VK_IMAGE_LAYOUT_GENERAL},
                                    {VK_NULL_HANDLE, view[0], VK_IMAGE_LAYOUT_GENERAL},
                                    {VK_NULL_HANDLE, view[1], VK_IMAGE_LAYOUT_GENERAL},
                                    {VK_NULL_HANDLE, view[2], VK_IMAGE_LAYOUT_GENERAL}};
    VkDescriptorImageInfo smpi{sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    VkDescriptorImageInfo dii[3][N];
    VkWriteDescriptorSet wr[6]{};
    for (auto& w : wr) { w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w.dstSet = ds; w.descriptorCount = 1; }
    wr[0].dstBinding = 0; wr[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; wr[0].pImageInfo = sii;
    wr[0].descriptorCount = N;
    wr[1].dstBinding = 1; wr[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; wr[1].pImageInfo = &smpi;
    wr[2].dstBinding = 2; wr[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[2].pBufferInfo = &tbi;
    for (int k = 0; k < 3; k++) {
        for (uint32_t e = 0; e < N; e++) dii[k][e] = {VK_NULL_HANDLE, view[k], VK_IMAGE_LAYOUT_GENERAL};
        wr[3 + k].dstBinding = 8 + k; wr[3 + k].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        wr[3 + k].pImageInfo = dii[k]; wr[3 + k].descriptorCount = N;
    }
    vkUpdateDescriptorSets(dev, 6, wr, 0, nullptr);

    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = code.size(); smi.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule sm; CHECK(vkCreateShaderModule(dev, &smi, nullptr, &sm));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 144};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pl; CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = sm; cpi.stage.pName = "main";
    cpi.layout = pl;
    VkPipeline pipe; CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe));

    VkBuffer rb[3]; void* rbp[3];
    for (int k = 0; k < 3; k++) CHECK(make_buffer(W * W * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rb[k], &rbp[k]));
    // Load mode: what the uint and sint images hold before the dispatch, texel (x, y) at i = 8y + x.
    // The last row of the D16 fixture exercises both sides of each 16-bit boundary.
    struct PackedLoad { uint32_t u; int32_t s, sum; };
    constexpr PackedLoad boundary[] = {
        {65535, 0, 65535}, {65536, 0, 65535}, {65537, 0, 65535}, {0, 32767, 32767},
        {0, 32768, 32767}, {0, -32768, -32768}, {0, -32769, -32768}, {65536, -32769, 32767},
    };
    static_assert(sizeof(boundary) / sizeof(boundary[0]) == W);
    VkBuffer fill[2]{}; void* fillp[2]{};
    if (load_mode) {
        for (int j = 0; j < 2; j++)
            CHECK(make_buffer(W * W * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &fill[j], &fillp[j]));
        for (uint32_t i = 0; i < W * W; i++) {
            const bool edge = packed_load && i / W == W - 1;
            static_cast<uint32_t*>(fillp[0])[i] = edge ? boundary[i % W].u : 100 + i;
            static_cast<int32_t*>(fillp[1])[i] = edge ? boundary[i % W].s : -2 * int32_t(i);
        }
    }
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpci.queueFamilyIndex = qi;
    VkCommandPool pool; CHECK(vkCreateCommandPool(dev, &cpci, nullptr, &pool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cmd; CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHECK(vkBeginCommandBuffer(cmd, &cbi));
    VkImageMemoryBarrier ib[3]{};
    for (int k = 0; k < 3; k++) {
        ib[k].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER; ib[k].image = img[k];
        ib[k].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; ib[k].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        ib[k].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        ib[k].srcQueueFamilyIndex = ib[k].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ib[k].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (load_mode && k > 0) {   // filled by a copy first
            ib[k].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; ib[k].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        }
    }
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                         nullptr, 3, ib);
    if (load_mode) {
        for (int k = 1; k < 3; k++) {
            VkBufferImageCopy bic{};
            bic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; bic.imageExtent = {W, W, 1};
            vkCmdCopyBufferToImage(cmd, fill[k - 1], img[k], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bic);
            ib[k].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; ib[k].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            ib[k].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            ib[k].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        }
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, 2, ib + 1);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
    uint32_t pc[36] = {};
    vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 144, pc);
    vkCmdDispatch(cmd, 1, 1, 1);
    for (int k = 0; k < 3; k++) {
        ib[k].oldLayout = VK_IMAGE_LAYOUT_GENERAL; ib[k].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        ib[k].srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT; ib[k].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    }
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 3, ib);
    for (int k = 0; k < 3; k++) {
        VkBufferImageCopy bic{};
        bic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; bic.imageExtent = {W, W, 1};
        vkCmdCopyImageToBuffer(cmd, img[k], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb[k], 1, &bic);
    }
    CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    CHECK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
    CHECK(vkQueueWaitIdle(q));

    uint32_t bad[3] = {};
    for (uint32_t y = 0; y < W; y++)
        for (uint32_t x = 0; x < W; x++) {
            uint32_t i = y * W + x;
            const uint8_t* c = static_cast<const uint8_t*>(rbp[0]) + 4 * i;
            int er = int(std::lround(x / 7.0 * 255)), eg = int(std::lround(y / 7.0 * 255));
            if (std::abs(c[0] - er) > 1 || std::abs(c[1] - eg) > 1 || std::abs(c[2] - 128) > 1 || c[3] != 255) bad[0]++;
            const bool edge = packed_load && y == W - 1;
            if (static_cast<const uint32_t*>(rbp[1])[i] != (edge ? boundary[x].u : 100 + i)) bad[1]++;
            // load: the uint texel (100 + 8y + x) plus the sint one (-2 (8y + x)), both read with image_load
            const int32_t want_s = edge ? boundary[x].sum : load_mode ? 100 - int32_t(i) : -int32_t(i);
            if (static_cast<const int32_t*>(rbp[2])[i] != want_s) bad[2]++;
        }
    const uint8_t* c = static_cast<const uint8_t*>(rbp[0]) + 4 * (3 * W + 5);
    std::printf("rgba8 unorm (float view): %u bad, texel(5,3) = %u %u %u %u\n", bad[0], c[0], c[1], c[2], c[3]);
    std::printf("r32 uint (uint view): %u bad, texel(5,3) = %u\n", bad[1], static_cast<const uint32_t*>(rbp[1])[3 * W + 5]);
    std::printf("r32 sint (sint view): %u bad, texel(5,3) = %d\n", bad[2], static_cast<const int32_t*>(rbp[2])[3 * W + 5]);
    return (bad[0] || bad[1] || bad[2]) ? 1 : 0;
}
