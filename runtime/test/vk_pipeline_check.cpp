// Driver-level check of the translated corpus: vk_pipeline_check <spirv dir> [pipeline_cache.bin]
// Reads <spirv dir>/index.json (written by tools/rdna2spirv.py), creates a VkShaderModule for every entry and, for
// compute shaders, a real compute pipeline with the translator's resource contract (set 0: sampled images 0, 3, 5,
// 6; samplers 1; descriptor hash table 2; storage images 4, 7 and 8..19; push constants 144 bytes). Device features
// the corpus needs are enabled when the GPU has them. Reports every failure; optionally saves the pipeline cache
// so a first boot doesn't compile these again.
#include <vulkan/vulkan.h>
#include <windows.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

static PFN_vkGetInstanceProcAddr gipa;

// Minimal reader for the index: {"name": {"stage": "cs", ...}, ...}. Only names and stages are needed.
static std::map<std::string, std::string> read_index(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::string s((std::istreambuf_iterator<char>(f)), {});
    std::map<std::string, std::string> out;
    size_t i = 0;
    int depth = 0;
    std::string key;
    while (i < s.size()) {
        char c = s[i];
        if (c == '{') { depth++; i++; continue; }
        if (c == '}') { depth--; i++; continue; }
        if (c == '"') {
            size_t j = s.find('"', i + 1);
            std::string tok = s.substr(i + 1, j - i - 1);
            i = j + 1;
            if (depth == 1) { key = tok; continue; }
            if (depth == 2 && tok == "stage") {
                size_t a = s.find('"', i), b = s.find('"', a + 1);
                out[key] = s.substr(a + 1, b - a - 1);
                i = b + 1;
            }
            continue;
        }
        i++;
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: vk_pipeline_check <spirv dir> [cache.bin]\n"); return 2; }
    std::string dir = argv[1];
    auto index = read_index(dir + "/index.json");
    std::fprintf(stderr, "%zu shaders in index\n", index.size());
    HMODULE lib = LoadLibraryA("vulkan-1.dll");
    gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(lib, "vkGetInstanceProcAddr"));
#define LOADI(name) auto name = reinterpret_cast<PFN_##name>(gipa(inst, #name))
    VkInstance inst = VK_NULL_HANDLE;
    LOADI(vkCreateInstance);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, nullptr, &inst)) return 1;
    LOADI(vkEnumeratePhysicalDevices); LOADI(vkGetPhysicalDeviceProperties); LOADI(vkCreateDevice);
    LOADI(vkGetDeviceProcAddr); LOADI(vkEnumerateDeviceExtensionProperties); LOADI(vkGetPhysicalDeviceFeatures2);
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(inst, &n, pds.data());
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    for (auto d : pds) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d, &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU || !pd) pd = d;
    }
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd, &props);
    std::fprintf(stderr, "device: %s\n", props.deviceName);
    uint32_t en = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, nullptr);
    std::vector<VkExtensionProperties> exts(en);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, exts.data());
    auto has = [&](const char* e) { for (auto& x : exts) if (!std::strcmp(x.extensionName, e)) return true; return false; };
    std::vector<const char*> enable;
    // Query what the GPU supports, then enable exactly that.
    VkPhysicalDeviceMeshShaderFeaturesEXT mesh{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR bary{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
    VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT lock{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT};
    VkPhysicalDeviceMaintenance8FeaturesKHR m8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    void* chain = nullptr;
    auto link = [&](void* s, VkBaseOutStructure* b, const char* ext) {
        if (ext && !has(ext)) return;
        if (ext) enable.push_back(ext);
        b->pNext = static_cast<VkBaseOutStructure*>(chain); chain = s;
    };
    link(&f12, reinterpret_cast<VkBaseOutStructure*>(&f12), nullptr);
    link(&f13, reinterpret_cast<VkBaseOutStructure*>(&f13), nullptr);
    link(&mesh, reinterpret_cast<VkBaseOutStructure*>(&mesh), VK_EXT_MESH_SHADER_EXTENSION_NAME);
    link(&bary, reinterpret_cast<VkBaseOutStructure*>(&bary), VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
    link(&lock, reinterpret_cast<VkBaseOutStructure*>(&lock), VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME);
    link(&m8, reinterpret_cast<VkBaseOutStructure*>(&m8), VK_KHR_MAINTENANCE_8_EXTENSION_NAME);
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = chain;
    vkGetPhysicalDeviceFeatures2(pd, &f2);            // fills every linked struct with what's supported
    mesh.multiviewMeshShader = mesh.primitiveFragmentShadingRateMeshShader = mesh.meshShaderQueries = VK_FALSE;
    mesh.taskShader = VK_FALSE;
    float prio = 1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = 0; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci; dci.pNext = &f2;
    dci.enabledExtensionCount = uint32_t(enable.size()); dci.ppEnabledExtensionNames = enable.data();
    VkDevice dev;
    if (VkResult r = vkCreateDevice(pd, &dci, nullptr, &dev)) { std::fprintf(stderr, "vkCreateDevice %d\n", r); return 1; }
#define LOADD(name) auto name = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(dev, #name))
    LOADD(vkCreateShaderModule); LOADD(vkDestroyShaderModule); LOADD(vkCreateDescriptorSetLayout);
    LOADD(vkCreatePipelineLayout); LOADD(vkCreateComputePipelines); LOADD(vkDestroyPipeline);
    LOADD(vkCreatePipelineCache); LOADD(vkGetPipelineCacheData);

    // Resource contract (docs/design.md): set 0, runtime-sized arrays as large bindless arrays.
    std::vector<VkDescriptorSetLayoutBinding> b;
    auto add = [&](uint32_t bind, VkDescriptorType t, uint32_t count) {
        b.push_back({bind, t, count, VK_SHADER_STAGE_ALL, nullptr});
    };
    add(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4096); add(1, VK_DESCRIPTOR_TYPE_SAMPLER, 1024);
    add(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1); add(3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024);
    add(4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024); add(5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 256);
    add(6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 256); add(7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256);
    for (uint32_t i = 8; i <= 19; i++) add(i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256);
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = uint32_t(b.size()); dli.pBindings = b.data();
    VkDescriptorSetLayout dsl;
    if (VkResult r = vkCreateDescriptorSetLayout(dev, &dli, nullptr, &dsl)) { std::fprintf(stderr, "layout %d\n", r); return 1; }
    VkPushConstantRange pcr{VK_SHADER_STAGE_ALL, 0, 144};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pl; vkCreatePipelineLayout(dev, &pli, nullptr, &pl);
    VkPipelineCacheCreateInfo pci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    VkPipelineCache cache; vkCreatePipelineCache(dev, &pci, nullptr, &cache);

    std::map<std::string, uint32_t> modules_ok, modules_bad, pipes_ok, pipes_bad;
    uint32_t done = 0;
    for (auto& [name, stage] : index) {
        std::ifstream f(dir + "/" + name + ".spv", std::ios::binary);
        std::vector<char> code((std::istreambuf_iterator<char>(f)), {});
        if (code.empty()) { modules_bad[stage]++; continue; }
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smi.codeSize = code.size(); smi.pCode = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule sm;
        if (vkCreateShaderModule(dev, &smi, nullptr, &sm)) { modules_bad[stage]++; std::printf("MODULE FAIL %s\n", name.c_str()); continue; }
        modules_ok[stage]++;
        if (stage == "cs") {
            VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = sm; cpi.stage.pName = "main";
            cpi.layout = pl;
            VkPipeline p;
            if (VkResult r = vkCreateComputePipelines(dev, cache, 1, &cpi, nullptr, &p)) {
                pipes_bad[stage]++; std::printf("PIPELINE FAIL %s (%d)\n", name.c_str(), r);
            } else {
                pipes_ok[stage]++; vkDestroyPipeline(dev, p, nullptr);
            }
        }
        vkDestroyShaderModule(dev, sm, nullptr);
        if (++done % 2000 == 0) std::fprintf(stderr, "  %u/%zu\n", done, index.size());
    }
    for (auto& [st, c] : modules_ok) std::printf("%s: %u modules ok, %u failed", st.c_str(), c, modules_bad[st]);
    std::printf("\n");
    for (auto& [st, c] : pipes_ok) std::printf("%s pipelines: %u ok, %u failed\n", st.c_str(), c, pipes_bad[st]);
    if (argc > 2) {
        size_t sz = 0;
        vkGetPipelineCacheData(dev, cache, &sz, nullptr);
        std::vector<char> data(sz);
        vkGetPipelineCacheData(dev, cache, &sz, data.data());
        std::ofstream(argv[2], std::ios::binary).write(data.data(), std::streamsize(sz));
        std::printf("pipeline cache: %zu bytes -> %s\n", sz, argv[2]);
    }
    return 0;
}
