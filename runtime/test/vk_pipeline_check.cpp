// Driver-level check of the translated corpus: vk_pipeline_check <spirv dir> [pipeline_cache.bin]
// Reads <spirv dir>/index.json (written by tools/rdna2spirv.py), creates a VkShaderModule for every entry and, for
// compute shaders, a real compute pipeline with the translator's resource contract (set 0: sampled images 0, 3, 5,
// 6; samplers 1; descriptor hash table 2; storage images 4, 7 and 8..19; push constants 144 bytes). Device features
// the corpus needs are enabled when the GPU has them. Reports every failure; optionally saves the pipeline cache
// so a first boot doesn't compile these again.
//   [cache.bin]   loaded first if it exists (a stopped run resumes: cached pipelines are instant), saved every
//                 10 minutes and at the end
//   --jobs N      compile on N threads (default: half the hardware threads)
//   --start I --count N   check only index entries [I, I+N) (sorted by name)
//   --slow SEC    print compute pipelines slower than SEC (default 30); compiles past 60 s are named while running
#include "vulkan_loader.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

static PFN_vkGetInstanceProcAddr gipa;

// Minimal reader for the index: {"name": {"stage": "cs", "subgroup_size": 64, ...}, ...}.
struct ShaderMeta {
    std::string stage;
    uint32_t subgroup_size = 0;
};

static std::map<std::string, ShaderMeta> read_index(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::string s((std::istreambuf_iterator<char>(f)), {});
    std::map<std::string, ShaderMeta> out;
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
                out[key].stage = s.substr(a + 1, b - a - 1);
                i = b + 1;
            } else if (depth == 2 && tok == "subgroup_size") {
                size_t a = s.find(':', i);
                if (a != std::string::npos) {
                    a++;
                    while (a < s.size() && (s[a] == ' ' || s[a] == '\t')) a++;
                    uint32_t val = 0;
                    while (a < s.size() && s[a] >= '0' && s[a] <= '9') {
                        val = val * 10 + uint32_t(s[a] - '0');
                        a++;
                    }
                    out[key].subgroup_size = val;
                    i = a;
                }
            }
            continue;
        }
        i++;
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: vk_pipeline_check <spirv dir> [cache.bin] [--jobs N] [--start I] [--count N] [--slow SEC]\n");
        return 2;
    }
    std::setvbuf(stdout, nullptr, _IOLBF, 1 << 16);   // failures reach a redirected log as they happen
    std::string dir = argv[1], cache_path;
    uint32_t jobs = std::max(1u, std::thread::hardware_concurrency() / 2);
    size_t start = 0, count = 0;
    double slow = 30.0;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        auto val = [&] { return i + 1 < argc ? std::string(argv[++i]) : std::string("0"); };
        if (a == "--jobs") jobs = std::max(1u, uint32_t(std::stoul(val())));
        else if (a == "--start") start = std::stoull(val());
        else if (a == "--count") count = std::stoull(val());
        else if (a == "--slow") slow = std::stod(val());
        else cache_path = a;
    }
    auto index = read_index(dir + "/index.json");
    std::fprintf(stderr, "%zu shaders in index\n", index.size());
    gipa = load_vulkan();
    if (!gipa) return 1;
#define LOADI(name) auto name = reinterpret_cast<PFN_##name>(gipa(inst, #name))
    VkInstance inst = VK_NULL_HANDLE;
    LOADI(vkCreateInstance);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, nullptr, &inst)) return 1;
    LOADI(vkEnumeratePhysicalDevices); LOADI(vkGetPhysicalDeviceProperties); LOADI(vkCreateDevice);
    LOADI(vkGetPhysicalDeviceProperties2);
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
    VkPhysicalDeviceSubgroupSizeControlProperties sgProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &sgProps;
    vkGetPhysicalDeviceProperties2(pd, &props2);
    std::fprintf(stderr, "device: %s (subgroup %u..%u)\n", props.deviceName, sgProps.minSubgroupSize, sgProps.maxSubgroupSize);
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
    std::vector<char> seed;
    if (!cache_path.empty()) {  // resume: pipelines already in the cache compile instantly
        std::ifstream cf(cache_path, std::ios::binary);
        seed.assign(std::istreambuf_iterator<char>(cf), {});
        if (!seed.empty()) std::fprintf(stderr, "seeded the pipeline cache with %zu bytes from %s\n", seed.size(), cache_path.c_str());
    }
    pci.initialDataSize = seed.size(); pci.pInitialData = seed.empty() ? nullptr : seed.data();
    VkPipelineCache cache;
    if (vkCreatePipelineCache(dev, &pci, nullptr, &cache)) {  // data from another driver: start empty
        pci.initialDataSize = 0; pci.pInitialData = nullptr;
        vkCreatePipelineCache(dev, &pci, nullptr, &cache);
    }
    auto save_cache = [&](bool final_save) {
        if (cache_path.empty()) return;
        size_t sz = 0;
        vkGetPipelineCacheData(dev, cache, &sz, nullptr);
        std::vector<char> data(sz);
        vkGetPipelineCacheData(dev, cache, &sz, data.data());
        const std::string tmp = cache_path + ".partial";
        std::ofstream(tmp, std::ios::binary).write(data.data(), std::streamsize(sz));
        std::remove(cache_path.c_str());
        std::rename(tmp.c_str(), cache_path.c_str());
        if (final_save) std::printf("pipeline cache: %zu bytes -> %s\n", sz, cache_path.c_str());
    };

    // Work list: index order (sorted by name), entries [start, start + count).
    using Item = std::pair<const std::string*, const ShaderMeta*>;
    std::vector<Item> all;
    for (auto& [name, meta] : index) all.push_back({&name, &meta});
    const size_t lo = std::min(start, all.size()), hi = std::min(all.size(), count ? lo + count : all.size());
    const std::vector<Item> work(all.begin() + lo, all.begin() + hi);
    std::fprintf(stderr, "checking %zu shaders [%zu, %zu) on %u threads\n", work.size(), lo, hi, jobs);

    std::mutex mu;  // guards the counters, stdout and the in-flight table
    std::map<std::string, uint32_t> modules_ok, modules_bad, pipes_ok, pipes_bad;
    uint32_t cs_w32_ok = 0, cs_w32_bad = 0, cs_w64_ok = 0, cs_w64_bad = 0, cs_done = 0;
    double cs_seconds = 0, cs_max = 0;
    std::string cs_max_name;
    std::atomic<size_t> next{0}, done{0};
    std::map<std::thread::id, std::pair<std::string, std::chrono::steady_clock::time_point>> inflight;
    const auto t_start = std::chrono::steady_clock::now();
    auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count(); };

    auto worker = [&] {
        for (size_t k; (k = next++) < work.size();) {
            const std::string& name = *work[k].first;
            const ShaderMeta& meta = *work[k].second;
            const std::string& stage = meta.stage;
            std::ifstream f(dir + "/" + name + ".spv", std::ios::binary);
            std::vector<char> code((std::istreambuf_iterator<char>(f)), {});
            VkShaderModule sm = VK_NULL_HANDLE;
            bool mod_ok = false;
            if (!code.empty()) {
                VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
                smi.codeSize = code.size(); smi.pCode = reinterpret_cast<const uint32_t*>(code.data());
                mod_ok = vkCreateShaderModule(dev, &smi, nullptr, &sm) == VK_SUCCESS;
            }
            {
                std::lock_guard<std::mutex> g(mu);
                if (mod_ok) modules_ok[stage]++;
                else { modules_bad[stage]++; std::printf("MODULE FAIL %s\n", name.c_str()); }
            }
            if (mod_ok && stage == "cs") {
                VkPipelineShaderStageRequiredSubgroupSizeCreateInfo reqSg{
                    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
                reqSg.requiredSubgroupSize = meta.subgroup_size;
                VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
                cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
                cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = sm; cpi.stage.pName = "main";
                if (f13.subgroupSizeControl &&
                    meta.subgroup_size >= sgProps.minSubgroupSize &&
                    meta.subgroup_size <= sgProps.maxSubgroupSize &&
                    (sgProps.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT)) {
                    cpi.stage.pNext = &reqSg;
                }
                cpi.layout = pl;
                const auto t0 = std::chrono::steady_clock::now();
                { std::lock_guard<std::mutex> g(mu); inflight[std::this_thread::get_id()] = {name, t0}; }
                VkPipeline p;
                VkResult r = vkCreateComputePipelines(dev, cache, 1, &cpi, nullptr, &p);
                const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (r == VK_SUCCESS) vkDestroyPipeline(dev, p, nullptr);
                std::lock_guard<std::mutex> g(mu);
                inflight.erase(std::this_thread::get_id());
                cs_seconds += sec;
                if (sec > cs_max) { cs_max = sec; cs_max_name = name; }
                if (sec > slow) std::printf("SLOW CS %s (w%u, %.1fs)\n", name.c_str(), meta.subgroup_size, sec);
                if (r) {
                    pipes_bad[stage]++;
                    if (meta.subgroup_size == 64) cs_w64_bad++; else cs_w32_bad++;
                    std::printf("PIPELINE FAIL %s (w%u, %d)\n", name.c_str(), meta.subgroup_size, r);
                } else {
                    pipes_ok[stage]++;
                    if (meta.subgroup_size == 64) cs_w64_ok++; else cs_w32_ok++;
                }
                if (++cs_done % 250 == 0) std::fprintf(stderr, "  cs %u (%.0fs elapsed)\n", cs_done, elapsed());
            }
            if (sm) vkDestroyShaderModule(dev, sm, nullptr);
            if (++done % 1000 == 0) {
                std::lock_guard<std::mutex> g(mu);
                std::fprintf(stderr, "  %zu/%zu (%.0fs)\n", size_t(done), work.size(), elapsed());
            }
        }
    };
    std::vector<std::thread> pool;
    for (uint32_t t = 0; t < jobs; t++) pool.emplace_back(worker);
    // Watchdog: names every compile running past 60 s (about once a minute) and saves the cache every 10 minutes.
    std::thread watch([&] {
        auto last_save = std::chrono::steady_clock::now();
        while (done < work.size()) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            const auto now = std::chrono::steady_clock::now();
            {
                std::lock_guard<std::mutex> g(mu);
                for (auto& [tid, job] : inflight) {
                    const double s = std::chrono::duration<double>(now - job.second).count();
                    if (s > 60 && int(s) % 60 < 5) std::fprintf(stderr, "  still compiling %s after %.0fs\n", job.first.c_str(), s);
                }
            }
            if (now - last_save > std::chrono::minutes(10)) { save_cache(false); last_save = now; }
        }
    });
    for (auto& t : pool) t.join();
    watch.join();

    for (auto& [st, c] : modules_ok) std::printf("%s: %u modules ok, %u failed\n", st.c_str(), c, modules_bad[st]);
    for (auto& [st, c] : modules_bad) if (!modules_ok.count(st)) std::printf("%s: 0 modules ok, %u failed\n", st.c_str(), c);
    if (cs_done > 0) {
        std::printf("cs pipelines: %u ok, %u failed\n", pipes_ok["cs"], pipes_bad["cs"]);
        std::printf("  cs wave32: %u ok, %u failed; wave64: %u ok, %u failed\n",
                    cs_w32_ok, cs_w32_bad, cs_w64_ok, cs_w64_bad);
        std::printf("  cs compile time: %.0fs total, %.2fs mean, slowest %s (%.1fs)\n",
                    cs_seconds, cs_seconds / cs_done, cs_max_name.c_str(), cs_max);
    }
    std::printf("wall time: %.0fs on %u threads\n", elapsed(), jobs);
    save_cache(true);
    return 0;
}
