// Tier 1 regression coverage and public-device capability checks.
#include "conservative_raster_fixture.h"
using namespace conservative_test;
namespace {
bool rejected_states(Fixture& f, bool enabled) {
    auto desc = description(f, true, 1, false);
    copycmp::Ptr<ID3D12PipelineState> pso;
    if (!enabled) {
        CHECK(f.run.device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso.p)) == E_INVALIDARG);
        desc.RasterizerState.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
        CHECK(SUCCEEDED(f.run.device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso.p))));
        return true;
    }
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_WIREFRAME;
    CHECK(f.run.device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso.p)) == E_INVALIDARG);
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
    CHECK(f.run.device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso.p)) == E_INVALIDARG);
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.RasterizerState.ForcedSampleCount = 4;
    CHECK(f.run.device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso.p)) == E_NOTIMPL);
    desc.RasterizerState.ForcedSampleCount = 0;
    desc.RasterizerState.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE(7);
    CHECK(f.run.device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso.p)) == E_INVALIDARG);
    return true;
}

bool owned_device_caps() {
    copycmp::Ptr<ID3D12Device> device;
    CHECK(SUCCEEDED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device.p))));
    const auto* info = rt::d3d12::vulkan_device_info(device.p);
    CHECK(info);
    const auto enumerate = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
        info->get_instance_proc(info->instance, "vkEnumerateDeviceExtensionProperties"));
    UINT count = 0;
    CHECK(enumerate(info->physical, nullptr, &count, nullptr) == VK_SUCCESS);
    std::vector<VkExtensionProperties> extensions(count);
    CHECK(enumerate(info->physical, nullptr, &count, extensions.data()) == VK_SUCCESS);
    const bool available = std::any_of(extensions.begin(), extensions.end(), [](const auto& e) {
        return std::strcmp(e.extensionName, VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME) == 0;
    });
    CHECK(info->conservative_rasterization == available);
    VkPhysicalDeviceConservativeRasterizationPropertiesEXT conservative{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONSERVATIVE_RASTERIZATION_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    if (available) properties.pNext = &conservative;
    reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(info->get_instance_proc(info->instance,
        "vkGetPhysicalDeviceProperties2"))(info->physical, &properties);
    const auto expected = leonida::conservative_raster::tier(available,
        properties.properties.limits.subPixelPrecisionBits, conservative.primitiveOverestimationSize,
        conservative.degenerateTrianglesRasterized == VK_TRUE);
    D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
    CHECK(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options))));
    CHECK(unsigned(options.ConservativeRasterizationTier) == expected);
    return true;
}
} // namespace

int main(int argc, char** argv) {
    const bool compare_warp_msaa4 = argc == 2 && std::strcmp(argv[1], "--warp-msaa4") == 0;
    if (argc != 1 && !compare_warp_msaa4) return 2;
    if (hostwine::host_is_wine()) return 77;
    vktest::Device vk;
    vk.request_buffer_device_address = true;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    VkPhysicalDeviceConservativeRasterizationPropertiesEXT conservative_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONSERVATIVE_RASTERIZATION_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    if (vk.conservative_rasterization) properties.pNext = &conservative_properties;
    reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(vk.instance_proc(vk.instance, "vkGetPhysicalDeviceProperties2"))(
        vk.physical, &properties);
    if (!vk.buffer_device_address || properties.properties.limits.maxPushConstantsSize < 144) {
        std::puts("SKIP: current graphics path needs bufferDeviceAddress and 144 push-constant bytes"); return 77;
    }
    const auto vs_program = d3d12test::fssm_program({0x43525653, 0xbf810000});
    const auto ps_program = d3d12test::fssm_program({0x43525053, 0xbf810000});
    const auto vs_bytes = d3d12test::shader_container(vs_program), ps_bytes = d3d12test::shader_container(ps_program);
    const auto directory = std::filesystem::temp_directory_path() / ("leonida_conservative_" + std::to_string(GetCurrentProcessId()));
    std::filesystem::create_directories(directory);
    std::filesystem::copy_file(LEONIDA_CONSERVATIVE_VS, directory / "vs.spv", std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(LEONIDA_CONSERVATIVE_PS, directory / "ps.spv", std::filesystem::copy_options::overwrite_existing);
    {
        std::ofstream index(directory / "index.json", std::ios::binary);
        index << "{\"vs\":{\"stage\":\"vs\",\"fssm_sha256\":\"" << leonida::sha256_hex(vs_program)
              << "\",\"bindings\":[],\"push_constant_bytes\":0,\"input_signature\":[{\"semantic\":\"POSITION\",\"semantic_index\":0,\"register\":0,\"mask\":3,\"rw_mask\":3,\"system_value\":0,\"component_type\":\"float\"}]},"
                 "\"ps\":{\"stage\":\"ps\",\"fssm_sha256\":\"" << leonida::sha256_hex(ps_program)
              << "\",\"bindings\":[],\"push_constant_bytes\":0,\"outputs\":[0]}}";
    }
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc; info.instance = vk.instance; info.physical = vk.physical;
    info.device = vk.device; info.queue = vk.queue; info.queue_family = vk.family;
    info.buffer_device_address = vk.buffer_device_address;
    info.conservative_rasterization = vk.conservative_rasterization;
    info.shader_dir = directory.string();
    copycmp::Ptr<ID3D12Device> ours, host;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;
    D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
    if (FAILED(ours->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options)))) return 1;
    const auto expected_tier = leonida::conservative_raster::tier(vk.conservative_rasterization,
        properties.properties.limits.subPixelPrecisionBits, conservative_properties.primitiveOverestimationSize,
        conservative_properties.degenerateTrianglesRasterized == VK_TRUE);
    if (unsigned(options.ConservativeRasterizationTier) != expected_tier) return 1;
    if (options.ConservativeRasterizationTier == D3D12_CONSERVATIVE_RASTERIZATION_TIER_NOT_SUPPORTED) {
        std::puts("SKIP: Vulkan device does not meet conservative raster Tier 1 requirements"); return 77;
    }
    if (options.ConservativeRasterizationTier > D3D12_CONSERVATIVE_RASTERIZATION_TIER_2) return 1;
    if (!owned_device_caps()) return 1;
    std::string adapter;
    if (!copycmp::create_host(host, adapter)) return 1;
    copycmp::Ptr<IDXGIFactory4> factory;
    copycmp::Ptr<IDXGIAdapter1> host_adapter;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
        FAILED(factory->EnumAdapterByLuid(host->GetAdapterLuid(), IID_PPV_ARGS(&host_adapter.p)))) return 1;
    DXGI_ADAPTER_DESC1 adapter_desc{};
    if (FAILED(host_adapter->GetDesc1(&adapter_desc))) return 1;
    const bool software_reference = (adapter_desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
    if (FAILED(host->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options)))) return 1;
    if (options.ConservativeRasterizationTier == D3D12_CONSERVATIVE_RASTERIZATION_TIER_NOT_SUPPORTED) return 77;
    copycmp::Ptr<ID3DBlob> vs, ps, error;
    const HMODULE compiler = LoadLibraryExA("d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto compile = compiler ? reinterpret_cast<pD3DCompile>(reinterpret_cast<void*>(
        GetProcAddress(compiler, "D3DCompile"))) : nullptr;
    if (!compile) { std::puts("SKIP: d3dcompiler_47 is needed for the native reference shaders"); return 77; }
    const char* hlsl = "float4 vs(float2 p:POSITION):SV_Position{return float4(p,0.5,1);} float4 ps():SV_Target{return float4(1,0,0,1);}";
    if (FAILED(compile(hlsl, std::strlen(hlsl), nullptr, nullptr, nullptr, "vs", "vs_5_0", 0, 0, &vs.p, &error.p)) ||
        FAILED(compile(hlsl, std::strlen(hlsl), nullptr, nullptr, nullptr, "ps", "ps_5_0", 0, 0, &ps.p, nullptr))) return 1;
    Fixture a, b;
    if (!setup(a, ours.p, true, {vs_bytes.data(), vs_bytes.size()}, {ps_bytes.data(), ps_bytes.size()}) ||
        !setup(b, host.p, false, {vs->GetBufferPointer(), vs->GetBufferSize()}, {ps->GetBufferPointer(), ps->GetBufferSize()})) return 1;
    if (!rejected_states(a, true)) return 1;
    {
        info.conservative_rasterization = false;
        copycmp::Ptr<ID3D12Device> disabled;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&disabled.p)))) return 1;
        if (FAILED(disabled->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options))) ||
            options.ConservativeRasterizationTier != D3D12_CONSERVATIVE_RASTERIZATION_TIER_NOT_SUPPORTED) return 1;
        Fixture f;
        if (!setup(f, disabled.p, true, {vs_bytes.data(), vs_bytes.size()}, {ps_bytes.data(), ps_bytes.size()}) ||
            !rejected_states(f, false)) return 1;
    }
    bool msaa4 = true;
    for (auto* device : {ours.p, host.p}) {
        D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS levels{DXGI_FORMAT_R8G8B8A8_UNORM, 4};
        if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &levels, sizeof(levels))) ||
            !levels.NumQualityLevels) msaa4 = false;
    }
    const Triangle scenes[] = {
        {{{3.0625,3.0625},{3.3125,3.0625},{3.0625,3.3125}}},
        {{{2.125,5.125},{25.125,6.125},{2.125,5.25}}},
        {{{4.125,4.125},{24.125,4.125},{4.125,24.125}}},
        {{{-0.25,2.125},{5.125,2.125},{2.125,15.125}}},
        {{{4.125,24.125},{24.125,4.125},{4.125,4.125}}},
        {{{3.0625,3.0625},{3.3125,3.0625},{3.0625,3.0625}}}, // Tier 1 permits degenerate triangles to be culled
    };
    unsigned comparisons = 0, host_comparisons = 0;
    for (const UINT samples : {1u, 4u}) {
      if (samples == 4 && !msaa4) continue;
      // WARP's 4x path was observed to cover outside the Tier 1 uncertainty bound for the subpixel fixture.
      // Keep Vulkan's 4x geometric oracle strict; use 1x for the ordinary software-reference comparisons.
      // --warp-msaa4 explicitly reproduces that reference discrepancy without weakening either oracle.
      const bool compare_host = samples == 1 || !software_reference || compare_warp_msaa4;
      for (const bool variant : {false, true})
        for (const auto scissor : {D3D12_RECT{0,0,32,32}, D3D12_RECT{4,4,20,20}})
            for (const auto& triangle : scenes) {
                Mask normal_a{}, normal_b{}, conservative_a{}, conservative_b{};
                if (!render(a, triangle, false, samples, variant, scissor, normal_a) ||
                    (compare_host && !render(b, triangle, false, samples, variant, scissor, normal_b)) ||
                    !render(a, triangle, true, samples, variant, scissor, conservative_a) ||
                    (compare_host && !render(b, triangle, true, samples, variant, scissor, conservative_b))) return 1;
                const bool ours_ok = coverage(triangle, scissor, conservative_a);
                const bool host_ok = !compare_host || coverage(triangle, scissor, conservative_b);
                if (!ours_ok || !host_ok) {
                    std::fprintf(stderr, "scene %zu samples %u variant %d scissor-left %ld: Vulkan=%d host=%d\n",
                                 std::size_t(&triangle - scenes), samples, int(variant), scissor.left, int(ours_ok), int(host_ok));
                    return 1;
                }
                for (unsigned i = 0; i < normal_a.size(); ++i)
                    if ((normal_a[i] && !conservative_a[i]) || (normal_b[i] && !conservative_b[i])) return 1;
                if (&triangle == &scenes[0] && samples == 1 && scissor.left == 0) {
                    if (std::any_of(normal_a.begin(), normal_a.end(), [](bool x) { return x; }) ||
                        std::any_of(normal_b.begin(), normal_b.end(), [](bool x) { return x; }) ||
                        !std::any_of(conservative_a.begin(), conservative_a.end(), [](bool x) { return x; }) ||
                        (compare_host && !std::any_of(conservative_b.begin(), conservative_b.end(), [](bool x) { return x; }))) return 1;
                }
                ++comparisons;
                if (compare_host) ++host_comparisons;
            }
    }
    for (auto* f : {&a, &b}) {
        Mask front{}, back{}, all{};
        const D3D12_RECT full{0,0,32,32};
        if (!render(*f, scenes[2], true, 1, false, full, all) ||
            !render(*f, scenes[2], true, 1, false, full, front, D3D12_CULL_MODE_FRONT) ||
            !render(*f, scenes[2], true, 1, false, full, back, D3D12_CULL_MODE_BACK)) return 1;
        for (unsigned i = 0; i < all.size(); ++i) if ((front[i] && back[i]) || ((front[i] || back[i]) != all[i])) return 1;
    }
    std::printf("conservative raster: %u Vulkan and %u %s scenario pairs; geometric bounds, culling, scissor, variants, rejections; Vulkan 4x MSAA %s\n",
                comparisons, host_comparisons, adapter.c_str(), msaa4 ? "checked" : "unavailable");
    if (software_reference && !compare_warp_msaa4)
        std::puts("WARP comparison is 1x; --warp-msaa4 separately checks its observed 4x coverage discrepancy");
    return vk.validation_errors() ? 1 : 0;
}
