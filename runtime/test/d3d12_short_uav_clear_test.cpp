// Short typed buffer UAV clears through the public COM API. The whole readback must match both the host and
// literal byte patterns, including all untouched neighbours. CTest selects WARP; manual runs can select a native
// adapter with LEONIDA_HOST_ADAPTER. Synthetic inputs only.
#include "vk_test_device.h"
#include "d3d12_copy_compare.h"
#include "host_wine.h"
#include <array>
#include <bit>

namespace {
struct Format {
    DXGI_FORMAT format;
    UINT bytes;
    bool floating;
    std::array<UINT, 4> values;
    std::array<unsigned char, 2> expected;
};

bool run(copycmp::Run& run, const Format& format, UINT first, UINT count,
         const std::vector<D3D12_RECT>& rects, std::vector<unsigned char>& actual) {
    constexpr UINT bytes = 256;
    copycmp::Ptr<ID3D12Resource> target, upload, readback;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (!run.texture(desc, D3D12_RESOURCE_STATE_COPY_DEST, target) ||
        !run.buffer(D3D12_HEAP_TYPE_UPLOAD, bytes, D3D12_RESOURCE_STATE_GENERIC_READ, upload) ||
        !run.buffer(D3D12_HEAP_TYPE_READBACK, bytes, D3D12_RESOURCE_STATE_COPY_DEST, readback)) return false;
    void* mapped = nullptr;
    const D3D12_RANGE none{};
    if (FAILED(upload->Map(0, &none, &mapped))) return false;
    std::memset(mapped, 0xa5, bytes);
    upload->Unmap(0, nullptr);
    copycmp::Ptr<ID3D12DescriptorHeap> cpu, gpu;
    D3D12_DESCRIPTOR_HEAP_DESC heap{};
    heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap.NumDescriptors = 1;
    if (FAILED(run.device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&cpu.p)))) return false;
    heap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(run.device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&gpu.p)))) return false;
    D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
    view.Format = format.format;
    view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    view.Buffer.FirstElement = first;
    view.Buffer.NumElements = count;
    run.device->CreateUnorderedAccessView(target.p, nullptr, &view, cpu->GetCPUDescriptorHandleForHeapStart());
    run.device->CreateUnorderedAccessView(target.p, nullptr, &view, gpu->GetCPUDescriptorHandleForHeapStart());
    run.begin();
    run.list->CopyBufferRegion(target.p, 0, upload.p, 0, bytes);
    run.barrier(target.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12DescriptorHeap* heaps[] = {gpu.p};
    run.list->SetDescriptorHeaps(1, heaps);
    if (format.floating) {
        std::array<float, 4> value{};
        for (unsigned i = 0; i < 4; ++i) value[i] = std::bit_cast<float>(format.values[i]);
        run.list->ClearUnorderedAccessViewFloat(gpu->GetGPUDescriptorHandleForHeapStart(),
            cpu->GetCPUDescriptorHandleForHeapStart(), target.p, value.data(), UINT(rects.size()), rects.data());
    } else {
        run.list->ClearUnorderedAccessViewUint(gpu->GetGPUDescriptorHandleForHeapStart(),
            cpu->GetCPUDescriptorHandleForHeapStart(), target.p, format.values.data(), UINT(rects.size()), rects.data());
    }
    run.barrier(target.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    run.list->CopyBufferRegion(readback.p, 0, target.p, 0, bytes);
    const auto error = run.submit();
    if (!error.empty()) { std::fprintf(stderr, "%s\n", error.c_str()); return false; }
    const D3D12_RANGE range{0, bytes};
    if (FAILED(readback->Map(0, &range, &mapped))) return false;
    actual.assign(static_cast<unsigned char*>(mapped), static_cast<unsigned char*>(mapped) + bytes);
    readback->Unmap(0, &none);
    return true;
}

std::vector<unsigned char> expected(const Format& format, UINT first, UINT count,
                                    const std::vector<D3D12_RECT>& rects) {
    std::vector<unsigned char> out(256, 0xa5);
    const auto write = [&](LONG left, LONG right) {
        for (UINT i = UINT(std::max(0L, left)); i < std::min(count, UINT(std::max(0L, right))); ++i)
            for (UINT byte = 0; byte < format.bytes; ++byte)
                out[(first + i) * format.bytes + byte] = format.expected[byte];
    };
    if (rects.empty()) write(0, LONG(count));
    else for (const auto& rect : rects) if (rect.top <= 0 && rect.bottom > 0) write(rect.left, rect.right);
    return out;
}
} // namespace

int main() {
    if (hostwine::host_is_wine()) return 77;
    vktest::Device vk;
    vk.request_buffer_device_address = true;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    if (!vk.buffer_device_address) return 77;
    copycmp::Ptr<ID3D12Device> ours, host;
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.buffer_device_address = vk.buffer_device_address;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;
    std::string adapter;
    if (!copycmp::create_host(host, adapter)) return 1;
    std::printf("short UAV clear reference: %s\n", adapter.c_str());
    copycmp::Run a, b;
    if (!a.init(ours.p, true) || !b.init(host.p, false)) return 1;
    const UINT half = std::bit_cast<UINT>(0.5f);
    const Format formats[] = {
        {DXGI_FORMAT_R8_UINT, 1, false, {0x7b, 0, 0, 0}, {0x7b, 0}},
        {DXGI_FORMAT_R16_UINT, 2, false, {0xbeef, 0, 0, 0}, {0xef, 0xbe}},
        {DXGI_FORMAT_R8G8_UINT, 2, false, {0x7b, 0x35, 0, 0}, {0x7b, 0x35}},
        {DXGI_FORMAT_R8_UNORM, 1, true, {half, 0, 0, 0}, {0x80, 0}},
        {DXGI_FORMAT_R16_UNORM, 2, true, {half, 0, 0, 0}, {0x00, 0x80}},
    };
    unsigned cases = 0;
    const auto check = [&](const Format& format, UINT first, UINT count, const std::vector<D3D12_RECT>& rects) {
        std::vector<unsigned char> actual, reference;
        const auto literal = expected(format, first, count, rects);
        if (!run(b, format, first, count, rects, reference) || reference != literal ||
            !run(a, format, first, count, rects, actual) || actual != reference) {
            std::fprintf(stderr, "format %u first %u count %u rects %zu failed\n",
                         unsigned(format.format), first, count, rects.size());
            return false;
        }
        ++cases;
        return true;
    };
    for (const auto& format : formats) {
        for (UINT first = 0; first < 4; ++first)
            for (UINT count = 1; count <= 8; ++count) if (!check(format, first, count, {})) return 1;
        if (!check(format, 256 / format.bytes - 1, 1, {})) return 1;
        if (!check(format, 1, 8, {{-3, 0, 1, 1}, {0, 0, 2, 1}, {3, 0, 4, 1}, {7, 0, 20, 1}, {0, 1, 8, 2}})) return 1;
        if (!check(format, 1, 8, {{-3, 0, 0, 1}, {0, 1, 8, 2}, {8, 0, 10, 1}})) return 1;
    }
    std::printf("short buffer UAV clears: %u cases matched host and literal bytes, including untouched neighbours\n", cases);
    return vk.validation_errors() ? 1 : 0;
}
