// ID3D12Device3::OpenExistingHeapFromAddress, OpenExistingHeapFromFileMapping and ID3D12Device5::CreateLifetimeTracker
// on arguments the host D3D12 runtime refuses (null address, a file mapping that is not a handle, a null owner), all
// E_INVALIDARG on the RTX 4080 SUPER, Intel UHD 770 and WARP. Checked on the Vulkan-backed device and, when present,
// on the host devices too. With VK_EXT_external_memory_host, OpenExistingHeapFromAddress on real host memory is
// compared case by case with the host devices (open_cases: RTX 4090 and WARP agree), and a heap is opened on a
// committed range; without it, the device reports EXISTING_HEAPS unsupported and answers an accepted range E_NOTIMPL.
// No shaders or game data.
// https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device3-openexistingheapfromaddress
#include "vk_test_device.h"
#include "host_adapter.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>
#include <string>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "%s line %d: %s\n", who, __LINE__, #condition); return 1; } \
} while (0)
#define EXPECT_HR(expr, want) do { \
    const HRESULT got_ = (expr); \
    if (got_ != (want)) { \
        std::fprintf(stderr, "%s line %d: %s = 0x%08lX, expected 0x%08lX\n", who, __LINE__, #expr, \
                     static_cast<unsigned long>(got_), static_cast<unsigned long>(want)); \
        return 1; \
    } \
} while (0)

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

int check(ID3D12Device* device, const char* who) {
    Ref<ID3D12Device3> device3;
    EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device3.p))));
    Ref<ID3D12Heap> heap;
    EXPECT_HR(device3->OpenExistingHeapFromAddress(nullptr, IID_PPV_ARGS(&heap.p)), E_INVALIDARG);
    EXPECT(heap.p == nullptr);
    EXPECT_HR(device3->OpenExistingHeapFromFileMapping(nullptr, IID_PPV_ARGS(&heap.p)), E_INVALIDARG);
    EXPECT_HR(device3->OpenExistingHeapFromFileMapping(reinterpret_cast<HANDLE>(0x1234), IID_PPV_ARGS(&heap.p)),
              E_INVALIDARG);
    EXPECT(heap.p == nullptr);
    Ref<ID3D12Device5> device5;
    if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device5.p)))) {
        Ref<ID3D12LifetimeTracker> tracker;
        EXPECT_HR(device5->CreateLifetimeTracker(nullptr, IID_PPV_ARGS(&tracker.p)), E_INVALIDARG);
        EXPECT(tracker.p == nullptr);
    }
    std::printf("%s: existing-heap and lifetime-tracker refusals match the host runtime\n", who);
    return 0;
}

// The calls the host accepts or refuses on real host memory, one result per case, for the comparison in main().
struct Case {
    const char* name;
    HRESULT hr = S_OK;
    D3D12_HEAP_DESC desc{};
};

std::vector<Case> open_cases(ID3D12Device* device) {
    std::vector<Case> out;
    Ref<ID3D12Device3> device3;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&device3.p)))) return out;
    auto run = [&](const char* name, const void* address) {
        Case c{name};
        Ref<ID3D12Heap> heap;
        c.hr = device3->OpenExistingHeapFromAddress(address, IID_PPV_ARGS(&heap.p));
        if (SUCCEEDED(c.hr) && heap.p) c.desc = heap->GetDesc();
        out.push_back(c);
    };
    constexpr SIZE_T k64 = 0x10000;
    // A whole 64 KiB committed read-write allocation, as the title's XMemVirtualAlloc gives it.
    void* whole = VirtualAlloc(nullptr, k64, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    run("64 KiB committed, at its base", whole);
    run("64 KiB committed, 4 KiB in", static_cast<char*>(whole) + 0x1000);
    // Two committed 64 KiB in one 192 KiB allocation whose last 64 KiB is only reserved.
    char* partial = static_cast<char*>(VirtualAlloc(nullptr, 3 * k64, MEM_RESERVE, PAGE_READWRITE));
    VirtualAlloc(partial, 2 * k64, MEM_COMMIT, PAGE_READWRITE);
    run("192 KiB reserved, first 128 KiB committed", partial);
    void* reserved = VirtualAlloc(nullptr, k64, MEM_RESERVE, PAGE_READWRITE);
    run("64 KiB reserved only", reserved);
    void* readonly = VirtualAlloc(nullptr, k64, MEM_RESERVE | MEM_COMMIT, PAGE_READONLY);
    run("64 KiB committed read-only", readonly);
    // The console title's graphics regions keep PAGE_WRITECOMBINE (or PAGE_NOCACHE) once XMemVirtualAlloc drops
    // the PAGE_GRAPHICS_* bits.
    void* combined = VirtualAlloc(nullptr, k64, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE | PAGE_WRITECOMBINE);
    run("64 KiB committed read-write write-combined", combined);
    void* uncached = VirtualAlloc(nullptr, k64, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE | PAGE_NOCACHE);
    run("64 KiB committed read-write uncached", uncached);
    void* big = VirtualAlloc(nullptr, 32 * k64, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    run("2 MiB committed", big);
    // The heap keeps no hold on the memory the test frees here; every heap above is already released.
    for (void* p : {whole, static_cast<void*>(partial), reserved, readonly, combined, uncached, big}) VirtualFree(p, 0, MEM_RELEASE);
    return out;
}

// Each case answers as the host does: the same HRESULT, and for an opened heap the same description.
int compare_cases(const std::vector<Case>& ours, const std::vector<Case>& host, const char* who) {
    int failures = 0;
    for (size_t i = 0; i < ours.size() && i < host.size(); ++i) {
        const D3D12_HEAP_DESC& a = ours[i].desc;
        const D3D12_HEAP_DESC& b = host[i].desc;
        const bool same = ours[i].hr == host[i].hr && a.SizeInBytes == b.SizeInBytes && a.Alignment == b.Alignment &&
                          a.Properties.Type == b.Properties.Type &&
                          a.Properties.CPUPageProperty == b.Properties.CPUPageProperty &&
                          a.Properties.MemoryPoolPreference == b.Properties.MemoryPoolPreference && a.Flags == b.Flags;
        if (!same) {
            std::fprintf(stderr, "vulkan vs %s: \"%s\" differs (hr 0x%08lX vs 0x%08lX)\n", who, ours[i].name,
                         static_cast<unsigned long>(ours[i].hr), static_cast<unsigned long>(host[i].hr));
            ++failures;
        }
    }
    if (!failures) std::printf("vulkan vs %s: %zu OpenExistingHeapFromAddress cases match\n", who, ours.size());
    return failures;
}

// The heap is the caller's memory: CheckFeatureSupport says so, and the heap stays valid while it is held.
// Without the extension the feature is unsupported and an accepted range is E_NOTIMPL.
int check_import(ID3D12Device* device, bool supported) {
    const char* who = "vulkan import";
    D3D12_FEATURE_DATA_EXISTING_HEAPS feature{};
    EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_EXISTING_HEAPS, &feature, sizeof(feature))));
    EXPECT(bool(feature.Supported) == supported);
    Ref<ID3D12Device3> device3;
    EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device3.p))));
    constexpr SIZE_T size = 4 * 0x10000;
    void* memory = VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    EXPECT(memory != nullptr);
    {
        Ref<ID3D12Heap> heap;
        const HRESULT hr = device3->OpenExistingHeapFromAddress(memory, IID_PPV_ARGS(&heap.p));
        if (!supported) {
            EXPECT_HR(hr, E_NOTIMPL);
            EXPECT(heap.p == nullptr);
        } else {
            EXPECT_HR(hr, S_OK);
            EXPECT(heap.p != nullptr);
            EXPECT(heap->GetDesc().SizeInBytes == size);
            // The CPU still owns the pages: a write through them is what the GPU will read.
            static_cast<unsigned char*>(memory)[size - 1] = 0x5a;
            // A placed resource in it is refused until the placed path addresses existing heaps.
            D3D12_RESOURCE_DESC buffer{};
            buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            buffer.Width = 0x10000;
            buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = 1;
            buffer.SampleDesc.Count = 1;
            buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            Ref<ID3D12Resource> placed;
            EXPECT_HR(device->CreatePlacedResource(heap.p, 0, &buffer, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                   IID_PPV_ARGS(&placed.p)),
                      E_NOTIMPL);
        }
    }
    EXPECT(VirtualFree(memory, 0, MEM_RELEASE));
    std::printf("%s: existing heap %s\n", who, supported ? "opened on the caller's memory" : "unsupported, as reported");
    return 0;
}

void print_cases(const std::vector<Case>& cases, const char* who) {
    for (const Case& c : cases)
        std::printf("%s: %-44s hr=0x%08lX size=%llu align=%llu type=%d cpu=%d pool=%d flags=0x%x\n", who, c.name,
                    static_cast<unsigned long>(c.hr), static_cast<unsigned long long>(c.desc.SizeInBytes),
                    static_cast<unsigned long long>(c.desc.Alignment), int(c.desc.Properties.Type),
                    int(c.desc.Properties.CPUPageProperty), int(c.desc.Properties.MemoryPoolPreference),
                    unsigned(c.desc.Flags));
}
}  // namespace

int main() {
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    int failures = 0;
    {
        Ref<ID3D12Device> ours;
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.host_pointer_alignment = vk.host_pointer_alignment;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;
        failures += check(ours.p, "vulkan");
        const std::vector<Case> vulkan_cases = open_cases(ours.p);
        print_cases(vulkan_cases, "vulkan");
        failures += check_import(ours.p, vk.host_pointer_alignment != 0);
        Ref<IDXGIFactory4> factory;
        if (!hostwine::host_is_wine() && SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
            for (const bool warp : {false, true}) {
                Ref<IDXGIAdapter> adapter;
                Ref<ID3D12Device> host;
                std::string adapter_name;
                if (!hostadapter::reference(factory.p, vk, warp, &adapter.p, adapter_name)) continue;
                if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) continue;
                failures += check(host.p, ("host " + adapter_name).c_str());
                const std::string who = "host " + adapter_name;
                const std::vector<Case> host_cases = open_cases(host.p);
                print_cases(host_cases, who.c_str());
                if (vk.host_pointer_alignment) failures += compare_cases(vulkan_cases, host_cases, who.c_str());
            }
        }
    }
    if (vk.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", vk.validation_errors());
        return 1;
    }
    return failures ? 1 : 0;
}
