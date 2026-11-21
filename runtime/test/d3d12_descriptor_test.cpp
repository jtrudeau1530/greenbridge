// Descriptors through the Xbox-ABI proxy over the Vulkan device: CopyDescriptors and CopyDescriptorsSimple.
// A descriptor is the bytes of its heap slot, so the copies are checked on those bytes.
// No shaders or game data. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slots 9/10: 8-byte handle in RAX

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() {
        if (p) p->Release();
    }
    T* operator->() const { return p; }
};

void* slot(void* object, unsigned index) { return (*reinterpret_cast<void***>(object))[index]; }

}  // namespace

int main() {
    vktest::Device d;
    EXPECT(d.create());
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);  // the proxy now owns `inner`

        // Two host-side heaps. Slot i of the source holds 32 bytes of (0x10 * (i + 1) + byte index).
        const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        constexpr UINT source_slots = 4, destination_slots = 8;
        Ref<ID3D12DescriptorHeap> source_heap, destination_heap;
        D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, source_slots, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&source_heap.p))));
        heap_desc.NumDescriptors = destination_slots;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&destination_heap.p))));
        auto* const source = reinterpret_cast<std::uint8_t*>(
            reinterpret_cast<XboxHeapHandleFn>(slot(source_heap.p, 9))(source_heap.p));
        auto* const destination = reinterpret_cast<std::uint8_t*>(
            reinterpret_cast<XboxHeapHandleFn>(slot(destination_heap.p, 9))(destination_heap.p));
        EXPECT(source && destination && step >= 16);
        for (UINT i = 0; i < source_slots; ++i)
            for (UINT b = 0; b < step; ++b) source[i * step + b] = std::uint8_t(0x10 * (i + 1) + b);
        const auto from = [&](UINT i) { return D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(source + i * step)}; };
        const auto to = [&](UINT i) { return D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(destination + i * step)}; };
        // Destination slot `at` holds source slot `want`, or is still zero (want < 0).
        const auto holds = [&](UINT at, int want) {
            const std::uint8_t* got = destination + at * step;
            if (want < 0) return std::all_of(got, got + step, [](std::uint8_t byte) { return byte == 0; });
            return std::memcmp(got, source + UINT(want) * step, step) == 0;
        };
        const auto clear = [&] { std::memset(destination, 0, std::size_t(destination_slots) * step); };
        const auto rtv = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;

        // CopyDescriptorsSimple: one run of descriptors.
        device->CopyDescriptorsSimple(2, to(1), from(2), rtv);
        EXPECT(holds(0, -1) && holds(1, 2) && holds(2, 3) && holds(3, -1));
        device->CopyDescriptorsSimple(0, to(5), from(0), rtv);
        EXPECT(holds(5, -1));

        // CopyDescriptors walks both range lists together: destination ranges {3, 4} and {0}, source ranges {0}
        // and {1, 2}, so 3 <- 0, 4 <- 1, 0 <- 2.
        clear();
        {
            const D3D12_CPU_DESCRIPTOR_HANDLE destinations[] = {to(3), to(0)}, sources[] = {from(0), from(1)};
            const UINT destination_sizes[] = {2, 1}, source_sizes[] = {1, 2};
            device->CopyDescriptors(2, destinations, destination_sizes, 2, sources, source_sizes, rtv);
            EXPECT(holds(3, 0) && holds(4, 1) && holds(0, 2));
            EXPECT(holds(1, -1) && holds(2, -1) && holds(5, -1) && holds(6, -1) && holds(7, -1));
        }
        // Null size arrays: every range is one descriptor.
        clear();
        {
            const D3D12_CPU_DESCRIPTOR_HANDLE destinations[] = {to(6), to(2)}, sources[] = {from(3), from(1)};
            device->CopyDescriptors(2, destinations, nullptr, 2, sources, nullptr, rtv);
            EXPECT(holds(6, 3) && holds(2, 1) && holds(7, -1) && holds(3, -1));
        }
        // One side sized, the other not, and a destination longer than the source: the copy stops with the source.
        clear();
        {
            const D3D12_CPU_DESCRIPTOR_HANDLE destinations[] = {to(1)}, sources[] = {from(0), from(3)};
            const UINT destination_sizes[] = {5};
            device->CopyDescriptors(1, destinations, destination_sizes, 2, sources, nullptr, rtv);
            EXPECT(holds(1, 0) && holds(2, 3) && holds(3, -1) && holds(4, -1) && holds(5, -1) && holds(0, -1));
        }
        // Nothing to copy from or to.
        clear();
        {
            const D3D12_CPU_DESCRIPTOR_HANDLE destinations[] = {to(0)}, sources[] = {from(0)};
            device->CopyDescriptors(1, destinations, nullptr, 0, sources, nullptr, rtv);
            device->CopyDescriptors(1, nullptr, nullptr, 1, sources, nullptr, rtv);
            device->CopyDescriptors(1, destinations, nullptr, 1, nullptr, nullptr, rtv);
            EXPECT(holds(0, -1));
        }
    }
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("descriptors: CopyDescriptorsSimple and CopyDescriptors copied the slots D3D12 names\n");
    return 0;
}
