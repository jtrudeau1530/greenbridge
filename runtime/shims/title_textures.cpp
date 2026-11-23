#include "title_textures.h"

#include <windows.h>

#include <d3d12.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

namespace rt::title_textures {
namespace {
struct Linear {
    std::vector<std::uint8_t> bytes;
    std::uint32_t row = 0, slice = 0;
};
struct Texture {
    ID3D12Resource* resource = nullptr;
    D3D12_RESOURCE_DESC desc{};
    bool dirty = true;
    bool uploaded = false;  // it was left in COMMON by an upload
};
struct Buffer {
    ID3D12Resource* resource = nullptr;
    std::uint8_t* mapped = nullptr;
    std::uint64_t size = 0;
};

std::mutex g_mutex;
std::map<std::uint64_t, std::map<std::uint32_t, Linear>> g_linear;
std::map<std::uint64_t, Texture> g_textures;
std::map<std::pair<std::uint64_t, std::uint64_t>, Buffer> g_buffers;
bool g_any_dirty = false;

bool readable(std::uint64_t p, std::uint64_t n) {
    MEMORY_BASIC_INFORMATION mbi{};
    return p > 0x10000 && VirtualQuery(reinterpret_cast<const void*>(p), &mbi, sizeof(mbi)) && mbi.State == MEM_COMMIT &&
           !(mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) &&
           reinterpret_cast<std::uint64_t>(mbi.BaseAddress) + mbi.RegionSize >= p + n;
}

bool log_first(std::atomic<int>& n, int limit) { return n.fetch_add(1, std::memory_order_relaxed) < limit; }

D3D12_RESOURCE_DESC clean(const D3D12_RESOURCE_DESC& in) {
    D3D12_RESOURCE_DESC d;
    std::memset(&d, 0, sizeof(d));
    d.Dimension = in.Dimension;
    d.Width = in.Width;
    d.Height = in.Height;
    d.DepthOrArraySize = in.DepthOrArraySize;
    d.MipLevels = in.MipLevels;
    d.Format = in.Format;
    d.SampleDesc.Count = in.SampleDesc.Count ? in.SampleDesc.Count : 1;
    return d;
}

ID3D12Resource* buffer(ID3D12Device* device, std::uint64_t size, D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES heap{type};
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = size;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* r = nullptr;
    device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                    IID_PPV_ARGS(&r));
    return r;
}

// One list, recorded and waited for per texture: the number of CPU-made textures is small.
void upload(ID3D12Device* device, Texture& t, const std::map<std::uint32_t, Linear>& data) {
    static ID3D12CommandQueue* queue = nullptr;
    static ID3D12CommandAllocator* allocator = nullptr;
    static ID3D12GraphicsCommandList* list = nullptr;
    static ID3D12Fence* fence = nullptr;
    static std::uint64_t value = 0;
    if (!queue) {
        D3D12_COMMAND_QUEUE_DESC q{};
        device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue));
        device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
        device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, IID_PPV_ARGS(&list));
        device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
        if (list) list->Close();
    }
    if (!queue || !allocator || !list || !fence) return;
    allocator->Reset();
    list->Reset(allocator, nullptr);
    const UINT layers = t.desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? 1 : t.desc.DepthOrArraySize;
    const UINT count = t.desc.MipLevels * layers;
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> prints(count);
    std::vector<UINT> rows(count);
    std::vector<UINT64> sizes(count);
    UINT64 total = 0;
    device->GetCopyableFootprints(&t.desc, 0, count, 0, prints.data(), rows.data(), sizes.data(), &total);
    ID3D12Resource* staging = total && total <= (512ull << 20) ? buffer(device, total, D3D12_HEAP_TYPE_UPLOAD) : nullptr;
    std::uint8_t* mapped = nullptr;
    if (!staging || FAILED(staging->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) {
        list->Close();
        return;
    }
    const auto transition = [&](D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = t.resource;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = from;
        b.Transition.StateAfter = to;
        list->ResourceBarrier(1, &b);
    };
    if (t.uploaded) transition(D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    for (const auto& [sub, in] : data) {
        if (sub >= count) continue;
        const auto& p = prints[sub];
        for (UINT z = 0; z < p.Footprint.Depth; ++z) {
            for (UINT y = 0; y < rows[sub]; ++y) {
                const std::uint64_t from = std::uint64_t(z) * in.slice + std::uint64_t(y) * in.row;
                const std::uint64_t n = std::min<std::uint64_t>(sizes[sub], in.row);
                if (from + n > in.bytes.size()) continue;
                std::memcpy(mapped + p.Offset + (std::uint64_t(z) * rows[sub] + y) * p.Footprint.RowPitch,
                            in.bytes.data() + from, static_cast<std::size_t>(n));
            }
        }
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource = t.resource;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.SubresourceIndex = sub;
        src.pResource = staging;
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint = p;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    transition(D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    const HRESULT hr = list->Close();
    if (SUCCEEDED(hr)) {
        ID3D12CommandList* lists[] = {list};
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(fence, ++value);
        fence->SetEventOnCompletion(value, nullptr);
        t.uploaded = true;
    }
    staging->Release();
    t.dirty = false;
    static std::atomic<int> n{0};
    if (log_first(n, 20))
        std::fprintf(stderr, "[title] uploaded %llux%u format %d (%zu subresources) -> 0x%08lx\n",
                     static_cast<unsigned long long>(t.desc.Width), t.desc.Height, int(t.desc.Format), data.size(),
                     static_cast<unsigned long>(hr));
}
}  // namespace

void note_linear_upload(std::uint64_t base, std::uint32_t subresource, const void* data, std::uint32_t row_pitch,
                        std::uint32_t slice_pitch, const std::uint32_t* rd) {
    D3D12_RESOURCE_DESC d{};
    std::memcpy(&d, rd, sizeof(d));
    const std::uint32_t mip = d.MipLevels ? subresource % d.MipLevels : 0;
    const std::uint64_t depth = d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D
                                    ? std::max<std::uint64_t>(1, d.DepthOrArraySize >> mip)
                                    : 1;
    const std::uint64_t n = depth * slice_pitch;
    if (!data || !n || n > (256ull << 20) || !readable(reinterpret_cast<std::uint64_t>(data), n)) return;
    const std::lock_guard lock(g_mutex);
    Linear& l = g_linear[base][subresource];
    l.bytes.assign(static_cast<const std::uint8_t*>(data), static_cast<const std::uint8_t*>(data) + n);
    l.row = row_pitch;
    l.slice = slice_pitch;
    if (const auto it = g_textures.find(base); it != g_textures.end()) {
        it->second.dirty = true;
        g_any_dirty = true;
    }
}

bool create_view(ID3D12Device* device, std::uint64_t resource_desc, std::uint64_t view_desc, std::uint64_t destination) {
    if (!readable(resource_desc, sizeof(D3D12_RESOURCE_DESC)) || !readable(view_desc, 0x30) || !readable(destination, 32))
        return false;
    D3D12_RESOURCE_DESC in{};
    std::memcpy(&in, reinterpret_cast<const void*>(resource_desc), sizeof(in));
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    std::memcpy(&srv, reinterpret_cast<const void*>(view_desc), sizeof(srv));
    std::uint64_t va = 0;
    std::memcpy(&va, reinterpret_cast<const void*>(view_desc + 0x28), sizeof(va));
    const D3D12_CPU_DESCRIPTOR_HANDLE dest{static_cast<SIZE_T>(destination)};
    const std::lock_guard lock(g_mutex);
    static std::atomic<int> n{0};
    if (in.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
        const std::uint64_t size = in.Width;
        if (!size || size > (4ull << 20) || !readable(va, size)) {
            if (log_first(n, 12))
                std::fprintf(stderr, "[title] buffer view of %llu bytes: null\n", static_cast<unsigned long long>(size));
            device->CreateShaderResourceView(nullptr, &srv, dest);
            return true;
        }
        Buffer& b = g_buffers[{va, size}];
        if (!b.resource) {
            b.resource = buffer(device, size, D3D12_HEAP_TYPE_UPLOAD);
            if (b.resource) b.resource->Map(0, nullptr, reinterpret_cast<void**>(&b.mapped));
            b.size = size;
        }
        device->CreateShaderResourceView(b.resource, &srv, dest);
        return true;
    }
    Texture& t = g_textures[va];
    const D3D12_RESOURCE_DESC d = clean(in);
    if (!t.resource || std::memcmp(&t.desc, &d, sizeof(d)) != 0) {
        t = Texture{};  // the old resource may be in flight: kept
        t.desc = d;
        D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
        const HRESULT hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST,
                                                           nullptr, IID_PPV_ARGS(&t.resource));
        if (FAILED(hr) && log_first(n, 12))
            std::fprintf(stderr, "[title] texture %llux%u format %d -> 0x%08lx\n", static_cast<unsigned long long>(d.Width),
                         d.Height, int(d.Format), static_cast<unsigned long>(hr));
        g_any_dirty = true;
    }
    device->CreateShaderResourceView(t.resource, &srv, dest);
    return true;
}

void refresh(ID3D12Device* device) {
    const std::lock_guard lock(g_mutex);
    for (auto& [key, b] : g_buffers)
        if (b.mapped && readable(key.first, b.size))
            std::memcpy(b.mapped, reinterpret_cast<const void*>(key.first), static_cast<std::size_t>(b.size));
    if (!g_any_dirty) return;
    g_any_dirty = false;
    for (auto& [va, t] : g_textures) {
        if (!t.dirty || !t.resource) continue;
        if (const auto it = g_linear.find(va); it != g_linear.end()) upload(device, t, it->second);
    }
}
}  // namespace rt::title_textures
