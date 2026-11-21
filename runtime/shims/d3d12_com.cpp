#include "d3d12_com.h"

#include "d3d12_stubs.h"  // generated
#include "d3d12_com_object.h"
#include "d3d12_trace.h"
#include "../src/gfx10_descriptors.h"

#include <array>
#include <atomic>
#include <bit>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace rt::d3d12 {

void stub(const char* method) {
    // Generated API stubs pass a bare method name; explanatory diagnostics are not separate calls.
    if (!std::strchr(method, '(')) d3d12_trace::record(method, "unsupported");
    static std::mutex mutex;
    static std::set<std::string> seen;
    std::lock_guard lock(mutex);
    if (seen.insert(method).second) std::fprintf(stderr, "[rt] d3d12: unimplemented %s\n", method);
}

namespace {

// Reports a deliberate simplification once (unlike stub, which reports a method not implemented).
void note_once(const char* what) {
    static std::mutex mutex;
    static std::set<std::string> seen;
    std::lock_guard lock(mutex);
    if (seen.insert(what).second) std::fprintf(stderr, "[rt] d3d12: %s\n", what);
}

const char* name(leonida::ReplayResult result) {
    switch (result) {
    case leonida::ReplayResult::success: return "success";
    case leonida::ReplayResult::list_open: return "a list is not closed";
    case leonida::ReplayResult::unresolved_resource: return "unknown resource";
    case leonida::ReplayResult::invalid_subresource: return "invalid subresource";
    case leonida::ReplayResult::invalid_clear: return "invalid clear";
    case leonida::ReplayResult::invalid_copy: return "invalid copy or resolve";
    case leonida::ReplayResult::invalid_query: return "invalid query";
    case leonida::ReplayResult::unsupported: return "unsupported";
    }
    return "?";
}

class CommandAllocator final
    : public Child<CommandAllocator, ID3D12CommandAllocatorStubs, ID3D12Object, ID3D12DeviceChild, ID3D12Pageable,
                    ID3D12CommandAllocator> {
public:
    // {7d0e6b51-34a2-4f0c-9a61-0c5b2e8f4a10}
    static constexpr GUID kTag{0x7d0e6b51, 0x34a2, 0x4f0c, {0x9a, 0x61, 0x0c, 0x5b, 0x2e, 0x8f, 0x4a, 0x10}};
    explicit CommandAllocator(D3D12_COMMAND_LIST_TYPE type) : type_(type) {}
    // Lists own their recorded commands, so there is nothing to free. D3D12 forbids resetting an allocator whose
    // lists are still executing; that is the application's job, as on D3D12.
    HRESULT STDMETHODCALLTYPE Reset() override { return S_OK; }
    D3D12_COMMAND_LIST_TYPE type() const { return type_; }
    static CommandAllocator* from(IUnknown* object) { return Child::from<CommandAllocator>(object); }

private:
    D3D12_COMMAND_LIST_TYPE type_;
};

class GraphicsCommandList final
    : public Child<GraphicsCommandList, ID3D12GraphicsCommandList8Stubs, ID3D12Object, ID3D12DeviceChild,
                    ID3D12CommandList, ID3D12GraphicsCommandList, ID3D12GraphicsCommandList1,
                    ID3D12GraphicsCommandList2, ID3D12GraphicsCommandList3, ID3D12GraphicsCommandList4,
                    ID3D12GraphicsCommandList5, ID3D12GraphicsCommandList6, ID3D12GraphicsCommandList7,
                    ID3D12GraphicsCommandList8> {
public:
    // {2f8a7c33-91d4-4b6e-8e05-6d3f1a9b2c47}
    static constexpr GUID kTag{0x2f8a7c33, 0x91d4, 0x4b6e, {0x8e, 0x05, 0x6d, 0x3f, 0x1a, 0x9b, 0x2c, 0x47}};
    explicit GraphicsCommandList(D3D12_COMMAND_LIST_TYPE type) : type_(type) {}

    const leonida::CommandList& commands() const { return list_; }
    UINT front_stencil_ref() const { return front_stencil_ref_; }
    UINT back_stencil_ref() const { return back_stencil_ref_; }

    D3D12_COMMAND_LIST_TYPE STDMETHODCALLTYPE GetType() override {
        d3d12_trace::record("ID3D12CommandList::GetType");
        return type_;
    }
    D3D12_COMMAND_LIST_TYPE type() const { return type_; }

    // E_INVALIDARG when a call during recording was invalid, or a query was begun and not ended, as D3D12's Close
    // reports it.
    HRESULT STDMETHODCALLTYPE Close() override {
        constexpr auto method = "ID3D12GraphicsCommandList::Close";
        if (list_.queries_open()) invalid_ = true;
        if (!list_.close()) return d3d12_trace::result(method, E_FAIL);
        return d3d12_trace::result(method, invalid_ ? E_INVALIDARG : S_OK);
    }

    HRESULT STDMETHODCALLTYPE Reset(ID3D12CommandAllocator* allocator, ID3D12PipelineState* state) override {
        constexpr auto method = "ID3D12GraphicsCommandList::Reset";
        CommandAllocator* command_allocator = CommandAllocator::from(allocator);
        if (!command_allocator || command_allocator->type() != type_)
            return d3d12_trace::result(method, E_INVALIDARG);
        if (!list_.reset()) return d3d12_trace::result(method, E_FAIL);  // D3D12 resets only closed lists
        invalid_ = false;
        front_stencil_ref_ = back_stencil_ref_ = 0;  // Reset returns every list state to its default
        stencil_ref_set_ = false;
        if (state) stub("ID3D12GraphicsCommandList::Reset(initial pipeline state)");
        return d3d12_trace::result(method, S_OK);
    }

    void STDMETHODCALLTYPE ExecuteBundle(ID3D12GraphicsCommandList* command_list) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::ExecuteBundle");
        if (type_ != D3D12_COMMAND_LIST_TYPE_DIRECT) return check(false);
        GraphicsCommandList* bundle = from(command_list);
        if (!bundle || bundle->type_ != D3D12_COMMAND_LIST_TYPE_BUNDLE || bundle->invalid_ ||
            !bundle->list_.closed())
            return check(false);
        if (!list_.execute_bundle(bundle->list_)) return check(false);

        // Only the sticky state currently recorded by this COM layer is carried back from the bundle.
        if (bundle->stencil_ref_set_) {
            front_stencil_ref_ = bundle->front_stencil_ref_;
            back_stencil_ref_ = bundle->back_stencil_ref_;
            stencil_ref_set_ = true;
        }
    }

    void STDMETHODCALLTYPE ResourceBarrier(UINT count, const D3D12_RESOURCE_BARRIER* barriers) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::ResourceBarrier");
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) return check(false);
        for (UINT i = 0; i < count; ++i) {
            const auto& b = barriers[i];
            switch (b.Type) {
            case D3D12_RESOURCE_BARRIER_TYPE_TRANSITION:
                check(list_.transition({resource_id(b.Transition.pResource), b.Transition.Subresource,
                                        static_cast<std::uint32_t>(b.Transition.StateBefore),
                                        static_cast<std::uint32_t>(b.Transition.StateAfter)},
                                       static_cast<std::uint32_t>(b.Flags)));
                break;
            case D3D12_RESOURCE_BARRIER_TYPE_ALIASING:
                check(!b.Flags && list_.aliasing_barrier({resource_id(b.Aliasing.pResourceBefore),
                                                          resource_id(b.Aliasing.pResourceAfter)}));
                break;
            case D3D12_RESOURCE_BARRIER_TYPE_UAV:
                check(!b.Flags && list_.uav_barrier({resource_id(b.UAV.pResource)}));
                break;
            default:
                check(false);
            }
        }
    }

    void STDMETHODCALLTYPE Barrier(UINT32 groups, const D3D12_BARRIER_GROUP* group) override {
        d3d12_trace::record("ID3D12GraphicsCommandList7::Barrier");
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) return check(false);
        for (UINT32 g = 0; g < groups; ++g) {
            for (UINT32 i = 0; i < group[g].NumBarriers; ++i) {
                switch (group[g].Type) {
                case D3D12_BARRIER_TYPE_GLOBAL: {
                    const auto& b = group[g].pGlobalBarriers[i];
                    check(list_.barrier(leonida::GlobalBarrier{u(b.SyncBefore), u(b.SyncAfter), u(b.AccessBefore),
                                                               u(b.AccessAfter)}));
                    break;
                }
                case D3D12_BARRIER_TYPE_BUFFER: {
                    const auto& b = group[g].pBufferBarriers[i];
                    check(list_.barrier(leonida::BufferBarrier{u(b.SyncBefore), u(b.SyncAfter), u(b.AccessBefore),
                                                               u(b.AccessAfter), resource_id(b.pResource)}));
                    break;
                }
                case D3D12_BARRIER_TYPE_TEXTURE: {
                    const auto& b = group[g].pTextureBarriers[i];
                    const auto& r = b.Subresources;
                    check(list_.barrier(leonida::TextureBarrier{
                        u(b.SyncBefore), u(b.SyncAfter), u(b.AccessBefore), u(b.AccessAfter), u(b.LayoutBefore),
                        u(b.LayoutAfter), resource_id(b.pResource),
                        {r.IndexOrFirstMipLevel, r.NumMipLevels, r.FirstArraySlice, r.NumArraySlices, r.FirstPlane,
                         r.NumPlanes},
                        u(b.Flags)}));
                    break;
                }
                default:
                    check(false);
                }
            }
        }
    }

    void STDMETHODCALLTYPE CopyBufferRegion(ID3D12Resource* dst, UINT64 dst_offset, ID3D12Resource* src,
                                            UINT64 src_offset, UINT64 bytes) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::CopyBufferRegion");
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) return check(false);
        check(list_.copy_buffer({resource_id(dst), dst_offset, resource_id(src), src_offset, bytes}));
    }

    void STDMETHODCALLTYPE CopyTextureRegion(const D3D12_TEXTURE_COPY_LOCATION* dst, UINT x, UINT y, UINT z,
                                             const D3D12_TEXTURE_COPY_LOCATION* src, const D3D12_BOX* box) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::CopyTextureRegion");
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) return check(false);
        if (!dst || !src) return check(false);
        leonida::CopyTexture copy{location(*dst), x, y, z, location(*src), box != nullptr, {}};
        if (box) copy.box = {box->left, box->top, box->front, box->right, box->bottom, box->back};
        check(list_.copy_texture(copy));
    }

    void STDMETHODCALLTYPE CopyResource(ID3D12Resource* dst, ID3D12Resource* src) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::CopyResource");
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) return check(false);
        check(list_.copy_resource({resource_id(dst), resource_id(src)}));
    }

    // The format picks the member of a typeless family to resolve as; the textures' own format is used.
    void STDMETHODCALLTYPE ResolveSubresource(ID3D12Resource* dst, UINT dst_subresource, ID3D12Resource* src,
                                              UINT src_subresource, DXGI_FORMAT) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::ResolveSubresource");
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) return check(false);
        check(list_.resolve_subresource({resource_id(dst), dst_subresource, resource_id(src), src_subresource}));
    }

    void STDMETHODCALLTYPE ResolveSubresourceRegion(ID3D12Resource* dst, UINT dst_subresource, UINT x, UINT y,
                                                     ID3D12Resource* src, UINT src_subresource, D3D12_RECT* rect,
                                                     DXGI_FORMAT format, D3D12_RESOLVE_MODE mode) override {
        d3d12_trace::record("ID3D12GraphicsCommandList1::ResolveSubresourceRegion");
        if (type_ != D3D12_COMMAND_LIST_TYPE_DIRECT || mode != D3D12_RESOLVE_MODE_AVERAGE ||
            !color_resolve_format_matches(format, dst, src))
            return check(false);
        leonida::ResolveRegion region{x, y, std::nullopt};
        if (rect) region.src_rect = leonida::Rect{rect->left, rect->top, rect->right, rect->bottom};
        check(list_.resolve_subresource({resource_id(dst), dst_subresource, resource_id(src), src_subresource, region}));
    }

    // A null buffer turns predication off (docs/d3d12_vulkan.md section 4.7). Bundles may not set it.
    void STDMETHODCALLTYPE SetPredication(ID3D12Resource* buffer, UINT64 offset, D3D12_PREDICATION_OP op) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::SetPredication");
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) return check(false);
        check(list_.set_predication({resource_id(buffer), offset, u(op)}));
    }

    void STDMETHODCALLTYPE ClearRenderTargetView(D3D12_CPU_DESCRIPTOR_HANDLE view, const FLOAT color[4], UINT count,
                                                 const D3D12_RECT* rects) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::ClearRenderTargetView");
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) return check(false);
        if (!view.ptr || !color) return check(false);
        check(list_.clear_render_target(*reinterpret_cast<const leonida::TargetView*>(view.ptr),
                                        {color[0], color[1], color[2], color[3]}, to_rects(count, rects)));
    }

    void STDMETHODCALLTYPE ClearDepthStencilView(D3D12_CPU_DESCRIPTOR_HANDLE view, D3D12_CLEAR_FLAGS flags,
                                                 FLOAT depth, UINT8 stencil, UINT count,
                                                 const D3D12_RECT* rects) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::ClearDepthStencilView");
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) return check(false);
        if (!view.ptr) return check(false);
        check(list_.clear_depth_stencil(*reinterpret_cast<const leonida::TargetView*>(view.ptr), u(flags), depth,
                                        stencil, to_rects(count, rects)));
    }

    // A null region discards the whole resource. Not allowed in a bundle.
    void STDMETHODCALLTYPE DiscardResource(ID3D12Resource* resource, const D3D12_DISCARD_REGION* region) override {
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) return check(false);
        if (!region) return check(list_.discard({resource_id(resource), 0, 0, false, true}));
        if (region->NumRects && !region->pRects) return check(false);
        check(list_.discard({resource_id(resource), region->FirstSubresource, region->NumSubresources,
                             region->NumRects != 0}));
    }

    // GCL2. A null pModes means DEFAULT for every write. The address is resolved when the list is executed.
    // Allowed in a bundle (the API restrictions table of "Recording command lists and bundles").
    void STDMETHODCALLTYPE WriteBufferImmediate(UINT count, const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER* params,
                                                const D3D12_WRITEBUFFERIMMEDIATE_MODE* modes) override {
        if (count && !params) return check(false);
        for (UINT i = 0; i < count; ++i)
            check(list_.write_immediate({params[i].Dest, params[i].Value, modes ? u(modes[i]) : 0}));
    }

    // GCL1. Not allowed in a bundle, as no copy is.
    void STDMETHODCALLTYPE AtomicCopyBufferUINT(ID3D12Resource* dst, UINT64 dst_offset, ID3D12Resource* src,
                                                UINT64 src_offset, UINT count, ID3D12Resource* const* resources,
                                                const D3D12_SUBRESOURCE_RANGE_UINT64* ranges) override {
        atomic_copy(4, dst, dst_offset, src, src_offset, count, resources, ranges);
    }
    void STDMETHODCALLTYPE AtomicCopyBufferUINT64(ID3D12Resource* dst, UINT64 dst_offset, ID3D12Resource* src,
                                                  UINT64 src_offset, UINT count, ID3D12Resource* const* resources,
                                                  const D3D12_SUBRESOURCE_RANGE_UINT64* ranges) override {
        atomic_copy(8, dst, dst_offset, src, src_offset, count, resources, ranges);
    }

    // The CPU handle's slot holds the view as the GFX10 descriptor a shader would load (docs/d3d12_vulkan.md
    // section 3.4): a T# for a texture, a V# for a buffer. It is read now, because the slot may be rewritten before
    // the list runs. The GPU handle names the same view in a shader-visible heap and is not needed.
    void STDMETHODCALLTYPE ClearUnorderedAccessViewUint(D3D12_GPU_DESCRIPTOR_HANDLE, D3D12_CPU_DESCRIPTOR_HANDLE view,
                                                        ID3D12Resource* resource, const UINT values[4], UINT count,
                                                        const D3D12_RECT* rects) override {
        if (!values) return check(false);
        clear_unordered_access(view, resource, false, {values[0], values[1], values[2], values[3]}, count, rects);
    }

    void STDMETHODCALLTYPE ClearUnorderedAccessViewFloat(D3D12_GPU_DESCRIPTOR_HANDLE, D3D12_CPU_DESCRIPTOR_HANDLE view,
                                                         ID3D12Resource* resource, const FLOAT values[4], UINT count,
                                                         const D3D12_RECT* rects) override {
        if (!values) return check(false);
        clear_unordered_access(view, resource, true,
                               {std::bit_cast<std::uint32_t>(values[0]), std::bit_cast<std::uint32_t>(values[1]),
                                std::bit_cast<std::uint32_t>(values[2]), std::bit_cast<std::uint32_t>(values[3])},
                               count, rects);
    }

    // Queries (docs/d3d12_vulkan.md section 4.6). Heap types, index ranges and the destination buffer are checked
    // at ExecuteCommandLists, when the heap and the buffer are known.
    void STDMETHODCALLTYPE BeginQuery(ID3D12QueryHeap* heap, D3D12_QUERY_TYPE type, UINT index) override {
        check(list_.begin_query({query_heap_id(heap), u(type), index}));
    }

    void STDMETHODCALLTYPE EndQuery(ID3D12QueryHeap* heap, D3D12_QUERY_TYPE type, UINT index) override {
        check(list_.end_query({query_heap_id(heap), u(type), index}));
    }

    void STDMETHODCALLTYPE ResolveQueryData(ID3D12QueryHeap* heap, D3D12_QUERY_TYPE type, UINT first, UINT count,
                                            ID3D12Resource* dst, UINT64 offset) override {
        check(list_.resolve_queries({query_heap_id(heap), u(type), first, count, resource_id(dst), offset}));
    }

    // GCL8. Vulkan can set the two faces independently with vkCmdSetStencilReference(VK_STENCIL_FACE_FRONT_BIT /
    // BACK_BIT), but that state only matters when a pipeline uses it, and this layer has no pipeline or draw path
    // yet. Retain the values for that path and report that nothing consumed them, rather than emit a call whose
    // effect no later command can observe.
    void STDMETHODCALLTYPE OMSetFrontAndBackStencilRef(UINT front, UINT back) override {
        d3d12_trace::record("ID3D12GraphicsCommandList8::OMSetFrontAndBackStencilRef");
        if (list_.closed()) return check(false);
        front_stencil_ref_ = front;
        back_stencil_ref_ = back;
        stencil_ref_set_ = true;
        stub("ID3D12GraphicsCommandList8::OMSetFrontAndBackStencilRef (recorded; no pipeline path applies it)");
    }

    static GraphicsCommandList* from(IUnknown* object) { return Child::from<GraphicsCommandList>(object); }

private:
    template<class E>
    static std::uint32_t u(E value) {
        return static_cast<std::uint32_t>(value);
    }

    static leonida::CopyLocation location(const D3D12_TEXTURE_COPY_LOCATION& l) {
        if (l.Type == D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT) {
            const auto& f = l.PlacedFootprint;
            return {resource_id(l.pResource), true, 0, f.Offset, f.Footprint.Width, f.Footprint.Height,
                    f.Footprint.Depth, f.Footprint.RowPitch};
        }
        return {resource_id(l.pResource), false, l.SubresourceIndex, 0, 0, 0, 0, 0};
    }

    static std::vector<leonida::Rect> to_rects(UINT count, const D3D12_RECT* rects) {
        std::vector<leonida::Rect> out;
        for (UINT i = 0; rects && i < count; ++i)
            out.push_back({rects[i].left, rects[i].top, rects[i].right, rects[i].bottom});
        return out;
    }

    void check(bool ok) {
        if (!ok) invalid_ = true;
    }

    void atomic_copy(std::uint32_t bytes, ID3D12Resource* dst, UINT64 dst_offset, ID3D12Resource* src,
                     UINT64 src_offset, UINT count, ID3D12Resource* const* resources,
                     const D3D12_SUBRESOURCE_RANGE_UINT64* ranges) {
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE || (count && (!resources || !ranges))) return check(false);
        leonida::AtomicCopyBuffer copy{resource_id(dst), dst_offset, resource_id(src), src_offset, bytes, {}};
        for (UINT i = 0; i < count; ++i)
            copy.dependencies.push_back(
                {resource_id(resources[i]), ranges[i].Subresource, ranges[i].Range.Begin, ranges[i].Range.End});
        check(list_.atomic_copy(copy));
    }

    void clear_unordered_access(D3D12_CPU_DESCRIPTOR_HANDLE view, ID3D12Resource* resource, bool is_float,
                                const std::array<std::uint32_t, 4>& values, UINT count, const D3D12_RECT* rects) {
        leonida::UavView target{};
        if (!view.ptr || !resource || !unordered_access_view(view, resource, target)) return check(false);
        check(list_.clear_unordered_access(target, is_float, values, to_rects(count, rects)));
    }

    // The Vulkan format a descriptor's FORMAT and DST_SEL name: a UAV has no component mapping of its own, so its
    // DST_SEL is the one its format reads R, G, B, A with (gfx10::format_swizzle), which tells B8G8R8A8 from
    // R8G8B8A8. VK_FORMAT_UNDEFINED: not a format the table has.
    static VkFormat descriptor_format(std::uint32_t code, const leonida::gfx10::Swizzle& swizzle) {
        for (const auto& row : leonida::gfx10::format_table)
            if (row.gfx10 == code && row.channels == swizzle) return row.vulkan;
        return leonida::gfx10::vulkan_format(code);
    }

    // A V#: stride 0 and no format is a raw buffer of NUM_RECORDS bytes, a stride without a format a structured one
    // of NUM_RECORDS records, and a format a typed one of NUM_RECORDS elements. Its address is a GPU VA, so the
    // view starts at that minus the resource's own.
    static bool unordered_access_view(D3D12_CPU_DESCRIPTOR_HANDLE handle, ID3D12Resource* resource,
                                      leonida::UavView& out) {
        namespace g = leonida::gfx10;
        g::ImageWords words{};
        std::memcpy(words.data(), reinterpret_cast<const void*>(handle.ptr), sizeof(words));
        out.resource = resource_id(resource);
        if (g::is_image(words)) {
            g::Image image{};
            if (!g::decode(words, image)) return false;
            const bool volume = image.type == g::ImageType::image_3d;
            out.buffer = false;
            out.format = std::uint32_t(descriptor_format(image.format, image.swizzle));
            out.mip = image.first_mip;
            out.first_layer = volume ? 0 : image.first_layer;  // a 3D view clears every slice of its mip
            out.layer_count = volume ? 1 : image.layer_count;
            return out.format != VK_FORMAT_UNDEFINED;
        }
        g::BufferWords buffer_words{};
        std::memcpy(buffer_words.data(), words.data(), sizeof(buffer_words));
        g::Buffer buffer{};
        if (!g::decode(buffer_words, buffer)) return false;
        const UINT64 base = resource->GetGPUVirtualAddress();
        if (buffer.address < base) return false;
        out.buffer = true;
        out.offset = buffer.address - base;
        if (buffer.format) {
            const VkFormat format = descriptor_format(buffer.format, buffer.swizzle);
            const auto* layout = leonida::uav_clear::layout(format);
            if (!layout) return false;  // not a typed-UAV format
            out.format = std::uint32_t(format);
            out.element_size = layout->bytes;
            out.size = std::uint64_t(buffer.num_records) * layout->bytes;
        } else if (buffer.stride) {
            out.element_size = buffer.stride;
            out.size = std::uint64_t(buffer.num_records) * buffer.stride;
        } else {
            out.element_size = 4;
            out.size = buffer.num_records;
        }
        return true;
    }

    D3D12_COMMAND_LIST_TYPE type_;
    leonida::CommandList list_;
    bool invalid_ = false;
    UINT front_stencil_ref_ = 0;
    UINT back_stencil_ref_ = 0;
    bool stencil_ref_set_ = false;
};

class Fence final
    : public Child<Fence, ID3D12Fence1Stubs, ID3D12Object, ID3D12DeviceChild, ID3D12Pageable, ID3D12Fence,
                    ID3D12Fence1> {
public:
    // {b4c39e02-6f71-4d8a-b25e-3e9d0f6c1a88}
    static constexpr GUID kTag{0xb4c39e02, 0x6f71, 0x4d8a, {0xb2, 0x5e, 0x3e, 0x9d, 0x0f, 0x6c, 0x1a, 0x88}};

    Fence(leonida::VulkanContext& context, UINT64 initial, D3D12_FENCE_FLAGS flags)
        : fence_(context, initial), flags_(flags), highest_(initial) {}

    ~Fence() override {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        changed_.notify_all();
        if (waiter_.joinable()) waiter_.join();
    }

    bool ok() const { return fence_.semaphore() != VK_NULL_HANDLE; }
    VkSemaphore semaphore() const { return fence_.semaphore(); }

    // A Vulkan timeline only increases; D3D12 allows setting a fence lower. Such a signal is reported and dropped.
    bool claim(UINT64 value) {
        UINT64 current = highest_.load();
        while (value > current)
            if (highest_.compare_exchange_weak(current, value)) return true;
        std::fprintf(stderr, "[rt] d3d12: fence signal %llu is not above %llu; Vulkan timelines cannot go back\n",
                     static_cast<unsigned long long>(value), static_cast<unsigned long long>(current));
        return false;
    }

    UINT64 STDMETHODCALLTYPE GetCompletedValue() override { return fence_.completed(); }
    D3D12_FENCE_FLAGS STDMETHODCALLTYPE GetCreationFlags() override { return flags_; }

    HRESULT STDMETHODCALLTYPE Signal(UINT64 value) override {
        if (claim(value) && !fence_.signal(value)) return E_FAIL;
        return S_OK;
    }

    // A null event blocks until the value is reached, as D3D12 does.
    HRESULT STDMETHODCALLTYPE SetEventOnCompletion(UINT64 value, HANDLE event) override {
        if (fence_.completed() >= value) {
            if (event) SetEvent(event);
            return S_OK;
        }
        if (!event) return fence_.wait(value, UINT64_MAX) ? S_OK : E_FAIL;
        {
            std::lock_guard lock(mutex_);
            pending_.emplace_back(value, event);
            if (!waiter_.joinable()) waiter_ = std::thread([this] { wait_loop(); });
        }
        changed_.notify_all();
        return S_OK;
    }

    static Fence* from(IUnknown* object) { return Child::from<Fence>(object); }

private:
    void wait_loop() {
        std::unique_lock lock(mutex_);
        while (true) {
            changed_.wait(lock, [this] { return stop_ || !pending_.empty(); });
            if (stop_) return;
            UINT64 lowest = UINT64_MAX;
            for (const auto& entry : pending_) lowest = std::min(lowest, entry.first);
            lock.unlock();
            fence_.wait(lowest, 10'000'000);  // 10 ms, so destruction is never held up for long
            const UINT64 done = fence_.completed();
            lock.lock();
            std::erase_if(pending_, [done](const auto& entry) {
                if (entry.first > done) return false;
                SetEvent(entry.second);
                return true;
            });
        }
    }

    leonida::VulkanFence fence_;
    D3D12_FENCE_FLAGS flags_;
    std::atomic<UINT64> highest_;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::vector<std::pair<UINT64, HANDLE>> pending_;
    std::thread waiter_;
    bool stop_ = false;
};

class CommandQueue final
    : public Child<CommandQueue, ID3D12CommandQueueStubs, ID3D12Object, ID3D12DeviceChild, ID3D12Pageable,
                    ID3D12CommandQueue> {
public:
    // {e1a85d76-0b3f-4c29-a7d4-58f2c6e0b913}
    static constexpr GUID kTag{0xe1a85d76, 0x0b3f, 0x4c29, {0xa7, 0xd4, 0x58, 0xf2, 0xc6, 0xe0, 0xb9, 0x13}};

    CommandQueue(leonida::VulkanContext& context, VkQueue queue, std::uint32_t family,
                 const D3D12_COMMAND_QUEUE_DESC& desc)
        : desc_(desc), context_(context), queue_(context, queue, family) {}

    bool ok() const { return queue_.ok(); }
    leonida::ReplayResult last_result() const { return last_result_; }

    D3D12_COMMAND_QUEUE_DESC STDMETHODCALLTYPE GetDesc() override {
        d3d12_trace::record("ID3D12CommandQueue::GetDesc");
        return desc_;
    }

    void STDMETHODCALLTYPE ExecuteCommandLists(UINT count, ID3D12CommandList* const* lists) override {
        constexpr auto method = "ID3D12CommandQueue::ExecuteCommandLists";
        std::lock_guard lock(mutex_);
        std::vector<const leonida::CommandList*> recorded;
        for (UINT i = 0; i < count; ++i) {
            const GraphicsCommandList* list = GraphicsCommandList::from(lists[i]);
            if (!list || list->type() == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
                std::fprintf(stderr, "[rt] d3d12: ExecuteCommandLists: list %u is not a direct command list\n", i);
                last_result_ = leonida::ReplayResult::unsupported;
                d3d12_trace::record(method, "failed");
                return;
            }
            recorded.push_back(&list->commands());
        }
        last_result_ = queue_.execute(recorded);
        d3d12_trace::record(method, last_result_ == leonida::ReplayResult::success ? "ok" : "failed");
        if (last_result_ != leonida::ReplayResult::success)
            std::fprintf(stderr, "[rt] d3d12: ExecuteCommandLists failed: %s; nothing was submitted\n",
                         name(last_result_));
    }

    // Reserved resources are fully backed at creation (docs/d3d12_vulkan.md section 4.2), so tile mappings are not
    // emulated: every tile keeps its own memory. What that loses is reported once each: two resources mapped to the
    // same heap tiles do not alias, and a tile unmapped (or remapped) keeps its contents.
    void STDMETHODCALLTYPE UpdateTileMappings(ID3D12Resource*, UINT, const D3D12_TILED_RESOURCE_COORDINATE*,
                                              const D3D12_TILE_REGION_SIZE*, ID3D12Heap*, UINT,
                                              const D3D12_TILE_RANGE_FLAGS*, const UINT*, const UINT*,
                                              D3D12_TILE_MAPPING_FLAGS) override {
        d3d12_trace::record("ID3D12CommandQueue::UpdateTileMappings", "ignored");
        note_once("ID3D12CommandQueue::UpdateTileMappings ignored: reserved resources are fully backed, mappings are not emulated");
    }

    void STDMETHODCALLTYPE CopyTileMappings(ID3D12Resource*, const D3D12_TILED_RESOURCE_COORDINATE*, ID3D12Resource*,
                                            const D3D12_TILED_RESOURCE_COORDINATE*, const D3D12_TILE_REGION_SIZE*,
                                            D3D12_TILE_MAPPING_FLAGS) override {
        d3d12_trace::record("ID3D12CommandQueue::CopyTileMappings", "ignored");
        note_once("ID3D12CommandQueue::CopyTileMappings ignored: reserved resources are fully backed, mappings are not emulated");
    }

    HRESULT STDMETHODCALLTYPE Signal(ID3D12Fence* fence, UINT64 value) override {
        constexpr auto method = "ID3D12CommandQueue::Signal";
        Fence* f = Fence::from(fence);
        if (!f) return d3d12_trace::result(method, E_INVALIDARG);
        std::lock_guard lock(mutex_);
        if (!f->claim(value)) return d3d12_trace::result(method, S_OK);
        return d3d12_trace::result(method, queue_.signal(f->semaphore(), value) ? S_OK : E_FAIL);
    }

    HRESULT STDMETHODCALLTYPE Wait(ID3D12Fence* fence, UINT64 value) override {
        constexpr auto method = "ID3D12CommandQueue::Wait";
        Fence* f = Fence::from(fence);
        if (!f) return d3d12_trace::result(method, E_INVALIDARG);
        std::lock_guard lock(mutex_);
        queue_.wait(f->semaphore(), value);
        return d3d12_trace::result(method, S_OK);
    }

    // Ticks per second of the timestamps EndQuery writes: one tick is timestampPeriod nanoseconds. Every D3D12
    // queue of the device shares one VkQueue, so they share one frequency. E_FAIL when that queue has no
    // timestamps (timestampValidBits 0), as D3D12 reports for a queue that does not support them.
    HRESULT STDMETHODCALLTYPE GetTimestampFrequency(UINT64* frequency) override {
        if (!frequency) return E_INVALIDARG;
        const double period = context_.timestamp_period();
        if (!(period > 0)) return E_FAIL;
        *frequency = static_cast<UINT64>(1e9 / period + 0.5);
        return S_OK;
    }

    static CommandQueue* from(IUnknown* object) { return Child::from<CommandQueue>(object); }

private:
    D3D12_COMMAND_QUEUE_DESC desc_;
    leonida::VulkanContext& context_;
    std::mutex mutex_;
    leonida::VulkanQueue queue_;
    leonida::ReplayResult last_result_ = leonida::ReplayResult::success;
};

}  // namespace

HRESULT create_command_queue(leonida::VulkanContext& context, VkQueue queue, std::uint32_t queue_family,
                             const D3D12_COMMAND_QUEUE_DESC& desc, REFIID riid, void** out, IUnknown* device) {
    if (!out) return E_POINTER;
    *out = nullptr;
    auto* object = new CommandQueue(context, queue, queue_family, desc);
    if (!object->ok()) {
        object->Release();
        return E_FAIL;
    }
    object->attach_device(device);
    return hand_out(object, riid, out);
}

HRESULT create_command_allocator(D3D12_COMMAND_LIST_TYPE type, REFIID riid, void** out, IUnknown* device) {
    if (!out) return E_POINTER;
    *out = nullptr;
    auto* object = new CommandAllocator(type);
    object->attach_device(device);
    return hand_out(object, riid, out);
}

HRESULT create_command_list(D3D12_COMMAND_LIST_TYPE type, ID3D12CommandAllocator* allocator, REFIID riid,
                            void** out, IUnknown* device) {
    if (!out) return E_POINTER;
    *out = nullptr;
    CommandAllocator* command_allocator = CommandAllocator::from(allocator);
    if (!command_allocator || command_allocator->type() != type) return E_INVALIDARG;
    if (type != D3D12_COMMAND_LIST_TYPE_DIRECT && type != D3D12_COMMAND_LIST_TYPE_BUNDLE)
        stub("create_command_list(non-direct type)");
    auto* object = new GraphicsCommandList(type);  // created open, as D3D12 does
    object->attach_device(device);
    return hand_out(object, riid, out);
}

HRESULT create_fence(leonida::VulkanContext& context, UINT64 initial_value, D3D12_FENCE_FLAGS flags, REFIID riid,
                     void** out, IUnknown* device) {
    if (!out) return E_POINTER;
    *out = nullptr;
    auto* object = new Fence(context, initial_value, flags);
    if (!object->ok()) {
        object->Release();
        return E_FAIL;
    }
    object->attach_device(device);
    return hand_out(object, riid, out);
}

leonida::ReplayResult last_execute_result(ID3D12CommandQueue* queue) {
    const CommandQueue* q = CommandQueue::from(queue);
    return q ? q->last_result() : leonida::ReplayResult::unsupported;
}

bool front_back_stencil_ref(ID3D12GraphicsCommandList8* list, UINT& front, UINT& back) {
    const GraphicsCommandList* l = GraphicsCommandList::from(list);
    if (!l) return false;
    front = l->front_stencil_ref();
    back = l->back_stencil_ref();
    return true;
}

}  // namespace rt::d3d12
