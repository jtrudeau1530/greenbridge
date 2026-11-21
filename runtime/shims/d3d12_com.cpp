#include "d3d12_com.h"

#include "d3d12_stubs.h"  // generated
#include "d3d12_com_object.h"

#include <atomic>
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
    static std::mutex mutex;
    static std::set<std::string> seen;
    std::lock_guard lock(mutex);
    if (seen.insert(method).second) std::fprintf(stderr, "[rt] d3d12: unimplemented %s\n", method);
}

namespace {

const char* name(leonida::ReplayResult result) {
    switch (result) {
    case leonida::ReplayResult::success: return "success";
    case leonida::ReplayResult::list_open: return "a list is not closed";
    case leonida::ReplayResult::unresolved_resource: return "unknown resource";
    case leonida::ReplayResult::invalid_subresource: return "invalid subresource";
    case leonida::ReplayResult::invalid_clear: return "invalid clear";
    case leonida::ReplayResult::invalid_copy: return "invalid copy";
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

    D3D12_COMMAND_LIST_TYPE STDMETHODCALLTYPE GetType() override { return type_; }

    // E_INVALIDARG when a call during recording was invalid, as D3D12's Close reports it.
    HRESULT STDMETHODCALLTYPE Close() override {
        if (!list_.close()) return E_FAIL;
        return invalid_ ? E_INVALIDARG : S_OK;
    }

    HRESULT STDMETHODCALLTYPE Reset(ID3D12CommandAllocator* allocator, ID3D12PipelineState* state) override {
        if (!allocator) return E_INVALIDARG;
        if (!list_.reset()) return E_FAIL;  // D3D12 resets only closed lists
        invalid_ = false;
        front_stencil_ref_ = back_stencil_ref_ = 0;  // Reset returns every list state to its default
        if (state) stub("ID3D12GraphicsCommandList::Reset(initial pipeline state)");
        return S_OK;
    }

    void STDMETHODCALLTYPE ResourceBarrier(UINT count, const D3D12_RESOURCE_BARRIER* barriers) override {
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
        check(list_.copy_buffer({resource_id(dst), dst_offset, resource_id(src), src_offset, bytes}));
    }

    void STDMETHODCALLTYPE CopyTextureRegion(const D3D12_TEXTURE_COPY_LOCATION* dst, UINT x, UINT y, UINT z,
                                             const D3D12_TEXTURE_COPY_LOCATION* src, const D3D12_BOX* box) override {
        if (!dst || !src) return check(false);
        leonida::CopyTexture copy{location(*dst), x, y, z, location(*src), box != nullptr, {}};
        if (box) copy.box = {box->left, box->top, box->front, box->right, box->bottom, box->back};
        check(list_.copy_texture(copy));
    }

    void STDMETHODCALLTYPE CopyResource(ID3D12Resource* dst, ID3D12Resource* src) override {
        check(list_.copy_resource({resource_id(dst), resource_id(src)}));
    }

    void STDMETHODCALLTYPE ClearRenderTargetView(D3D12_CPU_DESCRIPTOR_HANDLE view, const FLOAT color[4], UINT count,
                                                 const D3D12_RECT* rects) override {
        if (!view.ptr || !color) return check(false);
        check(list_.clear_render_target(*reinterpret_cast<const leonida::TargetView*>(view.ptr),
                                        {color[0], color[1], color[2], color[3]}, to_rects(count, rects)));
    }

    void STDMETHODCALLTYPE ClearDepthStencilView(D3D12_CPU_DESCRIPTOR_HANDLE view, D3D12_CLEAR_FLAGS flags,
                                                 FLOAT depth, UINT8 stencil, UINT count,
                                                 const D3D12_RECT* rects) override {
        if (!view.ptr) return check(false);
        check(list_.clear_depth_stencil(*reinterpret_cast<const leonida::TargetView*>(view.ptr), u(flags), depth,
                                        stencil, to_rects(count, rects)));
    }

    // GCL8. Vulkan can set the two faces independently with vkCmdSetStencilReference(VK_STENCIL_FACE_FRONT_BIT /
    // BACK_BIT), but that state only matters when a pipeline uses it, and this layer has no pipeline or draw path
    // yet. Retain the values for that path and report that nothing consumed them, rather than emit a call whose
    // effect no later command can observe.
    void STDMETHODCALLTYPE OMSetFrontAndBackStencilRef(UINT front, UINT back) override {
        front_stencil_ref_ = front;
        back_stencil_ref_ = back;
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

    D3D12_COMMAND_LIST_TYPE type_;
    leonida::CommandList list_;
    bool invalid_ = false;
    UINT front_stencil_ref_ = 0;
    UINT back_stencil_ref_ = 0;
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
        : desc_(desc), queue_(context, queue, family) {}

    bool ok() const { return queue_.ok(); }
    leonida::ReplayResult last_result() const { return last_result_; }

    D3D12_COMMAND_QUEUE_DESC STDMETHODCALLTYPE GetDesc() override { return desc_; }

    void STDMETHODCALLTYPE ExecuteCommandLists(UINT count, ID3D12CommandList* const* lists) override {
        std::lock_guard lock(mutex_);
        std::vector<const leonida::CommandList*> recorded;
        for (UINT i = 0; i < count; ++i) {
            const GraphicsCommandList* list = GraphicsCommandList::from(lists[i]);
            if (!list) {
                std::fprintf(stderr, "[rt] d3d12: ExecuteCommandLists: list %u is not one of ours\n", i);
                last_result_ = leonida::ReplayResult::unsupported;
                return;
            }
            recorded.push_back(&list->commands());
        }
        last_result_ = queue_.execute(recorded);
        if (last_result_ != leonida::ReplayResult::success)
            std::fprintf(stderr, "[rt] d3d12: ExecuteCommandLists failed: %s; nothing was submitted\n",
                         name(last_result_));
    }

    HRESULT STDMETHODCALLTYPE Signal(ID3D12Fence* fence, UINT64 value) override {
        Fence* f = Fence::from(fence);
        if (!f) return E_INVALIDARG;
        std::lock_guard lock(mutex_);
        if (!f->claim(value)) return S_OK;
        return queue_.signal(f->semaphore(), value) ? S_OK : E_FAIL;
    }

    HRESULT STDMETHODCALLTYPE Wait(ID3D12Fence* fence, UINT64 value) override {
        Fence* f = Fence::from(fence);
        if (!f) return E_INVALIDARG;
        std::lock_guard lock(mutex_);
        queue_.wait(f->semaphore(), value);
        return S_OK;
    }

    static CommandQueue* from(IUnknown* object) { return Child::from<CommandQueue>(object); }

private:
    D3D12_COMMAND_QUEUE_DESC desc_;
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
    if (!allocator) return E_INVALIDARG;
    if (type != D3D12_COMMAND_LIST_TYPE_DIRECT) stub("create_command_list(non-direct type)");
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
