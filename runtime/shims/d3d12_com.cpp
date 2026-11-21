#include "d3d12_com.h"

#include "d3d12_stubs.h"  // generated

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
    case leonida::ReplayResult::unsupported: return "unsupported";
    }
    return "?";
}

// IUnknown and ID3D12Object for one implementation class. QueryInterface answers the listed public interfaces
// and the class's private Derived::kTag (which returns the implementation itself); anything else is logged with
// its IID and refused, as section 3.1 asks.
template<class Derived, class Stubs, class... Interfaces>
class Object : public Stubs {
public:
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG left = --refs_;
        if (!left) delete static_cast<Derived*>(this);
        return left;
    }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (riid == Derived::kTag) {
            *out = static_cast<Derived*>(this);
        } else if (riid == __uuidof(IUnknown) || ((riid == __uuidof(Interfaces)) || ...)) {
            *out = static_cast<Stubs*>(this);  // one inheritance chain: every interface shares this pointer
        } else {
            std::fprintf(stderr,
                         "[rt] d3d12: QueryInterface: unknown IID {%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x}\n",
                         static_cast<unsigned long>(riid.Data1), riid.Data2, riid.Data3, riid.Data4[0], riid.Data4[1],
                         riid.Data4[2], riid.Data4[3], riid.Data4[4], riid.Data4[5], riid.Data4[6], riid.Data4[7]);
            return E_NOINTERFACE;
        }
        AddRef();
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID guid, UINT* size, void* data) override {
        if (!size) return E_INVALIDARG;
        std::lock_guard lock(data_mutex_);
        for (const auto& [key, bytes] : data_) {
            if (key != guid) continue;
            if (data) {
                if (*size < bytes.size()) return DXGI_ERROR_MORE_DATA;
                std::memcpy(data, bytes.data(), bytes.size());
            }
            *size = static_cast<UINT>(bytes.size());
            return S_OK;
        }
        *size = 0;
        return DXGI_ERROR_NOT_FOUND;
    }

    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID guid, UINT size, const void* data) override {
        std::lock_guard lock(data_mutex_);
        std::erase_if(data_, [&](const auto& entry) { return entry.first == guid; });
        if (data && size) {
            const auto* bytes = static_cast<const std::uint8_t*>(data);
            data_.emplace_back(guid, std::vector<std::uint8_t>(bytes, bytes + size));
        }
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE SetName(LPCWSTR) override { return S_OK; }

protected:
    template<class T>
    static T* from(IUnknown* object) {
        if (!object) return nullptr;
        void* implementation = nullptr;
        if (FAILED(object->QueryInterface(T::kTag, &implementation))) return nullptr;
        static_cast<T*>(implementation)->Release();  // the caller holds its own reference
        return static_cast<T*>(implementation);
    }

    virtual ~Object() = default;

private:
    std::atomic<ULONG> refs_{1};
    std::mutex data_mutex_;
    std::vector<std::pair<GUID, std::vector<std::uint8_t>>> data_;
};

template<class T>
HRESULT hand_out(T* object, REFIID riid, void** out) {
    const HRESULT hr = object->QueryInterface(riid, out);
    object->Release();
    return hr;
}

class CommandAllocator final
    : public Object<CommandAllocator, ID3D12CommandAllocatorStubs, ID3D12Object, ID3D12DeviceChild, ID3D12Pageable,
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
    : public Object<GraphicsCommandList, ID3D12GraphicsCommandList7Stubs, ID3D12Object, ID3D12DeviceChild,
                    ID3D12CommandList, ID3D12GraphicsCommandList, ID3D12GraphicsCommandList1,
                    ID3D12GraphicsCommandList2, ID3D12GraphicsCommandList3, ID3D12GraphicsCommandList4,
                    ID3D12GraphicsCommandList5, ID3D12GraphicsCommandList6, ID3D12GraphicsCommandList7> {
public:
    // {2f8a7c33-91d4-4b6e-8e05-6d3f1a9b2c47}
    static constexpr GUID kTag{0x2f8a7c33, 0x91d4, 0x4b6e, {0x8e, 0x05, 0x6d, 0x3f, 0x1a, 0x9b, 0x2c, 0x47}};
    explicit GraphicsCommandList(D3D12_COMMAND_LIST_TYPE type) : type_(type) {}

    const leonida::CommandList& commands() const { return list_; }

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

    static GraphicsCommandList* from(IUnknown* object) { return Object::from<GraphicsCommandList>(object); }

private:
    template<class E>
    static std::uint32_t u(E value) {
        return static_cast<std::uint32_t>(value);
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
};

class Fence final
    : public Object<Fence, ID3D12Fence1Stubs, ID3D12Object, ID3D12DeviceChild, ID3D12Pageable, ID3D12Fence,
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

    static Fence* from(IUnknown* object) { return Object::from<Fence>(object); }

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
    : public Object<CommandQueue, ID3D12CommandQueueStubs, ID3D12Object, ID3D12DeviceChild, ID3D12Pageable,
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

    static CommandQueue* from(IUnknown* object) { return Object::from<CommandQueue>(object); }

private:
    D3D12_COMMAND_QUEUE_DESC desc_;
    std::mutex mutex_;
    leonida::VulkanQueue queue_;
    leonida::ReplayResult last_result_ = leonida::ReplayResult::success;
};

}  // namespace

HRESULT create_command_queue(leonida::VulkanContext& context, VkQueue queue, std::uint32_t queue_family,
                             const D3D12_COMMAND_QUEUE_DESC& desc, REFIID riid, void** out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    auto* object = new CommandQueue(context, queue, queue_family, desc);
    if (!object->ok()) {
        object->Release();
        return E_FAIL;
    }
    return hand_out(object, riid, out);
}

HRESULT create_command_allocator(D3D12_COMMAND_LIST_TYPE type, REFIID riid, void** out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    return hand_out(new CommandAllocator(type), riid, out);
}

HRESULT create_command_list(D3D12_COMMAND_LIST_TYPE type, ID3D12CommandAllocator* allocator, REFIID riid,
                            void** out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    if (!allocator) return E_INVALIDARG;
    if (type != D3D12_COMMAND_LIST_TYPE_DIRECT) stub("create_command_list(non-direct type)");
    return hand_out(new GraphicsCommandList(type), riid, out);  // created open, as D3D12 does
}

HRESULT create_fence(leonida::VulkanContext& context, UINT64 initial_value, D3D12_FENCE_FLAGS flags, REFIID riid,
                     void** out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    auto* object = new Fence(context, initial_value, flags);
    if (!object->ok()) {
        object->Release();
        return E_FAIL;
    }
    return hand_out(object, riid, out);
}

leonida::ReplayResult last_execute_result(ID3D12CommandQueue* queue) {
    const CommandQueue* q = CommandQueue::from(queue);
    return q ? q->last_result() : leonida::ReplayResult::unsupported;
}

}  // namespace rt::d3d12
