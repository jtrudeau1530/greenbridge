// D3D-97: cross-queue Wait-before-Signal must make progress. No game files.
#include "../shims/d3d12_com.h"

#include <cstdio>

namespace {
template<class T> struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

bool wait_for(ID3D12Fence* fence, UINT64 value) {
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return false;
    const bool complete = SUCCEEDED(fence->SetEventOnCompletion(value, event)) &&
                          WaitForSingleObject(event, 3000) == WAIT_OBJECT_0;
    CloseHandle(event);
    return complete;
}
}  // namespace

int main() {
    Ref<ID3D12Device> device;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device.p)))) return 77;
    D3D12_COMMAND_QUEUE_DESC desc{};
    Ref<ID3D12CommandQueue> copy, compute, direct_a, direct_b;
    desc.Type = D3D12_COMMAND_LIST_TYPE_COPY;
    if (FAILED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&copy.p)))) return 77;
    desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    if (FAILED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&compute.p)))) return 77;
    desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&direct_a.p))) ||
        FAILED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&direct_b.p)))) return 77;

    Ref<ID3D12Fence> gate_copy, gate_direct, done_copy, done_direct;
    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate_copy.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate_direct.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done_copy.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done_direct.p)))) return 1;

    // Submission order is deliberately opposite the dependency order.
    const bool submitted = SUCCEEDED(copy->Wait(gate_copy.p, 1)) &&
        SUCCEEDED(copy->Signal(done_copy.p, 1)) &&
        SUCCEEDED(direct_a->Wait(gate_direct.p, 1)) &&
        SUCCEEDED(direct_a->Signal(done_direct.p, 1)) &&
        SUCCEEDED(compute->Signal(gate_copy.p, 2)) &&
        SUCCEEDED(direct_b->Signal(gate_direct.p, 2));
    const bool copy_complete = submitted && wait_for(done_copy.p, 1);
    const bool direct_complete = submitted && wait_for(done_direct.p, 1);
    if (!copy_complete || !direct_complete) {
        // Unblock submitted waits before queue destructors call vkQueueWaitIdle.
        gate_copy->Signal(1);
        gate_direct->Signal(1);
        std::fprintf(stderr, "cross-queue Wait-before-Signal did not complete\n");
        return 1;
    }
    std::puts("copy/compute and direct/direct Wait-before-Signal completed");
    return 0;
}
