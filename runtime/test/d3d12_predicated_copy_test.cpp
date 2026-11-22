// Synthetic byte-for-byte CopyBufferRegion oracle. --warp-only measures the native reference without Vulkan.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>
#include <array>
#include <vector>
#include <cstring>
#include <cstdio>
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return false; } } while (0)
namespace {
template<class T> struct Ref { T* p = nullptr; ~Ref() { if (p) p->Release(); } T* operator->() const { return p; } };
struct Case { UINT64 source_bytes, destination_bytes, source_offset, destination_offset, bytes; };
bool buffer(ID3D12Device* device, UINT64 size, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state, Ref<ID3D12Resource>& out) {
    const D3D12_HEAP_PROPERTIES properties{heap};
    D3D12_RESOURCE_DESC desc{}; desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size; desc.Height = desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&out.p))));
    return true;
}
bool write(ID3D12Resource* resource, const void* bytes, SIZE_T size) {
    void* mapped = nullptr; const D3D12_RANGE none{};
    CHECK(SUCCEEDED(resource->Map(0, &none, &mapped)) && mapped);
    std::memcpy(mapped, bytes, size); const D3D12_RANGE written{0, size}; resource->Unmap(0, &written); return true;
}
bool run(ID3D12Device* device, const Case& c, UINT64 predicate, D3D12_PREDICATION_OP op,
         bool enabled, std::vector<unsigned char>& actual) {
    Ref<ID3D12Resource> source, destination, seed, condition, readback;
    CHECK(buffer(device,c.source_bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,source));
    CHECK(buffer(device,c.destination_bytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_COPY_DEST,destination));
    CHECK(buffer(device,c.destination_bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,seed));
    CHECK(buffer(device,24,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,condition));
    CHECK(buffer(device,c.destination_bytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,readback));
    std::vector<unsigned char> input(SIZE_T(c.source_bytes)), sentinel(SIZE_T(c.destination_bytes));
    for (SIZE_T i=0;i<input.size();++i) input[i]=static_cast<unsigned char>(i*37+11);
    for (SIZE_T i=0;i<sentinel.size();++i) sentinel[i]=static_cast<unsigned char>(i*13+173);
    const UINT64 conditions[3]={UINT64_MAX,predicate,0};
    CHECK(write(source.p,input.data(),input.size()) && write(seed.p,sentinel.data(),sentinel.size()) && write(condition.p,conditions,sizeof(conditions)));
    Ref<ID3D12CommandQueue> queue; Ref<ID3D12CommandAllocator> allocator; Ref<ID3D12GraphicsCommandList> list; Ref<ID3D12Fence> fence;
    const D3D12_COMMAND_QUEUE_DESC desc{};
    CHECK(SUCCEEDED(device->CreateCommandQueue(&desc,IID_PPV_ARGS(&queue.p))));
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.p,nullptr,IID_PPV_ARGS(&list.p))));
    CHECK(SUCCEEDED(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence.p))));
    list->CopyBufferRegion(destination.p,0,seed.p,0,c.destination_bytes);
    // Explicit dependency also covers the unpredicated control path.
    D3D12_RESOURCE_BARRIER seed_barrier{}; seed_barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    seed_barrier.Transition.pResource=destination.p;
    seed_barrier.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    seed_barrier.Transition.StateBefore=D3D12_RESOURCE_STATE_COPY_DEST;
    seed_barrier.Transition.StateAfter=D3D12_RESOURCE_STATE_COMMON;
    list->ResourceBarrier(1,&seed_barrier);
    std::swap(seed_barrier.Transition.StateBefore,seed_barrier.Transition.StateAfter);
    list->ResourceBarrier(1,&seed_barrier);
    if(enabled) list->SetPredication(condition.p,8,op);
    list->CopyBufferRegion(destination.p,c.destination_offset,source.p,c.source_offset,c.bytes);
    list->SetPredication(nullptr,0,D3D12_PREDICATION_OP_EQUAL_ZERO);
    D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource=destination.p; barrier.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore=D3D12_RESOURCE_STATE_COPY_DEST; barrier.Transition.StateAfter=D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1,&barrier); list->CopyBufferRegion(readback.p,0,destination.p,0,c.destination_bytes);
    CHECK(SUCCEEDED(list->Close())); ID3D12CommandList* lists[]={list.p}; queue->ExecuteCommandLists(1,lists);
    CHECK(SUCCEEDED(queue->Signal(fence.p,1))); HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr); CHECK(event);
    const HRESULT hr=fence->SetEventOnCompletion(1,event); const DWORD wait=SUCCEEDED(hr)?WaitForSingleObject(event,10000):WAIT_FAILED;
    CloseHandle(event); CHECK(SUCCEEDED(hr)&&wait==WAIT_OBJECT_0);
    void* mapped=nullptr; const D3D12_RANGE range{0,SIZE_T(c.destination_bytes)};
    CHECK(SUCCEEDED(readback->Map(0,&range,&mapped))&&mapped);
    actual.assign(static_cast<unsigned char*>(mapped),static_cast<unsigned char*>(mapped)+c.destination_bytes);
    const D3D12_RANGE none{}; readback->Unmap(0,&none);
    const bool executes=!enabled || (op==D3D12_PREDICATION_OP_EQUAL_ZERO ? predicate!=0 : predicate==0);
    if(executes && c.bytes) std::memcpy(sentinel.data()+c.destination_offset,input.data()+c.source_offset,SIZE_T(c.bytes));
    for(SIZE_T i=0;i<sentinel.size();++i) if(actual[i]!=sentinel[i]) {
        std::fprintf(stderr,"copy %llu/%llu +%llu/%llu size %llu predicate %016llx op %u enabled %d byte %zu got %02x expected %02x\n",
            c.source_bytes,c.destination_bytes,c.source_offset,c.destination_offset,c.bytes,predicate,unsigned(op),int(enabled),i,unsigned(actual[i]),unsigned(sentinel[i])); return false;
    }
    return true;
}
}
int main(int argc,char** argv) {
    const bool native_only=argc==2&&std::strcmp(argv[1],"--warp-only")==0;
    if(argc!=1&&!native_only) return 2;
    vktest::Device vk;
    Ref<IDXGIFactory4> factory; Ref<IDXGIAdapter> adapter; Ref<ID3D12Device> host,ours;
    if(FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p))) ||
       FAILED(D3D12CreateDevice(adapter.p,D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&host.p)))) return 1;
    if(!native_only) {
        if(!vk.create())return 1;
        if(!vk.conditional_rendering || !vk.storage_buffer_8bit) {
            std::puts("predicated buffer copies: skipped (requires conditional rendering and storageBuffer8BitAccess)");
            return 0;
        }
        rt::d3d12::VulkanDeviceInfo info{}; info.get_instance_proc=vk.instance_proc; info.instance=vk.instance;
        info.physical=vk.physical; info.device=vk.device; info.queue=vk.queue; info.queue_family=vk.family;
        info.conditional_rendering=vk.conditional_rendering;
        info.storage_buffer_8bit=vk.storage_buffer_8bit;
        if(FAILED(rt::d3d12::create_vulkan_device(info,IID_PPV_ARGS(&ours.p))))return 1;
    }
    const Case cases[]={{1,1,0,0,1},{2,3,1,2,1},{7,11,1,3,5},{13,17,3,1,7},{33,35,0,3,32},
        {257,259,1,3,256},{4099,4103,3,7,4096},{4099,4103,4098,4102,1},{13,17,13,17,0}};
    unsigned count=0;
    for(const auto& c:cases)for(UINT64 value:{UINT64_C(0),UINT64_C(1),UINT64_C(1)<<32,(UINT64_C(1)<<32)|7,UINT64_MAX})
        for(auto op:{D3D12_PREDICATION_OP_EQUAL_ZERO,D3D12_PREDICATION_OP_NOT_EQUAL_ZERO})for(bool enabled:{false,true}) {
            std::vector<unsigned char> expected,actual;
            if(!run(host.p,c,value,op,enabled,expected)) {std::fprintf(stderr,"WARP reference failed\n");return 1;}
            if(!native_only&&(!run(ours.p,c,value,op,enabled,actual)||actual!=expected)){std::fprintf(stderr,"Vulkan comparison failed\n");return 1;}
            ++count;
        }
    if(!native_only&&vk.validation_errors())return 1;
    std::printf("predicated buffer copies: %u exact cases (%s)\n",count,native_only?"WARP + literal byte oracle":"Vulkan + WARP + literal byte oracle");
}
