// Synthetic byte-copy, forced descriptor windows and pending-submit lifetime regression. No game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"
#include <array>
#include <cstring>
#include <cstdio>
#define EXPECT(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s\n",__LINE__,#x); return 1; } } while(0)
int main() {
    vktest::Device d;
    if(!d.create())return 1;
    if(!d.conditional_rendering || !d.storage_buffer_8bit) {
        std::puts("predicated copy: skipped (requires conditional rendering and storageBuffer8BitAccess)"); return 0;
    }
    using namespace leonida;
    VkPhysicalDeviceMemoryProperties memory{};
    d.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(d.physical,&memory);
    VkPhysicalDeviceProperties properties{};
    d.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(d.physical,&properties);
    auto limits=properties.limits;
    // Smaller advertised limits force both dispatch and descriptor-window splits on ordinary GPUs.
    limits.maxStorageBufferRange=static_cast<std::uint32_t>(std::max<VkDeviceSize>(256,limits.minStorageBufferOffsetAlignment));
    limits.maxComputeWorkGroupCount[0]=1;
    VulkanContext context(d.device_proc,d.device,true,true);
    context.configure_predication(d.device_proc,memory,properties.limits.maxStorageBufferRange,d.family);
    VulkanQueue queue(context,d.queue,d.family);
    VulkanFence gate(context,0);
    EXPECT(context.loaded() && queue.ok() && gate.semaphore());
    constexpr std::size_t bytes=8207;
    constexpr auto usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    void *src_data=nullptr,*dst_data=nullptr,*pred_data=nullptr;
    auto src=d.buffer(bytes,usage,&src_data),dst=d.buffer(bytes,usage,&dst_data);
    auto pred=d.buffer(16,usage,&pred_data);
    EXPECT(src&&dst&&pred&&src_data&&dst_data&&pred_data);
    std::array<unsigned char,bytes> input{},expected{};
    for(std::size_t i=0;i<bytes;++i){ input[i]=static_cast<unsigned char>(37*i+11); expected[i]=static_cast<unsigned char>(13*i+173); }
    std::memcpy(src_data,input.data(),bytes); std::memcpy(dst_data,expected.data(),bytes);
    const std::uint64_t predicates[]={UINT64_C(1)<<32,0}; std::memcpy(pred_data,predicates,sizeof(predicates));
    ResourceBinding a{};a.buffer=src;a.buffer_offset=1;a.size=bytes-1;
    ResourceBinding b{};b.buffer=dst;b.buffer_offset=3;b.size=bytes-3;
    ResourceBinding p{};p.buffer=pred;p.size=16;
    context.add_resource(1,a);context.add_resource(2,b);context.add_resource(3,p);
    CommandList list;
    EXPECT(list.set_predication({3,0,d3d12_predication::equal_zero}));
    EXPECT(list.copy_buffer({2,4,1,2,4096}));  // high-word-only condition; offsets include arena prefixes
    EXPECT(list.copy_buffer({2,6000,2,4,33})); // dependent, non-overlapping copy within one VkBuffer
    EXPECT(list.set_predication({3,8,d3d12_predication::equal_zero}));
    EXPECT(list.copy_buffer({2,0,1,0,3}));    // suppressed: preserves destination prefix exactly
    EXPECT(list.set_predication({3,0,d3d12_predication::not_equal_zero}));
    EXPECT(list.copy_buffer({2,7000,1,0,37}));// suppressed: high word must not be truncated
    EXPECT(list.set_predication({0,0,0}));
    EXPECT(list.copy_buffer({2,5000,1,0,17}));// reset restores ordinary transfer path
    EXPECT(list.close());
    std::memcpy(expected.data()+7,input.data()+3,4096);
    std::memcpy(expected.data()+6003,expected.data()+7,33);
    std::memcpy(expected.data()+5003,input.data()+1,17);
    const CommandList* pointer=&list;
    // A device without the helper must submit nothing, rather than run a transfer unconditionally.
    EXPECT(queue.execute({&pointer,1})==ReplayResult::unsupported);
    EXPECT(std::memcmp(dst_data,expected.data(),bytes)!=0); // expected includes the not-yet-executed copies
    for(std::size_t i=0;i<bytes;++i) EXPECT(static_cast<unsigned char*>(dst_data)[i]==static_cast<unsigned char>(13*i+173));
    context.configure_predicated_copy(d.device_proc,limits);
    queue.wait(gate.semaphore(),1);
    bool submitted=true;
    for(unsigned i=0;i<32;++i) if(queue.execute({&pointer,1})!=ReplayResult::success){submitted=false;break;}
    // Release even on preparation failure, before any destructor can wait for the queue.
    EXPECT(gate.signal(1));EXPECT(queue.idle());EXPECT(submitted);
    EXPECT(std::memcmp(dst_data,expected.data(),bytes)==0 && std::memcmp(src_data,input.data(),bytes)==0);
    for(unsigned i=0;i<32;++i) EXPECT(queue.execute({&pointer,1})==ReplayResult::success);
    EXPECT(queue.idle());
    EXPECT(std::memcmp(dst_data,expected.data(),bytes)==0 && std::memcmp(src_data,input.data(),bytes)==0);
    EXPECT(d.validation_errors()==0);
    std::puts("predicated copy: 32 gated + 32 reused submissions; byte offsets, forced windows, copy dependency, rebinding/reset and untouched bytes passed");
}
