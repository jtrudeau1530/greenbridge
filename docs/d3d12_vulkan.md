# Xbox D3D12 (`d3d12_xs`) to Vulkan

**Status: in progress. A Vulkan-backed `ID3D12Device` (queues, allocators, command lists, fences, descriptor heaps,
render-target and depth views, committed buffers and 2D textures, copies) runs behind the Xbox-ABI device proxy
(section 3.2). `D3D12XboxCreateDevice` uses it when `LEONIDA_D3D12=vulkan` is set and the host PC device otherwise.
Shader views, root signatures, PSOs, draws and the swapchain are not implemented.** This is the plan for replacing the console's
`d3d12_xs.dll` with an implementation that records D3D12 commands and replays them on Vulkan, while producing
exactly the shader-visible contract that `tools/rdna2spirv.py` already emits against (see
[design.md](design.md#translator-status-and-runtime-contract)).

The document is deliberately conservative about what is known. Every claim is marked:

- **Verified** — observed in this repository or in the game package, with a way to reproduce it.
- **Public** — documented in the Windows SDK, Microsoft's public GDK/D3D12 documentation, or the Vulkan
  specification.
- **Assumption** — a design choice that must be checked against the game's executable when it is available.
- **Open** — unknown; listed with a way to resolve it.

Nothing here uses game data, keys or decryption. The package is read-only; the executable that would answer most
of the open questions is not in it.

## 1. The job

The game is an Xbox Series build (`game_scarlett_master.exe`, `d3d12_xs.h` / `d3d12_xs.lib`). It was compiled
against the Xbox flavour of D3D12 and calls the console driver through COM interfaces. The layer must:

1. **Intercept the D3D12 surface.** Free functions exported by `d3d12_xs.dll` (device creation, root-signature
   serialization) and every COM method the game calls on the returned objects. The objects are our own C++
   objects; the vtables must have the exact method order the game was compiled against.
2. **Preserve D3D12 semantics that affect output.** Command lists record; execution is deferred; descriptors are
   values (GPU virtual addresses or CPU handles); resource states are explicit; root arguments are set at record
   time; allocators and fences control reuse.
3. **Produce the translator's shader contract.** The translated SPIR-V is already written against a bindless set
   0, a 144-byte push-constant block, a descriptor hash table and one GPU address space. Changing that contract
   would invalidate 26,503 translated shaders and the GPU tests that check them.

The layer is not a general D3D12-on-Vulkan compatibility layer. It only needs the calls this game makes, but each
call it accepts must behave like D3D12.

## 2. What is verifiable today

| Fact | Evidence | Confidence |
|---|---|---|
| The public D3D12 interface chain and method order are in the Windows SDK `d3d12.h` (10.0.26100.0 here). The Xbox public reference lists the same base interfaces and method names under `d3d12_xs.h` / `d3d12_xs.lib`; the measured Xbox device ABI is mostly that order, with deviations at slots 62/76 and heap slots 9/10 (section 7). | Local SDK header; [Xbox public graphics reference](https://devdocs.xbox.com/reference/graphics/d3d12); e.g. [CreateCommittedResource](https://devdocs.xbox.com/reference/graphics/d3d12/interfaces/id3d12device/methods/id3d12device_createcommittedresource_public) | Public; measured deviations in section 7 |
| The game's raytracing helper is a shipped DLL, `XboxBvhBuilder.dll`, whose export signatures name `ID3D12Device11`, `ID3D12GraphicsCommandList4`, `ID3D12GraphicsCommandList6` and `ID3D12GraphicsCommandList8`, plus Xbox XDXR batched/streaming acceleration-structure types. | PE export table of `XboxBvhBuilder.dll` (884,224 bytes); see Appendix A | Verified |
| That DLL imports only `d3d12_xs.dll!D3D12SerializeVersionedRootSignature` and `xmem.dll!XMemVirtualQuery` from console libraries. It receives the device and command lists from the game. | PE import table; Appendix A | Verified |
| `amd_fidelityfx_upscaler_xbox.dll` imports only `d3d12_xs.dll!D3D12CreateRootSignatureDeserializer` and contains the base `ID3D12Device`, `ID3D12Resource`, `ID3D12PipelineState`, `ID3D12RootSignature`, `ID3D12Heap`, `ID3D12DescriptorHeap` and `ID3D12CommandSignature` IIDs. | PE import table and GUID byte scan; Appendix A | Verified |
| None of the other scanned non-CRT DLLs (`amd_fidelityfx_loader_xbox`, `Bink2Scarlett`, `libHttpClient.GDK`, `XCurl`) imports `d3d12_xs`. | PE import scan; Appendix A | Verified |
| The package layout lists `game_scarlett_master.exe`, but the file is not in the package folder. | `layout_..._xs.xml`, chunk 1001; directory listing | Verified |
| `gameos.xvd` is an XVD (`msft-xvd` at file offset 0x200, format version 3) with a high-entropy payload and no plaintext PE headers or D3D12 strings. Extracting `d3d12_xs.dll` from it would require keys, which this project does not do. | Entropy and string scan; Appendix A | Verified |
| Prior art runs the package DLLs against a PC D3D12 device and measured the Xbox ABI: 6 Xbox-specific static imports across the 16 DLLs, `D3D12XboxCreateDevice`, device slots 62/76, descriptor-heap start handles in RAX, root-signature flags `0x03000000`, and the BVH builder's Xbox DXIL shaders. | [PLATFORM_SURFACE.md](https://github.com/realdody/americas-recomp/blob/xbox-series/docs/xbox/PLATFORM_SURFACE.md), [platform README](https://github.com/realdody/americas-recomp/blob/xbox-series/tools/xbox/platform/README.md) | Verified by the prior art (running the package DLLs / static imports; section 7) |
| The translator and its GPU tests already define the shader contract: set 0 bindings 0–19, 144-byte push constants, heap/code base, FNV-1a descriptor hash table, buffer device addresses. | [design.md](design.md#translator-status-and-runtime-contract); `runtime/test/vk_texture_test.cpp` | Verified |

What this does **not** tell us: the full `d3d12_xs.dll` export list, the `D3D12XboxCreateDevice` params/flag
layout, the remaining Xbox slot deviations (presentation, command lists, queues, fences), the
root-signature-to-`s0..s31` mapping, the console descriptor layout, the present path, the raytracing path, and the
GPU VA window. Section 8.2 lists each with a way to resolve it.

The first run with the executable should be a **trace build**: every free function and COM method is a logging
stub that returns a failure, and the methods are filled in in the order the game calls them. Section 3.1 describes
the interception; section 8.2 turns the trace into a work list.

## 3. Architecture

### 3.1 Interception and COM

The current shim (`runtime/shims/d3d12_xs.cpp`) forwards only two free functions to the host `d3d12.dll`. That is
enough for the bundled middleware's `d3d12_xs` calls, not for the game. The layer adds:

- **Free functions.** Every `d3d12_xs.dll` export the executable imports. The import table is the source of truth.
  Prior art names the device entry point `D3D12XboxCreateDevice(adapter, params, riid, out)`; its 24-word params
  struct is only logged there, and the signature is taken from public documentation rather than a header. Treat the
  params/flag layout as **Open**, accept the call, log the words, and create the device. Do not assume
  `D3D12CreateDevice` is the console entry point.
- **COM objects.** `Device`, `CommandQueue`, `CommandAllocator`, `GraphicsCommandList`, `RootSignature`,
  `PipelineState`, `DescriptorHeap`, `Resource`, `Heap`, `Fence`, `QueryHeap`, `CommandSignature`,
  `PipelineLibrary`, and the raytracing/lifetime objects if used.
- **Vtable layout.** Use the Windows SDK's `d3d12.h` declarations for the public chain, but treat the vtable as
  **mostly public, not universally public**. Prior art measured device slots 1, 15, 18, 19, 27 and 38 in the
  desktop order, but also an extra texture-creation method at slot 76, a `CreateComputePipelineState` call at
  slot 62 where the PC interface it forwarded to presents `CreateStateObject` (different arity), and descriptor-heap
  start handles returned in RAX at heap slots 9/10. Implement the public chain plus those measured overrides
  explicitly; do not assume every slot. Device 11 adds `CreateSampler2`; command list 8 adds
  `OMSetFrontAndBackStencilRef` (public SDK).
- **Unknown IIDs and slots.** `QueryInterface` must log the IID and return `E_NOINTERFACE` if it is not in the
  public chain. If the game then fails, the IID and the call site identify the Xbox-specific interface to reverse.
  Do not invent a vtable for an unknown IID: a wrong slot is worse than a clean failure.
- **No host D3D12 or DXGI.** The PE loader currently lets unknown system DLLs bind to the host. `dxgi.dll` and
  `d3d12.dll` must be shimmed, not resolved from System32. Host DXGI would hand the game a desktop adapter and
  swapchain that cannot see our Vulkan device, and host D3D12 would create a second, unrelated device. This is an
  integration requirement, not an optimisation.
- **Trace first.** Each method wrapper records its name and arguments at a low rate (first call, then every N),
  with a `--trace-d3d12` switch. The trace must not quote game data; API names and counts only.

**Measured, not assumed:** the Xbox device ABI is not uniformly the desktop order. Slots 1/15/18/19/27/38 matched
in the prior art, while slots 62 and 76 and the heap start-handle return convention did not. Treat every slot as
unverified until traced; the public interface names and IIDs may still be usable, but the slot mapping is not
guaranteed.

### 3.2 Record and replay

D3D12 command lists are recorded, then executed later. Vulkan command buffers are also recorded, but the layer
must not simply translate each D3D12 call into a Vulkan call as it arrives, because:

- descriptor heaps can be updated between recording and execution;
- a command list can be reset and rerecorded;
- a bundle is recorded once and executed many times.

The design is therefore an **IR per command list**:

- `SetPipelineState`, root arguments, vertex/index buffers, viewport/scissor, render targets, descriptor-heap
  bindings and the other sticky state update a state block.
- Each draw/dispatch/copy/barrier appends a compact command with a snapshot of the state that affects it.
  GPU descriptor tables can refer to shader-visible heap entries at execution. CPU descriptors such as RTVs
  are consumed while recording, so their view contents go into the IR. An application can reuse the CPU
  descriptor slot after the call. See Microsoft's [descriptor overview][descriptor-overview].
- `ExecuteCommandLists` replays the IR into a fresh `VkCommandBuffer`, resolving descriptors, uploading dirty
  descriptor ranges and the hash table, and emitting barriers. Bundles are inlined at record time.
- `Reset` on a command list discards its IR; the allocator owns the memory. D3D12 already requires the application
  not to reset an allocator whose lists are in flight; the layer can rely on that and assert in debug builds.
- `Close` freezes the IR. `ExecuteBundle` copies the bundle's commands into the parent IR at the current state.
- A queue's `ExecuteCommandLists` submits in order with one timeline semaphore signal. `Wait`/`Signal` become
  timeline-semaphore waits/signals. `Present` queues after the frame's work.

The command-list surface maps to the IR as follows:

| Recorded call | IR / replay |
|---|---|
| `SetPipelineState` | state; bind the `VkPipeline` before the next draw/dispatch |
| `SetGraphicsRootSignature` / `SetComputeRootSignature` | state; select the pipeline layout and root map |
| `Set*Root32BitConstant(s)` / `Set*Root*View` / `Set*RootDescriptorTable` | update root args; build the push-constant block at the draw/dispatch |
| `SetDescriptorHeaps` | state; scan/populate the bindless arrays and hash table at execute |
| `IASetVertexBuffers` / `IASetIndexBuffer` / `IASetPrimitiveTopology` | state; bind arena buffer + offset, dynamic topology |
| `RSSetViewports` / `RSSetScissorRects` | state; dynamic viewport/scissor |
| `OMSetRenderTargets` / `OMSetBlendFactor` / `OMSetStencilRef` | state; rendering scope and dynamic state |
| `DrawInstanced` / `DrawIndexedInstanced` | command with a state snapshot |
| `Dispatch` / `DispatchMesh` | command with a state snapshot |
| `Copy*` / `Resolve*` / `Clear*` | copy/clear command |
| `ResourceBarrier` / `Barrier` | barrier command; state tracking |
| `BeginQuery` / `EndQuery` / `ResolveQueryData` | query commands |
| `ExecuteIndirect` | command; signature + argument buffer |
| `BeginRenderPass` / `EndRenderPass` | rendering scope |
| `ExecuteBundle` | inline the bundle's IR at the current state |

The IR is the place to add the trace, validation and "unsupported call" reporting. It also lets the layer replay a
frame deterministically from a capture, which is valuable for comparing against the game.

What is implemented (all of it testable without the game):

- **IR** (`runtime/src/command_list.h`): legacy barriers (transition, UAV, aliasing; split barriers record their end
  half), enhanced barriers (global, buffer, texture; the half with `SyncAfter = SPLIT` is dropped and the other is
  replayed as a full barrier), `ClearRenderTargetView` / `ClearDepthStencilView`, whole view or rectangles, and
  `CopyBufferRegion` / `CopyTextureRegion` / `CopyResource`.
  Views and rectangles are copied while recording, so the CPU descriptor slot can be rewritten afterwards.
- **Replay** (`runtime/src/vulkan_replay.h`): `prepare` resolves and checks a list without writing anything, `emit`
  writes it. Legacy states map through the section 4.3 table (`vulkan_access`), enhanced barriers through
  `vulkan_sync` / `vulkan_barrier_access` / `vulkan_layout` to `vkCmdPipelineBarrier2`. Whole-view clears are
  transfer clears inside a layout round trip; rectangle clears are `vkCmdClearAttachments` in a dynamic-rendering
  scope, with rectangles clipped to the mip. Copies become `vkCmdCopyBuffer`, `vkCmdCopyBufferToImage`,
  `vkCmdCopyImageToBuffer` and `vkCmdCopyImage`. A placed footprint's row pitch becomes `bufferRowLength`, and
  block-compressed extents stop at the mip edge (D3D12 pads footprints to whole blocks, Vulkan does not). Each
  range and buffer bound is checked before anything is written. A copy runs in `TRANSFER_SRC`/`TRANSFER_DST`, or
  in place in `GENERAL` when the texture is in `COMMON`: that is D3D12's implicit promotion to a copy state.
  2D only: 3D boxes and depth slices are rejected as unsupported.
- **Layout tracking** (`LayoutTracker`): the current Vulkan layout of every image subresource, so barriers start
  from the real layout instead of the application's `before`. This is how implicit promotion and decay work: a
  texture in `COMMON` sits in `GENERAL`, which every promotable access can use, so a later barrier naming the
  promoted state still starts from `GENERAL`, and decay changes no layout. Buffers have no layout, and
  simultaneous-access textures stay in `GENERAL` for life. With `separateDepthStencilLayouts` the depth and stencil
  planes are tracked apart, so a depth-only clear or a one-plane transition leaves the other plane where it is.
- **Execution** (`runtime/src/vulkan_backend.h`): `VulkanContext` (registered resources, the tracker, a view cache),
  `VulkanQueue` (`ExecuteCommandLists` prepares every list before writing any, so a bad list submits nothing and
  leaves the tracker as it was; `Signal`; `Wait`, which is attached to the next submission because a Vulkan wait
  only holds back its own batch) and `VulkanFence` (a timeline semaphore).
- **COM** (`runtime/shims/d3d12_com.cpp`): `ID3D12GraphicsCommandList7`, `ID3D12CommandQueue`,
  `ID3D12CommandAllocator` and `ID3D12Fence1` (with `SetEventOnCompletion` events). Every method they do not
  implement is a stub generated at build time from the SDK's `d3d12.h` by `tools/gen_d3d12_stubs.py`. The stub
  reports its name once, so the vtables follow the SDK order exactly and nothing derived from the SDK is checked
  in. Unknown IIDs are logged and refused. Resources are identified by their `ID3D12Resource` pointer, and
  RTV/DSV CPU handles point at a `TargetView`.
- **Device** (`runtime/shims/d3d12_vulkan_device.cpp`): an `ID3D12Device11` (PC ABI, 80 slots) that the Xbox-ABI
  proxy (`d3d12_xs_device.cpp`) forwards to, so slot 14's heap-handle conversion and slot 76's resource creation
  work unchanged. It creates the objects above, plus:
  - descriptor heaps whose slots are 32 bytes (`GetDescriptorHandleIncrementSize`; the console's increments are
    question 4). RTV/DSV slots hold a `TargetView`; shader-visible heaps return a GPU handle, but nothing reads it
    yet;
  - `CreateRenderTargetView` / `CreateDepthStencilView` for whole resources, `TEXTURE2D` and `TEXTURE2DARRAY`;
  - committed buffers (upload and readback heaps mapped persistently and zeroed) and 2D textures in the common
    render-target, depth and texture formats (typeless families become mutable-format images);
  - `CheckFeatureSupport` for feature levels (up to 12_1) and architecture. Every other query is logged and
    refused until the trace shows what the game asks (question 11);
  - `GetCopyableFootprints` for buffers and 2D textures, including BC1–7 and the two-plane depth-stencil formats.
    The test compares every output with the PC runtime's for 13 descriptions; that comparison found that planar
    depth-stencil rows are padded to 512 bytes, not 256.

  BC1–7 textures need `textureCompressionBC`, which the device enables when the GPU has it.

  New textures are not touched at creation. They are tracked as `UNDEFINED`, and the first command that uses a
  part of one zero-fills it (D3D12 committed resources start zeroed; `CREATE_NOT_ZEROED` skips this) and moves it
  to the layout that command expects, so creation needs no submission. Block-compressed textures are not
  zero-filled, because Vulkan cannot clear them; they start undefined until the application uploads them. Each
  submission ends with a barrier to the host, so a readback buffer holds the GPU's writes once the fence wait
  returns. Children hold a reference to the proxy, not
  the inner device, so `GetDevice` always returns the Xbox-ABI object. All D3D12 queues share one graphics
  `VkQueue` behind a submit lock (question 12).

Known gaps: shader-resource, UAV and constant-buffer views, samplers, root signatures, PSOs, draws, 3D and array
copies through footprints with depth, placed and reserved resources, `GetGPUVirtualAddress` (the shared buffer arena), MSAA, 1D/3D textures and the swapchain;
default-heap buffers are not zero-filled yet; a fence cannot be signalled to a lower value (Vulkan timelines only
increase; such a signal is reported and dropped); enhanced-barrier access bits are not filtered against the
sync scope, so a combination D3D12 accepts but Vulkan does not would show up as a validation error.

Build `runtime/` (Vulkan SDK and Python 3 on `PATH`) and run `ctest --test-dir <build> --output-on-failure` for the
CPU tests:
- `command_list_test` covers the recording rules.
- `d3d12_state_map_test` covers both state tables, subresource and barrier-range decoding, and layout tracking. On
  Windows it also checks every D3D12 constant the IR copies against the SDK's `d3d12.h`.

The GPU tests read their results back. With `LEONIDA_VK_VALIDATION=1` they run under `VK_LAYER_KHRONOS_validation`,
and any validation error fails them.
- `vk_command_replay_test`, part A: only Vulkan 1.0 functions and no tracker.
- `vk_command_replay_test`, part B: rectangle clears, a depth clear while the stencil plane is read-only, a promoted
  texture, and enhanced barriers including a discard.
- `d3d12_com_test`: drives the COM objects through their vtables. It also checks that a queue `Wait` on a fence
  the CPU has not signalled holds the GPU back.
- `d3d12_vulkan_device_test`: the Vulkan device behind the Xbox-ABI proxy, called the way the game will call it.
  It creates heaps through slot 14 and reads their start handles in RAX, creates a render target through slot 76
  with the Xbox layout word, then creates views, records clears and barriers, executes and waits on a fence.
  Results come back through D3D12 itself (`CopyTextureRegion` into a readback heap, then `Map`): the render
  target, the depth and stencil planes, and a never-written texture (which must be zero). A second list uploads a
  pattern and copies it on: whole into a texture created in `COMMON`, a boxed part into a smaller texture, a BC1
  round trip and a buffer region. `GetCopyableFootprints` is compared with the PC runtime's, and a short run
  repeats on a standalone device, the path `LEONIDA_D3D12=vulkan` takes.

**Open:** whether the game uses command-list flags, bundles, or `ExecuteCommandLists` with many lists at once. All
are supported by the IR; the trace will show which are hot.

[descriptor-overview]: https://learn.microsoft.com/en-us/windows/win32/direct3d12/descriptors-overview

### 3.3 One GPU address space

The shader contract is the constraint that shapes everything else. The translator computes:

```
host_address = heap_base + console_va
```

where `heap_base` is push-constant dwords 32–33 and `console_va` is an Xbox GPU virtual address the game writes
into a descriptor, a root argument, or a buffer. This applies to buffer, descriptor-table and shader-code
addresses. Image descriptors (T#) are hashed by the translator, not dereferenced, so an image's base can be a
separate key range (below). Some shaders also force the high dword of a descriptor-table pointer to zero, so
**shader-visible descriptor heaps must live below 4 GB** in this address space.

The layer therefore owns the console VAs. It returns them from `ID3D12Resource::GetGPUVirtualAddress` and from
`GetGPUDescriptorHandleForHeapStart`; the game only sees values the layer chose. The design is:

| Piece | Design |
|---|---|
| Address space | Two ranges. A **BDA arena** holds buffers, descriptor heaps and shader code; its device address is `heap_base`. A separate **image key range** supplies unique console VAs for images; those keys are never added to `heap_base`. The arena's `VkBuffer` is created with all buffer usages the game can bind (uniform, storage, vertex, index, indirect, transfer) plus `SHADER_DEVICE_ADDRESS`, and its memory with `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`. |
| Buffers | A buffer resource is a range in the arena's single `VkBuffer`: offset = its console VA, size = the resource size. Buffer commands bind the arena buffer with that offset, so its device address is exactly `arena_bda + offset`; no per-resource `VkBuffer` object is needed. |
| Images | Each image gets a unique console VA key and lives in its own `VkDeviceMemory` (or a suballocated device-local image heap) with optimal tiling. Shader image access goes through the T#/hash table, so nothing dereferences the key. Keeping images out of the arena avoids forcing optimal-tiled images into the arena's memory type and keeps the arena sized by buffers, not textures. |
| Descriptor heaps | Shader-visible CBV/SRV/UAV and sampler heaps are allocated in the low 4 GB of the arena. RTV/DSV heaps are host-side only. |
| Shader code | Each FSSM blob is uploaded to the arena; its console VA is the code base. |
| Upload/readback | `D3D12_HEAP_TYPE_UPLOAD`/`READBACK` resources are separate host-visible buffers outside the arena. Their console VAs are only used by copy APIs; if the game binds one as a vertex/index buffer or reads it in a shader, the layer biases the view/descriptor by `-heap_base` so the shader's addition lands on the host BDA (**Assumption**, to verify). |

Non-arena buffers (upload/readback) can still be addressed by the shader without putting them in the arena: pass
`console_va = host_bda - heap_base` (mod 2^64) in root descriptors and vertex/index/indirect views, so the shader's
`heap_base + console_va` equals the host BDA. For descriptors, patch the V# base field the same way when the layer
copies or scans them. This relies on the game not using the raw VA for anything except shader addressing
(**Assumption**); if it does, the resource must move into the arena instead.

Why not give each buffer its own BDA and set `heap_base = 0`? Because many shaders truncate descriptor-table
pointers to 32 bits, and host BDAs are not guaranteed to be below 4 GB. The arena keeps console VAs low and stable.

Why one arena rather than several? Because the shader gets exactly one `heap_base` for buffer, descriptor-table and
code accesses. A draw can reference a descriptor table in one heap and a vertex buffer in another; both must
satisfy the same `heap_base + va` equation. Several non-contiguous VkBuffers cannot provide that equation.

Image keys must fit the T# base field. Keep them below 1 TB (a conservative 40-bit assumption; confirm against the
T# layout the decoder uses) and above the arena. **Open:** the exact image-key range and whether any shader ever
uses an image VA as a raw pointer.

**Memory type policy.** The first boot needs correctness, not bandwidth:

- M0–M3: allocate the arena as `HOST_VISIBLE | HOST_COHERENT` (preferably also `DEVICE_LOCAL` when large BAR is
  available). The CPU can then write descriptor heaps, shader blobs and uploaded buffers directly. Images are
  already device-local, so rendering still uses fast memory for textures. This is slow for dynamic buffers but
  removes staging and host/console address translation from the bring-up path.
- M4+: allocate the arena `DEVICE_LOCAL` and stage CPU writes (descriptor heaps, shader blobs, buffer uploads)
  through a host-visible staging buffer, copied before the commands that use them. The hash table is not in the
  arena; it is a normal storage buffer, so it can stay host-visible and be updated directly.

**Capacity.** The arena is sized by buffers, descriptor heaps and shader code, not by textures. The allocator
tracks the high-water mark; start with a configurable capacity (for example 4 GiB for the loading/menu path) and
fail with a clear message when a console VA would fall outside it. Growing means creating a new arena, copying,
and updating `heap_base` for all subsequent commands; do it at a frame boundary, never with work in flight.

**Open:** the game's real buffer VA range and peak working set are unknown until it runs. `maxBufferSize` (with
`VK_KHR_maintenance4`) and `maxMemoryAllocationSize` bound the arena; a large open-world buffer working set may not
fit on smaller PC GPUs. This is a product requirement, not something the layer can hide.

**Reserved resources.** `CreateReservedResource` plus tile mappings is a sparse resource. Vulkan sparse binding
works on images and buffers, but the layer's arena model assumes all buffer memory is in one BDA range; a sparse
buffer would need its bound memory at the console VA. For the first boot, reserve the full size as ordinary arena
memory and log the tile calls; implement sparse binding only if the game uses it for streaming. Sparse images can
use their own VkDeviceMemory and are easier. **Open.**

### 3.4 Descriptor heaps and the bindless model

The game can use D3D12 descriptors in three ways, and the layer must handle all of them:

1. **Root descriptor tables.** `SetGraphicsRootDescriptorTable` supplies a GPU handle; the shader receives the
   table's console VA in its user data and loads T#/S# structures from it.
2. **Direct heap indexing.** The root signature has `D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED`
   and/or `..._SAMPLER_HEAP_DIRECTLY_INDEXED`, and the shader indexes `ResourceDescriptorHeap` /
   `SamplerDescriptorHeap` directly. The console driver supplies the bound heap base and stride; the shader loads
   the descriptor at `base + index * stride`. **Open:** where the heap base/stride arrive (user or system SGPRs; if
   system, the translator needs a new runtime-supplied constant).
3. **Game-built descriptor memory.** The game writes raw T#/S# bytes into a heap itself, without calling
   `CreateShaderResourceView`. The shaders then read those bytes exactly as in case 1.

The layer's descriptor model:

- A `DescriptorHeap` has a host shadow and, if shader-visible, an arena range. CPU handles point into the shadow;
  GPU handles are arena VAs. `Create*View`, `CopyDescriptors` and `CopyDescriptorsSimple` write the shadow and mark
  the range dirty. The execute-time scan below is the correctness path, so a game that writes raw descriptor bytes
  through a CPU handle still works; dirty tracking is only an optimisation.
- **Heap start-handle ABI.** Prior art measured `GetCPUDescriptorHandleForHeapStart` and
  `GetGPUDescriptorHandleForHeapStart` (heap slots 9/10) returning the 8-byte handle in RAX on the Xbox build,
  while the PC call returns it through a hidden pointer in RDX. A generic sret/IDL thunk must special-case these
  two slots or the handle is garbage. This is a method-level ABI quirk; the heap object is still the public
  `ID3D12DescriptorHeap` interface.
- The bytes written for an image/sampler descriptor are **GFX10 T#/S# structures**, the same structures the
  translator reads. The layer writes them itself in `Create*View`; this does not depend on what the console driver
  wrote. If the game builds descriptors in memory itself, it must use the same format (the shader reads raw T#/S#),
  which is the assumption to verify. If it uses an opaque format, the layer must translate at bind time.
- At execution, for each bound table (or the whole bound heap when direct indexing is enabled), the layer walks
  the declared descriptor range with the heap's stride, parses each T#/S#, and ensures a bindless entry exists:
  - image T# → `VkImageView` (format, view type, mip range, array layers) → bindless image index;
  - sampler S# → `VkSampler` (filter, address modes, LOD, compare, anisotropy) → bindless sampler index;
  - buffer V# / CBV → no Vulkan view; the descriptor already contains a console VA the shader dereferences.
- Unbounded descriptor ranges (`NumDescriptors = UINT_MAX`) have no end. Scan from the table handle to the end of
  the bound heap, and cache parsed descriptors by key so unchanged slots cost nothing. **Open:** whether the game
  uses unbounded tables; the trace will show the root-signature ranges.
- The heap stride is the value the layer returns from `GetDescriptorHandleIncrementSize`. Prior art forwards device
  slot 15 (`GetDescriptorHandleIncrementSize`) unchanged, but the actual increment values are still **Open**. Choose
  32 bytes for CBV/SRV/UAV (the T# size) and 32 bytes for samplers (16 bytes of S# plus padding) pending the console
  value. The game may hardcode the console increments or infer them from shader address arithmetic; the shader's
  shift amount gives the stride it expects, and the layer must match it.
- `CopyDescriptors` copies raw bytes. If the descriptor was produced by our `Create*View`, the copy is a valid
  T#/S#; if it came from game memory, it is still valid raw data. No re-encoding is needed.
- Handle validation: a GPU descriptor handle is an arena VA; the heap registry maps the containing range to a
  heap. A table handle outside the bound heaps is a validation error; log it and resolve to fallback index 0
  rather than reading arbitrary memory.
- RTV/DSV creation builds a `VkImageView` and stores it in the host-side heap; `OMSetRenderTargets` resolves those
  handles to views for the rendering scope. RTV/DSV descriptors are never shader-visible.
- View creation maps `D3D12_SRV_DIMENSION`/`D3D12_UAV_DIMENSION` to `VkImageViewType`, the mip and layer ranges,
  sample count, and `D3D12_SHADER_COMPONENT_MAPPING` to `VkComponentMapping`. `ResourceMinLODClamp` needs
  `VK_EXT_image_view_min_lod`; log if the device lacks it instead of silently ignoring the clamp.
- Set 0 is a fixed bindless layout. The GPU tests use 4096 sampled 2D images, 1024 samplers, 1024 2D-array
  images, 256 1D, 256 3D, 1024 storage images and 256 per storage dimension/class; start there and grow by
  recreating the set at a frame boundary. Use `UPDATE_AFTER_BIND` and `PARTIALLY_BOUND` so a new view does not
  require rewriting every descriptor; without those features, rebuild the set when it changes.
- The command buffer binds set 0 once; descriptor-heap changes only update the arrays and the hash table.

**Descriptor lifetime.** A bindless index may be referenced by in-flight command buffers after the resource is
destroyed. Views/samplers are retired on a fence-value queue; the hash-table entry can stay (a stale key is only a
miss for a new descriptor, not a correctness problem) until the slot is reused. Reusing a slot for a new resource
updates the index in place; the key match makes the lookup correct.

### 3.5 Root signature to user SGPRs

This is the least verifiable part of the layer. D3D12 gives the application a root signature; the console driver
turns it into the user SGPRs (`s0..s31`) that the shader reads. The translator already copies those 32 dwords from
push constants into `s0..s31`. The layer must produce them.

The default mapping (implemented by `rt::rootsig::pack` in `runtime/src/root_signature.{h,cpp}`) follows root
parameter order:

| Root parameter | User-data dwords |
|---|---|
| `D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS` | one dword per constant |
| `D3D12_ROOT_PARAMETER_TYPE_CBV` / `_SRV` / `_UAV` | two dwords: the 64-bit console VA passed to `Set*Root*View` |
| `D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE` | two dwords: the table's console VA (a `D3D12_GPU_DESCRIPTOR_HANDLE`) |
| static sampler | none; registered in the hash table at root-signature creation |

Before the draw/dispatch, the layer copies the current root arguments into a 32-dword block in that order and
uploads it as push constants. Missing parameters are zero.

Spill policy: a signature needing more than 32 user SGPRs, or more than the 64-dword D3D12 cost limit, is
refused loudly (`pack` returns the offending parameter) instead of being truncated. The console spills through
`__XB_GetRootSignatureSpillPointer`; until that layout is measured, refusing is the only correct behaviour.

Reconciliation: `tools/root_sig_contract.py` checks a candidate root signature against every shader's
`user_sgpr_reads` from `spirv/index.json` (const32 vs constant dwords, 64-bit VAs vs view/table pairs) and fails
the build on the first contradiction. Until the console mapping in §3.5 is confirmed by a PSO, this check — not
a rendered frame — is the acceptance gate for the mapping above.

**Root-signature flag quirk.** Prior art measured Xbox root-signature descriptors carrying flags `0x03000000`; the
PC serializer rejects them with `E_INVALIDARG` because its flags live in the low 16 bits. Clear the high byte on a
private copy before calling the host serializer/deserializer; never modify the game's descriptor. The meaning of the
bits is unknown. The v1.0 `D3D12SerializeRootSignature` path keeps flags at offset +32, and a descriptor with
`0x3000001` also fails on the PC runtime.

Things that can invalidate this simple model, all **Open**:

- **Shader visibility.** The layer starts with one global block in root-parameter order and treats visibility as
  access control, not as a per-stage reordering. If the console driver instead omits invisible parameters per
  stage, a VS and a PS would need different values at the same push-constant offsets, which Vulkan cannot express
  with one block; that would force per-stage blocks or a per-stage indirection. The shader read sets will show
  which model is right.
- **Root signature larger than 32 dwords.** D3D12 allows up to 64 dwords. The console has 32 user SGPRs, so a
  driver must spill the rest to memory, probably behind a pointer in the user data. The current shader contract has
  no representation for that. The likely answer is that this game keeps its root signatures small (bindless
  descriptors, a few constants), but it must be measured. If a large root signature appears, the spill layout has
  to be reverse-engineered from the shader's loads.
- **Direct heap indexing.** The bound heap base and stride are not root parameters. They come from a
  driver-managed slot, possibly a system SGPR rather than user data. The layer needs to know which register the
  shader reads for them; if it is a system SGPR, the translator must expose it as a runtime-supplied value. The
  root signature flag tells us direct indexing is enabled; the shader's address arithmetic identifies the base.
- **Static samplers.** D3D12 static samplers do not consume root parameters. The console driver may place their
  S# values in a fixed table. The layer registers every static sampler's S# in the hash table, so a shader that
  loads it by value works; a shader that expects a driver table needs that table's location (**Open**).
- **Descriptor-table high-dword truncation.** If a shader forces the high dword of a table pointer to zero, the
  table must be below 4 GB; the layer allocates all shader-visible heaps there. If a shader forces a resource VA's
  high dword to zero, the resource must also be low; that would constrain the arena layout. The disassembly shows
  which addresses are truncated; collect them per shader.

**What the console's own D3D12 libraries show** (names only, read statically from a user's console system-file
dump; nothing from those files is in this repository):

- Two root-signature ABIs exist. The driver's validation messages name a flag
  `D3D12XBOX_ROOT_SIGNATURE_FLAG_FORCE_MEMORY_BASED_ABI`, and require it for ray tracing (XDXR) and for pipelines
  with an amplification shader. So the default is the register-based ABI this section describes, and a root
  signature can opt into a memory-based one. The shader compiler has the matching switch,
  `XBOX_FORCE_MEMORY_BASED_ABI`.
- The ABI is decided when the shader is compiled, not by the driver at bind time. The shader compiler has the
  intrinsics `__XB_GetShaderUserData` and `__XB_GetRootSignatureSpillPointer`. That answers the "larger than
  32 dwords" question above: parameters that don't fit in the user SGPRs spill to memory, and the shader reads a
  spill pointer from its user data. Because the game ships precompiled shaders, each shader's own user-SGPR reads
  (and any loads through a spill pointer) are the authoritative map; the metadata below is how to recover it.
- `d3d12_xs.dll` is a thin front end (about 50 KB). It loads the user-mode driver, `xgs12_10.dll`, which implements
  the device and command lists and exports `UMD*` entry points. `xg_xs.dll` exports the texture layout functions
  (`XGComputeTexture2DLayout`, `XGComputeOptimalSwizzleMode`, ...), the reference for untiling textures.
- The driver keeps the standard D3D12 root-signature cost limit: 64 dwords, with descriptor tables costing 1 and
  root views 2.

**Required build metadata.** To solve the mapping without guessing, `spirv/index.json` should gain, per shader:

- the set of user-SGPR dwords read before first write, and whether each is used as a 32-bit constant or a 64-bit
  pointer;
- the descriptor stride(s) implied by descriptor address arithmetic;
- whether the shader uses the code base (`s_getpc_b64`) and, if so, in which stage;
- the literal S# values it builds from immediates;
- the input signature (semantic name/index, register, mask, component type), not just the register numbers;
- the FSSM content hash (the PSO lookup key, section 3.6);
- the descriptor-table / heap-direct-indexing mode the shader was compiled for.

The runtime can then reconcile the root signature with the shader's reads and fail loudly on a mismatch instead of
rendering silently wrong.

`tools/shader_meta.py` (invoked by `tools/rdna2spirv.py` and verified by `tests/test_shader_meta.py`) now emits these
fields into `spirv/index.json`: `user_sgpr_reads` (`const32`, `ptr64_lo`/`ptr64_hi`, `ptr32_lo`, `vsharp`, `tsharp`,
`ssharp`), `descriptor_strides`, `uses_code_base`, `literal_samplers`, `input_signature` (ISG1 semantic name/index,
register, masks, system value, component type), `fssm_sha256`, and `descriptor_access` (DXBC `SFI0` direct-indexing
bits 25/26 plus table-load and high-dword-truncation flags).

### 3.6 Shader identity, PSOs and feature selection

A D3D12 PSO carries DXBC containers, not SPIR-V. The layer identifies a shader by hashing the **FSSM payload** (the
native RDNA2 program inside the container) and looking it up in a build-time table keyed by `(stage, hash)`. The
build emits that hash for every translation; the runtime extracts the FSSM chunk from the DXBC container the same
way `tools/fssm_unpack.py` does. The current `spirv/index.json` is keyed by carve name, which the runtime may not
have; add the content hash to the index.

Pipeline creation:

- `D3D12_GRAPHICS_PIPELINE_STATE_DESC` and the `D3D12_PIPELINE_STATE_STREAM_DESC` subobjects are both supported.
  Parse the stream subobjects in order; unknown types are a hard error with the type number logged.
- Build `VkPipelineShaderStageCreateInfo` from the translated module and the index entry. Mesh shaders use
  `VK_EXT_mesh_shader` (`AS`/`MS` subobjects → task/mesh stages); geometry uses the geometry stage; compute uses a
  compute pipeline.
- The index's `max_vertices`/`max_primitives` already size the SPIR-V mesh output arrays. Vulkan has no
  per-pipeline mesh output limit: check them against
  `VkPhysicalDeviceMeshShaderPropertiesEXT::maxMeshOutputVertices`/`maxMeshOutputPrimitives` and fail at pipeline
  creation if the device is smaller.
- Vertex input comes from the PSO input layout plus the shader's ISG1 signature. The translator's rule is
  **location = D3D input register**, not the semantic's HLSL slot. Match the layout element's semantic name/index
  to the ISG1 element to get the register, then create a `VkVertexInputAttributeDescription` with that location
  and the DXGI format. Resolve `D3D12_APPEND_ALIGNED_ELEMENT` against the preceding element and honour
  `InputSlotClass`/`InstanceDataStepRate`. The index entry needs the semantic map (section 3.5).
- Enable the features the index entry lists: barycentrics, fragment interlock, mesh, geometry, Float64, Int64,
  buffer device address, storage image without format, maintenance8.
- Request `requiredSubgroupSize = 64` for wave64 shaders when `VK_EXT_subgroup_size_control` offers it (AMD).
  Otherwise use the translator's 32-wide approximation and say so in the log.
- Pipeline caches: use a `VkPipelineCache` keyed by driver/pipeline-cache UUID; the `pipelines` build stage already
  produces one. D3D12 `CachedPSO` blobs are driver-specific and are ignored (the PSO is rebuilt from its
  descriptors). `ID3D12PipelineLibrary` is an in-memory map plus the Vulkan cache.

**Open:** whether the game uses `D3D12_PIPELINE_STATE_STREAM_DESC` for mesh/RT and which subobject types appear.
The trace will enumerate them.

## 4. Object model mapping

| D3D12 / Xbox object | Vulkan / layer object | Notes |
|---|---|---|
| `ID3D12Device` … `ID3D12Device11` | one `Device` object with the public chain plus measured Xbox overrides | The shipped BVH builder names Device11; slots 1/15/18/19/27/38 matched desktop in prior art, slot 62 is called as `CreateComputePipelineState`, slot 76 is Xbox texture creation, and heap slots 9/10 return in RAX. Unknown Xbox IIDs/slots are logged and refused. |
| `ID3D12CommandQueue` | `Queue` over one or more `VkQueue`s | Start with one graphics+compute+transfer queue; add dedicated queues when the game uses async compute. `PresentX` is Xbox-specific and not in the public vtable (**Open**). |
| `ID3D12CommandAllocator` | IR arena + `VkCommandPool` | Reset only when the previous submission's fence is complete. |
| `ID3D12GraphicsCommandList` … `8` | `CommandList` with IR | Replay at `ExecuteCommandLists`; bundles inlined. Supports render passes, mesh dispatch, enhanced barriers, front/back stencil ref. |
| `ID3D12RootSignature` | `RootSignature` + `VkPipelineLayout` | Deserialize the blob with the host `D3D12CreateVersionedRootSignatureDeserializer` when possible; clear the measured `0x03000000` flag byte on a copy first; fall back to a local parser. Build the user-SGPR map and static samplers. |
| `ID3D12PipelineState` | `VkPipeline` + a `PipelineInfo` | Shader hash → SPIR-V + index entry; PSO state → Vulkan state; dynamic state where possible. The BVH builder's compute PSOs arrive at device slot 62 with a plain `D3D12_COMPUTE_PIPELINE_STATE_DESC`. |
| `ID3D12DescriptorHeap` | `DescriptorHeap` (shadow + optional arena range) | CBV/SRV/UAV and sampler heaps can be shader-visible; RTV/DSV are host-only. Start handles use the measured RAX return at heap slots 9/10. Increments are layer-chosen (32/32) pending the console value. |
| `ID3D12Resource` | `Resource` (a range in the arena buffer; a separate `VkImage` for textures) | Committed and placed; reserved is stubbed as linear for now. Xbox textures may arrive through device slot 76 rather than `CreateCommittedResource` (section 4.2). `GetGPUVirtualAddress` returns a console VA (a buffer offset or an image key). |
| `ID3D12Heap` | `Heap` (a reserved VA range) | Buffer ranges are backed by the arena; images get their own `VkDeviceMemory`. Residency calls are no-ops. True buffer/image aliasing inside a heap is **Open**. |
| `ID3D12Fence` / `ID3D12Fence1` | `Fence` over a timeline `VkSemaphore` | `GetCompletedValue` = counter value; `SetEventOnCompletion` = wait thread + Win32 event; CPU `Signal` = host signal. |
| Resource barriers (legacy) | `vkCmdPipelineBarrier` | Per-subresource state tracking; `BEGIN_ONLY`/`END_ONLY` become conservative full barriers at the begin point. |
| Enhanced barriers (`ID3D12GraphicsCommandList7::Barrier`) | `vkCmdPipelineBarrier2` | Direct mapping of sync/access/layout; prefer it when the game uses it. |
| `IDXGISwapChain` / `IDXGIFactory` | `Swapchain` over `VkSwapchainKHR` + Win32 surface | Shim DXGI; do not bind host DXGI. `PresentX` planes need a compose path (**Open**). |
| `ID3D12QueryHeap` | `VkQueryPool` | Occlusion, binary occlusion, pipeline statistics, timestamps. Binary results clamped to 0/1. |
| `ID3D12CommandSignature` | `IndirectLayout` | Simple draw/dispatch signatures map to indirect calls; root-constant/vertex/index/descriptor changes need expansion. |
| `ID3D12PipelineLibrary` | map + `VkPipelineCache` | D3D12 cached blobs are ignored. |
| `ID3D12StateObject` (ray tracing) | likely unused | RT work includes compute traversal (`image_bvh*`) plus `XboxBvhBuilder`; a DXR state-object path is stubbed until the trace shows it. |
| `ID3D12ShaderCacheSession` | `VkPipelineCache` | Stub with logging. |
| DRED, protected sessions, meta commands, shared handles | stubs | Not needed for single-player boot; return a clean failure and log. |

### 4.1 Device and feature queries

`CheckFeatureSupport` is a boot-critical diagnostic. Answer from the Vulkan device, not from a wish list:

- `D3D12_FEATURE_ARCHITECTURE`: report the host GPU's UMA property. The console is UMA; a discrete PC GPU is not.
  Reporting the truth may change the game's upload strategy; reporting a lie can make it map memory it cannot map.
  Start truthful and log the game's reaction.
- `D3D12_FEATURE_LEVELS`: report the highest feature level the Vulkan device can satisfy (12_2 when mesh/RT
  features are present).
- `D3D12_OPTIONS` families: report the tier the Vulkan feature set can honour (resource binding tier 3 when
  descriptor indexing and update-after-bind are available), plus wave ops, 16-bit, barycentrics, render passes,
  VRS, mesh shader tier, and RT tier 1.1 only if the layer can honour them. The game may branch on these; a false
  "supported" is worse than a false "unsupported" at boot.
- `D3D12_FEATURE_FORMAT_SUPPORT`: map to `vkGetPhysicalDeviceFormatProperties`.
- `D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT` (`D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT`): report the
  arena's limits, not the console's.
- `GetResourceAllocationInfo` and `GetCopyableFootprints`: implement D3D12's rules locally; the game uses them to
  suballocate and to lay out uploads. They are deterministic and public. Validate against a host D3D12 device in
  tests if one is available; do not require one at runtime.

### 4.2 Resources, heaps, placed and reserved

- **Xbox texture creation is a separate device method at slot 76**, not `CreateCommittedResource`. Prior art
  reports its shape as heap properties / heap flags / description / initial state / clear value / iid / output,
  forwarded on the PC side to `CreateCommittedResource` slot 27. The description is 56 bytes plus extension words,
  and its layout field is `0x101` (tiled) or `0x1` (row-major buffer). The PC proxy replaces the layout with a
  driver-chosen one; our layer must recognise slot 76 and `0x101` explicitly and must not pass `0x101` to the public
  layout enum. The FidelityFX upscaler creates 10 default-heap resources through this slot, so it is not optional
  for middleware. The actual console tiled data layout and the detile path remain **Open**.
- `CreateCommittedResource`: allocate an aligned range in the arena for a buffer, or a separate `VkDeviceMemory`
  for an image; record the initial state. Map `D3D12_HEAP_TYPE_DEFAULT` to device-local memory, `UPLOAD`/`READBACK`
  to separate host-visible buffers. This remains the path for buffers and for any PC-style committed resource.
- `CreateHeap`: reserve a VA range with the requested size/alignment. `CreatePlacedResource`: assign the resource
  the range at `heap_va + offset`; a buffer uses that range in the arena, an image uses the offset as a key and
  binds its own memory. Vulkan image alignment can exceed D3D12's 64 KiB; because images do not share the arena,
  this mostly affects true aliasing, which is **Open** until tested on the target GPU.
- `CreateReservedResource`: reserve the full size linearly; tile mapping calls are logged, not emulated, for the
  first boot.
- `D3D12_RESOURCE_DESC::Layout`: the public `ROW_MAJOR` and `64KB_STANDARD_SWIZZLE` values plus the Xbox `0x101`
  tiled value from slot 76. Row-major textures need `VK_IMAGE_TILING_LINEAR` (not usable for RT/DS) or a staging
  copy; `64KB_STANDARD_SWIZZLE` and `0x101` have no direct Vulkan equivalent. If the game uploads console-tiled data,
  the layer needs a detile step or a matching tiled image. **Open**; the ABI value is known, the data layout is not.
- Formats: a `DXGI_FORMAT` → `VkFormat` table, including sRGB and typeless variants. Typeless resources get a
  concrete image format plus `VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT` and a `VK_KHR_image_format_list` when views cast
  within the family; unsupported casts are logged and fall back to the resource format. Depth/stencil, BCn and
  packed formats map directly; any format the device rejects is a hard error with the DXGI value.
- `Map`/`Unmap`: upload/readback buffers map their host-visible memory; default-heap resources return
  `E_INVALIDARG` as D3D12 does. `WriteToSubresource`/`ReadFromSubresource` use a staging copy at the next submit.
- Resource flags map to Vulkan usage: `ALLOW_RENDER_TARGET` → `COLOR_ATTACHMENT`, `ALLOW_DEPTH_STENCIL` →
  `DEPTH_STENCIL_ATTACHMENT`, `ALLOW_UNORDERED_ACCESS` → `STORAGE`, `ALLOW_SIMULTANEOUS_ACCESS` → concurrent
  sharing, `DENY_*` → omit the corresponding usage. Buffers get the union of usages their states and flags imply;
  images copy dimension, format, mip levels, array layers and sample count.
- Copies: `CopyBufferRegion` → `vkCmdCopyBuffer` with arena offsets; `CopyTextureRegion` →
  `vkCmdCopyBufferToImage`/`vkCmdCopyImageToBuffer`/`vkCmdCopyImage`, using the `D3D12_PLACED_SUBRESOURCE_FOOTPRINT`
  row pitch and height; `CopyResource` → a whole-subresource copy; `ResolveSubresource` → `vkCmdResolveImage`.
  Partial resolve regions need a shader or a region copy.
- Clears: a full-subresource clear maps to `vkCmdClearColorImage`/`vkCmdClearDepthStencilImage` (UAV buffers to
  `vkCmdFillBuffer`); a rect clear uses `vkCmdClearAttachments` inside the rendering scope. UAV image clears use
  `vkCmdClearColorImage` in `GENERAL`; non-32-bit buffer patterns need a compute clear.
- `D3D12XBOX_RESOURCE_FLAG_FORCE_TEXTURE_COMPATIBILITY` (public release note) and DCC: ignore the flag, let the
  Vulkan driver choose compression, and fail loudly if the game aliases a buffer over a compressed image or reads
  raw DCC metadata.

### 4.3 Barriers and layouts

Track a current state/layout per subresource. D3D12 states are bitmasks; a combined state takes the union of the
mapped stages/accesses, and if no single layout serves all uses, use `GENERAL` and log it. Legacy
`D3D12_RESOURCE_STATE_*` maps to a (stage, access, layout) triple:

| D3D12 state | Vulkan layout | Stage / access |
|---|---|---|
| `COMMON` | `GENERAL` for images, no layout for buffers | Promote on first use |
| `RENDER_TARGET` | `COLOR_ATTACHMENT_OPTIMAL` | `COLOR_ATTACHMENT_OUTPUT` / write |
| `DEPTH_WRITE` / `DEPTH_READ` | `DEPTH_STENCIL_ATTACHMENT_OPTIMAL` / `DEPTH_STENCIL_READ_ONLY_OPTIMAL` | early/late tests |
| `PIXEL_SHADER_RESOURCE` | `SHADER_READ_ONLY_OPTIMAL` | fragment shader / read |
| `NON_PIXEL_SHADER_RESOURCE` | `SHADER_READ_ONLY_OPTIMAL` | vertex/compute/etc. / read |
| `UNORDERED_ACCESS` | `GENERAL` | shader / read+write |
| `COPY_DEST` / `COPY_SOURCE` | `TRANSFER_DST_OPTIMAL` / `TRANSFER_SRC_OPTIMAL` | transfer |
| `RESOLVE_DEST` / `RESOLVE_SOURCE` | `TRANSFER_DST_OPTIMAL` / `TRANSFER_SRC_OPTIMAL` | resolve |
| `STREAM_OUT` | no layout for buffers | transform feedback write (if used) |
| `VERTEX_BUFFER` / `INDEX_BUFFER` / `INDIRECT_ARGUMENT` | no layout for buffers | vertex input / index / draw indirect |
| `PRESENT` | `PRESENT_SRC_KHR` for swapchain images, `GENERAL` otherwise | present |
| `RAYTRACING_ACCELERATION_STRUCTURE` | buffer, no layout | acceleration structure read/write |

- Split barriers (`BEGIN_ONLY`/`END_ONLY`, and enhanced `SYNC_SPLIT`) are an optimisation. The resource is unusable
  between the halves, so one full barrier at either point is correct; the IR records it at the end point
  (`command_list.h`). Use `VK_KHR_synchronization2` events if profiling needs the overlap.
- Implicit promotion and decay need no extra barriers: `COMMON` is `GENERAL`, and the layout tracker makes later
  barriers start from it (section 3.2).
- Aliasing barriers: if image aliasing is implemented (one `VkDeviceMemory` per D3D12 heap range, with
  `VK_IMAGE_CREATE_ALIAS_BIT`), transition the old image to `UNDEFINED`, issue a global memory barrier, then
  transition the new image. Until then, log aliasing and allocate distinct memory; do not silently return the
  wrong contents.
- Enhanced barriers (`D3D12_BARRIER_SYNC_*`, `D3D12_BARRIER_ACCESS_*`, `D3D12_BARRIER_LAYOUT_*`) map directly to
  `VkPipelineStageFlags2`, `VkAccessFlags2` and `VkImageLayout` (`vulkan_sync`, `vulkan_barrier_access`,
  `vulkan_layout`). Queue-specific layouts map like their generic forms; ray tracing, video and split scopes use
  `ALL_COMMANDS`; `LayoutBefore = UNDEFINED` or the discard flag becomes `oldLayout = UNDEFINED`.
- Render passes: `BeginRenderPass`/`EndRenderPass` map to `vkCmdBeginRendering`/`vkCmdEndRendering` (dynamic
  rendering). The `D3D12_RENDER_PASS_BEGINNING_ACCESS`/`ENDING_ACCESS` types give the load/store ops and resolve
  targets. Suspending/resuming passes have no direct Vulkan equivalent; split them into two rendering scopes with
  the same attachments and `LOAD`/`STORE`. If the game does not use render passes, build a rendering scope per draw
  from the bound RTVs/DSV and let explicit `Clear*View` calls do the clears.

### 4.4 Fences, queues and frames in flight

- One timeline semaphore per `ID3D12Fence`. Queue `Signal` = `vkQueueSubmit` signal; queue `Wait` = submit wait.
- `SetEventOnCompletion(value, hEvent)`: a small wait thread calls `vkWaitSemaphores` and `SetEvent`. Reuse one
  thread with a condition variable; do not create a thread per fence.
- `GetCompletedValue` is `vkGetSemaphoreCounterValue`; CPU `Signal` is `vkSignalSemaphore`.
- The swapchain owns acquire/present semaphores and N frames in flight. A present waits for the frame's
  render-finished semaphore; the next acquire waits for the image's present-complete semaphore.
- Queue types: a direct queue maps to a graphics+compute queue, a compute queue to a compute-only family if one
  exists, a copy queue to a transfer-only family. If the host exposes only one family, serialize and log; async
  compute is a performance feature, not a correctness one.
- `GetTimestampFrequency`/`GetClockCalibration` use `VK_EXT_calibrated_timestamps` where available.

### 4.5 Swapchain and present

The game package does not include `dxgi.dll`, so the executable imports it from the system (or from the GDK).
That import must be shimmed. The shim provides:

- `CreateDXGIFactory`/`CreateDXGIFactory2` → one fake factory with one adapter describing the host Vulkan device.
  `IDXGIAdapter::GetDesc` reports host memory/LUID; the game may use these for settings. **Open:** whether it
  expects Xbox Series X values (16 GB, fixed clocks) instead.
- `IDXGIFactory::CreateSwapChain`/`CreateSwapChainForHwnd` → `VkSurfaceKHR` for the HWND + `VkSwapchainKHR`.
  `GetBuffer(i)` wraps each swapchain image as an `ID3D12Resource` so the game can create RTVs.
- `Present` → `vkQueuePresentKHR`; `ResizeBuffers`/`ResizeTarget` recreate the swapchain; `SetFullscreenState`
  toggles borderless/fullscreen; `GetFrameStatistics` is a stub.
- `DXGI_PRESENT_TEST`/`DO_NOT_WAIT` map to a no-op/`VK_NOT_READY` path.

**PresentX.** Public Xbox documentation mentions `ID3D12CommandQueue::PresentX`,
`D3D12XBOX_FRAME_PIPELINE_TOKEN` and `D3D12XBOX_PRESENT_PLANE_PARAMETERS` in the game-streaming APIs
([example](https://docs.xbox.com/reference/system/xgamestreaming/functions/xgamestreamingsetresolution)). PresentX
is not in the public `d3d12.h` vtable; its slot and struct layouts are **Open**. If the game uses it, the layer
must compose the present planes (source rects, destination placement) into the swapchain image and present. For
the first boot, implement the DXGI path and stub PresentX with a clear log.

### 4.6 Queries

- Occlusion: `D3D12_QUERY_TYPE_OCCLUSION` → `VK_QUERY_TYPE_OCCLUSION` with `VK_QUERY_CONTROL_PRECISE_BIT`;
  `BINARY_OCCLUSION` → the same pool without the precise bit, clamping the result to 0/1. `ResolveQueryData` →
  `vkCmdCopyQueryPoolResults` with `VK_QUERY_RESULT_64_BIT` (D3D12 returns UINT64).
- Pipeline statistics: `VK_QUERY_TYPE_PIPELINE_STATISTICS` with the flags implied by
  `D3D12_QUERY_DATA_PIPELINE_STATISTICS`; map the fields one-to-one.
- Timestamps: `VK_QUERY_TYPE_TIMESTAMP`; `GetTimestampFrequency` from the Vulkan timestamp period; calibration
  from `VK_EXT_calibrated_timestamps`.
- D3D12 query resolve destinations must be in `COPY_DEST`; honour the game's offsets and alignment.
- Xbox-specific query heap types (if any) are **Open**; log unknown enum values.

### 4.7 Indirect commands and command signatures

- `D3D12_COMMAND_SIGNATURE_DESC` describes the argument stride and a list of argument types. Simple draw,
  draw-indexed, dispatch and mesh-dispatch signatures map to `vkCmdDraw*Indirect`, `vkCmdDispatchIndirect` and
  `vkCmdDrawMeshTasksIndirectEXT`; use the `*Count` variants (`VK_KHR_draw_indirect_count`, core 1.2) when the
  `ExecuteIndirect` call supplies a count buffer.
- Signatures that change root constants, vertex/index buffers or descriptors per argument cannot be expressed by
  one Vulkan indirect call. Use a compute expansion pass that writes a Vulkan-shaped indirect buffer, or
  `VK_NV_device_generated_commands` on NVIDIA. Start with the simple signatures; the trace will show which
  signatures the game creates.
- Predication (`SetPredication`) maps to `VK_EXT_conditional_rendering`; D3D12's 64-bit equal/not-equal value
  becomes a 32-bit condition when the high dword is zero, or a small compute pass that evaluates the full 64-bit
  comparison into the condition buffer.

### 4.8 Ray tracing and XDXR

This is the largest Xbox-specific area, and the shipped `XboxBvhBuilder.dll` gives concrete evidence. The DLL's
own names use the `XDXR` prefix for Xbox's raytracing path:

- Its constructor takes `ID3D12Device11*`; its build/copy/rehydrate methods take
  `ID3D12GraphicsCommandList4/6/8*`.
- Prior art measured its constructor creating 53 compute pipelines through device slot 62, called with the
  compute-PSO shape (a plain `D3D12_COMPUTE_PIPELINE_STATE_DESC`, two zero words, iid, out); on the PC interface used
  there slot 62 is `CreateStateObject` (a different arity), so the proxy overrides it. The layer must implement
  slot 62 as compute-PSO creation, not as a state object.
- It exposes standard `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC` builds, batched Xbox builds
  (`D3D12XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2`), XDXR stream descriptions
  (`D3D12XBOX_XDXR_STREAM_DESC`), post-build offline statistics, and a `DEHYDRATE` build flag. The exact layouts
  of the Xbox types are not public; they are **Open**.
- The translator's `image_bvh_intersect_ray` / `image_bvh64_intersect_ray` already emulate the AMD hardware BVH
  node format: node id = address >> 3 | type, box16/box32/triangle nodes, 256-byte base units. That is the format
  the runtime must produce for the game's compute traversal.

**The builder's own shaders are a separate problem.** They are Xbox DXIL using `@xbox.*` intrinsics, not FSSM
RDNA2: prior art found 44 of the 53 call non-standard `@xbox.*` functions and fail the PC D3D12 shader compiler,
and 22 have been lowered to standard DXIL and accepted as pipelines on an RX 7800 XT. Thirty-one of the 53 use raw
64-bit addresses (global load/store/atomic and `s_load`), so lowering also needs an address-to-resource mapping.
The existing FSSM translator does not cover these shaders, and their results are not yet checked against a
reference. If the game uses the builder, the layer needs a DXIL path (or a substitute builder); this is on the
critical path for the RT effects, not a footnote.

The design choice follows from that: **build AMD-format BVH data, not a `VkAccelerationStructureKHR`, unless the
shaders are changed to `OpTraceRayKHR`.** The game's 64 RT compute shaders traverse the nodes themselves; a
Vulkan AS would be invisible to them. Two paths:

1. If `XboxBvhBuilder` does the build with its own compute dispatches (the likely case, given it ships as a DLL
   with a device and command lists), the layer must translate and run its Xbox DXIL compute shaders — it cannot
   feed DXIL to Vulkan directly. Buffers, UAVs, descriptors, barriers and indirect dispatch are necessary but not
   sufficient; the 53 builder shaders need the DXIL path described above, and the 31 raw-address shaders need a
   VA-to-resource mapping. The standard `BuildRaytracingAccelerationStructure` calls may never reach our command
   list.
2. If the game calls `ID3D12GraphicsCommandList4::BuildRaytracingAccelerationStructure`, the layer must implement
   it. Produce the AMD node layout in the arena, with the same geometry/instance rules the translator assumes,
   and make `GetRaytracingAccelerationStructurePrebuildInfo` return sizes for that layout.

**Open:** which path the game uses. Resolve by logging the first call to the standard AS methods and by tracing the
builder's command-list usage. If XDXR batched/streaming builds are used, their structs and compression formats need
to be recovered from call sites (or the NDA GDK headers, if a developer has access); they cannot be guessed.

### 4.9 Xbox-specific surface

| Feature | Public evidence | Design |
|---|---|---|
| No ESRAM | Xbox Series X\|S uses unified GDDR6; ESRAM was Xbox One | Do not implement ESRAM-era APIs. The `d3d12_xs.cpp` comment about "ESRAM-era extensions" should be read as Xbox One, not Series. |
| Binning | The GDK topic index names Xbox binning modes (NDA content) | Should be a no-op if the game uses it only as an optimisation: it changes how a render pass uses on-chip memory, not the result. Log and ignore for the first boot; revisit for performance. **Open** whether the game relies on binning for correctness. |
| DCC / texture compatibility | Public release note for `D3D12XBOX_RESOURCE_FLAG_FORCE_TEXTURE_COMPATIBILITY` | Let Vulkan choose compression; fail if the game aliases/reads raw compressed data. |
| PresentX / present planes | Public game-streaming docs | Compose planes into the swapchain; stub until the trace shows the call. **Open** slot/layout. |
| VRS | Public `D3D12_SHADING_RATE` / `D3D12_AXIS_SHADING_RATE` | `VK_KHR_fragment_shading_rate`; `RSSetShadingRate`/image-based rate map to `vkCmdSetFragmentShadingRateKHR`. |
| Mesh shader tier | Public `D3D12_MESH_SHADER_TIER` | `VK_EXT_mesh_shader`; the 9,282 translated mesh shaders make this a hard requirement for any content that uses them. |
| Wave MMA | Public `D3D12_WAVE_MMA_TIER` | If shaders use `v_mfma`, map to `VK_KHR_cooperative_matrix` or emulate. Not seen in the translator's coverage list; **Open**. |
| Enhanced barriers | Public Xbox Series `D3D12_BARRIER_GROUP` | `VK_KHR_synchronization2`; implement directly. |
| XDXR batched/streaming BVH | `XboxBvhBuilder.dll` exports | See 4.8. |

## 5. Producing the shader contract

The table below is the contract from [design.md](design.md#translator-status-and-runtime-contract) and where the
runtime gets each piece. The shader side is already proven by `runtime/test/vk_texture_test.cpp` and the compute,
store, load, mesh, geometry and BVH tests; this layer is the other half.

| Contract | Runtime source |
|---|---|
| `s0..s31` user data | Root signature state at draw/dispatch: root constants copied, root CBV/SRV/UAV addresses and descriptor-table base addresses as 64-bit console VAs. Mapping per section 3.5; absent parameters zero. |
| `heap_base` (dwords 32–33) | The arena's `vkGetBufferDeviceAddress`. One value per command buffer; updated only if the arena is recreated. |
| `code_base` (dwords 34–35) | The console VA of the current shader's FSSM blob in the arena. Per-stage conflict: see below. |
| Descriptor hash table (set 0, binding 2) | `uint mask` + entries of `key[8], index, pad`; FNV-1a over the 8 key dwords; linear probe up to 8; miss = index 0. The layer inserts every T#/S# it sees and keeps a fallback image/sampler at index 0. |
| Bindless images (bindings 0, 3, 5, 6) | 2D, 2D-array/cube, 1D, 3D sampled-image arrays; one `VkImageView` per distinct T#. Cube textures are 6-layer 2D arrays, as the translator expects. |
| Samplers (binding 1) | One `VkSampler` per distinct S#; literal S# values from shader constants are pre-registered. |
| Storage images (bindings 4, 7, 8–19) | r32ui/float/sint views by dimension and numeric class; the layer binds each storage image into the float/uint/sint arrays of its dimension (the translator picks by the T# format at run time), with all three arrays as the safe default. |
| Buffer/image binding | Descriptors contain console VAs, never host BDAs. Buffers are in the arena at their VA; images are keyed by T# through the hash table. |
| Vertex fetch | PSO input layout + ISG1 signature; location = D3D input register; format from `DXGI_FORMAT`; instance step rate from the layout. The fetch subroutine is already replaced by Vulkan inputs. |
| PS system inputs | `FragCoord` (xy/z/w), `FrontFacing`, barycentrics; enable `fragmentShaderBarycentric` and `fragmentShaderPixelInterlock` when the index entry asks. |
| Mesh/geometry stages | `VK_EXT_mesh_shader`; the index's max vertices/primitives size the SPIR-V output arrays and are checked against the device's mesh properties; geometry stage for the 195 voxelize shaders; `DispatchMesh` → `vkCmdDrawMeshTasksEXT`. |
| POPS collision | The runtime puts 0 (no collision, bit 31 clear) in the POPS collision SGPR; the translator uses fragment-shader interlock for ordering. **Open:** the exact SGPR location is a translator/driver detail not named in the current contract. |
| Wave size | `requiredSubgroupSize = 64` when supported (AMD); otherwise the translator's 32-wide approximation is explicit in the log. |
| Code-embedded data | The whole FSSM blob is in the arena; `s_getpc_b64` becomes `code_base + next-instruction offset`. |

### 5.1 Descriptor hash table details

The exact layout the shader reads (from `tools/rdna2spirv.py`, `desc_lookup`):

```
uint32 mask;                 // slot count - 1
struct entry {               // 10 uints = 40 bytes
    uint32 key[8];           // T#: 8 dwords; S#: 4 dwords then 0xFFFFFFFF x4
    uint32 index;            // bindless index
    uint32 pad;              // unused by the shader
} entries[mask + 1];
```

- Hash: FNV-1a over the eight key dwords as dwords (not bytes): `h = 2166136261; h = (h ^ key[i]) * 16777619`.
- Probe: `slot = (h + k) & mask` for `k = 0..7`; the first entry whose eight key dwords match yields its
  `index`; no match yields 0.
- The layer must keep the table sparse enough that no live key needs more than eight probes. Grow (rebuild into a
  larger power-of-two table) when an insertion would exceed eight probes, or at 50% load. Update binding 2 after
  the rebuild, at a point where no in-flight command buffer reads the old table.
- Insert on `CreateShaderResourceView`, `CreateUnorderedAccessView`, `CreateSampler`, `CopyDescriptors` and heap
  scans. A duplicate key updates the index in place.
- Index 0 is the fallback: a 1×1 texture and a default sampler (and a valid dummy storage image for the storage
  arrays). The translator returns 0 on a miss, so index 0 must always be valid.
- The table buffer is a normal storage buffer, not part of the arena. It can be host-visible and updated directly
  (host-coherent, or flushed before submit); if the game is sensitive to PCIe traffic, stage it like any other
  upload. The shader only reads it.
- Inserts can come from several threads (view creation is often parallel). Serialise them with a mutex, or stage
  per-thread and merge at submit. The shader only reads the table, and D3D12 already forbids modifying a descriptor
  that an in-flight command list is using, so the same rule applies.

### 5.2 Per-stage code base

The current 144-byte block has one 64-bit code base at dwords 34–35. A graphics pipeline has at least a vertex and
a pixel shader, and each may use `s_getpc_b64` for its own constant table. One block cannot hold two different
code bases at the same offset. The pipeline layout declares the block for every stage that may read it (or for all
graphics stages), and `vkCmdPushConstants` is called with those stage bits. This is a real gap in the current
contract, not a runtime bug:

- Only 155 shaders use code-embedded data, and a graphics pipeline has at least two stages (typically VS+PS, or
  MS+PS). If only one stage in a pipeline uses it, the layer can set that stage's code base; the other stage never
  reads it. This probably covers the first boot. **Assumption.**
- If both stages use it, the layer cannot satisfy both with the current contract. The clean fix is a translator
  change: give each stage its own code-base slot (for example dwords 34–35 VS, 36–37 PS, 38–39 CS, 40–41 GS/MS,
  extending the block to 168 bytes) and have each compiled shader read its stage's slot. A 168-byte push-constant
  block is still under the 256-byte limit of common desktop GPUs, but above Vulkan's 128-byte minimum; the device
  requirement must be checked.
- A second option is a per-stage uniform/storage buffer for the code base, patched in at pipeline creation. That
  adds a descriptor binding and a load per `s_getpc`, but avoids growing push constants. Decide after measuring
  how many real PSOs have two code-base stages.

**Open:** the exact count of such PSOs, and the minimum `maxPushConstantsSize` across target GPUs. If a GPU only
supports 128 bytes, the current 144-byte contract already fails; the layer needs a fallback (uniform buffer for the
tail) regardless.

### 5.3 Descriptor T#/S# parsing

The runtime needs a C++ decoder for the structures the shaders read. The fields it needs:

- **T# (image SRV):** base console VA, format (GFX10 format code), width/height/depth, mip levels, array layers,
  sample count, destination swizzle, dimension. The format table is `tools/gfx10_formats.json` (Mesa's
  `gfx10-rsrc.json`), the same one the translator uses.
- **S# (sampler):** filter modes, address modes (U/V/W), border color, min/max LOD, LOD bias, compare function,
  anisotropy, coordinate normalization.
- **V# (buffer):** base console VA, stride, record count, format, destination swizzle. Buffer descriptors are
  dereferenced directly by the shader; the layer only needs to validate the VA and the format for typed loads.

If the game calls `CreateShaderResourceView`/`CreateSampler`, the layer can encode these structures itself from
the D3D12 descriptors. If the game writes them, the layer decodes them. Both paths meet in the same parser. The
GPU texture test already proves the shader-side contract on a synthetic T#/S#; the runtime parser is new code.

## 6. Boot milestones

Each milestone is a capability gate with a synthetic test; the game is the integration test once it is available.
M0–M2 do not need the executable.

| # | Scope | What it proves | Done when |
|---|---|---|---|
| **M0** | `D3D12XboxCreateDevice` and the measured device overrides (heap start handles in RAX), one direct queue, allocator/list, fence, fake DXGI factory/adapter/swapchain, clear + present | COM interception, the known Xbox device entry point and ABI overrides, swapchain, frame loop, clean shutdown | A window shows a stable clear colour for 60 frames, no Vulkan validation errors, every D3D12 call traced |
| **M1** | Root signature with root constants (mask the measured `0x03000000` flag byte on a copy), one translated VS+PS, PSO, vertex buffer in the arena, draw, push constants `s0..s31` + heap base | PSO translation, shader hash lookup, vertex input, root-constant packing, heap base, present | The triangle matches a CPU/GPU reference; the existing texture-test style comparison passes with error 0 |
| **M2** | Descriptor heap (heap slots 9/10 return in RAX), `CreateShaderResourceView`, sampler, root descriptor table, hash-table insert, bindless sampling | The full D3D12 descriptor path and the shader's T#/S# contract | A textured quad matches the synthetic 8×8 gradient or a known texture with error 0; the hash table has the expected entries |
| **M3** | Depth/stencil, multiple render targets, resource barriers, render passes, occlusion queries, fences, 2–3 frames in flight; run the game's first frames | Real frame structure and the first game code | The game reaches its first rendered frame (loading screen/menu or whatever the executable does) or stops on a single unsupported call with a clean report |
| **M4** | Compute PSOs, UAVs, storage images, indirect dispatch, copy queues, async streaming | Asset loading and GPU compute | The world's streaming path runs without device loss; compute results match a CPU/reference check |
| **M5** | Mesh shaders, `ExecuteIndirect`, bindless-table pressure, multi-threaded command lists, pipeline cache | World rendering | A stable open-world frame at any frame rate; then profile and optimise |
| **M6** | Ray tracing/XDXR: standard AS builds or the shipped `XboxBvhBuilder` compute path; BVH node format; Xbox DXIL path for the builder's 53 shaders (22 lowered so far) | RT shadows and any RT effects | RT output matches a reference; the builder's shaders run through the DXIL path or a substitute; no unknown XDXR call remains |
| **M7** | HDR, VRS, PresentX planes, resize, alt-tab, frame pacing | Presentation polish | HDR and resize work on the target display; no validation errors over a long run |

### 6.1 Minimum API set for the first frame

M0–M3's synthetic tests need only this subset. The game itself may need more as soon as its first frame does;
everything else can be absent or a clean failure:

- `D3D12XboxCreateDevice` and `CheckFeatureSupport`; the measured device overrides (heap start handles in RAX;
  slot 76 texture creation if the game uses it; slot 62, compute-PSO creation in the builder's ABI, is not needed
  for the first frame); one direct queue, one allocator, one command list, one fence;
- DXGI factory/adapter/swapchain, `GetBuffer`, `Present`;
- a root signature with root constants and one descriptor table, masking `0x03000000` on a copy; one VS+PS PSO and
  an input layout;
- a shader-visible CBV/SRV/UAV heap and a sampler heap; `CreateShaderResourceView`, `CreateSampler`,
  `SetDescriptorHeaps`, `SetGraphicsRootDescriptorTable`;
- one committed vertex/index buffer, one texture, upload through `Map`/`Unmap` or `WriteToSubresource`,
  `CopyBufferToImage`;
- resource barriers, RTV/DSV, `OMSetRenderTargets`, `Clear*View`, `DrawIndexedInstanced`, `Present`;
- the runtime side: descriptor hash table, bindless set 0, push constants (`s0..s31`, heap base, code base),
  vertex input from the PSO layout.

Not needed for the first frame: compute, mesh, geometry, ray tracing, VRS, PresentX, reserved/tiled resources,
heap aliasing, indirect draws, multiple queues, pipeline libraries.

**Why this order.** M0–M2 isolate the layer from the game: they exercise every contract piece with known data.
M3 is the first point where the game's own root signatures, PSOs and descriptors appear; expect the trace to
rewrite parts of section 3.5. M4/M5 are ordered by dependency: assets must stream before the world can draw, and
the corpus's 9,282 mesh shaders (and, very likely, indirect draws) are world-rendering features. M6 is last
because the game ships its own BVH builder; the layer may only need to run its compute work, not implement a full
DXR path.

**First-boot minimum.** Strictly, the first boot needs M0–M3 plus whatever the game does before its first frame.
If the first frame is a loading screen with a simple UI draw, M3 may be enough; if it is the world, M5 is the real
minimum. The trace decides, not the milestone labels.

## 7. Prior art and measured Xbox ABI

A GPL-2.0-only fork of AnyPS5 (`americas-recomp`, branch `xbox-series`) recompiles the same GTA VI Xbox Series
shaders and has run the package's DLLs against a PC D3D12 device behind an Xbox-ABI proxy. Its measurements are
prior art for this design. The findings below are paraphrased; no code or long passages are copied. They were
verified there by running the package DLLs (`XS_TRACE=1`, per-slot hit counters) or by static import analysis, not
independently re-measured here. **Basis:** *static imports* means PE import-table analysis; *run* means the package
DLLs were executed against a PC D3D12 device through the Xbox-ABI proxy.

| Finding | Basis and link | Consequence for this design |
|---|---|---|
| Across the 16 package DLLs there are only 6 Xbox-specific static imports: `xmem.dll` → `XMemAlloc`, `XMemFree`, `XMemGetWorkingSetStatistics`, `XMemVirtualQuery`; `d3d12_xs.dll` → `D3D12SerializeVersionedRootSignature`, `D3D12CreateRootSignatureDeserializer`. Both `d3d12_xs` names exist with the same names in the PC `d3d12.dll`. The other 492 functions are standard Windows and 113 are shipped in the package. Argument shapes for the less common `xmem` calls come from caller disassembly; some attribute bits are inferred. | *Static imports* — [PLATFORM_SURFACE.md](https://github.com/realdody/americas-recomp/blob/xbox-series/docs/xbox/PLATFORM_SURFACE.md) | Confirms the middleware surface is tiny. The repo's `xmem` shim already covers all four; `d3d12_xs` still forwards only the two root-signature functions. The device/queue/command-list surface belongs to the executable. |
| `D3D12XboxCreateDevice(adapter, params, riid, out)` returns a PC `ID3D12Device` behind the Xbox-ABI proxy. The params struct is 24 words and is only logged; the signature comes from public documentation, not a header. | *Run* — [platform README](https://github.com/realdody/americas-recomp/blob/xbox-series/tools/xbox/platform/README.md), [PLATFORM_SURFACE.md](https://github.com/realdody/americas-recomp/blob/xbox-series/docs/xbox/PLATFORM_SURFACE.md) | Corrects "device entry point unknown": the entry point is named. The params/flag layout remains **Open**. Do not assume `D3D12CreateDevice`. |
| Xbox texture creation is device slot 76 (heap properties / heap flags / description / initial state / clear value / iid / out), not `CreateCommittedResource`. The description is 56 bytes plus extension words; its layout field is `0x101` (tiled) or `0x1` (row-major buffer), and the PC proxy forwards it to `CreateCommittedResource` slot 27 with a driver-chosen layout. The FidelityFX upscaler creates 10 default-heap resources this way. | *Run* — [PLATFORM_SURFACE.md](https://github.com/realdody/americas-recomp/blob/xbox-series/docs/xbox/PLATFORM_SURFACE.md) | Adds a required Xbox-specific device method and a non-public layout value. `0x101` must not be passed to the public layout enum; the tiled data layout/detile path remains **Open**. See section 4.2. |
| Device slot 14 is the public `CreateDescriptorHeap`; the returned heap is wrapped because `GetCPUDescriptorHandleForHeapStart` / `GetGPUDescriptorHandleForHeapStart` (heap slots 9/10) return the 8-byte handle in RAX on the Xbox build. The PC call returns it through a hidden pointer in RDX, so the proxy special-cases the return. | *Run* — [PLATFORM_SURFACE.md](https://github.com/realdody/americas-recomp/blob/xbox-series/docs/xbox/PLATFORM_SURFACE.md) | Adds a return-convention quirk to the descriptor-heap ABI. A generic IDL/sret thunk will corrupt the handle. See section 3.4. |
| Xbox root-signature descriptors carry flags `0x03000000`. The PC serializer rejects them with `E_INVALIDARG`; the proxy clears the high byte on a copy. The v1.0 `D3D12SerializeRootSignature` path keeps flags at +32, and `0x3000001` also fails on the PC runtime. | *Run* — [PLATFORM_SURFACE.md](https://github.com/realdody/americas-recomp/blob/xbox-series/docs/xbox/PLATFORM_SURFACE.md) | Narrows the root-signature reflection risk. Mask the high byte on a private copy; the meaning of the bits remains **Open**. See section 3.5. |
| The BVH builder's constructor creates 53 compute pipelines through device slot 62, called with a compute-PSO shape (a plain `D3D12_COMPUTE_PIPELINE_STATE_DESC`, two zero words, iid, out). On the PC interface the prior art forwards to, slot 62 is `CreateStateObject` (wrong arity), so the proxy overrides it. | *Run* — [PLATFORM_SURFACE.md](https://github.com/realdody/americas-recomp/blob/xbox-series/docs/xbox/PLATFORM_SURFACE.md) | Corrects the assumption that the whole device vtable is public order. Slot 62 must be compute-PSO creation for the builder, not a state object. Slots 1/15/18/19/27/38 did match the desktop order. |
| A reusable `xbox_device_proxy.h` forwards every slot to a PC object, counts hits per slot and logs first calls with `XS_TRACE=1`; the observed device slots include 1, 14, 15, 16, 18, 19, 27, 38, 62 and 76. Their `xgr` stand-in covers 48 of 398 methods; Xodus (LGPL-2.1) covers 152 of 396 via IDL-generated stubs. | *Run* — [platform README](https://github.com/realdody/americas-recomp/blob/xbox-series/tools/xbox/platform/README.md), [PLATFORM_SURFACE.md](https://github.com/realdody/americas-recomp/blob/xbox-series/docs/xbox/PLATFORM_SURFACE.md) | Confirms the trace-first approach and shows the long tail. We adopt the *method* (per-slot counters, first-call logging), not the code. |
| The BVH builder's 53 compute shaders are Xbox DXIL with `@xbox.*` intrinsics; 44 of 53 fail the PC D3D12 shader compiler, 22 have been lowered to standard DXIL and accepted as pipelines, and 31 use raw 64-bit GPU addresses. | *Run* — [PLATFORM_SURFACE.md](https://github.com/realdody/americas-recomp/blob/xbox-series/docs/xbox/PLATFORM_SURFACE.md) | A separate translation path from the FSSM RDNA2 corpus. See section 4.8 and the paragraph below. |

### 7.1 What this confirms, corrects and narrows

**Confirms.**

- The 16 middleware DLLs need only four `xmem` functions and the two `d3d12_xs` root-signature functions. Our
  earlier statement that the bundled DLLs only import the two root-signature calls was right but incomplete; the
  four `xmem` imports complete the static surface. The repo's `xmem` shim already implements all four.
- Device slots 1 (AddRef), 15 (`GetDescriptorHandleIncrementSize`), 18 (`CreateShaderResourceView`),
  19 (`CreateUnorderedAccessView`), 27 (`CreateCommittedResource`) and 38 (`GetCopyableFootprints`) match the
  desktop order. Device slot 14 is the public `CreateDescriptorHeap`; only the returned heap's start-handle return
  is wrapped. The public chain is a usable base for those methods, not a guess.
- The BVH builder receives the device and command lists from the caller; prior art runs it exactly that way.

**Corrects.**

- **Device creation.** `D3D12XboxCreateDevice` is the entry point. The 24-word params struct is only logged, so
  the flag/layout meaning is still unknown; the name and call shape are no longer unknown. Prior art returns a PC
  D3D12 device behind the proxy; this design returns the Vulkan-backed `Device` behind the same call shape.
- **Vtable.** The Xbox device is not uniformly the public order. Slot 76 is an extra texture-creation method,
  slot 62 is called with a compute-PSO signature where the PC interface has a state-object method, and the
  descriptor-heap start handles use a different return convention. Implement the public chain plus explicit
  measured overrides; do not trust a single generic thunk for every slot.
- **Texture layout.** Xbox `0x101` tiled is not a public `D3D12_TEXTURE_LAYOUT` value, and slot 76 is the texture
  path. Our earlier `D3D12_RESOURCE_DESC::Layout` discussion covered only the public enum; it now includes the
  Xbox value and slot 76.
- **Root signatures.** The `0x03000000` flags must be masked on a copy before the host serializer/deserializer.
  Our earlier "forward the blob to the host" was too simple; the PC serializer rejects the Xbox bits.
- **BVH builder.** Its shaders are Xbox DXIL, not FSSM. "Run the builder's dispatches" therefore needs a DXIL
  translation/lowering path plus an address-to-resource mapping for the raw-address shaders; the FSSM translator
  does not cover them.

**Narrows.**

- Presentation (`PresentX` and plane parameters), command-list extensions, and queue/fence creation slots remain
  unknown; the prior art explicitly stops before them.
- Descriptor start-handle ABI is known; descriptor bytes and heap increments are still **Open**. Slot 15 is
  forwarded unchanged, but the actual increment values were not measured.
- Texture creation ABI and the `0x101` value are known; the tiled data layout, detile path and DCC remain
  **Open**.
- The device params (command-queue ring size, scratch memory, debug flags) are unused and their meanings are
  still unknown.
- The `xgr` 48/398 and Xodus 152/396 coverage numbers show how large the long tail is. They do not change the
  plan: implement what the game calls, in the order it calls it.

### 7.2 The BVH builder's Xbox DXIL path

The builder's 53 compute shaders are a second shader front end. They are Xbox DXIL with `@xbox.*` intrinsics that
map one-to-one to RDNA instructions (global load/store/atomic with 64-bit addresses, `s_load`, min/med/max,
bitfield insert, alignbyte, cndmask, packed f16 min/max and lane/wave operations). The standard PC DXIL compiler
rejects 44 of the 53; 22 have been lowered to standard DXIL and compile; the 31 that use raw 64-bit addresses also
need a table mapping those addresses to resources. This is separate from the FSSM RDNA2 translator, which covers
the game's 26,503 shaders. A port that uses `XboxBvhBuilder` needs both: FSSM → SPIR-V for the game, and an Xbox
DXIL → standard DXIL → SPIR-V path (or a substitute builder) for the BVH helper. The prior art has compiled
lowered shaders but has no reference output yet, so correctness is unverified.

### 7.3 Licence

`americas-recomp` / AnyPS5 is **GPL-2.0-only**. GPL-2.0-only is not compatible with this repository's
GPL-3.0-or-later, so no code, headers, generated stubs or other implementation text from that tree may be copied
into Leonida. This section records ABI/format facts and links only; the designs in this document are independent.
Xodus is LGPL-2.1 and is cited only for its coverage number, not as a code source.

## 8. Known vs unknown

### 8.1 Known

- The public D3D12 ABI and method order (Windows SDK `d3d12.h` 10.0.26100.0); the Xbox public reference lists the
  same base interfaces and method names under `d3d12_xs.h`, but the measured Xbox device ABI is only mostly the
  public order: slots 1/15/18/19/27/38 matched, while slots 62 and 76 and the descriptor-heap start-handle return
  did not (section 7).
- The translator/runtime shader contract, and that it works on a real GPU for the tested paths.
- The package evidence in section 2 and Appendix A: the BVH builder's Device11/CommandList4/6/8 signatures and
  XDXR types, the FSR IIDs, the other DLLs' lack of `d3d12_xs` imports, `gameos.xvd`'s encrypted payload, and the
  absent executable.
- The prior art's measured Xbox ABI (section 7): 6 Xbox-specific static imports across the 16 DLLs,
  `D3D12XboxCreateDevice`, device slots 62/76, descriptor-heap start handles in RAX, root-signature flags
  `0x03000000`, and the BVH builder's Xbox DXIL shaders.

### 8.2 Unknown / open questions

| # | Question | How to resolve |
|---|---|---|
| 1 | The device entry point is now named `D3D12XboxCreateDevice(adapter, params, riid, out)`. What do its 24-word params and flag bits mean? | Public documentation and the executable's call sites. Log the words and accept any flag bits; do not treat a wrong params layout as fatal until the fields are understood. |
| 2 | Which Xbox device/heap/queue/command-list slots differ? Prior art measured device slots 62 and 76 and descriptor-heap slots 9/10; slot 14 is the public `CreateDescriptorHeap`, with only the returned heap's start-handle return wrapped. Presentation, command-list, queue and fence slots remain unknown. | Trace every `QueryInterface` and every method slot with per-slot hit counters. A wrong slot shows as an immediate crash or wrong argument; compare against the SDK vtable. |
| 3 | How does the console map a root signature to `s0..s31` (order, per-stage visibility, spills above 32 dwords, static samplers, direct-heap base)? | Dump the game's root-signature blobs with the host deserializer; mask the `0x03000000` flag byte on a copy first; match against the per-shader user-SGPR read sets. If a signature exceeds 32 dwords, reverse the spill pointer from the shader. |
| 4 | Are Xbox descriptor bytes GFX10 T#/S#, and what are the heap increments? Does the game use `Create*View` or build descriptors itself? | Dump a shader-visible heap after the first `CreateShaderResourceView` and after the first draw; compare the bytes the shader loads with the T# parser. The start-handle return ABI is known (RAX); the increment values still need measuring. |
| 5 | What is the buffer/descriptor/code VA range, and what arena capacity is needed? | Log every `GetGPUVirtualAddress` and every descriptor base; track the high-water mark. Choose a capacity; grow at a frame boundary if needed. |
| 6 | Which swapchain/present path is used: DXGI, `PresentX`, or both? | The executable's imports and the first present call. Implement DXGI first; add the PresentX plane compose if the trace shows it. |
| 7 | Does the game build acceleration structures through the standard D3D12 calls or through `XboxBvhBuilder`'s own compute path? | Log `BuildRaytracingAccelerationStructure`, `CopyRaytracingAccelerationStructure`, `Emit...PostbuildInfo`, and the builder's command-list method calls on the first RT frame. The builder's 53 shaders are Xbox DXIL; 44/53 fail the PC compiler and 22 have been lowered. If the builder path is used, the layer needs a DXIL plan. |
| 8 | What are the layouts of `D3D12XBOX_XDXR_STREAM_DESC`, the batched build desc, and the offline statistics desc? | Recover from the builder's call sites, or use the NDA GDK headers if available. Do not guess; a wrong layout corrupts memory. |
| 9 | Xbox texture creation is device slot 76 with layout `0x101` (tiled) or `0x1` (row-major). What is the tiled data layout, and does the game upload console-tiled or DCC data? | Trace slot 76, `CopyTextureRegion` and the resource descs. If the game uploads console-tiled data, add a detile step or a matching layout. |
| 10 | Does the game use binning, DCC, VRS, wave MMA, stream output, tessellation, or amplification shaders? | Scan the PSO/stream subobjects and the shader corpus. Ignore binning for correctness; implement VRS/mesh if present; map WMMA only if `v_mfma` appears. |
| 11 | What should `CheckFeatureSupport` answer for UMA, resource binding tier, mesh tier, RT tier? | Log every feature query and the game's subsequent behaviour; answer truthfully from Vulkan first. |
| 12 | Does the game rely on async compute or copy queues for correctness? | Trace queue creation and cross-queue fences. If only performance, serialize on one queue for boot. |
| 13 | Is 144 bytes of push constants available on all target GPUs? | Check `maxPushConstantsSize` at device creation; if below 144, implement the uniform-buffer fallback and adjust the translator contract. |
| 14 | Are there PSOs with two code-base stages? | Count from the PSO/shader hash table and the index metadata. If yes, implement per-stage code-base slots (section 5.2). |
| 15 | Is `ExecuteIndirect` used with signatures Vulkan cannot express directly? | Trace `CreateCommandSignature` and `ExecuteIndirect`; implement simple signatures first, expand the complex ones with compute. |
| 16 | Does the game use append/consume UAV counters or typed buffer formats the translator does not model? | Trace UAV descs and shader `buffer_*` ops; extend the translator/runtime together. |
| 17 | Do upload/readback resources ever reach shaders (as vertex/index buffers or descriptors), and does the `-heap_base` bias work for them? | Trace `CreateCommittedResource` heap types and every VB/IB/descriptor that points at an upload resource; test the bias with a synthetic draw. |
| 18 | Does the game alias buffers/images in a heap, or use reserved/tiled resources? | Trace `CreatePlacedResource` ranges, aliasing barriers, `CreateReservedResource` and tile mappings. If yes, implement shared memory per heap range or fall back to linear allocations with a clear log. |
| 19 | What do the `0x03000000` root-signature flag bits mean? | Masking them on a copy works around the PC serializer; the semantics remain unknown. Compare against the public flag enum and observe the game's behaviour with and without the mask. |
| 20 | Which of the BVH builder's 53 Xbox DXIL shaders does the game actually use, and can the 31 raw-address shaders be lowered without a reference? | Dump the slot-62 compute PSOs on the first RT frame; lower and compile each; compare against the builder's CPU validation (`CpuValidateResultX`) or a captured reference. |

## 9. Risks and non-goals

**Risks.**

- **Xbox ABI deviations.** The device vtable is mostly public but not uniformly: slot 76 adds texture creation,
  slot 62 is compute-PSO creation for the BVH builder, descriptor-heap slots 9/10 return in RAX, and root signatures
  carry `0x03000000`. Use a per-slot proxy with counters; presentation, command-list, queue and fence slots remain
  unknown. A wrong return convention or slot mapping corrupts state rather than failing cleanly.
- **Root-signature mapping.** A wrong `s0..s31` packing produces silently wrong rendering, not a crash. The
  per-shader read-set metadata and a strict mismatch check are the mitigation. The `0x03000000` flag mask is a
  workaround, not an explanation.
- **The arena.** A single BDA arena is required by the shader contract for buffers, descriptor tables and code; the
  game's buffer working set may exceed `maxBufferSize` or PC VRAM. Host-visible arenas are slow; device-local
  arenas need staging and careful upload/readback handling. This is the main performance risk.
- **Hash-table cost.** Every image/sampler access hashes a T#/S# and probes a storage buffer. The GPU test proves
  correctness, not speed. Measure early; optimise the table or cache indices without changing the shader contract.
- **Wave64.** On NVIDIA, wave64 shaders use the translator's 32-wide approximation; cross-half ops are approximate.
  AMD GPUs with 64-wide subgroups are exact. This is a known fidelity gap, not specific to the layer.
- **Xbox tiling/DCC.** Slot 76 and layout `0x101` are known, but the tiled data layout, detile path and DCC remain
  unknown. If the game assumes console tiling or compression metadata, the layer needs a detile/repack step.
- **BVH builder DXIL.** 44 of the builder's 53 Xbox DXIL shaders fail the PC compiler; 22 have been lowered to
  standard DXIL and compile, and 31 use raw 64-bit addresses. The lowered shaders have no reference output yet, and
  this is a separate front end from the FSSM translator.
- **Ray tracing.** The game's RT work includes 64 compute shaders that traverse AMD-format BVH nodes, and it ships
  a BVH builder; a standard D3D12 AS path may also exist. If it is used, the layer must build AMD-format nodes, not
  a Vulkan AS.

**Non-goals.**

- Xbox One ESRAM or `d3d12_x` (the Xbox One flavour); this is a Series build.
- D3D11, D3D12 video, protected content, cross-process sharing, meta commands, DRED tooling.
- A general D3D12 compatibility layer for arbitrary applications.
- Any use of game data, keys or decryption.
- Copying code, headers or generated stubs from the GPL-2.0-only prior art; its ABI/format facts are cited with
  links only.
- Replacing the translator's shader contract. The layer adapts to it; changes to it (per-stage code base, extra
  metadata) are coordinated with `tools/rdna2spirv.py`.

## 10. References

Prior art (facts and links only; no code copied; GPL-2.0-only, see section 7.3):

- [americas-recomp, branch `xbox-series`](https://github.com/realdody/americas-recomp/tree/xbox-series)
- [PLATFORM_SURFACE.md: package import surface and measured device ABI](https://github.com/realdody/americas-recomp/blob/xbox-series/docs/xbox/PLATFORM_SURFACE.md)
- [platform tools README: proxy, `XS_TRACE`, xgr, first-run workflow](https://github.com/realdody/americas-recomp/blob/xbox-series/tools/xbox/platform/README.md)
- [GPL-2.0 license text](https://github.com/realdody/americas-recomp/blob/xbox-series/LICENSE)

Public D3D12:

- [D3D12 programming guide](https://learn.microsoft.com/en-us/windows/win32/direct3d12/directx-12-programming-guide)
- [Root signatures](https://learn.microsoft.com/en-us/windows/win32/direct3d12/root-signatures)
- [`D3D12_ROOT_SIGNATURE_FLAGS`](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_root_signature_flags)
- [Resource binding in HLSL](https://learn.microsoft.com/en-us/windows/win32/direct3d12/resource-binding-in-hlsl)
- [Descriptor heaps](https://learn.microsoft.com/en-us/windows/win32/direct3d12/descriptor-heaps)
- [Recording command lists and bundles](https://learn.microsoft.com/en-us/windows/win32/direct3d12/recording-command-lists-and-bundles)
- [Using resource barriers](https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-resource-barriers-to-synchronize-resource-states-in-direct3d-12)
- [Enhanced barriers (DirectX spec)](https://microsoft.github.io/DirectX-Specs/d3d/D3D12EnhancedBarriers.html)
- [Executing and synchronizing command lists](https://learn.microsoft.com/en-us/windows/win32/direct3d12/executing-and-synchronizing-command-lists)
- [Fence-based resource management](https://learn.microsoft.com/en-us/windows/win32/direct3d12/fence-based-resource-management)
- [Queries](https://learn.microsoft.com/en-us/windows/win32/direct3d12/queries)
- [Mesh shader (DirectX spec)](https://microsoft.github.io/DirectX-Specs/d3d/MeshShader.html)
- [Direct3D 12 raytracing](https://learn.microsoft.com/en-us/windows/win32/direct3d12/direct3d-12-raytracing)
- [`ID3D12Device10`](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nn-d3d12-id3d12device10),
  [`ID3D12GraphicsCommandList7`](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nn-d3d12-id3d12graphicscommandlist7)
  (the Win32 reference does not yet have pages for `Device11`/`CommandList8`; the Windows SDK header defines them)

Xbox public documentation:

- [Xbox graphics API reference](https://devdocs.xbox.com/reference/graphics/d3d12)
- [Xbox `ID3D12Device::CreateCommittedResource`](https://devdocs.xbox.com/reference/graphics/d3d12/interfaces/id3d12device/methods/id3d12device_createcommittedresource_public)
- [`PresentX` mentioned in `XGameStreamingSetResolution`](https://docs.xbox.com/reference/system/xgamestreaming/functions/xgamestreamingsetresolution)
- [GDK release notes mentioning `D3D12XBOX_RESOURCE_FLAG_FORCE_TEXTURE_COMPATIBILITY`](https://docs.xbox.com/announcements/gdk/2510/release-notes-2510)

Vulkan:

- [Descriptor indexing](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_descriptor_indexing.html)
- [Buffer device address](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_buffer_device_address.html)
- [Synchronization2](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_synchronization2.html)
- [Dynamic rendering](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_dynamic_rendering.html)
- [Timeline semaphores](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_timeline_semaphore.html)
- [Mesh shader](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_mesh_shader.html)
- [Fragment shader barycentric](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_fragment_shader_barycentric.html)
- [Fragment shader interlock](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_fragment_shader_interlock.html)
- [Subgroup size control](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_subgroup_size_control.html)
- [Maintenance4](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_maintenance4.html)

## Appendix A: evidence and reproduction

REA was used for the PE import/export and string inventory. The XVD header and entropy scan used local .NET,
because `gameos.xvd` is an XVD container rather than a PE. Nothing was decrypted or executed.

### A.1 `gameos.xvd`

- Size 348,069,888 bytes. `msft-xvd` magic at file offset **0x200**; the 512 bytes before it are a high-entropy
  block, consistent with a signature. Using the standard XVD header layout, the first two dwords after the magic
  are `volume_flags = 1` and `format_version = 3`.
- 64 KiB Shannon entropy sampled at offsets 0, 0x1000, 0x10000, 1 MiB, 16 MiB, 64 MiB, 128 MiB, 256 MiB:
  6.40, 6.69, 7.68, 7.68, 7.997, 7.998, 7.997, 7.997 bits/byte.
- A chunked ASCII scan (4 MiB chunks) of the whole file found no `d3d12_xs.dll`, `d3d12.dll`,
  `D3D12CreateDevice`, `ID3D12Device`, `gameos`, or `xboxgip` strings. A token spanning a chunk boundary could be
  missed, but the entropy result does not depend on the scan.
- Conclusion: the payload is encrypted or compressed. It cannot supply the D3D12 API surface without keys, which
  are out of scope. No key search or decryption was attempted.

Reproduce: read the first 4 KiB, locate `msft-xvd`; compute per-block entropy; scan for the ASCII tokens above.
The file is an XVD container, not a PE; PE-oriented import tools do not apply.

### A.2 `XboxBvhBuilder.dll`

- Size 884,224 bytes; PE32+ x86-64; export directory RVA 861,536, 12 named exports.
- The exported C++ signatures name:
  - `ID3D12Device11` (constructor, rehydration),
  - `ID3D12GraphicsCommandList4` (standard build, copy, CPU validation),
  - `ID3D12GraphicsCommandList6` (batched build),
  - `ID3D12GraphicsCommandList8` (batched rehydration),
  - `D3D12XBOX_XDXR_STREAM_DESC` and `D3D12XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2`,
  - `D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC` and the standard build/copy/prebuild types.
- Imports: `d3d12_xs.dll!D3D12SerializeVersionedRootSignature` (ILT RVA 864,368, IAT RVA 148,048) and
  `xmem.dll!XMemVirtualQuery` (ILT RVA 864,384, IAT RVA 148,064). No other console graphics imports.
- GUID byte scan against the Windows SDK `d3d12.h` found `IID_ID3D12PipelineState` at file offset 0x0c9ed8 and
  `IID_ID3D12RootSignature` at 0x0c9ee8. Presence in a read-only data table is evidence of the interface version
  the DLL was compiled against, not proof that every method is called.

Reproduce: `rea.inspect_native_exports` and `rea.inspect_pe_imports` on the DLL, then a byte scan for the
little-endian GUIDs from `d3d12.h`.

### A.3 `amd_fidelityfx_upscaler_xbox.dll`

- Size 786,432 bytes; PE32+ x86-64.
- Import: `d3d12_xs.dll!D3D12CreateRootSignatureDeserializer` (IAT RVA 164,672).
- GUID byte scan found `IID_ID3D12Device` (0x0472f0), `IID_ID3D12Resource` (0x047310),
  `IID_ID3D12RootSignatureDeserializer` (0x047320), `IID_ID3D12PipelineState` (0x047330),
  `IID_ID3D12Heap` (0x047340), `IID_ID3D12RootSignature` (0x047350), `IID_ID3D12DescriptorHeap` (0x0472e0) and
  `IID_ID3D12CommandSignature` (0x047300). Again, presence ≠ use.

### A.4 Other DLLs and the executable

- `amd_fidelityfx_loader_xbox.dll`, `Bink2Scarlett.dll`, `libHttpClient.GDK.dll`, `XCurl.dll`: no `d3d12_xs`
  imports.
- The package layout XML, chunk 1001, lists `game_scarlett_master.exe`; the file is absent from the folder. No
  claim in this document depends on the executable's contents.
- The repository's existing import survey (`docs/design.md`, "Baseline already recorded") records that the 16
  bundled DLLs need only `xmem` and the two `d3d12_xs` root-signature calls; our scan of the six non-CRT DLLs is
  consistent.

## Appendix B: interface versions seen in shipped binaries

| Interface | Evidence | Public methods beyond the previous version |
|---|---|---|
| `ID3D12Device11` | `XboxBvhBuilder.dll` constructor and rehydration signatures | `CreateSampler2` (public SDK) |
| `ID3D12GraphicsCommandList4` | standard BVH build/copy/validation signatures | `BeginRenderPass`/`EndRenderPass`, `BuildRaytracingAccelerationStructure`, `EmitRaytracingAccelerationStructurePostbuildInfo`, `CopyRaytracingAccelerationStructure`, `DispatchRays`, `SetPipelineState1` |
| `ID3D12GraphicsCommandList6` | batched BVH build signature | `DispatchMesh` |
| `ID3D12GraphicsCommandList8` | batched rehydration signatures | `OMSetFrontAndBackStencilRef` |

The public Xbox graphics index retrieved for this document lists interfaces up to `ID3D12Device10` and
`ID3D12GraphicsCommandList7`; the shipped DLL names `Device11` and `CommandList8`, which the index does not cover.
This is why the layer targets the Windows SDK's newer public chain rather than assuming the Xbox public index is
complete.
