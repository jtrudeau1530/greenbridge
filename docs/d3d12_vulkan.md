# Xbox D3D12 (`d3d12_xs`) to Vulkan

**Status: in progress. A Vulkan-backed `ID3D12Device` (queues, allocators, command lists, fences, descriptor heaps,
render-target and depth views, texture views and samplers, committed and reserved buffers, 2D textures and 3D texture transfers and color clears, and query heaps)
runs behind the Xbox-ABI device proxy, and a presenter puts back buffers in a window (section 3.2). The M0
milestone passes in a synthetic test (section 6). `D3D12XboxCreateDevice` uses it when `LEONIDA_D3D12=vulkan` is
set and the host PC device otherwise. Buffer views, root signatures, PSOs, draws and the game's own present
entry point (DXGI or PresentX) are not implemented.** This is the plan for replacing the console's
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
- **Trace first.** `gta6rt --trace-d3d12` emits numbered records for device entry points, Vulkan device operations,
  command-list recording and queue submission. Records contain fixed API names and outcomes only. See
  [the runtime guide](../runtime/README.md#first-boot-d3d12-trace) for collecting a trace.

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
- `Close` freezes the IR. `ExecuteBundle` is accepted only on a direct list, with an already-closed bundle as its callee; the bundle IR is copied by value at the call site, so the source can be reused or reset without changing the parent. Command-list creation and reset require an allocator of the same type. A bundle cannot be submitted directly to a command queue.
- Bundle restrictions are checked while recording for the IR operations currently implemented here (barriers, clears, copies, resolves, `SetPredication`, `DiscardResource` and the atomic copies; `WriteBufferImmediate` is allowed in a bundle); those calls are not legal inside a D3D12 bundle and make `Close` return `E_INVALIDARG`. The currently implemented sticky front/back stencil reference is carried back to the parent when the bundle sets it. Draw and other graphics-state commands are still stubs, so this establishes validation and the IR insertion point but does not yet render bundle geometry.
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
| `DiscardResource` / `WriteBufferImmediate` / `AtomicCopyBufferUINT(64)` | discard, immediate-write and atomic-copy commands (section 4.2) |
| `BeginQuery` / `EndQuery` / `ResolveQueryData` | query commands |
| `ExecuteIndirect` | command; signature + argument buffer |
| `BeginRenderPass` / `EndRenderPass` | rendering scope |
| `ExecuteBundle` | inline the bundle's IR at the current state |

The IR is the place to add the trace, validation and "unsupported call" reporting. It also lets the layer replay a
frame deterministically from a capture, which is valuable for comparing against the game.

What is implemented (all of it testable without the game):

- **IR** (`runtime/src/command_list.h`): legacy barriers (transition, UAV, aliasing; split barriers record their end
  half), enhanced barriers (global, buffer, texture; the half with `SyncAfter = SPLIT` is dropped and the other is
  replayed as a full barrier), `ClearRenderTargetView` / `ClearDepthStencilView`, whole view or rectangles,
  `CopyBufferRegion` / `CopyTextureRegion` / `CopyResource`, colour resolves,
  `ClearUnorderedAccessViewUint` / `ClearUnorderedAccessViewFloat` (whole view or rectangles), (D3D-26)
  `DiscardResource`, `WriteBufferImmediate` and `AtomicCopyBufferUINT` / `AtomicCopyBufferUINT64`, and `BeginQuery` /
  `EndQuery` / `ResolveQueryData` (section 4.6).
  Views and rectangles are copied while recording, so the CPU descriptor slot can be rewritten afterwards.
- **Replay** (`runtime/src/vulkan_replay.h`): `prepare` resolves and checks a list without writing anything, `emit`
  writes it. Legacy states map through the section 4.3 table (`vulkan_access`), enhanced barriers through
  `vulkan_sync` / `vulkan_barrier_access` / `vulkan_layout` to `vkCmdPipelineBarrier2`. Whole-view clears are
  transfer clears inside a layout round trip; rectangle clears are `vkCmdClearAttachments` in a dynamic-rendering
  scope, with rectangles clipped to the mip. Copies become `vkCmdCopyBuffer`, `vkCmdCopyBufferToImage`,
  `vkCmdCopyImageToBuffer` and `vkCmdCopyImage`. A placed footprint's row pitch becomes `bufferRowLength`, and
  block-compressed extents stop at the mip edge (D3D12 pads footprints to whole blocks, Vulkan does not; a
  texture-to-texture box of a block-compressed format is rounded up to whole blocks, as on the PC runtime, must fit
  the destination's padded mip, and the Vulkan extent is cut where it reaches an image's edge; Vulkan wants that per
  image, so a copy whose cut extent is neither a block multiple nor at the edge of both images is refused). Each
  range and buffer bound is checked before anything is written. A copy runs in `TRANSFER_SRC`/`TRANSFER_DST`, or
  in place in `GENERAL` when the texture is in `COMMON`: that is D3D12's implicit promotion to a copy state.
  A 3D mip is one subresource. Boxed volume transfers preserve the full footprint's row and slice strides.
  Whole-resource depth/stencil copies issue one `vkCmdCopyImage` per aspect, with a transfer write-to-read/write
  dependency between them; all mips and layers stay batched within each aspect. This avoids stencil corruption
  observed on RX 5700 XT, Windows driver 32.0.21045.5002, when both aspects share one multi-region copy command.
  A standalone synthetic Vulkan multi-mip array reproduces the corruption with validation clean; this is a
  conservative compatibility measure, not evidence that the Vulkan specification requires this dependency.
  `d3d12_stencil_copy_test` checks D24S8/D32S8 across mip/layer counts and odd dimensions against WARP (CTest)
  or `LEONIDA_HOST_ADAPTER` (manual runs), plus the deterministic upload pattern for both planes and sources.
  **Depth ↔ colour copies** (`CopyResource` / `CopyTextureRegion` between `D32_FLOAT` and `R32_FLOAT/UINT/SINT`,
  or `D16_UNORM` and `R16_FLOAT/UNORM`, single-sampled, single-plane depth) copy the bits as the PC runtime does.
  `vkCmdCopyImage` refuses mismatched aspects, so each region goes image → scratch buffer → image
  (`src/vulkan_scratch.h`: device-local chunks kept until the submission completes, on every GPU). Not covered:
  depth with stencil (two planes), a D24 stored as D32, and multisampled textures; they are still refused. A float
  written into a `D32_SFLOAT` image outside [0, 1] is undefined in Vulkan without `VK_EXT_depth_range_unrestricted`,
  where D3D12 keeps the bits; `d3d12_copy_xfmt_test` uses values in [0, 1).
  **D24_UNORM_S8_UINT without a native D24** (the Radeon 880M, AMD hardware generally, RADV): the layer stores
  `D32_SFLOAT_S8_UINT` and makes it behave as D24 (`src/depth24.h`, `src/vulkan_depth24.h`). A footprint
  of the depth plane holds 24-bit UNORM words (bits 0-23; the top byte is unspecified, the PC runtime reads it as
  nothing and writes 0), the stencil plane is plain bytes. An upload converts the words with a compute pass
  (`src/depth24_convert.comp`, integer arithmetic) into a scratch buffer of floats, the nearest float to
  `d / 16777215`, which `vkCmdCopyBufferToImage` then moves into the image; a readback copies the image into the
  scratch and the pass writes the words (a float is multiplied by 16777215 in float and rounded to nearest, ties to
  even, which is what WARP and AMD's D3D12 driver do; NaN and negatives are 0, 1 and above the maximum). Every
  24-bit word makes the round trip unchanged (checked on all 16777216 by `depth24_convert_test`). Texture to
  texture copies move the floats as they are. `ClearDepthStencilView` of any D24 texture, native or stored as D32, clears with the float of the quantised value (an exact 24-bit value, so no driver tie rounding is left),
  so the texture reads back as D3D's quantisation of the clear value. The scratch buffers and descriptors live
  until the submission that used them completes. A copy larger than the device's `maxStorageBufferRange` (128 MiB at
  least) is split into several dispatches, by slices or by rows of one slice. The dispatch runs even under a
  `SetPredication` that would skip copies: the layer runs copies unpredicated (section 4.7) and the conversion
  must go with them, so conditional rendering is ended around it. The D3D11.3 functional spec (3.2.3.6, FLOAT to
  UNORM) says: NaN is 0, the value is clamped to 0..1, multiplied by 2^n - 1 in float, 0.5 is added and the
  fraction dropped, with a tolerance of 0.6 ULP on the integer side; read literally a tie rounds up, and 3.2.2
  lists this conversion as an exception to round-to-nearest-even without fixing ties. WARP and AMD's D3D12 driver
  were measured to round the float product to nearest even (2.5 gives 2, 0.5 gives 0), which the tolerance
  allows; NVIDIA's D3D12 driver (and a native D24 clear in its Vulkan driver) breaks some of those ties the other way;
  the layer follows WARP and AMD there, a measurement and not a rule of the spec. Not covered: a depth test or
  a depth write by the rasteriser works on D32 and may differ from D24 by up to the 24-bit step, and a depth
  texture sampled in a shader sees the float; only copies, clears and readbacks are bit-exact.
  `ResolveSubresource` becomes `vkCmdResolveImage` in the same layouts as a copy: a multisampled colour
  subresource into a single-sampled one of the same size and format. `ResolveSubresourceRegion` uses the same
  Vulkan operation for non-integer colour with average mode, taking a source rectangle (or the entire subresource)
  and a destination offset. A `ResolveSubresourceRegion` of one depth or stencil plane (the subresource's plane
  slice, the same on both sides) runs in a rendering scope of its own with no draws: the multisampled plane is the
  depth or stencil attachment (load `LOAD`, store `NONE`), the destination its resolve target with the mode, the
  render area the source rectangle. Both move from their resolve layouts to `DEPTH_STENCIL_ATTACHMENT_OPTIMAL` and
  back around it; the destination's other plane is not attached and keeps its contents. The format is the depth
  format or the plane's own (`R32_FLOAT`, `R24_UNORM_X8_TYPELESS`, `X24_TYPELESS_G8_UINT`, and the D32S8 pair, untested).
  What the PC runtime accepts was measured on an RTX 5070 (driver 621.24) and on WARP: `MIN` and `MAX` of either
  plane by either format, and `AVERAGE` of the depth plane by its own format only (its `Close` fails for `AVERAGE`
  by the depth format or of stencil); the layer accepts the same. WARP is no reference for two of them: a resolve
  with the format `D32_FLOAT` passes its `Close` but removes the device when run (`DXGI_ERROR_DRIVER_INTERNAL_ERROR`),
  and a one-plane D24S8 resolve writes both planes; the RTX 5070 resolves only the plane named. A scope resolves each pixel in place, so a
  destination offset other than the source rectangle's corner is refused, as is a mode missing from the device's
  `supportedDepthResolveModes`/`supportedStencilResolveModes` (queried at device creation; one plane per scope, so
  `independentResolve` never applies). Refused calls fail `Close` and record nothing. `d3d12_depth_resolve_test`
  compares D32 and D24S8 resolves (typed and typeless resources, whole and rectangle) with the PC runtime; clears give
  every sample of a pixel one value, so the modes over differing samples are not checked yet. `DECOMPRESS`,
  `ResolveSubresource` of depth (which D3D12 does not offer) and resolves between two formats of one typeless family
  remain open. A multisampled texture copies only to one with the same sample count
  and has no footprint.
  Discards, immediate writes and atomic copies replay as section 4.2 describes.
- **UAV clears** (`ClearUnorderedAccessViewUint` / `ClearUnorderedAccessViewFloat`). The CPU handle's slot is read
  while recording as the GFX10 descriptor a shader would load (section 3.4), so the clear works with any
  view-creation code that writes them and with descriptors the game builds itself; the GPU handle is not needed.
  A T# gives the view's format (FORMAT plus DST_SEL, which tells BGRA from RGBA), mip and layers; a 3D view clears
  every slice of its mip. A V# gives a byte range: its address minus the resource's `GetGPUVirtualAddress`, and
  NUM_RECORDS bytes (stride 0, no format: raw), records (a stride: structured) or elements (a format: typed). A view
  that reaches past the end of the buffer stops there, as the hardware's bounds check would.
  - Values follow D3D's rules (`runtime/src/uav_clear.h`): Uint copies the low bits of each value into its channel
    with no conversion, whatever the format's type; Float converts (UNORM/SNORM clamp and round to nearest, NaN is
    0; 16-bit and the packed 11/10-bit floats round to nearest even). A raw or structured view takes value 0 for
    every 32-bit word (Float: its bits). D3D11 calls Float on an integer view invalid and D3D12 does not say; here
    the value is converted (toward zero, clamped, NaN 0) so the result is defined. These are checked against
    hand-computed elements in `d3d12_uav_clear_test`.
  - Buffers: elements of up to 4 bytes repeat in every aligned word, so `vkCmdFillBuffer` writes the aligned middle
    and `vkCmdCopyBuffer` copies the unaligned ends from it (an 8-bit view at an odd offset, say). 8- and 16-byte
    elements are written with `vkCmdUpdateBuffer` (up to 64 KiB) and doubled with copies. A rectangle is an
    element range `[left, right)` on row 0. A view with unaligned ends and less than one whole aligned word between
    them (a few bytes of an 8- or 16-bit format) is refused as `unsupported`.
  - Textures: the values become one element of the view's format, and that element is cleared into the image as
    the value that decodes it in the image's own format, so a typeless texture viewed as UINT gets exactly the bits
    asked for. Exact for every typed-UAV format except the SNORM pattern `0x80..` (Vulkan writes -1.0 as `0x81..`)
    and NaN payloads. A whole view is `vkCmdClearColorImage`; rectangles are `vkCmdClearAttachments` in a
    dynamic-rendering scope, so a texture created with `ALLOW_UNORDERED_ACCESS` also gets `COLOR_ATTACHMENT` usage
    when its format supports it. Both run in place in `GENERAL`, the UAV layout. Rectangle clears of volumes, or of
    formats without attachment support, are refused as `unsupported`.
  - The clear is fenced as the UAV state widened to transfers, so a following UAV barrier, transition, or enhanced
    barrier from `SYNC_CLEAR_UNORDERED_ACCESS_VIEW` sees it.
  - Under `SetPredication`, a whole-view texture clear runs as one rectangle through the attachment path, as
    render-target clears do, so conditional rendering applies. Buffer clears, and texture clears that cannot use an
    attachment, are transfers that conditional rendering does not cover; they run unconditionally and are counted
    with the copies (`unpredicated_copies`).
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
- **COM** (`runtime/shims/d3d12_com.cpp`): `ID3D12GraphicsCommandList8` (the GCL7 chain plus
  `OMSetFrontAndBackStencilRef`), `ID3D12CommandQueue`, `ID3D12CommandAllocator` and `ID3D12Fence1` (with
  `SetEventOnCompletion` events). Every method they do not implement is a stub generated at build time from the
  SDK's `d3d12.h` by `tools/gen_d3d12_stubs.py`. The stub reports its name once, so the vtables follow the SDK
  order exactly and nothing derived from the SDK is checked in. Unknown IIDs are logged and refused. Resources
  are identified by their `ID3D12Resource` pointer, and RTV/DSV CPU handles point at a `TargetView`.
  `OMSetFrontAndBackStencilRef` records its two `UINT`s on the list (and `Reset` clears them), but nothing applies
  them yet: Vulkan can set the faces with `vkCmdSetStencilReference` and a face mask, but only a pipeline that
  consumes the dynamic state can, and this layer has no pipeline or draw path.
- **Device** (`runtime/shims/d3d12_vulkan_device.cpp`): an `ID3D12Device11` (PC ABI, 80 slots) that the Xbox-ABI
  proxy (`d3d12_xs_device.cpp`) forwards to, so slot 14's heap-handle conversion and slot 76's resource creation
  work unchanged. It creates the objects above, plus:
  - descriptor heaps whose slots are 32 bytes (`GetDescriptorHandleIncrementSize`; the console's increments are
    question 4). RTV/DSV slots hold a `TargetView`; shader-visible heaps return a GPU handle, but nothing reads it
    yet;
  - `CreateRenderTargetView` / `CreateDepthStencilView` for whole resources, `TEXTURE2D`, `TEXTURE2DMS`,
    `TEXTURE2DARRAY` and `TEXTURE2DMSARRAY` (a slice range; 1D and 1D-array DSVs wait for 1D textures, D3D-63);
  - `CreateRenderTargetView` / `CreateDepthStencilView` for whole resources, `TEXTURE2D`, `TEXTURE2DMS` and
    `TEXTURE2DARRAY`;
  - `CopyDescriptors` / `CopyDescriptorsSimple`. A descriptor is the bytes of its slot, so a copy is a byte copy:
    the destination and source range lists are walked together, one descriptor at a time, and null size arrays
    mean one descriptor per range;
  - committed buffers (upload and readback heaps mapped persistently and zeroed), 1D textures and 1D arrays, and
    2D textures in the common render-target, depth and texture formats (typeless families become mutable-format
    images), plus non-attachment 3D textures with mipmapped upload, image copy and readback, and multisampled 2D
    textures (single-mip render targets or depth-stencils, in the sample counts the GPU has for the format);
  - packed `R9G9B9E5_SHAREDEXP` textures map to `VK_FORMAT_E5B9G9R9_UFLOAT_PACK32` for uploads and sampled views.
    When `ALLOW_UNORDERED_ACCESS` is requested on a GPU whose `VK_FORMAT_E5B9G9R9_UFLOAT_PACK32` lacks native
    `STORAGE_IMAGE` support (for example NVIDIA), the layer backs the texture with a mutable `VK_FORMAT_R32_UINT`
    image (`MUTABLE_FORMAT | EXTENDED_USAGE` + `VkImageFormatListCreateInfo` `{R32_UINT, E5B9G9R9_UFLOAT_PACK32}`),
    refusing `ALLOW_RENDER_TARGET`, `ALLOW_DEPTH_STENCIL` and `DENY_SHADER_RESOURCE`: `CheckFeatureSupport` advertises
    `TYPED_UNORDERED_ACCESS_VIEW` and `UAV_TYPED_STORE` (plus `UAV_TYPED_LOAD` with `R32_UINT` storage atomics) while
    keeping `RENDER_TARGET` off; SRV and UAV descriptors emit the `E5B9G9R9` T# (`FORMAT 132`, RGB1) and reject
    incompatible casts, while set 0 creates a `VK_IMAGE_USAGE_SAMPLED_BIT` `E5B9G9R9` sampled view and a
    `VK_IMAGE_USAGE_STORAGE_BIT` `R32_UINT` storage view (bound in `as_uint` / `r32ui`, never `as_float`);
    `ClearUnorderedAccessViewFloat` packs RGB9E5 with round-half-up (clamping negative/NaN to 0 and +Inf/overflow to
    65408, bumping on 512) and `ClearUnorderedAccessViewUint` writes 0, matching WARP (`d3d12_r9g9b9e5_uav_test`,
    `d3d12_uav_clear_test`);
  - the descriptor shader samples `B5G5R5A1_UNORM` through a 2D SRV. Opaque and transparent red check bit 15,
    and mixed-channel texels check the five-bit color fields;
  - `R10G10B10_XR_BIAS_A2_UNORM` maps to `VK_FORMAT_A2B10G10R10_UNORM_PACK32`, the same packed 10:10:10:2
    layout; the bias and scale are the application's shader work. Like the PC runtime, it accepts only a
    single-mip, single-sample 2D texture: extra mips, multisampling and 3D are refused in creation, allocation
    and footprints. Format support reports `TEXTURE2D` and `CAST_WITHIN_BIT_LAYOUT`, not the host's display,
    back-buffer-cast or video-processor bits;
  - `CheckFeatureSupport` for feature levels (up to 12_1), architecture, D3D12 options (`OPTIONS`, `OPTIONS1`–`5`,
    `7`–`9`, `11`–`21`), GPU virtual-address support, format support, format plane counts, multisample quality
    levels, shader model and root signature version. `D3D12_OPTIONS8` reports unaligned block textures when the
    Vulkan logical device enables BC texture compression, and `D3D12_OPTIONS12` reports
    `RelaxedFormatCastingSupported = TRUE`, so a committed or placed texture accepts a castable-format list. Other
    queries (including `OPTIONS6` and `OPTIONS10`) are logged and refused until the trace shows what the game asks
    (question 11);
  - Single-sampled `A8_UNORM` 2D textures use one byte per texel in `VK_FORMAT_R8_UNORM`. Their SRVs expose
    (0, 0, 0, A) before applying the caller's component mapping. Format support includes `MIP` when the Vulkan
    image supports multiple levels. Render-target, depth-stencil and UAV creation are unsupported for this format.
  - reserved resources and `GetResourceTiling` (section 4.2);
  - query heaps for occlusion, timestamps and pipeline statistics, and `GetTimestampFrequency` on the queue
    (section 4.6);
  - `GetCopyableFootprints` for buffers, 1D and 1D-array textures, 2D textures and volumes, including BC1–7 and the
    two-plane depth-stencil formats.
    The test compares every output with the PC runtime's; that comparison found that planar
    depth-stencil rows are padded to 512 bytes, not 256.
  - the `1`/`2`/`3` variants as wrappers over the methods above (D3D-28): `CreateCommittedResource1/2/3`,
    `CreateCommandQueue1` and `GetCopyableFootprints1`. `D3D12_RESOURCE_DESC1` drops to `D3D12_RESOURCE_DESC`,
    and the initial state or `D3D12_BARRIER_LAYOUT` is not used, because new textures start `UNDEFINED` and move
    on first use. The newer arguments are answered as the PC runtime answers them. `d3d12_method_variants_test`
    compares each answer below with the host runtime live, measured on an RTX 4090 (driver 621.14):
  - `CreateCommandSignature` (`ID3D12CommandSignature`) and `ID3D12GraphicsCommandList::ExecuteIndirect` (including
    GPU-written `CountBuffer` via Vulkan 1.2 core `vkCmdDrawIndirectCount` / `vkCmdDrawIndexedIndirectCount`).
  - Optional `VK_KHR_acceleration_structure` + `VK_KHR_deferred_host_operations` + `VK_KHR_ray_query`:
    `ID3D12Device5::GetRaytracingAccelerationStructurePrebuildInfo`, `ID3D12GraphicsCommandList4::BuildRaytracingAccelerationStructure`
    (BLAS + TLAS), `vkGetBufferDeviceAddress` in `ID3D12Resource::GetGPUVirtualAddress()`, and RT-aware `UAV`
    barriers (cleanly disabled/skipped on GPUs without RT).

- **DXGI Swapchain** (`runtime/shims/dxgi_swapchain.h`, `runtime/shims/dxgi_com.{h,cpp}`): `IDXGIFactory4` and
  `IDXGISwapChain3` over `VK_KHR_win32_surface` + `VkSwapchainKHR`, exposed via `rt::d3d12::create_dxgi_factory`.
  Swapchain backbuffers are real `ID3D12Resource` COM objects (`BackBufferResource` inheriting from
  `ID3D12ResourceStubs`) registered directly with `VulkanContext`, supporting `CreateSwapChainForHwnd`,
  `GetCurrentBackBufferIndex`, `GetBuffer`, `CreateRenderTargetView`, `Present`, and `ResizeBuffers` (preserving
  explicit buffer dimensions even when the HWND client size is unchanged via an offscreen backbuffer + blit path).

    | Argument | Answer |
    |---|---|
    | a protected resource session | `E_INVALIDARG`; the layer reports no `D3D12_FEATURE_PROTECTED_RESOURCE_SESSION_SUPPORT` |
    | `SamplerFeedbackMipRegion` on a texture | `E_INVALIDARG` from `CreateCommittedResource2/3`; `GetCopyableFootprints1` returns its failure answer (every output all ones) |
    | `SamplerFeedbackMipRegion` on a buffer | ignored |
    | a buffer's initial layout other than `UNDEFINED` | `E_INVALIDARG` |
    | a castable-format count with a null list | no list (`S_OK`) |
    | a castable-format list | accepted on a texture when the formats map to Vulkan formats: the image is mutable and carries a `VkImageFormatListCreateInfo`; a buffer or an unmapped format is `E_INVALIDARG`. A cast whose texel size differs from the resource's is also `E_INVALIDARG` — the host refuses other sizes, and a size-mismatched format list is a Vulkan validation error (`VUID-VkImageCreateInfo-pNext-06722`). The list is kept on the resource, so an SRV whose format is neither the resource format nor in the list leaves a null descriptor. The resource's own format is always accepted and is added to the Vulkan format list when the D3D12 list omits it. A listed same-size cross-family cast, such as `R8G8B8A8_UINT` to `R32_UINT`, is accepted. An RTV in a listed RGBA8 family format on a typed `R8G8B8A8_UNORM` texture clears through the cast, the same reinterpretation a typeless RTV uses; without a list a typed resource still refuses a different-format RTV. `D3D12_OPTIONS12` reports `RelaxedFormatCastingSupported = TRUE` (this layer requires Vulkan 1.3, where the format list is core). `CreatePlacedResource2`'s list reaches its placed image through the same rules (D3D-90) |
    | `CreateCommandQueue1`'s creator ID | ignored |

    `CreatePlacedResource1/2` wrap the placed path above; `2` carries the castable-format list. `CreateHeap1`
    remains a stub.
  - `CreateSampler2` as a wrapper over `CreateSampler` (D3D-28, part 2). Without flags it writes the same S#.
    `D3D12_SAMPLER_FLAG_UINT_BORDER_COLOR` reads the union as integers. `(0,0,0,0)`, `(0,0,0,1)` and `(1,1,1,1)`
    take the S#'s transparent-black, opaque-black and opaque-white types. The S# has no separate integer types,
    since the hardware reads them by the image's format. Vulkan does not: an integer image sampled with a
    `VK_BORDER_COLOR_FLOAT_*` border reads 0. Vonsek found this on an RTX 5080 (#538), and `d3d12_descriptor_test`
    reproduces it on an RTX 4090 when the layer uses the float enums. So the layer marks an integer border in the S#
    with `BORDER_COLOR_PTR = 0x800` (`gfx10::kIntegerBorderPtr`). The hardware reads that pointer only for a table
    colour, and Mesa writes it only for `SQ_TEX_BORDER_COLOR_REGISTER`. The mark gives the integer border its own
    key in the descriptor hash table, so it does not share the float border's `VkSampler`, and `vulkan_sampler`
    makes it `VK_BORDER_COLOR_INT_*`. Any other colour is a table colour, still transparent black like
    `CreateSampler`'s custom colours, with no integer mark yet. That includes the bits of `1.0f`, which as integers
    are not white.
    `D3D12_SAMPLER_FLAG_NON_NORMALIZED_COORDINATES` sets `FORCE_UNNORMALIZED` (word 0, bit 15) and changes nothing
    else; `vulkan_sampler` makes it `unnormalizedCoordinates`. The layer answers `D3D12_OPTIONS17` with
    `NonNormalizedCoordinateSamplersSupported` (core Vulkan, with the same restrictions) and no
    `ManualWriteTrackingResourceSupported`. That is what the RTX 4090 and WARP report. The runtime on a Radeon 680M
    (D3D12 driver 31.0.14084.6002) reports no non-normalized samplers, and leaves every non-normalized description
    unwritten, including those the table below says are written.

    A description the PC runtime refuses leaves the slot as it was. The runtime itself does this, without the
    debug layer, on the RTX 4090 (driver 621.14) and on WARP; the debug layer's message names the rule.
    `d3d12_sampler2_test` compares each row with the host runtime live:

    | `CreateSampler2` input | Answer |
    |---|---|
    | a null description | the slot is left as it was |
    | a flag other than the two | the slot is left as it was |
    | non-normalized, with MIN ≠ MAG, a linear MIP filter, anisotropy or comparison | the slot is left as it was; `MINIMUM`/`MAXIMUM` filters are accepted |
    | non-normalized, with `AddressU` or `AddressV` not `CLAMP`/`BORDER` | the slot is left as it was; `AddressW` is free, as in Vulkan |
    | non-normalized, with `MinLOD`, `MaxLOD` or `MipLODBias` not 0 | the slot is left as it was; `-0` is 0 |
    | non-normalized, a point filter with `MaxAnisotropy` or `ComparisonFunc` set | written; both are ignored (the debug layer warns about `ComparisonFunc`) |
    | both flags | written |

  BC1–7 textures need `textureCompressionBC`, which the device enables when the GPU has it, as it does
  `occlusionQueryPrecise` and `pipelineStatisticsQuery` for the query heaps.
- **Presentation** (`leonida::VulkanPresenter` in `runtime/src/vulkan_present.h`, which
  `rt::d3d12::create_window_presenter` makes for a device and a window). This is the part every
  present front end needs; DXGI or `PresentX` is still open (question 6). It takes a back buffer, an ordinary
  texture in the D3D12 `PRESENT` state, and blits it to an acquired `VkSwapchainKHR` image, scaling when the window
  size differs. The blit goes on the same `VkQueue` after the application's work, so the frame contains everything
  submitted before the present. Details:
  - two frames in flight, and a present semaphore per swapchain image;
  - FIFO when vsync is asked for, otherwise MAILBOX or IMMEDIATE;
  - the swapchain is recreated on resize or when the surface reports it out of date or suboptimal, and a minimised
    window skips presents;
  - a back buffer that was never written is zero-filled like any other texture;
  - `presented()` counts the presents the engine took (`VK_SUCCESS` or `VK_SUBOPTIMAL_KHR`), not ones it reported
    out of date;
  - a recreation is published only when the swapchain, its images and every per-image semaphore exist. On a
    failure the partial objects are released and the next present tries again, so no image is acquired without
    its semaphore;
  - if the blit's submission fails after an image was acquired, an empty batch takes the acquire semaphore and
    signals the frame's fence, and the swapchain is recreated next time to get the image back. The next present
    therefore does not wait forever on a fence nothing signals. If even that batch fails, or a fence wait or reset
    fails, the presenter stops (`ok()` is false) and its owner makes a new one.
  - for `PresentX` (section 4.5): a source rectangle, a blank present, an sRGB swapchain format for an sRGB source,
    and an optional capture of each presented frame to host memory (what `xbox_present_test` reads back).

  The device needs `VK_KHR_swapchain` (and the instance the surface extensions). The standalone device enables
  them when present.

  New textures are not touched at creation. They are tracked as `UNDEFINED`, and the first command that uses a
  part of one zero-fills it (D3D12 committed resources start zeroed; `CREATE_NOT_ZEROED` skips this) and moves it
  to the layout that command expects, so creation needs no submission. Block-compressed textures are not
  zero-filled, because Vulkan cannot clear them; they start undefined until the application uploads them. Each
  submission ends with a barrier to the host, so a readback buffer holds the GPU's writes once the fence wait
  returns. Children hold a reference to the proxy, not
  the inner device, so `GetDevice` always returns the Xbox-ABI object. Each live D3D12 queue leases a distinct
  graphics-family `VkQueue`; the standalone device requests up to 16. The queues still share a submit lock for
  replay state, but Wait-before-Signal between them can make progress (D3D-97). Caller-owned Vulkan devices can
  supply additional handles through `VulkanDeviceInfo::command_queues`; without them they retain legacy single-queue
  behavior. When supplied handles are exhausted, `CreateCommandQueue` fails instead of silently sharing a queue.
  Dedicated compute/transfer
  *families* would also require resource queue-family ownership handling and remain future work.

Descriptors and the bindless set (milestone M2, without the pipelines that will bind the set), from the shader's
side inwards:
- **`runtime/src/vulkan_bindless.h`**: set 0 includes the twenty bindings the translator declares, and
  `DescriptorTable` (`runtime/src/descriptor_table.h`, section 5.1) behind binding 2. One set for the device.
  The translator declares every array as a runtime array, so the counts are the runtime's choice. Details:
  - The device sizes the arrays once, when it creates the set. It does not recreate the set, because pipeline
    layouts are built on `bindless_set_layout()`. `bindless::size_bindings` reads the update-after-bind limits
    (`maxPerStageDescriptorUpdateAfterBind*`, `maxDescriptorSetUpdateAfterBind*`,
    `maxPerStageUpdateAfterBindResources`, `maxUpdateAfterBindDescriptorsInAllPools`), or the plain limits for a
    set without update-after-bind. It starts from the default counts below (4096 sampled 2D images, 1024
    samplers, 1024 2D-array, 256 1D, 256 3D and 256 per storage array). Where a limit is lower, it clamps the
    sampler, storage and 2D arrays, in that order, and the device logs it. Next come the MS pairs that fit. Then it
    grows the arrays, in the same order, towards the maximums:
    - samplers: up to 4096, and at most `maxSamplerAllocationCount` less 64 for samplers outside the set. That
      covers a full 2048-sampler D3D12 heap plus room for literal and static S#s.
    - 2D images: up to 65536.
    - storage arrays: up to 4096 each.

    Growth leaves 64 descriptors of each limit to the other sets of a pipeline layout. It also stays within
    `maxPerSetDescriptors`, which never shrinks the default counts, because a layout past it may still be
    supported. The other arrays keep their default counts. On an RTX 5070 (driver 621.24, 1048576 of each type
    and 4000 samplers) the set has 65536 2D images, 3936 samplers and 4096 storage indices in 2D: 93793
    descriptors, 3-6 MiB at 32-64 bytes a descriptor. The device prints the counts it chose, and
    `bindless_sizing_test` checks the rules on synthetic limits.
  - An image has one index in its dimension (2D, 2D array, 3D, 1D, 2DMS or 2DMS array), shared by the
    bindings for that dimension. The runtime adds sampled-image bindings in pairs for 2DMS and 2DMS array:
    20/21 for floating-point and normalized color, 22/23 for unsigned integers, and 24/25 for signed integers.
    The runtime enables pairs that fit the device's descriptor limits and have a multisampled zero fallback
    of the matching numeric class. A missed key reads zero at texel (0, 0), layer 0, sample 0.
  - The storage arrays are shorter (`BindlessSet::storage_capacity`). Each array holds at most as many images
    as its dimension, and the 2D r32ui array is the same size as the other 2D storage arrays. So an image with a
    storage view gets an index the storage arrays hold, and a read-only one goes above where there is room.
  - Index 0 of every array is a fallback (a 1x1 image of zeros, a nearest clamping sampler), because a miss is
    index 0.
  - Changes are queued and written by `flush()`, one write a slot, so a view that is removed before the next
    flush is never written.
  - `flush()` also uploads the table to its buffer: device-local memory, written from a mapped staging copy when
    the buffers are created (first use or growth) and with `vkCmdUpdateBuffer` for slot changes. Adding a key only
    puts a new entry in an empty slot, and the copy of an entry is not atomic; a running lookup that does not know
    the key cannot resolve to it, which is accepted. Erasing a key compacts its probe cluster, which moves other
    keys' entries, and a table that has grown gets a new buffer behind binding 2: `flush_needs_idle()` says when a
    flush would do either, and the device waits for the queue before flushing in that case.
- **`runtime/src/gfx10_descriptors.h`**: the T#, S# and V# structures (section 5.3), encoded and decoded, with
  their mapping to Vulkan formats, view types, component mappings and samplers. Field positions are Mesa's
  (`src/amd/registers/gfx10-rsrc.json`) and agree with what the translator and the GPU tests already use. Bits
  that only the console's driver could give a meaning (tiling, compression metadata, performance hints) are
  written as zero.
- **Device**: `CreateShaderResourceView` and `CreateUnorderedAccessView` of textures write a T# into the slot,
  `CreateSampler` an S#; each then makes sure set 0 has the view or sampler and the table has the key. Details:
  - The key is the descriptor's bytes, so a copied descriptor needs nothing more, and an unordered-access view
    and a shader-resource view of the same mip are one entry.
  - A D3D12 comparison sampler with function NEVER and an ordinary sampler both encode zero in the hardware
    depth comparison field. The layer marks the NEVER comparison in S# word 3, bit 12 (unused with
    DERIV_ADJUST_EN clear) to give it a separate lookup key and a comparison-enabled Vulkan sampler.
  - A view the layer cannot show (below) leaves a null descriptor, all zeros, which finds the fallback.
  - A typed multisampled color texture has one mip. Its default SRV covers every layer. An explicit
    `TEXTURE2DMS` view reads layer zero, while `TEXTURE2DMSARRAY` selects a range starting at
    `FirstArraySlice` (`ArraySize = UINT_MAX` selects the remaining layers). The format-support query
    advertises `MULTISAMPLE_LOAD` when the format can be sampled with MSAA and its binding pair is available.
  - A texture is named in its T#s by an address of its own, above 4 GiB and 256 bytes apart, that nothing
    dereferences; its entries leave the set and the table when it is destroyed.
  - `VulkanDeviceInfo::bindless` says the Vulkan device has the features this needs: `runtimeDescriptorArray`,
    dynamic indexing of sampled- and storage-image arrays, and update after bind for sampled images, storage
    images and storage buffers. The standalone device enables them where the GPU has them.
  - Every `ExecuteCommandLists` flushes the set before its commands are written.
  - `rt::d3d12::bindless_set` and `bindless_set_layout` hand out the set and its layout: what a pipeline whose
    shader declares set-0 bindings is created with and binds. Nothing in the layer creates such a pipeline yet.
- **Layouts.** A descriptor in set 0 has one image layout for as long as it lives, but D3D12 lets a texture be
  sampled in a shader-resource state and also while promoted from `COMMON`, which is `GENERAL` here (section
  4.3). So with the bindless set, a texture's shader-resource states use `GENERAL` too
  (`ResourceBinding::sampled_in_general`), and every descriptor is written with `GENERAL`. This follows from the
  Vulkan rule that a descriptor's layout matches the image's; no test here can see it broken, because the
  validation layer does not follow which element of a runtime array a shader reads. A depth texture sampled
  while it is also bound read-only (`DEPTH_READ` with a shader-resource state) is in
  `DEPTH_STENCIL_READ_ONLY_OPTIMAL` and is not handled.

A typed single-sampled 3D texture can use a default SRV or an explicit `TEXTURE3D` SRV of the same format.
An `R8G8B8A8_TYPELESS` volume supports explicit `R8G8B8A8_UNORM` and `R8G8B8A8_UNORM_SRGB` SRVs.
The sRGB view decodes RGB and preserves linear alpha. A typeless volume requires an explicit typed view.
The T# records the base depth and the view's mip range. Binding 6 holds the 3D Vulkan view, which has one
array layer. `d3d12_descriptor_test` checks samples at four W coordinates through two views of the same
volume. A nonzero `ResourceMinLODClamp` on a 3D view is logged and leaves a null descriptor.

Known gaps in the descriptor path: buffer shader-resource and unordered-access views and constant-buffer views
(the V# a shader would load for them; the structure is in `gfx10_descriptors.h`, the views are not written);
1D-array SRVs, 1D and 3D storage views, other typeless volume formats, multisampled depth/stencil and typeless
color SRVs, and shader-translator lowering for multisample loads. Single-sampled depth-plane views (`R24_UNORM_X8_TYPELESS`,
`R32_FLOAT_X8X24_TYPELESS`) are a T# of the texture's depth format (8_24_UNORM, X24_8_32_FLOAT) through the
depth aspect (the texture's backing format, so X24_8_32_FLOAT for a D24 texture where the layer stores it as
`D32_SFLOAT_S8_UINT`), each on its own family only (the other family's is a null
descriptor, as D3D12 refuses it), and stencil-plane views (`PlaneSlice` 1, as
`X24_TYPELESS_G8_UINT` or `X32_TYPELESS_G8X24_UINT`) are an 8_UINT T# that set 0 shows through the stencil
aspect, in its float-typed sampled arrays like other single-sampled UINT views. Other gaps include descriptors
the game writes itself, which need
the execute-time scan of section 3.4 (only descriptors made through the device are registered); static samplers
of root signatures, which are not registered yet; the literal S#s of shaders (`literal_samplers` in the index),
which `rt::d3d12::register_samplers` registers without a heap slot but which nothing passes to it yet (the call at
pipeline creation comes with the shader index, #291);
custom border colours, minimum/maximum filters, and anisotropy or mirror-once addressing without the Vulkan
feature, each logged and approximated; the storage arrays hold 255 images a dimension at the default counts, more
where the device has room (4095 2D and 1023 2D-array on an RTX 5070); nothing writes through a
storage view in a D3D12-level test yet (`vk_bindless_test` goes through the storage bindings at the Vulkan
level). The set is flushed at every `ExecuteCommandLists`; a flush that erases keys or moves the table to a larger
buffer (`BindlessSet::flush_needs_idle()`) first waits for the device to be idle and then frees the retired table
buffers, so no running lookup misses (D3D-76). That wait stalls every queue, which is correct but slow if the game
destroys textures every frame; a per-queue wait is open.

A typed 3D color target supports default and explicit RTVs. The default covers mip 0; `FirstWSlice` and
`WSize` select the depth slices in an explicit view's mip. `d3d12_clear_test --volume` checks full and
rectangular clears of overlapping slice ranges against both Windows WARP and expected RGBA bytes.

The compute path, end to end:
- `runtime/src/shader_library.h` finds the translated SPIR-V for a shader the game hands to D3D12 (section 3.6).
  It extracts the FSSM program from the DXBC container (DXIL → XDAT entry 0x0D → XBCH → zlib) and looks up its
  SHA-256 as `fssm_sha256` in `spirv/index.json`. Its own inflate, SHA-256 and JSON reader keep it free of
  dependencies (`shader_library_test`).
  - A container with two different 0x0D payloads (195 of 26,503) is read at the last one, as
    `tools/fssm_unpack.py` does by default and so as `fssm_sha256` was computed.
  - Index entries that share a program are all kept (789 groups of 1,182 entries in the corpus). A PSO takes the
    first entry that fits (for a VS: whose input signature the input layout feeds) and declares no set-0
    bindings, since pipelines do not bind set 0 yet.
  - A malformed index (bad numbers or `\u` escapes) is refused with an error instead of throwing into the shim,
    and an integer field outside 0..2^32−1 reads as 0.
- Root signatures use the model already on `main` (`runtime/src/root_signature.h`, `rt::rootsig`): `parse_rts0`
  reads the serialized blob and `pack()` lays out the 32 user-data dwords by the section 3.5 default mapping,
  refusing anything that would spill.
- **GPU address arena** (section 3.3, when the device has `bufferDeviceAddress`): one host-visible,
  device-addressable buffer, 256 MiB by default (`LEONIDA_ARENA_MB`, at most 4095 MiB, so every VA fits 32-bit
  table pointers).
  - Default-heap buffers are 64 KiB-aligned ranges in it, zeroed through its mapping; their console VA is the offset.
  - Shader-visible descriptor heaps also live in it: the CPU handle is its host mapping and the GPU handle is the VA,
    so descriptor bytes the application writes itself are what shaders load.
  - Upload and readback buffers keep their own memory, and their VA is their device address minus the heap base.
  - Shader programs (FSSM blobs) are copied in as the code base.
- **`CreateRootSignature`** parses the blob with `parse_rts0` and keeps each parameter's place from `pack()`; a
  signature `pack()` refuses (more than 32 user dwords: the spill layout is open) is refused with its reason.
  **`CreateComputePipelineState`** (PC slot 11, Xbox slot 62 through the proxy) finds the shader by its FSSM
  hash, builds a pipeline with the 144-byte push-constant range and registers it for replay. Shaders that declare
  set-0 bindings are refused with a log while pipelines do not declare or bind set 0 (M2).
- **Command list**: `Set{Compute,Graphics}RootSignature` (a different signature clears the arguments, as D3D12
  leaves them undefined; setting the bound one again keeps them, as D3D12 does), `SetPipelineState`, the 32-bit
  constant, root view and descriptor-table setters, `SetDescriptorHeaps` and `ClearState`. A wrong parameter type
  or out-of-range constants makes `Close` return `E_INVALIDARG`.
  - The list holds a reference to every root signature and pipeline state it is given, until `Reset` or its own
    release. D3D12 lets an application release them right after setting them, so the root layout the setters read
    and the pipeline `ExecuteCommandLists` looks up must outlive that. Keeping them alive while the GPU runs the
    submission stays the application's job, as in D3D12.
  `Dispatch` records the block as it is at that moment. Replay binds the pipeline, pushes s0..s31, the heap base
  and the code base, and dispatches.
- **Graphics pipelines and draws (M1).** `CreateGraphicsPipelineState` (PC slot 10) finds the VS and PS by FSSM hash.
  - **Vertex input:** each input-layout element is matched to the VS input signature by semantic name and index,
    and its location is the element's register (the translator's vertex-fetch rule). An input the layout leaves
    unfed is refused, as D3D12 refuses it.
  - **Fixed-function state:** rasterizer, blend and depth-stencil state map one to one. Viewport, scissor,
    topology, stencil references and the blend factor are dynamic, and so are vertex strides when the pipeline has
    vertex bindings. A pipeline without them (a full-screen pass from `SV_VertexID`) does not declare the stride
    state, since nothing would set it (`VUID-vkCmdDraw-pStrides-04913`).
  - **Logic ops** come from `RenderTarget[0]`, as in D3D12, and apply to every target (Vulkan has one per
    pipeline; it skips float formats). **`DepthClipEnable = FALSE`** is depth clamping: D3D12 clamps output depth
    to the viewport's range anyway.
  - **Formats:** dynamic rendering takes the RTV and DSV formats. A draw that leaves a slot unbound for which the
    PSO names a format (D3D12 allows it and discards those writes) uses a variant of the pipeline with that format
    undefined, built on first use and kept with the PSO. Bit i of the variant key is colour slot i, and bit 8 is
    depth/stencil.
  - **Instance step rate 0:** every instance reads element 0, done with a stride of 0 for that slot at draw time.
  - **Features:** independent blend state, wireframe fill, a depth-bias clamp, SRC1 blend factors, logic ops and
    disabled depth clipping need `independentBlend`, `fillModeNonSolid`, `depthBiasClamp`, `dualSrcBlend`, `logicOp`
    and `depthClamp`. The standalone device enables those the GPU has and reports them in
    `VulkanDeviceInfo::features`. A PSO that needs a missing one is refused with a log.
  - **Shader features:** the standalone device also enables `shaderInt64` (declared by 25,803 of 26,503 index
    entries), `shaderFloat64`, `shaderInt16`, `shaderClipDistance` and `shaderCullDistance` where the GPU has them.
    Without one, `vkCreateShaderModule` reports the capability (`VUID-VkShaderModuleCreateInfo-pCode-08740`). The
    index's `features` list is not read yet, so a shader that needs a feature the GPU lacks is not refused up front.
  - **Refused with a log:** hull, domain and geometry stages, stream output, MSAA, set-0 bindings, and two stages
    that both read their code base (question 14, section 5.2). More than eight render targets is `E_INVALIDARG`.
  - **Push constants:** both PSO paths declare the 144-byte block as one range. Its stage flags are the whole
    stage table the graphics path accepts, and the replay pushes with the same flags. A device whose
    `maxPushConstantsSize` is below 144 (#224's `push_constants_need_fallback`) has every PSO refused, with the
    limit in the log, until the uniform-buffer tail exists.

  The command list keeps the input-assembler, viewport, scissor, render-target (read at `OMSetRenderTargets`, as
  D3D12 does), blend-factor and stencil-reference state, including CommandList8's front/back references. Each
  `DrawInstanced` / `DrawIndexedInstanced` records a copy of that state. A draw with no viewport (never set, or
  unbound with `RSSetViewports(0, …)`) or an empty one rasterizes nothing in D3D12. Such a draw is not recorded,
  since Vulkan forbids a 0×0 viewport.

  At replay:
  - **Buffers:** vertex and index views resolve to buffers by VA (arena or upload). A view that runs past its
    buffer is cut at the buffer's end. An unbound vertex slot, or an indexed draw with no index buffer, reads the
    arena's reserved first 64 KiB, which is kept zero (stride 0), so it gets the zeros D3D12 gives.
  - **Attachments:** they move to attachment layouts through the tracker, and the draw runs in a
    dynamic-rendering scope. Its layer count is the smallest layer count of the bound views.
  - **Viewport:** it is flipped (negative height), because translated vertex shaders write D3D clip coordinates;
    front-face winding then matches D3D12's.
  - Only the first viewport and scissor are used, and primitive restart is not modelled yet.

  **Resources and heaps:**
  - A buffer of 0 bytes, or one whose size wraps when aligned, is `E_INVALIDARG`. One the arena cannot hold is
    `E_OUTOFMEMORY`, and so is a shader-visible descriptor heap the arena cannot hold (in host memory its GPU
    handle would be 0).
  - If the arena cannot be created, its buffer is released, so other buffers keep their own memory and get no VA.
  - With `VK_EXT_conditional_rendering`, the arena is created with the conditional-rendering usage, and
    `SetPredication` on a default-heap buffer reads at the buffer's offset in the arena plus the call's offset.
  - `CreateHeap` reserves the requested size in the arena at 64 KiB, and a buffer placed in it gets the heap's
    place plus the placed offset as its console VA. A placed texture keeps its own `VkDeviceMemory` and a null VA,
    and a tiled (`64KB_UNDEFINED_SWIZZLE`) placement is refused with `E_NOTIMPL`. `CreatePlacedResource2`'s
    castable-format list reaches the placed texture's image as the committed path's does (D3D-90).

Known gaps: set 0 in pipelines (the device keeps the bindless set, but no pipeline declares or binds it yet) and
therefore every translated shader that declares bindings; tessellation, geometry and mesh pipelines,
multisampled pipelines, several viewports, primitive restart; `requiredSubgroupSize = 64` for wave64 compute;
the game's present entry point (`PresentX`; the DXGI factory/swapchain and the presenter above are what it
calls), HDR and fullscreen; buffer and constant-buffer views (UAV clears read the descriptor bytes, so they do
not wait on view creation), typeless volume RTVs, depth-stencil resolves to a destination offset other than
the source rectangle's;
default-heap buffers are not zero-filled yet; a fence cannot be signalled to a lower value (Vulkan timelines only
increase; such a signal is reported and dropped); instance step rates above 1 (they need vertex attribute
divisors); read-only DSVs (the depth image always moves to `DEPTH_STENCIL_ATTACHMENT_OPTIMAL`, which matters once
depth SRVs are sampled next to a read-only DSV, M2); the index buffer is bound without its view's size
(`vkCmdBindIndexBuffer2`, Vulkan 1.4, takes one).

Known gaps found in review, not fixed here:
- **Wave64:** a wave64 compute shader runs at whatever subgroup size the driver picks, because no
  `requiredSubgroupSize` is requested (claimed as a follow-up in #313).
- **Queues:** every D3D12 queue submits to one `VkQueue`, which runs batches in order. So a queue that waits on a
  fence another queue signals later (`B->Wait(f, n)` recorded before `A->Signal(f, n)`) holds back that signal,
  and the GPU never progresses (woreksami's probe). Queues on separate Vulkan queues (#306 for copy and compute)
  or holding back a batch until its wait value is submitted would fix it.
- **Fences:** a CPU `ID3D12Fence::Signal` checks only the semaphore's current value, so it can signal below a GPU
  signal that is still pending (`VUID-VkSemaphoreSignalInfo-value-03258`).
- **Lookups:** `buffer_at` searches every resource for each vertex and index view; an ordered map by VA would do
  when it shows.

Build `runtime/` (Vulkan SDK and Python 3 on `PATH`) and run `ctest --test-dir <build> --output-on-failure` for the
CPU tests:
- `command_list_test` covers the recording rules.
- `d3d12_state_map_test` covers both state tables, subresource and barrier-range decoding, layout tracking, the
  subresource ranges `DiscardResource` prepares, Resizable BAR detection, and the memory type order on a no-ReBAR
  heap layout. On Windows it also checks every D3D12 constant the IR copies against the SDK's `d3d12.h`.
- `gfx10_descriptors_test` covers the T#, S# and V#: every field alone at its bit position, round trips, what
  decoding refuses and ignores, the T#s the image test shaders build, sampler words as radv writes them, and the
  format table. `tests/test_gfx10_descriptors.py` compares that table with `tools/gfx10_formats.json`.

The GPU tests read their results back. With `LEONIDA_VK_VALIDATION=1` they run under `VK_LAYER_KHRONOS_validation`,
and any validation error fails them.
- `vk_command_replay_test`, part A: only Vulkan 1.0 functions and no tracker.
- `vk_command_replay_test`, part B: rectangle clears, a depth clear while the stencil plane is read-only, a promoted
  texture, and enhanced barriers including a discard.
- `vk_command_replay_test`, part C: a 4x4x4 volume uploaded from a footprint, copied as a 2x2x2 block and whole,
  and read back; a 4x multisampled target cleared and resolved; and the copies and resolves that must be refused
  (a z range past the volume or its footprint, a footprint past the buffer, a resolve from a single-sampled
  texture or into another size, a multisampled texture copied to a footprint or to another sample count).
- `vk_command_replay_test`, part D: every D3D12 sync bit against every access bit (360 pairs), each as a global,
  a buffer and a texture barrier. Under the validation layer none may be rejected; before
  `vulkan_barrier_scope` there were 462 validation errors on Intel(R) Graphics.
- `vk_command_replay_test`, part F (D3D-26): `WriteBufferImmediate` in all three modes at console VAs, one of them
  in a binding that is a range inside another buffer (as arena resources are); UINT and UINT64 atomic copies
  ordered between the writes so that a missing barrier would copy a stale or a later value; a buffer discard
  (kept), a texture cleared, discarded and cleared again, and a rectangle discard (kept); and the lists that must
  be refused (an address past a binding or in none, no `vkCmdUpdateBuffer`, a copy or dependent range past its
  buffer, an unknown dependent resource, a subresource the resource does not have, and an explicit discard count of
  `UINT_MAX`, which is a range and not the whole resource).
- `d3d12_vulkan_device_test` also runs both atomic copies and the discards through COM, and checks that
  `WriteBufferImmediate` is refused at execution while no buffer has a GPU virtual address on `main`.
- `vk_bindless_test` (run by `runtime/test/shaders/check.py` and `scripts/run_gpu_tests.py`): the translator's
  own lookup against the table and the set. `ps_tex.asm`, a hand-written RDNA2 pixel shader built by `build.py`,
  loads a T# and an S# from a descriptor table and samples. The test draws the gradient with the T# at the fourth
  slot probed, the fallback after the key is erased, the gradient again after the table has grown into a new
  buffer, another view at the same index, and the fallback after removal, with and without the
  descriptor-indexing flags. Last, it draws through a 2D image past 5000 others and a sampler past 2100 others.
- `vk_sampler_test` (run the same way): the S# and T# mip fields as `gfx10_descriptors.h` maps them. `ps_ssharp.asm`
  loads a T# and an S# from the heap and samples a 128x128 two-level texture twice, with `image_sample_lz` (LOD 0)
  and `image_sample` (LOD 1 at this size). The sampler is `vulkan_sampler(decode(S#))` and the view's levels come
  from `decode(T#)`. `check.py`'s `SAMPLER_CASES` cover a point mip filter, MAX_LOD 0, MIN_LOD 1, LOD_BIAS ±1,
  MIP_FILTER_NONE with and without MIN_LOD, and a T# whose BASE_LEVEL is 1. Each pixel is judged against levels
  worked out from the S# and T# bits in Python, independently of the header. On Vulkan, the sampler's minLod and
  mipLodBias apply to an explicit LOD as well. That this matches the console for `image_sample_lz` and
  `image_sample_l` is taken from radv, which writes these Vulkan fields straight into the S#; it was not measured
  on RDNA2 here. The T#'s MIN_LOD (D3D12's ResourceMinLODClamp) is not covered by this test.
- `d3d12_com_test`: drives the COM objects through their vtables. It also checks that a queue `Wait` on a fence
  the CPU has not signalled holds the GPU back.
- `dxgi_swapchain_test`: creates a Win32 HWND and exercises `rt::d3d12::create_dxgi_factory`,
  `IDXGIFactory4::CreateSwapChainForHwnd`, `IDXGISwapChain3::GetCurrentBackBufferIndex`, `GetBuffer` + `GetDesc()`,
  backbuffer `CreateRenderTargetView` + clear + readback verification + `Present`, and `ResizeBuffers` with 0
  validation errors.
- `d3d12_indirect_rt_test`: validates `ID3D12Device::CreateCommandSignature` + `ExecuteIndirect` with a GPU
  `CountBuffer` (`vkCmdDrawIndirectCount`, `count = 0` skipped and `count = 1` rendered) and optional
  `VK_KHR_acceleration_structure` + `VK_KHR_ray_query` (`GetRaytracingAccelerationStructurePrebuildInfo`, BLAS + TLAS
  `BuildRaytracingAccelerationStructure`, and a `RayQueryKHR` compute shader verifying hit vs. miss; clean SKIP on
  GPUs without RT).
- `d3d12_gcl8_test`: the public `ID3D12GraphicsCommandList8` surface: `QueryInterface` for its IID,
  `OMSetFrontAndBackStencilRef(1, 2)` with the recorded values, and the reference count around the query. It
  creates only an allocator and a list, so it needs no GPU or Vulkan loader.
- `d3d12_depth_resolve_test`: depth and stencil `ResolveSubresourceRegion` of a 4x D32 or D24S8 target cleared with
  three overlapping values, whole and by rectangle, compared with the PC runtime on the default adapter and WARP;
  the refused calls must fail `Close` and leave the destination as cleared.
- `d3d12_vulkan_device_test`: the Vulkan device behind the Xbox-ABI proxy, called the way the game will call it.
  It creates heaps through slot 14 and reads their start handles in RAX, creates a render target through slot 76
  with the Xbox layout word, then creates views, records clears and barriers, executes and waits on a fence.
  Results come back through D3D12 itself (`CopyTextureRegion` into a readback heap, then `Map`): the render
  target, the depth and stencil planes, and a never-written texture (which must be zero). A second list uploads a
  pattern and copies it on: whole into a texture created in `COMMON`, a boxed part into a smaller texture, a BC1
  round trip, a 1D texture and a 1D-array layer, and a buffer region. A 4x multisampled render target is cleared through a `TEXTURE2DMS` view and
  resolved into a texture that is read back; the multisampled shapes D3D12 refuses are refused, and so is a
  resolve from a single-sampled texture. `GetCopyableFootprints` is compared with the PC runtime's, and a short
  run repeats on a standalone device, the path `LEONIDA_D3D12=vulkan` takes. That standalone run also covers
  heaps and placed resources: a castable list on a placed texture is cleared through an RTV in a cast format and
  read back against the host's answer, and refused lists match the host's `E_INVALIDARG` (D3D-90).
  The comparison tests run against a PC D3D12 device: the default adapter, or the one named by
  `LEONIDA_HOST_ADAPTER` (`warp`, or part of an adapter's name; the adapter used is printed). Some answers are the
  driver's choice, so pick the host side when two adapters disagree: `GetResourceTiling` splits standard and packed
  mips differently on an NVIDIA, an AMD and a WARP host, and the test accepts any split.
  - Copies against the host (D3D-46, `d3d12_copy_compare.h`): the same scenario runs on the layer and on the host and
    the bytes read back are compared; a failing case prints its name, the region (subresource, mip, slice, plane)
    and the first differing byte. Every case uploads deterministic bytes into a texture and then runs
    `CopyResource`, `CopyTextureRegion` texture to texture (an inner box to the origin, and the origin box into
    another array slice at an offset), footprint to texture (an inner box at an offset) and texture to footprint
    (an inner box at an offset into a readback buffer prefilled with `0xcd`), and reads every subresource back.
    Formats: R8, R8G8 SNORM, RGBA8 (UNORM, SRGB, typeless), BGRA8, R16 UNORM, RG16 and RGBA16 float, R32 UINT,
    RG32 UINT, RGBA32 float, R10G10B10A2, R11G11B10, R9G9B9E5, B5G6R5, B5G5R5A1, D16, D32, D24S8, D32S8 and BC1-BC7
    (SRGB and signed variants included), as a 36x20 texture with three slices and six mips (odd mip sizes), 3D
    textures with boxes in depth, a single-mip single-slice texture and a one-block BC7. Buffers: `CopyBufferRegion`
    at odd offsets and sizes, a `CopyResource` of a 4093-byte buffer, and an upload buffer straight to a readback
    buffer. Only the bytes inside a footprint row are compared (row padding is the driver's; a difference there is
    printed, not failed); every byte a partial copy must leave alone is compared. It found:
    - `R8G8_SNORM`, `R16_UNORM`, `R32G32_UINT`, `R9G9B9E5_SHAREDEXP`, `B5G6R5_UNORM` and `B5G5R5A1_UNORM` textures were
      refused (no Vulkan mapping); they map to `R8G8_SNORM`, `R16_UNORM`, `R32G32_UINT`, `E5B9G9R9_UFLOAT_PACK32`,
      `R5G6B5_UNORM_PACK16` and `A1R5G5B5_UNORM_PACK16`.
    - A texture-to-texture box on a block-compressed texture that reaches the padded edge of a mip (for example a
      box of whole blocks over a 9x5 mip) was refused as `invalid_copy`; the PC runtime accepts it.

    Copies between different mips of one texture are compared too (mips 24, 12, 6, 3, 1; RGBA8, BC1, BC7): sizes that
    are no block multiple, boxes to a padded edge, and boxes that do not fit the destination, which both sides must
    refuse (the host at `Close` with `E_INVALIDARG`, the layer at execution with `invalid_copy`). Known gaps shown
    by it and by the formats, not fixed:
    - **Block-compressed copies into a mip whose size is no block multiple**, for example a 0..8 box into a 6-wide
      mip, a 6x6 mip into a larger one, or a 4x4 box into a 3x3 mip: the host copies whole blocks, `vkCmdCopyImage`
      cannot (its extent must be a block multiple or reach each image's edge), so the layer refuses them with
      `invalid_copy`. A fix needs a staging copy through a buffer. The test prints these as known differences.
    - **`TEXTURE1D`** resources are not supported by the layer (creation, footprints and allocation info refuse
      them), so 1D copies are not compared.
    - On a GPU with a native D24, a readback leaves the unspecified `X8` byte of the depth plane untouched where the PC
      runtime writes 0; the comparison ignores it. Where the layer stores D24 as D32 (see section 3.2) it writes 0.
    A PC driver's copy flushes denormal float depth to zero, so the test keeps its values normal. A
    `LEONIDA_HOST_ADAPTER` that matches no adapter fails the run instead of skipping the comparisons.
  - D24 against the host (`d3d12_depth24_compare.h`, `depth24_convert_test`): the D24S8 row of the copy test
    above runs with random 24-bit words on every GPU, the Radeon 880M's `D32_SFLOAT_S8_UINT` fallback included
    (72 regions, byte for byte, upload, readback and copies). `d3d12_depth24_compare.h` clears 5215 depth values
    (steps, half steps and the floats around them, powers of two, random) as a whole-view clear of a 4x4 texture
    (the first 160) and as one rectangle clear per value of a wide texture, with a stencil value, and compares both
    planes read back. The layer is checked against the reference (`depth24_quantize`), and against the host byte for byte
    where the host agrees with it; NVIDIA's host rounds some ties the other way (478 of 5215 values) and is logged as a host quirk, not a failure. It also clears an
    `R24G8_TYPELESS` texture through a typed D24 DSV (200 values, compared the same way), and reports whether
    shader-resource views of both planes (`R24_UNORM_X8_TYPELESS`, `X24_TYPELESS_G8_UINT`) write the T# of the
    texture's backing format (D3D-75): X24_8_32_FLOAT for the depth plane where the layer stores the texture as
    `D32_SFLOAT_S8_UINT`, 8_24_UNORM where the GPU has D24, and 8_UINT for the stencil plane. An SRV read is not compared with the host (no shader runs in these tests and the host's
    descriptor is opaque). It also runs an upload and readback under a `SetPredication` that skips copies, which the
    layer must still convert and which is compared with the host's unpredicated run. `depth24_convert_test` runs the
    compute pass on the GPU for all 16777216 words (to float, and back), 16777216 float bit patterns against
    `depth24_quantize` (with a 1 MiB range so the copy splits into 65 dispatches), and a pitched copy at an offset
    split by a 1 KiB range. It tests the pass, which needs no D24 absence, on whatever device
    `VK_LOADER_DRIVERS_SELECT` picks; the layer's own D24 path (the D3D12 tests above) is used only on a GPU without
    D24 depth-attachment support, so run those with the AMD driver selected (the Radeon 880M, RADV).
- `d3d12_method_variants_test`: each variant against the method it wraps, on the device directly (the proxy's
  slot 76 is the Xbox texture method, not `CreateCommittedResource3`). It runs 12 creation cases (buffers on the
  three heap types, textures, a volume, a multisampled target, and three shapes the base refuses) through every
  `CreateCommittedResource` variant, which must give the same result, description, heap properties and `Map`
  answer. It runs the edge cases in the table above and compares them with the host's PC runtime when there is
  one; protected sessions are checked on the layer alone, and castable lists have their own success/refusal
  checks. `GetCopyableFootprints1` must match `GetCopyableFootprints` field by field, including the failure
  answer. A texture from `CreateCommittedResource3`
  is uploaded, copied back and compared on a queue from `CreateCommandQueue1`.
- `d3d12_sampler2_test`: `CreateSampler2` against `CreateSampler`, on the slots' bytes. It checks:
  - 10 descriptions without flags, which must be byte for byte the same;
  - 6 UINT and float-bit border colours against their float equivalents (the three built-in UINT colours with
    the integer mark, which must decode to `VK_BORDER_COLOR_INT_*`);
  - 3 non-normalized samplers, which must be `CreateSampler`'s S# with `FORCE_UNNORMALIZED` set;
  - the `D3D12_OPTIONS17` answer;
  - the 22 written-or-refused cases behind the table in section 3.2, compared with the host's PC runtime when
    there is one.
- `xbox_present_test`: `PresentX` and the frame events (section 4.5, "what is implemented"), checked on swapchain pixels.
    there is one. A host that reports no non-normalized samplers is compared only on the 3 cases without the
    flag. The other 19 are then checked against their recorded answers only.
- `d3d12_descriptor_test`: descriptors through the proxy.
  - Copies, checked on the slots' bytes: one run, range lists of different shapes on each side, null size
    arrays, and a destination longer than its source.
  - What views and samplers write, decoded: a view of an uploaded 8x8 texture and a sampler, mips 1 and 2 of a
    BGRA cube with a component mapping, an unordered-access view (the same bytes as the shader-resource view of
    that mip), and the null descriptors of a missing resource, a texture without `ALLOW_UNORDERED_ACCESS`, a view
    past the mips and a volume.
  - UINT border colours (Vonsek's case on #538): an `R8G8B8A8_UINT` texture sampled one tile to the right
    through `CreateSampler2` samplers with a UINT `(0,0,0,0)` and `(1,1,1,1)` border reads 0 and 1 in every
    channel. A float white border on the UNORM texture reads 255; its S# differs from the UINT white one only in
    the integer mark.
  - That set 0 holds them: a compute shader (`test/glsl/cs_sample.comp`, GLSL, compiled with `glslangValidator`;
    skipped without it) is handed a slot's T# and the sampler's S#, looks them up in the hash table as a
    translated shader does, and samples. The sampler wraps in U and the shader samples one tile to the right,
    so a missed S# shows too: the fallback sampler clamps. The layer has no pipelines yet, so the test creates
    this one in Vulkan and binds the device's set. It reads the texture through the view's own slot and through
    a copy of it, and gets the fallback from a slot of bytes no view was made for and, after the texture is
    destroyed, from the view's own bytes.
  - Capacity: 5000 distinct SRVs of that texture and one more, and 2100 distinct wrapping samplers in two
    2048-slot heaps. The last view through the last sampler reads the texture one tile to the right. With the
    sizes of before, both were past their arrays ("array is full") and the shader read the fallback.
- `d3d12_compute_test`: the compute path through the proxy. It creates a root signature from the PC serializer's
  blob, and a PSO through slot 62 from a container shaped like the game's, found in a translator-style index by its
  FSSM hash (an unknown program is refused). It sets root constants, a root UAV and a descriptor table whose
  application-written entry points at an upload buffer, then runs two dispatches that change s0 and the UAV in
  between. The results come back through a readback buffer. The shader, `test/glsl/cs_user_data.comp`, is GLSL
  written to the translator's push-constant and address contract, compiled with `glslangValidator`. It also checks:
  - the dispatches' root signature and PSO are released right after being set, and the same signature is set
    again after the arguments, so the list must hold both and keep the arguments (without the hold the test
    crashes);
  - the index has an entry with set-0 bindings for the same program ahead of the binding-free one, and the PSO
    must take the binding-free one;
  - a device that reports 128 bytes of push constants refuses the PSO.
- `d3d12_draw_test`: milestone M1, through the proxy.
  - **Pipeline:** a VS+PS PSO found by FSSM hash. The input layout is POSITION then COLOR, while the VS has them in
    the opposite register order, so a match by position would fail.
  - **Buffers:** the vertex buffer is copied into the arena, and the index buffer is read from an upload heap.
  - **Draws:** two indexed draws with different root constants and root CBVs, then a back-facing triangle that must
    be culled. Each of these also checks a fix:
    - the first draw uses a PSO with a second RT format and a DSV format, with neither bound (a variant);
    - before the second draw, the same root signature is set again after the arguments, which must keep them;
    - two more draws, through a viewport on the top-right quadrant, read COLOR per instance with step rate 0 (two
      instances that must both read element 0) and from an unbound slot (zeros).
  - **Second pass** on the same target, cleared again: a full-screen triangle from a VS with no inputs
    (`test/glsl/vs_fullscreen.vert`), first in its submission so no vertex buffer was bound before it, through a
    viewport on the bottom-right quadrant; then, where the GPU has `logicOp`, the red quad XORed over the blue
    top-left quadrant, which must read `(255,0,255,0)`.
  - **Refusals:**
    - PSOs that need wireframe fill, independent blend, a depth-bias clamp or depth clamping are refused on this
      device, which enables none of them;
    - nine render targets is refused, at PSO creation and at `OMSetRenderTargets`;
    - 0-byte and wrapping buffer sizes are refused, and a buffer or shader-visible heap larger than the arena runs
      out of memory;
    - a draw with no viewport executes cleanly and draws nothing.
  - **Check:** all 4096 pixels of the read-back target are compared. Where the red and green quadrants land also
    checks the viewport flip and the winding. The test fails with any of these fixes undone: the rebind, step
    rate and logic op show wrong pixels, the variant breaks `dynamicRenderingUnusedAttachments-08912`, and the
    stride state breaks `pStrides-04913`.

  The shaders are GLSL written to the translator's contract (`test/glsl/vs_draw.vert`, `vs_fullscreen.vert`,
  `ps_draw.frag`).
- `d3d12_present_test`: milestone M0. It opens a window and presents 60 frames, each one: reset the list, move
  the back buffer `PRESENT` → `RENDER_TARGET`, clear, move it back, execute, present, signal a fence. It then
  resizes the window and presents 10 more (70 in all), which recreates the swapchain, scales the blit and switches
  to unsynchronised presents. Last, it reads the colour on screen with GDI when the window is in front.
  - The device and the presenter get their Vulkan functions through a driver double in the test. It forwards to
    the real driver, counts the presents the driver reports out of date (the frame counts allow for them), and
    tracks the binary semaphores alive.
  - Through the double, the test reports one successful present as out of date (not counted) and one as
    suboptimal (counted).
  - It fails the second per-image semaphore of a recreation: only the two acquire semaphores may remain, and a
    retry at the same vsync must make a complete swapchain.
  - It fails one blit submission: the next four presents must go through without hanging.
  - Run against the presenter before these fixes, the test fails at the frame count after the resize, where the
    driver reported a real out-of-date present (RTX 5060 Ti, driver 621.24).

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
| Heaps and placed resources | `CreateHeap` reserves the requested size in the arena, so a buffer placed at a 64 KiB-aligned offset is that offset into the arena and its console VA is the heap's place plus the offset. A placed texture gets its own `VkDeviceMemory` and a null VA, as in D3D12. Tiled (`64KB_UNDEFINED_SWIZZLE`) placements are refused with `E_NOTIMPL`; reserved resources are not implemented. |
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
  - A memory type is only tried when its heap can hold the whole allocation (`memory_type_order` in
    `runtime/src/vulkan_memory.h`, beside D3D-96's `resizable_bar`). Without Resizable BAR the only `DEVICE_LOCAL | HOST_VISIBLE` type sits in a
    heap of 256 MiB or less (214 MiB on an RTX 3090 with driver 616.56), so the 256 MiB default arena lands in
    system memory there. Asking that heap for it broke VUID-vkAllocateMemory-pAllocateInfo-01713 even though the
    driver returned `VK_SUCCESS` (reported by CharlyRien).
- M4+: allocate the arena `DEVICE_LOCAL` and stage CPU writes (descriptor heaps, shader blobs, buffer uploads)
  through a host-visible staging buffer, copied before the commands that use them. The hash table is not in the
  arena either: it is a normal storage buffer in `DEVICE_LOCAL` memory, uploaded from a mapped staging copy before
  the commands that read it (D3D-76).
- Upload-heap buffers prefer `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` memory when the GPU has Resizable BAR
  (D3D-96): the CPU writes them through the BAR and the GPU reads them from VRAM. `leonida::resizable_bar`
  (`runtime/src/vulkan_memory.h`) calls it Resizable BAR when such a type sits in a device-local heap larger than
  256 MiB. Without Resizable BAR that type lives in a window of 256 MiB or less (214 MiB on an RTX 3090) that every
  upload buffer would share, so they stay in system memory. If VRAM is full, the allocation falls back to system
  memory. The device logs which one it uses (`[rt] d3d12: upload heaps in ...`), and `LEONIDA_UPLOAD_SYSMEM=1`
  keeps them in system memory to compare the two. Readback buffers keep preferring `HOST_CACHED`.

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
  sample count, and `D3D12_SHADER_COMPONENT_MAPPING` to `VkComponentMapping`. Single-layer 1D textures have default and
  explicit shader-resource views on sampled binding 5. Single-sampled 1D, 2D and 2D-array colour SRVs apply the T# minimum LOD in resource mip coordinates. A clamp inside the exposed mip range uses
  `VK_EXT_image_view_min_lod`. Vulkan allows the image-view minimum to be rounded down to an integer.
  With nearest mip filtering, the UHD 770 and Arrow Lake Xe-LPG returned mip 0 for a 2D minimum of 0.75.
  The Xe-LPG also returned mip 1 for a 2D-array minimum of 1.75. RTX and Radeon returned mips 1 and 2.
  See [Vulkan image-level selection](https://docs.vulkan.org/spec/latest/chapters/textures.html#textures-image-level-selection)
  and the [Intel readbacks](https://github.com/ItsNotRexed/Leonida/pull/1015#issuecomment-7075135647).
  If the logical device lacks the feature, the runtime logs the unsupported view
  and its descriptor resolves to the fallback. A clamp at or below the first mip needs no extension. Above the
  last mip, the view retains its dimensions and maps sampled channels to zero while preserving constant channels.
- Set 0 is a fixed bindless layout, sized once when the device creates it. The defaults are 4096 sampled 2D
  images, 1024 samplers, 1024 2D-array images, 256 1D, 256 3D and 256 per storage dimension/class. The 2D, sampler and
  storage arrays grow to what the device's limits allow, up to 65536, 4096 and 4096 (see `vulkan_bindless.h`
  above). The set is not recreated to grow, because recreating it would invalidate the pipeline
  layouts built on it. Use `UPDATE_AFTER_BIND` and `PARTIALLY_BOUND` so a new view does not require rewriting
  every descriptor; without those features, rebuild the set when it changes.
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

Blob format: `rt::rootsig::parse_rts0` reads the `RTS0` chunk of a serialized root signature, versions 1.0, 1.1
and 1.2. `root_signature_oracle_test` checks it on Windows against blobs the PC `d3d12.dll` serializes. These
layouts are measured, not assumed:
- a descriptor table's payload is `{num_ranges, ranges_offset}`, with the ranges at that offset;
- ranges are 20 bytes, or 24 with the 1.1/1.2 flags dword, in `D3D12_DESCRIPTOR_RANGE(1)` order:
  `{type, count, base register, space, offset}` in 1.0 and `{type, count, base register, space, flags, offset}` in
  1.1/1.2, so the flags come *before* `OffsetInDescriptorsFromTableStart` (a range with `DATA_STATIC` and offset 7
  serializes as `… 8 7`). The parser keeps all six fields;
- root descriptors carry a flags dword from 1.1 on;
- static samplers are 13 dwords (`D3D12_STATIC_SAMPLER_DESC` order). In 1.2 they are 14: `D3D12_STATIC_SAMPLER_DESC1`
  appends `Flags`, i.e. UINT border colours and non-normalized coordinates.

Static samplers take no user-data dwords, so 1.2 changes nothing in `pack`.

**Root-signature flag quirk.** Prior art measured Xbox root-signature descriptors carrying flags `0x03000000`; the
PC serializer rejects them with `E_INVALIDARG` because its flags live in the low 16 bits. Clear the high byte on a
private copy before calling the host serializer/deserializer; never modify the game's descriptor. The meaning of the
bits is unknown. The v1.0 `D3D12SerializeRootSignature` path keeps flags at offset +32, and a descriptor with
`0x3000001` also fails on the PC runtime.

**Version 1.2 on older runtimes.** The in-box `d3d12.dll` of Windows 10 (measured on 22H2, 10.0.19045, without the
Agility SDK) and of Windows 11 23H2 (#361) serializes root signatures only up to 1.1. On Windows 10 it rejects even an
empty 1.2 description with `E_INVALIDARG`, and `D3D12_FEATURE_ROOT_SIGNATURE` reports `HighestVersion` 1.1 (#528).
Version 1.2 differs from 1.1 only in
`D3D12_STATIC_SAMPLER_DESC1::Flags`. So on such a host, the `d3d12_xs` serializer rewrites a 1.2 description whose
static samplers all have `Flags == 0` as the same root signature in 1.1. It refuses one that uses sampler flags (a
UINT border colour or non-normalized coordinates) with a log line, since 1.1 can't express them. The host is probed
once with an empty 1.2 description. `root_sig_test` checks the rewrite field by field and through the host
deserializer.

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
| `ID3D12CommandQueue` | one leased graphics-family `VkQueue` per live queue | Distinct Vulkan queues allow cross-queue waits and signals to make progress. Dedicated compute/transfer families remain open because resource ownership transfers are not yet implemented. `PresentX`, `SuspendX`, `ResumeX` are Xbox queue methods, not in the public vtable; public GDKX samples present only through them (§4.5). Call shapes known, slots **Open**. |
| `ID3D12CommandAllocator` | IR arena + `VkCommandPool` | Reset only when the previous submission's fence is complete. |
| `ID3D12GraphicsCommandList` … `8` | `CommandList` with IR | Replay at `ExecuteCommandLists`; bundles inlined. Supports render passes, mesh dispatch, enhanced barriers; records the GCL8 front/back stencil references (applied only once a pipeline path consumes them). |
| `ID3D12RootSignature` | `RootSignature` + `VkPipelineLayout` | Deserialize the blob with the host `D3D12CreateVersionedRootSignatureDeserializer` when possible; clear the measured `0x03000000` flag byte on a copy first; fall back to a local parser. Build the user-SGPR map and static samplers. |
| `ID3D12PipelineState` | `VkPipeline` + a `PipelineInfo` | Shader hash → SPIR-V + index entry; PSO state → Vulkan state; dynamic state where possible. The BVH builder's compute PSOs arrive at device slot 62 with a plain `D3D12_COMPUTE_PIPELINE_STATE_DESC`. |
| `ID3D12DescriptorHeap` | `DescriptorHeap` (shadow + optional arena range) | CBV/SRV/UAV and sampler heaps can be shader-visible; RTV/DSV are host-only. Start handles use the measured RAX return at heap slots 9/10. Increments are layer-chosen (32/32) pending the console value. |
| `ID3D12Resource` | `Resource` (a range in the arena buffer; a separate `VkImage` for textures) | Committed and placed; reserved is stubbed as linear for now. Xbox textures may arrive through device slot 76 rather than `CreateCommittedResource` (section 4.2). `GetGPUVirtualAddress` returns a console VA (a buffer offset or an image key). |
| `ID3D12Heap` | `Heap` (a reserved VA range) | Buffer ranges are backed by the arena; images get their own `VkDeviceMemory`. Residency calls are no-ops. True buffer/image aliasing inside a heap is **Open**. |
| `ID3D12Fence` / `ID3D12Fence1` | `Fence` over a timeline `VkSemaphore` | `GetCompletedValue` = counter value; `SetEventOnCompletion` = wait thread + Win32 event; CPU `Signal` = host signal. |
| Resource barriers (legacy) | `vkCmdPipelineBarrier` | Per-subresource state tracking; `BEGIN_ONLY`/`END_ONLY` become conservative full barriers at the begin point. |
| Enhanced barriers (`ID3D12GraphicsCommandList7::Barrier`) | `vkCmdPipelineBarrier2` | Direct mapping of sync/access/layout; prefer it when the game uses it. |
| `IDXGISwapChain` / `IDXGIFactory` | Factory/adapter/output: fake objects. `Swapchain` over `VkSwapchainKHR` + Win32 surface: only for the PC-style path | Shim DXGI; do not bind host DXGI. A GDKX title likely creates no swapchain (§4.5); it needs `IDXGIDevice1`→`GetAdapter`→`EnumOutputs` and a `PresentX` + frame-event presenter instead. |
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
- `D3D12_FEATURE_SHADER_MODEL`: answer the shader model the layer can honour, not the driver's maximum. Nothing in
  the runtime compiles DXIL: the game's shaders arrive as SPIR-V from the offline translator, and the only DXIL path
  (the BVH builder) is lowered to standard DXIL, whose baseline is 6.0. A request above it is clamped down to 6.0,
  a request for 5.1 is answered with 5.1 (D3D12's own baseline, which a caller asking for the minimum expects), and a
  value below 5.1 is refused with `E_INVALIDARG`. Measured against the host PC runtime (616.86) on an RTX 5070:
  5.1-6.8 echo the request, 6.9 clamps to 6.8, and `0` is `E_INVALIDARG`; `0x5f` crashed the host runtime, which the
  test records in a comment rather than repeating. WARP answers a 5.1 request the same way.
- `D3D12_FEATURE_ROOT_SIGNATURE`: 1.1, the version the proxy can always produce. A 1.2 blob is rewritten as 1.1 when
  the host's serializer stops at 1.1 and nothing is lost (`d3d12_xs_rootsig.cpp`), and Windows 10's `d3d12.dll` stops
  there, so 1.1 is what the game can rely on everywhere. A higher request is clamped down like the shader model, and
  an undefined value is refused. Measured against the host PC runtime (616.86): 1.0-1.2 echo the request, and a
  wrong `DataSize` is `E_INVALIDARG`; the host crashed on a null buffer, so the test never gives it one.
- `D3D12_FEATURE_D3D12_OPTIONS1`: answer from the Vulkan device's subgroups. `WaveOps` is true only when every
  operation group the wave intrinsics map to is present (`BASIC`, `VOTE`, `BALLOT`, `SHUFFLE`, `ARITHMETIC`),
  because D3D12 promises them all behind that one bool. `WaveLaneCountMin`/`Max` are the subgroup size range
  (`VkPhysicalDeviceSubgroupSizeControlProperties`, or the single `subgroupSize` without those 1.3 properties).
  The exception is an Intel integrated GPU (D3D-121): Vulkan reports 8..32 there, Intel's D3D12 driver a fixed
  16..16, and the layer answers 16..16 like the host. Measured on Xe-LP: UHD 770 (32.0.101.6129) and Raptor Lake-S
  Mobile (101.7092). Discrete Arc keeps the Vulkan range until it is measured (TEST-57).
  `TotalLaneCount` has no Vulkan equivalent: D3D12 counts every lane of the device, and the layer answers one full
  wave, so a game that sizes work from it plans for less parallelism than the GPU has, never more. Measured
  against the host PC runtime (616.86) on an RTX 5070: `S_OK` with `WaveOps` 1, lanes 32..32 and a total of 6144,
  and `E_INVALIDARG` for a `DataSize` one field too short or too long. The host crashed the process on a null
  buffer, so the layer refuses one without asking the host (as with root signatures).
- `D3D12_OPTIONS` families: report the tier the Vulkan feature set can honour (resource binding tier 3 when
  descriptor indexing and update-after-bind are available), plus wave ops, 16-bit, barycentrics, render passes,
  VRS, mesh shader tier, and RT tier 1.1 only if the layer can honour them. The game may branch on these; a false
  "supported" is worse than a false "unsupported" at boot. For `OPTIONS2`–`5`, `7`, `9`, `11`, `13`–`16`, and
  `18`–`21` (D3D-86; leaving `OPTIONS1` to #745, `OPTIONS6`/`10` to #469, `OPTIONS8` to #715, and `OPTIONS12` to
  #693):
  - `OPTIONS4` reports `MSAA64KBAlignedTextureSupported = TRUE` (implemented in `GetResourceAllocationInfo`);
  - `OPTIONS19` maps `SupportedSampleCountsWithNoOutputs` from `framebufferNoAttachmentsSampleCounts`, ties
    `AnisoFilterWithPointMipSupported` to `VulkanDeviceInfo::sampler_anisotropy`, and reports the baseline heap
    limits (`2048` samplers, `1000000` views);
  - `OPTIONS21` reports baseline `D3D12_EXECUTE_INDIRECT_TIER_1_0`, and unimplemented features/tiers (`DepthBounds`,
    `CastingFullyTypedFormat`, MSAA UAVs, independent stencil masks, render passes, RT, mesh shaders, sampler
    feedback, wave MMA, work graphs, GPU upload heaps) stay `FALSE`/`NOT_SUPPORTED`/`TIER_0`. Checked live against
    the host D3D12 runtime on the same adapter (`NVIDIA GeForce RTX 5080`, driver `616.56`, Windows 11): 15/15 feature
    queries compared with `0` mismatches (`MSAA64KBAlignedTextureSupported = TRUE`,
    `SupportedSampleCountsWithNoOutputs = 0x1d`, `LEONIDA_VK_VALIDATION=1` with `0` validation errors).
- `D3D12_FEATURE_FORMAT_SUPPORT`: map to `vkGetPhysicalDeviceFormatProperties`; multisample render-target
  and resolve bits also require the format's image-usage/sample-count query to succeed. D3D12's per-format
  rules then apply, as native D3D12 reports them (RX 9070 XT, AMD 26.11.2, and WARP):
  - `MIP` wherever the image can have more than one level.
  - `SHADER_GATHER` with `SHADER_SAMPLE`.
  - `UAV_TYPED_LOAD` with every typed UAV. Typed UAV reads use the sampled binding (`image_read_typed`), not a
    storage-image read.
  - The UAV atomic bits other than exchange for `R32_UINT`, the one format with the `r32ui` storage view. There
    is no `image_atomic_swap`, and the translator's image atomics cover 2D and 3D textures only.
  - No integer resolve, no 1D block-compressed, no 3D or shader reads of depth formats.
  - TYPELESS formats report texture dimensions and `MIP` only.

  `d3d12_vulkan_device_test` compares these bits with native D3D12 on the same adapter for every format both
  support.
- `D3D12_FEATURE_FORMAT_INFO`: report the DXGI plane count independently of Vulkan texture support. Both
  depth/stencil families (including their view formats), NV12, P010, P016 and NV11 have two planes. The
  device test compares counts and HRESULTs with native D3D12 on the same adapter. If an older host rejects the
  sampler-feedback formats or `A4B4G4R4_UNORM` with `E_INVALIDARG`, the test still requires the layer to return
  `S_OK` and one plane for them.
- `D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS`: query `vkGetPhysicalDeviceImageFormatProperties` for the matching
  colour or depth/stencil attachment usage, plus the transfer and sampled usages used by the resource implementation.
  Vulkan exposes one standard sample pattern, so a supported sample count reports one quality level (quality index 0);
  only D3D12's legal sample counts (powers of two up to 32) are exposed.
- `D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT` (`D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT`): report the
  arena's limits, not the console's.
- `GetResourceAllocationInfo` and `GetCopyableFootprints`: implement D3D12's rules locally; the game uses them to
  suballocate and to lay out uploads. They are deterministic and public. Validate against a host D3D12 device in
  tests if one is available; do not require one at runtime. `GetResourceAllocationInfo` (and `1`/`2`): a buffer
  takes its width rounded up to 64 KiB, aligned to 64 KiB. A texture's size is the memory its `VkImage` needs
  (`vkGetDeviceImageMemoryRequirements`), rounded up to its D3D12 alignment. That alignment is 64 KiB, or 4 MiB with
  MSAA. The small alignments (4 KiB, or 64 KiB with MSAA) apply only when the application asks for them. 4 KiB
  also needs the texture to fit in 64 KiB and not be a render target or depth buffer; if it doesn't qualify, the
  description is invalid (`UINT64_MAX`), as the PC runtime answers. The MSAA 64 KiB alignment has no size limit
  on the PC runtime, so none here either. Texture sizes are the driver's on both runtimes, so only buffer sizes,
  alignments and invalid descriptions are compared with the host. One known difference: a mip count beyond the
  full chain is invalid here (Vulkan can't create it), while the PC runtime still answers with a size.
  A total over several resources that passes 2^64 saturates at `0xFFFFFFFFFFFF0000` here. Hosts disagree on it:
  WARP and the RTX 5070 host of #1204 saturate too, while on Windows 26300.9974 (D3D12Core 10.0.26100.9818) the
  Intel Raptor Lake-S (101.7092) and RTX 5060 Laptop (621.24) hosts wrap, two 2^63 buffers to 0, and WARP on that
  PC still saturates. A wrapped total could make a caller allocate less than one of its resources, so the layer
  keeps saturating, and the test accepts a host that wraps.
- `GetAdapterLuid` (implemented): `VkPhysicalDeviceIDProperties::deviceLUID`, which on Windows is the DXGI
  adapter's LUID; `d3d12_fence_timing_test` finds it with `IDXGIFactory4::EnumAdapterByLuid` (RTX 4090, driver
  621.14). A driver that reports `deviceLUIDValid = VK_FALSE` gets a zero LUID, logged.
- `GetCustomHeapProperties` (implemented): the D3D12 documentation's table for a NUMA adapter, matching the
  `UMA = FALSE` that `D3D12_FEATURE_ARCHITECTURE` reports: `DEFAULT` is `NOT_AVAILABLE`/`L1`, `UPLOAD`
  `WRITE_COMBINE`/`L0`, `READBACK` `WRITE_BACK`/`L0`, `GPU_UPLOAD` `WRITE_COMBINE`/`L1`, all with type `CUSTOM` and
  node masks 1. `CUSTOM`, unknown types and node masks above 1 get an all-zero answer, logged. The test compares
  `DEFAULT`, `UPLOAD` and `READBACK` byte for byte with the host runtime's answer (RTX 4090: identical).
- Residency and power (D3D-33) are no-ops that still check their arguments (`d3d12_vulkan_device.cpp`, tested in
  `d3d12_vulkan_device_test`). Vulkan has no residency API: memory from `vkAllocateMemory` stays usable until
  `vkFreeMemory`, and paging under pressure is the kernel driver's job, so every pageable object is always resident.
  - `MakeResident` and `Evict` return `S_OK`, or `E_INVALIDARG` for a nonzero count with a null array. D3D12
    documents `Evict` as a hint the runtime may defer, so dropping it is allowed.
  - `SetResidencyPriority` (Device1): `S_OK`, or `E_INVALIDARG` for a nonzero count with a null object or priority
    array. Priorities only order what the OS evicts first; any `UINT` is a valid priority.
  - `EnqueueMakeResident` (Device3) signals its fence from the CPU with the given value before returning, since
    the work is already complete. A queue `Wait` on that fence must not hang, which is why the signal stays.
    `E_INVALIDARG` for a null fence, a nonzero count with a null array, or flags other than `NONE` and
    `DENY_OVERBUDGET`; a refused call does not signal.
  - `SetStablePowerState` returns `S_OK`: Vulkan has no clock control, and the result of a frame does not depend
    on the clocks.
  - `SetBackgroundProcessingMode` (Device6): no Vulkan API steers driver background work, so nothing runs. The
    completion event is set at once and `*pbFurtherMeasurementsDesired` is `FALSE`. A mode or action outside the
    SDK's enum is `E_INVALIDARG`.

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
- `ID3D12Resource::GetDesc` returns the description with its defaults filled in, as the PC runtime does: `MipLevels`
  0 becomes the full chain, and `Alignment` 0 becomes the placement alignment, 64 KiB, or 4 MiB for a multisampled
  texture (an alignment the description asked for is kept). **Implemented** and compared field by field with the
  PC runtime and WARP by `d3d12_resource_desc_test`: 25 committed and reserved buffers and 2D, array, volume and
  multisampled textures (measured on an RTX 3060 host and WARP, Windows 10).
- `CreateHeap`: reserve a VA range with the requested size/alignment. `CreatePlacedResource`: assign the resource
  the range at `heap_va + offset`; a buffer uses that range in the arena, an image uses the offset as a key and
  binds its own memory. Vulkan image alignment can exceed D3D12's 64 KiB; because images do not share the arena,
  this mostly affects true aliasing, which is **Open** until tested on the target GPU.
- `CreateReservedResource`: reserve the full size linearly; tile mapping calls are logged, not emulated, for the
  first boot. **Implemented** (`d3d12_vulkan_device.cpp`, `src/resource_tiling.h`, checked by
  `d3d12_vulkan_device_test` and `resource_tiling_test`):
  - `CreateReservedResource`, `CreateReservedResource1` (no protected sessions) and `CreateReservedResource2`
    (castable formats ignored) create a committed resource on a default heap, so every tile is backed and starts
    zeroed (D3D12 tier 2 reads unmapped tiles as zero). D3D12's description rules are kept, as measured on the PC
    runtime: buffers are `ROW_MAJOR`, textures `64KB_UNDEFINED_SWIZZLE` (anything else, or a format without a tile
    shape, is `E_INVALIDARG`; `64KB_STANDARD_SWIZZLE` too, since `CheckFeatureSupport` reports
    `StandardSwizzle64KBSupported` as FALSE), and `GetHeapProperties` on a reserved resource is `E_INVALIDARG`.
    `CheckFeatureSupport` still reports `TiledResourcesTier` as not supported, so a game that checks it creates
    committed resources instead; raising the tier is a decision for when the trace shows what the game does. The Xbox tiled layout word `0x101` is accepted as `64KB_UNDEFINED_SWIZZLE` (**Assumption**:
    that the console uses it for reserved resources too).
  - `GetResourceTiling`: the standard 64 KiB tile shapes, standard and packed mips (a mip stays standard while it is
    wider than half the tile or taller than the tile, depth not counting, as AMD's PC runtime does on RDNA2 and RDNA4
    and so, most likely, the console (D3D-65); an NVIDIA GPU packs from the first mip smaller than a tile on any axis.
    D3D11.3 functional spec 5.9.2.8.5 only requires that
    a mip filling a tile in every dimension stays standard, a smaller one "may or may not be packed, depending on the
    IHV"), each standard mip's tile counts, and the tile order (each array slice holds
    its standard mips, then its packed tiles). The test compares the tile shape and the tile counts of the mips both
    sides keep standard with the PC runtime, and checks that no mip filling a tile is packed, for 18
    buffer, 2D, array, block-compressed and volume layouts, and skips a layout the host refuses: which ones a PC
    driver accepts varies (an RTX 3090 on driver 616.56, at tier 3, refuses texture arrays whose mips reach the
    packed tail, measured by @CharlyRien in #295; an RTX 5070 on 621.01 accepts them). How many tiles the packed mips take is the driver's
    choice: here it is their tight linear size in whole tiles; the PC runtime on an RTX 5070 reported one tile more
    for 1000x300 and 100x100 RGBA8 and a 128^3 R8 volume, and the same count for the other layouts probed. Two
    further measured differences: on array slices after the first, the PC runtime
    writes a tile index into packed subresources instead of the documented `D3D12_PACKED_TILE` (ours follows the
    documentation), and a resource that is not reserved has no tiles (all outputs zero), as on the PC runtime.
  - **Measured on AMD** (RX 9070 XT, RDNA4, driver 32.0.31041.1004, Windows 11, `main` `91b36cf`): the AMD PC
    runtime splits standard and packed mips differently, which `d3d12_vulkan_device_test`'s tiling comparison used
    to report as a failure; the spec leaves the split to the driver, so the comparison now takes any split (a Radeon
    880M on 26.11.2 behaves the same, and WARP keeps every mip standard: it packs nothing, down to 1x1).
    Tile shapes agree in all 17 texture layouts, and 11 of them match exactly; in the other 6, AMD keeps more
    mips as standard tiles before the packed tail:

    | Layout | Tile | Standard mips: NVIDIA (the layer before D3D-65) | AMD (the layer now) |
    |---|---|---|---|
    | 1000x300 RGBA8 sRGB, 3 slices | 128x128 | 2 | 4 |
    | 2048x1024 BC7, 2 slices | 256x256 | 3 | 4 |
    | 1000x600 BC3 | 256x256 | 2 | 3 |
    | 100x100 RGBA8 | 128x128 | 0 | 1 |
    | 4096x64 BGRA8 | 128x128 | 0 | 6 |
    | 256x256x64 RGBA8 volume | 32x32x16 | 3 | 4 |

    An RDNA2 GPU (the Ryzen 7 9800X3D's, driver 32.0.21045.5002) gives the same splits as the RDNA4 one
    ([#374](https://github.com/ItsNotRexed/Leonida/issues/374#issuecomment-7071869388)). With 30 more layouts that
    probe the edges (a mip exactly at or just past half a tile, thin and tall textures, non-square 2D and volume
    tiles), the rule that fits all 47 on both GPUs is: **a mip stays standard while it is wider than half the tile or
    taller than the tile; depth does not count**. An earlier "every axis at most half the tile's largest side" fitted
    the first 17 but missed four tall or deep edge cases. All 47 layouts and AMD's values are in
    `runtime/test/resource_tiling_test.cpp`; the packed tile count is AMD's in 45 of them, and AMD reserves one more
    for the two deep volumes that pack from mip 0. The console is RDNA2, so the layer follows this rule (D3D-65;
    **Assumption**: that the console's runtime splits like the PC's AMD driver).
  - `UpdateTileMappings` / `CopyTileMappings` log once and do nothing. What that loses: two resources mapped to the
    same heap tiles do not alias, and a tile unmapped or remapped keeps its contents. If the trace shows the game
    relies on either, the next step is sparse binding (`sparseBinding`, `sparseResidencyImage2D`) with the heaps
    from D3D-16. `CopyTiles` on the command list is not implemented.
- `D3D12_RESOURCE_DESC::Layout`: the public `ROW_MAJOR` and `64KB_STANDARD_SWIZZLE` values plus the Xbox `0x101`
  tiled value from slot 76. Row-major textures need `VK_IMAGE_TILING_LINEAR` (not usable for RT/DS) or a staging
  copy; `64KB_STANDARD_SWIZZLE` and `0x101` have no direct Vulkan equivalent. If the game uploads console-tiled data,
  the layer needs a detile step or a matching tiled image. **Open**; the ABI value is known, the data layout is not.
- Formats: a `DXGI_FORMAT` → `VkFormat` table, including sRGB and typeless variants. Typeless resources get a
  concrete image format plus `VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT` and a `VK_KHR_image_format_list` when views cast
  within the family; unsupported casts are logged and fall back to the resource format. `CreateCommittedResource3`'s
  castable-format lists are mapped to a `VkImageFormatListCreateInfo`; buffers and unmapped formats are refused.
  `CreatePlacedResource2`'s list is mapped the same way on the placed path (D3D-90). Depth/stencil, BCn and packed
  formats map directly; any format the device rejects is a hard error with the DXGI value.
- `Map`/`Unmap`: upload/readback buffers map their host-visible memory; default-heap resources return
  `E_INVALIDARG` as D3D12 does. A texture in a CPU-visible custom heap maps only with a null pointer (its layout is
  the driver's), as on D3D12.
- `WriteToSubresource`/`ReadFromSubresource` (implemented in `d3d12_vulkan_device.cpp`) complete before they
  return, as D3D12 requires; a read cannot wait for a later submit. Textures in CPU-visible custom heaps are
  created with `VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT` when the device supports that usage with the requested
  image format and flags, moved to
  `GENERAL` and zero-filled on the host at creation, and copied with `vkCopyMemoryToImageEXT`/
  `vkCopyImageToMemoryEXT` while they stay in `GENERAL` (D3D12's `COMMON`). Every other case (no extension, a
  texture a barrier has moved, a default-heap texture, which D3D12 does not allow but the layer copies and logs) is
  a `CopyTextureRegion` from or to a host-visible staging buffer, replayed on the device's queue and waited for, so
  the layout tracker and a first use's zero fill behave exactly as for the application's own copies.
- 96-bit textures (`R32G32B32_FLOAT`/`_UINT`/`_SINT`/`_TYPELESS`, D3D-62) are `R32G32B32A32` images
  (`ResourceBinding::rgb96`), on every GPU. Most GPUs have no optimal-tiling RGB32 image (an RX 9070 XT reports
  no optimal features for `R32G32B32_SFLOAT`), and one that has would not avoid the repack: D3D12 places
  subresources at multiples of 512 bytes and Vulkan requires a copy's buffer offset to be a multiple of the
  12-byte texel (VUID-vkCmdCopyBufferToImage-dstImage-07975). What the application sees stays D3D12's:
  `GetCopyableFootprints` gives 12-byte texels (compared with WARP in `d3d12_rgb96_test`), and a view's T# names
  `GFX10_FORMAT_32_32_32_*`, whose swizzle reads alpha as one, so the image's fourth channel is never read.
  - `CopyTextureRegion` between a texture and a buffer is repacked by the D24 compute pass
    (`src/depth24_convert.comp` modes 2 and 3, `Depth24Job::rgb96`): the same scratch buffers, parts and lifetime,
    with texels of three words in the footprint and four in the scratch, copied as they are (an upload writes the
    fourth word as 0). The words are not interpreted, so NaN payloads, denormals and integers pass unchanged. On
    an RX 9070 XT (AMD 26.11.2) an upload and a readback take about 5 and 2 ms at 1024x1024 and about 20 and 45 ms
    at 4096x4096 (`d3d12_rgb96_test` with `LEONIDA_RGB96_LARGE` set; single runs that vary by a few ms, each the
    first submission of a new queue). The converter is created on every device for this, whether or not the
    GPU has a native D24.
  - Texture-to-texture copies and `CopyResource` are plain image copies, and only between two 96-bit textures.
  - `WriteToSubresource`/`ReadFromSubresource` always take the staged copy, as a D24 texture stored as D32 does:
    a host image copy would not repack.
  - Not done: render targets. WARP creates 96-bit render targets; this layer refuses `ALLOW_RENDER_TARGET` and
    leaves `RENDER_TARGET` out of `FORMAT_SUPPORT`, since a writable fourth channel changes destination-alpha
    blending. `ALLOW_UNORDERED_ACCESS` and `ALLOW_DEPTH_STENCIL` are refused, as on WARP (measured in the test).
    `GetResourceAllocationInfo` gives no size for a description creation refuses. A typed 96-bit texture has
    views of its own format only; the `TYPELESS` one has `FLOAT`, `UINT` and `SINT` views (sampled in
    `d3d12_descriptor_test`). `GetResourceTiling` reports nothing for a 96-bit texture: the tile shape of a
    12-byte texel is not measured.
- Resource flags map to Vulkan usage: `ALLOW_RENDER_TARGET` → `COLOR_ATTACHMENT`, `ALLOW_DEPTH_STENCIL` →
  `DEPTH_STENCIL_ATTACHMENT`, `ALLOW_UNORDERED_ACCESS` → `STORAGE`, `ALLOW_SIMULTANEOUS_ACCESS` → concurrent
  sharing, `DENY_*` → omit the corresponding usage. Buffers get the union of usages their states and flags imply;
  images copy dimension, format, mip levels, array layers and sample count.
- Copies: `CopyBufferRegion` → `vkCmdCopyBuffer` with arena offsets; `CopyTextureRegion` →
  `vkCmdCopyBufferToImage`/`vkCmdCopyImageToBuffer`/`vkCmdCopyImage`, using the `D3D12_PLACED_SUBRESOURCE_FOOTPRINT`
  row pitch and height; `CopyResource` → a whole-subresource copy; colour resolves → `vkCmdResolveImage`;
  depth and stencil resolves → a `vkCmdBeginRendering` scope with a resolve attachment (section 3.2).
  Partial resolve regions to another offset need a shader or a region copy.
- `WriteBufferImmediate` (D3D-26) → `vkCmdUpdateBuffer` of 4 bytes. The address is resolved when the list is
  executed, through `ReplayFunctions::buffer_at`: the buffer whose `ResourceBinding::va` range holds it, at
  `buffer_offset + (address - va)`. These are the same fields and lookup as #291's arena, and the device fills
  `va` once `GetGPUVirtualAddress` lands (#157/#291); until then, a list with an immediate write is refused as
  `unresolved_resource`. `DEFAULT` adds no barrier: it is ordered like any copy, by the application's barriers.
  `MARKER_OUT` waits for every earlier command to finish and sees its writes (`ALL_COMMANDS`/`MEMORY_WRITE` →
  `TRANSFER`/`TRANSFER_WRITE`). `MARKER_IN` only needs earlier commands to have started. Vulkan cannot express
  that, because `TOP_OF_PIPE` in a first synchronisation scope is no stage at all, so it gets an execution-only
  `ALL_COMMANDS` → `TRANSFER` dependency: stricter than D3D12 requires, never weaker. Recording refuses a null
  or unaligned address and an unknown mode (the debug layer's `WRITEBUFFERIMMEDIATE_*` messages, IDs 1090-1092 in
  `d3d12sdklayers.h`).
- `AtomicCopyBufferUINT` / `AtomicCopyBufferUINT64` (D3D-26) → one `vkCmdCopyBuffer` region of 4 or 8 bytes, after
  a `ALL_COMMANDS`/`MEMORY_WRITE` → `TRANSFER`/`TRANSFER_READ|WRITE` barrier. That barrier covers every earlier
  write, those to the dependent resources among them, so the copy sees them all, as D3D12 requires. The dependent
  ranges are checked against their buffers but add nothing a global barrier does not already cover. Recording
  follows the debug layer's `ATOMICCOPYBUFFER_*` rules (IDs 1027-1040): offsets aligned to the copy size and at
  least one dependency. **Assumption:** Vulkan promises no atomicity for a transfer, and the layer relies on a
  naturally aligned 4- or 8-byte copy being a single write on the GPU. This matters only to a reader racing the
  copy on another queue or on the host.
- Clears: a full-subresource clear maps to `vkCmdClearColorImage`/`vkCmdClearDepthStencilImage` (UAV buffers to
  `vkCmdFillBuffer`); a rect clear uses `vkCmdClearAttachments` inside the rendering scope. UAV image clears use
  `vkCmdClearColorImage` in `GENERAL`, and rectangles `vkCmdClearAttachments`. Buffer patterns of other sizes
  need no compute clear: 8/16-bit elements are a 32-bit fill with copied ends, 64/128-bit ones an update doubled by
  copies (section 3.2, UAV clears).
- **Typed UINT RTV clears:** `ClearRenderTargetView` still takes `FLOAT[4]`. For the currently exposed
  `R8G8B8A8_UINT`, `R32_UINT` and `R32G32B32A32_UINT` resources, convert those values numerically into
  `VkClearColorValue::uint32` for both image and rectangle clears; copying float bits gives incorrect stored
  integers without a Vulkan validation error. **Measured:** Windows D3D12 on WARP and AMD RX 5700 XT clamps
  negatives/NaN to zero, positive overflow/infinity to the component maximum, and truncates fractions.
  `d3d12_uint_clear_test` compares full and rectangular D3D12 readbacks with WARP over seven input sets on all
  three formats; `clear_color_test` checks both replay paths without a GPU, including unchanged float values.
  This does not add typed-view reinterpretation of typeless resources or new integer formats.
- **RGBA8 typeless RTV clears:** explicit UNORM, SRGB, UINT, SINT and SNORM RTVs on a
  `R8G8B8A8_TYPELESS` 2D resource with sample count 1, selecting a mip and a contiguous array-layer range.
  `TargetView` captures its format by value; copies and layout tracking keep the resource's base format.
  Reinterpreted full clears use dynamic rendering because `vkCmdClearColorImage` cannot select a view format.
  Rectangle clears use the same typed attachment view. The view-cache key includes the format, and views
  request only color-attachment usage so SRGB does not inherit an unrelated storage requirement.
  Same-format image clears and legacy float-only replay clients retain their existing path.
  On an RX 5700 XT, 13 synthetic full/rectangle readbacks match Windows WARP (UINT exact, normalized
  channels within one byte), including descriptor rewrites, format switches and an SRGB RTV on a UAV resource.
  The same tests fail without this change and when the cache ignores format. CPU replay covers 31 scenarios;
  17 unsupported/invalid frontend views fail `Close` without submission. Other typeless families,
  typed-resource casts, MSAA and 3D typeless RTVs remain unsupported.
  Phase 2 adds 54 mip/array and 12 non-array higher-mip readbacks against WARP, including untouched sentinels.
  D3D-161 adds SINT and SNORM views when the Vulkan format supports color attachments. SINT uses the existing
  integer clear conversion, and SNORM passes floating-point values to the attachment. Signed channels matched
  WARP byte for byte in full and rectangular clears on an RX 9070 XT (driver 32.0.32029.2010) and the integrated
  Radeon in a Ryzen 5 9600X (32.0.21045.29003), including mip and array selections. The SNORM vectors avoid
  half-step rounding ties.
- **Typed RTV clears:** `ClearRenderTargetView` still takes `FLOAT[4]`. For every integer RTV format the device
  exposes (`R8G8`, `R8G8B8A8`, `R16G16`, `R16G16B16A16`, `R32`, `R32G32` and `R32G32B32A32` in `_UINT` and `_SINT`,
  plus `R10G10B10A2_UINT`; `R8`/`R16` `_UINT`/`_SINT` are handled in replay too) convert those values numerically
  into `VkClearColorValue` for both image and rectangle clears; copying float bits gives incorrect stored integers
  without a Vulkan validation error. **Measured:** Windows D3D12 on WARP clamps negatives/NaN to zero, positive
  overflow/infinity to the component maximum (and signed below the minimum), and truncates fractions; signed formats
  use `int32` with the same NaN/overflow rules. `R10G10B10A2_UINT` clamps RGB to 1023 and alpha to 3 (inputs 4, 4.5
  and 2^32 in alpha match WARP). `d3d12_clear_test` compares full and rectangular D3D12 readbacks with WARP across
  UNORM, SNORM, FLOAT, all 15 exposed UINT/SINT render targets (11 value vectors, including 8-/16-/10-/2-bit range
  boundaries) and depth/stencil; `clear_color_test` checks both replay paths for 19 integer formats without a GPU,
  including unchanged float values. SRGB clears are omitted from the comparison: the
  linear→sRGB encoding rounds differently at the `.5` boundary on WARP vs Vulkan (a 1-ulp driver difference).
  **Open:** on NVIDIA RTX 5080 (driver 616.56) clears that land exactly on a `.5` UNORM/SNORM step round down where
  WARP rounds up: `R8G8B8A8_UNORM`/`SNORM` 0.5 store `7f`/`3f` vs WARP `80`/`40`, and a `D16_UNORM` 0.5 depth clear
  stores `0x7fff` vs WARP `0x8000` (all on `main` before this change; same 1-ulp class as SRGB, not addressed here).
  `D32_FLOAT` and `D32_FLOAT_S8X24_UINT` match WARP.
  This does not add typed-view reinterpretation of typeless resources.
- `D3D12XBOX_RESOURCE_FLAG_FORCE_TEXTURE_COMPATIBILITY` (public release note) and DCC: ignore the flag, let the
  Vulkan driver choose compression, and fail loudly if the game aliases a buffer over a compressed image or reads
  raw DCC metadata.
- **G-buffer render targets stay 8-bit UNORM.** Reported by @Silliana in #603 (not reproduced here): the material
  shaders pack data into plain bytes (the normal is octahedral-packed across three bytes of one target, and ID
  bytes sit in another) and the lighting shaders unpack them with a `* 255` and a truncate. If the layer or a test
  gives those targets another format (say f16 for everything), a byte is lost in many pixels: the normals go wrong
  and the specular sparkles. Create them with the format the game asks for (`R8G8B8A8_UNORM` and the like) and don't
  substitute a wider or float format for them.

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
- `DiscardResource` (D3D-26): each discarded image subresource gets a barrier from `oldLayout = UNDEFINED` to the
  layout the tracker already has for it, so the driver may drop the contents while later commands see no layout
  change. The other cases are no-ops, which "contents undefined" allows: buffers, discards with rectangles (only
  part of each subresource), a part never used yet (it is still to be zero-filled on first use), a list replayed
  without a tracker, and one plane of a combined depth-stencil image without `separateDepthStencilLayouts`
  (discarding one plane would also discard the other). A null region is recorded as `whole`; a region is always an
  explicit range, even with `NumSubresources = UINT_MAX`, so its `FirstSubresource` is checked like any other.
- Access bits and sync scopes are not one-to-one. D3D12 lets an access be named with a sync scope whose Vulkan
  stages cannot perform it: `SYNC_ALL_SHADING` with `ACCESS_VERTEX_BUFFER` is valid D3D12, but
  `VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT` needs `VERTEX_ATTRIBUTE_INPUT`, which shading stages are not. Vulkan
  rejects such a barrier. `vulkan_barrier_scope` therefore widens each side's stages by the narrowest stage that
  can perform each access without one. The table is the synchronization2 chapter's "supported access types". The
  stages it adds:
  - `DRAW_INDIRECT` for indirect arguments;
  - `INDEX_INPUT` and `VERTEX_ATTRIBUTE_INPUT` for index and vertex buffers;
  - all shading for constant buffers, SRVs and UAVs;
  - `COLOR_ATTACHMENT_OUTPUT` for render targets;
  - the fragment tests for depth/stencil;
  - `ALL_TRANSFER` for copies and resolves. `CLEAR` alone cannot read, and `ALL_GRAPHICS` contains no transfer stage.

  The access itself is never dropped, so the dependency still covers everything the application named, over a
  slightly larger scope.
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
- `SetEventOnMultipleFenceCompletion` (implemented). With a null event it is one blocking `vkWaitSemaphores` over
  every fence's semaphore, with `VK_SEMAPHORE_WAIT_ANY_BIT` for `D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY` and no flag for
  `ALL`. With an event, the device's fence wait thread counts down one shared waiter (1 for `ANY`, the fence
  count for `ALL`, kept as the API's unsigned `UINT`) and the hit that reaches zero sets the event, so no thread is
  added; later hits stay at zero (`d3d12_fence_waiter_test` checks counts past `INT_MAX`). Values already reached are
  counted before the call returns. Zero fences, other flags, a null array or a fence from elsewhere are
  `E_INVALIDARG`, as on the PC runtime. A fence released while an event waits on it counts as reached for that
  wait, as on the PC runtime (RTX 5070 and WARP): `SetEventOnCompletion` and `ANY` set the event, `ALL` counts it
  as one fence (`d3d12_fence_events_test`).
- The swapchain owns acquire/present semaphores and N frames in flight. A present waits for the frame's
  render-finished semaphore; the next acquire waits for the image's present-complete semaphore.
- Queue types: a direct queue maps to a graphics+compute queue, a compute queue to a compute-only family if one
  exists, a copy queue to a transfer-only family. If the host exposes only one family, serialize and log; async
  compute is a performance feature, not a correctness one.
- `GetTimestampFrequency`/`GetClockCalibration` use `VK_EXT_calibrated_timestamps` where available.
  `GetClockCalibration` (implemented) returns the GPU stamp in device ticks (`1e9 / timestampPeriod` per second,
  the units of `GetTimestampFrequency`) and the CPU stamp in `QueryPerformanceCounter` units. When the device has
  `VK_KHR_calibrated_timestamps` or `VK_EXT_calibrated_timestamps` enabled (`VulkanDeviceInfo::calibrated_timestamps`;
  the standalone device enables whichever the GPU has) and offers the `DEVICE` and `QUERY_PERFORMANCE_COUNTER`
  domains, it is one `vkGetCalibratedTimestampsKHR` call. Otherwise the queue waits until idle, submits a
  `vkCmdWriteTimestamp` and waits for it, and answers with the midpoint of the `QueryPerformanceCounter` reads
  around that submission. This stalls the queue, which is acceptable for a call made once per profiling capture. A
  queue family with `timestampValidBits = 0` fails with `E_FAIL`. `d3d12_fence_timing_test` takes two calibrations
  50 ms apart and compares the GPU and CPU intervals. On an RTX 4090 (driver 621.14) the gap is 2 us with calibrated
  timestamps, 20 us with the fallback, and 0.4 ms with the fallback under the validation layer. The test fails
  above 2 ms.

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

**What exists.** The back end of this path is `leonida::VulkanPresenter` (section 3.2): surface, swapchain,
blit, present, resize. A DXGI shim's `Present` and a `PresentX` implementation would both end in its `present()`.
`GetBuffer(i)` therefore does not need to wrap swapchain images: the buffers can be ordinary committed textures,
blitted at present time, which also frees the game's buffer count and format from the surface's.

**PresentX and frame events (GDKX).** Public Microsoft samples present on Xbox without a DXGI swapchain. Facts
below come from MIT/zlib-licensed code and docs, read directly; none is from the game or the SDK. Sources:
[SimpleDeviceAndSwapChain](https://github.com/microsoft/Xbox-GDK-Samples/tree/main/Samples/IntroGraphics/SimpleDeviceAndSwapChain)
(readme: "Instead of using DXGI for presentation, this uses the new PresentX API"),
[FramePacing](https://github.com/microsoft/Xbox-GDK-Samples/tree/main/Samples/Graphics/FramePacing),
[DirectXTK12 wiki, DeviceResources](https://github.com/Microsoft/DirectXTK12/wiki/DeviceResources) ("The GDKX
platform uses PresentX, so the method WaitForOrigin has to be called each frame loop"),
[directxtk12modelviewer `DeviceResourcesGXDK.cpp`](https://github.com/walbourn/directxtk12modelviewer) (MIT) and
SDL2 `src/render/direct3d12/SDL_render_d3d12_xbox.cpp` (zlib).

*Fact (call sequence in those samples):*

1. `D3D12XboxCreateDevice(nullptr, &params, iid, &device)`: adapter pointer null, a params struct (ring and scratch
   sizes, `D3D12XBOX_DEFAULT_SIZE_BYTES` = use default), IID, out pointer. SDL asks for `ID3D12Device1`.
2. Back buffers are ordinary `CreateCommittedResource` calls with `D3D12_HEAP_FLAG_ALLOW_DISPLAY`, initial state
   `D3D12_RESOURCE_STATE_PRESENT`, `ALLOW_RENDER_TARGET`, 2D, one mip (SDL: `B8G8R8A8_UNORM`). They are the
   game's own resources, not objects owned by a swapchain.
3. DXGI is used only to find the output: QI the device for `IDXGIDevice1`, `GetAdapter`, `EnumOutputs(0)`.
4. `device->SetFrameIntervalX(output, D3D12XBOX_FRAME_INTERVAL_60_HZ, n-1, D3D12XBOX_FRAME_INTERVAL_FLAG_NONE)`:
   4 arguments; the third is the allowed frame latency (SimpleDeviceAndSwapChain passes buffer count minus 1,
   SDL passes 1). Then `device->ScheduleFrameEventX(D3D12XBOX_FRAME_EVENT_ORIGIN, 0, nullptr, ..._FLAG_NONE)`:
   4 arguments (event, a `UINT` passed as 0, a pointer passed as null, flags).
5. Per frame: `device->WaitFrameEventX(D3D12XBOX_FRAME_EVENT_ORIGIN, INFINITE, nullptr, ..._FLAG_NONE, &token)`:
   5 arguments, the last an out pointer to a `D3D12XBOX_FRAME_PIPELINE_TOKEN` (SDL stores it in a `UINT64`; the
   null value is `D3D12XBOX_FRAME_PIPELINE_TOKEN_NULL`). Then render, `ExecuteCommandLists`, then
   `queue->PresentX(planeCount, planes, params)`: 3 arguments (`UINT`, pointer to an array of
   `D3D12XBOX_PRESENT_PLANE_PARAMETERS`, optional pointer to `D3D12XBOX_PRESENT_PARAMETERS`; null in the simple
   samples).
6. Plane fields used: `Token` (the value from the wait), `ResourceCount` (1), `ppResources` (pointer to the
   resource pointer(s)), `pSrcViewRects` (pointer to a rect, HDR case only) and `ColorSpace` (a
   `DXGI_COLOR_SPACE_TYPE`). The HDR path uses 2 planes sharing the token: plane 0 `RGB_FULL_G2084_NONE_P2020`
   (HDR10), plane 1 `RGB_FULL_G22_NONE_P709` (the GameDVR/SDR copy, its own back-buffer set). Non-HDR uses 1
   plane. This agrees with the UMD limit of 2 planes in `d3d12_xs_surface.md` §7.
7. `D3D12XBOX_PRESENT_PARAMETERS` has at least `ImmediateThresholdPercent` (float). Per the FramePacing text: a
   late frame is shown immediately only if scanout has not passed that percentage of the screen; 0 disables
   limited tearing, 100 allows a late flip anywhere in the scan.
8. Suspend: `queue->SuspendX(0)` (one argument, passed 0; no other thread may be rendering). Resume:
   `queue->ResumeX()`, then **re-run** steps 4's `SetFrameIntervalX`/`ScheduleFrameEventX`. The samples
   say frame events must be registered again after resume.
9. Shutdown: `PresentX(0, nullptr, nullptr)` presents a blank screen before resources are released.
10. Statistics: `device->GetFrameStatisticsX(token, typeMask, &count, array)`: 4 arguments. Type bits seen:
    `FRAME_INTERVAL`, `FRAME_EVENT`, `RENDER`, `PRESENT`, `DISPLAY`; `count` is in/out; `S_FALSE` or a short
    count means "not available yet"; no statistics for the first presented frame; render/present/display
    entries carry a `PlaneIndex`; display entries carry `PercentScanned`.
11. Xbox titles do not handle `DXGI_ERROR_DEVICE_REMOVED`/`RESET`.

*Not measured:* vtable slots of all `*X` methods, exact `D3D12XBOX_*` struct layouts and sizes, enum values, and
whether the token is exactly 64 bits in the ABI. The samples give argument counts and field names only.

*Hypothesis (the game is not observed):* a GDKX title follows this pattern, so the DXGI swapchain shim (D3D-2)
is probably **not** on its boot path. What is on it: `D3D12XboxCreateDevice`, `IDXGIDevice1`/`IDXGIAdapter`/
`IDXGIOutput` enumeration, committed `ALLOW_DISPLAY` resources, the frame events and `PresentX`. Without the
frame events and `PresentX` no frame reaches the screen. To confirm: trace which of `CreateSwapChain*` or the
queue's `PresentX` slot the game calls first; check the executable's `dxgi`/`d3d12_xs` imports.

Layer plan:

- Keep the DXGI factory, adapter and output enumeration (adapter `GetDesc`, `EnumOutputs`); no host DXGI.
- `ALLOW_DISPLAY` committed resources are normal `Resource`s that the presenter may blit to a Vulkan
  swapchain image; `PRESENT` state maps per the resource-state table (`PRESENT_SRC_KHR` only for swapchain images).
- Presentation core: `PresentX` with 0-2 planes presents plane 0 into the `VkSwapchainKHR` (plane compose
  stays **Open** for HDR/GameDVR); `SetFrameIntervalX`/`ScheduleFrameEventX`/`WaitFrameEventX` run on a vblank
  timer and hand the token through to the present; `GetFrameStatisticsX` minimal; `SuspendX`/`ResumeX`; 0
  planes = blank screen. Test the methods directly with a fake clock, before slots are known.
- The DXGI `CreateSwapChain*` path stays as the fallback and for PC-style tests, not as the first-boot target.

**What is implemented (the presentation core).** Everything in the plan above that does not need a slot exists as
methods of `rt::d3d12::XboxPresentation` (`runtime/shims/xbox_present.h`), on top of the presenter of section 3.2 and
`leonida::FramePacer` (`runtime/src/frame_pacer.h`, no Vulkan, no Windows headers). The struct layouts, constant
values, HRESULTs and the 64-bit token are **our own provisional definitions**: the samples give call shapes and field
names only, so the real slots, layouts and enum values are still unmeasured and the class is the surface the proxy
will forward to. The queue-side calls ignore which queue they came on (one `VkQueue` behind all of them).

- *Frame pacing* (`FramePacer`, time from an injectable clock; real time is a steady timer, a host vblank source
  can replace it). `SetFrameIntervalX` takes the interval as a period in microseconds and the latency, and starts
  the vblank grid. `ScheduleFrameEventX` arms the origin event, delayed by an offset inside the frame.
  `WaitFrameEventX` returns at the next origin with the next token (1, 2, 3, ...; 0 is the null token) and does not
  return while `latency` presented frames are still waiting for their vblank. A wait that comes late returns at once
  with the newest passed origin; the missed ones are skipped. A finite timeout gives
  `HRESULT_FROM_WIN32(WAIT_TIMEOUT)`; nothing registered, or suspended, gives `HRESULT_FROM_WIN32(ERROR_INVALID_STATE)`.
  `GetFrameStatisticsX` keeps the last 64 frames and reports the event, present and display times (not render, not
  `PercentScanned`); `S_FALSE` with count 0 for what it does not have.
- *`PresentX`*: 0 planes presents a black frame (no token needed). 1 plane blits plane 0's first resource (a
  single-sample 2D colour texture in `PRESENT`) to the swapchain image, limited to `pSrcViewRects[0]` when set
  (bilinear, so the rectangle's outer texels blend with their neighbours). The blit converts between formats; an sRGB
  source gets an sRGB swapchain format and every other source a UNORM one, so the encoded bytes go through unchanged.
  A source the GPU cannot blit gives `DXGI_ERROR_UNSUPPORTED` and presents nothing: the checks are
  `vkGetPhysicalDeviceFormatProperties` (`BLIT_SRC` on the source, `BLIT_DST` on the swapchain format, cached per
  format; the scaling filter is linear only with `SAMPLED_IMAGE_FILTER_LINEAR`, else nearest), plus integer formats,
  which a UNORM swapchain cannot take. Multisampled planes are refused the same way: resolving one into a temporary
  image is not done. 2 planes: plane 0 is shown, plane 1 is
  dropped, and the missing HDR/GameDVR composition is logged once; they must share one token. More than 2 planes,
  null planes, or an empty/outside rectangle give `E_INVALIDARG`. A missing `ALLOW_DISPLAY` heap flag is logged once
  and presented anyway. `ImmediateThresholdPercent` is ignored (logged once when non-zero): a late frame waits for
  its vblank.
- *Tokens*: what the console does with a null, repeated or out-of-order token is **unmeasured**, and a title that
  presents a loading screen with `Token = 0`, presents again with one token, or presents from a render thread out of
  order must not freeze. So those are presented anyway and logged once per kind; only a token that was never handed
  out gives `E_INVALIDARG` (nothing presented). Skipping a token is fine.
- *`SuspendX`/`ResumeX`*: suspend drops the interval, the origin event and the frames in flight, wakes blocked waits
  with `ERROR_INVALID_STATE` and refuses presents (a `PresentX` waiting for a swapchain image polls the suspended
  state every 50 ms and gives up, so `SuspendX` never waits on it); resume allows presents again, but waits fail until
  `SetFrameIntervalX` and `ScheduleFrameEventX` are called again, as the samples do. Tokens keep counting. Calling
  `SetFrameIntervalX` again outside suspend keeps the origin event armed and re-bases the vblank grid on that moment.
- *Not done*: HDR/GameDVR plane composition, resolving multisampled planes, a mutable-format swapchain (an sRGB/UNORM
  alternation recreates the swapchain; `VK_KHR_swapchain_mutable_format` is not enabled on the device), `ImmediateThresholdPercent`, a host-vblank clock (the timer drifts from the
  display; with FIFO presents the swapchain holds the queue to the real refresh anyway), render statistics,
  `PercentScanned`, and the vtable wiring. Slots to measure (all unknown): the device's `SetFrameIntervalX`,
  `ScheduleFrameEventX`, `WaitFrameEventX`, `GetFrameStatisticsX` and the queue's `PresentX`, `SuspendX`, `ResumeX`
  (the list is in the header).
- *Tests*: `frame_pacer_test` (fake clock: interval, offset, late waits, timeouts, latency blocking, tokens,
  statistics, suspend/resume, a real 10 ms timer; no GPU) and `xbox_present_test` (the Vulkan device, reading the
  swapchain image back: blank, solid, two colours, source rectangles, RGBA/BGRA/sRGB sources, 2 planes, token errors,
  statistics, suspend/resume, 30 frames at 60 Hz in real time; skipped, exit 77, without a Vulkan device).

### 4.6 Queries

Implemented (`QueryHeap` in `runtime/shims/d3d12_vulkan_device.cpp`, the query commands in `command_list.h` and
`vulkan_replay.h`), checked on one GPU by `d3d12_vulkan_device_test` under the validation layer:

- **Heaps.** One `VkQueryPool` per `ID3D12QueryHeap`, registered with the `VulkanContext` by its pointer, as
  resources are. `OCCLUSION` heaps are `VK_QUERY_TYPE_OCCLUSION` pools and hold both `OCCLUSION` and
  `BINARY_OCCLUSION` queries, as in D3D12. `TIMESTAMP` and `COPY_QUEUE_TIMESTAMP` heaps are
  `VK_QUERY_TYPE_TIMESTAMP` pools; they are refused when the queue family's `timestampValidBits` is 0.
  `PIPELINE_STATISTICS` heaps count all eleven Vulkan statistics (`0x7ff`), whose result order is the field order
  of `D3D12_QUERY_DATA_PIPELINE_STATISTICS` (IA vertices, IA primitives, VS, GS invocations, GS primitives,
  clipper invocations and primitives, PS, HS, DS, CS), so one 88-byte result per query is copied as is. They need
  `pipelineStatisticsQuery`. `SO_STATISTICS`, `VIDEO_DECODE_STATISTICS` and `PIPELINE_STATISTICS1` (mesh and
  amplification counts) are logged and refused with `E_NOTIMPL`.
- **Recording.** The list keeps D3D12's rules for the calls: timestamps only have `EndQuery`; every other query is
  begun and ended in the same list with the same type, is not begun again while it runs, and is not resolved while
  it runs; resolve offsets are multiples of 8. A list closed with a query still running reports `E_INVALIDARG` and
  is refused at execution. Heap types, index ranges and the destination buffer are checked at
  `ExecuteCommandLists`, when the heap and buffer are known (`ReplayResult::invalid_query`).
- **Replay.** D3D12 queries need no reset; a Vulkan query does. Each `BeginQuery` and each timestamp `EndQuery` is
  preceded by `vkCmdResetQueryPool` of that one query. Nothing in the layer runs inside a render pass yet, so the
  reset is legal where it stands; a draw path that opens a render pass around queries must reset them before it.
  `OCCLUSION` begins with `VK_QUERY_CONTROL_PRECISE_BIT` when `occlusionQueryPrecise` is enabled. A timestamp is
  `vkCmdWriteTimestamp` at `BOTTOM_OF_PIPE` (every earlier command finished). `ResolveQueryData` is
  `vkCmdCopyQueryPoolResults` with `VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT`. A query keeps its result
  across executions until it is begun or written again, as in D3D12.
- **Binary occlusion.** The queue resolves to a scratch buffer in chunks, reduces each 64-bit value to 0 or 1
  with a compute dispatch, and copies it into the destination. This works with readback buffers, which need
  only transfer-destination usage. The helper runs outside conditional rendering so a predicate cannot suppress
  the resolve. `vk_binary_occlusion` checks more than one scratch chunk, gaps, suballocated destinations,
  consecutive submissions and an ordinary occlusion count on a real Vulkan device.
- **Queries never ended.** D3D12 leaves their result undefined; a Vulkan copy with `WAIT_BIT` would wait for them
  forever. Each heap records on the host which queries were ended, in submission order, and a resolve writes zeros
  (`vkCmdFillBuffer`) for the ones that were not.
- **Overlapping ordinary occlusion.** On a graphics queue with compute, the replay opens a new physical query
  whenever the set of active ordinary queries changes. Each logical query keeps a contiguous range in the
  submission's physical pool. Resolving it sums the 64-bit counts on the GPU, then copies the result to the
  destination. Both occlusion compute helpers reuse one scratch buffer per queue. Logical results and in-flight
  submissions retain the pool independently, so later resolves and
  overwrites can be recorded before the earlier commands complete. `vk_overlapping_occlusion` checks crossing
  and nested scopes, more than one scratch chunk, alternating binary and ordinary resolves under a predicate,
  and gated submissions on a GPU.
- **Resolves during a pipeline-statistics query** (D3D-162). The helper dispatches above (binary normalisation,
  occlusion sums) and the statistics sum below would count in the application's `CSInvocations` if they ran inside
  its query. So a statistics query that contains a resolve other than a timestamp copy is split. Its counters go to
  physical `PIPELINE_STATISTICS` segments in a pool of the submission's own; a segment ends before each such resolve
  and a new one begins after it. Its result is the sum of its segments, added per counter by a compute dispatch
  (`statistics_sum.comp`) on the same scratch buffer, which is itself kept out of any statistics query still running.
  Whether a resolve dispatches is only known once the list is planned against the heaps, so every such resolve
  splits the query. `vk_binary_occlusion` checks that a binary resolve and a statistics sum inside statistics queries
  leave every counter at 0. `vk_overlapping_occlusion` puts a native draw in each segment and checks the split sums
  against unsplit queries with the same draws, word by word.
- **Timestamps.** `GetTimestampFrequency` is `1e9 / timestampPeriod` (1 GHz on the test GPU). Every D3D12 queue
  shares one `VkQueue` for now (question 12), so they share one frequency.

Not implemented yet:

- **Overlapping queries.** Replay refuses overlapping pipeline-statistics queries or occlusion overlaps involving
  a binary query with `ReplayResult::unsupported`.
- **Other helper dispatches during a pipeline-statistics query.** Replay still refuses `SetPredication` and
  predicated copies while pipeline statistics are active, since their dispatches would change the reported
  compute-invocation count. The single-list `replay()` helper, which has no segment pool, keeps refusing the
  occlusion resolves above that need a dispatch there.
- Stream-output and mesh-shader statistics. (`GetClockCalibration` is implemented: see section 4.4, "Fences,
  queues and frames in flight".)
- Xbox-specific query heap types (if any) are **Open**; unknown types are logged.

### 4.7 Indirect commands and command signatures

- `D3D12_COMMAND_SIGNATURE_DESC` describes the argument stride and a list of argument types. Simple draw,
  draw-indexed, dispatch and mesh-dispatch signatures map to `vkCmdDraw*Indirect`, `vkCmdDispatchIndirect` and
  `vkCmdDrawMeshTasksIndirectEXT`; use the `*Count` variants (`VK_KHR_draw_indirect_count`, core 1.2) when the
  `ExecuteIndirect` call supplies a count buffer.
- Signatures that change root constants, vertex/index buffers or descriptors per argument cannot be expressed by
  one Vulkan indirect call. Use a compute expansion pass that writes a Vulkan-shaped indirect buffer, or
  `VK_NV_device_generated_commands` on NVIDIA. Start with the simple signatures; the trace will show which
  signatures the game creates.
- Predication (`SetPredication`) maps to `VK_EXT_conditional_rendering` (D3D-18). Done: `GraphicsCommandList::
  SetPredication` records a `Predication` command (`runtime/src/command_list.h`; offset 8-byte aligned, op
  `EQUAL_ZERO` or `NOT_EQUAL_ZERO`, a null buffer turns it off). The replay (`runtime/src/vulkan_replay.h`) begins
  conditional rendering on the captured condition, plain for `EQUAL_ZERO` (Vulkan discards when the word is 0) and
  `VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT` for `NOT_EQUAL_ZERO`, ends it before a new predicate (it cannot
  nest) and at the end of the list (predication does not carry over to the next list). While a predicate is set,
  whole-view clears are replayed as one rectangle with `vkCmdClearAttachments`, because `vkCmdClearColorImage` and
  `vkCmdClearDepthStencilImage` ignore conditional rendering. The standalone device enables the extension and its
  `conditionalRendering` feature when the GPU has them, and then creates buffers with
  `VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT`; without it a predicated list is rejected (`unsupported`).
  `D3D12_RESOURCE_STATE_PREDICATION` is the same bit as `INDIRECT_ARGUMENT` (and `SYNC_PREDICATION` /
  `ACCESS_PREDICATION` the same as `EXECUTE_INDIRECT` / `INDIRECT_ARGUMENT`), so with the extension enabled
  barriers into and out of that state also cover `VK_ACCESS_CONDITIONAL_RENDERING_READ_BIT_EXT` at
  `VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT`, where the predicate is read: without it a `ResolveQueryData`
  into the predicate had no dependency on the read (`vulkan_access`, `vulkan_barrier_scope`).
  Checked by part E of `vk_command_replay_test` on an NVIDIA GeForce RTX 3070, and end to end by
  `d3d12_vulkan_device_test` (an occlusion query resolved into a predicate, through legacy and enhanced barriers)
  on an NVIDIA GeForce RTX 5070.
  The 64-bit value is captured on the GPU at each `SetPredication`: an internal eight-byte copy and a
    one-invocation compute pass OR the two dwords into a private 32-bit condition. Later writes to the source do
    not change that snapshot. A null buffer disables predication; rebinding ends the previous conditional scope
    before the capture, so the internal dispatch is never predicated. Bounds checks require all eight bytes.
    Capture arenas and descriptors remain alive until their command buffer's timeline ticket completes. No
    CPU readback or per-predicate wait is used. This helper requires a compute-capable configured queue family;
    unsupported queues fail preparation instead of falling back to the low word. The synthetic GLSL source and
    embedded SPIR-V are shipped together; normal builds do not need a shader compiler.
    Capturing a new predicate inside an active pipeline-statistics query is rejected as unsupported, because the
    helper's compute invocation would otherwise pollute the application's result; capturing before the query is
    allowed. Virtualising that query is not part of this change.
    Integration notes for #291 and D3D-27: the capture helper binds its own compute pipeline, descriptor set 0
    and push constants. The compute path must restore those bindings afterward, or bind its application state
    before each dispatch. Copies and compute dispatches cannot execute inside dynamic rendering, so render-pass
    integration must place the capture outside the rendering scope while preserving capture at `SetPredication`.
    These are requirements for those future integrations, not compute-dispatch or render-pass support added here.
  `CopyBufferRegion` under a predicate instead uses a byte-granular compute dispatch inside the captured
  conditional scope; both operations, unaligned offsets and lengths, and the full 64-bit predicate are covered.
  Zero-byte copies are no-ops, as checked against native WARP. Unpredicated copies retain `vkCmdCopyBuffer`.
  The compute helper requires enabled Vulkan 1.2 `storageBuffer8BitAccess` (not `shaderInt8`) and a compute queue;
  without it, preparation rejects the predicated buffer copy rather than executing it unconditionally.
  Descriptor windows respect storage-buffer alignment/range and dispatch limits, including arena offsets.
  Copy descriptors are owned by command-buffer slots until the submission timeline completes. Memory dependencies order prior writes,
  subsequent reads and successive windows. Direct and compute queues support the helper; copy queues reject
  active predication. Preparation is serialized under the tracker lock, preserving resource -> tracker -> submit
  lock order and the copy-then-commit tracker. Readback writes accumulate COMPUTE_SHADER for the helper and COPY
  for ordinary copies; mixed lists accumulate both before main's queue-specific host-stage filtering.
  This helper also binds compute pipeline/set 0/push constants, so the
  application-state restoration requirement above applies. A predicated copy inside an active pipeline-statistics
  query is rejected to avoid counting internal compute work.
  Synthetic checks: `predicated_copy_test` covers preparation and rejection paths without a GPU;
  `d3d12_predicated_copy_test` compares 180 byte-exact cases against WARP and an independent literal oracle,
  plus 16 query-to-copy cases: real empty ordinary/binary occlusion queries resolve zero into a default-heap
  predicate at offsets 0/16, both predicate operations govern copies, and executing overwrite copies replace zero
  with a high-word-only value. Four paired controls verify that recapturing the same address changes exactly one
  13-byte destination region; without recapture, later copies keep using the captured zero. Predicate-buffer guards
  and all untouched destination bytes are checked too. These query cases do not cover nonzero visible-draw counts;
  `vk_predicated_copy_test` forces small windows and dispatches, arena prefixes, dependent copies, offset rebinding,
  reset, 32 gated plus 32 reused submissions, failed-submit cleanup and host stages on all three queue kinds.
  Binary and overlapping occlusion fixtures also verify that query results resolve under a false predicate while
  a dedicated copy destination remains unchanged; disabling predication copies into a separate region.
  The embedded shader is generated from `predicated_copy.comp`
  with glslang 16.6.0 (`-V --target-env vulkan1.2 -Os`) and validated for Vulkan 1.2.

  Open:
  - Texture copies, `CopyResource` and resolves. Vulkan's conditional rendering does not cover transfer
    commands, so these still run unconditionally and `PreparedList::unpredicated_copies` counts them.

  - Draws, dispatches and `ExecuteIndirect` are covered by conditional rendering as soon as they are replayed
    (D3D-3, D3D-4, D3D-17); `ClearUnorderedAccessView*` and `ResolveSubresource` need the same check when added.

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
  (`D3D12XBOX_XDXR_STREAM_DESC`), post-build offline statistics, and a `DEHYDRATE` build flag. The Xbox types are
  not public. **Measured** from the DLL's own reads ([xdxr.md](xdxr.md), `runtime/src/xdxr_descs.h`): `BATCHED_DESC2`
  is the public build desc, the postbuild count and pointer, and a 16-byte `XDXR_STREAM_DESC` (`0x50` bytes); the
  stream desc is a size and a pointer to 16-byte typed records. The DLL rejects the offline-statistics postbuild type
  (`0x10`) for every build, so that layout is not in the package.
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
   The layout is the XDXR BVH4 runtime form ([bvh.md](bvh.md) §3.1). `tools/xdxr_bvh.py` encodes it on the
   CPU, with a reference traversal and a validator ([bvh.md](bvh.md) §7a). It is the reference the layer's compute
   builder is checked against.

**Open:** which path the game uses. Resolve by logging the first call to the standard AS methods and by tracing the
builder's command-list usage. If XDXR batched/streaming builds are used, their structs and compression formats need
to be recovered from call sites (or the NDA GDK headers, if a developer has access); they cannot be guessed.

### 4.9 Xbox-specific surface

| Feature | Public evidence | Design |
|---|---|---|
| No ESRAM | Xbox Series X\|S uses unified GDDR6; ESRAM was Xbox One | Do not implement ESRAM-era APIs. The `d3d12_xs.cpp` comment about "ESRAM-era extensions" should be read as Xbox One, not Series. |
| Binning | The GDK topic index names Xbox binning modes (NDA content) | Should be a no-op if the game uses it only as an optimisation: it changes how a render pass uses on-chip memory, not the result. Log and ignore for the first boot; revisit for performance. **Open** whether the game relies on binning for correctness. |
| DCC / texture compatibility | Public release note for `D3D12XBOX_RESOURCE_FLAG_FORCE_TEXTURE_COMPATIBILITY` | Let Vulkan choose compression; fail if the game aliases/reads raw compressed data. |
| PresentX / present planes / frame events | Public GDK samples (MIT/zlib), §4.5 | Likely the only present path of a GDKX title. Presenter + vblank-timed frame events; plane 0 into the Vulkan swapchain. Call shapes known; slots/layouts **Open**. |
| VRS | Public `D3D12_SHADING_RATE` / `D3D12_AXIS_SHADING_RATE` | `VK_KHR_fragment_shading_rate`; `RSSetShadingRate`/image-based rate map to `vkCmdSetFragmentShadingRateKHR`. |
| Mesh shader tier | Public `D3D12_MESH_SHADER_TIER` | `VK_EXT_mesh_shader`; the 9,282 translated mesh shaders make this a hard requirement for any content that uses them. |
| Wave MMA | Public `D3D12_WAVE_MMA_TIER` | Not used: no shader sets `WAVE_MMA` in SFI0 and none contains `v_mfma`/`v_wmma` (all 26,503, `tools/feature_scan.py`). Answer the tier query truthfully; no cooperative-matrix mapping is needed. |
| Enhanced barriers | Public Xbox Series `D3D12_BARRIER_GROUP` | `VK_KHR_synchronization2`; implement directly. |
| XDXR batched/streaming BVH | `XboxBvhBuilder.dll` exports | See 4.8. |

### 4.10 Debug markers and private data

Games and middleware call these every frame (PIX events), so they must be cheap and never fail. Implemented
(D3D-24) in `runtime/shims/d3d12_com.cpp`, `runtime/shims/d3d12_com_object.h`, `runtime/src/command_list.h`,
`runtime/src/vulkan_replay.h` and `runtime/src/vulkan_backend.h`; tested by `runtime/test/d3d12_debug_markers_test.cpp`.

- **Labels, only with `VK_EXT_debug_utils`.** It is an instance extension, so `VulkanDeviceInfo::debug_utils` says
  whether the instance enabled it; `create_vulkan_device_standalone` enables it when the loader lists it. Without
  it every marker call is a no-op (nothing logged, nothing recorded on the queue).
- **Command lists.** `SetMarker`/`BeginEvent`/`EndEvent` are recorded in the IR (`DebugMarker`) in order with the
  other commands and replayed as `vkCmdInsertDebugUtilsLabelEXT`/`vkCmdBeginDebugUtilsLabelEXT`/
  `vkCmdEndDebugUtilsLabelEXT` into the submission's command buffer.
- **Queues.** The same three methods on `ID3D12CommandQueue` call `vkQueueInsert/Begin/EndDebugUtilsLabelEXT` under
  the device's submit lock, so they fall between the submissions issued before and after them.
- **Balancing.** D3D12 does not make a list balance its events; Vulkan requires every end to have a begin.
  One `ExecuteCommandLists` is one command buffer, so a begin in one list may be ended by a later list of the
  same call. An end with nothing open is dropped, and labels still open when the last list ends are ended
  before the command buffer is ended. Queue labels are balanced per queue the same way (an unmatched end is
  dropped; open ones are ended when the queue is destroyed). Command-buffer labels never span two
  `ExecuteCommandLists` calls.
- **Labels.** The `Metadata` argument says what the data is, following WinPixEventRuntime's `pix3_win.h`:
  0 (`PIX_EVENT_UNICODE_VERSION`) a UTF-16 string, 1 (`PIX_EVENT_ANSI_VERSION`) an ANSI string, both cut at the
  terminator or at `Size` bytes and converted to UTF-8; 2 (`PIX_EVENT_PIX3BLOB_VERSION`) PIX's own encoded event,
  which is not decoded and gets the fixed label `PIX event` so the nesting is still visible (question 21). Other
  values get the same label and log once.
- **Private data, on every object.** `SetPrivateData`, `GetPrivateData`, `SetPrivateDataInterface` and `SetName`
  live in the shared `Object` base, so the device, queues, allocators, lists, fences, descriptor heaps and
  resources all have them. `SetPrivateDataInterface` holds a reference until the entry is replaced (by bytes or
  another interface), removed (a null interface or zero size) or the object is destroyed; `GetPrivateData` returns
  the pointer with a reference added, as D3D12 does. A buffer too small for the entry gets `DXGI_ERROR_MORE_DATA`,
  nothing copied and no reference added, and `*pDataSize` set to the size needed, so the caller can retry (as
  vkd3d-proton's private store does). `SetName` stores the name as `WKPDID_D3DDebugObjectNameW`
  (terminator included), which is where D3D12 keeps it. Names are not forwarded to `vkSetDebugUtilsObjectNameEXT`
  yet: one D3D12 object may own several Vulkan objects.

## 5. Producing the shader contract

The table below is the contract from [design.md](design.md#translator-status-and-runtime-contract) and where the
runtime gets each piece. The shader side is already proven by `runtime/test/vk_texture_test.cpp` and the compute,
store, load, mesh, geometry and BVH tests; this layer is the other half.

| Contract | Runtime source |
|---|---|
| `s0..s31` user data | Root signature state at draw/dispatch: root constants copied, root CBV/SRV/UAV addresses and descriptor-table base addresses as 64-bit console VAs. Mapping per section 3.5; absent parameters zero. |
| `heap_base` (dwords 32–33) | The arena's `vkGetBufferDeviceAddress`. One value per command buffer; updated only if the arena is recreated. |
| `code_base` (dwords 34–35) | The console VA of the current shader's FSSM blob in the arena. Per-stage conflict: see below. |
| Descriptor hash table (set 0, binding 2) | `uint mask` + entries of `key[8], index, pad`; FNV-1a over the 8 key dwords; linear probe up to 8; miss = index 0. Implemented in `runtime/src/descriptor_table.h` (`DescriptorTable`). The layer inserts every T#/S# it sees and keeps a fallback image/sampler at index 0. |
| Bindless images (bindings 0, 3, 5, 6) | 2D, 2D-array/cube, 1D, 3D sampled-image arrays; one `VkImageView` per distinct T#. Cube textures are 6-layer 2D arrays and cube arrays 6N-layer ones, as the translator expects. |
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
- The layer must keep the table sparse enough that no live key needs more than eight probes. When all eight slots are occupied,
  the runtime tries to move one resident to the next empty slot within that resident's own probe window.
  A relocation requires an idle flush because it overwrites a live slot. Grow when this fails or when the next
  insertion would exceed 50% load. Update binding 2 after
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

**Measured (D3D-56, `runtime/test/vk_descriptor_bench.cpp`).** Measured on an NVIDIA GeForce RTX 5080 (discrete, `0x10de:0x2c02`, subgroup 32, driver `NVIDIA 616.56`) using 64-bit Vulkan GPU timestamps (`1.000 ns` period) against `leonida::DescriptorTable` (90% T# / 10% S#) with the exact `tools/rdna2spirv.py` `desc_probe` loop. Unless noted, dispatches run `4096 x 64` invocations x `64` iterations (`16.78M` single / `33.55M` T#+S# pair lookups, median of 25 reps); wave-uniform sweeps use `group_stride = iters` (`64`), `lane_stride = 0` so all distinct entries are visited (at 100k: `90,000` T#, `10,000` S#, `100,000` T#+S# pairs, `131,072` miss keys; `uniform_hot_tsharp` repeats `1` key and measures `4.6–4.8 us/1M` in `device_local`):

| Entries (cap / load / bytes / mean probes) | Placement | `uniform_tsharp_hit` (us/1M, Glk/s) | `uniform_pair_ts` (us/1M, Glk/s) | `divergent_tsharp_hit` (us/1M, Glk/s) | `divergent_pair_ts` (us/1M, Glk/s) | `divergent_miss` (us/1M, Glk/s) |
|---|---|---|---|---|---|---|
| 1k (4,096 / 24.4% / 160 KiB / 1.316; 900 T#, 1k pair, 1,024 miss) | `device_local` | 5.892 (169.73) | 5.076 (196.99) | 25.715 (38.89) | 25.193 (39.69) | 98.454 (10.16) |
| 1k | `host_visible` | 6.969 (143.48) | 5.385 (185.69) | 28.193 (35.47) | 24.448 (40.90) | 430.849 (2.32) |
| 1k | `host_device_local` | 5.926 (168.74) | 5.095 (196.25) | 25.862 (38.67) | 25.267 (39.58) | 100.008 (10.00) |
| 10k (65,536 / 15.3% / 2.50 MiB / 1.310; 9k T#, 10k pair, 16,384 miss) | `device_local` | 6.123 (163.33) | 5.368 (186.28) | 37.697 (26.53) | 35.273 (28.35) | 203.882 (4.90) |
| 10k | `host_visible` | 58.622 (17.06) | 19.384 (51.59) | 7626.535 (0.13) | 6392.434 (0.16) | 135855.749 (0.01; `4096x17`, n=9) |
| 10k | `host_device_local` | 6.124 (163.28) | 5.344 (187.11) | 36.499 (27.40) | 34.738 (28.79) | 202.518 (4.94) |
| 100k (2,097,152 / 4.8% / 80.00 MiB / 1.088; 90k T#, 100k pair, 131,072 miss) | `device_local` | 5.432 (184.09) | 4.766 (209.84) | 34.504 (28.98) | 39.566 (25.27) | 217.159 (4.60) |
| 100k | `host_visible` | 151.640 (6.59) | 129.732 (7.71) | 11548.004 (0.09) | 2336.777 (0.43) | 163852.450 (0.01; `4096x15`, n=9) |
| 100k | `host_device_local` | 5.468 (182.87) | 4.764 (209.93) | 72.233 (13.84) | 99.000 (10.10) | 690.168 (1.45) |

Three consequences for the runtime and table layout (additional GPU measurements follow, including AMD RDNA3):

1. **In `DEVICE_LOCAL` VRAM, the hash probe itself is cheap at all three sizes:** wave-uniform full-table sweeps run at `5.4–6.1 us / 1M` (`3.0–3.7 us / 1M` net after subtracting the `baseline_uniform` key-read dispatch), and divergent lookups hit `25.7–37.7 us / 1M` (`17.2–28.9 us / 1M` net).
2. **`BindlessSet::create()` currently allocates PCIe `HOST_VISIBLE | HOST_COHERENT` system memory (`runtime/src/vulkan_bindless.h`):** once working sets exceed cache at 10k–100k entries, `host_visible` slows down by **9.6x–28x** even on wave-uniform full sweeps, **202x–335x** on divergent T# hits (`59x–181x` on T#+S# pairs), and **666x–755x** on 8-probe misses. `host_device_local` (ReBAR `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`) matches `device_local` across all 1k–10k cases and 100k uniform sweeps, and is 2.1x–3.2x slower than pure `device_local` on 100k divergent lookups (80 MiB, spilling L2) while staying 24x–237x faster than `host_visible`. Staging binding 2 to `DEVICE_LOCAL` (or preferring `DEVICE_LOCAL | HOST_VISIBLE` when small) avoids this cliff without touching the shader contract.
3. **Power-of-two linear probing with an 8-probe hard limit inflates large tables:** `DescriptorTable` reaches only `15.3%` load at 10k (`2.50 MiB`) and `4.8%` load at 100k (`80.00 MiB`, `max_probes=5`) because any 9-probe collision forces a full 2x capacity double. Bounded-window displacement (for example Robin Hood / backshift within the existing 8-slot window) can raise load factor without changing translated shaders.

**Measured (D3D-56, AMD RDNA3).** The same benchmark and 64-bit timestamps on an AMD Radeon RX 7900 XTX (discrete, `0x1002:0x744c`, subgroup 64, driver `AMD proprietary 26.10.1 (LLPC)`, Vulkan 1.4.359), Windows 11 build 26200, revision `2aed144`. Dispatches, iterations and reps match the NVIDIA run above, but the table code does not: this revision includes `bc8e604`'s bounded slot relocation, so at 100k it settles at `524,288` slots (`20.00 MiB`, `19.07%` load, `max_probes=8`), where the NVIDIA table above - taken before that change - grew instead (`2,097,152` slots, `80.00 MiB`, `4.8%`, `5`). The two tables therefore describe different `DescriptorTable` code: compare rows within each table, not across the two, and read no vendor difference into the 100k rows.

| Entries (cap / load / bytes / mean probes) | Placement | `uniform_tsharp_hit` (us/1M, Glk/s) | `uniform_pair_ts` (us/1M, Glk/s) | `divergent_tsharp_hit` (us/1M, Glk/s) | `divergent_pair_ts` (us/1M, Glk/s) | `divergent_miss` (us/1M, Glk/s) |
|---|---|---|---|---|---|---|
| 1k (4,096 / 24.4% / 160 KiB / 1.316) | `device_local` | 6.220 (160.76) | 5.705 (175.27) | 15.285 (65.42) | 14.051 (71.17) | 50.461 (19.82) |
| 1k | `host_visible` | 9.508 (105.17) | 5.783 (172.93) | 15.869 (63.02) | 13.465 (74.27) | 51.978 (19.24) |
| 1k | `host_device_local` | 8.783 (113.85) | 5.360 (186.58) | 15.380 (65.02) | 13.974 (71.56) | 49.765 (20.09) |
| 10k (65,536 / 15.3% / 2.50 MiB / 1.310) | `device_local` | 5.152 (194.09) | 4.880 (204.90) | 27.998 (35.72) | 20.506 (48.77) | 146.217 (6.84) |
| 10k | `host_visible` | 7.536 (132.69) | 6.236 (160.36) | 72.932 (13.71) | 54.183 (18.46) | 334.096 (2.99) |
| 10k | `host_device_local` | 5.066 (197.38) | 4.841 (206.57) | 28.322 (35.31) | 20.641 (48.45) | 144.532 (6.92) |
| 100k (524,288 / 19.1% / 20.00 MiB / 1.416) | `device_local` | 7.618 (131.28) | 6.005 (166.54) | 53.558 (18.67) | 54.927 (18.21) | 290.279 (3.45) |
| 100k | `host_visible` | 47.905 (20.87) | 29.887 (33.46) | 3492.770 (0.29) | 2240.356 (0.45) | 32108.853 (0.03) |
| 100k | `host_device_local` | 7.660 (130.54) | 6.040 (165.55) | 53.983 (18.52) | 54.804 (18.25) | 292.561 (3.42) |

On this GPU:

1. **`device_local` is stable and cheap at every size.** Wave-uniform sweeps run `4.9–7.6 us / 1M` (the 100k `uniform_tsharp_hit` is `7.618`), divergent lookups `14.1–54.9 us / 1M`, and the 8-probe miss `50.5–290.3 us / 1M`. Repeating the 10k measurement keeps `divergent_miss` at `141–143 us / 1M` net, so this column is not noise.
2. **`host_visible` falls off a cliff at 100k, and is unreliable below it.** At 100k it is `5.0x–6.3x` slower than `device_local` on wave-uniform sweeps, `41x–65x` on divergent hits and pairs, and `111x` on 8-probe misses. At 10k the same rows varied by up to 6x between runs (`divergent_miss` net `147–845 us / 1M` over five runs), so the 10k host row above is one representative run; the uniform rows moved much less (`6.1–8.7 us / 1M`). The conclusion does not change: `device_local` is stable, `host_visible` is not.
3. **`host_device_local` (ReBAR) is a safe fallback on AMD.** It is within 1% of `device_local` at 100k and within 2% at 10k; at 1k it is within 1.5% on the divergent and miss rows, `uniform_pair_ts` is 6% faster (`5.36` vs `5.71 us / 1M`) and `uniform_tsharp_hit` is 41% slower (`8.78` vs `6.22 us / 1M`). The `DEVICE_LOCAL | HOST_VISIBLE` allocation succeeds and runs here; it is not guaranteed on every Windows driver.
4. **The 100k `divergent_miss` difference is table geometry, not the vendor.** `bc8e604`'s bounded relocation is already in this run and keeps the table at `524,288` slots (`19.07%` load, `max_probes=8`); the NVIDIA table above predates it and grew to `2,097,152` slots (`4.8%`, `max_probes=5`). More probes per lookup is why `290 us / 1M` here exceeds the NVIDIA `217 us / 1M`, even though the 1k–10k rows are close. The relocation is therefore already paying for itself (a 4x smaller 100k table), and the two 100k rows are not a vendor comparison.

Raw CSV rows: [D3D-56 report](https://github.com/ItsNotRexed/Leonida/issues/313#issuecomment-7074181793), addendum with the repeatability runs: [same issue](https://github.com/ItsNotRexed/Leonida/issues/313#issuecomment-7074220711).

**Measured (D3D-56, Intel integrated + RTX 5070 Ti Laptop).** These synthetic runs use `vk_descriptor_bench` from revision `f85582e`, Windows 11 Pro Insider Preview `10.0.29704`, MSVC 19.51, Vulkan SDK 1.4.363.0, and the exact `4096 x 64` invocations x `64` iterations (`16.78M` single / `33.55M` T#+S# pair lookups, median of 25 reps). NVIDIA runs used `--sizes 1000,1000,10000,100000 --warmup 200`; the Intel run additionally duplicated 100k (`--sizes 1000,1000,10000,100000,100000 --warmup 200`). The first 1k block is excluded because its `baseline_uniform` timing differed from the duplicate (Intel device-local: `0.771 ms` vs `0.594 ms`; NVIDIA device-local: `0.0555 ms` vs `0.0765 ms`). For Intel 100k, the second block is retained after the duplicate confirmed stable timing per allocation. Retained rows were not auto-scaled. Intel's device-local and host-visible placements ran together on the integrated GPU. NVIDIA device-local and ReBAR were measured in separate processes. The RTX 5070 Ti Laptop is `0x10de:0x2f58` (driver `621.01`, Vulkan `1.4.357`); Intel Graphics is `0x8086:0x7d67` (driver `32.0.101.8132`, Vulkan `1.4.325`).

| GPU / entries (cap / load / bytes / mean probes) | Placement | `uniform_tsharp_hit` (us/1M, Glk/s) | `uniform_pair_ts` (us/1M, Glk/s) | `divergent_tsharp_hit` (us/1M, Glk/s) | `divergent_pair_ts` (us/1M, Glk/s) | `divergent_miss` (us/1M, Glk/s) |
|---|---|---|---|---|---|---|
| Intel iGPU, 1k (4,096 / 24.4% / 160 KiB / 1.316) | `device_local` | 85.669 (11.67) | 76.483 (13.07) | 468.524 (2.13) | 431.528 (2.32) | 1,735.402 (0.58) |
| Intel iGPU, 1k | `host_visible` | 85.356 (11.72) | 76.500 (13.07) | 468.750 (2.13) | 425.419 (2.35) | 1,741.669 (0.57) |
| Intel iGPU, 10k (65,536 / 15.3% / 2.50 MiB / 1.310) | `device_local` | 86.806 (11.52) | 76.642 (13.05) | 597.170 (1.67) | 507.250 (1.97) | 2,495.724 (0.40) |
| Intel iGPU, 10k | `host_visible` | 87.665 (11.41) | 76.415 (13.09) | 597.099 (1.67) | 506.874 (1.97) | 2,496.882 (0.40) |
| Intel iGPU, 100k (524,288 / 19.1% / 20.00 MiB / 1.416) | `device_local` | 101.132 (9.89) | 84.286 (11.86) | 672.021 (1.49) | 964.268 (1.04) | 5,580.289 (0.18) |
| Intel iGPU, 100k | `host_visible` | 100.269 (9.97) | 84.437 (11.84) | 668.788 (1.50) | 975.549 (1.02) | 5,581.019 (0.18) |
| RTX 5070 Ti Laptop, 1k (4,096 / 24.4% / 160 KiB / 1.316) | `device_local` | 11.028 (90.68) | 9.652 (103.60) | 48.388 (20.67) | 48.201 (20.75) | 195.543 (5.11) |
| RTX 5070 Ti Laptop, 1k | `host_device_local` (ReBAR) | 11.072 (90.32) | 9.682 (103.29) | 51.245 (19.51) | 48.427 (20.65) | 194.840 (5.13) |
| RTX 5070 Ti Laptop, 10k (65,536 / 15.3% / 2.50 MiB / 1.310) | `device_local` | 11.562 (86.49) | 10.164 (98.38) | 63.725 (15.69) | 66.871 (14.95) | 367.062 (2.72) |
| RTX 5070 Ti Laptop, 10k | `host_device_local` (ReBAR) | 11.494 (87.00) | 10.184 (98.19) | 63.858 (15.66) | 66.909 (14.95) | 297.104 (3.37) |
| RTX 5070 Ti Laptop, 100k (524,288 / 19.1% / 20.00 MiB / 1.416) | `device_local` | 11.993 (83.38) | 46.172 (21.66) | 116.020 (8.62) | 82.513 (12.12) | 788.086 (1.27) |
| RTX 5070 Ti Laptop, 100k | `host_device_local` (ReBAR) | 9.186 (108.86) | 46.162 (21.66) | 99.737 (10.03) | 141.439 (7.07) | 632.504 (1.58) |

On these devices:

1. **The Intel placements are close on the lookup workloads.** Gross dispatch times for `device_local` and `host_visible` track within 1.5%; after subtracting `baseline_uniform`, differences are within 1.7% through 10k and within 11.2% on 100k uniform hits. At 100k, the baseline itself is repeatable per allocation but differs between them (`0.730 ms` device-local vs `0.608 ms` host-visible), so the net uniform rows carry more uncertainty. Both allocations use the iGPU's shared system memory; this does not predict discrete-GPU PCIe behavior.
2. **NVIDIA ReBAR is close to device-local for uniform sweeps at 1k–10k and at 100k T#+S# pairs.** The 100k divergent and miss cases differ by workload (ReBAR is not uniformly faster or slower), so this run does not support a blanket placement ranking.
3. **The NVIDIA host-visible run is omitted.** A separate full run returned `vkWaitForFences -> -4` while measuring the 10k host-visible cases; those partial timings are not reported. A safer capped profile is needed before drawing conclusions about PCIe system-memory lookup on this laptop.

The run is synthetic and contains no game data. Raw retained rows and the corrected methodology are recorded in [the D3D-56 report on issue #313](https://github.com/ItsNotRexed/Leonida/issues/313#issuecomment-7078191651).

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

Measured per stage (all 26,503, `tools/feature_scan.py`): `s_getpc_b64` is kept to read code-embedded data in 54
VS, 93 PS, 266 MS and 263 CS (0 GS); the 147 VS/PS match the translator's own note. Both stages of a VS+PS or
MS+PS pipeline can therefore need a code base.

**Measured:** across the GPU test fleet (Issue #166), desktop NVIDIA (Turing, Ampere, Ada, Blackwell: 256 bytes), desktop AMD (RDNA2: 256 bytes) and Apple Silicon (MoltenVK: 4096 bytes) all satisfy the 144-byte contract natively (`maxPushConstantsSize >= 144`). Intel Arc is not uniform: the earlier fleet entries report the 128-byte specification minimum (`maxPushConstantsSize == 128`) for Arc, but an Arc B580 (Battlemage) on Windows 11, driver 32.0.101.9175, reports 256 bytes (#166, @rtrexmass), so there D3D-14's fallback never engages. The 128 figure may be specific to another OS or driver (Mesa ANV on Linux is the guess; no Arc run on Linux that records the OS is cited here, and an ANV result would settle it). So does an integrated AMD RDNA2 GPU, a Radeon 680M on the Windows driver 22.40.84.06, so the minimum is not Intel-only. 128 bytes cannot fit the 16-byte tail (dwords 32–35: heap base and code base) inside push constants. The runtime detects `maxPushConstantsSize < 144` at device creation (`d3d12_vulkan_device.cpp`) and flags `push_constants_need_fallback` for the uniform-buffer fallback.
**Open:** the exact count of such PSOs (the pairing is in the executable), and the minimum `maxPushConstantsSize` across target GPUs. If a GPU only
supports 128 bytes, the current 144-byte contract already fails; the layer needs a fallback (uniform buffer for the
tail) regardless.

At run time, `CreateGraphicsPipelineState` refuses a pair where both stages read their code base, with a log, so
the first boot measures the pair count. A device whose `maxPushConstantsSize` is below 144 has every PSO refused,
with the limit in the log, until the uniform-buffer tail exists.

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
| **M0** | `D3D12XboxCreateDevice` and the measured device overrides (heap start handles in RAX), one direct queue, allocator/list, fence, fake DXGI factory/adapter/output, `ALLOW_DISPLAY` back buffers, frame events + `PresentX` (DXGI swapchain only if the game creates one), clear + present | COM interception, the known Xbox device entry point and ABI overrides, presentation (`PresentX` + frame events, or swapchain), frame loop, clean shutdown | A window shows a stable clear colour for 60 frames, no Vulkan validation errors, every D3D12 call traced. **Synthetic part met** by `d3d12_present_test`: 60 frames through the proxy on Vulkan, validation clean, colour checked on screen. Still open: the game's present front end (frame events + `PresentX`, or a DXGI swapchain) and the per-call trace |
| **M1** | Root signature with root constants (mask the measured `0x03000000` flag byte on a copy), one translated VS+PS, PSO, vertex buffer in the arena, draw, push constants `s0..s31` + heap base | PSO translation, shader hash lookup, vertex input, root-constant packing, heap base, present | The triangle matches a CPU/GPU reference; the existing texture-test style comparison passes with error 0. **Met with stand-in shaders** by `d3d12_draw_test`: every pixel of two indexed draws and a culled triangle matches, through the proxy, validation clean. With translated game shaders it waits for set 0 in pipelines (M2), since most of them declare bindings |
| **M2** | Descriptor heap (heap slots 9/10 return in RAX), `CreateShaderResourceView`, sampler, root descriptor table, hash-table insert, bindless sampling | The full D3D12 descriptor path and the shader's T#/S# contract | A textured quad matches the synthetic 8×8 gradient or a known texture with error 0; the hash table has the expected entries |
| **M3** | Depth/stencil, multiple render targets, resource barriers, render passes, occlusion queries, fences, 2–3 frames in flight; run the game's first frames | Real frame structure and the first game code | The game reaches its first rendered frame (loading screen/menu or whatever the executable does) or stops on a single unsupported call with a clean report |
| **M4** | Compute PSOs, UAVs, storage images, indirect dispatch, copy queues, async streaming | Asset loading and GPU compute | The world's streaming path runs without device loss; compute results match a CPU/reference check |
| **M5** | Mesh shaders, `ExecuteIndirect`, bindless-table pressure, multi-threaded command lists, pipeline cache | World rendering | A stable open-world frame at any frame rate; then profile and optimise |
| **M6** | Ray tracing/XDXR: standard AS builds or the shipped `XboxBvhBuilder` compute path; BVH node format; Xbox DXIL path for the builder's 53 shaders (22 lowered so far) | RT shadows and any RT effects | RT output matches a reference; the builder's shaders run through the DXIL path or a substitute; no unknown XDXR call remains |
| **M7** | HDR, VRS, `PresentX` planes (2-plane HDR + GameDVR compose, `ImmediateThresholdPercent`), resize, alt-tab, `SuspendX`/`ResumeX`, frame pacing and `GetFrameStatisticsX` | Presentation polish | HDR and resize work on the target display; no validation errors over a long run |

### 6.1 Minimum API set for the first frame

M0–M3's synthetic tests need only this subset. The game itself may need more as soon as its first frame does;
everything else can be absent or a clean failure:

- `D3D12XboxCreateDevice` and `CheckFeatureSupport`; the measured device overrides (heap start handles in RAX;
  slot 76 texture creation if the game uses it; slot 62, compute-PSO creation in the builder's ABI, is not needed
  for the first frame); one direct queue, one allocator, one command list, one fence;
- DXGI factory/adapter/output enumeration; frame events and `PresentX` (or `GetBuffer`/`Present` if the game creates a swapchain);
- a root signature with root constants and one descriptor table, masking `0x03000000` on a copy; one VS+PS PSO and
  an input layout;
- a shader-visible CBV/SRV/UAV heap and a sampler heap; `CreateShaderResourceView`, `CreateSampler`,
  `SetDescriptorHeaps`, `SetGraphicsRootDescriptorTable`;
- one committed vertex/index buffer, one texture, upload through `Map`/`Unmap` or `WriteToSubresource`,
  `CopyBufferToImage`;
- resource barriers, RTV/DSV, `OMSetRenderTargets`, `Clear*View`, `DrawIndexedInstanced`, `PresentX`/`Present`;
- the runtime side: descriptor hash table, bindless set 0, push constants (`s0..s31`, heap base, code base),
  vertex input from the PSO layout.

Not needed for the first frame: compute, mesh, geometry, ray tracing, VRS, PresentX plane compose, reserved/tiled resources,
heap aliasing, indirect draws, multiple queues, pipeline libraries. (`PresentX` itself is needed; only its plane compose, HDR and statistics are not.)

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

- Presentation slots and struct layouts (`PresentX` and plane parameters; call shapes are known from public samples, §4.5), command-list extensions, and queue/fence creation slots remain
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
| 2 | Which Xbox device/heap/queue/command-list slots differ? Prior art measured device slots 62 and 76 and descriptor-heap slots 9/10; slot 14 is the public `CreateDescriptorHeap`, with only the returned heap's start-handle return wrapped. **Partly measured** with the FSR DLL ([d3d12_command_objects.md §3.5a](d3d12_command_objects.md)): device slots 14, 15, 16, 18, 19, 27, 38, 62 and 76 at their expected shapes (24 in the code, not called), slot 67 a console-only pipeline call; command-list slots 14, 15, 25 and 26 at their PC positions, then shifted (+4 from `SetDescriptorHeaps` at 32, +2 at `ExecuteIndirect` at 61), with console-only methods at 85 and 141. The BVH builder's call sites add the rest of the +4 root group (37, 39, 43, 45) and console slots 71, 73, 80, 146 (the GCL7 `Barrier`) and 148 (an upload allocator whose pointer is also the GPU address) ([d3d12_command_objects.md §3.3, §3.3a](d3d12_command_objects.md)). Presentation, queue and fence slots remain unknown. | Trace every `QueryInterface` and every method slot with per-slot hit counters. A wrong slot shows as an immediate crash or wrong argument; compare against the SDK vtable. |
| 3 | How does the console map a root signature to `s0..s31` (order, per-stage visibility, spills above 32 dwords, static samplers, direct-heap base)? | Dump the game's root-signature blobs with the host deserializer; mask the `0x03000000` flag byte on a copy first; match against the per-shader user-SGPR read sets. If a signature exceeds 32 dwords, reverse the spill pointer from the shader. |
| 4 | Are Xbox descriptor bytes GFX10 T#/S#, and what are the heap increments? Does the game use `Create*View` or build descriptors itself? | Dump a shader-visible heap after the first `CreateShaderResourceView` and after the first draw; compare the bytes the shader loads with the T# parser. The start-handle return ABI is known (RAX); the increment values still need measuring. |
| 5 | What is the buffer/descriptor/code VA range, and what arena capacity is needed? | Log every `GetGPUVirtualAddress` and every descriptor base; track the high-water mark. Choose a capacity; grow at a frame boundary if needed. |
| 6 | Which swapchain/present path is used: DXGI, `PresentX`, or both? | **Partly answered from public samples (§4.5):** GDKX samples use `PresentX` + frame events and no DXGI swapchain (DXGI only for `IDXGIDevice1`→adapter→output). Hypothesis for the game, not observed: same. Still open: vtable slots, `D3D12XBOX_*` layouts, and whether the game also creates a swapchain. Check imports and the first present call. Implement the `PresentX` + frame-event presenter first; keep DXGI swapchain as fallback. |
| 7 | Does the game build acceleration structures through the standard D3D12 calls or through `XboxBvhBuilder`'s own compute path? | Log `BuildRaytracingAccelerationStructure`, `CopyRaytracingAccelerationStructure`, `Emit...PostbuildInfo`, and the builder's command-list method calls on the first RT frame. The builder's 53 shaders are Xbox DXIL; 44/53 fail the PC compiler and 22 have been lowered. If the builder path is used, the layer needs a DXIL plan. |
| 8 | What are the layouts of `D3D12XBOX_XDXR_STREAM_DESC`, the batched build desc, and the offline statistics desc? | **Measured** from `XboxBvhBuilder.dll`'s reads ([xdxr.md](xdxr.md)): `BATCHED_DESC2` (`0x50`) and `XDXR_STREAM_DESC` (`0x10`) with every field's offset, and the stream records' offsets, types and defaults. Still open: the record names, and the offline statistics layout, which the DLL never writes (it rejects that postbuild type). |
| 9 | Xbox texture creation is device slot 76 with layout `0x101` (tiled) or `0x1` (row-major). What is the tiled data layout, and does the game upload console-tiled or DCC data? | Trace slot 76, `CopyTextureRegion` and the resource descs. If the game uploads console-tiled data, add a detile step or a matching layout. |
| 11 | What should `CheckFeatureSupport` answer for UMA, resource binding tier, mesh tier, RT tier? | **Implemented** (D3D-11). Reports truthful host architecture (discrete desktop GPU: non-UMA), ResourceBindingTier 3 (descriptor indexing / bindless support), 48-bit GPU virtual address support matching the arena, and maps `D3D12_FEATURE_FORMAT_SUPPORT` to Vulkan format properties plus multisample render-target/resolve support and one quality level per supported sample count. |
| 12 | Does the game rely on async compute or copy queues for correctness? | Trace queue creation and cross-queue fences. If only performance, serialize on one queue for boot. |
| 13 | Is 144 bytes of push constants available on all target GPUs? | **Resolved / Measured** (D3D-14). Checked at device creation in `runtime/shims/d3d12_vulkan_device.cpp`. Measured across the test fleet (#166): NVIDIA (256 B), desktop AMD (256 B), and Apple M1/MoltenVK (4096 B) satisfy 144 B natively. A Radeon 680M on the Windows driver 22.40.84.06 and the earlier Intel Arc entries expose only the core minimum 128 B (`maxPushConstantsSize == 128`), but an Arc B580 on Windows 11 (driver 32.0.101.9175) reports 256 B (#166), so the 128 B figure for Arc depends on the OS or driver; an Arc result on Linux (Mesa ANV) would settle which side it is. `d3d12_vulkan_device_test` accepts either answer and checks that the fallback flag matches it. Below 144 B, the device logs a warning and marks `push_constants_need_fallback`, routing the 16-byte tail (dwords 32..35: heap base and code base) to a uniform-buffer fallback. |
| 10 | Does the game use binning, DCC, VRS, wave MMA, stream output, tessellation, or amplification shaders? | **Shaders: answered** by `tools/feature_scan.py` over all 26,503 (1.0.2.1): no hull, domain or amplification shader (DXIL program kinds: 2,236 VS, 12,101 PS, 195 GS, 2,689 CS, 9,282 MS); no `WAVE_MMA` or `SHADING_RATE` in any SFI0, and no `v_mfma`/`v_wmma`/`v_smfmac` in the disassembly; no output-signature element on a stream other than 0 (all 195 GS write stream 0 only) and no GDS / `ds_ordered_count` / GWS instruction, so no stream-output path. Also from SFI0: 10 PS need ROVs (pixel interlock), 30 shaders `TILED_RESOURCES` (see #18), 685 `ATOMIC_INT64_ON_TYPED_RESOURCE`. Binning and DCC are not shader-visible; they stay open for the PSO/stream subobjects. |
| 14 | Are there PSOs with two code-base stages? | Count from the PSO/shader hash table and the index metadata. If yes, implement per-stage code-base slots (section 5.2). **Per stage, measured** (`tools/feature_scan.py`, `s_getpc_b64` kept for data rather than a far branch): 54 VS, 93 PS, 266 MS, 263 CS, 0 GS; the translator's `uses_code_base` gives the same counts (with LLVM 23 that needs #223, without which these 1,471 shaders are rejected). So VS+PS and MS+PS pipelines with two code-base stages are possible; whether any exists depends on which shaders a PSO pairs, which is in the executable, not in `shaders.rpf`. |
| 15 | Is `ExecuteIndirect` used with signatures Vulkan cannot express directly? | Trace `CreateCommandSignature` and `ExecuteIndirect`; implement simple signatures first, expand the complex ones with compute. The simple signatures are D3D-17 (#313). |
| 16 | Does the game use append/consume UAV counters or typed buffer formats the translator does not model? | Trace UAV descs and shader `buffer_*` ops; extend the translator/runtime together. |
| 17 | Do upload/readback resources ever reach shaders (as vertex/index buffers or descriptors), and does the `-heap_base` bias work for them? | Trace `CreateCommittedResource` heap types and every VB/IB/descriptor that points at an upload resource; test the bias with a synthetic draw. |
| 18 | Does the game alias buffers/images in a heap, or use reserved/tiled resources? | Trace `CreatePlacedResource` ranges, aliasing barriers, `CreateReservedResource` and tile mappings. If yes, implement shared memory per heap range or fall back to linear allocations with a clear log. Shaders: 30 declare `TILED_RESOURCES` in SFI0 (`tools/feature_scan.py`), so reserved/tiled resources are likely used. Placed resources are D3D-16 and reserved resources D3D-20 (#313). |
| 19 | What do the `0x03000000` root-signature flag bits mean? | Masking them on a copy works around the PC serializer; the semantics remain unknown. Compare against the public flag enum and observe the game's behaviour with and without the mask. |
| 20 | Which of the BVH builder's 53 Xbox DXIL shaders does the game actually use, and can the 31 raw-address shaders be lowered without a reference? | Dump the slot-62 compute PSOs on the first RT frame; lower and compile each; compare against the builder's CPU validation (`CpuValidateResultX`) or a captured reference. |
| 21 | Which `Metadata` values and blob format does the console's PIX use for `SetMarker`/`BeginEvent`? The layer reads 0 and 1 as strings and labels 2 (the PC PIX3 blob) `PIX event` (section 4.10). | Log the `Metadata` values the game passes; if it is 2, decode the console blob from the executable's PIX calls rather than assuming the PC WinPixEventRuntime layout. |

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
- **Hash-table cost.** Every image/sampler access hashes a T#/S# and probes a storage buffer. **Measured (D3D-56, section 5.1)** on an RTX 5080 (driver 616.56): in `DEVICE_LOCAL` VRAM the lookup itself stays fast up to 100k entries (`3.0–3.7 us/1M` net wave-uniform full sweeps, `17.2–28.9 us/1M` net divergent hits), whereas `HOST_VISIBLE | HOST_COHERENT` system memory drops 9.6x–28x on uniform full sweeps and 59x–755x on divergent lookups at 10k–100k entries, and linear probing's 8-probe limit doubles a 100k table to `80.00 MiB` (`4.8%` load). **The runtime now keeps that table in `DEVICE_LOCAL` memory** and uploads only the slots that changed, from a mapped staging copy, with the copy and its barrier in the command buffer that will read it (D3D-76; 46.2x on a divergent miss at 100k on an RTX 2060 SUPER). Improving slot placement within the 8-probe window is the open optimisation (SHADER-100); Intel, RTX 5070 Ti Laptop and AMD RDNA3 are measured in section 5.1; Apple numbers remain open via `vk_descriptor_bench`.
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
- [Xbox-GDK-Samples (MIT): SimpleDeviceAndSwapChain, FramePacing](https://github.com/microsoft/Xbox-GDK-Samples)
- [DirectXTK12 wiki: DeviceResources](https://github.com/Microsoft/DirectXTK12/wiki/DeviceResources); [directxtk12modelviewer](https://github.com/walbourn/directxtk12modelviewer) (MIT); SDL2 `SDL_render_d3d12_xbox.cpp` (zlib)
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

Constructed by `runtime/test/bvh_builder_test.cpp` (1.0.2.1: `rt::Loader`, `Loader::initialise()` for TLS and
`DllMain`, `.pdata` registered by the loader (#176), then `XboxBvhBuilder(ID3D12Device11*, bool)`
on a PC device behind the Xbox-ABI proxy). Measured:

- 0 unimplemented imports; `D3D12SerializeVersionedRootSignature` and `XMemVirtualQuery` bind to shims. The constructor writes 24 bytes of the object (`sizeof(XboxBvhBuilder)` >= 24;
  the rest of its state is elsewhere). It prints nothing through `OutputDebugString` and never calls
  `TerminateProcess`. On failure it throws the failing `HRESULT` as a C++ `long`.
- **Device slot 62 is called as `(desc, 0, 0, &IID_ID3D12PipelineState, out)`**, with a plain
  `D3D12_COMPUTE_PIPELINE_STATE_DESC` (root signature set, node mask 0, flags 0). The proxy used to read
  `(desc, riid, out)`, so the first call returned `E_POINTER` and the constructor threw it.
- With slot 62 read that way, on **WARP** the constructor returns after 62 ms: 17 root signatures (slot 16) and 53
  compute PSOs (slot 62, 53 of 53 `S_OK`), CS bytecode 4–75 KB each. `~XboxBvhBuilder` returns and makes no device
  call. Created is not the same as correct: the shaders are Xbox DXIL with `@xbox.*` intrinsics (section 7.2), and
  nothing has been dispatched yet.
- **NVIDIA (RT-54).** On an RTX 3060 (616.64), an RTX 4090 (621.14) and an RTX 5070 Ti (621.24), all on Windows 11,
  the first 48 PSOs return `S_OK`, and the 49th (38,248 bytes) never does. Some runs return `E_OUTOFMEMORY` for it,
  which the constructor throws. Other runs end the process inside `nvwgf2umx.dll` with `0xC0000409`, after the
  driver's LLVM 7.0.1 bitcode reader prints `Invalid record` or `Never resolved value found in function`. That
  can't be caught. The D3D12 debug layer prints nothing first.
  - **Cause: the input, not Leonida.** The proxy passes the bytecode through unchanged, so nothing is wrong in what
    it forwards. The input is the problem: `LEONIDA_BVH_RECORD62=1 LEONIDA_BVH_DXIL=1` shows that every one of the
    53 containers is signed, has the same 9 parts, and is shader model 6.5 / DXIL 1.5. But dxil.dll (Windows SDK
    10.0.26100) rejects all 53 (`0x80aa0009`): the target triple is `dxil-ms-xb2` and the module carries
    `xbox.compiler.options` metadata. Most also have Xbox-only externals (44 of 53), `extractvalue` on non-DXIL types
    (16) or other disallowed instruction types (5). No PC driver has to accept this, and WARP happens to.
  - **What sets the 49th apart: one error kind.** It is the only shader of the 53 that the validator reports for a
    resource in a function signature (a non-entry function that takes a resource). Its other counts are 265
    `extractvalue`, 6 Xbox externals and 514 `i64` mentions; the 36th shader has the same 265 and 468 and is created.
  - **Workaround: the test uses WARP on every hardware vendor.** AMD doesn't take these shaders either: on an RX 7900
    XTX (32.0.32029.2010) the constructor faults with `0xC0000005` right after the first slot 62 call (RT-89, #166).
    So `bvh_builder_test` uses WARP whenever there is a hardware adapter, unless `LEONIDA_BVH_HARDWARE=1`, and it
    prints the reason with the adapter's PCI vendor id. WARP creates all 53 PSOs, and the constructor and destructor
    return.
  - **The runtime itself still shouldn't hand Xbox DXIL to a PC driver.** These shaders need lowering to PC DXIL or a
    translation of their own (section 7.2, M6) before they can run on hardware.

### A.3 `amd_fidelityfx_upscaler_xbox.dll`

- Size 786,432 bytes; PE32+ x86-64.
- Import: `d3d12_xs.dll!D3D12CreateRootSignatureDeserializer` (IAT RVA 164,672).
- GUID byte scan found `IID_ID3D12Device` (0x0472f0), `IID_ID3D12Resource` (0x047310),
  `IID_ID3D12RootSignatureDeserializer` (0x047320), `IID_ID3D12PipelineState` (0x047330),
  `IID_ID3D12Heap` (0x047340), `IID_ID3D12RootSignature` (0x047350), `IID_ID3D12DescriptorHeap` (0x0472e0) and
  `IID_ID3D12CommandSignature` (0x047300). Again, presence ≠ use.
- Run through the public FidelityFX API by `runtime/test/fsr_dispatch_test.cpp` (1.0.2.1: `rt::Loader` and
  `Loader::initialise()`, with #93's static TLS, which the DLL's thread-safe statics need, and `.pdata` registered by
  the loader (#176); PC D3D12 device on an RTX 4090 behind the Xbox-ABI proxy). Measured:
  - It maps with 0 unimplemented imports: `D3D12CreateRootSignatureDeserializer` and `XMemAlloc` /
    `XMemFree` / `XMemGetWorkingSetStatistics` bind to the shims, 82 imports to host DLLs. Its TLS template is
    8 bytes with no callbacks. `DllMain(DLL_PROCESS_ATTACH)` returns TRUE.
  - `ffxConfigure(GLOBALDEBUG)` and the CPU-only upscale queries accept the **PC struct type ids**: the
    Performance ratio is 2.0 and 3840x2160 Quality renders at 2560x1440. `GET_VERSIONS` for the upscaler lists one
    provider, `0xf5a5ca1e01001001` "4.1.1".
  - `ffxCreateContext` reports, through the message callback, that only one configuration is built in:
    `maxUpscaleSize` 1920x1080 ("only the fixed-resolution Quality shader tables are included", so a 1280x720
    render size), with `HIGH_DYNAMIC_RANGE`, `DEPTH_INVERTED` and `AUTO_EXPOSURE` all required. It also requires an
    `ffxCreateContextDescUpscaleVersion` in the chain; 4.1.2 is rejected as invalid, 4.1.1 and 4.0.0 are accepted.
  - With all of that, `ffxCreateContext` returns `FFX_API_RETURN_ERROR` with no message, no XMem allocation and no
    call on the device proxy, for the PC DX12 backend descriptor (`0x2`), for the same `{header, device}` layout
    under `0x02000000`, `0x02000001` and `0x02000002`, and with no backend descriptor at all.
- **The Xbox backend descriptor, measured from the DLL** (Ghidra 12.2 through REA; RVAs, image base `0x180000000`):
  - `ffxCreateContext` (`0x11d0`) picks its one provider (`ffxProvider_FSR4_Int8` by RTTI) by finding a device in
    the chain: type 4 (`GET_VERSIONS`) at `+0x18`, types `0x10009` and **`0x01000002`** at `+0x10`.
  - The provider's create (`0xc9b0`) wants `CreateContextDescUpscale` (`0x10000`) first, checks the version
    descriptor (`0x1000b`, 4.0.0 to 4.1.1), then walks `pNext` for **type `0x01000002`: `{header, device}`, the device
    at `+0x10`**, the same layout as the PC DX12 descriptor. Two of them, or none, is `FFX_API_RETURN_ERROR`
    without a message: that is what the ids above got. An optional type `0x01000003` sets function pointers: when the
    one at `+0x20` is set (with `+0x28`), the backend calls it instead of `CreateHeap` (device slot 28, `0x7390`); what
    its other fields do is open.
  - The HRESULT checks at the calls seen here (`0x33d4`, `0x65b3`, `0x74a8`) go through one helper (`0x1c60`):
    `FormatMessageW` of the code, then `throw 1`.
- **Run with that descriptor** (`fsr_dispatch_test`, PC device on an RTX 2000 Ada, driver 597.06, behind the
  Xbox-ABI proxy): the DLL creates its backend through the proxy. Device slots called: 14 `CreateDescriptorHeap` (5),
  15 `GetDescriptorHandleIncrementSize`, 18 `CreateShaderResourceView`, 19 `CreateUnorderedAccessView` and **76** (4),
  plus heap slot **9** `GetCPUDescriptorHandleForHeapStart` (Xbox RAX form). No earlier test in this repository
  reached slot 76 or heap slot 9 with a real client; their measured shapes hold. Slot 76 is called with the 7
  arguments of `CreateCommittedResource`
  (heap properties, heap flags, resource desc, state, clear value, IID, out; call at `0x33cc`), the desc being the
  56-byte `D3D12_RESOURCE_DESC` with layout `0x101` and the word 3 at `+0x34`.
  - On that GPU it stops at its 4th slot-76 texture: 1920x1080 `DXGI_FORMAT_R9G9B9E5_SHAREDEXP` with
    `ALLOW_UNORDERED_ACCESS`, which the NVIDIA driver refuses (`E_INVALIDARG`); WARP creates it. The console build
    creates it, so the console takes that format as a UAV. On the same GPU Vulkan has no `STORAGE_IMAGE` support for
    `VK_FORMAT_E5B9G9R9_UFLOAT_PACK32` either (vulkaninfo, driver 597.06), so the layer needs an emulation (32-bit
    storage plus a pack/unpack) where a GPU lacks it. @Vonsek measured the same texture on more devices (#864):
    refused on an RTX 5080 (616.92), created on an AMD Radeon and on WARP, where typed UAV writes round-trip; and on
    both GPUs one allocation can serve both uses, a mutable `R32_UINT` Vulkan image with `E5B9G9R9_UFLOAT_PACK32` in
    its format list, written through an `R32_UINT` storage view and sampled through an `E5B9G9R9` view.
    **Implemented** in `d3d12_vulkan_device.cpp` and `src/uav_clear.h` (`d3d12_r9g9b9e5_uav_test`): on an RTX 3060 Ti
    (driver 621.01, `LEONIDA_FSR_DEVICE=vulkan LEONIDA_FSR_NO_STANDIN=1`, `LEONIDA_VK_VALIDATION=1`),
    `ffxCreateContext` and `ffxDestroyContext` both return OK without `standin_slot76` and with 0 validation errors.
    Open: a float store to that format rounds differently on PC devices (@Vonsek, #864: AMD Radeon truncates the
    mantissas, WARP rounds to nearest even; 609 of 794 test triples differ). Which rule the console FSR needs, and
    whether its shaders store floats to that texture at all (requiring shader-side packing into the `R32_UINT` storage
    view), waits on decoding its `FSPX` shaders.
  - With that one texture created as `R32_UINT` instead (the test's PC stand-in, `standin_slot76`), or as is on
    WARP, `ffxCreateContext` and `ffxDestroyContext` both return OK. Device slots called: 14 (5), 15 (26), 18 (10),
    19 (16), **27
    `CreateCommittedResource` (4)**, **38 `GetCopyableFootprints` (4)**, 76 (10); heap slot 9 (26). So the console
    device also serves the PC `CreateCommittedResource` at slot 27, and slots 14, 15, 18, 19, 27 and 38 keep their PC
    positions for this client. No compute pipeline or root signature is created at context creation.
  - On that context, one `ffxDispatch` recorded into a logging command list measures the **console command list**:
    slots 14, 15, 25 and 26 are at their PC positions, then `SetDescriptorHeaps` is at 32 (PC 28), the compute root
    setters four slots later (33, 35, 41; two of them inferred from their shapes), `ExecuteIndirect` two slots later
    (61), and there are console-only methods at 85 and 141. The list object also carries a write cursor and limits
    (`+0x18`, `+0x28`, `+0x30`) the DLL writes commands through while there is room. Table and evidence:
    [d3d12_command_objects.md §3.5a](d3d12_command_objects.md).
    Device slot 67 is a console-only compute-pipeline call (a code blob in its desc), not the PC method at that
    index; the proxy now refuses it. The dispatch stops where the PC runtime refuses the DLL's compute code at slot
    62: the DLL's shaders carry an `ACCC` part instead of DXIL.
  - Its `throw 1` reaches the caller: `_CxxThrowException` finds no image base for a manually mapped image, so the
    DLL's own handlers cannot catch it (the loader gap `tiny_game`'s `LEONIDA_RUN_EH` check also shows).
  - `xmem.dll!XMemGetWorkingSetStatistics` takes two arguments, `(0x20000001, stats*)`, at all seven call sites;
    the one-argument shim wrote through the constant. Fixed in `runtime/shims/xmem.cpp`.

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
