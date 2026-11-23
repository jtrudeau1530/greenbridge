# The retail boot chain: what stops `gta6rt --run`, in order

**Status: measured on one machine, 2026-11-23. A measurement for RT-5, not a claim; RT-5 stays with its owners
on the coordination issue.** Each row is the first failure with every fix above it in place. Nothing here quotes
the executable: no addresses, offsets, IID values or bytes; only export names, public SDK types, vtable slot
numbers, call shapes and counts. [boot.md](boot.md) has the runbook that produced the traces; #6715 maps the same
executable under `--dry-run`.

Setup: `main` `1cc6e3c12` with #6709, #4531, #6582, #4313, #4348 and #2490 merged locally, plus the local
stand-ins named below (not pull requests yet). Own 1.0.2.1 copy, Windows 11, RTX 4090 (driver 617.42),
`LEONIDA_D3D12=vulkan`, MSVC 2022 `RelWithDebInfo` build.

## 1. The chain

| # | Stop | What clears it |
|---|---|---|
| 1 | The first `XMemVirtualAlloc` calls: two 64 KiB reserve forms, then 2 MiB pages | #6709, #4531 |
| 2 | The 2 MiB reserve+commit calls carry attribute words the shim refuses, and `PAGE_GRAPHICS_*` bits (0x800..0x40000) on top of `PAGE_READWRITE` | Local: accept the attribute word, drop the graphics bits and keep the CPU protection. All seven startup allocations then succeed. #6843 measures the attribute words; no PR changes the check yet |
| 3 | `xgameruntime.lib`'s thunk treats a missing `XErrorReport` as fatal | #6582 |
| 4 | Access violation in the title's allocator: its `/GA` code reads its TLS from module index 0 | #4313 |
| 5 | Fatal assert when `DStorageGetFactory` returns null | #4348 |
| 6 | `XMemAllocatePhysicalPages` unimplemented; the title asserts on FALSE | Local stand-in, §2 |
| 7 | `D3D12XboxCreateDevice` succeeds; device slot 26 is called with its arguments swapped | Local, §3.1 |
| 8 | Device slot 48 is forwarded to the PC slot 48 and fails | #6954 |
| 9 | Device slot 73 is forwarded to the PC slot 73; the title then reads through a null list | #2490's list proxy plus local routing, §3.3 |
| 10 | An `XMemVirtualAlloc` reserve of about 185 MiB with type `0x42000`, attribute 7 and `PAGE_NOACCESS` is refused with `ERROR_INVALID_PARAMETER`. `0x40000` is `MEM_RESERVE_PLACEHOLDER` on the PC, which `VirtualAlloc` refuses and only `VirtualAlloc2` accepts; stop 11 supports the same reading on the console | Local: reserve it with `VirtualAlloc2(MEM_RESERVE \| MEM_RESERVE_PLACEHOLDER)` |
| 11 | The title carves five sub-ranges out of that placeholder with `MEM_RESERVE \| MEM_REPLACE_PLACEHOLDER \| MEM_PHYSICAL \| MEM_LARGE_PAGES` (`0x20406000`) at addresses inside it, then allocates a second batch of physical pages | Local: a physical reserve inside a known placeholder stays a placeholder (the page maps split it); other `MEM_REPLACE_PLACEHOLDER` calls split the range and allocate it in place |
| 12 | Device slot 133 is called; the proxy table had 128 slots, so the call went through a null pointer | Local: a 192-slot table, and a slot 133 stand-in, §3.4 |
| 13 | **Current stop**: right after the first `CreateDescriptorHeap`, the title calls the heap's slot 9 (`GetCPUDescriptorHandleForHeapStart` on the PC), takes the handle from `rax` and asserts that it equals a value it already holds. Hypothesis: a console descriptor heap starts at an address the title chose or can predict, which the PC handle doesn't match | Open: a question for the descriptor model ([boot.md](boot.md) Q4) |

At stop 13 the D3D12 trace holds 1,185 calls: the device, 5 command queues, 5 fences, 7 command allocators,
581 command lists created and closed, a shader cache session, `XboxIDXGIDevice::GetAdapter`, one failed
`ID3D12Device10::CreateReservedResource2`, the first `CreateSampler2`, one committed resource (the slot 133
stand-in) and the first descriptor heap.

Without a debugger, the unhandled `int3` of a fatal assert ends the process with `C0000374` (heap corruption) or
`C0000005` during teardown, not `80000003`. An exit code of `C0000374` from `--run` can be such an assert, not a
heap bug.

## 2. Physical pages (`xmem.dll`)

The shapes, from the call sites and the checks that follow them:

| Export | Shape |
|---|---|
| `XMemAllocatePhysicalPages` | `BOOL (ULONG64 attributes, ULONG_PTR* page_count /*in/out, 64 KiB pages*/, ULONG_PTR* page_array)`; the title checks both TRUE and the count written back. Attributes seen: `0x280`, `0x380` |
| `XMemVirtualAlloc` | the range for those pages is reserved with `MEM_RESERVE \| MEM_PHYSICAL \| MEM_LARGE_PAGES` (`0x20402000`) and no `MEM_COMMIT` |
| `XMemMapPhysicalPages` | `(va, page_count, page_array)`; a null `page_array` unmaps; non-null return on success |
| `XMemMapPhysicalPagesScatter` | `(va_array, page_count, page_array)`, the same with one VA per page |
| `XMemFreePhysicalPages` | `(page_count, page_array)` |

The local stand-in that gets past stop 6 is CPU-only: one pagefile section per 64 KiB page
(`CreateFileMappingW(INVALID_HANDLE_VALUE, …, SEC_COMMIT)`), page numbers handed out as indexes into that table,
the physical reserve made with `VirtualAlloc2(MEM_RESERVE | MEM_RESERVE_PLACEHOLDER)`, each map a
`VirtualFree(MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)` split followed by `MapViewOfFile3(MEM_REPLACE_PLACEHOLDER)`,
and each unmap `UnmapViewOfFile2(MEM_PRESERVE_PLACEHOLDER)`. #5381 has the CPU half of physical pages with dirty
tracking and #5378 a Vulkan sparse-binding mapper; neither is wired to these exports.

## 3. Device calls whose Xbox shape differs from the PC

### 3.1 Struct-returning methods take the return buffer first

Device slot 26 (`GetCustomHeapProperties`, which returns a `D3D12_HEAP_PROPERTIES`) is called as
`(return buffer, this, node_mask, heap_type)`: the first register holds the return buffer, the second the device
proxy. The PC MSVC method expects `(this, return buffer, …)`. A proxy thunk can tell the two apart: in the swapped
call the `this` register does not point at the proxy, and the second argument does. Calling the real method with
the two exchanged and returning the buffer pointer gets past it. Other struct-returning slots probably follow the
same rule; only slot 26 was reached.

### 3.2 Device slot 48 creates a command queue

Xbox device slot 48 is `(desc, riid, out)` with `riid = IID_ID3D12CommandQueue`. `desc` starts with a PC
`D3D12_COMMAND_QUEUE_DESC` (Type, Priority, Flags, NodeMask) followed by Xbox words. On the PC, slot 48 is
`OpenExistingHeapFromAddress`, so forwarding by number fails. Creating the queue from the 16-byte PC prefix, with
`Flags` limited to `D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT`, through the existing queue proxy works: four
queues come from this slot, the first of them a compute queue. The first queue comes from slot 8, as on the PC.

### 3.3 Device slot 73 creates a command list

Xbox device slot 73 is `(desc, allocator, pso, riid, out)`. `desc` is 16 bytes and its first word reads as the
`D3D12_COMMAND_LIST_TYPE`. `riid` is a console command-list IID that is not in the PC SDK (#3520 records that the
package doesn't export it). The shim's unknown-IID log names it on the player's machine. Creating an
`ID3D12GraphicsCommandList` of that type and handing out #2490's console list
proxy gets all 581 lists created and closed.

### 3.4 Device slot 133 places a resource at a virtual address

Xbox device slot 133 is `(va, const D3D12_RESOURCE_DESC*, state, clear, riid, out)` with `riid = IID_ID3D12Resource`
and `state = 0xFFFFFFFF`, which isn't a valid PC state. At the first call, `va` is the start of a physical-page
range the title had just reserved, and `desc` is a buffer of that range's size with Xbox bits in `Flags` above
the PC flags. The shape matches the console `CreatePlacedResourceX`. The local stand-in creates a committed
DEFAULT-heap buffer from the desc, with `Flags` limited to the PC bits and `D3D12_RESOURCE_STATE_COMMON`. It is
not tied to the VA, so what the title writes there doesn't reach the GPU. A real implementation needs the page
mapper (#5378) or an imported host allocation.

The proxy's slot table must also be longer than the title's highest slot. A shorter table leaves a null pointer,
not an `E_NOTIMPL`.

### 3.5 Committed resources take their memory through the title's XMem hook

On the console the D3D runtime allocates a committed resource's memory with `XMemAlloc`, so a hook the title
installs through `__XMemAllocationHooks__` sees it (stop 17 counts it against an arena). The device proxy does the
same for device slots 27 (`CreateCommittedResource`), 53 (`CreateCommittedResource1`), 76 and 127: it asks the
installed hook for the PC allocation size with the committed-resource attribute word
(`kXmemCommittedResourceAttributes`) and gives the memory back through the hook's free when the resource dies.

What the memory is used for depends on the resource:

- **A buffer** is made on that memory when the device behind the proxy is the Vulkan layer. The handler slot 133
  uses (`create_buffer_at_host_address`) imports it with `VK_EXT_external_memory_host`. The buffer's GPU VA is the
  hook's address, and the title's CPU reads and writes there are the buffer's bytes, with no copy. The memory is
  zeroed first unless the heap flags include `D3D12_HEAP_FLAG_CREATE_NOT_ZEROED`. `d3d12_existing_heap_test`
  checks both directions on a GPU.
- **A texture** stays on the device's own memory, and the hook memory only keeps the title's accounting. The
  console lays a texture out in its own tiling, and no Vulkan image on imported host memory reproduces it, so the
  title's CPU view of those bytes would not match the image anyway.
- **A buffer the handler cannot make** (a PC D3D12 device behind the proxy, no `VK_EXT_external_memory_host`, an
  address off the import alignment) and **a shared heap** (`D3D12_HEAP_FLAG_SHARED`, `_SHARED_CROSS_ADAPTER`) fall
  back to the device's memory, with the hook memory as bookkeeping. The hook is asked only once per resource.

GPU reads of host memory cross the bus, so a heavily read DEFAULT buffer is slower there than in device memory.
`LEONIDA_XMEM_RESOURCE_BACKING=0` keeps every committed resource on the device's memory (bookkeeping only) to
compare a run. The hook attribute word for a resource is the one #7235 measured at stop 17. The object type field
is not measured and stays 0.

## 4. What this does not show

These are one run's first failures, not a list of what is missing. Every stand-in above is local and untested
beyond "the title goes on to the next call". Nothing has reached a draw or a present yet.

## 5. The startup hang check

A stop that is a time limit, not a call: it ends a boot wherever the boot happens to be when the time is up. On
other machines it showed inside `XboxBvhBuilder.dll`'s constructor ("stop 18" in #7333), because that is where a
cold start spends its time.

Measured 2026-11-23 on a second machine: Windows 11, RTX 4080 (driver 610.88), i9-12900K, MSVC 2022
`RelWithDebInfo`, `LEONIDA_D3D12=vulkan`, own 1.0.2.1 copy. The tree is `Pipodi:boot-integration` at `b2e606620`
(`main` `1cc6e3c12` with the pull requests and stand-ins of #6982) plus #6943, run with that branch's
`LEONIDA_SKIP_ASSERTS=20` and the builder's 52 programs translated with `--wave64-emulate`.

### 5.1 What the title does

From a static read of the check's loop in the executable and from the timing of the runs: import names, counts and
values only.

| Fact | Value |
|---|---|
| The check's thread | started by the title during startup; it is not the main thread |
| One round | `Sleep(1000)`, imported from `api-ms-win-core-synch-l1-2-0.dll`, with `QueryPerformanceCounter` read before and after |
| The limit | 30 rounds in a row in which the main thread's progress mark did not change |
| At the limit | the thread writes to address 0; without a debugger the process ends with `C0000005` |
| A late round | a round whose `Sleep` took more than 2 s by `QueryPerformanceCounter` starts the count again at 30, as a suspended console would look to it |

In six runs the kill came 32.3 to 34.1 s after process start, whatever the title was doing then. In one of them
it came before `D3D12XboxCreateDevice`, with no D3D12 call made; in the other five it came inside the builder's
compute pipeline compiles, at 1,641 D3D12 calls. So up to that point of the boot the main thread
does not move its mark: everything from the check's start to there has to fit into about 31 s.

On the console the builder's pipelines are made from native code and take no time worth counting. Here each one
is a driver compile of a translated module on the title's thread.

### 5.2 What clears it

The layer marks the driver's compile of a compute pipeline as a host-only stall (`runtime/src/host_stall.h`), and
the runtime serves `Sleep` (`runtime/shims/host_stall.cpp`): a guest `Sleep` that a stall overlaps does not return
while the stall is open, and then lasts as much longer as stalls overlapped it. The check's round then comes back
late, and the title's own late-round rule starts the count again. No pipeline is skipped and the check stays armed:
a boot that hangs with no stall open still ends after 30 rounds.

| Run | Driver cache | `Sleep` | First slot 62 call | Outcome | D3D12 calls |
|---|---|---|---|---|---:|
| before | cold | host | 4.1 s | killed by the check at 32.4 s, inside the compiles | 1,641 |
| after | cold | held during stalls | 7.4 s | the compiles end at 63.1 s (55.7 s of them); the run goes on to the next stop at 63.4 s | 1,661 |
| reference | warm | host | 8.4 s | the compiles take 13.2 s; the next stop at 22.3 s | 1,661 |

For the two cold runs each of the 52 modules got one unused constant, a different one per run, so the driver's
disk cache missed. An earlier pair gave the same result (killed at 32.3 s; compiles of 57.5 s survived). The next stop is the same in the last two rows: device slots 74, 86 and 108 and the slot 67
programs, which #7323, #7201, #7180 and #7529 are about.

### 5.3 What it does not cover

- Time lost outside a marked stall counts as before. The run that ended before `D3D12XboxCreateDevice` sat for
  more than 31 s after the first `IDStorageFactoryX::OpenFile`; the next three runs took 25 s, 7 s and 3 s to get
  from there to the first `XDisplayTryEnableHdrMode`. Hypothesis: a cold file cache. Nothing marks that time.
- Graphics pipelines are built on first use, from whichever thread replays the list, and are not marked.
- Only `Sleep` is held. A check built on `SleepEx`, a wait with a timeout or a clock would not see the stall.
- Every guest `Sleep` that overlaps a stall is held, not only the check's: in the "after" run the first one held
  was a `Sleep(10)` of another title thread.
