// xmem.dll: the GDK's tracked allocator. Mapped onto the CRT's aligned heap and VirtualQueryEx.
//
// The measured ABI (docs from the clean-room spec) shows four imports:
//   XMemAlloc(size, attributes)
//   XMemFree(ptr, attributes)
//   XMemVirtualQuery(process, address, out64)
//   XMemGetWorkingSetStatistics(kind, out)   (kind 0x20000001 at all seven FSR call sites)
//
// ASSUMPTION: the attribute word is not decoded by the measured call sites (one passes 0x5e50000). Until a call
// site shows the alignment/type fields, every block is 64-byte aligned (covers cache lines and SIMD) and a size of
// zero is rounded up to one byte so the result is a stable, non-null pointer that XMemFree can release.
#include <windows.h>
#include <intrin.h>
#include <malloc.h>
#include <psapi.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <set>
#include <utility>

#include "shims.h"
#include "startup_trace.h"

namespace {

constexpr size_t kAlign = 64;

// gta6rt --trace FILE: the first XMemAlloc and the first XMemFree for each distinct raw attribute word go to the
// startup trace as an "xmem_attributes" event (api, the word, the size for an allocation, the caller as module+RVA). The
// bits of the attribute word are not known yet; this records them from a real run without logging every allocation,
// which would fill the file. Off: one relaxed load per call.
void note_attributes(bool is_alloc, uint64_t attributes, size_t size, const void* caller) {
    if (!rt::startup_trace::enabled()) return;
    static std::mutex mutex;
    static std::set<std::pair<bool, uint64_t>> seen;
    {
        const std::lock_guard lock(mutex);
        if (!seen.insert({is_alloc, attributes}).second) return;
    }
    std::string fields = std::string("\"api\":\"") + (is_alloc ? "XMemAlloc" : "XMemFree") +
                         "\",\"attributes\":" + rt::startup_trace::quote(rt::startup_trace::hex(attributes));
    if (is_alloc) fields += ",\"size\":" + std::to_string(size);
    fields += ",\"caller\":" +
              rt::startup_trace::quote(rt::startup_trace::where(reinterpret_cast<uint64_t>(caller)));
    rt::startup_trace::event("xmem_attributes", fields);
}

void* __stdcall XMemAlloc(size_t size, uint64_t attributes) {
    note_attributes(true, attributes, size, _ReturnAddress());
    return _aligned_malloc(size ? size : 1, kAlign);
}

void __stdcall XMemFree(void* p, uint64_t attributes) {
    note_attributes(false, attributes, 0, _ReturnAddress());
    _aligned_free(p);
}

// Measured call sites pass a zeroed 64-byte structure. Only the MEMORY_BASIC_INFORMATION-compatible prefix is
// known: callers read at least the base and size fields. Fill that prefix from VirtualQueryEx and zero the rest.
constexpr size_t kVirtualQueryOutSize = 64;
static_assert(sizeof(MEMORY_BASIC_INFORMATION) <= kVirtualQueryOutSize,
              "the 64-byte XMemVirtualQuery output must hold MEMORY_BASIC_INFORMATION");

// Returns the number of bytes written (the VirtualQuery convention) or 0 on failure. The exact return type and the
// meaning of every output field are not fully known; callers in the package only test for success.
SIZE_T __stdcall XMemVirtualQuery(HANDLE process, const void* address, void* out) {
    if (!out) return 0;
    std::memset(out, 0, kVirtualQueryOutSize);
    if (!address) return 0;

    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr,
                     "[rt] xmem: XMemVirtualQuery: fields after MEMORY_BASIC_INFORMATION are unknown; "
                     "the 64-byte output is zero-filled there\n");
    }

    // A null handle means the current process; GetCurrentProcess() is the (HANDLE)-1 pseudo handle.
    HANDLE target = process ? process : GetCurrentProcess();
    MEMORY_BASIC_INFORMATION info{};
    SIZE_T written = VirtualQueryEx(target, address, &info, sizeof(info));
    if (!written) return 0;

    std::memcpy(out, &info, written < kVirtualQueryOutSize ? written : kVirtualQueryOutSize);
    return written;
}

// BOOL XMemGetWorkingSetStatistics(UINT32 kind, XMEM_WORKING_SET_STATISTICS* stats). Measured at all seven call sites
// in amd_fidelityfx_upscaler_xbox.dll (1.0.2.1; calls at RVAs 0x26b3, 0x2ea1, 0x39a9, 0x7452, 0x74d2, 0x7568,
// 0x75b1): ECX is the constant 0x20000001 (its meaning is not known) and RDX points at a stack buffer (0x80 bytes
// where the frame shows it). The caller reads a 64-bit size at +8 and treats a zero return as "no data". Taking the
// pointer from the first argument wrote to address 0x20000001.
// The measured FSR sites establish argument shape, kind and the 64-bit working-set read at +8; +0's title-total memory
// limit interpretation is inferred from other Series implementations (xcl, xbox-emulator) and title engine telemetry
// sizing memory pools (RT-220). Note that if GlobalMemoryStatusEx reports 0 physical memory, XMem falls back to the
// 12 GiB title limit, whereas XSystemGetAppMemoryUsageLimit remains 0.
BOOL __stdcall XMemGetWorkingSetStatistics(uint32_t /*kind*/, void* stats) {
    if (!stats) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true, std::memory_order_relaxed)) {
            std::fprintf(stderr, "[rt] xmem: XMemGetWorkingSetStatistics(nullptr) -> FALSE\n");
        }
        return FALSE;
    }

    PROCESS_MEMORY_COUNTERS pmc{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return FALSE;

    MEMORYSTATUSEX ms{sizeof(ms)};
    uint64_t total_memory = 12ULL * 1024 * 1024 * 1024;
    if (GlobalMemoryStatusEx(&ms) && ms.ullTotalPhys != 0) {
        total_memory = ms.ullTotalPhys;
    }

    uint8_t* bytes = static_cast<uint8_t*>(stats);
    std::memset(bytes, 0, 16);
    std::memcpy(bytes + 0, &total_memory, sizeof(total_memory));
    uint64_t working_set = pmc.WorkingSetSize;
    std::memcpy(bytes + 8, &working_set, sizeof(working_set));
    return TRUE;
}

// The console's allocation type selects the page size with bits the host reads differently, so they are never
// forwarded as Win32 flags:
//   MEM_LARGE_PAGES (0x20000000) alone                        2 MiB pages
//   MEM_LARGE_PAGES | MEM_PHYSICAL (0x20400000)               64 KiB pages
//   MEM_4MB_PAGES (0x80000000), with or without MEM_LARGE_PAGES  4 MiB pages
// Values: the GDK's names as read in OpenScarlett (include/openscarlett/kernel/kernelx_abi.h, MIT) and xbox-emulator
// (crates/compat/xmem/src/virt.rs, GPL-3.0-or-later), which also picked this approach: on the host, MEM_LARGE_PAGES
// needs SeLockMemoryPrivilege and MEM_4MB_PAGES is rejected outright, so the shim serves every page size from
// ordinary pages and keeps what a title can see of the page size: the base is aligned to it and the region is a
// whole number of pages. The host still pages 4 KiB at a time underneath, which a title cannot observe except through
// TLB timing. Any other combination of these bits fails with ERROR_INVALID_PARAMETER.
constexpr DWORD kXmemLargePages = 0x20000000;
constexpr DWORD kXmemPhysical = 0x00400000;
constexpr DWORD kXmem64kPages = kXmemLargePages | kXmemPhysical;
constexpr DWORD kXmem4MbPages = 0x80000000;
constexpr DWORD kXmemPageModeMask = kXmem64kPages | kXmem4MbPages;
constexpr SIZE_T kXmemPageSize64k = 0x10000;
constexpr SIZE_T kXmemPageSize2M = 0x200000;
constexpr SIZE_T kXmemPageSize4M = 0x400000;
constexpr DWORD kXmemHostAllocationTypeMask =
    MEM_COMMIT | MEM_RESERVE | MEM_RESET | MEM_TOP_DOWN | MEM_WRITE_WATCH | 0x01000000;

// The guest page size the allocation type asks for, or 0 for the host's own pages. False for a combination that names
// no page size.
bool guest_page_size(DWORD allocation_type, SIZE_T& page) {
    switch (allocation_type & kXmemPageModeMask) {
        case 0: page = 0; return true;
        case kXmem64kPages: page = kXmemPageSize64k; return true;
        case kXmemLargePages: page = kXmemPageSize2M; return true;
        case kXmem4MbPages:
        case kXmem4MbPages | kXmemLargePages: page = kXmemPageSize4M; return true;
        default: return false;
    }
}

bool round_up_size(SIZE_T value, SIZE_T alignment, SIZE_T& rounded) {
    const SIZE_T max = (std::numeric_limits<SIZE_T>::max)();
    if (alignment == 0 || (alignment & (alignment - 1)) != 0 || value > max - (alignment - 1)) return false;
    rounded = (value + alignment - 1) & ~(alignment - 1);
    return true;
}

// VirtualAlloc2 (Windows 10 1803 and later; Wine has it too) reserves at an alignment in one call. It is looked up
// rather than imported so the runtime still loads where it is missing. LEONIDA_XMEM_NO_VIRTUALALLOC2 (any value)
// makes the shim behave as if it were missing, so the test suite runs the fallback below as well (xmem_probe).
using VirtualAlloc2Fn = PVOID(WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*, ULONG);

VirtualAlloc2Fn virtual_alloc2() {
    static const VirtualAlloc2Fn fn = []() -> VirtualAlloc2Fn {
        if (GetEnvironmentVariableW(L"LEONIDA_XMEM_NO_VIRTUALALLOC2", nullptr, 0) != 0) return nullptr;
        HMODULE kernelbase = GetModuleHandleW(L"kernelbase.dll");
        return kernelbase ? reinterpret_cast<VirtualAlloc2Fn>(
                                reinterpret_cast<void*>(GetProcAddress(kernelbase, "VirtualAlloc2")))
                          : nullptr;
    }();
    return fn;
}

// A new region of `size` bytes (a multiple of `alignment`) whose base is a multiple of `alignment`, which is larger than
// the host's 64 KiB allocation granularity. Without VirtualAlloc2, or when it refuses the alignment parameter, reserve
// `size + alignment` to find a free range, release it and reserve the aligned part of it; another thread can take
// that range in between, so try again a bounded number of times (the method of xbox-emulator's reserve_aligned in
// crates/compat/xmem/src/sys.rs). Each region is a single host allocation, so VirtualFree(base, 0, MEM_RELEASE) and
// VirtualQuery see exactly what was returned.
void* reserve_aligned(SIZE_T size, SIZE_T alignment, DWORD host_type, DWORD protect) {
    if (const VirtualAlloc2Fn alloc2 = virtual_alloc2()) {
        MEM_ADDRESS_REQUIREMENTS requirements{};
        requirements.Alignment = alignment;
        MEM_EXTENDED_PARAMETER parameter{};
        parameter.Type = MemExtendedParameterAddressRequirements;
        parameter.Pointer = &requirements;
        if (void* p = alloc2(GetCurrentProcess(), nullptr, size, host_type | MEM_RESERVE, protect, &parameter, 1)) {
            return p;
        }
        // Only a refused parameter can mean an older host without alignment support; anything else is final.
        if (GetLastError() != ERROR_INVALID_PARAMETER) return nullptr;
    }

    if (size > (std::numeric_limits<SIZE_T>::max)() - alignment) {
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
        return nullptr;
    }
    const SIZE_T probe_size = size + alignment;
    for (int attempt = 0; attempt < 64; ++attempt) {
        void* const probe = VirtualAlloc(nullptr, probe_size, MEM_RESERVE | (host_type & MEM_TOP_DOWN), PAGE_NOACCESS);
        if (!probe) return nullptr;
        const uintptr_t start = reinterpret_cast<uintptr_t>(probe);
        uintptr_t aligned = (start + alignment - 1) & ~(static_cast<uintptr_t>(alignment) - 1);
        // Top-down: take the highest aligned base inside the probe, as the host would have.
        if (host_type & MEM_TOP_DOWN) aligned = (start + alignment) & ~(static_cast<uintptr_t>(alignment) - 1);
        VirtualFree(probe, 0, MEM_RELEASE);
        if (void* p = VirtualAlloc(reinterpret_cast<void*>(aligned), size, host_type | MEM_RESERVE, protect)) return p;
        const DWORD error = GetLastError();
        if (error != ERROR_INVALID_ADDRESS) return nullptr;  // anything but losing the race is final
    }
    SetLastError(ERROR_NOT_ENOUGH_MEMORY);
    return nullptr;
}

// XMemVirtualAlloc's and a title's VirtualAlloc's common part. `default_operation`: no MEM_RESERVE/MEM_COMMIT/MEM_RESET
// bit means reserve and commit (XMemVirtualAlloc); a title's VirtualAlloc requires an operation, as on Win32.
void* console_virtual_alloc(void* address, SIZE_T size, DWORD allocation_type, DWORD protect, bool default_operation) {
    if (size == 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }
    SIZE_T guest_page = 0;
    if (!guest_page_size(allocation_type, guest_page)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    const DWORD unsupported = allocation_type & ~(kXmemHostAllocationTypeMask | kXmemPageModeMask);
    if (unsupported != 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    DWORD host_type = allocation_type & kXmemHostAllocationTypeMask;
    if (host_type == 0) {
        if (!default_operation) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return nullptr;
        }
        host_type = MEM_RESERVE | MEM_COMMIT;
    }

    SYSTEM_INFO system_info{};
    GetSystemInfo(&system_info);
    const SIZE_T page_granularity = guest_page ? guest_page : system_info.dwPageSize;
    const SIZE_T base_granularity =
        guest_page ? guest_page
                   : ((host_type & MEM_RESERVE) ? system_info.dwAllocationGranularity : system_info.dwPageSize);
    if (page_granularity == 0 || (page_granularity & (page_granularity - 1)) != 0 || base_granularity == 0 ||
        (base_granularity & (base_granularity - 1)) != 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    if (address != nullptr) {
        // Round the range out to whole guest pages: a commit inside a 2 MiB reservation commits that whole page. The
        // page size comes from this call's bits; nothing remembers the page size a region was reserved with.
        const uintptr_t requested = reinterpret_cast<uintptr_t>(address);
        const uintptr_t max = (std::numeric_limits<uintptr_t>::max)();
        if (size > max - requested) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return nullptr;
        }
        const uintptr_t requested_end = requested + size;
        const uintptr_t base = requested & ~(static_cast<uintptr_t>(base_granularity) - 1);
        if (requested_end > max - (static_cast<uintptr_t>(page_granularity) - 1)) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return nullptr;
        }
        const uintptr_t end = (requested_end + static_cast<uintptr_t>(page_granularity) - 1) &
                              ~(static_cast<uintptr_t>(page_granularity) - 1);
        address = reinterpret_cast<void*>(base);
        size = static_cast<SIZE_T>(end - base);
    } else if (guest_page != 0 && !round_up_size(size, page_granularity, size)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    // A new region with pages coarser than the host's 64 KiB allocation granularity needs an aligned reservation.
    // (MEM_COMMIT without MEM_RESERVE and no address also makes a new region on Win32.) MEM_RESET with no address is
    // left to the host to refuse.
    if (address == nullptr && guest_page > system_info.dwAllocationGranularity &&
        (host_type & (MEM_RESERVE | MEM_COMMIT))) {
        return reserve_aligned(size, guest_page, host_type, protect);
    }
    return VirtualAlloc(address, size, host_type, protect);
}

void* WINAPI XMemVirtualAlloc(void* address, SIZE_T size, DWORD allocation_type, ULONGLONG attributes, DWORD protect) {
    // Host-backed support covers ordinary, 64 KiB, 2 MiB and 4 MiB guest pages. Nonzero attribute flags still fail
    // until their host mapping is known.
    if (size == 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }
    if (attributes != 0) {
        SetLastError(ERROR_NOT_SUPPORTED);
        return nullptr;
    }
    return console_virtual_alloc(address, size, allocation_type, protect, true);
}

UINT WINAPI XMemGetWriteWatch(DWORD flags, void* base_address, SIZE_T region_size, void** addresses, ULONG_PTR* count,
                              ULONG* granularity) {
    return GetWriteWatch(flags, base_address, region_size, addresses, count, granularity);
}

UINT WINAPI XMemResetWriteWatch(void* base_address, SIZE_T region_size) {
    return ResetWriteWatch(base_address, region_size);
}

// kernel32!VirtualAlloc as a title calls it. The console's allocation type has two bits of its own, MEM_GRAPHICS
// (0x10000000, memory the GPU reads) and MEM_TITLE (0x40000000, the title's own heap). On a PC the host rejects both
// with ERROR_INVALID_PARAMETER (measured on Windows 11 26200: VirtualAlloc(NULL, 64 KiB, MEM_COMMIT | MEM_RESERVE |
// 0x10000000, PAGE_READWRITE) returns NULL; so does 0x40000000), so an engine allocating its heaps this way gets
// NULL at boot. Both are hints about where the memory lives; the host has no such split, so they are dropped and
// the rest is passed through. A page-size request (MEM_LARGE_PAGES or MEM_4MB_PAGES, see above) is the console's,
// not the host's MEM_LARGE_PAGES, which needs SeLockMemoryPrivilege: it goes through XMemVirtualAlloc's page
// handling, keeping Win32's rule that an operation is required. MEM_PHYSICAL without MEM_LARGE_PAGES is passed
// through. Not done: the console protection bits, and whether graphics memory has to come from an arena the D3D12
// layer can see (#313).
constexpr DWORD kMemGraphics = 0x10000000;
constexpr DWORD kMemTitle = 0x40000000;

LPVOID WINAPI TitleVirtualAlloc(LPVOID address, SIZE_T size, DWORD allocation_type, DWORD protect) {
    allocation_type &= ~(kMemGraphics | kMemTitle);
    if (allocation_type & (kXmemLargePages | kXmem4MbPages)) {
        return console_virtual_alloc(address, size, allocation_type, protect, false);
    }
    return VirtualAlloc(address, size, allocation_type, protect);
}

}  // namespace

#define TITLE_VIRTUAL_ALLOC(dll) {dll, "VirtualAlloc", reinterpret_cast<void*>(&TitleVirtualAlloc)}

// The classic libraries and the memory api sets a title's imports name.
SHIM_TABLE(title_memory,
    TITLE_VIRTUAL_ALLOC("kernel32.dll"),
    TITLE_VIRTUAL_ALLOC("kernelbase.dll"),
    TITLE_VIRTUAL_ALLOC("api-ms-win-core-memory-l1-1-0.dll"),
    TITLE_VIRTUAL_ALLOC("api-ms-win-core-memory-l1-1-1.dll"),
    TITLE_VIRTUAL_ALLOC("api-ms-win-core-memory-l1-1-2.dll"),
    TITLE_VIRTUAL_ALLOC("api-ms-win-core-memory-l1-1-3.dll"),
    TITLE_VIRTUAL_ALLOC("api-ms-win-core-memory-l1-1-4.dll"),
    TITLE_VIRTUAL_ALLOC("api-ms-win-core-memory-l1-1-5.dll"),
    TITLE_VIRTUAL_ALLOC("api-ms-win-core-memory-l1-1-6.dll"),
);

SHIM_TABLE(xmem,
    {"xmem.dll", "XMemAlloc", reinterpret_cast<void*>(&XMemAlloc)},
    {"xmem.dll", "XMemFree", reinterpret_cast<void*>(&XMemFree)},
    {"xmem.dll", "XMemVirtualQuery", reinterpret_cast<void*>(&XMemVirtualQuery)},
    {"xmem.dll", "XMemGetWorkingSetStatistics", reinterpret_cast<void*>(&XMemGetWorkingSetStatistics)},
    {"xmem.dll", "XMemVirtualAlloc", reinterpret_cast<void*>(&XMemVirtualAlloc)},
    {"xmem.dll", "XMemGetWriteWatch", reinterpret_cast<void*>(&XMemGetWriteWatch)},
    {"xmem.dll", "XMemResetWriteWatch", reinterpret_cast<void*>(&XMemResetWriteWatch)},
);
