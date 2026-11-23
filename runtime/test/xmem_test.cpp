// Checks the registered xmem.dll shim against the measured Xbox ABI.
//
// Every function is resolved through rt::shims::find, so the test exercises the same shim table and calling
// convention the runtime installs for xmem.dll imports instead of a private copy of the implementation. The checks
// cover what the measured call sites use: 64-byte-aligned allocations, XMemFree, working-set statistics with the
// size at +8, and XMemVirtualQuery's MEMORY_BASIC_INFORMATION prefix inside a zeroed 64-byte output, plus the
// console's 64 KiB, 2 MiB and 4 MiB page modes in XMemVirtualAlloc and a title's VirtualAlloc.
#include <windows.h>
#include <psapi.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "shims.h"
#include "startup_trace.h"

namespace {

using XMemAllocFn = void*(__stdcall*)(size_t, uint64_t);
using XMemFreeFn = void(__stdcall*)(void*, uint64_t);
using XMemVirtualQueryFn = SIZE_T(__stdcall*)(HANDLE, const void*, void*);
// BOOL XMemGetWorkingSetStatistics(UINT32 kind, stats*): the FSR DLL passes kind 0x20000001 at every call site.
using XMemGetWorkingSetStatisticsFn = BOOL(__stdcall*)(uint32_t, void*);
using XMemVirtualAllocFn = void*(WINAPI*)(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
using XMemGetWriteWatchFn = UINT(WINAPI*)(DWORD, void*, SIZE_T, void**, ULONG_PTR*, ULONG*);
using XMemResetWriteWatchFn = UINT(WINAPI*)(void*, SIZE_T);
constexpr uint32_t kMeasuredKind = 0x20000001;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

bool aligned_64(const void* p) { return p && (reinterpret_cast<uintptr_t>(p) & 63u) == 0; }

void test_alloc(XMemAllocFn alloc, XMemFreeFn free_mem) {
    void* const zero = alloc(0, 0);
    check(zero != nullptr, "XMemAlloc(0, 0) returns non-null");
    check(aligned_64(zero), "XMemAlloc(0, 0) returns a 64-byte aligned block");

    bool zero_readable = false;
    if (zero) {
        auto* bytes = static_cast<uint8_t*>(zero);
        bytes[0] = 0xa5;
        zero_readable = bytes[0] == 0xa5;
    }
    check(zero_readable, "XMemAlloc(0, 0) block is writable and readable");

    void* const big = alloc(4096, 0x5e50000);
    check(big != nullptr, "XMemAlloc(4096, 0x5e50000) returns non-null");
    check(aligned_64(big), "XMemAlloc(4096, 0x5e50000) returns a 64-byte aligned block");

    bool pattern_ok = big != nullptr;
    if (big) {
        auto* bytes = static_cast<uint8_t*>(big);
        for (size_t i = 0; i < 4096; ++i) bytes[i] = static_cast<uint8_t>((i * 37u + 11u) & 0xffu);
        for (size_t i = 0; i < 4096; ++i) {
            if (bytes[i] != static_cast<uint8_t>((i * 37u + 11u) & 0xffu)) {
                pattern_ok = false;
                break;
            }
        }
    }
    check(pattern_ok, "XMemAlloc(4096, 0x5e50000) block holds a write/read pattern");

    free_mem(zero, 0);
    free_mem(big, 0x5e50000);
    check(true, "XMemFree accepts both blocks (no crash)");
}

void test_working_set(XMemGetWorkingSetStatisticsFn get_stats) {
    alignas(64) uint8_t stats[64];
    std::memset(stats, 0xa5, sizeof(stats));
    // The pointer is the second argument: the first (0x20000001 here, as the FSR DLL passes it) must never be written
    // through, which is what the one-argument form did.
    const BOOL ok = get_stats(kMeasuredKind, stats);
    check(ok != FALSE, "XMemGetWorkingSetStatistics(kind, 64-byte out) returns TRUE");

    uint64_t total_memory = 0;
    std::memcpy(&total_memory, stats + 0, sizeof(total_memory));
    check(ok != FALSE && total_memory != 0, "XMemGetWorkingSetStatistics writes a non-zero total memory at +0");

    MEMORYSTATUSEX ms{sizeof(ms)};
    if (GlobalMemoryStatusEx(&ms) && ms.ullTotalPhys != 0) {
        check(ok != FALSE && total_memory == ms.ullTotalPhys,
              "XMemGetWorkingSetStatistics total memory (+0) matches GlobalMemoryStatusEx ullTotalPhys");
    } else {
        check(ok != FALSE && total_memory == 12ULL * 1024 * 1024 * 1024,
              "XMemGetWorkingSetStatistics total memory (+0) falls back to 12 GiB title limit");
    }

    uint64_t working_set = 0;
    std::memcpy(&working_set, stats + 8, sizeof(working_set));
    check(ok != FALSE && working_set != 0, "XMemGetWorkingSetStatistics writes a non-zero size at +8");

    check(ok != FALSE && total_memory >= working_set,
          "XMemGetWorkingSetStatistics total memory (+0) >= working set (+8)");

    bool tail_untouched = true;
    for (size_t i = 16; i < sizeof(stats); ++i) {
        if (stats[i] != 0xa5) {
            tail_untouched = false;
            break;
        }
    }
    check(ok != FALSE && tail_untouched, "XMemGetWorkingSetStatistics leaves bytes 16..63 untouched");

    check(get_stats(kMeasuredKind, nullptr) == FALSE,
          "XMemGetWorkingSetStatistics(kind, nullptr) returns FALSE without crashing");
}

void test_virtual_query(XMemVirtualQueryFn query) {
    uint8_t local = 0;
    const void* const local_address = &local;

    alignas(64) uint8_t out[64];
    std::memset(out, 0xa5, sizeof(out));
    const SIZE_T written = query(GetCurrentProcess(), local_address, out);
    check(written != 0, "XMemVirtualQuery(current process, stack local) returns non-zero");

    MEMORY_BASIC_INFORMATION info{};
    std::memcpy(&info, out, sizeof(info));
    const uintptr_t base = reinterpret_cast<uintptr_t>(info.BaseAddress);
    const uintptr_t address = reinterpret_cast<uintptr_t>(local_address);
    const bool contains_local =
        base != 0 && info.RegionSize != 0 && address >= base && address - base < info.RegionSize;
    check(written != 0 && contains_local,
          "XMemVirtualQuery fills MEMORY_BASIC_INFORMATION with a region containing the local");

    bool tail_zero = true;
    for (size_t i = sizeof(info); i < sizeof(out); ++i) {
        if (out[i] != 0) {
            tail_zero = false;
            break;
        }
    }
    check(written != 0 && tail_zero, "XMemVirtualQuery leaves the 64-byte output tail zero");

    alignas(64) uint8_t bad_out[64];
    std::memset(bad_out, 0xa5, sizeof(bad_out));
    // Address 1 is queryable (it falls in the free region starting at 0), so use a non-canonical address that
    // VirtualQueryEx rejects.
    const void* const unmappable = reinterpret_cast<const void*>(~static_cast<uintptr_t>(0));
    const SIZE_T bad = query(GetCurrentProcess(), unmappable, bad_out);
    check(bad == 0, "XMemVirtualQuery on an invalid address returns 0 without crashing");
}

void test_virtual_alloc(XMemVirtualAllocFn virtual_alloc, XMemGetWriteWatchFn get_write_watch,
                       XMemResetWriteWatchFn reset_write_watch) {
    SYSTEM_INFO system_info{};
    GetSystemInfo(&system_info);
    const SIZE_T page_size = system_info.dwPageSize;
    check(page_size != 0, "GetSystemInfo reports a non-zero page size");
    if (page_size == 0) return;

    const SIZE_T region_size = page_size * 3;
    void* const base = virtual_alloc(nullptr, region_size, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, 0,
                                     PAGE_READWRITE);
    check(base != nullptr, "XMemVirtualAlloc creates a synthetic write-watch region");
    if (!base) return;

    void* watched[4]{};
    ULONG_PTR count = 4;
    ULONG granularity = 0;
    UINT status = get_write_watch(0, base, region_size, watched, &count, &granularity);
    check(status == NO_ERROR && count == 0, "new write-watch region starts with no dirty pages");
    check(status == NO_ERROR && granularity == page_size, "write-watch reports the host page granularity");

    auto* const bytes = static_cast<volatile uint8_t*>(base);
    bytes[0] = 0xa1;
    bytes[page_size * 2] = 0xb2;

    count = 4;
    status = get_write_watch(0, base, region_size, watched, &count, &granularity);
    bool saw_first_page = false;
    bool saw_third_page = false;
    bool all_addresses_are_page_aligned = count <= 4;
    for (ULONG_PTR i = 0; i < count && i < 4; ++i) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(watched[i]);
        const uintptr_t start = reinterpret_cast<uintptr_t>(base);
        all_addresses_are_page_aligned &= address >= start && (address - start) % page_size == 0;
        saw_first_page |= address == start;
        saw_third_page |= address == start + page_size * 2;
    }
    check(status == NO_ERROR && count == 2 && saw_first_page && saw_third_page,
          "write-watch reports each written page by its relative address");
    check(all_addresses_are_page_aligned, "write-watch addresses are page aligned and within the region");

    count = 4;
    status = get_write_watch(0, base, region_size, watched, &count, &granularity);
    check(status == NO_ERROR && count == 2, "query-only write-watch leaves dirty pages recorded");

    status = reset_write_watch(base, region_size);
    check(status == NO_ERROR, "XMemResetWriteWatch clears the synthetic region");
    count = 4;
    status = get_write_watch(0, base, region_size, watched, &count, &granularity);
    check(status == NO_ERROR && count == 0, "reset write-watch query returns no dirty pages");

    bytes[page_size] = 0xc3;
    count = 4;
    status = get_write_watch(0, base, region_size, watched, &count, &granularity);
    check(status == NO_ERROR && count == 1 &&
              reinterpret_cast<uintptr_t>(watched[0]) == reinterpret_cast<uintptr_t>(base) + page_size,
          "write-watch records writes again after reset");

    check(VirtualFree(base, 0, MEM_RELEASE), "release the synthetic write-watch region");

    SetLastError(ERROR_SUCCESS);
    void* const attributed = virtual_alloc(nullptr, page_size, MEM_RESERVE | MEM_COMMIT, 1, PAGE_READWRITE);
    check(attributed == nullptr && GetLastError() == ERROR_NOT_SUPPORTED,
          "nonzero XMem allocation attributes are rejected before host allocation");
}

// gta6rt --trace: one "xmem_attributes" event per distinct raw attribute word and API, none with the trace off.
void test_attribute_trace(XMemAllocFn alloc, XMemFreeFn free_mem) {
    wchar_t temp[MAX_PATH];
    GetTempPathW(MAX_PATH, temp);
    const std::wstring path =
        std::wstring(temp) + L"xmem_trace_test_" + std::to_wstring(GetCurrentProcessId()) + L".jsonl";
    DeleteFileW(path.c_str());

    void* untraced = alloc(64, 0x1111);  // trace not open yet: nothing to write, nothing to crash
    check(untraced != nullptr, "XMemAlloc with the trace off still allocates");
    free_mem(untraced, 0x1111);

    check(rt::startup_trace::open(path.c_str()), "open the startup trace file");
    void* a = alloc(4096, 0x5e50000);
    void* b = alloc(8192, 0x5e50000);  // same word again: no second event
    void* c = alloc(16, 0x99);
    free_mem(a, 0x5e50000);
    free_mem(b, 0x5e50000);  // same word again: no second event
    free_mem(c, 0x99);
    free_mem(nullptr, 0x77);  // a free-only word is recorded too
    rt::startup_trace::close();

    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    const std::string log = text.str();
    auto count = [&](const std::string& needle) {
        int n = 0;
        for (size_t at = log.find(needle); at != std::string::npos; at = log.find(needle, at + 1)) ++n;
        return n;
    };
    check(count("\"kind\":\"xmem_attributes\"") == 5, "five events: two XMemAlloc words and three XMemFree words");
    check(count("\"api\":\"XMemAlloc\",\"attributes\":\"0x5e50000\",\"size\":4096") == 1,
          "XMemAlloc 0x5e50000 is recorded once, with the size of the first call");
    check(count("\"api\":\"XMemAlloc\",\"attributes\":\"0x99\",\"size\":16") == 1, "XMemAlloc 0x99 is recorded once");
    check(count("\"api\":\"XMemFree\",\"attributes\":\"0x5e50000\"") == 1, "XMemFree 0x5e50000 is recorded once");
    check(count("\"api\":\"XMemFree\",\"attributes\":\"0x77\"") == 1, "XMemFree 0x77 is recorded");
    check(count("\"attributes\":\"0x1111\"") == 0, "nothing was recorded while the trace was off");
    check(count("\"caller\":\"") == 5, "every event names its caller");
    DeleteFileW(path.c_str());
}

// The console's page-size bits in the allocation type: MEM_LARGE_PAGES alone is 2 MiB pages, MEM_4MB_PAGES (with or
// without MEM_LARGE_PAGES) 4 MiB pages, and MEM_LARGE_PAGES | MEM_PHYSICAL 64 KiB pages. None of them needs a host
// privilege: the region is ordinary pages, reserved at the page size's alignment and rounded to whole pages.
constexpr DWORD kLargePages = 0x20000000;
constexpr DWORD k4MbPages = 0x80000000;
constexpr DWORD k64kPages = 0x20400000;
constexpr SIZE_T k2MiB = 0x200000;
constexpr SIZE_T k4MiB = 0x400000;

struct PageMode {
    DWORD bits;
    SIZE_T page;
    const char* name;
};
constexpr PageMode kLargeModes[] = {
    {kLargePages, k2MiB, "2 MiB (0x20000000)"},
    {k4MbPages, k4MiB, "4 MiB (0x80000000)"},
    {k4MbPages | kLargePages, k4MiB, "4 MiB (0xa0000000)"},
};

bool is_aligned(const void* p, SIZE_T alignment) {
    return p && (reinterpret_cast<uintptr_t>(p) & (alignment - 1)) == 0;
}

// One run of pages as VirtualQuery sees it from `base`: exactly `size` bytes in `state`, inside the allocation that
// starts at `allocation` (by default `base` itself). Writes the first and last byte when it is committed.
bool region_is(void* base, SIZE_T size, DWORD state, void* allocation = nullptr) {
    MEMORY_BASIC_INFORMATION info{};
    if (VirtualQuery(base, &info, sizeof(info)) != sizeof(info)) return false;
    if (!allocation) allocation = base;
    if (info.BaseAddress != base || info.AllocationBase != allocation || info.RegionSize != size ||
        info.State != state) {
        std::printf("  query %p: base %p allocation base %p size %#zx state %#lx\n", base, info.BaseAddress,
                    info.AllocationBase, static_cast<size_t>(info.RegionSize), static_cast<unsigned long>(info.State));
        return false;
    }
    if (state == MEM_COMMIT) {
        auto* const bytes = static_cast<volatile uint8_t*>(base);
        bytes[0] = 0x11;
        bytes[size - 1] = 0x22;
        return bytes[0] == 0x11 && bytes[size - 1] == 0x22;
    }
    return true;
}

bool released(void* base) {
    if (!VirtualFree(base, 0, MEM_RELEASE)) return false;
    MEMORY_BASIC_INFORMATION info{};
    return VirtualQuery(base, &info, sizeof(info)) == sizeof(info) && info.State == MEM_FREE;
}

void test_virtual_alloc_large_pages(XMemVirtualAllocFn virtual_alloc) {
    char what[200];
    for (const PageMode& mode : kLargeModes) {
        // Not a whole page, so the size has to round up: 1.5 pages -> 2 pages.
        const SIZE_T request = mode.page + mode.page / 2;
        const SIZE_T rounded = mode.page * 2;
        SetLastError(ERROR_SUCCESS);
        void* const p = virtual_alloc(nullptr, request, MEM_RESERVE | MEM_COMMIT | mode.bits, 0, PAGE_READWRITE);
        std::snprintf(what, sizeof what, "XMemVirtualAlloc with %s pages returns memory (error %lu)", mode.name,
                      p ? 0ul : static_cast<unsigned long>(GetLastError()));
        check(p != nullptr, what);
        if (!p) continue;
        std::snprintf(what, sizeof what, "XMemVirtualAlloc with %s pages is aligned to the page size", mode.name);
        check(is_aligned(p, mode.page), what);
        std::snprintf(what, sizeof what, "XMemVirtualAlloc with %s pages commits whole pages", mode.name);
        check(region_is(p, rounded, MEM_COMMIT), what);
        std::snprintf(what, sizeof what, "VirtualFree releases a %s page region", mode.name);
        check(released(p), what);

        // No operation bits: XMemVirtualAlloc reserves and commits, as for the other page modes.
        void* const defaulted = virtual_alloc(nullptr, 1, mode.bits, 0, PAGE_READWRITE);
        std::snprintf(what, sizeof what, "XMemVirtualAlloc(1 byte, %s pages, no operation) commits one page",
                      mode.name);
        check(is_aligned(defaulted, mode.page) && region_is(defaulted, mode.page, MEM_COMMIT), what);
        if (defaulted) VirtualFree(defaulted, 0, MEM_RELEASE);

        // Reserve, then commit by an address inside the second page: the commit covers that whole page and starts at
        // its base, the console's commit granularity.
        void* const reserved =
            virtual_alloc(nullptr, mode.page * 3, MEM_RESERVE | MEM_TOP_DOWN | mode.bits, 0, PAGE_READWRITE);
        std::snprintf(what, sizeof what, "a %s page reservation with MEM_TOP_DOWN is aligned and stays reserved",
                      mode.name);
        check(is_aligned(reserved, mode.page) && region_is(reserved, mode.page * 3, MEM_RESERVE), what);
        if (!reserved) continue;
        auto* const second = static_cast<uint8_t*>(reserved) + mode.page;
        void* const committed = virtual_alloc(second + 0x1234, 0x10, MEM_COMMIT | mode.bits, 0, PAGE_READWRITE);
        std::snprintf(what, sizeof what, "committing inside a %s page commits that page from its base", mode.name);
        check(committed == second && region_is(second, mode.page, MEM_COMMIT, reserved), what);
        std::snprintf(what, sizeof what, "the pages around a %s page commit stay reserved", mode.name);
        check(region_is(reserved, mode.page, MEM_RESERVE) &&
                  region_is(second + mode.page, mode.page, MEM_RESERVE, reserved),
              what);
        std::snprintf(what, sizeof what, "VirtualFree releases the whole %s page reservation", mode.name);
        check(released(reserved), what);
    }

    // Combinations no page size answers stay invalid, before anything is allocated.
    for (const DWORD bits : {DWORD{k64kPages | k4MbPages}, DWORD{k4MbPages | 0x00400000}, DWORD{0x00400000}}) {
        SetLastError(ERROR_SUCCESS);
        void* const p = virtual_alloc(nullptr, k4MiB, MEM_RESERVE | MEM_COMMIT | bits, 0, PAGE_READWRITE);
        std::snprintf(what, sizeof what, "XMemVirtualAlloc rejects page bits %#lx with ERROR_INVALID_PARAMETER",
                      static_cast<unsigned long>(bits));
        check(p == nullptr && GetLastError() == ERROR_INVALID_PARAMETER, what);
        if (p) VirtualFree(p, 0, MEM_RELEASE);
    }

    SetLastError(ERROR_SUCCESS);
    void* const attributed = virtual_alloc(nullptr, k2MiB, MEM_RESERVE | MEM_COMMIT | kLargePages, 1, PAGE_READWRITE);
    check(attributed == nullptr && GetLastError() == ERROR_NOT_SUPPORTED,
          "a nonzero attribute word is still rejected with 2 MiB pages");
}

using HostVirtualAllocFn = void*(WINAPI*)(void*, SIZE_T, DWORD, DWORD);

// RT-217: a title's kernel32!VirtualAlloc carries the console's MEM_GRAPHICS / MEM_TITLE type bits, which the host
// rejects. The registered shim has to drop them and still allocate.
void test_title_virtual_alloc() {
    constexpr DWORD kMemGraphics = 0x10000000;
    constexpr DWORD kMemTitle = 0x40000000;
    const char* const dlls[] = {"kernel32.dll", "kernelbase.dll", "api-ms-win-core-memory-l1-1-0.dll"};
    for (const char* dll : dlls) {
        const auto alloc = reinterpret_cast<HostVirtualAllocFn>(rt::shims::find(dll, "VirtualAlloc"));
        char what[160];
        std::snprintf(what, sizeof what, "registry exposes %s!VirtualAlloc", dll);
        check(alloc != nullptr, what);
        if (!alloc) continue;

        for (const DWORD bits : {kMemGraphics, kMemTitle, kMemGraphics | kMemTitle, DWORD{0}}) {
            void* const p = alloc(nullptr, 0x10000, MEM_RESERVE | MEM_COMMIT | bits, PAGE_READWRITE);
            std::snprintf(what, sizeof what, "%s!VirtualAlloc with type bits %#lx returns memory", dll,
                          static_cast<unsigned long>(bits));
            check(p != nullptr, what);
            if (!p) continue;
            static_cast<volatile uint8_t*>(p)[0] = 0x5a;
            check(static_cast<volatile uint8_t*>(p)[0] == 0x5a, "the allocation is committed and writable");
            check(VirtualFree(p, 0, MEM_RELEASE), "release the allocation");
        }
    }
}

// The same page-size bits through a title's kernel32!VirtualAlloc. The host reads 0x20000000 as its own
// MEM_LARGE_PAGES (which needs SeLockMemoryPrivilege) and rejects 0x80000000, so these failed before the shim
// handled them. Win32 rules stay: an operation (reserve or commit) is required.
void test_title_virtual_alloc_large_pages() {
    constexpr DWORD kMemGraphics = 0x10000000;
    constexpr DWORD kMemTitle = 0x40000000;
    const char* const dlls[] = {"kernel32.dll", "kernelbase.dll", "api-ms-win-core-memory-l1-1-0.dll"};
    char what[200];
    for (const char* dll : dlls) {
        const auto alloc = reinterpret_cast<HostVirtualAllocFn>(rt::shims::find(dll, "VirtualAlloc"));
        if (!alloc) continue;  // test_title_virtual_alloc reports the missing registration
        for (const PageMode& mode : kLargeModes) {
            for (const DWORD tags : {DWORD{0}, DWORD{kMemGraphics | kMemTitle}}) {
                SetLastError(ERROR_SUCCESS);
                void* const p = alloc(nullptr, mode.page + 1, MEM_RESERVE | MEM_COMMIT | mode.bits | tags,
                                      PAGE_READWRITE);
                std::snprintf(what, sizeof what, "%s!VirtualAlloc, %s pages, tags %#lx: aligned, two pages (error %lu)",
                              dll, mode.name, static_cast<unsigned long>(tags),
                              p ? 0ul : static_cast<unsigned long>(GetLastError()));
                check(is_aligned(p, mode.page) && region_is(p, mode.page * 2, MEM_COMMIT), what);
                if (p) check(released(p), "release the title's large-page region");
            }
        }

        void* const sixty_four = alloc(nullptr, 0x1001, MEM_RESERVE | MEM_COMMIT | k64kPages, PAGE_READWRITE);
        std::snprintf(what, sizeof what, "%s!VirtualAlloc with 64 KiB pages commits one 64 KiB page", dll);
        check(is_aligned(sixty_four, 0x10000) && region_is(sixty_four, 0x10000, MEM_COMMIT), what);
        if (sixty_four) VirtualFree(sixty_four, 0, MEM_RELEASE);

        SetLastError(ERROR_SUCCESS);
        void* const no_operation = alloc(nullptr, k2MiB, kLargePages, PAGE_READWRITE);
        std::snprintf(what, sizeof what, "%s!VirtualAlloc with 2 MiB pages and no operation fails as on Win32", dll);
        check(no_operation == nullptr && GetLastError() == ERROR_INVALID_PARAMETER, what);
        if (no_operation) VirtualFree(no_operation, 0, MEM_RELEASE);
    }
}

void print_result() {
    if (g_failures == 0) {
        std::printf("xmem_test: OK (%d checks)\n", g_checks);
    } else {
        std::printf("xmem_test: FAILED (%d checks, %d failure%s)\n", g_checks, g_failures,
                    g_failures == 1 ? "" : "s");
    }
}

}  // namespace

int main() {
    std::printf("xmem_test: registered xmem.dll shim ABI\n");

    const auto alloc = reinterpret_cast<XMemAllocFn>(rt::shims::find("xmem.dll", "XMemAlloc"));
    const auto free_mem = reinterpret_cast<XMemFreeFn>(rt::shims::find("xmem.dll", "XMemFree"));
    const auto query = reinterpret_cast<XMemVirtualQueryFn>(rt::shims::find("xmem.dll", "XMemVirtualQuery"));
    const auto get_stats =
        reinterpret_cast<XMemGetWorkingSetStatisticsFn>(rt::shims::find("xmem.dll", "XMemGetWorkingSetStatistics"));
    const auto virtual_alloc =
        reinterpret_cast<XMemVirtualAllocFn>(rt::shims::find("xmem.dll", "XMemVirtualAlloc"));
    const auto get_write_watch =
        reinterpret_cast<XMemGetWriteWatchFn>(rt::shims::find("xmem.dll", "XMemGetWriteWatch"));
    const auto reset_write_watch =
        reinterpret_cast<XMemResetWriteWatchFn>(rt::shims::find("xmem.dll", "XMemResetWriteWatch"));

    check(alloc != nullptr, "registry exposes xmem.dll!XMemAlloc");
    check(free_mem != nullptr, "registry exposes xmem.dll!XMemFree");
    check(query != nullptr, "registry exposes xmem.dll!XMemVirtualQuery");
    check(get_stats != nullptr, "registry exposes xmem.dll!XMemGetWorkingSetStatistics");
    check(virtual_alloc != nullptr, "registry exposes xmem.dll!XMemVirtualAlloc");
    check(get_write_watch != nullptr, "registry exposes xmem.dll!XMemGetWriteWatch");
    check(reset_write_watch != nullptr, "registry exposes xmem.dll!XMemResetWriteWatch");

    if (!alloc || !free_mem || !query || !get_stats) {
        print_result();
        return 1;
    }

    test_alloc(alloc, free_mem);
    test_working_set(get_stats);
    test_virtual_query(query);
    if (virtual_alloc && get_write_watch && reset_write_watch) {
        test_virtual_alloc(virtual_alloc, get_write_watch, reset_write_watch);
        test_virtual_alloc_large_pages(virtual_alloc);
    }
    test_attribute_trace(alloc, free_mem);

    test_title_virtual_alloc();
    test_title_virtual_alloc_large_pages();

    print_result();
    return g_failures ? 1 : 0;
}
