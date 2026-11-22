#include "pe_loader.h"

#include <windows.h>
#include <intrin.h>
#include <psapi.h>
#include <tlhelp32.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "../shims/shims.h"
#include "../shims/startup_trace.h"
#include "base_reservation.h"

namespace rt {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

namespace {

std::vector<uint8_t> read_file(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    return {std::istreambuf_iterator<char>(f), {}};
}

template <class T>
T* at(uint8_t* base, uint32_t rva) { return reinterpret_cast<T*>(base + rva); }

// Every RVA followed below comes out of the image itself, so each one is checked first. A truncated or corrupt
// image then fails with a message instead of reading, writing or protecting memory outside its allocation.
void require_range(const std::string& name, uint64_t rva, uint64_t len, uint32_t image_size, const char* what) {
    if (rva > image_size || len > image_size - rva)
        throw std::runtime_error(name + ": " + what + " runs past the image");
}

std::string read_string(const std::string& name, uint8_t* base, uint32_t image_size, uint32_t rva, const char* what) {
    if (rva >= image_size) throw std::runtime_error(name + ": " + what + " is outside the image");
    auto* s = reinterpret_cast<const char*>(base + rva);
    auto* end = static_cast<const char*>(std::memchr(s, 0, image_size - rva));
    if (!end) throw std::runtime_error(name + ": " + what + " isn't terminated inside the image");
    return {s, end};
}

// PE/COFF: NumberOfRvaAndSizes counts the data directories; entries past it are not directories, whatever they hold.
IMAGE_DATA_DIRECTORY directory(const IMAGE_NT_HEADERS64* nt, unsigned index) {
    return index < nt->OptionalHeader.NumberOfRvaAndSizes ? nt->OptionalHeader.DataDirectory[index] : IMAGE_DATA_DIRECTORY{};
}

// The TLS directory holds virtual addresses, which relocations have already moved to this mapping.
uint64_t rva_in(const Module& m, uint64_t va, const char* what) {
    const uint64_t base = reinterpret_cast<uint64_t>(m.base);
    if (va < base || va - base >= m.size) throw std::runtime_error(m.name + ": " + what + " is outside the image");
    return va - base;
}

IMAGE_TLS_DIRECTORY64* tls_directory(Module& m) {
    auto* nt = at<IMAGE_NT_HEADERS64>(m.base, reinterpret_cast<IMAGE_DOS_HEADER*>(m.base)->e_lfanew);
    // Through directory(), as every other directory: an entry past NumberOfRvaAndSizes is not a TLS directory.
    const auto dir = directory(nt, IMAGE_DIRECTORY_ENTRY_TLS);
    if (!dir.Size || !dir.VirtualAddress) return nullptr;
    require_range(m.name, dir.VirtualAddress, sizeof(IMAGE_TLS_DIRECTORY64), m.size, "the TLS directory");
    return at<IMAGE_TLS_DIRECTORY64>(m.base, dir.VirtualAddress);
}

// __declspec(thread) reads TEB->ThreadLocalStoragePointer[_tls_index] at gs:[58h]. That array is indexed by the
// loader-assigned per-module TLS index and is a different index space from TlsAlloc/TlsGetValue (TlsSlots). Only the OS
// loader can allocate a module index, so the runtime ships a minimal DLL whose only content is a TLS anchor
// (src/tls_host.cpp). Loading it makes the OS reserve an index and give every thread a block for it; the native loader
// reuses that index for the mapped image. When more than one manually mapped module uses static TLS at once, the loader
// clones gta6rt_tls_host.dll to a per-process temporary path on demand and loads it, so the OS assigns an independent
// module TLS index for each simultaneous module (up to kMaxMappedTlsModules).
//
// A mapped executable is different: MSVC code built with /GA (an EXE-only optimisation) skips _tls_index and reads
// TEB->ThreadLocalStoragePointer[0] directly (mov rax, gs:[58h]; mov rax, [rax]), because Windows always gives the
// process executable module TLS index 0. Measured with a `thread_local int v = 42;` EXE built /O2 /GA /MD: natively
// tls_index=0 value=42; mapped with a helper index (6) the code still read slot 0 and printed value=0. Index 0 belongs to
// the runtime's own EXE (gta6rt.exe has a TLS block, 1350 bytes when measured, and no TLS callbacks; each test EXE
// that links this file has one too), and every TLS read in its disassembly goes through _tls_index (the only direct gs:[58h] read is
// module_tls_slots below, which wants the array itself). So index 0 is freed by moving the owner:
// move_index0_owner swaps slots 0 and H of every thread, H being the index of a further helper clone that exists for
// this, and writes H into the owner's _tls_index. The owner then finds its block through H, and slot 0 holds the
// helper's spare block, which the mapped executable's copy replaces (kMainImageSlot) exactly as a DLL's copy replaces
// a helper's. The swap is permanent: both pointers are blocks the OS allocated for this thread, and at thread exit it
// frees every slot by index whatever block each holds, so nothing has to be swapped back.
constexpr DWORD kMaxModuleTlsSlots = 64;
constexpr size_t kMaxMappedTlsModules = 16;
// The slot for a mapped executable, which reads module TLS index 0 (see above). Only one at a time; a second
// executable mapped while one is live gets an ordinary helper slot. kTlsSlotCount sizes the per-slot tables.
constexpr size_t kMainImageSlot = kMaxMappedTlsModules;
constexpr size_t kTlsSlotCount = kMaxMappedTlsModules + 1;
void** module_tls_slots() { return reinterpret_cast<void**>(__readgsqword(0x58)); }

void __cdecl on_host_thread_event(DWORD reason);

struct TlsHostModule {
    HMODULE module = nullptr;
    DWORD index = TLS_OUT_OF_INDEXES;
};

std::filesystem::path tls_host_base_path() {
    std::wstring exe_path;
    DWORD size = MAX_PATH;
    while (true) {
        exe_path.resize(size);
        DWORD len = GetModuleFileNameW(nullptr, exe_path.data(), size);
        if (len == 0) return {};
        if (len < size - 1) {
            exe_path.resize(len);
            break;
        }
        size *= 2;
    }
    return std::filesystem::path(exe_path).parent_path() / L"gta6rt_tls_host.dll";
}

std::filesystem::path tls_host_clone_dir() {
    wchar_t buf[MAX_PATH];
    DWORD n = GetTempPathW(MAX_PATH, buf);
    if (n > 0 && n < MAX_PATH) return std::filesystem::path(buf);
    return std::filesystem::temp_directory_path();
}

struct TlsCloneRecord {
    HMODULE module = nullptr;
    std::filesystem::path path;
};

std::mutex g_tls_alloc_mutex;
std::array<TlsHostModule, kMaxMappedTlsModules> g_tls_hosts{};
std::array<std::atomic<Module*>, kTlsSlotCount> g_tls_slot_owners{};
std::array<std::atomic<uint64_t>, kTlsSlotCount> g_tls_slot_generations{};
std::atomic<uint64_t> g_next_tls_generation{1};
std::vector<TlsCloneRecord> g_tls_clones;

// The move of module TLS index 0 (see kMainImageSlot), done at most once per process. `moved` is written under both
// g_tls_alloc_mutex and an exclusive g_tls_lock, so thread events read it under the shared lock; the other fields are
// set before it and never change after.
struct Index0Move {
    bool moved = false;
    TlsHostModule home;                // the clone whose index now holds the owner's block; never freed
    DWORD mark = TLS_OUT_OF_INDEXES;   // TlsAlloc slot, non-zero on a thread whose slots 0 and home are swapped
};
Index0Move g_index0;

// Frees every TLS helper clone this process loaded and deletes its files. load() registers the wrapper with atexit;
// the --run path in main() must also call this directly before ExitProcess, which skips the atexit table. Call it
// only when no mapped code can run again: it unmaps the helper images.
void cleanup_tls_clones_internal() {
    for (const auto& rec : g_tls_clones) {
        if (rec.module) FreeLibrary(rec.module);
        if (!rec.path.empty()) DeleteFileW(rec.path.c_str());
    }
    g_tls_clones.clear();
}

void cleanup_tls_clones_at_exit() { cleanup_tls_clones_internal(); }

// The owner process named by "leonida_tls_host_<pid>_<slot>.dll"; 0 when the name does not carry one.
DWORD clone_owner_pid(const wchar_t* name) {
    constexpr wchar_t prefix[] = L"leonida_tls_host_";
    const wchar_t* p = name;
    for (size_t i = 0; i < sizeof(prefix) / sizeof(wchar_t) - 1; ++i)
        if (*p++ != prefix[i]) return 0;
    if (*p < L'0' || *p > L'9') return 0;
    unsigned long pid = 0;
    while (*p >= L'0' && *p <= L'9') {
        pid = pid * 10 + (*p - L'0');
        if (pid > 0xFFFFFFFFul) return 0;
        ++p;
    }
    return static_cast<DWORD>(pid);
}

bool process_alive(DWORD pid) {
    // PROCESS_QUERY_LIMITED_INFORMATION needs no handle rights a normal process could deny; a failure means the
    // process is gone. A still-mapped or still-open file is caught by the checks below even for a misjudged PID.
    HANDLE p = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (!p) return false;
    CloseHandle(p);
    return true;
}

// Deletes clones left behind by processes that have exited. Deletion is scoped to files whose owner is gone: a
// clone of a live process is either still open by its copy step's handle or mapped as an image, so it is skipped
// (or DeleteFileW fails harmlessly), and a concurrent process between CopyFileW and LoadLibraryExW keeps its file.
void sweep_stale_tls_clones(const std::filesystem::path& dir) {
    WIN32_FIND_DATAW fd{};
    const std::filesystem::path pattern = dir / L"leonida_tls_host_*.dll";
    HANDLE h = FindFirstFileW(pattern.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        const std::filesystem::path path = dir / fd.cFileName;
        const DWORD owner = clone_owner_pid(fd.cFileName);
        if (owner && owner != GetCurrentProcessId() && process_alive(owner)) continue;
        // An exclusive open (no sharing) conflicts with any handle another process holds on the file, so a failure
        // means the copy->load sequence or a mapped image still owns it. Only unowned files reach DeleteFileW, and
        // that stays harmless for a mapped image.
        HANDLE probe =
            CreateFileW(path.c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (probe == INVALID_HANDLE_VALUE) continue;
        CloseHandle(probe);
        DeleteFileW(path.c_str());
    } while (FindNextFileW(h, &fd));
    FindClose(h);
}

TlsHostModule load_tls_host_image(const std::filesystem::path& path, bool hook_thread_events) {
    TlsHostModule host;
    if (path.empty()) return host;
    // The helper is linked with the static CRT (/MT), so it needs only system DLLs. Restricting the dependency search
    // to System32 keeps a game-supplied vcruntime next to the executable from being picked up.
    host.module = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!host.module) return host;

    auto* base = reinterpret_cast<uint8_t*>(host.module);
    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(base);
    auto* nt = at<IMAGE_NT_HEADERS64>(base, dos->e_lfanew);
    const auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_TLS];
    if (!dir.Size || !dir.VirtualAddress) {
        FreeLibrary(host.module);
        host.module = nullptr;
        return host;
    }
    auto* tls = at<IMAGE_TLS_DIRECTORY64>(base, dir.VirtualAddress);
    if (!tls->AddressOfIndex) {
        FreeLibrary(host.module);
        host.module = nullptr;
        return host;
    }
    host.index = *reinterpret_cast<DWORD*>(tls->AddressOfIndex);
    if (hook_thread_events) {
        // Only the primary helper reports thread attach/detach events; on_host_thread_event iterates all active slots.
        using SetThreadEvent = void(__cdecl*)(void(__cdecl*)(DWORD));
        if (auto set = reinterpret_cast<SetThreadEvent>(GetProcAddress(host.module, "gta6rt_tls_host_set_thread_event")))
            set(&on_host_thread_event);
    }
    return host;
}

// The primary helper: slot 0's index, and the only helper whose DllMain reports thread events (on_host_thread_event
// covers every slot). Must be called with g_tls_alloc_mutex held and WITHOUT g_tls_lock held (LoadLibraryExW takes the
// Windows loader lock).
bool ensure_primary_tls_host() {
    if (g_tls_hosts[0].index != TLS_OUT_OF_INDEXES) return true;
    const std::filesystem::path primary = tls_host_base_path();
    if (primary.empty()) return false;
    g_tls_hosts[0] = load_tls_host_image(primary, true);
    return g_tls_hosts[0].index != TLS_OUT_OF_INDEXES;
}

// Copies gta6rt_tls_host.dll to "leonida_tls_host_<pid>_<tag>.dll" in the clone directory and loads it, so the OS
// gives the copy an index of its own. On failure the file is gone again and the result has no index. The caller
// records the clone (g_tls_clones) if it is to be freed at exit. Same locking rule as ensure_primary_tls_host.
TlsHostModule load_tls_host_clone(const std::wstring& tag, std::filesystem::path& clone_path) {
    const std::filesystem::path primary = tls_host_base_path();
    if (primary.empty()) return {};
    // Clone gta6rt_tls_host.dll to a distinct path using CopyFileW (Wine 9 lacks CopyFile2 used by std::filesystem).
    const std::filesystem::path dir = tls_host_clone_dir();
    static bool swept = false;
    if (!swept) {
        swept = true;
        sweep_stale_tls_clones(dir);
        std::atexit(&cleanup_tls_clones_at_exit);
    }
    const std::wstring name = L"leonida_tls_host_" + std::to_wstring(GetCurrentProcessId()) + L"_" + tag + L".dll";
    clone_path = dir / name;
    // The staging name must not match the sweep pattern, which only covers final clone names ending in .dll.
    const std::filesystem::path staging = clone_path.wstring() + L".new";
    if (!CopyFileW(primary.c_str(), staging.c_str(), FALSE)) return {};

    // Open the guard before the clone exists under its final name and hold it across LoadLibraryExW: the rename
    // below needs DELETE access, which the handle's FILE_SHARE_DELETE grants, and no purge can remove the clone
    // while the handle is open (an exclusive open conflicts with it) or once the image is mapped. Copying straight
    // to the final name would leave a window where a concurrent sweep sees an unowned file.
    HANDLE guard = CreateFileW(staging.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (guard == INVALID_HANDLE_VALUE) {
        DeleteFileW(staging.c_str());
        return {};
    }
    if (!MoveFileExW(staging.c_str(), clone_path.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        CloseHandle(guard);
        DeleteFileW(staging.c_str());
        return {};
    }
    TlsHostModule loaded = load_tls_host_image(clone_path, false);
    CloseHandle(guard);

    if (loaded.index == TLS_OUT_OF_INDEXES) DeleteFileW(clone_path.c_str());
    return loaded;
}

// Must be called with g_tls_alloc_mutex held and WITHOUT g_tls_lock held (LoadLibraryExW takes the Windows loader lock).
TlsHostModule ensure_tls_host(size_t slot) {
    if (slot >= kMaxMappedTlsModules) return {};
    if (g_tls_hosts[slot].index != TLS_OUT_OF_INDEXES) return g_tls_hosts[slot];
    // Slot 0 is the primary helper; it is loaded first in any case so thread attach/detach events are hooked.
    if (!ensure_primary_tls_host()) return {};
    if (slot == 0) return g_tls_hosts[0];

    std::filesystem::path clone_path;
    TlsHostModule loaded = load_tls_host_clone(std::to_wstring(slot), clone_path);
    if (loaded.index == TLS_OUT_OF_INDEXES) return {};
    // Index 0 is the runtime EXE's, and after the move the home clone's index holds that EXE's block.
    bool taken = loaded.index == 0 || loaded.index == g_index0.home.index;
    for (size_t i = 0; i < kMaxMappedTlsModules; ++i) taken |= i != slot && g_tls_hosts[i].index == loaded.index;
    if (taken) {
        FreeLibrary(loaded.module);
        DeleteFileW(clone_path.c_str());
        return {};
    }
    g_tls_clones.push_back({loaded.module, clone_path});
    g_tls_hosts[slot] = loaded;
    return g_tls_hosts[slot];
}

struct ThreadTlsSlot {
    Module* module = nullptr;
    uint64_t generation = 0;
    void* saved = nullptr;
    void* block = nullptr;
    uint32_t index = kNoTlsIndex;
};

// Per-thread installation state for each reserved TLS slot.
thread_local std::array<ThreadTlsSlot, kTlsSlotCount> g_thread_tls_slots{};

// Orders the TLS owners' publication, their `initialised` flags and their retirement against the thread events. Shared
// in on_host_thread_event, exclusive only for those short writes (no allocation and no call that takes the Windows
// loader lock, which the events run under) and for the one-time index-0 move, which lists and stops threads under it
// with the loader lock already held.
SRWLOCK g_tls_lock = SRWLOCK_INIT;
struct SharedTlsLock {
    SharedTlsLock() { AcquireSRWLockShared(&g_tls_lock); }
    ~SharedTlsLock() { ReleaseSRWLockShared(&g_tls_lock); }
    SharedTlsLock(const SharedTlsLock&) = delete;
    SharedTlsLock& operator=(const SharedTlsLock&) = delete;
};
struct ExclusiveTlsLock {
    ExclusiveTlsLock() { AcquireSRWLockExclusive(&g_tls_lock); }
    ~ExclusiveTlsLock() { ReleaseSRWLockExclusive(&g_tls_lock); }
    ExclusiveTlsLock(const ExclusiveTlsLock&) = delete;
    ExclusiveTlsLock& operator=(const ExclusiveTlsLock&) = delete;
};

// x64 TEB fields the move reads and writes on other threads (the same offsets on Wine): ThreadLocalStoragePointer,
// and TlsSlots[64], the array TlsGetValue reads for an index below 64.
constexpr size_t kTebTlsPointer = 0x58;
constexpr size_t kTebTlsSlots = 0x1480;

// The image that owns module TLS index 0: the one whose TLS directory's AddressOfIndex holds 0. Only images the
// Windows loader mapped are listed, so their headers are read unchecked.
struct Index0Owner {
    const uint8_t* base = nullptr;
    size_t size = 0;
    DWORD* index = nullptr;  // the owner's _tls_index
    std::string name;        // UTF-8 file name, for the trace and messages
};

std::string module_file_name(HMODULE module) {
    wchar_t path[MAX_PATH];
    const DWORD n = GetModuleFileNameW(module, path, MAX_PATH);
    if (!n || n >= MAX_PATH) return "?";
    const wchar_t* name = path;
    for (const wchar_t* p = path; *p; ++p)
        if (*p == L'\\' || *p == L'/') name = p + 1;
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, name, -1, nullptr, 0, nullptr, nullptr);
    if (bytes <= 1) return "?";
    std::string out(static_cast<size_t>(bytes - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, name, -1, out.data(), bytes, nullptr, nullptr);
    return out;
}

// Empty on success, else why index 0 has no single movable owner.
std::string find_index0_owner(Index0Owner& owner) {
    std::vector<HMODULE> modules(256);
    DWORD needed = 0;
    for (;;) {
        const DWORD bytes = static_cast<DWORD>(modules.size() * sizeof(HMODULE));
        if (!EnumProcessModules(GetCurrentProcess(), modules.data(), bytes, &needed))
            return "the process's modules cannot be listed";
        if (needed <= bytes) break;
        modules.resize(needed / sizeof(HMODULE) + 16);  // a DLL loaded in between needs room too
    }
    modules.resize(needed / sizeof(HMODULE));
    int found = 0;
    HMODULE owner_module = nullptr;
    for (HMODULE module : modules) {
        auto* base = reinterpret_cast<uint8_t*>(module);
        auto* nt = at<IMAGE_NT_HEADERS64>(base, reinterpret_cast<IMAGE_DOS_HEADER*>(base)->e_lfanew);
        const auto dir = directory(nt, IMAGE_DIRECTORY_ENTRY_TLS);
        if (!dir.Size || !dir.VirtualAddress) continue;
        auto* tls = at<IMAGE_TLS_DIRECTORY64>(base, dir.VirtualAddress);
        if (!tls->AddressOfIndex || *reinterpret_cast<DWORD*>(tls->AddressOfIndex) != 0) continue;
        ++found;
        owner_module = module;
        owner.base = base;
        owner.size = nt->OptionalHeader.SizeOfImage;
        owner.index = reinterpret_cast<DWORD*>(tls->AddressOfIndex);
    }
    if (found == 0) return "no module holds it";
    if (found > 1) return "more than one module reports it";
    owner.name = module_file_name(owner_module);
    // The helpers are loaded after the EXE, so one of them holds index 0 only in a process whose EXE has no TLS; DLL
    // slot 0 then already maps DLLs there.
    for (const auto& host : g_tls_hosts)
        if (host.module == owner_module) return "it belongs to gta6rt_tls_host.dll";
    return {};
}

using NtQueryInformationThreadFn = LONG(NTAPI*)(HANDLE, int, PVOID, ULONG, PULONG);
using LdrLockLoaderLockFn = LONG(NTAPI*)(ULONG, ULONG*, ULONG_PTR*);
using LdrUnlockLoaderLockFn = LONG(NTAPI*)(ULONG, ULONG_PTR);

struct ThreadBasicInformation {  // THREAD_BASIC_INFORMATION (NtQueryInformationThread class 0)
    LONG exit_status;
    PVOID teb;
    PVOID process_id;
    PVOID thread_id;
    ULONG_PTR affinity;
    LONG priority;
    LONG base_priority;
};

struct OtherThread {
    HANDLE handle = nullptr;
    uint8_t* teb = nullptr;
    bool suspended = false;
};

enum class SwapResult { Done, Busy, Failed };

// One attempt at the swap, with every other thread of the process stopped. The exclusive g_tls_lock keeps every
// thread event out, and with it the adoption on threads that start meanwhile. Busy: a thread was stopped inside the
// owner's code, perhaps between its load of _tls_index and its read of the slot, so the caller resumes everything and
// tries again. Measured in ntdll (Windows 11 26300): a starting thread fills its whole TLS array before it stores it in
// its TEB (LdrpAllocateTls), an ending one clears that TEB field before it frees the blocks (LdrpFreeTls), and
// LoadLibrary of a DLL with TLS has the kernel update every thread's array in one system call (NtSetInformationProcess,
// class 0x23), which GetThreadContext below waits out on the calling thread. So an array seen here is complete and
// still owned; a thread with none yet gets one with the owner's block at 0 and adopts the swap in DLL_THREAD_ATTACH.
// Wine builds a starting thread's array and grows every thread's array for LoadLibrary in user mode, both under its
// loader lock, so the Windows loader lock is held as well; it comes before g_tls_lock, the order the thread events
// (run under it) take them in.
SwapResult swap_index0_on_all_threads(const Index0Owner& owner, DWORD home, DWORD mark, std::string& error) {
    static const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    static const auto query =
        reinterpret_cast<NtQueryInformationThreadFn>(GetProcAddress(ntdll, "NtQueryInformationThread"));
    static const auto lock_loader = reinterpret_cast<LdrLockLoaderLockFn>(GetProcAddress(ntdll, "LdrLockLoaderLock"));
    static const auto unlock_loader =
        reinterpret_cast<LdrUnlockLoaderLockFn>(GetProcAddress(ntdll, "LdrUnlockLoaderLock"));
    if (!query || !lock_loader || !unlock_loader) {
        error = "ntdll lacks NtQueryInformationThread or the loader lock calls";
        return SwapResult::Failed;
    }
    ULONG disposition = 0;
    ULONG_PTR cookie = 0;
    if (lock_loader(0, &disposition, &cookie) < 0) {
        error = "the Windows loader lock cannot be taken";
        return SwapResult::Failed;
    }
    struct LoaderUnlock {
        ULONG_PTR cookie;
        ~LoaderUnlock() { unlock_loader(0, cookie); }
    } loader_unlock{cookie};
    ExclusiveTlsLock lock;

    // Listed under the lock: a thread that starts later runs its DLL_THREAD_ATTACH after the swap and adopts it there
    // (adopt_index0_move_on_this_thread).
    std::vector<OtherThread> threads;
    {
        HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if (snapshot == INVALID_HANDLE_VALUE) {
            error = "the process's threads cannot be listed";
            return SwapResult::Failed;
        }
        THREADENTRY32 entry{};
        entry.dwSize = sizeof(entry);
        const DWORD pid = GetCurrentProcessId(), self = GetCurrentThreadId();
        for (BOOL more = Thread32First(snapshot, &entry); more; more = Thread32Next(snapshot, &entry)) {
            if (entry.th32OwnerProcessID != pid || entry.th32ThreadID == self) continue;
            HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION | SYNCHRONIZE,
                                  FALSE, entry.th32ThreadID);
            if (!h) continue;  // ended since the snapshot
            ThreadBasicInformation info{};
            if (query(h, 0, &info, sizeof(info), nullptr) < 0 || !info.teb) {
                CloseHandle(h);
                continue;
            }
            threads.push_back({h, static_cast<uint8_t*>(info.teb)});
        }
        CloseHandle(snapshot);
    }

    // From the first SuspendThread to the last ResumeThread nothing below allocates or takes a lock: a stopped thread
    // can hold the heap lock.
    bool busy = false;
    for (auto& t : threads) {
        if (SuspendThread(t.handle) == static_cast<DWORD>(-1)) {
            if (WaitForSingleObject(t.handle, 0) == WAIT_OBJECT_0) continue;  // it ended; its TEB is gone
            busy = true;  // a live thread left unswapped would read the helper's block for the owner's
            break;
        }
        t.suspended = true;
        // SuspendThread only asks; GetThreadContext returns once the thread has stopped.
        CONTEXT context{};
        context.ContextFlags = CONTEXT_CONTROL;
        if (!GetThreadContext(t.handle, &context)) {
            if (WaitForSingleObject(t.handle, 0) == WAIT_OBJECT_0) continue;  // it ended; its TEB is gone
            busy = true;
            break;
        }
        if (context.Rip - reinterpret_cast<uint64_t>(owner.base) < owner.size) {
            busy = true;
            break;
        }
    }
    if (!busy) {
        for (auto& t : threads) {
            if (!t.suspended || WaitForSingleObject(t.handle, 0) == WAIT_OBJECT_0) continue;
            // No array yet: the thread is still starting, and its DLL_THREAD_ATTACH adopts the swap.
            auto** slots = *reinterpret_cast<void***>(t.teb + kTebTlsPointer);
            if (!slots) continue;
            std::swap(slots[0], slots[home]);
            reinterpret_cast<void**>(t.teb + kTebTlsSlots)[mark] = reinterpret_cast<void*>(1);
        }
        // The owner's code reads the new index from here on, this thread's included.
        void** own = module_tls_slots();
        std::swap(own[0], own[home]);
        TlsSetValue(mark, reinterpret_cast<void*>(1));
        *owner.index = home;
        g_index0.moved = true;
    }
    for (auto& t : threads) {
        if (t.suspended) ResumeThread(t.handle);
        CloseHandle(t.handle);
    }
    return busy ? SwapResult::Busy : SwapResult::Done;
}

// Frees module TLS index 0 for a mapped executable (see kMainImageSlot): loads the home clone, then swaps slots 0
// and home on every thread and writes home into the owner's _tls_index. Empty on success (also when it was done
// before), else why not; nothing has changed then. Must be called with g_tls_alloc_mutex held and WITHOUT g_tls_lock
// held. Never inlined, so the caller reads its own thread_local data through the new index once this returns.
__declspec(noinline) std::string move_index0_owner() {
    if (g_index0.moved) return {};
    if (!ensure_primary_tls_host()) return "gta6rt_tls_host.dll is missing or has no module TLS index";
    Index0Owner owner;
    if (std::string why = find_index0_owner(owner); !why.empty()) return "module TLS index 0: " + why;
    if (g_index0.home.index == TLS_OUT_OF_INDEXES) {
        // Kept loaded for the process lifetime: FreeLibrary would make the OS free every thread's slot at its index,
        // which holds the owner's block from now on. Not in g_tls_clones for that reason; the sweep of a later run
        // deletes the file once this process has ended.
        std::filesystem::path path;
        TlsHostModule home = load_tls_host_clone(L"home", path);
        if (home.index == TLS_OUT_OF_INDEXES) return "the TLS helper clone for index 0 cannot be loaded";
        bool taken = home.index == 0 || home.index >= kMaxModuleTlsSlots;
        for (const auto& host : g_tls_hosts) taken |= host.index == home.index;
        if (taken) {
            FreeLibrary(home.module);
            DeleteFileW(path.c_str());
            return "the TLS helper clone for index 0 got an unusable index";
        }
        g_index0.home = home;
    }
    if (g_index0.mark == TLS_OUT_OF_INDEXES) {
        const DWORD mark = TlsAlloc();
        if (mark == TLS_OUT_OF_INDEXES) return "no TlsAlloc slot is free";
        if (mark >= 64) {  // another thread's TlsSlots can only be written for the first 64
            TlsFree(mark);
            return "no TlsAlloc slot below 64 is free";
        }
        g_index0.mark = mark;
    }
    // _tls_index sits in .data; made writable first in case a linker merged it into a read-only section.
    DWORD old_protect = 0;
    if (!VirtualProtect(owner.index, sizeof(DWORD), PAGE_READWRITE, &old_protect))
        return "the owner's _tls_index cannot be written";
    // A thread caught in the owner's code usually leaves it within microseconds; one that keeps computing there for
    // the whole budget makes the move give up (the executable then maps with a helper index), and a later executable
    // tries again.
    std::string error;
    SwapResult result = SwapResult::Busy;
    const ULONGLONG give_up = GetTickCount64() + 250;
    for (bool first = true; result == SwapResult::Busy && (first || GetTickCount64() < give_up); first = false) {
        if (!first) Sleep(1);
        result = swap_index0_on_all_threads(owner, g_index0.home.index, g_index0.mark, error);
    }
    VirtualProtect(owner.index, sizeof(DWORD), old_protect, &old_protect);
    if (result == SwapResult::Busy) return "a thread kept running inside " + owner.name;
    if (result == SwapResult::Failed) return error;
    if (startup_trace::enabled())
        startup_trace::event("tls_index0", "\"owner\":" + startup_trace::quote(owner.name) +
                                               ",\"moved_to\":" + std::to_string(g_index0.home.index));
    return {};
}

// Puts a thread started after the move in the same state as the threads the move swapped. Called first in this
// thread's DLL_THREAD_ATTACH, under the shared lock, before anything reads the runtime's own thread_local data: the
// OS built this thread's array with the owner's block at 0 and the home clone's at home. DLLs loaded before the
// helper get their DLL_THREAD_ATTACH earlier, and the EXE's TLS callbacks run later (measured on Windows 11 26300:
// DllMain of every DLL, then the EXE's callbacks, for attach and for detach), so only runtime code a vectored
// exception handler runs in that window would read the home clone's block in place of its own.
void adopt_index0_move_on_this_thread() {
    if (!g_index0.moved || TlsGetValue(g_index0.mark)) return;
    void** slots = module_tls_slots();
    if (!slots) return;
    std::swap(slots[0], slots[g_index0.home.index]);
    TlsSetValue(g_index0.mark, reinterpret_cast<void*>(1));
}

// Puts the helper DLL's own block back in this thread's slot and frees the copy that replaced it. Uses only this
// thread's record, so it also works after the module that installed the copy is gone.
void release_thread_slot(size_t slot) {
    if (slot >= kTlsSlotCount) return;
    ThreadTlsSlot& entry = g_thread_tls_slots[slot];
    if (!entry.module && !entry.block) return;
    if (entry.block) {
        void** slots = module_tls_slots();
        if (slots && entry.index < kMaxModuleTlsSlots && slots[entry.index] == entry.block)
            slots[entry.index] = entry.saved;
        VirtualFree(entry.block, 0, MEM_RELEASE);
    }
    entry = ThreadTlsSlot{};
}

void install_thread_block(Module& m) {
    if (m.tls_slot >= kTlsSlotCount) return;
    const size_t slot = m.tls_slot;
    ThreadTlsSlot& entry = g_thread_tls_slots[slot];
    if (entry.module == &m && entry.generation == m.tls_generation) return;
    if (entry.module) {
        // A copy left by a module whose Loader is gone (or from an earlier generation) is released; a live owner's is a conflict.
        if (entry.module == g_tls_slot_owners[slot].load() &&
            entry.generation == g_tls_slot_generations[slot].load()) {
            throw std::runtime_error("another manually mapped module's TLS is already installed in this thread slot");
        }
        release_thread_slot(slot);
    }

    const uint64_t total = static_cast<uint64_t>(m.tls_raw_size) + m.tls_zero_fill;
    if (!total) {
        // Callback-only TLS module (no data block): still record attachment in this slot so DLL_THREAD_DETACH runs.
        entry.module = &m;
        entry.generation = m.tls_generation;
        return;
    }
    if (m.tls_index == kNoTlsIndex) return;
    void** slots = module_tls_slots();
    if (!slots || m.tls_index >= kMaxModuleTlsSlots)
        throw std::runtime_error(m.name + ": cannot install TLS in this thread");

    void* block = VirtualAlloc(nullptr, static_cast<SIZE_T>(total), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!block) throw std::runtime_error(m.name + ": TLS block allocation failed");
    if (m.tls_raw_size) std::memcpy(block, m.tls_template, m.tls_raw_size);
    if (m.tls_zero_fill)
        std::memset(static_cast<uint8_t*>(block) + m.tls_raw_size, 0, m.tls_zero_fill);

    void* saved = slots[m.tls_index];
    slots[m.tls_index] = block;
    entry.module = &m;
    entry.generation = m.tls_generation;
    entry.saved = saved;
    entry.block = block;
    entry.index = m.tls_index;
}

void uninstall_thread_block(Module& m) {
    if (m.tls_slot >= kTlsSlotCount) return;
    const size_t slot = m.tls_slot;
    ThreadTlsSlot& entry = g_thread_tls_slots[slot];
    if (entry.module != &m || entry.generation != m.tls_generation) return;
    if (m.tls_block == entry.block) {
        m.tls_block = nullptr;
        m.tls_saved_block = nullptr;
    }
    release_thread_slot(slot);
}

// `trace`: record each callback in the startup trace. Only the Loader's own calls (process attach and detach) pass
// true; the thread events run under the Windows loader lock and stay out of the trace.
void run_callbacks(Module& m, uint32_t reason, bool trace = false) {
    auto* tls = tls_directory(m);
    if (!tls || !tls->AddressOfCallBacks) return;
    const uint64_t table_rva = rva_in(m, tls->AddressOfCallBacks, "the TLS callback table");
    // The callback table has no length field: it ends at a zero entry, so every step is checked against the image.
    for (uint64_t i = 0;; ++i) {
        const uint64_t entry_rva = table_rva + i * sizeof(uint64_t);
        require_range(m.name, entry_rva, sizeof(uint64_t), m.size, "a TLS callback table entry");
        const uint64_t fn = *at<uint64_t>(m.base, static_cast<uint32_t>(entry_rva));
        if (!fn) break;
        auto callback =
            reinterpret_cast<PIMAGE_TLS_CALLBACK>(m.base + rva_in(m, fn, "a TLS callback"));
        if (trace && startup_trace::enabled())
            startup_trace::event("tls_callback", "\"module\":" + startup_trace::quote(m.name) + ",\"at\":" +
                                                     startup_trace::quote(startup_trace::where(reinterpret_cast<uint64_t>(callback))) +
                                                     ",\"reason\":" + std::to_string(reason));
        callback(m.base, static_cast<DWORD>(reason), nullptr);
    }
}

// The body of on_host_thread_event, called with the shared lock held. A new thread gets its own copy of each active
// TLS owner's template; an ending one gives each helper's block back before the Windows loader frees it. TLS callbacks
// get the thread events only after the module's DLL_PROCESS_ATTACH (initialise()), as the Windows loader orders them.
// The mapped executable comes last both ways, as Windows runs an EXE's TLS callbacks after every DllMain for attach
// and for detach (measured on Windows 11 26300); kMainImageSlot is the highest slot, so attach runs in slot order and
// detach runs the DLL slots in reverse and that slot after them.
__declspec(noinline) void deliver_thread_event(DWORD reason) {
    // The try/catch is per slot: a throw from one module's install or callbacks must not take the thread events
    // away from the other slots. Nothing may leave DllMain.
    if (reason == DLL_THREAD_ATTACH) {
        for (size_t slot = 0; slot < kTlsSlotCount; ++slot) {
            Module* m = g_tls_slot_owners[slot].load();
            if (!m) continue;
            try {
                install_thread_block(*m);
                if (m->initialised) run_callbacks(*m, DLL_THREAD_ATTACH);
            } catch (const std::exception& e) {
                // The thread runs on without this module's TLS; later slots still get theirs.
                std::fprintf(stderr, "[rt] loader: thread attach, slot %zu (%s): %s\n", slot, m->name.c_str(),
                             e.what());
            }
        }
    } else if (reason == DLL_THREAD_DETACH) {
        for (size_t step = 0; step < kTlsSlotCount; ++step) {
            const size_t i = step < kMaxMappedTlsModules ? kMaxMappedTlsModules - 1 - step : kMainImageSlot;
            Module* m = g_tls_slot_owners[i].load();
            if (m && g_thread_tls_slots[i].module == m &&
                g_thread_tls_slots[i].generation == m->tls_generation) {
                try {
                    if (m->initialised) run_callbacks(*m, DLL_THREAD_DETACH);
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "[rt] loader: thread detach, slot %zu (%s): %s\n", i, m->name.c_str(),
                                 e.what());
                }
                // Released even when the callbacks threw: a leaked block would be handed to the OS loader as the
                // helper's own TLS block and freed with it at thread exit.
                uninstall_thread_block(*m);
            } else {
                release_thread_slot(i);  // installed by a module whose Loader is gone (or nothing to do)
            }
        }
    }
}

// Runs in gta6rt_tls_host.dll's DllMain for each thread the process starts or ends once the helper is loaded, under
// the Windows loader lock, on that thread. It reads no thread_local data itself: on a new thread the index-0 move has
// to be adopted before the runtime's own TLS is where its _tls_index says (deliver_thread_event is not inlined).
void __cdecl on_host_thread_event(DWORD reason) {
    SharedTlsLock lock;
    if (reason == DLL_THREAD_ATTACH) adopt_index0_move_on_this_thread();
    deliver_thread_event(reason);
}

// The Xbox builds of the MSVC runtime import the same CRT the desktop does, so the host's own copies are used.
bool prefer_host_copy(const std::string& dll) {
    static const char* names[] = {"vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll",
                                  "msvcp140_2.dll", "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll",
                                  "concrt140.dll", "vccorlib140.dll", "vcomp140.dll"};
    for (auto* n : names)
        if (dll == n) return true;
    return false;
}

// MSVC C++ exceptions in a mapped image. _CxxThrowException (vcruntime140.dll) puts the throwing image's base in the
// exception record, from RtlPcToFileHeader(pThrowInfo); the frame handler adds the ThrowInfo's RVAs (catchable types,
// copy constructor, destructor) to it. RtlPcToFileHeader only knows images the Windows loader mapped, so for ours the
// base is 0 and the exception can't be matched to a catch. The host copies the loader binds get their import of it
// redirected: the Windows answer first, then the modules a live Loader has mapped (shims::mapped_image_base).
using PcToFileHeaderFn = PVOID(NTAPI*)(PVOID, PVOID*);
std::atomic<PcToFileHeaderFn> g_host_pc_to_file_header{nullptr};

PVOID NTAPI pc_to_file_header(PVOID pc, PVOID* base) {
    if (PVOID found = g_host_pc_to_file_header.load()(pc, base)) return found;
    uint64_t mapped = 0;
    if (!shims::mapped_image_base(reinterpret_cast<uint64_t>(pc), mapped)) {
        *base = nullptr;
        return nullptr;
    }
    *base = reinterpret_cast<PVOID>(mapped);
    return *base;
}

// Points every import of RtlPcToFileHeader by name in a host module at pc_to_file_header, once per module. The
// module is a system DLL the Windows loader has already validated and bound, so its tables are read unchecked.
// The redirect stays for the process lifetime; with no live Loader it answers exactly as Windows does.
void redirect_pc_to_file_header(HMODULE host) {
    static std::mutex mutex;
    static std::vector<HMODULE> done;
    std::lock_guard lock(mutex);
    if (std::find(done.begin(), done.end(), host) != done.end()) return;
    done.push_back(host);

    auto* base = reinterpret_cast<uint8_t*>(host);
    auto* nt = at<IMAGE_NT_HEADERS64>(base, reinterpret_cast<IMAGE_DOS_HEADER*>(base)->e_lfanew);
    const auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!dir.Size || !dir.VirtualAddress) return;
    for (auto* d = at<IMAGE_IMPORT_DESCRIPTOR>(base, dir.VirtualAddress); d->Name; ++d) {
        if (!d->OriginalFirstThunk) continue;  // no name table to match against
        auto* names = at<IMAGE_THUNK_DATA64>(base, d->OriginalFirstThunk);
        auto* iat = at<IMAGE_THUNK_DATA64>(base, d->FirstThunk);
        for (; names->u1.AddressOfData; ++names, ++iat) {
            if (IMAGE_SNAP_BY_ORDINAL64(names->u1.Ordinal)) continue;
            auto* by_name = at<IMAGE_IMPORT_BY_NAME>(base, static_cast<uint32_t>(names->u1.AddressOfData));
            if (std::strcmp(reinterpret_cast<const char*>(by_name->Name), "RtlPcToFileHeader") != 0) continue;
            auto* slot = reinterpret_cast<void**>(&iat->u1.Function);
            if (*slot == reinterpret_cast<void*>(&pc_to_file_header)) continue;
            PcToFileHeaderFn expected = nullptr;  // the first bound address is the Windows function
            g_host_pc_to_file_header.compare_exchange_strong(expected, reinterpret_cast<PcToFileHeaderFn>(*slot));
            DWORD old = 0;
            if (!VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &old)) continue;
            *slot = reinterpret_cast<void*>(&pc_to_file_header);
            VirtualProtect(slot, sizeof(void*), old, &old);
        }
    }
}

// The module a host import actually resolved into. GetProcAddress follows forwarders, so this can be another DLL
// than the one imported (on Windows vcruntime140.dll implements _CxxThrowException itself); that module's
// RtlPcToFileHeader import is the one the throw calls.
void redirect_pc_to_file_header_at(const void* resolved) {
    HMODULE owner = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCWSTR>(resolved), &owner) && owner)
        redirect_pc_to_file_header(owner);
}

}  // namespace

// Declared in pe_loader.h; the body needs the internal clone registry above.
void cleanup_tls_clones() { cleanup_tls_clones_internal(); }

namespace {
bool lookup_in_loader(void* ctx, uint64_t address, std::string& module, uint32_t& rva) {
    for (const Module* m : static_cast<Loader*>(ctx)->modules()) {
        const auto base = reinterpret_cast<uint64_t>(m->base);
        if (m->base && address >= base && address - base < m->size) {
            module = m->name;
            rva = static_cast<uint32_t>(address - base);
            return true;
        }
    }
    return false;
}

void* bundled_find_loaded(void* ctx, const std::string& dll_lower) {
    return static_cast<Loader*>(ctx)->find_loaded_module(dll_lower);
}

void* bundled_load(void* ctx, const std::string& dll_lower, uint32_t flags, bool& handled) {
    return static_cast<Loader*>(ctx)->load_runtime_module(dll_lower, flags, handled);
}

void* bundled_get_proc(void* ctx, void* module, const std::string& sym, bool& owns_handle) {
    return static_cast<Loader*>(ctx)->resolve_module_export(module, sym, owns_handle);
}

bool bundled_owns_handle(void* ctx, void* module) {
    return static_cast<Loader*>(ctx)->owns_module_handle(module);
}

const shims::BundledModuleOps kBundledOps{
    &bundled_find_loaded,
    &bundled_load,
    &bundled_get_proc,
    &bundled_owns_handle,
};
}  // namespace

Loader::Loader(std::filesystem::path game_dir) : game_dir_(std::move(game_dir)) {
    shims::add_module_lookup(this, &lookup_in_loader);
    shims::add_bundled_loader(this, &kBundledOps);
}

Loader::~Loader() {
    // Importers must release their resources before their dependencies. Keep every mapping, unwind table and TLS
    // block alive throughout the notifications: a detach handler can still call an imported function or free an
    // FLS slot whose destructor lives in another mapped image.
    for (auto it = init_order_.rbegin(); it != init_order_.rend(); ++it) {
        Module& m = **it;
        if (!m.initialised) continue;  // a map-only load never opts in to running image code
        set_initialised(m, false);  // under the TLS lock: thread events stop running its callbacks
        try {
            run_tls_callbacks(m, DLL_PROCESS_DETACH);
        } catch (const std::exception& e) {
            // A module may have changed its callback table since attach. A destructor cannot propagate the validation
            // error; report it and still let DllMain release resources before the image is freed.
            std::fprintf(stderr, "[rt] TLS process detach failed for %s: %s\n", m.name.c_str(), e.what());
        }
        auto* nt = at<IMAGE_NT_HEADERS64>(m.base, reinterpret_cast<IMAGE_DOS_HEADER*>(m.base)->e_lfanew);
        if ((nt->FileHeader.Characteristics & IMAGE_FILE_DLL) && m.entry_rva) {
            auto entry = reinterpret_cast<BOOL(NTAPI*)(void*, DWORD, void*)>(m.base + m.entry_rva);
            entry(m.base, DLL_PROCESS_DETACH, nullptr);  // explicit unload; the return value is ignored
        }
    }
    shims::remove_bundled_loader(this);
    shims::remove_module_lookup(this);
    for (auto* m : order_) {
        // TLS state must go away before the image and its TLS block are unmapped.
        if (m->tls_index != kNoTlsIndex && m->tls_slot < kTlsSlotCount) {
            // Retired first, under the lock: no thread event installs it or runs its callbacks after this, and a
            // thread that still holds a copy frees it when it ends (on_host_thread_event).
            {
                ExclusiveTlsLock lock;
                Module* owner = m;
                if (g_tls_slot_owners[m->tls_slot].compare_exchange_strong(owner, nullptr))
                    g_tls_slot_generations[m->tls_slot].store(0);
            }
            uninstall_tls_for_thread(*m);
        }
        // Before the image is freed: a stale table would let a later stack walk read unmapped memory.
        if (m->function_table) RtlDeleteFunctionTable(static_cast<PRUNTIME_FUNCTION>(m->function_table));
        if (m->tls_block) VirtualFree(m->tls_block, 0, MEM_RELEASE);
        if (m->base) VirtualFree(m->base, 0, MEM_RELEASE);
        delete m;
    }
}

Module& Loader::load(const std::filesystem::path& image) {
    std::lock_guard lock(mutex_);
    std::string key = lower(image.filename().string());
    if (auto it = by_name_.find(key); it != by_name_.end()) return *it->second;
    Module* mapped = nullptr;
    try {
        mapped = &map_image(image);
        bind_imports(*mapped);  // loads every bundled module m imports first, so they land in init_order_ before m
        protect_sections(*mapped);
        register_unwind(*mapped);
    } catch (...) {
        // A failed load, nested ones (a dependency of a run-time load) included, must not stay findable by name:
        // a later lookup would get a half-mapped image with unbound imports. ~Loader still frees it through order_.
        by_name_.erase(key);
        throw;
    }
    Module& m = *mapped;
    init_order_.push_back(&m);
    if (startup_trace::enabled())
        startup_trace::event("map", "\"module\":" + startup_trace::quote(m.name) +
                                        ",\"base\":" + startup_trace::quote(startup_trace::hex(reinterpret_cast<uint64_t>(m.base))) +
                                        ",\"size\":" + startup_trace::quote(startup_trace::hex(m.size)) +
                                        ",\"relocated\":" + (m.relocated ? "true" : "false"));
    return m;
}

void* Loader::find_loaded_module(const std::string& dll_lower) const {
    std::lock_guard lock(mutex_);
    auto it = by_name_.find(dll_lower);
    return it == by_name_.end() ? nullptr : static_cast<void*>(it->second->base);
}

void* Loader::load_runtime_module(const std::string& dll_lower, uint32_t flags, bool& handled) {
    handled = false;
    if (dll_lower.empty() || prefer_host_copy(dll_lower)) return nullptr;

    std::lock_guard lock(mutex_);
    const auto existing = by_name_.find(dll_lower);
    const std::filesystem::path candidate = game_dir_ / dll_lower;
    std::error_code ec;
    const bool is_bundled = existing != by_name_.end() || std::filesystem::is_regular_file(candidate, ec);
    if (!is_bundled) return nullptr;

    handled = true;
    // Win32 DONT_RESOLVE_DLL_REFERENCES (0x1) forbids resolving dependencies or running TLS/DllMain; refuse it
    // explicitly instead of silently performing a normal resolving + initialising load.
    if ((flags & DONT_RESOLVE_DLL_REFERENCES) != 0) {
        std::fprintf(stderr,
                     "[rt] loader: LoadLibraryEx(%s, DONT_RESOLVE_DLL_REFERENCES) is not supported for bundled DLLs\n",
                     dll_lower.c_str());
        SetLastError(ERROR_NOT_SUPPORTED);
        return nullptr;
    }

    if (existing != by_name_.end()) {
        try {
            initialise();
            return static_cast<void*>(existing->second->base);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[rt] loader: run-time initialise of %s failed: %s\n", dll_lower.c_str(), e.what());
            SetLastError(ERROR_DLL_INIT_FAILED);
            return nullptr;
        }
    }

    try {
        Module& m = load(candidate);
        initialise();
        std::fprintf(stderr, "[rt] loader: %s mapped from game_dir (run-time load)\n", m.name.c_str());
        return static_cast<void*>(m.base);
    } catch (const std::exception& e) {
        // load() drops a module that failed to map or bind (dependencies included); this also drops one whose
        // initialise() failed, so later GetModuleHandle/LoadLibrary calls don't see it. ~Loader frees it via order_.
        by_name_.erase(dll_lower);
        std::fprintf(stderr, "[rt] loader: run-time load of %s failed: %s\n", dll_lower.c_str(), e.what());
        SetLastError(ERROR_DLL_INIT_FAILED);
        return nullptr;
    }
}

void* Loader::resolve_module_export(void* module_handle, const std::string& sym, bool& owns_handle) {
    owns_handle = false;
    if (!module_handle) return nullptr;
    std::lock_guard lock(mutex_);
    Module* target = nullptr;
    for (Module* m : order_) {
        if (m->base == module_handle) {
            target = m;
            break;
        }
    }
    if (!target) return nullptr;
    owns_handle = true;
    if (auto it = target->exports.find(sym); it != target->exports.end()) return it->second;
    if (auto it = target->forwarders.find(sym); it != target->forwarders.end()) {
        try {
            Binding binding = Binding::Unimplemented;
            void* fn = resolve_forwarder(it->second, binding, 0);
            if (binding == Binding::Module) initialise();
            return fn;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[rt] loader: forwarder %s from %s failed: %s\n", it->second.c_str(),
                         target->name.c_str(), e.what());
            return nullptr;
        }
    }
    return nullptr;
}

bool Loader::owns_module_handle(void* module_handle) const {
    if (!module_handle) return false;
    std::lock_guard lock(mutex_);
    for (const Module* m : order_)
        if (m->base == module_handle) return true;
    return false;
}

// x64 has no frame-pointer chain to fall back on: without its .pdata registered, an exception raised in (or passing
// through) a mapped image can't be dispatched past that frame, and the process dies. Images the system loader maps
// are found through its own module list; ours are VirtualAlloc'd, so they are registered as dynamic function tables.
//
// The table is checked before it is handed over, because RtlLookupFunctionEntry trusts it: it binary-searches the
// entries by BeginAddress and follows UnwindData without bounds checks. Measured over 3,780 x64 images in
// System32 (5,023,071 entries) and the package's FidelityFX DLLs: every table is sorted and non-overlapping and in
// bounds, and one shipped image (symcryptk.dll) has a zero-length entry (BeginAddress == EndAddress), so that is
// allowed. No entry in that set uses the indirect form (UnwindData bit 0).
void Loader::register_unwind(Module& m) {
    auto* nt = at<IMAGE_NT_HEADERS64>(m.base, reinterpret_cast<IMAGE_DOS_HEADER*>(m.base)->e_lfanew);
    const auto dir = directory(nt, IMAGE_DIRECTORY_ENTRY_EXCEPTION);
    // Like the system loader, a trailing partial entry is ignored.
    const uint32_t count = dir.Size / sizeof(RUNTIME_FUNCTION);
    if (!count) return;
    require_range(m.name, dir.VirtualAddress, static_cast<uint64_t>(count) * sizeof(RUNTIME_FUNCTION), m.size,
                  "the exception directory");
    auto* table = at<RUNTIME_FUNCTION>(m.base, dir.VirtualAddress);
    auto reject = [&](uint32_t i, const char* why) {
        throw std::runtime_error(m.name + ": unwind entry " + std::to_string(i) + " " + why);
    };
    uint32_t previous_end = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const RUNTIME_FUNCTION& f = table[i];
        if (f.EndAddress < f.BeginAddress || f.EndAddress > m.size) reject(i, "covers code outside the image");
        if (f.BeginAddress < previous_end) reject(i, "is out of order or overlaps the one before it");
        // UNWIND_INFO starts with a 4-byte header; its codes and handler data are the unwinder's to read.
        const uint32_t info = f.UnwindData & ~1u;
        if (info > m.size || m.size - info < 4) reject(i, "has unwind data outside the image");
        previous_end = f.EndAddress;
    }
    if (!RtlAddFunctionTable(table, count, reinterpret_cast<DWORD64>(m.base)))
        throw std::runtime_error(m.name + ": RtlAddFunctionTable failed");
    m.function_table = table;
    m.function_count = count;
}

void Loader::initialise() {
    // Not order_: map_image pushes a module there before bind_imports maps its dependencies, so order_ lists an
    // importer ahead of the DLLs it imports. init_order_ is filled once a module's imports are all loaded.
    for (auto* m : init_order_) initialise(*m);
}

void Loader::initialise(Module& m) {
    if (m.initialised) return;

    // TLS was installed by load(); initialise() only delivers the events, so no slot is allocated or written twice.
    run_tls_callbacks(m, DLL_PROCESS_ATTACH);

    // Only a DLL gets DllMain. An EXE's entry point has a different signature and is the caller's to run.
    auto* nt = at<IMAGE_NT_HEADERS64>(m.base, reinterpret_cast<IMAGE_DOS_HEADER*>(m.base)->e_lfanew);
    if ((nt->FileHeader.Characteristics & IMAGE_FILE_DLL) && m.entry_rva) {
        auto entry = reinterpret_cast<BOOL(NTAPI*)(void*, DWORD, void*)>(m.base + m.entry_rva);
        const BOOL ok = entry(m.base, DLL_PROCESS_ATTACH, nullptr);
        if (startup_trace::enabled())
            startup_trace::event("dllmain", "\"module\":" + startup_trace::quote(m.name) +
                                                ",\"result\":" + std::to_string(ok));
        if (!ok) {
            set_initialised(m, true);  // it ran: a second DLL_PROCESS_ATTACH would be wrong
            throw std::runtime_error(m.name + ": DllMain returned FALSE");
        }
    }
    set_initialised(m, true);
}

// Under the TLS lock, which orders it against the thread events: they run a module's TLS callbacks only while this is
// true (from initialise() until ~Loader starts the process detach).
void Loader::set_initialised(Module& m, bool value) {
    ExclusiveTlsLock lock;
    m.initialised = value;
}

Module& Loader::map_image(const std::filesystem::path& path) {
    auto file = read_file(path);
    const std::string name = path.filename().string();
    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(file.data());
    if (file.size() < sizeof(IMAGE_DOS_HEADER) || dos->e_magic != IMAGE_DOS_SIGNATURE)
        throw std::runtime_error(path.string() + ": not an MZ image");
    if (dos->e_lfanew < 0 || static_cast<uint64_t>(dos->e_lfanew) + sizeof(IMAGE_NT_HEADERS64) > file.size())
        throw std::runtime_error(name + ": the NT headers are outside the file");
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(file.data() + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE || nt->FileHeader.Machine != IMAGE_FILE_MACHINE_AMD64)
        throw std::runtime_error(name + ": not an x86-64 PE");
    if (nt->FileHeader.SizeOfOptionalHeader < sizeof(IMAGE_OPTIONAL_HEADER64) ||
        nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC)
        throw std::runtime_error(name + ": not a PE32+ image");
    auto& oh = nt->OptionalHeader;
    const uint64_t sections = static_cast<uint64_t>(dos->e_lfanew) + 4 + sizeof(IMAGE_FILE_HEADER) +
                              nt->FileHeader.SizeOfOptionalHeader;
    const uint64_t headers_end =
        sections + static_cast<uint64_t>(nt->FileHeader.NumberOfSections) * sizeof(IMAGE_SECTION_HEADER);
    if (headers_end > file.size())
        throw std::runtime_error(name + ": the section table runs past the end of the file");
    if (!oh.SizeOfImage || oh.SizeOfHeaders < headers_end || oh.SizeOfHeaders > oh.SizeOfImage ||
        oh.SizeOfHeaders > file.size())
        throw std::runtime_error(name + ": SizeOfHeaders and SizeOfImage don't fit the headers and file");
    if (oh.AddressOfEntryPoint && oh.AddressOfEntryPoint >= oh.SizeOfImage)
        throw std::runtime_error(name + ": the entry point is outside the image");

    auto* m = new Module;
    m->name = name;
    m->path = path;
    m->preferred_base = oh.ImageBase;
    m->size = oh.SizeOfImage;
    m->entry_rva = oh.AddressOfEntryPoint;
    by_name_[lower(m->name)] = m;
    order_.push_back(m);

    // Inside the range gta6rt reserved for this image before the process started (base_reservation.h), else as usual.
    m->base = static_cast<uint8_t*>(base_reservation::commit_reserved(oh.ImageBase, oh.SizeOfImage));
    if (!m->base)
        m->base = static_cast<uint8_t*>(VirtualAlloc(reinterpret_cast<void*>(oh.ImageBase), oh.SizeOfImage,
                                                     MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!m->base) {
        const auto relocs = directory(nt, IMAGE_DIRECTORY_ENTRY_BASERELOC);
        if (!relocs.Size)
            throw std::runtime_error(m->name + ": preferred base is taken and the image has no relocations");
        m->base = static_cast<uint8_t*>(VirtualAlloc(nullptr, oh.SizeOfImage, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (!m->base) throw std::runtime_error(m->name + ": VirtualAlloc failed");
    }

    std::memcpy(m->base, file.data(), oh.SizeOfHeaders);
    auto* sec = IMAGE_FIRST_SECTION(nt);
    for (unsigned i = 0; i < nt->FileHeader.NumberOfSections; ++i, ++sec) {
        uint32_t n = std::min(sec->SizeOfRawData, sec->Misc.VirtualSize ? sec->Misc.VirtualSize : sec->SizeOfRawData);
        if (!n) continue;
        if (sec->PointerToRawData > file.size() || n > file.size() - sec->PointerToRawData)
            throw std::runtime_error(m->name + ": section " + std::to_string(i) + " reads past the end of the file");
        require_range(m->name, sec->VirtualAddress, n, m->size, "a section's data");
        std::memcpy(m->base + sec->VirtualAddress, file.data() + sec->PointerToRawData, n);
    }

    int64_t delta = reinterpret_cast<int64_t>(m->base) - static_cast<int64_t>(oh.ImageBase);
    if (delta) {
        m->relocated = true;
        const auto dir = directory(nt, IMAGE_DIRECTORY_ENTRY_BASERELOC);
        if (!dir.Size) throw std::runtime_error(m->name + ": the image has no relocations and isn't at its base");
        require_range(m->name, dir.VirtualAddress, dir.Size, m->size, "the relocation table");
        auto* block = at<IMAGE_BASE_RELOCATION>(m->base, dir.VirtualAddress);
        auto* end = reinterpret_cast<uint8_t*>(block) + dir.Size;
        while (reinterpret_cast<uint8_t*>(block) < end) {
            const size_t remaining = static_cast<size_t>(end - reinterpret_cast<uint8_t*>(block));
            if (remaining < sizeof(IMAGE_BASE_RELOCATION) || block->SizeOfBlock < sizeof(IMAGE_BASE_RELOCATION) ||
                block->SizeOfBlock > remaining)
                throw std::runtime_error(m->name + ": a relocation block doesn't fit the relocation table");
            auto* entries = reinterpret_cast<uint16_t*>(block + 1);
            size_t count = (block->SizeOfBlock - sizeof(IMAGE_BASE_RELOCATION)) / 2;
            for (size_t i = 0; i < count; ++i) {
                uint16_t type = entries[i] >> 12, off = entries[i] & 0xFFF;
                if (type == IMAGE_REL_BASED_DIR64) {
                    require_range(m->name, static_cast<uint64_t>(block->VirtualAddress) + off, sizeof(uint64_t), m->size,
                                  "a relocation");
                    *at<uint64_t>(m->base, block->VirtualAddress + off) += delta;
                } else if (type == IMAGE_REL_BASED_HIGHLOW) {
                    // A 32-bit absolute address (code built /LARGEADDRESSAWARE:NO), which the x64 Windows loader
                    // also applies. Where the new address no longer fits in 32 bits, Windows would truncate it; the
                    // image would then jump or read through a wrong pointer, so the load is refused instead.
                    require_range(m->name, static_cast<uint64_t>(block->VirtualAddress) + off, sizeof(uint32_t),
                                  m->size, "a relocation");
                    auto* slot = at<uint32_t>(m->base, block->VirtualAddress + off);
                    const int64_t moved = static_cast<int64_t>(*slot) + delta;
                    if (moved < 0 || moved > 0xFFFFFFFFll)
                        throw std::runtime_error(m->name + ": a 32-bit (HIGHLOW) relocation does not fit at this base");
                    *slot = static_cast<uint32_t>(moved);
                } else if (type != IMAGE_REL_BASED_ABSOLUTE) {
                    throw std::runtime_error(m->name + ": unsupported relocation type " + std::to_string(type));
                }
            }
            block = reinterpret_cast<IMAGE_BASE_RELOCATION*>(reinterpret_cast<uint8_t*>(block) + block->SizeOfBlock);
        }
    }
    // The TLS directory and the exception directory hold virtual addresses, so both are read after relocation. TLS is
    // installed here, before bind_imports can return control to the caller and before any image code can run.
    initialize_tls(*m);
    parse_exports(*m);
    return *m;
}

// Static TLS for the thread that calls load(). The PE TLS directory only carries the template (Start..End) and the
// zero-fill size; it has no field naming the TLS block. The Windows loader allocates a block per module per thread and
// puts it in TEB->ThreadLocalStoragePointer[module index], which is what __declspec(thread) code reads via _tls_index.
// Only the OS loader can allocate that module index, so the loader reuses the index the OS assigned to
// gta6rt_tls_host.dll and maps a fresh clone of the helper for every additional simultaneous TLS module, each clone
// getting its own index. All kMaxMappedTlsModules slots can be live at once; a further module is rejected rather than
// silently sharing (and corrupting) a slot. An executable takes index 0 instead (kMainImageSlot), as on Windows.
void Loader::initialize_tls(Module& m) {
    auto* tls = tls_directory(m);
    if (!tls) return;
    // A present but all-zero directory is treated like an absent one.
    if (!tls->StartAddressOfRawData && !tls->EndAddressOfRawData && !tls->AddressOfIndex && !tls->AddressOfCallBacks &&
        !tls->SizeOfZeroFill)
        return;
    if (tls->EndAddressOfRawData < tls->StartAddressOfRawData)
        throw std::runtime_error(m.name + ": TLS template end precedes its start");
    uint64_t raw = tls->EndAddressOfRawData - tls->StartAddressOfRawData;
    uint64_t total = raw + tls->SizeOfZeroFill;
    if (total > 0xFFFFFFFFull) throw std::runtime_error(m.name + ": TLS block is too large");
    const uint32_t raw_rva =
        raw ? static_cast<uint32_t>(rva_in(m, tls->StartAddressOfRawData, "the TLS template")) : 0;
    if (raw) require_range(m.name, raw_rva, raw, m.size, "the TLS template");
    uint64_t index_rva = 0;
    if (tls->AddressOfIndex) {
        index_rva = rva_in(m, tls->AddressOfIndex, "the TLS index");
        require_range(m.name, index_rva, sizeof(uint32_t), m.size, "the TLS index");
    }

    if (!module_tls_slots()) throw std::runtime_error(m.name + ": this thread has no TLS array");

    auto* nt = at<IMAGE_NT_HEADERS64>(m.base, reinterpret_cast<IMAGE_DOS_HEADER*>(m.base)->e_lfanew);
    const bool executable = (nt->FileHeader.Characteristics & IMAGE_FILE_DLL) == 0;

    size_t slot = kTlsSlotCount;
    DWORD index = TLS_OUT_OF_INDEXES;
    {
        std::lock_guard alloc_lock(g_tls_alloc_mutex);
        if (executable) {
            // Load/clone OUTSIDE ExclusiveTlsLock, as below; the move takes that lock itself.
            std::string why = move_index0_owner();
            if (why.empty() && g_tls_slot_owners[kMainImageSlot].load() == nullptr) {
                slot = kMainImageSlot;
                index = 0;
            } else {
                // Mapped on as a DLL is: code that reads _tls_index works, /GA code reads another image's slot 0.
                if (why.empty()) why = "another mapped executable holds it";
                std::fprintf(stderr,
                             "[rt] loader: %s does not get module TLS index 0 (%s); its code built with /GA will not "
                             "find its TLS\n",
                             m.name.c_str(), why.c_str());
            }
        }
        if (slot == kTlsSlotCount) {
            for (size_t i = 0; i < kMaxMappedTlsModules; ++i) {
                if (g_tls_slot_owners[i].load() == nullptr) {
                    slot = i;
                    break;
                }
            }
            if (slot >= kMaxMappedTlsModules)
                throw std::runtime_error(m.name + ": all " + std::to_string(kMaxMappedTlsModules) +
                                         " manually mapped static TLS slots are in use");

            // Load/clone gta6rt_tls_host.dll OUTSIDE ExclusiveTlsLock so LoadLibraryExW cannot deadlock with
            // on_host_thread_event (which runs under the Windows loader lock and takes SharedTlsLock).
            index = ensure_tls_host(slot).index;
            if (index == TLS_OUT_OF_INDEXES)
                throw std::runtime_error(m.name + ": gta6rt_tls_host.dll is missing or has no module TLS index");
            if (index >= kMaxModuleTlsSlots)
                throw std::runtime_error(m.name + ": reserved TLS index is out of range");
        }

        // The fields are complete before a thread event can see the owner.
        ExclusiveTlsLock lock;
        Module* expected = nullptr;
        if (!g_tls_slot_owners[slot].compare_exchange_strong(expected, &m))
            throw std::runtime_error(m.name + ": static TLS slot was claimed concurrently");

        m.tls_slot = static_cast<uint32_t>(slot);
        m.tls_generation = g_next_tls_generation.fetch_add(1, std::memory_order_relaxed);
        g_tls_slot_generations[slot].store(m.tls_generation);
        m.tls_index = index;
        m.tls_template = raw ? m.base + raw_rva : nullptr;
        m.tls_raw_size = static_cast<uint32_t>(raw);
        m.tls_zero_fill = tls->SizeOfZeroFill;
        m.tls_block_size = static_cast<uint32_t>(total);
        // The index lives in a writable section and protect_sections() has not run yet, so the write needs no
        // VirtualProtect dance.
        if (index_rva) *reinterpret_cast<uint32_t*>(m.base + index_rva) = index;
    }
    try {
        install_thread_block(m);
    } catch (...) {
        ExclusiveTlsLock lock;
        m.tls_index = kNoTlsIndex;
        m.tls_slot = kNoTlsIndex;
        m.tls_generation = 0;
        g_tls_slot_generations[slot].store(0);
        g_tls_slot_owners[slot].store(nullptr);
        throw;
    }
    // The loading thread's copy is the one the Module records (and ~Loader frees on that thread).
    if (g_thread_tls_slots[slot].module == &m && g_thread_tls_slots[slot].generation == m.tls_generation) {
        m.tls_block = g_thread_tls_slots[slot].block;
        m.tls_saved_block = g_thread_tls_slots[slot].saved;
    }
}

void Loader::install_tls_for_thread(Module& m) { install_thread_block(m); }

void Loader::uninstall_tls_for_thread(Module& m) { uninstall_thread_block(m); }

void Loader::run_tls_callbacks(Module& m, uint32_t reason) { run_callbacks(m, reason, /*trace=*/true); }

void Loader::parse_exports(Module& m) {
    auto* nt = at<IMAGE_NT_HEADERS64>(m.base, reinterpret_cast<IMAGE_DOS_HEADER*>(m.base)->e_lfanew);
    const auto dir = directory(nt, IMAGE_DIRECTORY_ENTRY_EXPORT);
    if (!dir.Size) return;
    require_range(m.name, dir.VirtualAddress, sizeof(IMAGE_EXPORT_DIRECTORY), m.size, "the export directory");
    auto* ed = at<IMAGE_EXPORT_DIRECTORY>(m.base, dir.VirtualAddress);
    require_range(m.name, ed->AddressOfFunctions, static_cast<uint64_t>(ed->NumberOfFunctions) * sizeof(uint32_t), m.size,
                  "the export address table");
    require_range(m.name, ed->AddressOfNames, static_cast<uint64_t>(ed->NumberOfNames) * sizeof(uint32_t), m.size,
                  "the export name table");
    require_range(m.name, ed->AddressOfNameOrdinals, static_cast<uint64_t>(ed->NumberOfNames) * sizeof(uint16_t), m.size,
                  "the export ordinal table");
    auto* funcs = at<uint32_t>(m.base, ed->AddressOfFunctions);
    auto* names = at<uint32_t>(m.base, ed->AddressOfNames);
    auto* ords = at<uint16_t>(m.base, ed->AddressOfNameOrdinals);
    // An export whose RVA lands inside the export directory is a forwarder string ("KERNEL32.Sleep",
    // "api-ms-win-core-file-l1-1-0.CreateFileW"), not code. Storing base + that RVA would hand callers a pointer
    // into the string, so forwarders are kept apart and followed when an import asks for one.
    const uint64_t dir_begin = dir.VirtualAddress, dir_end = dir_begin + dir.Size;
    auto add_export = [&](const std::string& key, uint32_t rva) {
        if (!rva) return;  // An unused address-table slot is absent even when a name refers to it.
        if (rva >= dir_begin && rva < dir_end)
            m.forwarders[key] = read_string(m.name, m.base, m.size, rva, "a forwarded export");
        else if (rva >= m.size)  // base + rva would point past the mapping, and a caller would jump there
            throw std::runtime_error(m.name + ": export " + key + " points outside the image");
        else
            m.exports[key] = m.base + rva;
    };
    for (uint32_t i = 0; i < ed->NumberOfFunctions; ++i)
        if (funcs[i]) add_export("#" + std::to_string(ed->Base + i), funcs[i]);
    for (uint32_t i = 0; i < ed->NumberOfNames; ++i) {
        if (ords[i] >= ed->NumberOfFunctions)
            throw std::runtime_error(m.name + ": an export name points past the address table");
        add_export(read_string(m.name, m.base, m.size, names[i], "an export name"), funcs[ords[i]]);
    }
}

// Follows a forwarder string to the module and export it names. Bundled modules are searched first, with or without
// the extension, because a forwarder may point at another mapped module; system DLLs go through LoadLibraryEx, whose
// own GetProcAddress already follows its forwarders. Errors are "not found": the caller decides what to bind.
void* Loader::resolve_forwarder(const std::string& spec, Binding& binding, int depth) {
    if (depth >= 8) return nullptr;  // a longer chain is a loop between modules, not a re-export
    auto dot = spec.rfind('.');
    if (dot == std::string::npos || dot == 0 || dot + 1 >= spec.size()) return nullptr;
    const std::string module = spec.substr(0, dot), symbol = spec.substr(dot + 1);

    // A forwarder can name a module the loader shims. Shims take precedence over a bundled or host copy, exactly as
    // they do for a direct import in resolve(); without this, a forwarded xmem or d3d12_xs call becomes a failure thunk.
    std::string shim_key = lower(module);
    if (shim_key.size() < 4 || shim_key.compare(shim_key.size() - 4, 4, ".dll") != 0) shim_key += ".dll";
    if (void* p = shims::find(shim_key, symbol)) {
        binding = Binding::Shim;
        return p;
    }

    const bool by_ordinal = symbol.size() > 1 && symbol[0] == '#';
    uint32_t ordinal = 0;
    if (by_ordinal) {
        for (size_t i = 1; i < symbol.size(); ++i) {
            if (symbol[i] < '0' || symbol[i] > '9') return nullptr;
            ordinal = ordinal * 10 + static_cast<uint32_t>(symbol[i] - '0');
            if (ordinal > 0xFFFF) return nullptr;  // checked after the digit, so "#65537" can't truncate to 1
        }
    }

    // shim_key, not the bare module name: forwarder strings usually leave out ".dll" ("MSVCP140.?foo"), and the host
    // CRT copies are named with it, so a forwarded CRT symbol must skip the bundled Xbox build as a direct import does.
    const std::filesystem::path candidates[] = {game_dir_ / module, game_dir_ / (module + ".dll")};
    for (const auto& candidate : candidates) {
        std::error_code ec;
        if (prefer_host_copy(shim_key) || !std::filesystem::exists(candidate, ec)) continue;
        Module& dep = load(candidate);
        if (auto it = dep.exports.find(symbol); it != dep.exports.end()) {
            binding = Binding::Module;
            return it->second;
        }
        if (auto it = dep.forwarders.find(symbol); it != dep.forwarders.end())
            return resolve_forwarder(it->second, binding, depth + 1);
        return nullptr;  // the bundled module exists and does not export it; don't look for a host namesake
    }

    HMODULE h = LoadLibraryExA(module.c_str(), nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!h) return nullptr;
    if (prefer_host_copy(shim_key)) redirect_pc_to_file_header(h);
    FARPROC p = GetProcAddress(h, by_ordinal ? MAKEINTRESOURCEA(static_cast<WORD>(ordinal)) : symbol.c_str());
    if (!p) return nullptr;
    if (prefer_host_copy(shim_key)) redirect_pc_to_file_header_at(reinterpret_cast<const void*>(p));
    binding = Binding::Host;
    return reinterpret_cast<void*>(p);
}

void* Loader::resolve(Module& m, const std::string& dll, const std::string& name, uint16_t ordinal, bool by_ordinal) {
    std::string key = lower(dll);
    std::string sym = by_ordinal ? "#" + std::to_string(ordinal) : name;

    if (void* p = shims::find(key, sym)) {
        m.imports.push_back({dll, sym, Binding::Shim});
        return p;
    }
    // Bundled game module (an Xbox build that ships beside the game)?
    auto bundled = game_dir_ / dll;
    if (!prefer_host_copy(key) && std::filesystem::exists(bundled)) {
        Module& dep = load(bundled);
        if (auto it = dep.exports.find(sym); it != dep.exports.end()) {
            m.imports.push_back({dll, sym, Binding::Module});
            return it->second;
        }
        if (auto it = dep.forwarders.find(sym); it != dep.forwarders.end()) {
            Binding binding = Binding::Unimplemented;
            if (void* p = resolve_forwarder(it->second, binding, 0)) {
                m.imports.push_back({dll, sym, binding});
                return p;
            }
        }
    }
    // Host Windows: system DLLs and API sets. LOAD_LIBRARY_SEARCH_SYSTEM32 keeps the game folder out of the search.
    if (HMODULE h = LoadLibraryExA(dll.c_str(), nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32)) {
        if (prefer_host_copy(key)) redirect_pc_to_file_header(h);  // before any mapped code can throw through it
        FARPROC p = GetProcAddress(h, by_ordinal ? MAKEINTRESOURCEA(ordinal) : name.c_str());
        if (p) {
            if (prefer_host_copy(key)) redirect_pc_to_file_header_at(reinterpret_cast<const void*>(p));
            m.imports.push_back({dll, sym, Binding::Host});
            return reinterpret_cast<void*>(p);
        }
    }
    m.imports.push_back({dll, sym, Binding::Unimplemented});
    return shims::make_unimplemented_thunk(dll, sym);
}

void Loader::bind_imports(Module& m) {
    auto* nt = at<IMAGE_NT_HEADERS64>(m.base, reinterpret_cast<IMAGE_DOS_HEADER*>(m.base)->e_lfanew);

    // Thunk tables have no length field: they end at a zero entry, so each step is checked against the image.
    auto bind_thunks = [&](const std::string& dll, uint64_t lookup_rva, uint64_t iat_rva) {
        for (;; lookup_rva += sizeof(IMAGE_THUNK_DATA64), iat_rva += sizeof(IMAGE_THUNK_DATA64)) {
            require_range(m.name, lookup_rva, sizeof(IMAGE_THUNK_DATA64), m.size, "an import lookup table");
            require_range(m.name, iat_rva, sizeof(IMAGE_THUNK_DATA64), m.size, "an import address table");
            auto* lookup = at<IMAGE_THUNK_DATA64>(m.base, static_cast<uint32_t>(lookup_rva));
            if (!lookup->u1.AddressOfData) return;
            bool by_ord = IMAGE_SNAP_BY_ORDINAL64(lookup->u1.Ordinal);
            uint16_t ord = by_ord ? (uint16_t)IMAGE_ORDINAL64(lookup->u1.Ordinal) : 0;
            std::string name;
            if (!by_ord) {
                require_range(m.name, lookup->u1.AddressOfData, sizeof(uint16_t), m.size, "an import name");
                uint32_t hint = static_cast<uint32_t>(lookup->u1.AddressOfData);
                name = read_string(m.name, m.base, m.size, hint + sizeof(uint16_t), "an import name");
            }
            at<IMAGE_THUNK_DATA64>(m.base, static_cast<uint32_t>(iat_rva))->u1.Function =
                reinterpret_cast<uint64_t>(resolve(m, dll, name, ord, by_ord));
        }
    };

    const auto dir = directory(nt, IMAGE_DIRECTORY_ENTRY_IMPORT);
    if (dir.Size) {
        for (uint64_t i = 0;; ++i) {
            uint64_t rva = dir.VirtualAddress + i * sizeof(IMAGE_IMPORT_DESCRIPTOR);
            require_range(m.name, rva, sizeof(IMAGE_IMPORT_DESCRIPTOR), m.size, "the import table");
            auto* d = at<IMAGE_IMPORT_DESCRIPTOR>(m.base, static_cast<uint32_t>(rva));
            if (!d->Name) break;
            std::string dll = read_string(m.name, m.base, m.size, d->Name, "an imported DLL name");
            uint32_t lookup = d->OriginalFirstThunk ? d->OriginalFirstThunk : d->FirstThunk;
            bind_thunks(dll, lookup, d->FirstThunk);
        }
    }
    // Delay imports are bound eagerly: simpler, and the full import surface shows up in --dry-run.
    const auto ddir = directory(nt, IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT);
    if (ddir.Size) {
        for (uint64_t i = 0;; ++i) {
            uint64_t rva = ddir.VirtualAddress + i * sizeof(IMAGE_DELAYLOAD_DESCRIPTOR);
            require_range(m.name, rva, sizeof(IMAGE_DELAYLOAD_DESCRIPTOR), m.size, "the delay-import table");
            auto* d = at<IMAGE_DELAYLOAD_DESCRIPTOR>(m.base, static_cast<uint32_t>(rva));
            if (!d->DllNameRVA) break;
            // x64 descriptors always carry dlattrRva (0x1). Without it the fields would be 32-bit virtual
            // addresses, which cannot express a 64-bit image base, so the descriptor is not usable as written.
            if (!d->Attributes.RvaBased)
                throw std::runtime_error(m.name + ": a delay-import descriptor without the RVA attribute");
            if (!d->ImportAddressTableRVA)
                throw std::runtime_error(m.name + ": a delay-import descriptor has no address table");
            std::string dll = read_string(m.name, m.base, m.size, d->DllNameRVA, "a delay-imported DLL name");
            // Without a name table the IAT still holds delay thunks, not import names, so there is no symbol to bind
            // eagerly. Every MSVC delay descriptor and every System32 DLL has one; reject the form instead of reading
            // thunk addresses as name RVAs.
            if (!d->ImportNameTableRVA)
                throw std::runtime_error(m.name + ": a delay-import descriptor has no name table");
            bind_thunks(dll, d->ImportNameTableRVA, d->ImportAddressTableRVA);
        }
    }
}

void Loader::protect_sections(Module& m) {
    auto* nt = at<IMAGE_NT_HEADERS64>(m.base, reinterpret_cast<IMAGE_DOS_HEADER*>(m.base)->e_lfanew);
    DWORD old;
    VirtualProtect(m.base, nt->OptionalHeader.SizeOfHeaders, PAGE_READONLY, &old);
    auto* sec = IMAGE_FIRST_SECTION(nt);
    for (unsigned i = 0; i < nt->FileHeader.NumberOfSections; ++i, ++sec) {
        bool x = sec->Characteristics & IMAGE_SCN_MEM_EXECUTE, w = sec->Characteristics & IMAGE_SCN_MEM_WRITE;
        DWORD prot = x ? (w ? PAGE_EXECUTE_READWRITE : PAGE_EXECUTE_READ) : (w ? PAGE_READWRITE : PAGE_READONLY);
        uint64_t size = sec->Misc.VirtualSize ? sec->Misc.VirtualSize : sec->SizeOfRawData;
        if (sec->VirtualAddress >= m.size) continue;
        size = std::min<uint64_t>(size, m.size - sec->VirtualAddress);
        if (size) VirtualProtect(m.base + sec->VirtualAddress, static_cast<SIZE_T>(size), prot, &old);
    }
    FlushInstructionCache(GetCurrentProcess(), m.base, m.size);
}

}  // namespace rt
