// End-to-end checks for module TLS index 0 and a mapped executable, against locally built fixtures.
//
// Windows gives the process executable module TLS index 0, and MSVC code built with /GA relies on it: it reads
// TEB->ThreadLocalStoragePointer[0] without loading _tls_index. tiny_tls_ga.exe is such an image (a thread_local int
// with the initializer 42, built /GA). The loader frees index 0 by moving this test EXE's own block (index 0 before
// the first executable is mapped) to a helper clone's index on every thread. The checks: the executable reads 42 on
// the loading thread, on a thread that existed before the move, and on threads started later; this EXE's own
// thread_local data stays its own on all of them; tiny_tls.dll keeps a helper index alongside; two runs one after the
// other both get index 0, and a second executable mapped meanwhile gets a helper index.
//
// gta6rt_tls_host.dll must sit next to this EXE, as for loader_tls_test. No game files are needed.
#include <windows.h>
#include <intrin.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>

#include "pe_loader.h"

extern "C" unsigned long _tls_index;  // this EXE's module TLS index, read through its TLS directory's AddressOfIndex

namespace {

namespace fs = std::filesystem;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

// This EXE's own TLS: thread_local reads here go through _tls_index, so they follow the move.
thread_local int t_host = 1234;

__declspec(noinline) int host_value() { return t_host; }
__declspec(noinline) void set_host_value(int value) { t_host = value; }
__declspec(noinline) const void* host_address() { return &t_host; }

void** module_tls_slots() { return reinterpret_cast<void**>(__readgsqword(0x58)); }

// Whether t_host lies in the block this thread's slot _tls_index holds (the block is larger than this file's TLS).
bool host_tls_in_own_slot() {
    const auto* block = static_cast<const uint8_t*>(module_tls_slots()[_tls_index]);
    const auto* p = static_cast<const uint8_t*>(host_address());
    return block && p >= block && p < block + 0x10000;
}

template <class Fn>
Fn find_export(const rt::Module& m, const char* name) {
    const auto it = m.exports.find(name);
    if (it == m.exports.end()) throw std::runtime_error(std::string("missing export ") + name);
    return reinterpret_cast<Fn>(it->second);
}

struct Ga {
    int(__cdecl* get)();
    void(__cdecl* set)(int);
    int*(__cdecl* address)();
    unsigned long(__cdecl* tls_index)();
    unsigned long*(__cdecl* tls_index_address)();
};

Ga ga_exports(const rt::Module& m) {
    return {find_export<int(__cdecl*)()>(m, "tiny_ga_get"), find_export<void(__cdecl*)(int)>(m, "tiny_ga_set"),
            find_export<int*(__cdecl*)()>(m, "tiny_ga_address"),
            find_export<unsigned long(__cdecl*)()>(m, "tiny_ga_tls_index"),
            find_export<unsigned long*(__cdecl*)()>(m, "tiny_ga_tls_index_address")};
}

// Code bytes of a function, past an incremental-linking jump stub (Debug builds export the stub).
const uint8_t* code_of(const void* fn) {
    const auto* p = static_cast<const uint8_t*>(fn);
    if (p[0] == 0xE9) {
        int32_t rel;
        std::memcpy(&rel, p + 1, 4);
        p += 5 + rel;
    }
    return p;
}

// The /GA shape of the getter, up to its ret: a load of gs:[58h] (65 REX.W 8B /r with a SIB of disp32 0x58) and no
// rip-relative operand that lands on the image's _tls_index.
bool reads_slot0_without_index(const void* fn, const void* tls_index) {
    const uint8_t* p = code_of(fn);
    bool gs58 = false, index = false;
    for (size_t i = 0; i < 64 && p[i] != 0xC3; ++i) {
        if (p[i] == 0x65 && (p[i + 1] & 0xF8) == 0x48 && p[i + 2] == 0x8B && (p[i + 3] & 0xC7) == 0x04 &&
            p[i + 4] == 0x25 && p[i + 5] == 0x58 && !p[i + 6] && !p[i + 7] && !p[i + 8])
            gs58 = true;
        int32_t disp;
        std::memcpy(&disp, p + i, 4);
        for (int tail = 4; tail <= 5; ++tail)  // the displacement ends the instruction or is followed by an imm8
            index |= p + i + tail + disp == tls_index;
    }
    return gs58 && !index;
}

// What a thread started after load() sees, without any install call.
struct NewThreadView {
    int ga = -1, ga_after_set = -1, host = -1;
    bool host_in_own_slot = false;
    void* slot0 = nullptr;
    int dll = -1;
};

// ga may be null: then only this EXE's own TLS is looked at.
NewThreadView run_new_thread(const Ga* ga, int(__cdecl* dll_get)() = nullptr) {
    NewThreadView v;
    std::thread t([&] {
        v.slot0 = module_tls_slots()[0];
        if (ga) {
            v.ga = ga->get();
            ga->set(9);
            v.ga_after_set = ga->get();
        }
        v.host = host_value();
        set_host_value(4321);
        v.host_in_own_slot = host_tls_in_own_slot();
        if (dll_get) v.dll = dll_get();
    });
    t.join();
    return v;
}

// A thread that exists before any executable is mapped. It holds its own value in t_host across the move, then
// installs the executable's copy itself (install_tls_for_thread, as for any module and a thread older than load()).
struct EarlyThread {
    HANDLE ready = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    HANDLE go = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    HANDLE done = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    HANDLE quit = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    rt::Loader* loader = nullptr;
    rt::Module* module = nullptr;
    Ga ga{};
    int host_after_move = -1, ga_installed = -1, ga_after_set = -1;
    bool host_in_own_slot = false, block_in_slot0 = false, slot0_restored = false;
    std::thread thread;

    EarlyThread() {
        thread = std::thread([this] {
            set_host_value(777);
            SetEvent(ready);
            WaitForSingleObject(go, INFINITE);
            if (!loader) return;  // the test stopped before it got this far
            host_after_move = host_value();
            host_in_own_slot = host_tls_in_own_slot();
            void* before = module_tls_slots()[0];
            loader->install_tls_for_thread(*module);
            block_in_slot0 = module_tls_slots()[0] != before;
            ga_installed = ga.get();
            ga.set(5);
            ga_after_set = ga.get();
            loader->uninstall_tls_for_thread(*module);
            slot0_restored = module_tls_slots()[0] == before;
            SetEvent(done);
            WaitForSingleObject(quit, INFINITE);  // ends after both Loaders are gone
        });
        WaitForSingleObject(ready, INFINITE);
    }
    ~EarlyThread() {
        SetEvent(go);
        SetEvent(quit);
        if (thread.joinable()) thread.join();
        for (HANDLE h : {ready, go, done, quit}) CloseHandle(h);
    }
};

// A thread that keeps running inside this EXE's code could be stopped between its load of _tls_index and its read
// of the slot, so the move waits for it and gives up after its time budget: nothing changes, the executable maps with
// a helper index, and the move is tried again for the next executable.
void test_busy_thread_refuses(const fs::path& exe) {
    std::printf("\n-- a thread busy in this EXE's code holds the move off --\n");
    std::atomic<bool> stop{false}, wrong{false};
    std::atomic<long> spins{0};
    std::thread spinner([&] {
        set_host_value(31337);
        while (!stop.load(std::memory_order_relaxed)) {
            if (host_value() != 31337) wrong = true;
            spins.fetch_add(1, std::memory_order_relaxed);
        }
    });
    while (spins.load() == 0) Sleep(0);
    {
        rt::Loader loader(exe.parent_path());
        rt::Module& m = loader.load(exe);
        check(m.tls_index != 0 && m.tls_index != rt::kNoTlsIndex,
              "while a thread runs inside this EXE's code the executable gets a helper index");
        check(_tls_index == 0 && host_value() == 555, "this EXE keeps index 0 and its value then");
    }
    stop = true;
    spinner.join();
    check(!wrong, "the busy thread read its own TLS value on every one of its spins");
}

void test_first_run(const fs::path& exe, const fs::path& dll, EarlyThread& early, unsigned long& moved_to) {
    std::printf("\n-- first run: %s --\n", exe.string().c_str());
    void* block = nullptr;
    void* saved = nullptr;
    {
        rt::Loader loader(exe.parent_path());
        rt::Module& m = loader.load(exe);
        const Ga ga = ga_exports(m);
        moved_to = _tls_index;
        std::printf("     this EXE's _tls_index is now %lu; the executable's block=%p\n", moved_to, m.tls_block);

        check(m.tls_index == 0, "the mapped executable gets module TLS index 0");
        check(ga.tls_index() == 0, "its AddressOfIndex (_tls_index) holds 0");
        check(m.tls_block && module_tls_slots()[0] == m.tls_block, "slot 0 of this thread holds its template copy");
        check(ga.get() == 42, "its /GA getter reads the template value 42 on the loading thread");
        check(moved_to != 0 && moved_to != rt::kNoTlsIndex, "this EXE's own _tls_index moved off 0");
        check(host_value() == 555, "this EXE's thread_local value survives the move on the loading thread");
        check(host_tls_in_own_slot(), "this EXE's TLS is in the slot its new _tls_index names");
        check(m.tls_saved_block && m.tls_saved_block != module_tls_slots()[moved_to],
              "the block saved from slot 0 is the helper clone's, not this EXE's");

        // The getter follows slot 0, whatever _tls_index says: a stand-in block there changes what it reads.
        const auto offset = reinterpret_cast<const uint8_t*>(ga.address()) - static_cast<const uint8_t*>(m.tls_block);
        alignas(64) static uint8_t stand_in[4096];
        bool offset_ok = offset >= 0 && static_cast<uint64_t>(offset) + sizeof(int) <= m.tls_block_size;
        check(offset_ok, "the executable's thread_local lies inside its slot-0 block");
        if (offset_ok && static_cast<size_t>(offset) + sizeof(int) <= sizeof(stand_in)) {
            const int marker = 77;
            std::memcpy(stand_in + offset, &marker, sizeof(marker));
            void* real = module_tls_slots()[0];
            module_tls_slots()[0] = stand_in;
            const int seen = ga.get();
            module_tls_slots()[0] = real;
            check(seen == 77, "the executable's getter reads slot 0 directly");
        }
#if LEONIDA_GA_FIXTURE_IS_GA
        check(reads_slot0_without_index(reinterpret_cast<const void*>(ga.get), ga.tls_index_address()),
              "the fixture's getter is /GA code: gs:[58h] and no load of _tls_index");
#else
        std::printf("SKIP the fixture's /GA code shape (this compiler may ignore /GA)\n");
#endif

        ga.set(43);
        const NewThreadView fresh = run_new_thread(&ga);
        check(fresh.ga == 42, "a thread started after load() reads 42 without an install call");
        check(fresh.ga_after_set == 9 && ga.get() == 43, "its write stays in its own copy");
        check(fresh.slot0 != m.tls_block, "its slot 0 is its own block, not the loading thread's");
        check(fresh.host == 1234 && fresh.host_in_own_slot,
              "this EXE's thread_local starts at its initializer there, in the moved slot");
        check(host_value() == 555, "this EXE's value on the loading thread is unchanged");

        early.loader = &loader;
        early.module = &m;
        early.ga = ga;
        SetEvent(early.go);
        WaitForSingleObject(early.done, INFINITE);
        check(early.host_after_move == 777 && early.host_in_own_slot,
              "a thread that existed before the move keeps this EXE's TLS value (777)");
        check(early.block_in_slot0 && early.ga_installed == 42 && early.ga_after_set == 5,
              "install_tls_for_thread gives that thread its own copy in slot 0");
        check(early.slot0_restored, "uninstall_tls_for_thread puts that thread's slot 0 back");

        loader.initialise();
        check(m.initialised, "initialise() runs the executable's TLS callbacks (it has none)");
        check(run_new_thread(&ga).ga == 42, "after initialise() a new thread still reads 42");

        // A DLL with TLS keeps a helper index next to the executable's 0.
        rt::Module& d = loader.load(dll);
        const auto dll_get = find_export<int(__cdecl*)()>(d, "tiny_tls_get");
        check(d.tls_index != rt::kNoTlsIndex && d.tls_index != 0 && d.tls_index != moved_to,
              "tiny_tls.dll gets a helper index, neither 0 nor this EXE's");
        check(dll_get() == 7 && ga.get() == 43 && host_value() == 555,
              "the DLL, the executable and this EXE each read their own value on the loading thread");
        const NewThreadView both = run_new_thread(&ga, dll_get);
        check(both.dll == 7 && both.ga == 42 && both.host == 1234,
              "a new thread reads the DLL's 7, the executable's 42 and this EXE's 1234");

        block = m.tls_block;
        saved = m.tls_saved_block;
    }
    check(module_tls_slots()[0] == saved, "~Loader puts the helper clone's block back in slot 0");
    MEMORY_BASIC_INFORMATION info{};
    check(VirtualQuery(block, &info, sizeof(info)) == 0 || info.State == MEM_FREE, "~Loader frees the copy");
    check(_tls_index == moved_to && host_value() == 555, "this EXE keeps its moved index and value after the run");
}

void test_second_run(const fs::path& exe, unsigned long moved_to) {
    std::printf("\n-- second run, and a second executable at once --\n");
    wchar_t temp[MAX_PATH];
    GetTempPathW(MAX_PATH, temp);
    const fs::path copy = fs::path(temp) / ("leonida_tls_index0_copy_" + std::to_string(GetCurrentProcessId()) + ".exe");
    // CopyFileW, not fs::copy_file: MSVC's copy_file calls CopyFile2, which Wine 9 doesn't implement.
    check(CopyFileW(exe.c_str(), copy.c_str(), FALSE) != 0, "copied the executable under a second name");
    {
        rt::Loader loader(exe.parent_path());
        rt::Module& m = loader.load(exe);
        const Ga ga = ga_exports(m);
        loader.initialise();
        check(m.tls_index == 0 && ga.get() == 42, "a second Loader maps the executable at index 0 again, value 42");
        check(_tls_index == moved_to, "the move is done once: this EXE's index stays where the first run put it");
        check(run_new_thread(&ga).ga == 42 && host_value() == 555, "new threads and this EXE's value still check out");

        rt::Module& other = loader.load(copy);
        const Ga other_ga = ga_exports(other);
        check(other.tls_index != 0 && other.tls_index != rt::kNoTlsIndex && other_ga.tls_index() == other.tls_index,
              "a second executable mapped meanwhile gets a helper index in its _tls_index");
        check(m.tls_index == 0 && ga.get() == 42, "the first keeps index 0 and its value");
        // The documented limit: /GA code of the second one reads slot 0, the first one's block.
        check(other_ga.address() == ga.address(), "the second executable's /GA code reads the first one's slot 0");
    }
    std::error_code ec;
    fs::remove(copy, ec);
}

void print_result() {
    if (g_failures == 0)
        std::printf("\nloader_tls_index0_test: OK (%d checks)\n", g_checks);
    else
        std::printf("\nloader_tls_index0_test: FAILED (%d checks, %d failure%s)\n", g_checks, g_failures,
                    g_failures == 1 ? "" : "s");
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("loader_tls_index0_test: module TLS index 0 for a mapped executable\n");
    if (argc != 3) {
        std::fprintf(stderr, "usage: loader_tls_index0_test <tiny_tls_ga.exe> <tiny_tls.dll>\n");
        return 2;
    }
    const fs::path exe = argv[1], dll = argv[2];
    try {
        check(fs::exists(exe) && fs::exists(dll), "the fixtures exist");
        // Measured, not assumed: before anything is mapped, index 0 is this EXE's (Windows gives it to the process
        // executable when that has TLS, as every EXE linking pe_loader.cpp does).
        check(_tls_index == 0, "before any executable is mapped, this test EXE holds module TLS index 0");
        set_host_value(555);
        unsigned long moved_to = 0;
        {
            EarlyThread early;
            test_busy_thread_refuses(exe);
            test_first_run(exe, dll, early, moved_to);
            test_second_run(exe, moved_to);
        }  // the early thread ends here, after both Loaders
        check(run_new_thread(nullptr).host == 1234,
              "a thread started after both runs still gets this EXE's TLS at its moved index");
    } catch (const std::exception& e) {
        ++g_checks;
        ++g_failures;
        std::printf("FAIL unexpected exception: %s\n", e.what());
    }
    print_result();
    return g_failures ? 1 : 0;
}
