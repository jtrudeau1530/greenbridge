// End-to-end checks for the loader's static TLS, against locally built fixtures.
//
// tiny_tls.dll has a __declspec(thread) variable, a TLS callback in .CRT$XLB and a DllMain that counts calls. The test
// proves that load() installs the TLS template and index without running any image code, and that the callbacks only
// run through the opt-in run_tls_callbacks().
//
// The loader reserves its module TLS index with gta6rt_tls_host.dll, so that DLL must sit next to this EXE; CTest
// builds and places both in the build directory. No game files are needed and no real package DLL is mapped.
#include <windows.h>
#include <intrin.h>
#include <process.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "pe_loader.h"

namespace {

namespace fs = std::filesystem;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

std::vector<uint8_t> read_file(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path.string());
    return {std::istreambuf_iterator<char>(f), {}};
}

void write_file(const fs::path& path, const void* data, size_t size) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot create " + path.string());
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!f) throw std::runtime_error("cannot write " + path.string());
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() / ("leonida_loader_tls_test_" + std::to_string(GetCurrentProcessId()));
        std::error_code ec;
        fs::remove_all(path, ec);
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
};

const IMAGE_TLS_DIRECTORY64* mapped_tls(const rt::Module& m) {
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(m.base);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(m.base + dos->e_lfanew);
    const auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_TLS];
    if (!dir.Size || !dir.VirtualAddress) return nullptr;
    return reinterpret_cast<const IMAGE_TLS_DIRECTORY64*>(m.base + dir.VirtualAddress);
}

const IMAGE_RUNTIME_FUNCTION_ENTRY* mapped_pdata(const rt::Module& m, uint32_t* count) {
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(m.base);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(m.base + dos->e_lfanew);
    const auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXCEPTION];
    *count = dir.Size / sizeof(IMAGE_RUNTIME_FUNCTION_ENTRY);
    if (!dir.Size || !dir.VirtualAddress) return nullptr;
    return reinterpret_cast<const IMAGE_RUNTIME_FUNCTION_ENTRY*>(m.base + dir.VirtualAddress);
}

// The loader only initializes the thread that calls load(). install_tls_for_thread()/uninstall_tls_for_thread() are
// the seam the runtime must use for every thread it creates; the worker case below exercises it.
void** module_tls_slots() { return reinterpret_cast<void**>(__readgsqword(0x58)); }

template <class Fn>
Fn find_export(const rt::Module& m, const char* name) {
    const auto it = m.exports.find(name);
    if (it == m.exports.end()) throw std::runtime_error(std::string("missing export ") + name);
    return reinterpret_cast<Fn>(it->second);
}

// Writes copies of the fixture with one optional-header data directory zeroed. Relocations and headers are intact.
fs::path copy_without_directory(const fs::path& source, const fs::path& destination, size_t directory) {
    std::vector<uint8_t> bytes = read_file(source);
    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(bytes.data());
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(bytes.data() + dos->e_lfanew);
    nt->OptionalHeader.DataDirectory[directory].VirtualAddress = 0;
    nt->OptionalHeader.DataDirectory[directory].Size = 0;
    write_file(destination, bytes.data(), bytes.size());
    return destination;
}

void test_tls(const fs::path& dll) {
    std::printf("\n-- static TLS: %s --\n", dll.string().c_str());
    uint32_t index = rt::kNoTlsIndex;
    void* block = nullptr;
    void* saved = nullptr;
    {
        rt::Loader loader(dll.parent_path());
        rt::Module& m = loader.load(dll);

        const auto* tls = mapped_tls(m);
        check(tls != nullptr, "the mapped fixture has a TLS directory");
        check(m.tls_index != rt::kNoTlsIndex, "load() assigns the reserved module TLS index");
        check(m.tls_block != nullptr && m.tls_block_size >= sizeof(int), "load() installs a TLS template block");
        if (tls) {
            check(tls->AddressOfIndex != 0, "AddressOfIndex is non-null");
            if (tls->AddressOfIndex) {
                const auto stored = *reinterpret_cast<const DWORD*>(static_cast<uintptr_t>(tls->AddressOfIndex));
                check(stored == m.tls_index, "AddressOfIndex holds the module TLS index");
            }
            check(tls->AddressOfCallBacks != 0, "AddressOfCallBacks is non-null");
            const uint64_t template_size =
                tls->EndAddressOfRawData - tls->StartAddressOfRawData + tls->SizeOfZeroFill;
            check(m.tls_block_size == template_size, "the allocated block covers the template and zero-fill");
        }
        std::printf("     tls_index=%lu block=%p size=%u\n", static_cast<unsigned long>(m.tls_index), m.tls_block,
                    m.tls_block_size);
        check(m.tls_saved_block != nullptr && m.tls_saved_block != m.tls_block,
              "the loader saved the helper DLL's block before replacing it");
        check(module_tls_slots()[m.tls_index] == m.tls_block,
              "the module TLS array (gs:[58h]) points at the template copy");

        const auto get = find_export<int(__cdecl*)()>(m, "tiny_tls_get");
        const auto set = find_export<void(__cdecl*)(int)>(m, "tiny_tls_set");
        const auto callback_count = find_export<int(__cdecl*)()>(m, "tiny_tls_callback_count");
        const auto attach_count = find_export<int(__cdecl*)()>(m, "tiny_tls_attach_count");
        const auto detach_count = find_export<int(__cdecl*)()>(m, "tiny_tls_detach_count");
        const auto last_reason = find_export<DWORD(__cdecl*)()>(m, "tiny_tls_last_reason");
        const auto callback_base = find_export<PVOID(__cdecl*)()>(m, "tiny_tls_callback_base");
        const auto callback_reserved = find_export<PVOID(__cdecl*)()>(m, "tiny_tls_callback_reserved");
        const auto dllmain_count = find_export<int(__cdecl*)()>(m, "tiny_tls_dllmain_count");

        check(get() == 7, "__declspec(thread) reads the template value 7");
        set(42);
        check(get() == 42, "the per-thread copy is writable");

        check(callback_count() == 0, "load() does not run TLS callbacks");
        check(dllmain_count() == 0, "load() does not run DllMain");

        loader.run_tls_callbacks(m, DLL_PROCESS_ATTACH);
        check(callback_count() == 1, "run_tls_callbacks(DLL_PROCESS_ATTACH) calls the callback once");
        check(attach_count() == 1 && last_reason() == DLL_PROCESS_ATTACH,
              "the callback receives DLL_PROCESS_ATTACH");
        check(callback_base() == m.base, "the callback receives the mapped image base");
        check(callback_reserved() == nullptr, "the callback receives nullptr as the reserved argument");

        loader.run_tls_callbacks(m, DLL_PROCESS_DETACH);
        check(callback_count() == 2, "run_tls_callbacks(DLL_PROCESS_DETACH) calls the callback again");
        check(detach_count() == 1 && last_reason() == DLL_PROCESS_DETACH,
              "the callback receives DLL_PROCESS_DETACH");

        // A new thread starts with the helper DLL's block in the module slot. The runtime installs the mapped
        // module's template through the thread seam, then removes it before the thread exits.
        int worker_before = -1;
        int worker_after = -1;
        void* worker_previous = nullptr;
        std::thread worker([&] {
            worker_previous = module_tls_slots()[m.tls_index];
            loader.install_tls_for_thread(m);
            worker_before = get();
            set(99);
            worker_after = get();
            loader.uninstall_tls_for_thread(m);
        });
        worker.join();
        check(worker_previous != nullptr && worker_previous != m.tls_block,
              "a new thread starts with the helper DLL's block, not the mapped module's data");
        check(worker_before == 7, "the new thread sees the template value 7, not the main thread's 42");
        check(worker_after == 99, "the new thread's write is confined to its own block");
        check(get() == 42, "the main thread's TLS value is unchanged after the worker exit");

        const uint32_t first_index = m.tls_index;
        rt::Module& again = loader.load(dll);
        check(&again == &m, "load(dll) twice returns the same Module");
        check(again.tls_index == first_index && loader.modules().size() == 1,
              "duplicate load does not install a second TLS index");

        index = m.tls_index;
        block = m.tls_block;
        saved = m.tls_saved_block;
    }
    check(module_tls_slots()[index] == saved,
          "~Loader restores the helper DLL's block in the current thread");
    MEMORY_BASIC_INFORMATION info{};
    check(VirtualQuery(block, &info, sizeof(info)) == 0 || info.State == MEM_FREE, "~Loader frees the TLS block");
}

void test_second_tls_module_rejected(const fs::path& dll, const fs::path& temp) {
    std::printf("\n-- one mapped TLS module at a time --\n");
    const fs::path copy = temp / "tiny_tls_second.dll";
    // CopyFileW, not fs::copy_file: MSVC's copy_file calls CopyFile2, which Wine 9 doesn't implement.
    check(CopyFileW(dll.c_str(), copy.c_str(), FALSE) != 0, "copied the TLS fixture under a second name");

    {
        rt::Loader loader(dll.parent_path());
        rt::Module& first = loader.load(dll);
        check(first.tls_index != rt::kNoTlsIndex, "the first TLS module is initialized");

        bool rejected = false;
        std::string message;
        try {
            loader.load(copy);
        } catch (const std::runtime_error& e) {
            rejected = true;
            message = e.what();
        }
        check(rejected, "a second mapped module with TLS is rejected instead of sharing a slot");
        check(message.find("only one manually mapped module") != std::string::npos,
              "the rejection explains the one-module TLS limit");
    }

    // After the first loader restores its slot, the copy loads normally.
    {
        rt::Loader loader(copy.parent_path());
        rt::Module& m = loader.load(copy);
        check(m.tls_index != rt::kNoTlsIndex, "the copied TLS module loads after the first loader is destroyed");
        check(find_export<int(__cdecl*)()>(m, "tiny_tls_get")() == 7, "the copied module reads its template value");
    }
}

void print_result() {
    if (g_failures == 0)
        std::printf("\nloader_tls_test: OK (%d checks)\n", g_checks);
    else
        std::printf("\nloader_tls_test: FAILED (%d checks, %d failure%s)\n", g_checks, g_failures,
                    g_failures == 1 ? "" : "s");
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("loader_tls_test: static TLS against locally built fixtures\n");
    if (argc != 2) {
        std::fprintf(stderr, "usage: loader_tls_test <tiny_tls.dll>\n");
        return 2;
    }
    const fs::path tls_dll = argv[1];

    try {
        check(fs::exists(tls_dll), "tiny_tls fixture exists");
        if (!fs::exists(tls_dll)) {
            print_result();
            return 1;
        }

        TempDir temp;
        test_tls(tls_dll);
        test_second_tls_module_rejected(tls_dll, temp.path);
    } catch (const std::exception& e) {
        ++g_checks;
        ++g_failures;
        std::printf("FAIL unexpected exception: %s\n", e.what());
    }

    print_result();
    return g_failures ? 1 : 0;
}
