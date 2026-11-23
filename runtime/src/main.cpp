// gta6rt: native runtime for the Xbox Series build.
//
//   gta6rt --dry-run <image> [--game-dir DIR] [--overlay DIR]
//       Maps the image and its bundled modules, applies relocations, binds every import, installs static TLS for
//       this thread and prints a report. No code from the mapped images
//       runs. TLS needs the runtime's own gta6rt_tls_host.dll, which reserves the module TLS index; system DLLs are
//       loaded to resolve host imports, as before.
//
//   gta6rt --init <image> [--game-dir DIR] [--overlay DIR]
//       The same mapping, then each loaded DLL's TLS callbacks and DllMain(DLL_PROCESS_ATTACH) are run.
//
//   gta6rt --run <exe> [--game-dir DIR] [--overlay DIR] [--trace-d3d12] [--trace FILE] [-- game arguments...]
//       The same, then the EXE's own TLS callbacks and its entry point, on a game thread. The game's argv and
//       command line are <exe name> plus the arguments after "--"; exit/ExitProcess from the game end the run
//       (shims/process.cpp). gta6rt prints the binding report first and returns the game's exit code.
//
// --game-dir is also where a title's GAME: paths read from; anything it writes or deletes goes to the overlay
// (--overlay, default <game-dir>/overlay) so the package is never modified (shims/file_system.cpp).
#include <windows.h>

#include <cstdio>
#include <filesystem>
#include <map>
#include <string>
#include <system_error>
#include <vector>

#include "base_reservation.h"
#include "mods.h"
#include "pe_loader.h"
#include "../shims/d3d12_trace.h"
#include "../shims/file_system.h"
#include "../shims/process_shim.h"
#include "../shims/shims.h"
#include "../shims/startup_trace.h"

namespace {

const char* binding_name(rt::Binding b) {
    switch (b) {
        case rt::Binding::Shim: return "shim";
        case rt::Binding::Host: return "host";
        case rt::Binding::Module: return "module";
        default: return "UNIMPLEMENTED";
    }
}

int usage() {
    std::fprintf(stderr,
                 "usage: gta6rt --dry-run <image> [--game-dir DIR] [--overlay DIR] [--mods DIR] [--trace FILE]\n"
                 "       gta6rt --init <image> [--game-dir DIR] [--overlay DIR] [--mods DIR] [--trace-d3d12] [--trace FILE]\n"
                 "       gta6rt --run <exe> [--game-dir DIR] [--overlay DIR] [--mods DIR] [--trace-d3d12] [--trace FILE] [-- game arguments...]\n");
    return 2;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
    std::filesystem::path image, game_dir, overlay_dir, mods_dir;
    bool dry_run = false, init = false, run = false, trace_d3d12 = false;
    std::vector<std::wstring> game_args;  // wide: path::string() throws on a character the code page lacks (RT-84)
    std::wstring trace_file;
    for (int i = 1; i < argc; ++i) {
        std::wstring a = argv[i];
        if (a == L"--") {
            for (++i; i < argc; ++i) game_args.push_back(argv[i]);
            break;
        }
        if (a == L"--dry-run") dry_run = true;
        else if (a == L"--init") init = true;
        else if (a == L"--run") run = true;
        else if (a == L"--trace-d3d12") trace_d3d12 = true;
        else if (a == L"--trace" && i + 1 < argc) trace_file = argv[++i];
        else if (a == L"--game-dir" && i + 1 < argc) game_dir = argv[++i];
        else if (a == L"--overlay" && i + 1 < argc) overlay_dir = argv[++i];
        else if (a == L"--mods" && i + 1 < argc) mods_dir = argv[++i];
        else if (image.empty()) image = a;
        else return usage();
    }
    if (image.empty()) return usage();
    // Exactly one mode: --dry-run --run would otherwise initialise the modules and call the entry point.
    if (dry_run + init + run != 1) return usage();
    // The console folders are fixed once, as absolute paths. A bare image name (`gta6rt --run game.exe` from the
    // game folder) has an empty parent path, which left GAME: with no root; and a relative --game-dir or --overlay
    // would follow the title's later SetCurrentDirectory. If absolute() fails the path stays as given.
    const auto absolute_or_same = [](const std::filesystem::path& p) {
        std::error_code ec;
        std::filesystem::path a = std::filesystem::absolute(p, ec);
        return ec ? p : a;
    };
    game_dir = game_dir.empty() ? absolute_or_same(image).parent_path() : absolute_or_same(game_dir);
    if (!overlay_dir.empty()) overlay_dir = absolute_or_same(overlay_dir);
    // An image without relocations whose range this process already uses (arm64 Wine on macOS) runs in a new
    // process that has the range reserved from the start (base_reservation.h); that process does not do this again.
    if (!rt::base_reservation::inherited())
        if (const auto range = rt::base_reservation::needed(image)) return rt::base_reservation::relaunch(*range);
    // GAME: paths read under --game-dir; whatever a title writes lands in the overlay, never in the package.
    if (overlay_dir.empty()) overlay_dir = game_dir / "overlay";
    rt::fs::configure(game_dir, overlay_dir);
    // Mods (docs/modding.md): <folder of gta6rt.exe>\mods unless --mods names another. Loaded and reported here;
    // the archetype swaps take effect once the game's archetype lookup is hooked for this executable version.
    if (mods_dir.empty()) {
        std::wstring self(32768, L'\0');
        const DWORD n = GetModuleFileNameW(nullptr, self.data(), static_cast<DWORD>(self.size()));
        if (n && n < self.size()) mods_dir = std::filesystem::path(self.substr(0, n)).parent_path() / L"mods";
    }
    {
        const leonida::mods::LoadResult mods = leonida::mods::load_mods(mods_dir);
        for (const std::string& e : mods.errors) std::fprintf(stderr, "[rt] mods: skipped %s\n", e.c_str());
        for (const auto& m : mods.mods) {
            std::fprintf(stderr, "[rt] mods: %s%s\n", m.name.c_str(), m.enabled ? "" : " (disabled)");
            for (const auto& sw : m.archetype_swaps)
                std::fprintf(stderr, "[rt] mods:   archetype %s (0x%08X) -> %s (0x%08X)\n", sw.from_name.c_str(),
                             sw.from, sw.to_name.c_str(), sw.to);
        }
        leonida::mods::set_active(leonida::mods::ArchetypeSwaps(mods.mods));
        if (leonida::mods::active().size())
            std::fprintf(stderr, "[rt] mods: %zu archetype swap(s) loaded; not applied yet: the archetype lookup hook "
                                 "for this executable is not known (docs/modding.md)\n", leonida::mods::active().size());
    }
    rt::d3d12_trace::set_enabled(trace_d3d12);
    if (int trace_error = 0; !trace_file.empty() && !rt::startup_trace::open(trace_file.c_str(), &trace_error)) {
        std::fprintf(stderr, trace_error == EEXIST ? "error: the trace file already exists; --trace only creates a new file\n"
                                                   : "error: cannot create the trace file\n");
        return 2;
    }

    try {
        rt::Loader loader(game_dir);
        rt::Module& main_module = loader.load(image);  // maps, binds and installs TLS; runs no image code
        if (run) rt::process::set_command_line(image.filename().wstring(), game_args);
        if (init || run) {
            loader.initialise();  // TLS callbacks, then DllMain(DLL_PROCESS_ATTACH) for DLLs
            std::printf("%zu module(s) initialised\n", loader.modules().size());
        }
        int unimplemented = 0, unimplemented_hresult = 0;
        for (auto* m : loader.modules()) {
            std::map<rt::Binding, int> counts;
            for (auto& r : m->imports) counts[r.binding]++;
            std::printf("%-36s base=%p%s size=%u entry=+%#x  shim=%d host=%d module=%d unimplemented=%d\n",
                        m->name.c_str(), m->base, m->relocated ? " (relocated)" : "", m->size, m->entry_rva,
                        counts[rt::Binding::Shim], counts[rt::Binding::Host], counts[rt::Binding::Module],
                        counts[rt::Binding::Unimplemented]);
            for (auto& r : m->imports)
                if (r.binding == rt::Binding::Unimplemented || r.binding == rt::Binding::Shim) {
                    const bool missing = r.binding == rt::Binding::Unimplemented;
                    const bool hresult =
                        missing && rt::shims::classify_import(r.dll, r.name) == rt::shims::ReturnClass::HResult;
                    std::printf("    %-13s %s!%s%s\n", binding_name(r.binding), r.dll.c_str(), r.name.c_str(),
                                hresult ? "  [returns HRESULT]" : "");
                    unimplemented += missing;
                    unimplemented_hresult += hresult;
                }
        }
        std::printf("%zu module(s) mapped, %d unimplemented import(s)\n", loader.modules().size(), unimplemented);
        std::printf("unimplemented by return policy: hresult=%d other=%d\n", unimplemented_hresult,
                    unimplemented - unimplemented_hresult);
        if (!run) {
            rt::startup_trace::event("exit", "\"code\":0");
            return 0;
        }
        if (!main_module.entry_rva) {
            std::fprintf(stderr, "error: %s has no entry point\n", main_module.name.c_str());
            return 1;
        }
        const std::wstring paths = L"console paths: GAME: -> " + game_dir.native() + L", writes -> " + overlay_dir.native();
        const HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD mode;
        bool printed = false;
        if (GetConsoleMode(out, &mode)) {
            std::fflush(stdout);
            const std::wstring line = paths + L"\r\n";
            DWORD written;
            printed = WriteConsoleW(out, line.data(), static_cast<DWORD>(line.size()), &written, nullptr) != FALSE;
        }
        if (!printed) {
            const auto game = game_dir.u8string();
            const auto overlay = overlay_dir.u8string();
            std::printf("console paths: GAME: -> %s, writes -> %s\n",
                        reinterpret_cast<const char*>(game.c_str()), reinterpret_cast<const char*>(overlay.c_str()));
        }
        std::printf("running %s\n", main_module.name.c_str());
        std::fflush(stdout);
        const int code = rt::process::run_entry(main_module.base + main_module.entry_rva);
        std::fflush(stdout);
        std::printf("game exited with code %d\n", code);
        rt::shims::report_called_unimplemented();
        rt::startup_trace::event("exit", "\"code\":" + std::to_string(code));
        std::fflush(stdout);
        std::fflush(stderr);
        // Not `return`: a normal host exit runs the CRT's atexit table, which the game shares with gta6rt (its
        // handlers would run against a game that has stopped) and whose last word decides the exit status.
        // ExitProcess also skips atexit, so the TLS clone cleanup registered there is run here instead.
        rt::cleanup_tls_clones();
        ExitProcess(static_cast<UINT>(code));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        rt::startup_trace::event("error", "\"what\":" + rt::startup_trace::quote(e.what()));
        return 1;
    }
}
