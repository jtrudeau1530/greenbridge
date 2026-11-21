// A stand-in game executable for `gta6rt --run`: what a real title does in its first second, run through the
// loader instead of the Windows one. Test code, not game content. Run as
//   gta6rt --run tiny_game.exe -- hello 42
// It checks its arguments, a static initializer, the heap, a file round trip, a second thread, and the
// xgameruntime.dll imports (answered by the shim registry: the stub DLL it links against would fail them), then
// calls exit() with 0 when everything held. Two checks are opt-in: thread_local needs static TLS (#93;
// LEONIDA_RUN_TLS=1), and a C++ exception (LEONIDA_RUN_EH=1) needs more than the unwind tables #176 registers on MSVC:
// _CxxThrowException finds the throwing image with RtlPcToFileHeader, which only knows modules the Windows loader
// mapped. With MinGW (libgcc's SEH unwinder) the #176 tables are enough and the check passes.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

extern "C" __declspec(dllimport) HRESULT __stdcall XGameRuntimeInitialize();
extern "C" __declspec(dllimport) void __stdcall XGameRuntimeUninitialize();
extern "C" __declspec(dllimport) uint32_t __stdcall XSystemGetDeviceType();

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("tiny_game: %s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

struct StaticInit {
    int value;
    StaticInit() : value(1234) {}
};
StaticInit g_static;

bool env_on(const char* name) {
    char buf[8] = {};
    return GetEnvironmentVariableA(name, buf, sizeof buf) > 0 && buf[0] == '1';
}

thread_local int t_counter = 5;

}  // namespace

int main(int argc, char** argv) {
    std::printf("tiny_game: started, argc=%d\n", argc);
    check(argc == 3 && std::strcmp(argv[1], "hello") == 0 && std::strcmp(argv[2], "42") == 0,
          "argv is the game's command line (\"hello\" \"42\"), not gta6rt's");
    check(std::strstr(GetCommandLineA(), "hello 42") != nullptr, "GetCommandLineA is the game's command line");
    check(g_static.value == 1234, "a static initializer ran before main");

    std::vector<int> v(1 << 20);
    for (size_t i = 0; i < v.size(); ++i) v[i] = int(i);
    check(v[123456] == 123456, "a 4 MiB heap allocation works");

    char tmp[MAX_PATH] = {};
    GetTempPathA(MAX_PATH, tmp);
    std::string path = std::string(tmp) + "leonida_tiny_game_" + std::to_string(GetCurrentProcessId()) + ".txt";
    bool file_ok = false;
    if (FILE* f = std::fopen(path.c_str(), "wb")) {
        std::fputs("leonida", f);
        std::fclose(f);
        char back[16] = {};
        if (FILE* r = std::fopen(path.c_str(), "rb")) {
            std::fread(back, 1, 7, r);
            std::fclose(r);
            file_ok = std::strcmp(back, "leonida") == 0;
        }
        std::remove(path.c_str());
    }
    check(file_ok, "a file round trip through the CRT works");

    long long sum = 0;
    std::thread worker([&] { for (int i = 1; i <= 1000; ++i) sum += i; });
    worker.join();
    check(sum == 500500, "a second thread ran and joined");

    check(XGameRuntimeInitialize() == S_OK, "XGameRuntimeInitialize is answered by the gta6rt shim (S_OK)");
    check(XSystemGetDeviceType() == 7, "XSystemGetDeviceType is answered by the gta6rt shim (Series X)");
    XGameRuntimeUninitialize();

    if (env_on("LEONIDA_RUN_TLS")) {
        int seen = -1;
        std::thread t([&] { seen = t_counter; t_counter = 9; });
        t.join();
        check(t_counter == 5 && seen == 5, "thread_local starts at its initializer on each thread");
    } else {
        std::printf("tiny_game: SKIP thread_local (needs static TLS, #93; LEONIDA_RUN_TLS=1)\n");
    }
    if (env_on("LEONIDA_RUN_EH")) {
        bool caught = false;
        try {
            throw std::string("boom");
        } catch (const std::string& e) {
            caught = e == "boom";
        }
        check(caught, "a C++ exception is thrown and caught inside the image");
    } else {
        std::printf("tiny_game: SKIP C++ exception (MSVC needs the image known to RtlPcToFileHeader; LEONIDA_RUN_EH=1)\n");
    }

    std::printf("tiny_game: %s\n", g_failures ? "FAILED" : "OK");
    std::fflush(stdout);
    std::exit(g_failures ? 1 : 0);  // through the shimmed exit(): ends the run, not gta6rt
}
