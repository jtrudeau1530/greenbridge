// The process shim's command line round-trips: what GetCommandLineW says must parse with CommandLineToArgvW (the
// documented rules the CRT's argv parser shares) back to the __wargv the shim hands out, and __wargv must hold what
// main.cpp passed, whatever the system code page (RT-84). No game files.
#include <windows.h>
#include <shellapi.h>

#include <cstdio>
#include <string>
#include <vector>

#include "process_shim.h"
#include "shims.h"

namespace {
int failures = 0;

wchar_t** wargv() {
    using Fn = wchar_t*** (__cdecl*)();
    return *reinterpret_cast<Fn>(rt::shims::find("ucrtbase.dll", "__p___wargv"))();
}

char** argv() {
    using Fn = char*** (__cdecl*)();
    return *reinterpret_cast<Fn>(rt::shims::find("ucrtbase.dll", "__p___argv"))();
}

void round_trip(const std::vector<std::wstring>& args, const char* what) {
    rt::process::set_command_line(L"game.exe", args);
    using Fn = LPWSTR(WINAPI*)();
    const LPWSTR line = reinterpret_cast<Fn>(rt::shims::find("kernel32.dll", "GetCommandLineW"))();
    int count = 0;
    LPWSTR* parsed = CommandLineToArgvW(line, &count);
    bool ok = parsed && count == int(args.size()) + 1;
    for (int i = 0; ok && i < count; ++i) ok = wargv()[i] && wcscmp(parsed[i], wargv()[i]) == 0;
    if (!ok) {
        std::fwprintf(stderr, L"FAIL %hs: GetCommandLineW [%ls] parses to %d argument(s)\n", what, line, count);
        ++failures;
    }
    LocalFree(parsed);
}

// __wargv[1] must be `want` exactly, and the narrow argv[1] its ANSI code page text (the CRT's replacement for a
// character the code page lacks), not a crash.
void check_non_ascii(const std::wstring& want, const char* what) {
    round_trip({want}, what);
    if (!wargv()[1] || want != wargv()[1]) {
        std::fprintf(stderr, "FAIL %s: __wargv[1] is not the argument passed\n", what);
        ++failures;
    }
    char expected[64] = {};
    WideCharToMultiByte(CP_ACP, 0, want.c_str(), -1, expected, sizeof(expected), nullptr, nullptr);
    if (!argv()[1] || std::string(expected) != argv()[1]) {
        std::fprintf(stderr, "FAIL %s: argv[1] is not the code page %u text of the argument\n", what, GetACP());
        ++failures;
    }
}
}  // namespace

int main() {
    round_trip({L"plain", L"with space"}, "plain and spaced arguments");
    round_trip({L"C:\\My Games\\", L"next"}, "a quoted argument ending in a backslash");
    round_trip({L"a\\\"b"}, "a backslash before a quote");
    round_trip({L"\\\\server\\share\\x y"}, "a UNC path with a space");

    // In code page 1252 but not 1253 (Greek), where path::string() used to throw (RT-84).
    check_non_ascii(L"caf\u00e9", "a Latin-1 argument");
    // Greek, Cyrillic and CJK together: no single-byte code page holds all of them, so this covers every PC.
    check_non_ascii(L"\u0394\u0416\u6F22 x", "an argument outside any single-byte code page");

    if (failures) return 1;
    std::printf("process_cmdline: GetCommandLineW parses back to __wargv, non-ASCII arguments intact (code page %u)\n",
                GetACP());
    return 0;
}
