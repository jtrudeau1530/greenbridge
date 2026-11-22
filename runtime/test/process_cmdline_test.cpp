// The process shim's command line round-trips: what GetCommandLineW says must parse with CommandLineToArgvW (the
// documented rules the CRT's argv parser shares) back to the __wargv the shim hands out, and __wargv must hold what
// main.cpp passed. No game files.
#include <windows.h>
#include <shellapi.h>

#include <cstdio>
#include <filesystem>
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

void round_trip(const std::vector<std::string>& args, const char* what) {
    rt::process::set_command_line("game.exe", args);
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
}  // namespace

int main() {
    round_trip({"plain", "with space"}, "plain and spaced arguments");
    round_trip({"C:\\My Games\\", "next"}, "a quoted argument ending in a backslash");
    round_trip({"a\\\"b"}, "a backslash before a quote");
    round_trip({"\\\\server\\share\\x y"}, "a UNC path with a space");

    // main.cpp passes game arguments as std::filesystem::path::string(): ANSI code page text, not UTF-8.
    const std::wstring cafe = L"caf\u00e9";
    round_trip({std::filesystem::path(cafe).string()}, "a non-ASCII argument");
    if (!wargv()[1] || cafe != wargv()[1]) {
        std::fprintf(stderr, "FAIL: __wargv[1] is not L\"caf\\u00e9\"\n");
        ++failures;
    }

    if (failures) return 1;
    std::printf("process_cmdline: GetCommandLineW parses back to __wargv, non-ASCII arguments intact\n");
    return 0;
}
