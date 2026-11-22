// The process shim's command line round-trips: what GetCommandLineW says must parse with CommandLineToArgvW (the
// documented rules the CRT's argv parser shares) back to the __wargv the shim hands out, and __wargv must hold what
// main.cpp passed, whatever the system code page (RT-84). The WinMain line (lpCmdLine) is the game's too, and the
// command line and ExitProcess are served under kernelbase.dll and the API-set names as well as kernel32.dll.
// No game files.
#include <windows.h>
#include <shellapi.h>

#include <cstdio>
#include <cstring>
#include <cwchar>
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

// An executable can import GetCommandLineA/W and ExitProcess from kernel32.dll, kernelbase.dll or the API set its
// umbrella library names, and the loader matches the DLL name exactly: every name must hand out the same shim.
void check_import_names() {
    for (const char* dll : {"kernelbase.dll", "api-ms-win-core-processenvironment-l1-1-0.dll"})
        for (const char* name : {"GetCommandLineA", "GetCommandLineW"}) {
            void* fn = rt::shims::find(dll, name);
            if (!fn || fn != rt::shims::find("kernel32.dll", name)) {
                std::fprintf(stderr, "FAIL %s!%s is not the kernel32.dll shim\n", dll, name);
                ++failures;
            }
        }
    for (const char* dll : {"kernelbase.dll", "api-ms-win-core-processthreads-l1-1-0.dll"}) {
        void* fn = rt::shims::find(dll, "ExitProcess");
        if (!fn || fn != rt::shims::find("kernel32.dll", "ExitProcess")) {
            std::fprintf(stderr, "FAIL %s!ExitProcess is not the kernel32.dll shim\n", dll);
            ++failures;
        }
    }
    for (const char* dll : {"api-ms-win-crt-runtime-l1-1-0.dll", "ucrtbase.dll", "ucrtbased.dll"})
        for (const char* name : {"_get_narrow_winmain_command_line", "_get_wide_winmain_command_line"})
            if (!rt::shims::find(dll, name)) {
                std::fprintf(stderr, "FAIL %s!%s is not shimmed\n", dll, name);
                ++failures;
            }
}

// lpCmdLine as the CRT's WinMain startup gets it from _get_wide/_get_narrow_winmain_command_line: the game's command
// line after the program name and the blanks that follow it (the narrow one in the ANSI code page).
void check_winmain(const std::wstring& program, const std::vector<std::wstring>& args, const wchar_t* want,
                   const char* what) {
    rt::process::set_command_line(program, args);
    using Wide = wchar_t*(__cdecl*)();
    using Narrow = char*(__cdecl*)();
    for (const char* dll : {"api-ms-win-crt-runtime-l1-1-0.dll", "ucrtbase.dll"}) {
        const auto wide_fn = reinterpret_cast<Wide>(rt::shims::find(dll, "_get_wide_winmain_command_line"));
        const auto narrow_fn = reinterpret_cast<Narrow>(rt::shims::find(dll, "_get_narrow_winmain_command_line"));
        if (!wide_fn || !narrow_fn) continue;  // check_import_names has reported the missing shim
        const wchar_t* wide = wide_fn();
        const char* line = narrow_fn();
        char expected[256] = {};
        WideCharToMultiByte(CP_ACP, 0, want, -1, expected, sizeof(expected), nullptr, nullptr);
        if (!wide || std::wcscmp(wide, want) != 0) {
            std::fwprintf(stderr, L"FAIL %hs (%hs): the wide WinMain line is [%ls], want [%ls]\n", what, dll,
                          wide ? wide : L"(null)", want);
            ++failures;
        }
        if (!line || std::strcmp(line, expected) != 0) {
            std::fprintf(stderr, "FAIL %s (%s): the narrow WinMain line is [%s], want [%s]\n", what, dll,
                         line ? line : "(null)", expected);
            ++failures;
        }
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

    check_import_names();
    check_winmain(L"game.exe", {L"-foo", L"with space"}, L"-foo \"with space\"", "arguments after a plain name");
    check_winmain(L"C:\\My Games\\game.exe", {L"-x"}, L"-x", "a quoted program name with a space");
    check_winmain(L"game.exe", {}, L"", "no arguments");
    // A byte above 0x7f in the narrow program name is part of the name, not the end of it.
    check_winmain(L"caf\u00e9.exe", {L"-x"}, L"-x", "a non-ASCII program name");

    if (failures) return 1;
    std::printf("process_cmdline: GetCommandLineW parses back to __wargv, non-ASCII arguments intact (code page %u); "
                "the WinMain line and the kernelbase/API-set names are the game's\n",
                GetACP());
    return 0;
}
