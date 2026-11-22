// Check the exact process exit code returned by gta6rt after the synthetic guest exits.
#include <windows.h>

#include <cstdio>
#include <string>

int wmain(int argc, wchar_t** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: tiny_game_exit_code_test <gta6rt.exe> <tiny_game.exe>\n");
        return 1;
    }

    std::wstring command = L"\"" + std::wstring(argv[1]) + L"\" --run \"" + argv[2] + L"\" -- hello 42 exit-code";
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION child{};
    if (!CreateProcessW(argv[1], command.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &startup, &child)) {
        std::fprintf(stderr, "could not start gta6rt (Win32 error %lu)\n", GetLastError());
        return 1;
    }
    CloseHandle(child.hThread);
    DWORD code = 0;
    const DWORD wait = WaitForSingleObject(child.hProcess, 30000);
    const bool exited = wait == WAIT_OBJECT_0 && GetExitCodeProcess(child.hProcess, &code);
    if (wait == WAIT_TIMEOUT) TerminateProcess(child.hProcess, 1);
    CloseHandle(child.hProcess);
    if (!exited) {
        std::fprintf(stderr, "could not collect gta6rt's exit code (wait result %#lx)\n", wait);
        return 1;
    }
    if (code != 42) {
        std::fprintf(stderr, "gta6rt exited with code %lu, expected guest exit code 42\n", code);
        return 1;
    }
    std::printf("gta6rt returned the expected guest exit code 42\n");
    return 0;
}
