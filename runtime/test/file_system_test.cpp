// Checks the console path mapping the file shims install for a mapped title (RT-97).
//
// Every function is resolved through rt::shims::find under the DLL and name an import uses, so the test runs against
// the table the runtime installs rather than a private copy. It works on folders it creates in %TEMP%: a package
// folder that stands in for the game install, an overlay folder, and a "GAME:" path in each variant.
#include <windows.h>

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "file_system.h"
#include "shims.h"

namespace {

using CreateFileWFn = HANDLE(WINAPI*)(LPCWSTR, DWORD, DWORD, LPSECURITY_ATTRIBUTES, DWORD, DWORD, HANDLE);
using CreateFileAFn = HANDLE(WINAPI*)(LPCSTR, DWORD, DWORD, LPSECURITY_ATTRIBUTES, DWORD, DWORD, HANDLE);
using CreateFile2Fn = HANDLE(WINAPI*)(LPCWSTR, DWORD, DWORD, DWORD, LPCREATEFILE2_EXTENDED_PARAMETERS);
using FindFirstFileWFn = HANDLE(WINAPI*)(LPCWSTR, LPWIN32_FIND_DATAW);
using FindFirstFileAFn = HANDLE(WINAPI*)(LPCSTR, LPWIN32_FIND_DATAA);
using FindFirstFileExWFn =
    HANDLE(WINAPI*)(LPCWSTR, FINDEX_INFO_LEVELS, LPVOID, FINDEX_SEARCH_OPS, LPVOID, DWORD);
using FindFirstFileExAFn =
    HANDLE(WINAPI*)(LPCSTR, FINDEX_INFO_LEVELS, LPVOID, FINDEX_SEARCH_OPS, LPVOID, DWORD);
using FindNextFileWFn = BOOL(WINAPI*)(HANDLE, LPWIN32_FIND_DATAW);
using FindCloseFn = BOOL(WINAPI*)(HANDLE);
using GetFileAttributesWFn = DWORD(WINAPI*)(LPCWSTR);
using GetFileAttributesAFn = DWORD(WINAPI*)(LPCSTR);
using GetFileAttributesExWFn = BOOL(WINAPI*)(LPCWSTR, GET_FILEEX_INFO_LEVELS, LPVOID);
using DeleteFileWFn = BOOL(WINAPI*)(LPCWSTR);
using DeleteFileAFn = BOOL(WINAPI*)(LPCSTR);

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

template <typename Fn>
Fn resolve(const char* dll, const char* name) {
    return reinterpret_cast<Fn>(rt::shims::find(dll, name));
}

void write_path(const std::filesystem::path& path, const std::string& text) {
    std::filesystem::create_directories(path.parent_path());
    const HANDLE h =
        ::CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return;
    DWORD written = 0;
    ::WriteFile(h, text.data(), static_cast<DWORD>(text.size()), &written, nullptr);
    ::CloseHandle(h);
}

std::string read_handle(HANDLE h) {
    std::string out;
    char buffer[256];
    DWORD got = 0;
    while (::ReadFile(h, buffer, sizeof(buffer), &got, nullptr) && got != 0) out.append(buffer, got);
    return out;
}

std::string read_path(const std::filesystem::path& path) {
    const HANDLE h = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                   OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return {};
    std::string text = read_handle(h);
    ::CloseHandle(h);
    return text;
}

std::vector<std::wstring> list(FindFirstFileWFn first, FindNextFileWFn next, FindCloseFn close, const wchar_t* pattern) {
    std::vector<std::wstring> names;
    WIN32_FIND_DATAW found{};
    const HANDLE h = first(pattern, &found);
    if (h == INVALID_HANDLE_VALUE) return names;
    names.emplace_back(found.cFileName);
    while (next(h, &found)) names.emplace_back(found.cFileName);
    close(h);
    return names;
}

int count(const std::vector<std::wstring>& names, const wchar_t* name) {
    int n = 0;
    for (const auto& candidate : names)
        if (_wcsicmp(candidate.c_str(), name) == 0) ++n;
    return n;
}

bool contains(const std::vector<std::wstring>& names, const wchar_t* name) { return count(names, name) != 0; }

}  // namespace

int main() {
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / ("leonida_file_system_" + std::to_string(GetCurrentProcessId()));
    std::filesystem::remove_all(root);
    const std::filesystem::path package = root / "package";
    const std::filesystem::path overlay = root / "overlay";
    write_path(package / "data" / "hello.txt", "package");
    write_path(package / "data" / "dup.txt", "package");
    write_path(package / "data" / "bye.txt", "package");
    write_path(package / "data" / "sub" / "nested.txt", "nested");

    rt::fs::configure(package, overlay);
    check(rt::fs::game_dir() == package, "configure() sets the game folder");
    check(rt::fs::overlay_dir() == overlay, "configure() sets the overlay folder");

    // The table the runtime installs: kernel32 and the api sets a GDK title links against.
    const auto create_w = resolve<CreateFileWFn>("kernel32.dll", "CreateFileW");
    const auto create_a = resolve<CreateFileAFn>("kernelbase.dll", "CreateFileA");
    const auto first_w = resolve<FindFirstFileWFn>("api-ms-win-core-file-l1-1-0.dll", "FindFirstFileW");
    const auto first_a = resolve<FindFirstFileAFn>("kernel32.dll", "FindFirstFileA");
    const auto next_w = resolve<FindNextFileWFn>("kernel32.dll", "FindNextFileW");
    const auto find_close = resolve<FindCloseFn>("kernel32.dll", "FindClose");
    const auto attributes_w = resolve<GetFileAttributesWFn>("kernel32.dll", "GetFileAttributesW");
    const auto attributes_a = resolve<GetFileAttributesAFn>("kernel32.dll", "GetFileAttributesA");
    const auto attributes_ex_w = resolve<GetFileAttributesExWFn>("kernel32.dll", "GetFileAttributesExW");
    const auto delete_w = resolve<DeleteFileWFn>("kernel32.dll", "DeleteFileW");
    const auto delete_a = resolve<DeleteFileAFn>("api-ms-win-core-file-l1-1-0.dll", "DeleteFileA");
    check(create_w && create_a && first_w && first_a && next_w && find_close && attributes_w && attributes_a &&
              attributes_ex_w && delete_w && delete_a,
          "the file API resolves through the registry under kernel32/kernelbase and the api sets");

    // A read of a package file, through the mixed-case prefix and through forward slashes.
    {
        HANDLE h = create_w(L"GAME:\\data\\hello.txt", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        check(h != INVALID_HANDLE_VALUE, "CreateFileW opens GAME:\\data\\hello.txt from the package");
        check(read_handle(h) == "package", "the package file's bytes come back unchanged");
        if (h != INVALID_HANDLE_VALUE) ::CloseHandle(h);
        h = create_w(L"game:/data/hello.txt", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        check(h != INVALID_HANDLE_VALUE, "the prefix is case-insensitive and accepts forward slashes");
        if (h != INVALID_HANDLE_VALUE) ::CloseHandle(h);
        h = create_a("GAME:\\data\\hello.txt", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        check(h != INVALID_HANDLE_VALUE, "CreateFileA takes the same path");
        if (h != INVALID_HANDLE_VALUE) ::CloseHandle(h);
    }

    // Attributes, wide and narrow.
    check(attributes_w(L"GAME:\\data\\hello.txt") != INVALID_FILE_ATTRIBUTES, "GetFileAttributesW stats a package file");
    check(attributes_a("GAME:\\data\\hello.txt") != INVALID_FILE_ATTRIBUTES, "GetFileAttributesA stats a package file");
    const std::filesystem::path host_file = package / "data" / "hello.txt";
    check(attributes_w(host_file.c_str()) != INVALID_FILE_ATTRIBUTES, "a host path is passed through unchanged");

    // A write-open copies the package file into the overlay first, and the package is left alone.
    {
        const HANDLE h =
            create_w(L"GAME:\\data\\hello.txt", GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
        check(h != INVALID_HANDLE_VALUE, "a write-open of a package file succeeds");
        DWORD written = 0;
        if (h != INVALID_HANDLE_VALUE) {
            ::WriteFile(h, "overlay", 7, &written, nullptr);
            ::CloseHandle(h);
        }
        check(read_path(package / "data" / "hello.txt") == "package", "the package file is not modified");
        check(read_path(overlay / "data" / "hello.txt") == "overlay", "the write landed in the overlay copy");
        HANDLE r = create_w(L"GAME:\\data\\hello.txt", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        check(r != INVALID_HANDLE_VALUE && read_handle(r) == "overlay", "reads prefer the overlay copy");
        if (r != INVALID_HANDLE_VALUE) ::CloseHandle(r);
    }

    // A new file, in folders the package does not have.
    {
        const HANDLE h = create_w(L"GAME:\\save\\slot1.bin", GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, 0, nullptr);
        check(h != INVALID_HANDLE_VALUE, "CREATE_ALWAYS makes a new console file");
        DWORD written = 0;
        if (h != INVALID_HANDLE_VALUE) {
            ::WriteFile(h, "12345", 5, &written, nullptr);
            ::CloseHandle(h);
        }
        check(read_path(overlay / "save" / "slot1.bin") == "12345", "the new file is in the overlay");
        check(!std::filesystem::exists(package / "save"), "nothing was created in the package");

        WIN32_FILE_ATTRIBUTE_DATA data{};
        check(attributes_ex_w(L"GAME:\\save\\slot1.bin", GetFileExInfoStandard, &data) != FALSE &&
                  data.nFileSizeLow == 5,
              "GetFileAttributesExW reports the overlay copy's size");
    }

    // One listing, merged, overlay first and without duplicates.
    {
        const std::vector<std::wstring> names = list(first_w, next_w, find_close, L"GAME:\\data\\*");
        check(contains(names, L"hello.txt") && count(names, L"hello.txt") == 1, "the merged listing has hello.txt once");
        check(contains(names, L"dup.txt"), "the merged listing has the package's own files");
        check(contains(names, L"sub"), "the merged listing has the package's folders");
        check(!contains(names, L".gta6rt-deleted"), "the overlay's marker folder is not listed");

        const std::vector<std::wstring> root_names = list(first_w, next_w, find_close, L"GAME:\\");
        check(contains(root_names, L"data") && contains(root_names, L"save"),
              "GAME:\\ lists the package's folder and the overlay's");

        WIN32_FIND_DATAA ansi{};
        const HANDLE h = first_a("GAME:\\data\\*", &ansi);
        check(h != INVALID_HANDLE_VALUE && ansi.cFileName[0] != '\0', "FindFirstFileA fills an ANSI listing too");
        if (h != INVALID_HANDLE_VALUE) find_close(h);

        SetLastError(0);
        WIN32_FIND_DATAW none{};
        const HANDLE missing = first_w(L"GAME:\\missing\\*", &none);
        check(missing == INVALID_HANDLE_VALUE && GetLastError() == ERROR_FILE_NOT_FOUND,
              "a pattern with no match reports ERROR_FILE_NOT_FOUND");
    }

    // A delete hides the package file without removing it.
    {
        check(delete_w(L"GAME:\\data\\dup.txt") != FALSE, "DeleteFileW accepts a package file");
        check(std::filesystem::exists(package / "data" / "dup.txt"), "the package file is still there");
        SetLastError(0);
        check(attributes_w(L"GAME:\\data\\dup.txt") == INVALID_FILE_ATTRIBUTES &&
                  GetLastError() == ERROR_FILE_NOT_FOUND,
              "the deleted name no longer resolves");
        SetLastError(0);
        const HANDLE h =
            create_w(L"GAME:\\data\\dup.txt", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        check(h == INVALID_HANDLE_VALUE && GetLastError() == ERROR_FILE_NOT_FOUND, "and cannot be opened");
        check(!contains(list(first_w, next_w, find_close, L"GAME:\\data\\*"), L"dup.txt"),
              "and is absent from the listing");

        // A write-open of the hidden name fails, and must leave the name deleted: no copy-up from the package, no
        // overlay file, and still hidden afterwards.
        SetLastError(0);
        const HANDLE write_open =
            create_w(L"GAME:\\data\\dup.txt", GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
        check(write_open == INVALID_HANDLE_VALUE && GetLastError() == ERROR_FILE_NOT_FOUND,
              "a write-open of the deleted name fails with ERROR_FILE_NOT_FOUND");
        if (write_open != INVALID_HANDLE_VALUE) ::CloseHandle(write_open);
        check(!std::filesystem::exists(overlay / "data" / "dup.txt"), "the failed write-open left no overlay copy");
        SetLastError(0);
        check(attributes_w(L"GAME:\\data\\dup.txt") == INVALID_FILE_ATTRIBUTES &&
                  GetLastError() == ERROR_FILE_NOT_FOUND,
              "and the name is still deleted");

        // Re-creating it makes it readable again, and it is listed.
        const HANDLE again = create_w(L"GAME:\\data\\dup.txt", GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, 0, nullptr);
        DWORD written = 0;
        check(again != INVALID_HANDLE_VALUE, "the name can be created again");
        if (again != INVALID_HANDLE_VALUE) {
            ::WriteFile(again, "again", 5, &written, nullptr);
            ::CloseHandle(again);
        }
        check(read_path(overlay / "data" / "dup.txt") == "again", "the new copy is in the overlay");
        check(contains(list(first_w, next_w, find_close, L"GAME:\\data\\*"), L"dup.txt"),
              "and it is listed again");

        check(delete_a("GAME:\\data\\bye.txt") != FALSE, "DeleteFileA takes a console path");
        check(std::filesystem::exists(package / "data" / "bye.txt"), "the narrow delete leaves the package file too");
        SetLastError(0);
        check(delete_w(L"GAME:\\data\\bye.txt") == FALSE && GetLastError() == ERROR_FILE_NOT_FOUND,
              "deleting an already hidden file reports ERROR_FILE_NOT_FOUND");
        SetLastError(0);
        check(delete_w(L"GAME:\\nope\\missing.txt") == FALSE && GetLastError() == ERROR_FILE_NOT_FOUND,
              "deleting a file that never existed reports ERROR_FILE_NOT_FOUND");

        // Deleting a nested package file creates parent directories inside .gta6rt-deleted/; those marker parent
        // directories must not hide the corresponding ancestor directories in the package.
        check(delete_w(L"GAME:\\data\\sub\\nested.txt") != FALSE, "DeleteFileW accepts a nested package file");
        const DWORD sub_attrs = attributes_w(L"GAME:\\data\\sub");
        check(sub_attrs != INVALID_FILE_ATTRIBUTES && (sub_attrs & FILE_ATTRIBUTE_DIRECTORY) != 0,
              "deleting a nested package file does not hide its parent folder");
        check(contains(list(first_w, next_w, find_close, L"GAME:\\data\\*"), L"sub"),
              "the parent folder stays listed after deleting a nested package file");
    }

    // Nothing leaves the package and the overlay.
    {
        SetLastError(0);
        const HANDLE h = create_w(L"GAME:\\..\\escape.txt", GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, 0, nullptr);
        check(h == INVALID_HANDLE_VALUE && GetLastError() == ERROR_INVALID_NAME, "a path above GAME: is refused");
        if (h != INVALID_HANDLE_VALUE) ::CloseHandle(h);
        check(!std::filesystem::exists(root / "escape.txt"), "and nothing was created outside");
        SetLastError(0);
        check(attributes_w(L"GAME:\\..\\x") == INVALID_FILE_ATTRIBUTES && GetLastError() == ERROR_INVALID_NAME,
              "a stat above GAME: is refused");
        SetLastError(0);
        check(create_w(L"GAME:\\C:\\x", GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, 0, nullptr) == INVALID_HANDLE_VALUE &&
                  GetLastError() == ERROR_INVALID_NAME,
              "a drive inside GAME: is refused");
        SetLastError(0);
        WIN32_FIND_DATAW found{};
        check(first_w(L"GAME:\\..\\*", &found) == INVALID_HANDLE_VALUE && GetLastError() == ERROR_INVALID_NAME,
              "a listing above GAME: is refused");
        SetLastError(0);
        check(first_w(L"GAME:\\data\\*", nullptr) == INVALID_HANDLE_VALUE &&
                  GetLastError() == ERROR_INVALID_PARAMETER,
              "a null listing buffer is refused");
    }

    // CreateFile2 and FindFirstFileExA/W (used by GDK/OneCore titles and std::filesystem).
    {
        const auto create2 = resolve<CreateFile2Fn>("api-ms-win-core-file-l1-2-0.dll", "CreateFile2");
        const auto first_ex_w = resolve<FindFirstFileExWFn>("api-ms-win-core-file-l1-1-0.dll", "FindFirstFileExW");
        const auto first_ex_a = resolve<FindFirstFileExAFn>("kernelbase.dll", "FindFirstFileExA");
        check(create2 && first_ex_w && first_ex_a, "CreateFile2 and FindFirstFileExA/W are registered");

        // CreateFile2 reads an untouched package file without modifying it.
        write_path(package / "data" / "cf2_pkg.txt", "pkg-cf2");
        CREATEFILE2_EXTENDED_PARAMETERS ext{};
        ext.dwSize = sizeof(ext);
        ext.dwFileAttributes = FILE_ATTRIBUTE_NORMAL;
        const HANDLE h_read = create2(L"GAME:\\data\\cf2_pkg.txt", GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, &ext);
        check(h_read != INVALID_HANDLE_VALUE && read_handle(h_read) == "pkg-cf2",
              "CreateFile2 reads a GAME: file from the package");
        if (h_read != INVALID_HANDLE_VALUE) ::CloseHandle(h_read);

        // CreateFile2 write goes to the overlay, leaving the package untouched.
        const HANDLE h_write = create2(L"GAME:\\data\\cf2.txt", GENERIC_WRITE, 0, CREATE_ALWAYS, &ext);
        check(h_write != INVALID_HANDLE_VALUE, "CreateFile2 creates a new GAME: file in the overlay");
        if (h_write != INVALID_HANDLE_VALUE) {
            DWORD written = 0;
            ::WriteFile(h_write, "cf2", 3, &written, nullptr);
            ::CloseHandle(h_write);
        }
        check(read_path(overlay / "data" / "cf2.txt") == "cf2" && !std::filesystem::exists(package / "data" / "cf2.txt"),
              "CreateFile2 writes land in the overlay and never in the package");

        // CreateFile2 refuses path traversal.
        SetLastError(0);
        const HANDLE h_esc = create2(L"GAME:\\..\\cf2_escape.txt", GENERIC_WRITE, 0, CREATE_ALWAYS, &ext);
        check(h_esc == INVALID_HANDLE_VALUE && GetLastError() == ERROR_INVALID_NAME,
              "CreateFile2 refuses a path above GAME:");
        if (h_esc != INVALID_HANDLE_VALUE) ::CloseHandle(h_esc);

        // FindFirstFileExW merges package + overlay and honours FindExInfoBasic & FindExSearchLimitToDirectories.
        WIN32_FIND_DATAW fd{};
        std::vector<std::wstring> basic_names;
        bool alt_empty = true;
        HANDLE hf = first_ex_w(L"GAME:\\data\\*", FindExInfoBasic, &fd, FindExSearchNameMatch, nullptr, 0);
        check(hf != INVALID_HANDLE_VALUE, "FindFirstFileExW opens a GAME: directory listing");
        if (hf != INVALID_HANDLE_VALUE) {
            do {
                basic_names.emplace_back(fd.cFileName);
                if (fd.cAlternateFileName[0] != L'\0') alt_empty = false;
            } while (next_w(hf, &fd));
            find_close(hf);
        }
        check(contains(basic_names, L"hello.txt") && contains(basic_names, L"cf2.txt") && !contains(basic_names, L"bye.txt"),
              "FindFirstFileExW merges overlay + package and hides deleted files");
        check(alt_empty, "FindFirstFileExW(FindExInfoBasic) leaves cAlternateFileName empty");

        std::vector<std::wstring> dir_names;
        HANDLE hdir = first_ex_w(L"GAME:\\data\\*", FindExInfoStandard, &fd, FindExSearchLimitToDirectories, nullptr, 0);
        check(hdir != INVALID_HANDLE_VALUE, "FindFirstFileExW(FindExSearchLimitToDirectories) succeeds");
        if (hdir != INVALID_HANDLE_VALUE) {
            do {
                dir_names.emplace_back(fd.cFileName);
            } while (next_w(hdir, &fd));
            find_close(hdir);
        }
        check(contains(dir_names, L"sub") && !contains(dir_names, L"hello.txt"),
              "FindFirstFileExW(FindExSearchLimitToDirectories) returns only directories");

        // FindFirstFileExA works on GAME: paths too.
        WIN32_FIND_DATAA fda{};
        HANDLE hfa = first_ex_a("GAME:\\data\\cf2.txt", FindExInfoBasic, &fda, FindExSearchNameMatch, nullptr, 0);
        check(hfa != INVALID_HANDLE_VALUE && std::string(fda.cFileName) == "cf2.txt",
              "FindFirstFileExA finds an overlay file under GAME:");
        if (hfa != INVALID_HANDLE_VALUE) find_close(hfa);
    }

    std::filesystem::remove_all(root);
    std::printf("%d check(s), %d failure(s)\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
