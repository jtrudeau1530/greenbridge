// file_system: the file API of a mapped title, seen through the console's drives.
//
// A GDK title names its files on the console's drives, "GAME:\\data\\x.sps" being one in the installed game folder, so
// the imports it makes of CreateFile*, FindFirstFile*/FindNextFile*/FindClose, GetFileAttributes*/GetFileAttributesEx*
// and DeleteFile* are answered here (docs/runtime_imports.md, "Console paths and the write overlay"; the prefixes come
// from ivall/xcl, GPL-2.0-only, as facts). The rules:
//   - "GAME:" (any case, "\\" or "/") maps under --game-dir; any other path is a host path and goes to the host;
//   - an open that can write, and DeleteFile, go to the overlay (--overlay, default <game-dir>\\overlay), so the
//     package stays exactly as installed. A write-open of a package file copies it into the overlay first, so
//     OPEN_EXISTING keeps its bytes; a delete of a package file leaves an empty marker under
//     <overlay>\\.gta6rt-deleted, and reads and listings treat the name as gone. The marker is never removed and a
//     deleted name is never copied up: a re-created file is its overlay copy, which reads and listings already
//     prefer, while an open of a deleted name that fails leaves the name deleted;
//   - reads prefer the overlay copy and then the package;
//   - FindFirstFile merges the two folders, the overlay winning on a name, and hides deleted names and the marker
//     folder;
//   - a path that climbs above the root with ".." or names another drive is refused with ERROR_INVALID_NAME, so a
//     title cannot reach outside the package and the overlay.
// Only images mapped by the loader see these: they are bound through the registry, the runtime's own code calls the
// host directly.
#include <windows.h>

#include <cstring>
#include <cwchar>
#include <cwctype>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "file_system.h"
#include "shims.h"

namespace rt::fs {
namespace {

constexpr wchar_t kGamePrefix[] = L"game:";
constexpr size_t kGamePrefixLength = 5;
constexpr wchar_t kDeletedDir[] = L".gta6rt-deleted";

std::mutex g_mutex;
std::filesystem::path g_game_dir;
std::filesystem::path g_overlay_dir;

struct Roots {
    std::filesystem::path game, overlay;
};

Roots roots() {
    std::lock_guard lock(g_mutex);
    return {g_game_dir, g_overlay_dir};
}

std::wstring lower(std::wstring s) {
    for (auto& c : s) c = static_cast<wchar_t>(std::towlower(c));
    return s;
}

bool same_name(const wchar_t* a, const wchar_t* b) { return _wcsicmp(a, b) == 0; }

// The narrow API (CreateFileA and friends) takes ANSI code page text, as the CRT does; the loader's own paths are
// handled as wide strings everywhere else.
std::wstring widen_ansi(const char* s) {
    if (!s) return {};
    const int n = MultiByteToWideChar(CP_ACP, 0, s, -1, nullptr, 0);
    if (n <= 1) return {};
    std::wstring w(static_cast<size_t>(n - 1), L'\0');
    MultiByteToWideChar(CP_ACP, 0, s, -1, w.data(), n);
    return w;
}

void narrow_ansi(const std::wstring& wide, char* out, size_t capacity) {
    if (capacity == 0) return;
    const int n = WideCharToMultiByte(CP_ACP, 0, wide.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1 || static_cast<size_t>(n) > capacity) {
        out[0] = '\0';
        return;
    }
    WideCharToMultiByte(CP_ACP, 0, wide.c_str(), -1, out, static_cast<int>(capacity), nullptr, nullptr);
}

enum class Parsed { Host, Console, Refused };

// Splits a console path into the components below the root. A component that names a drive or a stream ("C:", "x:y")
// and a ".." that would leave the root are refused: the package and the overlay are the only folders a title may name.
Parsed parse_console(const std::wstring& path, std::vector<std::wstring>& parts) {
    if (path.size() < kGamePrefixLength || lower(path.substr(0, kGamePrefixLength)) != kGamePrefix) return Parsed::Host;
    parts.clear();
    std::wstring part;
    auto flush = [&]() -> bool {
        if (part.empty() || part == L".") {
            part.clear();
            return true;
        }
        if (part == L"..") {
            if (parts.empty()) return false;
            parts.pop_back();
        } else if (part.find(L':') != std::wstring::npos) {
            return false;
        } else {
            parts.push_back(part);
        }
        part.clear();
        return true;
    };
    for (size_t i = kGamePrefixLength; i < path.size(); ++i) {
        const wchar_t c = path[i];
        if (c == L'\\' || c == L'/') {
            if (!flush()) return Parsed::Refused;
        } else {
            part.push_back(c);
        }
    }
    return flush() ? Parsed::Console : Parsed::Refused;
}

std::filesystem::path join(const std::filesystem::path& root, const std::vector<std::wstring>& parts) {
    std::filesystem::path p = root;
    for (const auto& part : parts) p /= part;
    return p;
}

struct Target {
    std::filesystem::path path;
    DWORD error = 0;  // non-zero: the caller fails with this error and does not touch the host
};

Target fail(DWORD error) {
    Target t;
    t.error = error;
    return t;
}

// The overlay keeps one empty marker per deleted package path, outside the mirrored tree so it cannot collide with a
// file the title creates.
std::filesystem::path marker(const std::filesystem::path& overlay, const std::vector<std::wstring>& parts) {
    return join(overlay / kDeletedDir, parts);
}

bool deleted(const std::filesystem::path& overlay, const std::vector<std::wstring>& parts) {
    if (overlay.empty() || parts.empty()) return false;
    std::error_code ec;
    // Only regular files are deletion markers; parent directories inside .gta6rt-deleted/ exist only to hold markers
    // for nested deleted files and must not hide the corresponding ancestor directories in the package.
    return std::filesystem::is_regular_file(marker(overlay, parts), ec);
}

Target read_target(const std::wstring& path) {
    std::vector<std::wstring> parts;
    const Parsed parsed = parse_console(path, parts);
    if (parsed == Parsed::Host) {
        Target t;
        t.path = std::filesystem::path(path);
        return t;
    }
    if (parsed == Parsed::Refused) return fail(ERROR_INVALID_NAME);

    const Roots r = roots();
    if (r.game.empty()) return fail(ERROR_PATH_NOT_FOUND);

    std::error_code ec;
    if (!r.overlay.empty()) {
        const std::filesystem::path overlay_path = join(r.overlay, parts);
        if (std::filesystem::exists(overlay_path, ec)) {
            Target t;
            t.path = overlay_path;
            return t;
        }
    }
    // A package file the title deleted is gone even though the package still holds it.
    if (deleted(r.overlay, parts)) return fail(ERROR_FILE_NOT_FOUND);

    Target t;
    t.path = join(r.game, parts);
    return t;
}

Target write_target(const std::wstring& path, bool copy_up) {
    std::vector<std::wstring> parts;
    const Parsed parsed = parse_console(path, parts);
    if (parsed == Parsed::Host) {
        Target t;
        t.path = std::filesystem::path(path);
        return t;
    }
    if (parsed == Parsed::Refused) return fail(ERROR_INVALID_NAME);

    const Roots r = roots();
    if (r.overlay.empty()) return fail(ERROR_PATH_NOT_FOUND);
    if (parts.empty()) return fail(ERROR_INVALID_NAME);  // "GAME:" is a folder, not a file

    Target t;
    t.path = join(r.overlay, parts);

    std::error_code ec;
    std::filesystem::create_directories(t.path.parent_path(), ec);
    if (ec) {
        t.error = static_cast<DWORD>(ec.value());
        return t;
    }

    // A deleted package file is not there to copy: copying it up would bring the name back, and an open that keeps
    // the contents (OPEN_EXISTING, TRUNCATE_EXISTING) has to fail on it instead. CREATE_ALWAYS and CREATE_NEW still
    // make a new overlay file, which reads prefer over the package and so over the marker.
    const std::filesystem::path package_path = join(r.game, parts);
    if (copy_up && !deleted(r.overlay, parts) && !std::filesystem::exists(t.path, ec) &&
        std::filesystem::is_regular_file(package_path, ec)) {
        std::filesystem::copy_file(package_path, t.path, std::filesystem::copy_options::none, ec);
        if (ec) {
            t.error = static_cast<DWORD>(ec.value());
            return t;
        }
    }
    return t;
}

bool writes(DWORD access, DWORD disposition) {
    constexpr DWORD kWriteAccess = GENERIC_WRITE | GENERIC_ALL | FILE_WRITE_DATA | FILE_APPEND_DATA | FILE_WRITE_EA |
                                   FILE_WRITE_ATTRIBUTES | DELETE | WRITE_DAC | WRITE_OWNER;
    if ((access & kWriteAccess) != 0) return true;
    return disposition == CREATE_NEW || disposition == CREATE_ALWAYS || disposition == OPEN_ALWAYS ||
           disposition == TRUNCATE_EXISTING;
}

Target open_target(const std::wstring& path, DWORD access, DWORD disposition) {
    if (!writes(access, disposition)) return read_target(path);
    // OPEN_ALWAYS and TRUNCATE_EXISTING read what is already there, and a write-open must not lose a package file's
    // bytes; CREATE_ALWAYS and CREATE_NEW replace it.
    const bool keeps_contents =
        disposition == OPEN_EXISTING || disposition == OPEN_ALWAYS || disposition == TRUNCATE_EXISTING;
    return write_target(path, keeps_contents);
}

// ---- FindFirstFile: the overlay and the package merged into one listing
//
// The host enumerates each folder; the results are merged here, overlay first, so a name that exists in both appears
// once with the overlay's times and size, a name the title deleted does not appear at all, and the overlay's marker
// folder never does.
struct Enumeration {
    std::vector<WIN32_FIND_DATAW> entries;
    size_t next = 0;
};

std::mutex g_enum_mutex;
std::map<HANDLE, std::unique_ptr<Enumeration>> g_enumerations;  // keyed by the pseudo-handle handed to the title

HANDLE register_enumeration(std::unique_ptr<Enumeration> e) {
    std::lock_guard lock(g_enum_mutex);
    const HANDLE h = reinterpret_cast<HANDLE>(e.get());
    g_enumerations.emplace(h, std::move(e));
    return h;
}

bool unregister_enumeration(HANDLE h) {
    std::lock_guard lock(g_enum_mutex);
    return g_enumerations.erase(h) != 0;
}

bool collect(const std::wstring& pattern, const std::vector<std::wstring>& parts, std::vector<WIN32_FIND_DATAW>& out,
             DWORD& error) {
    // "GAME:" and "GAME:\\dir\\" name the folder itself: list everything in it.
    std::vector<std::wstring> dirs;
    std::wstring leaf;
    const bool trailing_separator = !pattern.empty() && (pattern.back() == L'\\' || pattern.back() == L'/');
    if (parts.empty() || trailing_separator) {
        dirs = parts;
        leaf = L"*";
    } else {
        dirs.assign(parts.begin(), parts.end() - 1);
        leaf = parts.back();
    }

    const Roots r = roots();
    if (r.game.empty()) {
        error = ERROR_PATH_NOT_FOUND;
        return false;
    }

    auto add_root = [&](const std::filesystem::path& root, bool overlay_root) {
        if (root.empty()) return;
        std::filesystem::path search = join(root, dirs);
        search /= leaf;
        WIN32_FIND_DATAW found{};
        const HANDLE h = ::FindFirstFileW(search.c_str(), &found);
        if (h == INVALID_HANDLE_VALUE) return;
        do {
            if (overlay_root && same_name(found.cFileName, kDeletedDir)) continue;
            const bool dot = same_name(found.cFileName, L".") || same_name(found.cFileName, L"..");
            if (!overlay_root && !dot) {
                std::vector<std::wstring> entry_parts = dirs;
                entry_parts.push_back(found.cFileName);
                if (deleted(r.overlay, entry_parts)) continue;
            }
            bool duplicate = false;
            for (const auto& e : out)
                if (same_name(e.cFileName, found.cFileName)) {
                    duplicate = true;
                    break;
                }
            if (!duplicate) out.push_back(found);
        } while (::FindNextFileW(h, &found));
        ::FindClose(h);
    };
    add_root(r.overlay, true);
    add_root(r.game, false);

    if (out.empty()) {
        error = ERROR_FILE_NOT_FOUND;
        return false;
    }
    return true;
}

void to_ansi(const WIN32_FIND_DATAW& wide, WIN32_FIND_DATAA* ansi) {
    *ansi = WIN32_FIND_DATAA{};
    ansi->dwFileAttributes = wide.dwFileAttributes;
    ansi->ftCreationTime = wide.ftCreationTime;
    ansi->ftLastAccessTime = wide.ftLastAccessTime;
    ansi->ftLastWriteTime = wide.ftLastWriteTime;
    ansi->nFileSizeHigh = wide.nFileSizeHigh;
    ansi->nFileSizeLow = wide.nFileSizeLow;
    ansi->dwReserved0 = wide.dwReserved0;
    ansi->dwReserved1 = wide.dwReserved1;
    narrow_ansi(wide.cFileName, ansi->cFileName, MAX_PATH);
    narrow_ansi(wide.cAlternateFileName, ansi->cAlternateFileName, 14);
}

}  // namespace

void configure(const std::filesystem::path& game_dir, const std::filesystem::path& overlay_dir) {
    std::lock_guard lock(g_mutex);
    g_game_dir = game_dir;
    g_overlay_dir = overlay_dir;
}

std::filesystem::path game_dir() {
    std::lock_guard lock(g_mutex);
    return g_game_dir;
}

std::filesystem::path overlay_dir() {
    std::lock_guard lock(g_mutex);
    return g_overlay_dir;
}

HANDLE WINAPI create_file_w(LPCWSTR name, DWORD access, DWORD share, LPSECURITY_ATTRIBUTES attributes,
                            DWORD disposition, DWORD flags, HANDLE template_file) {
    if (!name) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_HANDLE_VALUE;
    }
    const Target t = open_target(name, access, disposition);
    if (t.error) {
        SetLastError(t.error);
        return INVALID_HANDLE_VALUE;
    }
    return ::CreateFileW(t.path.c_str(), access, share, attributes, disposition, flags, template_file);
}

HANDLE WINAPI create_file_a(LPCSTR name, DWORD access, DWORD share, LPSECURITY_ATTRIBUTES attributes,
                            DWORD disposition, DWORD flags, HANDLE template_file) {
    if (!name) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_HANDLE_VALUE;
    }
    const std::wstring wide = widen_ansi(name);
    return create_file_w(wide.c_str(), access, share, attributes, disposition, flags, template_file);
}

DWORD WINAPI get_file_attributes_w(LPCWSTR name) {
    if (!name) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_FILE_ATTRIBUTES;
    }
    const Target t = read_target(name);
    if (t.error) {
        SetLastError(t.error);
        return INVALID_FILE_ATTRIBUTES;
    }
    return ::GetFileAttributesW(t.path.c_str());
}

DWORD WINAPI get_file_attributes_a(LPCSTR name) {
    if (!name) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_FILE_ATTRIBUTES;
    }
    const std::wstring wide = widen_ansi(name);
    return get_file_attributes_w(wide.c_str());
}

BOOL WINAPI get_file_attributes_ex_w(LPCWSTR name, GET_FILEEX_INFO_LEVELS level, LPVOID info) {
    if (!name || !info) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    const Target t = read_target(name);
    if (t.error) {
        SetLastError(t.error);
        return FALSE;
    }
    return ::GetFileAttributesExW(t.path.c_str(), level, info);
}

BOOL WINAPI get_file_attributes_ex_a(LPCSTR name, GET_FILEEX_INFO_LEVELS level, LPVOID info) {
    if (!name || !info) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    const std::wstring wide = widen_ansi(name);
    return get_file_attributes_ex_w(wide.c_str(), level, info);
}

HANDLE WINAPI find_first_file_w(LPCWSTR pattern, LPWIN32_FIND_DATAW data) {
    if (!pattern || !data) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_HANDLE_VALUE;
    }
    std::vector<std::wstring> parts;
    const Parsed parsed = parse_console(pattern, parts);
    if (parsed == Parsed::Host) return ::FindFirstFileW(pattern, data);
    if (parsed == Parsed::Refused) {
        SetLastError(ERROR_INVALID_NAME);
        return INVALID_HANDLE_VALUE;
    }

    std::vector<WIN32_FIND_DATAW> entries;
    DWORD error = 0;
    if (!collect(pattern, parts, entries, error)) {
        SetLastError(error);
        return INVALID_HANDLE_VALUE;
    }
    auto e = std::make_unique<Enumeration>();
    e->entries = std::move(entries);
    e->next = 1;
    *data = e->entries[0];
    return register_enumeration(std::move(e));
}

HANDLE WINAPI find_first_file_a(LPCSTR pattern, LPWIN32_FIND_DATAA data) {
    if (!pattern || !data) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_HANDLE_VALUE;
    }
    const std::wstring wide = widen_ansi(pattern);
    WIN32_FIND_DATAW found{};
    const HANDLE h = find_first_file_w(wide.c_str(), &found);
    if (h != INVALID_HANDLE_VALUE) to_ansi(found, data);
    return h;
}

BOOL WINAPI find_next_file_w(HANDLE find, LPWIN32_FIND_DATAW data) {
    if (!find || !data) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    {
        std::lock_guard lock(g_enum_mutex);
        const auto it = g_enumerations.find(find);
        if (it != g_enumerations.end()) {
            Enumeration& e = *it->second;
            if (e.next >= e.entries.size()) {
                SetLastError(ERROR_NO_MORE_FILES);
                return FALSE;
            }
            *data = e.entries[e.next++];
            SetLastError(ERROR_SUCCESS);
            return TRUE;
        }
    }
    return ::FindNextFileW(find, data);  // a host listing
}

BOOL WINAPI find_next_file_a(HANDLE find, LPWIN32_FIND_DATAA data) {
    if (!find || !data) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    WIN32_FIND_DATAW found{};
    const BOOL ok = find_next_file_w(find, &found);
    if (ok) to_ansi(found, data);
    return ok;
}

BOOL WINAPI find_close(HANDLE find) {
    if (!find) {
        SetLastError(ERROR_INVALID_HANDLE);
        return FALSE;
    }
    if (unregister_enumeration(find)) return TRUE;
    return ::FindClose(find);
}

BOOL WINAPI delete_file_w(LPCWSTR name) {
    if (!name) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    std::vector<std::wstring> parts;
    const Parsed parsed = parse_console(name, parts);
    if (parsed == Parsed::Host) return ::DeleteFileW(name);
    if (parsed == Parsed::Refused) {
        SetLastError(ERROR_INVALID_NAME);
        return FALSE;
    }

    const Roots r = roots();
    if (r.overlay.empty() || parts.empty()) {
        SetLastError(ERROR_INVALID_NAME);
        return FALSE;
    }

    bool removed = false;
    std::error_code ec;
    const std::filesystem::path overlay_path = join(r.overlay, parts);
    if (std::filesystem::exists(overlay_path, ec)) {
        if (!::DeleteFileW(overlay_path.c_str())) return FALSE;
        removed = true;
    }
    if (!r.game.empty() && !deleted(r.overlay, parts)) {
        const std::filesystem::path package_path = join(r.game, parts);
        if (std::filesystem::exists(package_path, ec)) {
            if (std::filesystem::is_directory(package_path, ec)) {
                SetLastError(ERROR_ACCESS_DENIED);
                return FALSE;
            }
            // The package file stays where it is; the marker is what hides it.
            const std::filesystem::path mark = marker(r.overlay, parts);
            std::filesystem::create_directories(mark.parent_path(), ec);
            if (ec) {
                SetLastError(static_cast<DWORD>(ec.value()));
                return FALSE;
            }
            const HANDLE h = ::CreateFileW(mark.c_str(), GENERIC_WRITE,
                                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                           CREATE_ALWAYS, FILE_ATTRIBUTE_HIDDEN, nullptr);
            if (h == INVALID_HANDLE_VALUE) return FALSE;  // CreateFileW's error
            ::CloseHandle(h);
            removed = true;
        }
    }
    if (!removed) {
        SetLastError(ERROR_FILE_NOT_FOUND);
        return FALSE;
    }
    return TRUE;
}

BOOL WINAPI delete_file_a(LPCSTR name) {
    if (!name) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    const std::wstring wide = widen_ansi(name);
    return delete_file_w(wide.c_str());
}

}  // namespace rt::fs

namespace {
using namespace rt::fs;

// CreateFile, the FindFirstFile family, the attribute queries and DeleteFile arrive through kernel32/kernelbase
// imports and through the api sets a GDK title is linked against (file-l1-1-0 for the find and attribute calls,
// file-l1-2-0 and file-l2-1-0 for CreateFile).
#define FILE_API(dll)                                                                                    \
    {dll, "CreateFileA", reinterpret_cast<void*>(&create_file_a)},                                       \
        {dll, "CreateFileW", reinterpret_cast<void*>(&create_file_w)},                                   \
        {dll, "FindFirstFileA", reinterpret_cast<void*>(&find_first_file_a)},                            \
        {dll, "FindFirstFileW", reinterpret_cast<void*>(&find_first_file_w)},                            \
        {dll, "FindNextFileA", reinterpret_cast<void*>(&find_next_file_a)},                              \
        {dll, "FindNextFileW", reinterpret_cast<void*>(&find_next_file_w)},                              \
        {dll, "FindClose", reinterpret_cast<void*>(&find_close)},                                        \
        {dll, "GetFileAttributesA", reinterpret_cast<void*>(&get_file_attributes_a)},                    \
        {dll, "GetFileAttributesW", reinterpret_cast<void*>(&get_file_attributes_w)},                    \
        {dll, "GetFileAttributesExA", reinterpret_cast<void*>(&get_file_attributes_ex_a)},               \
        {dll, "GetFileAttributesExW", reinterpret_cast<void*>(&get_file_attributes_ex_w)},               \
        {dll, "DeleteFileA", reinterpret_cast<void*>(&delete_file_a)},                                   \
        {dll, "DeleteFileW", reinterpret_cast<void*>(&delete_file_w)}
}  // namespace

SHIM_TABLE(file_system,
    FILE_API("kernel32.dll"),
    FILE_API("kernelbase.dll"),
    FILE_API("api-ms-win-core-file-l1-1-0.dll"),
    FILE_API("api-ms-win-core-file-l1-2-0.dll"),
    FILE_API("api-ms-win-core-file-l2-1-0.dll"),
);
