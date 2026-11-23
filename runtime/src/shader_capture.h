// Shader containers the layer cannot translate, kept so a build step can translate them later.
//
// The retail title builds some shader programs at run time, so they are not in shaders.rpf and the shader stage
// never carves them: the first one stops a run at the pipeline that asks for it (the 31 measured on retail 1.0.2.1
// in #8851 A40/A44). The layer gets the container in hand before it fails, so with this switch on it writes it out
// and a later build step translates it with the same three tools the carve uses:
//
//   LEONIDA_SHADER_CAPTURE=<dir>   <dir>/<fssm_sha256>.dxbc   the container as the title passed it
//                                  <dir>/<fssm_sha256>.json   {"stage":"cs"} - what tools/rdna2spirv.py reads from
//                                                             an index entry and a container does not carry
//
// These files are game data: the folder is for the machine that has the title and must never be committed, attached
// or published (CONTRIBUTING.md). Nothing here logs their contents, only the name it wrote.
//
// Off by default: without the variable each call is one relaxed load. On, the first call for a program writes it
// once and later calls for the same program do nothing, so a run that asks for the same shader twice writes one file.
#pragma once

#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <windows.h>

namespace leonida {

// The folder LEONIDA_SHADER_CAPTURE names, or empty for off. Read per call: it is only asked when a shader is
// missing, which happens a handful of times per run, and an uncached read is what lets a test exercise the on and
// off paths in one process.
inline std::filesystem::path shader_capture_directory() {
    constexpr wchar_t env_name[] = L"LEONIDA_SHADER_CAPTURE";
    const DWORD size = ::GetEnvironmentVariableW(env_name, nullptr, 0);
    if (!size) return {};
    std::wstring value(size, L'\0');
    const DWORD length = ::GetEnvironmentVariableW(env_name, value.data(), size);
    if (!length || length >= size) return {};
    value.resize(length);
    return std::filesystem::path(value);
}

// Writes `container` for the program whose SHA-256 is `program_sha256`, with its stage, when the switch is on.
// Returns true when the container is on disk (written now or by an earlier call), false when nothing was written.
// Never throws: the D3D12 shim runs inside the title, and a capture that fails must not turn into an exception.
inline bool capture_unresolved_shader(std::span<const std::uint8_t> container, std::string_view stage,
                                      const std::string& program_sha256) {
    const std::filesystem::path& dir = shader_capture_directory();
    if (dir.empty() || container.empty() || program_sha256.empty()) return false;
    // One file per program per run: the same container comes back for every pipeline that uses it.
    static std::mutex mutex;
    static std::set<std::string> seen;
    {
        const std::lock_guard lock(mutex);
        if (!seen.insert(program_sha256).second) return true;
    }
    try {
        std::error_code error;
        std::filesystem::create_directories(dir, error);
        if (!std::filesystem::is_directory(dir, error)) return false;
        const std::filesystem::path container_path = dir / (program_sha256 + ".dxbc");
        const std::filesystem::path stage_path = dir / (program_sha256 + ".json");
        {
            std::ofstream out(container_path, std::ios::binary | std::ios::trunc);
            if (!out) return false;
            out.write(reinterpret_cast<const char*>(container.data()),
                      static_cast<std::streamsize>(container.size()));
            if (!out) return false;
        }
        {
            std::ofstream out(stage_path, std::ios::binary | std::ios::trunc);
            if (!out) return false;
            out << "{\"stage\":\"" << std::string(stage) << "\"}\n";
            if (!out) return false;
        }
        std::fprintf(stderr,
                     "[rt] d3d12: captured the unresolved %s container as %s (game data: keep this folder local)\n",
                     std::string(stage).c_str(), container_path.string().c_str());
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace leonida
