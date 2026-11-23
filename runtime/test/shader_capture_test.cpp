// LEONIDA_SHADER_CAPTURE (src/shader_capture.h): when the variable names a folder, a shader container the layer
// cannot translate is written there, with a sidecar naming its stage, so a build step can translate it later. Off by
// default. Nothing here is game data: the container is synthetic, which is what A44 asks for.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "shader_capture.h"

namespace {

int g_failed = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok   " : "FAIL ", what);
    if (!ok) ++g_failed;
}

std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// A container shaped like the title's: a DXBC header, then a payload with NULs in it, so a text-mode write shows.
std::vector<std::uint8_t> synthetic_container() {
    std::vector<std::uint8_t> bytes = {'D', 'X', 'B', 'C', 0x00, 0x01, 0x00, 0x00};
    for (const char* word : {"FSSM", "\x00\x01\x01\x00", "\x2c\x00\x00\x00"})
        for (const char* p = word; *p; ++p) bytes.push_back(static_cast<std::uint8_t>(*p));
    bytes.push_back(0x00);
    bytes.push_back(0xFF);
    bytes.push_back(0x7F);
    return bytes;
}

constexpr const char* kHash = "b5316b2eabca2b441f9d247699612f346be5851971dd3c6439019e757d6bfdc0";
constexpr const char* kOtherHash = "fe43ab80aed8255a966007c63bd061a5119a91c1e8fbfa69cfc0c7be075451dc";

}  // namespace

int main() {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("leonida_capture_" + std::to_string(::GetCurrentProcessId()));
    std::error_code error;
    std::filesystem::remove_all(dir, error);

    // Off: no variable, nothing written.
    ::SetEnvironmentVariableW(L"LEONIDA_SHADER_CAPTURE", nullptr);
    const auto container = synthetic_container();
    check(leonida::shader_capture_directory().empty(), "without the variable the switch is off");
    check(!leonida::capture_unresolved_shader(container, "cs", kHash), "off: a capture writes nothing");
    check(!std::filesystem::exists(dir / (std::string(kHash) + ".dxbc")), "off: no container on disk");

    // On: the container and its stage land in the folder, byte for byte.
    ::SetEnvironmentVariableW(L"LEONIDA_SHADER_CAPTURE", dir.wstring().c_str());
    check(!leonida::shader_capture_directory().empty(), "with the variable set the switch is on");
    check(leonida::capture_unresolved_shader(container, "cs", kHash), "on: a capture reports the file written");
    const auto written = read_bytes(dir / (std::string(kHash) + ".dxbc"));
    check(written == container, "the captured container is byte-identical, NULs and high bytes included");
    check(read_text(dir / (std::string(kHash) + ".json")) == "{\"stage\":\"cs\"}\n",
          "the sidecar names the stage rdna2spirv.py cannot read from a container");

    // A second program in the same folder keeps its own name, and its stage is whatever the layer asked for.
    check(leonida::capture_unresolved_shader(container, "ps", kOtherHash), "on: a second program is captured");
    check(read_text(dir / (std::string(kOtherHash) + ".json")) == "{\"stage\":\"ps\"}\n",
          "each sidecar carries its own stage");

    // The same program again: already written this run, so nothing is rewritten and the answer is still true.
    const auto before = std::filesystem::last_write_time(dir / (std::string(kHash) + ".dxbc"), error);
    check(leonida::capture_unresolved_shader(container, "cs", kHash), "on: asking for a captured program again is true");
    check(std::filesystem::last_write_time(dir / (std::string(kHash) + ".dxbc"), error) == before,
          "asking again does not rewrite the file");

    // Nothing to write: no bytes, or no program name.
    check(!leonida::capture_unresolved_shader({}, "cs", std::string(kOtherHash) + "0"),
          "an empty container writes nothing");
    check(!leonida::capture_unresolved_shader(container, "cs", ""), "a program without a hash writes nothing");
    check(!std::filesystem::exists(dir / (std::string(kOtherHash) + "0.dxbc")), "no file for either refusal");

    // Off again in the same process: the read is not cached, so turning the switch off stops new captures.
    ::SetEnvironmentVariableW(L"LEONIDA_SHADER_CAPTURE", nullptr);
    check(!leonida::capture_unresolved_shader(container, "cs", std::string(kHash) + "1"),
          "turning the switch off mid-run stops capturing");

    std::filesystem::remove_all(dir, error);
    std::printf("shader_capture_test: %s (%d failures)\n", g_failed ? "FAILED" : "OK", g_failed);
    return g_failed ? 1 : 0;
}
