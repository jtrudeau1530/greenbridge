// Pipelines one session creates are found in the pipeline cache of the next, and the file that carries them is kept safe.
// A D3D12 device is made on a Vulkan device again and again in one process, each time with LEONIDA_PIPELINE_CACHE
// naming a scratch file, and creates compute PSOs; VK_EXT_pipeline_creation_feedback (rt::d3d12::pipeline_cache_stats)
// says how many came from the cache. What is checked, each against a way it went wrong:
//   - a second session finds every pipeline of the first, and a session that made nothing new leaves the file alone;
//   - a file that is junk, cut short or not this GPU's is ignored and replaced; an unwritable path, a limit of 0 MB and
//     LEONIDA_PIPELINE_CACHE=off cost the session nothing (and off writes no file anywhere);
//   - a name that the process's code page cannot hold works;
//   - another process holding the file open (and mapped) neither stops the load nor the replacement;
//   - the build's pipeline_cache.bin beside the executable is never written: not after the working directory changes, not
//     after a link is retargeted; a seed rebuilt during a session wins over what the session saved;
//   - two sessions that started from the same file and each learned something both survive the second one's save;
//   - a device without feedback still saves.
// The shaders are hand-written SPIR-V no-ops that differ in their workgroup size, so each PSO is a pipeline of its own.
// No game data. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/pipeline_cache_store.h"
#include "../src/shader_library.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace store = leonida::pipeline_cache_store;

int failures = 0;

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); ++failures; } \
} while (0)
#define REQUIRE(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

constexpr unsigned pipeline_count = 16;  // one full session; the library holds 32 distinct pipelines

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

using SerializeFn = HRESULT(WINAPI*)(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC*, ID3DBlob**, ID3DBlob**);

void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(std::uint8_t(v >> (8 * i)));
}

// zlib with one stored block (RFC 1950/1951), enough to wrap a test program.
std::vector<std::uint8_t> zlib_stored(const std::vector<std::uint8_t>& data) {
    std::vector<std::uint8_t> out = {0x78, 0x01, 0x01};
    const auto length = std::uint16_t(data.size());
    out.insert(out.end(), {std::uint8_t(length), std::uint8_t(length >> 8), std::uint8_t(~length),
                           std::uint8_t(~length >> 8)});
    out.insert(out.end(), data.begin(), data.end());
    std::uint32_t a = 1, b = 0;
    for (const std::uint8_t x : data) {
        a = (a + x) % 65521;
        b = (b + a) % 65521;
    }
    const std::uint32_t adler = (b << 16) | a;
    out.insert(out.end(), {std::uint8_t(adler >> 24), std::uint8_t(adler >> 16), std::uint8_t(adler >> 8),
                           std::uint8_t(adler)});
    return out;
}

// DXBC { DXIL { stub, XDAT { 0x0D: XBCH + zlib(fssm) } } }, the layout tools/fssm_unpack.py reads.
std::vector<std::uint8_t> shader_container(const std::vector<std::uint8_t>& fssm) {
    const auto stream = zlib_stored(fssm);
    std::vector<std::uint8_t> xbch = {'X', 'B', 'C', 'H'};
    for (const std::uint32_t v : {1u, std::uint32_t(stream.size()), 0u, std::uint32_t(fssm.size()), 0u}) put32(xbch, v);
    xbch.insert(xbch.end(), stream.begin(), stream.end());
    std::vector<std::uint8_t> dxil(16, 0);
    dxil.insert(dxil.end(), {'X', 'D', 'A', 'T'});
    put32(dxil, 1);
    put32(dxil, 12);
    put32(dxil, 0x0d);
    put32(dxil, 16 + 12 + 12);
    put32(dxil, std::uint32_t(xbch.size()));
    dxil.insert(dxil.end(), xbch.begin(), xbch.end());
    std::vector<std::uint8_t> out = {'D', 'X', 'B', 'C'};
    out.resize(28, 0);
    put32(out, 1);
    put32(out, 36);
    out.insert(out.end(), {'D', 'X', 'I', 'L'});
    put32(out, std::uint32_t(dxil.size()));
    out.insert(out.end(), dxil.begin(), dxil.end());
    return out;
}

std::vector<std::uint8_t> serialize(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& desc) {
    const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto fn = d3d12 ? reinterpret_cast<SerializeFn>(
                                reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12SerializeVersionedRootSignature")))
                          : nullptr;
    std::vector<std::uint8_t> out;
    ID3DBlob* blob = nullptr;
    if (fn && SUCCEEDED(fn(&desc, &blob, nullptr)) && blob) {
        const auto* p = static_cast<const std::uint8_t*>(blob->GetBufferPointer());
        out.assign(p, p + blob->GetBufferSize());
        blob->Release();
    }
    return out;
}

// void main() {} with a workgroup of 32 * y * z invocations: a compute shader of no use, valid, and one pipeline per size.
std::vector<std::uint32_t> noop_compute_spirv(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    return {0x07230203, 0x00010000, 0, 5, 0,                                // header, ids 1..4
            0x00020011, 1,                                                  // OpCapability Shader
            0x0003000e, 0, 1,                                               // OpMemoryModel Logical GLSL450
            0x0005000f, 5, 1, 0x6e69616d, 0,                                // OpEntryPoint GLCompute %1 "main"
            0x00060010, 1, 17, x, y, z,                                     // OpExecutionMode %1 LocalSize x y z
            0x00020013, 2,                                                  // %2 = OpTypeVoid
            0x00030021, 3, 2,                                               // %3 = OpTypeFunction %2
            0x00050036, 2, 1, 0, 3,                                         // %1 = OpFunction %2 None %3
            0x000200f8, 4,                                                  // %4 = OpLabel
            0x000100fd,                                                     // OpReturn
            0x00010038};                                                    // OpFunctionEnd
}

constexpr unsigned variants = 32;
std::uint32_t workgroup_y(unsigned i) { return 1 + i % 8; }
std::uint32_t workgroup_z(unsigned i) { return 1 + i / 8; }  // 32 * 8 * 4 = 1024 invocations at most

// The program the PSO carries: its SHA-256 is what the translator index finds the module by.
std::vector<std::uint8_t> fssm_of(unsigned i) {
    std::vector<std::uint8_t> fssm = {'F', 'S', 'S', 'M'};
    put32(fssm, 5000 + i);
    for (std::uint32_t k = 0; k < 14; ++k) put32(fssm, 0xbf810000u + k);
    return fssm;
}

bool write_library(const fs::path& dir) {
    std::string index = "{";
    for (unsigned i = 0; i < variants; ++i) {
        const std::string name = "psocache_" + std::to_string(i);
        const auto words = noop_compute_spirv(32, workgroup_y(i), workgroup_z(i));
        std::ofstream spv(dir / (name + ".spv"), std::ios::binary);
        spv.write(reinterpret_cast<const char*>(words.data()), std::streamsize(words.size() * 4));
        if (!spv) return false;
        index += std::string(i ? ",\n " : "") + "\"" + name + "\": {\"stage\": \"cs\", \"fssm_sha256\": \"" +
                 leonida::sha256_hex(fssm_of(i)) + "\", \"bindings\": [], \"push_constant_bytes\": 144, " +
                 "\"local_size\": [32, " + std::to_string(workgroup_y(i)) + ", " + std::to_string(workgroup_z(i)) +
                 "], \"uses_code_base\": true, \"wave64\": false}";
    }
    std::ofstream out(dir / "index.json", std::ios::binary);
    out << index << "}";
    return bool(out);
}

struct Setup {
    vktest::Device& d;
    fs::path library;
    std::vector<std::uint8_t> root_blob;
    bool feedback = true;  // ask for pipeline creation feedback (Vulkan 1.3); false: the size-based path
};

struct Session {
    bool ok = false;
    rt::d3d12::PipelineCacheStats stats;
    double wall_ms = 0;  // the whole CreateComputePipelineState calls, as the caller sees them
};

// One device with PSOs [first, first + count); the PSOs, the root signature and the device are released with it, and
// the device saves its cache while it is destroyed.
struct Live {
    Ref<ID3D12Device> device;
    Ref<ID3D12RootSignature> root;
    std::vector<Ref<ID3D12PipelineState>> psos;
    Session session;
};

bool open_session(Live& live, const Setup& setup, unsigned first, unsigned count) {
    auto& d = setup.d;
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = d.instance_proc;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    info.buffer_device_address = true;
    info.subgroup = d.subgroup;
    info.pipeline_creation_feedback = setup.feedback && d.core_1_3();
    info.shader_dir = setup.library.string();
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&live.device.p)))) return false;
    if (FAILED(live.device->CreateRootSignature(0, setup.root_blob.data(), setup.root_blob.size(),
                                                IID_PPV_ARGS(&live.root.p))))
        return false;
    live.psos.resize(count);
    LARGE_INTEGER frequency, begin, end;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&begin);
    for (unsigned i = 0; i < count; ++i) {
        const auto container = shader_container(fssm_of(first + i));
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature = live.root.p;
        desc.CS = {container.data(), container.size()};
        if (FAILED(live.device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&live.psos[i].p)))) return false;
    }
    QueryPerformanceCounter(&end);
    live.session.wall_ms = 1000.0 * double(end.QuadPart - begin.QuadPart) / double(frequency.QuadPart);
    live.session.stats = rt::d3d12::pipeline_cache_stats(live.device.p);
    live.session.ok = true;
    return true;
}

// One whole session; `before_release` runs with the PSOs made and the device still alive.
Session run_session(const Setup& setup, unsigned first = 0, unsigned count = pipeline_count,
                    const std::function<void()>& before_release = {}) {
    Live live;
    if (!open_session(live, setup, first, count)) return {};
    if (before_release) before_release();
    return live.session;
}

void print(const char* label, const Session& s) {
    std::printf("%s: %u pipelines, %u from the pipeline cache, %.2f ms creating them (%.2f ms through the API), "
                "%llu bytes loaded\n",
                label, s.stats.pipelines, s.stats.cache_hits, double(s.stats.create_nanoseconds) / 1e6, s.wall_ms,
                static_cast<unsigned long long>(s.stats.loaded_bytes));
}

bool set_env(const wchar_t* name, const wchar_t* value) { return SetEnvironmentVariableW(name, value) != FALSE; }
bool set_cache(const fs::path& path) { return set_env(L"LEONIDA_PIPELINE_CACHE", path.c_str()); }

std::string read_bytes(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), {}};
}

void write_bytes(const fs::path& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), std::streamsize(bytes.size()));
}

// The driver's own bytes in a session file (what the build's seed holds as is).
std::string payload_of(const fs::path& session_file) {
    const std::string bytes = read_bytes(session_file);
    return bytes.compare(0, 4, "LPCS") == 0 ? bytes.substr(store::kFrameBytes) : bytes;
}

std::set<std::wstring> names_in(const fs::path& dir) {
    std::set<std::wstring> out;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(dir, ec)) out.insert(entry.path().filename().wstring());
    return out;
}

std::wstring original_cwd;
void restore_cwd() { SetCurrentDirectoryW(original_cwd.c_str()); }

fs::path executable_dir() {
    wchar_t path[MAX_PATH * 4] = {};
    GetModuleFileNameW(nullptr, path, DWORD(std::size(path)));
    return fs::path(path).parent_path();
}

// Holds a file open for writing and mapped, as a process reading it through a view would, until released or destroyed.
class Holder {
public:
    explicit Holder(const fs::path& path, unsigned milliseconds) {
        HANDLE ready = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        thread_ = std::thread([=, this] {
            HANDLE file = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                                      FILE_ATTRIBUTE_NORMAL, nullptr);
            HANDLE mapping = file == INVALID_HANDLE_VALUE
                                 ? nullptr
                                 : CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
            void* view = mapping ? MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0) : nullptr;
            held_ = view != nullptr;
            SetEvent(ready);
            Sleep(milliseconds);
            if (view) UnmapViewOfFile(view);
            if (mapping) CloseHandle(mapping);
            if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
        });
        WaitForSingleObject(ready, INFINITE);
        CloseHandle(ready);
    }
    ~Holder() { thread_.join(); }
    bool held() const { return held_; }

private:
    std::thread thread_;
    std::atomic<bool> held_{false};
};

// A build's pipeline_cache.bin beside the executable for as long as this lives (never over a real one), and the
// working directory put back.
class FakeSeed {
public:
    FakeSeed(const std::string& payload) : path_(executable_dir() / "pipeline_cache.bin") {
        if (fs::exists(path_)) return;
        write_bytes(path_, payload);
        made_ = true;
    }
    ~FakeSeed() {
        restore_cwd();
        std::error_code ec;
        if (made_) fs::remove(path_, ec);
    }
    bool usable() const { return made_; }
    const fs::path& path() const { return path_; }

private:
    fs::path path_;
    bool made_ = false;
};

}  // namespace

int main(int argc, char** argv) {
    // --periodic (its own process: the interval is read once): a session saves while it runs, see below.
    const bool periodic = argc == 2 && !std::strcmp(argv[1], "--periodic");
    if (periodic) REQUIRE(set_env(L"LEONIDA_PIPELINE_CACHE_SAVE_SECONDS", L"1"));
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    REQUIRE(device_status == vktest::DeviceStatus::ready);

    wchar_t cwd[MAX_PATH * 4] = {};
    GetCurrentDirectoryW(DWORD(std::size(cwd)), cwd);
    original_cwd = cwd;
    const testfiles::ScratchDir scratch("leonida_pso_cache_test");
    const auto& dir = scratch.path();
    REQUIRE(write_library(dir));
    const fs::path session_file = dir / "session_cache.bin";
    REQUIRE(set_cache(session_file));

    // The console compute layout of d3d12_compute_test: two root constants, a root UAV and a one-CBV table.
    D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 1, 0, 0};
    D3D12_ROOT_PARAMETER params[3] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants = {0, 0, 2};
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[1].Descriptor = {0, 0};
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[2].DescriptorTable = {1, &range};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0 = {3, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    Setup setup{d, dir, serialize(root_desc)};
    REQUIRE(!setup.root_blob.empty());

    if (periodic) {
        // A session that is never torn down (a title that is killed, or ends through TerminateProcess) still leaves
        // its pipelines behind: with a 1 s interval, the first pipeline created more than a second into the session
        // saves the cache while the device is alive, and the next session starts from that file.
        Live live;
        REQUIRE(open_session(live, setup, 0, pipeline_count - 1));
        Sleep(1200);
        const auto container = shader_container(fssm_of(pipeline_count - 1));
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature = live.root.p;
        desc.CS = {container.data(), container.size()};
        Ref<ID3D12PipelineState> late;
        REQUIRE(SUCCEEDED(live.device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&late.p))));
        CHECK(fs::is_regular_file(session_file));  // the device is still alive
        const store::Stamp while_alive = store::stamp_of(session_file);
        CHECK(payload_of(session_file).size() > 64);
        // Nothing new since: a pipeline the cache has, a second later, does not write the file again.
        Sleep(1200);
        Ref<ID3D12PipelineState> same;
        REQUIRE(SUCCEEDED(live.device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&same.p))));
        const bool feedback = rt::d3d12::pipeline_cache_stats(live.device.p).pipelines == pipeline_count + 1;
        if (feedback) CHECK(store::stamp_of(session_file) == while_alive);
        std::printf("periodic: the cache was saved while the session ran (%llu bytes)\n",
                    static_cast<unsigned long long>(fs::file_size(session_file)));
        if (failures) std::printf("%d check(s) failed\n", failures);
        return failures ? 1 : 0;
    }

    // 1. The second session finds what the first made; the file survives a session that made nothing new.
    const Session first = run_session(setup);
    REQUIRE(first.ok);
    print("session 1, no cache file", first);
    REQUIRE(fs::is_regular_file(session_file));
    // A driver that gives no (valid) feedback cannot show the hits: everything else is still checked.
    const bool feedback_valid = first.stats.pipelines == pipeline_count;
    CHECK(first.stats.created == pipeline_count);
    const store::Stamp saved = store::stamp_of(session_file);
    const Session second = run_session(setup);
    REQUIRE(second.ok);
    print("session 2, cache saved by session 1", second);
    CHECK(second.stats.loaded_bytes > 0);
    CHECK(second.stats.created == pipeline_count);
    if (feedback_valid) CHECK(second.stats.cache_hits == pipeline_count);
    CHECK(store::stamp_of(session_file) == saved);

    // 2. A file that is not a pipeline cache, or is one cut short, is ignored and then replaced by a real one.
    const std::string good = read_bytes(session_file);
    write_bytes(session_file, std::string(100, 'x'));
    const Session junk = run_session(setup);
    print("session 3, junk in the cache file", junk);
    CHECK(junk.ok && junk.stats.loaded_bytes == 0 && junk.stats.cache_hits == 0);
    CHECK(fs::file_size(session_file) > 100);
    const std::string payload = payload_of(session_file);
    CHECK(payload.size() > 64);
    for (const std::string& cut : {good.substr(0, good.size() - 1), payload.substr(0, 33),
                                   good.substr(0, store::kFrameBytes + 33)}) {
        write_bytes(session_file, cut);
        const Session cut_short = run_session(setup);
        CHECK(cut_short.ok && cut_short.stats.loaded_bytes == 0 && cut_short.stats.cache_hits == 0);
        CHECK(fs::file_size(session_file) > cut.size());  // replaced, not kept
    }
    {
        std::string flipped = good;
        flipped[flipped.size() / 2] ^= 0x40;  // a byte of the driver's data changed
        write_bytes(session_file, flipped);
        const Session damaged = run_session(setup);
        CHECK(damaged.ok && damaged.stats.loaded_bytes == 0);
    }
    const Session again = run_session(setup);
    CHECK(again.ok && again.stats.loaded_bytes > 0);
    const std::string seed_payload = payload_of(session_file);  // pipelines 0 to 15, as the build would have made them

    // 3. A device without feedback (Vulkan 1.2) saves when its cache grew, and leaves the file alone when it did not.
    {
        const fs::path plain = dir / "plain.bin";
        REQUIRE(set_cache(plain));
        Setup without = setup;
        without.feedback = false;
        const Session grown = run_session(without);
        CHECK(grown.ok && grown.stats.pipelines == 0 && grown.stats.created == pipeline_count);
        CHECK(fs::is_regular_file(plain));
        const store::Stamp plain_stamp = store::stamp_of(plain);
        const Session same = run_session(without);
        CHECK(same.ok && same.stats.loaded_bytes > 0 && store::stamp_of(plain) == plain_stamp);
    }

    // 4. A name the process's code page cannot hold: saving and loading it must not throw.
    {
        const fs::path wide = dir / L"\u6f22\u5b57\u0416.bin";
        REQUIRE(set_cache(wide));
        const Session cold = run_session(setup);
        CHECK(cold.ok && fs::is_regular_file(wide));
        const Session warm = run_session(setup);
        CHECK(warm.ok && warm.stats.loaded_bytes > 0);
        if (feedback_valid) CHECK(warm.stats.cache_hits == pipeline_count);
    }

    // 5. Another process has the file open for writing and mapped while this one starts, learns something and saves: it
    // is still loaded, and the replacement waits for the other to let go (about 0.8 s here).
    {
        REQUIRE(set_cache(session_file));
        const std::uint64_t before = fs::file_size(session_file);
        const Holder holder(session_file, 800);
        CHECK(holder.held());
        const Session held = run_session(setup, pipeline_count, 8);
        CHECK(held.ok && held.stats.loaded_bytes > 0);
        // The file is the old one plus 8 pipelines once it has been replaced.
        CHECK(fs::file_size(session_file) > before);
    }

    // 6. A session that started from a file and learned something does not drop what another session saved meanwhile.
    {
        const fs::path shared = dir / "shared.bin";
        REQUIRE(set_cache(shared));
        const Session base = run_session(setup, 0, 8);
        CHECK(base.ok && fs::is_regular_file(shared));
        {
            Live a, b;
            CHECK(open_session(a, setup, 8, 8));
            CHECK(open_session(b, setup, 16, 8));
            a.psos.clear();
            a.root.reset();
            a.device.reset();  // A saves first
        }                      // then B, which started from the same file
        const Session all = run_session(setup, 0, 24);
        print("session after two that learned in parallel", all);
        CHECK(all.ok && all.stats.created == 24);
        if (feedback_valid) CHECK(all.stats.cache_hits == 24);
    }

    // 7. The build's pipeline_cache.bin beside the executable is never written, and a rebuilt one wins.
    {
        REQUIRE(set_cache(dir / "seeded.bin"));
        const std::string& seed_bytes = seed_payload;
        FakeSeed seed(seed_bytes);
        if (seed.usable()) {
            const std::string untouched = read_bytes(seed.path());
            // The seed is rebuilt while a session runs: that session's file would outrank it, so it is not written.
            const fs::path rebuilt = dir / "rebuilt.bin";
            REQUIRE(set_cache(rebuilt));
            const Session during = run_session(setup, pipeline_count, 8, [&] {
                std::error_code ignored;
                fs::last_write_time(seed.path(), fs::file_time_type::clock::now() + std::chrono::seconds(60), ignored);
            });
            CHECK(during.ok && during.stats.loaded_bytes == seed_bytes.size());
            CHECK(!fs::exists(rebuilt));
            const Session next = run_session(setup, pipeline_count, 8);
            CHECK(next.ok && next.stats.loaded_bytes == seed_bytes.size());  // started from the new seed
            // The name is made another name for the seed after the check: the file it reaches is the seed's.
            {
                const fs::path late = executable_dir() / "pipeline_cache_late.bin";  // same volume as the seed
                REQUIRE(set_cache(late));
                const Session aliased = run_session(setup, pipeline_count + 8, 8, [&] {
                    std::error_code made;
                    fs::create_hard_link(seed.path(), late, made);
                    if (made) std::printf("no hard links here: %s\n", made.message().c_str());
                });
                CHECK(aliased.ok);
                if (fs::exists(late)) CHECK(read_bytes(late) == untouched && read_bytes(seed.path()) == untouched);
                std::error_code ignored;
                fs::remove(late, ignored);
            }

            // The working directory changes between the check and the save.
            const fs::path work = dir / "work";
            fs::create_directory(work);
            SetCurrentDirectoryW(work.c_str());
            REQUIRE(set_env(L"LEONIDA_PIPELINE_CACHE", L"pipeline_cache.bin"));
            const Session moved = run_session(setup, pipeline_count, 8, [&] { SetCurrentDirectoryW(executable_dir().c_str()); });
            CHECK(moved.ok && moved.stats.loaded_bytes == seed_bytes.size());
            CHECK(read_bytes(seed.path()) == untouched);
            CHECK(fs::is_regular_file(work / "pipeline_cache.bin"));
            restore_cwd();

            // A link on the way is pointed at the executable's folder after the check.
            const fs::path link = dir / "link";
            std::error_code ec;
            fs::create_directory_symlink(work, link, ec);
            if (!ec && fs::is_directory(link)) {  // Wine leaves a link to a DOS path dangling
                REQUIRE(set_cache(link / "pipeline_cache.bin"));
                const Session retargeted = run_session(setup, pipeline_count + 8, 8, [&] {
                    std::error_code ignored;
                    fs::remove(link, ignored);
                    fs::create_directory_symlink(executable_dir(), link, ignored);
                });
                CHECK(retargeted.ok);
                CHECK(read_bytes(seed.path()) == untouched);
                std::error_code ignored;
                fs::remove(link, ignored);
            } else {
                std::printf("no usable symbolic links here: the retargeted link is not checked\n");
            }

        } else {
            std::printf("a pipeline_cache.bin is already beside the executable: the seed checks are skipped\n");
        }
        restore_cwd();
    }

    // 8. A limit of 0 MB saves nothing, "off" loads and saves nothing and writes no file where it is run from.
    {
        const fs::path limited = dir / "limited.bin";
        REQUIRE(set_cache(limited));
        REQUIRE(set_env(L"LEONIDA_PIPELINE_CACHE_MAX_MB", L"0"));
        const Session over = run_session(setup);
        CHECK(over.ok && over.stats.created == pipeline_count && !fs::exists(limited));
        REQUIRE(set_env(L"LEONIDA_PIPELINE_CACHE_MAX_MB", nullptr));

        const fs::path probe = dir / "probe";
        fs::create_directory(probe);
        SetCurrentDirectoryW(probe.c_str());
        REQUIRE(set_env(L"LEONIDA_PIPELINE_CACHE", L"off"));
        const auto in_probe = names_in(probe), in_scratch = names_in(dir), in_exe = names_in(executable_dir());
        const Session off = run_session(setup);
        CHECK(off.ok && off.stats.created == pipeline_count && off.stats.loaded_bytes == 0);
        CHECK(names_in(probe) == in_probe && names_in(dir) == in_scratch && names_in(executable_dir()) == in_exe);
        restore_cwd();
    }

    // 9. A path nothing can be saved to costs the session nothing but the save.
    {
        const fs::path unwritable = dir / "a_directory";
        REQUIRE(fs::create_directory(unwritable));
        REQUIRE(set_cache(unwritable));
        const Session blocked = run_session(setup);
        CHECK(blocked.ok && blocked.stats.created == pipeline_count);
        CHECK(fs::is_directory(unwritable));
        std::size_t entries = 0;
        for (const auto& entry : fs::directory_iterator(dir))
            entries += entry.path().filename().wstring().rfind(L"a_directory", 0) == 0;
        CHECK(entries == 1);  // no half-written staging file left beside it
    }

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    if (!feedback_valid) {
        std::fprintf(stderr, "SKIP (after every other check): the driver gave no valid pipeline creation feedback\n");
        return 77;
    }
    std::printf("pipeline cache persistence: ok\n");
    return 0;
}
