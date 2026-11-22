// The XDXR BVH4 GPU builder (src/xdxr_build.comp, xdxr_build.h) on synthetic scenes. Each case is built on the
// device the Vulkan loader picks and written to <out dir>/<case>.bin, the acceleration structure as the GPU left it,
// with <case>.json describing the scene: its address and the triangles (or instances) it was built from.
// `python -I tools/xdxr_bvh.py check-build <out dir>` then validates every structure and traces rays through it
// against brute force (ctest runs both; see CMakeLists.txt).
//
// Cases: one triangle; five (two levels); 1000 random triangles; an indexed R16 grid with a 16-byte vertex stride;
// R32 indices with R32G32 vertices; a geometry transform; two geometries with an inactive (NaN) triangle; no
// triangles; and a TLAS of four instances (rotated and scaled, one with mask 0, one with no BLAS).
// Exit 77 when there is no Vulkan device with bufferDeviceAddress.
// `vk_xdxr_build_test <out dir> <triangles>` instead times BLAS builds of that many random triangles (wall clock of
// one submission each, after one warm-up; result and scratch in device-local memory) and writes nothing.
#include "vk_test_device.h"
#include "../src/xdxr_build.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using namespace leonida;

struct Tri { float v[3][3]; std::uint32_t geometry, primitive; };

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    void* map = nullptr;
    std::uint64_t address = 0;
};

Buffer make(vktest::Device& gpu, std::size_t bytes, bool device_local = false) {
    Buffer b;
    b.buffer = gpu.buffer(bytes ? bytes : 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, device_local ? nullptr : &b.map);
    if (b.buffer) b.address = gpu.device_address(b.buffer);
    return b;
}

void barrier_to_host(vktest::Device& gpu, VkCommandBuffer cb) {
    VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    gpu.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                                            VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr, 0,
                                                            nullptr);
}

std::string num(double x) {
    if (std::isnan(x)) return "\"nan\"";
    char s[64];
    std::snprintf(s, sizeof s, "%.9g", x);
    return s;
}

bool write_file(const std::string& path, const void* data, std::size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    f.write(static_cast<const char*>(data), std::streamsize(bytes));
    return bool(f);
}

std::string tris_json(const std::vector<Tri>& tris) {
    std::string s = "[";
    for (std::size_t i = 0; i < tris.size(); ++i) {
        const Tri& t = tris[i];
        s += i ? ",\n  [" : "\n  [";
        for (int k = 0; k < 3; ++k)
            s += (k ? ", [" : "[") + num(t.v[k][0]) + ", " + num(t.v[k][1]) + ", " + num(t.v[k][2]) + "]";
        s += ", " + std::to_string(t.geometry) + ", " + std::to_string(t.primitive) + "]";
    }
    return s + "]";
}

struct Geometry {
    std::vector<float> vertex_data;          // as the buffer holds it (stride / 4 floats per vertex)
    std::uint32_t stride = 12;
    bool xy = false;
    std::vector<std::uint32_t> indices;       // empty = none
    bool u16 = false;
    bool transform = false;
    float m[12] = {};
    std::uint32_t triangle_count = 0;
};

// The world-space triangles a geometry describes, as the reference must see them.
void expand(const Geometry& g, std::uint32_t geometry, std::vector<Tri>& out) {
    const std::uint32_t floats = g.stride / 4;
    for (std::uint32_t t = 0; t < g.triangle_count; ++t) {
        Tri tri{};
        for (int k = 0; k < 3; ++k) {
            const std::uint32_t v = g.indices.empty() ? 3 * t + k : g.indices[3 * t + k];
            float p[3] = {g.vertex_data[v * floats], g.vertex_data[v * floats + 1],
                          g.xy ? 0.0f : g.vertex_data[v * floats + 2]};
            for (int c = 0; c < 3; ++c)
                tri.v[k][c] = g.transform ? g.m[4 * c] * p[0] + g.m[4 * c + 1] * p[1] + g.m[4 * c + 2] * p[2] + g.m[4 * c + 3]
                                          : p[c];
        }
        tri.geometry = geometry;
        tri.primitive = t;
        out.push_back(tri);
    }
}

struct Built {
    std::uint64_t address = 0;
    std::vector<Tri> tris;
};

int build_blas(vktest::Device& gpu, XdxrBuilder& builder, const std::string& dir, const std::string& name,
               const std::vector<Geometry>& geometries, Built& built) {
    std::vector<Buffer> keep;
    std::vector<XdxrTriangles> descs;
    std::uint32_t total = 0;
    for (std::size_t g = 0; g < geometries.size(); ++g) {
        const Geometry& geo = geometries[g];
        Buffer v = make(gpu, geo.vertex_data.size() * 4);
        EXPECT(v.buffer);
        std::memcpy(v.map, geo.vertex_data.data(), geo.vertex_data.size() * 4);
        XdxrTriangles d;
        d.vertices = v.address;
        d.vertex_stride = geo.stride;
        d.xy_only = geo.xy;
        d.triangle_count = geo.triangle_count;
        d.flags = 1;
        keep.push_back(v);
        if (!geo.indices.empty()) {
            Buffer ib = make(gpu, geo.indices.size() * 4);
            EXPECT(ib.buffer);
            if (geo.u16) {
                auto* p = static_cast<std::uint16_t*>(ib.map);
                for (std::size_t i = 0; i < geo.indices.size(); ++i) p[i] = std::uint16_t(geo.indices[i]);
            } else {
                std::memcpy(ib.map, geo.indices.data(), geo.indices.size() * 4);
            }
            d.indices = ib.address;
            d.index_type = geo.u16 ? 1 : 2;
            keep.push_back(ib);
        }
        if (geo.transform) {
            Buffer tb = make(gpu, 48);
            EXPECT(tb.buffer);
            std::memcpy(tb.map, geo.m, 48);
            d.transform = tb.address;
            keep.push_back(tb);
        }
        descs.push_back(d);
        expand(geo, std::uint32_t(g), built.tris);
        total += geo.triangle_count;
    }
    XdxrLayout layout;
    EXPECT(layout.compute(total, false));
    Buffer dst = make(gpu, layout.size + 256), scratch = make(gpu, std::size_t(layout.scratch));
    EXPECT(dst.buffer && scratch.buffer);
    const std::uint64_t base = (dst.address + 255) & ~std::uint64_t(255);   // 256-byte aligned, as D3D12 requires
    bool recorded = false;
    EXPECT(gpu.run([&](VkCommandBuffer cb) {
        recorded = builder.record_blas(cb, layout, descs, base, scratch.address);
        barrier_to_host(gpu, cb);
        return recorded;
    }));
    built.address = base;
    const auto* bytes = static_cast<const char*>(dst.map) + (base - dst.address);
    EXPECT(write_file(dir + "/" + name + ".bin", bytes, layout.size));
    const std::string json = "{\"kind\": \"blas\", \"address\": " + std::to_string(base) +
                             ", \"triangles\": " + tris_json(built.tris) + "}\n";
    EXPECT(write_file(dir + "/" + name + ".json", json.data(), json.size()));
    std::printf("%s: %u triangles, %u bytes, %u nodes\n", name.c_str(), total, layout.size, layout.nodes);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: vk_xdxr_build_test <out dir>\n");
        return 2;
    }
    const std::string dir = argv[1];
    vktest::Device gpu;
    gpu.request_buffer_device_address = true;
    const auto status = gpu.initialize();
    if (status == vktest::DeviceStatus::unavailable || (status == vktest::DeviceStatus::ready && !gpu.buffer_device_address)) {
        std::printf("vk_xdxr_build_test: no Vulkan device with bufferDeviceAddress, skipped\n");
        return 77;
    }
    if (status != vktest::DeviceStatus::ready) return 1;
    XdxrBuilder builder(gpu.device_proc, gpu.device);
    EXPECT(builder.loaded());

    if (argc > 2) {                                   // timing only
        const std::uint32_t n = std::uint32_t(std::strtoul(argv[2], nullptr, 10));
        EXPECT(n > 0);
        std::mt19937 trng(3);
        std::uniform_real_distribution<float> u(-100.0f, 100.0f), j(-0.5f, 0.5f);
        Buffer v = make(gpu, std::size_t(n) * 36);
        EXPECT(v.buffer);
        auto* f = static_cast<float*>(v.map);
        for (std::uint32_t t = 0; t < n; ++t) {
            const float c[3] = {u(trng), u(trng), u(trng)};
            for (int k = 0; k < 9; ++k) f[9 * t + k] = c[k % 3] + j(trng);
        }
        XdxrTriangles d;
        d.vertices = v.address;
        d.triangle_count = n;
        XdxrLayout layout;
        EXPECT(layout.compute(n, false));
        // in VRAM, where the layer keeps acceleration structures (host-visible memory on a discrete GPU is PCIe)
        Buffer dst = make(gpu, layout.size, true), scratch = make(gpu, std::size_t(layout.scratch), true);
        EXPECT(dst.buffer && scratch.buffer);
        double best = 1e30;
        for (int rep = 0; rep < 4; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            EXPECT(gpu.run([&](VkCommandBuffer cb) { return builder.record_blas(cb, layout, {d}, dst.address, scratch.address); }));
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (rep) best = std::min(best, ms);
        }
        std::printf("%u triangles: %.2f ms (best of 3), result %u bytes, scratch %llu bytes, sort size %u\n", n, best,
                    layout.size, static_cast<unsigned long long>(layout.scratch), layout.sort_size);
        return 0;
    }

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> spread(-4.0f, 4.0f), jitter(-0.6f, 0.6f);
    auto soup = [&](std::uint32_t n) {
        Geometry g;
        g.triangle_count = n;
        for (std::uint32_t t = 0; t < n; ++t) {
            const float c[3] = {spread(rng), spread(rng), spread(rng)};
            for (int k = 0; k < 3; ++k)
                for (int a = 0; a < 3; ++a) g.vertex_data.push_back(c[a] + jitter(rng));
        }
        return g;
    };
    auto grid = [&](std::uint32_t n, std::uint32_t floats_per_vertex, bool xy) {
        Geometry g;
        g.stride = 4 * floats_per_vertex;
        g.xy = xy;
        for (std::uint32_t j = 0; j <= n; ++j)
            for (std::uint32_t i = 0; i <= n; ++i) {
                const float p[4] = {float(i) - n * 0.5f, float(j) - n * 0.5f, xy ? 0.0f : 0.25f * float((i * 7 + j * 3) % 5), 9.0f};
                for (std::uint32_t f = 0; f < floats_per_vertex; ++f) g.vertex_data.push_back(p[f]);
            }
        for (std::uint32_t j = 0; j < n; ++j)
            for (std::uint32_t i = 0; i < n; ++i) {
                const std::uint32_t a = j * (n + 1) + i;
                const std::uint32_t q[6] = {a, a + 1, a + n + 2, a, a + n + 2, a + n + 1};
                g.indices.insert(g.indices.end(), q, q + 6);
            }
        g.triangle_count = 2 * n * n;
        return g;
    };

    Built one, five, random, grid16, xy32, moved, multi, empty;
    if (build_blas(gpu, builder, dir, "blas_one", {soup(1)}, one)) return 1;
    if (build_blas(gpu, builder, dir, "blas_five", {soup(5)}, five)) return 1;
    if (build_blas(gpu, builder, dir, "blas_random", {soup(1000)}, random)) return 1;
    Geometry g16 = grid(12, 4, false);
    g16.u16 = true;
    if (build_blas(gpu, builder, dir, "blas_grid_u16_stride16", {g16}, grid16)) return 1;
    if (build_blas(gpu, builder, dir, "blas_grid_u32_xy", {grid(9, 2, true)}, xy32)) return 1;
    Geometry gt = grid(6, 3, false);
    gt.transform = true;
    const float m[12] = {0, 2, 0, 1, 0, 0, 1, -2, 2, 0, 0, 3};   // exact in float: permute, scale by 2, translate
    std::memcpy(gt.m, m, sizeof m);
    if (build_blas(gpu, builder, dir, "blas_transform", {gt}, moved)) return 1;
    Geometry a = soup(40), b = soup(25);
    b.vertex_data[3 * 3 * 7] = std::nanf("");                    // triangle 7 of geometry 1: inactive
    if (build_blas(gpu, builder, dir, "blas_two_geometries", {a, b}, multi)) return 1;
    if (build_blas(gpu, builder, dir, "blas_empty", {}, empty)) return 1;

    // TLAS: instances over blas_random and blas_grid_u16_stride16. An instance names its BLAS by a 40-bit address
    // (256-byte units in 32 bits), as on the console; device addresses can be wider (llvmpipe's are host pointers).
    // The instances hold application-style addresses, device address - bias, as the layer's address window does.
    const std::uint64_t bias = (std::min(random.address, grid16.address) - 0x10000) & ~std::uint64_t(0xffff);
    auto va = [&](std::uint64_t device) { return device ? device - bias : 0; };
    struct Inst { float m[12]; std::uint32_t id_mask; std::uint32_t contrib; std::uint64_t blas; int which; };
    const float c = std::cos(0.7f), s = std::sin(0.7f);
    const Inst insts[] = {
        {{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}, 0xff000000u | 10, 0, va(random.address), 0},
        {{c, -s, 0, 9, s, c, 0, -1, 0, 0, 1, 0.5f}, 0x01000000u | 11, 3, va(grid16.address), 1},
        {{2, 0, 0, -9, 0, 0.5f, 0, 2, 0, 0, 1.5f, 1}, 0x00000000u | 12, 0, va(random.address), 0},   // mask 0
        {{c, 0, s, 0, 0, 1, 0, 9, -s, 0, c, -2}, 0xff000000u | 13, 0, 0, -1},                  // no BLAS
    };
    const std::uint32_t count = sizeof insts / sizeof insts[0];
    for (const Inst& i : insts) EXPECT(i.blas < (std::uint64_t(1) << 40));
    Buffer descs = make(gpu, 64 * count);
    EXPECT(descs.buffer);
    for (std::uint32_t i = 0; i < count; ++i) {
        auto* d = static_cast<char*>(descs.map) + 64 * i;
        std::memcpy(d, insts[i].m, 48);
        std::memcpy(d + 48, &insts[i].id_mask, 4);
        std::memcpy(d + 52, &insts[i].contrib, 4);
        std::memcpy(d + 56, &insts[i].blas, 8);
    }
    XdxrLayout layout;
    EXPECT(layout.compute(count, true));
    Buffer dst = make(gpu, layout.size + 256), scratch = make(gpu, std::size_t(layout.scratch));
    EXPECT(dst.buffer && scratch.buffer);
    const std::uint64_t base = (dst.address + 255) & ~std::uint64_t(255);
    bool recorded = false;
    EXPECT(gpu.run([&](VkCommandBuffer cb) {
        recorded = builder.record_tlas(cb, layout, descs.address, base, scratch.address, bias);
        barrier_to_host(gpu, cb);
        return recorded;
    }));
    EXPECT(write_file(dir + "/tlas.bin", static_cast<const char*>(dst.map) + (base - dst.address), layout.size));
    std::string json = "{\"kind\": \"tlas\", \"address\": " + std::to_string(base) + ", \"bias\": " +
                       std::to_string(bias) + ", \"blas\": [\"blas_random\", " +
                       "\"blas_grid_u16_stride16\"], \"instances\": [";
    for (std::uint32_t i = 0; i < count; ++i) {
        json += i ? ", {" : "{";
        json += "\"transform\": [";
        for (int k = 0; k < 12; ++k) json += (k ? ", " : "") + num(insts[i].m[k]);
        json += "], \"mask\": " + std::to_string(insts[i].id_mask >> 24) + ", \"blas\": " + std::to_string(insts[i].which) + "}";
    }
    json += "]}\n";
    EXPECT(write_file(dir + "/tlas.json", json.data(), json.size()));
    std::printf("tlas: %u instances, %u bytes\nvk_xdxr_build_test: built every case\n", count, layout.size);
    return 0;
}
