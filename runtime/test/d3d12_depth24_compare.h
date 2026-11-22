// D3D-47: D24_UNORM_S8_UINT against the host PC runtime (or WARP), byte for byte. Included by
// d3d12_vulkan_device_test.cpp after d3d12_copy_compare.h, whose Run and host selection (LEONIDA_HOST_ADAPTER) it uses.
//
// On a GPU without D24 the layer stores D32_SFLOAT_S8_UINT (depth24.h): a clear must leave the 24-bit value D3D's
// quantisation gives, and a footprint copy converts. The copy paths are compared with random 24-bit words in
// d3d12_copy_compare.h; this adds the clears (whole view, and a rectangle per value) with values on and between the
// 24-bit steps, a stencil value, and the depth and stencil planes read back through a footprint.
#pragma once
#include "d3d12_copy_compare.h"

#include <cmath>

namespace copycmp {

// Clear values: exact steps, ties (0.5 * 16777215 is a half), steps of the float grid around them, and the ends.
inline std::vector<float> depth24_clear_values() {
    std::vector<float> v = {0.0f, 1.0f, 0.5f, 0.25f, 0.75f, 1.0f / 3.0f, 2.0f / 3.0f, 0.1f, 0.9f, 0.123456789f,
                            1e-8f, 3e-8f, 6e-8f, 1.0e-7f, 1e-3f, 0.999999f, 0.99999994f, 0.9999999f, 0.5000001f,
                            0.49999997f, 8388607.0f / 16777215.0f, 8388608.0f / 16777215.0f, 1.0f / 16777215.0f,
                            0.5f / 16777215.0f, 1.5f / 16777215.0f, 2.5f / 16777215.0f, 16777214.5f / 16777215.0f,
                            16777214.0f / 16777215.0f, 5.9604645e-8f, 8.940697e-8f, 2.9802322e-8f, 1.0f / 65535.0f};
    // 1/64 steps, uniform random values, values at every scale (a random exponent), and the floats around random
    // half steps ((k + 0.5) / 16777215, where the rounding rule shows) and one step of the float grid either side.
    for (int i = 1; i < 64; ++i) v.push_back(float(i) / 64.0f + 1e-9f);
    std::uint32_t x = 12345;
    const auto next = [&] { return x = x * 1664525u + 1013904223u; };
    for (int i = 0; i < 1024; ++i) v.push_back(float(next() >> 8) / 16777216.0f);
    for (int i = 0; i < 1024; ++i) v.push_back(std::ldexp(1.0f + float(next() >> 9) / 8388608.0f, -int(next() >> 27) - 1));
    for (int i = 0; i < 1024; ++i) {
        const float half = float((double(next() >> 8) + 0.5) / 16777215.0);
        v.push_back(half);
        v.push_back(std::nextafter(half, 0.0f));
        v.push_back(std::nextafter(half, 1.0f));
    }
    return v;
}

// One scenario on one device: every value cleared into a 4x4 texture (a whole-view clear) and into a column of a wide
// one (a rectangle clear), the planes of both read back. out.bytes holds them, regions say where.
inline void run_depth24_clears(Run& run, ID3D12Device* host, const std::vector<float>& values, Outcome& out) {
    const UINT count = UINT(values.size());
    const UINT whole = std::min<UINT>(count, 160);  // whole-view clears, one texture each; the wide one takes all
    D3D12_RESOURCE_DESC tiny{};
    tiny.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    tiny.Width = 4;
    tiny.Height = 4;
    tiny.DepthOrArraySize = 1;
    tiny.MipLevels = 1;
    tiny.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    tiny.SampleDesc.Count = 1;
    tiny.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_RESOURCE_DESC wide = tiny;
    wide.Width = count;

    // Footprints come from the host: the layer's must be the same, and are compared elsewhere.
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT tiny_foot[2], wide_foot[2];
    UINT tiny_rows[2], wide_rows[2];
    UINT64 tiny_row_bytes[2], wide_row_bytes[2], tiny_total = 0, wide_total = 0;
    host->GetCopyableFootprints(&tiny, 0, 2, 0, tiny_foot, tiny_rows, tiny_row_bytes, &tiny_total);
    host->GetCopyableFootprints(&wide, 0, 2, 0, wide_foot, wide_rows, wide_row_bytes, &wide_total);
    const UINT64 tiny_stride = (tiny_total + 511) / 512 * 512, wide_stride = (wide_total + 511) / 512 * 512;
    const UINT64 wide_base = tiny_stride * whole;
    const UINT64 size = wide_base + wide_stride;

    Ptr<ID3D12Resource> rb, wide_tex;
    std::vector<Ptr<ID3D12Resource>> tex(whole);
    Ptr<ID3D12DescriptorHeap> heap;
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, whole + 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    if (!run.buffer(D3D12_HEAP_TYPE_READBACK, size, D3D12_RESOURCE_STATE_COPY_DEST, rb) ||
        !run.texture(wide, D3D12_RESOURCE_STATE_DEPTH_WRITE, wide_tex) ||
        FAILED(run.device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p)))) {
        out.error = "resource creation failed";
        return;
    }
    for (UINT k = 0; k < whole; ++k)
        if (!run.texture(tiny, D3D12_RESOURCE_STATE_DEPTH_WRITE, tex[k])) {
            out.error = "resource creation failed";
            return;
        }
    std::uint8_t* mapped = nullptr;
    if (FAILED(rb->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) {
        out.error = "Map failed";
        return;
    }
    std::memset(mapped, 0xcd, size_t(size));

    const UINT step = run.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    // The layer's proxy heap answers in the Xbox shape (slot 9 returns the 8-byte handle in RAX); the host's is the PC's.
    D3D12_CPU_DESCRIPTOR_HANDLE start{};
    if (run.ours) {
        using XboxHandle = std::uint64_t(WINAPI*)(void*);
        start.ptr = SIZE_T(reinterpret_cast<XboxHandle>((*reinterpret_cast<void***>(heap.p))[9])(heap.p));
    } else {
        start = heap->GetCPUDescriptorHandleForHeapStart();
    }
    const auto handle = [&](UINT i) {
        D3D12_CPU_DESCRIPTOR_HANDLE h = start;
        h.ptr += size_t(i) * step;
        return h;
    };
    D3D12_DEPTH_STENCIL_VIEW_DESC dsv{};
    dsv.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    for (UINT k = 0; k < whole; ++k) {
        run.device->CreateDepthStencilView(tex[k].p, &dsv, handle(k));
    }
    run.device->CreateDepthStencilView(wide_tex.p, &dsv, handle(whole));

    // Whole-view clears, one per texture; the wide texture first gets a stencil-only clear, then a column per value.
    for (UINT k = 0; k < whole; ++k)
        run.list->ClearDepthStencilView(handle(k), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, values[k],
                                        UINT8(0x40 + k % 64), 0, nullptr);
    run.list->ClearDepthStencilView(handle(whole), D3D12_CLEAR_FLAG_STENCIL, 0.0f, 0x11, 0, nullptr);
    for (UINT k = 0; k < count; ++k) {
        const D3D12_RECT column{LONG(k), 0, LONG(k + 1), 4};
        run.list->ClearDepthStencilView(handle(whole), D3D12_CLEAR_FLAG_DEPTH, values[k], 0, 1, &column);
    }
    for (UINT k = 0; k < whole; ++k) run.barrier(tex[k].p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE);
    run.barrier(wide_tex.p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE);

    const auto copy_out = [&](ID3D12Resource* texture, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT* foot, UINT64 base,
                              const UINT* rows, const UINT64* row_bytes, const char* what, UINT index) {
        for (UINT plane = 0; plane < 2; ++plane) {
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT at = foot[plane];
            at.Offset += base;
            const auto dst = foot_loc(rb.p, at);
            const auto src = sub_loc(texture, plane);
            run.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            char label[96];
            std::snprintf(label, sizeof(label), "%s %u (value %.9g) %s plane", what, index,
                          index < values.size() ? double(values[index]) : 0.0, plane ? "stencil" : "depth");
            out.regions.push_back({label, size_t(at.Offset), at.Footprint.RowPitch, size_t(row_bytes[plane]), rows[plane],
                                   1, plane == 0});
        }
    };
    for (UINT k = 0; k < whole; ++k) copy_out(tex[k].p, tiny_foot, tiny_stride * k, tiny_rows, tiny_row_bytes, "whole-view clear", k);
    copy_out(wide_tex.p, wide_foot, wide_base, wide_rows, wide_row_bytes, "rectangle clears", 0);
    out.error = run.submit();
    out.bytes.assign(mapped, mapped + size_t(size));
    rb->Unmap(0, nullptr);
}

// Returns the number of differences, or -1 with no host device.
inline int compare_depth24(ID3D12Device* ours) {
    Ptr<ID3D12Device> host;
    std::string adapter;
    if (!create_host(host, adapter)) return -1;
    const std::vector<float> values = depth24_clear_values();
    Outcome a, b;
    Run ours_run, host_run;
    if (!ours_run.init(ours, true) || !host_run.init(host.p, false)) {
        std::fprintf(stderr, "depth24 clears: could not create the queue and list\n");
        return 1;
    }
    run_depth24_clears(ours_run, host.p, values, a);
    run_depth24_clears(host_run, host.p, values, b);
    const int r = compare("D24 clears", a, b);
    if (!r) std::printf("depth24 clears: %zu values, whole-view and rectangle clears match the host (%zu regions)\n",
                        values.size(), b.regions.size());
    return r;
}

}  // namespace copycmp
