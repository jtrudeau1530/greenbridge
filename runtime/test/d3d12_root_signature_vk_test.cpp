#include "../shims/d3d12_com.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

void push_u32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    out.push_back(static_cast<std::uint8_t>(v & 0xFFu));
    out.push_back(static_cast<std::uint8_t>((v >> 8) & 0xFFu));
    out.push_back(static_cast<std::uint8_t>((v >> 16) & 0xFFu));
    out.push_back(static_cast<std::uint8_t>((v >> 24) & 0xFFu));
}

// Builds a minimal bare RTS0 payload (accepted by rt::rootsig::parse_rts0):
// version = 1 (V1_0), num_params, params_offset = 24, num_samplers = 0, samplers_offset = 24, flags = 0.
std::vector<std::uint8_t> make_rts0_with_one_constant(std::uint32_t num_values) {
    std::vector<std::uint8_t> blob;
    push_u32(blob, 1);   // version = 1 (V1_0)
    push_u32(blob, 1);   // num_params = 1
    push_u32(blob, 24);  // params_offset = 24
    push_u32(blob, 0);   // num_samplers = 0
    push_u32(blob, 24);  // samplers_offset = 24
    push_u32(blob, 0);   // flags = 0

    // Parameter 0 entry: {d3d_type = 1 (32BIT_CONSTANTS), visibility = 0 (ALL), offset = 36}
    push_u32(blob, 1);
    push_u32(blob, 0);
    push_u32(blob, 36);

    // Parameter 0 payload at offset 36: {shader_register = 0, register_space = 0, num_values}
    push_u32(blob, 0);
    push_u32(blob, 0);
    push_u32(blob, num_values);
    return blob;
}

std::vector<std::uint8_t> make_empty_rts0() {
    std::vector<std::uint8_t> blob;
    push_u32(blob, 1);   // version = 1 (V1_0)
    push_u32(blob, 0);   // num_params = 0
    push_u32(blob, 24);  // params_offset = 24
    push_u32(blob, 0);   // num_samplers = 0
    push_u32(blob, 24);  // samplers_offset = 24
    push_u32(blob, 0);   // flags = 0
    return blob;
}

}  // namespace

int main() {
    ID3D12Device* device = nullptr;
    const HRESULT dev_hr = rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device));
    if (FAILED(dev_hr) || !device) {
        std::fprintf(stderr, "FAIL: create_vulkan_device_standalone hr=0x%08lX\n",
                     static_cast<unsigned long>(dev_hr));
        return 1;
    }

    // 1. Negative tests: null/empty/invalid blobs must fail cleanly with E_INVALIDARG.
    ID3D12RootSignature* bad_sig = nullptr;
    if (device->CreateRootSignature(0, nullptr, 16, IID_PPV_ARGS(&bad_sig)) != E_INVALIDARG || bad_sig != nullptr) {
        std::fprintf(stderr, "FAIL: nullptr blob should return E_INVALIDARG\n");
        device->Release();
        return 1;
    }

    const std::uint8_t garbage[4] = {0xDE, 0xAD, 0xBE, 0xEF};
    if (device->CreateRootSignature(0, garbage, sizeof(garbage), IID_PPV_ARGS(&bad_sig)) != E_INVALIDARG ||
        bad_sig != nullptr) {
        std::fprintf(stderr, "FAIL: truncated/garbage blob should return E_INVALIDARG\n");
        device->Release();
        return 1;
    }

    // 2. Valid empty RTS0 root signature.
    const auto empty_blob = make_empty_rts0();
    ID3D12RootSignature* empty_sig = nullptr;
    HRESULT hr = device->CreateRootSignature(0, empty_blob.data(), empty_blob.size(), IID_PPV_ARGS(&empty_sig));
    if (FAILED(hr) || !empty_sig) {
        std::fprintf(stderr, "FAIL: empty CreateRootSignature hr=0x%08lX\n", static_cast<unsigned long>(hr));
        device->Release();
        return 1;
    }
    empty_sig->Release();

    // 3. Valid RTS0 root signature with 4x 32-bit root constants.
    const auto const_blob = make_rts0_with_one_constant(4);
    ID3D12RootSignature* const_sig = nullptr;
    hr = device->CreateRootSignature(0, const_blob.data(), const_blob.size(), IID_PPV_ARGS(&const_sig));
    if (FAILED(hr) || !const_sig) {
        std::fprintf(stderr, "FAIL: const CreateRootSignature hr=0x%08lX\n", static_cast<unsigned long>(hr));
        device->Release();
        return 1;
    }

    // Verify COM QueryInterface / Release roundtrip.
    ID3D12DeviceChild* as_child = nullptr;
    hr = const_sig->QueryInterface(IID_PPV_ARGS(&as_child));
    if (FAILED(hr) || !as_child) {
        std::fprintf(stderr, "FAIL: QueryInterface(ID3D12DeviceChild) hr=0x%08lX\n", static_cast<unsigned long>(hr));
        const_sig->Release();
        device->Release();
        return 1;
    }
    as_child->Release();
    const_sig->Release();

    device->Release();
    std::printf("PASS: d3d12_root_signature_vk_test (negative, empty, constants + VkPipelineLayout)\n");
    return 0;
}
