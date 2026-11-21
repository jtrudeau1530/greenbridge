// Shared IUnknown/ID3D12Object/ID3D12DeviceChild implementation for the Vulkan-backed D3D12 COM objects
// (d3d12_com.cpp, d3d12_vulkan_device.cpp). Private to the runtime; include after the generated d3d12_stubs.h.
#pragma once

#include <windows.h>
#include <d3d12.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

namespace rt::d3d12 {

void stub(const char* method);

// IUnknown and ID3D12Object for one implementation class. QueryInterface answers the listed public interfaces
// and the class's private Derived::kTag (which returns the implementation itself); anything else is logged with
// its IID and refused, as docs/d3d12_vulkan.md section 3.1 asks.
template<class Derived, class Stubs, class... Interfaces>
class Object : public Stubs {
public:
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG left = --refs_;
        if (!left) delete static_cast<Derived*>(this);
        return left;
    }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (riid == Derived::kTag) {
            *out = static_cast<Derived*>(this);
        } else if (riid == __uuidof(IUnknown) || ((riid == __uuidof(Interfaces)) || ...)) {
            *out = static_cast<Stubs*>(this);  // one inheritance chain: every interface shares this pointer
        } else {
            std::fprintf(stderr,
                         "[rt] d3d12: QueryInterface: unknown IID {%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x}\n",
                         static_cast<unsigned long>(riid.Data1), riid.Data2, riid.Data3, riid.Data4[0], riid.Data4[1],
                         riid.Data4[2], riid.Data4[3], riid.Data4[4], riid.Data4[5], riid.Data4[6], riid.Data4[7]);
            return E_NOINTERFACE;
        }
        AddRef();
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID guid, UINT* size, void* data) override {
        if (!size) return E_INVALIDARG;
        std::lock_guard lock(data_mutex_);
        for (const auto& [key, bytes] : data_) {
            if (key != guid) continue;
            if (data) {
                if (*size < bytes.size()) return DXGI_ERROR_MORE_DATA;
                std::memcpy(data, bytes.data(), bytes.size());
            }
            *size = static_cast<UINT>(bytes.size());
            return S_OK;
        }
        *size = 0;
        return DXGI_ERROR_NOT_FOUND;
    }

    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID guid, UINT size, const void* data) override {
        std::lock_guard lock(data_mutex_);
        std::erase_if(data_, [&](const auto& entry) { return entry.first == guid; });
        if (data && size) {
            const auto* bytes = static_cast<const std::uint8_t*>(data);
            data_.emplace_back(guid, std::vector<std::uint8_t>(bytes, bytes + size));
        }
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE SetName(LPCWSTR) override { return S_OK; }

    // The implementation behind a public interface pointer, or null when it is not a T. No reference is added.
    template<class T>
    static T* from(IUnknown* object) {
        if (!object) return nullptr;
        void* implementation = nullptr;
        if (FAILED(object->QueryInterface(T::kTag, &implementation))) return nullptr;
        static_cast<T*>(implementation)->Release();  // the caller holds its own reference
        return static_cast<T*>(implementation);
    }

protected:
    virtual ~Object() = default;

private:
    std::atomic<ULONG> refs_{1};
    std::mutex data_mutex_;
    std::vector<std::pair<GUID, std::vector<std::uint8_t>>> data_;
};

// An ID3D12DeviceChild: holds a reference to its device, as D3D12 children do, and GetDevice returns it. When the
// device sits behind the Xbox-ABI proxy, the reference is to the proxy, so the game never sees the inner device.
template<class Derived, class Stubs, class... Interfaces>
class Child : public Object<Derived, Stubs, Interfaces...> {
public:
    void attach_device(IUnknown* device) {
        if (device) device->AddRef();
        if (device_) device_->Release();
        device_ = device;
    }

    HRESULT STDMETHODCALLTYPE GetDevice(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        return device_ ? device_->QueryInterface(riid, out) : E_NOINTERFACE;
    }

protected:
    ~Child() override {
        if (device_) device_->Release();
    }

private:
    IUnknown* device_ = nullptr;
};

template<class T>
HRESULT hand_out(T* object, REFIID riid, void** out) {
    const HRESULT hr = object->QueryInterface(riid, out);
    object->Release();
    return hr;
}

}  // namespace rt::d3d12
