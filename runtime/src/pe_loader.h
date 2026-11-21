// Native PE mapper for Xbox Series (x86-64) images.
// Maps an image, applies relocations and binds every import through the shim registry.
// Mapping never runs code from the image. Loader::initialise() is the one call that does: it sets up the
// module's TLS (slot, template and callbacks) and calls a DLL's entry point as DllMain.
#pragma once
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace rt {

enum class Binding { Shim, Host, Module, Unimplemented };

struct ImportRecord {
    std::string dll;
    std::string name;   // "#123" for ordinals
    Binding binding;
};

struct Module {
    std::string name;
    std::filesystem::path path;
    uint8_t* base = nullptr;
    uint64_t preferred_base = 0;
    uint32_t size = 0;
    uint32_t entry_rva = 0;
    bool relocated = false;

    // Filled in by initialise(). The TLS slot and block are owned by the loader and released with it.
    void* tls_block = nullptr;
    uint64_t tls_block_size = 0;
    unsigned tls_slot = 0;
    bool has_tls_slot = false;
    bool initialised = false;
    std::vector<ImportRecord> imports;
    std::map<std::string, void*> exports;           // by name, plus "#ordinal"
    std::map<std::string, std::string> forwarders;  // same keys, holding "module.symbol" for re-exports
};

class Loader {
public:
    explicit Loader(std::filesystem::path game_dir) : game_dir_(std::move(game_dir)) {}
    ~Loader();

    // Loads an image and (recursively) the bundled modules it imports. Throws std::runtime_error on failure.
    Module& load(const std::filesystem::path& image);

    const std::vector<Module*>& modules() const { return order_; }

    // Runs each loaded module's TLS callbacks and then its DllMain(DLL_PROCESS_ATTACH), dependencies first.
    // Only DLLs get DllMain called: an EXE's entry point is the caller's to run. Throws on a corrupt TLS
    // directory, a TLS callback outside its image, or a DllMain that returns FALSE.
    void initialise();

private:
    void initialise(Module& m);
    Module& map_image(const std::filesystem::path& path);
    void bind_imports(Module& m);
    void* resolve(Module& m, const std::string& dll, const std::string& name, uint16_t ordinal, bool by_ordinal);
    void* resolve_forwarder(const std::string& spec, Binding& binding, int depth);
    void parse_exports(Module& m);
    void protect_sections(Module& m);

    std::filesystem::path game_dir_;
    std::map<std::string, Module*> by_name_;  // lower-case file name
    std::vector<Module*> order_;
};

std::string lower(std::string s);

}  // namespace rt
