// Tests for the native PE loader. Every image here is built byte by byte, so nothing from the game is needed and
// nothing is read from the repo: the only files written go to a temporary directory.
//
// What it pins down: a valid image maps exactly as its headers say, and a truncated or corrupt one is rejected with
// a message instead of being followed out of bounds. The truncation case covers every length of a valid image.
#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/pe_loader.h"

namespace {

int g_checks = 0;
int g_failed = 0;
unsigned g_files = 0;
std::filesystem::path g_dir;

// Sentinels that the synthesised TLS callback and DllMain stubs write, so a test can see execution happen.
uint32_t g_tls_ran = 0;
uint32_t g_main_ran = 0;

void pass(const char* what) {
    ++g_checks;
    std::printf("ok    %s\n", what);
}

void failed(const char* what, const std::string& why) {
    ++g_checks;
    ++g_failed;
    std::printf("FAIL  %s: %s\n", what, why.c_str());
}

// A minimal but valid x64 image: DOS stub, PE32+ headers, one .text section holding a recognisable byte pattern.
// No imports, exports or relocations unless a case adds them.
std::vector<uint8_t> make_image() {
    std::vector<uint8_t> bytes(0x400, 0);
    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(bytes.data());
    dos->e_magic = IMAGE_DOS_SIGNATURE;
    dos->e_lfanew = 0x80;
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(bytes.data() + dos->e_lfanew);
    nt->Signature = IMAGE_NT_SIGNATURE;
    nt->FileHeader.Machine = IMAGE_FILE_MACHINE_AMD64;
    nt->FileHeader.NumberOfSections = 1;
    nt->FileHeader.SizeOfOptionalHeader = sizeof(IMAGE_OPTIONAL_HEADER64);
    nt->OptionalHeader.Magic = IMAGE_NT_OPTIONAL_HDR64_MAGIC;
    nt->OptionalHeader.ImageBase = 0x140000000;
    nt->OptionalHeader.AddressOfEntryPoint = 0x1000;
    nt->OptionalHeader.SizeOfHeaders = 0x200;
    nt->OptionalHeader.SizeOfImage = 0x2000;
    nt->OptionalHeader.Subsystem = IMAGE_SUBSYSTEM_WINDOWS_CUI;
    auto* sec = IMAGE_FIRST_SECTION(nt);
    std::memcpy(sec->Name, ".text", 6);
    sec->VirtualAddress = 0x1000;
    sec->Misc.VirtualSize = 0x200;
    sec->PointerToRawData = 0x200;
    sec->SizeOfRawData = 0x200;
    sec->Characteristics = IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ;
    for (int i = 0; i < 0x200; ++i) bytes[0x200 + i] = static_cast<uint8_t>(i);
    return bytes;
}

IMAGE_NT_HEADERS64* nt_of(std::vector<uint8_t>& bytes) {
    return reinterpret_cast<IMAGE_NT_HEADERS64*>(bytes.data() + 0x80);
}

// The section's bytes as they sit in the file, ready to compare against what was mapped.
const uint8_t* text_raw(const std::vector<uint8_t>& bytes) { return bytes.data() + 0x200; }

std::filesystem::path write_image(const std::vector<uint8_t>& bytes) {
    auto p = g_dir / (std::to_string(++g_files) + ".bin");
    std::ofstream out(p, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + p.string());
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<long long>(bytes.size()));
    return p;
}

std::filesystem::path write_image_as(const std::vector<uint8_t>& bytes, const std::string& name) {
    auto p = g_dir / name;
    std::ofstream out(p, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + p.string());
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<long long>(bytes.size()));
    return p;
}

// An image whose only export, "fwd", forwards to `target`: the address-table entry points at a string inside the
// export directory, which is how a forwarder is encoded.
std::vector<uint8_t> make_forwarder_image(const char* target, uint64_t image_base) {
    auto img = make_image();
    std::fill(img.begin() + 0x200, img.begin() + 0x400, 0);  // real tables end in a zero descriptor
    nt_of(img)->OptionalHeader.ImageBase = image_base;
    auto* ed = reinterpret_cast<IMAGE_EXPORT_DIRECTORY*>(img.data() + 0x200);
    ed->Base = 1;
    ed->NumberOfFunctions = 1;
    ed->AddressOfFunctions = 0x1100;
    ed->NumberOfNames = 1;
    ed->AddressOfNames = 0x1120;
    ed->AddressOfNameOrdinals = 0x1140;
    *reinterpret_cast<uint32_t*>(img.data() + 0x300) = 0x1160;  // funcs[0]: inside the export directory
    *reinterpret_cast<uint32_t*>(img.data() + 0x320) = 0x1180;  // the export's name
    *reinterpret_cast<uint16_t*>(img.data() + 0x340) = 0;
    std::memcpy(img.data() + 0x360, target, std::strlen(target) + 1);
    std::memcpy(img.data() + 0x380, "fwd\0", 4);
    auto& dir = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
    dir.VirtualAddress = 0x1000;
    dir.Size = 0x200;  // covers the forwarder string at RVA 0x1160
    return img;
}

// An image importing `dll`!`symbol` by name.
std::vector<uint8_t> make_import_image(const char* dll, const char* symbol) {
    auto img = make_image();
    std::fill(img.begin() + 0x200, img.begin() + 0x400, 0);
    auto* d = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(img.data() + 0x200);
    d->Name = 0x1060;  // past the zero descriptor that ends the table
    d->OriginalFirstThunk = 0x1100;
    d->FirstThunk = 0x1180;  // the address table the loader fills
    *reinterpret_cast<uint64_t*>(img.data() + 0x300) = 0x1120;  // hint/name
    *reinterpret_cast<uint16_t*>(img.data() + 0x320) = 0;
    std::memcpy(img.data() + 0x322, symbol, std::strlen(symbol) + 1);
    std::memcpy(img.data() + 0x260, dll, std::strlen(dll) + 1);
    auto& dir = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    dir.VirtualAddress = 0x1000;
    dir.Size = sizeof(IMAGE_IMPORT_DESCRIPTOR);
    return img;
}

void* host_symbol(const char* dll, const char* name) {
    HMODULE h = GetModuleHandleA(dll);
    return h ? reinterpret_cast<void*>(GetProcAddress(h, name)) : nullptr;
}

// 0 when the image loaded, 1 when it was rejected. A crash here is a test failure, which is the point.
int load(const std::filesystem::path& p, std::string& error) {
    try {
        rt::Loader loader(g_dir);
        loader.load(p);
        return 0;
    } catch (const std::exception& e) {
        error = e.what();
        return 1;
    }
}

void expect_loads(const char* what, const std::vector<uint8_t>& bytes) {
    auto p = write_image(bytes);
    try {
        rt::Loader loader(g_dir);
        rt::Module& m = loader.load(p);
        if (!m.base) return failed(what, "the image mapped to a null base");
        if (m.size != 0x2000) return failed(what, "SizeOfImage came back as " + std::to_string(m.size));
        if (std::memcmp(m.base + 0x1000, text_raw(bytes), 0x200) != 0)
            return failed(what, "the mapped section doesn't match the file");
        pass(what);
    } catch (const std::exception& e) {
        failed(what, std::string("rejected a valid image: ") + e.what());
    }
}

void expect_rejects(const char* what, const std::vector<uint8_t>& bytes) {
    std::string error;
    if (load(write_image(bytes), error) == 0) return failed(what, "the image loaded");
    pass(what);
}

void test_valid_image() { expect_loads("a valid image maps with its section intact", make_image()); }

void test_header_bounds() {
    {
        auto img = make_image();
        nt_of(img)->OptionalHeader.ImageBase = 0x140000000;
        reinterpret_cast<IMAGE_DOS_HEADER*>(img.data())->e_lfanew = 0x10000;
        expect_rejects("e_lfanew past the end of the file", img);
    }
    {
        auto img = make_image();
        reinterpret_cast<IMAGE_DOS_HEADER*>(img.data())->e_lfanew = -1;
        expect_rejects("a negative e_lfanew", img);
    }
    {
        auto img = make_image();
        img.resize(0x80 + 0x40);  // cut inside the NT headers
        expect_rejects("NT headers truncated by the file size", img);
    }
    {
        auto img = make_image();
        nt_of(img)->OptionalHeader.SizeOfImage = 0;
        expect_rejects("SizeOfImage of 0", img);
    }
    {
        auto img = make_image();
        nt_of(img)->OptionalHeader.SizeOfHeaders = 0x4000;
        expect_rejects("SizeOfHeaders larger than SizeOfImage", img);
    }
    {
        auto img = make_image();
        nt_of(img)->FileHeader.SizeOfOptionalHeader = 0;
        expect_rejects("an optional header too small for PE32+", img);
    }
    {
        // A PE32 image: the magic says 32-bit, so reading it through a PE32+ header would be wrong.
        auto img = make_image();
        nt_of(img)->OptionalHeader.Magic = IMAGE_NT_OPTIONAL_HDR32_MAGIC;
        expect_rejects("a PE32 image in place of PE32+", img);
    }
    {
        auto img = make_image();
        nt_of(img)->FileHeader.Machine = IMAGE_FILE_MACHINE_I386;
        expect_rejects("an i386 machine type", img);
    }
    {
        auto img = make_image();
        nt_of(img)->FileHeader.NumberOfSections = 200;  // section table past the end of the file
        expect_rejects("a section table past the end of the file", img);
    }
}

void test_section_bounds() {
    {
        auto img = make_image();
        IMAGE_FIRST_SECTION(nt_of(img))->VirtualAddress = 0x1F00;  // 0x1F00 + 0x200 > SizeOfImage
        expect_rejects("a section that would be written past the image", img);
    }
    {
        auto img = make_image();
        IMAGE_FIRST_SECTION(nt_of(img))->PointerToRawData = 0x3F0;  // 0x3F0 + 0x200 > file size
        expect_rejects("a section that reads past the end of the file", img);
    }
}

void test_truncation() {
    // Every length of a valid image: no length below the full one may load, and none may crash.
    auto full = make_image();
    for (size_t len = 0; len < full.size(); ++len) {
        std::vector<uint8_t> img(full.begin(), full.begin() + static_cast<long long>(len));
        std::string error;
        if (load(write_image(img), error) == 0)
            return failed("every truncation of a valid image is rejected",
                          "a " + std::to_string(len) + "-byte image loaded");
    }
    pass("every truncation of a valid image is rejected");
}

void test_relocations() {
    // ImageBase 0 makes the loader relocate, which is what exercises the table walk.
    auto relocated = make_image();
    auto* nt = nt_of(relocated);
    nt->OptionalHeader.ImageBase = 0;
    auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_BASERELOC];
    dir.VirtualAddress = 0x1000;
    dir.Size = 0x0A;  // header (8) + one DIR64 entry (2)
    auto* block = reinterpret_cast<IMAGE_BASE_RELOCATION*>(relocated.data() + 0x200);
    block->VirtualAddress = 0x1000;
    block->SizeOfBlock = 0x0A;
    auto* entries = reinterpret_cast<uint16_t*>(block + 1);
    entries[0] = IMAGE_REL_BASED_DIR64 << 12;  // type DIR64, offset 0
    {
        // A correct table has to keep working: the pointer at 0x1000 moves by the difference from ImageBase.
        auto img = relocated;
        auto qword = [&] {
            uint64_t v;
            std::memcpy(&v, img.data() + 0x200, sizeof(v));
            return v;
        }();
        auto p = write_image(img);
        try {
            rt::Loader loader(g_dir);
            rt::Module& m = loader.load(p);
            uint64_t moved;
            std::memcpy(&moved, m.base + 0x1000, sizeof(moved));
            int64_t delta = reinterpret_cast<int64_t>(m.base) - 0;
            if (moved != qword + static_cast<uint64_t>(delta))
                failed("a valid relocation is applied", "the pointer wasn't moved by the load delta");
            else
                pass("a valid relocation is applied");
        } catch (const std::exception& e) {
            failed("a valid relocation is applied", e.what());
        }
    }
    {
        auto img = relocated;
        auto* d = &nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_BASERELOC];
        d->VirtualAddress = 0x1FF8;  // the table itself runs past the image
        expect_rejects("a relocation table past the image", img);
    }
    {
        auto img = relocated;
        auto* b = reinterpret_cast<IMAGE_BASE_RELOCATION*>(img.data() + 0x200);
        b->SizeOfBlock = 2;  // smaller than its own header
        expect_rejects("a relocation block shorter than its header", img);
    }
    {
        auto img = relocated;
        auto* b = reinterpret_cast<IMAGE_BASE_RELOCATION*>(img.data() + 0x200);
        b->VirtualAddress = 0x1FFA;  // the entry's 8 bytes end past the image
        expect_rejects("a relocation entry past the image", img);
    }
}

void test_exports() {
    {
        auto img = make_image();
        auto& dir = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
        dir.VirtualAddress = 0x1FF0;
        dir.Size = 0x28;
        expect_rejects("an export directory past the image", img);
    }
    {
        // A directory whose address table runs off the end of the image.
        auto img = make_image();
        auto* ed = reinterpret_cast<IMAGE_EXPORT_DIRECTORY*>(img.data() + 0x200);
        ed->NumberOfFunctions = 4;
        ed->AddressOfFunctions = 0x1FF8;
        ed->NumberOfNames = 0;
        auto& dir = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(IMAGE_EXPORT_DIRECTORY);
        expect_rejects("an export address table past the image", img);
    }
    {
        // A name whose ordinal doesn't index the address table.
        auto img = make_image();
        auto* ed = reinterpret_cast<IMAGE_EXPORT_DIRECTORY*>(img.data() + 0x200);
        ed->NumberOfFunctions = 1;
        ed->AddressOfFunctions = 0x1100;
        ed->NumberOfNames = 1;
        ed->AddressOfNames = 0x1120;
        ed->AddressOfNameOrdinals = 0x1140;
        auto* funcs = reinterpret_cast<uint32_t*>(img.data() + 0x300);
        funcs[0] = 0x1000;
        auto* name_rva = reinterpret_cast<uint32_t*>(img.data() + 0x320);
        *name_rva = 0x1160;
        auto* ord = reinterpret_cast<uint16_t*>(img.data() + 0x340);
        *ord = 7;  // past NumberOfFunctions
        std::memcpy(img.data() + 0x360, "f\0", 2);
        auto& dir = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(IMAGE_EXPORT_DIRECTORY);
        expect_rejects("an export name with an ordinal past the address table", img);
    }
}

void test_imports() {
    {
        auto img = make_image();
        auto& dir = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
        dir.VirtualAddress = 0x1FF0;
        dir.Size = sizeof(IMAGE_IMPORT_DESCRIPTOR);
        expect_rejects("an import table past the image", img);
    }
    {
        // A descriptor whose DLL name sits past the end of the allocation.
        auto img = make_image();
        auto* d = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(img.data() + 0x200);
        d->Name = 0x2000;  // exactly SizeOfImage: no byte of the name is inside the image
        d->FirstThunk = 0;
        d->OriginalFirstThunk = 0;
        auto& dir = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(IMAGE_IMPORT_DESCRIPTOR);
        expect_rejects("an imported DLL name past the image", img);
    }
    {
        // A thunk table with no terminating zero entry: the walk has to stop at the end of the image.
        auto img = make_image();
        nt_of(img)->OptionalHeader.SizeOfImage = 0x1200;  // the image ends where the section does
        auto* d = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(img.data() + 0x200);
        std::memcpy(img.data() + 0x220, "xmem.dll\0", 9);
        d->Name = 0x1020;
        d->FirstThunk = 0x11F8;  // room for one entry before the image ends
        d->OriginalFirstThunk = 0x11F8;
        auto* thunk = reinterpret_cast<IMAGE_THUNK_DATA64*>(img.data() + 0x3F8);  // file offset of RVA 0x11F8
        thunk->u1.AddressOfData = 0x140;  // a name RVA that is also not where a hint sits
        auto& dir = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(IMAGE_IMPORT_DESCRIPTOR);
        expect_rejects("an import thunk table that runs past the image", img);
    }
}

void test_delay_imports() {
    {
        auto img = make_image();
        auto& dir = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT];
        dir.VirtualAddress = 0x1FF0;  // the descriptor itself runs past the image
        dir.Size = sizeof(IMAGE_DELAYLOAD_DESCRIPTOR);
        expect_rejects("a delay-import table past the image", img);
    }
    {
        // A descriptor whose DLL name sits past the end of the allocation.
        auto img = make_image();
        auto* d = reinterpret_cast<IMAGE_DELAYLOAD_DESCRIPTOR*>(img.data() + 0x200);
        d->DllNameRVA = 0x2000;  // exactly SizeOfImage
        auto& dir = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(IMAGE_DELAYLOAD_DESCRIPTOR);
        expect_rejects("a delay-imported DLL name past the image", img);
    }
    {
        // A delay-import thunk table with no terminating zero entry.
        auto img = make_image();
        nt_of(img)->OptionalHeader.SizeOfImage = 0x1200;
        auto* d = reinterpret_cast<IMAGE_DELAYLOAD_DESCRIPTOR*>(img.data() + 0x200);
        std::memcpy(img.data() + 0x220, "xmem.dll\0", 9);
        d->DllNameRVA = 0x1020;
        d->ImportNameTableRVA = 0x11F8;  // one entry fits before the image ends
        d->ImportAddressTableRVA = 0x11F8;
        auto* thunk = reinterpret_cast<IMAGE_THUNK_DATA64*>(img.data() + 0x3F8);  // file offset of RVA 0x11F8
        thunk->u1.AddressOfData = 0xFFFF;  // non-zero, so the walk needs another entry and the image is over
        auto& dir = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(IMAGE_DELAYLOAD_DESCRIPTOR);
        expect_rejects("a delay-import thunk table that runs past the image", img);
    }
}

// movabs rax, addr ; mov dword [rax], value ; mov eax, eax_value ; ret
void emit_store_stub(std::vector<uint8_t>& img, size_t off, uint64_t addr, uint32_t value, uint32_t eax_value) {
    size_t n = off;
    img[n++] = 0x48;
    img[n++] = 0xB8;
    for (int i = 0; i < 8; ++i) img[n++] = static_cast<uint8_t>(addr >> (8 * i));
    img[n++] = 0xC7;
    img[n++] = 0x00;
    for (int i = 0; i < 4; ++i) img[n++] = static_cast<uint8_t>(value >> (8 * i));
    img[n++] = 0xB8;
    for (int i = 0; i < 4; ++i) img[n++] = static_cast<uint8_t>(eax_value >> (8 * i));
    img[n++] = 0xC3;
}

constexpr uint64_t kInitBase = 0x140000000;

// make_image() plus a TLS directory and an entry point, both aimed at real machine code: a stub that stores a
// sentinel in this process, so the test can tell a callback ran rather than only that it was parsed.
std::vector<uint8_t> make_init_image() {
    auto img = make_image();
    auto* nt = nt_of(img);
    nt->FileHeader.Characteristics |= IMAGE_FILE_DLL;
    nt->OptionalHeader.AddressOfEntryPoint = 0x1040;
    auto va = [](uint32_t rva) { return kInitBase + rva; };
    constexpr size_t text = 0x200;  // file offset of RVA 0x1000
    emit_store_stub(img, text + 0x00, reinterpret_cast<uint64_t>(&g_tls_ran), 1, 0);
    emit_store_stub(img, text + 0x40, reinterpret_cast<uint64_t>(&g_main_ran), 2, 1);
    std::memcpy(img.data() + text + 0x80, "TLS!", 4);
    auto* callbacks = reinterpret_cast<uint64_t*>(img.data() + text + 0x90);
    callbacks[0] = va(0x1000);
    callbacks[1] = 0;
    auto* tls = reinterpret_cast<IMAGE_TLS_DIRECTORY64*>(img.data() + text + 0xB0);
    tls->StartAddressOfRawData = va(0x1080);
    tls->EndAddressOfRawData = va(0x1084);
    tls->AddressOfIndex = va(0x10A0);
    tls->AddressOfCallBacks = va(0x1090);
    tls->SizeOfZeroFill = 8;
    auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_TLS];
    dir.VirtualAddress = 0x10B0;
    dir.Size = sizeof(IMAGE_TLS_DIRECTORY64);
    return img;
}

void expect_init_rejects(const char* what, const std::vector<uint8_t>& bytes) {
    auto p = write_image(bytes);
    try {
        rt::Loader loader(g_dir);
        loader.load(p);
        loader.initialise();
        failed(what, "the module initialised");
    } catch (const std::exception&) {
        pass(what);
    }
}

void test_initialise() {
    g_tls_ran = 0;
    g_main_ran = 0;
    auto p = write_image(make_init_image());
    try {
        rt::Loader loader(g_dir);
        rt::Module& m = loader.load(p);
        loader.initialise();
        if (g_tls_ran != 1)
            failed("initialise runs the TLS callback", "the callback never wrote its sentinel");
        else
            pass("initialise runs the TLS callback");
        if (g_main_ran != 2)
            failed("initialise calls DllMain for a DLL", "the entry point never wrote its sentinel");
        else
            pass("initialise calls DllMain for a DLL");
        uint32_t slot = *reinterpret_cast<const uint32_t*>(m.base + 0x10A0);
        void* block = slot == TLS_OUT_OF_INDEXES ? nullptr : TlsGetValue(slot);
        if (!block || std::memcmp(block, "TLS!", 4) != 0)
            failed("the TLS template is copied into the allocated slot", "the slot doesn't hold the template");
        else
            pass("the TLS template is copied into the allocated slot");
    } catch (const std::exception& e) {
        failed("initialise runs a TLS callback and DllMain", e.what());
    }
}

void test_initialise_rejects() {
    {
        auto img = make_init_image();
        nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_TLS].VirtualAddress = 0x1FF0;
        expect_init_rejects("a TLS directory past the image is rejected", img);
    }
    {
        auto img = make_init_image();
        auto* callbacks = reinterpret_cast<uint64_t*>(img.data() + 0x290);  // file offset of RVA 0x1090
        callbacks[0] = kInitBase + 0x9000;                                  // well past SizeOfImage
        expect_init_rejects("a TLS callback outside the image is rejected", img);
    }
    {
        g_main_ran = 0;
        auto img = make_init_image();
        emit_store_stub(img, 0x240, reinterpret_cast<uint64_t>(&g_main_ran), 3, 0);  // DllMain returns FALSE
        expect_init_rejects("a DllMain that returns FALSE is rejected", img);
    }
}

void test_forwarded_exports() {
    {
        const char* what = "a forwarded export binds to its target instead of into the string";
        write_image_as(make_forwarder_image("KERNEL32.GetCurrentProcessId", 0x180000000), "prov.dll");
        auto importer = write_image_as(make_import_image("prov.dll", "fwd"), "importer.exe");
        try {
            rt::Loader loader(g_dir);
            rt::Module& m = loader.load(importer);
            void* expected = host_symbol("kernel32.dll", "GetCurrentProcessId");
            if (!expected) return failed(what, "GetCurrentProcessId is not loaded in this process");
            if (m.imports.size() != 1)
                return failed(what, "the importer has " + std::to_string(m.imports.size()) + " import record(s)");
            if (m.imports[0].binding != rt::Binding::Host)
                return failed(what, "the import did not bind through the forwarder to the host");
            if (*reinterpret_cast<void**>(m.base + 0x1180) != expected)
                return failed(what, "the address table does not hold the forwarded target");
            rt::Module* prov = nullptr;
            for (auto* mod : loader.modules())
                if (mod->name == "prov.dll") prov = mod;
            if (!prov) return failed(what, "the provider module was not loaded");
            if (prov->exports.count("fwd")) return failed(what, "the forwarder was stored as a code export");
            auto it = prov->forwarders.find("fwd");
            if (it == prov->forwarders.end() || it->second != "KERNEL32.GetCurrentProcessId")
                return failed(what, "the forwarder string was not recorded");
            pass(what);
        } catch (const std::exception& e) {
            failed(what, e.what());
        }
    }
    {
        const char* what = "a forwarder to a module that isn't there binds as unimplemented";
        write_image_as(make_forwarder_image("NOPE_MISSING_TARGET.f", 0x180000000), "prov.dll");
        auto importer = write_image_as(make_import_image("prov.dll", "fwd"), "importer.exe");
        try {
            rt::Loader loader(g_dir);
            rt::Module& m = loader.load(importer);
            if (m.imports.size() != 1 || m.imports[0].binding != rt::Binding::Unimplemented)
                return failed(what, "the import did not end up unimplemented");
            if (!*reinterpret_cast<void**>(m.base + 0x1180))
                return failed(what, "the address table was left null instead of a failure thunk");
            pass(what);
        } catch (const std::exception& e) {
            failed(what, e.what());
        }
    }
    {
        const char* what = "forwarders that point at each other stop instead of recursing";
        write_image_as(make_forwarder_image("prov_b.fwd", 0x180000000), "prov_a.dll");
        write_image_as(make_forwarder_image("prov_a.fwd", 0x190000000), "prov_b.dll");
        auto importer = write_image_as(make_import_image("prov_a.dll", "fwd"), "importer.exe");
        try {
            rt::Loader loader(g_dir);
            rt::Module& m = loader.load(importer);
            if (m.imports.size() != 1 || m.imports[0].binding != rt::Binding::Unimplemented)
                return failed(what, "a forwarder loop did not end as unimplemented");
            pass(what);
        } catch (const std::exception& e) {
            failed(what, e.what());
        }
    }
}

void test_delay_descriptor_fields() {
    {
        // Without dlattrRva the fields are 32-bit virtual addresses, which cannot address an x64 image.
        auto img = make_image();
        std::fill(img.begin() + 0x200, img.begin() + 0x400, 0);
        auto* d = reinterpret_cast<IMAGE_DELAYLOAD_DESCRIPTOR*>(img.data() + 0x200);
        d->Attributes.AllAttributes = 0;
        d->DllNameRVA = 0x1060;  // past the zero descriptor that ends the table
        d->ImportAddressTableRVA = 0x1180;
        std::memcpy(img.data() + 0x260, "kernel32.dll\0", 13);
        auto& dir = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(IMAGE_DELAYLOAD_DESCRIPTOR);
        expect_rejects("a delay-import descriptor without the RVA attribute", img);
    }
    {
        const char* what = "a delay import with no name table binds through its address table";
        auto img = make_image();
        std::fill(img.begin() + 0x200, img.begin() + 0x400, 0);
        auto* d = reinterpret_cast<IMAGE_DELAYLOAD_DESCRIPTOR*>(img.data() + 0x200);
        d->Attributes.AllAttributes = 1;  // dlattrRva
        d->DllNameRVA = 0x1060;  // past the zero descriptor that ends the table
        d->ImportNameTableRVA = 0;
        d->ImportAddressTableRVA = 0x1180;
        std::memcpy(img.data() + 0x260, "kernel32.dll\0", 13);
        *reinterpret_cast<uint64_t*>(img.data() + 0x380) = 0x1120;  // RVA 0x1180: the single hint/name entry
        *reinterpret_cast<uint16_t*>(img.data() + 0x320) = 0;
        std::memcpy(img.data() + 0x322, "GetCurrentProcessId\0", 20);
        auto& dir = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(IMAGE_DELAYLOAD_DESCRIPTOR);
        try {
            rt::Loader loader(g_dir);
            rt::Module& m = loader.load(write_image_as(img, "delay.exe"));
            void* expected = host_symbol("kernel32.dll", "GetCurrentProcessId");
            if (m.imports.size() != 1) return failed(what, "expected one bound import");
            if (m.imports[0].binding != rt::Binding::Host) return failed(what, "the delay import did not bind to the host");
            if (*reinterpret_cast<void**>(m.base + 0x1180) != expected)
                return failed(what, "the address table does not hold the host function");
            pass(what);
        } catch (const std::exception& e) {
            failed(what, e.what());
        }
    }
}

}  // namespace

int main() {
    g_dir = std::filesystem::temp_directory_path() / "leonida_loader_test";
    std::error_code ec;
    std::filesystem::remove_all(g_dir, ec);
    std::filesystem::create_directories(g_dir, ec);
    if (ec) {
        std::printf("FAIL  cannot create %s: %s\n", g_dir.string().c_str(), ec.message().c_str());
        return 1;
    }

    test_valid_image();
    test_header_bounds();
    test_section_bounds();
    test_truncation();
    test_relocations();
    test_exports();
    test_imports();
    test_delay_imports();
    test_initialise();
    test_initialise_rejects();
    test_delay_descriptor_fields();
    test_forwarded_exports();

    std::filesystem::remove_all(g_dir, ec);
    std::printf("\n%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
