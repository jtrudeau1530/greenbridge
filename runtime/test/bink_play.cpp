// bink_play: play a Bink 2 stream in a window through the package's own Bink2Scarlett.dll, mapped by rt::Loader (the
// same path bink_decode_test checks). A local viewing tool for movies carved from your own copy; nothing it reads or
// shows is written anywhere.
//
//   bink_play <movie.bk2 | archive.rpf@OFFSET> [--game-dir DIR] [--frames N] [--headless] [--loop]
//
// A movie inside an archive plays where it is: `movies.rpf@OFFSET` hands BinkOpen the archive with
// BinkSetFileOffset, so nothing is extracted (tools/bink_play.py finds the offset from the movie's name). A relative
// archive path is taken from the game folder.
//
// --game-dir defaults to LEONIDA_GAME_DIR, the movie to LEONIDA_MOVIE. Without a package or a movie it prints
// "SKIP (no package/movie)" and exits 77 (CTest skip). --headless decodes without a window (as fast as it can) and
// prints the decode rate, so CTest can run it; --frames stops after N frames.
//
// Keys: Space pause/resume, Left/Right one second back/forward, Home restart, L loop on/off, Esc or closing the
// window quits. The frame is scaled to the window, keeping its aspect ratio.
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../shims/shims.h"
#include "../src/pe_loader.h"

namespace {

[[noreturn]] void fatal(const std::string& what) {
    std::fprintf(stderr, "FAIL %s\n", what.c_str());
    std::exit(1);
}

std::string env(const char* name) {
    const char* v = std::getenv(name);
    return v ? v : "";
}

uint32_t read_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

struct MovieHeader {
    uint32_t frames = 0, width = 0, height = 0, fps_num = 0, fps_den = 0;
};

// The fixed Bink 2 header (docs/movies.md): frame count at 0x08, size at 0x14/0x18, frame rate at 0x1C/0x20.
MovieHeader read_movie_header(const std::filesystem::path& path, uint64_t offset) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fatal("cannot open movie " + path.string());
    f.seekg((std::streamoff)offset);
    uint8_t h[0x2C]{};
    f.read(reinterpret_cast<char*>(h), sizeof(h));
    if (f.gcount() != (std::streamsize)sizeof(h)) fatal("movie is shorter than the 44-byte Bink 2 header");
    if (h[0] != 'K' || h[1] != 'B' || h[2] != '2' || h[3] < 'a' || h[3] > 'z')
        fatal("movie does not start with KB2[a-z]");
    MovieHeader out;
    out.frames = read_u32(h + 0x08);
    out.width = read_u32(h + 0x14);
    out.height = read_u32(h + 0x18);
    out.fps_num = read_u32(h + 0x1C);
    out.fps_den = read_u32(h + 0x20);
    if (!out.frames || !out.width || !out.height || !out.fps_num || !out.fps_den)
        fatal("movie header has a zero frame count, size or fps");
    if (out.width > 16384 || out.height > 16384) fatal("movie header has an implausible frame size");
    return out;
}

struct BinkApi {
    void* (*open)(const char*, uint32_t) = nullptr;
    uint32_t (*do_frame)(void*) = nullptr;
    uint32_t (*copy)(void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) = nullptr;
    void (*next)(void*) = nullptr;
    void (*go_to)(void*, uint32_t, uint32_t) = nullptr;
    void (*close)(void*) = nullptr;
    const char* (*get_error)() = nullptr;
    void (*set_file_offset)(uint64_t) = nullptr;  // applies to the next BinkOpen
};

template <class F> void bind(rt::Module& m, const char* name, F& out) {
    auto it = m.exports.find(name);
    if (it == m.exports.end()) fatal(std::string("Bink2Scarlett.dll does not export ") + name);
    out = reinterpret_cast<F>(it->second);
}

constexpr uint32_t kCopyBgra32 = 3;  // the BinkCopyToBuffer format bink_decode_test measured as 32-bit BGRA
constexpr uint32_t kFileOffset = 0x20;  // BinkOpen flag: start at the offset BinkSetFileOffset gave

struct Player {
    BinkApi api;
    void* bink = nullptr;
    MovieHeader header;
    std::vector<uint8_t> frame;
    uint32_t index = 0;   // the frame in `frame`
    bool paused = false;
    bool loop = false;
    bool quit = false;
    bool seek_pending = false;
    uint32_t seek_to = 0;

    void decode() {
        api.do_frame(bink);
        api.copy(bink, frame.data(), header.width * 4, header.height, 0, 0, kCopyBgra32);
    }
    // Bink frame numbers in BinkGoto are 1-based.
    void go_to(uint32_t target) {
        index = std::min(target, header.frames - 1);
        api.go_to(bink, index + 1, 0);
        decode();
    }
    uint32_t fps_frames(double seconds) const {
        return (uint32_t)(seconds * header.fps_num / header.fps_den);
    }
};

Player* g_player = nullptr;

void paint(HWND hwnd) {
    PAINTSTRUCT ps;
    HDC dc = BeginPaint(hwnd, &ps);
    RECT rc;
    GetClientRect(hwnd, &rc);
    const int cw = rc.right - rc.left, ch = rc.bottom - rc.top;
    const Player& p = *g_player;
    // Fit the frame in the client area, keeping its aspect ratio, and black out the bars.
    int w = cw, h = (int)((int64_t)cw * p.header.height / p.header.width);
    if (h > ch) {
        h = ch;
        w = (int)((int64_t)ch * p.header.width / p.header.height);
    }
    const int x = (cw - w) / 2, y = (ch - h) / 2;
    HBRUSH black = (HBRUSH)GetStockObject(BLACK_BRUSH);
    RECT bars[] = {{0, 0, cw, y}, {0, y + h, cw, ch}, {0, y, x, y + h}, {x + w, y, cw, y + h}};
    for (const RECT& b : bars) FillRect(dc, &b, black);
    BITMAPINFO bi{};
    bi.bmiHeader.biSize = sizeof(bi.bmiHeader);
    bi.bmiHeader.biWidth = (LONG)p.header.width;
    bi.bmiHeader.biHeight = -(LONG)p.header.height;  // top-down
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    bi.bmiHeader.biCompression = BI_RGB;
    SetStretchBltMode(dc, COLORONCOLOR);
    StretchDIBits(dc, x, y, w, h, 0, 0, (int)p.header.width, (int)p.header.height, p.frame.data(), &bi,
                  DIB_RGB_COLORS, SRCCOPY);
    EndPaint(hwnd, &ps);
}

LRESULT CALLBACK window_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    Player& p = *g_player;
    switch (msg) {
        case WM_PAINT: paint(hwnd); return 0;
        case WM_ERASEBKGND: return 1;
        case WM_KEYDOWN:
            switch (wp) {
                case VK_ESCAPE: DestroyWindow(hwnd); return 0;
                case VK_SPACE: p.paused = !p.paused; return 0;
                case 'L': p.loop = !p.loop; return 0;
                case VK_HOME: p.seek_pending = true; p.seek_to = 0; return 0;
                case VK_LEFT: {
                    const uint32_t back = p.fps_frames(1.0);
                    p.seek_pending = true;
                    p.seek_to = p.index > back ? p.index - back : 0;
                    return 0;
                }
                case VK_RIGHT:
                    p.seek_pending = true;
                    p.seek_to = std::min(p.index + p.fps_frames(1.0), p.header.frames - 1);
                    return 0;
            }
            break;
        case WM_DESTROY: p.quit = true; PostQuitMessage(0); return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

void set_title(HWND hwnd, const Player& p, const std::wstring& name, double rate) {
    wchar_t title[256];
    swprintf(title, 256, L"bink_play - %s - frame %u/%u - decode %.1f fps%s%s", name.c_str(), p.index + 1,
             p.header.frames, rate, p.paused ? L" - paused" : L"", p.loop ? L" - loop" : L"");
    SetWindowTextW(hwnd, title);
}

}  // namespace

int main(int argc, char** argv) {
    std::string game_dir_arg = env("LEONIDA_GAME_DIR"), movie_arg = env("LEONIDA_MOVIE");
    long max_frames = -1;
    bool headless = false, loop = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--game-dir" && i + 1 < argc) game_dir_arg = argv[++i];
        else if (a == "--frames" && i + 1 < argc) {
            char* end = nullptr;
            max_frames = std::strtol(argv[++i], &end, 10);
            if (!end || *end || max_frames <= 0) fatal("--frames takes a positive count");
        } else if (a == "--headless") headless = true;
        else if (a == "--loop") loop = true;
        else if (!a.empty() && a[0] != '-') movie_arg = a;
        else {
            std::fprintf(stderr, "usage: bink_play <movie.bk2> [--game-dir DIR] [--frames N] [--headless] [--loop]\n");
            return 2;
        }
    }
    if (game_dir_arg.empty() || movie_arg.empty()) {
        std::printf("SKIP (no package/movie)\n");
        return 77;
    }
    const std::filesystem::path game_dir = game_dir_arg;
    std::filesystem::path movie = movie_arg;
    uint64_t offset = 0;
    if (const size_t at = movie_arg.rfind('@'); at != std::string::npos) {
        char* end = nullptr;
        offset = std::strtoull(movie_arg.c_str() + at + 1, &end, 0);
        if (at + 1 == movie_arg.size() || !end || *end) fatal("the movie's @OFFSET is not a number");
        movie = movie_arg.substr(0, at);
        if (movie.is_relative() && !std::filesystem::exists(movie)) movie = game_dir / movie;
    }
    std::error_code ec;
    if (!std::filesystem::is_regular_file(game_dir / "Bink2Scarlett.dll", ec))
        fatal("the game folder does not contain Bink2Scarlett.dll");
    if (!std::filesystem::is_regular_file(movie, ec)) fatal("the movie is not a file");

    Player player;
    g_player = &player;
    player.header = read_movie_header(movie, offset);
    player.loop = loop;
    const MovieHeader& h = player.header;
    std::printf("movie: %ux%u, %u frames, %u/%u fps\n", h.width, h.height, h.frames, h.fps_num, h.fps_den);

    try {
        rt::Loader loader(game_dir);
        rt::Module& dll = loader.load(game_dir / "Bink2Scarlett.dll");
        for (auto& r : dll.imports)
            if (r.binding == rt::Binding::Unimplemented && rt::lower(r.dll).find("xaudio") == std::string::npos)
                fatal("Bink2Scarlett.dll import " + r.dll + "!" + r.name + " is unimplemented");
        BinkApi& api = player.api;
        bind(dll, "BinkOpen", api.open);
        bind(dll, "BinkDoFrame", api.do_frame);
        bind(dll, "BinkCopyToBuffer", api.copy);
        bind(dll, "BinkNextFrame", api.next);
        bind(dll, "BinkGoto", api.go_to);
        bind(dll, "BinkClose", api.close);
        bind(dll, "BinkGetError", api.get_error);
        if (offset) {
            bind(dll, "BinkSetFileOffset", api.set_file_offset);
            api.set_file_offset(offset);
        }

        player.bink = api.open(movie.string().c_str(), offset ? kFileOffset : 0);
        if (!player.bink) {
            const char* e = api.get_error ? api.get_error() : nullptr;
            fatal(std::string("BinkOpen failed: ") + (e && *e ? e : "(no BinkGetError text)"));
        }
        player.frame.assign((size_t)h.width * 4 * h.height, 0);
        player.decode();

        using clock = std::chrono::steady_clock;
        const auto frame_time = std::chrono::duration<double>((double)h.fps_den / h.fps_num);
        const uint32_t last = max_frames > 0 ? (uint32_t)std::min<long>(max_frames, (long)h.frames) : h.frames;

        if (headless) {
            const auto start = clock::now();
            uint32_t decoded = 1;
            while (decoded < last) {
                api.next(player.bink);
                player.decode();
                ++decoded;
            }
            const double s = std::chrono::duration<double>(clock::now() - start).count();
            std::printf("decoded %u frames in %.2f s (%.1f fps; the movie runs at %.2f fps)\n", decoded, s,
                        s > 0 ? decoded / s : 0.0, (double)h.fps_num / h.fps_den);
            api.close(player.bink);
            std::printf("RESULT PASS\n");
            return 0;
        }

        WNDCLASSW wc{};
        wc.lpfnWndProc = window_proc;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wc.lpszClassName = L"LeonidaBinkPlay";
        RegisterClassW(&wc);
        // Start at up to 1280 wide (or the movie's size if smaller), the movie's aspect ratio.
        const int start_w = (int)std::min<uint32_t>(h.width, 1280);
        RECT r{0, 0, start_w, (int)((int64_t)start_w * h.height / h.width)};
        AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
        HWND hwnd = CreateWindowW(wc.lpszClassName, L"bink_play", WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT,
                                  CW_USEDEFAULT, r.right - r.left, r.bottom - r.top, nullptr, nullptr, wc.hInstance,
                                  nullptr);
        if (!hwnd) fatal("CreateWindow failed");
        const std::wstring name = movie.filename().wstring() + (offset ? L"@" + std::to_wstring(offset) : L"");

        auto due = clock::now() + frame_time;
        auto rate_start = clock::now();
        uint32_t rate_frames = 0, shown = 1;
        double rate = 0;
        while (!player.quit) {
            MSG msg;
            while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            if (player.quit) break;
            if (player.seek_pending) {
                player.seek_pending = false;
                player.go_to(player.seek_to);
                InvalidateRect(hwnd, nullptr, FALSE);
                due = clock::now() + frame_time;
            }
            if (player.paused || clock::now() < due) {
                set_title(hwnd, player, name, rate);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            if (player.index + 1 >= h.frames || (max_frames > 0 && shown >= last)) {
                if (!player.loop) break;
                player.go_to(0);
            } else {
                api.next(player.bink);
                player.decode();
                ++player.index;
            }
            ++shown;
            InvalidateRect(hwnd, nullptr, FALSE);
            UpdateWindow(hwnd);
            ++rate_frames;
            const double elapsed = std::chrono::duration<double>(clock::now() - rate_start).count();
            if (elapsed >= 1.0) {
                rate = rate_frames / elapsed;
                rate_frames = 0;
                rate_start = clock::now();
            }
            set_title(hwnd, player, name, rate);
            // Hold the movie's frame rate; when decoding falls behind, don't try to catch up in a burst.
            due += std::chrono::duration_cast<clock::duration>(frame_time);
            if (clock::now() > due + std::chrono::duration_cast<clock::duration>(frame_time * 4)) due = clock::now();
        }
        if (IsWindow(hwnd)) DestroyWindow(hwnd);
        api.close(player.bink);
        std::printf("played %u frames\n", shown);
        return 0;
    } catch (const std::exception& e) {
        fatal(e.what());
    }
}
