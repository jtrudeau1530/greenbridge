// Checks the offline WinHTTP the shims give a mapped title (runtime/shims/winhttp_offline.cpp).
//
// Every function is resolved through rt::shims::find under winhttp.dll, so the test runs against the table the
// runtime installs. Nothing here touches the network: the server names use the reserved .invalid top-level domain, and
// the only host WinHTTP call the shims make is WinHttpCrackUrl, which parses a string.
#include <windows.h>
#include <winhttp.h>
#include <fcntl.h>
#include <io.h>
#include <string>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>

#include "shims.h"

namespace {

using OpenFn = HINTERNET(WINAPI*)(LPCWSTR, DWORD, LPCWSTR, LPCWSTR, DWORD);
using ConnectFn = HINTERNET(WINAPI*)(HINTERNET, LPCWSTR, INTERNET_PORT, DWORD);
using OpenRequestFn = HINTERNET(WINAPI*)(HINTERNET, LPCWSTR, LPCWSTR, LPCWSTR, LPCWSTR, LPCWSTR*, DWORD);
using SendRequestFn = BOOL(WINAPI*)(HINTERNET, LPCWSTR, DWORD, LPVOID, DWORD, DWORD, DWORD_PTR);
using ReceiveResponseFn = BOOL(WINAPI*)(HINTERNET, LPVOID);
using ReadDataFn = BOOL(WINAPI*)(HINTERNET, LPVOID, DWORD, LPDWORD);
using QueryHeadersFn = BOOL(WINAPI*)(HINTERNET, DWORD, LPCWSTR, LPVOID, LPDWORD, LPDWORD);
using AddRequestHeadersFn = BOOL(WINAPI*)(HINTERNET, LPCWSTR, DWORD, DWORD);
using SetTimeoutsFn = BOOL(WINAPI*)(HINTERNET, int, int, int, int);
using SetOptionFn = BOOL(WINAPI*)(HINTERNET, DWORD, LPVOID, DWORD);
using QueryOptionFn = BOOL(WINAPI*)(HINTERNET, DWORD, LPVOID, LPDWORD);
using SetStatusCallbackFn = WINHTTP_STATUS_CALLBACK(WINAPI*)(HINTERNET, WINHTTP_STATUS_CALLBACK, DWORD, DWORD_PTR);
using CloseHandleFn = BOOL(WINAPI*)(HINTERNET);
using CrackUrlFn = BOOL(WINAPI*)(LPCWSTR, DWORD, DWORD, LPURL_COMPONENTS);

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

template <typename Fn>
Fn resolve(const char* name) {
    return reinterpret_cast<Fn>(rt::shims::find("winhttp.dll", name));
}

struct StderrCapture {
    int saved_fd = -1;
    int read_fd = -1;

    StderrCapture() {
        std::fflush(stderr);
        const int stderr_fd = _fileno(stderr);
        int fds[2] = {-1, -1};
        if (stderr_fd < 0 || _pipe(fds, 4096, _O_TEXT) != 0) return;
        const int dup_saved = _dup(stderr_fd);
        if (dup_saved < 0 || _dup2(fds[1], stderr_fd) != 0) {
            if (dup_saved >= 0) _close(dup_saved);
            _close(fds[0]);
            _close(fds[1]);
            return;
        }
        _close(fds[1]);
        saved_fd = dup_saved;
        read_fd = fds[0];
    }

    ~StderrCapture() { stop(); }

    std::string stop() {
        std::string out;
        if (saved_fd >= 0) {
            std::fflush(stderr);
            _dup2(saved_fd, _fileno(stderr));
            _close(saved_fd);
            saved_fd = -1;
        }
        if (read_fd >= 0) {
            char buf[512];
            for (;;) {
                const int n = _read(read_fd, buf, static_cast<unsigned int>(sizeof(buf)));
                if (n <= 0) break;
                out.append(buf, static_cast<size_t>(n));
            }
            _close(read_fd);
            read_fd = -1;
        }
        return out;
    }
};

struct CreatedRequestProbe {
    OpenRequestFn open_request = nullptr;
    SendRequestFn send = nullptr;
    CloseHandleFn close = nullptr;
    int created_calls = 0;
    HINTERNET created_handle = nullptr;
    HINTERNET request_handle = nullptr;
    BOOL send_result = TRUE;
    DWORD send_error = 0;
    BOOL request_closed = FALSE;
};
CreatedRequestProbe* g_created_probe = nullptr;

void CALLBACK on_created_send_request(HINTERNET handle, DWORD_PTR, DWORD status, LPVOID, DWORD) {
    if (status != WINHTTP_CALLBACK_STATUS_HANDLE_CREATED || !g_created_probe) return;
    CreatedRequestProbe& p = *g_created_probe;
    if (p.created_calls++ != 0) return;
    p.created_handle = handle;
    p.request_handle = p.open_request(handle, L"GET", L"/created", nullptr, WINHTTP_NO_REFERER,
                                      WINHTTP_DEFAULT_ACCEPT_TYPES, 0);
    if (p.request_handle) {
        SetLastError(0);
        p.send_result = p.send(p.request_handle, nullptr, 0, nullptr, 0, 0, 0);
        p.send_error = GetLastError();
        p.request_closed = p.close(p.request_handle);
    }
}

struct CreatedCloseProbe {
    CloseHandleFn close = nullptr;
    int created_calls = 0;
    int closing_calls = 0;
    bool in_created = false;
    bool closing_during_created = false;
    HINTERNET created_handle = nullptr;
    HINTERNET closing_handle = nullptr;
    BOOL close_result = FALSE;
};
CreatedCloseProbe* g_close_probe = nullptr;

void CALLBACK on_created_close_handle(HINTERNET handle, DWORD_PTR, DWORD status, LPVOID info, DWORD length) {
    if (!g_close_probe) return;
    CreatedCloseProbe& p = *g_close_probe;
    if (status == WINHTTP_CALLBACK_STATUS_HANDLE_CREATED) {
        ++p.created_calls;
        p.created_handle = handle;
        p.in_created = true;
        if (p.close) p.close_result = p.close(handle);
        p.in_created = false;
    } else if (status == WINHTTP_CALLBACK_STATUS_HANDLE_CLOSING && info && length == sizeof(HINTERNET) &&
               *static_cast<HINTERNET*>(info) == handle) {
        if (p.in_created) p.closing_during_created = true;
        ++p.closing_calls;
        p.closing_handle = handle;
    }
}

// What the status callback saw, for the asynchronous checks.
struct Seen {
    std::atomic<int> request_errors{0};
    std::atomic<int> closing{0};
    std::atomic<int> created{0};
    std::atomic<DWORD> api{0};
    std::atomic<DWORD> error{0};
    std::atomic<DWORD_PTR> context{0};
    std::atomic<DWORD> thread{0};
    std::atomic<bool> closing_after_error{true};
    HANDLE error_event = nullptr;
    HANDLE closing_event = nullptr;
    CloseHandleFn close_in_callback = nullptr;  // when set, the error callback closes the request itself
};
Seen* g_seen = nullptr;

void CALLBACK on_status(HINTERNET handle, DWORD_PTR context, DWORD status, LPVOID info, DWORD length) {
    Seen& s = *g_seen;
    if (status == WINHTTP_CALLBACK_STATUS_HANDLE_CREATED) {
        ++s.created;
    } else if (status == WINHTTP_CALLBACK_STATUS_REQUEST_ERROR && info && length == sizeof(WINHTTP_ASYNC_RESULT)) {
        const auto* result = static_cast<const WINHTTP_ASYNC_RESULT*>(info);
        s.api = static_cast<DWORD>(result->dwResult);
        s.error = result->dwError;
        s.context = context;
        s.thread = GetCurrentThreadId();
        ++s.request_errors;
        if (s.close_in_callback) s.close_in_callback(handle);
        SetEvent(s.error_event);
    } else if (status == WINHTTP_CALLBACK_STATUS_HANDLE_CLOSING) {
        if (s.request_errors.load() == 0) s.closing_after_error = false;
        if (info && length == sizeof(HINTERNET) && *static_cast<HINTERNET*>(info) == handle) ++s.closing;
        SetEvent(s.closing_event);
    }
}

double ms_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

int main() {
    const auto open = resolve<OpenFn>("WinHttpOpen");
    const auto connect = resolve<ConnectFn>("WinHttpConnect");
    const auto open_request = resolve<OpenRequestFn>("WinHttpOpenRequest");
    const auto send = resolve<SendRequestFn>("WinHttpSendRequest");
    const auto receive = resolve<ReceiveResponseFn>("WinHttpReceiveResponse");
    const auto read = resolve<ReadDataFn>("WinHttpReadData");
    const auto query_headers = resolve<QueryHeadersFn>("WinHttpQueryHeaders");
    const auto add_headers = resolve<AddRequestHeadersFn>("WinHttpAddRequestHeaders");
    const auto set_timeouts = resolve<SetTimeoutsFn>("WinHttpSetTimeouts");
    const auto set_option = resolve<SetOptionFn>("WinHttpSetOption");
    const auto query_option = resolve<QueryOptionFn>("WinHttpQueryOption");
    const auto set_callback = resolve<SetStatusCallbackFn>("WinHttpSetStatusCallback");
    const auto close = resolve<CloseHandleFn>("WinHttpCloseHandle");
    const auto crack = resolve<CrackUrlFn>("WinHttpCrackUrl");
    check(open && connect && open_request && send && receive && read && query_headers && add_headers &&
              set_timeouts && set_option && query_option && set_callback && close && crack &&
              rt::shims::find("winhttp.dll", "WinHttpQueryDataAvailable") &&
              rt::shims::find("winhttp.dll", "WinHttpWriteData"),
          "all 16 WinHTTP imports of the package resolve through the registry");
    check(rt::shims::owns_dll("winhttp.dll"), "winhttp.dll is a module the shims own (LoadLibrary gets a pseudo-handle)");
    if (!(open && connect && open_request && send && close && set_callback)) {
        std::printf("%d check(s), %d failure(s)\n", g_checks, g_failures + 1);
        return 1;
    }

    // Synchronous: the send fails at once, nothing is resolved or connected.
    {
        const HINTERNET session = open(L"test", WINHTTP_ACCESS_TYPE_NO_PROXY, nullptr, nullptr, 0);
        check(session != nullptr, "WinHttpOpen returns a session");
        const HINTERNET connection = connect(session, L"example.invalid", INTERNET_DEFAULT_HTTPS_PORT, 0);
        check(connection != nullptr, "WinHttpConnect returns a connection without resolving the name");
        const HINTERNET request =
            open_request(connection, L"GET", L"/x", nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
                         WINHTTP_FLAG_SECURE);
        check(request != nullptr, "WinHttpOpenRequest returns a request");
        check(add_headers(request, L"X-Test: 1", static_cast<DWORD>(-1), WINHTTP_ADDREQ_FLAG_ADD) != FALSE,
              "request headers are accepted");
        check(set_timeouts(request, 1000, 1000, 1000, 1000) != FALSE, "timeouts are accepted");

        // The same answers the host's winhttp.dll gives (measured on Windows 11 26300).
        const auto rejects = [](BOOL ok) { return ok == FALSE && GetLastError() == ERROR_INVALID_PARAMETER; };
        SetLastError(0);
        check(rejects(add_headers(request, nullptr, 0, WINHTTP_ADDREQ_FLAG_ADD)), "a null header block is rejected");
        SetLastError(0);
        check(rejects(add_headers(request, L"", static_cast<DWORD>(-1), WINHTTP_ADDREQ_FLAG_ADD)),
              "an empty header block is rejected");
        SetLastError(0);
        check(rejects(add_headers(request, L"nocolon", static_cast<DWORD>(-1), WINHTTP_ADDREQ_FLAG_ADD)),
              "a header with no colon is rejected");
        SetLastError(0);
        check(rejects(add_headers(request, L"X-C: b", 2, WINHTTP_ADDREQ_FLAG_ADD)),
              "a length that cuts the header before its colon is rejected");
        check(add_headers(request, L"X-D: b", 6, WINHTTP_ADDREQ_FLAG_ADD) != FALSE,
              "an explicit length that covers the colon is accepted");
        check(set_timeouts(request, 0, 0, 0, 0) != FALSE && set_timeouts(request, -1, -1, -1, -1) != FALSE,
              "timeouts of 0 and -1 (infinite) are accepted");
        for (int field = 0; field < 4; ++field) {
            int t[4] = {0, 0, 0, 0};
            t[field] = -2;
            SetLastError(0);
            check(rejects(set_timeouts(request, t[0], t[1], t[2], t[3])), "a timeout below -1 is rejected");
        }

        SetLastError(0);
        const auto start = std::chrono::steady_clock::now();
        const BOOL sent = send(request, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0);
        const DWORD error = GetLastError();
        const double elapsed = ms_since(start);
        check(sent == FALSE && error == ERROR_WINHTTP_CANNOT_CONNECT,
              "WinHttpSendRequest fails with ERROR_WINHTTP_CANNOT_CONNECT");
        check(elapsed < 50.0, "and fails at once (under 50 ms)");
        SetLastError(0);
        check(receive(request, nullptr) == FALSE && GetLastError() == ERROR_WINHTTP_INCORRECT_HANDLE_STATE,
              "there is no response to receive");
        char buffer[16];
        DWORD got = 0;
        SetLastError(0);
        check(read(request, buffer, sizeof(buffer), &got) == FALSE &&
                  GetLastError() == ERROR_WINHTTP_INCORRECT_HANDLE_STATE,
              "and nothing to read");
        DWORD status = 0, size = sizeof(status);
        SetLastError(0);
        check(query_headers(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX,
                            &status, &size, WINHTTP_NO_HEADER_INDEX) == FALSE &&
                  GetLastError() == ERROR_WINHTTP_INCORRECT_HANDLE_STATE,
              "and no status code");

        // Options: the context value round-trips, the handle type is answered, the rest is refused when queried.
        DWORD_PTR value = 0x1234;
        check(set_option(request, WINHTTP_OPTION_CONTEXT_VALUE, &value, sizeof(value)) != FALSE,
              "WINHTTP_OPTION_CONTEXT_VALUE can be set");
        DWORD_PTR back = 0;
        DWORD back_size = sizeof(back);
        check(query_option(request, WINHTTP_OPTION_CONTEXT_VALUE, &back, &back_size) != FALSE && back == 0x1234,
              "and queried back");
        DWORD type = 0, type_size = sizeof(type);
        check(query_option(connection, WINHTTP_OPTION_HANDLE_TYPE, &type, &type_size) != FALSE &&
                  type == WINHTTP_HANDLE_TYPE_CONNECT,
              "WINHTTP_OPTION_HANDLE_TYPE names a connection");
        DWORD needed = 0;
        SetLastError(0);
        check(query_option(request, WINHTTP_OPTION_CONTEXT_VALUE, nullptr, &needed) == FALSE &&
                  GetLastError() == ERROR_INSUFFICIENT_BUFFER && needed == sizeof(DWORD_PTR),
              "a query without a buffer reports the size it needs");
        DWORD flags = 0, flags_size = sizeof(flags);
        SetLastError(0);
        check(query_option(request, WINHTTP_OPTION_SECURITY_FLAGS, &flags, &flags_size) == FALSE &&
                  GetLastError() == ERROR_WINHTTP_INVALID_OPTION,
              "other options are not answered");
        DWORD protocols = WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_2;
        check(set_option(session, WINHTTP_OPTION_SECURE_PROTOCOLS, &protocols, sizeof(protocols)) != FALSE,
              "but setting them is accepted");

        // Wrong handles.
        SetLastError(0);
        check(connect(request, L"example.invalid", 80, 0) == nullptr &&
                  GetLastError() == ERROR_WINHTTP_INCORRECT_HANDLE_TYPE,
              "a request is not a session");
        SetLastError(0);
        check(send(connection, nullptr, 0, nullptr, 0, 0, 0) == FALSE &&
                  GetLastError() == ERROR_WINHTTP_INCORRECT_HANDLE_TYPE,
              "a connection is not a request");
        SetLastError(0);
        check(connect(session, nullptr, 80, 0) == nullptr && GetLastError() == ERROR_INVALID_PARAMETER,
              "a connection needs a server name");

        check(close(request) != FALSE && close(connection) != FALSE && close(session) != FALSE,
              "the handles close, children first");
        SetLastError(0);
        check(close(request) == FALSE && GetLastError() == ERROR_INVALID_HANDLE, "a closed handle is gone");
        SetLastError(0);
        check(send(request, nullptr, 0, nullptr, 0, 0, 0) == FALSE && GetLastError() == ERROR_INVALID_HANDLE,
              "and cannot be used");
    }

    // Asynchronous: the send returns TRUE, the failure arrives through the inherited callback on another thread with
    // the request's context, and HANDLE_CLOSING comes last.
    Seen seen;
    seen.error_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    seen.closing_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    g_seen = &seen;
    {
        const HINTERNET session = open(L"test", WINHTTP_ACCESS_TYPE_NO_PROXY, nullptr, nullptr, WINHTTP_FLAG_ASYNC);
        check(set_callback(session, &on_status, WINHTTP_CALLBACK_FLAG_ALL_NOTIFICATIONS, 0) == nullptr,
              "WinHttpSetStatusCallback returns the previous callback (none)");
        const HINTERNET connection = connect(session, L"example.invalid", 443, 0);
        const HINTERNET request = open_request(connection, L"POST", L"/y", nullptr, WINHTTP_NO_REFERER,
                                               WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE);
        check(connection && request && seen.created.load() == 2,
              "the callback is inherited and reports both new handles as created");

        const auto start = std::chrono::steady_clock::now();
        const BOOL sent = send(request, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0xC0FFEE);
        check(sent != FALSE, "an asynchronous WinHttpSendRequest returns TRUE");
        const bool fired = WaitForSingleObject(seen.error_event, 2000) == WAIT_OBJECT_0;
        const double elapsed = ms_since(start);
        check(fired && seen.request_errors.load() == 1, "WINHTTP_CALLBACK_STATUS_REQUEST_ERROR arrives once");
        check(seen.api.load() == API_SEND_REQUEST && seen.error.load() == ERROR_WINHTTP_CANNOT_CONNECT,
              "for API_SEND_REQUEST with ERROR_WINHTTP_CANNOT_CONNECT");
        check(seen.context.load() == 0xC0FFEE, "with the context the request was sent with");
        check(seen.thread.load() != GetCurrentThreadId(), "on another thread");
        check(elapsed < 500.0, "within 500 ms");

        check(close(request) != FALSE, "the request closes");
        check(WaitForSingleObject(seen.closing_event, 2000) == WAIT_OBJECT_0 && seen.closing.load() == 1 &&
                  seen.closing_after_error.load(),
              "and HANDLE_CLOSING follows once, after the error");
        close(connection);
        close(session);
        check(seen.closing.load() == 3, "every handle reports HANDLE_CLOSING once");
    }

    // A callback that closes its request while handling the error, as libHttpClient-style code does.
    {
        seen.request_errors = 0;
        seen.closing = 0;
        seen.close_in_callback = close;
        ResetEvent(seen.closing_event);
        const HINTERNET session = open(L"test", WINHTTP_ACCESS_TYPE_NO_PROXY, nullptr, nullptr, WINHTTP_FLAG_ASYNC);
        set_callback(session, &on_status, WINHTTP_CALLBACK_FLAG_ALL_NOTIFICATIONS, 0);
        const HINTERNET connection = connect(session, L"example.invalid", 443, 0);
        const HINTERNET request = open_request(connection, L"GET", L"/z", nullptr, WINHTTP_NO_REFERER,
                                               WINHTTP_DEFAULT_ACCEPT_TYPES, 0);
        check(send(request, nullptr, 0, nullptr, 0, 0, 7) != FALSE, "a second asynchronous send is queued");
        check(WaitForSingleObject(seen.error_event, 2000) == WAIT_OBJECT_0, "its error arrives");
        check(WaitForSingleObject(seen.closing_event, 2000) == WAIT_OBJECT_0 && seen.closing.load() == 1,
              "closing the request inside the callback reports HANDLE_CLOSING once");
        seen.close_in_callback = nullptr;
        close(connection);
        close(session);
    }

    // A request closed before its queued failure is delivered gets no error callback, only HANDLE_CLOSING.
    {
        seen.request_errors = 0;
        seen.closing = 0;
        const HINTERNET session = open(L"test", WINHTTP_ACCESS_TYPE_NO_PROXY, nullptr, nullptr, WINHTTP_FLAG_ASYNC);
        set_callback(session, &on_status, WINHTTP_CALLBACK_FLAG_HANDLES, 0);  // closing only
        const HINTERNET connection = connect(session, L"example.invalid", 443, 0);
        const HINTERNET request = open_request(connection, L"GET", L"/w", nullptr, WINHTTP_NO_REFERER,
                                               WINHTTP_DEFAULT_ACCEPT_TYPES, 0);
        send(request, nullptr, 0, nullptr, 0, 0, 0);
        close(request);
        close(connection);
        close(session);
        for (int i = 0; i < 100 && seen.closing.load() < 3; ++i) Sleep(10);
        check(seen.request_errors.load() == 0 && seen.closing.load() == 3,
              "notifications not asked for are not sent, and each handle still closes once");
    }
    g_seen = nullptr;
    CloseHandle(seen.error_event);
    CloseHandle(seen.closing_event);

    // WinHttpCrackUrl is parsing only and stays the host's.
    {
        wchar_t host[64] = {}, path[64] = {};
        URL_COMPONENTS parts{};
        parts.dwStructSize = sizeof(parts);
        parts.lpszHostName = host;
        parts.dwHostNameLength = 64;
        parts.lpszUrlPath = path;
        parts.dwUrlPathLength = 64;
        check(crack(L"https://example.invalid:8443/a/b", 0, 0, &parts) != FALSE && parts.nPort == 8443 &&
                  parts.nScheme == INTERNET_SCHEME_HTTPS && std::wcscmp(host, L"example.invalid") == 0 &&
                  std::wcscmp(path, L"/a/b") == 0,
              "WinHttpCrackUrl parses a URL");
    }

    // Sending a request from inside HANDLE_CREATED must see the connection's server and port already initialized.
    {
        CreatedRequestProbe probe{open_request, send, close};
        g_created_probe = &probe;
        const HINTERNET session = open(L"test", WINHTTP_ACCESS_TYPE_NO_PROXY, nullptr, nullptr, 0);
        check(set_callback(session, &on_created_send_request, WINHTTP_CALLBACK_FLAG_HANDLES, 0) == nullptr,
              "HANDLE_CREATED probe callback registers");
        StderrCapture capture;
        const HINTERNET connection = connect(session, L"created-cb.invalid", 8443, 0);
        const std::string logged = capture.stop();
        g_created_probe = nullptr;
        check(connection != nullptr && probe.created_handle == connection && probe.request_handle != nullptr &&
                  probe.send_result == FALSE && probe.send_error == ERROR_WINHTTP_CANNOT_CONNECT &&
                  probe.request_closed != FALSE,
              "request sent during connection HANDLE_CREATED fails with ERROR_WINHTTP_CANNOT_CONNECT");
        check(logged.find("[rt] winhttp: offline, refused created-cb.invalid:8443") != std::string::npos &&
                  logged.find("[rt] winhttp: offline, refused :0") == std::string::npos,
              "refused-server log inside HANDLE_CREATED records server:port instead of :0");
        close(connection);
        close(session);
    }

    // Closing the connection handle inside HANDLE_CREATED must not trigger a use-after-free, must report
    // HANDLE_CLOSING after HANDLE_CREATED returns, and must fail WinHttpConnect with ERROR_WINHTTP_OPERATION_CANCELLED
    // (measured against the host's winhttp.dll on Windows 11 26200).
    {
        CreatedCloseProbe probe{close};
        g_close_probe = &probe;
        const HINTERNET session = open(L"test", WINHTTP_ACCESS_TYPE_NO_PROXY, nullptr, nullptr, 0);
        check(set_callback(session, &on_created_close_handle, WINHTTP_CALLBACK_FLAG_HANDLES, 0) == nullptr,
              "HANDLE_CREATED close callback registers");
        SetLastError(0);
        const HINTERNET connection = connect(session, L"closed-in-handle-created-callback-long-host.invalid", 8443, 0);
        const DWORD connect_error = GetLastError();
        g_close_probe = nullptr;
        check(connection == nullptr && connect_error == ERROR_WINHTTP_OPERATION_CANCELLED &&
                  probe.created_calls == 1 && probe.created_handle != nullptr && probe.close_result != FALSE &&
                  !probe.closing_during_created && probe.closing_calls == 1 &&
                  probe.closing_handle == probe.created_handle,
              "closing connection handle inside HANDLE_CREATED reports HANDLE_CLOSING after HANDLE_CREATED and fails with ERROR_WINHTTP_OPERATION_CANCELLED");
        SetLastError(0);
        check(close(probe.created_handle) == FALSE && GetLastError() == ERROR_INVALID_HANDLE,
              "connection handle closed inside HANDLE_CREATED is already invalid after WinHttpConnect returns");
        g_close_probe = &probe;
        const BOOL session_closed = close(session);
        g_close_probe = nullptr;
        check(session_closed != FALSE && probe.closing_calls == 2 && probe.closing_handle == session,
              "closing session afterwards drops the last reference and reports HANDLE_CLOSING once");
    }

    std::printf("%d check(s), %d failure(s)\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
