// xgameruntime: XGameEventImpl, XGameUiImpl and XGameStreamingImpl, offline: three small classes a title calls early.
//
// Each XGameEvent*, XGameUi* and XGameStreaming* function in xgameruntime.lib asks QueryApiImpl for its class and
// calls one slot with its own arguments after `this`. The class and interface ids and the method order are
// xodus-gaming/xgameruntime-docs' COM pages (GPL-3.0; facts only, nothing copied); IUnknown takes slots 0-2, so each
// class's first method is slot 3. Signatures follow Microsoft's public GDK reference (learn.microsoft.com/gaming/gdk).
// No GDK header or library code is used.
//
//   XGameEventImpl      XGameEventWrite (telemetry) succeeds and is dropped; the first event's name is logged, and
//                       LEONIDA_LOG_GAME_EVENTS=1 logs every one.
//   XGameUiImpl         a message dialog or error dialog has no screen: its title and text go to the log and it
//                       completes at once with the dialog's default button (error dialogs: dismissed). Every other UI
//                       (profile cards, pickers, text entry, web authentication, invites) fails before starting.
//   XGameStreamingImpl  the game is never streamed: initialising succeeds, IsStreaming is false, there are no
//                       clients, the connection-state registration hands out a token that never fires.
//
// Slots whose signature the public pages don't give answer with a value that is safe whichever way the caller reads
// it: see `client_not_connected` (XGameStreaming) and `offline_failure`.
#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <mutex>

#include <XAsync.h>
#include <XAsyncProvider.h>
#include <XTaskQueue.h>

namespace {

using XUserHandle = void*;

// XGameErr.h values, from the GDK "Error codes" page.
constexpr HRESULT E_GAMESTREAMING_NOT_INITIALIZED = static_cast<HRESULT>(0x89245400);

constexpr GUID kIUnknown = {0x00000000, 0x0000, 0x0000, {0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}};

#define FN(f) reinterpret_cast<const void*>(&f)

// A process-wide singleton per class, as the other runtime classes: AddRef and Release don't free.
ULONG STDMETHODCALLTYPE object_add_ref(void*) { return 2; }
ULONG STDMETHODCALLTYPE object_release(void*) { return 1; }

// One class: its object (a pointer to its vtable) and the interface ids it answers with that vtable.
struct ClassObject {
    const void* const* vtable;
};

HRESULT answer(ClassObject* object, const GUID* begin, const GUID* end, REFIID iid, void** out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    if (iid == kIUnknown) {
        *out = object;
        return S_OK;
    }
    for (const GUID* id = begin; id != end; ++id)
        if (iid == *id) {
            *out = object;
            return S_OK;
        }
    return E_NOINTERFACE;
}

// A slot the docs don't name: E_NOTIMPL, so nothing reads outputs it didn't write.
template <const char* Class, int Slot> HRESULT STDMETHODCALLTYPE unknown_slot(void*) {
    static std::atomic<bool> s_logged{false};
    if (!s_logged.exchange(true))
        std::fprintf(stderr, "[rt] xgameruntime: %s slot %d (unknown in the docs) called; E_NOTIMPL\n", Class, Slot);
    return E_NOTIMPL;
}

// A UI the offline runtime doesn't have. The *Async function fails before starting, so the XAsyncBlock is never used
// and no callback runs; its Result partners return the same error. Arguments don't matter (x64: the caller owns the
// argument area).
template <const char* Class, int Slot> HRESULT STDMETHODCALLTYPE offline_failure(void*) {
    static std::atomic<bool> s_logged{false};
    if (!s_logged.exchange(true))
        std::fprintf(stderr, "[rt] xgameruntime: %s slot %d is not available offline (E_FAIL)\n", Class, Slot);
    return E_FAIL;
}

// ======================= XGameEventImpl =======================

constexpr char kEventName[] = "XGameEventImpl";
constexpr GUID kEventClass = {0xbbfbdcc7, 0xbfe7, 0x409b, {0xa5, 0xca, 0xed, 0xf0, 0x54, 0x96, 0x0b, 0x4d}};
std::atomic<long> g_events{0};

bool log_every_event() {
    static const bool on = [] {
        char value[4] = {};
        size_t length = 0;
        return getenv_s(&length, value, "LEONIDA_LOG_GAME_EVENTS") == 0 && length == 2 && value[0] == '1';
    }();
    return on;
}

// 3: HRESULT XGameEventWrite(XUserHandle user, const char* serviceConfigId, const char* playSessionId,
//                            const char* eventName, const char* dimensionsJson, const char* measurementsJson)
HRESULT STDMETHODCALLTYPE event_write(void*, XUserHandle, const char*, const char*, const char* name,
                                      const char* dimensions, const char* measurements) {
    if (!name) return E_INVALIDARG;
    const long n = g_events.fetch_add(1);
    if (log_every_event())
        std::fprintf(stderr, "[rt] game event %s dimensions=%s measurements=%s\n", name, dimensions ? dimensions : "",
                     measurements ? measurements : "");
    else if (n == 0)
        std::fprintf(stderr, "[rt] xgameruntime: game events are dropped offline (first: %s; "
                             "LEONIDA_LOG_GAME_EVENTS=1 logs them)\n", name);
    return S_OK;
}

HRESULT STDMETHODCALLTYPE event_query_interface(void*, REFIID iid, void** out);
const void* const kEventVtable[] = {
    FN(event_query_interface), FN(object_add_ref), FN(object_release),   // 0-2 IUnknown
    FN(event_write),                                                     // 3 XGameEventWrite
};
ClassObject g_event_object{kEventVtable};
const GUID kEventInterfaces[] = {kEventClass};
HRESULT STDMETHODCALLTYPE event_query_interface(void*, REFIID iid, void** out) {
    return answer(&g_event_object, std::begin(kEventInterfaces), std::end(kEventInterfaces), iid, out);
}

// ======================= XGameUiImpl =======================

constexpr char kUiName[] = "XGameUiImpl";
constexpr GUID kUiClass = {0xdfcd4649, 0x4ff8, 0x4043, {0xba, 0x07, 0x35, 0xd6, 0x07, 0xdf, 0x98, 0xb0}};
const GUID kUiInterfaces[] = {
    kUiClass,   // the class id, which a thunk may pass for its implemented interface (IXGameUiImpl4)
    {0x6eeaa73e, 0x9669, 0x43ad, {0xa2, 0xc7, 0xd0, 0xda, 0x4e, 0x1f, 0x50, 0xa1}},   // IXGameUiImpl
    {0x36a03122, 0x9ea3, 0x4a3a, {0xa8, 0xa4, 0x89, 0x9c, 0xfd, 0x85, 0xd7, 0xdb}},   // IXGameUiImpl2
    {0xade7eba1, 0x2093, 0x42ce, {0xa5, 0x44, 0xa5, 0x23, 0xa6, 0x67, 0x90, 0xe0}},   // IXGameUiImpl3
    {0xeaf669df, 0x5542, 0x4590, {0x99, 0xa3, 0x8d, 0xc0, 0x61, 0xf8, 0x37, 0xcc}},   // IXGameUiImpl4
};

enum class XGameUiMessageDialogButton : uint32_t { First = 0, Second = 1, Third = 2 };

const char kDialogIdentity = 0, kErrorDialogIdentity = 0;

// The answer of a message dialog rides in the async context: the button, which never needs more.
HRESULT CALLBACK dialog_provider(XAsyncOp op, const XAsyncProviderData* data) {
    switch (op) {
    case XAsyncOp::Begin:
        return XAsyncSchedule(data->async, 0);
    case XAsyncOp::DoWork:
        XAsyncComplete(data->async, S_OK, sizeof(XGameUiMessageDialogButton));
        return S_OK;
    case XAsyncOp::GetResult:
        if (!data->buffer || data->bufferSize < sizeof(XGameUiMessageDialogButton)) return E_INVALIDARG;
        *static_cast<XGameUiMessageDialogButton*>(data->buffer) =
            static_cast<XGameUiMessageDialogButton>(reinterpret_cast<uintptr_t>(data->context));
        return S_OK;
    default:
        return S_OK;
    }
}

HRESULT CALLBACK error_dialog_provider(XAsyncOp op, const XAsyncProviderData* data) {
    if (op == XAsyncOp::Begin) return XAsyncSchedule(data->async, 0);
    if (op == XAsyncOp::DoWork) XAsyncComplete(data->async, S_OK, 0);
    return S_OK;
}

// 3: HRESULT XGameUiShowMessageDialogAsync(XAsyncBlock* async, const char* titleText, const char* contentText,
//        const char* firstButtonText, const char* secondButtonText, const char* thirdButtonText,
//        XGameUiMessageDialogButton defaultButton, XGameUiMessageDialogButton cancelButton)
// ASSUMPTION: the parameter list is the one the GDK reference gives; only the result-page and the enum were checked
// for this file. The dialog is "answered" with its default button.
HRESULT STDMETHODCALLTYPE ui_message_dialog_async(void*, XAsyncBlock* async, const char* title, const char* content,
                                                  const char* first, const char* second, const char* third,
                                                  XGameUiMessageDialogButton default_button, XGameUiMessageDialogButton) {
    if (!async) return E_POINTER;
    const auto button = static_cast<uint32_t>(default_button);
    if (button > 2) return E_INVALIDARG;
    const char* labels[] = {first, second, third};
    std::fprintf(stderr, "[rt] game dialog: \"%s\": %s [buttons: %s%s%s%s%s; answered: %s]\n", title ? title : "",
                 content ? content : "", first ? first : "", second ? " / " : "", second ? second : "",
                 third ? " / " : "", third ? third : "", labels[button] ? labels[button] : "(default)");
    return XAsyncBegin(async, reinterpret_cast<void*>(static_cast<uintptr_t>(button)), &kDialogIdentity,
                       "XGameUiShowMessageDialogAsync", dialog_provider);
}

// 4: HRESULT XGameUiShowMessageDialogResult(XAsyncBlock* async, XGameUiMessageDialogButton* resultButton)
HRESULT STDMETHODCALLTYPE ui_message_dialog_result(void*, XAsyncBlock* async, XGameUiMessageDialogButton* button) {
    if (!button) return E_POINTER;
    return XAsyncGetResult(async, &kDialogIdentity, sizeof(*button), button, nullptr);
}

// 14: HRESULT XGameUiShowErrorDialogAsync(XAsyncBlock* async, HRESULT errorCode, const char* context)
HRESULT STDMETHODCALLTYPE ui_error_dialog_async(void*, XAsyncBlock* async, HRESULT code, const char* context) {
    if (!async) return E_POINTER;
    std::fprintf(stderr, "[rt] game error dialog: HRESULT 0x%08lX%s%s [dismissed]\n", static_cast<unsigned long>(code),
                 context ? " context " : "", context ? context : "");
    return XAsyncBegin(async, nullptr, &kErrorDialogIdentity, "XGameUiShowErrorDialogAsync", error_dialog_provider);
}

// 15: HRESULT XGameUiShowErrorDialogResult(XAsyncBlock* async)
HRESULT STDMETHODCALLTYPE ui_error_dialog_result(void*, XAsyncBlock* async) {
    return XAsyncGetResult(async, &kErrorDialogIdentity, 0, nullptr, nullptr);
}

// 16: XGameUiSetNotificationPositionHint(position): there are no notifications to place.
HRESULT STDMETHODCALLTYPE ui_set_notification_hint(void*) { return S_OK; }

template <int Slot> HRESULT STDMETHODCALLTYPE ui_offline(void* self) { return offline_failure<kUiName, Slot>(self); }
template <int Slot> HRESULT STDMETHODCALLTYPE ui_unknown(void* self) { return unknown_slot<kUiName, Slot>(self); }

HRESULT STDMETHODCALLTYPE ui_query_interface(void*, REFIID iid, void** out);
const void* const kUiVtable[] = {
    FN(ui_query_interface), FN(object_add_ref), FN(object_release),   // 0-2 IUnknown
    FN(ui_message_dialog_async),       // 3  XGameUiShowMessageDialogAsync
    FN(ui_message_dialog_result),      // 4  XGameUiShowMessageDialogResult
    FN(ui_offline<5>),                 // 5  XGameUiShowSendGameInviteAsync
    FN(ui_offline<6>),                 // 6  XGameUiShowSendGameInviteResult
    FN(ui_offline<7>),                 // 7  XGameUiShowPlayerProfileCardAsync
    FN(ui_offline<8>),                 // 8  XGameUiShowPlayerProfileCardResult
    FN(ui_offline<9>),                 // 9  XGameUiShowAchievementsAsync
    FN(ui_offline<10>),                // 10 XGameUiShowAchievementsResult
    FN(ui_offline<11>),                // 11 XGameUiShowPlayerPickerAsync
    FN(ui_offline<12>),                // 12 XGameUiShowPlayerPickerResultCount
    FN(ui_offline<13>),                // 13 XGameUiShowPlayerPickerResult
    FN(ui_error_dialog_async),         // 14 XGameUiShowErrorDialogAsync
    FN(ui_error_dialog_result),        // 15 XGameUiShowErrorDialogResult
    FN(ui_set_notification_hint),      // 16 XGameUiSetNotificationPositionHint
    FN(ui_offline<17>),                // 17 XGameUiShowTextEntryAsync
    FN(ui_offline<18>),                // 18 XGameUiShowTextEntryResultSize
    FN(ui_offline<19>),                // 19 XGameUiShowTextEntryResult
    FN(ui_unknown<20>),                // 20 (unknown in the docs)
    FN(ui_unknown<21>),                // 21 (unknown in the docs)
    FN(ui_unknown<22>),                // 22 (unknown in the docs)
    FN(ui_unknown<23>),                // 23 (unknown in the docs)
    FN(ui_offline<24>),                // 24 XGameUiShowWebAuthenticationAsync
    FN(ui_offline<25>),                // 25 XGameUiShowWebAuthenticationResultSize
    FN(ui_offline<26>),                // 26 XGameUiShowWebAuthenticationResult
    FN(ui_offline<27>),                // 27 XGameUiShowWebAuthenticationWithOptionsAsync
    FN(ui_unknown<28>),                // 28 (unknown in the docs)
    FN(ui_unknown<29>),                // 29 (unknown in the docs)
    FN(ui_offline<30>),                // 30 XGameUiShowMultiplayerActivityGameInviteAsync (2)
    FN(ui_offline<31>),                // 31 XGameUiShowMultiplayerActivityGameInviteResult (2)
    FN(ui_unknown<32>),                // 32 (unknown in the docs) (2)
    FN(ui_unknown<33>),                // 33 (unknown in the docs) (2)
    FN(ui_offline<34>),                // 34 XGameUiTextEntryOpen (2)
    FN(ui_offline<35>),                // 35 XGameUiTextEntryClose (2)
    FN(ui_offline<36>),                // 36 XGameUiTextEntryGetState (2)
    FN(ui_offline<37>),                // 37 XGameUiTextEntryGetExtents (2)
    FN(ui_offline<38>),                // 38 XGameUiTextEntryUpdatePositionHint (2)
    FN(ui_offline<39>),                // 39 XGameUiTextEntryUpdateVisibility (2)
    FN(ui_offline<40>),                // 40 XGameUiShowStateShareAsync (3)
    FN(ui_offline<41>),                // 41 XGameUiShowStateShareResult (3)
    FN(ui_unknown<42>),                // 42 XGameUiSetUiCallbacks (4, PC-hosted UI; signature not given)
    FN(ui_unknown<43>),                // 43 XGameUiSetMessageDialogUiResponse (4)
    FN(ui_unknown<44>),                // 44 XGameUiSetPlayerPickerUiResponse (4)
    FN(ui_unknown<45>),                // 45 XGameUiSetTextEntryUiResponse (4)
    FN(ui_unknown<46>),                // 46 XGameUiSetPlayerProfileCardUiResponse (4)
    FN(ui_unknown<47>),                // 47 XGameUiSetSendGameInviteUiResponse (4)
    FN(ui_unknown<48>),                // 48 XGameUiSetAchievementsUiResponse (4)
    FN(ui_unknown<49>),                // 49 XGameUiSetMultiplayerActivityGameInviteUiResponse (4)
    FN(ui_unknown<50>),                // 50 XGameUiSetErrorDialogUiResponse (4)
};
static_assert(std::size(kUiVtable) == 51, "IUnknown + IXGameUiImpl's 27 methods + 10 + 2 + 9");
ClassObject g_ui_object{kUiVtable};
HRESULT STDMETHODCALLTYPE ui_query_interface(void*, REFIID iid, void** out) {
    return answer(&g_ui_object, std::begin(kUiInterfaces), std::end(kUiInterfaces), iid, out);
}

// ======================= XGameStreamingImpl =======================

constexpr GUID kStreamingClass = {0x0a2192aa, 0xb2d5, 0x4d58, {0x83, 0xbe, 0x38, 0x3b, 0x6d, 0x80, 0x79, 0x9e}};
const GUID kStreamingInterfaces[] = {
    kStreamingClass,   // the class id, for its implemented interface (IXGameStreamingImpl3)
    {0x8aff07f5, 0xa1bf, 0x4db8, {0x80, 0xa5, 0x31, 0xcc, 0xa0, 0xde, 0x51, 0xb7}},   // IXGameStreamingImpl
    {0x5f5e5169, 0x746c, 0x4001, {0xad, 0x1c, 0xda, 0x72, 0x8d, 0x01, 0xc9, 0xeb}},   // IXGameStreamingImpl2
    {0x57786622, 0x6605, 0x46d0, {0xb9, 0x17, 0x0f, 0x22, 0xbb, 0xcd, 0x9c, 0x52}},   // IXGameStreamingImpl3
};

std::mutex g_streaming_mutex;
uint64_t g_next_token = 1;

// 3: HRESULT XGameStreamingInitialize()
HRESULT STDMETHODCALLTYPE streaming_initialize(void*) { return S_OK; }
// 4: void XGameStreamingUninitialize()
void STDMETHODCALLTYPE streaming_uninitialize(void*) {}
// 5: bool XGameStreamingIsStreaming(): no client is ever connected.
bool STDMETHODCALLTYPE streaming_is_streaming(void*) { return false; }
// 9: uint32_t XGameStreamingGetClientCount()
uint32_t STDMETHODCALLTYPE streaming_client_count(void*) { return 0; }

// 12: HRESULT XGameStreamingRegisterConnectionStateChanged(XTaskQueueHandle queue, void* context,
//        XGameStreamingConnectionStateChangedCallback* callback, XTaskQueueRegistrationToken* token)
// ASSUMPTION: that parameter list (the GDK reference's, not re-checked for this file). No client connects, so the
// callback never runs.
HRESULT STDMETHODCALLTYPE streaming_register_connection(void*, XTaskQueueHandle, void*, void* callback,
                                                        XTaskQueueRegistrationToken* token) {
    if (!callback || !token) return E_INVALIDARG;
    std::lock_guard<std::mutex> lock(g_streaming_mutex);
    token->token = g_next_token++;
    return S_OK;
}

// Unregistering (7, 13): nothing is ever pending. Returns true, in case the caller reads a bool; a void caller
// ignores it.
bool STDMETHODCALLTYPE streaming_unregister(void*) { return true; }

// Every other slot asks about a client, its touch controls or its stream, and there is none. The answer is
// E_GAMESTREAMING_NOT_INITIALIZED (0x89245400), chosen for its shape: a caller reading an HRESULT sees a failure, and
// one reading a bool (al) sees false, since its low byte is 0. The slots returning an enum or a count are the ones
// above, answered explicitly.
template <int Slot> HRESULT STDMETHODCALLTYPE client_not_connected(void*) {
    static std::atomic<bool> s_logged{false};
    if (!s_logged.exchange(true))
        std::fprintf(stderr, "[rt] xgameruntime: XGameStreamingImpl slot %d: no streaming client offline\n", Slot);
    return E_GAMESTREAMING_NOT_INITIALIZED;
}
static_assert((static_cast<uint32_t>(E_GAMESTREAMING_NOT_INITIALIZED) & 0xFF) == 0, "reads as false in al");

// 11: XGameStreamingConnectionState XGameStreamingGetConnectionState(XGameStreamingClientId client): 0, the first
// enumerator. ASSUMPTION: it is Disconnected; there is no client to ask about either way.
uint32_t STDMETHODCALLTYPE streaming_connection_state(void*) { return 0; }

HRESULT STDMETHODCALLTYPE streaming_query_interface(void*, REFIID iid, void** out);
const void* const kStreamingVtable[] = {
    FN(streaming_query_interface), FN(object_add_ref), FN(object_release),   // 0-2 IUnknown
    FN(streaming_initialize),              // 3  XGameStreamingInitialize
    FN(streaming_uninitialize),            // 4  XGameStreamingUninitialize
    FN(streaming_is_streaming),            // 5  XGameStreamingIsStreaming
    FN(client_not_connected<6>),           // 6  XGameStreamingRegisterClientPropertiesChanged (per client)
    FN(streaming_unregister),              // 7  XGameStreamingUnregisterClientPropertiesChanged
    FN(client_not_connected<8>),           // 8  XGameStreamingGetStreamPhysicalDimensions
    FN(streaming_client_count),            // 9  XGameStreamingGetClientCount
    FN(client_not_connected<10>),          // 10 XGameStreamingGetClients
    FN(streaming_connection_state),        // 11 XGameStreamingGetConnectionState
    FN(streaming_register_connection),     // 12 XGameStreamingRegisterConnectionStateChanged
    FN(streaming_unregister),              // 13 XGameStreamingUnregisterConnectionStateChanged
    FN(client_not_connected<14>),          // 14 XGameStreamingGetStreamAddedLatency
    FN(client_not_connected<15>),          // 15 XGameStreamingGetServerLocationNameSize
    FN(client_not_connected<16>),          // 16 XGameStreamingGetServerLocationName
    FN(client_not_connected<17>),          // 17 XGameStreamingHideTouchControls
    FN(client_not_connected<18>),          // 18 XGameStreamingShowTouchControlLayout
    FN(client_not_connected<19>),          // 19 XGameStreamingHideTouchControlsOnClient
    FN(client_not_connected<20>),          // 20 XGameStreamingShowTouchControlLayoutOnClient
    FN(client_not_connected<21>),          // 21 XGameStreamingIsTouchInputEnabled (reads as false)
    FN(client_not_connected<22>),          // 22 XGameStreamingGetLastFrameDisplayed
    FN(client_not_connected<23>),          // 23 XGameStreamingGetAssociatedFrame
    FN(client_not_connected<24>),          // 24 XGameStreamingGetGamepadPhysicality
    FN(client_not_connected<25>),          // 25 XGameStreamingUpdateTouchControlsState
    FN(client_not_connected<26>),          // 26 XGameStreamingUpdateTouchControlsStateOnClient
    FN(client_not_connected<27>),          // 27 XGameStreamingShowTouchControlsWithStateUpdate
    FN(client_not_connected<28>),          // 28 XGameStreamingShowTouchControlsWithStateUpdateOnClient
    FN(client_not_connected<29>),          // 29 XGameStreamingGetTouchBundleVersionNameSize
    FN(client_not_connected<30>),          // 30 XGameStreamingGetTouchBundleVersion
    FN(client_not_connected<31>),          // 31 XGameStreamingGetClientIPAddress
    FN(client_not_connected<32>),          // 32 XGameStreamingGetSessionId (2)
    FN(client_not_connected<33>),          // 33 XGameStreamingGetDisplayDetails (3)
    FN(client_not_connected<34>),          // 34 XGameStreamingSetResolution (3)
};
static_assert(std::size(kStreamingVtable) == 35, "IUnknown + IXGameStreamingImpl's 29 methods + 1 + 2");
ClassObject g_streaming_object{kStreamingVtable};
HRESULT STDMETHODCALLTYPE streaming_query_interface(void*, REFIID iid, void** out) {
    return answer(&g_streaming_object, std::begin(kStreamingInterfaces), std::end(kStreamingInterfaces), iid, out);
}

#undef FN

}  // namespace

namespace rt::shims {

// Called by xgameruntime.cpp's QueryApiImpl for every class: true when the class is one of this file's, with *hr set.
bool xgameruntime_offline_query(const GUID& class_id, REFIID interface_id, void** out, HRESULT* hr) {
    if (class_id == kEventClass) {
        *hr = event_query_interface(nullptr, interface_id, out);
        return true;
    }
    if (class_id == kUiClass) {
        *hr = ui_query_interface(nullptr, interface_id, out);
        return true;
    }
    if (class_id == kStreamingClass) {
        *hr = streaming_query_interface(nullptr, interface_id, out);
        return true;
    }
    return false;
}

// For tests: how many game events were written.
long xgameruntime_offline_game_events() { return g_events.load(); }

}  // namespace rt::shims
