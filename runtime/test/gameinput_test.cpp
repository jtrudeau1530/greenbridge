// Checks the GameInput shim the way a title uses it, on SDL virtual joysticks, so no controller is needed.
//
// GameInputCreate is resolved through the shim registry like any import. Real controller backends are turned off
// with SDL hints, then virtual pads are attached and driven: an Xbox-style pad with trigger rumble, a DualSense
// (Sony vendor/product id, so SDL treats it as one and the shim takes the trigger-effect path) and a plain pad
// without trigger motors. The test covers enumeration and device callbacks, readings and their button/axis mapping,
// UI navigation, the guide button, rumble and trigger rumble as SDL receives them, the DualSense effect bytes for
// each LEONIDA_DS_TRIGGERS mode, the lightbar, a dispatcher, and unplugging and replugging. No game data.
#include <windows.h>

#include <SDL3/SDL.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "gameinput.h"
#include "gameinput_abi.h"
#include "shims.h"

using namespace gi;

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

bool approx(float a, float b) { return std::fabs(a - b) < 0.002f; }

// What SDL handed to one virtual pad.
struct Pad {
    SDL_JoystickID id = 0;
    SDL_Joystick* js = nullptr;
    int rumble_calls = 0, trigger_calls = 0, effect_calls = 0, led_calls = 0;
    uint16_t low = 0, high = 0, left = 0, right = 0;
    uint8_t effect[64] = {};
    int effect_size = 0;
    uint8_t led[3] = {};
    int player = -100;
};

bool SDLCALL on_rumble(void* u, Uint16 lo, Uint16 hi) {
    auto* p = static_cast<Pad*>(u);
    ++p->rumble_calls;
    p->low = lo;
    p->high = hi;
    return true;
}
bool SDLCALL on_triggers(void* u, Uint16 l, Uint16 r) {
    auto* p = static_cast<Pad*>(u);
    ++p->trigger_calls;
    p->left = l;
    p->right = r;
    return true;
}
bool SDLCALL on_led(void* u, Uint8 r, Uint8 g, Uint8 b) {
    auto* p = static_cast<Pad*>(u);
    ++p->led_calls;
    p->led[0] = r;
    p->led[1] = g;
    p->led[2] = b;
    return true;
}
bool SDLCALL on_effect(void* u, const void* data, int size) {
    auto* p = static_cast<Pad*>(u);
    ++p->effect_calls;
    p->effect_size = size;
    std::memcpy(p->effect, data, std::min<size_t>(size, sizeof p->effect));
    return true;
}
void SDLCALL on_player(void* u, int index) { static_cast<Pad*>(u)->player = index; }

void attach(Pad& pad, const char* name, uint16_t vendor, uint16_t product, bool trigger_motors) {
    SDL_VirtualJoystickDesc d;
    SDL_INIT_INTERFACE(&d);
    d.type = SDL_JOYSTICK_TYPE_GAMEPAD;
    d.vendor_id = vendor;
    d.product_id = product;
    d.naxes = SDL_GAMEPAD_AXIS_COUNT;
    d.nbuttons = SDL_GAMEPAD_BUTTON_COUNT;
    d.button_mask = (1u << SDL_GAMEPAD_BUTTON_COUNT) - 1;
    d.axis_mask = (1u << SDL_GAMEPAD_AXIS_COUNT) - 1;
    d.name = name;
    d.userdata = &pad;
    d.Rumble = on_rumble;
    d.RumbleTriggers = trigger_motors ? on_triggers : nullptr;
    d.SetLED = on_led;
    d.SendEffect = on_effect;
    d.SetPlayerIndex = on_player;
    pad.id = SDL_AttachVirtualJoystick(&d);
    pad.js = pad.id ? SDL_OpenJoystick(pad.id) : nullptr;
    if (pad.js) {  // triggers at rest: the full-range virtual axis maps -32768 to 0
        SDL_SetJoystickVirtualAxis(pad.js, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, -32768);
        SDL_SetJoystickVirtualAxis(pad.js, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, -32768);
    }
}

void detach(Pad& pad) {
    if (pad.js) SDL_CloseJoystick(pad.js);
    SDL_DetachVirtualJoystick(pad.id);
    pad.js = nullptr;
}

// Callback recorders.
struct DeviceEvent {
    IGameInputDevice* device;
    uint32_t current, previous;
    GameInputCallbackToken token;
};
std::vector<DeviceEvent> g_device_events;
int g_reading_events = 0;
std::vector<bool> g_guide_events;

void CALLBACK on_device(GameInputCallbackToken token, void*, IGameInputDevice* device, uint64_t, uint32_t current,
                        uint32_t previous) {
    g_device_events.push_back({device, current, previous, token});
}
void CALLBACK on_reading(GameInputCallbackToken, void*, IGameInputReading* reading, bool) {
    if (reading) ++g_reading_events;
}
void CALLBACK on_guide(GameInputCallbackToken, void*, IGameInputDevice*, uint64_t, bool pressed) {
    g_guide_events.push_back(pressed);
}

void set_env(const char* name, const char* value) { SetEnvironmentVariableA(name, value); }

// ---- keyboard pad: a fake key source instead of GetAsyncKeyState ----
bool g_keys[256] = {};
bool g_focus = false;
bool fake_key_down(int vk) { return vk >= 0 && vk < 256 && g_keys[vk]; }
bool fake_focus() { return g_focus; }

GameInputGamepadState pad_state(IGameInput* input, IGameInputDevice* dev, uint32_t* ui = nullptr) {
    GameInputGamepadState st{};
    IGameInputReading* r = nullptr;
    if (input->GetCurrentReading(GameInputKindGamepad, dev, &r) == S_OK && r) {
        r->GetGamepadState(&st);
        if (ui) {
            GameInputUiNavigationState nav{};
            r->GetUiNavigationState(&nav);
            *ui = nav.buttons;
        }
        r->Release();
    }
    return st;
}

// Runs with no other pad connected and the device callback from main still registered.
void test_keyboard_pad(IGameInput* input) {
    rt::gameinput::set_keyboard_source(fake_key_down, fake_focus);

    set_env("LEONIDA_KEYBOARD_PAD", nullptr);
    rt::gameinput::reload_config();
    rt::gameinput::poll();
    check(rt::gameinput::connected_count() == 0, "keyboard pad: off by default");

    set_env("LEONIDA_KEYBOARD_PAD", "1");
    rt::gameinput::reload_config();
    check(rt::gameinput::config().keyboard_pad == rt::gameinput::KeyboardPadMode::Focused,
          "keyboard pad: LEONIDA_KEYBOARD_PAD=1 is the focused mode");
    const size_t events_before = g_device_events.size();
    rt::gameinput::poll();
    check(rt::gameinput::connected_count() == 1, "keyboard pad: attached as one gamepad");
    // Main leaves three device callbacks registered (blocking, async and blocking enumeration), so one connect is
    // reported to each of them: all with the same device.
    IGameInputDevice* kb = g_device_events.size() > events_before ? g_device_events.back().device : nullptr;
    bool connect_events = g_device_events.size() == events_before + 3;
    for (size_t i = events_before; i < g_device_events.size(); ++i)
        connect_events &= g_device_events[i].device == kb && (g_device_events[i].current & GameInputDeviceConnected) &&
                          g_device_events[i].previous == GameInputDeviceNoStatus;
    check(connect_events, "keyboard pad: device callbacks on connect");
    if (!kb) return;
    const GameInputDeviceInfo* info = kb->GetDeviceInfo();
    check(info->displayName && std::strcmp(info->displayName->data, "Keyboard pad (Leonida)") == 0,
          "keyboard pad: display name");
    check((info->supportedInput & GameInputKindGamepad) && info->supportedRumbleMotors == 0,
          "keyboard pad: a gamepad without motors");
    check(pad_state(input, kb).buttons == 0, "keyboard pad: at rest");

    // Out of focus, keys are ignored.
    g_focus = false;
    g_keys[VK_RETURN] = true;
    rt::gameinput::poll();
    check(pad_state(input, kb).buttons == 0, "keyboard pad: no buttons while another window has focus");

    // In focus, every binding presses its button.
    g_focus = true;
    g_keys[VK_RETURN] = false;
    struct Expect { int vk; uint32_t button; const char* what; };
    const Expect expect[] = {
        {VK_UP, GameInputGamepadDPadUp, "keyboard pad: Up is d-pad up"},
        {VK_DOWN, GameInputGamepadDPadDown, "keyboard pad: Down is d-pad down"},
        {VK_LEFT, GameInputGamepadDPadLeft, "keyboard pad: Left is d-pad left"},
        {VK_RIGHT, GameInputGamepadDPadRight, "keyboard pad: Right is d-pad right"},
        {VK_RETURN, GameInputGamepadA, "keyboard pad: Enter is A"},
        {VK_BACK, GameInputGamepadB, "keyboard pad: Backspace is B"},
        {'X', GameInputGamepadX, "keyboard pad: X is X"},
        {'Y', GameInputGamepadY, "keyboard pad: Y is Y"},
        {'Q', GameInputGamepadLeftShoulder, "keyboard pad: Q is the left shoulder"},
        {'E', GameInputGamepadRightShoulder, "keyboard pad: E is the right shoulder"},
        {VK_ESCAPE, GameInputGamepadMenu, "keyboard pad: Esc is Menu"},
        {VK_TAB, GameInputGamepadView, "keyboard pad: Tab is View"},
    };
    for (const Expect& e : expect) {
        g_keys[e.vk] = true;
        rt::gameinput::poll();
        check(pad_state(input, kb).buttons == e.button, e.what);
        g_keys[e.vk] = false;
    }
    int bindings = 0;
    rt::gameinput::keyboard_bindings(&bindings);
    check(bindings == static_cast<int>(std::size(expect)), "keyboard pad: every binding is tested");

    // Two keys at once, and UI navigation from the d-pad and A.
    g_keys[VK_DOWN] = g_keys[VK_RETURN] = true;
    rt::gameinput::poll();
    uint32_t ui = 0;
    const GameInputGamepadState both = pad_state(input, kb, &ui);
    check(both.buttons == (GameInputGamepadDPadDown | GameInputGamepadA), "keyboard pad: Down and Enter together");
    check((ui & GameInputUiNavigationDown) && (ui & GameInputUiNavigationAccept), "keyboard pad: UI navigation down and accept");
    check(approx(both.leftTrigger, 0) && approx(both.rightTrigger, 0) && approx(both.leftThumbstickX, 0) &&
              approx(both.rightThumbstickY, 0),
          "keyboard pad: sticks and triggers stay at rest");

    // Losing focus releases held keys.
    g_focus = false;
    rt::gameinput::poll();
    check(pad_state(input, kb).buttons == 0, "keyboard pad: losing focus releases every button");

    // The always mode ignores focus.
    set_env("LEONIDA_KEYBOARD_PAD", "always");
    rt::gameinput::reload_config();
    rt::gameinput::poll();
    check(pad_state(input, kb).buttons == (GameInputGamepadDPadDown | GameInputGamepadA),
          "keyboard pad: LEONIDA_KEYBOARD_PAD=always reads keys without focus");
    g_keys[VK_DOWN] = g_keys[VK_RETURN] = false;

    // Turning it off removes the pad; turning it on again brings back the same device object.
    set_env("LEONIDA_KEYBOARD_PAD", "off");
    rt::gameinput::reload_config();
    rt::gameinput::poll();
    check(rt::gameinput::connected_count() == 0 && kb->GetDeviceStatus() == GameInputDeviceNoStatus,
          "keyboard pad: off removes it");
    set_env("LEONIDA_KEYBOARD_PAD", "1");
    rt::gameinput::reload_config();
    rt::gameinput::poll();
    check(rt::gameinput::connected_count() == 1 && g_device_events.back().device == kb &&
              (kb->GetDeviceStatus() & GameInputDeviceConnected),
          "keyboard pad: on again is the same device object");

    set_env("LEONIDA_KEYBOARD_PAD", nullptr);
    rt::gameinput::reload_config();
    rt::gameinput::poll();
    check(rt::gameinput::connected_count() == 0, "keyboard pad: removed at the end");
    rt::gameinput::set_keyboard_source(nullptr, nullptr);
}

}  // namespace

int main() {
    // Only virtual pads: no real controller on the test machine may show up.
    for (const char* hint : {SDL_HINT_JOYSTICK_HIDAPI, SDL_HINT_JOYSTICK_RAWINPUT, SDL_HINT_JOYSTICK_WGI,
                             SDL_HINT_JOYSTICK_GAMEINPUT, SDL_HINT_XINPUT_ENABLED, SDL_HINT_JOYSTICK_DIRECTINPUT})
        SDL_SetHintWithPriority(hint, "0", SDL_HINT_OVERRIDE);
    set_env("LEONIDA_GAMEINPUT_THREAD", "0");
    set_env("LEONIDA_DS_TRIGGERS", nullptr);
    set_env("LEONIDA_DS_LIGHTBAR", nullptr);
    set_env("LEONIDA_RUMBLE", nullptr);

    auto create = reinterpret_cast<GameInputCreateFn>(rt::shims::find("gameinput.dll", "GameInputCreate"));
    check(create != nullptr, "gameinput.dll!GameInputCreate is registered");
    check(rt::shims::find("xgameruntime.dll", "GameInputCreate") == reinterpret_cast<void*>(create),
          "xgameruntime.dll!GameInputCreate is the same function");
    if (!create) return 1;

    IGameInput* input = nullptr;
    check(create(&input) == S_OK && input, "GameInputCreate returns an instance");
    check(create(nullptr) == E_POINTER, "GameInputCreate(nullptr) is E_POINTER");
    IUnknown* unk = nullptr;
    check(input->QueryInterface(IID_IGameInput, reinterpret_cast<void**>(&unk)) == S_OK && unk == input,
          "QueryInterface(IID_IGameInput)");
    if (unk) unk->Release();

    rt::gameinput::poll();  // initialises SDL on this thread
    check(rt::gameinput::connected_count() == 0, "no gamepads before any is attached");

    IGameInputReading* reading = nullptr;
    check(input->GetCurrentReading(GameInputKindGamepad, nullptr, &reading) == GAMEINPUT_E_READING_NOT_FOUND && !reading,
          "GetCurrentReading with no device is READING_NOT_FOUND");

    GameInputCallbackToken device_token = 0, reading_token = 0, guide_token = 0;
    check(input->RegisterDeviceCallback(nullptr, GameInputKindGamepad, GameInputDeviceConnected,
                                        GameInputBlockingEnumeration, nullptr, on_device, &device_token) == S_OK &&
              device_token != 0,
          "RegisterDeviceCallback (blocking enumeration, nothing connected)");
    check(g_device_events.empty(), "no enumeration callback without devices");
    check(input->RegisterReadingCallback(nullptr, GameInputKindGamepad, 0.0f, nullptr, on_reading, &reading_token) == S_OK,
          "RegisterReadingCallback");
    check(input->RegisterGuideButtonCallback(nullptr, nullptr, on_guide, &guide_token) == S_OK, "RegisterGuideButtonCallback");

    // ---- an Xbox-style pad with impulse triggers ----
    Pad xbox;
    attach(xbox, "Leonida virtual Xbox pad", 0x045E, 0x0B12, true);
    check(xbox.js != nullptr, "virtual Xbox pad attached");
    rt::gameinput::poll();
    check(rt::gameinput::connected_count() == 1, "the shim opened it");
    check(g_device_events.size() == 1 && g_device_events[0].previous == GameInputDeviceNoStatus &&
              (g_device_events[0].current & GameInputDeviceConnected),
          "device callback: connected");
    IGameInputDevice* dev = g_device_events.empty() ? nullptr : g_device_events[0].device;
    if (!dev) return 1;
    dev->AddRef();

    const GameInputDeviceInfo* info = dev->GetDeviceInfo();
    check(info->infoSize == sizeof(GameInputDeviceInfo), "info.infoSize");
    check(info->vendorId == 0x045E && info->productId == 0x0B12, "info vendor and product id");
    check((info->supportedInput & GameInputKindGamepad) && (info->supportedInput & GameInputKindUiNavigation),
          "info.supportedInput has Gamepad and UiNavigation");
    check(info->supportedRumbleMotors == (GameInputRumbleLowFrequency | GameInputRumbleHighFrequency |
                                          GameInputRumbleLeftTrigger | GameInputRumbleRightTrigger),
          "info.supportedRumbleMotors: both motors and both triggers");
    check(info->controllerAxisCount == 6 && info->controllerAxisInfo && info->gamepadInfo && info->displayName &&
              std::strcmp(info->displayName->data, "Leonida virtual Xbox pad") == 0,
          "info axis table, gamepad info and display name");
    check(dev->GetDeviceStatus() & GameInputDeviceConnected, "GetDeviceStatus: connected");
    check(xbox.player == 0, "player index 0 for the first pad");

    // Initial reading: everything at rest.
    check(input->GetCurrentReading(GameInputKindGamepad, dev, &reading) == S_OK && reading, "initial reading");
    GameInputGamepadState st{};
    if (reading) {
        check(reading->GetGamepadState(&st) && st.buttons == 0 && approx(st.leftTrigger, 0) && approx(st.leftThumbstickX, 0),
              "initial gamepad state is at rest");
        IGameInputDevice* rd = nullptr;
        reading->GetDevice(&rd);
        check(rd == dev, "reading->GetDevice is the device");
        if (rd) rd->Release();
    }
    const uint64_t seq0 = reading ? reading->GetSequenceNumber(GameInputKindGamepad) : 0;

    // Press A, push the left stick up-right, pull the left trigger, press the right shoulder and dpad down.
    const int reading_events_before = g_reading_events;
    SDL_SetJoystickVirtualButton(xbox.js, SDL_GAMEPAD_BUTTON_SOUTH, true);
    SDL_SetJoystickVirtualButton(xbox.js, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, true);
    SDL_SetJoystickVirtualButton(xbox.js, SDL_GAMEPAD_BUTTON_DPAD_DOWN, true);
    SDL_SetJoystickVirtualButton(xbox.js, SDL_GAMEPAD_BUTTON_START, true);
    SDL_SetJoystickVirtualAxis(xbox.js, SDL_GAMEPAD_AXIS_LEFTX, 32767);
    SDL_SetJoystickVirtualAxis(xbox.js, SDL_GAMEPAD_AXIS_LEFTY, -32768);  // SDL: up is negative
    SDL_SetJoystickVirtualAxis(xbox.js, SDL_GAMEPAD_AXIS_RIGHTX, -32768);
    SDL_SetJoystickVirtualAxis(xbox.js, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, 32767);
    rt::gameinput::poll();
    check(g_reading_events > reading_events_before, "reading callback after a change");

    IGameInputReading* next = nullptr;
    check(input->GetNextReading(reading, GameInputKindGamepad, nullptr, &next) == S_OK && next,
          "GetNextReading after a change");
    if (next) next->Release();
    if (reading) reading->Release();
    reading = nullptr;
    check(input->GetCurrentReading(GameInputKindGamepad, nullptr, &reading) == S_OK && reading,
          "GetCurrentReading(any device)");
    if (reading) {
        st = {};
        reading->GetGamepadState(&st);
        std::printf("     buttons=0x%x lt=%.3f rt=%.3f lx=%.3f ly=%.3f rx=%.3f ry=%.3f\n", st.buttons, st.leftTrigger,
                    st.rightTrigger, st.leftThumbstickX, st.leftThumbstickY, st.rightThumbstickX, st.rightThumbstickY);
        check(st.buttons == (GameInputGamepadA | GameInputGamepadRightShoulder | GameInputGamepadDPadDown | GameInputGamepadMenu),
              "buttons: A, right shoulder, dpad down, Menu (Start)");
        check(approx(st.leftThumbstickX, 1) && approx(st.leftThumbstickY, 1), "left stick up-right is (+1, +1)");
        check(approx(st.rightThumbstickX, -1) && approx(st.rightThumbstickY, 0), "right stick left is (-1, 0)");
        check(approx(st.leftTrigger, 1) && approx(st.rightTrigger, 0), "left trigger 1, right trigger 0");
        check(reading->GetSequenceNumber(GameInputKindGamepad) > seq0, "sequence number advanced");
        GameInputUiNavigationState ui{};
        check(reading->GetUiNavigationState(&ui), "GetUiNavigationState");
        const uint32_t want_ui = GameInputUiNavigationAccept | GameInputUiNavigationMenu | GameInputUiNavigationDown |
                                 GameInputUiNavigationUp | GameInputUiNavigationRight | GameInputUiNavigationPageRight |
                                 GameInputUiNavigationPageUp | GameInputUiNavigationScrollLeft;
        check(ui.buttons == want_ui, "UI navigation derived from the pad");
        float axes[8] = {};
        check(reading->GetControllerAxisState(8, axes) == 6 && approx(axes[0], 1) && approx(axes[3], 0.5f) && approx(axes[4], 1),
              "controller axes (0..1, sticks centred at 0.5)");
        bool buttons[64] = {};
        check(reading->GetControllerButtonState(64, buttons) == SDL_GAMEPAD_BUTTON_COUNT && buttons[SDL_GAMEPAD_BUTTON_SOUTH] &&
                  !buttons[SDL_GAMEPAD_BUTTON_EAST],
              "controller buttons");
        check(!reading->GetMotionState(nullptr) && !reading->GetMouseState(nullptr), "no motion or mouse state");
        IGameInputReading* none = nullptr;
        check(input->GetNextReading(reading, GameInputKindGamepad, nullptr, &none) == GAMEINPUT_E_READING_NOT_FOUND && !none,
              "GetNextReading at the newest reading is READING_NOT_FOUND");
    }
    // Nothing changed: no new reading, no callback.
    const int events_now = g_reading_events;
    rt::gameinput::poll();
    check(g_reading_events == events_now, "no reading callback without a change");

    // Guide button.
    SDL_SetJoystickVirtualButton(xbox.js, SDL_GAMEPAD_BUTTON_GUIDE, true);
    rt::gameinput::poll();
    SDL_SetJoystickVirtualButton(xbox.js, SDL_GAMEPAD_BUTTON_GUIDE, false);
    rt::gameinput::poll();
    check(g_guide_events.size() == 2 && g_guide_events[0] && !g_guide_events[1], "guide button callback: down, up");

    // Rumble: both motors and both trigger motors reach SDL.
    GameInputRumbleParams rumble{0.5f, 1.0f, 0.25f, 0.75f};
    dev->SetRumbleState(&rumble);
    rt::gameinput::poll();
    std::printf("     rumble low=%u high=%u left=%u right=%u\n", xbox.low, xbox.high, xbox.left, xbox.right);
    check(xbox.low == 32768 && xbox.high == 65535, "SetRumbleState motors -> SDL_RumbleGamepad");
    check(xbox.left == 16384 && xbox.right == 49151, "SetRumbleState triggers -> SDL_RumbleGamepadTriggers");
    check(xbox.effect_calls == 0, "no DualSense effect for an Xbox pad");
    const int rumble_calls = xbox.rumble_calls;
    rt::gameinput::poll();
    check(xbox.rumble_calls == rumble_calls, "unchanged rumble is not re-sent every poll");
    dev->SetRumbleState(nullptr);
    rt::gameinput::poll();
    check(xbox.low == 0 && xbox.high == 0 && xbox.left == 0 && xbox.right == 0, "SetRumbleState(nullptr) stops rumble");

    set_env("LEONIDA_RUMBLE", "0");
    rt::gameinput::reload_config();
    dev->SetRumbleState(&rumble);
    rt::gameinput::poll();
    check(xbox.low == 0 && xbox.left == 0, "LEONIDA_RUMBLE=0 keeps the motors off");
    set_env("LEONIDA_RUMBLE", nullptr);
    rt::gameinput::reload_config();
    dev->SetRumbleState(nullptr);
    rt::gameinput::poll();

    // ---- a DualSense: trigger rumble becomes adaptive-trigger effects ----
    set_env("LEONIDA_DS_LIGHTBAR", "ff8000");
    rt::gameinput::reload_config();
    Pad ds;
    attach(ds, "Leonida virtual DualSense", 0x054C, 0x0CE6, true);
    rt::gameinput::poll();
    rt::gameinput::OutputState out{};
    check(rt::gameinput::connected_count() == 2 && rt::gameinput::output_state(1, &out) && out.is_dualsense,
          "a Sony 054c:0ce6 pad is handled as a DualSense");
    check(ds.player == 1, "player index 1 for the second pad");
    check(ds.led_calls >= 1 && ds.led[0] == 0xFF && ds.led[1] == 0x80 && ds.led[2] == 0x00, "LEONIDA_DS_LIGHTBAR=ff8000");
    IGameInputDevice* dsdev = g_device_events.empty() ? nullptr : g_device_events.back().device;
    check(dsdev && dsdev != dev && (dsdev->GetDeviceInfo()->supportedRumbleMotors & GameInputRumbleLeftTrigger),
          "DualSense device reported with trigger motors");
    if (!dsdev) return 1;

    GameInputRumbleParams ds_rumble{0.0f, 0.0f, 1.0f, 0.5f};
    dsdev->SetRumbleState(&ds_rumble);
    rt::gameinput::poll();
    std::printf("     effect size=%d bits=%02x R=%02x %02x %02x %02x L=%02x %02x %02x %02x\n", ds.effect_size,
                ds.effect[0], ds.effect[10], ds.effect[11], ds.effect[12], ds.effect[13], ds.effect[21], ds.effect[22],
                ds.effect[23], ds.effect[24]);
    check(ds.effect_size == 47 && ds.effect[0] == 0x0C, "effect block: 47 bytes, right+left trigger flags");
    check(ds.effect[10] == 0x06 && ds.effect[11] == 30 && ds.effect[12] == 128, "rumble mode: right trigger vibrates at half");
    check(ds.effect[21] == 0x06 && ds.effect[23] == 255, "rumble mode: left trigger vibrates at full");
    check(ds.trigger_calls == 0, "a DualSense gets effects, not SDL_RumbleGamepadTriggers");

    set_env("LEONIDA_DS_TRIGGERS", "resistance");
    rt::gameinput::reload_config();
    ds_rumble = {0, 0, 0.0f, 1.0f};
    dsdev->SetRumbleState(&ds_rumble);
    rt::gameinput::poll();
    check(ds.effect[10] == 0x01 && ds.effect[12] == 255 && ds.effect[21] == 0x05,
          "resistance mode: right trigger resists, left trigger off");

    set_env("LEONIDA_DS_TRIGGERS", "off");
    rt::gameinput::reload_config();
    const int effect_calls = ds.effect_calls;
    ds_rumble = {0, 0, 1.0f, 1.0f};
    dsdev->SetRumbleState(&ds_rumble);
    rt::gameinput::poll();
    check(ds.effect_calls == effect_calls, "LEONIDA_DS_TRIGGERS=off sends no trigger effect");
    set_env("LEONIDA_DS_TRIGGERS", nullptr);
    set_env("LEONIDA_DS_LIGHTBAR", nullptr);
    rt::gameinput::reload_config();

    // The most recent reading of any device: move the DualSense.
    SDL_SetJoystickVirtualButton(ds.js, SDL_GAMEPAD_BUTTON_EAST, true);
    rt::gameinput::poll();
    if (reading) reading->Release();
    reading = nullptr;
    if (input->GetCurrentReading(GameInputKindGamepad, nullptr, &reading) == S_OK && reading) {
        IGameInputDevice* rd = nullptr;
        reading->GetDevice(&rd);
        st = {};
        reading->GetGamepadState(&st);
        check(rd == dsdev && st.buttons == GameInputGamepadB, "GetCurrentReading(any) follows the latest change (B on the DualSense)");
        if (rd) rd->Release();
    } else {
        check(false, "GetCurrentReading(any) with two pads");
    }

    // ---- a pad without trigger motors ----
    Pad plain;
    attach(plain, "Leonida virtual plain pad", 0x057E, 0x2009, false);  // Switch Pro ids: rumble, no trigger motors
    rt::gameinput::poll();
    IGameInputDevice* plaindev = g_device_events.empty() ? nullptr : g_device_events.back().device;
    check(plaindev && (plaindev->GetDeviceInfo()->supportedRumbleMotors & GameInputRumbleLeftTrigger) == 0,
          "a pad without trigger motors does not report them");
    if (plaindev) {
        GameInputRumbleParams p{0.25f, 0.25f, 1.0f, 1.0f};
        plaindev->SetRumbleState(&p);
        rt::gameinput::poll();
        check(plain.low == 16384 && plain.trigger_calls == 0 && plain.effect_calls == 0,
              "its body motors rumble and the trigger request is dropped");
    }
    detach(plain);
    rt::gameinput::poll();

    // ---- dispatcher: callbacks wait for Dispatch ----
    IGameInputDispatcher* dispatcher = nullptr;
    check(input->CreateDispatcher(&dispatcher) == S_OK && dispatcher, "CreateDispatcher");
    HANDLE wait = nullptr;
    check(dispatcher && dispatcher->OpenWaitHandle(&wait) == S_OK && wait, "OpenWaitHandle");
    const int before_dispatch = g_reading_events;
    SDL_SetJoystickVirtualButton(xbox.js, SDL_GAMEPAD_BUTTON_NORTH, true);
    rt::gameinput::poll();
    check(g_reading_events == before_dispatch, "with a dispatcher, poll does not run callbacks");
    check(wait && WaitForSingleObject(wait, 0) == WAIT_OBJECT_0, "the wait handle is signalled");
    if (dispatcher) dispatcher->Dispatch(1000);
    check(g_reading_events > before_dispatch, "Dispatch runs them");
    if (wait) CloseHandle(wait);
    if (dispatcher) dispatcher->Release();

    // ---- unplug and replug ----
    const size_t events_before_unplug = g_device_events.size();
    detach(xbox);
    rt::gameinput::poll();
    check(g_device_events.size() == events_before_unplug + 1 && g_device_events.back().device == dev &&
              g_device_events.back().current == GameInputDeviceNoStatus &&
              (g_device_events.back().previous & GameInputDeviceConnected),
          "device callback: disconnected");
    check(dev->GetDeviceStatus() == GameInputDeviceNoStatus, "GetDeviceStatus after unplug");
    IGameInputReading* gone = nullptr;
    check(input->GetCurrentReading(GameInputKindGamepad, dev, &gone) == GAMEINPUT_E_DEVICE_DISCONNECTED && !gone,
          "GetCurrentReading on an unplugged device is DEVICE_DISCONNECTED");
    AppLocalDeviceId id = info->deviceId;
    IGameInputDevice* found = nullptr;
    check(input->FindDeviceFromId(&id, &found) == S_OK && found == dev, "FindDeviceFromId");
    if (found) found->Release();

    Pad xbox2;
    attach(xbox2, "Leonida virtual Xbox pad", 0x045E, 0x0B12, true);
    rt::gameinput::poll();
    check(g_device_events.back().device == dev && (g_device_events.back().current & GameInputDeviceConnected),
          "the same pad plugged back in is the same device object");
    check(xbox2.player == 0, "it takes the free player slot 0");

    // Async enumeration delivers on the next poll; unregistering stops callbacks.
    std::vector<DeviceEvent> before = g_device_events;
    GameInputCallbackToken async_token = 0;
    input->RegisterDeviceCallback(nullptr, GameInputKindGamepad, GameInputDeviceConnected, GameInputAsyncEnumeration,
                                  nullptr, on_device, &async_token);
    check(g_device_events.size() == before.size(), "async enumeration does not call back before returning");
    rt::gameinput::poll();
    size_t async_events = 0;
    for (size_t i = before.size(); i < g_device_events.size(); ++i) async_events += g_device_events[i].token == async_token;
    check(async_events == 2, "async enumeration reports both connected pads to the new callback only");

    check(input->UnregisterCallback(reading_token, 0), "UnregisterCallback");
    check(!input->UnregisterCallback(reading_token, 0), "a second UnregisterCallback returns false");
    const int after_unregister = g_reading_events;
    SDL_SetJoystickVirtualButton(xbox2.js, SDL_GAMEPAD_BUTTON_WEST, true);
    rt::gameinput::poll();
    check(g_reading_events == after_unregister, "no reading callback after unregistering");

    // Blocking enumeration with pads present runs before returning.
    before = g_device_events;
    GameInputCallbackToken blocking_token = 0;
    input->RegisterDeviceCallback(nullptr, GameInputKindGamepad, GameInputDeviceConnected, GameInputBlockingEnumeration,
                                  nullptr, on_device, &blocking_token);
    check(g_device_events.size() == before.size() + 2, "blocking enumeration reports both pads before returning");

    if (reading) reading->Release();
    dev->Release();
    detach(xbox2);
    detach(ds);
    rt::gameinput::poll();
    check(rt::gameinput::connected_count() == 0, "all pads closed after detaching");

    test_keyboard_pad(input);
    input->Release();

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
