// GameInput shim on SDL3: runtime-side controls and test seams. The GameInput API itself is reached the way a title
// reaches it, through GameInputCreate in the shim registry (gameinput.dll / xgameruntime.dll). See docs/gameinput.md.
#pragma once
#include <cstdint>

namespace rt::gameinput {

// How the impulse-trigger motors (GameInputRumbleParams::leftTrigger/rightTrigger) reach a DualSense.
enum class DsTriggerMode { Off, Rumble, Resistance };

// The keyboard pad: one extra virtual gamepad driven by the keyboard. Focused reads keys only while a window of this
// process is in the foreground; Always reads them whatever has focus (console tools such as gameinput_probe).
enum class KeyboardPadMode { Off, Focused, Always };

struct Config {
    DsTriggerMode ds_triggers = DsTriggerMode::Rumble;  // LEONIDA_DS_TRIGGERS=off|rumble|resistance
    float ds_trigger_gain = 1.0f;                       // LEONIDA_DS_TRIGGER_GAIN, 0..4, scales the effect strength
    float mouse_sensitivity = 1.0f;                     // LEONIDA_MOUSE_SENSITIVITY, 0.1..10, scales mouse movement
    uint8_t ds_trigger_frequency = 30;                  // LEONIDA_DS_TRIGGER_FREQ, Hz for the rumble mode, 1..255
    bool lightbar_set = false;                          // LEONIDA_DS_LIGHTBAR=RRGGBB|off; unset keeps SDL's
    uint8_t lightbar[3] = {0, 0, 0};                    //   player colour
    bool rumble = true;                                 // LEONIDA_RUMBLE=0 turns all rumble off
    bool background_thread = true;                      // LEONIDA_GAMEINPUT_THREAD=0: poll only from poll()
    KeyboardPadMode keyboard_pad = KeyboardPadMode::Off;  // LEONIDA_KEYBOARD_PAD=1|focus|always|off
};

// The configuration in use. Read from the environment on the first GameInputCreate.
const Config& config();
// Re-reads the environment (tests). Takes effect for the next rumble update.
void reload_config();

// One step of the backend, on the calling thread: initialise SDL if needed, open new gamepads, close removed ones,
// take a reading from each, apply pending rumble/trigger output, and deliver queued callbacks unless a dispatcher
// owns them. With the background thread on (the default) this runs every 2 ms by itself; with it off
// (LEONIDA_GAMEINPUT_THREAD=0) a test calls it, always from the same thread, because SDL's joystick calls must be.
void poll();

// Number of gamepads currently open by the backend (tests, gameinput_probe).
int connected_count();

// What the shim last sent to SDL for the device that is index `i` in connection order. Test seam.
struct OutputState {
    uint16_t low = 0, high = 0;           // SDL_RumbleGamepad
    uint16_t left = 0, right = 0;         // SDL_RumbleGamepadTriggers
    uint8_t ds_right[11] = {}, ds_left[11] = {};  // last DualSense trigger effects sent
    bool is_dualsense = false;
    bool trigger_rumble = false;          // the pad takes SDL_RumbleGamepadTriggers
};
bool output_state(int i, OutputState* out);

// Where the keyboard pad reads keys from. Test seam: key_down(virtual_key) replaces GetAsyncKeyState, has_focus
// replaces the foreground-window check of KeyboardPadMode::Focused. nullptr restores the default for either.
using KeyDownFn = bool (*)(int virtual_key);
using HasFocusFn = bool (*)();
void set_keyboard_source(KeyDownFn key_down, HasFocusFn has_focus);

// Where the keyboard pad reads the mouse cursor from (screen pixels; false when it cannot be read). Test seam:
// replaces GetCursorPos. nullptr restores the default.
using CursorPosFn = bool (*)(int* x, int* y);
void set_cursor_source(CursorPosFn cursor_pos);

// One row of the keyboard pad's key map: a Windows virtual-key code and the SDL gamepad button it presses.
struct KeyBinding {
    int virtual_key;
    int sdl_button;  // SDL_GamepadButton
    const char* key_name;
};
// The fixed key map (docs/gameinput.md lists it). `count` receives the number of rows.
const KeyBinding* keyboard_bindings(int* count);

}  // namespace rt::gameinput
