// GameInput shim on SDL3: runtime-side controls and test seams. The GameInput API itself is reached the way a title
// reaches it, through GameInputCreate in the shim registry (gameinput.dll / xgameruntime.dll). See docs/gameinput.md.
#pragma once
#include <cstdint>

namespace rt::gameinput {

// How the impulse-trigger motors (GameInputRumbleParams::leftTrigger/rightTrigger) reach a DualSense.
enum class DsTriggerMode { Off, Rumble, Resistance };

struct Config {
    DsTriggerMode ds_triggers = DsTriggerMode::Rumble;  // LEONIDA_DS_TRIGGERS=off|rumble|resistance
    float ds_trigger_gain = 1.0f;                       // LEONIDA_DS_TRIGGER_GAIN, 0..4, scales the effect strength
    uint8_t ds_trigger_frequency = 30;                  // LEONIDA_DS_TRIGGER_FREQ, Hz for the rumble mode, 1..255
    bool lightbar_set = false;                          // LEONIDA_DS_LIGHTBAR=RRGGBB|off; unset keeps SDL's
    uint8_t lightbar[3] = {0, 0, 0};                    //   player colour
    bool rumble = true;                                 // LEONIDA_RUMBLE=0 turns all rumble off
    bool background_thread = true;                      // LEONIDA_GAMEINPUT_THREAD=0: poll only from poll()
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

}  // namespace rt::gameinput
