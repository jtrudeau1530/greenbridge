# Controller input: the GameInput shim on SDL3

The Xbox build reads controllers through Microsoft's GameInput API. `runtime/shims/gameinput.cpp` answers
`GameInputCreate` with its own implementation of that API, backed by [SDL3](https://libsdl.org) gamepads, so
every controller SDL supports works the way an Xbox pad would: DualSense, DualShock 4, Switch Pro, Xbox pads, and the
rest of SDL's controller database. On a DualSense the impulse-trigger rumble becomes adaptive-trigger effects, and
the lightbar colour can be set.

Without the shim the loader would bind `GameInputCreate` to the host's `GameInput.dll` if there is one
([#279](https://github.com/ItsNotRexed/Leonida/pull/279)). That works for Xbox pads only, and gives no DualSense
features. The game executable is not available yet, so whether it imports GameInput this way, and under which
module, is unverified (see [VERIFY](#verify)).

## Building

The shim is optional, and the build downloads nothing unless asked. SDL3 is zlib-licensed. CMake builds the shim when
SDL3 is available in one of these ways:

- **Installed SDL3**, 3.2 or newer, found by `find_package(SDL3)`: set `SDL3_DIR` or `CMAKE_PREFIX_PATH`, for example
  `-DSDL3_DIR=C:/SDL3/cmake`. No flag is needed.
- **`-DLEONIDA_SDL3_FETCH=ON`**: downloads the pinned release `SDL3-3.4.20.tar.gz` from SDL's GitHub releases once at
  configure time, checks its SHA-256, and builds it static with only the gamepad parts (no audio, video, GPU or
  render). That takes about a minute on the first build. `scripts/build_runtime.ps1 -FetchSdl3` passes it.
- **A local copy of that tarball**: extract it and pass `-DFETCHCONTENT_SOURCE_DIR_SDL3=<path to SDL3-3.4.20>`; it is
  built the same way, without a download.

Without any of these, or with `-DLEONIDA_SDL3=OFF`, configure prints a message and `gta6rt` is built without the shim:
it has no `GameInputCreate`, as before, so the loader binds it to the host's `GameInput.dll` or the unimplemented
thunk. `gameinput_test` and `gameinput_probe` are not built, and CTest lists `gameinput` as not run (disabled) rather
than failing.

`scripts/build_runtime.ps1` (what `make_game.py` runs) builds the shim only on its CMake path, when Visual Studio's
C++ CMake tools are installed. Its direct-`cl.exe` fallback compiles a fixed file list without the shim and warns
about it.

## Mapping

GameInput's buttons are positional, as on an Xbox pad, and SDL's are too, so the face buttons go by position, not by
label:

| GameInput | SDL3 | DualSense | Switch Pro |
|---|---|---|---|
| `A` | south | Cross | B |
| `B` | east | Circle | A |
| `X` | west | Square | Y |
| `Y` | north | Triangle | X |
| `View` | back | Create | Minus |
| `Menu` | start | Options | Plus |
| shoulders, stick clicks, d-pad | same | L1/R1, L3/R3, d-pad | L/R, stick clicks, d-pad |
| guide button callback | guide | PS | Home |

The DualSense touchpad click, mute button and paddles are not in GameInput's gamepad state; they are in the
controller-button array (`GetControllerButtonState`, indexed in SDL's `SDL_GamepadButton` order).

- Sticks: -1..1 with +Y up (SDL reports +Y down; the shim flips it). Triggers: 0..1. No deadzone is applied, as on
  console: the title applies its own.
- `GetUiNavigationState` is derived from the pad: A accept, B cancel, X/Y context 1/2, stick clicks context 3/4, d-pad
  or left stick past half for up/down/left/right, shoulders for page left/right, triggers past half for page up/down,
  right stick past half for scrolling, Menu and View as themselves.
- Controller axes (`GetControllerAxisState`): LX, LY, RX, RY, LT, RT in 0..1, sticks at rest at 0.5.
- Every pad is reported as family `GameInputFamilyXboxOne` with its real USB vendor and product id.
- Player LEDs (and SDL's default lightbar colour) follow connection order: the first pad is player 1, and a pad
  plugged in later takes the lowest free slot.

## Keyboard pad

The title reads gamepads only, so keyboard input has to arrive as a gamepad. With `LEONIDA_KEYBOARD_PAD` set, the
shim attaches one SDL virtual gamepad named `Keyboard pad (Leonida)` and presses its buttons from the keyboard on
every poll. It then goes through the same path as a real pad: a device object, the next free player slot, readings,
reading and device callbacks, and UI navigation derived from its buttons. It reports no rumble motors. It is off by
default.

| key | button |
|---|---|
| arrow keys | d-pad |
| Enter | `A` |
| Backspace | `B` |
| X | `X` |
| Y | `Y` |
| Q / E | left / right shoulder |
| Esc | `Menu` |
| Tab | `View` |
| W / A / S / D | left stick up / left / down / right |
| mouse movement | right stick |
| right mouse button | left trigger (aim) |
| left mouse button | right trigger (fire) |

Letters are Windows virtual-key codes, so `Y` is the key labelled Y on the active layout (it sits where Z does on a
QWERTY keyboard with a German layout). Mouse buttons are the physical ones: `GetAsyncKeyState` ignores the
swapped-buttons setting.

WASD pushes the left stick to full deflection. Opposite keys cancel (W+S is centered on Y), and a diagonal sits on
the round gate as a real stick reports it: about 0.707 on each axis, not 1.

The right stick follows the mouse's movement, not the cursor's position. Each poll (2 ms with the background thread)
the stick keeps 7/8 of its value and adds 1024 axis units (of 32767, times `LEONIDA_MOUSE_SENSITIVITY`) per count the mouse moved since the last poll,
clamped to full. Steady movement of 4 counts a poll (2000 a second) is full deflection; when the mouse stops, the
stick falls back from full to exactly 0 in 67 polls (134 ms). Moving the mouse down pushes the stick down.

The movement comes from raw input (`WM_INPUT`): what the mouse itself reports, so it does not stop at the edge of the
screen and Windows' pointer acceleration does not change it. When the keyboard pad attaches, the shim starts one
thread with a message-only window (never shown) registered for raw mouse input (`RIDEV_INPUTSINK`), which adds each
relative movement to a running total; the poll takes the difference of that total between polls. Absolute reports
(remote desktop, some virtual machines, pen tablets) are ignored. If the window or the registration fails, the shim
logs it and falls back to `GetCursorPos` deltas, which stop at the screen edge. A count is one step of the mouse's
sensor, about a pixel at Windows' default pointer speed without acceleration, so the feel depends on the mouse's DPI.

With raw input on and a window of this process in the foreground, the cursor is kept inside that window's client
area (`ClipCursor`, re-applied every 100 ms) so that a click cannot land on another window; it is released when focus
moves away or the keyboard pad is turned off. The cursor stays visible: hiding it has to happen on the thread that
owns the game's window, which is the title's, not the shim's.

Keys are read with `GetAsyncKeyState`, which needs no window or message loop (SDL's keyboard is part of its video
subsystem, which this build leaves out) and works under Wine. With `LEONIDA_KEYBOARD_PAD=1` keys count only while a
window of this process is in the foreground, so typing in another application presses nothing, and every button,
stick and trigger is released when focus moves away (the cursor is read afresh when focus returns, so the right
stick does not jump). `LEONIDA_KEYBOARD_PAD=always` reads them whatever has focus; a console tool such as
`gameinput_probe` needs it, because its console window belongs to the terminal, not to the tool.

To try it by hand: `set LEONIDA_KEYBOARD_PAD=always`, then `gameinput_probe --seconds 20` and press keys.

## Rumble and triggers

`IGameInputDevice::SetRumbleState` takes four motor levels, 0..1. The two body motors go to `SDL_RumbleGamepad` on
every pad that has rumble. The two trigger motors depend on the pad:

- Xbox One and Series pads (and anything else SDL reports trigger rumble for): `SDL_RumbleGamepadTriggers`, the same
  impulse triggers as on console.
- DualSense: the trigger level becomes an adaptive-trigger effect sent with `SDL_SendGamepadEffect`, chosen by
  `LEONIDA_DS_TRIGGERS`.
- Others (DualShock 4, Switch Pro): the trigger levels are dropped and the device reports no trigger motors.

**This is an approximation.** The Xbox title only sends an impulse-trigger vibration level per frame. The PS5 version
of a game has effects authored per weapon and vehicle (weapon resistance, a click at a trigger position, and so on);
none of that exists in the Xbox build, so the shim can only turn the level it gets into a generic effect. Expect it to
feel like the Xbox signal played through the DualSense's trigger motors, not like the PS5 release.

The DualSense effect is SDL's 47-byte DualSense effects block (`DS5EffectsState_t` in SDL's `SDL_hidapi_ps5.c`, the
format `test/testcontroller.c` uses): byte 0 holds the valid flags `0x04 | 0x08` (right and left trigger effect),
bytes 10-20 the right trigger effect and bytes 21-31 the left one. The effect modes are the three SDL's test program
uses:

| `LEONIDA_DS_TRIGGERS` | level 0 | level > 0 |
|---|---|---|
| `rumble` (default) | `05` (effect off) | `06 <freq> <level*255> 00`: vibration at `LEONIDA_DS_TRIGGER_FREQ` Hz from the start of the pull |
| `resistance` | `05` | `01 00 <level*255>`: constant resistance over the whole pull |
| `off` | nothing is sent; the triggers stay as they are | |

An effect is sent only when its bytes change, so a steady level costs one report.

## Motion

Pads whose SDL gamepad reports both a gyroscope and an accelerometer (a DualSense, a Switch Pro, Steam Input pads) get
`GameInputKindMotion` in `supportedInput` and a `motionInfo` (`GameInputMotionInfo`, Microsoft's v0 layout); `IGameInputReading::GetMotionState`
returns a `GameInputMotionState` on their readings, and `GetCurrentReading`/`GetNextReading` answer a
`GameInputKindMotion` query for such a pad and return `GAMEINPUT_E_READING_NOT_FOUND` for a pad without sensors. The
sensors are enabled when the pad is opened, at SDL's requested rate, and every poll reads the latest sample.

- The state carries **signed fractions of the maxima `motionInfo` reports** (1.0 = full magnitude), which is the
  convention Microsoft documents for `GameInputMotionState`. SDL delivers accel in m/s² (gravity included) and gyro
  in rad/s, so the shim divides by the maxima. The maxima are this shim's choice — SDL does not expose a sensor
  range — and are 2 g (19.6133 m/s²) and 2000 deg/s (34.9066 rad/s), a DualSense's accelerometer and gyro ranges
  (VERIFY against the game and a real pad).
- A pad at rest sits at about 0.5 on the gravity axis and 0.0 on all gyro axes. SDL answers sensor reads with zeros
  before the device's first sample arrives; since a real accelerometer never reads exactly zero (gravity), all-zero
  accel means "no sample yet" and the reading carries no motion at all — `GetMotionState` returns false.
- SDL's gamepad API has no compass and the shim synthesises no orientation quaternion, so the magnetic-field and
  orientation members are zero with `GameInputMotionAccuracyUnknown`.
- A pad with sensors streams a reading every ~8 ms (~125 Hz) even when nothing else changes, so motion is not stuck
  at the last button event; pads without sensors keep the old change-driven behaviour. The 8 ms window and the 512-
  reading history cap bound the cost.
- `gameinput_probe` prints the accel and gyro fractions when the pad has sensors (a resting pad shows ~0.5 g).

## Environment variables

| variable | default | effect |
|---|---|---|
| `LEONIDA_DS_TRIGGERS` | `rumble` | DualSense trigger effects: `off`, `rumble` or `resistance` (above) |
| `LEONIDA_DS_TRIGGER_GAIN` | `1` | scales the trigger level before it becomes an effect, 0..4 |
| `LEONIDA_DS_TRIGGER_FREQ` | `30` | vibration frequency of the `rumble` mode, 1..255 |
| `LEONIDA_DS_LIGHTBAR` | unset | `RRGGBB` (hex) sets the DualSense/DualShock 4 lightbar when the pad connects; `off` turns it dark; unset keeps SDL's player colour |
| `LEONIDA_RUMBLE` | `1` | `0` turns all rumble and trigger output off |
| `LEONIDA_GAMEINPUT_THREAD` | `1` | `0` stops the background poll thread; only `rt::gameinput::poll()` then reads the pads (tests) |
| `LEONIDA_KEYBOARD_PAD` | `off` | `1` adds the [keyboard pad](#keyboard-pad), read while this process has focus; `always` reads keys whatever has focus |
| `LEONIDA_MOUSE_SENSITIVITY` | `1` | scales how far the mouse moves the keyboard pad's right stick, 0.1..10 |

SDL's own hints (`SDL_JOYSTICK_*`, see SDL's `SDL_hints.h`) also apply, for example `SDL_JOYSTICK_HIDAPI_PS5=0` to
leave a DualSense to the generic driver, or `SDL_GAMECONTROLLERCONFIG` to add a mapping for an unknown pad.

## How it runs

One thread owns SDL: a background thread started by the first `GameInputCreate` that polls every 2 ms. It opens new
pads, closes removed ones, takes a reading of each, and applies rumble requests. Title calls never touch SDL: they read
the latest reading or leave a rumble request, so a call costs a lock and a copy. Callbacks (device, reading, guide
button) run on that thread, as GameInput runs them on its own threads. Once the title creates a dispatcher
(`IGameInput::CreateDispatcher`), callbacks are queued instead, the dispatcher's wait handle is signalled, and
`IGameInputDispatcher::Dispatch` runs them on the title's thread.

The API surface:

| interface | implemented | not implemented |
|---|---|---|
| `IGameInput` | `GetCurrentTimestamp` (µs), `GetCurrentReading`, `GetNextReading`, `GetPreviousReading`, `RegisterReadingCallback`, `RegisterDeviceCallback` (blocking and async enumeration), `RegisterGuideButtonCallback`, `RegisterKeyboardLayoutCallback` (never fires), `StopCallback`, `UnregisterCallback`, `CreateDispatcher`, `FindDeviceFromId`, `EnableOemDeviceSupport` (no-op), `SetFocusPolicy` (no-op) | `GetTemporalReading` (`E_NOTIMPL`, as in GameInput v0 itself; see below), `CreateAggregateDevice`, `FindDeviceFromObject/PlatformHandle/PlatformString` |
| `IGameInputReading` | kind, sequence number, timestamp, device, gamepad state, UI navigation state, controller axes and buttons, motion state | raw report, keyboard, mouse, touch, arcade/flight stick, racing wheel (all return nothing) |
| `IGameInputDevice` | device info, status, battery, `SetRumbleState` | force feedback and haptic motors (none reported), raw device I/O, power off, input synchronization (no-ops) |
| `IGameInputDispatcher` | `Dispatch`, `OpenWaitHandle` | the quota (the queue is short) |

A device object stays alive for the whole process. When a pad is unplugged its status drops to
`GameInputDeviceNoStatus` and its device callback runs; when the same pad is plugged in again (matched by SDL GUID and
serial or device path) the title gets the same object back, connected again.

## Reading history

Each pad keeps the readings the poll took for it, one per change it saw, so `GetNextReading` and `GetPreviousReading`
can step through input the title has not read yet, such as a button pressed and released between two of its frames.
A reading leaves the history once the reading after it is more than 500 ms old, so the current state stays reachable
however long the pad then idles. A pad keeps at most 512 readings, and none once it is unplugged. A reading the title
holds stays valid after it leaves the history.

- With a device the search covers that pad; with none, every connected pad. Readings of different pads are ordered by
  when the shim took them, not by timestamp, since pads polled together share one.
- `GetNextReading` returns the first reading after the reference, `GetPreviousReading` the last one before it.
- The horizon is the newest reading dropped from the histories searched (with no device, the reference's pad counts
  too, even unplugged). A step that would cross it fails rather than skip input.

| situation | result |
|---|---|
| output null; reference null | `E_POINTER`; `E_INVALIDARG` |
| no supported kind in the mask | `GAMEINPUT_E_READING_NOT_FOUND` |
| the device given is unplugged | `GAMEINPUT_E_DEVICE_DISCONNECTED` |
| next: the horizon is after the reference; previous: it is not before it | `GAMEINPUT_E_REFERENCE_READING_TOO_OLD` |
| nothing newer, or nothing older after the horizon | `GAMEINPUT_E_READING_NOT_FOUND` |

After a pad is plugged back in, its last reading before the unplug leads to the reconnect snapshot, older ones are
`REFERENCE_READING_TOO_OLD`, and sequence numbers carry on. Timestamps are poll times, so a change shorter than one poll
(2 ms with the thread) is not seen.

### Device identity

`AppLocalDeviceId` is the SDL GUID in bytes 0-15 and an FNV-1a hash of the serial or path in bytes 16-31. Pads that
SDL cannot tell apart, such as two of the same model with no serial and no path, would share that id, and
`FindDeviceFromId` would return the first for both. They are twins: the first keeps the id, so a single pad's id is
unchanged, and each further one gets an id hashed from its twin number (`t:<n>:` before the serial or path). A new id
is checked against every device seen in the process and moved to the next number if taken, so no two devices share
one. A pad that comes back takes the lowest-numbered disconnected twin's object and id; a connected device is never
reused.

That is assignment by order, not identification. With one twin unplugged, the right object comes back if the same pad
returns, but another identical pad plugged in instead gets it too. With two or more unplugged, nothing tells them
apart: if they return in a different order, the objects and ids are swapped between the physical pads, and the shim
cannot detect it. The same sequence of plugs gives the same ids in every run, since the ids depend only on the GUID,
the serial or path and the order; for pads already connected at start, that order is SDL's (see [VERIFY](#verify)).

## ABI

`runtime/shims/gameinput_abi.h` declares the v0 GameInput API as Leonida's own code: the interfaces in vtable order,
the enums the shim uses and the structs it has to lay out. It was written against Microsoft's public v0 header (the
GDK version, also published as the `Microsoft.GameInput` NuGet package 0.2303.22621.3038). The header is not in the
repository. Its struct offsets were checked by compiling both side by side with `offsetof` asserts for every field the
shim fills (`GameInputDeviceInfo` 320 bytes with `deviceFamily` at 96 and the info pointers from 160,
`GameInputGamepadState` 28 bytes, `GameInputControllerAxisInfo` 48, and the rest in `gameinput_abi.h`).

## Tests

- `gameinput_test` (CTest, no controller): turns SDL's real backends off and drives the shim through
  `rt::shims::find` with SDL virtual joysticks, an Xbox-style pad with trigger motors, a DualSense (Sony
  `054c:0ce6`) and a pad without trigger motors. It checks enumeration and device callbacks, the button, stick,
  trigger and UI-navigation mapping, reading callbacks and sequence numbers, the reading history (a tap read after
  the fact, both directions, two pads interleaved and filtered by device, ageing out, held readings, unplug and
  replug), the guide button, motor and trigger
  rumble as SDL receives them, the DualSense effect bytes in each `LEONIDA_DS_TRIGGERS` mode, the lightbar, a
  dispatcher, async and blocking enumeration, unregistering, unplugging and replugging, and three identical pads
  (an id each, `FindDeviceFromId` and device-filtered readings per pad, replugging one, two and all three, and the
  first twin's id equal to the single-pad id). Motion runs on a virtual pad with sensors (`nsensors` in the
  virtual-joystick description, fed with `SDL_SendJoystickVirtualSensorData`): the kind and `motionInfo` on a sensor
  pad and not on a plain one, no motion before the first sample, normalisation against the `motionInfo` maxima, the
  accuracy values, the ±1 clamp, the ~8 ms cadence, and that a motion query on a pad without sensors is
  `GAMEINPUT_E_READING_NOT_FOUND`. The keyboard pad is
  driven through `rt::gameinput::set_keyboard_source` and `set_cursor_source` with fake keys, focus and cursor: every
  binding, two keys at once, each WASD direction, opposite keys cancelling, diagonals, the mouse moving the right
  stick, clamping and decaying to 0, the mouse-button triggers, no input out of focus, release of buttons, sticks and
  triggers on focus loss, the `always` mode, and turning it off and on again.
- `gameinput_probe [--seconds N] [--rumble]` (by hand, with a pad connected): lists the pads, plays each motor and
  trigger for a second with `--rumble`, then prints the live reading.

## VERIFY

These need the game executable or a test on hardware:

- **Which API version and module.** The shim is GameInput v0 and registers `GameInputCreate` under `gameinput.dll`
  and `xgameruntime.dll`. GameInput v1 and later (NuGet 1.x and up) changed the interfaces; if the executable was
  built against one of those, the vtables need a second set.
- **IIDs.** `QueryInterface` answers the v0 IIDs from the public header.
- **Reading history.** Microsoft's [next reading](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/input/gameinput/interfaces/igameinput/methods/igameinput_getnextreading)
  and [previous reading](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/input/gameinput/interfaces/igameinput/methods/igameinput_getpreviousreading) pages describe half a second of readings per device and failure when old readings leave the stream. The table above records the shim's HRESULT choices. Changes faster than the poll are not seen.
  `GetTemporalReading` returns `E_NOTIMPL` with a null reading on purpose: Microsoft's
  [reference page](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/input/gameinput/deprecated/interfaces/igameinput/methods/igameinput_gettemporalreading)
  says it "was never implemented in v0" and returns `E_NOTIMPL`, and v1 removed it. The v0 header (NuGet
  `Microsoft.GameInput` 0.2303.22621.3038) declares it as `(timestamp, device, reading)`, the layout the shim has;
  the page's five-parameter form does not match that header.
- **UI navigation mapping** and the **controller-axis convention** (0..1 with sticks at 0.5) are this shim's choice,
  not measured from a console.
- **Battery capacity** is reported as a fraction (remaining 0..1 of full 1), not in device units.
- **`UnregisterCallback` timeout** is not honoured; it waits for a callback that is running.
- **Motion normalisation maxima** (2 g accel, 2000 deg/s gyro) are this shim's choice; SDL exposes no sensor range,
  and whether the game reads motion, and against what scale, is unknown until it runs. The accelerometer range a
  DualSense exposes can also be wider than 2 g, so hard shakes may clamp at 1.0.
- **Motion members the shim does not provide** (magnetic field, orientation) are zero with
  `GameInputMotionAccuracyUnknown`; whether a title wants a synthesised quaternion there is unknown.
- **Device family**: every pad is reported as `GameInputFamilyXboxOne`; a title that filters on family sees only
  that.
- **Identical pads at start**: twins are numbered in the order the shim opens them. For pads already connected when
  it starts, that is SDL's enumeration order, not measured on Windows for identical pads; if it varies between runs,
  so does which pad gets which id.
- **Raw mouse registration**: Windows keeps one raw-input target window per device type and process, so if the title
  registers raw mouse input itself (a console title is not expected to), it replaces the shim's registration and the
  right stick falls back to nothing. The shim does not detect that yet.
- **Keyboard pad as a second pad**: it is one more connected gamepad, so with a controller plugged in the title sees
  two. A title that follows only the first pad it saw, or asks the player to press a button to pick the active pad,
  may ignore one of them; whether the game does either is unknown.
- **Mouse on the right stick**: the scale (1024 units per pixel per poll) and decay (7/8 per poll) are a starting
  point, not tuned in game. The title adds its own stick acceleration and deadzone on top, so aiming may feel slow
  near center or jumpy.
- **DualSense effect parameters**: the `06` (vibration) and `01` (resistance) byte meanings follow SDL's
  `testcontroller.c`; how strong 255 feels and what frequency suits the Xbox signal needs a session on a real pad.
