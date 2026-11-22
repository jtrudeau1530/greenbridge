// GameInput v0 ABI: the part of Microsoft's GameInput API a title uses for gamepads.
//
// These are Leonida's own declarations, written to match the binary layout of the public v0 GameInput API (the
// in-box GDK version, also shipped as the Microsoft.GameInput 0.2303.x NuGet package). Only what the shim implements
// or must lay out is declared; interface methods are in vtable order and every struct is checked with static_asserts
// below. Names follow the public API so the docs at learn.microsoft.com/gaming/gdk/docs/reference/input/gameinput
// apply directly.
//
// VERIFY: GameInput v1 and later (the 1.x-4.x NuGet packages) changed the interfaces. Which version the console
// executable was built against is unknown until it is available; this file is v0.
#pragma once
#include <windows.h>
#include <unknwn.h>

#include <cstddef>
#include <cstdint>

namespace gi {

enum GameInputKind : uint32_t {
    GameInputKindUnknown = 0x00000000,
    GameInputKindRawDeviceReport = 0x00000001,
    GameInputKindControllerAxis = 0x00000002,
    GameInputKindControllerButton = 0x00000004,
    GameInputKindControllerSwitch = 0x00000008,
    GameInputKindController = 0x0000000E,
    GameInputKindKeyboard = 0x00000010,
    GameInputKindMouse = 0x00000020,
    GameInputKindTouch = 0x00000100,
    GameInputKindMotion = 0x00001000,
    GameInputKindArcadeStick = 0x00010000,
    GameInputKindFlightStick = 0x00020000,
    GameInputKindGamepad = 0x00040000,
    GameInputKindRacingWheel = 0x00080000,
    GameInputKindUiNavigation = 0x01000000,
};

enum GameInputEnumerationKind : int32_t {
    GameInputNoEnumeration = 0,
    GameInputAsyncEnumeration = 1,
    GameInputBlockingEnumeration = 2,
};

enum GameInputGamepadButtons : uint32_t {
    GameInputGamepadNone = 0x00000000,
    GameInputGamepadMenu = 0x00000001,
    GameInputGamepadView = 0x00000002,
    GameInputGamepadA = 0x00000004,
    GameInputGamepadB = 0x00000008,
    GameInputGamepadX = 0x00000010,
    GameInputGamepadY = 0x00000020,
    GameInputGamepadDPadUp = 0x00000040,
    GameInputGamepadDPadDown = 0x00000080,
    GameInputGamepadDPadLeft = 0x00000100,
    GameInputGamepadDPadRight = 0x00000200,
    GameInputGamepadLeftShoulder = 0x00000400,
    GameInputGamepadRightShoulder = 0x00000800,
    GameInputGamepadLeftThumbstick = 0x00001000,
    GameInputGamepadRightThumbstick = 0x00002000,
};

enum GameInputUiNavigationButtons : uint32_t {
    GameInputUiNavigationNone = 0x00000000,
    GameInputUiNavigationMenu = 0x00000001,
    GameInputUiNavigationView = 0x00000002,
    GameInputUiNavigationAccept = 0x00000004,
    GameInputUiNavigationCancel = 0x00000008,
    GameInputUiNavigationUp = 0x00000010,
    GameInputUiNavigationDown = 0x00000020,
    GameInputUiNavigationLeft = 0x00000040,
    GameInputUiNavigationRight = 0x00000080,
    GameInputUiNavigationContext1 = 0x00000100,
    GameInputUiNavigationContext2 = 0x00000200,
    GameInputUiNavigationContext3 = 0x00000400,
    GameInputUiNavigationContext4 = 0x00000800,
    GameInputUiNavigationPageUp = 0x00001000,
    GameInputUiNavigationPageDown = 0x00002000,
    GameInputUiNavigationPageLeft = 0x00004000,
    GameInputUiNavigationPageRight = 0x00008000,
    GameInputUiNavigationScrollUp = 0x00010000,
    GameInputUiNavigationScrollDown = 0x00020000,
    GameInputUiNavigationScrollLeft = 0x00040000,
    GameInputUiNavigationScrollRight = 0x00080000,
};

enum GameInputDeviceStatus : uint32_t {
    GameInputDeviceNoStatus = 0x00000000,
    GameInputDeviceConnected = 0x00000001,
    GameInputDeviceInputEnabled = 0x00000002,
    GameInputDeviceOutputEnabled = 0x00000004,
    GameInputDeviceRawIoEnabled = 0x00000008,
    GameInputDeviceAudioCapture = 0x00000010,
    GameInputDeviceAudioRender = 0x00000020,
    GameInputDeviceSynchronized = 0x00000040,
    GameInputDeviceWireless = 0x00000080,
    GameInputDeviceUserIdle = 0x00100000,
    GameInputDeviceAnyStatus = 0x00FFFFFF,
};

enum GameInputBatteryStatus : int32_t {
    GameInputBatteryUnknown = -1,
    GameInputBatteryNotPresent = 0,
    GameInputBatteryDischarging = 1,
    GameInputBatteryIdle = 2,
    GameInputBatteryCharging = 3,
};

enum GameInputDeviceFamily : int32_t {
    GameInputFamilyVirtual = -1,
    GameInputFamilyAggregate = 0,
    GameInputFamilyXboxOne = 1,
    GameInputFamilyXbox360 = 2,
    GameInputFamilyHid = 3,
    GameInputFamilyI8042 = 4,
};

enum GameInputDeviceCapabilities : uint32_t {
    GameInputDeviceCapabilityNone = 0x00000000,
    GameInputDeviceCapabilityAudio = 0x00000001,
    GameInputDeviceCapabilityPluginModule = 0x00000002,
    GameInputDeviceCapabilityPowerOff = 0x00000004,
    GameInputDeviceCapabilitySynchronization = 0x00000008,
    GameInputDeviceCapabilityWireless = 0x00000010,
};

enum GameInputRumbleMotors : uint32_t {
    GameInputRumbleNone = 0x00000000,
    GameInputRumbleLowFrequency = 0x00000001,
    GameInputRumbleHighFrequency = 0x00000002,
    GameInputRumbleLeftTrigger = 0x00000004,
    GameInputRumbleRightTrigger = 0x00000008,
};

enum GameInputSwitchPosition : int32_t { GameInputSwitchCenter = 0 };
enum GameInputFocusPolicy : uint32_t { GameInputDefaultFocusPolicy = 0 };

using GameInputCallbackToken = uint64_t;
constexpr GameInputCallbackToken kInvalidCallbackToken = 0;

// Error codes (facility 0x38A).
constexpr HRESULT GAMEINPUT_E_DEVICE_DISCONNECTED = static_cast<HRESULT>(0x838A0001L);
constexpr HRESULT GAMEINPUT_E_DEVICE_NOT_FOUND = static_cast<HRESULT>(0x838A0002L);
constexpr HRESULT GAMEINPUT_E_READING_NOT_FOUND = static_cast<HRESULT>(0x838A0003L);
constexpr HRESULT GAMEINPUT_E_REFERENCE_READING_TOO_OLD = static_cast<HRESULT>(0x838A0004L);
constexpr HRESULT GAMEINPUT_E_TIMESTAMP_OUT_OF_RANGE = static_cast<HRESULT>(0x838A0005L);
constexpr HRESULT GAMEINPUT_E_INSUFFICIENT_FORCE_FEEDBACK_RESOURCES = static_cast<HRESULT>(0x838A0006L);

struct AppLocalDeviceId {  // APP_LOCAL_DEVICE_ID
    uint8_t value[32];
};

struct GameInputGamepadState {
    uint32_t buttons;  // GameInputGamepadButtons
    float leftTrigger;
    float rightTrigger;
    float leftThumbstickX;
    float leftThumbstickY;
    float rightThumbstickX;
    float rightThumbstickY;
};

struct GameInputUiNavigationState {
    uint32_t buttons;  // GameInputUiNavigationButtons
};

struct GameInputBatteryState {
    float chargeRate;
    float maxChargeRate;
    float remainingCapacity;
    float fullChargeCapacity;
    GameInputBatteryStatus status;
};

struct GameInputRumbleParams {
    float lowFrequency;
    float highFrequency;
    float leftTrigger;
    float rightTrigger;
};

struct GameInputString {
    uint32_t sizeInBytes;
    uint32_t codePointCount;
    const char* data;  // UTF-8, zero-terminated
};

struct GameInputUsage {
    uint16_t page;
    uint16_t id;
};

struct GameInputVersion {
    uint16_t major, minor, build, revision;
};

struct GameInputRawDeviceReportInfo;
struct GameInputRawDeviceReportItemInfo;

struct GameInputControllerAxisInfo {
    uint32_t mappedInputKinds;  // GameInputKind
    bool isContinuous;
    bool isNonlinear;
    bool isQuantized;
    bool hasRestValue;
    float restValue;
    uint64_t resolution;
    uint16_t legacyDInputIndex;
    uint16_t legacyHidIndex;
    uint32_t rawReportIndex;
    const GameInputRawDeviceReportInfo* inputReport;
    const GameInputRawDeviceReportItemInfo* inputReportItem;
};

struct GameInputControllerButtonInfo {
    uint32_t mappedInputKinds;  // GameInputKind
    uint16_t legacyDInputIndex;
    uint16_t legacyHidIndex;
    uint32_t rawReportIndex;
    const GameInputRawDeviceReportInfo* inputReport;
    const GameInputRawDeviceReportItemInfo* inputReportItem;
};

// Empty in v0 (one byte in C++); a title only checks the pointer.
struct GameInputGamepadInfo {};
struct GameInputUiNavigationInfo {};

struct GameInputDeviceInfo {
    uint32_t infoSize;
    uint16_t vendorId;
    uint16_t productId;
    uint16_t revisionNumber;
    uint8_t interfaceNumber;
    uint8_t collectionNumber;
    GameInputUsage usage;
    GameInputVersion hardwareVersion;
    GameInputVersion firmwareVersion;
    AppLocalDeviceId deviceId;
    AppLocalDeviceId deviceRootId;
    GameInputDeviceFamily deviceFamily;
    uint32_t capabilities;           // GameInputDeviceCapabilities
    uint32_t supportedInput;         // GameInputKind
    uint32_t supportedRumbleMotors;  // GameInputRumbleMotors
    uint32_t inputReportCount;
    uint32_t outputReportCount;
    uint32_t featureReportCount;
    uint32_t controllerAxisCount;
    uint32_t controllerButtonCount;
    uint32_t controllerSwitchCount;
    uint32_t touchPointCount;
    uint32_t touchSensorCount;
    uint32_t forceFeedbackMotorCount;
    uint32_t hapticFeedbackMotorCount;
    uint32_t deviceStringCount;
    uint32_t deviceDescriptorSize;
    const GameInputRawDeviceReportInfo* inputReportInfo;
    const GameInputRawDeviceReportInfo* outputReportInfo;
    const GameInputRawDeviceReportInfo* featureReportInfo;
    const GameInputControllerAxisInfo* controllerAxisInfo;
    const GameInputControllerButtonInfo* controllerButtonInfo;
    const void* controllerSwitchInfo;
    const void* keyboardInfo;
    const void* mouseInfo;
    const void* touchSensorInfo;
    const void* motionInfo;
    const void* arcadeStickInfo;
    const void* flightStickInfo;
    const GameInputGamepadInfo* gamepadInfo;
    const void* racingWheelInfo;
    const GameInputUiNavigationInfo* uiNavigationInfo;
    const void* forceFeedbackMotorInfo;
    const void* hapticFeedbackMotorInfo;
    const GameInputString* displayName;
    const GameInputString* deviceStrings;
    const void* deviceDescriptorData;
};

// Layout checks. The offsets were compared with the public v0 header by compiling both side by side (see
// docs/gameinput.md); they hold for x64 MSVC.
static_assert(sizeof(GameInputGamepadState) == 28);
static_assert(sizeof(GameInputRumbleParams) == 16);
static_assert(sizeof(GameInputBatteryState) == 20);
static_assert(sizeof(GameInputString) == 16);
static_assert(sizeof(GameInputControllerAxisInfo) == 48);
static_assert(sizeof(GameInputControllerButtonInfo) == 32);
static_assert(offsetof(GameInputDeviceInfo, deviceId) == 32);
static_assert(offsetof(GameInputDeviceInfo, deviceFamily) == 96);
static_assert(offsetof(GameInputDeviceInfo, inputReportCount) == 112);
static_assert(offsetof(GameInputDeviceInfo, inputReportInfo) == 160);
static_assert(offsetof(GameInputDeviceInfo, gamepadInfo) == 256);
static_assert(offsetof(GameInputDeviceInfo, displayName) == 296);
static_assert(sizeof(GameInputDeviceInfo) == 320);

struct IGameInputDevice;
struct IGameInputReading;
struct IGameInputDispatcher;

using GameInputReadingCallback = void(CALLBACK*)(GameInputCallbackToken, void* context, IGameInputReading* reading,
                                                 bool hasOverrunOccurred);
using GameInputDeviceCallback = void(CALLBACK*)(GameInputCallbackToken, void* context, IGameInputDevice* device,
                                                uint64_t timestamp, uint32_t currentStatus, uint32_t previousStatus);
using GameInputGuideButtonCallback = void(CALLBACK*)(GameInputCallbackToken, void* context, IGameInputDevice* device,
                                                     uint64_t timestamp, bool isPressed);
using GameInputKeyboardLayoutCallback = void(CALLBACK*)(GameInputCallbackToken, void* context,
                                                        IGameInputDevice* device, uint64_t timestamp,
                                                        uint32_t currentLayout, uint32_t previousLayout);

// {11BE2A7E-4254-445A-9C09-FFC40F006918}
constexpr GUID IID_IGameInput = {0x11BE2A7E, 0x4254, 0x445A, {0x9C, 0x09, 0xFF, 0xC4, 0x0F, 0x00, 0x69, 0x18}};
// {2156947A-E1FA-4DE0-A30B-D812931DBD8D}
constexpr GUID IID_IGameInputReading = {0x2156947A, 0xE1FA, 0x4DE0, {0xA3, 0x0B, 0xD8, 0x12, 0x93, 0x1D, 0xBD, 0x8D}};
// {31DD86FB-4C1B-408A-868F-439B3CD47125}
constexpr GUID IID_IGameInputDevice = {0x31DD86FB, 0x4C1B, 0x408A, {0x86, 0x8F, 0x43, 0x9B, 0x3C, 0xD4, 0x71, 0x25}};
// {415EED2E-98CB-42C2-8F28-B94601074E31}
constexpr GUID IID_IGameInputDispatcher = {0x415EED2E, 0x98CB, 0x42C2, {0x8F, 0x28, 0xB9, 0x46, 0x01, 0x07, 0x4E, 0x31}};

// Interfaces, methods in vtable order. Pointer types the shim never implements are void*.
struct IGameInput : IUnknown {
    virtual uint64_t STDMETHODCALLTYPE GetCurrentTimestamp() = 0;
    virtual HRESULT STDMETHODCALLTYPE GetCurrentReading(uint32_t inputKind, IGameInputDevice* device,
                                                        IGameInputReading** reading) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetNextReading(IGameInputReading* referenceReading, uint32_t inputKind,
                                                     IGameInputDevice* device, IGameInputReading** reading) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetPreviousReading(IGameInputReading* referenceReading, uint32_t inputKind,
                                                         IGameInputDevice* device, IGameInputReading** reading) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetTemporalReading(uint64_t timestamp, IGameInputDevice* device,
                                                         IGameInputReading** reading) = 0;
    virtual HRESULT STDMETHODCALLTYPE RegisterReadingCallback(IGameInputDevice* device, uint32_t inputKind,
                                                              float analogThreshold, void* context,
                                                              GameInputReadingCallback callbackFunc,
                                                              GameInputCallbackToken* callbackToken) = 0;
    virtual HRESULT STDMETHODCALLTYPE RegisterDeviceCallback(IGameInputDevice* device, uint32_t inputKind,
                                                             uint32_t statusFilter, GameInputEnumerationKind enumerationKind,
                                                             void* context, GameInputDeviceCallback callbackFunc,
                                                             GameInputCallbackToken* callbackToken) = 0;
    virtual HRESULT STDMETHODCALLTYPE RegisterGuideButtonCallback(IGameInputDevice* device, void* context,
                                                                  GameInputGuideButtonCallback callbackFunc,
                                                                  GameInputCallbackToken* callbackToken) = 0;
    virtual HRESULT STDMETHODCALLTYPE RegisterKeyboardLayoutCallback(IGameInputDevice* device, void* context,
                                                                     GameInputKeyboardLayoutCallback callbackFunc,
                                                                     GameInputCallbackToken* callbackToken) = 0;
    virtual void STDMETHODCALLTYPE StopCallback(GameInputCallbackToken callbackToken) = 0;
    virtual bool STDMETHODCALLTYPE UnregisterCallback(GameInputCallbackToken callbackToken,
                                                      uint64_t timeoutInMicroseconds) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateDispatcher(IGameInputDispatcher** dispatcher) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateAggregateDevice(uint32_t inputKind, IGameInputDevice** device) = 0;
    virtual HRESULT STDMETHODCALLTYPE FindDeviceFromId(const AppLocalDeviceId* value, IGameInputDevice** device) = 0;
    virtual HRESULT STDMETHODCALLTYPE FindDeviceFromObject(IUnknown* value, IGameInputDevice** device) = 0;
    virtual HRESULT STDMETHODCALLTYPE FindDeviceFromPlatformHandle(HANDLE value, IGameInputDevice** device) = 0;
    virtual HRESULT STDMETHODCALLTYPE FindDeviceFromPlatformString(LPCWSTR value, IGameInputDevice** device) = 0;
    virtual HRESULT STDMETHODCALLTYPE EnableOemDeviceSupport(uint16_t vendorId, uint16_t productId,
                                                             uint8_t interfaceNumber, uint8_t collectionNumber) = 0;
    virtual void STDMETHODCALLTYPE SetFocusPolicy(uint32_t policy) = 0;
};

struct IGameInputReading : IUnknown {
    virtual uint32_t STDMETHODCALLTYPE GetInputKind() = 0;
    virtual uint64_t STDMETHODCALLTYPE GetSequenceNumber(uint32_t inputKind) = 0;
    virtual uint64_t STDMETHODCALLTYPE GetTimestamp() = 0;
    virtual void STDMETHODCALLTYPE GetDevice(IGameInputDevice** device) = 0;
    virtual bool STDMETHODCALLTYPE GetRawReport(void** report) = 0;
    virtual uint32_t STDMETHODCALLTYPE GetControllerAxisCount() = 0;
    virtual uint32_t STDMETHODCALLTYPE GetControllerAxisState(uint32_t stateArrayCount, float* stateArray) = 0;
    virtual uint32_t STDMETHODCALLTYPE GetControllerButtonCount() = 0;
    virtual uint32_t STDMETHODCALLTYPE GetControllerButtonState(uint32_t stateArrayCount, bool* stateArray) = 0;
    virtual uint32_t STDMETHODCALLTYPE GetControllerSwitchCount() = 0;
    virtual uint32_t STDMETHODCALLTYPE GetControllerSwitchState(uint32_t stateArrayCount,
                                                                GameInputSwitchPosition* stateArray) = 0;
    virtual uint32_t STDMETHODCALLTYPE GetKeyCount() = 0;
    virtual uint32_t STDMETHODCALLTYPE GetKeyState(uint32_t stateArrayCount, void* stateArray) = 0;
    virtual bool STDMETHODCALLTYPE GetMouseState(void* state) = 0;
    virtual uint32_t STDMETHODCALLTYPE GetTouchCount() = 0;
    virtual uint32_t STDMETHODCALLTYPE GetTouchState(uint32_t stateArrayCount, void* stateArray) = 0;
    virtual bool STDMETHODCALLTYPE GetMotionState(void* state) = 0;
    virtual bool STDMETHODCALLTYPE GetArcadeStickState(void* state) = 0;
    virtual bool STDMETHODCALLTYPE GetFlightStickState(void* state) = 0;
    virtual bool STDMETHODCALLTYPE GetGamepadState(GameInputGamepadState* state) = 0;
    virtual bool STDMETHODCALLTYPE GetRacingWheelState(void* state) = 0;
    virtual bool STDMETHODCALLTYPE GetUiNavigationState(GameInputUiNavigationState* state) = 0;
};

struct IGameInputDevice : IUnknown {
    virtual const GameInputDeviceInfo* STDMETHODCALLTYPE GetDeviceInfo() = 0;
    virtual uint32_t STDMETHODCALLTYPE GetDeviceStatus() = 0;
    virtual void STDMETHODCALLTYPE GetBatteryState(GameInputBatteryState* state) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateForceFeedbackEffect(uint32_t motorIndex, const void* params,
                                                                void** effect) = 0;
    virtual bool STDMETHODCALLTYPE IsForceFeedbackMotorPoweredOn(uint32_t motorIndex) = 0;
    virtual void STDMETHODCALLTYPE SetForceFeedbackMotorGain(uint32_t motorIndex, float masterGain) = 0;
    virtual HRESULT STDMETHODCALLTYPE SetHapticMotorState(uint32_t motorIndex, const void* params) = 0;
    virtual void STDMETHODCALLTYPE SetRumbleState(const GameInputRumbleParams* params) = 0;
    virtual void STDMETHODCALLTYPE SetInputSynchronizationState(bool enabled) = 0;
    virtual void STDMETHODCALLTYPE SendInputSynchronizationHint() = 0;
    virtual void STDMETHODCALLTYPE PowerOff() = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateRawDeviceReport(uint32_t reportId, int32_t reportKind, void** report) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetRawDeviceFeature(uint32_t reportId, void** report) = 0;
    virtual HRESULT STDMETHODCALLTYPE SetRawDeviceFeature(void* report) = 0;
    virtual HRESULT STDMETHODCALLTYPE SendRawDeviceOutput(void* report) = 0;
    virtual HRESULT STDMETHODCALLTYPE SendRawDeviceOutputWithResponse(void* requestReport, void** responseReport) = 0;
    virtual HRESULT STDMETHODCALLTYPE ExecuteRawDeviceIoControl(uint32_t controlCode, size_t inputBufferSize,
                                                                const void* inputBuffer, size_t outputBufferSize,
                                                                void* outputBuffer, size_t* outputSize) = 0;
    virtual bool STDMETHODCALLTYPE AcquireExclusiveRawDeviceAccess(uint64_t timeoutInMicroseconds) = 0;
    virtual void STDMETHODCALLTYPE ReleaseExclusiveRawDeviceAccess() = 0;
};

struct IGameInputDispatcher : IUnknown {
    virtual bool STDMETHODCALLTYPE Dispatch(uint64_t quotaInMicroseconds) = 0;
    virtual HRESULT STDMETHODCALLTYPE OpenWaitHandle(HANDLE* waitHandle) = 0;
};

using GameInputCreateFn = HRESULT(WINAPI*)(IGameInput** gameInput);

}  // namespace gi
