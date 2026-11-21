// Tiny fixture DLL for the rt::Loader tests. It is ordinary test code, not game content.
//
// The imports must stay non-empty: tiny_add calls GetCurrentProcessId and records the result, and the test calls
// tiny_last_pid to prove the IAT entry really points at the host function. The string literal below gives the image
// a .rdata section, and tiny_text_ptr is initialised with its address so the image also contains a DIR64 base
// relocation that the test can verify after forcing a relocated mapping.
#include <windows.h>

namespace {

const char kTinyText[] = "leonida tiny fixture";
DWORD g_last_pid = 0;

}  // namespace

extern "C" __declspec(dllexport) int __cdecl tiny_add(int a, int b) {
    g_last_pid = GetCurrentProcessId();
    return a + b;
}

extern "C" __declspec(dllexport) unsigned long __cdecl tiny_last_pid() { return g_last_pid; }

extern "C" __declspec(dllexport) int tiny_exported_value = 1234;

extern "C" __declspec(dllexport) const char* tiny_text_ptr = kTinyText;
