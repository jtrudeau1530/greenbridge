// Minimal TLS anchor DLL for the native loader.
//
// Loading this image makes the Windows loader assign it a per-module TLS index and allocate a block for that index in
// every thread's TEB. Loader::initialize_tls reuses that reserved index for the manually mapped image, so
// __declspec(thread) access in mapped code reads a valid TEB slot. The anchor itself is never used; the loader saves
// and restores the helper's own block, so the helper's CRT TLS state is not disturbed.
#include <windows.h>

__declspec(thread) unsigned char g_tls_host_anchor = 0;

// Never called. The export keeps the TLS variable (and therefore the image's TLS directory) from being dropped by
// the linker without making DllMain touch TLS during thread attach/detach.
extern "C" __declspec(dllexport) void gta6rt_tls_host_keep() { g_tls_host_anchor = 1; }

BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) { return TRUE; }
