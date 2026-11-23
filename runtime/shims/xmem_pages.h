// The xmem shim's physical pages, for the other shims that are handed page numbers instead of addresses (console
// DirectStorage: DSTORAGE_REQUEST_OPTIONS::DestinationIsPhysicalPages).
#pragma once

#include <cstddef>
#include <cstdint>

namespace rt::xmem {

// The size of one page XMemAllocatePhysicalPages hands out (shims/xmem.cpp).
constexpr size_t kPhysicalPageBytes = 0x10000;

// Copies `size` bytes from `source` into the physical pages page_array names: the data starts `offset` bytes into
// page_array[0] and runs on through the entries after it, (offset + size) rounded up to whole pages of them. The
// pages are written through their backing, so it makes no difference whether a page is mapped, mapped at several
// addresses or not mapped at all, and a mapped view sees the bytes.
//
// Every page number is checked before the first byte is written. false, with nothing written: a null argument, an
// offset of a page or more, or an entry that is not a live page of XMemAllocatePhysicalPages (0 included; in the map
// calls 0 means "unmap", which is not a place to write to). `page_array` and `source` must be readable: a caller that
// takes them from a title copies them first.
bool write_physical_pages(const uint64_t* page_array, size_t offset, const void* source, size_t size);

}  // namespace rt::xmem
