#pragma once

#include <cstddef>

namespace rt::xmem {

bool commit_physical_range(const void* address, std::size_t size);

}
