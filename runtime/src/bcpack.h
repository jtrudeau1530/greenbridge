#pragma once

// Native BCPack decoder: a dependency-free C++ port of tools/bcpack.py (RT-186).
//
// Decodes one Xbox Series DirectStorage BCPack chunk (the interleaved 12/16-lane rANS entropy layer, then the
// BC1/BC3/BC4/BC5/BC6H/BC7 block rebuilders) into standard BCn blocks. It gives the same bytes, the same refusals
// and the same provisional labels as `bcpack.decode_with_info(raw, codec, out_size, allow_provisional=...)` with
// `history=None`: the built-in BC1/BC3/BC4/BC5 history tables of tools/bcpack_tables.py, and no BC6H/BC7 initial
// history, so BC6H/BC7 chunks decode only when their delta chains stay inside the chunk. The tables come from
// runtime/src/bcpack_tables.inc, which tools/gen_bcpack_tables.py generates from the Python tables;
// tests/test_bcpack_native.py compares the two decoders on synthetic chunks. docs/textures.md has the format.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace leonida::bcpack {

// DSTORAGE_BCPACK_MODE: 1 = BC1, 3 = BC3, 4 = BC4, 5 = BC5, 6 = BC6H, 7 = BC7.
enum class Status : std::uint8_t {
    ok = 0,
    error = 1,        // bcpack.BcpackError: the chunk is malformed
    provisional = 2,  // bcpack.BcpackProvisionalError: needs an unconfirmed or unknown built-in history entry
    unsupported = 3,  // bcpack.BcpackUnsupportedCodec: no block rebuilder for this codec id
};

struct Result {
    Status status = Status::ok;
    std::string message;                   // why the chunk was refused (empty when it decoded)
    std::vector<std::string> provisional;  // with allow_provisional: what the output relies on, sorted
    explicit operator bool() const { return status == Status::ok; }
};

// Bytes per BCn block of a codec id, 0 for an unknown one.
std::size_t block_bytes(int codec);

// Decodes `raw` into `out`, whose size is the chunk's decoded size (a whole number of blocks). `out` is only
// meaningful when the result is ok. Strict by default: any reference to an unconfirmed or unknown built-in history
// entry is refused; `allow_provisional` accepts them and lists them in `Result::provisional`.
Result decode(std::span<const std::uint8_t> raw, int codec, std::span<std::uint8_t> out,
              bool allow_provisional = false);

// The entropy layer alone (bcpack.decode_lanes): one symbol list per lane, and the input bytes it consumed.
Result decode_lanes(std::span<const std::uint8_t> raw, int codec, std::size_t out_size,
                    std::vector<std::vector<std::uint16_t>>& lanes, std::size_t* bytes_used = nullptr);

}  // namespace leonida::bcpack
