# Save files

The game's save containers, measured from one save set (two story saves, a profile record and settings) taken from a
retail copy, with every file already plain on disk. `tools/saves.py` reads, rebuilds and imports them using only
the standard library. Nothing here decrypts anything. The one high-entropy layer (`SLAYER1`, below) is described
and left alone.

Confidence: **confirmed** means it holds on every sample or is proven by a byte-identical rebuild. **likely** means
it's consistent with the data. **hypothesis** means it's a guess.

Never commit or attach real saves, extracted blocks or `save.pretty.json`. They're the player's data, and the block
contents are game data.

## Overview

| Container | Holds | Where it's seen |
|---|---|---|
| `JSAV` v3 | The story save: an index plus one JSON document | One file per story slot (manual and autosave) |
| `ASAV` v1 | Generic header + payload | Profile record (kind 3), settings (kind 2) |
| `SLAYER1` | A named wrapper around a packed `ASAV` (kind 1) | Next to each story save: the same save packed, with its slot name and title |

## JSAV: story save (confirmed)

```
0x00  char[4]  "JSAV"
0x04  u32      version            = 3
0x08  u32      n_code_blocks
0x0C  u32      n_script_blocks
0x10  entry[n_code + n_script], 20 bytes each:
        u32 hash      block id, also the block's inner key
        u32 flags     1 on every sample
        u32 reserved  0
        u32 offset    absolute file offset of the block's JSON text
        u32 size      byte length of the block's JSON text
      zero padding to a 16-byte boundary
body  UTF-8 JSON, CRLF between blocks:
        {"CodeSaveBlocks":[\r\n  b0,\r\n b1 ... bN],\r\n"ScriptSaveBlocks":[\r\n  ... bM]\r\n}
      zero padding to a 16-byte boundary
```

Both samples have 38 code and 44 script blocks. `tools/saves.py verify` rebuilds them byte for byte from the parsed
blocks, which confirms the framing, padding and offsets. The JSAV carries no checksum of its own.

Block conventions:
- Each block is `{"$0x<group>":{"$0x<hash>":{...}}}`, with one group hash for code blocks and one for script blocks. The inner hash equals the index entry's hash. **confirmed**
- Keys are 32-bit hashes written as `"$0x%08x"`. They aren't the lowercase joaat of obvious names; the hash function and seed are a **hypothesis** to check against the executable.
- Reserved, unhashed keys **(confirmed)**:
  - `$x $y $z $w` vectors and quaternions
  - `$content` + `$data` typed arrays (`int_array`, `int64_array`, `char_array`, `float_array`, `short_array`)
  - `$data:"0x…"` a hash value stored as a string
  - `$key`/`$value` map entries
  - `$type`, `$bits`
  - `$v` version (settings)
- Floats are printed `%f` (six decimals) and ints in decimal. A JSON library re-serialises them differently, so `pack` copies block text verbatim. **confirmed**
- Diffing an autosave against a manual save taken about a minute later shows 62 of 82 blocks identical. The other 20 are positions, counters and stats, so blocks are independent units. **confirmed**

## ASAV: header + payload

```
0x000 char[4]  "ASAV"
0x004 u32      version = 1
0x008 u32      kind: 1 story (packed), 2 settings, 3 profile               likely
0x00C u32      1 plain, 3 packed                                           hypothesis: storage mode
0x010 u32      0 plain, 0x180 packed                                       hypothesis
0x014 u32      payload offset = 0x170                                      confirmed
0x018 u32      0 plain, stored size packed                                 likely
0x01C u32      file size (plain), unpacked size (packed)                   confirmed plain / likely packed
0x020 u64      unix time, seconds, UTC                                     confirmed
0x028 u16[128] display title, UTF-16LE                                     confirmed
0x128 char[56] slot name, ASCII                                            confirmed
0x160 u32      checksum: algorithm unknown
0x164 u32      0
0x168 u64      file size again (plain)
0x170          payload
```

The checksum at 0x160 matched none of these:
- CRC-32, CRC-32C, Adler-32, FNV-1/1a, joaat (seeds 0, `0x3fac7125`, `0xffffffff`), byte sum, MD5, SHA-1 or SHA-256 (truncated to 32 bits)
- tried over every 4-aligned sub-range, with the field zeroed and raw, in both byte orders

It's left as **open**. Copy the header verbatim whenever the payload is unchanged.

Payloads: settings (kind 2) are hash-keyed JSON text with tabs and CRLF **(confirmed)**. The profile record (kind 3)
is 1,644 bytes, only the first 112 non-zero, starting `u32 10, u32 3`. It isn't decoded yet.

## SLAYER1: packed copy (not decoded)

```
0x00  char[8]  "SLAYER1\0"
0x08  u32      version = 1
0x0C  u32      name length n
0x10  char[n]  name, no terminator (so the rest may be unaligned)
...   u64      payload size
...   ASAV, kind 1: header as above (title, slot name, time), then 16 high-entropy bytes at +0x170
      and a high-entropy body at +0x180 (~30% of the JSAV's size)
```

The body is compressed and/or encrypted. Following AGENTS.md, it's recorded here and not decoded. The plain JSAV next
to it holds the same save. The wrapper's header is the only place the autosave's slot name and title appear, so
`saves.py` reads it for those.

## Importing into the runtime

`runtime/shims/xgameruntime_services.cpp` (#279) keeps `XGameSave` data in
`LEONIDA_SAVE_DIR\local-user\<container>\<blob>`, defaulting to `%LOCALAPPDATA%\Leonida\saves`, with names
%-encoded. `python -I tools/saves.py import <save folder>... [--out DIR] [--map OLD=CONTAINER/BLOB]` writes payloads
in that layout with the same encoding, plus `import-manifest.json` (source, size, SHA-256).

Checked: four imported blobs (2 KB to 1.2 MB) read back byte-identical through the shim's `XGameSaveReadBlobData`.

**VERIFY:** the title's real container and blob names. The default is container = slot name and blob = payload file
name, and `--map` overrides it. Confirm or fix them from the executable's `XGameSaveCreateContainer` and
`XGameSaveSubmitBlobWrite` calls.

## Tool

```powershell
python -I tools/saves.py info    <file or folder>...
python -I tools/saves.py verify  <JSAV>                 # parse every block, rebuild, compare byte for byte
python -I tools/saves.py extract <JSAV|ASAV> <outdir>   # blocks/*.json, index.json, save.pretty.json
python -I tools/saves.py pack    <outdir> <JSAV>        # after editing blocks/*.json
python -I tools/saves.py diff    <JSAV a> <JSAV b>
python -I tools/saves.py import  <save folder>... [--out DIR] [--map OLD=CONTAINER/BLOB] [--dry-run] [--force]
```

`tests/test_saves.py` covers all of it on synthetic containers built in the test.
