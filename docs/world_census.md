# Census of `levels.rpf`

`levels.rpf` is 76,155,065,056 bytes. Its table of contents is encrypted, so the archive cannot be
enumerated by name. `tools/world_census.py` measures the data area directly: it inflates every zlib
stream, validates each `0DRP` / version `0x7a` resource, joins continuation streams, and counts the
map resources and placement entities they contain.

## Result

| Measure | Value |
|---|---:|
| Data area scanned | 76,154,954,352 bytes (start `0x1B070`) |
| Map resources (`kind=map`) | **23,535** |
| Placement entities | **1,670,442** |
| Complete / validated map rows | **23,535 / 23,535** |
| Exact entity counts | **1,670,442 / 1,670,442** |
| Shards complete | **36 / 36** |
| Zlib streams inflated | **1,122,614** |
| Stored stream bytes | 75,165,813,806 |
| Inflated bytes (all streams) | 175,587,890,693 |
| Rejected zlib candidates | 906 |
| Adopted continuation streams | 174 |
| Metadata streams | 165,880 |
| Duplicate content groups | 0 |
| Scan time | 290.8 s |

A map row is a resource whose schema includes `CMapData`. `maps_exact` means every entity block
counted against the declared `entity_size` and no entity block was left in a graphics page.
`entities_exact` is the same check summed over all rows. Six entities of the 23,535 maps were split
across a continuation; the joiner adopted 174 streams in total.

### Placement maps vs metadata

The 23,535 rows above are **placement maps**: a `CMapData` root and at least one countable
`CEntityDef` block. The same pass also validates **138,839 `CMapData`-only metadata resources** with
no `CEntityDef` block, plus **26,075 other `0x7a` metadata resources**. Counting every `CMapData`
root regardless of entity schema gives **162,374** resources, but only the 23,535 with entities are
placement maps in this census. Keeping the two classes separate avoids inflating the map count with
map-definition metadata.

| Detail | Count |
|---|---:|
| Placement maps (`CMapData` + `CEntityDef`) | 23,535 |
| `CMapData`-only metadata (no `CEntityDef`) | 138,839 |
| Other `0x7a` metadata | 26,075 |
| Placement maps joining a continuation | 2,868 |
| Placement maps using a raw continuation page | 3 |
| Metadata resources using a raw continuation page | 1 |
| Byte-identical duplicate map payloads | 0 |
| Duplicate 32-bit map-name hashes | 0 |
| Entity blocks in graphics pages | 0 |
| Unresolved or truncated map resources | 0 |


## Coverage

| Byte class | Bytes | Share of data area |
|---|---:|---:|
| Accepted stream payload | 74,877,281,159 | 98.32% |
| Raw graphics pages (`0x6`) | 2,762,232 | 0.004% |
| Raw system continuation | 44,304 | 0.0001% |
| Zero padding | 1,058,806,784 | 1.39% |
| Unread non-zero bytes | 216,059,873 | 0.28% |
| **Accounted** | **76,154,954,352** | **100%** |

The unread bytes are mostly small alignment and metadata gaps between streams. The gap audit found
**23** `0DRP` signatures in unread ranges, but none resolved to a map or metadata resource
(`raw_maps = 0`, `raw_meta = 0`), so the census is not missing a map class. No unaligned zlib stream
that inflates to a `0DRP` header was found in a gap (`missed_zlib_meta = 0`). The 906 rejected
candidates are zlib streams that do not form a complete `0x7a` resource; they are ordinary archive
payload, not counted maps.

## Exactness condition

`summary.json` sets `census_exact` only when all of these hold:

- `complete` is true and every expected shard was written;
- `maps_exact == maps` and `entities_exact == entities`;
- `entity_gfx == 0` and `entity_unresolved == 0`;
- `incomplete == 0` and `invalid == 0`;
- `raw_maps == 0` and `missed_zlib_meta == 0`.

The run above satisfies every condition. `accounted_ratio` is 1.0 because every byte is assigned to
accepted payload, a raw page, zero padding or counted unread data.

## Method

The scan starts at the data-area offset from the `9FPR` header. It walks 16-byte-aligned zlib
candidates in independent shards and:

1. inflates each stream, keeping the first bytes needed to recognize `0DRPz`;
2. validates the little-endian `0x7a` header, struct table, block table and member offsets;
3. follows a contiguous split resource, either another zlib page or a raw system page whose length
   the resource's pointers require;
4. counts `CEntityDef` blocks against the declared entity size, and records `0x6` graphics extents;
5. merges shard rows, checks duplicate names and content hashes, and audits every gap for a missed
   `0DRP` signature or unaligned zlib stream.

Each shard is written to `shard_<index>_<start>_<end>.json`; `summary.json` and `maps.json` are
rebuilt from the shards, so an interrupted run resumes with `--resume` or `--merge-only`.

```powershell
python -I tools\world_census.py <levels.rpf> --out work\world\census --shard-size 2147483648 --jobs 12
python -I tools\world_census.py <levels.rpf> --out work\world\census --merge-only
python -I tests\test_world_census.py
```

The default shard size is 2 GiB, the default job count is half the logical CPUs, and the scan is
bounded by `--start` / `--limit` when a partial pass is wanted. `--jobs 1` runs the shards in-process
instead of a process pool, which is useful for small synthetic archives and tests.

## Limits

- The census counts resources, not names. It does not decrypt the RPF9 TOC, so it cannot say which
  archive entry owns a map; `docs/world.md` correlates the measured bands with the placement files
  it could carve.
- Only `0x7a` resources are validated. Other resource types contribute to stream and byte counts but
  are not classified.
- A map row proves the resource has a `CMapData` root and a self-consistent entity table. It does not
  prove the map is reachable, loaded, or complete in the runtime.
- Output contains resource offsets, sizes, counts and content hashes only. It is aggregate metadata;
  keep it out of git along with the archive and carved streams.
