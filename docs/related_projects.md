# Related projects and prior art

Leonida is **GPL-3.0-or-later**. Some useful reverse-engineering work lives under other licences, and some of it
cannot be mixed with this repository. This page records what we may learn from each project and where the line is.

The rule is the same for every entry: **facts are fine; code is not.** A format layout, an ABI observation or a
measured behaviour can be reimplemented independently. The source that expressed it cannot be copied.

## rpatty-dev/PercyStar

<https://github.com/rpatty-dev/PercyStar>

A source-available Rust explorer for RPF9/TTPF, with an RDNA2 disassembler, texture decoders and an `ADAT` Opus audio
decoder. Its format observations are relevant to the archive, texture and audio work here.

**Licence:** the "PercyStar Non-Commercial Attribution & Copyleft License (v1.2)". It is source-available, says
"All rights reserved", and adds non-commercial, attribution, naming and share-alike conditions. It is not
GPL-compatible, so **no PercyStar code, comments or tables may be copied into Leonida**.

**What we use it for:** reading about format facts, then implementing them independently. If a fact is useful,
record the observation in our own words, cite the project and verify it against the package or a public
specification. Do not paste even a small function "to save time".

## realdody/americas-recomp

<https://github.com/realdody/americas-recomp>

A **GPL-2.0-only** fork of [boykopovar/AnyPS5](https://github.com/boykopovar/AnyPS5), on its `xbox-series` branch. It
recompiles the same Xbox Series shaders and has run the package's DLLs against a PC D3D12 device behind an Xbox-ABI
proxy.

**Licence:** GPL-2.0-only is **incompatible with GPL-3.0-or-later**. GPLv2-only code cannot be combined into a
GPLv3 work, so **no code may be copied**, and there is no "just this one file" exception.

**What we use it for:** the measured ABI facts only. The useful findings are summarised in
[d3d12_vulkan.md §7](d3d12_vulkan.md#7-prior-art-and-measured-xbox-abi): the Xbox device entry point, the slot
deviations, the descriptor-heap start-handle return convention, the root-signature flag word and the BVH builder's
compute-pipeline path. Those are observations to verify, not code to port. The document paraphrases them; no code or
long passages are copied.

## The line

| Fine | Not fine |
|---|---|
| A format layout, field offset or enum value | Copying the function that parses it |
| An ABI measurement (slot number, return register, flag word) | Copying the proxy, thunk or struct definitions |
| A behaviour observed by running a tool | Copying comments, identifiers or test code |
| A public specification or vendor documentation | A line-by-line port, a translated port, or a "cleaned up" copy |
| A table of facts you rebuilt from your own observations | A table copied from the source, even with renamed columns |

A table of constants is still expression. Rebuild it from your own measurements or a public specification. A
struct definition is still expression even if the field names are changed. If the only way to use a fact is to
copy the surrounding code, stop and find the fact another way.

When in doubt:

1. Write down the fact, not the code.
2. Cite the project and the section or file where you read it.
3. Verify the fact against the game package, a public specification or a test of your own.
4. Implement it independently, in this repository's style.

Never paste code from either project into a pull request, an issue, a commit message or a chat log that becomes
part of the repository history. If a contribution was written while looking at incompatible source, say so in the
pull request so the provenance can be checked.

Other dependencies keep their own licences; see the Credits and License sections of [README.md](../README.md).
