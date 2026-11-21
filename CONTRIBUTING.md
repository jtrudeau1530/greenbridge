# Contributing

Thanks for helping. A few rules keep the project safe to host:

- **No game data, ever.** That covers archives, the executable, carved files, shaders (DXBC, FSSM, disassembly,
  SPIR-V), textures, models, audio, videos, and reports or logs that quote them. `.gitignore` blocks the common
  types; check `git status` before you commit anyway.
- **No keys, decryption or protection workarounds**, and no links to where to get them. The tools only read files
  that are already plain on disk.
- Short excerpts needed to explain a format (a few header bytes, one instruction sequence) are fine in docs and
  comments.

## Working on the code

- Python tools run with `python -I` and use only the standard library unless a file says otherwise.
- `tools/rpf9dump` needs the submodule: `git submodule update --init`.
- A new shader-translator rule goes into the contract table in `docs/design.md`, with how it was confirmed: a GPU
  test, a Mesa/LLVM reference, or a count over the corpus.
- `python -I tests/test_<name>.py` runs one of the unit tests (standard library only; CI runs them all).
- GPU tests are small hand-written RDNA2 programs in `runtime/test/shaders`. `build.py` there turns them into
  SPIR-V (it needs `llvm-mc`, so WSL or a Linux container), and `check.py` runs them on your GPU and compares
  the results. A new test gets its expected values in the comment at its top and a check in `check.py`.
- Say what you tested and on which GPU in the pull request.
- Run the translator tests with `python -m unittest discover -s tests/semantic -v` (see `tests/semantic/README.md`). A new
  translator rule should come with a small hand-written `.asm` fixture and its expected output words.
