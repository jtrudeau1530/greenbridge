# Contributing

Thanks for helping. A few rules keep the project safe to host:

- **No game data, ever.** That covers archives, the executable, carved files, shaders (DXBC, FSSM, disassembly,
  SPIR-V), textures, models, audio, videos, and reports or logs that quote them. `.gitignore` blocks the common
  types; check `git status` before you commit anyway.
- **No keys, decryption or protection workarounds**, and no links to where to get them. The tools only read files
  that are already plain on disk.
- Short excerpts needed to explain a format (a few header bytes, one instruction sequence) are fine in docs and
  comments.

## Before you start

Claim the work in [#156](https://github.com/ItsNotRexed/Leonida/issues/156) first: check its tables and the open
pull requests, post `claiming: <ID>` there (or a one-line description for something not listed yet), and put the ID
in your pull request. When two pull requests overlap anyway, the one that lands first is merged and the other is
asked to rebase or close. GPU test results go in [#166](https://github.com/ItsNotRexed/Leonida/issues/166).

## Working on the code

- Python tools run with `python -I` and use only the standard library unless a file says otherwise.
- `tools/rpf9dump` needs the submodule: `git submodule update --init`.
- A new shader-translator rule goes into the contract table in `docs/design.md`, with how it was confirmed: a GPU
  test, a Mesa/LLVM reference, or a count over the corpus.
- `python -I tests/test_<name>.py` runs one of the unit tests (standard library only; CI runs them all).
- Runtime tests: `cmake -S runtime -B runtime/build`, `cmake --build runtime/build`, then
  `ctest --test-dir runtime/build --output-on-failure`. The CPU tests that need no game data are configured
  on Windows. Tests that need the retail package read `LEONIDA_GAME_DIR` (and `LEONIDA_MOVIE` for the Bink
  decode test) in place and print `SKIP` when the variable is unset; never copy or commit those files.
- Viewer streaming tests: `node --test tests/test_viewer_streaming.cjs` (Node.js 22+, no packages,
  browser, network or game data). These execute the viewer streaming code with a synthetic loader and clock.
- `loader_test` writes small PE images to a `leonida_loader_test` folder in `%TEMP%`. Antivirus that blocks newly
  written executables makes it stop with that reason; exclude the folder or set `LEONIDA_TEST_DIR` to one it skips.
- Tool checks: `python -m compileall -q make_game.py tools tests`. The optional
  `vk_pipeline_check <spirv dir> [cache.bin] [--jobs N]` creates a module (and, for compute shaders, a
  pipeline) for every translated shader and reports failures plus the saved cache path.
- GPU tests are small hand-written RDNA2 programs in `runtime/test/shaders`. `build.py` there turns them into
  SPIR-V (it needs `llvm-mc`, so WSL or a Linux container), and `check.py` runs them on your GPU and compares
  the results. A new test gets its expected values in the comment at its top and a check in `check.py`.
- Say what you tested and on which GPU in the pull request.
- Run the translator tests with `python -m unittest discover -s tests/semantic -v` (see `tests/semantic/README.md`). A new
  translator rule should come with a small hand-written `.asm` fixture and its expected output words.
