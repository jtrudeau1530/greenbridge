# Security

Leonida builds everything from the player's own copy, so the two things that can hurt the project are a
security bug in the tools and game data that reaches the repository. This page covers both.

## Report it privately

Never put the details in a public issue, pull request, comment or Discord channel. Quoting the problem in public
spreads it further.

- **Preferred:** the repository's **Security** tab → **Advisories** → **Report a vulnerability**. Only the
  maintainers see it.
- **If that button is missing:** ask a maintainer for a private channel on [Discord](https://discord.gg/rexed).
  Your first message should say only that you have something to report, with no details.

Say what you found, where (file, commit, pull request or issue number) and when you noticed it. Don't attach the
file itself.

## Game data in the repository

This covers everything [CONTRIBUTING.md](CONTRIBUTING.md) rules out: archives, the executable or DLLs, carved
files, shaders in any form (DXBC, FSSM, disassembly, SPIR-V), textures, models, audio, videos, keys, and logs or
reports that quote any of them. It applies wherever it shows up: a commit, a branch, a pull request, an issue or
a comment attachment.

**If you see it:** report it privately as above. Don't download it, mirror it, fork the repository to keep it,
or comment on the pull request or issue that holds it.

**If you pushed it yourself:**

1. Report it privately right away, even if you have already deleted the file. A later commit that deletes a file
   leaves it in the history.
2. Don't push again to fix it, and don't open another pull request. The maintainers rewrite the history once.
3. If it is in your own fork, delete the fork or clean it the same way.

**What the maintainers do:**

1. Close or merge the open pull requests that touch the affected history, then rewrite every ref with
   [git-filter-repo](https://github.com/newren/git-filter-repo) (2.47 or later, `--sensitive-data-removal`,
   `--invert-paths --path <file>` for each path the file ever had) and check that the data is gone.
2. Force-push with `git push --force --mirror origin`. Pull request refs are read-only and keep the old history.
3. Ask GitHub Support to dereference or delete the affected pull requests and to remove cached views. Give the
   repository name, the number of affected pull requests and the first changed commit from the git-filter-repo
   output.
4. Ask the owners of forks that contain the commits to delete them.
5. Hide or delete issue and comment attachments that hold the data.
6. Tell contributors to re-clone or rebase onto the new history. A `git pull` and `git push` from an old clone
   puts the data back.

## Security bugs in the tools

The tools and the runtime read large binary files and run external programs (`git`, `llvm-mc`, `spirv-as`,
`ffmpeg`). In scope:

- A file in a game folder that makes a tool write outside `--out` or `--work`, or run a command it shouldn't.
- Unsafe parsing in `tools/` or `runtime/` (for example `eval`, path traversal, or unchecked sizes and offsets).
- The build downloading or running anything other than the pinned `third_party/rpf9-explorer` submodule.
- `viewer/serve.py` being reachable from anywhere but `127.0.0.1`.
- CI workflows that expose tokens or run untrusted code with write access.

Out of scope: the game itself, Xbox or Rockstar services, and any copy protection. Leonida does not work around
protection, and reports that ask it to are closed.

Only the latest `main` gets fixes.
