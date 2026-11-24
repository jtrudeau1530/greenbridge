# Source and validation recovery

Greenbridge continues the available Leonida source and validation record. Original
authors, commit messages, license text, copyright notices and third-party attribution
are retained. New recovery documentation does not claim authorship of upstream fixes.

## Available source history

The local repository contained 85 saved refs and 10,648 available commits before
export filtering. It was shallow: missing ancestors cannot be reconstructed from the
saved objects. The exported history closes those available boundaries and removes
the old logo assets and agent instruction files. Consequently commit IDs differ.

- [ref-map.json](ref-map.json) maps saved refs and original source heads to exported refs.
- [commit-map.txt](commit-map.txt) maps available original commits to exported commits.
- `main` starts from the last saved GitHub main, original `de4c37383cad386eaa851047628d7b92c51009c5`.
- `archive/codex/forge/main` retains the separate saved forge line, original
  `27210111ce9d223c6057d130d254e7a82da30e1f`.
- Other `archive/` branches preserve saved PR heads, local committed experiments and
  fork references. Their existence is not a claim that they were merged, tested or adopted.

The historical GitHub and forge refs were separate development lines. Do not infer
that either includes every fix from the other. Some original references may now be
unavailable. The submodule remains a pinned dependency with its original URL; its
remote availability is separate from preservation of this repository.

## Frozen source snapshots

[source-snapshots.json](source-snapshots.json) indexes 59 local snapshots, including
controls, translators, diagnostic builds and an unbuilt alternative. Each entry gives
an exported base, a source-only patch, its SHA-256 and the resulting Git tree identity.
Private provenance, generated output, build directories and internal launchers are
not part of a patch. Original synthetic shader fixtures remain source code.

To reconstruct a snapshot, use its `export_base` and patch from the index:

```sh
git worktree add ../greenbridge-checkpoint <export_base>
git -C ../greenbridge-checkpoint apply --index ../greenbridge/recovery/source-patches/<snapshot>.patch
git -C ../greenbridge-checkpoint write-tree
```

The resulting tree must equal `reconstructed_tree`. This checks source reconstruction;
it does not build or execute the snapshot. Build instructions remain in the main
README and `docs/testing.md`. Diagnostic snapshots may require local settings and
user-owned inputs that are intentionally not distributed.

All snapshots are archived experiments. Several include intrusive readbacks, incomplete
instrumentation or known concurrency limitations. They are evidence for continuing
investigation, not production defaults. See [validation corrections](../docs/validation-recovery.md)
before relying on an older interpretation.

## Recovered test records

[RX 7900 XT validation](../gpu-test-results/2026-11-23-rx7900xt/README.md) follows the
original project's GPU-results layout and separates runtime comparisons, synthetic
suites and historical reports. JSON companions retain per-test outcomes and bounded
run statistics. These records were reconstructed from available files, not invented
or rerun. Original community GPU reports remain in `docs/gpu_test_results.md` and
`gpu-test-results/` with their existing attribution.

The local source repositories, raw evidence and frozen experiments were not modified
by preparing this export. Game-derived data and private machine/coordination records
remain local.
