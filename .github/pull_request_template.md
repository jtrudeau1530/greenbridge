Claimed in the coordination issue as: <!-- the item ID from the open "Coordinating who works on what" issue,
     for example TEST-17, or a link to your claim -->

Depends on: <!-- #N for every pull request this one builds on or conflicts with (the coordination issue's
     Commands section, AGENTS.md); delete the line if it builds on none and conflicts with none -->

### What and why

<!-- What this changes and the problem it solves. -->

### Tested

<!-- What you ran (e.g. `python -I scripts/preflight_pr.py`) and on which GPU and driver (CONTRIBUTING.md).
     For the shader translator: how many shaders translate and pass spirv-val, before and after.
     For hardware/vendor-specific workarounds: state the premise and the direct measurement that confirmed it. -->

AI-assisted: <!-- yes or no (AGENTS.md asks a coding agent to say so) -->

### Checklist

- [ ] No game data, keys or decryption (CONTRIBUTING.md), including in logs or reports quoted here.
- [ ] `python -I scripts/preflight_pr.py` passes on a fresh rebase of `main`.
- [ ] A new translator rule is added under `docs/contracts/` and rendered into `docs/design.md` via `python -I tools/build_design_doc.py`.
