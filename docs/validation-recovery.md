# Recovered Linux/Wine validation

This page records the evidence available on 2026-11-23. The original project layout,
runtime tests, translator tests, GPU result notes and design documentation are retained.
The additional results are in [the RX 7900 XT archive](../gpu-test-results/2026-11-23-rx7900xt/README.md).
They were reconstructed from saved local results; the reconstruction did not rerun the tests.

## Environment and scope

The recorded validation machine used an AMD Radeon RX 7900 XT, Arch Linux, Wine 11.18
and RADV Mesa 26.2.3. Later runtime builds used clang-cl 23.1.6 with xwin and
RelWithDebInfo. Earlier records may use a different compiler version. Synthetic tests,
shader translation coverage and bounded retail observations are separate measurements.

`main` retains the last saved GitHub source tree, with Greenbridge documentation and
recovery material added. It does not silently adopt the experimental patches.
The tested forge-based snapshots and the saved PR heads are available separately through
[the source recovery index](../recovery/README.md).

## What was established

| Comparison | Observed result | Limit |
|---|---|---|
| Console list Reset routing, c3r to c3s | A PC Reset and successful first Close replaced the earlier Close failure. | The worker-thread assertion remained; its earlier HRESULT was not recorded. |
| Five, then seven missing compute programs | Existing #6446 and #8526 changes translated and validated the selected programs; all seven known misses were served in the corresponding bounded run. | Shader validity and successful lookup do not establish correct rendered output. The assertion remained on the older base. |
| Forge 4fe to 5972, with the same shader set | The 5972 comparison reached the 30-second bound without the earlier exception, with repeated submissions and at least 256 successful PresentX calls. | Credit belongs to the combined source delta; no single fix was isolated. No visible frame was established. |
| Presenter capture and positive control | Normal sampled frames were entirely black. The diagnostic magenta clear appeared in captured output. | Synthetic colour validates part of the presentation path, not game rendering. Capture copied every frame and affected timing. |
| Two 120-second capture runs | No exception was observed within the bounds; every saved normal sample was black, including samples through presenter frame 1681. | Later unsampled images and completion of boot are unknown. Some original black PPMs were deleted after statistics were saved. |
| Rectangle variants | The variant was created and bound; synthetic pixel and binding tests passed, and the negative control failed the intended assertion. | Normal scene and presented samples remained zero. Changing topology did not produce game content. |
| Pixel/vertex probes and triangle control | A solid-green pixel probe covered the scene target, while the composite target retained its clear. The old triangle path behaved the same for the composite. | The fullscreen vertex probe also caused rectangle-variant refusal and triangle fallback; it changed multiple variables and left interpolants undefined. |
| Composite input-layout inspection | The observed position attributes, stride, first vertex and three-vertex byte span were compatible and in bounds. | CPU input bytes and a valid layout do not prove valid post-vertex positions or GPU execution. |
| Selected three-slot descriptor pull | A bounded diagnostic copied title-side data into the layer at the selected table's middle slot. Two synthetic controls distinguished callback installation from scan extent. | Other selected slots remained null; all six sampled scene/plane images stayed black. This is not a production synchronization policy. |
| Pull plus #9586 mirror | Another selected slot became non-null in both title and layer samples; the matched composite range's mirror flag was off. All six images stayed black. | The origin of that data is unproved. A newly added WARP mirror assertion failed, despite the same set of failing test names. |

## Corrections that take precedence over older reports

Historical reports retain their original observations and interpretations. Apply these
corrections when reading them:

- A surviving magenta clear means no different colour survived at the sampled pixels.
  It does not prove that no draw ever reached a target: later clears, identical output
  and different sampling times can obscure writes.
- An alpha minimum of zero, maximum of 255 and rounded mean of 127.500 does not prove
  exactly half the pixels had alpha 255. Only later exact histograms establish counts.
- A 120-second timeout is an experiment bound, not completed loading or boot.
- Capture limits gated file writes, not the per-present staging copy. Reaching the file
  limit did not establish that capture overhead stopped.
- The early `vsin2` zero-input explanation used global latest-address watches and log
  adjacency, not submitted-draw identity. Later tagged measurements found equal,
  non-zero CPU prefixes at the replay hook. The earlier causal claim is withdrawn.
- `vsin3` still had a process-global pointer to stack copy statistics, global signature
  state and callback numbering, and sampled a 64-byte vertex-buffer prefix without
  accounting for the draw's first vertex and stride. Its counters and prefix equality
  do not prove race-free per-submission attribution, actual vertex values or GPU visibility.
- A null CPU descriptor sample does not, by itself, prove that a shader sampled that
  descriptor. Exact stage identity and accesses were inspected separately. The numeric
  transformed vertex result was not evaluated on the GPU.
- The slot-origin diagnostic used incomplete writer instrumentation and historical
  range notes. “No API write recorded” is narrower than “no writer exists.” Direct
  stores, unhooked paths, lifetime/reuse and large-copy limits remain relevant.
- Non-zero descriptor-shaped bytes are not a validated descriptor decode. Data appearing
  after an experiment is not provenance of a title memcpy or proof of a particular producer.
- `shader_library_corpus` sometimes printed SKIP without failing its process. Such a
  run did not exercise the corpus, even when an aggregate pass count included it.
- The pullmirror suite retained five failing test names but added failing assertions
  inside `device_proxy` and `device_proxy_trace`. Its WARP mirror branch was exercised
  and failed; it must not be described as merely skipped or unchanged baseline failure.
- Pullmirror required a test-only integration adaptation: a fake non-null SRV description
  used by the upstream mirror fixture was replaced by a null description because the
  combined mock copies non-null descriptions. The mirror assertion stayed unchanged.

## Tests and controls

Most late 70-test runs recorded 64 passed, one skipped and five failed. The common
failures were `device_proxy`, `device_proxy_trace`, `dstorage`,
`run_tiny_game_abort_exit_codes` and `run_tiny_tls_ga`. The recovered per-test records
are authoritative for each individual invocation, including skips. Different earlier
bases had different tests and failure sets.

The selected-pull fixture adds one passing case and two expected-failure controls:
an uninstalled source must fail the pull assertions; using the original one-slot
span must fail coverage of the second and third slots. The mirror control disables
only the mirror copy and must fail the mirror fixture's first assertion. A passing
expected-failure CTest entry means the control failed as intended, not that its
disabled feature worked.

## What remains unresolved

There is no verified non-black game image or playable state in these records.
Descriptor synchronization and the composite's constant inputs remain open lines of
investigation. Translation coverage gaps also remain, but no causal link from those
gaps to black output was established. Experimental source snapshots preserve these
investigations, including failed and superseded hypotheses, without enabling them
on the default branch.

The saved upstream ownership and frontier threads were #9677 and #8851. Later reads
returned HTTP 404 and source fetches failed. These records cannot establish newer
ownership, merges, handoffs, or why the original services became unavailable.

## Privacy and reconstruction limits

The archive contains counts, API/test names, source commit identities and sanitized
historical reports. It omits game files, extracted/translated shaders, entry names,
game-derived hashes, guest addresses, screenshots, raw run logs, local paths, credentials
and private coordination records. Missing results are labelled missing, not inferred
to pass. Reconstruction is not a fresh test of Greenbridge's default branch.
