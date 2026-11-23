# Mods

**Status: the mod format and loader are in place; swaps are not applied yet.** `gta6rt` reads every
`mods/<folder>/mod.json`, checks it and logs what it would change. A swap takes effect once the game's
archetype lookup is hooked, and that needs the executable (RT-5): its address differs per version.

A mod never contains game data. It names models the game already has, by the names (or `joaat` hashes)
the game uses, and says what to swap. Everything stays on your own PC, like the rest of the build.

## Where mods go

`gta6rt` looks in a `mods` folder next to `gta6rt.exe` (the port's folder, `Documents\Leonida` by
default), or in the folder `--mods DIR` names. Each mod is a folder with a `mod.json`; a folder without
one is ignored. Folders load in name order, and a later one wins when two swap the same model.

```text
Documents\Leonida\
  gta6rt.exe
  mods\
    mandy_prison_intro\
      mod.json
```

At start the runtime prints what it loaded, for example:

```text
[rt] mods: Mandy in the prison intro
[rt] mods:   archetype player_story000_00 (0xE472B687) -> cs_mandy (0xD7C0CDFF)
[rt] mods: 1 archetype swap(s) loaded; not applied yet: the archetype lookup hook for this executable is not known
```

## mod.json

```json
{
  "name": "Mandy in the prison intro",
  "enabled": true,
  "game_version": "1.0.2.1",
  "archetype_swaps": [
    { "from": "player_story000_00", "to": "cs_mandy", "when": "always" }
  ]
}
```

| Key | Meaning |
|---|---|
| `name` | Shown in the log and the launcher. Defaults to the folder name. |
| `enabled` | `false` keeps the mod installed but off. |
| `game_version` | The package version the mod was made for. |
| `archetype_swaps` | Each entry: when the game asks for the model `from`, it gets `to`. |

`from` and `to` are model names, hashed with `joaat` (lower case, as the game does), or 32-bit hashes
written `0x…`. A swap is one step: a model swapped to another that is itself swapped is not followed
again, so two mods cannot loop. `when` is `always`; a narrower condition (one mission, one cutscene) is
refused until a hook provides the game's mission state, so a mod never silently does more than it says.

The format is in `runtime/src/mods.h`; `runtime/test/mods_test.cpp` (ctest `mods`) covers it.

## The Mandy mod (`mods/mandy_prison_intro`)

Lucia's player model is `player_story000_00` (model id `0xE472B687`); Mandy is `cs_mandy`
(`0xD7C0CDFF`). Both ids are measured in the packs tables ([character_skeleton.md](character_skeleton.md),
"The Mandy search") and are the `joaat` of those names. Mandy has no drawable of her own: `cs_mandy`
uses the same shared female body as Lucia, and her look comes from her own nine-entry variation table.
Swapping the archetype therefore hands the game Mandy's model info, variation table included, rather
than pasting a mesh over Lucia; the game assembles her as it would in her own scenes.

What is not known yet, and decides how far the swap reaches:

- **The intro.** The prison intro's outfit is a component variation on `player_story000_00`. Which
  mission or cutscene id the intro is, and whether its outfit components exist on `cs_mandy`, is not
  measured; with `when: always` Mandy replaces Lucia everywhere the player model is asked for.
- **Mandy's components.** None of the 18 hashes in her variation table resolves to a name, so her
  head, hair and outfit parts are only reachable through her own model info, not by name.

## Day one: finding the hook

The swap needs one function in the executable: the archetype lookup by name hash (RAGE's
`fwArchetypeManager::GetArchetypeFromHashKey`). Ways to find it in `game_scarlett_master.exe`:

1. **Constant search.** The player model id `0xE472B687` should appear as an immediate in the code that
   sets up the story player. Its callers pass it to the lookup (or to a function that does).
2. **Shape.** The lookup takes a 32-bit hash (and an optional out-index), hashes into a bucket table
   and walks a chain comparing a 32-bit key; it is called from hundreds of sites.
3. **Cross-check.** Called with `0xD7C0CDFF` it must return a valid archetype too (`cs_mandy` is a live
   model id: a `story000.rpf` cutscene table uses it).

Once its RVA is known for 1.0.2.1, the loader installs a hook there that calls
`leonida::mods::active().remap(hash)` on the argument before the original runs. That is the only
version-specific piece; the manifests stay as they are.
