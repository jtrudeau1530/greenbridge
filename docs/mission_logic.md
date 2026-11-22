# Mission state tables in the script archives

`story000_script.rpf` and `script.rpf` are mostly not scripts in the sense of an instruction stream. The
containers that carry mission structure are **state machines**: arrays of 32-byte records keyed by a joaat
hash of a state name, each record pointing at a pool element that carries world transforms and an
archetype hash. This page records the layout, the vocabulary that resolved, and the shape of the flows.
Counts and offsets only — no payload, and only the small name excerpt needed to explain the key column.

## Where they are

| | |
|---|---|
| containers | `PSIN`-tagged, 2,309 of them across the two archives |
| record chunks | 8,986 |
| records | 40,804 |
| distinct record hashes | 6,065 |
| files with a record array | 2,035 |

Note that `PSIN` here is the same chunk family the reader meets elsewhere; the mission arrays live in the
`LSNS`-style payload side of those containers, not in the `PSIN` descriptor side.

## Record layout

32 bytes:

```
u32 key │ u32 0 │ u32 colB │ u32 0 │ u32 (u16 kind, u16 offset BE) │ u32 band │ u32 (u16 n, u16 n) │ u32 0
```

Two details cost a false start and are worth stating:

- **`band` is not zero.** It is an ascending band counter (`0,0,1,2,2,…`). A validator that requires
  `field[5] == 0` misclassifies the real tables as opaque blobs.
- **`offset` is a byte offset, big-endian in its half**, and it is **file-absolute for most records**:
  2,087 of 2,207 entity-bearing records resolve that way, 429 resolve through the "next even-tagged pool
  chunk" rule, which only fits one table family. The proof of byte-ness: consecutive deltas equal
  `count * 4`, and values pack back to back from 0.

## The pool element

64 bytes, 16 dwords:

```
u32 archetype hash │ 0 │ 0 │ 0 │ f32 pos.xyz │ f32 pos.xyz (dup) │ rot.a,b,c (dup) │ u32 hash │ u32 linkHash │ 0 │ 0
```

The 16-dword stride is a histogram peak (319 windows against 25/25 for neighbours), and the duplicated
archetype hash plus the duplicated position components are what identify the shape. Positions are true
world coordinates in RAGE space, Z up.

## The key column

Keys are **joaat of the state name, lower-cased first**: joaat here folds case, which is not true of every
implementation of it. Measured on 340 ground-truth pairs: a folding implementation reproduces **340/340**,
a case-sensitive one only **91/340** (329 reference names contain an uppercase byte). Any name lookup must
fold before hashing, or it will invent phantom names.

Resolved keys fall in four provenance tiers, kept separate because they are not equally strong:

| tier | count | how it was resolved |
|---|---:|---|
| document-proven | 22 | the name appears literally in the package's own text metadata |
| register-measured | 178 | found by grammar search, in-register, discounting ≈3.6 expected chance collisions |
| earlier register | 114 | same, from a weaker pass |
| positional | 4 | still unnamed, but distinguishable by position in the machine |

Coverage: 167 → 485 of 6,065 distinct hashes (2.75 % → 8.00 %); record keys 5,846 → 8,314 of 40,804
(14.33 % → 20.38 %); real (non-structural) states named **53.5 %** (2,675 of 4,996). 26.1 % of real states
are still hex, and 65 % of records sit in structural states rather than gameplay states.

## Shape of the flows

The corpus is highly templated: **113 identical-state-set clusters cover 1,676 of 2,035 files (82.4 %)**.
Naming one cluster therefore names dozens of files. Repeated clusters seen so far include
`g1798–g1804`, `g2209–g2212`, `g0986–g0995`.

Flows read as ordered states with an entity count per state. Representative shapes (names where resolved,
hex otherwise; `<…>` marks a positional label):

```
g0082/83   guardaggro → startRobbery → fleeRobbery → robberyCombat → disrupt → INTRO
g1599      stealvehicle → delivervehicle
g0554      fleestore → selfEscape
g1774–79   stealboat → dropoff → boatchase → deliver
g1096–1126 Approach → Leave → <pre-HOPOUT> → HOPOUT → INTRO
```

**135 initiation→terminal shaped jobs** were listed; only 3 are fully bounded end to end.

Phase tagging over the named states (primary match): setup 958, exit 400, action 344, cutscene 130,
aftermath 97, law reaction 29. Law reaction being the rarest phase is a property of the corpus, not of the
tagging.

## What a flow does and does not say

A flow gives **ordering, states and entities**. It carries no conditions, objectives, or durations: there
are no opcodes here, and no jump table. Two consequences for a runtime:

1. State order in the file is not necessarily play order.
2. The remaining unnamed hashes still have positional meaning — e.g. one is the single looping beat
   between two cutscenes, another is the "get the car into position" beat before a vehicle exit — so a
   trace is interpretable even where the name is missing.

## Mission identity

The package's own text metadata carries a 145-entry mission table. Only **one** of those ids is provably
bound to one of these tables, so most outlines are unattributed. Joining ids to tables is an open problem;
a plain joaat of the ids does not appear in the program data (0 of 290 variants in 170.9 MB), so the join
is not by name hash.
