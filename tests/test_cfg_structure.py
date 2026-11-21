"""cfg_structure.structure: the tree runs exactly the blocks the graph does, in the same order.

    python -I tests/test_cfg_structure.py

Random graphs, random branch outcomes. For each graph the tree is interpreted and compared with a plain walk of
the graph that makes the same choices. Irreducible graphs must be refused (None), reducible ones must not be.
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import cfg_structure as S  # noqa: E402


def chooser(succ, seed):
    """Branch outcomes that depend only on the block and how often it has run, so two runs that execute the
    same blocks make the same choices."""
    visits = {}

    def choose(b):
        k = visits[b] = visits.get(b, 0) + 1
        options = list(succ[b]) or [S.END]
        return options[random.Random(f"{seed}/{b}/{k}").randrange(len(options))]
    return choose


def walk(succ, choose, limit):
    trace, b = [], 0
    while b != S.END and len(trace) < limit:
        trace.append(b)
        b = choose(b)
    return trace


def reducible(succ):
    """Independent test (T1/T2): repeatedly delete self-loops and merge a block into its only predecessor."""
    g = {v: set(s) for v, s in enumerate(succ)}
    seen, work = {0}, [0]
    while work:
        for w in g[work.pop()]:
            if w not in seen:
                seen.add(w)
                work.append(w)
    g = {v: {w for w in s if w in seen} for v, s in g.items() if v in seen}
    changed = True
    while changed and len(g) > 1:
        changed = False
        for v in list(g):
            g[v].discard(v)
            preds = [p for p in g if v in g[p]]
            if v != 0 and len(preds) == 1:
                p = preds[0]
                g[p] |= g.pop(v)
                g[p].discard(v)
                changed = True
                break
    return len(g) == 1


def random_graph(rng):
    n = rng.randint(1, 14)
    kind = rng.random()
    succ = []
    for v in range(n):
        k = rng.choice((0, 1, 1, 2, 2, 2))
        if kind < 0.5:                               # anything goes: many of these are irreducible
            succ.append([rng.randrange(n) for _ in range(k)])
        else:                                        # mostly forward, with jumps back: loops, breaks, early exits
            succ.append([rng.randrange(v + 1, n) if v + 1 < n and rng.random() < 0.75 else rng.randrange(v + 1)
                         for _ in range(k)])
    return succ


def blocks_in(tree):
    out = []
    for it in tree:
        out += [it[1]] if it[0] == "block" else blocks_in(it[2])
    return out


def test_random_graphs():
    rng = random.Random(20261121)
    structured = refused = loops = 0
    for case in range(20000):
        succ = random_graph(rng)
        tree = S.structure(succ)
        assert (tree is not None) == reducible(succ), f"reducibility disagrees for {succ}"
        if tree is None:
            refused += 1
            continue
        structured += 1
        loops += '"loop"' in repr(tree).replace("'", '"')
        placed = blocks_in(tree)
        assert len(placed) == len(set(placed)), f"a block is placed twice: {succ}"
        for seed in range(6):
            want = walk(succ, chooser(succ, seed), 200)
            got = S.run(tree, chooser(succ, seed), 200)
            assert got == want, f"graph {succ} seed {seed}: tree ran {got}, graph runs {want}"
    assert structured > 5000 and refused > 1000 and loops > 2000, (structured, refused, loops)
    return structured, refused, loops


def test_shapes():
    assert S.structure([[]]) == [("block", 0)]
    assert S.structure([[1], [2], []]) == [("block", 0), ("block", 1), ("block", 2)]          # no tests needed
    # A diamond: the two arms in either order (here 2 first), then the join, which must come last.
    assert S.structure([[1, 2], [3], [3], []]) == [("block", 0), ("guard", 2, [("block", 2)]),
                                                    ("guard", 1, [("block", 1)]), ("guard", 3, [("block", 3)])]
    assert S.structure([[0, 1], []]) == [("loop", 0, [("block", 0)]), ("guard", 1, [("block", 1)])]
    assert S.structure([[1, 2], [2], [1]]) is None                                           # two-entry loop


if __name__ == "__main__":
    test_shapes()
    print("ok: %d graphs structured and run, %d irreducible refused, %d with loops" % test_random_graphs())
