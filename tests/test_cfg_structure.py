"""cfg_structure.structure: the tree runs exactly the blocks the graph does, in the same order.

    python -I tests/test_cfg_structure.py

Random graphs, random branch outcomes. For each graph the tree is interpreted and compared with a plain walk of
the graph that makes the same choices. Irreducible graphs must be refused (None), reducible ones must not be.
"""
from functools import cache
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import cfg_structure as S  # noqa: E402


@cache
def branch_index(seed, block, visit, count):
    # The choice depends only on these numbers, so different graphs can reuse it.
    return random.Random(f"{seed}/{block}/{visit}").randrange(count)


def chooser(succ, seed):
    """Branch outcomes that depend only on the block and how often it has run, so two runs that execute the
    same blocks make the same choices."""
    visits = {}

    def choose(b):
        options = succ[b]
        if len(options) < 2:
            return options[0] if options else S.END
        k = visits[b] = visits.get(b, 0) + 1
        return options[branch_index(seed, b, k, len(options))]
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
    structured = refused = loops = split_ok = split_cap = 0
    for case in range(20000):
        succ = random_graph(rng)
        tree = S.structure(succ)
        assert (tree is not None) == reducible(succ), f"reducibility disagrees for {succ}"
        if tree is None:
            refused += 1
            res = S.split_irreducible(succ)
            if res is not None:
                new_succ, orig = res
                assert len(new_succ) <= min(2 * len(succ), len(succ) + 64)
                stree = S.structure(new_succ)
                assert stree is not None, f"split_irreducible returned irreducible graph for {succ}"
                splaced = blocks_in(stree)
                assert len(splaced) == len(set(splaced)), f"a split block is placed twice: {succ}"
                for seed in range(6):
                    want = walk(succ, chooser(succ, seed), 200)
                    visits = {}

                    def split_choose(b, _orig=orig, _ns=new_succ, _seed=seed, _v=visits):
                        ob = _orig[b]
                        k = _v[ob] = _v.get(ob, 0) + 1
                        opts = _ns[b]
                        if not opts:
                            return S.END
                        return opts[random.Random(f"{_seed}/{ob}/{k}").randrange(len(opts))]

                    got = [orig[b] for b in S.run(stree, split_choose, 200)]
                    assert got == want, f"split graph {succ} seed {seed}: tree ran {got}, graph runs {want}"
                split_ok += 1
            else:
                split_cap += 1
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
    assert split_ok > 1000 and split_ok + split_cap == refused, (split_ok, split_cap, refused)
    return structured, refused, loops, split_ok, split_cap


def test_shapes():
    assert S.structure([[]]) == [("block", 0)]
    assert S.structure([[1], [2], []]) == [("block", 0), ("block", 1), ("block", 2)]          # no tests needed
    # A diamond: the two arms in either order (here 2 first), then the join, which must come last.
    assert S.structure([[1, 2], [3], [3], []]) == [("block", 0), ("guard", 2, [("block", 2)]),
                                                    ("guard", 1, [("block", 1)]), ("guard", 3, [("block", 3)])]
    assert S.structure([[0, 1], []]) == [("loop", 0, [("block", 0)]), ("guard", 1, [("block", 1)])]
    assert S.structure([[1, 2], [2], [1]]) is None                                           # two-entry loop
    ns, orig = S.split_irreducible([[1, 2], [2], [1]])
    assert S.structure(ns) is not None and len(ns) == 4 and orig == [0, 1, 2, 2]
    # Nested irreducible cycle {2, 3} inside an outer reducible loop headed by 1.
    nested = [[1], [2, 3], [3, 4], [2, 4], [1, 5], []]
    assert S.structure(nested) is None
    ns2, orig2 = S.split_irreducible(nested)
    assert S.structure(ns2) is not None and len(ns2) == 7
    # Regression for single-node infinite unrolling: SCC {1, 2, 3} with entries 1 (h) and 2 (e), plus an
    # internal subcycle 2 <-> 3 that does not pass through 1. Single-node cloning bounces 2' -> 3 -> 3' -> 2 -> ...
    # until hitting the cap; closed-region cloning duplicates R = {2, 3} once (+2 blocks, 5 -> 7).
    subcycle = [[1, 2], [3, 4], [3], [1, 2], []]
    assert S.structure(subcycle) is None
    ns3, orig3 = S.split_irreducible(subcycle)
    assert S.structure(ns3) is not None and len(ns3) == 7
    for seed in range(6):
        want = walk(subcycle, chooser(subcycle, seed), 200)
        visits = {}

        def sc_choose(b, _o=orig3, _ns=ns3, _s=seed, _v=visits):
            ob = _o[b]
            k = _v[ob] = _v.get(ob, 0) + 1
            opts = _ns[b]
            return S.END if not opts else opts[random.Random(f"{_s}/{ob}/{k}").randrange(len(opts))]

        assert [orig3[b] for b in S.run(S.structure(ns3), sc_choose, 200)] == want
    # Cap enforcement: max_splits=0 must refuse an irreducible graph cleanly.
    assert S.split_irreducible([[1, 2], [2], [1]], max_splits=0) is None


if __name__ == "__main__":
    test_shapes()
    print("ok: %d graphs structured and run, %d irreducible (%d split-structured, %d capped), %d with loops" %
          (lambda s, r, l, ok, cap: (s, r, ok, cap, l))(*test_random_graphs()))
