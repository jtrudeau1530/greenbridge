"""Arrange a control-flow graph as nested loops and conditionals driven by a block-number variable.

    structure(succ) -> tree, or None when the graph is irreducible

`succ[b]` lists the blocks that block b can go to next (none = the program ends there); block 0 is the entry.
Every block ends by storing its successor's number in a variable `pc` (END when it has none). The tree says where
each block's code goes:

    ("block", b)             run block b
    ("guard", b, [items])    if pc == b: run the items (they start with block b, or with the loop headed by b)
    ("loop", b, [items])     repeat the items, which start with block b, for as long as pc == b afterwards

Why this instead of one loop around a switch: a shader compiler has to rediscover the real control flow from a
switch on a variable that every block writes, which takes it minutes on shaders with a thousand blocks. Here the
loops are loops, and the guards are tests against a value that was just assigned, which fold at once.

How it is built (reducible graphs only, i.e. every loop has a single entry):
  - Each natural loop becomes a "loop" item. Leaving it needs no break: pc simply no longer names the header.
  - Inside a loop (and at the top level) the members are nested by dominance: after a member come the members it
    immediately dominates, each behind a guard, in an order where every forward edge points later. A jump to a
    member that isn't below the current one falls out of the guards it is in, to the level where a later guard
    matches. A jump out of the loop falls out of all of them.
A block is never skipped or run twice in one pass, so any path through the graph is followed exactly.
"""
import collections

END = -1


def _order(succ):
    """Reverse postorder from block 0, the retreating edges of that walk, and the predecessors of each block."""
    state, order, back = {0: 1}, [], []
    stack = [(0, iter(succ[0]))]
    while stack:
        v, it = stack[-1]
        for w in it:
            if w not in state:
                state[w] = 1
                stack.append((w, iter(succ[w])))
                break
            if state[w] == 1:
                back.append((v, w))
        else:
            state[v] = 2
            order.append(v)
            stack.pop()
    rpo = order[::-1]
    preds = collections.defaultdict(list)
    for v in rpo:
        for w in succ[v]:
            preds[w].append(v)
    return rpo, back, preds


def _dominators(rpo, preds):
    """Immediate dominator of each reachable block (Cooper, Harvey and Kennedy)."""
    pos = {v: i for i, v in enumerate(rpo)}
    idom = {rpo[0]: rpo[0]}
    changed = True
    while changed:
        changed = False
        for v in rpo[1:]:
            new = None
            for p in preds[v]:
                if p not in idom:
                    continue
                if new is None:
                    new = p
                    continue
                a, b = p, new
                while a != b:
                    while pos[a] > pos[b]:
                        a = idom[a]
                    while pos[b] > pos[a]:
                        b = idom[b]
                new = a
            if idom.get(v) != new:
                idom[v] = new
                changed = True
    return idom, pos


def structure(succ):
    rpo, back, preds = _order(succ)
    idom, pos = _dominators(rpo, preds)

    def dominates(a, b):
        while pos[b] > pos[a]:
            b = idom[b]
        return a == b

    if not all(dominates(w, v) for v, w in back):
        return None                                   # a loop with two entries: no nesting expresses it

    # Natural loops, and the innermost loop each block is in (named by its header; None = not in a loop).
    bodies = {}
    for v, h in back:
        body = bodies.setdefault(h, {h})
        work = [v]
        while work:
            x = work.pop()
            if x not in body:
                body.add(x)
                work.extend(preds[x])
    inner = {}
    for h in sorted(bodies, key=lambda h: len(bodies[h])):
        for x in bodies[h]:
            inner.setdefault(x, h)
    outer = {}                                        # loop -> the loop around it
    for h in bodies:
        around = [g for g in bodies if g != h and h in bodies[g]]
        outer[h] = min(around, key=lambda g: len(bodies[g])) if around else None

    def member(x, region):
        """The member of `region` that block x belongs to: x itself, or the loop inside the region holding it.
        Members are ("b", block) and ("l", header)."""
        loop = inner.get(x)
        if loop == region:
            return ("b", x)
        while outer[loop] != region:
            loop = outer[loop]
        return ("l", loop)

    # Dominance among the members of each region. A member's parent is the member holding the immediate
    # dominator of its first block; that dominator is always inside the same region.
    children = collections.defaultdict(list)
    for x in rpo[1:]:
        region = inner.get(x)
        if region == x:                               # a loop header: it is placed as its loop, in the outer region
            children[member(idom[x], outer[x])].append(("l", x))
        else:
            children[member(idom[x], region)].append(("b", x))

    def emit(m):
        kind, x = m
        items = [("loop", x, emit(("b", x)))] if kind == "l" else [("block", x)]
        below = sorted(children[m], key=lambda c: pos[c[1]])
        if kind == "b" and len(below) == 1 and list(succ[x]) == [below[0][1]]:
            return items + emit(below[0])             # the only way on, and nothing else leads there: no test
        return items + [("guard", c[1], emit(c)) for c in below]

    return emit(("l", 0) if 0 in bodies else ("b", 0))


def run(tree, choose, limit):
    """Interpret a tree: the blocks it executes, in order, stopping after `limit`. `choose(b)` gives the block
    that b goes to next (END to finish) and stands for whatever b stores in pc."""
    trace, pc = [], [0]

    class Stop(Exception):
        pass

    def go(items):
        for it in items:
            if it[0] == "block":
                trace.append(it[1])
                if len(trace) >= limit:
                    raise Stop
                pc[0] = choose(it[1])
            elif it[0] == "guard":
                if pc[0] == it[1]:
                    go(it[2])
            else:
                while True:
                    go(it[2])
                    if pc[0] != it[1]:
                        break

    try:
        go(tree)
    except Stop:
        pass
    return trace
