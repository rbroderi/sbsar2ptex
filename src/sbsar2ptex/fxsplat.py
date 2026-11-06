"""FX-Maps precomputed: a tree's patterns as a list of splats, packed for GLSL.

A scattering FX-Map can place each of thousands of patterns anywhere, so a
per-pixel tree walk visits them all. Instead the tree is walked once here,
with the graph inputs at their defaults and Substance's FX-Map generator
(sbsar2ptex.fxrng), into splats: where each pattern lands, its size,
rotation and value. They are packed into an RGBA8 data texture, one float32
per texel (bit for bit; Material Maker reads PNG bytes exactly), with a grid
of cells each listing the splats that can reach it, in drawing order. The
shader then draws only its cell's splats. The cost: the map no longer
follows its parameters live.

The splats are in Substance's drawing order (fxdraw.order), in pattern
space: the shader maps each pixel to it through the render region, and
draws each splat's tile copies (fxdraw.copies).

Texture layout, texel indices: splats from 0 (RECORD texels each: center x
and y, half size x and y, rotation in radians, pattern, supplement, blend
max as 0 or 1, then the instance color: gray's, or premultiplied rgba),
then the grid's cells (start and count in the index list, as uints), then
the index list (splat numbers, as uints).
"""

import math
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import MutableMapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sbsar2ptex import fxdraw
from sbsar2ptex import fxmap
from sbsar2ptex import fxseed
from sbsar2ptex.function import SETSLOT
from sbsar2ptex.function import Diverged
from sbsar2ptex.function import FunctionError
from sbsar2ptex.function import Lanes
from sbsar2ptex.function import decode
from sbsar2ptex.function import evaluate
from sbsar2ptex.function import evaluate_lanes
from sbsar2ptex.function import glsl_type
from sbsar2ptex.fxrng import FxRng
from sbsar2ptex.fxrng import FxRngLanes
from sbsar2ptex.fxrng import depth_step
from sbsar2ptex.fxrng import depth_steps
from sbsar2ptex.fxrng import iterate_step
from sbsar2ptex.fxrng import iterate_steps
from sbsar2ptex.rng import UINT32_MASK
from sbsar2ptex.sbsasm import Node

WIDTH = 1024
"""Texels per row of the data texture."""

MAX_SPLATS = 1_000_000
"""More than this and the map stays live (its texture would be too big)."""

CHILD = ((-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25))


class TooBig(Exception):  # noqa: N818
    """The tree draws more splats than MAX_SPLATS."""


@dataclass(slots=True)
class Splat:
    """One pattern as drawn: where, how big, how turned, which, how strong."""

    x: float
    y: float
    hx: float
    hy: float
    angle: float
    pattern: int
    """A pattern id, or an image's (fxmap.image_id)."""
    suppl: float
    blend_max: bool
    value: list[float]
    """The instance color (float32s): gray, [opacity times weight]; color,
    premultiplied rgba, alpha the opacity's times weight."""


def splats(
    node: Node,
    tree: fxmap.Tree | None,
    settings: fxmap.Settings,
    inputs: dict[int, list[float]],
    *,
    color: bool,
    size: int,
    slots: MutableMapping[int, list] | None = None,
) -> Sequence[Splat]:
    """Return the patterns the FX-Map ``node`` draws, in drawing order.

    In pattern space (the render region maps it to the image). ``inputs``
    are the graph inputs' values by uid (defaults); ``slots`` the graph's
    slot store, which get slots read. Raises FunctionError for
    what cannot be evaluated here (sampling an input) and TooBig past
    MAX_SPLATS.

    The tree is walked in Substance's order of calls, but the subtrees that
    keep no state the rest of the walk reads (``_deferrable``) are left for
    later and drawn together, a whole tree level at a time (``_lanes``).
    Should that raise anything, or lanes part ways, the plain walk runs
    instead: it raises, or draws, as it would. Only TooBig found once
    every function ran without error (_Counted) stands: the plain walk
    would have run them too, and drawn past MAX_SPLATS.
    """
    kw = {"color": color, "size": size, "slots": slots}
    try:
        return _walk(node, tree, settings, inputs, defer=True, **kw)
    except _Counted:
        raise
    except Exception:  # noqa: BLE001 - the plain walk raises it again, in its order
        return _walk(node, tree, settings, inputs, defer=False, **kw)


class _Counted(TooBig):
    """TooBig, found with the whole tree evaluated."""


@dataclass
class _Deferred:
    """A subtree the walk left for ``_lanes``: where it was entered."""

    tree: Any
    incoming: int
    pos: tuple[float, float]
    depth: int
    number: float
    store: dict[int, list]
    """The variables it reads (none of which it sets)."""


def _walk(
    node: Node,
    tree: fxmap.Tree | None,
    settings: fxmap.Settings,
    inputs: dict[int, list[float]],
    *,
    color: bool,
    size: int,
    defer: bool,
    slots: MutableMapping[int, list] | None = None,
) -> Sequence[Splat]:
    """Return ``splats``, ``_deferrable`` subtrees left for ``_lanes`` if ``defer``."""
    n = float(1 << size)
    weights = depth_weights(settings.hurst, fxmap.max_depth(tree))
    later = _deferrable(node, tree, settings) if defer else {}

    blocks: dict[int, Any] = {}
    # One variable store for the whole map: its setup sets variables the
    # tree's functions read, and they keep state from one call to the next
    # (a walk over a grid, pass by pass), in Substance's order of calls.
    variables: dict[int, list] = {}
    seeds = fxseed.Seeds()

    def block_at(code: int) -> Any:
        block = blocks.get(code)
        if block is None:
            block = blocks[code] = decode(node.at(code, (node.end - code) // 4))
        return block

    def run(code: int, state: int, lane: str, ctx: dict[int, list[float]]) -> list:
        rand = FxRng(state, lane).rand
        return evaluate(block_at(code), inputs, ctx, rand, variables, slots=slots)

    def param(
        p: fxmap.Param | None,
        default: list[float],
        state: int,
        lane: str,
        ctx: dict[int, list[float]],
    ) -> list[float]:
        if p is None:
            return default
        vals = run(p.code, state, lane, ctx) if p.state == fxmap.CODE else p.value
        vals = [float(v) for v in vals]
        return vals * len(default) if len(vals) == 1 else vals[: len(default)]

    if settings.seed is not None:
        root = settings.seed
    elif settings.seed_code is not None:
        root = int(run(settings.seed_code, 0, "frameoffset", {})[0])
    else:
        root = 0
    if settings.setup_code is not None:
        n_px = [n, n]
        run(settings.setup_code, 0, "frameoffset", {1: n_px, 3: [float(size)] * 2})
    # (the regions' functions run here, in Substance's order: they may read
    # or set the map's variables)
    _regions(settings, lambda code: run(code, 0, "frameoffset", {}))
    out: list[Splat | _Deferred] = []

    def enter(
        t: fxmap.Tree | None,
        incoming: int,
        pos: tuple[float, float],
        cell: float,
        depth: int,
        number: float,
    ) -> None:
        if t is None:
            return
        reads = later.get(id(t))
        if reads is not None:
            store = {v: variables[v] for v in reads if v in variables}
            out.append(_Deferred(t, incoming, pos, depth, number, store))
            return
        ctx = {
            0: [0.0],
            1: [n, n],
            3: [float(size), float(size)],
            8: list(pos),
            10: [number],
            12: [float(depth)],
        }
        own = t.seed or 0
        if t.seed_code is not None:  # first of the node's functions, each visit
            block = block_at(t.seed_code)
            own = seeds.seed(
                block, inputs, ctx, variables, incoming=incoming, slots=slots
            )
        inherited = not isinstance(t, (fxmap.Iterate, fxmap.Switch)) or t.inherited
        e = ((incoming if inherited else 0) + own) & UINT32_MASK
        if isinstance(t, fxmap.Quadrant):
            p = t.params
            op = param(p.get("opacity"), [1.0] * 4, e, "opacity", ctx)
            bo = param(p.get("branchoffset"), [0.0, 0.0], e, "branchoffset", ctx)
            fo = param(p.get("frameoffset"), [0.0, 0.0], e, "frameoffset", ctx)
            sz = param(p.get("patternsize"), [1.0, 1.0], e, "patternsize", ctx)
            rot = param(p.get("patternrotation"), [0.0], e, "patternrotation", ctx)
            sup = param(p.get("patternsuppl"), [0.0], e, "patternsuppl", ctx)
            at = (pos[0] + bo[0], pos[1] + bo[1])
            pattern = t.pattern
            if t.image_index is not None:
                index = int(run(t.image_index.code, e, "imageindex", ctx)[0])
                pattern = fxmap.image_id(index, t.image_flags)
            elif t.image is not None and t.image < len(settings.patterns):
                pattern = fxmap.image_id(t.image, t.image_flags)
            if pattern is not None:
                value = instance_color(
                    op, weights[depth], settings.opacity, color=color
                )
                # (nothing drawn: no color, no supplement added, no max)
                faint = not any(value[3:] if color else value) and not (
                    (t.blend_max and not color)
                    or (isinstance(pattern, int) and pattern in fxmap.OFFSET and sup[0])
                )
                if not faint:
                    out.append(
                        Splat(
                            at[0] + fo[0],
                            at[1] + fo[1],
                            cell * sz[0] / 2,
                            cell * sz[1] / 2,
                            fxdraw.f32(fxdraw.f32(rot[0]) * fxdraw.f32(2 * math.pi)),
                            pattern,
                            sup[0],
                            t.blend_max,
                            value,
                        )
                    )
                    if len(out) > MAX_SPLATS:
                        raise TooBig
            spawn = e
            for kid, (ox, oy) in zip(t.children, CHILD, strict=False):
                spawn = depth_step(spawn, depth + 1)
                if kid is not None:
                    child = (at[0] + ox * cell, at[1] + oy * cell)
                    enter(kid, spawn, child, cell / 2, depth + 1, number)
        elif isinstance(t, fxmap.Iterate):
            if isinstance(t.count, fxmap.Param):
                count = int(run(t.count.code, e, "iteratecount", ctx)[0])
            else:
                count = t.count
            # (output 0 gets the incoming state, not the effective one: 2.6)
            enter(t.first, incoming, pos, cell, depth, number)
            passes = e
            for i in range(count):
                passes = iterate_step(passes)
                enter(t.repeated, passes, pos, cell, depth, float(i))
        else:
            if isinstance(t.value, fxmap.Param):
                value = bool(run(t.value.code, e, "switch", ctx)[0])
            else:
                value = t.value
            enter(t.if_true if value else t.if_false, e, pos, cell, depth, number)

    enter(tree, root & UINT32_MASK, (0.5, 0.5), 1.0, 0, 0.0)
    if not defer:
        return [out[k] for k in fxdraw.order(out)]
    at = [k for k, d in enumerate(out) if isinstance(d, _Deferred)]
    later = [out[k] for k in at]
    # the walk's own splats met before each entry's next one (all, the last's)
    met_own = np.cumsum([isinstance(s, Splat) for s in out], dtype=np.int64)
    own = [int(met_own[k]) for k in at[1:]] + [int(met_own[-1]) if out else 0]
    kw = {"color": color, "size": size, "own": own, "slots": slots}
    entry, cols, values = _lanes(later, block_at, inputs, weights, settings, **kw)
    walked = [k for k, s in enumerate(out) if isinstance(s, Splat)]
    mine = [out[k] for k in walked]
    # the walk's own splats and the deferred ones, by where the walk met them
    # (a deferred entry's splats in their order, as the sort is stable)
    met = np.array(at, dtype=np.int64)[entry]
    where = np.concatenate([np.array(walked, dtype=np.int64), met])
    order = np.argsort(where, kind="stable")
    if len(order) > MAX_SPLATS:
        raise _Counted
    own = _columns(mine)
    width = 4 if color else 1
    rows = np.array([s.value for s in mine], dtype=np.float64).reshape(-1, width)
    cols = {k: np.concatenate([own[k], v])[order] for k, v in cols.items()}
    values = np.concatenate([rows, values])[order]
    drawing = fxdraw.order_of(cols["pattern"], cols["blend_max"])
    return SplatTable({k: v[drawing] for k, v in cols.items()}, values[drawing])


def _reads_sets(block: Any) -> tuple[set[int], set[int]]:
    """Return the variables a block reads before it sets them, and those it sets.

    A set inside a while loop does not count as before a read: run no
    times, it leaves what an earlier call set.
    """
    inside: set[int] = set()
    for loop in block.loops().values():
        inside.update(range(loop.start, loop.at + 1))
    reads: set[int] = set()
    sets: set[int] = set()
    set_here: set[int] = set()
    for k, ins in enumerate(block.instrs):
        if ins.op == SETVAR:
            sets.add(ins.operands[1])
            if k not in inside:
                set_here.add(ins.operands[1])
        elif ins.op == GETVAR and ins.operands[0] not in set_here:
            reads.add(ins.operands[0])
    return reads, sets


def _own_codes(t: fxmap.Tree) -> list[int]:
    """Return the function blocks a tree node itself runs, its seed's first."""
    seed = [t.seed_code] if t.seed_code is not None else []
    if isinstance(t, fxmap.Quadrant):
        codes = [p.code for p in t.params.values() if p.state == fxmap.CODE]
        image = [t.image_index.code] if t.image_index is not None else []
        return seed + codes + image
    param = t.count if isinstance(t, fxmap.Iterate) else t.value
    return seed + ([param.code] if isinstance(param, fxmap.Param) else [])


def _kids(t: fxmap.Tree) -> list[fxmap.Tree | None]:
    if isinstance(t, fxmap.Quadrant):
        return list(t.children)
    if isinstance(t, fxmap.Iterate):
        return [t.first, t.repeated]
    return [t.if_false, t.if_true]


def _deferrable(
    node: Node, tree: fxmap.Tree | None, settings: fxmap.Settings
) -> dict[int, list[int]]:
    """Return the subtrees the walk may leave for later (by id), and what each reads.

    A subtree may be drawn out of the walk's order when nothing it sets is
    read by any of the tree's functions before they set it themselves:
    then no call outside it sees what it does, and within it the variables
    it reads (set before it is entered) hold still. Only the outermost
    such subtrees, as the walk meets them, are left for later. None are
    when a tree function sets a value slot (the graph's store, which get
    slots read): it would change under them. Nor is a subtree holding a
    node whose seed is a function (fxseed): the plain walk runs those, and
    may defer below them.
    """
    try:
        found: dict[int, tuple[set[int], set[int]]] = {}
        for code in _codes(tree, fxmap.Settings()):  # (the tree's alone)
            block = decode(node.at(code, (node.end - code) // 4))
            if any(ins.op == SETSLOT for ins in block.instrs):
                return {}
            found[code] = _reads_sets(block)
    except Exception:  # noqa: BLE001 - unreadable functions: the walk's to raise
        return {}
    shared = set().union(*(r for r, _ in found.values()))
    memo: dict[int, tuple[set[int], set[int]]] = {}
    seeded: set[int] = set()
    going: set[int] = set()

    def visit(t: fxmap.Tree | None) -> tuple[set[int], set[int]]:
        """Return a subtree's variables read (before set) and set."""
        if t is None:
            return set(), set()
        if id(t) in going:
            raise RecursionError  # a tree that loops: walk it as it is
        if id(t) not in memo:
            going.add(id(t))
            reads, sets = set(), set()
            for code in _own_codes(t):
                reads |= found[code][0]
                sets |= found[code][1]
            if t.seed_code is not None:
                seeded.add(id(t))
            for kid in _kids(t):
                r, s = visit(kid)
                reads |= r
                sets |= s
                if kid is not None and id(kid) in seeded:
                    seeded.add(id(t))
            memo[id(t)] = (reads, sets)
            going.discard(id(t))
        return memo[id(t)]

    try:
        visit(tree)
    except RecursionError:
        return {}
    return {
        k: sorted(r) for k, (r, s) in memo.items() if not s & shared and k not in seeded
    }


LANES_MAX = 1 << 22
"""Most lanes a run of deferred subtrees may spread to, all levels together."""


class _Spread(Diverged):
    """The deferred subtrees spread to more than LANES_MAX lanes."""


@dataclass
class _Front:
    """Lanes entering one tree node: each one of the walk's entries into it."""

    incoming: np.ndarray
    """Incoming states (uint64s holding 32 bits)."""
    x: np.ndarray
    y: np.ndarray
    number: np.ndarray
    path: list[np.ndarray]
    """Where each lane's splats go: its deferred subtree's number, then the
    child slots and Iterate passes (output 0 as 0, pass i as 1 + i) taken."""
    store: dict[int, list[np.ndarray]]
    """The variables the subtree reads, a component array each."""

    def __len__(self) -> int:
        return len(self.incoming)

    def take(self, sel: np.ndarray) -> "_Front":
        """Return the lanes ``sel`` picks (a mask or indices)."""
        return _Front(
            self.incoming[sel],
            self.x[sel],
            self.y[sel],
            self.number[sel],
            [k[sel] for k in self.path],
            {v: [c[sel] for c in cs] for v, cs in self.store.items()},
        )

    @staticmethod
    def join(fronts: list["_Front"]) -> "_Front":
        """Return the lanes of ``fronts``, one after another."""
        if len(fronts) == 1:
            return fronts[0]
        first = fronts[0]

        def cat(get: Callable[["_Front"], np.ndarray]) -> np.ndarray:
            return np.concatenate([get(f) for f in fronts])

        return _Front(
            cat(lambda f: f.incoming),
            cat(lambda f: f.x),
            cat(lambda f: f.y),
            cat(lambda f: f.number),
            [cat(lambda f, k=k: f.path[k]) for k in range(len(first.path))],
            {
                v: [cat(lambda f, v=v, c=c: f.store[v][c]) for c in range(len(cs))]
                for v, cs in first.store.items()
            },
        )


def _ints(v: np.ndarray) -> np.ndarray:
    """Return int(v) of each lane (int64), raising where int() would."""
    if v.dtype.kind == "f":
        if not np.isfinite(v).all():
            msg = "cannot convert float NaN or infinity to integer"
            raise ValueError(msg)
        if (np.abs(v) >= float(LANES_MAX) ** 2).any():
            raise Diverged
        v = np.trunc(v)
    return v.astype(np.int64)


def _stores(entries: list[_Deferred]) -> dict[int, list[np.ndarray]]:
    """Return the deferred entries' variables as lane arrays (one kind each)."""
    out: dict[int, list[np.ndarray]] = {}
    for v in entries[0].store:
        values = [d.store.get(v) for d in entries]
        if any(x is None or len(x) != len(values[0]) for x in values):
            raise Diverged
        comps = []
        for c in range(len(values[0])):
            col = [x[c] for x in values]
            ints = [isinstance(x, int) for x in col]
            if any(ints) != all(ints):
                raise Diverged
            comps.append(np.array(col, dtype=np.int64 if ints[0] else np.float64))
        out[v] = comps
    if any(d.store.keys() != entries[0].store.keys() for d in entries):
        raise Diverged
    return out


def _lanes(
    entries: list[_Deferred],
    block_at: Callable[[int], Any],
    inputs: dict[int, list[float]],
    weights: list[float],
    settings: fxmap.Settings,
    *,
    color: bool,
    size: int,
    own: Sequence[int] = (),
    slots: MutableMapping[int, list] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Return the deferred subtrees' splats in the walk's order (_gather's).

    ``own``: how many of the walk's own splats it met before each entry's
    next one (by entry; all of them, for the last).

    Entries into one node at one depth run together, as lanes: each node's
    functions run once for all its lanes (function.evaluate_lanes,
    fxrng.FxRngLanes), the walk's arithmetic over arrays, so the splats
    are the walk's, bit for bit. A Quadrant's children that are one node
    take all four slots' lanes at once. Each splat keeps its path, to be
    put in the walk's order at the end.
    """
    n = float(1 << size)
    drawn: list[dict[str, Any]] = []
    spread = 0
    f = np.float32
    tau = fxdraw.f32(2 * math.pi)

    def run(code: int, at: tuple[np.ndarray, _Front], lane: str, ctx: dict) -> list:
        """Run a node's function: ``at`` its lanes' effective states and lanes."""
        states, front = at
        rand = FxRngLanes(states, lane).rand
        block = block_at(code)
        store = dict(front.store)
        return evaluate_lanes(block, inputs, ctx, rand, len(front), store, slots=slots)

    def param(
        p: fxmap.Param | None,
        default: list[float],
        at: tuple[np.ndarray, _Front],
        lane: str,
        ctx: dict,
    ) -> list:
        if p is None:
            return default
        if p.state == fxmap.CODE:
            vals = [v.astype(np.float64) for v in run(p.code, at, lane, ctx)]
        else:
            vals = [float(v) for v in p.value]
        return vals * len(default) if len(vals) == 1 else vals[: len(default)]

    def enter(t: fxmap.Tree | None, front: _Front, depth: int) -> None:
        nonlocal spread
        count = len(front)
        if t is None or not count:
            return
        spread += count
        if spread > LANES_MAX:
            raise _Spread
        cell = 0.5**depth
        if t.seed_code is not None:  # (never deferred: _deferrable)
            raise Diverged
        inherited = not isinstance(t, (fxmap.Iterate, fxmap.Switch)) or t.inherited
        base = front.incoming if inherited else np.zeros_like(front.incoming)
        e = (base + np.uint64(t.seed or 0)) & np.uint64(UINT32_MASK)
        here = (e, front)
        ctx = {
            0: [0.0],
            1: [n, n],
            3: [float(size), float(size)],
            8: [front.x, front.y],
            10: [front.number],
            12: [float(depth)],
        }
        if isinstance(t, fxmap.Quadrant):
            p = t.params
            op = param(p.get("opacity"), [1.0] * 4, here, "opacity", ctx)
            bo = param(p.get("branchoffset"), [0.0, 0.0], here, "branchoffset", ctx)
            fo = param(p.get("frameoffset"), [0.0, 0.0], here, "frameoffset", ctx)
            sz = param(p.get("patternsize"), [1.0, 1.0], here, "patternsize", ctx)
            rot = param(p.get("patternrotation"), [0.0], here, "patternrotation", ctx)
            sup = param(p.get("patternsuppl"), [0.0], here, "patternsuppl", ctx)
            at = (front.x + bo[0], front.y + bo[1])
            pattern: Any = t.pattern
            if t.image_index is not None:
                index = _ints(run(t.image_index.code, here, "imageindex", ctx)[0])
                pattern = fxmap.image_id(index, t.image_flags)  # (as arrays)
            elif t.image is not None and t.image < len(settings.patterns):
                pattern = fxmap.image_id(t.image, t.image_flags)
            if pattern is not None:
                _draw(t, front, depth, cell, (op, fo, sz, rot, sup), at, pattern)
            spawn = e
            slots: dict[int, tuple[fxmap.Tree, list[_Front]]] = {}
            kids = zip(t.children, CHILD, strict=False)
            for slot, (kid, (ox, oy)) in enumerate(kids):
                spawn = depth_steps(spawn, depth + 1)
                if kid is not None:
                    lanes = _Front(
                        spawn,
                        np.broadcast_to(at[0] + ox * cell, (count,)),
                        np.broadcast_to(at[1] + oy * cell, (count,)),
                        front.number,
                        [*front.path, np.full(count, slot)],
                        front.store,
                    )
                    slots.setdefault(id(kid), (kid, []))[1].append(lanes)
            for kid, fronts in slots.values():
                enter(kid, _Front.join(fronts), depth + 1)
        elif isinstance(t, fxmap.Iterate):
            if isinstance(t.count, fxmap.Param):
                passes = _ints(run(t.count.code, here, "iteratecount", ctx)[0])
            else:
                passes = np.full(count, t.count, dtype=np.int64)
            first = _Front(  # (output 0: the incoming states, 2.6)
                front.incoming, front.x, front.y, front.number,
                [*front.path, np.zeros(count, dtype=np.int64)], front.store,
            )  # fmt: skip
            enter(t.first, first, depth)
            if t.repeated is None:
                return
            passes = np.maximum(passes, 0)
            if int(passes.sum()) > LANES_MAX:
                raise _Spread
            lane = np.repeat(np.arange(count), passes)
            i = np.arange(len(lane)) - (np.cumsum(passes) - passes)[lane]
            again = front.take(lane)
            again.incoming = iterate_steps(e[lane], i)
            again.number = i.astype(np.float64)
            again.path.append(1 + i)
            enter(t.repeated, again, depth)
        else:
            if isinstance(t.value, fxmap.Param):
                value = run(t.value.code, here, "switch", ctx)[0] != 0
            else:
                value = np.full(count, bool(t.value))
            for branch, sel in ((t.if_true, value), (t.if_false, ~value)):
                taken = front.take(sel)
                taken.incoming = e[sel]
                enter(branch, taken, depth)

    def _draw(
        t: fxmap.Quadrant,
        front: _Front,
        depth: int,
        cell: float,
        params: tuple,
        at: tuple,
        pattern: Any,
    ) -> None:
        op, fo, sz, rot, sup = params
        count = len(front)
        w, o = f(weights[depth]), f(settings.opacity)
        if color:  # instance_color, a lane each
            a = np.asarray(op[3], dtype=np.float64).astype(f) * w * o
            value = [np.asarray(c, dtype=np.float64).astype(f) * a for c in op[:3]]
            value.append(a)
            lit = value[3] != 0
        else:
            value = [np.asarray(op[-1], dtype=np.float64).astype(f) * w * o]
            lit = value[0] != 0
        offset = np.isin(pattern, list(fxmap.OFFSET)) & (np.asarray(sup[0]) != 0)
        keep = np.broadcast_to(lit | (t.blend_max and not color) | offset, (count,))
        if not keep.any():
            return
        rows = {
            "x": at[0] + fo[0],
            "y": at[1] + fo[1],
            "hx": cell * sz[0] / 2,
            "hy": cell * sz[1] / 2,
            "angle": Lanes.f32(Lanes.f32(rot[0]) * tau),
            "pattern": pattern,
            "suppl": sup[0],
        }
        batch = {
            k: np.broadcast_to(np.asarray(v), (count,))[keep] for k, v in rows.items()
        }
        batch["value"] = [
            np.broadcast_to(np.asarray(c, dtype=np.float64), (count,))[keep]
            for c in value
        ]
        batch["blend_max"] = t.blend_max
        batch["path"] = [k[keep] for k in front.path]
        drawn.append(batch)

    def run_entries(first: int, stop: int) -> None:
        """Draw entries ``first`` to ``stop``: into one node at one depth, together."""
        groups: dict[tuple[int, int], list[int]] = {}
        for k in range(first, stop):
            d = entries[k]
            groups.setdefault((id(d.tree), d.depth), []).append(k)
        for members in groups.values():
            group = [entries[k] for k in members]
            front = _Front(
                np.array([d.incoming for d in group], dtype=np.uint64),
                np.array([d.pos[0] for d in group], dtype=np.float64),
                np.array([d.pos[1] for d in group], dtype=np.float64),
                np.array([d.number for d in group], dtype=np.float64),
                [np.array(members, dtype=np.int64)],
                _stores(group),
            )
            enter(group[0].tree, front, group[0].depth)

    # Entries run in the walk's order, as many at a time as LANES_MAX allows
    # (a run that spreads further is halved). Once the runs so far, with the
    # walk's own splats met before the next entry, are past MAX_SPLATS,
    # every function before that point in the walk's order has run without
    # error (the walk's own ones ran before), so the plain walk would get
    # there too and raise TooBig: _Counted.
    first, chunk, total = 0, len(entries), 0
    with np.errstate(all="ignore"):
        while first < len(entries):
            stop = min(first + chunk, len(entries))
            mark, spread = len(drawn), 0
            try:
                run_entries(first, stop)
            except _Spread:
                if stop - first == 1:
                    raise
                del drawn[mark:]
                chunk = (stop - first) // 2
                continue
            total += sum(len(b["x"]) for b in drawn[mark:])
            if total + (own[stop - 1] if own else 0) > MAX_SPLATS:
                raise _Counted
            first = stop
    return _gather(drawn, 4 if color else 1)


def _gather(
    drawn: list[dict[str, Any]], width: int
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Return _lanes' batches in the walk's order: (deferred entry, columns, values).

    By path: the walk draws a node's own splat before its children's, so a
    shorter path is padded with -1s, to sort first. Columns as _columns',
    values a row of ``width`` a splat.
    """
    if not drawn:
        return np.zeros(0, dtype=np.int64), _columns([]), np.zeros((0, width))
    deepest = max(len(b["path"]) for b in drawn)
    columns = [
        np.concatenate(
            [
                b["path"][level] if level < len(b["path"]) else np.full(len(b["x"]), -1)
                for b in drawn
            ]
        )
        for level in range(deepest)
    ]
    order = np.lexsort(columns[::-1])
    kinds = {"pattern": np.int64, "blend_max": bool}
    cols = {
        key: np.concatenate([np.asarray(b[key], dtype=kinds.get(key, np.float64))
                             for b in drawn])[order]
        for key in ("x", "y", "hx", "hy", "angle", "suppl", "pattern")
    }  # fmt: skip
    blend = np.concatenate([np.full(len(b["x"]), b["blend_max"]) for b in drawn])
    cols["blend_max"] = blend[order]
    values = np.stack(
        [np.concatenate([b["value"][c] for b in drawn])[order] for c in range(width)],
        axis=1,
    )
    return columns[0][order], cols, values


class SplatTable(Sequence):
    """Splats kept as columns of arrays, made into Splat objects as they are read.

    What ``splats`` returns for a map it drew as lanes: the same splats,
    field for field (Python floats, ints, bools), without making hundreds
    of thousands of objects that packing and raster read as arrays anyway.
    """

    def __init__(self, cols: dict[str, np.ndarray], values: np.ndarray) -> None:
        """Splats from _columns-like columns and their values' rows."""
        self.cols = cols
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, k: Any) -> Any:
        if isinstance(k, slice):
            return list(self)[k]
        c = self.cols
        return Splat(
            float(c["x"][k]), float(c["y"][k]), float(c["hx"][k]), float(c["hy"][k]),
            float(c["angle"][k]), int(c["pattern"][k]), float(c["suppl"][k]),
            bool(c["blend_max"][k]), self.values[k].tolist(),
        )  # fmt: skip

    def __iter__(self) -> Iterator[Splat]:
        c = {k: v.tolist() for k, v in self.cols.items()}
        fields = zip(
            c["x"], c["y"], c["hx"], c["hy"], c["angle"], c["pattern"], c["suppl"],
            c["blend_max"], self.values.tolist(), strict=True,
        )  # fmt: skip
        return (Splat(*f) for f in fields)


def has_images(spl: Sequence[Splat]) -> bool:
    """Whether any splat draws an image (not a built-in pattern)."""
    return bool((_columns(spl)["pattern"] < 0).any())


def depth_weights(hurst: float, deepest: int) -> list[float]:
    """Return each depth's weight (float32s), 0 to ``deepest``.

    hurst^depth over the sum of hurst^j for j to the deepest level (any
    Quadrant's, drawing or not); all 1 for hurst 0.
    """
    if hurst == 0:
        return [1.0] * (deepest + 1)
    total = sum(hurst**j for j in range(deepest + 1))
    return [fxdraw.f32(hurst**d / total) for d in range(deepest + 1)]


def instance_color(
    op: list[float], weight: float, opacity: float, *, color: bool
) -> list[float]:
    """Return a pattern's color: its opacity times weight and global opacity.

    Gray: the opacity's last component (an inline gray opacity is one value;
    a function's float4 gives its w). Color: premultiplied rgba. Float32.
    """
    f = np.float32
    if not color:
        return [float(f(op[-1]) * f(weight) * f(opacity))]
    a = f(op[3]) * f(weight) * f(opacity)
    return [float(f(op[0]) * a), float(f(op[1]) * a), float(f(op[2]) * a), float(a)]


UNIT_REGION = [0.0, 1.0, 0.0, 1.0]
"""The render and tiling regions' default: the image is the tile."""


def _regions(
    settings: fxmap.Settings, run: Callable[[int], list]
) -> tuple[list[float], list[float]]:
    """Return the map's render and tiling regions, x0 x1 y0 y1 each.

    ``run`` evaluates a code block, after the map's setup. The tiling region
    is a tile of pattern space that wraps, drawn only within itself; the
    render region is the part of pattern space the image shows.
    """
    out = []
    for roi in (settings.render_roi, settings.tiling_roi):
        if isinstance(roi, int):  # a function, of the setup's variables
            roi = [float(v) for v in run(roi)]  # noqa: PLW2901
        out.append(UNIT_REGION if roi is None else roi)
    return out[0], out[1]


def regions(
    node: Node,
    settings: fxmap.Settings,
    inputs: dict[int, list[float]],
    size: int,
    *,
    slots: MutableMapping[int, list] | None = None,
) -> tuple[list[float], list[float]]:
    """Return ``_regions`` for the FX-Map ``node``, its setup run first.

    ``slots`` is the graph's slot store, which get slots read.
    """
    variables: dict[int, list] = {}
    n = float(1 << size)

    def run(code: int, ctx: dict[int, list[float]]) -> list:
        block = decode(node.at(code, (node.end - code) // 4))
        rand = FxRng(0, "frameoffset").rand
        return evaluate(block, inputs, ctx, rand, variables, slots=slots)

    if settings.setup_code is not None:
        run(settings.setup_code, {1: [n, n], 3: [float(size)] * 2})
    return _regions(settings, lambda code: run(code, {}))


def stateful(node: Node, tree: fxmap.Tree | None, settings: fxmap.Settings) -> bool:
    """Whether the map's functions share variables (one reads what another set).

    Then each function's GLSL alone cannot know them: the map must be
    walked in order, as ``splats`` does.
    """
    for code in _codes(tree, settings):
        block = decode(node.at(code, (node.end - code) // 4))
        set_here: set[int] = set()
        for ins in block.instrs:
            if ins.op == SETVAR:
                set_here.add(ins.operands[1])
            elif ins.op == GETVAR and ins.operands[0] not in set_here:
                return True
    return False


def _codes(tree: fxmap.Tree | None, settings: fxmap.Settings) -> list[int]:
    """Return every function block of a map: its setup, then its tree's."""
    codes: list[int] = [settings.setup_code] if settings.setup_code else []
    seen: set[int] = set()

    def walk(t: fxmap.Tree | None) -> None:
        if t is None or id(t) in seen:
            return
        seen.add(id(t))
        if t.seed_code is not None:  # (run first of the node's)
            codes.append(t.seed_code)
        if isinstance(t, fxmap.Quadrant):
            codes.extend(p.code for p in t.params.values() if p.state == fxmap.CODE)
            if t.image_index is not None:
                codes.append(t.image_index.code)
            for c in t.children:
                walk(c)
        elif isinstance(t, fxmap.Iterate):
            if isinstance(t.count, fxmap.Param):
                codes.append(t.count.code)
            walk(t.first)
            walk(t.repeated)
        else:
            if isinstance(t.value, fxmap.Param):
                codes.append(t.value.code)
            walk(t.if_false)
            walk(t.if_true)

    walk(tree)
    return codes


def variable_types(
    node: Node, tree: fxmap.Tree | None, settings: fxmap.Settings
) -> dict[int, str]:
    """Return the GLSL type of every variable the map's functions use.

    Its type is what sets it (else what reads it); one set as two types
    is an error.
    """
    found: dict[int, str] = {}
    read: dict[int, str] = {}
    for code in _codes(tree, settings):
        for ins in decode(node.at(code, (node.end - code) // 4)).instrs:
            if ins.op == SETVAR:
                kind = glsl_type(ins.type, ins.width)
                if found.setdefault(ins.operands[1], kind) != kind:
                    message = f"FX-Map variable {ins.operands[1]} is set as two types"
                    raise FunctionError(message)
            elif ins.op == GETVAR:
                read.setdefault(ins.operands[0], glsl_type(ins.type, ins.width))
    return {**read, **found}


GETVAR, SETVAR = 0x04, 0x07
"""Function ops reading and writing a variable."""


RECORD = {False: 9, True: 12}
"""Texels per splat record: gray, color."""


@dataclass
class Packed:
    """A data texture and where its parts start."""

    image: np.ndarray
    """RGBA8, WIDTH texels a row: each texel a little-endian word."""
    grid: int
    """Cells per side."""
    cells: int
    """The first cell's texel."""
    index: int
    """The index list's first texel."""


Region = tuple[float, float, float, float]
"""A render or tiling region: x0 x1 y0 y1 of pattern space."""


def _columns(spl: Sequence[Splat]) -> dict[str, np.ndarray]:
    """Return the splats' scalar fields as arrays (float64; pattern int64)."""
    if isinstance(spl, SplatTable):
        return spl.cols
    f = np.float64
    return {
        "x": np.array([s.x for s in spl], dtype=f),
        "y": np.array([s.y for s in spl], dtype=f),
        "hx": np.array([s.hx for s in spl], dtype=f),
        "hy": np.array([s.hy for s in spl], dtype=f),
        "angle": np.array([s.angle for s in spl], dtype=f),
        "suppl": np.array([s.suppl for s in spl], dtype=f),
        "pattern": np.array([s.pattern for s in spl], dtype=np.int64),
        "blend_max": np.array([s.blend_max for s in spl], dtype=bool),
    }


def _floors(values: np.ndarray, up: bool = False) -> np.ndarray:  # noqa: FBT001, FBT002
    """Return math.floor (ceil if ``up``) of each value, raising as it would.

    The first value (in order) past the finite raises math's error.
    """
    bad = ~np.isfinite(values)
    if bad.any():
        first = float(values.ravel()[np.flatnonzero(bad.ravel())[0]])
        (math.ceil if up else math.floor)(first)
    return (np.ceil(values) if up else np.floor(values)).astype(np.int64)


def _ragged(lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return, for runs of ``lengths``, each entry's run and place in it."""
    run = np.repeat(np.arange(len(lengths)), lengths)
    starts = np.cumsum(lengths) - lengths
    return run, np.arange(len(run)) - starts[run]


def _axis_copies(
    c: np.ndarray, b: np.ndarray, lo: float, hi: float, *, wrap: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Return fxdraw.copies' offsets along one axis: (splat, offset), in order."""
    w = hi - lo
    k0 = _floors((lo - b - c) / w) - 1  # (found, and raising, either way)
    k1 = _floors((hi + b - c) / w, up=True) + 2
    if wrap:
        run, at = _ragged(np.maximum(k1 - k0, 0))
        offs = (k0[run] + at).astype(np.float64) * w
    else:
        run = np.arange(len(c))
        offs = np.zeros(len(c)) * w
    cc, bb = c[run], b[run]
    keep = (cc + offs - bb < hi) & (cc + offs + bb > lo)
    return run[keep], offs[keep]


def _pairs(
    n: int, xs: tuple[np.ndarray, np.ndarray], ys: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return each splat's copies, y outer and x inner: (splat, x part, y part).

    ``xs`` and ``ys`` are per axis (splat, value) runs, splats in order;
    the parts are values of those.
    """
    nx = np.bincount(xs[0], minlength=n)
    ny = np.bincount(ys[0], minlength=n)
    sx, sy = np.cumsum(nx) - nx, np.cumsum(ny) - ny
    run, at = _ragged(nx * ny)
    per = nx[run]
    return run, xs[1][sx[run] + at % per], ys[1][sy[run] + at // per]


def _copy_boxes(
    cols: dict[str, np.ndarray],
    render: Region,
    tiling: Region,
    tile: tuple[bool, bool],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``_boxes`` of every splat, in order: (splat, x0 x1 y0 y1 rows)."""
    angle = cols["angle"].tolist()  # (math's cos and sin: fxdraw.box's)
    c = np.abs(np.array([math.cos(a) for a in angle], dtype=np.float64))
    s = np.abs(np.array([math.sin(a) for a in angle], dtype=np.float64))
    hx, hy = np.abs(cols["hx"]), np.abs(cols["hy"])
    bx, by = hx * c + hy * s, hx * s + hy * c
    x0, x1, y0, y1 = tiling
    xs = _axis_copies(cols["x"], bx, x0, x1, wrap=tile[0])
    ys = _axis_copies(cols["y"], by, y0, y1, wrap=tile[1])
    k, ox, oy = _pairs(len(angle), xs, ys)
    rx0, rx1, ry0, ry1 = render
    rw, rh = rx1 - rx0, ry1 - ry0
    x, y, bx, by = cols["x"][k], cols["y"][k], bx[k], by[k]
    boxes = np.stack(
        [
            (x + ox - bx - rx0) / rw,
            (x + ox + bx - rx0) / rw,
            (y + oy - by - ry0) / rh,
            (y + oy + by - ry0) / rh,
        ],
        axis=1,
    )
    return k, boxes


def _value_rows(spl: Sequence[Splat]) -> np.ndarray | None:
    """Return the splats' values as rows (float64), None if of several widths."""
    if isinstance(spl, SplatTable):
        return spl.values
    if len({len(s.value) for s in spl}) > 1:
        return None
    if not spl:
        return np.zeros((0, 0))
    return np.array([s.value for s in spl], dtype=np.float64)


def _records(spl: Sequence[Splat], rec: int) -> np.ndarray:
    """Return the splat records' words (uint32s), RECORD[color] floats a splat."""
    rows = _value_rows(spl)
    if rows is None or (spl and rows.shape[1] != rec - 8):
        floats: list[float] = []
        for s in spl:
            floats += [s.x, s.y, s.hx, s.hy, s.angle, float(s.pattern), s.suppl]
            floats += [1.0 if s.blend_max else 0.0, *s.value]
        if len(floats) != rec * len(spl):
            msg = "splat records of the wrong size"
            raise ValueError(msg)
        return np.array(floats, dtype="<f4").view("<u4")
    cols = _columns(spl)
    table = np.zeros((len(spl), rec), dtype=np.float64)
    for j, key in enumerate(("x", "y", "hx", "hy", "angle", "pattern", "suppl")):
        table[:, j] = cols[key]
    table[:, 7] = cols["blend_max"]
    if spl:
        table[:, 8:] = rows
    return table.astype("<f4").view("<u4").ravel()


def pack(
    spl: Sequence[Splat],
    *,
    color: bool,
    tile: tuple[bool, bool],
    render: Region = (0.0, 1.0, 0.0, 1.0),
    tiling: Region = (0.0, 1.0, 0.0, 1.0),
) -> Packed:
    """Return the splats and a cell grid of them as a data texture.

    The grid (8 to 256 cells a side) covers the image; a cell lists the
    splats any copy of which (fxdraw.copies) reaches it, in drawing order.
    It is the finest whose index list stays within 16 entries a splat on
    average.
    """
    rec = RECORD[color]
    words = _records(spl, rec)
    k, boxes = _copy_boxes(_columns(spl), render, tiling, tile)
    count = max(len(spl), 1)

    def lists(g: int) -> np.ndarray:
        """Return the (cell, splat) pairs as cell * count + splat, sorted."""
        lo = _floors(boxes * g)
        cx0, cx1 = np.maximum(lo[:, 0], 0), np.minimum(lo[:, 1], g - 1)
        cy0, cy1 = np.maximum(lo[:, 2], 0), np.minimum(lo[:, 3], g - 1)
        nx, ny = np.maximum(cx1 - cx0 + 1, 0), np.maximum(cy1 - cy0 + 1, 0)
        run, at = _ragged(nx * ny)
        cells = (cy0[run] + at // nx[run]) * g + cx0[run] + at % nx[run]
        return np.unique(cells * count + k[run])

    grid, cells = 8, lists(8)
    g = 16
    while g <= 256:  # noqa: PLR2004
        finer = lists(g)
        if len(finer) > 16 * count + g * g:
            break
        grid, cells = g, finer
        g *= 2
    members = np.bincount(cells // count, minlength=grid * grid)
    table = np.stack([np.cumsum(members) - members, members], axis=1).ravel()
    start = len(words)
    index_at = start + 2 * grid * grid
    data = np.concatenate([words, table.astype("<u4"), (cells % count).astype("<u4")])
    # A power of two of rows: texel centers then sit exactly where the shader
    # samples. At 4187 or 5251 rows, NVIDIA's sampler lands a sliver off a
    # center and blends in its neighbor, garbling the word.
    rows = 1 << (-(-len(data) // WIDTH) - 1).bit_length()
    flat = np.zeros(rows * WIDTH, dtype="<u4")
    flat[: len(data)] = data
    return Packed(flat.view(np.uint8).reshape(rows, WIDTH, 4), grid, start, index_at)


RASTER_FLOATS = 16
"""Floats per splat for the sbs plugin's Splats node (raster)."""

_TAU = np.float32(6.2831855)


def raster(
    spl: Sequence[Splat],
    tile: tuple[bool, bool],
    tiling: Region = (0.0, 1.0, 0.0, 1.0),
    render: Region = (0.0, 1.0, 0.0, 1.0),
) -> np.ndarray | None:
    """Return gray splats as the sbs plugin's Splats node draws them; None if unsure.

    Substance's GPU engine draws a pattern as one triangle over its
    bounding box, which it sets up in float32 from the pattern's size and
    its rotation, in turns, from radians and back: RASTER_FLOATS a tile
    copy (_copies32, of the tiling region), in order: the color
    (straight rgba: the gray value, opaque), the box's size and center,
    the pattern's size, the rotation and the supplement, then the pattern
    and max blending (1 or 0), and two zeros. Patterns of no area draw
    nothing and are left out. None when a copy that reaches the render
    region (the image) ends at the tiling region's edge (_copies32): which
    side Substance takes it is not known. Float32s, a flat array.
    """
    f = np.float32
    every = _columns(spl)
    drawn = ~((every["hx"] == 0) | (every["hy"] == 0))
    cols = {k: v[drawn] for k, v in every.items()}
    w, h = (2 * cols["hx"]).astype(f), (2 * cols["hy"]).astype(f)
    turns = cols["angle"].astype(f) / _TAU
    angle = turns * _TAU
    c, n = np.abs(np.cos(angle)), np.abs(np.sin(angle))
    bx = np.abs(w) * c + np.abs(h) * n
    by = np.abs(w) * n + np.abs(h) * c
    copies = _copies32(cols, tiling, tile, render)
    if copies is None:
        return None
    k, x, y = copies
    if isinstance(spl, SplatTable):
        first = spl.values[:, 0]
    else:
        first = np.array([s.value[0] for s in spl], dtype=np.float64)
    v = first[drawn].astype(f)
    out = np.zeros((len(k), RASTER_FLOATS), dtype=f)
    out[:, 0] = out[:, 1] = out[:, 2] = v[k]
    out[:, 3] = 1.0
    out[:, 4], out[:, 5], out[:, 6], out[:, 7] = bx[k], by[k], x, y
    out[:, 8], out[:, 9], out[:, 10] = w[k], h[k], turns[k]
    out[:, 11] = cols["suppl"].astype(f)[k]
    out[:, 12] = cols["pattern"].astype(np.float64)[k]
    out[:, 13] = cols["blend_max"][k]
    return out.ravel()


def float_bytes(values: np.ndarray | list[float]) -> bytes:
    """Return values as little-endian float32s (``raster``'s, for a ptex)."""
    return np.asarray(values, dtype="<f4").tobytes()


def _copies32(
    cols: dict[str, np.ndarray],
    tiling: Region,
    tile: tuple[bool, bool],
    render: Region,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return the centers of the patterns' tile copies (float32); None if one is unsure.

    As (pattern, x, y), each pattern's copies y outer, x inner.
    fxdraw.copies' test: the copy's box overlaps the tiling region,
    strictly. A box that ends at the region's edge, to float32 rounding,
    may be drawn or not: x_region5's disc ends at 0.6 - 0.5 + 0.15, which
    is 0.25 + 2^-25 in float32, and neither Substance nor the inline
    shader draws it (where a strict test in float32 would). Such a copy
    that reaches the render region (whose pixels it would draw) makes it
    None; one past it draws nothing either way (a full-size pattern's
    copies at the default regions). Float32 throughout, numpy's scalar
    rules: a Python float meeting a float32 is rounded to one first.
    """
    f = np.float32
    a = cols["angle"].astype(f)
    ca, sa = np.cos(a), np.sin(a)
    hx, hy = cols["hx"].astype(f), cols["hy"].astype(f)
    b = (np.abs(hx * ca) + np.abs(hy * sa), np.abs(hx * sa) + np.abs(hy * ca))
    count = len(a)
    axes, unsure = [], []
    for c, bb, (lo, hi), (r0, r1), wrap in (
        (cols["x"].astype(f), b[0], tiling[:2], render[:2], tile[0]),
        (cols["y"].astype(f), b[1], tiling[2:], render[2:], tile[1]),
    ):
        w = f(hi - lo)
        if wrap:
            k0 = _floors((lo - bb - c) / w) - 1
            k1 = _floors((hi + bb - c) / w, up=True) + 2
            run, at = _ragged(np.maximum(k1 - k0, 0))
            ks = k0[run] + at
        else:
            run, ks = np.arange(count), np.zeros(count, dtype=np.int64)
        cs = c[run] + ks.astype(f) * w
        bb = bb[run]  # noqa: PLW2901
        near = 4 * np.finfo(np.float32).eps * max(abs(lo), abs(hi), 1.0)
        keep = ((cs - bb) < f(hi)) & ((cs + bb) > f(lo))
        doubt = (
            ((np.abs((cs + bb) - lo) <= near) | (np.abs((cs - bb) - hi) <= near))
            & ((cs - bb) < (r1 - near))
            & ((cs + bb) > (r0 + near))
        )
        axes.append((run[keep], cs[keep]))
        unsure.append(np.bincount(run[doubt], minlength=count) > 0)
    drawn = [np.bincount(r, minlength=count) > 0 for r, _ in axes]
    # an unsure copy along one axis, drawn (or unsure) along the other
    for k in (0, 1):
        if (unsure[k] & (drawn[1 - k] | unsure[1 - k])).any():
            return None
    return _pairs(count, axes[0], axes[1])


def write_png(path: Path, packed: Packed) -> None:
    """Write the data texture as an 8-bit RGBA PNG."""
    Image.fromarray(packed.image, "RGBA").save(path)


def pixel_work(
    spl: Sequence[Splat],
    render: Region = (0.0, 1.0, 0.0, 1.0),
    tiling: Region = (0.0, 1.0, 0.0, 1.0),
    tile: tuple[bool, bool] = (True, True),
) -> float:
    """Return about how many patterns the shader draws a pixel, on average."""
    _, boxes = _copy_boxes(_columns(spl), render, tiling, tile)
    x0, x1, y0, y1 = boxes.T
    # Python's min(a, b) is b only when b < a, max(a, b) b only when b > a
    w = np.where(x1 > 1.0, 1.0, x1) - np.where(x0 < 0.0, 0.0, x0)
    h = np.where(y1 > 1.0, 1.0, y1) - np.where(y0 < 0.0, 0.0, y0)
    areas = np.where(w < 0.0, 0.0, w) * np.where(h < 0.0, 0.0, h)
    total = 0.0
    for a in areas.tolist():  # summed in order, as the copies come
        total += a
    return total


BAKE_KINDS = 16
"""Most kinds of splat (pattern, size, angle, supplement) a bake takes."""


def bake(spl: list[Splat], size: int, tile: tuple[bool, bool]) -> np.ndarray | None:
    """Return the gray map the splats add, 2**size a side, or None.

    Only a map tiling both ways with the default regions, of gray splats of
    patterns (not images, nor those adding their supplement) that add (not
    max), of few kinds: each kind's patterns are one kernel (fxdraw.value),
    wrapped as the map tiles, convolved with their centers and values by
    FFT (each center spread bilinearly over its four nearest pixels).
    Unclamped (the map's store clamps).
    """
    if not all(tile) or any(
        len(s.value) != 1 or s.pattern < 0 or s.pattern in fxmap.OFFSET or s.blend_max
        for s in spl
    ):
        return None
    kinds: dict[tuple, list[Splat]] = {}
    for s in spl:
        kinds.setdefault((s.pattern, s.hx, s.hy, s.angle, s.suppl), []).append(s)
    if len(kinds) > BAKE_KINDS:
        return None
    n = 1 << size
    d = (np.arange(n) + 0.5) / n - 0.5  # pixel centers' offsets from a center
    dx, dy = np.meshgrid(d, d)
    total = np.zeros((n, n))
    for (pid, hx, hy, angle, suppl), members in kinds.items():
        if hx == 0 or hy == 0:
            continue
        bx, by = fxdraw.box(members[0])
        kx, ky = math.ceil(bx + 0.5), math.ceil(by + 0.5)
        ca, sa = math.cos(angle), math.sin(angle)
        kernel = np.zeros((n, n))
        for j in range(-ky, ky + 1):
            for i in range(-kx, kx + 1):
                ox, oy = dx + i, dy + j
                qx = (ca * ox + sa * oy) / hx
                qy = (-sa * ox + ca * oy) / hy
                kernel += fxdraw.value(pid, qx, qy, suppl)[1]
        # rolled: kernel[i] is the pattern (i + 1/2) / n from a center, and
        # points[k] a center at k / n
        kernel = np.roll(kernel, (-(n // 2), -(n // 2)), axis=(0, 1))
        points = np.zeros((n, n))
        for s in members:
            px, py = s.x * n, s.y * n
            x0, y0 = math.floor(px), math.floor(py)
            fx, fy = px - x0, py - y0
            for yy, wy in ((y0, 1 - fy), (y0 + 1, fy)):
                for xx, wx in ((x0, 1 - fx), (x0 + 1, fx)):
                    points[yy % n, xx % n] += s.value[0] * wx * wy
        total += np.fft.irfft2(np.fft.rfft2(points) * np.fft.rfft2(kernel), s=(n, n))
    return total.astype(np.float32)


def write_floats(path: Path, image: np.ndarray) -> None:
    """Write float32s as an RGBA8 PNG, one float a texel (bit for bit)."""
    rgba = np.ascontiguousarray(image, dtype="<f4").view(np.uint8)
    Image.fromarray(rgba.reshape(*image.shape, 4), "RGBA").save(path)


def evaluable(err: Exception) -> bool:
    """Whether an error from ``splats`` just means: keep the map live."""
    return isinstance(err, (FunctionError, KeyError, TooBig, IndexError))
