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

Texture layout, texel indices: splats from 0 (RECORD texels each: center x
and y, half size x and y, rotation in radians, pattern, supplement, blend
max as 0 or 1, then the value: gray's, or premultiplied-later rgba), then the
grid's cells (start and count in the index list, as uints), then the index
list (splat numbers, as uints).
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sbsar2ptex import fxmap
from sbsar2ptex.function import FunctionError
from sbsar2ptex.function import decode
from sbsar2ptex.function import evaluate
from sbsar2ptex.fxrng import FxRng
from sbsar2ptex.fxrng import depth_step
from sbsar2ptex.fxrng import iterate_step
from sbsar2ptex.rng import UINT32_MASK
from sbsar2ptex.sbsasm import Node

WIDTH = 1024
"""Texels per row of the data texture."""

MAX_SPLATS = 1_000_000
"""More than this and the map stays live (its texture would be too big)."""

CHILD = ((-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25))


class TooBig(Exception):  # noqa: N818
    """The tree draws more splats than MAX_SPLATS."""


@dataclass
class Splat:
    """One pattern as drawn: where, how big, how turned, which, how strong."""

    x: float
    y: float
    hx: float
    hy: float
    angle: float
    pattern: int
    """A pattern id, or -1 - an image input's index."""
    suppl: float
    blend_max: bool
    value: list[float]
    """Gray: [luminance times weight]; color: rgba, alpha times weight."""


def splats(
    node: Node,
    tree: fxmap.Tree | None,
    settings: fxmap.Settings,
    inputs: dict[int, list[float]],
    *,
    color: bool,
    size: int,
) -> list[Splat]:
    """Return the patterns the FX-Map ``node`` draws, in drawing order.

    ``inputs`` are the graph inputs' values by uid (defaults). Raises
    FunctionError for what cannot be evaluated here (sampling an input) and
    TooBig past MAX_SPLATS.
    """
    n = float(1 << size)
    deepest = fxmap.max_depth(tree)
    h = settings.hurst
    total = sum(h**j for j in range(deepest + 1)) if h not in (0, 1) else 1.0

    def weight(depth: float) -> float:
        if h == 0:
            return 1.0
        if h == 1:
            return 1.0 / (deepest + 1)
        return h**depth / total

    blocks: dict[int, Any] = {}

    def run(code: int, state: int, lane: str, ctx: dict[int, list[float]]) -> list:
        block = blocks.get(code)
        if block is None:
            block = blocks[code] = decode(node.at(code, (node.end - code) // 4))
        return evaluate(block, inputs, ctx, FxRng(state, lane).rand)

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
    out: list[Splat] = []

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
        inherited = not isinstance(t, (fxmap.Iterate, fxmap.Switch)) or t.inherited
        e = ((incoming if inherited else 0) + (t.seed or 0)) & UINT32_MASK
        ctx = {
            0: [0.0],
            1: [n, n],
            3: [float(size), float(size)],
            8: list(pos),
            10: [number],
            12: [float(depth)],
        }
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
                pattern = -1 - int(run(t.image_index.code, e, "imageindex", ctx)[0])
            elif t.image is not None and t.image < len(settings.patterns):
                pattern = -1 - t.image
            if pattern is not None:
                w = weight(depth) * settings.opacity
                value = [op[0], op[1], op[2], op[3] * w] if color else [op[1] * w]
                if any(value if not color else value[3:]):
                    out.append(
                        Splat(
                            at[0] + fo[0],
                            at[1] + fo[1],
                            cell * sz[0] / 2,
                            cell * sz[1] / 2,
                            rot[0] * 2 * math.pi,
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
            enter(t.first, e, pos, cell, depth, number)
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
    return out


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


def _reach(s: Splat) -> float:
    return math.hypot(s.hx, s.hy)


def pack(spl: list[Splat], *, color: bool, tile: tuple[bool, bool]) -> Packed:
    """Return the splats and a cell grid of them as a data texture.

    The grid (8 to 256 cells a side) is the finest whose index list stays
    within 16 entries a splat on average.
    """
    rec = RECORD[color]
    floats: list[float] = []
    for s in spl:
        floats += [s.x, s.y, s.hx, s.hy, s.angle, float(s.pattern), s.suppl]
        floats += [1.0 if s.blend_max else 0.0, *s.value]
    words = np.array(floats, dtype="<f4").view("<u4").tolist()
    if len(words) != rec * len(spl):
        msg = "splat records of the wrong size"
        raise ValueError(msg)

    def lists(g: int) -> list[list[int]]:
        cells: list[list[int]] = [[] for _ in range(g * g)]
        for k, s in enumerate(spl):
            r = _reach(s)
            spans = []
            for c, wrap in ((s.x, tile[0]), (s.y, tile[1])):
                lo, hi = math.floor((c - r) * g), math.floor((c + r) * g)
                if wrap:
                    spans.append(
                        range(g)
                        if hi - lo + 1 >= g
                        else [i % g for i in range(lo, hi + 1)]
                    )
                else:
                    spans.append(range(max(lo, 0), min(hi, g - 1) + 1))
            for cy in spans[1]:
                for cx in spans[0]:
                    cells[cy * g + cx].append(k)
        return cells

    grid, cells = 8, lists(8)
    g = 16
    while g <= 256:  # noqa: PLR2004
        finer = lists(g)
        if sum(map(len, finer)) > 16 * max(len(spl), 1) + g * g:
            break
        grid, cells = g, finer
        g *= 2
    start = len(words)
    index_at = start + 2 * grid * grid
    table, index = [], []
    for members in cells:
        table += [len(index), len(members)]
        index += members
    words += table + index
    rows = -(-len(words) // WIDTH)
    flat = np.zeros(rows * WIDTH, dtype="<u4")
    flat[: len(words)] = words
    return Packed(flat.view(np.uint8).reshape(rows, WIDTH, 4), grid, start, index_at)


def write_png(path: Path, packed: Packed) -> None:
    """Write the data texture as an 8-bit RGBA PNG."""
    Image.fromarray(packed.image, "RGBA").save(path)


def evaluable(err: Exception) -> bool:
    """Whether an error from ``splats`` just means: keep the map live."""
    return isinstance(err, (FunctionError, KeyError, TooBig, IndexError))
