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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sbsar2ptex import fxdraw
from sbsar2ptex import fxmap
from sbsar2ptex.function import FunctionError
from sbsar2ptex.function import decode
from sbsar2ptex.function import evaluate
from sbsar2ptex.function import glsl_type
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
) -> list[Splat]:
    """Return the patterns the FX-Map ``node`` draws, in drawing order.

    In pattern space (the render region maps it to the image). ``inputs``
    are the graph inputs' values by uid (defaults). Raises FunctionError for
    what cannot be evaluated here (sampling an input) and TooBig past
    MAX_SPLATS.
    """
    n = float(1 << size)
    weights = depth_weights(settings.hurst, fxmap.max_depth(tree))

    blocks: dict[int, Any] = {}
    # One variable store for the whole map: its setup sets variables the
    # tree's functions read, and they keep state from one call to the next
    # (a walk over a grid, pass by pass), in Substance's order of calls.
    variables: dict[int, list] = {}

    def run(code: int, state: int, lane: str, ctx: dict[int, list[float]]) -> list:
        block = blocks.get(code)
        if block is None:
            block = blocks[code] = decode(node.at(code, (node.end - code) // 4))
        return evaluate(block, inputs, ctx, FxRng(state, lane).rand, variables)

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
    return [out[k] for k in fxdraw.order(out)]


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
    node: Node, settings: fxmap.Settings, inputs: dict[int, list[float]], size: int
) -> tuple[list[float], list[float]]:
    """Return ``_regions`` for the FX-Map ``node``, its setup run first."""
    variables: dict[int, list] = {}
    n = float(1 << size)

    def run(code: int, ctx: dict[int, list[float]]) -> list:
        block = decode(node.at(code, (node.end - code) // 4))
        return evaluate(block, inputs, ctx, FxRng(0, "frameoffset").rand, variables)

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


def _boxes(
    s: Splat, render: Region, tiling: Region, tile: tuple[bool, bool]
) -> list[tuple[float, float, float, float]]:
    """Return the image-space boxes (x0 x1 y0 y1, uv) of a splat's copies."""
    bx, by = fxdraw.box(s)
    rx0, rx1, ry0, ry1 = render
    rw, rh = rx1 - rx0, ry1 - ry0
    return [
        (
            (s.x + ox - bx - rx0) / rw,
            (s.x + ox + bx - rx0) / rw,
            (s.y + oy - by - ry0) / rh,
            (s.y + oy + by - ry0) / rh,
        )
        for ox, oy in fxdraw.copies(s, tiling, tile)
    ]


def pack(
    spl: list[Splat],
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
    floats: list[float] = []
    for s in spl:
        floats += [s.x, s.y, s.hx, s.hy, s.angle, float(s.pattern), s.suppl]
        floats += [1.0 if s.blend_max else 0.0, *s.value]
    words = np.array(floats, dtype="<f4").view("<u4").tolist()
    if len(words) != rec * len(spl):
        msg = "splat records of the wrong size"
        raise ValueError(msg)
    boxes = [_boxes(s, render, tiling, tile) for s in spl]

    def lists(g: int) -> list[list[int]]:
        cells: list[set[int]] = [set() for _ in range(g * g)]
        for k, bs in enumerate(boxes):
            for x0, x1, y0, y1 in bs:
                cx0, cx1 = max(math.floor(x0 * g), 0), min(math.floor(x1 * g), g - 1)
                cy0, cy1 = max(math.floor(y0 * g), 0), min(math.floor(y1 * g), g - 1)
                for cy in range(cy0, cy1 + 1):
                    for cx in range(cx0, cx1 + 1):
                        cells[cy * g + cx].add(k)
        return [sorted(c) for c in cells]

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
    # A power of two of rows: texel centers then sit exactly where the shader
    # samples. At 4187 or 5251 rows, NVIDIA's sampler lands a sliver off a
    # center and blends in its neighbor, garbling the word.
    rows = 1 << (-(-len(words) // WIDTH) - 1).bit_length()
    flat = np.zeros(rows * WIDTH, dtype="<u4")
    flat[: len(words)] = words
    return Packed(flat.view(np.uint8).reshape(rows, WIDTH, 4), grid, start, index_at)


def write_png(path: Path, packed: Packed) -> None:
    """Write the data texture as an 8-bit RGBA PNG."""
    Image.fromarray(packed.image, "RGBA").save(path)


def pixel_work(
    spl: list[Splat],
    render: Region = (0.0, 1.0, 0.0, 1.0),
    tiling: Region = (0.0, 1.0, 0.0, 1.0),
    tile: tuple[bool, bool] = (True, True),
) -> float:
    """Return about how many patterns the shader draws a pixel, on average."""
    total = 0.0
    for s in spl:
        for x0, x1, y0, y1 in _boxes(s, render, tiling, tile):
            w = min(x1, 1.0) - max(x0, 0.0)
            h = min(y1, 1.0) - max(y0, 0.0)
            total += max(w, 0.0) * max(h, 0.0)
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
