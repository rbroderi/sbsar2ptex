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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

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


class RegionError(Exception):
    """A render or tiling region the splats cannot follow."""


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
    render, tiling = _regions(settings, lambda code: run(code, 0, "frameoffset", {}))
    if tiling != render:  # the image is not the tile (translate's to handle)
        msg = f"FX-Map tiling region {tiling} is not its render region {render}"
        raise RegionError(msg)
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
    if render != UNIT_REGION:  # the region drawn fills the image
        x0, x1, y0, y1 = render
        w, h = x1 - x0, y1 - y0
        if w != h and any(s.angle % math.pi for s in out):
            # stretching a turned pattern shears it: no splat can draw that
            msg = f"FX-Map render region {render} stretches turned patterns"
            raise RegionError(msg)
        for s in out:
            s.x, s.y = (s.x - x0) / w, (s.y - y0) / h
            s.hx, s.hy = s.hx / w, s.hy / h
    return out


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


def _smooth(e: float, x: np.ndarray) -> np.ndarray:
    t = np.clip(x / max(e, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def pattern(pid: int, qx: np.ndarray, qy: np.ndarray, s: float) -> np.ndarray:
    """Return pattern ``pid`` at (qx, qy), supplement ``s`` (fxmap.PATTERN_GLSL)."""
    r2 = qx * qx + qy * qy
    r = np.sqrt(r2)
    bell = np.maximum(1.0 - r2, 0.0) ** 4
    one = np.ones_like(qx)
    values = {
        0: lambda: one,
        1: lambda: (r <= 1.0) * one,
        2: lambda: np.maximum(1.0 - r2, 0.0),
        3: lambda: bell,
        4: lambda: np.exp(-11.1 * r2),
        5: lambda: np.maximum(1.0 - r, 0.0) ** 4,
        6: lambda: np.maximum(1.0 - np.maximum(abs(qx), abs(qy)), 0.0),
        7: lambda: _smooth(s, 1.0 - abs(qx)) * _smooth(s, 1.0 - abs(qy)),
        8: lambda: (qx + 1.0) * 0.5 * (1.0 - s) * one,
        9: lambda: (
            np.maximum(1.0 - r2, 0.0)
            * (0.5 + 0.5 * np.cos((3.0 + 12.0 * s) * math.pi * qx))
        ),
        10: lambda: np.where(qx > 0.0, bell * (1.0 - _smooth(1.0 - s, qx)), bell),
        11: lambda: bell * (1.0 - abs(qx)) ** 2,
        12: lambda: (
            np.where(r <= 1.0, 6.75 * r2 * (1.0 - r2) ** 2, 0.0)
            * (1.0 - _smooth(2.0 - 2.0 * s, qx + 1.0))
        ),
        13: lambda: np.maximum(
            1.0 - 20.0 * (np.maximum(abs(qx) - 0.77 * s, 0.0) ** 2 + qy * qy), 0.0
        ),
        14: lambda: np.maximum(1.0 - r, 0.0),
        15: lambda: (qx + 1.0) * 0.5 + s,
    }
    if pid in values:
        v = values[pid]()
    else:
        v = np.where(r <= 1.0, np.sqrt(np.maximum(1.0 - r2, 0.0)) + s, 0.0)
    return np.where((abs(qx) <= 1.0) & (abs(qy) <= 1.0), v, 0.0)


def _copies(s: Splat) -> int:
    """Return the tile copies a side the shader draws of ``s`` (FX_DRAW_GLSL)."""
    ext = math.hypot(max(abs(s.hx), 1e-8), max(abs(s.hy), 1e-8))
    return 2 * (math.ceil(ext - 0.5) if ext > 0.5 else 0) + 1  # noqa: PLR2004


def pixel_work(spl: list[Splat]) -> float:
    """Return about how many patterns the shader draws a pixel, on average."""
    return sum(min(1.0, (2 * _reach(s)) ** 2) * _copies(s) ** 2 for s in spl)


BAKE_KINDS = 16
"""Most kinds of splat (pattern, size, angle, supplement) a bake takes."""


def bake(spl: list[Splat], size: int, tile: tuple[bool, bool]) -> np.ndarray | None:
    """Return the gray map the splats draw, 2**size a side, or None.

    Only a tiling map of gray splats of patterns (not images) that add
    (not max), of few kinds: each kind's patterns are one kernel, wrapped
    as the map tiles, convolved with their centers and values by FFT (each
    center spread bilinearly over its four nearest pixels). Clamped.
    """
    if not all(tile) or any(
        len(s.value) != 1 or s.pattern < 0 or s.blend_max for s in spl
    ):
        return None
    kinds: dict[tuple, list[Splat]] = {}
    for s in spl:
        kinds.setdefault((s.pattern, s.hx, s.hy, s.angle, s.suppl), []).append(s)
    if len(kinds) > BAKE_KINDS:
        return None
    n = 1 << size
    d = np.arange(n) / n
    d = np.where(d >= 0.5, d - 1.0, d)  # noqa: PLR2004 (offsets, wrapped)
    dx, dy = np.meshgrid(d, d)
    total = np.zeros((n, n))
    for (pid, hx, hy, angle, suppl), members in kinds.items():
        hx_, hy_ = max(abs(hx), 1e-8), max(abs(hy), 1e-8)
        k = (_copies(members[0]) - 1) // 2 + 1
        ext = math.hypot(hx_, hy_)
        ca, sa = math.cos(angle), math.sin(angle)
        kernel = np.zeros((n, n))
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                if math.hypot(max(abs(i) - 0.5, 0), max(abs(j) - 0.5, 0)) > ext:
                    continue  # (that copy cannot reach the tile)
                ox, oy = dx + i, dy + j
                qx = (ca * ox + sa * oy) / hx_
                qy = (-sa * ox + ca * oy) / hy_
                kernel += pattern(pid, qx, qy, suppl)
        points = np.zeros((n, n))
        for s in members:
            px, py = s.x * n - 0.5, s.y * n - 0.5
            x0, y0 = math.floor(px), math.floor(py)
            fx, fy = px - x0, py - y0
            for yy, wy in ((y0, 1 - fy), (y0 + 1, fy)):
                for xx, wx in ((x0, 1 - fx), (x0 + 1, fx)):
                    points[yy % n, xx % n] += s.value[0] * wx * wy
        total += np.fft.irfft2(np.fft.rfft2(points) * np.fft.rfft2(kernel), s=(n, n))
    return np.clip(total, 0.0, 1.0).astype(np.float32)


def write_floats(path: Path, image: np.ndarray) -> None:
    """Write float32s as an RGBA8 PNG, one float a texel (bit for bit)."""
    rgba = np.ascontiguousarray(image, dtype="<f4").view(np.uint8)
    Image.fromarray(rgba.reshape(*image.shape, 4), "RGBA").save(path)


def evaluable(err: Exception) -> bool:
    """Whether an error from ``splats`` just means: keep the map live."""
    return isinstance(err, (FunctionError, KeyError, TooBig, IndexError))
