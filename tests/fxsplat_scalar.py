"""fxsplat's packing and raster as they were, a splat at a time: the reference.

test_lanes checks the vectorized versions against these, bit for bit.
"""

import math

import numpy as np

from sbsar2ptex import fxdraw
from sbsar2ptex.fxsplat import RECORD
from sbsar2ptex.fxsplat import WIDTH
from sbsar2ptex.fxsplat import Packed
from sbsar2ptex.fxsplat import Region
from sbsar2ptex.fxsplat import Splat

_TAU = np.float32(6.2831855)


def boxes(
    s: Splat, render: Region, tiling: Region, tile: tuple[bool, bool]
) -> list[tuple[float, float, float, float]]:
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
    rec = RECORD[color]
    floats: list[float] = []
    for s in spl:
        floats += [s.x, s.y, s.hx, s.hy, s.angle, float(s.pattern), s.suppl]
        floats += [1.0 if s.blend_max else 0.0, *s.value]
    words = np.array(floats, dtype="<f4").view("<u4").tolist()
    if len(words) != rec * len(spl):
        msg = "splat records of the wrong size"
        raise ValueError(msg)
    all_boxes = [boxes(s, render, tiling, tile) for s in spl]

    def lists(g: int) -> list[list[int]]:
        cells: list[set[int]] = [set() for _ in range(g * g)]
        for k, bs in enumerate(all_boxes):
            for x0, x1, y0, y1 in bs:
                cx0, cx1 = max(math.floor(x0 * g), 0), min(math.floor(x1 * g), g - 1)
                cy0, cy1 = max(math.floor(y0 * g), 0), min(math.floor(y1 * g), g - 1)
                for cy in range(cy0, cy1 + 1):
                    for cx in range(cx0, cx1 + 1):
                        cells[cy * g + cx].add(k)
        return [sorted(c) for c in cells]

    grid, cells = 8, lists(8)
    g = 16
    while g <= 256:
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
    rows = 1 << (-(-len(words) // WIDTH) - 1).bit_length()
    flat = np.zeros(rows * WIDTH, dtype="<u4")
    flat[: len(words)] = words
    return Packed(flat.view(np.uint8).reshape(rows, WIDTH, 4), grid, start, index_at)


def raster(
    spl: list[Splat],
    tile: tuple[bool, bool],
    tiling: Region = (0.0, 1.0, 0.0, 1.0),
    render: Region = (0.0, 1.0, 0.0, 1.0),
) -> list[float] | None:
    f = np.float32
    out: list[float] = []
    for s in spl:
        if s.hx == 0 or s.hy == 0:
            continue
        w, h = f(2 * s.hx), f(2 * s.hy)
        turns = f(f(s.angle) / _TAU)
        angle = f(turns * _TAU)
        c, n = f(abs(np.cos(angle))), f(abs(np.sin(angle)))
        bx = f(f(abs(w) * c) + f(abs(h) * n))
        by = f(f(abs(w) * n) + f(abs(h) * c))
        v = f(s.value[0])
        centers = copies32(s, tiling, tile, render)
        if centers is None:
            return None
        for x, y in centers:
            out += [v, v, v, 1.0, bx, by, x, y, w, h, turns, f(s.suppl)]
            out += [float(s.pattern), 1.0 if s.blend_max else 0.0, 0.0, 0.0]
    return [float(x) for x in out]


def copies32(
    s: Splat, tiling: Region, tile: tuple[bool, bool], render: Region
) -> list[tuple[np.float32, np.float32]] | None:
    f = np.float32
    a = f(s.angle)
    ca, sa = f(np.cos(a)), f(np.sin(a))
    hx, hy = f(s.hx), f(s.hy)
    b = (f(abs(f(hx * ca)) + abs(f(hy * sa))), f(abs(f(hx * sa)) + abs(f(hy * ca))))
    axes, unsure = [], []
    for c, bb, (lo, hi), (r0, r1), wrap in (
        (f(s.x), b[0], tiling[:2], render[:2], tile[0]),
        (f(s.y), b[1], tiling[2:], render[2:], tile[1]),
    ):
        w = f(hi - lo)
        ks = (
            range(math.floor((lo - bb - c) / w) - 1, math.ceil((hi + bb - c) / w) + 2)
            if wrap
            else (0,)
        )
        cs = [f(c + f(f(k) * w)) for k in ks]
        near = 4 * np.finfo(np.float32).eps * max(abs(lo), abs(hi), 1.0)
        axes.append([cc for cc in cs if f(cc - bb) < f(hi) and f(cc + bb) > f(lo)])
        unsure.append({
            cc for cc in cs
            if (abs(f(cc + bb) - lo) <= near or abs(f(cc - bb) - hi) <= near)
            and cc - bb < r1 - near and cc + bb > r0 + near
        })  # fmt: skip
    for k in (0, 1):
        other = set(axes[1 - k]) | unsure[1 - k]
        if unsure[k] and other:
            return None
    return [(x, y) for y in axes[1] for x in axes[0]]


def pixel_work(
    spl: list[Splat],
    render: Region = (0.0, 1.0, 0.0, 1.0),
    tiling: Region = (0.0, 1.0, 0.0, 1.0),
    tile: tuple[bool, bool] = (True, True),
) -> float:
    total = 0.0
    for s in spl:
        for x0, x1, y0, y1 in boxes(s, render, tiling, tile):
            w = min(x1, 1.0) - max(x0, 0.0)
            h = min(y1, 1.0) - max(y0, 0.0)
            total += max(w, 0.0) * max(h, 0.0)
    return total
