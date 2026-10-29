"""How Substance's GPU engine draws an FX-Map's patterns: a numpy model.

The tree walk (fxsplat.splats) gives each pattern drawn: its center, half
sizes, rotation, pattern, supplement, blending and color. This module draws
them as sbsrender's default (GPU) engine does, from a behavioral
specification of it; translate's FX-Map GLSL is the same drawing, and this
model is what tests and fxsplat.bake check it with.

- Order: by pattern (square first ... hemisphere, then images, grouped by
  input, the inputs in the reverse order of their first appearance), then
  blending (add, then max or alpha blending), then the tree's (``order``).
- Copies: wherever the map tiles, a pattern is drawn again at every whole
  tile offset whose copy's bounding box overlaps the tiling region, whole
  (``copies``).
- Coverage: no antialiasing; a pixel is drawn when its center is in the
  pattern's square or disc (fxmap.SQUARE), and not on its bounding box's
  left or top edge (the quad's corners are rounded to pixel boundaries,
  halves up). A pixel not drawn is left as it is.
- Value: the pattern's (``value``) times the instance color (gray, or
  premultiplied rgba), plus the supplement after for gradation offset and
  hemisphere. An image is read at its texels' centers, zero past its
  edges: nearest, bilinear with fractions rounded to 1/256, or bilinear in
  one level of a mip chain of stored 2x2 means (``mip_chain``, ``level``).
- Blending: into a float32 accumulator started at the background, never
  clamped: gray adds or takes the max; color adds or blends over (rgb), and
  always composites alpha over.
- Store: the accumulator clamped once, stored in the graph's format. Where
  GPUs differ (stores, mip means), ``amd`` takes the rules measured on an
  AMD GPU; the default are NVIDIA's predicted ones.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from sbsar2ptex import fxmap

F32 = np.float32


def f32(x: float) -> float:
    """Return ``x`` rounded to float32."""
    return float(np.float32(x))


def _ss(t: np.ndarray) -> np.ndarray:
    return t * t * (F32(3.0) - F32(2.0) * t)


def wave(f: np.ndarray) -> np.ndarray:
    """Return the waves pattern's polynomial cosine of ``f`` cycles."""
    f = np.asarray(f, dtype=F32)
    r = np.where(f >= 0, np.floor(f + F32(0.5)), np.ceil(f - F32(0.5))).astype(F32)
    e = F32(0.25) - np.abs(f - r)
    u = F32(1.0) - F32(4.0) * np.abs(e)
    w = F32(1.0) - u * u
    return np.sign(e) * w * (F32(0.77606332) + F32(0.22393668) * w)


def value(
    pid: int, qx: np.ndarray, qy: np.ndarray, s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return where built-in pattern ``pid`` covers (qx, qy), and its value.

    Supplement ``s``; float32 (fxmap.PATTERN_GLSL). Gradation offset and
    hemisphere's supplement is not added here (fxmap.OFFSET).
    """
    qx = np.asarray(qx, dtype=F32)
    qy = np.asarray(qy, dtype=F32)
    s = F32(s)
    one = F32(1.0)
    half = F32(0.5)
    r2 = qx * qx + qy * qy
    square = np.maximum(np.abs(qx), np.abs(qy))
    inside = square <= 1 if pid in fxmap.SQUARE else r2 <= 1
    m = one - r2
    m2 = m * m
    bell = m2 * m2
    ax, ay = np.abs(qx), np.abs(qy)
    with np.errstate(all="ignore"):
        if pid in (0, 1):
            v = np.ones_like(qx)
        elif pid == 2:  # noqa: PLR2004
            v = m
        elif pid == 3:  # noqa: PLR2004
            v = bell
        elif pid == 4:  # noqa: PLR2004
            v = np.clip(np.exp(F32(-11.1) * r2) - F32(1.51e-5), 0, 1).astype(F32)
        elif pid == 5:  # noqa: PLR2004
            t = one - np.sqrt(np.maximum(r2, 0))
            t = t * t
            v = t * t
        elif pid == 6:  # noqa: PLR2004
            v = one - np.maximum(ax, ay)
        elif pid == 7:  # noqa: PLR2004
            e = s + F32(1e-10)
            v = _ss(np.clip((one - ax) / e, 0, 1)) * _ss(np.clip((one - ay) / e, 0, 1))
        elif pid == 8:  # noqa: PLR2004
            v = half * (qx + one) * (one - s)
        elif pid == 9:  # noqa: PLR2004
            v = m * (half + half * wave((F32(6.0) * s + F32(1.5)) * qx))
        elif pid == 10:  # noqa: PLR2004
            v = bell * (one - _ss(np.clip(qx / (one - s), 0, 1)))
        elif pid == 11:  # noqa: PLR2004
            t = one - ax
            v = bell * (t * t)
        elif pid == 12:  # noqa: PLR2004
            t = np.clip(half * (qx + one) / (one - s), 0, 1)
            v = F32(6.75) * r2 * m2 * (one - _ss(t))
        elif pid == 13:  # noqa: PLR2004
            d = np.clip(ax - F32(0.77) * s, 0, 1)
            v = np.clip(one - F32(20.0) * (qy * qy + d * d), 0, 1)
        elif pid == 14:  # noqa: PLR2004
            v = one - np.sqrt(np.maximum(r2, 0))
        elif pid == 15:  # noqa: PLR2004
            v = half * (qx + one)
        else:
            v = np.sqrt(np.maximum(m, 0))
    return inside, np.where(inside, v, 0).astype(F32)


# Stores: a float32 in [0, 1] (clamped first) to an integer level.


def u16(v: np.ndarray) -> np.ndarray:
    """Return 16-bit levels of float32s, to the nearest (AMD)."""
    v = np.clip(np.asarray(v, dtype=np.float64), 0, 1)
    return np.floor(v * 65535 + 0.5).astype(np.int64)


def s16n(v: np.ndarray) -> np.ndarray:
    """Return 16-bit levels of float32s, truncated to 2^-20 first (NVIDIA)."""
    v = np.clip(np.asarray(v, dtype=np.float64), 0, 1)
    return np.floor(np.floor(v * 2**20) / 2**20 * 65535 + 0.5).astype(np.int64)


def s8(v: np.ndarray) -> np.ndarray:
    """Return 8-bit levels of float32s, through a half float's 11 bits (AMD)."""
    v = np.clip(np.asarray(v, dtype=F32), 0, 1)
    h = (v.view(np.uint32) & np.uint32(0xFFFFE000)).view(F32).astype(np.float64)
    return np.floor(h * 255 + 0.5).astype(np.int64)


def s8n(v: np.ndarray) -> np.ndarray:
    """Return 8-bit levels of float32s, truncated to 2^-12 first (NVIDIA)."""
    v = np.clip(np.asarray(v, dtype=np.float64), 0, 1)
    return np.floor((np.floor(v * 4096) * 255 + 2047) / 4096).astype(np.int64)


def store(v: np.ndarray, bits: int, *, amd: bool = False) -> np.ndarray:
    """Return ``v`` stored in ``bits`` (8 or 16) as a GPU does: integer levels."""
    if bits == 16:  # noqa: PLR2004
        return u16(v) if amd else s16n(v)
    return s8(v) if amd else s8n(v)


def _rounded_up(num: np.ndarray, den: int) -> np.ndarray:
    """Return whether float32(num / den) is above num / den (integers num)."""
    out = np.zeros(np.shape(num), dtype=bool)
    for k, n in np.ndenumerate(np.asarray(num, dtype=np.int64)):
        x = Fraction(int(n), den)
        if x <= 0:
            continue
        e = x.numerator.bit_length() - x.denominator.bit_length()
        if x < Fraction(2) ** e:
            e -= 1
        scaled = x * Fraction(2) ** (23 - e)  # in [2^23, 2^24)
        top = math.floor(scaled)
        rest = scaled - top
        out[k] = rest > Fraction(1, 2) or (rest == Fraction(1, 2) and top % 2 == 1)
    return out


def mip_mean(t: np.ndarray, bits: int, *, amd: bool = False) -> np.ndarray:
    """Return the next mip level of integer levels ``t``: stored 2x2 means.

    16-bit, AMD: the mean as a float32 value rounded to the nearest level (a
    half goes the way the float32 rounded); NVIDIA (predicted): halves up.
    8-bit: the mean value stored as ``s8`` (AMD), or as ``s8n`` with 0.03 of
    a 1/4096 step added before its truncation (NVIDIA).
    """
    t = np.asarray(t, dtype=np.int64)
    s = t[0::2, 0::2] + t[1::2, 0::2] + t[0::2, 1::2] + t[1::2, 1::2]
    if bits == 16:  # noqa: PLR2004
        q, r = s // 4, s % 4
        if not amd:
            return q + (r >= 2)  # noqa: PLR2004
        tie = r == 2  # noqa: PLR2004
        up = np.zeros_like(tie)
        if tie.any():
            # (q + 1/2) / 65535 = (2q + 1) / 131070: as float32(2q + 1 / 65535)
            up[tie] = _rounded_up(2 * q[tie] + 1, 65535)
        return q + (r > 2) + (tie & up)  # noqa: PLR2004
    if amd:
        out = np.zeros_like(s)
        for k, n in np.ndenumerate(s):  # n / 1020, truncated to 11 bits
            x = Fraction(int(n), 1020)
            if x == 0:
                continue
            e = x.numerator.bit_length() - x.denominator.bit_length()
            if x < Fraction(2) ** e:
                e -= 1
            h = Fraction(math.floor(x * Fraction(2) ** (10 - e))) * Fraction(2) ** (
                e - 10
            )
            out[k] = math.floor(h * 255 + Fraction(1, 2))
        return out
    # floor(s * 4096 / 1020 + 0.03), in integers (measured on an RTX 3090)
    return ((s * 409600 + 3060) // 102000 * 255 + 2047) // 4096


def mip_chain(t: np.ndarray, bits: int, *, amd: bool = False) -> list[np.ndarray]:
    """Return integer-level mip levels of square ``t``, from itself to 1x1."""
    out = [np.asarray(t, dtype=np.int64)]
    while out[-1].shape[0] > 1:
        out.append(mip_mean(out[-1], bits, amd=amd))
    return out


LOD_TIE = 1e-3
"""How near a half step of 1/256 a level's log2 counts as on it (in 1/256ths):
rounded down there, past float noise."""


def level(hx: float, hy: float, angle: float, m: int, px: float, py: float) -> int:
    """Return the mip level an image pattern reads (filtering 0).

    ``m`` is the image's size, ``px`` and ``py`` a pixel's step in pattern
    space. The longer of the screen derivatives of the texel coordinates;
    its log2 rounded to 1/256 (ties down), then to the nearest level.
    """
    if hx == 0 or hy == 0:
        return int(math.log2(m))
    c, s = math.cos(angle), math.sin(angle)
    gx = m * px / 2 * math.hypot(c / hx, s / hy)
    gy = m * py / 2 * math.hypot(s / hx, c / hy)
    lam = math.log2(max(gx, gy))
    lq = math.ceil(lam * 256 - 0.5 - LOD_TIE) / 256
    return min(max(math.floor(lq + 0.5), 0), int(math.log2(m)))


def frac256(f: np.ndarray) -> np.ndarray:
    """Return bilinear fractions rounded to 1/256, as a texture unit does."""
    return np.floor(f * 256 + 0.5) / 256


@dataclass
class Instance:
    """A pattern drawn: fxsplat.Splat's fields."""

    x: float
    y: float
    hx: float
    hy: float
    angle: float
    pattern: int
    suppl: float
    blend_max: bool
    value: list[float]


def order(instances: Sequence) -> list[int]:
    """Return the drawing order of ``instances`` (Splat-like), as indices.

    By pattern, images after all others (grouped by input, the inputs in the
    reverse order of their first appearance), then blending: add before
    max, then the tree's own order.
    """
    first: dict[int, int] = {}
    for s in instances:
        if s.pattern < 0:
            first.setdefault(fxmap.image_parts(s.pattern)[0], len(first))

    def key(k: int) -> tuple:
        s = instances[k]
        if s.pattern >= 0:
            group = s.pattern
        else:
            group = 1000 + len(first) - first[fxmap.image_parts(s.pattern)[0]]
        return (group, s.blend_max, k)

    return sorted(range(len(instances)), key=key)


def box(s: object) -> tuple[float, float]:
    """Return a turned pattern's bounding box's half extents."""
    c, n = abs(math.cos(s.angle)), abs(math.sin(s.angle))
    hx, hy = abs(s.hx), abs(s.hy)
    return hx * c + hy * n, hx * n + hy * c


def copies(
    s: object, tiling: Sequence[float], tile: tuple[bool, bool]
) -> list[tuple[float, float]]:
    """Return the tile offsets at which a pattern is drawn.

    Every whole tile offset (along the axes that tile) whose copy's
    bounding box overlaps the tiling region, x0 x1 y0 y1, strictly.
    """
    x0, x1, y0, y1 = tiling
    tw, th = x1 - x0, y1 - y0
    bx, by = box(s)
    axes = []
    for c, b, lo, hi, w, wrap in (
        (s.x, bx, x0, x1, tw, tile[0]),
        (s.y, by, y0, y1, th, tile[1]),
    ):
        ks = range(math.floor((lo - b - c) / w) - 1, math.ceil((hi + b - c) / w) + 2)
        offs = [k * w for k in (ks if wrap else (0,))]
        axes.append([o for o in offs if c + o - b < hi and c + o + b > lo])
    return [(ox, oy) for oy in axes[1] for ox in axes[0]]


def render(
    instances: Sequence,
    n: int,
    *,
    color: bool,
    background: Sequence[float] | np.ndarray | None = None,
    render_roi: Sequence[float] = (0.0, 1.0, 0.0, 1.0),
    tiling_roi: Sequence[float] = (0.0, 1.0, 0.0, 1.0),
    tile: tuple[bool, bool] = (True, True),
    images: Sequence[np.ndarray] = (),
    bits: int | None = 16,
    amd: bool = False,
) -> np.ndarray:
    """Return the FX-Map drawn: ``n`` pixels a side, stored levels or floats.

    ``instances`` are Splat-like (gray value [g]; color premultiplied
    rgba), in the tree's order; ``images`` the pattern inputs as integer
    levels in ``bits`` (n x n, or n x n x 4 in color). ``bits`` None returns
    the float32 accumulator, unclamped.
    """
    ch = 4 if color else 1
    acc = np.zeros((n, n, ch), dtype=F32)
    if background is not None:
        acc[...] = np.asarray(background, dtype=F32).reshape(-1)[:ch]
    rx0, rx1, ry0, ry1 = render_roi
    rw, rh = rx1 - rx0, ry1 - ry0
    centers = (np.arange(n) + 0.5) / n
    px_x = rx0 + centers * rw
    px_y = ry0 + centers * rh
    chains: dict[int, list[np.ndarray]] = {}
    full = (1 << 16) - 1 if bits == 16 else 255  # noqa: PLR2004
    for k in order(instances):
        s = instances[k]
        if s.hx == 0 or s.hy == 0:
            continue
        col = np.asarray(s.value, dtype=F32)
        ang = F32(s.angle)
        ca, sa = float(np.cos(ang)), float(np.sin(ang))
        bx, by = box(s)
        for ox, oy in copies(s, tiling_roi, tile):
            cx, cy = s.x + ox, s.y + oy
            # the quad's pixels: its corners rounded to pixel boundaries
            i0 = max(math.floor((cx - bx - rx0) / rw * n + 0.5), 0)
            i1 = min(math.floor((cx + bx - rx0) / rw * n + 0.5), n)
            j0 = max(math.floor((cy - by - ry0) / rh * n + 0.5), 0)
            j1 = min(math.floor((cy + by - ry0) / rh * n + 0.5), n)
            if i0 >= i1 or j0 >= j1:
                continue
            dx = px_x[None, i0:i1] - cx
            dy = px_y[j0:j1, None] - cy
            qx = (dx * ca + dy * sa) / s.hx
            qy = (-dx * sa + dy * ca) / s.hy
            if s.pattern >= 0:
                inside, v = value(s.pattern, qx, qy, s.suppl)
                src = col[None, None, :] * v[..., None]
                if s.pattern in fxmap.OFFSET:
                    src = src + F32(s.suppl)
            else:
                index, filtering, premul = fxmap.image_parts(s.pattern)
                inside = np.maximum(np.abs(qx), np.abs(qy)) <= 1
                img = images[index]
                if img.ndim == 2:  # noqa: PLR2004
                    img = img[..., None]
                if index not in chains:
                    per = [
                        mip_chain(img[..., c], bits or 16, amd=amd)
                        for c in range(img.shape[-1])
                    ]
                    chains[index] = [
                        np.stack([p[lv] for p in per], -1) for lv in range(len(per[0]))
                    ]
                lv = 0
                if filtering == fxmap.MIPMAPPED:
                    lv = level(s.hx, s.hy, s.angle, n, rw / n, rh / n)
                tex = chains[index][lv].astype(np.float64) / full
                m = tex.shape[0]
                u = (qx * 0.5 + 0.5).astype(np.float64)
                w = (qy * 0.5 + 0.5).astype(np.float64)

                def texel(ix: np.ndarray, iy: np.ndarray, tex=tex, m=m) -> np.ndarray:
                    ok = (ix >= 0) & (ix < m) & (iy >= 0) & (iy < m)
                    got = tex[np.clip(iy, 0, m - 1), np.clip(ix, 0, m - 1)]
                    return np.where(ok[..., None], got, 0.0)

                if filtering == fxmap.NEAREST:
                    c = texel(np.floor(u * m).astype(int), np.floor(w * m).astype(int))
                else:
                    tx, ty = u * m - 0.5, w * m - 0.5
                    x0, y0 = np.floor(tx), np.floor(ty)
                    fx = frac256(tx - x0)[..., None]
                    fy = frac256(ty - y0)[..., None]
                    x0, y0 = x0.astype(int), y0.astype(int)
                    c = (
                        (1 - fx) * (1 - fy) * texel(x0, y0)
                        + fx * (1 - fy) * texel(x0 + 1, y0)
                        + (1 - fx) * fy * texel(x0, y0 + 1)
                        + fx * fy * texel(x0 + 1, y0 + 1)
                    )
                c = c.astype(F32)
                if color:
                    if c.shape[-1] == 1:
                        c = np.concatenate([c, c, c, np.ones_like(c)], -1)
                    if not premul:
                        c = np.concatenate([c[..., :3] * c[..., 3:], c[..., 3:]], -1)
                else:
                    c = c[..., :1]
                src = col[None, None, :] * c
            src = src.astype(F32)
            d = acc[j0:j1, i0:i1]
            if color:
                a = src[..., 3:]
                if s.blend_max:
                    rgb = src[..., :3] + d[..., :3] * (F32(1) - a)
                else:
                    rgb = d[..., :3] + src[..., :3]
                new = np.concatenate([rgb, a + d[..., 3:] * (F32(1) - a)], -1)
            elif s.blend_max:
                new = np.maximum(d, src)
            else:
                new = d + src
            acc[j0:j1, i0:i1] = np.where(inside[..., None], new, d)
    if bits is None:
        return acc if color else acc[..., 0]
    out = store(acc, bits, amd=amd)
    return out if color else out[..., 0]
