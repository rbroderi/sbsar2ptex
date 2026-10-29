"""Reference model of Substance's directional motion blur (``dirmotionblur``).

    uv run python scripts/dmb_reference.py REFERENCE_DIR [--size 256|1024]

A float64 model of the GPU engine's blur as written in its behavioral
specification (Substance's own code is not used): the box of half-width R
steps along the angle, through bilinear reads whose fractions are rounded
to 1/256 (the texture filter). Up to R = 1.5 three reads; past it the
image is sheared into a frame whose columns follow the blur, boxed along
them (with the frame's own seam rule and, at 0 and 90 degrees, end taps
merged into one read) and resampled back.

With a directory of reference renders (``<input>_s<L>_b16_<A>_<I>.png``
next to ``grid256.png``, ``tex256.png``, ...), compares the model with
each and prints the worst error and the pixels more than 1/65535 off.

In 8-bit graphs (``blur8``, ``--eight``: the ``noise8_<N>`` renders) every
intermediate is stored in 8 bits through a half-float truncation, boxes
add aligned blocks of a pyramid of stored pair means, and the rotated box
reads only the stored frame and adds 1/512; it prints the identical pixels.

In 16-bit graphs (``blur16``) the passes are the 8-bit graph's, every
intermediate stored as an integer level: the pyramid's mean of two levels
is a half when their sum is odd, and NVIDIA GPUs send every half up; a
computed value is stored truncated to 2^-20 first on NVIDIA (s16n). Terms
and sums are float32 in the engine's order, the rotated box adds 0.4 of a
level. ``--sixteen`` compares it with the specification's T2 renders,
``--t0`` runs the tie-direction probes, ``--p2`` the small-path store
probe, ``--t1`` the cells of a folder of long boxes at 0 degrees.
``--amd`` takes AMD's rules (a half rounds as its float32 does, a value is
stored to the nearest level) here as for 8-bit stores: the renders in
``r/`` were made on AMD; NVIDIA's are the default.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

ANGLES = (0.0, 0.0625, 0.125, 0.1875, 0.25, 0.4375, 0.1, -0.125)
INTENSITIES = (0.3, 0.7326, 1.5, 2.93, 5.86, 12.0)


def q(u: np.ndarray | float) -> np.ndarray | float:
    """Return a bilinear fraction (in [0, 1)) rounded to 1/256, halves up."""
    return np.floor(np.asarray(u) * 256.0 + 0.5) / 256.0


def setup(intensity: float, angle: float, n_px: int) -> tuple[bool, float, float]:
    """Return (y-major, slope t, radius R in steps) for the blur's settings."""
    a = 2 * math.pi * angle
    c, s = math.cos(a), math.sin(a)
    if abs(abs(c) - abs(s)) < 1e-6:  # a diagonal: the GPU's rounding decides
        ymajor = round(abs(angle) * 8) % 4 == 1
        t = math.copysign(1.0, c * s)
    else:
        ymajor = abs(s) > abs(c)
        t = c / s if ymajor else s / c
        if abs(t) < 1e-6:
            t = 0.0
    radius = 0.5 + (intensity * n_px / 256 - 0.5) * max(abs(c), abs(s))
    return ymajor, t, radius


def read_minor(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return ``img`` at integer columns ``x`` and fractional rows ``y``, wrapping."""
    n = img.shape[0]
    y0 = np.floor(y)
    fy = q(y - y0)
    xi = x.astype(np.int64) % n
    yi = y0.astype(np.int64)
    return (1 - fy) * img[yi % n, xi] + fy * img[(yi + 1) % n, xi]


def blur_xmajor(img: np.ndarray, t: float, radius: float) -> np.ndarray:
    """Return the x-major blur of ``img`` (rows y, columns x), slope ``t``."""
    n_px = img.shape[0]
    if radius <= 0.5:
        return img.copy()
    n = math.ceil(radius - 0.5)
    f = radius + 0.5 - n
    y, x = np.mgrid[0:n_px, 0:n_px].astype(np.float64)
    if n == 1:  # the small path: three reads
        out = (
            f * read_minor(img, x - 1, y - t)
            + read_minor(img, x, y)
            + f * read_minor(img, x + 1, y + t)
        )
        return out / (2 * radius)
    x0 = (n_px - 1) / 2
    # The frame: frame[c, r] = in(r, c + (r - X0) t), column c along the blur.
    cc, rr = y, x
    frame = read_minor(img, rr, cc + (rr - x0) * t)
    cols = np.arange(n_px, dtype=np.float64)[:, None]

    def frame_at(r: np.ndarray) -> np.ndarray:
        """Return frame rows ``r`` (any integers) for every column, seam rule."""
        k = np.floor(r / n_px)
        rm = (r - k * n_px).astype(np.int64)
        cf = cols + k * n_px * t
        c0 = np.floor(cf)
        phi = q(cf - c0)
        c0i = c0.astype(np.int64)
        return (1 - phi) * frame[c0i % n_px, rm] + phi * frame[(c0i + 1) % n_px, rm]

    xs = np.arange(n_px, dtype=np.float64)[None, :]
    weights = {k: (f if abs(k) == n else 1.0) for k in range(-n, n + 1)}
    box = np.zeros_like(img)
    for k, w in weights.items():
        box += w * frame_at(np.broadcast_to(xs + k, (n_px, n_px)))
    if t == 0:  # end taps merged into one bilinear read with their neighbors
        e = (1 + f) * (q(f / (1 + f)) - f / (1 + f))
        xi = np.arange(n_px)
        first = xi - (n - 1)
        plus = np.ones(n_px, bool) if n == 2 else first % 2 == 0
        minus = np.ones(n_px, bool) if n == 2 else first % 2 == 1
        for side, on in ((n, plus), (-n, minus)):
            inner = side - int(math.copysign(1, side))
            delta = e * (frame_at(xs + side) - frame_at(xs + inner))
            box += np.where(on[None, :], delta, 0.0)
    # Back to the image: the frame column through each pixel, linearly.
    chat = y - (x - x0) * t
    c0 = np.floor(chat)
    g = q(chat - c0)
    c0i = c0.astype(np.int64)
    xi = x.astype(np.int64)
    out = (1 - g) * box[c0i % n_px, xi] + g * box[(c0i + 1) % n_px, xi]
    return out / (2 * radius)


def blur(img: np.ndarray, intensity: float, angle: float) -> np.ndarray:
    """Return the blurred ``img`` (values in [0, 1]), unrounded."""
    ymajor, t, radius = setup(intensity, angle, img.shape[0])
    if ymajor:
        return blur_xmajor(img.T, t, radius).T
    return blur_xmajor(img, t, radius)


def single_pass_xmajor(img: np.ndarray, t: float, radius: float) -> np.ndarray:
    """Return blur_xmajor computed as the translator's one shader computes it.

    Per pixel: the two frame columns either side, each boxed over its
    taps, a tap's two frame values from three texels (four past the seam),
    then mixed. The same arithmetic as the frame, box and back-resample.
    """
    n_px = img.shape[0]
    if radius <= 0.5 or math.ceil(radius - 0.5) == 1 or t == 0:
        return blur_xmajor(img, t, radius)  # the same reads either way
    n = math.ceil(radius - 0.5)
    f = radius + 0.5 - n
    x0 = (n_px - 1) / 2
    y, x = np.mgrid[0:n_px, 0:n_px].astype(np.float64)

    def p(i: np.ndarray, j: np.ndarray) -> np.ndarray:
        return img[j.astype(np.int64) % n_px, i.astype(np.int64) % n_px]

    chat = y - (x - x0) * t
    c0 = np.floor(chat)
    acc, acc1 = np.zeros_like(img), np.zeros_like(img)
    for k in range(-n, n + 1):
        w = f if abs(k) == n else 1.0
        r = x + k
        kk = np.floor(r / n_px)
        rm = r - kk * n_px
        cf = c0 + kk * n_px * t
        ca = np.floor(cf)
        phi = q(cf - ca)
        yy = ca + (rm - x0) * t
        y0 = np.floor(yy)
        wy = q(yy - y0)
        t0, t1, t2, t3 = (p(rm, y0 + j) for j in range(4))
        f0 = (1 - wy) * t0 + wy * t1
        f1 = (1 - wy) * t1 + wy * t2
        f2 = (1 - wy) * t2 + wy * t3
        acc += w * ((1 - phi) * f0 + phi * f1)  # phi is 0 inside the image
        acc1 += w * ((1 - phi) * f1 + phi * f2)
    g = q(chat - c0)
    return ((1 - g) * acc + g * acc1) / (2 * radius)


def blur_single_pass(img: np.ndarray, intensity: float, angle: float) -> np.ndarray:
    """Return ``blur`` computed as the translator's one shader computes it."""
    ymajor, t, radius = setup(intensity, angle, img.shape[0])
    if ymajor:
        return single_pass_xmajor(img.T, t, radius).T
    return single_pass_xmajor(img, t, radius)


F32 = np.float32


def s8(v: np.ndarray) -> np.ndarray:
    """Return Substance's 8-bit store of ``v``: half-float truncation, then 1/255.

    The value is clamped, truncated toward zero to 11 significant bits (a
    float32 with its low 13 mantissa bits cleared), rounded to the nearest
    255th (halves up) and read back as k / 255 in float32.
    """
    v = np.clip(np.asarray(v, F32), F32(0), F32(1)).astype(F32)
    h = (v.view(np.uint32) & np.uint32(0xFFFFE000)).view(F32)
    k = np.floor(h.astype(np.float64) * 255 + 0.5)
    return (k / 255).astype(F32)


def s8n(v: np.ndarray, c: float) -> np.ndarray:
    """Return the 8-bit store on NVIDIA GPUs: 1/4096 truncation, 1/255 ties down.

    ``c`` (in 4096ths) is 0 for sums the shader computes, 0.03 for values
    straight from a bilinear read, which the texture filter returns a hair
    high.
    """
    v = np.clip(np.asarray(v, F32), F32(0), F32(1)).astype(np.float64)
    x = np.floor(v * 4096 + c)
    return (np.floor((x * 255 + 2047) / 4096) / 255).astype(F32)


NVIDIA = [True]
"""Whether the 8-bit model stores as NVIDIA GPUs do (s8n), else as AMD's (s8)."""
SUM, READ = 0.0, 0.03
"""s8n's offsets: a computed sum; a bilinear read's value."""


def store(v: np.ndarray, c: float) -> np.ndarray:
    """Return the 8-bit store of ``v``, a value from source ``c`` (SUM, READ)."""
    return s8n(v, c) if NVIDIA[0] else s8(v)


def read32(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return read_minor in float32 (the 8-bit graph's arithmetic)."""
    n = img.shape[0]
    y0 = np.floor(y)
    fy = q(y - y0).astype(F32)
    xi = x.astype(np.int64) % n
    yi = y0.astype(np.int64)
    return (F32(1) - fy) * img[yi % n, xi] + fy * img[(yi + 1) % n, xi]


def pyramid(rows: np.ndarray) -> list[np.ndarray]:
    """Return the stored pair-mean levels of ``rows`` along axis 1 (level 0 first).

    Each level is made from the previous one as stored, not from level 0.
    """
    levels = [rows]
    while levels[-1].shape[1] > 1:
        a = levels[-1]
        levels.append(store((a[:, 0::2] + a[:, 1::2]) / F32(2), READ))
    return levels


def blocks(start: int, end: int, n_px: int) -> tuple[list[tuple[int, int]], int]:
    """Return the aligned power-of-two blocks (start, level) covering [start, end).

    Greedy: each the largest that fits and whose size divides its start
    (modulo the period); blocks of one are left over. Also the start left.
    """
    out = []
    while end - start >= 2:
        s = start % n_px
        level = min(
            (end - start).bit_length() - 1, (s & -s).bit_length() - 1 if s else 99
        )
        if level < 1:
            break
        out.append((start, level))
        start += 1 << level
    return out, start


def axis8(img: np.ndarray, radius: float) -> np.ndarray:
    """Return the 8-bit graph's x-major blur at t = 0: pyramid blocks, merged ends."""
    n_px = img.shape[0]
    n = math.ceil(radius - 0.5)
    f = radius + 0.5 - n
    qf, opf, ff = F32(q(f / (1 + f))), F32(1 + f), F32(f)
    levels = pyramid(img)

    def col(j: int) -> np.ndarray:
        return img[:, j % n_px]

    def merged(whole: int, part: int) -> np.ndarray:
        return opf * ((F32(1) - qf) * col(whole) + qf * col(part))

    out = np.empty_like(img)
    for x in range(n_px):
        first, end = x - (n - 1), x + n
        if n == 2:
            acc = (
                merged(first, first - 1) + col(first + 1) + merged(first + 2, first + 3)
            )
        else:
            if first % 2:
                acc, start = merged(first, first - 1), first + 1
            else:
                acc, start = ff * col(first - 1), first
            found, start = blocks(start, end, n_px)
            for s0, level in found:
                acc = acc + F32(1 << level) * levels[level][:, (s0 % n_px) >> level]
            if end - start == 1:
                acc = acc + merged(start, start + 1)
            else:
                acc = acc + ff * col(start)
        out[:, x] = store(acc / F32(2 * radius), SUM)
    return out


def rotated8(img: np.ndarray, t: float, radius: float) -> np.ndarray:
    """Return the 8-bit graph's x-major blur at t != 0: stored frame, pyramid, box.

    The box reads only the stored frame and its levels, merges no ends and
    adds 1/512 before its store; the back-resample is stored too.
    """
    n_px = img.shape[0]
    n = math.ceil(radius - 0.5)
    f = F32(radius + 0.5 - n)
    x0 = (n_px - 1) / 2
    y, x = np.mgrid[0:n_px, 0:n_px].astype(np.float64)
    frame = store(read32(img, x, y + (x - x0) * t), READ)  # frame[c, r]
    levels = pyramid(frame)
    cvec = np.arange(n_px, dtype=np.float64)

    def lev(level: int, s0: int) -> np.ndarray:
        """Return every column's level-``level`` block at absolute row ``s0``."""
        k = s0 // n_px
        i = (s0 - k * n_px) >> level
        if k == 0:
            return levels[level][:, i]
        cf = cvec + k * n_px * t
        ca = np.floor(cf)
        phi = q(cf - ca).astype(F32)
        cai = ca.astype(np.int64)
        lv = levels[level]
        return (F32(1) - phi) * lv[cai % n_px, i] + phi * lv[(cai + 1) % n_px, i]

    box = np.empty_like(frame)
    for r in range(n_px):
        first, end = r - (n - 1), r + n
        acc = f * lev(0, first - 1)
        start = first
        if first % 2:
            acc = acc + lev(0, first)
            start += 1
        found, start = blocks(start, end, n_px)
        for s0, level in found:
            acc = acc + F32(1 << level) * lev(level, s0)
        if end - start == 1:
            acc = acc + lev(0, start)
            start += 1
        acc = acc + f * lev(0, start)
        box[:, r] = store(acc / F32(2 * radius) + F32(1 / 512), SUM)
    chat = y - (x - x0) * t
    c0 = np.floor(chat)
    g = q(chat - c0).astype(F32)
    c0i, xi = c0.astype(np.int64), x.astype(np.int64)
    return store(
        (F32(1) - g) * box[c0i % n_px, xi] + g * box[(c0i + 1) % n_px, xi], READ
    )


def blur8(img: np.ndarray, intensity: float, angle: float) -> np.ndarray:
    """Return the blur in an 8-bit graph: ``img`` holds k / 255; so does the result."""
    ymajor, t, radius = setup(intensity, angle, img.shape[0])
    img = img.astype(F32)
    if ymajor:
        img = img.T.copy()
    if radius <= 0.5:
        out = img
    elif math.ceil(radius - 0.5) == 1:
        f = radius - 0.5
        y, x = np.mgrid[0 : img.shape[0], 0 : img.shape[0]].astype(np.float64)
        m, p = read32(img, x - 1, y - t), read32(img, x + 1, y + t)
        ws, wc = F32(f / (2 * radius)), F32(1 / (2 * radius))
        out = store((m * ws + p * ws) + img * wc, SUM)
    elif t == 0:
        out = axis8(img, radius)
    else:
        out = rotated8(img, t, radius)
    return out.T if ymajor else out


LEVELS = 65535
BETA16 = F32(0.4 / 65535)
"""The rotated box's bias in a 16-bit graph: 0.4 of a level."""
HALVES: list[str | None] = [None]
"""Force every pyramid half "up" or "down" (the T0 probe); None: the vendor's rule."""


def ratio(num: np.ndarray, den: int) -> np.ndarray:
    """Return num / den rounded once to float32 (exact float64 quotient first)."""
    return (np.asarray(num, np.float64) / den).astype(F32)


def u16(v: np.ndarray) -> np.ndarray:
    """Return the 16-bit store of a computed float32 ``v``: the nearest level, halves up."""
    v = np.clip(np.asarray(v, F32), F32(0), F32(1)).astype(np.float64)
    return np.floor(v * LEVELS + 0.5).astype(np.int64)


def s16n(v: np.ndarray) -> np.ndarray:
    """Return NVIDIA's 16-bit store of a float32 ``v``: truncated to 2^-20, then rounded."""
    v = np.clip(np.asarray(v, F32), F32(0), F32(1)).astype(np.float64)
    x = np.floor(v * 2**20).astype(np.int64)
    return (x * LEVELS + 2**19) >> 20


def store16(v: np.ndarray) -> np.ndarray:
    """Return the 16-bit store of a computed float32 ``v``, as the vendor makes it."""
    return s16n(v) if NVIDIA[0] else u16(v)


def u16_read(k0: np.ndarray, k1: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Return the stored level of a bilinear read of levels k0, k1 at weight j / 256.

    The filter is exact and its result rounded to float32, which is stored
    (store16; NVIDIA's store of a read is assumed, not measured).
    """
    return store16(ratio((256 - j) * k0 + j * k1, 256 * LEVELS))


def read16(k0: np.ndarray, k1: np.ndarray, j: np.ndarray | int) -> np.ndarray:
    """Return a bilinear read of levels k0, k1 at weight j / 256, as float32."""
    return ratio((256 - j) * k0 + j * k1, 256 * LEVELS)


def pairmean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the stored mean of two stored levels: a half rounds by the vendor's rule.

    NVIDIA sends every half up; AMD stores the float32 of (a + b) / 131070
    exactly, so a half goes up when that float is not below it.
    """
    s = a + b
    if HALVES[0] is not None:
        return (s + (HALVES[0] == "up")) >> 1
    if NVIDIA[0]:
        return (s + 1) >> 1
    return u16(ratio(s, 2 * LEVELS))


def pyramid16(rows: np.ndarray) -> list[np.ndarray]:
    """Return the 16-bit stored pair-mean levels of ``rows`` (integers) along axis 1."""
    levels = [rows]
    while levels[-1].shape[1] > 1:
        a = levels[-1]
        levels.append(pairmean(a[:, 0::2], a[:, 1::2]))
    return levels


def setup16(intensity: float, angle: float, n_px: int) -> tuple:
    """Return (y-major, t, R, n, f, 1 / 2R) as a 16-bit graph's engine has them.

    R, f and 1 / 2R are float32 from float32 steps; max(|cos|, |sin|) is 1
    exactly only at angle 0, a float32 ulp below it at other quarter turns.
    """
    ymajor, t, _ = setup(intensity, angle, n_px)
    lpx = F32(F32(intensity) * F32(n_px)) / F32(256)
    if angle == 0:
        mx = F32(1)
    elif angle * 4 == round(angle * 4):
        mx = F32(1 - 2**-24)
    else:
        a = 2 * math.pi * angle
        mx = F32(max(abs(math.cos(a)), abs(math.sin(a))))
    radius = F32(0.5) + F32(lpx - F32(0.5)) * mx
    n = math.ceil(float(radius) - 0.5)
    f = radius + F32(0.5) - F32(n)
    inv = F32(1) / (F32(2) * radius)
    return ymajor, t, radius, n, f, inv


def total(terms: list[np.ndarray], ymajor: bool) -> np.ndarray:
    """Return the float32 sum of the box's terms (listed from its - end).

    x-major sums from the + end, y-major from the - end.
    """
    order = terms if ymajor else terms[::-1]
    acc = order[0]
    for term in order[1:]:
        acc = (acc + term).astype(F32)
    return acc


def small16(k: np.ndarray, t: float, f: np.float32, inv: np.float32) -> np.ndarray:
    """Return the 16-bit graph's small path (x-major): three reads, from the + end."""
    n_px = k.shape[0]
    y, x = np.mgrid[0:n_px, 0:n_px]

    def read(dx: int, dy: float) -> np.ndarray:
        yy = y + dy
        y0 = np.floor(yy)
        j = np.round(q(yy - y0) * 256).astype(np.int64)
        y0i = y0.astype(np.int64)
        xi = (x + dx) % n_px
        return read16(k[y0i % n_px, xi], k[(y0i + 1) % n_px, xi], j)

    ws = f * inv
    p, c, m = read(1, t), ratio(k, LEVELS), read(-1, -t)
    return store16(
        ((p * ws).astype(F32) + (c * inv).astype(F32)) + (m * ws).astype(F32)
    )


def axis16(k: np.ndarray, n: int, f: np.float32, inv: np.float32, ymajor: bool):
    """Return the 16-bit graph's x-major blur at t = 0, in levels.

    The 8-bit graph's decomposition (axis8) over a pyramid of integer pair
    means; each term a float32, summed in the major axis's order.
    """
    n_px = k.shape[1]
    j = int(np.floor(float(f) / (1 + float(f)) * 256 + 0.5))
    w = f + F32(1)
    levels = pyramid16(k)

    def tex(i: int) -> np.ndarray:
        return ratio(k[:, i % n_px], LEVELS)

    def merged(whole: int, part: int) -> np.ndarray:
        return read16(k[:, whole % n_px], k[:, part % n_px], j) * w

    out = np.empty_like(k)
    for x in range(n_px):
        first, end = x - (n - 1), x + n
        if n == 2:
            terms = [merged(first, first - 1), tex(first + 1)]
            terms.append(merged(first + 2, first + 3))
            acc = total(terms, ymajor=True)  # this order on both axes
        else:
            if first % 2:
                terms, start = [merged(first, first - 1)], first + 1
            else:
                terms, start = [f * tex(first - 1)], first
            found, start = blocks(start, end, n_px)
            for s0, level in found:
                lv = levels[level][:, (s0 % n_px) >> level]
                terms.append(ratio(lv, LEVELS) * F32(1 << level))
            if end - start == 1:
                terms.append(merged(start, start + 1))
            else:
                terms.append(f * tex(start))
            acc = total(terms, ymajor)
        out[:, x] = store16(acc * inv)
    return out


def rotated16(k: np.ndarray, t: float, n: int, f, inv, ymajor: bool) -> np.ndarray:
    """Return the 16-bit graph's x-major blur at t != 0, in levels.

    rotated8's passes: the frame and the back-resample stored bilinear
    reads, the pyramid integer pair means, the box's terms float32 (past
    the seam a read between two columns' stored levels), stored with 0.4 of
    a level added.
    """
    n_px = k.shape[0]
    x0 = (n_px - 1) / 2
    y, x = np.mgrid[0:n_px, 0:n_px].astype(np.float64)
    yy = y + (x - x0) * t  # frame[c, r]: c rows, r columns
    y0 = np.floor(yy)
    j = np.round(q(yy - y0) * 256).astype(np.int64)
    y0i, xi = y0.astype(np.int64), x.astype(np.int64)
    frame = u16_read(k[y0i % n_px, xi], k[(y0i + 1) % n_px, xi], j)
    levels = pyramid16(frame)
    cvec = np.arange(n_px, dtype=np.float64)

    def lev(level: int, s0: int) -> np.ndarray:
        kk = s0 // n_px
        i = (s0 - kk * n_px) >> level
        lv = levels[level]
        if kk == 0:
            return ratio(lv[:, i], LEVELS)
        cf = cvec + kk * n_px * t
        ca = np.floor(cf)
        jj = np.round(q(cf - ca) * 256).astype(np.int64)
        cai = ca.astype(np.int64)
        return read16(lv[cai % n_px, i], lv[(cai + 1) % n_px, i], jj)

    box = np.empty_like(frame)
    for r in range(n_px):
        first, end = r - (n - 1), r + n
        terms = [f * lev(0, first - 1)]
        start = first
        if first % 2:
            terms.append(lev(0, first))
            start += 1
        found, start = blocks(start, end, n_px)
        for s0, level in found:
            terms.append(lev(level, s0) * F32(1 << level))
        if end - start == 1:
            terms.append(lev(0, start))
            start += 1
        terms.append(f * lev(0, start))
        box[:, r] = store16((total(terms, ymajor) * inv).astype(F32) + BETA16)
    chat = y - (x - x0) * t
    c0 = np.floor(chat)
    jg = np.round(q(chat - c0) * 256).astype(np.int64)
    c0i = c0.astype(np.int64)
    return u16_read(box[c0i % n_px, xi], box[(c0i + 1) % n_px, xi], jg)


def blur16(k: np.ndarray, intensity: float, angle: float) -> np.ndarray:
    """Return the blur in a 16-bit graph: ``k`` and the result are integer levels."""
    ymajor, t, radius, n, f, inv = setup16(intensity, angle, k.shape[0])
    k = k.astype(np.int64)
    if ymajor:
        k = k.T.copy()
    if radius <= 0.5:
        out = k
    elif n == 1:
        out = small16(k, t, f, inv)
    elif t == 0:
        out = axis16(k, n, f, inv, ymajor)
    else:
        out = rotated16(k, t, n, f, inv, ymajor)
    return out.T if ymajor else out


def tag(v: float) -> str:
    """Return a value as the reference file names write it."""
    text = f"{v:g}".replace(".", "p")
    return text.replace("-", "m")


def main() -> None:
    """Compare the model with the reference renders, case by case."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("refs", type=Path, help="folder of reference renders")
    parser.add_argument("--size", type=int, nargs="*", default=[256, 1024])
    parser.add_argument("--inputs", nargs="*", default=["grid", "tex"])
    parser.add_argument(
        "--single-pass", action="store_true", help="the one-shader form (the same)"
    )
    parser.add_argument(
        "--eight",
        action="store_true",
        help="8-bit graphs: noise8_<N> renders, %% identical pixels / max levels",
    )
    parser.add_argument("--angles", type=float, nargs="*", default=list(ANGLES))
    parser.add_argument(
        "--intensities", type=float, nargs="*", default=list(INTENSITIES)
    )
    parser.add_argument(
        "--sixteen",
        action="store_true",
        help="16-bit graphs: the T2 cells, %% identical pixels / max levels",
    )
    parser.add_argument(
        "--t0", action="store_true", help="16-bit graphs: the pyramid-halves probes"
    )
    parser.add_argument(
        "--p2", action="store_true", help="16-bit graphs: the small-path store probe"
    )
    parser.add_argument(
        "--t1",
        action="store_true",
        help="16-bit graphs: REFERENCE_DIR holds in_<input>_<N>.png and"
        " ref_i<I>_<input>_<N>.png (0 degrees)",
    )
    parser.add_argument(
        "--amd",
        action="store_true",
        help="stores as AMD GPUs make them (8 bits: s8, not s8n; 16 bits: a"
        " pyramid half rounds by float32, not always up)",
    )
    args = parser.parse_args()
    NVIDIA[0] = not args.amd
    if args.eight:
        main8(args)
        return
    if args.sixteen:
        main16(args)
        return
    if args.t0:
        main_t0(args)
        return
    if args.t1:
        main_t1(args.refs)
        return
    if args.p2:
        main_p2(args)
        return
    model = blur_single_pass if args.single_pass else blur
    for n_px in args.size:
        level = n_px.bit_length() - 1
        for name in args.inputs:
            src = args.refs / f"{name}{n_px}.png"
            if not src.exists():
                src = args.refs.parent / src.name
            img = np.asarray(Image.open(src)).astype(np.float64) / 65535
            print(f"{name}{n_px}: worst |error| (1/65535) / pixels > 1")
            print("A \\ I    " + "".join(f"{i:>14g}" for i in INTENSITIES))
            for angle in ANGLES:
                row = []
                for intensity in INTENSITIES:
                    ref_path = args.refs / (
                        f"{name}{n_px}_s{level}_b16_{tag(angle)}_{tag(intensity)}.png"
                    )
                    ref = np.asarray(Image.open(ref_path)).astype(np.float64)
                    got = np.clip(model(img, intensity, angle), 0, 1) * 65535
                    err = np.abs(got - ref)
                    row.append(f"{err.max():8.2f}/{int((err > 1).sum()):<5d}")
                print(f"{angle:<9g}" + "".join(f"{r:>14}" for r in row))
            print()


T2 = (
    ("small and axis, 0", [(s, 0, i) for s in ("tex256", "tex1024") for i in INTENSITIES]),
    ("axis, y-major", [("tex256", 0.25, i) for i in INTENSITIES[2:]]
     + [("tex1024", 0.25, i) for i in INTENSITIES[1:]]),
    ("axis, large R", [("tex1024", 0, i) for i in (8, 16, 48)]
     + [(s, 0, i) for s in ("smooth1024", "smooth2048") for i in (8, 16, 32, 48)]
     + [("hash12_1024", 0, i) for i in (8, 16, 32)]),
    ("axis, both axes", [("tex256", a, 5.86) for a in (0.5, 1, -0.5, 0.75, -0.25)]
     + [("tex256", a, 12) for a in (0.5, 0.75)] + [("tex256", a, 3.3) for a in (0, 0.25)]),
    ("rotated", [("tex256", a, i) for a in (0.125, -0.125) for i in (5.86, 12)]
     + [("tex1024", a, i) for a in (0.125, -0.125) for i in (1.5, 5.86)]
     + [("tex256", 0.1875, i) for i in (2.93, 5.86, 12)]),
    ("rotated, mx near-ties", [("tex1024", a, i) for a in (0.125, -0.125) for i in (2.93, 12)]),
    ("grid", [("grid256", a, i) for a in (0, 0.125, -0.125, 0.1875, 0.25, 0.1)
              for i in INTENSITIES]
     + [("grid1024", a, i) for a in (0, 0.125, -0.125, 0.25, 0.1) for i in INTENSITIES]),
    ("grid, fraction ties", [("grid1024", a, i) for a in (0.0625, 0.1875, 0.4375)
                             for i in INTENSITIES]),
    ("n = 2", [("tex256", 0, i) for i in (1.75, 2, 2.25)]
     + [("tex256", 0.25, 2.25), ("tex256", 0.125, 2.93)]),
    ("small, off the x axis", [("tex256", a, 0.7326) for a in (0.25, 0.125, -0.125, 0.4375)]
     + [("tex1024", 0.25, 0.3)]),
    ("oblique, large", [("tex256", a, i) for a in (0.0625, 0.1875, 0.1, 0.4375)
                        for i in (2.93, 5.86, 12)]),
)  # fmt: skip
"""The specification's T2 cells (AMD renders): (input, angle, intensity) by group."""


def levels16(refs: Path, name: str) -> np.ndarray:
    """Return an input image as 16-bit levels (an 8-bit image's k as 257 k)."""
    src = refs / f"{name}.png"
    if not src.exists():
        src = refs.parent / src.name
    img = np.asarray(Image.open(src))
    return img.astype(np.int64) * (257 if img.dtype == np.uint8 else 1)


def render16(refs: Path, name: str, n_px: int, angle: float, intensity: float):
    """Return a 16-bit graph's reference render as levels, or None if missing."""
    level = n_px.bit_length() - 1
    path = refs / f"{name}_s{level}_b16_{tag(angle)}_{tag(intensity)}.png"
    return np.asarray(Image.open(path)).astype(np.int64) if path.exists() else None


def main16(args: argparse.Namespace) -> None:
    """Compare the 16-bit model with the T2 cells: % identical pixels / max levels."""
    for group, cells in T2:
        print(group)
        for name, angle, intensity in cells:
            img = levels16(args.refs, name)
            ref = render16(args.refs, name, img.shape[0], angle, intensity)
            if ref is None:
                continue
            diff = blur16(img, intensity, angle) - ref
            print(
                f"  {name:12} {angle:<8g} {intensity:<7g}"
                f" {100 * (diff == 0).mean():9.4f}%  max {np.abs(diff).max()}"
            )


def main_t0(args: argparse.Namespace) -> None:
    """Run the T0 probes: per row, whether its pyramid halves went up or down.

    Each row's model is run with every half down and every half up; a row
    is counted for the one that matches all its pixels. Then probe4 and the
    model with the vendor's rule as is.
    """
    probes = (("alt256", 12, None), ("seqA", 3, None), ("seqB", 3, None),
              ("seqC", 3, 1009))  # fmt: skip
    for name, intensity, rows in probes:
        img = levels16(args.refs, name)
        ref = render16(args.refs, name, img.shape[0], 0, intensity)[:rows]
        got = {}
        for way in ("down", "up"):
            HALVES[0] = way
            got[way] = blur16(img, intensity, 0)[:rows]
        HALVES[0] = None
        down = (got["down"] == ref).all(axis=1)
        up = (got["up"] == ref).all(axis=1)
        rule = blur16(img, intensity, 0)[:rows]
        top = img.max(axis=1, keepdims=True)[:rows]
        print(
            f"{name:7} rows up {int((up & ~down).sum())}, down {int((down & ~up).sum())},"
            f" either {int((up & down).sum())}, neither {int((~up & ~down).sum())}"
            f" of {len(ref)}; vendor rule {100 * (rule == ref).mean():.4f}% identical,"
            f" {100 * (rule == top).mean():.4f}% its row's larger value"
        )
    img = levels16(args.refs, "probe4")
    ref = render16(args.refs, "probe4", img.shape[0], 0, 3)
    got = blur16(img, 3, 0)
    a = img.min(axis=1, keepdims=True)
    print(
        f"probe4  {100 * (got == ref).mean():.4f}% identical; render - a:"
        f" +1 at {int((ref - a == 1).sum())}, 0 at {int((ref - a == 0).sum())};"
        f" model - a: +1 at {int((got - a == 1).sum())}, 0 at"
        f" {int((got - a == 0).sum())} pixels"
    )


def p2_rows(out: np.ndarray, img: np.ndarray) -> str:
    """Return P2's even-x up share and its rows' runs (u up, d down) as text."""
    rows = 1009
    even = out[:rows, 0::2] - img[:rows, 0::2]
    up = (even == 1).mean(axis=1) > 0.5
    runs, k = [], 0
    while k < rows:
        j = k
        while j < rows and up[j] == up[k]:
            j += 1
        runs.append(f"{'u' if up[k] else 'd'}{j - k}")
        k = j
    return (
        f"even-x up share {(even == 1).mean():.4f}, rows up {int(up.sum())},"
        f" down {rows - int(up.sum())}; runs {' '.join(runs)}"
    )


def main_p2(args: argparse.Namespace) -> None:
    """Run the P2 probe: the small path (0 degrees, I 0.266) on alternating levels."""
    img = levels16(args.refs, "probe_p2")
    got = blur16(img, 0.266, 0)
    print(f"model   {p2_rows(got, img)}")
    ref = render16(args.refs, "probe_p2", img.shape[0], 0, 0.266)
    if ref is not None:
        same = 100 * (got[:1009] == ref[:1009]).mean()
        print(f"render  {p2_rows(ref, img)}")
        print(f"model = render at {same:.4f}% of rows 0-1008")


def main_t1(folder: Path) -> None:
    """Compare the 16-bit model with the T1 cells (0 degrees, I 8 to 48).

    ``folder`` holds ``in_{hash,smooth}_{1024,2048}.png`` and the renders
    ``ref_i<I>_<input>_<N>.png``: per cell the identical pixels, the
    largest difference and the mean (model - render), in levels.
    """
    for kind in ("hash", "smooth"):
        for n_px in (1024, 2048):
            src = folder / f"in_{kind}_{n_px}.png"
            if not src.exists():
                continue
            img = np.asarray(Image.open(src)).astype(np.int64)
            for intensity in (8, 16, 32, 48):
                path = folder / f"ref_i{intensity}_{kind}_{n_px}.png"
                if not path.exists():
                    continue
                diff = blur16(img, intensity, 0) - np.asarray(Image.open(path))
                print(
                    f"{kind:6} {n_px:5} I {intensity:<3}"
                    f" {100 * (diff == 0).mean():9.4f}%  max {np.abs(diff).max()}"
                    f"  mean {diff.mean():+.4f}"
                )


def main8(args: argparse.Namespace) -> None:
    """Compare the 8-bit model with the noise8_<N> 8-bit graph renders."""
    for n_px in args.size:
        level = n_px.bit_length() - 1
        src = args.refs / f"noise8_{n_px}.png"
        if not src.exists():
            src = args.refs.parent / src.name
        img = np.asarray(Image.open(src)).astype(np.float64) / 255
        print(f"noise8_{n_px}: % identical pixels / max |diff| in levels")
        print("A \\ I    " + "".join(f"{i:>14g}" for i in args.intensities))
        for angle in args.angles:
            row = []
            for intensity in args.intensities:
                ref_path = args.refs / (
                    f"noise8_{n_px}_s{level}_b8_{tag(angle)}_{tag(intensity)}.png"
                )
                if not ref_path.exists():
                    row.append("-")
                    continue
                ref = np.asarray(Image.open(ref_path)).astype(np.float64) / 257
                got = np.round(blur8(img, intensity, angle).astype(np.float64) * 255)
                diff = np.abs(got - np.round(ref))
                row.append(f"{100 * (diff == 0).mean():.4f}/{int(diff.max())}")
            print(f"{angle:<9g}" + "".join(f"{r:>14}" for r in row))
        print()


if __name__ == "__main__":
    main()
