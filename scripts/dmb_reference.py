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
        levels.append(s8((a[:, 0::2] + a[:, 1::2]) / F32(2)))
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
        out[:, x] = s8(acc / F32(2 * radius))
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
    frame = s8(read32(img, x, y + (x - x0) * t))  # frame[c, r]
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
        box[:, r] = s8(acc / F32(2 * radius) + F32(1 / 512))
    chat = y - (x - x0) * t
    c0 = np.floor(chat)
    g = q(chat - c0).astype(F32)
    c0i, xi = c0.astype(np.int64), x.astype(np.int64)
    return s8((F32(1) - g) * box[c0i % n_px, xi] + g * box[(c0i + 1) % n_px, xi])


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
        out = s8((m * ws + p * ws) + img * wc)
    elif t == 0:
        out = axis8(img, radius)
    else:
        out = rotated8(img, t, radius)
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
    args = parser.parse_args()
    if args.eight:
        main8(args)
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
