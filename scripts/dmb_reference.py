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
    args = parser.parse_args()
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
                    got = np.clip(blur(img, intensity, angle), 0, 1) * 65535
                    err = np.abs(got - ref)
                    row.append(f"{err.max():8.2f}/{int((err > 1).sum()):<5d}")
                print(f"{angle:<9g}" + "".join(f"{r:>14}" for r in row))
            print()


if __name__ == "__main__":
    main()
