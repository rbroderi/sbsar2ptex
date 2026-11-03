"""Model a 16-bit Warp on the CPU and score sampling models against Substance's image.

    uv run python scripts/warp_fit.py INPUT GRADIENT TRUTH --intensity K
        [--mm IMAGE ...] [--split]

INPUT, GRADIENT and TRUTH are Substance's 16-bit renders (PNG) of a warp's
input, its gradient input and the warp itself (Biomech's n1534, n1526 and
n1536), all one size. Each model computes the warp's sample position as
translate.warp_body does (float32, Substance's order) and then reads the
input its own way; the table gives each model's pixels exact, within one
level, and its largest error against TRUTH. ``--mm`` images (Material
Maker's renders of the warp; ``--split``: high byte in red, low in green)
are scored too, and against the "ours" model, which is what the
translation computes on NVIDIA (warp_glsl), so that a mismatch there shows
the model is not the shader. Then, for the pixels "ours" misses: how far
the sample moved, its 1/256 fractions, and the spread of the four texels
it blends.
"""

import argparse
from pathlib import Path

import numpy as np
import png16
from PIL import Image

F = np.float32


def levels(path: Path, *, split: bool = False) -> np.ndarray:
    """Return a 16-bit image's first channel's levels (``split``: hi, lo bytes)."""
    if split:
        enc = np.asarray(Image.open(path)).astype(np.int64)
        return enc[..., 0] * 256 + enc[..., 1]
    a = png16.read(path)[..., 0]
    return a * 257 if png16.bit_depth(path) == 8 else a


def value(lv: np.ndarray) -> np.ndarray:
    """Return the float32 nearest each level / 65535 (warp_glsl's _val)."""
    return (lv / 65535.0).astype(F)


def position(grad: np.ndarray, k: float, *, rounding: str) -> tuple[np.ndarray, ...]:
    """Return each pixel's sample q (in 256ths of a texel, from texel 0's center).

    As warp_body: backward differences of the gradient's values, times
    K = k (n / 256), added to the pixel's center, all float32. Also the
    unrounded q, for the models that read between 256ths.
    """
    n = grad.shape[0]
    h = value(grad)
    hx = value(np.roll(grad, 1, axis=1))
    hy = value(np.roll(grad, 1, axis=0))
    big_k = F(F(k) * F(n / 256))
    o = np.stack([(h - hx) * big_k, (h - hy) * big_k], -1).astype(F)
    py, px = np.mgrid[0:n, 0:n]
    c = np.stack([(px + F(0.5)) / F(n), (py + F(0.5)) / F(n)], -1).astype(F)
    raw = ((c + o).astype(F) * F(n * 256.0) - F(128.0)).astype(F)
    if rounding == "half_up":
        q = np.floor(raw + F(0.5))
    elif rounding == "even":
        q = np.round(raw)
    elif rounding == "floor":
        q = np.floor(raw)
    else:
        q = raw
    return q.astype(np.float64), raw.astype(np.float64)


def corners(inp: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return the four texels' levels around q and its fractions (in 1/256)."""
    n = inp.shape[0]
    b = np.floor(q / 256.0)
    f = q - 256.0 * b
    x0 = b[..., 0].astype(np.int64) % n
    y0 = b[..., 1].astype(np.int64) % n
    x1, y1 = (x0 + 1) % n, (y0 + 1) % n
    return inp[y0, x0], inp[y0, x1], inp[y1, x0], inp[y1, x1], f[..., 0], f[..., 1]


def ours(inp: np.ndarray, q: np.ndarray) -> np.ndarray:
    """NVIDIA as warp_glsl has it: corner weight fx fy / 256 half up, sum's 256th half up."""
    a, b, c, d, fx, fy = corners(inp, q)
    fx, fy = fx.astype(np.int64), fy.astype(np.int64)
    w11 = (fx * fy + 128) >> 8
    s = (256 - fx - fy + w11) * a + (fx - w11) * b + (fy - w11) * c + w11 * d
    return (s + 128) >> 8


def exact(inp: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Separable 256ths weights, the blend exact, rounded half up."""
    a, b, c, d, fx, fy = corners(inp, q)
    fx, fy = fx.astype(np.int64), fy.astype(np.int64)
    s = (256 - fx) * (256 - fy) * a + fx * (256 - fy) * b
    s += (256 - fx) * fy * c + fx * fy * d
    return (s + 32768) >> 16


def lerp32(inp: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Two float32 lerps of the values by fx / 256 then fy / 256, stored rounded."""
    a, b, c, d, fx, fy = corners(inp, q)
    wx, wy = (fx / 256).astype(F), (fy / 256).astype(F)
    va, vb, vc, vd = (value(t) for t in (a, b, c, d))
    top = (va + (vb - va) * wx).astype(F)
    bot = (vc + (vd - vc) * wx).astype(F)
    v = (top + (bot - top) * wy).astype(F)
    return np.floor(v.astype(np.float64) * 65535 + 0.5).astype(np.int64)


def score(got: np.ndarray, want: np.ndarray) -> str:
    """Return exact, within one level and the largest error, as a table row."""
    d = np.abs(got - want)
    return (
        f"exact {100 * float((d == 0).mean()):8.4f}%  +-1 {100 * float((d <= 1).mean()):8.4f}%"
        f"  max {int(d.max()):6}  mean {float((got - want).mean()):+.3f}"
    )


def main() -> None:
    """Score the models and describe the pixels "ours" misses."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("gradient", type=Path)
    parser.add_argument("truth", type=Path)
    parser.add_argument("--intensity", type=float, required=True)
    parser.add_argument("--mm", type=Path, nargs="*", default=[])
    parser.add_argument("--split", action="store_true")
    args = parser.parse_args()
    inp, grad, truth = (levels(p) for p in (args.input, args.gradient, args.truth))
    rows = {}
    for rounding in ("half_up", "even", "floor"):
        q, raw = position(grad, args.intensity, rounding=rounding)
        for name, model in (("ours", ours), ("exact", exact), ("lerp32", lerp32)):
            rows[f"{name}/{rounding}"] = model(inp, q)
    # unrounded: the blend at the float position, 1/256 not imposed
    _, raw = position(grad, args.intensity, rounding="none")
    a, b, c, d, fx, fy = corners(inp, raw)
    wx, wy = fx / 256, fy / 256
    s = (1 - wx) * (1 - wy) * a + wx * (1 - wy) * b + (1 - wx) * wy * c + wx * wy * d
    rows["float/none"] = np.floor(s + 0.5).astype(np.int64)
    for name, got in rows.items():
        print(f"{name:14} {score(got, truth)}")
    mine = rows["ours/half_up"]
    for p in args.mm:
        got = levels(p, split=args.split)
        print(f"{p.name[:14]:14} {score(got, truth)}   vs ours: {score(got, mine)}")
    miss = mine != truth
    q, raw = position(grad, args.intensity, rounding="half_up")
    py, px = np.mgrid[0 : truth.shape[0], 0 : truth.shape[0]]
    move = np.hypot(raw[..., 0] / 256 + 0.5 - px, raw[..., 1] / 256 + 0.5 - py)
    a, b, c, d, fx, fy = corners(inp, q)
    spread = np.max([a, b, c, d], 0) - np.min([a, b, c, d], 0)
    frac = np.abs(raw - np.round(raw)).max(-1)
    for label, v, bins in (
        ("move (texels)", move, [0, 0.01, 0.1, 0.5, 1, 2, 4, 16, 1e9]),
        ("texel spread", spread, [0, 1, 16, 256, 1024, 8192, 65536]),
        ("dist to 1/256 tie", frac, [0, 0.01, 0.05, 0.25, 0.45, 0.5]),
    ):
        h_all = np.histogram(v, bins)[0]
        h_miss = np.histogram(v[miss], bins)[0]
        print(f"{label}: bins {bins}")
        print(f"   all  {h_all.tolist()}")
        print(f"   miss {h_miss.tolist()}")
    ys, xs = np.nonzero(miss)
    for i in range(min(8, len(xs))):
        y, x = ys[i], xs[i]
        print(
            f"  ({x},{y}) truth {truth[y, x]} ours {mine[y, x]} q {raw[y, x].tolist()}"
            f" corners {[int(t[y, x]) for t in (a, b, c, d)]}"
            f" g {int(grad[y, x])} gx {int(grad[y, x - 1])} gy {int(grad[y - 1, x])}"
        )


if __name__ == "__main__":
    main()
