"""Write the corpus's deterministic test patterns (stdlib only).

    python make_patterns.py OUT_DIR [--size N]     (N = 256 default, or 128)

Every pattern is N x N (a power of two, at most 256). Pixel (x, y) has x to
the right and y DOWN from the top-left corner (PNG row order). Let
s = 256 / N (1 at 256 px, 2 at 128 px).

coded_rgba.png    RGBA 8-bit. R = s*x, G = s*y, B = R XOR G, A = 255. (R, G)
                  is unique per pixel, so any resampling, flip or offset can
                  be decoded from an output pixel.
noise_rgba.png    RGBA 8-bit. R, G, B = successive bytes of rng(SEED_RGB),
                  row by row, pixel by pixel, R then G then B; A = 255.
coded_gray16.png  Gray 16-bit. v = (y * N + x) * 65536 / N^2: every pixel
                  unique, a raster ramp over the 16-bit range (0 top-left).
noise_gray.png    Gray 8-bit. v = successive bytes of rng(SEED_GRAY).
ramp_h.png        Gray 8-bit. v = s*x (a horizontal ramp, for gradient maps).
bumps16.png       Gray 16-bit, tileable, smooth. v = round(65535 * (0.5 +
                  0.25 * sin(2 pi * 2x / N) + 0.25 * sin(2 pi * 3y / N))):
                  a warp / slope intensity input.
mask.png          Gray 8-bit, binary 0/255. In 256-px units (u, v) = (s*x,
                  s*y): a disc of radius 64 at (80, 96) and a rectangle u in
                  [150, 230), v in [40, 220); for distance and blend opacity.

rng(seed): xorshift32 (x ^= x << 13; x ^= x >> 17; x ^= x << 5, mod 2^32),
state = seed, each step's low byte is the next byte. SEED_RGB = 0x1234567,
SEED_GRAY = 0x2468ACE.
"""

import argparse
import math
import struct
import zlib
from pathlib import Path

SEED_RGB = 0x1234567
SEED_GRAY = 0x2468ACE


def rng(seed: int):
    x = seed
    while True:
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        yield x & 0xFF


def png(path: Path, rows: list[bytes], color_type: int, depth: int) -> None:
    """Write a PNG: color_type 0 gray / 6 RGBA, depth 8 or 16."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xFFFFFFFF
        )

    bpp = {0: 1, 6: 4}[color_type] * depth // 8
    width = len(rows[0]) // bpp
    raw = bytearray()
    prev = bytes(len(rows[0]))
    for row in rows:  # per row the smaller of filters Up, Sub, None
        up = bytes((row[i] - prev[i]) & 0xFF for i in range(len(row)))
        sub = bytes(
            (row[i] - (row[i - bpp] if i >= bpp else 0)) & 0xFF for i in range(len(row))
        )
        best = min(((2, up), (1, sub), (0, row)),
                   key=lambda t: sum(min(b, 256 - b) for b in t[1]))  # fmt: skip
        raw += bytes([best[0]]) + best[1]
        prev = row
    ihdr = struct.pack(">IIBBBBB", width, len(rows), depth, color_type, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--size", type=int, default=256)
    a = ap.parse_args()
    n, out = a.size, a.out
    s = 256 // n
    out.mkdir(parents=True, exist_ok=True)
    rows = range(n)
    png(out / "coded_rgba.png",
        [bytes(v for x in rows for v in (s * x, s * y, (s * x) ^ (s * y), 255)) for y in rows],
        6, 8)  # fmt: skip
    r = rng(SEED_RGB)
    png(out / "noise_rgba.png",
        [bytes(v for x in rows for v in (next(r), next(r), next(r), 255)) for y in rows], 6, 8)
    step = 65536 // (n * n)
    png(out / "coded_gray16.png",
        [b"".join(struct.pack(">H", (y * n + x) * step) for x in rows) for y in rows], 0, 16)
    g = rng(SEED_GRAY)
    png(out / "noise_gray.png", [bytes(next(g) for x in rows) for y in rows], 0, 8)
    png(out / "ramp_h.png", [bytes(s * x for x in rows) for y in rows], 0, 8)

    def bump(x: int, y: int) -> int:
        v = 0.5 + 0.25 * math.sin(2 * math.pi * 2 * x / n) + 0.25 * math.sin(2 * math.pi * 3 * y / n)
        return round(65535 * v)

    png(out / "bumps16.png",
        [b"".join(struct.pack(">H", bump(x, y)) for x in rows) for y in rows], 0, 16)

    def inside(x: int, y: int) -> bool:
        u, v = s * x, s * y
        return (u - 80) ** 2 + (v - 96) ** 2 < 64**2 or (150 <= u < 230 and 40 <= v < 220)

    png(out / "mask.png", [bytes(255 if inside(x, y) else 0 for x in rows) for y in rows], 0, 8)


if __name__ == "__main__":
    main()
