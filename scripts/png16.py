"""Read a PNG's samples at full depth (Pillow reads 16-bit RGBA as 8 bits)."""

import struct
import zlib
from pathlib import Path

import numpy as np

CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}
"""Samples a pixel by PNG color type (gray, RGB, gray-alpha, RGBA)."""


def read(path: Path) -> np.ndarray:
    """Return a non-interlaced 8- or 16-bit PNG's samples, height x width x channels."""
    data = path.read_bytes()
    at, idat = 8, []
    width = height = depth = kind = 0
    while at < len(data):
        (size,) = struct.unpack(">I", data[at : at + 4])
        tag = data[at + 4 : at + 8]
        body = data[at + 8 : at + 8 + size]
        if tag == b"IHDR":
            width, height, depth, kind, _, _, interlace = struct.unpack(
                ">IIBBBBB", body
            )
            if interlace or kind not in CHANNELS or depth not in (8, 16):
                msg = f"{path}: unsupported PNG layout"
                raise ValueError(msg)
        elif tag == b"IDAT":
            idat.append(body)
        at += 12 + size
    ch = CHANNELS[kind]
    bpp = ch * depth // 8
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), np.uint8)
    raw = raw.reshape(height, 1 + width * bpp)
    filters, rows = raw[:, 0], raw[:, 1:].reshape(height, width, bpp).astype(np.int32)
    out = np.zeros((height, width, bpp), np.int32)
    zero = np.zeros(bpp, np.int32)
    # Each pixel needs its left, upper and upper-left neighbors decoded:
    # decode along anti-diagonals, every row at once.
    r_all = np.arange(height)
    for d in range(height + width - 1):
        r = r_all[(d - r_all >= 0) & (d - r_all < width)]
        x = d - r
        a = np.where((x > 0)[:, None], out[r, np.maximum(x - 1, 0)], zero)
        b = np.where((r > 0)[:, None], out[np.maximum(r - 1, 0), x], zero)
        c = np.where(
            ((x > 0) & (r > 0))[:, None],
            out[np.maximum(r - 1, 0), np.maximum(x - 1, 0)],
            zero,
        )
        p = a + b - c
        pa, pb, pc = np.abs(p - a), np.abs(p - b), np.abs(p - c)
        paeth = np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))
        f = filters[r][:, None]
        pred = np.select(
            [f == 1, f == 2, f == 3, f == 4],
            [a, b, (a + b) // 2, paeth],
            0,
        )
        out[r, x] = (rows[r, x] + pred) & 255
    if depth == 16:
        out = out.reshape(height, width, ch, 2)
        return (out[..., 0] << 8 | out[..., 1]).astype(np.int64)
    return out.reshape(height, width, ch).astype(np.int64)


def bit_depth(path: Path) -> int:
    """Return a PNG's bits per sample."""
    return path.read_bytes()[24]
