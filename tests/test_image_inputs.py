"""Embedded image inputs, resolved by uid (docs/image-inputs.md section 4).

Synthetic .sbsasm files: a data area from 0x38, then image input nodes.
"""

import struct
from pathlib import Path

import pytest

from sbsar2ptex.procedural import bitmaps
from sbsar2ptex.sbsasm import BASE
from sbsar2ptex.sbsasm import Asm

RAW = bytes(range(16))
"""A 4 x 4 gray 8-bit raw image."""
JPEG = b"\xff\xd8\xff\xe0" + b"jpeg!!"
"""A file with a JPEG header, 10 bytes (padded to 12 after it)."""


def encoded(blob: bytes) -> bytes:
    """Return a length word, then ``blob``, padded to 4 bytes."""
    data = struct.pack("<I", len(blob)) + blob
    return data + b"\0" * (-len(data) % 4)


def image_node(fmt: int, uid: int) -> list[int]:
    """Return an image input node's words: x = 0x22 (4 x 4)."""
    return [fmt << 24 | 0x22 << 8 | 32, uid]


def sbsam(area: bytes, nodes: list[list[int]]) -> Asm:
    """Return a cook of ``area`` (from 0x38) then ``nodes``; no graph inputs."""
    data = bytearray(b"SBAM" + b"\0" * (0x38 - 4) + area)
    struct.pack_into("<H", data, 6, 9)
    starts = []
    for words in nodes:
        starts.append(len(data))
        data += struct.pack(f"<{len(words)}I", *words)
    table = len(data)
    data += struct.pack(f"<{len(starts)}I", *(s - BASE for s in starts))
    outputs = len(data)
    data += struct.pack("<II", 0xC0, len(starts) - 1)
    data += struct.pack("<HHI", 1, 0, 0x1234)
    data += struct.pack("<4I", len(starts), table - BASE, outputs - BASE, 0)
    return Asm(bytes(data))


def test_1_raw_then_jpeg(tmp_path: Path) -> None:
    area = RAW + encoded(JPEG)
    asm = sbsam(area, [image_node(1, 4), image_node(8, 0x14)])
    a, b = asm.nodes()
    assert a.unsupported is None
    assert b.unsupported is None
    assert asm.bitmaps == []  # the scan from 0x38 stops at the raw image
    assert asm.raw_image(a).pixels == RAW
    at, length = asm.encoded_image(b)
    assert asm.d[at : at + length] == JPEG
    files = bitmaps(asm, [a, b], tmp_path)
    assert set(files) == {4, 0x14}  # no "no image for graph input 20"
    assert Path(files[0x14]).read_bytes() == JPEG


def test_2_jpeg_first(tmp_path: Path) -> None:
    area = encoded(JPEG) + RAW
    raw_uid = 0x38 - BASE + len(encoded(JPEG))
    asm = sbsam(area, [image_node(1, raw_uid), image_node(8, 4)])
    a, b = asm.nodes()
    assert a.unsupported is None
    assert b.unsupported is None
    files = bitmaps(asm, [a, b], tmp_path)
    assert set(files) == {raw_uid, 4}
    assert Path(files[4]).read_bytes() == JPEG


def test_3_bad_length_is_malformed() -> None:
    area = struct.pack("<I", 1000) + JPEG + b"\0\0"
    asm = sbsam(area, [image_node(8, 4)])
    (b,) = asm.nodes()
    assert asm.encoded_image(b) is None
    assert b.unsupported is not None
    assert "malformed" in b.unsupported


def test_a_file_without_a_known_header_is_malformed() -> None:
    asm = sbsam(encoded(b"not an image"), [image_node(8, 4)])
    (b,) = asm.nodes()
    assert "malformed" in (b.unsupported or "")


@pytest.mark.parametrize("fmt", [0, 4, 9])
def test_4_an_unknown_format_is_unsupported(fmt: int) -> None:
    asm = sbsam(RAW, [image_node(fmt, 4)])
    (n,) = asm.nodes()
    assert "unsupported" in (n.unsupported or "")
