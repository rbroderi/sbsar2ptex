"""Text nodes, ops 34 and 35: docs/op-34.md section 4, on synthetic cooks.

One node at offset 0x100; its text and font records in the data area
before it (as the cooker places them), reached only through their offsets.
"""

import struct

import numpy as np
import pytest
from test_fxmap_seed import block

from sbsar2ptex import fxglsl
from sbsar2ptex import translate
from sbsar2ptex.function import FLOAT
from sbsar2ptex.sbsasm import BASE
from sbsar2ptex.sbsasm import Asm
from sbsar2ptex.sbsasm import Node
from sbsar2ptex.translate import GraphInput
from sbsar2ptex.translate import Sub
from sbsar2ptex.translate import UnsupportedError
from sbsar2ptex.translate import node_shader
from sbsar2ptex.translate import text_params

AT = 0x100
SEED_UID, SIZE_UID = 0x00C0FFEE, 0x00C0FFEF
SZ = [0x0A420001, SIZE_UID]
"""A size function: load $outputsize (integer, width 2)."""
SB = [0x0A020001, SEED_UID]
"""A seed block: $randomseed."""
T_AT, FT_AT = 0x2C, 0x5C
"""Where the text and font records go (offsets: file 0x60 and 0x90)."""


def f32w(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def f1(v: float) -> list[int]:
    return [0x09000001, f32w(v)]


def f2(x: float, y: float) -> list[int]:
    return [0x11400001, f32w(x), f32w(y)]


def f4(*v: float) -> list[int]:
    return [0x21C00001, *(f32w(x) for x in v)]


def gray_of(v: float) -> list[int]:
    """Return a gray colour function: float4 (v, v, v, v), then its mean (2.2)."""
    third = 1 / 3
    return block(
        (0x00, FLOAT, 4, _halves(v, v, v, v)),
        (0x00, FLOAT, 4, [0, *_halves(third, third, third, 0.0)]),  # (aligned)
        (0x18, FLOAT, 1, [0, 1]),
    )  # fmt: skip


def _halves(*v: float) -> list[int]:
    out = []
    for x in v:
        w = f32w(x)
        out += [w & 0xFFFF, w >> 16]
    return out


def cook(
    head: list[int | str],
    blocks: list[tuple[str, list[int]]] = (),
    *,
    text: list[int] | None = None,
    font: bytes | None = None,
) -> Node:
    """Parse a cook whose one node, at AT, is ``head`` then ``blocks``.

    A str in ``head`` is the offset of the block of that name; "T" and "FT"
    are the text's and font's records (``text``, ``font``).
    """
    at = AT + 4 * len(head)
    where = {"T": T_AT, "FT": FT_AT}
    for name, ws in blocks:
        where[name] = at
        at += 4 * len(ws)
    words = [where[w] if isinstance(w, str) else w for w in head]
    for _, ws in blocks:
        words += ws
    node_at = AT + BASE
    defaults = node_at + 4 * len(words)
    uids = defaults + 12
    end = uids + 4 + 4 + 16
    data = bytearray(end + 16)
    data[:4] = b"SBAM"
    struct.pack_into("<H", data, 6, 9)
    struct.pack_into("<I", data, 0x40, AT)  # the node table
    struct.pack_into("<II", data, 0x44, 0xC0, 0)  # the outputs
    if text is not None:
        struct.pack_into(f"<{1 + len(text)}I", data, T_AT + BASE, len(text), *text)
    if font is not None:
        struct.pack_into("<II", data, FT_AT + BASE, 0x1234, len(font))
        data[FT_AT + BASE + 8 : FT_AT + BASE + 8 + len(font)] = font
        assert FT_AT + BASE + 8 + len(font) <= node_at
    struct.pack_into(f"<{len(words)}I", data, node_at, *words)
    struct.pack_into("<iii", data, defaults, 0, 6, 6)
    struct.pack_into("<HHI", data, uids, 1, 2, 0x1234)
    struct.pack_into("<4I", data, uids + 8, 4, SEED_UID, 8, SIZE_UID)
    struct.pack_into("<4I", data, end, 1, 0x40 - BASE, 0x44 - BASE, defaults - BASE)
    (node,) = Asm(bytes(data)).nodes()
    return node


@pytest.fixture(autouse=True)
def ctx() -> None:
    translate._CTX = translate._Context(
        size=8,
        inputs={
            SEED_UID: GraphInput("$randomseed", 4, [0]),
            SIZE_UID: GraphInput("$outputsize", 8, [6, 6]),
        },
    )


FONT = b"\x00\x01\x00\x00" + bytes(12)
"""A font record's file (an sfnt header's start; never rendered here)."""
AB, HH = [0x41, 0x42], [0x48, 0x0A, 0x48]


def fill(node: Node) -> tuple[dict, str]:
    """Return the node's shader and its stored instance code."""
    sh, inputs = node_shader(node, "t0", size=8)
    assert inputs == []
    assert translate.stored(Sub([sh], [], [], "t0"), node)
    return sh, sh["shader_model"]["instance"]


def fill8(v: float) -> int:
    """fill8_glsl in float32: the 8-bit level a uniform fills v as."""
    v = np.float32(v)
    b = v * np.float32(256) - np.float32(0.5)
    r = b - v
    k = np.ceil(r)
    return int(k + np.float32(r == k and b - k > v))


# Layout


def test_1_defaults() -> None:
    node = cook([0x03098822, 0x00001000, 0, 0, 0, "SZ"], [("SZ", SZ)])
    assert (node.op, node.flags >> 8 & 3, node.fields) == (34, 3, [0x1000, 0, 0, 0])
    assert node.size_code == AT + 4 * 6 + BASE
    assert node.inline == []
    assert (node.text, node.font) == ([], None)
    params = text_params(node, color=False)
    assert all(state == "default" for state, _ in params.values())
    sh, code = fill(node)
    assert sh["parameters"]["background"] == 0.0
    assert fxglsl.fill8_glsl("float") in code


def test_2_and_14_the_manhole_shape() -> None:
    node = cook(
        [0x03198822, 0x00001100, 0, "T", 0, "SZ", 0x3C75C28F], [("SZ", SZ)], text=AB
    )
    assert node.text == AB
    assert node.font is None
    assert node.inline == [0x3C75C28F]
    params = text_params(node, color=False)
    assert params["fontsize"] == ("const", [pytest.approx(0.015)])
    assert params["background"] == ("default", None)
    sh, _ = fill(node)  # a uniform 0, whatever the text
    assert sh["parameters"]["background"] == 0.0
    assert "return $background;" in sh["shader_model"]["instance"]


ALL = [0x3E999999, 0x3F199999, *map(f32w, (2.0, 0.5, 0.25, 3.0, 0.1, 0.2)),
       0x3E99999A]  # fmt: skip


def test_3_all_constants_gray() -> None:
    node = cook([0x03098822, 0x00002554, 0, "T", 0, "SZ", *ALL], [("SZ", SZ)], text=AB)
    p = text_params(node, color=False)
    approx = pytest.approx
    assert p["background"] == ("const", [approx(0.3)])
    assert p["fontcolor"] == ("const", [approx(0.6)])
    assert p["matrix"] == ("const", approx([2.0, 0.5, 0.25, 3.0]))
    assert p["position"] == ("const", approx([0.1, 0.2]))
    assert p["fontsize"] == ("const", [approx(0.3)])
    assert node.fields[0] >> 12 & 3 == 2  # alignment: right
    sh, _ = fill(node)
    assert sh["parameters"]["background"] == approx(0.3)


def test_4_all_constants_color() -> None:
    words = [*map(f32w, (0.2, 0.3, 0.4, 0.9, 0.5, 0.6, 0.7, 0.8)),
             *map(f32w, (2.0, 0.5, 0.25, 3.0, 0.1, 0.2, 0.3))]  # fmt: skip
    node = cook([0x03098823, 0x00002554, 0, "T", 0, "SZ", *words], [("SZ", SZ)])
    assert len(node.inline) == 15
    p = text_params(node, color=True)
    approx = pytest.approx
    assert p["background"] == ("const", approx([0.2, 0.3, 0.4, 0.9]))
    assert p["fontcolor"] == ("const", approx([0.5, 0.6, 0.7, 0.8]))
    assert p["matrix"] == ("const", approx([2.0, 0.5, 0.25, 3.0]))
    assert p["position"] == ("const", approx([0.1, 0.2]))
    assert p["fontsize"] == ("const", [approx(0.3)])


@pytest.mark.parametrize("color", [False, True])
def test_5_all_functions(*, color: bool) -> None:
    colour = f4(0.2, 0.3, 0.4, 0.9) if color else f1(0.3)
    blocks = [("SZ", SZ), ("bg", colour), ("fc", colour), ("m", f4(1, 0, 0, 1)),
              ("pos", f2(0, 0)), ("size", f1(0.25))]  # fmt: skip
    word = 0x03098823 if color else 0x03098822
    node = cook([word, 0x00001AA8, 0, 0, 0, "SZ", "bg", "fc", "m", "pos", "size"],
                blocks)  # fmt: skip
    p = text_params(node, color=color)
    offsets = [p[k] for k in ("background", "fontcolor", "matrix", "position",
                              "fontsize")]  # fmt: skip
    assert all(state == "func" for state, _ in offsets)
    starts = [v for _, v in offsets]
    assert starts == sorted(starts)
    assert starts[0] == node.size_code + 4 * len(SZ)
    sh, code = fill(node)
    assert "background" not in sh["parameters"]  # the function, live
    assert "return " in code


def test_6_a_seed_block() -> None:
    node = cook(
        [0x03898822, 0x00001200, 0, 0, 0, "SB", "SZ", "size"],
        [("SB", SB), ("SZ", SZ), ("size", f1(0.25))],
    )
    assert node.seed_state == 2
    assert node.seed_code == AT + 4 * 8 + BASE
    assert node.size_code == node.seed_code + 4 * len(SB)
    assert text_params(node, color=False)["fontsize"][0] == "func"
    fill(node)


def test_7_tiling_0() -> None:
    node = cook([0x00098822, 0x00001000, 0, 0, 0, "SZ"], [("SZ", SZ)])
    assert node.flags >> 8 == 0
    assert node.inline == []
    fill(node)


def test_8_no_size_function() -> None:
    node = cook([0x03088822, 0x00001100, 0, 0, 0, 0x3C75C28F])
    assert node.size_code is None
    assert node.inline == [0x3C75C28F]
    assert text_params(node, color=False)["fontsize"] == (
        "const", [pytest.approx(0.015)],
    )  # fmt: skip


@pytest.mark.parametrize("text", [AB, HH])
def test_9_text_record(text: list[int]) -> None:
    node = cook([0x03098822, 0x00001000, 0, "T", 0, "SZ"], [("SZ", SZ)], text=text)
    assert node.text == text


def test_10_font_record() -> None:
    node = cook([0x03098822, 0x00001000, 0, 0, "FT", "SZ"], [("SZ", SZ)], font=FONT)
    assert node.font == FONT
    assert node.text == []


@pytest.mark.parametrize(
    ("f", "second"),
    [(0x1003, 0), (0x100C, 0), (0x1030, 0), (0x10C0, 0), (0x1300, 0), (0x1C00, 0),
     (0x3000, 0), (0x1001, 0), (0x5000, 0), (0x81000, 0), (0x1000, 1)],
)  # fmt: skip
def test_11_rejections(f: int, second: int) -> None:
    node = cook([0x03098822, f, second, 0, 0, "SZ", *[0] * 8], [("SZ", SZ)])
    with pytest.raises(UnsupportedError):
        node_shader(node, "t0", size=8)


# Semantics


@pytest.mark.parametrize(
    ("text", "font"), [(AB, None), ([], FONT), ([0x20], FONT), ([0x20] * 3, FONT)]
)
def test_12_no_ink_color(text: list[int], font: bytes | None) -> None:
    bg = [*map(f32w, (0.2, 0.3, 0.4, 0.9))]
    node = cook(
        [0x03098823, 0x00001004, 0, "T" if text else 0, "FT" if font else 0, "SZ", *bg],
        [("SZ", SZ)],
        text=text,
        font=font,
    )
    sh, code = fill(node)
    c = sh["parameters"]["background"]
    assert [c[k] for k in "rgba"] == pytest.approx([0.2, 0.3, 0.4, 0.9])
    assert fxglsl.fill8_glsl("vec4") in code
    levels = [fill8(struct.unpack("<f", struct.pack("<I", w))[0]) for w in bg]
    assert levels == [51, 77, 102, 229]


@pytest.mark.parametrize("text", [AB, HH, [0x20, 0x41]])
def test_12_ink_with_a_font_is_unsupported(text: list[int]) -> None:
    node = cook(
        [0x03098822, 0x00001000, 0, "T", "FT", "SZ"], [("SZ", SZ)], text=text, font=FONT
    )
    with pytest.raises(UnsupportedError, match="embedded font"):
        node_shader(node, "t0", size=8)


@pytest.mark.parametrize(
    ("v", "want"), [(0.3, 77), (0.43333334, 111), (0.1, 26), (0.7, 178)]
)
@pytest.mark.parametrize("color", [False, True])
def test_13_a_background_function(v: float, want: int, *, color: bool) -> None:
    bg = f4(v, v, v, v) if color else gray_of(v)
    word = 0x03098823 if color else 0x03098822
    node = cook([word, 0x00001008, 0, "T", 0, "SZ", "bg"], [("SZ", SZ), ("bg", bg)],
                text=AB)  # fmt: skip
    translate._CTX.helpers = []
    state, offset = text_params(node, color=color)["background"]
    assert state == "func"
    value = translate.dynamic(node, offset)
    assert value.width == (4 if color else 1)
    assert [fill8(x) for x in value.static] == [want] * value.width
    translate._CTX.helpers = []
    sh, code = fill(node)
    assert "background" not in sh["parameters"]
    assert "_$(name)(uv)" in code  # (its function, called per pixel)
