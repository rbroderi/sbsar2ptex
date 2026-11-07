"""The SVG node, ops 10 and 11 (docs/svg-filter.md)."""

import random
import struct
from pathlib import Path

import numpy as np
import pytest
from test_motion_blur import node_file
from test_slots import compiles

from sbsar2ptex import svg
from sbsar2ptex.sbsasm import BASE
from sbsar2ptex.translate import UnsupportedError
from sbsar2ptex.translate import translate_graph

OFF = -1


def f(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def record(words: list[int]) -> list[svg.Paint] | None:
    """Return ``words`` read as a record at offset 0 of some data."""
    data = bytes(BASE) + struct.pack(f"<{len(words)}I", *words)
    return svg.read_record(data, 0)


# Layout (section 6, tests 1-7): an empty record (the word 0) at offset 4,
# ahead of the node table, as the cooker places it.


@pytest.mark.parametrize(
    ("word", "rest", "background", "has_size"),
    [
        (0x0009880B, [4, OFF], (0, None), True),  # 1: corpus default, colour
        (0x0009880A, [4, OFF], (0, None), True),  # 2: gray
        (0x0008880B, [4], (0, None), False),  # 3: fixed size
        (0x0208880B, [4, f(0.2), f(0.4), f(0.6), f(0.8)],
         (1, [pytest.approx(v) for v in (0.2, 0.4, 0.6, 0.8)]), False),  # 4
        (0x0209880A, [4, OFF, f(0.4)], (1, [pytest.approx(0.4)]), True),  # 5
        (0x0608880B, [4, 1], (3, 1), False),  # 6: background input
    ],
)  # fmt: skip
def test_layouts(
    word: int,
    rest: list[int],
    background,
    has_size: bool,  # noqa: FBT001
) -> None:
    node = node_file(word, rest).nodes()[-1]
    assert node.unsupported is None
    assert node.vector == []  # the empty record
    assert node.params["background"] == background
    assert (node.size_code is not None) == has_size
    if background[0] == 3:
        assert node.inputs == {"background": 1}


def test_a_background_function_is_unsupported() -> None:
    node = node_file(0x0408880B, [4, OFF]).nodes()[-1]  # 7: P bits 1-2 = 2
    assert node.unsupported is not None


def tri_record() -> list[int]:
    # test 8: one solid paint (255, 128, 0), one 3-vertex strip
    return [0x000407FB, 0x00000019, 0x20002000, 0x4000DFFF, 0xDFFF6000, 0, 0]


def test_record_parsing() -> None:
    (paint,) = record(tri_record())
    assert paint.color == (255, 128, 0)
    assert paint.items == [("strip", [(0x2000, 0x2000), (0xDFFF, 0x4000),
                                      (0x6000, 0xDFFF)])]  # fmt: skip
    # 9: two paints
    v3 = [0x20002000, 0x4000DFFF, 0xDFFF6000]
    red, green = record([0x7FB, 0x19, *v3, 0x0007F803, 0x19, *v3, 0, 0])
    assert (red.color, green.color) == ((255, 0, 0), (0, 255, 0))
    # 10: a gradient, its plane, a 4-vertex strip
    (grad,) = record(gradient_record())
    assert grad.stops == [(0, (255, 0, 0, 255)), (0xFFFF, (0, 0, 255, 255))]
    assert grad.plane == (2.0, 0.0, -0.5)
    assert len(grad.items[0][1]) == 4
    # 11: tags 6 and 7 are malformed
    assert record([0x7FB, 0x0E, 0]) is None
    assert record([0x7FB, 0x0F, 0]) is None


def gradient_record(plane=(2.0, 0.0, -0.5)) -> list[int]:
    return [0x14, 0x00FF0000, 0x8000FF00, 0x0000FFFF, 0x8000FFFF, 0x1D,
            *(f(x) for x in plane), 0x21, 0x4000BFFF, 0x40004000, 0xBFFFBFFF,
            0xBFFF4000, 0, 0]  # fmt: skip


def levels(img: np.ndarray) -> np.ndarray:
    return np.round(img * 255).astype(int)


def test_12_triangle() -> None:
    img = levels(svg.render(record(tri_record()), 256, 256))
    covered = (img == [255, 128, 0, 255]).all(axis=2)
    assert covered.sum() == 17440
    assert covered[100, 100]
    assert not covered[10, 10]
    assert (img[~covered] == 0).all()


def strip(*pts: tuple[int, int]) -> list[int]:
    return [len(pts) << 3 | 1, *(x | y << 16 for x, y in pts)]


def test_13_tie_rule() -> None:
    paints = record([0x7FFFFFB, *strip((25728, 128), (128, 128), (25728, 25728),
                                       (128, 25728)), 0, 0])  # fmt: skip
    covered = svg.render(paints, 256, 256)[..., 3] == 1
    assert covered.sum() == 10000
    assert covered[:100, :100].all()


@pytest.mark.parametrize(("size", "want"), [(64, 1714), (256, 27500), (1024, 440000)])
def test_14_snapping(size: int, want: int) -> None:
    paints = record([0x7FFFFFB, *strip((2560, 2560), (63999, 7680), (5120, 61439)), 0])
    assert (svg.render(paints, size, size)[..., 3] == 1).sum() == want


def square(x0: int, x1: int) -> list[int]:
    a, b = round(65535 * x0 / 256), round(65535 * x1 / 256)
    return strip((a, a), (b, a), (a, b), (b, b))


def test_16_painters_order() -> None:
    paints = record([0x7FB, *square(64, 192), 0xFF << 19 | 3, *square(128, 240), 0])
    img = levels(svg.render(paints, 256, 256))
    assert img[128, 128].tolist() == [0, 0, 255, 255]
    assert img[100, 100].tolist() == [255, 0, 0, 255]


def test_17_gray() -> None:
    paints = record([0x03FBFBFB, *square(64, 192), 0])
    img = levels(svg.render(paints, 256, 256))[..., 0]
    assert set(np.unique(img).tolist()) == {0, 127}


def test_19_gradient() -> None:
    img = levels(svg.render(record(gradient_record()), 256, 256))
    row = img[128]
    assert row[64, :3].tolist() == [254, 0, 0]
    assert row[65, :3].tolist() == [252, 0, 2]
    assert row[68, :3].tolist() == [246, 0, 8]
    assert row[191, :3].tolist() == [0, 0, 254]
    assert img[10, 10].tolist() == [0, 0, 0, 0]


def whole(stops: list[tuple[float, tuple[int, int, int]]], plane) -> list[int]:
    words = [len(stops) << 3 | 4]
    for o, (r, g, b) in stops:
        words += [round(o * 65535) | r << 16 | g << 24, b | 0xFF << 8 | 0x8000 << 16]
    return [*words, 0x1D, *(f(x) for x in plane), *square(0, 256), 0]


def test_20_slow_gradients() -> None:
    bw = [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))]
    img = levels(svg.render(record(whole(bw, (0.25, 0.0, 0.0))), 256, 256))
    assert img[7, :, 0].tolist() == [i // 4 for i in range(256)]
    img = levels(svg.render(record(whole(bw, (0.0, 0.1, 0.0))), 256, 256))
    assert img[:, 7, 1].tolist() == [j // 10 for j in range(256)]


def test_21_three_stops() -> None:
    stops = [(0.25, (0, 0, 0)), (0.5, (255, 255, 255)), (0.75, (0, 0, 0))]
    row = levels(svg.render(record(whole(stops, (0.5, 0.0, 0.0))), 256, 256))[5, :, 0]
    assert (row[:128] == 0).all()
    assert row[128:131].tolist() == [1, 2, 4]
    assert row[255] == 253


def test_22_a_nan_plane_is_the_first_stop() -> None:
    nan = struct.unpack("<f", struct.pack("<I", 0xFFC00000))[0]
    img = levels(svg.render(record(gradient_record((nan, nan, nan))), 256, 256))
    assert img[128, 128].tolist() == [255, 0, 0, 255]


def test_23_polyline() -> None:
    paints = record([0x7FFFFFB, 0x12, 0x0A800A00, 0x7840C7FF, 0])
    covered = svg.render(paints, 256, 256)[..., 3] == 1
    assert covered.sum() == 190
    cols = np.nonzero(covered.any(axis=0))[0]
    assert (cols.min(), cols.max()) == (10, 199)
    assert covered[62, 100]
    assert covered[10, 10]
    assert covered[119, 199]


def test_the_split_sign_is_exact() -> None:
    rng = random.Random(3)  # noqa: S311
    lim = (1 << 22) - 1
    for _ in range(20000):
        a, b, c, d = (rng.randint(-lim, lim) for _ in range(4))
        if rng.random() < 0.2:
            c, d = b, a
        want = (a * b > c * d) - (a * b < c * d)
        assert svg.sgn_split(a, b, c, d) == want


# Translation


def test_the_corpus_default_is_the_background() -> None:
    nodes = node_file(0x0009880B, [4, OFF]).nodes()
    mm, _, outs = translate_graph(nodes, {"out": 9}, {}, size=8)
    made = next(n for n in mm if n["name"] == "n9")
    assert made["parameters"]["background"] == {"r": 0.0, "g": 0.0, "b": 0.0, "a": 0.0}
    assert "sbs_svg" not in made["shader_model"]["instance"]
    assert outs == {"out": (outs["out"][0], 0)}


def test_a_drawn_record_in_glsl(tmp_path: Path) -> None:
    helpers, body = svg.glsl(record(tri_record()), "N", "N", color=True)
    assert "ivec2 sbs_svg_snap_$(name)(vec2 q, vec2 n)" in helpers
    assert "const ivec4 svi[1] = ivec4[1](ivec4(0, 1, 2, 0));" in body
    assert "vec2 svn = vec2(N, N);" in body
    assert svg.glsl([], "N", "N", color=True) == ("", "")
    # the whole of it compiles: a gradient, a line and a solid paint
    paints = record([*gradient_record()[:-2], 0x7FFFFFB, 0x12, 0x0A800A00, 0x7840C7FF,
                     0x7FB, *square(0, 64), 0])  # fmt: skip
    helpers, body = svg.glsl(paints, "N", "N", color=True)
    compiles(
        "#version 450\nconst float N = 256.0;\n"
        + svg.SGN_GLSL
        + helpers.replace("$(name)", "x")
        + "layout(location = 0) out vec4 color;\n"
        + "void main() {\n\tvec2 uv = gl_FragCoord.xy / N;\n\tvec4 col = vec4(0.0);\n"
        + body.replace("$(name)", "x")
        + "\n\tcolor = col;\n}\n",
        tmp_path,
    )


def test_a_gray_gradient_is_unsupported() -> None:
    with pytest.raises(UnsupportedError, match="gradient"):
        svg.gray_record(record(gradient_record()))
