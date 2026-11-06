"""The neighborhood filters' parameter words (docs/node-params.md section 5).

Synthetic nodes at offset 0x100 (test_seed.cook), each block a well-formed
one after the node's words.
"""

import pytest
from test_seed import SB0
from test_seed import SZ
from test_seed import assemble
from test_seed import cook
from test_seed import f32w

from sbsar2ptex import translate
from sbsar2ptex.function import decode
from sbsar2ptex.function import evaluate
from sbsar2ptex.sbsasm import BASE
from sbsar2ptex.sbsasm import Node
from sbsar2ptex.translate import UnsupportedError
from sbsar2ptex.translate import flag_params
from sbsar2ptex.translate import word_params

PS = [0x11400001, f32w(1.0), f32w(1.0)]
"""A float2 block: the constant (1, 1)."""
F1 = [0x09000001, f32w(0.5)]
"""A float block: 0.5."""
F4 = [0x21C00001, f32w(0.1), f32w(0.2), f32w(0.3), f32w(1.0)]
"""A float4 block."""
B = [0x04000001, 1]
"""A bool block: true."""
BLOCKS = {"SZ": SZ, "PS": PS, "F1": F1, "F4": F4, "B": B, "SB0": SB0}


def node(*head: str | float) -> Node:
    """Cook a node: ``head`` its words (str a block's name, float an f32)."""
    words = [f32w(w) if isinstance(w, float) else w for w in head]
    names = [w for w in words if isinstance(w, str)]
    return cook(assemble(words, [(n, BLOCKS[n]) for n in names]))


def word(n: Node, k: int) -> int:
    """Return the file offset ``rest[k]`` points at."""
    return n.words[1 + k] + BASE


def value(n: Node, at: int) -> list:
    """Evaluate the block at file offset ``at``."""
    return evaluate(decode(n.at(at, (n.end - at) // 4)), {})


@pytest.fixture
def ctx() -> None:
    translate._CTX = translate._Context(size=8, inputs={})


def test_1_blur_intensity_function(ctx: None) -> None:
    n = node(0x2B198814, 7, "SZ", "PS", "F1")
    assert n.unsupported is None
    assert n.size_code == word(n, 1)
    assert n.pixel_size_code == word(n, 2)
    assert n.params == {"intensity": (2, word(n, 3))}
    assert n.inline == [n.words[4]]
    p = word_params(n, [("intensity", 4, 10.0)], {})
    assert p["intensity"].static == [0.5]


def test_2_blur_constant_intensity() -> None:
    n = node(0x1B198814, 7, "SZ", "PS", 7.25)
    assert n.unsupported is None
    assert n.pixel_size_code == word(n, 2)
    assert word_params(n, [("intensity", 4, 10.0)], {}) == {"intensity": 7.25}


def test_3_blur_default_pixel_size() -> None:
    n = node(0x13198814, 7, "SZ", 7.25)
    assert n.unsupported is None
    assert n.code == [n.size_code]
    assert n.pixel_size is None
    assert n.pixel_size_code is None
    assert word_params(n, [("intensity", 4, 10.0)], {}) == {"intensity": 7.25}


def test_4_blur_inline_pixel_size() -> None:
    n = node(0x17198814, 7, "SZ", 2.0, 3.0, 7.25)
    assert n.unsupported is None
    assert n.pixel_size == [2.0, 3.0]
    assert word_params(n, [("intensity", 4, 10.0)], {}) == {"intensity": 7.25}


def test_5_blur_without_a_size_function() -> None:
    n = node(0x13188814, 7, 7.25)
    assert n.unsupported is None
    assert n.code == []
    assert word_params(n, [("intensity", 4, 10.0)], {}) == {"intensity": 7.25}


def test_6_blur_with_a_seed_block() -> None:
    n = node(0x2B998814, 7, "SB0", "SZ", "PS", "F1")
    assert n.unsupported is None
    assert n.seed_code == word(n, 1)
    assert n.size_code == word(n, 2)
    assert n.pixel_size_code == word(n, 3)
    assert n.params["intensity"] == (2, word(n, 4))


def test_7_tiling_adds_no_word() -> None:
    n = node(0x10198814, 7, "SZ", 7.25)
    assert n.unsupported is None
    assert word_params(n, [("intensity", 4, 10.0)], {}) == {"intensity": 7.25}


def test_8_warp_v9_constant_intensity() -> None:
    n = node(0x2B19880E, 0, 3, 4, "SZ", "PS", 1.25)
    assert n.unsupported is None
    assert n.inputs == {"input": 3, "gradient": 4}
    assert n.params == {"intensity": (1, [1.25])}
    assert word_params(n, [("intensity", 5, 1.0)], {}) == {"intensity": 1.25}


def test_9_warp_v9_intensity_function(ctx: None) -> None:
    n = node(0x4B19880E, 0, 3, 4, "SZ", "PS", "F1")
    assert n.unsupported is None
    assert n.params == {"intensity": (2, word(n, 5))}
    assert word_params(n, [("intensity", 5, 1.0)], {})["intensity"].static == [0.5]


def test_10_warp_v9_default_pixel_size() -> None:
    n = node(0x2319880E, 0, 3, 4, "SZ", 1.25)
    assert n.unsupported is None
    assert n.code == [n.size_code]
    assert word_params(n, [("intensity", 5, 1.0)], {}) == {"intensity": 1.25}


def test_11_directional_motion_blur(ctx: None) -> None:
    n = node(0x0B198816, 0x9, 7, "SZ", "PS", 7.25, "F1")
    assert n.unsupported is None
    assert n.params == {"intensity": (1, [7.25]), "angle": (2, word(n, 5))}
    spec = [("intensity", 0, 10.0), ("angle", 2, 0.0)]
    p = flag_params(n.fields[0], n.inline, spec, 1, n)
    assert p["intensity"] == 7.25
    assert p["angle"].static == [0.5]


def test_12_directional_warp(ctx: None) -> None:
    n = node(0x03198818, 0x12, 7, 8, "SZ", 7.25, "F1")
    assert n.unsupported is None
    assert n.inputs == {"input": 7, "intensity": 8}
    spec = [("intensity", 1, 10.0), ("angle", 3, 0.0)]
    p = flag_params(n.fields[0], n.inline, spec, 1, n)
    assert p["intensity"] == 7.25
    assert p["angle"].static == [0.5]


def test_13_emboss_colour() -> None:
    n = node(0x0B198811, 0xCA, 7, 8, "SZ", "PS", 7.25, 0.3, "F4", 0.1, 0.2, 0.3, 1.0)
    assert n.unsupported is None
    assert n.params["intensity"] == (1, [7.25])
    assert n.params["angle"] == (1, [pytest.approx(0.3)])
    assert n.params["highlight"][0] == 2
    assert value(n, n.params["highlight"][1]) == pytest.approx([0.1, 0.2, 0.3, 1.0])
    assert n.params["shadow"] == (1, pytest.approx([0.1, 0.2, 0.3, 1.0]))
    # grayscale: the same F, a constant shadow of one f32
    g = node(0x0B198810, 0xCA, 7, 8, "SZ", "PS", 7.25, 0.3, "F1", 0.2)
    assert g.unsupported is None
    assert g.params["shadow"] == (1, [pytest.approx(0.2)])


def test_14_normal() -> None:
    n = node(0x03198825, 0x19, 7, "SZ", 3.0, "B")
    assert n.unsupported is None
    assert n.params == {
        "intensity": (1, [3.0]),
        "inverty": (2, word(n, 4)),
        "alpha": (1, None),
    }
    assert value(n, n.params["inverty"][1]) == [True]


def test_15_distance() -> None:
    n = node(0x0B19882A, 0xB, 5, 6, "SZ", "PS", "F1")
    assert n.unsupported is None
    assert n.inputs == {"mask": 5, "source": 6}
    assert n.params == {"distance": (2, word(n, 5))}


def test_16_state_3_is_unsupported(ctx: None) -> None:
    n = node(0x1F198814, 7, "SZ", 7.25)
    assert n.unsupported is not None
    assert "unsupported" in n.unsupported


def test_17_region_mismatch_is_malformed() -> None:
    n = node(0x1B198814, 7, "SZ", "PS", 7.25, 9.5)
    assert n.unsupported is not None
    assert "malformed" in n.unsupported
    assert f32w(9.5) not in n.inline


@pytest.mark.parametrize(
    "head",
    [
        (0x1F198814, 7, "SZ", 7.25),  # test 16
        (0x1B198814, 7, "SZ", "PS", 7.25, 9.5),  # test 17
    ],
)
def test_an_unsupported_node_is_not_translated(ctx: None, head: tuple) -> None:
    with pytest.raises(UnsupportedError):
        translate.node_shader(node(*head), "n0")
