"""A node's own random seed (docs/node-seed.md) and gradient map layouts.

docs/gradient-map.md section 5's layout tests too, on synthetic cooks: one
node at offset 0x100, its seed read from $randomseed.
"""

import struct

import pytest

from sbsar2ptex import translate
from sbsar2ptex.fxmap import read_settings
from sbsar2ptex.sbsasm import BASE
from sbsar2ptex.sbsasm import Asm
from sbsar2ptex.sbsasm import Node
from sbsar2ptex.sbsasm import fixed_size
from sbsar2ptex.translate import GraphInput
from sbsar2ptex.translate import UnsupportedError
from sbsar2ptex.translate import _gradient_keys
from sbsar2ptex.translate import node_seed
from sbsar2ptex.translate import node_seed_glsl
from sbsar2ptex.translate import node_shader

AT = 0x100
"""The node's offset (from BASE)."""
SEED_UID, SIZE_UID = 0x00C0FFEE, 0x00C0FFEF
"""$randomseed's and $outputsize's uids."""


def sb(n: int) -> list[int]:
    """SB(N): the seed block $randomseed + N."""
    return [0x0A020003, SEED_UID, 0x00000E00, n & 0xFFFFFFFF, 0x00010A12, 0]


SB0 = [0x0A020001, SEED_UID]
"""The seed block $randomseed."""
SZ = [0x0A420001, SIZE_UID]
"""A size function: load $outputsize (integer, width 2)."""
FLOAT_BLOCK = [0x09000001, 0x3F000000]
"""A float function: the constant 0.5."""


def f32w(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def cook(words: list[int], *, seed: int = 0, size: int = 6) -> Node:
    """Parse an SBAM file whose one node, at offset AT, is ``words``.

    Its graph inputs: $randomseed (an int, ``seed``) and $outputsize (an
    int2, ``size`` twice); the node table and output map lie before it.
    """
    node_at = AT + BASE
    defaults = node_at + 4 * len(words)
    uids = defaults + 12
    end = uids + 4 + 4 + 16
    data = bytearray(end + 16)
    data[:4] = b"SBAM"
    struct.pack_into("<H", data, 6, 9)
    struct.pack_into("<I", data, 0x40, AT)  # the node table
    struct.pack_into("<II", data, 0x44, 0xC0, 0)  # the outputs
    struct.pack_into(f"<{len(words)}I", data, node_at, *words)
    struct.pack_into("<iii", data, defaults, seed, size, size)
    struct.pack_into("<HHI", data, uids, 1, 2, 0x1234)
    struct.pack_into("<4I", data, uids + 8, 4, SEED_UID, 8, SIZE_UID)
    struct.pack_into("<4I", data, end, 1, 0x40 - BASE, 0x44 - BASE, defaults - BASE)
    (node,) = Asm(bytes(data)).nodes()
    return node


def assemble(head: list[int | str], blocks: list[tuple[str, list[int]]]) -> list[int]:
    """Return a node's words: ``head``, then ``blocks`` in order.

    A str in ``head`` is the offset of the block of that name.
    """
    at = AT + 4 * len(head)
    where = {}
    for name, words in blocks:
        where[name] = at
        at += 4 * len(words)
    out = [where[w] if isinstance(w, str) else w for w in head]
    for _, words in blocks:
        out += words
    return out


def seed_at(node: Node, graph_seed: int) -> int:
    return node_seed(node, {SEED_UID: [graph_seed], SIZE_UID: [6, 6]})


@pytest.fixture
def ctx() -> None:
    translate._CTX = translate._Context(
        size=8,
        inputs={
            SEED_UID: GraphInput("$randomseed", 4, [0]),
            SIZE_UID: GraphInput("$outputsize", 8, [6, 6]),
        },
    )


def live_seed(node: Node) -> str:
    """Return node_seed_glsl's GLSL, checked to read the $randomseed knob."""
    glsl = node_seed_glsl(node)
    (name,) = translate._CTX.params
    assert f"${name}" in translate._CTX.helpers[-1]
    return glsl


# docs/node-seed.md section 7


def test_1_state_0_with_a_size_function() -> None:
    node = cook([0x0209880D, 0x110, f32w(0.5), f32w(0.25), *SZ])
    assert node.seed_state == 0
    assert seed_at(node, 0) == seed_at(node, 3) == 0
    assert node.seed_code is None
    assert node.size_code == 0x110 + BASE
    assert node.inline == [f32w(0.5), f32w(0.25)]


def test_2_state_1_with_a_size_function() -> None:
    node = cook([0x0249880D, 5, 0x114, f32w(0.5), f32w(0.25), *SZ])
    assert seed_at(node, 0) == seed_at(node, 7) == 5
    assert node.size_code == 0x114 + BASE
    assert node.inline == [f32w(0.5), f32w(0.25)]


def test_3_an_inline_seed_that_looks_like_an_offset() -> None:
    node = cook([0x0249880D, 0x110, 0x114, f32w(0.5), f32w(0.25), *SZ])
    assert node.seed == 272
    assert node.seed_code is None
    assert node.size_code == 0x114 + BASE
    assert node.code == [0x114 + BASE]


def test_4_a_negative_inline_seed(ctx: None) -> None:
    node = cook([0x0249880D, 0xFFFFFFFE, 0x114, f32w(0.5), f32w(0.25), *SZ])
    assert seed_at(node, 0) == -2
    assert seed_at(node, 0) % 2**32 == 4294967294
    assert node_seed_glsl(node) == "-2"


MATRIX = [f32w(2.0), 0, 0, f32w(2.0)]


def transformation(seed_block: list[int]) -> Node:
    """Test 5's node: a fixed-size transformation, its seed a block."""
    head = [0x03985504, 0x0400005F, 7, "seed", *MATRIX, "fn"]
    return cook(assemble(head, [("fn", FLOAT_BLOCK), ("seed", seed_block)]))


def test_5_state_2_without_a_size_function(ctx: None) -> None:
    node = transformation(sb(1))
    assert node.seed_code == node.words[3] + BASE  # rest[2]
    assert node.fields == [0x0400005F, 7]
    assert node.inputs == {"input": 7}
    assert node.size_code is None
    assert node.inline[:4] == MATRIX  # rest[3..6]
    assert node.inline[4] == node.words[8]  # the function's offset
    assert seed_at(node, 0) == 1
    assert seed_at(node, 10) == 11
    # In the graph: a function of the $randomseed knob, not the number 1.
    glsl = live_seed(node)
    assert glsl.endswith("(uv)")
    assert "float(" not in glsl  # an int, wrapping at 32 bits


def test_6_state_2_with_a_size_function(ctx: None) -> None:
    head = [0x00998802, 0x05, 3, 4, "seed", "size"]
    node = cook(assemble(head, [("size", SZ), ("seed", SB0)]))
    assert node.inputs == {"destination": 3, "source": 4}
    assert node.seed_code == node.words[4] + BASE  # rest[3]
    assert node.size_code == node.words[5] + BASE  # rest[4]
    assert node.code == [node.size_code]
    assert seed_at(node, 0) == 0
    assert seed_at(node, 9) == 9
    live_seed(node)
    translate._CTX.inputs[SIZE_UID] = GraphInput("$outputsize", 8, [9, 9])
    assert translate._output_size(node, 11) == 9


def test_7_a_negative_relative_seed() -> None:
    node = transformation(sb(0xFFFFFFFE))
    assert seed_at(node, 1) == -1
    assert seed_at(node, 1) % 2**32 == 0xFFFFFFFF
    assert seed_at(node, 0) == -2


def test_8_pixel_processor_state_1() -> None:
    node = cook(assemble([0x00598829, 0x00020001, 4, 42, "size"], [("size", SZ)]))
    assert node.inputs == {"input": 4}
    assert node.fields == [0x00020001, 4]
    assert seed_at(node, 0) == 42
    assert node.size_code == node.words[4] + BASE  # rest[3]
    assert node.inline == []


def gradient_9() -> Node:
    """Test 9's gradient: keys, seed and size offsets, the seed's block last."""
    return cook([0x00998800, 5, 2, 0x118, 0x128, 0x120, 0x00000000, 0xFFFFFFFF,
                 *SZ, *sb(3)])  # fmt: skip


def test_9_a_gradient_with_a_seed_block_and_a_size_function(ctx: None) -> None:
    node = gradient_9()
    assert node.inputs == {"input": 5}
    assert _gradient_keys(node, color=False) == [(0.0, [0.0]), (1.0, [1.0])]
    assert node.seed_code == 0x128 + BASE
    assert node.size_code == 0x120 + BASE
    assert seed_at(node, 0) == 3
    assert translate._output_size(node, 11) == 6  # $outputsize, not the seed


def test_10_curve_state_2() -> None:
    # Op 44 is 0x2C (the spec's node word 0x00998844 has op byte 0x44, 68).
    head = [0x0099882C, 3, 0, "points", "seed", "size"]
    node = cook(assemble(head, [("points", [0, 0]), ("size", SZ), ("seed", SB0)]))
    assert node.inputs == {"input": 3}
    assert node.seed_code == node.words[4] + BASE  # rest[3]
    assert node.code == [node.words[3] + BASE, node.size_code]
    assert node.size_code == node.words[5] + BASE
    assert seed_at(node, 5) == 5


TREE = [0x00000008]
"""An FX-Map tree: one Quadrant drawing nothing."""


def fxmap(head: list[int | str], *blocks: tuple[str, list[int]]) -> Node:
    return cook(assemble(head, [("tree", TREE), *blocks]))


def test_11_fxmap_state_0() -> None:
    node = fxmap([0x00198808, 0, "tree", 0x1FF8, "size"], ("size", SZ))
    st = read_settings(node, color=False)
    assert st.patterns == [0x1FF8]
    assert node.inputs == {"pattern0": 0x1FF8}
    assert (st.seed, st.seed_code) == (None, None)
    assert seed_at(node, 3) == 0
    assert st.setup_code == node.size_code == node.words[4] + BASE  # rest[3]
    assert node.seed_code is None


def test_12_fxmap_state_1() -> None:
    node = fxmap([0x00598808, 0, "tree", 0x20BB, 3, "size"], ("size", SZ))
    st = read_settings(node, color=False)
    assert st.patterns == [0x20BB]
    assert node.inputs == {"pattern0": 0x20BB}
    assert st.seed == 3
    assert seed_at(node, 0) == 3
    assert st.setup_code == node.words[5] + BASE


def test_13_fxmap_state_1_without_patterns() -> None:
    node = fxmap([0x00598808, 0, "tree", 42, "size"], ("size", SZ))
    st = read_settings(node, color=False)
    assert st.patterns == []
    assert node.inputs == {}
    assert st.seed == 42
    assert seed_at(node, 0) == 42


def test_14_fxmap_state_2() -> None:
    node = fxmap(
        [0x00998808, 0, "tree", 0x5DD, "seed", "size"], ("size", SZ), ("seed", sb(1))
    )
    st = read_settings(node, color=False)
    assert st.patterns == [0x5DD]
    assert st.seed_code == node.seed_code == node.words[4] + BASE
    assert st.setup_code == node.words[5] + BASE
    assert seed_at(node, 0) == 1
    assert seed_at(node, 4) == 5


@pytest.mark.parametrize("state", [1, 2])
def test_15_layout_invariance(state: int) -> None:
    plain = [0x0209880D, 0x110, f32w(0.5), f32w(0.25), *SZ]
    if state == 1:
        words = [0x0249880D, 9, 0x114, f32w(0.5), f32w(0.25), *SZ]
    else:  # its block appended
        words = [0x0289880D, 0x11C, 0x114, f32w(0.5), f32w(0.25), *SZ, *SB0]
    a, b = cook(plain), cook(words)
    assert (b.fields, b.inline, b.inputs) == (a.fields, a.inline, a.inputs)
    assert b.code == [c + 4 for c in a.code]
    assert b.size_code == a.size_code + 4
    assert b.at(b.size_code, 2) == a.at(a.size_code, 2)
    assert seed_at(b, 0) == (9 if state == 1 else 0)


def test_16_state_3_is_unsupported(ctx: None) -> None:
    node = cook([0x00C9880D, 0x110, f32w(0.5), f32w(0.25), *SZ])
    assert node.unsupported is not None
    with pytest.raises(UnsupportedError):
        node_seed(node, {SEED_UID: [0]})
    with pytest.raises(UnsupportedError):
        node_seed_glsl(node)
    with pytest.raises(UnsupportedError):
        node_shader(node, "n0")
    fx = fxmap([0x00D98808, 0, "tree", 0x5DD, "seed", "size"], ("size", SZ),
               ("seed", SB0))  # fmt: skip
    with pytest.raises(UnsupportedError):
        read_settings(fx, color=False)


def test_a_version_9_warp_with_an_inline_seed() -> None:
    # Section 2.2: flag word (version 9), input, gradient, then the slot. The
    # seed looks like neither a node index nor an offset test must decide by.
    head = [0x0249880E, 0x05, 7, 8, 0x114, "size", f32w(0.5)]
    node = cook(assemble(head, [("size", SZ)]))
    assert node.inputs == {"input": 7, "gradient": 8}
    assert node.seed == 0x114
    assert node.size_code == node.words[5] + BASE
    assert node.inline == [f32w(0.5)]


def test_a_seed_block_of_two_components_is_unsupported() -> None:
    # The slot's block must return one integer: a size function there is
    # a misread layout, not a seed.
    node = transformation(SZ)
    with pytest.raises(UnsupportedError):
        seed_at(node, 0)


def test_an_fxmap_without_a_slot_or_setup_is_unsupported() -> None:
    # State 0, flag bit 0 clear: patterns and inline values can't be told.
    node = fxmap([0x00188808, 0, "tree", 0x1FF8, f32w(0.5)])
    assert node.unsupported is not None
    with pytest.raises(UnsupportedError):
        read_settings(node, color=False)


# docs/gradient-map.md section 5, layout tests


def test_gradient_5_flags_0x0099() -> None:
    keys = [0, 0x8000FFFF, 0x0000FFFF, 0xFFFFFFFF]  # 4 keys of 4 bytes
    head = [0x00998800, 5, 4, "keys", "seed", "size"]
    node = cook(assemble(head, [("keys", keys), ("seed", sb(2)), ("size", SZ)]))
    assert node.code[0] == node.words[3] + BASE  # rest[2]
    assert node.seed_code == node.words[4] + BASE  # rest[3]: one integer
    assert node.size_code == node.words[5] + BASE  # rest[4]: two
    assert len(_gradient_keys(node, color=False)) == 4
    assert seed_at(node, 0) == 2


def test_gradient_6_flags_0x0019(ctx: None) -> None:
    keys = [0, 0x8000FFFF, 0x0000FFFF, 0xFFFFFFFF]
    node = cook(
        assemble([0x00198800, 5, 4, "keys", "size"], [("keys", keys), ("size", SZ)])
    )
    assert node.code == [node.words[3] + BASE, node.words[4] + BASE]
    assert node.seed_code is None
    assert seed_at(node, 5) == 0
    assert node.size_code == node.words[4] + BASE  # rest[3]
    assert len(_gradient_keys(node, color=False)) == 4
    assert translate._output_size(node, 11) == 6


def test_gradient_7_fixed_size() -> None:
    node = cook([0x00188400, 5, 2, 0x110, 0x00000000, 0xFFFFFFFF])
    assert node.code == [0x110 + BASE]
    assert node.size_code is None
    assert _gradient_keys(node, color=False) == [(0.0, [0.0]), (1.0, [1.0])]
    assert fixed_size(node) == (4, 8)  # 16 pixels wide, 256 high


def test_gradient_8_extra_key_bytes_are_malformed() -> None:
    keys = [0, 0x8000FFFF, 0xFFFFFFFF, 0]  # 3 keys (12 bytes), then 4 more
    node = cook(
        assemble([0x00198800, 5, 3, "keys", "size"], [("keys", keys), ("size", SZ)])
    )
    with pytest.raises(UnsupportedError):
        _gradient_keys(node, color=False)
