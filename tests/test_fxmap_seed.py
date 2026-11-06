"""FX-Map tree nodes whose seed is a function: docs/fxmap-node-0x09.md T1-T12."""

import struct

import pytest

from sbsar2ptex import fxmap
from sbsar2ptex import fxseed
from sbsar2ptex import fxsplat
from sbsar2ptex.function import FLOAT
from sbsar2ptex.function import INT
from sbsar2ptex.function import FunctionError
from sbsar2ptex.function import decode
from sbsar2ptex.fxrng import FxRng
from sbsar2ptex.fxrng import iterate_step
from sbsar2ptex.sbsasm import BASE
from sbsar2ptex.sbsasm import Node

# The spec's blocks, word for word.
B_SEED3 = [0x09000002, 0x40400000, 0x00000611]
B_SEEDV0 = [0x05040002, 0x06110000, 0x00000000]
B_INT0 = [0x06040001, 0x00000000]
B_ADD1 = [0x09000004, 0x40400000, 0x00000611, 0x00000E00, 0x00000001,
          0x00010A12, 0x00000002]  # fmt: skip
B_SETNUM = [0x05010004, 0x0907000A, 0x00000000, 0x00010440, 0x0001084C, 0x00000002]
B_COUNT2 = [0x0A000001, 0x00000000, 0x00000002]  # const int 2


def f32_word(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def block(*instrs: tuple) -> list[int]:
    """Assemble (op, kind, width, operands), or ("f" or "i", constant), as words."""
    half = [len(instrs)]
    for ins in instrs:
        if ins[0] in ("f", "i"):
            fmt = "<f" if ins[0] == "f" else "<i"
            word = struct.unpack("<I", struct.pack(fmt, ins[1]))[0]
            kind = FLOAT if ins[0] == "f" else INT
            ops = [word & 0xFFFF, word >> 16]
            if (len(half) + 1) % 2:  # a u32 is 4-byte aligned
                ops = [0, *ops]
            half += [len(ops) << 10 | kind << 8, *ops]
            continue
        op, kind, width, operands = ins
        half += [len(operands) << 10 | kind << 8 | (width - 1) << 6 | op, *operands]
    half += [0] * (len(half) % 2)
    return [half[k] | half[k + 1] << 16 for k in range(0, len(half), 2)]


FRAME_RAND = block(
    ("f", 1.0), (0x32, FLOAT, 1, [0]), ("f", 1.0), (0x32, FLOAT, 1, [2]),
    (0x0D, FLOAT, 2, [1, 3]),
)  # fmt: skip
"""A frame offset of two rand(1) draws."""

OPACITY_RAND = block(("f", 1.0), (0x32, FLOAT, 1, [0]))


def tree_node(at: dict[int, list[int]], root: int = 0x100) -> Node:
    """Return an FX-Map node holding words at tree offsets, its root at ``root``."""
    top = max(off + 4 * len(ws) for off, ws in at.items())
    words = [0] * (top // 4)
    words[0], words[2] = 0x03000008, root
    for off, ws in at.items():
        words[off // 4 : off // 4 + len(ws)] = ws
    return Node(index=0, start=BASE, end=BASE + 4 * len(words), op=8, words=words)


def seed_of(words: list[int], variables: dict | None = None) -> int:
    ctx = {10: [0.0]}
    return fxseed.Seeds().seed(decode(words), {}, ctx, variables or {})


def leaf_states(
    monkeypatch: pytest.MonkeyPatch, node: Node, tree: fxmap.Tree, seed: int = 0
) -> list[tuple[int, str]]:
    """Return each FxRng the plain walk starts: (state, lane)."""
    started: list[tuple[int, str]] = []
    real = fxsplat.FxRng

    def record(state: int, lane: str = "frameoffset") -> FxRng:
        started.append((state, lane))
        return real(state, lane)

    monkeypatch.setattr(fxsplat, "FxRng", record)
    st = fxmap.Settings(seed=seed)
    fxsplat._walk(node, tree, st, {}, color=False, size=8, defer=False)
    return started


# T1-T5: parsing.


def test_t1_an_iterate_with_a_seed_function() -> None:
    node = tree_node({
        0x100: [0x000001AB, 0x200, 0x300, 0x400],
        0x200: B_SEED3, 0x300: B_COUNT2, 0x400: [0x00000048],
    })  # fmt: skip
    it = fxmap.read_tree(node)
    assert isinstance(it, fxmap.Iterate)
    assert it.inherited
    assert it.seed is None
    assert it.seed_code == 0x200 + BASE
    assert it.count == fxmap.Param(fxmap.CODE, code=0x300 + BASE)
    assert isinstance(it.repeated, fxmap.Quadrant)
    assert it.repeated.pattern == 0
    assert it.first is None


def test_t2_an_inline_count_with_a_seed_function() -> None:
    node = tree_node({
        0x100: [0x0000062B, 0x200, 0x400], 0x200: B_SEED3, 0x400: [0x48],
    })  # fmt: skip
    it = fxmap.read_tree(node)
    assert isinstance(it, fxmap.Iterate)
    assert it.count == 6
    assert it.seed_code == 0x200 + BASE
    assert isinstance(it.repeated, fxmap.Quadrant)


def test_t3_output0_after_output1() -> None:
    node = tree_node({
        0x100: [0x000001EB, 0x200, 0x300, 0x400, 0x500],
        0x200: B_SEED3, 0x300: B_COUNT2, 0x400: [0x48], 0x500: [0x148],
    })  # fmt: skip
    it = fxmap.read_tree(node)
    assert isinstance(it, fxmap.Iterate)
    assert (it.seed_code, it.count) == (
        0x200 + BASE, fxmap.Param(fxmap.CODE, code=0x300 + BASE),
    )  # fmt: skip
    assert isinstance(it.repeated, fxmap.Quadrant)
    assert isinstance(it.first, fxmap.Quadrant)
    assert (it.repeated.pattern, it.first.pattern) == (0, 1)


def test_t4_a_switch() -> None:
    value = block(("f", 1.0), (0x32, FLOAT, 1, [0]), ("f", 0.5), (0x1F, 0, 1, [1, 2]))
    node = tree_node({
        0x100: [0x000000A9, 0x200, 0x300, 0x400, 0x500],
        0x200: B_SEED3, 0x300: value, 0x400: [0x48], 0x500: [0x148],
    })  # fmt: skip
    sw = fxmap.read_tree(node)
    assert isinstance(sw, fxmap.Switch)
    assert sw.inherited
    assert sw.seed_code == 0x200 + BASE
    assert sw.value == fxmap.Param(fxmap.CODE, code=0x300 + BASE)
    assert isinstance(sw.if_false, fxmap.Quadrant)
    assert isinstance(sw.if_true, fxmap.Quadrant)
    assert (sw.if_false.pattern, sw.if_true.pattern) == (0, 1)
    node = tree_node({
        0x100: [0x00000069, 0x200, 0x400, 0x500],
        0x200: B_SEED3, 0x400: [0x48], 0x500: [0x148],
    })  # fmt: skip
    sw = fxmap.read_tree(node)
    assert isinstance(sw, fxmap.Switch)
    assert sw.value is True
    assert sw.seed_code == 0x200 + BASE
    assert isinstance(sw.if_false, fxmap.Quadrant)
    assert isinstance(sw.if_true, fxmap.Quadrant)
    assert sw.if_true.pattern == 1


def test_t5_a_quadrant() -> None:
    node = tree_node({
        0x100: [0x02000168, 0x108], 0x108: [0x200, 0x3E99999A, 0x3E99999A],
        0x200: B_SEED3,
    })  # fmt: skip
    quad = fxmap.read_tree(node)
    assert isinstance(quad, fxmap.Quadrant)
    assert quad.pattern == 1
    assert quad.seed is None
    assert quad.seed_code == 0x200 + BASE
    size = quad.params["patternsize"]
    assert size.state == fxmap.INLINE
    assert size.value == pytest.approx([0.3, 0.3])
    assert quad.children == []


# T6-T9: seed values.


def test_t6_a_seed_from_a_float_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    assert seed_of(B_SEED3) == 0x40400000
    for word, want in ((0x6B, 0x40400005), (0x63, 0x40400000)):
        # an Iterate drawing output0 (a Quadrant inheriting its state), count 0
        node = tree_node({
            0x100: [word, 0x200, 0, 0x400], 0x200: B_SEED3,
            0x400: [0x00100048, 0x408], 0x408: [0x500], 0x500: OPACITY_RAND,
        })  # fmt: skip
        tree = fxmap.read_tree(node)
        assert leaf_states(monkeypatch, node, tree, seed=5) == [(want, "opacity")]
        monkeypatch.undo()
    assert FxRng(0x40400000, "frameoffset").rand(1.0) == pytest.approx(
        0.25302231, abs=1e-8
    )


def test_t7_integer_add_on_patterns() -> None:
    assert seed_of(B_ADD1) == 0x40400001
    four = block(("f", 4.0), (0x11, INT, 1, [0]))
    assert seed_of(four) == 0x40800000 != seed_of(B_ADD1)
    for x, n, want in ((16777216.0, 3, 0x4B800003), (3.0, 16777216, 0x41400000)):
        words = list(B_ADD1)
        words[1], words[4] = f32_word(x), n
        assert seed_of(words) == want


def test_t8_floor() -> None:
    for x, want in ((2.04, 0x000000), (-2.49996, 0x400000)):
        words = [B_SEED3[0], f32_word(x), B_SEED3[2]]
        assert seed_of(words) & 0xFFFFFF == want
    assert seed_of([B_SEED3[0], f32_word(-2.49996), B_SEED3[2]]) == 0xC0400000
    assert seed_of([B_SEED3[0], f32_word(1.02), B_SEED3[2]]) == 0x3F800000


def test_t9_an_unset_integer_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    assert seed_of(B_INT0) == 0
    node = tree_node({
        0x100: [0x63, 0x200, 0, 0x400], 0x200: B_INT0,
        0x400: [0x00100048, 0x408], 0x408: [0x500], 0x500: OPACITY_RAND,
    })  # fmt: skip
    tree = fxmap.read_tree(node)
    assert leaf_states(monkeypatch, node, tree, seed=5) == [(0, "opacity")]


# T10-T11: per-visit evaluation and order.


def t10_node() -> Node:
    return tree_node({
        0x100: [0x0000030B, 0x200],
        0x200: [0x00000089, 0x600, 0x0, 0x300],
        0x300: [0x00000123, 0x700, 0x400],
        0x400: [0x01000048, 0x408], 0x408: [0x800],  # (frame offset: FRAME_RAND)
        0x600: B_SETNUM, 0x700: B_SEEDV0, 0x800: FRAME_RAND,
    })  # fmt: skip


def test_t10_per_visit_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    node = t10_node()
    tree = fxmap.read_tree(node)
    assert isinstance(tree, fxmap.Iterate)
    started = leaf_states(monkeypatch, node, tree)
    entered = [s for s, lane in started if lane == "switch"]
    assert entered == [0x000038C5, 0x5C49B922, 0xECB3A657]
    leaf = [s for s, lane in started if lane == "frameoffset"]
    assert leaf == [0x000038C5, 0xE38038C5, 0x400038C5]
    effective = [0x00000000, 0x3F800000, 0x40000000]
    assert [iterate_step(e) for e in effective] == leaf
    monkeypatch.undo()
    # the splats, the deferred walk's as the plain walk's
    st = fxmap.Settings()
    kw = {"color": False, "size": 8}
    want = fxsplat._walk(node, tree, st, {}, defer=False, **kw)
    got = fxsplat.splats(node, tree, st, {}, **kw)
    pairs = [(s.x - 0.5, s.y - 0.5) for s in want]
    assert pairs[0] == pytest.approx((0.95981741, 0.48948157), abs=1e-7)
    assert pairs[1] == pytest.approx((0.45981741, 0.98948157), abs=1e-7)
    assert pairs[2] == pairs[0]
    assert [(s.x, s.y, s.hx, s.value) for s in got] == [
        (s.x, s.y, s.hx, s.value) for s in want
    ]


def test_t11_the_seed_runs_before_the_count() -> None:
    seed = block(("f", 6.0), (0x07, FLOAT, 1, [0, 0]), ("i", 0))
    count = block((0x04, FLOAT, 1, [0]), (0x11, INT, 1, [0]))
    node = tree_node({
        0x100: [0x000000AB, 0x200, 0x300, 0x400],
        0x200: seed, 0x300: count, 0x400: [0x01000048, 0x408], 0x408: [0x800],
        0x800: FRAME_RAND,
    })  # fmt: skip
    tree = fxmap.read_tree(node)
    spl = fxsplat.splats(node, tree, fxmap.Settings(), {}, color=False, size=8)
    assert len(spl) == 6


def test_a_constant_seed_function_seeds_every_pass_alike(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # an outer Iterate (count 3) of an Iterate (count 2, not inheriting, its
    # seed a constant function): every inner pass at the same states
    node = tree_node({
        0x100: [0x0000030B, 0x200], 0x200: [0x00000223, 0x700, 0x400],
        0x400: [0x01000048, 0x408], 0x408: [0x800],
        0x700: B_SEED3, 0x800: FRAME_RAND,
    })  # fmt: skip
    tree = fxmap.read_tree(node)
    leaf = [s for s, _ in leaf_states(monkeypatch, node, tree)]
    one = [iterate_step(0x40400000), iterate_step(iterate_step(0x40400000))]
    assert leaf == one * 3


# T12: rejections.


@pytest.mark.parametrize("word", [0x000000BB, 0x00000039, 0x00000038])
def test_t12_seed_state_3_is_malformed(word: int) -> None:
    node = tree_node({
        0x100: [word, 0x200, 0x300, 0x400, 0x500],
        0x200: B_SEED3, 0x300: B_COUNT2, 0x400: [0x48], 0x500: [0x48],
    })  # fmt: skip
    with pytest.raises(ValueError, match="seed state 3"):
        fxmap.read_tree(node)


@pytest.mark.parametrize("slot", [0x202, 0x4000, 0x0])
def test_t12_a_seed_slot_not_an_offset_is_malformed(slot: int) -> None:
    node = tree_node({0x100: [0x0000062B, slot, 0x400], 0x200: B_SEED3, 0x400: [0x48]})
    with pytest.raises(ValueError, match="not a block"):
        fxmap.read_tree(node)


def test_t12_a_seed_function_not_an_int_is_malformed() -> None:
    floats = block(("f", 3.0))
    vec = block(("f", 3.0), (0x11, INT, 2, [0]))
    for words in (floats, vec):
        node = tree_node(
            {0x100: [0x0000062B, 0x200, 0x400], 0x200: words, 0x400: [0x48]}
        )
        with pytest.raises(ValueError, match="not an int"):
            fxmap.read_tree(node)


def test_t12_a_code_block_reached_as_a_node_is_an_error() -> None:
    node = tree_node({0x100: [0x0000000B, 0x300], 0x300: [0x09000010, 0, 0]})
    with pytest.raises(ValueError, match="unknown FX-Map node word 0x09000010"):
        fxmap.read_tree(node)
    # read without the seed slot, the T1 Iterate's count block would be output1
    node = tree_node({
        0x100: [0x000001AB, 0x200, 0x300, 0x400],
        0x200: B_SEED3, 0x300: [0x09000010, 0, 0], 0x400: [0x48],
    })  # fmt: skip
    assert isinstance(fxmap.read_tree(node), fxmap.Iterate)


def test_untested_integer_ops_on_float_patterns_raise() -> None:
    sub = block(("f", 3.0), (0x11, INT, 1, [0]), ("i", 1), (0x13, INT, 1, [1, 2]))
    with pytest.raises(FunctionError, match="untested"):
        seed_of(sub)
    plain = block(("i", 7), ("i", 2), (0x13, INT, 1, [0, 1]))
    assert seed_of(plain) == 5
    with pytest.raises(FunctionError):  # rand: its generator is untested
        seed_of(block(("f", 1.0), (0x32, FLOAT, 1, [0]), (0x11, INT, 1, [1])))
