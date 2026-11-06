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


def seed_of(
    words: list[int],
    variables: dict | None = None,
    *,
    incoming: int = 0,
    number: float = 0.0,
) -> int:
    ctx = {10: [number]}
    v = {} if variables is None else variables
    return fxseed.Seeds().seed(decode(words), {}, ctx, v, incoming=incoming)


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
    for word, want in ((0x12B, 0x40400005), (0x123, 0x40400000)):
        # an Iterate of one pass (output1, a Quadrant inheriting its state).
        # (Output0 would see the incoming state, 2.6: T18.)
        node = tree_node({
            0x100: [word, 0x200, 0x400], 0x200: B_SEED3,
            0x400: [0x00100048, 0x408], 0x408: [0x500], 0x500: OPACITY_RAND,
        })  # fmt: skip
        tree = fxmap.read_tree(node)
        got = leaf_states(monkeypatch, node, tree, seed=5)
        assert got == [(iterate_step(want), "opacity")]
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
        0x100: [0x123, 0x200, 0x400], 0x200: B_INT0,
        0x400: [0x00100048, 0x408], 0x408: [0x500], 0x500: OPACITY_RAND,
    })  # fmt: skip
    tree = fxmap.read_tree(node)
    got = leaf_states(monkeypatch, node, tree, seed=5)
    assert got == [(iterate_step(0), "opacity")]


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


# T13-T18: rand and arithmetic in seed blocks, and Iterate output0.

B_RAND24 = [0x09000003, 0x4B800000, 0x00000532, 0x00010611]
"""const float 16777216.0; rand float (#0); convert int (#1)."""


def test_t13_rand_in_a_seed_block() -> None:
    assert seed_of(B_RAND24) == 0x4B201D31  # (bit 0 used: not 0x4B201D30)
    assert seed_of(B_RAND24, incoming=2) == 0x4B11650B
    thousand = [B_RAND24[0], f32_word(1000.0), *B_RAND24[2:]]
    assert seed_of(thousand) == 0x441C4000
    assert seed_of(thousand, incoming=777) == 0x438F8000
    three = block(
        ("f", 16777216.0), (0x32, FLOAT, 1, [0]), (0x32, FLOAT, 1, [0]),
        (0x32, FLOAT, 1, [0]), (0x11, INT, 1, [3]),
    )  # fmt: skip
    assert seed_of(three) == 0x4AA3514E


def test_t13_rand_seeds_from_the_incoming_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # an Iterate of one pass at incoming state 2: inheriting adds it
    for word, want in ((0x12B, 0x4B11650D), (0x123, 0x4B11650B)):
        node = tree_node({
            0x100: [word, 0x200, 0x400], 0x200: B_RAND24,
            0x400: [0x00100048, 0x408], 0x408: [0x500], 0x500: OPACITY_RAND,
        })  # fmt: skip
        tree = fxmap.read_tree(node)
        got = leaf_states(monkeypatch, node, tree, seed=2)
        assert got == [(iterate_step(want), "opacity")]
        monkeypatch.undo()


def test_t14_the_seed_blocks_generator_is_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # T10's tree, the inner Iterate's seed block setvar r <- rand(1); 0
    with_rand = block(("f", 1.0), (0x32, FLOAT, 1, [0]), (0x07, FLOAT, 1, [1, 1]),
                      ("i", 0))  # fmt: skip
    without = block(("f", 0.5), (0x07, FLOAT, 1, [0, 1]), ("i", 0))
    states = []
    for seed in (with_rand, without):
        node = tree_node({
            0x100: [0x0000030B, 0x200],
            0x200: [0x00000089, 0x600, 0x0, 0x300],
            0x300: [0x00000123, 0x700, 0x400],
            0x400: [0x01000048, 0x408], 0x408: [0x800],
            0x600: B_SETNUM, 0x700: seed, 0x800: FRAME_RAND,
        })  # fmt: skip
        tree = fxmap.read_tree(node)
        states.append(leaf_states(monkeypatch, node, tree))
        monkeypatch.undo()
    assert states[0] == states[1]
    leaf = [s for s, lane in states[0] if lane == "frameoffset"]
    assert leaf == [iterate_step(0)] * 3  # (the effective state is 0)
    # r: the value generator's first value from the inner node's incoming state
    for incoming in (0x000038C5, 0x5C49B922, 0xECB3A657):
        variables: dict = {}
        assert seed_of(with_rand, variables, incoming=incoming) == 0
        first = (incoming * 0x7FF8A3ED + 0x2AA01D31) & 0xFFFFFFFF
        assert variables[1, FLOAT, 1] == [(first & 0xFFFFFF) / (1 << 24)]
    # a second run starts afresh
    variables = {}
    seed_of(with_rand, variables)
    again: dict = {}
    seed_of(with_rand, again)
    assert variables == again


def test_t15_add_adds_patterns() -> None:
    nan = block(("f", 3.0), ("f", 1.0), (0x12, FLOAT, 1, [0, 1]), (0x11, INT, 1, [2]))
    assert seed_of(nan) == 0x7FC00000
    for c, number, want in (
        (2.0, 1.0, 0x7F800000),
        (9.0, 1.0, 0xBF800000),
        (5.0, 1.0, 0x80000000),
        (3.0, 0.0, 0x40400000),
    ):
        words = block(
            (0x01, FLOAT, 1, [10]), ("f", c), (0x12, FLOAT, 1, [0, 1]),
            (0x11, INT, 1, [2]),
        )  # fmt: skip
        assert seed_of(words, number=number) == want, (c, number)


def _int_op(op: int, a: float, b: int | None) -> list[int]:
    head = [("i", a)] if isinstance(a, int) else [("f", a), (0x11, INT, 1, [0])]
    k = len(head)
    if b is None:
        return block(*head, (op, INT, 1, [k - 1]))
    return block(*head, ("i", b), (op, INT, 1, [k - 1, k]))


@pytest.mark.parametrize(
    ("op", "a", "b", "want"),
    [
        (0x13, 7, 3, 0),
        (0x14, 7, 3, 0),
        (0x17, 7, None, 0x80000007),
        (0x13, -5, 3, 0xFFFFFFFB),
        (0x30, -5, 1, 0),
        (0x15, 3.0, 2, 0x7F800000),
        (0x13, 3.0, 1, 0x40400000),
        (0x23, -5, None, 0x7FFFFFFB),
        (0x31, 3.0, 1, 0x40400000),
        (0x31, -5, 1, 0),
        (0x30, 3.0, 1, 0),
    ],
)
def test_t16_float_ops_on_patterns(op: int, a: float, b: int | None, want: int) -> None:
    assert seed_of(_int_op(op, a, b)) == want


@pytest.mark.parametrize(("op", "a", "b"), [(0x15, 7, 2), (0x16, 3.0, 3)])
def test_t16_nans(op: int, a: float, b: int) -> None:
    got = seed_of(_int_op(op, a, b))
    assert got & 0xFFFFFF == 0xC00000
    assert got & 0x7F800000 == 0x7F800000


def test_t16_spec_tables() -> None:
    # section 2.1's decoded low 24 bits: a = convert(x), x 3.0 and 6.5
    def a_then(*rest: tuple, x: float) -> list[int]:
        return block(("f", x), (0x11, INT, 1, [0]), *rest)

    for x, sub1, div2 in ((3.0, 0x400000, 0x800000), (6.5, 0xC00000, 0x800000)):
        assert seed_of(a_then(("i", 1), (0x13, INT, 1, [1, 2]), x=x)) & 0xFFFFFF == sub1
        assert seed_of(a_then(("i", 2), (0x15, INT, 1, [1, 2]), x=x)) & 0xFFFFFF == div2
        assert seed_of(a_then((0x12, INT, 1, [1, 1]), x=x)) & 0xFFFFFF == 0x800000
        ten = a_then(("i", 10), (0x13, INT, 1, [2, 1]), x=x)
        assert seed_of(ten) & 0xFFFFFF == sub1
    # float ops then convert: x - 1.0, x * 2.0, x / 2.0
    for x, want in (
        (3.0, (0, 0xC00000, 0x800000)),
        (6.5, (0xA00000, 0x500000, 0x400000)),
    ):
        for op, c, w in (
            (0x13, 1.0, want[0]),
            (0x14, 2.0, want[1]),
            (0x15, 2.0, want[2]),
        ):
            words = block(
                ("f", x), ("f", c), (op, FLOAT, 1, [0, 1]), (0x11, INT, 1, [2])
            )
            assert seed_of(words) & 0xFFFFFF == w, (x, op)
    # plain integers 7 and -5
    for i, plus3, minus3 in ((7, 0xA, 0), (-5, 0xFFFFFE, 0xFFFFFB)):
        assert seed_of(_int_op(0x12, i, 3)) & 0xFFFFFF == plus3
        assert seed_of(_int_op(0x13, i, 3)) & 0xFFFFFF == minus3


def test_t16_number_read_directly() -> None:
    for p in range(4):
        words = block((0x01, FLOAT, 1, [10]), (0x11, INT, 1, [0]))
        assert seed_of(words, number=float(p)) == f32_word(float(p))
        words = block(
            (0x01, FLOAT, 1, [10]), ("f", 3.0), (0x14, FLOAT, 1, [0, 1]),
            (0x11, INT, 1, [2]),
        )  # fmt: skip
        assert seed_of(words, number=float(p)) == f32_word(3.0 * p)


def test_t17_floor() -> None:
    for x, want in ((-0.5, 0xBF800000), (-8388609.5, 0xCB000002), (2.7, 0x40000000)):
        assert seed_of([B_SEED3[0], f32_word(x), B_SEED3[2]]) == want


def test_t18_iterate_output0_gets_the_incoming_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # an Iterate with inline seed 5, inheriting, count 1, output0 and output1
    node = tree_node({
        0x100: [0x0000015B, 5, 0x400, 0x300],
        0x300: [0x00100048, 0x308], 0x308: [0x500],
        0x400: [0x00100048, 0x408], 0x408: [0x500], 0x500: OPACITY_RAND,
    })  # fmt: skip
    tree = fxmap.read_tree(node)
    assert isinstance(tree, fxmap.Iterate)
    assert (tree.seed, tree.count) == (5, 1)
    assert tree.first is not None
    assert tree.repeated is not None
    got = leaf_states(monkeypatch, node, tree)
    assert got == [(0, "opacity"), (0xBF8B3662, "opacity")]
    assert iterate_step(5) == 0xBF8B3662
    monkeypatch.undo()
    # the deferred (lanes) walk agrees
    st = fxmap.Settings()
    kw = {"color": False, "size": 8}
    want = fxsplat._walk(node, tree, st, {}, defer=False, **kw)
    got_lanes = fxsplat.splats(node, tree, st, {}, **kw)
    assert [s.value for s in got_lanes] == [s.value for s in want]
    assert len(want) == 2


def test_untested_ops_in_seed_blocks_raise() -> None:
    compare = block(("i", 7), ("i", 2), (0x1F, 0, 1, [0, 1]),
                    (0x09, INT, 1, [2, 0, 1]))  # fmt: skip
    with pytest.raises(FunctionError, match="untested"):
        seed_of(compare)
    floor = block(("f", 3.5), (0x24, FLOAT, 1, [0]), (0x11, INT, 1, [1]))
    with pytest.raises(FunctionError, match="untested"):
        seed_of(floor)
    vector = block(("i", 1), (0x0D, INT, 2, [0, 0]), (0x12, INT, 2, [1, 1]),
                   (0x10, INT, 1, [2, 0]))  # fmt: skip
    with pytest.raises(FunctionError, match="untested"):
        seed_of(vector)
