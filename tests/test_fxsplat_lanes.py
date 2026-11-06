"""fxsplat's vectorized paths against the scalar ones, bit for bit.

splats (deferred subtrees drawn as lanes) against the plain walk, on
synthetic FX-Maps; packing, raster and pixel work against their
splat-at-a-time originals (fxsplat_scalar), on random splats.
"""

import struct

import fxsplat_scalar
import numpy as np
import pytest

from sbsar2ptex import fxdraw
from sbsar2ptex import fxmap
from sbsar2ptex import fxsplat
from sbsar2ptex.function import BOOL
from sbsar2ptex.function import FLOAT
from sbsar2ptex.function import INT
from sbsar2ptex.sbsasm import Node

# FX-Maps: code blocks assembled into a node's words, trees built directly.


class Asm:
    """Assemble code blocks into one node's words; offsets are file offsets."""

    def __init__(self) -> None:
        """Start with no blocks."""
        self.words: list[int] = []

    def block(self, *instrs: tuple) -> int:
        """Add a block of (op, kind, width, operands), or ("f" or "i", constant)."""
        half = [len(instrs)]
        for ins in instrs:
            if ins[0] in ("f", "i"):
                fmt = "<f" if ins[0] == "f" else "<i"
                word = struct.unpack("<I", struct.pack(fmt, ins[1]))[0]
                kind = FLOAT if ins[0] == "f" else INT
                ops = [word & 0xFFFF, word >> 16]
                if (len(half) + 1) % 2:  # a u32 is 4-byte aligned
                    ops = [0, *ops]
                half += [len(ops) << 10 | kind << 8 | 0x00, *ops]
                continue
            op, kind, width, operands = ins
            half += [len(operands) << 10 | kind << 8 | (width - 1) << 6 | op, *operands]
        half += [0] * (len(half) % 2)
        offset = 4 * len(self.words)
        self.words += [half[k] | half[k + 1] << 16 for k in range(0, len(half), 2)]
        return offset

    def node(self) -> Node:
        """Return a node holding the blocks."""
        return Node(index=0, start=0, end=4 * len(self.words), words=self.words)


def code(offset: int) -> fxmap.Param:
    return fxmap.Param(fxmap.CODE, code=offset)


def bits(spl: list[fxsplat.Splat]) -> list[tuple]:
    def b(x: float) -> bytes:
        return struct.pack("<d", x)

    return [
        (
            b(s.x), b(s.y), b(s.hx), b(s.hy), b(s.angle), s.pattern, b(s.suppl),
            s.blend_max, tuple(b(v) for v in s.value),
            tuple(type(v) for v in (s.x, s.y, s.hx, s.hy, s.angle, s.suppl, *s.value)),
            type(s.pattern), type(s.blend_max),
        )
        for s in spl
    ]  # fmt: skip


def splats_both(node: Node, tree: fxmap.Tree, st: fxmap.Settings, **kw: object) -> list:
    """Return splats, checking the deferred walk drew them as the plain walk does."""
    kw = {"color": False, "size": 8, **kw}
    want = fxsplat._walk(node, tree, st, {}, defer=False, **kw)
    got = fxsplat._walk(node, tree, st, {}, defer=True, **kw)
    assert bits(got) == bits(want)
    assert bits(fxsplat.splats(node, tree, st, {}, **kw)) == bits(want)
    return got


def quad_blocks(asm: Asm, var: int | None = None) -> dict[str, fxmap.Param]:
    """Return a Quadrant's functions of rand, $pos, $depth, $number (and ``var``)."""
    read = [(0x04, FLOAT, 1, [var])] if var is not None else [("f", 0.0)]
    return {
        "opacity": code(asm.block(
            ("f", 1.0), (0x32, FLOAT, 1, [0]), (0x01, FLOAT, 1, [12]),
            (0x27, FLOAT, 1, [2]), (0x14, FLOAT, 1, [1, 3]), (0x23, FLOAT, 1, [4]),
            ("f", 0.1), (0x12, FLOAT, 1, [5, 6]),
        )),
        "branchoffset": code(asm.block(
            ("f", 0.05), (0x32, FLOAT, 1, [0]), *read, (0x12, FLOAT, 1, [1, 2]),
            (0x01, FLOAT, 1, [10]), ("f", 0.01), (0x14, FLOAT, 1, [4, 5]),
            (0x0D, FLOAT, 2, [3, 6]),
        )),
        "patternsize": code(asm.block(
            ("f", 0.5), (0x32, FLOAT, 1, [0]), ("f", 0.5), (0x12, FLOAT, 1, [1, 2]),
        )),
        "patternrotation": code(asm.block(
            (0x01, FLOAT, 2, [8]), (0x10, FLOAT, 1, [0, 0]), ("f", 3.0),
            (0x14, FLOAT, 1, [1, 2]), (0x07, FLOAT, 1, [3, 40]),
            (0x04, FLOAT, 1, [40]), (0x26, FLOAT, 1, [5]),
        )),  # (sets and reads variable 40 itself: local)
        "patternsuppl": fxmap.Param(fxmap.INLINE, [0.25]),
    }  # fmt: skip


def test_a_stateless_tree_draws_as_lanes() -> None:
    asm = Asm()
    params = quad_blocks(asm)
    tip = fxmap.Quadrant(pattern=3, params=params)
    leaf = fxmap.Quadrant(pattern=7, params=params, children=[tip, tip, None, tip])
    mid = fxmap.Quadrant(pattern=2, params=params, children=[leaf] * 4)
    other = fxmap.Quadrant(pattern=5, blend_max=True, params=params, seed=77)
    count = code(asm.block(("f", 12.0), (0x32, FLOAT, 1, [0]), ("i", 1),
                           (0x11, INT, 1, [1]), (0x12, INT, 1, [3, 2])))  # fmt: skip
    switch = code(asm.block(("f", 1.0), (0x32, FLOAT, 1, [0]), ("f", 0.4),
                            (0x1F, BOOL, 1, [1, 2])))  # fmt: skip
    pick = fxmap.Switch(value=switch, if_true=mid, if_false=other, seed=5)
    loop = fxmap.Iterate(count=count, seed=9, first=other, repeated=pick)
    root = fxmap.Quadrant(pattern=1, params=params, children=[loop, None, mid, other])
    st = fxmap.Settings(seed=1234, hurst=0.7, opacity=0.8)
    node = asm.node()
    assert id(root) in fxsplat._deferrable(node, root, st)
    gray = splats_both(node, root, st)
    assert len(gray) > 60
    assert splats_both(node, root, st, color=True)
    # drawn as lanes, the splats stay arrays: packing and raster read them so
    assert isinstance(gray, fxsplat.SplatTable)
    assert not fxsplat.has_images(gray)
    assert bits(gray[3:5]) == bits([gray[3], gray[4]]) == bits(list(gray)[3:5])
    want = fxsplat_scalar.pack(list(gray), color=False, tile=(True, True))
    got = fxsplat.pack(gray, color=False, tile=(True, True))
    assert got.image.tobytes() == want.image.tobytes()
    want_raster = fxsplat_scalar.raster(list(gray), (True, True))
    assert want_raster is not None
    got_raster = fxsplat.raster(gray, (True, True))
    expected = struct.pack(f"<{len(want_raster)}f", *want_raster)
    assert fxsplat.float_bytes(got_raster) == expected


def test_too_many_splats_drawn_as_lanes_is_too_big(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asm = Asm()
    params = quad_blocks(asm)
    leaf = fxmap.Quadrant(pattern=3, params=params)
    root = fxmap.Quadrant(pattern=2, params=params, children=[leaf] * 4)
    st = fxmap.Settings(seed=8)
    node = asm.node()
    monkeypatch.setattr(fxsplat, "MAX_SPLATS", 3)
    with pytest.raises(fxsplat.TooBig):
        fxsplat._walk(node, root, st, {}, color=False, size=8, defer=False)
    # found with every function run: it stands, no plain walk after it
    plain = fxsplat._walk
    monkeypatch.setattr(
        fxsplat,
        "_walk",
        lambda *a, defer, **k: (
            plain(*a, defer=defer, **k) if defer else pytest.fail("walked again")
        ),
    )
    with pytest.raises(fxsplat.TooBig):
        fxsplat.splats(node, root, st, {}, color=False, size=8)


def stateful_tree(asm: Asm) -> tuple[fxmap.Tree, ...]:
    """Return (root, sub, noisy, pick): a walk ``sub`` reads the state of."""
    # Iterate: count 40, setting v5 = 0.25; each pass's Switch reads v5,
    # adds rand, keeps its fraction in v5 and v6 = 0.3 v5, true past 0.3;
    # the Quadrant subtree reads v6.
    count = code(asm.block(("f", 0.25), (0x07, FLOAT, 1, [0, 5]), ("i", 40)))
    switch = asm.block(
        (0x04, FLOAT, 1, [5]), ("f", 1.0), (0x32, FLOAT, 1, [1]),
        (0x12, FLOAT, 1, [0, 2]), ("f", 1.0), (0x16, FLOAT, 1, [3, 4]),
        (0x07, FLOAT, 1, [5, 5]), ("f", 0.3), (0x14, FLOAT, 1, [6, 7]),
        (0x07, FLOAT, 1, [8, 6]), (0x1F, BOOL, 1, [6, 7]),
    )  # fmt: skip
    params = quad_blocks(asm, var=6)
    leaf = fxmap.Quadrant(pattern=4, params=params)
    sub = fxmap.Quadrant(pattern=2, params=params, children=[leaf, None, leaf, leaf])
    # a subtree that sets v5 (which the Switch reads first): it must not wait
    setter = code(asm.block(("f", 0.5), (0x07, FLOAT, 1, [0, 5]), ("f", 1.0)))
    noisy = fxmap.Quadrant(pattern=6, params={"opacity": setter})
    pick = fxmap.Switch(value=code(switch), if_true=sub, if_false=noisy)
    return fxmap.Iterate(count=count, repeated=pick), sub, noisy, pick


def test_a_stateful_walk_defers_its_stateless_subtrees() -> None:
    asm = Asm()
    root, sub, noisy, pick = stateful_tree(asm)
    st = fxmap.Settings(seed=99)
    node = asm.node()
    later = fxsplat._deferrable(node, root, st)
    assert id(sub) in later
    assert later[id(sub)] == [(6, FLOAT, 1)]  # v6, a float: (id, type, width)
    assert id(noisy) not in later
    assert id(pick) not in later
    assert id(root) not in later
    assert len(splats_both(node, root, st)) > 40


def test_deferred_subtrees_past_lanes_max_run_in_smaller_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asm = Asm()
    root = stateful_tree(asm)[0]
    st = fxmap.Settings(seed=99)
    node = asm.node()
    # one entry spreads to 4 lanes: runs of entries are halved till they fit
    monkeypatch.setattr(fxsplat, "LANES_MAX", 9)
    assert len(splats_both(node, root, st)) > 40
    # one entry alone spreads further: the plain walk draws
    monkeypatch.setattr(fxsplat, "LANES_MAX", 3)
    kw = {"color": False, "size": 8}
    with pytest.raises(fxsplat._Spread):
        fxsplat._walk(node, root, st, {}, defer=True, **kw)
    want = fxsplat._walk(node, root, st, {}, defer=False, **kw)
    assert bits(fxsplat.splats(node, root, st, {}, **kw)) == bits(want)


def test_deferred_runs_past_max_splats_are_too_big_without_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asm = Asm()
    root = stateful_tree(asm)[0]
    st = fxmap.Settings(seed=99)
    node = asm.node()
    monkeypatch.setattr(fxsplat, "LANES_MAX", 9)
    lanes: list[int] = []
    evaluate_lanes = fxsplat.evaluate_lanes
    monkeypatch.setattr(
        fxsplat,
        "evaluate_lanes",
        lambda *a, **k: lanes.append(a[4]) or evaluate_lanes(*a, **k),
    )
    fxsplat._walk(node, root, st, {}, color=False, size=8, defer=True)
    every = sum(lanes)
    lanes.clear()
    monkeypatch.setattr(fxsplat, "MAX_SPLATS", 60)
    with pytest.raises(fxsplat.TooBig):
        fxsplat._walk(node, root, st, {}, color=False, size=8, defer=False)
    with pytest.raises(fxsplat._Counted):
        fxsplat._walk(node, root, st, {}, color=False, size=8, defer=True)
    # it stopped once the runs so far drew past MAX_SPLATS
    assert 0 < sum(lanes) < every
    # TooBig, or the same splats, as the plain walk, whatever MAX_SPLATS
    kw = {"color": False, "size": 8}
    for most in range(0, 130, 3):
        monkeypatch.setattr(fxsplat, "MAX_SPLATS", most)
        try:
            want = bits(fxsplat._walk(node, root, st, {}, defer=False, **kw))
        except fxsplat.TooBig:
            want = "TooBig"
        try:
            got = bits(fxsplat.splats(node, root, st, {}, **kw))
        except fxsplat.TooBig:
            got = "TooBig"
        assert got == want, most


def test_seeded_nodes_walk_and_their_subtrees_defer() -> None:
    # A stateless tree but for nodes whose seeds are functions (of $number,
    # of $depth, a constant): those and their ancestors are walked, the
    # subtrees below them drawn as lanes, bit for bit as the plain walk.
    asm = Asm()
    params = quad_blocks(asm)
    by_number = asm.block((0x01, FLOAT, 1, [10]), ("f", 3.5), (0x12, FLOAT, 1, [0, 1]),
                          (0x11, INT, 1, [2]))  # fmt: skip
    by_depth = asm.block((0x01, FLOAT, 1, [12]), (0x11, INT, 1, [0]), ("i", 9),
                         (0x12, INT, 1, [1, 2]))  # fmt: skip
    const = asm.block(("f", 3.0), (0x11, INT, 1, [0]))
    tip = fxmap.Quadrant(pattern=3, params=params)
    leaf = fxmap.Quadrant(pattern=7, params=params, children=[tip, tip, None, tip])
    seeded_quad = fxmap.Quadrant(
        pattern=2, params=params, seed_code=by_depth, children=[leaf] * 4
    )
    switch = code(asm.block(("f", 1.0), (0x32, FLOAT, 1, [0]), ("f", 0.4),
                            (0x1F, BOOL, 1, [1, 2])))  # fmt: skip
    pick = fxmap.Switch(
        value=switch, seed_code=const, inherited=False, if_true=seeded_quad,
        if_false=leaf,
    )  # fmt: skip
    inner = fxmap.Iterate(count=3, seed_code=by_number, repeated=pick, first=tip)
    loop = fxmap.Iterate(count=4, seed=9, repeated=inner)
    root = fxmap.Quadrant(pattern=1, params=params, children=[loop, None, leaf, tip])
    st = fxmap.Settings(seed=1234, hurst=0.7, opacity=0.8)
    node = asm.node()
    later = fxsplat._deferrable(node, root, st)
    for walked in (root, loop, inner, pick, seeded_quad):
        assert id(walked) not in later
    assert id(leaf) in later
    assert id(tip) in later
    gray = splats_both(node, root, st)
    assert len(gray) > 40
    assert isinstance(gray, fxsplat.SplatTable)
    assert splats_both(node, root, st, color=True)


# Packing, raster and pixel work.


def random_splats(seed: int, n: int, *, color: bool = False) -> list[fxsplat.Splat]:
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        size = rng.choice([0.0, 0.002, 0.03, 0.2, 0.7, 1.3])
        value = [float(np.float32(v)) for v in rng.uniform(0, 1, 4 if color else 1)]
        out.append(
            fxsplat.Splat(
                float(rng.uniform(-0.6, 1.6)),
                float(rng.uniform(-0.6, 1.6)),
                float(size * rng.uniform(0.2, 1)),
                float(size * rng.uniform(0.2, 1)),
                float(np.float32(rng.uniform(-7, 7))) if k % 3 else 0.0,
                int(rng.integers(0, 17)),
                float(rng.uniform(0, 1)),
                bool(k % 5 == 0),
                value,
            )
        )
    return out


REGIONS = [
    ((0.0, 1.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0)),
    ((-3.0, 4.0, -3.0, 4.0), (-3.0, 4.0, -3.0, 4.0)),
    ((0.1, 0.9, 0.2, 0.7), (0.0, 0.5, 0.25, 1.0)),
    ((0.0, 1.0, 0.0, 1.0), (0.1, 0.6, 0.0, 0.75)),
]


@pytest.mark.parametrize("tile", [(True, True), (True, False), (False, False)])
@pytest.mark.parametrize(("render", "tiling"), REGIONS)
def test_pack_and_pixel_work_as_the_originals(
    render: tuple, tiling: tuple, tile: tuple[bool, bool]
) -> None:
    for seed, n in ((1, 0), (2, 1), (3, 40), (4, 400)):
        for color in (False, True):
            spl = random_splats(seed, n, color=color)
            kw = {"render": render, "tiling": tiling}
            want = fxsplat_scalar.pack(spl, color=color, tile=tile, **kw)
            got = fxsplat.pack(spl, color=color, tile=tile, **kw)
            parts = (got.grid, got.cells, got.index)
            assert parts == (want.grid, want.cells, want.index)
            assert got.image.tobytes() == want.image.tobytes()
        args = (spl, render, tiling, tile)
        work = fxsplat.pixel_work(*args), fxsplat_scalar.pixel_work(*args)
        assert struct.pack("<d", work[0]) == struct.pack("<d", work[1])


def test_a_dense_pack_takes_the_finest_grid_that_fits() -> None:
    spl = [s for s in random_splats(8, 3000) if s.hx < 0.05]
    want = fxsplat_scalar.pack(spl, color=False, tile=(True, True))
    got = fxsplat.pack(spl, color=False, tile=(True, True))
    assert got.grid == want.grid > 8
    assert got.image.tobytes() == want.image.tobytes()


@pytest.mark.parametrize("tile", [(True, True), (False, True), (False, False)])
@pytest.mark.parametrize(("render", "tiling"), REGIONS)
def test_raster_as_the_original(
    render: tuple, tiling: tuple, tile: tuple[bool, bool]
) -> None:
    outcomes = set()
    for seed in range(12):
        spl = random_splats(100 + seed, 1 + seed * 7)
        want = fxsplat_scalar.raster(spl, tile, tiling, render)
        got = fxsplat.raster(spl, tile, tiling, render)
        outcomes.add(want is None)
        if want is None:
            assert got is None
        else:
            assert got is not None
            assert fxsplat.float_bytes(got) == struct.pack(f"<{len(want)}f", *want)
    # a disc ending at the tiling region's edge to float32 rounding (x_region5)
    edge = fxsplat.Splat(0.6, 0.5, 0.15, 0.15, 0.0, 3, 0.0, False, [1.0])  # noqa: FBT003
    for spl in ([edge], [*random_splats(7, 5), edge]):
        args = (spl, (True, True), (0.25, 0.75, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0))
        want = fxsplat_scalar.raster(*args)
        got = fxsplat.raster(*args)
        assert (got is None) == (want is None)
        if want is not None:
            assert fxsplat.float_bytes(got) == struct.pack(f"<{len(want)}f", *want)


def test_the_drawing_order_as_the_original() -> None:
    rng = np.random.default_rng(3)
    for n in (0, 1, 7, 500):
        spl = random_splats(n, n)
        for s in spl:  # some images, of a few inputs, filterings, premultiplied
            if rng.uniform() < 0.4:
                index, flags = int(rng.integers(0, 5)), int(rng.integers(0, 32))
                s.pattern = fxmap.image_id(index, flags)
        assert fxdraw.order(spl) == fxsplat_scalar.order(spl)
