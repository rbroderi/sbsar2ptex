"""fxsplat's vectorized paths against the scalar ones, bit for bit.

splats (deferred subtrees drawn as lanes) against the plain walk, on
synthetic FX-Maps.
"""

import struct

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
    assert len(splats_both(node, root, st)) > 60
    assert splats_both(node, root, st, color=True)


def test_a_stateful_walk_defers_its_stateless_subtrees() -> None:
    asm = Asm()
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
    root = fxmap.Iterate(count=count, repeated=pick)
    st = fxmap.Settings(seed=99)
    node = asm.node()
    later = fxsplat._deferrable(node, root, st)
    assert id(sub) in later
    assert later[id(sub)] == [6]
    assert id(noisy) not in later
    assert id(pick) not in later
    assert id(root) not in later
    assert len(splats_both(node, root, st)) > 40
