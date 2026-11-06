"""FX-Map variables set as different types (docs/fxmap-variables.md section 3).

Synthetic functions run in tree order with one shared store; the store and
the GLSL globals are keyed by (id, type, width) (function.var_key).
"""

import pytest
from test_fxsplat_lanes import Asm
from test_fxsplat_lanes import code

from sbsar2ptex import fxmap
from sbsar2ptex import fxsplat
from sbsar2ptex.function import FLOAT
from sbsar2ptex.function import decode
from sbsar2ptex.function import evaluate
from sbsar2ptex.function import glsl_type
from sbsar2ptex.function import to_glsl

GET, SET, VEC2 = 0x04, 0x07, 0x0D


def set2(asm: Asm, var: int, x: float, y: float) -> int:
    """Add a block setting ``var`` to the float2 (x, y)."""
    return asm.block(
        ("f", x), ("f", y), (VEC2, FLOAT, 2, [0, 1]), (SET, FLOAT, 2, [2, var])
    )


def set1(asm: Asm, var: int, x: float) -> int:
    """Add a block setting ``var`` to the float x."""
    return asm.block(("f", x), (SET, FLOAT, 1, [0, var]))


def get(asm: Asm, var: int, width: int) -> int:
    """Add a block reading ``var`` as a float of ``width``."""
    return asm.block((GET, FLOAT, width, [var]))


def run(asm: Asm, order: list[int]) -> list[list]:
    """Run the blocks at ``order`` with one store; return each one's value."""
    node = asm.node()
    store: dict = {}
    return [
        evaluate(decode(node.at(at, (node.end - at) // 4)), {}, variables=store)
        for at in order
    ]


def two_types() -> tuple[Asm, int, int, int, int]:
    asm = Asm()
    p, pc = set2(asm, 7, 0.25, 0.75), get(asm, 7, 2)
    s, sc = set1(asm, 7, 0.5), get(asm, 7, 1)
    return asm, p, pc, s, sc


def tree(p: int, pc: int, s: int, sc: int) -> fxmap.Tree:
    """P's child reads v7 as float2; a later sibling S's child, as float."""
    pchild = fxmap.Quadrant(pattern=1, params={"opacity": code(pc)})
    schild = fxmap.Quadrant(pattern=1, params={"opacity": code(sc)})
    pq = fxmap.Quadrant(pattern=0, params={"opacity": code(p)}, children=[pchild])
    sq = fxmap.Quadrant(pattern=0, params={"opacity": code(s)}, children=[schild])
    return fxmap.Quadrant(pattern=0, children=[pq, sq])


def test_1_same_id_two_types_separate_paths() -> None:
    asm, p, pc, s, sc = two_types()
    got = run(asm, [p, pc, s, sc])
    assert got[1] == [0.25, 0.75]
    assert got[3] == [0.5]
    # the map translates: no "set as two types"
    node, st = asm.node(), fxmap.Settings()
    shared = fxsplat.variable_types(node, tree(p, pc, s, sc), st)
    assert shared == {(7, FLOAT, 2): "vec2", (7, FLOAT, 1): "float"}
    names = {fxsplat.variable_name(k, shared) for k in shared}
    assert len(names) == 2


def test_2_overwrite() -> None:
    asm, p, pc, s, sc = two_types()
    got = run(asm, [s, p, pc, sc])
    assert got[2] == [0.25, 0.75]
    assert got[3] == [0.5]


def test_3_mosaic_shape() -> None:
    asm = Asm()
    a_opacity = asm.block(
        ("f", 1.0), ("f", 2.0), (VEC2, FLOAT, 2, [0, 1]), (SET, FLOAT, 2, [2, 47]),
        ("f", 3.0), (SET, FLOAT, 1, [4, 51]),
    )  # fmt: skip
    a_size, a_rotation = get(asm, 47, 2), get(asm, 51, 1)
    b_count = asm.block(
        ("f", 4.0), (SET, FLOAT, 1, [0, 47]),
        ("f", 5.0), ("f", 6.0), (VEC2, FLOAT, 2, [2, 3]), (SET, FLOAT, 2, [4, 51]),
    )  # fmt: skip
    c_opacity = get(asm, 47, 1)
    got = run(asm, [a_opacity, a_size, a_rotation, b_count, c_opacity])
    assert got[1] == [1.0, 2.0]
    assert got[2] == [3.0]
    assert got[4] == [4.0]


def test_4_unset_read() -> None:
    # A get whose key is never set reads as any unset variable does
    # (docs/function-ops.md: 0), not another type's value of its id.
    asm = Asm()
    got = run(asm, [set2(asm, 7, 0.25, 0.75), get(asm, 7, 1)])
    assert got[1] == [0.0]


def test_the_glsl_reads_each_types_own_global() -> None:
    asm, p, pc, s, sc = two_types()
    node = asm.node()
    shared = fxsplat.variable_types(node, tree(p, pc, s, sc), fxmap.Settings())
    glob = {k: (fxsplat.variable_name(k, shared), t) for k, t in shared.items()}

    def glsl(at: int) -> str:
        block = decode(node.at(at, (node.end - at) // 4))
        lines, _ = to_glsl(
            block, "f", lambda *_: "0.0", lambda *_: "0.0", None, None, glob
        )
        return "\n".join(lines)

    two, one = glob[7, FLOAT, 2][0], glob[7, FLOAT, 1][0]
    assert f"{two} = vec2(" in glsl(p)
    assert f"{one} = float(" in glsl(s)
    assert two in glsl(pc)
    assert one in glsl(sc)
    assert one not in glsl(pc)


@pytest.mark.parametrize("width", [1, 2])
def test_one_type_keeps_its_plain_name(width: int) -> None:
    key = (9, FLOAT, width)
    assert fxsplat.variable_name(key, [key]) == "fxv9_$(name)"
    assert glsl_type(FLOAT, width) in ("float", "vec2")
