"""function.evaluate_lanes and the generator's lane steps, bit for bit.

evaluate_lanes against evaluate a lane at a time, and fxrng's lane steps
against its scalar steps, on synthetic blocks and values.
"""

import itertools
import math
import struct

import numpy as np
import pytest

from sbsar2ptex.function import _BINARY
from sbsar2ptex.function import _UNARY
from sbsar2ptex.function import BOOL
from sbsar2ptex.function import FLOAT
from sbsar2ptex.function import INT
from sbsar2ptex.function import NO_VALUE
from sbsar2ptex.function import Block
from sbsar2ptex.function import Diverged
from sbsar2ptex.function import Instr
from sbsar2ptex.function import evaluate
from sbsar2ptex.function import evaluate_lanes
from sbsar2ptex.fxrng import PARAM_LCG
from sbsar2ptex.fxrng import FxRng
from sbsar2ptex.fxrng import FxRngLanes
from sbsar2ptex.fxrng import depth_step
from sbsar2ptex.fxrng import depth_steps
from sbsar2ptex.fxrng import iterate_step
from sbsar2ptex.fxrng import iterate_steps

FLOATS = [
    0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, 2.5, -2.5, 3.0, 7.0, 0.1, -0.3,
    1234.5678, 16777217.0, 1e-30, 1.4e-45, 3.4e38, -3.4e38, 2e9, -3e9,
    math.inf, -math.inf, math.nan, 6.2831855, 100.25,
]  # fmt: skip
INTS = [0, 1, -1, 2, -2, 3, 7, -7, 65536, 2**31 - 1, -(2**31), 1000003]
BOOLS = [0, 1]
POOLS = {FLOAT: FLOATS, INT: INTS, BOOL: BOOLS}


def load(kind: int, slot: int, width: int = 1) -> Instr:
    """Return an instruction reading built-in ``slot`` as ``kind``."""
    return Instr(op=0x01, type=kind, width=width, operands=[slot])


def lane_array(values: list) -> np.ndarray:
    """Return built-in values as a lane array (ints int64, floats float64)."""
    ints = all(isinstance(v, int) for v in values)
    return np.array(values, dtype=np.int64 if ints else np.float64)


def same(scalar: list, lanes: list[np.ndarray], lane: int) -> bool:
    """Whether a lane's value is the scalar one: floats bit for bit."""
    if len(scalar) != len(lanes):
        return False
    for s, arr in zip(scalar, lanes, strict=True):
        got = arr[lane].item()
        if isinstance(s, float) or isinstance(got, float):
            if not (isinstance(s, float) and isinstance(got, float)):
                return False
            if struct.pack("<d", s) != struct.pack("<d", got):
                return False
        elif int(s) != int(got):
            return False
    return True


def check(block: Block, columns: dict[int, list], rand_lane: str | None = None) -> None:
    """Evaluate ``block`` a lane at a time and as lanes: the same values, or errors.

    ``columns`` are the built-ins, a value a lane each. With ``rand_lane``,
    rand calls draw from FxRng (lane k's state: k * 7919) and FxRngLanes.
    """
    n = len(next(iter(columns.values())))
    states = [k * 7919 for k in range(n)]
    want, errors = [], set()
    for k in range(n):
        builtins = {slot: [vals[k]] for slot, vals in columns.items()}
        rand = FxRng(states[k], rand_lane).rand if rand_lane else None
        try:
            want.append(evaluate(block, {}, builtins, rand, {}))
        except Exception as e:  # noqa: BLE001
            errors.add(type(e))
            want.append(None)
    lanes_in = {slot: [lane_array(vals)] for slot, vals in columns.items()}
    rand = FxRngLanes(np.array(states), rand_lane).rand if rand_lane else None
    if errors:
        with pytest.raises((*errors, Diverged)):
            evaluate_lanes(block, {}, lanes_in, rand, n)
        return
    try:
        got = evaluate_lanes(block, {}, lanes_in, rand, n)
    except Diverged:
        return  # (the caller then evaluates each lane)
    bad = [k for k in range(n) if not same(want[k], got, k)]
    assert not bad, [(want[k], [g[k] for g in got]) for k in bad[:5]]


def pairs(kinds: tuple[int, int]) -> dict[int, list]:
    """Return two built-in columns: every pair of the kinds' pools."""
    rows = list(itertools.product(POOLS[kinds[0]], POOLS[kinds[1]]))
    return {0: [a for a, _ in rows], 1: [b for _, b in rows]}


KIND_PAIRS = [(FLOAT, FLOAT), (INT, INT), (FLOAT, INT), (INT, FLOAT), (BOOL, BOOL)]


@pytest.mark.parametrize("kinds", KIND_PAIRS)
@pytest.mark.parametrize("kind", [FLOAT, INT, BOOL])
@pytest.mark.parametrize("op", sorted({*_BINARY, 0x15, 0x16}))
def test_binary_ops_as_lanes(op: int, kind: int, kinds: tuple[int, int]) -> None:
    block = Block([load(kinds[0], 0), load(kinds[1], 1), Instr(op, kind, 1, [0, 1])])
    columns = pairs(kinds)
    if op == 0x36:  # pow: Python's own a lane at a time; keep it from overflowing
        columns = {
            s: [v for v, w in zip(c, columns[1], strict=True) if abs(w) < 40]
            for s, c in columns.items()
        }
    check(block, columns)


@pytest.mark.parametrize("src", [FLOAT, INT, BOOL])
@pytest.mark.parametrize("kind", [FLOAT, INT, BOOL])
@pytest.mark.parametrize("op", sorted(_UNARY))
def test_unary_ops_as_lanes(op: int, kind: int, src: int) -> None:
    pool = [*POOLS[src], *(np.linspace(-20, 20, 97).tolist() if src == FLOAT else [])]
    check(Block([load(src, 0), Instr(op, kind, 1, [0])]), {0: pool})


def test_trig_and_mixed_ops_as_lanes() -> None:
    rng = np.random.default_rng(5)
    xs = np.concatenate([rng.uniform(-50, 50, 400), rng.uniform(-1e6, 1e6, 50)])
    ys = rng.uniform(-3, 3, len(xs))
    cols = {0: xs.tolist(), 1: ys.tolist(), 2: (rng.uniform(0, 1, len(xs))).tolist()}
    instrs = [
        load(FLOAT, 0), load(FLOAT, 1), load(FLOAT, 2),            # 0 1 2
        Instr(0x0D, FLOAT, 2, [0, 1]),                             # 3 vec2
        Instr(0x2E, FLOAT, 2, [2, 0]),                             # 4 cartesian
        Instr(0x18, FLOAT, 1, [3, 4]),                             # 5 dot
        Instr(0x2F, FLOAT, 2, [3, 4, 2]),                          # 6 lerp
        Instr(0x2D, FLOAT, 1, [6]),                                # 7 atan2
        Instr(0x10, FLOAT, 2, [6, 0b0001]),                        # 8 swizzle yx
        Instr(0x1F, BOOL, 1, [5, 2]),                              # 9 dot > t
        Instr(0x09, FLOAT, 2, [9, 8, 3]),                          # 10 if/else
        Instr(0x11, INT, 2, [10]),                                 # 11 convert
        Instr(0x16, INT, 2, [11, 1]),                              # 12 int mod
        Instr(0x0E, FLOAT, 3, [7, 12]),                            # 13 vec3
        Instr(0x07, FLOAT, 3, [13, 4]),                            # 14 set v4
        Instr(0x04, FLOAT, 3, [4]),                                # 15 get v4
        Instr(0x26, FLOAT, 3, [15]),                               # 16 cos
        Instr(0x0C, INT, 3, [16, NO_VALUE]),                       # 17 dropped
    ]  # fmt: skip
    check(Block(instrs), cols)


@pytest.mark.parametrize("lane", ["opacity", "patternsize", "imageindex"])
def test_rand_as_lanes(lane: str) -> None:
    cols = {0: [0.0, 1.0, 0.5, 3.0, -2.0, 1e30, math.inf, math.nan] * 8}
    instrs = [
        load(FLOAT, 0),
        Instr(0x32, FLOAT, 1, [0]),
        Instr(0x32, FLOAT, 1, [1]),
        Instr(0x12, FLOAT, 1, [1, 2]),
    ]
    check(Block(instrs), cols, rand_lane=lane)


def loop_block(limit: int) -> Block:
    """Return: i = 0; while not (i >= limit): i = i + step; then i (step built-in 0)."""
    zero = struct.unpack("<I", struct.pack("<f", 0.0))[0]
    lim = struct.unpack("<I", struct.pack("<f", float(limit)))[0]
    return Block([
        load(FLOAT, 0),                                                 # 0 step
        Instr(0x00, FLOAT, 1, [zero & 0xFFFF, zero >> 16]),             # 1 0.0
        Instr(0x07, FLOAT, 1, [1, 3]),                                  # 2 set i
        Instr(0x04, FLOAT, 1, [3]),                                     # 3 get i
        Instr(0x00, FLOAT, 1, [lim & 0xFFFF, lim >> 16]),               # 4 limit
        Instr(0x20, BOOL, 1, [3, 4]),                                   # 5 i >= limit
        Instr(0x04, FLOAT, 1, [3]),                                     # 6 get i
        Instr(0x12, FLOAT, 1, [6, 0]),                                  # 7 i + step
        Instr(0x07, FLOAT, 1, [7, 3]),                                  # 8 set i
        Instr(0x0B, FLOAT, 1, [2, 5, 8, 0, 0]),                         # 9 while
    ])  # fmt: skip


def test_a_loop_the_lanes_agree_on_runs_together() -> None:
    check(loop_block(5), {0: [1.0] * 6})
    got = evaluate_lanes(loop_block(5), {}, {0: [np.full(3, 1.0)]}, None, 3)
    assert got[0].tolist() == [5.0] * 3


def test_lanes_that_part_ways_are_refused() -> None:
    with pytest.raises(Diverged):  # a loop run a different number of times
        evaluate_lanes(loop_block(5), {}, {0: [np.array([1.0, 2.5])]}, None, 2)
    # an if/else choosing between an int and a float
    block = Block([
        load(BOOL, 0), load(INT, 1), load(FLOAT, 1), Instr(0x09, FLOAT, 1, [0, 1, 2])
    ])  # fmt: skip
    with pytest.raises(Diverged):
        evaluate_lanes(block, {}, {0: [np.array([0, 1])], 1: [np.ones(2)]}, None, 2)


def test_generator_steps_as_lanes() -> None:
    states = np.array([0, 1, 0xFFFFFFFF, 0x12345678, 0xDEADBEEF], dtype=np.uint64)
    for depth in range(20):
        want = [depth_step(int(s), depth) for s in states]
        assert depth_steps(states, depth).tolist() == want
    passes = np.array([0, 3, 1, 7, 2])
    want = []
    for s, p in zip(states.tolist(), passes.tolist(), strict=True):
        for _ in range(p + 1):
            s = iterate_step(s)  # noqa: PLW2901
        want.append(s)
    assert iterate_steps(states, passes).tolist() == want
    for lane in PARAM_LCG:
        scalar = [FxRng(int(s), lane) for s in states]
        lanes = FxRngLanes(states, lane)
        for a in (1.0, 0.25, 7.0, 0.0):
            got = lanes.rand(np.full(len(states), a)).tolist()
            assert got == [r.rand(a) for r in scalar]
