"""While loops without a condition, and nested (docs/function-ops.md 1.4.1, 1.4.2).

Section 1.8's tests E to I, on synthetic bytecode: the scalar evaluator,
the lane evaluator and the GLSL translation.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from test_function import ADD
from test_function import GETVAR
from test_function import GTEQ
from test_function import SEQ
from test_function import SETVAR
from test_function import WHILE
from test_function import compiles
from test_function import const
from test_function import glsl_function
from test_function import ins
from test_function import loop_ops

from sbsar2ptex.function import BOOL
from sbsar2ptex.function import FLOAT
from sbsar2ptex.function import MAX_ITERATIONS
from sbsar2ptex.function import NO_VALUE
from sbsar2ptex.function import Block
from sbsar2ptex.function import FunctionError
from sbsar2ptex.function import Instr
from sbsar2ptex.function import evaluate
from sbsar2ptex.function import evaluate_lanes
from sbsar2ptex.function import loops
from sbsar2ptex.function import to_glsl

N = NO_VALUE
DIV = 0x15


def no_condition(cap: int) -> Block:
    """Return Test E: i from 7, i += 10, cap times; the block's value i / 100."""
    return Block([
        const(7.0),                                   # 0
        ins(SETVAR, 0, 0),                            # 1  i = 7
        ins(SEQ, 1, N),                               # 2
        const(0.0),                                   # 3
        ins(SETVAR, 3, 1),                            # 4  hidden = 0
        ins(SEQ, 2, 4),                               # 5  init
        ins(GETVAR, 0),                               # 6
        const(10.0),                                  # 7
        ins(ADD, 6, 7),                               # 8
        ins(SETVAR, 8, 0),                            # 9  i += 10
        ins(SETVAR, 9, 1),                            # 10 body
        ins(WHILE, *loop_ops(5, N, 10, cap, pad=True)),  # 11
        ins(GETVAR, 1),                               # 12
        ins(SEQ, 11, 12),                             # 13
        const(100.0),                                 # 14
        ins(DIV, 13, 14),                             # 15
    ])  # fmt: skip


def f32(x: float) -> float:
    return float(np.float32(x))


@pytest.mark.parametrize(("cap", "value", "i"), [(3, 0.37, 37.0), (1, 0.17, 17.0),
                                                 (5, 0.57, 57.0)])  # fmt: skip
def test_e_a_loop_without_a_condition_runs_cap_times(
    cap: int, value: float, i: float
) -> None:
    blk = no_condition(cap)
    (lp,) = loops(blk).values()  # the ranges are accepted
    assert (lp.init, lp.cond, lp.body, lp.cap, lp.at) == (5, N, 10, cap, 11)
    assert not lp.has_cond
    assert lp.body_start == 6
    variables: dict = {}
    assert evaluate(blk, {}, variables=variables) == [f32(f32(i) / 100.0)]
    assert variables[0, FLOAT, 1] == [i]
    assert evaluate(blk, {})[0] == pytest.approx(value)
    # the lanes: the same, in every lane
    (lanes,) = evaluate_lanes(blk, {}, {}, None, 3)
    assert list(lanes) == [f32(f32(i) / 100.0)] * 3


@pytest.mark.parametrize("cap", [3, 1, 5])
def test_e_glsl_has_no_break_and_loops_cap_times(cap: int, tmp_path: Path) -> None:
    lines, source = glsl_function(no_condition(cap))
    compiles(source, tmp_path)
    text = [line.strip() for line in lines]
    assert not any("break" in line for line in text)
    assert f"for (int f_it11 = 0; f_it11 < {cap}; f_it11++) {{" in text
    assert "f_11 = f_10;" in text
    assert not any("65535" in line for line in text)


def test_f_no_condition_and_no_cap_is_endless() -> None:
    # Test F: the evaluator gives up at its limit (the value is unknown) ...
    blk = no_condition(0)
    with pytest.raises(FunctionError, match=f"after {MAX_ITERATIONS} iterations"):
        evaluate(blk, {})
    # ... and the GLSL translation reports it as unsupported.
    with pytest.raises(FunctionError, match="endless"):
        to_glsl(blk, "f", lambda *_: "0.0", lambda *_: "0.0")


def nested_no_condition() -> Block:
    """Return Test G: section 1.4.2's block, an outer cap 3 around an inner cap 2."""
    return Block([
        const(7.0),                                   # 0
        ins(SETVAR, 0, 0),                            # 1  i = 7
        const(0.0),                                   # 2
        ins(SETVAR, 2, 1),                            # 3  outer hidden = 0
        ins(SEQ, 1, 3),                               # 4
        ins(SEQ, 4, N),                               # 5  outer init
        const(0.0),                                   # 6
        ins(SETVAR, 6, 2),                            # 7  j = 0
        ins(SETVAR, 6, 3),                            # 8  inner hidden = 0
        ins(SEQ, 7, 8),                               # 9
        ins(SEQ, 9, N),                               # 10 inner init
        ins(GETVAR, 0),                               # 11
        const(10.0),                                  # 12
        ins(ADD, 11, 12),                             # 13
        ins(SETVAR, 13, 0),                           # 14 i += 10
        ins(SETVAR, 14, 3),                           # 15 inner body
        ins(WHILE, *loop_ops(10, N, 15, 2, pad=True)),  # 16
        ins(GETVAR, 3),                               # 17
        ins(SEQ, 16, 17),                             # 18
        ins(SETVAR, 18, 1),                           # 19 outer body
        ins(WHILE, *loop_ops(5, N, 19, 3, pad=True)),   # 20
        ins(GETVAR, 1),                               # 21
        ins(SEQ, 20, 21),                             # 22
    ])  # fmt: skip


def test_g_nested_loops_without_a_condition(tmp_path: Path) -> None:
    blk = nested_no_condition()
    found = loops(blk)
    assert {lp.at: (lp.init, lp.cond, lp.body, lp.cap) for lp in found.values()} == {
        16: (10, N, 15, 2),
        20: (5, N, 19, 3),
    }
    variables: dict = {}
    assert evaluate(blk, {}, variables=variables) == [67.0]
    assert variables[0, FLOAT, 1] == [67.0]
    (lanes,) = evaluate_lanes(blk, {}, {}, None, 2)
    assert list(lanes) == [67.0, 67.0]
    lines, source = glsl_function(blk)
    compiles(source, tmp_path)
    text = [line.strip() for line in lines]
    assert not any("break" in line for line in text)
    outer = text.index("for (int f_it20 = 0; f_it20 < 3; f_it20++) {")
    inner = text.index("for (int f_it16 = 0; f_it16 < 2; f_it16++) {")
    # the inner loop's init range runs in each outer iteration, before it
    j_reset = next(n for n, line in enumerate(text) if line.startswith("f_var2 = "))
    assert outer < j_reset < inner


@pytest.mark.parametrize(
    ("operands", "match"),
    [
        ([5, N, 9, 0, 3, 0], "not just before it"),  # body not just before
        ([5, 12, 10, 0, 3, 0], "not just before it"),  # cond not in (init, body)
        ([N, 7, 10, 0, 3, 0], "unsupported"),  # init NO_VALUE
    ],
)
def test_h_malformed_ranges_are_refused(operands: list[int], match: str) -> None:
    blk = no_condition(3)
    blk.instrs[11] = ins(WHILE, *operands)
    with pytest.raises(FunctionError, match=match):
        loops(blk)
    with pytest.raises(FunctionError):
        evaluate(blk, {})


def _loop(
    out: list[Instr],
    hidden: int,
    cap: int,
    cond: Callable[[list[Instr]], int] | None,
    body: Callable[[list[Instr]], int],
) -> int:
    """Append a loop in the cooker's form; return its value (sequence) index.

    ``cond(out)`` appends the condition range and returns its last index;
    ``body(out)`` appends the body range (nested loops too) and returns the
    index of the body's value.
    """
    out.append(const(0.0))
    out.append(ins(SETVAR, len(out) - 1, hidden))
    init = len(out) - 1
    c = cond(out) if cond is not None else N
    value = body(out)
    out.append(ins(SETVAR, value, hidden))
    b = len(out) - 1
    out.append(ins(WHILE, *loop_ops(init, c, b, cap, pad=True)))
    at = len(out) - 1
    out.append(ins(GETVAR, hidden))
    out.append(ins(SEQ, at, len(out) - 1))
    return len(out) - 1


def _counter_cond(var: int, limit: float):
    """Return a condition range builder: var += 1 first; stop when var > limit."""

    def cond(out: list[Instr]) -> int:
        out.append(ins(GETVAR, var))
        out.append(const(1.0))
        out.append(ins(ADD, len(out) - 2, len(out) - 1))
        out.append(ins(SETVAR, len(out) - 1, var))
        out.append(const(limit + 0.5))
        out.append(ins(GTEQ, len(out) - 2, len(out) - 1, kind=BOOL))
        return len(out) - 1

    return cond


def _reset(var: int, then):
    """Return a body builder: var = 0 (a nested loop's own init), then ``then``."""

    def body(out: list[Instr]) -> int:
        out.append(const(0.0))
        out.append(ins(SETVAR, len(out) - 1, var))
        return then(out)

    return body


def mosaic_shaped() -> Block:
    """Return five nested loops as mosaic's pixel processors hold them (1.6).

    Outermost first: a condition loop (2 runs), a condition loop (2), two
    without a condition (cap 3 each), and innermost a condition loop of cap
    1 whose condition is never true. The innermost body counts in v0.
    """
    out: list[Instr] = [const(0.0), ins(SETVAR, 0, 0)]  # total = 0

    def count(o: list[Instr]) -> int:
        o.append(ins(GETVAR, 0))
        o.append(const(1.0))
        o.append(ins(ADD, len(o) - 2, len(o) - 1))
        o.append(ins(SETVAR, len(o) - 1, 0))
        return len(o) - 1

    def l491(o):
        return _loop(o, 15, 1, _counter_cond(5, 1e9), count)

    def l500(o):
        return _loop(o, 14, 3, None, l491)

    def l509(o):
        return _loop(o, 13, 3, None, l500)

    def l519(o):
        return _loop(o, 12, 0, _counter_cond(2, 2.0), _reset(3, l509))

    # the outer counter v1 and the next one's v2 start at 0
    out += [ins(SETVAR, 0, 1)]
    _loop(out, 11, 0, _counter_cond(1, 2.0), _reset(2, l519))
    out.append(ins(GETVAR, 0))
    return Block(out)


def test_i_mosaic_shaped_nesting(tmp_path: Path) -> None:
    # Test I's shape: no-condition loops (cap 3) inside condition loops, a
    # condition loop of cap 1 innermost. 2 x 2 x 3 x 3 x 1 = 36 counts.
    blk = mosaic_shaped()
    found = sorted(loops(blk).values(), key=lambda lp: lp.start)
    assert [(lp.has_cond, lp.cap) for lp in found] == [
        (True, 0), (True, 0), (False, 3), (False, 3), (True, 1),
    ]  # fmt: skip
    assert evaluate(blk, {}) == [36.0]
    (lanes,) = evaluate_lanes(blk, {}, {}, None, 4)
    assert list(lanes) == [36.0] * 4
    lines, source = glsl_function(blk)
    compiles(source, tmp_path)
    text = [line.strip() for line in lines]
    assert sum("break" in line for line in text) == 3  # the condition loops only
    assert sum(line.startswith("while (true)") for line in text) == 2
    assert sum(" < 3; " in line for line in text) == 2
    assert sum(" < 1; " in line for line in text) == 1
