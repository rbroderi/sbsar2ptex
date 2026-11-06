"""Function values at the edge of the float range (docs/function-float-range.md)."""

import math
import re
import struct

import pytest

from sbsar2ptex.function import FLOAT
from sbsar2ptex.function import INT
from sbsar2ptex.function import Block
from sbsar2ptex.function import Instr
from sbsar2ptex.function import _flt
from sbsar2ptex.function import bounds
from sbsar2ptex.function import evaluate
from sbsar2ptex.function import evaluate_lanes
from sbsar2ptex.function import to_glsl


def bits_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def float_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def reads_back(text: str) -> int:
    """Return the float32 bits a GLSL literal ``text`` compiles to."""
    m = re.fullmatch(r"uintBitsToFloat\((\d+)u\)", text)
    if m:
        return int(m.group(1))
    return float_bits(float(text))  # a decimal: to the nearest float32


@pytest.mark.parametrize(
    "bits",
    [0x7F7FFFFF, 0xFF7FFFFF, 0x7F7FFFFE, 0x00000001, 0x3F800000],  # L1-L3, L7, L8
)
def test_finite_literals_keep_their_bits(bits: int) -> None:
    text = _flt(bits_float(bits))
    assert "uintBitsToFloat" not in text
    assert reads_back(text) == bits


def test_flt_max_is_its_shortest_decimal() -> None:
    # The spec's example is 3.40282347e+38; the shortest decimal reading back
    # to FLT_MAX has 8 digits. Both lie above FLT_MAX by less than half an ulp.
    assert _flt(bits_float(0x7F7FFFFF)) == "3.4028235e+38"  # L1
    assert _flt(bits_float(0xFF7FFFFF)) == "-3.4028235e+38"  # L2
    assert _flt(1.0) == "1.0"  # L8


@pytest.mark.parametrize("bits", [0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000])
def test_infinities_and_nans_are_written_by_their_bits(bits: int) -> None:
    # L4-L6: a real infinity (not 1e30), a NaN (not the text nan)
    text = _flt(bits_float(bits))
    assert text == f"uintBitsToFloat({bits}u)"
    assert "1e30" not in text
    assert "nan" not in text


def test_every_float_of_the_top_exponent_formats() -> None:
    # Section 4: format, parse, compare bits, over the finite floats with
    # exponent 0xFE (a stride through them, both signs, and both ends).
    mantissas = [*range(0, 1 << 23, 4099), (1 << 23) - 2, (1 << 23) - 1]
    for sign in (0, 1 << 31):
        for m in mantissas:
            bits = sign | 0xFE << 23 | m
            assert reads_back(_flt(bits_float(bits))) == bits


def iconst(value: int) -> Instr:
    word = value & 0xFFFFFFFF
    return Instr(op=0x00, type=INT, width=1, operands=[word & 0xFFFF, word >> 16])


def fconst(value: float) -> Instr:
    word = float_bits(value)
    return Instr(op=0x00, type=FLOAT, width=1, operands=[word & 0xFFFF, word >> 16])


ONE = 7
"""The uid of a float graph input holding 1: a value the cooker cannot fold."""


def one() -> Instr:
    return Instr(op=0x02, type=FLOAT, width=1, operands=[ONE, 0])


def to_int(x: float) -> list[Instr]:
    """Return int(x * one): a run-time float-to-int convert (#0 to #3)."""
    return [
        fconst(x),
        one(),
        Instr(op=0x14, type=FLOAT, width=1, operands=[0, 1]),
        Instr(op=0x11, type=INT, width=1, operands=[2]),
    ]


@pytest.mark.parametrize(
    ("x", "want"),
    [
        (2.7, 2),
        (-2.7, -3),  # floor, not truncation (R2)
        (3e9, 3000000000),  # no saturation, no wrap (R2)
        (-3e9, -3000000000),
        (1e10, 10000000000),
        (2.0**31, 2147483648),
        (16777216.0, 16777216),
        (bits_float(0x7F7FFFFF), int(bits_float(0x7F7FFFFF))),  # int(M * one)
        (-bits_float(0x7F7FFFFF), -int(bits_float(0x7F7FFFFF))),
    ],
)
def test_convert_to_int_floors_with_no_wrap(x: float, want: int) -> None:
    assert evaluate(Block(to_int(x)), {ONE: [1.0]}) == [want]


def int_plus_one(c: int) -> Block:
    """Return integer constant c + int(one) (section 3.2's table)."""
    return Block(
        [
            iconst(c),
            one(),
            Instr(op=0x11, type=INT, width=1, operands=[1]),
            Instr(op=0x12, type=INT, width=1, operands=[0, 2]),
        ]
    )


def test_integer_sums_round_to_float32_and_do_not_wrap() -> None:
    assert evaluate(int_plus_one(2147483647), {ONE: [1.0]}) == [2147483648]  # R2
    assert evaluate(int_plus_one(16777217), {ONE: [1.0]}) == [16777216]
    neg = Block(
        [
            iconst(-2147483648),
            fconst(-1.0),
            one(),
            Instr(op=0x14, type=FLOAT, width=1, operands=[1, 2]),
            Instr(op=0x11, type=INT, width=1, operands=[3]),
            Instr(op=0x12, type=INT, width=1, operands=[0, 4]),
        ]
    )
    assert evaluate(neg, {ONE: [1.0]}) == [-2147483648]
    # int(3e9 * one) + int(one): 3e9 (float32 rounding of 3e9 + 1)
    blk = Block(
        [
            *to_int(3e9),
            one(),
            Instr(op=0x11, type=INT, width=1, operands=[4]),
            Instr(op=0x12, type=INT, width=1, operands=[3, 5]),
        ]
    )
    assert evaluate(blk, {ONE: [1.0]}) == [3000000000]


def test_lanes_agree_with_evaluate_on_the_int_model() -> None:
    blk = Block(to_int(-2.7))
    (got,) = evaluate_lanes(blk, {ONE: [1.0]}, {}, None, 3)
    assert got.tolist() == [-3, -3, -3]


def test_float_overflow_is_an_infinity() -> None:
    # R1: M * s is +inf, M * s * 0 NaN, 1 / (M * s) 0
    m = bits_float(0x7F7FFFFF)
    instrs = [
        fconst(m),
        fconst(2.0),
        Instr(op=0x14, type=FLOAT, width=1, operands=[0, 1]),
    ]
    assert evaluate(Block(instrs), {}) == [math.inf]
    zero = [*instrs, fconst(0.0), Instr(op=0x14, type=FLOAT, width=1, operands=[2, 3])]
    assert math.isnan(evaluate(Block(zero), {})[0])
    inv = [*instrs, fconst(1.0), Instr(op=0x15, type=FLOAT, width=1, operands=[3, 2])]
    assert evaluate(Block(inv), {}) == [0.0]


def test_glsl_convert_to_int_floors() -> None:
    lines, name = to_glsl(
        Block(to_int(-2.7)),
        "f",
        lambda *_: "$one",
        lambda *_: "0.0",
    )
    assert any("int(floor(f_2))" in line for line in lines)
    assert name == "f_3"


def test_glsl_writes_flt_max_exactly() -> None:
    # C3: a bounding box's starting value as the literal of L1 and L2
    m = bits_float(0x7F7FFFFF)
    lines, _ = to_glsl(Block([fconst(m), fconst(-m)]), "f", None, None)  # type: ignore[arg-type]
    text = "\n".join(lines)
    assert "f_0 = 3.4028235e+38;" in text
    assert "f_1 = -3.4028235e+38;" in text


def test_convert_bounds_are_the_floor() -> None:
    def inputs(uid: int, width: int) -> list[tuple[float, float]]:
        return [(0.5, 2.5)]

    assert bounds(Block(to_int(1.0)), inputs, {}) == [(0, 2)]
