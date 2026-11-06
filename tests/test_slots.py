"""Value slots (get slot 0x03, set slot 0x06), docs/function-ops.md section 2.

Synthetic bytecode only: section 2.8's acceptance tests S1-S13, the
NO_VALUE operand (2.5), and Value Processors in a translated graph.
"""

import json
import math
import re
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

import pytest

from sbsar2ptex import translate
from sbsar2ptex.function import BOOL
from sbsar2ptex.function import FLOAT
from sbsar2ptex.function import INT
from sbsar2ptex.function import Block
from sbsar2ptex.function import FunctionError
from sbsar2ptex.function import bounds
from sbsar2ptex.function import decode
from sbsar2ptex.function import evaluate
from sbsar2ptex.function import f32
from sbsar2ptex.function import to_glsl
from sbsar2ptex.sbsasm import BASE
from sbsar2ptex.sbsasm import Asm
from sbsar2ptex.sbsasm import Node
from sbsar2ptex.sbsasm import field_count
from sbsar2ptex.sbsasm import value_width
from sbsar2ptex.slots import Aliased
from sbsar2ptex.slots import SlotStore
from sbsar2ptex.slots import Writer
from sbsar2ptex.slots import builtins as slot_builtins
from sbsar2ptex.slots import slot_bounds
from sbsar2ptex.slots import template_slots
from sbsar2ptex.slots import writers
from sbsar2ptex.translate import GraphInput
from sbsar2ptex.translate import UnsupportedError
from sbsar2ptex.translate import translate_graph

U = 0x5A3B1E01
"""A float graph input's uid."""
CONST, INPUT, GETSLOT, GETVAR, SETSLOT, SETVAR, IFELSE, SEQ = (
    0x00, 0x02, 0x03, 0x04, 0x06, 0x07, 0x09, 0x0C,
)  # fmt: skip
VEC2, VEC3, VEC4, SWIZZLE, CONVERT = 0x0D, 0x0E, 0x0F, 0x10, 0x11
ADD, MUL, DIV, EQ, NEQ, GT, LT = 0x12, 0x14, 0x15, 0x1D, 0x1E, 0x1F, 0x21
NO_VALUE = 0xFFFF

Ins = tuple[int, list[int], list[int]]
"""An instruction to assemble: opcode without its count, u16s, u32s."""


def base(op: int, kind: int = FLOAT, width: int = 1) -> int:
    return kind << 8 | (width - 1) << 6 | op


def ins(op: int, *operands: int, kind: int = FLOAT, width: int = 1) -> Ins:
    """Return an instruction of u16 operands."""
    return base(op, kind, width), list(operands), []


def fconst(x: float) -> Ins:
    (w,) = struct.unpack("<I", struct.pack("<f", x))
    return base(CONST), [], [w]


def iconst(x: int) -> Ins:
    return base(CONST, INT), [], [x & 0xFFFFFFFF]


def bconst(*, value: bool) -> Ins:
    return base(CONST, BOOL), [int(value)], []


def load(uid: int = U, kind: int = FLOAT, width: int = 1) -> Ins:
    return base(INPUT, kind, width), [], [uid]


def getslot(slot: int, kind: int = FLOAT, width: int = 1) -> Ins:
    return ins(GETSLOT, slot, kind=kind, width=width)


def setslot(value: int, slot: int, kind: int = FLOAT, width: int = 1) -> Ins:
    return ins(SETSLOT, value, slot, kind=kind, width=width)


def half(*instrs: Ins) -> list[int]:
    """Return a block's u16 stream; u32 operands 4-byte aligned (padded)."""
    out = [len(instrs)]
    for opcode, small, big in instrs:
        ops = list(small)
        if big and (len(out) + 1 + len(ops)) % 2:
            ops.append(0)  # padding before the u32s, counted
        for w in big:
            ops += [w & 0xFFFF, w >> 16]
        out += [len(ops) << 10 | opcode, *ops]
    return out


def words(stream: list[int]) -> list[int]:
    """Return a u16 stream as u32 words, low half first."""
    s = stream + [0] * (len(stream) % 2)
    return [s[k] | s[k + 1] << 16 for k in range(0, len(s), 2)]


def code(*instrs: Ins) -> list[int]:
    """Return a block's u32 words."""
    return words(half(*instrs))


def blk(*instrs: Ins) -> Block:
    """Return a block, assembled to bytecode and decoded."""
    return decode(code(*instrs))


def run(*blocks: Block, x: float = 0.0, store: dict | None = None) -> list[list]:
    """Run ``blocks`` in order with one slot store; each its own variables."""
    store = {} if store is None else store
    return [evaluate(b, {U: [x]}, slots=store) for b in blocks]


# S1: decoding (2.2's streams).


@pytest.mark.parametrize(
    ("stream", "want"),
    [
        ([0x0001, 0x0503, 0x0033], [("getslot", FLOAT, 1, [51])]),
        (
            [0x0002, 0x0900, 0x0000, 0x3F80, 0x0906, 0x0000, 0x0033],
            [("const", FLOAT, 1, [0, 0x3F80]), ("setslot", FLOAT, 1, [0, 51])],
        ),
        (
            [0x0002, 0x0440, 0x0000, 0x0846, 0x0000, 0x003E],
            [("const", BOOL, 1, [0]), ("setslot", BOOL, 1, [0, 62])],
        ),
        (
            [0x0003, 0x0643, 0x0004, 0x0A50, 0x0000, 0x0004, 0x0A46, 0x0001, 0x000C],
            [
                ("getslot", INT, 2, [4]),
                ("swizzle", INT, 2, [0, 4]),
                ("setslot", INT, 2, [1, 12]),
            ],
        ),
    ],
)
def test_s1_slot_ops_decode(stream: list[int], want: list[tuple]) -> None:
    got = decode(words(stream)).instrs
    assert [(i.name, i.type, i.width, i.operands) for i in got] == want
    # each instruction's length is its opcode's: the stream is used up exactly
    assert 1 + sum(1 + len(i.operands) for i in got) == len(stream)


@pytest.mark.parametrize(
    ("opcode", "kind", "width"),
    [(0x0503, FLOAT, 1), (0x0543, FLOAT, 2), (0x0583, FLOAT, 3), (0x0603, INT, 1),
     (0x0643, INT, 2), (0x0443, BOOL, 1)],
)  # fmt: skip
def test_s1_getslot_opcodes(opcode: int, kind: int, width: int) -> None:
    (got,) = decode(words([1, opcode, 7])).instrs
    assert (got.name, got.type, got.width) == ("getslot", kind, width)
    assert got.operands == [7]


@pytest.mark.parametrize(
    ("opcode", "kind", "width"),
    [(0x0906, FLOAT, 1), (0x0986, FLOAT, 3), (0x0A06, INT, 1), (0x0A46, INT, 2),
     (0x0846, BOOL, 1)],
)  # fmt: skip
def test_s1_setslot_opcodes(opcode: int, kind: int, width: int) -> None:
    (got,) = decode(words([1, opcode, 0, 9])).instrs
    assert (got.name, got.type, got.width) == ("setslot", kind, width)
    assert got.operands == [0, 9]


# S2-S11: evaluation, one slot store shared in node order.


def test_s2_a_constant_value() -> None:
    w = blk(fconst(7.25), setslot(0, 0))
    r = blk(getslot(0), fconst(100.0), ins(ADD, 0, 1))
    store: dict[int, list] = {}
    assert run(w, r, store=store) == [[7.25], [107.25]]
    assert store == {0: [7.25]}


def writer_x_plus_1(slot: int = 0) -> Block:
    """Return S3's writer: input U + 1 into ``slot``."""
    return blk(load(), fconst(1.0), ins(ADD, 0, 1), setslot(2, slot))


@pytest.mark.parametrize(("x", "want"), [(3.5, 4.5), (10.25, 11.25)])
def test_s3_a_live_input(x: float, want: float) -> None:
    assert run(writer_x_plus_1(), blk(getslot(0)), x=x)[1] == [want]


def test_s4_integers() -> None:
    w = blk(iconst(-9), setslot(0, 0, INT))
    r = blk(getslot(0, INT), ins(CONVERT, 0))
    assert run(w, r)[1] == [-9.0]
    w = blk(iconst(-2147483648), setslot(0, 0, INT))
    r = blk(getslot(0, INT), iconst(0), ins(LT, 0, 1, kind=BOOL))
    assert run(w, r)[1] == [True]


@pytest.mark.parametrize(("value", "want"), [(True, 1.0), (False, 0.0)])
def test_s5_bool(*, value: bool, want: float) -> None:
    w = blk(bconst(value=value), setslot(0, 0, BOOL))
    r = blk(getslot(0, BOOL), fconst(1.0), fconst(0.0), ins(IFELSE, 0, 1, 2))
    assert run(w, r)[1] == [want]


def s6_writers() -> list[Block]:
    """Return S6's five writers: a float3, a float, an int2, a float4, a bool."""
    return [
        blk(load(), ins(VEC2, 0, 0, width=2), ins(VEC3, 1, 0, width=3),
            setslot(2, 0, FLOAT, 3)),
        blk(load(), fconst(1.0), ins(ADD, 0, 1), setslot(2, 3)),
        blk(load(), ins(VEC2, 0, 0, width=2), ins(CONVERT, 1, kind=INT, width=2),
            setslot(2, 4, INT, 2)),
        blk(load(), ins(VEC2, 0, 0, width=2), ins(VEC3, 1, 0, width=3),
            ins(VEC4, 2, 0, width=4), setslot(3, 6, FLOAT, 4)),
        blk(load(), fconst(1.0), ins(GT, 0, 1, kind=BOOL), setslot(2, 10, BOOL)),
    ]  # fmt: skip


S6_READER = (
    getslot(0, FLOAT, 3),  # 0
    ins(SWIZZLE, 0, 2),  # 1 z
    getslot(3),  # 2
    ins(ADD, 1, 2),  # 3
    getslot(6, FLOAT, 4),  # 4
    ins(SWIZZLE, 4, 3),  # 5 w
    getslot(10, BOOL),  # 6
    fconst(1.0),  # 7
    fconst(0.0),  # 8
    ins(IFELSE, 6, 7, 8),  # 9
    ins(ADD, 5, 9),  # 10
    ins(ADD, 3, 10),  # 11
)


@pytest.mark.parametrize(
    ("x", "want", "store_words"),
    [
        (2.0, 8.0, [2, 2, 2, 3, 2, 2, 2, 2, 2, 2, True]),
        (0.5, 2.5, [0.5, 0.5, 0.5, 1.5, 0, 0, 0.5, 0.5, 0.5, 0.5, False]),
    ],
)
def test_s6_layout_and_mixed_widths(x: float, want: float, store_words: list) -> None:
    store: dict[int, list] = {}
    got = run(*s6_writers(), blk(*S6_READER), x=x, store=store)
    assert got[-1] == [want]
    # packed without gaps: each value's words from its slot
    flat = []
    for slot in sorted(store):
        assert slot == len(flat)
        flat += store[slot]
    assert flat == store_words
    assert all(isinstance(v, int) for v in store[4])


W1_CHAIN = (load(), fconst(1.0), ins(ADD, 0, 1), setslot(2, 1))
W2_CHAIN = (getslot(1), fconst(10.0), ins(MUL, 0, 1), setslot(2, 0))


@pytest.mark.parametrize(("x", "want"), [(0.5, 15.0), (2.0, 30.0)])
def test_s7_a_chain(x: float, want: float) -> None:
    assert run(blk(*W1_CHAIN), blk(*W2_CHAIN), blk(getslot(0)), x=x)[2] == [want]


def chain_table(w1_node: int = 0, w2_node: int = 1) -> dict[int, list[Writer]]:
    return {
        1: [Writer(w1_node, blk(*W1_CHAIN), FLOAT, 1)],
        0: [Writer(w2_node, blk(*W2_CHAIN), FLOAT, 1)],
    }


@pytest.mark.parametrize(("x", "want"), [(0.5, 15.0), (2.0, 30.0)])
def test_s7_the_store_runs_writers_in_node_order(x: float, want: float) -> None:
    # The lazy store runs each slot's writer (and its writers) when read,
    # with the inputs it is given: live, not frozen.
    store = SlotStore(chain_table(), {U: [x]}, 11, reader=2)
    assert evaluate(blk(getslot(0)), {U: [x]}, slots=store) == [want]
    assert store.values == {1: [x + 1], 0: [f32((x + 1) * 10)]}


def test_s7_a_writer_after_its_reader_is_an_ordering_error() -> None:
    store = SlotStore(chain_table(w1_node=1, w2_node=0), {U: [0.5]}, 11, reader=2)
    with pytest.raises(FunctionError, match="later node"):
        evaluate(blk(getslot(0)), {}, slots=store)
    store = SlotStore(chain_table(), {U: [0.5]}, 11, reader=1)  # reads node 1's
    with pytest.raises(FunctionError, match="later node"):
        evaluate(blk(getslot(0)), {}, slots=store)


def test_s8_two_reads_of_one_slot() -> None:
    r = blk(getslot(0), getslot(0), ins(ADD, 0, 1))
    assert run(writer_x_plus_1(), r, x=2.0)[1] == [6.0]


def s9_writer() -> Block:
    return blk(
        fconst(5.0),
        ins(SETVAR, 0, 0),
        ins(GETVAR, 0),
        fconst(1.0),
        ins(ADD, 2, 3),
        ins(SEQ, 1, 4),
        setslot(5, 0),
    )


def test_s9_variables_do_not_cross_blocks() -> None:
    r = blk(getslot(0), ins(ADD, 0, NO_VALUE))
    assert r.instrs[1].operands == [0, NO_VALUE]
    assert run(s9_writer(), r) == [[6.0], [6.0]]
    variant = blk(getslot(0), ins(GETVAR, 0), ins(ADD, 0, 1))
    assert run(s9_writer(), variant) == [[6.0], [6.0]]


@pytest.mark.parametrize("x", [3.0, 0.7])
def test_s10_exact_transfer(x: float) -> None:
    w = blk(load(), fconst(0.1), ins(MUL, 0, 1), setslot(2, 0))
    r = blk(getslot(0), load(), fconst(0.1), ins(MUL, 1, 2), ins(EQ, 0, 3, kind=BOOL))
    assert run(w, r, x=x)[1] == [True]


def test_s11_special_values() -> None:
    w = blk(fconst(1.0), load(), ins(DIV, 0, 1), setslot(2, 0))
    r = blk(getslot(0), fconst(3e38), ins(GT, 0, 1, kind=BOOL))
    assert run(w, r, x=0.0)[1] == [True]
    w = blk(load(), load(), ins(DIV, 0, 1), setslot(2, 0))
    r = blk(getslot(0), getslot(0), ins(NEQ, 0, 1, kind=BOOL))
    store: dict[int, list] = {}
    assert run(w, r, x=0.0, store=store)[1] == [True]  # d3d11 and sse2's
    assert math.isnan(store[0][0])


# S12: GLSL.


def compiles(source: str, tmp_path: Path) -> None:
    """Assert that glslangValidator compiles ``source`` (a fragment shader)."""
    validator = shutil.which("glslangValidator")
    if validator is None:
        pytest.skip("glslangValidator not installed")
    shader = tmp_path / "slots.frag"
    shader.write_text(source, encoding="utf-8")
    done = subprocess.run(  # noqa: S603
        [validator, str(shader)], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stdout + source


def shader_source(lines: list[str], result: str, t: str = "float") -> str:
    body = "\n\t".join(lines)
    return (
        "#version 450\n"
        f"{t} f(float p) {{\n\t{body}\n\treturn {result};\n}}\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(f(2.0)); }\n"
    )


def glsl_value(lines: list[str], result: str, p: float) -> Any:
    """Run to_glsl's statements in Python: the few forms S12 has."""

    class Vec(list):
        def __getattr__(self, name: str) -> Any:
            return self["xyzw".index(name)]

    env: dict[str, Any] = {
        "p": p,
        "vec3": lambda a: Vec([a] * 3),
        "vec4": lambda a: Vec([a] * 4),
    }
    for line in lines:
        name, expr = line.strip().removesuffix(";").split(" = ", 1)
        expr = re.sub(r"\((\w+) \? (\w+) : (\w+)\)", r"(\2 if \1 else \3)", expr)
        env[name.split()[-1]] = eval(expr, {}, env)  # noqa: S307
    return env[result]


def no_input(*_: int) -> str:
    raise AssertionError


def test_s12_glsl_reads_the_resolvers_expression(tmp_path: Path) -> None:
    r = blk(getslot(0), fconst(100.0), ins(ADD, 0, 1))
    asked = []

    def resolve(slot: int, kind: int, width: int) -> str:
        asked.append((slot, kind, width))
        return "7.25"

    lines, result = to_glsl(r, "f", no_input, no_input, slots=resolve)
    compiles(shader_source(lines, result), tmp_path)
    assert asked == [(0, FLOAT, 1)]
    assert lines[0] == "float f_0 = 7.25;"
    assert not any("slot" in line for line in lines)  # no variable for the slot
    assert glsl_value(lines, result, 0.0) == 107.25


@pytest.mark.parametrize(("p", "want"), [(2.0, 8.0), (0.5, 2.5)])
def test_s12_glsl_of_mixed_widths(p: float, want: float, tmp_path: Path) -> None:
    exprs = {
        (0, FLOAT, 3): "vec3(p)",
        (3, FLOAT, 1): "p + 1.0",
        (6, FLOAT, 4): "vec4(p)",
        (10, BOOL, 1): "p > 1.0",
    }
    lines, result = to_glsl(
        blk(*S6_READER), "f", no_input, no_input, slots=lambda *k: exprs[k]
    )
    compiles(shader_source(lines, result), tmp_path)
    assert glsl_value(lines, result, p) == want
    # the same as S6's evaluation with the writers
    assert run(*s6_writers(), blk(*S6_READER), x=p)[-1] == [want]


def test_s12_glsl_without_a_resolver_fails_clearly() -> None:
    with pytest.raises(FunctionError, match="0x03"):
        to_glsl(blk(getslot(0)), "f", no_input, no_input)
    with pytest.raises(FunctionError, match="0x03"):
        evaluate(blk(getslot(0)), {})


# S13 and 2.5: operands that name no value.


def test_s13_a_no_value_operand_decodes_and_evaluates() -> None:
    stream = [2, 0x0503, 0x0000, 0x0912, 0x0000, 0xFFFF]
    r = decode(words(stream))
    assert [i.name for i in r.instrs] == ["getslot", "add"]
    assert r.instrs[1].operands == [0, NO_VALUE]
    assert evaluate(r, {}, slots={0: [6.0]}) == [6.0]
    # a vector's zero spreads too
    v = blk(ins(GETSLOT, 0, width=3), ins(ADD, NO_VALUE, 0, width=3))
    assert evaluate(v, {}, slots={0: [1.0, 2.0, 3.0]}) == [1.0, 2.0, 3.0]


def test_a_no_value_operand_glsl_and_bounds(tmp_path: Path) -> None:
    r = blk(getslot(0), ins(ADD, 0, NO_VALUE))
    lines, result = to_glsl(r, "f", no_input, no_input, slots=lambda *_: "p")
    assert lines[1] == "float f_1 = (f_0 + float(0));"
    compiles(shader_source(lines, result), tmp_path)
    got = bounds(r, no_input, {}, slots=lambda *_: [(1.0, 2.0)])
    assert got == [(1.0, 2.0)]
    assert bounds(r, no_input, {}) == [(-math.inf, math.inf)]


V = 0x5A3B1E02
"""S14's second float input (y)."""
N = NO_VALUE
NEG, MOD, MIN, MAX, DOT, LERP, POW, SIN, COS = (
    0x17, 0x16, 0x30, 0x31, 0x18, 0x2F, 0x36, 0x27, 0x26,
)  # fmt: skip

S14 = [
    ([ins(ADD, N, 0)], [3.0]),
    ([ins(0x13, N, 0)], [-3.0]),
    ([ins(0x13, 0, N)], [3.0]),
    ([ins(MUL, N, 0)], [0.0]),
    ([ins(DIV, N, 0)], [0.0]),
    ([ins(DIV, 0, N)], [math.inf]),
    ([ins(MOD, N, 0)], [0.0]),
    ([ins(MIN, N, 0)], [0.0]),
    ([ins(NEG, 0), ins(MAX, N, 2)], [0.0]),
    ([ins(NEG, 0), ins(GT, N, 2, kind=BOOL)], [True]),
    ([ins(EQ, N, 1, kind=BOOL)], [False]),
    ([ins(LERP, N, 0, 1)], [1.5]),
    ([ins(POW, 0, N)], [1.0]),
    ([ins(GT, 0, 1, kind=BOOL), ins(IFELSE, 2, N, 0)], [0.0]),
    ([ins(VEC2, N, 0, width=2)], [3.0, 0.0]),
    ([ins(VEC2, 0, N, width=2)], [3.0, 0.0]),
    ([ins(VEC2, 0, N, width=2), ins(VEC3, 2, 0, width=3)], [3.0, 0.0, 3.0]),
    ([ins(VEC2, 0, 1, width=2), ins(DOT, N, 2)], [0.0]),
    ([ins(CONVERT, 0, kind=INT), ins(ADD, N, 2, kind=INT)], [3]),
]
"""Section 2.8's S14: (instructions after input U and input V, value)."""


def s14_block(tail: list[Ins]) -> Block:
    return blk(load(U), load(V), *tail)


def two_inputs(uid: int, *_: int) -> str:
    return {U: "p", V: "q"}[uid]


@pytest.mark.parametrize(("tail", "want"), S14)
def test_s14_no_value_operands(tail: list[Ins], want: list, tmp_path: Path) -> None:
    b = s14_block(tail)
    got = evaluate(b, {U: [3.0], V: [0.5]})
    assert got == want
    assert all(type(g) is type(w) for g, w in zip(got, want, strict=True))
    lines, result = to_glsl(b, "f", two_inputs, no_input)
    last = b.instrs[-1]
    t = translate.glsl_type(last.type, last.width)
    body = "\n\t".join(lines)
    x = ".x" if last.width > 1 else ""
    compiles(
        "#version 450\n"
        f"{t} f(float p, float q) {{\n\t{body}\n\treturn {result};\n}}\n"
        "layout(location = 0) out vec4 color;\n"
        f"void main() {{ color = vec4(float(f(3.0, 0.5){x})); }}\n",
        tmp_path,
    )
    # bounds at the inputs' values hold the value
    fixed = {U: [(3.0, 3.0)], V: [(0.5, 0.5)]}
    got_bounds = bounds(b, lambda uid, _w: fixed[uid], {})
    for (lo, hi), w in zip(got_bounds, want, strict=True):
        assert lo <= float(w) <= hi


def test_s14_vector_glsl_pads_with_zeros() -> None:
    lines, _ = to_glsl(s14_block(S14[14][0]), "f", two_inputs, no_input)
    assert lines[-1] == "vec2 f_2 = vec2(f_0, float(0));"
    lines, _ = to_glsl(s14_block(S14[17][0]), "f", two_inputs, no_input)
    assert lines[-1] == "float f_3 = dot(vec2(0), f_2);"  # the other's type
    b = s14_block([ins(VEC2, N, 0, width=2)])
    assert bounds(b, lambda _u, _w: [(1.0, 2.0)], {}) == [(1.0, 2.0), (0.0, 0.0)]


def test_unlisted_ops_read_no_value_as_zero() -> None:
    # 2.5: [inferred] for an op not listed: a unary op, a swizzle, a
    # convert, a set variable and a set slot
    for tail, want in (
        ([ins(SIN, N)], [0.0]),
        ([ins(SWIZZLE, N, 0b0100, width=2)], [0.0, 0.0]),
        ([ins(CONVERT, N, kind=INT)], [0]),
        ([ins(SETVAR, N, 0)], [0.0]),
        ([setslot(N, 0)], [0.0]),
    ):
        assert evaluate(s14_block(tail), {U: [3.0], V: [0.5]}, slots={}) == want
        to_glsl(s14_block(tail), "f", two_inputs, no_input)  # no IndexError


# S15, S16: a Value Processor's built-ins and trig (2.4).


@pytest.mark.parametrize(
    ("builtin", "selector", "want"),
    [(1, 0, 8.0), (1, 1, 2.0), (3, 0, 3.0), (3, 1, 1.0), (8, 0, 0.0), (8, 1, 0.0)],
)
def test_s15_value_processor_builtins(builtin: int, selector: int, want: float) -> None:
    w = blk(ins(0x01, builtin, width=2), ins(SWIZZLE, 0, selector), setslot(1, 0))
    store: dict[int, list] = {}
    evaluate(w, {}, slot_builtins((3, 1)), slots=store)  # $outputsize (3, 1)
    assert store == {0: [want]}


def test_s15_time_number_and_depth_read_zero() -> None:
    for ident in (0, 10, 12):
        w = blk(ins(0x01, ident), fconst(1.0), ins(ADD, 0, 1), setslot(2, 0))
        table = {0: [Writer(0, w, FLOAT, 1)]}
        assert SlotStore(table, {}, 5, reader=1)[0] == [1.0]
    assert slot_builtins(4)[1] == [16.0, 16.0]
    assert slot_builtins(4)[3] == [4.0, 4.0]


@pytest.mark.parametrize(
    ("op", "x", "bits", "true_bits"),
    [(SIN, 1.382, 0x3F7B7469, 0x3F7B737B), (COS, 0.6, 0x3F53496A, 0x3F534932)],
)
def test_s16_cpu_trig_in_a_value_processor(
    op: int, x: float, bits: int, true_bits: int
) -> None:
    w = blk(load(), ins(op, 0), setslot(1, 0))
    store = SlotStore({0: [Writer(0, w, FLOAT, 1)]}, {U: [x]}, 11, reader=1)
    (got,) = store[0]
    assert struct.unpack("<I", struct.pack("<f", got))[0] == bits
    assert bits != true_bits


def test_s16_a_value_processors_glsl_uses_the_cpu_trig() -> None:
    vp = make_node(0, 40, vp_fields(1), code(load(), ins(SIN, 0), setslot(1, 0)))
    pp = make_node(1, 40, [0], code(getslot(0)))
    mm, _, _ = translate_graph([vp, pp], {"out": 1}, {}, size=10, inputs=amount())
    (n1,) = (n for n in mm if n["name"] == "n1")
    slot_fn = n1["shader_model"]["instance"].split("sbs_slot0_$(name)(vec2 uv)")[1]
    assert "sbs_sin(" in slot_fn.split("return")[0]
    assert "float sbs_sin(float x)" in n1["shader_model"]["global"]


def test_get_slot_edge_cases() -> None:
    # A set slot is its value, store or not; a slot never written reads as
    # zero (2.6); a value of another width is refused.
    assert evaluate(blk(fconst(2.5), setslot(0, 3)), {}) == [2.5]
    assert evaluate(blk(getslot(5, INT, 2)), {}, slots={}) == [0, 0]
    with pytest.raises(FunctionError, match="width"):
        evaluate(blk(getslot(5, FLOAT, 2)), {}, slots={5: [1.0]})
    # rounded to its type, as every op's result is
    assert evaluate(blk(getslot(0)), {}, slots={0: [0.1]}) == [f32(0.1)]


def test_slot_bounds_are_the_writers() -> None:
    table = chain_table()
    bound = slot_bounds(table, lambda _uid, w: [(0.0, 1.0)] * w, 11, reader=2)
    assert bound(0, FLOAT, 1) == [(10.0, 20.0)]  # (U + 1) * 10, U in [0, 1]
    assert bound(0, INT, 1) == [(-math.inf, math.inf)]  # not as written
    assert bound(7, FLOAT, 1) == [(-math.inf, math.inf)]  # no writer


# Value Processors in a graph.


def vp_fields(width: int, images: int = 0) -> list[int]:
    return [width << 16 | images]


def test_value_processor_fields() -> None:
    # 2.4: the first field's high 16 bits are the width, the low the image
    # inputs: only the latter are fields after it.
    def is_offset(w: int) -> bool:
        return w == 0x1000

    assert field_count(40, [0x10000, 0x1000], is_offset) == 1
    assert field_count(40, [0x20001, 5, 0x1000], is_offset) == 2
    assert field_count(40, [2, 5, 6, 0x1000], is_offset) == 3  # a pixel processor
    vp = Node(index=3, start=0, end=0, op=40, fields=[0x30000])
    assert value_width(vp) == 3
    assert value_width(Node(index=3, start=0, end=0, op=40, fields=[1, 2])) == 0
    assert value_width(Node(index=3, start=0, end=0, op=12, fields=[0x30000])) == 0


def test_a_value_processor_in_a_file() -> None:
    # Node 0 a uniform; node 1 a Value Processor (0x20001: a float2, one
    # image input, node 0) whose one block sets slot 0.
    block = code(fconst(1.0), ins(VEC2, 0, 0, width=2), setslot(1, 0, FLOAT, 2))
    vp_at = 0x44
    vp = [1 << 16 | 0x88 << 8 | 40, 0x20001, 0, vp_at + 16 - BASE, *block]
    table = vp_at + 4 * len(vp)
    outputs = table + 8
    data = bytearray(outputs + 32)
    data[:4] = b"SBAM"
    struct.pack_into("<I", data, 0x40, 12)
    struct.pack_into(f"<{len(vp)}I", data, vp_at, *vp)
    struct.pack_into("<2I", data, table, 0x40 - BASE, vp_at - BASE)
    struct.pack_into("<II", data, outputs, 0, 0)
    struct.pack_into("<HHI", data, outputs + 8, 1, 0, 0x1234)
    struct.pack_into("<4I", data, outputs + 16, 2, table - BASE, outputs - BASE, 0)
    nodes = Asm(bytes(data)).nodes()
    assert nodes[1].fields == [0x20001, 0]
    assert nodes[1].inputs == {"input": 0}
    assert nodes[1].code == [vp_at + 16]
    assert value_width(nodes[1]) == 2
    (w,) = writers(nodes)[0]
    assert (w.node, w.type, w.width) == (1, FLOAT, 2)
    assert evaluate(w.block, {}, slots=(store := {})) == [1.0, 1.0]
    assert store == {0: [1.0, 1.0]}


def make_node(index: int, op: int, fields: list[int], *blocks: list[int]) -> Node:
    """Return a node with code (node word flag bit 0) and ``blocks``' offsets."""
    start = 0x1000 * (index + 1)
    node_words = [1 << 16 | 0x88 << 8 | op, *fields, *[0] * len(blocks)]
    offsets = []
    for b in blocks:
        offsets.append(start + 4 * len(node_words))
        node_words += b
    node_words[1 + len(fields) : 1 + len(fields) + len(blocks)] = [
        o - BASE for o in offsets
    ]
    return Node(index=index, start=start, end=start + 4 * len(node_words), op=op,
                flags=1, fields=fields, code=offsets, words=node_words)  # fmt: skip


def amount() -> dict[int, GraphInput]:
    return {U: GraphInput("amount", 0, [3.5])}


def instances(mm: list[dict]) -> dict[str, str]:
    return {
        n["name"]: n["shader_model"]["instance"] for n in mm if n["type"] == "shader"
    }


def glsl_of(instance: str, node: str) -> str:
    """Return a shader node's instance code as a compilable fragment shader."""
    text = instance.replace("$(name)", node).replace("$p_amount", "p_amount")
    return (
        "#version 450\nconst float p_amount = 3.5;\n"
        f"{text}layout(location = 0) out vec4 color;\n"
        f"void main() {{ color = vec4({node}_fct(vec2(0.5))); }}\n"
    )


def test_a_pixel_processor_reads_a_value_live(tmp_path: Path) -> None:
    # n0 is a Value Processor (amount + 1 into slot 0); n1, a pixel
    # processor, returns slot 0 per pixel: the writer's expression, of the
    # amount's parameter, not its value at the default (4.5).
    vp = make_node(
        0, 40, vp_fields(1), code(load(), fconst(1.0), ins(ADD, 0, 1), setslot(2, 0))
    )
    pp = make_node(1, 40, [0], code(getslot(0)))
    assert set(writers([vp, pp])) == {0}
    mm, _, outs = translate_graph([vp, pp], {"out": 1}, {}, size=10, inputs=amount())
    assert outs == {"out": ("n1", 0)}
    assert not any(n["name"].startswith("n0") for n in mm)  # nothing rendered
    code_ = instances(mm)["n1"]
    assert "float sbs_slot0_$(name)(vec2 uv) {" in code_
    assert "$p_amount" in code_
    assert "4.5" not in code_
    assert "sbs_slot0_$(name)(vec2(0.0))" in code_
    assert code_.index("sbs_slot0_$(name)(vec2 uv)") < code_.index("f1_$(name)(vec2")
    compiles(glsl_of(code_, "n1"), tmp_path)
    remote = next(n for n in mm if n["name"] == "parameters")
    assert [w["label"] for w in remote["widgets"]] == ["amount"]


def test_a_chain_of_value_processors_in_glsl(tmp_path: Path) -> None:
    # S7 as nodes: n0 writes slot 1, n1 reads it into slot 0, n2 reads that
    w1 = make_node(0, 40, vp_fields(1), code(*W1_CHAIN))
    w2 = make_node(1, 40, vp_fields(1), code(*W2_CHAIN))
    pp = make_node(2, 40, [0], code(getslot(0), getslot(0), ins(ADD, 0, 1)))
    mm, _, _ = translate_graph([w1, w2, pp], {"out": 2}, {}, size=10, inputs=amount())
    code_ = instances(mm)["n2"]
    assert code_.count("float sbs_slot0_$(name)(vec2 uv)") == 1  # shared by both
    assert code_.count("float sbs_slot1_$(name)(vec2 uv)") == 1
    assert code_.index("sbs_slot1_$(name)(vec2 uv)") < code_.index(
        "sbs_slot0_$(name)(vec2 uv)"
    )
    compiles(glsl_of(code_, "n2"), tmp_path)


def test_an_output_size_reads_a_slot() -> None:
    # A pixel processor's output size block (its first) reads an int2
    # slot that a Value Processor sets from an int2 graph input: 2^9 in a
    # 2^10 graph, from the input's value.
    size_input = 0x11110002
    vp = make_node(
        0, 40, vp_fields(2),
        code(load(size_input, INT, 2), setslot(0, 0, INT, 2)),
    )  # fmt: skip
    pp = make_node(1, 40, [0], code(getslot(0, INT, 2)), code(fconst(0.25)))
    pp.size_code = pp.code[0]  # flag bit 0: its first block, as Asm reads it
    for log2 in (9, 8):
        inputs = {size_input: GraphInput("sizes", 8, [log2, log2])}
        mm, _, _ = translate_graph([vp, pp], {"out": 1}, {}, size=10, inputs=inputs)
        sizes = {n["name"]: n["parameters"]["size"] for n in mm if "size" in n["name"]}
        assert sizes["n1_size"] == log2


def test_a_reader_before_its_writer_is_refused() -> None:
    pp = make_node(0, 40, [0], code(getslot(0)))
    vp = make_node(1, 40, vp_fields(1), code(fconst(1.0), setslot(0, 0)))
    with pytest.raises(UnsupportedError, match="later node"):
        translate_graph([pp, vp], {"out": 0}, {}, size=10)


def test_unknown_or_mismatched_slots_are_refused() -> None:
    vp = make_node(0, 40, vp_fields(1), code(fconst(1.0), setslot(0, 0)))
    for reader in (code(getslot(1)), code(getslot(0, INT))):
        pp = make_node(1, 40, [0], reader)
        with pytest.raises(UnsupportedError, match="slot"):
            translate_graph([vp, pp], {"out": 1}, {}, size=10)


def test_a_value_processors_built_ins() -> None:
    # 2.4: its $size is the graph's ($outputsize 10: 1024), its $pos (0, 0);
    # here read by a pixel processor whose own size is 2^9.
    vp = make_node(
        0, 40, vp_fields(2),
        code(ins(0x01, 1, width=2), ins(0x01, 8, width=2), ins(ADD, 0, 1, width=2),
             setslot(2, 0, FLOAT, 2)),
    )  # fmt: skip
    size_block = code(iconst(9), ins(VEC2, 0, 0, kind=INT, width=2))
    pp = make_node(1, 40, [0], size_block, code(getslot(0, FLOAT, 2)))
    mm, _, _ = translate_graph([vp, pp], {"out": 1}, {}, size=10)
    code_ = instances(mm)["n1"]
    slot_fn = code_.split("sbs_slot0_$(name)(vec2 uv) {")[1].split("}")[0]
    assert "vec2(1024.0, 1024.0)" in slot_fn
    assert "vec2(0.0, 0.0)" in slot_fn
    store = SlotStore(writers([vp, pp]), {}, 10, reader=1)
    assert store[0] == [1024.0, 1024.0]


def test_a_value_processors_image_is_never_read() -> None:
    vp = make_node(0, 40, vp_fields(1), code(fconst(1.0), setslot(0, 0)))
    with pytest.raises(UnsupportedError, match="Value Processor"):
        translate_graph([vp], {"out": 0}, {}, size=10)


def test_an_fx_map_function_reads_a_slot(monkeypatch) -> None:
    vp = make_node(0, 40, vp_fields(1), code(fconst(0.5), setslot(0, 0)))
    fx = make_node(1, 8, [0], code(getslot(0), ins(0x01, 12), ins(ADD, 0, 1)))
    ctx = translate._Context(size=10, slot_writers=writers([vp]), reader=1)
    monkeypatch.setattr(translate, "_CTX", ctx)
    call = translate._fx_function(fx, fx.code[0])
    assert call.endswith("(pos, number, depth, e)")
    helpers = "".join(ctx.helpers)
    assert "float sbs_slot0_$(name)(vec2 uv)" in helpers
    assert "sbs_slot0_$(name)(vec2(0.0))" in helpers
    # its structure's evaluation reads the store
    store = translate._slot_store({})
    assert store is not None
    assert evaluate(decode(fx.words[3:]), {}, {12: [2.0]}, slots=store) == [2.5]


# S17: a library template's slots (2.6).

K = 0x0000AB01
"""The template graph's input k."""


def s17_template() -> list[Node]:
    """Return graph sub: T (k + 1 into slot 0), then R reading slot 0."""
    w = code(load(K), fconst(1.0), ins(ADD, 0, 1), setslot(2, 0))
    t = make_node(0, 40, vp_fields(1), w)
    r = make_node(1, 40, [0], code(getslot(0)))
    return [t, r]


def s17_graph() -> list[Node]:
    """Return graph main and two instances of sub.

    Main's own writer sets 20 into slot 3; the instances, k = 3 and k = 5,
    have their writers folded (4 into slot 2, 6 into slot 1).
    """
    return [
        make_node(0, 40, vp_fields(1), code(fconst(20.0), setslot(0, 3))),
        make_node(1, 40, vp_fields(1), code(fconst(4.0), setslot(0, 2))),
        make_node(2, 40, [0], code(getslot(2))),
        make_node(3, 40, vp_fields(1), code(fconst(6.0), setslot(0, 1))),
        make_node(4, 40, [0], code(getslot(1))),
    ]


def test_s17_template_slots_map_by_writer_node() -> None:
    template, graph = s17_template(), s17_graph()
    assert template_slots({1: 2}, template, graph) == {0: 2}
    assert template_slots({1: 4}, template, graph) == {0: 1}
    # the template reader evaluated as each instance: 4 and 6
    r = writers(template)  # (the template's own slot table: unused here)
    assert set(r) == {0}
    for gk, alias, want in ((2, {0: 2}, 4.0), (4, {0: 1}, 6.0)):
        store = Aliased(SlotStore(writers(graph), {}, 10, reader=gk), alias)
        assert evaluate(blk(getslot(0)), {}, slots=store) == [want]
    # an unmatched template slot is refused
    with pytest.raises(FunctionError, match="matched to no writer"):
        evaluate(blk(getslot(0)), {}, slots=Aliased(SlotStore({}, {}, 10), {}))


def test_s17_template_slots_that_do_not_line_up_map_nothing() -> None:
    template, graph = s17_template(), s17_graph()
    graph[2] = make_node(2, 40, [0], code(getslot(2), getslot(2), ins(ADD, 0, 1)))
    assert template_slots({1: 2}, template, graph) == {}  # two reads, one
    graph[2] = make_node(2, 40, [0], code(getslot(2, INT)))
    assert template_slots({1: 2}, template, graph) == {}  # another type


def test_s17_template_readers_translate_to_their_instances_writers(
    tmp_path: Path,
) -> None:
    template, graph = s17_template(), s17_graph()
    library = {2: (template[1], {}), 4: (template[1], {})}
    maps = {2: template_slots({1: 2}, template, graph),
            4: template_slots({1: 4}, template, graph)}  # fmt: skip
    mm, _, _ = translate_graph(
        graph, {"a": 2, "b": 4}, {}, size=10, library=library, template_slots=maps
    )
    code_ = instances(mm)
    assert "float sbs_slot2_$(name)(vec2 uv)" in code_["n2"]
    assert "4.0" in code_["n2"].split("sbs_slot2_$(name)(vec2 uv)")[1]
    assert "float sbs_slot1_$(name)(vec2 uv)" in code_["n4"]
    assert "6.0" in code_["n4"].split("sbs_slot1_$(name)(vec2 uv)")[1]
    assert "sbs_slot0" not in code_["n2"] + code_["n4"]  # not the template's 0
    compiles(glsl_of(code_["n2"], "n2"), tmp_path)
    # without a map, the template's reads are refused
    with pytest.raises(UnsupportedError, match="matched to no writer"):
        translate_graph(graph, {"a": 2}, {}, size=10, library=library)


def test_slot_reads_stay_live_in_static_values(monkeypatch) -> None:
    # dynamic()'s static value (its graph inputs at their defaults) reads
    # the store: a blur's tap count and the like see the writer's value;
    # its GLSL stays the writer's expression.
    w = code(load(), fconst(1.0), ins(ADD, 0, 1), setslot(2, 0))
    vp = make_node(0, 40, vp_fields(1), w)
    pp = make_node(1, 40, [0], code(getslot(0)))
    ctx = translate._Context(
        size=10, inputs=amount(), slot_writers=writers([vp]), reader=1
    )
    monkeypatch.setattr(translate, "_CTX", ctx)
    value = translate.dynamic(pp, pp.code[0])
    assert value.static == [4.5]
    assert value.text == "f1_$(name)(uv)"
    helpers = json.dumps(ctx.helpers)
    assert "sbs_slot0_$(name)(vec2(0.0))" in helpers
    assert "$p_amount" in helpers
