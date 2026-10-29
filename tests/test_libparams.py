"""Recovering a library node's settings from its cooked instance."""

import struct

from sbsar2ptex.function import FLOAT
from sbsar2ptex.function import decode
from sbsar2ptex.function import evaluate
from sbsar2ptex.libparams import _typed
from sbsar2ptex.libparams import knobs
from sbsar2ptex.libparams import solve
from sbsar2ptex.recognize import Match
from sbsar2ptex.recognize import Template
from sbsar2ptex.sbsasm import BASE
from sbsar2ptex.sbsasm import Node
from sbsar2ptex.translate import GraphInput

RADIUS = 0x1234ABCD


def word(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def block(*instrs: list[int]) -> list[int]:
    half = [len(instrs)]
    for ins in instrs:
        half += ins
    half += [0] * (len(half) % 2)
    return [half[k] | half[k + 1] << 16 for k in range(0, len(half), 2)]


def half_radius() -> list[int]:
    """Return a function: Radius * 0.5 (float32)."""
    load = [2 << 10 | FLOAT << 8 | 0x02, RADIUS & 0xFFFF, RADIUS >> 16]
    half = word(0.5)
    const = [3 << 10 | FLOAT << 8 | 0x00, 0, half & 0xFFFF, half >> 16]  # padded
    mul = [2 << 10 | FLOAT << 8 | 0x14, 0, 1]
    return block(load, const, mul)


def levels(
    start: int, *, low: float | None, code: list[int], high: float | None = None
) -> Node:
    """Return a grayscale levels node: input low inline (``low``), else ``code``.

    ``high``: input high inline too (else its default).
    """
    words = [30, 0, 0, 0, *code]  # node word, a field, padding; code at +16
    inline = [word(low)] if low is not None else [start + 16 - BASE]
    flags = 1 if low is not None else 2
    if high is not None:
        inline.append(word(high))
        flags |= 1 << 2
    return Node(index=0, start=start, end=start + 4 * len(words), op=30,
                fields=[flags], inline=inline, words=words)  # fmt: skip


def instance_of(low: float, high: float | None = None) -> tuple[Match, list[Node]]:
    template = levels(0x100, low=None, code=half_radius())
    t = Template("highpass", "pkg://levels", "out", [template], 0, members=[0])
    t.inputs = {RADIUS: GraphInput("Radius", 0, [6.0], "Radius")}
    graph = [levels(0x400, low=low, code=[], high=high)]
    return Match(t, 0, {0: 0}, {}), graph


def test_the_function_is_evaluated_in_float32() -> None:
    assert evaluate(decode(half_radius()), {RADIUS: [0.3]}) == [
        struct.unpack("<f", struct.pack("<f", 0.15))[0]
    ]


def test_a_constant_gives_back_the_parameter_that_made_it() -> None:
    m, graph = instance_of(0.15)  # 0.3 * 0.5, cooked
    solved = solve(m, graph, [8, 8])
    assert solved == {RADIUS: [0.3]}  # printed as the author typed it
    (radius,) = knobs(m, solved, {}).values()
    assert (radius.label, radius.default, radius.scope) == ("Radius", [0.3], "n0")


def test_an_instance_differing_where_the_template_is_fixed_is_declined() -> None:
    # the template's input high is its default (1): no parameter makes 0.9
    m, graph = instance_of(0.15, high=0.9)
    assert solve(m, graph, [8, 8]) is None


def test_values_print_as_typed() -> None:
    assert _typed(struct.unpack("<f", struct.pack("<f", 3.22))[0]) == 3.22


def test_an_integer_parameter_is_a_live_knob() -> None:
    # Slope blur's Samples: its template is cooked for any value, so it is
    # a knob like any other, not its solved value frozen.
    m, _ = instance_of(0.15)
    samples = 0x0BADF00D
    m.template.inputs[samples] = GraphInput("Samples", 4, [8], "Samples")
    reads = knobs(m, {RADIUS: [0.3], samples: [32]}, {})
    assert (reads[samples].type, reads[samples].default) == (4, [32])
    assert reads[samples].scope == "n0"
