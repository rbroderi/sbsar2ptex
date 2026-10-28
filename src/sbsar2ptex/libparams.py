"""Recover a library node's own settings from its cooked instance.

A library graph cooked alone keeps its parameters (Highpass's Radius, Blur
HQ's Intensity) as graph inputs, its atomic nodes computing their settings
from them through functions; an author's graph has those folded into
constants. ``solve`` finds the parameter values whose functions give every
one of an instance's constants back, bit for bit in float32 (the cooker's
arithmetic), so the instance can be translated from its template with the
values as live knobs (translate_graph's ``library``): the same numbers, now
editable as the author's were in Designer.

Fields are paired by reading both nodes the way translate's readers do
(``slots``). Declined, so the instance keeps its constants: FX-Maps and
pixel processors, a field the instance computes by a function of its own,
and anything not reproduced exactly.
"""

import math
import struct
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from sbsar2ptex.function import BOOL
from sbsar2ptex.function import FLOAT
from sbsar2ptex.function import INT
from sbsar2ptex.function import NO_VALUE
from sbsar2ptex.function import Block
from sbsar2ptex.function import FunctionError
from sbsar2ptex.function import constant
from sbsar2ptex.function import decode
from sbsar2ptex.recognize import Match
from sbsar2ptex.sbsasm import BASE
from sbsar2ptex.sbsasm import COLOR_OPS
from sbsar2ptex.sbsasm import Node
from sbsar2ptex.translate import LEVELS
from sbsar2ptex.translate import GraphInput

UNSOLVED_OPS = (8, 9, 40, 41)
"""FX-Maps and pixel processors: their settings are trees and blocks."""
INT_SLOTS = ("blend_mode", "tiling", "select", "flip")
ULPS = 64
"""How far (float32 steps) around an estimate exact values are looked for."""
INT_RANGE = range(-64, 1025)
SECANT_STEPS = 40
IN_FLOAT, IN_INT = 0, 4

type Slot = tuple[str, Any, list[float | int]]
"""(state, payload, default): "default"; "const", its u32 words; "func",
its code block's offset."""


def _offset(node: Node, word: int) -> bool:
    """Whether ``word`` is a code block's offset inside ``node``."""
    return node.start < word + BASE < node.end and word % 4 == 0


def _strip(node: Node, out: dict[str, Slot]) -> list[int]:
    """Return ``node``'s inline words after its seed and dynamic size (node_shader)."""
    inline = list(node.inline)
    if (node.flags >> 6) & 3 == 1:
        out["seed"] = ("const", inline[:1], [0])
        inline = inline[1:]
    if inline and _offset(node, inline[0]):
        target = inline[0] + BASE
        last = decode(node.at(target, (node.end - target) // 4)).instrs[-1]
        if last.type != FLOAT and last.width == 2:  # noqa: PLR2004
            inline = inline[1:]
    return inline


def _flag(
    node: Node,
    flags: int,
    inline: list[int],
    spec: list[tuple],
    width: int,
    out: dict[str, Slot],
) -> None:
    at = 0
    for name, bit, default, *own in spec:
        w = own[0] if own else width
        state = (flags >> bit) & 3
        d = default if isinstance(default, list) else [default] * w
        if state == 0:
            out[name] = ("default", None, d)
        elif state == 1:
            out[name] = ("const", inline[at : at + w], d)
            at += w
        elif at < len(inline) and _offset(node, inline[at]):
            out[name] = ("func", inline[at] + BASE, d)
            at += 1
        else:
            msg = f"no function for {name}"
            raise ValueError(msg)


def _word(
    node: Node,
    inline: list[int],
    spec: list[tuple],
    widths: dict[str, int],
    out: dict[str, Slot],
) -> None:
    states = node.flags >> 8
    at = 0
    for name, bit, default in spec:
        state = (states >> bit) & 3
        w = widths.get(name, 1)
        d = default if isinstance(default, list) else [default] * w
        if state == 0:
            out[name] = ("default", None, d)
        elif state == 1:
            out[name] = ("const", inline[at : at + w], d)
            at += w
        else:
            out[name] = ("func", inline[at] + BASE, d)
            at += 1


def slots(node: Node) -> dict[str, Slot]:
    """Return ``node``'s settings, as translate's readers find them.

    Raises ValueError for a node this cannot read (UNSOLVED_OPS among them).
    """
    if node.op in UNSOLVED_OPS:
        msg = f"op {node.op}"
        raise ValueError(msg)
    out: dict[str, Slot] = {}
    inline = _strip(node, out)
    op, color = node.op, node.op in COLOR_OPS
    n = 4 if color else 1
    if op in (2, 3):
        flags = node.fields[0]
        state = flags >> 4 & 3
        if state == 3:  # noqa: PLR2004 (an opacity input, its multiplier's state)
            state = (flags >> 9) & 3
        out["blend_mode"] = ("const", [flags & 0xF], [0])
        if state == 2:  # noqa: PLR2004
            out["opacity"] = ("func", inline[0] + BASE, [1.0])
        elif state == 1:
            out["opacity"] = ("const", inline[:1], [1.0])
        else:
            out["opacity"] = ("default", None, [1.0])
    elif op in (4, 5):
        flags = node.fields[0]
        for key, state, w, d in (
            ("matrix", (flags >> 6) & 3, 4, [1.0, 0.0, 0.0, 1.0]),
            ("offset", (flags >> 25) & 3, 2, [0.0, 0.0]),
        ):
            if state == 1:
                out[key] = ("const", inline[:w], d)
                inline = inline[w:]
            elif state == 2:  # noqa: PLR2004
                out[key] = ("func", inline[0] + BASE, d)
                inline = inline[1:]
            else:
                out[key] = ("default", None, d)
        out["tiling"] = ("const", [node.flags >> 8 & 3], [3])
    elif op == 6:  # noqa: PLR2004
        spec = [("weights", 0, [1 / 3, 1 / 3, 1 / 3, 0.0]), ("matte", 2, 0.0)]
        _word(node, inline, spec, {"weights": 4}, out)
    elif op == 7:  # noqa: PLR2004
        out["select"] = ("const", [node.fields[0]], [0])
    elif op in (12, 13):
        state = (node.flags >> 8) & 3
        d = [0.0, 0.0, 0.0, 1.0] if color else [0.0]
        if state == 2:  # noqa: PLR2004
            out["value"] = ("func", inline[0] + BASE, d)
        elif state == 1:
            out["value"] = ("const", inline[:n], d)
        else:
            out["value"] = ("default", None, d)
    elif op in (14, 15):
        _word(node, inline, [("intensity", 5, 1.0)], {}, out)
    elif op in (20, 21, 26, 27):
        _word(node, inline, [("intensity", 4, 10.0 if op < 26 else 1.0)], {}, out)  # noqa: PLR2004
    elif op in (22, 23):
        spec = [("intensity", 0, 10.0), ("angle", 2, 0.0)]
        _flag(node, node.fields[0], inline, spec, 1, out)
    elif op in (24, 25):
        spec = [("intensity", 1, 10.0), ("angle", 3, 0.0)]
        _flag(node, node.fields[0], inline, spec, 1, out)
    elif op == 29:  # noqa: PLR2004
        spec = [("hue", 0, 0.5), ("saturation", 2, 0.5), ("luminosity", 4, 0.5)]
        _word(node, inline, spec, {}, out)
    elif op in (30, 31):
        _flag(node, node.fields[0], inline, LEVELS, n, out)
    elif op == 37:  # noqa: PLR2004
        _flag(node, node.fields[0] & 3, inline, [("intensity", 0, 1.0)], 1, out)
        out["flip"] = ("const", [node.fields[0] & 4], [0])
    elif op in (32, 33):
        pass
    else:
        msg = f"op {op}"
        raise ValueError(msg)
    return out


def _f32(x: float) -> float:
    """Return ``x`` rounded to float32 (infinite past its range)."""
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def _wrap(x: float) -> int:
    x = int(x) & 0xFFFFFFFF
    return x - (1 << 32) if x >> 31 else x


def _div(a: float, b: float, kind: int) -> float:
    if kind == INT:
        return int(a / b) if b else 0
    if b:
        return a / b
    return math.copysign(math.inf, a) if a else math.nan


def _guard(fn: Callable[[float], float]) -> Callable[[float], float]:
    def safe(a: float) -> float:
        try:
            return fn(a)
        except (ValueError, OverflowError):
            return math.nan

    return safe


_BINARY: dict[int, Callable[[float, float], Any]] = {
    0x12: lambda a, b: a + b,
    0x13: lambda a, b: a - b,
    0x14: lambda a, b: a * b,
    0x16: lambda a, b: (a - b * math.floor(a / b)) if b else 0.0,
    0x1A: lambda a, b: bool(a) and bool(b),
    0x1B: lambda a, b: bool(a) or bool(b),
    0x1D: lambda a, b: a == b,
    0x1E: lambda a, b: a != b,
    0x1F: lambda a, b: a > b,
    0x20: lambda a, b: a >= b,
    0x21: lambda a, b: a < b,
    0x22: lambda a, b: a <= b,
    0x30: min,
    0x31: max,
    0x36: lambda a, b: a**b if a >= 0 else math.nan,
}
_UNARY: dict[int, Callable[[float], Any]] = {
    0x17: lambda a: -a,
    0x1C: lambda a: not a,
    0x23: abs,
    0x24: _guard(math.floor),
    0x25: _guard(math.ceil),
    0x26: _guard(math.cos),
    0x27: _guard(math.sin),
    0x28: lambda a: math.sqrt(a) if a >= 0 else math.nan,
    0x29: lambda a: math.log(a) if a > 0 else -math.inf,
    0x2A: _guard(math.exp),
    0x2B: _guard(lambda a: 2.0**a),
    0x2C: _guard(math.tan),
    0x35: lambda a: math.log2(a) if a > 0 else -math.inf,
}


def _spread(a: list, b: list) -> tuple[list, list]:
    """Return two operands of one width: a scalar repeated to the other's."""
    if len(a) == len(b):
        return a, b
    return (a * len(b), b) if len(a) == 1 else (a, b * len(a))


def _as(kind: int, values: list) -> list:
    if kind == FLOAT:
        return [_f32(float(x)) for x in values]
    if kind == INT:
        return [_wrap(x) if math.isfinite(x) else 0 for x in values]
    return [bool(x) for x in values]


def evaluate32(block: Block, inputs: dict[int, list]) -> list:
    """Return ``block``'s value, every step rounded to float32 as cooking does.

    Only what a library parameter's function reads: graph inputs, no
    built-ins, random numbers or samples (FunctionError).
    """
    vals: list[list] = []
    var: dict[int, list] = {}
    for ins in block.instrs:
        ops, op = ins.operands, ins.op
        if op == 0x00:
            v = constant(ins)
        elif op == 0x02:  # noqa: PLR2004
            v = inputs[ins.u32s()[0]]
        elif op == 0x04:  # noqa: PLR2004
            v = var[ops[0]]
        elif op == 0x07:  # noqa: PLR2004
            var[ops[1]] = v = vals[ops[0]]
        elif op == 0x09:  # noqa: PLR2004
            v = vals[ops[1]] if vals[ops[0]][0] else vals[ops[2]]
        elif op == 0x0C:  # noqa: PLR2004
            v = [0.0] * ins.width if ops[1] == NO_VALUE else vals[ops[1]]
        elif op in (0x0D, 0x0E, 0x0F):
            v = [x for o in ops for x in vals[o]]
        elif op == 0x10:  # noqa: PLR2004
            src, sel = vals[ops[0]], ops[1]
            v = [src[sel >> (2 * k) & 3] for k in range(ins.width)]
        elif op == 0x11:  # noqa: PLR2004
            v = vals[ops[0]]
        elif op == 0x18:  # noqa: PLR2004
            acc = 0.0
            for x, y in zip(vals[ops[0]], vals[ops[1]], strict=True):
                acc = _f32(acc + _f32(x * y))
            v = [acc]
        elif op == 0x2F:  # noqa: PLR2004
            a, b = vals[ops[0]], vals[ops[1]]
            t = vals[ops[2]] * (len(a) if len(vals[ops[2]]) == 1 else 1)
            v = [
                _f32(p + _f32(_f32(q - p) * s)) for p, q, s in zip(a, b, t, strict=True)
            ]
        elif op in _BINARY or op == 0x15:  # noqa: PLR2004 (0x15 divides)
            a, b = _spread(vals[ops[0]], vals[ops[1]])
            if op == 0x15:  # noqa: PLR2004
                v = [_div(p, q, ins.type) for p, q in zip(a, b, strict=True)]
            else:
                v = [_BINARY[op](p, q) for p, q in zip(a, b, strict=True)]
        elif op in _UNARY:
            v = [_UNARY[op](p) for p in vals[ops[0]]]
        else:
            msg = f"cannot evaluate {ins.name} here"
            raise FunctionError(msg)
        if op not in (0x07, 0x09, 0x0C) and ins.type in (FLOAT, INT, BOOL):
            v = _as(ins.type, v)
        vals.append(v)
    return vals[-1]


def _words(values: list, kind: int) -> list[int]:
    """Return values as the u32 words a node stores them in."""
    if kind == FLOAT:
        return [
            struct.unpack("<I", struct.pack("<f", _f32(float(x))))[0] for x in values
        ]
    return [int(x) & 0xFFFFFFFF for x in values]


def _ulps(x: float, k: int) -> list[float]:
    """Return the float32 values within ``k`` steps of ``x``, nearest first."""
    if not math.isfinite(x):
        return []
    bits = struct.unpack("<i", struct.pack("<f", _f32(x)))[0]
    out = []
    for d in sorted(range(-k, k + 1), key=abs):
        b = bits + d
        if -(1 << 31) <= b < (1 << 31):
            v = struct.unpack("<f", struct.pack("<i", b))[0]
            if math.isfinite(v):
                out.append(v)
    return out


def _short(v: float) -> tuple[int, float]:
    """Order values by how short they print: the author's typed values first."""
    return len(repr(_typed(v))), abs(v)


def _typed(v: float) -> float:
    """Return the shortest decimal that is float32 ``v``: 3.22, not 3.2200000286."""
    for digits in range(1, 10):
        x = float(f"{v:.{digits}g}")
        if _f32(x) == _f32(v):
            return x
    return v


class _Equation:
    """A template function that must give an instance's stored words."""

    def __init__(self, block: Block, target: list[int], kind: int) -> None:
        self.block, self.target, self.kind = block, target, kind
        self.reads = block.uids()

    def holds(self, values: dict[int, list]) -> bool:
        try:
            got = evaluate32(self.block, values)
        except (FunctionError, KeyError, IndexError, ZeroDivisionError):
            return False
        return _words(got, self.kind) == self.target

    def value(self, values: dict[int, list]) -> float | None:
        try:
            got = evaluate32(self.block, values)
        except (FunctionError, KeyError, IndexError, ZeroDivisionError):
            return None
        return float(got[0]) if got else None


def _solve_one(eq: _Equation, uid: int, gi: GraphInput, values: dict) -> list | None:
    """Return the value of input ``uid`` (all else known) that makes ``eq`` hold."""

    def at(x: float) -> dict:
        return {**values, uid: [x]}

    if gi.type == IN_INT:
        for x in sorted(INT_RANGE, key=lambda k: abs(k - (gi.default or [0])[0])):
            if eq.holds(at(x)):
                return [x]
        return None
    if gi.type != IN_FLOAT or eq.kind != FLOAT:
        return None
    target = struct.unpack("<f", struct.pack("<I", eq.target[0]))[0]
    guesses = [float(gi.default[0]) if gi.default else 0.0, target]
    y0, y1 = eq.value(at(0.0)), eq.value(at(1.0))
    if y0 is not None and y1 is not None and y1 != y0:
        guesses.append((target - y0) / (y1 - y0))
    # a few secant steps from each guess, on the first component
    for start in list(guesses):
        x0, x1 = start, start * 1.01 + 0.01
        for _ in range(SECANT_STEPS):
            f0, f1 = eq.value(at(x0)), eq.value(at(x1))
            if f0 is None or f1 is None or f1 == f0:
                break
            x0, x1 = x1, x1 - (f1 - target) * (x1 - x0) / (f1 - f0)
            if not math.isfinite(x1):
                break
        if math.isfinite(x1):
            guesses.append(x1)
    found = [x for g in guesses for x in _ulps(g, ULPS) if eq.holds(at(x))]
    return [_typed(min(found, key=_short))] if found else None


def solve(  # noqa: PLR0911
    m: Match, nodes: list[Node], outputsize: list[int]
) -> dict[int, list] | None:
    """Return the library parameters (by template uid) for instance ``m``'s constants.

    None when a member cannot be read, or not every constant is given back
    exactly. ``outputsize`` is the $outputsize templates' functions see.
    """
    t = m.template
    builtin = {
        uid: (list(outputsize) if gi.identifier == "$outputsize" else [0])
        for uid, gi in t.inputs.items()
        if gi.identifier.startswith("$")
    }
    params = {uid: gi for uid, gi in t.inputs.items() if uid not in builtin}
    eqs: list[_Equation] = []
    for tk, gk in m.members.items():
        tn, gn = t.nodes[tk], nodes[gk]
        try:
            ts, gs = slots(tn), slots(gn)
        except (ValueError, IndexError):
            return None
        if ts.keys() - {"seed"} != gs.keys() - {"seed"}:
            return None
        for key, (state, payload, default) in ts.items():
            if key == "seed":
                continue
            g_state, g_payload, _ = gs[key]
            kind = INT if key in INT_SLOTS else FLOAT
            if g_state == "func":
                return None  # the instance computes it: not handled yet
            target = (
                [int(w) for w in g_payload]
                if g_state == "const"
                else _words(default, kind)
            )
            if state == "func":
                block = decode(tn.at(payload, (tn.end - payload) // 4))
                last = block.instrs[-1]
                kind = FLOAT if last.type == FLOAT else INT
                eqs.append(_Equation(block, target, kind))
            elif (
                [int(w) for w in payload] if state == "const" else _words(default, kind)
            ) != target:
                return None  # a constant of the template's differs: not this node
    values: dict[int, list] = {
        uid: list(gi.default) for uid, gi in params.items()
    } | builtin
    known = set(builtin)
    progress = True
    while progress:
        progress = False
        for eq in eqs:
            open_ = eq.reads - known
            if len(open_) == 1:
                (uid,) = open_
                got = _solve_one(eq, uid, params[uid], values)
                if got is None:
                    return None
                values[uid] = got
                known.add(uid)
                progress = True
    if not all(eq.holds(values) for eq in eqs):
        return None
    read = set().union(*(eq.reads for eq in eqs)) if eqs else set()
    return {uid: values[uid] for uid in read if uid in params}


def knobs(
    m: Match, solved: dict[int, list], inputs: dict[int, GraphInput]
) -> dict[int, GraphInput]:
    """Return the inputs translating instance ``m`` from its template reads.

    Its solved parameters, as the instance's own (named after its root, so
    that 131 Highpasses keep 131 Radiuses); the template's $outputsize and
    $randomseed as the graph's (``inputs``). An integer parameter often
    decides the cooked graph's structure (Slope blur's Samples: its count
    of steps), which a knob could not change: it stays the solved constant.
    """
    by_identifier = {gi.identifier: gi for gi in inputs.values()}
    out = {}
    for uid, gi in m.template.inputs.items():
        if gi.identifier.startswith("$"):
            if gi.identifier in by_identifier:
                out[uid] = by_identifier[gi.identifier]
        elif uid in solved:
            out[uid] = replace(
                gi, default=solved[uid], scope=f"n{m.root}", fixed=gi.type == IN_INT
            )
    return out
