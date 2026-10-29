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
from dataclasses import replace
from typing import Any

from sbsar2ptex.function import FLOAT
from sbsar2ptex.function import INT
from sbsar2ptex.function import Block
from sbsar2ptex.function import FunctionError
from sbsar2ptex.function import decode
from sbsar2ptex.function import evaluate
from sbsar2ptex.function import f32
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


def _words(values: list, kind: int) -> list[int]:
    """Return values as the u32 words a node stores them in."""
    if kind == FLOAT:
        return [
            struct.unpack("<I", struct.pack("<f", f32(float(x))))[0] for x in values
        ]
    return [int(x) & 0xFFFFFFFF for x in values]


def _ulps(x: float, k: int) -> list[float]:
    """Return the float32 values within ``k`` steps of ``x``, nearest first."""
    if not math.isfinite(x):
        return []
    bits = struct.unpack("<i", struct.pack("<f", f32(x)))[0]
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
        if f32(x) == f32(v):
            return x
    return v


class _Equation:
    """A template function that must give an instance's stored words."""

    def __init__(self, block: Block, target: list[int], kind: int) -> None:
        self.block, self.target, self.kind = block, target, kind
        self.reads = block.uids()

    def holds(self, values: dict[int, list]) -> bool:
        try:
            got = evaluate(self.block, values)
        except (FunctionError, KeyError, IndexError, ZeroDivisionError):
            return False
        return _words(got, self.kind) == self.target

    def value(self, values: dict[int, list]) -> float | None:
        try:
            got = evaluate(self.block, values)
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
    $randomseed as the graph's (``inputs``). Integer parameters are knobs
    too: a solved one is read by the template's functions only, its nodes
    cooked for any value of it (Slope blur's 32 warps and 31 blends,
    whatever its Samples, which divides the warps' intensity and switches
    the blends past it off).
    """
    by_identifier = {gi.identifier: gi for gi in inputs.values()}
    out = {}
    for uid, gi in m.template.inputs.items():
        if gi.identifier.startswith("$"):
            if gi.identifier in by_identifier:
                out[uid] = by_identifier[gi.identifier]
        elif uid in solved:
            out[uid] = replace(gi, default=solved[uid], scope=f"n{m.root}")
    return out
