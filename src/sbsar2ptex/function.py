"""Substance function bytecode: decode, evaluate and translate to GLSL.

A parameter "driven by a function" (a Substance function graph) is compiled
to a code block: a u16 instruction count, then instructions, in SSA form
(each instruction's value is numbered in order; operands name earlier
values by number). The value of the block is its last instruction's.

An instruction is a u16 opcode, then its operand u16s. The opcode is

    (operand u16 count << 10) | (type << 8) | (width - 1) << 6 | op

with type 1 float, 2 int, 0 bool (whose width field reads 1), so every
instruction's length is in its opcode. Operands that are u32 or f32
(uids, constants) are 4-byte aligned, a u16 of padding first where
needed (it is counted).

Ops, from function graphs cooked by sbscooker (see sbsasm-probe):

    0x00 constant       0x01 built-in (id)   0x02 graph input (uid)
    0x04 get variable   0x07 set variable    0x09 if/else (c, a, b)
    0x0c sequence       0x0d-0x0f vector of 2-4 parts   0x10 swizzle
    0x11 convert        0x12 add  0x13 sub   0x14 mul  0x15 div  0x16 mod
    0x17 neg  0x18 dot  0x1a and  0x1b or  0x1c not
    0x1d ==  0x1e !=  0x1f >  0x20 >=  0x21 <  0x22 <=
    0x23 abs  0x24 floor  0x25 ceil  0x26 cos  0x27 sin  0x28 sqrt
    0x29 log  0x2a exp  0x2b pow2  0x2c tan  0x2d atan2  0x2e cartesian
    0x2f lerp  0x30 min  0x31 max  0x32 rand  0x35 log2  0x36 pow
    0x33 samplelum, 0x34 samplecol (position, input index[, filtering])

Built-ins: 1 $size, 3 $sizelog2, 8 $pos (a pixel processor's pixel).
"""

import math
import struct
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any

FLOAT, INT, BOOL = 1, 2, 0
NO_VALUE = 0xFFFF
"""An operand naming no value: a sequence whose result the cooker dropped."""

OP_NAMES = {
    0x00: "const", 0x01: "builtin", 0x02: "input", 0x04: "getvar", 0x07: "setvar",
    0x09: "ifelse", 0x0C: "sequence", 0x0D: "concat", 0x0E: "concat3",
    0x0F: "concat4", 0x10: "swizzle",
    0x11: "convert", 0x12: "add", 0x13: "sub", 0x14: "mul", 0x15: "div",
    0x16: "mod", 0x17: "neg", 0x18: "dot", 0x1A: "and", 0x1B: "or", 0x1C: "not",
    0x1D: "eq", 0x1E: "neq", 0x1F: "gt", 0x20: "gteq", 0x21: "lt", 0x22: "lteq",
    0x23: "abs", 0x24: "floor", 0x25: "ceil", 0x26: "cos", 0x27: "sin",
    0x28: "sqrt", 0x29: "log", 0x2A: "exp", 0x2B: "pow2", 0x2C: "tan",
    0x2D: "atan2", 0x2E: "cartesian", 0x2F: "lerp", 0x30: "min", 0x31: "max",
    0x32: "rand", 0x33: "samplelum", 0x34: "samplecol", 0x35: "log2", 0x36: "pow",
}  # fmt: skip
BUILTINS = {1: "$size", 3: "$sizelog2", 8: "$pos"}
"""Built-in values by id (more to come: $pos, $time...)."""


class FunctionError(Exception):
    """Bytecode the decoder or translator does not understand yet."""


@dataclass
class Instr:
    """One instruction: its op, result type and width, and raw operand u16s."""

    op: int
    type: int
    width: int
    operands: list[int]

    @property
    def name(self) -> str:
        """The op's name."""
        return OP_NAMES.get(self.op, f"op{self.op:#x}")

    def aligned(self) -> list[int]:
        """Return the operands without the alignment padding before u32s."""
        n = len(self.operands)
        if self.op in (0x00, 0x02) and n % 2 and self.type != BOOL:
            return self.operands[1:]  # bools take one u16 a component, no padding
        return self.operands

    def u32s(self) -> list[int]:
        """Return the (aligned) operands as u32s."""
        ops = self.aligned()
        return [ops[k] | ops[k + 1] << 16 for k in range(0, len(ops) - 1, 2)]


@dataclass
class Block:
    """A decoded code block: instructions; its value is the last one's."""

    instrs: list[Instr] = field(default_factory=list)

    def uids(self) -> set[int]:
        """Return the graph inputs (uids) the block reads."""
        return {i.u32s()[0] for i in self.instrs if i.op == 0x02}  # noqa: PLR2004


def u16s(words: list[int]) -> list[int]:
    """Return u32 words as their u16 halves, low first."""
    out: list[int] = []
    for w in words:
        out += [w & 0xFFFF, w >> 16]
    return out


def decode(words: list[int]) -> Block:
    """Decode the code block at the start of ``words`` (u32s)."""
    half = u16s(words)
    count, at = half[0], 1
    block = Block()
    for _ in range(count):
        code = half[at]
        n = code >> 10
        kind = code >> 8 & 3
        width = (code >> 6 & 3) + 1
        if kind == BOOL:
            width = 1
        block.instrs.append(Instr(code & 0x3F, kind, width, half[at + 1 : at + 1 + n]))
        at += 1 + n
    return block


def constant(instr: Instr) -> list[float] | list[int] | list[bool]:
    """Return a constant instruction's value, a list of its components."""
    ops = instr.aligned()
    if instr.type == BOOL:
        return [bool(ops[0])]
    words = [ops[k] | ops[k + 1] << 16 for k in range(0, 2 * instr.width, 2)]
    if instr.type == FLOAT:
        return [struct.unpack("<f", struct.pack("<I", w))[0] for w in words]
    return [struct.unpack("<i", struct.pack("<I", w))[0] for w in words]


# Python evaluation (for checking and constant folding). Values are lists
# of components (one for scalars).

_BINARY: dict[int, Callable[[float, float], Any]] = {
    0x12: lambda a, b: a + b,
    0x13: lambda a, b: a - b,
    0x14: lambda a, b: a * b,
    0x15: lambda a, b: a / b if b else math.inf,
    0x16: lambda a, b: a - b * math.floor(a / b) if b else 0.0,
    0x1A: lambda a, b: a and b,
    0x1B: lambda a, b: a or b,
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
    0x24: math.floor,
    0x25: math.ceil,
    0x26: math.cos,
    0x27: math.sin,
    0x28: lambda a: math.sqrt(a) if a >= 0 else math.nan,
    0x29: lambda a: math.log(a) if a > 0 else -math.inf,
    0x2A: math.exp,
    0x2B: lambda a: 2.0**a,
    0x2C: math.tan,
    0x35: lambda a: math.log2(a) if a > 0 else -math.inf,
}


def _bcast(a: list, b: list) -> tuple[list, list]:
    if len(a) == len(b):
        return a, b
    if len(a) == 1:
        return a * len(b), b
    return a, b * len(a)


def evaluate(
    block: Block,
    inputs: dict[int, list[float]],
    builtins: dict[int, list[float]],
    rand: Callable[[float], float] | None = None,
    variables: dict[int, list] | None = None,
) -> list:
    """Return the block's value with graph ``inputs`` (by uid) and ``builtins``.

    ``rand(a)``, when given, is the generator rand instructions call, in
    instruction order. ``variables``, when given, is the variable store,
    read and written in place: blocks that share one (an FX-Map's) see each
    other's variables.
    """
    vals: list[list] = []
    var: dict[int, list] = {} if variables is None else variables
    for ins in block.instrs:
        ops = ins.operands
        if ins.op == 0x00:
            v = constant(ins)
        elif ins.op == 0x01:
            v = builtins[ops[0]]
        elif ins.op == 0x02:  # noqa: PLR2004
            v = inputs[ins.u32s()[0]]
        elif ins.op == 0x04:  # noqa: PLR2004
            v = var.get(ops[0], [0.0] * ins.width)
        elif ins.op == 0x07:  # noqa: PLR2004
            var[ops[1]] = vals[ops[0]]
            v = vals[ops[0]]
        elif ins.op == 0x09:  # noqa: PLR2004
            v = vals[ops[1]] if vals[ops[0]][0] else vals[ops[2]]
        elif ins.op == 0x0C:  # noqa: PLR2004
            v = [0.0] * ins.width if ops[1] == NO_VALUE else vals[ops[1]]
        elif ins.op in (0x0D, 0x0E, 0x0F):
            v = [x for o in ops for x in vals[o]]
        elif ins.op == 0x10:  # noqa: PLR2004
            src, sel = vals[ops[0]], ops[1]
            v = [src[sel >> (2 * k) & 3] for k in range(ins.width)]
        elif ins.op == 0x11:  # noqa: PLR2004
            conv = float if ins.type == FLOAT else int
            v = [conv(x) for x in vals[ops[0]]]
        elif ins.op == 0x18:  # noqa: PLR2004
            v = [sum(x * y for x, y in zip(vals[ops[0]], vals[ops[1]], strict=True))]
        elif ins.op == 0x2D:  # noqa: PLR2004
            x, y = vals[ops[0]][:2]
            v = [math.atan2(y, x)]
        elif ins.op == 0x2E:  # noqa: PLR2004
            rho, theta = vals[ops[0]][0], vals[ops[1]][0]
            v = [rho * math.cos(theta), rho * math.sin(theta)]
        elif ins.op == 0x2F:  # noqa: PLR2004
            a, b = _bcast(vals[ops[0]], vals[ops[1]])
            x = vals[ops[2]] * (len(a) if len(vals[ops[2]]) == 1 else 1)
            v = [p + (q - p) * t for p, q, t in zip(a, b, x, strict=True)]
        elif ins.op == 0x32 and rand is not None:  # noqa: PLR2004
            v = [rand(vals[ops[0]][0])]
        elif ins.op in _BINARY:
            a, b = _bcast(vals[ops[0]], vals[ops[1]])
            v = [_BINARY[ins.op](p, q) for p, q in zip(a, b, strict=True)]
        elif ins.op in _UNARY:
            v = [_UNARY[ins.op](p) for p in vals[ops[0]]]
        else:
            msg = f"cannot evaluate {ins.name}"
            raise FunctionError(msg)
        vals.append(v)
    return vals[-1]


# Interval bounds (for culling): each component's (low, high), or infinite.

Interval = tuple[float, float]
_ALL = (-math.inf, math.inf)


def _imul(a: Interval, b: Interval) -> Interval:
    products = [
        x * y if not (x == 0 or y == 0) else 0.0
        for x in a
        for y in b
    ]  # fmt: skip
    return min(products), max(products)


def _iunary(op: int, a: Interval) -> Interval:  # noqa: PLR0911
    lo, hi = a
    if op == 0x17:  # noqa: PLR2004
        return -hi, -lo
    if op == 0x23:  # noqa: PLR2004
        if lo >= 0:
            return a
        return (0.0, max(-lo, hi)) if hi >= 0 else (-hi, -lo)
    if op == 0x24:  # noqa: PLR2004
        return math.floor(lo) if lo > -math.inf else lo, math.floor(
            hi
        ) if hi < math.inf else hi
    if op == 0x25:  # noqa: PLR2004
        return math.ceil(lo) if lo > -math.inf else lo, math.ceil(
            hi
        ) if hi < math.inf else hi
    if op in (0x26, 0x27):
        return -1.0, 1.0
    if op == 0x28:  # noqa: PLR2004
        return math.sqrt(max(lo, 0.0)), math.sqrt(max(hi, 0.0))
    if op == 0x1C:  # noqa: PLR2004
        return 0.0, 1.0
    if op == 0x2A:  # noqa: PLR2004
        return math.exp(min(lo, 700)), math.exp(min(hi, 700))
    if op == 0x2B:  # noqa: PLR2004
        return 2.0 ** min(lo, 1000), 2.0 ** min(hi, 1000)
    return _ALL


def bounds(
    block: Block,
    inputs: Callable[[int, int], list[Interval]],
    builtins: dict[int, list[Interval]],
) -> list[Interval]:
    """Return bounds on the block's value, each component's (low, high).

    ``inputs(uid, width)`` and ``builtins`` bound the graph inputs and built-ins;
    rand(a) is within [0, a]. Ops without a simple bound give (-inf, inf).
    """
    vals: list[list[Interval]] = []
    var: dict[int, list[Interval]] = {}
    for ins in block.instrs:
        ops = ins.operands

        def arg(k: int, ops: list[int] = ops) -> list[Interval]:
            return vals[ops[k]]

        if ins.op == 0x00:
            v = [(float(x), float(x)) for x in constant(ins)]
        elif ins.op == 0x01:
            v = builtins.get(ops[0], [_ALL] * ins.width)
        elif ins.op == 0x02:  # noqa: PLR2004
            v = inputs(ins.u32s()[0], ins.width)
        elif ins.op == 0x04:  # noqa: PLR2004
            v = var.get(ops[0], [_ALL] * ins.width)
        elif ins.op == 0x07:  # noqa: PLR2004
            var[ops[1]] = v = arg(0)
        elif ins.op == 0x09:  # noqa: PLR2004
            a, b = _bcast(arg(1), arg(2))
            v = [(min(p[0], q[0]), max(p[1], q[1])) for p, q in zip(a, b, strict=True)]
        elif ins.op == 0x0C:  # noqa: PLR2004
            v = [(0.0, 0.0)] * ins.width if ops[1] == NO_VALUE else arg(1)
        elif ins.op in (0x0D, 0x0E, 0x0F):
            v = [x for o in ops for x in vals[o]]
        elif ins.op == 0x10:  # noqa: PLR2004
            src, sel = arg(0), ops[1]
            v = [src[sel >> (2 * k) & 3] for k in range(ins.width)]
        elif ins.op == 0x11:  # noqa: PLR2004
            v = [_iunary(0x24, x) if ins.type != FLOAT else x for x in arg(0)]
            if ins.type != FLOAT:  # truncation toward zero: widen by one
                v = [(lo - 1, hi + 1) for lo, hi in v]
        elif ins.op in (0x12, 0x13, 0x14, 0x30, 0x31):
            a, b = _bcast(arg(0), arg(1))
            v = []
            for p, q in zip(a, b, strict=True):
                if ins.op == 0x12:  # noqa: PLR2004
                    v.append((p[0] + q[0], p[1] + q[1]))
                elif ins.op == 0x13:  # noqa: PLR2004
                    v.append((p[0] - q[1], p[1] - q[0]))
                elif ins.op == 0x14:  # noqa: PLR2004
                    v.append(_imul(p, q))
                elif ins.op == 0x30:  # noqa: PLR2004
                    v.append((min(p[0], q[0]), min(p[1], q[1])))
                else:
                    v.append((max(p[0], q[0]), max(p[1], q[1])))
        elif ins.op == 0x15:  # noqa: PLR2004
            a, b = _bcast(arg(0), arg(1))
            v = [
                _imul(p, (1 / q[1], 1 / q[0])) if q[0] > 0 or q[1] < 0 else _ALL
                for p, q in zip(a, b, strict=True)
            ]
        elif ins.op == 0x16:  # noqa: PLR2004
            a, b = _bcast(arg(0), arg(1))
            v = [
                (min(0.0, q[0]), max(0.0, q[1])) if q[0] > 0 or q[1] < 0 else _ALL
                for _, q in zip(a, b, strict=True)
            ]
        elif ins.op in (0x1A, 0x1B, 0x1C) or 0x1D <= ins.op <= 0x22:  # noqa: PLR2004
            v = [(0.0, 1.0)] * ins.width
        elif ins.op == 0x18:  # noqa: PLR2004
            terms = [_imul(p, q) for p, q in zip(arg(0), arg(1), strict=True)]
            v = [(sum(t[0] for t in terms), sum(t[1] for t in terms))]
        elif ins.op == 0x2D:  # noqa: PLR2004
            v = [(-math.pi, math.pi)]
        elif ins.op == 0x2E:  # noqa: PLR2004
            r = max(abs(arg(0)[0][0]), abs(arg(0)[0][1]))
            v = [(-r, r), (-r, r)]
        elif ins.op == 0x2F:  # noqa: PLR2004
            a, b = _bcast(arg(0), arg(1))
            t = arg(2) * (len(a) if len(arg(2)) == 1 else 1)
            v = []
            for p, q, x in zip(a, b, t, strict=True):
                d = (q[0] - p[1], q[1] - p[0])
                m = _imul(d, x)
                v.append((p[0] + m[0], p[1] + m[1]))
        elif ins.op == 0x32:  # noqa: PLR2004
            lo, hi = arg(0)[0]
            v = [(min(0.0, lo), max(0.0, hi))]
        elif ins.op in (0x33, 0x34):
            v = [(0.0, 1.0)] * ins.width
        elif ins.op in _UNARY:
            v = [_iunary(ins.op, x) for x in arg(0)]
        else:
            v = [_ALL] * ins.width
        # inf - inf and the like: no bound.
        vals.append(
            [
                (
                    -math.inf if math.isnan(lo) else lo,
                    math.inf if math.isnan(hi) else hi,
                )
                for lo, hi in v
            ]
        )
    return vals[-1]


# GLSL translation.


def glsl_type(kind: int, width: int) -> str:
    """Return the GLSL type of a value."""
    if kind == BOOL:
        return "bool"
    if kind == INT:
        return "int" if width == 1 else f"ivec{width}"
    return "float" if width == 1 else f"vec{width}"


def _lit(kind: int, vals: list) -> str:
    if kind == BOOL:
        return "true" if vals[0] else "false"
    if kind == INT:
        inner = ", ".join(str(int(v)) for v in vals)
        return inner if len(vals) == 1 else f"ivec{len(vals)}({inner})"
    inner = ", ".join(_flt(v) for v in vals)
    return inner if len(vals) == 1 else f"vec{len(vals)}({inner})"


def _flt(v: float) -> str:
    if math.isinf(v):
        return "1e30" if v > 0 else "-1e30"
    for digits in range(1, 10):
        text = f"{v:.{digits}g}"
        if struct.pack("<f", float(text)) == struct.pack("<f", v):
            break
    else:
        text = repr(v)
    return text if any(c in text for c in ".en") else f"{text}.0"


_GLSL_BINARY = {
    0x12: "({a} + {b})", 0x13: "({a} - {b})", 0x14: "({a} * {b})",
    0x15: "({a} / {b})", 0x16: "mod({a}, {b})", 0x1A: "({a} && {b})",
    0x1B: "({a} || {b})", 0x1D: "({a} == {b})", 0x1E: "({a} != {b})",
    0x1F: "({a} > {b})", 0x20: "({a} >= {b})", 0x21: "({a} < {b})",
    0x22: "({a} <= {b})", 0x30: "min({a}, {b})", 0x31: "max({a}, {b})",
    0x36: "pow({a}, {b})",
}  # fmt: skip
_GLSL_UNARY = {
    0x17: "(-{a})", 0x1C: "(!{a})", 0x23: "abs({a})", 0x24: "floor({a})",
    0x25: "ceil({a})", 0x26: "cos({a})", 0x27: "sin({a})", 0x28: "sqrt({a})",
    0x29: "log({a})", 0x2A: "exp({a})", 0x2B: "exp2({a})", 0x2C: "tan({a})",
    0x35: "log2({a})",
}  # fmt: skip


def to_glsl(
    block: Block,
    prefix: str,
    inputs: Callable[[int, int, int], str],
    builtins: Callable[[int, int], str],
    samples: Callable[[str, int, str, int], str] | None = None,
    rand: Callable[[str, int], str] | None = None,
    variables: Mapping[int, tuple[str, str]] | None = None,
) -> tuple[list[str], str]:
    """Return GLSL statements computing ``block`` and the name of its value.

    Values are named ``{prefix}_{k}``; ``inputs(uid, type, width)`` and
    ``builtins(id, width)`` give GLSL expressions for graph inputs and
    built-ins (a Material Maker parameter such as ``$p_amount``, a
    constant...); ``samples(kind, input index, position, filtering)``, kind
    "lum" or "col", gives a pixel processor's input sampled; ``rand(a,
    k)`` a random value in [0, a) for instruction k. ``variables`` maps
    variables shared beyond the block (an FX-Map's) to a GLSL global and
    its type, which the block reads and writes instead of locals.
    """
    lines: list[str] = []
    names: list[str] = []
    kinds: list[tuple[int, int]] = []
    declared: dict[int, str] = {}
    """Variables declared so far, with their GLSL types."""
    for k, ins in enumerate(block.instrs):
        ops = ins.operands
        name = f"{prefix}_{k}"
        t = glsl_type(ins.type, ins.width)

        def v(i: int) -> str:
            return names[i]

        if ins.op == 0x00:
            expr = _lit(ins.type, constant(ins))
        elif ins.op == 0x01:
            expr = builtins(ops[0], ins.width)
        elif ins.op == 0x02:  # noqa: PLR2004
            expr = inputs(ins.u32s()[0], ins.type, ins.width)
            if ins.type == BOOL:  # a bool input is a number parameter
                expr = f"bool({expr})"
        elif ins.op == 0x04 and variables and ops[0] in variables:  # noqa: PLR2004
            glob, kind = variables[ops[0]]
            expr = glob if kind == t else f"{t}({glob})"
        elif ins.op == 0x07 and variables and ops[1] in variables:  # noqa: PLR2004
            glob, kind = variables[ops[1]]
            lines.append(f"{glob} = {kind}({v(ops[0])});")
            expr = v(ops[0])
        elif ins.op == 0x04:  # noqa: PLR2004
            # A variable this block never set (one an FX-Map node's
            # ancestors set, it seems) reads as zero, as in evaluate.
            if ops[0] not in declared:
                declared[ops[0]] = t
                lines.append(f"{t} {prefix}_var{ops[0]} = {t}(0);")
            expr = f"{prefix}_var{ops[0]}"
            if declared[ops[0]] != t:
                expr = f"{t}({expr})"
        elif ins.op == 0x07:  # noqa: PLR2004
            if ops[1] in declared:
                lines.append(f"{prefix}_var{ops[1]} = {declared[ops[1]]}({v(ops[0])});")
            else:
                declared[ops[1]] = t
                lines.append(f"{t} {prefix}_var{ops[1]} = {v(ops[0])};")
            expr = v(ops[0])
        elif ins.op == 0x09:  # noqa: PLR2004
            expr = f"({v(ops[0])} ? {v(ops[1])} : {v(ops[2])})"
        elif ins.op == 0x0C:  # noqa: PLR2004
            expr = f"{t}(0)" if ops[1] == NO_VALUE else v(ops[1])
        elif ins.op in (0x0D, 0x0E, 0x0F):
            expr = f"{t}({', '.join(v(o) for o in ops)})"
        elif ins.op == 0x10:  # noqa: PLR2004
            sel = "".join("xyzw"[ops[1] >> (2 * c) & 3] for c in range(ins.width))
            expr = f"{v(ops[0])}.{sel}"
        elif ins.op == 0x11:  # noqa: PLR2004
            expr = f"{t}({v(ops[0])})"
        elif ins.op == 0x18:  # noqa: PLR2004
            expr = f"dot({v(ops[0])}, {v(ops[1])})"
        elif ins.op == 0x2D:  # noqa: PLR2004
            expr = f"atan({v(ops[0])}.y, {v(ops[0])}.x)"
        elif ins.op == 0x2E:  # noqa: PLR2004
            r, th = v(ops[0]), v(ops[1])
            expr = f"vec2({r} * cos({th}), {r} * sin({th}))"
        elif ins.op == 0x2F:  # noqa: PLR2004
            expr = f"mix({v(ops[0])}, {v(ops[1])}, {v(ops[2])})"
        elif ins.op == 0x32 and rand is not None:  # noqa: PLR2004
            expr = rand(v(ops[0]), k)
        elif ins.op in (0x33, 0x34) and samples is not None:
            kind = "lum" if ins.op == 0x33 else "col"  # noqa: PLR2004
            filtering = ops[2] if len(ops) > 2 else 0  # noqa: PLR2004
            expr = samples(kind, ops[1], v(ops[0]), filtering)
        elif ins.op == 0x16 and ins.type == INT:  # noqa: PLR2004 (GLSL's mod is float)
            expr = f"({v(ops[0])} % {v(ops[1])})"
        elif ins.op in _GLSL_BINARY:
            expr = _GLSL_BINARY[ins.op].format(a=v(ops[0]), b=v(ops[1]))
        elif ins.op in _GLSL_UNARY:
            expr = _GLSL_UNARY[ins.op].format(a=v(ops[0]))
        else:
            msg = f"cannot translate {ins.name}"
            raise FunctionError(msg)
        lines.append(f"{t} {name} = {expr};")
        names.append(name)
        kinds.append((ins.type, ins.width))
    return lines, names[-1]
