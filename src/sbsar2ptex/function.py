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
"""

import math
import struct
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import Any

FLOAT, INT, BOOL = 1, 2, 0

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
    0x32: "rand", 0x35: "log2", 0x36: "pow",
}  # fmt: skip
BUILTINS = {1: "$size", 3: "$sizelog2"}
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
    block: Block, inputs: dict[int, list[float]], builtins: dict[int, list[float]]
) -> list:
    """Return the block's value with graph ``inputs`` (by uid) and ``builtins``."""
    vals: list[list] = []
    var: dict[int, list] = {}
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
            v = vals[ops[1]]
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
) -> tuple[list[str], str]:
    """Return GLSL statements computing ``block`` and the name of its value.

    Values are named ``{prefix}_{k}``; ``inputs(uid, type, width)`` and
    ``builtins(id, width)`` give GLSL expressions for graph inputs and
    built-ins (a Material Maker parameter such as ``$p_amount``, a
    constant...).
    """
    lines: list[str] = []
    names: list[str] = []
    kinds: list[tuple[int, int]] = []
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
        elif ins.op == 0x04:  # noqa: PLR2004
            expr = f"{prefix}_var{ops[0]}"
        elif ins.op == 0x07:  # noqa: PLR2004
            lines.append(f"{t} {prefix}_var{ops[1]} = {v(ops[0])};")
            expr = v(ops[0])
        elif ins.op == 0x09:  # noqa: PLR2004
            expr = f"({v(ops[0])} ? {v(ops[1])} : {v(ops[2])})"
        elif ins.op == 0x0C:  # noqa: PLR2004
            expr = v(ops[1])
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
