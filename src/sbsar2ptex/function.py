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
    0x0b while (init, cond, body, u32 cap: see Loop)
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

import numpy as np

FLOAT, INT, BOOL = 1, 2, 0
NO_VALUE = 0xFFFF
"""An operand naming no value: a sequence whose result the cooker dropped."""

OP_NAMES = {
    0x00: "const", 0x01: "builtin", 0x02: "input", 0x04: "getvar", 0x07: "setvar",
    0x09: "ifelse", 0x0B: "while", 0x0C: "sequence", 0x0D: "concat", 0x0E: "concat3",
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
        # A bool constant takes one u16 a component, no padding; a graph
        # input, bool or not, is read by its u32 uid.
        if n % 2 and (self.op == 0x02 or (self.op == 0x00 and self.type != BOOL)):  # noqa: PLR2004
            return self.operands[1:]
        return self.operands

    def u32s(self) -> list[int]:
        """Return the (aligned) operands as u32s."""
        ops = self.aligned()
        return [ops[k] | ops[k + 1] << 16 for k in range(0, len(ops) - 1, 2)]


@dataclass
class Block:
    """A decoded code block: instructions; its value is the last one's."""

    instrs: list[Instr] = field(default_factory=list)
    _loops: "dict[int, Loop] | None" = field(
        default=None, init=False, repr=False, compare=False
    )
    _steps: list | None = field(default=None, init=False, repr=False, compare=False)

    def uids(self) -> set[int]:
        """Return the graph inputs (uids) the block reads."""
        return {i.u32s()[0] for i in self.instrs if i.op == 0x02}  # noqa: PLR2004

    def loops(self) -> "dict[int, Loop]":
        """Return ``loops(self)``, found once (the instructions as they are then)."""
        if self._loops is None:
            self._loops = loops(self)
        return self._loops

    def steps(self) -> list:
        """Return the instructions compiled (_compile), once."""
        if self._steps is None:
            self._steps = [_compile(ins) for ins in self.instrs]
        return self._steps


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


WHILE = 0x0B
MAX_ITERATIONS = 1 << 16
"""Iterations after which the evaluator gives up on a while loop (where
Substance, without a cap, would hang): its value is unknown (FunctionError)."""
GLSL_INT_MAX = (1 << 31) - 1


@dataclass(frozen=True)
class Loop:
    """A while loop (op 0x0B, docs/function-ops.md section 1).

    Its instructions are three consecutive ranges just before it: the init
    range, ending at ``init`` (run once); the condition range, (init,
    cond]; the body range, (cond, body]; the loop instruction, ``at``,
    comes right after ``body``. A ``cap`` of 0 is no cap.
    """

    init: int
    cond: int
    body: int
    cap: int
    at: int

    @property
    def start(self) -> int:
        """The first instruction the loop repeats (its condition range's)."""
        return self.init + 1


def while_operands(ins: Instr) -> tuple[int, int, int, int]:
    """Return a while instruction's (init, cond, body, cap).

    Three u16 SSA indices, then the u32 cap, 4-byte aligned: five operand
    u16s, or six with a u16 of padding before the cap.
    """
    ops = ins.operands
    if len(ops) == 6:  # noqa: PLR2004
        ops = [*ops[:3], *ops[4:]]
    elif len(ops) != 5:  # noqa: PLR2004
        msg = f"while with {len(ops)} operand u16s"
        raise FunctionError(msg)
    return ops[0], ops[1], ops[2], ops[3] | ops[4] << 16


def loops(block: Block) -> dict[int, Loop]:
    """Return the block's while loops, by their start (``Loop.start``).

    Raises FunctionError for ranges that are not as section 1.2 has them,
    or loops that overlap without one inside a range of the other.
    """
    found: dict[int, Loop] = {}
    for k, ins in enumerate(block.instrs):
        if ins.op != WHILE:
            continue
        init, cond, body, cap = while_operands(ins)
        if not 0 <= init < cond < body == k - 1:
            msg = f"while #{k}: ranges ({init}, {cond}, {body}) not just before it"
            raise FunctionError(msg)
        if init + 1 in found:
            msg = f"while #{k}: two loops start at #{init + 1}"
            raise FunctionError(msg)
        found[init + 1] = Loop(init, cond, body, cap, k)
    for outer in found.values():
        for inner in found.values():
            if not outer.start < inner.start <= outer.body:
                continue
            end = outer.cond if inner.start <= outer.cond else outer.body
            if inner.at > end:
                msg = f"while #{inner.at} straddles a range of while #{outer.at}"
                raise FunctionError(msg)
    return found


def _enclosing(block: Block, found: dict[int, Loop]) -> tuple[list, list]:
    """Return, per instruction, its outermost and innermost enclosing loop.

    An instruction is inside a loop when it is in its condition or body
    range (None when it is in none).
    """
    n = len(block.instrs)
    outer: list[Loop | None] = [None] * n
    inner: list[Loop | None] = [None] * n
    for loop in sorted(found.values(), key=lambda lp: lp.start):
        for k in range(loop.start, loop.body + 1):
            if outer[k] is None:
                outer[k] = loop
            inner[k] = loop  # later starts are nested deeper
    return outer, inner


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
# of components (one for scalars). Every step is rounded to its type, as
# Substance evaluates functions: floats to float32, ints wrapping at 32
# bits. A Tile Sampler's hash of large products (1234 / (n + 4321) *
# 3.435974e10, mod 1e6...) lands elsewhere in float64. A double operation
# rounded once is the float32 one for + - * / and sqrt; mod rounds each of
# its steps.


def f32(x: float) -> float:
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


def _mod(a: float, b: float, kind: int) -> float:
    """Return a - b * floor(a / b), each step rounded to float32 for floats."""
    if not b:
        return 0.0
    if kind != FLOAT:
        return a - b * math.floor(a / b)
    return f32(a - f32(b * math.floor(f32(a / b))))


_TURN = f32(1 / (2 * math.pi))
_QUARTER = f32(math.pi / 2)
_SIN = (f32(2 * math.pi), f32(41.3417015), f32(81.6052475), f32(74.1371841))
"""sin(2 pi t) ~ t (c1 - t^2 (c3 - t^2 (c5 - t^2 c7))) on [-1/4, 1/4]: the
cooker's function evaluator's polynomial (its c7 makes sin(pi / 2) 1 to
float32's precision; Taylor's 76.7 would leave it 1.6e-4 low). Its error
peaks at 1.4e-5 by t = 0.22; an FX-Map's splats are where its cartesian puts
them (Biomech's n2369: 92.4% of pixels
exact at 16 bits with math.sin, 98.5% with this)."""


def engine_sin(x: float) -> float:
    """Return sin(x) as the cooker's function evaluator does, in float32."""
    if not math.isfinite(x):
        return math.nan
    t = f32(x * _TURN)
    t = f32(t - round(t))  # [-1/2, 1/2], halves to even as rint
    if t > 0.25:  # noqa: PLR2004 - sin(pi - a) = sin(a)
        t = f32(0.5 - t)
    elif t < -0.25:  # noqa: PLR2004
        t = f32(-0.5 - t)
    c1, c3, c5, c7 = _SIN
    t2 = f32(t * t)
    p = f32(c5 - f32(c7 * t2))
    p = f32(f32(p * t2) - c3)
    p = f32(f32(p * t2) + c1)
    return f32(p * t)


def engine_cos(x: float) -> float:
    """Return cos(x) as the cooker's function evaluator does: sin(x + pi / 2)."""
    return engine_sin(f32(f32(x) + _QUARTER))


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
    0x26: engine_cos,
    0x27: engine_sin,
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
        return [f32(float(x)) for x in values]
    if kind == INT:
        return [_wrap(x) if math.isfinite(x) else 0 for x in values]
    return [bool(x) for x in values]


def _dropped(ins: Instr, first: list) -> list:
    """Return a sequence's value when the cooker dropped its result.

    Its first part's, as the sequence's type: a float to the nearest int.
    A pixel processor's output size relative to its input's, adding
    sequence(set v = (rand, rand), (0, 0)), comes out as the size plus v
    rounded (check_filters' pp_param_var: 4096 at seed 0, 4096 x 2048 at
    3). Zero where the widths differ.
    """
    if len(first) != ins.width:
        return [0.0] * ins.width
    if ins.type == INT:
        return [round(x) if math.isfinite(x) else 0 for x in first]
    return first


class _Scalars:
    """evaluate's arithmetic: a value's components are Python numbers."""

    f32 = staticmethod(f32)
    sin = staticmethod(engine_sin)
    cos = staticmethod(engine_cos)
    div = staticmethod(_div)
    mod = staticmethod(_mod)
    as_kind = staticmethod(_as)
    dropped = staticmethod(_dropped)

    @staticmethod
    def lift(values: list) -> list:
        return values

    @staticmethod
    def binary(op: int, kind: int) -> Callable[[Any, Any], Any]:  # noqa: ARG004
        return _BINARY[op]

    @staticmethod
    def unary(op: int, kind: int) -> Callable[[Any], Any]:  # noqa: ARG004
        return _UNARY[op]

    @staticmethod
    def atan2(y: Any, x: Any, kind: int) -> Any:  # noqa: ARG004
        return math.atan2(y, x)

    @staticmethod
    def select(c: Any, a: list, b: list) -> list:
        return a if c else b

    @staticmethod
    def truth(c: Any) -> bool:
        return bool(c)

    @staticmethod
    def unconverted(ins: Instr) -> None:
        """Leave an instruction of no known type as computed."""


_SCALARS = _Scalars()


def evaluate(
    block: Block,
    inputs: dict[int, list],
    builtins: dict[int, list[float]] | None = None,
    rand: Callable[[float], float] | None = None,
    variables: dict[int, list] | None = None,
) -> list:
    """Return the block's value with graph ``inputs`` (by uid) and ``builtins``.

    ``rand(a)``, when given, is the generator rand instructions call, in
    instruction order. ``variables``, when given, is the variable store,
    read and written in place: blocks that share one (an FX-Map's) see each
    other's variables. Samples, and what ``builtins`` and ``rand`` leave
    out, raise FunctionError, as does a while loop still running after
    MAX_ITERATIONS.

    Instructions run in the order the block's while loops run them (each
    loop's condition and body ranges again per iteration), so each value is
    stored by its index and a read sees its most recent run.
    """
    var: dict[int, list] = {} if variables is None else variables
    return _run(block, inputs, builtins, rand, var, _SCALARS)


class _Env:
    """What a block's instructions read besides earlier values: _run's arguments."""

    __slots__ = ("builtins", "inputs", "num", "rand", "var")

    def __init__(
        self,
        inputs: Mapping[int, list],
        builtins: Mapping[int, list] | None,
        rand: Callable[[Any], Any] | None,
        var: dict[int, list],
        num: Any,
    ) -> None:
        self.inputs, self.builtins, self.rand, self.var, self.num = (
            inputs, builtins, rand, var, num,
        )  # fmt: skip


Step = Callable[[list, _Env], list]
"""One instruction, compiled: its value from the values so far and the _Env."""


def _run(
    block: Block,
    inputs: Mapping[int, list],
    builtins: Mapping[int, list] | None,
    rand: Callable[[Any], Any] | None,
    var: dict[int, list],
    num: Any,
) -> list:
    """Run ``block`` with ``num``'s arithmetic (_Scalars, or Lanes)."""
    instrs = block.instrs
    steps = block.steps()
    vals: list[list] = [[] for _ in instrs]
    starts = block.loops()
    env = _Env(inputs, builtins, rand, var, num)

    def run(lo: int, hi: int, own: Loop | None = None) -> None:
        """Run instructions lo to hi, both included, and the loops among them."""
        k = lo
        while k <= hi:
            loop = starts.get(k)
            if loop is not None and loop is not own:
                if loop.at > hi:
                    msg = f"while #{loop.at} ends past its range"
                    raise FunctionError(msg)
                run_loop(loop)
                k = loop.at + 1
            else:
                vals[k] = steps[k](vals, env)
                k += 1

    def run_loop(loop: Loop) -> None:
        """Run ``loop`` (its init range has run): section 1.4."""
        n = 0
        while not (loop.cap and n >= loop.cap):
            if n >= MAX_ITERATIONS:
                msg = f"while #{loop.at} still running after {n} iterations"
                raise FunctionError(msg)
            run(loop.start, loop.cond, loop)
            if num.truth(vals[loop.cond][0]):
                break
            run(loop.cond + 1, loop.body)
            n += 1
        ins = instrs[loop.at]
        zero = num.as_kind(ins.type, num.lift([0] * ins.width))
        vals[loop.at] = vals[loop.body] if n else zero

    if starts:
        run(0, len(instrs) - 1)
    else:  # (straight code: each instruction once, in order)
        for k, step in enumerate(steps):
            vals[k] = step(vals, env)
    return vals[-1]


def _compile(ins: Instr) -> Step:
    """Return one instruction as a Step: its op chosen once, not each run.

    Its value is converted to its type (floats rounded to float32, ints
    wrapped...) but for set, if/else and sequence, which pass a value on.
    """
    body = _body(ins)
    kind = ins.type
    if ins.op in (0x07, 0x09, 0x0C):
        return body
    if kind in (FLOAT, INT, BOOL):

        def converted(vals: list, env: _Env) -> list:
            return env.num.as_kind(kind, body(vals, env))

        return converted

    def unconverted(vals: list, env: _Env) -> list:
        v = body(vals, env)
        env.num.unconverted(ins)
        return v

    return unconverted


def _body(ins: Instr) -> Step:  # noqa: PLR0911
    """Return an instruction's Step before its conversion."""
    ops, op, kind, width = ins.operands, ins.op, ins.type, ins.width

    def cannot(vals: list, env: _Env) -> list:
        msg = f"cannot evaluate {ins.name}"
        raise FunctionError(msg)

    if op == 0x00:
        known: list = []

        def const(vals: list, env: _Env) -> list:
            if not known:
                known.append(constant(ins))
            return env.num.lift(known[0])

        return const
    if op == 0x01:

        def builtin(vals: list, env: _Env) -> list:
            if env.builtins is None:
                return cannot(vals, env)
            return env.num.lift(env.builtins[ops[0]])

        return builtin
    if op == 0x02:  # noqa: PLR2004
        uids = ins.u32s()
        return lambda _vals, env: env.num.lift(env.inputs[uids[0]])
    if op == 0x04:  # noqa: PLR2004

        def getvar(vals: list, env: _Env) -> list:
            v = env.var.get(ops[0])
            return env.num.lift([0.0] * width) if v is None else v

        return getvar
    if op == 0x07:  # noqa: PLR2004

        def setvar(vals: list, env: _Env) -> list:
            env.var[ops[1]] = v = vals[ops[0]]
            return v

        return setvar
    if op == 0x09:  # noqa: PLR2004
        return lambda vals, env: env.num.select(
            vals[ops[0]][0], vals[ops[1]], vals[ops[2]]
        )
    if op == 0x0C:  # noqa: PLR2004

        def sequence(vals: list, env: _Env) -> list:
            if ops[1] != NO_VALUE:
                return vals[ops[1]]
            return env.num.dropped(ins, vals[ops[0]])

        return sequence
    if op in (0x0D, 0x0E, 0x0F):
        return lambda vals, _env: [x for o in ops for x in vals[o]]
    if op == 0x10:  # noqa: PLR2004

        def swizzle(vals: list, env: _Env) -> list:
            src, sel = vals[ops[0]], ops[1]
            return [src[sel >> (2 * k) & 3] for k in range(width)]

        return swizzle
    if op == 0x11:  # noqa: PLR2004 (converted by _compile, by its type)
        return lambda vals, _env: vals[ops[0]]
    if op == 0x18:  # noqa: PLR2004

        def dot(vals: list, env: _Env) -> list:
            f = env.num.f32
            acc = 0.0
            for x, y in zip(vals[ops[0]], vals[ops[1]], strict=True):
                acc = f(acc + f(x * y))
            return [acc]

        return dot
    if op == 0x2D:  # noqa: PLR2004

        def atan2(vals: list, env: _Env) -> list:
            x, y = vals[ops[0]][:2]
            return [env.num.atan2(y, x, kind)]

        return atan2
    if op == 0x2E:  # noqa: PLR2004

        def cartesian(vals: list, env: _Env) -> list:
            rho, theta = vals[ops[0]][0], vals[ops[1]][0]
            num = env.num
            return [num.f32(rho * num.cos(theta)), num.f32(rho * num.sin(theta))]

        return cartesian
    if op == 0x2F:  # noqa: PLR2004

        def lerp(vals: list, env: _Env) -> list:
            a, b = _spread(vals[ops[0]], vals[ops[1]])
            t = vals[ops[2]] * (len(a) if len(vals[ops[2]]) == 1 else 1)
            f = env.num.f32
            return [f(p + f(f(q - p) * s)) for p, q, s in zip(a, b, t, strict=True)]

        return lerp
    if op == 0x32:  # noqa: PLR2004

        def rand(vals: list, env: _Env) -> list:
            if env.rand is None:
                return cannot(vals, env)
            return [env.rand(vals[ops[0]][0])]

        return rand
    if op in (0x15, 0x16):

        def divide(vals: list, env: _Env) -> list:
            a, b = _spread(vals[ops[0]], vals[ops[1]])
            fn = env.num.div if op == 0x15 else env.num.mod  # noqa: PLR2004
            return [fn(p, q, kind) for p, q in zip(a, b, strict=True)]

        return divide
    if op in _BINARY:

        def binary(vals: list, env: _Env) -> list:
            a, b = _spread(vals[ops[0]], vals[ops[1]])
            fn = env.num.binary(op, kind)
            return [fn(p, q) for p, q in zip(a, b, strict=True)]

        return binary
    if op in _UNARY:

        def unary(vals: list, env: _Env) -> list:
            fn = env.num.unary(op, kind)
            return [fn(p) for p in vals[ops[0]]]

        return unary
    return cannot


# Lanes: one block evaluated for many inputs at once (an FX-Map's tree
# nodes, all of a level), each component a numpy array of one value a lane.
# The arithmetic is evaluate's, bit for bit: floats are float64 arrays
# holding what evaluate's Python floats hold, rounded to float32 where it
# rounds; ints (and bools, as 0 or 1: Python's bools add and compare as
# ints) are int64 arrays. Ops numpy has no exact twin of (exp, log, pow,
# atan2...) run evaluate's own function a lane at a time.


class Diverged(Exception):  # noqa: N818
    """Lanes that evaluate cannot follow together: run them one by one."""


_INT_SAFE = 1 << 31
"""Ints past this (a dropped sequence's rounding) could overflow int64 math."""


def _is_int(x: np.ndarray) -> bool:
    return x.dtype.kind in "iub"


class Lanes:
    """evaluate's arithmetic over ``n`` lanes: components are numpy arrays."""

    def __init__(self, n: int) -> None:
        """Lanes ``n`` wide."""
        self.n = n

    def lift(self, values: list) -> list[np.ndarray]:
        """Return components as lane arrays (numbers repeated)."""
        out = []
        for x in values:
            if isinstance(x, np.ndarray):
                out.append(x.astype(np.int64) if x.dtype.kind == "b" else x)
            else:
                kind = np.int64 if isinstance(x, int) else np.float64
                out.append(np.full(self.n, x, dtype=kind))
        return out

    @staticmethod
    def f32(x: Any) -> np.ndarray:
        """Return ``x`` rounded to float32, as float64s (f32's overflow: inf)."""
        return np.asarray(x, dtype=np.float64).astype(np.float32).astype(np.float64)

    def lanewise(self, kind: int, fn: Callable, *args: np.ndarray) -> np.ndarray:
        """Return evaluate's ``fn`` of each lane, converted to ``kind``."""
        lanes = zip(*(a.tolist() for a in args), strict=True)
        out = [_as(kind, [fn(*xs)])[0] for xs in lanes]
        return np.array(out, dtype=np.float64 if kind == FLOAT else np.int64)

    def sin(self, x: np.ndarray) -> np.ndarray:
        """Return engine_sin of each lane."""
        x = np.asarray(x, dtype=np.float64)
        f = self.f32
        t = f(x * _TURN)
        ok = np.isfinite(x)
        if (ok & ~np.isfinite(t)).any():  # round(inf) raises: as evaluate
            return self.lanewise(FLOAT, engine_sin, x)
        t = f(t - (np.rint(t) + 0.0))  # (round's int: no -0.0)
        t = np.where(t > 0.25, f(0.5 - t), np.where(t < -0.25, f(-0.5 - t), t))  # noqa: PLR2004
        c1, c3, c5, c7 = _SIN
        t2 = f(t * t)
        p = f(c5 - f(c7 * t2))
        p = f(f(p * t2) - c3)
        p = f(f(p * t2) + c1)
        return np.where(ok, f(p * t), np.nan)

    def cos(self, x: np.ndarray) -> np.ndarray:
        """Return engine_cos of each lane."""
        return self.sin(self.f32(self.f32(x) + _QUARTER))

    def div(self, a: np.ndarray, b: np.ndarray, kind: int) -> np.ndarray:
        """Return _div of each lane."""
        nz = b != 0
        q = a / np.where(nz, b, 1)
        if kind == INT:
            if not np.isfinite(q[nz]).all():  # int(inf) raises: as evaluate
                return self.lanewise(kind, lambda p, r: _div(p, r, kind), a, b)
            return np.where(nz, np.trunc(q), 0.0)
        zero = np.where(a != 0, np.copysign(np.inf, a), np.nan)
        return np.where(nz, q, zero)

    def mod(self, a: np.ndarray, b: np.ndarray, kind: int) -> np.ndarray:
        """Return _mod of each lane."""
        ints = _is_int(a), _is_int(b)
        if ints[0] != ints[1] or (kind == FLOAT and ints[1]):
            # exact int products of a float's floor: evaluate's own
            return self.lanewise(kind, lambda p, r: _mod(p, r, kind), a, b)
        zero = b == 0
        bs = np.where(zero, 1, b)
        q = a / bs if kind != FLOAT else self.f32(a / bs)
        if not np.isfinite(q[~zero]).all():  # math.floor(inf) raises
            return self.lanewise(kind, lambda p, r: _mod(p, r, kind), a, b)
        fl = np.floor(q) + 0.0  # (math.floor's int: no -0.0)
        if kind != FLOAT:
            r = a - bs * (fl.astype(np.int64) if ints[1] else fl)
        else:
            r = self.f32(a - self.f32(bs * fl))
        return np.where(zero, 0.0, r)

    def binary(self, op: int, kind: int) -> Callable[[Any, Any], np.ndarray]:
        """Return _BINARY[op] over lanes."""
        return lambda a, b: self.binary_lanes(op, a, b, kind)

    def unary(self, op: int, kind: int) -> Callable[[Any], np.ndarray]:
        """Return _UNARY[op] over lanes."""
        return lambda a: self.unary_lanes(op, a, kind)

    def binary_lanes(  # noqa: PLR0911
        self, op: int, a: np.ndarray, b: np.ndarray, kind: int
    ) -> np.ndarray:
        """Return _BINARY[op] of each lane."""
        if op == 0x12:  # noqa: PLR2004
            return a + b
        if op == 0x13:  # noqa: PLR2004
            return a - b
        if op == 0x14:  # noqa: PLR2004
            return a * b
        if op == 0x1A:  # noqa: PLR2004
            return ((a != 0) & (b != 0)).astype(np.int64)
        if op == 0x1B:  # noqa: PLR2004
            return ((a != 0) | (b != 0)).astype(np.int64)
        compare = _COMPARE.get(op)
        if compare is not None:
            return compare(a, b).astype(np.int64)
        if op == 0x30:  # noqa: PLR2004 - min(a, b): b only when below
            return np.where(b < a, b, a)
        if op == 0x31:  # noqa: PLR2004
            return np.where(b > a, b, a)
        return self.lanewise(kind, _BINARY[op], a, b)

    def unary_lanes(self, op: int, a: np.ndarray, kind: int) -> np.ndarray:  # noqa: PLR0911
        """Return _UNARY[op] of each lane."""
        if op == 0x17:  # noqa: PLR2004
            return -a
        if op == 0x1C:  # noqa: PLR2004
            return (a == 0).astype(np.int64)
        if op == 0x23:  # noqa: PLR2004
            return np.abs(a)
        if op in (0x24, 0x25):  # math's floor and ceil: nan past the finite
            if _is_int(a):
                return a
            r = np.floor(a) if op == 0x24 else np.ceil(a)  # noqa: PLR2004
            # (math's are ints: no -0.0)
            return np.where(np.isfinite(a), r + 0.0, np.nan)
        if op == 0x26:  # noqa: PLR2004
            return self.cos(a)
        if op == 0x27:  # noqa: PLR2004
            return self.sin(a)
        if op == 0x28:  # noqa: PLR2004
            ok = a >= 0
            return np.where(ok, np.sqrt(np.where(ok, a, 0).astype(np.float64)), np.nan)
        return self.lanewise(kind, _UNARY[op], a)

    def atan2(self, y: np.ndarray, x: np.ndarray, kind: int) -> np.ndarray:
        """Return math.atan2 of each lane."""
        return self.lanewise(kind, math.atan2, y, x)

    def as_kind(self, kind: int, values: list) -> list[np.ndarray]:
        """Return _as of each lane: floats to float32, ints wrapped, bools 0 or 1."""
        if kind == FLOAT:
            return [self.f32(x) for x in values]
        if kind == INT:
            return [self.wrap(x) for x in values]
        return [(np.asarray(x) != 0).astype(np.int64) for x in values]

    @staticmethod
    def wrap(x: np.ndarray) -> np.ndarray:
        """Return _as(INT): truncated, wrapped to 32 bits; 0 past the finite."""
        if not _is_int(x):
            ok = np.isfinite(x)
            # fmod is exact: the truncation's low 32 bits, as int(x) & mask
            x = np.where(ok, np.fmod(np.trunc(np.where(ok, x, 0)), 2.0**32), 0)
            x = x.astype(np.int64)
        return ((x.astype(np.int64) + _INT_SAFE) & 0xFFFFFFFF) - _INT_SAFE

    @staticmethod
    def select(c: np.ndarray, a: list, b: list) -> list:
        """Return each lane's a where c, else b (of one width and kind)."""
        if len(a) != len(b) or any(
            _is_int(x) != _is_int(y) for x, y in zip(a, b, strict=True)
        ):
            raise Diverged
        pick = c != 0
        return [np.where(pick, x, y) for x, y in zip(a, b, strict=True)]

    @staticmethod
    def truth(c: np.ndarray) -> bool:
        """Return a loop condition all lanes agree on (else Diverged)."""
        t = c != 0
        if t.all():
            return True
        if not t.any():
            return False
        raise Diverged

    def dropped(self, ins: Instr, first: list) -> list:
        """Return _dropped of each lane."""
        if len(first) != ins.width:
            return self.lift([0.0] * ins.width)
        if ins.type != INT:
            return first
        out = []
        for x in first:
            if not _is_int(x):
                ok = np.isfinite(x)
                if (np.abs(x[ok]) >= _INT_SAFE).any():
                    raise Diverged
                x = np.where(ok, np.rint(np.where(ok, x, 0)), 0).astype(np.int64)  # noqa: PLW2901
            out.append(x)
        return out

    @staticmethod
    def unconverted(ins: Instr) -> None:  # noqa: ARG004
        """Values left as computed (no known type) may mix kinds lane by lane."""
        raise Diverged


_COMPARE: dict[int, Callable[[Any, Any], Any]] = {
    0x1D: np.equal, 0x1E: np.not_equal, 0x1F: np.greater,
    0x20: np.greater_equal, 0x21: np.less, 0x22: np.less_equal,
}  # fmt: skip


def evaluate_lanes(
    block: Block,
    inputs: Mapping[int, list],
    builtins: Mapping[int, list],
    rand: Callable[[np.ndarray], np.ndarray] | None,
    n: int,
    variables: dict[int, list] | None = None,
) -> list[np.ndarray]:
    """Return ``evaluate`` of the block over ``n`` lanes at once, bit for bit.

    ``builtins`` components may be lane arrays (or numbers, the same in all
    lanes); ``rand(a)`` takes and returns lane arrays. ``variables``, read
    and written in place, holds lane arrays (int64 for ints and bools,
    float64 for floats); empty if not given. Raises what evaluate raises,
    or Diverged where lanes would part ways (an if/else between values of
    different widths or kinds, a loop condition they disagree on): then
    evaluate each lane.
    """
    var: dict[int, list] = {} if variables is None else variables
    with np.errstate(all="ignore"):
        return _run(block, inputs, builtins, rand, var, Lanes(n))


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

    One pass, in order, even over while loops: a variable a loop sets is
    unbounded from the loop's start on, and so is the loop's value. Loops
    whose ranges cannot be read leave the whole block unbounded.
    """
    vals: list[list[Interval]] = []
    var: dict[int, list[Interval]] = {}
    try:
        starts = loops(block)
    except FunctionError:
        return [_ALL] * block.instrs[-1].width
    loose: set[int] = set()
    """Variables a loop sets: unbounded."""
    for k, ins in enumerate(block.instrs):
        ops = ins.operands
        loop = starts.get(k)
        if loop is not None:
            for i in block.instrs[loop.start : loop.at + 1]:
                if i.op == 0x07:  # noqa: PLR2004
                    loose.add(i.operands[1])
                    var[i.operands[1]] = [_ALL] * i.width

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
            v = arg(0)
            var[ops[1]] = [_ALL] * len(v) if ops[1] in loose else v
        elif ins.op == 0x09:  # noqa: PLR2004
            a, b = _spread(arg(1), arg(2))
            v = [(min(p[0], q[0]), max(p[1], q[1])) for p, q in zip(a, b, strict=True)]
        elif ins.op == 0x0C:  # noqa: PLR2004
            if ops[1] != NO_VALUE:
                v = arg(1)
            elif len(arg(0)) != ins.width:  # as _dropped
                v = [(0.0, 0.0)] * ins.width
            elif ins.type == INT:  # rounded: a bound a bound still
                v = [
                    tuple(round(x) if math.isfinite(x) else x for x in iv)
                    for iv in arg(0)
                ]
            else:
                v = arg(0)
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
            a, b = _spread(arg(0), arg(1))
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
            a, b = _spread(arg(0), arg(1))
            v = [
                _imul(p, (1 / q[1], 1 / q[0])) if q[0] > 0 or q[1] < 0 else _ALL
                for p, q in zip(a, b, strict=True)
            ]
        elif ins.op == 0x16:  # noqa: PLR2004
            a, b = _spread(arg(0), arg(1))
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
            a, b = _spread(arg(0), arg(1))
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


TRIG_OPS = (0x26, 0x27, 0x2E)
"""cos, sin, cartesian: what engine_sin and engine_cos compute."""

ENGINE_TRIG_GLSL = """
float sbs_sin(float x) {
	precise float t = x * 0.159154937;
	t = t - roundEven(t);
	t = t > 0.25 ? 0.5 - t : (t < -0.25 ? -0.5 - t : t);
	precise float t2 = t * t;
	precise float p = 81.6052475 - 74.1371841 * t2;
	p = p * t2 - 41.3417015;
	p = p * t2 + 6.28318548;
	return p * t;
}
float sbs_cos(float x) {
	precise float a = x + 1.57079637;
	return sbs_sin(a);
}
"""
"""engine_sin and engine_cos in GLSL, for functions the cooker evaluates on
the CPU (an FX-Map's, a parameter's); ``precise`` keeps each step rounded."""


def to_glsl(
    block: Block,
    prefix: str,
    inputs: Callable[[int, int, int], str],
    builtins: Callable[[int, int], str],
    samples: Callable[[str, int, str, int], str] | None = None,
    rand: Callable[[str, int], str] | None = None,
    variables: Mapping[int, tuple[str, str]] | None = None,
    *,
    engine_trig: bool = False,
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
    ``engine_trig`` computes cos, sin and cartesian with ENGINE_TRIG_GLSL's
    functions (the caller adds them), as the cooker's CPU evaluator does.

    A while loop becomes one loop statement (docs/function-ops.md 1.7): its
    result declared before it, zero; ``cap`` iterations, or ``while (true)``
    without a cap; the condition range, a break when it is true, then the
    body range and the result set to the body's value. Variables first
    used in a loop, and values a loop computes that are read after it, are
    declared before the outermost loop, zero, and only assigned inside.
    """
    cos, sin = ("sbs_cos", "sbs_sin") if engine_trig else ("cos", "sin")
    instrs = block.instrs
    found = loops(block)
    hoisted_vars, hoisted_vals = _hoisted(block, found, variables or {})
    early = {j for js in hoisted_vals.values() for j in js}
    ends: dict[int, tuple[bool, Loop]] = {}
    """A loop's condition (False) or body (True) ending at an instruction."""
    for lp in found.values():
        ends[lp.cond], ends[lp.body] = (False, lp), (True, lp)
    lines: list[str] = []
    names = [f"{prefix}_{k}" for k in range(len(instrs))]
    kinds = [(i.type, i.width) for i in instrs]
    declared: dict[int, str] = {}
    """Variables declared so far, with their GLSL types."""
    depth = 0

    def emit(line: str) -> None:
        lines.append("\t" * depth + line)

    def define(k: int, expr: str) -> None:
        if k in early:
            emit(f"{names[k]} = {expr};")
        else:
            emit(f"{glsl_type(*kinds[k])} {names[k]} = {expr};")

    def v(i: int) -> str:
        return names[i]

    for k, ins in enumerate(instrs):
        ops = ins.operands
        t = glsl_type(ins.type, ins.width)
        loop = found.get(k)
        if loop is not None:
            for var, vt in hoisted_vars.get(k, []):
                declared[var] = vt
                emit(f"{vt} {prefix}_var{var} = {vt}(0);")
            for j in hoisted_vals.get(k, []):
                jt = glsl_type(*kinds[j])
                emit(f"{jt} {names[j]} = {jt}(0);")
            define(loop.at, f"{glsl_type(*kinds[loop.at])}(0)")
            cap = min(loop.cap, GLSL_INT_MAX)
            it = f"{prefix}_it{loop.at}"
            emit(
                f"for (int {it} = 0; {it} < {cap}; {it}++) {{"
                if cap
                else "while (true) {"
            )
            depth += 1
        if ins.op == WHILE:
            continue  # declared and computed by its loop statement
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
            emit(f"{glob} = {kind}({v(ops[0])});")
            expr = v(ops[0])
        elif ins.op == 0x04:  # noqa: PLR2004
            # A variable this block never set (one an FX-Map node's
            # ancestors set, it seems) reads as zero, as in evaluate.
            if ops[0] not in declared:
                declared[ops[0]] = t
                emit(f"{t} {prefix}_var{ops[0]} = {t}(0);")
            expr = f"{prefix}_var{ops[0]}"
            if declared[ops[0]] != t:
                expr = f"{t}({expr})"
        elif ins.op == 0x07:  # noqa: PLR2004
            if ops[1] in declared:
                emit(f"{prefix}_var{ops[1]} = {declared[ops[1]]}({v(ops[0])});")
            else:
                declared[ops[1]] = t
                emit(f"{t} {prefix}_var{ops[1]} = {v(ops[0])};")
            expr = v(ops[0])
        elif ins.op == 0x09:  # noqa: PLR2004
            expr = f"({v(ops[0])} ? {v(ops[1])} : {v(ops[2])})"
        elif ins.op == 0x0C:  # noqa: PLR2004
            if ops[1] != NO_VALUE:
                expr = v(ops[1])
            elif kinds[ops[0]][1] != ins.width:  # as _dropped
                expr = f"{t}(0)"
            elif ins.type == INT and kinds[ops[0]][0] == FLOAT:
                expr = f"{t}(round({v(ops[0])}))"
            else:
                expr = f"{t}({v(ops[0])})"
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
            expr = f"vec2({r} * {cos}({th}), {r} * {sin}({th}))"
        elif ins.op in (0x26, 0x27):
            expr = f"{cos if ins.op == 0x26 else sin}({v(ops[0])})"  # noqa: PLR2004
        elif ins.op == 0x2F:  # noqa: PLR2004
            expr = f"mix({v(ops[0])}, {v(ops[1])}, {v(ops[2])})"
        elif ins.op == 0x32 and rand is not None:  # noqa: PLR2004
            expr = rand(v(ops[0]), k)
        elif ins.op in (0x33, 0x34) and samples is not None:
            kind = "lum" if ins.op == 0x33 else "col"  # noqa: PLR2004
            # 0 nearest (the cooker leaves it out), 1 bilinear
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
        define(k, expr)
        if k in ends:
            is_body, lp = ends[k]
            if not is_body:
                cond = v(k) if ins.type == BOOL else f"bool({v(k)})"
                emit(f"if ({cond}) break;")
            else:
                lt = glsl_type(*kinds[lp.at])
                value = v(k) if lt == t else f"{lt}({v(k)})"
                emit(f"{names[lp.at]} = {value};")
                depth -= 1
                emit("}")
    return lines, names[-1]


def _reads(ins: Instr) -> list[int]:
    """Return the values (SSA indices) an instruction reads."""
    ops = ins.operands
    if ins.op in (0x00, 0x01, 0x02, 0x04):
        return []
    if ins.op in (0x07, 0x10, 0x11, 0x33, 0x34):
        return ops[:1]
    if ins.op == WHILE:
        return list(while_operands(ins)[:3])
    return [o for o in ops if o != NO_VALUE]


def _hoisted(
    block: Block, found: dict[int, Loop], shared: Mapping[int, tuple[str, str]]
) -> tuple[dict[int, list[tuple[int, str]]], dict[int, list[int]]]:
    """Return what to_glsl declares before each outermost loop (by its start).

    Variables first used inside a loop, (id, GLSL type), as its first use
    has them; and values computed inside a loop that something outside
    their innermost loop reads (a while reading its own ranges is the loop
    statement, inside them).
    """
    outer, inner = _enclosing(block, found)
    variables: dict[int, list[tuple[int, str]]] = {}
    seen: set[int] = set()
    for k, ins in enumerate(block.instrs):
        if ins.op not in (0x04, 0x07):
            continue
        var = ins.operands[0] if ins.op == 0x04 else ins.operands[1]  # noqa: PLR2004
        if var in shared or var in seen:
            continue
        seen.add(var)
        lp = outer[k]
        if lp is not None:
            entry = (var, glsl_type(ins.type, ins.width))
            variables.setdefault(lp.start, []).append(entry)
    early: set[int] = set()
    for r, ins in enumerate(block.instrs):
        if ins.op == WHILE:
            continue
        for j in _reads(ins):
            lp = inner[j] if 0 <= j < len(inner) else None
            if lp is not None and not lp.start <= r <= lp.body:
                early.add(j)
    values: dict[int, list[int]] = {}
    for j in sorted(early):
        lp = outer[j]
        if lp is not None:
            values.setdefault(lp.start, []).append(j)
    return variables, values
