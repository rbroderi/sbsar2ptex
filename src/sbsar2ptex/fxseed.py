"""FX-Map tree seeds computed by functions (docs/fxmap-node-0x09.md section 2).

A Quadrant, Iterate or Switch whose seed state is 2 (fxmap) runs a function
block for its seed each time the walk enters it, before its other functions,
with the variables as they are then. The block's value is an integer; the
seed is that integer's 32-bit pattern (section 2.1).

A seed block runs on raw 32-bit patterns, whatever an op's declared type:

- constants, graph inputs, builtins: a float's IEEE float32 bits, an
  integer's two's complement; ``getvar``: the pattern stored there;
- ``setvar``, ``sequence``, ``ifelse``, swizzles and vectors copy patterns;
- ``convert`` to an integer: the float32 bits of floor(x), x the pattern
  read as a float; ``convert`` to a float: the pattern unchanged;
- ``add``: the 32-bit integer sum of the patterns, ints or floats;
- ``sub``, ``mul``, ``div``, ``mod``, ``min``, ``max``: the float32 operation
  on the patterns read as floats, denormals as zero (``_read``);
- ``neg`` and ``abs``: bit 31 flipped or cleared;
- ``rand``: the ordinary-filter value generator started from the node's
  incoming state, private to this run of the block, bits 0-23 (2.5).

What the spec leaves untested raises FunctionError rather than guess
(section 6): comparisons and logic, ``ifelse`` on a computed condition,
other math ops (floor, pow, sqrt, trigonometry, lerp...), integer vector
arithmetic, samples, set slots, a sequence whose result was dropped, and
while loops. Only the low 24 bits of a seed are ever observable.

The variables a block sets hold their patterns: the rest of the walk reads
a float variable as the float of its pattern, an int one as its two's
complement (every value being one pattern, 2.1).
"""

import math
import struct
from collections.abc import Mapping
from collections.abc import MutableMapping

from sbsar2ptex.function import BOOL
from sbsar2ptex.function import FLOAT
from sbsar2ptex.function import INT
from sbsar2ptex.function import NO_VALUE
from sbsar2ptex.function import Block
from sbsar2ptex.function import FunctionError
from sbsar2ptex.function import Instr
from sbsar2ptex.function import constant
from sbsar2ptex.function import f32
from sbsar2ptex.valuerng import VALUE_LCG_C
from sbsar2ptex.valuerng import VALUE_LCG_M

MASK = 0xFFFFFFFF
SIGN = 0x80000000
EXPONENT = 0x7F800000
QUIET_NAN = 0xFFC00000
"""0 / 0's NaN (its sign is not observable, 2.1)."""

CONST, BUILTIN, INPUT, GETSLOT, GETVAR, SETVAR = 0x00, 0x01, 0x02, 0x03, 0x04, 0x07
SELECT, SEQUENCE, SWIZZLE, CONVERT = 0x09, 0x0C, 0x10, 0x11
ADD, SUB, MUL, DIV, MOD, NEG, ABS, MIN, MAX, RAND = (
    0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x23, 0x30, 0x31, 0x32,
)  # fmt: skip
CONCAT = (0x0D, 0x0E, 0x0F)
LEAVES = (BUILTIN, INPUT, GETSLOT)
FLOAT_OPS = (SUB, MUL, DIV, MOD, MIN, MAX)


def bits(x: float) -> int:
    """Return float ``x``'s float32 bits (rounded to float32)."""
    return struct.unpack("<I", struct.pack("<f", f32(x)))[0]


def float_bits(x: float) -> int:
    """Return the float32 bits of floor(x) (of x itself, if not finite)."""
    y = float(math.floor(x)) if math.isfinite(x) else x
    return bits(y)


def _is_nan(p: int) -> bool:
    return p & EXPONENT == EXPONENT and bool(p & 0x7FFFFF)


def _read(p: int) -> float:
    """Return pattern ``p`` read as a float: a denormal as a zero of its sign."""
    if not p & EXPONENT:
        return -0.0 if p & SIGN else 0.0
    return struct.unpack("<f", struct.pack("<I", p & MASK))[0]


def _floor(p: int) -> int:
    """Return ``convert``'s pattern of float pattern ``p`` (2.1)."""
    if p & EXPONENT == EXPONENT:  # NaN or infinity: kept
        return p
    x = _read(p)
    if x == 0:  # a zero or a denormal: the signed zero
        return p & SIGN
    return bits(float(math.floor(x)))


def _float_op(op: int, a: int, b: int) -> int:  # noqa: PLR0911
    """Return a float op's pattern on patterns ``a`` and ``b`` (2.1)."""
    if op in (MIN, MAX):  # (the second operand when either is a NaN)
        if _is_nan(a) or _is_nan(b):
            return b if _is_nan(b) else bits(_read(b))
        x, y = _read(a), _read(b)
        pick = x < y if op == MIN else x > y
        return bits(x if pick else y)
    if _is_nan(a):
        return a
    if _is_nan(b):
        return b
    x, y = _read(a), _read(b)
    if op == SUB:
        return bits(x - y)
    if op == MUL:
        return bits(x * y)  # (inf times 0: a NaN)
    if op == DIV:
        if y == 0:
            if x == 0:
                return QUIET_NAN
            negative = (x < 0) != (math.copysign(1.0, y) < 0)
            return 0xFF800000 if negative else EXPONENT
        return bits(x / y)  # (inf / inf: a NaN)
    # mod: x - y * floor(x / y), each step in float32 (by 0: a NaN)
    if y == 0 or math.isinf(x):
        return QUIET_NAN
    if math.isinf(y):
        return a
    q = f32(x / y)
    if math.isinf(q):
        return QUIET_NAN
    return bits(x - f32(y * math.floor(q)))


def _to_pattern(kind: int, values: list) -> list[int]:
    """Return values of a type as patterns: float bits, two's complement."""
    if kind == FLOAT:
        return [bits(float(v)) for v in values]
    return [int(v) & MASK for v in values]


def _from_pattern(kind: int, pats: list[int]) -> list:
    """Return patterns as the values the rest of the walk reads, by type."""
    if kind == FLOAT:
        return [struct.unpack("<f", struct.pack("<I", p))[0] for p in pats]
    if kind == INT:
        return [p - (1 << 32) if p & SIGN else p for p in pats]
    return [bool(p) for p in pats]


class Seeds:
    """Runs seed blocks (one per walk)."""

    def seed(
        self,
        block: Block,
        inputs: Mapping[int, list],
        builtins: dict[int, list[float]],
        variables: dict[int, list],
        *,
        incoming: int = 0,
        slots: MutableMapping[int, list] | None = None,
    ) -> int:
        """Run seed ``block`` (setting ``variables``) and return its seed, a u32.

        ``incoming`` is the state the walk hands the node: rand's start.
        """
        instrs = block.instrs
        if not instrs or instrs[-1].type != INT or instrs[-1].width != 1:
            msg = "an FX-Map seed function whose value is not an int"
            raise FunctionError(msg)
        if block.loops():
            msg = "a while loop in an FX-Map seed function"
            raise FunctionError(msg)
        run = _Patterns(block, inputs, builtins, variables, incoming, slots)
        pats = run.run()
        for name, (kind, p) in run.stored.items():
            variables[name] = _from_pattern(kind, p)
        return pats[0] & MASK


class _Patterns:
    """One run of a seed block, on 32-bit patterns."""

    def __init__(
        self,
        block: Block,
        inputs: Mapping[int, list],
        builtins: dict[int, list[float]],
        variables: Mapping[int, list],
        incoming: int,
        slots: Mapping[int, list] | None,
    ) -> None:
        self.block, self.inputs, self.builtins = block, inputs, builtins
        self.variables, self.slots = variables, slots
        self.state = incoming & MASK
        self.pats: list[list[int]] = []
        self.stored: dict[int, tuple[int, list[int]]] = {}

    def run(self) -> list[int]:
        """Return the block's value: its last instruction's pattern."""
        for ins in self.block.instrs:
            self.pats.append(self.pattern(ins))
        last = self.pats[-1]
        if not last:
            msg = "an FX-Map seed function of no value"
            raise FunctionError(msg)
        return last

    def of(self, o: int, width: int = 1) -> list[int]:
        """Return operand ``o``'s pattern (NO_VALUE: ``width`` zeros)."""
        return [0] * width if o == NO_VALUE else self.pats[o]

    def untested(self, ins: Instr, why: str = "") -> FunctionError:
        what = why or f"{ins.name} (op {ins.op:#04x})"
        return FunctionError(f"an FX-Map seed function's {what} (untested)")

    def rand(self, amplitude: int) -> int:
        """Return rand's pattern: the value generator's next value (2.5)."""
        self.state = (self.state * VALUE_LCG_M + VALUE_LCG_C) & MASK
        a = _read(amplitude)
        return bits(a * ((self.state & 0xFFFFFF) / (1 << 24)))

    def leaf(self, ins: Instr) -> list:
        """Return a builtin's, graph input's or get slot's value."""
        op, ops = ins.op, ins.operands
        if op == BUILTIN:
            if ops[0] not in self.builtins:
                msg = f"an FX-Map seed function reads builtin {ops[0]}"
                raise FunctionError(msg)
            return list(self.builtins[ops[0]])
        if op == INPUT:
            return list(self.inputs[ins.u32s()[0]])
        if self.slots is None:
            msg = "cannot evaluate getslot (op 0x03) without a slot store"
            raise FunctionError(msg)
        return list(self.slots.get(ops[0], [0] * ins.width))

    def pattern(self, ins: Instr) -> list[int]:  # noqa: PLR0911
        """Return the next instruction's pattern, ``ins``."""
        op, ops, width, kind = ins.op, ins.operands, ins.width, ins.type
        if op == CONST:
            return _to_pattern(kind, constant(ins))
        if op in LEAVES:
            return _to_pattern(kind, self.leaf(ins))[:width]
        if op == GETVAR:
            name = ops[0]
            if name in self.stored:
                return self.stored[name][1]
            value = self.variables.get(name)
            return [0] * width if value is None else _to_pattern(kind, value)
        if op == SETVAR:
            p = self.of(ops[0], width)
            self.stored[ops[1]] = (kind, p)
            return p
        if op == SELECT:
            c = ops[0]
            source = self.block.instrs[c] if c != NO_VALUE else None
            if source is not None and source.op not in (CONST, INPUT, GETVAR):
                raise self.untested(ins, f"ifelse on a computed {source.name}")
            cond = c != NO_VALUE and bool(self.pats[c][0])
            return self.of(ops[1] if cond else ops[2], width)
        if op == SEQUENCE:
            if ops[1] == NO_VALUE:
                raise self.untested(ins, "sequence of a dropped result")
            return self.of(ops[1], width)
        if op == SWIZZLE:
            src = self.of(ops[0], 4)
            picks = [ops[1] >> (2 * c) & 3 for c in range(width)]
            if max(picks) >= len(src):
                raise self.untested(ins, "swizzle past its source")
            return [src[p] for p in picks]
        if op in CONCAT:
            parts = [x for o in ops if o != NO_VALUE for x in self.of(o)]
            return (parts + [0] * width)[:width]
        if op == CONVERT:
            src = ops[0]
            p = self.of(src, width)[:width]
            from_float = src != NO_VALUE and self.block.instrs[src].type == FLOAT
            if kind == INT and from_float:
                return [_floor(x) for x in p]
            if BOOL in (kind, self.block.instrs[src].type if src != NO_VALUE else kind):
                raise self.untested(ins, "convert of a bool")
            return p
        if op == RAND:
            if kind != FLOAT or width != 1:
                raise self.untested(ins, "rand not a float")
            return [self.rand(self.of(ops[0])[0])]
        if op in (NEG, ABS):
            p = self.of(ops[0], width)
            return [x ^ SIGN if op == NEG else x & ~SIGN & MASK for x in p]
        if op == ADD or op in FLOAT_OPS:
            if kind == INT and width > 1:
                raise self.untested(ins, f"integer vector {ins.name}")
            a, b = self.of(ops[0], width), self.of(ops[1], width)
            if len(a) != len(b):
                a, b = (a * len(b), b) if len(a) == 1 else (a, b * len(a))
            if op == ADD:
                return [(x + y) & MASK for x, y in zip(a, b, strict=True)]
            return [_float_op(op, x, y) for x, y in zip(a, b, strict=True)]
        raise self.untested(ins)
