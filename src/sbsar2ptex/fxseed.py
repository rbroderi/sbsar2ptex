"""FX-Map tree seeds computed by functions (docs/fxmap-node-0x09.md section 2).

A Quadrant, Iterate or Switch whose seed state is 2 (fxmap) runs a function
block for its seed each time the walk enters it, before its other functions,
with the variables as they are then. The block's value is an integer; the
seed is that integer's 32-bit pattern, which depends on how it was made:

- an integer constant N: N as two's complement;
- ``convert`` of a float x: the IEEE float32 bits of floor(x);
- ``add`` of two integers: the sum of their patterns, mod 2^32;
- ``getvar``: the pattern stored there (an unset variable: 0).

Only the low 24 bits of a seed are ever observable (section 2.1).

Patterns are followed through the ops that only pass integers on (if/else,
sequence, swizzle, concatenation, set variable). Any other integer op on a
value whose pattern is not its two's complement (a float conversion's) is
untested and raises FunctionError rather than guess; so do while loops and
rand in a seed block (section 6). A variable keeps a pattern only within
the seed block that set it: one set by another function reads as its value.
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
from sbsar2ptex.function import evaluate

MASK = 0xFFFFFFFF

GETVAR, SETVAR, SELECT, SEQUENCE, SWIZZLE, CONVERT, ADD = (
    0x04, 0x07, 0x09, 0x0C, 0x10, 0x11, 0x12,
)  # fmt: skip
CONCAT = (0x0D, 0x0E, 0x0F)
LEAVES = (0x01, 0x02, 0x03)
"""Builtins, graph inputs and get slots: their value's pattern is plain."""


def float_bits(x: float) -> int:
    """Return the float32 bits of floor(x) (of x itself, if not finite)."""
    y = float(math.floor(x)) if math.isfinite(x) else x
    return struct.unpack("<I", struct.pack("<f", y))[0]


class Seeds:
    """Runs seed blocks, prefixes of them kept compiled (one per walk)."""

    def __init__(self) -> None:
        """Keep no prefixes yet."""
        self._prefixes: dict[tuple[int, int], Block] = {}

    def _prefix(self, block: Block, k: int) -> Block:
        key = (id(block), k)
        found = self._prefixes.get(key)
        if found is None:
            found = self._prefixes[key] = Block(list(block.instrs[: k + 1]))
        return found

    def seed(
        self,
        block: Block,
        inputs: Mapping[int, list],
        builtins: dict[int, list[float]],
        variables: dict[int, list],
        *,
        slots: MutableMapping[int, list] | None = None,
    ) -> int:
        """Run seed ``block`` (setting ``variables``) and return its seed, a u32."""
        instrs = block.instrs
        if not instrs or instrs[-1].type != INT or instrs[-1].width != 1:
            msg = "an FX-Map seed function whose value is not an int"
            raise FunctionError(msg)
        if block.loops():
            msg = "a while loop in an FX-Map seed function"
            raise FunctionError(msg)
        before = dict(variables)
        # The real run: what it sets, the rest of the walk sees. No rand (its
        # generator is untested): evaluate raises FunctionError.
        evaluate(block, dict(inputs), builtins, None, variables, slots=slots)
        return _Patterns(self, block, inputs, builtins, before, slots).last()


class _Patterns:
    """One run's integer patterns, an instruction at a time."""

    def __init__(
        self,
        seeds: Seeds,
        block: Block,
        inputs: Mapping[int, list],
        builtins: dict[int, list[float]],
        before: dict[int, list],
        slots: MutableMapping[int, list] | None,
    ) -> None:
        self.seeds, self.block, self.inputs, self.builtins = (
            seeds, block, inputs, builtins,
        )  # fmt: skip
        self.before, self.slots = before, slots
        self.memo: dict[int, list] = {}
        self.pats: list[list[int] | None] = [None] * len(block.instrs)
        self.stored: dict[int, list[int]] = {}

    def value(self, k: int) -> list:
        """Return instruction ``k``'s value: its prefix run, on a copy."""
        if k not in self.memo:
            self.memo[k] = evaluate(
                self.seeds._prefix(self.block, k),  # noqa: SLF001
                dict(self.inputs), self.builtins, None, dict(self.before),
                slots=self.slots,
            )  # fmt: skip
        return self.memo[k]

    def plain(self, k: int) -> list[int]:
        """Return instruction ``k``'s value as two's complement words."""
        return [int(v) & MASK for v in self.value(k)]

    def of(self, o: int, width: int = 1) -> list[int]:
        """Return operand ``o``'s pattern (an int's, or a bool's)."""
        if o == NO_VALUE:
            return [0] * width
        p = self.pats[o]
        if p is not None:
            return p
        if self.block.instrs[o].type == BOOL:
            return self.plain(o)
        msg = f"an FX-Map seed function reads float #{o} as an int"
        raise FunctionError(msg)

    def last(self) -> int:
        """Return the block's seed: its last instruction's pattern."""
        for k, ins in enumerate(self.block.instrs):
            if ins.op == SETVAR:
                if ins.type == INT:
                    p = self.of(ins.operands[0], ins.width)
                    self.pats[k] = self.stored[ins.operands[1]] = p
                else:
                    self.stored.pop(ins.operands[1], None)
            elif ins.type == INT:
                self.pats[k] = self.pattern(k, ins)
        last = self.pats[-1]
        if not last:
            msg = "an FX-Map seed function of no value"
            raise FunctionError(msg)
        return last[0] & MASK

    def pattern(self, k: int, ins: Instr) -> list[int]:  # noqa: PLR0911
        """Return integer instruction ``k``'s pattern."""
        op, ops, width = ins.op, ins.operands, ins.width
        if op == 0x00:
            return [x & MASK for x in constant(ins)]
        if op in LEAVES:
            return self.plain(k)
        if op == GETVAR:
            return self.stored.get(ops[0]) or self.plain(k)
        if op == CONVERT:
            src = ops[0]
            if src == NO_VALUE:
                return [0] * width
            if self.block.instrs[src].type == FLOAT:
                return [float_bits(x) for x in self.value(src)][:width]
            return self.of(src, width)[:width]
        if op == ADD:
            a, b = self.of(ops[0], width), self.of(ops[1], width)
            if len(a) != len(b):
                a, b = (a * len(b), b) if len(a) == 1 else (a, b * len(a))
            return [(x + y) & MASK for x, y in zip(a, b, strict=True)]
        if op == SELECT:
            cond = ops[0] != NO_VALUE and bool(self.value(ops[0])[0])
            return self.of(ops[1] if cond else ops[2], width)
        if op == SEQUENCE:
            return self.of(ops[1], width) if ops[1] != NO_VALUE else self.plain(k)
        if op == SWIZZLE:
            src = self.of(ops[0], 4)
            picks = [ops[1] >> (2 * c) & 3 for c in range(width)]
            if max(picks) >= len(src):
                return self.plain(k)
            return [src[p] for p in picks]
        if op in CONCAT:
            parts = [x for o in ops if o != NO_VALUE for x in self.of(o)]
            return (parts + [0] * width)[:width]
        # Any other op: only on plain integers (their two's complement).
        for o in ops:
            if o == NO_VALUE or o >= k or self.block.instrs[o].type != INT:
                continue
            if self.of(o) != self.plain(o):
                msg = (
                    f"an FX-Map seed function's {ins.name} of a float-converted "
                    "integer (untested)"
                )
                raise FunctionError(msg)
        return self.plain(k)
