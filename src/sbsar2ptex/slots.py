"""Value slots: the values Value Processors pass to later nodes.

docs/function-ops.md section 2. A Value Processor (op 40, its value's width
in its first field's high 16 bits: sbsasm.value_width) has no image: its
function, its last code block, ends with a set slot (op 0x06) storing its
value at a word offset of the graph's slot store. Any later node's
functions read it with get slot (op 0x03): pixel processors' per-pixel
functions, FX-Map functions, parameter and output-size functions, other
Value Processors'.

A read is the writer's value, which may depend on graph inputs: it stays
live. Translated to GLSL, a get slot is the writer's block, as a function
of the graph inputs (translate's resolver); evaluated here (output sizes,
an FX-Map's structure), the writer's block is run with the current input
values (SlotStore). Every writer comes before its readers in the node
table (2.3); a read that breaks that, or that no single writer of the same
type and width answers, is refused (FunctionError).
"""

from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import MutableMapping
from collections.abc import Sequence
from dataclasses import dataclass

from sbsar2ptex.function import SETSLOT
from sbsar2ptex.function import Block
from sbsar2ptex.function import FunctionError
from sbsar2ptex.function import Interval
from sbsar2ptex.function import bounds
from sbsar2ptex.function import decode
from sbsar2ptex.function import evaluate
from sbsar2ptex.sbsasm import Node
from sbsar2ptex.sbsasm import value_width


@dataclass(frozen=True)
class Writer:
    """A Value Processor's set-slot block: the slot's value."""

    node: int
    """The Value Processor's index in the node table."""
    block: Block
    type: int
    width: int


Writers = dict[int, list[Writer]]
"""Each slot's writers (one in cooked files), by slot."""


def writers(nodes: Sequence[Node]) -> Writers:
    """Return the graph's Value Processors' set-slot blocks, by slot.

    The block is the node's last code offset's (2.4); one that does not end
    with a set slot writes nothing known (a read of its slot is refused).
    """
    out: Writers = {}
    for node in nodes:
        if not value_width(node) or not node.code:
            continue
        at = node.code[-1]
        try:
            block = decode(node.at(at, (node.end - at) // 4))
        except IndexError:
            continue
        if not block.instrs or block.instrs[-1].op != SETSLOT:
            continue
        last = block.instrs[-1]
        out.setdefault(last.operands[1], []).append(
            Writer(node.index, block, last.type, last.width)
        )
    return out


def writer(
    table: Writers, slot: int, kind: int | None, width: int | None, reader: int | None
) -> Writer:
    """Return the one writer of ``slot`` that node ``reader`` may read.

    Its type and width must be the read's (``kind`` and ``width``, when
    known), and it must come before ``reader`` in the node table (2.3).
    Anything else (never seen in cooked files: 2.10) raises FunctionError.
    """
    found = table.get(slot, [])
    if not found:
        msg = f"slot {slot}: no Value Processor writes it"
        raise FunctionError(msg)
    if len(found) > 1:
        msg = f"slot {slot}: written by nodes {[w.node for w in found]}"
        raise FunctionError(msg)
    (w,) = found
    if kind is not None and (w.type, w.width) != (kind, width):
        msg = (
            f"slot {slot}: read as type {kind} width {width},"
            f" written as type {w.type} width {w.width}"
        )
        raise FunctionError(msg)
    if reader is not None and w.node >= reader:
        msg = f"slot {slot}: read by node {reader}, written by later node {w.node}"
        raise FunctionError(msg)
    return w


def builtins(size: int) -> dict[int, list[float]]:
    """Return a Value Processor's built-ins: $size the graph's, $pos (0, 0).

    2.4: $size (1) was the graph's output size; $pos (8) is (0, 0). Others
    were not probed: reading them raises FunctionError (evaluate's).
    """
    n = float(1 << size)
    return {1: [n, n], 8: [0.0, 0.0]}


class SlotStore(MutableMapping[int, list]):
    """A graph's slot store that runs each slot's writer when it is read.

    Writers run with ``inputs`` (graph input values by uid) and Value
    Processor built-ins at graph size ``size``; a writer's own get slots
    read this store, as earlier nodes (``reader`` is the reading node, None
    if unknown). A value once run is kept. Set slots write it.
    """

    def __init__(
        self,
        table: Writers,
        inputs: dict[int, list],
        size: int,
        reader: int | None = None,
    ) -> None:
        """Make an empty store over the graph's writers ``table``."""
        self.table, self.inputs, self.size = table, inputs, size
        self.reader = reader
        self.values: dict[int, list] = {}

    def __getitem__(self, slot: int) -> list:
        """Return ``slot``'s value: its writer's, run with the inputs if not yet."""
        w = writer(self.table, slot, None, None, self.reader)
        if slot not in self.values:
            outer, self.reader = self.reader, w.node
            try:
                fixed = builtins(self.size)
                evaluate(w.block, self.inputs, fixed, None, {}, slots=self)
            finally:
                self.reader = outer
        return self.values[slot]

    def __setitem__(self, slot: int, value: list) -> None:
        """Store ``value`` at ``slot`` (a set slot)."""
        self.values[slot] = list(value)

    def __delitem__(self, slot: int) -> None:
        """Forget ``slot``'s value."""
        del self.values[slot]

    def __iter__(self) -> Iterator[int]:
        """Iterate over the slots run so far."""
        return iter(self.values)

    def __len__(self) -> int:
        """Return how many slots have run."""
        return len(self.values)


def slot_bounds(
    table: Writers,
    inputs: Callable[[int, int], list[Interval]],
    size: int,
    reader: int | None = None,
) -> Callable[[int, int, int], list[Interval]]:
    """Return a get slot's bound for function.bounds: its writer's value's.

    The writer's block bounded with ``inputs``, the Value Processor
    built-ins, and its own get slots so bounded; unbounded where the read
    is refused (``writer``).
    """
    fixed = {k: [(x, x) for x in v] for k, v in builtins(size).items()}

    def bound(slot: int, kind: int, width: int, reader: int | None = reader) -> list:
        try:
            w = writer(table, slot, kind, width, reader)
        except FunctionError:
            return [(-float("inf"), float("inf"))] * width

        def inner(s: int, k: int, n: int) -> list[Interval]:
            return bound(s, k, n, w.node)

        return bounds(w.block, inputs, fixed, slots=inner)

    return bound
