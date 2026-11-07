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

from sbsar2ptex.function import GETSLOT
from sbsar2ptex.function import SETSLOT
from sbsar2ptex.function import Block
from sbsar2ptex.function import FunctionError
from sbsar2ptex.function import Interval
from sbsar2ptex.function import bounds
from sbsar2ptex.function import decode
from sbsar2ptex.function import evaluate
from sbsar2ptex.sbsasm import BASE
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


Size = int | tuple[int, int]
"""A graph's output size, log2: square, or (width, height)."""


def builtins(size: Size) -> dict[int, list[float]]:
    """Return a Value Processor's built-ins at the graph's output ``size``.

    2.4: $size (1) is the graph's size in pixels and $sizelog2 (3) its log2,
    even with an image input; $pos (8) is (0, 0); $time (0), $number (10)
    and $depth (12) are 0. Others raise FunctionError (evaluate's).
    """
    w, h = (size, size) if isinstance(size, int) else size
    return {
        0: [0.0],
        1: [float(1 << w), float(1 << h)],
        3: [float(w), float(h)],
        8: [0.0, 0.0],
        10: [0.0],
        12: [0.0],
    }


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
        size: Size,
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
    size: Size,
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


# Library templates (2.6): a template's slots are its own cook's. Its
# writers are matched to the instance's by node, through the readers the
# recognizer matched, never by slot number.


class Aliased(MutableMapping[int, list]):
    """A template's view of the graph's slot ``store``.

    Its slot s is the graph's ``alias[s]`` (template_slots). A slot not
    mapped, or a write, raises FunctionError.
    """

    def __init__(self, store: MutableMapping[int, list], alias: dict[int, int]) -> None:
        """Wrap ``store`` for a template whose slots ``alias`` maps."""
        self.store, self.alias = store, alias

    def _graph(self, slot: int) -> int:
        if slot not in self.alias:
            msg = f"template slot {slot}: matched to no writer of this graph"
            raise FunctionError(msg)
        return self.alias[slot]

    def __getitem__(self, slot: int) -> list:
        """Return the value of the graph slot ``slot`` maps to."""
        return self.store[self._graph(slot)]

    def __setitem__(self, slot: int, value: list) -> None:
        """Refuse: a template's reader writes no slot."""
        msg = f"template slot {slot}: written by a reader"
        raise FunctionError(msg)

    def __delitem__(self, slot: int) -> None:
        """Refuse, as a write."""
        self[slot] = []

    def __iter__(self) -> Iterator[int]:
        """Iterate over the mapped slots."""
        return iter(self.alias)

    def __len__(self) -> int:
        """Return how many slots are mapped."""
        return len(self.alias)


def _blocks(node: Node) -> list[int]:
    """Return the file offsets of ``node``'s code blocks, in file order.

    Its code offsets, and its fields' and inline words that point into it
    (parameter functions); a template node and its instance's lay them out
    alike, so their blocks pair in this order.
    """
    found = set(node.code)
    for w in (*node.fields, *node.inline):
        at = w + BASE
        if node.start < at < node.end and w % 4 == 0:
            found.add(at)
    return sorted(found)


def reads(node: Node) -> list[tuple[int, int, int]]:
    """Return the get slots of ``node``'s blocks, (slot, type, width), in order.

    Blocks that do not decode are skipped.
    """
    out: list[tuple[int, int, int]] = []
    for at in _blocks(node):
        try:
            block = decode(node.at(at, (node.end - at) // 4))
        except (IndexError, ValueError):
            continue
        out += [
            (i.operands[0], i.type, i.width)
            for i in block.instrs
            if i.op == GETSLOT and i.operands  # (words that are no block: none)
        ]
    return out


def template_slots(
    members: dict[int, int], template: Sequence[Node], nodes: Sequence[Node]
) -> dict[int, int]:
    """Return a template instance's slot map: template slot -> graph slot.

    ``members`` maps the template's nodes to the instance's (recognize's
    Match). Each matched node's get slots pair, in order, with its
    template node's when both read as many, of the same types and widths;
    each pair matches the template slot's writer node to the graph slot's.
    A template writer matched to two graph writers, or the reverse, is
    left out, as is any pairing that does not line up: its template reads
    are then refused (Aliased).
    """
    t_table, g_table = writers(template), writers(nodes)
    pairs: dict[int, set[int]] = {}
    back: dict[int, set[int]] = {}
    for tk, gk in members.items():
        if not (0 <= tk < len(template) and 0 <= gk < len(nodes)):
            continue
        t_reads, g_reads = reads(template[tk]), reads(nodes[gk])
        if len(t_reads) != len(g_reads):
            continue
        if any(t[1:] != g[1:] for t, g in zip(t_reads, g_reads, strict=True)):
            continue
        for (ts, _, _), (gs, _, _) in zip(t_reads, g_reads, strict=True):
            tw, gw = t_table.get(ts, []), g_table.get(gs, [])
            if len(tw) == 1 and len(gw) == 1:
                pairs.setdefault(tw[0].node, set()).add(gw[0].node)
                back.setdefault(gw[0].node, set()).add(tw[0].node)
    slot_of = {w[0].node: s for s, w in t_table.items() if len(w) == 1}
    g_slot_of = {w[0].node: s for s, w in g_table.items() if len(w) == 1}
    out: dict[int, int] = {}
    for tw, gws in pairs.items():
        if len(gws) == 1:
            (gw,) = gws
            if len(back[gw]) == 1:
                out[slot_of[tw]] = g_slot_of[gw]
    return out
