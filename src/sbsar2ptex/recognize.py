"""Recognize Substance library nodes inside a compiled graph.

Cooking inlines every library node an author placed (Blur HQ, Tile
Generator, a noise...) into atomic nodes, so a compiled graph is thousands
of them with the author's choices gone. But library graphs cook the same
way on their own: each of their outputs is a template, a DAG of atomic ops
from the output back to the library graph's inputs. Finding a template's
DAG in the compiled graph, wired the same, finds an instance, and its nodes
can become one node again, named as the author knew it.

Matching walks back from a candidate root: ops must be equal (image type
included), and so the named inputs, recursively; a template's graph inputs
match any node (consistently). FX-Maps compare their trees' shapes and
pixel processors their functions' ops, not values: instances differ in
parameters, which cooking folds into constants. Cooking also drops a blend
whose opacity the author left constant at 0 or 1: library graphs choose a
quality, a mode, a blade count with such switches, their opacity a function
of the parameter in the template. So matching may skip a template blend
whose opacity is a function, to the input it would pass on; the author's
choice was constant, so no knob is lost. A match folded below MIN_SHARE of
its template is not evidence of it, and of overlapping matches the one
with the most nodes matched, less half those skipped, wins. A template with
fewer than MIN_NODES nodes is too common to mean anything. ``resembler``
names what no template matches after the template it most resembles.
"""

import math
import os
import re
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from sbsar2ptex import fxmap
from sbsar2ptex.archive import member
from sbsar2ptex.archive import read_archive
from sbsar2ptex.function import decode
from sbsar2ptex.sbsasm import BLEND_OPACITY
from sbsar2ptex.sbsasm import Asm
from sbsar2ptex.sbsasm import Node
from sbsar2ptex.translate import GraphInput
from sbsar2ptex.translate import graph_inputs

MIN_NODES = 3
MIN_SHARE = 0.45
"""Least share of its template a match must keep (the rest skipped switches)."""
BUDGET = (40, 500)
"""Walk steps a match may take: so many per template node, and more
(skipping a switch backtracks; a match that fails late is costly)."""
PARALLEL_FROM = 64
"""Fewer templates than this are tried in this process."""
SKIP_DEPTH = 4
"""How many switches deep an instance's root may sit in its template."""
FXMAP_OPS = (8, 9)
INPUT_OPS = (32, 33)
BLEND_OPS = (2, 3)
FUNCTION_OPACITY = 0x20
"""A blend's opacity state (BLEND_OPACITY bits): a function."""


@dataclass
class Template:
    """One output of a library graph, as atomic nodes."""

    package: str
    graph: str
    """The graph's package URL, pkg://name."""
    output: str
    nodes: list[Node]
    root: int
    members: list[int] = field(default_factory=list)
    """Its nodes (indices into ``nodes``), graph inputs excluded."""
    label: str = ""
    inputs: dict[int, GraphInput] = field(default_factory=dict)
    """The package's (non-image) inputs by uid: the library node's parameters,
    which its nodes' functions read."""


@dataclass
class Match:
    """A template found in a graph."""

    template: Template
    root: int
    members: dict[int, int]
    """Template node -> graph node, for the template's own nodes."""
    inputs: dict[int, int]
    """Template input node -> the graph node it reads."""


def _shape(node: Node) -> tuple:
    """Return what must agree between a template node and a graph node."""
    key: tuple = (node.op, tuple(sorted(node.inputs)))
    if node.op in FXMAP_OPS:
        try:
            color = node.op == FXMAP_OPS[1]
            key += (_tree_shape(fxmap.read_tree(node, color=color)),)
        except (ValueError, IndexError):
            key += ("?",)
    elif node.op in (40, 41) and node.code:
        try:
            block = decode(node.at(node.code[-1], (node.end - node.code[-1]) // 4))
            key += (tuple(i.op for i in block.instrs),)
        except (IndexError, ValueError):
            key += ("?",)
    return key


def _tree_shape(tree: fxmap.Tree | None) -> tuple:
    seen: dict[int, int] = {}

    def walk(t: fxmap.Tree | None) -> tuple:
        if t is None:
            return ()
        if id(t) in seen:
            return ("ref", seen[id(t)])
        seen[id(t)] = len(seen)
        if isinstance(t, fxmap.Quadrant):
            return ("q", t.pattern, t.image, tuple(sorted(t.params)),
                    tuple(walk(c) for c in t.children))  # fmt: skip
        if isinstance(t, fxmap.Iterate):
            return ("i", walk(t.first), walk(t.repeated))
        return ("s", walk(t.if_false), walk(t.if_true))

    return walk(tree)


def templates(sbsar: Path) -> list[Template]:
    """Return the templates of a cooked library package."""
    files = read_archive(sbsar)
    xml = member(files, ".xml").decode("utf-8")
    asm = Asm(member(files, ".sbsasm"))
    nodes = asm.nodes()
    names = {int(u): n for u, n in re.findall(r'uid="(\d+)" identifier="([^"]*)"', xml)}
    # Which graph each output belongs to, in the description's order.
    owner: dict[int, tuple[str, str]] = {}
    for graph in re.finditer(r'<graph pkgurl="([^"]+)"(.*?)</graph>', xml, re.DOTALL):
        label = re.search(r'label="([^"]*)"', graph.group(2))
        for uid in re.findall(r'<output uid="(\d+)"', graph.group(2)):
            owner[int(uid)] = (graph.group(1), label.group(1) if label else "")
    inputs = graph_inputs(asm, xml)
    out = []
    for uid, (_, root) in zip(asm.output_uids, asm.output_nodes, strict=True):
        graph, label = owner.get(uid, ("?", ""))
        t = Template(sbsar.stem, graph, names.get(uid, str(uid)), nodes, root)
        t.label = label
        t.inputs = inputs
        seen: set[int] = set()
        todo = [root]
        while todo:
            k = todo.pop()
            if k in seen or not 0 <= k < len(nodes):
                continue
            seen.add(k)
            if nodes[k].op not in INPUT_OPS:
                t.members.append(k)
                todo += nodes[k].inputs.values()
        if len(t.members) >= MIN_NODES:
            out.append(t)
    return out


class _OverBudgetError(Exception):
    """A match took more than BUDGET steps."""


def skippable(node: Node) -> tuple[str, ...]:
    """Return the inputs a template node may be skipped to (a switch blend)."""
    if (
        node.op in BLEND_OPS
        and node.fields
        and node.fields[0] & BLEND_OPACITY == FUNCTION_OPACITY
    ):
        return tuple(k for k in ("destination", "source") if k in node.inputs)
    return ()


def match_at(t: Template, nodes: list[Node], root: int, shapes: dict) -> Match | None:
    """Return the template's instance rooted at graph node ``root``, if any.

    Template switch blends may be skipped (``skippable``); a match that
    takes more than BUDGET steps is given up.
    """
    members: dict[int, int] = {}
    inputs: dict[int, int] = {}
    used: set[int] = set()
    trail: list[tuple[bool, int]] = []  # (a member, not an input; template node)
    steps = 0

    def shape(ns: list[Node], k: int) -> tuple:
        cache = shapes.setdefault(id(ns), {})
        if k not in cache:
            cache[k] = _shape(ns[k])
        return cache[k]

    def undo(mark: int) -> None:
        while len(trail) > mark:
            is_member, k = trail.pop()
            if is_member:
                used.discard(members.pop(k))
            else:
                inputs.pop(k)

    limit = BUDGET[0] * len(t.nodes) + BUDGET[1]

    def walk(tk: int, gk: int) -> bool:  # noqa: PLR0911
        nonlocal steps
        steps += 1
        if steps > limit:
            raise _OverBudgetError
        tn = t.nodes[tk]
        if tn.op in INPUT_OPS:
            if tk in inputs:
                return inputs[tk] == gk
            inputs[tk] = gk
            trail.append((False, tk))
            return True
        if tk in members:
            return members[tk] == gk
        if not 0 <= gk < len(nodes):
            return False
        mark = len(trail)
        if gk not in used and shape(t.nodes, tk) == shape(nodes, gk):
            members[tk] = gk
            used.add(gk)
            trail.append((True, tk))
            gn = nodes[gk]
            if all(walk(tn.inputs[name], gn.inputs[name]) for name in tn.inputs):
                return True
            undo(mark)
        for name in skippable(tn):
            if walk(tn.inputs[name], gk):
                return True
            undo(mark)
        return False

    try:
        found = walk(t.root, root)
    except _OverBudgetError:
        return None
    return Match(t, root, members, inputs) if found else None


def heads(t: Template, k: int, depth: int = 0) -> list[int]:
    """Return the template nodes an instance's root can match: ``k``, or a switch's."""
    out = [k]
    if depth < SKIP_DEPTH:
        for name in skippable(t.nodes[k]):
            child = t.nodes[k].inputs[name]
            if t.nodes[child].op not in INPUT_OPS:
                out += heads(t, child, depth + 1)
    return out


def score(m: Match) -> float:
    """Return a match's weight: nodes matched, less half those it skipped.

    A big template skipped down to a fragment loses to a small one matched
    whole.
    """
    return len(m.members) - 0.5 * (len(m.template.members) - len(m.members))


_WORK: tuple[list[Node], list[Template]] = ([], [])
"""A worker's graph and library (set once per process)."""


def _start_worker(nodes: list[Node], library: list[Template]) -> None:
    global _WORK  # noqa: PLW0603 (one graph a worker)
    _WORK = (nodes, library)


def _found_by(which: list[int]) -> list[tuple[int, int, dict, dict]]:
    """Return the instances of library templates ``which`` (indices) in the graph.

    As (template index, root, members, inputs): an instance's inner nodes
    (all but its root) must not feed nodes outside it, or it is not one
    node's insides.
    """
    nodes, library = _WORK
    users: dict[int, set[int]] = {}
    for n in nodes:
        for v in n.inputs.values():
            users.setdefault(v, set()).add(n.index)
    by_root: dict[tuple, list[int]] = {}
    for i in which:
        t = library[i]
        for key in {_shape(t.nodes[h]) for h in heads(t, t.root)}:
            by_root.setdefault(key, []).append(i)
    shapes: dict = {}
    found = []
    for n in nodes:
        for i in by_root.get(_shape(n), []):
            t = library[i]
            m = match_at(t, nodes, n.index, shapes)
            if m is None or len(m.members) < max(MIN_NODES, MIN_SHARE * len(t.members)):
                continue
            mine = set(m.members.values())
            if all(users.get(k, set()) <= mine for k in mine - {m.root}):
                found.append((i, m.root, m.members, m.inputs))
    return found


def recognize(
    nodes: list[Node], library: list[Template], jobs: int | None = None
) -> list[Match]:
    """Return library instances in ``nodes``: the weightiest first, not overlapping.

    Templates are tried in ``jobs`` processes (all the CPU's by default; one
    for a small library).
    """
    jobs = jobs or os.cpu_count() or 1
    if jobs == 1 or len(library) < PARALLEL_FROM:
        _start_worker(nodes, library)
        raw = _found_by(list(range(len(library))))
    else:
        chunks = [list(range(k, len(library), jobs)) for k in range(jobs)]
        with ProcessPoolExecutor(
            jobs, initializer=_start_worker, initargs=(nodes, library)
        ) as pool:
            raw = [m for part in pool.map(_found_by, chunks) for m in part]
    found = [Match(library[i], root, members, ins) for i, root, members, ins in raw]
    # Ties: the smaller template, then a published graph over a helper.
    found.sort(key=lambda m: (-score(m), len(m.template.members), not m.template.label))
    taken: set[int] = set()
    kept = []
    for m in found:
        mine = set(m.members.values())
        if mine & taken:
            continue
        taken |= mine
        kept.append(m)
    return kept


RESEMBLE = 0.97
"""How alike (cosine of op counts) a run must be to a template to share its name."""


def resembler(
    nodes: list[Node], library: list[Template]
) -> Callable[[list[int]], str | None]:
    """Return a namer for runs of ``nodes`` no template matches exactly.

    A run is named "≈ <template>" after the template of a similar size
    (within a quarter) whose op counts are most alike, if alike enough: the
    same library node from another version, or pruned by its settings.
    """
    shapes = [(t, Counter(t.nodes[k].op for k in t.members)) for t in library]

    def cosine(a: Counter, b: Counter) -> float:
        dot = sum(a[k] * b[k] for k in a)
        norm = math.sqrt(
            sum(v * v for v in a.values()) * sum(v * v for v in b.values())
        )
        return dot / norm if norm else 0.0

    def name(members: list[int]) -> str | None:
        ops = Counter(nodes[k].op for k in members)
        best, score = None, RESEMBLE
        for t, tops in shapes:
            if not 0.8 <= len(t.members) / len(members) <= 1.25:  # noqa: PLR2004
                continue
            c = cosine(ops, tops)
            if c >= score:
                best, score = t, c
        if best is None:
            return None
        return "≈ " + (best.label or best.graph.removeprefix("pkg://"))

    return name
