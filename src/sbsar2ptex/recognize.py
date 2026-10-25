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
parameters, which cooking folds into constants. A template with fewer than
MIN_NODES nodes is too common to mean anything. ``resembler`` names what
no template matches exactly after the template it most resembles.
"""

import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from sbsar2ptex import fxmap
from sbsar2ptex.archive import member
from sbsar2ptex.archive import read_archive
from sbsar2ptex.function import decode
from sbsar2ptex.sbsasm import Asm
from sbsar2ptex.sbsasm import Node

MIN_NODES = 3
FXMAP_OPS = (8, 9)
INPUT_OPS = (32, 33)


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
    out = []
    for uid, (_, root) in zip(asm.output_uids, asm.output_nodes, strict=True):
        graph, label = owner.get(uid, ("?", ""))
        t = Template(sbsar.stem, graph, names.get(uid, str(uid)), nodes, root)
        t.label = label
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


def match_at(t: Template, nodes: list[Node], root: int, shapes: dict) -> Match | None:
    """Return the template's instance rooted at graph node ``root``, if any."""
    members: dict[int, int] = {}
    inputs: dict[int, int] = {}

    def shape(ns: list[Node], k: int, cache: dict) -> tuple:
        if k not in cache:
            cache[k] = _shape(ns[k])
        return cache[k]

    def walk(tk: int, gk: int) -> bool:  # noqa: PLR0911
        tn = t.nodes[tk]
        if tn.op in INPUT_OPS:
            if tk in inputs:
                return inputs[tk] == gk
            inputs[tk] = gk
            return True
        if tk in members:
            return members[tk] == gk
        if not 0 <= gk < len(nodes):
            return False
        if shape(t.nodes, tk, shapes.setdefault(id(t.nodes), {})) != shape(
            nodes, gk, shapes.setdefault(id(nodes), {})
        ):
            return False
        if gk in members.values():
            return False
        members[tk] = gk
        gn = nodes[gk]
        return all(walk(tn.inputs[name], gn.inputs[name]) for name in tn.inputs)

    if walk(t.root, root):
        return Match(t, root, members, inputs)
    return None


def recognize(nodes: list[Node], library: list[Template]) -> list[Match]:
    """Return library instances in ``nodes``: the biggest first, not overlapping.

    An instance's inner nodes (all but its root) must not feed nodes outside
    it, or it is not one node's insides.
    """
    users: dict[int, set[int]] = {}
    for n in nodes:
        for v in n.inputs.values():
            users.setdefault(v, set()).add(n.index)
    by_root: dict[tuple, list[Template]] = {}
    for t in library:
        by_root.setdefault(_shape(t.nodes[t.root]), []).append(t)
    shapes: dict = {}
    found: list[Match] = []
    for n in nodes:
        for t in by_root.get(_shape(n), []):
            m = match_at(t, nodes, n.index, shapes)
            if m is None:
                continue
            inner = set(m.members.values()) - {m.root}
            if all(users.get(k, set()) <= set(m.members.values()) for k in inner):
                found.append(m)
    found.sort(key=lambda m: -len(m.members))
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
