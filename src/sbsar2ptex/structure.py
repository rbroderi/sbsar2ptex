"""Give a translated graph the shape an author would have built.

A compiled Substance graph is flat: thousands of atomic nodes, each library
node the author placed (a Blur HQ, a noise, a Tile Generator) inlined into
dozens of them. ``nest`` folds them back into Material Maker subgraphs.

The cooker writes each placed node's atomic nodes together, in the order
it walked the author's graph (recognized library instances are contiguous
in 158 of 164 cases in BiomechZombieCircuits), so stretches of the node
order are the parts of the author's graph: ``segment`` cuts the order
where the fewest values cross the cut, as an author's frames would hold
the parts of a material, into two levels (parts, then regions of a part),
never inside a recognized library node (sbsar2ptex.recognize). The library
nodes become named groups inside their regions; a region is named after
what it holds.

Every group becomes a Material Maker ``graph`` node with ``gen_inputs`` and
``gen_outputs`` ports (a region can have several of each), and parameters
are passed up through each level's ``gen_parameters`` remote to the top
"parameters" remote.
"""

import itertools
from collections import Counter
from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from sbsar2ptex.sbsasm import OPS
from sbsar2ptex.sbsasm import Node
from sbsar2ptex.translate import GRAPH_PARAM

PART = (300, 1200)
"""Smallest and largest part, in Substance nodes."""

REGION = (15, 80)
"""Smallest and largest region of a part."""


@dataclass
class Group:
    """A subgraph: a stretch of the compiled order, or a library node."""

    members: list[int]
    label: str
    parent: "Group | None" = None
    children: list["Group"] = field(default_factory=list)
    name: str = ""
    mm: list[dict[str, Any]] = field(default_factory=list)
    wires: list[dict[str, Any]] = field(default_factory=list)
    inputs: dict[tuple[str, int], int] = field(default_factory=dict)
    """Source at the parent's level -> this group's input port."""
    input_types: list[str] = field(default_factory=list)
    outputs: dict[tuple[str, int], int] = field(default_factory=dict)
    """Source inside -> this group's output port."""
    output_types: list[str] = field(default_factory=list)


def crossings(nodes: list[Node], order: list[int]) -> list[int]:
    """Return, for each cut before order[i], how many values cross it."""
    pos = {k: i for i, k in enumerate(order)}
    last = list(range(len(order)))
    for i, k in enumerate(order):
        for v in nodes[k].inputs.values():
            if v in pos:
                last[pos[v]] = max(last[pos[v]], i)
    delta = [0] * (len(order) + 1)
    for i, end in enumerate(last):
        if end > i:
            delta[i + 1] += 1
            delta[end + 1] -= 1
    out, live = [], 0
    for i in range(len(order)):
        live += delta[i]
        out.append(live)
    return out


def segment(sizes: list[int], cost: list[int], lo: int, hi: int) -> list[int]:
    """Return item indices to cut before, splitting items into pieces.

    ``sizes`` are the items' node counts and ``cost[i]`` what a cut before
    item i costs (the values crossing it). Pieces hold lo..hi nodes, or one
    item of any size; the cuts minimize the total cost. Too few nodes to
    split: one piece.
    """
    n = len(sizes)
    if sum(sizes) <= hi:
        return []
    inf = float("inf")
    best = [inf] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0
    for i in range(1, n + 1):
        here = cost[i] if i < n else 0
        total = 0
        for j in range(i - 1, -1, -1):
            total += sizes[j]
            if total > hi and j < i - 1:
                break
            if (lo <= total or j == i - 1 or (j == 0 and i == n)) and (
                best[j] + here < best[i]
            ):
                best[i], back[i] = best[j] + here, j
    cuts, i = [], n
    while i:
        i = back[i]
        if i:
            cuts.append(i)
    return sorted(cuts)


def _token(nodes: list[Node], order: list[int], pos: dict[int, int], i: int) -> tuple:
    """Return what makes node order[i] the same as another, wiring and all.

    Its op and, for each input, how far back it is (if near; else only that
    it comes from outside), not its constants.
    """
    node = nodes[order[i]]
    wiring = tuple(
        sorted(
            (name, i - pos[v] if v in pos and i - pos[v] <= REACH else 0)
            for name, v in node.inputs.items()
        )
    )
    return (node.op, wiring)


REACH = 96
"""How far back an input is told apart from an outside one."""

LENGTHS = (256, 192, 128, 96, 64, 48, 32, 24, 16, 12, 8)
"""Run lengths tried for repeats, longest first."""


def repeats(nodes: list[Node], order: list[int], taken: set[int]) -> list[list[range]]:
    """Return repeated runs of the order: each a list of its occurrences.

    The same library node placed twice with the same settings cooks to the
    same run of ops, wired the same; runs are found longest first, each
    occurrence extended while all occurrences agree, and never over
    ``taken`` positions or each other.
    """
    pos = {k: i for i, k in enumerate(order)}
    tokens = [_token(nodes, order, pos, i) for i in range(len(order))]
    ids: dict[tuple, int] = {}
    seq = [ids.setdefault(t, len(ids)) for t in tokens]
    used = set(taken)
    found: list[list[range]] = []
    for length in LENGTHS:
        windows: dict[tuple, list[int]] = {}
        for i in range(len(seq) - length + 1):
            windows.setdefault(tuple(seq[i : i + length]), []).append(i)
        for starts in windows.values():
            if len(starts) < 2:  # noqa: PLR2004
                continue
            free, last = [], -1
            for i in starts:
                span = range(i, i + length)
                if i >= last and not any(k in used for k in span):
                    free.append(i)
                    last = i + length
            if len(free) < 2:  # noqa: PLR2004
                continue
            n = length
            while all(
                i + n < len(seq)
                and i + n not in used
                and seq[i + n] == seq[free[0] + n]
                for i in free
            ) and all(free[k] + n + 1 <= free[k + 1] for k in range(len(free) - 1)):
                n += 1
            spans = [range(i, i + n) for i in free]
            for span in spans:
                used.update(span)
            found.append(spans)
    return found


GENERATORS = {8: "Pattern", 9: "Pattern", 40: "Pixel processor",
              41: "Pixel processor", 0: "Gradient", 1: "Gradient",
              42: "Distance", 43: "Distance"}  # fmt: skip
"""Ops worth naming a group after: what makes its content."""


def _counted(names: Iterable[str], most: int = 2) -> str:
    """Return the commonest names, counted: "Highpass x6, Clouds"."""
    return ", ".join(
        f"{name} x{n}" if n > 1 else name
        for name, n in Counter(names).most_common(most)
    )


def _describe(nodes: list[Node], members: list[int], inner: list[str]) -> str:
    """Name a region: its library nodes, or what makes it and what it ends in."""
    made = [GENERATORS[nodes[k].op] for k in members if nodes[k].op in GENERATORS]
    if inner:
        extra = f" + {_counted(made, 1)}" if made else ""
        return _counted(inner) + extra
    last = OPS.get(nodes[members[-1]].op, "node").capitalize()
    if made:
        return f"{_counted(made, 1)} → {last}"
    ops = Counter(OPS.get(nodes[k].op, "node") for k in members)
    return _counted(name.capitalize() for name in ops.elements())


Resemble = Callable[[list[int]], str | None]
"""Name a run of nodes after the library node it resembles, if any."""


def groups(
    nodes: list[Node],
    need: set[int],
    instances: list[tuple[list[int], str]],
    resemble: Resemble | None = None,
) -> tuple[dict[int, Group], list[Group]]:
    """Return each node's innermost group and the top groups (the parts).

    ``instances`` are recognized library nodes: their members and names;
    ``resemble`` names repeated runs that are not.
    """
    order = sorted(need)
    pos = {k: i for i, k in enumerate(order)}
    cost = crossings(nodes, order)
    # Items: what stays whole. Recognized library nodes, then repeated runs
    # (copies of one placed node), then single nodes.
    spans: list[tuple[range, str, list[int] | None]] = []
    taken: set[int] = set()
    for members, label in instances:
        ps = sorted(pos[k] for k in members if k in pos)
        if ps and ps[-1] - ps[0] + 1 == len(ps):
            span = range(ps[0], ps[-1] + 1)
            spans.append((span, label, members))
            taken.update(span)
    unit = 0
    for occurrences in repeats(nodes, order, taken):
        first = [order[i] for i in occurrences[0]]
        label = resemble(first) if resemble else None
        if label is None:
            unit += 1
            label = f"Unit {_letters(unit)}"
        spans += [(span, label, None) for span in occurrences]
    spans.sort(key=lambda sp: sp[0].start)
    items: list[tuple[int, int, str | None]] = []  # start, stop, label
    at = 0
    for span, label, _ in spans:
        items += [(i, i + 1, None) for i in range(at, span.start)]
        items.append((span.start, span.stop, label))
        at = span.stop
    items += [(i, i + 1, None) for i in range(at, len(order))]
    owner: dict[int, Group] = {}
    top: list[Group] = []

    def pieces(lo: int, hi: int, first: int, last: int) -> list[tuple[int, int]]:
        """Split items first..last-1: their (start item, stop item) pieces."""
        sizes = [items[t][1] - items[t][0] for t in range(first, last)]
        cuts = segment(sizes, [cost[items[t][0]] for t in range(first, last)], lo, hi)
        edges = [first, *(first + c for c in cuts), last]
        return list(itertools.pairwise(edges))

    def span_of(a: int, b: int) -> list[int]:
        return order[items[a][0] : items[b - 1][1]]

    for number, (a, b) in enumerate(pieces(*PART, 0, len(items)), 1):
        part = Group(span_of(a, b), "")
        top.append(part)
        for c, d in pieces(*REGION, a, b):
            region = Group(span_of(c, d), "", parent=part)
            part.children.append(region)
            for k in region.members:
                owner[k] = region
            for t in range(c, d):
                i0, i1, label = items[t]
                if label is None:
                    continue
                inner = Group(order[i0:i1], label, parent=region)
                region.children.append(inner)
                for k in inner.members:
                    owner[k] = inner
            region.label = _describe(
                nodes, region.members, [g.label for g in region.children]
            )
        named = [g.label for r in part.children for g in r.children]
        part.label = f"Part {number}: " + (
            _counted(named, 3) or _describe(nodes, part.members, [])
        )
    # A region that is one item is that item's group; a part of one region,
    # that region.
    for part in top:
        for region in list(part.children):
            if len(region.children) == 1 and len(region.children[0].members) == len(
                region.members
            ):
                only = region.children[0]
                only.parent = part
                part.children[part.children.index(region)] = only
        if len(part.children) == 1:
            only = part.children[0]
            part.children = only.children
            part.label = f"{part.label.split(':')[0]}: {only.label}"
            for c in part.children:
                c.parent = part
            for k, g in owner.items():
                if g is only:
                    owner[k] = part
    return owner, top


def _letters(n: int) -> str:
    """Return 1 -> A, 26 -> Z, 27 -> AA..."""
    out = ""
    while n:
        n, r = divmod(n - 1, 26)
        out = chr(65 + r) + out
    return out


def nest(
    mm: list[dict[str, Any]],
    wires: list[dict[str, Any]],
    outs: dict[str, tuple[str, int]],
    origins: dict[str, int],
    nodes: list[Node],
    instances: list[tuple[list[int], str]] | None = None,
    resemble: Resemble | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, tuple[str, int]]]:
    """Return the translated graph folded into nested subgraphs.

    ``mm``, ``wires`` and ``outs`` are translate_graph's results and
    ``origins`` its Material Maker node -> Substance node map;
    ``instances`` the recognized library nodes (members, name).
    """
    need = set(origins.values())
    owner, top = groups(nodes, need, instances or [], resemble)
    by_name = {n["name"]: n for n in mm}
    group_of: dict[str, Group | None] = {}
    for n in mm:
        k = origins.get(n["name"])
        group_of[n["name"]] = owner.get(k) if k is not None else None
    count = Counter()

    def name_groups(gs: list[Group]) -> None:
        for g in gs:
            base = (
                "".join(c if c.isalnum() else "_" for c in g.label.lower()) or "group"
            )
            count[base] += 1
            g.name = f"{base}_{count[base]}"
            name_groups(g.children)

    name_groups(top)
    for n in mm:
        g = group_of[n["name"]]
        if g is not None:
            g.mm.append(n)
    top_nodes = [n for n in mm if group_of[n["name"]] is None]
    top_wires: list[dict[str, Any]] = []

    def chain(g: Group | None) -> list[Group]:
        out = []
        while g is not None:
            out.append(g)
            g = g.parent
        return out

    def out_type(node: str) -> str:
        n = by_name.get(node)
        if n is None:
            return "rgba"
        if n["type"] == "shader":
            return n["shader_model"]["outputs"][0]["type"]
        return "rgba"

    def wires_of(g: Group | None) -> list[dict[str, Any]]:
        return g.wires if g is not None else top_wires

    def add(g: Group | None, src: tuple[str, int], dst: tuple[str, int]) -> None:
        w = {"from": src[0], "from_port": src[1], "to": dst[0], "to_port": dst[1]}
        if w not in wires_of(g):
            wires_of(g).append(w)

    def route(
        src: tuple[str, int], ga: Group | None, dst: tuple[str, int], gb: Group | None
    ) -> None:
        a, b = chain(ga), chain(gb)
        common = next((g for g in a if g in b), None)
        kind = out_type(src[0])
        for g in a[: a.index(common) if common in a else len(a)]:
            if src not in g.outputs:
                g.outputs[src] = len(g.outputs)
                g.output_types.append(kind)
                add(g, src, ("gen_outputs", g.outputs[src]))
            src = (g.name, g.outputs[src])
        down = b[: b.index(common) if common in b else len(b)]
        level = common
        for g in reversed(down):
            if src not in g.inputs:
                g.inputs[src] = len(g.inputs)
                g.input_types.append(kind)
                add(level, src, (g.name, g.inputs[src]))
            src = ("gen_inputs", g.inputs[src])
            level = g
        add(level, src, dst)

    for w in wires:
        route(
            (w["from"], w["from_port"]),
            group_of.get(w["from"]),
            (w["to"], w["to_port"]),
            group_of.get(w["to"]),
        )
    new_outs = {}
    for name, (node, port) in outs.items():
        src = (node, port)
        for g in chain(group_of.get(node)):
            if src not in g.outputs:
                g.outputs[src] = len(g.outputs)
                g.output_types.append(out_type(src[0]))
                add(g, src, ("gen_outputs", g.outputs[src]))
            src = (g.name, g.outputs[src])
        new_outs[name] = src
    declared = {
        prm["name"]: prm
        for n in mm
        if n["type"] == "shader"
        for prm in n["shader_model"]["parameters"]
    }

    def params_of(n: dict[str, Any]) -> list[str]:
        """Return the graph inputs a node reads (its own knobs stay on it)."""
        if n["type"] == "shader":
            return [
                p["name"]
                for p in n["shader_model"]["parameters"]
                if p["name"].startswith(GRAPH_PARAM)
            ]
        if n["type"] == "graph":
            return list(n["parameters"])
        return []

    def build(g: Group) -> dict[str, Any]:
        inner = [build(c) for c in g.children] + g.mm
        exposed: dict[str, list[str]] = {}
        for n in inner:
            for p in params_of(n):
                exposed.setdefault(p, []).append(n["name"])
        ios = [
            _ios("gen_inputs", g.input_types),
            _ios("gen_outputs", g.output_types),
            _remote("gen_parameters", exposed, declared),
        ]
        layout([*inner, *ios[:2]], g.wires)
        return {
            "name": g.name,
            "label": g.label,
            "type": "graph",
            "node_position": {"x": 0, "y": 0},
            "parameters": {p: declared[p]["default"] for p in exposed},
            "seed_int": 0,
            "nodes": [*inner, *ios],
            "connections": g.wires,
        }

    graphs = [build(g) for g in top]
    top_nodes = [n for n in top_nodes if n["type"] != "remote"] + graphs
    exposed: dict[str, list[str]] = {}
    for n in top_nodes:
        for p in params_of(n):
            exposed.setdefault(p, []).append(n["name"])
    layout(top_nodes, top_wires)
    if exposed:
        top_nodes.append(_remote("parameters", exposed, declared))
    return top_nodes, top_wires, new_outs


def _ios(name: str, types: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "type": "ios",
        "node_position": {"x": 0, "y": 0},
        "parameters": {},
        "ports": [{"name": f"port{k}", "type": t} for k, t in enumerate(types)],
    }


def _remote(
    name: str, exposed: dict[str, list[str]], declared: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Return a remote node exposing each parameter (by name) across its users."""
    widgets = [
        {
            "name": p,
            "label": declared[p]["label"],
            "type": "linked_control",
            "linked_widgets": [{"node": u, "widget": p} for u in users],
        }
        for p, users in sorted(exposed.items())
    ]
    return {
        "name": name,
        "type": "remote",
        "node_position": {"x": -400, "y": 0},
        "parameters": {p: declared[p]["default"] for p in sorted(exposed)},
        "seed_int": 0,
        "widgets": widgets,
    }


def layout(mm: list[dict[str, Any]], wires: list[dict[str, Any]]) -> None:
    """Place nodes in columns by their distance from the sources, left to right."""
    names = {n["name"] for n in mm}
    preds: dict[str, set[str]] = {n: set() for n in names}
    for w in wires:
        if w["to"] in names and w["from"] in names and w["from"] != w["to"]:
            preds[w["to"]].add(w["from"])
    col: dict[str, int] = {}
    left = {n: len(p) for n, p in preds.items()}
    succ: dict[str, list[str]] = {n: [] for n in names}
    for n, ps in preds.items():
        for p in ps:
            succ[p].append(n)
    ready = [n for n, k in left.items() if k == 0]
    while ready:
        n = ready.pop()
        col[n] = 1 + max((col[p] for p in preds[n]), default=-1)
        for m in succ[n]:
            left[m] -= 1
            if left[m] == 0:
                ready.append(m)

    def depth(n: str) -> int:
        return col.get(n, 0)

    rows: Counter = Counter()
    for n in mm:
        c = depth(n["name"])
        if n["name"] == "gen_inputs":
            c = -1
        if n["name"] == "gen_outputs":
            c = max(col.values(), default=0) + 1
        n["node_position"] = {"x": 260 * c, "y": 160 * rows[c]}
        rows[c] += 1
