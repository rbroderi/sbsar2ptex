"""Loops of a translated graph as Material Maker iterate buffers (--experimental).

Substance cooks a loop unrolled: Slope blur is 32 warps, each of the last,
and 31 blends keeping their running mean, 63 nodes and (translated) 34
buffers. The material-maker-mod fork's ``iterate_buffer_v2`` runs one
pass of such a loop as often as asked, its states in textures: the same
shaders, the same stored values, a pass at a time.

Slope blur (grayscale, a solved library instance: libparams) becomes one
iterate of two states, the image and the mean, both 32-bit floats (the
buffers they replace, holding stored() values exactly), or the image in
16 bits as its buffers were where its format keeps floats. Both start as
the first warp of the input (the template's warp, translated as today, so
its intensity is Intensity / Samples, live, and reading the input through
the same buffer); pass j warps the image again and mixes that into the
mean as blend j does, at its weight 1 / (j + 1) (its exact float32, from
its function), switched off past Samples. The passes are Samples - 1 (a
named parameter of the loop's subgraph, which the iteration count reads),
up to the 31 the template has. Each loop is a subgraph of its own, which
merging, nesting and fusing take whole.

A chain the author repeated by hand (Biomech's reaction-diffusion: 16
periods of a threshold, a Blur HQ and a Highpass, each reading the last)
becomes one iterate too (``chains``, ``Chains``): one period of the
finished translation, its buffers made the iterate's states. A period's
buffers depend on one another, so one pass of the iterate is one phase of
a period, the phases cycling (``$iteration``): in each phase the states
whose buffers are rendered then take their new values, the others keep
theirs. The pixels are those of the unrolled translation: the same
shaders, reading the same values from textures of the same format.
"""

import copy
import json
import re
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from sbsar2ptex.function import decode
from sbsar2ptex.libparams import evaluate32
from sbsar2ptex.libparams import slots
from sbsar2ptex.recognize import Match
from sbsar2ptex.sbsasm import Node
from sbsar2ptex.translate import GRAPH_PARAM
from sbsar2ptex.translate import INSTANCE_PARAM
from sbsar2ptex.translate import STORE_STEPS
from sbsar2ptex.translate import GraphInput
from sbsar2ptex.translate import Sub
from sbsar2ptex.translate import buffer
from sbsar2ptex.translate import input_param
from sbsar2ptex.translate import node_shader
from sbsar2ptex.translate import num
from sbsar2ptex.translate import output_type
from sbsar2ptex.translate import shader
from sbsar2ptex.translate import stored
from sbsar2ptex.translate import subgraph
from sbsar2ptex.translate import wire

FEATURES = ("iterate_buffer_v2", "iteration_variable")
"""What the fork must have for the loops (its "requires" features)."""
SLOPE_BLURS = ("pkg://slope_blur_grayscale", "pkg://slope_blur_grayscale_2")
STEPS = 32
"""Slope blur's warps: the most passes its template has."""
WARP, BLEND = 14, 2
"""Grayscale warp and blend ops."""
SWITCH = 7
PROBES = ((10.0, 8), (0.37, 32), (1.15, 3))
"""(Intensity, Samples) the warps' functions are compared at."""
FLOAT32, FLOAT16 = 1, 0
"""iterate_buffer_v2 state formats."""


@dataclass
class SlopeBlur:
    """A Slope blur instance, made one iterate (``make``).

    ``warps`` and ``blends`` are its graph nodes, in order; ``templates``
    their template nodes; ``weights`` the blends' (blend i's at i - 1);
    ``samples`` its Samples knob.
    """

    warps: list[int]
    blends: list[int]
    templates: dict[int, Node]
    weights: list[float]
    samples: GraphInput
    inputs: dict[str, int]
    """The graph nodes it reads: "source", "gradient"."""
    made: list[str] = field(default_factory=list)

    @property
    def members(self) -> set[int]:
        """Return the nodes the iterate computes."""
        return {*self.warps, *self.blends}

    @property
    def end(self) -> int:
        """Return the node whose image is the iterate's."""
        return self.blends[-1]

    def make(self, name: str, nodes: list[Node], size: int) -> Sub:
        """Return the iterate, in its subgraph ``name``, and the buffers it reads.

        ``size`` is its images' (log2). Called with the instance's knobs as
        the translation's inputs (translate_graph's ``library``).
        """
        loop, mean = f"{name}_loop", f"{name}_mean"
        first, last = nodes[self.warps[0]], nodes[self.end]
        samples = input_param(self.samples)

        def warp(shader_name: str) -> tuple[dict[str, Any], bool]:
            made = node_shader(first, shader_name, size, self.templates[first.index])
            out = next(n for n in made.nodes if n["name"] == shader_name)
            rounded = stored(Sub([out], [], [], shader_name), first)
            # Samples is the loop subgraph's named parameter, which the
            # iteration count reads: the shaders read that one, not their own.
            model = out["shader_model"]
            model["parameters"] = [
                p for p in model["parameters"] if p["name"] != samples
            ]
            out["parameters"].pop(samples, None)
            return out, rounded

        # The first warp reads the input as the unrolled one does (through
        # the same buffer, so the same filtering): it starts both states.
        start, _ = warp(f"{name}_warp0")
        step, rounded = warp(f"{name}_warp")
        # Pass j makes warp j + 1 and blend j, switching as _blend does (the
        # same arithmetic, so the same pixels), at its weight, off past
        # Samples. Precise: no fused multiply-add, as the blend compiles to
        # alone (a half weight lands on 16-bit ties, which an ulp flips).
        table = ", ".join(num(w) for w in [0.0, *self.weights])
        gate = f"clamp(float(int(${samples})) - float(j), 0.0, 1.0)"
        mean_shader = shader(
            mean,
            [("mean", "f"), ("warp", "f")],
            "f",
            "\n".join(
                [
                    "vec4 s = vec4($warp(uv)); vec4 d = vec4($mean(uv));",
                    "int j = int($iteration);",
                    f"const float w[{STEPS}] = float[{STEPS}]({table});",
                    f"float o = w[clamp(j, 0, {STEPS - 1})] * {gate};",
                    "vec4 f = s;",
                    "precise float r = d.x + (f.x - d.x) * o;",
                    "return clamp(r, 0.0, 1.0);",
                ]
            ),
        )
        stored(Sub([mean_shader], [], [], mean), last)
        iterate = {
            "name": loop,
            "type": "iterate_buffer_v2",
            "node_position": {"x": 0, "y": 0},
            "parameters": {
                "states": 2,
                "size_mode": 0,
                "size": size,
                "size_offset": 0,
                "iterations": f"clamp(${samples}, 1, {STEPS}) - 1",
                "filter": True,
                "rounding": 0,
                "format_0": FLOAT32 if rounded else FLOAT16,
                "rg_0": False,
                "format_1": FLOAT32,
                "rg_1": False,
            },
            "seed_int": 0,
        }
        w0, w = start["name"], step["name"]
        inner_wires = [
            wire(w0, loop, 0),
            wire(w0, loop, 1),
            {"from": loop, "from_port": 2, "to": w, "to_port": 0},
            {"from": loop, "from_port": 3, "to": mean, "to_port": 0},
            wire(w, mean, 1),
            wire(w, loop, 2),
            wire(mean, loop, 3),
        ]
        graph = subgraph(
            name,
            "Slope blur (iterated)",
            [iterate, start, step, mean_shader],
            inner_wires,
            [("f", w0, 0), ("f", w0, 1)],
            (loop, 1, "f"),
        )
        graph["connections"].append(
            {"from": "gen_inputs", "from_port": 1, "to": w, "to_port": 1}
        )
        # Never an automatic buffer in the loop's body: one there is rendered
        # again on every pass, which starts the loop over.
        for c in graph["connections"]:
            if c["to"] in (w0, w, mean):
                c["buffer"] = "never"
        default = float(self.samples.default[0])
        controls = next(n for n in graph["nodes"] if n["name"] == "gen_parameters")
        controls["widgets"].append(
            {
                "name": samples,
                "label": self.samples.label or self.samples.identifier,
                "type": "named_parameter",
                "min": 0.0,
                "max": float(max(STEPS, default)),
                "step": 1.0,
                "default": default,
            }
        )
        controls["parameters"][samples] = default
        graph["parameters"][samples] = default
        # the warps' buffers, as the unrolled ones have
        source, gradient = f"{name}_input_buf", f"{name}_gradient_buf"
        out = Sub(
            [graph, buffer(source, size), buffer(gradient, size, precise=True)],
            [wire(source, name, 0), wire(gradient, name, 1)],
            [("source", source, 0), ("gradient", gradient, 0)],
            name,
        )
        self.made.append(name)
        return out

    def rounds(self, nodes: list[Node]) -> bool:
        """Return whether the iterate's image is stored in 8 or 16 bits (stored)."""
        return (nodes[self.end].flags >> 4 & 3) in STORE_STEPS


def _function(node: Node, slot: str) -> Any:
    """Return ``node``'s function for setting ``slot``; None if it has none."""
    state, payload, _ = slots(node)[slot]
    if state != "func":
        return None
    return decode(node.at(payload, (node.end - payload) // 4))


def _chain(m: Match) -> tuple[list[int], list[int], list[float]] | None:  # noqa: PLR0911
    """Return a Slope blur template's warps, blends and mean weights, in order.

    None unless it is 32 warps, each of the last, all alike and reading one
    gradient, and 31 switch blends, blend i mixing warp i + 1 into the one
    before at c_i clamp(Samples - i, 0, 1) (c_i its weight).
    """
    t = m.template
    members = set(t.members)
    warps = [k for k in members if t.nodes[k].op == WARP]
    blends = [k for k in members if t.nodes[k].op == BLEND]
    starts = [k for k in warps if t.nodes[k].inputs.get("input") not in members]
    if len(starts) != 1:
        return None
    after = {t.nodes[k].inputs.get("input"): k for k in warps}
    chain = starts
    while chain[-1] in after and len(chain) <= STEPS:
        chain.append(after[chain[-1]])
    if len(chain) != STEPS or set(chain) != set(warps):
        return None
    if len({t.nodes[k].inputs.get("gradient") for k in chain} - members) != 1:
        return None
    ids = {uid: gi.identifier for uid, gi in t.inputs.items()}

    def at(intensity: float, samples: int) -> dict[int, list]:
        return {
            uid: [intensity if name == "Intensity" else samples]
            for uid, name in ids.items()
            if name in ("Intensity", "Samples")
        }

    try:
        fns = [_function(t.nodes[k], "intensity") for k in chain]
        if any(f is None for f in fns):
            return None
        for probe in PROBES:
            if len({tuple(evaluate32(f, at(*probe))) for f in fns}) != 1:
                return None
        mean, weights, dest = [], [], chain[0]
        for i, w in enumerate(chain[1:], 1):
            b = next(
                (
                    k
                    for k in blends
                    if t.nodes[k].inputs == {"destination": dest, "source": w}
                ),
                None,
            )
            if b is None or slots(t.nodes[b])["blend_mode"][:2] != ("const", [SWITCH]):
                return None
            opacity = _function(t.nodes[b], "opacity")
            if opacity is None:
                return None
            c = evaluate32(opacity, at(1.0, STEPS))
            if (
                evaluate32(opacity, at(1.0, i)) != [0.0]
                or evaluate32(opacity, at(1.0, i + 1)) != c
            ):
                return None
            mean.append(b)
            weights.append(float(c[0]))
            dest = b
    except (ValueError, IndexError, KeyError):
        return None
    if set(mean) != set(blends) - {t.root}:
        return None
    return chain, mean, weights


def slope_blurs(
    found: list[Match],
    nodes: list[Node],
    library: dict[int, tuple[Node, dict[int, GraphInput]]],
) -> dict[int, SlopeBlur]:
    """Return the Slope blurs to make iterates, by their last node.

    ``found`` are the graph's library node instances, ``library`` the
    translation's (procedural.library_nodes): only solved instances, their
    knobs live, whose chain no node outside it reads but at its end.
    """
    readers: dict[int, set[int]] = {}
    for n in nodes:
        for k in n.inputs.values():
            readers.setdefault(k, set()).add(n.index)
    out = {}
    for m in found:
        if m.template.graph not in SLOPE_BLURS:
            continue
        chain = _chain(m)
        if chain is None:
            continue
        warps_t, blends_t, weights = chain
        if not all(k in m.members for k in (*warps_t, *blends_t)):
            continue
        warps = [m.members[k] for k in warps_t]
        blends = [m.members[k] for k in blends_t]
        if not all(k in library for k in (*warps, *blends)):
            continue
        inside = {*warps, *blends}
        if any(readers.get(k, set()) - inside for k in inside - {blends[-1]}):
            continue
        if (
            len({nodes[k].flags >> 4 & 3 for k in warps}) != 1
            or len({nodes[k].flags >> 4 & 3 for k in blends}) != 1
        ):
            continue
        knobs = library[blends[-1]][1]
        samples = next(
            (gi for gi in knobs.values() if gi.identifier == "Samples" and gi.scope),
            None,
        )
        if samples is None:
            continue
        first = nodes[warps[0]]
        out[blends[-1]] = SlopeBlur(
            warps,
            blends,
            {g: m.template.nodes[t] for t, g in m.members.items()},
            weights,
            samples,
            {"source": first.inputs["input"], "gradient": first.inputs["gradient"]},
        )
    return out


MIN_PERIODS = 3
"""The fewest periods a chain is made an iterate from."""
LONGEST = 64
"""The longest period (nodes) chains are looked for with."""
MAX_STATES = 4
"""iterate_buffer_v2's most states."""
NODE_NAME = re.compile(r"(?<![A-Za-z0-9])n(\d+)(?![0-9])")
"""A translated node's name (or a library instance's scope): n and its index."""
HELPER = re.compile(r"(?<![A-Za-z0-9_])f(\d+)_")
"""A function's GLSL helper (translate.dynamic): numbered as translated."""


def _helpers_renumbered(text: str) -> str:
    """Return ``text`` with its function helpers numbered in order of appearance."""
    order: dict[str, int] = {}
    return HELPER.sub(lambda m: f"f@{order.setdefault(m[1], len(order))}_", text)


@dataclass
class Chain:
    """Periods of alike nodes, each reading the one before's end (``chains``).

    ``periods`` are their nodes, in order, each ending in the node the next
    reads; ``source`` is the node the first reads in its place.
    """

    periods: list[list[int]]
    source: int

    @property
    def end(self) -> int:
        """Return the last period's end: the chain's image."""
        return self.periods[-1][-1]


def _shape(node: Node) -> tuple:
    return (node.op, node.flags, tuple(sorted(node.inputs)))


def _match(nodes: list[Node], a: int, p: int) -> int | None:
    """Compare the period of ``p`` nodes from ``a`` with the next.

    The next one's nodes must be like this one's (op, flags, inputs by
    name), each reading nodes of its own period at the same offsets or this
    period's end ("carried"). Returns the one node this period reads where
    the next reads this one's end (before it), None if they differ.
    """
    b = a + p
    carried: set[int] = set()
    for t in range(p):
        x, y = nodes[a + t], nodes[b + t]
        if _shape(x) != _shape(y):
            return None
        for name, sy in y.inputs.items():
            sx = x.inputs[name]
            if b <= sy < b + p:
                if sx != sy - p:
                    return None
            elif sy == b - 1:
                carried.add(sx)
            else:
                return None
    if len(carried) != 1:
        return None
    (source,) = carried
    return source if source < a else None


def _chain_at(nodes: list[Node], end: int, p: int) -> tuple[list[int], int] | None:
    """Return the starts of ``p``-node periods ending at ``end``, and their source."""
    starts = [end - p + 1]
    if starts[0] < 1:
        return None
    source = None
    while starts[0] >= p:
        a = starts[0] - p
        got = _match(nodes, a, p)
        if got is None:
            break
        starts.insert(0, a)
        if got != a - 1:  # what the first reads is not a period of the chain
            source = got
            break
    if len(starts) < 2:  # noqa: PLR2004
        return None
    return starts, starts[0] - 1 if source is None else source


def chains(
    nodes: list[Node], ends: set[int] | None = None, minimum: int = MIN_PERIODS
) -> list[Chain]:
    """Return the graph's chains of at least ``minimum`` alike periods.

    Found from their last node down, the shortest period first. ``ends``
    are nodes read from outside the graph (its outputs). Only a period's
    end may be read outside it, by the next period: a period with another
    node read elsewhere is left out (the chain split around it), and an end
    read elsewhere ends a chain (the rest another).
    """
    readers: dict[int, set[int]] = {}
    for n in nodes:
        for k in n.inputs.values():
            readers.setdefault(k, set()).add(n.index)
    outside = ends or set()
    taken: set[int] = set()
    found: list[Chain] = []
    for end in range(len(nodes) - 1, 0, -1):
        if end in taken:
            continue
        got = None
        for p in range(2, LONGEST + 1):
            got = _chain_at(nodes, end, p)
            if got is not None and len(got[0]) >= minimum:
                break
        else:
            continue
        assert got is not None  # noqa: S101 (the loop broke on it)
        starts, source = got
        periods = [list(range(s, s + p)) for s in starts]
        taken.update(k for period in periods for k in period)
        run: list[list[int]] = []
        for j, period in enumerate(periods):
            inner = period[:-1]
            if any(readers.get(k, set()) - set(period) or k in outside for k in inner):
                if len(run) >= minimum:
                    found.append(Chain(run, source))
                run, source = [], period[-1]
                continue
            run.append(period)
            last = period[-1]
            after = set(periods[j + 1]) if j + 1 < len(periods) else None
            if after is not None and (
                readers.get(last, set()) - after or last in outside
            ):
                if len(run) >= minimum:
                    found.append(Chain(run, source))
                run, source = [], last
        if len(run) >= minimum:
            found.append(Chain(run, source))
    return sorted(found, key=lambda c: c.end)


class UnfoldableError(Exception):
    """A chain left unrolled, and why."""


@dataclass
class Chains:
    """Chains made iterates: translate_graph's ``fold``.

    ``found`` are the graph's (``chains``). Of each, the longest run of
    periods translated alike (``minimum`` at least) is made one iterate
    from one of them as translated (its buffers the iterate's states); the
    periods before and after it stay unrolled. ``made`` are the iterates'
    subgraphs, with the periods (indices in their chain) each runs;
    ``left`` the chains left unrolled (by their ends) and why.
    """

    found: list[Chain]
    minimum: int = MIN_PERIODS
    made: dict[str, range] = field(default_factory=dict)
    left: dict[int, str] = field(default_factory=dict)

    def __call__(
        self,
        mm: list[dict[str, Any]],
        wires: list[dict[str, Any]],
        origins: dict[str, int],
        out_of: dict[int, str],
        exposed: dict[str, list[str]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return ``mm`` and ``wires`` with the chains folded (translate.Fold)."""
        for chain in self.found:
            try:
                mm, wires, name, run = _fold(
                    chain, mm, wires, origins, out_of, exposed, self.minimum
                )
            except UnfoldableError as e:
                self.left[chain.end] = str(e)
            else:
                self.made[name] = run
        return mm, wires


def _has_buffer(graph: dict[str, Any]) -> bool:
    return any(
        n["type"] == "buffer" or (n["type"] == "graph" and _has_buffer(n))
        for n in graph["nodes"]
    )


def _inline(
    g: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    wires: list[dict[str, Any]],
    controls: dict[str, dict[str, Any]],
) -> dict[str, tuple[str, int]]:
    """Put subgraph ``g``'s nodes in its place in ``nodes`` and ``wires``.

    Its nodes are named after it; its knobs become ``controls`` (name ->
    label, value, linked widgets), but for the graph's and instances'
    parameters, which its nodes declare. Returns where its outputs now come
    from, by port.
    """
    gname = g["name"]
    del nodes[gname]
    inner = {n["name"]: n for n in g["nodes"] if n["type"] not in ("ios", "remote")}
    renamed = {k: f"{gname}_{k}" for k in inner}
    for k, n in inner.items():
        nodes[renamed[k]] = {**copy.deepcopy(n), "name": renamed[k]}
    feeding: dict[int, list[tuple[str, int]]] = {}
    for w in wires:
        if w["to"] == gname:
            feeding.setdefault(w["to_port"], []).append((w["from"], w["from_port"]))
    rest = [w for w in wires if gname not in (w["to"], w["from"])]
    outputs: dict[int, tuple[str, int]] = {}
    for w in g["connections"]:
        f, t = w["from"], w["to"]
        sources = (
            feeding.get(w["from_port"], [])
            if f == "gen_inputs"
            else [(renamed[f], w["from_port"])]
        )
        if t == "gen_outputs":
            if len(sources) != 1:
                msg = f"subgraph {gname} has an unconnected output"
                raise UnfoldableError(msg)
            outputs[w["to_port"]] = sources[0]
            continue
        rest += [
            {"from": s, "from_port": sp, "to": renamed[t], "to_port": w["to_port"]}
            for s, sp in sources
        ]
    for w in wires:
        if w["from"] == gname:
            s, sp = outputs[w["from_port"]]
            rest.append({**w, "from": s, "from_port": sp})
    wires[:] = rest
    remote = next((n for n in g["nodes"] if n["type"] == "remote"), None)
    for wd in remote["widgets"] if remote else []:
        if wd["type"] != "linked_control":
            msg = f"subgraph {gname} has a {wd['type']}"
            raise UnfoldableError(msg)
        if wd["name"].startswith((GRAPH_PARAM, INSTANCE_PARAM)):
            continue  # its nodes declare it: the loop's control
        controls[f"{gname}_{wd['name']}"] = {
            "label": wd["label"],
            "value": g["parameters"].get(wd["name"]),
            "links": [
                {"node": renamed[x["node"]], "widget": x["widget"]}
                for x in wd["linked_widgets"]
            ],
        }
    return outputs


def _fold(
    chain: Chain,
    mm: list[dict[str, Any]],
    wires: list[dict[str, Any]],
    origins: dict[str, int],
    out_of: dict[int, str],
    exposed: dict[str, list[str]],
    minimum: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, range]:
    """Return ``mm`` and ``wires`` with ``chain`` one iterate, its name and periods.

    Raises UnfoldableError, changing nothing, for a chain without
    ``minimum`` periods in a row translated alike, read elsewhere than at
    their end, or that do not fit an iterate.
    """
    by_name = {n["name"]: n for n in mm}
    into: dict[str, list[dict[str, Any]]] = {}
    out_wires: dict[str, list[dict[str, Any]]] = {}
    for w in wires:
        into.setdefault(w["to"], []).append(w)
        out_wires.setdefault(w["from"], []).append(w)
    periods = chain.periods
    period_of = {k: i for i, period in enumerate(periods) for k in period}
    if any(p[-1] not in out_of for p in periods) or chain.source not in out_of:
        msg = "not translated"
        raise UnfoldableError(msg)
    ends = [out_of[p[-1]] for p in periods]
    # what each period reads in the place of the one before's end
    entries = [out_of[chain.source], *ends[:-1]]

    def copies(name: str, source: str) -> bool:
        """Whether ``name`` is a buffer of ``source``'s image."""
        return by_name.get(name, {}).get("type") == "buffer" and [
            (w["from"], w["from_port"]) for w in into.get(name, [])
        ] == [(source, 0)]

    cores: list[set[str]] = [set() for _ in periods]
    for name, k in origins.items():
        i = period_of.get(k)
        if i is not None and name in by_name and not copies(name, ends[i]):
            cores[i].add(name)

    def renamer(i: int) -> Any:
        start = periods[i][0]

        def rename(text: str) -> str:
            return NODE_NAME.sub(
                lambda m: (
                    f"n@{int(m[1]) - start}" if period_of.get(int(m[1])) == i else m[0]
                ),
                text,
            )

        return rename

    def entry_key(name: str) -> str:
        params = json.dumps(by_name[name]["parameters"], sort_keys=True)
        return f"<entry buffer {params}>"

    def canon(i: int) -> str | None:
        """Return period ``i`` as text, its own names relative; None if open.

        Open: a node of it read outside it (but its end), or reading outside
        it (but the one before's end, or a buffer of that).
        """
        core, rename = cores[i], renamer(i)
        body, links = [], []
        for name in core:
            node = {k: v for k, v in by_name[name].items() if k != "node_position"}
            body.append(_helpers_renumbered(rename(json.dumps(node, sort_keys=True))))
            for w in into.get(name, []):
                f = w["from"]
                if f in core:
                    src = rename(f)
                elif f == entries[i]:
                    src = "<entry>"
                elif copies(f, entries[i]):
                    src = entry_key(f)
                else:
                    return None
                links.append(f"{src}:{w['from_port']}>{rename(name)}:{w['to_port']}")
            if name != ends[i] and any(
                w["to"] not in core for w in out_wires.get(name, [])
            ):
                return None
        return json.dumps([sorted(body), sorted(links)])

    # The longest run of periods translated alike (the latest of equals).
    texts = [canon(i) if ends[i] in cores[i] else None for i in range(len(periods))]
    first, last = 0, -1
    start = 0
    for i, text in enumerate(texts):
        if text is None or text != texts[start]:
            start = i
        if text is not None and i - start >= last - first:
            first, last = start, i
    count = last - first + 1
    if count < minimum:
        msg = f"at most {count} of its periods are translated alike"
        raise UnfoldableError(msg)

    # What goes: the periods, and buffers of one's end the next reads.
    removed: set[str] = set().union(*cores[first : last + 1])
    for i in range(first, last):
        removed |= {
            w["to"] for w in out_wires.get(ends[i], []) if copies(w["to"], ends[i])
        }
    # The first period's buffers of what it reads (then: the iterate's).
    removed -= {n for n in cores[first] if copies(n, entries[first])}
    feeds: list[str] = []
    for w in wires:
        f_in, t_in = w["from"] in removed, w["to"] in removed
        if t_in and not f_in:
            if not copies(w["from"], entries[first]):
                msg = f"it reads {w['from']}"
                raise UnfoldableError(msg)
            if w["from"] not in feeds:
                feeds.append(w["from"])
        elif f_in and not t_in and w["from"] != ends[last]:
            msg = f"{w['from']} is read outside it"
            raise UnfoldableError(msg)

    # The last period, its buffers states.
    core = cores[last]
    nodes = {n: copy.deepcopy(by_name[n]) for n in sorted(core)}
    tw: list[dict[str, Any]] = []
    carried: list[str] = []
    for name in sorted(core):
        for w in into.get(name, []):
            f = w["from"]
            if f in core:
                tw.append(dict(w))
            elif f == entries[last]:
                if by_name[name]["type"] != "buffer":
                    msg = "a period reads the one before's image unbuffered"
                    raise UnfoldableError(msg)
                if name not in carried:
                    carried.append(name)
            else:  # a buffer of the one before's image
                if f not in nodes:
                    nodes[f] = copy.deepcopy(by_name[f])
                    carried.append(f)
                tw.append(dict(w))
    first_rename, last_rename = renamer(first), renamer(last)

    def key_of(name: str, i: int, rename: Any) -> str:
        return rename(name) if name in cores[i] else entry_key(name)

    feed_of = {}
    for c in carried:
        f = next(
            (
                f
                for f in feeds
                if key_of(f, first, first_rename) == key_of(c, last, last_rename)
            ),
            None,
        )
        if f is None:
            msg = f"no first buffer like {c}"
            raise UnfoldableError(msg)
        feed_of[c] = f
    controls: dict[str, dict[str, Any]] = {}
    image = (ends[last], 0)
    while True:
        g = next(
            (n for n in nodes.values() if n["type"] == "graph" and _has_buffer(n)), None
        )
        if g is None:
            break
        moved = _inline(g, nodes, tw, controls)
        if image[0] == g["name"]:
            image = moved[image[1]]
    kinds = {n["type"] for n in nodes.values()}
    if kinds - {"shader", "buffer", "graph"}:
        msg = f"its period has {sorted(kinds - {'shader', 'buffer', 'graph'})}"
        raise UnfoldableError(msg)
    states_of = [n for n, node in nodes.items() if node["type"] == "buffer"]
    looks = {
        (b["parameters"]["size"], b["parameters"]["filter"], b["parameters"]["mipmap"])
        for b in (nodes[n] for n in states_of)
    }
    if len(looks) != 1:
        msg = "its buffers differ in size or filtering"
        raise UnfoldableError(msg)
    ((size, filtering, _),) = looks
    feeding: dict[str, list[dict[str, Any]]] = {}
    for w in tw:
        feeding.setdefault(w["to"], []).append(w)
    source: dict[str, tuple[str, int]] = dict.fromkeys(carried, image)
    for b in states_of:
        if b not in carried:
            ws = feeding.get(b, [])
            if len(ws) != 1:
                msg = f"buffer {b} has {len(ws)} inputs"
                raise UnfoldableError(msg)
            source[b] = (ws[0]["from"], ws[0]["from_port"])

    memo: dict[str, frozenset[str]] = {}

    def cone(name: str) -> frozenset[str]:
        """Return the buffers ``name``'s image reads (it and its inputs inline)."""
        if name in source:
            return frozenset([name])
        if name not in memo:
            got: set[str] = set()
            for w in feeding.get(name, []):
                if w["from"] not in nodes:
                    msg = f"{name} reads {w['from']}"
                    raise UnfoldableError(msg)
                got |= cone(w["from"])
            memo[name] = frozenset(got)
        return memo[name]

    depth: dict[str, int] = dict.fromkeys(carried, 0)

    def deep(b: str) -> int:
        if b not in depth:
            depth[b] = 1 + max((deep(x) for x in cone(source[b][0])), default=0)
        return depth[b]

    phases = 1 + max((deep(x) for x in cone(image[0])), default=0)
    writes = sorted(
        [(deep(b) - 1, b) for b in states_of if b not in carried]
        + [(phases - 1, c) for c in carried]
    )
    if any(p >= phases for p, _ in writes):
        msg = "a buffer the period's image does not read"
        raise UnfoldableError(msg)
    reads: dict[str, list[int]] = {b: [] for b in states_of}
    for p, b in writes:
        for x in cone(source[b][0]):
            reads[x].append(p)

    def blocked(p: int, b: str) -> set[int]:
        """Return the phases ``b``'s state is not written in: it is still read."""
        if not reads[b]:
            return set()
        if b in carried:  # from the period before
            return set(range(max(reads[b])))
        return set(range(p + 1, max(reads[b])))

    def kind(name: str) -> str:
        node = nodes[name]
        if node["type"] == "buffer":
            return kind(source[name][0])
        return output_type(node)

    states: list[dict[str, Any]] = []
    where: dict[str, int] = {}
    for p, b in writes:
        fmt = FLOAT32 if nodes[b]["parameters"]["f32"] else FLOAT16
        key, mine = (fmt, kind(source[b][0])), blocked(p, b)
        k = next(
            (
                k
                for k, st in enumerate(states)
                if st["key"] == key
                and all(
                    p != q and p not in theirs and q not in mine
                    for q, _, theirs in st["values"]
                )
            ),
            len(states),
        )
        if k == len(states):
            states.append({"key": key, "values": []})
        states[k]["values"].append((p, b, mine))
        where[b] = k
    if len(states) > MAX_STATES:
        msg = f"its period needs {len(states)} states"
        raise UnfoldableError(msg)
    exact = [c for c in carried if states[where[c]]["key"][0] == FLOAT32]
    if not exact:
        msg = "its image is kept in 16 bits"
        raise UnfoldableError(msg)

    # The iterate's subgraph.
    n = len(states)
    loop = "loop"
    inner = [node for node in nodes.values() if node["type"] != "buffer"]
    links: list[dict[str, Any]] = []
    for w in tw:
        if w["to"] in source:
            continue  # a buffer's input: its state's, below
        if w["from"] in source:
            links.append({**w, "from": loop, "from_port": n + where[w["from"]]})
        else:
            links.append(dict(w))

    def read(src: tuple[str, int]) -> tuple[str, int]:
        return (loop, n + where[src[0]]) if src[0] in source else src

    feed_ports: list[str] = []
    for k, st in enumerate(states):
        fmt, t = st["key"]
        values = sorted(st["values"])
        lines = [f"int p = (int($iteration) - 1) % {phases};"]
        lines += [f"if (p == {q}) return $p{q}(uv);" for q, _, _ in values]
        lines.append("return $current(uv);")
        name = f"state{k}"
        inputs = [("current", t)] + [(f"p{q}", t) for q, _, _ in values]
        inner.append(shader(name, inputs, t, "\n".join(lines)))
        links.append({"from": loop, "from_port": n + k, "to": name, "to_port": 0})
        links.append({"from": name, "from_port": 0, "to": loop, "to_port": n + k})
        for j, (_, b, _) in enumerate(values):
            s, sp = read(source[b])
            links.append({"from": s, "from_port": sp, "to": name, "to_port": 1 + j})
        held = next((b for _, b, _ in values if b in carried), None)
        init = f"init{k}"
        if held is None:  # read only after it is written
            zero = "vec4(0.0)" if t == "rgba" else "0.0"
            inner.append(shader(init, [], t, f"return {zero};"))
        else:
            inner.append(shader(init, [("input", t)], t, "return $input(uv);"))
            feed = feed_of[held]
            if feed not in feed_ports:
                feed_ports.append(feed)
            links.append(
                {"from": "gen_inputs", "from_port": feed_ports.index(feed),
                 "to": init, "to_port": 0}
            )  # fmt: skip
        links.append({"from": init, "from_port": 0, "to": loop, "to_port": k})
    out_state = where[exact[0]]
    out_type = states[out_state]["key"][1]
    links.append(
        {"from": loop, "from_port": out_state, "to": "gen_outputs", "to_port": 0}
    )
    # Never an automatic buffer in the loop's body (one restarts the loop).
    for w in links:
        if w["to"] != "gen_outputs":
            w["buffer"] = "never"
    iterate = {
        "name": loop,
        "type": "iterate_buffer_v2",
        "node_position": {"x": 0, "y": 0},
        "parameters": {
            "states": n, "size_mode": 0, "size": size, "size_offset": 0,
            "iterations": f"{phases} * {count}", "filter": filtering, "rounding": 0,
            **{f"format_{k}": st["key"][0] for k, st in enumerate(states)},
            **{f"rg_{k}": False for k in range(n)},
        },
        "seed_int": 0,
    }  # fmt: skip
    users: dict[str, list[str]] = {}
    values: dict[str, Any] = {}
    for node in inner:
        if node["type"] == "shader":
            names = [p["name"] for p in node["shader_model"]["parameters"]]
        elif node["type"] == "graph":
            names = list(node["parameters"])
        else:
            continue
        for p in names:
            if p.startswith((GRAPH_PARAM, INSTANCE_PARAM)):
                users.setdefault(p, []).append(node["name"])
                values.setdefault(p, node["parameters"][p])
    labels = {
        p["name"]: p.get("label", p["name"])
        for node in inner
        if node["type"] == "shader"
        for p in node["shader_model"]["parameters"]
    }
    widgets = [
        {"name": p, "label": labels.get(p, p), "type": "linked_control",
         "linked_widgets": [{"node": u, "widget": p} for u in us]}
        for p, us in sorted(users.items())
    ] + [
        {"name": c, "label": d["label"], "type": "linked_control",
         "linked_widgets": d["links"]}
        for c, d in sorted(controls.items())
    ]  # fmt: skip
    values |= {c: d["value"] for c, d in controls.items()}
    io = {"node_position": {"x": 0, "y": 0}, "parameters": {}, "type": "ios"}
    inner = [iterate, *inner]
    for k, node in enumerate(inner):
        node["node_position"] = {"x": 220 * (k % 12), "y": 180 * (k // 12)}
    name = f"{ends[last]}_loop"
    in_types = [kind(next(c for c in carried if feed_of[c] == f)) for f in feed_ports]
    ports_in = [{"name": f"port{k}", "type": t} for k, t in enumerate(in_types)]
    ports_out = [{"name": "port0", "type": out_type}]
    graph = {
        "name": name,
        "label": f"Loop ({count} periods of {phases} passes)",
        "type": "graph",
        "node_position": {"x": 0, "y": 0},
        "parameters": values,
        "seed_int": 0,
        "nodes": [
            *inner,
            {"name": "gen_inputs", **io, "ports": ports_in},
            {"name": "gen_outputs", **io, "ports": ports_out},
            {"name": "gen_parameters", "type": "remote",
             "node_position": {"x": 0, "y": -200}, "parameters": dict(values),
             "seed_int": 0, "widgets": widgets},
        ],
        "connections": links,
    }  # fmt: skip

    # In the graph: it in the place of the chain's nodes.
    new_wires = []
    for w in wires:
        f_in, t_in = w["from"] in removed, w["to"] in removed
        if t_in:
            continue
        new_wires.append({**w, "from": name, "from_port": 0} if f_in else w)
    new_wires += [wire(f, name, k) for k, f in enumerate(feed_ports)]
    for r in removed:
        origins.pop(r, None)
    end = periods[last][-1]
    origins[name] = end
    for f in feed_ports:
        if f in cores[first]:  # a buffer of the first period's: now the loop's
            origins[f] = end
    out_of[end] = name
    for p in list(exposed):
        kept = [u for u in exposed[p] if u not in removed]
        if p in users:
            kept.append(name)
        if kept:
            exposed[p] = kept
        else:
            del exposed[p]
    mm = [n for n in mm if n["name"] not in removed] + [graph]
    return mm, new_wires, name, range(first, last + 1)
