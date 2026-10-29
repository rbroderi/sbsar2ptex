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
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from sbsar2ptex.function import decode
from sbsar2ptex.libparams import evaluate32
from sbsar2ptex.libparams import slots
from sbsar2ptex.recognize import Match
from sbsar2ptex.sbsasm import Node
from sbsar2ptex.translate import STORE_STEPS
from sbsar2ptex.translate import GraphInput
from sbsar2ptex.translate import Sub
from sbsar2ptex.translate import buffer
from sbsar2ptex.translate import input_param
from sbsar2ptex.translate import node_shader
from sbsar2ptex.translate import num
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
