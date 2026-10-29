"""Loops made iterate buffers (--experimental), on synthetic graphs.

Slope blur, and chains of alike periods. Their pixels are checked against
the unrolled translation, through the material-maker-mod fork, by
scripts/check_loops.py and scripts/check_chains.py; these check the
detection, the emitted nodes and their wiring.
"""

import json
import struct

import pytest

from sbsar2ptex.function import FLOAT
from sbsar2ptex.function import INT
from sbsar2ptex.libparams import knobs
from sbsar2ptex.loops import STEPS
from sbsar2ptex.loops import Chains
from sbsar2ptex.loops import SlopeBlur
from sbsar2ptex.loops import chains
from sbsar2ptex.loops import slope_blurs
from sbsar2ptex.recognize import Match
from sbsar2ptex.recognize import Template
from sbsar2ptex.sbsasm import BASE
from sbsar2ptex.sbsasm import Node
from sbsar2ptex.translate import GraphInput
from sbsar2ptex.translate import translate_graph

SAMPLES, INTENSITY = 0x5A3B1E01, 0x1A7E0C02
SOURCE, GRADIENT = 500, 501
"""The graph inputs' uids (image nodes)."""
WARPS = list(range(2, 2 + STEPS))
BLENDS = list(range(2 + STEPS, 1 + 2 * STEPS))
END, ROOT = BLENDS[-1], 1 + 2 * STEPS
"""The last blend (the instance's output) and the template's root blend,
which mixes the input back in when Intensity is 0 (cooked away here)."""


def word(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def code(*instrs: list[int]) -> list[int]:
    """Return a code block's u32 words: its count, then its instructions."""
    half = [len(instrs)]
    for ins in instrs:
        half += ins
    half += [0] * (len(half) % 2)
    return [half[k] | half[k + 1] << 16 for k in range(0, len(half), 2)]


def load(uid: int, kind: int) -> list[int]:
    return [3 << 10 | kind << 8 | 0x02, 0, uid & 0xFFFF, uid >> 16]


def const(x: float) -> list[int]:
    w = word(x)
    return [3 << 10 | FLOAT << 8 | 0x00, 0, w & 0xFFFF, w >> 16]


def op(o: int, *args: int) -> list[int]:
    return [len(args) << 10 | FLOAT << 8 | o, *args]


def with_code(k: int, opcode: int, flags: int, fields: list[int], block: list[int]):
    """Return node ``k`` whose one setting is the function ``block``."""
    start = 0x1000 + 0x400 * k
    words = [opcode, 0, 0, 0, *block]
    return Node(index=k, start=start, end=start + 4 * len(words), op=opcode,
                flags=flags, fields=fields, inline=[start + 16 - BASE],
                words=words)  # fmt: skip


def wiring(k: int) -> dict[str, int]:
    """Return node ``k``'s inputs: each warp of the last, the blends' mean."""
    if k in WARPS:
        return {"input": k - 1 if k > WARPS[0] else 0, "gradient": 1}
    if k == ROOT:
        return {"destination": END, "source": 0}
    i = k - BLENDS[0] + 1  # blend i mixes warp i + 1 into the mean so far
    return {"destination": k - 1 if i > 1 else WARPS[0], "source": WARPS[i]}


def template(fmt: int) -> Template:
    """Return Slope blur grayscale's template, as Designer's cooks."""
    warp = code(load(INTENSITY, FLOAT), load(SAMPLES, INT), op(0x11, 1),
                op(0x15, 0, 2))  # fmt: skip
    nodes = [Node(index=0, start=0, end=4, op=32), Node(index=1, start=4, end=8, op=32)]
    nodes += [with_code(k, 14, fmt << 4 | 2 << 13, [], warp) for k in WARPS]
    for i, k in enumerate(BLENDS, 1):
        gate = [load(SAMPLES, INT), op(0x11, 0), const(i), op(0x13, 1, 2),
                const(1.0), op(0x30, 3, 4), const(0.0), op(0x31, 5, 6)]  # fmt: skip
        block = code(*gate, const(f32(1 / (i + 1))), op(0x14, 8, 7))
        nodes.append(with_code(k, 2, fmt << 4, [7 | 2 << 4], block))
    nodes.append(with_code(ROOT, 2, fmt << 4, [7 | 2 << 4], code(const(0.0))))
    for n in nodes[2:]:
        n.inputs = wiring(n.index)
    t = Template(
        "slope_blur", "pkg://slope_blur_grayscale", "Slope_Blur", nodes, ROOT,
        members=[*WARPS, *BLENDS, ROOT], label="Slope blur grayscale",
    )  # fmt: skip
    t.inputs = {
        SAMPLES: GraphInput("Samples", 4, [8], "Samples"),
        INTENSITY: GraphInput("Intensity", 0, [10.0], "Intensity"),
    }
    return t


def instance(fmt: int = 1, samples: int = 32, intensity: float = 0.37):
    """Return a graph of one solved Slope blur, and what translate_graph needs.

    (graph nodes, the match, translate_graph's ``library``.)
    """
    t = template(fmt)
    graph = [Node(index=0, start=0, end=4, op=32, uid=SOURCE),
             Node(index=1, start=4, end=8, op=32, uid=GRADIENT)]  # fmt: skip
    for k in [*WARPS, *BLENDS]:
        n = t.nodes[k]
        graph.append(Node(index=k, start=n.start, end=n.end, op=n.op, flags=n.flags,
                          inputs=wiring(k)))  # fmt: skip
    m = Match(t, END, {k: k for k in [*WARPS, *BLENDS]}, {0: 0, 1: 1})
    reads = knobs(m, {SAMPLES: [samples], INTENSITY: [intensity]}, {})
    library = {k: (t.nodes[k], reads) for k in m.members.values()}
    return graph, m, library


def translated(fmt: int = 1, *, loops: bool = True) -> list[dict]:
    graph, m, library = instance(fmt)
    found = slope_blurs([m], graph, library) if loops else {}
    mm, _, outs = translate_graph(
        graph, {"out": END}, {SOURCE: "a.png", GRADIENT: "b.png"}, size=8,
        library=library, loops=found,
    )  # fmt: skip
    assert outs == {"out": (f"n{END}", 0)}
    return mm


def test_slope_blur_is_found_with_its_weights() -> None:
    graph, m, library = instance()
    found = slope_blurs([m], graph, library)
    assert list(found) == [END]
    loop = found[END]
    assert isinstance(loop, SlopeBlur)
    assert (loop.warps, loop.blends) == (WARPS, BLENDS)
    assert loop.inputs == {"source": 0, "gradient": 1}
    # the template's own float32 weights: 1 / (i + 1) for blend i
    assert loop.weights == [f32(1 / (i + 1)) for i in range(1, STEPS)]
    assert loop.weights[:3] == [0.5, f32(1 / 3), 0.25]
    assert (loop.samples.identifier, loop.samples.scope) == ("Samples", f"n{END}")


def test_a_chain_read_from_outside_stays_unrolled() -> None:
    graph, m, library = instance()
    graph.append(Node(index=len(graph), start=0, end=4, op=30,
                      inputs={"input1": WARPS[5]}))  # fmt: skip
    assert slope_blurs([m], graph, library) == {}


def test_an_unsolved_instance_stays_unrolled() -> None:
    graph, m, library = instance()
    del library[BLENDS[3]]  # its constants not given back by any knobs
    assert slope_blurs([m], graph, library) == {}


def test_other_library_nodes_are_not_loops() -> None:
    graph, m, library = instance()
    m.template.graph = "pkg://highpass_grayscale"
    assert slope_blurs([m], graph, library) == {}


@pytest.mark.parametrize(("fmt", "steps"), [(0, "255.0"), (1, "65535.0")])
def test_the_iterate_replaces_the_unrolled_nodes(fmt: int, steps: str) -> None:
    mm = translated(fmt)
    names = {n["name"] for n in mm}
    assert not names & {f"n{k}" for k in [*WARPS, *BLENDS[:-1]]}
    assert {f"n{END}_gradient_buf", f"n{END}_input_buf"} <= names
    by_name = {n["name"]: n for n in mm}
    assert by_name[f"n{END}_gradient_buf"]["parameters"]["f32"]
    sub = by_name[f"n{END}"]
    assert sub["type"] == "graph"
    inner = {n["name"]: n for n in sub["nodes"]}
    loop = inner[f"n{END}_loop"]
    assert loop["type"] == "iterate_buffer_v2"
    samples = f"i_n{END}_Samples"
    assert loop["parameters"] | {"iterations": None} == {
        "states": 2, "size_mode": 0, "size": 8, "size_offset": 0,
        "iterations": None, "filter": True, "rounding": 0,
        "format_0": 1, "rg_0": False, "format_1": 1, "rg_1": False,
    }  # fmt: skip
    # Samples - 1 passes: the first warp starts both states
    iterations = f"clamp(${samples}, 1, {STEPS}) - 1"
    assert loop["parameters"]["iterations"] == iterations
    w0, w, mean = (inner[f"n{END}_{k}"] for k in ("warp0", "warp", "mean"))
    # Samples is the subgraph's named parameter, which the count reads too
    controls = inner["gen_parameters"]
    named = [x for x in controls["widgets"] if x["type"] == "named_parameter"]
    assert [(x["name"], x["default"]) for x in named] == [(samples, 32.0)]
    assert sub["parameters"][samples] == 32.0
    for warp in (w0, w):
        assert samples not in warp["parameters"]
        assert samples not in [p["name"] for p in warp["shader_model"]["parameters"]]
        assert f"int(${samples})" in warp["shader_model"]["instance"]
    assert f"i_n{END}_Intensity" in sub["parameters"]
    # stored as the blends and warps are, in the graph's format
    for shader in (w0, w, mean):
        assert f"v * {steps}" in shader["shader_model"]["instance"]
    body = mean["shader_model"]["instance"]
    assert "int j = int($iteration);" in body
    assert "const float w[32] = float[32](0.0, 0.5, 0.33333334, 0.25" in body
    assert f"clamp(float(int(${samples})) - float(j), 0.0, 1.0)" in body
    links = {
        (x["from"], x["from_port"], x["to"], x["to_port"]) for x in sub["connections"]
    }
    lp, w0p, wp, mp = (f"n{END}_{k}" for k in ("loop", "warp0", "warp", "mean"))
    assert links == {
        ("gen_inputs", 0, w0p, 0), ("gen_inputs", 1, w0p, 1),  # the first warp
        (w0p, 0, lp, 0), (w0p, 0, lp, 1),  # starts both states
        (lp, 2, wp, 0), ("gen_inputs", 1, wp, 1),  # warp the image
        (lp, 3, mp, 0), (wp, 0, mp, 1),  # into the mean
        (wp, 0, lp, 2), (mp, 0, lp, 3),
        (lp, 1, "gen_outputs", 0),
    }  # fmt: skip
    assert all(
        x.get("buffer") == "never"
        for x in sub["connections"]
        if x["to"] in (w0p, wp, mp)
    )


def test_samples_is_live_in_the_unrolled_translation() -> None:
    mm = translated(loops=False)
    warps = [n for n in mm if n["name"] in {f"n{k}" for k in WARPS}]
    assert len(warps) == STEPS
    for n in warps:
        assert n["parameters"][f"i_n{END}_Samples"] == 32.0
        assert f"int($i_n{END}_Samples)" in n["shader_model"]["instance"]


def test_without_loops_the_translation_is_unchanged() -> None:
    graph, _, library = instance()
    args = (graph, {"out": END}, {SOURCE: "a.png", GRADIENT: "b.png"})
    plain = translate_graph(*args, size=8, library=library)
    none = translate_graph(*args, size=8, library=library, loops={})
    assert json.dumps(plain) == json.dumps(none)


def periodic(count: int, extra: dict[int, dict[str, int]] | None = None) -> list[Node]:
    """Return an image, then ``count`` periods of a directional blur and a blur.

    Each period's blur reads its directional blur, which reads the period
    before's blur (the first: the image). 8 bits, knobs inline. ``extra``
    adds inputs to nodes.
    """
    nodes = [Node(index=0, start=0, end=4, op=32, uid=SOURCE)]
    for k in range(count):
        d, b = 1 + 2 * k, 2 + 2 * k
        nodes.append(
            Node(index=d, start=4 * d, end=4 * d + 4, op=22, fields=[1 | 1 << 2],
                 inline=[word(5.86), word(0.0625)], inputs={"input": d - 1})
        )  # fmt: skip
        nodes.append(
            Node(index=b, start=4 * b, end=4 * b + 4, op=20, flags=1 << 12,
                 inline=[word(2.0)], inputs={"input": d})
        )  # fmt: skip
    for k, more in (extra or {}).items():
        nodes[k].inputs |= more
    return nodes


def test_a_chain_of_alike_periods_is_found() -> None:
    (chain,) = chains(periodic(5))
    assert chain.periods == [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    assert chain.source == 0
    assert chain.end == 10
    assert chains(periodic(2)) == []  # too few


def test_a_chain_read_inside_is_split() -> None:
    # node 6 (a period's blur, not the chain's end) read by an outside node
    nodes = [*periodic(9), Node(index=19, start=76, end=80, op=20, flags=1 << 12,
                                inline=[word(2.0)], inputs={"input": 6})]  # fmt: skip
    found = chains(nodes)
    assert [(c.periods[0][0], c.end, c.source) for c in found] == [
        (1, 6, 0),
        (7, 18, 6),
    ]
    # a period reading something else than the one before's end is no period
    assert chains(periodic(5, {5: {"other": 0}})) == []


def test_a_chain_becomes_one_iterate() -> None:
    nodes = periodic(5)
    images = {SOURCE: "a.png"}
    fold = Chains(chains(nodes))
    mm, wires, outs = translate_graph(nodes, {"out": 10}, images, size=8, fold=fold)
    # The first period reads the image through a 16-bit buffer, the others
    # the 8-bit blur before through a 32-bit one (exact): it stays unrolled.
    assert fold.made == {"n10_loop": range(1, 5)}
    assert outs == {"out": ("n10_loop", 0)}
    names = {n["name"] for n in mm}
    assert names == {"n0", "n1_input_buf", "n1", "n2", "n3_input_buf", "n10_loop"}
    assert {(w["from"], w["to"]) for w in wires} == {
        ("n0", "n1_input_buf"), ("n1_input_buf", "n1"), ("n1", "n2"),
        ("n2", "n3_input_buf"),  # the second period's buffer, the iterate's
        ("n3_input_buf", "n10_loop"),
    }  # fmt: skip
    sub = next(n for n in mm if n["name"] == "n10_loop")
    inner = {n["name"]: n for n in sub["nodes"]}
    loop = inner["loop"]["parameters"]
    # the blur's two 16-bit buffers one state, the 32-bit one another:
    # three phases a period
    assert loop == {
        "states": 2, "size_mode": 0, "size": 8, "size_offset": 0,
        "iterations": "3 * 4", "filter": True, "rounding": 0,
        "format_0": 0, "format_1": 1, "rg_0": False, "rg_1": False,
    }  # fmt: skip
    # the period's shaders, as translated, their knobs live: the blur's
    # passes inlined, its intensity one control
    assert {"n9", "n10_pass_x", "n10_pass_y"} <= set(inner)
    assert inner["n9"]["parameters"] == pytest.approx(
        {"intensity": 5.86, "angle": 0.0625}
    )
    assert "5.86" not in inner["n9"]["shader_model"]["instance"]
    controls = inner["gen_parameters"]["widgets"]
    assert [(c["name"], c["linked_widgets"]) for c in controls] == [
        (
            "n10_intensity",
            [
                {"node": "n10_pass_x", "widget": "intensity"},
                {"node": "n10_pass_y", "widget": "intensity"},
            ],
        )
    ]
    assert sub["parameters"] == {"n10_intensity": 2.0}
    body = inner["state1"]["shader_model"]["instance"]
    assert "int p = (int($iteration) - 1) % 3;" in body
    assert "if (p == 2) return $p2(uv);" in body
    assert "return $current(uv);" in body
    links = {
        (x["from"], x["from_port"], x["to"], x["to_port"]) for x in sub["connections"]
    }
    assert {
        ("gen_inputs", 0, "init1", 0), ("init1", 0, "loop", 1),  # the image
        ("init0", 0, "loop", 0),
        ("loop", 3, "n9", 0),  # the directional blur reads it
        ("n9", 0, "state0", 1),  # into the blur's first buffer (phase 0)
        ("loop", 2, "n10_pass_x", 0), ("n10_pass_x", 0, "state0", 2),  # 1
        ("loop", 2, "n10_pass_y", 0), ("n10_pass_y", 0, "state1", 1),  # 2
        ("loop", 2, "state0", 0), ("state0", 0, "loop", 2),
        ("loop", 3, "state1", 0), ("state1", 0, "loop", 3),
        ("loop", 1, "gen_outputs", 0),
    } == links  # fmt: skip
    assert all(
        x.get("buffer") == "never"
        for x in sub["connections"]
        if x["to"] != "gen_outputs"
    )


def test_a_chain_read_inside_stays_unrolled_there() -> None:
    nodes = periodic(5)
    images = {SOURCE: "a.png"}
    # node 6's image an output too: a chain ends there (the rest too short)
    fold = Chains(chains(nodes, {6, 10}))
    assert [c.periods for c in fold.found] == [[[1, 2], [3, 4], [5, 6]]]
    plain = translate_graph(nodes, {"a": 10, "b": 6}, images, size=8)
    folded = translate_graph(nodes, {"a": 10, "b": 6}, images, size=8, fold=fold)
    # its first period reads a 16-bit buffer, the others 32-bit: only two alike
    assert fold.left == {6: "at most 2 of its periods are translated alike"}
    assert json.dumps(plain) == json.dumps(folded)
