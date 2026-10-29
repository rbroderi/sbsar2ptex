"""Slope blur made an iterate buffer (--experimental), on a synthetic instance.

Its pixels are checked against the unrolled translation, through the
material-maker-mod fork, by scripts/check_loops.py; these check the
detection, the emitted nodes and their wiring.
"""

import json
import struct

import pytest

from sbsar2ptex.function import FLOAT
from sbsar2ptex.function import INT
from sbsar2ptex.libparams import knobs
from sbsar2ptex.loops import STEPS
from sbsar2ptex.loops import SlopeBlur
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
