"""The translator's Material Maker graphs, on the cooked probe fixtures.

Pixel accuracy is checked against sbsrender by scripts/check_filters.py
(needs Substance Designer and Material Maker); these check the wiring.
"""

import re
from pathlib import Path

import pytest

from sbsar2ptex.archive import member
from sbsar2ptex.archive import read_archive
from sbsar2ptex.sbsasm import Asm
from sbsar2ptex.sbsasm import Node
from sbsar2ptex.translate import UnsupportedError
from sbsar2ptex.translate import translate_graph

DATA = Path(__file__).parent / "data"


def translated(name: str):
    files = read_archive(DATA / f"{name}.sbsar")
    xml = member(files, ".xml").decode("utf-8")
    uids = {int(u) for u in re.findall(r'uid="(\d+)" identifier="input_', xml)}
    asm = Asm(member(files, ".sbsasm"))
    return translate_graph(
        asm.nodes(),
        {"output": asm.output_nodes[0][1]},
        {u: f"{u}.png" for u in uids},
    )


def test_uniform_color() -> None:
    nodes, wires, outs = translated("uniform_color")
    (node,) = nodes
    assert wires == []
    assert outs == {"output": (node["name"], 0)}
    assert "vec4(0.3, 0.4, 0.5, 1.0)" in node["shader_model"]["instance"]


def test_levels_wiring() -> None:
    nodes, wires, outs = translated("levels_gray")
    kinds = {n["name"]: n["type"] for n in nodes}
    assert sorted(kinds.values()) == ["image", "shader"]
    (wire,) = wires
    assert kinds[wire["from"]] == "image"
    assert wire["to"] == outs["output"][0]
    assert (
        "float lo = float(0.1)"
        in next(n for n in nodes if n["type"] == "shader")["shader_model"]["instance"]
    )


def test_blend_ports_and_opacity() -> None:
    nodes, wires, outs = translated("blend_gray")
    blend = outs["output"][0]
    by_port = {w["to_port"]: w["from"] for w in wires if w["to"] == blend}
    types = {n["name"]: n["type"] for n in nodes}
    assert set(by_port) == {0, 1, 2}  # source, destination, opacity
    assert types[by_port[1]] == "image"
    instance = next(n for n in nodes if n["name"] == blend)["shader_model"]["instance"]
    assert "0.37 * $opacity(uv)" in instance


def test_unsupported_op() -> None:
    odd = Node(index=0, start=0, end=4, op=99, words=[99])
    with pytest.raises(UnsupportedError, match="op 99"):
        translate_graph([odd], {"output": 0}, {})


def test_distance_jump_floods() -> None:
    nodes, _, outs = translated("distance_gray_chain")
    passes = [n for n in nodes if "_pass" in n["name"]]
    assert len(passes) >= 2  # halving steps, then one texel
    seeds = [n for n in nodes if n["type"] == "buffer" and "_jfa" in n["name"]]
    assert all(not b["parameters"]["filter"] for b in seeds)
    assert outs["output"][0] in {n["name"] for n in nodes}


def test_neighborhood_filters_sample_buffers() -> None:
    nodes, wires, outs = translated("warp_gray_chain")
    types = {n["name"]: n["type"] for n in nodes}
    warp = outs["output"][0]
    into_warp = {w["from"] for w in wires if w["to"] == warp}
    assert {types[n] for n in into_warp} == {"buffer"}  # input and gradient
    blur_passes = [
        n for n in nodes if n["type"] == "shader" and n["name"].endswith("_x")
    ]
    assert len(blur_passes) == 1  # the chained blur: x pass, then y
