"""The translator's Material Maker graphs, on the cooked probe fixtures.

Pixel accuracy is checked against sbsrender by scripts/check_filters.py
(needs Substance Designer and Material Maker); these check the wiring.
"""

import re
import struct
from pathlib import Path

import pytest

from sbsar2ptex.archive import member
from sbsar2ptex.archive import read_archive
from sbsar2ptex.sbsasm import Asm
from sbsar2ptex.sbsasm import Node
from sbsar2ptex.translate import UnsupportedError
from sbsar2ptex.translate import _distance
from sbsar2ptex.translate import _gradient_keys
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
    assert node["parameters"]["color"] == pytest.approx(
        {"r": 0.3, "g": 0.4, "b": 0.5, "a": 1.0}
    )


def test_levels_wiring() -> None:
    nodes, wires, outs = translated("levels_gray")
    kinds = {n["name"]: n["type"] for n in nodes}
    assert sorted(kinds.values()) == ["image", "shader"]
    (wire,) = wires
    assert kinds[wire["from"]] == "image"
    assert wire["to"] == outs["output"][0]
    levels = next(n for n in nodes if n["type"] == "shader")
    assert levels["parameters"]["in_low"] == pytest.approx(0.1)


def test_blend_ports_and_opacity() -> None:
    nodes, wires, outs = translated("blend_gray")
    blend = outs["output"][0]
    by_port = {w["to_port"]: w["from"] for w in wires if w["to"] == blend}
    types = {n["name"]: n["type"] for n in nodes}
    assert set(by_port) == {0, 1, 2}  # source, destination, opacity
    assert types[by_port[1]] == "image"
    blend_node = next(n for n in nodes if n["name"] == blend)
    assert blend_node["parameters"]["opacity_mult"] == pytest.approx(0.37)


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


def test_color_distance_preserves_source_rgb_outside_range() -> None:
    node = Node(
        index=0,
        start=0,
        end=4,
        op=43,
        fields=[7, 1, 0],
        inline=[struct.unpack("<I", struct.pack("<f", 12.0))[0]],
    )
    made = _distance(node, "distance", color=True, size=8)
    final = next(n for n in made.nodes if n["name"] == "distance")
    instance = final["shader_model"]["instance"]
    assert "$source(f > 0.0 ? c.xy : uv)" in instance


def test_gradient_ignores_stored_midpoints() -> None:
    words = [
        0,
        0xFF000000,
        0x2000FF00,
        0xFFFFFFFF,
        0xE000FFFF,
    ]
    node = Node(
        index=0,
        start=0,
        end=len(words) * 4,
        fields=[0, 2],
        code=[4],
        words=words,
    )
    assert _gradient_keys(node, color=True) == [
        (0.0, [0.0, 1.0, 0.0, 1.0]),
        (1.0, [1.0, 1.0, 1.0, 1.0]),
    ]


def test_neighborhood_filters_sample_buffers() -> None:
    nodes, wires, outs = translated("warp_gray_chain")
    types = {n["name"]: n["type"] for n in nodes}
    warp = outs["output"][0]
    into_warp = {w["from"] for w in wires if w["to"] == warp}
    assert {types[n] for n in into_warp} == {"buffer"}  # input and gradient
    (blur,) = [n for n in nodes if n["type"] == "graph"]  # the chained blur
    inner = {n["name"]: n for n in blur["nodes"]}
    assert {"pass_x", "pass_y", "buf_x", "buf_y"} <= set(inner)
    # One intensity control, linked to both passes.
    (control,) = inner["gen_parameters"]["widgets"]
    assert control["name"] == "intensity"
    assert {w["node"] for w in control["linked_widgets"]} == {"pass_x", "pass_y"}
