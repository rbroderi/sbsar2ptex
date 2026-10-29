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
from sbsar2ptex.translate import _blur
from sbsar2ptex.translate import _dirmotionblur
from sbsar2ptex.translate import _distance
from sbsar2ptex.translate import _gradient_keys
from sbsar2ptex.translate import _levels
from sbsar2ptex.translate import _normal
from sbsar2ptex.translate import buffer
from sbsar2ptex.translate import drop_copies
from sbsar2ptex.translate import graph_inputs
from sbsar2ptex.translate import merge_same
from sbsar2ptex.translate import shader
from sbsar2ptex.translate import translate_graph
from sbsar2ptex.translate import wire

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


def test_outputsize_is_the_render_size() -> None:
    # A function reading $outputsize sees the size the graph renders at, not
    # the archive's default for it (8 here).
    files = read_archive(DATA / "pp_reads_outputsize.sbsar")
    asm = Asm(member(files, ".sbsasm"))
    inputs = graph_inputs(asm, member(files, ".xml").decode("utf-8"))
    assert [gi.default for gi in inputs.values() if gi.identifier == "$outputsize"] == [
        [8, 8]
    ]
    xml = member(files, ".xml").decode("utf-8")
    images = {
        int(u): f"{u}.png" for u in re.findall(r'uid="(\d+)" identifier="input_', xml)
    }
    nodes, _, _ = translate_graph(
        asm.nodes(), {"output": asm.output_nodes[0][1]}, images, size=10, inputs=inputs
    )
    params = {k: v for n in nodes for k, v in n.get("parameters", {}).items()}
    assert params["p_outputsize_x"] == params["p_outputsize_y"] == 10


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


@pytest.mark.parametrize("color", [False, True])
def test_empty_levels_range_ramps(*, color: bool) -> None:
    # in_low == in_high ramps over 1e-4, as Substance's renders of a 16-bit
    # input near the edge show, rather than stepping; the knobs stay live.
    edge = struct.unpack("<I", struct.pack("<f", 0.5))[0]
    node = Node(index=0, start=0, end=4, op=1, fields=[1 | 1 << 2], inline=[edge, edge])
    sh, _ = _levels(node, "levels", color=color, size=8)
    code = sh["shader_model"]["instance"]
    assert "1e-4" in code
    assert "step(ceil" not in code
    assert "x = clamp((x - lo) / d, 0.0, 1.0);" in code
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


def test_identical_nodes_merge() -> None:
    # Two copies of a node of one input, and a node reading each: both pairs
    # merge (the second once the first has); a different one stays.
    mm = [
        shader("a", [], "f", "return uv.x;"),
        shader("b1", [("in", "f")], "f", "return $in(uv) * 0.5;"),
        shader("b2", [("in", "f")], "f", "return $in(uv) * 0.5;"),
        shader("c1", [("in", "f")], "f", "return 1.0 - $in(uv);"),
        shader("c2", [("in", "f")], "f", "return 1.0 - $in(uv);"),
        shader("d", [("in", "f")], "f", "return $in(uv) * 0.25;"),
    ]
    wires = [wire("a", "b1"), wire("a", "b2"), wire("b1", "c1"), wire("b2", "c2")]
    wires += [wire("a", "d")]
    nodes, out, merged = merge_same(mm, wires)
    assert merged == {"b2": "b1", "c2": "c1"}
    assert {n["name"] for n in nodes} == {"a", "b1", "c1", "d"}
    assert out == [wire("a", "b1"), wire("b1", "c1"), wire("a", "d")]


def test_buffers_copying_a_buffer_go() -> None:
    # A buffer of a buffer of its size and filtering is the same image; a
    # 16-bit one of a 32-bit one is not, nor an unfiltered one of a
    # filtered one. A blur fed by a buffer drops its own first buffer.
    blur = _blur(Node(index=0, start=0, end=4), "blur", color=False, size=8)
    mm = [
        shader("src", [], "f", "return uv.x;"),
        buffer("first", 8),
        buffer("copy", 8),
        buffer("coarse", 8, filtering=False),
        buffer("wide", 8, precise=True),
        buffer("narrow", 8),
        shader("read", [("in", "f")], "f", "return $in(uv);"),
        *blur.nodes,
    ]
    wires = [
        wire("src", "first"), wire("first", "copy"), wire("copy", "read"),
        wire("first", "coarse"), wire("src", "wide"), wire("wide", "narrow"),
        wire("first", "blur"),
    ]  # fmt: skip
    nodes, out, moved = drop_copies(mm, wires)
    assert moved == {"copy": "first"}
    assert wire("first", "read") in out
    inner = {n["name"] for n in next(n for n in nodes if n["name"] == "blur")["nodes"]}
    assert "buf_x" not in inner
    assert "buf_y" in inner


@pytest.mark.parametrize("input2alpha", [0, 0x10])
def test_normal_alpha_is_height_only_if_asked(input2alpha: int) -> None:
    # Flag 0x10 (input2alpha) puts the height in alpha; off, alpha is 1. A
    # live knob either way.
    three = struct.unpack("<I", struct.pack("<f", 3.0))[0]
    node = Node(
        index=0, start=0, end=4, op=37, fields=[1 | input2alpha], inline=[three]
    )
    made = _normal(node, "normal", color=False, size=8)
    sh = next(n for n in made.nodes if n["type"] == "shader")
    assert sh["parameters"]["height_alpha"] == (1.0 if input2alpha else 0.0)
    assert "$height_alpha > 0.5 ? h : 1.0" in sh["shader_model"]["instance"]


@pytest.mark.parametrize("kind", ["f", "rgba"])
@pytest.mark.parametrize("bits", [8, 16, 32])
def test_directional_blur_knobs_stay_live(bits: int, kind: str) -> None:
    color = kind == "rgba"
    # Intensity and angle, inline in the node; its output format in bits 4-5
    # (0 8 bits, 1 16 bits, 3 32-bit float).
    words = [struct.unpack("<I", struct.pack("<f", v))[0] for v in (5.86, 0.0625)]
    flags = {8: 0, 16: 1, 32: 3}[bits] << 4
    node = Node(
        index=0, start=0, end=4, op=22, flags=flags, fields=[1 | 1 << 2], inline=words
    )
    made = _dirmotionblur(node, "dmb", color=color, size=8)
    # One shader, its knobs live (never constants in its GLSL), reading the
    # input's buffer; in an 8-bit graph with Substance's 8-bit stores, in a
    # 16-bit one with its 16-bit stores, on a 32-bit buffer (exact levels).
    by_type = {n["type"]: n for n in made.nodes}
    assert len(made.nodes) == 2
    buf = by_type["buffer"]["parameters"]
    assert buf["filter"] is True
    assert buf["f32"] is (bits == 16)
    sh = by_type["shader"]
    assert sh["parameters"] == pytest.approx({"intensity": 5.86, "angle": 0.0625})
    code = sh["shader_model"]["instance"]
    assert "$intensity" in code
    assert "$angle" in code
    assert "5.86" not in code
    assert "0.0625" not in code
    assert ("_s8(" in code) == (bits == 8)
    assert ("_pm(" in code) == ("_st(" in code) == (bits == 16)
    if bits == 16:
        # Integer levels per channel; NVIDIA's rules unless asked: halves
        # up, a value truncated to 2^-20 before its store.
        assert ("uvec4" in code) == color
        assert "floor((a + b + 1.0) * 0.5)" in code
        assert "1048576.0" in code
