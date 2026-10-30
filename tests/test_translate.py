"""The translator's Material Maker graphs, on the cooked probe fixtures.

Pixel accuracy is checked against sbsrender by scripts/check_filters.py
(needs Substance Designer and Material Maker); these check the wiring.
"""

import json
import re
import struct
from pathlib import Path

import pytest

from sbsar2ptex import translate
from sbsar2ptex.archive import member
from sbsar2ptex.archive import read_archive
from sbsar2ptex.sbsasm import BASE
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
    # settings outside [0, 1] are clamped, as Substance's are; still knobs
    assert code.count("clamp(") == 6
    assert "$in_low" in code
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


def test_blur_x_pass_graph_inputs_follow_the_blur() -> None:
    # A graph input its x pass read is exposed on the blur, not on the pass's
    # first name ("Node not found" in Material Maker).
    translate._CTX = translate._Context(size=8)
    translate._CTX.exposed["p_seed"] = ["blur_x", "blur", "other"]
    _blur(Node(index=0, start=0, end=4), "blur", color=False, size=8)
    assert translate._CTX.exposed == {"p_seed": ["blur", "other"]}


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


def test_blend_cropping_area() -> None:
    # Two words of 16-bit values times 32768: left and top, right and
    # bottom, after the opacity's words (check_filters blendcrop_g: 0.1 0.6
    # 0.2 0.7; Biomech's n1623: its right half). Live knobs.
    crop = Node(index=0, start=0, end=4, op=2, fields=[0x175])
    got = translate.blend_crop(crop, [0x199A0CCD, 0x599A4CCD])
    assert got == pytest.approx([0.1, 0.6, 0.2, 0.7], abs=1e-4)
    n1623 = Node(index=0, start=0, end=4, op=2, fields=[0x145])
    assert translate.blend_crop(n1623, [0x4000, 0x80008000]) == [0.5, 1.0, 0.0, 1.0]
    plain = Node(index=0, start=0, end=4, op=2, fields=[0x105])
    assert translate.blend_crop(plain, []) == [0.0, 1.0, 0.0, 1.0]
    node = Node(
        index=0, start=0, end=4, op=2, fields=[0x145], inline=[0x4000, 0x80008000]
    )
    sh, _ = translate._blend(node, "b", color=False, size=8)
    assert sh["parameters"]["crop_left"] == 0.5
    assert "$crop_left" in sh["shader_model"]["instance"]


def test_a_node_without_a_size_function_has_a_fixed_size() -> None:
    # The node word's second byte, log2 width and height, whatever the
    # graph's size (Biomech's n4677: 2048 in a 1024 graph, then read
    # averaged down; check_filters size_levels_abs6_g). Not square: the
    # graph's, as for a size function's.
    word = 0x0318BB04
    n4677 = Node(index=0, start=0, end=4, op=4, flags=word >> 16, words=[word])
    assert translate._output_size(n4677, 10) == 11
    assert translate._output_size(n4677, 12) == 11
    wide = Node(index=0, start=0, end=4, op=4, flags=0x318, words=[0x03184804])
    assert translate._output_size(wide, 10) == 10
    image = Node(index=0, start=0, end=4, op=32, flags=0, words=[0xBB20])
    assert translate._output_size(image, 10) == 10


def test_a_functions_size_is_its_nodes() -> None:
    # $size in a node's functions is the node's own size: Biomech's n4677
    # (2048 in a 1024 graph, its words) shifts by 0.5 / $size, half of its
    # pixels (Substance's render of it: n4676 shifted half a pixel, exact).
    # The cooker folds it where it can (check_filters size_shift_abs6_g).
    words = [0x0318BB04, 0x0400003F, 0x1244, 0xF80B54, 0x05410006, 0x09100001, 0,
             0xD00, 0x3F000000, 0x00020915, 0x09000001, 0, 0x0003094D, 4]  # fmt: skip
    start = words[3] + BASE - 16  # its offset function at the fifth word
    n4677 = Node(index=1, start=start, end=start + 4 * len(words), op=4,
                 flags=words[0] >> 16, fields=words[1:3], inline=words[3:],
                 inputs={"input": 0}, words=words)  # fmt: skip
    image = Node(index=0, start=0, end=4, op=32, uid=500)
    mm, _, _ = translate_graph([image, n4677], {"out": 1}, {500: "a.png"}, size=10)
    code = json.dumps(mm)
    assert "vec2(2048.0)" in code
    assert "vec2(1024.0)" not in code


def test_transformation_shrinking_reads_a_mip_level() -> None:
    # A mip level from log2 of the longer texel step, from 1.5x: 3x reads
    # level 2 (Biomech's n2333: tiling seams' rows blend, check_filters
    # transform_x3_g). Its input is buffered then, as stored; the matrix
    # stays knobs, the level found from them in the shader.
    assert translate.transform_minifies([3.0, 0.0, 0.0, 3.0])
    assert translate.transform_minifies([1.5, 0.0, 0.0, 1.0])
    assert not translate.transform_minifies([1.4, 0.0, 0.0, 1.4])
    assert not translate.transform_minifies([0.25, 0.0, 0.0, 0.25])
    assert not translate.transform_minifies([0.0, 0.0, 0.0, 0.0])
    three = struct.unpack("<I", struct.pack("<f", 3.0))[0]
    node = Node(
        index=0, start=0, end=4, op=4, fields=[1 << 6], inline=[three, 0, 0, three]
    )
    made = translate._transformation(node, "tf", color=False, size=8)
    assert isinstance(made, translate.Sub)
    by_type = {n["type"]: n for n in made.nodes}
    assert set(by_type) == {"buffer", "shader"}
    code = by_type["shader"]["shader_model"]["instance"]
    assert "int ml = 256 >> L;" in code
    assert "tmlvin_$(name)(x, 0)" in code
    assert "$m11" in code
    # one read of the input: a chain inlined in one shader stays its length
    assert code.count("$input(") == 1
