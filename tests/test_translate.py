"""The translator's Material Maker graphs, on the cooked probe fixtures.

Pixel accuracy is checked against sbsrender by scripts/check_filters.py
(needs Substance Designer and Material Maker); these check the wiring.
"""

import json
import re
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sbsar2ptex import fxdraw
from sbsar2ptex import fxglsl
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
    # the archive's default for it (8 here): the material's, live, not a
    # knob (the graph is built at 10, and so is the node).
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
    assert "p_outputsize_x" not in params
    assert "ivec2(int(round(log2($pass_size.x))))" in json.dumps(nodes)


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


def test_distance_seeds_are_substances() -> None:
    # Biomech's n232, distance_strip1_g: seeds to 1/65535 (16-bit unorm),
    # a mask pixel at distance 0 from its own, neighbours in Substance's
    # order: 91.11% exact before, 99.99% now
    node = Node(index=0, start=0, end=4, op=43, fields=[6, 1, 0],
                inline=[struct.unpack("<I", struct.pack("<f", 1.28))[0]])  # fmt: skip
    made = _distance(node, "distance", color=False, size=11)
    code = {n["name"]: n["shader_model"]["instance"]
            for n in made.nodes if n["type"] == "shader"}  # fmt: skip
    assert "floor(uv * 65535.0 + 0.5) / 65535.0" in code["distance_seed"]
    assert "ivec2(1, 0), ivec2(-1, 0), ivec2(0, 1)" in code["distance_pass0"]
    assert "lessThan(abs(c.xy - uv)" in code["distance"]


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
        op=1,
        flags=0x119,
        fields=[0, 2],
        code=[4],
        words=words,
    )
    assert _gradient_keys(node, color=True) == [
        (0.0, [0.0, 1.0, 0.0, 1.0]),
        (1.0, [1.0, 1.0, 1.0, 1.0]),
    ]


def gradient_node(op: int, flags: int, count: int, key_bytes: str) -> Node:
    """Return a gradient map whose keys (hex, in file order) follow its node word."""
    raw = bytes.fromhex(key_bytes)
    raw += bytes(-len(raw) % 4)
    words = [flags << 16 | op, *struct.unpack(f"<{len(raw) // 4}I", raw)]
    return Node(index=0, start=0, end=4 * len(words), op=op, flags=flags,
                fields=[0, count], code=[4], words=words)  # fmt: skip


RED, GREEN, BLUE = [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]


@pytest.mark.parametrize("mid", ["0000", "3333", "0080", "cccc", "ffff"])
def test_gradient_keys_gray_with_trailing_u16(mid: str) -> None:
    # docs/gradient-map.md, acceptance test 1: flag 0x100, 6-byte gray keys.
    node = gradient_node(0, 0x119, 2, f"0000 0000 {mid} ffff ffff {mid}")
    assert _gradient_keys(node, color=False) == [(0.0, [0.0]), (1.0, [1.0])]


def test_gradient_keys_gray_without_trailing_u16() -> None:
    # Acceptance test 2: no flag 0x100, 4-byte gray keys.
    node = gradient_node(0, 0x19, 4, "0000 0000 0080 ffff 1481 0000 ffff ffff")
    keys = _gradient_keys(node, color=False)
    assert [v for _, v in keys] == [[0.0], [1.0], [0.0], [1.0]]
    assert [p for p, _ in keys] == pytest.approx([0.0, 0.5, 0.50421, 1.0], abs=1e-5)


@pytest.mark.parametrize("mid", ["0000", "3333", "0080", "cccc", "ffff"])
def test_gradient_keys_color_with_trailing_u16(mid: str) -> None:
    # Acceptance test 3: flag 0x100, 8-byte color keys.
    node = gradient_node(
        1, 0x119, 3,
        f"0000 ff0000ff {mid} 0080 00ff00ff {mid} ffff 0000ffff {mid}",
    )  # fmt: skip
    keys = _gradient_keys(node, color=True)
    assert [v for _, v in keys] == [RED, GREEN, BLUE]
    assert [p for p, _ in keys] == pytest.approx([0.0, 0.50001, 1.0], abs=1e-5)


def test_gradient_keys_color_without_trailing_u16() -> None:
    # Acceptance test 4: no flag 0x100, 6-byte color keys.
    node = gradient_node(1, 0x19, 2, "0000 000000ff ffff ffffffff")
    assert _gradient_keys(node, color=True) == [
        (0.0, [0.0, 0.0, 0.0, 1.0]),
        (1.0, [1.0, 1.0, 1.0, 1.0]),
    ]


def test_gradient_keys_bad_data_is_unsupported() -> None:
    keys = "0000 0000 0080 ffff ffff 0080"
    with pytest.raises(UnsupportedError):  # more keys than bytes
        _gradient_keys(gradient_node(0, 0x119, 3, keys), color=False)
    with pytest.raises(UnsupportedError):  # no keys' offset
        _gradient_keys(replace(gradient_node(0, 0x19, 2, keys), code=[]), color=False)
    node = replace(gradient_node(0, 0x119, 2, keys), code=[4, 12])
    with pytest.raises(UnsupportedError):  # running into the next block
        _gradient_keys(node, color=False)
    node = replace(gradient_node(0, 0x19, 2, keys), fields=[0])
    with pytest.raises(UnsupportedError):  # no key count
        _gradient_keys(node, color=False)


def test_gradient_fixed_size() -> None:
    # A gradient without flag bit 0 (0x18) has no size function: its node
    # word's second byte is its size.
    node = gradient_node(0, 0x18, 2, "0000 0000 ffff ffff")
    node.words[0] |= 0x55 << 8
    assert translate._output_size(node, 11) == 5


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


def test_buffers_follow_the_materials_size_unless_fixed(monkeypatch) -> None:
    # A fixed size stays at any graph size (absolute); the graph's size, or
    # a size function's of it, follows the material's (size_mode 1, by the
    # size less the graph's build size).
    word = 0x0318BB04
    n4677 = Node(index=0, start=0, end=4, op=4, flags=word >> 16, words=[word])
    wide = Node(index=0, start=0, end=4, op=4, flags=0x318, words=[0x03184804])
    monkeypatch.setattr(translate, "_CTX", translate._Context(size=10, graph_size=10))
    assert translate.absolute_size(n4677, 10)
    assert not translate.absolute_size(wide, 10)
    assert translate.absolute_size(n4677, translate.MAX_SIZE)  # (one smaller)
    rel = translate.buffer("b", 8)["parameters"]
    assert (rel["size"], rel["size_mode"], rel["size_offset"]) == (8, 1, -2.0)
    fixed = translate.buffer("b", 11, absolute=True)["parameters"]
    assert (fixed["size"], fixed["size_mode"], fixed["size_offset"]) == (11, 0, 0.0)
    translate._CTX.absolute = True  # the node being translated's, by default
    assert translate.buffer("b", 11)["parameters"]["size_mode"] == 0


def test_size_literals_are_live_but_an_fxmaps(monkeypatch) -> None:
    # A node's width is its pass's ($pass_size: each node renders at its own
    # size), another size's scaled from it; in an FX-Map, the build size's.
    monkeypatch.setattr(translate, "_CTX", translate._Context(size=9, graph_size=10))
    assert translate.px_glsl() == "$pass_size.x"
    assert translate.px_glsl(7) == "($pass_size.x * 0.25)"
    assert translate.px_int_glsl() == "int($pass_size.x)"
    translate._CTX.fixed = True
    assert translate.px_glsl() == "512.0"
    assert translate.px_glsl(7) == "128.0"


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
    # live: its pass's, a buffer its own fixed size (absolute: 2048 at any
    # graph size)
    assert "vec2($pass_size.x)" in code
    assert "vec2(1024.0)" not in code
    (own,) = [n for n in mm if n["name"] == "n1_size"]
    assert own["parameters"]["size"] == 11
    assert own["parameters"]["size_mode"] == 0


def test_a_gradient_maps_size_function_follows_its_keys() -> None:
    # Biomech's n3904 (its words): the keys, then a size function of
    # $outputsize (32 pixels in a 1024 graph): its offset the word after
    # the keys', as sbsasm reads it (Node.size_code)
    words =[0x01195500, 0xF3F, 4, 0xF578B0, 0xF578C8, 0, 0x80008000, 0x8000FFFF,
             0x8114, 0xFFFF8000, 0x8000FFFF, 0xA420015, 0xE3411820, 0x551, 0x10910,
             0x9100000, 0x10001, 0x2085F, 0x9130003, 0x30002, 0xD00, 0, 0x5094D,
             0x9130006, 0x20003, 0xD00, 0, 0x9094D, 0xD490008, 0x70004, 0x1140000A,
             0x40A00000, 0x40A00000, 0xC0952, 0x953000B, 0x1000D, 0xE0947,
             0x11400000, 0, 0, 0xE0970, 0x6510010, 0xA4C0011, 0x12000F, 0xA52,
             0x13]  # fmt: skip
    start = 16087248
    n3904 = Node(index=1, start=start, end=start + 4 * len(words), op=0,
                 flags=words[0] >> 16, fields=words[1:3],
                 code=[words[3] + BASE, words[4] + BASE], size_code=words[4] + BASE,
                 inputs={"input": 0}, words=words)  # fmt: skip
    image = Node(index=0, start=0, end=4, op=32, uid=500)
    size = translate.GraphInput("$outputsize", 5, [10, 10])
    mm, _, _ = translate_graph(
        [image, n3904], {"out": 1}, {500: "a.png"}, size=10, inputs={words[12]: size}
    )
    buffers = {n["name"]: n["parameters"] for n in mm if n["type"] == "buffer"}
    assert buffers["n1_size"]["size"] == 5
    # its function gives 32 pixels at any graph size (9, 10, 11): absolute
    assert buffers["n1_size"]["size_mode"] == 0


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
    assert "int nin = int(float($pass_size.x) * sc + 0.5);" in code
    assert "int ml = nin >> L;" in code
    assert "tmlvin_$(name)(x, 0)" in code
    assert "$m11" in code
    # one read of the input: a chain inlined in one shader stays its length
    assert code.count("$input(") == 1


def _nv_warp_level(p: tuple[int, int, int, int], fx: int, fy: int) -> int:
    """NVIDIA's bilinear read of 16-bit levels p00, p10, p01, p11, stored."""
    w11 = (fx * fy + 128) >> 8
    w = (256 - fx - fy + w11, fx - w11, fy - w11, w11)
    return (sum(a * b for a, b in zip(w, p, strict=True)) + 128) >> 8


def _glsl_warp_level(code: str, p: tuple[int, int, int, int], fx: int, fy: int):
    """Run warp_glsl's integer _mix and _st (16 bits) in Python."""
    mix = code.split("uint fy = uint(f.y);\n")[1].split("\treturn", 1)[0]
    store = code.split("float $(name)_st(uint num) {\n\treturn float(")[1]
    env = {"fx": fx, "fy": fy}
    for corner, level in zip("abcd", p, strict=True):
        mix = mix.replace(f"$(name)_lv({corner})", str(level))
    for line in mix.split(";")[:-1]:
        name, expr = line.replace("uint ", "").split("=", 1)
        env[name.strip()] = eval(re.sub(r"(\d)u\b", r"\1", expr), {}, env)  # noqa: S307
    expr = re.sub(r"(\d)u\b", r"\1", store.split(");\n")[0])
    return eval(expr, {}, env)  # noqa: S307


@pytest.mark.parametrize("bits", [8, 16])
def test_a_blur_is_two_directional_blurs(bits: int) -> None:
    # Substance renders a blur as directional blurs at 0 and a quarter turn
    # (harness dmb2_322_*): the first's image stored in the node's format,
    # both on 32-bit buffers, the intensity one live knob.
    translate._CTX = translate._Context(size=8)
    word = struct.unpack("<I", struct.pack("<f", 3.22))[0]
    node = Node(
        index=0, start=0, end=4, flags={8: 0, 16: 1}[bits] << 4 | 1 << 12,
        inline=[word],
    )  # fmt: skip
    made = _blur(node, "blur", color=False, size=8)
    (group,) = made.nodes
    inner = {n["name"]: n for n in group["nodes"]}
    assert all(inner[b]["parameters"]["f32"] for b in ("buf_x", "buf_y"))
    x = inner["pass_x"]["shader_model"]["instance"]
    y = inner["pass_y"]["shader_model"]["instance"]
    assert "float A = 0.0;" in x
    assert "float A = 0.25;" in y
    assert "$intensity" in x
    assert "3.22" not in x
    step = {8: "255.0", 16: "65535.0"}[bits]
    assert "$(name)_raw(" in x
    assert step in x.split("$(name)_raw(vec2 uv)")[-1]
    assert ("_s8(" in x) == (bits == 8)


def test_nvidia_warp_rounds_only_the_corner_weight(monkeypatch) -> None:
    monkeypatch.setattr(translate, "_DMB_AMD", False)
    code = translate.warp_glsl("float", 1, "n")
    cases = [
        # (texels, fx, fy, level): the corner's 9/256 rounds to 0, not the
        # exact blend's 9; its 128/256 = 1/2 rounds up, 256 not 128
        ((0, 0, 0, 65535), 3, 3, 0),
        ((0, 0, 0, 65535), 16, 8, 256),
        ((0, 0, 0, 65535), 255, 255, 65023),
        ((100, 200, 300, 401), 128, 128, 250),
        # one axis: the exact blend, halves up
        ((1, 2, 7, 9), 128, 0, 2),
        ((1, 2, 7, 9), 0, 128, 4),
        ((1000, 2000, 3000, 4000), 0, 0, 1000),
        ((0, 65535, 0, 65535), 255, 0, 65279),
    ]
    for p, fx, fy, level in cases:
        assert _nv_warp_level(p, fx, fy) == level
        assert _glsl_warp_level(code, p, fx, fy) == level
    p = (12345, 54321, 999, 65535)
    for fx in (0, 1, 77, 128, 255):
        for fy in (0, 1, 128, 200, 255):
            w11 = (fx * fy + 128) >> 8
            assert min(256 - fx - fy + w11, fx - w11, fy - w11) >= 0
            assert _glsl_warp_level(code, p, fx, fy) == _nv_warp_level(p, fx, fy)
            if 0 in (fx, fy):  # one axis: the exact blend, halves up
                num = (
                    (256 - fx) * (256 - fy) * p[0]
                    + fx * (256 - fy) * p[1]
                    + (256 - fx) * fy * p[2]
                    + fx * fy * p[3]
                )
                assert _nv_warp_level(p, fx, fy) == (num + 32768) >> 16


def _glsl_level16(code: str, v: np.ndarray) -> np.ndarray:
    """Run fxglsl.level16_glsl's float lines in float32 numpy; return k."""
    env = {
        "v": v.astype(np.float32),
        "floor": np.floor,
        "step": lambda e, x: (x >= e).astype(np.float32),
    }
    for line in code.split(";")[:-1]:
        name, expr = line.strip().removeprefix("float ").split("=", 1)
        name = name.strip()
        if name.endswith("+"):
            name = name[:-1].strip()
            expr = f"{name} + ({expr})"
        env[name] = np.float32(1) * eval(expr, {}, env)  # noqa: S307
        env[name] = env[name].astype(np.float32)
    return env["k"].astype(np.int64)


def test_16_bit_stores_are_the_gpus() -> None:
    # Substance's 16-bit store (ramp16_g, every pixel): AMD's the nearest
    # level exactly, where floor(65535 v + 0.5) in float32 sends values
    # just under a half up; NVIDIA's of v truncated to 2^-20, a half down.
    rng = np.random.default_rng(3)
    v = rng.random(200000, dtype=np.float32)
    k = np.arange(1, 65535)
    v = np.concatenate([v, ((k - 0.5) / 65535).astype(np.float32)])
    exact = np.floor(v.astype(np.float64) * 65535 + 0.5).astype(np.int64)
    amd = _glsl_level16(fxglsl.level16_glsl("float", amd=True), v)
    assert (amd == exact).all()
    naive = np.floor(v * np.float32(65535) + np.float32(0.5)).astype(np.int64)
    assert (naive != exact).any()
    nvidia = _glsl_level16(fxglsl.level16_glsl("float", amd=False), v)
    assert (nvidia == fxdraw.s16n(v)).all()
    assert (nvidia < exact).any()


def test_a_blurs_8_bit_stores_are_the_gpus(monkeypatch) -> None:
    # check_filters blur2_8 / blur322_8 on AMD: each pass's 8-bit store
    # through a half float as AMD's (89.74% / 87.88% with NVIDIA's, 99.997%)
    monkeypatch.setattr(translate, "_DMB_AMD", True)
    amd = translate._dmb_s8("float")
    level = fxglsl.level8_glsl("float", amd=True)
    assert all(line in amd for line in level.splitlines())
    monkeypatch.setattr(translate, "_DMB_AMD", False)
    assert "4096.0 + c" in translate._dmb_s8("float")


def test_a_dynamic_gradient_reads_texel_by_texel() -> None:
    # check_filters dyngrad_g (coordinate 0.3): the coordinate maps to texel
    # centers as the input does, and the gradient is read four texels with
    # fractions to 1/256, clamped (95.81% before, 100% now)
    body = translate.bilinear256_glsl("$gradient", "p", "2048.0", "f")
    assert "vec2 tf = floor((tt - tb) * 256.0 + 0.5) / 256.0;" in body
    assert body.count("$gradient(") == 4
    assert "float a = mix(" in body
    assert "vec4 a = mix(" in translate.bilinear256_glsl("$g", "p", "8.0", "rgba")


def test_hsl_is_substances() -> None:
    # check_filters hsl_c (hue 0.62, saturation 0.4, lightness 0.45): hue
    # from the smallest channel with offsets of 3 and 1 2^-12, saturation's
    # 1 + 2^-12, the hue shift in sixths (95.61% before, 99.9995% now).
    # This is HSL_GLSL in float32, on the colours plain HSL got wrong.
    f = np.float32
    want = {(0, 11, 128): (69, 8, 48), (0, 58, 129): (69, 8, 70)}
    for rgb, out in want.items():
        r, g, b = (f(x / 255) for x in rgb)
        mx = max(r, g, b)
        x, y, mn, base = (g, b, r, 3) if r <= min(g, b) else (
            (b, r, g, 5) if g <= min(r, b) else (r, g, b, 1))  # fmt: skip
        t, d = mn + mx, mx - mn
        h = f(base) - (f(0.000732421875) + x - y) / (f(0.000244140625) + d)
        h += f(12 * (0.62 - 0.5))
        s = min(max(d / (f(1.000244140625) - abs(f(1) - t)) + f(-0.2), 0), 1)
        lum = min(max(f(0.5) * t + f(-0.1), 0), 1)
        u = h * f(1 / 6)
        q = lum + s * (f(0.5) - abs(f(0.5) - lum))
        ramp = np.abs(f(u - np.floor(u)) - np.array([0.5, 1 / 3, 2 / 3], f))
        w = np.clip(ramp * np.array([6, -6, -6], f) + np.array([-1, 2, 2], f), 0, 1)
        v = (f(2) * lum - q) + (q - (f(2) * lum - q)) * w
        assert tuple(int(k) for k in fxdraw.s8(v.astype(f))) == out
    assert "0.000732421875" in translate.HSL_GLSL
    assert "1.000244140625" in translate.HSL_GLSL


def test_a_samples_filtering_0_is_the_nearest_texel(monkeypatch) -> None:
    # check_filters pp_filter0_noise / pp_filter1_noise (every pixel): a
    # pixel processor's sample with filtering 0 (no operand) reads the
    # nearest texel, 1 bilinear (the other way round before: 1.16% each)
    monkeypatch.setattr(translate, "_CTX", translate._Context(size=11))
    translate._CTX.sampled = ["input"]
    # (the node's size, live: its pass's)
    assert translate._sample("col", 0, "p", 0) == (
        "$input((floor(fract(p) * $pass_size.x) + 0.5) / $pass_size.x)"
    )
    assert translate._sample("col", 0, "p", 1) == "$input(fract(p))"
    # in an FX-Map, the build size's (its records render elsewhere)
    translate._CTX.fixed = True
    assert translate._sample("col", 0, "p", 0) == (
        "$input((floor(fract(p) * 2048.0) + 0.5) / 2048.0)"
    )


def test_8_bit_node_stores_are_the_gpus(monkeypatch) -> None:
    # check_filters' 8-bit blends (exact inputs), every pixel: AMD's level
    # through a half float, NVIDIA's truncated to 2^-12 (nvfit8 on an RTX
    # 3090: blend0..6_g, blend_addsub_half_g); a uniform fills its image:
    # 1.3 / 3 is 111, not 110
    translate._CTX = translate._Context(size=8)

    def stored_code(*, fill: bool) -> str:
        out = translate.shader("s", [], "f", "return 0.5;")
        assert translate.store_shader(out, 255, fill=fill)
        return out["shader_model"]["instance"]

    for amd in (True, False):
        monkeypatch.setattr(translate, "_DMB_AMD", amd)
        assert fxglsl.level8_glsl("float", amd=amd) in stored_code(fill=False)
        assert fxglsl.fill8_glsl("float") in stored_code(fill=True)
    v = np.float32(0.43333334)  # the gray's float32, a hair over 110.5 / 255
    assert fxdraw.s8(np.array([v]))[0] == fxdraw.s8n(np.array([v]))[0] == 110

    def fill8(v: np.float32) -> np.float32:  # fill8_glsl, in float32
        b = v * np.float32(256) - np.float32(0.5)
        r = b - v
        k = np.ceil(r)
        return k + np.float32(r == k and b - k > v)

    # fn_uniform_c's green, 0.3f: 255 v rounds to 76.5 in float32, so
    # ceil(255 v - 0.5) was 76 (75% of pixels); unrounded it is 77
    for x, want in ((v, 111), (np.float32(0.5) * np.float32(0.6), 77)):
        assert np.ceil(np.float64(x) * 255 - 0.5) == want
        assert fill8(x) == want
    assert np.ceil(np.float32(0.3) * np.float32(255) - np.float32(0.5)) == 76
    # NVIDIA's ties go down: an add-sub at half opacity, (a + b) / 255 - 0.5
    ties = (np.arange(1, 255) + 0.5) / 255
    assert (fxdraw.s8n(ties.astype(np.float32)) == np.arange(1, 255)).all()


def test_an_8_bit_warp_reads_its_inputs_bits() -> None:
    # Biomech's n4001: an 8-bit warp of 16-bit images reads their levels
    # whole (harness warpm12_grad16_8), an image in 8 bits (warp_8); both
    # on 32-bit buffers
    translate._CTX = translate._Context(size=8)
    word = struct.unpack("<I", struct.pack("<f", -1.2075))[0]
    node = Node(index=0, start=0, end=4, flags=1 << 12, inline=[word])
    for bits, eight in ((16, "false"), (8, "true")):
        translate._CTX.input_bits = {"input": 16, "gradient": bits}
        made = translate._warp(node, "w", color=False, size=8)
        warp = next(n for n in made.nodes if n["name"] == "w")
        code = warp["shader_model"]["instance"]
        assert "$(name)_level(v, false);" in code  # the input's
        assert f"wf_$(name)($(name)_level(v, {eight}))" in code
        assert all(n["parameters"]["f32"] for n in made.nodes if n["type"] == "buffer")


def test_amd_warp_blends_exactly(monkeypatch) -> None:
    monkeypatch.setattr(translate, "_DMB_AMD", True)
    code = translate.warp_glsl("float", 1, "n")
    assert "w11" not in code
    assert "uint num = (256u - fx) * (256u - fy) * $(name)_lv(a)" in code
