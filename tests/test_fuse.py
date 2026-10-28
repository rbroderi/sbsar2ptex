"""Fusing shaders into their one reader (pixels: a whole Biomech export)."""

from sbsar2ptex.fuse import fuse
from sbsar2ptex.translate import shader
from sbsar2ptex.translate import wire


def knobbed(name, inputs, body, knobs):
    made = shader(name, inputs, "f", body)
    made["shader_model"]["parameters"] = [
        {"name": k, "label": k.title(), "type": "float", "default": v}
        for k, v in knobs.items()
    ]
    made["parameters"] = dict(knobs)
    return made


def test_a_single_reader_shader_becomes_its_helper() -> None:
    # a -> b (b's second input); b also reads x. Both have a knob "gain" and
    # read the graph input p_seed.
    a = knobbed(
        "a", [("input", "f")], "return $input(uv) * $gain + $p_seed;",
        {"gain": 2.0, "p_seed": 0.0},
    )  # fmt: skip
    b = knobbed(
        "b", [("x", "f"), ("in", "f")], "return $x(uv) + $in(uv) * $gain;",
        {"gain": 3.0, "p_seed": 0.0},
    )  # fmt: skip
    nodes, wires = fuse(
        [a, b], [wire("src", "a"), wire("other", "b", 0), wire("a", "b", 1)]
    )
    (made,) = nodes
    m = made["shader_model"]
    assert [i["name"] for i in m["inputs"]] == ["x", "u1_input"]
    assert wires == [wire("src", "b", 1), wire("other", "b", 0)]
    assert made["parameters"] == {"gain": 3.0, "p_seed": 0.0, "u1_gain": 2.0}
    code = m["instance"]
    assert "$(name)_u1_fct(vec2 uv)" in code
    assert "return $u1_input(uv) * $u1_gain + $p_seed;" in code
    assert "return $x(uv) + $(name)_u1_fct(uv) * $gain;" in code


def test_shaders_read_twice_or_out_of_the_level_stay() -> None:
    a = shader("a", [], "f", "return uv.x;")
    b = shader("b", [("in", "f")], "f", "return $in(uv);")
    c = shader("c", [("in", "f")], "f", "return $in(uv);")
    d = shader("d", [], "f", "return uv.y;")
    wires = [wire("a", "b"), wire("a", "c"), wire("d", "gen_outputs")]
    nodes, out = fuse([a, b, c, d], wires)
    assert [n["name"] for n in nodes] == ["a", "b", "c", "d"]
    assert out == wires
