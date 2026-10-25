"""Translate decompiled .sbsasm nodes into Material Maker shader nodes.

Each Substance atomic node becomes a Material Maker ``shader`` node whose
GLSL reproduces the filter; ``translate_graph`` wires them as the compiled
graph is wired. The formulas were fitted against sbsrender renders of
single-filter graphs (see tests/test_translate.py and sbsasm-probe):

- levels: v = clamp((x - in_low) / (in_high - in_low)); v = pow(v, g(mid)),
  g = 1 + 16 (mid - 0.5) above 0.5, 1 / (1 + 16 (0.5 - mid)) below; out =
  mix(out_low, out_high, v).
- blend: out = clamp(d + (f(s, d) - d) * o), o = opacity (times the opacity
  input, and for color the source's alpha); color alpha is "over",
  sa o + da (1 - sa o); switch mixes rgba by o alone.
- transformation: samples the input at M^T (uv - 0.5) + 0.5 + offset,
  wrapping (matrix22 "a b c d" is M = [[a, b], [c, d]]).
- uniform: a constant (a grayscale one keeps the color's green).

Parameters driven by function bytecode ("dynamic") are not translated yet.
"""

import itertools
import struct
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from sbsar2ptex.sbsasm import COLOR_OPS
from sbsar2ptex.sbsasm import Node


class UnsupportedError(Exception):
    """A node (op or parameter) the translator cannot express yet."""


def f32(w: int) -> float:
    """Return the u32 ``w`` read as an IEEE float."""
    return struct.unpack("<f", struct.pack("<I", w))[0]


@dataclass
class Params:
    """A node's parameters read from its flag word and inline data."""

    values: dict[str, Any] = field(default_factory=dict)


def flag_params(
    flags: int,
    inline: list[int],
    spec: list[tuple[str, int, Any]],
    width: int,
) -> dict[str, Any]:
    """Read parameters stored two flag bits each: 1 inline, 2 dynamic.

    ``spec`` lists (name, bit position, default); each inline parameter takes
    ``width`` floats, in spec order.
    """
    out: dict[str, Any] = {}
    at = 0
    for name, bit, default in spec:
        state = (flags >> bit) & 3
        if state == 0:
            out[name] = default
        elif state == 1:
            vals = [f32(w) for w in inline[at : at + width]]
            at += width
            out[name] = vals if width > 1 else vals[0]
        else:
            msg = f"parameter {name} is driven by a function"
            raise UnsupportedError(msg)
    return out


LEVELS = [
    ("in_low", 0, 0.0),
    ("in_high", 2, 1.0),
    ("in_mid", 4, 0.5),
    ("out_low", 6, 0.0),
    ("out_high", 8, 1.0),
]
BLEND_MODES = {
    0: "s",  # copy
    1: "d + s",  # add
    2: "d - s",  # subtract
    3: "s * d",  # multiply
    4: "d + 2.0 * s - 1.0",  # add sub
    5: "max(s, d)",
    6: "min(s, d)",
    7: "s",  # switch
    8: (  # divide; Substance takes x / 0 as +infinity, even 0 / 0
        "mix(vec4(1e9), d / max(s, vec4(1e-12)), step(vec4(1e-12), s))"
    ),
    9: "mix(2.0 * s * d, 1.0 - 2.0 * (1.0 - s) * (1.0 - d), step(0.5, d))",  # overlay
    10: "1.0 - (1.0 - s) * (1.0 - d)",  # screen
    11: (  # soft light (Photoshop's)
        "mix(2.0 * d * s + d * d * (1.0 - 2.0 * s),"
        " 2.0 * d * (1.0 - s) + sqrt(d) * (2.0 * s - 1.0), step(0.5, s))"
    ),
}


def num(v: float) -> str:
    """Return the shortest decimal that reads back as float32 ``v``."""
    target = struct.pack("<f", v)
    for digits in range(1, 10):
        text = f"{v:.{digits}g}"
        if struct.pack("<f", float(text)) == target:
            break
    else:
        text = repr(v)
    # GLSL (Godot's, strictly) wants float literals to look like floats.
    return text if any(c in text for c in ".en") else f"{text}.0"


def vec(values: list[float] | float, n: int = 4) -> str:
    """Return a GLSL vecN (or float) literal."""
    if isinstance(values, (int, float)):
        values = [values] * n
    nums = ", ".join(num(v) for v in values[:n])
    return f"vec{n}({nums})" if n > 1 else f"float({nums})"


def shader(
    name: str, inputs: list[tuple[str, str]], out_type: str, body: str
) -> dict[str, Any]:
    """Return a Material Maker shader node computing ``body`` (a GLSL function body).

    ``inputs`` are (name, "f" | "rgba"); in the body each is read as
    ``$name(uv)``, and the function returns vec4 (rgba) or float (f).
    """
    ret = "vec4" if out_type == "rgba" else "float"
    return {
        "name": name,
        "type": "shader",
        "node_position": {"x": 0, "y": 0},
        "parameters": {},
        "seed_int": 0,
        "shader_model": {
            "name": name,
            "code": "",
            "global": "",
            "inputs": [
                {
                    "name": i,
                    "type": t,
                    "default": "vec4(0.0)" if t == "rgba" else "0.0",
                    "function": True,
                    "label": i,
                }
                for i, t in inputs
            ],
            "instance": f"{ret} $(name)_fct(vec2 uv) {{\n{body}\n}}\n",
            "outputs": [{"type": out_type, out_type: "$(name)_fct($uv)"}],
            "parameters": [],
        },
    }


Made = tuple[dict[str, Any], list[str]]
"""A translated node: its Material Maker node and its input names by port."""


def _uniform(node: Node, name: str, *, color: bool, size: int) -> Made:
    vals = [f32(w) for w in node.inline]
    if color:
        return shader(name, [], "rgba", f"return {vec(vals or [1, 1, 1, 1])};"), []
    return shader(name, [], "f", f"return {num(vals[0]) if vals else '1.0'};"), []


def _levels(node: Node, name: str, *, color: bool, size: int) -> Made:
    t, n = ("rgba", 4) if color else ("f", 1)
    p = flag_params(node.fields[0], node.inline, LEVELS, n)
    g = "vec4" if color else "float"
    lines = [
        f"{g} x = $input(uv);",
        f"{g} lo = {vec(p['in_low'], n)};",
        f"{g} hi = {vec(p['in_high'], n)};",
        f"{g} m = {vec(p['in_mid'], n)};",
        "x = clamp((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0);",
        (
            "x = pow(x, mix(1.0 / (1.0 + 16.0 * (0.5 - m)),"
            " 1.0 + 16.0 * (m - 0.5), step(0.5, m)));"
        ),
        f"return mix({vec(p['out_low'], n)}, {vec(p['out_high'], n)}, x);",
    ]
    return shader(name, [("input", t)], t, "\n".join(lines)), ["input"]


def _blend(node: Node, name: str, *, color: bool, size: int) -> Made:
    t = "rgba" if color else "f"
    flags = node.fields[0]
    mode = flags & 0xF
    if mode not in BLEND_MODES:
        msg = f"blend mode {mode}"
        raise UnsupportedError(msg)
    inline = list(node.inline)
    has_input = bool(flags & 0x20)
    if has_input:
        inline = inline[1:]
    state = (flags >> 9) & 3
    if state == 2:  # noqa: PLR2004
        msg = "blend opacity is driven by a function"
        raise UnsupportedError(msg)
    opacity = f32(inline[0]) if state == 1 else 1.0
    o = num(opacity) + (" * $opacity(uv)" if has_input else "")
    expr = BLEND_MODES[mode]
    inputs = [("source", t), ("destination", t)]
    if has_input:
        inputs.append(("opacity", "f"))
    if color:
        lines = ["vec4 s = $source(uv); vec4 d = $destination(uv);", f"float o = {o};"]
        if mode == 7:  # noqa: PLR2004 (switch)
            lines.append("return mix(d, s, o);")
        else:
            lines += [
                f"vec4 f = {expr};",
                "float k = o * s.a;",
                "vec3 rgb = clamp(d.rgb + (f.rgb - d.rgb) * k, 0.0, 1.0);",
                "return vec4(rgb, k + d.a * (1.0 - k));",
            ]
    else:
        lines = [
            "vec4 s = vec4($source(uv)); vec4 d = vec4($destination(uv));",
            f"float o = {o};",
            f"vec4 f = {expr};",
            "return clamp(d.x + (f.x - d.x) * o, 0.0, 1.0);",
        ]
    return shader(name, inputs, t, "\n".join(lines)), [i for i, _ in inputs]


def _transformation(node: Node, name: str, *, color: bool, size: int) -> Made:
    t = "rgba" if color else "f"
    flags = node.fields[0]
    matrix = [1.0, 0.0, 0.0, 1.0]
    offset = [0.0, 0.0]
    inline = list(node.inline)
    state_m = (flags >> 6) & 3
    state_o = (flags >> 25) & 3
    if 2 in (state_m, state_o):  # noqa: PLR2004
        msg = "transformation driven by a function"
        raise UnsupportedError(msg)
    if state_m == 1:
        matrix, inline = [f32(w) for w in inline[:4]], inline[4:]
    if state_o == 1:
        offset = [f32(w) for w in inline[:2]]
    a, b, c, d = matrix
    lines = [
        "vec2 p = uv - vec2(0.5);",
        (
            f"vec2 q = vec2({num(a)} * p.x + {num(c)} * p.y,"
            f" {num(b)} * p.x + {num(d)} * p.y);"
        ),
        f"return $input(fract(q + vec2(0.5) + {vec(offset, 2)}));",
    ]
    return shader(name, [("input", t)], t, "\n".join(lines)), ["input"]


def word_params(
    node: Node, spec: list[tuple[str, int, Any]], widths: dict[str, int]
) -> dict[str, Any]:
    """Read parameters stored two bits each in the node word's top byte."""
    out: dict[str, Any] = {}
    states = node.flags >> 8
    at = 0
    for pname, bit, default in spec:
        state = (states >> bit) & 3
        width = widths.get(pname, 1)
        if state == 0:
            out[pname] = default
        elif state == 1:
            vals = [f32(w) for w in node.inline[at : at + width]]
            at += width
            out[pname] = vals if width > 1 else vals[0]
        else:
            msg = f"parameter {pname} is driven by a function"
            raise UnsupportedError(msg)
    return out


def _gradient_keys(node: Node, *, color: bool) -> list[tuple[float, list[float]]]:
    """Return the gradient keys: (position, rgba or [gray]), by position.

    Color keys are two words, [u16 position | u8 r | u8 g][u8 b | u8 a | u16
    midpoint]; grayscale keys three u16s, position, value, midpoint. A
    midpoint other than 0x8000 means interpolation with midpoints, which is
    not translated yet.
    """
    count = node.fields[1]
    data = node.at(node.code[0], 2 * count)  # the first offset points at the keys
    keys: list[tuple[float, list[float]]] = []
    if color:
        for k in range(count):
            w0, w1 = data[2 * k], data[2 * k + 1]
            pos, mid = (w0 & 0xFFFF) / 0xFFFF, w1 >> 16
            rgba = [(w0 >> 16 & 0xFF), (w0 >> 24), (w1 & 0xFF), (w1 >> 8 & 0xFF)]
            keys.append((pos, [c / 255 for c in rgba]))
            if mid != 0x8000:  # noqa: PLR2004
                msg = "gradient with midpoints"
                raise UnsupportedError(msg)
    else:
        u16 = []
        for w in data:
            u16 += [w & 0xFFFF, w >> 16]
        for k in range(count):
            pos, value, mid = u16[3 * k : 3 * k + 3]
            keys.append((pos / 0xFFFF, [value / 0xFFFF]))
            if mid != 0x8000:  # noqa: PLR2004
                msg = "gradient with midpoints"
                raise UnsupportedError(msg)
    return sorted(keys, key=lambda k: k[0])


def _gradient(node: Node, name: str, *, color: bool, size: int) -> Made:
    keys = _gradient_keys(node, color=color)
    t, n = ("rgba", 4) if color else ("f", 1)
    g = "vec4" if color else "float"
    lines = ["float x = $input(uv);", f"{g} v = {vec(keys[0][1], n)};"]
    for (p0, _), (p1, c1) in itertools.pairwise(keys):
        lines.append(
            f"v = mix(v, {vec(c1, n)},"
            f" clamp((x - {num(p0)}) / max({num(p1 - p0)}, 1e-9), 0.0, 1.0));"
        )
    lines.append("return v;")
    return shader(name, [("input", "f")], t, "\n".join(lines)), ["input"]


def _shuffle(node: Node, name: str, *, color: bool, size: int) -> Made:
    select = node.fields[0]
    parts = []
    for c in range(4):
        sel = select >> (8 * c) & 0xFF
        src = "a" if sel < 4 else "b"  # noqa: PLR2004
        parts.append(f"{src}.{'rgba'[sel % 4]}")
    lines = [
        "vec4 a = $input1(uv); vec4 b = $input2(uv);",
        f"return vec4({', '.join(parts)});",
    ]
    inputs = [("input1", "rgba"), ("input2", "rgba")]
    return shader(name, inputs, "rgba", "\n".join(lines)), ["input1", "input2"]


HSL_GLSL = """vec3 hsl_$(name)(vec3 c) {
\tfloat mx = max(c.r, max(c.g, c.b)); float mn = min(c.r, min(c.g, c.b));
\tfloat l = (mx + mn) * 0.5; float d = mx - mn;
\tif (d < 1e-9) return vec3(0.0, 0.0, l);
\tfloat s = l <= 0.5 ? d / (mx + mn) : d / (2.0 - mx - mn);
\tfloat h = mx == c.r ? mod((c.g - c.b) / d, 6.0)
\t\t: (mx == c.g ? (c.b - c.r) / d + 2.0 : (c.r - c.g) / d + 4.0);
\treturn vec3(fract(h / 6.0), s, l);
}
float hue_$(name)(float p, float q, float t) {
\tt = fract(t);
\tif (t < 1.0 / 6.0) return p + (q - p) * 6.0 * t;
\tif (t < 0.5) return q;
\tif (t < 2.0 / 3.0) return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
\treturn p;
}
vec3 rgb_$(name)(vec3 hsl) {
\tfloat q = hsl.z < 0.5 ? hsl.z * (1.0 + hsl.y) : hsl.z + hsl.y - hsl.z * hsl.y;
\tfloat p = 2.0 * hsl.z - q;
\treturn vec3(hue_$(name)(p, q, hsl.x + 1.0 / 3.0), hue_$(name)(p, q, hsl.x),
\t\thue_$(name)(p, q, hsl.x - 1.0 / 3.0));
}
"""


def _hsl(node: Node, name: str, *, color: bool, size: int) -> Made:
    p = word_params(
        node, [("hue", 0, 0.5), ("saturation", 2, 0.5), ("luminosity", 4, 0.5)], {}
    )
    lines = [
        "vec4 c = $input(uv);",
        "vec3 h = hsl_$(name)(c.rgb);",
        f"h.x += {num(2 * (p['hue'] - 0.5))};",
        f"h.y = clamp(h.y + {num(2 * (p['saturation'] - 0.5))}, 0.0, 1.0);",
        f"h.z = clamp(h.z + {num(2 * (p['luminosity'] - 0.5))}, 0.0, 1.0);",
        "return vec4(rgb_$(name)(h), c.a);",
    ]
    made = shader(name, [("input", "rgba")], "rgba", "\n".join(lines))
    made["shader_model"]["instance"] = HSL_GLSL + made["shader_model"]["instance"]
    return made, ["input"]


def _grayscale(node: Node, name: str, *, color: bool, size: int) -> Made:
    p = word_params(
        node,
        [("weights", 0, [1 / 3, 1 / 3, 1 / 3, 0.0]), ("matte", 2, 0.0)],
        {"weights": 4},
    )
    alpha_mult = bool(node.flags >> 8 & 0x10)
    lines = ["vec4 c = $input(uv);", f"float g = dot(c, {vec(p['weights'])});"]
    lines.append(
        f"return mix({num(p['matte'])}, g, c.a);" if alpha_mult else "return g;"
    )
    return shader(name, [("input", "rgba")], "f", "\n".join(lines)), ["input"]


def _curve_points(words: list[int], at: int) -> tuple[list[list[float]], int]:
    """Read one curve list at word ``at``: a count, then six floats a point.

    Returns the points (x, y, left x, left y, right x, right y) and the
    index after them.
    """
    count = words[at]
    floats = [f32(w) for w in words[at + 1 : at + 1 + 6 * count]]
    return [floats[6 * k : 6 * k + 6] for k in range(count)], at + 1 + 6 * count


def _curve_glsl(fn: str, points: list[list[float]]) -> str:
    """Return GLSL ``float fn(float x)`` evaluating a Bezier curve's y at x.

    Each segment runs from a point through its right handle and the next
    point's left handle; t is found by bisection on x(t), which the handles
    keep monotonic. Outside the points the end values hold.
    """
    pts = sorted(points, key=lambda p: p[0])
    lines = [
        f"float {fn}(float x) {{",
        f"\tif (x <= {num(pts[0][0])}) return {num(pts[0][1])};",
    ]
    for p0, p1 in itertools.pairwise(pts):
        a, b = f"vec2({num(p0[0])}, {num(p0[1])})", f"vec2({num(p0[4])}, {num(p0[5])})"
        c, d = f"vec2({num(p1[2])}, {num(p1[3])})", f"vec2({num(p1[0])}, {num(p1[1])})"
        lines += [
            f"\tif (x <= {num(p1[0])}) {{",
            "\t\tfloat lo = 0.0; float hi = 1.0; vec2 q = vec2(0.0);",
            "\t\tfor (int i = 0; i < 24; i++) {",
            "\t\t\tfloat t = 0.5 * (lo + hi); float s = 1.0 - t;",
            (
                f"\t\t\tq = s * s * s * {a} + 3.0 * s * s * t * {b}"
                f" + 3.0 * s * t * t * {c} + t * t * t * {d};"
            ),
            "\t\t\tif (q.x < x) lo = t; else hi = t;",
            "\t\t}",
            "\t\treturn q.y;",
            "\t}",
        ]
    lines += [f"\treturn {num(pts[-1][1])};", "}"]
    return "\n".join(lines) + "\n"


def _curve(node: Node, name: str, *, color: bool, size: int) -> Made:
    """Curve: per-channel curves (color) first, then the luminance curve.

    The curve data is at the node's first offset: the luminance curve, then
    for color red, green, blue and alpha; an empty curve leaves the channel.
    """
    words = node.at(node.code[0], len(node.words))
    lum, at = _curve_points(words, 0)
    helpers = []
    if lum:
        helpers.append(_curve_glsl("lum_$(name)", lum))
    if not color:
        body = "float x = $input(uv);\n" + (
            "return lum_$(name)(x);" if lum else "return x;"
        )
        made = shader(name, [("input", "f")], "f", body)
    else:
        lines = ["vec4 c = $input(uv);"]
        for ch in "rgba":
            pts, at = _curve_points(words, at)
            if pts:
                helpers.append(_curve_glsl(f"{ch}_$(name)", pts))
                lines.append(f"c.{ch} = {ch}_$(name)(c.{ch});")
        if lum:
            lines += [f"c.{ch} = lum_$(name)(c.{ch});" for ch in "rgb"]
        lines.append("return c;")
        made = shader(name, [("input", "rgba")], "rgba", "\n".join(lines))
    made["shader_model"]["instance"] = (
        "".join(helpers) + made["shader_model"]["instance"]
    )
    return made, ["input"]


def _dyngradient(node: Node, name: str, *, color: bool, size: int) -> Made:
    """Dynamic gradient: sample the gradient at (input, coordinate).

    The input value maps to texel centers, x (N - 1) / N + 0.5 / N, sampled
    with clamping; the U/V selector (node-word flag 0x08) swaps the axes.
    """
    t = "rgba" if color else "f"
    coordinate = word_params(node, [("coordinate", 1, 0.5)], {})["coordinate"]
    swap = bool(node.flags >> 8 & 0x08)
    n = float(1 << size)
    x = f"$input(uv) * {num((n - 1) / n)} + {num(0.5 / n)}"
    p = f"vec2({num(coordinate)}, {x})" if swap else f"vec2({x}, {num(coordinate)})"
    lo, hi = num(0.5 / n), num(1 - 0.5 / n)
    body = f"return $gradient(clamp({p}, vec2({lo}), vec2({hi})));"
    inputs = [("input", "f"), ("gradient", t)]
    return shader(name, inputs, t, body), ["input", "gradient"]


TRANSLATORS = {
    0: _gradient,
    1: _gradient,
    2: _blend,
    3: _blend,
    4: _transformation,
    5: _transformation,
    6: _grayscale,
    7: _shuffle,
    12: _uniform,
    13: _uniform,
    29: _hsl,
    30: _levels,
    31: _levels,
    38: _dyngradient,
    39: _dyngradient,
    44: _curve,
    45: _curve,
}
"""Each op's translator: (node, MM node name, *, color, size log2) -> Made."""


def node_shader(node: Node, name: str, size: int = 11) -> Made:
    """Return the shader node for ``node`` and its input names, in port order."""
    translate = TRANSLATORS.get(node.op)
    if translate is None:
        msg = f"op {node.op}"
        raise UnsupportedError(msg)
    return translate(node, name, color=node.op in COLOR_OPS, size=size)


def translate_graph(
    nodes: list[Node],
    outputs: dict[str, int],
    images: dict[int, str],
    size: int = 11,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, tuple[str, int]]]:
    """Return (MM nodes, connections, output name -> (MM node, port)).

    ``images`` gives each graph input's uid an image file for its image node.
    Only the nodes the outputs need are translated.
    """
    need: set[int] = set()
    todo = list(outputs.values())
    while todo:
        k = todo.pop()
        if k not in need:
            need.add(k)
            todo += nodes[k].inputs.values()
    mm: list[dict[str, Any]] = []
    wires: list[dict[str, Any]] = []
    for k in sorted(need):
        node = nodes[k]
        name = f"n{k}"
        if node.op in (32, 33):
            if node.uid not in images:
                msg = f"no image for graph input {node.uid}"
                raise UnsupportedError(msg)
            mm.append(
                {
                    "name": name,
                    "type": "image",
                    "node_position": {"x": 0, "y": 0},
                    "parameters": {"image": images[node.uid]},
                    "seed_int": 0,
                }
            )
            continue
        made, ports = node_shader(node, name, size)
        mm.append(made)
        for port, inp in enumerate(ports):
            wires.append(
                {
                    "from": f"n{node.inputs[inp]}",
                    "from_port": 0,
                    "to": name,
                    "to_port": port,
                }
            )
    for row, made in enumerate(mm):
        made["node_position"] = {"x": 220 * (row % 12), "y": 180 * (row // 12)}
    return mm, wires, {o: (f"n{k}", 0) for o, k in outputs.items()}
