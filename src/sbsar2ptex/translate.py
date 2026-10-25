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
    8: "d / max(s, vec4(1e-9))",  # divide
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


def node_shader(node: Node, name: str) -> tuple[dict[str, Any], list[str]]:
    """Return the shader node for ``node`` and its input names, in port order."""
    color = node.op in COLOR_OPS
    t = "rgba" if color else "f"
    op = node.op & ~1 if node.op not in (1, 7, 29, 37) else node.op
    flags = node.fields[0] if node.fields else 0
    if op == 12:  # noqa: PLR2004 (uniform)
        vals = [f32(w) for w in node.inline]
        if color:
            body = f"return {vec(vals or [1, 1, 1, 1])};"
        else:
            body = f"return {num(vals[0]) if vals else '1.0'};"
        return shader(name, [], t, body), []
    if op == 30:  # noqa: PLR2004 (levels)
        p = flag_params(flags, node.inline, LEVELS, 4 if color else 1)
        n = 4 if color else 1
        mid = p["in_mid"]
        lines = [
            f"{'vec4' if color else 'float'} x = $input(uv);",
            f"{'vec4' if color else 'float'} lo = {vec(p['in_low'], n)};",
            f"{'vec4' if color else 'float'} hi = {vec(p['in_high'], n)};",
            f"{'vec4' if color else 'float'} m = {vec(mid, n)};",
            "x = clamp((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0);",
            (
                "x = pow(x, mix(1.0 / (1.0 + 16.0 * (0.5 - m)),"
                " 1.0 + 16.0 * (m - 0.5), step(0.5, m)));"
            ),
            f"return mix({vec(p['out_low'], n)}, {vec(p['out_high'], n)}, x);",
        ]
        return shader(name, [("input", t)], t, "\n".join(lines)), ["input"]
    if op == 2:  # noqa: PLR2004 (blend)
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
            lines = [
                "vec4 s = $source(uv); vec4 d = $destination(uv);",
                f"float o = {o};",
            ]
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
        names = [i for i, _ in inputs]
        return shader(name, inputs, t, "\n".join(lines)), names
    if op == 4:  # noqa: PLR2004 (transformation)
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
    msg = f"op {node.op}"
    raise UnsupportedError(msg)


def translate_graph(
    nodes: list[Node],
    outputs: dict[str, int],
    images: dict[int, str],
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
        made, ports = node_shader(node, name)
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
