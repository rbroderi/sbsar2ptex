"""The "sbs" Material Maker plugin: one node per Substance atomic node.

``definitions`` makes the plugin's nodes (``sbs_<atom>``, ``sbs_<atom>_gray``
for an atom's grayscale variant where it has both), ``install`` writes them
with their plugin.cfg (``sbsar2ptex --install-mm-plugin DIR``), and ``Maker``
makes a translated graph's nodes plugin nodes (``--experimental``).

A plugin node computes what translate's inline translation of the same atom
computes, with the same GLSL: its settings are its knobs, with Substance's
names (``levelinlow``, ``blendingmode``, ``intensity``...), its output format
(Substance's 8 and 16 bits stores, ``format``) one too, and the image width
is ``$pass_size``, the size of the texture rendered (the fork's), not a
number. What a node reads around a pixel is buffered by the graph, as the
inline translation buffers it (one buffer for its readers, as precise as
the most precise of them), not by the node. Nodes of several passes are
graphs with their own buffers, sized relative to the material
(``outputsize``, Substance's output size, log2), or an iterate buffer (the
distance's jump flooding).

Not plugin nodes: gradient maps and curves (their keys and Bezier points
are no Material Maker parameter: its gradients and curves interpolate
otherwise), pixel processors and FX-Maps (functions, not settings), and
any node one of whose settings is a Substance function (other than a graph
input itself, which a plugin node's knob follows): those stay translated
inline.
"""

import json
from collections.abc import Container
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from pathlib import Path
from typing import Any

from sbsar2ptex.function import FLOAT as FUNC_FLOAT
from sbsar2ptex.function import decode
from sbsar2ptex.sbsasm import BASE
from sbsar2ptex.sbsasm import COLOR_OPS
from sbsar2ptex.sbsasm import Node
from sbsar2ptex.translate import BLEND_MODES
from sbsar2ptex.translate import BLEND_NAMES
from sbsar2ptex.translate import HSL_GLSL
from sbsar2ptex.translate import IN_FLOAT
from sbsar2ptex.translate import IN_FLOAT4
from sbsar2ptex.translate import LEVELS
from sbsar2ptex.translate import PLUGIN_LINKS
from sbsar2ptex.translate import PLUGIN_PREFIX
from sbsar2ptex.translate import RCP_GLSL
from sbsar2ptex.translate import TORUS_GLSL
from sbsar2ptex.translate import Sub
from sbsar2ptex.translate import UnsupportedError
from sbsar2ptex.translate import _dynamic_offset
from sbsar2ptex.translate import box_glsl_live
from sbsar2ptex.translate import buffer
from sbsar2ptex.translate import dmb_glsl
from sbsar2ptex.translate import dmb_setup_glsl
from sbsar2ptex.translate import f32
from sbsar2ptex.translate import graph_input
from sbsar2ptex.translate import input_params
from sbsar2ptex.translate import levels_glsl
from sbsar2ptex.translate import wire

VERSION = "1.0"
"""The plugin's version: graphs require it (``requires``)."""
PLUGIN_ID = "sbs"
N = "$pass_size.x"
"""The rendered texture's width: the plugin's image size."""
MASK = f"(int({N}) - 1)"
INPUT_OP = 0x02
"""The function bytecode op reading a graph input."""
JFA_STEPS = 13
"""Jump flooding's largest step is 2^12: images up to 8192 pixels."""

FORMATS = (
    "8 bits per channel",
    "16 bits per channel",
    "HDR low precision (16F)",
    "HDR high precision (32F)",
)
TILINGS = ("No tiling", "Horizontal tiling", "Vertical tiling", "H and V tiling")
CHANNELS = [
    f"Input {i} {c} channel" for i in (1, 2) for c in ("red", "green", "blue", "alpha")
]


# -- Parameters --------------------------------------------------------------


def _float(name: str, label: str, default: float, lo: float, hi: float) -> dict:
    return {"name": name, "label": label, "type": "float", "default": default,
            "min": lo, "max": hi, "step": 0.001}  # fmt: skip


def _color(name: str, label: str, default: list[float]) -> dict:
    r, g, b, a = default
    return {"name": name, "label": label, "type": "color",
            "default": {"r": r, "g": g, "b": b, "a": a}}  # fmt: skip


def _bool(name: str, label: str, *, default: bool) -> dict:
    return {"name": name, "label": label, "type": "boolean", "default": default}


def _enum(name: str, label: str, names: list[str] | tuple, default: int) -> dict:
    values = [{"name": n, "value": str(k)} for k, n in enumerate(names)]
    return {"name": name, "label": label, "type": "enum", "default": default,
            "values": values}  # fmt: skip


def _value(name: str, label: str, default: Any, *, color: bool) -> dict:
    """Return a setting that is a float4 in color, a float in grayscale."""
    if color:
        vals = default if isinstance(default, list) else [default] * 4
        return _color(name, label, vals)
    return _float(name, label, default, 0.0, 1.0)


FORMAT = _enum("format", "Output format", FORMATS, 0)
OUTPUTSIZE = {
    "name": "outputsize", "label": "Output size", "type": "float", "default": 0.0,
    "min": -6.0, "max": 4.0, "step": 1.0,
    "longdesc": "The node's size relative to the material's, as a power of 2",
}  # fmt: skip


# -- Node definitions --------------------------------------------------------


@dataclass
class Port:
    """An input of a plugin node.

    ``name`` is Substance's connector (the GLSL's ``$name``), ``key`` the
    decompiled node's input it is wired from (sbsasm LAYOUTS).
    """

    name: str
    kind: str
    label: str
    key: str
    default: str = ""


def _store(t: str) -> str:
    """Return ``$(name)_out``: ``$(name)_fct`` stored in its format (as stored)."""
    g = "vec4" if t == "rgba" else "float"
    return (
        f"{g} $(name)_out(vec2 uv) {{\n"
        "\tif ($format == 0) {\n"
        f"\t\t{g} v = clamp($(name)_fct(uv), 0.0, 1.0);\n"
        "\t\treturn ceil(v * 255.0 - 0.5) / 255.0;\n"
        "\t}\n"
        "\tif ($format == 1) {\n"
        f"\t\t{g} v = clamp($(name)_fct(uv), 0.0, 1.0);\n"
        "\t\treturn floor(v * 65535.0 + 0.5) / 65535.0;\n"
        "\t}\n"
        "\treturn $(name)_fct(uv);\n"
        "}\n"
    )


def _inputs(ports: list[Port]) -> list[dict[str, Any]]:
    return [
        {
            "name": p.name,
            "label": p.label,
            "shortdesc": p.label,
            "type": p.kind,
            "default": p.default or ("vec4(0.0)" if p.kind == "rgba" else "0.0"),
            "function": True,
        }
        for p in ports
    ]


def _shader_model(
    label: str,
    ports: list[Port],
    out: str,
    body: str,
    params: list[dict],
    helpers: str = "",
    shared: str = "",
    *,
    store: bool = True,
) -> dict[str, Any]:
    """Return a shader_model: ``body`` the function's, its output stored."""
    ret = "vec4" if out == "rgba" else "float"
    instance = f"{helpers}{ret} $(name)_fct(vec2 uv) {{\n{body}\n}}\n"
    if store:
        instance += _store(out)
    return {
        "name": label,
        "shortdesc": label,
        "longdesc": "",
        "code": "",
        "global": shared,
        "inputs": _inputs(ports),
        "instance": instance,
        "outputs": [
            {
                "type": out,
                out: f"$(name)_{'out' if store else 'fct'}($uv)",
                "shortdesc": "Output",
            }
        ],
        "parameters": [*params, FORMAT] if store else params,
    }


@dataclass
class Atom:
    """A plugin node type: its definition (.mmg) and how it is wired."""

    type: str
    label: str
    description: str
    ports: list[Port]
    output: str
    params: list[dict]
    """Its knobs (the format and output size too)."""
    definition: dict[str, Any] = field(default_factory=dict)


def _shader_atom(
    type_: str,
    label: str,
    description: str,
    ports: list[Port],
    out: str,
    body: str,
    params: list[dict],
    helpers: str = "",
    shared: str = "",
) -> Atom:
    model = _shader_model(label, ports, out, body, params, helpers, shared)
    model["longdesc"] = description
    definition = {
        "name": type_,
        "type": "shader",
        "node_position": {"x": 0, "y": 0},
        "parameters": {p["name"]: p["default"] for p in model["parameters"]},
        "shader_model": model,
    }
    return Atom(type_, label, description, ports, out, model["parameters"], definition)


def _kinds(*, color: bool) -> tuple[str, str, str]:
    """Return (MM type, GLSL type, type suffix)."""
    return ("rgba", "vec4", "") if color else ("f", "float", "_gray")


def _uniform(*, color: bool) -> Atom:
    t, _, sfx = _kinds(color=color)
    black = [0.0, 0.0, 0.0, 1.0] if color else 0.0
    return _shader_atom(
        f"sbs_uniform{sfx}", "Uniform color",
        "A constant color (Substance's Uniform color).",
        [], t, "return $outputcolor;",
        [_value("outputcolor", "Output color", black, color=color)],
    )  # fmt: skip


def _levels(*, color: bool) -> Atom:
    t, _, sfx = _kinds(color=color)
    names = {"in_low": "levelinlow", "in_high": "levelinhigh", "in_mid": "levelinmid",
             "out_low": "leveloutlow", "out_high": "levelouthigh"}  # fmt: skip
    labels = {"in_low": "Level in low", "in_high": "Level in high",
              "in_mid": "Level in mid", "out_low": "Level out low",
              "out_high": "Level out high"}  # fmt: skip
    params = [_value(names[k], labels[k], d, color=color) for k, _, d in LEVELS]
    order = ("in_low", "in_high", "in_mid", "out_low", "out_high")
    body = levels_glsl(*(f"${names[k]}" for k in order), color=color)
    return _shader_atom(
        f"sbs_levels{sfx}", "Levels",
        "Substance's Levels: every setting clamped to [0, 1].",
        [Port("input1", t, "Input", "input")], t,
        body.replace("$input(", "$input1("), params,
    )  # fmt: skip


BLEND_GLOBAL = (
    "vec4 sbs_blend_mode(int m, vec4 s, vec4 d) {\n"
    + "".join(f"\tif (m == {m}) return ({e});\n" for m, e in BLEND_MODES.items())
    + "\treturn s;\n}\n"
)
"""Each blending mode's color (translate.BLEND_MODES), by mode."""


def _blend(*, color: bool) -> Atom:
    t, _, sfx = _kinds(color=color)
    head = (
        "vec4 s = $source(uv); vec4 d = $destination(uv);"
        if color
        else "vec4 s = vec4($source(uv)); vec4 d = vec4($destination(uv));"
    )
    lines = [head, "float o = $opacitymult * $opacity(uv);"]
    if color:
        lines += [
            "if ($blendingmode == 7) return mix(d, s, o);",
            "vec4 f = sbs_blend_mode($blendingmode, s, d);",
            "float k = o * s.a;",
            "vec3 rgb = clamp(d.rgb + (f.rgb - d.rgb) * k, 0.0, 1.0);",
            "return vec4(rgb, k + d.a * (1.0 - k));",
        ]
    else:
        lines += [
            "vec4 f = sbs_blend_mode($blendingmode, s, d);",
            "return clamp(d.x + (f.x - d.x) * o, 0.0, 1.0);",
        ]
    names = {1: "Add (linear dodge)", 5: "Max (lighten)", 6: "Min (darken)"}
    modes = [names.get(m, BLEND_NAMES[m]) for m in range(12)]
    return _shader_atom(
        f"sbs_blend{sfx}", "Blend",
        "Substance's Blend: the foreground over the background.",
        [Port("source", t, "Foreground", "source"),
         Port("destination", t, "Background", "destination"),
         Port("opacity", "f", "Opacity", "opacity", "1.0")],
        t, "\n".join(lines),
        [_float("opacitymult", "Opacity", 1.0, 0.0, 1.0),
         _enum("blendingmode", "Blending mode", modes, 0)],
        shared=BLEND_GLOBAL,
    )  # fmt: skip


def _transformation(*, color: bool) -> Atom:
    t, _, sfx = _kinds(color=color)
    empty = "vec4(0.0)" if color else "0.0"
    lines = [
        "vec2 p = uv - vec2(0.5);",
        (
            "vec2 q = vec2($matrix22_x * p.x + $matrix22_z * p.y,"
            " $matrix22_y * p.x + $matrix22_w * p.y);"
        ),
        "q += vec2(0.5) + vec2($offset_x, $offset_y);",
        f"if (($tiling & 1) == 0 && (q.x < 0.0 || q.x >= 1.0)) return {empty};",
        f"if (($tiling & 2) == 0 && (q.y < 0.0 || q.y >= 1.0)) return {empty};",
        "return $input1(fract(q));",
    ]
    params = [
        _float(f"matrix22_{k}", f"Transform matrix {k}", float(k in "xw"), -2.0, 2.0)
        for k in "xyzw"
    ]
    params += [_float(f"offset_{k}", f"Offset {k}", 0.0, -1.0, 1.0) for k in "xy"]
    params.append(_enum("tiling", "Tiling mode", TILINGS, 3))
    return _shader_atom(
        f"sbs_transformation{sfx}", "Transformation 2D",
        "Substance's Transformation 2D: the input sampled at M^T (uv - 0.5) + 0.5"
        " + offset (matrix22 x y z w: M = [[x, y], [z, w]]), empty past its edges"
        " where it does not tile.",
        [Port("input1", t, "Input", "input")], t, "\n".join(lines), params,
    )  # fmt: skip


def _grayscale() -> Atom:
    lines = [
        "vec4 c = $input1(uv);",
        "float g = dot(c, $channelsweights);",
        "return $alphamult ? mix($mattelevel, g, c.a) : g;",
    ]
    return _shader_atom(
        "sbs_grayscaleconversion", "Grayscale conversion",
        "Substance's Grayscale conversion: the channels' weighted sum, over a"
        " background value where Flatten alpha is set.",
        [Port("input1", "rgba", "Input", "input")], "f", "\n".join(lines),
        [_color("channelsweights", "Channels weights", [0.33, 0.33, 0.33, 0.0]),
         _bool("alphamult", "Flatten alpha", default=False),
         _float("mattelevel", "Background value", 1.0, 0.0, 1.0)],
    )  # fmt: skip


def _shuffle() -> Atom:
    channels = [f"{'ab'[k // 4]}.{'rgba'[k % 4]}" for k in range(8)]
    params = []
    for k, c in enumerate(("red", "green", "blue", "alpha")):
        p = _enum(f"channel{c}", f"{c.capitalize()} channel", CHANNELS, k)
        pairs = zip(CHANNELS, channels, strict=True)
        p["values"] = [{"name": n, "value": v} for n, v in pairs]
        params.append(p)
    lines = [
        "vec4 a = $input1(uv); vec4 b = $input2(uv);",
        "return vec4($channelred, $channelgreen, $channelblue, $channelalpha);",
    ]
    return _shader_atom(
        "sbs_shuffle", "Channels shuffle",
        "Substance's Channels shuffle: each channel any channel of either input.",
        [Port("input1", "rgba", "Input 1", "input1"),
         Port("input2", "rgba", "Input 2", "input2")],
        "rgba", "\n".join(lines), params,
    )  # fmt: skip


def _hsl() -> Atom:
    lines = [
        "vec4 c = $input1(uv);",
        "vec3 h = hsl_$(name)(c.rgb);",
        "h.x += (2.0 * ($hue - 0.5));",
        "h.y = clamp(h.y + (2.0 * ($saturation - 0.5)), 0.0, 1.0);",
        "h.z = clamp(h.z + (2.0 * ($luminosity - 0.5)), 0.0, 1.0);",
        "return vec4(rgb_$(name)(h), c.a);",
    ]
    return _shader_atom(
        "sbs_hsl", "HSL", "Substance's HSL: hue, saturation and lightness shifted.",
        [Port("input1", "rgba", "Input", "input")], "rgba", "\n".join(lines),
        [_float("hue", "Hue", 0.5, 0.0, 1.0),
         _float("saturation", "Saturation", 0.5, 0.0, 1.0),
         _float("luminosity", "Lightness", 0.5, 0.0, 1.0)],
        helpers=HSL_GLSL,
    )  # fmt: skip


def _dyngradient(*, color: bool) -> Atom:
    t, _, sfx = _kinds(color=color)
    x = f"$input1(uv) * (({N} - 1.0) / {N}) + (0.5 / {N})"
    lo, hi = f"(0.5 / {N})", f"(1.0 - 0.5 / {N})"
    p = f"($uvselector == 1 ? vec2($coordinate, {x}) : vec2({x}, $coordinate))"
    body = f"return $input2(clamp({p}, vec2({lo}), vec2({hi})));"
    return _shader_atom(
        f"sbs_dyngradient{sfx}", "Gradient (Dynamic)",
        "Substance's Gradient (Dynamic): the gradient input sampled at (input,"
        " position), or (position, input) vertically, at texel centers.",
        [Port("input1", "f", "Grayscale input", "input"),
         Port("input2", t, "Gradient input", "gradient")],
        t, body,
        [_float("coordinate", "Gradient input position", 0.0, 0.0, 1.0),
         _enum("uvselector", "Gradient orientation", ("Horizontal", "Vertical"), 0)],
    )  # fmt: skip


def _sharpen(*, color: bool) -> Atom:
    t, _, sfx = _kinds(color=color)
    lines = [
        f"vec2 e = vec2(1.0 / {N}, 0.0);",
        "vec4 c = vec4($input1(uv));",
        (
            "vec4 s = vec4($input1(fract(uv + e))) + vec4($input1(fract(uv - e)))"
            " + vec4($input1(fract(uv + e.yx))) + vec4($input1(fract(uv - e.yx)));"
        ),
        "vec4 r = clamp(c + $intensity * (4.0 * c - s), 0.0, 1.0);",
        "return r;" if color else "return r.x;",
    ]
    return _shader_atom(
        f"sbs_sharpen{sfx}", "Sharpen",
        "Substance's Sharpen: x + intensity (4 x - its four neighbors).",
        [Port("input1", t, "Input", "input")], t, "\n".join(lines),
        [_float("intensity", "Intensity", 1.0, 0.0, 4.0)],
    )  # fmt: skip


def _warp(*, color: bool) -> Atom:
    t, _, sfx = _kinds(color=color)
    lines = [
        f"vec2 e = vec2(1.0 / {N}, 0.0);",
        "float h = $inputgradient(uv);",
        (
            "vec2 g = vec2(h - $inputgradient(fract(uv - e)),"
            " h - $inputgradient(fract(uv - e.yx)));"
        ),
        f"return $input1(fract(uv + (($intensity * {N}) / 256.0) * g));",
    ]
    return _shader_atom(
        f"sbs_warp{sfx}", "Warp",
        "Substance's Warp: the input sampled at uv + intensity / 256 times the"
        " gradient input's slope (per UV unit).",
        [Port("input1", t, "Input", "input"),
         Port("inputgradient", "f", "Gradient input", "gradient")],
        t, "\n".join(lines), [_float("intensity", "Intensity", 1.0, 0.0, 32.0)],
    )  # fmt: skip


def _directionalwarp(*, color: bool) -> Atom:
    t, _, sfx = _kinds(color=color)
    a = "6.28318531 * $warpangle"
    d = f"$intensity / 256.0 * vec2(cos({a}), sin({a}))"
    body = f"return $input1(fract(uv + $inputintensity(uv) * {d}));"
    return _shader_atom(
        f"sbs_directionalwarp{sfx}", "Directional warp",
        "Substance's Directional warp: the input sampled at uv + intensity /"
        " 256 times the intensity input along the angle (turns).",
        [Port("input1", t, "Input", "input"),
         Port("inputintensity", "f", "Intensity input", "intensity")],
        t, body,
        [_float("intensity", "Intensity", 10.0, 0.0, 64.0),
         _float("warpangle", "Warp angle", 0.0, 0.0, 1.0)],
    )  # fmt: skip


def _normal() -> Atom:
    k = f"($intensity * ({N} / 256.0))"
    lines = [
        f"vec2 e = vec2(1.0 / {N}, 0.0);",
        "float h = $input1(uv);",
        "vec2 g = vec2(h - $input1(fract(uv - e)), h - $input1(fract(uv - e.yx)));",
        (
            f"vec3 n = normalize(vec3((-{k}) * g.x,"
            f" (($inversedy ? 1.0 : -1.0) * {k}) * g.y, 1.0));"
        ),
        "return vec4(n * 0.5 + 0.5, $input2alpha ? h : 1.0);",
    ]
    return _shader_atom(
        "sbs_normal", "Normal",
        "Substance's Normal: the height's slope (backward differences) as a"
        " normal map, intensity per 256 pixels.",
        [Port("input1", "f", "Input", "input")], "rgba", "\n".join(lines),
        [_float("intensity", "Intensity", 1.0, 0.0, 8.0),
         _bool("inversedy", "Normal format (DirectX)", default=False),
         _bool("input2alpha", "Alpha channel content (height)", default=True)],
    )  # fmt: skip


def _emboss(*, color: bool) -> Atom:
    t, _, sfx = _kinds(color=color)
    a = "6.28318531 * $lightangle"
    lines = [
        f"vec2 e = vec2(1.0 / {N}, 0.0);",
        "float h = $inputgradient(uv);",
        (
            "vec2 g = vec2(h - $inputgradient(fract(uv - e)),"
            " h - $inputgradient(fract(uv - e.yx)));"
        ),
        f"vec2 light = vec2(cos({a}), sin({a}));",
        f"float s = (($intensity * {N}) / 256.0) * dot(g, light);",
    ]
    if color:
        lines += [
            "vec4 c = $input1(uv);",
            (
                "vec3 rgb = c.rgb + max(s, 0.0) * $highlightcolor.xyz"
                " - max(-s, 0.0) * (vec3(1.0) - $shadowcolor.xyz);"
            ),
            "return vec4(clamp(rgb, 0.0, 1.0), c.a);",
        ]
    else:
        lines.append(
            "return clamp($input1(uv) + max(s, 0.0) * $highlightcolor"
            " - max(-s, 0.0) * (1.0 - $shadowcolor), 0.0, 1.0);"
        )
    return _shader_atom(
        f"sbs_emboss{sfx}", "Emboss",
        "Substance's Emboss: the input lit by the intensity input's slope along"
        " the light angle (turns).",
        [Port("input1", t, "Input", "input"),
         Port("inputgradient", "f", "Intensity input", "gradient")],
        t, "\n".join(lines),
        [_float("intensity", "Intensity", 5.0, 0.0, 16.0),
         _float("lightangle", "Light angle", 0.0, 0.0, 1.0),
         _value("highlightcolor", "Highlight color", 1.0, color=color),
         _value("shadowcolor", "Shadow color", 0.0, color=color)],
    )  # fmt: skip


def _dirmotionblur(*, color: bool) -> Atom:
    t, g, sfx = _kinds(color=color)
    helpers, calls = [], []
    for fmt in (0, 1, 2):
        setup = dmb_setup_glsl("$intensity", "$mblurangle", N, lpx=fmt == 1)
        h, body = dmb_glsl(g, fmt, N, MASK, setup)
        tag = f"$(name)_v{fmt}_"
        helpers.append(h.replace("$(name)_", tag))
        body = body.replace("$(name)_", tag)
        helpers.append(f"{g} {tag}body(vec2 uv) {{\n{body}\n}}\n")
        calls.append(f"{tag}body(uv)")
    text = "".join(helpers).replace("$input(", "$input1(")
    body = (
        f"if ($format == 0) return {calls[0]};\n"
        f"if ($format == 1) return {calls[1]};\n"
        f"return {calls[2]};"
    )
    return _shader_atom(
        f"sbs_dirmotionblur{sfx}", "Directional blur",
        "Substance's Directional blur (dirmotionblur): a box of intensity / 256"
        " of the image along the angle (turns), computed as Substance computes"
        " it in its output format (8 bits, 16 bits or floats). Its input is"
        " read by texel (buffer it; 32 bits for a 16 bits blur).",
        [Port("input1", t, "Input", "input")], t, body,
        [_float("intensity", "Intensity", 10.0, 0.0, 32.0),
         _float("mblurangle", "Angle", 0.0, 0.0, 1.0)],
        helpers=text,
    )  # fmt: skip


def _plugin_buffer(name: str, *, filtering: bool = True, f32: bool = False) -> dict:
    """Return a buffer the size of the material times 2^outputsize."""
    return {
        "name": name,
        "type": "buffer",
        "node_position": {"x": 0, "y": 0},
        "parameters": {
            "size": 9, "f32": f32, "filter": filtering, "mipmap": False,
            "size_mode": 1, "size_offset": 0.0,
        },
        "seed_int": 0,
        "version": 2,
    }  # fmt: skip


def _inner_shader(
    name: str, ports: list[Port], out: str, body: str, params: list[dict],
    helpers: str = "", *, store: bool = False,
) -> dict[str, Any]:  # fmt: skip
    model = _shader_model(name, ports, out, body, params, helpers, store=store)
    return {
        "name": name,
        "type": "shader",
        "node_position": {"x": 0, "y": 0},
        "parameters": {p["name"]: p["default"] for p in model["parameters"]},
        "seed_int": 0,
        "shader_model": model,
    }


def _graph_atom(
    type_: str,
    label: str,
    description: str,
    ports: list[Port],
    out: str,
    inner: list[dict[str, Any]],
    wires: list[dict[str, Any]],
    inputs: list[tuple[str, int]],
    output: tuple[str, int],
    knobs: list[tuple[dict, list[tuple[str, str]]]],
) -> Atom:
    """Return a node of several passes: a graph of ``inner`` nodes.

    ``inputs`` are the inner (node, port) each of ``ports`` feeds, ``output``
    the inner one it outputs; ``knobs`` its parameters and the inner (node,
    widget) each sets.
    """
    links = [
        {"from": "gen_inputs", "from_port": k, "to": n, "to_port": p}
        for k, (n, p) in enumerate(inputs)
    ]
    links.append(
        {"from": output[0], "from_port": output[1], "to": "gen_outputs", "to_port": 0}
    )
    io = {"node_position": {"x": 0, "y": 0}, "parameters": {}, "type": "ios"}
    params = [p for p, _ in knobs]
    values = {p["name"]: p["default"] for p in params}
    remote = {
        "name": "gen_parameters",
        "type": "remote",
        "node_position": {"x": 0, "y": -200},
        "parameters": dict(values),
        "seed_int": 0,
        "widgets": [
            {"name": p["name"], "label": p["label"], "type": "linked_control",
             "linked_widgets": [{"node": n, "widget": w} for n, w in to]}
            for p, to in knobs
        ],
    }  # fmt: skip
    for k, n in enumerate(inner):
        n["node_position"] = {"x": 220 * k, "y": 0}
    definition = {
        "name": type_,
        "label": label,
        "shortdesc": label,
        "longdesc": description,
        "type": "graph",
        "node_position": {"x": 0, "y": 0},
        "parameters": values,
        "nodes": [
            *inner,
            {"name": "gen_inputs", **io,
             "ports": [{"name": p.name, "type": p.kind} for p in ports]},
            {"name": "gen_outputs", **io, "ports": [{"name": "output", "type": out}]},
            remote,
        ],
        "connections": [*wires, *links],
    }  # fmt: skip
    return Atom(type_, label, description, ports, out, params, definition)


def _blur(*, color: bool) -> Atom:
    t, _, sfx = _kinds(color=color)
    radius = f"$intensity * ({N} / 256.0)"
    knob = [_float("intensity", "Intensity", 10.0, 0.0, 32.0)]
    port = [Port("input", t, "Input", "")]
    x = _inner_shader(
        "pass_x", port, t, box_glsl_live(t, "vec2(1.0, 0.0)", radius, N), knob
    )
    y = _inner_shader(
        "pass_y", port, t, box_glsl_live(t, "vec2(0.0, 1.0)", radius, N), knob,
        store=True,
    )  # fmt: skip
    inner = [_plugin_buffer("buf_x"), x, _plugin_buffer("buf_y"), y]
    wires = [
        {"from": "buf_x", "from_port": 0, "to": "pass_x", "to_port": 0},
        {"from": "pass_x", "from_port": 0, "to": "buf_y", "to_port": 0},
        {"from": "buf_y", "from_port": 0, "to": "pass_y", "to_port": 0},
    ]
    sized = [("buf_x", "size_offset"), ("buf_y", "size_offset")]
    return _graph_atom(
        f"sbs_blur{sfx}", "Blur",
        "Substance's Blur: a separable box, half-width intensity / 256 of the"
        " image, its taps counted in the shader.",
        [Port("input1", t, "Input", "input")], t, inner, wires,
        [("buf_x", 0)], ("pass_y", 0),
        [(knob[0], [("pass_x", "intensity"), ("pass_y", "intensity")]),
         (FORMAT, [("pass_y", "format")]), (OUTPUTSIZE, sized)],
    )  # fmt: skip


def _distance(*, color: bool) -> Atom:
    """Distance: jump flooding (translate._distance) in an iterate buffer.

    Its iterations are the steps 2^12 down to 1, then 1 again; each larger
    than the translation's first step (from the distance, as it computes
    it) keeps the positions as they are, so the others are its passes.
    """
    t, _, sfx = _kinds(color=color)
    reach = _float("distance", "Maximum distance", 10.0, 0.0, 256.0)
    seed = _inner_shader(
        "seed", [Port("mask", "f", "Mask", "")], "rgba",
        "return $mask(uv) > 0.5 ? vec4(uv, 1.0, 1.0) : vec4(0.0);", [],
    )  # fmt: skip
    step = [
        "int i = int($iteration);",
        f"float N = {N};",
        "float rp = $distance * N / 256.0;",
        "float s = 1.0;",
        "for (int k = 0; k < 16 && s < min(rp, N / 2.0); k++) s *= 2.0;",
        f"float st = i <= {JFA_STEPS} ? float(1 << ({JFA_STEPS} - i)) : 1.0;",
        "if (st > s) return $input(uv);",
        "vec4 best = $input(uv);",
        "float bd = best.z > 0.5 ? tdist_$(name)(uv, best.xy) : 1e9;",
        "for (int y = -1; y <= 1; y++) {",
        "\tfor (int x = -1; x <= 1; x++) {",
        "\t\tvec2 o = vec2(float(x), float(y));",
        "\t\tvec4 c = $input(fract(uv + o * (st / N)));",
        "\t\tfloat d = tdist_$(name)(uv, c.xy);",
        "\t\tif (c.z > 0.5 && d < bd) { best = c; bd = d; }",
        "\t}",
        "}",
        "return best;",
    ]
    jfa = _inner_shader("jfa", [Port("input", "rgba", "Positions", "")], "rgba",
                        "\n".join(step), [reach], TORUS_GLSL)  # fmt: skip
    lines = [
        "vec4 c = $input(uv);",
        f"float r = max($distance * ({N} / 256.0), 1e-6);",
        f"float d = tdist_$(name)(uv, c.xy) * {N};",
        # The translation's distance is a number, which the GPU's compiler
        # divides by as a product by its reciprocal, rounded: so here too.
        "float f = c.z > 0.5 ? clamp(1.0 - d * $(name)_rcp(r), 0.0, 1.0) : 0.0;",
    ]
    if color:  # the falloff goes into alpha only
        lines += [
            "vec4 s = $source(f > 0.0 ? c.xy : uv);",
            "return vec4(s.rgb, s.a * f);",
        ]
    else:
        lines.append("return f * $source(c.xy);")
    source = Port("source", t, "Source", "", "vec4(1.0)" if color else "1.0")
    final = _inner_shader(
        "final", [Port("input", "rgba", "Positions", ""), source], t,
        "\n".join(lines), [reach], TORUS_GLSL + RCP_GLSL, store=True,
    )  # fmt: skip
    loop = {
        "name": "loop",
        "type": "iterate_buffer_v2",
        "node_position": {"x": 0, "y": 0},
        "parameters": {
            "states": 1, "size_mode": 1, "size": 9, "size_offset": 0.0,
            "iterations": JFA_STEPS + 1, "filter": False, "rounding": 0,
            "format_0": 1, "rg_0": False,
        },
        "seed_int": 0,
    }  # fmt: skip
    wires = [
        {"from": "seed", "from_port": 0, "to": "loop", "to_port": 0},
        {"from": "loop", "from_port": 1, "to": "jfa", "to_port": 0},
        {"from": "jfa", "from_port": 0, "to": "loop", "to_port": 1},
        {"from": "loop", "from_port": 0, "to": "final", "to_port": 0},
    ]
    return _graph_atom(
        f"sbs_distance{sfx}", "Distance",
        "Substance's Distance: 1 - d / distance from the nearest mask pixel"
        " (mask > 0.5), times the source there (in color, its alpha), by jump"
        " flooding; up to 8192 pixels.",
        [Port("mask", "f", "Mask input", "mask"),
         Port("source", t, "Source input", "source", source.default)],
        t, [seed, loop, jfa, final], wires, [("seed", 0), ("final", 1)], ("final", 0),
        [(reach, [("jfa", "distance"), ("final", "distance")]),
         (FORMAT, [("final", "format")]), (OUTPUTSIZE, [("loop", "size_offset")])],
    )  # fmt: skip


def _atoms() -> dict[str, Atom]:
    made = [
        _grayscale(), _shuffle(), _hsl(), _normal(),
        *(f(color=c) for c in (True, False) for f in (
            _uniform, _levels, _blend, _transformation, _dyngradient, _sharpen,
            _warp, _directionalwarp, _emboss, _dirmotionblur, _blur, _distance,
        )),
    ]  # fmt: skip
    return {a.type: a for a in made}


_ATOMS: dict[str, Atom] = {}


def atoms() -> dict[str, Atom]:
    """Return the plugin's node types by name."""
    if not _ATOMS:
        _ATOMS.update(_atoms())
    return _ATOMS


def output_kind(node_type: str) -> str:
    """Return a plugin node type's output type: "f" or "rgba"."""
    atom = atoms().get(node_type)
    return atom.output if atom is not None else "rgba"


def definitions() -> dict[str, dict[str, Any]]:
    """Return the plugin's node definitions (.mmg contents) by node type."""
    return {name: atom.definition for name, atom in atoms().items()}


DESCRIPTION = (
    "Nodes that match Substance Designer's atomic nodes (one Material Maker"
    " node per Substance node, with the same parameters), used by graphs"
    " converted by sbsar2ptex --experimental. Generated by sbsar2ptex"
    " (sbsar2ptex --install-mm-plugin)."
)
PLUGIN_CFG = f"""[plugin]

name="Substance atomic nodes"
description="{DESCRIPTION}"
version="{VERSION}"
nodes="nodes"
enabled_by_default=true
"""


def install(folder: Path) -> list[Path]:
    """Write the plugin in ``folder`` (its plugin.cfg and nodes); return the files.

    ``folder`` is the plugin's own (for Material Maker: its user folder's
    plugins/sbs).
    """
    nodes = folder / "nodes"
    nodes.mkdir(parents=True, exist_ok=True)
    cfg = folder / "plugin.cfg"
    cfg.write_text(PLUGIN_CFG, encoding="utf-8")
    written = [cfg]
    for name, definition in sorted(definitions().items()):
        path = nodes / f"{name}.mmg"
        path.write_text(json.dumps(definition, indent="\t") + "\n", encoding="utf-8")
        written.append(path)
    return written


# -- Converter ---------------------------------------------------------------


@dataclass
class _Driven:
    """A setting a Substance function computes: its code block's file offset."""

    offset: int


def _flag_settings(
    flags: int, inline: list[int], spec: list[tuple], width: int, node: Node
) -> dict[str, Any]:
    """Read settings as translate.flag_params does; functions as _Driven."""
    out: dict[str, Any] = {}
    at = 0
    for name, bit, default, *own in spec:
        w = own[0] if own else width
        state = (flags >> bit) & 3
        if state == 0:
            out[name] = default
        elif state == 1:
            vals = [f32(x) for x in inline[at : at + w]]
            at += w
            out[name] = vals if w > 1 else vals[0]
        else:
            offset, at = _dynamic_offset(node, inline, at)
            out[name] = _Driven(offset)
    return out


def _word_settings(
    node: Node, spec: list[tuple[str, int, Any]], widths: dict[str, int]
) -> dict[str, Any]:
    """Read settings as translate.word_params does; functions as _Driven."""
    out: dict[str, Any] = {}
    states = node.flags >> 8
    at = 0
    for name, bit, default in spec:
        state = (states >> bit) & 3
        width = widths.get(name, 1)
        if state == 0:
            out[name] = default
        elif state == 1:
            vals = [f32(w) for w in node.inline[at : at + width]]
            at += width
            out[name] = vals if width > 1 else vals[0]
        else:
            out[name] = _Driven(node.inline[at] + BASE)
            at += 1
    return out


class _Unpluggable(Exception):  # noqa: N818
    """A node a plugin node cannot be (a setting a function computes)."""


@dataclass
class _Knobs:
    """A plugin node's parameter values, and its settings that are graph inputs."""

    node: Node
    values: dict[str, Any] = field(default_factory=dict)
    links: dict[str, dict[str, Any]] = field(default_factory=dict)

    def _input(self, value: _Driven, kinds: Container[int]) -> Any:
        """Return the graph input ``value``'s function reads, if that is all it does."""
        node = self.node
        try:
            block = decode(node.at(value.offset, (node.end - value.offset) // 4))
        except (ValueError, IndexError) as e:
            raise _Unpluggable from e
        if len(block.instrs) != 1 or block.instrs[0].op != INPUT_OP:
            raise _Unpluggable
        gi = graph_input(block.instrs[0].u32s()[0])
        if gi is None or gi.type not in kinds:
            raise _Unpluggable
        return gi

    def _link(self, gi: Any, widgets: list[str]) -> list[Any]:
        """Link graph input ``gi``'s parameters to ``widgets``; return their values."""
        params, _ = input_params(gi)
        out = []
        for prm, widget in zip(params, widgets, strict=True):
            link = self.links.setdefault(prm["name"], {"decl": prm, "widgets": []})
            link["widgets"].append(widget)
            out.append(prm["default"])
        return out

    def float(self, name: str, value: Any) -> None:
        """Set float knob ``name``: a number, or a graph input's."""
        if isinstance(value, _Driven):
            gi = self._input(value, (IN_FLOAT,))
            (self.values[name],) = self._link(gi, [name])
        else:
            self.values[name] = float(value)

    def color(self, name: str, value: Any) -> None:
        """Set color knob ``name``: four numbers, or a float4 graph input's."""
        if isinstance(value, _Driven):
            gi = self._input(value, (IN_FLOAT4,))
            (self.values[name],) = self._link(gi, [name])
        else:  # a number is all four (a default: translate.knob's float)
            vals = value if isinstance(value, list) else [value] * 4
            r, g, b, a = (float(v) for v in vals)
            self.values[name] = {"r": r, "g": g, "b": b, "a": a}

    def floats(self, name: str, value: Any, width: int) -> None:
        """Set knobs ``name``_x, _y...: numbers, or a float vector graph input's."""
        widgets = [f"{name}_{'xyzw'[k]}" for k in range(width)]
        if isinstance(value, _Driven):
            if width not in (2, 3):  # a float4 input is one color parameter
                raise _Unpluggable
            gi = self._input(value, (width - 1,))  # float2 1, float3 2
            vals = self._link(gi, widgets)
        else:
            vals = [float(v) for v in value]
        self.values.update(zip(widgets, vals, strict=True))

    def value(self, name: str, value: Any, *, color: bool) -> None:
        """Set a knob that is a color in color, a float in grayscale."""
        if color:
            self.color(name, value)
        else:
            self.float(name, value)

    def plain(self, name: str, value: Any) -> None:
        """Set an enum or boolean knob (a constant)."""
        if isinstance(value, _Driven):
            raise _Unpluggable
        self.values[name] = value


def _read_levels(node: Node, k: _Knobs, *, color: bool) -> None:
    p = _flag_settings(node.fields[0], node.inline, LEVELS, 4 if color else 1, node)
    names = {"in_low": "levelinlow", "in_high": "levelinhigh", "in_mid": "levelinmid",
             "out_low": "leveloutlow", "out_high": "levelouthigh"}  # fmt: skip
    for key, v in p.items():
        k.value(names[key], v, color=color)


def _read_uniform(node: Node, k: _Knobs, *, color: bool) -> None:
    if (node.flags >> 8) & 3 == 2:  # noqa: PLR2004
        offset, _ = _dynamic_offset(node, node.inline, 0)
        k.value("outputcolor", _Driven(offset), color=color)
        return
    vals = [f32(w) for w in node.inline]
    if color:
        k.color("outputcolor", vals[:4] or [0.0, 0.0, 0.0, 1.0])
    else:
        k.float("outputcolor", vals[0] if vals else 0.0)


def _read_blend(node: Node, k: _Knobs, *, color: bool) -> None:
    flags = node.fields[0]
    mode = flags & 0xF
    if mode not in BLEND_MODES:
        raise _Unpluggable
    inline = list(node.inline)
    state = flags >> 4 & 3
    if state == 3:  # noqa: PLR2004 (an opacity input)
        inline = inline[1:]
        state = (flags >> 9) & 3
    if state == 2:  # noqa: PLR2004
        k.float("opacitymult", _Driven(inline[0] + BASE))
    else:
        k.float("opacitymult", f32(inline[0]) if state == 1 else 1.0)
    k.plain("blendingmode", mode)


def _read_transformation(node: Node, k: _Knobs, *, color: bool) -> None:
    flags = node.fields[0]
    matrix: Any = [1.0, 0.0, 0.0, 1.0]
    offset: Any = [0.0, 0.0]
    inline = list(node.inline)
    state_m, state_o = (flags >> 6) & 3, (flags >> 25) & 3
    if state_m == 1:
        matrix, inline = [f32(w) for w in inline[:4]], inline[4:]
    elif state_m == 2:  # noqa: PLR2004 (a float4: a color input, no knob of it)
        raise _Unpluggable
    if state_o == 1:
        offset = [f32(w) for w in inline[:2]]
    elif state_o == 2:  # noqa: PLR2004
        offset = _Driven(inline[0] + BASE)
    k.floats("matrix22", matrix, 4)
    k.floats("offset", offset, 2)
    k.plain("tiling", node.flags >> 8 & 3)


def _read_grayscale(node: Node, k: _Knobs, *, color: bool) -> None:
    p = _word_settings(
        node,
        [("weights", 0, [1 / 3, 1 / 3, 1 / 3, 0.0]), ("matte", 2, 0.0)],
        {"weights": 4},
    )
    k.color("channelsweights", p["weights"])
    k.float("mattelevel", p["matte"])
    k.plain("alphamult", bool(node.flags >> 8 & 0x10))


def _read_shuffle(node: Node, k: _Knobs, *, color: bool) -> None:
    for c, ch in enumerate(("red", "green", "blue", "alpha")):
        k.plain(f"channel{ch}", node.fields[0] >> (8 * c) & 0xFF)


def _read_hsl(node: Node, k: _Knobs, *, color: bool) -> None:
    p = _word_settings(
        node, [("hue", 0, 0.5), ("saturation", 2, 0.5), ("luminosity", 4, 0.5)], {}
    )
    for name, v in p.items():
        k.float(name, v)


def _read_dyngradient(node: Node, k: _Knobs, *, color: bool) -> None:
    p = _word_settings(node, [("coordinate", 1, 0.5)], {})
    k.float("coordinate", p["coordinate"])
    k.plain("uvselector", 1 if node.flags >> 8 & 0x08 else 0)


def _read_word_intensity(bit: int, default: float) -> Any:
    def read(node: Node, k: _Knobs, *, color: bool) -> None:
        p = _word_settings(node, [("intensity", bit, default)], {})
        k.float("intensity", p["intensity"])

    return read


def _read_dirmotionblur(node: Node, k: _Knobs, *, color: bool) -> None:
    spec = [("intensity", 0, 10.0), ("angle", 2, 0.0)]
    p = _flag_settings(node.fields[0], node.inline, spec, 1, node)
    k.float("intensity", p["intensity"])
    k.float("mblurangle", p["angle"])


def _read_directionalwarp(node: Node, k: _Knobs, *, color: bool) -> None:
    spec = [("intensity", 1, 10.0), ("angle", 3, 0.0)]
    p = _flag_settings(node.fields[0], node.inline, spec, 1, node)
    k.float("intensity", p["intensity"])
    k.float("warpangle", p["angle"])


def _read_normal(node: Node, k: _Knobs, *, color: bool) -> None:
    flags = node.fields[0]
    p = _flag_settings(flags & 3, node.inline, [("intensity", 0, 1.0)], 1, node)
    k.float("intensity", p["intensity"])
    k.plain("inversedy", bool(flags & 4))
    k.plain("input2alpha", bool(flags & 0x10))


def _read_emboss(node: Node, k: _Knobs, *, color: bool) -> None:
    n = 4 if color else 1
    p = _flag_settings(
        node.fields[0],
        node.inline,
        [
            ("intensity", 1, 5.0),
            ("angle", 3, 0.0),
            ("highlight", 5, [1.0] * n if color else 1.0, n),
            ("shadow", 7, [0.0] * n if color else 0.0, n),
        ],
        1,
        node,
    )
    k.float("intensity", p["intensity"])
    k.float("lightangle", p["angle"])
    k.value("highlightcolor", p["highlight"], color=color)
    k.value("shadowcolor", p["shadow"], color=color)


def _read_distance(node: Node, k: _Knobs, *, color: bool) -> None:
    p = _flag_settings(node.fields[0], node.inline, [("distance", 2, 10.0)], 1, node)
    k.float("distance", p["distance"])


@dataclass
class _Op:
    """How a decompiled op becomes a plugin node."""

    base: str
    read: Any
    sampled: tuple[str, ...] = ()
    """The node's inputs it reads around a pixel: buffered by the graph."""
    precise: tuple[str, ...] = ()
    """Those buffered in 32 bits."""
    variants: bool = True
    """Whether it has a grayscale variant (``_gray``)."""


OPS: dict[int, _Op] = {
    **dict.fromkeys((2, 3), _Op("blend", _read_blend)),
    **dict.fromkeys((4, 5), _Op("transformation", _read_transformation)),
    6: _Op("grayscaleconversion", _read_grayscale, variants=False),
    7: _Op("shuffle", _read_shuffle, variants=False),
    **dict.fromkeys((12, 13), _Op("uniform", _read_uniform)),
    **dict.fromkeys(
        (14, 15),
        _Op("warp", _read_word_intensity(5, 1.0), ("input", "gradient"), ("gradient",)),
    ),
    **dict.fromkeys(
        (16, 17), _Op("emboss", _read_emboss, ("gradient",), ("gradient",))
    ),
    **dict.fromkeys((20, 21), _Op("blur", _read_word_intensity(4, 10.0))),
    **dict.fromkeys((22, 23), _Op("dirmotionblur", _read_dirmotionblur, ("input",))),
    **dict.fromkeys(
        (24, 25), _Op("directionalwarp", _read_directionalwarp, ("input",))
    ),
    **dict.fromkeys(
        (26, 27), _Op("sharpen", _read_word_intensity(4, 1.0), ("input",), ("input",))
    ),
    29: _Op("hsl", _read_hsl, variants=False),
    **dict.fromkeys((30, 31), _Op("levels", _read_levels)),
    37: _Op("normal", _read_normal, ("input",), ("input",), variants=False),
    **dict.fromkeys((38, 39), _Op("dyngradient", _read_dyngradient)),
    **dict.fromkeys((42, 43), _Op("distance", _read_distance, ("source",))),
}
"""The ops that are plugin nodes (translate.TRANSLATORS's op numbers)."""


def prepared(node: Node, template: Node | None) -> Node:
    """Return the node whose settings are read, as translate.node_shader finds it.

    Without its seed's inline word (no plugin node uses a seed), from its
    library ``template`` if any, without a dynamic output size's word.
    """
    translated = node
    if (node.flags >> 6) & 3 == 1 and node.op not in (8, 9, 40, 41):
        translated = replace(node, inline=node.inline[1:])
    if template is not None:
        translated = replace(template, index=node.index, inputs=node.inputs)
        if (template.flags >> 6) & 3 == 1:
            translated = replace(translated, inline=template.inline[1:])
    if translated.inline:
        target = translated.inline[0] + BASE
        if translated.start < target < translated.end:
            block = decode(translated.at(target, (translated.end - target) // 4))
            last = block.instrs[-1]
            if last.type != FUNC_FLOAT and last.width == 2:  # noqa: PLR2004 (output size)
                translated = replace(translated, inline=translated.inline[1:])
    return translated


@dataclass
class Maker:
    """Makes translated nodes plugin nodes (translate.Plugins).

    ``size`` is the graph's (log2). ``loops`` are nodes a loop may compute
    (sbsar2ptex.loops folds their translation into an iterate buffer's
    body): they are plugin nodes of one pass only, no buffer of their own.
    ``made`` and ``inline`` count the nodes made plugin nodes, by type, and
    those that could not be, by reason.
    """

    size: int
    loops: Container[int] = frozenset()
    made: dict[str, int] = field(default_factory=dict)
    inline: dict[str, int] = field(default_factory=dict)

    def _skip(self, why: str) -> None:
        self.inline[why] = self.inline.get(why, 0) + 1

    def __call__(
        self, node: Node, name: str, size: int, template: Node | None
    ) -> Sub | None:
        """Return ``node`` as a plugin node (and its buffers), or None."""
        op = OPS.get(node.op)
        if op is None:
            self._skip(f"op {node.op}: no plugin node")
            return None
        color = node.op in COLOR_OPS
        type_ = PLUGIN_PREFIX + op.base + ("" if color or not op.variants else "_gray")
        atom = atoms()[type_]
        if node.index in self.loops and atom.definition["type"] != "shader":
            self._skip(f"{op.base}: in a loop (its buffers)")
            return None
        knobs = _Knobs(prepared(node, template))
        try:
            op.read(knobs.node, knobs, color=color)
        except (_Unpluggable, UnsupportedError):
            self._skip(f"{op.base}: a setting is a function")
            return None
        fmt = node.flags >> 4 & 3
        knobs.values["format"] = fmt
        if any(p["name"] == "outputsize" for p in atom.params):
            knobs.values["outputsize"] = float(size - self.size)
        made: dict[str, Any] = {
            "name": name,
            "type": type_,
            "node_position": {"x": 0, "y": 0},
            "parameters": knobs.values,
            "seed_int": 0,
        }
        if knobs.links:
            made[PLUGIN_LINKS] = knobs.links
        # Its inputs buffered as the inline translation buffers them.
        precise = set(op.precise)
        if op.base == "dirmotionblur" and fmt == 1:  # 16 bits: the input's levels
            precise.add("input")
        nodes, wires, ports = [made], [], []
        for port, p in enumerate(atom.ports):
            if p.key not in node.inputs:
                continue
            if p.key in op.sampled:
                buf = f"{name}_{p.key}_buf"
                nodes.append(buffer(buf, size, precise=p.key in precise))
                wires.append(wire(buf, name, port))
                ports.append((p.key, buf, 0))
            else:
                ports.append((p.key, name, port))
        self.made[type_] = self.made.get(type_, 0) + 1
        return Sub(nodes, wires, ports, name, rounded=fmt in (0, 1))
