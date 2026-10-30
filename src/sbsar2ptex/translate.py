"""Translate decompiled .sbsasm nodes into Material Maker shader nodes.

Each Substance atomic node becomes a Material Maker ``shader`` node whose
GLSL reproduces the filter; ``translate_graph`` wires them as the compiled
graph is wired. The formulas were fitted against sbsrender renders of
single-filter graphs (see tests/test_translate.py and sbsasm-probe):

- levels: v = clamp((x - in_low) / (in_high - in_low)), the range at least
  1e-4 wide; v = pow(v, g(mid)),
  g = 1 + 16 (mid - 0.5) above 0.5, 1 / (1 + 16 (0.5 - mid)) below; out =
  mix(out_low, out_high, v); every setting clamped to [0, 1] first.
- blend: out = clamp(d + (f(s, d) - d) * o), o = opacity (times the opacity
  input, and for color the source's alpha); color alpha is "over",
  sa o + da (1 - sa o); switch mixes rgba by o alone.
- transformation: samples the input at M^T (uv - 0.5) + 0.5 + offset,
  wrapping (matrix22 "a b c d" is M = [[a, b], [c, d]]).
- uniform: a constant (a grayscale one keeps the color's green).

Parameters driven by function bytecode ("dynamic") are translated to helper
functions, including Substance's context-specific random generators.
"""

import itertools
import json
import math
import re
import struct
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from pathlib import Path
from typing import Any
from typing import Protocol

from sbsar2ptex import fxdraw
from sbsar2ptex import fxglsl
from sbsar2ptex import fxmap
from sbsar2ptex import fxsplat
from sbsar2ptex.function import FLOAT as FUNC_FLOAT
from sbsar2ptex.function import FunctionError
from sbsar2ptex.function import bounds
from sbsar2ptex.function import decode
from sbsar2ptex.function import evaluate
from sbsar2ptex.function import glsl_type
from sbsar2ptex.function import to_glsl
from sbsar2ptex.fxrng import FX_RNG_GLSL
from sbsar2ptex.fxrng import PARAM_LCG
from sbsar2ptex.rng import RNG_GLSL
from sbsar2ptex.sbsasm import BASE
from sbsar2ptex.sbsasm import COLOR_OPS
from sbsar2ptex.sbsasm import Node
from sbsar2ptex.valuerng import VALUE_RNG_GLSL


class UnsupportedError(Exception):
    """A node (op or parameter) the translator cannot express yet."""


def f32(w: int) -> float:
    """Return the u32 ``w`` read as an IEEE float."""
    return struct.unpack("<f", struct.pack("<I", w))[0]


@dataclass
class GraphInput:
    """A Substance graph input: becomes a Material Maker parameter."""

    identifier: str
    type: int
    """The .sbsasm type: 0 float, 1 float2, 2 float3, 3 float4, 4 int, ..."""
    default: list[float | int]
    label: str = ""
    scope: str = ""
    """A library node instance's (libparams.knobs): its parameter is the
    instance's, INSTANCE_PARAM-named, not the graph's."""


@dataclass
class _Context:
    """What translating the current graph needs to know and collects."""

    size: int = 11
    """The output size (log2) of the node being translated: its functions'
    $size, its samples' grid; the graph's before the first."""
    inputs: dict[int, GraphInput] = field(default_factory=dict)
    helpers: list[str] = field(default_factory=list)
    """GLSL helper functions for the next shader node made."""
    globals: list[str] = field(default_factory=list)
    """Shared GLSL (Material Maker's global code) for the next shader node."""
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Material Maker parameters for the next shader node made."""
    exposed: dict[str, list[str]] = field(default_factory=dict)
    """Parameter name -> the MM nodes declaring it (for the remote node)."""
    sampled: list[str] | None = None
    """A pixel processor's input names by index, while translating it."""
    sampled_read: set[str] = field(default_factory=set)
    """Those of them a function has sampled."""
    sample_nearest: bool = False
    """Whether samples read the nearest texel whatever their filtering (an
    FX-Map's functions: check_filters fx_sampled_x_g on a sharp input)."""
    value_seed: str = "0"
    """Resolved random seed of the ordinary filter currently being translated."""
    pixel_seed: str = "0"
    """The pixel processor being translated's seed (GLSL): its own, the
    graph's unless set; its per-pixel generators start from it."""
    count: int = 0
    data_dir: Path | None = None
    fx_variables: dict[int, tuple[str, str]] | None = None
    """The FX-Map being translated's shared variables: GLSL global, type."""
    """Where precomputed FX-Map textures go (none: FX-Maps stay live)."""


_CTX = _Context()


class GLExpr:
    """A GLSL expression for a value computed by function bytecode.

    Arithmetic with numbers builds bigger expressions; ``static`` is the
    value with the graph inputs at their defaults, for uses that need a
    number when translating (a blur's tap count, say).
    """

    def __init__(self, text: str, width: int, static: list[float]) -> None:
        """Wrap GLSL ``text`` of ``width`` components."""
        self.text, self.width, self.static = text, width, static

    def _op(self, other: Any, fmt: str) -> "GLExpr":
        o = other.text if isinstance(other, GLExpr) else num(float(other))
        ostatic = other.static if isinstance(other, GLExpr) else [float(other)]
        a, b = self.static, ostatic
        if len(a) != len(b):
            a, b = (a * len(b), b) if len(a) == 1 else (a, b * len(a))
        ops = {"+": float.__add__, "-": float.__sub__, "*": float.__mul__, "/": _div}
        sym = fmt.split()[1]
        if fmt.startswith("{o}"):
            vals = [ops[sym](float(q), float(p)) for p, q in zip(a, b, strict=True)]
        else:
            vals = [ops[sym](float(p), float(q)) for p, q in zip(a, b, strict=True)]
        return GLExpr(
            "(" + fmt.format(s=self.text, o=o) + ")", max(self.width, len(b)), vals
        )

    def __add__(self, o: Any) -> "GLExpr":
        return self._op(o, "{s} + {o}")

    def __radd__(self, o: Any) -> "GLExpr":
        return self._op(o, "{o} + {s}")

    def __sub__(self, o: Any) -> "GLExpr":
        return self._op(o, "{s} - {o}")

    def __rsub__(self, o: Any) -> "GLExpr":
        return self._op(o, "{o} - {s}")

    def __mul__(self, o: Any) -> "GLExpr":
        return self._op(o, "{s} * {o}")

    def __rmul__(self, o: Any) -> "GLExpr":
        return self._op(o, "{o} * {s}")

    def __truediv__(self, o: Any) -> "GLExpr":
        return self._op(o, "{s} / {o}")

    def __neg__(self) -> "GLExpr":
        return GLExpr(f"(-{self.text})", self.width, [-v for v in self.static])

    def __getitem__(self, k: int) -> "GLExpr":
        """Return component ``k``."""
        return GLExpr(f"{self.text}.{'xyzw'[k]}", 1, [self.static[k]])

    def __iter__(self):
        """Iterate over the components."""
        return (self[k] for k in range(self.width))

    def __float__(self) -> float:
        """Return the static value (a scalar's)."""
        return float(self.static[0])


def _div(a: float, b: float) -> float:
    return a / b if b else math.inf


def static(v: Any) -> Any:
    """Return ``v`` as numbers: a GLExpr's value at the graph's defaults."""
    if isinstance(v, GLExpr):
        return v.static if v.width > 1 else v.static[0]
    return v


_INPUT_GLSL = {0: 1, 1: 2, 2: 3, 3: 4, 4: 1, 8: 2, 9: 3, 10: 4}
"""Components of each graph input type."""


IN_FLOAT, IN_FLOAT4, IN_INT, IN_INT2 = 0, 3, 4, 8
"""Graph input types (.sbsasm) the parameters below distinguish."""


def _float_decl(
    name: str, label: str, default: float, *, integer: bool
) -> dict[str, Any]:
    """Return a float parameter's declaration (a slider from 0 to 1, or wider)."""
    return {
        "name": name,
        "label": label,
        "type": "float",
        "default": default,
        "min": min(0.0, default),
        "max": max(1.0, default),
        "step": 1.0 if integer else 0.001,
    }


GRAPH_PARAM = "p_"
"""Graph inputs' parameter names start so; a node's own knobs do not."""
INSTANCE_PARAM = "i_"
"""A library node instance's parameters' names start so (GraphInput.scope):
shared by the instance's nodes, shown on the instance, not the graph."""


def knob(
    pname: str, label: str, value: Any, lo: float | None = None, hi: float | None = None
) -> Any:
    """Return a node setting as a Material Maker parameter of the next shader.

    A constant becomes an editable parameter, named ``pname`` (a float, a
    color for four values in [0, 1], else a float per component), returned
    as a GLExpr reading it; a function-driven value (a GLExpr) stays as it
    is. ``lo`` and ``hi`` bound the slider (default: 0 to 1, widened to
    the value).
    """
    if isinstance(value, GLExpr):
        return value
    if isinstance(value, (list, tuple)):
        vals = [float(v) for v in value]
        if len(vals) == 4 and all(0.0 <= v <= 1.0 for v in vals):  # noqa: PLR2004
            r, g, b, a = vals
            _CTX.params[pname] = {
                "name": pname,
                "label": label,
                "type": "color",
                "default": {"r": r, "g": g, "b": b, "a": a},
            }
            return GLExpr(f"${pname}", 4, vals)
        parts = []
        for k, v in enumerate(vals):
            part = f"{pname}_{'xyzw'[k]}"
            knob(part, f"{label} {'xyzw'[k]}", v, lo, hi)
            parts.append(f"${part}")
        return GLExpr(f"vec{len(vals)}({', '.join(parts)})", len(vals), vals)
    v = float(value)
    low = min(0.0 if lo is None else lo, v)
    high = max(1.0 if hi is None else hi, v)
    _CTX.params[pname] = {
        "name": pname,
        "label": label,
        "type": "float",
        "default": v,
        "min": low,
        "max": high,
        "step": 0.001,
    }
    return GLExpr(f"${pname}", 1, [v])


def param_name(identifier: str) -> str:
    """Return a graph input's Material Maker parameter name.

    That is p_ and its word characters, with single underscores (GLSL
    reserves double ones).
    """
    word = "".join(c if c.isalnum() else "_" for c in identifier)
    return "p_" + "_".join(part for part in word.split("_") if part)


def input_param(gi: GraphInput) -> str:
    """Return the Material Maker parameter name of graph input ``gi``.

    A library node instance's (its scope) is INSTANCE_PARAM-named.
    """
    name = param_name(gi.identifier)
    if gi.scope:
        return INSTANCE_PARAM + gi.scope + "_" + name.removeprefix(GRAPH_PARAM)
    return name


def _param_for(uid: int) -> str:
    """Declare graph input ``uid`` as a parameter of the next shader; return GLSL.

    Floats and ints become float parameters, float4s colors; other vectors a
    float parameter per component.
    """
    gi = _CTX.inputs.get(uid)
    if gi is None:
        msg = f"function reads unknown input {uid:#x}"
        raise UnsupportedError(msg)
    params, glsl = input_params(gi)
    for prm in params:
        _CTX.params[prm["name"]] = prm
    return glsl


def graph_input(uid: int) -> GraphInput | None:
    """Return the input ``uid`` of the graph (or library node) being translated."""
    return _CTX.inputs.get(uid)


def input_params(gi: GraphInput) -> tuple[list[dict[str, Any]], str]:
    """Return graph input ``gi`` as Material Maker parameters, and GLSL reading it.

    Floats and ints are a float parameter, float4s a color; other vectors a
    float parameter per component (named with _x, _y...).
    """
    integer = gi.type >= IN_INT
    name = input_param(gi)
    label = gi.label or gi.identifier
    if gi.type == IN_FLOAT4:
        r, g, b, a = (list(gi.default) + [0.0] * 4)[:4]
        color = {
            "name": name,
            "label": label,
            "type": "color",
            "default": {"r": r, "g": g, "b": b, "a": a},
        }
        return [color], f"${name}"
    if gi.type in (IN_FLOAT, IN_INT):
        default = float(gi.default[0]) if gi.default else 0.0
        prm = _float_decl(name, label, default, integer=integer)
        return [prm], f"int(${name})" if integer else f"${name}"
    width = _INPUT_GLSL.get(gi.type, 1)
    parts, params = [], []
    for k in range(width):
        pname = f"{name}_{'xyzw'[k]}"
        default = float(gi.default[k]) if k < len(gi.default) else 0.0
        params.append(
            _float_decl(pname, f"{label} {'xyzw'[k]}", default, integer=integer)
        )
        parts.append(f"int(${pname})" if integer else f"${pname}")
    glsl = f"{'ivec' if gi.type >= IN_INT2 else 'vec'}{width}({', '.join(parts)})"
    return params, glsl


def _sample(kind: str, index: int, pos: str, filtering: int) -> str:
    """Return GLSL sampling a pixel processor's input ``index`` at ``pos``.

    Filtering 0 is bilinear (the inputs are buffered with filtering);
    otherwise, and always in an FX-Map's functions, the nearest texel.
    Positions wrap.
    """
    if _CTX.sampled is None or index >= len(_CTX.sampled):
        msg = f"sample of input {index} outside a pixel processor"
        raise UnsupportedError(msg)
    n = num(float(1 << _CTX.size))
    at = f"fract({pos})"
    if filtering or _CTX.sample_nearest:
        at = f"(floor(fract({pos}) * {n}) + 0.5) / {n}"
    _CTX.sampled_read.add(_CTX.sampled[index])
    value = f"${_CTX.sampled[index]}({at})"
    # samplelum is the mean of R, G and B (a grayscale input's value).
    return f"dot({value}.rgb, vec3(1.0 / 3.0))" if kind == "lum" else value


def _rand(a: str, k: int) -> str:
    """Return GLSL for Substance's rand(a): uniform in [0, a).

    In a pixel processor, Substance's own generator (docs/random-generator.md,
    sbsar2ptex.rng): a per-pixel state, ``rng``, advanced by each call. Its
    An ordinary value function instead uses its node's resolved random seed
    and the once-per-node generator from docs/random-generator.md section 11.
    """
    if _CTX.sampled is not None:
        return f"sbs_rng_next(rng, float({a}))"
    return f"sbs_value_rng_next(rng, float({a}))"


RAND = 0x32
"""The function bytecode op of rand."""
SETVAR = 0x07
"""The function bytecode op setting a variable."""


def _seed_glsl() -> str:
    """Return GLSL for the graph's random seed: its $randomseed input, else 0."""
    for uid, gi in _CTX.inputs.items():
        if gi.identifier == "$randomseed":
            return _param_for(uid)
    return "0"


def dynamic(node: Node, offset: int, before: Sequence[int] = ()) -> GLExpr:
    """Return the value of the function block at file ``offset`` in ``node``.

    The blocks at ``before`` (a pixel processor's parameters) run first, once
    per node in Substance, with the node's value generator; the variables
    they set are the block's to read.
    """
    block = decode(node.at(offset, (node.end - offset) // 4))
    _CTX.count += 1
    prefix = f"f{_CTX.count}"
    n = float(1 << _CTX.size)
    builtin_glsl = {
        1: f"vec2({num(n)})",
        3: f"vec2({num(float(_CTX.size))})",
        8: "uv",  # $pos: the pixel, in a pixel processor
    }
    builtin_vals = {1: [n, n], 3: [float(_CTX.size)] * 2}

    def builtins(ident: int, width: int) -> str:
        if ident not in builtin_glsl:
            msg = f"function reads built-in {ident}"
            raise UnsupportedError(msg)
        return builtin_glsl[ident]

    def param(uid: int, *_: int) -> str:
        return _param_for(uid)

    shared: dict[int, tuple[str, str]] = {}
    first: list[str] = []
    try:
        for at in before:
            pre = decode(node.at(at, (node.end - at) // 4))
            sets = [i for i in pre.instrs if i.op == SETVAR]
            if not sets:
                continue
            for i in sets:
                shared.setdefault(
                    i.operands[1],
                    (f"{prefix}_pv{i.operands[1]}", glsl_type(i.type, i.width)),
                )
            _CTX.count += 1
            pre_lines, _ = to_glsl(
                pre, f"f{_CTX.count}", param, builtins, None,
                lambda a, _k: f"sbs_value_rng_next(rng, float({a}))", shared,
            )  # fmt: skip
            if any(i.op == RAND for i in pre.instrs):
                pre_lines.insert(0, f"uint rng = uint({_CTX.value_seed});")
                if VALUE_RNG_GLSL not in _CTX.globals:
                    _CTX.globals.append(VALUE_RNG_GLSL)
            first += ["{", *pre_lines, "}"]
        first[:0] = [f"{t} {v} = {t}(0);" for v, t in shared.values()]
        lines, result = to_glsl(
            block, prefix, param, builtins, _sample, _rand, shared or None
        )
        values = {
            uid: [float(x) for x in gi.default] or [0.0]
            for uid, gi in _CTX.inputs.items()
        }
    except FunctionError as e:
        raise UnsupportedError(str(e)) from e
    try:
        value = evaluate(block, values, builtin_vals)
    except (FunctionError, KeyError):  # per-pixel: no single value
        value = [0.0] * block.instrs[-1].width
    last = block.instrs[-1]
    t = glsl_type(last.type, last.width)
    if _CTX.sampled is not None and any(i.op == RAND for i in block.instrs):
        # rand: the pixel's generator, from the graph seed and the pixel.
        lines.insert(0, f"uint rng = sbs_rng_init(uint({_CTX.pixel_seed}), uv);")
        if RNG_GLSL not in _CTX.globals:
            _CTX.globals.append(RNG_GLSL)
    elif any(i.op == RAND for i in block.instrs):
        lines.insert(0, f"uint rng = uint({_CTX.value_seed});")
        if VALUE_RNG_GLSL not in _CTX.globals:
            _CTX.globals.append(VALUE_RNG_GLSL)
    body = "\n\t".join(first + lines)
    # Material Maker appends seed arguments to instance functions, so they
    # need a parameter list of their own.
    _CTX.helpers.append(
        f"{t} {prefix}_$(name)(vec2 uv) {{\n\t{body}\n\treturn {result};\n}}\n"
    )
    call = f"{prefix}_$(name)(uv)"
    if last.type != FUNC_FLOAT:
        call = f"float({call})" if last.width == 1 else f"vec{last.width}({call})"
    return GLExpr(call, last.width, [float(x) for x in value])


def _dynamic_offset(node: Node, inline: list[int], at: int) -> tuple[int, int]:
    """Return a dynamic parameter's function block (file offset) and next inline index.

    The parameter's inline word is the block's offset. Older cookers (2018)
    put a neighborhood filter's function-driven intensity in its second
    code block instead, with no inline word: then the word at ``at`` is not
    an offset into the node, and it stays for the next parameter.
    """
    if at < len(inline):
        target = inline[at] + BASE
        if node.start < target < node.end and inline[at] % 4 == 0:
            return target, at + 1
    if len(node.code) > 1:
        return node.code[-1], at
    msg = "no function block for a dynamic parameter"
    raise UnsupportedError(msg)


def flag_params(
    flags: int,
    inline: list[int],
    spec: list[tuple],
    width: int,
    node: Node | None = None,
) -> dict[str, Any]:
    """Read parameters stored two flag bits each: 1 inline, 2 dynamic.

    ``spec`` lists (name, bit position, default[, width]); each inline
    parameter takes its width (else ``width``) floats, in spec order; a
    dynamic one, one word: the offset of its function block in ``node``.
    """
    out: dict[str, Any] = {}
    at = 0
    for name, bit, default, *own in spec:
        width = own[0] if own else width
        state = (flags >> bit) & 3
        if state == 0:
            out[name] = default
        elif state == 1:
            vals = [f32(w) for w in inline[at : at + width]]
            at += width
            out[name] = vals if width > 1 else vals[0]
        else:
            if node is None:
                msg = f"parameter {name} is driven by a function"
                raise UnsupportedError(msg)
            offset, at = _dynamic_offset(node, inline, at)
            out[name] = dynamic(node, offset)
    return out


LEVELS = [
    ("in_low", 0, 0.0),
    ("in_high", 2, 1.0),
    ("in_mid", 4, 0.5),
    ("out_low", 6, 0.0),
    ("out_high", 8, 1.0),
]
BLEND_NAMES = {
    0: "Copy", 1: "Add", 2: "Subtract", 3: "Multiply", 4: "Add sub", 5: "Max",
    6: "Min", 7: "Switch", 8: "Divide", 9: "Overlay", 10: "Screen",
    11: "Soft light",
}  # fmt: skip
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


def num(v: Any) -> str:
    """Return the shortest decimal that reads back as float32 ``v``.

    A GLExpr is its GLSL (a vector's green, as a grayscale node's constant).
    """
    if isinstance(v, GLExpr):
        return v.text if v.width == 1 else f"{v.text}.y"
    target = struct.pack("<f", v)
    for digits in range(1, 10):
        text = f"{v:.{digits}g}"
        if struct.pack("<f", float(text)) == target:
            break
    else:
        text = repr(v)
    # GLSL (Godot's, strictly) wants float literals to look like floats.
    return text if any(c in text for c in ".en") else f"{text}.0"


def vec(values: Any, n: int = 4) -> str:
    """Return a GLSL vecN (or float) literal, or a GLExpr shaped to N."""
    if isinstance(values, GLExpr):
        if values.width == n:
            return values.text
        if values.width == 1:
            return values.text if n == 1 else f"vec{n}({values.text})"
        if n == 1:
            return f"{values.text}.y"
        if values.width < n:  # fill out: zeros, and an opaque alpha
            fill = ["0.0"] * (n - values.width)
            if n == 4:  # noqa: PLR2004
                fill[-1] = "1.0"
            return f"vec{n}({values.text}, {', '.join(fill)})"
        return f"{values.text}.{'xyzw'[:n]}"
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
    helpers, params = "".join(_CTX.helpers), list(_CTX.params.values())
    shared = "".join(_CTX.globals)
    _CTX.helpers, _CTX.params, _CTX.globals = [], {}, []
    for prm in params:  # graph inputs only: a node's knobs stay its own
        if prm["name"].startswith(GRAPH_PARAM):
            _CTX.exposed.setdefault(prm["name"], []).append(name)
    return {
        "name": name,
        "type": "shader",
        "node_position": {"x": 0, "y": 0},
        "parameters": {prm["name"]: prm["default"] for prm in params},
        "seed_int": 0,
        "shader_model": {
            "name": name,
            "code": "",
            "global": shared,
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
            "instance": f"{helpers}{ret} $(name)_fct(vec2 uv) {{\n{body}\n}}\n",
            "outputs": [{"type": out_type, out_type: "$(name)_fct($uv)"}],
            "parameters": params,
        },
    }


@dataclass
class Sub:
    """A node translated to several Material Maker nodes.

    ``ports`` wires the Substance node's inputs: (input name, MM node, port);
    ``output`` is the MM node whose port 0 is the result.
    """

    nodes: list[dict[str, Any]]
    wires: list[dict[str, Any]]
    ports: list[tuple[str, str, int]]
    output: str
    rounded: bool = False
    """Its output is stored in 8 or 16 bits already (a plugin node's format;
    stored() makes shaders so)."""


PLUGIN_PREFIX = "sbs_"
"""The sbs plugin's node types start so (sbsar2ptex.plugin)."""
PLUGIN_LINKS = "sbs_inputs"
"""A plugin node's settings that are graph inputs, while translating: graph
parameter -> {"decl": its declaration, "widgets": the node's knobs it sets}
(the knobs have the plugin's names, not the parameter's)."""


def is_plugin(node: dict[str, Any]) -> bool:
    """Whether ``node`` is an sbs plugin node (sbsar2ptex.plugin)."""
    return str(node.get("type", "")).startswith(PLUGIN_PREFIX)


def widgets_of(node: dict[str, Any] | None, p: str) -> list[str]:
    """Return the widgets of ``node`` a remote sets for its parameter ``p``.

    A shader's parameter is its own widget; a plugin node's graph inputs
    set its knobs (PLUGIN_LINKS).
    """
    links = (node or {}).get(PLUGIN_LINKS, {})
    return list(links[p]["widgets"]) if p in links else [p]


def strip_links(graph: dict[str, Any]) -> None:
    """Drop PLUGIN_LINKS from ``graph``'s nodes, its subgraphs' too."""
    for n in graph["nodes"]:
        n.pop(PLUGIN_LINKS, None)
        if n["type"] == "graph":
            strip_links(n)


class Plugins(Protocol):
    """Makes a node an sbs plugin node (sbsar2ptex.plugin.Maker)."""

    def __call__(  # noqa: D102
        self, node: Node, name: str, size: int, template: Node | None
    ) -> Sub | None: ...


class Loop(Protocol):
    """Library nodes made one Material Maker node (sbsar2ptex.loops).

    ``members`` are the nodes it computes, ``inputs`` the nodes it reads,
    by its Sub's port names.
    """

    @property
    def members(self) -> set[int]: ...  # noqa: D102

    inputs: dict[str, int]

    def make(self, name: str, nodes: list[Node], size: int) -> Sub:
        """Return it as Material Maker nodes, ``size`` (log2) square."""
        ...

    def rounds(self, nodes: list[Node]) -> bool:
        """Return whether its image is stored in 8 or 16 bits (stored)."""
        ...


class Fold(Protocol):
    """A pass over a whole translation (sbsar2ptex.loops.Chains).

    Given the Material Maker nodes and connections, each one's Substance
    node (``origins``), each Substance node's Material Maker node
    (``out_of``, updated in place) and the graph inputs' users (``exposed``,
    updated in place), it returns the nodes and connections it made of them.
    """

    def __call__(  # noqa: D102
        self,
        mm: list[dict[str, Any]],
        wires: list[dict[str, Any]],
        origins: dict[str, int],
        out_of: dict[int, str],
        exposed: dict[str, list[str]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]: ...


def buffer(
    name: str, size: int, *, filtering: bool = True, precise: bool = False
) -> dict[str, Any]:
    """Return a Material Maker buffer node: its input rendered to a texture.

    Filters that read neighboring pixels sample buffers, so their input's
    graph is computed once, not once per tap. 16-bit float (images need no
    more), or 32-bit when ``precise`` (what slopes are taken of: half
    floats' steps would show in a normal map) or unfiltered (positions);
    linear filtering unless not ``filtering``. Version 2: the version-1
    buffer Material Maker reads otherwise ignores both (always filtered,
    always 16-bit).
    """
    return {
        "name": name,
        "type": "buffer",
        "node_position": {"x": 0, "y": 0},
        "parameters": {
            "size": size,
            "f32": precise or not filtering,
            "filter": filtering,
            "mipmap": False,
        },
        "seed_int": 0,
        "version": 2,
    }


def cache_shared(
    mm: list[dict[str, Any]],
    wires: list[dict[str, Any]],
    size: int,
    origins: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``mm`` and ``wires`` with nodes several others read buffered.

    Material Maker builds one shader from everything up to a buffer, each
    node a function called wherever it is read: a node read along several
    paths is computed once per path, per pixel, which multiplies through a
    graph whose paths keep meeting again (hours for a material of
    thousands of nodes). A buffer computes it once: its existing (filtered)
    buffer if it has one, else a new one, which all its readers then read.
    The buffer is 32-bit when the node's image reaches a 32-bit buffer
    before any other (a slope is taken of it downstream), else 16-bit.
    """
    by_name = {n["name"]: n for n in mm}
    out_of: dict[str, list[dict[str, Any]]] = {}
    for w in wires:
        out_of.setdefault(w["from"], []).append(w)

    def kind(name: str) -> str:
        node = by_name.get(name, {})
        return "graph" if is_plugin(node) else node.get("type", "")

    def filtered_buffer(name: str) -> bool:
        return kind(name) == "buffer" and by_name[name]["parameters"]["filter"]

    precise_memo: dict[str, bool] = {}

    def precise(name: str) -> bool:
        """Whether ``name``'s image reaches a 32-bit buffer unbuffered."""
        if name not in precise_memo:
            precise_memo[name] = False  # (a DAG; this guards recursion)
            precise_memo[name] = any(
                by_name[w["to"]]["parameters"]["f32"]
                if kind(w["to"]) == "buffer"
                else kind(w["to"]) in ("shader", "graph") and precise(w["to"])
                for w in out_of.get(name, [])
            )
        return precise_memo[name]

    for name in [n["name"] for n in mm if kind(n["name"]) in ("shader", "graph")]:
        for port in sorted({w["from_port"] for w in out_of.get(name, [])}):
            ws = [w for w in out_of[name] if w["from_port"] == port]
            if len({w["to"] for w in ws}) < 2:  # noqa: PLR2004
                continue
            caches = [w["to"] for w in ws if filtered_buffer(w["to"])]
            if caches:
                cache = caches[0]
            else:
                cache = f"{name}_cache{port or ''}"
                made = buffer(cache, size, precise=precise(name))
                mm = [*mm, made]
                by_name[cache] = made
                wires = [
                    *wires,
                    {"from": name, "from_port": port, "to": cache, "to_port": 0},
                ]
                if origins is not None and name in origins:
                    origins[cache] = origins[name]
            for w in ws:
                if kind(w["to"]) != "buffer":  # unfiltered buffers: positions
                    w["from"], w["from_port"] = cache, 0
    return mm, wires


def merge_buffers(
    mm: list[dict[str, Any]], wires: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``mm`` and ``wires`` with one buffer per buffered source.

    Each filter buffers its own inputs, so a node several filters read gets
    as many buffers; they are the same image (the same source, size and
    filtering), so all but one go, and the one kept is 32-bit if any was.
    """
    by_name = {n["name"]: n for n in mm}
    source: dict[str, tuple[str, int]] = {}
    for w in wires:
        if by_name.get(w["to"], {}).get("type") == "buffer":
            source[w["to"]] = (w["from"], w["from_port"])
    kept: dict[tuple, str] = {}
    gone: dict[str, str] = {}
    for name, src in source.items():
        prm = by_name[name]["parameters"]
        key = (src, prm["size"], prm["filter"])
        if key in kept:
            gone[name] = kept[key]
            first = by_name[kept[key]]["parameters"]
            first["f32"] = first["f32"] or prm["f32"]
        else:
            kept[key] = name
    out_wires = []
    for w in wires:
        if w["to"] in gone:
            continue
        if w["from"] in gone:
            w = {**w, "from": gone[w["from"]]}  # noqa: PLW2901
        out_wires.append(w)
    return [n for n in mm if n["name"] not in gone], out_wires


def _forget(gone: dict[str, str], origins: dict[str, int] | None) -> None:
    """Point the graph's exposed parameters at the nodes ``gone`` keeps instead."""
    for p, users in _CTX.exposed.items():
        _CTX.exposed[p] = list(dict.fromkeys(gone.get(u, u) for u in users))
    if origins is not None:
        for name in gone:
            origins.pop(name, None)


def merge_same(
    mm: list[dict[str, Any]],
    wires: list[dict[str, Any]],
    origins: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """Return ``mm`` and ``wires`` with identical nodes merged, and what went where.

    Substance graphs repeat work (a library node's copies of one transform of
    one input, an FX-Map's pattern inputs all one image); nodes of the same
    code and parameters reading the same inputs compute the same image, so
    one is kept and read in their place. Taken from the inputs on, so that
    nodes reading merged ones merge too.
    """
    into: dict[str, list[dict[str, Any]]] = {}
    for w in wires:
        into.setdefault(w["to"], []).append(w)
    rep: dict[str, str] = {}
    seen: dict[tuple[str, tuple], str] = {}
    done: set[str] = set()
    by_name = {n["name"]: n for n in mm}
    for start in by_name:
        todo = [start]
        while todo:  # inputs first, iteratively: graphs run thousands deep
            name = todo[-1]
            if name in done:
                todo.pop()
                continue
            pending = [w["from"] for w in into.get(name, []) if w["from"] not in done]
            if pending:
                todo += pending
                continue
            todo.pop()
            done.add(name)
            node = by_name[name]
            body = {k: v for k, v in node.items() if k not in ("name", "node_position")}
            if "shader_model" in body:
                body["shader_model"] = {
                    k: v for k, v in body["shader_model"].items() if k != "name"
                }
            ins = tuple(
                sorted(
                    (w["to_port"], rep.get(w["from"], w["from"]), w["from_port"])
                    for w in into.get(name, [])
                )
            )
            key = (json.dumps(body, sort_keys=True), ins)
            if key in seen:
                rep[name] = seen[key]
            else:
                seen[key] = name
    out_wires = []
    for w in wires:
        if w["to"] in rep:
            continue
        if w["from"] in rep:
            w = {**w, "from": rep[w["from"]]}  # noqa: PLW2901
        out_wires.append(w)
    _forget(rep, origins)
    return [n for n in mm if n["name"] not in rep], out_wires, rep


def _copies(buffer_node: dict[str, Any], source: dict[str, Any]) -> bool:
    """Whether ``buffer_node``, reading buffer ``source``, holds the same image.

    Rendered at the same size it samples each texel at its center: the same
    values, if it is no less precise; read the same way if it filters alike.
    """
    b, s = buffer_node["parameters"], source["parameters"]
    return (
        b["size"] == s["size"]
        and b["filter"] == s["filter"]
        and (b["f32"] or not s["f32"])
    )


def drop_copies(
    mm: list[dict[str, Any]],
    wires: list[dict[str, Any]],
    origins: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """Return ``mm`` and ``wires`` without buffers copying a buffer (_copies).

    A filter buffers its input, and the input is often a buffer already (a
    cache, another filter's): its readers read that one instead. Inside a
    subgraph too (a blur's first buffer, when what feeds the blur is one).
    """
    by_name = {n["name"]: n for n in mm}
    src: dict[str, list[dict[str, Any]]] = {}
    for w in wires:
        src.setdefault(w["to"], []).append(w)

    def buffer_at(name: str) -> dict[str, Any] | None:
        node = by_name.get(name)
        return node if node is not None and node["type"] == "buffer" else None

    gone: dict[str, str] = {}
    for name, node in by_name.items():
        ws = src.get(name, [])
        if node["type"] == "buffer" and len(ws) == 1:
            first = buffer_at(gone.get(ws[0]["from"], ws[0]["from"]))
            if first is not None and _copies(node, first):
                gone[name] = first["name"]
        elif node["type"] == "graph":
            for w in ws:
                first = buffer_at(gone.get(w["from"], w["from"]))
                if first is not None:
                    _drop_inner_copy(node, w["to_port"], first)

    def final(name: str) -> str:
        while name in gone:
            name = gone[name]
        return name

    out_wires = []
    for w in wires:
        if w["to"] in gone:
            continue
        if w["from"] in gone:
            w = {**w, "from": final(w["from"])}  # noqa: PLW2901
        out_wires.append(w)
    moved = {k: final(k) for k in gone}
    _forget(moved, origins)
    return [n for n in mm if n["name"] not in gone], out_wires, moved


def _drop_inner_copy(graph: dict[str, Any], port: int, source: dict[str, Any]) -> None:
    """Drop ``graph``'s first buffer on input ``port`` if it copies ``source``."""
    inner = {n["name"]: n for n in graph["nodes"]}
    links = graph["connections"]
    outs = [w for w in links if w["from"] == "gen_inputs" and w["from_port"] == port]
    if len(outs) != 1 or inner.get(outs[0]["to"], {}).get("type") != "buffer":
        return
    buf = outs[0]["to"]
    if not _copies(inner[buf], source):
        return
    graph["nodes"] = [n for n in graph["nodes"] if n["name"] != buf]
    graph["connections"] = [
        {**w, "from": "gen_inputs", "from_port": port} if w["from"] == buf else w
        for w in links
        if w["to"] != buf
    ]


def wire(src: str, dst: str, port: int = 0) -> dict[str, Any]:
    """Return a connection from ``src``'s port 0 to ``dst``'s ``port``."""
    return {"from": src, "from_port": 0, "to": dst, "to_port": port}


def buffered(
    made: dict[str, Any],
    names: list[str],
    sampled: set[str],
    size: int,
    precise: Iterable[str] = (),
) -> Sub:
    """Return ``made`` (a shader node) with buffers before its ``sampled`` inputs.

    ``precise`` inputs get 32-bit buffers (see buffer).
    """
    nodes, wires, ports = [made], [], []
    for port, inp in enumerate(names):
        if inp in sampled:
            buf = f"{made['name']}_{inp}_buf"
            nodes.append(buffer(buf, size, precise=inp in precise))
            wires.append(wire(buf, made["name"], port))
            ports.append((inp, buf, 0))
        else:
            ports.append((inp, made["name"], port))
    return Sub(nodes, wires, ports, made["name"])


def box_taps(radius: float) -> list[tuple[float, float]]:
    """Return (offset, weight) taps of Substance's box blur of half-width ``radius``.

    The exact integral over [-radius, radius] of the linearly interpolated
    signal, normalized: interior taps 1, the end taps by the trapezoid.
    """
    radius = max(radius, 1e-6)
    n = int(radius)
    t = radius - n
    taps = [(float(k), 1.0) for k in range(-n + 1, n)]
    if n == 0:
        taps = [(0.0, 2 * t - t * t)]
    else:
        end = 0.5 + t - t * t / 2
        taps += [(float(-n), end), (float(n), end)]
    if t > 0:
        taps += [(float(-n - 1), t * t / 2), (float(n + 1), t * t / 2)]
    total = sum(w for _, w in taps)
    return [(k, w / total) for k, w in taps]


def box_glsl_live(t: str, direction: str, radius: str, n: str) -> str:
    """Return a GLSL body: the box of box_taps, its radius (texels) a GLSL value.

    ``direction`` is a GLSL vec2 in texels, ``n`` the image width (a GLSL
    float). The taps are counted in the shader, so the radius can be a
    parameter or a function.
    """
    g = "vec4" if t == "rgba" else "float"
    return "\n".join(
        [
            f"{g} acc = {g}(0.0);",
            f"float r = max({radius}, 1e-6);",
            "int n = int(r);",
            "float t = r - float(n);",
            f"vec2 d = {direction} / {n};",
            "for (int i = -n + 1; i < n; i++) {",
            "\tacc += $input(fract(uv + float(i) * d));",
            "}",
            "float total = float(max(2 * n - 1, 0));",
            "if (n == 0) {",
            "\tfloat w = 2.0 * t - t * t;",
            "\tacc += w * $input(uv);",
            "\ttotal += w;",
            "} else {",
            "\tfloat w = 0.5 + t - 0.5 * t * t;",
            "\tvec2 o = float(n) * d;",
            "\tacc += w * ($input(fract(uv - o)) + $input(fract(uv + o)));",
            "\ttotal += 2.0 * w;",
            "}",
            "if (t > 0.0) {",
            "\tfloat w = 0.5 * t * t;",
            "\tvec2 o = float(n + 1) * d;",
            "\tacc += w * ($input(fract(uv - o)) + $input(fract(uv + o)));",
            "\ttotal += 2.0 * w;",
            "}",
            "return acc / total;",
        ]
    )


Made = tuple[dict[str, Any], list[str]] | Sub
"""A translated node: its Material Maker node and its input names by port."""


def _uniform(node: Node, name: str, *, color: bool, size: int) -> Made:
    # The value's state is node word bits 24-25: 0 none (the default, opaque
    # black), 1 inline, 2 a function, its inline word the block's offset.
    if (node.flags >> 8) & 3 == 2:  # noqa: PLR2004
        offset, _ = _dynamic_offset(node, node.inline, 0)
        value = dynamic(node, offset)
        if color:
            return shader(name, [], "rgba", f"return {vec(value, 4)};"), []
        return shader(name, [], "f", f"return {vec(value, 1)};"), []
    vals = [f32(w) for w in node.inline]
    if color:
        c = knob("color", "Color", vals[:4] or [0.0, 0.0, 0.0, 1.0])
        return shader(name, [], "rgba", f"return {vec(c)};"), []
    v = knob("value", "Value", vals[0] if vals else 0.0)
    return shader(name, [], "f", f"return {num(v)};"), []


def _levels(node: Node, name: str, *, color: bool, size: int) -> Made:
    t, n = ("rgba", 4) if color else ("f", 1)
    p = flag_params(node.fields[0], node.inline, LEVELS, n, node)
    labels = {"in_low": "Input low", "in_high": "Input high", "in_mid": "Midtones",
              "out_low": "Output low", "out_high": "Output high"}  # fmt: skip
    p = {k: knob(k, labels[k], v) for k, v in p.items()}
    body = levels_glsl(
        *(vec(p[k], n) for k in ("in_low", "in_high", "in_mid", "out_low", "out_high")),
        color=color,
    )
    return shader(name, [("input", t)], t, body), ["input"]


def levels_glsl(
    in_low: str, in_high: str, in_mid: str, out_low: str, out_high: str, *, color: bool
) -> str:
    """Return Levels' GLSL body: its settings GLSL values, its input ``$input``."""
    g = "vec4" if color else "float"
    # every setting is clamped to [0, 1], as Substance's are
    lines = [
        f"{g} x = $input(uv);",
        f"{g} lo = clamp({in_low}, 0.0, 1.0);",
        f"{g} hi = clamp({in_high}, 0.0, 1.0);",
        f"{g} m = clamp({in_mid}, 0.0, 1.0);",
        # an input range narrower than 1e-4 is 1e-4 wide: an empty one ramps
        # over the 1e-4 above in_low, which a 16-bit input lands on
        f"{g} d = hi - lo;",
        (
            "d = mix(d, vec4(1e-4), step(abs(d), vec4(1e-4)));"
            if color
            else "d = abs(d) <= 1e-4 ? 1e-4 : d;"
        ),
        "x = clamp((x - lo) / d, 0.0, 1.0);",
        (
            "x = pow(x, mix(1.0 / (1.0 + 16.0 * (0.5 - m)),"
            " 1.0 + 16.0 * (m - 0.5), step(0.5, m)));"
        ),
        f"{g} ol = clamp({out_low}, 0.0, 1.0);",
        f"return mix(ol, clamp({out_high}, 0.0, 1.0), x);",
    ]
    return "\n".join(lines)


CROP_SIDES = ("left", "right", "top", "bottom")
"""A blend's cropping area (Substance's maskrectangle), in its order."""


def blend_crop(node: Node, inline: list[int]) -> list[Any]:
    """Return a blend's cropping area: left, right, top and bottom (uv).

    Its state is in the node's bits 6-7, its words follow the opacity's
    (``inline`` starts at them): inline, two words of four 16-bit values,
    left and top, right and bottom, each a value times 32768; a function,
    its block's offset (a float4 in maskrectangle's order).
    """
    state = node.fields[0] >> 6 & 3
    if state == 1:
        w0, w1 = inline[:2]
        return [x / 32768 for x in (w0 & 0xFFFF, w1 & 0xFFFF, w0 >> 16, w1 >> 16)]
    if state == 2:  # noqa: PLR2004
        return list(dynamic(node, inline[0] + BASE))
    return [0.0, 1.0, 0.0, 1.0]


def crop_glsl(crop: Sequence[str], n: str, *, color: bool) -> str:
    """Return a blend's lines giving the destination outside its cropping area.

    In pixels of an image of side ``n``: from floor(left n) up to, not
    including, floor(right n); the same down (check_filters blendcrop*).
    """
    left, right, top, bottom = crop
    d = "d" if color else "d.x"
    return (
        f"vec2 cpx = floor(uv * {n});\n"
        f"vec2 clo = floor(vec2({left}, {top}) * {n});\n"
        f"vec2 chi = floor(vec2({right}, {bottom}) * {n});\n"
        f"if (any(lessThan(cpx, clo)) || any(greaterThanEqual(cpx, chi))) return {d};"
    )


def _blend(node: Node, name: str, *, color: bool, size: int) -> Made:
    t = "rgba" if color else "f"
    flags = node.fields[0]
    mode = flags & 0xF
    if mode not in BLEND_MODES:
        msg = f"blend mode {mode}"
        raise UnsupportedError(msg)
    inline = list(node.inline)
    # Opacity, bits 4-5: 1 inline, 2 a function, 3 an input, which a
    # multiplier (its state in bits 9-10) scales.
    state = flags >> 4 & 3
    has_input = state == 3  # noqa: PLR2004
    if has_input:
        inline = inline[1:]
        state = (flags >> 9) & 3
    if state == 2:  # noqa: PLR2004
        opacity = dynamic(node, inline[0] + BASE)
    else:
        opacity = f32(inline[0]) if state == 1 else 1.0
    opacity = knob("opacity_mult", "Opacity", opacity)
    o = num(opacity) + (" * $opacity(uv)" if has_input else "")
    if state in (1, 2):
        inline = inline[1:]
    crop = [
        knob(f"crop_{side}", f"Cropping {side}", v)
        for side, v in zip(CROP_SIDES, blend_crop(node, inline), strict=True)
    ]
    expr = BLEND_MODES[mode]
    if mode != 7:  # noqa: PLR2004 (switch mixes, it does not blend)
        modes = [m for m in BLEND_MODES if m != 7]  # noqa: PLR2004
        _CTX.params["mode"] = {
            "name": "mode",
            "label": "Blending mode",
            "type": "enum",
            "default": modes.index(mode),
            "values": [
                {"name": BLEND_NAMES[m], "value": f"({BLEND_MODES[m]})"} for m in modes
            ],
        }
        expr = "$mode"
    inputs = [("source", t), ("destination", t)]
    if has_input:
        inputs.append(("opacity", "f"))
    outside = crop_glsl([num(v) for v in crop], num(float(1 << size)), color=color)
    if color:
        lines = [
            "vec4 s = $source(uv); vec4 d = $destination(uv);",
            outside,
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
            outside,
            f"float o = {o};",
            f"vec4 f = {expr};",
            "return clamp(d.x + (f.x - d.x) * o, 0.0, 1.0);",
        ]
    return shader(name, inputs, t, "\n".join(lines)), [i for i, _ in inputs]


def transform_mip_glsl(formats: Iterable[int]) -> str:
    """Return a transformation's mip filtering helpers (transform_read_lines).

    ``tmlevel_$(name)(gx, gy, n)``: the level read at texel-coordinate
    derivatives gx and gy (per pixel) in an image of side n, as
    fxdraw.level, at most MIP_LIVE_MAX. ``tmlvin``, ``tmlvval`` and
    ``tmmean`` ``_$(name)(..., fmt)``: fxglsl's levels and 2x2 means of a
    format (0: 8 bits, 1: 16 bits, else floats), of ``formats`` only; for
    one format, ``fmt`` is not looked at. None reads the input.
    """
    formats = list(formats)
    out = ""
    for fmt in formats:
        bits = {0: 8, 1: 16}.get(fmt)
        helpers = fxglsl.levels_glsl(bits) + fxglsl.mean_glsl(bits, amd=_FX_AMD)
        if bits is not None:
            helpers = fxglsl.val_glsl() + helpers
        out += helpers.replace("_$(name)", f"{fmt}_$(name)")
    for fn, args, call in (
        ("tmlvin", "vec4 x", "fxlvin{}_$(name)(x)"),
        ("tmlvval", "vec4 x", "fxlvval{}_$(name)(x)"),
        ("tmmean", "vec4 a, vec4 b, vec4 c, vec4 d", "fxmean{}_$(name)(a, b, c, d)"),
    ):
        picks = "".join(
            f"\tif (fmt == {fmt}) return {call.format(fmt)};\n" for fmt in formats[:-1]
        )
        out += (
            f"vec4 {fn}_$(name)({args}, int fmt) {{\n"
            f"{picks}\treturn {call.format(formats[-1])};\n}}\n"
        )
    out += (
        "int tmlevel_$(name)(vec2 gx, vec2 gy, int n) {\n"
        "\tfloat lam = 0.5 * log2(max(dot(gx, gx), dot(gy, gy)));\n"
        + (
            f"\tfloat lq = ceil(lam * 256.0 - {num(0.5 + fxdraw.LOD_TIE)}) / 256.0;\n"
            if _FX_AMD
            else "\tfloat lq = lam;\n"
        )
        # (not a number, say a matrix of infinities: level 0)
        + "\tif (!(lq >= 0.5)) return 0;\n"
        f"\treturn int(min(floor(lq + 0.5), min(log2(float(n)), {MIP_LIVE_MAX}.0)));\n"
        "}\n"
    )
    return out


MIP_LIVE_MAX = 4
"""The deepest mip level a transformation reads (4^level input reads a
texel, built on the fly): beyond it, it reads this one. Translated graphs'
static matrices read at most level 3 (Biomech's); a level from a knob or a
function past it is a sharper image than Substance's, not a stalled render."""


def transform_read_lines(
    inp: str, matrix: Sequence[str], n: str, fmt: str, *, color: bool
) -> list[str]:
    """Return a transformation's last lines: its input ``$<inp>`` read at q.

    At level 0 (tmlevel, from ``matrix`` a, b, c, d: GLSL), at q itself;
    past it bilinear (fractions to 1/256), wrapped, in that mip level of
    the input of side ``n`` stored in ``fmt`` (GLSL ints): each texel the
    level of its 4^L texels below, in Morton order, each four completed
    replaced by their mean (fxglsl.mip_readers). One read of the input in
    the code: a chain of transformations inlined in one shader grows as
    the chain, not 4^k times.
    """
    a, b, c, d = matrix
    return [
        f"int L = tmlevel_$(name)(vec2({a}, {b}), vec2({c}, {d}), {n});",
        f"int ml = {n} >> L;",
        "vec2 tt = fract(q) * float(ml) - 0.5;",
        "vec2 tb = floor(tt);",
        "vec2 tf = floor((tt - tb) * 256.0 + 0.5) / 256.0;",
        "ivec2 i0 = (ivec2(tb) + ml) % ml;",
        "vec4 acc = vec4(0.0);",
        f"vec4 v[{4 * MIP_LIVE_MAX}];",
        "for (int k = 0; k < (L > 0 ? 4 : 1); k++) {",
        "\tivec2 corner = ivec2(k & 1, k >> 1);",
        "\tivec2 tc = (i0 + corner) % ml;",
        "\tvec2 w = mix(1.0 - tf, tf, vec2(corner));",
        "\tvec4 m = vec4(0.0);",
        "\tfor (int i = 0; i < 1 << (2 * L); i++) {",
        "\t\tivec2 o = ivec2(0);",
        "\t\tfor (int l = 0; l < L; l++) {",
        "\t\t\tint c = (i >> (2 * l)) & 3;",
        "\t\t\to += ivec2(c & 1, c >> 1) << l;",
        "\t\t}",
        f"\t\tvec2 at = L > 0 ? (vec2((tc << L) + o) + 0.5) / float({n}) : fract(q);",
        f"\t\tvec4 x = vec4(${inp}(at));",
        "\t\tif (L == 0) {",
        "\t\t\tm = x;",
        "\t\t\tbreak;",
        "\t\t}",
        f"\t\tm = tmlvin_$(name)(x, {fmt});",
        "\t\tfor (int l = 0; l < L; l++) {",
        "\t\t\tint c = (i >> (2 * l)) & 3;",
        "\t\t\tv[4 * l + c] = m;",
        "\t\t\tif (c != 3) break;",
        (
            "\t\t\tm = tmmean_$(name)(v[4 * l], v[4 * l + 1], v[4 * l + 2],"
            f" v[4 * l + 3], {fmt});"
        ),
        "\t\t}",
        "\t}",
        f"\tacc += L > 0 ? w.x * w.y * tmlvval_$(name)(m, {fmt}) : m;",
        "}",
        "return acc;" if color else "return acc.x;",
    ]


def transform_minifies(matrix: Sequence[float]) -> bool:
    """Return whether a transformation of ``matrix`` reads a mip level."""
    a, b, c, d = matrix
    g = max(math.hypot(a, b), math.hypot(c, d))
    if g == 0:
        return False
    return transform_level(g) > 0


def transform_level(g: float) -> int:
    """Return the mip level a transformation reads where a pixel spans ``g`` texels."""
    if g <= 0:
        return 0
    lam = math.log2(g)
    if _FX_AMD:
        lam = math.ceil(lam * 256 - 0.5 - fxdraw.LOD_TIE) / 256
    return math.floor(lam + 0.5) if lam >= 0.5 else 0  # noqa: PLR2004


def transform_feed(node: Node, have: int, own: int) -> int:
    """Return the size (log2) a transformation ``own`` in size reads its input at.

    An input larger than the node is read in the mip level the matrix
    picks for its texels (Biomech's 16-pixel min/max pyramid zooms into
    its 1024-pixel input: level 5, not 6 as the node's size): that level
    averaged down, which the shader reads at level 0. A constant matrix
    only: a function's is read at the node's size.
    """
    if have <= own or (node.fields[0] >> 6) & 3 == 2:  # noqa: PLR2004
        return own
    a, b, c, d = (float(v) for v in _transform_matrix(node)[0])
    g = max(math.hypot(a, b), math.hypot(c, d)) * (1 << have - own)
    return min(max(have - transform_level(g), own), have)


def _transform_matrix(node: Node) -> tuple[list[Any], list[Any]]:
    """Return a transformation's matrix and offset: numbers or GLExprs."""
    flags = node.fields[0]
    matrix: list[Any] = [1.0, 0.0, 0.0, 1.0]
    offset: list[Any] = [0.0, 0.0]
    inline = list(node.inline)
    state_m = (flags >> 6) & 3
    state_o = (flags >> 25) & 3
    if state_m == 1:
        matrix, inline = [f32(w) for w in inline[:4]], inline[4:]
    elif state_m == 2:  # noqa: PLR2004
        matrix, inline = list(dynamic(node, inline[0] + BASE)), inline[1:]
    if state_o == 1:
        offset = [f32(w) for w in inline[:2]]
    elif state_o == 2:  # noqa: PLR2004
        offset = dynamic(node, inline[0] + BASE)
    return matrix, offset


def _transformation(node: Node, name: str, *, color: bool, size: int) -> Made | Sub:
    """Read the input at M^T (uv - 0.5) + 0.5 + offset: Transformation 2D.

    Where it shrinks the image, Substance reads a mip level of it (as an
    FX-Map's image patterns, fxdraw.level): transform_read_lines, the input
    buffered (as stored) where the matrix shrinks it with the graph's
    defaults.
    """
    t = "rgba" if color else "f"
    matrix, offset = _transform_matrix(node)
    a, b, c, d = (knob(k, f"Matrix {k}", v, -2.0, 2.0) for k, v in zip(
        ("m11", "m12", "m21", "m22"), matrix, strict=True))  # fmt: skip
    offset = knob("offset", "Offset", offset, -1.0, 1.0)
    # Tiling, node word bits 24-25: 1 across, 2 down, 3 both; where it
    # does not tile, the image is empty beyond its edges.
    tiling = node.flags >> 8 & 3
    lines = [
        "vec2 p = uv - vec2(0.5);",
        (
            f"vec2 q = vec2({num(a)} * p.x + {num(c)} * p.y,"
            f" {num(b)} * p.x + {num(d)} * p.y);"
        ),
        f"q += vec2(0.5) + {vec(offset, 2)};",
    ]
    for axis, bit in (("x", 1), ("y", 2)):
        if not tiling & bit:
            empty = "vec4(0.0)" if color else "0.0"
            lines.append(f"if (q.{axis} < 0.0 || q.{axis} >= 1.0) return {empty};")
    fmt = node.flags >> 4 & 3
    _CTX.helpers.append(transform_mip_glsl([fmt]))
    lines += transform_read_lines(
        "input", [num(x) for x in (a, b, c, d)], str(1 << size), str(fmt), color=color
    )
    made = shader(name, [("input", t)], t, "\n".join(lines))
    # A function's matrix by its value with the graph's defaults (zero, so
    # no buffer, where it varies per pixel)
    if transform_minifies([float(v) for v in matrix]):
        # its mip levels are of the input as stored
        return buffered(made, ["input"], {"input"}, size, ["input"] if fmt == 1 else [])
    return made, ["input"]


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
            out[pname] = dynamic(node, node.inline[at] + BASE)
            at += 1
    return out


def _gradient_keys(node: Node, *, color: bool) -> list[tuple[float, list[float]]]:
    """Return the gradient keys: (position, rgba or [gray]), by position.

    Color keys are two words, [u16 position | u8 r | u8 g][u8 b | u8 a | u16
    midpoint]; grayscale keys three u16s, position, value, midpoint. Substance
    stores the midpoint words but does not use them during interpolation.
    """
    count = node.fields[1]
    data = node.at(node.code[0], 2 * count)  # the first offset points at the keys
    keys: list[tuple[float, list[float]]] = []
    if color:
        for k in range(count):
            w0, w1 = data[2 * k], data[2 * k + 1]
            pos = (w0 & 0xFFFF) / 0xFFFF
            rgba = [(w0 >> 16 & 0xFF), (w0 >> 24), (w1 & 0xFF), (w1 >> 8 & 0xFF)]
            keys.append((pos, [c / 255 for c in rgba]))
    else:
        u16 = []
        for w in data:
            u16 += [w & 0xFFFF, w >> 16]
        for k in range(count):
            pos, value, _mid = u16[3 * k : 3 * k + 3]
            keys.append((pos / 0xFFFF, [value / 0xFFFF]))
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
    p = {k: knob(k, k.capitalize(), v) for k, v in p.items()}
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
    p["weights"] = knob("weights", "Channel weights", p["weights"], -1.0, 1.0)
    p["matte"] = knob("matte", "Matte", p["matte"])
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


def _blur(node: Node, name: str, *, color: bool, size: int) -> Sub:
    """Blur: a separable box, half-width intensity / 256 of the image."""
    t = "rgba" if color else "f"
    p = word_params(node, [("intensity", 4, 10.0)], {})
    scale = num((1 << size) / 256)
    width = num(float(1 << size))
    k = knob("intensity", "Intensity", p["intensity"], 0.0, 32.0)
    x = shader(
        f"{name}_x", [("input", t)], t,
        box_glsl_live(t, "vec2(1.0, 0.0)", f"{num(k)} * {scale}", width),
    )  # fmt: skip
    # Read again: a function's helper goes with the shader made next.
    p = word_params(node, [("intensity", 4, 10.0)], {})
    k = knob("intensity", "Intensity", p["intensity"], 0.0, 32.0)
    y = shader(
        name, [("input", t)], t,
        box_glsl_live(t, "vec2(0.0, 1.0)", f"{num(k)} * {scale}", width),
    )  # fmt: skip
    # the graph's inputs the x pass reads are read by the blur, as the y
    # pass's (named after it) are: not by the x pass's name, which goes
    _forget({x["name"]: name}, None)
    x["name"], y["name"] = "pass_x", "pass_y"
    group = subgraph(
        name,
        "Blur",
        [buffer("buf_x", size), x, buffer("buf_y", size), y],
        [wire("buf_x", "pass_x"), wire("pass_x", "buf_y"), wire("buf_y", "pass_y")],
        [(t, "buf_x", 0)],
        ("pass_y", 0, t),
    )
    return Sub([group], [], [("input", name, 0)], name)


def _dmb_setup(node: Node, size: int, *, lpx: bool = False) -> str:
    """Return GLSL deriving a directional motion blur's geometry from its knobs.

    Declares the intensity and angle knobs of the next shader (each of the
    blur's shaders has them, one control on the blur's subgraph), and
    computes from them, in the shader (dmb_setup_glsl).
    """
    p = flag_params(
        node.fields[0],
        node.inline,
        [("intensity", 0, 10.0), ("angle", 2, 0.0)],
        1,
        node,
    )
    k = knob("intensity", "Intensity", p["intensity"], 0.0, 32.0)
    turn = knob("angle", "Angle", p["angle"])
    return dmb_setup_glsl(num(k), num(turn), num(float(1 << size)), lpx=lpx)


def dmb_setup_glsl(k: str, turn: str, n: str, *, lpx: bool = False) -> str:
    """Return GLSL deriving a directional motion blur's geometry.

    From its intensity ``k`` and angle ``turn`` (GLSL values) and the image
    width ``n`` (a GLSL float): ``N`` (texels), ``ymaj`` (steps go along
    y), the slope ``t`` (minor-axis texels per step), the radius ``R`` in
    steps, the tap count ``n`` either side, the end taps' weight ``f``, and
    ``X0``, the image center the frame is sheared about; with ``lpx`` also
    ``Lpx``, the length in texels.
    """
    return "\n".join(
        [
            f"float N = {n};",
            f"float A = {turn};",
            "float ang = 6.28318531 * A;",
            "float cs = cos(ang);",
            "float sn = sin(ang);",
            "bool ymaj = abs(sn) > abs(cs);",
            "float t = ymaj ? cs / sn : sn / cs;",
            # The diagonals: Substance's rounding picks the axis (measured).
            "float a8 = abs(A) * 8.0;",
            "float m8 = floor(a8 + 0.5);",
            "if (abs(a8 - m8) < 1e-5 && mod(m8, 2.0) == 1.0) {",
            "\tymaj = mod(m8, 4.0) == 1.0;",
            "\tt = cs * sn > 0.0 ? 1.0 : -1.0;",
            "}",
            "if (abs(t) < 1e-6) t = 0.0;",
            f"float R = 0.5 + ({k} * N / 256.0 - 0.5) * max(abs(cs), abs(sn));",
            "int n = int(ceil(R - 0.5));",
            "float f = R + 0.5 - float(n);",
            "float X0 = 0.5 * (N - 1.0);",
            *([f"float Lpx = {k} * N / 256.0;"] if lpx else []),
        ]
    )


_DMB_Q = "float $(name)_q(float u) {\n\treturn floor(u * 256.0 + 0.5) / 256.0;\n}\n"
"""A bilinear fraction rounded to 1/256, as the texture filter rounds it."""


def _dmb_image_reads(g: str, n: str, *, eight: bool) -> str:
    """Return GLSL reading the blur's input buffer by texel.

    ``_p(i, j, ymaj)`` is the texel at major-axis index i, minor j (its
    center: no filtering); ``_read(i, y, ymaj)`` reads at integer i and
    fractional y, linearly in y with the fraction rounded to 1/256. Indices
    wrap (``n`` is the image width, a GLSL float). In an 8-bit graph the
    input holds k / 255, which the (16-bit float) buffer keeps only nearly:
    texels are rounded back to it.
    """
    texel = f"$input((mod(xy, {n}) + 0.5) / {n})"
    if eight:
        texel = f"floor({texel} * 255.0 + 0.5) / 255.0"
    return (
        _DMB_Q + f"{g} $(name)_p(float i, float j, bool ymaj) {{\n"
        "\tvec2 xy = ymaj ? vec2(j, i) : vec2(i, j);\n"
        f"\treturn {texel};\n"
        "}\n"
        f"{g} $(name)_read(float i, float y, bool ymaj) {{\n"
        "\tfloat y0 = floor(y);\n"
        "\tfloat w = $(name)_q(y - y0);\n"
        "\treturn mix($(name)_p(i, y0, ymaj), $(name)_p(i, y0 + 1.0, ymaj), w);\n"
        "}\n"
    )


_DMB_SUM, _DMB_READ = "0.0", "0.03"
"""The 8-bit store's offset (in 4096ths) for a sum the shader computes, and
for a value straight from a bilinear read (NVIDIA's filter returns it a
hair high)."""


def _dmb_s8(g: str) -> str:
    """Return GLSL for Substance's 8-bit store, as on NVIDIA GPUs.

    Clamped, truncated to 1/4096 (after the offset ``c``: _DMB_SUM or
    _DMB_READ by where the value comes from), then the nearest 255th, ties
    down. Exact in float32: X 255 + 2047 stays below 2^20.
    """
    return (
        f"{g} $(name)_s8({g} v, float c) {{\n"
        f"\t{g} x = floor(clamp(v, 0.0, 1.0) * 4096.0 + c);\n"
        "\treturn floor((x * 255.0 + 2047.0) / 4096.0) / 255.0;\n"
        "}\n" + _DMB_BITS
    )


_DMB_BITS = (
    # floor(log2(m)) and the power of two dividing s (s = 0: any).
    "int $(name)_msb(int m) {\n"
    "\tint l = 0;\n"
    "\twhile ((2 << l) <= m) l++;\n"
    "\treturn l;\n"
    "}\n"
    "int $(name)_lsb(int s) {\n"
    "\tif (s == 0) return 99;\n"
    "\tint l = 0;\n"
    "\twhile (((s >> l) & 1) == 0) l++;\n"
    "\treturn l;\n"
    "}\n"
)
"""GLSL the box's aligned blocks (_dmb_blocks) use."""


def _dmb_blocks(mask: str, use: str) -> list[str]:
    """Return GLSL adding the box's aligned blocks from ``start`` to ``end``.

    Greedy, as Substance's pyramid holds them: each the largest power of
    two that fits and divides its start (modulo the period: ``mask`` is the
    image width less one, a GLSL int), of two at least. ``use`` is GLSL
    adding block (``start``, level ``L``).
    """
    return [
        "for (int it = 0; it < 64 && end - start >= 2; it++) {",
        f"\tint s = start & {mask};",
        "\tint L = min($(name)_msb(end - start), $(name)_lsb(s));",
        "\tif (L < 1) break;",
        f"\t{use}",
        "\tstart += 1 << L;",
        "}",
    ]


def _dmb_eight(g: str, n: str, mask: str) -> tuple[str, list[str]]:
    """Return an 8-bit graph's large paths: GLSL helpers and body lines.

    Substance stores every intermediate texture in 8 bits (_dmb_s8) and
    boxes with a pyramid of stored pair means, recursive, along the major
    axis: a block's value is a tree of stored means of its texels, here
    rebuilt per pixel. At 0 and 90 degrees the box runs on the input, its
    partial end taps merged with their neighbors as in 16 bits, but by the
    aligned blocks (both ends when n = 2). Otherwise every value is the
    stored frame's (blocks a fractional column over past its seam), with no
    merging, the box stored with 1/512 added, then the read back stored.
    ``n`` is the image width (a GLSL float), ``mask`` it less one (int).
    """
    helpers = (
        _dmb_s8(g)
        # A partial end tap (at b) read with its whole neighbor (a) at once.
        + f"{g} $(name)_mrg(float a, float b, float mn, float qf, float f,"
        " bool ymaj) {\n"
        "\treturn (1.0 + f) * ((1.0 - qf) * $(name)_p(a, mn, ymaj)"
        " + qf * $(name)_p(b, mn, ymaj));\n"
        "}\n"
        # Level L at major index s0 along minor index mn: input texels.
        f"{g} $(name)_blk(float s0, int L, float mn, bool ymaj) {{\n"
        f"\t{g} st[13];\n"
        "\tfor (int j = 0; j < (1 << L); j++) {\n"
        f"\t\t{g} v = $(name)_p(s0 + float(j), mn, ymaj);\n"
        "\t\tint l = 0;\n"
        "\t\tfor (int jj = j; (jj & 1) == 1; jj >>= 1) {\n"
        "\t\t\tv = $(name)_s8((st[l] + v) / 2.0, " + _DMB_READ + ");\n"
        "\t\t\tl++;\n"
        "\t\t}\n"
        "\t\tst[l] = v;\n"
        "\t}\n"
        "\treturn st[L];\n"
        "}\n"
        # Level L of the stored frame at row s0, columns c and c + 1: both
        # from three texels a row (the same fraction), past the seam three
        # columns from four, a fractional column over.
        f"void $(name)_blk2(float c, float s0, int L, float t, float X0, bool ymaj,"
        f" out {g} a, out {g} b) {{\n"
        f"\tfloat k = floor(s0 / {n});\n"
        f"\tfloat rm = s0 - k * {n};\n"
        f"\tfloat cf = c + k * {n} * t;\n"
        "\tfloat ca = floor(cf);\n"
        "\tbool seam = k != 0.0;\n"
        f"\t{g} s0a[13]; {g} s1a[13]; {g} s2a[13];\n"
        "\tfor (int j = 0; j < (1 << L); j++) {\n"
        "\t\tfloat r = rm + float(j);\n"
        "\t\tfloat y = ca + (r - X0) * t;\n"
        "\t\tfloat y0 = floor(y);\n"
        "\t\tfloat w = $(name)_q(y - y0);\n"
        f"\t\t{g} p1 = $(name)_p(r, y0 + 1.0, ymaj);\n"
        f"\t\t{g} p2 = $(name)_p(r, y0 + 2.0, ymaj);\n"
        f"\t\t{g} v0 = $(name)_s8(mix($(name)_p(r, y0, ymaj), p1, w), "
        + _DMB_READ
        + ");\n"
        f"\t\t{g} v1 = $(name)_s8(mix(p1, p2, w), " + _DMB_READ + ");\n"
        f"\t\t{g} v2 = seam ? $(name)_s8(mix(p2, $(name)_p(r, y0 + 3.0, ymaj), w),"
        " " + _DMB_READ + ")"
        f" : {g}(0.0);\n"
        "\t\tint l = 0;\n"
        "\t\tfor (int jj = j; (jj & 1) == 1; jj >>= 1) {\n"
        "\t\t\tv0 = $(name)_s8((s0a[l] + v0) / 2.0, " + _DMB_READ + ");\n"
        "\t\t\tv1 = $(name)_s8((s1a[l] + v1) / 2.0, " + _DMB_READ + ");\n"
        "\t\t\tif (seam) v2 = $(name)_s8((s2a[l] + v2) / 2.0, " + _DMB_READ + ");\n"
        "\t\t\tl++;\n"
        "\t\t}\n"
        "\t\ts0a[l] = v0; s1a[l] = v1; s2a[l] = v2;\n"
        "\t}\n"
        "\tif (!seam) {\n"
        "\t\ta = s0a[L]; b = s1a[L];\n"
        "\t\treturn;\n"
        "\t}\n"
        "\tfloat phi = $(name)_q(cf - ca);\n"
        "\ta = mix(s0a[L], s1a[L], phi);\n"
        "\tb = mix(s1a[L], s2a[L], phi);\n"
        "}\n"
    )
    axis = [
        "if (t == 0.0) {",
        "\tfloat Qf = $(name)_q(f / (1.0 + f));",
        "\tint first = int(pq.x) - (n - 1);",
        "\tint end = first + 2 * n - 1;",
        "\tint start = first;",
        "\tfloat mn = pq.y;",
        f"\t{g} acc = $(name)_mrg(float(first), float(first - 1), mn, Qf, f, ymaj);",
        "\tif (n == 2) {",
        "\t\tacc = acc + $(name)_p(float(first + 1), mn, ymaj);",
        "\t\tstart = first + 2;",
        "\t} else if ((first & 1) == 1) {",
        "\t\tstart = first + 1;",
        "\t} else {",
        "\t\tacc = f * $(name)_p(float(first - 1), mn, ymaj);",
        "\t}",
        *(
            "\t" + line
            for line in _dmb_blocks(
                mask, "acc += float(1 << L) * $(name)_blk(float(start), L, mn, ymaj);"
            )
        ),
        "\tif (end - start == 1) {",
        "\t\tacc += $(name)_mrg(float(start), float(start + 1), mn, Qf, f, ymaj);",
        "\t} else {",
        "\t\tacc += f * $(name)_p(float(start), mn, ymaj);",
        "\t}",
        "\treturn $(name)_s8(acc / (2.0 * R), " + _DMB_SUM + ");",
        "}",
    ]
    call = "$(name)_blk2(c0, float({at}), {lv}, t, X0, ymaj, a, b);"
    rotated = [
        "float ch = pq.y - (pq.x - X0) * t;",
        "float c0 = floor(ch);",
        "int first = int(pq.x) - (n - 1);",
        "int end = first + 2 * n - 1;",
        "int start = first;",
        f"{g} a; {g} b;",
        call.format(at="first - 1", lv="0"),
        f"{g} accA = f * a;",
        f"{g} accB = f * b;",
        "if ((first & 1) == 1) {",
        "\t" + call.format(at="first", lv="0"),
        "\taccA += a; accB += b;",
        "\tstart += 1;",
        "}",
        *_dmb_blocks(
            mask,
            call.format(at="start", lv="L")
            + " accA += float(1 << L) * a; accB += float(1 << L) * b;",
        ),
        "if (end - start == 1) {",
        "\t" + call.format(at="start", lv="0"),
        "\taccA += a; accB += b;",
        "\tstart += 1;",
        "}",
        call.format(at="start", lv="0"),
        "accA += f * a; accB += f * b;",
        f"{g} gA = $(name)_s8(accA / (2.0 * R) + 1.0 / 512.0, " + _DMB_SUM + ");",
        f"{g} gB = $(name)_s8(accB / (2.0 * R) + 1.0 / 512.0, " + _DMB_SUM + ");",
        "return $(name)_s8(mix(gA, gB, $(name)_q(ch - c0)), " + _DMB_READ + ");",
    ]
    return helpers, axis + rotated


_DMB_AMD = False
"""Round a 16-bit graph's halves as AMD GPUs do, not NVIDIA's (always up).

Internal: for checking the shader against renders made on AMD.
"""


def _dmb_sixteen_helpers(g: str) -> str:
    """Return GLSL for a 16-bit graph's stores, on levels held in floats.

    A level is the integer k of a stored k / 65535; sums of a few of them
    stay exact in float32. ``_lv`` is an input texel's level; ``_val`` the
    float32 nearest k / 65535, as a texture read returns it (built from
    the integer quotient: GPU division is not rounded to nearest; ``up``
    whether it rounded up); ``_rd`` a bilinear read of two levels, the
    filter's exact result rounded to float32. ``_st`` stores a computed
    float32, exactly: on NVIDIA truncated to a multiple of 2^-20, then to
    the nearest level; on AMD straight to the nearest level (halves up).
    ``_sr`` stores a bilinear read: on NVIDIA its exact value to the
    nearest level, halves up; on AMD its float32 so. ``_pm`` is the
    pyramid's mean of two levels, a half always up (NVIDIA) or as its
    float32 rounded (AMD). ``_nc`` passes a product through an opaque bit
    operation, so it is not fused into the sum it feeds.
    """
    u = "uvec4" if g == "vec4" else "uint"
    i = "ivec4" if g == "vec4" else "int"
    if _DMB_AMD:
        # floor(65535 v + 0.5) = floor(65536 v) + floor(frac - v + 0.5).
        store = (
            "\tv = clamp(v, 0.0, 1.0);\n"
            f"\t{g} x = v * 65536.0;\n"
            f"\t{g} k = floor(x);\n"
            f"\t{g} fx = x - k;\n"
            "\treturn k + step(v, fx - 0.5) + step(v, fx + 0.5) - 1.0;\n"
        )
        stored = (
            f"\t{g} s = (256.0 - j) * a + j * b;\n"
            f"\t{g} h = floor(s / 256.0);\n"
            f"\t{g} lo = s - 256.0 * h;\n"
            f"\t{g} up;\n"
            "\t$(name)_val(s, up);\n"
            f"\t{g} tie = 1.0 - min(abs(lo - 128.0), {g}(1.0));\n"
            f"\treturn h + step({g}(128.5), lo) + tie * up;\n"
        )
        mean = (
            f"\t{g} s = a + b;\n"
            f"\t{g} h = floor(s * 0.5);\n"
            f"\t{g} up;\n"
            "\t$(name)_val(s, up);\n"
            "\treturn h + (s - 2.0 * h) * up;\n"
        )
    else:
        # x = 16 q + r, v truncated to 2^-20: (65535 x + 2^19) / 2^20 exactly.
        store = (
            "\tv = clamp(v, 0.0, 1.0);\n"
            f"\t{g} x = floor(v * 1048576.0);\n"
            f"\t{g} q = floor(x / 16.0);\n"
            "\treturn q + floor(((x - 16.0 * q) * 65536.0 + 524288.0 - x)"
            " / 1048576.0);\n"
        )
        # A read's exact value, halves up: no truncation (exact integers).
        stored = "\treturn floor(((256.0 - j) * a + j * b + 128.0) / 256.0);\n"
        mean = "\treturn floor((a + b + 1.0) * 0.5);\n"
    return (
        _DMB_BITS + f"{g} $(name)_lv(float i, float j, bool ymaj) {{\n"
        "\treturn floor($(name)_p(i, j, ymaj) * 65535.0 + 0.5);\n"
        "}\n"
        f"{g} $(name)_val({g} k, out {g} up) {{\n"
        f"\t{u} n = {u}(max(k, {g}(1.0)));\n"
        f"\t{i} m = findMSB(n);\n"
        f"\t{u} a = n << {u}(23 - m);\n"
        # The quotient reaching 2^24: it is in the next binade.
        f"\t{u} hz = (a + a / 65535u) >> 24u;\n"
        "\ta = a >> hz;\n"
        f"\t{u} r = a % 65535u;\n"
        f"\t{u} big = (2u * r) / 65536u;\n"
        f"\tup = {g}(max(big, {u}(1u) - min(r, {u}(1u))));\n"
        f"\t{u} bits = ({u}(m + 111) << 23u) + (hz << 23u) + a + a / 65535u + big"
        " - 8388608u;\n"
        f"\treturn uintBitsToFloat(bits) * min(k, {g}(1.0));\n"
        "}\n"
        f"{g} $(name)_v({g} k) {{\n"
        f"\t{g} up;\n"
        "\treturn $(name)_val(k, up);\n"
        "}\n"
        f"{g} $(name)_rd({g} a, {g} b, float j) {{\n"
        "\treturn $(name)_v((256.0 - j) * a + j * b) / 256.0;\n"
        "}\n"
        f"{g} $(name)_st({g} v) {{\n{store}}}\n"
        f"{g} $(name)_sr({g} a, {g} b, float j) {{\n{stored}}}\n"
        f"{g} $(name)_pm({g} a, {g} b) {{\n{mean}}}\n"
        f"{g} $(name)_nc({g} v, uint z) {{\n"
        "\treturn uintBitsToFloat(floatBitsToUint(v) ^ z);\n"
        "}\n"
        "float $(name)_ncf(float v, uint z) {\n"
        "\treturn uintBitsToFloat(floatBitsToUint(v) ^ z);\n"
        "}\n"
        # c + h t (c an integer, h a multiple of 1/2 under 2^12): its floor,
        # and the fraction's Q-rounded weight j in 256ths. h times t's top
        # 11 bits is exact, so only t's own rounding is left.
        "float $(name)_at(float c, float h, float t, out float j) {\n"
        "\tfloat th = uintBitsToFloat(floatBitsToUint(t) & 4294959104u);\n"
        "\tfloat ph = h * th;\n"
        "\tfloat b = floor(ph);\n"
        "\tfloat fr = (ph - b) + h * (t - th);\n"
        "\tfloat fl = floor(fr);\n"
        "\tj = floor((fr - fl) * 256.0 + 0.5);\n"
        "\treturn c + b + fl;\n"
        "}\n" + RCP_GLSL
    )


RCP_GLSL = (
    # 1 / x to the nearest float32 (x > 0, normal): long division.
    "float $(name)_rcp(float x) {\n"
    "\tuint bits = floatBitsToUint(x);\n"
    "\tuint e = bits >> 23u;\n"
    "\tuint m = (bits & 8388607u) | 8388608u;\n"
    "\tif (m == 8388608u) return uintBitsToFloat((254u - e) << 23u);\n"
    "\tuint q = 0u;\n"
    "\tuint r = 8388608u;\n"
    "\tfor (int it = 0; it < 24; it++) {\n"
    "\t\tr *= 2u;\n"
    "\t\tq *= 2u;\n"
    "\t\tif (r >= m) { r -= m; q += 1u; }\n"
    "\t}\n"
    "\tif (2u * r > m) q += 1u;\n"
    "\treturn uintBitsToFloat(((253u - e) << 23u) + q - 8388608u);\n"
    "}\n"
)
"""GLSL ``$(name)_rcp(x)``: 1 / x correctly rounded (a GPU's is not)."""


def _dmb_sixteen(g: str, n: str, mask: str) -> tuple[str, list[str]]:
    """Return a 16-bit graph's paths past R = 0.5: GLSL helpers and body lines.

    The 8-bit graph's passes (_dmb_eight), Substance's intermediates stored
    in 16 bits: integer pair means in the pyramid, the rotated frame and
    back-resample stored reads, its box stored with 0.4 of a level added.
    R, 1 / 2R and every term are float32 as the engine computes them (max
    |cos|, |sin| exactly 1 only at angle 0, an ulp below at other quarter
    turns), the terms summed from the + end along x, the - end along y.
    ``n`` is the image width (a GLSL float), ``mask`` it less one (int).
    """
    helpers = (
        _dmb_sixteen_helpers(g)
        # Level L at major index s0 along minor index mn: input levels.
        + f"{g} $(name)_blk(float s0, int L, float mn, bool ymaj) {{\n"
        f"\t{g} st[13];\n"
        "\tfor (int j = 0; j < (1 << L); j++) {\n"
        f"\t\t{g} v = $(name)_lv(s0 + float(j), mn, ymaj);\n"
        "\t\tint l = 0;\n"
        "\t\tfor (int jj = j; (jj & 1) == 1; jj >>= 1) {\n"
        "\t\t\tv = $(name)_pm(st[l], v);\n"
        "\t\t\tl++;\n"
        "\t\t}\n"
        "\t\tst[l] = v;\n"
        "\t}\n"
        "\treturn st[L];\n"
        "}\n"
        # Level L of the stored frame at row s0, columns c and c + 1, as
        # read: past the seam a fractional column over, between two
        # columns' levels.
        f"void $(name)_blk2(float c, float s0, int L, float t, float X0, bool ymaj,"
        f" out {g} a, out {g} b) {{\n"
        f"\tfloat k = floor(s0 / {n});\n"
        f"\tfloat rm = s0 - k * {n};\n"
        "\tfloat phi;\n"
        f"\tfloat ca = $(name)_at(c, k * {n}, t, phi);\n"
        "\tbool seam = k != 0.0;\n"
        f"\t{g} s0a[13]; {g} s1a[13]; {g} s2a[13];\n"
        "\tfor (int j = 0; j < (1 << L); j++) {\n"
        "\t\tfloat r = rm + float(j);\n"
        "\t\tfloat w;\n"
        "\t\tfloat y0 = $(name)_at(ca, r - X0, t, w);\n"
        f"\t\t{g} p1 = $(name)_lv(r, y0 + 1.0, ymaj);\n"
        f"\t\t{g} p2 = $(name)_lv(r, y0 + 2.0, ymaj);\n"
        f"\t\t{g} v0 = $(name)_sr($(name)_lv(r, y0, ymaj), p1, w);\n"
        f"\t\t{g} v1 = $(name)_sr(p1, p2, w);\n"
        f"\t\t{g} v2 = seam ? $(name)_sr(p2, $(name)_lv(r, y0 + 3.0, ymaj), w)"
        f" : {g}(0.0);\n"
        "\t\tint l = 0;\n"
        "\t\tfor (int jj = j; (jj & 1) == 1; jj >>= 1) {\n"
        "\t\t\tv0 = $(name)_pm(s0a[l], v0);\n"
        "\t\t\tv1 = $(name)_pm(s1a[l], v1);\n"
        "\t\t\tif (seam) v2 = $(name)_pm(s2a[l], v2);\n"
        "\t\t\tl++;\n"
        "\t\t}\n"
        "\t\ts0a[l] = v0; s1a[l] = v1; s2a[l] = v2;\n"
        "\t}\n"
        "\tif (!seam) {\n"
        "\t\ta = $(name)_v(s0a[L]); b = $(name)_v(s1a[L]);\n"
        "\t\treturn;\n"
        "\t}\n"
        "\ta = $(name)_rd(s0a[L], s1a[L], phi);\n"
        "\tb = $(name)_rd(s1a[L], s2a[L], phi);\n"
        "}\n"
    )
    # Every product a sum takes goes through _nc: the engine's sums, unfused.
    setup = [
        "uint z = uint(uv.x > 1e30);",
        # cos and sin to about an ulp (the GPU's own are coarser, and the
        # slope places every frame read): the angle less its nearest
        # quarter turn, by series.
        "float ra = A - floor(A + 0.5);",
        "float qd = floor(ra * 4.0 + 0.5);",
        "float x = (ra - 0.25 * qd) * 6.28318531;",
        "float x2 = x * x;",
        (
            "float sx = x * (1.0 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0"
            " + x2 * (-1.0 / 5040.0 + x2 * (1.0 / 362880.0)))));"
        ),
        (
            "float cx = 1.0 + x2 * (-0.5 + x2 * (1.0 / 24.0 + x2 * (-1.0 / 720.0"
            " + x2 * (1.0 / 40320.0 + x2 * (-1.0 / 3628800.0)))));"
        ),
        "float qm = mod(qd, 4.0);",
        "cs = qm == 0.0 ? cx : qm == 1.0 ? -sx : qm == 2.0 ? -cx : sx;",
        "sn = qm == 0.0 ? sx : qm == 1.0 ? cx : qm == 2.0 ? -sx : -cx;",
        "if (abs(a8 - m8) >= 1e-5 || mod(m8, 2.0) != 1.0) {",
        "\tt = ymaj ? cs / sn : sn / cs;",
        "\tif (abs(t) < 1e-6) t = 0.0;",
        "}",
        "float mx = max(abs(cs), abs(sn));",
        "if (abs(a8 - m8) < 1e-5 && mod(m8, 2.0) == 1.0) mx = 0.70710677;",
        "float a4 = A * 4.0;",
        "if (a4 == floor(a4)) mx = A == 0.0 ? 1.0 : 0.99999994;",
        "R = 0.5 + $(name)_ncf((Lpx - 0.5) * mx, z);",
        "n = int(ceil(R - 0.5));",
        "f = R + 0.5 - float(n);",
        "float inv = $(name)_rcp(2.0 * R);",
    ]
    small = [
        "if (n <= 1) {",
        "\tfloat ws = f * inv;",
        "\tfloat j;",
        "\tfloat y0 = $(name)_at(pq.y, 1.0, t, j);",
        (
            f"\t{g} p = $(name)_rd($(name)_lv(pq.x + 1.0, y0, ymaj),"
            " $(name)_lv(pq.x + 1.0, y0 + 1.0, ymaj), j);"
        ),
        "\ty0 = $(name)_at(pq.y, -1.0, t, j);",
        (
            f"\t{g} m = $(name)_rd($(name)_lv(pq.x - 1.0, y0, ymaj),"
            " $(name)_lv(pq.x - 1.0, y0 + 1.0, ymaj), j);"
        ),
        f"\t{g} c = $(name)_v($(name)_lv(pq.x, pq.y, ymaj));",
        (
            f"\t{g} v = ($(name)_nc(p * ws, z) + $(name)_nc(c * inv, z))"
            " + $(name)_nc(m * ws, z);"
        ),
        "\treturn $(name)_v($(name)_st(v));",
        "}",
    ]

    # The box's terms are summed from its + end along x, from its - end
    # along y: its aligned blocks found from that end (the same blocks).
    def blocks(use: str) -> list[str]:
        return [
            "if (ymaj) {",
            "\tint start = sB;",
            *("\t" + line for line in _dmb_blocks(mask, use)),
            "} else {",
            "\tint e = eB;",
            "\tfor (int it = 0; it < 64 && e - sB >= 2; it++) {",
            f"\t\tint L = min($(name)_msb(e - sB), $(name)_lsb(e & {mask}));",
            "\t\tint start = e - (1 << L);",
            f"\t\t{use}",
            "\t\te = start;",
            "\t}",
            "}",
        ]

    # The whole texels left at either end: the - end's when the first is
    # odd, the + end's when one is over after the blocks.
    ends = [
        "int first = int(pq.x) - (n - 1);",
        "int end = first + 2 * n - 1;",
        "bool lo = (first & 1) == 1;",
        "int sB = lo ? first + 1 : first;",
        "bool hi = ((end - sB) & 1) == 1;",
        "int eB = hi ? end - 1 : end;",
    ]
    # A partial end tap (at b) read with its whole neighbor (a) at once.
    mrg = (
        "$(name)_nc($(name)_rd($(name)_lv({a}, mn, ymaj), $(name)_lv({b}, mn, ymaj),"
        " Qj) * (f + 1.0), z)"
    )
    lone = "$(name)_nc(f * $(name)_v($(name)_lv({a}, mn, ymaj)), z)"
    axis = [
        "if (t == 0.0) {",
        "\tfloat Qj = 256.0 * $(name)_q(f / (1.0 + f));",
        "\tfloat mn = pq.y;",
        *("\t" + line for line in ends),
        "\tif (n == 2) {",  # in this order on both axes
        f"\t\t{g} e0 = " + mrg.format(a="float(first)", b="float(first - 1)") + ";",
        f"\t\t{g} e1 = " + mrg.format(a="float(first + 2)", b="float(first + 3)") + ";",
        f"\t\t{g} mid = $(name)_v($(name)_lv(float(first + 1), mn, ymaj));",
        "\t\treturn $(name)_v($(name)_st($(name)_nc(((e0 + mid) + e1) * inv, z)));",
        "\t}",
        # The ends: merged with their whole neighbors, or partial alone.
        f"\t{g} e0 = lo ? "
        + mrg.format(a="float(first)", b="float(first - 1)")
        + " : "
        + lone.format(a="float(first - 1)")
        + ";",
        f"\t{g} e1 = hi ? "
        + mrg.format(a="float(eB)", b="float(eB + 1)")
        + " : "
        + lone.format(a="float(eB)")
        + ";",
        f"\t{g} acc = ymaj ? e0 : e1;",
        *(
            "\t" + line
            for line in blocks(
                "acc = acc + float(1 << L)"
                " * $(name)_v($(name)_blk(float(start), L, mn, ymaj));"
            )
        ),
        "\tacc = acc + (ymaj ? e1 : e0);",
        "\treturn $(name)_v($(name)_st($(name)_nc(acc * inv, z)));",
        "}",
    ]

    def call(at: str, lv: str = "0") -> str:
        return f"$(name)_blk2(c0, float({at}), {lv}, t, X0, ymaj, a, b);"

    add = "accA = accA + {k}a; accB = accB + {k}b;"
    whole = {
        "lo": ["if (lo) {", "\t" + call("first"), "\t" + add.format(k=""), "}"],
        "hi": ["if (hi) {", "\t" + call("end - 1"), "\t" + add.format(k=""), "}"],
    }
    rotated = [
        "float jg;",
        "float c0 = $(name)_at(pq.y, X0 - pq.x, t, jg);",
        *ends,
        f"{g} a; {g} b;",
        call("first - 1"),
        f"{g} e0A = $(name)_nc(f * a, z); {g} e0B = $(name)_nc(f * b, z);",
        call("end"),
        f"{g} e1A = $(name)_nc(f * a, z); {g} e1B = $(name)_nc(f * b, z);",
        f"{g} accA = ymaj ? e0A : e1A; {g} accB = ymaj ? e0B : e1B;",
        "if (ymaj) {",
        *("\t" + line for line in whole["lo"]),
        "} else {",
        *("\t" + line for line in whole["hi"]),
        "}",
        *blocks(call("start", "L") + " " + add.format(k="float(1 << L) * ")),
        "if (ymaj) {",
        *("\t" + line for line in whole["hi"]),
        "} else {",
        *("\t" + line for line in whole["lo"]),
        "}",
        "accA = accA + (ymaj ? e1A : e0A); accB = accB + (ymaj ? e1B : e0B);",
        # 0.4 of a level, added to the rounded product.
        f"{g} gA = $(name)_st($(name)_nc(accA * inv, z) + 6.103609e-06);",
        f"{g} gB = $(name)_st($(name)_nc(accB * inv, z) + 6.103609e-06);",
        "return $(name)_v($(name)_sr(gA, gB, jg));",
    ]
    return helpers, setup + small + axis + rotated


def _dirmotionblur(node: Node, name: str, *, color: bool, size: int) -> Sub:
    """Directional motion blur: a box of half-width R steps along the angle.

    The angle (turns) picks the major axis, the one its direction moves
    along more; a step is one texel along it and t texels across. R is
    0.5 + (intensity N / 256 - 0.5) max(|cos|, |sin|) steps: n = ceil(R -
    0.5) taps either side, the end ones weighted R + 0.5 - n, the sum
    divided by 2R. Up to R = 0.5 the image is kept; up to 1.5 it is three
    reads, linear across the major axis.

    Past that Substance shears the image into a frame whose column c is
    the line minor = c + (major - X0) t, boxes along the columns and reads
    the result back linearly across them. One shader does it all: per
    pixel, the two frame columns either side of it, each boxed over its
    taps (both from three texels a tap: their reads share the fraction),
    then mixed. A tap past the image's edge reads the frame's own row a
    fractional column over (the frame's seam). At 0 and 90 degrees the
    frame is the image, and Substance's box reads its partial end tap and
    its neighbor as one, so their weights move by that read's rounding,
    at one end or the other by the parity of the major-axis index (both
    when n = 2). Every linear read rounds its fraction to 1/256, as the
    GPU's texture filter does; texels are read at their centers.

    A node stored in 8 bits follows Substance's 8-bit arithmetic instead
    (_dmb_eight): its stores, its pyramid, its bias. One stored in 16 bits
    follows the same passes with 16-bit stores (_dmb_sixteen), on its
    input's exact levels (a 32-bit buffer).

    Intensity and angle are live: everything is computed from them in the
    shader, so the tap loops run to a bound counted per pixel.
    """
    t = "rgba" if color else "f"
    g = "vec4" if color else "float"
    fmt = node.flags >> 4 & 3
    setup = _dmb_setup(node, size, lpx=fmt == 1)
    helpers, body = dmb_glsl(g, fmt, num(float(1 << size)), str((1 << size) - 1), setup)
    made = shader(name, [("input", t)], t, body)
    made["shader_model"]["instance"] = helpers + made["shader_model"]["instance"]
    # 16 bits: the input's levels exactly, so a 32-bit buffer.
    precise = {"input"} if fmt == 1 else set()
    return buffered(made, ["input"], {"input"}, size, precise=precise)


def dmb_glsl(g: str, fmt: int, n: str, mask: str, setup: str) -> tuple[str, str]:
    """Return a directional motion blur's GLSL: helpers and the body.

    ``g`` is its value type (float or vec4), ``fmt`` the node's output
    format (_dirmotionblur), ``n`` the image width (a GLSL float), ``mask``
    it less one (a GLSL int), ``setup`` dmb_setup_glsl's lines. The body
    reads the input as ``$input``.
    """
    eight, sixteen = fmt == 0, fmt == 1
    start = [
        setup,
        f"vec2 pq = floor(uv * {n});",
        "if (ymaj) pq = pq.yx;",
        "if (R <= 0.5) return $(name)_p(pq.x, pq.y, ymaj);",
    ]
    if sixteen:
        helpers, large = _dmb_sixteen(g, n, mask)
        lines = [*start, *large]
    elif eight:
        helpers, large = _dmb_eight(g, n, mask)
        lines = [
            *start,
            "if (n <= 1) {",
            "\tfloat ws = f / (2.0 * R);",
            "\tfloat wc = 1.0 / (2.0 * R);",
            f"\t{g} m = $(name)_read(pq.x - 1.0, pq.y - t, ymaj);",
            f"\t{g} p = $(name)_read(pq.x + 1.0, pq.y + t, ymaj);",
            f"\t{g} c = $(name)_p(pq.x, pq.y, ymaj);",
            "\treturn $(name)_s8((m * ws + p * ws) + c * wc, " + _DMB_SUM + ");",
            "}",
            *large,
        ]
    else:
        helpers = (
            # Frame columns c and c + 1 at row r (any integer): three texels
            # in the image's row, or past the edge the frame's own row, a
            # fractional column over (four texels).
            f"void $(name)_cols(float c, float r, float t, float X0, bool ymaj,"
            f" out {g} a, out {g} b) {{\n"
            f"\tfloat k = floor(r / {n});\n"
            "\tif (k == 0.0) {\n"
            "\t\tfloat y = c + (r - X0) * t;\n"
            "\t\tfloat y0 = floor(y);\n"
            "\t\tfloat w = $(name)_q(y - y0);\n"
            f"\t\t{g} p1 = $(name)_p(r, y0 + 1.0, ymaj);\n"
            "\t\ta = mix($(name)_p(r, y0, ymaj), p1, w);\n"
            "\t\tb = mix(p1, $(name)_p(r, y0 + 2.0, ymaj), w);\n"
            "\t\treturn;\n"
            "\t}\n"
            f"\tfloat rm = r - k * {n};\n"
            f"\tfloat cf = c + k * {n} * t;\n"
            "\tfloat ca = floor(cf);\n"
            "\tfloat phi = $(name)_q(cf - ca);\n"
            "\tfloat y = ca + (rm - X0) * t;\n"
            "\tfloat y0 = floor(y);\n"
            "\tfloat w = $(name)_q(y - y0);\n"
            f"\t{g} p1 = $(name)_p(rm, y0 + 1.0, ymaj);\n"
            f"\t{g} p2 = $(name)_p(rm, y0 + 2.0, ymaj);\n"
            f"\t{g} f0 = mix($(name)_p(rm, y0, ymaj), p1, w);\n"
            f"\t{g} f1 = mix(p1, p2, w);\n"
            f"\t{g} f2 = mix(p2, $(name)_p(rm, y0 + 3.0, ymaj), w);\n"
            "\ta = mix(f0, f1, phi);\n"
            "\tb = mix(f1, f2, phi);\n"
            "}\n"
        )
        lines = [
            *start,
            "if (n <= 1) {",
            "\treturn (f * $(name)_read(pq.x - 1.0, pq.y - t, ymaj)",
            "\t\t+ $(name)_read(pq.x, pq.y, ymaj)",
            "\t\t+ f * $(name)_read(pq.x + 1.0, pq.y + t, ymaj)) / (2.0 * R);",
            "}",
            f"{g} acc = {g}(0.0);",
            "if (t == 0.0) {",
            # Merged end reads: the +n end when the first whole tap's
            # index is even, the -n end when odd; both when n = 2.
            "\tfloat h = f / (1.0 + f);",
            "\tfloat E = (1.0 + f) * ($(name)_q(h) - h);",
            "\tfloat par = mod(pq.x - float(n - 1), 2.0);",
            "\tbool mp = n == 2 || par < 0.5;",
            "\tbool mm = n == 2 || par > 0.5;",
            "\tfor (int k = -n; k <= n; k++) {",
            "\t\tfloat w = abs(k) == n ? f : 1.0;",
            "\t\tif ((k == n && mp) || (k == -n && mm)) w += E;",
            "\t\tif ((k == n - 1 && mp) || (k == 1 - n && mm)) w -= E;",
            "\t\tacc += w * $(name)_p(pq.x + float(k), pq.y, ymaj);",
            "\t}",
            "\treturn acc / (2.0 * R);",
            "}",
            "float ch = pq.y - (pq.x - X0) * t;",
            "float c0 = floor(ch);",
            f"{g} acc1 = {g}(0.0);",
            "for (int k = -n; k <= n; k++) {",
            "\tfloat w = abs(k) == n ? f : 1.0;",
            f"\t{g} a; {g} b;",
            "\t$(name)_cols(c0, pq.x + float(k), t, X0, ymaj, a, b);",
            "\tacc += w * a;",
            "\tacc1 += w * b;",
            "}",
            "return mix(acc, acc1, $(name)_q(ch - c0)) / (2.0 * R);",
        ]
    return _dmb_image_reads(g, n, eight=eight) + helpers, "\n".join(lines)


def _sharpen(node: Node, name: str, *, color: bool, size: int) -> Sub:
    """Sharpen: x + k (4 x - its four neighbors), k the intensity."""
    t = "rgba" if color else "f"
    k = word_params(node, [("intensity", 4, 1.0)], {})["intensity"]
    k = knob("intensity", "Intensity", k, 0.0, 4.0)
    e = num(1 / (1 << size))
    lines = [
        f"vec2 e = vec2({e}, 0.0);",
        "vec4 c = vec4($input(uv));",
        (
            "vec4 s = vec4($input(fract(uv + e))) + vec4($input(fract(uv - e)))"
            " + vec4($input(fract(uv + e.yx))) + vec4($input(fract(uv - e.yx)));"
        ),
        f"vec4 r = clamp(c + {num(k)} * (4.0 * c - s), 0.0, 1.0);",
        "return r;" if color else "return r.x;",
    ]
    return buffered(
        shader(name, [("input", t)], t, "\n".join(lines)),
        ["input"], {"input"}, size, precise={"input"},
    )  # fmt: skip


def _warp(node: Node, name: str, *, color: bool, size: int) -> Sub:
    """Warp: sample at uv + intensity / 256 times the gradient's slope.

    The slope is a backward difference over one texel, per UV unit.
    """
    t = "rgba" if color else "f"
    k = word_params(node, [("intensity", 5, 1.0)], {})["intensity"]
    k = knob("intensity", "Intensity", k, 0.0, 32.0)
    n = 1 << size
    lines = [
        f"vec2 e = vec2({num(1 / n)}, 0.0);",
        "float h = $gradient(uv);",
        "vec2 g = vec2(h - $gradient(fract(uv - e)), h - $gradient(fract(uv - e.yx)));",
        f"return $input(fract(uv + {num(k * n / 256)} * g));",
    ]
    made = shader(name, [("input", t), ("gradient", "f")], t, "\n".join(lines))
    return buffered(
        made, ["input", "gradient"], {"input", "gradient"}, size, precise={"gradient"}
    )


def _directionalwarp(node: Node, name: str, *, color: bool, size: int) -> Sub:
    """Directional warp: sample at uv + intensity / 256 map(uv) (cos, sin)."""
    t = "rgba" if color else "f"
    flags = node.fields[0]
    p = flag_params(
        flags, node.inline, [("intensity", 1, 10.0), ("angle", 3, 0.0)], 1, node
    )
    k = knob("warp_intensity", "Intensity", p["intensity"], 0.0, 64.0)
    turn = knob("angle", "Angle", p["angle"])
    a = f"6.28318531 * {num(turn)}"
    d = f"{num(k)} / 256.0 * vec2(cos({a}), sin({a}))"
    body = f"return $input(fract(uv + $intensity(uv) * {d}));"
    made = shader(name, [("input", t), ("intensity", "f")], t, body)
    return buffered(made, ["input", "intensity"], {"input"}, size)


def _normal(node: Node, name: str, *, color: bool, size: int) -> Sub:
    """Translate normal: normalize(-k dx, -k dy, 1) / 2 + 1/2, backward differences.

    k is the intensity times N / 256 (N the image width); flag 0x4 flips y
    (inverse dy); alpha is the height if flag 0x10 (input2alpha), else 1.
    """
    flags = node.fields[0]
    k = flag_params(flags & 3, node.inline, [("intensity", 0, 1.0)], 1, node)[
        "intensity"
    ]
    k = knob("intensity", "Intensity", k, 0.0, 8.0)
    height = knob("height_alpha", "Height in alpha", 1.0 if flags & 0x10 else 0.0)
    k *= (1 << size) / 256  # per texel at 256 px: resolution independent
    sy = 1.0 if flags & 4 else -1.0
    lines = [
        f"vec2 e = vec2({num(1 / (1 << size))}, 0.0);",
        "float h = $input(uv);",
        "vec2 g = vec2(h - $input(fract(uv - e)), h - $input(fract(uv - e.yx)));",
        f"vec3 n = normalize(vec3({num(-k)} * g.x, {num(sy * k)} * g.y, 1.0));",
        f"return vec4(n * 0.5 + 0.5, {num(height)} > 0.5 ? h : 1.0);",
    ]
    made = shader(name, [("input", "f")], "rgba", "\n".join(lines))
    return buffered(made, ["input"], {"input"}, size, precise={"input"})


def _emboss(node: Node, name: str, *, color: bool, size: int) -> Sub:
    """Emboss: light the input by the gradient image's slope.

    s = intensity N / 256 times the backward-difference slope along (cos,
    sin) of the light angle; out = in + max(s, 0) highlight - max(-s, 0) (1 - shadow).
    """
    t = "rgba" if color else "f"
    n = 4 if color else 1  # colors: four floats, or one (grayscale)
    p = flag_params(
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
    turn = knob("angle", "Light angle", p["angle"])
    p["intensity"] = knob("intensity", "Intensity", p["intensity"], 0.0, 16.0)
    hi = knob("highlight", "Highlight", p["highlight"])
    sh = knob("shadow", "Shadow", p["shadow"])
    a = f"6.28318531 * {num(turn)}"
    lines = [
        f"vec2 e = vec2({num(1 / (1 << size))}, 0.0);",
        "float h = $gradient(uv);",
        "vec2 g = vec2(h - $gradient(fract(uv - e)), h - $gradient(fract(uv - e.yx)));",
        f"vec2 light = vec2(cos({a}), sin({a}));",
        f"float s = {num(p['intensity'] * (1 << size) / 256)} * dot(g, light);",
    ]
    if color:
        lines += [
            "vec4 c = $input(uv);",
            (
                f"vec3 rgb = c.rgb + max(s, 0.0) * {vec(hi, 3)}"
                f" - max(-s, 0.0) * (vec3(1.0) - {vec(sh, 3)});"
            ),
            "return vec4(clamp(rgb, 0.0, 1.0), c.a);",
        ]
    else:
        lines.append(
            f"return clamp($input(uv) + max(s, 0.0) * {num(hi)}"
            f" - max(-s, 0.0) * (1.0 - {num(sh)}), 0.0, 1.0);"
        )
    made = shader(name, [("input", t), ("gradient", "f")], t, "\n".join(lines))
    return buffered(
        made, ["input", "gradient"], {"gradient"}, size, precise={"gradient"}
    )


def _pixelprocessor(node: Node, name: str, *, color: bool, size: int) -> Sub:
    """Pixel processor: its per-pixel function (the last code block) as GLSL.

    $pos is the pixel (uv); samples read the inputs, buffered. Its own
    seed (inline, or a function: the first block) replaces the graph's
    for the per-pixel generators (check_filters pp_rand_nodeseed).
    """
    count = node.fields[0]
    names = [f"input{n or ''}" for n in range(count)]
    params = node.code[:-1]
    state = (node.flags >> 6) & 3
    _CTX.pixel_seed = _seed_glsl()
    if state == 1:
        _CTX.value_seed = _CTX.pixel_seed = str(node.inline[0])
    elif state == 2:  # noqa: PLR2004
        _CTX.value_seed = "0"
        _CTX.value_seed = _CTX.pixel_seed = num(dynamic(node, params[0]))
        params = params[1:]
    _CTX.sampled = names
    try:
        value = dynamic(node, node.code[-1], params)
    finally:
        _CTX.sampled = None
    t = "rgba" if color else "f"
    body = f"return {vec(value, 4)};" if color else f"return {num(value)};"
    made = shader(name, [(nm, "rgba") for nm in names], t, body)
    return buffered(made, names, set(names), size, precise=set(names))


TORUS_GLSL = """float tdist_$(name)(vec2 a, vec2 b) {
\tvec2 d = abs(a - b);
\treturn length(min(d, vec2(1.0) - d));
}
"""


def _distance(node: Node, name: str, *, color: bool, size: int) -> Sub:
    """Translate distance: a falloff from the nearest mask pixel.

    1 - d / D from the nearest mask pixel (mask > 0.5), d the Euclidean
    distance (tiling) and D the distance parameter (/ 256 of the image);
    with a source, that times the source at the nearest mask pixel (for
    color, the source's alpha only: RGB is the source's). The
    nearest mask pixel is found by jump flooding: buffers of seed
    positions (nearest sampling), steps halving from the largest power of
    two that covers D down to one texel, then one more one-texel pass.
    """
    t = "rgba" if color else "f"
    flags = node.fields[0]
    has_source = bool(flags & 1)
    reach = flag_params(flags, node.inline, [("distance", 2, 10.0)], 1, node)[
        "distance"
    ]
    n = 1 << size
    reach_px = static(reach) * n / 256
    steps = []
    s = 1
    while s < min(reach_px, n / 2):
        s *= 2
    while s >= 1:
        steps.append(s)
        s //= 2
    steps.append(1)
    nodes: list[dict[str, Any]] = []
    wires: list[dict[str, Any]] = []
    init = shader(
        f"{name}_seed",
        [("mask", "f")],
        "rgba",
        "return $mask(uv) > 0.5 ? vec4(uv, 1.0, 1.0) : vec4(0.0);",
    )
    nodes.append(init)
    last = init["name"]
    for k, step in enumerate(steps):
        buf = buffer(f"{name}_jfa{k}", size, filtering=False)
        body = "\n".join(
            [
                "vec4 best = $input(uv);",
                "float bd = best.z > 0.5 ? tdist_$(name)(uv, best.xy) : 1e9;",
                "for (int y = -1; y <= 1; y++) {",
                "\tfor (int x = -1; x <= 1; x++) {",
                "\t\tvec2 o = vec2(float(x), float(y));",
                f"\t\tvec4 c = $input(fract(uv + o * {num(step / n)}));",
                "\t\tfloat d = tdist_$(name)(uv, c.xy);",
                "\t\tif (c.z > 0.5 && d < bd) { best = c; bd = d; }",
                "\t}",
                "}",
                "return best;",
            ]
        )
        jfa = shader(f"{name}_pass{k}", [("input", "rgba")], "rgba", body)
        jfa["shader_model"]["instance"] = TORUS_GLSL + jfa["shader_model"]["instance"]
        nodes += [buf, jfa]
        wires += [wire(last, buf["name"]), wire(buf["name"], jfa["name"])]
        last = jfa["name"]
    final_buf = buffer(f"{name}_near", size, filtering=False)
    lines = [
        "vec4 c = $input(uv);",
        f"float r = max({num(reach)} * {num(n / 256)}, 1e-6);",
        f"float d = tdist_$(name)(uv, c.xy) * {num(float(n))};",
        "float f = c.z > 0.5 ? clamp(1.0 - d / r, 0.0, 1.0) : 0.0;",
    ]
    inputs = [("input", "rgba")]
    if has_source:
        inputs.append(("source", t))
        if color:  # the falloff goes into alpha only
            lines += [
                "vec4 s = $source(f > 0.0 ? c.xy : uv);",
                "return vec4(s.rgb, s.a * f);",
            ]
        else:
            lines.append("return f * $source(c.xy);")
    else:
        lines.append("return vec4(vec3(1.0), f);" if color else "return f;")
    final = shader(name, inputs, t, "\n".join(lines))
    final["shader_model"]["instance"] = TORUS_GLSL + final["shader_model"]["instance"]
    nodes += [final_buf, final]
    wires += [wire(last, final_buf["name"]), wire(final_buf["name"], name, 0)]
    ports = [("mask", init["name"], 0)]
    if has_source:
        source_buf = buffer(f"{name}_source_buf", size)
        nodes.append(source_buf)
        wires.append(wire(source_buf["name"], name, 1))
        ports.append(("source", source_buf["name"], 0))
    return Sub(nodes, wires, ports, name)


FX_ARGS = "vec2 pos, float number, float depth, uint rs"
"""What an FX-Map function reads: $pos, $number, $depth and its node's
random state (docs/random-generator.md section 8)."""

_FX_CHILD = ((-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25))
"""Each Quadrant output's cell center, in parent cells from the parent's."""


def _fx_function(node: Node, offset: int, parameter: str = "frameoffset") -> str:
    """Return a GLSL call for the FX-Map function block at file ``offset``.

    $pos, $number and $depth are the tree function's arguments; rand is
    Substance's FX-Map generator (sbsar2ptex.fxrng), private to the call and
    started from its tree node's state ``e``.
    """
    block = decode(node.at(offset, (node.end - offset) // 4))
    _CTX.count += 1
    prefix = f"f{_CTX.count}"
    n = float(1 << _CTX.size)
    glsl = {
        0: "0.0",
        1: f"vec2({num(n)})",
        3: f"vec2({num(float(_CTX.size))})",
        8: "pos",
        10: "number",
        12: "depth",
    }

    def builtins(ident: int, width: int) -> str:
        if ident not in glsl:
            msg = f"FX-Map function reads built-in {ident}"
            raise UnsupportedError(msg)
        return glsl[ident]

    def rand(a: str, k: int) -> str:
        m, c = PARAM_LCG[parameter]
        return f"fx_rng_next(rng, float({a}), 0x{m:08X}u, 0x{c:08X}u)"

    try:
        lines, result = to_glsl(
            block,
            prefix,
            lambda uid, *_: _param_for(uid),
            builtins,
            _sample,
            rand,
            _CTX.fx_variables,
        )
    except FunctionError as e:
        raise UnsupportedError(str(e)) from e
    last = block.instrs[-1]
    t = glsl_type(last.type, last.width)
    if any(i.op == RAND for i in block.instrs):
        lines.insert(0, "uint rng = rs;")
    body = "\n\t".join(lines)
    _CTX.helpers.append(
        f"{t} {prefix}_$(name)({FX_ARGS}) {{\n\t{body}\n\treturn {result};\n}}\n"
    )
    return f"{prefix}_$(name)(pos, number, depth, e)"


def _fx_param(node: Node, param: Any, default: list[float], parameter: str) -> str:
    """Return GLSL for a tree parameter: its value, function, or ``default``."""
    width = len(default)
    if param is None:
        return vec(default, width)
    if param.state == fxmap.CODE:
        call = _fx_function(node, param.code, parameter)
        return f"vec{width}({call})" if width > 1 else f"float({call})"
    value = param.value * width if len(param.value) == 1 else param.value
    return vec(value, width)


Reach = tuple[float, float]
"""How far a tree node can draw from its $pos: a + b * its cell size."""


def _fx_reach(node: Node, tree: fxmap.Tree | None, size: int) -> dict[int, Reach]:
    """Return each tree node's reach (by id), for those with a finite one.

    Interval bounds on its parameters (sbsar2ptex.function.bounds), rand(a)
    within [0, a]; graph inputs, which can change, and $pos are unbounded,
    so what depends on them is never culled. A node used in several places
    takes the largest reach.
    """
    n = float(1 << size)
    out: dict[int, Reach] = {}
    inf = float("inf")
    memo: dict[tuple, Reach | None] = {}

    def span(param: Any, default: list[float], fixed: dict[int, list]) -> list:
        if param is None:
            return [(v, v) for v in default]
        if param.state != fxmap.CODE:
            vals = param.value * len(default) if len(param.value) == 1 else param.value
            return [(v, v) for v in vals]
        block = decode(node.at(param.code, (node.end - param.code) // 4))
        got = bounds(block, lambda _uid, width: [(-inf, inf)] * width, fixed)
        return got * len(default) if len(got) == 1 else got

    def length(iv: list) -> float:
        return math.sqrt(sum(max(abs(lo), abs(hi)) ** 2 for lo, hi in iv))

    def reach(t: fxmap.Tree | None, number: tuple, depth: tuple) -> Reach | None:
        if t is None:
            return (-inf, -inf)
        key = (id(t), number, depth)
        if key in memo:
            return memo[key]
        memo[key] = None  # (a cycle cannot happen; this guards recursion)
        fixed = {
            0: [(0.0, 0.0)],
            1: [(n, n)] * 2,
            3: [(float(size), float(size))] * 2,
            8: [(-inf, inf)] * 2,
            10: [number],
            12: [depth],
        }
        got: Reach | None
        if isinstance(t, fxmap.Quadrant):
            p = t.params
            bo = length(span(p.get("branchoffset"), [0.0, 0.0], fixed))
            fo = length(span(p.get("frameoffset"), [0.0, 0.0], fixed))
            sz = length(span(p.get("patternsize"), [1.0, 1.0], fixed)) / 2
            a, b = -inf, -inf
            if t.pattern is not None or t.image is not None:
                a, b = bo + fo, sz
            deeper = (depth[0] + 1, depth[1] + 1)
            for kid in {id(k): k for k in t.children if k is not None}.values():
                r = reach(kid, number, deeper)
                if r is None:
                    a = inf
                    continue
                a, b = max(a, bo + r[0]), max(b, math.sqrt(2) / 4 + r[1] / 2)
            got = (a, b)
        elif isinstance(t, fxmap.Iterate):
            count = t.count if isinstance(t.count, int) else None
            passes = (0.0, float(count - 1)) if count else (0.0, inf)
            r0, r1 = reach(t.first, number, depth), reach(t.repeated, passes, depth)
            got = (
                None
                if r0 is None or r1 is None
                else (max(r0[0], r1[0]), max(r0[1], r1[1]))
            )
        else:
            r0 = reach(t.if_false, number, depth)
            r1 = reach(t.if_true, number, depth)
            got = (
                None
                if r0 is None or r1 is None
                else (max(r0[0], r1[0]), max(r0[1], r1[1]))
            )
        if got is not None and not all(math.isfinite(x) or x == -inf for x in got):
            got = None
        memo[key] = got
        old = out.get(id(t), (-inf, -inf))
        out[id(t)] = (
            (inf, inf) if got is None else (max(old[0], got[0]), max(old[1], got[1]))
        )
        return got

    reach(tree, (0.0, 0.0), (0.0, 0.0))
    return {k: v for k, v in out.items() if math.isfinite(v[0]) or v[0] == -inf}


LIVE_SPLATS = 64
"""FX-Maps drawing more patterns than this are precomputed (fxsplat)."""

FX_RECORDS = 4096
"""The most patterns a recorded FX-Map draws (see _fxmaps); more are lost."""


def _fx_draw_bound(node: Node, tree: fxmap.Tree | None) -> int | None:
    """Return how many patterns the tree draws at most, None if unknown.

    Both branches of a Switch count; an Iterate count that is a function
    is evaluated with the graph's default inputs, and is unknown when it
    reads anything else (a variable, an image).
    """
    defaults = {
        uid: [float(x) for x in gi.default] or [0.0] for uid, gi in _CTX.inputs.items()
    }
    builtins = {0: [0.0], 1: [float(1 << _CTX.size)] * 2, 3: [float(_CTX.size)] * 2}

    def count(t: fxmap.Tree | None) -> int:
        if t is None:
            return 0
        if isinstance(t, fxmap.Quadrant):
            own = (
                t.pattern is not None
                or t.image is not None
                or t.image_index is not None
            )
            return int(own) + sum(count(k) for k in t.children)
        if isinstance(t, fxmap.Iterate):
            n = t.count
            if isinstance(n, fxmap.Param):
                block = decode(node.at(n.code, (node.end - n.code) // 4))
                n = int(evaluate(block, defaults, builtins, lambda a: a / 2, {})[0])
            return count(t.first) + max(n, 0) * count(t.repeated)
        return max(count(t.if_false), count(t.if_true))

    try:
        return count(tree)
    except (FunctionError, KeyError, IndexError):
        return None


def _fx_splats(
    node: Node,
    tree: fxmap.Tree | None,
    st: fxmap.Settings,
    layout: fxglsl.Layout,
) -> tuple[fxsplat.Packed | None, Path] | None:
    """Precompute the FX-Map's patterns into a data texture, when worth it.

    Only with a data folder (translate_graph's ``data_dir``), when the tree
    can be evaluated here and draws more than LIVE_SPLATS patterns, or its
    functions share variables (fxsplat.stateful), which per-pixel GLSL
    cannot follow. Returns
    the packing and the texture's file, else None (the map stays live).
    When drawing the splats would cost a pixel more than BAKE_WORK patterns
    and fxsplat.bake can, the texture is the drawn map instead (one float a
    texel), the packing None.
    """
    if _CTX.data_dir is None:
        return None
    defaults = {
        uid: [float(x) for x in gi.default] or [0.0] for uid, gi in _CTX.inputs.items()
    }
    color, size = layout.color, layout.size
    try:
        spl = fxsplat.splats(node, tree, st, defaults, color=color, size=size)
    except Exception as e:
        if fxsplat.evaluable(e):
            return None
        raise
    if len(spl) <= LIVE_SPLATS and not fxsplat.stateful(node, tree, st):
        return None
    _CTX.data_dir.mkdir(parents=True, exist_ok=True)
    path = _CTX.data_dir / f"fxmap_{node.index}.png"
    regions = (tuple(layout.render), tuple(layout.tiling))
    tile = (st.tile_x, st.tile_y)
    unit = tuple(fxsplat.UNIT_REGION)
    if fxsplat.pixel_work(spl, *regions, tile) > BAKE_WORK and regions == (unit, unit):
        baked = fxsplat.bake(spl, size, tile)
        if baked is not None:
            fxsplat.write_floats(path, baked)
            return None, path
    packed = fxsplat.pack(
        spl, color=color, tile=tile, render=regions[0], tiling=regions[1]
    )
    fxsplat.write_png(path, packed)
    return packed, path


BAKE_WORK = 2048
"""Patterns a pixel past which a precomputed FX-Map is baked, if it can be."""


def _fx_baked_lines() -> list[str]:
    """Return GLSL adding the baked map (fxsplat.bake), filtered bilinearly.

    The bake's pixel k is the map at k / n (fxsplat.bake's centers).
    """
    n = 1 << _CTX.size
    _CTX.helpers.append(
        "float fxbaked_$(name)(ivec2 t) {\n"
        f"\tt = (t % {n} + {n}) % {n};\n"
        f"\tvec4 s = floor($splats((vec2(t) + 0.5) / {num(float(n))}) * 255.0 + 0.5);\n"
        "\treturn uintBitsToFloat(uint(s.r) | (uint(s.g) << 8u)"
        " | (uint(s.b) << 16u) | (uint(s.a) << 24u));\n"
        "}\n"
    )
    return [
        f"vec2 bp = uv * {num(float(n))} - 0.5;",
        "ivec2 b0 = ivec2(floor(bp));",
        "vec2 bf = bp - floor(bp);",
        "float b00 = fxbaked_$(name)(b0), b10 = fxbaked_$(name)(b0 + ivec2(1, 0));",
        "float b01 = fxbaked_$(name)(b0 + ivec2(0, 1));",
        "float b11 = fxbaked_$(name)(b0 + ivec2(1, 1));",
        "acc.x += mix(mix(b00, b10, bf.x), mix(b01, b11, bf.x), bf.y);",
    ]


def _fx_splat_lines(packed: fxsplat.Packed, *, color: bool) -> list[str]:
    """Return GLSL drawing the pixel's cell's splats from the data texture."""
    rows = packed.image.shape[0]
    rec = fxsplat.RECORD[color]
    _CTX.helpers.append(
        "uint fxword_$(name)(int i) {\n"
        f"\tvec2 t = (vec2(float(i % {fxsplat.WIDTH}), float(i / {fxsplat.WIDTH}))"
        f" + 0.5) / vec2({num(float(fxsplat.WIDTH))}, {num(float(rows))});\n"
        "\tvec4 s = floor($splats(t) * 255.0 + 0.5);\n"
        "\treturn uint(s.r) | (uint(s.g) << 8u) | (uint(s.b) << 16u)"
        " | (uint(s.a) << 24u);\n"
        "}\n"
        "float fxfloat_$(name)(int i) {\n"
        "\treturn uintBitsToFloat(fxword_$(name)(i));\n"
        "}\n"
    )
    g = packed.grid
    col = (
        "vec4(fxfloat_$(name)(s + 8), fxfloat_$(name)(s + 9),"
        " fxfloat_$(name)(s + 10), fxfloat_$(name)(s + 11))"
        if color
        else "vec4(fxfloat_$(name)(s + 8))"
    )
    return [
        f"ivec2 gc = min(ivec2(floor(uv * {num(float(g))})), ivec2({g - 1}));",
        f"int cell = {packed.cells} + 2 * (gc.y * {g} + gc.x);",
        "int first = int(fxword_$(name)(cell));",
        "int count = int(fxword_$(name)(cell + 1));",
        "for (int k = 0; k < count; k++) {",
        f"\tint s = int(fxword_$(name)({packed.index} + first + k)) * {rec};",
        "\tvec2 c = vec2(fxfloat_$(name)(s), fxfloat_$(name)(s + 1));",
        "\tvec2 hs = vec2(fxfloat_$(name)(s + 2), fxfloat_$(name)(s + 3));",
        "\tbool mx = fxfloat_$(name)(s + 7) > 0.5;",
        "\tfloat a = fxfloat_$(name)(s + 4);",
        "\tint id = int(fxfloat_$(name)(s + 5));",
        f"\tfxdraw_$(name)(P, c, hs, a, id, fxfloat_$(name)(s + 6), mx, {col}, acc);",
        "}",
    ]


def _fxmaps(node: Node, name: str, *, color: bool, size: int) -> Sub:
    """FX-Map: drawn (_fxmap_drawn) through its render and tiling regions.

    The render region is the part of pattern space the image shows; the
    tiling region the tile of it that repeats (fxdraw.copies). Both are
    constants here: inline, or their functions evaluated with the graph's
    defaults.
    """
    st = fxmap.read_settings(node, color=color)
    render = tiling = fxsplat.UNIT_REGION
    if st.render_roi is not None or st.tiling_roi is not None:
        defaults = {
            uid: [float(x) for x in gi.default] or [0.0]
            for uid, gi in _CTX.inputs.items()
        }
        try:
            render, tiling = fxsplat.regions(node, st, defaults, size)
        except Exception as e:
            if fxsplat.evaluable(e):
                msg = "an FX-Map's regions, not evaluable here"
                raise UnsupportedError(msg) from e
            raise
    fmt = node.flags >> 4 & 3
    layout = fxglsl.Layout(
        size=size,
        render=tuple(render),
        tiling=tuple(tiling),
        tile=(st.tile_x, st.tile_y),
        color=color,
        bits={0: 8, 1: 16}.get(fmt),
        amd=_FX_AMD,
    )
    return _fxmap_drawn(node, name, layout)


_FX_AMD = False
"""Draw FX-Maps by an AMD GPU's rules (stores, mip means), not NVIDIA's.

Internal: for checking the shader against renders made on AMD.
"""


@dataclass
class _FxGroups:
    """An FX-Map's drawing groups (fxdraw.order), as the tree can draw them."""

    keys: list[tuple]
    """Each group's (pattern rank, max blending), in drawing order."""
    images: list[int]
    """The pattern inputs, in the order of their first appearance."""

    def key(self, pid: int, *, blend: bool) -> tuple:
        """Return a pattern's group key (fxdraw.order's, by static appearance)."""
        if pid >= 0:
            return (pid, blend)
        k = fxmap.image_parts(pid)[0]
        return (1000 + len(self.images) - self.images.index(k), blend)

    def index(self, pid: int, *, blend: bool) -> int:
        """Return a pattern's group's number."""
        return self.keys.index(self.key(pid, blend=blend))


def _fx_groups(tree: fxmap.Tree | None, n_images: int) -> tuple[_FxGroups, dict]:
    """Return a tree's drawing groups, and each node's (by id) groups below it.

    Image inputs appear in the tree's static order (a Quadrant, then its
    children; an Iterate's first output, then its repeated one; a Switch's
    false output, then its true one); one drawn through an index function
    may be any of them, the ones never named appearing last.
    """
    order: list[int] = []
    drawn: list[tuple[int | None, bool]] = []  # (pattern id or None: any image)
    seen: set[int] = set()

    def visit(t: fxmap.Tree | None) -> None:
        if t is None or id(t) in seen:
            return
        seen.add(id(t))
        if isinstance(t, fxmap.Quadrant):
            if t.image_index is not None:
                drawn.append((None, t.blend_max))
            elif t.image is not None and t.image < n_images:
                if t.image not in order:
                    order.append(t.image)
                drawn.append((fxmap.image_id(t.image, t.image_flags), t.blend_max))
            elif t.pattern is not None:
                drawn.append((t.pattern, t.blend_max))
            for c in t.children:
                visit(c)
        elif isinstance(t, fxmap.Iterate):
            visit(t.first)
            visit(t.repeated)
        else:
            visit(t.if_false)
            visit(t.if_true)

    visit(tree)
    order += [k for k in range(n_images) if k not in order]
    groups = _FxGroups([], order)

    def keys_of(pid: int | None, *, blend: bool) -> set[tuple]:
        if pid is None:
            return {groups.key(-1 - k, blend=blend) for k in range(n_images)}
        return {groups.key(pid, blend=blend)}

    every = set()
    for pid, blend in drawn:
        every |= keys_of(pid, blend=blend)
    groups.keys = sorted(every)
    below: dict[int, set[int]] = {}

    def under(t: fxmap.Tree | None) -> set[int]:
        if t is None:
            return set()
        if id(t) in below:
            return below[id(t)]
        below[id(t)] = set()  # (a guard; trees have no cycles)
        got: set[int] = set()
        if isinstance(t, fxmap.Quadrant):
            if t.image_index is not None:
                got = {groups.keys.index(k) for k in keys_of(None, blend=t.blend_max)}
            elif t.image is not None and t.image < n_images:
                pid = fxmap.image_id(t.image, t.image_flags)
                got = {groups.index(pid, blend=t.blend_max)}
            elif t.pattern is not None:
                got = {groups.index(t.pattern, blend=t.blend_max)}
            for c in t.children:
                got |= under(c)
        elif isinstance(t, fxmap.Iterate):
            got = under(t.first) | under(t.repeated)
        else:
            got = under(t.if_false) | under(t.if_true)
        below[id(t)] = got
        return got

    under(tree)
    return groups, below


def _fx_group_glsl(groups: _FxGroups) -> str:
    """Return ``fxgroup_$(name)(id, mx)``: a pattern's drawing group's number."""
    lines = ["int fxgroup_$(name)(int id, bool mx) {", "\tint key;"]
    lines.append("\tif (id >= 0) key = 2 * id + int(mx);")
    lines.append("\telse {")
    lines.append(f"\t\tint k = (-1 - id) % {fxmap.IMAGE_STRIDE};")
    lines.append("\t\tkey = -1;")
    for pos, k in enumerate(groups.images):
        rank = 1000 + len(groups.images) - pos
        lines.append(f"\t\tif (k == {k}) key = 2 * {rank} + int(mx);")
    lines.append("\t}")
    for g, (rank, blend) in enumerate(groups.keys):
        lines.append(f"\tif (key == {2 * rank + int(blend)}) return {g};")
    lines += ["\treturn -1;", "}", ""]
    return "\n".join(lines)


def _fx_mips(tree: fxmap.Tree | None, n_images: int) -> list[int]:
    """Return the pattern inputs the tree can draw in mip levels (filtering 0)."""
    out: set[int] = set()
    seen: set[int] = set()

    def visit(t: fxmap.Tree | None) -> None:
        if t is None or id(t) in seen:
            return
        seen.add(id(t))
        if isinstance(t, fxmap.Quadrant):
            if t.image_flags >> 2 & 3 == fxmap.MIPMAPPED:
                if t.image_index is not None:
                    out.update(range(n_images))
                elif t.image is not None and t.image < n_images:
                    out.add(t.image)
            for c in t.children:
                visit(c)
        elif isinstance(t, fxmap.Iterate):
            visit(t.first)
            visit(t.repeated)
        else:
            visit(t.if_false)
            visit(t.if_true)

    visit(tree)
    return sorted(out)


def _fx_mip_atlas(
    name: str, k: int, image_buf: str, layout: fxglsl.Layout
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Return the mip atlas of a map's pattern input ``k``: nodes, wires, last buffer.

    One shader and buffer a pass (fxglsl.mip_passes): two up to 4096 px.
    """
    size, bits, amd = layout.size, layout.bits, layout.amd
    nodes, wires = [], []
    prev = image_buf
    for first, last in fxglsl.mip_passes(size):
        sname = f"{name}_mip{k}_{first}"
        _CTX.helpers = [
            fxglsl.val_glsl() if bits is not None else "",
            fxglsl.levels_glsl(bits),
            fxglsl.mean_glsl(bits, amd=amd),
            fxglsl.mip_readers(size, first, last),
        ]
        made = shader(
            sname,
            [("input", "rgba")],
            "rgba",
            "\n".join(fxglsl.mip_body(size, first, last, copy_below=first > 1)),
        )
        buf = f"{sname}_buf"
        nodes += [made, buffer(buf, size, filtering=False)]
        wires += [wire(prev, sname), wire(sname, buf)]
        prev = buf
    return nodes, wires, prev


def _fxmap_drawn(node: Node, name: str, layout: fxglsl.Layout) -> Sub:
    """FX-Map: its tree evaluated per pixel (see sbsar2ptex.fxmap), drawn.

    Drawing follows sbsar2ptex.fxdraw (fxglsl): each pattern's tile copies,
    coverage, value and blending into an accumulator, stored once.

    Each tree node is a GLSL function drawing what it draws over the pixel
    into ``acc`` (premultiplied rgba; gray in x), its children's functions
    defined first. The tree is walked once per drawing group present
    (pattern and blending, fxdraw.order), each walk drawing only its
    group's patterns, the tree's order kept within it. A node returns at
    once when nothing it draws can reach the pixel (_fx_reach), or none of
    the walk's group is below it.

    A map whose functions share variables cannot skip any of the tree, nor
    walk it twice, so it is recorded instead: a small buffer holds each
    pattern it draws (three texels: center and half size; angle, pattern,
    supplement and blend; color), a pixel of it running the tree up to its
    pattern, and the map's shader draws the buffer's patterns, group by
    group, skipping those out of reach. Per pixel that is a loop over the
    patterns, not over the tree's every function.
    """
    color, size = layout.color, layout.size
    tree = fxmap.read_tree(node, color=color)
    st = fxmap.read_settings(node, color=color)
    weights = fxsplat.depth_weights(st.hurst, fxmap.max_depth(tree))
    _CTX.helpers.append(
        "float fxw_$(name)(float depth) {\n"
        + "".join(
            f"\tif (depth < {num(d + 0.5)}) return {num(w)};\n"
            for d, w in enumerate(weights[:-1])
        )
        + f"\treturn {num(weights[-1])};\n}}\n"
    )
    opacity = num(fxdraw.f32(st.opacity))
    images = [f"pattern{n}" for n in range(len(st.patterns))]
    groups, below = _fx_groups(tree, len(images))
    mips = _fx_mips(tree, len(images))
    t = "rgba" if color else "f"
    names: dict[int, str] = {}
    # Functions that share variables run in Substance's order, every one of
    # them, on per-pixel globals: no node may be skipped (it could set one).
    stateful = fxsplat.stateful(node, tree, st)
    reaches = {} if stateful else _fx_reach(node, tree, size)
    splats = _fx_splats(node, tree, st, layout)
    recording = stateful and splats is None
    walks = len(groups.keys) if not recording else 1
    masked = 1 < walks <= 32  # noqa: PLR2004

    def instance(op: str) -> list[str]:
        """Return GLSL lines setting ``col``, the instance color, from ``op``."""
        if color:
            return [
                f"float al = {op}.a * fxw_$(name)(depth) * {opacity};",
                f"vec4 col = vec4({op}.rgb * al, al);",
            ]
        return [f"vec4 col = vec4({op}.w * fxw_$(name)(depth) * {opacity});"]

    def emit(tree: fxmap.Tree | None) -> str | None:
        if tree is None:
            return None
        if id(tree) in names:
            return names[id(tree)]
        lines: list[str] = []
        if recording:  # its pattern is recorded: the rest need not run
            lines.append("if (fxi_$(name) < 0) return;")
        if masked:
            mask = sum(1 << g for g in below.get(id(tree), set()))
            if mask != (1 << walks) - 1:
                lines.append(f"if (({mask}u >> uint(grp) & 1u) == 0u) return;")
        args = "uv, {pos}, {cell}, {depth}, {number}, {st}, grp, acc"
        reach = reaches.get(id(tree))
        if reach is not None:  # nothing it draws can reach this pixel: skip
            a, b = (max(x, 0.0) for x in reach)
            lines.append(
                f"if (!fxnear_$(name)(uv, pos, {num(a)} + {num(b)} * cell)) return;"
            )
        # The node's random state: its incoming state plus its own seed.
        incoming = (
            "st"
            if not isinstance(tree, (fxmap.Iterate, fxmap.Switch)) or tree.inherited
            else "0u"
        )
        lines.append(f"uint e = {incoming} + {tree.seed or 0}u;")
        if isinstance(tree, fxmap.Quadrant):
            kids = [emit(c) for c in tree.children]
            p = tree.params
            draw_id: int | str | None = tree.pattern
            if tree.image is not None and tree.image < len(images):
                draw_id = fxmap.image_id(tree.image, tree.image_flags)
            elif tree.image is not None:
                draw_id = None
            draws = draw_id is not None or tree.image_index is not None
            # Its parameters in Substance's order (opacity, branch and frame
            # offsets, size, rotation, supplement, image index): their
            # functions may set variables the next ones read. Walked per
            # group, what only draws is evaluated in its group's walk.
            params = []
            for local, key, default in (
                ("vec4 op", "opacity", [1.0] * 4),
                ("vec2 bo", "branchoffset", [0.0, 0.0]),
                ("vec2 fo", "frameoffset", [0.0, 0.0]),
                ("vec2 sz", "patternsize", [1.0, 1.0]),
                ("float rot", "patternrotation", [0.0]),
                ("float sup", "patternsuppl", [0.0]),
            ):
                if draws or key == "branchoffset" or (stateful and key in p):
                    value = _fx_param(node, p.get(key), default, key)
                    params.append((key, f"{local} = {value};"))
            if tree.image_index is not None:
                index = _fx_function(node, tree.image_index.code, "imageindex")
                strided = fxmap.image_id(0, tree.image_flags) + 1
                params.append(
                    ("imageindex", f"int image = {strided - 1} - int({index});")
                )
                draw_id = "image"
            mx = "true" if tree.blend_max else "false"
            if recording:
                lines += [ln for _, ln in params]
                lines.append("vec2 p = pos + bo;")
                if draw_id is not None:
                    lines += [
                        *instance("op"),
                        "if (fxi_$(name) == 0) {",
                        "\tfxr0_$(name) = vec4(p + fo, cell * sz * 0.5);",
                        (
                            "\tfxr1_$(name) = vec4(rot * 6.28318531,"
                            f" float({draw_id}), sup, {float(tree.blend_max)});"
                        ),
                        "\tfxr2_$(name) = col;",
                        "}",
                        "fxi_$(name)--;",
                    ]
            else:
                lines += [ln for key, ln in params if key == "branchoffset"]
                lines.append("vec2 p = pos + bo;")
                if draw_id is not None:
                    if isinstance(draw_id, int):
                        cond = f"grp == {groups.index(draw_id, blend=tree.blend_max)}"
                        inner = [ln for key, ln in params if key != "branchoffset"]
                    else:
                        cond = f"fxgroup_$(name)(image, {mx}) == grp"
                        lines += [ln for key, ln in params if key == "imageindex"]
                        inner = [
                            ln for key, ln in params
                            if key not in ("branchoffset", "imageindex")
                        ]  # fmt: skip
                    lines += [
                        f"if ({cond}) {{",
                        *(f"\t{ln}" for ln in inner),
                        *(f"\t{ln}" for ln in instance("op")),
                        (
                            "\tfxdraw_$(name)(uv, p + fo, cell * sz * 0.5,"
                            f" rot * 6.28318531, {draw_id}, sup, {mx}, col, acc);"
                        ),
                        "}",
                    ]
            shared = len(set(map(id, tree.children))) == 1 and kids and kids[0]
            # Each slot's state: the child depth's transform, once more per slot.
            if shared:
                lines += [
                    "uint sp = e;",
                    "for (int c = 0; c < 4; c++) {",
                    "\tsp = fx_depth_step(sp, depth + 1.0);",
                    "\tvec2 o = vec2(float(c % 2) - 0.5, float(c / 2) - 0.5) * 0.5;",
                    f"\t{kids[0]}("
                    + args.format(
                        pos="p + o * cell",
                        cell="cell * 0.5",
                        depth="depth + 1.0",
                        number="number",
                        st="sp",
                    )
                    + ");",
                    "}",
                ]
            elif any(kids):
                lines.append("uint sp = e;")
                for kid, (ox, oy) in zip(kids, _FX_CHILD, strict=False):
                    lines.append("sp = fx_depth_step(sp, depth + 1.0);")
                    if kid:
                        lines.append(
                            f"{kid}("
                            + args.format(
                                pos=f"p + vec2({num(ox)}, {num(oy)}) * cell",
                                cell="cell * 0.5",
                                depth="depth + 1.0",
                                number="number",
                                st="sp",
                            )
                            + ");"
                        )
        elif isinstance(tree, fxmap.Iterate):
            first, repeated = emit(tree.first), emit(tree.repeated)
            if isinstance(tree.count, fxmap.Param):
                count = f"int({_fx_function(node, tree.count.code, 'iteratecount')})"
            else:
                count = str(tree.count)
            same = args.format(
                pos="pos", cell="cell", depth="depth", number="number", st="e"
            )
            lines.append(f"int n = {count};")  # first: it may set variables
            if first:
                lines.append(f"{first}({same});")
            if repeated:
                # Each pass's state: one more Iterate step.
                lines += [
                    "uint ps = e;",
                    "for (int i = 0; i < n; i++) {",
                    "\tps = fx_iter_step(ps);",
                    f"\t{repeated}("
                    + args.format(
                        pos="pos",
                        cell="cell",
                        depth="depth",
                        number="float(i)",
                        st="ps",
                    )
                    + ");",
                    "}",
                ]
        else:
            if_false, if_true = emit(tree.if_false), emit(tree.if_true)
            if isinstance(tree.value, fxmap.Param):
                cond = f"bool({_fx_function(node, tree.value.code, 'switch')})"
            else:
                cond = "true" if tree.value else "false"
            same = args.format(
                pos="pos", cell="cell", depth="depth", number="number", st="e"
            )
            lines.append(f"if ({cond}) {{")
            if if_true:
                lines.append(f"\t{if_true}({same});")
            lines.append("} else {")
            if if_false:
                lines.append(f"\t{if_false}({same});")
            lines.append("}")
        fname = f"fxn{len(names)}_$(name)"
        names[id(tree)] = fname
        body = "\n\t".join(lines)
        _CTX.helpers.append(
            f"void {fname}(vec2 uv, vec2 pos, float cell, float depth,"
            f" float number, uint st, int grp, inout vec4 acc) {{\n\t{body}\n}}\n"
        )
        return fname

    def draw_helpers() -> None:
        """Add the patterns and drawing helpers (before what draws)."""
        _CTX.helpers.append(fxglsl.draw_glsl(layout, images, mips))
        _CTX.helpers.append(_fx_group_glsl(groups))

    if not recording:  # the tree's functions draw
        draw_helpers()
    if FX_RNG_GLSL not in _CTX.globals:
        _CTX.globals.append(FX_RNG_GLSL)
    # The root's state: the FX-Map's random seed (a function of $randomseed,
    # or its own value).
    seed = str(st.seed or 0)
    if st.seed_code is not None:
        seed = f"int({num(dynamic(node, st.seed_code))})"
    inputs = [(nm, "rgba") for nm in images]
    root = None
    start: list[str] = []
    _CTX.sampled_read = set()
    if splats is None:
        _CTX.sampled = images  # functions may sample the pattern inputs
        _CTX.sample_nearest = True
        if stateful:
            shared = fxsplat.variable_types(node, tree, st)
            _CTX.fx_variables = {k: (f"fxv{k}_$(name)", t) for k, t in shared.items()}
            _CTX.helpers.append(
                "".join(f"{t} fxv{k}_$(name);\n" for k, t in shared.items())
                + "int fxi_$(name);\nvec4 fxr0_$(name), fxr1_$(name), fxr2_$(name);\n"
            )
        try:
            root = emit(tree)
            if stateful:  # the per-pixel start: variables zero, then the setup
                start = [f"fxv{k}_$(name) = {t}(0);" for k, t in shared.items()]
                if st.setup_code is not None:
                    setup = _fx_function(node, st.setup_code)
                    start.append(
                        setup.replace(
                            "(pos, number, depth, e)", "(vec2(0.5), 0.0, 0.0, 0u)"
                        )
                        + ";"
                    )
        finally:
            _CTX.sampled = None
            _CTX.sample_nearest = False
            _CTX.fx_variables = None

    def run(grp: int, at: str = "P") -> str:
        return f"{root}({at}, vec2(0.5), 1.0, 0.0, 0.0, uint({seed}), {grp}, acc);"

    recorder: dict[str, Any] | None = None
    if recording:
        # The record buffer: pixel 3k + g holds pattern k's texel g; past
        # the last pattern drawn, an end mark (blend -1). Room for twice the
        # patterns the default inputs draw: inputs can raise a count.
        bound = _fx_draw_bound(node, tree)
        records = FX_RECORDS if bound is None else min(max(2 * bound, 64), FX_RECORDS)
        side = max(1, math.ceil(math.log2(3 * (records + 1)) / 2))
        n = num(float(1 << side))
        recorder = shader(
            f"{name}_records",
            inputs,
            "rgba",
            "\n".join(
                [
                    f"ivec2 px = ivec2(floor(uv * {n}));",
                    f"int at = px.y * {1 << side} + px.x;",
                    "fxi_$(name) = at / 3;",
                    "fxr0_$(name) = vec4(0.0);",
                    "fxr1_$(name) = vec4(0.0, 0.0, 0.0, -1.0);",
                    "fxr2_$(name) = vec4(0.0);",
                    "vec4 acc = vec4(0.0);",
                    *start,
                    run(0, "vec2(0.0)"),
                    "int g = at % 3;",
                    (
                        "return g == 0 ? fxr0_$(name)"
                        " : (g == 1 ? fxr1_$(name) : fxr2_$(name));"
                    ),
                ]
            ),
        )
        draw_helpers()
        _CTX.helpers.append(
            "vec2 fxrt_$(name)(int i) {\n"
            f"\treturn (vec2(float(i % {1 << side}), float(i / {1 << side}))"
            f" + 0.5) / {n};\n}}\n"
        )
    lines = [*fxglsl.pixel_lines(layout), "vec4 acc = vec4(0.0);"]
    if isinstance(st.background, int):
        inputs.append(("background", t))
        # (as stored: an image's levels, through Material Maker's half floats)
        bg = "fxlvval_$(name)(fxlvin_$(name)(vec4($background(uv))))"
        lines.append(f"acc = {bg};" if color else f"acc.x = {bg}.x;")
    elif st.background is not None:
        lines.append(
            f"acc = {vec(st.background, 4)};"
            if color
            else f"acc.x = {num(st.background[0])};"
        )
    if recording:
        inputs.append(("records", "rgba"))
        groups_loop = len(groups.keys) > 1
        if groups_loop:
            lines.append(f"for (int grp = 0; grp < {len(groups.keys)}; grp++) {{")
        lines += [
            f"for (int k = 0; k < {records}; k++) {{",
            "\tvec4 r1 = $records(fxrt_$(name)(3 * k + 1));",
            "\tif (r1.w < -0.5) break;",
            "\tvec4 r0 = $records(fxrt_$(name)(3 * k));",
            "\tbool mx = r1.w > 0.5;",
        ]
        if groups_loop:
            lines.append("\tif (fxgroup_$(name)(int(r1.y), mx) != grp) continue;")
        lines += [
            "\tif (!fxnear_$(name)(P, r0.xy, length(r0.zw) * 1.001 + 1e-6)) continue;",
            "\tvec4 r2 = $records(fxrt_$(name)(3 * k + 2));",
            "\tfxdraw_$(name)(P, r0.xy, r0.zw, r1.x, int(r1.y), r1.z, mx, r2, acc);",
            "}",
        ]
        if groups_loop:
            lines.append("}")
    elif root:
        lines += [*start, *(run(g) for g in range(walks))]
    if splats is not None:
        packed, path = splats
        inputs.append(("splats", "rgba"))
        lines += (
            _fx_baked_lines()
            if packed is None
            else _fx_splat_lines(packed, color=color)
        )
    mip_inputs = [(f"mip{k}", "rgba") for k in mips]
    inputs[len(images) : len(images)] = mip_inputs
    lines.append(
        "return fxstore_$(name)(acc);" if color else "return fxstore_$(name)(acc).x;"
    )
    _CTX.helpers.append(fxglsl.store_glsl(layout))
    made = shader(name, inputs, t, "\n".join(lines))
    # (16-bit float buffers: an 8-bit image's levels come back exactly
    # from them, fxglsl.levels_glsl, as from Material Maker's images; a
    # function's samples, as a pixel processor's, read 32-bit ones: a 16-bit
    # input's value, which can place a pattern, to the level)
    sub = buffered(
        made, [nm for nm, _ in inputs], set(images), size, precise=_CTX.sampled_read
    )
    _CTX.sampled_read = set()
    for k in mips:
        nodes, wires_, last = _fx_mip_atlas(name, k, f"{name}_pattern{k}_buf", layout)
        sub.nodes += nodes
        sub.wires += [*wires_, wire(last, name, len(images) + mips.index(k))]
        sub.ports = [pt for pt in sub.ports if pt[0] != f"mip{k}"]
    if recorder is not None:
        # The recorder reads the map's (buffered) images too.
        buf = f"{name}_records_buf"
        sub.nodes += [recorder, buffer(buf, side, filtering=False)]
        sub.wires += [
            wire(recorder["name"], buf),
            wire(buf, name, [nm for nm, _ in inputs].index("records")),
            *(
                wire(f"{name}_{nm}_buf", recorder["name"], port)
                for port, nm in enumerate(images)
            ),
        ]
        sub.ports = [pt for pt in sub.ports if pt[0] != "records"]
    if splats is not None:
        data = f"{name}_splats"
        port = [nm for nm, _ in inputs].index("splats")
        sub.nodes.append(
            {
                "name": data,
                "type": "image",
                "node_position": {"x": 0, "y": 0},
                "parameters": {"image": path.as_posix()},
                "seed_int": 0,
            }
        )
        sub.wires.append(wire(data, name, port))
        sub.ports = [pt for pt in sub.ports if pt[0] != "splats"]
    sub.rounded = layout.bits is not None  # (fxstore stores it)
    return sub


TRANSLATORS = {
    0: _gradient,
    1: _gradient,
    2: _blend,
    3: _blend,
    4: _transformation,
    5: _transformation,
    6: _grayscale,
    7: _shuffle,
    8: _fxmaps,
    9: _fxmaps,
    12: _uniform,
    14: _warp,
    15: _warp,
    16: _emboss,
    17: _emboss,
    20: _blur,
    21: _blur,
    22: _dirmotionblur,
    23: _dirmotionblur,
    24: _directionalwarp,
    25: _directionalwarp,
    26: _sharpen,
    27: _sharpen,
    13: _uniform,
    29: _hsl,
    37: _normal,
    40: _pixelprocessor,
    41: _pixelprocessor,
    42: _distance,
    43: _distance,
    30: _levels,
    31: _levels,
    38: _dyngradient,
    39: _dyngradient,
    44: _curve,
    45: _curve,
}
"""Each op's translator: (node, MM node name, *, color, size log2) -> Made."""


def _bspline(name: str, size: int, t: str) -> dict[str, Any]:
    """Return a shader reading its input through a cubic B-spline.

    The input is a buffer ``size`` in log2 pixels; four linear fetches, each
    placed between two texels so that it weighs them as the spline does.
    """
    n = num(float(1 << size))
    g = "vec4" if t == "rgba" else "float"
    lines = [
        f"vec2 p = uv * {n} - 0.5;",
        "vec2 i = floor(p);",
        "vec2 f = p - i;",
        "vec2 w0 = (1.0 - f) * (1.0 - f) * (1.0 - f) / 6.0;",
        "vec2 w1 = (4.0 - 6.0 * f * f + 3.0 * f * f * f) / 6.0;",
        "vec2 w3 = f * f * f / 6.0;",
        "vec2 w2 = 1.0 - w0 - w1 - w3;",
        "vec2 g0 = w0 + w1;",
        "vec2 g1 = w2 + w3;",
        f"vec2 h0 = (i - 1.0 + w1 / g0 + 0.5) / {n};",
        f"vec2 h1 = (i + 1.0 + w3 / g1 + 0.5) / {n};",
        f"{g} a = $input(fract(h0)), b = $input(fract(vec2(h1.x, h0.y)));",
        f"{g} c = $input(fract(vec2(h0.x, h1.y))), d = $input(fract(h1));",
        "return g0.y * (g0.x * a + g1.x * b) + g1.y * (g0.x * c + g1.x * d);",
    ]
    return shader(name, [("input", t)], t, "\n".join(lines))


def _output_size(node: Node, size: int) -> int:
    """Return ``node``'s output size (log2), ``size`` unless the node says.

    A node without a size function (node word bit 16 clear) has a fixed
    size: the node word's second byte, log2 width (low nibble) and height
    (high nibble), whatever the graph's size. The size function is the
    node's first code block, or the block its first inline word points at
    (node_shader drops that word); a gradient's or a curve's first block is
    its data instead. Square sizes only, one pixel at least. A graph
    input's image is its own size.
    """
    if node.op in (0, 1, 32, 33, 44, 45):
        return size
    if not node.flags & 1 and node.words:
        fixed = node.words[0] >> 8 & 0xFF
        return fixed & 15 if fixed & 15 == fixed >> 4 else size
    offsets = list(node.code[:1])
    if not offsets and node.inline and node.start < node.inline[0] + BASE < node.end:
        offsets = [node.inline[0] + BASE]
    defaults = {
        uid: [float(x) for x in gi.default] or [0.0] for uid, gi in _CTX.inputs.items()
    }
    builtins = {1: [float(1 << size)] * 2, 3: [float(size)] * 2}
    for off in offsets:
        try:
            block = decode(node.at(off, (node.end - off) // 4))
            v = evaluate(block, defaults, builtins, lambda a: a / 2, {})
        except (FunctionError, KeyError, IndexError):
            return size
        if len(v) == 2 and v[0] == v[1] and float(v[0]).is_integer():  # noqa: PLR2004
            return max(int(v[0]), 0)  # one pixel at least
    return size


STORE_STEPS = {0: 255, 1: 65535}
"""A node's output format (node word flag bits 4-5) -> its integer steps:
8 and 16 bits; the float formats (2, 3) keep what they are given."""


def stored(made: Sub, node: Node) -> bool:
    """Make ``made``'s output read as Substance stores ``node``'s: in its format.

    Substance keeps every node's image in the node's format, so an 8-bit
    node's value is rounded to 1/255 before anything reads it, a 16-bit
    one's to 1/65535 (thresholds downstream turn the difference into whole
    pixels). Ties between 8-bit steps break down, as on NVIDIA GPUs (see
    _levels). The output shader computes the unrounded value as
    ``$(name)_raw`` and its function returns that rounded. Returns whether
    it does (exact_buffers keeps such values exact in buffers).
    """
    steps = STORE_STEPS.get(node.flags >> 4 & 3)
    out = next((n for n in made.nodes if n["name"] == made.output), None)
    if out is not None and out.get("type") == "graph":  # its output's shader
        last = next(
            (w["from"] for w in out["connections"] if w["to"] == "gen_outputs"), None
        )
        out = next((n for n in out["nodes"] if n["name"] == last), None)
    if steps is None or out is None or out.get("type") != "shader":
        return False
    m = out["shader_model"]
    kind = m["outputs"][0]["type"]
    if m.get("code") or m["outputs"][0].get(kind) != "$(name)_fct($uv)":
        return False
    ret, g = ("vec4", "vec4") if kind == "rgba" else ("float", "float")
    k = num(float(steps))
    tie = (
        f"ceil(v * {k} - 0.5) / {k}"
        if steps == 255  # noqa: PLR2004
        else f"floor(v * {k} + 0.5) / {k}"
    )
    m["instance"] = m["instance"].replace("$(name)_fct(", "$(name)_raw(") + (
        f"{ret} $(name)_fct(vec2 uv) {{\n"
        f"{g} v = clamp($(name)_raw(uv), 0.0, 1.0);\nreturn {tie};\n}}\n"
    )
    return True


def exact_buffers(
    mm: list[dict[str, Any]], wires: list[dict[str, Any]], rounded: set[str]
) -> None:
    """Make the buffers of ``rounded`` nodes' outputs 32-bit.

    A 16-bit float holds k / 255 only nearly: half-opacity blends of such
    values land on 8-bit ties, which then round by the sign of that error
    instead of down (Biomech's reaction-diffusion loop drifts from its
    first Highpass). A 32-bit float holds them exactly enough.
    """
    by_name = {n["name"]: n for n in mm}
    for w in wires:
        target = by_name.get(w["to"])
        if w["from"] in rounded and target is not None and target["type"] == "buffer":
            target["parameters"]["f32"] = True


def _seed_block(node: Node) -> int:
    """Return the offset of the code block computing ``node``'s random seed.

    The seed function returns one integer. A node with a dynamic output size
    too has that function's block as well (two integers), after the seed's
    or before it.
    """
    for offset in reversed(node.code):
        last = decode(node.at(offset, (node.end - offset) // 4)).instrs[-1]
        if last.type != FUNC_FLOAT and last.width == 1:
            return offset
    msg = f"no seed function in node {node.index}"
    raise UnsupportedError(msg)


def node_shader(
    node: Node, name: str, size: int = 11, template: Node | None = None
) -> Made:
    """Return the shader node for ``node`` and its input names, in port order.

    With ``template``, a library graph's node ``node`` is an instance of:
    its settings are read from it (functions of the library node's
    parameters: translate_graph's ``library``), the seed and the wiring are
    ``node``'s.
    """
    translate = TRANSLATORS.get(node.op)
    if translate is None:
        msg = f"op {node.op}"
        raise UnsupportedError(msg)
    # FX-Maps and pixel processors seed their own generators.
    seed_state = 0 if node.op in (8, 9, 40, 41) else (node.flags >> 6) & 3
    translated = node
    if seed_state == 1:
        _CTX.value_seed = str(node.inline[0])
        translated = replace(node, inline=node.inline[1:])
    elif seed_state == 2:  # noqa: PLR2004
        _CTX.value_seed = "0"  # not the last node's, inside the seed's function
        _CTX.value_seed = num(dynamic(node, _seed_block(node)))
    else:
        _CTX.value_seed = "0"
    if template is not None:
        translated = replace(template, index=node.index, inputs=node.inputs)
        if (template.flags >> 6) & 3 == 1:  # its own seed, inline
            translated = replace(translated, inline=template.inline[1:])
    if translated.inline:
        target = translated.inline[0] + BASE
        if translated.start < target < translated.end:
            block = decode(translated.at(target, (translated.end - target) // 4))
            last = block.instrs[-1]
            if (  # dynamic output size
                last.type != FUNC_FLOAT and last.width == 2  # noqa: PLR2004
            ):
                translated = replace(translated, inline=translated.inline[1:])
    return translate(translated, name, color=translated.op in COLOR_OPS, size=size)


def graph_inputs(asm: Any, xml: str) -> dict[int, GraphInput]:
    """Return a cooked graph's non-image inputs by uid: identifier, type, default.

    ``asm`` is the sbsasm.Asm; ``xml`` the archive's description (names and
    labels).
    """
    names = dict(re.findall(r'<input uid="(\d+)" identifier="([^"]*)"', xml))
    labels = dict(
        re.findall(r'<input uid="(\d+)"[^>]*>\s*<inputgui[^>]*label="([^"]*)"', xml)
    )
    out: dict[int, GraphInput] = {}
    for uid, kind in asm.inputs:
        if kind == 5 or str(uid) not in names:  # noqa: PLR2004 (images)
            continue
        out[uid] = GraphInput(
            names[str(uid)],
            kind,
            list(asm.defaults.get(uid, [])),
            labels.get(str(uid), ""),
        )
    return out


def translate_graph(
    nodes: list[Node],
    outputs: dict[str, int],
    images: dict[int, str],
    size: int = 11,
    inputs: dict[int, GraphInput] | None = None,
    data_dir: Path | None = None,
    origins: dict[str, int] | None = None,
    aliases: dict[str, str] | None = None,
    library: dict[int, tuple[Node, dict[int, GraphInput]]] | None = None,
    loops: dict[int, Loop] | None = None,
    fold: Fold | None = None,
    plugins: Plugins | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, tuple[str, int]]]:
    """Return (MM nodes, connections, output name -> (MM node, port)).

    ``images`` gives each graph input's uid an image file for its image node;
    ``inputs`` the graph's (non-image) inputs by uid, which functions read:
    they become parameters of the shaders that use them, exposed together on
    a "parameters" remote node. Only the nodes the outputs need are
    translated. ``origins``, if given, is filled with each Material Maker
    node's Substance node (by name); ``aliases`` with each node merged away
    (merge_same, drop_copies) and the one computing its image now.
    ``library`` gives the nodes of library node instances whose settings
    were recovered (libparams) their template node and the inputs its
    functions read: they are translated from it (node_shader), the
    library node's parameters live knobs of the instance. ``loops``
    (sbsar2ptex.loops), by their last node, are made one node each in
    place of all theirs, which must be library ones. ``fold`` is a last
    pass over the finished translation (loops.Chains). ``plugins`` makes
    nodes sbs plugin nodes where it can (sbsar2ptex.plugin), the others
    translated as they are without it.
    """
    global _CTX  # noqa: PLW0603 (one graph at a time)
    if fold is not None and origins is None:
        origins = {}  # the fold finds a node's nodes by them
    _CTX = _Context(size=size, inputs=dict(inputs or {}), data_dir=data_dir)
    # Functions reading $outputsize see the size the graph renders at, not
    # the archive's default for it.
    for uid, gi in _CTX.inputs.items():
        if gi.identifier == "$outputsize":
            _CTX.inputs[uid] = replace(gi, default=[size, size])
    need: set[int] = set()
    todo = list(outputs.values())
    while todo:
        k = todo.pop()
        if k not in need:
            need.add(k)
            todo += nodes[k].inputs.values()
    mm: list[dict[str, Any]] = []
    wires: list[dict[str, Any]] = []
    out_of: dict[int, str] = {}
    # A node's own output size: a smaller one is rendered into a buffer that
    # size, which larger nodes read through a cubic B-spline (Substance's
    # enlargement; bilinear leaves its texels' corners).
    sizes = {k: _output_size(nodes[k], size) for k in need}
    # a loop's nodes all one size, or it stays unrolled
    loops = {
        k: loop
        for k, loop in (loops or {}).items()
        if k in need
        and k in (library or {})
        and len({sizes[m] for m in loop.members}) == 1
    }
    folded = {m for k, loop in loops.items() for m in loop.members if m != k}
    levels: dict[tuple[int, int], str] = {}
    rounded: set[str] = set()  # outputs stored in 8 or 16 bits (stored)
    smooth: dict[int, str] = {}

    def fed(m: int, own: int) -> str:
        """Return input ``m`` as a node ``own`` in size reads it.

        Smaller, it is enlarged (_bspline). Larger, it is averaged down: each
        halving buffer samples the one above at its texels' shared corners,
        so linear filtering averages them two by two (a mipmap). A
        transformation reads it in the level its matrix picks (transform_feed),
        no smaller than ``own``: pass that as ``own``.
        """
        if sizes[m] < own:
            if m not in smooth:
                smooth[m] = f"n{m}_smooth"
                t = "rgba" if nodes[m].op in COLOR_OPS else "f"
                mm.append(_bspline(smooth[m], sizes[m], t))
                wires.append(wire(out_of[m], smooth[m]))
            return smooth[m]
        if sizes[m] == own:
            return out_of[m]
        src = out_of[m]
        # the top level is the input itself, rendered (once) at its size
        top = sizes[m] if sizes[m] == size else sizes[m] - 1
        for s in range(top, own - 1, -1):
            if (m, s) not in levels:
                levels[m, s] = f"n{m}_level{s}"
                mm.append(buffer(levels[m, s], s))
                wires.append(wire(src, levels[m, s]))
            src = levels[m, s]
        return src

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
            out_of[k] = name
            if origins is not None:
                origins[name] = k
            continue
        if k in folded:
            continue
        own = sizes[k]
        _CTX.size = own
        loop = loops.get(k)
        sources = loop.inputs if loop is not None else node.inputs
        if library and k in library:
            template, knobs = library[k]
            graph_wide = _CTX.inputs
            # the template's $outputsize and $randomseed are the graph's own
            by_id = {gi.identifier: gi for gi in graph_wide.values()}
            _CTX.inputs = graph_wide | {
                uid: by_id.get(gi.identifier, gi) if gi.identifier[:1] == "$" else gi
                for uid, gi in knobs.items()
            }
            try:
                if loop is not None:
                    made = loop.make(name, nodes, own)
                else:
                    made = plugins and plugins(node, name, own, template)
                    made = made or node_shader(node, name, own, template)
            finally:
                _CTX.inputs = graph_wide
        else:
            made = plugins and plugins(node, name, own, None)
            made = made or node_shader(node, name, own)
        if not isinstance(made, Sub):
            shader_node, names = made
            made = Sub(
                [shader_node], [], [(n, name, i) for i, n in enumerate(names)], name
            )
        for n in made.nodes:  # a plugin node's graph inputs (shader() does it)
            for p in n.get(PLUGIN_LINKS, {}):
                if p.startswith(GRAPH_PARAM):
                    _CTX.exposed.setdefault(p, []).append(n["name"])
        if (
            made.rounded
            or stored(made, node)
            or (loop is not None and loop.rounds(nodes))
        ):
            rounded.add(made.output)
        if own != size:
            made.nodes.append(buffer(f"{name}_size", own))
            made.wires.append(wire(made.output, f"{name}_size"))
            made.output = f"{name}_size"
        mm += made.nodes
        if origins is not None:
            origins.update((n["name"], k) for n in made.nodes)
        wires += made.wires
        for inp, to, port in made.ports:
            m = sources[inp]
            at = transform_feed(node, sizes[m], own) if node.op in (4, 5) else own
            wires.append(wire(fed(m, at), to, port))
        out_of[k] = made.output
    mm, wires = merge_buffers(mm, wires)
    mm, wires = cache_shared(mm, wires, size, origins)
    # After caching: a node read inline by each of its copies' readers stays
    # read inline, at full precision, not through a (16-bit) cache.
    mm, wires, merged = merge_same(mm, wires, origins)
    out_of = {k: merged.get(v, v) for k, v in out_of.items()}
    mm, wires, moved = drop_copies(mm, wires, origins)
    out_of = {k: moved.get(v, v) for k, v in out_of.items()}
    exact_buffers(mm, wires, {merged.get(n, n) for n in rounded})
    if fold is not None and origins is not None:
        mm, wires = fold(mm, wires, origins, out_of, _CTX.exposed)
    if aliases is not None:
        aliases.update({k: moved.get(v, v) for k, v in merged.items()})
        aliases.update(moved)
    for row, made in enumerate(mm):
        made["node_position"] = {"x": 220 * (row % 12), "y": 180 * (row // 12)}
    if _CTX.exposed:
        mm.append(remote(_CTX.exposed, mm))
    return mm, wires, {o: (out_of[k], 0) for o, k in outputs.items()}


def output_type(node: dict[str, Any]) -> str:
    """Return a Material Maker node's output type: "f" or "rgba"."""
    if node["type"] == "shader":
        return node["shader_model"]["outputs"][0]["type"]
    if node["type"] == "graph":
        outs = next(c for c in node["nodes"] if c["name"] == "gen_outputs")
        return outs["ports"][0]["type"]
    if is_plugin(node):
        from sbsar2ptex.plugin import output_kind  # noqa: PLC0415 (it imports this)

        return output_kind(node["type"])
    return "rgba"


def declared_params(mm: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the parameters shader nodes declare, inside subgraphs too.

    And a subgraph's named parameters (an iterate's count reads one: loops).
    """
    out: dict[str, dict[str, Any]] = {}
    for node in mm:
        for p, link in node.get(PLUGIN_LINKS, {}).items():
            out.setdefault(p, link["decl"])
        if node["type"] == "shader":
            for prm in node["shader_model"]["parameters"]:
                out.setdefault(prm["name"], prm)
        elif node["type"] == "remote":
            for w in node["widgets"]:
                if w["type"] == "named_parameter":
                    out.setdefault(w["name"], {**w, "type": "float"})
        elif node["type"] == "graph":
            out.update(
                (k, v)
                for k, v in declared_params(node["nodes"]).items()
                if k not in out
            )
    return out


def subgraph(
    name: str,
    label: str,
    inner: list[dict[str, Any]],
    wires: list[dict[str, Any]],
    inputs: list[tuple[str, str, int]],
    output: tuple[str, int, str],
) -> dict[str, Any]:
    """Return ``inner`` as one Material Maker graph node named ``name``.

    ``inputs`` are (type, inner node, port) per input port; ``output`` the
    inner (node, port, type) it outputs. Every parameter of its nodes is one
    control on it, linked to all the nodes declaring it (a blur's two
    passes, one intensity); graph inputs among them are exposed as the
    graph node's, by the name they had.
    """
    declared = declared_params(inner)
    users: dict[str, list[str]] = {}
    for n in inner:
        if n["type"] == "shader":
            for prm in n["shader_model"]["parameters"]:
                users.setdefault(prm["name"], []).append(n["name"])
    for p, us in _CTX.exposed.items():  # the graph's inputs: now read here
        kept = [u for u in us if u not in {n["name"] for n in inner}]
        if len(kept) != len(us):
            _CTX.exposed[p] = [*kept, name]
    ports_in = [{"name": f"port{k}", "type": t} for k, (t, _, _) in enumerate(inputs)]
    links = [
        {"from": "gen_inputs", "from_port": k, "to": node, "to_port": port}
        for k, (_, node, port) in enumerate(inputs)
    ]
    node, port, kind = output
    links.append({"from": node, "from_port": port, "to": "gen_outputs", "to_port": 0})
    io = {"node_position": {"x": 0, "y": 0}, "parameters": {}, "type": "ios"}
    controls = {
        "name": "gen_parameters",
        "type": "remote",
        "node_position": {"x": 0, "y": -200},
        "parameters": {p: declared[p]["default"] for p in users},
        "seed_int": 0,
        "widgets": [
            {
                "name": p,
                "label": declared[p]["label"],
                "type": "linked_control",
                "linked_widgets": [{"node": u, "widget": p} for u in us],
            }
            for p, us in users.items()
        ],
    }
    for k, n in enumerate(inner):
        n["node_position"] = {"x": 220 * k, "y": 0}
    return {
        "name": name,
        "label": label,
        "type": "graph",
        "node_position": {"x": 0, "y": 0},
        "parameters": {p: declared[p]["default"] for p in users},
        "seed_int": 0,
        "nodes": [
            *inner,
            {"name": "gen_inputs", "ports": ports_in, **io},
            {"name": "gen_outputs", "ports": [{"name": "port0", "type": kind}], **io},
            controls,
        ],
        "connections": [*wires, *links],
    }


def remote(exposed: dict[str, list[str]], mm: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a remote node exposing each parameter across the nodes declaring it."""
    declared = declared_params(mm)
    by_name = {n["name"]: n for n in mm}
    widgets, values = [], {}
    for k, (pname, users) in enumerate(sorted(exposed.items())):
        prm = declared[pname]
        widgets.append(
            {
                "name": f"param{k}",
                "label": prm["label"],
                "type": "linked_control",
                "linked_widgets": [
                    {"node": u, "widget": w}
                    for u in users
                    for w in widgets_of(by_name.get(u), pname)
                ],
            }
        )
        values[f"param{k}"] = prm["default"]
    return {
        "name": "parameters",
        "type": "remote",
        "node_position": {"x": -400, "y": 0},
        "parameters": values,
        "seed_int": 0,
        "widgets": widgets,
    }
