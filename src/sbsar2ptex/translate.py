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

Parameters driven by function bytecode ("dynamic") are translated to helper
functions, including Substance's context-specific random generators.
"""

import itertools
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


@dataclass
class _Context:
    """What translating the current graph needs to know and collects."""

    size: int = 11
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
    value_seed: str = "0"
    """Resolved random seed of the ordinary filter currently being translated."""
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


def _float_param(name: str, label: str, default: float, *, integer: bool) -> None:
    """Declare a float parameter for the next shader node."""
    _CTX.params[name] = {
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


def _param_for(uid: int) -> str:
    """Declare graph input ``uid`` as a parameter of the next shader; return GLSL.

    Floats and ints become float parameters, float4s colors; other vectors a
    float parameter per component.
    """
    gi = _CTX.inputs.get(uid)
    if gi is None:
        msg = f"function reads unknown input {uid:#x}"
        raise UnsupportedError(msg)
    name = param_name(gi.identifier)
    label = gi.label or gi.identifier
    integer = gi.type >= IN_INT
    if gi.type == IN_FLOAT4:
        r, g, b, a = (list(gi.default) + [0.0] * 4)[:4]
        _CTX.params[name] = {
            "name": name,
            "label": label,
            "type": "color",
            "default": {"r": r, "g": g, "b": b, "a": a},
        }
        return f"${name}"
    if gi.type in (IN_FLOAT, IN_INT):
        default = float(gi.default[0]) if gi.default else 0.0
        _float_param(name, label, default, integer=integer)
        return f"int(${name})" if integer else f"${name}"
    width = _INPUT_GLSL.get(gi.type, 1)
    parts = []
    for k in range(width):
        pname = f"{name}_{'xyzw'[k]}"
        default = float(gi.default[k]) if k < len(gi.default) else 0.0
        _float_param(pname, f"{label} {'xyzw'[k]}", default, integer=integer)
        parts.append(f"int(${pname})" if integer else f"${pname}")
    return f"{'ivec' if gi.type >= IN_INT2 else 'vec'}{width}({', '.join(parts)})"


def _sample(kind: str, index: int, pos: str, filtering: int) -> str:
    """Return GLSL sampling a pixel processor's input ``index`` at ``pos``.

    Filtering 0 is bilinear (the inputs are buffered with filtering);
    otherwise the nearest texel. Positions wrap.
    """
    if _CTX.sampled is None or index >= len(_CTX.sampled):
        msg = f"sample of input {index} outside a pixel processor"
        raise UnsupportedError(msg)
    n = num(float(1 << _CTX.size))
    at = f"fract({pos})"
    if filtering:
        at = f"(floor(fract({pos}) * {n}) + 0.5) / {n}"
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
        lines.insert(0, f"uint rng = sbs_rng_init(uint({_seed_glsl()}), uv);")
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
        return by_name.get(name, {}).get("type", "")

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

    for name in [n["name"] for n in mm if n["type"] in ("shader", "graph")]:
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


def box_glsl_live(t: str, direction: str, radius: str, size: int) -> str:
    """Return a GLSL body: the box of box_taps, its radius (texels) a GLSL value.

    ``direction`` is a GLSL vec2 in texels. The taps are counted in the
    shader, so the radius can be a parameter or a function.
    """
    g = "vec4" if t == "rgba" else "float"
    n = num(float(1 << size))
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
    g = "vec4" if color else "float"
    lines = [
        f"{g} x = $input(uv);",
        f"{g} lo = {vec(p['in_low'], n)};",
        f"{g} hi = {vec(p['in_high'], n)};",
        f"{g} m = {vec(p['in_mid'], n)};",
        # an empty input range thresholds, strictly, on the input as Substance
        # stored it: in 8 bits a computed 0.5 is 127/255, below a 0.5
        # threshold (ties break down, as on NVIDIA, whose renders match
        # Adobe's baked maps; an AMD GPU rounds them up)
        (
            "x = mix(clamp((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0),"
            " 1.0 - step(ceil(x * 255.0 - 0.5) / 255.0, lo),"
            " step(hi - lo, vec4(1e-6)));"
            if color
            else (
                "x = hi - lo <= 1e-6 ? 1.0 - step(ceil(x * 255.0 - 0.5) / 255.0, lo)"
                " : clamp((x - lo) / (hi - lo), 0.0, 1.0);"
            )
        ),
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
    if state_m == 1:
        matrix, inline = [f32(w) for w in inline[:4]], inline[4:]
    elif state_m == 2:  # noqa: PLR2004
        matrix, inline = list(dynamic(node, inline[0] + BASE)), inline[1:]
    if state_o == 1:
        offset = [f32(w) for w in inline[:2]]
    elif state_o == 2:  # noqa: PLR2004
        offset = dynamic(node, inline[0] + BASE)
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
    lines.append("return $input(fract(q));")
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
    k = knob("intensity", "Intensity", p["intensity"], 0.0, 32.0)
    x = shader(
        f"{name}_x", [("input", t)], t,
        box_glsl_live(t, "vec2(1.0, 0.0)", f"{num(k)} * {scale}", size),
    )  # fmt: skip
    # Read again: a function's helper goes with the shader made next.
    p = word_params(node, [("intensity", 4, 10.0)], {})
    k = knob("intensity", "Intensity", p["intensity"], 0.0, 32.0)
    y = shader(
        name, [("input", t)], t,
        box_glsl_live(t, "vec2(0.0, 1.0)", f"{num(k)} * {scale}", size),
    )  # fmt: skip
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


def _dirmotionblur(node: Node, name: str, *, color: bool, size: int) -> Sub:
    """Directional motion blur: the box along (cos, sin) of the angle (turns)."""
    t = "rgba" if color else "f"
    p = flag_params(
        node.fields[0],
        node.inline,
        [("intensity", 0, 10.0), ("angle", 2, 0.0)],
        1,
        node,
    )
    k = knob("intensity", "Intensity", p["intensity"], 0.0, 32.0)
    turn = knob("angle", "Angle", p["angle"])
    a = f"6.28318531 * {num(turn)}"
    made = shader(
        name, [("input", t)], t,
        box_glsl_live(
            t, f"vec2(cos({a}), sin({a}))", f"{num(k)} * {num((1 << size) / 256)}",
            size,
        ),
    )  # fmt: skip
    return buffered(made, ["input"], {"input"}, size)


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
    (inverse dy); alpha is the height.
    """
    flags = node.fields[0]
    k = flag_params(flags & 3, node.inline, [("intensity", 0, 1.0)], 1, node)[
        "intensity"
    ]
    k = knob("intensity", "Intensity", k, 0.0, 8.0)
    k *= (1 << size) / 256  # per texel at 256 px: resolution independent
    sy = 1.0 if flags & 4 else -1.0
    lines = [
        f"vec2 e = vec2({num(1 / (1 << size))}, 0.0);",
        "float h = $input(uv);",
        "vec2 g = vec2(h - $input(fract(uv - e)), h - $input(fract(uv - e.yx)));",
        f"vec3 n = normalize(vec3({num(-k)} * g.x, {num(sy * k)} * g.y, 1.0));",
        "return vec4(n * 0.5 + 0.5, h);",
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

    $pos is the pixel (uv); samples read the inputs, buffered.
    """
    count = node.fields[0]
    names = [f"input{n or ''}" for n in range(count)]
    params = node.code[:-1]
    if (node.flags >> 6) & 3 == 2:  # noqa: PLR2004 (a seed function: the first block)
        _CTX.value_seed = "0"
        _CTX.value_seed = num(dynamic(node, params[0]))
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

# The signature stays on one GLSL line: Material Maker rewrites it line by line.
FX_DRAW_GLSL = (
    "vec4 fxdraw_$(name)(vec2 uv, vec2 c, vec2 hs, float a, int id, float s,"
    " bool mx) {\n"
    """\tvec2 d0 = uv - c;
\td0 -= round(d0) * $(tile);
\tvec2 h = max(abs(hs), vec2(1e-8));
\tfloat ext = length(h);
\tint k = ext > 0.5 ? int(ceil(ext - 0.5)) : 0;
\tint kx = $(tile).x > 0.5 ? k : 0;
\tint ky = $(tile).y > 0.5 ? k : 0;
\tfloat ca = cos(a), sa = sin(a);
\tvec4 v = vec4(0.0);
\tfor (int j = -ky; j <= ky; j++) {
\t\tfor (int i = -kx; i <= kx; i++) {
\t\t\tvec2 d = d0 + vec2(float(i), float(j));
\t\t\tvec2 q = vec2(ca * d.x + sa * d.y, -sa * d.x + ca * d.y) / h;
\t\t\tvec4 p = vec4(0.0);
\t\t\tif (id >= 0) {
\t\t\t\tp = vec4(fxpat_$(name)(id, q, s));
\t\t\t} else if (abs(q.x) <= 1.0 && abs(q.y) <= 1.0) {
\t\t\t\tp = fximg_$(name)(-1 - id, (q + 1.0) * 0.5);
\t\t\t}
\t\t\tv = mx ? max(v, p) : v + p;
\t\t}
\t}
\treturn v;
}
"""
)
"""A pattern (id >= 0) or image (-1 - index) drawn at c, wrapping."""

FX_NEAR_GLSL = """bool fxnear_$(name)(vec2 uv, vec2 p, float r) {
\tvec2 d = uv - p;
\td -= round(d) * $(tile);
\treturn dot(d, d) <= r * r;
}
"""
"""Whether the pixel is within r of p (wrapping as the FX-Map tiles)."""

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
    *,
    color: bool,
    size: int,
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
    try:
        spl = fxsplat.splats(node, tree, st, defaults, color=color, size=size)
    except fxsplat.RegionError as e:
        raise UnsupportedError(str(e)) from e
    except Exception as e:
        if fxsplat.evaluable(e):
            return None
        raise
    if len(spl) <= LIVE_SPLATS and not fxsplat.stateful(node, tree, st):
        return None
    _CTX.data_dir.mkdir(parents=True, exist_ok=True)
    path = _CTX.data_dir / f"fxmap_{node.index}.png"
    if fxsplat.pixel_work(spl) > BAKE_WORK:
        baked = fxsplat.bake(spl, size, (st.tile_x, st.tile_y))
        if baked is not None:
            fxsplat.write_floats(path, baked)
            return None, path
    packed = fxsplat.pack(spl, color=color, tile=(st.tile_x, st.tile_y))
    fxsplat.write_png(path, packed)
    return packed, path


BAKE_WORK = 2048
"""Patterns a pixel past which a precomputed FX-Map is baked, if it can be."""


def _fx_baked_lines() -> list[str]:
    """Return GLSL adding the baked map (fxsplat.bake), filtered bilinearly."""
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
    lines = [
        f"ivec2 gc = min(ivec2(floor(fract(uv) * {num(float(g))})), ivec2({g - 1}));",
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
        "\tvec4 v = fxdraw_$(name)(uv, c, hs, a, id, fxfloat_$(name)(s + 6), mx);",
    ]
    if color:
        lines += [
            "\tvec2 rg = vec2(fxfloat_$(name)(s + 8), fxfloat_$(name)(s + 9));",
            "\tvec2 ba = vec2(fxfloat_$(name)(s + 10), fxfloat_$(name)(s + 11));",
            "\tvec4 src = v * vec4(rg, ba);",
            "\tacc.a = src.a + acc.a * (1.0 - src.a);",
            "\tvec3 over = src.rgb * src.a + acc.rgb * (1.0 - src.a);",
            "\tacc.rgb = mx ? over : acc.rgb + src.rgb * src.a;",
        ]
    else:
        lines += [
            "\tfloat value = v.x * fxfloat_$(name)(s + 8);",
            "\tacc.x = mx ? max(acc.x, value) : acc.x + value;",
        ]
    return [*lines, "}"]


def _fxmaps(node: Node, name: str, *, color: bool, size: int) -> Sub:
    """FX-Map: drawn (_fxmap_drawn) through its render and tiling regions.

    The tiling region is a tile of pattern space that wraps; the render
    region the part of pattern space the image shows. Equal, the image is
    the tile: precomputed patterns are placed in it (fxsplat.splats). A
    render region inside the default tile, a window onto it (make_it_tile
    stretches a column), is the tile drawn, then sampled through the window.
    """
    st = fxmap.read_settings(node, color=color)
    if st.render_roi is None and st.tiling_roi is None:
        return _fxmap_drawn(node, name, color=color, size=size, regions=False)
    defaults = {
        uid: [float(x) for x in gi.default] or [0.0] for uid, gi in _CTX.inputs.items()
    }
    try:
        render, tiling = fxsplat.regions(node, st, defaults, size)
    except Exception as e:
        if fxsplat.evaluable(e):
            msg = "an FX-Map's regions, not evaluable here"
            raise UnsupportedError(msg) from e
        raise
    unit = fxsplat.UNIT_REGION
    if render == tiling:
        return _fxmap_drawn(node, name, color=color, size=size, regions=render != unit)
    x0, x1, y0, y1 = render
    if tiling != unit or not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        msg = f"an FX-Map's render region {render} outside its tiling region {tiling}"
        raise UnsupportedError(msg)
    # The tile drawn finer by the window's magnification (up to four times),
    # so that sampling it keeps the patterns' edges.
    finer = size + min(2, max(0, math.ceil(-math.log2(min(x1 - x0, y1 - y0)))))
    sub = _fxmap_drawn(node, f"{name}_tile", color=color, size=finer, regions=False)
    t = "rgba" if color else "f"
    buf = f"{name}_tile_buf"
    window = shader(
        name, [("input", t)], t,
        f"return $input(fract(vec2({num(x0)}, {num(y0)})"
        f" + uv * vec2({num(x1 - x0)}, {num(y1 - y0)})));",
    )  # fmt: skip
    sub.nodes += [buffer(buf, finer), window]
    sub.wires += [wire(sub.output, buf), wire(buf, name)]
    sub.output = name
    return sub


def _fxmap_drawn(
    node: Node, name: str, *, color: bool, size: int, regions: bool
) -> Sub:
    """FX-Map: its tree evaluated per pixel (see sbsar2ptex.fxmap).

    With ``regions``, precomputed patterns are placed by the render region
    (equal to the tiling region); a map drawn per pixel cannot be.

    Each tree node is a GLSL function adding what it draws over the pixel
    to ``acc`` (premultiplied rgba; gray in x), its children's functions
    defined first. A node returns at once when nothing it draws can reach
    the pixel (_fx_reach); otherwise every pattern is visited.

    A map whose functions share variables cannot skip any of the tree, so
    it is recorded instead: a small buffer holds each pattern it draws
    (three texels: center and half size; angle, pattern, supplement and
    blend; color), a pixel of it running the tree up to its pattern, and
    the map's shader draws the buffer's patterns, skipping those out of
    reach. Per pixel that is a loop over the patterns, not over the
    tree's every function.
    """
    tree = fxmap.read_tree(node, color=color)
    st = fxmap.read_settings(node, color=color)
    if not regions:
        st = replace(st, render_roi=None, tiling_roi=None)
    deepest = fxmap.max_depth(tree)
    h = st.hurst
    if h == 0:
        weight = "1.0"
    elif h == 1:
        weight = num(1.0 / (deepest + 1))
    else:
        total = sum(h**j for j in range(deepest + 1))
        weight = f"pow({num(h)}, depth) * {num(1.0 / total)}"
    scale = f"{weight} * {num(st.opacity)}"
    images = [f"pattern{n}" for n in range(len(st.patterns))]
    t = "rgba" if color else "f"
    names: dict[int, str] = {}
    # Functions that share variables run in Substance's order, every one of
    # them, on per-pixel globals: no node may be skipped (it could set one).
    stateful = fxsplat.stateful(node, tree, st)
    reaches = {} if stateful else _fx_reach(node, tree, size)
    splats = _fx_splats(node, tree, st, color=color, size=size)
    if splats is None and regions:
        msg = "an FX-Map's render and tiling region, drawn per pixel"
        raise UnsupportedError(msg)
    recording = stateful and splats is None

    def emit(tree: fxmap.Tree | None) -> str | None:
        if tree is None:
            return None
        if id(tree) in names:
            return names[id(tree)]
        lines: list[str] = []
        if recording:  # its pattern is recorded: the rest need not run
            lines.append("if (fxi_$(name) < 0) return;")
        args = "uv, {pos}, {cell}, {depth}, {number}, {st}, acc"
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
            draw_id = tree.pattern if tree.pattern is not None else None
            if tree.image is not None and tree.image < len(images):
                draw_id = -1 - tree.image
            # Its parameters in Substance's order (opacity, branch and frame
            # offsets, size, rotation, supplement, image index): their
            # functions may set variables the next ones read.
            draws = draw_id is not None or tree.image_index is not None
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
                    lines.append(f"{local} = {value};")
            if tree.image_index is not None:
                index = _fx_function(node, tree.image_index.code, "imageindex")
                lines.append(f"int image = -1 - int({index});")
                draw_id = "image"
            lines.append("vec2 p = pos + bo;")
            if draw_id is not None and recording:
                mx = "1.0" if tree.blend_max else "0.0"
                lines += [
                    "if (fxi_$(name) == 0) {",
                    "\tfxr0_$(name) = vec4(p + fo, cell * sz * 0.5);",
                    (
                        "\tfxr1_$(name) = vec4(rot * 6.28318531,"
                        f" float({draw_id}), sup, {mx});"
                    ),
                    (
                        f"\tfxr2_$(name) = vec4(op.rgb, op.a * {scale});"
                        if color
                        else f"\tfxr2_$(name) = vec4(op.y * {scale});"
                    ),
                    "}",
                    "fxi_$(name)--;",
                ]
            elif draw_id is not None:
                mx = "true" if tree.blend_max else "false"
                lines += [
                    (
                        "vec4 v = fxdraw_$(name)(uv, p + fo, cell * sz * 0.5,"
                        f" rot * 6.28318531, {draw_id}, sup, {mx});"
                    ),
                ]
                if color:
                    lines += [
                        "vec4 src = v * op;",
                        f"src.a *= {scale};",
                        "acc.a = src.a + acc.a * (1.0 - src.a);",
                        (
                            "acc.rgb = src.rgb * src.a + acc.rgb * (1.0 - src.a);"
                            if tree.blend_max
                            else "acc.rgb += src.rgb * src.a;"
                        ),
                    ]
                else:
                    value = f"v.x * op.y * {scale}"
                    lines.append(
                        f"acc.x = max(acc.x, {value});"
                        if tree.blend_max
                        else f"acc.x += {value};"
                    )
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
            f" float number, uint st, inout vec4 acc) {{\n\t{body}\n}}\n"
        )
        return fname

    tile = f"vec2({1.0 if st.tile_x else 0.0}, {1.0 if st.tile_y else 0.0})"

    def draw_helpers() -> None:
        """Add the patterns and drawing helpers (before what draws)."""
        _CTX.helpers.append(fxmap.PATTERN_GLSL)
        sample = [
            f"\tif (k == {n}) return {'$' + nm}(fract(q));"
            for n, nm in enumerate(images)
        ]
        _CTX.helpers.append(
            "vec4 fximg_$(name)(int k, vec2 q) {\n"
            + "\n".join(sample)
            + "\n\treturn vec4(0.0);\n}\n"
        )
        _CTX.helpers.append(FX_DRAW_GLSL.replace("$(tile)", tile))
        _CTX.helpers.append(FX_NEAR_GLSL.replace("$(tile)", tile))

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
    if splats is None:
        _CTX.sampled = images  # functions may sample the pattern inputs
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
            _CTX.fx_variables = None
    run = f"{root}(uv, vec2(0.5), 1.0, 0.0, 0.0, uint({seed}), acc);"
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
                    run,
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
    lines = ["vec4 acc = vec4(0.0);"]
    if isinstance(st.background, int):
        inputs.append(("background", t))
        lines.append("acc = $background(uv);" if color else "acc.x = $background(uv);")
    elif st.background is not None:
        lines.append(
            f"acc = {vec(st.background, 4)};"
            if color
            else f"acc.x = {num(st.background[0])};"
        )
    if recording:
        inputs.append(("records", "rgba"))
        lines += [
            f"for (int k = 0; k < {records}; k++) {{",
            "\tvec4 r1 = $records(fxrt_$(name)(3 * k + 1));",
            "\tif (r1.w < -0.5) break;",
            "\tvec4 r0 = $records(fxrt_$(name)(3 * k));",
            "\tfloat ext = length(r0.zw);",  # patterns stay in their square
            (
                "\tif (ext <= 0.5 && !fxnear_$(name)(uv, r0.xy, ext * 1.001 + 1e-6))"
                " continue;"
            ),
            "\tvec4 r2 = $records(fxrt_$(name)(3 * k + 2));",
            "\tbool mx = r1.w > 0.5;",
            "\tvec4 v = fxdraw_$(name)(uv, r0.xy, r0.zw, r1.x, int(r1.y), r1.z, mx);",
        ]
        if color:
            lines += [
                "\tvec4 src = v * r2;",
                "\tacc.a = src.a + acc.a * (1.0 - src.a);",
                "\tvec3 over = src.rgb * src.a + acc.rgb * (1.0 - src.a);",
                "\tacc.rgb = mx ? over : acc.rgb + src.rgb * src.a;",
            ]
        else:
            lines += [
                "\tfloat value = v.x * r2.x;",
                "\tacc.x = mx ? max(acc.x, value) : acc.x + value;",
            ]
        lines.append("}")
    elif root:
        lines += [*start, run]
    if splats is not None:
        packed, path = splats
        inputs.append(("splats", "rgba"))
        lines += (
            _fx_baked_lines()
            if packed is None
            else _fx_splat_lines(packed, color=color)
        )
    lines.append(
        "return clamp(acc, 0.0, 1.0);" if color else "return clamp(acc.x, 0.0, 1.0);"
    )
    made = shader(name, inputs, t, "\n".join(lines))
    sub = buffered(made, [nm for nm, _ in inputs], set(images), size)
    if recorder is not None:
        # The recorder reads the map's (buffered) images too.
        buf = f"{name}_records_buf"
        sub.nodes += [recorder, buffer(buf, side, filtering=False)]
        sub.wires += [
            wire(recorder["name"], buf),
            wire(buf, name, len(inputs) - 1),
            *(
                wire(f"{name}_{nm}_buf", recorder["name"], port)
                for port, nm in enumerate(images)
            ),
        ]
        sub.ports = [pt for pt in sub.ports if pt[0] != "records"]
    if splats is not None:
        data = f"{name}_splats"
        port = len(inputs) - 1
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
    """Return ``node``'s output size (log2), ``size`` unless its function says.

    The size function is the node's first code block, or the block its first
    inline word points at (node_shader drops that word); a gradient's or a
    curve's first block is its data instead. Square sizes only, one pixel
    at least.
    """
    if node.op in (0, 1, 44, 45):
        return size
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


def node_shader(node: Node, name: str, size: int = 11) -> Made:
    """Return the shader node for ``node`` and its input names, in port order."""
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
        _CTX.value_seed = num(dynamic(node, node.code[-1]))
    else:
        _CTX.value_seed = "0"
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, tuple[str, int]]]:
    """Return (MM nodes, connections, output name -> (MM node, port)).

    ``images`` gives each graph input's uid an image file for its image node;
    ``inputs`` the graph's (non-image) inputs by uid, which functions read:
    they become parameters of the shaders that use them, exposed together on
    a "parameters" remote node. Only the nodes the outputs need are
    translated. ``origins``, if given, is filled with each Material Maker
    node's Substance node (by name).
    """
    global _CTX  # noqa: PLW0603 (one graph at a time)
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
    levels: dict[tuple[int, int], str] = {}
    smooth: dict[int, str] = {}

    def fed(m: int, own: int) -> str:
        """Return input ``m`` as a node ``own`` in size reads it.

        Smaller, it is enlarged (_bspline). Larger, it is averaged down: each
        halving buffer samples the one above at its texels' shared corners,
        so linear filtering averages them two by two (a mipmap).
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
        own = sizes[k]
        made = node_shader(node, name, own)
        if not isinstance(made, Sub):
            shader_node, names = made
            made = Sub(
                [shader_node], [], [(n, name, i) for i, n in enumerate(names)], name
            )
        if own != size:
            made.nodes.append(buffer(f"{name}_size", own))
            made.wires.append(wire(made.output, f"{name}_size"))
            made.output = f"{name}_size"
        mm += made.nodes
        if origins is not None:
            origins.update((n["name"], k) for n in made.nodes)
        wires += made.wires
        wires += [
            wire(fed(node.inputs[inp], own), to, port) for inp, to, port in made.ports
        ]
        out_of[k] = made.output
    mm, wires = merge_buffers(mm, wires)
    mm, wires = cache_shared(mm, wires, size, origins)
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
    return "rgba"


def declared_params(mm: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the parameters shader nodes declare, inside subgraphs too."""
    out: dict[str, dict[str, Any]] = {}
    for node in mm:
        if node["type"] == "shader":
            for prm in node["shader_model"]["parameters"]:
                out.setdefault(prm["name"], prm)
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
    widgets, values = [], {}
    for k, (pname, users) in enumerate(sorted(exposed.items())):
        prm = declared[pname]
        widgets.append(
            {
                "name": f"param{k}",
                "label": prm["label"],
                "type": "linked_control",
                "linked_widgets": [{"node": u, "widget": pname} for u in users],
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
