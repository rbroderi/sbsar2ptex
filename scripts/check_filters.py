"""Check translated filters against Substance, pixel by pixel.

    uv run python scripts/check_filters.py --mm PATH/TO/material_maker.exe [-o DIR] [TAG ...]

For each case (a filter, image type and parameters): cook a one-node graph
with sbscooker, render it with sbsrender on test images, decompile and
translate it to a Material Maker graph fed the same images, export that with
Material Maker, and compare. Needs Substance 3D Designer (sbscooker,
sbsrender) and Material Maker 1.x.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from sbsar2ptex.archive import member
from sbsar2ptex.archive import read_archive
from sbsar2ptex.probe import COLOR
from sbsar2ptex.probe import COOKER
from sbsar2ptex.probe import FLOAT2
from sbsar2ptex.probe import FLOAT3
from sbsar2ptex.probe import FLOAT4
from sbsar2ptex.probe import GRAY
from sbsar2ptex.probe import const
from sbsar2ptex.probe import cook
from sbsar2ptex.probe import curve_array
from sbsar2ptex.probe import fn
from sbsar2ptex.probe import function_graph
from sbsar2ptex.probe import fx
from sbsar2ptex.probe import fxmap_graph
from sbsar2ptex.probe import get
from sbsar2ptex.probe import gradient_array
from sbsar2ptex.probe import input_kinds
from sbsar2ptex.probe import sbs
from sbsar2ptex.sbsasm import Asm
from sbsar2ptex.translate import UnsupportedError
from sbsar2ptex.translate import graph_inputs
from sbsar2ptex.translate import output_type
from sbsar2ptex.translate import param_name
from sbsar2ptex.translate import shader
from sbsar2ptex.translate import translate_graph

SBSRENDER = Path("C:/Program Files/Adobe/Adobe Substance 3D Designer/sbsrender.exe")
SIZE = 11
"""log2 pixels; Material Maker's export is 2048 px."""
F4 = "constantValueFloat4"
F1 = "constantValueFloat1"


def cases() -> list[tuple]:
    """Return (tag, filter, image type, parameters[, arrays]) per case."""
    out = []
    for kind, k in ((GRAY, "g"), (COLOR, "c")):
        out += [
            (
                f"uniform_{k}",
                "uniform",
                kind,
                (("outputcolor", F4, "0.3 0.4 0.5 0.8"),),
            ),
            (f"uniform_default_{k}", "uniform", kind, ()),
            (
                f"levels_{k}",
                "levels",
                kind,
                (
                    ("levelinlow", F4, "0.1 0.15 0.2 0.05"),
                    ("levelinhigh", F4, "0.9 0.8 0.85 0.95"),
                    ("levelinmid", F4, "0.35 0.6 0.5 0.45"),
                    ("leveloutlow", F4, "0.05 0.1 0.0 0.0"),
                    ("levelouthigh", F4, "0.95 0.9 1.0 1.0"),
                ),
            ),
            *(
                (
                    f"levels_step{i}_{k}",
                    "levels",
                    kind,
                    (("levelinlow", F4, edges), ("levelinhigh", F4, edges)),
                )
                for i, edges in enumerate(("0.5 0.5 0.5 0.5", "0 0.25 0.75 1"))
            ),
            # A computed 0.5 (half of min(2 v, 1), from v = 0.5 up) clears a
            # 0.5 threshold: stored in 8 bits it rounds to 128 / 255 (on
            # Direct3D and Vulkan here; a GPU may break the tie down).
            {
                "tag": f"levels_step_stored_{k}",
                "filter": "levels",
                "kind": kind,
                "params": (
                    ("levelinlow", F4, "0.5 0.5 0.5 0.5"),
                    ("levelinhigh", F4, "0.5 0.5 0.5 0.5"),
                ),
                "chain": (
                    "levels",
                    "".join(
                        f'<parameter><name v="{name}"/><relativeTo v="0"/>'
                        f'<paramValue><constantValueFloat4 v="{v} {v} {v} {v}"/>'
                        "</paramValue></parameter>"
                        for name, v in (("levelinhigh", 0.5), ("levelouthigh", 0.5))
                    ),
                ),
            },
            (
                f"transform_{k}",
                "transformation",
                kind,
                (
                    ("matrix22", F4, "0.9 0.2 -0.3 1.1"),
                    ("offset", "constantValueFloat2", "0.25 0.1"),
                ),
            ),
        ]
        out += [
            (
                f"transform_tile{tiling}_{k}",
                "transformation",
                kind,
                (
                    ("matrix22", F4, "1.6 0.2 -0.3 1.4"),
                    ("offset", "constantValueFloat2", "0.15 0.1"),
                    ("tiling", "constantValueInt32", str(tiling)),
                ),
            )
            for tiling in range(3)
        ]
        out += [
            (
                f"blend{m}_{k}",
                "blend",
                kind,
                (
                    ("blendingmode", "constantValueInt32", str(m)),
                    ("opacitymult", "constantValueFloat1", "0.8"),
                ),
            )
            for m in range(12)
        ]
    keys = gradient_array(
        [
            (0.0, (0.1, 0.2, 0.3, 0.4)),
            (0.3, (0.5, 0.6, 0.7, 0.8)),
            (1.0, (0.9, 0.8, 0.7, 0.6)),
        ]
    )
    gray = (("colorswitch", "constantValueBool", "0"),)
    curve = [
        ((0, 0.1), (0, 0.1), (0.2, 0.15)),
        ((0.5, 0.7), (0.35, 0.6), (0.65, 0.8)),
        ((1, 0.9), (0.85, 0.95), (1, 0.9)),
    ]
    red = [((0, 0), (0, 0), (0.3, 0.0)), ((1, 1), (0.7, 1.0), (1, 1))]
    i32 = "constantValueInt32"
    out += [
        ("gradient_c", "gradient", GRAY, (), keys),
        ("gradient_g", "gradient", GRAY, gray, keys),
        ("hsl_c", "hsl", COLOR, (
            ("hue", "constantValueFloat1", "0.62"),
            ("saturation", "constantValueFloat1", "0.4"),
            ("luminosity", "constantValueFloat1", "0.45"),
        )),
        ("grayconv_c", "grayscaleconversion", COLOR, (
            ("channelsweights", F4, "0.2 0.3 0.4 0.1"),
        )),
        ("grayconv_am_c", "grayscaleconversion", COLOR, (
            ("channelsweights", F4, "0.2 0.3 0.4 0.1"),
            ("alphamult", "constantValueBool", "1"),
            ("mattelevel", "constantValueFloat1", "0.35"),
        )),
        ("shuffle_c", "shuffle", COLOR, (
            ("channelred", i32, "6"), ("channelgreen", i32, "0"),
            ("channelblue", i32, "3"), ("channelalpha", i32, "5"),
        )),
        ("curve_g", "curve", GRAY, (), curve_array(curve)),
        ("curve_c", "curve", COLOR, (), curve_array(curve) + curve_array(red, name="curvered")),
        ("dyngrad_g", "dyngradient", GRAY, (("coordinate", "constantValueFloat1", "0.3"),)),
        ("dyngrad_c", "dyngradient", COLOR, (("coordinate", "constantValueFloat1", "0.3"),)),
        ("dyngrad_uv_c", "dyngradient", COLOR, (
            ("coordinate", "constantValueFloat1", "0.3"), ("uvselector", i32, "1"),
        )),
    ]  # fmt: skip
    hi = ("highlightcolor", F4, "0.9 0.7 0.5 1")
    lo = ("shadowcolor", F4, "0.2 0.3 0.1 1")
    for kind, k in ((GRAY, "g"), (COLOR, "c")):
        out += [
            (
                f"blur_{k}",
                "blur",
                kind,
                (("intensity", F1, "1.7" if k == "c" else "3"),),
                "",
                True,
            ),
            (
                f"dmblur_{k}",
                "dirmotionblur",
                kind,
                (("intensity", F1, "4"), ("mblurangle", F1, "0.1")),
                "",
                True,
            ),
            (f"sharpen_{k}", "sharpen", kind, (("intensity", F1, "0.7"),), "", True),
            (f"warp_{k}", "warp", kind, (("intensity", F1, "2"),), "", True),
            (
                f"dwarp_{k}",
                "directionalwarp",
                kind,
                (("intensity", F1, "4"), ("warpangle", F1, "0.1")),
                "",
                True,
            ),
            (
                f"emboss_{k}",
                "emboss",
                kind,
                (("intensity", F1, "4"), ("lightangle", F1, "0.1"), hi, lo),
                "",
                True,
            ),
        ]
    out += [
        ("normal_g", "normal", GRAY, (("intensity", F1, "3"),), "", True),
        (
            "normal_inv_g",
            "normal",
            GRAY,
            (("intensity", F1, "3"), ("inversedy", "constantValueBool", "1")),
            "",
            True,
        ),
    ]
    return out + function_cases()


def function_cases() -> list[dict]:
    """Return cases whose parameters are driven by functions of graph inputs.

    Each runs at the inputs' defaults and with other values (``set``),
    which the translated graph takes through its exposed parameters.
    """
    pa, pb = get("pa"), get("pb")

    def vec4(x: tuple) -> tuple:
        v2 = fn("vector2", FLOAT2, componentsin=x, componentslast=x)
        v3 = fn("vector3", FLOAT3, componentsin=v2, componentslast=x)
        return fn("vector4", FLOAT4, componentsin=v3, componentslast=const(1.0))

    def dyn(tree: tuple) -> str:
        return function_graph(tree)

    floats = {"pa": 0.25, "pb": 0.5}
    levels = (("levelinlow", "dynamicValue", dyn(vec4(fn("mul", a=pa, b=const(0.8))))),)
    opacity = (
        ("opacitymult", "dynamicValue",
         dyn(fn("add", a=fn("mul", a=fn("sin", a=pa), b=const(0.4)), b=const(0.5)))),
    )  # fmt: skip
    offset = (
        ("offset", "dynamicValue",
         dyn(fn("vector2", FLOAT2, componentsin=pa, componentslast=fn("mul", a=pb, b=const(0.3))))),
    )  # fmt: skip
    hue = (("hue", "dynamicValue", dyn(pa)),)
    blur = (("intensity", "dynamicValue", dyn(fn("mul", a=pa, b=const(8.0)))),)
    # distinct components, so a gray uniform shows which one it takes
    rg = fn(
        "vector2", FLOAT2, componentsin=pa, componentslast=fn("mul", a=pb, b=const(0.6))
    )
    rgb = fn("vector3", FLOAT3, componentsin=rg, componentslast=const(0.2))
    color = (
        ("outputcolor", "dynamicValue",
         dyn(fn("vector4", FLOAT4, componentsin=rgb, componentslast=const(0.9)))),
    )  # fmt: skip
    out = []
    for tag, filt, kind, params, values in (
        ("fn_uniform", "uniform", GRAY, color, {"pa": 0.8}),
        ("fn_uniform_c", "uniform", COLOR, color, {"pa": 0.8, "pb": 0.1}),
        ("fn_levels", "levels", GRAY, levels, {"pa": 0.4}),
        ("fn_blend", "blend", GRAY, opacity, {"pa": 1.2}),
        ("fn_transform", "transformation", GRAY, offset, {"pa": 0.1, "pb": 0.9}),
        ("fn_hsl", "hsl", COLOR, hue, {"pa": 0.7}),
    ):
        base = {"filter": filt, "kind": kind, "params": params, "floats": floats}
        out += [{"tag": tag, **base}, {"tag": f"{tag}_set", **base, "set": values}]
    # No opacity input: the opacity's own state (inline, a function).
    out += [
        {"tag": "blend_inline_g", "filter": "blend", "kind": GRAY, "params": None,
         "floats": floats, "connect": 2},
        {"tag": "fn_blend_inline", "filter": "blend", "kind": GRAY,
         "params": opacity, "floats": floats, "connect": 2, "set": {"pa": 1.2}},
        # Add sub at half opacity is (a + b) / 255 - 0.5: every value a tie
        # between two 8-bit steps; exact where a + b = 255 (0.5, stored 128).
        {"tag": "blend_addsub_half_g", "filter": "blend", "kind": GRAY,
         "connect": 2, "params": (("blendingmode", "constantValueInt32", "4"),
                                  ("opacitymult", "constantValueFloat1", "0.5"))},
    ]  # fmt: skip
    out.append(
        {
            "tag": "fn_blur",
            "filter": "blur",
            "kind": GRAY,
            "params": blur,
            "floats": floats,
        }
    )
    return out + pixel_cases() + distance_cases() + fxmap_cases()


def fxmap_cases() -> list[dict]:
    """Return FX-Map cases: trees of Quadrants, Iterates and Switches."""
    i1, f1, f2 = "constantValueInt32", "constantValueFloat1", "constantValueFloat2"

    def pat(t: int, *more: tuple, children: dict | None = None) -> tuple:
        return fx("quadrant", (("patterntype", i1, str(t)), *more), children)

    def dyn(name: str, tree: tuple) -> tuple:
        return (name, "dynamicValue", function_graph(tree))

    def quad4(types: tuple, sup: str = "0") -> tuple:
        return fx(
            "quadrant",
            (),
            {
                k: pat(t, ("patternsize", f2, "0.9 0.8"), ("patternsuppl", f1, sup))
                for k, t in enumerate(types)
            },
        )

    flat = (("hurst", f1, "0"),)
    number = get("$number")

    def vec2(x: tuple, y: tuple) -> tuple:
        return fn("vector2", FLOAT2, componentsin=x, componentslast=y)

    def lin(a: float, b: float) -> tuple:
        return fn("add", a=fn("mul", a=number, b=const(a)), b=const(b))

    leaf = pat(3, ("patternsize", f2, "0.8 0.6"), ("blendingmode", i1, "1"))
    chain = fx("quadrant", (), dict.fromkeys(range(4), leaf))
    for _ in range(2):
        shared = chain
        chain = fx("quadrant", (), dict.fromkeys(range(4), shared))
    swept = pat(
        4,
        ("patternsize", f2, "0.3 0.2"),
        dyn("frameoffset", vec2(lin(0.17, -0.3), lin(0.07, 0.1))),
        dyn("patternrotation", lin(0.1, 0.0)),
    )
    switch = fx(
        "switch",
        (("switch", "constantValueBool", "1"),),
        {0: pat(2, ("patternsize", f2, "0.5 0.5")), 1: pat(6)},
    )
    cases = [
        ("fx_shape_g", GRAY, pat(3, ("patternsize", f2, "0.5 0.3"),
                                 ("frameoffset", f2, "0.1 0.2"),
                                 ("patternrotation", f1, "0.1")), ()),
        ("fx_pats1_g", GRAY, quad4((2, 3, 4, 5)), flat),
        ("fx_pats2_g", GRAY, quad4((6, 7, 8, 9), "0.3"), flat),
        ("fx_pats3_g", GRAY, quad4((10, 12, 15, 16), "0.4"), flat),
        ("fx_pats4_g", GRAY, quad4((13, 14, 17, 18), "0.3"), flat),
        ("fx_wrap_g", GRAY, pat(8, ("patternsize", f2, "1.6 0.7"),
                                ("frameoffset", f2, "0.4 0.1")), ()),
        ("fx_grid_g", GRAY, chain, ()),
        ("fx_iter_g", GRAY, fx("iterate", (("numberadded", i1, "6"),),
                               {0: pat(5), 1: swept}), flat),
        ("fx_switch_g", GRAY, switch, ()),
        ("fx_hurst_g", GRAY, pat(2, ("patternsize", f2, "0.6 0.6"),
                                 children={1: quad4((3, 4, 5, 6))}),
         (("hurst", f1, "0.6"), ("opacitymult", f1, "0.8"))),
        ("fx_color_c", COLOR, fx("iterate", (("numberadded", i1, "1"),), {
            0: pat(2, ("opacity", F4, "0.2 0.4 0.6 0.5")),
            1: pat(3, ("opacity", F4, "0.9 0.3 0.1 0.7"), ("blendingmode", i1, "1"),
                   ("patternsize", f2, "0.7 0.7"))}), flat),
    ]  # fmt: skip

    # rand: each function's own generator, from its node's state.
    def rnd(a: float) -> tuple:
        return fn("rand", a=const(a))

    gray4 = fn(
        "vector4",
        FLOAT4,
        componentsin=fn(
            "vector3", FLOAT3, componentsin=vec2(rnd(1.0), rnd(1.0)),
            componentslast=rnd(1.0),
        ),
        componentslast=const(1.0),
    )  # fmt: skip
    dot = pat(
        3,
        ("randomseed", i1, "5"),
        dyn("patternsize", vec2(rnd(0.6), rnd(0.6))),
        dyn("frameoffset", vec2(fn("sub", a=rnd(0.5), b=const(0.25)), rnd(0.2))),
        dyn("opacity", gray4),
        ("blendingmode", i1, "1"),
    )
    grid = fx("quadrant", (("randomseed", i1, "3"),), dict.fromkeys(range(4), dot))
    grid = fx("quadrant", (), dict.fromkeys(range(4), grid))
    splat = pat(
        6,
        ("patternsize", f2, "0.2 0.2"),
        dyn("frameoffset", vec2(rnd(1.0), rnd(1.0))),
        dyn("patternrotation", rnd(1.0)),
    )
    splats = fx(
        "iterate",
        (("numberadded", i1, "40"), ("randomseed", i1, "3")),
        {1: splat},
    )
    out = [
        {"tag": tag, "filter": "fxmaps", "kind": kind, "params": params,
         "extra": fxmap_graph(root), "connect": 0, "format16": True}
        for tag, kind, root, params in cases
    ]  # fmt: skip
    # More than translate.LIVE_SPLATS patterns: precomputed (fxsplat).
    jitter = fn("sub", a=rnd(0.3), b=const(0.15))
    leaf = pat(
        5,
        dyn("branchoffset", vec2(jitter, fn("sub", a=rnd(0.3), b=const(0.15)))),
        dyn("patternsize", vec2(fn("add", a=rnd(1.5), b=const(0.5)), rnd(2.0))),
        dyn("patternrotation", rnd(1.0)),
    )
    scatter = leaf
    for _ in range(4):
        scatter = fx("quadrant", (), dict.fromkeys(range(4), scatter))
    many = fx(
        "iterate",
        (("numberadded", i1, "300"),),
        {1: pat(3, ("patternsize", f2, "0.05 0.03"), ("blendingmode", i1, "1"),
                dyn("frameoffset", vec2(rnd(1.0), rnd(1.0))),
                dyn("patternrotation", rnd(1.0)))},
    )  # fmt: skip
    rand_trees = (
        ("fx_rand_grid_g", grid),
        ("fx_rand_iter_g", splats),
        ("fx_scatter_g", scatter),
        ("fx_many_g", many),
    )
    out += [
        {"tag": f"{tag}{seed}", "filter": "fxmaps", "kind": GRAY,
         "params": (), "extra": fxmap_graph(root), "connect": 0,
         "format16": True, "set": {"$randomseed": seed}}
        for tag, root in rand_trees
        for seed in (0, 7)
    ]  # fmt: skip
    # A render region, x0 x1 y0 y1 of pattern space, fills the image (and
    # wraps, as its tiling region): Biomech's grid FX-Maps draw -3 to 4.
    roi = "-0.5 1.5 -0.25 1.75"
    out.append(
        {"tag": "fx_roi_g", "filter": "fxmaps", "kind": GRAY,
         "params": (("renderroi", F4, roi), ("tilingroi", F4, roi)),
         "extra": fxmap_graph(many), "connect": 0, "format16": True}
    )  # fmt: skip
    # Without a tiling region, the default tile wraps; a render region in it
    # is a window onto it, stretched over the image (make_it_tile's).
    out.append(
        {"tag": "fx_window_g", "filter": "fxmaps", "kind": GRAY,
         "params": (("renderroi", F4, "0.2 0.7 0.3 0.8"),),
         "extra": fxmap_graph(many), "connect": 0, "format16": True}
    )  # fmt: skip
    # Functions sharing a variable and sampling the input: recorded.
    x = get("x")
    step = fn(
        "set",
        data='<constantValueString v="x"/>',
        value=fn("mul", a=number, b=const(0.04)),
    )
    # (a get is a variable only where its function also sets it)
    same = fn("set", data='<constantValueString v="x"/>', value=x)
    lum = fn(
        "samplelum", pos=vec2(same, const(0.5)), data='<constantValueInt2 v="0 0"/>'
    )
    walk = pat(
        3,
        dyn("frameoffset", vec2(fn("sub", a=step, b=const(0.5)), const(0.0))),
        dyn(
            "patternsize",
            vec2(
                fn("add", a=fn("mul", a=lum, b=const(0.05)), b=const(0.01)), const(0.1)
            ),
        ),
    )
    state = fx("iterate", (("numberadded", i1, "24"),), {1: walk})
    out.append(
        {"tag": "fx_state_g", "filter": "fxmaps", "kind": GRAY, "params": flat,
         "extra": fxmap_graph(state), "connect": 1, "format16": True}
    )  # fmt: skip
    # Patterns larger than the map, many: baked (fxsplat.bake).
    blob = pat(
        6,  # gaussian
        ("patternsize", f2, "1.5 1.2"),
        ("patternrotation", f1, "0.1"),
        dyn("frameoffset", vec2(rnd(1.0), rnd(1.0))),
    )
    blobs = fx("iterate", (("numberadded", i1, "300"),), {1: blob})
    faint = (*flat, ("opacitymult", f1, "0.01"))
    out.append(
        {"tag": "fx_bake_g", "filter": "fxmaps", "kind": GRAY, "params": faint,
         "extra": fxmap_graph(blobs), "connect": 0, "format16": True}
    )  # fmt: skip
    return out


def distance_cases() -> list[dict]:
    """Return distance cases: mask A (> 0.5) and source B, or the mask alone."""
    reach = (("distance", F1, "12"),)
    return [
        {"tag": "distance_g", "filter": "distance", "kind": GRAY, "params": reach},
        {"tag": "distance_mask_g", "filter": "distance", "kind": GRAY, "params": reach,
         "connect": 1},
        {"tag": "distance_c", "filter": "distance", "kind": COLOR, "params": reach},
    ]  # fmt: skip


def pixel_cases() -> list[dict]:
    """Return pixel processor cases: per-pixel functions that sample inputs."""
    pos = get("$pos", FLOAT2)
    pa = get("pa")

    def at(dx: float, dy: float) -> tuple:
        shift = fn("const_float2", FLOAT2, data=f'<constantValueFloat2 v="{dx} {dy}"/>')
        return fn("add", FLOAT2, a=pos, b=shift)

    def lum(where: tuple, index: int, filtering: int = 0) -> tuple:
        data = f'<constantValueInt2 v="{index} {filtering}"/>'
        return fn("samplelum", pos=where, data=data)

    def col(where: tuple, index: int) -> tuple:
        data = f'<constantValueInt2 v="{index} 0"/>'
        return fn("samplecol", FLOAT4, pos=where, data=data)

    def per(tree: tuple) -> tuple:
        return (("perpixel", "dynamicValue", function_graph(tree)),)

    gray = fn(
        "add",
        a=fn("mul", a=lum(at(0.013, -0.021), 0), b=pa),
        b=fn("mul", a=lum(pos, 1), b=fn("sub", a=const(1.0), b=pa)),
    )
    color = fn("lerp", FLOAT4, a=col(pos, 0), b=col(at(0.05, 0.1), 1), x=pa)
    scale = fn("const_float2", FLOAT2, data='<constantValueFloat2 v="0.37 0.37"/>')
    nearest = lum(fn("mul", FLOAT2, a=pos, b=scale), 0, 1)
    l4 = lum(pos, 0)
    v2 = fn("vector2", FLOAT2, componentsin=l4, componentslast=l4)
    v3 = fn("vector3", FLOAT3, componentsin=v2, componentslast=l4)
    lum_of_color = fn("vector4", FLOAT4, componentsin=v3, componentslast=const(1.0))
    rand_mix = fn(
        "add",
        a=fn("mul", a=fn("rand", a=const(0.5)), b=lum(pos, 0)),
        b=fn("rand", a=fn("add", a=pa, b=fn("rand", a=const(0.2)))),
    )
    floats = {"pa": 0.3}
    out = []
    for tag, kind, tree in (
        ("pp_gray", GRAY, gray),
        ("pp_color", COLOR, color),
        ("pp_nearest", GRAY, nearest),
        ("pp_lum_color", COLOR, lum_of_color),
        ("pp_rand", GRAY, fn("rand", a=const(1.0))),
        ("pp_rand_mix", GRAY, rand_mix),
    ):
        out.append(
            {
                "tag": tag,
                "filter": "pixelprocessor",
                "kind": kind,
                "params": per(tree),
                "floats": floats,
            }
        )
    out.append(
        {
            "tag": "pp_rand_seed",
            "filter": "pixelprocessor",
            "kind": GRAY,
            "params": per(fn("rand", a=const(1.0))),
            "floats": floats,
            "set": {"$randomseed": 3},
        }
    )
    # A parameter function sets a variable, once per node with its value
    # generator; the per-pixel function reads it. Biomech hides one in an
    # output size relative to the input's, adding (0, 0); Substance renders
    # that at twice the size, so the image is flat, the variable alone.
    v = '<constantValueString v="v"/>'
    same = fn("const_int2", 32, data='<constantValueInt2 v="0 0"/>')
    noise = fn("vector2", FLOAT2, componentsin=fn("rand", a=const(1.0)),
               componentslast=fn("rand", a=const(1.0)))  # fmt: skip
    sets = fn("sequence", 32, seqin=fn("set", FLOAT2, data=v, value=noise), seqout=same)
    weights = fn("const_float2", FLOAT2, data='<constantValueFloat2 v="0.6 0.3"/>')
    shade = fn("dot", a=get("v", FLOAT2), b=weights)
    for tag, seed in (("pp_param_var", {}), ("pp_param_var_seed", {"$randomseed": 3})):
        out.append(
            {
                "tag": tag,
                "filter": "pixelprocessor",
                "kind": GRAY,
                "params": (
                    *per(shade),
                    ("outputsize", "dynamicValue", function_graph(sets), 1),
                ),
                "floats": floats,
                **({"set": seed} if seed else {}),
            }
        )
    return out


def test_images(out: Path) -> dict[str, Path]:
    """Write gray a, b, c and color A, B (alpha varying) test images."""
    n = 1 << SIZE
    y, x = np.mgrid[0:n, 0:n] / n
    gray = {
        "a": 0.5
        + 0.25 * np.sin(2 * np.pi * 3 * x)
        + 0.25 * y * np.cos(2 * np.pi * 2 * y),
        "b": x * 0.7 + 0.3 * (0.5 + 0.5 * np.sin(2 * np.pi * 5 * (x + y))),
        "c": 0.5 + 0.5 * np.cos(2 * np.pi * 2 * x) * np.sin(2 * np.pi * 3 * y),
    }
    color = {
        "A": np.stack(
            [x, y, 0.5 + 0.5 * np.sin(2 * np.pi * 4 * x * y), 0.3 + 0.7 * y], -1
        ),
        "B": np.stack(
            [1 - y, 0.5 + 0.5 * np.sin(2 * np.pi * 3 * x), x * y, 0.2 + 0.8 * x], -1
        ),
    }
    paths = {}
    for k, v in gray.items():
        paths[k] = out / f"in_gray_{k}.png"
        # 8-bit: Material Maker reads images at 8 bits, so Substance must too.
        Image.fromarray((v.clip(0, 1) * 255).round().astype(np.uint8), "L").save(
            paths[k]
        )
    for k, v in color.items():
        paths[k] = out / f"in_color_{k}.png"
        Image.fromarray((v.clip(0, 1) * 255).round().astype(np.uint8), "RGBA").save(
            paths[k]
        )
    return paths


def load(p: Path) -> np.ndarray:
    """Return an image as floats in [0, 1]."""
    im = Image.open(p)
    a = np.asarray(im).astype(np.float64)
    return a / (65535 if im.mode.startswith("I") or a.max() > 255 else 255)


def material_graph(mm_nodes, wires, source: tuple[str, int], color: bool) -> dict:
    """Return a graph: ``source`` into a Material's albedo (and alpha into opacity)."""
    material = {
        "name": "Material",
        "type": "material",
        "node_position": {"x": 3000, "y": 0},
        "export_paths": {},
        "seed_int": 0,
        "parameters": {
            "albedo_color": {"type": "Color", "r": 1, "g": 1, "b": 1, "a": 1},
            "metallic": 1,
            "roughness": 1,
            "emission_energy": 1,
            "normal": 1,
            "ao": 1,
            "depth_scale": 0.5,
            "flags_transparent": color,
            "sss": 0,
            "size": SIZE,
        },
    }
    nodes = [*mm_nodes, material]
    wires = [
        *wires,
        {"from": source[0], "from_port": source[1], "to": "Material", "to_port": 0},
    ]
    if color:
        nodes.append(shader("alpha", [("input", "rgba")], "f", "return $input(uv).a;"))
        wires += [
            {"from": source[0], "from_port": source[1], "to": "alpha", "to_port": 0},
            {"from": "alpha", "from_port": 0, "to": "Material", "to_port": 7},
        ]
    return {
        "name": "check",
        "type": "graph",
        "label": "Graph",
        "shortdesc": "",
        "longdesc": "",
        "node_position": {"x": 0, "y": 0},
        "parameters": {},
        "seed_int": 0,
        "nodes": nodes,
        "connections": wires,
    }


def case(c: tuple | dict) -> dict:
    """Return a case as a dict.

    Keys: tag, filter, kind, params, arrays, format16, floats (graph float
    inputs and defaults) and set (input values to render with, in both
    Substance and Material Maker).
    """
    if isinstance(c, dict):
        return {"arrays": "", "format16": False, "floats": {}, "set": {}, **c}
    tag, filt, kind, params, *rest = c
    return {
        "tag": tag,
        "filter": filt,
        "kind": kind,
        "params": params,
        "arrays": rest[0] if rest else "",
        "format16": bool(rest[1]) if len(rest) > 1 else False,
        "floats": {},
        "set": {},
    }


def main() -> None:
    """Run the cases the command line names, or all."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("tags", nargs="*")
    parser.add_argument("--mm", type=Path, required=True, help="material_maker.exe")
    parser.add_argument("-o", "--out", type=Path, default=Path("check_out"))
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    images = test_images(out)

    todo = [case(c) for c in cases()]
    todo = [c for c in todo if not args.tags or c["tag"] in args.tags]
    made: list[tuple[str, int]] = []
    for c in todo:
        tag, filt, kind = c["tag"], c["filter"], c["kind"]
        src = out / f"{tag}.sbs"
        src.write_text(
            sbs(
                filt,
                kind,
                params=c["params"],
                arrays=c["arrays"],
                format16=c["format16"],
                floats=c["floats"],
                connect=c.get("connect"),
                extra=c.get("extra", ""),
                chain=c.get("chain", False),
            ),
            encoding="utf-8",
        )
        if cook(COOKER, src, out) is None:
            print(f"{tag:14} cook failed")
            continue
        ka, kb, _ = input_kinds(filt, kind)
        a = "A" if ka == COLOR else "a"
        b = "B" if kb == COLOR else "b"
        subprocess.run(
            [
                str(SBSRENDER), "render", "--input", str(out / f"{tag}.sbsar"),
                "--set-entry", f"input_a@{images[a]}", "--set-entry", f"input_b@{images[b]}",
                "--set-entry", f"input_c@{images['c']}",
                "--set-value", f"$outputsize@{SIZE},{SIZE}",
                *(a for k, v in c["set"].items() for a in ("--set-value", f"{k}@{v}")),
                "--output-path", str(out), "--output-name", f"ref_{tag}",
                "--output-bit-depth", "16", "--no-report",
            ],
            check=True,
            capture_output=True,
        )  # fmt: skip
        files = read_archive(out / f"{tag}.sbsar")
        xml = member(files, ".xml").decode("utf-8")
        names = {
            int(u): n for u, n in re.findall(r'uid="(\d+)" identifier="([^"]*)"', xml)
        }
        asm = Asm(member(files, ".sbsasm"))
        feeds = {"input_a": images[a], "input_b": images[b], "input_c": images["c"]}
        by_uid = {u: feeds[n].as_posix() for u, n in names.items() if n in feeds}
        given = graph_inputs(asm, xml)
        for gi in given.values():  # precomputed FX-Maps take values when made
            if gi.identifier in c["set"]:
                gi.default = [c["set"][gi.identifier]]
        try:
            mm_nodes, wires, outs = translate_graph(
                asm.nodes(),
                {"output": asm.output_nodes[0][1]},
                by_uid,
                inputs=given,
                data_dir=out / "data" / tag,
            )
        except UnsupportedError as e:
            print(f"{tag:14} unsupported: {e}")
            continue
        for k, v in c["set"].items():  # the same input values in Material Maker
            k = param_name(k)  # noqa: PLW2901
            for n in mm_nodes:
                if n["type"] == "shader" and k in n["parameters"]:
                    n["parameters"][k] = float(v)
            for n in mm_nodes:
                if n["type"] == "remote":
                    for w in n["widgets"]:
                        if w["linked_widgets"][0]["widget"] == k:
                            n["parameters"][w["name"]] = float(v)
        source = next(n for n in mm_nodes if n["name"] == outs["output"][0])
        color = output_type(source) == "rgba"
        graph = material_graph(mm_nodes, wires, outs["output"], color)
        (out / f"{tag}.ptex").write_text(json.dumps(graph, indent=1), encoding="utf-8")
        made.append((tag, kind))

    mm_out = out / "mm"
    mm_out.mkdir(exist_ok=True)
    for tag, _ in made:  # an earlier run's image must not pass for this one's
        (mm_out / f"{tag}_albedo.png").unlink(missing_ok=True)
    subprocess.run(
        [str(args.mm), "--export", "-o", mm_out.as_posix(),
         *(f"{(out / t).as_posix()}.ptex" for t, _ in made)],
        cwd=args.mm.parent,
        check=False,
        capture_output=True,
        timeout=600,
    )  # fmt: skip

    worst = 0.0
    for tag, _ in made:
        got_path = mm_out / f"{tag}_albedo.png"
        if not got_path.exists():
            print(f"{tag:14} Material Maker made no image")
            continue
        ref, got = load(out / f"ref_{tag}.png"), load(got_path)
        if ref.ndim == 2:
            got = got[..., 0]
        note = ""
        if ref.shape[:2] != got.shape[:2]:  # a case whose output size Substance moves
            note = f"  (Substance at {ref.shape[1]}x{ref.shape[0]})"
            ref = ref[:: ref.shape[0] // got.shape[0], :: ref.shape[1] // got.shape[1]]
        diff = np.abs(ref - got[..., : ref.shape[-1]] if ref.ndim == 3 else ref - got)
        p99 = float(np.percentile(diff, 99)) * 255
        mean = float(diff.mean()) * 255
        worst = max(worst, p99)
        verdict = "ok" if p99 <= 4 and mean <= 1.5 else "OFF"
        print(
            f"{tag:14} mean {mean:6.2f}/255  p99 {p99:6.2f}/255  max {diff.max() * 255:6.1f}  {verdict}{note}"
        )
    print(f"worst p99 {worst:.2f}/255")
    sys.exit(0)


if __name__ == "__main__":
    main()
