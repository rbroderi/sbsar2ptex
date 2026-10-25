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
from sbsar2ptex.probe import get
from sbsar2ptex.probe import gradient_array
from sbsar2ptex.probe import input_kinds
from sbsar2ptex.probe import sbs
from sbsar2ptex.sbsasm import Asm
from sbsar2ptex.translate import UnsupportedError
from sbsar2ptex.translate import graph_inputs
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
    out = []
    for tag, filt, kind, params, values in (
        ("fn_levels", "levels", GRAY, levels, {"pa": 0.4}),
        ("fn_blend", "blend", GRAY, opacity, {"pa": 1.2}),
        ("fn_transform", "transformation", GRAY, offset, {"pa": 0.1, "pb": 0.9}),
        ("fn_hsl", "hsl", COLOR, hue, {"pa": 0.7}),
    ):
        base = {"filter": filt, "kind": kind, "params": params, "floats": floats}
        out += [{"tag": tag, **base}, {"tag": f"{tag}_set", **base, "set": values}]
    out.append(
        {
            "tag": "fn_blur",
            "filter": "blur",
            "kind": GRAY,
            "params": blur,
            "floats": floats,
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
        try:
            mm_nodes, wires, outs = translate_graph(
                asm.nodes(),
                {"output": asm.output_nodes[0][1]},
                by_uid,
                inputs=graph_inputs(asm, xml),
            )
        except UnsupportedError as e:
            print(f"{tag:14} unsupported: {e}")
            continue
        for k, v in c["set"].items():  # the same input values in Material Maker
            for n in mm_nodes:
                if n["type"] == "shader" and f"p_{k}" in n["parameters"]:
                    n["parameters"][f"p_{k}"] = float(v)
            for n in mm_nodes:
                if n["type"] == "remote":
                    for w in n["widgets"]:
                        if w["linked_widgets"][0]["widget"] == f"p_{k}":
                            n["parameters"][w["name"]] = float(v)
        source = next(n for n in mm_nodes if n["name"] == outs["output"][0])
        color = source["type"] == "image" or (
            source["shader_model"]["outputs"][0]["type"] == "rgba"
        )
        graph = material_graph(mm_nodes, wires, outs["output"], color)
        (out / f"{tag}.ptex").write_text(json.dumps(graph, indent=1), encoding="utf-8")
        made.append((tag, kind))

    mm_out = out / "mm"
    mm_out.mkdir(exist_ok=True)
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
        diff = np.abs(ref - got[..., : ref.shape[-1]] if ref.ndim == 3 else ref - got)
        p99 = float(np.percentile(diff, 99)) * 255
        mean = float(diff.mean()) * 255
        worst = max(worst, p99)
        verdict = "ok" if p99 <= 4 and mean <= 1.5 else "OFF"
        print(
            f"{tag:14} mean {mean:6.2f}/255  p99 {p99:6.2f}/255  max {diff.max() * 255:6.1f}  {verdict}"
        )
    print(f"worst p99 {worst:.2f}/255")
    sys.exit(0)


if __name__ == "__main__":
    main()
