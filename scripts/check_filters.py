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
from sbsar2ptex.probe import GRAY
from sbsar2ptex.probe import cook
from sbsar2ptex.probe import sbs
from sbsar2ptex.sbsasm import Asm
from sbsar2ptex.translate import UnsupportedError
from sbsar2ptex.translate import shader
from sbsar2ptex.translate import translate_graph

SBSRENDER = Path("C:/Program Files/Adobe/Adobe Substance 3D Designer/sbsrender.exe")
SIZE = 11
"""log2 pixels; Material Maker's export is 2048 px."""
F4 = "constantValueFloat4"


def cases() -> list[tuple[str, str, int, tuple[tuple[str, str, str], ...]]]:
    """Return (tag, filter, image type, parameters) per case."""
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
        Image.fromarray((v.clip(0, 1) * 65535).round().astype(np.uint16)).save(paths[k])
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

    todo = [c for c in cases() if not args.tags or c[0] in args.tags]
    made: list[tuple[str, int]] = []
    for tag, filt, kind, params in todo:
        src = out / f"{tag}.sbs"
        src.write_text(sbs(filt, kind, params=params), encoding="utf-8")
        if cook(COOKER, src, out) is None:
            print(f"{tag:14} cook failed")
            continue
        a, b = ("A", "B") if kind == COLOR else ("a", "b")
        subprocess.run(
            [
                str(SBSRENDER), "render", "--input", str(out / f"{tag}.sbsar"),
                "--set-entry", f"input_a@{images[a]}", "--set-entry", f"input_b@{images[b]}",
                "--set-entry", f"input_c@{images['c']}",
                "--set-value", f"$outputsize@{SIZE},{SIZE}", "--engine", "sse2",
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
                asm.nodes(), {"output": asm.output_nodes[0][1]}, by_uid
            )
        except UnsupportedError as e:
            print(f"{tag:14} unsupported: {e}")
            continue
        graph = material_graph(mm_nodes, wires, outs["output"], kind == COLOR)
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
    for tag, kind in made:
        got_path = mm_out / f"{tag}_albedo.png"
        if not got_path.exists():
            print(f"{tag:14} Material Maker made no image")
            continue
        ref, got = load(out / f"ref_{tag}.png"), load(got_path)
        if kind == GRAY:
            ref = ref[..., 0] if ref.ndim == 3 else ref
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
