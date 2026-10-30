"""Check the sbs plugin's nodes against the inline translation, through the fork.

    uv run python -I scripts/check_plugin.py --fork PATH/TO/materialmakermod
        [-o DIR] [--godot EXE] [TAG ...]

For each scripts/check_filters.py case: cook its one-node graph with
sbscooker, translate it as today ("inline") and with plugin nodes
(sbsar2ptex.plugin, as --experimental does, "plugin"), and export both
through the material-maker-mod fork (Godot, headless), which loads the
plugin from its material_maker/plugins/sbs (written there first: the
user's own plugins folder is left alone). Two exports a case and mode, on
the same test images: the output's 8-bit albedo (and alpha), the pixels;
and its float32 bits, a channel at a time, as the albedo's four bytes.
The plugin node must be identical to the inline translation in both.
Cases that make no plugin node (pixel processors, FX-Maps, curves,
gradient maps, settings driven by functions) are listed with why. Needs
Substance 3D Designer (sbscooker) and the fork.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))  # check_filters (python -I)

import check_filters

from sbsar2ptex import plugin
from sbsar2ptex.archive import member
from sbsar2ptex.archive import read_archive
from sbsar2ptex.probe import COLOR
from sbsar2ptex.probe import COOKER
from sbsar2ptex.probe import cook
from sbsar2ptex.probe import input_kinds
from sbsar2ptex.probe import sbs
from sbsar2ptex.sbsasm import Asm
from sbsar2ptex.translate import PLUGIN_LINKS
from sbsar2ptex.translate import UnsupportedError
from sbsar2ptex.translate import graph_inputs
from sbsar2ptex.translate import output_type
from sbsar2ptex.translate import param_name
from sbsar2ptex.translate import shader
from sbsar2ptex.translate import strip_links
from sbsar2ptex.translate import translate_graph

GODOT = Path(
    "C:/Users/richa/AppData/Local/Microsoft/WinGet/Packages/"
    "GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe/"
    "Godot_v4.7.2-stable_win64_console.exe"
)
SIZE = check_filters.SIZE
BATCH = 24
"""Graphs a fork run exports."""
MODES = ("inline", "plugin")
LOST = ("DEVICE_LOST", "device lost", "TDR")
"""What the fork's log says when the GPU driver reset (a TDR)."""


def material(*, color: bool) -> dict:
    """Return a Material node of the case's size (alpha used for ``color``)."""
    return {
        "name": "Material", "type": "material", "node_position": {"x": 3000, "y": 0},
        "export_paths": {}, "seed_int": 0,
        "parameters": {"albedo_color": {"type": "Color", "r": 1, "g": 1, "b": 1, "a": 1},
                       "metallic": 1, "roughness": 1, "emission_energy": 1, "normal": 1,
                       "ao": 1, "depth_scale": 0.5, "flags_transparent": color, "sss": 0,
                       "size": SIZE},
    }  # fmt: skip


def graph(nodes: list, wires: list) -> dict:
    """Return the graph of ``nodes``."""
    return {
        "name": "check", "type": "graph", "label": "Graph", "shortdesc": "",
        "longdesc": "", "node_position": {"x": 0, "y": 0}, "parameters": {},
        "seed_int": 0, "nodes": nodes, "connections": wires,
    }  # fmt: skip


def pixels(mm: list, wires: list, source: tuple[str, int], kind: str) -> dict:
    """Return ``source`` into a Material's albedo (and its alpha the opacity)."""
    color = kind == "rgba"
    nodes = [*mm, material(color=color)]
    wires = [*wires, {"from": source[0], "from_port": source[1], "to": "Material",
                      "to_port": 0}]  # fmt: skip
    if color:
        nodes.append(shader("alpha", [("input", "rgba")], "f", "return $input(uv).a;"))
        wires += [
            {"from": source[0], "from_port": source[1], "to": "alpha", "to_port": 0},
            {"from": "alpha", "from_port": 0, "to": "Material", "to_port": 7},
        ]
    return graph(nodes, wires)


def bits(mm: list, wires: list, source: tuple[str, int], kind: str, c: int) -> dict:
    """Return channel ``c`` of ``source`` as its float32's bytes: albedo RGB, alpha."""
    value = f"vec4($input(uv))[{c}]" if kind == "rgba" else "$input(uv)"
    enc = shader(
        "bits", [("input", kind)], "rgba",
        f"uint b = floatBitsToUint({value});\n"
        "return vec4(float(b & 255u), float((b >> 8u) & 255u),"
        " float((b >> 16u) & 255u), float(b >> 24u)) / 255.0 + 0.25 / 255.0;",
    )  # fmt: skip
    alpha = shader("alpha", [("input", "rgba")], "f", "return $input(uv).a;")
    wires = [
        *wires,
        {"from": source[0], "from_port": source[1], "to": "bits", "to_port": 0},
        {"from": "bits", "from_port": 0, "to": "Material", "to_port": 0},
        {"from": "bits", "from_port": 0, "to": "alpha", "to_port": 0},
        {"from": "alpha", "from_port": 0, "to": "Material", "to_port": 7},
    ]
    return graph([*mm, enc, alpha, material(color=True)], wires)


def set_inputs(mm: list, values: dict[str, float]) -> None:
    """Set graph inputs (by identifier) as check_filters does, plugin knobs too."""
    for k, v in values.items():
        p = param_name(k)
        for n in mm:
            if n["type"] == "shader" and p in n["parameters"]:
                n["parameters"][p] = float(v)
            for w in n.get(PLUGIN_LINKS, {}).get(p, {}).get("widgets", []):
                n["parameters"][w] = float(v)
            if n["type"] == "remote":
                for wd in n["widgets"]:
                    if wd["linked_widgets"][0]["widget"] in (p, *_widgets(mm, p)):
                        n["parameters"][wd["name"]] = float(v)


def _widgets(mm: list, p: str) -> list[str]:
    return [
        w for n in mm for w in n.get(PLUGIN_LINKS, {}).get(p, {}).get("widgets", [])
    ]


def translate(c: dict, out: Path, images: dict, maker: plugin.Maker | None) -> tuple:
    """Return the case's translation: MM nodes, connections, output."""
    tag, filt, kind = c["tag"], c["filter"], c["kind"]
    files = read_archive(out / f"{tag}.sbsar")
    xml = member(files, ".xml").decode("utf-8")
    names = {int(u): n for u, n in re.findall(r'uid="(\d+)" identifier="([^"]*)"', xml)}
    asm = Asm(member(files, ".sbsasm"))
    ka, kb, _ = input_kinds(filt, kind)
    feeds = {
        "input_a": images["A" if ka == COLOR else "a"],
        "input_b": images["B" if kb == COLOR else "b"],
        "input_c": images["c"],
    }
    by_uid = {u: feeds[n].as_posix() for u, n in names.items() if n in feeds}
    given = graph_inputs(asm, xml)
    for gi in given.values():
        if gi.identifier in c["set"]:
            gi.default = [c["set"][gi.identifier]]
    mm, wires, outs = translate_graph(
        asm.nodes(), {"output": asm.output_nodes[0][1]}, by_uid, size=SIZE,
        inputs=given, data_dir=out / "data" / tag, plugins=maker,
    )  # fmt: skip
    set_inputs(mm, c["set"])
    return mm, wires, outs["output"]


def export(godot: Path, fork: Path, paths: list[Path], out: Path, log: Path) -> None:
    """Export ``paths`` through the fork, a batch a run, again after a GPU reset."""
    for attempt in range(3):
        todo = [p for p in paths if not (out / f"{p.stem}_albedo.png").exists()]
        if not todo:
            return
        lost = False
        for k in range(0, len(todo), BATCH):
            with log.open("a", encoding="utf-8", errors="replace") as f:
                subprocess.run(
                    [str(godot), "--headless", "--path", str(fork),
                     "--export-material", "--target", "Godot/Godot 4 Standard",
                     "-o", str(out), *(str(p) for p in todo[k : k + BATCH])],
                    stdout=f, stderr=subprocess.STDOUT, check=False, timeout=3600,
                )  # fmt: skip
            text = log.read_text(encoding="utf-8", errors="replace")
            if any(s in text[-20000:] for s in LOST):
                lost = True
                print("GPU reset: waiting 60 s")
                time.sleep(60)
        if not lost and attempt:
            return


def load_bits(path: Path) -> np.ndarray:
    """Return the float32 bits an export of ``bits`` holds."""
    a = np.asarray(Image.open(path).convert("RGBA")).astype(np.uint32)
    return a[..., 0] | a[..., 1] << 8 | a[..., 2] << 16 | a[..., 3] << 24


def main() -> None:
    """Run the cases the command line names, or all."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("tags", nargs="*")
    parser.add_argument("--fork", type=Path, required=True, help="the fork's folder")
    parser.add_argument("--godot", type=Path, default=GODOT)
    parser.add_argument("-o", "--out", type=Path, default=Path("check_plugin_out"))
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    plugin.install(args.fork / "material_maker" / "plugins" / "sbs")
    images = check_filters.test_images(out)
    todo = [check_filters.case(c) for c in check_filters.cases()]
    todo = [c for c in todo if not args.tags or c["tag"] in args.tags]
    made: dict[str, int] = {}
    skipped: dict[str, str] = {}
    for c in todo:
        tag = c["tag"]
        src = out / f"{tag}.sbs"
        src.write_text(
            sbs(c["filter"], c["kind"], params=c["params"], arrays=c["arrays"],
                format16=c["format16"], floats=c["floats"], connect=c.get("connect"),
                extra=c.get("extra", ""), chain=c.get("chain", False)),
            encoding="utf-8",
        )  # fmt: skip
        if cook(COOKER, src, out) is None:
            skipped[tag] = "cook failed"
            continue
        maker = plugin.Maker(SIZE)
        try:
            translated = {
                "inline": translate(c, out, images, None),
                "plugin": translate(c, out, images, maker),
            }
        except UnsupportedError as e:
            skipped[tag] = f"unsupported: {e}"
            continue
        if not maker.made:
            skipped[tag] = "no plugin node: " + ", ".join(maker.inline)
            continue
        channels = 1
        for mode, (mm, wires, source) in translated.items():
            kind = output_type(next(n for n in mm if n["name"] == source[0]))
            channels = 4 if kind == "rgba" else 1
            graphs = {f"{tag}_{mode}": pixels(mm, wires, source, kind)}
            graphs |= {
                f"{tag}_{mode}_b{k}": bits(mm, wires, source, kind, k)
                for k in range(channels)
            }
            for name, g in graphs.items():
                strip_links(g)
                (out / f"{name}.ptex").write_text(json.dumps(g), encoding="utf-8")
        made[tag] = channels
        print(f"{tag:22} {', '.join(f'{t} {n}' for t, n in maker.made.items())}")

    mm_out = out / "mm"
    mm_out.mkdir(exist_ok=True)
    paths = [
        out / f"{tag}_{mode}{suffix}.ptex"
        for tag, channels in made.items()
        for mode in MODES
        for suffix in ("", *(f"_b{k}" for k in range(channels)))
    ]
    for p in paths:
        (mm_out / f"{p.stem}_albedo.png").unlink(missing_ok=True)
    log = out / "fork.log"
    log.write_text("", encoding="utf-8")
    export(args.godot, args.fork, paths, mm_out, log)

    bad = 0
    print(
        f"\n{'case':22} {'pixels':>10} {'float bits':>12}  (pixels differing of {1 << 2 * SIZE})"
    )
    for tag, channels in made.items():
        got = {}
        for mode in MODES:
            files = [mm_out / f"{tag}_{mode}{s}_albedo.png"
                     for s in ("", *(f"_b{k}" for k in range(channels)))]  # fmt: skip
            if not all(f.exists() for f in files):
                got = None
                break
            got[mode] = (
                np.asarray(Image.open(files[0])).astype(np.int64),
                [load_bits(f) for f in files[1:]],
            )
        if got is None:
            print(f"{tag:22} not exported")
            bad += 1
            continue
        (pa, ba), (pb, bb) = got["inline"], got["plugin"]
        dp = int((pa != pb).reshape(pa.shape[0], pa.shape[1], -1).any(-1).sum())
        db = int(sum((x != y).sum() for x, y in zip(ba, bb, strict=True)))
        bad += dp != 0 or db != 0
        print(
            f"{tag:22} {dp:>10} {db:>12}  {'identical' if dp == db == 0 else 'DIFFERENT'}"
        )
    for tag, why in skipped.items():
        print(f"{tag:22} {why}")
    print(
        f"{len(made) - bad} of {len(made)} plugin cases identical; {len(skipped)} without"
    )
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
