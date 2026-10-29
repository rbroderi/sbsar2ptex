"""Check loops made iterate buffers (--experimental) against their unrolled translation.

    uv run python -I scripts/check_loops.py --fork PATH/TO/materialmakermod [-o DIR]
        [--godot EXE] [--size 9] [TAG ...]

For each case (a Slope blur grayscale instance: Samples, Intensity, an 8- or
16-bit graph): cook a graph of it with sbscooker, render it with sbsrender
(16 bits) on test images, and translate it as today (its 63 nodes
unrolled, "norm"), the same with each blend reading the one before through
a 32-bit buffer ("stored"), and with its loop an iterate_buffer_v2
(sbsar2ptex.loops, "exp"). All are exported through the material-maker-mod
fork (Godot, headless), which alone has iterate_buffer_v2, their results
encoded into the albedo as 16-bit levels (high byte red, low byte green),
so that they compare exactly.

The iterate must be identical to "stored" in every pixel. Unrolled, a blend
recomputes the ones before it inline in its shader, where the GPU compiler
may fuse their arithmetic: a value an ulp off its stored one, which a later
half-level tie turns into a 16-bit level ("norm" differs there; the
iterate, like Substance, reads stored values). Substance's render is
compared too. Cooked with Samples under 32, an instance loses the blends
it switches off (no longer the library node): such a case translates the
Samples 32 graph with its live Samples knob set instead ("knob"), against
Substance's render of its own. Needs Substance 3D Designer (sbscooker,
sbsrender) and its cooked library.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from sbsar2ptex import library
from sbsar2ptex.archive import member
from sbsar2ptex.archive import read_archive
from sbsar2ptex.loops import slope_blurs
from sbsar2ptex.probe import BRIDGE_PARAMETERS
from sbsar2ptex.probe import COOKER
from sbsar2ptex.probe import cook
from sbsar2ptex.procedural import library_nodes
from sbsar2ptex.sbsasm import Asm
from sbsar2ptex.translate import buffer
from sbsar2ptex.translate import graph_inputs
from sbsar2ptex.translate import shader
from sbsar2ptex.translate import translate_graph

SBSRENDER = Path("C:/Program Files/Adobe/Adobe Substance 3D Designer/sbsrender.exe")
GODOT = Path(
    "C:/Users/richa/AppData/Local/Microsoft/WinGet/Packages/"
    "GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe/"
    "Godot_v4.7.2-stable_win64_console.exe"
)
DEPENDENCY = 1149349542
"""The uid the graph's slope_blur.sbs dependency goes by."""


def cases() -> list[dict]:
    """Return the cases: tag, Samples, Intensity, 16 bits."""
    return [
        {"tag": f"slope_s{s}_i{i.replace('.', 'p')}_{b}", "samples": s,
         "intensity": i, "format16": b == 16}
        for b in (8, 16)
        for i in ("0.37", "2.5")
        for s in (32, 8, 2, 1)  # 32 first: the others may need its graph
    ]  # fmt: skip


MODES = ("norm", "stored", "exp")
"""The unrolled translation; the same, each blend reading the one before
through a 32-bit buffer (as stored); the iterate."""


def read_stored(
    mm: list[dict], wires: list[dict], size: int
) -> tuple[list[dict], list[dict]]:
    """Return ``mm`` with a 32-bit buffer on each wire between two shaders.

    The unrolled blends read the one before inline, in one shader, which
    the GPU compiler may fuse and so round differently at a tie than the
    blend rendered alone; through a buffer each reads the stored value, as
    Substance's do and as the iterate's state holds it.
    """
    mm, wires = [dict(n) for n in mm], [dict(w) for w in wires]
    kind = {n["name"]: n["type"] for n in mm}
    for k, w in enumerate(wires):
        if kind.get(w["from"]) == kind.get(w["to"]) == "shader":
            name = f"{w['from']}_stored{k}"
            mm.append(buffer(name, size, precise=True))
            wires.append({"from": w["from"], "from_port": w["from_port"],
                          "to": name, "to_port": 0})  # fmt: skip
            wires[k] = {**w, "from": name, "from_port": 0}
    return mm, wires


def set_samples(mm: list[dict], value: float) -> None:
    """Set every Slope blur's Samples knob in ``mm`` (inside subgraphs too)."""
    for n in mm:
        for key in n.get("parameters", {}):
            if key.startswith("i_") and key.endswith("_Samples"):
                n["parameters"][key] = value
        for w in n.get("widgets", []):
            if w.get("type") == "named_parameter" and w["name"].endswith("_Samples"):
                w["default"] = value
        if n["type"] == "graph":
            set_samples(n["nodes"], value)


def sbs(samples: int, intensity: str, *, format16: bool) -> str:
    """Return a package: gray inputs a (Source), b (Effect) -> Slope blur -> output."""

    def paraminput(ident: str, uid: int) -> str:
        return (
            f'<paraminput><identifier v="{ident}"/><uid v="{uid}"/>'
            '<isConnectable v="1"/><type v="2"/><defaultValue>'
            '<constantValueFloat1 v="0"/></defaultValue>'
            '<defaultWidget><name v=""/><options/></defaultWidget></paraminput>'
        )

    def bridge(uid: int, entry: int) -> str:
        return (
            f'<compNode><uid v="{uid}"/><compOutputs><compOutput><uid v="{uid + 1}"/>'
            '<comptype v="2"/></compOutput></compOutputs><compImplementation>'
            f'<compInputBridge><entry v="{entry}"/>'
            f"<parameters>{BRIDGE_PARAMETERS}</parameters></compInputBridge>"
            "</compImplementation></compNode>"
        )

    def value(name: str, kind: str, v: str) -> str:
        return (
            f'<parameter><name v="{name}"/><relativeTo v="0"/><paramValue>'
            f'<{kind} v="{v}"/></paramValue></parameter>'
        )

    return (
        '<?xml version="1.0" encoding="UTF-8"?><package><identifier v="probe"/>'
        '<formatVersion v="1.1.0.202401"/><updaterVersion v="1.1.0.202401"/>'
        '<fileUID v="{00000000-0000-0000-0000-000000000001}"/><versionUID v="0"/>'
        '<dependencies><dependency><filename v="sbs://slope_blur.sbs"/>'
        f'<uid v="{DEPENDENCY}"/><type v="package"/><fileUID v="0"/>'
        '<versionUID v="0"/></dependency></dependencies>'
        '<content><graph><identifier v="probe"/><uid v="1"/>'
        f"<paraminputs>{paraminput('input_a', 10)}{paraminput('input_b', 11)}"
        '</paraminputs><primaryInput v="10"/><graphOutputs><graphoutput>'
        '<identifier v="output"/><uid v="20"/><channels v="2"/>'
        "</graphoutput></graphOutputs><compNodes>"
        f"{bridge(100, 10)}{bridge(110, 11)}"
        '<compNode><uid v="200"/><connections>'
        '<connection><identifier v="Source"/><connRef v="100"/>'
        '<connRefOutput v="101"/></connection>'
        '<connection><identifier v="Effect"/><connRef v="110"/>'
        '<connRefOutput v="111"/></connection></connections>'
        '<compOutputs><compOutput><uid v="201"/><comptype v="2"/></compOutput>'
        "</compOutputs><compImplementation><compInstance>"
        f'<path v="pkg:///slope_blur_grayscale?dependency={DEPENDENCY}"/><parameters>'
        f"{value('Samples', 'constantValueInt32', str(samples))}"
        f"{value('Intensity', 'constantValueFloat1', intensity)}"
        '</parameters><outputBridgings><outputBridging><uid v="201"/>'
        '<identifier v="Slope_Blur"/></outputBridging></outputBridgings>'
        "</compInstance></compImplementation></compNode>"
        '<compNode><uid v="300"/><connections><connection>'
        '<identifier v="inputNodeOutput"/><connRef v="200"/>'
        '<connRefOutput v="201"/></connection></connections>'
        '<compImplementation><compOutputBridge><output v="20"/>'
        "</compOutputBridge></compImplementation></compNode>"
        "</compNodes><baseParameters>"
        f"{value('format', 'constantValueInt32', '1' if format16 else '0')}"
        '</baseParameters><options><option><name v="defaultParentSize"/>'
        '<value v="8x8"/></option></options><root><rootOutputs><rootOutput>'
        '<output v="20"/><format v="0"/><usertag v=""/></rootOutput></rootOutputs>'
        "</root></graph></content></package>"
    )


def test_images(out: Path, size: int) -> dict[str, Path]:
    """Write the gray test images a and b (8 bits), ``size`` (log2) square."""
    n = 1 << size
    y, x = np.mgrid[0:n, 0:n] / n
    gray = {
        "a": 0.5
        + 0.25 * np.sin(2 * np.pi * 3 * x)
        + 0.25 * y * np.cos(2 * np.pi * 2 * y),
        "b": x * 0.7 + 0.3 * (0.5 + 0.5 * np.sin(2 * np.pi * 5 * (x + y))),
    }
    paths = {}
    for k, v in gray.items():
        paths[k] = out / f"in_gray_{k}.png"
        Image.fromarray((v.clip(0, 1) * 255).round().astype(np.uint8), "L").save(
            paths[k]
        )
    return paths


def encoded(mm_nodes: list, wires: list, source: tuple[str, int], size: int) -> dict:
    """Return a graph: ``source``'s 16-bit level as the albedo's red (high), green."""
    enc = shader(
        "enc", [("input", "f")], "rgba",
        "float k = floor(clamp($input(uv), 0.0, 1.0) * 65535.0 + 0.5);\n"
        "float h = floor(k / 256.0);\n"
        # a quarter step up: whole steps whether the export rounds or truncates
        "return vec4(h, k - 256.0 * h, 0.0, 255.0) / 255.0 + 0.25 / 255.0;",
    )  # fmt: skip
    material = {
        "name": "Material", "type": "material", "node_position": {"x": 3000, "y": 0},
        "export_paths": {}, "seed_int": 0,
        "parameters": {"albedo_color": {"type": "Color", "r": 1, "g": 1, "b": 1, "a": 1},
                       "metallic": 1, "roughness": 1, "emission_energy": 1, "normal": 1,
                       "ao": 1, "depth_scale": 0.5, "flags_transparent": False, "sss": 0,
                       "size": size},
    }  # fmt: skip
    wires = [
        *wires,
        {"from": source[0], "from_port": source[1], "to": "enc", "to_port": 0},
        {"from": "enc", "from_port": 0, "to": "Material", "to_port": 0},
    ]
    # No automatic buffers (a preference of Material Maker's): what the
    # translation buffers, and nothing else, as in the graphs it writes.
    wires = [{**w, "buffer": "never"} for w in wires]
    return {
        "name": "check", "type": "graph", "label": "Graph", "shortdesc": "",
        "longdesc": "", "node_position": {"x": 0, "y": 0}, "parameters": {},
        "seed_int": 0, "nodes": [*mm_nodes, enc, material], "connections": wires,
    }  # fmt: skip


def levels(path: Path) -> np.ndarray:
    """Return an exported albedo's 16-bit levels (``encoded``)."""
    a = np.asarray(Image.open(path)).astype(np.int64)
    return a[..., 0] * 256 + a[..., 1]


def same(a: np.ndarray, b: np.ndarray) -> str:
    """Return how many pixels two images agree in, and their largest difference."""
    d = np.abs(a - b)
    return f"{int((d == 0).sum()):>7} ({int(d.max())})"


def main() -> None:
    """Run the cases the command line names, or all."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("tags", nargs="*")
    parser.add_argument("--fork", type=Path, required=True, help="the fork's project")
    parser.add_argument("--godot", type=Path, default=GODOT)
    parser.add_argument("-o", "--out", type=Path, default=Path("check_loops"))
    parser.add_argument("--size", type=int, default=9, help="log2 pixels")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    size = args.size
    images = test_images(out, size)
    templates = library.load()
    todo = [c for c in cases() if not args.tags or c["tag"] in args.tags]
    made: list[str] = []
    knobbed: set[str] = set()
    built: dict[tuple, tuple] = {}
    for c in todo:
        tag = c["tag"]
        src = out / f"{tag}.sbs"
        src.write_text(
            sbs(c["samples"], c["intensity"], format16=c["format16"]), encoding="utf-8"
        )
        raw = cook(COOKER, src, out)
        if raw is None:
            print(f"{tag:22} cook failed")
            continue
        subprocess.run(
            [
                str(SBSRENDER), "render", "--input", str(out / f"{tag}.sbsar"),
                "--set-entry", f"input_a@{images['a']}",
                "--set-entry", f"input_b@{images['b']}",
                "--set-value", f"$outputsize@{size},{size}",
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
        asm = Asm(raw)
        nodes = asm.nodes()
        feeds = {"input_a": images["a"], "input_b": images["b"]}
        by_uid = {u: feeds[n].as_posix() for u, n in names.items() if n in feeds}
        inputs = graph_inputs(asm, xml)
        found = library.recognized(raw, nodes, templates, out / "matches")
        lib = library_nodes(found, nodes, inputs, size)
        loops = slope_blurs(found, nodes, lib)
        group = (c["format16"], c["intensity"])
        knob = None
        if loops:
            built[group] = (asm, nodes, by_uid, inputs, lib, loops)
        elif group in built:
            # Cooked with fewer Samples, the blends past them are folded
            # away: the Samples 32 instance, its live Samples knob set
            asm, nodes, by_uid, inputs, lib, loops = built[group]
            knob = float(c["samples"])
        else:
            print(f"{tag:22} no Slope blur to iterate (found {len(found)},"
                  f" solved {len(lib)} nodes)")  # fmt: skip
            continue
        for mode, these in (("norm", {}), ("exp", loops)):
            mm, wires, outs = translate_graph(
                nodes, {"output": asm.output_nodes[0][1]}, by_uid, size=size,
                inputs=inputs, library=lib, loops=these,
            )  # fmt: skip
            if knob is not None:
                set_samples(mm, knob)
            graph = encoded(mm, wires, outs["output"], size)
            (out / f"{tag}_{mode}.ptex").write_text(json.dumps(graph), encoding="utf-8")
            if mode == "norm":
                graph = encoded(*read_stored(mm, wires, size), outs["output"], size)
                (out / f"{tag}_stored.ptex").write_text(
                    json.dumps(graph), encoding="utf-8"
                )
        made.append(tag)
        if knob is not None:
            knobbed.add(tag)

    mm_out = out / "mm"
    mm_out.mkdir(exist_ok=True)
    paths = [out / f"{t}_{m}.ptex" for t in made for m in MODES]
    for p in paths:
        (mm_out / f"{p.stem}_albedo.png").unlink(missing_ok=True)
    if paths:
        with (out / "fork.log").open("w", encoding="utf-8", errors="replace") as log:
            subprocess.run(
                [str(args.godot), "--headless", "--path", str(args.fork),
                 "--export-material", "--target", "Godot/Godot 4 Standard",
                 "-o", str(mm_out), "--size", str(1 << size),
                 *(str(p) for p in paths)],
                stdout=log, stderr=subprocess.STDOUT, check=False, timeout=3600,
            )  # fmt: skip
    worst = 0
    print(
        f"{'':27} {'exp = stored':>14} {'exp = norm':>12}   vs Substance: same% / max"
    )
    for tag in made:
        files = {m: mm_out / f"{tag}_{m}_albedo.png" for m in MODES}
        missing = [m for m, f in files.items() if not f.exists()]
        if missing:
            print(f"{tag:22} no export of {', '.join(missing)}")
            worst = max(worst, 1 << 16)
            continue
        got = {m: levels(f) for m, f in files.items()}
        ref = np.asarray(Image.open(out / f"ref_{tag}.png")).astype(np.int64)
        ref = ref[..., 0] if ref.ndim == 3 else ref
        exp = got["exp"]
        worst = max(worst, int(np.abs(got["stored"] - exp).max()))

        subst = "  ".join(
            f"{m} {100 * float((got[m] == ref).mean()):5.2f}%"
            f" / {int(np.abs(got[m] - ref).max())}"
            for m in MODES
        )
        print(
            f"{tag:22}{' knob' if tag in knobbed else '     '}"
            f" {same(exp, got['stored'])} {same(exp, got['norm'])}   {subst}"
        )
    if not made:
        print("nothing compared")
        sys.exit(1)
    print(
        f"of {1 << 2 * size} px: exp is bit-identical to stored"
        if worst == 0
        else f"exp DIFFERS from stored (max {worst} levels)"
    )
    sys.exit(0 if worst == 0 else 1)


if __name__ == "__main__":
    main()
