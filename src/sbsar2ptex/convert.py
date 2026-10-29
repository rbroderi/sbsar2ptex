"""Turn a Substance archive (.sbsar) into a Material Maker graph (.ptex).

    sbsar2ptex MATERIAL.sbsar [-o OUT_DIR] [--size 8] [--set NAME=VALUE ...]
        [--maps DIR [--normal opengl|directx]] [--sbsrender PATH] [--embedded]
    sbsar2ptex MATERIAL.sbsar --procedural [--flat] [--no-library]
        [--library DIR] [--sbscooker PATH] [--refresh-library]

An .sbsar is a 7z archive of a description (XML: the graph's outputs, their
channels, its parameters) and a compiled graph (.sbsasm: Substance engine
bytecode). The bytecode is not a node graph Material Maker could rebuild, so
this renders the outputs and makes a .ptex of image nodes wired into a
Material node, one per channel Material Maker has:

- the maps come from Adobe's ``sbsrender`` (Substance 3D Automation Toolkit,
  or the one Substance Designer ships), found on PATH or in the usual install
  folders, or given by ``--sbsrender``; ``--set`` passes the archive's
  parameters to it (``--list`` prints them), ``--size`` is log2 pixels;
- or, with no ``sbsrender``, from ``--maps DIR``: maps exported by hand
  (Substance Player, Painter, Designer), found by output name or channel
  (``*_basecolor.png``, ``*_normal.png``, ...).

``--procedural`` instead translates the compiled graph into Material Maker
nodes (sbsar2ptex.procedural): live, editable, folded into subgraphs named
after the Substance library nodes found in it (unless ``--flat``). Naming
them needs Substance's library cooked: the first run cooks it with
Designer's sbscooker into a cache (``--library`` to choose where,
``--no-library`` to skip); the graph's files go in ``NAME_data/``.

``--embedded`` also writes out the bitmaps baked into the bytecode (often a
material's scanned grunge and scratch sources), for building on in Material
Maker. Everything lands in OUT_DIR (default: beside the archive):
``NAME.ptex`` and ``NAME_maps/``.
"""

import argparse
import json
import re
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from defusedxml import ElementTree

from sbsar2ptex.archive import read_archive

SBSRENDER_PLACES = (
    Path("C:/Program Files/Adobe/Adobe Substance 3D Automation Toolkit"),
    Path("C:/Program Files/Adobe/Adobe Substance 3D Designer"),
    Path("C:/Program Files/Allegorithmic/Substance Automation Toolkit"),
    Path("C:/Program Files/Allegorithmic/Substance Designer"),
    Path("/Applications/Adobe Substance 3D Designer.app/Contents/MacOS"),
    Path("/opt/Adobe/Adobe_Substance_3D_Automation_Toolkit"),
    Path("/opt/Allegorithmic/Substance_Automation_Toolkit"),
)
SLOTS = (
    # Material Maker's material node inputs, in port order, and the Substance
    # channel usages that fill each (first found wins).
    ("albedo", ("baseColor", "diffuse")),
    ("metallic", ("metallic",)),
    ("roughness", ("roughness",)),
    ("emission", ("emissive",)),
    ("normal", ("normal",)),
    ("ao", ("ambientOcclusion",)),
    ("depth", ("height",)),
    ("opacity", ("opacity",)),
)
LENGTH_SIZE = 4
"""Bytes of the length before each embedded bitmap."""
MAP_SUFFIXES = (".png", ".tga", ".jpg", ".jpeg", ".exr", ".tif", ".tiff", ".bmp")
INPUT_TYPES = {
    0: "float",
    1: "float2",
    2: "float3",
    3: "float4",
    4: "int",
    5: "image",
    6: "string",
    8: "int2",
    9: "int3",
    10: "int4",
}


@dataclass
class Output:
    """One of the graph's outputs: its identifier and the channels it carries."""

    identifier: str
    usages: list[str]
    packed: bool
    """More than one channel in one image (MRAO, DDNA, ...)."""


@dataclass
class Graph:
    """What an .sbsar's description says about one graph."""

    url: str
    label: str
    description: str
    outputs: list[Output] = field(default_factory=list)
    inputs: list[dict[str, str]] = field(default_factory=list)


def describe(xml: bytes) -> list[Graph]:
    """Return the graphs an .sbsar's description XML lists."""
    graphs: list[Graph] = []
    for g in ElementTree.fromstring(xml).iter("graph"):
        graph = Graph(g.get("pkgurl", ""), g.get("label", ""), g.get("description", ""))
        for o in g.iter("output"):
            channels = o.findall("./outputgui/channels/channel")
            usages = [c.get("names", "") for c in channels]
            packed = len(channels) > 1 or any(
                c.get("components", "RGB") not in ("RGB", "RGBA", "R") for c in channels
            )
            graph.outputs.append(Output(o.get("identifier", ""), usages, packed))
        for i in g.iter("input"):
            gui = i.find("inputgui")
            graph.inputs.append(
                {
                    "identifier": i.get("identifier", ""),
                    "type": INPUT_TYPES.get(
                        int(i.get("type", "-1")), i.get("type", "")
                    ),
                    "default": i.get("default", ""),
                    "label": gui.get("label", "") if gui is not None else "",
                }
            )
        graphs.append(graph)
    return graphs


def choose(graph: Graph) -> dict[str, Output]:
    """Return the output that fills each Material Maker slot.

    Single-channel outputs beat packed ones, and the OpenGL normal (Material
    Maker's convention) beats the DirectX one.
    """
    chosen: dict[str, Output] = {}
    for slot, usages in SLOTS:
        for usage in usages:
            found = [o for o in graph.outputs if usage in o.usages and not o.packed]
            if usage == "normal":
                found.sort(key=lambda o: "opengl" not in o.identifier.lower())
            if found:
                chosen[slot] = found[0]
                break
    return chosen


def default_size(graph: Graph) -> int:
    """Return the size (log2 pixels) the graph renders at unless told otherwise.

    Its $outputsize input's default (the width), or Substance's own default,
    8 (256 pixels), when it has none.
    """
    for i in graph.inputs:
        if i["identifier"] == "$outputsize" and i["default"]:
            return int(i["default"].split(",")[0])
    return 8


def find_sbsrender(given: str | None) -> Path | None:
    """Return sbsrender: the one given, on PATH, or in an install folder."""
    if given:
        return Path(given)
    on_path = shutil.which("sbsrender")
    if on_path:
        return Path(on_path)
    for place in SBSRENDER_PLACES:
        for name in ("sbsrender.exe", "sbsrender"):
            if (place / name).is_file():
                return place / name
    return None


def render(
    sbsrender: Path,
    sbsar: Path,
    graph: Graph,
    outputs: list[Output],
    maps: Path,
    size: int,
    values: list[str],
) -> dict[str, Path]:
    """Render ``outputs`` to PNGs in ``maps``; return them by output identifier."""
    command = [
        str(sbsrender),
        "render",
        "--input",
        str(sbsar),
        "--input-graph",
        graph.url,
        "--output-path",
        str(maps),
        "--output-name",
        "{outputNodeName}",
        "--output-format",
        "png",
        "--set-value",
        f"$outputsize@{size},{size}",
    ]
    for value in values:
        name, _, setting = value.partition("=")
        command += ["--set-value", f"{name}@{setting}"]
    for output in outputs:
        command += ["--input-graph-output", output.identifier]
    subprocess.run(command, check=True)  # noqa: S603 (the user's own tool)
    return {
        o.identifier: maps / f"{o.identifier}.png"
        for o in outputs
        if (maps / f"{o.identifier}.png").is_file()
    }


def match(given: Path, outputs: dict[str, Output], maps: Path) -> dict[str, Path]:
    """Copy hand-exported maps from ``given`` into ``maps``, by slot.

    A file matches a slot when its name ends in the output's identifier or one
    of the slot's channel names (``Foo_BaseColor.png``, ``foo-normal.tga``).
    """
    files = [p for p in given.iterdir() if p.suffix.lower() in MAP_SUFFIXES]

    def key(text: str) -> str:
        return re.sub(r"[^a-z0-9]", "", text.lower())

    found: dict[str, Path] = {}
    for slot, usages in SLOTS:
        names = ([outputs[slot].identifier] if slot in outputs else []) + list(usages)
        for name in map(key, names):
            hits = [p for p in files if key(p.stem).endswith(name)]
            if hits:
                if hits[0].parent.resolve() == maps.resolve():
                    found[slot] = hits[0]
                else:
                    target = maps / f"{slot}{hits[0].suffix.lower()}"
                    shutil.copyfile(hits[0], target)
                    found[slot] = target
                break
    return found


def embedded(sbsasm: bytes, out: Path) -> list[Path]:
    """Write the JPEG and PNG bitmaps baked into the bytecode; return them.

    Each sits after a little-endian uint32 of its length.
    """
    written: list[Path] = []
    for magic, suffix in ((b"\xff\xd8\xff", ".jpg"), (b"\x89PNG\r\n\x1a\n", ".png")):
        at = sbsasm.find(magic)
        while at >= LENGTH_SIZE:
            length = struct.unpack_from("<I", sbsasm, at - LENGTH_SIZE)[0]
            if 0 < length <= len(sbsasm) - at:
                path = out / f"embedded_{len(written):02d}{suffix}"
                path.write_bytes(sbsasm[at : at + length])
                written.append(path)
                at += length
            else:
                at += len(magic)
            at = sbsasm.find(magic, at)
    return written


def ptex(graph: Graph, maps: dict[str, Path], size: int, normal: str) -> dict[str, Any]:
    """Return a Material Maker graph: an image node per map into a Material.

    ``normal`` is the normal map's convention, "opengl" or "directx".
    """
    ports = {slot: n for n, (slot, _) in enumerate(SLOTS)}
    nodes: list[dict[str, Any]] = [
        {
            "name": "Material",
            "type": "material",
            "node_position": {"x": 400, "y": 0},
            "parameters": {
                "albedo_color": {"type": "Color", "r": 1, "g": 1, "b": 1, "a": 1},
                "metallic": 1,
                "roughness": 1,
                "emission_energy": 1,
                "normal": 1,
                "ao": 1,
                "depth_scale": 0.5,
                "flags_transparent": "opacity" in maps,
                "sss": 0,
                "size": size,
            },
            "export_paths": {},
            "seed_int": 0,
        }
    ]
    connections: list[dict[str, Any]] = []
    for row, (slot, path) in enumerate(sorted(maps.items(), key=lambda m: ports[m[0]])):
        name = f"image_{slot}"
        nodes.append(
            {
                "name": name,
                "type": "image",
                "node_position": {"x": 0, "y": row * 160},
                "parameters": {"image": path.resolve().as_posix()},
                "seed_int": 0,
            }
        )
        if slot == "normal":
            # The Material node takes normals in Material Maker's own form;
            # normal_map_convert makes it from an OpenGL (0) or DirectX (1) map.
            nodes.append(
                {
                    "name": "normal_convert",
                    "type": "normal_map_convert",
                    "node_position": {"x": 200, "y": row * 160},
                    "parameters": {"op": 1 if normal == "directx" else 0},
                    "seed_int": 0,
                }
            )
            connections.append(
                {"from": name, "from_port": 0, "to": "normal_convert", "to_port": 0}
            )
            name = "normal_convert"
        connections.append(
            {"from": name, "from_port": 0, "to": "Material", "to_port": ports[slot]}
        )
    parameters = "\n".join(
        f"{i['identifier']} ({i['type']}) = {i['default']}" for i in graph.inputs
    )
    return {
        "name": "material",
        "type": "graph",
        "label": graph.label or "Graph",
        "shortdesc": graph.label,
        "longdesc": f"{graph.description}\n\nConverted from {graph.url}. "
        f"Substance parameters:\n{parameters}",
        "node_position": {"x": 0, "y": 0},
        "parameters": {},
        "seed_int": 0,
        "nodes": nodes,
        "connections": connections,
    }


def main() -> None:
    """Convert the archive the command line names."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sbsar", type=Path)
    parser.add_argument(
        "-o", "--out", type=Path, help="output folder (default: beside it)"
    )
    parser.add_argument(
        "--graph", help="which graph, by pkg:// url (default: the first)"
    )
    parser.add_argument(
        "--size",
        type=int,
        help="log2 pixels (11 = 2048; default: the graph's own, often 8 = 256)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="a Substance parameter, e.g. Flesh_Amount=0.5 (sbsrender only)",
    )
    parser.add_argument(
        "--maps", type=Path, help="hand-exported maps, used without sbsrender"
    )
    parser.add_argument(
        "--normal",
        choices=("opengl", "directx"),
        default="opengl",
        help="the hand-exported normal map's convention (default opengl)",
    )
    parser.add_argument("--sbsrender", help="path to sbsrender")
    parser.add_argument(
        "--embedded",
        action="store_true",
        help="also write the bitmaps baked into the archive",
    )
    parser.add_argument(
        "--list", action="store_true", help="print outputs and parameters"
    )
    parser.add_argument(
        "--procedural",
        action="store_true",
        help="translate the graph into live Material Maker nodes, not images",
    )
    parser.add_argument(
        "--flat", action="store_true", help="with --procedural: no subgraphs"
    )
    parser.add_argument(
        "--experimental",
        action="store_true",
        help="with --procedural: use the material-maker-mod fork's iterate"
        " buffers for loops (Slope blur, repeated chains); the graph then"
        " requires the fork",
    )
    parser.add_argument(
        "--no-library",
        action="store_true",
        help="with --procedural: do not name Substance library nodes",
    )
    parser.add_argument(
        "--library", type=Path, help="cooked library cache (default: per user)"
    )
    parser.add_argument("--sbscooker", help="path to sbscooker (to cook the library)")
    parser.add_argument(
        "--refresh-library",
        action="store_true",
        help="re-read the cooked library (after a Designer update)",
    )
    args = parser.parse_args()

    files = read_archive(args.sbsar)
    xml = next((d for n, d in files.items() if n.endswith(".xml")), None)
    if xml is None:
        sys.exit(f"{args.sbsar}: no description XML; not an .sbsar?")
    graphs = describe(xml)
    graph = next((g for g in graphs if args.graph in (None, g.url)), None)
    if graph is None:
        sys.exit(f"no graph {args.graph}; have: {', '.join(g.url for g in graphs)}")
    chosen = choose(graph)
    if args.size is None:
        args.size = default_size(graph)

    if args.list:
        print(f"{graph.url}  {graph.label}")
        for o in graph.outputs:
            slot = next((s for s, c in chosen.items() if c is o), "")
            print(f"  output {o.identifier:20} {'+'.join(o.usages):30} {slot}")
        for i in graph.inputs:
            print(
                f"  input  {i['identifier']:40} {i['type']:7}",
                f"{i['default']:24} {i['label']}",
            )
        return

    stem = args.sbsar.stem
    out = args.out or args.sbsar.parent
    if args.procedural:
        procedural_main(args, files, graph, out / f"{stem}.ptex", out / f"{stem}_data")
        return
    maps = out / f"{stem}_maps"
    maps.mkdir(parents=True, exist_ok=True)

    if args.embedded:
        sbsasm = b"".join(d for n, d in files.items() if n.endswith(".sbsasm"))
        for path in embedded(sbsasm, maps):
            print("embedded", path)

    sbsrender = None if args.maps else find_sbsrender(args.sbsrender)
    if sbsrender is not None:
        rendered = render(
            sbsrender,
            args.sbsar,
            graph,
            list(chosen.values()),
            maps,
            args.size,
            args.set,
        )
        found = {
            s: rendered[o.identifier]
            for s, o in chosen.items()
            if o.identifier in rendered
        }
        # Substance's plain "normal" output is DirectX unless it says OpenGL.
        normal_id = chosen["normal"].identifier.lower() if "normal" in chosen else ""
        normal = "opengl" if "opengl" in normal_id else "directx"
    elif args.maps is not None:
        normal = args.normal
        if args.set:
            print("note: --set needs sbsrender; ignored for hand-exported maps")
        found = match(args.maps, chosen, maps)
    else:
        sys.exit(
            "no sbsrender found (Substance 3D Automation Toolkit or Designer):\n"
            "  give --sbsrender PATH, or export the maps yourself (Substance\n"
            "  Player/Painter/Designer) and give --maps DIR"
        )
    missing = [s for s, _ in SLOTS if s in chosen and s not in found]
    if missing:
        print("no map for:", ", ".join(missing))
    if not found:
        sys.exit("no maps; nothing to wire")

    target = out / f"{stem}.ptex"
    target.write_text(
        json.dumps(ptex(graph, found, args.size, normal), indent=1), encoding="utf-8"
    )
    for slot, path in found.items():
        print(f"{slot:9} {path}")
    print("wrote", target)


def procedural_main(
    args: argparse.Namespace,
    files: dict[str, bytes],
    graph: Graph,
    target: Path,
    data: Path,
) -> None:
    """Write the translated graph (``--procedural``)."""
    from sbsar2ptex import library  # noqa: PLC0415 (library imports convert's peers)
    from sbsar2ptex import procedural  # noqa: PLC0415

    templates = []
    if not (args.flat or args.no_library):
        templates = library.load(
            args.library, args.sbscooker, refresh=args.refresh_library
        )
        if not templates:
            print("no Substance library to name nodes by (no Designer found)")
    try:
        made = procedural.build(
            files,
            graph,
            data,
            size=args.size,
            library=templates,
            flat=args.flat,
            matches=(args.library or library.cache_dir()) / "matches",
            experimental=args.experimental,
        )
    except ValueError as e:  # bytecode this decoder cannot read
        sys.exit(f"{args.sbsar}: {e}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(made, indent=1), encoding="utf-8")
    print("wrote", target)


if __name__ == "__main__":
    main()
