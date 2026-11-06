"""Make an .sbsar's graph into a live Material Maker graph, node for node.

Where convert renders the outputs to images, this translates the compiled
graph itself (sbsar2ptex.translate): every atomic node a Material Maker
shader with its settings as parameters, the graph's inputs on a
"parameters" remote node, the outputs wired into a Material node. By
default the result is folded into nested subgraphs named after the library
nodes and repeated units found in it (sbsar2ptex.structure, with the
cooked library of sbsar2ptex.library).

Files a graph needs (embedded bitmaps, precomputed FX-Map textures) go in
a data folder beside the .ptex.
"""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sbsar2ptex.archive import member
from sbsar2ptex.convert import SLOTS
from sbsar2ptex.convert import Graph
from sbsar2ptex.convert import choose
from sbsar2ptex.libparams import knobs
from sbsar2ptex.libparams import solve
from sbsar2ptex.library import recognized
from sbsar2ptex.loops import FEATURES
from sbsar2ptex.loops import Chains
from sbsar2ptex.loops import chains
from sbsar2ptex.loops import slope_blurs
from sbsar2ptex.plugin import PLUGIN_ID
from sbsar2ptex.plugin import VERSION
from sbsar2ptex.plugin import Maker
from sbsar2ptex.recognize import Match
from sbsar2ptex.recognize import Template
from sbsar2ptex.recognize import resembler
from sbsar2ptex.sbsasm import BASE
from sbsar2ptex.sbsasm import Asm
from sbsar2ptex.sbsasm import Node
from sbsar2ptex.slots import template_slots
from sbsar2ptex.structure import nest
from sbsar2ptex.translate import GraphInput
from sbsar2ptex.translate import graph_inputs
from sbsar2ptex.translate import strip_links
from sbsar2ptex.translate import translate_graph

LENGTH_SIZE = 4
MM_MOD = ">=1.17"
"""The material-maker-mod version --experimental graphs need."""


SUFFIXES = {
    b"\x89PNG": ".png",
    b"DDS ": ".dds",
    b"v/1\x01": ".exr",
    b"II*\x00": ".tif",
    b"MM\x00*": ".tif",
}
"""An embedded file's suffix by its header (else a JPEG's)."""


def _suffix(blob: bytes) -> str:
    return next((s for magic, s in SUFFIXES.items() if blob.startswith(magic)), ".jpg")


def bitmaps(asm: Asm, nodes: list[Node], folder: Path) -> dict[int, str]:
    """Write the embedded bitmaps; return their files by the id inputs use.

    An image input node names its bitmap by the offset (from BASE) of the
    length word before it, or of its raw pixels (Asm.raw_image): those are
    written as 8-bit PNGs, as Material Maker loads images in 8 bits. Each
    node's image is resolved by its uid (docs/image-inputs.md): the files
    in an unbroken run from the data area's start, then any other encoded
    file a node's uid points at (Asm.encoded_image), past a raw image.
    """
    folder.mkdir(parents=True, exist_ok=True)
    out: dict[int, str] = {}
    for k, (at, length) in enumerate(asm.bitmaps):
        blob = asm.d[at : at + length]
        path = folder / f"bitmap_{k:02d}{_suffix(blob)}"
        path.write_bytes(blob)
        out[at - BASE - LENGTH_SIZE] = path.resolve().as_posix()
    for node in nodes:
        found = None if node.uid in out else asm.encoded_image(node)
        if found is not None:
            blob = asm.d[found[0] : found[0] + found[1]]
            path = folder / f"image_{node.index}{_suffix(blob)}"
            path.write_bytes(blob)
            out[node.uid] = path.resolve().as_posix()
    for node in nodes:
        raw = None if node.uid in out else asm.raw_image(node)
        if raw is None:
            continue
        dtype = np.uint8 if raw.depth == 1 else np.dtype("<u2")
        px = np.frombuffer(raw.pixels, dtype).reshape(
            raw.height, raw.width, raw.channels
        )
        if raw.depth == 2:  # noqa: PLR2004
            px = ((px.astype(np.uint32) * 255 + 32767) // 65535).astype(np.uint8)
        path = folder / f"raw_{node.index}.png"
        Image.fromarray(px[..., 0] if raw.channels == 1 else px).save(path)
        out[node.uid] = path.resolve().as_posix()
    return out


def material(
    slots: dict[str, tuple[str, int]], *, normal_directx: bool, size: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return a Material node (and a normal converter) and its wires."""
    ports = {slot: n for n, (slot, _) in enumerate(SLOTS)}
    nodes: list[dict[str, Any]] = [
        {
            "name": "Material",
            "type": "material",
            "node_position": {"x": 3000, "y": 0},
            "parameters": {
                "albedo_color": {"type": "Color", "r": 1, "g": 1, "b": 1, "a": 1},
                "metallic": 1,
                "roughness": 1,
                "emission_energy": 1,
                "normal": 1,
                "ao": 1,
                "depth_scale": 0.5,
                "flags_transparent": "opacity" in slots,
                "sss": 0,
                "size": size,
            },
            "export_paths": {},
            "seed_int": 0,
        }
    ]
    wires = []
    for slot, source in slots.items():
        node, port = source
        if slot == "normal":
            nodes.append(
                {
                    "name": "normal_convert",
                    "type": "normal_map_convert",
                    "node_position": {"x": 2800, "y": 400},
                    "parameters": {"op": 1 if normal_directx else 0},
                    "seed_int": 0,
                }
            )
            wires.append(
                {"from": node, "from_port": port, "to": "normal_convert", "to_port": 0}
            )
            node, port = "normal_convert", 0
        wires.append(
            {"from": node, "from_port": port, "to": "Material", "to_port": ports[slot]}
        )
    return nodes, wires


def library_nodes(
    found: list[Match], nodes: list[Node], inputs: dict[int, GraphInput], size: int
) -> dict[int, tuple[Node, dict[int, GraphInput]]]:
    """Return translate_graph's ``library``: instances' nodes and their templates'.

    For each instance whose library parameters are recovered (libparams):
    its every node's template node, and the inputs those read, the
    parameters the instance's own.
    """
    out: dict[int, tuple[Node, dict[int, GraphInput]]] = {}
    for m in found:
        solved = solve(m, nodes, [size, size])
        if solved is None:
            continue
        reads = knobs(m, solved, inputs)
        for tk, gk in m.members.items():
            out[gk] = (m.template.nodes[tk], reads)
    return out


def library_slots(
    found: list[Match], nodes: list[Node], lib: dict[int, Any]
) -> dict[int, dict[int, int]]:
    """Return translate_graph's ``template_slots``: each library node's slot map.

    An instance's template slots map to its own writers', matched by node
    (slots.template_slots, docs/function-ops.md 2.6).
    """
    out: dict[int, dict[int, int]] = {}
    for m in found:
        if not any(gk in lib for gk in m.members.values()):
            continue
        alias = template_slots(m.members, m.template.nodes, nodes)
        out.update((gk, alias) for gk in m.members.values() if gk in lib)
    return out


def build(
    files: dict[str, bytes],
    graph: Graph,
    data: Path,
    *,
    size: int = 11,
    library: list[Template] | None = None,
    flat: bool = False,
    aliases: dict[str, str] | None = None,
    matches: Path | None = None,
    experimental: bool = False,
    loops: bool = True,
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the Material Maker graph translating ``graph``.

    ``files`` is the archive's contents; ``data`` the folder for files the
    graph uses; ``library`` the templates for naming library nodes (none:
    groups are named by content); ``flat``: no subgraphs; ``aliases`` is
    filled as translate_graph fills it; ``matches`` a folder keeping the
    library nodes found (library.recognized). ``experimental`` uses the
    material-maker-mod fork's features (sbsar2ptex.loops) and the sbs
    plugin's nodes (sbsar2ptex.plugin), and says so: a "requires" key and
    long description line, as the fork reads them; ``loops`` False keeps
    the loops unrolled (every node's image a node's, for scripts/descend.py
    to compare). ``stats`` is filled with
    the plugin nodes made, by type, and the nodes left inline, by reason.
    """
    xml = member(files, ".xml").decode("utf-8")
    asm = Asm(member(files, ".sbsasm"))
    nodes = asm.nodes()
    names = {int(u): n for u, n in re.findall(r'uid="(\d+)" identifier="([^"]*)"', xml)}
    node_of = {
        names.get(uid): node
        for uid, (_, node) in zip(asm.output_uids, asm.output_nodes, strict=True)
    }
    chosen = choose(graph)
    wanted = {o.identifier: node_of[o.identifier] for o in chosen.values()}
    origins: dict[str, int] = {}
    inputs = graph_inputs(asm, xml)
    found = (
        recognized(member(files, ".sbsasm"), nodes, library, matches)
        if library and not flat
        else []
    )
    lib = library_nodes(found, nodes, inputs, size)
    looped = experimental and loops
    slopes = slope_blurs(found, nodes, lib) if looped else {}
    repeated = chains(nodes, set(wanted.values())) if looped else []
    folded = Chains(repeated) if looped else None
    # Plugin nodes, of one pass in the chains (an iterate's body: no buffers)
    in_chains = {k for c in repeated for period in c.periods for k in period}
    maker = Maker(size, in_chains) if experimental else None
    mm, wires, outs = translate_graph(
        nodes,
        wanted,
        bitmaps(asm, nodes, data / "bitmaps"),
        size=size,
        inputs=inputs,
        data_dir=data / "fxmaps",
        origins=origins,
        aliases=aliases,
        library=lib,
        loops=slopes,
        fold=folded,
        plugins=maker,
        template_slots=library_slots(found, nodes, lib),
    )
    if maker is not None and stats is not None:
        stats["plugin nodes"] = dict(sorted(maker.made.items()))
        stats["inline"] = dict(sorted(maker.inline.items()))
    if not flat:
        instances = [
            (list(m.members.values()), m.template.label or m.template.graph)
            for m in found
        ]
        mm, wires, outs = nest(
            mm,
            wires,
            outs,
            origins,
            nodes,
            instances,
            resembler(nodes, library) if library else None,
        )
    slots = {slot: outs[o.identifier] for slot, o in chosen.items()}
    normal_id = chosen["normal"].identifier.lower() if "normal" in chosen else ""
    extra, extra_wires = material(
        slots, normal_directx="opengl" not in normal_id, size=size
    )
    made = {
        "name": "material",
        "type": "graph",
        "label": graph.label or "Graph",
        "shortdesc": graph.label,
        "longdesc": f"{graph.description}\n\nTranslated from {graph.url}.",
        "node_position": {"x": 0, "y": 0},
        "parameters": {},
        "seed_int": 0,
        "nodes": [*mm, *extra],
        "connections": [*wires, *extra_wires],
    }
    if experimental:
        iterated = any(loop.made for loop in slopes.values()) or bool(
            folded and folded.made
        )
        used = FEATURES if iterated else ()
        requires: dict[str, Any] = {
            "material_maker_mod": MM_MOD,
            "features": list(used),
        }
        if maker is not None and maker.made:
            requires["plugins"] = {PLUGIN_ID: f">={VERSION}"}
        made["requires"] = requires
        made["longdesc"] += f"\nrequires: {json.dumps(requires)}"
    strip_links(made)
    return made
