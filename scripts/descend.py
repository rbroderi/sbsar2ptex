"""Walk back from a node that differs from Substance to the first that does.

    uv run python scripts/descend.py SBSAR NODE --mm PATH/TO/material_maker.exe
        [--sbsrender PATH] [--work DIR] [--size 10] [--strict]
        [--bisect [PROBES]] [--pin NODES]

NODE is a compiled node index (``sbsasm-dump`` lists them; a graph output's
node is a good start). A node is rendered twice: in Substance, by
retargeting a graph output to it (a grayscale output for a grayscale node, a
color one for a color node) and running sbsrender; and in Material Maker, by
exporting the translated graph (flat, ``--procedural --flat``) pruned to the
node and the nodes feeding it. The nodes a step compares are rendered
together: one sbsrender run for as many as the graph has outputs, one
Material Maker run for them all (its start-up is most of an export). A node
matches when the two images agree pixel for pixel, or when they have the same
content placed elsewhere: the same value distribution and the same frequency
spectrum (random placement that differs only by GPU float rounding, such as
a hash of cos, moves things without changing them). From a differing node it
follows the input that matches worst, and stops at a node whose inputs all
match: the first node translated wrong. A color node matches when each of
its red, green and blue channels does, and its alpha too where a node
downstream reads that (alpha_read: an FX-Map's patterns, a shuffle's
picks, ...); Material Maker's alpha is exported apart, as gray, since its
albedo export is opaque. ``--strict`` turns placement off,
for a node fed by one placed elsewhere that a later step (a narrow levels,
a gradient lookup) magnifies: such an input can be the real difference.

``--bisect`` (strict) finds the node faster on a long chain: it bisects a
path up from the differing node instead of comparing every input of every
node on the way (see ``bisect``), and still checks every input of the node
it ends on.

``--pin`` takes Substance's render of a node in place of its translation,
for one known to differ for reasons no translation can fix (Biomech's n17
hashes with cos, which Substance's own engines round differently): the
nodes it feeds are then compared from the same input.

Material Maker exports 8 bits. A grayscale node whose Substance image spans
less than NARROW (1st to 99th percentile) is magnified before export,
(v - median) * k + 0.5 with its spread filling most of the range, and
Substance's image the same way, so that steps finer than 1/255 count.

The flat graph is built once into the work folder (``--rebuild`` to redo it).
Side-by-side images (Substance left) go to WORK/vs.
"""

import argparse
import copy
import json
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import py7zr
from PIL import Image

from sbsar2ptex import procedural
from sbsar2ptex.archive import member
from sbsar2ptex.archive import read_archive
from sbsar2ptex.convert import describe
from sbsar2ptex.sbsasm import COLOR_OPS
from sbsar2ptex.sbsasm import OPS
from sbsar2ptex.sbsasm import Asm
from sbsar2ptex.sbsasm import Node

SBSRENDER = Path("C:/Program Files/Adobe/Adobe Substance 3D Designer/sbsrender.exe")
SIDE = 512
"""Images are compared at this size."""
NARROW = 16 / 255
"""A grayscale image spanning less than this is compared magnified."""
MAX_EXPORTS = 40
"""Graphs one Material Maker run exports (its command line has a limit)."""


def load(path: Path, channels: int = 1) -> np.ndarray:
    """Return an image's first ``channels`` channels as floats in [0, 1].

    Each is SIDE square; with more than one, they are the last axis.
    """
    im = Image.open(path)
    a = np.asarray(im).astype(float)
    a = a / 65535 if im.mode.startswith("I") or a.max() > 255 else a / 255
    a = a[..., None] if a.ndim == 2 else a
    out = []
    for c in range(channels):
        ch = a[..., min(c, a.shape[2] - 1)]
        im = Image.fromarray((np.clip(ch, 0, 1) * 65535).astype(np.uint16))
        out.append(np.asarray(im.resize((SIDE, SIDE), Image.BILINEAR)) / 65535)
    return out[0] if channels == 1 else np.stack(out, axis=-1)


def spectrum(x: np.ndarray) -> np.ndarray:
    """Return the log of the radially averaged power spectrum."""
    p = np.abs(np.fft.fftshift(np.fft.fft2(x - x.mean()))) ** 2
    n = x.shape[0]
    y, xx = np.indices(p.shape) - n // 2
    r = np.hypot(y, xx).astype(int)
    radial = np.bincount(r.ravel(), p.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    return np.log10(radial[1 : n // 2] + 1e-12)


def compare(
    t: np.ndarray, o: np.ndarray, *, strict: bool = False
) -> tuple[bool, str, float]:
    """Return whether two images match, a summary, and their correlation.

    ``strict``: content placed elsewhere does not match.
    """
    flat_t, flat_o = t.std() < 1e-4, o.std() < 1e-4
    if flat_t or flat_o:
        # A flat image matches only one that is flat to within 8-bit
        # rounding: a sparse mask (a few pixels off a constant) moves the
        # mean too little to tell.
        flat = max(t.std(), o.std()) < 1 / 255
        c = 1.0 if flat and abs(t.mean() - o.mean()) < 0.02 else 0.0
    else:
        c = float(np.corrcoef(t.ravel(), o.ravel())[0, 1])
    err = float(np.abs(t - o).mean()) * 255
    q = np.linspace(0, 1, 101)
    gap = float(np.abs(np.quantile(t, q) - np.quantile(o, q)).max()) * 255
    if flat_t or flat_o:
        spec = 1.0 if flat_t and flat_o and gap < 6 else 0.0
    else:
        spec = float(np.corrcoef(spectrum(t), spectrum(o))[0, 1])
    same = c > 0.97 and err < 6
    moved = gap < 12 and spec > 0.95
    text = f"corr {c:.3f} err {err:.1f} quantile gap {gap:.1f} spectrum {spec:.3f}"
    return (
        same or (moved and not strict),
        text + (" (placed elsewhere)" if moved and not same else ""),
        c,
    )


def compare_color(
    t: np.ndarray, o: np.ndarray, *, strict: bool = False
) -> tuple[bool, str, float]:
    """Return ``compare`` over the channels (red, green, blue[, alpha]): the worst."""
    n = t.shape[-1]
    per = [compare(t[..., c], o[..., c], strict=strict) for c in range(n)]
    k = min(range(n), key=lambda c: (per[c][0], per[c][2]))
    _, text, c = per[k]
    return all(p[0] for p in per), f"{'rgba'[k]}: {text}", c


def alpha_read(node: Node, alpha: bool) -> set[str]:
    """Return the inputs whose alpha ``node``'s output depends on.

    ``alpha``: whether its own alpha counts (a node downstream reads it).
    An FX-Map draws its patterns through their alpha (a color one over its
    background); a shuffle reads the alpha its selector picks; a color blend
    composites by alpha; a grayscale conversion multiplies by it when its
    flag says. Other color nodes pass their inputs' alpha on to their own.
    """
    color_in = {name for name, m in node.inputs.items() if m is not None}
    out: set[str] = set()
    if node.op in (8, 9):
        out |= {name for name in node.inputs if name.startswith("pattern")}
        if node.op == 9:
            out |= {"background"} & color_in
    elif node.op == 7:
        picks = {node.fields[0] >> 8 * c & 0xFF for c in range(4)}
        out |= {"input1"} if 3 in picks else set()
        out |= {"input2"} if 7 in picks else set()
        return out & set(node.inputs)
    elif node.op == 3:
        out |= {"source", "destination"}
    elif node.op == 6 and node.flags >> 8 & 0x10:
        out |= {"input"}
    if alpha and node.op in COLOR_OPS:
        out |= color_in
    return out & set(node.inputs)


def wiring(graph: dict) -> dict[str, set[str]]:
    """Return each node's direct inputs, by name."""
    into: dict[str, set[str]] = {}
    for w in graph["connections"]:
        into.setdefault(w["to"], set()).add(w["from"])
    return into


def cone(into: dict[str, set[str]], name: str) -> set[str]:
    """Return ``name`` and every node feeding it."""
    seen, todo = set(), [name]
    while todo:
        n = todo.pop()
        if n not in seen:
            seen.add(n)
            todo += into.get(n, ())
    return seen


def magnifier(t: np.ndarray) -> tuple[float, float] | None:
    """Return (center, gain) to compare a narrow grayscale image by; None if wide.

    The gain spreads its 1st to 99th percentile over 0.1 to 0.9.
    """
    lo, mid, hi = np.quantile(t, [0.01, 0.5, 0.99])
    reach = max(hi - mid, mid - lo)
    if hi - lo >= NARROW or reach < 1e-6:
        return None
    return float(mid), float(min(0.4 / reach, 1024.0))


def magnified(t: np.ndarray, center: float, gain: float) -> np.ndarray:
    """Return ``t`` as the magnifying node renders it."""
    return np.clip((t - center) * gain + 0.5, 0, 1)


def build(
    graph: dict,
    into: dict[str, set[str]],
    src: str,
    path: Path,
    magnify: tuple[float, float] | None = None,
    *,
    alpha: bool = False,
) -> Path:
    """Write ``src`` alone (with what feeds it) as the albedo, to ``path``.

    ``magnify`` (center, gain) exports ``magnified`` of it instead; ``alpha``
    its alpha, as gray (the albedo's own alpha is exported opaque).
    """
    g = copy.deepcopy(graph)
    keep = cone(into, src) | {"Material"}
    g["nodes"] = [n for n in g["nodes"] if n["name"] in keep or n["type"] == "remote"]
    for n in g["nodes"]:
        if n["type"] == "remote":  # its widgets may point at pruned nodes
            for wd in n["widgets"]:
                wd["linked_widgets"] = [
                    lw for lw in wd["linked_widgets"] if lw["node"] in keep
                ]
    g["connections"] = [
        w for w in g["connections"]
        if w["to"] != "Material" and w["from"] in keep and w["to"] in keep
    ]  # fmt: skip
    out = src
    if magnify:
        out = f"{src}_magnified"
        c, k = magnify
        g["nodes"].append({
            "name": out, "type": "shader", "node_position": {"x": 0, "y": 0},
            "parameters": {}, "seed_int": 0,
            "shader_model": {
                "name": out, "code": "", "global": "", "parameters": [],
                "inputs": [{"name": "input", "type": "f", "default": "0.0",
                            "function": True, "label": "input"}],
                "instance": (
                    f"float $(name)_fct(vec2 uv) {{\n  return clamp(($input(uv) - {c!r})"
                    f" * {k!r} + 0.5, 0.0, 1.0);\n}}\n"
                ),
                "outputs": [{"type": "f", "f": "$(name)_fct($uv)"}],
            },
        })  # fmt: skip
        g["connections"].append({"from": src, "from_port": 0, "to": out, "to_port": 0})
    if alpha:
        out = f"{src}_alpha"
        g["nodes"].append({
            "name": out, "type": "shader", "node_position": {"x": 0, "y": 0},
            "parameters": {}, "seed_int": 0,
            "shader_model": {
                "name": out, "code": "", "global": "", "parameters": [],
                "inputs": [{"name": "input", "type": "rgba", "default": "vec4(0.0)",
                            "function": True, "label": "input"}],
                "instance": "float $(name)_fct(vec2 uv) {\n  return $input(uv).a;\n}\n",
                "outputs": [{"type": "f", "f": "$(name)_fct($uv)"}],
            },
        })  # fmt: skip
        g["connections"].append({"from": src, "from_port": 0, "to": out, "to_port": 0})
    g["connections"].append(
        {"from": out, "from_port": 0, "to": "Material", "to_port": 0}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(g), encoding="utf-8")
    return path


def mm_export(mm: Path, folder: Path, paths: list[Path]) -> None:
    """Export the graphs at ``paths`` into ``folder`` (NAME_albedo.png each).

    One Material Maker run exports them all (MAX_EXPORTS at a time): its
    start-up, not the graphs, takes most of an export. Its output goes to
    ``folder``/mm.log.
    """
    with (folder / "mm.log").open("w", encoding="utf-8", errors="replace") as log:
        for k in range(0, len(paths), MAX_EXPORTS):
            subprocess.run(
                [str(mm), "--export", "-o", folder.as_posix(),
                 *(p.as_posix() for p in paths[k : k + MAX_EXPORTS])],
                cwd=mm.parent, stdout=log, stderr=subprocess.STDOUT, check=False,
                timeout=7200,
            )  # fmt: skip


def export(
    graph: dict,
    into: dict[str, set[str]],
    src: str,
    mm: Path,
    folder: Path,
    magnify: tuple[float, float] | None = None,
    *,
    alpha: bool = False,
) -> Path | None:
    """Export ``src`` alone (``build``) into ``folder``; None on failure."""
    path = build(graph, into, src, folder / f"{src}.ptex", magnify, alpha=alpha)
    png = folder / f"{src}_albedo.png"
    png.unlink(missing_ok=True)
    mm_export(mm, folder, [path])
    return png if png.exists() else None


type Probe = tuple[int, bool]
"""A node to compare, and whether its alpha counts."""


class Search:
    """The two renderers and the translated graph, for one archive."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Read the archive and build (or reuse) the flat translated graph."""
        self.args = args
        self.work = args.work
        for sub in ("truth", "ours", "vs"):
            (self.work / sub).mkdir(parents=True, exist_ok=True)
        self.files = read_archive(args.sbsar)
        self.asm_name = next(n for n in self.files if n.endswith(".sbsasm"))
        self.asm = Asm(self.files[self.asm_name])
        self.nodes = self.asm.nodes()
        xml = member(self.files, ".xml").decode("utf-8")
        ids = {
            int(u): i for u, i in re.findall(r'uid="(\d+)" identifier="([^"]*)"', xml)
        }
        self.outputs = [ids.get(u) for u in self.asm.output_uids]
        # Graph outputs a node can be rendered through: grayscale ones for a
        # grayscale node, color ones for a color node.
        data = self.files[self.asm_name]
        self.slots: dict[bool, list[int]] = {False: [], True: []}
        for k, name in enumerate(self.outputs):
            if name is not None:
                (node,) = struct.unpack_from(
                    "<I", data, self.asm.outputs_at + 8 * k + 4
                )
                self.slots[self.nodes[node].op in COLOR_OPS].append(k)
        if not self.slots[False] or not self.slots[True]:
            sys.exit(f"need a grayscale and a color graph output: {self.outputs}")
        built = self.work / "graph"
        ptex, known = built / f"{args.sbsar.stem}.ptex", built / "aliases.json"
        if not ptex.exists() or not known.exists() or args.rebuild:
            aliases: dict[str, str] = {}
            made = procedural.build(
                self.files, describe(member(self.files, ".xml"))[0],
                built / f"{args.sbsar.stem}_data", size=args.size, flat=True,
                aliases=aliases,
            )  # fmt: skip
            built.mkdir(parents=True, exist_ok=True)
            ptex.write_text(json.dumps(made, indent=1), encoding="utf-8")
            known.write_text(json.dumps(aliases, indent=1), encoding="utf-8")
        self.graph = json.loads(ptex.read_text(encoding="utf-8"))
        # Nodes merged away (identical to another, or a copy of a buffer)
        # are compared by the one computing their image.
        self.aliases: dict[str, str] = json.loads(known.read_text(encoding="utf-8"))
        self.names = {n["name"] for n in self.graph["nodes"]}
        self.memo: dict[Probe, bool | None] = {}
        self.corr: dict[int, float] = {}
        self.pin([int(n) for n in args.pin.split(",") if n])
        self.into = wiring(self.graph)

    def mm_name(self, node: int) -> str:
        """Return the translated graph's node computing Substance node ``node``."""
        return self.aliases.get(f"n{node}", f"n{node}")

    def pin(self, nodes: list[int]) -> None:
        """Replace ``nodes`` in the translated graph by Substance's renders of them.

        For a node whose difference is known and not a translation's (a hash
        of cos that each GPU rounds its own way): what it feeds is then judged
        on its own, from the same input as Substance's.
        """
        truth = self.truths(nodes)
        for n in nodes:
            name, path = self.mm_name(n), truth[n]
            if path is None or name not in self.names:
                sys.exit(f"cannot pin n{n}: no Substance render or no such node")
            color = self.nodes[n].op in COLOR_OPS
            image = f"{name}_substance"
            self.graph["nodes"] = [
                x for x in self.graph["nodes"] if x["name"] != name
            ] + [
                {"name": image, "type": "image", "node_position": {"x": 0, "y": 0},
                 "parameters": {"image": path.resolve().as_posix()}, "seed_int": 0},
                {"name": name, "type": "shader", "node_position": {"x": 0, "y": 0},
                 "parameters": {}, "seed_int": 0,
                 "shader_model": {
                     "name": name, "code": "", "global": "", "parameters": [],
                     "inputs": [{"name": "input", "type": "rgba",
                                 "default": "vec4(0.0)", "function": True,
                                 "label": "input"}],
                     "instance": (
                         f"{'vec4' if color else 'float'} $(name)_fct(vec2 uv) {{\n"
                         f"  return $input(uv){'' if color else '.r'};\n}}\n"
                     ),
                     "outputs": [{"type": "rgba" if color else "f",
                                  "f": "$(name)_fct($uv)"}],
                 }},
            ]  # fmt: skip
            self.graph["connections"] = [
                w for w in self.graph["connections"] if w["to"] != name
            ] + [{"from": image, "from_port": 0, "to": name, "to_port": 0}]

    def truths(self, nodes: list[int]) -> dict[int, Path | None]:
        """Render ``nodes`` in Substance, by retargeting graph outputs to them.

        Each sbsrender run renders as many as the graph has outputs of their
        kind (grayscale, color); renders are kept in WORK/truth.
        """
        out = {n: self.work / "truth" / f"n{n}.png" for n in nodes}
        todo = {
            c: [n for n in dict.fromkeys(nodes) if not out[n].exists()
                and (self.nodes[n].op in COLOR_OPS) == c]
            for c in (False, True)
        }  # fmt: skip
        while todo[False] or todo[True]:
            batch = {}
            for c in (False, True):
                for k in self.slots[c]:
                    if todo[c]:
                        batch[k] = todo[c].pop(0)
            self.render(batch)
        return {n: p if p.exists() else None for n, p in out.items()}

    def render(self, batch: dict[int, int]) -> None:
        """Render node ``batch[k]`` through graph output ``k``, each."""
        data = bytearray(self.files[self.asm_name])
        for k, node in batch.items():
            struct.pack_into("<I", data, self.asm.outputs_at + 8 * k + 4, node)
        files = {**self.files, self.asm_name: bytes(data)}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "out").mkdir()
            sbsar = root / "t.sbsar"
            for n, b in files.items():
                (root / "x" / n).parent.mkdir(parents=True, exist_ok=True)
                (root / "x" / n).write_bytes(b)
            with py7zr.SevenZipFile(sbsar, "w") as z:
                for n in files:
                    z.write(root / "x" / n, n)
            subprocess.run(
                [str(self.args.sbsrender), "render", "--input", str(sbsar),
                 *(a for k in batch for a in ("--input-graph-output", self.outputs[k])),
                 "--set-value", f"$outputsize@{self.args.size},{self.args.size}",
                 "--output-path", str(root / "out"), "--output-name", "{outputNodeName}",
                 "--output-bit-depth", "16", "--no-report"],
                capture_output=True, check=False,
            )  # fmt: skip
            for k, node in batch.items():
                got = root / "out" / f"{self.outputs[k]}.png"
                if got.exists():
                    shutil.move(got, self.work / "truth" / f"n{node}.png")

    def check(self, probes: list[Probe]) -> dict[Probe, bool | None]:
        """Return whether each node matches Substance (None: not compared).

        Its alpha counts too when the probe says (and it is a color node).
        The nodes not compared before are rendered together: one sbsrender
        run per graph-output's worth, one Material Maker run for them all.
        """
        asked = probes  # answered as asked: a grayscale node's alpha is no probe
        probes = [(n, a and self.nodes[n].op in COLOR_OPS) for n, a in probes]
        new = [p for p in dict.fromkeys(probes) if p not in self.memo]
        for p in new:
            if self.mm_name(p[0]) not in self.names:
                self.memo[p] = None
        new = [p for p in new if p not in self.memo]
        truth = self.truths([n for n, _ in new])
        jobs: dict[Probe, tuple[np.ndarray, tuple[float, float] | None]] = {}
        paths, folder = [], self.work / "ours"
        for n, alpha in new:
            color = self.nodes[n].op in COLOR_OPS
            if truth[n] is None:
                print(f"  n{n}: no Substance image", flush=True)
                self.memo[n, alpha] = None
                continue
            t = load(truth[n], 4 if alpha else 3 if color else 1)
            magnify = None if color else magnifier(t)
            jobs[n, alpha] = t, magnify
            stems = [f"n{n}_magnified" if magnify else f"n{n}"]
            stems += [f"n{n}_alpha"] if alpha else []
            for stem in stems:
                path = folder / f"{stem}.ptex"
                (folder / f"{stem}_albedo.png").unlink(missing_ok=True)
                if path not in paths:
                    paths.append(path)
                    build(self.graph, self.into, self.mm_name(n), path,
                          magnify if stem.endswith("_magnified") else None,
                          alpha=stem.endswith("_alpha"))  # fmt: skip
        if paths:
            mm_export(self.args.mm, folder, paths)
        for (n, alpha), (t, magnify) in jobs.items():
            self.memo[n, alpha] = self.compare(n, alpha, t, magnify)
        return {a: self.memo[p] for a, p in zip(asked, probes, strict=True)}

    def compare(
        self,
        node: int,
        alpha: bool,
        t: np.ndarray,
        magnify: tuple[float, float] | None,
    ) -> bool:
        """Compare ``node``'s Substance image ``t`` with Material Maker's."""
        color = self.nodes[node].op in COLOR_OPS
        folder = self.work / "ours"
        png = folder / (f"n{node}_magnified" if magnify else f"n{node}")
        png = png.with_name(png.name + "_albedo.png")
        alpha_png = folder / f"n{node}_alpha_albedo.png"
        if not png.exists() or (alpha and not alpha_png.exists()):
            # Without the image the search cannot tell this node from a match.
            sys.exit(
                f"Material Maker made no image for n{node}: see {folder / 'mm.log'}"
            )
        o = load(png, 3 if color else 1)
        if alpha:
            o = np.concatenate([o, load(alpha_png)[..., None]], axis=-1)
        if magnify:
            t = magnified(t, *magnify)
        result, text, self.corr[node] = (compare_color if color else compare)(
            t, o, strict=self.args.strict
        )
        kind = OPS.get(self.nodes[node].op)
        if magnify:
            kind += " (magnified {1:.0f}x about {0:.4f})".format(*magnify)
        if alpha:
            kind += " (with alpha)"
            side = np.concatenate([t[..., 3], o[..., 3]], axis=1)
            Image.fromarray((side * 255).astype(np.uint8)).save(
                self.work / "vs" / f"n{node}_alpha.png"
            )
            t, o = t[..., :3], o[..., :3]
        print(f"  n{node} {kind}: {text} -> {'ok' if result else 'DIFF'}", flush=True)
        side = np.concatenate([t, o], axis=1)
        Image.fromarray((side * 255).astype(np.uint8)).save(
            self.work / "vs" / f"n{node}.png"
        )
        return result

    def inputs(self, probe: Probe) -> dict[int, bool]:
        """Return the node's inputs, each with whether its alpha counts."""
        node = self.nodes[probe[0]]
        reads = alpha_read(node, probe[1])
        return {m: name in reads for name, m in node.inputs.items()}

    def upstream(self, n: int) -> int:
        """Return how many nodes feed ``n``, directly or not."""
        seen, todo = set(), [n]
        while todo:
            k = todo.pop()
            if k not in seen:
                seen.add(k)
                todo += self.nodes[k].inputs.values()
        return len(seen)

    def path(self, probe: Probe) -> list[Probe]:
        """Return a path up from ``probe``: by the input most nodes feed."""
        out = [probe]
        while ins := self.inputs(out[-1]):
            m = max(ins, key=self.upstream)
            out.append((m, ins[m]))
        return [p for p in out if self.mm_name(p[0]) in self.names]

    def verdict(self, probe: Probe) -> Probe | None:
        """Return ``probe``'s worst differing input; None if they all match."""
        ins = self.inputs(probe)
        got = self.check(list(ins.items()))
        bad = [p for p, ok in got.items() if ok is False]
        return min(bad, key=lambda p: self.corr.get(p[0], 1.0)) if bad else None


def walk(search: Search, probe: Probe) -> Probe:
    """Return the first differing node: from ``probe``, by its worst input."""
    while (worse := search.verdict(probe)) is not None:
        probe = worse
    return probe


def bisect(search: Search, probe: Probe, probes: int) -> Probe:
    """Return a differing node whose inputs all match, by bisecting paths.

    Up the path from a differing node (``Search.path``), ``probes`` nodes
    at a time, rendered together: the first matching one bounds the search
    from above, the last differing one before it from below, until they
    are neighbors. A differing input of the node found starts a new path.
    A node that matches is taken to hide no difference above it (a mask
    could: the node found still differs while all its inputs match).
    """
    while True:
        path = search.path(probe)
        lo, hi = 0, len(path)  # path[lo] differs; path[hi] matches (or ends)
        while hi - lo > 1:
            step = (hi - lo) / (probes + 1)
            picks = sorted({lo + max(1, round(step * (k + 1))) for k in range(probes)})
            picks = [i for i in picks if lo < i < hi]
            got = search.check([path[i] for i in picks])
            for i in picks:
                if got[path[i]] is False:
                    lo = i
                else:  # a match (or no image): the difference enters below
                    hi = i
                    break
        probe = path[lo]
        worse = search.verdict(probe)
        if worse is None:
            return probe
        probe = worse


def main() -> None:
    """Descend from the node the command line names."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sbsar", type=Path)
    parser.add_argument("node", type=int)
    parser.add_argument("--mm", type=Path, required=True, help="material_maker.exe")
    parser.add_argument("--sbsrender", type=Path, default=SBSRENDER)
    parser.add_argument("--work", type=Path, default=Path("descend"))
    parser.add_argument("--size", type=int, default=10, help="log2 pixels")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the flat graph")
    parser.add_argument(
        "--strict", action="store_true", help="content placed elsewhere does not match"
    )
    parser.add_argument(
        "--bisect", type=int, nargs="?", const=3, default=0, metavar="PROBES",
        help="bisect paths, PROBES nodes a round (3; implies --strict)",
    )  # fmt: skip
    parser.add_argument(
        "--pin", default="", metavar="NODES",
        help="nodes (comma-separated) to give Substance's render of (see Search.pin)",
    )  # fmt: skip
    args = parser.parse_args()
    args.strict = args.strict or bool(args.bisect)
    search = Search(args)
    start = (args.node, False)
    first = search.check([start])[start]
    if first is not False:
        print(f"n{args.node} {'matches' if first else 'was not compared'}")
        return
    n, _ = bisect(search, start, args.bisect) if args.bisect else walk(search, start)
    x = search.nodes[n]
    print(f"FIRST DIFF: n{n} {OPS.get(x.op)} inputs {x.inputs} fields {x.fields[:6]}")
    print(f"  side by side: {search.work / 'vs' / f'n{n}.png'}")


if __name__ == "__main__":
    main()
