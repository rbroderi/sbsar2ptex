"""Walk back from a node that differs from Substance to the first that does.

    uv run python scripts/descend.py SBSAR NODE --mm PATH/TO/material_maker.exe
        [--sbsrender PATH] [--work DIR] [--size 10] [--output height]
        [--color-output basecolor] [--strict]

NODE is a compiled node index (``sbsasm-dump`` lists them; a graph output's
node is a good start). Each step renders a node twice: in Substance, by
retargeting a graph output to it (``--output`` for a grayscale node,
``--color-output`` for a color one) and running sbsrender; and in Material Maker, by exporting the translated graph (flat,
``--procedural --flat``) pruned to the node and the nodes feeding it. A node
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
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import py7zr
from PIL import Image

from sbsar2ptex.archive import member
from sbsar2ptex.archive import read_archive
from sbsar2ptex.sbsasm import COLOR_OPS
from sbsar2ptex.sbsasm import OPS
from sbsar2ptex.sbsasm import Asm
from sbsar2ptex.sbsasm import Node

SBSRENDER = Path("C:/Program Files/Adobe/Adobe Substance 3D Designer/sbsrender.exe")
SIDE = 512
"""Images are compared at this size."""
NARROW = 16 / 255
"""A grayscale image spanning less than this is compared magnified."""


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
    """Export ``src`` alone (with what feeds it) as the albedo; None on failure.

    ``magnify`` (center, gain) exports ``magnified`` of it instead; ``alpha``
    its alpha, as gray (the albedo's own alpha is exported opaque).
    Material Maker's output goes to ``folder``/mm.log.
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
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{src}.ptex"
    path.write_text(json.dumps(g), encoding="utf-8")
    png = folder / f"{src}_albedo.png"
    png.unlink(missing_ok=True)
    with (folder / "mm.log").open("w", encoding="utf-8", errors="replace") as log:
        subprocess.run(
            [str(mm), "--export", "-o", folder.as_posix(), path.as_posix()],
            cwd=mm.parent, stdout=log, stderr=subprocess.STDOUT, check=False, timeout=7200,
        )  # fmt: skip
    return png if png.exists() else None


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
        for output in (args.output, args.color_output):
            if output not in self.outputs:
                sys.exit(f"no graph output {output!r}: {self.outputs}")
        built = self.work / "graph"
        ptex = next(built.glob("*.ptex"), None)
        if ptex is None or args.rebuild:
            subprocess.run(
                [sys.executable, "-m", "sbsar2ptex.convert", str(args.sbsar), "-o", str(built),
                 "--size", str(args.size), "--procedural", "--flat"],
                check=True,
            )  # fmt: skip
            ptex = next(built.glob("*.ptex"))
        self.graph = json.loads(ptex.read_text(encoding="utf-8"))
        self.names = {n["name"] for n in self.graph["nodes"]}
        self.into = wiring(self.graph)
        self.memo: dict[tuple[int, bool], bool | None] = {}
        self.corr: dict[int, float] = {}

    def truth(self, node: int, output: str) -> Path | None:
        """Render ``node`` in Substance: retarget the graph ``output`` to it."""
        out = self.work / "truth" / f"n{node}.png"
        if out.exists():
            return out
        data = bytearray(self.files[self.asm_name])
        k = self.outputs.index(output)
        struct.pack_into("<I", data, self.asm.outputs_at + 8 * k + 4, node)
        files = {**self.files, self.asm_name: bytes(data)}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sbsar = root / "t.sbsar"
            for n, b in files.items():
                (root / "x" / n).parent.mkdir(parents=True, exist_ok=True)
                (root / "x" / n).write_bytes(b)
            with py7zr.SevenZipFile(sbsar, "w") as z:
                for n in files:
                    z.write(root / "x" / n, n)
            subprocess.run(
                [str(self.args.sbsrender), "render", "--input", str(sbsar),
                 "--input-graph-output", output,
                 "--set-value", f"$outputsize@{self.args.size},{self.args.size}",
                 "--output-path", str(out.parent), "--output-name", out.stem,
                 "--output-bit-depth", "16", "--no-report"],
                capture_output=True, check=False,
            )  # fmt: skip
        return out if out.exists() else None

    def ours(
        self,
        node: int,
        magnify: tuple[float, float] | None = None,
        *,
        alpha: bool = False,
    ) -> Path | None:
        """Export ``node`` alone (with what feeds it) from Material Maker."""
        tail = "_magnified" if magnify else "_alpha" if alpha else ""
        folder = self.work / "ours" / f"n{node}{tail}"
        png = export(
            self.graph,
            self.into,
            f"n{node}",
            self.args.mm,
            folder,
            magnify,
            alpha=alpha,
        )
        if png is None:
            # Without the image the search cannot tell this node from a match.
            sys.exit(
                f"Material Maker made no image for n{node}: see {folder / 'mm.log'}"
            )
        return png

    def matches(self, node: int, *, alpha: bool = False) -> bool | None:
        """Return whether ``node`` matches Substance; None if not compared.

        ``alpha``: a color node's alpha counts too (something reads it).
        """
        color = self.nodes[node].op in COLOR_OPS
        alpha = alpha and color
        if (node, alpha) in self.memo:
            return self.memo[node, alpha]
        result = None
        if f"n{node}" in self.names:
            output = self.args.color_output if color else self.args.output
            truth = self.truth(node, output)
            channels = 4 if alpha else 3 if color else 1
            t = load(truth, channels) if truth else None
            magnify = None if color or t is None else magnifier(t)
            ours = self.ours(node, magnify)
            if ours and alpha:
                ours_a = self.ours(node, alpha=True)
            if truth and ours:
                o = load(ours, 3 if color else 1)
                if alpha:
                    o = np.concatenate([o, load(ours_a)[..., None]], axis=-1)
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
                    Image.fromarray(
                        (np.concatenate([t[..., 3], o[..., 3]], axis=1) * 255).astype(
                            np.uint8
                        )
                    ).save(self.work / "vs" / f"n{node}_alpha.png")
                    t, o = t[..., :3], o[..., :3]
                print(
                    f"  n{node} {kind}: {text} -> {'ok' if result else 'DIFF'}",
                    flush=True,
                )
                side = np.concatenate([t, o], axis=1)
                Image.fromarray((side * 255).astype(np.uint8)).save(
                    self.work / "vs" / f"n{node}.png"
                )
            else:
                print(
                    f"  n{node}: no {'Substance' if not truth else 'Material Maker'} image",
                    flush=True,
                )
        self.memo[node, alpha] = result
        return result


def main() -> None:
    """Descend from the node the command line names."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sbsar", type=Path)
    parser.add_argument("node", type=int)
    parser.add_argument("--mm", type=Path, required=True, help="material_maker.exe")
    parser.add_argument("--sbsrender", type=Path, default=SBSRENDER)
    parser.add_argument("--work", type=Path, default=Path("descend"))
    parser.add_argument("--size", type=int, default=10, help="log2 pixels")
    parser.add_argument("--output", default="height", help="a grayscale graph output")
    parser.add_argument(
        "--color-output", default="basecolor", help="a color graph output"
    )
    parser.add_argument("--rebuild", action="store_true", help="rebuild the flat graph")
    parser.add_argument(
        "--strict", action="store_true", help="content placed elsewhere does not match"
    )
    args = parser.parse_args()
    search = Search(args)
    n, alpha = args.node, False
    if search.matches(n):
        print(f"n{n} matches")
        return
    while True:
        node = search.nodes[n]
        reads = alpha_read(node, alpha)
        flagged = {m: name in reads for name, m in node.inputs.items()}
        bad = [m for m, a in flagged.items() if search.matches(m, alpha=a) is False]
        if not bad:
            x = search.nodes[n]
            print(
                f"FIRST DIFF: n{n} {OPS.get(x.op)} inputs {x.inputs} fields {x.fields[:6]}"
            )
            print(f"  side by side: {search.work / 'vs' / f'n{n}.png'}")
            return
        n = min(bad, key=lambda m: search.corr.get(m, 1.0))
        alpha = flagged[n]


if __name__ == "__main__":
    main()
