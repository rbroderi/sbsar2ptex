"""Walk back from a node that differs from Substance to the first that does.

    uv run python scripts/descend.py SBSAR NODE --mm PATH/TO/material_maker.exe
        [--sbsrender PATH] [--work DIR] [--size 10] [--output height]
        [--color-output basecolor]

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
its red, green and blue channels does.

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

SBSRENDER = Path("C:/Program Files/Adobe/Adobe Substance 3D Designer/sbsrender.exe")
SIDE = 512
"""Images are compared at this size."""


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


def compare(t: np.ndarray, o: np.ndarray) -> tuple[bool, str, float]:
    """Return whether two images match, a summary, and their correlation."""
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
        same or moved,
        text + (" (placed elsewhere)" if moved and not same else ""),
        c,
    )


def compare_color(t: np.ndarray, o: np.ndarray) -> tuple[bool, str, float]:
    """Return ``compare`` over the red, green and blue channels: the worst one."""
    per = [compare(t[..., c], o[..., c]) for c in range(3)]
    k = min(range(3), key=lambda c: (per[c][0], per[c][2]))
    _, text, c = per[k]
    return all(p[0] for p in per), f"{'rgb'[k]}: {text}", c


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


def export(
    graph: dict, into: dict[str, set[str]], src: str, mm: Path, folder: Path
) -> Path | None:
    """Export ``src`` alone (with what feeds it) as the albedo; None on failure.

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
    g["connections"].append(
        {"from": src, "from_port": 0, "to": "Material", "to_port": 0}
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
        self.memo: dict[int, bool | None] = {}
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

    def ours(self, node: int) -> Path | None:
        """Export ``node`` alone (with what feeds it) from Material Maker."""
        folder = self.work / "ours" / f"n{node}"
        png = export(self.graph, self.into, f"n{node}", self.args.mm, folder)
        if png is None:
            # Without the image the search cannot tell this node from a match.
            sys.exit(
                f"Material Maker made no image for n{node}: see {folder / 'mm.log'}"
            )
        return png

    def matches(self, node: int) -> bool | None:
        """Return whether ``node`` matches Substance; None if not compared."""
        if node in self.memo:
            return self.memo[node]
        result = None
        if f"n{node}" in self.names:
            color = self.nodes[node].op in COLOR_OPS
            output = self.args.color_output if color else self.args.output
            truth, ours = self.truth(node, output), self.ours(node)
            if truth and ours:
                t, o = load(truth, 3 if color else 1), load(ours, 3 if color else 1)
                result, text, self.corr[node] = (compare_color if color else compare)(
                    t, o
                )
                kind = OPS.get(self.nodes[node].op)
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
        self.memo[node] = result
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
    args = parser.parse_args()
    search = Search(args)
    n = args.node
    if search.matches(n):
        print(f"n{n} matches")
        return
    while True:
        bad = [m for m in search.nodes[n].inputs.values() if search.matches(m) is False]
        if not bad:
            x = search.nodes[n]
            print(
                f"FIRST DIFF: n{n} {OPS.get(x.op)} inputs {x.inputs} fields {x.fields[:6]}"
            )
            print(f"  side by side: {search.work / 'vs' / f'n{n}.png'}")
            return
        n = min(bad, key=lambda m: search.corr.get(m, 1.0))


if __name__ == "__main__":
    main()
