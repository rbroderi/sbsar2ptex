"""Check chains made iterate buffers (--experimental) against their unrolled translation.

    uv run python -I scripts/check_chains.py --fork PATH/TO/materialmakermod
        [-o DIR] [--godot EXE] [--size 8] [--sbsar FILE] [END ...]

For each chain sbsar2ptex.loops finds in the material (Biomech's reaction-
diffusion, by default), or those whose last nodes are named: translate its
image as today (every period unrolled, "norm") and with the chain one
iterate_buffer_v2 ("exp"), and compare them exported through the material-
maker-mod fork (Godot, headless), each encoded into the albedo as 16-bit
levels (scripts/check_loops.py's encoding), so that they compare exactly.

Also, per chain: the iterate stopped after 1, 2 and 3 periods (its
iteration count edited) against the unrolled image of that period's end;
and both with every Intensity and Radius knob of the chain's library
nodes changed, which the iterate must follow as the unrolled nodes do.
The iterate must be identical to the unrolled translation in every pixel.
Every connection is pinned unbuffered (check_loops.encoded): Material
Maker's automatic buffers (16-bit, where it judges a node costly) fall
differently in the two graphs and change pixels of both.
"""

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from check_loops import GODOT
from check_loops import encoded
from check_loops import levels
from check_loops import same

from sbsar2ptex import library
from sbsar2ptex.archive import member
from sbsar2ptex.archive import read_archive
from sbsar2ptex.loops import Chains
from sbsar2ptex.loops import chains
from sbsar2ptex.procedural import bitmaps
from sbsar2ptex.procedural import library_nodes
from sbsar2ptex.sbsasm import Asm
from sbsar2ptex.translate import graph_inputs
from sbsar2ptex.translate import translate_graph

BIOMECH = Path("C:/Users/richa/Downloads/BiomechZombieCircuits.sbsar")
KNOBS = {"Intensity": 1.6, "Radius": 2.4}
"""Knobs set in the "knob" cases (by the end of their names)."""


def set_knobs(mm: list[dict], knobs: dict[str, float]) -> None:
    """Set every instance knob ending in one of ``knobs`` in ``mm`` (subgraphs too)."""
    for n in mm:
        for key in n.get("parameters", {}):
            for end, v in knobs.items():
                if key.startswith("i_") and key.endswith("_" + end):
                    n["parameters"][key] = v
        if n["type"] == "graph":
            set_knobs(n["nodes"], knobs)


def set_iterations(mm: list[dict], periods: int) -> None:
    """Stop every iterate in ``mm`` after ``periods`` periods."""
    for n in mm:
        if n["type"] == "iterate_buffer_v2":
            phases = n["parameters"]["iterations"].split("*")[0].strip()
            n["parameters"]["iterations"] = f"{phases} * {periods}"
        if n["type"] == "graph":
            set_iterations(n["nodes"], periods)


def main() -> None:
    """Run the chains the command line names, or all."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ends", nargs="*", type=int)
    parser.add_argument("--fork", type=Path, required=True, help="the fork's project")
    parser.add_argument("--godot", type=Path, default=GODOT)
    parser.add_argument("--sbsar", type=Path, default=BIOMECH)
    parser.add_argument("-o", "--out", type=Path, default=Path("check_chains"))
    parser.add_argument("--size", type=int, default=8, help="log2 pixels")
    parser.add_argument("--min", type=int, default=3, help="fewest periods")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    size = args.size
    files = read_archive(args.sbsar)
    raw = member(files, ".sbsasm")
    asm = Asm(raw)
    nodes = asm.nodes()
    inputs = graph_inputs(asm, member(files, ".xml").decode("utf-8"))
    cache = out / "found.pickle"
    if cache.exists():
        found = pickle.loads(cache.read_bytes())  # noqa: S301 (written here)
    else:
        found = library.recognized(
            raw, nodes, library.load(), library.cache_dir() / "matches"
        )
        cache.write_bytes(pickle.dumps(found))
    lib = library_nodes(found, nodes, inputs, size)
    images = bitmaps(asm, out / "bitmaps")
    data = out / "fxmaps"
    todo = [
        c
        for c in chains(nodes, minimum=args.min)
        if not args.ends or c.end in args.ends
    ]
    cases: list[tuple[str, str, str]] = []  # (tag, exp file, norm file)

    def write(tag: str, mm: list, wires: list, outs: dict) -> Path:
        path = out / f"{tag}.ptex"
        graph = encoded(mm, wires, outs["o"], size)
        path.write_text(json.dumps(graph), encoding="utf-8")
        return path

    for chain in todo:
        tag = f"c{chain.end}"
        norm = translate_graph(nodes, {"o": chain.end}, images, size=size,
                               inputs=inputs, library=lib, data_dir=data)  # fmt: skip
        fold = Chains([chain], minimum=args.min)
        exp = translate_graph(nodes, {"o": chain.end}, images, size=size,
                              inputs=inputs, library=lib, fold=fold,
                              data_dir=data)  # fmt: skip
        if not fold.made:
            print(f"{tag}: left unrolled: {fold.left}")
            continue
        write(f"{tag}_norm", *norm)
        write(f"{tag}_exp", *exp)
        cases.append((tag, f"{tag}_exp", f"{tag}_norm"))
        # the knobs changed, in both
        for mode, made in (("norm", norm), ("exp", exp)):
            mm = json.loads(json.dumps(made[0]))
            set_knobs(mm, KNOBS)
            write(f"{tag}_knob_{mode}", mm, made[1], made[2])
        cases.append((f"{tag} knobs", f"{tag}_knob_exp", f"{tag}_knob_norm"))
        # stopped early: against the unrolled image of that period's end
        # (the iterate's periods are the chain's last ones)
        run = next(iter(fold.made.values()))
        if run.stop != len(chain.periods):  # periods after it unrolled
            continue
        for k in (1, 2, 3):
            end = chain.periods[run.start + k - 1][-1]
            mm = json.loads(json.dumps(exp[0]))
            set_iterations(mm, k)
            write(f"{tag}_n{k}_exp", mm, exp[1], exp[2])
            write(f"{tag}_n{k}_norm", *translate_graph(
                nodes, {"o": end}, images, size=size, inputs=inputs, library=lib,
                data_dir=data))  # fmt: skip
            cases.append((f"{tag} N={k}", f"{tag}_n{k}_exp", f"{tag}_n{k}_norm"))

    mm_out = out / "mm"
    mm_out.mkdir(exist_ok=True)
    names = sorted({f for _, *fs in cases for f in fs})
    for f in names:
        (mm_out / f"{f}_albedo.png").unlink(missing_ok=True)
    if names:
        with (out / "fork.log").open("w", encoding="utf-8", errors="replace") as log:
            subprocess.run(
                [str(args.godot), "--headless", "--path", str(args.fork),
                 "--export-material", "--target", "Godot/Godot 4 Standard",
                 "-o", str(mm_out), "--size", str(1 << size),
                 *(str(out / f"{f}.ptex") for f in names)],
                stdout=log, stderr=subprocess.STDOUT, check=False, timeout=3600,
            )  # fmt: skip
    worst = 0
    print(f"{'':16} {'exp = norm':>14}  of {1 << 2 * size} px")
    for tag, e, nm in cases:
        paths = [mm_out / f"{f}_albedo.png" for f in (e, nm)]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            print(f"{tag:16} no export of {', '.join(missing)}")
            worst = max(worst, 1 << 16)
            continue
        a, b = (levels(p) for p in paths)
        worst = max(worst, int(np.abs(a - b).max()))
        print(f"{tag:16} {same(a, b)}")
    if not cases:
        print("nothing compared")
        sys.exit(1)
    print(
        "exp is bit-identical to norm"
        if worst == 0
        else f"exp DIFFERS from norm (max {worst} levels)"
    )
    sys.exit(0 if worst == 0 else 1)


if __name__ == "__main__":
    main()
