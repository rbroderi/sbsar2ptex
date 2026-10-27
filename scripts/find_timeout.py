"""Find the smallest part of a translated graph whose export fails.

    uv run python scripts/find_timeout.py GRAPH.ptex NODE --mm PATH/TO/material_maker.exe
        [--work DIR]

GRAPH is a flat translated graph (``sbsar2ptex --procedural --flat``; the one
``descend.py`` builds is in its work folder, under graph/), NODE a node of it
whose export fails, such as one Windows' GPU timeout (TDR) kills. A node
"fails" when Material Maker, exporting it with only the nodes feeding it,
makes no image. Adding nodes is taken to never rescue an export, so along a
path from NODE toward the graph's inputs the failing nodes come first: the
search follows the input with the most nodes behind it, binary searches that
path for the last failing node, then checks that node's other inputs, and
stops at a failing node whose inputs all export. Either that node's own
shader is too heavy, or the work it gathers is.

Each export's graph and Material Maker log are kept in WORK/<node>/.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from descend import cone
from descend import export
from descend import wiring


class Bisect:
    """Exports of a graph's nodes, each remembered."""

    def __init__(self, graph: dict, mm: Path, work: Path) -> None:
        """Keep the graph and where to export."""
        self.graph = graph
        self.into = wiring(graph)
        self.by_name = {n["name"]: n for n in graph["nodes"]}
        self.mm = mm
        self.work = work
        self.memo: dict[str, bool] = {}

    def inputs(self, name: str) -> list[str]:
        """Return ``name``'s inputs that are nodes of their own."""
        return sorted(
            m
            for m in self.into.get(name, ())
            if self.by_name.get(m, {}).get("type") != "remote"
        )

    def fails(self, name: str) -> bool:
        """Return whether exporting ``name`` (and what feeds it) fails."""
        if name not in self.memo:
            t0 = time.time()
            png = export(self.graph, self.into, name, self.mm, self.work / name)
            log = (self.work / name / "mm.log").read_text(
                encoding="utf-8", errors="replace"
            )
            lost = "VkResult error -4" in log or "device lost" in log.lower()
            self.memo[name] = png is None
            verdict = (
                "FAILS" + (" (device lost)" if lost else "")
                if png is None
                else "exports"
            )
            size = len(cone(self.into, name))
            print(
                f"  {name} ({size} nodes): {verdict} in {time.time() - t0:.0f} s",
                flush=True,
            )
        return self.memo[name]

    def path(self, name: str) -> list[str]:
        """Return a path from ``name`` that always takes the largest input."""
        out = [name]
        while ins := self.inputs(out[-1]):
            out.append(max(ins, key=lambda m: len(cone(self.into, m))))
        return out

    def describe(self, name: str) -> str:
        """Return a short account of a node's shader."""
        n = self.by_name[name]
        sm = n.get("shader_model", {})
        code = sm.get("global", "") + sm.get("instance", "")
        return (
            f"{name}: type {n['type']}, {len(code)} chars of GLSL, "
            f"{code.count('for (')} loops, inputs {self.inputs(name)}, "
            f"{len(cone(self.into, name))} nodes in all"
        )


def main() -> None:
    """Search from the node the command line names."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("graph", type=Path)
    parser.add_argument("node")
    parser.add_argument("--mm", type=Path, required=True, help="material_maker.exe")
    parser.add_argument("--work", type=Path, default=Path("find_timeout"))
    args = parser.parse_args()
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    node = args.node if args.node.startswith("n") else f"n{args.node}"
    b = Bisect(graph, args.mm, args.work)
    if not b.fails(node):
        sys.exit(f"{node} exports: nothing to find")
    while True:
        path = b.path(node)
        print(f"path from {node}: {len(path)} nodes", flush=True)
        lo, hi = 0, len(path)  # path[lo] fails; path[hi:] all export
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if b.fails(path[mid]):
                lo = mid
            else:
                hi = mid
        last = path[lo]
        others = [m for m in b.inputs(last) if m not in path]
        failing = next((m for m in others if b.fails(m)), None)
        if failing is None:
            print(f"SMALLEST FAILING: {b.describe(last)}")
            for m in b.inputs(last):
                print(f"  input {b.describe(m)}")
            return
        node = failing


if __name__ == "__main__":
    main()
