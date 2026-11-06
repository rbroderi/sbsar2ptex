"""Inventory the library graphs in Designer's bundled packages.

    python scan_library.py PACKAGES_DIR OUT.tsv

One row per graph a package defines (not hidden from the library):
package, graph identifier, label, category, graphtype, image inputs
(identifier:type, 1 color 2 gray 0 either), value inputs count, outputs.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

IMAGE_TYPES = {1: "c", 2: "g", 3: "cg"}  # paraminput type bitmasks seen: 1 color, 2 gray


def v(el, path, default=""):
    e = el.find(path)
    return e.get("v", default) if e is not None else default


def main() -> None:
    pkg_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = []
    for f in sorted(pkg_dir.glob("*.sbs")):
        try:
            root = ET.parse(f).getroot()
        except ET.ParseError as e:
            print("parse error", f.name, e)
            continue
        for g in root.iter("graph"):
            attrs = g.find("attributes")
            hidden = attrs is not None and v(attrs, "hideInLibrary") == "1"
            cat = v(attrs, "category") if attrs is not None else ""
            label = v(attrs, "label") if attrs is not None else ""
            tags = v(attrs, "tags") if attrs is not None else ""
            cat = f"{cat}/{tags.replace(',', ';').split(';')[0].strip()}" if tags else cat
            ins, values = [], 0
            for pi in g.iterfind("paraminputs/paraminput"):
                t = int(v(pi, "type", "0") or 0)
                ident = v(pi, "identifier")
                if t in (1, 2):
                    ins.append(f"{ident}:{IMAGE_TYPES[t]}")
                elif t == 3:
                    ins.append(f"{ident}:cg")
                else:
                    values += 1
            outs = []
            for go in g.iterfind("graphOutputs/graphoutput"):
                ch = v(go, "channels", "1") or "1"
                if ch not in ("1", "2"):
                    continue  # a value output, not an image
                outs.append(v(go, "identifier") + (":g" if ch == "2" else ":c"))
            if not outs:
                continue  # function graphs and other non-texture graphs
            rows.append(
                (f.name, v(g, "identifier"), label, cat, v(g, "graphtype"),
                 "1" if hidden else "0", " ".join(ins), str(values), " ".join(outs))
            )
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("package\tgraph\tlabel\tcategory\tgraphtype\thidden\timage_inputs\tvalue_inputs\toutputs\n")
        for r in rows:
            fh.write("\t".join(x.replace("\t", " ") for x in r) + "\n")
    print(len(rows), "graphs,", sum(r[5] == "0" for r in rows), "visible")


if __name__ == "__main__":
    main()
