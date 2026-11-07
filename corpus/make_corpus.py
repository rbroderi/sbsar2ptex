"""Build the node corpus: one small graph per Substance node, cooked and rendered.

    python make_corpus.py ROOT [--only atomic|library] [--match SUBSTR]
                          [--steps gen,cook,render] [--depth 8|16] [--jobs N]

ROOT holds patterns/ (make_patterns.py). This writes ROOT/sbs/<node>.sbs,
cooks each to ROOT/sbsar/<node>.sbsar with sbscooker, renders every output
to ROOT/truth/<node>/<output>.png with sbsrender, and writes
ROOT/inventory.tsv (and library inventory from ROOT/library_raw.tsv, see
scan_library.py).

Each graph: one Bitmap node per distinct pattern it uses (an embedded
resource, 16-bit, 256 x 256), the node under test with every image input
wired to a pattern, and one graph output per node output. Atomic nodes get
their main modes as variants; library nodes run at default parameters.

Run it with the sbsar2ptex venv's python (it uses sbsar2ptex.probe's XML
helpers for function graphs, FX-Maps, gradients and curves).
"""

import argparse
import csv
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sbsar2ptex.probe import FLOAT2, FLOAT4, curve_array, fn, function_graph, fx
from sbsar2ptex.probe import fxmap_graph, get, gradient_array

DESIGNER = Path("C:/Program Files/Adobe/Adobe Substance 3D Designer")
COOKER = DESIGNER / "sbscooker.exe"
RENDER = DESIGNER / "sbsrender.exe"
COLOR, GRAY = 1, 2
LOG2 = 8
"""log2 of the pattern and output size (--size)."""

PATTERNS = {
    # name: (file, image kind)
    "coded_rgba": ("coded_rgba.png", COLOR),
    "noise_rgba": ("noise_rgba.png", COLOR),
    "coded_gray16": ("coded_gray16.png", GRAY),
    "noise_gray": ("noise_gray.png", GRAY),
    "ramp_h": ("ramp_h.png", GRAY),
    "bumps16": ("bumps16.png", GRAY),
    "mask": ("mask.png", GRAY),
}
MAIN = {"coded": {COLOR: "coded_rgba", GRAY: "coded_gray16"},
        "noise": {COLOR: "noise_rgba", GRAY: "noise_gray"}}  # fmt: skip
OTHER = {"coded": "noise", "noise": "coded"}
ROLE_INPUTS = {"opacity": "mask", "mask": "mask", "inputgradient": "bumps16",
               "inputintensity": "bumps16"}  # fmt: skip
"""Atomic inputs that always take a role pattern, whatever the variant."""

F1, F2, F4, I1, I32, B = ("constantValueFloat1", "constantValueFloat2",
                          "constantValueFloat4", "constantValueInt1",
                          "constantValueInt32", "constantValueBool")  # fmt: skip


def p(name: str, kind: str, value: str) -> tuple:
    return (name, kind, value)


# ---------------------------------------------------------------- atomic nodes


def pixel_processor_variants(kind: int) -> list[tuple[str, tuple]]:
    pos = get("$pos", FLOAT2)

    def at(dx: float, dy: float) -> tuple:
        shift = fn("const_float2", FLOAT2, data=f'<constantValueFloat2 v="{dx} {dy}"/>')
        return fn("add", FLOAT2, a=pos, b=shift)

    if kind == COLOR:
        def s(where, i):
            return fn("samplecol", FLOAT4, pos=where, data=f'<constantValueInt2 v="{i} 0"/>')
        half = fn("const_float1", data='<constantValueFloat1 v="0.5"/>')
        mix = fn("lerp", FLOAT4, a=s(pos, 0), b=s(at(0.05, 0.1), 1), x=half)
    else:
        def s(where, i):
            return fn("samplelum", pos=where, data=f'<constantValueInt2 v="{i} 0"/>')
        half = fn("const_float1", data='<constantValueFloat1 v="0.5"/>')
        mix = fn("lerp", a=s(pos, 0), b=s(at(0.05, 0.1), 1), x=half)
    return [
        ("identity", (p("perpixel", "dynamicValue", function_graph(s(pos, 0))),)),
        ("offset_mix", (p("perpixel", "dynamicValue", function_graph(mix)),)),
    ]


def fxmap_variants() -> list[tuple[str, tuple]]:
    full = fx("quadrant", ((("patterntype", I32, "1")) ,))
    quad = fx("quadrant", (), dict.fromkeys(range(4), fx(
        "quadrant", (("patterntype", I32, "1"), ("patternrotation", F1, "0.1")))))
    return [("image", (), fxmap_graph(full)), ("quad_rot", (), fxmap_graph(quad))]


ROT30 = "0.866025 -0.5 0.5 0.866025"
BLEND_MODES = ["copy", "add", "subtract", "multiply", "addsub", "max", "min",
               "switch", "divide", "overlay", "screen", "softlight"]  # fmt: skip


def atomic_specs() -> list[dict]:
    """Each: name, filter, kind (image kind of main inputs), inputs
    [(identifier, kind)], output kind, variants [(tag, params, extra)].
    Variant 'default' (no params) comes first."""
    specs = []

    def add(filt, kinds, inputs, out=None, variants=(), label=None):
        for kind in kinds:
            ins = [(n, GRAY if n in ROLE_INPUTS or k == "g" else (COLOR if k == "c" else kind))
                   for n, k in inputs]
            okind = out if out is not None else kind
            vs = [("default", (), "")]
            for v in (variants(kind) if callable(variants) else variants):
                vs.append(v if len(v) == 3 else (*v, ""))
            specs.append({"filter": filt, "kind": kind, "inputs": ins, "out": okind,
                          "variants": vs, "label": label or filt})

    both = (GRAY, COLOR)
    add("bitmap", both, [])  # the pattern itself, through the Bitmap node
    add("blend", both, [("source", ""), ("destination", ""), ("opacity", "g")],
        variants=lambda k: [(f"m{i:02d}_{m}", (p("blendingmode", I1, str(i)),))
                            for i, m in enumerate(BLEND_MODES)
                            if i and (k == COLOR or i in (1, 2, 3, 8, 9))]
        + [("opacity037", (p("opacitymult", F1, "0.37"),))])
    add("blur", both, [("input1", "")],
        variants=[("i2_5", (p("intensity", F1, "2.5"),)), ("i40", (p("intensity", F1, "40"),))])
    add("shuffle", (COLOR,), [("input1", "c"), ("input2", "c")], out=COLOR,
        variants=[("swap", (p("channelred", I1, "6"), p("channelgreen", I1, "0"),
                            p("channelblue", I1, "5"), p("channelalpha", I1, "1")))])
    s_curve = curve_array([((0, 0), (0, 0), (0.25, 0.0)), ((0.5, 0.5), (0.4, 0.3), (0.6, 0.7)),
                           ((1, 1), (0.75, 1.0), (1, 1))])
    add("curve", both, [("input1", "")], variants=[("s_curve", (), f"<paramsArrays>{s_curve}</paramsArrays>")])
    add("dirmotionblur", both, [("input1", "")],
        variants=[("a0125_i20", (p("intensity", F1, "20"), p("mblurangle", F1, "0.125")))])
    add("directionalwarp", both, [("input1", ""), ("inputintensity", "g")],
        variants=[("a03_i20", (p("intensity", F1, "20"), p("warpangle", F1, "0.3")))])
    add("distance", both, [("mask", "g"), ("source", "")],
        variants=[("d50", (p("distance", F1, "50"),)),
                  ("combine", (p("combinedistance", B, "1"),))])
    add("dyngradient", both, [("input1", "g"), ("input2", "")],
        variants=[("vertical", (p("orientation", I1, "1"),))])
    add("emboss", both, [("input1", ""), ("inputgradient", "g")],
        variants=[("i20_a03", (p("intensity", F1, "20"), p("lightangle", F1, "0.3")))])
    grad = gradient_array([(0.0, (0.1, 0.0, 0.3, 1.0)), (0.4, (0.9, 0.2, 0.1, 1.0)),
                           (0.75, (0.2, 0.8, 0.3, 0.5)), (1.0, (1.0, 1.0, 0.6, 1.0))])
    add("gradient", (GRAY,), [("input1", "g")], out=COLOR,
        variants=[("keys4", (), f"<paramsArrays>{grad}</paramsArrays>")])
    add("grayscaleconversion", (COLOR,), [("input1", "c")], out=GRAY,
        variants=[("weights", (p("channelsweights", F4, "0.2 0.3 0.5 0"),))])
    add("hsl", (COLOR,), [("input1", "c")], out=COLOR,
        variants=[("hue02", (p("hue", F1, "0.2"),)), ("sat08", (p("saturation", F1, "0.8"),)),
                  ("lum065", (p("luminosity", F1, "0.65"),))])

    def levels(k):
        def v(x):
            return (F4, " ".join([x] * 4))  # Float4 in gray too (Float1 fails to cook)
        return [("in02_08", (p("levelinlow", *v("0.2")), p("levelinhigh", *v("0.8")))),
                ("mid03", (p("levelinmid", *v("0.3")),)),
                ("out01_09", (p("leveloutlow", *v("0.1")), p("levelouthigh", *v("0.9"))))]

    add("levels", both, [("input1", "")], variants=levels)
    add("normal", (GRAY,), [("input1", "g")], out=COLOR,
        variants=[("i3", (p("intensity", F1, "3"),)), ("inv_y", (p("inversedy", B, "1"),))])
    add("passthrough", both, [("input1", "")])
    add("sharpen", both, [("input1", "")], variants=[("i2", (p("intensity", F1, "2"),))])
    add("transformation", both, [("input1", "")],
        variants=[("offset", (p("offset", F2, "0.25 0.1"),)),
                  ("rot30", (p("matrix22", F4, ROT30),)),
                  ("scale2_notile", (p("matrix22", F4, "2 0 0 2"), p("tiling", I1, "0"))),
                  ("halfpx_nearest", (p("offset", F2, "0.001953125 0"), p("filtering", I1, "1")))])
    add("uniform", both, [], variants=[("color", (p("outputcolor", F4, "0.3 0.4 0.5 1"),))])
    add("warp", both, [("input1", ""), ("inputgradient", "g")],
        variants=[("i10", (p("intensity", F1, "10"),)), ("i50", (p("intensity", F1, "50"),))])
    add("pixelprocessor", both, [("input", ""), ("input:1", "")], variants=pixel_processor_variants)
    add("fxmaps", both, [("inputpattern", "")], variants=lambda k: fxmap_variants())
    add("text", both, [])
    add("svg", both, [])
    add("motionblur", both, [("input1", "")])
    return specs


# ------------------------------------------------------------------ XML writer


def param_xml(params) -> str:
    out = []
    for name, vkind, value in params:
        val = f"<{vkind}>{value}</{vkind}>" if vkind == "dynamicValue" else f'<{vkind} v="{value}"/>'
        out.append(f'<parameter><name v="{name}"/><relativeTo v="0"/><paramValue>{val}</paramValue></parameter>')
    return "".join(out)


COLOR_USAGES = ("baseColor", "emissive")
GRAY_USAGES = ("roughness", "metallic", "ambientOcclusion", "opacity", "height")
"""Channel usages given to graph outputs, in order, so that sbsar2ptex (which
maps outputs to Material Maker's material slots by usage) translates them.
Outputs beyond these get no usage."""


def assign_usages(kinds: list[int]) -> list[str]:
    """Return each output's usage ('' for none): color outputs take
    COLOR_USAGES in turn, grayscale ones GRAY_USAGES."""
    c, g = iter(COLOR_USAGES), iter(GRAY_USAGES)
    return [next(c, "") if k == COLOR else next(g, "") for k in kinds]


def package(node_xml: str, used: list[str], outputs: list[tuple[str, int, int, int]],
            dependencies: str = "") -> str:
    """A package: bitmap nodes for ``used`` patterns (uids 100+10i, output
    +1), ``node_xml`` (the node(s) under test), and graph outputs
    (identifier, kind, source node uid, source output uid)."""
    bitmaps = "".join(
        f'<compNode><uid v="{100 + 10 * i}"/><compOutputs><compOutput><uid v="{101 + 10 * i}"/>'
        f'<comptype v="{PATTERNS[n][1]}"/></compOutput></compOutputs><compImplementation>'
        '<compFilter><filter v="bitmap"/><parameters>'
        + param_xml((("bitmapresourcepath", "constantValueString", f"pkg:///Resources/{n}?dependency=1"),
                     ("colorswitch", B, "1" if PATTERNS[n][1] == COLOR else "0"),
                     ("format", I32, "1"), ("outputsize", "constantValueInt2", f"{LOG2} {LOG2}")))
        + "</parameters></compFilter></compImplementation></compNode>"
        for i, n in enumerate(used)
    )
    usages = assign_usages([kind for _, kind, _, _ in outputs])
    graph_outputs = "".join(
        f'<graphoutput><identifier v="{ident}"/><uid v="{20 + i}"/>'
        + (f'<usages><usage><components v="RGBA"/><name v="{u}"/></usage></usages>' if u else "")
        + f'<channels v="{kind}"/></graphoutput>'
        for i, ((ident, kind, _, _), u) in enumerate(zip(outputs, usages))
    )
    bridges = "".join(
        f'<compNode><uid v="{300 + i}"/><connections><connection><identifier v="inputNodeOutput"/>'
        f'<connRef v="{src}"/><connRefOutput v="{srco}"/></connection></connections>'
        f'<compImplementation><compOutputBridge><output v="{20 + i}"/></compOutputBridge>'
        "</compImplementation></compNode>"
        for i, (_, _, src, srco) in enumerate(outputs)
    )
    roots = "".join(f'<rootOutput><output v="{20 + i}"/><format v="0"/><usertag v=""/></rootOutput>'
                    for i in range(len(outputs)))
    resources = "".join(
        f'<resource><identifier v="{n}"/><uid v="{3 + i}"/><type v="bitmap"/><format v="png"/>'
        f'<filepath v="../patterns/{PATTERNS[n][0]}"/></resource>'
        for i, n in enumerate(PATTERNS)
        if n in used
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?><package><identifier v="corpus"/>'
        '<formatVersion v="1.1.0.202401"/><updaterVersion v="1.1.0.202401"/>'
        '<fileUID v="{00000000-0000-0000-0000-000000000002}"/><versionUID v="0"/>'
        '<dependencies><dependency><filename v="?himself"/><uid v="1"/><type v="package"/>'
        f'<fileUID v="0"/><versionUID v="0"/></dependency>{dependencies}</dependencies>'
        '<content><graph><identifier v="corpus"/><uid v="2"/><graphtype v="filter"/>'
        f'<paraminputs/><graphOutputs>{graph_outputs}</graphOutputs><compNodes>'
        f"{bitmaps}{node_xml}{bridges}</compNodes><baseParameters/>"
        f'<options><option><name v="defaultParentSize"/><value v="{LOG2}x{LOG2}"/></option></options>'
        f"<root><rootOutputs>{roots}</rootOutputs></root></graph>"
        '<group><uid v="50"/><identifier v="Resources"/><content>'
        f"{resources}</content></group></content></package>"
    )


def wire(inputs: list[tuple[str, int]], variant: str, used: list[str]) -> str:
    """Connections for ``inputs`` [(identifier, kind)]: the first input of a
    kind takes the variant's pattern, later ones of that kind the other
    variant's, role inputs their role pattern."""
    conns, seen = [], set()
    for ident, kind in inputs:
        base = ident.split(":")[0]
        if base in ROLE_INPUTS:
            pat = ROLE_INPUTS[base]
        elif kind in seen:
            pat = MAIN[OTHER[variant]][kind]
        else:
            pat = MAIN[variant][kind]
            seen.add(kind)
        if pat not in used:
            used.append(pat)
        i = used.index(pat)
        conns.append(f'<connection><identifier v="{ident}"/><connRef v="{100 + 10 * i}"/>'
                     f'<connRefOutput v="{101 + 10 * i}"/></connection>')
    return "".join(conns)


def atomic_cases() -> list[dict]:
    cases = []
    for s in atomic_specs():
        k = "c" if s["kind"] == COLOR else "g"
        for vi, (tag, params, extra) in enumerate(s["variants"]):
            for pat in (("coded", "noise") if vi == 0 else ("coded",)):
                name = f"a_{s['filter']}_{k}_{tag}_{pat}"
                used: list[str] = []
                if s["filter"] == "bitmap":
                    used.append(MAIN[pat][s["kind"]])
                    xml = package("", used, [("output", s["out"], 100, 101)])
                else:
                    conns = wire(s["inputs"], pat, used)
                    node = (
                        f'<compNode><uid v="200"/><connections>{conns}</connections>'
                        f'<compOutputs><compOutput><uid v="201"/><comptype v="{s["out"]}"/></compOutput>'
                        f'</compOutputs><compImplementation><compFilter><filter v="{s["filter"]}"/>'
                        f"<parameters>{param_xml(params)}</parameters>{extra}"
                        "</compFilter></compImplementation></compNode>"
                    )
                    xml = package(node, used, [("output", s["out"], 200, 201)])
                cases.append({
                    "node": name, "category": "Atomic", "package": f"atomic:{s['filter']}",
                    "variant": f"{k}/{tag}/{pat}",
                    "inputs": " ".join(f"{i}:{'c' if t == COLOR else 'g'}" for i, t in s["inputs"]),
                    "outputs": f"output:{'c' if s['out'] == COLOR else 'g'}"
                               f"[{assign_usages([s['out']])[0]}]", "xml": xml,
                })
    return cases


def library_cases(root: Path) -> list[dict]:
    rows = list(csv.DictReader((root / "library_raw.tsv").open(encoding="utf-8"), delimiter="\t"))
    rows = [r for r in rows if r["hidden"] == "0"]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["graph"]] = counts.get(r["graph"], 0) + 1
    cases = []
    for r in rows:
        gid = r["graph"]
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", gid)
        if counts[gid] > 1:
            safe += "__" + Path(r["package"]).stem
        ins = [tuple(x.rsplit(":", 1)) for x in r["image_inputs"].split()]
        outs = [tuple(x.rsplit(":", 1)) for x in r["outputs"].split()]
        used: list[str] = []
        conns = []
        for ident, k in ins:
            kind = GRAY if k == "g" else COLOR
            low = ident.lower()
            pat = "mask" if "mask" in low and kind == GRAY else MAIN["coded"][kind]
            if pat not in used:
                used.append(pat)
            i = used.index(pat)
            conns.append(f'<connection><identifier v="{ident}"/><connRef v="{100 + 10 * i}"/>'
                         f'<connRefOutput v="{101 + 10 * i}"/></connection>')
        comp_outs = "".join(
            f'<compOutput><uid v="{210 + j}"/><comptype v="{GRAY if k == "g" else COLOR}"/></compOutput>'
            for j, (_, k) in enumerate(outs))
        bridgings = "".join(
            f'<outputBridging><uid v="{210 + j}"/><identifier v="{o}"/></outputBridging>'
            for j, (o, _) in enumerate(outs))
        node = (
            f'<compNode><uid v="200"/><connections>{"".join(conns)}</connections>'
            f"<compOutputs>{comp_outs}</compOutputs><compImplementation><compInstance>"
            f'<path v="pkg:///{gid}?dependency=60"/><parameters/>'
            f"<outputBridgings>{bridgings}</outputBridgings></compInstance>"
            "</compImplementation></compNode>"
        )
        dep = (f'<dependency><filename v="sbs://{r["package"]}"/><uid v="60"/><type v="package"/>'
               '<fileUID v="0"/><versionUID v="0"/></dependency>')
        gouts = [(re.sub(r"[^A-Za-z0-9_]+", "_", o), GRAY if k == "g" else COLOR, 200, 210 + j)
                 for j, (o, k) in enumerate(outs)]
        cases.append({
            "node": f"l_{safe}", "category": r["category"], "package": f"{r['package']}#{gid}",
            "variant": "default", "inputs": r["image_inputs"],
            "outputs": " ".join(f"{o}:{k}[{u}]" for (o, k), u in zip(
                outs, assign_usages([GRAY if k == "g" else COLOR for _, k in outs]))),
            "xml": package(node, used, gouts, dep),
        })
    return cases


# ------------------------------------------------------------- cook and render


def cook(root: Path, node: str) -> tuple[bool, str]:
    out = root / "sbsar"
    r = subprocess.run([str(COOKER), "--inputs", str(root / "sbs" / f"{node}.sbs"),
                        "--output-path", str(out), "--quiet"],
                       capture_output=True, text=True, check=False)
    ok = r.returncode == 0 and (out / f"{node}.sbsar").is_file()
    return ok, "" if ok else (r.stdout + r.stderr).strip().replace("\n", " | ")[:300]


def render(root: Path, node: str, depth: str) -> tuple[bool, str]:
    out = root / "truth" / node
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    r = subprocess.run([str(RENDER), "render", "--input", str(root / "sbsar" / f"{node}.sbsar"),
                        "--output-path", str(out), "--output-name", "{outputNodeName}",
                        "--output-bit-depth", depth, "--set-value", f"$outputsize@{LOG2},{LOG2}",
                        "--png-format-compression", "best_compression", "--no-report"],
                       capture_output=True, text=True, check=False)
    ok = r.returncode == 0 and any(out.glob("*.png"))
    return ok, "" if ok else (r.stdout + r.stderr).strip().replace("\n", " | ")[:300]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--only", choices=("atomic", "library"))
    ap.add_argument("--match", default="")
    ap.add_argument("--steps", default="gen,cook,render")
    ap.add_argument("--depth", default="16", help="8, 16, or auto (16 if any output is gray, else 8)")
    ap.add_argument("--size", type=int, default=256, help="pattern and output size (256 or 128)")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--inventory", default="inventory.tsv")
    a = ap.parse_args()
    root = a.root.resolve()
    global LOG2
    LOG2 = a.size.bit_length() - 1
    cases = []
    if a.only != "library":
        cases += atomic_cases()
    if a.only != "atomic":
        cases += library_cases(root)
    cases = [c for c in cases if a.match in c["node"]]
    (root / "sbs").mkdir(exist_ok=True)
    (root / "sbsar").mkdir(exist_ok=True)
    steps = a.steps.split(",")
    if "gen" in steps:
        for c in cases:
            (root / "sbs" / f"{c['node']}.sbs").write_text(c["xml"], encoding="utf-8")
    inv_path = root / a.inventory
    old = {}
    if inv_path.is_file():
        old = {r["node"]: r for r in csv.DictReader(inv_path.open(encoding="utf-8"), delimiter="\t")}
    with ThreadPoolExecutor(a.jobs) as ex:
        if "cook" in steps:
            for c, (ok, err) in zip(cases, ex.map(lambda c: cook(root, c["node"]), cases)):
                c["cooked"], c["cook_error"] = ok, err
                print("cook", "ok " if ok else "FAIL", c["node"], err, flush=True)
        if "render" in steps:
            todo = [c for c in cases if (root / "sbsar" / f"{c['node']}.sbsar").is_file()]
            for c, (ok, err) in zip(todo, ex.map(lambda c: render(root, c["node"], a.depth if a.depth != "auto" else ("16" if ":g" in c["outputs"] else "8")), todo)):
                c["rendered"], c["render_error"] = ok, err
                print("render", "ok " if ok else "FAIL", c["node"], err, flush=True)
    cols = ["node", "category", "package", "variant", "inputs", "outputs",
            "cooked_ok", "rendered_ok", "error"]
    for c in cases:
        prev = old.get(c["node"], {})
        cooked = c.get("cooked", prev.get("cooked_ok") == "1")
        rendered = c.get("rendered", prev.get("rendered_ok") == "1" if "cooked" not in c else False)
        if "cook" in steps and not cooked:
            rendered = False
        err = c.get("cook_error") or c.get("render_error") or ("" if "cooked" in c else prev.get("error", ""))
        old[c["node"]] = {"node": c["node"], "category": c["category"], "package": c["package"],
                          "variant": c["variant"], "inputs": c["inputs"], "outputs": c["outputs"],
                          "cooked_ok": "1" if cooked else "0", "rendered_ok": "1" if rendered else "0",
                          "error": err}
    with inv_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in sorted(old.values(), key=lambda r: (not r["node"].startswith("a_"), r["node"])):
            fh.write("\t".join(str(r[k]).replace("\t", " ") for k in cols) + "\n")
    print(len(cases), "cases;", sum(c.get("cooked", False) for c in cases), "cooked;",
          sum(c.get("rendered", False) for c in cases), "rendered")


if __name__ == "__main__":
    sys.exit(main())
