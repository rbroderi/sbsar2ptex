"""Find which .sbsasm op each Substance atomic filter compiles to.

    sbsasm-probe [--cooker PATH] [--keep DIR] [--chain] [FILTER ...]

For each atomic filter (or those named), in grayscale and in color, this
writes a graph of one image input, that filter and one output, cooks it with
Substance Designer's ``sbscooker`` and prints the node words the compiled
graph holds (see sbsar2ptex.sbsasm): the filter's op is the one that is
not the input's. ``--keep`` keeps the .sbs and cooked files there.
"""

import argparse
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

from sbsar2ptex.archive import read_archive

GRAY_INPUTS = {"opacity", "inputgradient", "inputintensity", "intensity", "gradient"}
"""Filter inputs that take a grayscale image whatever the filter's own type."""
COOKER = Path("C:/Program Files/Adobe/Adobe Substance 3D Designer/sbscooker.exe")
NODE_MARK = 0x88
GRAY, COLOR = 2, 1
"""Substance's image types: comptype / paraminput type."""
FILTERS: dict[str, tuple[str, ...]] = {
    # Atomic filter: its image inputs, fed by graph inputs A, B and C in turn.
    "blend": ("source", "destination", "opacity"),
    "blur": ("input1",),
    "shuffle": ("input1", "input2"),
    "curve": ("input1",),
    "dirmotionblur": ("input1",),
    "directionalwarp": ("input1", "inputintensity"),
    "distance": ("mask", "source"),
    "dyngradient": ("input1", "input2"),
    "emboss": ("input1", "inputgradient"),
    "gradient": ("input1",),
    "grayscaleconversion": ("input1",),
    "hsl": ("input1",),
    "levels": ("input1",),
    "normal": ("input1",),
    "passthrough": ("input1",),
    "sharpen": ("input1",),
    "transformation": ("input1",),
    "uniform": (),
    "warp": ("input1", "inputgradient"),
}

PARAMETERS: dict[str, tuple[tuple[str, str, str], ...]] = {
    # Non-default settings, so the cooker cannot fold the node away.
    "blend": (
        ("opacitymult", "constantValueFloat1", "0.37"),
        ("blendingmode", "constantValueInt1", "4"),
    ),
    "blur": (("intensity", "constantValueFloat1", "7.25"),),
    "dirmotionblur": (
        ("intensity", "constantValueFloat1", "7.25"),
        ("mblurangle", "constantValueFloat1", "0.1"),
    ),
    "directionalwarp": (
        ("intensity", "constantValueFloat1", "7.25"),
        ("warpangle", "constantValueFloat1", "0.1"),
    ),
    "distance": (("distance", "constantValueFloat1", "7.25"),),
    "emboss": (("intensity", "constantValueFloat1", "7.25"),),
    "hsl": (("hue", "constantValueFloat1", "0.3"),),
    "levels": (("levelinlow", "constantValueFloat4", "0.1 0.1 0.1 0.1"),),
    "normal": (("intensity", "constantValueFloat1", "3"),),
    "sharpen": (("intensity", "constantValueFloat1", "2"),),
    "transformation": (("offset", "constantValueFloat2", "0.25 0.1"),),
    "uniform": (("outputcolor", "constantValueFloat4", "0.3 0.4 0.5 1"),),
    "warp": (("intensity", "constantValueFloat1", "7.25"),),
    "grayscaleconversion": (
        ("channelsweights", "constantValueFloat4", "0.2 0.3 0.5 0"),
    ),
}

BRIDGE_PARAMETERS = "".join(
    f'<parameter><name v="{name}"/><relativeTo v="1"/><paramValue>'
    f'<{kind} v="{value}"/></paramValue></parameter>'
    for name, kind, value in (
        ("outputsize", "constantValueInt2", "0 0"),
        ("format", "constantValueInt1", "0"),
        ("pixelsize", "constantValueFloat2", "1 1"),
        ("pixelratio", "constantValueInt1", "0"),
        ("tiling", "constantValueInt1", "0"),
        ("randomseed", "constantValueInt1", "0"),
    )
)


def input_kinds(filt: str, kind: int) -> list[int]:
    """Return the image type of graph inputs A, B and C in ``filt``'s probe.

    Input k feeds the filter's k-th input: grayscale where that input always
    is (opacity, gradients, intensities, a dynamic gradient's input1), else
    the probe's own image type.
    """
    inputs = FILTERS[filt]
    return [
        GRAY
        if k < len(inputs)
        and (inputs[k] in GRAY_INPUTS or (filt, inputs[k]) == ("dyngradient", "input1"))
        else kind
        for k in range(3)
    ]


def gradient_array(
    keys: list[tuple[float, tuple[float, float, float, float]]],
    *,
    name: str = "gradientrgba",
) -> str:
    """Return a gradient's <paramsArray>: keys as (position, rgba)."""
    cells = "".join(
        f'<paramsArrayCell><uid v="{900 + i}"/><parameters>'
        '<parameter><name v="position"/><paramValue>'
        f'<constantValueFloat1 v="{pos}"/></paramValue></parameter>'
        '<parameter><name v="midpoint"/><paramValue>'
        '<constantValueFloat1 v="0.5"/></paramValue></parameter>'
        '<parameter><name v="value"/><paramValue>'
        f'<constantValueFloat4 v="{" ".join(map(str, rgba))}"/></paramValue>'
        "</parameter></parameters></paramsArrayCell>"
        for i, (pos, rgba) in enumerate(keys)
    )
    return (
        f'<paramsArray><name v="{name}"/><uid v="899"/>'
        f"<paramsArrayCells>{cells}</paramsArrayCells></paramsArray>"
    )


def curve_array(
    points: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]],
    *,
    name: str = "curveluminance",
) -> str:
    """Return a curve's <paramsArray>: points as (position, left, right) handles."""

    def f2(pname: str, xy: tuple[float, float]) -> str:
        return (
            f'<parameter><name v="{pname}"/><paramValue>'
            f'<constantValueFloat2 v="{xy[0]} {xy[1]}"/></paramValue></parameter>'
        )

    def flag(pname: str) -> str:
        return (
            f'<parameter><name v="{pname}"/><paramValue>'
            '<constantValueBool v="0"/></paramValue></parameter>'
        )

    cells = "".join(
        f'<paramsArrayCell><uid v="{900 + i}"/><parameters>'
        f"{f2('position', pos)}{f2('left', left)}{flag('isLeftBroken')}"
        f"{f2('right', right)}{flag('isRightBroken')}{flag('isLocked')}"
        "</parameters></paramsArrayCell>"
        for i, (pos, left, right) in enumerate(points)
    )
    return (
        f'<paramsArray><name v="{name}"/><uid v="899"/>'
        f"<paramsArrayCells>{cells}</paramsArrayCells></paramsArray>"
    )


def sbs(
    filt: str,
    kind: int,
    *,
    chain: bool = False,
    params: tuple[tuple[str, str, str], ...] | None = None,
    arrays: str = "",
    format16: bool = False,
) -> str:
    """Return a package: graph inputs -> ``filt`` -> output, of image ``kind``.

    With ``chain``, input A reaches the filter through a blur, so the filter's
    first input is not node 0 (which reads the same as a zero flag word).
    ``params`` ((name, value kind, value) each) replaces PARAMETERS[filt];
    ``arrays`` is the filter's <paramsArrays> content (gradient keys, curve
    points), as .sbs XML. ``format16`` makes the graph 16-bit throughout (its
    inputs are 8-bit otherwise, which slope-based filters amplify).
    """
    inputs = FILTERS[filt]
    first = 150 if chain else 100
    connections = "".join(
        f'<connection><identifier v="{name}"/>'
        f'<connRef v="{first if k == 0 else 100 + 10 * k}"/>'
        f'<connRefOutput v="{first + 1 if k == 0 else 101 + 10 * k}"/></connection>'
        for k, name in enumerate(inputs)
    )
    chained = (
        '<compNode><uid v="150"/><connections><connection>'
        '<identifier v="input1"/><connRef v="100"/><connRefOutput v="101"/>'
        '</connection></connections><compOutputs><compOutput><uid v="151"/>'
        f'<comptype v="{kind}"/></compOutput></compOutputs><compImplementation>'
        '<compFilter><filter v="blur"/><parameters><parameter>'
        '<name v="intensity"/><relativeTo v="0"/><paramValue>'
        '<constantValueFloat1 v="3"/></paramValue></parameter></parameters>'
        "</compFilter></compImplementation></compNode>"
        if chain
        else ""
    )
    parameters = "".join(
        f'<parameter><name v="{name}"/><relativeTo v="0"/><paramValue>'
        f'<{vkind} v="{value}"/></paramValue></parameter>'
        for name, vkind, value in (
            PARAMETERS.get(filt, ()) if params is None else params
        )
    )
    kinds = input_kinds(filt, kind)

    def paraminput(ident: str, uid: int) -> str:
        k = kinds[uid - 10]
        default = (
            '<constantValueFloat4 v="0 0 0 0"/>'
            if k == COLOR
            else '<constantValueFloat1 v="0"/>'
        )
        return (
            f'<paraminput><identifier v="{ident}"/><uid v="{uid}"/>'
            f'<isConnectable v="1"/><type v="{k}"/>'
            f"<defaultValue>{default}</defaultValue>"
            '<defaultWidget><name v=""/><options/></defaultWidget></paraminput>'
        )

    def bridge(uid: int, entry: int) -> str:
        return (
            f'<compNode><uid v="{uid}"/><compOutputs><compOutput><uid v="{uid + 1}"/>'
            f'<comptype v="{kinds[entry - 10]}"/></compOutput></compOutputs>'
            "<compImplementation>"
            f'<compInputBridge><entry v="{entry}"/><parameters>{BRIDGE_PARAMETERS}'
            "</parameters></compInputBridge></compImplementation></compNode>"
        )

    output_kind = GRAY if filt == "grayscaleconversion" else kind
    if filt in ("gradient", "shuffle", "hsl", "normal"):
        output_kind = COLOR
    if filt == "gradient" and any(
        name == "colorswitch" and value == "0" for name, _, value in params or ()
    ):
        output_kind = GRAY  # a grayscale gradient map
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?><package><identifier v="probe"/>'
        '<formatVersion v="1.1.0.202401"/><updaterVersion v="1.1.0.202401"/>'
        '<fileUID v="{00000000-0000-0000-0000-000000000001}"/><versionUID v="0"/>'
        '<dependencies/><content><graph><identifier v="probe"/><uid v="1"/>'
        '<graphtype v="filter"/><paraminputs>'
        f"{paraminput('input_a', 10)}{paraminput('input_b', 11)}"
        f"{paraminput('input_c', 12)}"
        '</paraminputs><primaryInput v="10"/><graphOutputs><graphoutput>'
        f'<identifier v="output"/><uid v="20"/><channels v="{output_kind}"/>'
        "</graphoutput></graphOutputs><compNodes>"
        # The graph inputs.
        f"{bridge(100, 10)}{bridge(110, 11)}{bridge(120, 12)}{chained}"
        # The filter.
        f'<compNode><uid v="200"/><connections>{connections}</connections>'
        '<compOutputs><compOutput><uid v="201"/>'
        f'<comptype v="{output_kind}"/></compOutput></compOutputs>'
        f'<compImplementation><compFilter><filter v="{filt}"/>'
        f"<parameters>{parameters}</parameters>"
        f"{f'<paramsArrays>{arrays}</paramsArrays>' if arrays else ''}"
        "</compFilter></compImplementation></compNode>"
        # The output.
        '<compNode><uid v="300"/><connections><connection>'
        '<identifier v="inputNodeOutput"/><connRef v="200"/>'
        '<connRefOutput v="201"/></connection></connections>'
        '<compImplementation><compOutputBridge><output v="20"/>'
        "</compOutputBridge></compImplementation></compNode>"
        '</compNodes><baseParameters/><options><option><name v="defaultParentSize"/>'
        '<value v="8x8"/></option></options><root><rootOutputs><rootOutput>'
        '<output v="20"/><format v="0"/><usertag v=""/></rootOutput></rootOutputs>'
        "</root></graph></content></package>"
    )
    if format16:
        xml = xml.replace(
            '<name v="format"/><relativeTo v="1"/><paramValue>'
            '<constantValueInt1 v="0"/>',
            '<name v="format"/><relativeTo v="0"/><paramValue>'
            '<constantValueInt1 v="1"/>',
        ).replace(
            "<parameters>" + parameters + "</parameters>",
            "<parameters>"
            + parameters
            + '<parameter><name v="format"/><relativeTo v="0"/><paramValue>'
            '<constantValueInt32 v="1"/></paramValue></parameter></parameters>',
        )
    return xml


def node_words(sbsasm: bytes) -> list[tuple[int, int]]:
    """Return each node word's (op, flags), in file order."""
    found: list[tuple[int, int]] = []
    for at in range(0, len(sbsasm) - 3, 4):
        w = struct.unpack_from("<I", sbsasm, at)[0]
        if (w >> 8) & 0xFF == NODE_MARK and w >> 24 < 0x40:  # noqa: PLR2004
            found.append((w & 0xFF, w >> 16))
    return found


def cook(cooker: Path, source: Path, out: Path) -> bytes | None:
    """Cook ``source`` into ``out``; return its .sbsasm, or None on failure."""
    result = subprocess.run(  # noqa: S603 (the user's own tool)
        [str(cooker), "--inputs", str(source), "--output-path", str(out), "--quiet"],
        capture_output=True,
        text=True,
        check=False,
    )
    sbsar = out / f"{source.stem}.sbsar"
    if result.returncode or not sbsar.is_file():
        print(f"  cook failed: {(result.stdout + result.stderr).strip()[:300]}")
        return None
    files = read_archive(sbsar)
    return next(d for n, d in files.items() if n.endswith(".sbsasm"))


def main() -> None:
    """Probe the filters the command line names, or all of them."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("filters", nargs="*", default=list(FILTERS))
    parser.add_argument("--cooker", type=Path, default=COOKER)
    parser.add_argument("--keep", type=Path, help="keep the probe files here")
    parser.add_argument(
        "--chain", action="store_true", help="feed input A through a blur first"
    )
    args = parser.parse_args()

    work = args.keep or Path(tempfile.mkdtemp(prefix="sbsprobe"))
    work.mkdir(parents=True, exist_ok=True)
    suffix = "_chain" if args.chain else ""
    try:
        for filt in args.filters:
            for kind, label in ((GRAY, "gray"), (COLOR, "color")):
                source = work / f"{filt}_{label}{suffix}.sbs"
                source.write_text(sbs(filt, kind, chain=args.chain), encoding="utf-8")
                data = cook(args.cooker, source, work)
                if data is not None:
                    words = ", ".join(
                        f"op {op} ({flags:#06x})" for op, flags in node_words(data)
                    )
                    print(f"{filt:20} {label:5} {words}")
    finally:
        if args.keep is None:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
