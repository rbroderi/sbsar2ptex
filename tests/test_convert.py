"""Choosing outputs, matching hand-exported maps and writing the .ptex."""

from pathlib import Path

import pytest

from sbsar2ptex.convert import SLOTS
from sbsar2ptex.convert import choose
from sbsar2ptex.convert import describe
from sbsar2ptex.convert import match
from sbsar2ptex.convert import ptex


def output(identifier: str, *channels: tuple[str, str]) -> str:
    rows = "".join(f'<channel components="{c}" names="{n}"/>' for n, c in channels)
    return (
        f'<output uid="1" identifier="{identifier}" format="0">'
        f'<outputgui label="{identifier}"><channels>{rows}</channels>'
        "</outputgui></output>"
    )


XML = (
    '<?xml version="1.0"?><sbsdescription><graphs count="1">'
    '<graph pkgurl="pkg://Test" label="Test" description="A test material.">'
    "<outputs>"
    + output("basecolor", ("baseColor", "RGB"))
    + output("roughness", ("roughness", "RGB"))
    + output("normal", ("normal", "RGB"))
    + output("normalopengl", ("normal", "RGB"))
    + output("height", ("height", "RGB"))
    + output("mrao", ("metallic", "R"), ("roughness", "G"), ("ambientOcclusion", "B"))
    + output("opacity", ("opacity", "R"))
    + "</outputs><inputs>"
    '<input uid="2" identifier="Amount" type="0" default="0.5">'
    '<inputgui label="Amount"/></input>'
    "</inputs></graph></graphs></sbsdescription>"
).encode()


@pytest.fixture
def graph():
    (graph,) = describe(XML)
    return graph


def test_describe(graph) -> None:
    assert graph.url == "pkg://Test"
    assert [o.identifier for o in graph.outputs if o.packed] == ["mrao"]
    assert graph.inputs == [
        {"identifier": "Amount", "type": "float", "default": "0.5", "label": "Amount"}
    ]


def test_choose_prefers_single_channel_and_opengl(graph) -> None:
    chosen = {slot: o.identifier for slot, o in choose(graph).items()}
    assert chosen == {
        "albedo": "basecolor",
        "roughness": "roughness",
        "normal": "normalopengl",
        "depth": "height",
        "opacity": "opacity",
    }


def test_match_by_channel_name(graph, tmp_path: Path) -> None:
    given = tmp_path / "given"
    given.mkdir()
    for name in (
        "Rock_BaseColor.png",
        "rock-normal.tga",
        "rock_height.png",
        "notes.txt",
    ):
        (given / name).write_bytes(b"x")
    maps = tmp_path / "maps"
    maps.mkdir()
    found = match(given, choose(graph), maps)
    assert sorted(found) == ["albedo", "depth", "normal"]
    assert found["normal"] == maps / "normal.tga"
    assert found["normal"].is_file()


@pytest.mark.parametrize(("normal", "op"), [("opengl", 0), ("directx", 1)])
def test_ptex_wiring(graph, tmp_path: Path, normal: str, op: int) -> None:
    maps = {slot: tmp_path / f"{slot}.png" for slot in ("albedo", "normal", "opacity")}
    made = ptex(graph, maps, 11, normal)
    ports = {slot: n for n, (slot, _) in enumerate(SLOTS)}
    wires = {(c["from"], c["to"], c["to_port"]) for c in made["connections"]}
    assert wires == {
        ("image_albedo", "Material", ports["albedo"]),
        ("image_normal", "normal_convert", 0),
        ("normal_convert", "Material", ports["normal"]),
        ("image_opacity", "Material", ports["opacity"]),
    }
    nodes = {n["name"]: n for n in made["nodes"]}
    assert nodes["normal_convert"]["parameters"]["op"] == op
    assert nodes["Material"]["parameters"]["flags_transparent"] is True
    assert nodes["image_albedo"]["parameters"]["image"].endswith("/albedo.png")
