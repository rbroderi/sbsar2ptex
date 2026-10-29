"""The sbs Material Maker plugin: its node definitions and the converter's use of them.

Their pixels are checked against the inline translation by
scripts/check_plugin.py (through the material-maker-mod fork).
"""

import json
import re
from pathlib import Path

import pytest

from sbsar2ptex import plugin
from sbsar2ptex.archive import member
from sbsar2ptex.archive import read_archive
from sbsar2ptex.sbsasm import Asm
from sbsar2ptex.translate import PLUGIN_LINKS
from sbsar2ptex.translate import TRANSLATORS
from sbsar2ptex.translate import GraphInput
from sbsar2ptex.translate import output_type
from sbsar2ptex.translate import remote
from sbsar2ptex.translate import translate_graph

DATA = Path(__file__).parent / "data"


def translated(name: str, maker: plugin.Maker | None):
    files = read_archive(DATA / f"{name}.sbsar")
    xml = member(files, ".xml").decode("utf-8")
    uids = {int(u) for u in re.findall(r'uid="(\d+)" identifier="input_', xml)}
    asm = Asm(member(files, ".sbsasm"))
    return translate_graph(
        asm.nodes(),
        {"output": asm.output_nodes[0][1]},
        {u: f"{u}.png" for u in uids},
        plugins=maker,
    )


def test_install_writes_the_plugin(tmp_path) -> None:
    written = plugin.install(tmp_path / "sbs")
    cfg = (tmp_path / "sbs" / "plugin.cfg").read_text(encoding="utf-8")
    assert f'version="{plugin.VERSION}"' in cfg
    assert 'nodes="nodes"' in cfg
    nodes = sorted((tmp_path / "sbs" / "nodes").glob("*.mmg"))
    assert len(written) == len(nodes) + 1
    for path in nodes:
        definition = json.loads(path.read_text(encoding="utf-8"))
        assert definition["name"] == path.stem
        assert path.stem.startswith("sbs_")
        # every node has Substance's output format among its knobs
        assert "format" in definition["parameters"]


def test_every_plugin_op_is_a_translated_one() -> None:
    assert set(plugin.OPS) <= set(TRANSLATORS)
    atoms = plugin.atoms()
    for op in plugin.OPS.values():
        assert f"sbs_{op.base}" in atoms
        if op.variants:
            assert f"sbs_{op.base}_gray" in atoms


def test_shaders_keep_substance_names() -> None:
    levels = plugin.definitions()["sbs_levels_gray"]
    names = [p["name"] for p in levels["shader_model"]["parameters"]]
    assert names == [
        "levelinlow", "levelinhigh", "levelinmid", "leveloutlow", "levelouthigh",
        "format",
    ]  # fmt: skip
    model = levels["shader_model"]
    assert [i["name"] for i in model["inputs"]] == ["input1"]
    # the image width is the rendered texture's, not a number
    warp = plugin.definitions()["sbs_warp"]["shader_model"]["instance"]
    assert "$pass_size.x" in warp


def test_levels_is_a_plugin_node() -> None:
    maker = plugin.Maker(11)
    nodes, wires, outs = translated("levels_gray", maker)
    node = next(n for n in nodes if n["name"] == outs["output"][0])
    assert node["type"] == "sbs_levels_gray"
    assert node["parameters"]["levelinlow"] == pytest.approx(0.1)
    assert node["parameters"]["format"] in (0, 1, 2, 3)
    assert output_type(node) == "f"
    (w,) = wires
    assert (w["to"], w["to_port"]) == (node["name"], 0)
    assert maker.made == {"sbs_levels_gray": 1}


def test_sampled_inputs_are_buffered_by_the_graph() -> None:
    # As inline: a warp's input and gradient are read through buffers, which
    # the graph merges and drops as it does the inline translation's.
    nodes, wires, outs = translated("warp_gray_chain", plugin.Maker(11))
    types = {n["name"]: n["type"] for n in nodes}
    warp = outs["output"][0]
    assert types[warp] == "sbs_warp_gray"
    into = {w["to_port"]: types[w["from"]] for w in wires if w["to"] == warp}
    assert into == {0: "buffer", 1: "buffer"}
    blur = next(n for n in nodes if n["type"] == "sbs_blur_gray")
    assert blur["parameters"]["outputsize"] == 0.0


def test_other_ops_stay_inline() -> None:
    maker = plugin.Maker(10)
    files = read_archive(DATA / "pp_reads_outputsize.sbsar")
    asm = Asm(member(files, ".sbsasm"))
    nodes = asm.nodes()
    op = nodes[asm.output_nodes[0][1]].op
    assert maker(nodes[asm.output_nodes[0][1]], "n", 10, None) is None
    assert maker.inline == {f"op {op}: no plugin node": 1}


def test_graph_inputs_set_the_knobs_they_drive() -> None:
    # A setting that is a graph input: the remote sets the plugin's knob.
    gi = GraphInput("Height_Shift", 0, [0.25], "Height shift")
    knobs = plugin._Knobs(None)  # type: ignore[arg-type]
    (value,) = knobs._link(gi, ["levelinlow"])
    assert value == 0.25
    node = {
        "name": "n1", "type": "sbs_levels_gray", "parameters": {"levelinlow": 0.25},
        PLUGIN_LINKS: knobs.links,
    }  # fmt: skip
    made = remote({"p_Height_Shift": ["n1"]}, [node])
    (widget,) = made["widgets"]
    assert widget["linked_widgets"] == [{"node": "n1", "widget": "levelinlow"}]
    assert widget["label"] == "Height shift"
