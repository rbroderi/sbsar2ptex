"""The sbs Material Maker plugin: its node definitions and the converter's use of them.

Their pixels are checked against the inline translation by
scripts/check_plugin.py (through the material-maker-mod fork).
"""

import base64
import json
import re
import struct
from pathlib import Path

import pytest

from sbsar2ptex import fxglsl
from sbsar2ptex import fxsplat
from sbsar2ptex import plugin
from sbsar2ptex import translate
from sbsar2ptex.archive import member
from sbsar2ptex.archive import read_archive
from sbsar2ptex.sbsasm import Asm
from sbsar2ptex.translate import PLUGIN_LINKS
from sbsar2ptex.translate import TRANSLATORS
from sbsar2ptex.translate import GraphInput
from sbsar2ptex.translate import graph_inputs
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
        inputs=graph_inputs(asm, xml),
        plugins=maker,
    )


def test_install_writes_the_plugin(tmp_path) -> None:
    written = plugin.install(tmp_path / "sbs")
    cfg = (tmp_path / "sbs" / "plugin.cfg").read_text(encoding="utf-8")
    assert f'version="{plugin.VERSION}"' in cfg
    assert 'nodes="nodes"' in cfg
    assert 'script="plugin.gd"' in cfg
    gd = (tmp_path / "sbs" / "plugin.gd").read_text(encoding="utf-8")
    assert 'add_node_type("sbs_fxmap", preload("gen_sbs_fxmap.gd"))' in gd
    fxmap = (tmp_path / "sbs" / "gen_sbs_fxmap.gd").read_text(encoding="utf-8")
    # a class_name in user:// would not register; the file format is the fork's
    assert "\nclass_name " not in fxmap
    assert 'return "sbs_fxmap"' in fxmap
    assert 'add_node_type("sbs_splats", preload("gen_sbs_splats.gd"))' in gd
    splats = (tmp_path / "sbs" / "gen_sbs_splats.gd").read_text(encoding="utf-8")
    assert "\nclass_name " not in splats
    assert 'return "sbs_splats"' in splats
    assert (tmp_path / "sbs" / "README.md").exists()
    nodes = sorted((tmp_path / "sbs" / "nodes").glob("*.mmg"))
    assert len(written) == len(nodes) + 5
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
        "format", "gpu",
    ]  # fmt: skip
    model = levels["shader_model"]
    assert [i["name"] for i in model["inputs"]] == ["input1"]
    # the image width is the rendered texture's, not a number
    warp = plugin.definitions()["sbs_warp"]["shader_model"]["instance"]
    assert "$pass_size.x" in warp


def test_plugin_stores_are_the_gpus(monkeypatch) -> None:
    # As the inline translation stores (translate.store_shader): its GPU's
    # 8 and 16 bits levels by $gpu, set from translate._DMB_AMD; a uniform's
    # fill at 8 bits the nearest, ties down (B's n526, n656: the plugin's
    # blend stored as before, 93.79% / 96.52%; inline 99.98% / 100%)
    blend = plugin.definitions()["sbs_blend_gray"]["shader_model"]["instance"]
    for amd in (False, True):
        assert f"if ($gpu == {int(amd)})" in blend
        for level in (fxglsl.level8_glsl("float", amd=amd),
                      fxglsl.level16_glsl("float", amd=amd)):  # fmt: skip
            assert all(line.strip() in blend for line in level.splitlines())
    uniform = plugin.definitions()["sbs_uniform_gray"]["shader_model"]["instance"]
    fill = fxglsl.fill8_glsl("float")
    assert all(line.strip() in uniform for line in fill.splitlines())
    assert "4096.0" not in uniform
    for amd in (False, True):
        monkeypatch.setattr(translate, "_DMB_AMD", amd)
        nodes, _, outs = translated("levels_gray", plugin.Maker(11))
        node = next(n for n in nodes if n["name"] == outs["output"][0])
        assert node["parameters"]["gpu"] == int(amd)


def test_plugin_blurs_store_as_the_gpu() -> None:
    # a directional blur's passes store as $gpu's (blur2_8 on AMD: 89.74%
    # with NVIDIA's stores, 99.997% with AMD's)
    dmb = plugin.definitions()["sbs_dirmotionblur_gray"]["shader_model"]["instance"]
    for gpu, tag in ((0, "n"), (1, "a")):
        assert (
            f"if ($gpu == {gpu} && $format == 0) return $(name)_v0{tag}_body(uv);"
            in dmb
        )
    assert "uintBitsToFloat" in dmb
    assert "4096.0 + c" in dmb


def test_plugin_warps_read_as_the_gpu() -> None:
    # a warp's bilinear read and rounding are $gpu's, as the inline
    # translation's are (warp_g on AMD: 88.96% with NVIDIA's, 100% now)
    for atom in ("sbs_warp_gray", "sbs_directionalwarp_gray"):
        code = plugin.definitions()[atom]["shader_model"]["instance"]
        for gpu, tag in ((0, "n"), (1, "a")):
            assert (
                f"if ($gpu == {gpu} && $format == 1) return $(name)_v1{tag}_body(uv);"
                in code
            )
        assert "roundEven(" in code  # AMD's
        assert "+ 0.5)" in code  # NVIDIA's


def test_a_nearest_transformation_reads_one_texel() -> None:
    # transform_nearx3_g/_c: a 3x zoom out reads mip level 2's texel q is in,
    # not a bilinear mix of four (66.6% / 87.3% before, 100% now)
    near = translate.transform_read_lines("input", ["3.0"] * 4, "256", "0",
                                          color=False, nearest="true")  # fmt: skip
    assert "bool tnear = true;" in near
    assert "for (int k = 0; k < (tnear ? 1 : 4); k++) {" in near
    # transform_g: bilinear at level 0 too texel by texel, fractions to
    # 1/256, not the GPU's filter (98.05% before, 100% now)
    assert "fract(q)" not in "".join(near)
    # transform_tile0_g: untiled, texels past the edges are empty
    assert "int tl = 3;" in near
    assert "\tif ((tout.x && !twrap.x) || (tout.y && !twrap.y)) continue;" in near
    t = plugin.definitions()["sbs_transformation_gray"]
    assert "bool tnear = ($filtering == 1);" in t["shader_model"]["instance"]
    assert t["parameters"]["filtering"] == 0


def test_plugin_distance_seeds_as_the_translation() -> None:
    # as translate._distance: 16-bit unorm seeds, Substance's neighbours
    graph = json.dumps(plugin.definitions()["sbs_distance_gray"])
    assert json.dumps(translate.JFA_SEED)[1:-1] in graph
    assert "ivec2(1, 0), ivec2(-1, 0), ivec2(0, 1)" in graph
    assert "lessThan(abs(c.xy - uv)" in graph


def test_plugin_fxmaps_draw_triangles() -> None:
    # a gradation turned a quarter (Biomech's n2): Substance's GPU draws its
    # patterns as triangles, so the Splats node does (the inline translation
    # draws them per pixel)
    nodes, wires, _ = translated("fxmap_gradation", plugin.Maker(11))
    drawn = [n for n in nodes if n["type"] == "sbs_splats"]
    assert len(drawn) == 1
    p = drawn[0]["parameters"]
    assert p["size"] == 11
    assert "bool fxpat_fx(int id, vec2 q, float s, out float v)" in p["pattern_code"]
    raw = base64.b64decode(p["splats"])
    floats = struct.unpack(f"<{len(raw) // 4}f", raw)
    assert len(floats) % fxsplat.RASTER_FLOATS == 0
    first = floats[: fxsplat.RASTER_FLOATS]
    assert first[:4] == (1.0, 1.0, 1.0, 1.0)  # its color, opaque
    assert first[8:11] == (1.0, 1.0, pytest.approx(0.25))  # size, turns
    assert first[12:14] == (8.0, 0.0)  # the gradation, adding
    assert (0.5, 0.5) in [
        floats[k + 6 : k + 8] for k in range(0, len(floats), fxsplat.RASTER_FLOATS)
    ]
    shader = next(n for n in nodes if "$raster(" in json.dumps(n))
    assert any(
        w["from"] == drawn[0]["name"] and w["to"] == shader["name"] for w in wires
    )
    inline, _, _ = translated("fxmap_gradation", None)
    assert not any(n["type"] == "sbs_splats" for n in inline)
    gd = (Path(plugin.__file__).parent / "mm_plugin" / "gen_sbs_splats.gd").read_text(
        encoding="utf-8"
    )
    assert "2.0/rw, -2.0/rh, size.x/rw, size.y/rh, rw/size.x, rh/size.y" in gd


def test_plugin_fxmaps_draw_their_regions() -> None:
    # x_region3: render and tiling regions -0.25 to 1.25 by 0.1 to 0.9
    # (Biomech's n2369: -3 to 4), the image the render region (99.93%
    # inline, 99.98% drawn)
    nodes, _, _ = translated("fxmap_region", plugin.Maker(8))
    (drawn,) = [n for n in nodes if n["type"] == "sbs_splats"]
    p = drawn["parameters"]
    region = (p["region_x"], p["region_y"], p["region_w"], p["region_h"])
    assert region == pytest.approx((-0.25, 0.9, 1.5, 0.8))
    # x_region5: a disc's copy ends at the tiling region's edge, in the
    # image: inline, as Substance (99.91%; 93.66% drawn)
    nodes, _, _ = translated("fxmap_region_edge", plugin.Maker(8))
    assert not any(n["type"] == "sbs_splats" for n in nodes)


def test_an_emboss_scales_its_slope_by_its_pixel_size() -> None:
    # Biomech's n4291: an emboss whose Pixel Size is 0.1 lights ten times
    # less (18.9% of pixels as Substance's before; emboss_px_g 100% now)
    inline, _, _ = translated("emboss_pixelsize", None)
    code = next(json.dumps(n) for n in inline if "$gradient(uv)" in json.dumps(n))
    assert "h - $gradient(fract(uv - e.yx))) * f" in code
    nodes, _, outs = translated("emboss_pixelsize", plugin.Maker(11))
    node = next(n for n in nodes if n["name"] == outs["output"][0])
    assert node["type"] == "sbs_emboss_gray"
    p = node["parameters"]
    assert (p["pixelsize_x"], p["pixelsize_y"]) == (pytest.approx(0.1),) * 2
    # Substance's default is 1
    shader = plugin.definitions()["sbs_emboss_gray"]["shader_model"]
    assert "vec2($pixelsize_x, $pixelsize_y)" in shader["instance"]
    defaults = {p["name"]: p["default"] for p in shader["parameters"]}
    assert defaults["pixelsize_x"] == defaults["pixelsize_y"] == 1.0


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
