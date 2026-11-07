"""Value Processor samples: samplelum and samplecol (docs/function-sample.md)."""

import json
from pathlib import Path

import pytest
from test_slots import VEC2
from test_slots import U
from test_slots import amount
from test_slots import code
from test_slots import compiles
from test_slots import fconst
from test_slots import getslot
from test_slots import glsl_of
from test_slots import instances
from test_slots import load
from test_slots import make_node
from test_slots import setslot
from test_slots import vp_fields

from sbsar2ptex.function import FLOAT
from sbsar2ptex.function import decode
from sbsar2ptex.sbsasm import Node
from sbsar2ptex.translate import UnsupportedError
from sbsar2ptex.translate import translate_graph

SAMPLELUM, SAMPLECOL = 0x33, 0x34


def words(stream: list[int]) -> list[int]:
    s = stream + [0] * (len(stream) % 2)
    return [s[k] | s[k + 1] << 16 for k in range(0, len(s), 2)]


def test_d1_a_value_processors_sample_decodes() -> None:
    stream = [0x0003, 0x1140, 0xFFFF, 0x3F7F, 0xFFFF, 0x3F7F, 0x0933, 0x0000, 0x0000,
              0x0906, 0x0001, 0x0003]  # fmt: skip
    const, sample, store = decode(words(stream)).instrs
    assert (const.op, const.width) == (0x00, 2)
    assert (sample.op, sample.type, sample.width) == (SAMPLELUM, FLOAT, 1)
    assert sample.operands == [0, 0]  # position #0, input 0, nearest
    assert (store.op, store.operands) == (0x06, [1, 3])


@pytest.mark.parametrize(
    ("opcode", "operands"),
    [(0x0DF4, [5, 0, 1]), (0x09F4, [5, 0])],  # D2 bilinear, D3 nearest
)
def test_d2_d3_samplecol(opcode: int, operands: list[int]) -> None:
    (sample,) = decode(words([1, opcode, *operands])).instrs
    assert (sample.op, sample.type, sample.width) == (SAMPLECOL, FLOAT, 4)
    assert sample.operands == operands  # operand 1 is the input index, a literal


def sample(op: int, pos: int, *, bilinear: bool = False):
    width = 1 if op == SAMPLELUM else 4
    ops = [pos, 0, 1] if bilinear else [pos, 0]
    return (FLOAT << 8 | (width - 1) << 6 | op, ops, [])


def image(index: int, op: int = 13) -> Node:
    """Return a uniform: an image (colour for op 13, grayscale for 12)."""
    return Node(index=index, start=0, end=4, op=op, words=[op])


def graph(vp_block: list[int], img_op: int = 13):
    """n0 an image, n1 a Value Processor sampling it into slot 0, n2 reading it."""
    vp = make_node(1, 40, [*vp_fields(1, images=1), 0], vp_block)
    vp.inputs = {"input": 0}  # (as Asm names it)
    pp = make_node(2, 40, [0], code(getslot(0)))
    return translate_graph(
        [image(0, img_op), vp, pp], {"out": 2}, {}, size=10, inputs=amount()
    )


def last_texel_lum() -> list[int]:
    """atlas_scatter n107's block (T1): the last texel's luminance."""
    one = 0.99999994
    return code(fconst(one), fconst(one), (FLOAT << 8 | 1 << 6 | VEC2, [0, 1], []),
                sample(SAMPLELUM, 2), setslot(3, 0))  # fmt: skip


def test_t1_the_reader_samples_the_image_at_its_own_size(tmp_path: Path) -> None:
    mm, wires, _ = graph(last_texel_lum())
    reader = next(n for n in mm if n["name"] == "n2")
    inputs = [i["name"] for i in reader["shader_model"]["inputs"]]
    assert inputs == ["vp0"]
    # an unfiltered (whole texels) 32-bit buffer of the image, at its size
    buf = next(n for n in mm if n["name"] == "n0_vp")
    assert buf["parameters"]["filter"] is False
    assert buf["parameters"]["f32"] is True
    assert [w["from"] for w in wires if w["to"] == "n0_vp"] in (["n0"], ["n0_size"])
    assert {"from": "n0_vp", "from_port": 0, "to": "n2", "to_port": 0} in wires
    code_ = instances(mm)["n2"]
    # nearest: the texel floor(p N) mod N; a colour image's mean of R, G, B
    assert "$vp0(fract(s0_2))" in code_
    assert "dot($vp0(fract(s0_2)).rgb, vec3(1.0 / 3.0))" in code_
    compiles(
        glsl_of(code_.replace("$vp0(", "vp0("), "n2").replace(
            "#version 450\n",
            "#version 450\nvec4 vp0(vec2 uv) { return vec4(uv, 0.0, 1.0); }\n",
        ),
        tmp_path,
    )


def test_a_grayscale_images_samples() -> None:
    mm, _, _ = graph(last_texel_lum(), img_op=12)
    assert ".r" in instances(mm)["n2"].split("sbs_slot0_$(name)(vec2 uv)")[1]
    col = code(fconst(0.0), fconst(0.0), (FLOAT << 8 | 1 << 6 | VEC2, [0, 1], []),
               sample(SAMPLECOL, 2), (FLOAT << 8 | 0x18, [3, 3], []),
               setslot(4, 0))  # fmt: skip
    mm, _, _ = graph(col, img_op=12)
    assert "vec4(vec3($vp0(fract(s0_2)).r), 1.0)" in instances(mm)["n2"]


def test_t4_a_position_from_an_input_stays_live() -> None:
    block = code(load(U), load(U), (FLOAT << 8 | 1 << 6 | VEC2, [0, 1], []),
                 sample(SAMPLELUM, 2), setslot(3, 0))  # fmt: skip
    mm, _, _ = graph(block)
    helper = instances(mm)["n2"].split("sbs_slot0_$(name)(vec2 uv)")[1]
    assert "$p_amount" in helper
    assert "3.5" not in helper  # not frozen at its default


def test_bilinear_mixes_four_whole_texels() -> None:
    block = code(fconst(0.1), fconst(0.2), (FLOAT << 8 | 1 << 6 | VEC2, [0, 1], []),
                 sample(SAMPLECOL, 2, bilinear=True), (FLOAT << 8 | 0x18, [3, 3], []),
                 setslot(4, 0))  # fmt: skip
    mm, _, _ = graph(block)
    code_ = instances(mm)["n2"]
    assert "vec4 sbs_vpbilinear0_$(name)(vec2 p)" in code_
    assert "vec2 t = p * n - 0.5;" in code_
    assert code_.count("$vp0(fract(") == 4


def test_a_sample_of_another_input_is_refused() -> None:
    bad = (FLOAT << 8 | SAMPLELUM, [2, 1], [])
    block = code(fconst(0.0), fconst(0.0), (FLOAT << 8 | 1 << 6 | VEC2, [0, 1], []),
                 bad, setslot(3, 0))  # fmt: skip
    with pytest.raises(UnsupportedError, match="input 1"):
        graph(block)


def test_an_image_only_a_value_processor_reads_is_translated() -> None:
    # n0 feeds nothing but the Value Processor: it is still made (its image)
    mm, _, _ = graph(last_texel_lum())
    assert any(n["name"] == "n0" for n in mm)
    assert json.dumps(mm).count("n0_vp") >= 1
