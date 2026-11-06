"""The decompiler, on graphs cooked from sbsasm-probe's one-node graphs.

Each fixture is input A -> (blur ->) one filter -> output, inputs B and C
wired to the filter's other inputs, cooked by Substance Designer 16.0.7.
"""

import re
import struct
from pathlib import Path

import pytest

from sbsar2ptex.archive import member
from sbsar2ptex.archive import read_archive
from sbsar2ptex.sbsasm import BASE
from sbsar2ptex.sbsasm import NO_NODE
from sbsar2ptex.sbsasm import UNWIRED
from sbsar2ptex.sbsasm import Asm
from sbsar2ptex.sbsasm import Node
from sbsar2ptex.sbsasm import field_count
from sbsar2ptex.sbsasm import pp_inputs
from sbsar2ptex.sbsasm import reachable
from sbsar2ptex.translate import word_params

DATA = Path(__file__).parent / "data"


def load(name: str) -> tuple[Asm, list[Node], dict[int, str]]:
    files = read_archive(DATA / f"{name}.sbsar")
    xml = member(files, ".xml").decode("utf-8")
    names = {int(u): n for u, n in re.findall(r'uid="(\d+)" identifier="([^"]*)"', xml)}
    asm = Asm(member(files, ".sbsasm"))
    return asm, asm.nodes(), names


def by_input(nodes: list[Node], names: dict[int, str]) -> dict[str, int]:
    """Graph input name -> the node that reads it."""
    return {names[n.uid]: n.index for n in nodes if n.uid is not None}


def only(nodes: list[Node], name: str) -> Node:
    (node,) = [n for n in nodes if n.name.startswith(name)]
    return node


def test_no_defaults_block() -> None:
    # previewsphere_sphere.sbsar (a version 2 cook) has no inputs and no
    # defaults block: offset 0. A decoy at 0x40 reads as a uid table ending
    # at the trailer, as its bitmap's bytes did.
    data = bytearray(0x84)
    data[:4] = b"SBAM"
    struct.pack_into("<H", data, 6, 2)
    struct.pack_into("<HH", data, 0x40, 12, 0)  # the decoy
    struct.pack_into("<I", data, 0x60, 0x50 - BASE)  # the node table: one node
    struct.pack_into("<II", data, 0x64, 0xC0, 0)  # the outputs: (format, node)
    struct.pack_into("<HHI", data, 0x6C, 1, 0, 0x1234)  # the uid table
    struct.pack_into("<4I", data, 0x74, 1, 0x60 - BASE, 0x64 - BASE, 0)
    asm = Asm(bytes(data))
    assert asm.output_uids == [0x1234]
    assert asm.output_nodes == [(0xC0, 0)]
    assert asm.inputs == []
    assert asm.defaults_at is None
    assert asm.uid_at == 0x6C
    (node,) = asm.nodes()
    assert (node.start, node.end) == (0x50, 0x60)  # the old layout: at the table


HALF = 0x3F000000
"""The f32 0.5."""
OFF = -1
"""In a warp's words below: a code offset (to a code block after them)."""


def warp_file(flags: int, rest: list[int], *, version: int = 0, count: int = 100):
    """Return an SBAM file: count - 1 uniforms, then a warp with ``rest``.

    Its node word is ``flags << 16 | 0x88 << 8 | 14``; each OFF in ``rest``
    is an offset to a code block (an empty one) after its words.
    """
    start = 0x40 + 4 * (count - 1)
    blocks = start + 4 * (1 + len(rest))
    offsets = iter(range(blocks, blocks + 8, 4))
    words = [flags << 16 | 0x88 << 8 | 14]
    words += [next(offsets) - BASE if w == OFF else w for w in rest]
    words += [0, 0] if OFF in rest else []
    table = start + 4 * len(words)
    outputs = table + 4 * count
    data = bytearray(outputs + 8 + 8 + 16)
    data[:4] = b"SBAM"
    struct.pack_into("<H", data, 6, version)
    for k in range(count - 1):
        struct.pack_into("<I", data, 0x40 + 4 * k, 12)  # a uniform
    struct.pack_into(f"<{len(words)}I", data, start, *words)
    starts = [0x40 + 4 * k for k in range(count - 1)] + [start]
    struct.pack_into(f"<{count}I", data, table, *(s - BASE for s in starts))
    struct.pack_into("<HHI", data, outputs + 8, 1, 0, 0x1234)  # the uid table
    struct.pack_into("<4I", data, outputs + 16, count, table - BASE, outputs - BASE, 0)
    return Asm(bytes(data))


@pytest.mark.parametrize(
    ("flags", "rest", "version", "gradient", "intensity"),
    [
        # The cook version decides the flag word, never the look of rest[2]
        # (docs/node-seed.md section 2.2, warp note).
        # 1: v5/v6, code; the second offset is the pixel size's, so P bits
        # 2-3 = 2 (docs/node-params.md: the warp-layout correction)
        (0x2B19, [7, 8, OFF, OFF, HALF], 5, 8, 0.5),
        (0x2B19, [0, 7, 8, OFF, OFF, HALF], 9, 8, 0.5),  # 2: v9, code
        (0x2318, [7, 8, HALF], 6, 8, 0.5),  # 3: no code (persian)
        (0x2318, [0, 7, 8, HALF], 9, 8, 0.5),  # 4: v9, no code
        (0x0318, [7, 8], 5, 8, 1.0),  # 5: no code, default intensity
        (0x2318, [7, NO_NODE, HALF], 5, NO_NODE, 0.5),  # 6: gradient unwired
        (0x2318, [7, 8, HALF], 2, 8, 0.5),  # 3 in a version 2 cook (probes)
        (0x2318, [7, 8, 0], 5, 8, 0.0),  # +0.0 reads as a node: the version decides
    ],
)
def test_warp_layouts(
    flags: int, rest: list[int], version: int, gradient: int, intensity: float
) -> None:
    # docs/warp-layout.md section 4, the node count 100.
    warp = warp_file(flags, rest, version=version).nodes()[-1]
    # an unwired gradient (NO_NODE) is no input at all (Asm._node)
    wired = {} if gradient == NO_NODE else {"gradient": gradient}
    assert warp.inputs == {"input": 7, **wired}
    assert bool(warp.code) == bool(flags & 1)
    assert word_params(warp, [("intensity", 5, 1.0)], {}) == {"intensity": intensity}


def test_a_warp_without_code_in_a_file() -> None:
    # docs/warp-layout.md section 4's container: test 3's warp as node 2.
    data = bytearray(0x84)
    data[:4] = b"SBAM"
    struct.pack_into("<I", data, 0x40, 12)
    struct.pack_into("<I", data, 0x44, 12)
    struct.pack_into("<4I", data, 0x48, 0x2318 << 16 | 0x88 << 8 | 14, 0, 1, HALF)
    struct.pack_into("<3I", data, 0x60, 0x40 - BASE, 0x44 - BASE, 0x48 - BASE)
    struct.pack_into("<HHI", data, 0x6C, 1, 0, 0x1234)
    struct.pack_into("<4I", data, 0x74, 3, 0x60 - BASE, 0x64 - BASE, 0)
    warp = Asm(bytes(data)).nodes()[2]
    assert warp.inputs == {"input": 0, "gradient": 1}
    assert warp.code == []
    assert warp.inline[0] == HALF


def test_pixel_processor_input_count_is_16_bits() -> None:
    # Version 9 cooks (mosaic.sbsar's n5) set 0x20000 above a pixel
    # processor's input count: one input, node 4, then its code offsets.
    rest = [0x20001, 4, 0x3F48, 0x3C4C, 0x3C40]

    def is_offset(w: int) -> bool:
        return w > 0x1000

    assert pp_inputs(rest[0]) == 1
    assert field_count(40, rest, is_offset) == 2
    assert field_count(41, [0x20000, 0x2BD8], is_offset) == 1
    assert field_count(40, [3, 1, 2, 3], is_offset) == 4  # (older cooks)


def test_an_unwired_input_is_no_input() -> None:
    # mosaic.sbsar's pixel processors: an input of node 0xFFFFFFFF
    data = bytearray(0x84)
    data[:4] = b"SBAM"
    struct.pack_into("<HHI", data, 0x6C, 1, 0, 0x1234)
    struct.pack_into("<4I", data, 0x74, 2, 0x60 - BASE, 0x64 - BASE, 0)
    # node 0 at 0x40: a uniform (op 12, no inputs); node 1 at 0x48: a pixel
    # processor of two inputs, node 0 and none
    struct.pack_into("<I", data, 0x40, 12)
    struct.pack_into("<4I", data, 0x48, 40, 2, 0, UNWIRED)
    struct.pack_into("<2I", data, 0x60, 0x40 - BASE, 0x48 - BASE)
    nodes = Asm(bytes(data)).nodes()
    assert nodes[1].inputs == {"input": 0}


def test_raw_image() -> None:
    # A 2 x 1 RGB image stored raw (format 2) just before the node reading
    # it, as stripe-new-1.sbsar's: the node word's x is log2 h << 4 | log2 w.
    data = bytearray(0x84)
    data[:4] = b"SBAM"
    struct.pack_into("<HHI", data, 0x6C, 1, 0, 0x1234)
    struct.pack_into("<4I", data, 0x74, 1, 0x60 - BASE, 0x64 - BASE, 0)
    data[0x48:0x4E] = bytes([10, 20, 30, 40, 50, 60])
    asm = Asm(bytes(data))
    word = 0x0200 << 16 | 0x01 << 8 | 33
    node = Node(index=0, start=0x50, end=0x58, op=33, flags=0x0200,
                uid=0x48 - BASE, words=[word, 0x48 - BASE])  # fmt: skip
    raw = asm.raw_image(node)
    assert raw is not None
    assert (raw.width, raw.height, raw.channels, raw.depth) == (2, 1, 3, 1)
    assert raw.pixels == bytes([10, 20, 30, 40, 50, 60])
    # pixels that would run into the node are not this node's
    node.start = 0x4C
    assert asm.raw_image(node) is None


def one_node(words: list[int]) -> Node:
    """Parse a cook whose one node, at 0x40, is ``words``."""
    at, table = 0x40, 0x40 + 4 * len(words)
    data = bytearray(table + 36)
    data[:4] = b"SBAM"
    struct.pack_into("<H", data, 6, 5)
    struct.pack_into(f"<{len(words)}I", data, at, *words)
    struct.pack_into("<I", data, table, at - BASE)  # the node table
    struct.pack_into("<II", data, table + 4, 0xC0, 0)  # the outputs
    struct.pack_into("<HHI", data, table + 12, 1, 0, 0x1234)  # the uid table
    struct.pack_into("<4I", data, table + 20, 1, table - BASE, table + 4 - BASE, 0)
    (node,) = Asm(bytes(data)).nodes()
    return node


@pytest.mark.parametrize(
    ("flags", "blocks"), [(0x18, 1), (0x19, 2), (0x119, 2), (0x9, 2), (0x99, 3)]
)
def test_gradient_block_offsets(flags: int, blocks: int) -> None:
    # docs/gradient-map.md: after the input and the key count, the keys'
    # offset (with or without flag bit 0), the seed block's (bits 6-7 = 2),
    # the size function's (bit 0); then the keys, 6 bytes each with 0x100.
    # The seed's offset is Node.seed_code, not one of Node.code.
    at = 0x40 + 4 * (3 + blocks)
    keys = [0, 0x80000000, 0xFFFFFFFF] if flags & 0x100 else [0, 0xFFFFFFFF]
    offsets = [at - BASE] + [at + 4 * (len(keys) + i) - BASE for i in range(blocks - 1)]
    node = one_node([flags << 16 | 0x88 << 8, 7, 2, *offsets, *keys, *[0] * blocks])
    assert node.op == 0
    assert node.inputs == {"input": 7}
    assert node.fields == [7, 2]
    seeded = flags >> 6 & 3 == 2
    assert node.seed_code == (offsets[1] + BASE if seeded else None)
    kept = [o for i, o in enumerate(offsets) if not seeded or i != 1]
    assert node.code == [o + BASE for o in kept]
    assert node.size_code == (offsets[-1] + BASE if flags & 1 else None)
    assert node.inline == []


def test_blend_inputs_and_opacity() -> None:
    _, nodes, names = load("blend_gray_chain")
    inputs = by_input(nodes, names)
    blur = only(nodes, "blur")
    blend = only(nodes, "blend")
    assert blur.inputs == {"input": inputs["input_a"]}
    assert blend.inputs == {
        "destination": inputs["input_b"],
        "source": blur.index,
        "opacity": inputs["input_c"],
    }
    assert 0.37 in blend.constants()


@pytest.mark.parametrize(
    ("fixture", "filt", "first", "second"),
    [
        ("warp_gray_chain", "warp", "input", "gradient"),
        ("distance_gray_chain", "distance", "mask", "source"),
        ("shuffle_gray_chain", "shuffle", "input1", "input2"),
    ],
)
def test_two_input_filters(fixture: str, filt: str, first: str, second: str) -> None:
    _, nodes, names = load(fixture)
    inputs = by_input(nodes, names)
    blur = only(nodes, "blur")
    node = only(nodes, filt)
    assert node.inputs == {first: blur.index, second: inputs["input_b"]}


def test_levels_constant() -> None:
    _, nodes, names = load("levels_gray")
    levels = only(nodes, "levels")
    assert levels.inputs == {"input": by_input(nodes, names)["input_a"]}
    assert levels.constants()[0] == pytest.approx(0.1)


def test_uniform_color() -> None:
    _, nodes, _ = load("uniform_color")
    (node,) = nodes
    assert node.name == "uniform"
    assert node.constants() == pytest.approx([0.3, 0.4, 0.5, 1.0])


@pytest.mark.parametrize("fixture", sorted(p.stem for p in DATA.glob("*.sbsar")))
def test_outputs_reach_every_node(fixture: str) -> None:
    asm, nodes, _ = load(fixture)
    roots = [node for _, node in asm.output_nodes]
    assert reachable(nodes, roots) == set(range(len(nodes)))
    assert all(not n.name.startswith("op") for n in nodes)
