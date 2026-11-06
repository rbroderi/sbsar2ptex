"""The .sbsasm container: trailer, output map, defaults, uid table, node extents.

docs/sbsasm-container.md section 7: T1-T6 are its synthetic files (u32
words, stored little-endian), T7 the corpus cooks.
"""

import re
import struct
from pathlib import Path

import pytest

from sbsar2ptex.archive import member
from sbsar2ptex.archive import read_archive
from sbsar2ptex.sbsasm import BASE
from sbsar2ptex.sbsasm import Asm

CORPUS = Path(__file__).parent.parent / "corpus" / "sbsar"


def build(hexdump: str) -> bytes:
    """Return the file a section 7 dump describes (position, then u32 words)."""
    words = []
    for line in hexdump.strip().splitlines():
        _, *row = line.split()
        words += [int(w, 16) for w in row]
    return struct.pack(f"<{len(words)}I", *words)


def extents(asm: Asm) -> list[tuple[int, int]]:
    """Each node's (start, end) as offsets (relative to BASE)."""
    return [(n.start - BASE, n.end - BASE) for n in asm.nodes()]


T1 = """
0000  4d414253 00020000 00000000 00000000
0010  0000007c 00000000 00000000 00000000
0020  00000000 00000000 00000000 00000000
0030  00000000 00000000 00000014 0000001c
0040  000000c0 00000001 00088820 00001000
0050  13088814 00000000 40e80000 00010001
0060  00002000 00000005 00001000 00000002
0070  00000004 0000000c 00000000
"""


def test_t1_image_input_no_defaults() -> None:
    asm = Asm(build(T1))
    assert asm.uid_at - BASE == 0x28
    assert asm.defaults_at is None
    assert asm.output_uids == [0x2000]
    assert asm.output_nodes == [(0xC0, 1)]
    assert asm.inputs == [(0x1000, 5)]
    assert asm.defaults == {}
    assert extents(asm) == [(0x14, 0x1C), (0x1C, 0x28)]


T2 = """
0000  4d414253 00020000 00000000 00000000
0010  00000070 00000000 00000000 00000000
0020  00000000 00000000 00000000 00000000
0030  00000000 00000000 00000010 00000000
0040  00000000 0108880d 3e800000 3f000000
0050  3f400000 3f800000 00000001 00002000
0060  00000001 00000004 00000008 00000000
"""


def test_t2_no_inputs() -> None:
    asm = Asm(build(T2))
    assert asm.uid_at - BASE == 0x24
    assert asm.output_uids == [0x2000]
    assert asm.output_nodes == [(0, 0)]
    assert asm.inputs == []
    (node,) = asm.nodes()
    assert (node.start - BASE, node.end - BASE) == (0x10, 0x24)
    assert len(node.words) == 5


T3 = """
0000  4d414253 00090000 00000000 00000000
0010  0000008c 00000000 00000000 00000000
0020  00000000 00000000 00000000 00000000
0030  00000000 00000000 00000010 00000400
0040  00000000 0108880d 3e800000 3f000000
0050  3f400000 3f800000 00000008 00000008
0060  00000000 00020001 00002000 00000008
0070  00001000 00000004 00003000 00000001
0080  00000004 00000008 00000024
"""


def test_t3_defaults_that_read_as_a_header() -> None:
    asm = Asm(build(T3))
    assert asm.uid_at - BASE == 0x30  # not the defaults' start, 0x24
    assert asm.output_uids == [0x2000]
    assert asm.output_nodes == [(0x400, 0)]
    assert asm.inputs == [(0x1000, 8), (0x3000, 4)]
    assert asm.defaults == {0x1000: [8, 8], 0x3000: [0]}
    assert extents(asm) == [(0x10, 0x24)]


T4 = """
0000  4d414253 00090000 00000000 00000000
0010  000000a8 00000000 00000000 00000000
0020  00000000 00000000 00000000 00000000
0030  00000000 00000000 00000014 0000001c
0040  000000c0 00000001 00088820 00001000
0050  13088814 00000000 40e80000 00000008
0060  00000008 3e800000 3f000000 3f400000
0070  3f800000 00000007 00030001 00002000
0080  00000008 00000900 00000005 00001000
0090  00000004 00003000 00000002 00000004
00a0  0000000c 00000028
"""


def test_t4_image_default_version_9() -> None:
    asm = Asm(build(T4))
    assert asm.uid_at - BASE == 0x44
    assert asm.inputs == [(0x900, 8), (0x1000, 5), (0x3000, 4)]
    assert asm.defaults == {
        0x900: [8, 8],
        0x1000: [0.25, 0.5, 0.75, 1.0],
        0x3000: [7],
    }
    assert extents(asm) == [(0x14, 0x1C), (0x1C, 0x28)]


T5 = """
0000  4d414253 00020000 00000000 00000000
0010  00000098 00000000 00000000 00000000
0020  00000000 00000000 00000000 00000000
0030  00000000 00000000 00000014 0000001c
0040  000000c0 00000001 00088820 00001000
0050  13088814 00000000 40e80000 00000007
0060  00000008 00000008 00030001 00002000
0070  00000004 00000800 00000005 00001000
0080  00000008 00003000 00000002 00000004
0090  0000000c 00000028
"""


def test_t5_image_default_version_2() -> None:
    asm = Asm(build(T5))
    assert asm.uid_at - BASE == 0x34
    assert asm.inputs == [(0x800, 4), (0x1000, 5), (0x3000, 8)]
    assert asm.defaults == {0x800: [7], 0x3000: [8, 8]}  # the image: none
    assert extents(asm) == [(0x14, 0x1C), (0x1C, 0x28)]


T6 = """
0000  4d414253 00020000 00000000 00000000
0010  000000a0 00000000 00000000 00000000
0020  00000000 00000000 00000000 00000000
0030  00000000 00000000 0000001c 00000030
0040  00000000 00000000 00000000 00000001
0050  0108880d 3e800000 3f000000 3f400000
0060  3f800000 0108880d 3e800000 3f000000
0070  3f400000 3f800000 3f000000 00010002
0080  00002000 00002001 00000000 00001000
0090  00000002 00000004 0000000c 00000044
"""


def test_t6_two_outputs_one_float() -> None:
    asm = Asm(build(T6))
    assert asm.uid_at - BASE == 0x48
    assert asm.output_uids == [0x2000, 0x2001]
    assert asm.output_nodes == [(0, 0), (0, 1)]
    assert asm.inputs == [(0x1000, 0)]
    assert asm.defaults == {0x1000: [0.5]}
    assert extents(asm) == [(0x1C, 0x30), (0x30, 0x44)]


def test_no_table_is_reported() -> None:
    data = bytearray(build(T2))
    struct.pack_into("<HH", data, 0x24 + BASE, 2, 0)  # n_out 2, not K = 1
    with pytest.raises(ValueError, match="no uid table"):
        Asm(bytes(data))


CORPUS_FILES = sorted(CORPUS.glob("*.sbsar"))


@pytest.mark.skipif(not CORPUS_FILES, reason="no corpus")
@pytest.mark.parametrize("path", CORPUS_FILES, ids=lambda p: p.stem)
def test_t7_corpus_counts(path: Path) -> None:
    files = read_archive(path)
    xml = member(files, ".xml").decode("utf-8")
    outputs = [int(u) for u in re.findall(r'<output uid="(\d+)"', xml)]
    inputs = sorted(int(u) for u in re.findall(r'<input uid="(\d+)"', xml))
    asm = Asm(member(files, ".sbsasm"))
    assert asm.output_uids == outputs
    assert [uid for uid, _ in asm.inputs] == inputs
    assert len(asm.output_nodes) == len(outputs)
    for node in asm.nodes():
        assert node.start < node.end
