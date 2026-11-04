"""Function bytecode: decoding, evaluating, and choosing a node's blocks."""

import struct

import numpy as np

from sbsar2ptex.function import BOOL
from sbsar2ptex.function import FLOAT
from sbsar2ptex.function import INT
from sbsar2ptex.function import NO_VALUE
from sbsar2ptex.function import Block
from sbsar2ptex.function import Instr
from sbsar2ptex.function import decode
from sbsar2ptex.function import evaluate
from sbsar2ptex.sbsasm import Node
from sbsar2ptex.translate import _seed_block


def block(*instrs: list[int]) -> list[int]:
    """Return a code block's u32 words: a u16 count, then the instructions."""
    half = [len(instrs)]
    for ins in instrs:
        half += ins
    half += [0] * (len(half) % 2)
    return [half[k] | half[k + 1] << 16 for k in range(0, len(half), 2)]


def load(kind: int, width: int, uid: int, *, pad: bool) -> list[int]:
    """Return an instruction reading graph input ``uid`` (padded to align it)."""
    ops = [0] * pad + [uid & 0xFFFF, uid >> 16]
    return [len(ops) << 10 | kind << 8 | (width - 1) << 6 | 0x02, *ops]


def test_a_bool_input_is_read_by_its_aligned_uid() -> None:
    # Count at u16 0, opcode at 1: the uid's u32 starts at u16 2, aligned...
    (aligned,) = decode(block(load(BOOL, 1, 0x1234ABCD, pad=False))).instrs
    assert aligned.u32s() == [0x1234ABCD]
    # ... and after another instruction (u16s 1-3) a u16 of padding first.
    first = load(INT, 1, 7, pad=False)
    _, padded = decode(block(first, load(BOOL, 1, 0x1234ABCD, pad=True))).instrs
    assert padded.u32s() == [0x1234ABCD]


def test_the_seed_is_the_block_of_one_integer() -> None:
    # Biomech's nodes keep the seed's block first and the output size's
    # after it, or the other way round: either way the seed is the int one.
    seed = block(load(INT, 1, 11, pad=False))
    size = block(load(INT, 2, 12, pad=False))
    for blocks in ((seed, size), (size, seed)):
        words = blocks[0] + blocks[1]
        node = Node(index=0, start=0, end=4 * len(words), words=words,
                    code=[0, 4 * len(blocks[0])])  # fmt: skip
        assert _seed_block(node) == (0 if blocks[0] is seed else 4 * len(blocks[0]))


def const(value: float) -> Instr:
    """Return a float constant instruction."""
    (word,) = struct.unpack("<I", struct.pack("<f", value))
    return Instr(op=0x00, type=FLOAT, width=1, operands=[word & 0xFFFF, word >> 16])


def test_a_dropped_sequence_result_is_its_first_parts_rounded() -> None:
    # pp_param_var's size: $outputsize + sequence(set v = (rand, rand),
    # (0, 0) dropped by the cooker). Substance renders it at 4096 x 4096 at
    # seed 0, v (0.625, 0.911), and 4096 x 2048 at seed 3, v (0.539, 0.376):
    # the size plus v to the nearest
    for v, size in (((0.625, 0.911), [12, 12]), ((0.539, 0.376), [12, 11])):
        instrs = [
            Instr(op=0x02, type=INT, width=2, operands=[7, 0]),
            const(v[0]),
            const(v[1]),
            Instr(op=0x0D, type=FLOAT, width=2, operands=[1, 2]),
            Instr(op=0x07, type=FLOAT, width=2, operands=[3, 0]),
            Instr(op=0x0C, type=INT, width=2, operands=[4, NO_VALUE]),
            Instr(op=0x12, type=INT, width=2, operands=[0, 5]),
        ]
        assert evaluate(Block(instrs), {7: [11, 11]}) == size


def test_mod_is_float32_each_step_rounded() -> None:
    # A Tile Sampler's hash, 1234 / 4325 * 3.435974e10 mod 1e6: Substance's
    # float32 a - b * floor(a / b), each step rounded (numpy's float32),
    # lands 192 from the exact (and float64) remainder, 449344.
    a = np.float32(1234.0) / np.float32(4325.0) * np.float32(3.435974e10)
    b = np.float32(1e6)
    mod = Instr(op=0x16, type=FLOAT, width=1, operands=[0, 1])
    got = evaluate(Block([const(float(a)), const(float(b)), mod]), {})
    assert got == [float(a - b * np.floor(a / b))] == [449536.0]
