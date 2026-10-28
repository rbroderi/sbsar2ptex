"""Function bytecode: decoding, and choosing a node's blocks."""

from sbsar2ptex.function import BOOL
from sbsar2ptex.function import INT
from sbsar2ptex.function import decode
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
