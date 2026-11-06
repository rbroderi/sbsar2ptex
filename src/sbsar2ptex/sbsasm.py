"""Decompile a compiled Substance graph (.sbsasm) to a node listing.

    sbsasm-dump MATERIAL.sbsar [-o OUT_DIR] [--nodes N]

Prints the nodes (atomic filter, inputs, constant parameters, graph
parameters read) and which node makes each output; ``-o`` also writes the
embedded bitmaps, nodes.txt (every node, with a hex dump) and graph.dot.

Format notes (SBAM v2-v5; Designer 16's cooker and a 2018 one), checked
against graphs cooked by sbsasm-probe (sbsar2ptex.probe). Integers are little-endian;
an "offset" is relative to BASE (0x34), where offset 0 is a u32 0.

Header: "SBAM", u16 0, u16 version, u32 cooker build, u32 asmuid, u32 file
size, ... Bitmaps follow at 0x38, each a u32 length then the image file
(JPEG, PNG, ...), the next 4-byte aligned after it.

Some cooks store an image raw instead: its pixels (rows top down, channels
R G B A, 16-bit ones little-endian) just before the image input node that
reads it, whose uid is their offset. The node word's x is the size, log2
height << 4 | log2 width, and its flags >> 8 the format (RAW_FORMATS).

Trailer (the file's last 16 bytes): u32 node count, offset of the node
table, offset of the output map, offset of the defaults block.

Node table: an offset per node, to its node word. Output map: per graph
output, in the XML's order, (u32 format, u32 node index). Defaults block: the
graph inputs' default values, in uid order (as the uid table), images taking
no space. Then the uid table: u16 output count, u16 input count, the output
uids, then (u32 type, u32 uid) per input, sorted by uid, $randomseed and
$outputsize among them; it ends at the trailer.

Node: the node word ``flags << 16 | x << 8 | op`` (op: the atomic filter,
see OPS; x is often 0x88 but not always, meaning unknown), then fields
(flags, input node indices: see LAYOUTS), then offsets of the node's code
blocks (each pointing further into the node), then inline
data (constants; some ops keep an optional input index here), then the code.
Ops pair up, even for a grayscale image, odd for color.

The node's own random seed (docs/node-seed.md): flag bits 6-7 are its
state, 0 none (seed 0), 1 an inline i32, 2 the offset of a block returning
one integer. Its word, the seed slot, sits at a fixed place: after the
fields (and a gradient's keys' or a curve's points' offset), before the
output-size function's offset (flag bit 0: "has a size function", not "has
code"). Node.seed, Node.seed_code and Node.size_code hold what is found
there; the slot is left out of Node.code and Node.inline, which read as
for a node without a seed.

Code blocks: function bytecode, a u16 stream in SSA form. An opcode is
``type << 8 | (width - 1) << 6 | op``: type 0x09 float, 0x0a int, others
unknown; op 0x02 loads a graph input by its uid (u32), so ``0a42 xxxx
<$outputsize uid>`` loads the output size; 0x00 with type 0x09/0x0d is an f32
constant (4-byte aligned, padded with u16 0); most others take u16 operand
indices (earlier values).
"""

import argparse
import re
import struct
import sys
from collections import Counter
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from sbsar2ptex.archive import read_archive

BASE = 0x34
MAGICS = (b"\xff\xd8\xff", b"\x89PNG", b"DDS ", b"v/1\x01", b"II*\x00", b"MM\x00*")
UNWIRED = 0xFFFFFFFF
"""The node index of an input no node feeds (mosaic.sbsar's pixel processors)."""
RAW_FORMATS = {1: (1, 1), 2: (3, 1), 3: (4, 1), 5: (1, 2), 6: (3, 2), 7: (4, 2)}
"""A raw image's format: (channels, bytes per channel). Checked against
sbsrender's renders of previewsphere_sphere.sbsar (1, 3) and the sizes of
stripe-new-1's (1, 2) and persian-mosaic-wall-tiels's (5, 6, 7) images."""
ENCODED_FORMAT = 8
"""An image input's storage format for an encoded file: at its uid, a u32
length, then the file (docs/image-inputs.md)."""
DATA_START = 0x38
"""Where the embedded images' data area starts."""
LENGTH_SIZE = 4
"""An encoded image's length word."""


@dataclass
class RawImage:
    """An image stored raw in the bytecode (see the module notes)."""

    width: int
    height: int
    channels: int
    depth: int
    """Bytes per channel: 1 or 2."""
    pixels: bytes


_PAIRS = {
    2: "blend",
    4: "transformation",
    8: "fxmaps",
    12: "uniform",
    14: "warp",
    16: "emboss",
    20: "blur",
    22: "dirmotionblur",
    24: "directionalwarp",
    26: "sharpen",
    30: "levels",
    32: "input",
    38: "dyngradient",
    40: "pixelprocessor",
    42: "distance",
    44: "curve",
}
OPS: dict[int, str] = {
    **{op: f"{name}" for op, name in _PAIRS.items()},
    **{op + 1: f"{name}" for op, name in _PAIRS.items()},
    0: "gradient",
    1: "gradient",
    6: "grayscaleconversion",
    7: "shuffle",
    29: "hsl",
    37: "normal",
}
"""Atomic filter by op, from sbsasm-probe (sbsar2ptex.probe) (Designer 16.0.7's cooker).

9 (fxmaps, color) is inferred from the pairing; value processors seem to
cook to 40 too (unconfirmed). The cooker folds no-op nodes away
(passthrough never appears) and adds uniform nodes where an image changes
between grayscale and color.
"""
COLOR_OPS = {op + 1 for op in _PAIRS} | {1, 7, 29, 37}

BLEND_OPACITY = 0x30
"""Blend flag word, bits 4-5: the opacity's state, 3 for an opacity input,
its index the first inline word (1 an inline value, 2 a function; the low
nibble is the blending mode)."""

_LAYOUT = {
    # Fields before the code offsets: "-" a flag word, "#" a count, else the
    # input the field is (a node index).
    0: ("input", "#keys"),
    1: ("input", "#keys"),
    2: ("-", "destination", "source"),
    4: ("-", "input"),
    6: ("input",),
    7: ("-", "input1", "input2"),
    12: (),
    14: ("input", "gradient"),
    16: ("-", "input", "gradient"),
    20: ("input",),
    22: ("-", "input"),
    24: ("-", "input", "intensity"),
    26: ("input",),
    29: ("input",),
    30: ("-", "input"),
    32: ("uid",),
    37: ("-", "input"),
    38: ("input", "gradient"),
    42: ("-", "mask", "source"),
    44: ("input", "-"),
}
LAYOUTS = {**_LAYOUT, **{op + 1: v for op, v in _LAYOUT.items() if op in _PAIRS}}
"""Each op's fields before its code offsets (from sbsasm-probe (sbsar2ptex.probe)).

Not here: pixel processors (40/41: a count, then that many inputs), FX-Maps
(8/9: a flag word, a code offset, their pattern inputs, two code offsets,
the background input) and blend's optional opacity input (BLEND_OPACITY).
Warps have a leading flag word in version 9 cooks (three fields, not two:
see warp_flag_word); distance
has its source input only when its flag word's bit 0 is set. A node whose
node-word flags lack bit 0 has no size function; its other code offsets
(if any) are read with its inline words. Gradient maps are 0 (grayscale
out) and 1 (color out), both from either.
"""
CODE_COUNT = dict.fromkeys((2, 3, 4, 5, 6, 7, 12, 13, 29, 30, 31, 32, 33, 38, 39), 1)
"""Code offsets each op's node has after its seed slot, with node flag bit 0.

The output-size function's block. Without bit 0 none are read as code. Not
here: FX-Maps and pixel processors, whose layouts differ, gradient maps and
curves (DATA_OPS), and the neighborhood filters (NEIGHBOR_PARAMS), which
have no fixed count: they walk their parameter list.
"""
BOOL = 0
"""A NEIGHBOR_PARAMS width: a boolean, whose state is its value (no word)."""
_NEIGHBOR = {
    # op: (where the states are: "P" the node word's top byte, "F" the flag
    # word), then (name, bit, width) per parameter, in bit order
    14: ("P", (("intensity", 5, 1),)),
    16: ("F", (("intensity", 1, 1), ("angle", 3, 1), ("highlight", 5, 1),
               ("shadow", 7, 1))),
    17: ("F", (("intensity", 1, 1), ("angle", 3, 1), ("highlight", 5, 4),
               ("shadow", 7, 4))),
    20: ("P", (("intensity", 4, 1),)),
    22: ("F", (("intensity", 0, 1), ("angle", 2, 1))),
    24: ("F", (("intensity", 1, 1), ("angle", 3, 1))),
    26: ("P", (("intensity", 4, 1),)),
    37: ("F", (("intensity", 0, 1), ("inverty", 2, BOOL), ("alpha", 4, BOOL))),
    42: ("F", (("distance", 2, 1),)),
}  # fmt: skip
NEIGHBOR_PARAMS = {
    **_NEIGHBOR,
    **{op + 1: v for op, v in _NEIGHBOR.items() if op in _PAIRS and op != 16},  # noqa: PLR2004
}
"""The neighborhood filters' parameters (docs/node-params.md section 1.2).

After the size offset come the pixel-size vector (P bits 2-3, a float2),
then these, in bit order: per state, nothing (0), ``width`` f32 (1; a
boolean none) or one offset (2), interleaved as they come (section 2).
"""
PIXEL_SIZE_BIT = 2
"""P bits 2-3: the pixel-size vector's state (P bits 0-1: tiling, no word)."""
DATA_OPS = frozenset({0, 1, 44, 45})
"""Gradient maps and curves: their data's offset (keys, points) comes
right after their fields, always, then the seed slot, then (bit 0) the size
function's offset (docs/node-seed.md section 2.1)."""
UNSEEDED_OPS = frozenset({32, 33})
"""Graph-input nodes: never a seed slot."""
SEED_NONE, SEED_INLINE, SEED_CODE = 0, 1, 2
"""Node flag bits 6-7, the seed state (docs/node-seed.md section 1)."""
UINT32 = 0xFFFFFFFF


class UnsupportedError(Exception):
    """A node (op or parameter) the translator cannot express yet."""


IMAGE_COUNT = 0xFFFF
"""A pixel processor's first field, low 16 bits: its image-input count. A
Value Processor (docs/function-ops.md 2.4) is op 40 with its value's width
in the high 16 bits (value_width)."""


def value_width(node: "Node") -> int:
    """Return a Value Processor's value width; 0 for any other node.

    It has no image of its own: its function (its last code block) sets a
    value slot that later nodes' functions read.
    """
    if node.op != 40 or not node.fields:  # noqa: PLR2004
        return 0
    return node.fields[0] >> 16


def seed_state(flags: int) -> int:
    """Return node ``flags``' seed state, bits 6-7: 0, 1, 2 (or 3, unknown)."""
    return flags >> 6 & 3


def signed32(w: int) -> int:
    """Return ``w`` modulo 2^32 as a signed 32-bit integer."""
    w &= UINT32
    return w - (1 << 32) if w >> 31 else w


def fxmap_layout(
    rest: list[int], is_offset, flags: int, start: int, count: int | None = None
) -> tuple[list[int], int]:
    """Return an FX-Map's pattern inputs and where its header ends.

    ``rest`` is the node's words after its node word (the node at file
    offset ``start``): the parameter states, the tree root's offset, the
    patterns, the seed slot (state 1 or 2), the setup / size function's
    offset (flag bit 0). The header ends at E, the first offset word at or
    after rest[2] or the word the tree root points at, whichever is first;
    the words from rest[2] to E are the patterns, then in state 1 the seed,
    the last of them (docs/node-seed.md section 2.3). Returns the patterns
    and the index in ``rest`` of the seed slot (state 1 or 2), else of the
    setup offset or what follows the header.

    Raises UnsupportedError for state 3; for state 2 without an offset at
    E, or state 1 without a word for its seed; and, ``count`` (the node
    count) given, when the tree bounds the header and a pattern word is not
    a node index (below ``count``, or NO_NODE): inline values could then sit
    among them (section 8).
    """
    state = seed_state(flags)
    if state == 3:  # noqa: PLR2004
        msg = "FX-Map with seed state 3"
        raise UnsupportedError(msg)
    end = 2
    while end < len(rest) and not is_offset(rest[end]):
        end += 1
    tree = len(rest)
    if len(rest) > 1 and is_offset(rest[1]):
        tree = (rest[1] + BASE - start) // 4 - 1
    by_tree = tree < end
    end = max(min(end, tree), 2)
    run = rest[2:end]
    if state == SEED_CODE:
        if end >= len(rest) or not is_offset(rest[end]):
            msg = "FX-Map: its seed slot is not an offset"
            raise UnsupportedError(msg)
        return run, end
    if state == SEED_INLINE:
        if not run:
            msg = "FX-Map with an inline seed but no word for it"
            raise UnsupportedError(msg)
        run, end = run[:-1], end - 1
    unbounded = by_tree and count is not None
    if unbounded and any(w != NO_NODE and w >= count for w in run):
        msg = "FX-Map: header words that are not node indices before its tree"
        raise UnsupportedError(msg)
    return run, end


def fixed_size(node: "Node") -> tuple[int, int] | None:
    """Return a node's fixed size (log2 width, log2 height); None with bit 0.

    A node without an output-size function (flag bit 0 clear) packs its
    size in its node word's second byte: log2 height << 4 | log2 width.
    """
    if node.flags & 1 or not node.words:
        return None
    x = node.words[0] >> 8 & 0xFF
    return x & 15, x >> 4


TYPE_SIZES = {0: 4, 1: 8, 2: 12, 3: 16, 4: 4, 5: 0, 8: 8, 9: 12, 10: 16}
"""Bytes a graph input of each type takes in the defaults block."""


def pp_inputs(word: int) -> int:
    """Return a pixel processor's input count from its first field.

    The low 16 bits (IMAGE_COUNT): newer cooks (version 6 and 9) set bits
    above them, then the inputs and the code offsets follow as in older
    ones; a Value Processor's are its value's width (value_width).
    """
    return word & IMAGE_COUNT


WARP_FLAG_WORD = {2: False, 5: False, 6: False, 9: True}
"""Whether a warp has a leading flag word, by cook version (docs/node-seed.md
section 2.2, warp note); other versions not seen."""
NO_NODE = UNWIRED
"""An input no node feeds."""


def warp_flag_word(version: int | None) -> bool | None:
    """Return whether a warp's words start with a flag word; None if unknown.

    The cook version decides (WARP_FLAG_WORD), never the look of rest[2]:
    a state-1 seed there is no offset either (docs/node-seed.md 2.2).
    """
    return WARP_FLAG_WORD.get(version) if version is not None else None


def field_count(
    op: int,
    rest: list[int],
    is_offset,
    *,
    flags: int = 1,
    count: int = 0,
    version: int | None = None,
) -> int | None:
    """Return how many fields precede ``op``'s code offsets, if known.

    ``rest`` is the node's words after its node word. Fields can hold values
    that look like code offsets (a flag word of 0x40), so where the layout
    is known it decides, not the look of the words. The cook ``version``
    places a warp's optional flag word (warp_flag_word; an unknown version
    reads as none: Asm marks such a warp unsupported). ``flags`` and
    ``count`` are no longer used.
    """
    del flags, count
    if op in (40, 41):  # a count word (its low 16 bits), then that many inputs
        return min(1 + pp_inputs(rest[0]), len(rest)) if rest else 0
    if op not in LAYOUTS:
        return None
    n = len(LAYOUTS[op])
    if op in (42, 43) and rest and not rest[0] & 1:
        n -= 1  # distance without a source
    if op in (14, 15) and warp_flag_word(version):
        n += 1  # a warp's leading flag word: three fields
    return min(n, len(rest))


@dataclass
class Node:
    """One node: its op, inputs and what else is read from it."""

    index: int
    start: int
    end: int
    op: int = -1
    flags: int = 0
    fields: list[int] = field(default_factory=list)
    code: list[int] = field(default_factory=list)
    """Block offsets (file) after the fields, the seed slot left out: a
    gradient's keys or a curve's points first, then the size function's
    (size_code) with flag bit 0, then the op's others."""
    inline: list[int] = field(default_factory=list)
    inputs: dict[str, int] = field(default_factory=dict)
    uid: int | None = None
    reads: list[int] = field(default_factory=list)
    words: list[int] = field(default_factory=list)
    """All the node's u32 words, from its node word on."""
    seed: int = 0
    """The node's own seed when a constant (seed state 1), a signed i32; 0
    in state 0. In state 2 see seed_code."""
    seed_code: int | None = None
    """Seed state 2: the seed function's block (file offset), one integer."""
    size_code: int | None = None
    """Flag bit 0: the output-size function's block (file offset)."""
    pixel_size: list[float] | None = None
    """A neighborhood filter's pixel-size vector when constant (state 1)."""
    pixel_size_code: int | None = None
    """A neighborhood filter's pixel-size function's block (state 2): also
    in Node.code, after size_code."""
    params: dict[str, tuple[int, Any]] = field(default_factory=dict)
    """A neighborhood filter's parameters (NEIGHBOR_PARAMS), by name: (state,
    payload), the payload None (state 0; a boolean in state 0 or 1), its f32
    values (state 1) or its block's file offset (state 2). Its words are
    also Node.inline, in the same order (the pixel size's are not)."""
    unsupported: str | None = None
    """Why the node's layout could not be read (seed state 3...), if so."""

    @property
    def seed_state(self) -> int:
        """The seed state, node flag bits 6-7 (docs/node-seed.md section 1)."""
        return seed_state(self.flags)

    def blocks(self) -> list[int]:
        """Return every block offset found (code, the seed's), in file order."""
        extra = [o for o in (self.seed_code, self.size_code) if o is not None]
        return sorted({*self.code, *extra})

    def at(self, offset: int, count: int) -> list[int]:
        """Return ``count`` words at file ``offset`` (inside this node)."""
        k = (offset - self.start) // 4
        return self.words[k : k + count]

    @property
    def name(self) -> str:
        """The atomic filter, with its image type."""
        base = OPS.get(self.op, f"op{self.op}")
        return f"{base}{'' if self.op in COLOR_OPS else ' (gray)'}"

    def constants(self) -> list[float]:
        """Return the inline words as floats (where they look like ones).

        An FX-Map's inline words are its structure, not constants: none.
        """
        if self.op in (8, 9):
            return []
        out: list[float] = []
        for w in self.inline:
            f = struct.unpack("<f", struct.pack("<I", w))[0]
            out.append(round(f, 5) if w and 1e-6 < abs(f) < 1e7 else w)  # noqa: PLR2004
        return out


class Asm:
    """A parsed .sbsasm."""

    def __init__(self, data: bytes) -> None:
        """Read the container: trailer, tables, bitmaps."""
        self.d = data
        if data[:4] != b"SBAM":
            msg = "not an SBAM file"
            raise ValueError(msg)
        self.version = self.u16(6)
        self.cooker, self.asmuid = struct.unpack_from("<II", data, 8)
        self.count, table, outputs, defaults = struct.unpack_from(
            "<4I", data, len(data) - 16
        )
        # A graph without (non-image) inputs may have no defaults block
        # (offset 0, as in old cooks): its nodes end at the node table.
        self.table, self.outputs_at = table + BASE, outputs + BASE
        self.defaults_at = defaults + BASE if defaults else self.table
        self.bitmaps: list[tuple[int, int]] = []
        at = 0x38
        while at < self.table and data[at + 4 : at + 8].startswith(MAGICS):
            self.bitmaps.append((at + 4, self.u32(at)))
            at = (at + 4 + self.u32(at) + 3) & ~3
        # The uid table ends at the trailer, and the defaults before it are
        # its inputs' (a default can look like a table that ends there).
        end = len(data) - 16

        def table_at(at: int) -> bool:
            n_out, n_in = struct.unpack_from("<HH", data, at)
            if at + 4 + 4 * n_out + 8 * n_in != end:
                return False
            if not defaults:
                return n_in == 0
            pairs = at + 4 + 4 * n_out
            kinds = [self.u32(pairs + 8 * i) for i in range(n_in)]
            if any(k not in TYPE_SIZES for k in kinds):
                return True  # (sizes unknown: take it)
            return sum(TYPE_SIZES[k] for k in kinds) == at - self.defaults_at

        # (after the defaults; without them, after the outputs: earlier bytes,
        # a bitmap's, can look like a table that ends at the trailer)
        first = self.defaults_at if defaults else self.outputs_at
        at = next((a for a in range(first, end, 4) if table_at(a)), None)
        if at is None:
            msg = "no uid table before the trailer"
            raise ValueError(msg)
        n_out, n_in = struct.unpack_from("<HH", data, at)
        self.output_uids = [self.u32(at + 4 + 4 * i) for i in range(n_out)]
        pairs = at + 4 + 4 * n_out
        self.inputs = [
            (self.u32(pairs + 8 * i + 4), self.u32(pairs + 8 * i)) for i in range(n_in)
        ]
        """(uid, type) per graph input, in uid order."""
        self.output_nodes = [
            struct.unpack_from("<II", data, self.outputs_at + 8 * i)
            for i in range(n_out)
        ]
        self.defaults: dict[int, list[float | int]] = {}
        at = self.defaults_at
        for uid, kind in self.inputs:
            size = TYPE_SIZES.get(kind)
            if size is None:
                break
            fmt = "<" + ("f" if kind < 4 else "i") * (size // 4)  # noqa: PLR2004
            self.defaults[uid] = list(struct.unpack_from(fmt, data, at))
            at += size

    def raw_image(self, node: "Node") -> RawImage | None:
        """Return the raw image input ``node`` reads, if it reads one.

        Its pixels must fit between BASE and the node (as they lie just
        before it): a uid of a graph input or bitmap file does not.
        """
        if node.op not in (32, 33) or node.uid is None or not node.words:
            return None
        form = RAW_FORMATS.get(node.flags >> 8)
        if form is None:
            return None
        sizes = node.words[0] >> 8 & 0xFF
        width, height = 1 << (sizes & 0xF), 1 << (sizes >> 4)
        start = node.uid + BASE
        size = width * height * form[0] * form[1]
        if start <= BASE or start + size > node.start:
            return None
        return RawImage(width, height, *form, self.d[start : start + size])

    def encoded_image(self, node: "Node") -> tuple[int, int] | None:
        """Return the encoded file (JPEG...) image input ``node`` reads, if so.

        docs/image-inputs.md: storage format 8 (the node word's top byte),
        at offset ``uid`` a u32 length, then the file's bytes; resolved by
        the uid, never by a walk of the data area. Returns (file offset,
        length); None for another node or format. One that does not end
        before the node or has no known image header is malformed:
        image_problem says so.
        """
        if node.op not in (32, 33) or node.uid is None or not node.words:
            return None
        if node.flags >> 8 != ENCODED_FORMAT:
            return None
        at = node.uid + BASE
        if at < DATA_START or at + LENGTH_SIZE > node.start:
            return None
        length = self.u32(at)
        start = at + LENGTH_SIZE
        if start + length > node.start or not self.d[start : start + length].startswith(
            MAGICS
        ):
            return None
        return start, length

    def image_problem(self, node: "Node") -> str | None:
        """Return why embedded image input ``node``'s image cannot be read, if so.

        docs/image-inputs.md section 3: formats 1-7 raw (raw_image), 8 an
        encoded file (encoded_image), others unsupported.
        """
        fmt = node.flags >> 8
        if fmt in RAW_FORMATS:
            if self.raw_image(node) is None:
                return f"node {node.index}: malformed raw image at uid {node.uid}"
            return None
        if fmt == ENCODED_FORMAT:
            if self.encoded_image(node) is None:
                return f"node {node.index}: malformed encoded image at uid {node.uid}"
            return None
        return f"node {node.index}: image storage format {fmt} (unsupported)"

    def u16(self, at: int) -> int:
        """Return the u16 at ``at``."""
        return struct.unpack_from("<H", self.d, at)[0]

    def u32(self, at: int) -> int:
        """Return the u32 at ``at``."""
        return struct.unpack_from("<I", self.d, at)[0]

    def nodes(self) -> list[Node]:
        """Every node, its fields read by its op's layout."""
        starts = [self.u32(self.table + 4 * i) + BASE for i in range(self.count)]
        bounds = sorted(starts)
        nxt = dict(zip(bounds, [*bounds[1:], self.defaults_at], strict=True))
        known = {uid for uid, _ in self.inputs}
        return [self._node(i, s, nxt[s], known) for i, s in enumerate(starts)]

    def _node(self, index: int, start: int, end: int, known: set[int]) -> Node:
        node = Node(index, start, end)
        words = [self.u32(o) for o in range(start, end, 4)]
        node.words = words
        word, rest = words[0], words[1:]
        node.op, node.flags = word & 0xFF, word >> 16

        def is_offset(w: int) -> bool:
            return start < w + BASE <= end and w % 4 == 0

        if node.op in (8, 9):
            k = self._fxmap_words(node, rest, is_offset)
        else:
            k = self._words(node, rest, is_offset)
        walked = node.op in NEIGHBOR_PARAMS and len(node.params) == len(
            NEIGHBOR_PARAMS[node.op][1]
        )  # (Asm._neighbor_words set its inline words)
        if not walked:
            blocks = node.blocks()
            stop = (blocks[0] - start) // 4 - 1 if blocks else len(rest)
            node.inline = rest[k:stop]
        node.reads = [w for w in words if w in known]
        self._inputs(node)
        if node.op in (32, 33) and node.uid not in known and node.unsupported is None:
            # an embedded image, not a graph input: found by its uid
            node.unsupported = self.image_problem(node)
        # An input no node feeds (UNWIRED): not an input at all, so that
        # what reads it sees nothing, as Substance's empty input.
        node.inputs = {k: v for k, v in node.inputs.items() if v != UNWIRED}
        return node

    def _words(self, node: Node, rest: list[int], is_offset) -> int:
        """Read an ordinary node's fields, seed slot and code offsets.

        In order (docs/node-seed.md section 2.1): the fields; a gradient's
        or curve's data offset; the seed slot (seed state 1 or 2), read by
        its position, never by its look; the size function's offset (flag
        bit 0); the op's other offsets. Returns the index in ``rest`` where
        the inline words start.
        """
        k = field_count(
            node.op, rest, is_offset,
            flags=node.flags, count=self.count, version=self.version,
        )  # fmt: skip
        known = k is not None
        if k is None:  # unknown layout: up to the first code offset
            k = 0
            while k < len(rest) and not is_offset(rest[k]):
                k += 1
        node.fields = rest[:k]
        if node.op in (14, 15) and warp_flag_word(self.version) is None:
            node.unsupported = (
                f"node {node.index}: a warp in cook version {self.version},"
                " whose leading flag word is unknown"
            )
        if known and node.op in DATA_OPS and k < len(rest) and is_offset(rest[k]):
            node.code.append(rest[k] + BASE)
            k += 1
        state = node.seed_state
        if known and node.op not in UNSEEDED_OPS:
            if state == 3:  # noqa: PLR2004
                node.unsupported = f"node {node.index}: seed state 3"
            elif state in (SEED_INLINE, SEED_CODE):
                k = self._seed_slot(node, rest, k, is_offset)
        if known and node.flags & 1 and k < len(rest) and is_offset(rest[k]):
            node.size_code = rest[k] + BASE
            node.code.append(node.size_code)
            k += 1
        if known and node.op in NEIGHBOR_PARAMS:
            return self._neighbor_words(node, rest, k, is_offset)
        fixed = CODE_COUNT.get(node.op)
        if fixed is not None:
            # Offsets after a set number of code offsets (none without bit 0)
            # belong to parameters (function blocks): inline.
            for _ in range(fixed - 1 if node.flags & 1 else 0):
                if k < len(rest) and is_offset(rest[k]):
                    node.code.append(rest[k] + BASE)
                    k += 1
        elif node.op not in DATA_OPS:
            while k < len(rest) and is_offset(rest[k]):
                node.code.append(rest[k] + BASE)
                k += 1
        return k

    @staticmethod
    def _neighbor_words(node: Node, rest: list[int], k: int, is_offset) -> int:  # noqa: PLR0911
        """Walk a neighborhood filter's parameter words from ``rest[k]``.

        docs/node-params.md section 2: the pixel-size vector, then the op's
        parameters in bit order (NEIGHBOR_PARAMS), each nothing, its f32
        values or one offset by its state. They must end exactly at the
        node's first block (or its end, with none). Returns the index where
        the inline words (the parameters') start; Asm._node takes them up to
        the first block. A state 3, a state-2 word that is no offset or a
        region that does not end at the first block marks the node
        unsupported.
        """
        where, spec = NEIGHBOR_PARAMS[node.op]
        top = node.flags >> 8
        states = top if where == "P" else (node.fields[0] if node.fields else 0)

        def bad(why: str) -> int:
            node.unsupported = f"node {node.index}: {why}"
            return min(k, len(rest))

        def f32s(words: list[int]) -> list[float]:
            return [struct.unpack("<f", struct.pack("<I", w))[0] for w in words]

        pixel = top >> PIXEL_SIZE_BIT & 3
        if pixel == 3:  # noqa: PLR2004
            return bad("pixel size in state 3 (unsupported)")
        if pixel == 1:
            if k + 2 > len(rest):
                return bad("malformed: no words for its pixel size")
            node.pixel_size = f32s(rest[k : k + 2])
            k += 2
        elif pixel == 2:  # noqa: PLR2004
            if k >= len(rest) or not is_offset(rest[k]):
                return bad("malformed: its pixel size's word is not an offset")
            node.pixel_size_code = rest[k] + BASE
            node.code.append(node.pixel_size_code)
            k += 1
        first = k
        offsets = []
        for name, bit, width in spec:
            state = states >> bit & 3
            if state == 3:  # noqa: PLR2004
                return bad(f"{name} in state 3 (unsupported)")
            if state == 0 or (width == BOOL and state == 1):
                node.params[name] = (state, None)
            elif state == 1:
                if k + width > len(rest):
                    return bad(f"malformed: no words for its {name}")
                node.params[name] = (1, f32s(rest[k : k + width]))
                k += width
            else:
                if k >= len(rest) or not is_offset(rest[k]):
                    return bad(f"malformed: its {name}'s word is not an offset")
                node.params[name] = (2, rest[k] + BASE)
                offsets.append(rest[k] + BASE)
                k += 1
        node.inline = rest[first:k]
        blocks = [*node.blocks(), *offsets]
        end = (min(blocks) - node.start) // 4 - 1 if blocks else len(rest)
        if k != end:
            bad(f"malformed: its parameter words end at word {k + 1}, not {end + 1}")
        return first

    @staticmethod
    def _seed_slot(node: Node, rest: list[int], k: int, is_offset) -> int:
        """Read the seed slot at ``rest[k]`` (state 1 or 2); return the next index."""
        if k >= len(rest):
            node.unsupported = f"node {node.index}: no word for its seed slot"
            return k
        if node.seed_state == SEED_INLINE:
            node.seed = signed32(rest[k])
        elif is_offset(rest[k]):
            node.seed_code = rest[k] + BASE
        else:
            node.unsupported = f"node {node.index}: its seed slot is not an offset"
        return k + 1

    def _fxmap_words(self, node: Node, rest: list[int], is_offset) -> int:
        """Read an FX-Map's words (fxmap_layout); return where its inline words start.

        Its fields are the parameter-states word; its code the tree root's
        offset, then the setup / size function's. The patterns and the
        background are its inputs (Asm._inputs).
        """
        node.fields = rest[:1]
        if len(rest) > 1 and is_offset(rest[1]):
            node.code.append(rest[1] + BASE)
        try:
            patterns, k = fxmap_layout(
                rest, is_offset, node.flags, node.start, self.count
            )
        except UnsupportedError as e:
            node.unsupported = f"node {node.index}: {e}"
            # Inputs as well as can be told: node indices before the next offset.
            k = 2
            while k < len(rest) and not is_offset(rest[k]) and rest[k] < node.index:
                node.inputs[f"pattern{k - 2}"] = rest[k]
                k += 1
            return len(rest)
        for n, w in enumerate(patterns):
            node.inputs[f"pattern{n}"] = w
        if node.seed_state in (SEED_INLINE, SEED_CODE):
            k = self._seed_slot(node, rest, k, is_offset)
        if node.flags & 1 and k < len(rest) and is_offset(rest[k]):
            node.size_code = rest[k] + BASE
            node.code.append(node.size_code)
            k += 1
        if rest[0] & 3 == 3 and k < len(rest):  # noqa: PLR2004 (a background input)
            node.inputs["background"] = rest[k]
        return k

    def _inputs(self, node: Node) -> None:
        """Name the node's inputs from its op's layout."""
        op, fields = node.op, node.fields
        if op in LAYOUTS:
            layout = LAYOUTS[op]
            if op in (14, 15) and len(fields) == 3:  # noqa: PLR2004
                layout = ("-", *layout)  # a leading flag word (field_count)
            if op in (42, 43) and fields and not fields[0] & 1:
                layout = layout[:2]  # no source input
            for name, value in zip(layout, fields, strict=False):
                if name == "uid":
                    node.uid = value
                elif name[0] not in "-#":
                    node.inputs[name] = value
            # Opacity state, bits 4-5: 1 inline, 2 a function, 3 an input.
            opacity = fields[0] & BLEND_OPACITY if op in (2, 3) and fields else 0
            if opacity == BLEND_OPACITY and node.inline:
                node.inputs["opacity"] = node.inline[0]
        elif op in (40, 41) and fields:
            for n, value in enumerate(fields[1 : 1 + pp_inputs(fields[0])]):
                node.inputs[f"input{n or ''}"] = value


def reachable(nodes: list[Node], roots: list[int]) -> set[int]:
    """Return the nodes ``roots`` depend on, themselves included."""
    seen: set[int] = set()
    todo = list(roots)
    while todo:
        k = todo.pop()
        if k in seen or not 0 <= k < len(nodes):
            continue
        seen.add(k)
        todo += nodes[k].inputs.values()
    return seen


def main() -> None:
    """Dump the archive the command line names."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sbsar", type=Path)
    parser.add_argument(
        "-o", "--out", type=Path, help="write bitmaps, nodes.txt, graph.dot here"
    )
    parser.add_argument(
        "--nodes", type=int, default=40, help="list this many nodes (0: all)"
    )
    args = parser.parse_args()

    files = read_archive(args.sbsar)
    data = next(d for n, d in files.items() if n.endswith(".sbsasm"))
    xml = next(d for n, d in files.items() if n.endswith(".xml")).decode("utf-8")
    names = {int(u): n for u, n in re.findall(r'uid="(\d+)" identifier="([^"]*)"', xml)}
    try:
        asm = Asm(data)
    except ValueError as e:
        sys.exit(f"{args.sbsar}: {e}")
    nodes = asm.nodes()

    def uid_name(uid: int) -> str:
        return names.get(uid, f"{uid:#010x}")

    print(f"SBAM v{asm.version} cooker {asm.cooker} asmuid {asm.asmuid}")
    print(f"{len(asm.bitmaps)} bitmaps, {asm.count} nodes")
    print("graph inputs:")
    for uid, kind in asm.inputs:
        print(
            f"  {uid_name(uid):40} type {kind:2} default {asm.defaults.get(uid, '?')}"
        )
    ops = Counter(n.name for n in nodes)
    print("nodes by filter:", ", ".join(f"{name} {c}" for name, c in ops.most_common()))
    roots = [node for _, node in asm.output_nodes]
    used = reachable(nodes, roots)
    print(f"outputs depend on {len(used)} of {len(nodes)} nodes")
    unknown = sorted({n.op for n in nodes if n.op not in OPS})
    if unknown:
        print("unidentified ops:", unknown)
    print("outputs:")
    for uid, (fmt, node) in zip(asm.output_uids, asm.output_nodes, strict=True):
        print(f"  {uid_name(uid):20} <- n{node} {nodes[node].name}  (format {fmt:#x})")

    def describe(n: Node) -> str:
        parts = [f"n{n.index}: {n.name}"]
        if n.uid is not None:
            parts.append(f"[{uid_name(n.uid)}]")
        if n.inputs:
            parts.append(", ".join(f"{k}=n{v}" for k, v in n.inputs.items()))
        consts = n.constants()
        if consts:
            parts.append(f"const {consts[:12]}{' ...' if len(consts) > 12 else ''}")  # noqa: PLR2004
        reads = sorted({uid_name(u) for u in n.reads} - {"$outputsize"})
        if reads:
            parts.append(f"reads {reads}")
        return "  ".join(parts)

    shown = nodes if args.nodes == 0 else nodes[: args.nodes]
    print(f"nodes{'' if args.nodes == 0 else f' (first {len(shown)})'}:")
    for n in shown:
        print(" ", describe(n))

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for k, (at, length) in enumerate(asm.bitmaps):
            suffix = ".png" if data[at : at + 4] == b"\x89PNG" else ".jpg"
            (args.out / f"bitmap_{k:02d}{suffix}").write_bytes(data[at : at + length])
        with (args.out / "nodes.txt").open("w", encoding="utf-8") as f:
            for n in nodes:
                hexdump = " ".join(
                    f"{asm.u32(o):08x}" for o in range(n.start, n.end, 4)
                )
                f.write(f"{describe(n)}\n  @{n.start:#x} {hexdump}\n")
        with (args.out / "graph.dot").open("w", encoding="utf-8") as f:
            f.write("digraph sbsasm {\n rankdir=LR; node [shape=box,fontsize=9];\n")
            for n in nodes:
                if n.index in used:
                    label = n.name + (
                        f"\\n{uid_name(n.uid)}" if n.uid is not None else ""
                    )
                    f.write(f' n{n.index} [label="n{n.index}\\n{label}"];\n')
                    f.writelines(
                        f' n{i} -> n{n.index} [label="{k}",fontsize=7];\n'
                        for k, i in n.inputs.items()
                    )
            for uid, (_, node) in zip(asm.output_uids, asm.output_nodes, strict=True):
                out = uid_name(uid)
                f.write(f' "{out}" [shape=ellipse]; n{node} -> "{out}";\n')
            f.write("}\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
