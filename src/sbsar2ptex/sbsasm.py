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

from sbsar2ptex.archive import read_archive

BASE = 0x34
MAGICS = (b"\xff\xd8\xff", b"\x89PNG", b"DDS ", b"v/1\x01", b"II*\x00", b"MM\x00*")

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

BLEND_OPACITY = 0x20
"""Blend flag: an opacity input, its index the first inline word (the flag
word's low nibble is the blending mode)."""

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
Warps sometimes have a leading flag word (three fields, not two); distance
has its source input only when its flag word's bit 0 is set. A node whose
node-word flags lack bit 0 has no code, its constants following its fields.
Gradient maps are 0 (grayscale out) and 1 (color out), both from either.
"""
TYPE_SIZES = {0: 4, 1: 8, 2: 12, 3: 16, 4: 4, 5: 0, 8: 8, 9: 12, 10: 16}
"""Bytes a graph input of each type takes in the defaults block."""


def field_count(op: int, rest: list[int], is_offset) -> int | None:
    """Return how many fields precede ``op``'s code offsets, if known.

    ``rest`` is the node's words after its node word. Fields can hold values
    that look like code offsets (a flag word of 0x40), so where the layout
    is known it decides, not the look of the words.
    """
    if op in (40, 41):
        return 1 + rest[0] if rest else 0
    if op not in LAYOUTS:
        return None
    n = len(LAYOUTS[op])
    if op in (42, 43) and rest and not rest[0] & 1:
        n -= 1  # distance without a source
    if op in (14, 15) and len(rest) > n and not is_offset(rest[n]):
        n += 1  # a warp's optional leading flag word: three fields
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
    inline: list[int] = field(default_factory=list)
    inputs: dict[str, int] = field(default_factory=dict)
    uid: int | None = None
    reads: list[int] = field(default_factory=list)
    words: list[int] = field(default_factory=list)
    """All the node's u32 words, from its node word on."""

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
            sys.exit("not an SBAM file")
        self.version = self.u16(6)
        self.cooker, self.asmuid = struct.unpack_from("<II", data, 8)
        self.count, table, outputs, defaults = struct.unpack_from(
            "<4I", data, len(data) - 16
        )
        self.table, self.outputs_at, self.defaults_at = (
            table + BASE,
            outputs + BASE,
            defaults + BASE,
        )
        self.bitmaps: list[tuple[int, int]] = []
        at = 0x38
        while at < self.table and data[at + 4 : at + 8].startswith(MAGICS):
            self.bitmaps.append((at + 4, self.u32(at)))
            at = (at + 4 + self.u32(at) + 3) & ~3
        # The uid table ends at the trailer.
        end = len(data) - 16
        for at in range(self.defaults_at, end, 4):
            n_out, n_in = struct.unpack_from("<HH", data, at)
            if at + 4 + 4 * n_out + 8 * n_in == end:
                break
        else:
            sys.exit("no uid table before the trailer")
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

        k = field_count(node.op, rest, is_offset)
        if k is None:  # unknown layout: up to the first code offset
            k = 0
            while k < len(rest) and not is_offset(rest[k]):
                k += 1
        node.fields = rest[:k]
        while k < len(rest) and is_offset(rest[k]):
            node.code.append(rest[k] + BASE)
            k += 1
        stop = (min(node.code) - start) // 4 - 1 if node.code else len(rest)
        node.inline = rest[k:stop]
        node.reads = [w for w in words if w in known]
        self._inputs(node)
        return node

    def _inputs(self, node: Node) -> None:
        """Name the node's inputs from its op's layout."""
        op, fields = node.op, node.fields
        if op in LAYOUTS:
            layout = LAYOUTS[op]
            if op in (14, 15) and len(fields) == 3:  # noqa: PLR2004
                layout = ("-", *layout)  # a leading flag word, sometimes
            if op in (42, 43) and fields and not fields[0] & 1:
                layout = layout[:2]  # no source input
            for name, value in zip(layout, fields, strict=False):
                if name == "uid":
                    node.uid = value
                elif name[0] not in "-#":
                    node.inputs[name] = value
            if op in (2, 3) and fields and fields[0] & BLEND_OPACITY and node.inline:
                node.inputs["opacity"] = node.inline[0]
        elif op in (40, 41) and fields:
            for n, value in enumerate(fields[1 : 1 + fields[0]]):
                node.inputs[f"input{n or ''}"] = value
        elif op in (8, 9):
            self._fxmap_inputs(node)

    def _fxmap_inputs(self, node: Node) -> None:
        """Read an FX-Map's inputs: [flags, code, patterns, code, code, background].

        A background of node 0 reads the same as none, so only others are taken.
        """
        words = [self.u32(o) for o in range(node.start + 4, node.end, 4)]

        def is_offset(w: int) -> bool:
            return node.start < w + BASE <= node.end and w % 4 == 0

        k = 1
        while k < len(words) and not is_offset(words[k]):
            k += 1
        k += 1
        n = 0
        while k < len(words) and not is_offset(words[k]) and words[k] < node.index:
            node.inputs[f"pattern{n}"] = words[k]
            n += 1
            k += 1
        while k < len(words) and is_offset(words[k]):
            k += 1
        if k < len(words) and 0 < words[k] < node.index:
            node.inputs["background"] = words[k]


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
    asm = Asm(data)
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
