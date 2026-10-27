"""FX-Maps: read a compiled FX-Map, and its patterns in GLSL.

An FX-Map (ops 8/9) draws patterns by a tree of Quadrant, Iterate and
Switch nodes. Its third word is the tree's root; each tree node is a word, then
offsets (from the file's BASE) and values, found by sbscooker probes
(sbsar2ptex.probe.fx):

Quadrant, low nibble 8:

    0x10        its own random seed, first of its parameters
    0x40        draws a pattern, id bits 8-15 (pattern type - 2: 0 square...)
    0x80        draws an image (a pattern input), its index bits 8-15
    bits 16-17  children: 1, four slots (0 for none); 2, one child on all four
    bit 18      blending: 0 add, 1 max (gray) or alpha blending (color)
    bits 19-30  parameter states, 2 bits each: opacity (4 floats, gray or
                not), branch offset (2), frame offset (2), pattern size (2),
                rotation (1), pattern supplement (1); a gray FX-Map's
                inline opacity is 1 float: 0 default, 1 inline,
                2 a function (code block offset)
    bit 31      an image's flags word follows the node word: filtering << 2,
                0x10 premultiplied, 0x2 its index a function

then the parameters' offset (if any are set), then the children. At the
parameters' offset: the seed, then each parameter set, in that order, then
the image index's code (if a function).

Iterate, low byte 0x0b (0x1b, its own seed; 0x4b, output0 used; 0x8b, its
count a function): the count (word >> 8, when not a function), then the seed,
the count's code, output1, then output0 if used.

Switch, low byte 0x09 (0x19, its own seed; 0x49, constant true; 0x89, a
function): the seed, the function's code, then output0 and output1.

The FX-Map node itself: its node word (bits 24-25 tiling, horizontal and
vertical), a word of parameter states (2 bits each: background, hurst, global
opacity, render and tiling regions...; 3 for the background, an input
image), code[0], the pattern inputs' node indices, two more code offsets (the
random seed's and the output size's), then the inline values in state order.

How it draws (fitted to sbsrender): a Quadrant's cell is the image at the
root and each child's a quarter of its parent's (output0 top left, 1 top
right, 2 bottom left, 3 bottom right). $pos is the cell's center plus the
branch offsets of the Quadrant and its ancestors; the pattern is drawn at
$pos plus the frame offset, pattern size times the cell wide, rotated
clockwise by rotation (turns), the image wrapping around. Offsets are in
image units. A pattern at depth k is weighted hurst^k over the sum of hurst^j
for j to the deepest level (all 1 for hurst 0), times the global opacity;
gray adds (clamped at the end) or takes the max, of opacity's green
component times the pattern. Iterate runs output0 once, then output1 count
times with $number 0, 1, ...; Switch runs output1 when its value is true,
else output0.
"""

import struct
from dataclasses import dataclass
from dataclasses import field

from sbsar2ptex.sbsasm import BASE
from sbsar2ptex.sbsasm import Node

PATTERNS = {
    0: "square", 1: "disc", 2: "paraboloid", 3: "bell", 4: "gaussian",
    5: "thorn", 6: "pyramid", 7: "brick", 8: "gradation", 9: "waves",
    10: "half bell", 11: "ridged bell", 12: "crescent", 13: "capsule",
    14: "cone",
}  # fmt: skip
"""Pattern names by compiled id (the .sbs pattern type less 2)."""

PATTERN_GLSL = """float fxss_$(name)(float e, float x) {
\tfloat t = clamp(x / max(e, 1e-6), 0.0, 1.0);
\treturn t * t * (3.0 - 2.0 * t);
}
float fxpat_$(name)(int id, vec2 q, float s) {
\tif (abs(q.x) > 1.0 || abs(q.y) > 1.0) return 0.0;
\tfloat r2 = dot(q, q);
\tfloat r = sqrt(r2);
\tfloat bell = pow(max(1.0 - r2, 0.0), 4.0);
\tif (id == 0) return 1.0;
\tif (id == 1) return r <= 1.0 ? 1.0 : 0.0;
\tif (id == 2) return max(1.0 - r2, 0.0);
\tif (id == 3) return bell;
\tif (id == 4) return exp(-11.1 * r2);
\tif (id == 5) return pow(max(1.0 - r, 0.0), 4.0);
\tif (id == 6) return max(1.0 - max(abs(q.x), abs(q.y)), 0.0);
\tif (id == 7) {
\t\treturn fxss_$(name)(s, 1.0 - abs(q.x)) * fxss_$(name)(s, 1.0 - abs(q.y));
\t}
\tif (id == 8) return (q.x + 1.0) * 0.5 * (1.0 - s);
\tif (id == 9) {
\t\tfloat f = (3.0 + 12.0 * s) * 3.14159265;
\t\treturn max(1.0 - r2, 0.0) * (0.5 + 0.5 * cos(f * q.x));
\t}
\tif (id == 10) return q.x > 0.0 ? bell * (1.0 - fxss_$(name)(1.0 - s, q.x)) : bell;
\tif (id == 11) return bell * (1.0 - abs(q.x)) * (1.0 - abs(q.x));
\tif (id == 12) {
\t\tfloat ring = r <= 1.0 ? 6.75 * r2 * (1.0 - r2) * (1.0 - r2) : 0.0;
\t\treturn ring * (1.0 - fxss_$(name)(2.0 - 2.0 * s, q.x + 1.0));
\t}
\tif (id == 13) {
\t\tfloat d = max(abs(q.x) - 0.77 * s, 0.0);
\t\treturn max(1.0 - 20.0 * (d * d + q.y * q.y), 0.0);
\t}
\tif (id == 14) return max(1.0 - r, 0.0);
\tif (id == 15) return (q.x + 1.0) * 0.5 + s;
\treturn r <= 1.0 ? sqrt(1.0 - r2) + s : 0.0;
}
"""
"""Each pattern's value at q in [-1, 1]^2 (x right, y down), supplement s."""

QUAD_PARAMS = (
    ("opacity", 19, 4),
    ("branchoffset", 21, 2),
    ("frameoffset", 23, 2),
    ("patternsize", 25, 2),
    ("patternrotation", 27, 1),
    ("patternsuppl", 29, 1),
)
"""A Quadrant's parameters: name, state bit, float count."""

INLINE, CODE, INPUT = 1, 2, 3
"""Parameter states: a value inline, a function's code block, an input."""


@dataclass
class Param:
    """A tree node's parameter: inline floats, or a code block's offset."""

    state: int
    value: list[float] = field(default_factory=list)
    code: int = 0


@dataclass
class Quadrant:
    """Draws a pattern in its cell; its children draw in its four quarters."""

    pattern: int | None = None
    image: int | None = None
    """The pattern input drawn, when an image."""
    image_index: "Param | None" = None
    """The image's index, when a function."""
    blend_max: bool = False
    seed: int | None = None
    params: dict[str, Param] = field(default_factory=dict)
    children: list["Tree | None"] = field(default_factory=list)


@dataclass
class Iterate:
    """Runs output0 once, then output1 ``count`` times ($number counting)."""

    count: int | Param = 0
    seed: int | None = None
    inherited: bool = True
    first: "Tree | None" = None
    repeated: "Tree | None" = None


@dataclass
class Switch:
    """Runs output1 when its value is true, else output0."""

    value: bool | Param = False
    seed: int | None = None
    inherited: bool = True
    if_false: "Tree | None" = None
    if_true: "Tree | None" = None


type Tree = Quadrant | Iterate | Switch


@dataclass
class Settings:
    """The FX-Map node's own parameters."""

    tile_x: bool = True
    tile_y: bool = True
    background: list[float] | int | None = None
    """Inline values, or an input node's index (state 3)."""
    hurst: float = 1.0
    opacity: float = 1.0
    patterns: list[int] = field(default_factory=list)
    """The pattern inputs' node indices."""
    seed: int | None = None
    """The FX-Map's own random seed, when a constant."""
    seed_code: int | None = None
    """Its random seed's code block (file offset), when a function."""
    setup_code: int | None = None
    """A function run before the tree (its output size's, in a plain FX-Map):
    library FX-Maps set variables there (regions from $outputsize) that the
    tree's functions read."""
    render_roi: list[float] | int | None = None
    """The region of pattern space drawn into the image, x0 x1 y0 y1: inline
    values, or its function's code (file offset). None: the image, 0 1 0 1."""
    tiling_roi: list[float] | int | None = None
    """The region that wraps, the same way."""


def _float(w: int) -> float:
    return struct.unpack("<f", struct.pack("<I", w))[0]


def read_settings(node: Node, *, color: bool) -> Settings:
    """Return the FX-Map ``node``'s settings (code-driven ones at defaults)."""
    words = node.words
    out = Settings(tile_x=bool(words[0] >> 24 & 1), tile_y=bool(words[0] >> 25 & 1))
    states = words[1]

    def is_offset(w: int) -> bool:
        return node.start < w + BASE <= node.end and w % 4 == 0

    k = 3  # after the states and code[0]
    while k < len(words) and not is_offset(words[k]):
        out.patterns.append(words[k])
        k += 1
    if words[0] >> 22 & 3 == INLINE and out.patterns:  # the seed, inline
        out.seed = out.patterns.pop()
    else:
        out.seed_code = words[k] + BASE if k < len(words) else None
        k += 1
    if k < len(words) and is_offset(words[k]):
        out.setup_code = words[k] + BASE
    k += 1  # the setup's code
    sizes = (
        ("background", 4 if color else 1),
        ("hurst", 1),
        ("opacity", 1),
        ("render_roi", 4),
        ("tiling_roi", 4),
    )
    for n, (name, size) in enumerate(sizes):
        state = states >> 2 * n & 3
        if state == INLINE:
            vals = [_float(w) for w in words[k : k + size]]
            setattr(
                out, name, vals if name.endswith(("background", "_roi")) else vals[0]
            )
            k += size
        elif state == INPUT and name == "background":
            out.background = words[k]
            k += 1
        elif state == CODE and name.endswith("_roi"):
            setattr(out, name, words[k] + BASE)
            k += 1
        elif state:
            k += 1  # a function: left at its default
    return out


def read_tree(node: Node, *, color: bool = False) -> Tree | None:
    """Return the FX-Map ``node``'s tree, shared subtrees read once.

    An inline opacity is one float in a grayscale FX-Map, four in color.
    """
    if len(node.words) < 3:  # noqa: PLR2004
        return None
    seen: dict[int, Tree] = {}

    def word(off: int) -> int:
        return node.at(off + BASE, 1)[0]

    def read(off: int) -> Tree | None:
        if not off:
            return None
        if off in seen:
            return seen[off]
        w = word(off)
        lo = w & 0xFF
        nxt = off + 4
        tree: Tree
        if lo & 0x0F == 0x08:  # noqa: PLR2004
            quad = Quadrant(
                pattern=(w >> 8 & 0xFF) if w & 0x40 else None,
                image=(w >> 8 & 0xFF) if w & 0x80 else None,
                blend_max=bool(w >> 18 & 1),
            )
            seen[off] = tree = quad
            states = {name: w >> bit & 3 for name, bit, _ in QUAD_PARAMS}
            image_flags = 0
            if w >> 31:
                image_flags = word(nxt)
                nxt += 4
            if w & 0x10 or any(states.values()) or image_flags & 2:
                at = word(nxt)
                nxt += 4
                if w & 0x10:
                    quad.seed = word(at)
                    at += 4
                for name, _, size in QUAD_PARAMS:
                    n = 1 if name == "opacity" and not color else size
                    if states[name] == INLINE:
                        quad.params[name] = Param(
                            INLINE, [_float(word(at + 4 * i)) for i in range(n)]
                        )
                        at += 4 * n
                    elif states[name] == CODE:
                        quad.params[name] = Param(CODE, code=word(at) + BASE)
                        at += 4
                if image_flags & 2:
                    quad.image_index = Param(CODE, code=word(at) + BASE)
            kids = w >> 16 & 3
            if kids == 1:
                quad.children = [read(word(nxt + 4 * i)) for i in range(4)]
            elif kids == 2:  # noqa: PLR2004
                quad.children = [read(word(nxt))] * 4
        elif lo & 0x07 == 0x03:  # noqa: PLR2004
            it = Iterate(inherited=bool(w & 0x08))
            seen[off] = tree = it
            if not w & 0x80:
                it.count = w >> 8
            if w & 0x10:
                it.seed = word(nxt)
                nxt += 4
            if w & 0x80:
                it.count = Param(CODE, code=word(nxt) + BASE)
                nxt += 4
            it.repeated = read(word(nxt))
            if w & 0x40:
                it.first = read(word(nxt + 4))
        elif lo & 0x07 == 0x01:
            sw = Switch(value=bool(w & 0x40), inherited=bool(w & 0x08))
            seen[off] = tree = sw
            if w & 0x10:
                sw.seed = word(nxt)
                nxt += 4
            if w & 0x80:
                sw.value = Param(CODE, code=word(nxt) + BASE)
                nxt += 4
            sw.if_false = read(word(nxt))
            sw.if_true = read(word(nxt + 4))
        else:
            msg = f"unknown FX-Map node word {w:#010x} at {off:#x}"
            raise ValueError(msg)
        return tree

    # The root is the word after the states (which can look like an offset,
    # so not node.code[0]).
    return read(node.words[2])


def max_depth(tree: Tree | None) -> int:
    """Return the deepest Quadrant level below ``tree`` (0 at the root)."""
    memo: dict[int, int] = {}

    def deep(t: Tree | None) -> int:
        if t is None:
            return -1
        if id(t) in memo:
            return memo[id(t)]
        if isinstance(t, Quadrant):
            d = max([0, *(1 + deep(c) for c in t.children if c is not None)])
        elif isinstance(t, Iterate):
            d = max(deep(t.first), deep(t.repeated))
        else:
            d = max(deep(t.if_false), deep(t.if_true))
        memo[id(t)] = d
        return d

    return max(deep(tree), 0)
