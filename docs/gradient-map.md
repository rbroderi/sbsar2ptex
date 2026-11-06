# Substance Gradient Map

## Clean-room behavioral specification

This document specifies how a compiled Gradient Map node (`.sbsasm` ops
0 and 1) stores its keys, and how the engine maps a value through them.
It is intended to be sufficient for an independent implementation
without access to the original implementation. No engine code is
reproduced here. It was recovered from three sources:

- gradient maps cooked with `sbscooker` (sbsar2ptex.probe);
- `sbsrender` renders of those, some with their key bytes patched;
- the node words of gradient maps in archives cooked by two cookers: a
  version 5 one, and the version 9 one of Designer 16.

Each statement is tagged:

- **[measured]**: established by cooking, patching or rendering.
- **[inferred]**: fits the data, but was not tested directly.

Integers are little-endian. An "offset" is from the file's BASE (see
sbsar2ptex.sbsasm).

## 1. Ops and node flags

Op 0 outputs grayscale and op 1 outputs color. Either can read a
grayscale input. The node word is `flags << 16 | x << 8 | op`. These flag
bits matter here:

| bit | mask | meaning |
|---|---|---|
| 0 | 0x001 | the node has an output-size function (its first code block) |
| 3, 4 | 0x018 | set on every gradient map seen |
| 6, 7 | 0x0C0 | state 2: an extra code block, the node's random seed (section 4) |
| 8 | 0x100 | each key carries a trailing u16 (section 2) |

**[measured]** The combinations seen:

- 0x119: Designer 16's cooker, every probe. Also every gradient map of a
  version-5 archive (Biomech), and some of version 9.
- 0x19 and 0x9: version 9.
- 0x18, plus 0x118 and 0x108: version 9. A fixed-size node with no size
  function; the node word's second byte holds its size (see below).
- 0x99: version 9.

**Fixed size.** **[measured]** With bit 0 clear, the node word's second
byte (`x`) packs the size as `log2 height << 4 | log2 width`. The low
nibble is the width and the high nibble the height, as for other
fixed-size nodes. Probes cooked with these absolute output sizes:

| output size (log2, w x h) | byte | rendered size (pixels, w x h) |
|---|---|---|
| 4 x 8 | 0x84 | 16 x 256 |
| 8 x 4 | 0x48 | 256 x 16 |
| 5 x 5 | 0x55 | 32 x 32 |

In the wild the byte runs from 0x55 to 0xBB (square sizes), plus 0x84 on
two 0x118 nodes of Concrete_Floor. The same packing holds for every op
with bit 0 clear, square or not, curves included. `docs/node-seed.md`
section 2.4 has the measurements.

**Words after the node word.** **[measured]** In this order:

1. The input node index.
2. The key count.
3. The keys' offset. It is present with or without bit 0.
4. The seed slot, only when bits 6-7 are non-zero. With bits 6-7 = 2 it
   is the seed block's offset. With bits 6-7 = 1 it would be the seed
   itself, an inline i32 (**[inferred]**, see section 6).
5. With bit 0: the size function's offset.

So a 0x99 node reads keys offset, then seed offset, then size offset
(Snow_Dune n1171, Lava n1568). The seed comes before the size function,
not after it. `docs/node-seed.md` gives the seed slot's rule for every
op.

Then come the inline data and the code.

**Key data length.** **[measured]** The keys run from their offset to the
next block's offset. With bit 0 clear and no seed block they run to the
end of the node, and the keys' offset points at the word right after
itself. The length is exactly the key count times the key size (section
2), rounded up to 4 bytes:

- This held in all 1,927 gradient maps of a 35-archive survey.
- No extra bytes follow, only the 0-3 bytes of padding the rounding
  implies.
- A reader may treat any other length as malformed.

**Key count.** **[measured]** The count is never 0. In the survey it runs
from 2 to 693. A gradient map whose `.sbs` has no keys (an empty key
array, or none) cooks to count 2: the default keys (0, black) and
(1, white). A single key cooks to count 1: 6 bytes of key, 8 with
padding.

## 2. Key layout

**[measured]** Keys are stored in position order. A position is a u16:
0 is 0.0 and 0xFFFF is 1.0.

| op | flags bit 8 (0x100) | key bytes | fields |
|---|---|---|---|
| 0 (gray) | set | 6 | u16 position, u16 value, u16 extra |
| 0 (gray) | clear | 4 | u16 position, u16 value |
| 1 (color) | set | 8 | u16 position, u8 r, g, b, a, u16 extra |
| 1 (color) | clear | 6 | u16 position, u8 r, g, b, a |

Examples (node number: count, byte length):

- Version 5, 0x119: Biomech n3163, gray, 3 keys, 20 bytes (18 rounded
  up). n5781, color, 2 keys, 16 bytes.
- Version 9, 0x19: Bark n139, gray, 4 keys, 16 bytes. n389, color, 2 keys,
  12 bytes. Read this way, n139's positions run 0, 0x8000, 0x8114, 0xFFFF.
  A 6-byte reading of them is not in order.
- Version 9, 0x119: Bark n1602, color, 34 keys of 8 bytes. The extra u16
  is 0 in every key.

**The extra u16 is the key's midpoint slot.**

- Designer 16's cooker writes 0x8000 (0.5) there, whatever midpoint the
  `.sbs` key gives. Probes with midpoints 0.2, 0.5 and 0.8 cooked to
  identical bytes.
- Version-5 archives hold 0x8000 there. Version-9 0x119 nodes hold 0.

## 3. Interpolation

**[measured]** The engine ignores the extra u16. Patched values of
0x0000, 0x3333, 0x8000, 0xCCCC and 0xFFFF gave identical renders:

- gray, 2 keys (0 black, 1 white), on a 16-bit ramp;
- color, 3 keys (0 red, 0.5 green, 1 blue).

Between two keys the value is linear in the input: `x = 0.127` gave
red 0.745 and green 0.255 (`t = 0.254`).

**[measured]** Fitted against renders of Biomech's gradient maps (n3904,
a drop from 1 to 0 between keys 0.00421 apart):

- The keys are first baked into a 256-entry table, one entry per 1/255
  step, stored in 16 bits.
- The input reads the table bilinearly, its fraction rounded to 1/256.

So a sharp drop between keys closer than 1/255 is blurred across one
table step.

## 4. Flags bits 6-7 = 2

**[measured]** Examples: Snow_Dune n1171 (gray, 4 keys) and n1577
(color), Lava n1568, Wood n1371. These nodes have a third block offset.

- The offsets run keys, then seed, then size function (section 1). The
  seed block returns one integer and the size function returns two.
- The seed block reads `$randomseed`, as other nodes' seed blocks do.
- Their size function also draws `rand` values and stores them in
  variables. Snow_Dune n1171 stores three; Wood n1371 stores two
  constants and two `rand` values.
- The keys are ordinary key data at the keys' offset, in the layout of
  section 2.

**[inferred]** Bits 6-7 = 2 marks a node whose random seed is a function
of the graph's seed. No key or other parameter of the node reads the
variables its size function sets: they are left over from the instanced
graph's functions. So the keys are static, and the node maps values as a
0x19 node does. Still to confirm: that n1171's input-to-output mapping
does not change with `$randomseed`.

## 5. Acceptance tests

Synthetic key data (hex bytes, in file order) and the keys a reader must
return as `(position, value)`:

| # | op, flags | count | bytes | keys |
|---|---|---|---|---|
| 1 | 0, 0x119 | 2 | `00 00 00 00 00 80 ff ff ff ff 00 80` | (0, 0), (1, 1) |
| 2 | 0, 0x19 | 4 | `00 00 00 00 00 80 ff ff 14 81 00 00 ff ff ff ff` | (0, 0), (0.5, 1), (0.50421, 0), (1, 1) |
| 3 | 1, 0x119 | 3 | `00 00 ff 00 00 ff 00 80 00 80 00 ff 00 ff 00 80 ff ff 00 00 ff ff 00 80` | (0, red), (0.50001, green), (1, blue), all with alpha 1 |
| 4 | 1, 0x19 | 2 | `00 00 00 00 00 ff ff ff ff ff ff ff` | (0, black with alpha 1), (1, white) |

Rendered, measured at 16 bits on a 256-pixel ramp `(x + 0.5) / 256`:

- **Test 1** gives the input back: 0.1270 at x = 32, 0.5019 at x = 128.
  That holds with the trailing u16s set to 0x0000, 0x3333, 0xCCCC or
  0xFFFF instead of 0x8000.
- **Test 3** gives (0.745, 0.255, 0) at
  x = 32 and (0, 0.749, 0.251) at x = 160, whatever the trailing u16s
  hold.

**Layout tests.** Each node starts at offset 0x100. The words listed come
after the node word, so the first is `rest[0]` at 0x104. Every offset
word points inside the node.

5. **Flags 0x0099.** Words: input, 4, keys offset, seed offset, size
   offset.
   - Keys: `rest[2]`.
   - Seed block: `rest[3]` (it returns one integer).
   - Size function: `rest[4]` (it returns two).
6. **Flags 0x0019.** Words: input, 4, keys offset, size offset.
   - Keys: `rest[2]`.
   - No seed.
   - Size function: `rest[3]`.
7. **Flags 0x0018, op 0, node word 0x00188400.** Words: input, 2, 0x110,
   then two 4-byte keys.
   - Keys: at 0x110, the word after the offset, 8 bytes to the node end.
   - Fixed size: 16 pixels wide, 256 high.
8. **Flags 0x0019, op 0, count 3.** The keys span 16 bytes before the
   next block.
   - Malformed: 3 keys of 4 bytes, rounded up, take 12 bytes.

## 6. Unknown

- **An inline seed on a gradient (bits 6-7 = 1).** It is not in any
  archive, and it was not probed. The cooker drops the seed of a node
  that draws no `rand` (state 0), and a gradient map's keys cannot be
  functions. **[inferred]** from every other op (`docs/node-seed.md`):
  the seed word sits where the seed block's offset would, after the
  keys' offset and before the size function's offset. Nothing in the
  gradient reads it.
- **A seed on a gradient with bit 0 clear.** Not seen. **[inferred]** The
  seed slot follows the keys' offset.
- How a gradient with one key maps values. It was not rendered.

- What makes a cooker set bit 8 on one node and not on another in the
  same archive. Seen: Bark n1602 (0x119) beside n389 (0x19).
- Whether the version-9 extra u16 (0 in every key seen) is ever
  non-zero.
