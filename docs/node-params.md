# Neighborhood-Filter Parameter Words (sbsasm ops 14-17, 20-27, 37, 42, 43)

## Clean-room behavioral specification

This document specifies, for the compiled neighborhood filters, which
words follow a node's output-size offset: where each parameter's 2-bit
state lives, how many words each state adds, and in what order. It
replaces the earlier belief that these ops carry a fixed "second code
block". It is intended to be sufficient for an independent
implementation without access to the original implementation. No engine
code and no file data are reproduced; third-party files are cited by
name and node number only.

It was recovered from:

- Probe graphs cooked with Designer's `sbscooker` (cook version 2): one
  filter, with each of its parameters in turn absent, constant or a
  function reading a float graph input; and with the base parameters
  tiling, pixel size and pixel ratio set.
- A survey of every node of these ops in 35 cooked archives (30 material
  archives and 5 community archives; cook versions 5, 6 and 9): 38,105
  nodes.

Each statement is tagged:

- **[measured]**: established by probes or by the survey.
- **[inferred]**: fits the data, but was not tested directly.

`docs/node-seed.md` section 0 defines words, offsets, `rest`, code
blocks and the node word (`flags << 16 | x << 8 | op`). Its section 2
places the fields, the seed slot and the size offset. This document
starts where that one stops.

## 1. Two state words

A parameter's **state** is a 2-bit number:

| state | numeric parameter | boolean parameter |
|---|---|---|
| 0 | the parameter's default; no word | false; no word |
| 1 | a constant, stored inline as `width` f32 words | true; no word |
| 2 | a function: one word, the offset of a code block | a function: one word, the offset of a code block |
| 3 | never seen (report as unsupported) | never seen |

The states live in two places:

- **P**, the node word's top byte (`flags >> 8`, bits 24-31 of the node
  word). Its layout below is for these ops only. Other ops (HSL, for
  one) use `P` differently.
- **F**, the op's **flag word**: the first of its fields, for the ops
  that have one (`docs/node-seed.md` section 2.2). The warp's version-9
  leading flag word is not an `F`: it holds no states (it is 0 in every
  warp seen).

### 1.1 P, common to all these ops

**[measured]**

| bits of P | meaning | words |
|---|---|---|
| 0-1 | **tiling**: an enumeration (0 none, 1 horizontal, 2 vertical, 3 both). It is a value, not a state. | none, whatever its value |
| 2-3 | state of the **pixel-size vector**, a float2 (width 2) | state 1: two f32; state 2: one offset to a float2 block |

Probes: tiling 0, 1 and 2 set bits 0-1 to 0, 1 and 2; the default is 3.
A tiling function is folded by the cooker to a constant. All four tiling
values occur in the survey, and none adds a word.

**The pixel-size vector.** **[measured]** The cooker gives every probe
node state 2 by default, with a block computing, from `$sizelog2`
(builtin 3, float2 `(log2 width, log2 height)`):

- for all ops but the warp: `pow2(min(s.yx - s.xy, 0))`, which is
  `(min(w, h) / w, min(w, h) / h)`: (1, 1) for a square image;
- for the warp: the same with the exponent doubled, `pow2(2 min(s.yx -
  s.xy, 0))`.

Setting the node's Pixel Size to (2, 3), absolute or relative, multiplies
that by the constant (2, 3) in the same block. A Pixel Size function
appends its own instructions to the block. Pixel ratio 1 changed
nothing. **[inferred]** State 0 means the vector (1, 1). How the engine
uses the vector when rendering is not established (section 6).

### 1.2 The op's own parameters

**[measured]** In bit order. "Width" is the number of f32 words a state-1
value takes. Widths not marked otherwise are the same for the grayscale
(even) and colour (odd) op.

| op (gray / color) | filter | state word | parameters: bits, width |
|---|---|---|---|
| 14 / 15 | warp | P | intensity: **bits 5-6**, width 1. Bit 4 is never set. |
| 16 / 17 | emboss | F | intensity 1-2 (w 1); light angle 3-4 (w 1); highlight colour 5-6 (w 1 gray, **4 colour**); shadow colour 7-8 (w 1 gray, **4 colour**). Bit 0: never set. |
| 20 / 21 | blur | P | intensity: bits 4-5, width 1 |
| 22 / 23 | directional motion blur | F | intensity 0-1 (w 1); angle 2-3 (w 1) |
| 24 / 25 | directional warp | F | intensity 1-2 (w 1); angle 3-4 (w 1). Bit 0: never set. |
| 26 / 27 | sharpen | P | intensity: bits 4-5, width 1 |
| 37 | normal | F | intensity 0-1 (w 1); invert Y 2-3 (**boolean**); alpha channel 4-5 (**boolean**, default true) |
| 42 / 43 | distance | F | bit 0: the source input is present (`docs/node-seed.md` 2.2); bit 1: combine distance, a plain bit, default 1; distance 2-3 (w 1) |

Bits of P above those listed are 0 in every node surveyed, and so are the
bits of F above those listed.

Notes:

- **Warp, bits 5-6.** **[measured]** A constant intensity sets P bit 5
  (state 1), a function sets bit 6 (state 2). Reading the warp's
  intensity from bits 4-5 instead turns a P of 0x23 (tiling 3, intensity
  state 1) into "intensity function", which is wrong: 5,096 surveyed
  warps have P = 0x23 and an inline f32.
- **Emboss in grayscale** stores each colour as one f32, the mean of its
  red, green and blue: a highlight colour of (0.9, 0.8, 0.7, 1) was
  stored as 0.8. In colour, as four f32 (r, g, b, a).
- **Booleans** (normal's two) never add an inline word: the state is the
  value. A function adds an offset to a block whose last instruction is
  type bool.
- **Distance bit 1** is a single bit. A function for it was folded by the
  cooker to a constant. The distance's colour switch did not change any
  word of the node in the probes.

## 2. The word order

**[measured]** After the node's fields, its data offset (gradient map and
curve only, not these ops), its seed slot (seed state 1 or 2) and its
size offset (flag bit 0), the words are, in this order:

1. the **pixel-size vector**: nothing (state 0), two f32 (state 1) or one
   offset (state 2);
2. the op's parameters, **in increasing bit order**, as listed in
   section 1.2: for each, nothing, `width` f32 or one offset, by its
   state;
3. then the code blocks begin.

Offsets and inline f32 are interleaved in that order. An offset is not
gathered ahead of the f32 values. For example, a directional warp with
intensity constant and angle a function reads intensity f32, then the
angle's offset.

**The region ends exactly at the first code block.** **[measured]** In all
38,105 surveyed nodes, the words this rule consumes end exactly at the
lowest block any of the node's offsets (seed, size, pixel size,
parameters) points at; with no offsets at all, at the node's end. A
reader should check this and report a mismatch as malformed.

**Every offset's block has the expected result.** **[measured]** Every
pixel-size offset in the survey points at a block whose last instruction
is float width 2. Parameter functions return float width 1, float width
4 (colour emboss colours), or bool (normal).

## 3. There is no second code block

**[measured]** None of these ops has a fixed second code offset. What was
taken for a "neighborhood second" is the **pixel-size function**, P bits
2-3 = 2. The number of offsets after the size offset equals the number
of parameters in state 2, counting the pixel-size vector as one, in
every one of the 38,105 nodes.

So a reader must not take a fixed count of offsets as code. It must walk
section 2's list. Concrete n1694 (op 20, P = 0x2B: tiling 3, pixel size
state 2, intensity state 2) has exactly the size offset, the pixel-size
offset and the intensity offset. A reader taking two offsets "as code"
eats the pixel-size one as the second code and then reads the intensity
as inline, past the words: this was the IndexError in Cobblestone
Pavement, Concrete, Concrete_Floor, Forest_Ground, Lava and
Road_Asphalt_GR.

With bit 0 clear there is no size offset, and the list starts right after
the fields (and seed slot). Such nodes in the survey (for example
persian-mosaic-wall-tiels n604, a warp with P = 0x23) read: fields, then
the intensity f32.

## 4. Survey

**[measured]** Every node of these ops in the 35 archives, 34 of which
contain at least one:

| op | nodes | op | nodes |
|---|---|---|---|
| 14 | 13,521 | 22 | 3,642 |
| 16 | 120 | 23 | 30 |
| 17 | 2 | 24 | 18,650 |
| 20 | 1,397 | 25 | 175 |
| 21 | 11 | 26 | 78 |
| 37 | 203 | 27 | 7 |
| 42 | 268 | 43 | 1 |

Total 38,105, with **no exception**:

- the rule of section 2 consumes exactly the words before the first block;
- the number of offsets after the size offset equals the number of
  state-2 parameters (pixel size included);
- no state is 3; no bit outside section 1 is set in P or F.

Pixel-size states: 0 in 18,977 nodes, 1 in 424, 2 in 18,704. Blur
intensity functions (P bits 4-5 = 2): 49 nodes. Warp intensity functions:
none in the survey, two in the probes.

## 5. Acceptance tests

Synthetic nodes; each node starts at offset 0x100. Words are given from
the node word on (`w0`; `w1` is `rest[0]`). "→ X" is an offset word
pointing at a well-formed block X later in the node:

- `SZ`: a size block (last instruction integer width 2);
- `PS`: a float2 block;
- `F1`: a float1 block;
- `F4`: a float4 block;
- `B`: a block whose last instruction is bool;
- `SB0`: a seed block (`docs/node-seed.md` section 7).

The first block starts right after the last word listed. Floats are
f32 bit patterns: 7.25 is 0x40E80000, 1.25 is 0x3FA00000.

1. **Blur, intensity function (the Concrete n1694 shape).** Op 20,
   `w0 = 0x2B198814`; `w1 = 7` (input), `w2` → `SZ`, `w3` → `PS`,
   `w4` → `F1`. Size: `rest[1]`. Pixel size: the function at `rest[2]`.
   Intensity: the function at `rest[3]`. No inline values. A reader must
   not raise IndexError.
2. **Blur, constant intensity.** `w0 = 0x1B198814`; 7, → `SZ`, → `PS`,
   7.25. Intensity 7.25.
3. **Blur, default pixel size.** `w0 = 0x13198814`; 7, → `SZ`, 7.25.
   Exactly one offset (the size). Intensity 7.25.
4. **Blur, inline pixel size.** `w0 = 0x17198814`; 7, → `SZ`, 2.0, 3.0,
   7.25. Pixel size (2, 3); intensity 7.25.
5. **Blur, no size function.** `w0 = 0x13188814`; 7, 7.25. No code.
6. **Blur with a seed block.** `w0 = 0x2B998814`; 7, → `SB0`, → `SZ`,
   → `PS`, → `F1`. Seed: the block at `rest[1]`; size `rest[2]`; pixel
   size `rest[3]`; intensity `rest[4]`.
7. **Tiling adds no word.** As test 3 with `w0 = 0x10198814` (tiling 0):
   the same words, intensity 7.25.
8. **Warp, version 9, constant intensity.** Op 14, `w0 = 0x2B19880E`;
   0 (flag word), 3 (input), 4 (gradient), → `SZ`, → `PS`, 1.25.
   Intensity 1.25 (P bits 5-6 = 1). Reading bits 4-5 (= 2) and taking
   the f32 as an offset is wrong.
9. **Warp, version 9, intensity function.** `w0 = 0x4B19880E`; 0, 3, 4,
   → `SZ`, → `PS`, → `F1`. Intensity: the function at `rest[5]`.
10. **Warp, version 9, default pixel size.** `w0 = 0x2319880E`; 0, 3, 4,
    → `SZ`, 1.25. One offset.
11. **Directional motion blur.** Op 22, `w0 = 0x0B198816`; F = 0x9
    (intensity 1, angle 2), 7, → `SZ`, → `PS`, 7.25, → `F1`.
    Intensity 7.25; angle: the function at `rest[5]`.
12. **Directional warp.** Op 24, `w0 = 0x03198818`; F = 0x12 (intensity
    1, angle 2), 7 (input), 8 (intensity input), → `SZ`, 7.25, → `F1`.
13. **Emboss, colour.** Op 17, `w0 = 0x0B198811`; F = 0xCA, 7, 8, → `SZ`,
    → `PS`, 7.25 (intensity), 0.3 (light angle), → `F4` (highlight), then
    0.1, 0.2, 0.3, 1.0 (shadow). The same F on op 16 (grayscale) with a
    constant shadow takes one f32 for it.
14. **Normal.** Op 37, `w0 = 0x03198825`; F = 0x19 (intensity 1, invert
    Y 2, alpha 1), 7, → `SZ`, 3.0, → `B`. Intensity 3; invert Y: the
    function; alpha: true, with no word.
15. **Distance.** Op 42, `w0 = 0x0B19882A`; F = 0xB (source present,
    combine, distance 2), 5 (mask), 6 (source), → `SZ`, → `PS`, → `F1`.
16. **State 3.** Test 3 with `w0 = 0x1F198814` (pixel size state 3):
    reported as unsupported.
17. **Region mismatch.** Test 2 with one extra f32 between 7.25 and the
    first block: reported as malformed, not read as a parameter.

## 6. Unknown

- **State 3** of any parameter: never seen.
- **Warp P bit 4**, **emboss F bit 0** and **directional warp F bit 0**:
  never set, in probes or survey. Their meaning is unknown.
- **The pixel-size vector's effect on rendering.** Its value is
  established (section 1.1), not how the engine uses it. Every probe
  image was square, where the default is (1, 1). **[inferred]** For a
  square image with Pixel Size left at its default, ignoring it changes
  nothing.
- **The distance's colour switch**: not found in the node's words.
- **Parameter defaults** (state 0) of each op: out of scope here.
- **Other ops' P.** HSL, levels and others use P with their own layouts;
  this document covers only the ops above.
