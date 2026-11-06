# Substance Node Random Seed

## Clean-room behavioral specification

This document specifies how a compiled Substance node (`.sbsasm`) stores
its own random seed: what node-word flag bits 6-7 mean, where the seed's
word sits among the node's other words, and what seed value the engine
uses in each case. It covers every op that can carry a seed, not only
gradient maps. It is intended to be sufficient for an independent
implementation without access to the original implementation. No engine
code and no file data are reproduced; third-party files are cited by
name and node number only. It was recovered from three sources:

- Probe graphs cooked with Designer 16's `sbscooker`: one filter whose
  parameter function draws `rand`, with the node's `randomseed` setting
  absent, absolute, relative or a function. Filters probed: uniform color,
  blend, transformation, levels, blur, warp, HSL, pixel processor, and a
  fixed-size uniform.
- `sbsrender` renders of those probes at several graph seeds
  (`--set-value $randomseed@N`).
- A survey of 35 cooked archives (cook versions 5, 6 and 9; about 170,000
  nodes), checking the rule below against every node.

Each statement is tagged:

- **[measured]**: established by cooking, rendering or the survey.
- **[inferred]**: fits the data, but was not tested directly.

## 0. Node basics

This is all the rest of this document assumes.

- **Words.** A compiled graph is little-endian. A node is a run of u32
  words. An **offset** is a byte position relative to byte 0x34 of the
  file.
- **Node word.** A node's first word is `flags << 16 | x << 8 | op`. The
  byte `x` does not matter here.
- **rest.** Below, `rest` means the node's u32 words after its node word,
  indexed from 0.
- **Order of a node's words:** its fields (flag words, input node indices,
  counts), then its offset words (each points at a code block later in
  the same node), then inline values, then the code blocks themselves.
- **Code block.** A code block is a u16 stream:
  - It starts with a u16, the instruction count.
  - Each instruction is a u16 opcode, then `count` operand u16s. The
    opcode is `count << 10 | type << 8 | (width - 1) << 6 | op`. `type` is
    0 bool, 1 float, 2 integer.
  - A u32 operand (a constant, a graph-input uid) is 4-byte aligned in the
    file. When needed, a padding u16 comes first, and it counts in
    `count`.
  - Instruction `k` defines value `#k`. The block's value is its last
    instruction's value.
- **Ops a seed block uses** (all integer, width 1):

  | op | name | operands | value |
  |---|---|---|---|
  | 0x02 | load graph input | u32 uid | that graph input's current value |
  | 0x00 | constant | u32 | the constant (a signed 32-bit integer) |
  | 0x12 | add | two u16 value indices | their sum |

- **`$randomseed`** is the graph input whose identifier is `$randomseed`.
  It is an integer graph input. Its uid comes from the graph's input list.

## 1. Flag bits 6-7: the seed state

**[measured]** Bits 6-7 of `flags` (mask 0x00C0 of `flags`, or 0x00C00000
of the node word) are the node's **seed state**:

| state | meaning | words it adds |
|---|---|---|
| 0 | No seed of its own. The seed is the constant 0. | none |
| 1 | A constant seed, stored inline. | one: the seed, an i32 |
| 2 | A seed function. | one: the offset of a code block returning one integer |
| 3 | Never seen. | unknown |

Nothing else in the node changes with the state. Bit 0 of `flags` is
independent of it. Bit 0 says whether the node has an output-size
function; it does not say whether the node has code. A node with bit 0
clear can still hold a seed block and parameter-function blocks.

## 2. Where the seed word sits

### 2.1 The rule

**[measured]** A state 1 or 2 node has one extra word, the **seed slot**.
It sits at a fixed position:

1. the op's fields (section 2.2);
2. for a gradient map or a curve: its data offset (keys or points);
3. **the seed slot** (state 1 or 2 only);
4. the output-size function's offset (only when `flags` bit 0 is set);
5. the op's remaining offset words, then inline values, then code, exactly
   as in a state-0 node. For the neighborhood filters (ops 14-17, 20-27,
   37, 42, 43) these are not a fixed number of code offsets: each
   parameter in state 2, the pixel-size vector included, adds one offset
   among the inline values, in parameter order (`docs/node-params.md`).

Read the slot by its position, never by how it looks. A state-1 seed can
be any i32, including one that happens to look like an offset into the
node, or like a node index.

**[measured]** Take a state-0 node and cook the same node with a non-zero
absolute seed. The only change is one word, the seed, inserted at the
slot. Every later word that is an offset grows by 4, because the code
moved 4 bytes. A state-2 node also gains its seed block. In every probe
and every surveyed node, the seed block is the node's last code block in
file order. Do not rely on that ordering; the slot's offset is
authoritative.

### 2.2 Fields before the slot, per op

**[measured]** In file order. "flag" is a flag word, "count" a count;
everything else is a node index (an input), except where noted.

| op (gray / color) | filter | fields before the slot |
|---|---|---|
| 0 / 1 | gradient map | input, key count, **then the keys' offset** |
| 2 / 3 | blend | flag, destination, source |
| 4 / 5 | transformation | flag, input |
| 6 | grayscale conversion | input |
| 7 | shuffle | flag, input 1, input 2 |
| 12 / 13 | uniform | none: the slot is `rest[0]` |
| 14 / 15 | warp | (flag, version 9 cooks only), input, gradient; see the warp note below |
| 16 / 17 | emboss | flag, input, gradient |
| 20 / 21 | blur | input |
| 22 / 23 | directional motion blur | flag, input |
| 24 / 25 | directional warp | flag, input, intensity |
| 26 / 27 | sharpen | input |
| 29 | HSL | input |
| 30 / 31 | levels | flag, input |
| 37 | normal | flag, input |
| 38 / 39 | dynamic gradient | input, gradient |
| 40 / 41 | pixel processor | count word (low 16 bits = n), then n inputs |
| 42 / 43 | distance | flag, mask, then source only if the flag's bit 0 is set |
| 44 / 45 | curve | input, a word, **then the points' offset** |
| 8 / 9 | FX-Map | see section 2.3 |

Graph-input nodes (32 / 33) never carry a seed.

**Warp note.** **[measured]** The warp's leading flag word depends on the
cook version (the `.sbsasm` header's u16 at byte 6). Decide it by
version, never by whether `rest[2]` looks like an offset: a state-1 seed
in `rest[2]` is not an offset either.

| cook version | warp fields | warps seen |
|---|---|---|
| 9 | flag, input, gradient: slot at `rest[3]` | 12,807 in the survey (states 0 and 2) |
| 5, 6 | input, gradient: slot at `rest[2]` | 714 in the survey (state 0 only) |
| 2 | input, gradient: slot at `rest[2]` | probes, states 0, 1 and 2, with and without bit 0 |

A version-2 probe warp in state 1 with bit 0 clear reads: input,
gradient, seed (5), then the inline intensity values and a parameter
offset. The rule of section 2.1 holds there.

### 2.3 FX-Maps (ops 8 and 9)

**[measured]** An FX-Map's words after its node word are, in order:

1. the parameter-states word;
2. the tree root's offset;
3. the pattern inputs' node indices, zero or more;
4. **the seed slot** (state 1 or 2);
5. the setup / output-size function's offset (when present);
6. the rest.

The setup / output-size offset is present exactly when `flags` bit 0 is
set. **[measured]** Flag bit 1 is clear in every node of every op, in the
survey and in the probes. Bit 1 never marks a setup function.

**Where the header ends.** A pattern index and an inline seed look alike:
both are small integers that are not offsets. Find the header's end `E`
as the smaller of two word positions:

- the first offset word at or after `rest[2]`;
- the word the tree root's offset points at. The tree's first word is
  stored inline in the same node.

The words from `rest[2]` up to `E` are the patterns, then, in state 1,
the seed: the last of them. In states 0 and 2 all of them are patterns.
In state 2, the word at `E` is the seed block's offset. With bit 0 set,
the setup offset follows the slot.

**[measured]** This holds for all 5,711 surveyed FX-Maps and all the
probes:

- In the survey the first offset always comes before the tree root.
- Probes of fixed-size FX-Maps (flags 0x0308 and 0x0348: bit 0 clear,
  states 0 and 1) have no offset after the header. Their tree root's
  word follows it directly, so it bounds the header.

For example, a fixed-size probe in state 1 with two pattern inputs reads:

| `rest` index | word |
|---|---|
| 0 | states |
| 1 | tree root offset, pointing at `rest[5]` |
| 2, 3 | patterns 1 and 0 |
| 4 | seed 5 |
| 5 | the tree's first word |

With no pattern inputs, it reads states, tree root offset (pointing at
`rest[3]`), then seed 5.

In state 0 there is no slot. The first offset after the patterns is the
setup / size function (bit 0), not a seed block. Surveyed examples:

- OvergrownRock n8185 and Cobblestone Pavement n2943: state 0.
- OvergrownRock n8380 (seed 3) and Mud n3418 (seed 42): state 1.

**Pattern indices.** **[measured]** All 3,904 surveyed pattern indices
are below the FX-Map's own node index. None is 0xFFFFFFFF (unwired) or
past the node count. A reader may reject an index at or past the node
count. 0xFFFFFFFF means no node feeds that input, as for any input. An
index between the node's own index and the node count was never seen
(section 8).

The FX-Map's resolved seed then feeds its tree as described in
`docs/random-generator.md`.

### 2.4 Fixed output size (bit 0 clear)

**[measured]** With bit 0 clear, the node has no size function. Its size
is the node word's second byte `x`, read as `log2 height << 4 | log2
width`: the low nibble is the width and the high nibble the height. The
rule is the same for every op, square or not.

Non-square fixed sizes occur in the survey on ops 0, 2, 3, 4, 8, 30, 40,
41 and 44. Examples:

- Concrete_Floor n1222 (op 4) and n1224 (op 0): 0x84.
- Cobblestone Pavement n3190 (op 44): 0x5B.
- stripe-new-1 n159 (op 40): 0x9A.

Probes cooked with an absolute output size of (log2) 4 x 8 all have
`x = 0x84`. Rendered, each is 16 pixels wide and 256 high:

- uniform (op 13);
- transformation (op 4);
- curve (op 44);
- warp (op 14);
- FX-Map (op 8);
- gradient map (op 0, `docs/gradient-map.md`).

**[measured]** A curve with bit 0 clear (probe flags 0x0008; survey flags
0x18) uses this byte like any other op. It still holds its points'
offset in `rest[2]`. With bit 0 set, the byte is not a size: it is
commonly 0x88, and the size function decides.

## 3. The seed's value

### 3.1 By state

**[measured]**

- **State 0:** the seed is 0. The graph's `$randomseed` is not used.
  Rendered at graph seeds 0 and 3, a state-0 probe gives the same image
  both times, equal to the inheriting probe's image at graph seed 0.
- **State 1:** the slot word is the seed, read as a signed 32-bit integer.
  It does not change with `$randomseed`.
- **State 2:** the seed is the value of the code block at the slot's
  offset, evaluated with the graph inputs' current values. The block
  returns one integer (its last instruction is type integer, width 1).
  The result is reduced modulo 2^32. For example, a relative seed of −2 at
  graph seed 0 renders exactly as an absolute seed of −2 (stored
  0xFFFFFFFE).

**Signedness.** **[measured]** Only the seed's 32-bit pattern matters:

- An inline −1 is stored as 0xFFFFFFFF.
- A relative −2 at graph seed 0 renders the same as an inline 0xFFFFFFFE.
- A relative −2 at graph seed 3 renders the same as an inline 1.

So holding the seed as an i32 and reinterpreting it as a u32 (−2 becomes
4294967294) is correct. The generator consumes the low 32 bits
(`docs/random-generator.md`). Signed and unsigned readings never differ
in output.

**Malformed seeds.** **[measured]** None of the following occurs in the
survey or the probes, and no `.sbs` setting tried produces one:

- state 3;
- a state-2 slot that is not an offset into the node;
- a state-2 block that does not return one integer;
- a node too short to hold its slot word.

Report each as unsupported or malformed. Do not guess a seed.

### 3.2 What the `.sbs` setting compiles to

**[measured]** For a node whose parameter functions draw `rand`:

| node `randomseed` in the `.sbs` | state | slot | effective seed |
|---|---|---|---|
| absent (inherits the graph's) | 2 | block: load `$randomseed` | `$randomseed` |
| relative, N = 0 | 2 | block: load `$randomseed` | `$randomseed` |
| relative, N ≠ 0 | 2 | block: load `$randomseed`, constant N, add | `$randomseed + N` |
| absolute, N = 0 | 0 | none | 0 |
| absolute, N ≠ 0 | 1 | N | N |
| absolute function returning a constant N | 1 | N (folded by the cooker) | N |

**[measured]** A node that draws no `rand` is cooked with state 0,
whatever its `randomseed` setting. A uniform with constant colour and an
absolute seed of 5 cooks to state 0.

### 3.3 Seed blocks found in the wild

**[measured]** Every state-2 slot in the survey points at a block of one
of these four shapes. All read `$randomseed`, and none reads any other
graph input:

| instructions | count |
|---|---|
| load `$randomseed` | 9,761 |
| load, constant, add | 4,290 |
| load, constant, add, constant, add | 786 |
| load, then three (constant, add) pairs | 21 |

The deeper shapes come from nested graph instances with relative seeds,
each adding its own offset. **[inferred]** The two-add and three-add
shapes evaluate as plain sums. Only the one-add shape was rendered.

### 3.4 Rendered cross-check values

**[measured]** A uniform colour node's output, `vector4(rand(1), rand(1),
rand(1), 1)`, rendered as 8-bit RGB, at each effective node seed. These
are the first three values of the node's once-per-node value generator
(`docs/random-generator.md` section 11), times 255 and rounded.
**[inferred]** The channels take the draws in instruction order.

| effective seed | R, G, B |
|---|---|
| 0 | 159, 232, 81 |
| 1 | 152, 17, 159 |
| 2 | 145, 56, 236 |
| 3 | 137, 96, 59 |
| 5 | 123, 175, 213 |
| 6 | 116, 214, 36 |
| 7 | 108, 254, 113 |
| 8 | 101, 38, 190 |
| 10 | 86, 117, 90 |
| −1 | 167, 193, 4 |
| −2 | 174, 153, 182 |

The probes that establish the semantics of section 3.2:

- Inheriting renders seed 0, 1 and 3 at graph seeds 0, 1 and 3.
- Relative 5 renders seed 5 at graph seed 0 and seed 8 at graph seed 3.
- Relative 7 renders seeds 7, 8 and 10 at graph seeds 0, 1 and 3.
- Relative −2 renders seed −2 at graph seed 0 and seed 1 at graph seed 3.
- Absolute 5 renders seed 5 at every graph seed.

## 4. Worked example: fixed-size transformations

**[measured]** Four nodes of this form appear in four cooked archives:

- Bricks n1168
- Cobblestone Pavement n603
- Concrete_Floor n1
- Concrete_Yellow n1

Each is op 4 (transformation, grayscale) with flags 0x0398:

- bit 0 clear: no size function;
- state 2.

Their `rest` reads:

| index | word | meaning |
|---|---|---|
| 0 | 0x0400005F | flag word |
| 1 | a node index (0 in Concrete_Floor) | input |
| 2 | an offset | **seed slot**: a block `$randomseed + 1` (Concrete_Floor, Concrete_Yellow) or `$randomseed + 3` (Bricks, Cobblestone Pavement) |
| 3-6 | four f32 | the inline 2x2 matrix |
| 7 | an offset | the offset's function block (it draws `rand`) |

A reader that finds code offsets only when bit 0 is set, or that looks for
the seed among a list of size-function offsets, finds no seed here. By
section 2.1 the seed is `rest[2]`'s block. At graph seed 0, the effective
seed is 1 or 3.

## 5. Survey

**[measured]** The rule of section 2 was checked against every node of 35
cooked archives:

- 30 material archives;
- 5 community archives, cook versions 5, 6 and 9.

"Holds" means: in state 2 the slot is an offset to an integer-width-1
block; in state 1 the slot is not an offset; and with bit 0 set, the word
after the slot (or after the fields, in state 0) is the offset of an
integer-width-2 block, the size function.

| family | state | bit 0 | nodes | rule holds |
|---|---|---|---|---|
| ordinary ops | 0 | 0 | 18,249 | all |
| ordinary ops | 0 | 1 | 136,810 | all |
| ordinary ops | 2 | 0 | 1,373 | all |
| ordinary ops | 2 | 1 | 7,787 | all |
| FX-Map | 0 | 1 | 7 | all |
| FX-Map | 1 | 1 | 6 | all |
| FX-Map | 2 | 0 | 768 | all |
| FX-Map | 2 | 1 | 4,930 | all |

No node has state 3. State 1 appears in the survey only on FX-Maps. The
probes measured it on uniform, blend, transformation, levels, blur, warp,
HSL and pixel processor nodes. Four op-34 nodes (layout unknown) are all
state 0.

The op/flag combinations with state 2 in the survey:

- 0x0099: ops 0, 1, 2, 3, 30, 40, 41, 44 and FX-Maps.
- 0x0398 and 0x0399: op 4.
- 0x0399: op 5.
- 0x0199: op 6.
- 0x0399, 0x0B99, 0x2399, 0x2B99: op 14.
- 0x0B99: ops 16 and 37.
- 0x1389, 0x1399, 0x1B99: op 20.
- 0x0389, 0x0399, 0x0B99: op 22.
- 0x0399, 0x0B99: op 24.
- 0x0499: op 29.
- 0x0089: op 30.
- 0x0088, 0x0089, 0x0098, 0x00B9: ops 40 and 41.
- 0x0089, 0x0098, 0x0199, 0x0299, 0x0388, 0x0389, 0x0398, 0x0399: op 8.
- 0x0399: op 9.

## 6. Corrections to earlier notes

- **Gradient-map order.** An earlier version of `docs/gradient-map.md`
  listed the size function's offset before the seed block's. It now
  agrees with this document. **[measured]** The seed slot comes first, as
  in every other op. Examples: Snow_Dune n1171 and
  Lava n1568 run keys offset, seed offset, size offset.
- **Bit 0.** Bit 0 of `flags` means "has an output-size function", not
  "has code" (section 1).

## 7. Acceptance tests

Synthetic nodes. Each node starts at offset 0x100. Words are given from
the node word on (`w0` is the node word, `w1` is `rest[0]`). Unless
stated otherwise, the offset words point at well-formed blocks inside the
node.

`SB(N)` is the 24-byte seed block `$randomseed + N`, as u32 words:

```
0x0A020003, U, 0x00000E00, N, 0x00010A12, 0x00000000
```

Here `U` is the `$randomseed` uid. In u16 terms:

1. count 3;
2. `0x0A02` with the uid as two u16s;
3. `0x0E00`, a padding u16, then N as a u32;
4. `0x0A12` with operands 1 and 0.

`SB0` is the 8-byte block `0x0A020001, U` (load `$randomseed` only).
`SZ` is any block whose last instruction is integer width 2 (a size
function).

1. **State 0, size function.** Op 13: `w0 = 0x0209880D`, `w1 = 0x110` (→
   `SZ`). Seed: constant 0. Size function: `rest[0]`. No seed slot.
2. **State 1, size function.** Op 13: `w0 = 0x0249880D`, `w1 = 5`,
   `w2 = 0x114` (→ `SZ`). Seed: 5. Size function: `rest[1]`.
3. **State 1, looks like an offset.** As test 2 with `w1 = 0x110`, a
   value inside the node. Seed: 272 (0x110), not a block. Size function:
   `rest[1]`.
4. **State 1, negative.** As test 2 with `w1 = 0xFFFFFFFE`. Seed: −2
   (equal to 4294967294 modulo 2^32).
5. **State 2, no size function** (the section 4 shape). Op 4:
   - `w0 = 0x03985504`, `w1 = 0x0400005F` (flag), `w2 = 7` (input);
   - `w3` → `SB(1)`;
   - `w4..w7` = 2.0, 0, 0, 2.0 as f32;
   - `w8` → a float block.

   Seed slot: `rest[2]`. Seed at `$randomseed` = 0: 1. At 10: 11. The
   matrix is `rest[3..6]`. Reading must not raise "no seed".
6. **State 2, size function.** Op 2:
   - `w0 = 0x00998802`, then fields flag, destination, source;
   - `w4` → `SB0`, `w5` → `SZ`.

   Seed: equal to `$randomseed`. Size function: `rest[4]`.
7. **State 2, negative relative.** As test 5 with `SB(0xFFFFFFFE)`. At
   `$randomseed` = 1: seed −1 (0xFFFFFFFF as u32). At 0: −2.
8. **Pixel processor, state 1.** Op 41:
   - `w0 = 0x00598829`;
   - `w1 = 0x00020001` (one input), `w2 = 4`;
   - `w3 = 42`, `w4` → `SZ`.

   Inputs: [4]. Seed: 42. Size function: `rest[3]`.
9. **Gradient with a seed block and a size function.** Op 0, flags
   0x0099 (bits 6-7 = 2, bit 0 set, bit 8 clear: 4-byte keys). Node at
   0x100, words from 0x100:
   - `0x00998800`, then 5 (input), then 2 (key count);
   - `0x118` (keys), `0x128` (seed), `0x120` (size);
   - at 0x118, the keys: `0x00000000`, `0xFFFFFFFF`;
   - at 0x120, the size block: `0x0A420001`, then `$outputsize`'s uid
     (load integer width 2);
   - at 0x128, `SB(3)`.

   Keys: (0, 0) and (1, 1), from `rest[2]`. Seed: the block at
   `rest[3]` = 0x128; at `$randomseed` = 0 it is 3. Size function: the
   block at `rest[4]` = 0x120. The seed block lies after the size block
   in the file, so a reader must take the roles from word order, not
   from block order. Swapping the roles makes the seed a two-component
   value, which is wrong.
10. **Curve, state 2.** Op 44 (0x2C): `w0 = 0x0099882C`, then input, a
    word, `w3` → points, `w4` → `SB0`, `w5` → `SZ`. Seed slot:
    `rest[3]`.

In tests 11-14 the tree root's offset points past every word listed.

11. **FX-Map, state 0.** Op 8:
    - `w0 = 0x00198808`, `w1` = states, `w2` → tree;
    - `w3 = 0x1FF8` (pattern), `w4` → `SZ`.

    Patterns: [0x1FF8]. Seed: 0. `rest[3]` is the setup / size function,
    not a seed block.
12. **FX-Map, state 1.** `w0 = 0x00598808`, states, tree, `w3 = 0x20BB`,
    `w4 = 3`, `w5` → `SZ`. Patterns: [0x20BB]. Seed: 3.
13. **FX-Map, state 1, no patterns.** `w0 = 0x00598808`, states, tree,
    `w3 = 42`, `w4` → `SZ`. Patterns: []. Seed: 42.
14. **FX-Map, state 2.** `w0 = 0x00998808`, states, tree, `w3 = 0x5DD`,
    `w4` → `SB(1)`, `w5` → `SZ`. Patterns: [0x5DD]. Seed:
    `$randomseed + 1`.
15. **Layout invariance.** Take test 1's node, insert a seed word after
    the fields, and add 4 to every later offset word. Every other field,
    inline value and block parses identically.
16. **State 3.** `w0 = 0x00C9880D`: reported as unsupported. It must not
    be silently read as 0, 1 or 2.
17. **Fixed-size FX-Map, state 1.** Node at 0x40. Words:
    - `0x03488408`, `0x800` (states), `0x58` (tree root, pointing at
      `w6`);
    - `1`, `0`, `5`;
    - `0x04000088` (the tree's first word), `0x60`, `0x64`.

    Patterns: [1, 0]. Seed: 5. There is no setup offset (bit 0 clear).
    Fixed size: 16 wide, 256 high (`x = 0x84`).
18. **Fixed-size FX-Map, state 0, no patterns.** Node at 0x10. Words:
    - `0x03088408`, `0` (states), `0x1C` (tree root, pointing at `w3`);
    - `0x04000148`, `0x24`.

    Patterns: []. Seed: 0. `0x04000148` is tree data, not a pattern.
19. **Warp, cook version 2, state 1, bit 0 clear.** Op 14:
    - `w0 = 0x4748840E`, then input 3, gradient 1;
    - `w3 = 5`, then the f32s 1.0 and 0.00390625.

    Seed: 5 (`rest[2]`). In a version-9 file, the same words after a
    leading flag word put the seed at `rest[3]`. Reading `rest[2] = 5` as
    "not an offset, so a flag word follows" is wrong.
20. **Malformed state 2.** As test 6, but `w4 = 0x00000005` (not an offset
    into the node): reported as malformed. The same holds when the node
    ends before the slot word.
21. **Signedness.** As test 2, but `w1 = 0xFFFFFFFF`. The seed is −1; as a
    u32 it is 4294967295. Both name the same generator state.

## 8. Unknown

- **State 3.** Never seen, in about 170,000 nodes. Its meaning, perhaps a
  seed taken from an input, is unknown.
- **Absolute seed functions that read `$randomseed`.** In probes the cooker
  folds them as if `$randomseed` were 0:
  - `$randomseed + 7` cooked to state 1 with 7;
  - `$randomseed * 2` cooked to state 0.

  One, `$randomseed * 2 + 1`, kept a state-2 block (load, constant,
  multiply, constant, add). Yet it rendered as seed 1 at graph seeds 0, 1
  and 3, as if the load gave 0 or the multiply 0. Why is not known. No
  surveyed block contains a multiply or any op other than load, constant
  and add.
- **Gradient maps and curves with state 1, or with state ≠ 0 and bit 0
  clear.** Not seen. **[inferred]** The slot still follows the data
  offset.
- **Multi-add seed blocks.** Not rendered. **[inferred]** They are plain
  integer sums (section 3.3).
- **Seed blocks reading other graph inputs.** Not seen.
- **Seed range.** The cooker stores an i32. It rejects a `.sbs` seed of
  4294967294. Section 3.1 shows signedness never changes the output.
- **FX-Maps with inline parameter values, bit 0 clear and state 0 or 1.**
  Not seen and not probed. The probes' fixed-size FX-Maps had no inline
  values before the tree. If inline values ever sat between the header
  and the tree, with no offset after the header, section 2.3 could not
  bound the header. Report such a node as unsupported when the pattern
  words are not all valid node indices.
- **FX-Map pattern indices at or above the node's own index, but below the
  node count.** Never seen. Whether the engine allows a forward
  reference is unknown.
- **Flag bit 1.** Never set on any node seen. Its meaning is unknown.
- **Warps in cook versions other than 2, 5, 6 and 9.** Not seen. Whether
  they have the leading flag word is unknown; the seed slot follows the
  fields either way.
