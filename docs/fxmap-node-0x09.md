# FX-Map Tree Nodes: the Seed-Function Slot ("node word 0x09000010")

## Clean-room behavioral specification

This document explains the error `unknown FX-Map node word 0x09000010`
and specifies the FX-Map tree layout and behaviour that cause it: a
Quadrant, Iterate or Switch node whose random seed is a function. It is
written for an implementation that has no access to the engine. No engine
code and no file data are reproduced here. Third-party archives are cited
by name and node number only.

It was recovered from three sources:

- probe graphs cooked with Designer's `sbscooker` (the `sbsar2ptex.probe`
  helpers), with one FX-Map whose tree nodes have their `randomseed` set
  absolutely, inline, or by a function graph;
- `sbsrender` renders of those probes at 256 x 256 in 16-bit, compared
  pixel for pixel with renders of reference trees that use inline seeds;
- a survey of the FX-Map trees in the 35 test archives (5 community
  files, 30 cooked-pack files).

Each statement is tagged:

- **[measured]**: established by cooking, rendering or the survey;
- **[inferred]**: fits the data, but was not tested directly.

Tree offsets in this document are the offsets the existing tree reader
uses: file offset minus `BASE` (0x34). Words are little-endian u32.

## 0. What 0x09000010 is

**[measured]** 0x09000010 is not a tree-node word. It is the first word
of a function code block. Read as code (`docs/function-ops.md`), its low
u16 0x0010 is the instruction count (16), and its high u16 0x0900 is the
first opcode (a float constant).

The reader lands on a code block because it ignores bit 5 (0x20) of the
tree-node word. That bit is part of the node's **seed state** (section
1.1). When a node's seed is a function, the node carries one extra slot,
the seed block's offset. A reader that does not count this slot reads
every later slot one place early, so an output offset ends up pointing
at a code block.

In all 32 failing FX-Maps, the culprit is an Iterate whose low byte is
0xAB or 0xA3 (section 4). Its slots are, in order: the seed block, the
count block, then output1. The reader takes the seed block as the count
block, and the count block as output1. Each count block there happens to
begin with 0x09000010.

## 1. Word layout

### 1.1 Seed state, bits 4-5 of every tree-node word

**[measured]** The node words of all three kinds (Quadrant, Iterate and
Switch) hold a 2-bit seed state in bits 4-5:

| state | bits 4-5 | node word bit | seed slot |
|---|---|---|---|
| 0 | 00 | none | no slot. The node's own seed is 0. |
| 1 | 01 | 0x10 | one word, the seed as an inline i32 |
| 2 | 10 | 0x20 | one word, the tree offset of a code block that computes the seed |
| 3 | 11 | 0x30 | never seen: reject (section 6) |

States 0 and 1 are what the existing reader already handles: 0x10 is
read as "its own seed". State 2 is the new case. The cooker writes it when
the node's `randomseed` parameter is a function graph that it cannot fold
to a constant. A function that folds to a constant N is cooked as state 1
with N inline, and N = 0 folds to state 0.

The seed slot sits where the inline seed sits in state 1. Its position per
kind is given below.

### 1.2 Iterate (low 3 bits 0b011)

| bits | meaning |
|---|---|
| 0-2 | 3 (the kind) |
| 3 (0x08) | `randominherited`: 1 inherits the incoming state |
| 4-5 | seed state (1.1) |
| 6 (0x40) | output0 is present |
| 7 (0x80) | the count is a function |
| 8-31 | the inline count, when bit 7 is clear (ignored when it is set) |

**[measured]** The slots that follow the node word, in this order:

1. the seed slot (state 1: the i32; state 2: the seed block's offset), if
   the state is 1 or 2;
2. the count block's offset, if bit 7 is set;
3. output1's offset (the repeated subtree);
4. output0's offset, if bit 6 is set.

Probe words: 0x0000062B (count 6 inline, seed function, inherited);
0x000001AB and 0x000001EB (count and seed functions, without and with
output0). The latter's slots are `seed, count, output1, output0`.

### 1.3 Switch (low 3 bits 0b001)

| bits | meaning |
|---|---|
| 0-2 | 1 (the kind) |
| 3 (0x08) | `randominherited` |
| 4-5 | seed state (1.1) |
| 6 (0x40) | the constant value is true (when bit 7 is clear) |
| 7 (0x80) | the value is a function |

**[measured]** The slots, in order:

1. the seed slot, if the state is 1 or 2;
2. the value block's offset, if bit 7 is set;
3. output0's offset;
4. output1's offset.

Probe words: 0x00000069 (constant true, seed function, inherited),
0x00000061 (the same, not inherited), and 0x000000A9 (value and seed
functions; slots `seed, value, output0, output1`).

### 1.4 Quadrant (low nibble 0x8)

The seed state is bits 4-5, as in 1.1. The rest of the word is as the
existing reader has it.

**[measured]** The parameter block's offset slot is present when any of
these holds:

- the seed state is non-zero;
- any parameter state is non-zero;
- the image index is a function.

At that offset the seed slot comes first, then the parameters in their
usual order. In state 2, the first word at the parameter offset is the
seed block's offset, not an inline seed.

Probe words: 0x02000168 (a pattern with inline pattern size and a seed
function: parameter block `[seed block offset, 0.3, 0.3]`); 0x02010168
(the same with four child slots); 0x02100168 (an opacity function and a
seed function).

## 2. Semantics

### 2.1 The seed value

**[measured]** A seed block returns one integer (its last instruction has
type int, width 1). The cooker rejects a `randomseed` function whose root
is float ("Root node type mismatch").

The node's seed is the 32-bit pattern of that integer. In this
evaluator an integer's 32-bit pattern depends on how the integer was
produced. Measured through the seed:

| how the integer value was produced | its 32-bit pattern |
|---|---|
| `convert` from a float x | the IEEE float32 bits of floor(x) |
| an integer constant N | N as two's complement |
| `add` of two integers | the sum of their patterns, mod 2^32 |
| `getvar` of a variable | the pattern stored there. A never-set integer variable is 0. |

Measured cases, by pattern equality of renders:

- convert of 3.0 → 0x40400000; of 1.02 → 0x3F800000; of 2.04 →
  0x40000000; of 1000 → 0x447A0000; of 12582912 → 0x4B400000; of
  12595257 → 0x4B403039; of −2.49996 → the pattern of −3.0. That last
  case shows floor, not truncation toward zero, in this context.
- `convert(3.0) + 1` → 0x40400001, not the pattern of 4 (0x40800000).
- `convert(3.0) + 16777216` → 0x41400000.
- `convert(16777216.0) + 3` → 0x4B800003.
- `convert(8388608.0) + 8388613` → 0x4B800005.
- `convert(16777216.0) + 1048577` → 0x4B900001.

So for the shapes that occur in the survey (section 4), the rules are:

- `getvar v; convert`, with v a float variable: the seed is the float32
  bits of floor(v).
- `getvar v` of an integer variable that nothing sets: the seed is 0.

**Only the low 24 bits matter.** **[inferred]** The generator
(`docs/random-generator.md`) moves a state only by affine steps mod 2^32,
and a draw reads only bits 1-23 of the advanced state. A difference in
bits 24-31 of a seed therefore stays in bits 24 and up, and never changes
any drawn value. Every equality above was observed through renders, so it
holds modulo 2^24. For example, the sign of −3.0 cannot be seen. An
implementation may compute the full 32 bits as stated. Nothing observable
depends on bits 24-31.

### 2.2 The effective state

**[measured]** A seed from a function is used exactly like an inline seed
(`docs/random-generator.md` 8.2):

```text
effective = ((inherited ? incoming : 0) + seed_u32) mod 2^32
```

Quadrants always inherit. Iterate and Switch inherit when bit 3 is set.
The checks:

- An Iterate with a seed function returning 3 renders exactly as one with
  the inline seed 0x40400000.
- At graph seed 3 it renders exactly as the inline seed 0x40400003. The
  incoming state is added.
- A Quadrant, a Switch, and a non-inheriting Switch whose seed functions
  return 3 each render exactly as the same node with the inline seed
  0x40400000.

Everything below the node derives its states from this effective state as
usual: Quadrant child transforms, Iterate pass steps, and each function's
private generator.

### 2.3 When it is evaluated

**[measured]**

- **On every visit.** The seed block runs each time the walk enters the
  node, with the variables as they are at that moment. Example: an outer
  Iterate (count 3), then a Switch whose value function sets variable x
  to `$number + c`, then an inner Iterate (count 1, not inheriting) with
  seed `convert(getvar x)`. This renders exactly as three inner Iterates
  with the inline seeds bits(c), bits(c + 1) and bits(c + 2), for c = 0,
  3 and 10. The same holds with a Switch in place of the inner Iterate.
- **Constant blocks give the same seed on every visit.** A non-foldable
  constant seed (graph input × 6 → 3) under a count-2 or count-3
  Iterate draws every pass at the same state.
- **First among the node's functions.** The seed block runs before the
  node's other functions, and the variables it sets are visible to them:
  - Iterate: a count function reading a variable that the seed function
    sets gets the new value (6 passes drawn).
  - Switch: a value function reading such a variable sees it set.
  - Quadrant: an opacity function reading such a variable sees it set.

  So the order within a node is: seed, then count or value, then the
  Quadrant parameters in their usual order.

### 2.4 What it changes in the drawing

A seed function has no direct effect on which splats are drawn or in what
order. The tree walk is unchanged: an Iterate runs output0 once, then
output1 once per pass, and a Switch picks output1 when its value is true.
The seed function changes only the effective state, and so every `rand`
drawn by the node and its descendants. **[inferred]** It can change which
splats exist, but only through those `rand` values: for example, a count,
switch value or opacity function that calls `rand`.

## 3. How to implement (behaviour only)

1. When parsing a tree node of any kind, read the seed state from bits
   4-5:
   - state 1: one inline i32;
   - state 2: one block offset;
   - state 3: reject.
   The slot goes where section 1 puts it.
2. When the walk enters the node, before the node's other functions, run
   the seed block in the current variable context. Turn the integer into
   its 32-bit pattern (2.1), then form the effective state (2.2).
3. For the corpus, supporting the two survey shapes (`getvar; convert`
   and an unset `getvar`) is enough. Only the low 24 bits need to be
   right.

## 4. Survey (35 archives)

**[measured]** Scanning every FX-Map tree in the 35 archives (the 5
community files and the 30 cooked-pack files):

- **The current reader** rejects exactly one word value, 0x09000010. It
  does so in 32 FX-Map nodes across 15 archives:

  | archive | failing FX-Map nodes |
  |---|---|
  | Concrete_Yellow | 642 |
  | Forest_Ground | 2673 |
  | Grass | 316, 2511, 8074, 9999, 10506 |
  | Manhole_Cover | 256, 257, 1194 |
  | Moss | 316, 3761 |
  | OvergrownRock | 158, 170, 4102 |
  | Plaster | 23, 4594 |
  | RiberStones | 492, 761, 951, 1152, 1354 |
  | Road_Asphalt_GR | 2538 |
  | Rock_Chunky_Desert_Cliff | 1935 |
  | Rock_Exfoliating_Cliff | 628 |
  | Rock_Layered_Cliff | 15, 16, 1583 |
  | Rock_Sharp_Cliff | 1183 |
  | Street_Cobblestone | 0, 1802 |
  | Wood | 127 |

  Forest_Ground and Road_Asphalt_GR currently stop earlier, on an
  unrelated IndexError.
- **A walk with the seed slot understood** rejects nothing in any of the
  35 archives.
- **Seed-state-2 nodes:** 64, all of them Iterates. There are no
  Quadrants or Switches with seed functions in the corpus:
  - 44 have low byte 0xAB (count function, seed function, inherited).
    Their seed block is `getvar (float); convert`.
  - 20 have low byte 0xA3 (the same, not inherited). Their seed block is
    a single `getvar` of an integer variable that nothing in the map sets.
    So their seed is 0, and their effective state is 0.
  - 54 of the 64 sit inside an enclosing Iterate's repeated output, so
    per-visit evaluation matters.
- **The typical shape** (Street_Cobblestone node 0) is an outer Iterate,
  then a Switch whose value function stores the outer `$number` (a
  float) in a variable, then an Iterate whose seed is `convert` of that
  variable. Each outer pass therefore seeds the inner Iterate with
  bits(float(floor(pass number))).
- **Other low bytes** seen, with no seed functions:
  - Iterate: 0x0B, 0x1B, 0x4B, 0x8B, 0x9B, 0xCB, 0xDB;
  - Switch: 0x09, 0x49, 0x89, 0x99;
  - Quadrant: seed states 0 and 1 only.

## 5. Acceptance tests

All tests use synthetic bytes. Each tree is a list of words at tree
offsets. "Block" means a code block in the `docs/function-ops.md`
encoding. An opcode is
`(operand count << 10) | (type << 8) | (width − 1) << 6 | op`:

| type | value |
|---|---|
| bool | 0 (width field 1) |
| float | 1 |
| int | 2 |

Blocks used:

```text
B_SEED3:  0x09000002 0x40400000 0x00000611
          (2 instrs: const float 3.0; convert int (#0))
B_SEEDV0: 0x05040002 0x06110000 0x00000000
          (2 instrs: getvar float var 0; convert int (#0); one padding u16)
B_INT0:   0x06040001 0x00000000
          (1 instr: getvar int var 0; never set)
B_ADD1:   0x09000004 0x40400000 0x00000611 0x00000E00 0x00000001
          0x00010A12 0x00000002
          (4 instrs: const float 3.0; convert int (#0); const int 1
           (a padding u16, then the u32); add int (#1, #2))
B_SETNUM: 0x05010004 0x0907000A 0x00000000 0x00010440 0x0001084C 0x00000002
          (4 instrs: builtin $number (10); setvar (#0, var 0);
           const bool true; sequence (#1, #2) -> bool)
```

**T1. An Iterate with a seed function (parsing).** At 0x100: `0x000001AB`,
`0x00000200`, `0x00000300`, `0x00000400`. At 0x400: `0x00000048` (a
Quadrant drawing a square, no parameters).

- Expected: an Iterate with:
  - inherited true;
  - seed: function, block at 0x200;
  - count: function, block at 0x300;
  - output1: the Quadrant at 0x400;
  - no output0.
- The reader must not treat 0x300 as an output, and must not read the
  word at 0x300 as a node.

**T2. An inline count with a seed function.** At 0x100: `0x0000062B`,
`0x00000200`, `0x00000400`.

- Expected: an Iterate with count 6 (inline), seed block at 0x200, and
  output1 at 0x400.

**T3. Output0 after output1.** At 0x100: `0x000001EB`, `0x00000200`,
`0x00000300`, `0x00000400`, `0x00000500`.

- Expected: seed block 0x200, count block 0x300, output1 0x400, output0
  0x500.

**T4. A Switch.** At 0x100: `0x000000A9`, `0x00000200`, `0x00000300`,
`0x00000400`, `0x00000500`.

- Expected: a Switch with:
  - inherited true;
  - seed block 0x200;
  - value block 0x300;
  - output0 0x400;
  - output1 0x500.
- `0x00000069`, `0x00000200`, `0x00000400`, `0x00000500` is a Switch
  with a constant true value, seed block 0x200, output0 0x400 and output1
  0x500.

**T5. A Quadrant.** At 0x100: `0x02000168`, `0x00000108`. At 0x108:
`0x00000200`, `0x3E99999A`, `0x3E99999A`.

- Expected: a Quadrant drawing pattern 1 (disc), with:
  - seed: function, block at 0x200;
  - pattern size inline (0.3, 0.3);
  - no children.

**T6. A seed value from a float conversion.** The seed block is B_SEED3.

- The seed is 0x40400000.
- With incoming state 5:
  - inherited: the effective state is 0x40400005;
  - not inherited: it is 0x40400000.
- The first `rand(1)` drawn from 0x40400000 on the frame-offset lane is
  0.25302231 (`docs/random-generator.md`).

**T7. Integer add on patterns.** The block is B_ADD1,
`const float 3.0; convert int (#0); const int 1; add int (#1, #2)`.

- The seed is 0x40400001. It is not 0x40800000, so it differs from
  a block that converts 4.0.
- Change the constants:

  | float constant | int constant | seed |
  |---|---|---|
  | 16777216.0 | 3 | 0x4B800003 |
  | 3.0 | 16777216 | 0x41400000 |

**T8. Floor.** In B_SEED3, replace 3.0 with:

| constant | seed (low 24 bits) |
|---|---|
| 2.04 | 0x000000 (the pattern of 2.0) |
| −2.49996 | 0x400000 (the pattern of −3.0, 0xC0400000) |

**T9. An unset integer variable.** The seed block is B_INT0, and nothing
sets variable 0. The seed is 0. A non-inheriting node's effective state
is 0.

**T10. Per-visit evaluation and order.** The tree:

- at 0x100, an Iterate: `0x0000030B`, `0x00000200` (count 3, inherited,
  output1 at 0x200);
- at 0x200, a Switch: `0x00000089`, `0x00000600`, `0x00000000`,
  `0x00000300` (value block at 0x600, which is B_SETNUM; output1 at
  0x300);
- at 0x300, an Iterate: `0x00000123`, `0x00000700`, `0x00000400` (count
  1 inline, seed function, not inherited; seed block at 0x700, which is
  B_SEEDV0; output1 at 0x400);
- at 0x400, `0x00000048`.

At root incoming state 0, the outer passes p = 0, 1 and 2 enter the
Switch with states 0x000038C5, 0x5C49B922 and 0xECB3A657. The inner
Iterate's effective states are:

| pass | effective state |
|---|---|
| 0 | 0x00000000 |
| 1 | 0x3F800000 |
| 2 | 0x40000000 |

Its single pass hands the leaf:

| pass | state to the leaf |
|---|---|
| 0 | 0x000038C5 |
| 1 | 0xE38038C5 |
| 2 | 0x400038C5 |

The first frame-offset `rand(1)` pairs are:

| pass | first pair |
|---|---|
| 0 | (0.95981741, 0.48948157) |
| 1 | (0.45981741, 0.98948157) |
| 2 | equal to pass 0 (the states differ only in bit 30) |

**T11. Order within a node.** Take an Iterate (0x000000AB) whose seed
block sets variable 0 to 6.0 and returns an integer. Its count block is
`getvar var 0; convert`. The count read is 6. Reading 0 would mean the
count ran first, which is wrong.

**T12. Rejections.**

- A tree-node word whose bits 4-5 are 0b11 (for example `0x000000BB`,
  `0x00000039` or `0x00000038`) is malformed: report it, do not guess.
- A seed slot that is not an offset into the node is also malformed.
- So is a seed block whose last instruction is not an int of width 1.
- The word 0x09000010 reached as a node is an error. With the layout
  above, it is never reached in the 35 archives.

## 6. Unknown

- **Seed state 3** (bits 4-5 = 0b11) never occurs, and no `.sbs`
  setting tried produces it.
- **A seed function that reads the built-in `$number` directly.**
  Reading it through a variable is fully measured and is the only shape
  in the corpus. Reading it in the seed block itself was measured only in
  part:
  - `convert($number × k)` matches the rule on passes 0 and 1.
  - `convert($number + c)` (with `c = graph input × const`) matches on
    pass 0 only. On later passes it renders as a state whose low 22 bits
    are 0, and that state does not follow c + p:

    | c | pass-1 seed's low 24 bits |
    |---|---|
    | 1, 4, 5, 6, 7 | 0x000000 |
    | 2, 9, 10, 20, 100, 1000, 12582912, 12595257 | 0x800000 |
    | 3 | 0xC00000 |

    In counts 3 and 4, the later passes repeat one of the first two
    states. This happens on both Iterate and Switch seeds, and it is not
    explained. Do not rely on direct `$number` in a seed block.
- **`rand` inside a seed block.** It never occurs in the corpus. Which
  state its generator starts from (the incoming state or 0, and which
  lane) is untested.
- **Other integer ops on float-converted values.** Sub, mul, div, mod
  and compare were not probed. Neither were other consumers of a
  converted integer (count, Switch value, image index) seeing the float
  pattern. Counts built as `convert(x)` behave as the number floor(x)
  (6 passes for 6.0). How a count built as `convert(x) + 1` reads is
  untested.
- **Floor versus truncation for negatives** rests on one sample
  (−2.49996). `docs/function-ops.md` measured truncation in a Value
  Processor. The FX-Map seed context may differ, or the sample may be
  misleading.
- **Bits 24-31 of a seed** cannot be observed in any render (2.1). The
  table entries are exact 32-bit statements only as far as the inferred
  value model goes.
- **Switch-seed-before-value and Quadrant-seed-before-parameters** were
  each measured with one probe. The relative order of the Quadrant
  parameter functions among themselves is outside this document.
