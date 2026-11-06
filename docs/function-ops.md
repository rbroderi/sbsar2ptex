# Substance Function Ops

## Clean-room behavioral specification

This document specifies function bytecode ops that sbsar2ptex.function
does not yet describe. It is intended to be sufficient for an
independent implementation without access to the original
implementation. It was recovered from three sources. No engine code is
reproduced here.

- Function graphs cooked with `sbscooker` (sbsar2ptex.probe).
- Renders of those graphs with `sbsrender` on each engine (`--engine vk`,
  `d3d11` and `sse2`).
- Compiled graphs found in the wild, cited by node number only.

Each statement is tagged:

- **[measured]**: established by cooking or rendering probes.
- **[inferred]**: fits the probes, but was not tested directly.

## 0. Bytecode basics

This summary is all the rest of this document assumes.

**Block.** A function compiles to a code block:

- The block starts with a u16, the instruction count.
- The instructions follow. The block's value is its last instruction's
  value.
- The values are in SSA form. Instruction `k` defines value `#k`, and
  operands name earlier values by number.

**Instruction.** An instruction is a u16 opcode, then its operand u16s.
The opcode is `count << 10 | type << 8 | (width - 1) << 6 | op`:

- `count` is the number of operand u16s that follow.
- `type` is 0 for bool, 1 for float and 2 for integer. A bool's width
  field reads 1.
- `width` is 1 to 4 components.

A u32 or f32 operand, such as a constant, a uid or this document's loop
cap, is 4-byte aligned in the file. A padding u16 comes first when
needed, and it is included in `count`.

**Ops used in this document:**

| op | name | operands | value |
|---|---|---|---|
| 0x00 | const | the constant | the constant |
| 0x02 | graph input | the input's uid, a u32 | the input's current value |
| 0x04 | get variable | variable id | the variable's current value (0 if never set) |
| 0x07 | set variable | value index, variable id | stores the value, and is that value |
| 0x09 | if/else | c, a, b | a if c is true, else b |
| 0x0C | sequence | a, b | b (a is run first) |
| 0x0D, 0x0E, 0x0F | vector | 2, 3 or 4 values | their components, joined in order |
| 0x10 | swizzle | a, selector | component `selector >> 2k & 3` of a, for each result component k |
| 0x11 | convert | a | a converted to the instruction's type (float to int truncates toward zero) |
| 0x12 | add | a, b | a + b |
| 0x13 | sub | a, b | a - b |
| 0x14 | mul | a, b | a * b |
| 0x15 | div | a, b | a / b |
| 0x1D | == | a, b | bool |
| 0x1E | != | a, b | bool |
| 0x1F | > | a, b | bool |
| 0x20 | >= | a, b | bool |
| 0x21 | < | a, b | bool |

Float values are float32 and integer values are 32-bit two's complement.
A bool constant takes one u16 operand (0 is false), with no padding.

Variables are named by small integer ids, which the cooker assigns. They
live for the whole evaluation of the block, so a value set in one place
is read later by a get.

## 1. While Loop (op 0x0B)

### 1.1 In the .sbs file

**[measured]** A While Loop is the function node `while`:

- Its inputs are `init`, `cond` and `loop`.
- Its funcData `while` holds the iteration cap as a `constantValueInt1`.
- Its type is the loop's value type: float or integer, of any width the
  graph uses.

### 1.2 Encoding

**[measured]** The instruction is opcode `0x0B` with the value's type and
width in the usual bits. Its operands are:

1. `init`, a u16 SSA index;
2. `cond`, a u16 SSA index (a bool);
3. `body`, a u16 SSA index;
4. the cap, a u32, 4-byte aligned. A padding u16 comes before it when the
   three u16s leave it unaligned.

So the operand count is 5 or 6. Seen: `[7, 15, 20, 0, 3, 0]` (padded, cap
3) and `[4, 12, 19, 0, 0]` (unpadded, cap 0).

The loop's instructions lie in three consecutive ranges before it:

- the init range, ending at `init`;
- the condition range, `(init, cond]`;
- the body range, `(cond, body]`.

The loop instruction comes right after `body`. Each range is run as a
whole, every time its part of the loop runs.

**Cap values:**

- A cap of -1 in the .sbs file cooks to 0, and 0 means unlimited. 300
  iterations ran with no cap on every engine.
- A cap of 0 in the .sbs file cooks to no loop at all. The node's value
  is then its init's value. In the probe, init's value was 7, and the
  result was 7.

### 1.3 How the cooker returns the loop's value

**[measured]** The cooker does not use the loop instruction's own value.
It makes a hidden variable, numbered after the graph's own variables,
for the result:

- The init range ends by setting the hidden variable to 0.
- The body range ends by setting it to the body's value.
- After the loop, a `sequence` (loop, then get hidden variable) gives the
  node's value.

So a loop that runs no iterations returns 0, not its init's value. A
loop that runs returns its last body's value.

**[measured]** In these probes, a value the condition and the body share
is copied: its instructions appear in both ranges and run in each. If
the shared value sets a variable, that happens twice per iteration.

- **Probe.** The condition is `c = c + 1; c >= 4`, and the body is that
  same `c = c + 1` node times 3. The result was 12.
- **Trace.**
  1. The condition makes `c` 1.
  2. The body makes it 2 and gives 6.
  3. The condition makes it 3.
  4. The body makes it 4 and gives 12.
  5. The condition makes it 5 and exits.

**[inferred]** Another cooker can reference the condition's SSA value
from the body instead of copying it. stripe-new-1's n179 does this: its
body instruction 540 reads 532, which is set in the condition range. The
body then sees the value the condition computed in the same iteration.
An evaluator should allow both.

### 1.4 Semantics

**[measured]** The loop runs as follows:

1. Run the init range once.
2. Repeat:
   1. If a cap is set (non-zero) and the iterations run so far have
      reached it, stop. The condition is not run again.
   2. Run the condition range. If its value is true, stop.
   3. Run the body range. Count one iteration.

Variables persist across iterations and after the loop. Every probe
gave this on vk, d3d11 and sse2:

| probe (`i` from 7, `i += 10` per body, `k` counts condition runs) | value | `k` | `i` |
|---|---|---|---|
| stop at `i >= 0`, no cap (zero iterations) | 0 | 1 | 7 |
| stop at `i >= 30`, no cap | 37 | 4 | 37 |
| stop at `i >= 1000`, cap 3 | 37 | 3 | 37 |
| stop at `i >= 30`, cap 3 (the cap and the stop coincide) | 37 | 3 | - |
| stop at `i >= 1000`, cap 1 | 17 | - | - |
| cap 0 in the .sbs file (no loop cooked) | 7 | - | - |
| integer: `j = 0`, stop at `j >= 5`, `j += 2` | 6 | - | - |
| 300 iterations, no cap | - | 301 | - |

### 1.5 Engines and paths

**[measured]** Two paths were tested:

- A **pixel processor**, which the GPU engines run per pixel. Its
  results were identical to 16 bits on vk, d3d11 and sse2.
- A **uniform color** whose color is the same function, which is a node
  parameter. That function cooks with its own copies of the loop and
  runs once per node. Its results were the same on all three engines,
  read at 8 bits.

So the cap and zero-iteration rules hold on the CPU (sse2) and on both
GPU engines.

### 1.6 Seen in

stripe-new-1 (cooked by a v9 cooker): pixel processors n158, n179, n205,
n579, n600, n619 and n641, all with cap 13.

### 1.7 Implementation notes

**Evaluator** [inferred from 1.4]. Do not walk a block's instructions
once, in order. Run them in the order the loop runs them:

    run the init range (instructions up to init)
    n = 0
    repeat:
        if cap != 0 and n >= cap: stop
        run the condition range (init, cond]
        if value[cond] is true: stop
        run the body range (cond, body]
        n = n + 1
    value[loop] = value[body] if n > 0, else the loop type's zero
    continue after the loop instruction

Storage:

- Each SSA value is overwritten each time its instruction runs, so values
  need indexed storage, not an append-only list.
- A read sees the most recent run.
- The cooker never consumes the loop instruction's own value (1.3).
  Giving it the hidden variable's value is a safe convention.

Limits:

- A range can hold a loop of its own. Nesting was not probed, but the
  rule above composes.
- With no cap, an offline evaluator should give up after some large
  count, where Substance would hang, and treat the value as unknown.

**GLSL.** Translate the condition and body ranges into one loop
statement:

1. Before the loop, declare the result, set to the type's zero.
2. Loop `cap` times, or without bound when the cap is 0.
3. In each iteration, emit the condition range, then break if its value
   is true.
4. Then emit the body range, and assign the body's value to the result.

Two scoping rules:

1. A variable (ops 0x04 and 0x07) first written inside a loop must be
   declared before the outermost enclosing loop, set to zero. Otherwise
   it would reset each iteration and would not exist after the loop. The
   after-loop `get` of the hidden variable needs it.
2. An SSA value computed inside the loop that is read outside it must
   also be declared before the outermost loop, and only assigned inside.

The cap is a u32: clamp it to 2^31 - 1 for a GLSL int. For bounds
analysis, a variable written in a loop is unbounded from the loop's
start onwards, and so is the loop's value.

### 1.8 Acceptance tests (synthetic bytecode)

**Test A: a counting loop in the cooker's form.** The cooker produced
this form in the 1.4 probes. Variables: v0 is `i`, v1 is `k` (it counts
condition runs) and v2 is the hidden result. Values are float, width 1,
except where marked.

    #0  const 0.0
    #1  const 7.0
    #2  setvar v0 <- #1          (i = 7)
    #3  setvar v1 <- #0          (k = 0)
    #4  setvar v2 <- #0          (hidden = 0)
    #5  sequence (#2, #3)
    #6  sequence (#5, #4)        init = 6
    #7  getvar v1
    #8  const 1.0
    #9  add (#7, #8)
    #10 setvar v1 <- #9          (k += 1)
    #11 getvar v0
    #12 const L
    #13 gteq (#11, #12)  bool
    #14 sequence (#10, #13) bool cond = 14
    #15 getvar v0
    #16 const 10.0
    #17 add (#15, #16)
    #18 setvar v0 <- #17         (i += 10)
    #19 setvar v2 <- #18         body = 19
    #20 while (6, 14, 19, CAP)   operands [6, 14, 19, cap_lo, cap_hi]
                                 or, padded, [6, 14, 19, 0, cap_lo, cap_hi]
    #21 getvar v2
    #22 sequence (#20, #21)      the block's value

The first five rows were measured on vk, d3d11 and sse2 (1.4, and
`L = 25`, which gave 27 with `k = 3`). The cooker chooses the padding
itself, so the padded row follows from the encoding rule (1.2):

| L | CAP | block value | k (v1) | i (v0) |
|---|---|---|---|---|
| 0 | 0 | 0 | 1 | 7 |
| 30 | 0 | 37 | 4 | 37 |
| 1000 | 3 | 37 | 3 | 37 |
| 1000 | 1 | 17 | 1 | 17 |
| 25 | 0 | 27 | 3 | 27 |
| 1000 | 3, padded encoding | 37 | 3 | 37 |

**Test B: the body reads the condition's SSA value.** This is
stripe-new-1 n179's form ([inferred], 1.3).

- Init: v1 = 0, v3 = 0, v2 (hidden) = 0.
- Condition: `#c = setvar v1 <- getvar v1 + 1`, then `#c > 3`.
- Body: `v3 <- getvar v3 + #c`, using the condition's own SSA value, then
  `v2 <- that`.
- After the loop: `sequence(loop, getvar v2)`.

Expected: 6 (1 + 2 + 3), with v1 = 4. The body must see the value the
condition computed in the same iteration.

**Test C: GLSL.** Test A's GLSL, wrapped in a function, must meet these
checks:

- It compiles with glslangValidator.
- It declares v0, v1 and v2 once each, before the loop.
- It holds exactly one `break`, guarded by #14's value and placed before
  the body's statements.
- With CAP 3 the loop is bounded by 3. With CAP 0 it is unbounded.

**Test D: a real file.** An implementation must decode, evaluate and
translate the loops of stripe-new-1's pixel processors n158, n179, n205,
n579, n600, n619 and n641 without error. Each block has 1157 to 3413
instructions and cap 13. A prototype of the GLSL form above compiled for
all seven.

### 1.9 Unknowns

- Nested loops: neither probed nor seen.
- Whether a cooker ever reads a condition-range or body-range SSA value
  after the loop. Not seen: results pass through variables.
- Caps above 2^31 - 1. Integer caps up to 16777215 appear in Designer's
  own packages.
- The loop instruction's own value when consumed directly: never seen
  consumed.

## 2. Value slots: get slot (op 0x03) and set slot (op 0x06)

### 2.1 What they are

**[measured]** Designer's *values* pass a number from one node to
another. A Value Processor node computes a value with its function, and
another node reads it through a value input (an identifier starting
with `#`, read in its function with `get_float1 "#name"`, `get_integer2
"#name"` and so on). The cooker compiles this to a graph-wide store of
32-bit words, the **slot store**:

- The writer's function ends with **set slot** (op 0x06), which stores
  its value at a word offset in the store.
- Every function that reads the value uses **get slot** (op 0x03) with
  the same word offset.

Slots are not variables (ops 0x04 and 0x07):

| | variables (0x04, 0x07) | slots (0x03, 0x06) |
|---|---|---|
| scope | one node's functions | the whole graph |
| id | a small variable id | a word offset into the slot store |
| written by | any function | only a Value Processor's function, once |

**[measured]** A variable set in a Value Processor's function is not
visible in the reading node's function. The probe set `q = 5` in the
writer and read `q` in the reader, and the reader saw 0 (test S9).

### 2.2 Encoding

**[measured]** Both ops carry the value's type and width in the usual
opcode bits. A bool's width field reads 1, as everywhere.

**Get slot, op 0x03.** It has one operand u16, the slot: a word offset
into the slot store. Its value is the slot's contents, with the
instruction's type and width.

| value | opcode |
|---|---|
| float, width 1 | `0x0503` |
| float, width 2 | `0x0543` |
| float, width 3 | `0x0583` |
| int, width 1 | `0x0603` |
| int, width 2 | `0x0643` |
| bool | `0x0443` |

**Set slot, op 0x06.** It has two operand u16s: first the SSA index of
the value to store, then the slot. Its type and width are the stored
value's. Its value is the stored value ([inferred]: the cooker always
makes it the block's last instruction, so the block's value is the
stored value, and nothing else reads it).

| value | opcode |
|---|---|
| float, width 1 | `0x0906` |
| float, width 3 | `0x0986` |
| int, width 1 | `0x0A06` |
| int, width 2 | `0x0A46` |
| bool | `0x0846` |

Neither op has a u32 operand, so neither is ever padded.

**Example u16 streams** (each starts with the block's instruction
count):

- `0001 0503 0033`: get slot, float, slot 51 (0x33).
- `0002 0900 0000 3F80 0906 0000 0033`: const float 1.0, then set slot
  (#0 into slot 51).
- `0002 0440 0000 0846 0000 003E`: const bool false, then set slot (#0
  into slot 62).
- `0003 0643 0004 0A50 0000 0004 0A46 0001 000C`: get slot int2 from slot
  4; swizzle int2 of #0 with selector 4 (components x, y); set slot (#1
  into slot 12).

### 2.3 The slot store

**[measured]** The store is an array of 32-bit words, one array for the
whole graph.

- A value of width `w` occupies the `w` consecutive words from its slot.
  A bool takes one word.
- The cooker packs values without gaps or overlaps. In a probe with five
  values, the cooker placed a float3 at 0, a float at 3, an int2 at 4, a
  float4 at 6 and a bool at 10.
- Float components hold float32 values, and int components hold int32
  values.
- Values pass through exactly. A float written as `x * 0.1` compared
  equal to the same product computed in the reader, for x = 3 and x =
  0.7. +infinity also passed through (S11).

**[measured]** In all 142 slots of the test corpus:

- every slot is written by exactly one node;
- the set slot is the last instruction of that node's block;
- every read has the same type and width as the write;
- the writing node comes earlier in the node table than every node that
  reads the slot;
- no slot is read without being written;
- no two slots overlap.

**[measured]** The cooker enforces the type match. A reader whose type
differs from the writer's (float against int, float against bool,
float against float3, or the reverse) fails to cook with "Input value
connection type mismatch". So a get slot never reinterprets a value: it
reads what the set slot of the same type and width stored.

**[measured]** A value input that nothing feeds is folded by the cooker
into a constant zero. No get slot is emitted.

### 2.4 Semantics

**Evaluation order** [measured, from the corpus rules in 2.3]. Run the
graph's nodes in node-table order. Each node's functions see every slot
written by earlier nodes. The writer's block runs once per graph
evaluation [inferred: its `$pos` is (0, 0), and its value is the same in
every pixel of a reader].

**Get slot.** Its value is the `w` words from the slot, as the
instruction's type: a float, an int or a bool. In a pixel processor the
value is the same for every pixel. **[measured]** Two get slots of the
same slot in one block both read the stored value (S8).

**Set slot.** It stores its operand's value at the slot, overwriting
the words, and its value is that value [inferred].

**Where the ops appear** [measured]:

- Set slot appears only in Value Processor nodes. In the .sbsasm these
  are atomic op 40 (the pixel processor's op) whose first field has the
  value's width in its high 16 bits and the image-input count in its low
  16 bits: 0x10000, 0x20000, 0x30000, 0x40000, or 0x20001 with one image
  input. The block with the set slot is the one named by the last of
  the node's code offsets, which is the earliest block in the file.
- Set slot never appears in atomic op 40's color twin, op 41. No op-41
  node in the corpus has a non-zero high half in its first field, and
  float3 and float4 Value Processors cooked to op 40 (fields 0x30000 and
  0x40000) in probes. Treat an op-41 node as a pixel processor, never as
  a Value Processor.
- Get slot appears in:
  - pixel processors' per-pixel functions;
  - FX-Map functions;
  - other nodes' parameter functions, such as a uniform color's color
    and a blend's parameters;
  - output-size blocks, as in stripe-new-1's n183 (2.7);
  - other Value Processors' functions, which chain values.

**A Value Processor's code offsets** [measured, all 142 writers in the
corpus]. In the order the node lists them:

1. **Seed block.** Usually a single graph-input read of `$randomseed`
   (int). One writer (RiberStones n3394) has a 3-instruction function
   here instead.
2. **Output-size block**, when present. Usually a single graph-input
   read of `$outputsize` (int2). Some writers have a longer function here
   (38 to 253 instructions), which is [inferred] the node's output-size
   function, like other nodes' size blocks.
3. **Set-slot block**, always last in the list and always earliest in
   the file. Its last instruction is the set slot.

A node-word flags value of 0x88 has only blocks 1 and 3 (stripe-new-1's
n159). The flags values 0x89, 0x99 and 0xB9 have all three. Bits 6 and 7
(value 2) mean "the seed is a function", as on other nodes.

**[measured]** The corpus also has 27 op-40 nodes with a width in their
first field but no set slot. Their last block is a single get slot, as
in stripe-new-1's n1, n3, …, n25. Their role is unknown (2.10). They
write nothing, so an evaluator can run them like any other block.

**Built-ins inside a Value Processor's function** [measured on vk,
d3d11 and sse2; reads cooked as op 0x01 unless noted]:

| built-in | id | value |
|---|---|---|
| `$size` | 1 | the graph's output size (width, height): (4, 4) at `$outputsize` (2, 2), (8, 2) at (3, 1) |
| `$sizelog2` | 3 | the graph's log2 output size: (2, 2) at (2, 2), (3, 1) at (3, 1) |
| `$pos` | 8 | (0, 0) |
| `$number` | 10 | 0 |
| `$depth` | 12 | 0 |
| `$time` | 0 | 0 |
| `$tiling` | - | folded by the cooker to a constant int (3 in the probe graph) |
| `$normalformat` | - | cooked as a graph input (op 0x02, uid 0x80000001), not a built-in; it read 0 |

`$size` and `$sizelog2` stay the graph's size even when the Value
Processor has an image input. In the probe, the input was a uniform set
to an absolute 8 x 8 and the graph was 4 x 4, and `$size` read 4 and
`$sizelog2` read 2. That the input really was 8 x 8 was not checked.

**Trig inside a Value Processor** [measured]. A Value Processor computes
sin and cos with the cooker's CPU polynomial, the one other
once-per-node functions use, not with the GPU's native functions. The
probe measured `(v - S) * 10^6`, where S is the float32 rounding of the
true value:

| function | x | CPU polynomial predicts | Value Processor, d3d11 / sse2 | Value Processor, vk | pixel processor, vk / d3d11 |
|---|---|---|---|---|---|
| sin | 1.382 | 14.19 | 14.19 | 14.22 | 0.03 / 0.00 |
| cos | 0.6 | 3.34 | 3.35 | 3.33 | -0.06 / -0.06 |

The output step is 0.016. So d3d11 and sse2 match the polynomial within
the step, and vk is within one float32 ulp of it. On sse2, a pixel
processor also gives the polynomial's values.

**Engines** [measured]. Every probe gave the same value on vk, d3d11
and sse2, with two exceptions:

- A NaN in the slot compared unequal to itself on d3d11 and sse2, but
  equal on vk. This looks like vk's comparison of NaN, not the slot.
- A Value Processor using `rand` gave a different value on sse2 (see
  2.10).
- A Value Processor's sin and cos on vk were up to one ulp off d3d11 and
  sse2 (see the trig table above).

### 2.5 Operands that name no value

**[measured]** In one probe, the reader's function read a variable that
only the writer had set. The cooker compiled the read as an add whose
second operand was 0xFFFF, which names no value (NO_VALUE). The add's
result was its other operand: 6 + 0 = 6 on all three engines.

**[measured]** The cooker emits 0xFFFF wherever a function reads a
variable that is never set, and the read's consumer is not folded away.
Probes on vk, d3d11 and sse2 used live inputs x = 3 and y = 0.5, with N
the unset read.

**Value ops.** N reads as the type's zero, broadcast to the other
operand's width:

| expression | operands | result |
|---|---|---|
| N + x | add [N, #] | 3 |
| N - x, x - N | sub, both sides | -3, 3 |
| N * x | mul | 0 |
| N / x | div [N, #] | 0 |
| x / N | div [#, N] | +infinity |
| mod(N, x) | mod | 0 |
| min(N, x), max(N, -x) | min, max | 0, 0 |
| N > -x | gt | true |
| N == y | eq | false |
| lerp(N, x, y) | lerp [N, #, #] | 1.5 |
| pow(x, N) | pow [#, N] | 1 |
| dot(N2, (x, y)) | dot [N, #] (N2 an unset float2) | 0 |
| if (x > y) N else x | if/else [#, N, #] | 0 |
| N + int(x), integer | add, integer | 3 |

**Vector ops (0x0D to 0x0F).** N contributes no components. The
remaining components are joined in order, and the result is padded with
zeros to the instruction's width:

| expression | operands | result |
|---|---|---|
| vector2(N, x) | [N, #] | (3, 0) |
| vector2(x, N) | [#, N] | (3, 0) |
| vector3(vector2(y, N), x) | inner [#, N], then outer | (·, 0, 3); the x component was not read |

**Unary ops.** neg, abs, floor, cos and the int-to-float convert of an
unset read were folded by the cooker into constants (-0.0, 0, 0, 1, 0).
No unary op was seen with an N operand.

**In the corpus**, N appears only as a sequence's second operand (the
dropped result, which function.py already handles; persian n1501). No
arithmetic op in a real file has it.

An evaluator must not index its value list with 0xFFFF. Every listed op
must accept it. For an op not listed, treating N as zero is
[inferred].

### 2.6 Implementation notes

**Evaluator** [inferred from 2.3 and 2.4]:

- Give evaluation a slot store, a map from slot to the list of the
  value's components, shared by every block of the graph. It works like
  the variable store, but at graph scope.
- Set slot stores `value[operand 0]` under `operand 1`. Get slot returns
  the stored list, rounded to its type as every op's result is.
- Run the Value Processors' blocks in node-table order before any block
  that reads their slots. Running all of a graph's blocks in node order
  satisfies this.
- If a slot was never written, which never happens in cooked files, use
  the type's zero.
- A value is one stored list per slot. Since reads always match writes
  (2.3), reinterpreting words is not needed. An implementation that
  keeps a flat word array instead must store floats as float32 bits.

**GLSL** [inferred]. A get slot's value is the value the writer's set
slot stores. The writer's function may depend on graph inputs, so keep
it live; do not freeze its value at the defaults:

- Translate the writer's block as a function of the graph inputs.
  Before its set slot, any get slot inside it resolves to its own
  writer, recursively.
- In the reader, replace get slot with that expression (or with a shared
  global computed once from it), of the get slot's GLSL type.
- A per-pixel reader sees it as a uniform. It never depends on the
  reader's `$pos`.

Give the translator a resolver callback, like the existing inputs and
built-ins callbacks: given (slot, type, width), it returns the GLSL
expression. With no resolver, a get slot cannot be translated.

**Library graphs and instances** [measured]. Slot numbers belong to the
cooked assembly, not to the graph that was authored:

- Designer's own library uses slots. In standalone cooks of the shipped
  packages, set slots and get slots appear in atlas_scatter, auto_crop,
  constant_values, floodfill_mapper, floodfill_to_index, min_max,
  multi_directional_warp and value_get_size (38 sets, 68 gets). So a
  template cooked from one of these reads and writes slots.
- When a graph is instanced, the cooker inlines the instance's nodes and
  gives its Value Processors new slots in the instancing assembly.

**Probe.** Graph `sub` has a Value Processor writing `k + 1`, read by
a pixel processor. Graph `main` has its own Value Processor (20.0) and
two instances of `sub`, with k = 3 and k = 5. One cook of the package
gave one assembly holding both graphs, with these slots:

| writer | value | slot |
|---|---|---|
| `sub`'s own nodes (its standalone graph) | k + 1, live | 0 |
| `main`'s Value Processor | 20 | 3 |
| instance with k = 3 | 4 (folded to a constant) | 2 |
| instance with k = 5 | 6 (folded to a constant) | 1 |

Renders gave 4, 6, 20 and 4 on all three engines. So:

- The same template slot became 2 in one instance and 1 in the other,
  and neither matches its standalone number, 0.
- The cooker folds an instance's Value Processor when its inputs are
  constant.

**Mapping a template's slots to an instance's** [inferred]. Match nodes,
not slot numbers. A template's set slot s belongs to a template Value
Processor node. The node matched to it in the instance has its own set
slot s'. Rewrite each of the template's get slot s as get slot s', or
resolve it to the matched writer directly. Neither the slot number nor
the slots' order carries over: in the probe, the two instances' slots
came out in reverse order. A template whose writer was folded to a
constant in the instance still maps one to one, writer to writer.

**Bounds.** A get slot's bound is the writer's value's bound, when
known, or unbounded.

### 2.7 Seen in

- **stripe-new-1** (v9 cooker): 49 get slots and 35 set slots. Slots 51
  to 63 are written by n0, n2, ..., n24 and read by n1, n3, ..., n25.
  Slot 4 (int2) is written by n159 and read by n160, n161, n180, n207,
  n580, n601, n620 and n624. Pixel processor n160's per-pixel function
  reads slot 4. Output-size blocks of n164, n183, n212, n583, n604, n623
  and n644 read slots. Float3 slots 22 to 40 chain through n653 to
  n660.
- **mosaic**: 457 get slots and 54 set slots, including FX-Map n11 and
  pixel processor n8 (get slot int, slot 12).
- **persian-mosaic-wall-tiels**: 53 get slots and 15 set slots.
- **Pack graphs** (get / set): Concrete 6 / 3, Mud 4 / 2, OvergrownRock
  19 / 13, Plaster 8 / 4, RiberStones 1 / 1, Rock_Chunky_Desert_Cliff
  3 / 2, Rock_Exfoliating_Cliff 4 / 3, Rock_Layered_Cliff 5 / 4,
  Rock_Sharp_Cliff 9 / 6. Their reads are mostly in parameter functions,
  whose blocks follow the node's fixed code offsets.

The rest of the test corpus uses neither op.

### 2.8 Acceptance tests (synthetic bytecode)

Notation as in 1.8. `U` is a float graph input's uid, and `x` is its
value. A test runs the blocks it lists in order, with one slot store
shared between them. Each block gets its own fresh variable store.
Expected values come from probes rendered on vk, d3d11 and sse2,
except where marked.

**S1: decoding.** The four streams in 2.2 must decode to:

1. `getslot f1 [51]`
2. `const f1 1.0`; `setslot f1 [0, 51]`
3. `const b false`; `setslot b [0, 62]`
4. `getslot i2 [4]`; `swizzle i2 [0, 4]`; `setslot i2 [1, 12]`

Each instruction's length must follow from its opcode alone.

**S2: a constant value.**

    W: #0 const f1 7.25
       #1 setslot f1 (#0, slot 0)
    R: #0 getslot f1 (slot 0)
       #1 const f1 100.0
       #2 add (#0, #1)

W's value is 7.25, the store holds slot 0 = [7.25], and R's value is
107.25.

**S3: a live input.** W = `input U; const 1.0; add (#0, #1); setslot f1
(#2, slot 0)`. R = `getslot f1 (slot 0)`.

| x | R |
|---|---|
| 3.5 | 4.5 |
| 10.25 | 11.25 |

**S4: integers.**

- W = `const i1 -9; setslot i1 (#0, slot 0)`. R = `getslot i1 (slot 0);
  convert f1 (#0)` gives -9.0.
- W = `const i1 -2147483648; setslot i1 (#0, slot 0)`. R = `getslot i1
  (slot 0); const i1 0; lt (#0, #1)` gives true. The cooker folded
  2147483647 + 1 to this constant.

**S5: bool.** W = `const b true; setslot b (#0, slot 0)`. R = `getslot
b (slot 0); const 1.0; const 0.0; ifelse (#0, #1, #2)` gives 1.0. With
`const b false` in W, R gives 0.0 [inferred].

**S6: layout and mixed widths.** Five writers, run in this order:

    W1: input U; vector2 (#0, #0) f2; vector3 (#1, #0) f3; setslot f3 (#2, slot 0)
    W2: input U; const 1.0; add (#0, #1); setslot f1 (#2, slot 3)
    W3: input U; vector2 (#0, #0) f2; convert i2 (#1); setslot i2 (#2, slot 4)
    W4: input U; vector2 (#0, #0) f2; vector3 (#1, #0) f3; vector4 (#2, #0) f4; setslot f4 (#3, slot 6)
    W5: input U; const 1.0; gt (#0, #1) b; setslot b (#2, slot 10)
    R:  #0 getslot f3 (slot 0)
        #1 swizzle f1 (#0, selector 2)        z
        #2 getslot f1 (slot 3)
        #3 add (#1, #2)
        #4 getslot f4 (slot 6)
        #5 swizzle f1 (#4, selector 3)        w
        #6 getslot b (slot 10)
        #7 const 1.0
        #8 const 0.0
        #9 ifelse (#6, #7, #8)
        #10 add (#5, #9)
        #11 add (#3, #10)

| x | R | store words 0 to 10 |
|---|---|---|
| 2.0 | 8.0 | 2 2 2, 3, 2 2, 2 2 2 2, true |
| 0.5 | 2.5 | 0.5 0.5 0.5, 1.5, 0 0, 0.5 0.5 0.5 0.5, false |

The int2 words (0 0 at x = 0.5) follow from truncation [inferred]. The
probe's reader did not read them.

**S7: a chain.**

    W1: input U; const 1.0; add (#0, #1); setslot f1 (#2, slot 1)
    W2: getslot f1 (slot 1); const 10.0; mul (#0, #1); setslot f1 (#2, slot 0)
    R:  getslot f1 (slot 0)

| x | R |
|---|---|
| 0.5 | 15.0 |
| 2.0 | 30.0 |

Running W2 before W1 is an ordering error. An implementation must run
the writers in node order.

**S8: two reads of one slot.** W = x + 1 into slot 0, as in S3. R =
`getslot f1 (slot 0); getslot f1 (slot 0); add (#0, #1)`. With x = 2.0,
R gives 6.0.

**S9: variables do not cross blocks.**

    W: #0 const 5.0
       #1 setvar v0 <- #0
       #2 getvar v0
       #3 const 1.0
       #4 add (#2, #3)
       #5 sequence (#1, #4)
       #6 setslot f1 (#5, slot 0)
    R: #0 getslot f1 (slot 0)
       #1 add (#0, NO_VALUE)          operands [0, 0xFFFF], as the cooker emitted it

W's value is 6.0, and R's value is 6.0. A variant of R with `#1 getvar
v0; #2 add (#0, #1)` must also give 6.0, because R's variable store is
fresh.

**S10: exact transfer.** W = `input U; const 0.1; mul (#0, #1); setslot
f1 (#2, slot 0)`. R = `getslot f1 (slot 0); input U; const 0.1; mul
(#1, #2); eq (#0, #3)` gives true for x = 3.0 and x = 0.7.

**S11: special values.**

- W = `const 1.0; input U; div (#0, #1); setslot f1 (#2, slot 0)` with
  x = 0. R = `getslot; const 3e38; gt (#0, #1)` gives true (+infinity).
- W = `input U; input U; div (#0, #1); setslot` with x = 0. R =
  `getslot; getslot; neq (#0, #1)` gives true (NaN). This was measured
  on d3d11 and sse2; vk gave false (2.4).

**S12: GLSL.**

- R of S2, translated with a resolver that returns the literal
  `7.25`, must compile with glslangValidator and declare no variable for
  the slot.
- R of S6, translated with a resolver that returns `vec3(p)`, `p + 1.0`,
  `vec4(p)` and `p > 1.0` for slots 0, 3, 6 and 10, must compile and
  give R's values for p = 2.0 and p = 0.5.
- R with no resolver must fail with a clear error naming op 0x03.

**S13: decoding robustness.** A block whose arithmetic operand is
0xFFFF (S9's R) must decode and evaluate without an index error.

**S14: NO_VALUE operands** (2.5). Use `#0 input U` (x = 3.0), `#1 input
V` (y = 0.5) and N = 0xFFFF:

| block (after #0, #1) | value |
|---|---|
| `add [N, 0]` | 3.0 |
| `sub [N, 0]` / `sub [0, N]` | -3.0 / 3.0 |
| `mul [N, 0]` | 0.0 |
| `div [N, 0]` / `div [0, N]` | 0.0 / +inf |
| `mod [N, 0]` | 0.0 |
| `min [N, 0]` | 0.0 |
| `neg (#0)` as #2, then `max [N, 2]` | 0.0 |
| `neg (#0)` as #2, then `gt [N, 2]` | true |
| `eq [N, 1]` | false |
| `lerp [N, 0, 1]` | 1.5 |
| `pow [0, N]` | 1.0 |
| `gt [0, 1]` as #2, then `ifelse [2, N, 0]` | 0.0 |
| `vector2 f2 [N, 0]` | (3.0, 0.0) |
| `vector2 f2 [0, N]` | (3.0, 0.0) |
| `vector2 f2 [0, N]` as #2, then `vector3 f3 [2, 0]` | (3.0, 0.0, 3.0) |
| `vector2 f2 [0, 1]` as #2, then `dot [N, 2]` | 0.0 |
| `convert i1 (#0)` as #2, then `add i1 [N, 2]` | 3 |

In the vector3 row, the probe measured only the y and z components, as
0 and x.

**S15: built-ins in a Value Processor** (2.4). With built-ins bound for
a graph of `$outputsize` (3, 1), W = `builtin f2 (1); swizzle f1 (#0, 0);
setslot f1 (#1, slot 0)` stores 8.0. With swizzle selector 1, it stores
2.0. With built-in 3, it stores 3.0 and 1.0. With built-in 8, it stores
0.0.

**S16: CPU trig in a Value Processor** (2.4). W = `input U; sin (#0);
setslot f1 (#1, slot 0)` with x = 1.382 stores 0.98224503 (float32 bits
0x3F7B7469), not the correctly rounded sine 0.98223084 (0x3F7B737B).
With cos and x = 0.6, it stores 0.82533896 (0x3F53496A), not 0.82533562
(0x3F534932). These bit patterns are what the cooker's CPU polynomial,
already used for once-per-node functions, gives. The renders matched them
to within 2e-8 on d3d11 and sse2.

**S17: template slot mapping** (2.6). A template with writer node T
(set slot 0) and reader R (get slot 0) is matched to an instance whose
node matched to T sets slot 2. The instance's R must then read slot 2.
A second instance, whose matched writer sets slot 1, must read slot 1.
Expected values: 4 and 6 for k = 3 and k = 5, as in the probe.

### 2.9 End-to-end test (a real file)

**stripe-new-1.sbsar**, graph `Stripe new 1`. Run these blocks in node
order with one slot store:

1. n159's set-slot block (2.4): `const i2 (2048, 2048); setslot i2 (#0,
   slot 4)`. The store then holds words 4 and 5 = 2048, 2048.
2. n180's set-slot block: get slot 4, swizzle (x, y), set slot 2.
3. n182's set-slot block: get slot 2, swizzle (x, y), set slot 0.
4. n183's output-size block (its first code offset), 28 instructions.
   It starts with `getslot i2 (slot 0)` and ends with `convert i2`.

Expected: n183's block gives (11, 11). Each component is computed as
ceil(log(2048 / k) / ln 2), then 2^that, log2 of that, and max with 6,
where k is 1.1348939 for x and 1.2793764 for y.

Also:

- n1's block `getslot f1 (slot 51)`, run after n0's set-slot block
  (`const 1.0; setslot (#0, slot 51)`), gives 1.0.
- Pixel processor n160's per-pixel block (111 instructions, get slot
  int2 from slot 4 at #4) translates to GLSL without error, given a
  resolver for slot 4.
- Converting the whole file no longer fails with `cannot translate
  op0x3`.

### 2.10 Unknowns

- The value of a set slot when another instruction consumes it: never
  seen consumed.
- A get slot whose type or width differs from its writer's: the cooker
  refuses to cook it, so it is never seen.
- A slot written twice, or read before it is written: never seen.
- Value Processors with an image input (first field 0x20001), such as
  mosaic's n5:
  - Sampling inside them was not tested. In a probe, a Value Processor's
    `samplelum` of a uniform image input was folded by the cooker to
    0.0.
  - `$size` read the graph's size, but that the input image really had
    another size was not verified (2.4).
- `rand` in a Value Processor: rand(1000) gave 2.984 on vk and d3d11
  and 3.140 on sse2. Which seed and generator each path uses is unknown.
  The seed block (2.4) is presumably its seed.
- Why vk's trig in a Value Processor is up to one ulp off the CPU
  polynomial (2.4).
- A NaN slot's behaviour on vk (2.4).
- The 27 op-40 nodes that carry a width but write no slot, whose last
  block is a single get slot (stripe-new-1's n1, n3, …, n25;
  persian-mosaic-wall-tiels n1, …). Perhaps the graph's value outputs.
- Unary ops (neg, abs, floor, trig, convert) taking a 0xFFFF operand:
  the cooker folded every probe, so none was seen (2.5).
- A value input on an instanced graph that an outer Value Processor
  feeds: how its slot appears inside the instance was not probed (2.6).
- The middle (output-size) block of a Value Processor: its role is
  [inferred] from other nodes' size blocks. A Value Processor has no
  image output, so the size it computes was not observed.
