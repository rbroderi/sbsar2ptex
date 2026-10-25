# Substance Random Generators

## Clean-room behavioral specification

This document specifies the observable random-number behavior recovered from
Substance 3D Designer pixel-processor renders. It is intended to be sufficient
for an independent implementation without access to the original
implementation.

## 1. Data model

- All generator state and hash arithmetic uses unsigned 32-bit integers.
- Integer addition, left shift, and multiplication discard bits above bit 31.
- Right shifts are logical shifts that insert zero bits.
- `float32` means IEEE-754 binary32 using round-to-nearest, ties-to-even.
- Bit 0 is the least-significant bit.

Constants:

```text
UINT32_MASK = 0xFFFFFFFF
LCG_M        = 0x4937573B
LCG_C        = 0x0000C7C1
```

## 2. Pixel coordinates

Each pixel owns an independent generator state. For pixel `(x, y)` in an
output of `width` by `height` pixels, calculate its normalized center:

```text
position_x = float32((x + 0.5) / width)
position_y = float32((y + 0.5) / height)
```

Reinterpret each binary32 value as its unsigned 32-bit IEEE-754 bit pattern:

```text
position_x_bits = bitcast_u32(position_x)
position_y_bits = bitcast_u32(position_y)
```

This is a bit reinterpretation, not a numeric float-to-integer conversion.
For example, `float32(0.5)` has the bit pattern `0x3F000000`.

## 3. Initial state

The initial state is the Jenkins one-at-a-time hash of three 32-bit words, in
this exact order:

1. the graph random seed;
2. `position_x_bits`;
3. `position_y_bits`.

Only the low 32 bits of the supplied seed participate.

Language-neutral pseudocode:

```text
function initial_state(seed, position_x_bits, position_y_bits):
    h = 0

    for word in [seed & UINT32_MASK, position_x_bits, position_y_bits]:
        h = (h + word) & UINT32_MASK
        h = (h + (h << 10)) & UINT32_MASK
        h = h XOR (h >>> 6)

    h = (h + (h << 3)) & UINT32_MASK
    h = h XOR (h >>> 11)
    h = (h + (h << 15)) & UINT32_MASK

    return h
```

`>>>` denotes an unsigned logical right shift.

## 4. Generating a value

Every random call first advances the pixel's state:

```text
state = (state * LCG_M + LCG_C) & UINT32_MASK
```

The returned unit value uses state bits 1 through 23:

```text
mantissa = (state >>> 1) & 0x007FFFFF
unit_value = mantissa / 8388608.0
```

The result is in the half-open interval `[0, 1)`. Because the denominator is
`2^23`, every possible result is exactly representable as binary32.

An equivalent bit-level conversion, matching the native operation, is:

```text
float_bits = 0x3F800000 OR mantissa
unit_value = bitcast_float32(float_bits) - float32(1.0)
```

The pixel processor's `rand(a)` operation returns the binary32 product:

```text
result = float32(unit_value * float32(a))
```

The argument does not alter the state transition. A zero argument still
consumes one random value.

## 5. Call and state lifetime

- Initialize one state for each pixel evaluation.
- Calls in the same pixel advance that state sequentially.
- Pixel states do not affect one another.
- A new graph evaluation recreates the state from the seed and pixel position.
- The first call uses the first LCG-advanced state, not the raw hash state.

Complete pseudocode:

```text
function create_pixel_rng(x, y, width, height, seed):
    px = float32((x + 0.5) / width)
    py = float32((y + 0.5) / height)

    rng.state = initial_state(
        seed,
        bitcast_u32(px),
        bitcast_u32(py)
    )
    return rng

function rand(rng, amplitude):
    rng.state = (
        rng.state * 0x4937573B + 0x0000C7C1
    ) & 0xFFFFFFFF

    mantissa = (rng.state >>> 1) & 0x007FFFFF
    unit_value = bitcast_float32(0x3F800000 OR mantissa) - 1.0f
    return float32(unit_value * float32(amplitude))
```

## 6. Conformance vectors

The hexadecimal states below are normative. The `u16` column is provided as a
convenient secondary check and is calculated as:

```text
u16 = round_to_nearest(unit_value * 65535)
```

### Pixel `(0, 0)`, size `1 x 1`, seed `0`

```text
position_x_bits = 0x3F000000
position_y_bits = 0x3F000000
initial state   = 0x0A1140EE

call  state       unit value          u16
1     0x002DA09B  0.1782318353652954  11680
2     0x4066787A  0.4002758264541626  26232
3     0x56C601DF  0.7734659910202026  50689
```

### Pixel `(0, 0)`, size `16 x 16`, seed `0`

```text
position_x_bits = 0x3D000000
position_y_bits = 0x3D000000
initial state   = 0x30B00BCA

call  state       unit value          u16
1     0x2EFB254F  0.9810379743576050  64292
2     0xB88939F6  0.5360406637191772  35129
3     0xC42DBD73  0.1786719560623169  11709
```

### Pixel `(1, 0)`, size `16 x 16`, seed `0`

```text
position_x_bits = 0x3DC00000
position_y_bits = 0x3D000000
initial state   = 0x83D861D7

call  state       unit value          u16
1     0x4050654E  0.3140457868576050  20581
2     0x26B7A2BB  0.7173267602920532  47010
3     0x9DCDD5DA  0.8040443658828735  52693
```

### Pixel `(0, 0)`, size `16 x 16`, seed `1`

```text
position_x_bits = 0x3D000000
position_y_bits = 0x3D000000
initial state   = 0xD013D7A6

call  state       unit value          u16
1     0x5D86E503  0.5269318819046021  34532
2     0x74909472  0.5647650957107544  37012
3     0x6643BC07  0.2645877599716187  17340
```

## 7. Common implementation errors

- Using signed right shifts instead of logical right shifts.
- Failing to wrap hash and LCG arithmetic to 32 bits.
- Hashing numeric coordinates instead of their binary32 bit patterns.
- Sampling pixel edges rather than normalized pixel centers.
- Returning a value before advancing the LCG state.
- Using state bits 0 through 22 instead of bits 1 through 23.
- Dividing by `2^23 - 1` instead of `2^23`.
- Treating separate pixels as one shared random stream.
- Skipping state advancement when `rand` has an amplitude of zero.

## 8. FX-Map `rand`

FX-Map function graphs use the float conversion described in section 5, but
they use a different LCG step and do **not** use the Jenkins pixel-position
hash. There is no pixel coordinate in an FX-Map random state.

Each Quadrant parameter has a separate generator lane. Every lane starts from
the node's effective state, but each parameter uses its own LCG pair:

| Parameter | Multiplier | Increment |
|---|---:|---:|
| opacity | `0x4937573B` | `0x0000C7C1` |
| branch offset | `0x5AD7FCD1` | `0x0000C72D` |
| frame offset | `0x3E1BD4C1` | `0x0000C613` |
| pattern size | `0x4458B04D` | `0x0000C4EB` |
| pattern rotation | `0x4A34F3F5` | `0x0000C3D3` |
| pattern supplement | `0x5FBCF1A6` | `0x0000C5D7` |
| dynamic image index | `0x0019660D` | `0x3C6EF35F` |
| Iterate count | `0x5690FB9D` | `0x0000C59F` |
| Switch value | `0x70DCB7DC` | `0x0000C3BF` |

The Switch pair is directly confirmed from generated native code. The image
index and Iterate-count assignments come from their positions immediately
after the six directly mapped Quadrant lanes in the native constant table and
evaluation order.

Given an FX-Map function's starting state, every `rand(a)` call is therefore:

```text
state = (state * PARAMETER_LCG_M + PARAMETER_LCG_C) & 0xFFFFFFFF
mantissa = (state >>> 1) & 0x007FFFFF
unit_value = bitcast_float32(0x3F800000 OR mantissa) - 1.0f
result = float32(unit_value * float32(a))
```

### 8.1 FX-Map root state

The FX-Map atomic node first resolves its own `randomseed` against the
containing graph's `$randomseed`:

```text
no local override: root_incoming = graph_randomseed
absolute override: root_incoming = fxmap_randomseed
relative override: root_incoming = graph_randomseed + fxmap_randomseed
```

All seed arithmetic wraps modulo `2^32`. In an `.sbs` file, the distinction
between absolute and relative overrides is the seed parameter's `relativeTo`
setting.

The root tree node then applies its own inline seed according to the node rules
below. With no FX-Map override and no tree-node seed, the first `rand` starts
directly from `$randomseed`.

For example, a root Quadrant at graph seed zero starts at state zero. Its first
call advances to `0x0000C613` and returns `0.003022313117980957`.

### 8.2 Effective tree-node state

Every tree node has an incoming state and may have an inline `randomseed`.

For a Quadrant:

```text
effective = (incoming + node_randomseed) & 0xFFFFFFFF
```

Quadrants always inherit their incoming state.

Iterate and Switch nodes also have a `randominherited` flag. Its default is
true:

```text
if randominherited:
    effective = (incoming + node_randomseed) & 0xFFFFFFFF
else:
    effective = node_randomseed & 0xFFFFFFFF
```

An absent inline node seed is zero.

In the compiled tree word for Iterate and Switch, bit `0x08` stores
`randominherited`: set means true and clear means false. This bit is independent
of the node kind. Identify Iterate with `low_byte & 0x07 == 0x03` and Switch
with `low_byte & 0x07 == 0x01`.

The effective state is used by that node's dynamic functions and as the basis
for deriving child states. A child's own inline seed is not added until the
child is entered.

### 8.3 Quadrant child states

A Quadrant derives a separate incoming state for each of its four child
slots. It uses one affine transform selected by the **child depth modulo 16**.
The root node is at depth zero.

```text
transform(s, depth) =
    (s * DEPTH_M[depth & 15] + DEPTH_C[depth & 15]) & 0xFFFFFFFF
```

The complete transform table is:

| Depth mod 16 | Multiplier | Increment |
|---:|---:|---:|
| 0  | `0x63DC7A18` | `0x0000CD5B` |
| 1  | `0x4019CF16` | `0x0000CD93` |
| 2  | `0x6454A406` | `0x0000CDC1` |
| 3  | `0x7F13AC69` | `0x0000CDE7` |
| 4  | `0x5CA1EBE2` | `0x0000CE0B` |
| 5  | `0x7527BAB1` | `0x0000CE47` |
| 6  | `0x6AC690C5` | `0x0000CE7B` |
| 7  | `0x5CDCFAA7` | `0x0000CC19` |
| 8  | `0x7549D83B` | `0x0000CC41` |
| 9  | `0x613B9CA8` | `0x0000CC61` |
| 10 | `0x7A3FFD4B` | `0x0000CC9B` |
| 11 | `0x56470DA9` | `0x0000CCE5` |
| 12 | `0x6A364C22` | `0x0000CD1F` |
| 13 | `0x5A76F899` | `0x0000CD3F` |
| 14 | `0x7C3448BE` | `0x0000CBDD` |
| 15 | `0x62AFDC7A` | `0x0000CAD9` |

Starting from the parent Quadrant's effective state, apply the selected
transform once before each slot:

```text
spawn = parent_effective
for slot in [0, 1, 2, 3]:
    spawn = transform(spawn, parent_depth + 1)
    child[slot].incoming = spawn
```

Consequently, slot `i` receives the transform applied `i + 1` times. Empty
slots still occupy their position in this sequence. The compact encoding in
which all four outputs reference one child follows the same rule; each cell
evaluation receives a different incoming state.

Evaluating a child subtree does not modify `spawn` and therefore cannot alter
the states assigned to its siblings.

### 8.4 Iterate child states

An Iterate node has a second affine transition:

```text
ITERATE_M = 0x598232B9
ITERATE_C = 0x000038C5

iterate_step(s) =
    (s * ITERATE_M + ITERATE_C) & 0xFFFFFFFF
```

The optional output 0 is evaluated once using the Iterate node's effective
state unchanged:

```text
output0.incoming = iterate_effective
```

For output 1, keep a private pass state. Advance it before every repeated
pass:

```text
pass_state = iterate_effective
for number in range(count):
    pass_state = iterate_step(pass_state)
    output1.incoming = pass_state
    output1.$number = number
    evaluate(output1)
```

Thus pass zero uses one Iterate step, pass one uses two, and so on. A repeated
child's inline seed is added after deriving that pass's incoming state. It
does not feed back into `pass_state`.

Evaluating output 0, the count function, or an earlier repeated subtree does
not consume or otherwise change the pass-state sequence.

### 8.5 Switch child states

A Switch does not apply an additional affine transform. Its selected child
receives the Switch node's effective state unchanged:

```text
selected_child.incoming = switch_effective
```

At the source graph level, a false `switch` value selects output 0 and a true
value selects output 1.

Evaluating a dynamic Switch value does not advance or replace the state later
passed to the selected child.

### 8.6 Dynamic-function lifetime and call order

There is no single mutable random stream running through an entire FX-Map
subtree.

Each dynamic function evaluation creates a private parameter generator
initialized from the effective state of the node being evaluated. Separate
parameters use the separate LCG lanes in section 8, while calls within one
parameter function advance that parameter's lane in compiled instruction
order. This applies independently to:

- Quadrant opacity;
- branch offset;
- frame offset;
- pattern size;
- rotation;
- pattern supplement;
- a dynamic image index;
- an Iterate count function;
- a Switch value function.

For example, opacity and branch offset both start from the effective node
state, but their first values generally differ because their LCG constants
differ. Random calls made by the opacity function do not advance the
branch-offset function's generator.

The native parameter evaluation order is opacity, branch offset, frame
offset, pattern size, rotation, then pattern supplement. This order does not
couple their random sequences because each parameter lane starts from a fresh
copy of the node state and has its own constants.

Within one compiled function, all `rand` instructions share that function's
private state and execute in function-bytecode instruction order. A
zero-amplitude call still advances it. Function-graph operands are computed
before the instruction that consumes them, so their random calls occur in
their compiled dependency order.

Iterate-count and Switch-value functions follow the same rule: they may use
`rand`, but their private calls do not affect child states.

### 8.7 FX-Map traversal pseudocode

```text
function enter_node(node, incoming, depth, context):
    if node is Iterate or node is Switch:
        base = incoming if node.randominherited else 0
    else:
        base = incoming

    effective = (base + node.randomseed) & 0xFFFFFFFF

    if node is Quadrant:
        evaluate_each_parameter_with_private_rng(effective, context)
        draw_pattern_if_present()

        spawn = effective
        child_depth = depth + 1
        for slot in [0, 1, 2, 3]:
            spawn = transform(spawn, child_depth)
            if node.child[slot] exists:
                enter_node(node.child[slot], spawn, child_depth, context)

    else if node is Iterate:
        count = evaluate_count_with_private_rng(effective, context)

        if node.output0 exists:
            enter_node(node.output0, effective, depth, context)

        pass_state = effective
        for number in range(count):
            pass_state = iterate_step(pass_state)
            pass_context = context with $number = number
            enter_node(node.output1, pass_state, depth, pass_context)

    else if node is Switch:
        value = evaluate_switch_with_private_rng(effective, context)
        selected = node.output1 if value else node.output0
        if selected exists:
            enter_node(selected, effective, depth, context)
```

Iterate and Switch do not increase Quadrant depth. Only descending through a
Quadrant child slot increments it.

## 9. FX-Map conformance vectors

### 9.1 Nested Quadrants

Use an effective FX-Map seed of `11` and this tree:

```text
root Quadrant, seed 3
  slot 2 -> middle Quadrant, seed 5
    slot 1 -> leaf Quadrant, seed 7
```

The node states are:

```text
root effective = 11 + 3
               = 0x0000000E

middle incoming = depth_transform_1^3(root effective)
middle effective = middle incoming + 5
                 = 0x91F05976

leaf incoming = depth_transform_2^2(middle effective)
leaf effective = leaf incoming + 7
               = 0xBEB0F8E6
```

The first three unit-amplitude calls in the leaf are:

```text
call  state       unit value          u16
1     0x45CCE379  0.8003458976745605  52451
2     0x979B784C  0.6073043346405029  39800
3     0x23D9675F  0.8492335081100464  55655
```

### 9.2 Iterate passes

Use an effective FX-Map seed of `11`, an Iterate seed of `3`, and a repeated
child seed of `5`. With inheritance enabled:

```text
iterate effective = 11 + 3 = 0x0000000E

pass 0 incoming = iterate_step(0x0000000E)
pass 0 child effective = pass 0 incoming + 5
                       = 0xE51EFEE8

pass 1 incoming = iterate_step(iterate_step(0x0000000E))
pass 1 child effective = pass 1 incoming + 5
                       = 0xFB74C0D5
```

The first unit-amplitude call in each pass is:

```text
pass  state after rand  unit value           u16
0     0x79EF12FB        0.9338833093643188   61202
1     0x2F2D8AA8        0.17789697647094727  11658
```

Drawing both full-frame square patterns in additive mode produces:

```text
round((0.9338833093643188 + 0.17789697647094727) * 65535)
    = 72861
```

An unsigned 16-bit output clamps this sum to `65535`.

### 9.3 Frame-offset function calls

At effective state zero:

```text
first rand(1)  = 0.003022313117980957
second rand(1) = 0.6204589605331421
```

Two calls inside one frame-offset function return those first and second
values. A separate evaluation of a frame-offset function returns
`0.003022313117980957` again because its lane restarts from state zero.

A literal `rand(0)` is not a reliable way to skip a value in a compiled
function: the compiler may constant-fold the call away, in which case it does
not advance the runtime generator.

### 9.4 Parameter-specific pattern-size lane

At effective state `0x90F1B74A`, two `rand(0.6)` calls in `patternsize`
use `M = 0x4458B04D` and `C = 0x0000C4EB` and return:

```text
0.09322082996368408
0.4085485339164734
```

Using the frame-offset pair for this function is incorrect even though the
same node's random frame offset is otherwise pixel-exact.

## 10. Additional FX-Map implementation errors

- Applying the pixel-position Jenkins hash to FX-Map states.
- Sharing one mutable RNG across parameter functions or the whole subtree.
- Letting a child's random calls affect its siblings.
- Adding a child node's seed before deriving its incoming slot/pass state.
- Feeding a repeated child's seed back into the Iterate pass sequence.
- Treating Quadrant slot variation as one universal transform at every depth.
- Forgetting that the depth-transform table wraps every 16 levels.
- Incrementing depth through Iterate or Switch nodes.
- Starting Iterate pass zero from the unadvanced Iterate state.
- Treating `randominherited=false` as also discarding the node's own seed.
- Letting random calls in Iterate-count or Switch-value functions advance the
  selected child state.

## 11. Ordinary-filter value-function `rand`

This section covers function-driven parameters on ordinary atomic filters,
such as a Blend opacity, Levels bound, Blur intensity, Directional Blur angle,
or Warp intensity. These functions are evaluated once for a node, not once
per output pixel.

### 11.1 Starting state

An ordinary filter resolves its base `randomseed` exactly like an FX-Map
filter:

```text
no local override: initial_state = graph_randomseed
absolute override: initial_state = node_randomseed
relative override: initial_state = graph_randomseed + node_randomseed
```

Seed addition wraps modulo `2^32`. There is no coordinate hash and no mix
based on the filter kind, parameter name, node UID, or graph position.
Changing only a compiled node's UID leaves its random values unchanged.

### 11.2 Generator

All tested ordinary-filter parameter functions use one common LCG pair:

```text
VALUE_LCG_M = 0x7FF8A3ED
VALUE_LCG_C = 0x2AA01D31

state = (state * VALUE_LCG_M + VALUE_LCG_C) & 0xFFFFFFFF
mantissa = (state >>> 1) & 0x007FFFFF
unit_value = bitcast_float32(0x3F800000 OR mantissa) - 1.0f
result = float32(unit_value * float32(amplitude))
```

The float conversion and scale operation are the same as sections 5 and 8.
The common pair was verified directly for a Uniform `outputcolor` function
and a Blend `opacitymult` function.

### 11.3 Lifetime and call order

Each dynamic parameter evaluation starts a new generator from the node's
resolved seed. Two different parameters on the same node do not share state.
For example, random Levels output-low and output-high functions both return
the first value when evaluated on black and white inputs.

Within one parameter function, all `rand` instructions share its generator
and execute in compiled function-bytecode dependency order. Function graph
operands are evaluated before the instruction that consumes them.

The generator is not carried into downstream nodes. A downstream node starts
from its own resolved seed; node identity and evaluation order do not add
variation. Nodes that need different values must have different effective
random seeds.

### 11.4 Conformance vectors

With effective seed zero:

```text
call  state       unit value          u16
1     0x2AA01D31  0.6254453659057617  40989
2     0x29E9568E  0.9114769697189331  59734
3     0xD551A8A7  0.3189796209335327  20904
```

With effective seed five:

```text
call  state       unit value           u16
1     0xAA7B50D2  0.48170197010040283  31568
2     0xFCAFA59B  0.6861206293106079   44965
3     0x42D61EB0  0.8364057540893555   54814
```

An absolute node seed of five produces the second sequence for every graph
seed. A relative node seed of five produces it at graph seed zero and starts
from six at graph seed one.

### 11.5 Common implementation errors

- Applying the pixel-position Jenkins hash.
- Reusing the pixel-processor or any FX-Map parameter LCG pair.
- Hashing the node UID, filter kind, parameter name, or evaluation index.
- Sharing one generator across separate parameters or nodes.
- Ignoring absolute versus relative node-seed resolution.
- Evaluating `rand` once per output pixel instead of once per node parameter.
