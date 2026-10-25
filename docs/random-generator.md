# Substance Pixel-Processor Random Generator

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
