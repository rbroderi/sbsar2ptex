# Function Values at the Edge of the Float Range

## Clean-room behavioral specification

This document specifies what Substance computes when a function value
reaches the end of the float32 range: the largest finite float
(`FLT_MAX`, bit pattern 0x7F7FFFFF), infinities, NaN, and integers past
2^24 or 2^31. It also says how a translator must write such a constant
as a GLSL literal. It is intended to be sufficient for an independent
implementation without access to the original implementation. No engine
code and no file data are reproduced; library graphs are cited by name
only. It was recovered from three sources:

- the two corpus graphs that failed with `OverflowError` in
  `function.py` `_flt`, and a third that fails the same way once its
  samples translate (`docs/function-sample.md`);
- the Designer library sources these graphs instance (`.sbs` text), for
  the literal values;
- probe graphs cooked with Designer's `sbscooker` and rendered with
  `sbsrender` to 32-bit float TIFF, on the `vk` and `sse2` engines:
  - 16 cases in one pixel processor;
  - the same 16 cases as 16 Value Processors read back by a pixel
    processor;
  - a second set of 16 integer cases in both.

`docs/function-ops.md` defines the bytecode: blocks, instructions and
op numbers. This document uses its op names.

Each statement is tagged:

- **[measured]**: established by the probes or the corpus.
- **[inferred]**: fits the data, but was not tested directly.

## 1. What fails, and why

**[measured]** In every failing graph the value is a float **constant**
(op 0x00) whose bits are 0x7F7FFFFF (`FLT_MAX`, 3.4028234663852886e38)
or 0xFF7FFFFF (`-FLT_MAX`). No arithmetic op produces it. The library
sources write it as the text `3.402823466e+38`, as a running minimum's or
bounding box's starting value:

| corpus graph | where | the constant |
|---|---|---|
| l_shape_splatter_v2 | a pixel processor's per-pixel function | a float4 `(FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX)` set into a variable, then narrowed by `min` / `max` with sampled values inside a loop: a bounding box |
| l_path_to_spline | a Value Processor's function, read by an FX-Map's iteration-count function through a value slot | a float1 `FLT_MAX` set into a variable: a running minimum |
| l_mask_to_paths_2 | its own copy of the same paths tools function | a float1 `FLT_MAX` |

The Designer packages hold only these two literals in this range: three
in `paths_tools.sbs` and one in `shape_splatter_v2.sbs`.

**The failure.** `_flt` looks for the shortest decimal that reads back
as the same float32. It tries 1 to 9 significant digits and packs each
candidate as a float32 to compare. For `FLT_MAX` the candidates with 1
to 3 digits read back as different finite floats. The 4-digit candidate,
`3.403e+38`, is above `FLT_MAX` plus half a unit in the last place, so
packing it as a float32 raises `OverflowError` instead of returning
false. So this is a converter error about a finite constant, not a
Substance overflow.

## 2. The literal a translator must write

**[measured]** A float constant's value is exactly its 32 bits. A
translation must keep those bits, including at the range's ends:

1. **Finite values, `FLT_MAX` included.** Use the shortest decimal that
   reads back to the same bits. A candidate above the float32 range is
   not a match: skip it and try more digits. Never let it raise. For
   `FLT_MAX` this gives `3.40282347e+38` (9 digits), and for `-FLT_MAX`
   `-3.40282347e+38`. **[inferred]** A GLSL compiler rounds a decimal
   literal to the nearest float32, as C does. Both literals are below
   `FLT_MAX` plus half an ulp, so they compile to 0x7F7FFFFF and
   0xFF7FFFFF.
2. **Infinities** (0x7F800000, 0xFF800000). Write a real infinity, not
   a large finite number, for example `uintBitsToFloat(0x7F800000u)`.
   The current `1e30` changes results: a later `x * 0` gives 0 instead
   of NaN, and a comparison with 1e31 flips.
3. **NaN** (exponent all ones, mantissa non-zero). Write it by its bits,
   `uintBitsToFloat(0x7FC00000u)` or the constant's own bits. Python's
   `nan` text is not GLSL.
4. **Subnormals** (exponent 0). A decimal works: `1e-45` reads back as
   0x00000001. **[inferred]** A GPU may flush a subnormal to zero
   whatever the literal says.

The same rule holds for each component of a vector constant.

**[measured]** Where the constant comes from does not matter: pixel
processor, Value Processor, FX-Map function or any other function block.
The constant's 32 bits are what the engine uses. In the probes,
`FLT_MAX` written out directly read back as 0x7F7FFFFF on both engines.

**The cooker's text parsing** (for writing probes, not for translating).
**[measured]** A Designer literal above the float32 range does not
saturate or become infinity in the cooked file. `3.40282347e+38` and
`3.5e38` both cooked to the constant 0.0. `3.402823466e+38` cooked to
`FLT_MAX`. `1e30` cooked to 0x7149F2CA, its nearest float32.

## 3. What Substance computes past the range

### 3.1 Floats

**[measured]** Float arithmetic is IEEE float32, rounded at every step.
Results past `FLT_MAX` become infinities. They do not saturate at
`FLT_MAX` and do not wrap. With `s` = 2, `one` = 1 and `M` = `FLT_MAX`,
each a live value the cooker cannot fold:

| expression | pixel processor, vk | pixel processor, sse2 | Value Processor, vk and sse2 |
|---|---|---|---|
| `M` | `FLT_MAX` | `FLT_MAX` | `FLT_MAX` |
| `M * s` | +inf | +inf | +inf |
| `-M * s` | -inf | -inf | -inf |
| `M * s * 0` | NaN (0xFFC00000) | NaN | NaN |
| `M + M * one` | +inf | +inf | +inf |
| `(M * s) / s` | +inf | +inf | +inf |
| `M * s - M` | **`FLT_MAX`** | +inf | +inf |
| `M - M * one` | 0 | 0 | 0 |
| `min(M, 5)` | 5 | 5 | 5 |
| `floor(M * one)` | `FLT_MAX` | `FLT_MAX` | `FLT_MAX` |
| `1 / (M * s)` | 0 | 0 | 0 |
| `M * s > 1` | true | true | true |

**The one disagreement.** The vk pixel processor gave `FLT_MAX` for
`M * s - M`, which is the fused multiply-add `fma(M, s, -M)` with no
rounding of the product. sse2 and both engines' Value Processors rounded
the product to +inf first. So the GPU compiler may contract `a * b + c`.
A translation that rounds every step matches sse2 and the Value
Processors. On vk it matches except where a contracted product would
overflow or lose bits, as here.

**For the corpus graphs.** `FLT_MAX` is only a starting value, consumed
by `min`, `max` and comparisons. These are exact, and no step overflows
unless nothing narrows the box or the minimum. In that case:

- a later `max - min` of the untouched box gives +inf, from
  `FLT_MAX - (-FLT_MAX)`;
- a product with 0 gives NaN.

A translation that writes the literal exactly and computes in float32
matches.

### 3.2 Integers

**[measured]** In pixel processors and Value Processors (vk and sse2),
integer values behave as float32 values whose float-to-int `convert` is
`floor`. They are not 32-bit two's complement. Neither saturation nor
wrapping happens:

| expression (each `int(...)` is a `convert` to integer, read back through a `convert` to float) | result, all four evaluators |
|---|---|
| `int(2.7 * one)` | 2 |
| `int(-2.7 * one)` | **-3** (floor, not truncation) |
| `int(3e9 * one)` | 3e9 (no saturation at 2^31 - 1, no wrap) |
| `int(-3e9 * one)` | -3e9 |
| `int(1e10 * one)` | 1e10 |
| `int(2^31 * one)` | 2147483648 |
| `int(M * one)` | `FLT_MAX` |
| `int(M * s)` | +inf |
| `int(M * s * 0)` | NaN |
| `int(-M * one)` | `-FLT_MAX` |
| `int(16777216 * one)` | 16777216 |
| integer constant 2147483647 `+ int(one)` | 2147483648 (no wrap to -2^31) |
| integer constant 16777217 `+ int(one)` | 16777216 (the constant and the sum both round to float32) |
| integer constant -2147483648 `+ int(-1 * one)` | -2147483648 (float32 rounding of -2147483649) |
| `int(3e9 * one) + int(one)` | 3e9 |
| `int(M * one) + int(one)` | `FLT_MAX` |

**This contradicts `docs/function-ops.md`** on two points, and corrects
it for these evaluators:

- Its section 1 says the float-to-int `convert` "truncates toward
  zero". It is `floor`: -2.7 becomes -3.
- Its section 1 also says "integer values are 32-bit two's complement".
  At run time they are float32 values.

What its section 2 measured still holds: the cooker's constant folding
wraps at 32 bits. It folded 2147483647 + 1 to -2147483648 (its S4), while
the same sum computed at run time gives +2147483648. **[inferred]** A
translation must not fold integer constants with 32-bit wrapping where
the engine would compute them at run time. Below 2^24, and for
non-negative operands of `convert`, the two models agree. Every integer
in the corpus failures is in that range.

**[inferred]** For GLSL: an integer value of a pixel processor is
modelled exactly as a `float` holding `floor` of the value. A GLSL `int`
is exact only while every intermediate stays within ±2^24 and every
float-to-int conversion is of a non-negative value or written as
`int(floor(x))`. The current translation's `int(x)` truncates in GLSL,
so it is wrong for negative non-integers.

## 4. Acceptance tests

**Literals** (unit tests of the float formatter):

| # | bits | expected | must not |
|---|---|---|---|
| L1 | 0x7F7FFFFF | a decimal reading back to 0x7F7FFFFF, e.g. `3.40282347e+38` | raise `OverflowError` |
| L2 | 0xFF7FFFFF | `-3.40282347e+38` or equivalent | raise |
| L3 | 0x7F7FFFFE | reads back to 0x7F7FFFFE (`3.40282326e+38`) | raise |
| L4 | 0x7F800000 | +infinity (by bits) | be `1e30` |
| L5 | 0xFF800000 | -infinity | be `-1e30` |
| L6 | 0x7FC00000 | a NaN | be the text `nan` |
| L7 | 0x00000001 | reads back to 0x00000001 | |
| L8 | 0x3F800000 | `1.0` (unchanged behaviour) | |

L1 to L3 cover every finite float whose shorter decimal candidates
overflow. A check over all finite floats with exponent 0xFE, of the
form "format, parse, compare bits", must pass without raising.

**Corpus:**

- **C1.** l_shape_splatter_v2 converts past its pixel processor. With
  the uid-table reader fix (`docs/sbsasm-container.md`) and this
  formatter, it writes a `.ptex`. This was checked with a study patch.
- **C2.** l_path_to_spline and l_mask_to_paths_2 no longer fail in
  `_flt`. They then stop at their Value Processor samples, which are
  specified in `docs/function-sample.md`. With both fixes they should
  convert.
- **C3.** The GLSL of C1 contains the bounding box's starting value as
  the literal of L1 and L2. A render of the translated pixel processor
  matches the truth wherever the box is narrowed.

**Run-time semantics** (render tests; 32-bit float output, s = 2 and
one = 1 live as in section 3):

- **R1.** `M * s` gives +inf, `-M * s` -inf, `M * s * 0` NaN, `1 / (M * s)` 0.
- **R2.** `int(-2.7 * one)` gives -3. `int(3e9 * one)` gives 3e9.
  Integer constant 2147483647 `+ int(one)` gives 2147483648.
- **R3.** A translation computing `M * s - M` may give +inf (stepwise)
  or `FLT_MAX` (contracted). Either is accepted; the truth is
  engine-dependent (section 3.1).

## 5. Unknown

- **FX-Map functions.** The run-time integer and overflow behaviour of
  an FX-Map's own functions (quadrant parameters, iteration counts) was
  not probed. They run on the CPU, so section 3 is **[inferred]** for
  them. In particular, how an iteration count of +inf, NaN or a value
  above 2^31 is taken is unknown.
- **Integer ops other than add and convert.** Integer division, modulo,
  and bitwise-like uses were not probed under the float model. The
  question is whether integer `div` floors, truncates or gives a
  fraction.
- **Which engines contract.** Whether a given GPU and driver fuses
  `a * b + c` in a pixel processor. Only one vk machine was measured.
- **Subnormals** on the GPU: flushed to zero or kept. Not probed.
- **An infinite or NaN constant in a cooked file.** Not seen. Designer's
  text field does not produce one; overflowing text cooks to 0.0.
- **NaN payloads and signs** past `x * 0`: the probe saw 0xFFC00000
  only.
