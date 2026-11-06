# Legacy Motion Blur (sbsasm ops 18 and 19)

## Clean-room behavioral specification

This document specifies the compiled legacy **Motion Blur** filter
(`.sbs` filter `motionblur`, Designer label "Motion blur"): op 18
(grayscale) and op 19 (colour). It gives the node's word layout and the
per-pixel result, exactly enough to reproduce `sbsrender` at 16 bits. It
is intended to be sufficient for an independent implementation without
access to the original implementation. No engine code and no file data
are reproduced.

It was recovered from:

- Probe graphs cooked with Designer's `sbscooker` (cook version 2): one
  Bitmap for `input1`, optionally one for `inputgradient`, and the motion
  blur with its parameters absent, constant or a function, and with the
  base parameters tiling, pixel size, output size and random seed set.
- `sbsrender` renders (vk engine, 16 bits) of those probes, at 256 × 256,
  512 × 512, 256 × 128 and 128 × 256, compared pixel by pixel with a
  float32 model of the rule below.
- The engine's decompiled per-pixel shader for this filter (one of the
  SPIR-V modules in `substance_vk_blend.dll`), read for the structure of
  the computation only.
- The node corpus graphs `a_motionblur_{g,c}_default_{coded,noise}`.

Each statement is tagged:

- **[measured]**: established by cooking or rendering probes.
- **[inferred]**: fits the data, but was not tested directly.

`docs/node-seed.md` section 0 defines words, offsets, `rest`, code blocks
and the node word (`flags << 16 | x << 8 | op`). `docs/node-params.md`
defines parameter states, `P` and `F`, and the pixel-size vector. This
document uses their terms.

## 1. What the filter does, in one paragraph

The filter has two image inputs: `input1`, the image to blur, and
`inputgradient`, a grayscale height map. At each pixel it takes the
backward differences of the height map along x and y, rotates that
vector by the **shift angle**, scales it by the **intensity**, and averages
six bilinear samples of `input1` spread along the resulting line,
symmetrically about the pixel. Where the height map is flat (or not
connected) the output is the input.

## 2. Node layout

### 2.1 The node word

**[measured]** `flags << 16 | x << 8 | op`, op 18 (grayscale) or 19
(colour). In `flags`:

- bit 0: the node has an output-size function (`docs/node-seed.md`);
- bits 6-7: the seed state (`docs/node-seed.md` section 1);
- `P = flags >> 8`, laid out as for the neighborhood filters
  (`docs/node-params.md` 1.1):

  | bits of P | meaning | words |
  |---|---|---|
  | 0-1 | tiling (0 none, 1 horizontal, 2 vertical, 3 both); a value, not a state | none |
  | 2-3 | state of the pixel-size vector (float2) | state 1: two f32; state 2: one offset |

  No other bit of `P` was set in any probe.

The byte `x` changes with the node's output format (0x18 at the default,
0x28 with format 2), as for other filters; it does not affect the layout.

### 2.2 The fields

**[measured]** `rest[0..2]` are, in order:

| `rest` | field |
|---|---|
| 0 | **F**, the flag word holding the op's parameter states |
| 1 | `input1`: a node index |
| 2 | `inputgradient`: a node index |

`F`'s layout:

| bits of F | parameter | width | default (state 0) |
|---|---|---|---|
| 0-1 | intensity | 1 | 1.0 **[measured]** |
| 2-3 | shift angle, in turns | 1 | 0.0 **[measured]** |

Probes: intensity 4 gives F = 1; shift angle 0.1 gives F = 4; both give
F = 5; intensity 2 with an angle function gives F = 9. An intensity of
exactly 1 is cooked as state 0 (F = 0, no word), and renders as 1.

When `inputgradient` is not connected, the cooker wires it to a node of
its own that produces a constant image (in the corpus graphs, a
transformation of a uniform). So the field is always a node index, and an
unconnected gradient behaves as a flat one: the output equals the input
(section 3.4).

### 2.3 The word order

**[measured]** The general rule of `docs/node-seed.md` section 2.1 and
`docs/node-params.md` section 2 holds:

1. the fields (F, input, gradient);
2. the seed slot (seed state 1 or 2);
3. the output-size function's offset (flags bit 0);
4. the pixel-size vector: nothing, two f32, or one offset;
5. intensity: nothing, one f32, or one offset;
6. shift angle: nothing, one f32, or one offset;
7. the code blocks.

Measured shapes (word values after the node word):

| probe | flags | rest |
|---|---|---|
| intensity 4 | 0x0318 | 1, in, grad, 4.0 |
| intensity 4, angle 0.1 | 0x0318 | 5, in, grad, 4.0, 0.1 |
| angle 0.1 | 0x0318 | 4, in, grad, 0.1 |
| intensity 4, pixel size (2, 1) | 0x0718 | 1, in, grad, 2.0, 1.0, 4.0 |
| intensity 4, relative output size | 0x0B19 | 1, in, grad, → size, → pixel size, 4.0 |
| intensity `1 + rand(4)` (seed from the graph) | 0x0398 | 2, in, grad, → seed block, → intensity |
| the same, random seed 5 | 0x0358 | 2, in, grad, 5, → intensity |
| intensity 2, angle `0.25 rand(0.5)` | 0x0398 | 9, in, grad, → seed block, 2.0, → angle |
| tiling 0 / tiling 1 | 0x0018 / 0x0118 | as intensity 4 |
| 256 × 128, intensity 1 | 0x0718 | 0, in, grad, 0.5, 1.0 |

A random seed with no parameter function drawing `rand` is dropped by
the cooker (seed state 0). A parameter function the cooker can fold is
stored as a constant (intensity `2 × 1.5` became the f32 3.0, state 1).

Read the shift angle and intensity as numbers in **turns** and **plain
units**. They are parameters like any other: a state-2 parameter is
evaluated once per node, before the pass, with the node's seed
(`docs/node-seed.md`, `docs/random-generator.md`).

## 3. Per-pixel semantics

### 3.1 Notation

- The output, `input1` and `inputgradient` have the same size, `Wd × Ht`
  texels. **[measured]** for equal sizes; see Unknown for unequal.
- A pixel `(i, j)` (column `i`, row `j`, row 0 the first row of the
  image as stored) has UV `p = ((i + 0.5) / Wd, (j + 0.5) / Ht)`. UV y
  grows with the row index.
- `h(q)` is the gradient's value (0 to 1) at texel `q`.
- `k` is the intensity, `a` the shift angle in turns, `v = (v.x, v.y)`
  the pixel-size vector: the inline pair (state 1), its block's value
  (state 2), or `(min(Wd, Ht) / Wd, min(Wd, Ht) / Ht)` in state 0
  **[measured at 512 × 512, where it is (1, 1); inferred otherwise:** the
  cooker stored that default as an inline pair at 256 × 128**]**.
- All arithmetic is float32 unless stated.

### 3.2 The formula

**[measured]** Per pixel:

1. **Slope.** The backward differences of the gradient, one texel along
   each axis:

       dx = h(i, j) - h(i - 1, j)
       dy = h(i, j) - h(i, j - 1)

   The gradient is read at texel centres, so these are exact texel
   values. The neighbour index is wrapped or clamped by the tiling
   (section 3.3). The step is always one texel; the pixel-size vector
   does not change it.

2. **Rotation** by the shift angle, with `c = cos(2π a)`, `s = sin(2π a)`
   (float32; see Unknown for their last bits):

       r.x = dx c - dy s
       r.y = dx s + dy c

   A positive angle turns +x toward +y (toward higher rows).

3. **Scale.** The line's half-step in UV:

       D.x = r.x × (m.x × K)
       D.y = r.y × (m.y × K)

   with `K = 0.335` (the float32 0.33500000834465027) and the per-axis
   factor

       m.x = k × v.x × (Wd / 256)
       m.y = k × v.y × (Ht / 256)

   Measured: at 256 × 256, `m = (k, k)`; pixel size (2, 1) gives `m =
   (2k, k)`; at 512 × 512, `m = (2k, 2k)`; at 256 × 128 and 128 × 256
   (default vector), `m = (k / 2, k / 2)`. So with the default vector
   the factor on both axes is `k × min(Wd, Ht) / 256`. In texels, the
   line's half-step along x is `D.x × Wd`.

   The product is formed in that order: `m × K` first (one float32
   rounding), then times `r`. This matters only at offsets of tens of
   texels and more.

4. **Taps.** Six bilinear samples of `input1` at

       q_t = p + t × D,   t ∈ {+0.5, -0.5, +1.5, -1.5, +2.5, -2.5}

   (each `p + t × D` rounded to float32 once per component), so the taps
   sit at `±1/6, ±1/2, ±5/6` of the line's full length `6 |D|`.

5. **Average.**

       out = W × (sum of the six samples),   W = 0.166666 (float32
       0.16666600108146667)

   `W` is not exactly 1/6: six equal samples of 1.0 give 0.999996, which
   rounds to 65535 at 16 bits. An implementation should use this
   constant, not `1.0 / 6.0`, to stay exact.

All four channels of a colour input are blurred alike, alpha included
**[measured for RGB; alpha inferred:** the probe's alpha was constant**]**.
The grayscale op blurs its one channel. The result is not clamped by the
filter; it is stored in the output format as usual (16 bits: rounded to
the nearest of 65535 levels).

### 3.3 Sampling, filtering and tiling

**[measured]**

- **Filtering.** The six taps are hardware bilinear samples with texel
  centres at integers + 0.5 in texel space: texel coordinate `u × Wd -
  0.5`. The sub-texel fraction is quantized to **1/256**: the weight of
  the right-hand texel is `round(256 × frac) / 256`. Modelling exact
  (unquantized) bilinear weights misses by up to 79 levels on 97% of
  pixels; truncating instead of rounding misses by up to 158 levels on
  99%. Rounding matches all but about 0.1% of pixels, which differ by at
  most 1 level, plus, on noise gradients at some intensities, a few
  dozen pixels per image (65 of 65,536 at intensity 0.3, none at 1 and
  4) where one tap's fraction falls at a quantization step's midpoint:
  those land on the other step and miss by up to 56 levels. The
  engine's exact tie rule was not recovered (Unknown). A GLSL
  translation that samples with `texture()` and linear filtering gets
  the same 8-bit fraction on common GPUs.
- **Tiling.** Per axis, with tiling on along that axis (P bits 0-1: 1
  horizontal, 2 vertical, 3 both), the input taps and the gradient's
  backward difference wrap around the image. With tiling off along an
  axis, both clamp to the edge texel: the tap is clamped, and at row 0
  (column 0) the "previous" gradient texel is the edge texel itself, so
  `dy` (`dx`) there is 0. Tiling 0 and tiling 1 were rendered: clamp on
  both axes, and wrap along x with clamp along y, each within 1 level.
- **No mip level.** Taps far outside the image (hundreds of texels) are
  still single bilinear samples, wrapped or clamped.

### 3.4 Special cases

- **Flat or unconnected gradient** (`dx = dy = 0`): all six taps sample
  the pixel itself; the output is `W × 6 × input`, which is the input at
  16 bits (corpus graphs `a_motionblur_*_default_*`, 0 levels).
- **Intensity sign.** `-k` gives `-D`; the tap set is symmetric, so the
  result is the same as `k`. **[inferred]**
- **Shift angle** is periodic with period 1. **[inferred]**

### 3.5 GLSL shape for a translation

Not engine code; a direct transcription of 3.2 for a Material Maker
shader, where `$in` and `$grad` are the two inputs, `size` the texture
size in texels, and `k`, `a`, `v` the parameters:

    vec2 e  = 1.0 / size;
    float h0 = $grad(uv).r;
    vec2 d  = vec2(h0 - $grad(uv - vec2(e.x, 0.0)).r,
                   h0 - $grad(uv - vec2(0.0, e.y)).r);
    float c = cos(6.28318530718 * a), s = sin(6.28318530718 * a);
    vec2 r  = vec2(d.x * c - d.y * s, d.x * s + d.y * c);
    vec2 D  = r * (k * v * size / 256.0 * 0.335);
    vec4 acc = vec4(0.0);
    for (int n = 1; n <= 3; n++) {
        float t = float(n) - 0.5;
        acc += $in(uv + D * t) + $in(uv - D * t);
    }
    out = acc * 0.166666;

The gradient reads must hit texel centres (`uv` is a pixel centre), and
the input reads need the tiling's wrap or clamp.

## 4. Acceptance tests

Renders are sbsrender 16-bit, 256 × 256, cook version 2. "Level" is one
step of 65535. Patterns: `impulse` is black with one texel of 1.0 at
column 128, row 128; `ramp_x` has texel value `(i + 0.5) / 256` in column
`i` (stored at 16 bits, so `dx` is 255 or 256 levels, and -65279 at
column 0, where it wraps); `ramp_y` the same along rows; `noise` and
`noise2` are independent uniform random 16-bit images.

Layout tests (synthetic words; "→ X" an offset to a well-formed block):

1. **Constant intensity.** Op 18, `w0 = 0x03180012`; 1, 7, 8, 4.0.
   Input 7, gradient 8, intensity 4, angle 0, pixel size state 0.
2. **Both constant.** `w0 = 0x03180012`; 5, 7, 8, 4.0, 0.1. Intensity 4,
   angle 0.1.
3. **Angle only.** `w0 = 0x03180012`; 4, 7, 8, 0.1. Intensity 1 (default).
4. **Inline pixel size.** `w0 = 0x07180012`; 1, 7, 8, 2.0, 1.0, 4.0.
   Pixel size (2, 1), intensity 4.
5. **Size function.** `w0 = 0x0B190012`; 1, 7, 8, → size, → float2,
   4.0. Pixel size: the function.
6. **Seed block and an intensity function.** `w0 = 0x03980012`; 2, 7, 8,
   → seed block, → float1.
7. **Inline seed.** `w0 = 0x03580012`; 2, 7, 8, 5, → float1. Seed 5.
8. **Mixed.** `w0 = 0x03980012`; 9, 7, 8, → seed block, 2.0, → float1.
   Intensity 2; angle: the function.
9. **Colour.** Tests 1-8 with op 19 read the same.

Render tests (expected values are the renders; a translation must match
within the stated levels):

- **R1. Impulse, intensity 4, `ramp_x`.** Row 128 only; columns 124 to
  132 hold 3840, 7211, 10794, 7339, 7253, 7339, 10794, 7211, 3840; every
  other pixel 0. (Taps at ±0.67, ±2.01, ±3.35 texels; fractions to
  1/256.) 0 levels.
- **R2. Impulse, intensity 40, `ramp_x`.** Row 128 only; non-zero at
  columns (relative to 128) ±6: 3285, ±7: 7637, ±20: 9813, ±21: 1109,
  ±33: 5461, ±34: 5461. Nothing at column 128 itself. 0 levels.
- **R3. As R2 with shift angle 0.25.** The same pattern along column 128
  (rows instead of columns; `cos(π/2)` is not exactly 0 in float32, so
  the levels differ slightly: ±6: 3541, ±7: 7381, ±20: 10666, ±21: 256,
  ±33: 6869, ±34: 4053). 0 levels.
- **R4. Impulse, intensity 4, `ramp_y`.** R1 transposed. 0 levels.
- **R5. Two impulses (column 64, row 64 and column 192, row 128),
  `ramp_x`, intensity 10, angle 0.125.** Each spreads along the down-right diagonal (rows and
  columns increasing together). 0 levels. Rotating the other way fails
  by thousands of levels.
- **R6. Noise input, `noise2` gradient, intensity 1, 4 and 0.3, angle 0.**
  Intensity 1 and 4: at most 1 level. Intensity 0.3: at most 1 level on
  99.9% of pixels (quantization ties, section 3.3).
- **R7. Noise, `ramp_x`, intensity 20, angle 0.1.** At most 1 level.
- **R8. Pixel size (2, 1), noise / noise2, intensity 4.** Equal to the
  model with `m = (8, 4)`; at most 1 level.
- **R9. Size.** Noise / noise2 at 512 × 512, intensity 1: `m = (2, 2)`. At
  256 × 128 and 128 × 256 with the default vector: `m = (0.5, 0.5)`. At
  most 1 level.
- **R10. Tiling.** Noise / noise2, intensity 4: tiling 0 is the clamped
  model, tiling 1 wraps x and clamps y. At most 1 level. Wrapping instead
  misses by up to 37,000 levels.
- **R11. Colour.** Op 19, an 8-bit RGBA noise input, `noise2` gradient,
  intensity 2, angle 0.125: each channel is the grayscale model of that
  channel, at most 1 level.
- **R12. Corpus.** `a_motionblur_g_default_coded`, `_g_default_noise`,
  `_c_default_coded`, `_c_default_noise` (gradient not connected): the
  output equals the input, 0 levels.

## 5. Unknown

- **The last bits of `cos` and `sin`.** At intensity 4 on a noise
  gradient the taps reach hundreds of texels, and one float32 ulp in `c`
  or `s` moves most pixels. The engine's values differ from correctly
  rounded `cos(2π a)` and `sin(2π a)` by a few ulps (absolute error up
  to about 2.4e-7). Fitted against renders, in ulps of (c, s): a = 0.05:
  (-1, 0); 0.1: (+1, 0); 0.2: (-7, -1); 0.3: (+7, -1); 0.37: (+1, 0);
  0.6: (-3, +4); 0.9: (+1, +1). With those values the renders match
  within 1 level; with correctly rounded ones, 2% to 4% of pixels miss
  by up to 130 levels. None of these reproduce the fitted values: float32
  or double argument `2π a`, the function evaluator's polynomial
  (`docs/function-ops.md`), or a common minimax `sincos`. For gradients
  of ordinary slope (R7: a full-range ramp at intensity 20) correctly
  rounded values are exact. An implementation can use plain `cos` and
  `sin` and report the residual as approximate for extreme offsets.
- **Quantization ties.** Which way the bilinear fraction rounds when it
  falls (nearly) on a 1/512 midpoint (section 3.3). Neither round-half-
  even in double nor round-half-up on a float32 texel coordinate matches
  all such pixels.
- **A colour image on `inputgradient`.** The shader reads the gradient
  as a weighted sum of its channels; the weights for a colour source
  were not probed. Designer types this input grayscale.
- **Inputs of different sizes** (gradient or input not the output's
  size): not probed. **[inferred]** The backward difference steps one
  texel of the output.
- **Varying alpha** on a colour input: the probe's alpha was constant.
- **Non-square sizes with an explicit Pixel Size**: whether the cooker
  multiplies the default vector into an inline value. The rule in 3.2
  uses the stored vector, whatever it is.
- **Seed state 3** and **state 3** of any parameter: never seen.
