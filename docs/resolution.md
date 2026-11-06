# Substance Output Resolution

## Clean-room behavioral specification

This document specifies how Substance 3D Designer's engine, as run by
`sbsrender`, changes what a node computes when the graph's output size
(`$outputsize`) changes. It is meant to tell sbsar2ptex which parameters
must be scaled by the image size, which are fixed in pixels, and where an
effect disappears below some size. It is intended to be sufficient for
an independent implementation without access to the original
implementation.

It was recovered from three sources. No engine code is reproduced here.

- Single-filter graphs cooked with `sbscooker` (sbsar2ptex.probe), fed
  generated inputs and rendered at 64 to 2048 pixels.
- Renders of BiomechZombieCircuits' nodes at 256 to 2048 pixels.
- The Vulkan engine's SPIR-V shaders, read for their formulas only. The
  engine is `substance_vk_blend.dll`, which embeds 70 modules.

Each statement is tagged:

- **[measured]**: established by renders.
- **[shader]**: read from the engine's shaders.
- **[inferred]**: fits the measurements, but was not tested directly.

`N` is the node's output width in pixels. "uv" is image space, [0, 1]
across the image. "Px@256" is a length given in pixels of a 256-pixel
image: `L` px@256 is `L N / 256` pixels at size `N`, or `L / 256` uv.

Graph inputs were 16-bit PNGs at the render size, so nothing was
resampled unless a test says so. Renders were 16-bit, on sbsrender's
default engine (d3d11). The vk engine's shaders agree with d3d11's renders
to within one level (docs/fxmap-splats.md, section 5).

## 1. The rule: lengths are image-relative, in px@256

**[measured]** In every filter tested, a length parameter is in px@256.
The cooker stores the value as the user typed it: blur 0.37 cooks to
0.37 and distance 7.68 to 7.68.

**[shader]** No shader scales by `N` itself. Each takes its offsets and
scales as uniforms, computed on the CPU: a texel size `1/N`, or a vector
already in uv. Nothing is clamped to a minimum of one pixel. So each
filter covers the same part of the image at every size, and at small
sizes it loses what falls below one pixel.

| filter | parameter | at size N | below one pixel |
|---|---|---|---|
| blur | intensity `I` | box of half-width `R = I N / 256` px, on both axes | `R <= 0.5`: identity |
| directional motion blur | intensity `I`, angle | box of `R` steps along the angle (1.1) | `R <= 0.5`: identity |
| warp | intensity `I` | moved `I / 256` uv per unit of gradient slope per uv | bilinear sub-pixel move |
| directional warp | intensity `I` | moved `I / 256` uv per unit of the intensity map | bilinear sub-pixel move |
| distance | distance `D` | `1 - d / (D N / 256)`, `d` in pixels | reach shrinks to the mask itself |
| emboss, normal | intensity `I` | `I / 256` per unit of slope per uv | none seen |
| sharpen | intensity `k` | fixed 1-pixel 4-neighbour kernel, not scaled | none: it is in pixels |
| transformation | matrix, offset | in uv; mip level from the matrix only (2) | sub-texel offsets interpolate |

### 1.1 Blur and directional motion blur

**[measured]** Each output pixel is the input averaged over a box of
half-width `R` pixels. The pixel at either end of the box is weighted by
how much of it lies inside. Tested with a one-pixel line, blurred across
the line:

- Intensity 1 gives `R = 1` px at 256: taps (1/4, 1/2, 1/4). At 2048 it
  gives `R = 8`: 15 taps of 1/16, with 1/32 at each end.
- Intensity 0.75 at 256 gives `R = 0.75`: (1/6, 2/3, 1/6).
- Intensity 0.37 at 512 gives `R = 0.74`: centre 1/1.48 = 0.6757.
- At or below `R = 0.5` the box lies inside the centre pixel, so the
  image is returned unchanged. That held for intensity 0.37 at 256 and
  below, for 0.5 at 256, for 1 at 128 and for 0.1 at 1024.
- Blur at intensity `I` and size `2N` equals blur at `2I` and size `N`,
  tap for tap. So the box is fixed in uv: `R / N = I / 256`.
- The directional motion blur at angle 0 gives exactly the same taps as
  blur at the same intensity.

**Off-axis angles.** The angle `a` is in turns. The major axis is the
one the direction moves along more.

- A step is one texel along the major axis and `t` texels across it,
  where `t` is the tangent of the angle measured from the major axis.
- The box is `R = 0.5 + (I N / 256 - 0.5) max(|cos 2 pi a|, |sin 2 pi a|)`
  steps long on each side.
- It takes `n = ceil(R - 0.5)` taps on each side of the centre. The two
  end taps are weighted `R + 0.5 - n`, the others 1, and the sum is
  divided by `2R`.
- Across a step, a tap is read linearly between the two texels it falls
  between.
- In pixels along the direction, the box is about `2 I N / 256` long at
  any angle. Its short-box correction depends on the angle.

**[measured]** Checks of this rule, with a one-pixel dot:

- At 45 degrees, intensity 1, size 256: `R = 0.5 + 0.5 cos 45 = 0.854`.
  The dot becomes three diagonal taps, 0.2071, 0.5858 and 0.2071. Those
  are `0.354 / 1.707`, `1 / 1.707` and `0.354 / 1.707`, as the rule
  gives.
- At 22.5 degrees, intensity 3, size 256: `R = 2.81`, so `n = 3`. The
  footprint spans x from -3 to +3. It sums to 1 and is centred.

**[shader]** Short and long boxes take different paths:

- **Up to `R = 1.5`**: one shader reads three taps, the centre and one
  step either side. The step goes through a 2x3 matrix, and the centre
  and side weights are uniforms.
- **Longer boxes**: the image is sheared into a frame whose columns are
  the blur's lines, then boxed along the columns, then read back.
  - Boxing builds each sum from blocks of `2^k` texels. They are read at
    level 0 from a reduced copy, with partial blocks at the two ends.
  - Reading back mixes the two frame columns either side of each pixel,
    with linear weights.
  - All the shaders' offsets are in texels of the node: uniforms set on
    the CPU from `N`.
  - At 45 degrees and size 1024 (`R = 2.975`), this shows as diagonal
    runs that are three texels wide (0.042, 0.084, 0.042). That is the
    linear read across frame columns.

### 1.2 Warp and directional warp

**[measured]** A one-pixel line was warped by a horizontal ramp (slope 1
per uv), or moved by a white intensity map at angle 0. The move was
`I / 256` uv at every size from 64 to 2048: -0.00391 uv for intensity 1,
-0.0156 uv for intensity 4. A move of part of a pixel is a bilinear
blend: intensity 0.1 at 2048 moves 0.8 px, and gives the line's two
pixels 0.8 and 0.2.

**[shader]**

- **Warp** reads the gradient image at the pixel and one texel back in x
  and in y. The texel size is a uniform. It moves the sample by those two
  differences times a per-axis scale uniform, which must be `I N / 256`
  for the measured move.
- **Directional warp** has no texel uniform. It moves the sample by
  `(map value - bias)` times a uv vector. The CPU sets that vector to
  `(I / 256)` times the angle's direction, so the move is independent of
  `N`.

### 1.3 Distance

**[measured]** For a one-pixel dot, the value is `1 - d / (D N / 256)`,
with `d` the distance in pixels to the nearest mask pixel:

- `D = 16`, 4 px away: 0.7499 at 256 (1 - 4/16), 0.9686 at 2048 (1 - 4/128).
- `D = 7.68`, 4 px away: 0.4789 at 256.
- Below one pixel only the mask is left: `D = 1` reaches 0 px at 64 and
  128, and half a pixel at 256.

**[shader]** The nearest mask pixel is found by jump-flooding passes. Each
pass stores a seed position per pixel. Seed distances are measured in
pixels:

- The wrapped uv difference is multiplied by a size uniform.
- The metric is chosen by a uniform: Euclidean, Manhattan (sum) or
  Chebyshev (max).

### 1.4 Emboss and normal

**[measured]** For a flat 0.5 input lit by a ramp gradient, the output's
maximum was `0.5 + I / 256` at every size: 0.5039 for intensity 1, 0.5156
for intensity 4.

**[shader]** Emboss and normal both take backward differences over one
texel (a texel-size uniform). They scale the differences by a per-axis
uniform:

- Emboss turns the differences, through that uniform, into one slope,
  and adds `slope * highlight` or `slope * shadow` by its sign.
- Normal builds `normalize(-dx k, -dy k, 1) / 2 + 1/2`.

The measured slope is per uv, so the scale uniform is `I N / 256`.

### 1.5 Sharpen

**[measured]** For a 0.75 line on 0.5 gray, each of the line's
neighbours moved by `-k (0.75 - 0.5)`, the same at every size from 64 to
2048. Only the line's immediate neighbours changed.

**[shader]** Sharpen reads the four neighbours one texel away (a texel
uniform). It returns `x (1 + 2 kx + 2 ky) - kx (left + right) - ky (up +
down)`, with `kx = ky = k`, not scaled by `N`. At 2048 it sharpens detail
eight times finer, in uv, than at 256.

## 2. Transformation

**[shader]** The transformation maps the output pixel's uv through a 2x3
matrix uniform, composed on the CPU, and reads the input at that point.
**[measured]** The composition scales and rotates about the image centre
(`p = M (uv - 1/2) + 1/2`, then the offset). Every scale in 2.1 matched
exactly that way.

- **Filtering 1 (nearest)**: the point is first snapped to the centre of
  the input texel it falls in, using the input's size and its inverse
  (uniforms). Then it is read.
- **Filtering 0 (bilinear)**: it is read with the GPU's bilinear filter.
- **Mipmaps**: both modes read with an implicit-level texture fetch. No
  level or bias is passed. **[inferred]** So the GPU picks the level from
  how fast the point moves per output pixel (2.1). Nearest filtering
  still reads that level.
- **No tiling**: outside [0, 1] on an axis that does not tile, the value
  fades to a background colour uniform. The fade is the clamped distance
  past the edge times a scale uniform, so a band about one texel wide
  (not measured).

**[measured]** In uv at every size:

- An offset of 1/256 uv moves a line 1 px at 256 and 4 px at 1024.
- Half a texel with bilinear filtering gives (0.5, 0.5).
- Half a texel with nearest filtering moves it a whole texel: a point
  exactly on a texel edge reads the texel above it.

### 2.1 The mip level

**[measured]** Inputs were noise at the node's size. Each output was
compared with every box-filtered mip level, read bilinearly or nearest at
the mapped point:

| matrix scale `s` | log2 s | level read (256 and 1024) |
|---|---|---|
| 0.5 (enlarging) | -1 | 0 |
| 1.2 | 0.26 | 0 |
| 1.41 | 0.496 | 0 |
| 1.42 | 0.506 | 1 |
| 1.5 | 0.58 | 1 |
| 2.5 | 1.32 | 1 |
| 2.82 | 1.496 | 1 |
| 2.83 | 1.501 | 2 |
| 3 | 1.58 | 2 |
| 6 | 2.58 | 3 |
| 8 | 3 | 3 |
| 4 in x, 1 in y | | 2 |

- The level is `round(log2 g)`, with `g` the larger of the matrix's axis
  scales in input texels per output pixel.
- So level `k` begins at `g = sqrt(2) 2^k`. This is the GPU's nearest-mip
  selection.
- Each level is the 2x2 box mean of the one below.
- A manual mip level of 0 at scale 4 read level 0.

`g` counts input texels per output pixel, so for a node the size of its
input the level does not depend on `N`. It does where input and output
differ in size: `g` is the matrix scale times `2^(input log2 - node log2)`.

## 3. Nodes of another size than their input

**[measured]** Biomech's n1497 renders at 16 px whether the graph
renders at 256 or at 2048. So a node with an absolute size keeps it.

**[measured]** Levels and blend nodes with a relative output size were
fed noise:

- **Smaller node** (-1, -2, -3): each output pixel is the box mean of the
  `2^k x 2^k` input block under it, a mip level. It matched to the
  levels node's own rounding (0.0005 mean).
- **Larger node** by one step (+1): plain bilinear enlargement of the
  input (0.0010 maximum error). A cubic B-spline is off by up to 0.17 on
  noise.
- **Larger node** by two steps (+2): two successive 2x bilinear
  enlargements, the second reading the first's result (0.0010 maximum).
  A single 4x bilinear read is off by up to 0.18. A B-spline is off by up
  to 0.12. Blend did the same as levels.

**[shader]** The per-pixel filters (levels, blend modes, HSL, gradient
map, curve, uniform) read their inputs with one plain texture fetch at
the pixel's uv. None has a size or texel uniform. The engine prepares
the input at the node's size beforehand: box mips going down, repeated
2x bilinear steps going up.

**[inferred]** Every px@256 rule in section 1 uses the node's own `N`,
not the graph's.

## 4. Shaders that read the image size

**[shader]** Uniforms in texels or pixels, by shader:

| shader | size-dependent uniform | use |
|---|---|---|
| warp | texel size | backward differences of the gradient |
| emboss | texel size | backward differences of the gradient |
| normal | texel size | backward differences of the height |
| sharpen | texel size | the four neighbours |
| motion blur, 3-tap | step vector, through a matrix | one step either side |
| motion blur, frame shear and box sums (four shaders) | column shift per row, texel size, block sizes | sheared frame, `2^k` block sums, reading back |
| four-way interleave (likely the motion blur's frame) | one-texel offset | picking a neighbour column |
| unidentified: 6 taps along a gradient's turned slope | texel size | backward differences of the gradient |
| transformation | input size and its inverse | nearest filtering's texel snap; edge fade |
| distance (jump flooding, several) | image size | seed distance in pixels |
| FX-Map vertex stage | grid cells per unit and cell size | quad corners snapped to pixels (docs/fxmap-splats.md) |
| 4x4 block codecs (five shaders) | block size in texels | compressed output formats |

These shaders take only uv or unit uniforms:

- directional warp (a uv vector);
- the per-pixel filters;
- the gradient generator (coefficients of uv);
- the FX-Map pattern fragments (pattern space).

`$size` and `$sizelog2` exist only in function bytecode, where the CPU
evaluates them (pixel processors and FX-Map functions). No shader has
them as a uniform.

## 5. Biomech

**[measured]** Std of each node at each size:

| node | parameter | 256 | 512 | 2048 |
|---|---|---|---|---|
| n2049 | ramp | 0.2887 | 0.2887 | 0.2887 |
| n2336 | blur 0.37 of n2049 | = n2049 | 0.2878 | 0.2878 |
| n2338 | blend 0.5 (n2049, 1 - n2336) | 0.0000 | 0.0101 | 0.0152 |
| n2339 | levels 0.63, 0.63 | 0.0000 | 0.0442 | 0.0312 |
| n2340 | distance 8.96 | 0.0000 | 0.1488 | 0.1502 |
| n2343 | pattern 0 of n2344 | 0.0000 | - | 0.1311 |
| n2344 | FX-Map | 0.0486 | - | 0.2029 |
| n2345 | motion blur 0.2838 of n2344 | = n2344 | - | 0.1939 |
| n2355, n2357 | | 0.0000 | - | 0.034, 0.051 |
| n2366 | levels 0.1287, 0.1313 (trace mask) | 0.2073 | - | 0.4989 |

The same chain runs through n2322 (directional warp 0.5), n2323 (blur
0.37), n2325, n2326 (levels 0.63, 0.63) and n2327 (distance 5.12), into
n2335 and n2355.

Blur 0.37 gives `R = 0.37 N / 256` px, so it is identity up to 346 px.
Motion blur 0.2838 is identity up to 451 px.

n1500 to n1593 is 1.0 (n1500) and flat at 256, 512, 1024 and 2048:

- The outer Iterate count is 1 and the inner one 30.
- Splat `k` is at `y = (k + 1/2) / 30` and `x = 1/2 - lum(n19)`.
- Each splat is `(1, 25.41 / 30)` cells: half size (0.5, 0.4235).
- Images are patterns 8 to 13 (n265, n1495, n1252, n1003, n757, n511),
  nearest filtering, chosen by `int(rand(6)) + 8`.
- Max blending, opacity 1.
- sbsar2ptex's splat walk and `fxdraw.render`, fed Substance's renders
  of n19 and n1496 to n1499, give 1.0 at every pixel at 256 and 2048.
- The translated graph's n1525 at 256 is a grid that repeats every 1/30
  of the image. Substance's is flat 0.9375.

## 6. Acceptance tests

Every expected value below was measured with sbsrender at 16 bits. The
inputs are 16-bit images at the node's size. In each image:

- a "line" is 1.0 in column `N/2` and 0 elsewhere;
- a "dot" is 1.0 at pixel `(N/2, N/2)` and 0 elsewhere;
- a "ramp" is `(x + 0.5) / N` along x;
- "noise" is uniform random values.

A tap list gives the output row through the line or dot, from the
leftmost non-zero pixel, at the given offset from column `N/2`.

| # | node | settings | N | expected |
|---|---|---|---|---|
| 1 | blur | intensity 1, line | 256 | taps at -1..1: 0.25, 0.5, 0.25 |
| 2 | blur | intensity 1, line | 2048 | 15 taps of 0.0625 at -7..7, and 0.03125 at -8 and +8 |
| 3 | blur | intensity 0.75, line | 256 | taps at -1..1: 1/6, 2/3, 1/6 |
| 4 | blur | intensity 0.37, line | 256 | the input unchanged |
| 5 | blur | intensity 0.37, line | 512 | taps at -1..1: 0.1622, 0.6757, 0.1622 |
| 6 | motion blur | intensity 1, angle 0, line | 256 | as test 1 |
| 7 | motion blur | intensity 1, angle 0.125, dot | 256 | (-1, -1): 0.2071, (0, 0): 0.5858, (+1, +1): 0.2071 |
| 8 | warp | intensity 1, line, gradient = ramp | 64 to 2048 | the line moved -N/256 px (-0.25 px at 64, bilinear: 0.75 and 0.25) |
| 9 | directional warp | intensity 4, angle 0, line, map = 1 | any | the line moved -N/64 px |
| 10 | distance | 16, dot | 256 | 4 px from the dot: 0.75 |
| 11 | distance | 16, dot | 2048 | 4 px from the dot: 0.96875 |
| 12 | sharpen | intensity 1, a 0.75 line on 0.5 | any | the line's neighbours 0.25, the line 1.0 (clamped) |
| 13 | emboss | intensity 4, input 0.5, gradient = ramp | any | maximum 0.5156 |
| 14 | transformation | scale 1.42, bilinear, noise | 256, 1024 | `input` mip level 1 (2x2 means) read bilinearly at `1.42 (uv - 1/2) + 1/2` |
| 15 | transformation | scale 1.41, bilinear, noise | 256, 1024 | as test 14 with level 0 |
| 16 | transformation | scale 4 in x and 1 in y, noise | 256 | level 2 |
| 17 | transformation | offset 1/256 uv, line | 256, 1024 | the line moved -1 px at 256 and -4 px at 1024 |
| 18 | levels (in 0 to 0.999), relative size -2 | noise | 256 | each output pixel is the mean of the 4x4 input block under it, divided by 0.999 |
| 19 | levels (in 0 to 0.999), relative size +2 | noise | 256 | two successive 2x bilinear enlargements, each sampled at pixel centres, divided by 0.999 |

Tests 14 to 19 match to the stated rule within 0.003, the rounding of
the GPU filter's fraction to 1/256. Tests 1 to 13 match to within one
16-bit level.

## 7. Unknown

- n1500's difference at 256 needs a Material Maker render of n1500 and
  its records buffer.
- The motion blur's long-box path was checked only through its
  footprint, not tap for tap.
- The transformation's edge fade width was not measured.
- The +3 or larger enlargement chain, and an enlargement of a node with
  an absolute size, were not measured.
