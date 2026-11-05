# Substance FX-Map Splats

## Behavioral specification

This document specifies how Substance 3D Designer's GPU engines draw FX-Map
splats, and how its CPU function evaluator computes trigonometry. It was
recovered from the engines' SPIR-V shaders (read for their formulas only),
constant tables in the engine DLLs, and numerical tests against Substance's
own 16-bit renders. No engine code is reproduced here.

Each statement is tagged:

- **[shader]**: read from the engine's shaders.
- **[DLL data]**: read from constant tables in the engine DLLs; the code
  that uses them was not found.
- **[measured]**: established by tests against Substance renders.
- **[inferred]**: fits the shaders and the tests, but was not read anywhere.

`float32` means IEEE-754 binary32, round to nearest, ties to even. `fl(x)`
is `x` rounded to float32.

## 1. Function trigonometry

FX-Map functions (branch offsets, sizes, colours...) and parameter functions
run on the CPU. Their `cos` (0x26), `sin` (0x27) and `cartesian` (0x2E) ops
use a polynomial, not the C library. **[DLL data]** The vk, d3d11 and sse2
engine DLLs all hold these float32 constants:

```text
1/(2 pi)  0.159154937
pi/2      1.57079637
c1        6.28318548   (2 pi)
c3        41.3417015   ((2 pi)^3 / 3!)
c5        81.6052475   ((2 pi)^5 / 5!)
c7        74.1371841   (not (2 pi)^7 / 7! = 76.7059)
```

`c7` is tuned so that the polynomial is 1 at a quarter turn to float32
precision. Taylor's coefficient would leave it 1.6e-4 low.

**[inferred, measured]** The sine of `x`, every step in float32:

1. `t = fl(x * 0.159154937)`, then `t = fl(t - rint(t))`, which puts `t` in
   [-1/2, 1/2].
2. Fold: if `t > 1/4` then `t = 1/2 - t`; if `t < -1/4` then `t = -1/2 - t`.
3. `t2 = t * t`; `p = c5 - c7 t2`; `p = p t2 - c3`; `p = p t2 + c1`; the
   result is `p t`.

`cos(x) = sin(fl(x + 1.57079637))`. `cartesian(rho, theta)` is
`(fl(rho cos theta), fl(rho sin theta))`.

The polynomial's error is non-negative on t in [0, 1/4]. It peaks at about
+1.42e-5 near t = 0.22, and it is 0 at t = 0 and t = 1/4.

**[measured]** In Biomech's n2369, 49 gaussian splats are placed by
`var12 + cartesian(rand(1.5), (rand(2) - 1) 2 pi)`. Using `math.cos` and
`math.sin` there moves each centre by about 1e-5 region units (about 0.002
px). That error is radial and depends on the angle. It leaves 92.43% of
pixels exact at 16 bits; this polynomial gives 98.48%.

The reduction order (step 1) cannot be told apart at this level. Computing
cos in turns, as `t + 1/4`, gives the same result.

Pixel processors run on the GPU and use its native `cos` and `sin`.

## 2. Vertex stage

**[shader]** One right triangle is drawn per splat instance. Its local
corners are (-1,-1), (3,-1) and (-1,3), so it covers the square [-1,1]^2.

Per-instance data:

- the colour, (r, g, b, a);
- the drawn box's full size, and its centre `C`;
- the pattern's full size `2h`, its rotation in turns, and the
  supplement.

The triangle is axis-aligned in region space, so the box is the rotated
pattern's bounding box. The rotation in radians is
`turns * fl(2 pi)`.

**Uniforms** [shader]: an origin `o` with an NDC scale `k`, so that
NDC = (P - o) k - 1, with y negated. Also a snapping grid: cells per unit
`g` and cell size `d`.

**[inferred]** `o` is the render region's origin, `k` = 2 / its size, `g` =
pixels per unit, and `d` = 1 / `g`.

**Snapping** [shader]. Each vertex `P = C + e/2`, with `e` the corner times
the box size, is moved onto the pixel grid:

- low corners go to the nearest pixel edge, halves rounding up;
- far corners round up, then move one more cell.

The pattern coordinate is interpolated from the corners' snapped
positions. It equals `R(theta) (x - C) / h` exactly, with
`R(theta)(x, y) = (x cos + y sin, -x sin + y cos)`. So snapping places the
triangle without changing the pattern.

**Coverage** [shader, inferred]. The low edges lie on the snapped corners.
The fragment discards outside the pattern's support. This is the same
coverage as "corners rounded to pixel boundaries, halves up".

**Tile copies** [inferred]. The vertex stage knows nothing of tiling, so each
tile copy is its own instance with an offset centre.

The colour is passed flat, premultiplied as (rgb a, a). The supplement is
passed flat.

## 3. Fragment stage

**[shader]** Every pattern shader is float32 throughout. None uses
RelaxedPrecision or 16-bit floats.

Each shader:

1. computes `q` (the pattern coordinate) and `r2 = q . q`;
2. discards outside its support (a disc `r2 > 1`, or the square);
3. writes colour times the pattern value.

The gaussian (built-in pattern 4) is `clamp(exp(-11.1 r2) - 1.51e-5, 0, 1)`.
The shaders map to `fxdraw.value`'s pattern ids as follows:

| id | support | value (`s` = supplement) |
|---|---|---|
| 0 | square | 1 |
| 1 | disc | 1 |
| 2 | disc | 1 - r2 |
| 3 | disc | (1 - r2)^4 |
| 4 | disc | clamp(exp(-11.1 r2) - 1.51e-5, 0, 1) |
| 5 | disc | (1 - sqrt r2)^4 |
| 6 | square | 1 - max(\|qx\|, \|qy\|) |
| 7 | square | product over axes of smoothstep((1 - \|q\|) / (s + 1e-10)), clamped |
| 8 | square | (qx + 1)(1 - s) / 2 |
| 9 | disc | (1 - r2)(1/2 + wave((6 s + 1.5) qx) / 2) |
| 10 | disc | bell (1 - smoothstep(qx / (1 - s))) |
| 11 | disc | bell (1 - \|qx\|)^2 |
| 12 | disc | 6.75 r2 (1 - r2)^2 (1 - smoothstep((qx + 1) / (2 (1 - s)))) |
| 13 | square | clamp(1 - 20 (qy^2 + clamp(\|qx\| - 0.77 s)^2)) step(\|qx\|, 1) |
| 14 | disc | 1 - sqrt r2 |
| 15 | square | colour (qx + 1) / 2, plus s on every channel |
| 16 | disc | colour sqrt(1 - r2), plus s on every channel |

In the table:

- `wave` is the polynomial `w (0.77606332 + 0.22393668 w)`.
- `bell` is pattern 3.
- smoothstep arguments are clamped to [0, 1].

Image patterns sample their input at `q / 2 + 1/2`.

## 4. Blending

**[measured]** Splats accumulate in a float32 target. These alternatives
match n2369 much worse:

| target | pixels off by more than half a level |
|---|---|
| float32 (with the polynomial trig) | 1.52% |
| unorm16, rounded after each splat | 70.0% |
| float16 | 95.7% |

The result is clamped and rounded to 16 bits once, at the end.

## 5. Engines

**[measured]** sbsrender's default engine is d3d11 (on an AMD Radeon 780M).
Its renders equal d3d11's. The vk engine agrees with d3d11 except on 0.026%
of n2369's pixels, each by 1 level. The sse2 CPU engine is a different
rasterizer and differs by up to 34 levels.

## 6. Open

After the trig fix, 1.52% of n2369's pixels are still more than half a level
off. What remains is:

- a global offset of about -0.006 levels;
- per-splat centre shifts of about 1e-6 units, with no angle pattern.

Emulating the vertex stage in float32 (centre plus tile offset rounded to
float32, in region units) takes that to 1.26%. The likely remaining sources
are:

- the exact order of the argument reduction;
- float32 sums along the position chain;
- the hardware's interpolation.
