# SVG Nodes (sbsasm ops 10 and 11)

## Clean-room behavioral specification

This document specifies the compiled Substance **SVG** filter: its op
numbers, its node words, the vector record it points at, and how that
record is rendered. It explains `UnsupportedError: op 10` and `op 11` in
the node corpus (`a_svg_g_*`, `a_svg_c_*`). It is intended to be
sufficient for an independent implementation without access to the
original implementation. No engine code and no file data are reproduced;
third-party files are not used.

It was recovered from:

- probe graphs cooked with Designer 16's `sbscooker` (cook version 2): one
  SVG node whose vector resource is a small hand-written `.svg` file
  (triangles, squares, a self-intersecting star under both fill rules,
  a square with a hole, a circle, transforms, several `width` / `height` /
  `viewBox` combinations, strokes, linear and radial gradients, colours in
  colour and grayscale mode), with and without a background colour or a
  background input, with absolute and relative output sizes;
- `sbsrender` renders of those probes (16-bit PNG, 64, 256 and 1024
  pixels, and 256 × 128, 64 × 256), compared pixel for pixel with an
  independent rasterization of the cooked record (sections 4 and 5);
- the corpus graphs `a_svg_{c,g}_default_{coded,noise}` and their truth
  renders.

Each statement is tagged:

- **[measured]**: established by probes or renders;
- **[inferred]**: fits the data, but was not tested directly.

`docs/node-seed.md` section 0 defines words, offsets (relative to byte
0x34 of the `.sbsasm`), `rest`, code blocks and the node word
(`flags << 16 | x << 8 | op`); its section 2 places the fields, the seed
slot and the size offset. `docs/node-params.md` section 1 defines
parameter states. This document uses both.

## 1. Identity

**[measured]** Cooking the atomic filter `svg` gives op **10** for a
grayscale node and op **11** for a colour node (the filter's
`colorswitch`, default true). The filter's parameters (Designer 16):

| parameter | type | default | stored |
|---|---|---|---|
| `vectorresourcepath` | string: a package resource of type `svg` | empty | the vector record (section 3) |
| `background` | colour, also connectable as an image input | (0, 0, 0, 0) | node word P and inline words, or an input (2.2) |
| `colorswitch` | bool | true | the op (10 / 11) |
| `$tiling` | enumeration | 3 | **nothing**: tiling 0 cooks to the same words |
| `$randomseed` | integer | 0 | nothing: a constant seed of 5 cooks to the same words |

The SVG text itself never reaches the `.sbsasm`. The cooker parses it,
flattens and tessellates it, and stores only the result (section 3).

## 2. Node layout

### 2.1 Node word

**[measured]**

- `op` 10 or 11. The byte `x` was 0x88 in every probe.
- `flags` bit 0: the node has an output-size function (a relative output
  size; the corpus nodes have it). Bit 3 was set in every probe and node;
  its meaning is unknown and it adds no word.
- `flags` bits 6-7: the seed state (`docs/node-seed.md` section 1). It was
  0 in every probe; states 1 and 2 were not seen.
- **P** = `flags >> 8` (node-word bits 24-31). Bits 1-2 are the
  **background state**:

  | P bits 1-2 | background | words |
  |---|---|---|
  | 0 | the default, (0, 0, 0, 0) (gray 0) | none |
  | 1 (P = 0x02) | a constant colour | op 10: one f32; op 11: four f32 (r, g, b, a) |
  | 3 (P = 0x06) | an image input | one word: a node index |
  | 2 | not seen; **[inferred]** a function, one code offset | |

  Bit 0 and bits 3-7 of P were 0 in every probe.

### 2.2 Words after the node word

**[measured]** In file order:

| word | present |
|---|---|
| the **record offset** | always: `rest[0]` |
| the seed slot | seed state 1 or 2 only (never seen; position **[inferred]** from `docs/node-seed.md` 2.1) |
| the output-size function's offset | `flags` bit 0 |
| the background: one or four f32, or one node index | P bits 1-2 = 1 or 3 |
| then the record and the code blocks | |

Examples:

- corpus `a_svg_c_default_coded` (flags 0x0009, P = 0): record offset,
  size offset;
- a fixed-size probe (flags 0x0008): record offset only;
- constant background, op 11, fixed size (node word 0x0208880B): record
  offset, then 0.2, 0.4, 0.6, 0.8;
- constant background, op 10, relative size (0x0209880A): record offset,
  size offset, then 0.4;
- wired background (0x0608880B): record offset, then the node index of the
  background's source.

**Gray background** **[measured]**: a constant colour (r, g, b, a) is
stored in op 10 as one f32, the mean of r, g and b: (0.2, 0.4, 0.6, 0.8)
is stored as 0.4.

**Background input** **[measured]**: when an image feeds `background`,
the cooker inserts a transformation node (op 4 gray / op 5 colour) between
the source and the SVG node; the node index points at that transformation.
This is ordinary graph structure and needs nothing special.

## 3. The vector record

### 3.1 Words

**[measured]** The record offset points at a run of u32 words, a tagged
stream. Each word's low 3 bits are its **tag**; `w >> 3` is its value.

| tag | meaning | value | followed by |
|---|---|---|---|
| (the word 0) | end of the record | | |
| 3 | **solid paint**: starts a new paint | the colour (below) | nothing |
| 4 | **gradient paint**: starts a new paint | `n`, the stop count | `2n` words: the stops |
| 5 | gradient plane (only right after a tag 4 and its stops) | 3 | three f32: `a`, `b`, `c` |
| 1 | **triangle strip** of the current paint | `k`, the vertex count | `k` vertex words |
| 2 | **polyline** (a stroke) of the current paint | `k`, the vertex count | `k` vertex words |

Reading: start at the record offset; read a word; stop at the word 0;
otherwise act on its tag and continue after its following words. Tags 0,
6 and 7 with a non-zero value were never seen: report them as malformed.
A paint applies to the strips and polylines that follow it, up to the next
paint. **[measured]** Every probe record was followed by a second zero
word after the terminating one; its meaning is unknown and a reader must
not need it.

**The empty record.** **[measured]** With no resource (the corpus
default), the record offset points at a single word 0 that the cooker
places at offset 4 of the file's offset space, ahead of the node table.
Nothing is drawn.

**Solid colour (tag 3).** `R = (w >> 3) & 0xFF`, `G = (w >> 11) & 0xFF`,
`B = (w >> 19) & 0xFF`; bits 27-31 were 0 in every probe. Examples:
`#FF8000` is 0x000407FB, `#102030` is 0x01810083, `#FFFFFF` is
0x07FFFFFB, a path with no `fill` (SVG's default black) is 0x00000003.

- **Grayscale (op 10)** **[measured]**: the cooker stores the gray value
  `g = floor((R + G + B) / 3)` in all three bytes: `#FF8000` becomes
  0x03FBFBFB (g = 127), `#FE0000` becomes g = 84, `#102030` g = 32.
- **Opacity is dropped** **[measured]**: `opacity`, `fill-opacity` and an
  `rgba()` alpha all cook to the same word as the opaque colour, and the
  render is opaque. `fill="none"` emits no paint and no strip.
- **CSS `style="fill:…"`** is honoured like the attribute.

**Vertex words** (tags 1 and 2). `X = w & 0xFFFF`, `Y = w >> 16`: the
point's position as a fraction of the document, `X / 65535` across and
`Y / 65535` down (y down, as in SVG). **[measured]** The cooker computes
`round(65535 · x_user / W_doc)` and `round(65535 · y_user / H_doc)`
(section 3.3 defines `W_doc`, `H_doc`); for example x = 224 in a
256-wide document is 0xDFFF and 32.3 is 8269.

**Gradient stops (tag 4).** Each stop is two words `s0, s1`:

- `offset = s0 & 0xFFFF` (the stop's offset × 65535, so 0.2 is 0x3333,
  0.5 is 0x8000, 1 is 0xFFFF);
- `R = (s0 >> 16) & 0xFF`, `G = s0 >> 24`, `B = s1 & 0xFF`,
  `A = (s1 >> 8) & 0xFF`;
- `s1 >> 16` was 0x8000 in every stop; its meaning is unknown.

`A` was 0xFF in every probe, including a stop with `stop-opacity` 0.5:
stop opacity is dropped too.

**Gradient plane (tag 5).** Three f32 `a, b, c`: the gradient parameter
at a document point `(u, v)` (fractions, y down) is `t = a·u + b·v + c`.
Probes: `x1 = 64, x2 = 192` (user space, a 256 document) gives (2, 0,
−0.5); `(0, 64) → (128, 192)` gives (1, 1, −0.25), the projection onto
the gradient vector divided by its squared length. A gradient in the
default `objectBoundingBox` units cooks to three NaN (0xFFC00000).

**Radial gradients** **[measured]** cook to a tag-3 paint of colour 0
(black), not to a gradient.

### 3.2 Geometry the cooker produces

**[measured]** The cooker does the vector work; the record holds its
output:

- **Fills** are triangulated into triangle strips (tag 1). Strip `k`
  vertices give triangles `(v_i, v_{i+1}, v_{i+2})` for `i = 0 .. k−3`.
  The cooker joins separate pieces (subpaths, holes, the star's points)
  into one strip with repeated vertices, so many triangles are
  degenerate (zero area); they draw nothing.
- **Fill rule.** The SVG `fill-rule` is applied by the cooker: a
  self-intersecting star cooks to a different triangulation under
  `evenodd` (its centre left empty) than under `nonzero` or no rule (the
  default, nonzero; centre filled). A square with an inner square of the
  same winding is filled solid; with the opposite winding it has a hole.
- **Curves** (a `circle`) are flattened by the cooker (264 strip vertices
  for a radius-100 circle at the default resource quality). The
  resource's `cookedQuality` presumably sets this; it was not varied.
- **Transforms** (`transform` on elements and groups) are applied by the
  cooker: `translate(32,16)` around `scale(0.5)` moves a 64–192 square to
  64–128 × 48–112.
- **Clipping.** Geometry outside the document box is clipped at cook time
  to `[0, 1]²` (a rectangle from −64 to 320 across became a triangulation
  of 0 to 1); nothing wraps.
- **Strokes** become tag-2 polylines of the stroke colour, the path's
  points in order (a closed path repeats its first point). The stroke
  width is not stored. In an element with both, the fill's paint and
  strips come first, then the stroke's.
- **Paint order** is document order.

### 3.3 Document size

**[measured]** `W_doc` and `H_doc` are the root `<svg>`'s `width` and
`height` when they are plain numbers. When those are absent (or carry a
unit such as `mm` or `%`, which the cooker ignores), they are the
`viewBox`'s width and height. The `viewBox` never scales and its origin
never shifts the drawing:

| root attributes | square 64–192 lands at (fraction × 256) |
|---|---|
| `width=256 height=256 viewBox="0 0 512 512"` | 64–192 (viewBox ignored) |
| `width=256 height=256 viewBox="64 64 256 256"` | 64–192 (origin ignored) |
| `width=128 height=128 viewBox="0 0 256 256"` | 128–256, clipped |
| `viewBox="0 0 512 512"` only | 32–96 |
| `viewBox="64 32 512 256"` only | x 32–96, y 64–192 |
| `width=512 height=256` only | x 32–96, y 64–192 |
| `width="100mm" height="100mm" viewBox="0 0 256 256"` | 64–192 |
| no size at all | the cook fails ("doesn't specify sizes") |

So the drawing is the user-space rectangle `[0, W_doc] × [0, H_doc]`
stretched onto the output. A converter of `.sbsar` files never needs this:
the record already holds fractions.

## 4. Rendering

### 4.1 Output

Let the output be `Wo × Ho` pixels (a power of two each; they need not be
equal). **[measured]** The output starts as the background:

- op 11: the background colour (r, g, b, a), or the background input's
  pixel; default (0, 0, 0, 0);
- op 10: the stored gray, or the input's pixel; default 0.

The background keeps its full precision: (0.2, 0.4, 0.6, 0.8) reads back
13107, 26214, 39321, 52428 in a 16-bit render.

Then each strip triangle and each polyline is drawn, in record order, with
the current paint. A drawn pixel **replaces** what is there:

- op 11: `(R / 255, G / 255, B / 255, 1)` (alpha is always 1);
- op 10: `(g / 255)`.

There is **no antialiasing and no blending**: every pixel is either
untouched or fully replaced. [measured] at 64, 256 and 1024 pixels: a
render has exactly the background's values and the paints' values.

### 4.2 Where a triangle covers

**[measured]** The engine draws the triangles on the GPU with standard
rasterization: a pixel is covered when its centre is inside the triangle,
with vertices snapped to 1/256 pixel and the top-left tie rule. Exactly:

1. Vertex to pixels, in **f32** arithmetic (each step rounded to f32):

   ```text
   s   = f32(X) / 65535
   ndc = s · 2 − 1
   px  = (ndc + 1) · (Wo / 2)          (and the same for Y with Ho)
   ```

2. Snap: `px' = round(px · 256) / 256` (nearest; ties not observed).
3. Pixel `(i, j)` (column `i`, row `j` from the top) is covered when its
   centre `(i + 0.5, j + 0.5)` is inside the triangle `(p0, p1, p2)` of
   snapped points. With the triangle ordered so that its signed area
   `E(p0, p1, p2) > 0`, where
   `E(a, b, p) = (b.x − a.x)(p.y − a.y) − (b.y − a.y)(p.x − a.x)`
   (y down), the centre `p` is inside when, for each edge
   `(a, b)` of `(p1, p2), (p2, p0), (p0, p1)`, either `E(a, b, p) > 0`, or
   `E(a, b, p) = 0` and the edge is a **top or left edge**: `b.y < a.y`,
   or `b.y = a.y` and `b.x > a.x`. Zero-area triangles cover nothing.

All snapped coordinates are multiples of 1/256 and centres of 1/2, so the
test is exact in double precision.

Why each step matters (renders of the star, 256 pixels):

- with the exact `X · Wo / 65535` instead of the f32 chain of step 1, 4
  pixels (nonzero) and 6 (evenodd) on one long edge come out wrong: they
  lie within 1e−5 of the edge;
- without the snap of step 2, 8 and 11 pixels are wrong, and 80 on a large
  triangle at 1024 pixels;
- `X / 65536` instead of `/ 65535`: 16 and 20;
- without the tie rule, a square from (0.5, 0.5) to (100.5, 100.5) gains a
  101st row and column.

With all four, every probe matched at every pixel: triangles, the
square-with-hole in both windings, both stars, the circle, overlapping
paints, two subpaths, at 64, 256 and 1024 square and at 256 × 128 and
64 × 256.

### 4.3 Polylines (strokes)

**[measured]** Each polyline segment is drawn as a one-pixel line, as GPU
line rasterization does, whatever the SVG `stroke-width`: a 128-unit
square outline drew exactly 511 pixels. Using the step-1/2 vertex
positions:

- an x-major segment (|Δx| ≥ |Δy|) from `x0` to `x1` draws one pixel in
  each column whose centre `xc` lies in `[min(x0, x1), max(x0, x1))`, in
  the row `ceil(y(xc)) − 1`, where `y(xc)` is the segment's y at `xc`
  (so a y exactly on a row boundary goes to the row above);
- a y-major segment likewise, with x and y exchanged.

Probes: `(10, 10.5) → (200, 120.25)` drew 190 pixels, columns 10 to 199
(200 excluded), row 62 at column 100; a horizontal edge at y = 64 drew
row 63; a vertical one at x = 64, column 63. **[inferred]** The half-open
end and the exact tie rule for a pixel shared by two segments (the
corners were consistent with each segment drawn on its own).

### 4.4 Gradient paints

**[measured]** For a gradient paint, a covered pixel's colour is:

1. `t = a·u + b·v + c` at the pixel centre, `u = (i + 0.5) / Wo`,
   `v = (j + 0.5) / Ho`.
2. A 256-entry **ramp**: entry `m` (0 to 255) is the stops' colour at
   `t_m = m / 255`, rounded to 8 bits per channel (nearest). The stops'
   colour at `t` is the stop colour below the first offset, the last
   stop's above the last offset, and between consecutive stops (offsets
   `o_k = offset / 65535`) the linear mix
   `C_k + (C_{k+1} − C_k) · (t − o_k) / (o_{k+1} − o_k)`, per channel.
3. Sample the ramp linearly: `x = clamp(256 t − 0.5, 0, 255)`,
   `m = floor(x)`, `f = x − m`,
   `value = ramp[m] · (1 − f) + ramp[min(m + 1, 255)] · f`.
4. Round to an integer, **halves to even**, and divide by 255.

This matched every pixel of five gradient probes (two stops, three stops,
stops not at 0 and 1, slopes of 1/1024 to 2/256 of t per pixel).
Rounding halves up instead misses 32,768 pixels of one probe; evaluating
the stops directly at `t` (no ramp) misses thousands. Alpha is 1.

A NaN plane (`objectBoundingBox` gradients) renders the **first stop's**
colour everywhere: [measured] (red for red→blue, green for a first stop
green at offset 0.2). **[inferred]** The NaN reaches the clamp in step 3
as 0.

## 5. What a translator needs for the corpus

All four corpus nodes (`a_svg_{c,g}_default_{coded,noise}`) have no
resource: the record is the empty record, there is no background, and the
node is a **uniform fill of 0** (op 10) or **(0, 0, 0, 0)** (op 11) at the
size its size function gives. Their truth renders are exactly that. The
same holds for any SVG node with an empty record: the output is the
background (default, constant or input) everywhere.

A translator of non-empty records rasterizes the record per section 4: a
fixed list of triangles and lines with constant or ramp colours. Because
there is no antialiasing, the exact coverage rule (4.2) decides edge
pixels; a shader that tests pixel centres against the triangles with the
same snapping and tie rule reproduces it.

## 6. Acceptance tests

Synthetic nodes; words are given from the node word on (`w0`; `w1` is
`rest[0]`). "→ X" is an offset word pointing at X later in the file. `SZ`
is a size block (integer width 2). Records are given as their word lists.
Renders are 256 × 256 unless stated; values are 8-bit levels.

**Layout**

1. **Corpus default, colour.** `w0 = 0x0009880B`; → `E`, → `SZ`, where `E`
   is a single word 0. No background. Output: every pixel (0, 0, 0, 0).
   Must not raise "op 11".
2. **Corpus default, gray.** `w0 = 0x0009880A`, the same words. Output 0.
3. **Fixed size.** `w0 = 0x0008880B`; → `R`. No size function; the record
   follows directly.
4. **Constant background, colour.** `w0 = 0x0208880B`; → `R`, 0.2, 0.4,
   0.6, 0.8 (f32). Background (0.2, 0.4, 0.6, 0.8).
5. **Constant background, gray, relative size.** `w0 = 0x0209880A`; → `R`,
   → `SZ`, 0.4. Background 0.4.
6. **Background input.** `w0 = 0x0608880B`; → `R`, 1. The background is
   node 1's output.
7. **P bits 1-2 = 2.** Report as unsupported (a background function was
   never seen).

**Record parsing**

8. `R` = 0x000407FB, 0x00000019, 0x20002000, 0x4000DFFF, 0xDFFF6000, 0,
   0: one solid paint (255, 128, 0) with one 3-vertex strip at
   (0x2000, 0x2000), (0xDFFF, 0x4000), (0x6000, 0xDFFF).
9. Two paints: 0x000007FB, 0x19, three vertices, 0x0007F803, 0x19, three
   vertices, 0, 0: red then green, each with one strip.
10. A gradient: 0x00000014, 0x00FF0000, 0x8000FF00, 0x0000FFFF,
    0x8000FFFF, 0x0000001D, 2.0, 0.0, −0.5 (f32), 0x00000021, 0x4000BFFF,
    0x40004000, 0xBFFFBFFF, 0xBFFF4000, 0, 0: stops (0, red) and (1,
    blue); `t = 2u − 0.5`; one 4-vertex strip (the square 64–192).
11. A word with tag 6 or 7: malformed.

**Rendering** (each is a render of the cooked probe, all pixels compared)

12. **Triangle (test 8).** 17,440 pixels are (255, 128, 0, 255), the
    rest (0, 0, 0, 0); pixel (100, 100) is covered, (10, 10) is not.
13. **Tie rule.** Strip (128, 128), (25728, 128), (128, 25728),
    (25728, 25728) in any strip order (the cooked one is (25728, 128),
    (128, 128), (25728, 25728), (128, 25728)): exactly columns and rows
    0 to 99 are covered (10,000 pixels).
14. **Snapping.** The triangle (2560, 2560), (63999, 7680), (5120, 61439)
    covers 1,714 pixels at 64, 27,500 at 256 and 440,000 at 1024.
15. **The f32 chain.** The nonzero star `M 128 10 L 200 240 L 10 90 L 246
    90 L 56 240 Z` (a 256 document), cooked: 17,946 pixels at 256,
    including (column 133, row 187) and (152, 202), which lie 1e−5 outside
    a triangle edge in exact arithmetic.
16. **Painter's order.** Red square 64–192 then blue square 128–240:
    pixel (128, 128) is blue, (100, 100) red.
17. **Gray.** Op 10 with paint 0x03FBFBFB over a square: covered pixels
    are 127, the rest 0.
18. **Non-square.** The record of a white two-triangle path and a gray
    triangle drawn at 256 × 128 and 64 × 256: the rule of 4.2 with `Wo`
    and `Ho` separate matches every pixel.
19. **Gradient (test 10).** Row 128, columns 64, 65, 68, 191: (254, 0, 0),
    (252, 0, 2), (246, 0, 8), (0, 0, 254); outside the square
    (0, 0, 0, 0).
20. **Gradient, slow.** Stops (0, black), (1, white), plane (0.25, 0, 0)
    over the whole image: every channel of pixel column `i` is
    `floor(i / 4)`. With plane (0, 0.1, 0): row `j` is `floor(j / 10)`.
21. **Gradient, three stops.** Stops (0.25, black), (0.5, white), (0.75,
    black), plane (0.5, 0, 0): columns 0-127 are 0; columns 128, 129, 130
    are 1, 2, 4; column 255 is 253.
22. **NaN plane.** Test 10 with the plane NaN, NaN, NaN: the square is all
    red.
23. **Polyline.** Paint white, tag-2 word 0x12, vertices (0x0A00, 0x0A80)
    and (0xC7FF, 0x7840): 190 pixels, one per column 10 to 199; column 100
    has row 62, column 10 row 10, column 199 row 119.

## 7. Unknown

- **Seed states 1 and 2**: not seen; the seed slot's position is
  inferred. No SVG parameter can draw random numbers, so probably never.
- **P bits 1-2 = 2** (a background function) and the other bits of P:
  never set.
- **`flags` bit 3**: set in every SVG node; meaning unknown.
- **The second zero word** after a record, and the **0x8000** half of each
  gradient stop's second word.
- **Curve flattening and `cookedQuality`**: done by the cooker, so a
  converter does not need it; how quality maps to segment count was not
  measured.
- **Gradients in op 10**: not probed. **[inferred]** Stops stored as gray
  bytes as for solid paints.
- **Gradient spread** (`spreadMethod` pad / reflect / repeat), gradient
  transforms, and `gradientUnits="objectBoundingBox"` beyond the NaN
  case: not probed beyond what 3.1 states.
- **Line ends and joins**: whether a polyline's last vertex pixel is
  drawn, and a pixel shared by consecutive segments, were consistent with
  each segment drawn half-open, but not isolated.
- **Other GPUs.** The coverage, snapping and rounding rules were measured
  with the `vk` engine on one machine. **[inferred]** Any Vulkan device
  with 8 sub-pixel bits rasterizes the same; the gradient's half-to-even
  rounding may differ elsewhere.
- **Text, images, patterns, masks, clip paths, `<use>`** inside the SVG:
  not probed.
