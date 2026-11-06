# Function Image Samples (function ops 0x33 samplelum, 0x34 samplecol)

## Clean-room behavioral specification

This document specifies the two function ops that read an image,
`samplelum` (0x33) and `samplecol` (0x34): how they are encoded, which
image they read, and what value they return. It covers every place they
occur in the node corpus, and in particular **Value Processors**, where
sbsar2ptex could not translate them. It is intended to be sufficient for
an independent implementation without access to the original
implementation. No engine code and no file data are reproduced;
third-party and library graphs are cited by name and node number only.

It extends `docs/function-ops.md`, whose section 0 (bytecode basics) and
section 2 (value slots, Value Processors) it assumes. Section 1.3 below
corrects one statement of its section 2.10.

It was recovered from:

- The 7 corpus graphs that failed with `cannot translate samplelum` or
  `cannot translate samplecol` (`nodecorpus/baseline.tsv`): every
  sample op in them, located by node and block.
- Probe graphs cooked with Designer's `sbscooker` and rendered with
  `sbsrender` (vk engine unless stated; also d3d11 and sse2), read back
  as 32-bit float TIFF. Each probe is: a pattern pixel processor `P` of
  a chosen size and format, 16 Value Processors each sampling `P` at one
  position, and a 4 × 4 reader pixel processor (format 32F) whose pixel
  `k` writes Value Processor `k`'s value. About 55 probe graphs, 16
  samples each.
- The function graphs of Designer's shipped library packages (`.sbs`),
  for how the authoring tool writes the op's data.

Each statement is tagged:

- **[measured]**: established by probes or by the corpus.
- **[inferred]**: fits the data, but was not tested directly.

`float32` means IEEE-754 binary32, round to nearest, ties to even.

## 1. Encoding

### 1.1 The instruction

**[measured]** The opcode carries the usual `count`, `type` and `width`
(`docs/function-ops.md` section 0). The type is always float. The width
is 1 for `samplelum` and 4 for `samplecol`. The operands are u16s:

| operand | meaning |
|---|---|
| 0 | SSA index of the **position**, a float2 value |
| 1 | the **input index**: a literal small integer, not an SSA index |
| 2 | the **filtering**, a literal: present only when it is 1 (bilinear) |

So the instruction has `count` 2 (nearest) or 3 (bilinear). The
opcodes seen are:

| op | nearest (`count` 2) | bilinear (`count` 3) |
|---|---|---|
| samplelum | `0x0933` | `0x0D33` |
| samplecol | `0x09F4` | `0x0DF4` |

Neither has a u32 operand, so neither is ever padded.

Example u16 stream (it starts with the block's instruction count):
`0003 1140 FFFF 3F7F FFFF 3F7F 0933 0000 0000 0906 0001 0003` is const
float2 (0.99999994, 0.99999994); samplelum (#0, input 0), nearest; set
slot (#1 into slot 3). This is atlas_scatter n107's block;
atlas_spliter n83 and floodfill_to_index n56 differ only in the slot.
A bilinear example is PBR_render n25's pixel-processor `samplelum
(#325, input 0, filtering 1)`.

### 1.2 In the .sbs file

**[measured]** The authoring tool writes the op's constant as an int2,
`<constantValueInt2 v="INDEX FILTERING"/>`, in every one of the shipped
library packages (index 0 to 3, filtering 0 or 1). The cooker drops a
filtering of 0 and keeps a filtering of 1 as operand 2.

**Filtering numbering.** For these ops, **0 is nearest and 1 is
bilinear** (section 3.4 measures it). This is the reverse of the
Transformation node's own filtering parameter (`docs/resolution.md`
section 2), where 1 is nearest. Do not share a table between the two.

### 1.3 Correction to docs/function-ops.md 2.10

**[measured]** That section reports that "a Value Processor's
`samplelum` of a uniform image input was folded by the cooker to 0.0".
The probe it cites wrote the op's constant as an **int1**
(`<constantValueInt1 v="0"/>`). Written that way, the cooker folds the
sample to a constant zero whatever the input is: a probe sampling a
non-uniform pixel processor with an int1 constant cooked to
`const float4 (0, 0, 0, 0); set slot` and rendered 0. With the int2
form, the same Value Processor keeps its sample and renders the input's
texel (section 3). So the folding came from the constant's form, not
from the uniform input. Whether the int2 form of a sample of a
*uniform* input is folded is untested (section 7).

## 2. Where the ops occur

### 2.1 Node kinds

**[measured]** In the 7 graphs, every sample op is in a block of an
op 40 or op 41 node (1,175 blocks: 252 in op 40 nodes, 923 in op 41
nodes). None is in an FX-Map's functions or in another filter's
parameter function. Those blocks are of two kinds:

- **Pixel processors** (op 41, and op 40 with a zero high half in its
  first field): the per-pixel function. sbsar2ptex already translates
  these.
- **Value Processors** (op 40 whose first field has a non-zero high
  half, the value's width, `docs/function-ops.md` 2.4): the set-slot
  block. **These 17 blocks are the cause of all 7 failures.** The
  converter translates a slot writer's block for its readers
  (`docs/function-ops.md` 2.6) but has no image to give its samples.

**[measured]** A Value Processor has exactly one image input: its first
field is `width << 16 | 1` (0x10001, 0x20001, 0x40001 in the corpus and
probes). Its sample ops all use input index 0, which is that input (the
node's single input field after the first field).

### 2.2 The 17 Value Processor samples

**[measured]** All use nearest filtering (two operands). "Input size" is
the image input node's output size, `(log2 w, log2 h)`, where it is
fixed; "function" means the input has an output-size function.

| graph | VP node | width | op | position | input node | input size |
|---|---|---|---|---|---|---|
| PBR_render | n21, n22 | 1 | samplecol | (0, 0) | n20 (op 41) | (0, 0): 1 × 1 |
| atlas_scatter | n107 | 1 | samplelum | (0.99999994, 0.99999994) | n106 (op 2, blend) | function |
| atlas_scatter | n131 | 1 | samplecol | (0, 0) | n120 (op 41) | function |
| atlas_spliter | n83 | 1 | samplelum | (0.99999994, 0.99999994) | n82 (op 2) | function |
| atlas_spliter | n96, n116, n126 | 1 | samplecol | (0, 0) | n95, n115, n125 (op 41) | function |
| atlas_spliter | n98, n128, n143 | 2 | samplecol | `(0 / s.x, floor(0 / s.x) / s.y)` with `s` a float2 get slot: (0, 0) | n87 (op 5) | function |
| floodfill_to_index | n56 | 1 | samplelum | (0.99999994, 0.99999994) | n54 (op 2) | (8, 8) |
| mask_to_paths_2 | n19 | 1 | samplelum | (0, 0) | n18 (op 4) | (0, 0): 1 × 1 |
| quantize_color | n50, n724 | 1 | samplelum | (0, 0) | n6 (op 4), n723 (op 40) | (4, 4), (8, 0) |
| rt_irradiance | n32, n33 | 1 | samplecol | (0, 0) | n31 (op 41) | (0, 0): 1 × 1 |

What the blocks do with the sample:

- PBR_render n21 / n22, rt_irradiance n32 / n33, atlas_scatter n131,
  atlas_spliter n96 / n116 / n126: `max(c.r, c.g)` or `1 - max(c.b,
  c.a)` of the sampled colour (a min/max reduction read back).
- atlas_scatter n107, atlas_spliter n83, floodfill_to_index n56: the
  last texel's luminance, stored as is.
- mask_to_paths_2 n19: `floor($size.x * $size.y * lum + 0.5)`, where
  `$size` is the graph's size (`docs/function-ops.md` 2.4), not the
  input's.
- quantize_color n50: 2^21 times the luminance.
- atlas_spliter n98 / n128 / n143: a 60-instruction float2 computed from
  the sampled colour's (b, a) and constants. Their position is (0, 0)
  whenever the slot `s` is non-zero.

The inputs read at (0, 0) of a 1 × 1 image, or at the last texel of a
larger one, are the ends of reduction chains: a sequence of pixel
processors (for example PBR_render n14 to n20), each combining several
samples of the one before into a smaller image.

## 3. Semantics

### 3.1 The image read

**[measured]** A sample reads the stored image of its input node: the
input's output buffer, at the **input's own size** `(W, H)`, holding
texels quantised to the input's own **output format**. It does not read
at the graph's size or at the reading node's size.

- Probes: the pattern was 8 × 8 (absolute size) while the graph was
  4 × 4; also 1 × 1, 16 × 1, 8 × 4 and 4 × 32. Every sample matched the
  pattern's texels at the pattern's size.
- Texel values, by the input's format parameter, measured on texel
  centres with nearest filtering:

  | format | stored value of a computed `v` | max error of that rule |
  |---|---|---|
  | 3, 32-bit float | `v` (float32), unclamped (values above 1 were read back) | 6e-8 |
  | 1, 16-bit | `round(clamp(v, 0, 1) × 65535) / 65535` | 3e-8 |
  | 2, 16-bit float | `v` converted to binary16 **rounding toward zero** | 0 |
  | 0, 8-bit | close to `round(clamp(v, 0, 1) × 255) / 255`, but see section 7 | one level |

  The format is a property of the input's buffer, and applies to any
  reader of it, not only to samples. It is given here because a sample
  returns it exactly.
- **Mip level.** Every sample returned a single texel's value (nearest)
  or an exact bilinear mix of four (section 3.4). So a sample reads the
  full-resolution level 0, never a mip level.

### 3.2 The texel addressing

The position `p = (p.x, p.y)` is in UV units of the input: (0, 0) is the
top-left corner of texel (0, 0) and (1, 1) the bottom-right corner of
texel (W - 1, H - 1). Texel `(i, j)`'s centre is `((i + 0.5) / W, (j +
0.5) / H)`. Row `j` counts down from the top, as in the image file.

**Wrap.** **[measured]** Positions outside [0, 1) wrap, on both axes,
always. Neither the Value Processor's nor the input's tiling parameter
(0, 1 or 2 probed on each) changed any sample. Negative positions wrap
as `mod` with a non-negative result.

### 3.3 Nearest (filtering 0)

**[measured]** The texel read is

    i = floor(p.x × W) mod W
    j = floor(p.y × H) mod H

with `p.x × W` computed in float32 (exact, since `W` is a power of two)
and `mod` the non-negative remainder. A position exactly on a texel
edge reads the texel to its right or below: x = 0.125 on an 8-wide
image reads column 1, x = 0.12499999 reads column 0, and x = 1.0 reads
column 0. Every probe position matched exactly, including -0.0001,
-0.9999, -3.3, 17.03, 0.99999, 1.0 and edges approached from both
sides to within 1e-5 texel widths.

### 3.4 Bilinear (filtering 1)

**[measured]** With texel centres at `(i + 0.5) / W`:

    u = p.x × W - 0.5            v = p.y × H - 0.5
    i0 = floor(u)                j0 = floor(v)
    fx = u - i0                  fy = v - j0
    T(i, j) = the stored texel (i mod W, j mod H)
    value = (1 - fx)(1 - fy) T(i0, j0) + fx (1 - fy) T(i0 + 1, j0)
          + (1 - fx) fy T(i0, j0 + 1) + fx fy T(i0 + 1, j0 + 1)

The neighbours wrap (section 3.2). The weights are full precision: the
measured values were within 1e-7 of this formula evaluated in float64,
where a GPU filter with 8-bit sub-texel weights would be off by up to
2e-4 on the probe's texel steps. Evaluating it in float32 (either as the
four weights or as two nested `mix`) is within 2e-7 of the measured
values.

Examples: at (0, 0) or (0.5, 0.5) on an 8 × 8 image the value is the
mean of the four texels around that corner; at (0, 0) those are the
four corner texels (0, 0), (7, 0), (0, 7) and (7, 7). On a 1 × 1 image
every position returns the single texel.

### 3.5 Luminance and colour

**[measured]**

| op | colour input (r, g, b, a) | grayscale input `v` |
|---|---|---|
| samplelum | `(r + g + b) / 3`: the mean of the three colour channels; alpha is ignored | `v` |
| samplecol | `(r, g, b, a)` | `(v, v, v, 1)` |

The mean is taken of the filtered colour (it is linear, so of the
texels' means likewise). Measured within 8e-8 of `(r + g + b) / 3` on a
32-bit float pattern, nearest and bilinear; luma weights such as (0.299,
0.587, 0.114) or (0.2126, 0.7152, 0.0722) are off by up to 0.2.

### 3.6 Evaluation inside a Value Processor

**[measured]** As for any Value Processor (`docs/function-ops.md` 2.4),
the block runs once per graph evaluation, before its readers. The
sample's value is the same in every pixel of every reader. The value
read back through a reader equalled the texel to 3e-8 (16-bit input)
and 6e-8 (32F input), the read-back's own precision, so the sample
passes through the slot unrounded. A Value Processor's `$pos` is (0,
0), so a sample at `$pos` reads position (0, 0).

### 3.7 Engines

**[measured]** vk and d3d11 gave identical samples (max difference 0 at
nearest, 9e-8 at bilinear). On sse2 a single Value Processor sampled
correctly (texel (3, 5) at (0.4375, 0.6875), and a relative-size input
at the graph's size). With 16 Value Processors in one graph, sse2's
reader showed the first one's value in every pixel; whether that is the
sample or the reader is not known (section 7). The vk results are
normative.

## 4. Translating to GLSL

This section is **[inferred]** from section 3: it states what an exact
translation must compute. It is not a measured behaviour of its own.

**The image.** A Value Processor's slot value depends on its input
image. A translation of the slot's writer (`docs/function-ops.md` 2.6,
"GLSL") must therefore have that image available wherever the writer is
evaluated, as a sampler of the input node's buffer, together with that
buffer's size `(W, H)` (the input's own size, section 3.1; it may be 1 ×
1 or 256 × 1 while the graph is 2048 × 2048). The writer's helper
becomes a function of the graph inputs **and** of that sampler. In every
reader pixel it computes the same value, because the position does not
depend on the reader's pixel.

Equivalently, an implementation may evaluate the writer once into a
1 × 1 buffer and let readers sample that; the value is the same.

**Nearest.** Read the stored texel exactly, with no filtering:

    ivec2 ij = ivec2(mod(floor(p * vec2(W, H)), vec2(W, H)));
    vec4 c = texelFetch(img, ij, 0);

If only a filtered `texture()` lookup is available, sample the texel's
centre, which is exact under any filter:

    vec2 at = (floor(fract(p) * vec2(W, H)) + 0.5) / vec2(W, H);
    vec4 c = texture(img, at);   // level 0

`floor(fract(p) × N)` equals `floor(p × N) mod N` for power-of-two `N`,
except when `fract` rounds (a negative `p` within one float32 ulp of a
texel edge).

**Bilinear.** For an exact result, compute section 3.4's formula from
four nearest reads rather than relying on the GPU's filter, whose
sub-texel weights are commonly 8-bit. A GPU `texture()` with linear
filtering and repeat wrap at `fract(p)` gives the same value to within
about 2e-4 of the texel difference.

**The value.** Then `samplelum` = `dot(c.rgb, vec3(1.0 / 3.0))` for a
colour input and `c.r` for a grayscale one (where the grayscale buffer is
stored as one channel; if it is stored as RGBA grey, `c.r`, `c.g` and
`c.b` are equal and the mean is also right). `samplecol` = `c` for a
colour input and `vec4(c.r, c.r, c.r, 1.0)` for a grayscale one.

**Quantisation.** The texels the engine reads are already quantised to
the input's format (section 3.1). A translation whose intermediate
buffers are float will read the unquantised value: off by up to 1/510
(8-bit) or 1/131070 (16-bit). For a reduction's last texel read as an
index or a count (mask_to_paths_2 n19 multiplies by the texel count and
floors), quantising the read value to the input's format reproduces the
engine; [inferred] it matters only where the result is floored or
compared.

**Bounds.** A sample's bound is [0, 1] per component for 8-bit and
16-bit inputs. A float input (formats 2 and 3) is unbounded.

## 5. The corpus graphs

**[measured]** Running the converter with the uid-table patch
(`nodecorpus/reader_gap/patched_convert.py`) and a stand-in for Value
Processor samples (each sample returning a constant), the graphs give:

| graph | then |
|---|---|
| l_PBR_render | converts (writes a .ptex) |
| l_atlas_scatter | converts |
| l_atlas_spliter | converts |
| l_floodfill_to_index | converts |
| l_quantize_color | converts |
| l_rt_irradiance | converts |
| l_mask_to_paths_2 | fails next with `OverflowError: float too large to pack with f format` (function.py `_flt`), a separate gap |

So implementing section 4 removes the sample failure from all 7 graphs,
and 6 of them then convert. Whether their renders match the truth was
not checked: that needs the real sample, not the stand-in.

## 6. Acceptance tests

### 6.1 Decoding (synthetic)

- **D1.** The stream `0003 1140 FFFF 3F7F FFFF 3F7F 0933 0000 0000 0906
  0001 0003` decodes to: const float2 (0.99999994, 0.99999994);
  samplelum, float width 1, position #0, input 0, nearest; set slot
  float width 1 (#1 into slot 3).
- **D2.** `0DF4 0005 0000 0001` (within a block where #5 is a float2) is
  samplecol, float width 4, position #5, input 0, **bilinear**.
- **D3.** `09F4 0005 0000` is the same with nearest. A reader must not
  take operand 1 as an SSA index.

### 6.2 Probe pattern

The probes' pattern is a pixel processor of size `W × H` whose texel
`(i, j)` (column `i`, row `j` from the top) is, in float32:

- colour: `r = (i + 0.3) × f32(1 / 8.7)`, `g = (j + 0.6) × f32(1 /
  8.9)`, `b = 0.2 + (0.05 i + 0.03 j)`, `a = 0.95 - 0.07 j`;
- grayscale: `v = ((i + 8 j) + 0.5) × f32(1 / 64.3)`.

Each test below gives a Value Processor sampling it with input index 0,
and the value its slot must hold (vk, read back at float32). An
implementation can check its sampler against these without rendering, by
building the pattern's texels from the formula and applying section 3.

### 6.3 Expected values

All with the pattern at 8 × 8, format 3 (32F), unless stated.

| # | op, filtering | position | expected (r, g, b, a) or value |
|---|---|---|---|
| S1 | samplecol, nearest | (0.0625, 0.0625) | (0.0344828, 0.0674157, 0.2, 0.95): texel (0, 0) |
| S2 | samplecol, nearest | (0.1, 0.2) | texel (0, 1): (0.0344828, 0.1797753, 0.23, 0.88) |
| S3 | samplecol, nearest | (0.99999, 0.99999) | texel (7, 7): (0.8390805, 0.8539326, 0.76, 0.46) |
| S4 | samplecol, nearest | (-0.1, -0.2) | texel (7, 6): (0.8390805, 0.7415730, 0.73, 0.53) |
| S5 | samplecol, nearest | (3.7, 2.15) | texel (5, 1): (0.6091954, 0.1797753, 0.48, 0.88) |
| S6 | samplecol, nearest | (1.0, 1.0) | texel (0, 0) |
| S7 | samplecol, nearest | (0.12499999, 0.5) / (0.12500001, 0.5) | texel (0, 4) / texel (1, 4): r 0.0344828 / 0.1494253 |
| S8 | samplecol, bilinear | (0.0625, 0.0625) | texel (0, 0), exactly |
| S9 | samplecol, bilinear | (0.1, 0.2) | (0.0689655, 0.1910113, 0.248, 0.873) |
| S10 | samplecol, bilinear | (0, 0) and (0.5, 0.5) | (0.4367816, 0.4606742, 0.48, 0.705): the four-corner mean, by wrap |
| S11 | samplecol, bilinear | (-0.1, -0.2) | (0.8045977, 0.7303371, 0.712, 0.537) |
| S12 | samplecol, bilinear | (3.7, 2.15) | (0.6206897, 0.1460675, 0.476, 0.901) |
| S13 | samplelum, nearest | (0.0625, 0.0625) | 0.1006328 = (r + g + b) / 3 of texel (0, 0) |
| S14 | samplelum, nearest | (0.99999, 0.99999) | 0.8176710 |
| S15 | samplelum, nearest, gray pattern format 1 | (0.1, 0.2) | 0.1321889 = round(texel (0, 1) × 65535) / 65535 |
| S16 | samplecol, nearest, gray pattern format 1 | (0.5, 0.5) | (0.5676509, 0.5676509, 0.5676509, 1) |
| S17 | samplecol, bilinear, pattern 1 × 1 | any of S1 to S12's positions | texel (0, 0) |
| S18 | samplecol, bilinear, pattern 16 × 1 | (0, 0) | (0.8965517, 0.0674157, 0.575, 0.95): mean of columns 15 and 0 |
| S19 | samplecol, nearest, pattern format 1 | (0.0625, 0.0625) | (2260, 4418, 13107, 62258) / 65535 |
| S20 | samplecol, nearest, pattern format 2 | (0.0625, 0.0625) | binary16 toward zero: (0.0344543, 0.0673828, 0.1999512, 0.9497070) |
| S21 | samplecol, nearest, VP tiling 0 | (-0.1, -0.2) | as S4: no clamp |

Tolerance: 2e-7 per component for S1 to S18 and S21; exact for S19 and
S20.

S19's integers are the 16-bit roundings of texel (0, 0): 0.0344828 ×
65535 = 2259.8 becomes 2260, and so on; the measured values equal them
within 3e-8.

### 6.4 Translation tests

- **T1.** Translating atlas_scatter n107's set-slot block yields a value
  equal to the luminance of input n106's texel `(W - 1, H - 1)`, where
  `(W, H)` is n106's output size. Converting l_atlas_scatter no longer
  fails with `cannot translate samplelum`.
- **T2.** Translating PBR_render n21's block yields `max(c.r, c.g)` of
  input n20's single texel (n20 is 1 × 1).
- **T3.** For all 7 graphs of section 5, conversion no longer raises
  `cannot translate samplelum` or `cannot translate samplecol`.
- **T4.** A writer whose sample position depends on a graph input
  stays live: changing the input changes the slot value as section 3
  says, with no value frozen at the defaults.

## 7. Unknown

- **8-bit storage.** With format 0, some texels were one level below
  `round(v × 255)`: in an 8 × 8 colour pattern, all 8 with alpha 0.81
  (206.55 levels, stored 206), and blue 76.5 and 132.6 (stored 76 and
  132); in an 8 × 8 grey pattern, 5 texels with fractions 0.50 to 0.61.
  Elsewhere round-to-nearest held (96.72 became 97, 184.66 became 185).
  The rule (dithering, a lower-precision computation of the pattern, or
  another rounding) is unknown. It concerns the buffer, not the sample.
- **A uniform input** with the int2 constant: whether the cooker folds
  the sample (section 1.3). Not probed.
- **sse2 with several Value Processors** (section 3.7): only the first
  one's value reached the reader.
- **Pixel processors and FX-Maps.** This document measured samples in
  Value Processors only. sbsar2ptex's existing pixel-processor
  translation uses the same addressing (wrap, nearest at the texel
  centre, bilinear), consistent with section 3, but that was not
  re-measured here. FX-Map function samples did not occur in the 7
  graphs.
- **Input index above 0 in a Value Processor**: never seen; a Value
  Processor has one image input.
- **Filtering values other than 0 and 1**: never seen in the library or
  the corpus.
- **Non-power-of-two input sizes** cannot occur in Substance; section
  3.3's exactness note assumes power-of-two sizes.
- **Whether the engine reads a mip level** for samples in a *pixel
  processor* whose position varies fast across pixels: in a Value
  Processor it reads level 0 (section 3.1).
