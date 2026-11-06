# Warp Node Layout (sbsasm ops 14 and 15)

## Clean-room behavioral specification

This document specifies how a compiled warp node's words are laid out:
which word is the flag word, which are the inputs and where the
intensity lives. It is intended to be sufficient for an independent
implementation without access to the original implementation. It was
recovered from compiled graphs in five
third-party .sbsar files. They are cited here by file name, cook version
and node number only. No engine code and no file data are reproduced.

Each statement is tagged:

- **[measured]**: holds for every warp in the files surveyed.
- **[inferred]**: fits the files, but is not tested directly.

**Node basics.** This is all the rest of this document assumes.

- **Header.** A compiled graph (`.sbsasm`) is little-endian u32 words.
  Its trailer, the file's last 16 bytes, holds:
  - the node count;
  - the offset of the node table (one offset per node, to its node word);
  - the offset of the output map;
  - the offset of the defaults block.
- **Offsets.** Every offset is relative to byte 0x34 of the file.
- **Node.** A node is its node word, then its fields (flag words and
  input node indices), then its code-block offsets (when it has code),
  then inline words, then the code blocks themselves. A node index is a
  position in the node table.
- **rest.** Below, "rest" means the node's u32 words after its node word.

**Correction.** `docs/node-params.md` supersedes the "two code offsets"
below. With bit 0 set, one offset (the size function) always follows the
fields. A second follows only when the pixel-size vector is a function
(bits 2-3 of `flags >> 8` = 2). The intensity's offset, when a function,
comes after it. The intensity's state is bits 5-6, as stated here.

## 1. The node word

**[measured]** The node word is `flags << 16 | x << 8 | op`, with op 14
(grayscale) or 15 (color).

- Bit 0 of `flags` is set when the node has code: two code offsets
  follow the fields. When it is clear, the node has no code, and its
  inline words follow its fields directly.
- `flags >> 8` holds two-bit parameter states. The intensity's state is
  bits 5-6 of that byte:
  - 0: the default, with no inline word;
  - 1: an inline f32;
  - 2: a function, whose inline word is a code offset.

  States 0 and 1 have both been seen. For example, Biomech's n5452 has
  state 0 and no inline word.

## 2. The words after the node word

**[measured]** In order:

1. **The flag word.** It is present in version 9 cooks only. In every
   warp seen it is 0; its bits' meaning is unknown. Section 6 extends
   this to versions 2, 3, 4 and 8, none of which has it.
2. **The input**: a node index.
3. **The gradient**, the displacement map: a node index. As for any
   input, 0xFFFFFFFF would mean no node feeds it.
4. **Code offsets.** There are two when node flag bit 0 is set, and none
   otherwise.
5. **Inline words.** The intensity is the first, when its state is 1 (an
   f32) or 2 (a code offset).

So the four shapes are:

| cook | code | rest |
|---|---|---|
| v5, v6 | yes | input, gradient, offset, offset, [intensity], ... |
| v5, v6 | no | input, gradient, [intensity] |
| v9 | yes | flag, input, gradient, offset, offset, [intensity], ... |
| v9 | no (**not seen**) | flag, input, gradient, [intensity] [inferred] |

Survey: every warp in each file has the shape shown.

| file | cook | warps | with code | without code | flag word |
|---|---|---|---|---|---|
| BiomechZombieCircuits | v5 | 525 | 525 | 0 | none |
| brick_wall | v5 | 399 | 399 | 0 | none |
| persian-mosaic-wall-tiels | v6 | 59 | 50 | 9 | none |
| mosaic | v9 | 16 | 16 | 0 | all, 0 |

persian-mosaic-wall-tiels has nine warps without code (node flags
0x2318): n604, n606, n609, n612, n615, n618, n621, n624 and n638. Their
rest is exactly three words: input, gradient and the intensity f32. Its
coded warps have the same intensity at inline word 0. n660 is one, with
the same value as n638.

## 3. Finding the flag word

The cook version alone separates the layouts seen: v5 and v6 have no
flag word, and v9 has one. Versions 7 and 8 were not seen. The structure
also tells, and the two agree on every warp surveyed.

- **With code [measured].** rest[2] is a code offset exactly when there
  is no flag word. With a flag word, rest[2] is the gradient, a node
  index. A code offset is a multiple of 4 that points into the node's own
  span, past its start.
- **Without code.** rest[2] is not a code offset either way, so that
  test cannot be used. It is the intensity f32 (no flag word), or the
  gradient (flag word), or absent when the intensity has state 0 and
  there is no flag word. **[inferred]:** take a flag word only when
  rest[2] reads as a node index (below the node count, or 0xFFFFFFFF).
  - An f32 intensity reads as a node index only if it is +0.0 or
    denormal. The surveyed intensities are normal floats, and a
    zero-intensity warp does nothing.
  - Where the version is known, check the two against each other.

**The converter's failure on persian-mosaic-wall-tiels.** The converter
applied the "with code" test to all warps, so it misread the nine warps
that have no code: their intensity became the gradient input, a "node"
far past the node list. The cause is the missing code-bit check, not the
cook version.

## 4. Acceptance tests (synthetic words)

`is_offset(w)` below is "a multiple of 4 within the node's span". The
node count is 100. "Intensity 0.5" is the f32 bit pattern 0x3F000000.

| # | node flags | rest | flag word | input | gradient | intensity |
|---|---|---|---|---|---|---|
| 1 | 0x2319 (code) | 7, 8, off, off, 0.5 | none | 7 | 8 | 0.5 |
| 2 | 0x2b19 (code) | 0, 7, 8, off, off, 0.5 | 0 | 7 | 8 | 0.5 |
| 3 | 0x2318 (no code) | 7, 8, 0.5 | none | 7 | 8 | 0.5 |
| 4 | 0x2318 (no code) | 0, 7, 8, 0.5 | 0 | 7 | 8 | 0.5 |
| 5 | 0x0318 (no code, intensity state 0) | 7, 8 | none | 7 | 8 | default |
| 6 | 0x2318 (no code) | 7, 0xFFFFFFFF, 0.5 | none | 7 | (unwired) | 0.5 |

Test 3 is the persian case. A layout chooser that only asks "is rest[2] a
code offset?" gets it wrong: it reads three fields, so the gradient
becomes 0x3F000000 and the intensity is lost.

**Container-level test.** A whole synthetic SBAM file, read by
sbsar2ptex.sbsasm.Asm:

- 0x84 bytes, "SBAM" at 0.
- The uid table at 0x6C: one output (uid 0x1234), no inputs.
- The trailer at 0x74: 3 nodes, node table at 0x60, outputs at 0x64,
  defaults offset 0. All offsets are relative to 0x34.
- Nodes 0 and 1 at 0x40 and 0x44: uniforms (word 12).
- Node 2 at 0x48: the warp of test 3 with inputs 0 and 1. Its node word
  is `0x2318 << 16 | 0x88 << 8 | 14`, followed by 0, 1 and 0x3F000000.
- The node table at 0x60: 0x40, 0x44 and 0x48, each minus 0x34.

Expected: node 2's inputs are {input: 0, gradient: 1}, it has no code,
and its first inline word is 0x3F000000.

## 5. Unknown

- What the v9 flag word's bits mean. All seen are 0.
- v7 cooks: whether they have the flag word. v8 is answered in section
  6 (no flag word).
- v9 warps without code: now seen (section 6). Test 4's shape holds.
- Whether a no-code warp can have a function-driven intensity (state 2).
  Function blocks are code, so probably not. Section 6 shows a no-code
  warp with a pixel-size function (P bits 2-3 = 2), so "no code" means
  only that there is no size function (`docs/node-seed.md` section 1).

## 6. Cook versions 3, 4 and 8

The cook version is the `.sbsasm` header's u16 at byte 6. It is not a
format generation: the same cooker writes versions 2 to 9, by the
features a graph uses (`docs/sbsasm-container.md` section 0). The
question here is which of those versions carry the warp's leading flag
word. The converter knew only 2, 5, 6 (none) and 9 (one), and refused
the rest. That refusal is what stopped the four corpus graphs
l_cracks_weathering, l_moss_weathering, l_rock_weathering and
l_rust_weathering, all cooked to version 4 ("a warp in cook version 4,
whose leading flag word is unknown").

### 6.1 The rule

**[measured]** The flag word is present **exactly in version 9**. Cook
versions 2, 3, 4, 5, 6 and 8 have none: their warp is input, gradient,
then the words of `docs/node-params.md` section 2.

| cook version | flag word | warps checked | source |
|---|---|---|---|
| 2 | no | 189, plus probes | corpus |
| 3 | no | 8 | corpus (l_crystal_2, l_grunge_map_002, ...) |
| 4 | **no** | 190 (71 with a size function, 119 without) | corpus, including the four failing graphs |
| 5 | no | 656 | corpus and brick_wall |
| 6 | no | 59 | persian-mosaic-wall-tiels |
| 8 | **no** | probe | a warp plus an image graph input with a default value |
| 9 | yes, 0 | 12,891 (12,817 with a size function, 74 without), plus a probe | corpus and mosaic |

So a reader takes the flag word when the version is 9, and otherwise
not. An unknown version (7, or above 9) stays unsupported.
**[inferred]** A version above 9 has the flag word too, but none was
seen.

### 6.2 How it was established

**Corpus survey [measured].** Every warp node (ops 14 and 15) was
checked in the 645 node-corpus archives, whose library graphs span
versions 2 to 9, the 5 community archives and the 33 material-pack
archives. Each warp was tested against both hypotheses, two fields (input, gradient) and three (flag 0, input, gradient), with the
word rule of `docs/node-params.md`:

- count the words the node's states call for: the seed slot (seed state
  1 or 2), the size offset (flag bit 0), the pixel-size vector (P bits
  2-3: 0, 2 or 1 words), and the intensity (P bits 5-6: 0, 1 or 1);
- the hypothesis fits when the fields plus those words end exactly at
  the node's first code block, or at the node's end when the node has no
  offsets, and, for three fields, when the first word is 0.

Every warp of versions 2 to 6 fits two fields only. Every warp of
version 9 fits three fields only. No warp fits both or neither. The last
node of a file was left out when it had no code, since its end cannot
be told from the next node.

The version-4 warps take both shapes of section 2's table:

- **No size function.** l_cracks_weathering n17: flags 0x2318, then
  `rest` = input 13, gradient 16, intensity 0.02 (three words).
  n123: flags 0x2718 (pixel-size state 1), then input 122, gradient
  121, the pixel-size vector as two f32, and the intensity (five words).
  A flag word would make the input 0 and leave a word over.
- **With a size function.** l_grunge_galvanic_large n24: `rest[2]` is a
  code offset, so there are two fields.

**Probes [measured].** One warp (intensity 1.25, both inputs from one
grayscale image graph input) was cooked with sbscooker in three graphs.
Only what else the graph held changed:

- the warp alone: version 2, no flag word;
- with the image input given a default value: version 8, no flag word;
- with a pixel processor whose function has a While Loop feeding the
  gradient: version 9, with the flag word 0.

In all three the rest of the warp's words were the same: input,
gradient, the size offset, the pixel-size offset, intensity 1.25. So
the flag word goes with the file's version, not with anything in the
warp itself.

**A no-code warp in version 9.** The survey has 74. For example,
l_grunge_splashes_dusty n134 has flags 0x2318 and `rest` = 0, 133, 133,
then the intensity f32. l_mg_dirt n127 has flags 0x0318 (intensity state
0) and `rest` = 0, 121, 126. Section 2's inferred v9 no-code shape is now
measured.

### 6.3 Acceptance tests

As in section 4. "v" is the cook version given to the reader; the node
count is 200.

| # | v | node flags | rest | flag word | input | gradient | intensity |
|---|---|---|---|---|---|---|---|
| 7 | 4 | 0x2318 | 13, 16, 0x3CA3D70A | none | 13 | 16 | 0.02 |
| 8 | 4 | 0x2718 | 122, 121, p, p, 0x3EC7A0F9 | none | 122 | 121 | 0.3899 (pixel size (p, p), p = 0x3F0C8A03) |
| 9 | 4 | 0x2319 | 23, 23, off, 0x3D8F5C29 | none | 23 | 23 | 0.07 |
| 10 | 3 | 0x2B19 | 4, 7, off, off, 0x3F4F5C29 | none | 4 | 7 | 0.81 |
| 11 | 8 | 0x2B19 | 0, 0, off, off, 1.25 | none | 0 | 0 | 1.25 |
| 12 | 9 | 0x2318 | 0, 133, 133, 0x3E6B851F | 0 | 133 | 133 | 0.23 |
| 13 | 9 | 0x0318 | 0, 121, 126 | 0 | 121 | 126 | default |
| 14 | 7 | any | any | reported as unsupported | | | |

Tests 7 and 8 are the failing corpus shapes: a reader must accept
version 4 and read two fields. Test 11 has input 0 and gradient 0. A
reader that looked for a flag word by "is `rest[0]` zero?" would take
the input as a flag word there, which is wrong. Decide by version.

**Corpus.** With version 3, 4 and 8 read as having no flag word, and
the uid-table reader fix applied, all four version-4 graphs convert to
a `.ptex`. Each then reports one node it handles approximately or not at
all, each unrelated to the warp:

- l_cracks_weathering: an op 1 node with no plugin node;
- l_moss_weathering: an op 40 node with no plugin node;
- l_rock_weathering: a warp whose pixel size is not (1, 1);
- l_rust_weathering: a blur whose pixel size is not (1, 1).

`docs/node-params.md` section 7 covers the pixel-size cases. The renders
were not compared to the truth.

### 6.4 Still unknown

- Version 7: never seen, from any graph or probe.
- What raises a file to version 9, beyond the loop seen here and the
  examples in `docs/sbsasm-container.md`. The rule does not need it.
- The flag word's meaning. It was 0 in all 12,891 version-9 warps.
