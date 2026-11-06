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
   warp seen it is 0; its bits' meaning is unknown.
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
- v7 and v8 cooks: whether they have the flag word.
- v9 warps without code: none seen, so their shape (test 4) is inferred.
- Whether a no-code warp can have a function-driven intensity (state 2).
  Function blocks are code, so probably not.
