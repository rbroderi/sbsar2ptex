# The .sbsasm Container: Trailer, Output Map, Defaults and Uid Table

## Clean-room behavioral specification

This document specifies how a compiled Substance graph (`.sbsasm`) lays out
the tables around its nodes: the trailer, the node table, the output map,
the defaults block and the uid table, and where each node ends. It explains
three ways the current reader goes wrong on small graphs:

- "no uid table before the trailer" on graphs whose only inputs are images
  (no `$outputsize` / `$randomseed` exposed), and on version 8 and 9 graphs
  with image inputs;
- a wrong uid table (eight outputs, no inputs) on every small graph with
  one output whose defaults block starts with `$outputsize` = (8, 8): the
  defaults read as a table header;
- a node running past its end (an index error) when there is no defaults
  block.

It is intended to be sufficient for an independent implementation without
access to the original implementation. No engine code and no file data are
reproduced; third-party files are cited by name only.

It was recovered from:

- Probe graphs cooked with Designer 16's `sbscooker`, with and without
  `--expose-output-size 0 --expose-random-seed 0` and
  `--expose-pixel-size 1`, with image graph inputs, float graph inputs, and
  image inputs with distinctive default colors.
- The node corpus (one graph per atomic and library node, 645 graphs),
  recooked in part without exposed inputs.
- A survey of every distinct `.sbsasm` in the work folders: 2,550 compiled
  graphs, versions 2 to 9 (v2 1,140; v3 236; v4 647; v5 147; v6 83; v8 10;
  v9 287), checked against each archive's `.xml` description.

Each statement is tagged **[measured]** (cooked, rendered or surveyed) or
**[inferred]** (fits the data, not tested directly).

## 0. Basics

- Little-endian throughout. An **offset** is a byte position relative to
  byte 0x34 of the file (`BASE`); the file position of offset `o` is
  `o + 0x34`.
- **Version.** The u16 at byte 6. **[measured]** It varies with the
  features a graph uses (the same cooker writes 2, 3, 4, 5, 6, 8 and 9; a
  uniform color alone is v2, a blend v4, an image graph input with a
  default value v8, some library pixel processors v9). It is not a format
  generation: the layout below is the same for every version, except the
  size of an image input's default (section 4).
- **end** is the file position of the trailer: file length − 16.

## 1. Section order

**[measured]** In every file but one the sections follow each other in
this order, each starting where the previous ends (on a 4-byte boundary):

1. the header (bytes 0x00 to 0x37);
2. the embedded image data area, from byte 0x38 (possibly empty;
   `docs/image-inputs.md`);
3. the **node table**;
4. the **output map**;
5. the **nodes**;
6. the **defaults block**, which may be absent;
7. the **uid table**;
8. the **trailer** (16 bytes).

Survey: 2,526 files with a defaults block (order: table, outputs, nodes,
defaults, uid table) and 23 without (table, outputs, nodes, uid table).

**Old layout.** **[measured]** One file (previewsphere_sphere, an older
cook, version 2) puts the nodes first, right after its image data:
nodes, node table, output map, (defaults,) uid table, trailer. Sections 3,
5 and 6 state the rules so that they cover both orders.

## 2. Trailer

**[measured]** The last 16 bytes are four u32:

| word | meaning |
|---|---|
| 0 | node count `N` |
| 1 | offset of the node table |
| 2 | offset of the output map |
| 3 | offset of the defaults block, or **0 when there is no defaults block** |

## 3. Node table and output map

**[measured]** The node table is `N` u32 offsets, one per node in node
index order, each pointing at that node's node word. Node offsets need not
increase with the index.

**[measured]** The output map is `K` entries of 8 bytes: u32 `format`
(meaning not needed here; seen 0, 0xC0, 0x100, 0x400, 0x100C0, ...), then
u32 node index. Its entries are in the order of the uid table's output uids
(the XML's output order).

**[measured]** `K` is not stored. The output map runs, without padding,
up to the next section, so

    K = (S − output map offset) / 8

where `S` is the smallest of these offsets that is larger than the output
map's: the node offsets, the defaults block's (when present) and the uid
table's. In the usual layout `S` is the first node, so `K` is known before
the uid table is found. In the old layout no node follows the output map;
`S` is then the defaults block, or, when that is absent, the uid table
itself (section 5 handles that case). This holds in every surveyed file,
and `K` always equals the uid table's output count and the XML's output
count.

## 4. Defaults block

**[measured]** The defaults block holds one default value per graph input,
in the uid table's input order (ascending uid), back to back, with no
padding. A value's size depends on its type (the uid table's type word):

| type | value | bytes |
|---|---|---|
| 0 | float | 4 |
| 1 | float2 | 8 |
| 2 | float3 | 12 |
| 3 | float4 | 16 |
| 4 | int (also bool; `$randomseed`) | 4 |
| 8 | int2 (`$outputsize`) | 8 |
| 9 | int3 | 12 |
| 10 | int4 | 16 |
| 5 | image, **version 2 to 6** | 0 |
| 5 | image, **version 8 and 9** | 16 |

**Image inputs in version 8 and 9.** **[measured]** An image input's
default takes 16 bytes: four f32, the input's default color (R, G, B, A).
A grayscale image input stores its default value in the first f32 and 0 in
the other three. Probes: a color image input with default (0.25, 0.5, 0.75,
1) stores 3E800000 3F000000 3F400000 3F800000; a grayscale one with
default 0.625 stores 3F200000 00000000 00000000 00000000. Survey: all 41
version 8/9 files with image inputs need 16 bytes per image; all 391
version 2 to 6 files with image inputs need 0. (Giving an image input a
default value is itself what raised a probe from version 2 to 8.)

**Absent block.** **[measured]** The trailer's word 3 is 0 exactly when
the block would be empty: no inputs at all, or only image inputs in a
version 2 to 6 graph. It is **not** a sign that the graph has no inputs. A
graph cooked without `$outputsize` and `$randomseed` whose only inputs are
images has no defaults block and a non-empty uid table.

**[measured]** No other type's size depends on the version: with the image
rule above, the sizes add up to the block's length in every surveyed file.

## 5. Uid table

**[measured]** The uid table is:

- u16 `n_out`, u16 `n_in`;
- `n_out` u32 output uids (the XML's output uids, in output map order);
- `n_in` pairs (u32 type, u32 uid), sorted by **strictly ascending** uid.
  `$outputsize`, `$randomseed` and `$pixelsize` are among them when exposed.

It ends exactly at the trailer: its start is
`end − 4 − 4·n_out − 8·n_in`.

**Finding it.** **[measured]** Its start cannot be read from the trailer,
and a forward search is unsafe: the defaults block can hold bytes that read
as a header which also ends at the trailer. Example: a graph with one output
and inputs `$outputsize` (8, 8) and `$randomseed`, with `$outputsize`'s uid
the smaller. Its defaults start with words 8, 8, 0, so the defaults' first
u16 pair reads `n_out` = 8, `n_in` = 0, a table of 36 bytes, and the real
table (4 + 4 + 16 = 24 bytes) follows 12 bytes of defaults: 12 + 24 = 36.
Every single-output graph in the node corpus has this shape (the cooker gives
`$outputsize` and `$randomseed` the same uids in every graph), and so do
556 of the 2,550 surveyed files.

The rule:

1. Compute `K` (section 3) when a node or the defaults block follows the
   output map. Then `n_out` must be `K`. (Old layout without a defaults
   block: `n_out` must instead equal (`at` − output map offset) / 8, the
   output map ending where the table starts.)
2. For each candidate start `at = end − 4 − 4·n_out − 8·m`, for `m` = 0, 1,
   2, ... (nearest the trailer first), while `at` is after the output map:
   accept the first `at` where
   - the header reads (`n_out`, `m`);
   - every type word is a known type (section 4);
   - the uids strictly increase;
   - the inputs' default sizes (section 4, using the file's version) add up
     to `at − defaults block position`, or to 0 when the block is absent.
3. If no candidate passes, the file is malformed: report "no uid table".

**[measured]** On 2,549 of the 2,550 surveyed files (all but the
old-layout one, which the survey did not try with the old-layout variant
of step 1) this finds exactly one table, and its counts equal the XML's
output and input counts. On the old-layout file the table ending at the
trailer has `n_out` 5, `n_in` 0, and 5 × 8 bytes of output map end exactly
at it, which the variant accepts. **[inferred]** that the variant is
right in general: one file.

**Current reader, for comparison.** **[measured]** Of the 2,550 files, the
current reader picks a different uid table on 556 (the
`$outputsize` = (8, 8) shape above in every case inspected; not each
checked), fails with "no uid table before the
trailer" on 58 (version 8/9 image defaults, and image-only graphs without
a defaults block), fails reading the output map on 1, and of those it does
read, 19 have a last node with no words (section 6).

## 6. Where each node ends

**[measured]** A node runs from its offset up to the next larger start
among: the other nodes' offsets, the node table, the output map, the
defaults block (if present) and the uid table. In the usual layout the
node table and output map lie before the nodes and are never the bound;
the last node ends at the defaults block, or at the uid table when the
block is absent. (Taking the node table as the last node's bound whenever
the block is absent gives it an empty or negative extent in the usual
layout.) In the old layout the last node ends at the node table.

## 7. Acceptance tests

Synthetic files built for this specification. Hex is u32 words,
little-endian as stored, file position at the left. Bytes 0x08 to 0x37
are zero except the file length at 0x10 (an implementation must not need
the other header words). Node words are those of an image input (op 32,
`00088820` then its uid), a blur (`13088814`, input node 0, an intensity
constant) and a uniform color (`0108880d` then four floats); their contents
do not matter here. "Extent" is (start offset, end offset) per node index.

**T1. No defaults block; one image graph input; version 2.** 124 bytes.

    0000  4d414253 00020000 00000000 00000000
    0010  0000007c 00000000 00000000 00000000
    0020  00000000 00000000 00000000 00000000
    0030  00000000 00000000 00000014 0000001c
    0040  000000c0 00000001 00088820 00001000
    0050  13088814 00000000 40e80000 00010001
    0060  00002000 00000005 00001000 00000002
    0070  00000004 0000000c 00000000

Expect: uid table at offset 0x28; outputs [uid 0x2000 ← node 1, format
0xC0]; inputs [(uid 0x1000, image)]; no defaults; extents node 0
(0x14, 0x1C), node 1 (0x1C, 0x28). Must not report "no uid table".

**T2. No inputs at all.** 112 bytes.

    0000  4d414253 00020000 00000000 00000000
    0010  00000070 00000000 00000000 00000000
    0020  00000000 00000000 00000000 00000000
    0030  00000000 00000000 00000010 00000000
    0040  00000000 0108880d 3e800000 3f000000
    0050  3f400000 3f800000 00000001 00002000
    0060  00000001 00000004 00000008 00000000

Expect: uid table at 0x24 (n_out 1, n_in 0); outputs [0x2000 ← node 0];
node 0's extent (0x10, 0x24), five words.

**T3. Defaults that read as a table header; version 9.** 140 bytes.

    0000  4d414253 00090000 00000000 00000000
    0010  0000008c 00000000 00000000 00000000
    0020  00000000 00000000 00000000 00000000
    0030  00000000 00000000 00000010 00000400
    0040  00000000 0108880d 3e800000 3f000000
    0050  3f400000 3f800000 00000008 00000008
    0060  00000000 00020001 00002000 00000008
    0070  00001000 00000004 00003000 00000001
    0080  00000004 00000008 00000024

Expect: uid table at 0x30, not at the defaults' start 0x24 (which reads
`n_out` 8, `n_in` 0 and also ends at the trailer); outputs [0x2000 ← node
0]; inputs [(0x1000, int2), (0x3000, int)]; defaults 0x1000 = (8, 8),
0x3000 = 0; node 0's extent (0x10, 0x24).

**T4. Image input default in version 9 (16 bytes).** 168 bytes.

    0000  4d414253 00090000 00000000 00000000
    0010  000000a8 00000000 00000000 00000000
    0020  00000000 00000000 00000000 00000000
    0030  00000000 00000000 00000014 0000001c
    0040  000000c0 00000001 00088820 00001000
    0050  13088814 00000000 40e80000 00000008
    0060  00000008 3e800000 3f000000 3f400000
    0070  3f800000 00000007 00030001 00002000
    0080  00000008 00000900 00000005 00001000
    0090  00000004 00003000 00000002 00000004
    00a0  0000000c 00000028

Expect: uid table at 0x44; inputs [(0x900, int2), (0x1000, image),
(0x3000, int)]; defaults 0x900 = (8, 8), 0x1000 = (0.25, 0.5, 0.75, 1.0),
0x3000 = 7; extents (0x14, 0x1C), (0x1C, 0x28).

**T5. Image input default in version 2 (0 bytes).** 152 bytes.

    0000  4d414253 00020000 00000000 00000000
    0010  00000098 00000000 00000000 00000000
    0020  00000000 00000000 00000000 00000000
    0030  00000000 00000000 00000014 0000001c
    0040  000000c0 00000001 00088820 00001000
    0050  13088814 00000000 40e80000 00000007
    0060  00000008 00000008 00030001 00002000
    0070  00000004 00000800 00000005 00001000
    0080  00000008 00003000 00000002 00000004
    0090  0000000c 00000028

Expect: uid table at 0x34; inputs [(0x800, int), (0x1000, image),
(0x3000, int2)]; defaults 0x800 = 7, 0x3000 = (8, 8), the image none.

**T6. Two outputs, one float input.** 160 bytes.

    0000  4d414253 00020000 00000000 00000000
    0010  000000a0 00000000 00000000 00000000
    0020  00000000 00000000 00000000 00000000
    0030  00000000 00000000 0000001c 00000030
    0040  00000000 00000000 00000000 00000001
    0050  0108880d 3e800000 3f000000 3f400000
    0060  3f800000 0108880d 3e800000 3f000000
    0070  3f400000 3f800000 3f000000 00010002
    0080  00002000 00002001 00000000 00001000
    0090  00000002 00000004 0000000c 00000044

Expect: K = 2; uid table at 0x48; outputs [0x2000 ← node 0, 0x2001 ←
node 1]; inputs [(0x1000, float)] = 0.5; extents (0x1C, 0x30),
(0x30, 0x44).

**T7. Real cooks.** Every graph of the node corpus (`corpus/sbsar`, one
output, `$outputsize` and `$randomseed` only) must read one output (the
XML's output uid) and two inputs. Recooked with `--expose-output-size 0
--expose-random-seed 0` (no inputs, no defaults block), each must read one
output, no inputs, and nodes that end at the uid table.

## 8. Unknown

- **The old layout** (nodes before the node table) was seen in one file
  only; whether old cooks with a defaults block place it as stated is
  **[inferred]**.
- **Version 7** was never seen, nor versions above 9. Image defaults are
  16 bytes from 8 on; for 7 it is unknown.
- What raises the version, beyond the examples in section 0.
- The output map's `format` word, and the header words from 0x14 to 0x37.
- Whether padding can ever appear between the output map and the first
  node (never seen; `K` would then be wrong). A reader can cross-check `K`
  against the XML's output count.
- A cooked multi-graph package (several graphs in one `.sbsasm`, as in
  some cooked library packages) has one uid table for all graphs' inputs
  and outputs; how its nodes split into graphs is not covered here.
