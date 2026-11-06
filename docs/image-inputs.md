# Embedded Image Inputs (sbsasm ops 32 and 33)

## Clean-room behavioral specification

This document specifies how a compiled image-input node finds its
embedded image: a raw pixel array or an encoded file (JPEG) stored in the
`.sbsasm` before the nodes. It explains the Tarp_Floor failure "no image
for graph input 262148". It is intended to be sufficient for an
independent implementation without access to the original
implementation. No engine code and no file data are reproduced;
third-party files are cited by name and node number only.

It was recovered from every op-32/33 node in the 35-archive corpus (46
nodes) and the files' own `.xml` descriptions. Each statement is tagged
**[measured]** or **[inferred]**. `docs/node-seed.md` section 0 defines
offsets (from byte 0x34) and the node word.

## 1. What the node holds

**[measured]** An image-input node is the node word, then one word, its
`uid`. The node word's byte `x` is the image size, `log2 height << 4 |
log2 width`. The node word's top byte (`flags >> 8`) is the image's
**storage format**:

| format | storage | at offset `uid` |
|---|---|---|
| 1 | raw, 1 channel, 8 bits | the pixels |
| 2 | raw, 3 channels, 8 bits | the pixels |
| 3 | raw, 4 channels, 8 bits | the pixels |
| 5 | raw, 1 channel, 16 bits | the pixels |
| 6 | raw, 3 channels, 16 bits | the pixels |
| 7 | raw, 4 channels, 16 bits | the pixels |
| 8 | an encoded file | a u32 byte length, then the file's bytes |

When `uid` is one of the graph's declared inputs (the uid table / the
`.xml` `<input>` list), the node is a real graph input instead; no node in
the corpus is.

A raw image has no length word: its size is `width × height × channels ×
bytes per channel`, from `x` and the format. Every format-8 file in the
corpus is a JPEG (it starts with FF D8 FF), and its own dimensions equal
`x`'s.

**The data area.** **[measured]** The blobs lie from byte 0x38 up to the
first node, back to back, each starting on a 4-byte boundary. Raw and
encoded blobs can be mixed. A reader cannot walk the area from 0x38 by
"length word, then a known file header", because a raw blob has neither.
It must resolve each node's image **by its uid**.

## 2. Tarp_Floor

**[measured]** Tarp_Floor (cook version 9) declares only two graph
inputs, `$outputsize` and `$randomseed`. It has three image-input nodes:

| node | format | `x` | uid | data |
|---|---|---|---|---|
| n909 | 1 (gray, 8 bits) | 0x99 (512 × 512) | 4 | 262,144 raw bytes from 0x38 |
| n1006 | 8 | 0xBB (2048 × 2048) | 262148 (0x40004) | length 1,007,873, a grayscale JPEG |
| n1008 | 8 | 0xBB (2048 × 2048) | 1270028 (0x13610C) | length 804,640, a grayscale JPEG |

262148 = 4 + 262144: n1006's blob starts right after n909's raw pixels.
A scan from 0x38 that expects a length word and a file header stops at
the raw image and never finds the two JPEGs. So "graph input 262148" is
not a graph input. It is n1006's embedded JPEG.

**Corpus.** Of the 30 format-8 nodes, 26 have their blob in an unbroken
run of encoded files from 0x38, and 4 do not: Tarp_Floor n1006 and n1008, and
Manhole_Cover n2183 and n2185. Resolving by uid finds all 30. The
16 raw nodes (formats 1, 2, 3, 5, 6, 7) are found by uid as before.

## 3. How to handle it

1. For an op-32/33 node whose uid is not a declared graph input, read
   the format byte.
2. Formats 1-7: read the raw pixels at offset `uid`, sized from `x`.
3. Format 8: read the u32 length at offset `uid`; the file is the next
   `length` bytes. Check that it ends before the node's start and that it
   starts with a known image header; else report it as malformed.
4. Any other format: report as unsupported.

The image is the node's output. 16-bit raw data and grayscale JPEGs keep
their channel count.

## 4. Acceptance tests

A synthetic `.sbsasm` with a data area from 0x38:

1. **Raw then JPEG.** A 4 × 4 gray 8-bit raw image (16 bytes) at 0x38,
   then at 0x48 a length word L and an L-byte JPEG, padded to 4 bytes.
   Node A: format 1, `x` = 0x22, uid 4. Node B: format 8, `x` = 0x22,
   uid 0x14. B's image is the JPEG; translation must not report "no
   image for graph input 20".
2. **JPEG first.** The same blobs in the other order: A's uid points
   after the padded JPEG. Both resolve.
3. **Bad length.** Format 8 with a length running past the node start:
   reported as malformed.
4. **Unknown format.** Format 4: reported as unsupported.

## 5. Unknown

- Formats 0 and 4, and 9 and up: never seen.
- Encoded formats other than JPEG (PNG, DDS, TIFF) under format 8: none
  in the corpus. **[inferred]** They use the same length-then-bytes
  layout.
- Whether the alignment padding after a blob is always to 4 bytes is
  inferred from the offsets seen, all multiples of 4.
