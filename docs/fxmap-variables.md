# FX-Map Variables Set as Different Types

## Clean-room behavioral specification

This document specifies what a compiled FX-Map means when one variable
id is set with different types in different functions of the same map,
as in mosaic.sbsar. It is intended to be sufficient for an independent
implementation without access to the original implementation. No engine
code and no file data are reproduced; third-party files are cited by
name and node number only.

It was recovered from:

- the FX-Maps of the 35-archive corpus, decoded;
- probe FX-Maps cooked with Designer's `sbscooker` and rendered with
  `sbsrender`.

Each statement is tagged **[measured]** or **[inferred]**.
`docs/function-ops.md` defines the code blocks and the variable ops:
get variable (0x04) and set variable (0x07). In an FX-Map, the functions
of the tree's nodes share one variable store, in evaluation order: a
value one node's function sets is read by later functions, for instance
its children's.

## 1. The rule

**[measured]** A variable id names storage, not a typed variable. A set
stores its value with the set's own type and width. A get yields the
value of the most recent set of that id in evaluation order.

**[measured]** The cooker guarantees that this most recent set has the
get's type. A probe FX-Map in which a parent quadrant's function sets
`v` and a child quadrant's function reads it:

| set in the parent | read in the child | cook result |
|---|---|---|
| float2 (0.25, 0.75) | get float2 | cooks; renders the child as (0.25, 0.75) |
| float 0.625 | get float | cooks; renders as 0.625 |
| float2 | get float | **refused**: "Variable type mismatch" |
| float | get float2 | **refused**: "Variable type mismatch" |

So no cooked map reads a variable with a type other than the one it was
last set with. The same id may still be set with different types in
different places, when every read follows a set of its own type.

**Consequence for a translator.** **[inferred]** Keying the store by
(id, type and width) instead of by id alone gives the same values as the
engine for every cooked map: each get reads the latest set of its own
key, which is the latest set of its id. Reject a map only if a get's key
is never set before it. Do not reject a map because one id is set with
two types.

## 2. mosaic.sbsar

**[measured]** The FX-Maps n11, n24, n37, n62, n67, n70 and n74 of
mosaic.sbsar share one shape. In each, variables 47, 49 and 51 are set
with two types each, in two parts of the tree:

1. Quadrant A, the repeated node under an Iterate of 2 under a Switch.
   Its opacity function sets 47 and 49 as float2 and 51 as float. Its
   frame offset, pattern size and rotation functions then read 49 and 47
   as float2 and 51 as float.
2. An Iterate B, the sibling run after that Switch. Its count function
   sets 47 and 49 as float and 51 as float2. Its repeated Quadrant C's
   opacity and pattern-supplement functions read 47 and 49 as float.

Each read follows, in evaluation order, a set of its own type (A's in
A's own opacity; C's in B's count).

**Corpus.** **[measured]** Of the 5,679 readable FX-Maps in the 35
archives (32 more could not be parsed), 98 set some id with two types:
mosaic 7, Manhole_Cover 23, OvergrownRock 6, RiberStones 34 and
Rock_Chunky_Desert_Cliff 28. Most reuse id 0 as float2 and float4, or as
float and float4. A type-flow check followed every map in evaluation
order: the setup function, then the tree depth-first, with a quadrant's
parameter functions in order before its children, an Iterate's count
before its first and repeated branches (the repeated branch looping),
and both Switch branches merged. It found **no** get whose latest set,
on any path, has another type, and no get of an id never set.
**[inferred]** The other files convert today only because their maps are not translated through the
check that raises the error.

## 3. Acceptance tests

Synthetic FX-Map functions run in tree order, with one shared store.

1. **Same id, two types, separate paths.** Function P sets v7 to float2
   (0.25, 0.75); its child's function reads v7 as float2 and yields
   (0.25, 0.75). A later sibling's function S sets v7 to float 0.5; its
   child reads v7 as float and yields 0.5. Translating the map must not
   raise an error.
2. **Overwrite.** As test 1, but run S before P: the float2 read still
   yields (0.25, 0.75) and the float read 0.5.
3. **mosaic shape.** Quadrant A's opacity sets v47 float2 (1, 2) and v51
   float 3; its pattern size reads v47 float2 = (1, 2); its rotation reads
   v51 = 3. Then B's count sets v47 float 4 and v51 float2 (5, 6); C's
   opacity reads v47 float = 4.
4. **Unset read.** A get whose key is never set before it is reported,
   as for any variable (`docs/function-ops.md`).

## 4. Unknown

- Whether the engine stores a set's full width or reinterprets on read.
  It cannot be observed, since the cooker refuses mismatched reads.
- Whether the cooker's check covers a read that, along one evaluation
  path, could see a set of another type made by an earlier sibling's
  subtree. Not probed: only parent-to-child paths were.
