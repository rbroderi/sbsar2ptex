# sbs plugin

Nodes that match Substance Designer's atomic nodes, for graphs converted by
[sbsar2ptex](https://github.com/rbroderi/sbsar2ptex) `--experimental`.
Generated and installed by `sbsar2ptex --install-mm-plugin DIR` (for Material
Maker: its user folder's `plugins/sbs`).

- The plugin's id is `sbs` (the folder name), and graphs require it with
  `"requires": {"plugins": {"sbs": ">=1.10"}}`.
- A node's *Output format* rounds to 8 or 16 bits as Substance's GPU engine
  stores them, which differs between GPUs: *GPU levels* picks NVIDIA's or
  AMD's (sbsar2ptex sets it).
- Node types are `sbs_<atom>` (for example `sbs_levels`), `.mmg` files in
  `nodes/`, with Substance's parameter names and ranges. `plugin.gd` adds the
  node types implemented by scripts: the FX-Map (Substance), `sbs_fxmap`.

## FX-Map (Substance) node

The **FX-Map (Substance)** node (type `sbs_fxmap`, `gen_sbs_fxmap.gd`) draws
splats of patterns, as Substance Designer's FX-Map node. It is meant for graphs
converted from Substance Designer, which write the code that generates the
splats, so all the FX-Map's parameters and inputs stay live (nothing is baked).
Material Maker's own FX-Map nodes are better suited to new graphs.

Splats are generated on the GPU by the node's *walk code* (GLSL), run once per
*invocation* (for example a single invocation that walks a whole tree of
quadrants with a sequential random generator, or one per independent subtree).
It calls

```glsl
fx_emit(vec2 position, vec2 size, float rotation, vec4 color, int pattern,
        int blend, float opacity, float pattern_parameter)
```

for each splat. `fx_invocation` is the invocation's index, `fx_seed` the node's
seed and `fx_size` the output size in pixels. `$name` in the code is a
parameter of the node (declared in its `fx_parameters`, so they can be edited
and connected to remote parameters) or a named parameter of the graph. The
node's *global code* is added before the walk (for example the FX-Map random
generator).

Positions are in UV space (splats that cross the borders are also drawn on the
other side), sizes are relative to the output, and rotations are in turns.
Patterns 1 to 15 are built-in shapes (square, disc, paraboloid, bell, gaussian,
thorn, pyramid, brick, gradation, waves, half bell, ridged bell, crescent,
capsule and cone), patterns 100 to 103 sample the pattern inputs, and patterns
from 200 call `fx_custom_pattern(pattern, p, parameter)` if the global code
defines `FX_CUSTOM_PATTERNS` (p is the position in the splat, from -0.5 to
0.5). Blending modes are 0 (add), 1 (subtract), 2 (max), 3 (min), 4 (alpha) and
5 (copy).

The splats are then sorted into a grid of *cells*, in the order they were
emitted, and each pixel draws the splats of its cell on the background (the
first input, or the background color).

Inputs:

- the background
- the pattern inputs (0 to 4, see *Pattern inputs*), rendered at the *pattern
  size*

Parameters:

- *size mode*, *size* and *size offset*, as in the Advanced Buffer node, the
  output's *format* and whether it is *grayscale*
- the number of *pattern inputs* and their *pattern size*
- the *background* color, used when the background input is not connected
- the number of *invocations*, the maximum number of *splats per invocation*,
  and the number of *cells* of the grid (per side)
- the parameters of the walk code (`fx_parameters`)
