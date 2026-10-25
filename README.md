# sbsar2ptex

Tools for Substance archives (`.sbsar`):

- **`sbsar2ptex`** turns an `.sbsar` into a [Material Maker](https://www.materialmaker.org/)
  graph (`.ptex`): it renders the material's maps with Adobe's `sbsrender` (or takes
  maps you exported) and wires them into a Material node.
- **`sbsasm-dump`** decompiles the compiled graph inside an `.sbsar` (the `.sbsasm`)
  into its atomic nodes — blend, levels, warp, FX-Map, pixel processor, … — with
  their connections, constants and the exposed parameters each reads.
- **`sbsasm-probe`** is the research tool behind the dumper: it writes one-node
  graphs, cooks them with Substance Designer's `sbscooker`, and shows how each atomic
  filter compiles.

An `.sbsar` holds a *compiled* graph, so a node-for-node translation into Material
Maker's own nodes isn't possible: Material Maker has no equivalents for Substance's
atomic filters, and the cooker rewrites graphs as it compiles them. `sbsar2ptex`
therefore bakes the maps; `sbsasm-dump` shows you what the graph actually does.

## Install

```bash
uv tool install git+https://github.com/rbroderi/sbsar2ptex
```

or `pip install git+https://github.com/rbroderi/sbsar2ptex`.

## Convert to Material Maker

```bash
sbsar2ptex Material.sbsar
```

This writes `Material.ptex` and `Material_maps/` beside the archive (`-o DIR` puts them
elsewhere). It needs `sbsrender`, which ships with **Substance 3D Designer** and the
**Substance 3D Automation Toolkit**; it is found on `PATH` or in the usual install
folders, or give `--sbsrender PATH`.

- `--list` prints the material's outputs and parameters;
- `--set NAME=VALUE` changes a parameter before rendering (repeatable);
- `--size 11` is the resolution as log2 (11 = 2048 px);
- `--embedded` also saves the bitmaps baked into the archive.

Without `sbsrender`, export the maps yourself (Substance Player, Painter or Designer)
and pass `--maps DIR`; files are matched by output or channel name (`*_basecolor.png`,
`*_normal.png`, `*-height.png`, …). `--normal directx` if your normal map is DirectX.

The graph has one image node per map (base color, metallic, roughness, emission,
normal, ambient occlusion, height, opacity) into a Material node. The normal map goes
through Material Maker's `normal_map_convert` node, since the Material node takes
normals in Material Maker's own convention.

## Decompile

```bash
sbsasm-dump Material.sbsar --nodes 0        # every node
sbsasm-dump Material.sbsar -o out/          # also nodes.txt, graph.dot, bitmaps
```

```
outputs:
  height               <- n3404 levels (gray)  (format 0x101c0)
  ...
nodes:
  n3404: levels (gray)  input=n3403  reads ['Basic_Parameters_Height_Shift']
  n6314: shuffle  input1=n6313, input2=n6293
  n1500: fxmaps (gray)  pattern0=n1497, pattern1=n1499, ...
```

The format notes — container, node layout, the opcode table and each op's fields — are
in the docstrings of [`sbsasm.py`](src/sbsar2ptex/sbsasm.py). They were worked out by
cooking known graphs with Designer 16's `sbscooker` (`sbsasm-probe`) and comparing, and
are checked against over a hundred cooked graphs, where every node is decoded and
reached from the outputs.

Not decoded yet: what the function bytecode computes (parameters driven by functions
are listed as "reads", not evaluated), value processors' links to the nodes that use
their values, and the meaning of most flag bits.

## Translate to Material Maker nodes (in progress)

`sbsar2ptex.translate` turns decompiled nodes into Material Maker shader
nodes whose GLSL reproduces each Substance filter, wired as the compiled graph
is; parameters driven by Substance functions become GLSL too, and the graph
inputs they read become live parameters on a "parameters" remote node.

Translated, each within 3/255 of `sbsrender` (99th percentile) in
`scripts/check_filters.py`: uniform color, levels, curve, HSL, grayscale
conversion, channel shuffle, gradient map, dynamic gradient, all twelve blend
modes, transformation, blur, directional blur, sharpen, warp, directional
warp, emboss, normal, distance (jump flooding) and pixel processors.

Pixel processors use Substance's own random generator (`rand`), implemented
clean-room from a behavioral specification ([docs/random-generator.md](docs/random-generator.md))
and identical to it bit for bit on every test render.

FX-Maps (`sbsar2ptex.fxmap`) are evaluated per pixel: the compiled Quadrant,
Iterate and Switch tree becomes GLSL functions, and the patterns, placement,
depth weighting (hurst), blending, image patterns and function-driven
parameters ($pos, $number, $depth) match `sbsrender` within 2/255 in the
harness. Their `rand` follows the FX-Map part of the specification
(sections 8–9, `sbsar2ptex.fxrng`). Renders match its first call from the
root state, but not its second call within a function or its Quadrant-slot
and Iterate-pass states yet, so random FX-Maps have the right distribution,
not the same values. Every pattern is visited for every pixel, so large
trees are slow. The ridged bell and crescent patterns are drawn as a bell.

`rand` in value functions is still approximated by a hash with the same
distribution, not the same values. Gradient maps with
midpoint interpolation, and RGB beyond a color distance's range (where alpha
is 0), are not reproduced.

## Develop

```bash
uv sync
uv run pytest
uv run ruff check
```

Substance, Substance 3D Designer and Substance 3D Painter are trademarks of Adobe. This
project is not affiliated with Adobe; it reads the format for interoperability.

## License

GPL-3.0-or-later.
