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
- `--size 11` is the resolution as log2 (11 = 2048 px); by default the material's
  own (its `$outputsize`, usually 8 = 256 px);
- `--embedded` also saves the bitmaps baked into the archive.

Without `sbsrender`, export the maps yourself (Substance Player, Painter or Designer)
and pass `--maps DIR`; files are matched by output or channel name (`*_basecolor.png`,
`*_normal.png`, `*-height.png`, …). `--normal directx` if your normal map is DirectX.

The graph has one image node per map (base color, metallic, roughness, emission,
normal, ambient occlusion, height, opacity) into a Material node. The normal map goes
through Material Maker's `normal_map_convert` node, since the Material node takes
normals in Material Maker's own convention.

### A live, editable graph

```bash
sbsar2ptex Material.sbsar --procedural
```

translates the compiled graph itself into Material Maker nodes instead of
rendering images (see "Translate to Material Maker nodes" below): nothing
is baked except big scattering FX-Maps, node settings are parameters, the
material's inputs are on a "parameters" node. It writes `Material.ptex` and
`Material_data/` (embedded bitmaps, precomputed FX-Map textures). No
`sbsrender` is needed.

The graph comes folded into subgraphs named after the Substance library
nodes found in it. Naming them needs Substance's library cooked: the first
run cooks Designer's `resources/packages` with its `sbscooker` into a cache
(a few minutes, once; LOCALAPPDATA/sbsar2ptex/library on Windows, else
~/.cache/sbsar2ptex/library).

- `--flat` leaves the graph flat (one node per atomic node);
- `--no-library` groups it without naming library nodes (no Designer needed);
- `--library DIR` keeps the cooked library elsewhere, `--sbscooker PATH`
  names the cooker, `--refresh-library` re-reads the cache;
- `--size` is the resolution the graph is built for (FX-Map textures, blur
  widths in texels); the material's own by default, as Substance renders it.
  Material Maker keeps every buffer in GPU memory, so a large graph built at
  2048 px can need more than a graphics card has.

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
(sections 8–9, `sbsar2ptex.fxrng`), including its parameter-specific LCGs and
the Quadrant-slot and Iterate-pass state transforms. Random Iterate cases
match pixel for pixel, and Quadrant-grid size vectors match the render probe.

Each tree node is skipped for pixels it cannot reach (interval bounds on its
offsets and sizes, `rand(a)` within [0, a]); what depends on graph inputs is
never skipped, since they can change.

Given `translate_graph(..., data_dir=...)`, an FX-Map drawing more than 64
patterns is precomputed instead (`sbsar2ptex.fxsplat`): its tree is walked
once, with the graph inputs at their values then, into a list of splats,
packed with a grid of cells into an RGBA8 data texture (one float32 a texel),
and the shader draws only its cell's splats. Such a map no longer follows
its parameters live. In BiomechZombieCircuits that is 16 of 102 FX-Maps, at
about 150 splats per pixel at worst.

`rand` in once-per-node value functions uses Substance's resolved node seed
and value-function LCG.

### Editable nodes

A node's own settings are Material Maker parameters on that node, not
constants in its GLSL: levels' five values, a blend's opacity and mode (a
drop-down), a transformation's matrix and offset, uniform colors, HSL,
grayscale weights, and the intensities and angles of blurs, warps, sharpen,
normal and emboss. Blurs count their taps in the shader, so their intensity
can change, and follow a function exactly too; a blur's two passes are one
subgraph with one intensity control. Settings driven by functions
stay functions. Graph inputs (`p_...`) are exposed together on the remote
"parameters" node; a node's own settings stay on it.

### Structure

A compiled graph is flat: every library node the author placed is inlined
into atomic nodes, thousands of them. `sbsar2ptex.structure.nest` folds a
translated graph back into nested Material Maker subgraphs, closer to how it
was built:

- Library nodes are found by structure (`sbsar2ptex.recognize`): Designer's
  own library, cooked, gives each library output's atomic DAG, matched
  backwards from its output with its inputs as wildcards. Those not matched
  exactly (other library versions, or pruned by their settings) but
  repeated in the graph (the same node placed again) are grouped too, and
  named after the template they resemble ("≈ Slope blur grayscale") or
  "Unit A", "Unit B"...
- The cooker writes a placed node's atomic nodes together, so the compiled
  order is cut into parts and regions where the fewest values cross,
  never inside one of those nodes; groups are named after what they hold.

In BiomechZombieCircuits, the whole material's Material Maker nodes (about
9,000) become seven parts, at most three levels deep; nesting changes no
pixel (a 382-node subgraph exports identically flat and nested).
`sbsar2ptex --procedural` translates and nests it in about 20 seconds with
the library cached.

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
