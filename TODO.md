# To do

- **Emulate Substance's float32 FX-Map vertex arithmetic.** In the splat
  drawing (fxsplat/fxdraw and the Splats node), round each copy's centre
  plus tile offset to float32 in region units. Then compute the triangle
  corners, snapping and pattern coordinate in float32, as the engine's
  vertex stage does (docs/fxmap-splats.md, section 2).
  - Biomech's n2369 should go from 1.52% to about 1.26% of pixels off by
    more than half a 16-bit level. The study agent measured that with a
    scratch emulation (vsemu.py).
  - Putting the vertex data in render-region UV units instead gave only
    1.44%, so keep region units.

- **Speed up FX-map splat evaluation.** brick_wall.sbsar (community file)
  times out after 20 minutes. It isn't stuck and it isn't a format
  problem: the splats are evaluated slowly in Python.
  - One FX-map node, n1321 (depth 9), has 348,160 splats and took 30.8 s.
  - Over the whole graph, splat evaluation took 702 s across 89 calls.
  - The profile points at function.py's per-instruction interpreter
    (`f32` rounding) running once per splat.
  - Possible fixes: vectorize the splat functions over all splats at once
    (numpy), or cache results that don't depend on the splat index.
  - Keep the output bit-identical. The existing tests and Biomech's FX-map
    nodes must match before and after.

- **Enlarge smaller nodes as Substance does: chained 2x bilinear steps.**
  Where a larger node reads a smaller one, translate_graph's `_bspline`
  reads it through a cubic B-spline. Substance's engine instead enlarges
  by repeated 2x bilinear steps, one per size doubling
  (docs/resolution.md).
  - Measured on noise: one step matches plain bilinear (error 0.0010).
    Two steps match two chained 2x bilinear passes (0.0010), not one 4x
    bilinear read (0.18) and not the B-spline (0.12 to 0.17).
  - Each step is a buffer at the next size, read bilinearly. Keep it live:
    relative sizes (`buffer`), widths from `px_glsl`.
  - Enlargements of 3 or more doublings were not measured; check one
    with check_filters before relying on it.
  - Clean-room: a fresh implementer works from docs/resolution.md alone,
    not the study's scripts or decompiled shaders.
