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
