"""Substance archives (.sbsar) to Material Maker graphs (.ptex), and back to nodes.

- ``sbsar2ptex`` (sbsar2ptex.convert): render an .sbsar's maps and wire them
  into a Material Maker graph;
- ``sbsasm-dump`` (sbsar2ptex.sbsasm): decompile an .sbsar's compiled graph
  to its atomic nodes and connections;
- ``sbsasm-probe`` (sbsar2ptex.probe): cook one-node graphs with Substance
  Designer's sbscooker to learn how each atomic filter compiles.
"""
