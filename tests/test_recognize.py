"""Library templates matched against graph nodes, on synthetic graphs."""

from sbsar2ptex.recognize import Template
from sbsar2ptex.recognize import heads
from sbsar2ptex.recognize import match_at
from sbsar2ptex.sbsasm import Node

UNWIRED = 0xFFFFFFFF
"""An input no node feeds: an index past the graph's nodes."""


def blend(k: int, source: int) -> Node:
    return Node(index=k, start=4 * k, end=4 * k + 4, op=2, fields=[0],
                inputs={"destination": 0, "source": source})  # fmt: skip


def graph(source: int) -> list[Node]:
    return [Node(index=0, start=0, end=4, op=32), blend(1, source)]


def test_unwired_template_input_matches_only_unwired() -> None:
    # Community .sbsar files (stripe-new-1, mosaic) have library graphs with
    # inputs left unwired; matching one indexed past the template's nodes.
    t = Template("lib", "pkg://lib", "out", graph(UNWIRED), 1, members=[1])
    assert heads(t, 1) == [1]
    m = match_at(t, graph(UNWIRED), 1, {})
    assert m is not None
    assert m.members == {1: 1}
    assert match_at(t, graph(0), 1, {}) is None
