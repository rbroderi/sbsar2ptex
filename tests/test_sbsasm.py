"""The decompiler, on graphs cooked from sbsasm-probe's one-node graphs.

Each fixture is input A -> (blur ->) one filter -> output, inputs B and C
wired to the filter's other inputs, cooked by Substance Designer 16.0.7.
"""

import re
from pathlib import Path

import pytest

from sbsar2ptex.archive import member
from sbsar2ptex.archive import read_archive
from sbsar2ptex.sbsasm import Asm
from sbsar2ptex.sbsasm import Node
from sbsar2ptex.sbsasm import reachable

DATA = Path(__file__).parent / "data"


def load(name: str) -> tuple[Asm, list[Node], dict[int, str]]:
    files = read_archive(DATA / f"{name}.sbsar")
    xml = member(files, ".xml").decode("utf-8")
    names = {int(u): n for u, n in re.findall(r'uid="(\d+)" identifier="([^"]*)"', xml)}
    asm = Asm(member(files, ".sbsasm"))
    return asm, asm.nodes(), names


def by_input(nodes: list[Node], names: dict[int, str]) -> dict[str, int]:
    """Graph input name -> the node that reads it."""
    return {names[n.uid]: n.index for n in nodes if n.uid is not None}


def only(nodes: list[Node], name: str) -> Node:
    (node,) = [n for n in nodes if n.name.startswith(name)]
    return node


def test_blend_inputs_and_opacity() -> None:
    _, nodes, names = load("blend_gray_chain")
    inputs = by_input(nodes, names)
    blur = only(nodes, "blur")
    blend = only(nodes, "blend")
    assert blur.inputs == {"input": inputs["input_a"]}
    assert blend.inputs == {
        "destination": inputs["input_b"],
        "source": blur.index,
        "opacity": inputs["input_c"],
    }
    assert 0.37 in blend.constants()


@pytest.mark.parametrize(
    ("fixture", "filt", "first", "second"),
    [
        ("warp_gray_chain", "warp", "input", "gradient"),
        ("distance_gray_chain", "distance", "mask", "source"),
        ("shuffle_gray_chain", "shuffle", "input1", "input2"),
    ],
)
def test_two_input_filters(fixture: str, filt: str, first: str, second: str) -> None:
    _, nodes, names = load(fixture)
    inputs = by_input(nodes, names)
    blur = only(nodes, "blur")
    node = only(nodes, filt)
    assert node.inputs == {first: blur.index, second: inputs["input_b"]}


def test_levels_constant() -> None:
    _, nodes, names = load("levels_gray")
    levels = only(nodes, "levels")
    assert levels.inputs == {"input": by_input(nodes, names)["input_a"]}
    assert levels.constants()[0] == pytest.approx(0.1)


def test_uniform_color() -> None:
    _, nodes, _ = load("uniform_color")
    (node,) = nodes
    assert node.name == "uniform"
    assert node.constants() == pytest.approx([0.3, 0.4, 0.5, 1.0])


@pytest.mark.parametrize("fixture", sorted(p.stem for p in DATA.glob("*.sbsar")))
def test_outputs_reach_every_node(fixture: str) -> None:
    asm, nodes, _ = load(fixture)
    roots = [node for _, node in asm.output_nodes]
    assert reachable(nodes, roots) == set(range(len(nodes)))
    assert all(not n.name.startswith("op") for n in nodes)
