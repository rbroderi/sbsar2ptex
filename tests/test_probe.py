"""The probe's one-node graphs are well-formed and wired as documented."""

import xml.etree.ElementTree as ET

import pytest

from sbsar2ptex.probe import COLOR
from sbsar2ptex.probe import FILTERS
from sbsar2ptex.probe import GRAY
from sbsar2ptex.probe import sbs


@pytest.mark.parametrize("filt", sorted(FILTERS))
@pytest.mark.parametrize("kind", [GRAY, COLOR])
@pytest.mark.parametrize("chain", [False, True])
def test_graph_wiring(filt: str, kind: int, chain: bool) -> None:  # noqa: FBT001
    root = ET.fromstring(sbs(filt, kind, chain=chain))  # noqa: S314 (our own XML)
    nodes = {n.find("uid").get("v"): n for n in root.iter("compNode")}
    probe = nodes["200"]
    assert probe.find(".//filter").get("v") == filt
    refs = [c.find("connRef").get("v") for c in probe.iter("connection")]
    expected = ["100", "110", "120"][: len(FILTERS[filt])]
    if chain and expected:
        expected[0] = "150"
        assert nodes["150"].find(".//filter").get("v") == "blur"
    assert refs == expected
