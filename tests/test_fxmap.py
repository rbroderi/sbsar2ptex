"""FX-Map settings read from a node's words."""

import pytest

from sbsar2ptex import fxmap
from sbsar2ptex.fxmap import read_settings
from sbsar2ptex.sbsasm import Node


def test_settings_without_a_setup_function() -> None:
    # BiomechZombieCircuits n3998: no setup function (header bit 16 clear),
    # so the inline opacity (1.00343) follows the seed's code at once.
    words = [0x03985508, 0x10, 0xF5BD2C, 0xF5BD3C, 0x3F807065, 0x02000448, 0xF5BD34]
    node = Node(index=3998, start=0xF5BD4C, end=0xF5BD94, op=8, words=words)
    st = read_settings(node, color=False)
    assert st.setup_code is None
    assert st.seed_code == 0xF5BD3C + 0x34
    assert st.opacity == pytest.approx(1.00343, abs=1e-5)


def test_settings_with_a_setup_function() -> None:
    # BiomechZombieCircuits n4004: pattern n4002, seed and setup functions,
    # then the background input n4003 and an inline hurst of 0.
    words = [0x03998809, 0x407, 0xF5BE3C, 0xFA2, 0xF5C0F4, 0xF5C0E0, 0xFA3, 0x0]
    node = Node(index=4004, start=0xF5BE50, end=0xF5C14C, op=9, words=words)
    st = read_settings(node, color=True)
    assert st.patterns == [0xFA2]
    assert st.seed_code == 0xF5C0F4 + 0x34
    assert st.setup_code == 0xF5C0E0 + 0x34
    assert st.background == 0xFA3
    assert st.hurst == 0.0


def test_an_image_id_carries_its_filtering() -> None:
    # image input 3, nearest (filtering 2 in bits 2-3; 0x2: an index function)
    k = -1 - fxmap.image_id(3, 2 << 2 | 0x2)
    assert (k % fxmap.IMAGE_STRIDE, k // fxmap.IMAGE_STRIDE) == (3, fxmap.NEAREST)
    assert fxmap.image_id(3, 0) == -4  # bilinear and mipmaps: the index alone
