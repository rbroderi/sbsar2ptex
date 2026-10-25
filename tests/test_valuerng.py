"""Conformance tests for ordinary-filter value-function randomness."""

import pytest

from sbsar2ptex.valuerng import VALUE_LCG_C
from sbsar2ptex.valuerng import VALUE_LCG_M
from sbsar2ptex.valuerng import VALUE_RNG_GLSL
from sbsar2ptex.valuerng import ValueRng


@pytest.mark.parametrize(
    ("seed", "vectors"),
    [
        (
            0,
            [
                (0x2AA01D31, 0.6254453659057617, 40989),
                (0x29E9568E, 0.9114769697189331, 59734),
                (0xD551A8A7, 0.3189796209335327, 20904),
            ],
        ),
        (
            1,
            [
                (0xAA98C11E, 0.5966967344284058, 39105),
                (0x5410FFF7, 0.06640565395355225, 4352),
                (0x519F59DC, 0.6224648952484131, 40793),
            ],
        ),
        (
            5,
            [
                (0xAA7B50D2, 0.48170197010040283, 31568),
                (0xFCAFA59B, 0.6861206293106079, 44965),
                (0x42D61EB0, 0.8364057540893555, 54814),
            ],
        ),
    ],
)
def test_rendered_vectors(seed, vectors):
    rng = ValueRng(seed)
    for state, value, u16 in vectors:
        assert rng.rand(1.0) == value
        assert rng.state == state
        assert round(value * 65535) == u16


def test_glsl_constants():
    assert f"0x{VALUE_LCG_M:08X}u + 0x{VALUE_LCG_C:08X}u" in VALUE_RNG_GLSL
