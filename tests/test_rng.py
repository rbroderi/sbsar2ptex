"""Conformance tests for the pixel-processor random generator."""

import pytest

from sbsar2ptex.rng import RNG_GLSL
from sbsar2ptex.rng import PixelRng
from sbsar2ptex.rng import float32
from sbsar2ptex.rng import float_bits
from sbsar2ptex.rng import initial_state

# (x, y, width, height, seed, x bits, y bits, initial state,
#  [(state, unit value, u16), ...]) from section 6 of docs/random-generator.md.
VECTORS = [
    (
        0, 0, 1, 1, 0, 0x3F000000, 0x3F000000, 0x0A1140EE,
        [
            (0x002DA09B, 0.1782318353652954, 11680),
            (0x4066787A, 0.4002758264541626, 26232),
            (0x56C601DF, 0.7734659910202026, 50689),
        ],
    ),
    (
        0, 0, 16, 16, 0, 0x3D000000, 0x3D000000, 0x30B00BCA,
        [
            (0x2EFB254F, 0.9810379743576050, 64292),
            (0xB88939F6, 0.5360406637191772, 35129),
            (0xC42DBD73, 0.1786719560623169, 11709),
        ],
    ),
    (
        1, 0, 16, 16, 0, 0x3DC00000, 0x3D000000, 0x83D861D7,
        [
            (0x4050654E, 0.3140457868576050, 20581),
            (0x26B7A2BB, 0.7173267602920532, 47010),
            (0x9DCDD5DA, 0.8040443658828735, 52693),
        ],
    ),
    (
        0, 0, 16, 16, 1, 0x3D000000, 0x3D000000, 0xD013D7A6,
        [
            (0x5D86E503, 0.5269318819046021, 34532),
            (0x74909472, 0.5647650957107544, 37012),
            (0x6643BC07, 0.2645877599716187, 17340),
        ],
    ),
]  # fmt: skip


@pytest.mark.parametrize("vector", VECTORS)
def test_position_bits(vector):
    x, y, width, height, _, x_bits, y_bits, *_ = vector
    assert float_bits((x + 0.5) / width) == x_bits
    assert float_bits((y + 0.5) / height) == y_bits


@pytest.mark.parametrize("vector", VECTORS)
def test_initial_state(vector):
    x, y, width, height, seed, x_bits, y_bits, state, _ = vector
    assert initial_state(seed, x_bits, y_bits) == state
    assert PixelRng.for_pixel(x, y, width, height, seed).state == state


@pytest.mark.parametrize("vector", VECTORS)
def test_calls(vector):
    x, y, width, height, seed, *_, calls = vector
    rng = PixelRng.for_pixel(x, y, width, height, seed)
    for state, unit, u16 in calls:
        value = rng.rand(1.0)
        assert rng.state == state
        assert value == pytest.approx(unit, abs=1e-15)
        assert value == float32(unit)
        assert round(value * 65535) == u16


def test_only_low_seed_bits_matter():
    assert initial_state(1 << 32, 0x3F000000, 0x3F000000) == 0x0A1140EE


def test_amplitude_scales_in_binary32():
    plain = PixelRng.for_pixel(0, 0, 16, 16, 0)
    scaled = PixelRng.for_pixel(0, 0, 16, 16, 0)
    for _ in range(3):
        unit = plain.rand(1.0)
        assert scaled.rand(0.3) == float32(unit * float32(0.3))
    negative = PixelRng.for_pixel(0, 0, 16, 16, 0)
    assert negative.rand(-2.0) == pytest.approx(-2 * 0.9810379743576050)


def test_zero_amplitude_still_advances():
    rng = PixelRng.for_pixel(0, 0, 1, 1, 0)
    assert rng.rand(0.0) == 0.0
    assert rng.state == 0x002DA09B
    assert rng.rand(1.0) == pytest.approx(0.4002758264541626, abs=1e-15)


def test_pixels_are_independent():
    a = PixelRng.for_pixel(0, 0, 16, 16, 0)
    b = PixelRng.for_pixel(1, 0, 16, 16, 0)
    for _ in range(5):
        a.rand(1.0)
    assert b.rand(1.0) == pytest.approx(0.3140457868576050, abs=1e-15)
    assert b.state == 0x4050654E
    fresh = PixelRng.for_pixel(0, 0, 16, 16, 0)
    assert fresh.rand(1.0) == pytest.approx(0.9810379743576050, abs=1e-15)


def test_glsl_helpers_declared():
    assert "uint sbs_rng_init(uint seed, vec2 position)" in RNG_GLSL
    assert "float sbs_rng_next(inout uint state, float amplitude)" in RNG_GLSL
    assert "floatBitsToUint" in RNG_GLSL
    assert "uintBitsToFloat" in RNG_GLSL
    assert "$(name)" not in RNG_GLSL
    assert "0x4937573Bu" in RNG_GLSL
    assert "0x0000C7C1u" in RNG_GLSL
