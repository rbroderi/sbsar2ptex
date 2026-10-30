"""The FX-Map drawing model (fxdraw): the specification's unit checks."""

import math
from fractions import Fraction

import numpy as np
import pytest

from sbsar2ptex import fxdraw
from sbsar2ptex import fxmap
from sbsar2ptex import fxsplat
from sbsar2ptex.fxdraw import Instance


def square(value: float, *, blend_max: bool = False, size: float = 1.0) -> Instance:
    """Return a full-cell square (size 1: the whole image) drawing ``value``."""
    return Instance(0.5, 0.5, size / 2, size / 2, 0.0, 0, 0.0, blend_max, [value])


def gray(value: float) -> float:
    """Return an instance's gray color as fxsplat makes it: float32."""
    return fxsplat.instance_color([value], 1.0, 1.0, color=False)[0]


def test_the_accumulator_is_float32_unclamped() -> None:
    # 100 squares of 0.3/65535 make 30 levels; 1000 of 0.01/65535 over 0.5
    # make 32779 (each float32 add rounds up to 3 ulps)
    many = [square(gray(0.3 / 65535)) for _ in range(100)]
    assert set(fxdraw.render(many, 16, color=False, amd=True).flat) == {30}
    more = [square(gray(0.5)), *(square(gray(0.01 / 65535)) for _ in range(1000))]
    assert set(fxdraw.render(more, 16, color=False, amd=True).flat) == {32779}
    # no intermediate clamp: 1.4 then -0.6 is 0.8
    over = [square(gray(1.4)), square(gray(-0.6))]
    assert set(fxdraw.render(over, 4, color=False, amd=True).flat) == {52428}


def test_a_gray_opacity() -> None:
    # inline: the mean of rgb (the cooker stores it); a function's float4: w
    inline = fxsplat.instance_color([0.4333333] * 4, 1.0, 1.0, color=False)
    assert fxdraw.u16(np.float32(inline[0])) == 28398
    func = fxsplat.instance_color([0.9, 0.1, 0.3, 0.7], 1.0, 1.0, color=False)
    assert fxdraw.u16(np.float32(func[0])) == 45874


def test_max_draws_after_adds() -> None:
    tree = [square(gray(0.1), blend_max=True), square(gray(0.2)), square(gray(0.3))]
    assert set(fxdraw.render(tree, 4, color=False, amd=True).flat) == {32768}


def test_the_order_is_by_pattern_then_blending() -> None:
    def inst(pattern: int, *, blend_max: bool = False) -> Instance:
        return Instance(0.5, 0.5, 0.1, 0.1, 0.0, pattern, 0.0, blend_max, [1.0])

    image = fxmap.image_id(0, 0)
    tree = [inst(image), inst(16), inst(15), inst(1), inst(0)]
    assert [tree[k].pattern for k in fxdraw.order(tree)] == [0, 1, 15, 16, image]
    # the type outranks the blending
    tree = [inst(1), inst(0, blend_max=True)]
    assert fxdraw.order(tree) == [1, 0]
    # images by input, the inputs in the reverse order of first appearance
    red, blue = fxmap.image_id(0, 0), fxmap.image_id(1, 0)
    tree = [inst(red), inst(blue), inst(red)]
    assert fxdraw.order(tree) == [1, 0, 2]


def test_a_color_map_blends() -> None:
    def color(rgba: tuple, *, blend_max: bool) -> Instance:
        a = rgba[3]
        value = [rgba[0] * a, rgba[1] * a, rgba[2] * a, a]
        return Instance(0.5, 0.5, 0.5, 0.5, 0.0, 0, 0.0, blend_max, value)

    a, b = (0.8, 0.4, 0.2, 0.5), (0.1, 0.6, 0.9, 0.7)
    adds = fxdraw.render(
        [color(a, blend_max=False), color(b, blend_max=False)], 4, color=True, bits=None
    )
    assert adds[0, 0] == pytest.approx([0.47, 0.62, 0.73, 0.85], abs=1e-6)
    over = fxdraw.render(
        [color(a, blend_max=True), color(b, blend_max=True)], 4, color=True, bits=None
    )
    assert over[0, 0] == pytest.approx([0.19, 0.48, 0.66, 0.85], abs=1e-6)


def test_stores() -> None:
    # 8 bits on AMD: through a half float's 11 bits
    assert fxdraw.s8(np.float32((128 + 0.61) / 255)) == 128
    assert fxdraw.s8(np.float32((128 + 0.63) / 255)) == 129
    assert fxdraw.s8n(np.float32((128 + 0.4) / 255)) == 128
    assert fxdraw.u16(np.float32(0.5)) == 32768
    assert fxdraw.s16n(np.float32(1.0)) == 65535
    # NVIDIA puts a half down (RTX 3090): 0.5 is 32767.5 levels
    assert fxdraw.s16n(np.float32(0.5)) == 32767
    assert fxdraw.s16n(np.float32(0.5000008)) == 32767


def test_waves_is_a_polynomial_cosine() -> None:
    f = np.linspace(-3, 3, 10001)
    err = np.abs(fxdraw.wave(f) - np.cos(2 * np.pi * f))
    assert 5e-4 < err.max() < 9.4e-4


def test_patterns_cover_their_square_or_disc() -> None:
    q = np.array([0.9, 0.0, 1.0])
    corner = np.array([0.9, 0.0, 0.0])
    for pid in range(17):
        inside, _ = fxdraw.value(pid, q, corner, 0.3)
        assert list(inside) == [pid in fxmap.SQUARE, True, True]


def test_gaussian_reaches_zero_at_its_edge() -> None:
    _, v = fxdraw.value(4, np.array([1.0]), np.array([0.0]), 0.0)
    assert v[0] < 0.5 / 65535
    _, v = fxdraw.value(4, np.array([0.0]), np.array([0.0]), 0.0)
    assert v[0] == pytest.approx(1 - 1.51e-5)


def test_tiling_copies_overlap_the_tiling_region() -> None:
    region = (0.25, 0.75, 0.25, 0.75)

    def at(x: float, r: float = 0.05) -> list[float]:
        disc = Instance(x, 0.5, r, r, 0.0, 1, 0.0, blend_max=False, value=[1.0])
        return sorted(
            round(x + ox, 6) for ox, _ in fxdraw.copies(disc, region, (True, True))
        )

    assert at(0.74) == [0.24, 0.74]
    assert at(0.1) == [0.6]
    assert at(0.2) == [0.7]  # its box ends at 0.25: touching is not overlapping


def test_the_top_left_edge_is_not_drawn() -> None:
    # edges on pixel centers 64 and 192 (of 256): columns 65 to 192
    edge = Instance(
        0.5 + 0.5 / 256, 0.5, 0.25, 0.6, 0.0, 0, 0.0, blend_max=False, value=[1.0]
    )
    row = fxdraw.render([edge], 256, color=False, tile=(False, False))[128]
    assert list(np.flatnonzero(row)) == list(range(65, 193))


def test_the_mip_tie_goes_as_float32_rounds() -> None:
    rng = np.random.default_rng(1)
    q = rng.integers(0, 65535, 2000)
    up = fxdraw._rounded_up(2 * q + 1, 65535)
    exact = [Fraction(2 * int(k) + 1, 131070) for k in q]
    f32 = [Fraction(float(np.float32((int(k) + 0.5) / 65535))) for k in q]
    assert list(up) == [f > e for f, e in zip(f32, exact, strict=True)]
    assert 0.3 < up.mean() < 0.7


def test_nearest_rounds_the_texel_coordinate_to_a_256th() -> None:
    # within 1/512 below a texel reads it; at the image's edge, the border
    t = np.array([511.5, 511.997, 511.9985, 512.0, 1023.999, 1023.9975])
    assert list(fxdraw.nearest(t, 256)) == [511, 511, 512, 512, 1024, 1023]


def test_a_mip_level_is_drawn_at_its_size() -> None:
    # an image drawn at size 2^-L, centered, shows mip level L at texel centers
    n = 64
    rng = np.random.default_rng(2)
    image = rng.integers(0, 65536, (n, n))
    chain = fxdraw.mip_chain(image, 16, amd=True)
    for lv in range(1, 4):
        h = 0.5 / 2**lv
        pid = fxmap.image_id(0, 0)
        inst = Instance(0.5, 0.5, h, h, 0.0, pid, 0.0, blend_max=False, value=[1.0])
        assert fxdraw.level(h, h, 0.0, n, 1 / n, 1 / n, amd=True) == lv
        got = fxdraw.render([inst], n, color=False, images=[image], amd=True)
        side = n >> lv
        lo = n // 2 - side // 2
        block = got[lo : lo + side, lo : lo + side]
        assert np.array_equal(block, chain[lv])


def test_the_level_follows_the_smaller_side() -> None:
    # (0.6, 0.3) and (0.37, 0.29) of a 256 image read level 2
    assert fxdraw.level(0.3, 0.15, 0.0, 256, 1 / 256, 1 / 256) == 2
    assert fxdraw.level(0.185, 0.145, 0.0, 256, 1 / 256, 1 / 256) == 2
    # a turn does not change an isotropic pattern's level
    turn = 2 * math.pi / 8
    assert fxdraw.level(0.3, 0.3, turn, 256, 1 / 256, 1 / 256) == 1
    assert fxdraw.level(0.375, 0.375, turn, 256, 1 / 256, 1 / 256) == 0
    # the switch from level 0 to 1: on AMD lambda 0.5 - 2/1024 stays at 0
    # (rounded to 1/256, ties down); on NVIDIA it is at 0.5 exactly
    for k, amd, want in ((-2, True, 0), (-1, True, 1), (-1, False, 0), (1, False, 1)):
        h = 0.5 * 2 ** -(0.5 + k / 1024)
        assert fxdraw.level(h, h, 0.0, 256, 1 / 256, 1 / 256, amd=amd) == want


def test_eight_bit_mip_means() -> None:
    # stored 2x2 means: through s8 (AMD), s8n (NVIDIA)
    t = np.array([[0, 1], [1, 1]])
    s = int(t.sum())
    assert fxdraw.mip_mean(t, 8, amd=True)[0, 0] == fxdraw.s8(np.float32(s / 1020))
    assert fxdraw.mip_mean(t, 8)[0, 0] == fxdraw.s8n(np.float32(s / 1020))
    # NVIDIA adds 0.03 of a 1/4096 step before truncating (RTX 3090): the
    # sum 62 is 248.97 steps, so 249, a level more than s8n's 15
    t = np.array([[15, 15], [16, 16]])
    assert fxdraw.s8n(np.float32(62 / 1020)) == 15
    assert fxdraw.mip_mean(t, 8)[0, 0] == 16
