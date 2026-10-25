"""Conformance tests for the FX-Map random generator."""

import pytest

from sbsar2ptex.fxrng import DEPTH_C
from sbsar2ptex.fxrng import DEPTH_M
from sbsar2ptex.fxrng import FX_LCG_C
from sbsar2ptex.fxrng import FX_LCG_M
from sbsar2ptex.fxrng import FX_RNG_GLSL
from sbsar2ptex.fxrng import FxRng
from sbsar2ptex.fxrng import depth_step
from sbsar2ptex.fxrng import effective_state
from sbsar2ptex.fxrng import iterate_pass_states
from sbsar2ptex.fxrng import iterate_step
from sbsar2ptex.fxrng import quadrant_child_states
from sbsar2ptex.rng import float32

# First three unit-amplitude leaf calls of section 9.1: (state, unit, u16).
LEAF_CALLS = [
    (0x45CCE379, 0.8003458976745605, 52451),
    (0x979B784C, 0.6073043346405029, 39800),
    (0x23D9675F, 0.8492335081100464, 55655),
]

# Section 9.2: (child effective, state after first rand, unit, u16) per pass.
PASSES = [
    (0xE51EFEE8, 0x79EF12FB, 0.9338833093643188, 61202),
    (0xFB74C0D5, 0x2F2D8AA8, 0.17789697647094727, 11658),
]


def test_root_state_zero_first_call():
    rng = FxRng(effective_state(0, 0))
    value = rng.rand(1.0)
    assert rng.state == 0x0000C613
    assert value == 0.003022313117980957


def test_rendered_consecutive_calls_from_zero():
    rng = FxRng(0)
    assert [rng.rand(1.0) for _ in range(3)] == [
        0.003022313117980957,
        0.6204589605331421,
        0.0581202507019043,
    ]


def test_rendered_iterate_pass_offsets():
    offsets = [
        FxRng(state).rand(0.8) - float32(0.4) for state in iterate_pass_states(0, 5)
    ]
    assert offsets == pytest.approx(
        [0.36785391, -0.01405391, -0.10554609, 0.19917861, -0.06880969],
        abs=1e-8,
    )


def test_rendered_quadrant_slot_offsets():
    offsets = [FxRng(state).rand(0.25) for state in quadrant_child_states(0, 0)]
    assert offsets == pytest.approx(
        [0.08837089, 0.07705319, 0.16057569, 0.03403562],
        abs=1e-8,
    )


def test_table_shape():
    assert len(DEPTH_M) == len(DEPTH_C) == 16
    assert depth_step(0, 16) == depth_step(0, 0) == DEPTH_C[0]
    assert depth_step(1, 17) == (DEPTH_M[1] + DEPTH_C[1]) & 0xFFFFFFFF


def _nested_quadrant_states():
    root = effective_state(11, 3)
    middle = effective_state(quadrant_child_states(root, 0)[2], 5)
    leaf = effective_state(quadrant_child_states(middle, 1)[1], 7)
    return root, middle, leaf


def test_nested_quadrant_states():
    root, middle, leaf = _nested_quadrant_states()
    assert root == 0x0000000E
    tripled = depth_step(depth_step(depth_step(root, 1), 1), 1)
    assert middle == effective_state(tripled, 5)
    assert middle == 0x91F05976
    assert leaf == 0xBEB0F8E6


def test_nested_quadrant_leaf_calls():
    *_, leaf = _nested_quadrant_states()
    rng = FxRng(leaf)
    for state, unit, u16 in LEAF_CALLS:
        value = rng.rand(1.0)
        assert rng.state == state
        assert value == pytest.approx(unit, abs=1e-15)
        assert value == float32(unit)
        assert round(value * 65535) == u16


def test_quadrant_slots_chain_transforms():
    states = quadrant_child_states(0x1234, 4)
    spawn = 0x1234
    for state in states:
        spawn = depth_step(spawn, 5)
        assert state == spawn


def test_iterate_passes():
    iterate = effective_state(11, 3, inherited=True)
    assert iterate == 0x0000000E
    pass_states = iterate_pass_states(iterate, 2)
    assert pass_states == [iterate_step(iterate), iterate_step(iterate_step(iterate))]
    total = 0.0
    for incoming, vector in zip(pass_states, PASSES, strict=True):
        effective, state, unit, u16 = vector
        child = effective_state(incoming, 5)
        assert child == effective
        rng = FxRng(child)
        value = rng.rand(1.0)
        assert rng.state == state
        assert value == pytest.approx(unit, abs=1e-15)
        assert round(value * 65535) == u16
        total += value
    assert round(total * 65535) == 72861


def test_not_inherited_keeps_own_seed():
    assert effective_state(0xDEADBEEF, 3, inherited=False) == 3
    assert effective_state(0xFFFFFFFF, 2) == 1


def test_independent_parameter_functions():
    opacity = FxRng(0)
    assert opacity.rand(1.0) == 0.003022313117980957
    assert opacity.rand(1.0) == pytest.approx(0.6204589605331421, abs=1e-15)
    offset = FxRng(0)
    assert offset.rand(1.0) == 0.003022313117980957


def test_zero_amplitude_still_advances():
    rng = FxRng(0)
    assert rng.rand(0.0) == 0.0
    assert rng.state == 0x0000C613
    assert rng.rand(1.0) == pytest.approx(0.6204589605331421, abs=1e-15)


def test_glsl_helpers_declared():
    assert "uint fx_depth_step(uint s, float depth)" in FX_RNG_GLSL
    assert "uint fx_iter_step(uint s)" in FX_RNG_GLSL
    assert "float fx_rng_next(inout uint s, float a)" in FX_RNG_GLSL
    assert "int(depth) & 15" in FX_RNG_GLSL
    assert "uintBitsToFloat" in FX_RNG_GLSL
    assert f"0x{FX_LCG_M:08X}u + 0x{FX_LCG_C:08X}u" in FX_RNG_GLSL
    for m, c in zip(DEPTH_M, DEPTH_C, strict=True):
        assert f"0x{m:08X}u + 0x{c:08X}u" in FX_RNG_GLSL
    assert "0x598232B9u + 0x000038C5u" in FX_RNG_GLSL
