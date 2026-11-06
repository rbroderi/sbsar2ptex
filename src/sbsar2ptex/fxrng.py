"""The FX-Map random generator, as docs/random-generator.md sections 8-9 specify.

FX-Map functions use their own LCG step and the pixel processor's float
conversion, but no pixel-position hash: each dynamic function starts a private
generator from the effective state of the tree node it belongs to. Tree nodes
derive their children's incoming states by affine transforms: Quadrant slots by
a transform chosen by the child depth modulo 16, Iterate passes by a single
Iterate step.

``FX_RNG_GLSL`` is the same transforms and generator as GLSL helpers for
Material Maker.
"""

import numpy as np

from sbsar2ptex.rng import MANTISSA_MASK
from sbsar2ptex.rng import ONE_BITS
from sbsar2ptex.rng import UINT32_MASK
from sbsar2ptex.rng import float32
from sbsar2ptex.rng import unit_value

PARAM_LCG = {
    "opacity": (0x4937573B, 0x0000C7C1),
    "branchoffset": (0x5AD7FCD1, 0x0000C72D),
    "frameoffset": (0x3E1BD4C1, 0x0000C613),
    "patternsize": (0x4458B04D, 0x0000C4EB),
    "patternrotation": (0x4A34F3F5, 0x0000C3D3),
    "patternsuppl": (0x5FBCF1A6, 0x0000C5D7),
    "imageindex": (0x0019660D, 0x3C6EF35F),
    "iteratecount": (0x5690FB9D, 0x0000C59F),
    "switch": (0x70DCB7DC, 0x0000C3BF),
}
FX_LCG_M, FX_LCG_C = PARAM_LCG["frameoffset"]

# Quadrant child-slot transforms, indexed by child depth modulo 16.
DEPTH_M = (
    0x63DC7A18, 0x4019CF16, 0x6454A406, 0x7F13AC69,
    0x5CA1EBE2, 0x7527BAB1, 0x6AC690C5, 0x5CDCFAA7,
    0x7549D83B, 0x613B9CA8, 0x7A3FFD4B, 0x56470DA9,
    0x6A364C22, 0x5A76F899, 0x7C3448BE, 0x62AFDC7A,
)  # fmt: skip
DEPTH_C = (
    0x0000CD5B, 0x0000CD93, 0x0000CDC1, 0x0000CDE7,
    0x0000CE0B, 0x0000CE47, 0x0000CE7B, 0x0000CC19,
    0x0000CC41, 0x0000CC61, 0x0000CC9B, 0x0000CCE5,
    0x0000CD1F, 0x0000CD3F, 0x0000CBDD, 0x0000CAD9,
)  # fmt: skip

ITERATE_M = 0x598232B9
ITERATE_C = 0x000038C5


def depth_step(state: int, depth: int) -> int:
    """Apply the Quadrant child transform for child depth ``depth`` once."""
    index = depth & 15
    return (state * DEPTH_M[index] + DEPTH_C[index]) & UINT32_MASK


def iterate_step(state: int) -> int:
    """Return the state after one Iterate pass step."""
    return (state * ITERATE_M + ITERATE_C) & UINT32_MASK


def effective_state(incoming: int, seed: int, *, inherited: bool = True) -> int:
    """Resolve a tree node's effective state from its incoming state and seed.

    Quadrants always inherit; ``inherited`` is the ``randominherited`` flag of
    Iterate and Switch nodes. Not inheriting discards only the incoming state,
    never the node's own seed.
    """
    base = incoming if inherited else 0
    return (base + seed) & UINT32_MASK


def quadrant_child_states(parent_effective: int, parent_depth: int) -> list[int]:
    """Return the incoming states of a Quadrant's four child slots, in order.

    Slot ``i`` receives the child-depth transform applied ``i + 1`` times to
    the parent's effective state; empty slots still take their place.
    """
    states = []
    spawn = parent_effective & UINT32_MASK
    for _ in range(4):
        spawn = depth_step(spawn, parent_depth + 1)
        states.append(spawn)
    return states


def iterate_pass_states(effective: int, count: int) -> list[int]:
    """Return the incoming states of an Iterate node's ``count`` repeated passes.

    Pass ``n`` receives ``n + 1`` Iterate steps of the node's effective state.
    Output 0, if any, receives the effective state itself.
    """
    states = []
    pass_state = effective & UINT32_MASK
    for _ in range(count):
        pass_state = iterate_step(pass_state)
        states.append(pass_state)
    return states


class FxRng:
    """One dynamic function's private generator, started from a node state."""

    __slots__ = ("increment", "multiplier", "state")

    def __init__(self, state: int, parameter: str = "frameoffset") -> None:
        """Start a parameter lane from a tree node's effective state."""
        self.state = state & UINT32_MASK
        self.multiplier, self.increment = PARAM_LCG[parameter]

    def rand(self, amplitude: float) -> float:
        """Advance the state and return a value in [0, amplitude), in binary32."""
        self.state = (self.state * self.multiplier + self.increment) & UINT32_MASK
        return float32(unit_value(self.state) * float32(amplitude))


class FxRngLanes:
    """FxRng over many lanes at once: states a uint64 array (32-bit values)."""

    __slots__ = ("increment", "multiplier", "state")

    def __init__(self, state: np.ndarray, parameter: str = "frameoffset") -> None:
        """Start a parameter lane from each lane's tree node state."""
        self.state = np.asarray(state, dtype=np.uint64) & np.uint64(UINT32_MASK)
        m, c = PARAM_LCG[parameter]
        self.multiplier, self.increment = np.uint64(m), np.uint64(c)

    def rand(self, amplitude: np.ndarray) -> np.ndarray:
        """Return FxRng.rand of each lane (float32 values, as float64s)."""
        # (a 32-bit product fits in 64 bits: the low 32 are the LCG's)
        s = (self.state * self.multiplier + self.increment) & np.uint64(UINT32_MASK)
        self.state = s
        a = np.asarray(amplitude, dtype=np.float64)
        a32 = a.astype(np.float32)
        if (np.isfinite(a) & ~np.isfinite(a32)).any():
            msg = "float too large to pack with f format"  # as struct's
            raise OverflowError(msg)
        return lane_float32(unit_values(s) * a32.astype(np.float64))


def lane_float32(x: np.ndarray) -> np.ndarray:
    """Return float32 of each lane (as float64s), raising where float32 would."""
    y = x.astype(np.float32)
    if (np.isfinite(x) & ~np.isfinite(y)).any():
        msg = "float too large to pack with f format"
        raise OverflowError(msg)
    return y.astype(np.float64)


def unit_values(states: np.ndarray) -> np.ndarray:
    """Return unit_value of each (advanced) state, as float64s."""
    mantissa = (states >> np.uint64(1)) & np.uint64(MANTISSA_MASK)
    bits = (mantissa | np.uint64(ONE_BITS)).astype(np.uint32)
    return bits.view(np.float32).astype(np.float64) - 1.0


def depth_steps(states: np.ndarray, depth: int) -> np.ndarray:
    """Return depth_step of each lane's state (uint64 arrays)."""
    index = depth & 15
    m, c = np.uint64(DEPTH_M[index]), np.uint64(DEPTH_C[index])
    return (states * m + c) & np.uint64(UINT32_MASK)


def iterate_steps(states: np.ndarray, passes: np.ndarray) -> np.ndarray:
    """Return each lane's state after ``passes + 1`` Iterate steps (uint64s).

    The steps compose to one affine map a pass count: m^k s + c (1 + m + ...
    + m^(k-1)), mod 2^32, tabled up to the most passes.
    """
    most = int(passes.max()) + 1 if passes.size else 0
    mul, add = [], []
    m, c = 1, 0
    for _ in range(most):
        m, c = (m * ITERATE_M) & UINT32_MASK, (c * ITERATE_M + ITERATE_C) & UINT32_MASK
        mul.append(m)
        add.append(c)
    k = passes.astype(np.intp)
    ms = np.array(mul, dtype=np.uint64)[k]
    cs = np.array(add, dtype=np.uint64)[k]
    return (states * ms + cs) & np.uint64(UINT32_MASK)


def _depth_glsl() -> str:
    """Write the depth-transform table as an if/else chain of GLSL returns."""
    lines = []
    for index, (m, c) in enumerate(zip(DEPTH_M, DEPTH_C, strict=True)):
        keyword = "if" if index == 0 else "} else if"
        body = f"        return s * 0x{m:08X}u + 0x{c:08X}u;"
        if index == len(DEPTH_M) - 1:
            lines.append("    } else {")
        else:
            lines.append(f"    {keyword} (d == {index}) {{")
        lines.append(body)
    lines.append("    }")
    return "\n".join(lines)


FX_RNG_GLSL = f"""\
uint fx_depth_step(uint s, float depth) {{
    int d = int(depth) & 15;
{_depth_glsl()}
}}

uint fx_iter_step(uint s) {{
    return s * 0x{ITERATE_M:08X}u + 0x{ITERATE_C:08X}u;
}}

float fx_rng_next(inout uint s, float a, uint m, uint c) {{
    s = s * m + c;
    uint mantissa = (s >> 1u) & 0x007FFFFFu;
    return (uintBitsToFloat(0x3F800000u | mantissa) - 1.0) * a;
}}
"""
