"""The FX-Map random generator, as docs/random-generator.md sections 8-9 specify.

FX-Map functions use the pixel processor's LCG step and float conversion, but
no pixel-position hash: each dynamic function starts a private generator from
the effective state of the tree node it belongs to. Tree nodes derive their
children's incoming states by affine transforms: Quadrant slots by a transform
chosen by the child depth modulo 16, Iterate passes by a single Iterate step.

``FX_RNG_GLSL`` is the same transforms and generator as GLSL helpers for
Material Maker.
"""

from sbsar2ptex.rng import UINT32_MASK
from sbsar2ptex.rng import advance
from sbsar2ptex.rng import float32
from sbsar2ptex.rng import unit_value

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

    __slots__ = ("state",)

    def __init__(self, state: int) -> None:
        """Start from a tree node's effective state."""
        self.state = state & UINT32_MASK

    def rand(self, amplitude: float) -> float:
        """Advance the state and return a value in [0, amplitude), in binary32."""
        self.state = advance(self.state)
        return float32(unit_value(self.state) * float32(amplitude))


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

float fx_rng_next(inout uint s, float a) {{
    s = s * 0x4937573Bu + 0x0000C7C1u;
    uint mantissa = (s >> 1u) & 0x007FFFFFu;
    return (uintBitsToFloat(0x3F800000u | mantissa) - 1.0) * a;
}}
"""
