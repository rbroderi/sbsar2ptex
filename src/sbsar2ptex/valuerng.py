"""Random generator for once-per-node ordinary-filter value functions."""

from sbsar2ptex.rng import UINT32_MASK
from sbsar2ptex.rng import float32
from sbsar2ptex.rng import unit_value

VALUE_LCG_M = 0x7FF8A3ED
VALUE_LCG_C = 0x2AA01D31


class ValueRng:
    """One value-function generator, initialized from its node's seed."""

    __slots__ = ("state",)

    def __init__(self, state: int) -> None:
        """Start from an ordinary filter's resolved random seed."""
        self.state = state & UINT32_MASK

    def rand(self, amplitude: float) -> float:
        """Advance and return a binary32 value in ``[0, amplitude)``."""
        self.state = (self.state * VALUE_LCG_M + VALUE_LCG_C) & UINT32_MASK
        return float32(unit_value(self.state) * float32(amplitude))


VALUE_RNG_GLSL = f"""\
float sbs_value_rng_next(inout uint state, float amplitude) {{
    state = state * 0x{VALUE_LCG_M:08X}u + 0x{VALUE_LCG_C:08X}u;
    uint mantissa = (state >> 1u) & 0x007FFFFFu;
    return (uintBitsToFloat(0x3F800000u | mantissa) - 1.0) * amplitude;
}}
"""
