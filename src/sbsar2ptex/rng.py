"""The pixel processor's random generator, as docs/random-generator.md specifies.

Each pixel owns a 32-bit state: the Jenkins one-at-a-time hash of the graph
seed and the binary32 bit patterns of the pixel's normalized center. Every
``rand(a)`` call first advances the state by an LCG, then turns state bits
1-23 into a unit value in [0, 1) and scales it by ``a`` in binary32.

``RNG_GLSL`` is the same generator as GLSL 4.50 helpers for Material Maker.
"""

import struct

UINT32_MASK = 0xFFFFFFFF
LCG_M = 0x4937573B
LCG_C = 0x0000C7C1
MANTISSA_MASK = 0x007FFFFF
ONE_BITS = 0x3F800000


def float32(value: float) -> float:
    """Round a float to the nearest IEEE-754 binary32 value."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def float_bits(value: float) -> int:
    """Return the binary32 bit pattern of ``value`` (rounded to binary32)."""
    return struct.unpack("<I", struct.pack("<f", value))[0]


def bits_float(bits: int) -> float:
    """Reinterpret a 32-bit pattern as a binary32 value."""
    return struct.unpack("<f", struct.pack("<I", bits & UINT32_MASK))[0]


def initial_state(seed: int, position_x_bits: int, position_y_bits: int) -> int:
    """Hash the seed and the position bit patterns into a pixel's first state."""
    h = 0
    for word in (seed & UINT32_MASK, position_x_bits, position_y_bits):
        h = (h + (word & UINT32_MASK)) & UINT32_MASK
        h = (h + (h << 10)) & UINT32_MASK
        h ^= h >> 6
    h = (h + (h << 3)) & UINT32_MASK
    h ^= h >> 11
    return (h + (h << 15)) & UINT32_MASK


def advance(state: int) -> int:
    """Return the state after one LCG step."""
    return (state * LCG_M + LCG_C) & UINT32_MASK


def unit_value(state: int) -> float:
    """Return the unit value in [0, 1) carried by an (advanced) state."""
    mantissa = (state >> 1) & MANTISSA_MASK
    return bits_float(ONE_BITS | mantissa) - 1.0


class PixelRng:
    """One pixel's generator; ``rand`` advances its state before each value."""

    __slots__ = ("state",)

    def __init__(self, state: int) -> None:
        """Start from an initial (hashed) state."""
        self.state = state & UINT32_MASK

    @classmethod
    def for_pixel(
        cls, x: int, y: int, width: int, height: int, seed: int
    ) -> "PixelRng":
        """Create pixel ``(x, y)``'s generator in a ``width`` x ``height`` image."""
        position_x = float_bits((x + 0.5) / width)
        position_y = float_bits((y + 0.5) / height)
        return cls(initial_state(seed, position_x, position_y))

    def rand(self, amplitude: float) -> float:
        """Advance the state and return a value in [0, amplitude), in binary32."""
        self.state = advance(self.state)
        return float32(unit_value(self.state) * float32(amplitude))


RNG_GLSL = """\
uint sbs_rng_init(uint seed, vec2 position) {
    uint words[3] = uint[3](
        seed, floatBitsToUint(position.x), floatBitsToUint(position.y));
    uint h = 0u;
    for (int i = 0; i < 3; ++i) {
        h += words[i];
        h += h << 10u;
        h ^= h >> 6u;
    }
    h += h << 3u;
    h ^= h >> 11u;
    h += h << 15u;
    return h;
}

float sbs_rng_next(inout uint state, float amplitude) {
    state = state * 0x4937573Bu + 0x0000C7C1u;
    uint mantissa = (state >> 1u) & 0x007FFFFFu;
    return (uintBitsToFloat(0x3F800000u | mantissa) - 1.0) * amplitude;
}
"""
