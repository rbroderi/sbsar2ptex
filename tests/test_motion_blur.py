"""The legacy motion blur, ops 18 and 19 (docs/motion-blur-legacy.md)."""

import struct

import numpy as np
import pytest

from sbsar2ptex.sbsasm import BASE
from sbsar2ptex.sbsasm import COLOR_OPS
from sbsar2ptex.sbsasm import Asm
from sbsar2ptex.translate import motion_blur_glsl
from sbsar2ptex.translate import translate_graph

OFF = -1
"""In a node's words below: an offset to a code block after its words."""


def f(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def node_file(word: int, rest: list[int], count: int = 10) -> Asm:
    """Return an SBAM file: count - 1 uniforms, then a node ``word`` with ``rest``.

    Each OFF in ``rest`` is an offset to its own (empty) code block after
    the node's words.
    """
    start = 0x40 + 4 * (count - 1)
    blocks = start + 4 * (1 + len(rest))
    n_off = rest.count(OFF)
    offsets = iter(range(blocks, blocks + 4 * n_off, 4))
    words = [word] + [next(offsets) - BASE if w == OFF else w for w in rest]
    words += [0] * n_off
    table = start + 4 * len(words)
    outputs = table + 4 * count
    data = bytearray(outputs + 8 + 8 + 16)
    data[:4] = b"SBAM"
    struct.pack_into("<H", data, 6, 2)
    for k in range(count - 1):
        struct.pack_into("<I", data, 0x40 + 4 * k, 12)
    struct.pack_into(f"<{len(words)}I", data, start, *words)
    starts = [0x40 + 4 * k for k in range(count - 1)] + [start]
    struct.pack_into(f"<{count}I", data, table, *(s - BASE for s in starts))
    struct.pack_into("<HHI", data, outputs + 8, 1, 0, 0x1234)
    struct.pack_into("<4I", data, outputs + 16, count, table - BASE, outputs - BASE, 0)
    return Asm(bytes(data))


@pytest.mark.parametrize("op", [0x12, 0x13])  # test 9: colour reads the same
@pytest.mark.parametrize(
    ("flags", "rest", "params", "pixel"),
    [
        # docs/motion-blur-legacy.md section 4, tests 1-8
        (0x0318, [1, 7, 8, f(4.0)], {"intensity": (1, [4.0]), "angle": (0, None)},
         None),
        (0x0318, [5, 7, 8, f(4.0), f(0.1)],
         {"intensity": (1, [4.0]), "angle": (1, [pytest.approx(0.1)])}, None),
        (0x0318, [4, 7, 8, f(0.1)],
         {"intensity": (0, None), "angle": (1, [pytest.approx(0.1)])}, None),
        (0x0718, [1, 7, 8, f(2.0), f(1.0), f(4.0)],
         {"intensity": (1, [4.0]), "angle": (0, None)}, [2.0, 1.0]),
        (0x0B19, [1, 7, 8, OFF, OFF, f(4.0)],
         {"intensity": (1, [4.0]), "angle": (0, None)}, "code"),
        (0x0398, [2, 7, 8, OFF, OFF], {"intensity": (2, "code"), "angle": (0, None)},
         None),
        (0x0358, [2, 7, 8, 5, OFF], {"intensity": (2, "code"), "angle": (0, None)},
         None),
        (0x0398, [9, 7, 8, OFF, f(2.0), OFF],
         {"intensity": (1, [2.0]), "angle": (2, "code")}, None),
    ],
)  # fmt: skip
def test_layouts(op: int, flags: int, rest: list[int], params: dict, pixel) -> None:
    node = node_file(flags << 16 | 0x18 << 8 | op, rest).nodes()[-1]
    assert node.unsupported is None
    assert node.inputs == {"input": 7, "gradient": 8}
    got = {
        name: (state, "code" if state == 2 else payload)
        for name, (state, payload) in node.params.items()
    }
    assert got == params
    if pixel == "code":
        assert node.pixel_size_code is not None
    else:
        assert node.pixel_size == pixel
    assert (op in COLOR_OPS) == (op == 0x13)
    if flags & 0xC0 == 0x40:  # test 7: the inline seed
        assert node.seed == 5
    if flags & 1:
        assert node.size_code is not None


# A float32 model of section 3.2 (the GLSL's arithmetic), for render tests.

K = np.float32(0.335)
W = np.float32(0.166666)


def model(
    inp: np.ndarray, grad: np.ndarray, k: float, turn: float, *, wrap: bool = True
) -> np.ndarray:
    n = inp.shape[0]
    f32 = np.float32
    ii, jj = np.meshgrid(np.arange(n), np.arange(n))  # column, row

    def idx(x: np.ndarray) -> np.ndarray:
        return np.mod(x, n) if wrap else np.clip(x, 0, n - 1)

    h0 = grad[jj, ii]
    dx = (h0 - grad[jj, idx(ii - 1)]).astype(f32)
    dy = (h0 - grad[idx(jj - 1), ii]).astype(f32)
    a = f32(f32(6.2831855) * f32(turn))
    c, s = f32(np.cos(np.float64(a))), f32(np.sin(np.float64(a)))
    rx, ry = (dx * c - dy * s).astype(f32), (dx * s + dy * c).astype(f32)
    m = f32(f32(k) * f32(n / 256.0))
    mk = f32(m * K)
    big_dx, big_dy = (rx * mk).astype(f32), (ry * mk).astype(f32)
    px = ((ii + 0.5) / n).astype(f32)
    py = ((jj + 0.5) / n).astype(f32)

    def tap(qx: np.ndarray, qy: np.ndarray) -> np.ndarray:
        x, y = (
            (qx * f32(n) - f32(0.5)).astype(f32),
            (qy * f32(n) - f32(0.5)).astype(f32),
        )
        i0, j0 = np.floor(x), np.floor(y)
        fx = np.floor((x - i0) * 256 + 0.5) / 256
        fy = np.floor((y - j0) * 256 + 0.5) / 256
        i0, j0 = i0.astype(int), j0.astype(int)

        def t(i: np.ndarray, j: np.ndarray) -> np.ndarray:
            return inp[idx(j), idx(i)]

        top = t(i0, j0) * (1 - fx) + t(i0 + 1, j0) * fx
        bot = t(i0, j0 + 1) * (1 - fx) + t(i0 + 1, j0 + 1) * fx
        return (top * (1 - fy) + bot * fy).astype(f32)

    acc = np.zeros_like(px)
    for tt in (0.5, 1.5, 2.5):
        plus = tap(f32(px + f32(tt) * big_dx), f32(py + f32(tt) * big_dy))
        minus = tap(f32(px - f32(tt) * big_dx), f32(py - f32(tt) * big_dy))
        acc = (acc + (plus + minus)).astype(f32)
    return (acc * W).astype(f32)


def levels(x: np.ndarray) -> np.ndarray:
    return np.round(np.clip(x, 0, 1) * 65535).astype(int)


def ramp_x(n: int = 256) -> np.ndarray:
    row = np.round((np.arange(n) + 0.5) / n * 65535) / 65535
    return np.tile(row, (n, 1)).astype(np.float32)


def impulse(n: int = 256) -> np.ndarray:
    img = np.zeros((n, n), dtype=np.float32)
    img[128, 128] = 1.0
    return img


def test_r1_impulse_intensity_4() -> None:
    out = levels(model(impulse(), ramp_x(), 4.0, 0.0))
    row = out[128, 124:133].tolist()
    assert row == [3840, 7211, 10794, 7339, 7253, 7339, 10794, 7211, 3840]
    assert out.sum() == sum(row)


def test_r2_impulse_intensity_40() -> None:
    out = levels(model(impulse(), ramp_x(), 40.0, 0.0))[128]
    want = {6: 3285, 7: 7637, 20: 9813, 21: 1109, 33: 5461, 34: 5461}
    for d, lv in want.items():
        assert out[128 + d] == lv
        assert out[128 - d] == lv
    assert out[128] == 0


def test_r4_transposed() -> None:
    out = levels(model(impulse(), ramp_x().T.copy(), 4.0, 0.0))
    col = out[124:133, 128].tolist()
    assert col == [3840, 7211, 10794, 7339, 7253, 7339, 10794, 7211, 3840]


def test_r12_a_flat_gradient_keeps_the_input() -> None:
    rng = np.random.default_rng(1)
    inp = (rng.integers(0, 65536, (64, 64)) / 65535).astype(np.float32)
    out = model(inp, np.full((64, 64), 0.5, np.float32), 4.0, 0.3)
    assert np.array_equal(levels(out), levels(inp))


def test_a_motion_blur_translates_with_live_knobs() -> None:
    nodes = node_file(
        0x0318 << 16 | 0x18 << 8 | 0x13, [5, 7, 8, f(4.0), f(0.1)]
    ).nodes()
    mm, wires, outs = translate_graph(nodes, {"out": 9}, {}, size=8)
    made = next(n for n in mm if n["name"] == "n9")
    assert made["parameters"]["intensity"] == 4.0
    assert made["parameters"]["angle"] == pytest.approx(0.1)
    assert [i["name"] for i in made["shader_model"]["inputs"]] == ["input", "gradient"]
    assert "$intensity" in made["shader_model"]["instance"]
    assert outs == {"out": ("n9", 0)}
    assert {w["to_port"] for w in wires if w["to"] == "n9"} == {0, 1}


def test_glsl_shape() -> None:
    helpers, body = motion_blur_glsl(
        "float", "N", "$intensity", "$angle", "vec2(1.0)", (True, False)
    )
    assert "mod(ij.x, N)" in helpers
    assert "clamp(ij.y, 0.0, N - 1.0)" in helpers
    assert "floor((x - i0) * 256.0 + 0.5) / 256.0" in helpers
    assert "vec2 D = r * (m * 0.335);" in body
    assert "fma(t, D, p)" in body
    assert body.endswith("return acc * 0.166666;")
