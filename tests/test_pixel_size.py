"""The pixel-size vector in the translations (docs/node-params.md section 7).

Acceptance tests R1 to R13 on synthetic nodes (test_node_params.node). They
do not render: each checks the translation's GLSL takes the vector as 7.2
has it (its x geometry times v.x, its y times v.y, each product float32),
and the effective per-axis values that GLSL computes, mirrored here with
numpy's float32, against the equivalences the spec measured.
"""

import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_function import block
from test_node_params import BLOCKS
from test_seed import assemble
from test_seed import cook
from test_seed import f32w

from sbsar2ptex import plugin
from sbsar2ptex import translate
from sbsar2ptex.function import FLOAT
from sbsar2ptex.function import decode
from sbsar2ptex.function import evaluate
from sbsar2ptex.sbsasm import Node

f32 = np.float32


def op(code: int, width: int, *operands: int) -> list[int]:
    """Return a float instruction's u16s: opcode, then operands."""
    return [len(operands) << 10 | FLOAT << 8 | (width - 1) << 6 | code, *operands]


SIZELOG2, SWIZZLE, ADD, SUB, MUL, NEG, POW2, MIN = (
    0x01,
    0x10,
    0x12,
    0x13,
    0x14,
    0x17,
    0x2B,
    0x30,
)

DEFAULT = block(
    op(SIZELOG2, 2, 3),  # 0 s = $sizelog2
    op(SWIZZLE, 2, 0, 1),  # 1 s.yx
    op(SUB, 2, 1, 0),  # 2 s.yx - s.xy
    op(SUB, 2, 0, 0),  # 3 (0, 0)
    op(MIN, 2, 2, 3),  # 4 min(s.yx - s.xy, 0)
    op(POW2, 2, 4),  # 5 the cooker's default (section 1.1)
)
"""The cooker's default pixel-size block: (min(w, h) / w, min(w, h) / h)."""
DEFAULT_X2 = block(
    op(SIZELOG2, 2, 3),
    op(SWIZZLE, 2, 0, 1),
    op(SUB, 2, 1, 0),
    op(SUB, 2, 0, 0),
    op(MIN, 2, 2, 3),
    op(POW2, 2, 4),
    op(POW2, 2, 3),  # 6 (1, 1)
    op(ADD, 2, 6, 6),  # 7 (2, 2)
    op(MUL, 2, 5, 7),  # 8 the default times Pixel Size (2, 2)
)
"""R12's block: the default times the Pixel Size (2, 2)."""
QUARTER_LOG = block(
    op(SIZELOG2, 2, 3),  # 0 $sizelog2
    op(SUB, 2, 0, 0),  # 1 (0, 0)
    op(POW2, 2, 1),  # 2 (1, 1)
    op(ADD, 2, 2, 2),  # 3 (2, 2)
    op(NEG, 2, 3),  # 4 (-2, -2)
    op(POW2, 2, 4),  # 5 (0.25, 0.25)
    op(MUL, 2, 0, 5),  # 6 $sizelog2 / 4: (2, 2) at 256 pixels
)
"""A pixel size that follows the output size: not (1, 1), so kept live."""
OWN = {**BLOCKS, "DEF": DEFAULT, "DEF2": DEFAULT_X2, "LOG4": QUARTER_LOG}


def node(*head: str | float) -> Node:
    """Cook a node: ``head`` its words (str a block's name, float an f32)."""
    words = [f32w(w) if isinstance(w, float) else w for w in head]
    names = [w for w in words if isinstance(w, str)]
    return cook(assemble(words, [(n, OWN[n]) for n in names]))


@pytest.fixture(autouse=True)
def ctx() -> None:
    translate._CTX = translate._Context(size=8, inputs={})


def shaders(got) -> list[dict]:
    """Return a translation's shader nodes (subgraphs' too), in order."""
    nodes = got.nodes if hasattr(got, "nodes") else [got[0]]
    out = []
    for nd in nodes:
        if nd.get("shader_model"):
            out.append(nd)
        out += [i for i in nd.get("nodes", []) if i.get("shader_model")]
    return out


def bodies(head: tuple) -> list[str]:
    """Return the instance GLSL of each shader the node translates to."""
    got = translate.node_shader(node(*head), "n0", size=8)
    return [nd["shader_model"]["instance"] for nd in shaders(got)]


def body(head: tuple) -> str:
    (text,) = bodies(head)
    return text


# Node words (docs/node-params.md 1.1, 1.2): pixel size inline (state 1,
# P bits 2-3 = 1), or a function (state 2), or none (state 0).


def blur(v: tuple | str | None, k: float = 1.0) -> tuple:
    """Op 20, intensity k constant (P bits 4-5 = 1)."""
    if v is None:
        return (0x13198814, 7, "SZ", k)
    if isinstance(v, str):
        return (0x1B198814, 7, "SZ", v, k)
    return (0x17198814, 7, "SZ", *map(float, v), k)


def warp(v: tuple | str | None, k: float = 1.0) -> tuple:
    """Op 14, version 9, intensity k constant (P bits 5-6 = 1)."""
    if v is None:
        return (0x2319880E, 0, 3, 4, "SZ", k)
    if isinstance(v, str):
        return (0x2B19880E, 0, 3, 4, "SZ", v, k)
    return (0x2719880E, 0, 3, 4, "SZ", *map(float, v), k)


def dmb(v: tuple, k: float, turn: float) -> tuple:
    """Op 22, intensity and angle constant (F = 0x5)."""
    return (0x07198816, 0x5, 7, "SZ", *map(float, v), k, turn)


def dwarp(v: tuple, k: float, turn: float) -> tuple:
    """Op 24, intensity and angle constant (F = 0xA)."""
    return (0x07198818, 0xA, 7, 8, "SZ", *map(float, v), k, turn)


def sharpen(v: tuple | None, k: float = 1.0) -> tuple:
    """Op 26, intensity constant."""
    if v is None:
        return (0x1319881A, 7, "SZ", k)
    return (0x1719881A, 7, "SZ", *map(float, v), k)


def normal(v: tuple | None, k: float = 1.0) -> tuple:
    """Op 37, intensity constant, invert Y false, alpha true (F = 0x11)."""
    if v is None:
        return (0x03198825, 0x11, 7, "SZ", k)
    return (0x07198825, 0x11, 7, "SZ", *map(float, v), k)


def distance(v: tuple | None, d: float) -> tuple:
    """Op 42 with a source, distance constant (F = 0x7)."""
    if v is None:
        return (0x0319882A, 0x7, 5, 6, "SZ", d)
    return (0x0719882A, 0x7, 5, 6, "SZ", *map(float, v), d)


def emboss(v: tuple, k: float) -> tuple:
    """Op 16, intensity and light angle constant (F = 0xA)."""
    return (0x07198810, 0xA, 7, 8, "SZ", *map(float, v), k, 0.1)


# What the GLSL computes, mirrored in float32.


def mul(a: float, b: float) -> f32:
    return f32(a) * f32(b)


def dmb_effective(k: float, turn: float, v: tuple) -> tuple[f32, float]:
    """Return dmb_pixel_size_glsl's (psk, psA): the v = (1, 1) blur this is."""
    vx, vy = f32(v[0]), f32(v[1])
    if vx == vy:
        return mul(k, vx), turn
    a4 = f32(turn) * f32(4.0)
    m4 = math.floor(a4 + 0.5)
    if a4 == m4:
        return mul(k, vx if m4 % 2 == 0 else vy), turn
    pa = 2 * math.pi * turn
    d = (k * vx * math.cos(pa), k * vy * math.sin(pa))
    return f32(math.hypot(*d)), math.atan2(d[1], d[0]) / (2 * math.pi)


# R1, R2: blur.


def test_r1_blur_isotropic() -> None:
    x, y = bodies(blur((2, 2)))
    assert "($intensity * vec2(2.0, 2.0).x) * N / 256.0" in x
    assert "($intensity * vec2(2.0, 2.0).y) * N / 256.0" in y
    # both passes: intensity 1 x 2 = 2
    assert mul(1.0, 2.0) == f32(2.0)


def test_r2_blur_per_axis() -> None:
    x, y = bodies(blur((2, 1)))
    assert "($intensity * vec2(2.0, 1.0).x)" in x
    assert "($intensity * vec2(2.0, 1.0).y)" in y
    assert (mul(1.0, 2.0), mul(1.0, 1.0)) == (2.0, 1.0)


# R3, R4: warp.


def test_r3_warp_bathroom_marble() -> None:
    text = body(warp((0.0444, 0.0444), 38.0))
    assert "vec2 psv = vec2(0.0444, 0.0444);" in text
    assert (
        "float Kx = wnc_$(name)($intensity * psv.x, z) * ($pass_size.x / 256.0);"
        in (text)
    )
    assert (
        "float Ky = wnc_$(name)($intensity * psv.y, z) * ($pass_size.x / 256.0);"
        in (text)
    )
    # the float32 product is the intensity Substance renders it as
    assert mul(38.0, 0.0444) == f32(1.6872)


def test_r4_warp_per_axis() -> None:
    text = body(warp((2, 1)))
    # x's difference times K along x, y's times K along y
    assert "float ox = wnc_$(name)((h - hx) * Kx, z);" in text
    assert "vec2 o = vec2(ox, wnc_$(name)((h - hy) * Ky, z));" in text
    assert "vec2 psv = vec2(2.0, 1.0);" in text
    kx, ky = mul(1.0, 2.0), mul(1.0, 1.0)
    # a gradient varying along x only (h - hy = 0): as v = (2, 2)
    assert (kx, f32(0.0) * ky) == (mul(1.0, 2.0), f32(0.0) * mul(1.0, 2.0))
    # along y only: as v = (1, 1)
    assert (f32(0.0) * kx, ky) == (f32(0.0) * mul(1.0, 1.0), mul(1.0, 1.0))


def test_a_unit_vector_leaves_the_translation_as_it_was() -> None:
    for make in (blur, warp):
        plain = bodies(make(None))
        assert bodies(make((1, 1))) == plain
        assert bodies(make("DEF")) == plain  # the cooker's default: (1, 1)
    assert bodies(sharpen((1, 1))) == bodies(sharpen(None))
    assert bodies(normal((1, 1))) == bodies(normal(None))
    assert bodies(distance((1, 1), 8.0)) == bodies(distance(None, 8.0))
    assert "psv" not in body(warp(None))


# R5: emboss (already translated with the vector).


def test_r5_emboss() -> None:
    text = body(emboss((2, 2), 1.0))
    assert "h - $gradient(fract(uv - e.yx))) * vec2(2.0, 2.0);" in text
    # s = (k N / 256) dot(g v, light): v = (2, 2) at k = 1 is k = 2, exactly
    rng = np.random.default_rng(5)
    g = rng.random((100, 2), dtype=np.float32) - f32(0.5)
    light = np.array([math.cos(0.2 * math.pi), math.sin(0.2 * math.pi)], np.float32)
    n = f32(1.0)  # N / 256 at 256 pixels
    a = (f32(1.0) * n) * ((g * f32(2.0)) @ light)
    b = (f32(2.0) * n) * (g @ light)
    assert np.array_equal(a, b)


# R6, R7: directional motion blur.


def test_r6_directional_motion_blur_per_axis() -> None:
    text = body(dmb((2, 1), 4.0, 0.0))
    assert "vec2 psv = vec2(2.0, 1.0);" in text
    assert "psk = $intensity * (mod(m4, 2.0) == 0.0 ? psv.x : psv.y);" in text
    assert "float A = psA;" in text
    assert "float R = 0.5 + (psk * N / 256.0 - 0.5)" in text
    # 0 turns: (2, 1) as (2, 2); 0.25 turns: (2, 1) as (1, 1)
    assert dmb_effective(4.0, 0.0, (2, 1)) == dmb_effective(4.0, 0.0, (2, 2))
    assert dmb_effective(4.0, 0.25, (2, 1)) == dmb_effective(4.0, 0.25, (1, 1))
    assert dmb_effective(4.0, 0.5, (2, 1))[0] == 8.0
    assert dmb_effective(4.0, 0.75, (2, 1))[0] == 4.0


def test_r7_directional_motion_blur_isotropic() -> None:
    assert "float psk = $intensity * psv.x;" in body(dmb((0.3, 0.3), 3.0, 0.1))
    k, turn = dmb_effective(3.0, 0.1, (0.3, 0.3))
    assert turn == 0.1
    assert k == mul(3.0, 0.3)  # float32: 0.90000004, not f32(0.9)
    assert abs(float(k) - 0.9) < 1e-6


def test_oblique_unequal_directional_motion_blur_is_the_vector_form() -> None:
    # 7.4: approximate; intensity 16 at 0.1 turns with (2, 1) is close to
    # intensity 27.543829 at 0.0554573 turns with (1, 1).
    text = body(dmb((2, 1), 16.0, 0.1))
    assert "approximate" in text
    k, turn = dmb_effective(16.0, 0.1, (2, 1))
    assert float(k) == pytest.approx(27.543829, abs=1e-4)
    assert turn == pytest.approx(0.0554573, abs=1e-6)


# R8: directional warp.


def test_r8_directional_warp_oblique() -> None:
    text = body(dwarp((2, 1), 4.0, 0.1))
    assert (
        "vec2 i8 = vec2(wnc_$(name)($warp_intensity * psv.x, z),"
        " wnc_$(name)($warp_intensity * psv.y, z)) * 0.00390625;"
    ) in text
    assert "vec2 kv = vec2(i8.x * C, i8.y * S);" in text

    def offset(k: float, turn: float, v: tuple) -> np.ndarray:
        c, s = math.cos(2 * math.pi * turn), math.sin(2 * math.pi * turn)
        i8 = np.array([mul(k, v[0]), mul(k, v[1])], np.float32) * f32(0.00390625)
        return i8 * np.array([c, s], np.float32)

    # the vector's own length and angle, with v = (1, 1): within a level's
    # worth of offset (5 levels at < 0.1% of pixels, measured)
    a = offset(4.0, 0.1, (2, 1))
    b = offset(6.885957, 0.0554573, (1, 1))
    assert np.allclose(a, b, rtol=0, atol=1e-6)
    # isotropic: exactly intensity k p
    assert np.array_equal(offset(4.0, 0.1, (0.5, 0.5)), offset(2.0, 0.1, (1, 1)))


# R9: sharpen.


def sharpened(c, sr, sl, su, sd, k: float, v: tuple) -> np.ndarray:
    """Return _sharpen's GLSL value (before its clamp), in float32."""
    s = sr + sl + su + sd
    kx, ky = mul(k, v[0]), mul(k, v[1])
    if f32(v[0]) == f32(v[1]):
        return c + kx * (f32(4.0) * c - s)
    return c + kx * (f32(2.0) * c - (sr + sl)) + ky * (f32(2.0) * c - (su + sd))


def test_r9_sharpen() -> None:
    text = body(sharpen((2, 1)))
    assert "? c + ($intensity * psv.x) * (4.0 * c - s)" in text
    assert (
        ": c + ($intensity * psv.x) * (2.0 * c - (sr + sl))"
        " + ($intensity * psv.y) * (2.0 * c - (su + sd));"
    ) in text
    # on noise: out(2, 1) = out(1, 1) + out(2, 2) - out(1, 2) where unclamped
    rng = np.random.default_rng(9)
    c, sr, sl, su, sd = (rng.random(4096, dtype=np.float32) for _ in range(5))
    out = {
        v: sharpened(c, sr, sl, su, sd, 1.0, v)
        for v in ((2, 1), (1, 1), (2, 2), (1, 2))
    }
    keep = np.all([(o >= 0) & (o <= 1) for o in out.values()], axis=0)
    assert keep.sum() > 100
    lhs = out[2, 1][keep]
    rhs = (out[1, 1] + out[2, 2] - out[1, 2])[keep]
    assert np.max(np.abs(lhs - rhs)) < 0.5 / 65535  # 0 levels at 16 bits
    # isotropic: as intensity k p
    assert np.array_equal(
        sharpened(c, sr, sl, su, sd, 1.0, (2, 2)),
        sharpened(c, sr, sl, su, sd, 2.0, (1, 1)),
    )


# R10: normal.


def test_r10_normal() -> None:
    text = body(normal((2, 1)))
    assert "-(($intensity * psv.x) * ($pass_size.x / 256.0)) * g.x" in text
    assert "-(($intensity * psv.y) * ($pass_size.x / 256.0)) * g.y" in text

    def slopes(dx: float, dy: float, v: tuple) -> tuple:
        return (
            -(mul(1.0, v[0]) * f32(1.0)) * f32(dx),
            -(mul(1.0, v[1]) * f32(1.0)) * f32(dy),
        )

    assert slopes(0.3, 0.0, (2, 1)) == slopes(0.3, 0.0, (2, 2))
    assert slopes(0.0, 0.3, (2, 1)) == slopes(0.0, 0.3, (1, 1))


# R11: distance.


def test_r11_distance() -> None:
    *passes, final = bodies(distance((2, 1), 8.0))
    jfa = [p for p in passes if "tdistv_" in p]
    assert len(jfa) == len(passes) - 1  # all but the seed's
    assert all("vec2 pss = vec2(2.0, 1.0);" in p for p in jfa)
    assert "float r = max((8.0 * psv.x) * ($pass_size.x / 256.0), 1e-6);" in final
    assert "q = length(dd / rr);" in final
    # the steps reach D v.x = 16 texels (256 pixels): 16 first
    assert "* 0.0625));" in jfa[0]
    n = 256.0

    def falloff(dx: float, dy: float, dist: float, v: tuple) -> float:
        r = max(float(mul(dist, v[0])) * n / 256, 1e-6)
        if v[0] == v[1]:
            q = math.hypot(dx, dy) / r
        else:
            rr = (float(mul(dist, v[0])) * n / 256, float(mul(dist, v[1])) * n / 256)
            q = math.hypot(dx / rr[0], dy / rr[1])
        return min(max(1.0 - q, 0.0), 1.0)

    # isotropic: distance 1 at (2, 2) is distance 2 at (1, 1)
    for dx, dy in ((0, 0), (1, 0), (1, 1), (3, 2)):
        assert falloff(dx, dy, 1.0, (2, 2)) == falloff(dx, dy, 2.0, (1, 1))
    # (2, 1): the field at (2j, i) is the (1, 1) field at (j, i)
    for i in range(-10, 11):
        for j in range(-10, 11):
            a = falloff(2 * j, i, 8.0, (2, 1))
            b = falloff(j, i, 8.0, (1, 1))
            assert abs(a - b) * 65535 < 4


# R12: a pixel-size function.


def test_r12_a_pixel_size_function() -> None:
    n = node(*blur("DEF2"))
    assert translate.pixel_size_static(n) == [2.0, 2.0]
    assert not translate.pixel_size_unit(n)
    x, y = bodies(blur("DEF2"))
    # live: the function's helper, called, not a constant
    assert re.search(r"\(\$intensity \* f\d+_\$\(name\)\(uv\)\.x\)", x)
    assert re.search(r"\(\$intensity \* f\d+_\$\(name\)\(uv\)\.y\)", y)
    assert "vec2(2.0, 2.0)" not in x + y
    # the default alone is (1, 1) at every size: nothing to keep live
    assert translate.pixel_size_unit(node(*blur("DEF")))
    # one that follows the size stays live (and is 2 at 256 pixels)
    live = node(*warp("LOG4"))
    assert not translate.pixel_size_unit(live)
    assert translate.pixel_size_static(live) == [2.0, 2.0]
    assert re.search(r"vec2 psv = f\d+_\$\(name\)\(uv\);", body(warp("LOG4")))


# R13: non-square (the default block's vector; Material Maker's images are square).


def test_r13_non_square_blur() -> None:
    blk = decode(DEFAULT)
    for w, h in ((256, 128), (128, 256)):
        v = evaluate(
            blk, {}, {1: [float(w), float(h)], 3: [math.log2(w), math.log2(h)]}
        )
        # intensity 2: k v_a N_a / 256 texels along each axis, 1 each
        assert [2.0 * v[0] * w / 256, 2.0 * v[1] * h / 256] == [1.0, 1.0]
    assert evaluate(blk, {}, {1: [256.0, 128.0], 3: [8.0, 7.0]}) == [0.5, 1.0]


# Plugin nodes: no pixel-size knob, so a vector not (1, 1) stays inline (live).


def test_a_pixel_sized_filter_is_not_a_plugin_node() -> None:
    maker = plugin.Maker(8)
    assert maker(node(*blur((2, 1))), "n0", 8, None) is None
    assert maker.inline == {"blur: its pixel size is not (1, 1)": 1}
    assert maker(node(*blur("DEF2")), "n1", 8, None) is None
    assert maker(node(*blur("DEF")), "n2", 8, None) is not None
    assert maker(node(*warp((1, 1))), "n3", 8, None) is not None
    assert maker.made == {"sbs_blur_gray": 1, "sbs_warp_gray": 1}


# Every translation compiles (Material Maker's placeholders stubbed).


HEADS = [
    blur((2, 1)), blur("DEF2"), warp((0.0444, 0.0444), 38.0), warp("LOG4"),
    dmb((2, 1), 16.0, 0.1), dwarp((2, 1), 4.0, 0.1), sharpen((2, 1)),
    normal((2, 1)), distance((2, 1), 8.0),
]  # fmt: skip


def stubbed(nd: dict) -> str:
    """Return a shader node's GLSL as a fragment shader, its $ names stubbed."""
    sm = nd["shader_model"]
    src = sm.get("global", "") + sm["instance"]
    stubs = []
    for port in sm["inputs"]:
        t = "vec4" if port["type"] == "rgba" else "float"
        stubs.append(f"{t} in_{port['name']}(vec2 uv) {{ return {t}(0.5); }}")
        src = src.replace(f"${port['name']}(", f"in_{port['name']}(")
    src = src.replace("$(name)", "nm").replace("$pass_size", "vec2(256.0)")
    for p in sorted(set(re.findall(r"\$(\w+)", src))):
        stubs.append(f"float p_{p} = 1.0;")
        src = src.replace(f"${p}", f"p_{p}")
    return (
        "#version 450\n" + "\n".join(stubs) + "\n" + src
        + "layout(location = 0) out vec4 color;\n"
        + "void main() { color = vec4(nm_fct(gl_FragCoord.xy / 256.0)); }\n"
    )  # fmt: skip


@pytest.mark.parametrize("head", HEADS)
def test_the_translations_compile(head: tuple, tmp_path: Path) -> None:
    validator = shutil.which("glslangValidator")
    if validator is None:
        pytest.skip("glslangValidator not installed")
    for k, nd in enumerate(shaders(translate.node_shader(node(*head), "n0", size=8))):
        source = stubbed(nd)
        f = tmp_path / f"s{k}.frag"
        f.write_text(source, encoding="utf-8")
        done = subprocess.run(  # noqa: S603
            [validator, str(f)], capture_output=True, text=True, check=False
        )
        assert done.returncode == 0, done.stdout + source
