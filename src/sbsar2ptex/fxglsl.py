"""FX-Map drawing in GLSL: sbsar2ptex.fxdraw's model, for a Material Maker shader.

Every function is named with ``$(name)``, the shader node's. A map's
shader maps its pixel to pattern space (``PIXEL_LINES``), starts its
accumulator at the background, draws each pattern with ``fxdraw_$(name)``
(its tile copies, coverage, value, color and blending) in Substance's
order, and stores the result (``fxstore_$(name)``) in the map's format.

Image patterns read their pattern input's buffer at texel centers, as the
graph stores it (8 or 16 bits). Bilinear filtering in a mip level
(filtering 0) reads levels 1 and up from a mip atlas: a buffer as large as
the image whose top half holds every level side by side (level L of side
M / 2^L at x = M - 2 M / 2^L), each the stored 2x2 means of the one before
(``mip_body``), built in two passes (levels 1 to MIP_PASS from the image,
the rest from level MIP_PASS) so that no texel sums more than 4^MIP_PASS.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from sbsar2ptex import fxdraw
from sbsar2ptex import fxmap

FLOAT = None
"""A map's format when neither 8 nor 16 bits (floats: no store)."""

MIP_PASS = 6
"""Levels an atlas pass derives from its base level."""


def num(x: float) -> str:
    """Return a GLSL float literal of ``x``."""
    s = repr(float(x))
    if "e" in s or "E" in s:
        return f"{float(x):.9e}"
    return s if "." in s else s + ".0"


def vec2(x: float, y: float) -> str:
    """Return a GLSL vec2 literal."""
    return f"vec2({num(x)}, {num(y)})"


@dataclass
class Layout:
    """Where a map draws: its image, regions and tiling (constants)."""

    size: int
    """log2 of the image's (and its pattern inputs') side."""
    render: Sequence[float] = (0.0, 1.0, 0.0, 1.0)
    tiling: Sequence[float] = (0.0, 1.0, 0.0, 1.0)
    tile: tuple[bool, bool] = (True, True)
    color: bool = False
    bits: int | None = 16
    """The map's format: 8, 16, or FLOAT."""
    amd: bool = False
    """Store and mip rules: AMD's measured ones, else NVIDIA's (predicted)."""


def val_glsl() -> str:
    """Return ``fxval_$(name)``: the float32 nearest k / 65535, and its direction.

    For integers k >= 0 in floats (any of them below 2^24); ``up`` is 1
    where the float32 is not below k / 65535.
    """
    return (
        "vec4 fxval_$(name)(vec4 k, out vec4 up) {\n"
        "\tuvec4 n = uvec4(max(k, vec4(1.0)));\n"
        "\tivec4 m = findMSB(n);\n"
        "\tuvec4 a = n << uvec4(23 - m);\n"
        # The quotient reaching 2^24: it is in the next binade.
        "\tuvec4 hz = (a + a / 65535u) >> 24u;\n"
        "\ta = a >> hz;\n"
        "\tuvec4 r = a % 65535u;\n"
        "\tuvec4 big = (2u * r) / 65536u;\n"
        "\tup = vec4(max(big, uvec4(1u) - min(r, uvec4(1u))));\n"
        "\tuvec4 bits = (uvec4(m + 111) << 23u) + (hz << 23u) + a + a / 65535u + big"
        " - 8388608u;\n"
        "\treturn uintBitsToFloat(bits) * min(k, vec4(1.0));\n"
        "}\n"
        "vec4 fxv_$(name)(vec4 k) {\n"
        "\tvec4 up;\n"
        "\treturn fxval_$(name)(k, up);\n"
        "}\n"
    )


def levels_glsl(bits: int | None) -> str:
    """Return ``fxlv_$(name)``/``fxlvval_$(name)``: a value's level, a level's value.

    A stored value's integer level (8 or 16 bits), and back: the float32 a
    texture read of that level returns. Floats pass as they are.

    ``fxlvin_$(name)`` is an input image's level: Material Maker hands
    images over as half floats (11 bits, truncated), so a value within a
    half float's step of an 8-bit level (an 8-bit file's, as Substance
    stores it) is that level exactly; others go to the nearest level.
    """
    if bits == 16:  # noqa: PLR2004
        to, back = "floor(v * 65535.0 + 0.5)", "fxv_$(name)(k)"
        near = (
            "\tvec4 k8 = floor(v * 255.0 + 0.5);\n"
            "\tvec4 snap = step(abs(v - k8 / 255.0), abs(v) / 1024.0 + 1e-7);\n"
            "\treturn mix(floor(v * 65535.0 + 0.5), k8 * 257.0, snap);\n"
        )
    elif bits == 8:  # noqa: PLR2004
        to, back = "floor(v * 255.0 + 0.5)", "fxv_$(name)(k * 257.0)"
        near = f"\treturn {to};\n"
    else:
        to, back = "v", "k"
        near = "\treturn v;\n"
    return (
        f"vec4 fxlv_$(name)(vec4 v) {{\n\treturn {to};\n}}\n"
        f"vec4 fxlvin_$(name)(vec4 v) {{\n{near}}}\n"
        f"vec4 fxlvval_$(name)(vec4 k) {{\n\treturn {back};\n}}\n"
    )


def store_glsl(layout: Layout) -> str:
    """Return ``fxstore_$(name)(vec4)``: the accumulator stored, as its value.

    Clamped, then to the map's format's level as the GPU stores it
    (fxdraw.store), returned as that level's value.
    """
    if layout.bits == 16:  # noqa: PLR2004
        if layout.amd:  # floor(65535 v + 0.5) = floor(65536 v) + floor(f - v + 0.5)
            body = (
                "\tvec4 x = v * 65536.0;\n"
                "\tvec4 k = floor(x);\n"
                "\tvec4 f = x - k;\n"
                "\tk += step(v, f - 0.5) + step(v, f + 0.5) - 1.0;\n"
            )
        else:  # x = 16 q + r, v truncated to 2^-20: (65535 x + 2^19) / 2^20
            body = (
                "\tvec4 x = floor(v * 1048576.0);\n"
                "\tvec4 q = floor(x / 16.0);\n"
                "\tvec4 k = q + floor(((x - 16.0 * q) * 65536.0 + 524288.0 - x)"
                " / 1048576.0);\n"
            )
        out = "fxv_$(name)(k)"
    elif layout.bits == 8:  # noqa: PLR2004
        if layout.amd:  # through a half float's 11 bits, truncated
            body = (
                "\tvec4 h = uintBitsToFloat(floatBitsToUint(v) & 4294959104u);\n"
                "\tvec4 k = floor(h * 255.0 + 0.5);\n"
            )
        else:
            body = "\tvec4 k = floor((floor(v * 4096.0) * 255.0 + 2047.0) / 4096.0);\n"
        out = "fxv_$(name)(k * 257.0)"
    else:
        body, out = "", "v"
    return (
        "vec4 fxstore_$(name)(vec4 v) {\n"
        "\tv = clamp(v, 0.0, 1.0);\n" + body + f"\treturn {out};\n}}\n"
    )


def mean_glsl(bits: int | None, *, amd: bool) -> str:
    """Return ``fxmean_$(name)``: the next mip level's level of four levels.

    fxdraw.mip_mean, in floats holding integers.
    """
    if bits == 16:  # noqa: PLR2004
        if amd:  # a half goes the way float32((q + 1/2) / 65535) rounded
            tie = (
                "\tvec4 up;\n"
                "\tfxval_$(name)(2.0 * q + 1.0, up);\n"
                "\treturn q + step(2.5, r) + (step(1.5, r) - step(2.5, r)) * up;\n"
            )
        else:
            tie = "\treturn q + step(1.5, r);\n"
        body = "\tvec4 s = a + b + c + d;\n\tvec4 q = floor(s * 0.25);\n"
        body += "\tvec4 r = s - 4.0 * q;\n" + tie
    elif bits == 8:  # noqa: PLR2004
        if amd:  # s / 1020 truncated to 11 significant bits, to the nearest
            body = (
                "\tuvec4 su = uvec4(a + b + c + d);\n"
                "\tvec4 k = vec4(0.0);\n"
                "\tfor (int i = 0; i < 4; i++) {\n"
                "\t\tif (su[i] == 0u) continue;\n"
                "\t\tint j = 0;\n"
                "\t\twhile ((su[i] << uint(j)) < 1020u) j++;\n"
                "\t\tuint m = (su[i] << uint(10 + j)) / 1020u;\n"
                "\t\tk[i] = floor(float(m * 255u) * exp2(float(-10 - j)) + 0.5);\n"
                "\t}\n"
                "\treturn k;\n"
            )
        else:
            body = (
                "\tuvec4 su = uvec4(a + b + c + d);\n"
                "\treturn vec4(((su * 4096u) / 1020u * 255u + 2047u) / 4096u);\n"
            )
    else:
        body = "\treturn (a + b + c + d) * 0.25;\n"
    return f"vec4 fxmean_$(name)(vec4 a, vec4 b, vec4 c, vec4 d) {{\n{body}}}\n"


def atlas_at(size: int, level: str, t: str) -> str:
    """Return GLSL: the atlas uv of texel ``t`` (ivec2) of mip ``level`` (int)."""
    m = 1 << size
    return (
        f"(vec2(float({m} - 2 * ({m} >> {level}) + {t}.x), float({t}.y)) + 0.5)"
        f" / {num(float(m))}"
    )


def mip_body(size: int, first: int, last: int, *, copy_below: bool) -> list[str]:
    """Return a mip atlas pass's shader body: levels ``first`` to ``last``.

    Its input ``$input`` is the image itself (``first`` 1) or the atlas
    holding level ``first - 1``; with ``copy_below``, the levels under
    ``first`` are copied from it. Texels are levels' values (fxlvval).
    Needs fxlv, fxlvval, fxmean and ``mip_readers``.
    """
    m = 1 << size
    return [
        f"ivec2 px = ivec2(floor(uv * {num(float(m))}));",
        f"if (px.y >= {m // 2}) return vec4(0.0);",
        "int L = 1;",
        f"while (L < {size} && px.x >= {m} - ({m} >> L)) L++;",
        f"ivec2 t = ivec2(px.x - ({m} - 2 * ({m} >> L)), px.y);",
        f"if (t.y >= ({m} >> L)) return vec4(0.0);",
        f"if (L < {first}) return {'$input(uv)' if copy_below else 'vec4(0.0)'};",
        f"if (L > {last}) return vec4(0.0);",
        f"return fxlvval_$(name)(fxmip_$(name)(t, L - {first - 1}));",
    ]


def mip_readers(size: int, first: int, last: int) -> str:
    """Return ``fxmip_$(name)(t, d)``: texel t's level, d levels past the base.

    The base is the pass's level ``first - 1`` (the image, or the atlas's);
    its 4^d texels under t are read in Morton order, each four completed
    replaced by their mean (fxmean) one level up, as a GPU builds mips.
    """
    base = first - 1
    read = "fxlvin" if base == 0 else "fxlv"
    if base == 0:
        at = f"(vec2(t) + 0.5) / {num(float(1 << size))}"
    else:
        at = atlas_at(size, str(base), "t")
    depth = last - base
    return (
        f"vec4 fxmip0_$(name)(ivec2 t) {{\n\treturn {read}_$(name)($input({at}));\n}}\n"
        "vec4 fxmip_$(name)(ivec2 t, int d) {\n"
        f"\tvec4 v[{4 * max(depth, 1)}];\n"
        "\tvec4 m = vec4(0.0);\n"
        "\tint total = 1 << (2 * d);\n"
        "\tfor (int i = 0; i < total; i++) {\n"
        "\t\tivec2 o = ivec2(0);\n"
        "\t\tfor (int l = 0; l < d; l++) {\n"
        "\t\t\tint c = (i >> (2 * l)) & 3;\n"
        "\t\t\to += ivec2(c & 1, c >> 1) << l;\n"
        "\t\t}\n"
        "\t\tm = fxmip0_$(name)((t << d) + o);\n"
        "\t\tfor (int l = 0; l < d; l++) {\n"
        "\t\t\tint c = (i >> (2 * l)) & 3;\n"
        "\t\t\tv[4 * l + c] = m;\n"
        "\t\t\tif (c != 3) break;\n"
        "\t\t\tm = fxmean_$(name)(v[4 * l], v[4 * l + 1], v[4 * l + 2],"
        " v[4 * l + 3]);\n"
        "\t\t}\n"
        "\t}\n"
        "\treturn m;\n"
        "}\n"
    )


def mip_passes(size: int) -> list[tuple[int, int]]:
    """Return the atlas passes' level spans, (first, last) each."""
    out = []
    first = 1
    while first <= size:
        last = min(first + MIP_PASS - 1, size)
        out.append((first, last))
        first = last + 1
    return out


def draw_glsl(
    layout: Layout,
    images: Sequence[str],
    mips: Sequence[int],
) -> str:
    """Return the drawing helpers: images, levels, ``fxdraw_$(name)``, ``fxnear``.

    ``images`` are the pattern inputs' names (read as ``$name``), ``mips``
    the indices of those read in mip levels (their atlases ``$mip<k>``).
    ``fxdraw_$(name)(P, c, hs, a, id, s, mx, col, acc)`` draws a pattern
    centered at c, half sizes hs, turned a radians (fxmap.image_id, or a
    built-in id), supplement s, max/alpha blending mx, instance color col
    (gray in x), at the pixel's pattern-space point P (the pixel itself in
    the global ``fxpx_$(name)``), into acc.
    """
    n = 1 << layout.size
    nf = num(float(n))
    rx0, rx1, ry0, ry1 = layout.render
    tx0, tx1, ty0, ty1 = layout.tiling
    tw, th = tx1 - tx0, ty1 - ty0
    r0, rs = vec2(rx0, ry0), vec2(rx1 - rx0, ry1 - ry0)
    out = fxmap.PATTERN_GLSL + "vec2 fxpx_$(name);\n"
    if layout.bits is not None:
        out += val_glsl()
    out += levels_glsl(layout.bits)
    if images:
        reads = "".join(
            f"\tif (k == {k}) v = ${nm}(at);\n" for k, nm in enumerate(images)
        )
        at = atlas_at(layout.size, "L", "t")
        atlas = "".join(f"\tif (k == {k}) return $mip{k}({at});\n" for k in mips)
        out += (
            # a texel of input k's mip level L as stored, zero past its edges
            "vec4 fxtex_$(name)(int k, int L, ivec2 t) {\n"
            f"\tint ml = {n} >> L;\n"
            "\tif (any(lessThan(t, ivec2(0)))"
            " || any(greaterThanEqual(t, ivec2(ml)))) {\n"
            "\t\treturn vec4(0.0);\n\t}\n"
            "\tif (L == 0) {\n"
            f"\t\tvec2 at = (vec2(t) + 0.5) / {nf};\n"
            "\t\tvec4 v = vec4(0.0);\n"
            f"{reads}"
            "\t\treturn fxlvval_$(name)(fxlvin_$(name)(v));\n"
            "\t}\n"
            f"{atlas}"
            "\treturn vec4(0.0);\n"
            "}\n"
            # the image's filtered value at q (straight)
            "vec4 fximg_$(name)(int k, int filtering, int L, vec2 q) {\n"
            "\tvec2 u = 0.5 * q + 0.5;\n"
            f"\tif (filtering == {fxmap.NEAREST}) {{\n"
            f"\t\treturn fxtex_$(name)(k, 0, ivec2(floor(u * {nf})));\n"
            "\t}\n"
            f"\tvec2 t = u * float({n} >> L) - 0.5;\n"
            "\tvec2 b = floor(t);\n"
            "\tvec2 f = floor((t - b) * 256.0 + 0.5) / 256.0;\n"
            "\tivec2 i = ivec2(b);\n"
            "\tvec4 t00 = fxtex_$(name)(k, L, i);\n"
            "\tvec4 t10 = fxtex_$(name)(k, L, i + ivec2(1, 0));\n"
            "\tvec4 t01 = fxtex_$(name)(k, L, i + ivec2(0, 1));\n"
            "\tvec4 t11 = fxtex_$(name)(k, L, i + ivec2(1, 1));\n"
            "\treturn (1.0 - f.x) * (1.0 - f.y) * t00 + f.x * (1.0 - f.y) * t10"
            " + (1.0 - f.x) * f.y * t01 + f.x * f.y * t11;\n"
            "}\n"
            # the mip level read: the longer texel-coordinate derivative
            "int fxlevel_$(name)(vec2 hs, float ca, float sa) {\n"
            f"\tvec2 px = {rs} * 0.5;\n"
            "\tvec2 gx = px.x * vec2(ca / hs.x, sa / hs.y);\n"
            "\tvec2 gy = px.y * vec2(sa / hs.x, ca / hs.y);\n"
            "\tfloat lam = 0.5 * log2(max(dot(gx, gx), dot(gy, gy)));\n"
            f"\tfloat lq = ceil(lam * 256.0 - {num(0.5 + fxdraw.LOD_TIE)}) / 256.0;\n"
            f"\treturn int(clamp(floor(lq + 0.5), 0.0, {num(float(layout.size))}));\n"
            "}\n"
        )
    if layout.color:
        blend = (
            "\t\tacc.rgb = mx ? src.rgb + acc.rgb * (1.0 - src.a)"
            " : acc.rgb + src.rgb;\n"
            "\t\tacc.a = src.a + acc.a * (1.0 - src.a);\n"
        )
        image_value = (
            "\t\t\tvec4 c = fximg_$(name)(ik % 64, ik / 64 % 4, L, q);\n"
            "\t\t\tif (ik < 256) c.rgb *= c.a;\n"
            "\t\t\tsrc = col * c;\n"
        )
    else:
        blend = "\t\tacc.x = mx ? max(acc.x, src.x) : acc.x + src.x;\n"
        image_value = "\t\t\tsrc = col * fximg_$(name)(ik % 64, ik / 64 % 4, L, q).x;\n"
    ranges = []
    for axis, wrap, w in (("x", layout.tile[0], tw), ("y", layout.tile[1], th)):
        if wrap:  # the copies near the pixel (a margin of one more each way)
            d, ww = f"P.{axis} - c.{axis}", num(w)
            ranges.append(
                f"\tint lo{axis} = int(floor(({d} - b.{axis}) / {ww})) - 1;\n"
                f"\tint hi{axis} = int(ceil(({d} + b.{axis}) / {ww})) + 1;\n"
            )
        else:
            ranges.append(f"\tint lo{axis} = 0, hi{axis} = 0;\n")
    out += (
        "void fxdraw_$(name)(vec2 P, vec2 c, vec2 hs, float a, int id, float s,"
        " bool mx, vec4 col, inout vec4 acc) {\n"
        "\tif (hs.x == 0.0 || hs.y == 0.0) return;\n"
        "\tfloat ca = cos(a), sa = sin(a);\n"
        "\tvec2 b = vec2(abs(hs.x * ca) + abs(hs.y * sa),"
        " abs(hs.x * sa) + abs(hs.y * ca));\n"
        + "".join(ranges)
        + "\tint ik = -1 - id;\n"
        "\tint L = 0;\n"
        + (
            "\tif (id < 0 && ik / 64 % 4 == 0) L = fxlevel_$(name)(hs, ca, sa);\n"
            if images
            else ""
        )
        + "\tfor (int j = loy; j <= hiy; j++) {\n"
        "\tfor (int i = lox; i <= hix; i++) {\n"
        f"\t\tvec2 cc = c + vec2(float(i) * {num(tw)}, float(j) * {num(th)});\n"
        # drawn: the copy's box overlaps the tiling region
        f"\t\tif (!(cc.x - b.x < {num(tx1)} && cc.x + b.x > {num(tx0)}"
        f" && cc.y - b.y < {num(ty1)} && cc.y + b.y > {num(ty0)})) continue;\n"
        # its pixels: the box's corners rounded to pixel boundaries
        f"\t\tvec2 lo = floor((cc - b - {r0}) / {rs} * {nf} + 0.5);\n"
        f"\t\tvec2 hi = floor((cc + b - {r0}) / {rs} * {nf} + 0.5);\n"
        "\t\tif (any(lessThan(fxpx_$(name), lo))"
        " || any(greaterThanEqual(fxpx_$(name), hi))) continue;\n"
        "\t\tvec2 d = P - cc;\n"
        "\t\tvec2 q = vec2(d.x * ca + d.y * sa, -d.x * sa + d.y * ca) / hs;\n"
        "\t\tvec4 src;\n"
        "\t\tif (id >= 0) {\n"
        "\t\t\tfloat v;\n"
        "\t\t\tif (!fxpat_$(name)(id, q, s, v)) continue;\n"
        "\t\t\tsrc = col * v;\n"
        "\t\t\tif (id >= 15) src += s;\n"
        "\t\t} else {\n"
        "\t\t\tif (max(abs(q.x), abs(q.y)) > 1.0) continue;\n"
        + (image_value if images else "\t\t\tcontinue;\n")
        + "\t\t}\n"
        + blend
        + "\t}\n\t}\n}\n"
    )
    near = []
    for axis, wrap, w in (("x", layout.tile[0], tw), ("y", layout.tile[1], th)):
        if wrap:
            near.append(f"\td.{axis} -= round(d.{axis} / {num(w)}) * {num(w)};\n")
    out += (
        "bool fxnear_$(name)(vec2 P, vec2 p, float r) {\n"
        "\tvec2 d = P - p;\n" + "".join(near) + "\treturn dot(d, d) <= r * r;\n}\n"
    )
    return out


PIXEL_LINES = (
    "fxpx_$(name) = floor(uv * {n});",
    "vec2 P = {r0} + (fxpx_$(name) + 0.5) / {n} * {rs};",
)
"""The shader's first lines: its pixel, and the pattern-space point it shows."""


def pixel_lines(layout: Layout) -> list[str]:
    """Return PIXEL_LINES for ``layout``."""
    rx0, rx1, ry0, ry1 = layout.render
    n = num(float(1 << layout.size))
    return [
        ln.format(n=n, r0=vec2(rx0, ry0), rs=vec2(rx1 - rx0, ry1 - ry0))
        for ln in PIXEL_LINES
    ]
