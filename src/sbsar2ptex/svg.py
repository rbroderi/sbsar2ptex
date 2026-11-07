"""The SVG node (sbsasm ops 10 and 11): its vector record and how it renders.

docs/svg-filter.md. The cooker parses, flattens, triangulates and clips the
SVG; the node holds only the result, a tagged record of paints, triangle
strips and polylines in document fractions (section 3). Rendering (section
4) draws them in record order over the background, each covered pixel
replaced by its paint, with no antialiasing: a pixel is covered when its
centre is inside a triangle of vertices snapped to 1/256 pixel, by the
top-left rule; a polyline is drawn as one-pixel lines.

``render`` is a reference rasterizer (numpy, exact integers); ``glsl``
writes the same computation as a Material Maker shader body, live in the
output size, with SGN_GLSL (shared global code) for its coverage tests.
"""

import math
import struct
from dataclasses import dataclass
from dataclasses import field

import numpy as np

from sbsar2ptex.sbsasm import BASE
from sbsar2ptex.sbsasm import UnsupportedError

TAG_STRIP, TAG_LINE, TAG_SOLID, TAG_GRADIENT, TAG_PLANE = 1, 2, 3, 4, 5
SNAP = 256
"""Sub-pixel steps of a snapped vertex (4.2: 8 sub-pixel bits)."""


@dataclass
class Paint:
    """A paint and what it draws (section 3.1)."""

    color: tuple[int, int, int] | None = None
    """A solid paint's bytes (R, G, B); None for a gradient."""
    stops: list[tuple[int, tuple[int, int, int, int]]] = field(default_factory=list)
    """A gradient's stops: (offset x 65535, (R, G, B, A))."""
    plane: tuple[float, float, float] | None = None
    """A gradient's t = a u + b v + c at document fractions (u, v)."""
    items: list[tuple[str, list[tuple[int, int]]]] = field(default_factory=list)
    """Its strips ("strip") and polylines ("line"): vertex words as (X, Y)."""


def _f32(w: int) -> float:
    return struct.unpack("<f", struct.pack("<I", w))[0]


def read_record(data: bytes, offset: int) -> list[Paint] | None:  # noqa: PLR0911
    """Return the vector record at ``offset`` (section 3.1); None if malformed.

    A word's low 3 bits are its tag, the rest its value; the word 0 ends it.
    A strip or polyline before any paint, a plane anywhere but right after
    a gradient's stops, a gradient with no plane, tags 0, 6 and 7 with a
    value, or a record past the data, are malformed.
    """
    at = offset + BASE
    paints: list[Paint] = []
    pending_plane = False

    def word() -> int:
        nonlocal at
        if at < 0 or at + 4 > len(data):
            raise IndexError
        (w,) = struct.unpack_from("<I", data, at)
        at += 4
        return w

    try:
        while True:
            w = word()
            if w == 0:
                return None if pending_plane else paints
            tag, value = w & 7, w >> 3
            if pending_plane and tag != TAG_PLANE:
                return None
            if tag == TAG_SOLID:
                paints.append(Paint(color=(value & 0xFF, value >> 8 & 0xFF,
                                           value >> 16 & 0xFF)))  # fmt: skip
            elif tag == TAG_GRADIENT:
                stops = []
                for _ in range(value):
                    s0, s1 = word(), word()
                    rgba = (s0 >> 16 & 0xFF, s0 >> 24, s1 & 0xFF, s1 >> 8 & 0xFF)
                    stops.append((s0 & 0xFFFF, rgba))
                if not stops:
                    return None
                paints.append(Paint(stops=stops))
                pending_plane = True
            elif tag == TAG_PLANE:
                if not pending_plane or value != 3:  # noqa: PLR2004
                    return None
                paints[-1].plane = (_f32(word()), _f32(word()), _f32(word()))
                pending_plane = False
            elif tag in (TAG_STRIP, TAG_LINE):
                if not paints:
                    return None
                verts = [(v & 0xFFFF, v >> 16) for v in (word() for _ in range(value))]
                paints[-1].items.append(
                    ("strip" if tag == TAG_STRIP else "line", verts)
                )
            else:
                return None
    except IndexError:
        return None


def gray_record(paints: list[Paint]) -> None:
    """Refuse what a grayscale node's record holds that was not measured.

    Its solid paints hold the gray in all three bytes (section 3.1); its
    gradients were never probed (section 7).
    """
    if any(p.color is None for p in paints):
        msg = "a gradient in a grayscale SVG node (not probed)"
        raise UnsupportedError(msg)


# Geometry: the f32 vertex chain (4.2 step 1), then the snap.


def f32(x: float) -> float:
    """Return ``x`` rounded to float32."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def unit(w: int) -> float:
    """Return a vertex word's (ndc + 1): f32(f32(f32(w) / 65535) * 2 - 1) + 1.

    Times the output size / 2 (a power of two: exact) it is the vertex's
    pixel position, so it is computed here once for every size.
    """
    s = f32(float(w) / 65535.0)
    ndc = f32(s * 2.0 - 1.0)
    return f32(ndc + 1.0)


def snapped(w: int, size: int) -> int:
    """Return a vertex coordinate in 1/256 pixels at an output ``size`` (pixels).

    round(px 256), px = unit(w) size / 2: ties (never observed) up.
    """
    return math.floor(unit(w) * size * (SNAP // 2) + 0.5)


def sgn_split(a: int, b: int, c: int, d: int) -> int:
    """Return the sign of a b - c d in 32-bit integer steps (|each| < 2^22).

    The GLSL helper's arithmetic (SGN_GLSL), mirrored for tests: each
    factor split at 2^11, the partial sums carried, no step past 2^31.
    """

    def split(x: int) -> tuple[int, int]:
        lo = x & 2047
        return (x - lo) // 2048, lo

    (ah, al), (bh, bl), (ch, cl), (dh, dl) = split(a), split(b), split(c), split(d)
    t2 = ah * bh - ch * dh
    t1 = ah * bl + al * bh - ch * dl - cl * dh
    t0 = al * bl - cl * dl
    c0 = t0 & 2047
    t1 += (t0 - c0) // 2048
    c1 = t1 & 2047
    t2 += (t1 - c1) // 2048
    if t2:
        return 1 if t2 > 0 else -1
    return 1 if (c1 | c0) else 0


SGN_GLSL = """int sbs_svg_sgn(int a, int b, int c, int d) {
	int al = a & 2047; int ah = (a - al) / 2048;
	int bl = b & 2047; int bh = (b - bl) / 2048;
	int cl = c & 2047; int ch = (c - cl) / 2048;
	int dl = d & 2047; int dh = (d - dl) / 2048;
	int t2 = ah * bh - ch * dh;
	int t1 = ah * bl + al * bh - ch * dl - cl * dh;
	int t0 = al * bl - cl * dl;
	int c0 = t0 & 2047;
	t1 += (t0 - c0) / 2048;
	int c1 = t1 & 2047;
	t2 += (t1 - c1) / 2048;
	return t2 != 0 ? (t2 > 0 ? 1 : -1) : ((c1 | c0) != 0 ? 1 : 0);
}
bool sbs_svg_edge(ivec2 a, ivec2 b, ivec2 c) {
	int e = sbs_svg_sgn(b.x - a.x, c.y - a.y, b.y - a.y, c.x - a.x);
	return e > 0 || (e == 0 && (b.y < a.y || (b.y == a.y && b.x > a.x)));
}
bool sbs_svg_tri(ivec2 p0, ivec2 p1, ivec2 p2, ivec2 c) {
	int o = sbs_svg_sgn(p1.x - p0.x, p2.y - p0.y, p1.y - p0.y, p2.x - p0.x);
	if (o == 0) return false;
	if (o < 0) { ivec2 t = p1; p1 = p2; p2 = t; }
	return sbs_svg_edge(p1, p2, c) && sbs_svg_edge(p2, p0, c)
		&& sbs_svg_edge(p0, p1, c);
}
bool sbs_svg_major(int a0, int a1, int b0, int b1, int ca, int cb) {
	if (a1 < a0) { int t = a0; a0 = a1; a1 = t; t = b0; b0 = b1; b1 = t; }
	if (a1 == a0 || ca < a0 || ca >= a1) return false;
	int da = a1 - a0, db = b1 - b0;
	int lo = cb - 128;
	return sbs_svg_sgn(b0 - lo, da, -db, ca - a0) > 0
		&& sbs_svg_sgn(b0 - lo - 256, da, -db, ca - a0) <= 0;
}
bool sbs_svg_line(ivec2 a, ivec2 b, ivec2 c) {
	if (abs(b.x - a.x) >= abs(b.y - a.y))
		return sbs_svg_major(a.x, b.x, a.y, b.y, c.x, c.y);
	return sbs_svg_major(a.y, b.y, a.x, b.x, c.y, c.x);
}
"""
"""The coverage tests (4.2, 4.3) in exact integer steps, in 1/256 pixels:
a pixel's centre ``c`` is (256 i + 128, 256 j + 128)."""


def _edge_cover(ax, ay, bx, by, cx, cy) -> np.ndarray:
    e = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    top_left = (by < ay) | ((by == ay) & (bx > ax))
    return (e > 0) | ((e == 0) & top_left)


def triangle_mask(
    p: list[tuple[int, int]], cx: np.ndarray, cy: np.ndarray
) -> np.ndarray:
    """Return the pixels whose centres a snapped triangle covers (4.2)."""
    (x0, y0), (x1, y1), (x2, y2) = p
    o = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    if o == 0:
        return np.zeros(cx.shape, dtype=bool)
    if o < 0:
        x1, y1, x2, y2 = x2, y2, x1, y1
    return (
        _edge_cover(x1, y1, x2, y2, cx, cy)
        & _edge_cover(x2, y2, x0, y0, cx, cy)
        & _edge_cover(x0, y0, x1, y1, cx, cy)
    )


def _major(a0, a1, b0, b1, ca, cb) -> np.ndarray:
    if a1 < a0:
        a0, a1, b0, b1 = a1, a0, b1, b0
    if a1 == a0:
        return np.zeros(ca.shape, dtype=bool)
    da, db = a1 - a0, b1 - b0
    lo = cb - SNAP // 2  # the row's top, 256 j
    num = (b0 - lo) * da + db * (ca - a0)  # (b(ca) - 256 j) da
    return (ca >= a0) & (ca < a1) & (num > 0) & (num - SNAP * da <= 0)


def line_mask(a, b, cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    """Return the pixels a one-pixel segment draws (4.3).

    Along its major axis, each pixel whose centre is in [min, max) of the
    segment, in the row (column) ceil(y(centre)) - 1.
    """
    (ax, ay), (bx, by) = a, b
    if abs(bx - ax) >= abs(by - ay):
        return _major(ax, bx, ay, by, cx, cy)
    return _major(ay, by, ax, bx, cy, cx)


def ramp(stops: list[tuple[int, tuple[int, int, int, int]]]) -> list[tuple[int, ...]]:
    """Return a gradient's 256-entry ramp (4.4 step 2): (R, G, B) bytes."""
    offs = [o / 65535 for o, _ in stops]
    out = []
    for m in range(256):
        t = m / 255
        if t <= offs[0]:
            c = stops[0][1][:3]
        elif t >= offs[-1]:
            c = stops[-1][1][:3]
        else:
            k = max(i for i in range(len(offs) - 1) if offs[i] <= t)
            lo, hi = stops[k][1][:3], stops[k + 1][1][:3]
            w = (t - offs[k]) / (offs[k + 1] - offs[k])
            c = tuple(lo[i] + (hi[i] - lo[i]) * w for i in range(3))
        out.append(tuple(math.floor(x + 0.5) for x in c))
    return out


def nan_plane(paint: Paint) -> bool:
    """Whether a gradient's plane is NaN (objectBoundingBox: 4.4)."""
    return paint.plane is not None and any(math.isnan(x) for x in paint.plane)


def render(
    paints: list[Paint],
    wo: int,
    ho: int,
    background: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    """Return the record drawn at ``wo`` x ``ho`` over ``background`` (section 4).

    Rows from the top, RGBA in [0, 1]; a covered pixel is its paint's
    colour, alpha 1.
    """
    out = np.empty((ho, wo, 4))
    out[:] = background
    jj, ii = np.mgrid[0:ho, 0:wo]
    cx, cy = ii * SNAP + SNAP // 2, jj * SNAP + SNAP // 2
    for paint in paints:
        color = _paint_image(paint, wo, ho)
        for kind, verts in paint.items:
            pts = [(snapped(x, wo), snapped(y, ho)) for x, y in verts]
            if kind == "strip":
                masks = [
                    triangle_mask(pts[i : i + 3], cx, cy) for i in range(len(pts) - 2)
                ]
            else:
                masks = [
                    line_mask(pts[i], pts[i + 1], cx, cy) for i in range(len(pts) - 1)
                ]
            for mask in masks:
                out[mask] = color[mask]
    return out


def _paint_image(paint: Paint, wo: int, ho: int) -> np.ndarray:
    """Return a paint's colour at every pixel, RGBA."""
    img = np.ones((ho, wo, 4))
    if paint.color is not None:
        img[..., :3] = np.array(paint.color) / 255
        return img
    if nan_plane(paint):
        img[..., :3] = np.array(paint.stops[0][1][:3]) / 255
        return img
    a, b, c = (np.float32(x) for x in paint.plane)
    jj, ii = np.mgrid[0:ho, 0:wo]
    u = ((ii + 0.5) / wo).astype(np.float32)
    v = ((jj + 0.5) / ho).astype(np.float32)
    t = (a * u + b * v).astype(np.float32) + c
    x = np.clip(np.float32(256) * t.astype(np.float32) - np.float32(0.5), 0, 255)
    m = np.floor(x).astype(int)
    fr = (x - m).astype(np.float32)
    table = np.array(ramp(paint.stops), dtype=np.float32)
    lo, hi = table[m], table[np.minimum(m + 1, 255)]
    value = lo * (1 - fr[..., None]) + hi * fr[..., None]
    img[..., :3] = np.round(value) / 255  # numpy rounds halves to even
    return img


def _num(x: float) -> str:
    """Return a GLSL float literal keeping ``x``'s float32 bits."""
    for digits in range(1, 10):
        text = f"{x:.{digits}g}"
        if f32(float(text)) == x:
            break
    return text if any(ch in text for ch in ".e") else f"{text}.0"


def glsl(
    paints: list[Paint], width: str, height: str, *, color: bool
) -> tuple[str, str]:
    """Return the record's GLSL: helpers and the lines of a body.

    The body leaves ``col`` (a vec4 the caller declares, the background)
    painted at the pixel ``uv``; ``width`` and ``height`` are the output's
    size in pixels (GLSL floats, live). An empty record adds nothing.
    """
    verts: list[tuple[int, int]] = []
    index: dict[tuple[int, int], int] = {}
    items: list[tuple[int, int, int, int]] = []

    def vertex(v: tuple[int, int]) -> int:
        if v not in index:
            index[v] = len(verts)
            verts.append(v)
        return index[v]

    for p, paint in enumerate(paints):
        for kind, vs in paint.items:
            ids = [vertex(v) for v in vs]
            if kind == "strip":
                for i in range(len(ids) - 2):
                    tri = ids[i : i + 3]
                    if len(set(tri)) == 3:  # (a repeated vertex: no area at any size)  # noqa: PLR2004
                        items.append((*tri, p))
            else:
                items += [(ids[i], ids[i + 1], -1, p) for i in range(len(ids) - 1)]
    if not items:
        return "", ""
    qs = ", ".join(f"vec2({_num(unit(x))}, {_num(unit(y))})" for x, y in verts)
    its = ", ".join(f"ivec4({a}, {b}, {c}, {p})" for a, b, c, p in items)
    branches = []
    for p, paint in enumerate(paints):
        if not paint.items:
            continue
        if paint.color is not None or nan_plane(paint):
            rgb = paint.color if paint.color is not None else paint.stops[0][1][:3]
            value = f"vec4({', '.join(_num(c / 255) for c in rgb)}, 1.0)"
            branches.append(f"\tif (p == {p}) return {value};")
            continue
        table = ", ".join(f"vec3({r}.0, {g}.0, {b}.0)" for r, g, b in ramp(paint.stops))
        a, b, c = (_num(f32(x)) for x in paint.plane)
        branches += [
            f"\tif (p == {p}) {{",
            f"\t\tconst vec3 rp[256] = vec3[256]({table});",
            f"\t\tprecise float t = {a} * uv.x + {b} * uv.y + {c};",
            "\t\tfloat x = clamp(256.0 * t - 0.5, 0.0, 255.0);",
            "\t\tint m = int(floor(x));",
            "\t\tfloat f = x - float(m);",
            "\t\tvec3 v = mix(rp[m], rp[min(m + 1, 255)], f);",
            "\t\treturn vec4(roundEven(v) / 255.0, 1.0);",
            "\t}",
        ]
    helpers = (
        "ivec2 sbs_svg_snap_$(name)(vec2 q, vec2 n) {\n"
        "\treturn ivec2(floor(q * (n * 128.0) + 0.5));\n}\n"
        "vec4 sbs_svg_paint_$(name)(int p, vec2 uv) {\n"
        + "\n".join(branches)
        + "\n\treturn vec4(0.0);\n}\n"
    )
    body = "\n".join(
        [
            f"const vec2 svq[{len(verts)}] = vec2[{len(verts)}]({qs});",
            f"const ivec4 svi[{len(items)}] = ivec4[{len(items)}]({its});",
            f"vec2 svn = vec2({width}, {height});",
            "ivec2 svc = ivec2(floor(uv * svn)) * 256 + 128;",
            f"for (int k = {len(items) - 1}; k >= 0; k--) {{",
            "\tivec4 it = svi[k];",
            "\tivec2 a = sbs_svg_snap_$(name)(svq[it.x], svn);",
            "\tivec2 b = sbs_svg_snap_$(name)(svq[it.y], svn);",
            (
                "\tbool hit = it.z < 0 ? sbs_svg_line(a, b, svc)"
                " : sbs_svg_tri(a, b, sbs_svg_snap_$(name)(svq[it.z], svn), svc);"
            ),
            (
                "\tif (hit) { col = sbs_svg_paint_$(name)(it.w,"
                " (vec2(svc) / 256.0) / svn); break; }"
            ),
            "}",
        ]
    )
    del color
    return helpers, body
