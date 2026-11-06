"""Function bytecode: decoding, evaluating, and choosing a node's blocks."""

import math
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sbsar2ptex.function import BOOL
from sbsar2ptex.function import ENGINE_TRIG_GLSL
from sbsar2ptex.function import FLOAT
from sbsar2ptex.function import INT
from sbsar2ptex.function import NO_VALUE
from sbsar2ptex.function import Block
from sbsar2ptex.function import FunctionError
from sbsar2ptex.function import Instr
from sbsar2ptex.function import bounds
from sbsar2ptex.function import decode
from sbsar2ptex.function import engine_cos
from sbsar2ptex.function import engine_sin
from sbsar2ptex.function import evaluate
from sbsar2ptex.function import f32
from sbsar2ptex.function import to_glsl
from sbsar2ptex.function import while_operands


def block(*instrs: list[int]) -> list[int]:
    """Return a code block's u32 words: a u16 count, then the instructions."""
    half = [len(instrs)]
    for ins in instrs:
        half += ins
    half += [0] * (len(half) % 2)
    return [half[k] | half[k + 1] << 16 for k in range(0, len(half), 2)]


def load(kind: int, width: int, uid: int, *, pad: bool) -> list[int]:
    """Return an instruction reading graph input ``uid`` (padded to align it)."""
    ops = [0] * pad + [uid & 0xFFFF, uid >> 16]
    return [len(ops) << 10 | kind << 8 | (width - 1) << 6 | 0x02, *ops]


def test_a_bool_input_is_read_by_its_aligned_uid() -> None:
    # Count at u16 0, opcode at 1: the uid's u32 starts at u16 2, aligned...
    (aligned,) = decode(block(load(BOOL, 1, 0x1234ABCD, pad=False))).instrs
    assert aligned.u32s() == [0x1234ABCD]
    # ... and after another instruction (u16s 1-3) a u16 of padding first.
    first = load(INT, 1, 7, pad=False)
    _, padded = decode(block(first, load(BOOL, 1, 0x1234ABCD, pad=True))).instrs
    assert padded.u32s() == [0x1234ABCD]


def const(value: float) -> Instr:
    """Return a float constant instruction."""
    (word,) = struct.unpack("<I", struct.pack("<f", value))
    return Instr(op=0x00, type=FLOAT, width=1, operands=[word & 0xFFFF, word >> 16])


def test_a_dropped_sequence_result_is_its_first_parts_rounded() -> None:
    # pp_param_var's size: $outputsize + sequence(set v = (rand, rand),
    # (0, 0) dropped by the cooker). Substance renders it at 4096 x 4096 at
    # seed 0, v (0.625, 0.911), and 4096 x 2048 at seed 3, v (0.539, 0.376):
    # the size plus v to the nearest
    for v, size in (((0.625, 0.911), [12, 12]), ((0.539, 0.376), [12, 11])):
        instrs = [
            Instr(op=0x02, type=INT, width=2, operands=[7, 0]),
            const(v[0]),
            const(v[1]),
            Instr(op=0x0D, type=FLOAT, width=2, operands=[1, 2]),
            Instr(op=0x07, type=FLOAT, width=2, operands=[3, 0]),
            Instr(op=0x0C, type=INT, width=2, operands=[4, NO_VALUE]),
            Instr(op=0x12, type=INT, width=2, operands=[0, 5]),
        ]
        assert evaluate(Block(instrs), {7: [11, 11]}) == size


def test_mod_is_float32_each_step_rounded() -> None:
    # A Tile Sampler's hash, 1234 / 4325 * 3.435974e10 mod 1e6: Substance's
    # float32 a - b * floor(a / b), each step rounded (numpy's float32),
    # lands 192 from the exact (and float64) remainder, 449344.
    a = np.float32(1234.0) / np.float32(4325.0) * np.float32(3.435974e10)
    b = np.float32(1e6)
    mod = Instr(op=0x16, type=FLOAT, width=1, operands=[0, 1])
    got = evaluate(Block([const(float(a)), const(float(b)), mod]), {})
    assert got == [float(a - b * np.floor(a / b))] == [449536.0]


def test_trig_is_the_cookers_polynomial() -> None:
    # Its sin(2 pi t) = t (c1 - t^2 (c3 - t^2 (c5 - t^2 c7))) is 1 at a
    # quarter turn to float32's precision and over sin by up to 1.42e-5, at
    # t ~ 0.22 (and 0.28, folded about the quarter); Biomech's n2369 puts
    # its gaussians with it (92.4% of pixels exact at 16 bits with math's sin
    # and cos, 98.5% with these).
    assert abs(engine_sin(math.pi / 2) - 1) < 1.2e-7
    assert abs(engine_cos(0.0) - 1) < 1.2e-7
    ts = np.linspace(-0.5, 0.5, 2001)
    over = [engine_sin(2 * math.pi * t) - math.sin(2 * math.pi * t) for t in ts]
    assert min(over[1000:]) > -1e-6
    assert 1.35e-5 < max(over) < 1.5e-5
    peak = abs(float(ts[int(np.argmax(over))]))
    assert abs(min(peak, 0.5 - peak) - 0.22) < 0.01
    # cartesian(rho, theta), the FX-Map branch offsets' op, uses them
    theta = 1.3
    cart = Instr(op=0x2E, type=FLOAT, width=2, operands=[0, 1])
    got = evaluate(Block([const(1.5), const(theta), cart]), {})
    assert got == [f32(1.5 * engine_cos(f32(theta))), f32(1.5 * engine_sin(f32(theta)))]
    assert got[0] != f32(1.5 * f32(math.cos(f32(theta))))


# While loops (docs/function-ops.md section 1).

SETVAR, GETVAR, SEQ, ADD, GTEQ, GT, WHILE = 0x07, 0x04, 0x0C, 0x12, 0x20, 0x1F, 0x0B


def ins(op: int, *operands: int, kind: int = FLOAT) -> Instr:
    """Return a width-1 instruction."""
    return Instr(op=op, type=kind, width=1, operands=list(operands))


def loop_ops(init: int, cond: int, body: int, cap: int, *, pad: bool) -> list[int]:
    """Return a while's operand u16s: three indices, (padding,) the u32 cap."""
    return [init, cond, body, *[0] * pad, cap & 0xFFFF, cap >> 16]


def counting(limit: float, cap: int, *, pad: bool = False) -> Block:
    """Return section 1.8's Test A: i from 7, i += 10, k counts conditions."""
    return Block([
        const(0.0),                                   # 0
        const(7.0),                                   # 1
        ins(SETVAR, 1, 0),                            # 2  i = 7
        ins(SETVAR, 0, 1),                            # 3  k = 0
        ins(SETVAR, 0, 2),                            # 4  hidden = 0
        ins(SEQ, 2, 3),                               # 5
        ins(SEQ, 5, 4),                               # 6  init
        ins(GETVAR, 1),                               # 7
        const(1.0),                                   # 8
        ins(ADD, 7, 8),                               # 9
        ins(SETVAR, 9, 1),                            # 10 k += 1
        ins(GETVAR, 0),                               # 11
        const(limit),                                 # 12
        ins(GTEQ, 11, 12, kind=BOOL),                 # 13
        ins(SEQ, 10, 13, kind=BOOL),                  # 14 cond
        ins(GETVAR, 0),                               # 15
        const(10.0),                                  # 16
        ins(ADD, 15, 16),                             # 17
        ins(SETVAR, 17, 0),                           # 18 i += 10
        ins(SETVAR, 18, 2),                           # 19 body
        ins(WHILE, *loop_ops(6, 14, 19, cap, pad=pad)),  # 20
        ins(GETVAR, 2),                               # 21
        ins(SEQ, 20, 21),                             # 22
    ])  # fmt: skip


@pytest.mark.parametrize(
    ("limit", "cap", "pad", "value", "k", "i"),
    [
        (0.0, 0, False, 0.0, 1.0, 7.0),
        (30.0, 0, False, 37.0, 4.0, 37.0),
        (1000.0, 3, False, 37.0, 3.0, 37.0),
        (1000.0, 1, False, 17.0, 1.0, 17.0),
        (25.0, 0, False, 27.0, 3.0, 27.0),
        (1000.0, 3, True, 37.0, 3.0, 37.0),
    ],
)
def test_a_while_loop_runs_as_substance(
    *,
    limit: float,
    cap: int,
    pad: bool,
    value: float,
    k: float,
    i: float,
) -> None:
    variables: dict[int, list] = {}
    assert evaluate(counting(limit, cap, pad=pad), {}, variables=variables) == [value]
    assert variables[1] == [k]
    assert variables[0] == [i]


def test_while_operands_with_and_without_padding() -> None:
    # The two encodings section 1.2 saw, decoded from words.
    for ops, want in (([7, 15, 20, 0, 3, 0], (7, 15, 20, 3)),
                      ([4, 12, 19, 0, 0], (4, 12, 19, 0))):  # fmt: skip
        opcode = len(ops) << 10 | FLOAT << 8 | WHILE
        (got,) = decode(block([opcode, *ops])).instrs
        assert got.name == "while"
        assert while_operands(got) == want


def test_the_body_reads_the_conditions_value() -> None:
    # Section 1.8's Test B: the body adds the condition's own #9.
    blk = Block([
        const(0.0),                       # 0
        ins(SETVAR, 0, 1),                # 1  v1 = 0
        ins(SETVAR, 0, 3),                # 2  v3 = 0
        ins(SETVAR, 0, 2),                # 3  hidden = 0
        ins(SEQ, 1, 2),                   # 4
        ins(SEQ, 4, 3),                   # 5  init
        ins(GETVAR, 1),                   # 6
        const(1.0),                       # 7
        ins(ADD, 6, 7),                   # 8
        ins(SETVAR, 8, 1),                # 9  #c: v1 += 1
        const(3.0),                       # 10
        ins(GT, 9, 10, kind=BOOL),        # 11 cond
        ins(GETVAR, 3),                   # 12
        ins(ADD, 12, 9),                  # 13
        ins(SETVAR, 13, 3),               # 14 v3 += #c
        ins(SETVAR, 14, 2),               # 15 body
        ins(WHILE, 5, 11, 15, 0, 0),      # 16
        ins(GETVAR, 2),                   # 17
        ins(SEQ, 16, 17),                 # 18
    ])  # fmt: skip
    variables: dict[int, list] = {}
    assert evaluate(blk, {}, variables=variables) == [6.0]
    assert variables[1] == [4.0]


def nested() -> Block:
    """Return two nested loops: 3 outer times, 2 inner times, total += 1."""
    return Block([
        const(0.0),                       # 0
        ins(SETVAR, 0, 0),                # 1  i = 0
        ins(SETVAR, 0, 1),                # 2  total = 0
        ins(SETVAR, 0, 9),                # 3  outer hidden
        ins(SEQ, 1, 2),                   # 4
        ins(SEQ, 4, 3),                   # 5  outer init
        ins(GETVAR, 0),                   # 6
        const(3.0),                       # 7
        ins(GTEQ, 6, 7, kind=BOOL),       # 8  outer cond
        const(0.0),                       # 9
        ins(SETVAR, 9, 2),                # 10 j = 0
        ins(SETVAR, 9, 8),                # 11 inner hidden
        ins(SEQ, 10, 11),                 # 12 inner init
        ins(GETVAR, 2),                   # 13
        const(2.0),                       # 14
        ins(GTEQ, 13, 14, kind=BOOL),     # 15 inner cond
        ins(GETVAR, 1),                   # 16
        const(1.0),                       # 17
        ins(ADD, 16, 17),                 # 18
        ins(SETVAR, 18, 1),               # 19 total += 1
        ins(GETVAR, 2),                   # 20
        ins(ADD, 20, 17),                 # 21
        ins(SETVAR, 21, 2),               # 22 j += 1
        ins(SETVAR, 22, 8),               # 23 inner body
        ins(WHILE, 12, 15, 23, 0, 0),     # 24
        ins(GETVAR, 0),                   # 25
        const(1.0),                       # 26
        ins(ADD, 25, 26),                 # 27
        ins(SETVAR, 27, 0),               # 28 i += 1
        ins(SETVAR, 28, 9),               # 29 outer body
        ins(WHILE, 5, 8, 29, 0, 0),       # 30
        ins(GETVAR, 1),                   # 31
        ins(SEQ, 30, 31),                 # 32
    ])  # fmt: skip


def test_nested_loops_compose() -> None:
    assert evaluate(nested(), {}) == [6.0]


def test_an_endless_loop_is_unknown() -> None:
    # No cap and a condition never true: Substance hangs; the value is unknown.
    with pytest.raises(FunctionError):
        evaluate(counting(math.inf, 0), {})


def test_bad_loop_ranges_are_refused() -> None:
    bad = counting(30.0, 0)
    bad.instrs[20] = ins(WHILE, 6, 14, 18, 0, 0)  # body not just before it
    with pytest.raises(FunctionError):
        evaluate(bad, {})
    assert bounds(bad, lambda _u, w: [(0.0, 0.0)] * w, {}) == [(-math.inf, math.inf)]


def test_loop_variables_are_unbounded() -> None:
    # i starts bounded, but the loop sets it: unbounded from its start on.
    blk = counting(30.0, 3)
    blk.instrs[21] = ins(GETVAR, 0)
    assert bounds(blk, lambda _u, w: [(0.0, 0.0)] * w, {}) == [(-math.inf, math.inf)]
    # the init range's own values stay bounded
    assert bounds(Block(blk.instrs[:3]), lambda *_: [], {}) == [(7.0, 7.0)]


def glsl_function(blk: Block) -> tuple[list[str], str]:
    """Return ``blk``'s GLSL statements and the source of a shader using them."""
    lines, result = to_glsl(blk, "f", lambda *_: "0.0", lambda *_: "0.0")
    body = "\n\t".join(lines)
    source = (
        "#version 450\n"
        f"float f() {{\n\t{body}\n\treturn {result};\n}}\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(f()); }\n"
    )
    return lines, source


def compiles(source: str, tmp_path: Path) -> None:
    """Assert that glslangValidator compiles ``source`` (a fragment shader)."""
    validator = shutil.which("glslangValidator")
    if validator is None:
        pytest.skip("glslangValidator not installed")
    shader = tmp_path / "loop.frag"
    shader.write_text(source, encoding="utf-8")
    done = subprocess.run(  # noqa: S603
        [validator, str(shader)], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stdout + source


@pytest.mark.parametrize("cap", [3, 0])
def test_a_while_loop_in_glsl(cap: int, tmp_path: Path) -> None:
    # Section 1.8's Test C.
    lines, source = glsl_function(counting(30.0, cap))
    compiles(source, tmp_path)
    text = [line.strip() for line in lines]
    head = next(n for n, line in enumerate(text) if line.startswith(("for", "while")))
    for var in ("f_var0", "f_var1", "f_var2"):
        decls = [n for n, line in enumerate(text) if line.startswith(f"float {var} ")]
        assert len(decls) == 1
        assert decls[0] < head
    assert "float f_20 = float(0);" in text[:head]
    breaks = [n for n, line in enumerate(text) if "break" in line]
    assert len(breaks) == 1
    assert text[breaks[0]] == "if (f_14) break;"
    first_body = next(
        n for n, line in enumerate(text) if line.startswith("float f_15 ")
    )
    assert head < breaks[0] < first_body
    assert "f_20 = f_19;" in text
    if cap:
        assert text[head] == "for (int f_it20 = 0; f_it20 < 3; f_it20++) {"
    else:
        assert text[head] == "while (true) {"


def test_glsl_declares_what_a_loop_leaves_before_it(tmp_path: Path) -> None:
    # Variables first used in a loop, and a body value read after it, live
    # before the outermost loop; the nested loops compile.
    lines, source = glsl_function(nested())
    compiles(source, tmp_path)
    head = next(n for n, line in enumerate(lines) if line.startswith("while"))
    assert "float f_var2 = float(0);" in lines[:head]
    assert "float f_var8 = float(0);" in lines[:head]
    blk = counting(30.0, 0)
    blk.instrs[22] = ins(SEQ, 20, 17)  # reads the body's i + 10
    assert evaluate(blk, {}) == [37.0]
    lines, source = glsl_function(blk)
    compiles(source, tmp_path)
    head = next(n for n, line in enumerate(lines) if line.startswith("while"))
    assert "float f_17 = float(0);" in lines[:head]
    assert "\tf_17 = (f_15 + f_16);" in lines


def test_cpu_functions_compute_trig_as_the_cooker_in_glsl() -> None:
    theta = Instr(op=0x00, type=FLOAT, width=1, operands=[0, 0x3FA6])
    blk = Block([theta, Instr(op=0x26, type=FLOAT, width=1, operands=[0]),
                 Instr(op=0x2E, type=FLOAT, width=2, operands=[0, 1])])  # fmt: skip
    plain, _ = to_glsl(blk, "f", lambda *_: "", lambda *_: "")
    cooked, _ = to_glsl(blk, "f", lambda *_: "", lambda *_: "", engine_trig=True)
    assert "sbs_" not in "".join(plain)
    assert "sbs_cos(f_0)" in cooked[1] and "sbs_sin(f_1)" in cooked[2]  # noqa: PT018
    assert "float sbs_sin(float x)" in ENGINE_TRIG_GLSL
