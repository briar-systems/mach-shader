#!/usr/bin/env python3
"""Generate mach-shader's src/math.mach.

The library is a naming rule applied to a table of SPIR-V instructions, so it is
generated rather than hand-maintained: adding a vector width or an instruction
is an edit to the table, not to every entry in the file.
"""

import re
import sys

sys.dont_write_bytecode = True
import machfmt

HEADER = '''# Shader-side maths: the functions that ARE SPIR-V instructions.
#
# Every declaration here is one SPIR-V instruction, named by an `#[intrinsic(target, set,
# name)]` decorator the compiler reads at the call site. A call to `normalize3`
# in a shader does not become a call at all: it becomes the instruction, inline,
# in the calling function's body.
#
# WHY THIS IS NOT `std.math`. `std.math.sqrt_f32` promises a documented IEEE
# result, a Newton refinement evaluated in double precision and accurate to under
# one f32 ulp, and it delivers that on every target. `GLSL.std.450 Sqrt` promises
# what the driver does. Those are different functions with the same name, and
# quietly substituting one for the other across a target boundary produces an
# answer that is nearly right, with nothing to report. Keeping the shader set in
# its own module makes the difference visible at the call site: `sh.sqrt(x)` and
# `math.sqrt_f32(x)` do not look alike.
#
# THE PRECISION CONTRACT. Nothing here is IEEE-exact and nothing here is portable
# between drivers. The SPIR-V specification leaves the precision of most of
# GLSL.std.450 implementation-defined, and Vulkan's own precision table gives
# bounds in ULP for some entries and none at all for others; `Pow`, `Exp`, `Log`
# and the trigonometric family are the loose ones. Two GPUs may return different
# bits for the same input, and the same GPU may return different bits under a
# different driver version. Code that needs a reproducible answer, meaning a
# hash, a checksum, or anything compared for equality across machines, must not
# use this module. Code that is shading a pixel should.
#
# THIS IS A SHADER-ONLY MODULE, DELIBERATELY. Nothing here has a body. On a
# SPIR-V target that is invisible, because no call to any of these is ever
# emitted as a call. On any other target a program that calls one fails to link,
# naming the symbol:
#
#   error: undefined symbol: _M6shader4mathN4sqrt
#
# That is the intended behaviour and it was chosen over the alternative. A native
# fallback would mean writing a second float-maths library beside the one
# `mach-std` already owns, and the only sensible implementation of each entry
# would be to call `std.math`. That is exactly the substitution the paragraph
# above rejects: it would make `sh.sqrt` and `math.sqrt_f32` the same function on
# a CPU and different functions on a GPU, which is the worst of the three options
# because it is the one that never errors. A link failure at build time is loud,
# early, and impossible to ship past.
#
# THE NAMING RULE. A scalar entry takes the plain name; a vector entry always
# carries its lane count after an underscore. `sqrt` is the scalar, `sqrt_2`,
# `sqrt_3` and `sqrt_4` are the vectors. The geometric entries follow the same
# rule even though they have no scalar form, so they are spelled `dot_3` and
# `normalize_3` rather than `dot` and `normalize`, and `cross_3` carries its
# width even though a cross product exists at no other one.
#
# The rule is uniform on purpose. The axes this module grows along are the
# instruction, the lane count, and eventually the component type, and a name that
# encodes its width is a drop-in at each of them. An unsuffixed `dot` meaning the
# 4-wide form would have to be broken the first time a shader wants the 3-wide
# one, which is the common case in lighting.
#
# THE SEPARATOR IS NOT DECORATION. `exp2` and `log2` are the base-2 entries and
# the digit is part of the instruction, not a width, so a bare digit suffix would
# make the 2-lane `exp` collide with the scalar `exp2`. The underscore keeps
# `exp2` and `exp_2` apart, and because they differ in operand type as well as in
# name, confusing one for the other is a type error rather than a wrong result.
#
# TYPES. `f32`, `f32x2`, `f32x3` and `f32x4`, and the integer rows over 16-, 32-
# and 64-bit integers at the same widths. GLSL.std.450 defines the float entries
# over 16-, 32- and 64-bit floats; this module carries the 32-bit family, which is
# what a mach shader spells today, and it is the one family whose names carry no
# type. Every other family carries its component type, as the integer rows at the
# end of the file do, so a future f16 or f64 family is an additional set of names
# under the same rule, not a change to these.
#
# This file is generated. See tools/genmath.py.
'''

# name, glsl instruction, parameter names, doc summary, doc lines per parameter,
# return description. `%V` in a doc stands for the operand's spelling.
UNARY = [
    ("abs",          "FAbs",        ["x"], "Absolute value.",                       "|x|"),
    ("sign",         "FSign",       ["x"], "Sign as -1, 0 or +1.",                  "-1.0 below zero, 0.0 at zero, 1.0 above"),
    ("floor",        "Floor",       ["x"], "Largest integer not greater than x.",   "floor(x)"),
    ("ceil",         "Ceil",        ["x"], "Smallest integer not less than x.",     "ceil(x)"),
    ("fract",        "Fract",       ["x"], "Fractional part.",                      "x - floor(x), always in [0, 1)"),
    ("radians",      "Radians",     ["x"], "Degrees to radians.",                   "the same angle in radians"),
    ("degrees",      "Degrees",     ["x"], "Radians to degrees.",                   "the same angle in degrees"),
    ("sin",          "Sin",         ["x"], "Sine of an angle in radians.",          "sin(x)"),
    ("cos",          "Cos",         ["x"], "Cosine of an angle in radians.",        "cos(x)"),
    ("tan",          "Tan",         ["x"], "Tangent of an angle in radians.",       "tan(x)"),
    ("asin",         "Asin",        ["x"], "Arcsine.",                              "asin(x) in [-pi/2, pi/2]"),
    ("acos",         "Acos",        ["x"], "Arccosine.",                            "acos(x) in [0, pi]"),
    ("atan",         "Atan",        ["x"], "Arctangent of a single ratio.",         "atan(x) in [-pi/2, pi/2]"),
    ("exp",          "Exp",         ["x"], "e raised to x.",                        "e ** x"),
    ("log",          "Log",         ["x"], "Natural logarithm.",                    "ln(x)"),
    ("exp2",         "Exp2",        ["x"], "2 raised to x.",                        "2 ** x"),
    ("log2",         "Log2",        ["x"], "Base-2 logarithm.",                     "log2(x)"),
    ("sqrt",         "Sqrt",        ["x"], "Square root.",                          "sqrt(x)"),
    ("inverse_sqrt", "InverseSqrt", ["x"], "Reciprocal square root, which is what a normalization actually wants.", "1 / sqrt(x)"),
]

BINARY = [
    ("atan2", "Atan2", ["y", "x"],    "Arctangent of y/x, quadrant-correct.", "atan2(y, x) in [-pi, pi]"),
    ("pow",   "Pow",   ["x", "e"],    "x raised to e.",                       "x ** e"),
    ("min",   "FMin",  ["a", "b"],    "Smaller of two values.",               "a where a < b, else b"),
    ("max",   "FMax",  ["a", "b"],    "Larger of two values.",                "a where a > b, else b"),
    ("step",  "Step",  ["edge", "x"], "A step from 0 to 1 at an edge.",       "0.0 below edge, else 1.0"),
    ("rem",   "OpFRem", ["a", "b"],   "Remainder of a division, as the driver computes it.", "a - b * trunc(a / b), with the sign of a", "core"),
]

TERNARY = [
    ("clamp",       "FClamp",     ["x", "lo", "hi"], "Constrain a value to a range.",             "min(max(x, lo), hi)"),
    ("mix",         "FMix",       ["a", "b", "t"],   "Linear interpolation.",                     "a * (1 - t) + b * t"),
    ("smooth_step", "SmoothStep", ["lo", "hi", "x"], "A smooth Hermite step between two edges.",  "0.0 at or below lo, 1.0 at or above hi, a smooth ramp between"),
    ("fma",         "Fma",        ["a", "b", "c"],   "Fused multiply-add.",                       "a * b + c"),
]

# the entries that return one part of a value and store the other through a pointer.
# name, glsl instruction, pointer parameter, the scalar it points to (`=` for the
# value's own type), doc summary, pointer parameter doc, return description
SPLIT = [
    ("modf",  "Modf",  "whole", "=",   "Split a value into its whole and fractional parts.",
     "Where the whole part is stored, with the sign of x.", "the fractional part, with the sign of x"),
    ("frexp", "Frexp", "exp",   "i32", "Split a value into a significand and a power of two.",
     "Where the exponent is stored.", "the significand, in [0.5, 1) in magnitude, or zero at zero"),
]

# Parameter wording that depends on the function rather than the parameter name.
PARAM_DOC_BY_FUN = {
    "radians":      {"x": "An angle in degrees."},
    "degrees":      {"x": "An angle in radians."},
    "sin":          {"x": "The angle in radians."},
    "cos":          {"x": "The angle in radians."},
    "tan":          {"x": "The angle in radians."},
    "asin":         {"x": "A value in [-1, 1]."},
    "acos":         {"x": "A value in [-1, 1]."},
    "atan":         {"x": "The ratio."},
    "atan2":        {"x": "The denominator."},
    "log":          {"x": "A positive value."},
    "log2":         {"x": "A positive value."},
    "sqrt":         {"x": "A non-negative value."},
    "inverse_sqrt": {"x": "A positive value."},
    "pow":          {"x": "The base, which must not be negative."},
    "mix":          {"a": "The value at t == 0.", "b": "The value at t == 1."},
    "smooth_step":  {"lo": "The lower edge.", "hi": "The upper edge.", "x": "The value."},
    "step":         {"x": "The value."},
    "fma":          {"a": "The first factor.", "b": "The second factor."},
    "rem":          {"a": "The dividend.", "b": "The divisor."},
}

# lines between an entry's summary and its parameters, saying what an operator does not give
NOTE_BY_FUN = {
    "rem": ["Not the `%` operator: a float `%` is computed exactly by code the compiler",
            "provides, and this is the core `OpFRem`, to whatever precision the driver gives."],
    "reverse_bits": ["The core `OpBitReverse`, which no operator writes."],
    "insert_bits": ["The core `OpBitFieldInsert`, which no operator writes. The offset and the",
                    "count must leave the field inside the 32 bits, or the result is undefined."],
    "extract_bits": ["The core `OpBitFieldSExtract` on a signed type, which fills the high bits with",
                     "the field's top bit, and `OpBitFieldUExtract` on an unsigned one, which fills",
                     "them with 0. The offset and the count must leave the field inside the 32 bits,",
                     "or the result is undefined."],
    "add_carry": ["The core `OpIAddCarry`: `+` wraps and gives no carry."],
    "sub_borrow": ["The core `OpISubBorrow`: `-` wraps and gives no borrow."],
}

PARAM_DOC = {
    "x":    "The value.",
    "y":    "The numerator.",
    "e":    "The exponent.",
    "a":    "The first value.",
    "b":    "The second value.",
    "c":    "The addend.",
    "t":    "The interpolant.",
    "lo":   "The lower bound.",
    "hi":   "The upper bound.",
    "edge": "Where the step happens.",
    "insert": "Bits to place in the field, taken from bit 0 up.",
    "offset": "Index of the field's lowest bit. A scalar at every width.",
    "count":  "Number of bits in the field. A scalar at every width.",
}


def widths():
    return [("", "f32", "scalar"), ("_2", "f32x2", "2"), ("_3", "f32x3", "3"), ("_4", "f32x4", "4")]


def decl(name, suffix, glsl, params, ty, summary, ret, per_lane, sset="GLSL.std.450", rty=None):
    # a parameter is a name, of the entry's type, or a (name, type) pair
    params = [p if isinstance(p, tuple) else (p, ty) for p in params]
    args = ", ".join(f"{p}: {t}" for p, t in params)
    lines = [f"# {summary}"]
    if per_lane:
        lines[0] = f"# {summary[:-1]}, per lane."
    if name in NOTE_BY_FUN:
        lines += ["#"] + [f"# {line}" for line in NOTE_BY_FUN[name]]
    lines.append("# ---")
    width = max(4, *(len(p) + 1 for p, _ in params))
    for p, _ in params:
        doc = PARAM_DOC_BY_FUN.get(name, {}).get(p, PARAM_DOC.get(p, "The operand."))
        if per_lane and doc.startswith("The ") and not doc.endswith("s."):
            doc = doc[:-1] + "s, one per lane."
        lines.append(f"# {p + ':':<{width}} {doc}")
    tail = ret + (" per lane" if per_lane else "")
    tail = f"{tail[0].lower()}{tail[1:]}" if tail[0].isupper() else tail
    lines.append(f"# {'ret:':<{width}} {tail}")
    lines.append(f'#[intrinsic("spirv", "{sset}", "{glsl}")]')
    lines.append(f"pub fun {name}{suffix}({args}) {rty or ty};")
    return "\n".join(lines)


def split(name, suffix, glsl, param, pty, ty, summary, pdoc, ret, per_lane):
    lanes = ty[len("f32"):]
    target = ty if pty == "=" else pty + lanes
    width = max(len(param), len("ret")) + 1
    lines = [f"# {summary[:-1]}, per lane." if per_lane else f"# {summary}", "# ---"]
    lines.append(f"# {'x:':<{width}} {'The values, one per lane.' if per_lane else 'The value.'}")
    lines.append(f"# {param + ':':<{width}} {pdoc}")
    lines.append(f"# {'ret:':<{width}} {ret}" + (" per lane" if per_lane else ""))
    lines.append(f'#[intrinsic("spirv", "GLSL.std.450", "{glsl}")]')
    lines.append(f"pub fun {name}{suffix}(x: {ty}, {param}: *{target}) {ty};")
    return "\n".join(lines)


def ldexp(suffix, ty, per_lane):
    exp = "i32" + ty[len("f32"):]
    lines = ["# Build a value from a significand and a power of two, per lane." if per_lane else "# Build a value from a significand and a power of two.", "# ---"]
    lines.append(f"# x:   {'The significands, one per lane.' if per_lane else 'The significand.'}")
    lines.append(f"# exp: {'The exponents, one per lane.' if per_lane else 'The exponent.'}")
    lines.append("# ret: x * 2 ** exp" + (" per lane" if per_lane else ""))
    lines.append('#[intrinsic("spirv", "GLSL.std.450", "Ldexp")]')
    lines.append(f"pub fun ldexp{suffix}(x: {ty}, exp: {exp}) {ty};")
    return "\n".join(lines)


DERIVATIVES = [
    ("dpdx",          "OpDPdx",         "Rate of change along the window's x axis.", "the difference between neighbouring fragments' values in x"),
    ("dpdy",          "OpDPdy",         "Rate of change along the window's y axis.", "the difference between neighbouring fragments' values in y"),
    ("fwidth",        "OpFwidth",       "Total rate of change in x and y.",           "abs(dpdx(p)) + abs(dpdy(p))"),
    ("dpdx_fine",     "OpDPdxFine",     "Rate of change along x, from the fragment's own neighbour.",  "the difference in x within the fragment's row of its quad"),
    ("dpdy_fine",     "OpDPdyFine",     "Rate of change along y, from the fragment's own neighbour.",  "the difference in y within the fragment's column of its quad"),
    ("fwidth_fine",   "OpFwidthFine",   "Total rate of change from the fragment's own neighbours.",    "abs(dpdx_fine(p)) + abs(dpdy_fine(p))"),
    ("dpdx_coarse",   "OpDPdxCoarse",   "Rate of change along x, one value for the whole quad.",       "the difference in x the quad shares"),
    ("dpdy_coarse",   "OpDPdyCoarse",   "Rate of change along y, one value for the whole quad.",       "the difference in y the quad shares"),
    ("fwidth_coarse", "OpFwidthCoarse", "Total rate of change, one value for the whole quad.",         "abs(dpdx_coarse(p)) + abs(dpdy_coarse(p))"),
]


def derivative(name, suffix, op, ty, summary, ret, per_lane):
    lines = [f"# {summary[:-1]}, per lane." if per_lane else f"# {summary}", "# ---"]
    lines.append(f"# p:   {'The values, one per lane.' if per_lane else 'The value.'}")
    lines.append(f"# ret: {ret}" + (" per lane" if per_lane else ""))
    lines.append(f'#[intrinsic("spirv", "core", "{op}")]')
    lines.append(f"pub fun {name}{suffix}(p: {ty}) {ty};")
    return "\n".join(lines)


out = [HEADER]

for suffix, ty, label in widths():
    if label == "scalar":
        out.append("\n# ---------- scalar ----------\n")
    else:
        out.append(f"\n# ---------- per lane of an {ty} ----------\n")
    per_lane = label != "scalar"
    for group in (UNARY, BINARY, TERNARY):
        for name, glsl, params, summary, ret, *iset in group:
            out.append(decl(name, suffix, glsl, params, ty, summary, ret, per_lane, *iset))
            out.append("")
    for name, glsl, param, pty, summary, pdoc, ret in SPLIT:
        out.append(split(name, suffix, glsl, param, pty, ty, summary, pdoc, ret, per_lane))
        out.append("")
    out.append(ldexp(suffix, ty, per_lane))
    out.append("")

out.append('''
# ---------- derivatives ----------
#
# A derivative is the difference between a value in neighbouring fragments of a 2x2
# quad, so it exists in a fragment stage only, and a call from any other stage is
# refused where it is written. The plain forms leave the choice of neighbour to the
# driver, the fine forms take the fragment's own, and the coarse forms one for the
# whole quad. Each is a core instruction over 32-bit floats.
''')

for suffix, ty, label in widths():
    out.append(f"\n# ----- {ty} -----\n")
    for name, op, summary, ret in DERIVATIVES:
        out.append(derivative(name, suffix, op, ty, summary, ret, label != "scalar"))
        out.append("")

out.append('''
# ---------- geometry ----------
#
# These consume vectors and have no scalar form, but they carry their width like
# every other vector entry: a lighting shader wants the 3-wide `dot3`, and a name
# that means the 4-wide form today would have to be broken to give it one.
#
# `dot` is the odd instruction of the group. GLSL spells it beside `length` and
# `normalize`, but it is not a GLSL.std.450 entry at all; it is core `OpDot`, and
# its decorator says so.
''')

GEOM_1 = [
    ("length",   "Length",   ["v"],               "Euclidean length.",                                   "sqrt(dot(v, v))", "scalar"),
    ("normalize","Normalize",["v"],               "A vector of length 1 in the same direction.",         "v / length(v)",   "vector"),
]
GEOM_2 = [
    ("distance", "Distance", ["a", "b"],          "Distance between two points.",                        "length(a - b)",   "scalar"),
    ("reflect",  "Reflect",  ["i", "n"],          "Reflect an incident vector about a normal.",          "i - 2 * dot(n, i) * n", "vector"),
]
GEOM_3 = [
    ("face_forward", "FaceForward", ["n", "i", "r"], "Flip a vector to face away from a surface.",       "n where dot(r, i) is below zero, else -n", "vector"),
]

GEOM_DOC = {
    "v": "The vector.",
    "a": "The first vector.",
    "b": "The second vector.",
    "i": "The incident vector.",
    "n": "The surface normal, which must already be normalized.",
    "r": "The reference normal.",
}


def geom(name, suffix, glsl, params, ty, summary, ret, result, sset="GLSL.std.450"):
    args = ", ".join(f"{p}: {ty}" for p in params)
    rty = "f32" if result == "scalar" else ty
    lines = [f"# {summary}", "# ---"]
    for p in params:
        lines.append(f"# {p}:{' ' * (4 - len(p))}{GEOM_DOC[p]}")
    lines.append(f"# ret: {ret}")
    lines.append(f'#[intrinsic("spirv", "{sset}", "{glsl}")]')
    lines.append(f"pub fun {name}{suffix}({args}) {rty};")
    return "\n".join(lines)


for suffix, ty, label in widths():
    if label == "scalar":
        continue
    out.append(f"\n# ----- {ty} -----\n")

    out.append(geom("dot", suffix, "OpDot", ["a", "b"], ty,
                    "Dot product.", "the sum of the per-lane products", "scalar", sset="core"))
    out.append("")

    for name, glsl, params, summary, ret, result in GEOM_1 + GEOM_2 + GEOM_3:
        out.append(geom(name, suffix, glsl, params, ty, summary, ret, result))
        out.append("")

    lines = ["# Refract an incident vector through a surface.", "# ---",
             "# i:   The incident vector, which must already be normalized.",
             "# n:   The surface normal, which must already be normalized.",
             "# eta: The ratio of indices of refraction. A scalar at every width.",
             "# ret: the refracted vector, or the zero vector under total internal reflection",
             '#[intrinsic("spirv", "GLSL.std.450", "Refract")]',
             f"pub fun refract{suffix}(i: {ty}, n: {ty}, eta: f32) {ty};"]
    out.append("\n".join(lines))
    out.append("")

    lines = ["# Scale a vector by a scalar.",
             "#",
             "# The core `OpVectorTimesScalar`. The `*` operator multiplies a vector by a vector",
             "# of one shape, so it scales only once the scalar is built into a vector.",
             "# ---",
             "# v:   The vector.",
             "# s:   The factor every lane is multiplied by. A scalar at every width.",
             "# ret: v with each lane multiplied by s",
             '#[intrinsic("spirv", "core", "OpVectorTimesScalar")]',
             f"pub fun scale{suffix}(v: {ty}, s: f32) {ty};"]
    out.append("\n".join(lines))
    out.append("")

    if suffix == "_3":
        lines = ["# Cross product.",
                 "#",
                 "# Defined at three lanes and no other, which is why it appears only here.",
                 "# It still carries the width, so the rule holds without an exception.",
                 "# ---",
                 "# a:   The first vector.",
                 "# b:   The second vector.",
                 "# ret: the vector perpendicular to both, right-handed",
                 '#[intrinsic("spirv", "GLSL.std.450", "Cross")]',
                 "pub fun cross_3(a: f32x3, b: f32x3) f32x3;"]
        out.append("\n".join(lines))
        out.append("")

out.append('''
# ---------- integers ----------
#
# The integer rows, over `i16`, `u16`, `i32`, `u32`, `i64` and `u64`. Each entry
# carries its component type after an underscore, and a vector entry its lane count
# after that: `min_i32`, `min_u64_3`. The float family above is the one unsuffixed
# type. The instruction follows the signedness of the type, so `min_i32` is `SMin` and
# `min_u32` is `UMin`, and `abs` and `sign` exist only signed. `find_msb_i32` finds the
# most significant bit that differs from the sign bit, so it scans for a 0 in a
# negative value.
#
# A 16-bit entry needs the target's `Int16` and a 64-bit one its `Int64`, where the
# target names an `env`. The bit scans are 32-bit only: GLSL.std.450 defines `FindILsb`,
# `FindSMsb` and `FindUMsb` on 32-bit integers alone. The count of set bits is the core
# `OpBitCount`, which no operator writes, and is 32-bit only too: Vulkan holds its operand
# to 32 bits unless the device enables `maintenance9`, which no Vulkan version guarantees.
# The same rule holds the other core bit rows, `reverse_bits`, `insert_bits` and
# `extract_bits`, to 32 bits.
#
# `add_carry` and `sub_borrow` are the core `OpIAddCarry` and `OpISubBorrow`, over unsigned
# integers only, as SPIR-V defines them. Each returns its two results as a record of two
# members of the operands' type.

# the result of an `add_carry`: the sum wrapped to the operands' width, and 1 in each lane
# where the sum overflowed, else 0
pub rec AddCarry[T] {
    sum:   T;
    carry: T;
}

# the result of a `sub_borrow`: the difference wrapped to the operands' width, and 1 in
# each lane where b was larger than a, else 0
pub rec SubBorrow[T] {
    difference: T;
    borrow:     T;
}
''')

# name, instruction for a signed and an unsigned type (absent where the signedness has
# no form), parameter names, doc summary, return description, the widths it takes, and
# the instruction set. the bit scans take 32 bits alone, since GLSL.std.450 defines them
# on no other width, and the bit count since vulkan holds OpBitCount to 32 bits
ANY_WIDTH = (16, 32, 64)
INTEGER = [
    ("abs",      {"i": "SAbs"},                       ["x"],             "Absolute value.",                "|x|", ANY_WIDTH),
    ("sign",     {"i": "SSign"},                      ["x"],             "Sign as -1, 0 or +1.",           "-1 below zero, 0 at zero, 1 above", ANY_WIDTH),
    ("min",      {"i": "SMin",     "u": "UMin"},      ["a", "b"],        "Smaller of two values.",         "a where a < b, else b", ANY_WIDTH),
    ("max",      {"i": "SMax",     "u": "UMax"},      ["a", "b"],        "Larger of two values.",          "a where a > b, else b", ANY_WIDTH),
    ("clamp",    {"i": "SClamp",   "u": "UClamp"},    ["x", "lo", "hi"], "Constrain a value to a range.",  "min(max(x, lo), hi)", ANY_WIDTH),
    ("find_lsb", {"i": "FindILsb", "u": "FindILsb"},  ["x"],             "Index of the least significant set bit.", "the bit index, or every bit set (-1) at zero", (32,)),
    ("find_msb", {"i": "FindSMsb", "u": "FindUMsb"},  ["x"],             "Index of the most significant bit that differs from the sign.", "the bit index, or every bit set (-1) where there is none", (32,)),
    ("count_ones", {"i": "OpBitCount", "u": "OpBitCount"}, ["x"],        "Number of set bits.",            "how many bits of x are 1", (32,), "core"),
    ("reverse_bits", {"i": "OpBitReverse", "u": "OpBitReverse"}, ["x"],  "Reverse the order of the bits.", "x with bit n moved to bit 31 - n", (32,), "core"),
    ("insert_bits", {"i": "OpBitFieldInsert", "u": "OpBitFieldInsert"}, ["x", "insert", ("offset", "u32"), ("count", "u32")],
     "Replace a field of bits.", "x with its count bits from offset replaced by the low bits of insert", (32,), "core"),
    ("extract_bits", {"i": "OpBitFieldSExtract", "u": "OpBitFieldUExtract"}, ["x", ("offset", "u32"), ("count", "u32")],
     "Read a field of bits.", "the count bits of x from offset, moved down to bit 0", (32,), "core"),
]

# name, instruction, the record it returns, doc summary, return description. unsigned only,
# as SPIR-V defines both
CARRY = [
    ("add_carry",  "OpIAddCarry",  "AddCarry",  "Add two values, keeping the carry.",       "the wrapped sum and the carry"),
    ("sub_borrow", "OpISubBorrow", "SubBorrow", "Subtract two values, keeping the borrow.", "the wrapped difference and the borrow"),
]

for bits in ANY_WIDTH:
    for sign in ("i", "u"):
        ity = f"{sign}{bits}"
        for suffix, fty, label in widths():
            ty = ity + fty[len("f32"):]
            out.append(f"\n# ----- {ty} -----\n")
            per_lane = label != "scalar"
            for name, glsl, params, summary, ret, takes, *iset in INTEGER:
                if sign in glsl and bits in takes:
                    out.append(decl(name, f"_{ity}{suffix}", glsl[sign], params, ty, summary, ret, per_lane, *iset))
                    out.append("")
            for name, op, record, summary, ret in CARRY if sign == "u" else ():
                out.append(decl(name, f"_{ity}{suffix}", op, ["a", "b"], ty, summary, ret, per_lane, "core", f"{record}[{ty}]"))
                out.append("")

text = "\n".join(out)

CONFORM_HEADER = '''# Calls every entry in `shader.math`.
#
# This exists to be compiled, not to shade anything. Each call forces the
# compiler to resolve an `#[intrinsic]` decorator and emit the instruction at that
# operand width, so a wrong instruction name or a signature SPIR-V rejects turns
# into a build failure here. The library on its own compiles to nothing and would
# catch neither.
#
# Every result is added to the output so no call can be eliminated as dead.
#
# This file is generated from the declarations `tools/genmath.py` writes, so an
# entry added to the table is called here without an edit. Run
# `python3 tools/genmath.py --conform > conform/src/conform_frag.mach`.

use sh: shader.math;

#[gpu.input(0)]
var a2: f32x2;
#[gpu.input(1)]
var b2: f32x2;
#[gpu.input(2)]
var a3: f32x3;
#[gpu.input(3)]
var b3: f32x3;
#[gpu.input(4)]
var a4: f32x4;
#[gpu.input(5)]
var b4: f32x4;
#[gpu.input(6)]
var s:  f32;

#[gpu.output(0)]
var out_colour: f32x4;
'''

SIGNATURE = re.compile(r"pub fun (\w+)\(([^)]*)\) ([\w\[\]]+);")
RECORD = re.compile(r"pub rec (\w+)\[T\] \{([^}]*)\}")


def lanes(ty):
    found = re.fullmatch(r"([iuf]\d+)(?:x(\d))?", ty)
    return found.group(1), int(found.group(2) or 1)


def conform(library):
    """A fragment shader calling every declaration in `library` once."""
    # every type alternates between two values, so no two arguments of a call are one
    # value, and an integer vector takes each lane from a different float input
    values = {"f32": ["s", "b2[0]"]}
    for n in (2, 3, 4):
        values[f"f32x{n}"] = [f"a{n}", f"b{n}"]
    prelude = []

    def value(ty, i):
        if ty not in values:
            scalar, n = lanes(ty)
            values[ty] = []
            for name, source in (("x", "a4"), ("y", "b4")):
                if n == 1:
                    init = f"{values['f32'][len(values[ty])]}::{ty}"
                else:
                    init = f"{ty}{{{', '.join(f'{source}[{k}]::{scalar}' for k in range(n))}}}"
                prelude.append(f"    val {name}_{ty}: {ty} = {init};")
                values[ty].append(f"{name}_{ty}")
        options = values[ty]
        return options[i % len(options)]

    members = {name: re.findall(r"(\w+):\s*T;", fields) for name, fields in RECORD.findall(library)}
    body = []
    for n, (fun, params, ret) in enumerate(SIGNATURE.findall(library)):
        args = []
        for i, param in enumerate(params.split(", ")):
            ty = param.split(": ")[1]
            if ty.startswith("*"):
                body.append(f"    var p{n}: {ty[1:]};")
                args.append(f"?p{n}")
            else:
                args.append(value(ty, i))
        record = re.fullmatch(r"(\w+)\[(\w+)\]", ret)
        body.append(f"    val r{n}: {'sh.' + ret if record else ret} = sh.{fun}({', '.join(args)});")
        # a record result adds each of its members, so neither can be eliminated
        places = [(f"r{n}.{m}", record.group(2)) for m in members[record.group(1)]] if record else [(f"r{n}", ret)]
        for place, ty in places:
            scalar, width = lanes(ty)
            lane = place if width == 1 else f"{place}[0]"
            body.append(f"    total = total + {lane if scalar == 'f32' else lane + '::f32'};")
    return "\n".join([CONFORM_HEADER, "#[gpu.stage(\"fragment\")]", "fun frag_main() {",
                      *prelude, "    var total: f32 = 0.0;", *body,
                      "    out_colour = f32x4{total, total, total, total};", "}", ""])


machfmt.emit("genmath", conform(text) if sys.argv[1:] == ["--conform"] else text)
