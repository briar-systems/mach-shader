#!/usr/bin/env python3
"""Generate mach-shader's src/math.mach.

The library is a naming rule applied to a table of SPIR-V instructions, so it is
generated rather than hand-maintained: adding a vector width or an instruction
is an edit to the table, not 130 edits to the file.
"""

HEADER = '''# Shader-side maths: the functions that ARE SPIR-V instructions.
#
# Every declaration here is one SPIR-V instruction, named by a `#[spirv_op(set,
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
# TYPES. `f32`, `f32x2`, `f32x3` and `f32x4`. GLSL.std.450 defines every entry
# here over 16-, 32- and 64-bit floats; this module carries the 32-bit family,
# which is what a mach shader spells today. A future f16 or f64 family is an
# additional set of names under the same rule, not a change to these.
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
]

TERNARY = [
    ("clamp",       "FClamp",     ["x", "lo", "hi"], "Constrain a value to a range.",             "min(max(x, lo), hi)"),
    ("mix",         "FMix",       ["a", "b", "t"],   "Linear interpolation.",                     "a * (1 - t) + b * t"),
    ("smooth_step", "SmoothStep", ["lo", "hi", "x"], "A smooth Hermite step between two edges.",  "0.0 at or below lo, 1.0 at or above hi, a smooth ramp between"),
    ("fma",         "Fma",        ["a", "b", "c"],   "Fused multiply-add.",                       "a * b + c"),
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
}


def widths():
    return [("", "f32", "scalar"), ("_2", "f32x2", "2"), ("_3", "f32x3", "3"), ("_4", "f32x4", "4")]


def decl(name, suffix, glsl, params, ty, summary, ret, per_lane, sset="GLSL.std.450"):
    args = ", ".join(f"{p}: {ty}" for p in params)
    lines = [f"# {summary}"]
    if per_lane:
        lines[0] = f"# {summary[:-1]}, per lane."
    lines.append("# ---")
    for p in params:
        doc = PARAM_DOC_BY_FUN.get(name, {}).get(p, PARAM_DOC.get(p, "The operand."))
        if per_lane and doc.startswith("The ") and not doc.endswith("s."):
            doc = doc[:-1] + "s, one per lane."
        lines.append(f"# {p}:{' ' * (4 - len(p))}{doc}")
    tail = ret + (" per lane" if per_lane else "")
    lines.append(f"# ret: {tail[0].lower()}{tail[1:]}" if tail[0].isupper() else f"# ret: {tail}")
    lines.append(f'#[spirv_op("{sset}", "{glsl}")]')
    lines.append(f"pub fun {name}{suffix}({args}) {ty};")
    return "\n".join(lines)


out = [HEADER]

for suffix, ty, label in widths():
    if label == "scalar":
        out.append("\n# ---------- scalar ----------\n")
    else:
        out.append(f"\n# ---------- per lane of an {ty} ----------\n")
    per_lane = label != "scalar"
    for group in (UNARY, BINARY, TERNARY):
        for name, glsl, params, summary, ret in group:
            out.append(decl(name, suffix, glsl, params, ty, summary, ret, per_lane))
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
    lines.append(f'#[spirv_op("{sset}", "{glsl}")]')
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
             '#[spirv_op("GLSL.std.450", "Refract")]',
             f"pub fun refract{suffix}(i: {ty}, n: {ty}, eta: f32) {ty};"]
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
                 '#[spirv_op("GLSL.std.450", "Cross")]',
                 "pub fun cross_3(a: f32x3, b: f32x3) f32x3;"]
        out.append("\n".join(lines))
        out.append("")

text = "\n".join(out)
while "\n\n\n\n" in text:
    text = text.replace("\n\n\n\n", "\n\n\n")
print(text.rstrip() + "\n", end="")
