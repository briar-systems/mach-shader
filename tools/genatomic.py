#!/usr/bin/env python3
"""Generate mach-shader's src/atomic.mach.

The module is a table of SPIR-V atomic instructions applied to every type an atomic
operates on, so it is generated rather than hand-maintained: adding a type or an
instruction is an edit to the tables below.
"""

import sys

sys.dont_write_bytecode = True
import machfmt

HEADER = '''# Shader-side atomics: read-modify-write operations on one value, indivisible across
# every invocation that reaches the same memory.
#
# Every declaration here is one SPIR-V instruction, named by `#[op]`, as in
# `shader.math`. Like a barrier, an atomic is never inlined away or reordered: the
# compiler treats it as reading and writing all memory, at every optimization level.
#
# THE POINTER TAKES ITS STORAGE CLASS FROM THE CALL. An atomic has no body, so its
# pointer parameter is whatever the argument points into: an element or member of a
# `#[storage]` buffer, of a `#[shared]` variable, or a storage image texel from
# `shader.texture`'s `image_texel_*`. `atomic_add_u32(?counts.n[i], ...)` and
# `atomic_add_u32(?tile[i], ...)` are the same declaration. A `"readonly"` binding
# admits only an atomic load and a `"writeonly"` one only an atomic store.
#
# SCOPE AND SEMANTICS ARE CONSTANTS. Each atomic takes the `SCOPE_*` it is atomic
# across and the `SEMANTICS_*` that order the memory around it, from `shader.sync`, as
# constants written at the call. A relaxed atomic passes `SEMANTICS_RELAXED`.
#
# THE TYPE DECIDES THE FEATURE. A 32-bit integer atomic is core in every storage
# class. Every other type needs the Vulkan device feature of its storage class, named
# in the target's `extensions`: `buffer_*` on buffer memory, `shared_*` on workgroup
# memory and `image_*` on an image texel, where Vulkan defines only a 64-bit integer
# and an `f32`. A 64-bit integer needs `*_int64_atomics`; an `f16`, `f32` or `f64`
# load, store or exchange `*_float{16,32,64}_atomics`; an add `*_float{16,32,64}_atomic_add`;
# a min or max `*_float{16,32,64}_atomic_min_max`. An `f16` atomic also needs `float16`,
# and an `f16` storage buffer `storage_buffer_16bit_access`. A call whose feature the
# target does not select is refused there, naming it.
#
# THE NAMING RULE. Every entry carries the type it operates on after an underscore:
# `atomic_add_u32`, `atomic_add_i64`, `atomic_add_f32`. There is no unsuffixed entry,
# so no type is the default that a later one would have to break. The signed and
# unsigned forms are distinct where the instruction is, so `atomic_min_i32` is
# `OpAtomicSMin` and `atomic_min_u32` is `OpAtomicUMin`, and a float's add, min and
# max are the `OpAtomicF*EXT` instructions.
#
# THIS IS A SHADER-ONLY MODULE. On any target other than SPIR-V a call to one of these
# fails to link, for the reason `shader.math` gives.
#
# This file is generated. See tools/genatomic.py.
'''

# the name suffix and the mach type of each type an atomic operates on
INTEGERS = [("u32", "u32"), ("i32", "i32"), ("u64", "u64"), ("i64", "i64")]
FLOATS = [("f16", "f16"), ("f32", "f32"), ("f64", "f64")]

POINTER = "The address of the value, in a storage buffer, workgroup memory or an image texel."
SCOPE = "The `SCOPE_*` the operation is atomic across."
SEMANTICS = "The `SEMANTICS_*` ordering memory around it."


def instruction(op, ty, signed):
    if op in ("min", "max"):
        if ty.startswith("f"):
            return "OpAtomicF" + op.capitalize() + "EXT"
        return "OpAtomic" + ("S" if signed else "U") + op.capitalize()
    if op == "add" and ty.startswith("f"):
        return "OpAtomicFAddEXT"
    return {
        "load": "OpAtomicLoad", "store": "OpAtomicStore", "exchange": "OpAtomicExchange",
        "compare_exchange": "OpAtomicCompareExchange", "increment": "OpAtomicIIncrement",
        "decrement": "OpAtomicIDecrement", "add": "OpAtomicIAdd", "sub": "OpAtomicISub",
        "and": "OpAtomicAnd", "or": "OpAtomicOr", "xor": "OpAtomicXor",
    }[op]


# name, summary, extra parameters after scope and semantics, return doc (None for none)
OPS = {
    "load":      ("Read the value atomically.", [], "the value"),
    "store":     ("Write a value atomically.", [("value", "The value to write.")], None),
    "exchange":  ("Write a value and return the one it replaced.", [("value", "The value to write.")], "the value before the write"),
    "increment": ("Add one.", [], "the value before the increment"),
    "decrement": ("Subtract one.", [], "the value before the decrement"),
    "add":       ("Add a value.", [("value", "The value to add.")], "the value before the add"),
    "sub":       ("Subtract a value.", [("value", "The value to subtract.")], "the value before the subtraction"),
    "min":       ("Keep the smaller of the value and another.", [("value", "The value to compare with.")], "the value before the operation"),
    "max":       ("Keep the larger of the value and another.", [("value", "The value to compare with.")], "the value before the operation"),
    "and":       ("Bitwise-and a mask into the value.", [("value", "The mask.")], "the value before the operation"),
    "or":        ("Bitwise-or a mask into the value.", [("value", "The mask.")], "the value before the operation"),
    "xor":       ("Bitwise-xor a mask into the value.", [("value", "The mask.")], "the value before the operation"),
}

INTEGER_OPS = ["load", "store", "exchange", "compare_exchange", "increment", "decrement",
               "add", "sub", "min", "max", "and", "or", "xor"]
FLOAT_OPS = ["load", "store", "exchange", "add", "min", "max"]


def decl(op, suffix, ty):
    signed = suffix.startswith("i")
    glsl = instruction(op, suffix, signed)
    lines = []
    if op == "compare_exchange":
        lines.append("# Write a value if the current one equals a comparator, and return the current one.")
        params = [("p", f"*{ty}", POINTER), ("scope", "u32", SCOPE),
                  ("equal", "u32", "The `SEMANTICS_*` when the write happens."),
                  ("unequal", "u32", "The `SEMANTICS_*` when it does not, no stronger than `equal` and never a release."),
                  ("value", ty, "The value to write."),
                  ("comparator", ty, "The value the current one must equal.")]
        ret = "the value before the operation, which equals `comparator` exactly when the write happened"
    else:
        summary, extra, ret = OPS[op]
        lines.append(f"# {summary}")
        params = [("p", f"*{ty}", POINTER), ("scope", "u32", SCOPE), ("semantics", "u32", SEMANTICS)]
        params += [(n, ty, d) for n, d in extra]
    lines.append("# ---")
    width = max(len(n) for n, _, _ in params + [("ret", "", "")]) + 1
    for n, _, d in params:
        lines.append(f"# {n + ':':<{width}} {d}")
    if ret is not None:
        lines.append(f"# {'ret:':<{width}} {ret}")
    lines.append(f'#[op("spirv", "core", "{glsl}")]')
    args = ", ".join(f"{n}: {t}" for n, t, _ in params)
    tail = f" {ty}" if ret is not None else ""
    lines.append(f"pub fun atomic_{op}_{suffix}({args}){tail};")
    return "\n".join(lines)


out = [HEADER, ""]
for suffix, ty in INTEGERS:
    for op in INTEGER_OPS:
        out.append(decl(op, suffix, ty))
        out.append("")
for suffix, ty in FLOATS:
    for op in FLOAT_OPS:
        out.append(decl(op, suffix, ty))
        out.append("")

machfmt.emit("genatomic", "\n".join(out))
