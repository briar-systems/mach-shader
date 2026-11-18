#!/usr/bin/env python3
"""Generate mach-shader's src/compare.mach, src/logic.mach and src/vector.mach.

Each module is a table of core SPIR-V instructions no language construct writes,
applied to the component types and lane counts each one takes, so it is generated
rather than hand-maintained: adding an instruction is one line in a table below. With
`--conform` it writes that module's conformance shader instead, a compute stage
calling every declaration the module run would write.

    python3 tools/gencore.py compare > src/compare.mach
    python3 tools/gencore.py compare --conform > conform/src/conform_compare_comp.mach
"""

import sys
import textwrap
from dataclasses import dataclass, field

sys.dont_write_bytecode = True
import machfmt

INTEGERS = ["i16", "u16", "i32", "u32", "i64", "u64"]
FLOATS = ["f16", "f32", "f64"]
NUMBERS = INTEGERS + FLOATS
LANES = (1, 2, 3, 4)

# the capability a component type needs where the target names an `env`
CAPABILITY = {"i16": "Int16", "u16": "Int16", "i64": "Int64", "u64": "Int64", "f16": "Float16", "f64": "Float64"}

# a truth value, as `shader.subgroup` takes and gives one
TRUTH = "u32"

SHADER_ONLY = '''#
# THIS IS A SHADER-ONLY MODULE. On any target other than SPIR-V a call to one of these
# fails to link, for the reason `shader.math` gives.
#
# This file is generated. See tools/gencore.py.
'''


def vec(ty, lanes):
    return ty if lanes == 1 else f"{ty}x{lanes}"


def kind(ty):
    return {"i": "i", "u": "u", "f": "f"}[ty[0]]


@dataclass
class Param:
    name: str
    ty: str
    doc: str
    # a literal is a constant the conformance shader writes at the call
    literal: int = None


@dataclass
class Decl:
    name: str
    op: str
    summary: str
    params: list
    ret: str
    ret_doc: str
    notes: list = field(default_factory=list)

    def text(self):
        lines = [f"# {self.summary}"]
        if self.notes:
            lines.append("#")
            for note in self.notes:
                lines += [f"# {line}" for line in textwrap.wrap(note, 86)]
        lines.append("# ---")
        width = max(len(p.name) for p in self.params + [Param("ret", "", "")]) + 1
        for p in self.params:
            lines.append(f"# {p.name + ':':<{width}} {p.doc}")
        lines.append(f"# {'ret:':<{width}} {self.ret_doc}")
        lines.append(f'#[intrinsic("spirv", "core", "{self.op}")]')
        lines.append(f"pub fun {self.name}({', '.join(f'{p.name}: {p.ty}' for p in self.params)}) {self.ret};")
        return "\n".join(lines)


def per_lane(text, lanes):
    """A scalar entry's summary, or a vector entry's with `per lane` before its stop."""
    return text if lanes == 1 else f"{text[:-1]}, per lane."


def needs(ty):
    cap = CAPABILITY.get(ty)
    return [f"Needs the target's `{cap}` where it names an `env`."] if cap else []


def operand(name, ty, lanes, scalar_doc, vector_doc):
    return Param(name, vec(ty, lanes), scalar_doc if lanes == 1 else vector_doc)


def suffix(ty, lanes):
    return f"_{ty}" + ("" if lanes == 1 else f"_{lanes}")


COMPARE_HEADER = '''# Shader-side comparisons: the relations no operator writes, each giving 1 or 0 per lane.
#
# Every declaration here is one core SPIR-V instruction, named by `#[intrinsic]`, as in
# `shader.math`. `<`, `<=`, `==` and `!=` are instructions of their own, and `>` and `>=`
# are the less-than instructions with their operands exchanged, so the compiler keeps those
# to the operators. What is here is the rest: the greater-than instructions themselves, the
# float relations no operator gives, and `OpIsNan`.
#
# ONE OR ZERO PER LANE. A comparison here returns a `u32`, or a `u32` vector of its
# operands' lane count, holding 1 where the relation holds and 0 where it does not. A
# comparison operator on vectors gives a mask instead, every bit of a lane set where it
# holds, in the unsigned integer of the operands' width. The 1 or 0 is the truth value
# `shader.logic` takes, and a count or a weight as it stands. A 1 or 0 per lane for
# less-than is `greater_than` with the operands exchanged.
#
# ORDERED AND UNORDERED. An ordered float relation is false where either operand is NaN,
# and an unordered one true. `<`, `<=`, `==`, `>` and `>=` are ordered and `!=` is
# unordered, so `greater_than_f32` and `greater_equal_f32` are ordered as `>` and `>=` are,
# and the ordered not-equal and the unordered forms of the others are here.
#
# TYPES. 16-, 32- and 64-bit integers and floats, scalar and 2, 3 and 4 lanes. The
# instruction follows the signedness, so `greater_than_i32` is `OpSGreaterThan` and
# `greater_than_u32` is `OpUGreaterThan`. A 16- or 64-bit entry needs the target's `Int16`,
# `Int64`, `Float16` or `Float64` where it names an `env`, and says which.
#
# THE NAMING RULE. Every entry carries its operands' type after an underscore, as in
# `shader.atomic`, and a vector entry its lane count after that: `greater_than_f32`,
# `unordered_less_than_f16_3`.
''' + SHADER_ONLY

ORDERED = "Ordered, as the operator is: 0 where either operand is NaN."
UNORDERED = "Unordered: 1 where either operand is NaN, which of the operators only `!=` gives."

# name, instruction by component kind, summary, return description, a note on a float
COMPARE = [
    ("greater_than",            {"i": "OpSGreaterThan", "u": "OpUGreaterThan", "f": "OpFOrdGreaterThan"},
     "Whether one value is greater than another.", "1 where a is greater than b, else 0", ORDERED),
    ("greater_equal",           {"i": "OpSGreaterThanEqual", "u": "OpUGreaterThanEqual", "f": "OpFOrdGreaterThanEqual"},
     "Whether one value is greater than or equal to another.", "1 where a is greater than or equal to b, else 0", ORDERED),
    ("ordered_not_equal",       {"f": "OpFOrdNotEqual"},
     "Whether two values differ and neither is NaN.", "1 where a and b differ and neither is NaN, else 0",
     "Ordered: 0 where either operand is NaN, where `!=` gives 1."),
    ("unordered_equal",         {"f": "OpFUnordEqual"},
     "Whether two values are equal or either is NaN.", "1 where a equals b or either is NaN, else 0", UNORDERED),
    ("unordered_less_than",     {"f": "OpFUnordLessThan"},
     "Whether one value is less than another or either is NaN.", "1 where a is less than b or either is NaN, else 0", UNORDERED),
    ("unordered_less_equal",    {"f": "OpFUnordLessThanEqual"},
     "Whether one value is less than or equal to another or either is NaN.",
     "1 where a is less than or equal to b or either is NaN, else 0", UNORDERED),
    ("unordered_greater_than",  {"f": "OpFUnordGreaterThan"},
     "Whether one value is greater than another or either is NaN.", "1 where a is greater than b or either is NaN, else 0", UNORDERED),
    ("unordered_greater_equal", {"f": "OpFUnordGreaterThanEqual"},
     "Whether one value is greater than or equal to another or either is NaN.",
     "1 where a is greater than or equal to b or either is NaN, else 0", UNORDERED),
]

# name, instruction by component kind, summary, return description
CLASSIFY = [
    ("is_nan", {"f": "OpIsNan"}, "Whether a value is NaN.", "1 where x is NaN, else 0"),
]


def compare():
    out = []
    for ty in NUMBERS:
        for lanes in LANES:
            for name, ops, summary, ret, note in COMPARE:
                if kind(ty) not in ops:
                    continue
                notes = ([note] if kind(ty) == "f" else []) + needs(ty)
                out.append(Decl(f"{name}{suffix(ty, lanes)}", ops[kind(ty)], per_lane(summary, lanes),
                                [operand("a", ty, lanes, "The left operand.", "The left operands, one per lane."),
                                 operand("b", ty, lanes, "The right operand.", "The right operands, one per lane.")],
                                vec(TRUTH, lanes), ret + (" in each lane" if lanes > 1 else ""), notes))
            for name, ops, summary, ret in CLASSIFY:
                if kind(ty) not in ops:
                    continue
                out.append(Decl(f"{name}{suffix(ty, lanes)}", ops[kind(ty)], per_lane(summary, lanes),
                                [operand("x", ty, lanes, "The value.", "The values, one per lane.")],
                                vec(TRUTH, lanes), ret + (" in each lane" if lanes > 1 else ""), needs(ty)))
    return COMPARE_HEADER, out


LOGIC_HEADER = '''# Shader-side logic: truth values combined lane by lane without a branch, a vector's lanes
# reduced to one truth value, and a choice between two values by a truth value.
#
# Every declaration here is one core SPIR-V instruction, named by `#[intrinsic]`, as in
# `shader.math`, and none is an operator's: `&&` and `||` branch, a truth value is an
# integer to every other operator, and the language has no reduction of a vector's lanes
# and no conditional expression.
#
# TRUTH VALUES. A truth value is a `u32`, or a `u32` vector holding one per lane, true where
# it is nonzero, as in `shader.subgroup`. An entry that answers yes or no returns 1 or 0 in
# each lane, and the comparisons of `shader.compare` return exactly that.
#
# NOTHING BRANCHES. `logical_and` and `logical_or` evaluate both operands, where `&&` and
# `||` skip the second, and `select_*` evaluates both values, where an `if` runs one. Both
# operands are computed whatever they hold, so neither may be something only the other's
# truth makes safe to compute.
#
# TYPES. `select_*` chooses between 16-, 32- and 64-bit integers and floats, scalar and 2, 3
# and 4 lanes, with a condition of the values' lane count. A 16- or 64-bit entry needs the
# target's `Int16`, `Int64`, `Float16` or `Float64` where it names an `env`, and says which.
#
# THE NAMING RULE. An entry over truth values alone has fixed types, so it takes the plain
# name, and a vector entry its lane count after an underscore: `logical_and`,
# `logical_and_3`, `all_4`. `select_*` carries the type of the values it chooses between:
# `select_f32`, `select_u16_3`.
''' + SHADER_ONLY

# name, instruction, operands, summary, return description, note
LOGICAL = [
    ("logical_and",       "OpLogicalAnd",      ["a", "b"], "Whether two truth values both hold.",
     "1 where a and b are both true, else 0", "Unlike `&&`, both operands are evaluated and nothing branches."),
    ("logical_or",        "OpLogicalOr",       ["a", "b"], "Whether either of two truth values holds.",
     "1 where a or b is true, else 0", "Unlike `||`, both operands are evaluated and nothing branches."),
    ("logical_not",       "OpLogicalNot",      ["a"],      "The opposite of a truth value.",
     "1 where a is false, else 0", "The result is 1 or 0, where `~` would flip every bit of the integer."),
    ("logical_equal",     "OpLogicalEqual",    ["a", "b"], "Whether two truth values agree.",
     "1 where a and b are both true or both false, else 0", "Any two nonzero values agree, where `==` compares the integers."),
    ("logical_not_equal", "OpLogicalNotEqual", ["a", "b"], "Whether two truth values differ.",
     "1 where exactly one of a and b is true, else 0", "Any two nonzero values agree, where `!=` compares the integers."),
]

# name, instruction, summary, return description
REDUCE = [
    ("all", "OpAll", "Whether every lane of a truth vector holds.", "1 where every lane is true, else 0"),
    ("any", "OpAny", "Whether any lane of a truth vector holds.",   "1 where at least one lane is true, else 0"),
]

TRUTH_DOC = ("A truth value, true where nonzero.", "The truth values, one per lane, each true where nonzero.")


def logic():
    out = []
    for lanes in LANES:
        for name, op, params, summary, ret, note in LOGICAL:
            out.append(Decl(name + ("" if lanes == 1 else f"_{lanes}"), op, per_lane(summary, lanes),
                            [operand(p, TRUTH, lanes, *TRUTH_DOC) for p in params],
                            vec(TRUTH, lanes), ret + (" in each lane" if lanes > 1 else ""), [note]))
    for lanes in LANES[1:]:
        for name, op, summary, ret in REDUCE:
            out.append(Decl(f"{name}_{lanes}", op, summary, [operand("v", TRUTH, lanes, *TRUTH_DOC)], TRUTH, ret))
    for ty in NUMBERS:
        for lanes in LANES:
            notes = ["Both values are evaluated and nothing branches." +
                     (" Each lane is chosen by its own lane of the condition." if lanes > 1 else "")] + needs(ty)
            out.append(Decl(f"select{suffix(ty, lanes)}", "OpSelect", per_lane("Choose between two values by a truth value.", lanes),
                            [operand("condition", TRUTH, lanes, *TRUTH_DOC),
                             operand("a", ty, lanes, "The value where the condition is true.", "The values where the condition is true, one per lane."),
                             operand("b", ty, lanes, "The value where it is false.", "The values where it is false, one per lane.")],
                            vec(ty, lanes), "a where condition is true, else b" + (", in each lane" if lanes > 1 else ""), notes))
    return LOGIC_HEADER, out


VECTOR_HEADER = '''# Shader-side lane operations: the lanes of two vectors rearranged into one, and one lane
# read or written at an index computed at run time.
#
# Every declaration here is one core SPIR-V instruction, named by `#[intrinsic]`, as in
# `shader.math`, and none is a construct's: `v[i]` reads one lane at a constant index,
# `v[start, count]` takes consecutive lanes of one vector, and a literal builds a vector
# from scalars.
#
# SHUFFLE. `shuffle_*` builds a vector from the lanes of two, numbered across both: the
# first vector's from 0 and the second's after them. Each index is a constant written at
# the call, below the two vectors' lane counts together, or 0xFFFFFFFF for a lane whose
# value is undefined. A swizzle of one vector passes it twice, so
# `shuffle_f32_4_4_3(v, v, 2, 1, 0)` is the first three lanes of v in reverse.
#
# A RUN-TIME INDEX. `extract_*` and `insert_*` take the lane as a `u32` computed at run
# time. An index at or past the vector's lane count is undefined behaviour, which nothing
# checks.
#
# TYPES. 16-, 32- and 64-bit integers and floats, at 2, 3 and 4 lanes. A 16- or 64-bit
# entry needs the target's `Int16`, `Int64`, `Float16` or `Float64` where it names an `env`,
# and says which.
#
# THE NAMING RULE. Every entry carries its component type after an underscore, as in
# `shader.atomic`, and its lane counts after that. `extract_f32_4` and `insert_f32_4` take
# an `f32x4`. A shuffle names three: the first vector's, the second's and the result's, so
# `shuffle_u32_4_2_3` builds a `u32x3` from a `u32x4` and a `u32x2`.
''' + SHADER_ONLY

LANE_NAMES = ["x", "y", "z", "w"]


def vector():
    out = []
    for ty in NUMBERS:
        for lanes in LANES[1:]:
            v = vec(ty, lanes)
            out.append(Decl(f"extract_{ty}_{lanes}", "OpVectorExtractDynamic", "One lane of a vector, chosen at run time.",
                            [Param("v", v, "The vector."), Param("index", "u32", "The lane, below the vector's lane count.")],
                            ty, "lane index of v", ["`v[i]` reads a lane at a constant index alone."] + needs(ty)))
            out.append(Decl(f"insert_{ty}_{lanes}", "OpVectorInsertDynamic", "A vector with one lane, chosen at run time, replaced.",
                            [Param("v", v, "The vector."), Param("value", ty, "The value the lane takes."),
                             Param("index", "u32", "The lane, below the vector's lane count.")],
                            v, "v with lane index replaced by value", ["Assigning `v[i]` writes a lane at a constant index alone."] + needs(ty)))
        for first in LANES[1:]:
            for second in LANES[1:]:
                for lanes in LANES[1:]:
                    # each result lane reads a different source lane, from both vectors where it can
                    indices = [Param(LANE_NAMES[k], "u32", f"The source lane of the result's lane {k}, a constant.",
                                     (2 * k + 1) % (first + second)) for k in range(lanes)]
                    out.append(Decl(f"shuffle_{ty}_{first}_{second}_{lanes}", "OpVectorShuffle",
                                    "A vector built from the lanes of two.",
                                    [Param("first", vec(ty, first), "The first vector, whose lanes are numbered from 0."),
                                     Param("second", vec(ty, second), f"The second vector, whose lanes are numbered from {first}.")] + indices,
                                    vec(ty, lanes), "the lane each index names, in order", needs(ty)))
    return VECTOR_HEADER, out


MODULES = {"compare": compare, "logic": logic, "vector": vector}

CONFORM_HEADER = '''# Calls every entry in `shader.{module}`.
#
# This exists to be compiled, not to be dispatched. Each call forces the compiler to
# resolve an `#[intrinsic]` decorator and emit the instruction at that type, so a wrong
# instruction name or a signature SPIR-V rejects turns into a build failure here, and
# `tools/instcheck.py` finds each instruction in the built module at its own types. The
# target names no `env`, so it holds every capability a 16- or 64-bit entry needs.
#
# Every value comes from a storage buffer and is cast only between integers or between
# floats, so the module holds no comparison or select the program did not call, and every
# result is added to a total written to a buffer, so no call can be eliminated as dead.
#
# This file is generated from the declarations `tools/gencore.py` writes, so an entry
# added to its table is called here without an edit. Run
# `python3 tools/gencore.py {module} --conform > conform/src/conform_{module}_comp.mach`.

use m: shader.{module};

rec Inputs {{
    u: [2]u32x4;
    i: [2]i32x4;
    f: [2]f32x4;
}}

rec Outputs {{
    u: u32;
    f: f32;
}}

#[gpu.storage(0, 0, "readonly")]
var inputs: Inputs;
#[gpu.storage(0, 1)]
var outputs: Outputs;
'''


def conform(module, decls):
    """A compute stage calling every declaration of `module` once."""
    prelude, made = [], {}

    def value(ty, i):
        # every type alternates between two values, read from the two rows of its kind
        scalar, lanes = (ty.split("x") + ["1"])[:2]
        row = "x" if i % 2 == 0 else "y"
        name = f"{row}_{ty}"
        if name not in made:
            base = {"i": "i32", "u": "u32", "f": "f32"}[scalar[0]]
            cast = "" if scalar == base else f"::{scalar}"
            source = f"inputs.{scalar[0]}[{i % 2}]"
            if lanes == "1":
                init = f"{source}[0]{cast}"
            else:
                init = f"{ty}{{{', '.join(f'{source}[{k}]{cast}' for k in range(int(lanes)))}}}"
            prelude.append(f"    val {name}: {ty} = {init};")
            made[name] = True
        return name

    body = []
    for n, d in enumerate(decls):
        args = [str(p.literal) if p.literal is not None else value(p.ty, i) for i, p in enumerate(d.params)]
        body.append(f"    val r{n}: {d.ret} = m.{d.name}({', '.join(args)});")
        scalar = d.ret.split("x")[0]
        lane = f"r{n}" if "x" not in d.ret else f"r{n}[0]"
        if scalar[0] == "f":
            body.append(f"    total_f = total_f + {lane if scalar == 'f32' else lane + '::f32'};")
        else:
            body.append(f"    total_u = total_u + {lane if scalar == 'u32' else lane + '::u32'};")
    return "\n".join([CONFORM_HEADER.format(module=module), '#[gpu.stage("compute")]', "#[gpu.workgroup(1, 1, 1)]",
                      "fun comp_main() {", *prelude, "    var total_u: u32 = 0;", "    var total_f: f32 = 0.0;", *body,
                      "    outputs.u = total_u;", "    outputs.f = total_f;", "}", ""])


def main():
    args = sys.argv[1:]
    if not args or args[0] not in MODULES or args[1:] not in ([], ["--conform"]):
        sys.exit(f"usage: gencore.py {{{'|'.join(MODULES)}}} [--conform]")
    header, decls = MODULES[args[0]]()
    if args[1:] == ["--conform"]:
        machfmt.emit("gencore", conform(args[0], decls))
    else:
        machfmt.emit("gencore", header + "\n" + "\n\n".join(d.text() for d in decls) + "\n")


main()
