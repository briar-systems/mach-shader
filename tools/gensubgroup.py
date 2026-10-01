#!/usr/bin/env python3
"""Generate mach-shader's src/subgroup.mach.

The module is the SPIR-V `OpGroupNonUniform*` families applied to every value type
they take, so it is generated rather than hand-maintained: adding a type or an
instruction is an edit to the tables below.
"""

import sys

sys.dont_write_bytecode = True
import machfmt

HEADER = '''# Shader-side subgroup operations: the invocations of one subgroup computing together,
# without a barrier or shared memory.
#
# Every declaration here is one SPIR-V `OpGroupNonUniform*` instruction, named by
# `#[op]`, as in `shader.math`. Each takes the `SCOPE_SUBGROUP` from `shader.sync` as
# its first operand, the one scope Vulkan admits, written at the call.
#
# EACH FAMILY IS A DEVICE FEATURE. Every subgroup operation needs SPIR-V 1.3, so
# `vulkan1.1` or later. Elect is every such device's, and each other family needs the
# target's `extensions` to name the feature Vulkan reports it by: `subgroup_vote`
# (all, any, all-equal), `subgroup_arithmetic` (a reduction or scan),
# `subgroup_clustered` (a clustered reduction), `subgroup_ballot` (ballot and
# broadcast), `subgroup_shuffle`, `subgroup_shuffle_relative` (shuffle up and down)
# and `subgroup_quad`. Vulkan guarantees them only in a compute stage, so a use from a
# vertex or fragment stage also needs `subgroup_graphics_stages`. A call whose
# feature the target does not select is refused there, naming it.
#
# CONSTANT OPERANDS. A `GROUP_*` operation, a broadcast's lane, a quad broadcast's
# index and a quad swap's direction are constants written at the call, never values
# computed at run time.
#
# TRUTH VALUES. A predicate is an integer, true where it is nonzero, and an operation
# that answers yes or no returns 1 or 0.
#
# THE NAMING RULE. An entry over a value the caller chooses carries that value's type
# after an underscore, as `shader.atomic` does: `subgroup_add_u32`, `subgroup_add_f32`.
# The signed and unsigned forms are distinct where the instruction is, so
# `subgroup_min_i32` is `OpGroupNonUniformSMin`. An entry whose operand types are
# fixed, a ballot's `u32x4` or a predicate, takes the plain name.
#
# THIS IS A SHADER-ONLY MODULE. On any target other than SPIR-V a call to one of these
# fails to link, for the reason `shader.math` gives.
#
# This file is generated. See tools/gensubgroup.py.

# the `GroupOperation` of a reduction or scan: the whole subgroup's result, each
# invocation's result over itself and the lower ones, the same excluding itself, and
# a reduction over each cluster of `cluster_size` invocations
pub val GROUP_REDUCE:           u32 = 0;
pub val GROUP_INCLUSIVE_SCAN:   u32 = 1;
pub val GROUP_EXCLUSIVE_SCAN:   u32 = 2;
pub val GROUP_CLUSTERED_REDUCE: u32 = 3;

# the direction a quad swap exchanges values in, within a quad of four invocations
pub val QUAD_SWAP_HORIZONTAL: u32 = 0;
pub val QUAD_SWAP_VERTICAL:   u32 = 1;
pub val QUAD_SWAP_DIAGONAL:   u32 = 2;
'''

TYPES = ["u32", "i32", "f32"]
INTEGER_TYPES = ["u32", "i32"]

SCOPE = ("scope", "u32", "`SCOPE_SUBGROUP`.")
OPERATION = ("operation", "u32", "A `GROUP_*` reduction or scan.")
CLUSTERED = ("operation", "u32", "`GROUP_CLUSTERED_REDUCE`.")
CLUSTER = ("cluster_size", "u32", "The invocations in each cluster, a constant power of two.")
PREDICATE = ("predicate", "u32", "The truth value, true where nonzero.")
BALLOT = ("ballot", "u32x4", "A ballot, one bit per invocation.")


def decl(name, op, summary, params, ret_ty, ret_doc):
    lines = [f"# {summary}", "# ---"]
    width = max(len(n) for n, _, _ in params + [("ret", "", "")]) + 1
    for n, _, d in params:
        lines.append(f"# {n + ':':<{width}} {d}")
    lines.append(f"# {'ret:':<{width}} {ret_doc}")
    lines.append(f'#[op("spirv", "core", "OpGroupNonUniform{op}")]')
    args = ", ".join(f"{n}: {t}" for n, t, _ in params)
    lines.append(f"pub fun {name}({args}) {ret_ty};")
    return "\n".join(lines)


out = [HEADER]

out.append(decl("subgroup_elect", "Elect", "Whether this is the lowest active invocation of the subgroup.",
                [SCOPE], "u32", "1 in exactly one active invocation, 0 in the others"))
out.append(decl("subgroup_all", "All", "Whether a predicate holds in every active invocation.",
                [SCOPE, PREDICATE], "u32", "1 where it holds in all of them, else 0"))
out.append(decl("subgroup_any", "Any", "Whether a predicate holds in any active invocation.",
                [SCOPE, PREDICATE], "u32", "1 where it holds in at least one, else 0"))
out.append(decl("subgroup_ballot", "Ballot", "The set of active invocations a predicate holds in.",
                [SCOPE, PREDICATE], "u32x4", "one bit per invocation, set where it holds, invocation 0 the lowest bit of lane 0"))
out.append(decl("subgroup_inverse_ballot", "InverseBallot", "Whether this invocation's bit is set in a ballot.",
                [SCOPE, BALLOT], "u32", "1 where it is set, else 0"))
out.append(decl("subgroup_ballot_bit_extract", "BallotBitExtract", "Whether one invocation's bit is set in a ballot.",
                [SCOPE, BALLOT, ("index", "u32", "The invocation whose bit is read.")], "u32", "1 where it is set, else 0"))
out.append(decl("subgroup_ballot_bit_count", "BallotBitCount", "Count the set bits of a ballot, over the whole subgroup or as a scan.",
                [SCOPE, ("operation", "u32", "`GROUP_REDUCE`, `GROUP_INCLUSIVE_SCAN` or `GROUP_EXCLUSIVE_SCAN`."), BALLOT],
                "u32", "the bits set over the invocations the operation covers"))
out.append(decl("subgroup_ballot_find_lsb", "BallotFindLSB", "The lowest invocation whose bit is set in a ballot.",
                [SCOPE, BALLOT], "u32", "its index"))
out.append(decl("subgroup_ballot_find_msb", "BallotFindMSB", "The highest invocation whose bit is set in a ballot.",
                [SCOPE, BALLOT], "u32", "its index"))

for t in TYPES:
    v = ("value", t, "This invocation's value.")
    out.append(decl(f"subgroup_all_equal_{t}", "AllEqual", "Whether a value is the same in every active invocation.",
                    [SCOPE, v], "u32", "1 where it is, else 0"))
    out.append(decl(f"subgroup_broadcast_{t}", "Broadcast", "One invocation's value, in every invocation.",
                    [SCOPE, v, ("id", "u32", "The invocation to read, a constant.")], t, "that invocation's value"))
    out.append(decl(f"subgroup_broadcast_first_{t}", "BroadcastFirst", "The lowest active invocation's value, in every invocation.",
                    [SCOPE, v], t, "that invocation's value"))
    out.append(decl(f"subgroup_shuffle_{t}", "Shuffle", "Another invocation's value, chosen per invocation.",
                    [SCOPE, v, ("id", "u32", "The invocation to read.")], t, "that invocation's value"))
    out.append(decl(f"subgroup_shuffle_xor_{t}", "ShuffleXor", "The value of the invocation whose index is this one's xor a mask.",
                    [SCOPE, v, ("mask", "u32", "The mask.")], t, "that invocation's value"))
    out.append(decl(f"subgroup_shuffle_up_{t}", "ShuffleUp", "The value of the invocation a distance below this one.",
                    [SCOPE, v, ("delta", "u32", "The distance.")], t, "that invocation's value"))
    out.append(decl(f"subgroup_shuffle_down_{t}", "ShuffleDown", "The value of the invocation a distance above this one.",
                    [SCOPE, v, ("delta", "u32", "The distance.")], t, "that invocation's value"))
    out.append(decl(f"subgroup_quad_broadcast_{t}", "QuadBroadcast", "One invocation's value, in every invocation of its quad.",
                    [SCOPE, v, ("index", "u32", "The invocation within the quad, `0` to `3`, a constant.")], t, "that invocation's value"))
    out.append(decl(f"subgroup_quad_swap_{t}", "QuadSwap", "The value of this invocation's neighbour in its quad.",
                    [SCOPE, v, ("direction", "u32", "A `QUAD_SWAP_*` direction, a constant.")], t, "the neighbour's value"))

# name, instruction per type (None where the type has none), summary
ARITHMETIC = [
    ("add", {"u32": "IAdd", "i32": "IAdd", "f32": "FAdd"}, "The sum"),
    ("mul", {"u32": "IMul", "i32": "IMul", "f32": "FMul"}, "The product"),
    ("min", {"u32": "UMin", "i32": "SMin", "f32": "FMin"}, "The least"),
    ("max", {"u32": "UMax", "i32": "SMax", "f32": "FMax"}, "The greatest"),
    ("and", {"u32": "BitwiseAnd", "i32": "BitwiseAnd"}, "The bitwise and"),
    ("or",  {"u32": "BitwiseOr", "i32": "BitwiseOr"}, "The bitwise or"),
    ("xor", {"u32": "BitwiseXor", "i32": "BitwiseXor"}, "The bitwise xor"),
]

for name, ops, what in ARITHMETIC:
    for t, op in ops.items():
        v = ("value", t, "This invocation's value.")
        out.append(decl(f"subgroup_{name}_{t}", op, f"{what} of a value over the subgroup, or a scan of it.",
                        [SCOPE, OPERATION, v], t, "the result over the invocations the operation covers"))
        out.append(decl(f"subgroup_clustered_{name}_{t}", op, f"{what} of a value over each cluster of invocations.",
                        [SCOPE, CLUSTERED, v, CLUSTER], t, "the result over this invocation's cluster"))

for name, op in [("and", "LogicalAnd"), ("or", "LogicalOr"), ("xor", "LogicalXor")]:
    out.append(decl(f"subgroup_logical_{name}", op, f"The logical {name} of a predicate over the subgroup, or a scan of it.",
                    [SCOPE, OPERATION, PREDICATE], "u32", "1 or 0 over the invocations the operation covers"))
    out.append(decl(f"subgroup_clustered_logical_{name}", op, f"The logical {name} of a predicate over each cluster of invocations.",
                    [SCOPE, CLUSTERED, PREDICATE, CLUSTER], "u32", "1 or 0 over this invocation's cluster"))

machfmt.emit("gensubgroup", "\n\n".join(out) + "\n")
