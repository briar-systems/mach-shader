#!/usr/bin/env python3
"""Check that a disassembled module holds one instruction per declaration of a module.

usage: instcheck.py <module.mach> <module.spvasm>

`tools/opcheck.py` proves each instruction name reaches some module, which a single
surviving width satisfies. This check keys every `#[intrinsic]` declaration of one module
by its instruction, result type and operand types, and requires the conformance module
built from it to hold exactly as many of each as are declared, so a call that was dropped
or folded away at one width fails here. An extended instruction is keyed by its set, and a
core one as core, so neither can pass as the other. The expected entries come from the
module itself, so a new entry is checked without editing this script.

SPIR-V integers carry no sign here, so `i32` and `u32` are one type in the module, and an
instruction both signednesses share, `FindILsb`, is expected twice at each width. A
literal operand is a 32-bit word, so it is keyed as a 32-bit integer. A `rec` of `T`
members is keyed as the structure of those members. A truth value is the integer the
program holds: the compiler gives a truth operand as that integer compared not equal to
zero, and turns a truth result into the integer by a select of the constants 1 and 0, so
each is keyed by the integer's type, and those selects are the compiler's, not counted.
"""

import collections
import re
import sys

OP = re.compile(r'#\[intrinsic\("spirv",\s*"([^"]+)",\s*"(\w+)"\)\]\s*\npub fun (\w+)\(([^)]*)\) ([\w\[\]]+);')
RECORD = re.compile(r"pub rec (\w+)\[T\] \{([^}]*)\}")


def signless(ty, records):
    pointer = ty.startswith("*")
    ty = ty.lstrip("*")
    record = re.fullmatch(r"(\w+)\[(\w+)\]", ty)
    if record:
        member = signless(record.group(2), records)
        return "struct(" + ",".join(member for _ in records[record.group(1)]) + ")"
    m = re.fullmatch(r"([iuf])(\d+)(?:x(\d))?", ty)
    kind, bits, lanes = m.groups()
    out = ("float" if kind == "f" else "int") + bits + (f"x{lanes}" if lanes else "")
    return "*" + out if pointer else out


def declared(path):
    text = open(path).read()
    records = {name: re.findall(r"(\w+):\s*T;", fields) for name, fields in RECORD.findall(text)}
    want = collections.Counter()
    names = {}
    for iset, name, fun, params, ret in OP.findall(text):
        args = tuple(signless(p.split(": ")[1], records) for p in params.split(", "))
        key = (iset, name, signless(ret, records), args)
        want[key] += 1
        names.setdefault(key, []).append(fun)
    return want, names


def emitted(path, core):
    types, values, constants = {}, {}, set()
    have = collections.Counter()
    glsl = None
    found = []
    for line in open(path).read().splitlines():
        m = re.match(r"\s*(%\S+) = (Op\w+)(.*)", line)
        if not m:
            continue
        rid, op, rest = m.group(1), m.group(2), m.group(3).split()
        if op == "OpTypeFloat":
            types[rid] = f"float{rest[0]}"
        elif op == "OpTypeInt":
            types[rid] = f"int{rest[0]}"
        elif op == "OpTypeBool":
            types[rid] = "bool"
        elif op == "OpTypeVector" and rest[0] in types:
            types[rid] = f"{types[rest[0]]}x{rest[1]}"
        elif op == "OpTypePointer" and rest[1] in types:
            types[rid] = "*" + types[rest[1]]
        elif op == "OpTypeStruct" and all(r in types for r in rest):
            types[rid] = "struct(" + ",".join(types[r] for r in rest) + ")"
        elif op == "OpExtInstImport" and rest[0] == '"GLSL.std.450"':
            glsl = rid
        elif rest and rest[0] in types:
            values[rid] = types[rest[0]]
            if op.startswith(("OpConstant", "OpSpecConstant")):
                constants.add(rid)
            found.append((rid, op, types[rest[0]], rest[1:]))

    # the compiler's truth conversions: an integer compared not equal to a constant zero
    # gives a truth operand, and a select of two constants gives a truth result its integer
    truth_in, truth_out, conversions = {}, {}, set()
    for rid, op, ty, operands in found:
        if op == "OpINotEqual" and ty.startswith("bool") and operands[1] in constants:
            truth_in[rid] = values.get(operands[0], "?")
        elif (op == "OpSelect" and not ty.startswith("bool") and values.get(operands[0], "").startswith("bool")
              and operands[1] in constants and operands[2] in constants):
            truth_out[operands[0]] = ty
            conversions.add(rid)

    def operand(token):
        if not token.startswith("%"):
            return "int32"
        ty = values.get(token, "?")
        return truth_in.get(token, ty) if ty.startswith("bool") else ty

    # a float cast to an integer saturates, through an NMax and an NMin the compiler emits
    # ahead of the conversion: those are no maths entry's, so they are not counted
    keys, converted, clamps = {}, set(), set()
    for rid, op, ty, operands in found:
        if op == "OpExtInst" and operands[0] == glsl:
            keys[rid] = (("GLSL.std.450", operands[1], ty, tuple(values.get(a, "?") for a in operands[2:])), operands[2:])
        elif op in ("OpConvertFToS", "OpConvertFToU"):
            converted.add(operands[0])
        elif op in core and rid not in conversions:
            result = truth_out.get(rid, ty) if ty.startswith("bool") else ty
            have[("core", op, result, tuple(operand(a) for a in operands))] += 1
    for rid, (key, operands) in keys.items():
        if key[1] == "NMin" and rid in converted and operands and operands[0] in keys and keys[operands[0]][0][1] == "NMax":
            clamps.update((rid, operands[0]))
    for rid, (key, _) in keys.items():
        if rid not in clamps:
            have[key] += 1
    return have


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: instcheck.py <module.mach> <module.spvasm>")
    want, names = declared(sys.argv[1])
    if not want:
        sys.exit(f"instcheck: no #[intrinsic] declarations found in {sys.argv[1]}")
    # every core instruction the module declares is counted wherever the module holds it
    core = {key[1] for key in want if key[0] == "core"}
    have = emitted(sys.argv[2], core)
    failed = False
    for key in sorted(set(want) | set(have)):
        if want[key] != have[key]:
            iset, name, ret, args = key
            funs = ", ".join(f"`{f}`" for f in names.get(key, [])) or "no declaration"
            print(f"::error::{sys.argv[2]}: {iset} {name} {ret}({', '.join(args)}) appears {have[key]} times, "
                  f"expected {want[key]} for {funs}")
            failed = True
    if failed:
        sys.exit(1)
    print(f"every entry of {sys.argv[1]} ({sum(want.values())} declarations) has its own instruction in the module")


main()
