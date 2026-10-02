#!/usr/bin/env python3
"""Check that a disassembled module holds one instruction per declared maths entry.

usage: mathcheck.py <math.mach> <module.spvasm>

`tools/opcheck.py` proves each instruction name reaches some module, which a single
surviving width satisfies. This check keys every `#[op]` declaration by its
instruction, result type and operand types, and requires the module to hold exactly
as many of each as are declared, so a call that was dropped or folded away at one
width fails here. `OpDot` is keyed as core, so it cannot pass as an extended
instruction. The expected entries come from the library itself, so a new entry
is checked without editing this script. SPIR-V integers carry no sign here, so
`i32` and `u32` are one type in the module, and an instruction both signednesses
share, `FindILsb`, is expected twice at each width.
"""

import collections
import re
import sys

OP = re.compile(r'#\[op\("spirv",\s*"([^"]+)",\s*"(\w+)"\)\]\s*\npub fun (\w+)\(([^)]*)\) (\w+);')


def signless(ty):
    pointer = ty.startswith("*")
    ty = ty.lstrip("*")
    m = re.fullmatch(r"([iuf])(\d+)(?:x(\d))?", ty)
    kind, bits, lanes = m.groups()
    out = ("float" if kind == "f" else "int") + bits + (f"x{lanes}" if lanes else "")
    return "*" + out if pointer else out


def declared(path):
    want = collections.Counter()
    names = {}
    for iset, name, fun, params, ret in OP.findall(open(path).read()):
        args = tuple(signless(p.split(": ")[1]) for p in params.split(", "))
        key = (iset, name, signless(ret), args)
        want[key] += 1
        names.setdefault(key, []).append(fun)
    return want, names


def emitted(path):
    types, values = {}, {}
    have = collections.Counter()
    glsl = None
    for line in open(path).read().splitlines():
        m = re.match(r"\s*(%\S+) = (Op\w+)(.*)", line)
        if not m:
            continue
        rid, op, rest = m.group(1), m.group(2), m.group(3).split()
        if op == "OpTypeFloat":
            types[rid] = f"float{rest[0]}"
        elif op == "OpTypeInt":
            types[rid] = f"int{rest[0]}"
        elif op == "OpTypeVector" and rest[0] in types:
            types[rid] = f"{types[rest[0]]}x{rest[1]}"
        elif op == "OpTypePointer" and rest[1] in types:
            types[rid] = "*" + types[rest[1]]
        elif op == "OpExtInstImport" and rest[0] == '"GLSL.std.450"':
            glsl = rid
        elif rest and rest[0] in types:
            values[rid] = types[rest[0]]
            if op == "OpExtInst" and rest[1] == glsl:
                have[("GLSL.std.450", rest[2], types[rest[0]], tuple(values.get(a, "?") for a in rest[3:]))] += 1
            elif op == "OpDot":
                have[("core", "OpDot", types[rest[0]], tuple(values.get(a, "?") for a in rest[1:]))] += 1
    return have


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: mathcheck.py <math.mach> <module.spvasm>")
    want, names = declared(sys.argv[1])
    have = emitted(sys.argv[2])
    if not want:
        sys.exit(f"mathcheck: no #[op] declarations found in {sys.argv[1]}")
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
    print(f"every maths entry ({sum(want.values())} declarations) has its own instruction in the module")


main()
