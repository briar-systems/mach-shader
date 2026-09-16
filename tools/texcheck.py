#!/usr/bin/env python3
"""Check that a disassembled module samples and combines every declared image shape.

usage: texcheck.py <texture.mach> <module.spvasm>

The expected shapes come from the `#[handle("spirv", "image", ...)]` declarations,
so a handle added to the library is checked without editing this script. A shape
is (texel scalar, dimensionality, arrayed) as SPIR-V spells it. SPIR-V integers
carry no sign here, so `TEXEL_I32` and `TEXEL_U32` images are one shape in the
module, and the signed and unsigned entries cannot be told apart by type.
"""

import re
import sys

TEXEL = {"TEXEL_F32": "float", "TEXEL_I32": "int", "TEXEL_U32": "int"}
DIM = {"DIM_1D": "1D", "DIM_2D": "2D", "DIM_3D": "3D", "DIM_CUBE": "Cube"}
ARRAYED = {"NONARRAYED": "0", "ARRAYED": "1"}

HANDLE = re.compile(r'#\[handle\("spirv",\s*"image",\s*(\w+),\s*(\w+),\s*\w+,\s*(\w+),')


def declared(path):
    shapes = {}
    lines = open(path).read().splitlines()
    for i, line in enumerate(lines):
        m = HANDLE.match(line.strip())
        if not m:
            continue
        name = re.match(r"pub def (\w+);", lines[i + 1].strip()).group(1)
        texel, dim, arrayed = m.groups()
        shapes.setdefault((TEXEL[texel], DIM[dim], ARRAYED[arrayed]), []).append(name)
    return shapes


def emitted(path):
    scalar, image, sampled, pointee, typeof = {}, {}, {}, {}, {}
    samples, combines = [], []
    for line in open(path):
        m = re.match(r"\s*(%\S+) = (Op\w+)\s*(.*)", line)
        if not m:
            continue
        rid, op, args = m.group(1), m.group(2), m.group(3).split()
        if op == "OpTypeFloat":
            scalar[rid] = "float"
        elif op == "OpTypeInt":
            scalar[rid] = "int"
        elif op == "OpTypeImage":
            image[rid] = (scalar[args[0]], args[1], args[3])
        elif op == "OpTypeSampledImage":
            sampled[rid] = args[0]
        elif op == "OpTypePointer":
            pointee[rid] = args[1]
        elif op == "OpVariable":
            typeof[rid] = pointee[args[0]]
        elif args and not op.startswith("OpType") and op not in ("OpConstant", "OpFunction"):
            typeof[rid] = args[0]
        if op == "OpImageSampleImplicitLod":
            samples.append(args[1])
        elif op == "OpSampledImage":
            combines.append(args[0])
    sampled_shape = {image[sampled[typeof[si]]] for si in samples}
    combined_shape = {image[sampled[t]] for t in combines}
    return sampled_shape, combined_shape


def main():
    shapes = declared(sys.argv[1])
    if not shapes:
        sys.exit(f"texcheck: no image handles found in {sys.argv[1]}")
    sampled, combined = emitted(sys.argv[2])
    failed = False
    for shape, names in sorted(shapes.items()):
        label = f"{'/'.join(names)} ({' '.join(shape)})"
        for what, seen in (("OpImageSampleImplicitLod", sampled), ("OpSampledImage", combined)):
            if shape not in seen:
                print(f"::error::no {what} over {label}")
                failed = True
    if failed:
        sys.exit(1)
    print(f"every declared image shape ({len(shapes)}) is sampled and combined")


main()
