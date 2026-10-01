#!/usr/bin/env python3
"""Check that the conformance shaders call every instruction a module declares.

usage: opcheck.py <module.mach>... -- <conform source or module.spvasm>...

Each `#[op("spirv", set, name)]` declaration in a module must be called, by its name,
from a conformance source, and its instruction must appear in a disassembled
conformance module. The declarations are read from the modules themselves, so an
entry added to the library is checked without editing this script, and an entry no
shader calls, or one whose call never reached the module, fails here.
"""

import re
import sys

OP = re.compile(r'#\[op\("spirv",\s*"([^"]+)",\s*"(\w+)"\)\]\s*\npub fun (\w+)\(')


def main():
    if "--" not in sys.argv:
        sys.exit("usage: opcheck.py <module.mach>... -- <conform source or module.spvasm>...")
    split = sys.argv.index("--")
    modules, evidence = sys.argv[1:split], sys.argv[split + 1:]
    sources = "".join(open(p).read() for p in evidence if p.endswith(".mach"))
    asm = "".join(open(p).read() for p in evidence if p.endswith(".spvasm"))
    called = set(re.findall(r"\.(\w+)\(", sources))
    emitted = set(re.findall(r"= (Op\w+)|^\s*(Op\w+)", asm, re.M))
    emitted = {a or b for a, b in emitted} | set(re.findall(r"OpExtInst %\S+ %\S+ (\w+)", asm))
    failed = False
    count = 0
    for path in modules:
        for iset, name, fun in OP.findall(open(path).read()):
            count += 1
            if fun not in called:
                print(f"::error::{path}: `{fun}` is declared but no conformance shader calls it")
                failed = True
            if name not in emitted:
                print(f"::error::{path}: `{fun}` names {iset} {name}, which no conformance module contains")
                failed = True
    if count == 0:
        sys.exit(f"opcheck: no #[op] declarations found in {' '.join(modules)}")
    if failed:
        sys.exit(1)
    print(f"every declared instruction ({count} declarations) is called and emitted")


main()
