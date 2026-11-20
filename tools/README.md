# tools

## genmath.py

Generates `src/math.mach` from a table of SPIR-V instructions and a list of
vector widths. Edit the table, not the generated file.

```sh
python3 tools/genmath.py > src/math.mach
python3 tools/genmath.py --conform > conform/src/conform_frag.mach
```

With `--conform` it writes the conformance shader instead, a fragment stage that
calls every declaration the library run would write, so a new entry is called
without an edit.

It writes the mach 7 forms and pipes its output through `mach fmt -`, so it needs
**mach 7.0.0 or newer**. It
uses the `mach` on `PATH`, or the compiler the `MACH` environment variable names,
and stops with a clear message when that compiler is older. `conform/verify.sh`
runs the generator both ways and diffs each result against its committed file, so the
table and both files must always be committed together.

## genatomic.py, gensubgroup.py

Generate `src/atomic.mach` and `src/subgroup.mach` from their tables of SPIR-V
instructions and the types each one takes, the same way and with the same compiler
requirement. Both share `machfmt.py` for the formatting step.

```sh
python3 tools/genatomic.py > src/atomic.mach
python3 tools/gensubgroup.py > src/subgroup.mach
```

`conform/verify.sh` diffs each against its committed file.

## gencore.py

Generates `src/compare.mach`, `src/logic.mach` and `src/vector.mach`, the core
instructions no language construct writes, from their tables and the component types
and lane counts each takes, and with `--conform` the conformance shader of each, a
compute stage calling every declaration, with the same compiler requirement.

```sh
python3 tools/gencore.py compare > src/compare.mach
python3 tools/gencore.py compare --conform > conform/src/conform_compare_comp.mach
```

`conform/verify.sh` diffs all six against their committed files.

## opcheck.py

Checks that the conformance shaders call every `#[intrinsic]` a module declares and that
each instruction reaches a disassembled conformance module. `conform/verify.sh`
runs it over every module.

## instcheck.py

Checks that a conformance module holds one instruction for each declaration of the
module it was generated from, keyed by instruction, result type and operand types, so a
call dropped or folded away at one width fails. A truth value is keyed as the integer
the program holds and a literal as a 32-bit word. `conform/verify.sh` runs it over
`shader.math` against `conform_frag`, and over `shader.compare`, `shader.logic` and
`shader.vector` against their compute stages at both optimization levels.

## texcheck.py

Checks that every plain sampled image shape `src/texture.mach` declares is sampled
and combined in `conform_tex_frag`.
