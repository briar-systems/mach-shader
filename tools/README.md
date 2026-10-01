# tools

## genmath.py

Generates `src/math.mach` from a table of SPIR-V instructions and a list of
vector widths. Edit the table, not the generated file.

```sh
python3 tools/genmath.py > src/math.mach
```

It pipes its output through `mach fmt -`, so it needs **mach 5.1.0 or newer**. It
uses the `mach` on `PATH`, or the compiler the `MACH` environment variable names,
and stops with a clear message when that compiler is older. `conform/verify.sh`
runs the generator and diffs the result against the committed file, so the two must always be
committed together.

## genatomic.py, gensubgroup.py

Generate `src/atomic.mach` and `src/subgroup.mach` from their tables of SPIR-V
instructions and the types each one takes, the same way and with the same compiler
requirement. Both share `machfmt.py` for the formatting step.

```sh
python3 tools/genatomic.py > src/atomic.mach
python3 tools/gensubgroup.py > src/subgroup.mach
```

`conform/verify.sh` diffs each against its committed file.

## opcheck.py

Checks that the conformance shaders call every `#[op]` a module declares and that
each instruction reaches a disassembled conformance module. `conform/verify.sh`
runs it over `shader.texture`, `shader.sync`, `shader.atomic` and `shader.subgroup`.

## texcheck.py

Checks that every plain sampled image shape `src/texture.mach` declares is sampled
and combined in `conform_tex_frag`.
