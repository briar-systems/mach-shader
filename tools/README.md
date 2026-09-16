# tools

## genmath.py

Generates `src/math.mach` from a table of SPIR-V instructions and a list of
vector widths. Edit the table, not the generated file.

```sh
python3 tools/genmath.py > src/math.mach
```

It pipes its output through `mach fmt -`, so it needs **mach 5.1.0 or newer**. It
uses the `mach` on `PATH`, or the compiler the `MACH` environment variable names,
and stops with a clear message when that compiler is older. CI runs the generator
and diffs the result against the committed file, so the two must always be
committed together.
