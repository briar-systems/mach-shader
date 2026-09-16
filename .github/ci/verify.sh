#!/usr/bin/env bash
# the library compiles to nothing, so the checks are on the generator and on the
# conformance shaders built from it
set -euo pipefail

# the table and the checked-in source drift the first time one is edited alone
MACH="$MACH_COMPILER" python3 tools/genmath.py | diff -u src/math.mach -
echo "src/math.mach matches tools/genmath.py"

for module in conform/out/*.spv; do
  spirv-val "$module"
  echo "$module: valid"
done

# a decorator can name a plausible instruction that never reaches the module.
# disassembling and counting proves the calls lowered rather than being dropped,
# and that OpDot came through as core rather than as an extended instruction.
spirv-dis conform/out/conform_frag.spv > "$RUNNER_TEMP/conform.spvasm"
for op in Cross Refract Reflect FaceForward Normalize Length Distance \
          FAbs Sqrt InverseSqrt Fract Floor Ceil Degrees Sin Pow \
          Exp2 Log2 Atan2 FMin FMax FClamp FMix Step SmoothStep Fma; do
  grep -q "OpExtInst .* $op " "$RUNNER_TEMP/conform.spvasm" \
    || { echo "::error::missing GLSL.std.450 $op"; exit 1; }
done
grep -q 'OpDot' "$RUNNER_TEMP/conform.spvasm" || { echo "::error::missing core OpDot"; exit 1; }
echo "every checked instruction is present"
