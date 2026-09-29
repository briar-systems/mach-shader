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

# the same proof for the texture module: every image shape the library declares
# is read by a sample and built by a combine, so a dropped call fails here
spirv-dis conform/out/conform_tex_frag.spv > "$RUNNER_TEMP/conform_tex.spvasm"
python3 tools/texcheck.py src/texture.mach "$RUNNER_TEMP/conform_tex.spvasm"

# the compute stage: an entry point that validates can still have lost its
# workgroup size, a built-in or the read-only buffer's decoration, and each of
# those is a pipeline the host builds against a different interface
spirv-dis conform/out/conform_comp.spv > "$RUNNER_TEMP/conform_comp.spvasm"
for want in 'OpEntryPoint GLCompute %[0-9a-z_]* "comp_main"' \
            'OpExecutionMode %[0-9a-z_]* LocalSize 64 1 1' \
            'BuiltIn GlobalInvocationId' 'BuiltIn LocalInvocationId' 'BuiltIn WorkgroupId' \
            'OpDecorate %[0-9a-z_]* NonWritable' 'StorageBuffer'; do
  grep -q "$want" "$RUNNER_TEMP/conform_comp.spvasm" \
    || { echo "::error::conform_comp is missing: $want"; exit 1; }
done
echo "the compute stage carries its workgroup size, built-ins and buffers"
