#!/usr/bin/env bash
# the library compiles to nothing, so the checks are on the generator and on the
# conformance shaders built from it. needs spirv-tools.
#   conform/verify.sh [mach]
set -euo pipefail

MACH_COMPILER="${1:-mach}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
scratch="$(mktemp -d "${TMPDIR:-/tmp}/mach-shader-conform.XXXXXX")"
trap 'rm -rf -- "$scratch"' EXIT

"$MACH_COMPILER" dep pull conform
"$MACH_COMPILER" build conform --all

# the tables and the checked-in sources drift the first time one is edited alone
for module in math atomic subgroup; do
  MACH="$MACH_COMPILER" python3 "tools/gen$module.py" | diff -u "src/$module.mach" -
  echo "src/$module.mach matches tools/gen$module.py"
done

# every module at both optimization levels, as SPIR-V and as Vulkan consumes it: the
# Vulkan rules are the ones a pipeline is created against, and some only it checks
for module in conform/out/*/*.spv; do
  spirv-val "$module"
  spirv-val --target-env vulkan1.3 "$module"
  echo "$module: valid"
done

# a decorator can name a plausible instruction that never reaches the module.
# disassembling and counting proves the calls lowered rather than being dropped,
# and that OpDot came through as core rather than as an extended instruction.
spirv-dis conform/out/release/conform_frag.spv > "$scratch/conform.spvasm"
for op in Cross Refract Reflect FaceForward Normalize Length Distance \
          FAbs Sqrt InverseSqrt Fract Floor Ceil Degrees Sin Pow \
          Exp2 Log2 Atan2 FMin FMax FClamp FMix Step SmoothStep Fma; do
  grep -q "OpExtInst .* $op " "$scratch/conform.spvasm" \
    || { echo "error: missing GLSL.std.450 $op"; exit 1; }
done
grep -q 'OpDot' "$scratch/conform.spvasm" || { echo "error: missing core OpDot"; exit 1; }
echo "every checked instruction is present"

# the same proof for the texture module: every image shape the library declares
# is read by a sample and built by a combine, so a dropped call fails here
spirv-dis conform/out/release/conform_tex_frag.spv > "$scratch/conform_tex.spvasm"
python3 tools/texcheck.py src/texture.mach "$scratch/conform_tex.spvasm"

# the compute stage: an entry point that validates can still have lost its
# workgroup size, a built-in or the read-only buffer's decoration, and each of
# those is a pipeline the host builds against a different interface
spirv-dis conform/out/release/conform_comp.spv > "$scratch/conform_comp.spvasm"
for want in 'OpEntryPoint GLCompute %[0-9a-z_]* "comp_main"' \
            'OpExecutionMode %[0-9a-z_]* LocalSize 64 1 1' \
            'BuiltIn GlobalInvocationId' 'BuiltIn LocalInvocationId' 'BuiltIn WorkgroupId' \
            'OpDecorate %[0-9a-z_]* NonWritable' 'StorageBuffer'; do
  grep -q "$want" "$scratch/conform_comp.spvasm" \
    || { echo "error: conform_comp is missing: $want"; exit 1; }
done
echo "the compute stage carries its workgroup size, built-ins and buffers"

# every declared instruction is called by a conformance shader and reaches its module
for module in conform/out/release/*.spv; do
  spirv-dis "$module" > "$scratch/$(basename "$module" .spv).spvasm"
done
python3 tools/opcheck.py src/texture.mach src/sync.mach src/atomic.mach src/subgroup.mach \
  -- conform/src/*.mach "$scratch"/*.spvasm
