# mach-shader

Shader-side maths for [Mach](https://github.com/briar-systems/mach): the functions
that lower to a single SPIR-V instruction.

```mach
use sh: shader.math;

#[gpu.input(0)]  var in_normal: f32x3;
#[gpu.input(1)]  var in_light:  f32x3;
#[gpu.output(0)] var out_colour: f32x4;

#[gpu.stage("fragment")]
fun frag_main() {
    val n: f32x3 = sh.normalize_3(in_normal);
    val d: f32   = sh.max(sh.dot_3(n, in_light), 0.0);
    out_colour   = sh.mix_4(ambient, albedo, f32x4{d, d, d, d});
}
```

`sh.normalize_3` does not compile to a call. Each declaration in `shader.math`
carries an `#[intrinsic(target, set, name)]` decorator naming the SPIR-V instruction it *is*,
and the compiler substitutes that instruction at the call site: an `OpExtInst` into
`GLSL.std.450` for most of them, core `OpDot` for the `dot_*` family. The extended instruction
set is imported once per module and only when something in that module uses it.

## Textures

`shader.texture` declares the image and sampler handles and the instructions that
read them. A handle is a bodyless `def` carrying `#[handle(target, constructor,
operands...)]`: the compiler knows only that a declaration with no body has its
definition supplied by the owning target, and the SPIR-V target owns what a
constructor is and what each operand means.

```mach
use tex: shader.texture;

#[gpu.sampler(0, 0)] var albedo: tex.Sampler2D;
#[gpu.input(0)] var in_uv: f32x2;
#[gpu.output(0)] var out_colour: f32x4;

#[gpu.stage("fragment")]
fun frag_main() {
    out_colour = tex.sample_2d(albedo, in_uv);
}
```

A `Sampler2D` is an `OpTypeSampledImage` over an `OpTypeImage`, and it NAMES the
image it wraps rather than restating that image's operands, so the two cannot
disagree about the shape they share. Adding a handle is a declaration in that file
rather than a compiler release.

A handle's operands are the seven `OpTypeImage` operands, the last of them the image
format, which is `FORMAT_UNKNOWN` for every sampled image. The module names each
operand value, the 42 formats included, so a shape the library does not declare is
one `def` in the consumer's own source.

Beside the implicit-lod samples it declares the explicit-lod ones a compute stage
samples with (`sample_lod_*`, `sample_grad_*`), the depth images and their comparisons
(`sample_compare_*`, `sample_compare_lod_*`, `gather_compare_*`), gathers
(`gather_*`), integer-coordinate fetches (`fetch_*`) and the size and level queries
(`size_*`, `levels_*`), and the image a combined handle is built over (`image_of_*`).
A multisampled image is never sampled and is not declared, and
the target refuses the `Rect` dimensionality by naming the operand.

### Storage images and texel buffers

A storage image is read and written texel by texel, with no sampler, and binds
through `#[gpu.storage]` like a buffer, `"readonly"`, `"writeonly"` and `"coherent"`
included:

```mach
use tex: shader.texture;

#[gpu.storage(0, 0, "readonly")]  var src: tex.StorageImage2D;
#[gpu.storage(0, 1, "writeonly")] var dst: tex.StorageImage2D;
#[gpu.builtin("global_invocation")] var gid: u32x3;

#[gpu.stage("compute")]
#[gpu.workgroup(8, 8, 1)]
fun blit() {
    val at: i32x2 = i32x2{gid[0]::i32, gid[1]::i32};
    tex.image_write_2d(dst, at, tex.image_read_2d(src, at));
}
```

The `StorageImage*` handles are `FORMAT_UNKNOWN`, so one handle binds a view of any
format, under the `StorageImageReadWithoutFormat` and `StorageImageWriteWithoutFormat`
capabilities, which `vulkan1.3` selects and an earlier `env` needs named in the target's
`ext`. The `*R32f`, `*R32i`, `*R32ui`, `*R64i` and
`*R64ui` images carry the formats Vulkan defines an image atomic on, and
`image_texel_*` gives an atomic in `shader.atomic` the address of one of their
texels. A texel buffer is a `TexelBuffer`, fetched and bound by `#[gpu.sampler]`, or a
`StorageTexelBuffer`, read and written and bound by `#[gpu.storage]`.

### Subpass inputs

A subpass input reads the attachment a previous subpass of the render pass wrote, at
the fragment's own place. It binds through `#[gpu.storage]` beside
`#[gpu.attachment(index)]`, the input attachment it reads, which the compiler requires,
and only a fragment stage may reach it:

```mach
use tex: shader.texture;

#[gpu.storage(0, 0)] #[gpu.attachment(0)] var lit: tex.SubpassInput;
#[gpu.output(0)] var colour: f32x4;

#[gpu.stage("fragment")]
fun resolve() {
    colour = tex.subpass_load(lit, i32x2{0, 0});
}
```

`SubpassInput`, `ISubpassInput` and `USubpassInput` read float, signed and unsigned
attachments, and `SubpassInputMS` one sample of a multisampled attachment through
`subpass_load_ms`. The coordinate is relative to the fragment, so it is always
`i32x2{0, 0}`.

## Compute

A compute stage is a `#[gpu.stage("compute")]` function sized by `#[gpu.workgroup(x, y,
z)]`, and it emits a GLCompute entry point with that `LocalSize`. It reads and
writes `#[gpu.storage(set, binding)]` buffers rather than varyings, since Vulkan has no
`Output` class in a compute stage, and a buffer marked `"readonly"` is emitted
`NonWritable`. Its ids are the `global_invocation`, `local_invocation` and
`workgroup_id` built-ins, each a `u32x3`. The maths in `shader.math` works in a
compute body exactly as it does in a fragment one.

```mach
use sh: shader.math;

rec Field { values: [256]f32; }

#[gpu.storage(0, 0, "readonly")] var src: Field;
#[gpu.storage(0, 1)]             var dst: Field;
#[gpu.builtin("global_invocation")] var gid: u32x3;

#[gpu.stage("compute")]
#[gpu.workgroup(64, 1, 1)]
fun step() {
    val i: u32 = gid[0];
    dst.values[i] = sh.sqrt(src.values[i]);
}
```

All of that is the compiler's. What a compute body calls is this library's, each
one a bodyless `#[intrinsic]` declaration like the maths:

- `shader.sync`: the `SCOPE_*` and `SEMANTICS_*` operand constants,
  `control_barrier` and `memory_barrier`, and the wrappers `workgroup_barrier`,
  `storage_barrier`, `image_barrier` and `subgroup_barrier`, and `terminate_invocation`,
  which ends a fragment's invocation and the block its call stands in.
- `shader.atomic`: every SPIR-V atomic over `u32`, `i32`, `u64` and `i64`, and the
  load, store, exchange, add, min and max over `f16`, `f32` and `f64`.
- `shader.subgroup`: the `OpGroupNonUniform*` families, the `GROUP_*` operations and
  the `QUAD_SWAP_*` directions.
- `shader.texture`: the storage images and texel buffers above.

```mach
use at: shader.atomic;
use sy: shader.sync;

rec Histogram { bins: [64]u32; }

#[gpu.storage(0, 0)] var histogram: Histogram;
#[gpu.shared]        var local_bins: [64]u32;
#[gpu.builtin("local_invocation_index")] var lid: u32;

#[gpu.stage("compute")]
#[gpu.workgroup(64, 1, 1)]
fun count() {
    at.atomic_add_u32(?local_bins[lid & 63], sy.SCOPE_WORKGROUP, sy.SEMANTICS_RELAXED, 1);
    sy.workgroup_barrier();
    at.atomic_add_u32(?histogram.bins[lid], sy.SCOPE_DEVICE, sy.SEMANTICS_RELAXED, local_bins[lid]);
}
```

An atomic's pointer takes its storage class from the call, so one declaration
covers a storage buffer, workgroup memory and an image texel. A scope, a memory
semantics, a group operation and a broadcast lane are constants SPIR-V takes by
value, and each is refused at the call unless it is a literal or a `pub val`.

That is also why the wrappers are only the four barriers. Each writes its operands as
constants in its own body, so they reach the instruction as constants whether the
wrapper is inlined (O2) or called (O0), and `conform_sync_comp` holds that at both
levels. A wrapper that took a scope or semantics as a parameter would compile at O2,
where inlining folds it, and be refused at O0, so none is shipped.

Every type other than a 32-bit integer atomic, and every subgroup family but
`Elect`, needs a Vulkan device feature, which the target names in its `ext` by the SPIR-V
capability it enables (`AtomicFloat32AddEXT`, `GroupNonUniformArithmetic`, and so on). The module headers
list them, and a call whose feature the target does not select is refused with its
name.

## Comparisons, logic and lanes

Three modules hold the core instructions no language construct writes, each a
bodyless `#[intrinsic]` declaration like the maths:

- `shader.compare`: the greater-than comparisons and the float relations no operator
  gives (`ordered_not_equal`, the `unordered_*` forms) and `is_nan`, over 16-, 32- and
  64-bit integers and floats.
- `shader.logic`: `logical_and`, `logical_or`, `logical_not`, `logical_equal` and
  `logical_not_equal`, the reductions `all_*` and `any_*`, and `select_*`.
- `shader.vector`: `shuffle_*`, which builds a vector from the lanes of two, and
  `extract_*` and `insert_*`, which read and write a lane at a run-time index.

```mach
use cmp: shader.compare;
use lg:  shader.logic;

fun lit(n_dot_l: f32x4, shadow: f32x4, colour: f32x4, ambient: f32x4) f32x4 {
    val facing: u32x4 = cmp.greater_than_f32_4(n_dot_l, f32x4{0.0, 0.0, 0.0, 0.0});
    val clear:  u32x4 = cmp.greater_equal_f32_4(shadow, f32x4{0.5, 0.5, 0.5, 0.5});
    ret lg.select_f32_4(lg.logical_and_4(facing, clear), colour, ambient);
}
```

A comparison here gives 1 or 0 in each lane, where a comparison operator on vectors
gives a mask, every bit of a lane set. A truth value is a `u32`, or a `u32` vector
with one per lane, true where it is nonzero, as in `shader.subgroup`. `logical_and`
and `logical_or` evaluate both operands, where `&&` and `||` skip the second, and
`select_*` evaluates both values, where an `if` runs one, so nothing here branches.

## Why this is not in the standard library

`std.math.sqrt_f32` promises a documented IEEE result and delivers it on every
target. `GLSL.std.450 Sqrt` promises what the driver does. They are different
functions, and substituting one for the other across a target boundary gives an
answer that is nearly right with nothing to report. Keeping the shader set in its
own module makes the difference visible where it matters, at the call site.

## Precision

Nothing here is IEEE-exact and nothing here is portable between drivers. SPIR-V
leaves most of GLSL.std.450 implementation-defined in precision; Vulkan's precision
table gives ULP bounds for some entries and none at all for `Pow`, `Exp`, `Log` and
the trigonometric family. Two GPUs may return different bits for the same input,
and so may one GPU under two driver versions.

Use it to shade a pixel. Do not use it for anything compared for equality across
machines.

## This is a shader-only module

Nothing here has a body. On a SPIR-V target that is invisible, because no call to
any of these is emitted as a call. On any other target, a program that calls one
fails to link:

```
error[link.input_invalid] the link refers to symbols nothing defines (src/main.mach)
  note nothing defines `shader.math.sqrt`
  note src/main.mach refers to `shader.math.sqrt`
```

That is deliberate. The alternative - native bodies, so the same source also runs
on a CPU - would mean writing a second float-maths library beside the one
`mach-std` already owns, and the only sensible implementation of each would be to
call `std.math`. That would make `sh.sqrt` and `math.sqrt_f32` the same function on
a CPU and different functions on a GPU: the one outcome of the three that never
errors and is always slightly wrong. A link failure at build time is loud and
early.

## Types and naming

`f32`, `f32x2`, `f32x3` and `f32x4`, and the integer rows over 16-, 32- and 64-bit
integers at the same widths, with 357 entries across them. `ldexp` takes its exponent
as an `i32` of the value's lane count, and the derivatives (`dpdx`, `dpdy`, `fwidth`
and their `_fine` and `_coarse` forms) are core instructions over 32-bit floats that
only a fragment stage may call.

A scalar entry takes the plain name. A vector entry always carries its lane count
after an underscore: `sqrt`, `sqrt_2`, `sqrt_3`, `sqrt_4`. The geometric entries
follow the same rule despite having no scalar form, so they are `dot_3` and
`normalize_3`, and `cross_3` carries its width even though a cross product exists
at no other one.

The separator is not decoration. `exp2` and `log2` are the base-2 entries, where
the digit belongs to the instruction rather than to a width, so a bare digit
suffix would make the 2-lane `exp` collide with the scalar `exp2`. Because the two
also differ in operand type, reaching for the wrong one is a type error rather
than a wrong result.

`modf` and `frexp` return one part of a value and store the other through a
pointer, which may address a local, a `#[gpu.shared]` variable or a storage buffer:
`sh.modf(x, ?whole)`. `frexp` stores its exponent as an `i32` with as many lanes as
the value.

The 32-bit float family is the one unsuffixed type in `shader.math`. The integer
rows carry their component type after an underscore and then the lane count:
`min_i32`, `min_u64_3`, `find_msb_i32_4`. The instruction follows the signedness,
so `min_i32` is `SMin` and `min_u32` is `UMin`, and `abs_*` and `sign_*` exist only
signed. `abs`, `sign`, `min`, `max` and `clamp` take 16-, 32- and 64-bit integers, and
a 16- or 64-bit entry needs the target's `Int16` or `Int64` where it names an `env`.
The bit scans `find_lsb` and `find_msb` are 32-bit only, the one width GLSL.std.450
defines them on, and so are the core bit rows `count_ones`, `reverse_bits`,
`insert_bits` and `extract_bits`, which Vulkan holds to 32 bits.

A few core instructions no operator writes sit in `shader.math` beside the entries
they belong with: `rem`, the driver's `OpFRem` rather than the exact remainder a
float `%` computes, `scale_2`, `scale_3` and `scale_4`, `OpVectorTimesScalar`, which
multiplies a vector by a scalar where `*` takes two vectors, and `add_carry_*` and
`sub_borrow_*`, `OpIAddCarry` and `OpISubBorrow` over unsigned integers, returning an
`AddCarry` or `SubBorrow` record of the wrapped result and the carry or borrow.

`shader.atomic`, `shader.subgroup`, `shader.compare`, `shader.vector` and `select_*`
in `shader.logic` grow along the component type as well, so an
entry over a value the caller chooses carries that value's mach type after the
underscore: `atomic_add_u32`, `atomic_add_f32`, `subgroup_min_i32`,
`greater_than_f32_3`. No type is the
unsuffixed default there. The signed and unsigned forms differ where the instruction
does, so `atomic_min_i32` is `OpAtomicSMin` and `atomic_min_u32` is `OpAtomicUMin`. An
entry whose operand types are fixed, like `subgroup_ballot` or `logical_and_3`, takes
the plain name, with its lane count where it has one.
`shader.texture` names the handle instead, as it always has: `image_read_2d_u`,
`fetch_buffer_i`.

The rule is uniform so that the axes this module grows along - the instruction,
the lane count, and eventually the component type - each stay a drop-in. An
unsuffixed `dot` meaning the 4-wide form would have had to break the first time a
shader wanted the 3-wide one, which is the common case in lighting.

## Using it

Add it with `mach dep add`, which declares the dependency at a caret range over
the newest compatible release and realizes it:

```sh
mach dep add . shader --git https://github.com/briar-systems/mach-shader
```

That writes this stanza to `mach.toml`:

```toml
[dep.shader]
git = "https://github.com/briar-systems/mach-shader"
version = "^0.7.0"
```

Requires a Mach compiler with `#[intrinsic]`, the `#[gpu.*]` shader decorators and
handle operands written as literals.

## How it is checked

`src/math.mach` is generated by `tools/genmath.py` from a table of instructions
and a list of widths, because the library is that rule applied to every width and
hand-maintaining it would let the widths drift apart. `src/atomic.mach` and
`src/subgroup.mach` are generated the same way, by `tools/genatomic.py` and
`tools/gensubgroup.py`, from their instructions and the types each takes, and
`src/compare.mach`, `src/logic.mach` and `src/vector.mach` by `tools/gencore.py`. Each
generator passes its output through `mach fmt -` and writes the mach 7 forms, so it needs
mach 7.0.0 or newer (see `tools/README.md`), either the `mach` on `PATH` or the one the `MACH`
environment variable names, and a formatted tree never drifts from it.

The library itself compiles to nothing - every declaration is bodyless - so
building it proves nothing. `conform/` is the actual check: `conform_frag`, a
fragment shader generated by `tools/genmath.py --conform`, calls every entry in
`shader.math`, writing each result to its output so no call can be eliminated. A
decorator naming an instruction that does not exist, or a signature SPIR-V will not
accept at that width, fails there. Beside it,
`conform_tex_frag` binds and samples every texture handle, `conform_sample_frag`
makes every comparison, gather, fetch and query and reads every subpass input, and `conform_comp` is a compute
stage with a workgroup size, every compute built-in and a read-only and a read-write
buffer. `conform_atomic_comp` calls every atomic on a storage buffer and on workgroup
memory, `conform_image_comp` reads, writes and queries every storage image and texel
buffer and reaches each atomic-format texel, `conform_sync_comp` calls every barrier
and `conform_subgroup_comp` every subgroup operation. `conform_compare_comp`,
`conform_logic_comp` and `conform_vector_comp`, generated by `tools/gencore.py
--conform`, each call every entry of their module from a compute stage.

`conform/verify.sh` (needs spirv-tools) builds them at O0 and O2, runs `spirv-val`
over each both as SPIR-V and against `vulkan1.3`, and checks the disassembly for
what validation alone would not catch: `tools/opcheck.py` requires every `#[intrinsic]`
declared in every module to be called by a conformance shader and to reach its
module, and `tools/instcheck.py` requires each entry of `shader.math`, `shader.compare`,
`shader.logic` and `shader.vector` to leave its own instruction, at its own types, in
the conformance shader generated from it.

## Releasing

Versions follow semver, judged from what has landed on `dev` since the last tag:
a breaking change is major, a feature is minor, and a fix is patch.

1. On a `chore/<issue>` branch off `dev`, set `version` in `mach.toml`, rename
   `## [Unreleased]` in CHANGELOG.md to `## [X.Y.Z] - YYYY-MM-DD`, open a fresh
   `## [Unreleased]` above it, and update the compare links. Merge it into `dev`.
2. Merge `dev` into `main` with a merge commit.
3. Tag that merge `vX.Y.Z` (annotated) and push the tag. The Release workflow
   (`.github/workflows/cd.yml`) checks the tag against the manifest, builds on
   every host the library ships to, and publishes the GitHub release with that version's changelog section as notes.
   Watch that run to success.
4. Merge `main` back into `dev`.

`gh workflow run cd.yml --ref dev` rehearses the same path without a tag,
and deletes its draft when it finishes.

## What is not here, and why

The GLSL.std.450 set is much larger than this. The omissions are reasoned rather
than pending:

- **`ModfStruct`, `FrexpStruct`** return a two-member struct that is not spellable
  as a Mach return type. `modf` and `frexp` give the same two results through a
  pointer.
- **`Determinant`, `MatrixInverse`** need a matrix type, which Mach does not have
  by design - matrices belong in a library over vectors.
- **the `Pack*` / `Unpack*` family** reinterprets one width's bits as another's,
  which logical addressing forbids.
- **`InterpolateAt*`** need the `InterpolationFunction` capability, which the
  emitted module does not declare.
- **`IMix`** was removed from GLSL.std.450, so there is no row to declare.
- **`NMin`, `NMax`, `NClamp`** differ from the `F` forms only in NaN handling. Both
  spellings existing with no visible difference is a trap; the `F` forms are what
  GLSL's own `min` / `max` / `clamp` compile to.

Each of those is a row in the compiler's table and a declaration here on the day
its blocker clears.

## Licence

MIT. See [LICENSE](LICENSE).
