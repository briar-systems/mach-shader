# mach-shader

Shader-side maths for [Mach](https://github.com/briar-systems/mach): the functions
that lower to a single SPIR-V instruction.

```mach
use sh: shader.math;

#[input(0)]  var in_normal: f32x3;
#[input(1)]  var in_light:  f32x3;
#[output(0)] var out_colour: f32x4;

#[stage("fragment")]
fun frag_main() {
    val n: f32x3 = sh.normalize_3(in_normal);
    val d: f32   = sh.max(sh.dot_3(n, in_light), 0.0);
    out_colour   = sh.mix_4(ambient, albedo, f32x4{d, d, d, d});
}
```

`sh.normalize_3` does not compile to a call. Each declaration in `shader.math`
carries an `#[op(target, set, name)]` decorator naming the SPIR-V instruction it *is*,
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

#[sampler(0, 0)] var albedo: tex.Sampler2D;
#[input(0)] var in_uv: f32x2;
#[output(0)] var out_colour: f32x4;

#[stage("fragment")]
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
(`size_*`, `levels_*`). A multisampled image is never sampled and is not declared, and
the target refuses the `Rect` and `SubpassData` dimensionalities by naming the
operand.

### Storage images and texel buffers

A storage image is read and written texel by texel, with no sampler, and binds
through `#[storage]` like a buffer, `"readonly"`, `"writeonly"` and `"coherent"`
included:

```mach
use tex: shader.texture;

#[storage(0, 0, "readonly")]  var src: tex.StorageImage2D;
#[storage(0, 1, "writeonly")] var dst: tex.StorageImage2D;
#[builtin("global_invocation")] var gid: u32x3;

#[stage("compute")]
#[workgroup(8, 8, 1)]
fun blit() {
    val at: i32x2 = i32x2{gid[0]::i32, gid[1]::i32};
    tex.image_write_2d(dst, at, tex.image_read_2d(src, at));
}
```

The `StorageImage*` handles are `FORMAT_UNKNOWN`, so one handle binds a view of any
format, under the `storage_read_without_format` and `storage_write_without_format`
extensions that `vulkan1.3` selects. The `*R32f`, `*R32i`, `*R32ui`, `*R64i` and
`*R64ui` images carry the formats Vulkan defines an image atomic on, and
`image_texel_*` gives an atomic in `shader.atomic` the address of one of their
texels. A texel buffer is a `TexelBuffer`, fetched and bound by `#[sampler]`, or a
`StorageTexelBuffer`, read and written and bound by `#[storage]`.

## Compute

A compute stage is a `#[stage("compute")]` function sized by `#[workgroup(x, y,
z)]`, and it emits a GLCompute entry point with that `LocalSize`. It reads and
writes `#[storage(set, binding)]` buffers rather than varyings, since Vulkan has no
`Output` class in a compute stage, and a buffer marked `"readonly"` is emitted
`NonWritable`. Its ids are the `global_invocation`, `local_invocation` and
`workgroup_id` built-ins, each a `u32x3`. The maths in `shader.math` works in a
compute body exactly as it does in a fragment one.

```mach
use sh: shader.math;

rec Field { values: [256]f32; }

#[storage(0, 0, "readonly")] var src: Field;
#[storage(0, 1)]             var dst: Field;
#[builtin("global_invocation")] var gid: u32x3;

#[stage("compute")]
#[workgroup(64, 1, 1)]
fun step() {
    val i: u32 = gid[0];
    dst.values[i] = sh.sqrt(src.values[i]);
}
```

All of that is the compiler's. What a compute body calls is this library's, each
one a bodyless `#[op]` declaration like the maths:

- `shader.sync`: the `SCOPE_*` and `SEMANTICS_*` operand constants,
  `control_barrier` and `memory_barrier`, and the wrappers `workgroup_barrier`,
  `storage_barrier`, `image_barrier` and `subgroup_barrier`.
- `shader.atomic`: every SPIR-V atomic over `u32`, `i32`, `u64` and `i64`, and the
  load, store, exchange, add, min and max over `f16`, `f32` and `f64`.
- `shader.subgroup`: the `OpGroupNonUniform*` families, the `GROUP_*` operations and
  the `QUAD_SWAP_*` directions.
- `shader.texture`: the storage images and texel buffers above.

```mach
use at: shader.atomic;
use sy: shader.sync;

rec Histogram { bins: [64]u32; }

#[storage(0, 0)] var histogram: Histogram;
#[shared]        var local_bins: [64]u32;
#[builtin("local_invocation_index")] var lid: u32;

#[stage("compute")]
#[workgroup(64, 1, 1)]
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
`Elect`, needs a Vulkan device feature that the target names in its `extensions`
(`buffer_float32_atomic_add`, `subgroup_arithmetic`, and so on). The module headers
list them, and a call whose feature the target does not select is refused with its
name.

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
error: undefined symbol: _M6shader4mathN4sqrt
```

That is deliberate. The alternative - native bodies, so the same source also runs
on a CPU - would mean writing a second float-maths library beside the one
`mach-std` already owns, and the only sensible implementation of each would be to
call `std.math`. That would make `sh.sqrt` and `math.sqrt_f32` the same function on
a CPU and different functions on a GPU: the one outcome of the three that never
errors and is always slightly wrong. A link failure at build time is loud and
early.

## Types and naming

`f32`, `f32x2`, `f32x3` and `f32x4`, and the integer rows over `i32` and `u32` at
the same widths, with 190 entries across them.

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
pointer, which may address a local, a `#[shared]` variable or a storage buffer:
`sh.modf(x, ?whole)`. `frexp` stores its exponent as an `i32` with as many lanes as
the value.

The 32-bit float family is the one unsuffixed type in `shader.math`. The integer
rows carry their component type after an underscore and then the lane count:
`min_i32`, `min_u32_3`, `find_msb_i32_4`. The instruction follows the signedness,
so `min_i32` is `SMin` and `min_u32` is `UMin`, and `abs_*` and `sign_*` exist only
for `i32`.

`shader.atomic` and `shader.subgroup` grow along the component type as well, so an
entry over a value the caller chooses carries that value's mach type after the
underscore: `atomic_add_u32`, `atomic_add_f32`, `subgroup_min_i32`. No type is the
unsuffixed default there. The signed and unsigned forms differ where the instruction
does, so `atomic_min_i32` is `OpAtomicSMin` and `atomic_min_u32` is `OpAtomicUMin`. An
entry whose operand types are fixed, like `subgroup_ballot`, takes the plain name.
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
version = "^0.5.0"
```

Requires Mach 6.10. The manifest declares `mach = "^6.10"`, the release that
declares a signed-format image over a signed texel type and carries the `Modf`,
`Frexp` and integer GLSL.std.450 rows.

## How it is checked

`src/math.mach` is generated by `tools/genmath.py` from a table of instructions
and a list of widths, because the library is that rule applied 190 times and
hand-maintaining it would let the widths drift apart. `src/atomic.mach` and
`src/subgroup.mach` are generated the same way, by `tools/genatomic.py` and
`tools/gensubgroup.py`, from their instructions and the types each takes. Each
generator passes its output through `mach fmt -`, so it needs mach 5.1.0 or newer
(see `tools/README.md`), either the `mach` on `PATH` or the one the `MACH`
environment variable names, and a formatted tree never drifts from it.

The library itself compiles to nothing - every declaration is bodyless - so
building it proves nothing. `conform/` is the actual check: a fragment shader that
calls one entry from every family at every width, writing the result to its output
so no call can be eliminated. A decorator naming an instruction that does not
exist, or a signature SPIR-V will not accept at that width, fails there. Beside it,
`conform_tex_frag` binds and samples every texture handle, `conform_sample_frag`
makes every comparison, gather, fetch and query, and `conform_comp` is a compute
stage with a workgroup size, every compute built-in and a read-only and a read-write
buffer. `conform_atomic_comp` calls every atomic on a storage buffer and on workgroup
memory, `conform_image_comp` reads, writes and queries every storage image and texel
buffer and reaches each atomic-format texel, `conform_sync_comp` calls every barrier
and `conform_subgroup_comp` every subgroup operation.

`conform/verify.sh` (needs spirv-tools) builds them at O0 and O2, runs `spirv-val`
over each both as SPIR-V and against `vulkan1.3`, and checks the disassembly for
what validation alone would not catch: `tools/opcheck.py` requires every `#[op]`
declared in `shader.texture`, `shader.sync`, `shader.atomic` and `shader.subgroup`
to be called by a conformance shader and to reach its module.

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
