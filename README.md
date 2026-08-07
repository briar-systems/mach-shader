# mach-shader

Shader-side maths for [Mach](https://github.com/briar-systems/mach): the functions
that lower to a single SPIR-V instruction.

```mach
use sh: shader.math;

#[input(0)]  var in_normal: f32x4;
#[input(1)]  var in_light:  f32x4;
#[output(0)] var out_colour: f32x4;

#[stage("fragment")]
fun frag_main() {
    val n: f32x4 = sh.normalize(in_normal);
    val d: f32   = sh.max(sh.dot(n, in_light), 0.0);
    out_colour   = sh.mix4(ambient, albedo, f32x4{d, d, d, d});
}
```

`sh.normalize` does not compile to a call. Each declaration in `shader.math`
carries a `#[spirv_op(set, name)]` decorator naming the SPIR-V instruction it *is*,
and the compiler substitutes that instruction at the call site: an `OpExtInst` into
`GLSL.std.450` for most of them, core `OpDot` for `dot`. The extended instruction
set is imported once per module and only when something in that module uses it.

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

## Types

`f32` and `f32x4`. The per-lane forms take a `4` suffix where a scalar of the same
name exists (`max` and `max4`); the geometric entries, which have no scalar form,
keep the plain name (`dot`, `length`, `normalize`, `reflect`).

`f32x3` is the gap that matters and is tracked in
[briar-systems/mach#2687](https://github.com/briar-systems/mach/issues/2687). Until
it lands, carry a 3-component normal in an `f32x4` with its fourth lane at 0 -
`normalize` and `length` count every lane.

## Using it

```toml
[dep.mach-shader]
git = "https://github.com/briar-systems/mach-shader"
ref = "branch/main"
```

Requires a Mach with `#[spirv_op]` support (briar-systems/mach#2688).

## What is not here, and why

The GLSL.std.450 set is much larger than this. The omissions are reasoned rather
than pending:

- **`Modf`, `Frexp`** take a pointer to write their second result through. SPIR-V
  here is logical addressing: a pointer has exactly one pointee type and none may
  be cast, and there is no way to hand the back end a Function-storage out-pointer
  at a call site. The `*Struct` forms return a two-member struct that is not
  spellable as a Mach return type.
- **`Cross`** is defined only for 3-component vectors, so it lands with `f32x3`.
- **`Determinant`, `MatrixInverse`** need a matrix type, which Mach does not have
  by design - matrices belong in a library over vectors.
- **the `Pack*` / `Unpack*` family** reinterprets one width's bits as another's,
  which logical addressing forbids.
- **`InterpolateAt*`** need the `InterpolationFunction` capability, which the
  emitted module does not declare.
- **the integer `UMin` / `SMin` / `UMax` / `SMax` / `UClamp` / `SClamp` and the
  `Find*Msb` bit-scan family** are ordinary integer arithmetic that a CPU computes
  identically. They are not the shader gap, and routing them through a shader-only
  module would make code less portable rather than more.
- **`NMin`, `NMax`, `NClamp`** differ from the `F` forms only in NaN handling. Both
  spellings existing with no visible difference is a trap; the `F` forms are what
  GLSL's own `min` / `max` / `clamp` compile to.

Each of those is a row in the compiler's table and a declaration here on the day
its blocker clears.

## Licence

MIT. See [LICENSE](LICENSE).
