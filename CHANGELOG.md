# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Copyright is attributed to Briar Systems LLC (#26).

## [0.3.0] - 2026-09-16

### Changed
- Requires Mach 5.0. The manifests use the 5.0 schema (#5).
- `src/math.mach` is laid out by `mach fmt`, and `tools/genmath.py` pipes its output through `mach fmt -` (#7).
- `tools/genmath.py` requires mach 5.1.0 and says so when the compiler is older (#14).
- CI runs the family's shared tiered workflow and ends in a `gate` job (#7, #10).

### Added
- CHANGELOG.md and a written release process (#13).
- `tools/texcheck.py`: CI asserts that every declared image shape is sampled and combined in `conform_tex_frag`, which now calls every `combine_*` (#15).

## [0.2.0] - 2026-08-09

### Added
- `shader.texture`, which declares the image, sampler and combined-sampler handles a shader binds, plus `combine_*` (`OpSampledImage`) and `sample_*` (`OpImageSampleImplicitLod`) for each shape (#2).
- `conform_tex_frag`, a conformance shader that binds and reads each declared handle shape.

### Changed
- **Breaking:** the `#[spirv_op(set, name)]` decorator is now `#[op(target, set, name)]` on every declaration, following mach#2888.

## [0.1.0] - 2026-08-08

### Added
- `shader.math`: 134 bodyless declarations over `f32`, `f32x2`, `f32x3` and `f32x4`. Each one lowers to a single SPIR-V instruction: a `GLSL.std.450` `OpExtInst`, or core `OpDot` for `dot_*`.
- The width-carrying naming rule: scalar entries take the plain name, and vector entries end in their lane count (`sqrt_3`, `dot_3`, `cross_3`).
- `tools/genmath.py`, which generates `src/math.mach`, and `conform/`, the conformance shader CI validates with `spirv-val`.

[Unreleased]: https://github.com/briar-systems/mach-shader/compare/v0.3.0...dev
[0.3.0]: https://github.com/briar-systems/mach-shader/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/briar-systems/mach-shader/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/briar-systems/mach-shader/releases/tag/v0.1.0
