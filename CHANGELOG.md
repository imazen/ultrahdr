# Changelog

All notable changes to this repository are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); crates follow [Semantic
Versioning](https://semver.org/).

The workspace ships two publishable crates — `ultrahdr-core` and `ultrahdr-rs` —
with independent version numbers, plus shared workspace tooling. Each has its
own section below.

## ultrahdr-core

### [Unreleased]

#### QUEUED BREAKING CHANGES
<!-- Breaking changes queued for the next major (or minor for 0.x) release.
     Batch them here instead of shipping piecemeal. -->
- `zentone` is now an optional default-on feature rather than a hard
  dependency. With `--no-default-features`, `color::tonemap` and the
  `StreamingTonemapper` re-exports disappear; decoders that only need to
  apply/compute gain maps can depend on ultrahdr-core without pulling in
  zentone. (Breaking only for callers building with custom feature sets
  that already compiled against zentone-free configurations.)

#### Added
- In-core luma gain map splitter: `LumaToneMap` trait, `LumaFn` closure
  adapter, `SplitConfig`, `SplitStats`, `LumaGainMapSplitter`, and a
  built-in `HableFilmic` (Uncharted 2) tone curve. Makes it possible to
  reduce HDR to (SDR, luma gain) and roundtrip back without depending on
  zentone. Re-exported at the crate root as `ultrahdr_core::HableFilmic`,
  `LumaToneMap`, `LumaGainMapSplitter`, `SplitConfig`, `SplitStats`.
- `impl LumaToneMap` for zentone's BT.2408 / BT.2446 A/B/C /
  `CompiledFilmicSpline` when the `zentone` feature is enabled, so
  callers can pass those curves directly to `LumaGainMapSplitter`.

#### Changed
- `zentone` moved from hard dep to optional default-on feature. With
  zentone off, `color::tonemap` is unavailable; `color::gamut` now owns
  `apply_matrix` / `apply_matrix_row` / `soft_clip_gamut` directly rather
  than re-exporting them.
- `compute_gainmap_tonemap` now dispatches through the in-core
  `LumaGainMapSplitter` rather than zentone's.

### [0.4.1] - 2026-04-10

#### Added
- ISO 21496-1 `common_denominator` parsing and `Iso21496Format` conversion helpers (2474e8e)
- `backward_direction` field on `GainMapMetadata`, parsed from ISO 21496-1 flags (dab5e27)
- Comprehensive ISO 21496-1 metadata format documentation (80f7970)
- `create_jpeg_iso_markers` high-level API for emitting both APP2 markers in one call (5f36b0d)
- SIMD tier consistency tests for gain map application (fc4f30e)
- aarch64 NEON SIMD implementation for gain map application (e85f7ce)
- Overflow guard on gain map math (269061c)

#### Changed
- Replaced platform-specific SIMD with portable magetypes generics (264ffff)
- Bumped `archmage` / `magetypes` 0.9.4 → 0.9.16 and `linear-srgb` 0.6.5 → 0.6.7 (e9c2931)
- Gated `proptest` behind `cfg(not(wasm32))` so wasm tests build cleanly (ba9310f)

#### Fixed
- Emit version-only ISO APP2 in the primary JPEG (4a14af1)
- Strip version byte from ISO 21496-1 JPEG APP2 and default wire format to `Both` (626063c)
- Use continued-fraction algorithm for ISO 21496-1 fraction encoding to match Chromium expectations (23f81c9)
- Resolved broken doc links in `ultrahdr-core` (22d851a)
- Collapsed nested `if let` in decoder XMP parsing to satisfy new clippy lints (e31032b)
- Prevent slice panic in MPF parsing when segment length < 2 (14b9423)
- Prevent panics from out-of-bounds JPEG segment offsets (aee9692)

### [0.3.0] - 2026-03-29

#### Changed
- 8 breaking API changes flagged by `cargo semver-checks`; see commit for details (8c07820)
- Switched gain map metadata types to the canonical zencodec versions, deleting ~1700 lines of duplication (0524d44)
- Introduced borrow-only `RawImageRef` / `RawImageRefMut` and made public enums `#[non_exhaustive]` (7acea4d)

### [0.2.0] - 2026-02-21

#### Changed
- Adapted to the new zenjpeg API, refreshed transitive deps (45e4e40)
- MSRV bumped to 1.92 (b22fdde)

### [0.1.1] - 2026-01-23

#### Added
- `set_existing_gainmap_jpeg()` for raw JPEG passthrough on the gain map side (eca5894)
- Dual streaming APIs for parallel decode/encode (13a609a)
- Low-memory streaming APIs for gain map processing (c80dd91)
- `AdaptiveTonemapper` for learned HDR-to-SDR curves (5e3eab8)
- `enough::Stop` cooperative cancellation hooks (e0d20b7)
- `GainMapLut` giving a 32-37% speedup on `apply_gainmap` (cf70399)
- `no_std` support for `ultrahdr-core` (db8594a)

### [0.1.0] - 2026-01-13

#### Added
- Initial pure-Rust Ultra HDR implementation (46ad2be).
- Split into `ultrahdr-core` (math + metadata, no codec dep) and `ultrahdr` crates (9f76f93).

## ultrahdr-rs

### [Unreleased]

#### QUEUED BREAKING CHANGES
<!-- Breaking changes queued for the next major (or minor for 0.x) release. -->

### [0.3.5] - 2026-04-10

#### Changed
- Yanked 0.4.0 — `cargo semver-checks` confirmed no breaking changes vs 0.3.4, so this is a patch release instead (b448736)
- Pinned `moxcms` to a concrete version for crates.io publish (d9f1305)

### [0.3.0] - 2026-03-29

#### Changed
- Added explicit `zenjpeg 0.7` version constraint (44081c0)
- Updated to `zencodec` 0.1.8 API (b2841c5)
- Required `zenpixels` 0.2.1 for gamut matrix, serde, ICC profiles, and bug fixes (126ba04)

### [0.2.0] - 2026-02-21

#### Changed
- Adapted to new zenjpeg API and refreshed deps (45e4e40)
- Updated README code examples for the 0.2.0 API (537e73b)

### [0.1.1] - 2026-01-23

#### Added
- Streaming encode/decode pipelines and tonemapper exposed through the top-level crate (1be054d, 8887946, 149aff2)
- `set_existing_gainmap()` for gain map reuse (4b25c6d)

#### Changed
- Renamed the `ultrahdr` crate to `ultrahdr-rs` to free the `ultrahdr` name on crates.io (c84d85d)

### [0.1.0] - 2026-01-13

#### Added
- Initial pure-Rust Ultra HDR encoder/decoder built on zenjpeg (46ad2be)
- WASM build and test infrastructure (a127e64, d6f9679)
- FFI parity test suite against libultrahdr (19b37e2)

## Workspace

### [Unreleased]

#### QUEUED BREAKING CHANGES
<!-- None queued. -->

#### Added
- Credits / Acknowledgments section in README for Google's Ultra HDR spec and the libultrahdr reference implementation.
- `.gitignore` entry for the `.workongoing` agent coordination marker.

#### Changed
- Cleaned up `unused manifest key` warnings by removing redundant `path =` / `version =` overrides on workspace dependencies in `ultrahdr-core` and `ultrahdr-rs`.

### 2026-04 — Tooling

- Nightly fuzz workflow: 60 s on push, 5 min nightly (aa66e35)
- Added 5 cargo-fuzz targets plus a custom JPEG dictionary (2f77e48)
- Committed fuzz infrastructure — seeds, dictionary, lockfile (9cd277b)
- Gitignored tooling noise (`.superwork/`, `.zenbench/`, etc.) and excluded it from published packages (b93ba96)

### 2026-03 — CI & Deps

- Tested the `zencodec` feature in CI alongside XMP / ISO 21496 parity tests (b71611a)
- Fixed `wasm-bench` build flags for edition 2024 and WASI cap-lints (fe21c23, 7bb494b, 642db55, f07c1b8, cd094fb, d24e3b3)
- Replaced local path dependencies with crates.io versions across the workspace (31e0bb1)

### 2026-01 — Initial Infrastructure

- Added cross-platform SIMD benchmarking via archmage / magetypes (c58f543)
- Added criterion benchmarks for core gain map operations (f1d2f39)
- Added WASI benchmark testing with wasmtime (1253735)
- Added WASM SIMD128 build support and `justfile` targets (413bede)
