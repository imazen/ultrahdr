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

#### Added
- `metadata::apple` — Apple iOS MakerNote HDR headroom parser. Extracts `0x21 HDRHeadroom`, `0x30 HDRGain`, `0x0a HDRImageType` (per exiftool `Apple.pm`) from EXIF TIFF bytes, computes HDR headroom via the Apple stops formula, and maps to `GainMapMetadata` (`from_apple_headroom`). Validated against 49 real iPhone 8/13/16/17 HEIC captures (parsed values match exiftool, tol 1e-3). `no_std` + `alloc`, zero new deps. Public API: `parse_exif_for_apple_hdr`, `parse_apple_makernote`, `from_apple_headroom`, `AppleHdrInfo`.
- `metadata::bplist` — minimal `bplist00` (Apple binary property list) reader for bplist-encoded MakerNote values (`RunTime`, AE state, …). Depth-bounded against cyclic refs. Public API: `parse_bplist`, `PlistValue`.

#### Changed
- Exclude `tests/` and `benches/` from published package to slim the tarball; local `cargo test`/`cargo bench` are unaffected (target declarations kept intact)

### [0.5.0] - 2026-04-26

#### Breaking changes
- Removed `RawImage`, `RawImageRef`, `RawImageRefMut`. All kernels now take
  `zenpixels::PixelBuffer` (owning) and `zenpixels::PixelSlice` /
  `PixelSliceMut` (borrowed) directly. Replace:
  - `RawImage::new(w, h, fmt)` → `ultrahdr_core::new_pixel_buffer(w, h, fmt, primaries, transfer)`
  - `RawImage::from_data(w, h, fmt, primaries, transfer, data)` →
    `ultrahdr_core::pixel_buffer_from_vec(data, w, h, fmt, primaries, transfer)`
    (note: `data` is now the first argument)
  - Field access `img.data` / `img.stride` / `img.width` / `img.height` /
    `img.format` / `img.gamut` / `img.transfer` →
    `img.as_slice().as_strided_bytes()` / `img.stride()` / `img.width()` /
    `img.height()` / `img.descriptor().pixel_format()` /
    `img.descriptor().primaries` / `img.descriptor().transfer()`.
  - `img.clone()` on a `PixelBuffer` → `ultrahdr_core::clone_pixel_buffer(&img)`
    (zenpixels intentionally doesn't derive `Clone` on `PixelBuffer` to
    discourage silent large-pixel copies).
  - `RawImageRef<'_>` / `RawImageRefMut<'_>` → `PixelSlice<'_>` /
    `PixelSliceMut<'_>`.
- **Deletions queued** (marked `#[doc(hidden)]` in the current release,
  dropped in 0.5.0). None of these are reached from ultrahdr-rs's
  documented decode/encode paths. If you use any of these, switch now.
  - `color::transfer::{SrgbEotfLut, PqEotfLut, HlgOetfInvLut}` — dead LUT
    types. Callers needing per-byte linearization should call
    `linear_srgb::tf::*` directly (no setup cost).
  - `color::transfer::{apply_oetf, apply_eotf}` — generic dispatchers
    with silent sRGB fallback on unknown transfers (footgun).
  - `color::transfer::{pq_to_nits, nits_to_pq}` — trivial `* 10000` /
    `/ 10000`; inline at call sites.
  - `color::transfer::hlg_oetf` (forward direction). The only useful
    direction is `hlg_oetf_inv` (decode); use `linear_srgb::tf::linear_to_hlg`
    directly if you need the forward path.
  - `color::streaming_tonemap::*` — pass-through re-export of
    `zentone::experimental::{StreamingTonemapConfig, StreamingTonemapper}`.
    Import directly from `zentone` instead.
  - `color::tonemap` zentone re-exports: `Bt2408Tonemapper`, `EetfSpace`,
    `CompiledFilmicSpline`, `FilmicSplineConfig`, `AgxLook`, `ToneMap`,
    `ToneMapCurve`, and the named curve functions (`aces_ap1`,
    `agx_tonemap`, `bt2390_tonemap`, `bt2390_tonemap_ext`,
    `filmic_narkowicz`, `hable_filmic`, `reinhard_extended`,
    `reinhard_jodie`, `reinhard_simple`). Pure pass-throughs; import
    from `zentone` / `zentone::curves` directly.
  - `gainmap::streaming::*` — `RowEncoder`, `StreamEncoder`,
    `RowDecoder`, `StreamDecoder` (+ their configs and stats types).
    The per-row kernels these wrap (`sample_gainmap_row_lut`,
    `apply_gain_row_presampled`) stay; the state-machine/ring-buffer
    glue is Ultra-HDR-specific and doesn't generalize (no imageflow or
    zenjpeg consumer for it).

#### Changed
- Re-introduced the `tonemap` feature flag (default-on) gating the
  zentone re-exports at the crate root (`LumaToneMap`,
  `LumaGainMapSplitter`, etc.). Decoder-only consumers can build with
  `--no-default-features --features std` to drop the transitive zentone
  dependency. Replaces the prior `zentone` feature that was removed in
  the splitter consolidation; this version gates only the re-exports
  (no in-core splitter to fall back on when off).

#### Added
- `HdrOutputFormat::LinearF16` — linear f16 RGBA HDR output.
  Mirrors libultrahdr's `UHDR_IMG_FMT_64bppRGBAHalfFloat`. 8 bytes/pixel
  vs `LinearFloat`'s 16. Use for direct compositor / GPU-texture handoff.
- `RgbaF16` and `RgbF16` accepted as encode HDR input (`compute_gainmap`)
  and as decode SDR input (`apply_gainmap`). PQ / HLG / sRGB transfers
  on float inputs are now properly EOTF-decoded — previously RgbaF32
  silently assumed Linear regardless of `descriptor().transfer()`.
- Reference parity tests against libultrahdr and libavif goldens
  (`tests/reference_parity.rs`). 5 tests: bit-exact agreement on
  `applyGain` (105 rows from libultrahdr), `applyGainCore` (35 rows
  from libavif), `avifGetGainMapWeight` (6 rows), a 35-point cross-check
  proving libultrahdr/libavif/ours all agree on shared inputs, and a
  documented-divergence test for libultrahdr's `computeGain` near-black
  clamp (we don't replicate it; ours uses configurable `min_boost`/
  `max_boost` from `GainMapConfig`).
- `gainmap::apply::calculate_weight` is now `pub` (was `pub(crate)`).
  Mirrors `avifGetGainMapWeight`; useful for callers that want to
  precompute the apply weight without going through `apply_gainmap`.
- Shepard's Inverse Distance Weighting upsample for gain map apply
  with integer-scale precomputed weight LUT, plus shared weights across
  channels and row-hoisted constants in the float fallback. Bit-exact
  parity with libultrahdr's CPU `sampleMap` (see f425292).
- Re-export `PixelBuffer`, `PixelSlice`, `PixelSliceMut` at the crate root.
- `new_pixel_buffer(w, h, fmt, primaries, transfer)` — allocate a
  zero-filled `PixelBuffer` with ultrahdr-core's stricter
  dimension/format caps applied.
- `pixel_buffer_from_vec(data, w, h, fmt, primaries, transfer)` — wrap
  an existing `Vec<u8>` as a `PixelBuffer` with the same validators.
- `clone_pixel_buffer(&buf)` — deep-copy a `PixelBuffer` for callers that
  need an owned duplicate.
- `validate_ultrahdr_dimensions`, `validate_ultrahdr_image`,
  `validate_ultrahdr_slice`, `require_supported_format`,
  `descriptor_for` — the validator/descriptor helpers used internally
  at public entry points.
- New `apply_gainmap_slice(sdr: PixelSlice, ...)` — the borrowed
  counterpart to `apply_gainmap(&PixelBuffer, ...)`.
- New `compute_gainmap_slice(hdr: PixelSlice, sdr: PixelSlice, ...)` —
  the borrowed counterpart to `compute_gainmap(&PixelBuffer, ...)`.

#### Removed
- `RawImage`, `RawImageRef`, `RawImageRefMut` — eliminated in favor of
  zenpixels types. (~1,100 LOC removed from `types.rs`.)
- `gainmap::splitter::*` (`LumaGainMapSplitter`, `SplitConfig`,
  `SplitStats`, `LumaToneMap`, `LumaFn`, `HableFilmic`) — moved to
  `zentone`; ultrahdr-core re-exports them at the crate root for
  back-compat. Use `zentone::LumaGainMapSplitter` etc. directly.
- `gainmap::compute::compute_gainmap_tonemap` — niche HDR-only compute
  with explicit tone-curve injection. Callers should drive
  `zentone::LumaGainMapSplitter::new(curve, config).split_row(hdr,
  sdr_out, gain_out, channels, &mut stats)` per row, then call
  `compute_gainmap(hdr, sdr, …)` (or pack the gain map themselves) on
  the resulting SDR.

#### Changed
- The `zentone` dependency is gated behind the new `tonemap` feature
  (default-on). The luma gain map splitter API + curve catalog comes
  from zentone; ultrahdr-core re-exports `LumaGainMapSplitter`,
  `SplitConfig`, `SplitStats`, `LumaToneMap`, `LumaFn`, `HableFilmic`,
  `Bt2408Yrgb`, and `ExtendedReinhardLuma` at the crate root when the
  feature is on. Decoder-only consumers can build with
  `--no-default-features --features std` to drop the transitive
  zentone dep. (Replaces the old `zentone` Cargo feature; same shape,
  renamed and rewired.)
- `color::gamut` owns `apply_matrix` / `apply_matrix_row` /
  `soft_clip_gamut` directly rather than re-exporting from zentone.

#### Fixed
- `gainmap::compute` now imports `alloc::vec` so the
  `--no-default-features` (no_std + alloc) build of `cargo build -p
  ultrahdr-core` compiles. Was a regression from earlier refactor work.
- `gainmap::streaming::RowEncoder` no longer keeps an internal
  `Vec<Vec<u8>>` of per-row gainmap bytes alongside the rows it returns
  for streaming output. Replaced with a single preallocated
  `gainmap_data: Vec<u8>` written in place by `compute_gainmap_row`;
  saves one full per-row `Vec` allocation per gainmap row (~50% fewer
  allocs in the streaming encode path; ~3 MB churn per 4K multi-channel
  encode).

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

#### Fixed
- **`decode_gainmap` decodes RGB (multi-channel) gain maps** (#27): the
  gain-map JPEG is now decoded as RGB and collapses to single-channel only
  when provably achromatic (`R == G == B` at every pixel). Previously it
  requested grayscale output unconditionally — failing outright for some
  color encodings ("unsupported color conversion"; e.g. the libavif seine
  sample, whose hdrgm metadata carries distinct per-channel triples) and
  silently luma-averaging any RGB map that did decode. Per-channel maps
  are mainstream (Adobe exports, iOS 18). Regressions in
  `tests/rgb_gainmap.rs` cover both the 3-channel and the
  achromatic-collapse paths via fully synthetic round-trips.
- **`Decoder::new` no longer aborts when the MPF index fails to parse**
  (#26): MPF is one of several gain-map discovery routes, so a malformed or
  unsupported index (e.g. zenjpeg#148 — valid big-endian `MM` MPF read as
  "zero images") now degrades to the JPEG-boundary fallback instead of
  erroring out of detection that the XMP scan already established. Files
  like the committed 7.6 KB MPF-first fixture previously lost their HDR
  rendition silently in every consumer; regression-pinned in
  `tests/decoder_robustness.rs::mpf_first_sample_detected_as_ultrahdr`
  (asserts detection, exact hdrgm values, and gain-map decode).

#### Changed
- Exclude `tests/` from published package; add `version = "0.1.3"` to the `libultrahdr_rs` git-only optional dep (required by `cargo package`)
- `Encoder::set_hdr_image` / `set_sdr_image` now take `PixelBuffer` (from
  zenpixels) instead of the former `RawImage`. `Decoder::decode_sdr` /
  `decode_hdr` / `decode_hdr_with_format` return `PixelBuffer`. See the
  ultrahdr-core section for the call-site migration table. ultrahdr-rs
  re-exports `PixelBuffer`, `PixelSlice`, `PixelSliceMut`,
  `new_pixel_buffer`, `pixel_buffer_from_vec`, `clone_pixel_buffer`,
  `descriptor_for` at the crate root.

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
- Versioned public-API surface snapshots: `docs/public-api/<crate>.txt` for `ultrahdr-core` and `ultrahdr-rs`, regenerated by `ultrahdr-core/tests/public_api_doc.rs` on every `cargo test` (`ZEN_API_DOC=check` verified by the new ci.yml `api-doc` job, `=off` elsewhere; `just api-doc` / `api-doc-check` recipes; `just fmt` regenerates). The test-only `ffi-tests` feature is excluded from the all-features section (dev/test deps only, no public surface).
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
